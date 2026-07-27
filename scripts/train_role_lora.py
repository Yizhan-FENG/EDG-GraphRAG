#!/usr/bin/env python3
"""Train one isolated role LoRA with resumable, weighted QLoRA segments.

Each invocation trains A1 or A3 only.  A segment checkpoint includes the PEFT
adapter, optimiser, scheduler, RNG state and weighted-sampling generator, so a
later invocation can continue the same training run without rebuilding or
mixing the other role's LoRA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml


@dataclass
class EncodedRow:
    input_ids: list[int]
    labels: list[int]
    source_name: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"role", "adapter_name", "base_model", "hf_home", "output_dir", "catalog", "train_sources"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing required config keys: {sorted(missing)}")
    # The original smoke configs use max_steps.  Keep them usable while formal
    # configs use explicit total_steps and segment_steps.
    config.setdefault("total_steps", config.get("max_steps", 160))
    config.setdefault("segment_steps", config["total_steps"])
    config.setdefault("validation_max_rows", 32)
    config.setdefault("evaluate_at_segment_end", True)
    return config


def read_catalog(path: Path) -> dict[str, Any]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog.get("dataset_version") != "0.3":
        raise ValueError(f"Unsupported catalog version: {catalog.get('dataset_version')}")
    return catalog


def source_rows(config: dict[str, Any], catalog: dict[str, Any], split: str) -> list[tuple[str, dict[str, Any]]]:
    role = config["role"]
    section = catalog[role]
    rows: list[tuple[str, dict[str, Any]]] = []
    forbidden = {Path(value) for value in catalog["forbidden_inputs"]}
    extra_names = {extra["name"] for extra in config.get("extra_train_jsonl", [])}
    for source_name in config["train_sources"]:
        if source_name in extra_names:
            continue
        if source_name not in section["counts"]:
            raise ValueError(f"{role}: source not found in catalog: {source_name}")
        source_path = Path(section[source_name][split])
        if source_path in forbidden:
            raise ValueError(f"Forbidden dataset selected: {source_path}")
        records = read_jsonl(source_path)
        expected = section["counts"][source_name][split]
        if len(records) != expected:
            raise ValueError(f"{source_name} {split}: expected {expected} rows, found {len(records)}")
        rows.extend((source_name, record) for record in records)
    # Explicitly versioned, training-only augmentation.  It never participates
    # in validation/test and is named separately so weighted sampling remains
    # visible in the run manifest.
    for extra in config.get("extra_train_jsonl", []) if split == "train" else []:
        source_name = extra["name"]
        source_path = Path(extra["path"])
        if source_name not in config["train_sources"]:
            raise ValueError(f"Extra training source {source_name} needs a sampling weight")
        rows.extend((source_name, record) for record in read_jsonl(source_path))
    if not rows:
        raise ValueError(f"No {split} records selected")
    return rows


def token_ids(processor: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> list[int]:
    ids = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    # Newer text tokenizers return BatchEncoding while the multimodal processor
    # may return bare IDs.  Both are valid chat-template results.
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def tokenizer_for(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def encode_record(processor: Any, record: dict[str, Any], max_length: int, source_name: str) -> EncodedRow | None:
    messages = record.get("messages", [])
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        sample_id = record.get("metadata", {}).get("sample_id", "<unknown>")
        raise ValueError(f"Invalid chat roles in {sample_id}")
    prompt_ids = token_ids(processor, messages[:2], add_generation_prompt=True)
    full_ids = token_ids(processor, messages, add_generation_prompt=False)
    if not full_ids or len(full_ids) <= len(prompt_ids):
        return None
    if full_ids[: len(prompt_ids)] != prompt_ids:
        common_prefix = 0
        for prompt_token, full_token in zip(prompt_ids, full_ids):
            if prompt_token != full_token:
                break
            common_prefix += 1
        prompt_ids = prompt_ids[:common_prefix]
    if len(full_ids) > max_length:
        answer_ids = full_ids[len(prompt_ids) :]
        prompt_budget = max_length - len(answer_ids)
        if prompt_budget <= 0:
            return None
        full_ids = prompt_ids[-prompt_budget:] + answer_ids
        prompt_length = prompt_budget
    else:
        prompt_length = len(prompt_ids)
    labels = [-100] * prompt_length + full_ids[prompt_length:]
    if all(label == -100 for label in labels):
        return None
    return EncodedRow(input_ids=full_ids, labels=labels, source_name=source_name)


def select_target_modules(model: Any) -> str:
    """Select language linear projections and permanently exclude vision/head layers."""

    import torch

    suffixes = set()
    for name, module in model.named_modules():
        lowered = name.lower()
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(marker in lowered for marker in ("vision", "visual", "image", "lm_head", "embed")):
            continue
        suffixes.add(name.rsplit(".", 1)[-1])
    if not suffixes:
        raise RuntimeError("No language-side linear modules found for LoRA")
    escaped = "|".join(sorted((suffix.replace(".", "\\.") for suffix in suffixes), key=len, reverse=True))
    return rf"^(?!.*(?:vision|visual|image|lm_head|embed)).*(?:{escaped})$"


def collate(rows: list[EncodedRow], pad_token_id: int) -> dict[str, Any]:
    import torch

    max_length = max(len(row.input_ids) for row in rows)
    input_ids, attention_mask, labels = [], [], []
    for row in rows:
        padding = max_length - len(row.input_ids)
        input_ids.append(row.input_ids + [pad_token_id] * padding)
        attention_mask.append([1] * len(row.input_ids) + [0] * padding)
        labels.append(row.labels + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def balanced_validation_rows(rows: list[EncodedRow], max_rows: int) -> list[EncodedRow]:
    """Keep validation bounded while retaining each selected source tier."""

    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    grouped: dict[str, deque[EncodedRow]] = defaultdict(deque)
    for row in rows:
        grouped[row.source_name].append(row)
    selected: list[EncodedRow] = []
    source_names = sorted(grouped)
    while len(selected) < max_rows and any(grouped.values()):
        for source_name in source_names:
            if grouped[source_name] and len(selected) < max_rows:
                selected.append(grouped[source_name].popleft())
    return selected


def evaluate(
    model: Any,
    rows: list[EncodedRow],
    pad_token_id: int,
    device: Any,
    on_progress: Callable[[int, int], None] | None = None,
) -> float:
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            if on_progress:
                on_progress(index, len(rows))
            batch = {name: tensor.to(device) for name, tensor in collate([row], pad_token_id).items()}
            losses.append(float(model(**batch).loss.detach().float().cpu()))
    model.train()
    return sum(losses) / len(losses) if losses else math.nan


def checkpoint_dir(output_dir: Path, global_step: int) -> Path:
    return output_dir / "checkpoints" / f"step-{global_step:04d}"


def save_checkpoint(
    checkpoint: Path,
    model: Any,
    processor: Any,
    optimizer: Any,
    scheduler: Any,
    sampler_generator: Any,
    metadata: dict[str, Any],
) -> None:
    import torch

    checkpoint.mkdir(parents=True, exist_ok=True)
    adapter_dir = checkpoint / "adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    state = {
        "metadata": metadata,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "sampler_state": sampler_generator.get_state(),
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    torch.save(state, checkpoint / "trainer_state.pt")
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_checkpoint_state(path: Path, role: str, base_model: str) -> dict[str, Any]:
    import torch

    state_path = path / "trainer_state.pt"
    adapter_path = path / "adapter" / "adapter_config.json"
    if not state_path.is_file() or not adapter_path.is_file():
        raise ValueError(f"Resume checkpoint is incomplete: {path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    metadata = state["metadata"]
    if metadata["role"] != role or metadata["base_model"] != base_model:
        raise ValueError("Resume checkpoint role/base model does not match the requested formal run")
    return state


def write_run_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_training_heartbeat(output_dir: Path, stage: str, **details: Any) -> None:
    """Atomically expose the current long-running stage for recovery diagnostics."""

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        **details,
    }
    target = output_dir / "training_heartbeat.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate data/config only; do not load model weights.")
    parser.add_argument("--max-steps", type=int, help="Smoke-run override: set total and segment steps without editing YAML.")
    parser.add_argument("--segment-steps", type=int, help="Override only the number of optimiser steps in this invocation.")
    parser.add_argument("--max-seq-length", type=int, help="Memory-constrained smoke-run override; does not edit YAML.")
    parser.add_argument("--output-dir", type=Path, help="Output override for isolated smoke runs.")
    parser.add_argument("--resume-from", type=Path, help="Checkpoint directory from a previous formal segment.")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip validation only for a short smoke run.")
    args = parser.parse_args()

    config = read_config(args.config)
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        config["total_steps"] = args.max_steps
        config["segment_steps"] = args.max_steps
    if args.segment_steps is not None:
        if args.segment_steps <= 0:
            raise ValueError("--segment-steps must be positive")
        config["segment_steps"] = args.segment_steps
    if args.max_seq_length is not None:
        if args.max_seq_length < 64:
            raise ValueError("--max-seq-length must be at least 64")
        config["max_seq_length"] = args.max_seq_length
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if int(config["segment_steps"]) > int(config["total_steps"]):
        raise ValueError("segment_steps cannot exceed total_steps")

    catalog = read_catalog(Path(config["catalog"]))
    train_records = source_rows(config, catalog, "train")
    validation_records = source_rows(config, catalog, "validation")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "role": config["role"],
                    "base_model": config["base_model"],
                    "train_rows": len(train_records),
                    "validation_rows": len(validation_records),
                    "total_steps": config["total_steps"],
                    "segment_steps": config["segment_steps"],
                    "train_sources": config["train_sources"],
                    "status": "dry_run_passed",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    os.environ["HF_HOME"] = str(config["hf_home"])
    os.environ.setdefault("HF_HUB_CACHE", str(Path(config["hf_home"]) / "hub"))
    Path(config["hf_home"]).mkdir(parents=True, exist_ok=True)
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
        get_linear_schedule_with_warmup,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this QLoRA configuration")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    model_mode = config.get("model_mode", "multimodal")
    if model_mode == "causal_lm":
        processor = AutoTokenizer.from_pretrained(config["base_model"])
    elif model_mode == "multimodal":
        processor = AutoProcessor.from_pretrained(config["base_model"])
    else:
        raise ValueError(f"Unsupported model_mode: {model_mode}")
    tokenizer = tokenizer_for(processor)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError("Tokenizer has neither pad token nor EOS token")
    train_encoded = [
        encoded
        for source_name, record in train_records
        if (encoded := encode_record(processor, record, config["max_seq_length"], source_name)) is not None
    ]
    validation_encoded = [
        encoded
        for source_name, record in validation_records
        if (encoded := encode_record(processor, record, config["max_seq_length"], source_name)) is not None
    ]
    validation_subset = balanced_validation_rows(validation_encoded, int(config["validation_max_rows"]))
    if not train_encoded or not validation_subset:
        raise RuntimeError("No usable encoded train/validation records")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_from = args.resume_from or (Path(config["resume_from"]) if config.get("resume_from") else None)
    resume_state = load_checkpoint_state(resume_from, config["role"], config["base_model"]) if resume_from else None

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_loader = AutoModelForCausalLM if model_mode == "causal_lm" else AutoModelForMultimodalLM
    model = model_loader.from_pretrained(
        config["base_model"],
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    target_modules = select_target_modules(model)
    if resume_state:
        model = PeftModel.from_pretrained(model, resume_from / "adapter", is_trainable=True)
        start_step = int(resume_state["metadata"]["global_step"])
        saved_best_loss = resume_state["metadata"]["best_validation_loss"]
        best_loss = math.inf if saved_best_loss is None else float(saved_best_loss)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=config["lora_r"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"],
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        start_step = 0
        best_loss = math.inf
    model.print_trainable_parameters()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(config["learning_rate"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(int(config["total_steps"]) * float(config["warmup_ratio"]))),
        num_training_steps=int(config["total_steps"]),
    )
    sampler_generator = torch.Generator(device="cpu")
    sampler_generator.manual_seed(int(config["seed"]))
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        sampler_generator.set_state(resume_state["sampler_state"])
        random.setstate(resume_state["python_random_state"])
        torch.set_rng_state(resume_state["torch_rng_state"])
        torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])

    total_steps = int(config["total_steps"])
    if start_step >= total_steps:
        raise ValueError(f"Checkpoint is already complete at step {start_step}/{total_steps}")
    end_step = min(total_steps, start_step + int(config["segment_steps"]))
    source_weights = torch.tensor([float(config["train_sources"][row.source_name]) for row in train_encoded], dtype=torch.double)
    device = torch.device("cuda:0")
    log_path = output_dir / "training_log.jsonl"
    optimizer.zero_grad(set_to_none=True)

    def heartbeat(stage: str, **details: Any) -> None:
        write_training_heartbeat(
            output_dir,
            stage,
            role=config["role"],
            global_step=details.pop("global_step", start_step),
            total_steps=total_steps,
            **details,
        )

    heartbeat("training_started", resumed_from=str(resume_from) if resume_from else None)

    for global_step in range(start_step + 1, end_step + 1):
        heartbeat("train_step_started", global_step=global_step)
        accumulated_loss = 0.0
        sampled_sources: dict[str, int] = defaultdict(int)
        for micro_step in range(1, int(config["gradient_accumulation_steps"]) + 1):
            heartbeat("micro_batch_started", global_step=global_step, micro_step=micro_step)
            index = int(torch.multinomial(source_weights, 1, replacement=True, generator=sampler_generator).item())
            row = train_encoded[index]
            sampled_sources[row.source_name] += 1
            batch = {name: tensor.to(device) for name, tensor in collate([row], pad_token_id).items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / int(config["gradient_accumulation_steps"])
            loss.backward()
            accumulated_loss += float(loss.detach().float().cpu())
            heartbeat("micro_batch_completed", global_step=global_step, micro_step=micro_step)
        heartbeat("optimizer_step_started", global_step=global_step)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        applied_learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        heartbeat("optimizer_step_completed", global_step=global_step)

        record: dict[str, Any] = {
            "global_step": global_step,
            "train_loss": accumulated_loss,
            "applied_learning_rate": applied_learning_rate,
            "next_learning_rate": scheduler.get_last_lr()[0],
            "sampled_sources": dict(sampled_sources),
        }
        evaluate_now = not args.skip_evaluation and (
            global_step % int(config["eval_every_steps"]) == 0
            or global_step == total_steps
            or (bool(config["evaluate_at_segment_end"]) and global_step == end_step)
        )
        if evaluate_now:
            heartbeat("validation_started", global_step=global_step, validation_rows=len(validation_subset))
            validation_loss = evaluate(
                model,
                validation_subset,
                pad_token_id,
                device,
                on_progress=lambda row_index, row_count: heartbeat(
                    "validation_row_started",
                    global_step=global_step,
                    validation_row=row_index,
                    validation_rows=row_count,
                ),
            )
            record["validation_loss"] = validation_loss
            record["validation_rows"] = len(validation_subset)
            if validation_loss < best_loss:
                heartbeat("best_adapter_saving", global_step=global_step, validation_loss=validation_loss)
                best_loss = validation_loss
                best_dir = output_dir / "best"
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    heartbeat("checkpoint_saving", global_step=end_step)
    checkpoint_metadata = {
        "role": config["role"],
        "adapter_name": config["adapter_name"],
        "base_model": config["base_model"],
        "model_mode": model_mode,
        "global_step": end_step,
        "total_steps": total_steps,
        "segment_steps_completed": end_step - start_step,
        "best_validation_loss": None if math.isinf(best_loss) else best_loss,
        "encoded_train_rows": len(train_encoded),
        "encoded_validation_rows": len(validation_encoded),
        "validation_subset_rows": len(validation_subset),
        "target_modules_regex": target_modules,
        "catalog": str(config["catalog"]),
        "train_sources": config["train_sources"],
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint = checkpoint_dir(output_dir, end_step)
    save_checkpoint(checkpoint, model, processor, optimizer, scheduler, sampler_generator, checkpoint_metadata)
    heartbeat("checkpoint_saved", global_step=end_step, checkpoint=str(checkpoint))
    complete = end_step == total_steps
    if complete:
        heartbeat("final_adapter_saving", global_step=end_step)
        final_dir = output_dir / "final"
        model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
    run_manifest = {
        **checkpoint_metadata,
        "status": "complete" if complete else "segment_complete_resume_required",
        "latest_checkpoint": str(checkpoint),
        "final_adapter": str(output_dir / "final") if complete else None,
        "evaluation_skipped": args.skip_evaluation,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_run_manifest(output_dir, run_manifest)
    heartbeat("complete" if complete else "segment_complete", global_step=end_step, manifest_status=run_manifest["status"])
    print(json.dumps(run_manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
