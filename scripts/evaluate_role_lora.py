#!/usr/bin/env python3
"""Evaluate an A1 or A3 adapter on the manifest's held-out test splits.

Without ``--run-model`` this validates the split and emits a gold-only
evaluation manifest.  ``--run-model`` sequentially loads exactly one adapter
and records JSON validity, exact structured overlap, and evidence grounding.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_kg.evaluation.role_metrics import (
    a1_claim_set,
    a1_entity_set,
    a3_statement_set,
    aggregate,
    assistant_json,
    evidence_ids,
    prf,
)
from agent_kg.runtime import RoleSandboxError, SharedQwenRoleRuntime
from agent_kg.config import load_agent_config


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_records(role: str) -> list[dict[str, Any]]:
    catalog = json.loads((ROOT / "data" / "manifests" / "fine_tuning_catalog_v0.3.json").read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for source_name, paths in catalog[role].items():
        if source_name in {"counts", "suggested_train_sampling_weight"}:
            continue
        for record in read_jsonl(Path(paths["test"])):
            record["_source_name"] = source_name
            records.append(record)
    return records


def prompt_and_allowed_evidence(record: dict[str, Any]) -> tuple[str, list[str]]:
    messages = record["messages"]
    prompt = next(item["content"] for item in messages if item["role"] == "user")
    gold = assistant_json(record)
    # The sandbox must expose every evidence item supplied to the model, not
    # merely the subset cited by the reference output.  Restricting to gold
    # citations creates false "outside sandbox" failures when an otherwise
    # grounded prediction cites another prompt-provided item.
    prompt_evidence = set(re.findall(r"(?m)^\[([A-Za-z0-9_.:-]+)\]", prompt))
    allowed = sorted(prompt_evidence | evidence_ids(gold))
    if not allowed:
        raise ValueError(f"{record.get('metadata', {}).get('sample_id')} has no target evidence IDs")
    return prompt, allowed


def score(role: str, gold: dict[str, Any], prediction: dict[str, Any], *, sample_id: str, source_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_id": sample_id,
        "source_name": source_name,
        "json_valid": True,
        "evidence": prf(evidence_ids(prediction), evidence_ids(gold)),
    }
    if role == "a1":
        result["entity"] = prf(a1_entity_set(prediction), a1_entity_set(gold))
        result["claim"] = prf(a1_claim_set(prediction), a1_claim_set(gold))
    else:
        result["statement"] = prf(a3_statement_set(prediction), a3_statement_set(gold))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["a1", "a3"], required=True)
    parser.add_argument("--run-model", action="store_true", help="Run the completed local LoRA adapter on the test split.")
    parser.add_argument("--source", help="Evaluate one manifest source tier only (for example external_reported_trigger_only).")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of held-out samples; 0 means all.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--adapter-name", help="Configured adapter key for this role; enables registered v1/v2 comparisons.")
    parser.add_argument("--label", help="Human-readable evaluation label included in the output directory.")
    args = parser.parse_args()
    records = test_records(args.role)
    if args.source:
        records = [record for record in records if record["_source_name"] == args.source]
        if not records:
            raise ValueError(f"No held-out {args.role} records found for source tier: {args.source}")
    if args.limit:
        records = records[: args.limit]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "experiments" / "evaluation" / f"{args.role}_{args.label + '_' if args.label else ''}{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "role": args.role,
        "mode": "model" if args.run_model else "gold_only",
        "sample_count": len(records),
        "source_counts": {source: sum(item["_source_name"] == source for item in records) for source in sorted({item["_source_name"] for item in records})},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.run_model:
        print(json.dumps({"status": "gold_only_manifest_created", "output_dir": str(output_dir), **manifest}, ensure_ascii=False))
        return
    config = load_agent_config(ROOT / "config" / "agents.yaml")
    if args.adapter_name:
        if args.adapter_name not in config.lora_adapters:
            raise ValueError(f"Unknown configured adapter: {args.adapter_name}")
        config.agents[args.role].adapter = args.adapter_name
    runtime = SharedQwenRoleRuntime(config)
    scored: list[dict[str, Any]] = []
    prediction_lines: list[str] = []
    try:
        for record in records:
            sample_id = record["metadata"]["sample_id"]
            gold = assistant_json(record)
            prompt, allowed = prompt_and_allowed_evidence(record)
            try:
                prediction = runtime.generate_json(args.role, prompt, allowed, max_new_tokens=args.max_new_tokens)
                row = score(args.role, gold, prediction, sample_id=sample_id, source_name=record["_source_name"])
                prediction_lines.append(json.dumps({"sample_id": sample_id, "prediction": prediction}, ensure_ascii=False))
            except (RoleSandboxError, RuntimeError) as exc:
                row = {"sample_id": sample_id, "source_name": record["_source_name"], "json_valid": False, "error": str(exc)}
            scored.append(row)
    finally:
        runtime.unload()
    (output_dir / "per_sample.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in scored) + "\n", encoding="utf-8")
    (output_dir / "predictions.jsonl").write_text("\n".join(prediction_lines) + ("\n" if prediction_lines else ""), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(aggregate(scored), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "evaluation_complete", "output_dir": str(output_dir), "summary": aggregate(scored)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
