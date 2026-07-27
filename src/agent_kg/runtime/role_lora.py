"""Single-GPU Qwen3-4B runtime that activates exactly one role LoRA per request.

The role sandbox is evaluated *before* any adapter is loaded.  It constrains
the prompt, evidence IDs and output contract; LoRA only specializes language
behaviour and never grants additional data or write permissions.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

from ..agents.a1_graph_builder import A1GraphBuilder
from ..agents.a3_diagnosis import A3DiagnosisAgent
from ..config import AgentProjectConfig, load_agent_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class RoleSandboxError(ValueError):
    """A request that violates role, evidence, or adapter isolation."""


class SharedQwenRoleRuntime:
    """Load one completed PEFT adapter at a time on the frozen shared backbone."""

    _instructions = {
        "a1": A1GraphBuilder.system_instruction,
        "a3": A3DiagnosisAgent.system_instruction,
    }
    _required_output_keys = {
        "a1": {"entities", "claims", "attributes", "evidence"},
        # A3 held-out cases cover observation-only, causal-chain and complete
        # report tasks, so a valid response must expose at least one contract
        # field rather than a nested evidence item.
        "a3": {"observations", "root_cause_assessment", "recommended_actions", "report_draft", "claim_reviews"},
    }

    def __init__(self, config: AgentProjectConfig, *, backbone_name: str = "m1_qwen3_4b") -> None:
        if backbone_name not in config.shared_backbones:
            raise RoleSandboxError(f"Unknown shared backbone: {backbone_name}")
        self.config = config
        self.backbone_name = backbone_name
        self.backbone = config.shared_backbones[backbone_name]
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._active_role: str | None = None
        self._adapter_enabled: bool | None = None

    @classmethod
    def from_project_config(cls, config_path: str | Path = "config/agents.yaml") -> "SharedQwenRoleRuntime":
        return cls(load_agent_config(config_path))

    def validate_request(self, role: str, evidence_ids: list[str], *, use_adapter: bool = True) -> Path | None:
        """Validate the sandbox before model/adapters can receive a request."""

        if role not in self._instructions or role not in self.backbone.roles:
            raise RoleSandboxError(f"Role {role!r} is not permitted on {self.backbone_name}.")
        if not evidence_ids or any(not isinstance(item, str) or not item.strip() for item in evidence_ids):
            raise RoleSandboxError("A role request requires explicit, non-empty evidence IDs.")
        agent = self.config.agents[role]
        if "baseline.kg.write" not in agent.sandbox.get("deny", []):
            raise RoleSandboxError(f"{role} sandbox must deny baseline graph writes.")
        # B0 is the frozen-backbone control.  It retains exactly the same
        # sandbox and output contract as the LoRA conditions; only adapter
        # activation changes.  This avoids accidentally measuring a different
        # prompt or evidence policy instead of the LoRA contribution.
        if not use_adapter:
            return None
        adapter_name = agent.adapter
        if not adapter_name or adapter_name not in self.config.lora_adapters:
            raise RoleSandboxError(f"{role} has no configured LoRA adapter.")
        adapter = self.config.lora_adapters[adapter_name]
        if adapter.role != role or adapter.backbone != self.backbone_name:
            raise RoleSandboxError("Adapter role/backbone does not match the requested sandbox.")
        completed_statuses = (
            "formal_training_complete",
            # A registered pilot may be loaded only to evaluate its frozen
            # held-out behaviour.  Its status remains visibly distinct from
            # a selected formal adapter and cannot be mistaken for default
            # production routing.
            "pilot_training_complete_pending_frozen_evaluation",
        )
        if not adapter.status.startswith(completed_statuses):
            raise RoleSandboxError(f"Adapter {adapter_name} is not a completed training artifact.")
        adapter_dir = Path(adapter.output_dir)
        if not (adapter_dir / "adapter_config.json").is_file():
            raise RoleSandboxError(f"Adapter files are incomplete: {adapter_dir}")
        return adapter_dir

    def build_messages(
        self,
        role: str,
        user_prompt: str,
        evidence_ids: list[str],
        *,
        use_adapter: bool = True,
        contract_override: set[str] | None = None,
        instruction_override: str | None = None,
    ) -> list[dict[str, str]]:
        if not user_prompt.strip():
            raise RoleSandboxError("Role request requires a non-empty user prompt.")
        self.validate_request(role, evidence_ids, use_adapter=use_adapter)
        instruction = instruction_override or self._instructions[role]
        guardrail = (
            "\n沙盒约束：仅使用请求中提供的证据；允许引用的 evidence_ids 为 "
            + json.dumps(sorted(set(evidence_ids)), ensure_ascii=False)
            + "。不得调用工具、不得写入图谱、不得写最终报告；输出严格 JSON。"
        )
        if role == "a1" and contract_override is not None:
            # Structured orchestration may intentionally request a small core
            # contract and deterministically add empty optional fields later.
            # This is an explicit, auditable runtime mode—not a relaxation of
            # the normal A1 production contract.
            guardrail += (
                "For this explicitly declared structured-evaluation request, "
                "the top-level JSON must contain exactly the required core keys: "
                + json.dumps(sorted(contract_override), ensure_ascii=False)
                + ". Do not emit other top-level fields."
            )
        elif role == "a1":
            guardrail += "顶层 JSON 必须同时包含 entities、claims、attributes、evidence 四个键，禁止只输出单个实体或关系。"
        else:
            guardrail += "顶层 JSON 必须至少包含 observations、root_cause_assessment、recommended_actions、report_draft 中的一个任务字段。"
        return [
            {"role": "system", "content": instruction + guardrail},
            {"role": "user", "content": user_prompt},
        ]

    def load_role(self, role: str, evidence_ids: list[str], *, use_adapter: bool = True) -> None:
        """Load the frozen base and precisely one PEFT adapter; never co-load A1/A3."""

        adapter_dir = self.validate_request(role, evidence_ids, use_adapter=use_adapter)
        if self._active_role == role and self._adapter_enabled == use_adapter and self._model is not None:
            return
        self.unload()
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:  # Keeps local unit tests independent of GPU extras.
            raise RuntimeError("Install the project's [train] dependencies before local LoRA inference.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Local Qwen3-4B LoRA inference requires a CUDA GPU.")
        if self.backbone.hf_home:
            os.environ["HF_HOME"] = self.backbone.hf_home
            # Do not inherit a stale global HF_HUB_CACHE from another Python
            # process.  The formal adapters and the Qwen3-4B snapshot live in
            # this project's configured D: cache and must resolve together.
            os.environ["HF_HUB_CACHE"] = str(Path(self.backbone.hf_home) / "hub")
            os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]
        # The formal training run already populated the D: cache.  For
        # evaluation, fail fast if it is unavailable rather than silently
        # downloading weights or competing with the research workflow.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_source = self._local_model_source()
        self._tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(
            model_source,
            quantization_config=quantization,
            dtype=torch.bfloat16,
            device_map={"": 0},
            local_files_only=True,
        )
        self._model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False) if use_adapter else base
        self._model.eval()
        self._active_role = role
        self._adapter_enabled = use_adapter

    def _local_model_source(self) -> str:
        """Resolve the exact cached HF snapshot, without cache-index ambiguity.

        Hugging Face's cache lookup can inherit a process-level cache setting
        before the configured project cache is applied.  Training artifacts
        are tied to one exact Qwen snapshot, so direct local loading is both
        more reproducible and safer than allowing a background run to fetch a
        different revision.
        """

        if not self.backbone.hf_home or not self.backbone.hf_base_model:
            raise RuntimeError("The shared backbone requires hf_home and hf_base_model for offline inference.")
        repository = "models--" + self.backbone.hf_base_model.replace("/", "--")
        cache_root = Path(self.backbone.hf_home) / "hub" / repository
        ref_path = cache_root / "refs" / "main"
        snapshot_id = ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else ""
        candidate = cache_root / "snapshots" / snapshot_id
        if not snapshot_id or not (candidate / "config.json").is_file():
            raise RuntimeError(f"Local model snapshot is incomplete: {candidate}")
        return str(candidate)

    def generate_json(
        self,
        role: str,
        user_prompt: str,
        evidence_ids: list[str],
        *,
        max_new_tokens: int = 512,
        use_adapter: bool = True,
        contract_override: set[str] | None = None,
        instruction_override: str | None = None,
    ) -> dict[str, Any]:
        """Generate one sandboxed JSON response and reject ungrounded citations."""

        messages = self.build_messages(
            role,
            user_prompt,
            evidence_ids,
            use_adapter=use_adapter,
            contract_override=contract_override,
            instruction_override=instruction_override,
        )
        self.load_role(role, evidence_ids, use_adapter=use_adapter)
        assert self._model is not None and self._tokenizer is not None
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        try:
            encoded = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:  # Compatibility with older tokenizer templates.
            encoded = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        # Qwen tokenizer versions may return either a tensor or BatchEncoding.
        # ``generate`` needs the concrete input_ids tensor plus its mask, not a
        # BatchEncoding object passed positionally.
        if hasattr(encoded, "keys"):
            input_ids = encoded["input_ids"].to(self._model.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._model.device)
        else:
            input_ids = encoded.to(self._model.device)
            attention_mask = None
        prompt_length = input_ids.shape[-1]

        class _StopAtFirstCompleteJSONObject(StoppingCriteria):
            """Stop deterministic decoding as soon as one complete JSON object exists."""

            def __call__(self, generated_ids: Any, scores: Any, **_: Any) -> bool:
                text = self_tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=True).strip()
                decoder = json.JSONDecoder()
                start = text.find("{")
                if start < 0:
                    return False
                try:
                    payload, _ = decoder.raw_decode(text[start:])
                except json.JSONDecodeError:
                    return False
                return isinstance(payload, dict)

        self_tokenizer = self._tokenizer
        with torch.inference_mode():
            output = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([_StopAtFirstCompleteJSONObject()]),
            )
        generated = output[0, input_ids.shape[-1] :]
        content = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        required_keys = contract_override or self._required_output_keys[role]
        payload = self._parse_json_object(
            content,
            role,
            required_keys=required_keys,
            require_all=role == "a1" and contract_override is None,
        )
        if not isinstance(payload, dict):
            raise RoleSandboxError(f"{role} adapter JSON output must be an object.")
        cited = self._collect_evidence_ids(payload)
        unknown = cited - set(evidence_ids)
        if unknown:
            raise RoleSandboxError(f"{role} output cited evidence outside the sandbox: {sorted(unknown)}")
        return payload

    def unload(self) -> None:
        """Release adapter/base memory before a different role obtains the single GPU."""

        self._model = None
        self._tokenizer = None
        self._active_role = None
        self._adapter_enabled = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def _collect_evidence_ids(value: Any) -> set[str]:
        if isinstance(value, dict):
            result = set(value.get("evidence_ids", [])) if isinstance(value.get("evidence_ids"), list) else set()
            for item in value.values():
                result |= SharedQwenRoleRuntime._collect_evidence_ids(item)
            return {item for item in result if isinstance(item, str)}
        if isinstance(value, list):
            return set().union(*(SharedQwenRoleRuntime._collect_evidence_ids(item) for item in value)) if value else set()
        return set()

    @staticmethod
    def _parse_json_object(content: str, role: str, *, required_keys: set[str], require_all: bool | None = None) -> dict[str, Any]:
        """Parse only the final JSON object, ignoring fences or preambles.

        Some local Qwen inference stacks prepend non-contract text even when a
        JSON-only instruction is supplied.  It is never returned or logged;
        only the first decodable object is admitted to the role sandbox.
        """

        decoder = json.JSONDecoder()
        for index, character in enumerate(content):
            if character != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(content[index:])
            except json.JSONDecodeError:
                continue
            must_include_all = role == "a1" if require_all is None else require_all
            if isinstance(payload, dict) and (required_keys <= set(payload) if must_include_all else bool(required_keys & set(payload))):
                return payload
        first_character = next((character for character in content if not character.isspace()), "<empty>")
        mentioned = sorted(key for key in required_keys if f'"{key}"' in content)
        # Keep a bounded, escaped generation preview for reproducible local
        # failure diagnosis.  This is model output only (not the source
        # prompt), and avoids silently treating malformed JSON as a valid
        # extraction result.
        preview = content[:400].replace("\n", "\\n").replace("\r", "\\r")
        raise RoleSandboxError(
            f"{role} adapter returned no complete top-level contract JSON "
            f"(chars={len(content)}, first={first_character!r}, contract_keys_seen={mentioned}, "
            f"output_preview={preview!r})."
        )
