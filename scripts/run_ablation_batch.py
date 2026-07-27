#!/usr/bin/env python3
"""Run a resumable, sequential B0/B1/B2 ablation on held-out A1 cases.

The script never parallelises model calls: one Qwen3-4B base and at most one
role adapter occupy the GPU.  B3 is intentionally refused while the A4 teacher
API account has no quota.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agent_kg.orchestration.ablation import load_terminal_result, stable_run_id, summarize_results
from agent_kg.orchestration.e2e_smoke import a3_prompt, draft_from_a3_payload, normalize_a1_payload, record_a1_to_a2
from agent_kg.orchestration.orchestrator import AgentWorkflowOrchestrator
from agent_kg.runtime import RoleSandboxError, SharedQwenRoleRuntime


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ("baseline_silver", "external_full_causal_chain", "external_reported_trigger_only")


class GenerationContractError(RoleSandboxError):
    """A failed generation together with its recorded retry attempts."""

    def __init__(self, message: str, attempts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = attempts


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def messages_value(record: dict[str, Any], role: str) -> str:
    for message in record.get("messages", []):
        if message.get("role") == role and isinstance(message.get("content"), str):
            return message["content"]
    raise ValueError(f"Dataset record has no {role!r} message.")


def evidence_ids(record: dict[str, Any]) -> list[str]:
    target = json.loads(messages_value(record, "assistant"))
    # Legacy silver A1 samples use ``evidence_id`` while the external
    # source-audited extension uses ``source_id``.  Both identify the
    # evidence token that the role sandbox must whitelist; treating either
    # schema as absent silently turns a data-contract mismatch into a fake
    # model structural failure.
    return [
        str(item.get("source_id") or item.get("evidence_id"))
        for item in target.get("evidence", [])
        if isinstance(item, dict) and (item.get("source_id") or item.get("evidence_id"))
    ]


def load_records(catalog: dict[str, Any], sources: list[str], limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in sources:
        try:
            test_path = Path(catalog["a1"][source]["test"])
        except KeyError as exc:
            raise ValueError(f"Unknown A1 evaluation source: {source}") from exc
        records.extend(read_jsonl(test_path))
    return records[:limit] if limit is not None else records


def generate_with_retry(
    runtime: SharedQwenRoleRuntime,
    role: str,
    prompt: str,
    allowed_ids: list[str],
    *,
    use_adapter: bool,
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Retry exactly once for a contract failure, preserving both attempts."""

    attempts: list[dict[str, Any]] = []
    for budget in (max_new_tokens, max_new_tokens * 2):
        started = time.perf_counter()
        try:
            payload = runtime.generate_json(role, prompt, allowed_ids, max_new_tokens=budget, use_adapter=use_adapter)
            attempts.append({"max_new_tokens": budget, "status": "complete", "latency_seconds": round(time.perf_counter() - started, 3)})
            return payload, attempts
        except (RoleSandboxError, RuntimeError, ValueError, OSError) as exc:
            attempts.append({"max_new_tokens": budget, "status": "structural_failure", "error": str(exc), "latency_seconds": round(time.perf_counter() - started, 3)})
    raise GenerationContractError(f"{role} failed its JSON contract after {len(attempts)} recorded attempts.", attempts)


def run_one(
    runtime: SharedQwenRoleRuntime,
    experiment_id: str,
    spec: dict[str, Any],
    record: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    run_id = stable_run_id(experiment_id, record)
    started = time.perf_counter()
    query = messages_value(record, "user")
    allowed_ids = evidence_ids(record)
    result: dict[str, Any] = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "dataset_sample_id": record.get("metadata", {}).get("sample_id"),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "controls": {
            "a1_adapter": bool(spec["a1_adapter"]),
            "a2_gate": bool(spec["a2_gate"]),
            "a3_adapter": bool(spec["a3_adapter"]),
            "a4_review": bool(spec["a4_review"]),
            "max_new_tokens": max_new_tokens,
            "decode": "greedy; thinking disabled",
        },
        "a1_attempts": [],
        "a3_attempts": [],
    }
    try:
        try:
            a1_raw, a1_attempts = generate_with_retry(
                runtime, "a1", query, allowed_ids, use_adapter=bool(spec["a1_adapter"]), max_new_tokens=max_new_tokens
            )
        except GenerationContractError as exc:
            result["a1_attempts"] = exc.attempts
            raise
        result["a1_attempts"] = a1_attempts
        result["a1_raw"] = a1_raw
        normalized = normalize_a1_payload(
            a1_raw,
            run_id=run_id,
            query=query,
            model_profile=("a1_lora" if spec["a1_adapter"] else "base_qwen"),
            allowed_evidence_ids=allowed_ids,
        )
        result["a1_type_normalizations"] = normalized.type_normalizations
        result["a1_evidence_normalizations"] = normalized.evidence_normalizations
        result["a1_predicate_normalizations"] = normalized.predicate_normalizations

        if spec["a2_gate"]:
            session = record_a1_to_a2(AgentWorkflowOrchestrator(), normalized)
            result["a2_result"] = session.a2_result.model_dump(mode="json") if session.a2_result else None
            result["workflow_state_after_a2"] = session.state.value
            accepted_claims = session.a2_result.accepted_claims if session.a2_result else []
            if not accepted_claims:
                result["status"] = "blocked_by_a2"
                result["a3_skipped_reason"] = "A2 admitted no diagnosis-eligible claim."
                return result
        else:
            session = None
            accepted_claims = normalized.proposal.claims
            result["a2_result"] = None
            result["workflow_state_after_a2"] = "a2_bypassed_by_ablation_design"

        prompt = a3_prompt(query, normalized.proposal, accepted_claims)
        try:
            a3_raw, a3_attempts = generate_with_retry(
                runtime, "a3", prompt, allowed_ids, use_adapter=bool(spec["a3_adapter"]), max_new_tokens=max_new_tokens
            )
        except GenerationContractError as exc:
            result["a3_attempts"] = exc.attempts
            raise
        result["a3_attempts"] = a3_attempts
        result["a3_raw"] = a3_raw
        draft = draft_from_a3_payload(run_id, a3_raw, normalized.proposal.evidence, "a3_lora" if spec["a3_adapter"] else "base_qwen")
        if session is not None:
            AgentWorkflowOrchestrator.submit_a3_draft(session, draft)
            result["workflow_state_after_a3"] = session.state.value
            result["workflow_audit"] = session.audit_payload()
        else:
            result["workflow_state_after_a3"] = "a3_draft_received_without_a2_by_ablation_design"
        result["status"] = "complete"
    except (RoleSandboxError, RuntimeError, ValueError, OSError) as exc:
        result["status"] = "structural_failure"
        result["error"] = str(exc)
    finally:
        result["latency_seconds"] = round(time.perf_counter() - started, 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="B2_dual_lora_a2")
    parser.add_argument("--source", action="append", choices=DEFAULT_SOURCES, help="May be repeated; default uses all held-out sources.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true", help="Skip only terminal, matching per-sample result files.")
    parser.add_argument("--dry-run", action="store_true", help="Validate registry, case selection and result paths without model/GPU calls.")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    registry = yaml.safe_load((ROOT / "config" / "experiments" / "ablation_matrix.yaml").read_text(encoding="utf-8"))
    experiments = registry.get("experiments", {})
    if args.experiment not in experiments:
        parser.error(f"Unknown experiment: {args.experiment}")
    spec = experiments[args.experiment]
    if spec.get("a4_review"):
        parser.error("B3/A4 is blocked until the configured API account has usable quota; no paid calls were attempted.")
    catalog_path = ROOT / registry["shared_controls"]["test_catalog"]
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    records = load_records(catalog, args.source or list(DEFAULT_SOURCES), args.limit)
    run_dir = ROOT / "experiments" / "ablation" / "runs" / args.experiment
    results: list[dict[str, Any]] = []
    skipped = 0
    runtime = None if args.dry_run else SharedQwenRoleRuntime.from_project_config(ROOT / "config" / "agents.yaml")
    try:
        for record in records:
            run_id = stable_run_id(args.experiment, record)
            target = run_dir / f"{run_id}.json"
            prior = load_terminal_result(target, args.experiment, run_id) if args.resume else None
            if prior is not None:
                results.append(prior)
                skipped += 1
                continue
            if args.dry_run:
                result = {"run_id": run_id, "experiment_id": args.experiment, "status": "planned_dry_run", "result_path": str(target)}
            else:
                assert runtime is not None
                result = run_one(runtime, args.experiment, spec, record, args.max_new_tokens)
                atomic_json_write(target, result)
            results.append(result)
    finally:
        if runtime is not None:
            runtime.unload()
    summary = summarize_results(results)
    summary.update({"experiment_id": args.experiment, "dry_run": args.dry_run, "resumed_terminal_cases": skipped, "sources": args.source or list(DEFAULT_SOURCES)})
    summary_path = ROOT / "experiments" / "ablation" / f"{args.experiment}_latest_summary.json"
    atomic_json_write(summary_path, summary)
    print(json.dumps({"status": "complete", "summary": summary, "summary_path": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
