#!/usr/bin/env python3
"""Run one held-out A1 -> A2 -> A3 smoke experiment with full local audit."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_kg.evaluation.role_metrics import assistant_json
from agent_kg.orchestration.e2e_smoke import a3_prompt, draft_from_a3_payload, normalize_a1_payload, record_a1_to_a2
from agent_kg.orchestration.orchestrator import AgentWorkflowOrchestrator
from agent_kg.runtime import SharedQwenRoleRuntime


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_case(source: str) -> dict[str, Any]:
    catalog = json.loads((ROOT / "data" / "manifests" / "fine_tuning_catalog_v0.3.json").read_text(encoding="utf-8"))
    records = read_jsonl(Path(catalog["a1"][source]["test"]))
    if not records:
        raise ValueError(f"No A1 held-out test case for source: {source}")
    return records[0]


def user_prompt(record: dict[str, Any]) -> str:
    return next(message["content"] for message in record["messages"] if message["role"] == "user")


def allowed_evidence(record: dict[str, Any]) -> list[str]:
    target = assistant_json(record)
    return [item["source_id"] for item in target.get("evidence", []) if isinstance(item, dict) and item.get("source_id")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="external_reported_trigger_only", choices=["baseline_silver", "external_full_causal_chain", "external_reported_trigger_only"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    record = select_case(args.source)
    query = user_prompt(record)
    evidence_ids = allowed_evidence(record)
    run_id = f"e2e-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    runtime = SharedQwenRoleRuntime.from_project_config(ROOT / "config" / "agents.yaml")
    orchestrator = AgentWorkflowOrchestrator()
    try:
        a1_raw = runtime.generate_json("a1", query, evidence_ids, max_new_tokens=args.max_new_tokens)
        normalized = normalize_a1_payload(a1_raw, run_id=run_id, query=query, model_profile="a1_graph_builder_lora")
        session = record_a1_to_a2(orchestrator, normalized)
        output: dict[str, Any] = {
            "run_id": run_id,
            "source": args.source,
            "a1_raw": a1_raw,
            "a1_type_normalizations": normalized.type_normalizations,
            "a2_result": session.a2_result.model_dump(mode="json") if session.a2_result else None,
            "workflow_state_after_a2": session.state.value,
        }
        if session.a2_result and session.a2_result.accepted_claims:
            prompt = a3_prompt(query, normalized.proposal, session.a2_result.accepted_claims)
            a3_raw = runtime.generate_json("a3", prompt, evidence_ids, max_new_tokens=args.max_new_tokens)
            draft = draft_from_a3_payload(run_id, a3_raw, normalized.proposal.evidence, "a3_diagnosis_lora")
            orchestrator.submit_a3_draft(session, draft)
            output["a3_raw"] = a3_raw
            output["workflow_state_after_a3"] = session.state.value
        else:
            output["a3_skipped_reason"] = "A2 accepted no diagnosis-eligible claims."
        target = ROOT / "experiments" / "runs" / f"{run_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "complete", "run_id": run_id, "output": str(target), "state": session.state.value}, ensure_ascii=False))
    finally:
        runtime.unload()


if __name__ == "__main__":
    main()
