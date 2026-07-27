#!/usr/bin/env python3
"""Confirmation-only E0/E2 interface evaluation on post-design official cases.

No item in the input JSONL may be used for LoRA training or rule design.  The
script reports only contracts, evidence isolation and A2 tiering; it has no
human relation gold and makes no semantic-accuracy claim.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_kg.agents.a2_quality_control import A2QualityController
from agent_kg.contracts import A3DiagnosisDraft
from agent_kg.evaluation.role_metrics import evidence_ids
from agent_kg.orchestration.e2e_smoke import a3_prompt, normalize_a1_payload, record_a1_to_a2
from agent_kg.orchestration.orchestrator import AgentWorkflowOrchestrator
from agent_kg.runtime import SharedQwenRoleRuntime


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "confirmation" / "official_2026_confirmation_cases_v1.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def a1_prompt(case: dict[str, Any]) -> str:
    evidence = case["evidence"]
    blocks = "\n".join(f"[{item['source_id']}] {item['excerpt']}" for item in evidence)
    return (
        "任务：从来源可追溯的电力事故案例中提取候选实体、候选关系、属性和证据。\n"
        "只依据下列证据；每个实体和关系引用 evidence_ids；候选不能声称已写入知识图谱。\n"
        f"案例 ID：{case['case_id']}\n来源：{case['source_title']}\n证据：\n{blocks}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="official_2026_confirmation_pilot_v1")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--a1-max-new-tokens", type=int, default=1024)
    parser.add_argument("--a3-max-new-tokens", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cases = read_jsonl(CASES)[: args.limit]
    out = ROOT / "experiments" / "confirmation" / args.label
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps({"status": "planned", "output": str(out), "sample_count": len(cases)}, ensure_ascii=False))
        return
    runtime = SharedQwenRoleRuntime.from_project_config(ROOT / "config" / "agents.yaml")
    intermediate: list[dict[str, Any]] = []
    try:
        # A1 is evaluated exactly once per case; downstream arms replay it.
        for case in cases:
            prompt = a1_prompt(case)
            allowed_ids = [item["source_id"] for item in case["evidence"]]
            raw = runtime.generate_json("a1", prompt, allowed_ids, max_new_tokens=args.a1_max_new_tokens)
            bridge = dict(raw)
            bridge["evidence"] = case["evidence"]  # immutable source bundle, not model-generated evidence
            intermediate.append({"case": case, "prompt": prompt, "raw": raw, "bridge": bridge, "allowed_ids": allowed_ids})

        rows: list[dict[str, Any]] = []
        for item in intermediate:
            for experiment_id, mode in (("E0_raw", "raw"), ("E2_full", "full")):
                normalized = normalize_a1_payload(item["bridge"], run_id=f"{experiment_id}--{item['case']['case_id']}", query=item["prompt"], model_profile="a1_qwen3_4b_confirmation", normalization_mode=mode, allowed_evidence_ids=item["allowed_ids"])
                orchestrator = AgentWorkflowOrchestrator(A2QualityController())
                session = record_a1_to_a2(orchestrator, normalized)
                result = session.a2_result
                assert result is not None
                claims = result.accepted_claims or result.held_claims
                tier = "confirmed" if result.accepted_claims else "candidate_only"
                if not result.accepted_claims and result.held_claims:
                    session.admit_candidate_graph_for_exploration(len(result.held_claims), "Confirmation candidate-only route; never promote to core graph.")
                row: dict[str, Any] = {"case_id": item["case"]["case_id"], "experiment_id": experiment_id, "normalization_mode": mode, "a1_raw": item["raw"], "a2_metrics": result.operational_metrics.model_dump(mode="json"), "normalization_audit": {"types": normalized.type_normalizations, "predicates": normalized.predicate_normalizations}, "a3_claim_tier": tier}
                if not claims:
                    row["status"] = "blocked_by_a2"
                else:
                    prediction = runtime.generate_json("a3", a3_prompt(item["prompt"], normalized.proposal, claims, claim_tier=tier), item["allowed_ids"], max_new_tokens=args.a3_max_new_tokens)
                    draft = A3DiagnosisDraft(run_id=normalized.proposal.run_id, diagnosis_object=prediction, report_sections=[], evidence=normalized.proposal.evidence, model_profile="a3_qwen3_4b_confirmation")
                    orchestrator.submit_a3_draft(session, draft)
                    cited = set(evidence_ids(prediction))
                    row.update({"status": "complete", "a3_prediction": prediction, "a3_json_valid": True, "a3_current_case_citation_valid": cited.issubset(set(item["allowed_ids"])), "workflow_state": session.state.value})
                rows.append(row)
    finally:
        runtime.unload()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["experiment_id"]].append(row)
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "scope": "confirmation-only post-design official cases; no training, no rule design, no semantic gold labels", "sample_count": len(cases), "groups": {key: {"n": len(values), "completed": sum(row["status"] == "complete" for row in values), "mean_confirmed": sum(row["a2_metrics"]["accepted_claims"] for row in values) / len(values), "mean_candidate": sum(row["a2_metrics"]["held_claims"] for row in values) / len(values), "a3_json_valid_rate": sum(bool(row.get("a3_json_valid")) for row in values) / len(values), "a3_current_case_citation_valid_rate": sum(bool(row.get("a3_current_case_citation_valid")) for row in values) / len(values)} for key, values in groups.items()}}
    (out / "per_case.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(out), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
