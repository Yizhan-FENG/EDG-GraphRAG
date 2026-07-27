#!/usr/bin/env python3
"""Paired A1→A2→A3 interface ablation over frozen A1 raw outputs.

E0/E1/E2 replay identical saved A1 outputs.  This isolates the A1-to-A2
ontology interface; it does not claim a fresh A1 generation comparison.
Candidate-only claims are labeled as such in the A3 input and never promoted.
"""

from __future__ import annotations

import argparse
import json
import re
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
SOURCE = ROOT / "experiments" / "ablation" / "runs" / "B2_dual_lora_a2_type_normalized_evidencefixed"
MODES = {"E0_raw": "raw", "E1_alias_only": "alias_only", "E2_full": "full"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def a3_record_by_case() -> dict[str, dict[str, Any]]:
    records = read_jsonl(ROOT / "data" / "a3" / "external_source_audited" / "test.jsonl")
    output: dict[str, dict[str, Any]] = {}
    for row in records:
        sample_id = str(row["metadata"]["sample_id"])
        match = re.search(r"ext-(\d+)-full_report$", sample_id)
        if match:
            output[f"EXT-{match.group(1)}"] = row
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="e2e_normalization_ablation_pilot_v1")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sources = sorted(SOURCE.glob("*a1-ext*.json"))[: args.limit]
    reports = a3_record_by_case()
    out = ROOT / "experiments" / "e2e_ablation" / args.label
    out.mkdir(parents=True, exist_ok=True)
    runtime = None if args.dry_run else SharedQwenRoleRuntime.from_project_config(ROOT / "config" / "agents.yaml")
    rows: list[dict[str, Any]] = []
    try:
        for source in sources:
            saved = json.loads(source.read_text(encoding="utf-8"))
            raw = saved["a1_raw"]
            sample_id = str(saved["dataset_sample_id"])
            case_id = "EXT-" + sample_id.split("-ext-")[1].split("-")[0]
            report = reports[case_id]
            query = next(item["content"] for item in report["messages"] if item["role"] == "user")
            for experiment_id, mode in MODES.items():
                normalized = normalize_a1_payload(raw, run_id=f"{experiment_id}--{sample_id}", query=query, model_profile="frozen_a1_raw", normalization_mode=mode)
                orchestrator = AgentWorkflowOrchestrator(A2QualityController())
                session = record_a1_to_a2(orchestrator, normalized)
                result = session.a2_result
                assert result is not None
                claims = result.accepted_claims or result.held_claims
                tier = "confirmed" if result.accepted_claims else "candidate_only"
                if not result.accepted_claims and result.held_claims:
                    session.admit_candidate_graph_for_exploration(len(result.held_claims), "Ablation candidate-only route; no core graph promotion.")
                row: dict[str, Any] = {"experiment_id": experiment_id, "normalization_mode": mode, "sample_id": sample_id, "case_id": case_id, "a2_metrics": result.operational_metrics.model_dump(mode="json"), "a2_claim_tier_for_a3": tier, "normalization_audit": {"types": normalized.type_normalizations, "predicates": normalized.predicate_normalizations}}
                if not claims:
                    row["status"] = "blocked_by_a2"
                elif args.dry_run:
                    row["status"] = "planned"
                else:
                    assert runtime is not None
                    allowed_ids = [item.source_id for item in normalized.proposal.evidence]
                    prediction = runtime.generate_json("a3", a3_prompt(query, normalized.proposal, claims, claim_tier=tier), allowed_ids, max_new_tokens=args.max_new_tokens)
                    draft = A3DiagnosisDraft(run_id=normalized.proposal.run_id, diagnosis_object=prediction, report_sections=[], evidence=normalized.proposal.evidence, model_profile="a3_diagnosis_lora")
                    orchestrator.submit_a3_draft(session, draft)
                    cited = set(evidence_ids(prediction))
                    row.update({"status": "complete", "a3_prediction": prediction, "a3_json_valid": True, "a3_citations_current_case_valid": cited.issubset(set(allowed_ids)), "workflow_state": session.state.value})
                rows.append(row)
    finally:
        if runtime is not None:
            runtime.unload()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["experiment_id"]].append(row)
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "scope": "paired frozen-A1 raw-output interface ablation; no raw A1 regeneration; candidate-only routes never promote to core graph", "sample_count": len(sources), "groups": {key: {"n": len(values), "completed": sum(row["status"] == "complete" for row in values), "mean_confirmed": sum(row["a2_metrics"]["accepted_claims"] for row in values) / len(values), "mean_candidate": sum(row["a2_metrics"]["held_claims"] for row in values) / len(values), "a3_json_valid_rate": sum(bool(row.get("a3_json_valid")) for row in values) / len(values), "a3_current_case_citation_valid_rate": sum(bool(row.get("a3_citations_current_case_valid")) for row in values) / len(values)} for key, values in grouped.items()}}
    (out / "per_sample.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(out), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
