#!/usr/bin/env python3
"""Run pre-registered A2 contract mutations over public-case identifiers.

This is a deterministic conformance suite, not a semantic extraction benchmark.
It does not claim that fixture relations occurred in the source cases. Each case
identifier supplies an independent run namespace; expected outcomes follow the
published A2 policy only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_kg.agents.a2_quality_control import A2QualityController
from agent_kg.contracts import A1GraphProposal, EvidenceRef, GraphClaim


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "evaluation" / "public_regulatory_case_registry_v1.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_proposal(case_id: str, mutation: str) -> A1GraphProposal:
    evidence_id = f"ev-{case_id.lower()}"
    evidence = [EvidenceRef(source_type="input_text", source_id=evidence_id, excerpt="故障状态导致设备异常，记录仅用于契约测试。")]
    claim: dict[str, Any] = {
        "claim_id": f"{case_id}-{mutation}",
        "subject_id": "e-fault",
        "subject": "故障状态",
        "subject_type": "fault_mechanism_or_condition",
        "predicate": "导致",
        "object_id": "e-device",
        "object": "受影响设备",
        "object_type": "equipment_or_component",
        "confidence": 0.9,
        "evidence_ids": [evidence_id],
        "qualifiers": {"case_id": case_id, "fixture_only": True, "mutation": mutation},
    }
    if mutation == "M1_missing_evidence":
        claim["evidence_ids"] = ["ev-not-in-packet"]
    elif mutation == "M3_cross_case_evidence":
        claim["evidence_ids"] = [f"ev-other-{case_id.lower()}"]
    elif mutation == "M4_self_loop":
        claim.update({"object_id": "e-fault", "object": "故障状态", "object_type": "fault_mechanism_or_condition"})
    elif mutation == "M5_low_confidence_candidate":
        claim["confidence"] = 0.25
    elif mutation == "M6_fault_modifier_dropped":
        claim.update({"subject_id": "e-device", "subject": "受影响设备", "subject_type": "equipment_or_component"})
        evidence[0].excerpt = "受影响设备故障状态导致设备异常，记录仅用于契约测试。"
    return A1GraphProposal(
        run_id=f"a2-contract-v1-{case_id}-{mutation}",
        original_query="Deterministic A2 contract robustness fixture; not a source semantic claim.",
        model_profile="no_llm_fixture",
        entities=[
            {"id": "e-fault", "name": "故障状态", "type": "fault_mechanism_or_condition"},
            {"id": "e-device", "name": "受影响设备", "type": "equipment_or_component"},
        ],
        evidence=evidence,
        claims=[GraphClaim(**claim)],
    )


def decision(result) -> str:
    if result.accepted_claims:
        return "accept"
    if result.held_claims:
        return "hold"
    return "reject"


EXPECTED = {
    "BASE": "accept",
    "M1_missing_evidence": "reject",
    "M3_cross_case_evidence": "reject",
    # Current policy deliberately retains this malformed relation in the
    # candidate tier for review. The safety contract is therefore "not accept",
    # operationalized here as the stable hold decision.
    "M4_self_loop": "hold",
    "M5_low_confidence_candidate": "hold",
    "M6_fault_modifier_dropped": "hold",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--label", default="a2_contract_robustness_public18_v1")
    args = parser.parse_args()
    rows = read_jsonl(args.registry)
    output = ROOT / "experiments" / "robustness" / args.label
    output.mkdir(parents=True, exist_ok=False)
    controller = A2QualityController()
    records = []
    for source in rows:
        for mutation, expected in EXPECTED.items():
            result = controller.inspect(make_proposal(source["case_id"], mutation))
            audit = result.claim_audits[0]
            observed = decision(result)
            records.append({
                "case_id": source["case_id"],
                "domain": source["domain"],
                "mutation": mutation,
                "expected_policy_decision": expected,
                "observed_decision": observed,
                "pass": observed == expected,
                "rule_ids": audit.rule_ids,
                "evidence_status": audit.evidence_status,
                "fixture_scope": "deterministic_contract_only_not_semantic_case_claim",
            })
    (output / "per_fixture.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in records) + "\n", encoding="utf-8")
    by_mutation = {}
    for mutation in EXPECTED:
        subset = [x for x in records if x["mutation"] == mutation]
        by_mutation[mutation] = {
            "count": len(subset),
            "passed": sum(x["pass"] for x in subset),
            "pass_rate": sum(x["pass"] for x in subset) / len(subset),
            "observed": dict(Counter(x["observed_decision"] for x in subset)),
        }
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "fixture_type": "pre_registered deterministic A2 contract mutation suite; not human gold and not semantic extraction evaluation",
        "public_case_namespaces": len(rows),
        "fixture_count": len(records),
        "passed": sum(x["pass"] for x in records),
        "pass_rate": sum(x["pass"] for x in records) / len(records),
        "by_mutation": by_mutation,
        "policy_version": "A2QualityController current project policy",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
