from __future__ import annotations

from agent_kg.agents.a2_quality_control import A2QualityController
from agent_kg.contracts import A1GraphProposal, EvidenceRef, GraphClaim
from agent_kg.evaluation import evaluate_a2_decisions


def proposal(*claims: GraphClaim) -> A1GraphProposal:
    return A1GraphProposal(
        run_id="test-run",
        original_query="test",
        model_profile="test",
        entities=[
            {"id": "e-fault", "name": "裂纹", "type": "fault_mechanism_or_condition"},
            {"id": "e-device", "name": "叶片", "type": "equipment_or_component"},
            {"id": "e-outcome", "name": "跳闸", "type": "protection_or_outcome"},
        ],
        evidence=[EvidenceRef(source_type="input_text", source_id="ev-1", excerpt="裂纹导致叶片损伤")],
        claims=list(claims),
    )


def valid_claim(claim_id: str = "c1", **updates: object) -> GraphClaim:
    values: dict[str, object] = {
        "claim_id": claim_id,
        "subject_id": "e-fault",
        "subject": "裂纹",
        "predicate": "导致",
        "object_id": "e-device",
        "object": "叶片",
        "confidence": 0.9,
        "evidence_ids": ["ev-1"],
        "qualifiers": {"case_id": "case-1"},
    }
    values.update(updates)
    return GraphClaim(**values)


def test_a2_accepts_a_well_formed_grounded_candidate() -> None:
    result = A2QualityController().inspect(proposal(valid_claim()))

    assert [claim.claim_id for claim in result.accepted_claims] == ["c1"]
    assert not result.held_claims
    assert not result.rejected_claims
    assert result.claim_audits[0].normalized_predicate == "causes"
    assert result.claim_audits[0].diagnosis_eligible
    assert result.operational_metrics.verified_evidence_rate == 1.0


def test_a2_rejects_missing_or_unknown_evidence() -> None:
    result = A2QualityController().inspect(proposal(valid_claim(evidence_ids=["not-present"])))

    assert [claim.claim_id for claim in result.rejected_claims] == ["c1"]
    assert result.claim_audits[0].evidence_status == "missing"
    assert "R03_EVIDENCE_REFERENCE_EXISTS" in result.claim_audits[0].rule_ids


def test_a2_holds_ambiguous_but_not_fatally_invalid_claims() -> None:
    # Policy v0.4 retains more candidates: 0.15 <= confidence < 0.35 is
    # routed to the candidate/review tier instead of being rejected.
    result = A2QualityController().inspect(proposal(valid_claim(confidence=0.25)))

    assert [claim.claim_id for claim in result.held_claims] == ["c1"]
    assert result.needs_semantic_escalation
    assert "R05_CONFIDENCE_HOLD" in result.claim_audits[0].rule_ids


def test_a2_holds_causal_subject_that_drops_explicit_fault_modifier() -> None:
    item = proposal(valid_claim(subject="叶片", subject_type="equipment_or_component"))
    item.evidence[0].excerpt = "叶片裂纹导致叶片损伤"
    result = A2QualityController().inspect(item)

    assert [claim.claim_id for claim in result.held_claims] == ["c1"]
    assert "R15_EVIDENCE_FAULT_MODIFIER_DROPPED" in result.claim_audits[0].rule_ids


def test_a2_distinguishes_cross_case_corroboration_from_same_case_duplicates() -> None:
    same_case = A2QualityController().inspect(proposal(valid_claim("c1"), valid_claim("c2")))
    cross_case = A2QualityController().inspect(
        proposal(valid_claim("c1"), valid_claim("c2", qualifiers={"case_id": "case-2"}))
    )

    assert {claim.claim_id for claim in same_case.held_claims} == {"c1", "c2"}
    assert {claim.claim_id for claim in cross_case.accepted_claims} == {"c1", "c2"}


def test_offline_a2_evaluation_requires_and_uses_adjudicated_labels() -> None:
    audit = [
        {"claim": {"claim_id": "c1"}, "decision": "accept"},
        {"claim": {"claim_id": "c2"}, "decision": "hold"},
        {"claim": {"claim_id": "c3"}, "decision": "reject"},
    ]
    gold = [
        {"claim_id": "c1", "gold_decision": "accept"},
        {"claim_id": "c2", "gold_decision": "hold"},
        {"claim_id": "c3", "gold_decision": "reject"},
        {"claim_id": "pending", "gold_decision": ""},
    ]

    report = evaluate_a2_decisions(audit, gold)

    assert report["evaluated_claims"] == 3
    assert report["coverage"]["pending_gold_labels"] == 1
    assert report["decision_accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["safety_indicators"]["unsafe_accept_rate_among_gold_rejects"] == 0.0
