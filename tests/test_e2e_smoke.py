from __future__ import annotations

from agent_kg.contracts import EvidenceRef
from agent_kg.orchestration.e2e_smoke import a3_prompt, draft_from_a3_payload, normalize_a1_payload, record_a1_to_a2
from agent_kg.orchestration.orchestrator import AgentWorkflowOrchestrator


def test_e2e_normalizer_records_type_aliases_and_allows_a2_transition() -> None:
    raw = {
        "entities": [
            {"name": "锅炉", "type": "equipment", "evidence_ids": ["ev-1"]},
            {"name": "机组跳闸", "type": "fault_consequence", "evidence_ids": ["ev-1"]},
        ],
        "claims": [
            {
                "subject": "锅炉",
                "predicate": "导致",
                "object": "机组跳闸",
                "subject_type": "equipment",
                "object_type": "fault_consequence",
                "confidence": 1.0,
                "evidence_ids": ["ev-1"],
            }
        ],
        "evidence": [{"source_type": "input_text", "source_id": "ev-1", "excerpt": "锅炉运行异常导致跳闸"}],
    }
    normalized = normalize_a1_payload(raw, run_id="e2e-test", query="锅炉故障", model_profile="a1-test")
    assert normalized.proposal.claims[0].subject_type == "equipment_or_component"
    assert normalized.proposal.claims[0].object_type == "protection_or_outcome"
    assert len(normalized.type_normalizations) == 4
    assert normalized.evidence_normalizations == []
    session = record_a1_to_a2(AgentWorkflowOrchestrator(), normalized)
    assert len(session.a2_result.accepted_claims) == 1  # type: ignore[union-attr]


def test_a3_draft_carries_nested_evidence_ids_to_report_section() -> None:
    draft = draft_from_a3_payload(
        "e2e-test",
        {"observations": [{"statement": "跳闸", "evidence_ids": ["ev-1"]}]},
        [EvidenceRef(source_type="input_text", source_id="ev-1")],
        "a3-test",
    )
    assert draft.report_sections[0].evidence_ids == ["ev-1"]


def test_e2e_normalizer_records_legacy_evidence_transport_aliases() -> None:
    raw = {
        "evidence": [
            {"source_file": "processed_data/legacy.txt", "evidence_id": "ev-legacy", "excerpt": "legacy text"}
        ]
    }
    normalized = normalize_a1_payload(
        raw,
        run_id="e2e-legacy",
        query="q",
        model_profile="a1-test",
        allowed_evidence_ids=["ev-legacy"],
    )
    assert normalized.proposal.evidence[0].source_type == "input_text"
    assert normalized.proposal.evidence[0].source_id == "ev-legacy"
    assert {item["field"] for item in normalized.evidence_normalizations} == {"source_type", "source_id"}


def test_e2e_normalizer_contextually_separates_component_from_degradation_state() -> None:
    raw = {
        "entities": [
            {"name": "管子母材", "type": "component"},
            {"name": "材质老化", "type": "component"},
            {"name": "硬度降低", "type": "component"},
        ],
        "claims": [
            {"subject": "管子母材", "predicate": "导致", "object": "材质老化", "subject_type": "component", "object_type": "component", "confidence": 1.0, "evidence_ids": ["ev-1"]},
            {"subject": "管子母材", "predicate": "导致", "object": "硬度降低", "subject_type": "component", "object_type": "component", "confidence": 1.0, "evidence_ids": ["ev-1"]},
        ],
        "evidence": [{"source_type": "input_text", "source_id": "ev-1", "excerpt": "母材存在材质老化和硬度降低。"}],
    }
    normalized = normalize_a1_payload(raw, run_id="e2e-context", query="q", model_profile="a1-test")
    assert [item["type"] for item in normalized.proposal.entities] == [
        "equipment_or_component", "fault_mechanism_or_condition", "fault_mechanism_or_condition"
    ]
    assert normalized.proposal.claims[0].object_type == "fault_mechanism_or_condition"
    assert normalized.proposal.claims[1].object_type == "fault_mechanism_or_condition"
    assert any(item.get("rule") == "degradation_or_failure_state" for item in normalized.type_normalizations)


def test_e2e_normalizer_maps_local_phenomenon_and_process_surface_labels() -> None:
    raw = {
        "entities": [
            {"name": "材质老化", "type": "phenomenon"},
            {"name": "逐步更换", "type": "process"},
        ]
    }
    normalized = normalize_a1_payload(raw, run_id="e2e-labels", query="q", model_profile="a1-test")
    assert [item["type"] for item in normalized.proposal.entities] == [
        "fault_mechanism_or_condition", "corrective_action"
    ]


def test_e2e_normalizer_maps_explicit_condition_containment_only_with_evidence() -> None:
    raw = {
        "claims": [{
            "subject": "管子母材", "predicate": "包含", "normalized_predicate": "contains", "object": "材质老化",
            "subject_type": "component", "object_type": "phenomenon", "confidence": 1.0, "evidence_ids": ["ev-1"],
        }],
        "evidence": [{"source_type": "input_text", "source_id": "ev-1", "excerpt": "管子母材存在材质老化。"}],
    }
    normalized = normalize_a1_payload(raw, run_id="e2e-predicate", query="q", model_profile="a1-test")
    assert normalized.proposal.claims[0].predicate_normalized == "has_defect"
    assert normalized.predicate_normalizations[0]["rule"] == "explicit_condition_containment"


def test_e2e_normalizer_keeps_evidenced_parameter_state_distinct_from_defect() -> None:
    raw = {
        "claims": [{
            "subject": "管子母材", "predicate": "包含", "object": "硬度降低",
            "subject_type": "component", "object_type": "phenomenon", "confidence": 1.0, "evidence_ids": ["ev-1"],
        }],
        "evidence": [{"source_type": "input_text", "source_id": "ev-1", "excerpt": "管子母材存在硬度降低。"}],
    }
    normalized = normalize_a1_payload(raw, run_id="e2e-parameter", query="q", model_profile="a1-test")
    assert normalized.proposal.claims[0].object_type == "operating_parameter"
    assert normalized.proposal.claims[0].predicate_normalized == "has_operating_parameter_state"


def test_a3_prompt_separates_candidate_exploration_from_confirmed_claims() -> None:
    raw = {
        "claims": [
            {"subject": "boiler", "predicate": "causes", "object": "trip", "confidence": 1.0, "evidence_ids": ["ev-1"]},
            {"subject": "candidate_cause", "predicate": "causes", "object": "trip", "confidence": 0.4, "evidence_ids": ["ev-1"]},
        ],
        "evidence": [{"source_type": "input_text", "source_id": "ev-1", "excerpt": "boiler issue causes trip"}],
    }
    normalized = normalize_a1_payload(raw, run_id="e2e-tier", query="q", model_profile="a1-test")
    prompt = a3_prompt("q", normalized.proposal, [normalized.proposal.claims[0]], candidate_claims=[normalized.proposal.claims[1]])
    assert '"confirmed_claims"' in prompt
    assert '"candidate_exploration_claims"' in prompt
    assert "must not be cited as evidence" in prompt
