from __future__ import annotations

import json

import pytest

from agent_kg.contracts import (
    A1GraphProposal,
    A3DiagnosisDraft,
    A4ReviewResult,
    EvidenceRef,
    GraphClaim,
)
from agent_kg.orchestration import AgentWorkflowOrchestrator, InvalidTransition, WorkflowState


RUN_ID = "workflow-test-001"


def a1_proposal() -> A1GraphProposal:
    return A1GraphProposal(
        run_id=RUN_ID,
        original_query="叶片裂纹故障诊断",
        model_profile="test",
        entities=[
            {"id": "fault", "name": "裂纹", "type": "fault_mechanism_or_condition"},
            {"id": "device", "name": "叶片", "type": "equipment_or_component"},
        ],
        evidence=[EvidenceRef(source_type="input_text", source_id="ev-1", excerpt="裂纹导致叶片损坏")],
        claims=[
            GraphClaim(
                claim_id="claim-1",
                subject_id="fault",
                subject="裂纹",
                predicate="导致",
                object_id="device",
                object="叶片",
                confidence=0.9,
                evidence_ids=["ev-1"],
                qualifiers={"case_id": "case-1"},
            )
        ],
    )


def a3_draft() -> A3DiagnosisDraft:
    return A3DiagnosisDraft(
        run_id=RUN_ID,
        diagnosis_object={"suspected_fault": "叶片裂纹"},
        report_sections=[],
        evidence=[EvidenceRef(source_type="input_text", source_id="ev-1", excerpt="裂纹导致叶片损坏")],
        model_profile="a3-test",
    )


def a4_review(decision: str) -> A4ReviewResult:
    return A4ReviewResult(run_id=RUN_ID, decision=decision, teacher_model_profile="a4-test")


def test_happy_path_requires_a2_before_a3_and_a4_approval_before_completion(tmp_path) -> None:
    orchestrator = AgentWorkflowOrchestrator()
    session = orchestrator.begin(a1_proposal())

    assert session.state is WorkflowState.A2_AUDITED
    assert len(session.a2_result.accepted_claims) == 1  # type: ignore[union-attr]
    orchestrator.submit_a3_draft(session, a3_draft())
    orchestrator.submit_a4_review(session, a4_review("approve"))
    orchestrator.complete(session)

    assert session.state is WorkflowState.COMPLETED
    path = orchestrator.write_audit_log(session, tmp_path / "workflow.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["a4_decision"] == "approve"
    assert payload["state"] == "completed"


def test_a4_revision_returns_only_to_a3() -> None:
    orchestrator = AgentWorkflowOrchestrator()
    session = orchestrator.begin(a1_proposal())
    orchestrator.submit_a3_draft(session, a3_draft())
    orchestrator.submit_a4_review(session, a4_review("revise"))

    assert session.state is WorkflowState.A3_REVISION_REQUIRED
    with pytest.raises(InvalidTransition):
        orchestrator.complete(session)
    orchestrator.submit_a3_draft(session, a3_draft())
    assert session.state is WorkflowState.A3_DRAFT_RECEIVED


def test_a3_cannot_bypass_a2() -> None:
    orchestrator = AgentWorkflowOrchestrator()
    from agent_kg.orchestration.state_machine import WorkflowSession

    session = WorkflowSession(run_id=RUN_ID)
    with pytest.raises(InvalidTransition):
        orchestrator.submit_a3_draft(session, a3_draft())


def test_a2_all_rejected_blocks_a3() -> None:
    proposal = a1_proposal().model_copy(
        update={
            "claims": [
                GraphClaim(
                    claim_id="bad-1",
                    subject="裂纹",
                    predicate="导致",
                    object="叶片",
                    confidence=0.1,
                    evidence_ids=[],
                )
            ]
        }
    )
    session = AgentWorkflowOrchestrator().begin(proposal)

    assert session.state is WorkflowState.BLOCKED
    with pytest.raises(InvalidTransition):
        AgentWorkflowOrchestrator.submit_a3_draft(session, a3_draft())
