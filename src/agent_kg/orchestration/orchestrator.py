"""Thin, explicit coordinator for A1 -> A2 -> A3 -> A4.

This class owns scheduling and audit-log persistence, never model weights,
baseline-KG writes, or final-report prose.  Model/API calls remain inside the
individual role adapters and are passed in as structured artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..agents.a2_quality_control import A2QualityController
from ..agents.a4_reviewer import A4Reviewer
from ..contracts import A1GraphProposal, A3DiagnosisDraft, A4ReviewResult
from .state_machine import WorkflowSession


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class AgentWorkflowOrchestrator:
    """Coordinate one bounded A1/A2/A3/A4 workflow run."""

    def __init__(self, a2_controller: A2QualityController | None = None) -> None:
        self.a2_controller = a2_controller or A2QualityController()

    def begin(self, proposal: A1GraphProposal) -> WorkflowSession:
        session = WorkflowSession(run_id=proposal.run_id)
        session.receive_a1_proposal(proposal)
        session.record_a2_result(self.a2_controller.inspect(proposal))
        return session

    @staticmethod
    def submit_a3_draft(session: WorkflowSession, draft: A3DiagnosisDraft) -> WorkflowSession:
        session.receive_a3_draft(draft)
        return session

    @staticmethod
    def submit_a4_review(session: WorkflowSession, review: A4ReviewResult) -> WorkflowSession:
        session.receive_a4_review(review)
        return session

    @staticmethod
    async def run_a4_api_review(session: WorkflowSession, reviewer: A4Reviewer) -> A4ReviewResult:
        """Run the configured A4 teacher only after A3 has submitted a draft."""

        if session.a3_draft is None:
            raise ValueError("A3 draft is required before an A4 API review.")
        review = await reviewer.review(session.a3_draft, session.a2_result)
        session.receive_a4_review(review)
        return review

    @staticmethod
    def resolve_a2_holds(session: WorkflowSession, accepted_claim_count: int, note: str) -> WorkflowSession:
        session.record_semantic_resolution(accepted_claim_count, note)
        return session

    @staticmethod
    def complete(session: WorkflowSession) -> WorkflowSession:
        session.complete()
        return session

    @staticmethod
    def write_audit_log(session: WorkflowSession, output_path: Path | str | None = None) -> Path:
        """Persist only the workflow audit; report rendering is a separate tool."""

        target = Path(output_path) if output_path else (
            PROJECT_ROOT / "experiments" / "logs" / f"workflow_{session.run_id}.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(session.audit_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target
