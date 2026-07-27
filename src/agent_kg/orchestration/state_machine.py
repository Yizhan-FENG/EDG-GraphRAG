"""State machine for the A1 -> A2 -> A3 -> A4 diagnosis workflow.

The state machine owns transitions and audit history; agents own their
specialist work.  It deliberately has no permission to write the baseline KG
or a final Word document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from ..contracts import A1GraphProposal, A2QualityResult, A3DiagnosisDraft, A4ReviewResult


class WorkflowState(str, Enum):
    CREATED = "created"
    A1_PROPOSAL_RECEIVED = "a1_proposal_received"
    A2_AUDITED = "a2_audited"
    WAITING_SEMANTIC_REVIEW = "waiting_semantic_review"
    A3_DRAFT_RECEIVED = "a3_draft_received"
    A3_REVISION_REQUIRED = "a3_revision_required"
    A4_APPROVED = "a4_approved"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class InvalidTransition(RuntimeError):
    """Raised when a caller attempts to bypass a role boundary."""


@dataclass(frozen=True)
class WorkflowEvent:
    timestamp: str
    from_state: WorkflowState
    to_state: WorkflowState
    actor: str
    reason: str


@dataclass
class WorkflowSession:
    """Mutable workflow state for one diagnosis/report run.

    HOLD facts are excluded from high-confidence diagnostic evidence.  A
    separately audited candidate-graph exploration route may expose them to
    A3 with explicit provenance, but never promotes them to the core graph or
    marks the workflow complete.
    """

    run_id: str
    state: WorkflowState = WorkflowState.CREATED
    a1_proposal: A1GraphProposal | None = None
    a2_result: A2QualityResult | None = None
    a3_draft: A3DiagnosisDraft | None = None
    a4_review: A4ReviewResult | None = None
    events: list[WorkflowEvent] = field(default_factory=list)
    block_reason: str | None = None

    def receive_a1_proposal(self, proposal: A1GraphProposal) -> None:
        self._require({WorkflowState.CREATED}, "A1 can only start a new workflow.")
        self._require_run_id(proposal.run_id, "A1 proposal")
        self.a1_proposal = proposal
        self._transition(WorkflowState.A1_PROPOSAL_RECEIVED, "a1", "candidate graph proposal received")

    def record_a2_result(self, result: A2QualityResult) -> None:
        self._require({WorkflowState.A1_PROPOSAL_RECEIVED}, "A2 must audit an A1 proposal before later roles run.")
        self._require_run_id(result.run_id, "A2 result")
        self.a2_result = result
        if result.accepted_claims:
            self._transition(
                WorkflowState.A2_AUDITED,
                "a2",
                f"{len(result.accepted_claims)} accepted, {len(result.held_claims)} held, {len(result.rejected_claims)} rejected claims",
            )
            return
        if result.held_claims:
            self._transition(
                WorkflowState.WAITING_SEMANTIC_REVIEW,
                "a2",
                "no admissible claim; held candidates require API/A4/human semantic review",
            )
            return
        self.block_reason = "A2 rejected every candidate claim. A3 is not permitted to draft a diagnosis from it."
        self._transition(WorkflowState.BLOCKED, "a2", self.block_reason)

    def record_semantic_resolution(self, accepted_claim_count: int, note: str) -> None:
        """Record an externally adjudicated resolution of an all-HOLD batch.

        A4/API/human review happens outside A2; this method only records the
        result and makes the next A3 step legal.  The resolved facts themselves
        must be persisted in a separate versioned review log before production
        graph promotion.
        """

        self._require({WorkflowState.WAITING_SEMANTIC_REVIEW}, "No semantic review is pending.")
        if accepted_claim_count <= 0:
            self.block_reason = note or "Semantic review accepted no claims."
            self._transition(WorkflowState.BLOCKED, "semantic_reviewer", self.block_reason)
            return
        self._transition(
            WorkflowState.A2_AUDITED,
            "semantic_reviewer",
            f"semantic review admitted {accepted_claim_count} claim(s): {note}",
        )

    def admit_candidate_graph_for_exploration(self, candidate_claim_count: int, note: str) -> None:
        """Permit a non-promoting A3 exploration over an all-HOLD candidate graph.

        This is a policy route for retrieval/report exploration, not semantic
        adjudication.  The audit log explicitly distinguishes it from
        ``record_semantic_resolution`` and downstream A4 approval is still
        required for any final report.
        """

        self._require({WorkflowState.WAITING_SEMANTIC_REVIEW}, "Candidate exploration is only needed for an all-HOLD batch.")
        if candidate_claim_count <= 0:
            raise InvalidTransition("Candidate exploration requires at least one held claim.")
        self._transition(
            WorkflowState.A2_AUDITED,
            "orchestrator_candidate_route",
            f"candidate-only exploration admitted {candidate_claim_count} held claim(s); no core-graph promotion: {note}",
        )

    def receive_a3_draft(self, draft: A3DiagnosisDraft) -> None:
        self._require(
            {WorkflowState.A2_AUDITED, WorkflowState.A3_REVISION_REQUIRED},
            "A3 may only draft after A2 admission or an A4 revision request.",
        )
        self._require_run_id(draft.run_id, "A3 draft")
        self.a3_draft = draft
        self._transition(WorkflowState.A3_DRAFT_RECEIVED, "a3", "diagnosis draft received")

    def receive_a4_review(self, review: A4ReviewResult) -> None:
        self._require({WorkflowState.A3_DRAFT_RECEIVED}, "A4 can only review an A3 draft.")
        self._require_run_id(review.run_id, "A4 review")
        self.a4_review = review
        if review.decision == "approve":
            self._transition(WorkflowState.A4_APPROVED, "a4", "independent review approved the draft")
        elif review.decision == "revise":
            self._transition(WorkflowState.A3_REVISION_REQUIRED, "a4", "A4 requested revision")
        else:
            self.block_reason = "A4 rejected the diagnosis draft."
            self._transition(WorkflowState.BLOCKED, "a4", self.block_reason)

    def complete(self) -> None:
        self._require({WorkflowState.A4_APPROVED}, "Only an A4-approved draft can be completed.")
        self._transition(
            WorkflowState.COMPLETED,
            "orchestrator",
            "approved structured draft handed to the non-LLM report builder and audit logger",
        )

    def audit_payload(self) -> dict[str, object]:
        """Return JSON-serialisable workflow evidence for the audit logger."""

        return {
            "run_id": self.run_id,
            "state": self.state.value,
            "block_reason": self.block_reason,
            "events": [
                {
                    "timestamp": event.timestamp,
                    "from_state": event.from_state.value,
                    "to_state": event.to_state.value,
                    "actor": event.actor,
                    "reason": event.reason,
                }
                for event in self.events
            ],
            "a2_policy_version": self.a2_result.policy_version if self.a2_result else None,
            "a2_operational_metrics": (
                self.a2_result.operational_metrics.model_dump() if self.a2_result else None
            ),
            "a4_decision": self.a4_review.decision if self.a4_review else None,
        }

    def _transition(self, to_state: WorkflowState, actor: str, reason: str) -> None:
        event = WorkflowEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            from_state=self.state,
            to_state=to_state,
            actor=actor,
            reason=reason,
        )
        self.events.append(event)
        self.state = to_state

    def _require(self, allowed: set[WorkflowState], message: str) -> None:
        if self.state not in allowed:
            states = ", ".join(state.value for state in sorted(allowed, key=str))
            raise InvalidTransition(f"{message} Current state: {self.state.value}; allowed: {states}.")

    def _require_run_id(self, run_id: str, artifact_name: str) -> None:
        if run_id != self.run_id:
            raise ValueError(f"{artifact_name} run_id {run_id!r} does not match workflow run_id {self.run_id!r}.")
