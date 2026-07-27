"""Deterministic evidence-authority guards used before A2 admission."""

from __future__ import annotations

from typing import Any

from ..contracts import A1GraphProposal


CAUSAL_PREDICATES = {"causes", "cause", "triggers", "damages", "导致", "瀵艰嚧"}


def enforce_evidence_scope_guard(
    proposal: A1GraphProposal,
    root_cause_allowed: Any,
) -> tuple[A1GraphProposal, list[str]]:
    """Demote causal candidates when the source has no root-cause authority.

    The guard preserves candidates for later review instead of deleting them.
    It is a provenance rule, not a semantic-correctness label.
    """

    if root_cause_allowed is True:
        return proposal, []
    guarded_ids: list[str] = []
    guarded_claims = []
    for claim in proposal.claims:
        predicate = str(claim.predicate_normalized or claim.predicate).strip().lower()
        if predicate in CAUSAL_PREDICATES:
            qualifiers = dict(claim.qualifiers or {})
            qualifiers["evidence_scope_guard"] = "causal_claim_demoted_no_root_cause_authority"
            guarded_claims.append(
                claim.model_copy(
                    update={
                        "confidence": min(claim.confidence, 0.25),
                        "qualifiers": qualifiers,
                    }
                )
            )
            guarded_ids.append(claim.claim_id)
        else:
            guarded_claims.append(claim)
    return proposal.model_copy(update={"claims": guarded_claims}), guarded_ids
