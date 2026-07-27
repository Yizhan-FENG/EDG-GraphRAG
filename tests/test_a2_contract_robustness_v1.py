from scripts.run_a2_contract_robustness_v1 import EXPECTED, decision, make_proposal
from agent_kg.agents.a2_quality_control import A2QualityController
from agent_kg.evaluation.scope_guard import enforce_evidence_scope_guard


def test_registered_mutations_match_current_a2_policy() -> None:
    controller = A2QualityController()
    for mutation, expected in EXPECTED.items():
        observed = decision(controller.inspect(make_proposal("PUB-TEST-01", mutation)))
        assert observed == expected, (mutation, observed, expected)


def test_scope_guard_demotes_causal_claim_without_deleting_it() -> None:
    proposal = make_proposal("PUB-TEST-02", "BASE")
    guarded, ids = enforce_evidence_scope_guard(proposal, root_cause_allowed=False)
    assert ids == [proposal.claims[0].claim_id]
    assert guarded.claims[0].confidence == 0.25
    assert decision(A2QualityController().inspect(guarded)) == "hold"
