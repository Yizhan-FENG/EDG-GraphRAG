"""Auditable A1 -> A2 -> A3 glue for a bounded held-out smoke run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..contracts import A1GraphProposal, A3DiagnosisDraft, DiagnosisSection, EvidenceRef, GraphClaim
from ..evaluation.role_metrics import evidence_ids as collect_evidence_ids
from .orchestrator import AgentWorkflowOrchestrator
from .state_machine import WorkflowSession


ENTITY_TYPE_ALIASES = {
    "equipment": "equipment_or_component",
    "component": "equipment_or_component",
    "fault": "fault_mechanism_or_condition",
    "phenomenon": "fault_mechanism_or_condition",
    # Observed local A1 adapter surface labels.  These aliases are a
    # transparent ontology-normalisation layer, not a semantic rewrite: the
    # original label remains in the raw audit and every conversion is logged.
    "fault_concept": "fault_mechanism_or_condition",
    "fault_consequence": "protection_or_outcome",
    "outcome": "protection_or_outcome",
    "protection": "protection_or_outcome",
    "protection_or_overhaul": "protection_or_outcome",
    "action": "corrective_action",
    "process": "corrective_action",
}

# These are deliberately small, lexical *context overrides* for the local
# electric-power ontology.  They correct an observed adapter interface issue:
# the A1 LoRA often emits the training-surface label ``component`` for both a
# physical part and a degradation state.  The raw A1 label is never replaced
# silently; each override is appended to ``type_normalizations`` with its rule.
# This is not an evidence or relation rewrite.
CONTEXTUAL_ENTITY_TYPE_RULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "degradation_or_failure_state",
        ("老化", "劣化", "降低", "裂纹", "开裂", "泄漏", "磨损", "失效", "故障", "腐蚀", "变形"),
        "fault_mechanism_or_condition",
    ),
    (
        "operating_parameter_state",
        ("温度", "压力", "流量", "振动", "电压", "电流", "硬度", "厚度"),
        "operating_parameter",
    ),
)


@dataclass(frozen=True)
class NormalizedA1Payload:
    proposal: A1GraphProposal
    type_normalizations: list[dict[str, str]]
    evidence_normalizations: list[dict[str, str]]
    predicate_normalizations: list[dict[str, str]]


def _canonical_type(
    value: Any,
    changes: list[dict[str, str]],
    *,
    field: str,
    entity_name: Any = None,
    apply_aliases: bool = True,
    apply_context: bool = True,
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    canonical = ENTITY_TYPE_ALIASES.get(value.strip(), value.strip()) if apply_aliases else value.strip()
    if canonical != value:
        changes.append({"field": field, "original": value, "canonical": canonical})
    if apply_context and isinstance(entity_name, str):
        for rule, cues, contextual in CONTEXTUAL_ENTITY_TYPE_RULES:
            if any(cue in entity_name for cue in cues) and canonical != contextual:
                changes.append(
                    {
                        "field": field,
                        "entity": entity_name,
                        "original": canonical,
                        "canonical": contextual,
                        "rule": rule,
                    }
                )
                canonical = contextual
                break
    return canonical


def normalize_a1_payload(
    raw: dict[str, Any],
    *,
    run_id: str,
    query: str,
    model_profile: str,
    allowed_evidence_ids: list[str] | None = None,
    normalization_mode: str = "full",
) -> NormalizedA1Payload:
    """Convert sandbox JSON into the A1 contract with explicit, logged aliases.

    ``processed_data`` and ``evidence_id`` occur in the legacy silver-data
    schema.  They are transport-field aliases, not a claim/evidence rewrite:
    the original field value is kept in the normalization audit.  A missing
    source ID is never guessed unless the input supplies exactly one allowed
    evidence ID, in which case the inference is explicitly logged.
    """

    if normalization_mode not in {"raw", "alias_only", "full"}:
        raise ValueError("normalization_mode must be raw, alias_only, or full")
    apply_aliases = normalization_mode != "raw"
    apply_context = normalization_mode == "full"
    changes: list[dict[str, str]] = []
    entities: list[dict[str, Any]] = []
    for entity in raw.get("entities", []):
        if not isinstance(entity, dict):
            continue
        item = dict(entity)
        item["type"] = _canonical_type(
            item.get("type"), changes, field="entity.type", entity_name=item.get("name")
            , apply_aliases=apply_aliases, apply_context=apply_context
        )
        entities.append(item)
    evidence_normalizations: list[dict[str, str]] = []
    evidence: list[EvidenceRef] = []
    for index, raw_item in enumerate(raw.get("evidence", []), start=1):
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        source_type = item.get("source_type")
        if source_type == "processed_data":
            item["source_type"] = "input_text"
            evidence_normalizations.append(
                {"index": str(index), "field": "source_type", "original": "processed_data", "canonical": "input_text"}
            )
        if not item.get("source_type") and isinstance(item.get("source_file"), str) and item["source_file"].startswith("processed_data/"):
            item["source_type"] = "input_text"
            evidence_normalizations.append(
                {"index": str(index), "field": "source_type", "original": "missing_processed_data_path", "canonical": "input_text"}
            )
        if not item.get("source_id") and item.get("evidence_id"):
            item["source_id"] = str(item["evidence_id"])
            evidence_normalizations.append(
                {"index": str(index), "field": "source_id", "original": "evidence_id", "canonical": str(item["source_id"])}
            )
        if not item.get("source_id") and allowed_evidence_ids and len(allowed_evidence_ids) == 1:
            item["source_id"] = allowed_evidence_ids[0]
            evidence_normalizations.append(
                {"index": str(index), "field": "source_id", "original": "missing_singleton_context", "canonical": allowed_evidence_ids[0]}
            )
        evidence.append(EvidenceRef.model_validate(item))
    evidence_excerpt_by_id = {item.source_id: item.excerpt or "" for item in evidence}
    predicate_normalizations: list[dict[str, str]] = []
    claims: list[GraphClaim] = []
    for index, claim in enumerate(raw.get("claims", []), start=1):
        if not isinstance(claim, dict):
            continue
        item = dict(claim)
        item.setdefault("claim_id", f"{run_id}-claim-{index:03d}")
        item["subject_type"] = _canonical_type(
            item.get("subject_type"), changes, field="claim.subject_type", entity_name=item.get("subject")
            , apply_aliases=apply_aliases, apply_context=apply_context
        )
        item["object_type"] = _canonical_type(
            item.get("object_type"), changes, field="claim.object_type", entity_name=item.get("object")
            , apply_aliases=apply_aliases, apply_context=apply_context
        )
        # A1's local data sometimes uses the generic surface predicate
        # "包含" for an explicitly evidenced defect condition.  Canonicalise
        # only this narrow pattern; generic containment is deliberately not
        # added as an A2 global alias.
        raw_predicate = str(item.get("predicate_normalized") or item.get("normalized_predicate") or item.get("predicate") or "")
        object_name = str(item.get("object") or "")
        excerpts = " ".join(evidence_excerpt_by_id.get(str(evidence_id), "") for evidence_id in item.get("evidence_ids", []))
        stated_condition = object_name and object_name in excerpts and any(cue in excerpts for cue in ("存在", "有", "出现"))
        if normalization_mode == "full" and (
            raw_predicate in {"contains", "包含"}
            and item.get("subject_type") == "equipment_or_component"
            and item.get("object_type") == "fault_mechanism_or_condition"
            and stated_condition
        ):
            item["predicate_normalized"] = "has_defect"
            predicate_normalizations.append(
                {
                    "claim_id": str(item["claim_id"]),
                    "original": raw_predicate,
                    "canonical": "has_defect",
                    "rule": "explicit_condition_containment",
                }
            )
        elif normalization_mode == "full" and (
            raw_predicate in {"contains", "包含"}
            and item.get("subject_type") == "equipment_or_component"
            and item.get("object_type") == "operating_parameter"
            and stated_condition
        ):
            item["predicate_normalized"] = "has_operating_parameter_state"
            predicate_normalizations.append(
                {
                    "claim_id": str(item["claim_id"]),
                    "original": raw_predicate,
                    "canonical": "has_operating_parameter_state",
                    "rule": "explicit_parameter_state_containment",
                }
            )
        elif "normalized_predicate" in item and "predicate_normalized" not in item:
            # Transport-field compatibility only; A2 still decides whether
            # that predicate is in its controlled vocabulary.
            item["predicate_normalized"] = item["normalized_predicate"]
        claims.append(GraphClaim.model_validate(item))
    proposal = A1GraphProposal(
        run_id=run_id,
        original_query=query,
        entities=entities,
        claims=claims,
        evidence=evidence,
        model_profile=model_profile,
    )
    return NormalizedA1Payload(
        proposal=proposal,
        type_normalizations=changes,
        evidence_normalizations=evidence_normalizations,
        predicate_normalizations=predicate_normalizations,
    )


def a3_prompt(
    query: str,
    proposal: A1GraphProposal,
    accepted_claims: list[GraphClaim],
    *,
    claim_tier: str = "confirmed",
    candidate_claims: list[GraphClaim] | None = None,
) -> str:
    graph = {
        "entities": proposal.entities,
        f"{claim_tier}_claims": [claim.model_dump(mode="json") for claim in accepted_claims],
        "candidate_exploration_claims": [claim.model_dump(mode="json") for claim in (candidate_claims or [])],
        "evidence": [item.model_dump(mode="json") for item in proposal.evidence],
    }
    return (
        "任务：根据 A2 放行的局部图和来源证据生成结构化诊断草案。"
        "只可使用给定 evidence_ids；若根因或措施证据不足，必须标记为待核验。\n"
        "confirmed_claims are the only claims that may be presented as current-case facts or causal conclusions. "
        "candidate_exploration_claims may only guide open verification questions or retrieval directions; they must not be cited as evidence or converted into current-case facts.\n"
        f"诊断任务：{query}\nA2 放行局部图与证据：\n{json.dumps(graph, ensure_ascii=False)}"
    )


def draft_from_a3_payload(run_id: str, payload: dict[str, Any], evidence: list[EvidenceRef], model_profile: str) -> A3DiagnosisDraft:
    cited = sorted(collect_evidence_ids(payload))
    return A3DiagnosisDraft(
        run_id=run_id,
        diagnosis_object=payload,
        report_sections=[DiagnosisSection(title="A3 结构化诊断草案", content=json.dumps(payload, ensure_ascii=False), evidence_ids=cited)],
        evidence=evidence,
        model_profile=model_profile,
    )


def record_a1_to_a2(orchestrator: AgentWorkflowOrchestrator, normalized: NormalizedA1Payload) -> WorkflowSession:
    """Pure, testable A1 -> A2 state transition; model invocation stays outside."""

    return orchestrator.begin(normalized.proposal)
