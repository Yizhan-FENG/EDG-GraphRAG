"""A2: deterministic, explainable quality control for candidate KG facts.

This module deliberately does not use an LLM as its default decision maker.
It separates fatal data defects (REJECT) from claims that are structurally
valid but need domain judgement (HOLD).  Only HOLD records may be escalated to
an API/A4/human reviewer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any
from unicodedata import normalize as unicode_normalize

import yaml

from ..contracts import (
    A1GraphProposal,
    A2OperationalMetrics,
    A2QualityResult,
    ClaimAuditRecord,
    GraphClaim,
    RuleFinding,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "a2_quality_policy.yaml"


def _normalise_text(value: str | None) -> str:
    """Return a stable comparison key without changing the displayed label."""

    return " ".join(unicode_normalize("NFKC", value or "").strip().casefold().split())


class A2QualityController:
    """Rule/tool quality gate for A1 graph candidates.

    The controller reports an operational metric vector for every run.  It
    intentionally does *not* claim P/R/F1 without a human-adjudicated gold
    set; use :mod:`agent_kg.evaluation.a2_metrics` for that offline analysis.
    """

    role_id = "a2"

    def __init__(self, policy_path: Path | str | None = None) -> None:
        path = Path(policy_path) if policy_path else DEFAULT_POLICY_PATH
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"A2 policy must be a YAML mapping: {path}")
        self.policy = payload
        self.policy_path = path
        self.policy_version = str(payload.get("version", "unversioned"))
        self.entity_types = set(payload.get("entity_types", []))
        thresholds = payload.get("thresholds", {})
        self.reject_confidence = float(thresholds.get("reject_confidence", 0.35))
        self.hold_confidence = float(thresholds.get("hold_confidence", 0.50))
        if not 0 <= self.reject_confidence <= self.hold_confidence <= 1:
            raise ValueError("A2 confidence thresholds must satisfy 0 <= reject <= hold <= 1")

        aliases: dict[str, str] = {}
        for canonical, values in payload.get("predicate_aliases", {}).items():
            aliases[_normalise_text(canonical)] = canonical
            for value in values or []:
                aliases[_normalise_text(str(value))] = canonical
        self.predicate_aliases = aliases
        self.relation_constraints: dict[str, dict[str, Any]] = payload.get("relation_constraints", {})
        self.cycle_sensitive_predicates = set(payload.get("cycle_sensitive_predicates", []))
        fidelity = payload.get("evidence_fidelity", {})
        self.fault_cues = tuple(str(item) for item in fidelity.get("fault_cues", []) if str(item).strip())

    def inspect(self, proposal: A1GraphProposal) -> A2QualityResult:
        """Audit every claim and return ACCEPT/HOLD/REJECT with reasons.

        No claim is written to the baseline graph here.  ``accepted_claims``
        are only eligible for a later transactional promotion after the
        project-level A4/human approval rule is also satisfied.
        """

        evidence_ids_available = {reference.source_id for reference in proposal.evidence}
        evidence_excerpt_by_id = {reference.source_id: reference.excerpt for reference in proposal.evidence}
        entity_types_by_id, entity_types_by_name = self._index_entity_types(proposal.entities)
        findings: list[RuleFinding] = []
        audits: list[ClaimAuditRecord] = []

        for index, raw_claim in enumerate(proposal.claims):
            audit = self._initial_audit(
                raw_claim,
                index=index,
                evidence_ids_available=evidence_ids_available,
                evidence_excerpt_by_id=evidence_excerpt_by_id,
                entity_types_by_id=entity_types_by_id,
                entity_types_by_name=entity_types_by_name,
                findings=findings,
            )
            audits.append(audit)

        # Cross-claim checks can only be performed after every individual
        # candidate has been normalised and structurally inspected.
        self._apply_entity_type_conflicts(audits, entity_types_by_name, findings)
        self._apply_duplicate_checks(audits, findings)
        self._apply_causal_cycle_checks(audits, findings)
        self._apply_polarity_conflict_checks(audits, findings)

        for audit in audits:
            constraint = self.relation_constraints.get(audit.normalized_predicate or "", {})
            audit.diagnosis_eligible = (
                audit.decision == "accept" and bool(constraint.get("diagnostic_relation", False))
            )
            audit.needs_semantic_escalation = audit.decision == "hold"

        accepted = [audit.claim for audit in audits if audit.decision == "accept"]
        held = [audit.claim for audit in audits if audit.decision == "hold"]
        rejected = [audit.claim for audit in audits if audit.decision == "reject"]
        return A2QualityResult(
            run_id=proposal.run_id,
            accepted_claims=accepted,
            held_claims=held,
            rejected_claims=rejected,
            findings=findings,
            claim_audits=audits,
            operational_metrics=self._operational_metrics(audits),
            policy_version=self.policy_version,
            needs_semantic_escalation=bool(held),
        )

    def _initial_audit(
        self,
        raw_claim: GraphClaim,
        *,
        index: int,
        evidence_ids_available: set[str],
        evidence_excerpt_by_id: dict[str, str],
        entity_types_by_id: dict[str, set[str]],
        entity_types_by_name: dict[str, set[str]],
        findings: list[RuleFinding],
    ) -> ClaimAuditRecord:
        subject_type = self._resolve_type(
            explicit=raw_claim.subject_type,
            entity_id=raw_claim.subject_id,
            name=raw_claim.subject,
            by_id=entity_types_by_id,
            by_name=entity_types_by_name,
        )
        object_type = self._resolve_type(
            explicit=raw_claim.object_type,
            entity_id=raw_claim.object_id,
            name=raw_claim.object,
            by_id=entity_types_by_id,
            by_name=entity_types_by_name,
        )
        normalized_predicate = self._canonical_predicate(raw_claim)
        claim = raw_claim.model_copy(
            update={
                "subject_type": subject_type,
                "object_type": object_type,
                "predicate_normalized": normalized_predicate,
            }
        )
        audit = ClaimAuditRecord(
            claim=claim,
            claim_index=index,
            decision="accept",
            normalized_predicate=normalized_predicate,
            evidence_status="verified" if evidence_ids_available else "unverifiable",
        )

        if not all(_normalise_text(value) for value in (claim.subject, claim.predicate, claim.object)):
            self._add_finding(
                audit, findings, "R01_REQUIRED_FIELDS", "error",
                "subject, predicate, and object must all be non-empty.",
                hard_schema_invalid=True,
            )

        if not claim.evidence_ids:
            audit.evidence_status = "missing"
            self._add_finding(
                audit, findings, "R02_EVIDENCE_REQUIRED", "error",
                "The claim has no evidence_ids and cannot enter the candidate fact graph.",
                hard_schema_invalid=True,
            )
        elif evidence_ids_available:
            unknown_ids = sorted(set(claim.evidence_ids) - evidence_ids_available)
            if unknown_ids:
                audit.evidence_status = "missing"
                self._add_finding(
                    audit, findings, "R03_EVIDENCE_REFERENCE_EXISTS", "error",
                    f"Evidence IDs are absent from the supplied evidence bundle: {unknown_ids}.",
                    hard_schema_invalid=True,
                )
            else:
                audit.evidence_status = "verified"
        else:
            audit.evidence_status = "unverifiable"
            self._add_finding(
                audit, findings, "R03_EVIDENCE_BUNDLE_UNAVAILABLE", "warning",
                "Evidence IDs are present but no evidence bundle was supplied; semantic grounding must be reviewed.",
            )

        if claim.confidence < self.reject_confidence:
            self._add_finding(
                audit, findings, "R04_CONFIDENCE_REJECT", "error",
                f"Candidate confidence {claim.confidence:.3f} is below the reject threshold {self.reject_confidence:.2f}.",
            )
        elif claim.confidence < self.hold_confidence:
            self._add_finding(
                audit, findings, "R05_CONFIDENCE_HOLD", "warning",
                f"Candidate confidence {claim.confidence:.3f} requires semantic review below {self.hold_confidence:.2f}.",
            )

        if not normalized_predicate:
            self._add_finding(
                audit, findings, "R06_UNKNOWN_PREDICATE", "warning",
                "Predicate is not in the controlled vocabulary; map it before graph promotion.",
            )
        else:
            self._check_relation_types(audit, findings)

        for endpoint, endpoint_type in (("subject", subject_type), ("object", object_type)):
            if not endpoint_type:
                self._add_finding(
                    audit, findings, "R07_TYPE_INCOMPLETE", "warning",
                    f"{endpoint} type is unavailable; entity resolution/type classification is required.",
                )
            elif endpoint_type not in self.entity_types:
                self._add_finding(
                    audit, findings, "R08_UNKNOWN_ENTITY_TYPE", "warning",
                    f"{endpoint} type '{endpoint_type}' is outside the controlled ontology.",
                )

        if _normalise_text(claim.subject) == _normalise_text(claim.object):
            self._add_finding(
                audit, findings, "R09_SELF_LOOP", "warning",
                "Self-loop candidate requires manual validation; it is not promoted automatically.",
            )
        self._check_evidence_modifier_loss(audit, evidence_excerpt_by_id, findings)
        return audit

    def _check_evidence_modifier_loss(
        self, audit: ClaimAuditRecord, evidence_excerpt_by_id: dict[str, str], findings: list[RuleFinding]
    ) -> None:
        """Hold a causal claim that drops an explicit fault modifier from evidence.

        This narrow deterministic guard catches e.g. ``superheater leakage``
        being weakened to ``superheater causes trip``.  It is a HOLD rather
        than a rejection because an expert may still resolve the intended
        entity boundary from a richer source bundle.
        """
        if audit.normalized_predicate not in {"causes", "triggers", "damages"}:
            return
        subject = _normalise_text(audit.claim.subject)
        if not subject or any(cue in subject for cue in self.fault_cues):
            return
        excerpts = " ".join(evidence_excerpt_by_id.get(item, "") for item in audit.claim.evidence_ids)
        if any(f"{audit.claim.subject}{cue}" in excerpts for cue in self.fault_cues):
            self._add_finding(
                audit, findings, "R15_EVIDENCE_FAULT_MODIFIER_DROPPED", "warning",
                "The causal subject drops an explicit fault modifier present in its cited evidence; hold for semantic review.",
            )

    def _check_relation_types(self, audit: ClaimAuditRecord, findings: list[RuleFinding]) -> None:
        constraint = self.relation_constraints.get(audit.normalized_predicate or "")
        if not constraint:
            return
        subject_allowed = set(constraint.get("subject_types", []))
        object_allowed = set(constraint.get("object_types", []))
        if audit.claim.subject_type and "*" not in subject_allowed and audit.claim.subject_type not in subject_allowed:
            self._add_finding(
                audit, findings, "R10_DOMAIN_RANGE_MISMATCH", "warning",
                f"Predicate '{audit.normalized_predicate}' does not normally accept subject type '{audit.claim.subject_type}'.",
            )
        if audit.claim.object_type and "*" not in object_allowed and audit.claim.object_type not in object_allowed:
            self._add_finding(
                audit, findings, "R10_DOMAIN_RANGE_MISMATCH", "warning",
                f"Predicate '{audit.normalized_predicate}' does not normally accept object type '{audit.claim.object_type}'.",
            )

    @staticmethod
    def _index_entity_types(entities: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        by_id: dict[str, set[str]] = defaultdict(set)
        by_name: dict[str, set[str]] = defaultdict(set)
        for entity in entities:
            entity_type = entity.get("type") or entity.get("entity_type")
            if not isinstance(entity_type, str) or not _normalise_text(entity_type):
                continue
            entity_id = entity.get("id") or entity.get("entity_id")
            name = entity.get("name") or entity.get("entity")
            if isinstance(entity_id, str) and _normalise_text(entity_id):
                by_id[entity_id].add(entity_type)
            if isinstance(name, str) and _normalise_text(name):
                by_name[_normalise_text(name)].add(entity_type)
        return dict(by_id), dict(by_name)

    @staticmethod
    def _resolve_type(
        *,
        explicit: str | None,
        entity_id: str | None,
        name: str,
        by_id: dict[str, set[str]],
        by_name: dict[str, set[str]],
    ) -> str | None:
        if explicit:
            return explicit
        if entity_id and len(by_id.get(entity_id, set())) == 1:
            return next(iter(by_id[entity_id]))
        matches = by_name.get(_normalise_text(name), set())
        return next(iter(matches)) if len(matches) == 1 else None

    def _canonical_predicate(self, claim: GraphClaim) -> str | None:
        for candidate in (claim.predicate_normalized, claim.predicate):
            normalised = _normalise_text(candidate)
            if normalised in self.predicate_aliases:
                return self.predicate_aliases[normalised]
        return None

    def _apply_entity_type_conflicts(
        self,
        audits: list[ClaimAuditRecord],
        types_by_name: dict[str, set[str]],
        findings: list[RuleFinding],
    ) -> None:
        ambiguous_names = {name for name, types in types_by_name.items() if len(types) > 1}
        if not ambiguous_names:
            return
        for audit in audits:
            if (
                _normalise_text(audit.claim.subject) in ambiguous_names
                or _normalise_text(audit.claim.object) in ambiguous_names
            ):
                self._add_finding(
                    audit, findings, "R11_ENTITY_TYPE_CONFLICT", "warning",
                    "The same canonical entity label has conflicting types in this proposal; resolve it before promotion.",
                )

    @staticmethod
    def _assertion_scope(audit: ClaimAuditRecord) -> str:
        """Return the source/case scope that separates corroboration from duplication.

        The same causal triple reported by two independent incidents is useful
        corroboration, not a duplicate to be held.  A1 should attach one of
        these qualifiers whenever a source document/case is available.
        """

        qualifiers = audit.claim.qualifiers
        for key in ("case_id", "document_id", "source_id", "time_window"):
            value = qualifiers.get(key)
            if value is not None and _normalise_text(str(value)):
                return _normalise_text(str(value))
        return "proposal_global"

    def _apply_duplicate_checks(self, audits: list[ClaimAuditRecord], findings: list[RuleFinding]) -> None:
        groups: dict[tuple[str, str, str, str], list[ClaimAuditRecord]] = defaultdict(list)
        for audit in audits:
            if audit.normalized_predicate:
                groups[
                    (
                        self._assertion_scope(audit),
                        _normalise_text(audit.claim.subject),
                        audit.normalized_predicate,
                        _normalise_text(audit.claim.object),
                    )
                ].append(audit)
        for group in groups.values():
            if len(group) > 1:
                for audit in group:
                    self._add_finding(
                        audit, findings, "R12_DUPLICATE_TRIPLE", "warning",
                        "Duplicate canonical triple detected in the same source/case scope; merge evidence or retain one canonical assertion.",
                    )

    def _apply_causal_cycle_checks(self, audits: list[ClaimAuditRecord], findings: list[RuleFinding]) -> None:
        edges: dict[tuple[str, str, str], list[ClaimAuditRecord]] = defaultdict(list)
        for audit in audits:
            if audit.normalized_predicate in self.cycle_sensitive_predicates:
                edges[
                    (audit.normalized_predicate, _normalise_text(audit.claim.subject), _normalise_text(audit.claim.object))
                ].append(audit)
        visited: set[tuple[str, str, str]] = set()
        for key, group in edges.items():
            predicate, subject, object_ = key
            reverse_key = (predicate, object_, subject)
            if reverse_key in edges and key not in visited and reverse_key not in visited:
                for audit in [*group, *edges[reverse_key]]:
                    self._add_finding(
                        audit, findings, "R13_CAUSAL_CYCLE", "warning",
                        "Bidirectional causal relation detected; retain as HOLD until temporal/causal review confirms a feedback loop.",
                    )
                visited.add(key)
                visited.add(reverse_key)

    def _apply_polarity_conflict_checks(self, audits: list[ClaimAuditRecord], findings: list[RuleFinding]) -> None:
        groups: dict[tuple[str, str, str, str], list[ClaimAuditRecord]] = defaultdict(list)
        for audit in audits:
            if audit.normalized_predicate:
                groups[
                    (
                        self._assertion_scope(audit),
                        _normalise_text(audit.claim.subject),
                        audit.normalized_predicate,
                        _normalise_text(audit.claim.object),
                    )
                ].append(audit)
        positive = {"positive", "affirmed", "true", "肯定", "存在"}
        negative = {"negative", "negated", "false", "否定", "不存在"}
        for group in groups.values():
            polarities = {_normalise_text(str(item.claim.qualifiers.get("polarity", ""))) for item in group}
            if polarities & positive and polarities & negative:
                for audit in group:
                    self._add_finding(
                        audit, findings, "R14_POLARITY_CONFLICT", "warning",
                        "Positive and negative assertions share the same canonical triple; check time/context before promotion.",
                    )

    def _add_finding(
        self,
        audit: ClaimAuditRecord,
        findings: list[RuleFinding],
        rule_id: str,
        severity: str,
        message: str,
        *,
        hard_schema_invalid: bool = False,
    ) -> None:
        if rule_id not in audit.rule_ids:
            audit.rule_ids.append(rule_id)
        target = audit.claim.claim_id or f"claim-{audit.claim_index}"
        findings.append(RuleFinding(rule_id=rule_id, severity=severity, target=target, message=message))
        if hard_schema_invalid:
            audit.hard_schema_valid = False
        if severity == "error":
            audit.decision = "reject"
        elif severity == "warning" and audit.decision != "reject":
            audit.decision = "hold"

    @staticmethod
    def _operational_metrics(audits: list[ClaimAuditRecord]) -> A2OperationalMetrics:
        total = len(audits)
        accepted = sum(audit.decision == "accept" for audit in audits)
        held = sum(audit.decision == "hold" for audit in audits)
        rejected = sum(audit.decision == "reject" for audit in audits)
        with_evidence = sum(bool(audit.claim.evidence_ids) for audit in audits)
        verified_evidence = sum(audit.evidence_status == "verified" for audit in audits)
        normalised_predicates = sum(audit.normalized_predicate is not None for audit in audits)
        typed = sum(bool(audit.claim.subject_type and audit.claim.object_type) for audit in audits)
        conflicts = sum(any(rule.startswith(("R11_", "R12_", "R13_", "R14_")) for rule in audit.rule_ids) for audit in audits)
        diagnosis_eligible = sum(audit.diagnosis_eligible for audit in audits)
        confidence_values = [audit.claim.confidence for audit in audits]
        denominator = total or 1
        return A2OperationalMetrics(
            total_claims=total,
            accepted_claims=accepted,
            held_claims=held,
            rejected_claims=rejected,
            acceptance_rate=accepted / denominator,
            hold_rate=held / denominator,
            rejection_rate=rejected / denominator,
            hard_schema_conformance_rate=sum(audit.hard_schema_valid for audit in audits) / denominator,
            evidence_id_coverage_rate=with_evidence / denominator,
            verified_evidence_rate=verified_evidence / denominator,
            predicate_normalization_rate=normalised_predicates / denominator,
            type_completeness_rate=typed / denominator,
            duplicate_or_conflict_rate=conflicts / denominator,
            diagnosis_eligible_rate=diagnosis_eligible / denominator,
            mean_candidate_confidence=fmean(confidence_values) if confidence_values else None,
            finding_count_by_rule=dict(
                Counter(rule_id for audit in audits for rule_id in audit.rule_ids)
            ),
        )
