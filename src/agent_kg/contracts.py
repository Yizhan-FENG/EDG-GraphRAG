"""A1-A4 间只能传递显式的数据契约，避免自由文本隐式耦合。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class EvidenceRef(BaseModel):
    source_type: Literal["kg_entity", "kg_relation", "rag_chunk", "input_text", "rule"]
    source_id: str
    excerpt: str = ""
    score: float | None = None


class GraphClaim(BaseModel):
    """A candidate fact proposed by A1.

    ``claim_id`` is deliberately stable across A1 -> A2 -> A3/A4.  It lets a
    human reviewer, an A4 review log, and the offline evaluation set refer to
    exactly the same candidate without relying on a potentially duplicated
    subject-predicate-object string.
    """

    claim_id: str | None = None
    subject_id: str | None = None
    subject: str
    predicate: str
    predicate_normalized: str | None = None
    object_id: str | None = None
    object: str
    subject_type: str | None = None
    object_type: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    qualifiers: dict[str, Any] = Field(default_factory=dict)


class A1GraphProposal(BaseModel):
    run_id: str
    original_query: str
    entities: list[dict] = Field(default_factory=list)
    claims: list[GraphClaim] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    model_profile: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RuleFinding(BaseModel):
    rule_id: str
    severity: Literal["info", "warning", "error"]
    target: str
    message: str


class ClaimAuditRecord(BaseModel):
    """The explainable, per-claim decision emitted by A2.

    ``decision`` is intentionally three-valued.  A structurally valid but
    semantically ambiguous claim must not be silently accepted or discarded:
    it is routed to A4/API/human review as ``hold``.
    """

    claim: GraphClaim
    claim_index: int
    decision: Literal["accept", "hold", "reject"]
    normalized_predicate: str | None = None
    rule_ids: list[str] = Field(default_factory=list)
    evidence_status: Literal["verified", "unverifiable", "missing"] = "unverifiable"
    hard_schema_valid: bool = True
    diagnosis_eligible: bool = False
    needs_semantic_escalation: bool = False


class A2OperationalMetrics(BaseModel):
    """Run-level, label-free quality indicators.

    These values describe what A2 observed in one run.  They are *not* a
    substitute for precision/recall/F1, which require human-adjudicated gold
    labels and are computed by the offline evaluator.
    """

    total_claims: int = 0
    accepted_claims: int = 0
    held_claims: int = 0
    rejected_claims: int = 0
    acceptance_rate: float = 0.0
    hold_rate: float = 0.0
    rejection_rate: float = 0.0
    hard_schema_conformance_rate: float = 0.0
    evidence_id_coverage_rate: float = 0.0
    verified_evidence_rate: float = 0.0
    predicate_normalization_rate: float = 0.0
    type_completeness_rate: float = 0.0
    duplicate_or_conflict_rate: float = 0.0
    diagnosis_eligible_rate: float = 0.0
    mean_candidate_confidence: float | None = None
    finding_count_by_rule: dict[str, int] = Field(default_factory=dict)


class A2QualityResult(BaseModel):
    run_id: str
    accepted_claims: list[GraphClaim] = Field(default_factory=list)
    held_claims: list[GraphClaim] = Field(default_factory=list)
    rejected_claims: list[GraphClaim] = Field(default_factory=list)
    findings: list[RuleFinding] = Field(default_factory=list)
    claim_audits: list[ClaimAuditRecord] = Field(default_factory=list)
    operational_metrics: A2OperationalMetrics = Field(default_factory=A2OperationalMetrics)
    policy_version: str = "unversioned"
    needs_semantic_escalation: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DiagnosisSection(BaseModel):
    title: str
    content: str
    evidence_ids: list[str] = Field(default_factory=list)


class A3DiagnosisDraft(BaseModel):
    run_id: str
    diagnosis_object: dict
    report_sections: list[DiagnosisSection]
    evidence: list[EvidenceRef]
    model_profile: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ReviewIssue(BaseModel):
    category: Literal["unsupported_claim", "contradiction", "missing_evidence", "safety", "report_quality"]
    severity: Literal["minor", "major", "critical"]
    message: str
    evidence_ids: list[str] = Field(default_factory=list)


class A4AuditMetadata(BaseModel):
    """Trace metadata for an A4 API review; it must never contain an API key."""

    provider: str
    model: str
    prompt_version: str
    response_id: str | None = None
    request_fingerprint: str
    latency_ms: int | None = None
    audit_log_path: str | None = None


class A4ReviewResult(BaseModel):
    run_id: str
    decision: Literal["approve", "revise", "reject"]
    issues: list[ReviewIssue] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)
    teacher_model_profile: str
    audit_metadata: A4AuditMetadata | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
