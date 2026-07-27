"""Gold-label-based evaluation for A2 decisions.

Operational A2 logs can measure schema and evidence coverage, but cannot
honestly measure correctness.  This module evaluates a frozen A2 audit against
human/A4-adjudicated labels and therefore produces paper-ready P/R/F1 values.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


DECISIONS = ("accept", "hold", "reject")


def _safe_divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _per_class_metrics(predicted: list[str], gold: list[str], label: str) -> dict[str, float | int | None]:
    true_positive = sum(pred == label and actual == label for pred, actual in zip(predicted, gold, strict=True))
    false_positive = sum(pred == label and actual != label for pred, actual in zip(predicted, gold, strict=True))
    false_negative = sum(pred != label and actual == label for pred, actual in zip(predicted, gold, strict=True))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    return {
        "support": sum(actual == label for actual in gold),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def evaluate_a2_decisions(
    audit_records: Iterable[dict[str, Any]], gold_rows: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Compare A2's frozen decisions with adjudicated, three-way gold labels.

    Each gold row requires ``claim_id`` and ``gold_decision``.  Valid labels
    are ``accept`` (correct and promotable), ``hold`` (ambiguous / needs more
    context), and ``reject`` (incorrect, unsupported, or structurally invalid).
    Blank labels are treated as pending annotation and excluded.
    """

    by_claim_id: dict[str, str] = {}
    duplicate_audit_ids: list[str] = []
    for record in audit_records:
        claim = record.get("claim", {})
        claim_id = claim.get("claim_id") or record.get("claim_id")
        decision = str(record.get("decision", "")).strip().lower()
        if not claim_id or decision not in DECISIONS:
            continue
        if claim_id in by_claim_id:
            duplicate_audit_ids.append(claim_id)
        by_claim_id[claim_id] = decision

    predicted: list[str] = []
    gold: list[str] = []
    missing_audit_ids: list[str] = []
    pending_gold_ids: list[str] = []
    duplicate_gold_ids: list[str] = []
    seen_gold_ids: set[str] = set()
    for row in gold_rows:
        claim_id = str(row.get("claim_id", "")).strip()
        label_raw = row.get("gold_decision")
        label = str(label_raw or "").strip().lower()
        if not claim_id:
            continue
        if claim_id in seen_gold_ids:
            duplicate_gold_ids.append(claim_id)
            continue
        seen_gold_ids.add(claim_id)
        if not label:
            pending_gold_ids.append(claim_id)
            continue
        if label not in DECISIONS:
            raise ValueError(f"{claim_id}: gold_decision must be one of {DECISIONS}, got {label!r}")
        if claim_id not in by_claim_id:
            missing_audit_ids.append(claim_id)
            continue
        predicted.append(by_claim_id[claim_id])
        gold.append(label)

    total = len(gold)
    per_class = {label: _per_class_metrics(predicted, gold, label) for label in DECISIONS}
    f1_values = [item["f1"] for item in per_class.values() if item["f1"] is not None]
    confusion_matrix = {
        actual: {prediction: sum(g == actual and p == prediction for p, g in zip(predicted, gold, strict=True)) for prediction in DECISIONS}
        for actual in DECISIONS
    }
    accept_precision = per_class["accept"]["precision"]
    gold_reject_count = sum(label == "reject" for label in gold)
    gold_accept_count = sum(label == "accept" for label in gold)
    unsafe_accepts = sum(prediction == "accept" and label == "reject" for prediction, label in zip(predicted, gold, strict=True))
    over_rejections = sum(prediction == "reject" and label == "accept" for prediction, label in zip(predicted, gold, strict=True))
    gold_hold_count = sum(label == "hold" for label in gold)
    review_captured = sum(
        prediction in {"hold", "reject"} and label == "hold"
        for prediction, label in zip(predicted, gold, strict=True)
    )

    return {
        "evaluation_protocol": "A2 three-way decision evaluation against frozen human/A4-adjudicated gold labels",
        "evaluated_claims": total,
        "coverage": {
            "gold_rows_seen": len(seen_gold_ids),
            "pending_gold_labels": len(pending_gold_ids),
            "missing_audit_records": len(missing_audit_ids),
            "duplicate_gold_ids": sorted(duplicate_gold_ids),
            "duplicate_audit_ids": sorted(set(duplicate_audit_ids)),
        },
        "decision_accuracy": _safe_divide(sum(p == g for p, g in zip(predicted, gold, strict=True)), total),
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "per_class": per_class,
        "safety_indicators": {
            "accepted_claim_error_rate": (1 - accept_precision) if accept_precision is not None else None,
            "unsafe_accept_rate_among_gold_rejects": _safe_divide(unsafe_accepts, gold_reject_count),
            "over_rejection_rate_among_gold_accepts": _safe_divide(over_rejections, gold_accept_count),
            "hold_capture_rate": _safe_divide(review_captured, gold_hold_count),
        },
        "confusion_matrix_gold_rows_predicted_columns": confusion_matrix,
        "predicted_distribution": dict(Counter(predicted)),
        "gold_distribution": dict(Counter(gold)),
    }
