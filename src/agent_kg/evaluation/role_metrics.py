"""Deterministic, evidence-centric metrics for held-out A1/A3 evaluation."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any


def assistant_json(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages", [])
    answer = next((item.get("content", "") for item in messages if item.get("role") == "assistant"), "")
    parsed = json.loads(answer)
    if not isinstance(parsed, dict):
        raise ValueError("Assistant target must be a JSON object.")
    return parsed


def evidence_ids(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = set(value.get("evidence_ids", [])) if isinstance(value.get("evidence_ids"), list) else set()
        for item in value.values():
            found |= evidence_ids(item)
        return {item for item in found if isinstance(item, str)}
    if isinstance(value, list):
        return set().union(*(evidence_ids(item) for item in value)) if value else set()
    return set()


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def a1_entity_set(payload: dict[str, Any]) -> set[str]:
    return {normalize(str(item.get("name", ""))) for item in payload.get("entities", []) if item.get("name")}


def a1_claim_set(payload: dict[str, Any]) -> set[str]:
    return {
        "|".join(normalize(str(item.get(key, ""))) for key in ("subject", "predicate", "object"))
        for item in payload.get("claims", [])
        if all(item.get(key) for key in ("subject", "predicate", "object"))
    }


def a3_statement_set(payload: dict[str, Any]) -> set[str]:
    statements: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in ("statement", "content", "evidence_summary"):
                if isinstance(value.get(key), str) and value[key].strip():
                    statements.add(normalize(value[key]))
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    return statements


def prf(predicted: set[str], gold: set[str]) -> dict[str, float | int]:
    correct = len(predicted & gold)
    precision = correct / len(predicted) if predicted else (1.0 if not gold else 0.0)
    recall = correct / len(gold) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"correct": correct, "predicted": len(predicted), "gold": len(gold), "precision": precision, "recall": recall, "f1": f1}


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    available: Counter[str] = Counter()
    for record in records:
        for metric in ("entity", "claim", "statement", "evidence"):
            values = record.get(metric)
            if isinstance(values, dict):
                available[metric] += 1
                totals[f"{metric}_correct"] += int(values.get("correct", 0))
                totals[f"{metric}_predicted"] += int(values.get("predicted", 0))
                totals[f"{metric}_gold"] += int(values.get("gold", 0))
    summary: dict[str, Any] = {
        "sample_count": len(records),
        "json_valid_rate": sum(bool(item.get("json_valid")) for item in records) / len(records) if records else 0.0,
        "structural_failure_count": sum(not bool(item.get("json_valid")) for item in records),
    }
    for metric in ("entity", "claim", "statement", "evidence"):
        if not available[metric]:
            summary[metric] = None
            continue
        predicted, gold, correct = totals[f"{metric}_predicted"], totals[f"{metric}_gold"], totals[f"{metric}_correct"]
        precision = correct / predicted if predicted else (1.0 if not gold else 0.0)
        recall = correct / gold if gold else 1.0
        summary[metric] = {"correct": correct, "predicted": predicted, "gold": gold, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return summary
