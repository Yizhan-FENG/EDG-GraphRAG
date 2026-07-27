#!/usr/bin/env python3
"""Evaluate an auditable lexical retrieval baseline over the candidate KG.

This is intentionally a CPU-only retrieval-integrity experiment, not a claim
of GraphRAG diagnosis quality.  The candidate extension contains case cards
from the external test sources, so every result records whether a retrieved
edge comes from the queried case (in-corpus support) or another case.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def tokens(text: str) -> list[str]:
    """Chinese character bigrams plus alphanumeric terms; deterministic/no model."""
    compact = re.sub(r"\s+", "", text.lower())
    cjk = re.findall(r"[\u4e00-\u9fff]+", compact)
    grams = [piece[i : i + 2] for piece in cjk for i in range(max(0, len(piece) - 1))]
    latin = re.findall(r"[a-z0-9]+", compact)
    return grams + latin


def message(record: dict[str, Any], role: str) -> str:
    for item in record.get("messages", []):
        if item.get("role") == role:
            return str(item.get("content", ""))
    raise ValueError(f"Missing {role} message")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def edge_text(edge: dict[str, Any]) -> str:
    return " ".join(str(edge.get(key, "")) for key in ("source_name", "predicate", "normalized_predicate", "target_name", "case_id"))


def score(query: list[str], document: list[str]) -> float:
    q, d = Counter(query), Counter(document)
    return float(sum(min(count, d[token]) for token, count in q.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--label", default="candidate_kg_lexical_integrity_v1")
    parser.add_argument("--exclude-query-case", action="store_true", help="Leave the query case out of the candidate index; prevents in-corpus case leakage.")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    candidate_root = ROOT / "kg_extensions" / "external_source_audited" / "v0.1"
    edges = read_jsonl(candidate_root / "candidate_edges.jsonl")
    docs = [(edge, tokens(edge_text(edge))) for edge in edges]
    test_path = ROOT / "data" / "a1" / "external_source_audited" / "test.jsonl"
    records = read_jsonl(test_path)
    rows: list[dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata", {})
        case_id = str(metadata.get("case_id", ""))
        eligible_docs = [(edge, doc_tokens) for edge, doc_tokens in docs if not args.exclude_query_case or str(edge.get("case_id")) != case_id]
        ranked = sorted(((score(tokens(message(record, "user")), doc_tokens), edge) for edge, doc_tokens in eligible_docs), key=lambda item: (-item[0], str(item[1].get("id"))))[: args.top_k]
        retrieved = [edge for value, edge in ranked if value > 0]
        same_case = [edge for edge in retrieved if str(edge.get("case_id")) == case_id]
        rows.append({
            "sample_id": metadata.get("sample_id"),
            "case_id": case_id,
            "query_case_in_candidate_kg": any(str(edge.get("case_id")) == case_id for edge in edges),
            "top_k": args.top_k,
            "retrieved_count": len(retrieved),
            "same_case_hit_count": len(same_case),
            "same_case_hit_at_k": bool(same_case),
            "cross_case_hit_count": len(retrieved) - len(same_case),
            "top_edges": [{"id": edge.get("id"), "case_id": edge.get("case_id"), "score": value, "evidence_ids": edge.get("evidence_ids", [])} for value, edge in ranked if value > 0],
        })
    n = len(rows)
    summary = {
        "experiment_id": args.label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "deterministic_character_bigram_overlap",
        "scope": "candidate_kg_retrieval_integrity_only_not_diagnosis_quality",
        "sample_count": n,
        "top_k": args.top_k,
        "exclude_query_case": args.exclude_query_case,
        "query_cases_present_in_candidate_kg": sum(row["query_case_in_candidate_kg"] for row in rows),
        "same_case_hit_at_k_rate": sum(row["same_case_hit_at_k"] for row in rows) / n if n else None,
        "mean_retrieved_count": sum(row["retrieved_count"] for row in rows) / n if n else None,
        "leakage_notice": "All queried case IDs are checked against the candidate KG. When exclude_query_case=true, same-case candidate edges are excluded; retrieval count remains a systems-availability statistic, not relevance or diagnostic accuracy.",
    }
    out = ROOT / "experiments" / "retrieval" / args.label
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_sample.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
