"""Frozen tier-aware lexical retrieval ablation over the relaxed candidate view.

This measures only retrieval availability and the fraction of retrieved
candidate-tier relations.  It has no human relevance labels and must not be
reported as diagnostic correctness or retrieval relevance quality.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "evaluation" / "rel_evid_60_pending_independent_review.jsonl"
ROUTED = ROOT / "experiments" / "candidate_routing" / "relaxed_candidate_v1_model_proxy" / "routed_claims.jsonl"
OUT = ROOT / "experiments" / "retrieval" / "tier_aware_relaxed_candidate_v1_model_proxy"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", text.lower())
    terms = re.findall(r"[\u4e00-\u9fff]+", compact)
    return [term[i : i + 2] for term in terms for i in range(max(0, len(term) - 1))] + re.findall(r"[a-z0-9]+", compact)


def score(query: list[str], doc: list[str]) -> float:
    query_counts, doc_counts = Counter(query), Counter(doc)
    return float(sum(min(count, doc_counts[token]) for token, count in query_counts.items()))


def main() -> None:
    source = read_jsonl(SOURCE)
    routed = read_jsonl(ROUTED)
    source_by_review = {f"REL-{index:03d}": row for index, row in enumerate(source, start=1)}
    edges = []
    for row in routed:
        original = source_by_review[row["source_review_code"]]
        claim = row["claim"]
        edges.append({
            "route_id": row["route_id"],
            "tier": row["tier"],
            "case_id": original["parent_sample_id"],
            "text": " ".join(str(claim.get(key, "")) for key in ("subject", "predicate", "object")),
        })
    query_contexts: dict[str, str] = {}
    for row in source:
        query_contexts.setdefault(row["parent_sample_id"], row["source_context"])

    results: dict[str, list[dict]] = {"confirmed_only": [], "confirmed_plus_candidate": []}
    for mode in results:
        pool = [edge for edge in edges if mode == "confirmed_plus_candidate" or edge["tier"] == "confirmed_model_proxy"]
        for query_case, context in sorted(query_contexts.items()):
            eligible = [edge for edge in pool if edge["case_id"] != query_case]
            ranked = sorted(
                ((score(tokens(context), tokens(edge["text"])), edge) for edge in eligible),
                key=lambda item: (-item[0], item[1]["route_id"]),
            )[:10]
            retrieved = [(value, edge) for value, edge in ranked if value > 0]
            results[mode].append({
                "query_case_id": query_case,
                "top_k": 10,
                "same_case_excluded": True,
                "indexed_edge_count": len(pool),
                "eligible_edge_count": len(eligible),
                "retrieved_count": len(retrieved),
                "candidate_tier_retrieved_count": sum(edge["tier"] == "candidate_exploration" for _, edge in retrieved),
                "top_edges": [{"route_id": edge["route_id"], "tier": edge["tier"], "score": value} for value, edge in retrieved],
            })

    OUT.mkdir(parents=True, exist_ok=True)
    for mode, rows in results.items():
        (OUT / f"{mode}_per_query.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    def summarise(rows: list[dict]) -> dict:
        total = sum(row["retrieved_count"] for row in rows)
        candidates = sum(row["candidate_tier_retrieved_count"] for row in rows)
        return {
            "query_count": len(rows),
            "mean_retrieved_count": sum(row["retrieved_count"] for row in rows) / len(rows),
            "nonempty_query_rate": sum(row["retrieved_count"] > 0 for row in rows) / len(rows),
            "candidate_tier_share_of_retrieved": candidates / total if total else None,
        }
    summary = {
        "experiment_id": "tier_aware_relaxed_candidate_v1_model_proxy",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "deterministic_character_bigram_overlap_leave_source_case_out",
        "comparison": {mode: summarise(rows) for mode, rows in results.items()},
        "invariants": {"queries": len(query_contexts), "top_k": 10, "same_source_case_excluded": True, "a3_fact_boundary": "candidate_exploration relations remain unavailable as diagnosis facts"},
        "scope_limit": "Availability and tier-composition only. No independent relevance labels, no diagnostic accuracy, and no semantic quality claim.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
