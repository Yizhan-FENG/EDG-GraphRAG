#!/usr/bin/env python3
"""Run a paired A3 GraphRAG context ablation on frozen full-report cases.

The retrieval arm appends cross-case relation cards *without their evidence
IDs*.  They are explicitly non-evidentiary priors: A3 may use them only to
organise questions or report structure, never to state a current-case fact.
Only the current case's source IDs are whitelisted by the role sandbox.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_kg.evaluation.role_metrics import assistant_json, evidence_ids
from agent_kg.runtime import SharedQwenRoleRuntime


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_LABEL = "candidate_kg_lexical_leave_one_case_out_v2_a2interface"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def message(record: dict[str, Any], role: str) -> str:
    return next(str(item["content"]) for item in record["messages"] if item["role"] == role)


def retrieval_context(case_id: str, rows: dict[str, dict[str, Any]], edges: dict[str, dict[str, Any]]) -> str:
    row = rows.get(case_id, {})
    cards: list[str] = []
    for item in row.get("top_edges", [])[:3]:
        edge = edges.get(str(item.get("id")), {})
        if not edge:
            continue
        cards.append(
            f"- prior case {edge.get('case_id')}: {edge.get('source_name')} | "
            f"{edge.get('normalized_predicate') or edge.get('predicate')} | {edge.get('target_name')}"
        )
    return (
        "\n\n[Cross-case retrieval context: non-evidentiary prior]\n"
        "The following cards are from different cases. They have no citation authority for the current case; "
        "do not cite them or turn them into current-case facts. You may only use them to organise analysis questions.\n"
        + ("\n".join(cards) if cards else "- no cross-case card retrieved")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="a3_graphrag_context_pilot_v1")
    parser.add_argument("--limit", type=int, default=2, help="Use 5 for the full frozen set after pilot verification.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    records = [row for row in read_jsonl(ROOT / "data" / "a3" / "external_source_audited" / "test.jsonl") if row["metadata"]["sample_id"].endswith("full_report")]
    records = records[: args.limit]
    retrieval_rows = {str(row["case_id"]): row for row in read_jsonl(ROOT / "experiments" / "retrieval" / RETRIEVAL_LABEL / "per_sample.jsonl")}
    candidate_edges = {str(row["id"]): row for row in read_jsonl(ROOT / "kg_extensions" / "external_source_audited" / "v0.1" / "candidate_edges.jsonl")}
    run_dir = ROOT / "experiments" / "graphrag_ablation" / args.label
    run_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    runtime = None if args.dry_run else SharedQwenRoleRuntime.from_project_config(ROOT / "config" / "agents.yaml")
    try:
        for record in records:
            sample_id = str(record["metadata"]["sample_id"])
            case_id = re.search(r"ext-(\d+)", sample_id).group(1)  # frozen A3 naming contract
            case_key = f"EXT-{case_id}"
            gold = assistant_json(record)
            allowed_ids = sorted(evidence_ids(gold))
            base_prompt = message(record, "user")
            arms = {"no_retrieval": base_prompt, "cross_case_retrieval": base_prompt + retrieval_context(case_key, retrieval_rows, candidate_edges)}
            for arm, prompt in arms.items():
                item: dict[str, Any] = {"sample_id": sample_id, "case_id": case_key, "arm": arm, "allowed_current_case_evidence_ids": allowed_ids}
                if args.dry_run:
                    item["status"] = "planned"
                else:
                    assert runtime is not None
                    prediction = runtime.generate_json("a3", prompt, allowed_ids, max_new_tokens=args.max_new_tokens)
                    produced = sorted(evidence_ids(prediction))
                    item.update({
                        "status": "complete",
                        "prediction": prediction,
                        "json_valid": True,
                        "section_key_coverage": len(set(gold) & set(prediction)) / len(gold) if gold else 1.0,
                        "produced_evidence_ids": produced,
                        "all_citations_current_case_valid": set(produced).issubset(set(allowed_ids)),
                    })
                results.append(item)
    finally:
        if runtime is not None:
            runtime.unload()
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "paired A3 structure/citation ablation; cross-case cards are non-evidentiary and cannot support current-case facts",
        "retrieval_experiment_id": RETRIEVAL_LABEL,
        "sample_count": len(records),
        "arms": ["no_retrieval", "cross_case_retrieval"],
        "completed_rows": sum(row.get("status") == "complete" for row in results),
        "json_valid_rate": sum(row.get("json_valid", False) for row in results) / len(results) if results else None,
        "current_case_citation_valid_rate": sum(row.get("all_citations_current_case_valid", False) for row in results) / len(results) if results else None,
    }
    (run_dir / "per_sample.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(run_dir), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
