#!/usr/bin/env python3
"""Evaluate A3 report generation without treating paraphrase as failure.

This CPU-only evaluator preserves exact statement overlap as a secondary
diagnostic, but reports primary contract/section coverage and evidence
grounding separately.  It is not a substitute for expert clinical/domain
judgement or factual correctness labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_kg.evaluation.role_metrics import assistant_json, evidence_ids


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluation_dir = args.evaluation_dir
    catalog = json.loads((ROOT / "data" / "manifests" / "fine_tuning_catalog_v0.3.json").read_text(encoding="utf-8"))
    gold_by_id: dict[str, dict[str, Any]] = {}
    for source, paths in catalog["a3"].items():
        if source in {"counts", "suggested_train_sampling_weight"}:
            continue
        for record in read_jsonl(Path(paths["test"])):
            gold_by_id[record["metadata"]["sample_id"]] = assistant_json(record)
    prediction_by_id = {row["sample_id"]: row["prediction"] for row in read_jsonl(evaluation_dir / "predictions.jsonl")}
    per_sample = read_jsonl(evaluation_dir / "per_sample.jsonl")
    records = []
    for row in per_sample:
        sample_id = row["sample_id"]
        gold = gold_by_id[sample_id]
        prediction = prediction_by_id.get(sample_id, {})
        expected_keys = sorted(gold)
        produced_keys = sorted(prediction)
        matched_keys = sorted(set(expected_keys) & set(produced_keys))
        records.append({
            "sample_id": sample_id,
            "json_valid": bool(row.get("json_valid")),
            "expected_top_level_keys": expected_keys,
            "produced_top_level_keys": produced_keys,
            "section_key_coverage": len(matched_keys) / len(expected_keys) if expected_keys else 1.0,
            "expected_evidence_ids": sorted(evidence_ids(gold)),
            "produced_evidence_ids": sorted(evidence_ids(prediction)),
            "evidence_metrics": row.get("evidence"),
        })
    valid = [item for item in records if item["json_valid"]]
    evidence_precision = [item["evidence_metrics"]["precision"] for item in valid if isinstance(item.get("evidence_metrics"), dict)]
    evidence_recall = [item["evidence_metrics"]["recall"] for item in valid if isinstance(item.get("evidence_metrics"), dict)]
    result = {
        "evaluation_scope": "A3 report contract and evidence grounding; paraphrase-tolerant structure metrics",
        "not_claimed": ["semantic factual correctness", "domain-expert report usefulness", "clinical or operational safety"],
        "sample_count": len(records),
        "json_valid_rate": sum(item["json_valid"] for item in records) / len(records) if records else 0.0,
        "mean_section_key_coverage": sum(item["section_key_coverage"] for item in records) / len(records) if records else 0.0,
        "mean_evidence_precision": sum(evidence_precision) / len(evidence_precision) if evidence_precision else None,
        "mean_evidence_recall": sum(evidence_recall) / len(evidence_recall) if evidence_recall else None,
        "records": records,
    }
    output = evaluation_dir / "a3_report_contract_evaluation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
