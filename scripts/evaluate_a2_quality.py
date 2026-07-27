#!/usr/bin/env python3
"""Evaluate a frozen A2 audit against human/A4-adjudicated gold labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_kg.evaluation import evaluate_a2_decisions


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True, help="A2 audit JSON with claim_audits")
    parser.add_argument("--gold", type=Path, required=True, help="Reviewed JSONL with claim_id and gold_decision")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "reports" / "a2_quality_evaluation.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    report = evaluate_a2_decisions(audit.get("claim_audits", []), read_jsonl(args.gold))
    report.update({"audit_path": str(args.audit), "gold_path": str(args.gold)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
