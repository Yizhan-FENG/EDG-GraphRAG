"""Run the existing A2/A3-bound A4 packets through a local Ollama reviewer.

This runner is deliberately separate from the authorized remote-API runner:
it creates a new experiment directory, sends no request outside localhost, and
never overwrites remote-A4 attempts.  Local LLM reviews are model-proxy audits,
not independent human annotations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_kg.agents.a4_reviewer import A4ApiReviewError, A4Reviewer


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "experiments" / "a4_review_queue" / "bound_offline_queue_v1" / "pending_requests.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


async def main(limit: int, label: str, sample_ids: set[str] | None, max_output_tokens: int) -> None:
    out = ROOT / "experiments" / "a4_reviews" / label
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.jsonl"
    reviewer = A4Reviewer.from_project_config(ROOT / "config" / "agents.yaml")
    if reviewer.profile.provider != "ollama":
        raise RuntimeError("This local runner requires the A4 profile to use provider='ollama'.")
    # The A4 contract is deliberately compact.  A bounded local completion
    # avoids one malformed item monopolising the single GPU for many minutes.
    reviewer.max_output_tokens = max_output_tokens
    reviewer.audit_dir = out / "audits"
    rows = read_jsonl(result_path) if result_path.exists() else []
    completed_ids = {row["queue_id"] for row in rows if row.get("status") == "complete"}
    for item in read_jsonl(QUEUE):
        if sample_ids is not None and item["sample_id"] not in sample_ids:
            continue
        if item["queue_id"] in completed_ids:
            continue
        if sum(row.get("status") == "complete" for row in rows) >= limit:
            break
        request = item["request"]
        if request.get("store") is not False or not request.get("text", {}).get("format", {}).get("strict"):
            raise RuntimeError(f"Queue contract is invalid: {item['queue_id']}")
        started = time.perf_counter()
        raw: dict[str, Any] | None = None
        try:
            raw = await reviewer._local_ollama_review(request)
            parsed = reviewer.parse_response(raw)
            latency_ms = round((time.perf_counter() - started) * 1000)
            audit_path = reviewer._write_audit(
                run_id=item["run_id"], request=request, raw_response=raw,
                parsed_response=parsed, request_fingerprint=item["request_fingerprint"], latency_ms=latency_ms,
            )
            rows.append({
                "queue_id": item["queue_id"], "sample_id": item["sample_id"], "run_id": item["run_id"],
                "status": "complete", "decision": parsed["decision"], "issues": parsed["issues"],
                "revision_instructions": parsed["revision_instructions"], "latency_ms": latency_ms,
                "model": raw.get("model", reviewer.profile.model), "usage": raw.get("usage"),
                "audit_log_path": str(audit_path), "request_fingerprint": item["request_fingerprint"],
                "storage_scope": "local_ollama_only", "review_label": "local_llm_model_proxy",
            })
        except A4ApiReviewError as exc:
            if raw is not None:
                diagnostic_dir = out / "parse_failures"
                diagnostic_dir.mkdir(parents=True, exist_ok=True)
                diagnostic = {
                    "queue_id": item["queue_id"], "sample_id": item["sample_id"], "run_id": item["run_id"],
                    "error": str(exc), "model": raw.get("model", reviewer.profile.model),
                    "output_text": raw.get("output_text"), "request_fingerprint": item["request_fingerprint"],
                }
                (diagnostic_dir / f"{item['run_id']}.json").write_text(
                    json.dumps(reviewer._redact(diagnostic), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            rows.append({
                "queue_id": item["queue_id"], "sample_id": item["sample_id"], "run_id": item["run_id"],
                "status": "failed", "error": str(exc), "latency_ms": round((time.perf_counter() - started) * 1000),
                "storage_scope": "local_ollama_only", "review_label": "local_llm_model_proxy",
            })
        write_jsonl(result_path, rows)
    completed = [row for row in rows if row["status"] == "complete"]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "queue_source": str(QUEUE), "requested": limit, "completed": len(completed),
        "failed": len(rows) - len(completed), "decision_counts": dict(Counter(row["decision"] for row in completed)),
        "mean_latency_ms": sum(row["latency_ms"] for row in completed) / len(completed) if completed else None,
        "model_profile": reviewer.profile_name, "model": reviewer.profile.model,
        "review_label": "local_llm_model_proxy_not_human_gold", "external_api_called": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2, help="Sequential packet count for this run.")
    parser.add_argument("--label", default="glm4_9b_local_pilot_v1")
    parser.add_argument("--sample-id", action="append", dest="sample_ids", help="Only run this sample ID; repeatable.")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.max_output_tokens < 64:
        raise SystemExit("--max-output-tokens must be at least 64")
    asyncio.run(main(args.limit, args.label, set(args.sample_ids) if args.sample_ids else None, args.max_output_tokens))
