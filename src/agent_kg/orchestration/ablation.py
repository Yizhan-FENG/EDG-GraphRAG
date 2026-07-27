"""Small, auditable helpers for sequential ablation execution.

The GPU runner deliberately writes one result per held-out case.  This makes a
power interruption, a malformed model response, or an A2 block observable and
resumable instead of silently changing the evaluated sample population.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_run_id(experiment_id: str, record: dict[str, Any]) -> str:
    """Create a filesystem-safe deterministic run ID from the dataset sample."""

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    sample_id = metadata.get("sample_id")
    if isinstance(sample_id, str) and sample_id.strip():
        safe = "".join(character if character.isalnum() or character in "-_" else "-" for character in sample_id)
        return f"{experiment_id}--{safe}"
    digest = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{experiment_id}--unlabeled-{digest}"


def is_terminal_result(payload: dict[str, Any], experiment_id: str, run_id: str) -> bool:
    """Only skip an existing file when it is a complete result for this exact case."""

    return (
        payload.get("experiment_id") == experiment_id
        and payload.get("run_id") == run_id
        and payload.get("status") in {"complete", "blocked_by_a2", "structural_failure", "blocked_by_a4"}
    )


def load_terminal_result(path: Path, experiment_id: str, run_id: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) and is_terminal_result(payload, experiment_id, run_id) else None


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    latencies: list[float] = []
    for result in results:
        status = str(result.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
        latency = result.get("latency_seconds")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
    terminal_count = sum(counts.get(status, 0) for status in ("complete", "blocked_by_a2", "structural_failure", "blocked_by_a4"))
    return {
        "sample_count": len(results),
        "status_counts": counts,
        "successful_completion_rate": counts.get("complete", 0) / terminal_count if terminal_count else None,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
    }
