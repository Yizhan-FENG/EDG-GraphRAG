from __future__ import annotations

import json

from agent_kg.orchestration.ablation import load_terminal_result, stable_run_id, summarize_results


def test_stable_run_id_uses_sample_id_and_is_filesystem_safe() -> None:
    record = {"metadata": {"sample_id": "case / 01: A"}}
    assert stable_run_id("B2_dual_lora_a2", record) == "B2_dual_lora_a2--case---01--A"


def test_terminal_result_can_be_resumed_only_for_matching_experiment(tmp_path) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"experiment_id": "B2", "run_id": "B2--s1", "status": "structural_failure"}), encoding="utf-8")
    assert load_terminal_result(path, "B2", "B2--s1") is not None
    assert load_terminal_result(path, "B1", "B2--s1") is None


def test_summary_keeps_failures_in_denominator() -> None:
    summary = summarize_results([{"status": "complete", "latency_seconds": 2.0}, {"status": "structural_failure", "latency_seconds": 4.0}])
    assert summary["successful_completion_rate"] == 0.5
    assert summary["status_counts"] == {"complete": 1, "structural_failure": 1}
