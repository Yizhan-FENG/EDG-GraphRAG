from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_kg.config import AgentProjectConfig
from agent_kg.evaluation.role_metrics import a1_claim_set, aggregate, evidence_ids, prf
from agent_kg.runtime.role_lora import RoleSandboxError, SharedQwenRoleRuntime


def sandbox_runtime(tmp_path: Path) -> SharedQwenRoleRuntime:
    hf_home = tmp_path / "hf"
    snapshot_id = "test-snapshot"
    snapshot = hf_home / "hub" / "models--Qwen--Qwen3-4B" / "snapshots" / snapshot_id
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    ref = hf_home / "hub" / "models--Qwen--Qwen3-4B" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text(snapshot_id, encoding="utf-8")

    adapters: dict[str, dict[str, str]] = {}
    agents: dict[str, dict[str, object]] = {}
    for role in ("a1", "a3"):
        adapter_dir = tmp_path / role / "best"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        adapter_name = f"{role}_adapter"
        adapters[adapter_name] = {
            "backbone": "m1_qwen3_4b",
            "role": role,
            "status": "formal_training_complete",
            "training_target": "synthetic test",
            "output_dir": str(adapter_dir),
        }
        agents[role] = {
            "display_name": role,
            "model_profile": "local",
            "backbone": "m1_qwen3_4b",
            "adapter": adapter_name,
            "sandbox": {"read": [], "write": [], "deny": ["baseline.kg.write"]},
        }

    config = AgentProjectConfig.model_validate(
        {
            "project": {"name": "test"},
            "model_profiles": {
                "local": {
                    "provider": "ollama",
                    "model": "qwen3:4b",
                    "base_url": "http://localhost:11434",
                    "temperature": 0,
                    "timeout_seconds": 10,
                    "max_context_tokens": 1024,
                }
            },
            "shared_backbones": {
                "m1_qwen3_4b": {
                    "model_profile": "local",
                    "hf_base_model": "Qwen/Qwen3-4B",
                    "hf_home": str(hf_home),
                    "roles": ["a1", "a3"],
                    "activation_rule": "one_adapter_per_request",
                    "base_model_training": "frozen",
                }
            },
            "lora_adapters": adapters,
            "agents": agents,
            "infrastructure": {},
        }
    )
    return SharedQwenRoleRuntime(config)


def test_a1_claim_metrics_and_evidence_are_deterministic() -> None:
    payload = {"claims": [{"subject": "裂纹", "predicate": "导致", "object": "泄漏", "evidence_ids": ["ev-1"]}]}
    assert a1_claim_set(payload) == {"裂纹|导致|泄漏"}
    assert evidence_ids(payload) == {"ev-1"}
    assert prf({"a"}, {"a", "b"})["f1"] == pytest.approx(2 / 3)
    summary = aggregate([{"json_valid": True, "evidence": {"correct": 1, "predicted": 2, "gold": 1}}])
    assert summary["evidence"]["precision"] == pytest.approx(0.5)


def test_completed_adapter_sandbox_rejects_unknown_role_and_missing_evidence(tmp_path: Path) -> None:
    runtime = sandbox_runtime(tmp_path)
    with pytest.raises(RoleSandboxError):
        runtime.validate_request("a2", ["ev-1"])
    with pytest.raises(RoleSandboxError):
        runtime.validate_request("a1", [])
    assert runtime.validate_request("a1", ["ev-1"]).name == "best"


def test_base_model_control_keeps_sandbox_but_skips_adapter_lookup(tmp_path: Path) -> None:
    runtime = sandbox_runtime(tmp_path)
    assert runtime.validate_request("a1", ["ev-1"], use_adapter=False) is None
    with pytest.raises(RoleSandboxError):
        runtime.validate_request("a1", [], use_adapter=False)


def test_runtime_resolves_the_explicit_local_qwen_snapshot(tmp_path: Path) -> None:
    runtime = sandbox_runtime(tmp_path)
    snapshot = runtime._local_model_source()
    assert Path(snapshot).name
    assert (Path(snapshot) / "config.json").is_file()


def test_role_sandbox_collects_only_string_evidence_ids() -> None:
    payload = json.loads('{"evidence_ids":["ev-1",3],"nested":{"evidence_ids":["ev-2"]}}')
    assert SharedQwenRoleRuntime._collect_evidence_ids(payload) == {"ev-1", "ev-2"}


def test_role_sandbox_parses_final_json_after_non_contract_preamble() -> None:
    assert SharedQwenRoleRuntime._parse_json_object("说明文字```json\n{\"evidence_ids\":[\"ev-1\"]}\n```", "a1", required_keys={"evidence_ids"}) == {
        "evidence_ids": ["ev-1"]
    }
