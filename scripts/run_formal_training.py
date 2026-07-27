#!/usr/bin/env python3
"""Serially finish resumable A1 then A3 formal QLoRA training on one GPU."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    ROOT / "config" / "training" / "a1_qwen3_4b_formal.yaml",
    ROOT / "config" / "training" / "a3_qwen3_4b_formal.yaml",
]


def latest_checkpoint(output_dir: Path) -> Path | None:
    manifest = output_dir / "run_manifest.json"
    if not manifest.is_file():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    checkpoint = data.get("latest_checkpoint")
    return Path(checkpoint) if checkpoint else None


def main() -> None:
    log_path = ROOT / "experiments" / "logs" / "formal_training_orchestrator.log"
    with log_path.open("a", encoding="utf-8") as log:
        for config_path in CONFIGS:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            output_dir = Path(config["output_dir"])
            while True:
                manifest_path = output_dir / "run_manifest.json"
                if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "complete":
                    print(f"{config['role']}: already complete", file=log, flush=True)
                    break
                command = [sys.executable, "scripts/train_role_lora.py", "--config", str(config_path)]
                checkpoint = latest_checkpoint(output_dir)
                if checkpoint:
                    command += ["--resume-from", str(checkpoint)]
                print("RUN " + " ".join(command), file=log, flush=True)
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("ALL_FORMAL_TRAINING_COMPLETE", file=log, flush=True)


if __name__ == "__main__":
    main()
