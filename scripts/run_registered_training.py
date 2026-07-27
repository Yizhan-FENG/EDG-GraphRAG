#!/usr/bin/env python3
"""Safely complete a registered segmented LoRA run from its latest checkpoint."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); args=p.parse_args()
    cfg_path=Path(args.config); cfg=yaml.safe_load(cfg_path.read_text(encoding="utf-8")); out=Path(cfg["output_dir"])
    log=ROOT/"experiments"/"logs"/(cfg["adapter_name"]+"_orchestrator.log")
    while True:
        manifest_path=out/"run_manifest.json"; manifest=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if manifest.get("status")=="complete": break
        command=[sys.executable,"scripts/train_role_lora.py","--config",str(cfg_path)]
        if manifest.get("latest_checkpoint"): command += ["--resume-from",manifest["latest_checkpoint"]]
        with log.open("a",encoding="utf-8") as f:
            f.write("RUN "+" ".join(command)+"\n"); f.flush()
            subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({"status":"complete","output_dir":str(out)},ensure_ascii=False))
if __name__=="__main__": main()
