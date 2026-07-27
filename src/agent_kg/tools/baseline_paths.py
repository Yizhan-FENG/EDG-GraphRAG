"""读取基线路径，不允许在 Agent 项目中原地修改基线输出。"""

from pathlib import Path
import yaml


def load_baseline_root(config_path: str | Path = "config/baseline.yaml") -> Path:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(config["baseline"]["root"])
    if not root.exists():
        raise FileNotFoundError(f"Baseline project not found: {root}")
    if config["baseline"].get("access_mode") != "read_only":
        raise RuntimeError("Agent project must keep the baseline access mode read_only")
    return root
