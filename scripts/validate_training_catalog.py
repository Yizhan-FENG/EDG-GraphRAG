#!/usr/bin/env python3
"""Validate explicit LoRA data sources, counts, weights, and forbidden inputs."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "manifests" / "fine_tuning_catalog_v0.3.json"


def count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    forbidden = {Path(value) for value in catalog["forbidden_inputs"]}
    for role in ("a1", "a3"):
        section = catalog[role]
        weights = section["suggested_train_sampling_weight"]
        if set(weights) != set(section["counts"]):
            raise ValueError(f"{role}: sources and sampling weights do not match")
        for source_name, split_counts in section["counts"].items():
            if source_name not in section:
                raise ValueError(f"{role}: missing paths for {source_name}")
            for split, expected_count in split_counts.items():
                path = Path(section[source_name][split])
                if path in forbidden:
                    raise ValueError(f"{role}: forbidden dataset listed as training source: {path}")
                if not path.is_file():
                    raise FileNotFoundError(f"{role}: missing dataset {path}")
                actual_count = count_jsonl(path)
                if actual_count != expected_count:
                    raise ValueError(
                        f"{role}: {source_name} {split} count mismatch: catalog={expected_count}, actual={actual_count}"
                    )
                if weights[source_name] <= 0:
                    raise ValueError(f"{role}: non-positive sampling weight for {source_name}")
    print("Training catalog validated: explicit paths, counts, positive weights, and no forbidden inputs.")


if __name__ == "__main__":
    main()
