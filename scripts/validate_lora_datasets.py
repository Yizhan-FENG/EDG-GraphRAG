#!/usr/bin/env python3
"""Validate role-separated JSONL datasets built for the A1/A3 LoRA adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON: {error}") from error
    return rows


def validate_chat_rows(rows: list[dict[str, Any]], expected_role: str, path: Path) -> set[str]:
    source_documents: set[str] = set()
    for index, row in enumerate(rows, start=1):
        messages = row.get("messages")
        metadata = row.get("metadata", {})
        if not isinstance(messages, list) or [message.get("role") for message in messages] != [
            "system",
            "user",
            "assistant",
        ]:
            raise ValueError(f"{path}:{index}: expected system/user/assistant messages")
        if metadata.get("role") != expected_role:
            raise ValueError(f"{path}:{index}: expected role {expected_role}")
        source_document = metadata.get("source_document")
        if not source_document:
            raise ValueError(f"{path}:{index}: source_document is required")
        source_documents.add(source_document)
        try:
            json.loads(messages[-1]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}:{index}: assistant target must be JSON") from error
    return source_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args()

    all_documents: dict[str, dict[str, set[str]]] = {"A1": {}, "A3": {}}
    specifications = {
        "A1": (args.data_root / "a1", "{split}.jsonl"),
        "A3": (args.data_root / "a3", "evidence_alignment_{split}.jsonl"),
    }
    for role, (directory, name_template) in specifications.items():
        for split in ("train", "validation", "test"):
            path = directory / name_template.format(split=split)
            rows = read_rows(path)
            all_documents[role][split] = validate_chat_rows(rows, role, path)
            print(f"{role} {split}: {len(rows)} rows, {len(all_documents[role][split])} source documents")

        splits = all_documents[role]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = splits[left] & splits[right]
            if overlap:
                raise ValueError(f"{role}: source-document leakage between {left} and {right}: {sorted(overlap)[:3]}")

    review_queue = args.data_root / "a3" / "unreviewed_trace_queue.jsonl"
    review_rows = read_rows(review_queue)
    if any(row.get("status") != "unreviewed" for row in review_rows):
        raise ValueError("A3 review queue may contain only unreviewed records")
    print(f"A3 review queue: {len(review_rows)} unreviewed legacy traces")
    print("Validation passed. Unreviewed traces are not training data.")


if __name__ == "__main__":
    main()
