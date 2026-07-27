"""Deterministic, text-triggered relation candidates for the A1-v5 ablation."""
from __future__ import annotations

import hashlib
import re
from typing import Any

from .role_metrics import normalize


# Legacy predicates preserve comparability with the frozen A1 labels; the
# ontology relation documents the v2 semantic interpretation.
TRIGGERS = [
    ("导致", "causes", ("导致", "造成", "引起", "使得")),
    ("表现为", "manifests_as", ("表现为", "出现", "呈现")),
    ("需要", "requires", ("需要", "必须", "应当")),
    ("采用", "diagnosed_by", ("采用", "使用", "通过")),
    ("预防", "prevents", ("预防", "防止", "避免")),
    ("包含", "part_of", ("包含", "包括", "由", "组成")),
]


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[。！？；\n]+", text) if part.strip()]


def _edge_id(subject: str, predicate: str, obj: str) -> str:
    return "e-" + hashlib.sha256(f"{subject}|{predicate}|{obj}".encode("utf-8")).hexdigest()[:12]


def candidate_edges(text: str, entities: list[dict[str, Any]], evidence_ids: list[str], *, limit: int = 24) -> list[dict[str, Any]]:
    """Create directed, non-self-loop candidates supported by textual triggers."""
    names = []
    seen = set()
    for entity in entities:
        name = str(entity.get("name", "")).strip()
        key = normalize(name)
        if name and key not in seen:
            names.append(name)
            seen.add(key)
    candidates: dict[str, dict[str, Any]] = {}
    for sentence_index, sentence in enumerate(_sentences(text)):
        occurrences = sorted((sentence.find(name), name) for name in names if sentence.find(name) >= 0)
        if len(occurrences) < 2:
            continue
        for predicate, ontology_relation, markers in TRIGGERS:
            marker_positions = [sentence.find(marker) for marker in markers if sentence.find(marker) >= 0]
            if not marker_positions:
                continue
            marker_position = min(marker_positions)
            left = [name for position, name in occurrences if position < marker_position]
            right = [name for position, name in occurrences if position > marker_position]
            for subject in left[-2:]:
                for obj in right[:2]:
                    if normalize(subject) == normalize(obj):
                        continue
                    edge_id = _edge_id(subject, predicate, obj)
                    candidates.setdefault(edge_id, {
                        "edge_id": edge_id,
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "ontology_relation": ontology_relation,
                        "evidence_ids": list(evidence_ids),
                        "trigger": next(marker for marker in markers if marker in sentence),
                        "sentence_index": sentence_index,
                    })
    return list(candidates.values())[:limit]
