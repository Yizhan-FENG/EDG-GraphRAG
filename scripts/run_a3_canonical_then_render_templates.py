"""Generate canonical A3 drafts, then deterministically render report templates.

The A3 LoRA is evaluated only on its trained ``report_draft`` contract.  A
non-LLM renderer owns presentation variants, so template changes cannot alter
the evidence sandbox or be misattributed to A3 reasoning.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_kg.config import load_agent_config
from agent_kg.runtime import RoleSandboxError, SharedQwenRoleRuntime


ROOT = Path(__file__).resolve().parents[1]
CASE_FILES = [ROOT / "data" / "confirmation" / "official_2026_confirmation_cases_v1.jsonl", ROOT / "data" / "confirmation" / "official_2026_extension_cases_v2.jsonl"]
TEMPLATE_PATH = ROOT / "config" / "evaluation" / "report_template_contracts_v1.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def canonical_prompt(case: dict[str, Any]) -> str:
    evidence = "\n".join(f"[{e['source_id']}] {e['excerpt']}" for e in case["evidence"])
    depth = case.get("evidence_depth", "reported_trigger_only")
    return (
        "Return only JSON with top-level key report_draft. report_draft must contain content and evidence_ids. "
        "Use only the supplied evidence. If the evidence depth is reported_trigger_only or reported_event_only, "
        "state that a full root-cause conclusion is not established.\n"
        f"Case: {case['case_id']}\nEvidence depth: {depth}\nEvidence:\n{evidence}"
    )


def render(template: dict[str, Any], draft: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    body = draft.get("content") if isinstance(draft.get("content"), str) else json.dumps(draft, ensure_ascii=False)
    cited = [item for item in draft.get("evidence_ids", []) if isinstance(item, str)] if isinstance(draft, dict) else []
    boundary = "Only source-reported facts are presented; a full root-cause conclusion is not established from this packet."
    output: dict[str, Any] = {}
    for key in template["required_keys"]:
        if key in {"evidence_boundary", "uncertainty_and_scope", "open_questions"}:
            content = boundary
        elif key in {"verification_or_action_request", "recommended_next_check"}:
            content = "Verify the event through the source-designated operational or maintenance process; do not infer additional cause facts."
        else:
            content = body
        output[key] = {"content": content, "evidence_ids": cited}
    return output


def collect_ids(value: Any) -> set[str]:
    if isinstance(value, dict):
        own = set(value.get("evidence_ids", [])) if isinstance(value.get("evidence_ids"), list) else set()
        return {x for x in own if isinstance(x, str)} | set().union(*(collect_ids(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(collect_ids(v) for v in value)) if value else set()
    return set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="a3_canonical_template_pilot_v1")
    parser.add_argument("--case-limit", type=int, default=2)
    args = parser.parse_args()
    cases = [case for file in CASE_FILES for case in read_jsonl(file)][: args.case_limit]
    templates = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))["templates"]
    out = ROOT / "experiments" / "multitemplate" / args.label
    out.mkdir(parents=True, exist_ok=True)
    runtime = SharedQwenRoleRuntime(load_agent_config(ROOT / "config" / "agents.yaml"))
    canonical_rows: list[dict[str, Any]] = []
    rendered_rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            allowed = [e["source_id"] for e in case["evidence"]]
            try:
                prediction = runtime.generate_json("a3", canonical_prompt(case), allowed, max_new_tokens=512, contract_override={"report_draft"}, instruction_override="Return only the canonical A3 report_draft JSON contract.")
                draft = prediction["report_draft"] if isinstance(prediction.get("report_draft"), dict) else prediction
                cited = collect_ids(draft)
                canonical_rows.append({"case_id": case["case_id"], "status": "complete", "prediction": prediction, "citation_whitelist_valid": cited <= set(allowed), "cited_evidence_ids": sorted(cited)})
                for template in templates:
                    rendered = render(template, draft, case)
                    rendered_ids = collect_ids(rendered)
                    rendered_rows.append({"case_id": case["case_id"], "template_id": template["template_id"], "status": "complete", "rendered": rendered, "template_complete": set(template["required_keys"]) <= set(rendered), "citation_whitelist_valid": rendered_ids <= set(allowed), "cited_evidence_ids": sorted(rendered_ids)})
            except RoleSandboxError as exc:
                canonical_rows.append({"case_id": case["case_id"], "status": "failed", "error": str(exc)})
            write_jsonl(out / "canonical_predictions.jsonl", canonical_rows)
            write_jsonl(out / "rendered_templates.jsonl", rendered_rows)
    finally:
        runtime.unload()
    complete_canonical = [r for r in canonical_rows if r["status"] == "complete"]
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "scope": "canonical A3 draft plus deterministic template rendering; not correctness evaluation", "case_count": len(cases), "template_count": len(templates), "canonical_json_rate": len(complete_canonical) / len(canonical_rows) if canonical_rows else 0.0, "canonical_citation_whitelist_rate": sum(r["citation_whitelist_valid"] for r in complete_canonical) / len(complete_canonical) if complete_canonical else 0.0, "render_count": len(rendered_rows), "template_complete_rate": sum(r["template_complete"] for r in rendered_rows) / len(rendered_rows) if rendered_rows else 0.0, "rendered_citation_whitelist_rate": sum(r["citation_whitelist_valid"] for r in rendered_rows) / len(rendered_rows) if rendered_rows else 0.0}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
