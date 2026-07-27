from __future__ import annotations

import json

import pytest

from agent_kg.agents.a4_reviewer import A4ApiReviewError, A4Reviewer
from agent_kg.config import ModelProfile
from agent_kg.contracts import A3DiagnosisDraft, DiagnosisSection, EvidenceRef


def reviewer(tmp_path) -> A4Reviewer:
    profile = ModelProfile(
        provider="openai_responses",
        model="test-model",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        temperature=0.0,
        timeout_seconds=30,
        max_context_tokens=4096,
    )
    return A4Reviewer("a4_test", profile, audit_dir=tmp_path)


def draft() -> A3DiagnosisDraft:
    return A3DiagnosisDraft(
        run_id="a4-test-001",
        diagnosis_object={"diagnosis": "synthetic"},
        report_sections=[DiagnosisSection(title="Conclusion", content="Supported synthetic conclusion.", evidence_ids=["ev-1"])],
        evidence=[EvidenceRef(source_type="input_text", source_id="ev-1", excerpt="Synthetic evidence.")],
        model_profile="a3-test",
    )


def test_a4_request_is_structured_and_opted_out_of_api_storage(tmp_path) -> None:
    request = reviewer(tmp_path).build_request(draft(), quality_result=None)

    assert request["store"] is False
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    assert request["input"][0]["content"][0]["type"] == "input_text"


def test_a4_parses_a_valid_structured_response(tmp_path) -> None:
    response = {
        "id": "resp_test",
        "model": "test-model",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(
                            {
                                "decision": "revise",
                                "issues": [
                                    {
                                        "category": "missing_evidence",
                                        "severity": "major",
                                        "message": "The conclusion needs one explicit evidence citation.",
                                        "evidence_ids": ["ev-1"],
                                    }
                                ],
                                "revision_instructions": ["Cite ev-1 in the conclusion."],
                            }
                        ),
                    }
                ],
            }
        ],
    }

    parsed = reviewer(tmp_path).parse_response(response)

    assert parsed["decision"] == "revise"
    assert parsed["issues"][0]["evidence_ids"] == ["ev-1"]


def test_a4_rejects_non_json_api_output(tmp_path) -> None:
    with pytest.raises(A4ApiReviewError):
        reviewer(tmp_path).parse_response({"output_text": "This is not structured JSON."})


def test_a4_rejects_issue_citations_outside_the_packet(tmp_path) -> None:
    parsed = {
        "decision": "revise",
        "issues": [{"category": "missing_evidence", "severity": "major", "message": "missing", "evidence_ids": ["ev-other"]}],
        "revision_instructions": [],
    }
    with pytest.raises(A4ApiReviewError, match="outside the bounded packet"):
        reviewer(tmp_path)._validate_issue_evidence_ids(parsed, {"ev-1"})


def test_a4_redacts_key_shaped_text_before_audit_persistence(tmp_path) -> None:
    value = {"nested": ["sk-" + "example_secret_123456789"]}

    assert reviewer(tmp_path)._redact(value)["nested"][0] == "[REDACTED_API_KEY]"


def test_a4_accepts_local_ollama_profile(tmp_path) -> None:
    profile = ModelProfile(
        provider="ollama",
        model="glm4:9b",
        base_url="http://localhost:11434",
        temperature=0.0,
        timeout_seconds=600,
        max_context_tokens=4096,
    )

    local = A4Reviewer("a4_glm4_9b_local", profile, audit_dir=tmp_path)

    assert local.profile.provider == "ollama"
    assert local.build_request(draft(), quality_result=None)["store"] is False
