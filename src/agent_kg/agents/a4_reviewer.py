"""A4: independent API-backed adversarial review for A3 diagnosis drafts.

A4 never writes the baseline graph or final report.  It receives a bounded
evidence package, A2's auditable gate result, and A3's draft; it returns a
strict, machine-readable review decision for the orchestrator.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from ..config import ModelProfile, load_agent_config
from ..contracts import (
    A2QualityResult,
    A3DiagnosisDraft,
    A4AuditMetadata,
    A4ReviewResult,
)
from ..llm.router import ModelRouter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_VERSION = "a4-evidence-adversarial-review-v1.0"
API_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{8,}")


class A4ApiReviewError(RuntimeError):
    """An API or structured-output failure that leaves the workflow unreviewed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code


class A4Reviewer:
    role_id = "a4"
    system_instruction = """You are A4, an independent adversarial reviewer for an electric-power fault diagnosis system.

Review only the supplied A3 draft, A2 audit result, and evidence bundle. Do not introduce external facts or infer an unstated root cause. Treat A2 HOLD/REJECT items as unavailable for diagnosis unless the supplied evidence independently supports the claim.

Return approve only when the draft's material diagnostic conclusions are grounded, internally consistent, and safe to present. For approve, return exactly an empty issues list and an empty revision_instructions list. Return revise for correctable evidence gaps, ambiguity, missing citations, or causal overreach. Return reject for fundamental unsupported, contradictory, or unsafe content. For revise or reject, provide at most three concise, actionable issues and cite only evidence IDs present in the input. Do not reveal chain-of-thought; provide conclusions and short review rationales only."""

    response_schema: dict[str, Any] = {
        "type": "json_schema",
        "name": "a4_review_result",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["decision", "issues", "revision_instructions"],
            "properties": {
                "decision": {"type": "string", "enum": ["approve", "revise", "reject"]},
                "issues": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["category", "severity", "message", "evidence_ids"],
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": [
                                    "unsupported_claim",
                                    "contradiction",
                                    "missing_evidence",
                                    "safety",
                                    "report_quality",
                                ],
                            },
                            "severity": {"type": "string", "enum": ["minor", "major", "critical"]},
                            "message": {"type": "string", "minLength": 1, "maxLength": 240},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "revision_instructions": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {"type": "string", "maxLength": 240},
                },
            },
        },
    }

    def __init__(
        self,
        profile_name: str,
        profile: ModelProfile,
        *,
        audit_dir: Path | str | None = None,
        max_output_tokens: int = 1200,
        max_attempts: int = 3,
    ) -> None:
        if profile.provider not in {"openai_responses", "ollama"}:
            raise ValueError("A4Reviewer supports only 'openai_responses' or local 'ollama' model profiles.")
        self.profile_name = profile_name
        self.profile = profile
        self.audit_dir = Path(audit_dir) if audit_dir else PROJECT_ROOT / "experiments" / "logs" / "a4_reviews"
        self.max_output_tokens = max_output_tokens
        self.max_attempts = max_attempts

    @classmethod
    def from_project_config(cls, config_path: str | Path = "config/agents.yaml") -> "A4Reviewer":
        """Create A4 from the versioned project config and local ``.env`` file."""

        load_dotenv(PROJECT_ROOT / ".env", override=False)
        config = load_agent_config(config_path)
        agent = config.agents["a4"]
        if not agent.model_profile:
            raise ValueError("A4 must have a configured model_profile.")
        return cls(agent.model_profile, config.model_profiles[agent.model_profile])

    def build_request(self, draft: A3DiagnosisDraft, quality_result: A2QualityResult | None) -> dict[str, Any]:
        """Build a bounded Responses API request with no key and no tools."""

        context = {
            "review_protocol": PROMPT_VERSION,
            "run_id": draft.run_id,
            "a3_draft": draft.model_dump(mode="json"),
            "a2_quality_result": quality_result.model_dump(mode="json") if quality_result else None,
            "review_constraints": {
                "evidence_only": True,
                "a2_held_or_rejected_claims_are_not_admissible": True,
                "required_output_language": "Chinese",
            },
        }
        return {
            "model": self.profile.model,
            "instructions": self.system_instruction,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": json.dumps(context, ensure_ascii=False)}]}],
            "text": {"format": self.response_schema},
            "max_output_tokens": self.max_output_tokens,
            # Diagnosis evidence can be sensitive operational material.  We
            # retain our own redacted audit trail but opt out of API storage.
            "store": False,
        }

    async def review(self, draft: A3DiagnosisDraft, quality_result: A2QualityResult | None) -> A4ReviewResult:
        """Run an auditable A4 review through the configured remote or local backend."""

        if quality_result and quality_result.run_id != draft.run_id:
            raise ValueError("A2 result and A3 draft must share the same run_id.")
        request = self.build_request(draft, quality_result)
        request_fingerprint = self._fingerprint(request)
        started = time.perf_counter()
        try:
            if self.profile.provider == "ollama":
                raw_response = await self._local_ollama_review(request)
            else:
                raw_response = await self._request_with_retry(request, self._api_key())
        except A4ApiReviewError as exc:
            self._write_failure_audit(
                run_id=draft.run_id,
                request=request,
                request_fingerprint=request_fingerprint,
                latency_ms=round((time.perf_counter() - started) * 1000),
                error=exc,
            )
            raise
        latency_ms = round((time.perf_counter() - started) * 1000)
        parsed = self.parse_response(raw_response)
        self._validate_issue_evidence_ids(parsed, {item.source_id for item in draft.evidence})
        audit_path = self._write_audit(
            run_id=draft.run_id,
            request=request,
            raw_response=raw_response,
            parsed_response=parsed,
            request_fingerprint=request_fingerprint,
            latency_ms=latency_ms,
        )
        return A4ReviewResult(
            run_id=draft.run_id,
            decision=parsed["decision"],
            issues=parsed["issues"],
            revision_instructions=parsed["revision_instructions"],
            teacher_model_profile=self.profile_name,
            audit_metadata=A4AuditMetadata(
                provider=self.profile.provider,
                model=str(raw_response.get("model", self.profile.model)),
                prompt_version=PROMPT_VERSION,
                response_id=raw_response.get("id"),
                request_fingerprint=request_fingerprint,
                latency_ms=latency_ms,
                audit_log_path=str(audit_path),
            ),
        )

    async def _local_ollama_review(self, request: dict[str, Any]) -> dict[str, Any]:
        """Ask a local Ollama model for the same constrained A4 contract.

        Ollama does not provide the Responses API's server-side JSON-schema
        guarantee, so parsing remains mandatory and invalid output is recorded
        as an A4 failure rather than silently repaired by the orchestrator.
        """

        context = request["input"][0]["content"][0]["text"]
        router = ModelRouter(self.profile_name, self.profile)
        try:
            result = await router.chat(
                [
                    {"role": "system", "content": self.system_instruction},
                    {
                        "role": "user",
                        "content": (
                            "Return only one JSON object matching the supplied schema. "
                            "Do not use Markdown fences or add commentary.\n\n"
                            f"Review packet:\n{context}"
                        ),
                    },
                ],
                format=self.response_schema["schema"],
                options={
                    "temperature": self.profile.temperature,
                    "num_ctx": self.profile.max_context_tokens,
                    "num_predict": self.max_output_tokens,
                },
            )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
            raise A4ApiReviewError("Local Ollama A4 request failed before a review was produced.") from exc
        raw = result["raw"]
        return {
            "id": None,
            "model": raw.get("model", self.profile.model),
            "output_text": result["content"],
            "usage": {
                "prompt_eval_count": raw.get("prompt_eval_count"),
                "eval_count": raw.get("eval_count"),
            },
            "local_backend": "ollama",
        }

    def parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Extract and validate the model's structured review, never free text."""

        output_text = self._extract_output_text(response)
        try:
            payload = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise A4ApiReviewError("A4 returned non-JSON output despite structured-output mode.") from exc
        try:
            result = A4ReviewResult(
                run_id="schema-validation-only",
                teacher_model_profile=self.profile_name,
                **payload,
            )
        except Exception as exc:  # Pydantic validation exposes no secrets, but normalise the public error.
            raise A4ApiReviewError("A4 output did not meet the required review schema.") from exc
        return {
            "decision": result.decision,
            "issues": [issue.model_dump() for issue in result.issues],
            "revision_instructions": result.revision_instructions,
        }

    @staticmethod
    def _validate_issue_evidence_ids(parsed: dict[str, Any], allowed_ids: set[str]) -> None:
        """Reject, rather than silently repair, a review citing absent evidence.

        A4 may only cite IDs in the bounded packet.  This protects the
        decision-to-routing interface from cross-case citation leakage.
        """
        invalid = sorted({
            evidence_id
            for issue in parsed.get("issues", [])
            for evidence_id in issue.get("evidence_ids", [])
            if evidence_id not in allowed_ids
        })
        if invalid:
            raise A4ApiReviewError(
                "A4 review cited evidence IDs outside the bounded packet: " + ", ".join(invalid)
            )

    async def _request_with_retry(self, request: dict[str, Any], api_key: str) -> dict[str, Any]:
        base_url = self.profile.resolved_base_url().rstrip("/")
        if not base_url:
            raise A4ApiReviewError("A4 OpenAI base URL is empty.")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "electric-agent-kg-a4/0.1",
        }
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.profile.timeout_seconds) as client:
            for attempt in range(1, self.max_attempts + 1):
                try:
                    response = await client.post(f"{base_url}/responses", headers=headers, json=request)
                    if response.status_code == 429 or response.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"Retryable A4 API status: {response.status_code}", request=response.request, response=response
                        )
                    response.raise_for_status()
                    return response.json()
                except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                    last_error = exc
                    retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                        exc.response.status_code == 429 or exc.response.status_code >= 500
                    )
                    if not retryable or attempt == self.max_attempts:
                        break
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
        if isinstance(last_error, httpx.HTTPStatusError):
            status = last_error.response.status_code
            error_type, error_code, message = self._api_error_details(last_error.response)
            suffix = f" ({error_type}/{error_code})" if error_type or error_code else ""
            detail = f" Details: {message}" if message else ""
            raise A4ApiReviewError(
                f"A4 API request failed with HTTP {status}{suffix}; no review was accepted.{detail}",
                status_code=status,
                error_type=error_type,
                error_code=error_code,
            ) from last_error
        raise A4ApiReviewError("A4 API request failed after retry; no review was accepted.") from last_error

    def _api_key(self) -> str:
        key_name = self.profile.api_key_env or "OPENAI_API_KEY"
        key = os.environ.get(key_name, "").strip()
        if not key:
            raise A4ApiReviewError(f"A4 API key is not configured in environment variable {key_name}.")
        return key

    @staticmethod
    def _extract_output_text(response: dict[str, Any]) -> str:
        direct = response.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    return content["text"].strip()
        raise A4ApiReviewError("A4 API response contains no output_text review.")

    @staticmethod
    def _fingerprint(request: dict[str, Any]) -> str:
        encoded = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _write_audit(
        self,
        *,
        run_id: str,
        request: dict[str, Any],
        raw_response: dict[str, Any],
        parsed_response: dict[str, Any],
        request_fingerprint: str,
        latency_ms: int,
    ) -> Path:
        """Write a future-distillation trace after redacting key-shaped strings."""

        timestamp = datetime.now(timezone.utc)
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
        target = self.audit_dir / f"{safe_run_id}_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
        record = {
            "audit_type": "a4_local_review" if self.profile.provider == "ollama" else "a4_api_review",
            "timestamp": timestamp.isoformat(),
            "provider": self.profile.provider,
            "model_profile": self.profile_name,
            "model": raw_response.get("model", self.profile.model),
            "prompt_version": PROMPT_VERSION,
            "response_id": raw_response.get("id"),
            "request_fingerprint": request_fingerprint,
            "latency_ms": latency_ms,
            "request": request,
            "raw_response": raw_response,
            "parsed_review": parsed_response,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self._redact(record), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    def _write_failure_audit(
        self,
        *,
        run_id: str,
        request: dict[str, Any],
        request_fingerprint: str,
        latency_ms: int,
        error: A4ApiReviewError,
    ) -> Path:
        """Persist a redacted failure trace without storing headers or credentials."""

        timestamp = datetime.now(timezone.utc)
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
        target = self.audit_dir / f"{safe_run_id}_{timestamp.strftime('%Y%m%dT%H%M%SZ')}_failed.json"
        record = {
            "audit_type": "a4_local_review_failure" if self.profile.provider == "ollama" else "a4_api_review_failure",
            "timestamp": timestamp.isoformat(),
            "provider": self.profile.provider,
            "model_profile": self.profile_name,
            "model": self.profile.model,
            "prompt_version": PROMPT_VERSION,
            "request_fingerprint": request_fingerprint,
            "latency_ms": latency_ms,
            "request": request,
            "error": {
                "message": str(error),
                "http_status": error.status_code,
                "type": error.error_type,
                "code": error.error_code,
            },
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self._redact(record), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @staticmethod
    def _api_error_details(response: httpx.Response) -> tuple[str | None, str | None, str | None]:
        """Extract compact API error metadata; never return headers or request data."""

        try:
            payload = response.json()
        except ValueError:
            return None, None, None
        error = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(error, dict):
            return None, None, None
        error_type = error.get("type") if isinstance(error.get("type"), str) else None
        error_code = error.get("code") if isinstance(error.get("code"), str) else None
        message = error.get("message") if isinstance(error.get("message"), str) else None
        return error_type, error_code, message

    @staticmethod
    def _redact(value: Any) -> Any:
        if isinstance(value, str):
            return API_KEY_PATTERN.sub("[REDACTED_API_KEY]", value)
        if isinstance(value, list):
            return [A4Reviewer._redact(item) for item in value]
        if isinstance(value, dict):
            return {key: A4Reviewer._redact(item) for key, item in value.items()}
        return value

    @staticmethod
    def revision_stub(draft: A3DiagnosisDraft, teacher_model_profile: str) -> A4ReviewResult:
        """Offline fallback used only when the API is intentionally not invoked."""

        return A4ReviewResult(
            run_id=draft.run_id,
            decision="revise",
            teacher_model_profile=teacher_model_profile,
            revision_instructions=["A4 API was not invoked; retain the draft for evidence-based review."],
        )
