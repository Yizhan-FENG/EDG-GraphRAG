"""按角色路由到本地 Ollama 或 OpenAI-compatible API。"""

from __future__ import annotations

import os
from typing import Any

import httpx

from ..config import ModelProfile


class ModelRouter:
    def __init__(self, profile_name: str, profile: ModelProfile):
        self.profile_name = profile_name
        self.profile = profile

    async def chat(self, messages: list[dict[str, str]], **extra: Any) -> dict[str, Any]:
        if self.profile.provider == "ollama":
            return await self._ollama_chat(messages, **extra)
        return await self._openai_compatible_chat(messages, **extra)

    async def _ollama_chat(self, messages: list[dict[str, str]], **extra: Any) -> dict[str, Any]:
        payload = {
            "model": self.profile.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.profile.temperature},
            **extra,
        }
        async with httpx.AsyncClient(timeout=self.profile.timeout_seconds) as client:
            response = await client.post(f"{self.profile.resolved_base_url()}/api/chat", json=payload)
            response.raise_for_status()
            result = response.json()
        return {
            "content": result.get("message", {}).get("content", ""),
            "raw": result,
            "model_profile": self.profile_name,
        }

    async def _openai_compatible_chat(self, messages: list[dict[str, str]], **extra: Any) -> dict[str, Any]:
        api_key = os.environ.get(self.profile.api_key_env or "")
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.profile.api_key_env}")
        payload = {
            "model": self.profile.model,
            "messages": messages,
            "temperature": self.profile.temperature,
            **extra,
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        async with httpx.AsyncClient(timeout=self.profile.timeout_seconds) as client:
            response = await client.post(
                f"{self.profile.resolved_base_url().rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()
        return {
            "content": result["choices"][0]["message"]["content"],
            "raw": result,
            "model_profile": self.profile_name,
        }
