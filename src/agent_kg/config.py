from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelProfile(BaseModel):
    provider: Literal["ollama", "openai_compatible", "openai_responses"]
    model: str
    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None
    temperature: float = Field(ge=0.0, le=2.0)
    timeout_seconds: int = Field(gt=0)
    max_context_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_endpoint(self) -> "ModelProfile":
        if self.provider == "ollama" and not self.base_url:
            raise ValueError("Ollama profile requires base_url")
        if self.provider in {"openai_compatible", "openai_responses"} and not (self.base_url or self.base_url_env):
            raise ValueError("OpenAI-compatible profile requires base_url or base_url_env")
        return self

    def resolved_base_url(self) -> str:
        return self.base_url or os.environ.get(self.base_url_env or "", "")


class AgentSpec(BaseModel):
    display_name: str
    model_profile: str | None = None
    fallback_model_profile: str | None = None
    backbone: str | None = None
    adapter: str | None = None
    role_mode: str | None = None
    sandbox: dict[str, list[str]]


class SharedBackboneSpec(BaseModel):
    model_profile: str
    hf_base_model: str | None = None
    hf_home: str | None = None
    roles: list[str]
    activation_rule: str
    base_model_training: str


class LoRAAdapterSpec(BaseModel):
    backbone: str
    role: str
    status: str
    training_target: str
    output_dir: str


class AgentProjectConfig(BaseModel):
    project: dict
    model_profiles: dict[str, ModelProfile]
    shared_backbones: dict[str, SharedBackboneSpec] = Field(default_factory=dict)
    lora_adapters: dict[str, LoRAAdapterSpec] = Field(default_factory=dict)
    agents: dict[str, AgentSpec]
    infrastructure: dict


def load_agent_config(config_path: str | Path = "config/agents.yaml") -> AgentProjectConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return AgentProjectConfig.model_validate(_expand_environment(data))


def _expand_environment(value: object) -> object:
    """Expand environment variables in configuration values, recursively."""

    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value
