"""Application configuration, read from the submission env-var contract.

LLM_NAME / LLM_BASE_URL / LLM_API_KEY are the fixed Hack Apertus contract
(template README); the whole app must run on any OpenAI-compatible endpoint.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LlmConfig:
    name: str
    base_url: str
    api_key: str
    configured: bool


def load_llm_config() -> LlmConfig:
    """Build the runtime LLM configuration from the process environment."""
    name: str = os.environ.get("LLM_NAME", "").strip()
    base_url: str = os.environ.get("LLM_BASE_URL", "").strip().rstrip("/")
    api_key: str = os.environ.get("LLM_API_KEY", "").strip()
    configured: bool = bool(name and base_url and api_key)
    config: LlmConfig = LlmConfig(name=name,
                                  base_url=base_url,
                                  api_key=api_key,
                                  configured=configured)
    return config


def looks_like_thinking_model(name: Optional[str]) -> bool:
    """Heuristic pre-selection for the 'natively thinking' GUI checkbox."""
    lowered: str = (name or "").lower()
    return "thinking" in lowered or "reasoning" in lowered
