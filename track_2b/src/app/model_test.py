"""'Test model — Hello World': one minimal chat completion, immediate diagnostics.

Sends exactly one request through the configured env-var trio and reports
reachability, auth, the model actually served, latency, and the visible error
on failure. Doubles as the on-site key-surprise smoke check (demo script).
"""

import time
from typing import Any, Dict, Optional

import httpx

from .config import LlmConfig

TEST_TIMEOUT_SECONDS = 30.0

# The GUI's token-budget slider is advisory: whatever it asks for is clamped into
# this band server-side, so a tampered or nonsensical value cannot make the probe
# expensive, and the effective number is echoed back rather than trusted.
#
# These are RAW tokens, the unit a chat completion takes ("max_tokens"). The
# platform's per-model budget is a different thing in a different unit — kilotokens,
# stored as "token_limit_kilo", where 200 means 200000 tokens.
#
# The default is 80000, below the endpoint's own limit of 100000: measured
# 2026-10-11 against https://llm.thalabus.ai/v1, max_tokens=200000 answers HTTP 400
# "cannot be greater than max_total_tokens=100000", while 32768 answers 200.
DEFAULT_MAX_TOKENS = 80000
MIN_MAX_TOKENS = 16
MAX_MAX_TOKENS = 300000


async def test_model(config: LlmConfig, thinking: bool = False,
                     max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Probe the configured endpoint with one 'Hello World' completion."""
    effective_max_tokens: int = _clamp_max_tokens(max_tokens)
    result: Dict[str, Any] = {
        "ok": False,
        "requested_model": config.name,
        "base_url": config.base_url,
        "model_served": None,
        "max_tokens": effective_max_tokens,
        "latency_ms": None,
        "reply": None,
        "error": None,
    }

    if not config.configured:
        result["error"] = {"type": "not_configured", "message": "LLM_NAME / LLM_BASE_URL / LLM_API_KEY are not all set (see .env.example)."}
        return result

    payload: Dict[str, Any] = {
        "model": config.name,
        "messages": [{"role": "user", "content": "Reply with exactly: Hello world"}],
        "max_tokens": effective_max_tokens,
        "temperature": 0.0,
    }
    if thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    request_url: str = f"{config.base_url}/chat/completions"
    started: float = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT_SECONDS) as client:
            response: httpx.Response = await client.post(request_url, json=payload, headers=headers)
        result["latency_ms"] = round((time.perf_counter() - started) * 1000)
        if response.status_code != 200:
            result["error"] = {"type": f"http_{response.status_code}", "message": _safe_body(response)}
            return result
        body: Dict[str, Any] = response.json()
        result["model_served"] = _extract_field(body, "model")
        result["reply"] = _extract_reply(body)
        result["ok"] = result["reply"] is not None

    except httpx.TimeoutException:
        result["error"] = {"type": "timeout", "message": f"No response within {TEST_TIMEOUT_SECONDS:.0f}s."}
    except (httpx.ConnectError, httpx.RequestError) as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    except ValueError as exc:
        result["error"] = {"type": "malformed_json", "message": str(exc)}

    return result


def _clamp_max_tokens(value: Optional[int]) -> int:
    """Bound the requested token budget, falling back to the default when unusable."""
    if value is None:
        return DEFAULT_MAX_TOKENS

    try:
        requested: int = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_TOKENS

    return max(MIN_MAX_TOKENS, min(MAX_MAX_TOKENS, requested))


def _extract_field(body: Dict[str, Any], key: str) -> Optional[str]:
    value: Optional[str] = body.get(key)
    return value if isinstance(value, str) else None


def _extract_reply(body: Dict[str, Any]) -> Optional[str]:
    try:
        content: Any = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    reply: Optional[str] = content if isinstance(content, str) and content else None
    return reply


def _safe_body(response: httpx.Response) -> str:
    text: str = response.text or ""
    return text[:500]
