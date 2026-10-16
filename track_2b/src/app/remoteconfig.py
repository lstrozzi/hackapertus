"""SaaS-plan remote model configuration — the T26 client (applies, not remounts).

The platform deliberately ships no mounted configuration file: its own LLM
entry is reconfigured over the public REST API at runtime (T26
`POST /v1/api/platform/config/model`), which is why this client exists. The
GUI exposes it as an explicit button rather than a boot push, so applying is
a visible act; the route diffs before writing, so pressing it again is safe
and after any platform restart it simply needs pressing again (T26 C3).

Trust model, per T26:

* the token must be the GLOBAL platform token (`platfadmin`); a session
  token authenticates but answers `403` on every `platform/*` route (C1);
* PATCH semantics: blank fields are left untouched, so a blank key here
  keeps the platform's own key and `tokenConfigured: true` in the reply;
* reads are redacted (`WithoutModelTokens`), so the read-back confirms
  `endpoint`/`model` only — a key that was read back would also be written
  back, and it never is;
* this module only proxies: the browser never learns the platform token and
  never talks to the platform directly.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from .thalabus import ThalabusConfig

CONFIGURE_TIMEOUT_SECONDS = 15.0
READBACK_TIMEOUT_SECONDS = 10.0

CONFIGURE_PATH = "/v1/api/platform/config/model"
READBACK_PATH = "/v1/api/platform/config/llm"

# The shipped default of ThalabusWeb/Configuration/LlmSettings.json (entry
# `DEFAULT`). The endpoint is a platform-side address on purpose: the value
# tells the PLATFORM where its model lives, not this laptop — the SaaS
# instance reaches its own GPU stack, the laptop does not (and must not).
DEFAULT_SAAS_ID = "DEFAULT"
DEFAULT_SAAS_NAME = "DEFAULT"
DEFAULT_SAAS_MODEL = "onprem-ai/Apertus-v1.5-70B-FP8"
DEFAULT_SAAS_BASE_URL = "http://thalabus-ai-3:8352/v1"
# Kilotokens, the unit the platform stores (LlmSetting.token_limit_kilo). 0 means
# 'leave the platform's own budget alone': this route is PATCH-like, and the
# shipped DEFAULT entry already carries a budget (128 = 128000 tokens). The unit is
# deliberately spelled out everywhere because a chat completion's max_tokens is raw
# tokens — the same words, two units, one order of magnitude apart.
DEFAULT_SAAS_TOKEN_LIMIT_KILO = 0


@dataclass(frozen=True)
class SaasModelDefaults:
    """The `SAAS_CFG_*` preset used by 'use default' and by source=default saves."""

    id: str
    name: str
    model: str
    base_url: str
    key: str
    reasoning: bool
    token_limit_kilo: int


_TRUE_WORDS = {"1", "true", "yes", "on"}


def _env_bool(name: str, fallback: bool) -> bool:
    raw: str = os.environ.get(name, "").strip().lower()
    if not raw:
        return fallback
    return raw in _TRUE_WORDS


def _env_str(name: str, fallback: str) -> str:
    return os.environ.get(name, "").strip() or fallback


def _env_int(name: str, fallback: int) -> int:
    """Read a whole-number setting, falling back when unset, blank, or not a number."""
    raw: str = os.environ.get(name, "").strip()
    if not raw:
        return fallback

    try:
        value: int = int(raw)
    except ValueError:
        return fallback

    return value if value > 0 else fallback


def load_saas_defaults() -> SaasModelDefaults:
    """Read the .env preset; every variable falls back to the shipped DEFAULT entry."""
    defaults: SaasModelDefaults = SaasModelDefaults(
        id=_env_str("SAAS_CFG_ID", DEFAULT_SAAS_ID),
        name=_env_str("SAAS_CFG_NAME", DEFAULT_SAAS_NAME),
        model=_env_str("SAAS_CFG_MODEL", DEFAULT_SAAS_MODEL),
        base_url=_env_str("SAAS_CFG_BASE_URL", DEFAULT_SAAS_BASE_URL),
        key=_env_str("SAAS_CFG_KEY", ""),
        reasoning=_env_bool("SAAS_CFG_REASONING", False),
        token_limit_kilo=_env_int("SAAS_CFG_TOKEN_LIMIT_KILO", DEFAULT_SAAS_TOKEN_LIMIT_KILO),
    )
    return defaults


def _payload_from(defaults: SaasModelDefaults) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": defaults.id,
        "name": defaults.name,
        "model_id": defaults.model,
        "base_url": defaults.base_url,
        "key": defaults.key,
        "reasoning": defaults.reasoning,
        "token_limit_kilo": defaults.token_limit_kilo,
    }
    for field in list(payload):
        value = payload[field]
        if value is None or (isinstance(value, str) and not value):
            del payload[field]

    # A blank SAAS_CFG_TOKEN_LIMIT_KILO means "do not touch the budget the platform already
    # has", and the route refuses a budget of zero — so an unset entry must send no field at
    # all rather than the 0 it defaults to. Silence here is a deliberate no-op, not an error.
    if isinstance(payload.get("token_limit_kilo"), int) and payload["token_limit_kilo"] <= 0:
        del payload["token_limit_kilo"]

    return payload


def _hint_for(status: int, body: str) -> str:
    hints: Dict[int, str] = {
        400: "the platform refused the body (it must carry the model 'id')",
        401: "THALABUS_TOKEN is absent or invalid — the platform refused authentication",
        403: "the token authenticated but is not a platform-admin token: this route "
             "requires the GLOBAL platform token; a session token always answers 403 (T26 C1)",
        404: "no model entry with that id on the platform — check SAAS_CFG_ID / the id field",
        500: "the platform could not load or save its settings — a real failure, never a no-op (T26)",
    }
    if status in hints:
        return hints[status]
    return body


async def get_view(thalabus: ThalabusConfig, defaults: SaasModelDefaults) -> Dict[str, Any]:
    """Everything the GUI card shows: applicability, preset (never the key), target."""
    hint: Optional[str] = None
    if thalabus.plan != "saas":
        hint = "SaaS-only card: loaded under Plan Local, where the platform is configured by its own stack."
    elif not thalabus.token:
        hint = "Plan SaaS needs THALABUS_TOKEN (the platform-admin token) before this card can save — set it in .env."

    return {
        "visible": thalabus.plan == "saas",
        "plan": thalabus.plan,
        "token_present": bool(thalabus.token),
        "platform_base_url": thalabus.base_url,
        "configure_endpoint": thalabus.base_url + CONFIGURE_PATH,
        "hint": hint,
        "defaults": {
            "id": defaults.id,
            "name": defaults.name,
            "model": defaults.model,
            "base_url": defaults.base_url,
            "key_present": bool(defaults.key),
            "reasoning": defaults.reasoning,
            "token_limit_kilo": defaults.token_limit_kilo,
            "token_budget_tokens": defaults.token_limit_kilo * 1000,
        },
    }


async def apply_model_config(thalabus: ThalabusConfig,
                             defaults: SaasModelDefaults,
                             selection: str,
                             overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Apply the preset or the custom form to the platform, then read back what it stored.

    The POST reply is the authority for `changed` and `tokenConfigured`; the
    GET is the authority for `endpoint`/`model` — it is redacted, so it can
    confirm the model without ever exposing the platform's key (T26 C1).
    """
    result: Dict[str, Any] = {
        "ok": False,
        "applied_via": None,
        "post_status": None,
        "platform_reply": None,
        "readback": None,
        "verify": {"endpoint_matches": False, "model_matches": False},
        "error": None,
        "hint": None,
    }

    if thalabus.plan != "saas":
        result["error"] = {"type": "wrong_plan", "message": f"plan is '{thalabus.plan}', the SaaS platform route is only used from Plan SaaS"}
        result["hint"] = "Run with PLAN=SaaS, or use the platform's own settings UI under Plan Local."
        return result
    if not thalabus.token:
        result["error"] = {"type": "no_token", "message": "THALABUS_TOKEN is empty"}
        result["hint"] = "Set THALABUS_TOKEN in .env (the GLOBAL platform token, not a session token), then recreate the app container."
        return result

    source: str = "custom" if selection == "custom" else "default"
    if source == "default":
        payload: Dict[str, Any] = _payload_from(defaults)
    else:
        payload = {}
        body: Dict[str, Any] = overrides or {}
        for field in ("id", "name", "model_id", "base_url", "key"):
            value: Any = body.get(field)
            if isinstance(value, str) and value.strip():
                payload[field] = value.strip()
        if isinstance(body.get("reasoning"), bool):
            payload["reasoning"] = body["reasoning"]
        requested_kilo: Any = body.get("token_limit_kilo")
        if isinstance(requested_kilo, int) and requested_kilo > 0:
            payload["token_limit_kilo"] = requested_kilo
        if "id" not in payload:
            result["error"] = {"type": "missing_id", "message": "custom saves must say which model entry to configure"}
            result["hint"] = "Fill the 'id' field (e.g. DEFAULT) — blank fields stay untouched by the platform."
            return result

    headers: Dict[str, str] = {"Authorization": f"Bearer {thalabus.token}"}
    url: str = thalabus.base_url + CONFIGURE_PATH
    try:
        async with httpx.AsyncClient(timeout=CONFIGURE_TIMEOUT_SECONDS) as client:
            response: httpx.Response = await client.post(url, json=payload, headers=headers)
            result["post_status"] = response.status_code
            if response.status_code != 200:
                body_text: str = (response.text or "")[:300]
                result["error"] = {"type": f"platform_http_{response.status_code}", "message": body_text}
                result["hint"] = _hint_for(response.status_code, body_text)
                return result
            reply: Dict[str, Any] = response.json() if response.content else {}
            changed = reply.get("changed") or []
            result["ok"] = True
            result["applied_via"] = source
            result["platform_reply"] = reply
            result["hint"] = f"applied (source: {source})" + (f"; changed: {', '.join(map(str, changed))}" if changed else "; nothing changed — platform already configured this way")

            readback: httpx.Response = await client.get(thalabus.base_url + READBACK_PATH, headers=headers)
            if readback.status_code == 200:
                entry = _find_model_entry(readback.json(), reply.get("id") or payload.get("id"))
                if entry:
                    result["readback"] = {
                        "id": entry.get("id"),
                        "name": entry.get("name"),
                        "model": entry.get("model"),
                        "endpoint": entry.get("endpoint"),
                        "is_reasoning": entry.get("is_reasoning"),
                        "token_limit_kilo": entry.get("token_limit_kilo"),
                        "token_redacted": entry.get("token") in (None, ""),
                    }
            if result["readback"]:
                result["verify"]["endpoint_matches"] = result["readback"].get("endpoint") == reply.get("endpoint")
                result["verify"]["model_matches"] = result["readback"].get("model") == reply.get("model")
                if payload.get("token_limit_kilo") is not None:
                    result["verify"]["token_limit_kilo_matches"] = result["readback"].get("token_limit_kilo") == payload.get("token_limit_kilo")

    except httpx.TimeoutException:
        result["error"] = {"type": "timeout", "message": f"platform did not answer within {CONFIGURE_TIMEOUT_SECONDS:.0f}s"}
        result["hint"] = "check reachability of the configured instance"
    except (httpx.ConnectError, httpx.RequestError) as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["hint"] = f"cannot reach {thalabus.base_url} from the app container"
    except ValueError as exc:
        result["error"] = {"type": "malformed_json", "message": str(exc)}
        result["hint"] = "the platform answered this route without a JSON body"

    return result


def _find_model_entry(body: Any, want_id: Optional[str]) -> Optional[Dict[str, Any]]:
    settings: Any = (body or {}).get("llmconfiguration") or (body or {}).get("llmConfiguration") or {}
    for entry in settings.get("models") or []:
        if isinstance(entry, dict) and str(entry.get("id", "")).lower() == str(want_id or "").lower():
            return entry

    return None
