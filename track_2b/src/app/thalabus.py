"""Thalabus platform connection — this module *is* the run-plan switch.

Both run plans (docs/RUN_PLANS.md) use this same client and the same config
contract; the only difference between them is which endpoint the configuration
names:

    Plan Local  THALABUS_BASE_URL=http://thalabusweb:8080   (platform started by `make run`)
    Plan SaaS   THALABUS_BASE_URL=https://demo.thalabus.ai  (platform already running)

That is why one image and one `make run` target cover both plans: the licensed
submission layer is client configuration, never an embedded platform component.

Access to a Thalabus instance is a run precondition obtained by agreement with
T2B and is explicitly *not* part of the submission — see PROVENANCE.md.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

PLAN_LOCAL: str = "local"
PLAN_SAAS: str = "saas"

# Legacy plan values from the former A/B naming, normalised on read.
_LEGACY_PLAN_ALIASES: Dict[str, str] = {"a": PLAN_LOCAL, "b": PLAN_SAAS}

PLAN_DEFAULT_BASE_URL: Dict[str, str] = {
    PLAN_LOCAL: "http://thalabusweb:8080",
    PLAN_SAAS: "https://demo.thalabus.ai",
}

# Browser-facing address per plan: in Plan Local the platform is reached through the
# published port of its own container, in Plan SaaS it is the instance itself.
PLAN_DEFAULT_UI_URL: Dict[str, str] = {
    PLAN_LOCAL: "http://localhost:8080",
    PLAN_SAAS: "https://demo.thalabus.ai",
}


@dataclass(frozen=True)
class ThalabusConfig:
    plan: str
    base_url: str
    ui_url: str
    token: str
    configured: bool


def load_thalabus_config() -> ThalabusConfig:
    """Build the platform connection from the environment.

    Precedence: `make run THALABUS_BASE_URL=…` / process env, else the default
    of the selected plan (`PLAN=local` local stack, `PLAN=SaaS` remote instance).
    """
    plan: str = (os.environ.get("THALABUS_PLAN", PLAN_LOCAL).strip().lower() or PLAN_LOCAL)
    plan = _LEGACY_PLAN_ALIASES.get(plan, plan)
    if plan not in PLAN_DEFAULT_BASE_URL:
        plan = PLAN_LOCAL
    base_url: str = (os.environ.get("THALABUS_BASE_URL", "").strip().rstrip("/")
                     or PLAN_DEFAULT_BASE_URL[plan])
    ui_url: str = (os.environ.get("THALABUS_UI_URL", "").strip().rstrip("/")
                   or PLAN_DEFAULT_UI_URL[plan])
    token: str = os.environ.get("THALABUS_TOKEN", "").strip()
    configured: bool = bool(base_url and token)
    config: ThalabusConfig = ThalabusConfig(plan=plan,
                                            base_url=base_url,
                                            ui_url=ui_url,
                                            token=token,
                                            configured=configured)
    return config


async def platform_status(config: ThalabusConfig, timeout: float = 3.0) -> Dict[str, Any]:
    """Cheap reachability probe of the configured platform.

    Only the base URL is requested (no API path is assumed): the platform answers
    with an authentication redirect/401 when it is up but the token is missing,
    which is still proof that the endpoint resolved and served — enough for a
    health pill. The API calls that need a token belong to the chat pipeline.
    """
    status: Dict[str, Any] = {
        "plan": config.plan,
        "base_url": config.base_url,
        "ui_url": config.ui_url or None,
        "configured": config.configured,
        "reachable": False,
        "http_status": None,
        "error": None,
    }
    if not config.base_url:
        status["error"] = "not_configured"
        return status

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response: httpx.Response = await client.get(config.base_url)
            status["reachable"] = True
            status["http_status"] = response.status_code

    except Exception as error:
        status["error"] = type(error).__name__

    return status


def platform_help_hint(config: ThalabusConfig) -> Optional[str]:
    """Actionable hint shown beside the status when the plan cannot be used."""
    hint: Optional[str] = None
    if not config.configured:
        if config.plan == PLAN_LOCAL:
            hint = ("Plan Local: the local platform is starting (first boot runs the "
                    "database migration); the token comes with the team session.")
        else:
            hint = "Plan SaaS: set THALABUS_TOKEN — instance access by agreement with T2B."
    return hint
