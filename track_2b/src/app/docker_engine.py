"""Docker Engine probe -- wizard step 4.

Every container in this stack is a Docker container, the one serving this page
included, so an Engine certainly exists. The question this step answers is
narrower and actually testable: does an Engine answer over the TCP address this
app is configured with, and is what answers really the Docker API rather than
some other HTTP server that happens to own the port.

Under Plan SaaS the step is waived rather than failed. The remote platform runs
on host hardware this submission has no claim on, its Engine is not ours to
reach, and no value in `.env` could be right for every reader's instance -- the
same rule the retrieval services have for the same reason, so the GUI greys the
step out instead of testing something else and calling it the same thing.

Read-only by construction, not by restraint: the probe issues two GETs, `/_ping`
and `/version`, and the module contains no POST, PUT, PATCH or DELETE. An
unauthenticated Engine API can mount the host filesystem and start a privileged
container, so the only safe contract with it is to ask it who it is and stop.

The step can also be switched off outright: DOCKER_ENGINE_CONTROL_ENABLE defaults to
false, because a local installation does not need Docker control. Off, the probe is
never called, the screen greys its buttons, and the step is completed by continuing
rather than by a test.
"""

__all__ = [
    "DockerEngineConfig",
    "docker_engine_help_hint",
    "engine_control_enabled",
    "load_docker_engine_config",
    "probe_docker_engine",
]

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

PROBE_TIMEOUT_SECONDS = 6.0

# Deliberately the platform's own two variables, not a submission-invented pair:
# ThalabusWeb/Configuration/EnvironmentVariables.json ships exactly these names and
# ServerConfiguration.DockerEngineIPAddress/Port read them, so one value in .env
# configures the platform's container-spawning path and this probe together, and the
# two can never disagree about where the Engine is.
DEFAULT_DOCKER_ENGINE_ADDR = "host.docker.internal"
DEFAULT_DOCKER_ENGINE_PORT = 2375

# Whether this deployment drives the Engine API at all. Off by default: a local
# installation has no need for Docker control, and a screen with nothing to test is
# honest about it rather than failing a probe it never needed to run.
DEFAULT_ENGINE_CONTROL_ENABLED = False
_ENGINE_CONTROL_TRUE = {"1", "true", "yes", "on"}


def engine_control_enabled() -> bool:
    """Read DOCKER_ENGINE_CONTROL_ENABLE. Anything unrecognised, including unset, means off."""
    raw: str = (os.getenv("DOCKER_ENGINE_CONTROL_ENABLE") or "").strip().lower()

    return raw in _ENGINE_CONTROL_TRUE if raw else DEFAULT_ENGINE_CONTROL_ENABLED

# The platform's own default for the address is 127.0.0.1, which is correct for a
# process on the host and wrong for one in a container, where it is the container's
# own loopback and nothing listens there. `host.docker.internal` is the name compose
# maps to the host through `extra_hosts: host-gateway`; docker-compose.plan-local.yml
# declares that mapping for this service.

# Longest reply fragment kept when an endpoint answers with something unexpected.
_MAX_REPLY_CHARS = 240


@dataclass(frozen=True)
class DockerEngineConfig:
    """Where the Engine is expected, and whether this plan asks at all."""

    addr: str
    port: int
    plan: str
    control_enabled: bool = True

    @property
    def base_url(self) -> str:
        """The URL the two GETs are sent to, in the form the Engine documents."""
        return f"http://{self.addr}:{self.port}"

    @property
    def asked(self) -> bool:
        """False when the switch is off or the plan waives the step: no request is made or implied."""
        return self.control_enabled and self.plan != "saas"


def load_docker_engine_config(plan: str) -> DockerEngineConfig:
    """Read the engine address from the environment, with the platform's defaults.

    An unset address is not treated as "localhost" here: inside this container that
    would be a confident wrong answer, so it becomes the host gateway name instead.
    """
    addr: str = (os.getenv("DOCKER_ENGINE_ADDR") or "").strip() or DEFAULT_DOCKER_ENGINE_ADDR
    raw_port: str = (os.getenv("DOCKER_ENGINE_PORT") or "").strip()
    try:
        port: int = int(raw_port) if raw_port else DEFAULT_DOCKER_ENGINE_PORT
    except ValueError:
        # A non-numeric port is a typo in .env, and guessing a port to probe would
        # report some other Engine as the configured one.
        port = -1

    return DockerEngineConfig(addr=addr, port=port, plan=plan, control_enabled=engine_control_enabled())


def _typed_error(kind: str, message: str) -> Dict[str, Any]:
    return {"type": kind, "message": message}


def probe_docker_engine(config: DockerEngineConfig) -> Dict[str, Any]:
    """Ask the configured address who it is. Never mutates anything.

    Every key is present on every path. A step the plan waives reports `asked:
    false` with a reason rather than omitting the endpoint's subject, because an
    absent field invites a reader to finish the sentence as "broken".
    """
    result: Dict[str, Any] = {
        "asked": config.asked,
        "configured": config.port > 0,
        "requested": config.base_url if config.port > 0 else None,
        "ok": False,
        "reachable": False,
        "pinged": False,
        "is_docker_engine": False,
        "engine_version": None,
        "api_version": None,
        "min_api_version": None,
        "platform_name": None,
        "os": None,
        "arch": None,
        "latency_ms": None,
        "reply_fragment": None,
        "error": None,
    }

    if not config.asked:
        if not config.control_enabled:
            result["error"] = _typed_error(
                "control_disabled",
                "Docker Engine control is disabled for this deployment "
                "(DOCKER_ENGINE_CONTROL_ENABLE=false): the local installation does not "
                "drive the Engine, so nothing was probed",
            )
        else:
            result["error"] = _typed_error(
                "not_applicable",
                "Plan SaaS does not ask: the platform runs on host hardware this submission "
                "cannot reach, and no address in .env could be right for every instance",
            )

        return result

    if config.port <= 0:
        result["error"] = _typed_error(
            "bad_configuration",
            "DOCKER_ENGINE_PORT is not a number, so nothing was probed rather than a guess made",
        )

        return result

    started: float = time.monotonic()
    try:
        with httpx.Client(timeout=PROBE_TIMEOUT_SECONDS) as client:
            ping: httpx.Response = client.get(config.base_url + "/_ping")
            result["pinged"] = ping.status_code == 200
            result["reply_fragment"] = _clip(ping.text)
            version_response: httpx.Response = client.get(config.base_url + "/version")
            payload: Dict[str, Any] = version_response.json() if version_response.status_code == 200 else {}

    except httpx.TimeoutException:
        result["error"] = _typed_error(
            "timeout",
            f"no answer within {PROBE_TIMEOUT_SECONDS:.0f}s -- the Engine does listen on every "
            "interface, so the usual cause is a host firewall dropping inbound 2375 from the "
            "container network rather than Engine down",
        )

        return result

    except (httpx.ConnectError, httpx.RequestError) as exc:
        result["error"] = _typed_error(type(exc).__name__, str(exc)[:300])

        return result

    result["latency_ms"] = round((time.monotonic() - started) * 1000, 1)

    if not isinstance(payload, dict) or not payload:
        # Something answered HTTP, which is not the same as the Engine answering; say
        # which of the two happened instead of reporting a green light.
        result["error"] = _typed_error(
            "not_the_engine",
            f"HTTP answered but /version returned no {version_response.status_code} JSON: "
            + _clip(version_response.text),
        )

        return result

    result["reachable"] = True
    result["engine_version"] = str(payload.get("Version") or "") or None
    result["api_version"] = str(payload.get("ApiVersion") or "") or None
    result["min_api_version"] = str(payload.get("MinAPIVersion") or "") or None
    platform_block: Any = payload.get("Platform")
    if isinstance(platform_block, dict):
        result["platform_name"] = str(platform_block.get("Name") or "") or None
    result["os"] = str(payload.get("Os") or "") or None
    result["arch"] = str(payload.get("Arch") or "") or None
    result["is_docker_engine"] = bool(result["api_version"]) and str(result["platform_name"] or "").startswith("Docker")
    result["ok"] = bool(result["reachable"] and result["is_docker_engine"])
    if not result["is_docker_engine"]:
        result["error"] = _typed_error(
            "not_the_engine",
            "the address answered, and not as a Docker Engine: Platform.Name was "
            + repr(result["platform_name"]),
        )

    return result


def _clip(body: Optional[str]) -> Optional[str]:
    """Whitespace-collapsed head of a reply body, for the detail line."""
    if not body:
        return None

    return " ".join(body.split())[:_MAX_REPLY_CHARS]


def docker_engine_help_hint(config: DockerEngineConfig, status: Dict[str, Any]) -> str:
    """One sentence on what to do, in the reader's terms, for each failure shape."""
    if not config.asked:
        if not config.control_enabled:
            return "Docker Engine control is disabled for this deployment; press Continue."

        return "Not part of this plan. The platform's own engine stays the platform's concern."

    error: Dict[str, Any] = status.get("error") or {}
    kind: str = str(error.get("type") or "")

    if kind == "timeout":
        return (f"The Engine listens on tcp://0.0.0.0:{config.port} on the host, and the host firewall "
                "appears to drop it from the container network: allow that port for the Docker "
                "bridge subnet, or point DOCKER_ENGINE_ADDR at a proxy that is allowed.")

    if kind == "ConnectError":
        return ("Nothing is listening on that address from here. Check DOCKER_ENGINE_ADDR "
                "(does it resolve?) and DOCKER_ENGINE_PORT, and whether the daemon was started "
                "with a -H tcp://... flag at all.")

    if kind == "bad_configuration":
        return "Fix DOCKER_ENGINE_PORT in .env: it has to be a number."

    if kind == "not_the_engine":
        return ("Something answered on that port that is not the Docker API. A port collision, "
                "or a proxy in the way; the reply fragment above says what it was.")

    if status.get("reachable"):
        return "Reached, identified, and only ever asked for its version."

    return str(error.get("message") or "not probed")
