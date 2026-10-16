"""Neo4j access layer — the application's own graph store.

Everything the submission needs from the database goes through this module, and
the driver only ever appears here. The connection parameters come from the
environment, so pointing the same image at a different database — the compose
service in either plan, the platform's own Neo4j in Plan Local, or an external
one — is an env change with no code change (see ``NEO4J_URI`` in ``.env.example``).

The GUI needs ``probe_neo4j`` only; a reasoning pipeline would use
``open_driver`` and run its own queries against the same credentials.
"""

__all__ = [
    "DEFAULT_DATABASE",
    "Neo4jConfig",
    "load_neo4j_config",
    "neo4j_help_hint",
    "open_driver",
    "probe_neo4j",
    "probe_platform_database",
    "run_read",
    "run_write",
]

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_DATABASE: str = "neo4j"
DEFAULT_URI: str = "bolt://graph:7687"
DEFAULT_USER: str = "neo4j"
PROBE_TIMEOUT_SECONDS: float = 2.5

# The database the platform itself uses (its Python server is configured with this name),
# and the address it listens on inside the Plan Local stack: the overlay's database service
# is called `neo4j` and publishes no port, so the service name is the only usable address and
# the app can reach it because it joins that stack's network.
PLATFORM_DATABASE_NAME: str = "platform"
PLATFORM_NEO4J_DEFAULT_URI: str = "bolt://neo4j:7687"


@dataclass(frozen=True)
class Neo4jConfig:
    """Connection settings for one Neo4j instance (password never leaves this object)."""

    uri: str
    user: str
    password: str
    database: str

    @property
    def configured(self) -> bool:
        """True when a URI and credentials are present."""
        return bool(self.uri) and bool(self.user) and bool(self.password)

    @property
    def target(self) -> str:
        """Loggable, credential-free description of where this points."""
        return self.uri


def _classify_error(error: Exception) -> str:
    """Name one failure family for a driver exception, across driver versions.

    The 5.x and 6.x drivers renamed their exception classes — ``Neo4jAuthError``
    became ``AuthError``, ``SessionExpired`` lost its suffix — so classes are
    matched by what they are called rather than by a pinned spelling. Classifying
    wrongly is worse than not classifying: the hint the GUI shows depends on it.
    """
    lineage: List[str] = [classic.__name__ for classic in type(error).__mro__]
    joined: str = " ".join(lineage)
    message: str = str(error)

    if "Auth" in joined:
        return "auth_failed"
    if "Unauthorized" in joined:
        return "unauthorized"
    # A stopped database most often arrives as a plain ValueError about the address
    # (or as OSError, depending on driver version), so this family is matched on the
    # message: the classes differ, the thing the user must do does not.
    if "resolve address" in message or "Unable to connect" in message or "Connection refused" in message:
        return "service_unavailable"
    if "ServiceUnavail" in joined:
        return "service_unavailable"
    if "Configuration" in joined:
        return "bad_uri"

    return type(error).__name__


def _import_driver() -> Any:
    """Return the installed driver's ``GraphDatabase``.

    Imported lazily so a missing dependency surfaces as a typed status entry
    rather than an app that will not start.

    The driver is only ever imported here. This layer lives at
    ``app/graphstore/neo4j.py`` — deliberately not at ``app/neo4j.py``: the image
    runs ``python -m uvicorn`` from ``/app/app``, which puts that directory first
    on ``sys.path``, so a ``neo4j.py`` there would be imported *instead of* the
    driver package and the failure would look like a broken dependency. Keeping
    the name one package deeper gives the layer the obvious name without that
    collision.
    """
    from neo4j import GraphDatabase

    return GraphDatabase


def load_neo4j_config() -> Neo4jConfig:
    """Read the Neo4j connection settings from the environment.

    The defaults resolve to the compose service on the compose network, so a
    clean checkout with no ``.env`` still connects; a password is deliberately
    not defaulted here — it is provided by ``.env`` or the environment.
    """
    config: Neo4jConfig = Neo4jConfig(
        uri=os.environ.get("NEO4J_URI", DEFAULT_URI).strip(),
        user=os.environ.get("NEO4J_USER", DEFAULT_USER).strip() or DEFAULT_USER,
        password=os.environ.get("NEO4J_PASSWORD", "") or "",
        database=os.environ.get("NEO4J_DATABASE", DEFAULT_DATABASE).strip() or DEFAULT_DATABASE,
    )

    return config


def open_driver(config: Neo4jConfig, timeout: float = PROBE_TIMEOUT_SECONDS):
    """Open a driver for ``config``; the caller owns closing it.

    Kept separate from the probe so a pipeline can hold one driver for its
    lifetime instead of reconnecting per query.
    """
    GraphDatabase = _import_driver()
    driver = GraphDatabase.driver(
        config.uri,
        auth=(config.user, config.password),
        connection_timeout=timeout,
    )

    return driver


def _server_databases(session) -> List[str]:
    """List the databases visible to this user (empty rather than fatal)."""
    names: List[str] = []
    try:
        names = [record["name"] for record in session.run("SHOW DATABASES YIELD name RETURN name")]
    except Exception:
        names = []

    return sorted(names)


def probe_neo4j(
    config: Neo4jConfig,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Connect, run one read query, and report what happened.

    Typed outcome rather than a boolean: a refused connection, a bad password
    and a slow start-up need different actions, and the GUI shows the type
    directly. Never raises.
    """
    status: Dict[str, Any] = {
        "configured": config.configured,
        "target": config.target,
        "database": config.database,
        "reachable": False,
        "latency_ms": None,
        "server_agent": None,
        "databases": [],
        "error": None,
    }

    if not config.configured:
        status["error"] = {"type": "not_configured", "message": "NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD unset"}

        return status

    try:
        _import_driver()
    except ImportError as missing_driver:
        status["error"] = {"type": "driver_missing", "message": str(missing_driver)}

        return status

    started: float = time.perf_counter()
    driver = None
    try:
        driver = open_driver(config, timeout)
        session = driver.session(database=config.database)
        try:
            # The summary is what proves the round trip: it carries the server's own
            # identification, returned with the query result rather than guessed from
            # a handshake. (The session has no server_info in the 5.x driver.)
            summary = session.run("RETURN 1 AS ok").consume()
            databases: List[str] = _server_databases(session)
            status["server_agent"] = summary.server.agent
            status["databases"] = databases
            status["probe"] = "RETURN 1 AS ok"
            # Set last: a payload that says reachable while carrying an error is
            # worse than one that says nothing, because the GUI renders it green.
            status["reachable"] = True
        finally:
            session.close()
    except Exception as driver_error:
        status["error"] = {
            "type": _classify_error(driver_error),
            "message": str(driver_error).strip()[:400],
        }

    if driver is not None:
        driver.close()

    status["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)

    return status


def _platform_neo4j_uri() -> str:
    """The platform's database address, or the default that Plan Local's stack implies.

    The overlay's database service is called ``neo4j``, publishes no port, and sits on a network
    this app joins -- so the service name is the only address that can resolve, and the default
    needs no host knowledge at all. ``NEO4J_PLATFORM_URI`` overrides it, which is what an
    external or separately-secured store needs.
    """
    explicit: str = os.environ.get("NEO4J_PLATFORM_URI", "").strip()
    if explicit:
        return explicit

    return PLATFORM_NEO4J_DEFAULT_URI


def probe_platform_database(thalabus_plan: str,
                            password_from_app: str = "",
                            timeout: float = PROBE_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """Ask the platform's own database the same questions, and answer them separately.

    Deliberately its own function and its own payload field rather than extra flags on
    :func:`probe_neo4j`, for a reason that turned up in testing: a single ``reachable`` flag
    cannot tell apart "server up, platform database not there", and reusing the app's
    credentials against a second store would have reported the app's own Neo4j as the
    platform's whenever both happen to share a password. Every fact is therefore reported
    against the address it was collected from, and under Plan SaaS nothing is probed at all
    but the answer is still returned, so a caller never has to guess why a field is missing.
    """
    result: Dict[str, Any] = {
        "requested": thalabus_plan == "local",
        "configured": False,
        # `reachable` means a server at this address answered a query -- not that the requested
        # database could be opened, which is a different answer and is reported by `exists`.
        # Collapsing the two would call a running platform database unreachable the moment its
        # `platform` database is missing, and it would call a missing one fine the moment a
        # different server answered at the same address.
        "reachable": False,
        "uri": None,
        "user": None,
        "database": PLATFORM_DATABASE_NAME,
        "databases": [],
        "exists": False,
        "asked_via": None,
        "server_agent": None,
        "latency_ms": None,
        "error": None,
        "hint": None,
    }

    if not result["requested"]:
        result["hint"] = ("Plan SaaS does not probe this: the platform's database belongs to the "
                          "instance you were given and is not exposed over its public endpoint. "
                          "Nothing here was inferred from the submission's own store.")

        return result

    uri: str = _platform_neo4j_uri()
    user: str = os.environ.get("PLATFORM_NEO4J_USER", DEFAULT_USER).strip() or DEFAULT_USER
    password: str = os.environ.get("PLATFORM_NEO4J_PASSWORD", "") or password_from_app
    result["uri"] = uri
    result["user"] = user
    result["configured"] = bool(uri and password)

    # Guard against the one mistake that would make this number meaningless: pointing the
    # platform address at this app's own store. The answer would then be true for the wrong
    # database, so it is refused here rather than reported.
    own_uri: str = os.environ.get("NEO4J_URI", DEFAULT_URI).strip().rstrip("/")
    if uri.rstrip("/") == own_uri:
        result["configured"] = False
        result["error"] = {"type": "same_store_as_app",
                           "message": "the platform address configured here is this app's own database"}
        result["hint"] = ("Refused: probing " + uri + " twice would report the submission's store as "
                          "the platform's. Give NEO4J_PLATFORM_URI the platform's address.")

        return result

    if not result["configured"]:
        result["error"] = {"type": "not_configured",
                           "message": "no password available for the platform's database"}
        result["hint"] = ("Set PLATFORM_NEO4J_PASSWORD, or leave it unset to reuse "
                          "NEO4J_PASSWORD as Plan Local does.")

        return result

    config: Neo4jConfig = Neo4jConfig(uri=uri, user=user, password=password,
                                      database=PLATFORM_DATABASE_NAME)
    probe: Dict[str, Any] = probe_neo4j(config, timeout)
    databases: List[str] = list(probe.get("databases") or [])

    result["reachable"] = bool(probe.get("reachable"))
    result["databases"] = databases
    result["server_agent"] = probe.get("server_agent")
    result["latency_ms"] = probe.get("latency_ms")
    result["exists"] = PLATFORM_DATABASE_NAME in databases
    result["error"] = probe.get("error")

    # Opening the requested database is the strict test, and on a platform stack that has been
    # created but never started it fails for a reason that is the opposite of down: the server
    # answers with "no such database". Repeating the question against a database that always
    # exists is what tells those two apart, so the attempt is retried once through the server's
    # own default database before anything is called unreachable.
    if not result["reachable"]:
        for fallback_database in (DEFAULT_DATABASE, "system"):
            fallback: Dict[str, Any] = probe_neo4j(
                Neo4jConfig(uri=uri, user=user, password=password, database=fallback_database),
                timeout)
            if fallback.get("reachable"):
                result["reachable"] = True
                result["asked_via"] = fallback_database
                result["databases"] = list(fallback.get("databases") or [])
                result["server_agent"] = fallback.get("server_agent")
                result["latency_ms"] = fallback.get("latency_ms")
                result["exists"] = PLATFORM_DATABASE_NAME in result["databases"]
                result["error"] = {"type": "database_unavailable",
                                   "message": str((probe.get("error") or {}).get("message")
                                                  or PLATFORM_DATABASE_NAME + " could not be opened")[:400]}
                break

    if result["exists"]:
        result["hint"] = ("Reached the platform's own database: " + PLATFORM_DATABASE_NAME
                          + " is present and a query ran against it.")
    elif result["reachable"]:
        result["hint"] = ("A server answers at " + uri + ", but " + PLATFORM_DATABASE_NAME
                          + " is not among its databases (visible: "
                          + (", ".join(result["databases"]) or "none") + "), so this is not the "
                          + "platform's store yet -- the platform creates " + PLATFORM_DATABASE_NAME
                          + " when it first starts.")
    else:
        result["hint"] = ("Nothing answered at " + uri + ". The submission's own store is "
                          "unaffected and is reported separately.")

    return result


def run_write(config: Neo4jConfig,
              statement: str,
              parameters: Optional[Dict[str, Any]] = None,
              timeout: float = PROBE_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """Execute one write statement and report what changed. Never raises.

    The wizard's state node is written through here, so a second module never has
    to import the driver: one statement, one session, one typed answer. Counters
    are read from the consumed summary, because an unconsumed result has not run
    yet — write statements produce no rows to iterate.
    """
    result: Dict[str, Any] = {
        "ok": False,
        "counters": {},
        "database": config.database,
        "error": None,
    }

    if not config.configured:
        result["error"] = {"type": "not_configured", "message": "NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD unset"}

        return result

    try:
        _import_driver()
    except ImportError as missing_driver:
        result["error"] = {"type": "driver_missing", "message": str(missing_driver)}

        return result

    driver = None
    try:
        driver = open_driver(config, timeout)
        session = driver.session(database=config.database)
        try:
            summary = session.run(statement, **(parameters or {})).consume()
            counters: Dict[str, int] = {}
            for name in ("nodes_created", "nodes_deleted", "properties_set"):
                counters[name] = int(getattr(summary.counters, name, 0))
            result["counters"] = counters
            result["ok"] = True
        finally:
            session.close()
    except Exception as driver_error:
        # Same classification as the probe: a write that fails on credentials is a
        # credentials problem, not a broken statement, and the GUI says so.
        result["error"] = {
            "type": _classify_error(driver_error),
            "message": str(driver_error).strip()[:400],
        }

    if driver is not None:
        driver.close()

    return result


def run_read(config: Neo4jConfig,
             statement: str,
             parameters: Optional[Dict[str, Any]] = None,
             timeout: float = PROBE_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """Run one read statement and return its records. Never raises.

    Records are collected by iterating the result — a Result summary carries no rows,
    so consuming first would return nothing and look like an empty database. Dicts are
    materialised here so no driver type escapes this module.
    """
    result: Dict[str, Any] = {"ok": False, "records": [], "error": None}

    if not config.configured:
        result["error"] = {"type": "not_configured", "message": "NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD unset"}

        return result

    driver = None
    try:
        driver = open_driver(config, timeout)
        session = driver.session(database=config.database)
        try:
            result["records"] = [dict(record) for record in session.run(statement, **(parameters or {}))]
            result["ok"] = True
        finally:
            session.close()
    except Exception as driver_error:
        result["error"] = {
            "type": _classify_error(driver_error),
            "message": str(driver_error).strip()[:400],
        }

    if driver is not None:
        driver.close()

    return result


def neo4j_help_hint(config: Neo4jConfig, status: Dict[str, Any]) -> Optional[str]:
    """One actionable hint for the current database state."""
    if status.get("reachable"):
        return None

    if not status.get("configured"):
        return "Set NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD in .env (the .env.example defaults match this compose file)."

    error_type: str = str((status.get("error") or {}).get("type", "unknown"))
    if error_type == "driver_missing":
        return "The Neo4j driver is not installed in this image — rebuild it so requirements.txt is applied: docker compose build"
    if error_type == "bad_uri":
        return f"NEO4J_URI '{config.uri}' is not a usable connection URI — it needs a scheme, for example bolt://graph:7687"
    if error_type == "auth_failed":
        return "Credentials rejected: check NEO4J_USER/NEO4J_PASSWORD (they must match NEO4J_AUTH in docker-compose.yml)."
    if error_type in {"service_unavailable", "OSError", "ConnectionRefusedError"}:
        return f"Nothing answered at {config.target} — is the database up? From this folder: docker compose -f docker-compose.yml ps"
    if error_type == "unauthorized":
        return f"The user is not allowed to use database '{config.database}': check NEO4J_DATABASE."

    return f"Cannot reach {config.target}; see the error detail above."
