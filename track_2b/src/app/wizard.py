"""Wizard flow — server-side navigation state for the first-run guide.

Six steps: welcome, configuration, additional models, Docker engine, ready,
operation. The server owns the flow so the GUI cannot be talked into skipping
verification: forward is locked until the previous step is complete, backward is
always open, step 3 probes the configured retrieval endpoints under either plan,
and the host-specific Docker Engine step is skipped under Plan SaaS.

The Docker Engine step was inserted after the first five shipped, so stored
nodes written by the old flow carry step numbers that mean something else; see
``FLOW_VERSION`` and the migration at the end of ``load_state``.

State lives in this process first. Once the graph store has been verified at
step 2, every later completion is also written to it, so a judge who reloads the
page or recreates the container resumes where they left off — see
``save_state`` and the single ``HackathonWizard`` node it merges.
"""

__all__ = [
    "STEPS",
    "WizardState",
    "complete_step",
    "goto_step",
    "load_state",
    "mark_engine_unverified",
    "mark_engine_verified",
    "mark_graph_verified",
    "mark_llm_verified",
    "reset_state",
    "restore_if_available",
    "save_state",
    "skip_step",
    "state_view",
    "waive_engine_step",
]

import asyncio
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .docker_engine import engine_control_enabled
from .graphstore.neo4j import Neo4jConfig, run_read, run_write
from .thalabus import PLAN_SAAS, ThalabusConfig

WIZARD_NODE_LABEL = "HackathonWizard"
WIZARD_NODE_KEY = "submission-guide"

# The flow gained the Docker Engine step after nodes were already in the store, so
# stored step numbers need to be read against the flow version they were written by.
FLOW_VERSION: int = 3

# One write per completion, and only after the store proved itself reachable.
SAVE_STATE_CYPHER = (
    f"MERGE (w:{WIZARD_NODE_LABEL} {{wizard_key: $key}}) "
    "SET w.current_step = $current_step, "
    "    w.completed_steps = $completed_steps, "
    "    w.skipped_steps = $skipped_steps, "
    "    w.plan = $plan, "
    "    w.graph_verified = $graph_verified, "
    "    w.llm_verified = $llm_verified, "
    "    w.services_verified = $services_verified, "
    "    w.engine_verified = $engine_verified, "
    "    w.engine_waived_under = $engine_waived_under, "
    "    w.flow_version = $flow_version, "
    "    w.updated_at = $updated_at, "
    "    w.updated_at_epoch = $updated_at_epoch"
)

LOAD_STATE_CYPHER = (
    f"MATCH (w:{WIZARD_NODE_LABEL} {{wizard_key: $key}}) "
    "RETURN w.current_step AS current_step, w.completed_steps AS completed_steps, "
    "       w.skipped_steps AS skipped_steps, w.plan AS plan, "
    "       w.graph_verified AS graph_verified, w.llm_verified AS llm_verified, "
    "       w.services_verified AS services_verified, "
    "       w.engine_verified AS engine_verified, w.engine_waived_under AS engine_waived_under, "
    "       w.flow_version AS flow_version, "
    "       w.updated_at AS updated_at"
)


@dataclass(frozen=True)
class StepDefinition:
    """Static description of one wizard step, shared by the API and the GUI."""

    number: int
    key: str
    title: str
    purpose: str
    saas_skippable: bool = False


STEPS: List[StepDefinition] = [
    StepDefinition(1, "welcome", "Welcome", "Orient the reader: what runs where, in each plan."),
    StepDefinition(2, "configuration", "Configuration & core tests",
                   "Test the LLM endpoint and this app's own graph store."),
    StepDefinition(3, "model-services", "Additional models",
                   "Test the reranker and the embeddings server."),
    StepDefinition(4, "docker-engine", "Docker Engine",
                   "Ask the configured Engine address who it is, over the same pair of "
                   "variables the platform itself uses.", saas_skippable=True),
    StepDefinition(5, "ready", "Platform ready", "Summary, and the choice to restart or move on."),
    StepDefinition(6, "operations", "Workspace",
                   "First operational screen — deliberately not implemented yet."),
]

LAST_STEP: int = len(STEPS)


@dataclass
class WizardState:
    """Where the reader stands. Mutable by design: one shared instance, guarded by a lock."""

    current_step: int = 1
    completed_steps: List[int] = None  # type: ignore[assignment]
    skipped_steps: List[int] = None  # type: ignore[assignment]
    graph_verified: bool = False
    llm_verified: bool = False
    services_verified: bool = False
    engine_verified: bool = False
    engine_waived_under: str = ""
    persisted_at: Optional[str] = None
    persistence_error: Optional[str] = None
    last_written_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.completed_steps is None:
            self.completed_steps = []
        if self.skipped_steps is None:
            self.skipped_steps = []


_LOCK = threading.Lock()
_STATE = WizardState()


def _is_saas(plan: str) -> bool:
    return plan == PLAN_SAAS


def _applicable_steps(plan: str) -> List[int]:
    """Step numbers the reader must actually complete on this plan."""
    required: List[int] = []
    for step in STEPS:
        if step.saas_skippable and _is_saas(plan):
            continue
        required.append(step.number)

    return required


def step_needs_services_proof(plan: str) -> bool:
    """Whether step 3 must be proved rather than skipped for this plan."""
    return 3 not in skipped_steps_for_plan(plan)


def skipped_steps_for_plan(plan: str) -> List[int]:
    """Steps the plan itself rules out, recorded so the breadcrumb shows why."""
    return [step.number for step in STEPS if step.saas_skippable and _is_saas(plan)]


def step_gate(plan: str, number: int) -> str:
    """Which proof, if any, must exist before this step may close.

    The gates were hardcoded to step numbers when step numbers were stable; a
    renumbering would silently move a gate onto the wrong step, so the rule now
    lives in one place and `complete_step` just honours what it is told.
    """
    if number == 3 and step_needs_services_proof(plan):
        return "requires_services_proof"

    # A deployment with DOCKER_ENGINE_CONTROL_ENABLE=false has no Engine test to pass,
    # so the step is completed by continuing; the probe itself refuses to run, and the
    # GUI says so, so the gate must match rather than strand the reader on the screen.
    if number == 4 and not _is_saas(plan) and engine_control_enabled():
        return "requires_engine_proof"

    return ""


def _is_complete(state: WizardState, number: int, plan: str) -> bool:
    if number in state.completed_steps:
        return True

    return number in skipped_steps_for_plan(plan) or number in state.skipped_steps


def _unlocked_through(state: WizardState, plan: str) -> int:
    """Highest step the reader may open right now.

    Walks the steps in order and stops at the first one that is neither complete
    nor ruled out by the plan: that one is the frontier, and nothing past it is
    reachable. Forward-only jumping would let a reader land on the operational
    screen without ever proving the database is up.
    """
    frontier: int = 1
    for number in range(1, LAST_STEP + 1):
        if _is_complete(state, number, plan):
            frontier = number + 1
        else:
            break

    return min(frontier, LAST_STEP)


def _step_view(state: WizardState, plan: str, step: StepDefinition, unlocked_through: int) -> Dict[str, Any]:
    completed: bool = _is_complete(state, step.number, plan)
    skipped_by_plan: bool = step.number in skipped_steps_for_plan(plan)
    unlocked: bool = skipped_by_plan or step.number <= unlocked_through

    if skipped_by_plan:
        status = "skipped"
        reason = "not required in this hackathon" if step.number == 4 else "Not part of this plan."
    elif step.number in _STATE.skipped_steps:
        # A waiver is displayed as skipped, never as verified: the reader chose to move on
        # past a test that did not pass, and the breadcrumb says so.
        status = "skipped"
        reason = _STATE.engine_waived_under or "Waived on this run: the test did not pass."
    elif completed:
        status = "completed"
        if step.number == 4 and not _STATE.engine_verified and not engine_control_enabled():
            # Completed, but not by a test: the deployment switched Engine control off,
            # and the summary must not borrow the wording of a proof that never happened.
            reason = "Completed without a test: Docker Engine control is disabled for this deployment."
        else:
            reason = "Verified in this run."
    elif unlocked:
        status = "current" if step.number == state.current_step else "available"
        reason = None
    else:
        status = "locked"
        reason = f"Complete step {step.number - 1} first."

    return {
        "number": step.number,
        "key": step.key,
        "title": step.title,
        "purpose": step.purpose,
        "status": status,
        "unlocked": unlocked,
        "completed": completed or skipped_by_plan,
        "skipped_by_plan": skipped_by_plan,
        "reason": reason,
    }


def state_view(thalabus: ThalabusConfig,
               neo4j: Optional[Neo4jConfig] = None,
               llm_configured: bool = False) -> Dict[str, Any]:
    """Everything the GUI needs to draw the breadcrumb and gate the buttons."""
    plan: str = thalabus.plan
    with _LOCK:
        unlocked_through: int = _unlocked_through(_STATE, plan)
        views: List[Dict[str, Any]] = [_step_view(_STATE, plan, step, unlocked_through) for step in STEPS]
        view: Dict[str, Any] = {
            "plan": plan,
            "current_step": _STATE.current_step,
            "unlocked_through": unlocked_through,
            "graph_verified": _STATE.graph_verified,
            "llm_verified": _STATE.llm_verified,
            "llm_configured": llm_configured,
            "services_verified": _STATE.services_verified,
            "engine_verified": _STATE.engine_verified,
            "engine_control_enabled": engine_control_enabled(),
            "steps": views,
            "persistence": {
                "graph_verified": _STATE.graph_verified,
                "restored_at": _STATE.persisted_at,
                "persistence_error": _STATE.persistence_error,
                "last_written_at": _STATE.last_written_at,
                "target": neo4j.target if neo4j is not None else None,
                "node": f"{WIZARD_NODE_LABEL} {{wizard_key: '{WIZARD_NODE_KEY}'}}",
                "explanation": "State lives in this process; once the graph store has answered, the same state is read from and written to it, so a reload or a recreated container resumes here.",
            },
        }

    return view


def complete_step(thalabus: ThalabusConfig, number: int, llm_configured: bool = False) -> Dict[str, Any]:
    """Mark a step done and move the reader forward. Returns the step view or a typed error."""
    result: Dict[str, Any] = {"ok": False, "step": None, "error": None}
    plan: str = thalabus.plan

    if number < 1 or number > LAST_STEP:
        result["error"] = {"type": "unknown_step", "message": f"there is no step {number} (1..{LAST_STEP})"}

        return result

    definition: StepDefinition = STEPS[number - 1]

    with _LOCK:
        unlocked_through: int = _unlocked_through(_STATE, plan)
        skipped_by_plan: bool = number in skipped_steps_for_plan(plan)
        if number > unlocked_through and not skipped_by_plan:
            result["error"] = {
                "type": "step_locked",
                "message": f"step {number} is not reachable yet — complete step {unlocked_through} first",
            }

            return result

        gate: str = step_gate(plan, number)
        if gate == "requires_services_proof" and not _STATE.services_verified:
            # Same rule as step 2, same reason: the point of the wizard is that a step
            # closes because its own test passed, not because a button was pressed.
            result["error"] = {
                "type": "not_verified",
                "message": "step 3 stays open until both the embeddings probe and the reranking probe pass",
            }

            return result

        if gate == "requires_engine_proof" and not _STATE.engine_verified:
            result["error"] = {
                "type": "not_verified",
                "message": "step 4 stays open until the Engine probe answers as a Docker Engine — "
                           "or is waived below, which records the waiver rather than a pass",
            }

            return result

        if number == 2 and not (_STATE.graph_verified and (_STATE.llm_verified or llm_configured)):
            # The model needs to be configured, not tested (a test is a plus); the graph flag is set
            # only by a probe that reached the store, which the page runs on load, so a body
            # claiming `verified` still changes nothing.
            missing: List[str] = []
            if not (_STATE.llm_verified or llm_configured):
                missing.append("no model is configured (LLM_NAME, LLM_BASE_URL, LLM_API_KEY)")
            if not _STATE.graph_verified:
                missing.append("the graph store has not answered a probe")
            result["error"] = {
                "type": "not_verified",
                "message": "step 2 stays open until: " + "; ".join(missing),
            }

            return result

        if number not in _STATE.completed_steps:
            _STATE.completed_steps.append(number)
        _STATE.current_step = min(number + 1, LAST_STEP)
        if number == 2:
            _STATE.graph_verified = True
        result["ok"] = True
        result["step"] = _step_view(_STATE, plan, definition, _unlocked_through(_STATE, plan))

    return result


def goto_step(thalabus: ThalabusConfig, number: int) -> Dict[str, Any]:
    """Move the reader to an already-unlocked step; forward jumps need the lock check."""
    result: Dict[str, Any] = {"ok": False, "step": None, "error": None}
    plan: str = thalabus.plan

    with _LOCK:
        unlocked_through: int = _unlocked_through(_STATE, plan)
        if number < 1 or number > LAST_STEP:
            result["error"] = {"type": "unknown_step", "message": f"there is no step {number} (1..{LAST_STEP})"}

            return result

        if number > unlocked_through:
            result["error"] = {
                "type": "step_locked",
                "message": f"step {number} is still locked — breadcrumbs go back, not forward",
            }

            return result

        _STATE.current_step = number
        result["ok"] = True
        result["step"] = _step_view(_STATE, plan, STEPS[number - 1], unlocked_through)

    return result


def mark_graph_verified() -> None:
    """Called when a graph probe comes back reachable, enabling persistence."""
    with _LOCK:
        _STATE.graph_verified = True


def mark_llm_verified() -> None:
    """Called when a chat completion actually came back, proving the LLM trio works."""
    with _LOCK:
        _STATE.llm_verified = True


def mark_services_verified() -> None:
    """Called when both retrieval-model probes came back correct."""
    with _LOCK:
        _STATE.services_verified = True


def mark_services_unverified() -> None:
    """A failed probe takes the step's proof away again."""
    with _LOCK:
        _STATE.services_verified = False


def mark_engine_verified() -> None:
    """Called when the Engine probe came back identified as a Docker Engine.

    A waiver cannot coexist with a passed test: whichever happened last is what the
    breadcrumb shows, so passing clears an earlier waiver.
    """
    with _LOCK:
        _STATE.engine_verified = True
        _STATE.engine_waived_under = ""
        if 4 in _STATE.skipped_steps:
            _STATE.skipped_steps.remove(4)


def mark_engine_unverified() -> None:
    """An Engine that stops answering stops holding the step open on a lie."""
    with _LOCK:
        _STATE.engine_verified = False


def waive_engine_step(thalabus: ThalabusConfig, reason: str) -> Dict[str, Any]:
    """Close the Docker step without its test, recording that is what happened.

    On this particular host the Engine API is reachable from the host and not from
    the container network -- the daemon listens, UFW is active, and port 22 proves
    the route while 2375 times out -- and the fix is a root firewall rule that no
    part of this submission should require in order to demo. A waiver keeps the
    honesty contract: it is stored and displayed as `skipped`, never as `done`, and
    the reader's own words for why are shown on the breadcrumb. Plan SaaS refuses a
    waiver because the step is already skipped there by plan, so there is nothing to
    waive.
    """
    result: Dict[str, Any] = {"ok": False, "step": None, "error": None}
    plan: str = thalabus.plan

    if _is_saas(plan):
        result["error"] = {"type": "not_applicable",
                           "message": "on Plan SaaS the step is ruled out by the plan, not waived by the reader"}

        return result

    note: str = (reason or "").strip()[:200]
    with _LOCK:
        if 4 not in _STATE.skipped_steps:
            _STATE.skipped_steps.append(4)
        _STATE.engine_verified = False
        _STATE.engine_waived_under = ("Waived on this run: " + note) if note else \
            "Waived on this run: the reader chose to move on past a test that did not pass."
        _STATE.current_step = min(5, LAST_STEP)
        result["ok"] = True
        result["step"] = _step_view(_STATE, plan, STEPS[3], _unlocked_through(_STATE, plan))

    return result


def mark_graph_unverified() -> None:
    """A store that stops answering stops being trusted as the state store."""
    with _LOCK:
        _STATE.graph_verified = False


def skip_step(thalabus: ThalabusConfig, number: int) -> Dict[str, Any]:
    """Skip a plan-inapplicable step, so the flow continues past it."""
    result: Dict[str, Any] = {"ok": False, "step": None, "error": None}
    plan: str = thalabus.plan

    if number not in skipped_steps_for_plan(plan):
        result["error"] = {"type": "not_skippable", "message": f"step {number} cannot be skipped on plan '{plan}'"}

        return result

    with _LOCK:
        if number not in _STATE.skipped_steps:
            _STATE.skipped_steps.append(number)
        _STATE.current_step = min(number + 1, LAST_STEP)
        result["ok"] = True
        result["step"] = _step_view(_STATE, plan, STEPS[number - 1], _unlocked_through(_STATE, plan))

    return result


def reset_state() -> Dict[str, Any]:
    """Start the guide over, keeping nothing but the knowledge that it can be redone."""
    global _STATE
    with _LOCK:
        _STATE = WizardState()

    return {"ok": True, "reset_at": _now_iso()}


def save_state(neo4j: Neo4jConfig, plan: str) -> Dict[str, Any]:
    """Persist the wizard node; only meaningful once the store is verified.

    Written on every completion rather than on close, because a browser tab can be
    closed and a container recreated without anyone saying goodbye.
    """
    result: Dict[str, Any] = {"written": False, "error": None}

    with _LOCK:
        if not _STATE.graph_verified:
            result["error"] = {"type": "graph_not_verified", "message": "the graph store has not been verified yet"}

            return result

        timestamp: str = _now_iso()
        write: Dict[str, Any] = run_write(neo4j, SAVE_STATE_CYPHER, {
            "key": WIZARD_NODE_KEY,
            "current_step": _STATE.current_step,
            "completed_steps": sorted(_STATE.completed_steps),
            "skipped_steps": sorted(_STATE.skipped_steps),
            "plan": plan,
            "graph_verified": _STATE.graph_verified,
            "llm_verified": _STATE.llm_verified,
            "services_verified": _STATE.services_verified,
            "engine_verified": _STATE.engine_verified,
            "engine_waived_under": _STATE.engine_waived_under,
            "flow_version": FLOW_VERSION,
            "updated_at": timestamp,
            "updated_at_epoch": int(time.time()),
        })

        if write["ok"]:
            _STATE.last_written_at = timestamp
            _STATE.persistence_error = None
            result["written"] = True
            result["counters"] = write.get("counters")
        else:
            _STATE.persistence_error = str((write.get("error") or {}).get("type", "write_failed"))
            result["error"] = write.get("error")

    return result


async def load_state(neo4j: Neo4jConfig, plan: str) -> Dict[str, Any]:
    """Read back the wizard node, so a recreated container resumes the guide.

    An unreadable store is reported rather than guessed at: resuming from a wrong step
    is worse than starting again, and the GUI shows the reason either way.

    Evidence does not travel between plans. The node records the plan it was written on, and a
    step that plan allowed to be waived cannot read as verified on a plan that requires it --
    otherwise switching plans would mint a completed test that was never run. Such a step is
    dropped here, and the number of dropped steps is returned so the GUI can say what it forgot
    and why, rather than silently rewinding.
    """
    result: Dict[str, Any] = {"loaded": False, "error": None, "forgotten": []}
    read: Dict[str, Any] = await asyncio.to_thread(run_read, neo4j, LOAD_STATE_CYPHER, {"key": WIZARD_NODE_KEY})

    if not read["ok"]:
        result["error"] = read.get("error")

        return result

    records: List[Dict[str, Any]] = read.get("records") or []
    if not records:
        result["error"] = {"type": "no_state", "message": "no wizard node in this store yet"}

        return result

    row: Dict[str, Any] = records[0]
    stored_plan: str = str(row.get("plan") or "")
    waived: List[int] = []
    with _LOCK:
        _STATE.current_step = int(row.get("current_step") or 1)
        completed: List[int] = [int(n) for n in (row.get("completed_steps") or [])]
        _STATE.skipped_steps = [int(n) for n in (row.get("skipped_steps") or [])]
        _STATE.graph_verified = bool(row.get("graph_verified"))
        _STATE.llm_verified = bool(row.get("llm_verified"))
        _STATE.services_verified = bool(row.get("services_verified"))
        _STATE.persisted_at = row.get("updated_at")

        # A node written before FLOW_VERSION 2 counts steps the old way, when step 4 was
        # "Platform ready" and there was no Engine test at all. Trusting those numbers would
        # let an old "ready" click read as a passed Docker test, so the stored flow is
        # reinterpreted instead: the old record keeps the steps its numbers still mean the
        # same for, its Operations completion moves to 6, its ungated "ready" is dropped --
        # it recorded no test -- and the reader lands on the new step because that test has
        # genuinely never run here. The reason is returned, not silently applied.
        migration: str = ""
        stored_flow: int = int(row.get("flow_version") or 0)
        _STATE.engine_verified = bool(row.get("engine_verified"))
        _STATE.engine_waived_under = str(row.get("engine_waived_under") or "")
        if stored_flow < 2:
            if 4 in completed:
                completed.remove(4)
            completed = [6 if n == 5 else n for n in completed]
            _STATE.skipped_steps = [n for n in _STATE.skipped_steps if n != 4]
            _STATE.engine_verified = False
            migration = ("Step 4 (Docker Engine) was added to the flow after this progress was stored. "
                         "The stored steps were re-read under the new numbering and the guide resumes "
                         "at the new step, because its test has not run yet.")
            _STATE.current_step = min(_STATE.current_step, _unlocked_through(_STATE, plan) or 1)

        if stored_flow < FLOW_VERSION and _is_saas(stored_plan) and 3 in completed and not _STATE.services_verified:
            completed.remove(3)
            _STATE.current_step = min(_STATE.current_step, 3)
            migration = ("Step 3 (Additional models) now requires its own SaaS embeddings and reranker probes. "
                         "Previous SaaS progress skipped those tests, so the guide resumes at step 3.")

        if stored_plan and stored_plan != plan:
            _STATE.services_verified = False
            if 3 in completed:
                completed.remove(3)
                waived.append(3)
            for definition in STEPS:
                # Only a step the stored plan could waive is suspect. Steps 1 and 2 test the same
                # thing under both plans, so their evidence carries over unchanged.
                if not definition.saas_skippable or definition.number not in completed:
                    continue
                if _is_saas(stored_plan) and not step_needs_services_proof(plan):
                    continue
                completed.remove(definition.number)
                waived.append(definition.number)
                if definition.number == 3:
                    _STATE.services_verified = False

        _STATE.completed_steps = completed
        # Rewind to the earliest step that has to be redone. Leaving the reader on a later
        # screen would mean the app is displaying a screen the frontier no longer permits --
        # most likely exactly when they return after switching plans.
        if waived and _STATE.current_step > min(waived):
            _STATE.current_step = min(waived)

        # And the mirror image of that rule, which is the one that strands a finished run: the
        # stored pointer can lag behind the steps the same node already lists as completed,
        # because it moves when a step closes and the reader then steps back to re-read an
        # earlier screen. `completed_steps=[1,2,3,4,5], current_step=2` is a guide that is
        # finished and was browsing backwards, not one that has to start over, so it resumes at
        # the frontier its own evidence permits rather than at the pointer.
        frontier: int = _unlocked_through(_STATE, plan)
        if _STATE.current_step < frontier:
            _STATE.current_step = frontier

    result["loaded"] = True
    result["forgotten"] = waived
    result["migration"] = migration
    result["plan_switched"] = bool(stored_plan and stored_plan != plan)
    result["stored_plan"] = stored_plan or None
    result["state"] = {"current_step": _STATE.current_step,
                       "completed_steps": sorted(_STATE.completed_steps),
                       "skipped_steps": sorted(_STATE.skipped_steps),
                       "updated_at": _STATE.persisted_at}

    return result


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_RESTORE_ATTEMPTED: bool = False


async def restore_if_available(neo4j: Neo4jConfig, plan: str) -> Dict[str, Any]:
    """Try once per process to resume from the store, and say what happened.

    Attempted before the probe, because the point of persisting is that a recreated
    container comes back mid-flow: waiting for a verified probe would mean the reader
    had to re-verify in order to be allowed to resume. A store that is not there simply
    reports `no_state` and the guide starts at the beginning.
    """
    global _RESTORE_ATTEMPTED

    result: Dict[str, Any] = {"attempted": False, "loaded": False, "error": None,
                              "forgotten": [], "migration": "", "plan_switched": False, "stored_plan": None}

    with _LOCK:
        if _RESTORE_ATTEMPTED:
            result["error"] = {"type": "already_attempted", "message": "restore runs once per app process"}

            return result

        _RESTORE_ATTEMPTED = True

    result["attempted"] = True
    loaded: Dict[str, Any] = await load_state(neo4j, plan)
    result["loaded"] = loaded["loaded"]
    result["error"] = loaded.get("error")
    result["state"] = loaded.get("state")
    result["forgotten"] = loaded.get("forgotten") or []
    result["migration"] = loaded.get("migration") or ""
    result["plan_switched"] = bool(loaded.get("plan_switched"))
    result["stored_plan"] = loaded.get("stored_plan")

    return result
