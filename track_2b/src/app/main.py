"""FastAPI application — submission spine.

Serves the demo GUI, a health probe for `make health`, and the model-test
endpoint. All LLM access goes through the env-var contract (config.py); no
other external calls are made at runtime (air-gap friendly except the LLM).
"""

import asyncio
import mimetypes
import os
import signal
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import load_llm_config, looks_like_thinking_model
from .graphstore.neo4j import (Neo4jConfig, load_neo4j_config, neo4j_help_hint, probe_neo4j,
                               probe_platform_database)
from .model_services import load_model_services_config, test_embeddings, test_reranker
from .docker_engine import (DockerEngineConfig, docker_engine_help_hint, load_docker_engine_config,
                            probe_docker_engine)
from .model_test import test_model
from .operations import render_operations_mockups
from .operations.chat_rag import RAG_MODES, answer_question
from .operations.document_categorizer import categorize_document, load_category_overview, propose_ontology_extension
from .operations.document_store import (PLATFORM_ID_PLACEHOLDER, approve_proposed_category, platform_document_ids,
                                        read_document_states, synchronize_documents)
from .operations.kg_extraction_runner import ONTOLOGY_ID, KgExtractionJob, extractable_documents
from .operations.kg_graph_view import load_graph as load_kg_graph
from .operations.kg_metamodel import load_metamodel
from .operations.local_document_ingester import LocalDocumentIngester
from .remoteconfig import apply_model_config, get_view, load_saas_defaults
from .thalabus import PLAN_LOCAL, load_thalabus_config, platform_help_hint, platform_status
from .wizard import (complete_step, goto_step, mark_engine_unverified, mark_graph_unverified, mark_graph_verified,
                     mark_llm_verified, mark_services_unverified, mark_services_verified,
                     reset_state, restore_if_available, save_state, skip_step, state_view, waive_engine_step)

APP_DIR: Path = Path(__file__).resolve().parent
SAMPLE_DIR: Path = Path(os.getenv("SAMPLE_DIR", APP_DIR.parent.parent / "data" / "sample"))
ALLOWED_SAMPLE_EXTENSIONS: set[str] = {".txt", ".md", ".pdf", ".html"}
MAX_SAMPLE_FILE_BYTES: int = 20 * 1024 * 1024
MAX_SAMPLE_FOLDER_BYTES: int = 100 * 1024 * 1024
document_ingester: LocalDocumentIngester = LocalDocumentIngester(SAMPLE_DIR)
kg_extraction: KgExtractionJob = KgExtractionJob()

app = FastAPI(title="Hack Apertus submission", version="0.1.0")

# How long a restart request waits before the process ends: long enough for the HTTP
# reply to be written and sent, short enough that the reader is not left staring at a
# spinner wondering whether anything happened.
RESTART_DELAY_SECONDS = 1.0

# Brand assets are vendored in src/app/static/img: the GUI renders its header with
# no external request (air-gap posture), so nothing is loaded from a CDN.
app.mount("/static", StaticFiles(directory=APP_DIR / "static", html=False), name="static")


class ModelTestRequest(BaseModel):
    thinking: bool | None = None
    # Sent by the GUI slider; model_test.py clamps it into the accepted band.
    max_tokens: int | None = None


class DocumentIngestionRequest(BaseModel):
    documents: list[str]


class ChatQuestionRequest(BaseModel):
    question: str
    # simple = shipped pipeline; advanced = T35 expansion behind the pair judge.
    # The dropdown can only send these two values; an unknown value is a
    # hand-crafted request and gets a 400 rather than a silently chosen tier.
    mode: str = "simple"


class CategoryRunRequest(BaseModel):
    # One file name from /api/operations/documents; both categorisation boxes run
    # the skill on a single selected document.
    document: str


class CategoryApproveRequest(BaseModel):
    # One proposed-category node id, as listed by /api/operations/categories.
    category_id: str


class DockerWaiveRequest(BaseModel):
    # Free text, stored with the waiver and shown on the breadcrumb; empty gets a
    # truthful fallback from wizard.waive_engine_step.
    reason: str = ""


class SaasModelConfigRequest(BaseModel):
    # The card is read-only: 'default' applies the SAAS_CFG_* preset server-side, so what is
    # saved is what the environment says, and the fields below only serve callers that post
    # source=custom directly to this endpoint (the GUI no longer offers custom).
    source: str = "default"
    id: str | None = None
    name: str | None = None
    model_id: str | None = None
    base_url: str | None = None
    key: str | None = None
    reasoning: bool | None = None
    # Kilotokens (the platform's unit for a model budget), not a chat completion's raw-token max_tokens.
    token_limit_kilo: int | None = None


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    config = load_llm_config()
    platform: Dict[str, Any] = await platform_status(load_thalabus_config())
    return {
        "status": "ok",
        "llm_configured": config.configured,
        "llm_name": config.name or None,
        "llm_base_url": config.base_url or None,
        "run_plan": platform["plan"],
        "platform": platform,
    }


@app.get("/api/platform/status")
async def platform() -> Dict[str, Any]:
    config = load_thalabus_config()
    result: Dict[str, Any] = await platform_status(config)
    result["hint"] = platform_help_hint(config)
    return result


@app.get("/api/neo4j/status")
async def neo4j_status() -> Dict[str, Any]:
    """Connectivity report for the graph store this app is configured to use.

    Its own endpoint rather than a field of /api/health: the compose healthcheck
    calls /api/health on a short timeout, and that must keep answering even while
    the database is still starting.

    The wizard watches the outcome: a store that answers becomes the state store,
    and step 2 cannot close without that answer.

    The answer covers two stores, reported apart. The flags above are the
    submission's own; ``platform_database`` is the platform's, probed at its own address
    with its own credentials. Under Plan SaaS that second field is present and states that
    it was not asked, rather than being omitted: a missing field invites a reader to finish
    the sentence themselves.
    """
    config: Neo4jConfig = load_neo4j_config()
    status: Dict[str, Any] = await asyncio.to_thread(probe_neo4j, config)
    status["hint"] = neo4j_help_hint(config, status)

    # Deliberately not merged into the flags above. One `reachable` cannot express "server up,
    # platform database not there", and answering the second question with the first store's
    # credentials would report the submission's own database as the platform's.
    planned: Any = load_thalabus_config()
    status["platform_database"] = await asyncio.to_thread(probe_platform_database,
                                                         planned.plan, config.password)

    if status.get("reachable"):
        mark_graph_verified()
    else:
        mark_graph_unverified()

    return status


@app.get("/api/model/config")
async def model_config() -> Dict[str, Any]:
    config = load_llm_config()
    platform = load_thalabus_config()
    services = load_model_services_config()
    return {
        "name": config.name or None,
        "base_url": config.base_url or None,
        "api_key_present": bool(config.api_key),
        "thinking_default": looks_like_thinking_model(config.name),
        # The chat test card is shown under both plans. Plan SaaS also shows the separate
        # remote-platform configuration card; its model settings do not replace this probe.
        "plan": platform.plan,
        "visible": True,
        # Retrieval services as configured for this run, for screen 3's cards. The key is
        # reported as present or absent only, never shown; Plan Local probes its in-network
        # services without one, so it reads absent there.
        "embeddings_model": services.embeddings_model,
        "reranker_model": services.reranker_model,
        "retrieval_key_present": bool(config.api_key) and platform.plan != PLAN_LOCAL,
    }


@app.post("/api/model/test")
async def model_test(request: ModelTestRequest) -> Dict[str, Any]:
    config = load_llm_config()
    thinking: bool = request.thinking if request.thinking is not None else looks_like_thinking_model(config.name)
    result: Dict[str, Any] = await test_model(config, thinking=thinking, max_tokens=request.max_tokens)
    if result.get("ok"):
        mark_llm_verified()

    return result


@app.get("/api/saas/model/config")
async def saas_model_config_view() -> Dict[str, Any]:
    """Card contents: preset (without the key), target platform, applicability."""
    return await get_view(load_thalabus_config(), load_saas_defaults())


@app.post("/api/saas/model/config")
async def saas_model_config_save(request: SaasModelConfigRequest) -> Dict[str, Any]:
    """Apply a model configuration to the remote platform, then read it back.

    The platform token never crosses this boundary in either direction: only
    the POST reply (changed fields, tokenConfigured) and the redacted read-back
    (endpoint/model, key stripped by WithoutModelTokens) are returned.
    """
    result: Dict[str, Any] = await apply_model_config(load_thalabus_config(),
                                                       load_saas_defaults(),
                                                       selection=request.source,
                                                       overrides=request.model_dump())
    return result


@app.get("/api/wizard")
async def wizard_state() -> Dict[str, Any]:
    """Where the reader stands, and what the store knows about it.

    The first call in a process tries to resume from the graph store, so a recreated
    container comes back mid-guide; later calls are free. The live plan is passed in because
    evidence does not cross plans: a step the stored plan could waive has to be asked again
    here, and ``forgotten_steps`` says that happened instead of the guide quietly rewinding.
    """
    platform: Any = load_thalabus_config()
    restore: Dict[str, Any] = await restore_if_available(load_neo4j_config(), platform.plan)
    view: Dict[str, Any] = state_view(platform, load_neo4j_config(), load_llm_config().configured)
    view["restore"] = restore
    view["persistence"]["forgotten_steps"] = restore.get("forgotten") or []
    view["persistence"]["migration_note"] = restore.get("migration") or ""
    if restore.get("forgotten"):
        view["persistence"]["forgotten_reason"] = (
            "Step " + ", ".join(str(n) for n in restore["forgotten"])
            + " was verified on the " + str(restore.get("stored_plan") or "other")
            + " plan, which does not require the test this plan does")

    return view


@app.post("/api/wizard/steps/{number}/complete")
async def wizard_step_complete(number: int) -> Dict[str, Any]:
    """Close a step. Step 2 is refused unless this process saw both core tests pass."""
    platform: Any = load_thalabus_config()
    result: Dict[str, Any] = complete_step(platform, number, load_llm_config().configured)
    if result["ok"]:
        result["persistence"] = await asyncio.to_thread(save_state, load_neo4j_config(), platform.plan)

    return result


@app.post("/api/wizard/steps/{number}/skip")
async def wizard_step_skip(number: int) -> Dict[str, Any]:
    """Step over a step this plan rules out, so the flow can continue past it."""
    platform: Any = load_thalabus_config()
    result: Dict[str, Any] = skip_step(platform, number)
    if result["ok"]:
        result["persistence"] = await asyncio.to_thread(save_state, load_neo4j_config(), platform.plan)

    return result


@app.post("/api/wizard/goto/{number}")
async def wizard_goto(number: int) -> Dict[str, Any]:
    """Follow a breadcrumb. Breadcrumbs point backwards; forward must be earned.

    Saved like a completion: the breadcrumb is where the reader chose to stand, and a
    position that survives only until the next restart is a position that was never stored.
    """
    platform: Any = load_thalabus_config()
    result: Dict[str, Any] = goto_step(platform, number)
    if result["ok"]:
        result["persistence"] = await asyncio.to_thread(save_state, load_neo4j_config(), platform.plan)

    return result


@app.post("/api/wizard/reset")
async def wizard_reset() -> Dict[str, Any]:
    """Start the guide again. The stored node is left alone, and said so."""
    platform: Any = load_thalabus_config()
    result: Dict[str, Any] = reset_state()
    result["note"] = "in-memory state cleared; a node written to the graph store is only overwritten by the next completion"

    return result


@app.post("/api/wizard/model-services")
async def wizard_model_services() -> Dict[str, Any]:
    """Probe the active plan's embeddings and reranker endpoints."""
    platform: Any = load_thalabus_config()
    result: Dict[str, Any] = {
        "ok": False,
        "applicable": True,
        "plan": platform.plan,
        "embeddings": None,
        "reranker": None,
        "error": None,
    }

    services = load_model_services_config(api_keys_enabled=platform.plan != PLAN_LOCAL)
    embeddings: Dict[str, Any] = await asyncio.to_thread(test_embeddings, services)
    reranker: Dict[str, Any] = await asyncio.to_thread(test_reranker, services)
    result["embeddings"] = embeddings
    result["reranker"] = reranker
    result["ok"] = bool(embeddings.get("ok") and reranker.get("ok"))
    if result["ok"]:
        mark_services_verified()
    else:
        mark_services_unverified()

    return result


@app.get("/api/docker/status")
async def docker_status() -> Dict[str, Any]:
    """Ask the configured Engine address who it is, and report exactly what came back.

    Read-only by plan and by construction: this handler issues no write of any kind and
    marks nothing. `docker_engine.probe_docker_engine` fetches ``/_ping`` and ``/version``
    over the same ``DOCKER_ENGINE_ADDR``/``DOCKER_ENGINE_PORT`` pair the platform itself
    is configured with -- the address the user configured to make the machine "correctly
    configured", tested at the very place that claim can fail: from inside the app's own
    network. The probe's own error kind names the failure; nothing here reinterprets it.
    """
    engine_config: DockerEngineConfig = load_docker_engine_config(load_thalabus_config().plan)
    status: Dict[str, Any] = await asyncio.to_thread(probe_docker_engine, engine_config)
    status["hint"] = docker_engine_help_hint(engine_config, status)

    return {"ok": status["ok"],
            "config": {"asked": engine_config.asked, "addr": engine_config.addr,
                       "port": engine_config.port, "base_url": status.get("requested")},
            "engine": status}


@app.post("/api/wizard/docker-test")
async def wizard_docker_test() -> Dict[str, Any]:
    """Run the Engine check and let its result decide the step's state.

    A reached Engine that identifies as Docker marks step 4; anything else takes the mark
    away again, so a reader who changes the address and retries sees the step follow the
    measurements rather than their click. Under Plan SaaS the step is ruled out by the plan
    and no request is made -- the platform's host is not this app's to probe.
    """
    platform: Any = load_thalabus_config()
    engine_config: DockerEngineConfig = load_docker_engine_config(platform.plan)
    status: Dict[str, Any] = await asyncio.to_thread(probe_docker_engine, engine_config)
    status["hint"] = docker_engine_help_hint(engine_config, status)
    result: Dict[str, Any] = {"ok": status["ok"], "applicable": engine_config.asked,
                              "plan": platform.plan,
                              "config": {"asked": engine_config.asked, "addr": engine_config.addr,
                                         "port": engine_config.port, "base_url": status.get("requested")},
                              "engine": status, "error": status.get("error")}

    if not engine_config.asked:
        return result

    if status["ok"]:
        mark_engine_verified()
    else:
        mark_engine_unverified()

    return result


@app.post("/api/wizard/docker/waive")
async def wizard_docker_waive(request: DockerWaiveRequest) -> Dict[str, Any]:
    """Move past the Engine step with the reason on record, not with a green tick.

    The step is skippable by plan under SaaS; this route is the Plan Local way out, for
    the case the submission found on its own host -- the Engine API answers from the host
    and times out from the container network because UFW is active, and opening the port
    is a root firewall edit no submission should need for a demo. The waiver is stored as
    `skipped` and shown as such; the guide never converts it into a pass.
    """
    platform: Any = load_thalabus_config()
    result: Dict[str, Any] = waive_engine_step(platform, request.reason)
    if result["ok"]:
        result["persistence"] = await asyncio.to_thread(save_state, load_neo4j_config(), platform.plan)

    return result


def _signal_self_termination() -> None:
    """Ask this process to stop, by signalling ourselves.

    SIGTERM rather than an exit call: uvicorn catches it and shuts down cleanly, which is
    what lets the reply above actually reach the client first. As the container's init
    process, our exit is the container's exit, and the restart policy turns that into a
    fresh start.
    """
    os.kill(os.getpid(), signal.SIGTERM)


@app.post("/api/admin/restart")
async def admin_restart() -> Dict[str, Any]:
    """End this process so Docker starts the container again.

    The app has no access to the Docker socket, so it cannot ask to be restarted; it can
    only stop itself, which works because the service declares a restart policy. When that
    is not true — a plain `uvicorn` on a developer machine — the call refuses instead of
    leaving the reader with a dead server and no way back.

    A restart reloads the environment as the container was created with it. Editing .env
    is a recreate, not a restart, and the GUI says so.
    """
    platform: Any = load_thalabus_config()
    pid1: bool = os.getpid() == 1
    containerised: bool = Path("/.dockerenv").exists()

    if not (pid1 and containerised):
        return {
            "ok": False,
            "planned": False,
            "error": {
                "type": "not_restartable",
                "message": "this process is not the container's init process, so ending it would leave nothing to restart it",
            },
            "note": "restart it the way you started it, for example `make run PLAN=" + platform.plan + "` or `docker compose ... restart app` on the host",
        }

    # The reply has to leave before the process does, so the exit is scheduled rather
    # than performed, and the shutdown runs on the loop that is still answering requests.
    loop = asyncio.get_running_loop()
    loop.call_later(RESTART_DELAY_SECONDS, _signal_self_termination)

    return {
        "ok": True,
        "planned": True,
        "in_seconds": RESTART_DELAY_SECONDS,
        "plan": platform.plan,
        "note": "the page will reload when the app answers again",
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "index.html")

@app.get("/api/operations/mockups", response_class=HTMLResponse)
async def operations_mockups() -> str:
    """Return the operation-box markup for the guide."""
    return render_operations_mockups()


@app.get("/api/operations/kg/metamodel")
async def operations_kg_metamodel() -> Dict[str, Any]:
    """Read-only metamodel of the hard-coded demo ontology (see operations/kg_metamodel.py)."""
    return load_metamodel()


@app.get("/api/operations/kg/graph")
async def operations_kg_graph() -> Dict[str, Any]:
    """Read-only entity graph for the Graph tab (see operations/kg_graph_view.py)."""
    return await asyncio.to_thread(load_kg_graph)


@app.get("/api/operations/kg/extraction/documents")
async def kg_extraction_documents() -> Dict[str, Any]:
    """Documents extractable right now: dossier documents in Ready/NeedsCategorization with text."""
    return await asyncio.to_thread(extractable_documents)


class KgExtractionRequest(BaseModel):
    document: str


@app.post("/api/operations/kg/extraction")
async def start_kg_extraction(request: KgExtractionRequest) -> Dict[str, Any]:
    """Start one background extraction run (in-app mirror of the KgFromDocument skill)."""
    started: Dict[str, Any] = await asyncio.to_thread(kg_extraction.start, request.document)
    if not started["ok"]:
        raise _kg_extraction_http_error(started)

    return {"started": True, "document": request.document, "ontology_id": ONTOLOGY_ID}


@app.get("/api/operations/kg/extraction/state")
async def kg_extraction_state() -> Dict[str, Any]:
    """Snapshot of the running or last finished extraction, consumed by the box poll loop."""
    return await asyncio.to_thread(kg_extraction.state_snapshot)


def _kg_extraction_http_error(started: Dict[str, Any]) -> HTTPException:
    """Translate the runner's typed start errors into the status codes the box renders."""
    codes: Dict[str, int] = {"not_configured": 503, "graph": 503, "graph_unavailable": 503,
                             "driver_missing": 503, "database_unavailable": 503,
                             "llm_not_configured": 502, "invalid_name": 404, "document_not_found": 404,
                             "not_extractable": 409, "ontology_missing": 409, "in_progress": 409}
    kind: str = str(started.get("error_type") or "failed")
    message: str = str(started.get("error") or "Extraction could not start")

    return HTTPException(status_code=codes.get(kind, 500), detail=f"{kind}: {message}")


def sample_documents() -> list[dict[str, Any]]:
    """List supported files currently stored in the sample folder."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    documents: list[dict[str, Any]] = []
    for path in sorted(SAMPLE_DIR.iterdir(), key=lambda item: item.name.casefold()):
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in ALLOWED_SAMPLE_EXTENSIONS:
            documents.append({"name": path.name, "size_bytes": path.stat().st_size})

    return documents


def tracked_sample_documents() -> list[dict[str, Any]]:
    """Return sample files with status and platform document id read from the graph."""
    documents: list[dict[str, Any]] = sample_documents()
    graph_result: Dict[str, Any] = synchronize_documents(documents)
    if not graph_result["ok"]:
        error: Dict[str, Any] = graph_result.get("error") or {}
        message: str = str(error.get("message") or "Unknown Neo4j error")
        raise RuntimeError(message)

    statuses: Dict[str, str] = graph_result["statuses"]
    states: Dict[str, Any] = {}
    states_result: Dict[str, Any] = read_document_states([document["name"] for document in documents])
    if states_result["ok"]:
        states = states_result["states"]
    for document in documents:
        state: Dict[str, Any] = states.get(document["name"]) or {}
        document["status"] = statuses.get(document["name"], str(state.get("status") or "Unknown"))
        # Why a status reads Failed, read off the node. A table that can only print "Failed"
        # makes an unreadable file indistinguishable from a broken deployment.
        document["status_error"] = str(state.get("error") or "")
        document["assigned_categories"] = int(state.get("assigned_count") or 0)
        document["proposed_categories"] = list(state.get("proposed") or [])

    # Read separately from the statuses above: this one answers a question about the
    # platform's numbering of the same file, and a file that was only ever uploaded here
    # has no answer, which is shown as `n/a` rather than reported as a failure.
    platform_result: Dict[str, Any] = platform_document_ids([document["name"] for document in documents])
    platform_ids: Dict[str, str] = platform_result.get("ids") or {}
    platform_statuses: Dict[str, str] = platform_result.get("statuses") or {}
    for document in documents:
        document["platform_document_id"] = platform_ids.get(document["name"]) or PLATFORM_ID_PLACEHOLDER
        document["platform_status"] = platform_statuses.get(document["name"]) or PLATFORM_ID_PLACEHOLDER

    return documents


@app.get("/api/operations/documents")
async def operations_documents() -> Dict[str, Any]:
    """List uploaded sample documents for the ingestion panel."""
    try:
        documents: list[dict[str, Any]] = await asyncio.to_thread(tracked_sample_documents)
    except RuntimeError as graph_error:
        raise HTTPException(status_code=503, detail=f"Could not read document status from Track 2B Neo4j: {graph_error}")

    return {"documents": documents}


@app.post("/api/operations/documents/ingest")
async def start_document_ingestion(request: DocumentIngestionRequest) -> Dict[str, Any]:
    """Start a serial Track 2B-only ingestion run for selected sample files."""
    documents: list[dict[str, Any]] = await asyncio.to_thread(sample_documents)
    document_by_name: Dict[str, dict[str, Any]] = {document["name"]: document for document in documents}
    missing: list[str] = [name for name in request.documents if name not in document_by_name]
    if missing:
        raise HTTPException(status_code=404, detail="Selected documents are not available in the sample folder.")

    selected_documents: list[dict[str, Any]] = [document_by_name[name] for name in request.documents]
    graph_result: Dict[str, Any] = await asyncio.to_thread(synchronize_documents, selected_documents)
    if not graph_result["ok"]:
        error: Dict[str, Any] = graph_result.get("error") or {}
        raise HTTPException(status_code=503, detail=f"Could not prepare Track 2B Neo4j documents: {error.get('message', 'Unknown error')}")

    started: Dict[str, Any] = await asyncio.to_thread(document_ingester.start, request.documents)
    if not started["ok"]:
        raise HTTPException(status_code=409, detail=started["error"])

    return started


@app.get("/api/operations/ingestion/progress")
async def document_ingestion_progress() -> Dict[str, Any]:
    """Return the in-process Track 2B ingestion progress snapshot."""
    progress: Dict[str, Any] = await asyncio.to_thread(document_ingester.progress)

    return progress


@app.post("/api/operations/chat")
async def operations_chat(request: ChatQuestionRequest) -> Dict[str, Any]:
    """Answer one question through the RAG pipeline: retrieve, (expand), rerank, generate.

    The pipeline itself never raises; a failed stage arrives as a typed error and is
    translated here into the matching HTTP status plus that same typed code in the body,
    so the GUI can name the stage that broke instead of guessing from the message.
    An unknown search mode is a client mistake, not a cost-tier preference: 400.
    """
    if request.mode not in RAG_MODES:
        raise HTTPException(status_code=400, detail=f"Unknown search mode '{request.mode}'. Use 'simple' or 'advanced'.")
    result: Dict[str, Any] = await asyncio.to_thread(answer_question, request.question, request.mode)
    if result["ok"]:
        return result

    error: Dict[str, Any] = result.get("error") or {}
    message: str = str(error.get("message") or "The Chat Q&A pipeline failed.")
    error_type: str = str(error.get("type") or "unknown")
    if error_type in {"missing_question", "question_too_long"}:
        raise HTTPException(status_code=400, detail=message)
    if error_type == "no_documents":
        raise HTTPException(status_code=409, detail=message)
    if error_type.startswith("store") or error_type == "store_not_configured":
        raise HTTPException(status_code=503, detail=message)

    return JSONResponse(status_code=502, content={"ok": False,
                                                  "question": result["question"],
                                                  "error": error,
                                                  "detail": message})


@app.get("/api/operations/categories")
async def operations_categories() -> Dict[str, Any]:
    """Read-only view of the ontology categories the categorisation boxes may use."""
    return await asyncio.to_thread(load_category_overview)


@app.post("/api/operations/categories/infer")
async def operations_categories_infer(request: CategoryRunRequest) -> Dict[str, Any]:
    """Run the categorisation skill on one document and keep the result as a proposal."""
    return await _run_categorization(request.document, apply=False)


@app.post("/api/operations/categories/apply")
async def operations_categories_apply(request: CategoryRunRequest) -> Dict[str, Any]:
    """Run the categorisation skill on one document and store the assignments explicitly."""
    return await _run_categorization(request.document, apply=True)


@app.post("/api/operations/categories/extend")
async def operations_categories_extend(request: CategoryRunRequest) -> Dict[str, Any]:
    """Ask the extension prompt on one document and file the answer as flagged proposals."""
    result: Dict[str, Any] = await propose_ontology_extension(request.document)
    if not result["ok"]:
        error: Dict[str, Any] = result.get("error") or {"type": "unknown", "message": "Extension failed."}
        raise HTTPException(status_code=_skill_error_status(error), detail=f"{error.get('type')}: {error.get('message')}")

    return result


@app.post("/api/operations/categories/approve")
async def operations_categories_approve(request: CategoryApproveRequest) -> Dict[str, Any]:
    """Clear the `proposed` flag on one flagged category — the explicit adoption step."""
    approved: Dict[str, Any] = await asyncio.to_thread(approve_proposed_category, request.category_id)
    if not approved["ok"]:
        error: Dict[str, Any] = approved.get("error") or {"type": "unknown", "message": "Approval failed."}
        status_code: int = 404 if str(error.get("type")) == "not_proposed" else 503
        raise HTTPException(status_code=status_code, detail=f"{error.get('type')}: {error.get('message')}")

    return approved


def _skill_error_status(error: Dict[str, Any]) -> int:
    """Translate one typed skill error into the HTTP status that names it."""
    return {
        "document_not_found": 404,
        "not_categorizable": 409,
        "ontology_missing": 409,
        "llm_not_configured": 503,
        "llm_timeout": 504,
        "not_configured": 503,
        "auth_failed": 503,
    }.get(str(error.get("type")), 502)


async def _run_categorization(name: str, apply: bool) -> Dict[str, Any]:
    """Share one skill run with an HTTP outcome that names why it could not run."""
    result: Dict[str, Any] = await categorize_document(name, apply=apply)
    if not result["ok"]:
        error: Dict[str, Any] = result.get("error") or {"type": "unknown", "message": "Categorisation failed."}
        raise HTTPException(status_code=_skill_error_status(error), detail=f"{error.get('type')}: {error.get('message')}")

    return result


@app.get("/api/operations/documents/{filename}")
async def open_operations_document(filename: str) -> FileResponse:
    """Serve a sample document inline for opening in the browser."""
    normalized_name: str = filename.replace("\\", "/")
    document_path: Path = SAMPLE_DIR / filename
    sample_root: Path = SAMPLE_DIR.resolve()
    if Path(normalized_name).name != filename or document_path.is_symlink():
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        resolved_path: Path = document_path.resolve(strict=True)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found.")

    if resolved_path.parent != sample_root or not resolved_path.is_file() or resolved_path.suffix.lower() not in ALLOWED_SAMPLE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Document not found.")

    headers: Dict[str, str] = {"X-Content-Type-Options": "nosniff"}
    if resolved_path.suffix.lower() == ".html":
        headers["Content-Security-Policy"] = "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:"
    media_type: str = mimetypes.guess_type(resolved_path.name)[0] or "application/octet-stream"

    return FileResponse(resolved_path, media_type=media_type, filename=resolved_path.name,
                        content_disposition_type="inline", headers=headers)


@app.post("/api/operations/documents")
async def upload_operations_documents(files: list[UploadFile] = File(...)) -> Dict[str, Any]:
    """Store accepted source files in the persistent sample folder."""
    if not files:
        raise HTTPException(status_code=400, detail="Select at least one document to upload.")

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    existing_paths: list[Path] = [path for path in SAMPLE_DIR.iterdir() if path.is_file()]
    total_bytes: int = sum(path.stat().st_size for path in existing_paths)
    reserved_names: set[str] = {path.name.casefold() for path in existing_paths}
    pending: list[tuple[str, bytes]] = []
    for upload in files:
        original_name: str = (upload.filename or "").replace("\\", "/")
        filename: str = Path(original_name).name
        extension: str = Path(filename).suffix.lower()
        if not filename or filename in {".", ".."} or extension not in ALLOWED_SAMPLE_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported document filename: {filename or 'empty'}")

        content: bytes = await upload.read(MAX_SAMPLE_FILE_BYTES + 1)
        if len(content) > MAX_SAMPLE_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"{filename} exceeds the 20 MB per-file limit.")
        total_bytes += len(content)
        if total_bytes > MAX_SAMPLE_FOLDER_BYTES:
            raise HTTPException(status_code=413, detail="The sample folder is limited to 100 MB.")

        stem: str = Path(filename).stem
        candidate: str = filename
        suffix_number: int = 1
        while candidate.casefold() in reserved_names:
            candidate = f"{stem} ({suffix_number}){extension}"
            suffix_number += 1
        reserved_names.add(candidate.casefold())
        pending.append((candidate, content))

    for filename, content in pending:
        destination: Path = SAMPLE_DIR / filename
        await asyncio.to_thread(destination.write_bytes, content)

    try:
        documents: list[dict[str, Any]] = await asyncio.to_thread(tracked_sample_documents)
    except RuntimeError as graph_error:
        raise HTTPException(status_code=503, detail=f"Files are stored, but Track 2B Neo4j metadata could not be saved: {graph_error}")

    return {"uploaded": [filename for filename, _ in pending], "documents": documents}
