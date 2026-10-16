"""KgFromDocument re-implemented on the submission's own two legs.

Same algorithm as the platform skill (``KgFromDocument_llm.py`` /
``KgFromDocument.py``: entity discovery, LLM-assisted deduplication,
ontology-gated relationships — same prompts, adapted).
This module runs on the OpenAI-compatible endpoint of the env-var trio alone and
reads and writes only this app's Neo4j store, so the extraction box works under
both run plans and with no Thalabus instance anywhere.

Persistence mirrors the platform's shapes so the extracted graph is readable by
platform-side queries too:

* entity nodes  : ``MERGE (e:Entity:{Label} {id})`` with ``e.name`` a list of
                  the name variants the text used (as the platform stores it);
* community     : one ``(:Community:Dataset {id})`` per ontology+document,
                  linked ``(:Ontology)-[:DEFINES]->()`` when the ontology node
                  exists in this store, with AGGREGATES edges to its entities;
* document link : every created entity gets ``(:Entity)-[:SOURCE]->(:Document)``
                  and ``(:Entity)-[:IS_DOCUMENTED {kind:'extracted'}]->(:Document)``
                  (T27's machine-managed edge, with kind/created_by/created_at);
* relations     : ``MATCH (a) MATCH (b) MERGE (a)-[r:TYPE]->(b) SET r.source_id``,
                  with the ontology's source/target label constraints enforced
                  here before the write (the platform's editor does the same).

The ontology is ``Parlament``, hard-coded by design exactly as in the other
operation boxes (see ``CATEGORY_ONTOLOGY_ID`` in ``document_store``); its
declarations are read live from this store when present and fall back to the
frozen snapshot shipped in ``kg_metamodel``.

The box runs the pipeline as an in-process job (one at a time, like the
ingestion job): :func:`start_extraction` launches the thread,
:func:`extraction_state` reports its progress and last result.
"""

__all__ = ["ONTOLOGY_ID", "extractable_documents", "KgExtractionJob"]

import json
import re
import threading
import time
from concurrent import futures
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..config import load_llm_config
from ..graphstore.neo4j import load_neo4j_config, run_read, run_write
from .document_store import read_document_by_id
from .kg_metamodel import label_list, load_metamodel

ONTOLOGY_ID: str = "Parlament"

# The dossier whose documents feed the box's dropdown: every document the
# platform links to it with (:Dossier)-[:CONTAINS]->(:Document) is offered, as
# long as its status is Ready/NeedsCategorization and it has extracted text.
SOURCE_DOSSIER_ID: str = "hackathon-temp"

# Same two ingestion statuses the categoriser accepts (the platform's own pair):
# a document is extractable once it has text, whether or not it has been
# categorised yet.
ALLOWED_SOURCE_STATUSES: List[str] = ["Ready", "NeedsCategorization"]

# Chunking mirrors the skill: accumulate top-level elements into chunks of at
# most this many characters; oversized blocks are flattened to their lines.
# In this app's store the ingester keeps plain text, so paragraphs are rebuilt
# as <p id="para-N"> elements — the ids what the LLM's source_id references point
# into for provenance (as they do on the platform, where documents are HTML).
CHUNK_MAX_CHARS: int = 3000
CHUNK_ELEMENT_MAX_CHARS: int = 1200
MAX_DOCUMENT_CHARS: int = 200_000
MAX_CHUNKS: int = 40

LLM_TIMEOUT_SECONDS: float = 180.0
MAX_PARALLELISM: int = 4
# Hard cap on concurrent chat requests across every pooled worker at once. Type
# fan-out nests chunk fan-out, so without this gate the in-flight count could
# reach MAX_PARALLELISM × MAX_PARALLELISM; the gateway keeps it at this number.
LLM_MAX_CONCURRENCY: int = 4

# How far the pipeline goes: 1 entities, 2 +entity attributes, 3 +relationships,
# 4 +relationship attributes (the skill's KGExtractionPhase). The demo stops at
# relationships: attribute sweeps are one LLM call per attribute × chunk and
# would lengthen the demo without adding visible structure. Raise to 4 to run
# the full phase set the skill runs by default.
STOP_AFTER_PHASE: int = 3

DOCUMENT_CONTEXT_MAX_CHARS: int = 8000

ENTITY_MAX_TOKENS: int = 8000
VALUES_MAX_TOKENS: int = 8000
RELATIONSHIPS_MAX_TOKENS: int = 4000
REASONING_ESCALATION_TOKENS: int = 16000

_LABEL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_PATTERN = re.compile(r"```(?:json|ya?ml)?\s*(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# Extractable documents (the box's dropdown)
# ---------------------------------------------------------------------------

_EXTRACTABLE_QUERY: str = (
    "MATCH (:Dossier {id: $dossier_id})-[:CONTAINS]->(document:Document) "
    "WHERE coalesce(document.status, document.Status) IN $statuses "
    "  AND NOT coalesce(document.text, document.Text) IS NULL "
    "RETURN document.id AS id, "
    "       coalesce(document.name, document.FileName, document.id) AS name, "
    "       coalesce(document.status, document.Status) AS status, "
    "       size(coalesce(document.text, document.Text)) AS chars "
    "ORDER BY name, id"
)


def extractable_documents() -> Dict[str, Any]:
    """List dossier documents that can be extracted: Ready or NeedsCategorization, with text."""
    listing: Dict[str, Any] = {"ok": False, "ontology_id": ONTOLOGY_ID, "documents": [], "error": None}
    config = load_neo4j_config()
    if not config.configured:
        listing["error"] = {"type": "not_configured",
                            "message": "Track 2B Neo4j is not configured."}

        return listing

    read_result: Dict[str, Any] = run_read(
        config, _EXTRACTABLE_QUERY,
        {"statuses": ALLOWED_SOURCE_STATUSES, "dossier_id": SOURCE_DOSSIER_ID},
        timeout=10.0,
    )
    if not read_result["ok"]:
        listing["error"] = read_result["error"]

        return listing

    listing["ok"] = True
    listing["documents"] = [
        {"id": str(record["id"]), "name": str(record["name"]),
         "status": str(record["status"]), "chars": int(record["chars"] or 0)}
        for record in read_result["records"]
    ]

    return listing


# ---------------------------------------------------------------------------
# The job (one extraction at a time, like the ingestion job)
# ---------------------------------------------------------------------------

class KgExtractionJob:
    """Runs the extraction pipeline for one document in a worker thread."""

    def __init__(self) -> None:
        self.lock: threading.Lock = threading.Lock()
        self.state: Dict[str, Any] = _idle_state()

    def start(self, name: str) -> Dict[str, Any]:
        """Validate the request, then launch the pipeline; the payload never raises."""
        started: Dict[str, Any] = {"ok": False, "error": None, "error_type": None}
        prepared: Dict[str, Any] = _prepare(name)
        if not prepared["ok"]:
            started["error"] = str((prepared.get("error") or {}).get("message") or "Extraction cannot start.")
            started["error_type"] = str((prepared.get("error") or {}).get("type") or "failed")

            return started

        with self.lock:
            if self.state["running"]:
                started["error"] = "An extraction is already in progress."
                started["error_type"] = "in_progress"

                return started

            self.state = {
                "running": True,
                "document": name,
                "phase": "starting",
                "percent": 0,
                "error": None,
                "result": None,
            }
            worker: threading.Thread = threading.Thread(
                target=self._run, args=(prepared,), daemon=True, name="kg-extraction")
            worker.start()
            started["ok"] = True
            started["document"] = name

        return started

    def state_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            snapshot: Dict[str, Any] = dict(self.state)

        return snapshot

    def _run(self, prepared: Dict[str, Any]) -> None:
        extraction: Dict[str, Any] = _extract_all(prepared, self._progress)
        finished: Dict[str, Any]
        if extraction["ok"]:
            finished = {"running": False, "document": prepared["name"], "phase": "done",
                        "percent": extraction["percent"], "error": None,
                        "result": extraction.get("result")}
        else:
            finished = {"running": False, "document": prepared["name"],
                        "phase": str((extraction.get("error") or {}).get("type") or "failed"),
                        "percent": extraction["percent"],
                        "error": extraction.get("error"), "result": None}

        with self.lock:
            self.state = finished

    def _progress(self, phase: str, percent: int) -> None:
        with self.lock:
            self.state = {**self.state, "phase": phase, "percent": percent}


def _idle_state() -> Dict[str, Any]:
    return {"running": False, "document": None, "phase": "idle",
            "percent": 0, "error": None, "result": None}


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

_DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


def _prepare(document_id: str) -> Dict[str, Any]:
    """Load document + declarations and check every precondition, with typed errors."""
    prepared: Dict[str, Any] = {"ok": False, "name": document_id, "document": None,
                                "elements": [], "ontology_present": False, "error": None}
    if not _DOCUMENT_ID_PATTERN.match(document_id or ""):
        prepared["error"] = {"type": "invalid_name", "message": "Not a known document id."}

        return prepared

    stored: Dict[str, Any] = read_document_by_id(document_id)
    if not stored["ok"]:
        prepared["error"] = stored["error"] or {"type": "graph_unavailable",
                                                "message": "Could not read the document."}

        return prepared

    document: Dict[str, Any] = stored["document"]
    status: str = str(document.get("status") or "Unknown")
    text: str = str(document.get("text") or "")
    if status not in ALLOWED_SOURCE_STATUSES:
        prepared["error"] = {"type": "not_extractable",
                             "message": f"Status '{status}' cannot be extracted — ingest the document first."}

        return prepared

    if not text.strip():
        prepared["error"] = {"type": "not_extractable",
                             "message": "No extracted text: ingest the document before extracting from it."}

        return prepared

    llm = load_llm_config()
    if not llm.configured:
        prepared["error"] = {"type": "llm_not_configured",
                             "message": "LLM_NAME / LLM_BASE_URL / LLM_API_KEY are not all set."}

        return prepared

    elements: List[Dict[str, Any]] = _read_local_elements()
    ontology_present: bool = bool(elements)
    if not elements:
        snapshot: Dict[str, Any] = load_metamodel()
        elements = [dict(element) for element in snapshot.get("elements") or []]
        prepared["note"] = str(snapshot.get("note") or "")

    entities: List[Dict[str, Any]] = [e for e in elements if e.get("kind") == "Entity" and e.get("name")]
    relationships: List[Dict[str, Any]] = [r for r in elements if r.get("kind") == "Relationship" and r.get("name")]
    if not entities and not relationships:
        prepared["error"] = {"type": "ontology_missing",
                             "message": f"Ontology '{ONTOLOGY_ID}' defines no entity or relationship types here."}

        return prepared

    prepared["ok"] = True
    prepared["document"] = document
    prepared["name"] = str(document.get("name") or document_id)
    prepared["elements"] = elements
    prepared["ontology_present"] = ontology_present

    return prepared


_ELEMENTS_QUERY: str = (
    "MATCH (o:Ontology {id: $ontology_id})-[:CONTAINS]->(e) "
    "WHERE e:EntityOntology OR e:RelationshipOntology "
    "RETURN CASE WHEN e:EntityOntology THEN 'Entity' ELSE 'Relationship' END AS kind, "
    "       last(labels(e)) AS name, "
    "       coalesce(e.entity_description, '') AS description, "
    "       coalesce(e.entity_source_labels, []) AS source_labels, "
    "       coalesce(e.entity_target_labels, []) AS target_labels, "
    "       [ (e)-[:HAS_ATTRIBUTE]->(a:OntologyAttribute) | "
    "           { name: a.name, type: a.type, description: coalesce(a.value, '') } ] AS attributes"
)


def _read_local_elements() -> List[Dict[str, Any]]:
    """The ontology as declared in this app's own store — the one the writes hit."""
    elements: List[Dict[str, Any]] = []
    read_result: Dict[str, Any] = run_read(load_neo4j_config(), _ELEMENTS_QUERY,
                                           {"ontology_id": ONTOLOGY_ID}, timeout=10.0)
    if read_result.get("ok"):
        for record in read_result.get("records") or []:
            elements.append({
                "kind": record.get("kind") or "",
                "name": record.get("name") or "",
                "description": record.get("description") or "",
                "source_labels": label_list(record.get("source_labels")),
                "target_labels": label_list(record.get("target_labels")),
                "attributes": [dict(a) for a in (record.get("attributes") or []) if a],
            })

    return elements


# ---------------------------------------------------------------------------
# The pipeline (mirrors KgFromDocument.execute)
# ---------------------------------------------------------------------------

def _extract_all(prepared: Dict[str, Any], progress: Any) -> Dict[str, Any]:
    """Run the full extraction for one prepared document; returns a typed payload."""
    run: Dict[str, Any] = {"ok": False, "percent": 5, "error": None, "result": None, "notes": []}
    document: Dict[str, Any] = prepared["document"] or {}
    text: str = _plain_text(str(document.get("text") or ""))[:MAX_DOCUMENT_CHARS]
    elements: List[Dict[str, Any]] = prepared["elements"]
    ontology_present: bool = bool(prepared.get("ontology_present"))
    entities: List[Dict[str, Any]] = [e for e in elements if e.get("kind") == "Entity" and e.get("name")]
    relationships: List[Dict[str, Any]] = [r for r in elements if r.get("kind") == "Relationship" and r.get("name")]
    document_id: str = str(document.get("id") or prepared.get("name") or "")
    document_context: str = text[:DOCUMENT_CONTEXT_MAX_CHARS]
    ontology_description: str = _describe_ontology(entities, relationships)
    notes: List[str] = []
    if prepared.get("note"):
        notes.append(str(prepared["note"]))
    if not ontology_present:
        notes.append(f"The Ontology node '{ONTOLOGY_ID}' is not present in this store — the "
                     "extraction used the declarations shipped with this submission and skipped the DEFINES link.")

    client: Optional[_Llm] = _Llm.create(progress_notes=notes)
    if client is None:
        run["error"] = {"type": "llm_not_configured",
                        "message": "LLM_NAME / LLM_BASE_URL / LLM_API_KEY are not all set."}

        return run

    try:
        progress("chunking document", 5)
        chunks: List[str] = _chunk_plain_text(text, notes)

        community_id: str = _ensure_community(document_id, ontology_present)
        progress("created community", 10)

        # Phase 1: entities, the declared types fanned out in parallel (chunk
        # fan-out nests inside each type; _Llm.complete gates the in-flight
        # requests; one failing type is noted, it no longer aborts the run).
        all_entities: Dict[str, Dict[str, Any]] = {}
        entity_tasks: List[Dict[str, Any]] = []
        for entity in entities:
            label: str = str(entity["name"])
            if _safe_label(label):
                entity_tasks.append(entity)
            else:
                notes.append(f"Skipped entity type with an unusable label: {label!r}")

        entity_work: Any = _make_entity_work(client, chunks, document_context, ontology_description,
                                             community_id, document_id)
        for _type_label, found in _run_declarations_in_parallel(client, entity_tasks, entity_work,
                                                                progress, "entity types", 10, 60):
            if found is None:
                continue

            for entity_id, candidate in found.items():
                existing_candidate: Optional[Dict[str, Any]] = all_entities.get(entity_id)
                if existing_candidate is None:
                    all_entities[entity_id] = {**candidate, "labels": [candidate["label"]]}
                else:
                    labels: set = set(existing_candidate.get("labels") or [existing_candidate["label"]])
                    labels.add(candidate["label"])
                    existing_candidate["labels"] = sorted(labels)
                    for name in candidate.get("names") or []:
                        if name not in existing_candidate.get("names", []):
                            existing_candidate.setdefault("names", []).append(name)

        if STOP_AFTER_PHASE < 3:
            entities_only: Dict[str, Any] = {"ok": True, "percent": int(95), "error": None, "notes": notes}
            entities_only["result"] = _result_payload(prepared, community_id, all_entities, [], notes)

            return entities_only

        # Phase 3: relationships, the declared types fanned out in parallel too.
        all_relationships: List[Dict[str, str]] = []
        relationship_tasks: List[Dict[str, Any]] = []
        for relationship in relationships:
            rel_label: str = str(relationship["name"])
            if _safe_label(rel_label):
                relationship_tasks.append(relationship)
            else:
                notes.append(f"Skipped relationship type with an unusable label: {rel_label!r}")

        relationship_work: Any = _make_relationship_work(client, chunks, all_entities,
                                                         document_context, ontology_description)
        for _rel_label, created in _run_declarations_in_parallel(client, relationship_tasks, relationship_work,
                                                                 progress, "relationship types", 70, 30):
            if created:
                all_relationships.extend(created)

        progress("persisted community", 100)
        run["ok"] = True
        run["percent"] = 100
        run["notes"] = notes
        run["result"] = _result_payload(prepared, community_id, all_entities, all_relationships, notes)

    except Exception as failure:  # noqa: BLE001 — a pipeline thread has no one to raise to
        run["error"] = {"type": type(failure).__name__, "message": str(failure)[:600]}

    finally:
        client.close()


    return run


def _result_payload(prepared: Dict[str, Any], community_id: str,
                    all_entities: Dict[str, Dict[str, Any]],
                    all_relationships: List[Dict[str, str]],
                    notes: List[str]) -> Dict[str, Any]:
    """The box's preview and persistence evidence, in one payload."""
    payload: Dict[str, Any] = {
        "ontology_id": ONTOLOGY_ID,
        "document": str(prepared.get("name") or ""),
        "community_id": community_id,
        "entity_count": len(all_entities),
        "relationship_count": len(all_relationships),
        "entities_by_type": {},
        "entities": sorted(
            [{"id": e["id"], "label": e["label"], "name": e["name"]} for e in all_entities.values()],
            key=lambda item: (item["label"], item["name"].lower())),
        "relationships": all_relationships,
        "notes": notes,
    }
    by_type: Dict[str, List[str]] = {}
    for entity in payload["entities"]:
        by_type.setdefault(str(entity["label"]), []).append(str(entity["name"]))
    payload["entities_by_type"] = by_type

    return payload


# ---------------------------------------------------------------------------
# Ontology description and community
# ---------------------------------------------------------------------------

def _describe_ontology(entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> str:
    """A compact schema dump injected into the prompts (the skill does the same)."""
    lines: List[str] = []
    for entity in entities:
        attributes: List[str] = [a.get("name", "") for a in entity.get("attributes") or [] if a.get("name")]
        lines.append(f"- {entity['name']} (entity): {entity.get('description') or ''}"
                     + (f" | attributes: {', '.join(attributes)}" if attributes else ""))
    for relationship in relationships:
        lines.append(f"- {relationship['name']} (relationship, "
                     f"{_labels_text(relationship.get('source_labels'))} -> "
                     f"{_labels_text(relationship.get('target_labels'))}): "
                     f"{relationship.get('description') or ''}")

    return "\n".join(lines)


def _labels_text(labels: Any) -> str:
    values: List[str] = [str(label) for label in (labels or []) if label]

    return ", ".join(values) if values else "any"


_COMMUNITY_IDS_QUERY: str = (
    "MATCH (c:Community) WHERE c.id = 'DefaultCommunity_' + $ontology_id "
    "   OR $ontology_id IN coalesce(c.ontologies, []) "
    "   OR EXISTS { MATCH (o:Ontology {id: $ontology_id})-[:DEFINES]->(c) } "
    "RETURN c.id AS id"
)


def _ensure_community(document_id: str, ontology_present: bool) -> str:
    """Resolve and ensure the extraction's community, like the skill's fallback."""
    community_id: str = ""
    read_result: Dict[str, Any] = run_read(load_neo4j_config(), _COMMUNITY_IDS_QUERY,
                                           {"ontology_id": ONTOLOGY_ID}, timeout=10.0)
    ids: List[str] = [str(record["id"]) for record in read_result.get("records") or [] if record.get("id")]
    if ids:
        community_id = next((cid for cid in ids if cid.upper() == "DEFAULT"),
                            next((cid for cid in ids if cid == f"DefaultCommunity_{ONTOLOGY_ID}"), ids[0]))
    if not community_id:
        community_id = f"DefaultCommunity_{ONTOLOGY_ID}"

    run_write(load_neo4j_config(),
              "MERGE (c:Community:Dataset {id: $community_id}) "
              "ON CREATE SET c.description = $description, c.created_at = datetime() "
              "ON MATCH SET c.updated_at = datetime()",
              {"community_id": community_id,
               "description": f"KG extracted from document '{document_id}' using ontology '{ONTOLOGY_ID}'"},
              timeout=10.0)
    if ontology_present:
        run_write(load_neo4j_config(),
                  "MATCH (o:Ontology {id: $ontology_id}) "
                  "MATCH (c:Community {id: $community_id}) "
                  "MERGE (o)-[:DEFINES]->(c)",
                  {"ontology_id": ONTOLOGY_ID, "community_id": community_id},
                  timeout=10.0)

    return community_id


# ---------------------------------------------------------------------------
# Phase 1 — entities (discover, dedup, persist)
# ---------------------------------------------------------------------------

def _extract_entities(client: "_Llm", entity: Dict[str, Any], chunks: List[str],
                      document_context: str, ontology_description: str,
                      community_id: str, document_id: str, progress: Any,
                      progress_base: float, progress_span: float) -> Dict[str, Dict[str, Any]]:
    """Discover, deduplicate and persist every instance of one declared entity type."""
    found: Dict[str, Dict[str, Any]] = {}
    label: str = str(entity["name"])
    description: str = str(entity.get("description") or "")
    existing: List[Dict[str, str]] = _existing_community_entities(community_id, label)
    raw: List[Dict[str, str]] = _run_over_chunks(
        client, chunks,
        lambda chunk: _call_extract_entities(client, chunk, label, description,
                                             existing, document_context, ontology_description))

    variants: Dict[str, List[str]] = {}
    first_seen: Dict[str, str] = {}
    for item in raw:
        entity_id: str = _tokenize(item.get("entity_id") or "")
        if not entity_id:
            continue

        name: str = _normalize_entity_name(item.get("name"))
        if not name:
            continue

        variants.setdefault(entity_id, [])
        first_seen.setdefault(entity_id, name)
        if name and name not in variants[entity_id]:
            variants[entity_id].append(name)

    if not variants:
        return found

    unique: List[Dict[str, Any]] = [
        {"entity_id": entity_id, "name": first_seen[entity_id], "names": names}
        for entity_id, names in variants.items()
    ]
    dedup_map, removals = _deduplicate(client, label, unique, document_context)
    removal_set: set = set(removals)
    merged: Dict[str, Dict[str, Any]] = {}
    for item in unique:
        if item["entity_id"] in removal_set:
            continue

        canonical: str = dedup_map.get(item["entity_id"], item["entity_id"])
        if canonical not in merged:
            merged[canonical] = {"entity_id": canonical, "name": item["name"], "names": list(item["names"])}
        else:
            for name in item["names"]:
                if name not in merged[canonical]["names"]:
                    merged[canonical]["names"].append(name)

    done: int = 0
    for item in merged.values():
        entity_id = item["entity_id"]
        created: bool = _persist_entity(label, item["names"], entity_id, community_id, document_id)
        if created:
            found[entity_id] = {"id": entity_id, "label": label, "name": item["name"],
                                "names": list(item["names"])}
            done += 1
        progress(f"persisted {label} {done}/{len(merged)}", int(progress_base + progress_span * ((done / max(len(merged), 1)))))

    return found


def _existing_community_entities(community_id: str, label: str) -> List[Dict[str, str]]:
    """Current members of this community, so chunks reuse their ids (as the skill does)."""
    existing: List[Dict[str, str]] = []
    if not _safe_label(label):
        return existing

    read_result: Dict[str, Any] = run_read(
        load_neo4j_config(),
        f"MATCH (c:Community {{id: $community_id}})-[:AGGREGATES]->(e:Entity:`{label}`) "
        "RETURN e.id AS entity_id, e.name AS name",
        {"community_id": community_id}, timeout=10.0)
    for record in read_result.get("records") or []:
        existing.append({"entity_id": str(record.get("entity_id")),
                         "name": _first_name(record.get("name"))})

    return existing


def _persist_entity(label: str, names: List[str], entity_id: str, community_id: str, document_id: str) -> bool:
    """Create/update the node and attach community, source and IS_DOCUMENTED edges."""
    if not _safe_label(label):
        return False

    write_result: Dict[str, Any] = run_write(
        load_neo4j_config(),
        f"MERGE (e:Entity {{id: $entity_id, ontology_id: $ontology_id}}) ON CREATE SET e.id = $entity_id "
        f"SET e:`{label}`, e.name = reduce(acc = coalesce(e.name, []), name IN $names | "
        "CASE WHEN name IN acc THEN acc ELSE acc + name END) "
        "WITH e "
        "MATCH (c:Community {id: $community_id}) MERGE (c)-[:AGGREGATES]->(e) "
        "WITH e "
        "MATCH (d:Document {id: $document_id}) "
        "MERGE (e)-[:SOURCE]->(d) "
        "MERGE (e)-[l:IS_DOCUMENTED {kind: 'extracted'}]->(d) "
        "  ON CREATE SET l.created_by = 'track2b', l.created_at = datetime()",
        {"entity_id": entity_id, "names": [n for n in names if n][:8],
         "community_id": community_id, "document_id": document_id, "ontology_id": ONTOLOGY_ID},
        timeout=10.0)

    return bool(write_result["ok"])


# ---------------------------------------------------------------------------
# Phase 3 — relationships (discover, validate, persist)
# ---------------------------------------------------------------------------

def _extract_relationships(client: "_Llm", relationship: Dict[str, Any], chunks: List[str],
                           all_entities: Dict[str, Dict[str, Any]], document_context: str,
                           ontology_description: str, rel_label: str, progress: Any,
                           progress_base: float, progress_span: float) -> List[Dict[str, str]]:
    """Discover and persist every instance of one declared relationship type."""
    created: List[Dict[str, str]] = []
    source_labels: List[str] = [str(x) for x in relationship.get("source_labels") or [] if x]
    target_labels: List[str] = [str(x) for x in relationship.get("target_labels") or [] if x]
    candidates: List[str] = []
    for entity_id, entity in all_entities.items():
        entity_labels: List[str] = entity.get("labels") or [str(entity.get("label") or "")]
        if not source_labels or any(label in source_labels for label in entity_labels):
            candidates.append(entity_id)
    allowed_targets: set = {
        entity_id for entity_id, entity in all_entities.items()
        if (not target_labels or any(label in target_labels for label in
                         (entity.get("labels") or [str(entity.get("label") or "")])))
    }
    if not candidates or not allowed_targets:
        return created

    known_ids: List[str] = list(dict.fromkeys(candidates + sorted(allowed_targets)))

    raw: List[Dict[str, str]] = _run_over_chunks(
        client, chunks,
        lambda chunk: _call_extract_relationships(
            client, chunk, rel_label, str(relationship.get("description") or ""),
            _labels_text(source_labels), _labels_text(target_labels),
            known_ids, document_context, ontology_description))

    seen: set = set()
    valid: List[Dict[str, str]] = []
    for item in raw:
        source: str = _tokenize(item.get("source_entity_id") or "")
        target: str = _tokenize(item.get("target_entity_id") or "")
        if source in candidates and target in allowed_targets and (source, target) not in seen:
            seen.add((source, target))
            valid.append({"source": source, "target": target,
                          "source_id": str(item.get("source_id") or "").strip()})

    done: int = 0
    for item in valid:
        if not _persist_relationship(rel_label, item["source"], item["target"], item["source_id"]):
            continue

        done += 1
        created.append({"type": rel_label, "source": item["source"], "target": item["target"]})
        progress(f"persisted {rel_label} {done}/{len(valid)}",
                 int(progress_base + progress_span * (done / max(len(valid), 1))))

    return created


def _persist_relationship(label: str, source_id: str, target_id: str, evidence: str) -> bool:
    """MERGE the ontology-typed edge between two entities, mirroring the editor."""
    if not _safe_label(label):
        return False

    write_result: Dict[str, Any] = run_write(
        load_neo4j_config(),
        f"MATCH (a:Entity {{id: $source, ontology_id: $ontology_id}}) WITH a "
        "MATCH (b:Entity {id: $target, ontology_id: $ontology_id}) "
        f"MERGE (a)-[r:`{label}`]->(b) "
        "SET r.source_id = $source_id, r.ontology_id = $ontology_id",
        {"source": source_id, "target": target_id, "source_id": evidence, "ontology_id": ONTOLOGY_ID},
        timeout=10.0)

    return bool(write_result["ok"])


# ---------------------------------------------------------------------------
# LLM calls (the prompts, adapted from the platform skill to JSON replies)
# ---------------------------------------------------------------------------

def _call_extract_entities(client: "_Llm", chunk: str, label: str, description: str,
                           existing: List[Dict[str, str]], document_context: str,
                           ontology_description: str) -> List[Dict[str, str]]:
    """One chunk, one entity type: names and ids, no attributes (yet)."""
    messages: List[Dict[str, str]] = [{
        "role": "system",
        "content": _prompt_extract_entities(label, description),
    }]
    if existing:
        lines: List[str] = [f"- entity_id: {item['entity_id']}, name: {item['name']}" for item in existing]
        messages.append({"role": "system",
                         "content": f"EXISTING {label.upper()} ENTITIES ALREADY IN THIS KNOWLEDGE GRAPH COMMUNITY:\n"
                                    + "\n".join(lines)
                                    + "\nIf the same entity appears in <TEXT_BLOCK_IN_FOCUS>, reuse its entity_id exactly."})
    if document_context:
        messages.append({"role": "system", "content": f"<DOCUMENT_CONTEXT>\n{document_context}\n</DOCUMENT_CONTEXT>"})
    if ontology_description:
        messages.append({"role": "system", "content": f"FULL ONTOLOGY SCHEMA:\n{ontology_description}"})
    messages.append({"role": "user", "content": f"<TEXT_BLOCK_IN_FOCUS>{chunk}</TEXT_BLOCK_IN_FOCUS>"})

    return _reply_items(client, messages, "entities", ENTITY_MAX_TOKENS)


def _call_extract_relationships(client: "_Llm", chunk: str, rel_label: str, description: str,
                                source_labels: str, target_labels: str, known_ids: List[str],
                                document_context: str, ontology_description: str) -> List[Dict[str, str]]:
    """One chunk, one relationship type, constrained to the entities already extracted."""
    known: str = ", ".join(known_ids[:200]) if known_ids else "(none)"
    messages: List[Dict[str, str]] = [{"role": "system",
                                       "content": _prompt_extract_relationships(rel_label, description,
                                                                                source_labels, target_labels, known)}]
    if document_context:
        messages.append({"role": "system", "content": f"<DOCUMENT_CONTEXT>\n{document_context}\n</DOCUMENT_CONTEXT>"})
    if ontology_description:
        messages.append({"role": "system", "content": f"FULL ONTOLOGY SCHEMA:\n{ontology_description}"})
    messages.append({"role": "user", "content": f"<TEXT_BLOCK_IN_FOCUS>{chunk}</TEXT_BLOCK_IN_FOCUS>"})

    return _reply_items(client, messages, "relationships", RELATIONSHIPS_MAX_TOKENS)


def _deduplicate(client: "_Llm", label: str, unique: List[Dict[str, Any]],
                 document_context: str) -> Tuple[Dict[str, str], List[str]]:
    """LLM-assisted merging of name variants and removal of wrong-type entities."""
    dedup_map: Dict[str, str] = {item["entity_id"]: item["entity_id"] for item in unique}
    removals: List[str] = []
    if len(unique) <= 1:
        return dedup_map, removals

    lines: List[str] = [f"- entity_id: {item['entity_id']}, name: {item['name']}" for item in unique]
    messages: List[Dict[str, str]] = [{"role": "system", "content": _prompt_deduplicate(label)}]
    if document_context:
        messages.append({"role": "system", "content": f"<DOCUMENT_CONTEXT>\n{document_context}\n</DOCUMENT_CONTEXT>"})
    user_prompt: str = ("Entities:\n" + "\n".join(lines)
                        + f"\n\nIdentify duplicates and flag any entity that is NOT a {label}.")
    messages.append({"role": "user", "content": user_prompt})

    answer: Optional[Dict[str, Any]] = client.complete(messages, VALUES_MAX_TOKENS)
    groups: List[Any] = _reply_list(answer, "duplicates")
    for group in groups:
        if not isinstance(group, dict):
            continue

        primary: str = _tokenize(str(group.get("primary_entity_id") or ""))
        for duplicate in group.get("duplicate_entity_ids") or []:
            duplicate_id: str = _tokenize(str(duplicate))
            if primary and duplicate_id in dedup_map:
                dedup_map[duplicate_id] = primary

    removals = [_tokenize(str(item)) for item in _reply_list(answer, "elements_to_remove") if _tokenize(str(item))]

    return dedup_map, [r for r in removals if r in dedup_map]


def _run_over_chunks(client: "_Llm", chunks: List[str], call: Any) -> List[Dict[str, str]]:
    """Fan one prompt-builder out over the chunks in parallel and pool the replies."""
    items: List[Dict[str, str]] = []
    with futures.ThreadPoolExecutor(max_workers=MAX_PARALLELISM) as pool:
        for chunk_items in pool.map(lambda future_chunk: _safe_chunk_call(client, call, future_chunk), chunks):
            items.extend(chunk_items)

    return items


def _safe_chunk_call(client: "_Llm", call: Any, chunk: str) -> List[Dict[str, str]]:
    """One chunk call that may fail without taking the whole extraction down."""
    items: List[Dict[str, str]] = []
    try:
        items = call(chunk)
    except Exception as failure:  # noqa: BLE001 — one chunk failing must not abort the run
        client.notes.append(f"A chunk call failed ({type(failure).__name__}: {str(failure)[:160]}); "
                            "its evidence was skipped.")

    return items


def _safe_declaration_call(client: "_Llm", work: Any, declaration: Dict[str, Any],
                           progress: Any, progress_base: float, progress_span: float) -> Tuple[str, Any]:
    """One ontology type's extraction, isolated so one failing type cannot abort the run."""
    label: str = str(declaration.get("name") or "")
    try:
        progress(f"extracting {label}", int(progress_base))

        return (label, work(declaration, progress, progress_base, progress_span))
    except Exception as failure:  # noqa: BLE001 — one type failing must not abort the run
        client.notes.append(f"{label}: extraction failed ({type(failure).__name__}: "
                            f"{str(failure)[:160]}); its evidence was skipped.")

        return (label, None)


def _run_declarations_in_parallel(client: "_Llm", declarations: List[Dict[str, Any]], work: Any,
                                  progress: Any, phase_word: str,
                                  progress_base: float, progress_span: float) -> List[Tuple[str, Any]]:
    """Fan the work callable out over the declared ontology types in parallel.

    Results return in declaration order so persistence stays deterministic. The
    chunk fan-out nests inside each type call; ``_Llm.complete`` gates every
    in-flight chat request to ``LLM_MAX_CONCURRENCY``, so the pools can stay
    simple while the real concurrency never exceeds that number."""
    results: List[Tuple[str, Any]] = []
    if not declarations:
        return results

    results_by_index: Dict[int, Tuple[str, Any]] = {}
    done: int = 0
    with futures.ThreadPoolExecutor(max_workers=min(MAX_PARALLELISM, len(declarations))) as pool:
        pending: Dict[Any, int] = {}
        for index, declaration in enumerate(declarations):
            base: float = progress_base + progress_span * (index / len(declarations))
            span: float = progress_span / len(declarations)
            pending[pool.submit(_safe_declaration_call, client, work, declaration,
                                progress, base, span)] = index
        for future in futures.as_completed(list(pending.keys())):
            index = pending[future]
            results_by_index[index] = future.result()
            done += 1
            progress(f"done {phase_word} {done}/{len(declarations)}",
                     int(progress_base + progress_span * (done / len(declarations))))

    results = [results_by_index[index] for index in sorted(results_by_index)]

    return results


def _make_entity_work(client: "_Llm", chunks: List[str], document_context: str, ontology_description: str,
                      community_id: str, document_id: str) -> Any:
    """Bind the shared pipeline inputs so one call handles a single entity type."""
    def work(entity: Dict[str, Any], progress: Any, progress_base: float,
             progress_span: float) -> Dict[str, Dict[str, Any]]:
        return _extract_entities(client, entity, chunks, document_context, ontology_description,
                                 community_id, document_id, progress, progress_base, progress_span)

    return work


def _make_relationship_work(client: "_Llm", chunks: List[str], all_entities: Dict[str, Dict[str, Any]],
                            document_context: str, ontology_description: str) -> Any:
    """Bind the shared pipeline inputs so one call handles a single relationship type."""
    def work(relationship: Dict[str, Any], progress: Any, progress_base: float,
             progress_span: float) -> List[Dict[str, str]]:
        return _extract_relationships(client, relationship, chunks, all_entities, document_context,
                                      ontology_description, str(relationship.get("name") or ""),
                                      progress, progress_base, progress_span)

    return work


def _prompt_extract_entities(label: str, description: str) -> str:
    """Entity discovery prompt — the skill's rules, adapted to a JSON reply."""
    return f"""You are a knowledge graph extraction assistant.
Task: extract ALL "{label}" entities from <TEXT_BLOCK_IN_FOCUS>.
Entity definition: {description or label}

Respond with exactly one JSON object:
{{"entities": [{{"entity_id": "<lowercase, underscores>", "name": "<exact name from the text>"}}]}}

RULES:
- Source: extract ONLY from <TEXT_BLOCK_IN_FOCUS>. Do NOT include entities that appear only in <DOCUMENT_CONTEXT>.
- <DOCUMENT_CONTEXT> is provided only to help you identify wrong-type entities to exclude.
- Extract ONLY "{label}" instances. Anything of another type must not appear.
- entity_id: lowercase, words joined by underscores, stable wording so the same real-world entity gets the same id.
- Output the JSON as soon as you have identified all qualifying entities; do not re-examine the same evidence.
- When there are none, respond with exactly {{"entities": []}}

EXAMPLE — extract "Flower" entities
TEXT_BLOCK_IN_FOCUS: "There was a vase of roses on the table, and some margheritas were visible out of the window."
DOCUMENT_CONTEXT: "Charly is a dog owned by Tom and Sarah, ... "
→ Roses and Margheritas are flowers. Tom, Sarah (Persons) and Charly (a dog) are not.
{{"entities": [{{"entity_id": "roses", "name": "Roses"}}, {{"entity_id": "margheritas", "name": "Margheritas"}}]}}"""


def _prompt_extract_relationships(rel_label: str, description: str, source_labels: str,
                                  target_labels: str, known_ids: str) -> str:
    """Relationship discovery prompt — the skill's rules, adapted to a JSON reply."""
    return f"""You are a knowledge graph extraction assistant.
Task: extract ALL "{rel_label}" relationships ({description or rel_label}) from <TEXT_BLOCK_IN_FOCUS>.
Direction: {source_labels} → {target_labels}
Known entity IDs (use ONLY these for source and target): {known_ids}

Respond with exactly one JSON object:
{{"relationships": [{{"source_entity_id": "<id from the known list>", "target_entity_id": "<id from the known list>", "source_id": "<para id of the evidence, if any>"}}]}}

RULES:
- Source: extract ONLY from <TEXT_BLOCK_IN_FOCUS>. Do NOT use <DOCUMENT_CONTEXT> as a source.
- Use ONLY entity IDs from the known list, and obey the direction and the allowed types.
- source_id: the id attribute of the <p id="..."> element in <TEXT_BLOCK_IN_FOCUS> that holds the evidence.
- Do not explain; output only the JSON.
- When there are none, respond with exactly {{"relationships": []}}"""


def _prompt_deduplicate(label: str) -> str:
    """Merge/wrong-type prompt — the skill's rules, adapted to a JSON reply."""
    return f"""Deduplicate "{label}" entities using <DOCUMENT_CONTEXT>.
Merge only if clearly the same real-world entity; remove only if clearly the wrong type. When in doubt, do nothing.
Examples:
  bob (a cat) among Persons → elements_to_remove: ["bob"]
  charly + charly_the_poodle (alias) → duplicates: [{{"primary_entity_id": "charly", "duplicate_entity_ids": ["charly_the_poodle"]}}]
  no issues → {{"duplicates": [], "elements_to_remove": []}}

Respond with exactly one JSON object:
{{"duplicates": [{{"primary_entity_id": "...", "duplicate_entity_ids": ["..."]}}], "elements_to_remove": ["..."]}}"""


# ---------------------------------------------------------------------------
# LLM transport (the env-var trio only)
# ---------------------------------------------------------------------------

class _Llm:
    """One OpenAI-compatible chat endpoint, used with blocking calls from the job thread.

    Apertus's chat template accepts ``system`` only as the first message, so
    messages are sanitised the way the platform's own apertus provider does
    (``provider_apertus.py``): the first ``system`` message stays, every later
    one is resent as ``user``.
    """

    SECONDARY_SYSTEM_ROLE: str = "user"

    def __init__(self, model: str, base_url: str, api_key: str, notes: List[str]) -> None:
        self.model: str = model
        self.notes: List[str] = notes
        self.client: httpx.Client = httpx.Client(timeout=LLM_TIMEOUT_SECONDS)
        self.base_url: str = base_url
        self.api_key: str = api_key
        self.gateway: threading.Semaphore = threading.BoundedSemaphore(LLM_MAX_CONCURRENCY)

    @classmethod
    def create(cls, progress_notes: List[str]) -> Optional["_Llm"]:
        """Build the transport from the env-var trio, or None when it is not configured."""
        config = load_llm_config()
        if not config.configured:
            return None

        return cls(config.name, f"{config.base_url}/chat/completions", config.api_key, progress_notes)

    def close(self) -> None:
        self.client.close()

    def complete(self, messages: List[Dict[str, str]], max_tokens: int) -> Optional[Dict[str, Any]]:
        """One chat completion returning the parsed JSON object, or None when it did not contain one.

        qwen models think before answering: the reply arrives as
        ``reasoning_content`` and the visible ``content`` stays empty whenever
        ``max_tokens`` is exhausted by the thinking (``finish_reason: length``,
        observed live). An empty content is therefore retried with the full
        escalation budget, and one truncated reply is salvaged from the tail of
        the reasoning text, where the model usually already wrote the JSON.
        """
        answer: Optional[Dict[str, Any]] = None
        budget: int = max_tokens
        gave_up: bool = False
        for attempt in range(3):
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": _sanitize_system_roles(messages, self.SECONDARY_SYSTEM_ROLE),
                "max_tokens": budget,
                "temperature": 0.0,
            }
            headers: Dict[str, str] = {"Content-Type": "application/json",
                                       "Authorization": f"Bearer {self.api_key}"}
            self.gateway.acquire()
            try:
                response: httpx.Response = self.client.post(self.base_url, json=payload, headers=headers)
            finally:
                self.gateway.release()
            if response.status_code != 200:
                self.notes.append(f"LLM call failed for {self.model}: HTTP {response.status_code} "
                                  f"{response.text[:200]}")

                break

            choice: Dict[str, Any] = {}
            try:
                choice = response.json()["choices"][0]
            except (KeyError, IndexError, TypeError, ValueError):
                pass
            message: Dict[str, Any] = dict(choice.get("message") or {})
            content: Any = message.get("content")
            if content:
                answer = _json_object(str(content))

                break
            if attempt == 0 and str(choice.get("finish_reason")) == "length":
                answer = _last_json_object(str(message.get("reasoning_content") or ""))

            if answer is not None:
                break
            budget = REASONING_ESCALATION_TOKENS
            gave_up = attempt == 2
            time.sleep(2)
        if answer is None and gave_up:
            self.notes.append(f"LLM returned no content for {self.model} after three attempts "
                              f"(up to {REASONING_ESCALATION_TOKENS} tokens); the call was skipped.")

        return answer


def _sanitize_system_roles(messages: List[Dict[str, str]], secondary_role: str) -> List[Dict[str, str]]:
    """Keep the first system message, resend every later one under `secondary_role`.

    Mirrors the platform's apertus provider: the model's chat template names
    "system" only as the first role, and the gateway rejects a request that
    carries a later one with HTTP 400 "Invalid message role".
    """
    sanitized: List[Dict[str, str]] = []
    seen_system: bool = False
    for message in messages:
        if message.get("role") == "system" and seen_system:
            sanitized.append({**message, "role": secondary_role})
        else:
            sanitized.append(message)
            seen_system = seen_system or message.get("role") == "system"

    return sanitized


def _reply_items(client: _Llm, messages: List[Dict[str, str]], key: str, max_tokens: int) -> List[Dict[str, str]]:
    """Complete and pull one item list out of the reply."""
    answer: Optional[Dict[str, Any]] = client.complete(messages, max_tokens)

    return [item for item in _reply_list(answer, key) if isinstance(item, dict)]


def _reply_list(answer: Optional[Dict[str, Any]], key: str) -> List[Any]:
    value: Any = (answer or {}).get(key)

    return value if isinstance(value, list) else []


def _json_object(text: str) -> Optional[Dict[str, Any]]:
    """First parseable JSON object in the reply (fences tolerated), None otherwise."""
    candidates: List[str] = [match for match in _FENCE_PATTERN.findall(text)] + [text]
    for candidate in candidates:
        match: Optional[re.Match[str]] = _JSON_BLOCK_PATTERN.search(candidate)
        if match is None:
            continue

        try:
            parsed: Any = json.loads(match.group())
        except ValueError:
            continue

        if isinstance(parsed, dict):
            return parsed

    return None


def _last_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Last parseable JSON object in the text — reasoning tails salvage real answers.

    A reasoning model truncated at ``max_tokens`` left the JSON answer inside
    ``reasoning_content``; the model writes the reply as its final thought, so
    the last complete object in the thinking text is the answer, while earlier
    objects are usually prompt echoes or examples.
    """
    answer: Optional[Dict[str, Any]] = None
    for match in _JSON_BLOCK_PATTERN.finditer(text):
        try:
            parsed: Any = json.loads(match.group())
        except ValueError:
            continue

        if isinstance(parsed, dict):
            answer = parsed

    return answer


# ---------------------------------------------------------------------------
# Text and identity helpers (the skill's, re-implemented)
# ---------------------------------------------------------------------------

_IMG_TAG_PATTERN: "re.Pattern" = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_DATA_URI_PATTERN: "re.Pattern" = re.compile(r"data:[a-zA-Z/+.;\-]{0,48}base64,[A-Za-z0-9+/=]{64,}")


def _plain_text(text: str) -> str:
    """Document text fit for a prompt: image markup gone, long base64 blobs gone.

    Ingested PDFs keep their text as HTML with inline page renders as
    ``<img src=... alt=...>``, repeating the rendered payload in both attributes.
    That noise steals the prompt budget and provokes empty completions
    (observed as content ``None``), so it never reaches a message.
    """
    cleaned: str = _IMG_TAG_PATTERN.sub(" ", text)
    cleaned = _DATA_URI_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned


def _chunk_plain_text(text: str, notes: List[str]) -> List[str]:
    """Paragraphs of plain text rebuilt as <p id="para-N"> chunks (skill's chunker shape)."""
    lines: List[str] = [line.strip() for line in text.splitlines()]
    blocks: List[str] = []
    for line in [line for line in lines if line]:
        if blocks and len(blocks[-1]) + len(line) + 2 < CHUNK_ELEMENT_MAX_CHARS:
            blocks[-1] = f"{blocks[-1]}\n{line}"
        else:
            blocks.append(line)

    chunks: List[str] = []
    current: List[str] = []
    size: int = 0
    index: int = 0
    for block in blocks:
        index += 1
        element: str = f'<p id="para-{index}">{_escape(block)}</p>'
        if current and size + len(element) > CHUNK_MAX_CHARS:
            chunks.append("\n".join(current))
            current = [element]
            size = len(element)
        else:
            current.append(element)
            size += len(element)
    if current:
        chunks.append("\n".join(current))

    if len(chunks) > MAX_CHUNKS:
        notes.append(f"The document spans {len(chunks)} chunks; extracting from the first {MAX_CHUNKS} only.")
        chunks = chunks[:MAX_CHUNKS]

    return chunks


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _tokenize(value: str) -> str:
    """Entity ids as the platform tokenizes them: lowercase, punctuation to underscores."""
    tokenized: str = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")

    return tokenized[:120]


def _safe_label(label: Any) -> bool:
    """Labels reach Cypher as backtipped tokens, so only plain identifiers pass."""
    return bool(_LABEL_PATTERN.fullmatch(str(label or "")))


def _first_name(value: Any) -> str:
    return _normalize_entity_name(value)


def _normalize_entity_name(value: Any) -> str:
    """Return a scalar name, never a container's runtime representation."""
    name: str = ""
    if isinstance(value, str):
        name = value.strip()
    elif isinstance(value, list):
        for item in value:
            name = _normalize_entity_name(item)
            if name:
                break
    elif isinstance(value, dict):
        for key in ("value", "name", "text"):
            if key in value:
                name = _normalize_entity_name(value[key])
                if name:
                    break
    if name.startswith("System.Collections.Generic.List`1["):
        name = ""

    return name
