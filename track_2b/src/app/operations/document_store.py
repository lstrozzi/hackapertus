"""Track 2B's isolated document metadata in its own Neo4j database."""

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Dict

from ..graphstore.neo4j import Neo4jConfig, load_neo4j_config, run_read, run_write

# The category ontology the categorisation boxes are built around. Hard-coded by
# design (it is the demo's subject), defined once here, and read through the same
# store the documents live in — under both run plans the categories are already
# present in the graph this app is configured with.
CATEGORY_ONTOLOGY_ID: str = "Parlament"


def document_node_id(name: str) -> str:
    """Return the fallback graph id for a sample file the graph does not know yet.

    Only a file with no existing node of that name lands on this id: every write
    first asks :func:`_existing_document_ids` (batch) or uses :data:`DOCUMENT_BY_NAME`
    (single document), so re-synchronising a file the platform already ingested
    under `docid-N` updates that node instead of creating a second one.
    """
    document_id: str = "docid-track2b-" + hashlib.sha256(name.casefold().encode("utf-8")).hexdigest()[:24]

    return document_id


# How the platform numbers documents, and the status shown for a node this app
# has adopted but never ingested (the platform stores its own status in the
# lowercase `status`, which is a different question and a different property).
PLATFORM_NUMERIC_ID_PREFIX: str = "docid-"
APP_STATUS_UNKNOWN: str = "NotYetProcessed"


def _is_platform_numbered(document_id: str) -> bool:
    """True for `docid-<digits>`: the platform's own numbering, not this app's."""
    remainder: str = document_id[len(PLATFORM_NUMERIC_ID_PREFIX):]

    return document_id.startswith(PLATFORM_NUMERIC_ID_PREFIX) and remainder.isdigit()


def _preferred_document_id(candidates: list[str], own_id: str) -> str:
    """Pick the one node a filename refers to, preferring the platform's numbering.

    The order is deliberate. A platform `docid-N` node already carries the text,
    the chapters and the dossier link, so adopting it keeps the sample and the
    platform document as one record (and matches the id the panel shows in its
    "platform document id" column). A node this app created itself ranks next, so
    a file that was only ever ingested here keeps its own node. Anything else in
    the graph (another corpus's node that happens to share a filename) is not
    adopted at all — the answer is then `own_id`, which creates this app's node.
    """
    numbered: list[str] = [item for item in candidates if _is_platform_numbered(item)]
    chosen: str = ""
    if numbered:
        chosen = max(numbered, key=lambda item: (len(item), item))
    elif own_id in candidates:
        chosen = own_id

    return chosen


def _existing_document_ids(config: Neo4jConfig, names: list[str]) -> Dict[str, Any]:
    """Resolve each sample filename to a node that already carries it.

    One read for the whole batch, matching the three spellings a filename is
    stored under across platform versions (`FileName` here, `name`/`label` for
    platform uploads). `ids` answers every requested name, empty string when the
    graph has no node for it yet, which is what tells the synchroniser to create.
    """
    result: Dict[str, Any] = {"ok": False, "ids": {}, "error": None}
    if not names:
        result["ok"] = True
    else:
        read_result: Dict[str, Any] = run_read(
            config,
            "UNWIND $names AS name "
            "OPTIONAL MATCH (document:Document) "
            "WHERE document.FileName = name OR document.name = name OR document.label = name "
            "WITH name, [node IN collect(document) WHERE node IS NOT NULL] AS documents "
            "RETURN name AS name, [node IN documents | node.id] AS ids",
            {"names": names},
            timeout=10.0,
        )
        if read_result["ok"]:
            ids: Dict[str, str] = {name: "" for name in names}
            for record in read_result["records"]:
                name: str = str(record["name"])
                ids[name] = _preferred_document_id([str(item) for item in (record.get("ids") or []) if item],
                                                   document_node_id(name))
            result["ok"] = True
            result["ids"] = ids
        else:
            result["error"] = read_result["error"]

    return result


# Addresses "the document for this file" in one statement, for every single-document
# write and read below. The app's own id wins when that node exists, so a file that
# was tracked here before stays where it is; otherwise the node that already carries
# the filename is adopted, in either property shape. Without the name matches each of
# these statements would silently match nothing after adoption (or recreate the node
# it was meant to reuse), which is how one file ended up with two Document nodes.
DOCUMENT_BY_NAME: str = (
    "MATCH (document:Document) "
    "WHERE document.id = $id "
    "   OR document.FileName = $name "
    "   OR document.name = $name "
    "   OR document.label = $name "
    "WITH document ORDER BY CASE WHEN document.id = $id THEN 0 ELSE 1 END, document.id DESC "
    "LIMIT 1 "
)


def synchronize_documents(documents: list[dict[str, Any]]) -> Dict[str, Any]:
    """Upsert sample-file metadata and return each node's persisted status.

    Existing first: each name is resolved against the graph before anything is
    written, so a file the platform has already ingested is updated in place and
    a file only ever ingested here is created once and then updated in place too.
    """
    config = load_neo4j_config()
    result: Dict[str, Any] = {"ok": False, "statuses": {}, "error": None}
    if not config.configured:
        result["error"] = {"type": "not_configured", "message": "Track 2B Neo4j is not configured."}
    else:
        existing: Dict[str, Any] = _existing_document_ids(
            config, [str(document["name"]) for document in documents])
        if not existing["ok"]:
            result["error"] = existing["error"]
        else:
            rows: list[Dict[str, Any]] = []
            for document in documents:
                name: str = str(document["name"])
                rows.append({
                    "id": str(existing["ids"].get(name) or "") or document_node_id(name),
                    "name": name,
                    "file_path": str(Path(name)),
                })

            write_result: Dict[str, Any] = run_write(
                config,
                """
                UNWIND $documents AS item
                MERGE (document:Document {id: item.id})
                ON CREATE SET document.Status = 'NotYetProcessed',
                              document.Type = 'Other',
                              document.CreatedAt = datetime(),
                              document.UpdatedAt = datetime()
                SET document.Name = item.name,
                    document.FileName = item.name,
                    document.FilePath = item.file_path
                """,
                {"documents": rows},
                timeout=10.0,
            )
            if write_result["ok"]:
                ids: list[str] = [row["id"] for row in rows]
                read_result: Dict[str, Any] = run_read(
                    config,
                    "MATCH (document:Document) WHERE document.id IN $ids "
                    "RETURN coalesce(document.FileName, document.name, document.label) AS name, "
                    "       coalesce(document.Status, $unknown) AS status",
                    {"ids": ids, "unknown": APP_STATUS_UNKNOWN},
                    timeout=10.0,
                )
                if read_result["ok"]:
                    result["ok"] = True
                    result["statuses"] = {
                        str(record["name"]): str(record["status"])
                        for record in read_result["records"]
                    }
                else:
                    result["error"] = read_result["error"]
            else:
                result["error"] = write_result["error"]

    return result


def update_document_status(name: str, status: str, text: str | None = None,
                           error: str | None = None) -> Dict[str, Any]:
    """Persist one document's processing state and extracted text."""
    config = load_neo4j_config()
    result: Dict[str, Any] = run_write(
        config,
        DOCUMENT_BY_NAME +
        "SET document.Status = $status, document.UpdatedAt = datetime(), "
        "document.Text = $text, document.Error = $error",
        {"id": document_node_id(name), "name": name,
         "status": status, "text": text, "error": error},
        timeout=10.0,
    )

    return result


# ── Categorisation support ──────────────────────────────────────────────────────
# The shapes below mirror the platform's own category model, in the submission's
# isolated node space (ids keyed by `document_node_id`): proposals are properties
# on the Track 2B document node, while applied assignments use the platform's
# edges, so a categorised sample can be read by platform queries too:
#   (document)-[:HAS_CATEGORY]->(category:Category)-[:OF_CATEGORY]->(definition:KeywordCategory)
#   (document)-[:FOLLOWS]->(ontology:Ontology)-[:CONTAINS]->(definition)


def read_document_states(names: list[str]) -> Dict[str, Any]:
    """Return per-document status, proposals and applied-category counts."""
    config = load_neo4j_config()
    result: Dict[str, Any] = {"ok": False, "states": {}, "error": None}
    if not config.configured:
        result["error"] = {"type": "not_configured", "message": "Track 2B Neo4j is not configured."}
    else:
        existing: Dict[str, Any] = _existing_document_ids(config, names)
        if existing["ok"]:
            rows: list[Dict[str, str]] = [{"name": name, "id": str(existing["ids"].get(name) or "")}
                                          for name in names if existing["ids"].get(name)]
            read_result: Dict[str, Any] = run_read(
                config,
                "UNWIND $rows AS row "
                "MATCH (document:Document {id: row.id}) "
                "OPTIONAL MATCH (document)-[:HAS_CATEGORY]->(assigned:Category) "
                "-[:OF_CATEGORY]->(:KeywordCategory)<-[:CONTAINS]-(:Ontology {id: $ontology_id}) "
                "RETURN coalesce(document.FileName, document.name, document.label) AS name, "
                "       document.Status AS status, "
                "       document.ProposedCategories AS proposed, "
                "       document.Text IS NOT NULL AS has_text, "
                "       count(assigned) AS assigned",
                {"rows": rows, "ontology_id": CATEGORY_ONTOLOGY_ID},
                timeout=10.0,
            )
            if read_result["ok"]:
                states: Dict[str, Any] = {}
                for record in read_result["records"]:
                    states[str(record["name"])] = _parse_document_state(record)
                result["ok"] = True
                result["states"] = states
            else:
                result["error"] = read_result["error"]
        else:
            result["error"] = existing["error"]

    return result


def _parse_document_state(record: Dict[str, Any]) -> Dict[str, Any]:
    """Materialise one state record, parsing the stored proposal JSON defensively."""
    proposed: list[Dict[str, Any]] = []
    raw: Any = record.get("proposed")
    if isinstance(raw, str) and raw:
        try:
            parsed: Any = json.loads(raw)
            if isinstance(parsed, list):
                proposed = [item for item in parsed if isinstance(item, dict)]
        except ValueError:
            proposed = []

    return {
        "status": str(record["status"] or "Unknown"),
        "has_text": bool(record.get("has_text")),
        "assigned_count": int(record.get("assigned") or 0),
        "proposed": proposed,
    }


def read_document(name: str) -> Dict[str, Any]:
    """Fetch one document's categorisation inputs: graph id, status and stored text."""
    config = load_neo4j_config()
    result: Dict[str, Any] = {"ok": False, "document": None, "error": None}
    if not config.configured:
        result["error"] = {"type": "not_configured", "message": "Track 2B Neo4j is not configured."}
    else:
        read_result: Dict[str, Any] = run_read(
            config,
            DOCUMENT_BY_NAME +
            "RETURN document.id AS id, document.FileName AS name, "
            "document.Status AS status, document.Text AS text",
            {"id": document_node_id(name), "name": name},
            timeout=10.0,
        )
        if read_result["ok"]:
            if read_result["records"]:
                record: Dict[str, Any] = dict(read_result["records"][0])
                record["text"] = str(record.get("text") or "")
                result["ok"] = True
                result["document"] = record
            else:
                result["error"] = {"type": "document_not_found",
                                   "message": f"{name} is not tracked in the Track 2B graph."}
        else:
            result["error"] = read_result["error"]

    return result


def read_document_by_id(document_id: str) -> Dict[str, Any]:
    """Fetch one document by its graph id, whichever property shape it stores.

    Platform documents (the dossier's own nodes, e.g. `docid-400`) keep lowercase
    `name`/`status`/`text`; this app's tracked samples keep uppercase
    `FileName`/`Status`/`Text`. Both shapes are coalesced so either node can be
    extraction input.
    """
    config = load_neo4j_config()
    result: Dict[str, Any] = {"ok": False, "document": None, "error": None}
    if not config.configured:
        result["error"] = {"type": "not_configured", "message": "Track 2B Neo4j is not configured."}
    else:
        read_result: Dict[str, Any] = run_read(
            config,
            "MATCH (document:Document {id: $id}) RETURN document.id AS id, "
            "coalesce(document.name, document.FileName) AS name, "
            "coalesce(document.status, document.Status) AS status, "
            "coalesce(document.text, document.Text) AS text LIMIT 1",
            {"id": document_id},
            timeout=10.0,
        )
        if read_result["ok"]:
            if read_result["records"]:
                record: Dict[str, Any] = dict(read_result["records"][0])
                record["text"] = str(record.get("text") or "")
                result["ok"] = True
                result["document"] = record
            else:
                result["error"] = {"type": "document_not_found",
                                   "message": f"{document_id} is not a document in the graph."}
        else:
            result["error"] = read_result["error"]

    return result


def save_proposed_categories(name: str, proposals: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Store a proposal set on the document node — reviewable, never authoritative."""
    config = load_neo4j_config()
    result: Dict[str, Any] = run_write(
        config,
        DOCUMENT_BY_NAME +
        "SET document.ProposedCategories = $json, document.ProposedAt = datetime(), "
        "    document.ProposedCount = $count, document.UpdatedAt = datetime()",
        {"id": document_node_id(name), "name": name,
         "json": json.dumps(proposals, ensure_ascii=False),
         "count": len(proposals)},
        timeout=10.0,
    )

    return result


def apply_categories(name: str, items: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Write assignments as Category nodes with the platform's own edge shape."""
    config = load_neo4j_config()
    result: Dict[str, Any] = {"ok": False, "applied": 0, "error": None}
    if not items:
        result["ok"] = True
    else:
        write_result: Dict[str, Any] = run_write(
            config,
            DOCUMENT_BY_NAME +
            "MERGE (ontology:Ontology {id: $ontology_id}) "
            "MERGE (document)-[:FOLLOWS]->(ontology) "
            "WITH document, ontology "
            "UNWIND $items AS item "
            "MATCH (ontology)-[:CONTAINS]->(definition:KeywordCategory {id: item.category_id}) "
            "MERGE (assignment:Category {text: item.text})-[:OF_CATEGORY]->(definition) "
            "  ON CREATE SET assignment.lang = null "
            "MERGE (document)-[:HAS_CATEGORY]->(assignment)",
            {"id": document_node_id(name), "name": name,
             "ontology_id": CATEGORY_ONTOLOGY_ID, "items": items},
            timeout=15.0,
        )
        if write_result["ok"]:
            count_result: Dict[str, Any] = run_read(
                config,
                DOCUMENT_BY_NAME +
                "MATCH (document)-[:HAS_CATEGORY]->(assignment:Category) "
                "-[:OF_CATEGORY]->(:KeywordCategory)<-[:CONTAINS]-(ontology:Ontology {id: $ontology_id}) "
                "RETURN count(assignment) AS applied",
                {"id": document_node_id(name), "name": name,
                 "ontology_id": CATEGORY_ONTOLOGY_ID},
                timeout=10.0,
            )
            if count_result["ok"]:
                records: list[Dict[str, Any]] = count_result["records"]
                result["ok"] = True
                result["applied"] = int(records[0]["applied"]) if records else 0
            else:
                result["error"] = count_result["error"]
        else:
            result["error"] = write_result["error"]

    return result


def promote_document_status(name: str, status: str) -> Dict[str, Any]:
    """Write only Status/UpdatedAt — leaves the extracted text and error untouched."""
    config = load_neo4j_config()
    result: Dict[str, Any] = run_write(
        config,
        DOCUMENT_BY_NAME +
        "SET document.Status = $status, document.UpdatedAt = datetime()",
        {"id": document_node_id(name), "name": name, "status": status},
        timeout=10.0,
    )

    return result


def promote_platform_document_status(name: str, status: str = "Ready") -> Dict[str, Any]:
    """Set the matched *platform* document's status, as the platform's C# does.

    In the real Thalabus the C# backend flips the platform document to `Ready`
    when the categorisation skill completes; this app shares that graph, so an
    apply that moved only the local record would leave the platform queue state
    wrong. The platform node is resolved with the same matcher and tie-break as
    every other platform-side read, then only its lowercase `status` is written.
    A file with no platform match is not a failure (`promoted: False`); a store
    that refuses the write reports `ok: False` with the typed error.
    """
    result: Dict[str, Any] = {"ok": False, "promoted": False, "platform_document_id": "", "error": None}
    found: Dict[str, Any] = read_platform_document(name, max_text_chars=1)
    if not found["ok"]:
        result["error"] = found["error"]
    elif found["document"] is None:
        result["ok"] = True
    else:
        platform_id: str = str(found["document"]["id"])
        write_result: Dict[str, Any] = run_write(
            load_neo4j_config(),
            "MATCH (document:Document {id: $id}) SET document.status = $status",
            {"id": platform_id, "status": status},
            timeout=10.0,
        )
        if write_result["ok"]:
            # run_write answers with summary counters: a set property is proof
            # the matched node existed and received the new status.
            properties_set: int = int((write_result.get("counters") or {}).get("properties_set") or 0)
            result["ok"] = True
            result["promoted"] = properties_set > 0
            result["platform_document_id"] = platform_id
        else:
            result["error"] = write_result["error"]

    return result


# ── Platform-side identity ───────────────────────────────────────────────────────
# The sample files are copies of documents the platform has already ingested under its
# own numbering (`docid-413`), so the ingestion panel can name the platform document each
# sample corresponds to. Two details decide the shape of this read:
#   * this app's own nodes carry the same filename, so they are excluded — the column
#     answers "which platform document is this?", not "which node of ours?";
#   * the platform records a filename under `name`/`label` for uploads while `FileName`
#     is this store's spelling, so all three are matched and the first attribute that
#     hits wins, which keeps the lookup working across platform versions.
TRACK2B_ID_PREFIX: str = "docid-track2b-"
PLATFORM_ID_PREFIX: str = "docid-"
PLATFORM_ID_PLACEHOLDER: str = "n/a"


def _latest_platform_id(candidates: list[str]) -> str:
    """Choose the id to show when one filename matches several platform documents."""
    numbered: list[str] = [item for item in candidates if item.startswith(PLATFORM_ID_PREFIX)]
    chosen: str = ""
    if numbered:
        chosen = max(numbered, key=lambda item: (len(item), item))

    return chosen


# The single platform status that makes a document eligible for category proposals.
# Deliberately not knowledge_graph_status: that tracks KG extraction, a different workflow.
PLATFORM_CATEGORIZABLE_STATUS: str = "NeedsCategorization"


def platform_status_categorizable(status: Any) -> bool:
    """One exact platform status awaits categorisation; everything else does not."""
    return str(status or "") == PLATFORM_CATEGORIZABLE_STATUS


def platform_document_ids(names: list[str]) -> Dict[str, Any]:
    """Resolve sample filenames to platform ids and statuses, answered `n/a` when unmatched."""
    config = load_neo4j_config()
    result: Dict[str, Any] = {"ok": False, "ids": {}, "statuses": {}, "error": None}
    if not config.configured:
        result["error"] = {"type": "not_configured", "message": "Track 2B Neo4j is not configured."}
    elif not names:
        result["ok"] = True
    else:
        read_result: Dict[str, Any] = run_read(
            config,
            "UNWIND $names AS name "
            "OPTIONAL MATCH (document:Document) "
            "WHERE NOT document.id STARTS WITH $own_prefix "
            "  AND (document.FileName = name OR document.name = name OR document.label = name) "
            "RETURN name AS name, collect({id: document.id, status: document.status}) AS documents",
            {"names": names, "own_prefix": TRACK2B_ID_PREFIX},
            timeout=10.0,
        )
        if read_result["ok"]:
            ids: Dict[str, str] = {}
            statuses: Dict[str, str] = {}
            for record in read_result["records"]:
                platform_documents: list[Dict[str, Any]] = [
                    item for item in (record.get("documents") or []) if item.get("id")
                ]
                chosen_id: str = _latest_platform_id([str(item["id"]) for item in platform_documents])
                name: str = str(record["name"])
                ids[name] = chosen_id
                for platform_document in platform_documents:
                    if str(platform_document["id"]) == chosen_id and platform_document.get("status"):
                        statuses[name] = str(platform_document["status"])
            result["ok"] = True
            result["ids"] = ids
            result["statuses"] = statuses
        else:
            result["error"] = read_result["error"]

    return result


def read_platform_document(name: str, max_text_chars: int = 60_001) -> Dict[str, Any]:
    """Resolve one sample filename to its platform document, with that document's text.

    This answers the question the inference box asks when the Track 2B record has no
    text: which platform document matches this filename, what is its `status`, and what
    text does it carry. The read is capped (`max_text_chars`) because the categoriser
    truncates anyway, and read-only by design — proposals must never reach the platform
    node. The same-id tie-break as the platform id column keeps the box consistent
    when one filename matches several uploads.
    """
    config = load_neo4j_config()
    result: Dict[str, Any] = {"ok": False, "document": None, "error": None}
    if not config.configured:
        result["error"] = {"type": "not_configured", "message": "Track 2B Neo4j is not configured."}
    else:
        read_result: Dict[str, Any] = run_read(
            config,
            "MATCH (document:Document) "
            "WHERE NOT document.id STARTS WITH $own_prefix "
            "  AND (document.FileName = $name OR document.name = $name OR document.label = $name) "
            "RETURN document.id AS id, document.status AS status, "
            "       substring(coalesce(document.text, ''), 0, $max_text) AS text",
            {"name": name, "own_prefix": TRACK2B_ID_PREFIX, "max_text": max_text_chars},
            timeout=15.0,
        )
        if read_result["ok"]:
            candidates: list[Dict[str, Any]] = [record for record in read_result["records"] if record.get("id")]
            chosen_id: str = _latest_platform_id([str(record["id"]) for record in candidates])
            if chosen_id:
                record: Dict[str, Any] = next(item for item in candidates if str(item["id"]) == chosen_id)
                result["document"] = {"id": chosen_id,
                                      "status": str(record.get("status") or ""),
                                      "text": str(record.get("text") or "")}
            result["ok"] = True
        else:
            result["error"] = read_result["error"]

    return result


# ── Ontology extension support ────────────────────────────────────────────────
# Mirrors the platform's ontology_service.save_proposed_categories: a proposal is
# a real KeywordCategory definition under the same Ontology, flagged `proposed:
# true`. Flagged nodes are inert for categorisation (the skill's category menu
# filters them out) until an explicit approval clears the flag; nothing else in
# the ontology and no document node is touched.
PROPOSED_CATEGORY_ID_PREFIX: str = "keyword_cat_"


def save_proposed_ontology_categories(proposals: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Create proposed KeywordCategory definitions under the demo ontology.

    Each proposal needs a non-empty `text_en`; `description`, the translations
    (`text_fr`/`text_de`/`text_it`) and a comma-joined `lov` string are optional
    and stored only when present, exactly like the platform's writer. Proposals
    whose name already exists in the ontology (approved or proposed, case
    -insensitive) are skipped, not duplicated. No node is ever updated here:
    approval is the separate, explicit `approve_proposed_category` step.
    """
    config = load_neo4j_config()
    result: Dict[str, Any] = {"ok": False, "saved": 0, "skipped": 0, "items": [], "error": None}
    if not config.configured:
        result["error"] = {"type": "not_configured", "message": "Track 2B Neo4j is not configured."}
    elif not proposals:
        result["error"] = {"type": "nothing_to_save", "message": "No proposals to save."}
    else:
        existing_result: Dict[str, Any] = run_read(
            config,
            "MATCH (:Ontology {id: $ontology_id})-[:CONTAINS]->(definition:KeywordCategory) "
            "RETURN toLower(definition.text_en) AS name",
            {"ontology_id": CATEGORY_ONTOLOGY_ID},
            timeout=10.0,
        )
        if not existing_result["ok"]:
            result["error"] = existing_result["error"]
        else:
            existing: set[str] = {str(record.get("name") or "").strip() for record in existing_result["records"]}
            items: list[Dict[str, Any]] = []
            batch_seen: set[str] = set()
            for proposal in proposals:
                text_en: str = str(proposal.get("text_en") or "").strip()
                key: str = text_en.casefold()
                if not text_en or key in existing or key in batch_seen:
                    result["skipped"] += 1
                    continue
                batch_seen.add(key)
                item: Dict[str, Any] = {"id": PROPOSED_CATEGORY_ID_PREFIX + uuid.uuid4().hex[:12],
                                        "text_en": text_en,
                                        "description": str(proposal.get("description") or "").strip(),
                                        "proposed": True}
                for language_key in ("text_fr", "text_de", "text_it"):
                    translation: str = str(proposal.get(language_key) or "").strip()
                    if translation:
                        item[language_key] = translation
                lov: str = str(proposal.get("lov") or "").strip()
                if lov:
                    item["lov"] = lov
                items.append(item)

            result["items"] = items
            if not items:
                result["ok"] = True
            else:
                write_result: Dict[str, Any] = run_write(
                    config,
                    "MATCH (ontology:Ontology {id: $ontology_id}) "
                    "UNWIND $items AS item "
                    "CREATE (category:KeywordCategory) "
                    "SET category += item "
                    "WITH ontology, category "
                    "CREATE (ontology)-[:CONTAINS]->(category)",
                    {"ontology_id": CATEGORY_ONTOLOGY_ID, "items": items},
                    timeout=15.0,
                )
                if write_result["ok"]:
                    # run_write answers with summary counters, never rows: every
                    # item is its own KeywordCategory node, so nodes_created is
                    # the number of definitions the graph actually accepted.
                    created: int = int((write_result.get("counters") or {}).get("nodes_created") or 0)
                    result["saved"] = created
                    result["skipped"] += len(items) - created
                    result["ok"] = True
                else:
                    result["error"] = write_result["error"]

    return result


def approve_proposed_category(category_id: str) -> Dict[str, Any]:
    """Clear the `proposed` flag on one flagged category — the explicit adoption step.

    Touches only that property on a node that is currently flagged and contained
    by the demo ontology; edges, descriptions and every other property are left
    exactly as they were. Rejects unknown ids and categories that are not (or no
    longer) proposed, so an approval can never edit an approved definition.
    """
    config = load_neo4j_config()
    result: Dict[str, Any] = {"ok": False, "approved": 0, "error": None}
    if not config.configured:
        result["error"] = {"type": "not_configured", "message": "Track 2B Neo4j is not configured."}
    else:
        write_result: Dict[str, Any] = run_write(
            config,
            "MATCH (ontology:Ontology {id: $ontology_id})-[:CONTAINS]->(category:KeywordCategory {id: $id}) "
            "WHERE category.proposed = true "
            "SET category.proposed = false",
            {"ontology_id": CATEGORY_ONTOLOGY_ID, "id": category_id},
            timeout=10.0,
        )
        if not write_result["ok"]:
            result["error"] = write_result["error"]
        else:
            # The guarded statement can only change the one flag, so a single
            # properties_set (and never more) is exact proof of one adoption;
            # zero means the id matched no flagged category.
            properties_set: int = int((write_result.get("counters") or {}).get("properties_set") or 0)
            if properties_set != 1:
                result["error"] = {"type": "not_proposed",
                                   "message": f"Category '{category_id}' is not a proposed category of this ontology."}
            else:
                result["approved"] = 1
                result["ok"] = True

    return result
