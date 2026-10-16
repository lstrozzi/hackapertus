"""Read-only metamodel of the hard-coded `Parlament` ontology.

The demo operates on exactly one ontology, `Parlament`, so its id is a literal
here by design, not a setting (owner decision, 2026-10-15). This module answers
two questions about that ontology:

* what does the connected platform declare?  For Plan Local the platform's own
  database is reachable over Bolt (see ``graphstore/neo4j.py``) and is queried
  read-only — the table then mirrors the live ``(:Ontology {id: 'Parlament'})``
  definitions as they are registered in the main Thalabus deployment;
* what should the box show when the platform database is not reachable —
  which is always the case under Plan SaaS, whose public endpoint does not
  expose the instance's database at all?  A frozen snapshot of the same
  declarations (5 entity types, 7 relationship types, 29 attributes) that
  ships with this submission, taken from the main Thalabus database and
  verified there on 2026-10-15. The payload always says which source it used.

Nothing here writes: no statement this module builds can modify the graph.
Attribute names, types and descriptions are the English ones as registered on
the definition nodes (``OntologyAttribute.name`` / ``.type`` / ``.value``).
"""

__all__ = ["PARLAMENT_ONTOLOGY_ID", "PARLAMENT_ONTOLOGY_NAME", "load_metamodel"]

import os
from typing import Any, Dict, List, Optional

from ..graphstore.neo4j import (PLATFORM_DATABASE_NAME, Neo4jConfig, load_neo4j_config,
                                probe_neo4j, run_read)

PARLAMENT_ONTOLOGY_ID: str = "Parlament"
PARLAMENT_ONTOLOGY_NAME: str = "Open Parlament Data"

QUERY_TIMEOUT_SECONDS: float = 5.0

# One read round trip: definition nodes with their attributes collected in the
# same statement. Entities sort before relationships, both alphabetical.
METAMODEL_QUERY: str = """
MATCH (o:Ontology {id: $ontology_id})-[:CONTAINS]->(e)
WHERE e:EntityOntology OR e:RelationshipOntology
RETURN CASE WHEN e:EntityOntology THEN 'Entity' ELSE 'Relationship' END AS kind,
       last(labels(e)) AS name,
       coalesce(e.entity_description, '') AS description,
       coalesce(e.entity_source_labels, []) AS source_labels,
       coalesce(e.entity_target_labels, []) AS target_labels,
       [ (e)-[:HAS_ATTRIBUTE]->(a:OntologyAttribute) |
           { name: a.name, type: a.type, description: coalesce(a.value, '') } ] AS attributes
ORDER BY kind ASC, name ASC
"""

# Frozen declarations, read from the main Thalabus database (Parlament ontology,
# 2026-10-15) with the same statement shape as METAMODEL_QUERY. Text shortened
# where the platform stores it; the name/type columns are verbatim.
_SNAPSHOT_ELEMENTS: List[Dict[str, Any]] = [
    {
        "kind": "Entity", "name": "Affair",
        "description": "A single parliamentary affair (motion, postulate, interpellation, "
                       "government business, ...) as published by the Swiss Federal Assembly and "
                       "mirrored in the OpenParlData corpus. Identified business-wise by business_id. "
                       "The instrument/kind of an affair is deliberately NOT an attribute: it is "
                       "expressed via the AFFAIR_TYPE document category.",
        "source_labels": [], "target_labels": [],
        "attributes": [
            {"name": "business_id", "type": "string",
             "description": "Official parliamentary business number. Stable re-link key across imports and updates."},
            {"name": "closed_on", "type": "date",
             "description": "Date on which the affair was closed or terminated."},
            {"name": "extraction_notes", "type": "string",
             "description": "Free-text classifier rationale recorded during ingestion/harmonization."},
            {"name": "scope", "type": "string",
             "description": "Parliament ID. Join key to Organisation."},
            {"name": "source_url", "type": "string",
             "description": "Canonical OpenParlData URI of the affair record. Provenance link back to the source."},
            {"name": "state", "type": "string",
             "description": "Lifecycle state, harmonized from events.title_harmonized. Allowed values: see the AFFAIR_STATE category LoV."},
            {"name": "submitted_on", "type": "date",
             "description": "Date on which the affair was submitted."},
            {"name": "title_ml", "type": "string",
             "description": "JSON object of per-language titles, e.g. {\"de\": \"...\", \"fr\": \"...\", \"it\": \"...\"}."},
            {"name": "type_wikidata", "type": "string",
             "description": "Wikidata Q-id of the affair instrument. Alignment hook only; the human-readable instrument lives in the AFFAIR_TYPE category."},
        ],
    },
    {
        "kind": "Entity", "name": "Canton",
        "description": "A Swiss canton: the 26 member states, grounded in OpenParlData bodies rows "
                       "with type=canton. A canton is a territory, not a parliament: its legislative "
                       "body is referenced by body_id and modelled as an Organisation. Persons attach "
                       "via FROM_CANTON.",
        "source_labels": [], "target_labels": [],
        "attributes": [
            {"name": "body_id", "type": "string",
             "description": "OpenParlData body id of the canton parliament. Join key to the legislature Organisation and to Affair.scope."},
            {"name": "canton_code", "type": "string",
             "description": "Two-letter canton key (bodies.canton_key), e.g. BE, SH, GR. Canonical join key."},
            {"name": "name_ml", "type": "string",
             "description": "JSON object of per-language names, e.g. {\"de\": \"Bern\", \"fr\": \"Berne\", \"it\": \"Berna\"}."},
            {"name": "wikidata_uri", "type": "string",
             "description": "Alignment URI to the canton article; alignment hook consistent with Affair.type_wikidata."},
        ],
    },
    {
        "kind": "Entity", "name": "Organisation",
        "description": "A collective actor in the parliamentary corpus: parliamentary body, committee, "
                       "department, party, or vote group. Affairs link via HANDLED_BY and PUBLISHED_BY; "
                       "persons via MEMBER_OF.",
        "source_labels": [], "target_labels": [],
        "attributes": [
            {"name": "level", "type": "string",
             "description": "Geographic/administrative level. Allowed values: country, canton, city, municipality."},
            {"name": "name_ml", "type": "string",
             "description": "JSON object of per-language names, e.g. {\"de\": \"...\", \"fr\": \"...\", \"it\": \"...\"}."},
            {"name": "official_key", "type": "string",
             "description": "Corpus body or party key. Stable re-link key."},
            {"name": "org_kind", "type": "string",
             "description": "Kind of organisation. Allowed values: body, committee, department, party, group."},
        ],
    },
    {
        "kind": "Entity", "name": "Person",
        "description": "A natural person in the parliamentary corpus: a contributor (author, cosigner, "
                       "speaker or assignee) and/or a voter. Persons connect to affairs via "
                       "CONTRIBUTES_TO, to organisations via MEMBER_OF, and to their canton via "
                       "FROM_CANTON.",
        "source_labels": [], "target_labels": [],
        "attributes": [
            {"name": "full_name", "type": "string", "description": "Name of the person as published in the corpus."},
            {"name": "party", "type": "string",
             "description": "Harmonized party acronym. Soft link to Organisation nodes of kind party."},
            {"name": "person_key", "type": "string",
             "description": "Corpus person ID. Stable key for re-linking contributor and vote records."},
        ],
    },
    {
        "kind": "Entity", "name": "Topic",
        "description": "A subject topic from the solution-owned topic scheme. Schema-declared; the "
                       "runtime signal may equally ride the TOPIC document category (open point C-1).",
        "source_labels": [], "target_labels": [],
        "attributes": [
            {"name": "eurovoc_uri", "type": "string",
             "description": "Optional alignment URI to a EuroVoc concept, kept separate from topic_uri."},
            {"name": "gnd_uri", "type": "string",
             "description": "Optional alignment URI to a GND subject heading."},
            {"name": "label_ml", "type": "string",
             "description": "JSON object of per-language preferred labels, e.g. {\"de\": \"...\", \"fr\": \"...\", \"it\": \"...\"}."},
            {"name": "scheme", "type": "string",
             "description": "Which topical layer this node came from. Allowed values: derived, eurovoc, hybrid."},
            {"name": "topic_uri", "type": "string",
             "description": "Solution-owned topic URI: stable identifier of the derived scheme."},
        ],
    },
    {
        "kind": "Relationship", "name": "ABOUT_TOPIC",
        "description": "An affair is about this subject topic; the weight carries the classifier "
                       "confidence rather than a binary judgement.",
        "source_labels": ["Affair"], "target_labels": ["Topic"],
        "attributes": [
            {"name": "weight", "type": "string",
             "description": "Classifier confidence for this affair-topic pair as a decimal string in 0..1."},
        ],
    },
    {
        "kind": "Relationship", "name": "CONTRIBUTES_TO",
        "description": "A person contributed to an affair. One relationship type for all contributor "
                       "roles: role_kind carries the distinction.",
        "source_labels": ["Person"], "target_labels": ["Affair"],
        "attributes": [
            {"name": "role_kind", "type": "string",
             "description": "Role of the person towards the affair. Allowed values: author, cosign, speaker, assignee."},
        ],
    },
    {
        "kind": "Relationship", "name": "FROM_CANTON",
        "description": "The canton a person comes from or represents (origin or constituency). An "
                       "ingestion derivation: resolve the person body via persons.body_id and read "
                       "bodies.canton_key.",
        "source_labels": ["Person"], "target_labels": ["Canton"],
        "attributes": [],
    },
    {
        "kind": "Relationship", "name": "HANDLED_BY",
        "description": "Committee/department routing of an affair: which organisation handles (or "
                       "handled) the affair at a given stage.",
        "source_labels": ["Affair"], "target_labels": ["Organisation"],
        "attributes": [
            {"name": "stage_kind", "type": "string",
             "description": "Which handling stage this edge represents. Allowed values: assigned_committee, leading_department."},
        ],
    },
    {
        "kind": "Relationship", "name": "IS_DOCUMENTED",
        "description": "Links an entity to the Document that describes or registers it. For Affair the "
                       "cardinality is 1:N. Machine-managed edge properties are owned by the writer, "
                       "not by this ontology; traversal is ACL-guarded.",
        "source_labels": ["Affair"], "target_labels": ["Document"],
        "attributes": [],
    },
    {
        "kind": "Relationship", "name": "MEMBER_OF",
        "description": "Membership of a person in an organisation (party affiliation, committee seat). "
                       "The classic cross-affair bridge.",
        "source_labels": ["Person"], "target_labels": ["Organisation"],
        "attributes": [
            {"name": "since", "type": "string",
             "description": "Optional. Year or date since which the person is a member of the organisation."},
        ],
    },
    {
        "kind": "Relationship", "name": "PUBLISHED_BY",
        "description": "The affair was published/issued by this organisation.",
        "source_labels": ["Affair"], "target_labels": ["Organisation"],
        "attributes": [],
    },
]


def _snapshot_payload(note: str) -> Dict[str, Any]:
    """Build the response with the frozen snapshot as its source."""
    return {
        "ok": True,
        "source": "snapshot",
        "note": note,
        "ontology": {"id": PARLAMENT_ONTOLOGY_ID, "name": PARLAMENT_ONTOLOGY_NAME},
        "elements": _SNAPSHOT_ELEMENTS,
    }


def _platform_metamodel_config() -> Optional[Neo4jConfig]:
    """Connection settings for the platform's own database, or None when unreachable by contract.

    Mirrors :func:`graphstore.neo4j.probe_platform_database`: Plan SaaS never sees it, the
    address must differ from this app's own store, and the password falls back to the
    app's, which is what the Plan Local stack shares. Only readable here, never logged.
    """
    plan: str = os.environ.get("THALABUS_PLAN", "local").strip().lower() or "local"
    if plan in ("b", "saas"):
        return None

    uri: str = os.environ.get("NEO4J_PLATFORM_URI", "").strip() or "bolt://neo4j:7687"
    own_uri: str = os.environ.get("NEO4J_URI", "bolt://graph:7687").strip().rstrip("/")
    if uri.rstrip("/") == own_uri:
        return None

    password: str = os.environ.get("PLATFORM_NEO4J_PASSWORD", "") or load_neo4j_config().password
    user: str = os.environ.get("PLATFORM_NEO4J_USER", "neo4j").strip() or "neo4j"
    database: str = os.environ.get("NEO4J_PLATFORM_DATABASE", PLATFORM_DATABASE_NAME).strip() or PLATFORM_DATABASE_NAME
    if not password:
        return None

    return Neo4jConfig(uri=uri, user=user, password=password, database=database)


def _live_elements(config: Neo4jConfig) -> Optional[List[Dict[str, Any]]]:
    """Run the metamodel read against the platform's database; None when it cannot be asked.

    ``run_read`` never raises: it reports a typed error instead, which this turns
    into a None, which the caller answers with the snapshot and a true note.
    """
    probe: Dict[str, Any] = probe_neo4j(config)
    if not probe.get("reachable"):
        return None

    result: Dict[str, Any] = run_read(config, METAMODEL_QUERY,
                                      {"ontology_id": PARLAMENT_ONTOLOGY_ID},
                                      timeout=QUERY_TIMEOUT_SECONDS)
    if not result.get("ok"):
        return None

    elements: List[Dict[str, Any]] = []
    for record in result.get("records") or []:
        attributes: List[Dict[str, Any]] = [dict(a) for a in (record.get("attributes") or []) if a]
        elements.append({
            "kind": record.get("kind") or "",
            "name": record.get("name") or "",
            "description": record.get("description") or "",
            "source_labels": label_list(record.get("source_labels")),
            "target_labels": label_list(record.get("target_labels")),
            "attributes": attributes,
        })

    return elements


def label_list(value: Any) -> List[str]:
    """Label constraints as declared, never exploded into characters.

    The platform stores these properties as plain strings (single label,
    ``"Affair"``) or comma-separated strings (``"Person,Animal"``, the
    KnowledgeGraphEditor convention), while the snapshot ships real lists. A
    naive ``list(value)`` would split ``"Affair"`` into ``['A', 'f', ...]``,
    making every type filter silently match nothing.
    """
    normalized: List[str] = []
    if isinstance(value, str):
        normalized = [part.strip() for part in value.split(",") if part.strip()]
    elif value:
        normalized = [str(x).strip() for x in value if str(x).strip()]

    return normalized


def load_metamodel() -> Dict[str, Any]:
    """Answer the metamodel box: the platform's declaration when reachable, else the frozen snapshot.

    The returned payload names the source (`platform` / `snapshot`) and carries a
    truthful note; the GUI renders note text instead of guessing.
    """
    config: Optional[Neo4jConfig] = _platform_metamodel_config()
    note: str = ""
    if config is None:
        note = ("Platform database not reachable under this run plan (Plan SaaS does not expose it): "
                "showing the frozen declarations shipped with this submission.")

        return _snapshot_payload(note)

    elements: Optional[List[Dict[str, Any]]] = _live_elements(config)
    if elements is None:
        note = ("No answer from the platform database at the configured address, so this cannot "
                "claim to be live: showing the frozen declarations shipped with this submission.")

        return _snapshot_payload(note)

    if not elements:
        note = (f"The platform answered but declares no ontology '{PARLAMENT_ONTOLOGY_ID}' yet "
                "(it is created by the T13 setup): showing the frozen declarations shipped with "
                "this submission.")

        return _snapshot_payload(note)

    return {
        "ok": True,
        "source": "platform",
        "note": f"Read the live definitions of ontology '{PARLAMENT_ONTOLOGY_ID}' from the platform's own database (read-only).",
        "ontology": {"id": PARLAMENT_ONTOLOGY_ID, "name": PARLAMENT_ONTOLOGY_NAME},
        "elements": elements,
    }
