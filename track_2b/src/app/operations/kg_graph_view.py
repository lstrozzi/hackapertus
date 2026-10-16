"""Graph tab box: a minimal read-only viewer of the extracted knowledge graph.

The markup is server-rendered; the behaviour lives in ``static/js/kg_graph_view.js``
which boots lazily on the first Graph-tab click, loads the same vis-network UMD
pair the platform's own KG page uses (vendored under ``static/lib/vis-network/``)
and renders the payload of ``GET /api/operations/kg/graph``.

The data is read from this app's own store — the one the extraction runner writes
to (see ``kg_extraction_runner.py``): entities are ``(:Entity:{Label})`` nodes
aggregated by a ``(:Community)`` node. The community is fixed to the demo's
``DefaultCommunity_Parlament`` — that selector renders disabled; this box shows
one graph, the one the demo builds. The entity-class select is enabled: it
filters the shown nodes client-side (``All`` shows every class) and is populated
from the loaded payload by ``kg_graph_view.js``.
Documents, ``SOURCE`` and ``IS_DOCUMENTED``
edges are deliberately excluded: the viewer shows the entity graph, not provenance.
Every statement here is a plain read; nothing in this module can modify the graph.
"""

__all__ = ["FIXED_COMMUNITY_ID", "render", "load_graph"]

from typing import Any, Dict, List

from ..graphstore.neo4j import Neo4jConfig, load_neo4j_config, run_read
from .kg_metamodel import PARLAMENT_ONTOLOGY_ID

QUERY_TIMEOUT_SECONDS: float = 10.0

FIXED_COMMUNITY_ID: str = f"DefaultCommunity_{PARLAMENT_ONTOLOGY_ID}"

_STRUCTURAL_RELATIONSHIPS: List[str] = ["SOURCE", "IS_DOCUMENTED"]

_ENTITIES_IN_COMMUNITY_QUERY: str = (
    "MATCH (:Community {id: $community_id})-[:AGGREGATES]->(e:Entity) "
    "RETURN e.id AS id, "
    "       last([l IN labels(e) WHERE l <> 'Entity']) AS label, "
    "       e.name AS name "
    "ORDER BY id"
)

_RELATIONSHIPS_IN_COMMUNITY_QUERY: str = (
    "MATCH (c:Community {id: $community_id})-[:AGGREGATES]->(a:Entity) "
    "MATCH (c)-[:AGGREGATES]->(b:Entity) "
    "MATCH (a)-[r]-(b) "
    f"WHERE NOT type(r) IN {_STRUCTURAL_RELATIONSHIPS} "
    "RETURN DISTINCT type(r) AS kind, startNode(r).id AS source, endNode(r).id AS target "
    "ORDER BY kind, source, target"
)


def render() -> str:
    """Render the graph-viewer box: toolbar, canvas, status strip, selection footer."""
    return f'''<div class="kg-graph-view" id="kg-graph-view">
  <header class="kg-graph-toolbar">
    <div class="kg-graph-control"><label for="kg-graph-community">Community:</label>
      <select id="kg-graph-community" disabled title="Fixed to the demo community"><option selected>{FIXED_COMMUNITY_ID}</option></select></div>
    <div class="kg-graph-control"><label for="kg-graph-class">Entity class:</label>
      <select id="kg-graph-class" title="Filter nodes by entity class"><option value="" selected>All</option></select></div>
    <button type="button" id="kg-graph-reload" disabled>Reload</button>
    <span class="operation-status kg-graph-status" id="kg-graph-status">Graph not loaded</span>
  </header>
  <div class="kg-graph-canvas" id="kg-graph-canvas" role="img" aria-label="Knowledge graph viewer"></div>
  <footer class="kg-graph-footer"><span id="kg-graph-selection">Select a node to inspect it. Drag to move, scroll to zoom.</span></footer>
</div>'''


def _display_name(raw_name: Any, entity_id: str) -> tuple[str, List[str]]:
    """Normalise ``e.name`` (a list of name variants by convention) to (display, all)."""
    names: List[str] = []
    if isinstance(raw_name, list):
        names = [str(item) for item in raw_name if item]
    elif raw_name:
        names = [str(raw_name)]

    return (names[0] if names else entity_id), names


def load_graph() -> Dict[str, Any]:
    """Read the fixed demo community's entity graph for the viewer. Never raises."""
    payload: Dict[str, Any] = {
        "ok": False,
        "source": "app_database",
        "community": FIXED_COMMUNITY_ID,
        "entities": [],
        "relationships": [],
        "error": None,
    }
    config: Neo4jConfig = load_neo4j_config()
    parameters: Dict[str, Any] = {"community_id": FIXED_COMMUNITY_ID}

    entities_result = run_read(config, _ENTITIES_IN_COMMUNITY_QUERY, parameters,
                               timeout=QUERY_TIMEOUT_SECONDS)
    if not entities_result["ok"]:
        payload["error"] = entities_result["error"]

        return payload

    entity_ids: set = set()
    entities_by_id: Dict[str, Dict[str, Any]] = {}
    for record in entities_result["records"]:
        entity_id: str = str(record.get("id") or "")
        if not entity_id:
            continue
        display: str
        names: List[str]
        display, names = _display_name(record.get("name"), entity_id)
        entity_ids.add(entity_id)
        # One node can carry several class labels (the extraction merges entities
        # of different types into one id), so the query may return the same id on
        # several rows. vis ids must be unique: keep the first row's label and
        # union the name variants instead of emitting the id twice.
        seen: Dict[str, Any] = entities_by_id.get(entity_id)
        if seen is not None:
            for name in names:
                if name not in seen["names"]:
                    seen["names"].append(name)
            continue
        entities_by_id[entity_id] = {
            "id": entity_id,
            "label": str(record.get("label") or "Entity"),
            "name": display,
            "names": names,
        }

    payload["entities"] = list(entities_by_id.values())

    relationships_result = run_read(config, _RELATIONSHIPS_IN_COMMUNITY_QUERY, parameters,
                                    timeout=QUERY_TIMEOUT_SECONDS)
    if relationships_result["ok"]:
        for record in relationships_result["records"]:
            source: str = str(record.get("source") or "")
            target: str = str(record.get("target") or "")
            if source not in entity_ids or target not in entity_ids:
                continue
            payload["relationships"].append({
                "relationship": str(record.get("kind") or ""),
                "source": source,
                "target": target,
            })
    else:
        # A missing edge layer still leaves a useful node view; the status line says which
        # of the two reads failed so the blank edge set is never mistaken for "no edges".
        payload["edges_error"] = relationships_result["error"]

    payload["community"] = FIXED_COMMUNITY_ID
    payload["ok"] = True

    return payload
