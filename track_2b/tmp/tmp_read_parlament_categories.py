"""Read-only dump of the Parlament ontology's KeywordCategory definitions.

Used once to build the frozen snapshot shipped with the submission; writes
JSON next to itself. Basic auth uses the local dev Neo4j password.
"""

import base64
import json
import urllib.request
from pathlib import Path

NEO4J_URL = "http://localhost:7474/db/neo4j/tx/commit"
AUTH = base64.b64encode(b"neo4j:20012001").decode("ascii")

# Exactly the statement document_categorizer.CATEGORIES_QUERY sends.
QUERY = (
    "MATCH (ontology:Ontology {id: $ontology_id})-[:CONTAINS]->(definition:KeywordCategory) "
    "OPTIONAL MATCH (definition)<-[:OF_CATEGORY]-(:Category)<-[:HAS_CATEGORY]-(document:Document) "
    "RETURN definition.id AS id, definition.text_en AS name, "
    "       definition.description AS description, definition.lov AS lov, "
    "       count(DISTINCT document) AS documents "
    "ORDER BY name"
)


def main() -> None:
    body = json.dumps({"statements": [{"statement": QUERY, "parameters": {"ontology_id": "Parlament"}}]}).encode("utf-8")
    request = urllib.request.Request(NEO4J_URL, data=body, method="POST",
                                     headers={"Authorization": "Basic " + AUTH, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("errors"):
        raise SystemExit(f"neo4j errors: {payload['errors']}")

    columns = payload["results"][0]["columns"]
    rows = [dict(zip(columns, item["row"])) for item in payload["results"][0]["data"]]

    out = Path(__file__).with_name("parlament_categories.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
