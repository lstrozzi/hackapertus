"""Reset one document's chunk marker so the next question re-embeds it at the deployed chunk sizes."""

from app.graphstore.neo4j import load_neo4j_config, run_write

config = load_neo4j_config()
deleted = run_write(config, "MATCH (s:Track2bRagState {documentId: $id}) DELETE s", {"id": "docid-413"}, timeout=10.0)
kept = run_write(config, "MATCH (c:Track2bChunk {documentId: $id}) DETACH DELETE c RETURN count(c) AS n",
                  {"id": "docid-413"}, timeout=10.0)
print("marker deleted:", deleted["ok"], "| probe chunks removed:", kept["records"])
