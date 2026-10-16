"""Print the evidence windows linking the Capaul affair across documents."""
from app.graphstore.neo4j import load_neo4j_config, run_read

IDS = ["docid-394", "docid-391", "docid-402", "docid-414"]
TERM = "Capaul"

query = ("UNWIND $ids AS did MATCH (d:Document {id: did}) "
         "RETURN d.id AS id, coalesce(d.title, '') AS title, coalesce(d.text, d.Text, '') AS body")
result = run_read(load_neo4j_config(), query, {"ids": IDS}, timeout=20.0)
if not result.get("ok"):
    raise SystemExit(f"read failed: {result.get('error')}")

for record in result["records"]:
    body = str(record["body"])
    print(f"=== {record['id']} | {record['title'][:70]} | {len(body)} chars")
    hits = []
    cursor = 0
    while True:
        found = body.find(TERM, cursor)
        if found < 0:
            break
        hits.append(found)
        cursor = found + len(TERM)
    windows = []
    for position in hits[:4]:
        start = max(0, position - 110)
        window = " ".join(body[start:position + 240].split())
        windows.append(window)
    for window in windows:
        print(f"   … {window}")
    print()
