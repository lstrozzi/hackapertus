"""Offline probe: does the fixed _ready_documents see the platform corpus, and what does chunking produce?

Run with PYTHONPATH set to this folder so the local `src` package wins over any other checkout.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

for line in Path(".env").read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if stripped and not stripped.startswith("#") and "=" in stripped:
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))

from src.app.operations import chat_rag
from src.app.config import load_llm_config
from src.app.graphstore.neo4j import load_neo4j_config
from src.app.model_services import load_model_services_config

print("module:", chat_rag.__file__)

config = load_neo4j_config()
print("neo4j configured:", config.configured, config.uri)

documents: List[Dict[str, Any]] = chat_rag._ready_documents(config)
if documents and "error" in documents[0]:
    print("READ ERROR:", documents[0]["error"])

else:
    print("eligible documents:", len(documents))
    print("total plain chars:", sum(len(d["text"]) for d in documents))
    munz: List[Dict[str, Any]] = [d for d in documents if "068597ae" in d["id"] or "068597ae" in d["name"]]
    for d in munz:
        print("\n== munz document ==", d["id"], "|", d["name"], "| plain chars:", len(d["text"]))
        print("--- first 400 plain chars ---")
        print(d["text"][:400])
        print("--- hidden markup survived? ---")
        for marker in ("ocr-textarea", "image-classification", "<p ", "class=\"class-name\""):
            print(f"  {marker}: {marker in d['text']}")
        chunks: List[str] = chat_rag._split_chunks(d["text"])
        print("chunks:", len(chunks), "| sizes:", [len(c) for c in chunks])
        print("hash:", chat_rag._text_hash(d["text"])[:16])

    biggest: List[Dict[str, Any]] = sorted(documents, key=lambda d: -len(d["text"]))[:5]
    print("\n largest docs:")
    for d in biggest:
        print("  ", len(d["text"]), d["name"])
    print("\nllm:", load_llm_config().name, "| embeddings:", load_model_services_config().embeddings_model)
