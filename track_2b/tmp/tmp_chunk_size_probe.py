"""Measure rerank quality on the Munz document with finer chunks, before changing any constant.

Re-embeds ONLY docid-413 with a smaller chunk target, re-scores the phrasings, and prints
the before/after so the chunk size decision rests on numbers instead of a guess.
"""

from typing import Any, Dict, List

from app.graphstore.neo4j import load_neo4j_config, run_write
from app.model_services import load_model_services_config
from app.operations import chat_rag

TARGET_DOC: str = "docid-413"
QUESTIONS: List[str] = [
    "Wer hat die Kleine Anfrage 2016/8 eingereicht und welchen Vorwurf enthaelt sie?",
    "Sieht die Regierung den Bericht der Sonntagszeitung als glaubwuerdig an?",
    "Wie entwickelte sich der Gewinn von Rising Tide Budapest?",
    "Hat eine der genannten Firmen Steuererleichterungen erhalten?",
    "Transparenz von Stiftungen und angesiedelten Firmen",
]

config = load_neo4j_config()
services = load_model_services_config()


def score(question: str) -> Dict[str, Any]:
    """Retrieve and rerank one question, returning the top two candidates."""
    vector: Any = chat_rag._embed_texts(services, [question])[0]
    ranked: List[Dict[str, Any]] = sorted(
        chat_rag._rerank(services, question, chat_rag._vector_search(config, vector)),
        key=lambda c: -c["rerank_score"])

    return {"top": ranked[0], "second": ranked[1] if len(ranked) > 1 else ranked[0], "count": len(ranked)}


def report(label: str) -> None:
    print(f"\n=== {label} (target {chat_rag.CONTEXT_TARGET_CHARS} / max {chat_rag.MAX_CHUNK_CHARS}) ===")
    for question in QUESTIONS:
            found: Dict[str, Any] = score(question)
            top: Dict[str, Any] = found["top"]
            verdict: str = "PASS" if top["rerank_score"] >= chat_rag.MIN_RERANK_SCORE else "refused"
            print(f"  {top['rerank_score']:.3f} {verdict:8} chunk {top['ordinal']} of {len(top['text'])} chars | "
                  f"runner-up {found['second']['rerank_score']:.3f} | {question[:48]}")


report("as deployed")

# Re-embed the Munz document alone with finer chunks.
chat_rag.CONTEXT_TARGET_CHARS = 320
chat_rag.MAX_CHUNK_CHARS = 600
run_write(config, "MATCH (s:Track2bRagState {documentId: $id}) DELETE s", {"id": TARGET_DOC}, timeout=30.0)
documents: List[Dict[str, Any]] = [d for d in chat_rag._ready_documents(config) if d["id"] == TARGET_DOC]
print("\nchunks at target 320:", [len(c) for c in chat_rag._split_chunks(documents[0]["text"])])
chat_rag._ensure_chunks_current(config, services, chat_rag._embed_texts(services, ["warm"])[0].__len__(), documents)
report("finer chunks")
