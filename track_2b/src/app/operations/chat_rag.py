"""Hard-coded single-pass RAG answering for the Chat Q&A box.

One request, one pipeline: embed the question, retrieve candidate fragments from
the graph by vector similarity, rerank them with the cross-encoder, keep the
fragments scoring at least ``MIN_RERANK_SCORE``, and have the chat model answer
from exactly those fragments with numbered citations. No query planner, no
augmentation, no agents — the same stages the platform RagService runs inside
its plan, reduced to a fixed sequence, as required for the submission.

Why not the sliced ``src.services.rag_service``: it is the platform's service
and expects the platform's data model (Chapter/Paragraph/Summary nodes, dataset
hints, ACL context, vector indexes created by platform ingestion). This
submission keeps its documents isolated as ``Document`` nodes in the wizard's
graph store, so retrieval runs against that store through the wizard's own
connection layer, following the RagService stage order.

Embeddings, reranking and generation reach the endpoints the wizard verified:
``EMBEDDINGS_URL``/``EMBEDDINGS_MODEL``, ``RERANKER_URL``/``RERANKER_MODEL`` and
the ``LLM_*`` trio. Fragment embeddings are persisted in the graph and reused,
so one question costs one embedding, one cosine retrieval, one rerank, one chat.
"""

__all__ = ["answer_question", "RAG_MODES"]

import hashlib
import html
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx

from ..config import LlmConfig, load_llm_config
from ..graphstore.neo4j import Neo4jConfig, load_neo4j_config, run_read, run_write
from ..model_services import ModelServicesConfig, load_model_services_config
from .chat_expansion import expand_seeds

# The contract of the Chat Q&A box: only fragments at least this relevant reach
# the model. Applied to the reranker score — the embedding score decides which
# fragments are candidates, the cross-encoder decides whether they are worth it.
MIN_RERANK_SCORE: float = 0.5

# Candidate pool, in order: ANN pre-selection from the vector index, then the
# cross-encoder reranker.
VECTOR_TOP_K: int = 40
RERANK_CANDIDATES: int = 30

# Chunking: paragraphs are packed towards CONTEXT_TARGET_CHARS, never above
# MAX_CHUNK_CHARS; oversized paragraphs split at sentence boundaries.
CONTEXT_TARGET_CHARS: int = 800
MAX_CHUNK_CHARS: int = 1200
EMBEDDING_BATCH_SIZE: int = 8
QUESTION_MAX_CHARS: int = 4000

# How much of the passing fragments is actually put into the prompt, and how the
# answer reports the fragments it was built from.
CONTEXT_MAX_CHARS: int = 14000
CITATION_EXCERPT_CHARS: int = 220

# One question means one generation, but a gateway hiccup on the chat route should
# not end the answer the retrieval already earned. Transient failures — the router
# could not reach the deployment (5xx/429), a connection reset, a timeout — are
# retried a bounded number of times with a growing pause; anything else surfaces
# immediately, because a wrong key or a malformed reply will not fix itself.
GENERATION_ATTEMPTS: int = 3
GENERATION_BACKOFF_SECONDS: float = 2.0
# Ceiling across all attempts of one question, so a wedged route cannot hold a
# question for attempts x timeout. One slow-but-working attempt still fits; the
# second only runs while real time remains.
GENERATION_DEADLINE_SECONDS: float = 150.0
TRANSIENT_GENERATION_ERRORS: frozenset = frozenset({
    "llm_timeout", "llm_malformed_json",
    "ConnectError", "ConnectTimeout", "ReadTimeout", "ReadError", "WriteError",
    "RemoteProtocolError", "PoolTimeout", "NetworkError",
})

CHUNK_INDEX_NAME: str = "track2b_chunk_embeddings"
STORE_TIMEOUT_SECONDS: float = 30.0
MODEL_SERVICES_TIMEOUT_SECONDS: float = 120.0
LLM_TIMEOUT_SECONDS: float = 180.0
LLM_TEMPERATURE: float = 0.2
LLM_MAX_TOKENS: int = 8192

SYSTEM_PROMPT: str = (
    "You are the Q&A assistant of the Thalabus Hack Apertus submission. "
    "Answer the user's question using ONLY the numbered context fragments provided. "
    "Cite every factual claim with the fragment number in square brackets, for example [2]. "
    "If the fragments do not contain the answer, state exactly which part is missing and invent nothing. "
    "Keep the answer concise and in the language of the question."
)

# Which ingestion states make a document RAG-eligible: both already carry final text.
RAG_DOCUMENT_STATUSES: Tuple[str, ...] = ("Ready", "NeedsCategorization")

# The two search modes. `simple` is the shipped pipeline, byte-for-byte; the
# dropdown can only ever send these two. An absent mode means simple and an
# unknown one is a hand-crafted request — rejected with 400 at the API boundary
# rather than silently charged to a cost tier (T35 acceptance criteria).
RAG_MODES: Tuple[str, ...] = ("simple", "advanced")
# Seeds are the documents of the passing fragments, best-scoring first. Each
# seed drives candidate retrieval and a pair judge per candidate, so the cap
# stays small even when 30 fragments pass.
ADVANCED_MAX_SEEDS: int = 6
# The gateway model thinks silently by default: 220 tokens gave finish_reason 'length' with no content
# and 1600 still took ~25 s per pair, so _ask_over switches thinking off (measured 2026-10-16: 0.8 s, 22 tokens).
JUDGE_MAX_TOKENS: int = 300
# Fragments kept per expanded document, so one verbose confirmation cannot push
# every seed fragment out of the context window.
EXPANDED_FRAGMENTS_PER_DOCUMENT: int = 5

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+")


class _ServiceError(Exception):
    """One interaction that ended with a typed failure, ready for the result dict."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type: str = error_type
        self.message: str = message

    def to_dict(self) -> Dict[str, str]:
        return {"type": self.error_type, "message": self.message}


def answer_question(question: str, mode: str = "simple") -> Dict[str, Any]:
    """Run the fixed RAG pipeline for one question and report the outcome.

    ``simple`` is the shipped pipeline: embed, retrieve, rerank, generate.
    ``advanced`` inserts the T35 expansion between rerank and generation: the
    passing fragments become seeds, the expansion engine proposes candidates
    through the category ladder and evidenced entity hops, a parallel pair
    judge decides every candidate, and the confirmed documents' fragments face
    the *same* reranker and the *same* 50% floor before the model answers from
    the enlarged set. Mode is never a filter, and a refusal stays a refusal.

    Never raises: every external failure lands in ``error`` as type plus message,
    so the caller always receives a typed answer object.
    """
    normalized_mode: str = mode if mode in RAG_MODES else "simple"
    result: Dict[str, Any] = {
        "ok": False,
        "mode": normalized_mode,
        "question": (question or "").strip(),
        "answer": None,
        "answer_source": None,
        "citations": [],
        "candidates": 0,
        "passed_threshold": 0,
        "expansion": None,
        "timings_ms": {},
        "error": None,
    }
    if not result["question"]:
        result["error"] = {"type": "missing_question", "message": "Type a question first."}

        return result
    if len(result["question"]) > QUESTION_MAX_CHARS:
        result["error"] = {"type": "question_too_long", "message": f"The question exceeds {QUESTION_MAX_CHARS} characters."}

        return result

    config: Neo4jConfig = load_neo4j_config()
    if not config.configured:
        result["error"] = {"type": "store_not_configured", "message": "NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD are not set."}

        return result

    started: float = time.perf_counter()
    documents: List[Dict[str, Any]] = _ready_documents(config)
    if documents and documents[0].get("error"):
        result["error"] = documents[0]["error"]

        return result
    result["timings_ms"]["documents"] = _elapsed_ms(started)

    if not documents:
        result["error"] = {"type": "no_documents", "message": "No ingested document with text found yet — ingest one in the Documents tab first."}

        return result

    started = time.perf_counter()
    services: ModelServicesConfig = load_model_services_config()
    try:
        query_vector: List[float] = _embed_texts(services, [result["question"]])[0]
    except _ServiceError as failure:
        result["error"] = failure.to_dict()

        return result
    result["timings_ms"]["embed_query"] = _elapsed_ms(started)

    started = time.perf_counter()
    try:
        _ensure_chunks_current(config, services, len(query_vector), documents)
    except _ServiceError as failure:
        result["error"] = failure.to_dict()

        return result
    result["timings_ms"]["index"] = _elapsed_ms(started)

    started = time.perf_counter()
    try:
        candidates: List[Dict[str, Any]] = _vector_search(config, query_vector)
    except _ServiceError as failure:
        result["error"] = failure.to_dict()

        return result
    result["timings_ms"]["search"] = _elapsed_ms(started)
    result["candidates"] = len(candidates)
    _attach_display_names(config, candidates)

    if not candidates:
        result["error"] = {"type": "nothing_indexed", "message": "The vector index returned no candidates for this question."}

        return result

    started = time.perf_counter()
    try:
        candidates = _rerank(services, result["question"], candidates)
    except _ServiceError as failure:
        result["error"] = failure.to_dict()

        return result
    result["timings_ms"]["rerank"] = _elapsed_ms(started)

    passing: List[Dict[str, Any]] = sorted(
        (candidate for candidate in candidates if candidate["rerank_score"] >= MIN_RERANK_SCORE),
        key=lambda candidate: candidate["rerank_score"],
        reverse=True,
    )
    result["passed_threshold"] = len(passing)

    if not passing:
        best: float = max(candidate["rerank_score"] for candidate in candidates)
        result["ok"] = True
        result["answer_source"] = "deterministic"
        result["answer"] = (
            "None of the ingested documents matched this question at the required 50% relevance "
            f"(the best candidate reached {best:.0%}). Rephrase the question, or ingest a document "
            "that covers the topic in the Documents tab."
        )
        if normalized_mode == "advanced":
            result["expansion"] = {"status": "no_seeds", "seeds": [], "documents": [], "rejections": [],
                                   "notes": ["Nothing to expand from: no fragment passed the 50% rerank floor."]}

        return result

    fragments: List[Dict[str, Any]] = list(passing)
    if normalized_mode == "advanced":
        seed_ids: List[str] = list(dict.fromkeys(str(candidate["document_id"]) for candidate in passing))[:ADVANCED_MAX_SEEDS]
        judge_llm: LlmConfig = load_llm_config()
        expansion: Dict[str, Any] = expand_seeds(
            config,
            RAG_DOCUMENT_STATUSES,
            seed_ids,
            result["question"],
            lambda system_prompt, user_prompt: _ask_over(judge_llm, system_prompt, user_prompt, JUDGE_MAX_TOKENS),
            judge_llm.name,
        )
        result["expansion"] = expansion
        if expansion.get("documents"):
            started = time.perf_counter()
            fragments = _merge_expanded_fragments(config, services, result["question"], fragments, expansion)
            result["timings_ms"]["expanded_rerank"] = _elapsed_ms(started)

    context: str = ""
    used: List[Dict[str, Any]] = []
    context, used = _build_context(fragments)
    result["citations"] = [_citation(ordinal, candidate) for ordinal, candidate in enumerate(used, start=1)]

    started = time.perf_counter()
    try:
        answer: str = _generate(load_llm_config(), result["question"], context)
    except _ServiceError as failure:
        result["error"] = failure.to_dict()

        return result
    result["timings_ms"]["generate"] = _elapsed_ms(started)

    result["ok"] = True
    result["answer_source"] = "llm"
    result["answer"] = answer

    return result


def _ready_documents(config: Neo4jConfig) -> List[Dict[str, Any]]:
    """Load every ingested document with its text, or one error entry.

    Two producers write this store and they disagree on the schema: the wizard's local
    ingester writes `Status`, `Text`, `FileName`, while the platform's document
    ingestion writes `status`, `text`, `name` and holds the body as HTML. Reading only
    one spelling hides the other's whole corpus without any error, so every field is
    coalesced across both here, as the Knowledge Graph tab already does.

    Ingestion writes NeedsCategorization first and the categorization feature later
    promotes it to Ready; both states carry final extraction text, so either makes a
    document RAG-eligible.
    """
    read: Dict[str, Any] = run_read(
        config,
        "MATCH (document:Document) "
        "WHERE coalesce(document.status, document.Status) IN $statuses "
        "  AND NOT coalesce(document.text, document.Text) IS NULL "
        "RETURN document.id AS id, "
        "       coalesce(document.name, document.FileName, document.label, document.id) AS name, "
        "       coalesce(document.text, document.Text) AS text "
        "ORDER BY name, id",
        {"statuses": list(RAG_DOCUMENT_STATUSES)},
        timeout=STORE_TIMEOUT_SECONDS,
    )
    if not read["ok"]:
        return [{"error": read["error"]}]

    documents: List[Dict[str, Any]] = []
    for record in read["records"]:
        text: str = _plain_text(str(record.get("text") or ""))
        if not record.get("id") or not text:
            continue
        documents.append({"id": str(record["id"]),
                          "name": str(record.get("name") or record["id"]),
                          "text": text})

    return documents


# The platform nests extraction noise inside the body: OCR drafts of logos and
# signatures, and image-classification probes, both hidden from the reader. They are
# metadata about the extraction, not document text, and must never be embedded or quoted.
_HIDDEN_BLOCK: Any = re.compile(r"<(?:ocr-textarea|div)\b[^>]*?(?:display:\s*none|image-classification)[^>]*>.*?</(?:ocr-textarea|div)>", re.DOTALL)
_IMG_TAG: Any = re.compile(r"<img\b[^>]*?/?>", re.DOTALL)
_BREAK_TAG: Any = re.compile(r"<br\s*/?>", re.IGNORECASE)
_BLOCK_END: Any = re.compile(r"</(?:p|h[1-6]|li|tr|div|section|article|table|blockquote)\s*>", re.IGNORECASE)
_ANY_TAG: Any = re.compile(r"<[^>]+>")


def _plain_text(text: str) -> str:
    """The body as the model should read it: text, without markup or hidden noise.

    Locally ingested documents are already plain and pass through untouched. Platform
    ingestion stores HTML, so hidden blocks are dropped, block ends become paragraph
    breaks, remaining tags are removed and entities unescaped. Chunking, reranking and
    citation excerpts then all see plain text, and chunk hashes are taken over this
    result, so a body that changes shape here is re-embedded on the next question.
    """
    plain: str = text.strip()
    if "<" not in text:
        return plain

    cleaned: str = _HIDDEN_BLOCK.sub(" ", text)
    cleaned = _IMG_TAG.sub(" ", cleaned)
    cleaned = _BREAK_TAG.sub("\n", cleaned)
    cleaned = _BLOCK_END.sub("\n\n", cleaned)
    cleaned = html.unescape(_ANY_TAG.sub(" ", cleaned))

    lines: List[str] = [line.strip() for line in cleaned.splitlines()]

    return "\n\n".join(line for line in lines if line)


def _ensure_chunks_current(config: Neo4jConfig, services: ModelServicesConfig,
                           dimensions: int, documents: List[Dict[str, Any]]) -> None:
    """Make sure every Ready document has current chunks behind the vector index."""
    read: Dict[str, Any] = run_read(
        config,
        "MATCH (state:Track2bRagState) RETURN state.documentId AS documentId, "
        "state.hash AS hash, state.model AS model",
        timeout=STORE_TIMEOUT_SECONDS,
    )
    if not read["ok"]:
        _raise_store(read["error"])
    markers: Dict[str, Dict[str, Any]] = {
        str(record["documentId"]): record
        for record in read["records"]
        if record.get("documentId")
    }

    _ensure_vector_index(config, dimensions)

    for document in documents:
        marker: Dict[str, Any] = markers.get(document["id"]) or {}
        text_hash: str = _text_hash(document["text"])
        if str(marker.get("hash", "")) == text_hash and str(marker.get("model", "")) == services.embeddings_model:
            continue
        _embed_document(config, services, document, text_hash)


def _embed_document(config: Neo4jConfig, services: ModelServicesConfig,
                    document: Dict[str, Any], text_hash: str) -> None:
    """Chunk, embed and store one document, replacing any previous version."""
    chunks: List[str] = _split_chunks(document["text"])

    write: Dict[str, Any] = run_write(
        config,
        "MATCH (chunk:Track2bChunk {documentId: $id}) DETACH DELETE chunk",
        {"id": document["id"]},
        timeout=STORE_TIMEOUT_SECONDS,
    )
    if not write["ok"]:
        _raise_store(write["error"])

    rows: List[Dict[str, Any]] = []
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch_texts: List[str] = chunks[start:start + EMBEDDING_BATCH_SIZE]
        vectors: List[List[float]] = _embed_texts(services, batch_texts)
        for offset, (chunk, vector) in enumerate(zip(batch_texts, vectors)):
            ordinal: int = start + offset
            rows.append({"id": _chunk_id(document["id"], ordinal), "ordinal": ordinal, "text": chunk, "embedding": vector})

    for start in range(0, len(rows), EMBEDDING_BATCH_SIZE):
        write = run_write(
            config,
            "UNWIND $rows AS row "
            "MERGE (chunk:Track2bChunk {id: row.id}) "
            "SET chunk.documentId = $id, chunk.documentName = $name, chunk.ordinal = row.ordinal, "
            "chunk.text = row.text, chunk.embedding = row.embedding, chunk.model = $model",
            {"id": document["id"], "name": document["name"], "model": services.embeddings_model, "rows": rows[start:start + EMBEDDING_BATCH_SIZE]},
            timeout=STORE_TIMEOUT_SECONDS,
        )
        if not write["ok"]:
            _raise_store(write["error"])

    write = run_write(
        config,
        "MERGE (state:Track2bRagState {documentId: $id}) "
        "SET state.hash = $hash, state.model = $model, state.chunks = $chunks, state.updatedAt = datetime()",
        {"id": document["id"], "hash": text_hash, "model": services.embeddings_model, "chunks": len(chunks)},
        timeout=STORE_TIMEOUT_SECONDS,
    )
    if not write["ok"]:
        _raise_store(write["error"])


def _ensure_vector_index(config: Neo4jConfig, dimensions: int) -> None:
    """One vector index over the submission's chunks; IF NOT EXISTS keeps it cheap."""
    statement: str = (
        f"CREATE VECTOR INDEX `{CHUNK_INDEX_NAME}` IF NOT EXISTS "
        "FOR (chunk:Track2bChunk) ON (chunk.embedding) "
        f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {int(dimensions)}, `vector.similarity_function`: 'cosine' }} }}"
    )
    write: Dict[str, Any] = run_write(config, statement, timeout=STORE_TIMEOUT_SECONDS)
    if not write["ok"]:
        _raise_store(write["error"])


def _vector_search(config: Neo4jConfig, query_vector: List[float]) -> List[Dict[str, Any]]:
    """Nearest-neighbour chunk candidates, scored server-side.

    Prefers the built-in vector index (`db.index.vector.queryNodes`, the name
    this store actually registers) and falls back to a brute-force cosine scan
    if the index is not queryable yet. The cosine scan mirrors the main Thalabus
    RAG fallback and is honest for a corpus this small. Only the reranker
    decides from here.
    """
    ann: Dict[str, Any] = run_read(
        config,
        f"CALL db.index.vector.queryNodes('{CHUNK_INDEX_NAME}', $top_k, $vector) "
        "YIELD node AS chunk, score "
        "RETURN chunk.id AS id, chunk.documentId AS documentId, chunk.documentName AS documentName, "
        "chunk.ordinal AS ordinal, chunk.text AS text, score "
        "ORDER BY score DESC LIMIT $result_limit",
        {"vector": query_vector, "top_k": VECTOR_TOP_K, "result_limit": RERANK_CANDIDATES},
        timeout=STORE_TIMEOUT_SECONDS,
    )
    if ann["ok"]:
        return _candidates_from_records(ann["records"])

    read: Dict[str, Any] = run_read(
        config,
        "MATCH (chunk:Track2bChunk) WHERE chunk.embedding IS NOT NULL "
        "WITH chunk, vector.similarity.cosine(chunk.embedding, $vector) AS score "
        "WHERE score IS NOT NULL "
        "RETURN chunk.id AS id, chunk.documentId AS documentId, chunk.documentName AS documentName, "
        "chunk.ordinal AS ordinal, chunk.text AS text, score "
        "ORDER BY score DESC LIMIT $result_limit",
        {"vector": query_vector, "result_limit": RERANK_CANDIDATES},
        timeout=STORE_TIMEOUT_SECONDS,
    )
    if not read["ok"]:
        _raise_store(read["error"])

    return _candidates_from_records(read["records"])


def _attach_display_names(config: Neo4jConfig, candidates: List[Dict[str, Any]]) -> None:
    """Give every candidate the human name of its document, plus a file name for the link.

    Chunk nodes store only the raw ingestion name — for platform-ingested PDFs that is a
    content-hash filename no reader can parse. The document carries a generated `title`
    whenever one exists, so citations prefer the title and fall back to the name, and the
    raw file name travels separately as the target of the citation's link (the same route
    the Documents tab uses to open sample files inline). Resolved at read time, so a
    renamed or re-titled document is current without re-embedding anything, and a store
    hiccup here never costs the answer — the candidate keeps its stored name.
    """
    if not candidates:
        return

    document_ids: List[str] = sorted({candidate["document_id"] for candidate in candidates})
    read: Dict[str, Any] = run_read(
        config,
        "UNWIND $ids AS document_id "
        "OPTIONAL MATCH (document:Document {id: document_id}) "
        "RETURN document_id AS id, document.title AS title, "
        "       coalesce(document.name, document.FileName, document.label) AS file_name",
        {"ids": document_ids},
        timeout=STORE_TIMEOUT_SECONDS,
    )
    if not read["ok"]:
        return

    display: Dict[str, Dict[str, Any]] = {str(record["id"]): record for record in read["records"]}
    for candidate in candidates:
        record: Dict[str, Any] = display.get(candidate["document_id"]) or {}
        title: str = str(record.get("title") or "").strip()
        file_name: str = str(record.get("file_name") or "").strip()
        candidate["document_caption"] = title or file_name or candidate["document_name"]
        candidate["document_file_name"] = file_name or None


def _candidates_from_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Shape raw graph records into the candidate rows the reranker consumes."""
    return [
        {
            "id": str(record["id"]),
            "document_id": str(record["documentId"]),
            "document_name": str(record.get("documentName") or "unknown document"),
            "ordinal": record.get("ordinal"),
            "text": str(record["text"]),
            "vector_score": float(record["score"]),
            "rerank_score": 0.0,
        }
        for record in records
        if record.get("id") and record.get("text")
    ]


def _rerank(services: ModelServicesConfig, question: str,
            candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score every candidate with the cross-encoder; an unanswered candidate keeps 0."""
    body: Dict[str, Any] = {
        "model": services.reranker_model,
        "query": question,
        "documents": [candidate["text"] for candidate in candidates],
    }
    response: Dict[str, Any] = _post_json(_service_endpoint(services.reranker_url, "rerank"), body, services.reranker_api_key, "reranker")
    results = response.get("results") or response.get("data") or []
    if not isinstance(results, list):
        raise _ServiceError("reranker_malformed_rerank", "the reranker returned no results array")

    for item in results:
        if not isinstance(item, dict):
            continue
        document: Any = item.get("document")
        index: Any = item.get("index")
        if index is None and isinstance(document, dict):
            index = document.get("index")
        score: Any = item.get("relevance_score", item.get("score"))
        if isinstance(index, int) and 0 <= index < len(candidates) and isinstance(score, (int, float)):
            candidates[index]["rerank_score"] = float(score)

    return candidates


def _build_context(passing: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Number the passing fragments, highest score first, up to the prompt budget."""
    parts: List[str] = []
    used: List[Dict[str, Any]] = []
    budget: int = 0
    for candidate in passing:
        block: str = f"[{len(parts) + 1}] (from: {candidate['document_name']})\n{candidate['text']}"
        if parts and budget + len(block) > CONTEXT_MAX_CHARS:
            break
        parts.append(block)
        used.append(candidate)
        budget += len(block)

    return "\n\n".join(parts), used


def _ask_over(llm: LlmConfig, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    """One bounded chat completion for the pair judge — same transport, tiny budget.

    Deliberately built as its own payload: the judge must not inherit the answer
    prompt's system text, its verbosity budget, or its sampling temperature.
    """
    payload: Dict[str, Any] = {
        "model": llm.name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    return _generate_payload(llm, payload)


def _merge_expanded_fragments(config: Neo4jConfig, services: ModelServicesConfig, question: str,
                              fragments: List[Dict[str, Any]],
                              expansion: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Submit the confirmed documents' chunks to the same reranker and floor.

    Expanded documents earn no privilege: a mode is not a filter, so their
    fragments pass exactly when a seed fragment would. Documents with no
    indexed fragments (never embedded, or embedded after the last question)
    are recorded truthfully — a citation they cannot support is not hidden.
    """
    entered_by: Dict[str, str] = {}
    for item in expansion.get("documents") or []:
        entered_by.setdefault(str(item["document_id"]), str(item.get("entered_by") or "expansion"))
    document_ids: List[str] = list(entered_by)
    known_chunks: set = {str(fragment["id"]) for fragment in fragments}

    read: Dict[str, Any] = run_read(
        config,
        "UNWIND $ids AS document_id "
        "MATCH (chunk:Track2bChunk {documentId: document_id}) "
        "RETURN chunk.id AS id, chunk.documentId AS documentId, chunk.documentName AS documentName, "
        "chunk.ordinal AS ordinal, chunk.text AS text, 0.0 AS score",
        {"ids": document_ids},
        timeout=STORE_TIMEOUT_SECONDS,
    )
    if not read["ok"]:
        detail: str = str((read.get("error") or {}).get("message") or "statement failed")
        expansion["notes"].append(f"Expanded fragments could not be read ({detail}); the answer stays with the seeds.")

        return fragments

    extra: List[Dict[str, Any]] = [candidate for candidate in _candidates_from_records(read["records"])
                                   if str(candidate["id"]) not in known_chunks]
    _attach_display_names(config, extra)
    if not extra:
        for document_id in document_ids:
            expansion["notes"].append(f"{document_id} has no indexed fragments — it cannot contribute citations.")

        return list(fragments)

    extra = _rerank(services, question, extra)
    passing_extra: List[Dict[str, Any]] = [candidate for candidate in extra
                                          if candidate["rerank_score"] >= MIN_RERANK_SCORE]
    kept: List[Dict[str, Any]] = []
    per_document: Dict[str, int] = {}
    for candidate in sorted(passing_extra, key=lambda item: item["rerank_score"], reverse=True):
        document_id = str(candidate["document_id"])
        if per_document.get(document_id, 0) >= EXPANDED_FRAGMENTS_PER_DOCUMENT:
            continue
        per_document[document_id] = per_document.get(document_id, 0) + 1
        candidate["entered_by"] = entered_by.get(document_id, "expansion")
        kept.append(candidate)
    passed_documents: set = {str(candidate["document_id"]) for candidate in kept}
    for document_id in document_ids:
        if document_id not in passed_documents:
            expansion["notes"].append(f"{entered_by.get(document_id, document_id)}: no fragment of this document cleared the 50% floor — it is reported as an expansion, not cited.")
    expansion["fragments_passed"] = len(kept)

    return sorted(list(fragments) + kept,
                  key=lambda fragment: (float(fragment["rerank_score"]),
                                         1 if str(fragment.get("entered_by") or "seed") == "seed" else 0),
                  reverse=True)


def _generate(llm: LlmConfig, question: str, context: str) -> str:
    """One chat completion over the prepared context, retried while the failure looks transient."""
    payload: Dict[str, Any] = {
        "model": llm.name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context fragments:\n\n{context}\n\nQuestion: {question}"},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }

    return _generate_payload(llm, payload)


def _generate_payload(llm: LlmConfig, payload: Dict[str, Any]) -> str:
    """Run one prepared payload under the shared transient-failure retry policy."""
    if not llm.configured:
        raise _ServiceError("llm_not_configured", "LLM_NAME/LLM_BASE_URL/LLM_API_KEY are not all set.")

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if llm.api_key:
        headers["Authorization"] = f"Bearer {llm.api_key}"

    answer: str = ""
    last_failure: Optional[_ServiceError] = None
    started: float = time.perf_counter()

    for attempt in range(1, GENERATION_ATTEMPTS + 1):
        try:
            answer = _request_completion(llm, payload, headers)
            last_failure = None
            break
        except _ServiceError as failure:
            last_failure = failure
            elapsed: float = _elapsed_ms(started) / 1000
            out_of_time: bool = elapsed >= GENERATION_DEADLINE_SECONDS
            if not _is_transient(failure) or attempt == GENERATION_ATTEMPTS or out_of_time:
                raise _ServiceError(failure.error_type, f"{failure.message} (after {attempt} attempt{'s' if attempt > 1 else ''}, {elapsed:.0f}s).") from failure
            time.sleep(GENERATION_BACKOFF_SECONDS * attempt)

    if last_failure is not None:
        raise last_failure

    return answer


def _is_transient(failure: _ServiceError) -> bool:
    """Whether the same request could plausibly succeed a moment later."""
    error_type: str = failure.error_type
    transient: bool = error_type in TRANSIENT_GENERATION_ERRORS
    if error_type.startswith("llm_http_"):
        transient = error_type in {"llm_http_429", "llm_http_500", "llm_http_502", "llm_http_503", "llm_http_504"}

    return transient


def _apertus_messages(payload: Dict[str, Any]) -> None:
    """Enforce the apertus chat-template rule on a payload, in place.

    The template accepts "system" only as the first message; every later one is
    resent as "user", exactly as the platform's own provider does
    (`src/core/llm/providers/provider_apertus.py`). A later system role makes the
    gateway reject the whole request with HTTP 400. The payload built here already
    obeys the rule; this guard keeps it true if the prompt ever grows.
    """
    sanitized: List[Dict[str, Any]] = []
    seen_system: bool = False
    for message in payload.get("messages") or []:
        if message.get("role") == "system" and seen_system:
            sanitized.append({**message, "role": "user"})
        else:
            seen_system = seen_system or message.get("role") == "system"
            sanitized.append(message)
    payload["messages"] = sanitized


def _request_completion(llm: LlmConfig, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
    """One chat completion request, typed errors included."""
    _apertus_messages(payload)
    try:
        with httpx.Client(timeout=LLM_TIMEOUT_SECONDS) as client:
            response: httpx.Response = client.post(f"{llm.base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
    except httpx.TimeoutException as failure:
        raise _ServiceError("llm_timeout", f"The chat model did not answer within {LLM_TIMEOUT_SECONDS:.0f}s.") from failure
    except httpx.RequestError as failure:
        raise _ServiceError(type(failure).__name__, str(failure)) from failure

    if response.status_code != 200:
        raise _ServiceError(f"llm_http_{response.status_code}", _error_text(response))

    try:
        body: Dict[str, Any] = response.json()
        content: Any = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as failure:
        raise _ServiceError("llm_malformed_response", str(failure)) from failure
    if not isinstance(content, str) or not content.strip():
        raise _ServiceError("llm_empty_reply", "the chat model returned an empty reply.")

    return content.strip()


def _embed_texts(services: ModelServicesConfig, texts: List[str]) -> List[List[float]]:
    """Embed one batch of texts and return the vectors in request order."""
    body: Dict[str, Any] = {"model": services.embeddings_model, "input": texts}
    response: Dict[str, Any] = _post_json(_service_endpoint(services.embeddings_url, "embeddings"), body, services.embeddings_api_key, "embeddings")

    vectors = response.get("data")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise _ServiceError("embeddings_malformed", "the embeddings service returned an unexpected data array.")

    by_index: Dict[int, List[float]] = {}
    for position, item in enumerate(vectors):
        if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
            raise _ServiceError("embeddings_malformed", "an embedding vector was missing.")
        vector: List[float] = [float(value) for value in item["embedding"]]
        by_index[int(item["index"]) if isinstance(item.get("index"), int) else position] = vector

    if any(position not in by_index for position in range(len(texts))):
        raise _ServiceError("embeddings_malformed", "the embeddings service skipped an input.")

    return [by_index[position] for position in range(len(texts))]


def _post_json(url: str, payload: Dict[str, Any], api_key: str, service: str) -> Dict[str, Any]:
    """One synchronous JSON POST shared by both retrieval services."""
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=MODEL_SERVICES_TIMEOUT_SECONDS) as client:
            response: httpx.Response = client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as failure:
        raise _ServiceError(f"{service}_timeout", f"{service} did not answer within {MODEL_SERVICES_TIMEOUT_SECONDS:.0f}s.") from failure
    except httpx.RequestError as failure:
        raise _ServiceError(f"{service}_{type(failure).__name__}", str(failure)) from failure

    if response.status_code != 200:
        raise _ServiceError(f"{service}_http_{response.status_code}", _error_text(response))
    try:
        body: Dict[str, Any] = response.json()
    except ValueError as failure:
        raise _ServiceError(f"{service}_malformed_json", str(failure)) from failure

    return body


def _error_text(response: httpx.Response) -> str:
    """Extract the service's own error message from a failed response, best effort."""
    try:
        body: Dict[str, Any] = response.json()
    except ValueError:
        return (response.text or "")[:400]
    error: Any = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return str(error["message"])[:400]
    if isinstance(error, str):
        return error[:400]
    return str(body)[:400]


def _service_endpoint(base_url: str, route: str) -> str:
    """Absolute route URL that tolerates a configured base already ending in the route."""
    endpoint: str = base_url.rstrip("/")
    if endpoint.rsplit("/", 1)[-1].lower() != route.lower():
        endpoint = f"{endpoint}/{route}"

    return endpoint


def _split_chunks(text: str) -> List[str]:
    """Pack the document into index-sized fragments, paragraphs first, sentences next."""
    chunks: List[str] = []
    pending: str = ""
    for block in (part.strip() for part in _PARAGRAPH_SPLIT.split(text) if part.strip()):
        paragraph: str = block
        while len(paragraph) > MAX_CHUNK_CHARS:
            head: str
            head, paragraph = _cut_sentence_prefix(paragraph, CONTEXT_TARGET_CHARS)
            if not head:
                break
            if pending and len(pending) + len(head) + 2 <= MAX_CHUNK_CHARS:
                pending = f"{pending}\n\n{head}"
            else:
                if pending:
                    chunks.append(pending)
                pending = head

        if not paragraph:
            continue
        if pending and len(pending) + len(paragraph) + 2 <= MAX_CHUNK_CHARS:
            pending = f"{pending}\n\n{paragraph}"
        else:
            if pending:
                chunks.append(pending)
            pending = paragraph

    if pending:
        chunks.append(pending)

    return chunks


def _cut_sentence_prefix(text: str, target: int) -> Tuple[str, str]:
    """Split at the last sentence boundary near ``target``, else whitespace, else a hard cut."""
    head: str = text[:target]
    cut: int = 0
    for match in _SENTENCE_SPLIT.finditer(head):
        cut = match.end()
    too_early: bool = cut < target // 2
    if too_early:
        space: int = head.rfind(" ")
        if space > target // 2:
            cut = space
        elif len(text) > target:
            cut = target

    return text[:cut].strip(), text[cut:].strip()


def _citation(ordinal: int, candidate: Dict[str, Any]) -> Dict[str, Any]:
    """A citation numbers the fragment exactly as the model was shown it."""
    excerpt: str = " ".join(candidate["text"].split())
    if len(excerpt) > CITATION_EXCERPT_CHARS:
        excerpt = f"{excerpt[:CITATION_EXCERPT_CHARS].rstrip()}…"

    file_name: Optional[str] = candidate.get("document_file_name")

    return {
        "ordinal": ordinal,
        "document": candidate.get("document_caption") or candidate["document_name"],
        "document_id": candidate["document_id"],
        "url": f"/api/operations/documents/{quote(str(file_name))}" if file_name else None,
        "score": round(candidate["rerank_score"], 3),
        "vector_score": round(candidate["vector_score"], 3),
        "entered_by": str(candidate.get("entered_by") or "seed"),
        "excerpt": excerpt,
    }


def _chunk_id(document_id: str, ordinal: int) -> str:
    """Stable id so one document and position always merge onto the same node."""
    digest: str = hashlib.sha256(f"{document_id}|{ordinal}".encode("utf-8")).hexdigest()[:24]

    return f"chunkid-track2b-{digest}"


def _text_hash(text: str) -> str:
    """Content fingerprint deciding when a document must be re-embedded."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _raise_store(error: Optional[Dict[str, Any]]) -> None:
    """Turn one store failure into the pipeline error without swallowing it."""
    detail: Dict[str, Any] = error or {}

    raise _ServiceError(str(detail.get("type") or "store"), str(detail.get("message") or "Neo4j statement failed"))


def _elapsed_ms(started: float) -> int:
    """Whole milliseconds since a ``time.perf_counter()`` reading."""
    return round((time.perf_counter() - started) * 1000)
