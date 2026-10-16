# Chat Q&A — one question, one fixed RAG pass

The fifth Workspace tab (`src/app/operations/chat.py`, rendering in `chat_qa.py`) answers a single
question from the ingested corpus and cites the fragments it used. It runs the same stages the
platform's `RagService` runs inside its plan, flattened into one hard-coded sequence, with no
planner, no augmentation, and no agent loop.

The corpus is written by two producers with two different schemas — the wizard's local ingester
(`Status`, `Text`, `FileName`, plain text) and the platform's document ingestion (`status`, `text`,
`name`, HTML with hidden OCR and image-classification blocks). The pipeline reads
`coalesce(lowercase, Uppercase)` for every field and flattens HTML to plain text before chunking,
because reading only one spelling hides the other's whole corpus without raising any error.

## Stage sequence

`POST /api/operations/chat` (body `{"question": "...", "mode": "simple" | "advanced"}`;
`mode` absent means `simple`, an unknown `mode` is rejected with `400`) → `answer_question()` in
`src/app/operations/chat_rag.py`:

1. **Documents** — every `Document` whose `coalesce(status, Status)` is `Ready` or
   `NeedsCategorization` and whose `coalesce(text, Text)` is not empty is RAG-eligible
   (`_ready_documents`, `RAG_DOCUMENT_STATUSES`). HTML bodies are flattened by `_plain_text`
   first, so hidden OCR drafts and image-classification probes never reach an embedding or a
   citation. Chunk markers (`Track2bRagState`) are compared with the hash of that plain text, so
   an edited or re-parsed document is re-chunked and re-embedded on the next question while an
   unchanged one costs no embedding call (`_ensure_chunks_current`). A freshly widened corpus
   pays one warm-up: the first question after the eligibility fix embedded 45 documents
   (950 kB of text) in 40 s, every question after it in 0.2 s.
2. **Embed the question** — one batch to `EMBEDDINGS_URL` with `EMBEDDINGS_MODEL`; the vector
   count must match the text count or the request fails as `<service>_count_mismatch`.
3. **Vector search** — cosine ANN over `track2b_chunk_embeddings`, `VECTOR_TOP_K = 40` chunks
   across all documents. When the index is missing the store is asked for it once
   (`_ensure_vector_index`), then a brute-force cosine pass keeps the box usable while
   `timings_ms.index` records the fallback.
4. **Rerank** — the cross-encoder (`RERANKER_URL`, `RERANKER_MODEL`) re-scores the top
   `RERANK_CANDIDATES = 30` fragments for the exact question.
5. **Threshold** — fragments with `relevance_score >= MIN_RERANK_SCORE` (0.5) survive, packed
   into the prompt newest-first up to `CONTEXT_MAX_CHARS`.
6. **Generate** — one chat completion (`LLM_NAME` at `LLM_BASE_URL`) over exactly those
   fragments, which the model must cite as `[1]`, `[2]`, … The request is retried up to
   `GENERATION_ATTEMPTS` times when the failure looks transient (429/5xx, connection reset,
   timeout); anything else surfaces immediately, and the eventual message says how many
   attempts were spent.
7. **Name the sources** — `_attach_display_names` looks up each cited document once
   (`UNWIND $ids … OPTIONAL MATCH`) and prefers its generated `title`, falling back to its name.
   Platform-ingested PDFs are stored under a content-hash file name, so a citation that printed
   the raw name would be unreadable; the raw name is kept only as the link target
   (`/api/operations/documents/<name>`, the same inline-serving route the Documents tab uses).
   Read at answer time, so a re-titled document is correct in the next citation without
   re-embedding anything, and a store hiccup here costs nothing — the fragment keeps its stored
   name instead of losing its citation.

If nothing clears the threshold the box answers deterministically instead of inventing
something, quoting the best score it did see, and `answer_source` says which producer answered.
The endpoints it uses are the ones the wizard verifies before this tab opens (`docs/RUN_PLANS.md`).
## Search modes (T35)

The box carries a **Simple / Advanced** dropdown; the request body carries `mode`, and an absent
`mode` means `simple` — the shipped behaviour, reproduced identically. `advanced` adds one stage
between rerank and generate, implemented in `src/app/operations/chat_expansion.py`:

1. The documents that *passed the 50 % floor* become **seeds** (≤ 6, in rerank order).
2. Two affinity signals propose more documents: a **category ladder** over the ontology-210
   approved-value hierarchy (shared values rank candidates; fewer than 3 shared with any single
   seed is dropped, and the value ids are clipped to the 6 strongest), and a one-hop walk through
   shared typed **entities** (`:Entity` nodes, filtered to the documents' own ontology so a foreign
   tenant's graph cannot leak in; entity names/aliases up to 5 per candidate, hub entities over 8
   documents skipped with a visible note).
3. Entity-only candidates must pass a **co-mention gate**: the bridging name must occur verbatim
   (case-insensitively, up to 240 000 chars) in **both** document bodies — a shared extracted
   relationship older than the current text is never trusted on its own.
4. Every surviving (seed, candidate) pair is judged by the **same chat model** with profiles of
   both documents (question + first/last 900/420 chars each — the model reads profiles, never
   bodies, in the gate) in parallel, one strict reply: `MATCH: yes|no`, `REASON: <one line>`.
   A YES is persisted as a canonical-order `(Document)-[:RELATES {method, matched, match_reason,
   judge_model, entry, seed_document, shared_values, shared_category_ids, shared_entity_ids,
   evidenced_labels, created_at}]->(Document)` edge; a NO/co-mention rejection is returned to the
   UI **without** an edge. Existing `:RELATES` verdicts are reused — re-asking costs no judge call.
5. The enlarged set is **re-reranked with the same 50 % floor** (a mode is not a filter) and
   capped to the same per-document fragment limit. A document that enters but contributes no
   passing fragment yields an explicit note, not a silent citation.

Invariants: `ok`/`error` semantics unchanged; advanced **never removes** a seed citation the
simple mode would print; `expansion.ok: false` degrades to the seeds with a visible note; the
judge and the persistence layer can fail independently and both say so. Response fields:
`mode`, `expansion` (`{status, seeds, documents[], rejections[], notes[], cached, waves,
pair_checks, judge_model, timings_ms{expand, pair_judging}}`, null in simple mode) and per
citation `entered_by` (`"seed"` for the baseline path). The UI meta line reports expansion
counts, the first rejections with their reasons, and cached-verdict reuse.

Latency: worst case is (≤ 30 pairs) parallel judge calls plus one re-rank plus one generation.
**Measurement pending** — requires a morning run of the feature against the live system
(server off-limits during development); acceptance requires the number before demo.
The engine was validated offline: all Cypher compiled via `EXPLAIN` through the Neo4j REST API
and read-executed read-only (`tmp/tmp_t35_validate_cypher.py`), and the whole expansion path is
tested without a server or LLM in `tests/chat_expansion_unit_test.py`.
## Response shape

The endpoint always answers JSON. A successful turn carries `ok: true`, `answer`,
`answer_source` (`"llm"` or `"deterministic"`), `citations[]` (`ordinal`, `document`, `url`,
`score`, `vector_score`, `excerpt`), `candidates`, `passed_threshold`, and `timings_ms`
(`documents`, `embed_query`, `index`, `search`, `rerank`, `generate`). Failure turns carry
`ok: false` and `error: {type, message}`; the HTTP status names the stage:

| Status | Meaning |
| --- | --- |
| 400 | Empty or over-long question (the box cannot use it) |
| 409 | No ingested document yet, or the graph store is unreachable |
| 502 | Embeddings, reranker, or chat model failed; the message is the service's own |

The note line under the box reports the run, not the spinner: while asking it names the pipeline,
under an error turn it quotes the service's own message (so a gateway problem is visible without
opening server logs), and after an answer it returns to its resting hint with the summed
`timings_ms`, which is what an unexpectedly slow question costs.

## What to check when it complains

- **502 with a chat-model message** (`litellm...`, `Connection error`, `No fallback was
  attempted`) — retrieval already succeeded; the chat route in wizard step 2 is failing
  upstream. `docker exec <app> python -c "..."` or a direct `POST $LLM_BASE_URL/chat/completions`
  with the `.env` key distinguishes a route outage from a key problem; other model handles on
  the same gateway answer while one deployment is down.
- **No citations under a correct answer** — the model was asked to answer in one word, or the
  answer came from `answer_source: "deterministic"`, which by definition has no fragments.
- **`0 fragments retrieved`** — no ingested document carries text, or the store was reseeded.
- **Refusal with a low percentage** — genuinely correct behaviour: the corpus has no fragment
  about that topic. The Documents tab shows which corpora exist.
- **Chunking questions** — chunk ids are `<documentId>_cNNNN`, so an identical document
  re-parsed from the same bytes reuses its embeddings; changing the text changes the hash and
  forces one re-embed of that document only.

- **A refusal naming a percentage far below the real relevance** — the reranker scores the exact
  wording of the question, so a long multi-part interrogative can score a genuinely on-topic
  fragment at 0.03 while a noun phrase from the same document scores 0.97. Reorder the question
  as the document words it, or ask the parts separately; splitting the same document into finer
  chunks was measured and did not change the outcome.

- `docs/RUN_PLANS.md` — where the chat, embeddings and reranker endpoints come from per plan.
- `docs/CATEGORIZATION.md` — its sibling on the Knowledge tab, same endpoints and client
  conventions (typed `x_y` error codes, no `AbortController`, a note line that reports the
  failing stage instead of a spinner).
- `docs/KG_EXTRACTION.md` — the extraction whose output this box retrieves fragments from.
- `src/app/operations/chat_rag.py` — the implementation, stage for stage.
- `src/app/static/css/operations/chat_qa.css` — the box's rules, including `.chat-citation-list`
  (the class the rendered turns carry) and the transcript's share of the card height.
- `src/app/static/index.html` — the fetcher for this box, the citation renderer, and the note line
  that names the failing stage.
