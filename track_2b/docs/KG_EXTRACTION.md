# Knowledge graph extraction on the Knowledge tab

Box **07 | Extract** is live. It runs a re-implementation of the Thalabus platform
skill `KgFromDocument` inside this app — plain chat calls to the configured
OpenAI-compatible endpoint, strict ontology constraints, no platform imports —
and writes the result into the track 2B Neo4j store.

| Box | Endpoint | Effect |
| --- | --- | --- |
| Knowledge graph extraction (07) | `GET /api/operations/kg/extraction/documents` | Lists the documents the extraction accepts right now: every document of the source dossier (`SOURCE_DOSSIER_ID = "hackathon-temp"`, reached through `(:Dossier)-[:CONTAINS]->(:Document)`) with status `Ready` or `NeedsCategorization` and extracted text present. Each entry carries its graph `id`, which is what the dropdown uses as its option value. |
| | `POST /api/operations/kg/extraction` `{"document": "<document id>"}` | Starts one background extraction run (one run at a time; typed error codes, never a hang). The id is resolved with `read_document_by_id`, which coalesces the platform's lowercase `name`/`status`/`text` and this app's uppercase `FileName`/`Status`/`Text` shapes, so platform dossier nodes (`docid-NNN`) and legacy tracked nodes (`docid-track2b-*`) both work as sources. |
| | `GET /api/operations/kg/extraction/state` | Phase, percent, and the finished result — the box polls it, and the finished result survives page reloads. |

## The ontology

```
ONTOLOGY_ID = "Parlament"   # src/app/operations/kg_extraction_runner.py
```

and `SOURCE_DOSSIER_ID = "hackathon-temp"` beside it — the dossier whose
`CONTAINS`-linked documents feed the box's dropdown.

Hard-coded by design, exactly like the categorisation boxes. The extraction is
driven entirely by the ontology **as stored in the track 2B graph**: entity types
and their attributes, relationship types and their node-label end constraints,
read live for the prompts. The metamodel falls back to the bundled snapshot that
`kg_metamodel.py` uses for inspection when the ontology node is absent.

## The pipeline, as the skill runs it

One document is split into HTML-shaped chunks (`<p>` elements, ≤3 KB each; a
plain text without HTML tags is wrapped paragraph by paragraph, so chunking and
`source_id` provenance keep working). The declared entity types are fanned out
in parallel up to `MAX_PARALLELISM` workers, and inside each type call the
chunks fan out too; every in-flight chat request additionally passes a
`LLM_MAX_CONCURRENCY` semaphore inside `_Llm.complete`, so one mostly
single-chunk document no longer means one serial LLM call. Per entity type: a
discovery pass — with the community's existing entities offered for id reuse — a
deduplication pass over everything that chunk calls agreed on, then persistence.
Relationship types fan out in parallel the same way, after the entity phase
merged its results (results are returned in declaration order, so persistence
stays deterministic). The prompt receives the union of source- and
target-eligible entity IDs, and each returned endpoint is validated against its
own type's label constraint before persistence. Sending only source IDs made
cross-type relationships impossible because valid target IDs were absent from
the prompt. The reply contract is one JSON object; the box shows counts, entity
chips, relationship rows and every degradation note.

Writes: one `(o:Ontology)-[:DEFINES]->(c:Community:Dataset)` community — reusing
`DefaultCommunity_Parlament` when it exists and creating it otherwise — entities
as `(:Entity:`<Label>` {id, name})` with `AGGREGATES` from the community, a
`SOURCE` and an `IS_DOCUMENTED` edge (kind `extracted`, `created_by: track2b`) to
the document node, and the extracted relationships as typed edges with their
`source_id` provenance. Entity identity is unique by `(ontology_id, id)` across
labels: later discoveries add their ontology label and distinct name variants
to the existing node rather than creating a second graph node with the same ID.

## What degrades and how

One LLM call failing does not abort the run: the chunk (or, at the outer fan-out,
the whole ontology type) is skipped, the reason is recorded, and `notes` are
rendered under the result. A run where every call
fails therefore ends as `Done: 0 entities, 0 links` with the failure reasons
listed (observed live with the shared endpoint in cooldown: `HTTP 429 all
deployments in cooldown`). A single document is queued at a time; a second
`POST` while a run is live is answered `409`.

## One system message only (Apertus chat-template constraint)

The prompts attach optional context blocks (community entities already present,
document context, full ontology schema) as additional `system` messages — a
payload that is fine for generic OpenAI-compatible endpoints but the Apertus
chat template names `system` only as the **first** role, and the gateway answers
`HTTP 400 Invalid message role` for anything else. The transport therefore
normalises every outgoing call the same way the platform's own Apertus provider
does (`PythonServer/src/core/llm/providers/provider_apertus.py`):
`_sanitize_system_roles` keeps the first `system` message and resends each later
one as `user` (`_Llm.SECONDARY_SYSTEM_ROLE`). With that in place an extraction
against `https://llm.thalabus.ai/v1` completes with zero failure notes
(verified live 2026-10-15).

## Label constraints are strings, not lists (relationship phase)

The platform stores `entity_source_labels` / `entity_target_labels` as **plain
strings** — a single label (`"Affair"`) or comma-separated (`"Person,Animal"`,
the `KnowledgeGraphEditor` convention) — while the bundled snapshot ships real
lists. `list("Affair")` explodes into `['A', 'f', 'f', ...]`, so every
candidate/allowed-target filter matched nothing and the relationship phase
closed itself before the first prompt: many entities, zero relationships, zero
LLM calls (observed live 2026-10-15). `kg_metamodel.label_list()` normalises
both shapes (split on comma when a string, itemise when a list) and both the
runner loader and the metamodel read path go through it.

## Image markup steals the prompt (document text)

Ingested PDFs keep their text as HTML that inlines page renders as
`<img src=... alt=...>` — the rendered payload repeated in **both** attributes.
Unfiltered, `<DOCUMENT_CONTEXT>` became megabytes of base64-ish noise and some
calls came back empty or off-target. `_plain_text()` now strips `<img>` tags and
long `data:` URIs from the text **before** the `MAX_DOCUMENT_CHARS` slice (so
the budget holds prose, not images); verified live 2026-10-15.

## The model thinks before it answers (reasoning truncation)

A raw response dump exposed the real reason for the empty content: the
gateway's model reasons first — the reply arrives in `reasoning_content`, and
when `max_tokens` is exhausted **by the thinking itself** the visible
`content` is empty with `finish_reason: length` (one probe prompt spent its
whole 8 000-token budget on a 15 k-character inner monologue and still had not
emitted the JSON). A fixed retry with the same budget therefore fails twice —
the emptiness is deterministic, not flaky. `_Llm.complete` now treats
`finish_reason: length` as its own case: it retries with
`REASONING_ESCALATION_TOKENS` (16 000), and if the first truncated attempt
already wrote the JSON inside its reasoning, the answer is salvaged from the
**last** parseable object of the thinking text (earlier ones are prompt echoes
or examples). Only after three escalating attempts is a call skipped, with a
note (verified live 2026-10-15).

## Curly braces inside f-string concatenations (entity persistence)

Every Cypher the runner writes reaches the gateway as one string built by f-line
plus plain-string concatenation. Plain concatenated segments therefore must use
**single** braces: the first line of the entity-`MERGE` correctly escapes
`{{id: $entity_id}}` (it is an f-line), but the `IS_DOCUMENTED` relationship
map was written `{{kind: 'extracted'}}` inside a plain line, so the statement
reached Neo4j with `{{ }}`, failed with a `CypherSyntaxError`, and
`_persist_entity` returned `False` — every entity silently vanished while the
run still reported `Done: 0 entities, 0 links`. Single braces on that line;
relationship persistence verified live 2026-10-15.

## End-to-end run (observed against the dev graph)

1. Every document of the source dossier with status `Ready`/`NeedsCategorization`
   and text appears in the dropdown (verified live: 39 dossier documents listed
   on 2026-10-15); without one, the box says so and the button stays disabled.
2. Pick it, click **Extract to graph** → the pill streams the phases
   (`extracting Canton entities (chunk 0/1)` … `persisted community`), the page
   may be reloaded at any point and the box re-attaches to the running state
   (verified live after a mid-run reload).
3. Finished: `Done: N entities, M links`, community `DefaultCommunity_Parlament`,
   entities grouped by type, relationship rows, notes if anything degraded.
4. In the store: community node + `DEFINES` edge from `(:Ontology {id:'Parlament'})`,
   entities as `(:Entity:`<Label>` {id, name})` with `AGGREGATES` from the
   community and `SOURCE`/`IS_DOCUMENTED` to the document — a full live run of
   `docid-394` finished `Done: 9 entities` (Affair 1, Canton 1, Organisation 2,
   Person 1, Topic 4), zero failure notes (2026-10-15).
