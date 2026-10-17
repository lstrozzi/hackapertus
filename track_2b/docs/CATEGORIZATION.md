# Categorisation on the Knowledge tab

The three top boxes of the Knowledge tab are live. They implement the same skill
the Thalabus platform runs as `DocumentCategorizer`, re-implemented for this app:
one OpenAI-compatible chat call, strict ontology constraints, no platform imports.

| Box | Endpoint | Effect |
| --- | --- | --- |
| Category inference (03) | `POST /api/operations/categories/infer` | Proposes categories for one document and stores the proposal on the document node (`ProposedCategories` JSON property). Assignments are **not** written. |
| Categories (04) | `GET /api/operations/categories` | Read-only list of every `KeywordCategory` in the categorisation ontology: name, description, list of values, and how many documents carry each. |
| Categorization (05) | `POST /api/operations/categories/apply` | Runs the same skill and stores the result as graph assignments, then promotes the document `NeedsCategorization → Ready`. |

## The ontology

```
CATEGORY_ONTOLOGY_ID = "Parlament"   # src/app/operations/document_categorizer.py
```

Hard-coded by design, mirroring the KG metamodel box. The ontology is the single
source of admissible categories: the model is shown only its `KeywordCategory`
definitions (`description` + `lov`) and must answer with exact category names and
valid list-of-values entries. The reply is parsed defensively — invented
categories, values outside a `lov`, and prose around the JSON are handled, and
every rejected item is reported in the box rather than silently dropped.

```
(o:Ontology {id: 'Parlament'})-[:CONTAINS]->(k:KeywordCategory)   ← allowed names
(d:Document)-[:HAS_CATEGORY]->(c:Category {text})-[:OF_CATEGORY]->(k)   ← stored assignment
(d)-[:FOLLOWS]->(o)
```

`load_category_overview()` in `document_categorizer.py` reads the definitions and
the assigned-document counts; if a future environment renames the ontology, point
`CATEGORY_ONTOLOGY_ID` at the right id and nothing else changes.

## Ingestion status contract

Ingestion leaves a processed document at **`NeedsCategorisation`**, exactly like
the platform's `DocIngester`: only the categorisation skill promotes to `Ready`.
The inference box lists precisely the sample files whose **matched platform
document** has `status = NeedsCategorization` exactly; a platform status of any
other value, or no platform match (`n/a`), is not offered. The picker never reads
`knowledge_graph_status` (that tracks KG extraction, a different workflow) and never
the app-local ingest status.
The apply box accepts `NeedsCategorization` and `Ready` (the platform's own input
contract); its document list labels show the same categorisation status — the
matched platform document's status when there is one, the local ingestion status
only for files with no platform match (e.g. `Failed`), so a freshly provided
sample never reads `NotYetProcessed` just because this app has not ingested it.
The Documents table's **Platform status** column reads the matched platform
document's `status` property, and falls back to this app's own ingestion status when
no platform document matches — the same rule the apply-box labels use, because under
Plan SaaS without an instance token no sample matches at all, and a bare `n/a` would
hide a `Failed` this app recorded along with its reason. That reason, written to
`Document.Error` by the ingester, is delivered as `status_error` in the same payload
and rendered beneath the status. Its summary counters are separate: they report this
app's local ingestion progress, not the platform's workflow status.

A document selected from that list is proposed **from whichever graph already has
the text**: the local record when this app ingested it, otherwise the matched
platform document (sample PDFs arrive with their platform counterpart ingested,
while the Track 2B record is still `NotYetProcessed` and textless). Proposal stays
proposal-only either way: the categories are stored as `ProposedCategories` on this
app's own document node, the platform document keeps its status and gains nothing,
and only the explicit apply edge-writes and promotes (locally).
`tests/categorization_unit_test.py` covers this contract offline (fake stores, no
LLM): the exact-status rule, local-first selection, the no-match and missing-text
rejections, and that a propose run never applies or promotes.

Both boxes address a document through the store's resolved identity, not a derived
key: proposals and assignments go to the node the synchroniser adopted, so a sample
whose text lives on a platform `docid-N` node is categorised on that node rather
than on a second node for the same file (see *Document identity* in the README and
`tests/document_identity_unit_test.py`).

## End-to-end run (verified against the dev graph)

1. Ingest a file on the Documents tab → pill turns `NeedsCategorization`.
2. Knowledge tab → the document is selectable in **Category inference** →
   "Propose categories" → the proposal table renders and survives reloads.
3. Select the document once in **Categorization** → "Categorize document" →
   `HAS_CATEGORY`/`FOLLOWS` edges appear, status pill → `Ready`, Categories
   counters increase, and the document disappears from the inference picker.

Run on 2026-10-15 against the platform dev graph (SaaS profile): a re-ingested
council-decisions PDF proposed 6 categories; one invalid `DECISION_BODY` value was
rejected with its reason shown in the box, the six valid proposals stored verbatim.
Applying the same document wrote the assignment edges, added `FOLLOWS`, promoted
the document to `Ready` (verified in the graph: 6 `HAS_CATEGORY` edges on
`docid-track2b-c2897cce1f794db44cef16d2`). Counts shown for a shared database also
reflect documents of the platform's own demo dossiers.

> Note (2026-10-16): that hash-derived id no longer resolves. It was one of the duplicate copies
> removed when the store started addressing a file by the node that already carries it (see
> *Document identity* in the README), so the file is now represented by its platform `docid-N`
> node and the edges above are recorded against the id the panel shows. The run itself is
> unchanged; only the address it was written to changed.

## LLM configuration

One call with `max_tokens=1600, temperature=0`, 180 s timeout; the prompt (in
`_system_prompt`) embeds the full definition list, the strict-JSON contract, and an
empty-categories escape hatch. If the configured model is unreachable, the endpoints
answer 502/504 with the gateway message and the box shows it as a visible error —
never as a silent empty result. The document is capped at 60 000 characters
(`MAX_DOCUMENT_CHARS`).

## Ontology extension (box 06)

`category_extension.py` copies the platform's
`DocumentCategorizer.ask_llm_for_category_proposals` skill
(`propose_new_categories: true`) into this submission's fixed `Parlament` ontology.
Same source-document eligibility, same LLM gateway, a condensed propose prompt, and
one storage difference: a reply is filed as **category definitions**, never as an
assignment —

- `POST /api/operations/categories/extend` (body `{"document"}`) asks the LLM for
  3–10 new categories, validates each (`text_en` + `description`, no clash with any
  existing name incl. flagged ones, batch-deduped, cap 10), and creates the
  survivors as `(:Ontology {id:"Parlament"})-[:CONTAINS]->(:KeywordCategory {id:
  "keyword_cat_…", text_en, description, proposed: true, …})`. Translations
  (`text_fr/de/it`) and an optional comma-joined `lov` pass through when present.
  A definition whose name exists is skipped, never updated — the response reports
  `saved`, `skipped`, the persisted `proposals` (with ids) and each rejection.
- A flagged category is **inert**: the categorisation menu (inference prompts and
  validators) filters `coalesce(cat.proposed, false) = false`, while inspection shows
  all of them with a `proposed` badge and an Approve button.
- `POST /api/operations/categories/approve` (body `{"category_id"}`) is the adoption
  step: it requires `proposed = true`, sets only `proposed = false`, touches no edge
  or other property, and answers 404 for anything not flagged.
- The document node is never touched by either call: same status, same proposals,
  same edges — proposals extend the ontology for review, nothing more.

Tests: `tests/category_extension_unit_test.py` (offline, stubbed graph and LLM),
plus `tests/categorization_unit_test.py` for the unchanged categorisation core.
Run both through the submission's own interpreter.

Both writes read `run_write`'s summary counters (`nodes_created`, `properties_set`),
not rows: `run_write` never returns a `records` key, so write statements carry no
`RETURN` — a stub that invents one hides the bug until the first live 500.

