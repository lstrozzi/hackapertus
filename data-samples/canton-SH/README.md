# Canton sample — Schaffhausen (`SH`), OpenParlData

Full single-canton reference corpus, pulled 2026-10-13 (daily provider batch). Licence: CC0-1.0 (OpenParlData). Local-only: this whole archive is git-ignored — `MANIFEST.md` carries size × sha256 per file.

## Why SH

Smallest-tier canton with a complete structured export **and** a populated original-document directory. German-only titles — ideal for the PDF→text extraction pipeline trials (T13/T14) whose reference corpus must be self-contained and re-pullable in minutes.

## Contents

| path | what | size / rows |
|---|---|---|
| `docs_SH.ndjson.gz` | complete documents table for the canton (from `exports/docs/`) | 40 MB gz · 4 451 rows |
| `news_SH.ndjson.gz` | complete news table for the canton (from `exports/news/`) | 1 414 rows |
| `slices/affairs.canton.ndjson.gz` | affairs of body 266 (`body_key 'SH'`) | 2 008 rows |
| `slices/agendas.canton.ndjson.gz` | agenda items | 864 rows |
| `slices/bodies.canton.ndjson.gz` | the canton parliament row | 1 row |
| `slices/groups.canton.ndjson.gz` | fractions/committees | 136 rows |
| `slices/memberships.canton.ndjson.gz` | person ⇄ group memberships | 1 493 rows |
| `slices/contributors.canton.ndjson.gz` | affair actors | 2 389 rows |
| `slices/persons.canton.ndjson.gz` | persons referenced above | 185 rows |
| `pdfs/` | 40 size-stratified original documents (3.4 KB – 3.7 MB, ≈ 19 MB) | 39 `.pdf` + 1 `.zip` |
| `pdf_listing.tsv` | full server listing of `/doc/SH/` (4 432 files, name × size × mtime) | — |

Note: `texts_SH` / `speeches_SH` shards do **not** exist — SH exposes no harmonized speech corpus via OpenParlData.

## Where the data lives (source links)

- Structured bulk: `https://files.openparldata.ch/exports/` → nested shards `exports/docs/docs_SH.ndjson.gz`, `exports/news/news_SH.ndjson.gz`; flat tables sliced locally from the T6 raw archive by `body_id = 266`.
- Original documents: `https://files.openparldata.ch/doc/SH/` — files are **extensionless content-hash names**; append `.pdf` to open. 39/40 sampled files are real PDFs (`%PDF`); one is a ZIP — always magic-byte-check before parsing.
- Every sampled PDF hash appears in a `docs_SH` row (metadata ⇄ file cross-link verified).

## Replay

```
python tests/scripts/t6_canton_sample_pull.py --canton SH --pdf-count 40
```

Idempotent: cached downloads are skipped, slices and `MANIFEST.md` are recomputed. Re-run ≤ 1 h before the build sprint (R6 — provider re-exports daily ≈ 04:00 GMT) for a fresh snapshot.
