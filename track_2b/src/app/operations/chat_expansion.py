"""Advanced-search expansion for the Chat Q&A box (T35).

The simple pipeline answers *where is it written*. This module serves the
harder question — *what happened* — by enlarging the seed set before
generation, following the plan recorded in
``Documentation/.../2026_10_HACKATHON.T35.md``:

1. Seeds are the documents whose fragments passed the rerank floor.
2. Two independent signals propose candidates for each seed:
   * the **category ladder** — documents sharing approved keyword values
     ``(doc)-[:HAS_CATEGORY]->(:Category)-[:OF_CATEGORY]->(:KeywordCategory)``,
     ordered by shared-value count, top-down from eight, floored at three;
   * the **shared-entity hop** — documents bridged through an extracted
     ``(entity)-[:SOURCE|IS_DOCUMENTED]->(document)`` node, walked only for
     entities specific enough (degree cap) **and whose name occurs verbatim in
     both bodies**. The co-mention rule is not decoration: the store already
     contains affair entities falsely claimed by unrelated documents (measured
     2026-10-15), so an unevidenced bridge may not move the answer.
3. Every surviving (seed, candidate) pair — profiles, never bodies — goes to a
   strict pair judge, run in parallel. A YES earns the candidate a place in the
   enlarged set; a NO is recorded as a visible rejection.
4. A YES is persisted as a ``(Document)-[:RELATES]->(Document)`` edge (written
   in canonical endpoint order, read undirected) carrying full provenance, so
   a repeat question reuses the verdict instead of paying for the judge again.
   The edge is an *inferred* link, kept apart from extraction output by
   ``method`` and by the evidence fields.

The store is only reached through the app's ``graphstore`` layer. The module
never embeds, reranks or generates: the caller injects ``ask`` (the pair
judge), and consumes a structured report. Failures land in ``notes`` or
``rejections`` — this stage degrades honestly, never silently.
"""

__all__ = ["expand_seeds"]

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..graphstore.neo4j import Neo4jConfig, run_read, run_write
from .document_store import CATEGORY_ONTOLOGY_ID

# The pair judge: (system_prompt, user_prompt) -> raw reply text. The caller
# owns the transport and the model; this module owns the prompts, the parsing,
# the parallelism, and the consequences.
PairAsk = Callable[[str, str], str]

# Ladder bounds: a candidate must share at least the floor in approved keyword
# values; ordering starts at the top rung, so strong pairs consume the budget
# first. The measured histogram says the ladder discriminates down to ~6; the
# floor exists to keep long-tail candidates admissible under the top-K cap.
EXPANSION_LADDER_TOP: int = 8
EXPANSION_MIN_SHARED_VALUES: int = 3

# Hard caps. Measured hub fan-out (33 neighbours at threshold 3) shows an
# uncapped expansion collapses to "half the corpus"; these are the brakes.
EXPANSION_MAX_CANDIDATES: int = 20
EXPANSION_MAX_DOCUMENTS: int = 12
EXPANSION_MAX_PAIRS: int = 60
# Waves of recursion from newly confirmed documents. 1 = seeds only; the loop
# supports the bounded second wave the plan allows for.
EXPANSION_MAX_WAVES: int = 1
EXPANSION_WAVES_CEILING: int = 2
# An entity documented by more documents than this is an institution, not a
# path (kantonsrat spans 16 documents: it connects everything, proves nothing).
EXPANSION_GENERIC_ENTITY_MAX_DOCS: int = 8

# The pair check presents profiles, never bodies — two full documents do not
# fit one prompt (they run past 200k chars), and feasibility made this a
# precondition.
EXPANSION_PROFILE_HEAD_CHARS: int = 900
EXPANSION_PROFILE_TAIL_CHARS: int = 420
EXPANSION_PROMPT_HEAD_CHARS: int = 640
EXPANSION_PROMPT_TAIL_CHARS: int = 280
# How far into a body the server-side co-mention CONTAINS scan reads.
EXPANSION_SCAN_CHARS: int = 240_000

# Bounding caps that hold no matter how the corpus grows.
EXPANSION_SHARED_IDS_SHOWN: int = 6
EXPANSION_ENTITIES_PER_CANDIDATE: int = 6
EXPANSION_LABELS_PER_ENTITY: int = 3
EXPANSION_MENTION_REFS_CAP: int = 400
EXPANSION_QUERY_ROW_CAP: int = 4000

STORE_TIMEOUT_SECONDS: float = 30.0
# Same formula the platform uses for LLM fan-out (src/config/constants.py),
# kept local because this submission configures itself.
EXPANSION_PARALLELISM: int = min(int((os.cpu_count() or 4) * 2.5), 12)

RELATES_METHOD: str = "category-affinity+llm-match"

JUDGE_SYSTEM_PROMPT: str = (
    "You are the document-pair judge inside the advanced search of a parliamentary Q&A system. "
    "A retrieval pass found document A directly relevant to the user's question; document B is an expansion "
    "candidate pulled in by category overlap and/or a shared extracted entity. "
    "Decide whether document B is about the SAME real-world affair as document A — one filing, motion or case "
    "seen at another stage (filed, debated, answered, budgeted) — or its direct continuation. "
    "Sharing only the canton, an institution, the language or a broad topic is NOT a match: the affair itself must be shared. "
    "The profile excerpts are your only evidence; co-mention lines state whether a bridging name occurs verbatim in each body. "
    "Reply in exactly two lines, no other text: 'MATCH: yes' or 'MATCH: no', then a line 'REASON: <why, max 25 words, name the deciding evidence>'."
)

_MATCH_RE = re.compile(r"MATCH\s*:?\s*(?:\[?\s*)?(yes|no)\b", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON\s*:\s*(.+)", re.IGNORECASE)


class _ExpansionError(Exception):
    """One store failure inside expansion — reported as a note, never a lost answer."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message


def expand_seeds(config: Neo4jConfig, statuses: Sequence[str], seed_ids: List[str], question: str,
                 ask: PairAsk, judge_model: str) -> Dict[str, Any]:
    """Enlarge the seed documents by category/entity affinity, under the caps.

    Returns a structured report — never raises:
      ``documents``  confirmed expansions (id, name, entered_by, reason, …)
      ``rejections`` rejected candidates, each with the visible reason why
      ``notes``      honest remarks (uncategorised seeds, skipped hubs, failures)
      ``cached``     pairs answered from stored :RELATES verdicts, judge-free
    """
    result: Dict[str, Any] = {"ok": True, "status": "expanded", "seeds": list(seed_ids),
                              "documents": [], "rejections": [], "notes": [], "cached": 0,
                              "waves": 0, "pair_checks": 0, "judge_model": judge_model,
                              "timings_ms": {}}
    if not seed_ids:
        result["status"] = "no_seeds"
        result["notes"].append("Nothing to expand from: no fragment passed the rerank floor.")

        return result

    visited: Set[str] = set(seed_ids)
    frontier: List[str] = list(seed_ids)
    waves: int = min(max(EXPANSION_MAX_WAVES, 1), EXPANSION_WAVES_CEILING)
    started: float = time.perf_counter()

    try:
        for wave in range(1, waves + 1):
            if not frontier or len(result["documents"]) >= EXPANSION_MAX_DOCUMENTS:
                break
            result["waves"] = wave
            _run_wave(config, statuses, frontier, visited, question, ask, judge_model, result)
            confirmed: List[str] = [str(item["document_id"]) for item in result["documents"]]
            visited.update(confirmed)
            frontier = [document_id for document_id in confirmed if document_id not in set(seed_ids)]

        if len(result["documents"]) >= EXPANSION_MAX_DOCUMENTS:
            result["notes"].append(f"Expansion stopped at the {EXPANSION_MAX_DOCUMENTS}-document cap.")
        if not result["documents"]:
            result["status"] = "no_candidates"
            if not any("expand" in note.lower() or "cap" in note.lower() for note in result["notes"]):
                result["notes"].append("The expansion found no candidate the checks could justify.")
    except _ExpansionError as failure:
        result["ok"] = False
        result["status"] = "failed"
        result["notes"].append("Expansion failed: " + failure.message + " — the answer is built from the seeds"
                               + (", plus the expansions confirmed before the failure." if result["documents"] else " alone."))
    finally:
        result["timings_ms"]["expand"] = _elapsed_ms(started)

    return result


def _run_wave(config: Neo4jConfig, statuses: Sequence[str], frontier: List[str], visited: Set[str],
              question: str, ask: PairAsk, judge_model: str, result: Dict[str, Any]) -> None:
    """One wave: signals → co-mention gate → cache → parallel pair judge → verdicts."""
    profiles: Dict[str, Dict[str, Any]] = {profile["id"]: profile for profile in _profiles(config, frontier)}
    unlabelled: List[str] = [document_id for document_id in frontier
                            if not (profiles.get(document_id) or {}).get("categories")]
    if unlabelled:
        result["notes"].append(
            f"{len(unlabelled)} seed document(s) carry no approved categories — the category ladder had nothing to walk from them.")

    candidates: Dict[str, Dict[str, Any]] = _merge_signals(
        config,
        _category_ladder_rows(config, statuses, frontier),
        _entity_bridge_rows(config, statuses, frontier),
        frontier,
        visited,
        result,
    )
    if not candidates:
        result["notes"].append("No candidate above the affinity floor was found for this wave.")

        return

    ordered: List[str] = sorted(candidates, key=lambda document_id: candidates[document_id]["priority"],
                                reverse=True)
    candidate_ids: List[str] = ordered[:EXPANSION_MAX_CANDIDATES]
    if len(ordered) > EXPANSION_MAX_CANDIDATES:
        result["notes"].append(f"{len(ordered) - EXPANSION_MAX_CANDIDATES} weaker candidates were cut by the top-{EXPANSION_MAX_CANDIDATES} cap.")

    profiles.update({profile["id"]: profile for profile in _profiles(config, candidate_ids)})
    _apply_co_mention_gate(config, candidates, candidate_ids, profiles, result)

    judged: List[Tuple[str, str]] = []
    for document_id in candidate_ids:
        record: Dict[str, Any] = candidates.get(document_id) or {}
        if not record or not record.get("admissible", True):
            continue
        for seed_id in record["via_seeds"]:
            if seed_id in frontier and len(judged) < EXPANSION_MAX_PAIRS:
                judged.append((seed_id, document_id))
    if len(judged) == EXPANSION_MAX_PAIRS:
        result["notes"].append(f"Pair checks were cut at the {EXPANSION_MAX_PAIRS}-pair cap.")

    cached: Dict[Tuple[str, str], Dict[str, Any]] = _cached_verdicts(config, judged)
    pending: List[Dict[str, str]] = []
    for (seed_id, document_id) in judged:
        pair: Tuple[str, str] = tuple(sorted((seed_id, document_id)))  # type: ignore[assignment]
        hit: Optional[Dict[str, Any]] = cached.get(pair)
        if hit is None:
            pending.append({"seed": seed_id, "candidate": document_id})
            continue
        shared = hit.get("shared_values")
        _confirm(config, result, candidates, profiles, seed_id, document_id, judge_model,
                 f"cached :RELATES verdict ({shared} shared value(s), method {hit.get('method') or RELATES_METHOD})")
        result["cached"] += 1

    if pending:
        result["pair_checks"] += len(pending)
        started_judging: float = time.perf_counter()
        verdicts: Dict[Tuple[str, str], Tuple[bool, str]] = _judge_pairs(pending, profiles, candidates,
                                                                         question, ask)
        result["timings_ms"]["pair_judging"] = _elapsed_ms(started_judging)
        for (seed_id, document_id), (matched, reason) in verdicts.items():
            if matched:
                _confirm(config, result, candidates, profiles, seed_id, document_id, judge_model, reason)
            else:
                _reject(result, candidates, profiles, document_id, f"pair check: {reason}")


# ── signals ────────────────────────────────────────────────────────────────────


def _profiles(config: Neo4jConfig, document_ids: List[str]) -> List[Dict[str, Any]]:
    """Title, assigned category ids and bounded head/tail excerpts per document."""
    if not document_ids:
        return []

    read: Dict[str, Any] = run_read(
        config,
        "UNWIND $ids AS document_id "
        "MATCH (document:Document {id: document_id}) "
        "OPTIONAL MATCH (document)-[:HAS_CATEGORY]->(:Category)-[:OF_CATEGORY]->(definition:KeywordCategory) "
        "RETURN document_id AS id, "
        "       coalesce(document.title, document.name, document.FileName, document.label, document_id) AS name, "
        "       left(coalesce(document.text, document.Text, ''), $head_chars) AS head, "
        "       right(coalesce(document.text, document.Text, ''), $tail_chars) AS tail, "
        "       collect(DISTINCT definition.id) AS categories",
        {"ids": document_ids, "head_chars": EXPANSION_PROFILE_HEAD_CHARS,
         "tail_chars": EXPANSION_PROFILE_TAIL_CHARS},
        timeout=STORE_TIMEOUT_SECONDS,
    )
    if not read["ok"]:
        raise _ExpansionError(str((read.get("error") or {}).get("message") or "profile query failed"))

    return [
        {
            "id": str(record["id"]),
            "name": str(record.get("name") or record["id"]),
            "head": str(record.get("head") or ""),
            "tail": str(record.get("tail") or ""),
            "categories": sorted({str(item) for item in (record.get("categories") or []) if item}),
        }
        for record in read["records"] if record.get("id")
    ]


def _category_ladder_rows(config: Neo4jConfig, statuses: Sequence[str], seed_ids: List[str]) -> List[Dict[str, Any]]:
    """Count approved keyword values shared between the seeds and other documents."""
    read: Dict[str, Any] = run_read(
        config,
        "UNWIND $seed_ids AS seed_id "
        "MATCH (seed:Document {id: seed_id})-[:HAS_CATEGORY]->(:Category)-[:OF_CATEGORY]->(seed_value:KeywordCategory) "
        "WITH seed_id, collect(DISTINCT seed_value.id) AS seed_values "
        "UNWIND seed_values AS seed_value_id "
        "MATCH (other:Document)-[:HAS_CATEGORY]->(:Category)-[:OF_CATEGORY]->(shared:KeywordCategory {id: seed_value_id}) "
        "WHERE other.id <> seed_id "
        "  AND coalesce(other.status, other.Status) IN $statuses "
        "  AND NOT coalesce(other.text, other.Text) IS NULL "
        "WITH other.id AS document_id, seed_value_id, collect(DISTINCT seed_id) AS via_seeds "
        "WITH document_id, count(seed_value_id) AS shared_values, "
        "     collect(seed_value_id)[0..$show] AS shared_ids, collect(via_seeds) AS seed_lists "
        "RETURN document_id, shared_values, shared_ids, seed_lists "
        "ORDER BY shared_values DESC, document_id ASC "
        "LIMIT $row_cap",
        {"seed_ids": seed_ids, "statuses": list(statuses), "show": EXPANSION_SHARED_IDS_SHOWN,
         "row_cap": EXPANSION_QUERY_ROW_CAP},
        timeout=STORE_TIMEOUT_SECONDS,
    )
    if not read["ok"]:
        raise _ExpansionError(str((read.get("error") or {}).get("message") or "category ladder query failed"))

    return [
        {
            "document_id": str(record["document_id"]),
            "shared_values": int(record.get("shared_values") or 0),
            "shared_ids": [str(item) for item in (record.get("shared_ids") or []) if item],
            "via_seeds": sorted({str(seed) for group in (record.get("seed_lists") or []) for seed in group if seed}),
        }
        for record in read["records"]
        if record.get("document_id") and int(record.get("shared_values") or 0) >= EXPANSION_MIN_SHARED_VALUES
    ]


def _entity_bridge_rows(config: Neo4jConfig, statuses: Sequence[str], seed_ids: List[str]) -> List[Dict[str, Any]]:
    """(seed, other, entity) triples bridged by entity evidence edges."""
    read: Dict[str, Any] = run_read(
        config,
        "UNWIND $seed_ids AS seed_id "
        "MATCH (seed:Document {id: seed_id})<-[:SOURCE|IS_DOCUMENTED]-(entity:Entity) "
        "WHERE entity.ontology_id = $ontology_id "
        "WITH DISTINCT seed_id, entity "
        "MATCH (entity)-[r:SOURCE|IS_DOCUMENTED]->(other:Document) "
        "WHERE other.id <> seed_id "
        "  AND coalesce(other.status, other.Status) IN $statuses "
        "  AND NOT coalesce(other.text, other.Text) IS NULL "
        "RETURN seed_id, other.id AS document_id, entity.id AS entity_id, entity.name AS raw_name "
        "LIMIT $row_cap",
        {"seed_ids": seed_ids, "statuses": list(statuses), "ontology_id": CATEGORY_ONTOLOGY_ID,
         "row_cap": EXPANSION_QUERY_ROW_CAP},
        timeout=STORE_TIMEOUT_SECONDS,
    )
    if not read["ok"]:
        raise _ExpansionError(str((read.get("error") or {}).get("message") or "entity bridge query failed"))

    return [
        {
            "seed": str(record["seed_id"]),
            "document_id": str(record["document_id"]),
            "entity_id": str(record["entity_id"]),
            "labels": _as_string_list(record.get("raw_name")),
        }
        for record in read["records"]
        if record.get("document_id") and record.get("entity_id")
    ]


def _entity_degrees(config: Neo4jConfig, entity_ids: List[str]) -> Dict[str, int]:
    """Documents per bridging entity — the generic-institution filter."""
    if not entity_ids:
        return {}

    read: Dict[str, Any] = run_read(
        config,
        "UNWIND $entity_ids AS entity_id "
        "MATCH (entity:Entity {id: entity_id, ontology_id: $ontology_id}) "
        "-[:SOURCE|IS_DOCUMENTED]->(document:Document) "
        "RETURN entity_id, count(DISTINCT document.id) AS degree",
        {"entity_ids": entity_ids, "ontology_id": CATEGORY_ONTOLOGY_ID},
        timeout=STORE_TIMEOUT_SECONDS,
    )
    if not read["ok"]:
        raise _ExpansionError(str((read.get("error") or {}).get("message") or "entity degree query failed"))

    return {str(record["entity_id"]): int(record.get("degree") or 0) for record in read["records"]}


def _mention_matrix(config: Neo4jConfig, refs: List[Dict[str, str]]) -> Dict[Tuple[str, str, str], bool]:
    """Server-side co-mention evidence for (document, entity, lowercase label).

    ``refs`` entries carry ``document_id``, ``entity_id`` and an already
    lower-cased ``label``. The scan runs server-side and is capped at
    EXPANSION_SCAN_CHARS, so evidence costs booleans over the wire, never
    document bodies.
    """
    if not refs:
        return {}

    read: Dict[str, Any] = run_read(
        config,
        "UNWIND $refs AS ref "
        "MATCH (document:Document {id: ref.document_id}) "
        "WITH ref, left(coalesce(document.text, document.Text, ''), $scan_limit) AS body "
        "RETURN ref.document_id AS document_id, ref.entity_id AS entity_id, ref.label AS label, "
        "       toLower(body) CONTAINS ref.label AS evidenced",
        {"refs": refs, "scan_limit": EXPANSION_SCAN_CHARS},
        timeout=STORE_TIMEOUT_SECONDS,
    )
    if not read["ok"]:
        raise _ExpansionError(str((read.get("error") or {}).get("message") or "mention matrix query failed"))

    return {
        (str(record["document_id"]), str(record["entity_id"]), str(record["label"])):
            bool(record.get("evidenced"))
        for record in read["records"]
    }


def _as_string_list(value: Any) -> List[str]:
    """Normalise ``Entity.name``: extraction stores it as a string or an alias list."""
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, str) and item]

    return []


# ── candidate assembly ─────────────────────────────────────────────────────────


def _merge_signals(config: Neo4jConfig, ladder_rows: List[Dict[str, Any]],
                   bridge_rows: List[Dict[str, Any]], frontier: List[str],
                   visited: Set[str], result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Fold both signals into one prioritised candidate record per document."""
    candidates: Dict[str, Dict[str, Any]] = {}

    for row in ladder_rows:
        document_id: str = str(row["document_id"])
        if document_id in visited:
            continue
        record: Dict[str, Any] = candidates.setdefault(document_id, _blank_candidate(document_id))
        record["shared_values"] = int(row["shared_values"])
        record["shared_ids"] = list(row["shared_ids"])[:EXPANSION_SHARED_IDS_SHOWN]
        via: List[str] = [seed for seed in row["via_seeds"] if seed in frontier]
        record["via_seeds"] = via if via else [seed for seed in row["via_seeds"] if seed][:1]

    bridge_seeds: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in bridge_rows:
        document_id = str(row["document_id"])
        if document_id in visited:
            continue
        record = candidates.setdefault(document_id, _blank_candidate(document_id))
        key: Tuple[str, str] = (document_id, str(row["entity_id"]))
        entity: Dict[str, Any] = bridge_seeds.setdefault(
            key, {"entity_id": str(row["entity_id"]), "labels": list(row["labels"]),
                  "via_seeds": [], "degree": 0, "evidence_label": ""})
        if row["seed"] in frontier and row["seed"] not in entity["via_seeds"]:
            entity["via_seeds"].append(str(row["seed"]))
        if len(record["entities"]) < EXPANSION_ENTITIES_PER_CANDIDATE and entity not in record["entities"]:
            record["entities"].append(entity)

    for record in candidates.values():
        record["via_seeds"] = sorted({seed for entity in record["entities"] for seed in entity["via_seeds"]}
                                     | set(record["via_seeds"]))

    bridging: List[str] = sorted({entity["entity_id"] for record in candidates.values()
                                  for entity in record["entities"]})
    degrees: Dict[str, int] = {}
    if bridging:
        try:
            degrees = _entity_degrees(config, bridging)
        except _ExpansionError as failure:
            result["notes"].append(f"Entity degree check failed ({failure.message}); the hub cap was skipped.")
    if degrees:
        hubs: Set[str] = {entity_id for entity_id, degree in degrees.items()
                          if degree > EXPANSION_GENERIC_ENTITY_MAX_DOCS}
        if hubs:
            result["notes"].append(
                f"{len(hubs)} bridging entities were skipped as institutional hubs "
                f"(documented by more than {EXPANSION_GENERIC_ENTITY_MAX_DOCS} documents).")
        for record in candidates.values():
            kept: List[Dict[str, Any]] = []
            for entity in record["entities"]:
                entity["degree"] = int(degrees.get(entity["entity_id"], 0))
                if entity["entity_id"] not in hubs:
                    kept.append(entity)
            record["entities"] = kept
            record["via_seeds"] = sorted({s for e in record["entities"] for s in e["via_seeds"]}
                                         | (set(record["via_seeds"]) if record["shared_values"] else set()))
        for document_id in [key for key, record in candidates.items()
                            if not record["shared_values"] and not record["entities"]]:
            del candidates[document_id]

    for record in candidates.values():
        if record["shared_values"]:
            record["priority"] = 100.0 + record["shared_values"]

    return candidates


def _blank_candidate(document_id: str) -> Dict[str, Any]:
    """A candidate slot with every field spelled out once."""
    return {"document_id": document_id, "shared_values": 0, "shared_ids": [], "entities": [],
            "via_seeds": [], "priority": 0.0, "admissible": True, "admission_note": "",
            "evidenced_entities": [], "missing_labels": []}


def _apply_co_mention_gate(config: Neo4jConfig, candidates: Dict[str, Dict[str, Any]],
                           candidate_ids: List[str], profiles: Dict[str, Dict[str, Any]],
                           result: Dict[str, Any]) -> None:
    """Entity-only candidates must show the bridging name in both current bodies.

    Candidates with approved category assignments skip the gate: an approved
    assignment is already a curated signal. This is the rule that keeps the
    two 2016 documents off the Capaul answer (measured 2026-10-15): they
    bridge through the affair entity, whose name appears in neither body.
    """
    gate_ids: List[str] = [document_id for document_id in candidate_ids
                           if not candidates[document_id]["shared_values"] and candidates[document_id]["entities"]]
    refs: List[Dict[str, str]] = []
    for document_id in gate_ids:
        record: Dict[str, Any] = candidates[document_id]
        for entity in record["entities"]:
            for label in entity["labels"][:EXPANSION_LABELS_PER_ENTITY]:
                refs.append({"document_id": document_id, "entity_id": entity["entity_id"], "label": label.lower()})
                for seed_id in entity["via_seeds"]:
                    refs.append({"document_id": seed_id, "entity_id": entity["entity_id"], "label": label.lower()})
    if not refs:
        return
    if len(refs) > EXPANSION_MENTION_REFS_CAP:
        refs = refs[:EXPANSION_MENTION_REFS_CAP]

    matrix: Dict[Tuple[str, str, str], bool] = _mention_matrix(config, refs)
    for document_id in gate_ids:
        record = candidates[document_id]
        evidenced: List[Dict[str, Any]] = []
        missing: List[str] = []
        for entity in record["entities"]:
            hit: str = ""
            for label in entity["labels"][:EXPANSION_LABELS_PER_ENTITY]:
                in_candidate: bool = matrix.get((document_id, entity["entity_id"], label.lower()), False)
                in_seed: bool = any(matrix.get((seed_id, entity["entity_id"], label.lower()), False)
                                    for seed_id in entity["via_seeds"])
                if in_candidate and in_seed:
                    hit = label
                    break
            if hit:
                entity["evidence_label"] = hit
                evidenced.append(entity)
            elif entity["labels"]:
                missing.append(entity["labels"][0])
        record["evidenced_entities"] = evidenced
        record["missing_labels"] = missing[:3]

    for document_id in [key for key in gate_ids if not candidates[key]["evidenced_entities"]]:
        record = candidates[document_id]
        record["admissible"] = False
        names: List[str] = record["missing_labels"] or [entity["entity_id"] for entity in record["entities"]]
        _reject(result, candidates, profiles, document_id,
                "co-mention check: bridging name(s) " + ", ".join(f"'{name}'" for name in names[:2])
                + " do not occur verbatim in both bodies")


# ── judgement ──────────────────────────────────────────────────────────────────


def _pair_prompt(profile_a: Dict[str, Any], profile_b: Dict[str, Any],
                 signal: Dict[str, Any], question: str) -> str:
    """The two document profiles and the shared signal — bounded, never bodies."""
    def _excerpt(profile: Dict[str, Any]) -> str:
        head: str = " ".join(str(profile.get("head") or "").split())[:EXPANSION_PROMPT_HEAD_CHARS]
        tail: str = " ".join(str(profile.get("tail") or "").split())[-EXPANSION_PROMPT_TAIL_CHARS:]

        return f"…{tail}… [excerpt] …{head}…" if tail else f"…{head}…"

    lines: List[str] = [
        f"User question (why we look): {question}",
        "",
        f"Document A — {profile_a.get('name') or profile_a.get('id')}",
        f"  Assigned categories: {', '.join(profile_a.get('categories') or []) or '(none)'}",
        f"  Excerpt: {_excerpt(profile_a)}",
        "",
        f"Document B — {profile_b.get('name') or profile_b.get('id')}",
        f"  Assigned categories: {', '.join(profile_b.get('categories') or []) or '(none)'}",
        f"  Excerpt: {_excerpt(profile_b)}",
        "",
    ]
    if signal.get("shared_values"):
        rung: int = min(int(signal["shared_values"]), EXPANSION_LADDER_TOP)
        lines.append(f"Shared signal: {signal['shared_values']} approved category value(s) in common — ladder rung {rung}.")
    for entity in signal.get("entities") or []:
        shown: str = (entity["labels"][0] if entity.get("labels") else entity["entity_id"])
        if entity.get("evidence_label"):
            lines.append(f"Bridging entity: {shown} — its name occurs verbatim in both bodies.")
        else:
            lines.append(f"Bridging entity: {shown} — its name does NOT occur verbatim in both bodies.")
    for name in signal.get("missing_labels") or []:
        lines.append(f"Bridging name '{name}' was not found in both bodies.")

    return "\n".join(lines)


def _judge_pairs(pending: List[Dict[str, str]], profiles: Dict[str, Dict[str, Any]],
                 candidates: Dict[str, Dict[str, Any]], question: str,
                 ask: PairAsk) -> Dict[Tuple[str, str], Tuple[bool, str]]:
    """Judge every pending pair in parallel; a failed judgement is a visible NO."""
    verdicts: Dict[Tuple[str, str], Tuple[bool, str]] = {}
    if not pending:
        return verdicts

    def _judge(item: Dict[str, str]) -> Tuple[Tuple[str, str], Tuple[bool, str]]:
        pair: Tuple[str, str] = (item["seed"], item["candidate"])  # caller unpacks (seed, candidate): never sort
        candidate: Dict[str, Any] = candidates.get(item["candidate"]) or {}
        signal: Dict[str, Any] = _signal_for(candidate, str(item["seed"]), question)
        user_prompt: str = _pair_prompt(profiles.get(item["seed"]) or {"id": item["seed"]},
                                        profiles.get(item["candidate"]) or {"id": item["candidate"]},
                                        signal, question)
        try:
            reply: str = ask(JUDGE_SYSTEM_PROMPT, user_prompt)
        except Exception as failure:  # noqa: BLE001 — transport failures must surface as a visible NO
            error_type: str = str(getattr(failure, "error_type", "") or "")
            message: str = str(getattr(failure, "message", "") or "")
            detail: str = f"{error_type}: {message}" if error_type and message else (error_type or type(failure).__name__)

            return pair, (False, f"pair judge unavailable ({detail})")
        verdict: Tuple[bool, str] = _parse_verdict(reply)

        return pair, verdict

    with ThreadPoolExecutor(max_workers=min(EXPANSION_PARALLELISM, len(pending))) as pool:
        for pair, verdict in pool.map(_judge, pending):
            verdicts[pair] = verdict

    return verdicts


def _signal_for(candidate: Dict[str, Any], seed_id: str, question: str) -> Dict[str, Any]:
    """The seed-specific signal view the judge sees for one pair."""
    entities: List[Dict[str, Any]] = [
        entity for entity in candidate.get("entities") or []
        if seed_id in entity.get("via_seeds", []) or not entity.get("via_seeds")
    ]

    return {
        "question": question,
        "shared_values": int(candidate.get("shared_values") or 0),
        "shared_ids": list(candidate.get("shared_ids") or [])[:EXPANSION_SHARED_IDS_SHOWN],
        "entities": entities[:EXPANSION_ENTITIES_PER_CANDIDATE],
        "missing_labels": list(candidate.get("missing_labels") or [])[:3],
    }


def _parse_verdict(reply: str) -> Tuple[bool, str]:
    """Strict two-line reply, parsed defensively; undecidable replies are NOs."""
    text: str = str(reply or "")
    match = _MATCH_RE.search(text)
    if match is None:
        return False, "pair judge reply did not contain a MATCH line"
    verdict: bool = match.group(1).lower() == "yes"
    reason = _REASON_RE.search(text)

    return verdict, (reason.group(1).strip()[:240] if reason
                     else ("pair judge confirmed" if verdict else "pair judge declined without a reason"))


# ── verdict consequences ───────────────────────────────────────────────────────


def _cached_verdicts(config: Neo4jConfig,
                     judged: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Stored :RELATES edges answer pending pairs without paying the judge again."""
    if not judged:
        return {}

    seen: Set[Tuple[str, str]] = set()
    pairs: List[Dict[str, str]] = []
    for (first, second) in judged:
        key: Tuple[str, str] = tuple(sorted((first, second)))  # type: ignore[assignment]
        if key not in seen:
            seen.add(key)
            pairs.append({"a": key[0], "b": key[1]})

    read: Dict[str, Any] = run_read(
        config,
        "UNWIND $pairs AS pair "
        "MATCH (a:Document {id: pair.a})-[r:RELATES]-(b:Document {id: pair.b}) "
        "RETURN DISTINCT pair.a AS a, pair.b AS b, r.method AS method, r.shared_values AS shared_values",
        {"pairs": pairs},
        timeout=STORE_TIMEOUT_SECONDS,
    )
    if not read["ok"]:
        raise _ExpansionError(str((read.get("error") or {}).get("message") or "cached verdict query failed"))

    return {
        tuple(sorted((str(record["a"]), str(record["b"])))):
            {"method": str(record.get("method") or ""), "shared_values": record.get("shared_values")}
        for record in read["records"]
    }


def _confirm(config: Neo4jConfig, result: Dict[str, Any], candidates: Dict[str, Dict[str, Any]],
             profiles: Dict[str, Dict[str, Any]], seed_id: str, document_id: str,
             judge_model: str, reason: str) -> None:
    """Record one confirmed expansion and persist its provenance-carrying edge.

    A second confirmation via a different seed only widens ``via_seeds`` — the
    document enters the enlarged set once, through the first door that opened.
    """
    record: Dict[str, Any] = candidates.get(document_id) or _blank_candidate(document_id)
    already = next((item for item in result["documents"] if item["document_id"] == document_id), None)
    if already is not None:
        if seed_id not in already["via_seeds"]:
            already["via_seeds"].append(seed_id)

        return

    entered: str = _entered_by(record)
    try:
        _write_relation(config, seed_id, document_id,
                        _relation_properties(record, seed_id, reason, entered, judge_model))
    except _ExpansionError as failure:
        _reject(result, candidates, profiles, document_id, f"confirmed but not persisted: {failure.message}")

        return

    profile: Dict[str, Any] = profiles.get(document_id) or {}
    result["documents"].append({
        "document_id": document_id,
        "document": str(profile.get("name") or document_id),
        "entered_by": entered,
        "reason": str(reason)[:240],
        "via_seeds": [seed_id],
        "shared_categories": list(record.get("shared_ids") or [])[:EXPANSION_SHARED_IDS_SHOWN],
        "shared_entities": [{"id": entity["entity_id"],
                             "label": (entity["labels"][0] if entity.get("labels") else entity["entity_id"]),
                             "evidenced": bool(entity.get("evidence_label"))}
                            for entity in (record.get("evidenced_entities") or record.get("entities") or [])
                            [:EXPANSION_ENTITIES_PER_CANDIDATE]],
    })


def _reject(result: Dict[str, Any], candidates: Dict[str, Dict[str, Any]],
            profiles: Dict[str, Dict[str, Any]], document_id: str, reason: str) -> None:
    """One visible rejection — a rejection is as much the demo as an expansion."""
    if any(item["document_id"] == document_id for item in result["rejections"]):
        return
    profile: Dict[str, Any] = profiles.get(document_id) or {}
    result["rejections"].append({
        "document_id": document_id,
        "document": str(profile.get("name") or document_id),
        "reason": str(reason)[:240],
    })
    candidates.pop(document_id, None)


def _entered_by(record: Dict[str, Any]) -> str:
    """The short honest phrase naming how this document entered."""
    shared: int = int(record.get("shared_values") or 0)
    if shared:
        return f"category affinity: {shared} shared value(s), pair check YES"
    if record.get("evidenced_entities"):
        return "shared-entity hop (co-mention verified), pair check YES"

    return "shared-entity hop, pair check YES"


def _relation_properties(record: Dict[str, Any], seed_id: str, reason: str,
                         entered: str, judge_model: str) -> Dict[str, Any]:
    """Every provenance field a reviewer needs to walk the path back."""
    return {
        "method": RELATES_METHOD,
        "matched": True,
        "match_reason": str(reason)[:240],
        "judge_model": str(judge_model or ""),
        "entry": entered,
        "seed_document": seed_id,
        "shared_values": int(record.get("shared_values") or 0),
        "shared_category_ids": [str(item) for item in (record.get("shared_ids") or [])][:16],
        "shared_entity_ids": [str(entity["entity_id"]) for entity in (record.get("entities") or [])][:16],
        "evidenced_labels": [str(entity["evidence_label"]) for entity in (record.get("entities") or [])
                             if entity.get("evidence_label")][:16],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_relation(config: Neo4jConfig, document_a: str, document_b: str,
                    props: Dict[str, Any]) -> None:
    """MERGE the RELATES edge in canonical endpoint order — one pair, one edge.

    An edge that fails to persist must not fake a confirmation: the caller
    downgrades that verdict to a visible rejection.
    """
    left: str = min(document_a, document_b)
    right: str = max(document_a, document_b)
    write: Dict[str, Any] = run_write(
        config,
        "MATCH (left:Document {id: $left}) "
        "MATCH (right:Document {id: $right}) "
        "MERGE (left)-[r:RELATES]->(right) "
        "SET r += $props",
        {"left": left, "right": right, "props": props},
        timeout=STORE_TIMEOUT_SECONDS,
    )
    if not write["ok"]:
        raise _ExpansionError(f"RELATES edge for {document_a} <-> {document_b}: "
                              f"{(write.get('error') or {}).get('message', 'write failed')}")


def _elapsed_ms(started: float) -> int:
    """Whole milliseconds since a ``time.perf_counter()`` reading."""
    return round((time.perf_counter() - started) * 1000)
