"""Ontology-driven categorisation of a single Track 2B document.

A slim re-use of the platform's DocumentCategorizer skill (PythonServer's
core/extensions/skillsets/Documents/): same contract — the ontology is the only
source of admissible categories, the LLM may not invent any, list-of-values
constraints are enforced after the reply, and the category assignment shape
matches the platform's graph model — but no platform imports: the skill runs on
the OpenAI-compatible endpoint of the env-var contract alone, against this
app's own store.

One ontology drives both boxes: CATEGORY_ONTOLOGY_ID in ``document_store``
(default "Parlament"), hard-coded by design. Read the definitions with
:func:`load_category_overview`; run the skill with :func:`categorize_document`,
which either stores a reviewable proposal (inference box) or writes the
assignment edges and promotes the document (categorisation box).
"""

__all__ = ["ONTOLOGY_ID", "categorize_document", "load_category_overview", "propose_ontology_extension"]

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

import httpx

from ..config import load_llm_config
from ..graphstore.neo4j import load_neo4j_config, run_read
from .document_store import (CATEGORY_ONTOLOGY_ID, PLATFORM_CATEGORIZABLE_STATUS, apply_categories,
                             platform_status_categorizable, promote_document_status,
                             promote_platform_document_status, read_document, read_platform_document,
                             save_proposed_categories, save_proposed_ontology_categories)

ONTOLOGY_ID: str = CATEGORY_ONTOLOGY_ID

# A categorisation reply is small but the document is not: the timeout is sized
# for one long-prompt completion, and the prompt for one very long file.
LLM_TIMEOUT_SECONDS: float = 180.0
MAX_DOCUMENT_CHARS: int = 60_000
MAX_LLM_TOKENS: int = 1600

# Same two statuses the platform's DocumentCategorizer accepts as input.
ALLOWED_SOURCE_STATUSES: List[str] = ["Ready", "NeedsCategorization"]

JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

CATEGORIES_QUERY: str = (
    "MATCH (ontology:Ontology {id: $ontology_id})-[:CONTAINS]->(definition:KeywordCategory) "
    "OPTIONAL MATCH (definition)<-[:OF_CATEGORY]-(:Category)<-[:HAS_CATEGORY]-(document:Document) "
    "RETURN definition.id AS id, definition.text_en AS name, "
    "       definition.description AS description, definition.lov AS lov, "
    "       coalesce(definition.proposed, false) AS proposed, "
    "       count(DISTINCT document) AS documents "
    "ORDER BY name"
)


def load_category_overview() -> Dict[str, Any]:
    """Read-only view of the ontology's categories, with assigned-document counts."""
    overview: Dict[str, Any] = {"ok": False, "ontology_id": ONTOLOGY_ID,
                                "categories": [], "error": None}
    config = load_neo4j_config()
    if not config.configured:
        overview["error"] = {"type": "not_configured",
                             "message": "Track 2B Neo4j is not configured."}
    else:
        read_result: Dict[str, Any] = run_read(config, CATEGORIES_QUERY,
                                               {"ontology_id": ONTOLOGY_ID}, timeout=10.0)
        if read_result["ok"]:
            overview["ok"] = True
            overview["categories"] = [
                {
                    "id": str(record.get("id") or ""),
                    "name": str(record.get("name") or record.get("id") or ""),
                    "description": str(record.get("description") or ""),
                    "lov": _parse_lov(record.get("lov")),
                    "proposed": bool(record.get("proposed")),
                    "documents": int(record.get("documents") or 0),
                }
                for record in read_result["records"]
            ]
        else:
            overview["error"] = read_result["error"]

    return overview


async def categorize_document(name: str, apply: bool) -> Dict[str, Any]:
    """Run the categorisation skill on one document, proposing or applying.

    ``apply=False`` stores the result as a proposal on the document node and
    changes no edges; ``apply=True`` writes the assignment edges and promotes
    the status to Ready, mirroring the platform skill's final step.
    """
    result: Dict[str, Any] = {"ok": False, "mode": "apply" if apply else "propose",
                              "document": name, "categories": [], "rejected": [],
                              "status": None, "error": None}
    prepared: Dict[str, Any] = await asyncio.to_thread(_prepare, name)
    if not prepared["ok"]:
        result["error"] = prepared["error"]

        return result

    document: Dict[str, Any] = prepared["document"]
    if document.get("source") == "platform":
        result["platform_document_id"] = str(document.get("id") or "")
    definitions: List[Dict[str, Any]] = prepared["categories"]
    llm: Dict[str, Any] = await _ask_llm(document, definitions)
    if not llm["ok"]:
        result["error"] = llm["error"]

        return result

    result["categories"], result["rejected"] = _validate(llm["response"], definitions)

    if apply:
        written: Dict[str, Any] = await asyncio.to_thread(
            apply_categories, name,
            [{"category_id": item["category_id"], "text": item["text"]} for item in result["categories"]])
        if not written["ok"]:
            result["error"] = written["error"]

            return result

        status: str = "Ready"
        promoted: Dict[str, Any] = await asyncio.to_thread(promote_document_status, name, status)
        if not promoted["ok"]:
            result["error"] = promoted["error"]

            return result

        # The platform's C# flips the platform document itself when the skill
        # completes; here we hold that graph, so the apply must do it. Never
        # fatal: the assignment and the local promotion already succeeded.
        platform_update: Dict[str, Any] = await asyncio.to_thread(promote_platform_document_status, name, status)
        result["platform_promoted"] = bool(platform_update["promoted"])
        result["platform_status"] = status if platform_update["promoted"] else None
        if not platform_update["promoted"]:
            if platform_update["ok"]:
                result["platform_note"] = "no platform document matched this file"
            else:
                platform_error: Dict[str, Any] = platform_update.get("error") or {}
                result["platform_note"] = f"platform status not updated: {str(platform_error.get('type'))}"

        result["applied"] = int(written.get("applied") or 0)
    else:
        saved: Dict[str, Any] = await asyncio.to_thread(save_proposed_categories, name, result["categories"])
        if not saved["ok"]:
            result["error"] = saved["error"]

            return result

        status = str(document.get("status") or "Unknown")

    result["status"] = status
    result["ok"] = True

    return result


def _resolve_source_document(name: str) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Return (document, rejection): the categorisable source document, or its typed reason.

    The text may come from either graph. The local Track 2B record wins when this
    app ingested the sample itself; otherwise the matched *platform* document is
    consulted, because a sample PDF arrives with its platform counterpart already
    ingested and awaiting categorisation while the local record is still
    `NotYetProcessed` and textless. Either source must offer a categorisable
    status and non-empty text; neither path may write — only an explicit apply
    assigns categories or promotes a status.
    """
    document: Optional[Dict[str, Any]] = None
    rejection: Dict[str, Any] = {}
    stored: Dict[str, Any] = read_document(name)
    if stored["ok"]:
        record: Dict[str, Any] = stored["document"]
        status: str = str(record.get("status") or "Unknown")
        if status not in ALLOWED_SOURCE_STATUSES:
            rejection = {"type": "not_categorizable",
                         "message": f"Status '{status}' cannot be categorised — ingest the document first."}
        elif not str(record.get("text") or "").strip():
            rejection = {"type": "not_categorizable",
                         "message": "No extracted text: ingest the document before categorising it."}
        else:
            document = record
    elif str((stored["error"] or {}).get("type")) == "document_not_found":
        rejection = stored["error"]
    else:
        return None, stored["error"] or {"type": "graph_unavailable",
                                        "message": "Could not read the document."}

    if document is None:
        platform_document, platform_error = _platform_source_document(name)
        if platform_document is not None:
            document = platform_document
        elif not (rejection
                  and str(platform_error.get("type")) == "document_not_found"):
            rejection = platform_error

    return document, rejection


def _prepare(name: str) -> Dict[str, Any]:
    """Load the source document and the admissible category definitions.

    Proposed categories (flagged `proposed`) are never admissible here: the
    categorisation skill speaks only the approved ontology, so both the prompt
    and the validator behind it see only unflagged definitions. Inspection still
    sees all of them through :func:`load_category_overview`.
    """
    prepared: Dict[str, Any] = {"ok": False, "document": None, "categories": [], "error": None}
    document: Optional[Dict[str, Any]]
    document, rejection = _resolve_source_document(name)
    if document is None:
        prepared["error"] = rejection or {"type": "document_not_found",
                                          "message": f"{name} could not be read or matched."}

        return prepared

    overview: Dict[str, Any] = load_category_overview()
    if not overview["ok"]:
        prepared["error"] = overview["error"]
    else:
        definitions: List[Dict[str, Any]] = [item for item in overview["categories"] if not item.get("proposed")]
        if not definitions:
            prepared["error"] = {"type": "ontology_missing",
                                 "message": f"Ontology '{ONTOLOGY_ID}' defines no approved categories in this graph."}
        else:
            prepared["ok"] = True
            prepared["document"] = document
            prepared["categories"] = definitions

    return prepared


def _platform_source_document(name: str) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Return (document, error): the matched platform document iff it awaits categorisation."""
    found: Dict[str, Any] = read_platform_document(name, MAX_DOCUMENT_CHARS + 1)
    if not found["ok"]:
        return None, found["error"] or {"type": "graph_unavailable",
                                        "message": "Could not read the platform document."}
    record: Optional[Dict[str, Any]] = found.get("document")
    if record is None:
        return None, {"type": "document_not_found",
                      "message": f"{name} matches no platform document awaiting categorisation."}
    platform_status: str = str(record.get("status") or "Unknown")
    if not platform_status_categorizable(platform_status):
        return None, {"type": "not_categorizable",
                      "message": f"Platform document '{record.get('id')}' has status "
                                 f"'{platform_status}' — only '{PLATFORM_CATEGORIZABLE_STATUS}' documents await categorisation."}
    text: str = str(record.get("text") or "")
    if not text.strip():
        return None, {"type": "not_categorizable",
                      "message": f"Platform document '{record.get('id')}' carries no extracted text yet."}

    document = {"id": str(record.get("id")), "name": name, "status": platform_status,
                "text": text, "source": "platform"}

    return document, {}


def _apertus_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Enforce the apertus chat-template rule on any message list.

    The template accepts "system" only as the first message; every later one is
    resent as "user", exactly as the platform's own provider does
    (`src/core/llm/providers/provider_apertus.py`). A later system role makes the
    gateway reject the whole request with HTTP 400.
    """
    sanitized: List[Dict[str, str]] = []
    seen_system: bool = False
    for message in messages:
        if message.get("role") == "system" and seen_system:
            sanitized.append({**message, "role": "user"})
        else:
            seen_system = seen_system or message.get("role") == "system"
            sanitized.append(message)

    return sanitized


async def _chat_completion(messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
    """One chat completion over the endpoint contract. Never raises.

    Shared transport for the two asks this skill makes — assigning existing
    categories and proposing new ones — which differ only in prompt and budget,
    never in error taxonomy.
    """
    answer: Dict[str, Any] = {"ok": False, "content": None, "error": None}
    config = load_llm_config()
    if not config.configured:
        answer["error"] = {"type": "llm_not_configured",
                           "message": "LLM_NAME / LLM_BASE_URL / LLM_API_KEY are not all set."}

        return answer

    payload: Dict[str, Any] = {
        "model": config.name,
        "messages": _apertus_messages(messages),
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if "qwen" in config.name.lower():
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
            response: httpx.Response = await client.post(f"{config.base_url}/chat/completions",
                                                         json=payload, headers=headers)
        if response.status_code != 200:
            answer["error"] = {"type": f"llm_http_{response.status_code}",
                               "message": (response.text or "")[:500]}

            return answer

        body: Dict[str, Any] = response.json()
        content: Any = body["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content:
            answer["error"] = {"type": "llm_empty_reply", "message": "The endpoint returned no content."}

            return answer

        answer["ok"] = True
        answer["content"] = content
    except httpx.TimeoutException:
        answer["error"] = {"type": "llm_timeout",
                           "message": f"No reply within {LLM_TIMEOUT_SECONDS:.0f}s."}
    except (httpx.ConnectError, httpx.RequestError) as exc:
        answer["error"] = {"type": type(exc).__name__, "message": str(exc)}
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        answer["error"] = {"type": "llm_malformed_response", "message": str(exc)}

    return answer


async def _ask_llm(document: Dict[str, Any], definitions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One chat completion asking for the JSON category assignment. Never raises."""
    answer: Dict[str, Any] = {"ok": False, "response": None, "error": None}
    text: str = str(document.get("text") or "")
    if len(text) > MAX_DOCUMENT_CHARS:
        text = text[:MAX_DOCUMENT_CHARS] + "\n[truncated]"

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _system_prompt(definitions)},
        {"role": "user", "content": f"""Categorize the following document using the allowed categories only.
<TITLE>{document.get("name") or "document.txt"}</TITLE>
<CONTENT>
{text}
</CONTENT>"""}
    ]
    completion: Dict[str, Any] = await _chat_completion(messages, MAX_LLM_TOKENS)
    if not completion["ok"]:
        answer["error"] = completion["error"]

        return answer

    try:
        answer["response"] = _parse_response_json(str(completion["content"]))
    except ValueError as exc:
        answer["error"] = {"type": "llm_malformed_response", "message": str(exc)}

        return answer

    answer["ok"] = True

    return answer


def _system_prompt(definitions: List[Dict[str, Any]]) -> str:
    """Category list + strict rules, condensed from the platform skill's prompt."""
    lines: List[str] = []
    for definition in definitions:
        lov: List[str] = definition.get("lov") or []
        suffix: str = f" (allowed values: {', '.join(lov)})" if lov else ""
        description: str = definition.get("description") or ""
        lines.append(f"- {definition['name']}: {description}{suffix}".strip())

    return f"""You are an expert document categorization AI. You MUST only use the categories listed below. Do NOT suggest, invent, or use any category that is not explicitly listed.

Allowed categories:
{chr(10).join(lines)}

STRICT RULES:
1. Only output categories from the list above, with the exact spelling and casing shown
2. Do NOT create new categories or variations of category names
3. Where a category lists allowed values, the value MUST be exactly one of them
4. Where a category lists no values, omit the value (null)
5. Only output categories that are actually supported by the document; an empty array is a valid answer

Output format: a single JSON object, nothing around it:
{{"categories": [{{"name": "CATEGORY_NAME", "value": "VALUE_OR_NULL"}}]}}"""


def _parse_response_json(content: str) -> Dict[str, Any]:
    """Extract the JSON object from a chat reply that may carry prose or fences."""
    match: Optional[re.Match[str]] = JSON_BLOCK_PATTERN.search(content)
    if match is None:
        raise ValueError("The reply contained no JSON object.")

    parsed: Any = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("The JSON reply was not an object.")

    return parsed


def _validate(raw: Dict[str, Any], definitions: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str]]:
    """Keep only admissible assignments; returns (accepted, rejected descriptions)."""
    accepted: List[Dict[str, Any]] = []
    rejected: List[str] = []
    allowed: Dict[str, Dict[str, Any]] = {str(d["name"]): d for d in definitions}
    lower_map: Dict[str, str] = {name.lower(): name for name in allowed}
    seen: set[str] = set()

    items: Any = raw.get("categories") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name: str = str(item.get("name") or item.get("text") or "").strip()
        canonical: Optional[str] = lower_map.get(name.lower())
        if canonical is None:
            rejected.append(f"{name or '(empty)'}: not an ontology category")
            continue
        if canonical in seen:
            continue

        definition: Dict[str, Any] = allowed[canonical]
        lov: List[str] = definition.get("lov") or []
        value: Any = item.get("value")
        value_text: Optional[str] = str(value).strip() if value not in (None, "") else None
        if value_text is not None:
            if lov:
                exact: Optional[str] = next((allowed_value for allowed_value in lov if allowed_value == value_text), None)
                if exact is None:
                    exact = next((allowed_value for allowed_value in lov if allowed_value.casefold() == value_text.casefold()), None)
                if exact is None:
                    rejected.append(f"{canonical}: '{value_text}' is not an allowed value")
                    continue
                value_text = exact
            else:
                value_text = value_text.upper()

        seen.add(canonical)
        accepted.append({"category_id": str(definition["id"]), "name": canonical,
                         "value": value_text, "text": value_text or canonical})

    return accepted, rejected


def _parse_lov(raw: Any) -> List[str]:
    """The platform stores a list of values as one comma-joined string."""
    if isinstance(raw, list):
        return [str(value).strip() for value in raw if str(value).strip()]
    if isinstance(raw, str) and raw.strip():
        return [value.strip() for value in raw.split(",") if value.strip()]

    return []


# ── Ontology extension ────────────────────────────────────────────────────────────
# The platform's DocumentCategorizer gains a second ask, `propose_new_categories:
# true`, whose replies are stored as real KeywordCategory definitions flagged
# `proposed: true` (see ontology_service.save_proposed_categories). Reproduced
# here in the submission's ontology: proposals extend the ontology for review,
# never a document; the flag keeps them out of the categorisation menu until an
# explicit approval clears it.
EXTENSION_LLM_TOKENS: int = 2400
MAX_EXTENSION_PROPOSALS: int = 10


async def propose_ontology_extension(name: str) -> Dict[str, Any]:
    """Ask the extension prompt on one document and file the answer as proposals.

    Mirrors the platform's propose flow: the reply is a *category definition set*,
    not an assignment. What survives validation is created under the ontology
    flagged `proposed: true` — the document keeps its status, edges and proposals
    untouched, and nothing is admissible to categorisation until approved.
    """
    result: Dict[str, Any] = {"ok": False, "mode": "extension", "document": name,
                              "proposals": [], "rejected": [], "created": 0, "skipped": 0,
                              "error": None}
    document: Optional[Dict[str, Any]]
    document, rejection = await asyncio.to_thread(_resolve_source_document, name)
    if document is None:
        result["error"] = rejection or {"type": "document_not_found",
                                        "message": f"{name} could not be read or matched."}

        return result

    overview: Dict[str, Any] = await asyncio.to_thread(load_category_overview)
    if not overview["ok"]:
        result["error"] = overview["error"]

        return result

    llm: Dict[str, Any] = await _ask_extension_llm(document, overview["categories"])
    if not llm["ok"]:
        result["error"] = llm["error"]

        return result

    result["proposals"], result["rejected"] = _validate_proposals(llm["response"], overview["categories"])
    if not result["proposals"]:
        result["ok"] = True

        return result

    saved: Dict[str, Any] = await asyncio.to_thread(save_proposed_ontology_categories, result["proposals"])
    if not saved["ok"]:
        result["error"] = saved["error"]

        return result

    result["created"] = int(saved.get("saved") or 0)
    result["skipped"] = int(saved.get("skipped") or 0)
    result["proposals"] = list(saved.get("items") or [])
    result["ok"] = True

    return result


async def _ask_extension_llm(document: Dict[str, Any], definitions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One chat completion asking for NEW category definitions. Never raises."""
    answer: Dict[str, Any] = {"ok": False, "response": None, "error": None}
    text: str = str(document.get("text") or "")
    if len(text) > MAX_DOCUMENT_CHARS:
        text = text[:MAX_DOCUMENT_CHARS] + "\n[truncated]"

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _extension_system_prompt(definitions)},
        {"role": "user", "content": f"""Analyze the following document and propose categories that would help classify this and similar documents.
<TITLE>{document.get("name") or "document.txt"}</TITLE>
<CONTENT>
{text}
</CONTENT>"""}
    ]
    completion: Dict[str, Any] = await _chat_completion(messages, EXTENSION_LLM_TOKENS)
    if not completion["ok"]:
        answer["error"] = completion["error"]

        return answer

    try:
        answer["response"] = _parse_response_json(str(completion["content"]))
    except ValueError as exc:
        answer["error"] = {"type": "llm_malformed_response", "message": str(exc)}

        return answer

    answer["ok"] = True

    return answer


def _extension_system_prompt(definitions: List[Dict[str, Any]]) -> str:
    """Condensed from the platform's propose_new_categories prompt."""
    existing: str = ""
    names: List[str] = [str(item.get("name") or "") for item in definitions if item.get("name")]
    if names:
        existing = f"\n\nEXISTING CATEGORIES (DO NOT PROPOSE THESE AGAIN):\n{', '.join(names)}"

    return f"""You are an expert document classification AI tasked with proposing new categories for document classification systems.
Analyze document content and propose meaningful categories that would be useful for organizing and classifying similar documents.{existing}

REQUIREMENTS:
1. Propose 3-{MAX_EXTENSION_PROPOSALS} relevant categories that would help classify this type of document
2. Only propose NEW categories that are not in the existing list above; an empty proposals array is a valid answer
3. Each category MUST have:
   - "text_en": a concise name in English, UPPERCASE, max 3 words, underscores between words (e.g. DOCUMENT_TYPE)
   - "description": what the category represents
   - translations "text_fr", "text_de", "text_it" (UPPERCASE)
   - optionally "lov": comma-separated string of allowed values, for finite sets
4. Categories must be general enough to apply to multiple documents, not just this one

Output format: a single JSON object, nothing around it:
{{"proposals": [{{"text_en": "...", "text_fr": "...", "text_de": "...", "text_it": "...", "description": "...", "lov": "..."}}]}}"""


def _validate_proposals(raw: Dict[str, Any], definitions: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str]]:
    """Keep only admissible proposals; returns (accepted, rejected descriptions).

    Same discipline as the assignment validator, aimed at definitions instead of
    answers: names must be present, must not repeat a category the ontology
    already has (approved or proposed, case-insensitive) or an earlier proposal
    in this reply, must carry a description, and the reply is capped at the
    prompt's own maximum. Translations and LoV pass through untouched.
    """
    accepted: List[Dict[str, Any]] = []
    rejected: List[str] = []
    known: set[str] = {str(item.get("name") or "").strip().casefold() for item in definitions}
    seen: set[str] = set()

    items: Any = raw.get("proposals") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text_en: str = str(item.get("text_en") or item.get("name") or "").strip()
        key: str = text_en.casefold()
        if not text_en:
            rejected.append("(empty): a proposal needs a name")
            continue
        if not str(item.get("description") or "").strip():
            rejected.append(f"{text_en}: no description")
            continue
        if key in known:
            rejected.append(f"{text_en}: the ontology already has this category")
            continue
        if key in seen:
            continue
        if len(accepted) >= MAX_EXTENSION_PROPOSALS:
            rejected.append(f"{text_en}: over the limit of {MAX_EXTENSION_PROPOSALS} proposals")
            continue

        seen.add(key)
        proposal: Dict[str, Any] = {"text_en": text_en, "description": str(item.get("description") or "").strip()}
        for language_key in ("text_fr", "text_de", "text_it"):
            translation: str = str(item.get(language_key) or "").strip()
            if translation:
                proposal[language_key] = translation
        lov: str = str(item.get("lov") or "").strip()
        if lov:
            proposal["lov"] = lov
        accepted.append(proposal)

    return accepted, rejected
