"""Offline unit tests for the Track 2B ontology extension (copy-adapted platform skill).

Everything is stubbed: no HTTP, no Neo4j, no LLM. The tests pin exactly the
invariants the feature must never break: proposals join the ontology flagged
`proposed`, they are inadmissible for categorisation, approval only clears the
flag, and no document changes when only proposing categories.
"""

import asyncio
import json
import pathlib
import sys
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.app.operations import category_extension, document_store  # noqa: E402
from src.app.operations import document_categorizer as dc  # noqa: E402

APP_ROOT = pathlib.Path(__file__).resolve().parent.parent
FAKE_CONFIG = SimpleNamespace(configured=True)
DOC = {"name": "sample.txt", "title": "Sample", "text": "The council rejected the motion.",
       "status": "NeedsCategorization"}


class _Stubs:
    """Swap module-level seams for one test and restore them afterwards."""

    def __init__(self) -> None:
        self.patches: List[Tuple[object, str, Any]] = []
        self.saved_items: List[Dict[str, Any]] = []

    def add(self, target: object, name: str, value: Callable) -> None:
        self.patches.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def __enter__(self) -> "_Stubs":
        return self

    def __exit__(self, *_: object) -> None:
        for target, name, original in reversed(self.patches):
            setattr(target, name, original)
        self.patches.clear()


def _resolve_stub(error: Optional[Dict[str, Any]] = None) -> Callable:
    def resolve(name: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        if error:
            return None, error
        return dict(DOC), {}
    return resolve


def _overview(with_proposed: bool = False) -> Dict[str, Any]:
    categories = [{"id": "cat-1", "name": "MOTION_PRACTICE", "description": "Motions.", "lov": [],
                   "documents": 2, "proposed": False}]
    if with_proposed:
        categories.append({"id": "keyword_cat_deadbeef", "name": "LOBBY_STATEMENTS", "description": "Pending.",
                           "lov": [], "documents": 0, "proposed": True})
    return {"ok": True, "ontology_id": dc.ONTOLOGY_ID, "total_definitions": len(categories),
            "categories": categories}


def _llm_stub(reply: str, capture: Dict[str, Any]) -> Callable:
    async def completion(messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
        capture["messages"] = messages
        capture["max_tokens"] = max_tokens
        return {"ok": True, "content": reply, "error": None}
    return completion


def _capture_save(stubs: _Stubs) -> Callable:
    def save(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        stubs.saved_items.extend(items)
        echoed = [{**item, "id": document_store.PROPOSED_CATEGORY_ID_PREFIX + "0123456789ab"}
                  for item in items]
        return {"ok": True, "saved": len(echoed), "skipped": 0, "items": echoed, "error": None}
    return save


def _never(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("a write was attempted where none is allowed")
def test_prepare_excludes_proposed() -> bool:
    with _Stubs() as stubs:
        stubs.add(dc, "_resolve_source_document", _resolve_stub())
        stubs.add(dc, "load_category_overview", lambda: _overview(with_proposed=True))
        prepared = dc._prepare("sample.txt")
        names = [category["name"] for category in prepared.get("categories") or []]
        return prepared["ok"] is True and names == ["MOTION_PRACTICE"]


def test_prompt_contract_and_flow() -> bool:
    reply = json.dumps({"proposals": [
        {"text_en": "COUNCIL_VOTING", "description": "Registered votes and rejected motions.",
         "text_de": "RATSBESCHLUSS", "lov": "a, b"},
        {"text_en": "MOTION_PRACTICE", "description": "Duplicate of an existing category."},
        {"text_en": "NO_DESC"},
    ]})
    capture: Dict[str, Any] = {}
    with _Stubs() as stubs:
        stubs.add(dc, "_resolve_source_document", _resolve_stub())
        stubs.add(dc, "load_category_overview", lambda: _overview(with_proposed=True))
        stubs.add(dc, "_chat_completion", _llm_stub(reply, capture))
        stubs.add(dc, "save_proposed_ontology_categories", _capture_save(stubs))
        stubs.add(dc, "apply_categories", _never)
        stubs.add(dc, "promote_document_status", _never)
        stubs.add(dc, "save_proposed_categories", _never)
        result = asyncio.run(dc.propose_ontology_extension("sample.txt"))
    system = capture["messages"][0]["content"]
    prompt_ok = ("DO NOT PROPOSE THESE AGAIN" in system and "MOTION_PRACTICE" in system
                 and "LOBBY_STATEMENTS" in system
                 and '{"proposals": [' in system
                 and capture["max_tokens"] == dc.EXTENSION_LLM_TOKENS
                 and "<TITLE>" in capture["messages"][1]["content"])
    saved = stubs.saved_items
    saved_ok = (len(saved) == 1 and "id" not in saved[0] and saved[0]["text_en"] == "COUNCIL_VOTING"
                and saved[0]["lov"] == "a, b")
    rejected_ok = (result["rejected"][0].startswith("MOTION_PRACTICE: the ontology already has")
                   and result["rejected"][-1] == "NO_DESC: no description")
    shape_ok = (result["ok"] is True and result["mode"] == "extension" and result["created"] == 1
                and result["proposals"][0]["id"].startswith(document_store.PROPOSED_CATEGORY_ID_PREFIX))
    return prompt_ok and saved_ok and rejected_ok and shape_ok


def test_validation_rules() -> bool:
    existing = _overview(with_proposed=True)["categories"]
    raw = {"proposals": [
        {"name": "FROM_NAME_KEY", "description": "Name-key fallback works."},
        {"text_en": "lobby_statements", "description": "Clashes with a flagged one."},
        {"text_en": "SILENT_DUPES", "description": "First."},
        {"text_en": "SILENT_DUPES", "description": "Second."},
        "not-a-dict",
    ]}
    accepted, rejected = dc._validate_proposals(raw, existing)
    names = [item["text_en"] for item in accepted]
    return (names == ["FROM_NAME_KEY", "SILENT_DUPES"]
            and any("already has" in reason for reason in rejected))
def test_cap_and_empty_are_clean() -> bool:
    existing = _overview()["categories"]
    raw = {"proposals": [{"text_en": f"NEW_CAT_{index}", "description": "d"}
                          for index in range(dc.MAX_EXTENSION_PROPOSALS + 3)]}
    accepted, rejected = dc._validate_proposals(raw, existing)
    if len(accepted) != dc.MAX_EXTENSION_PROPOSALS or not any("over the limit" in reason for reason in rejected):
        return False
    with _Stubs() as stubs:
        stubs.add(dc, "_resolve_source_document", _resolve_stub())
        stubs.add(dc, "load_category_overview", lambda: _overview())
        stubs.add(dc, "_chat_completion", _llm_stub('{"proposals": []}', {}))
        stubs.add(dc, "save_proposed_ontology_categories", _never)
        result = asyncio.run(dc.propose_ontology_extension("sample.txt"))
    return result["ok"] is True and result["created"] == 0 and stubs.saved_items == []


def test_failure_paths_write_nothing() -> bool:
    with _Stubs() as stubs:
        stubs.add(dc, "_resolve_source_document",
                  _resolve_stub(error={"type": "not_categorizable",
                                       "message": "The platform says this document is Ready."}))
        stubs.add(dc, "save_proposed_ontology_categories", _never)
        result = asyncio.run(dc.propose_ontology_extension("sample.txt"))
    return result["ok"] is False and (result["error"] or {}).get("type") == "not_categorizable"


def test_store_save_writes_flagged_nodes() -> bool:
    writes: List[Tuple[str, Dict[str, Any]]] = []

    def fake_read(config: Any, query: str, params: Dict[str, Any] = None, timeout: float = 0.0) -> Dict[str, Any]:
        return {"ok": True, "records": [{"name": "motion_practice"}], "error": None}

    def fake_write(config: Any, query: str, params: Dict[str, Any] = None, timeout: float = 0.0) -> Dict[str, Any]:
        writes.append((query, params or {}))
        created = len((params or {}).get("items") or [])
        return {"ok": True, "counters": {"nodes_created": created, "nodes_deleted": 0,
                                         "properties_set": created}, "database": "neo4j", "error": None}
    with _Stubs() as stubs:
        stubs.add(document_store, "load_neo4j_config", lambda: FAKE_CONFIG)
        stubs.add(document_store, "run_read", fake_read)
        stubs.add(document_store, "run_write", fake_write)
        result = document_store.save_proposed_ontology_categories([
            {"text_en": "LOBBY_STATEMENTS", "description": "Lobby.", "text_fr": "LOBBY", "lov": "a, b"},
            {"text_en": "MOTION_PRACTICE", "description": "Already exists."},
            {"text_en": "LOBBY_STATEMENTS", "description": "Batch dupe."},
        ])
    query, params = writes[0]
    item = params["items"][0]
    return (result["ok"] and result["saved"] == 1 and result["skipped"] == 2
            and item["proposed"] is True and item["text_fr"] == "LOBBY"
            and item["id"].startswith(document_store.PROPOSED_CATEGORY_ID_PREFIX)
            and params["ontology_id"] == document_store.CATEGORY_ONTOLOGY_ID
            and "UNWIND $items AS item" in query and "CREATE (category:KeywordCategory)" in query
            and "SET category += item" in query and "CONTAINS" in query)


def test_store_approve_only_clears_flag() -> bool:
    writes: List[Tuple[str, Dict[str, Any]]] = []

    def fake_write(config: Any, query: str, params: Dict[str, Any] = None, timeout: float = 0.0) -> Dict[str, Any]:
        writes.append((query, params or {}))
        return {"ok": True, "counters": {"nodes_created": 0, "nodes_deleted": 0,
                                        "properties_set": 1}, "database": "neo4j", "error": None}
    with _Stubs() as stubs:
        stubs.add(document_store, "load_neo4j_config", lambda: FAKE_CONFIG)
        stubs.add(document_store, "run_write", fake_write)
        result = document_store.approve_proposed_category("keyword_cat_0123456789ab")
    query, params = writes[0]
    return (result["ok"] and result["approved"] == 1
            and params["id"] == "keyword_cat_0123456789ab"
            and "WHERE category.proposed = true" in query
            and "SET category.proposed = false" in query
            and "DELETE" not in query and "DETACH" not in query)


def test_store_approve_rejects_non_proposed() -> bool:
    with _Stubs() as stubs:
        stubs.add(document_store, "load_neo4j_config", lambda: FAKE_CONFIG)
        stubs.add(document_store, "run_write",
                  lambda config, query, params=None, timeout=0.0: {"ok": True, "counters": {"nodes_created": 0,
                                                                                            "nodes_deleted": 0,
                                                                                            "properties_set": 0},
                                                                   "database": "neo4j", "error": None})
        result = document_store.approve_proposed_category("MOTION_PRACTICE")
    return result["ok"] is False and (result["error"] or {}).get("type") == "not_proposed"


def test_box_and_page_contract() -> bool:
    html = category_extension.render()
    required = ('data-extend-ontology disabled', 'id="category-extension-document"',
                'id="category-extension-status"', 'id="category-extension-results"')
    if not all(marker in html for marker in required):
        return False
    page = (APP_ROOT / "src" / "app" / "static" / "index.html").read_text(encoding="utf-8")
    tabs = (APP_ROOT / "src" / "app" / "operations" / "categories.py").read_text(encoding="utf-8")
    css = (APP_ROOT / "src" / "app" / "static" / "css" / "operations" / "category_extension.css").read_text(encoding="utf-8")
    init = (APP_ROOT / "src" / "app" / "operations" / "__init__.py").read_text(encoding="utf-8")
    return ('runOntologyExtension' in page and '/api/operations/categories/extend' in page
            and 'data-approve-category' in page and '/api/operations/categories/approve' in page
            and 'inferButton.disabled = picker.disabled || !picker.value' in page
            and "render_category_extension" in tabs and "category_extension" in init
            and ".category-proposed-badge" in css)


TESTS = [
    ("_prepare never admits flagged categories", test_prepare_excludes_proposed),
    ("extension prompt and full propose flow", test_prompt_contract_and_flow),
    ("validator rules (keep, reject, dedupe)", test_validation_rules),
    ("cap enforced and empty is clean", test_cap_and_empty_are_clean),
    ("failure paths write nothing", test_failure_paths_write_nothing),
    ("store writes flagged KeywordCategory nodes", test_store_save_writes_flagged_nodes),
    ("approve only clears the flag", test_store_approve_only_clears_flag),
    ("approve rejects non-proposed ids", test_store_approve_rejects_non_proposed),
    ("box, tab, page, css wiring", test_box_and_page_contract),
]


if __name__ == "__main__":
    failed = 0
    for case_name, check in TESTS:
        try:
            passed = bool(check())
            detail = ""
        except Exception as error:
            passed = False
            detail = f" ({type(error).__name__}: {error})"
        print(f"{'PASS' if passed else 'FAIL'}  {case_name}{detail}")
        failed += 0 if passed else 1
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
