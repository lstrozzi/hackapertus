"""Focused unit tests for the Knowledge-tab category-inference flow.

Covers, without Neo4j and without an LLM:
  * the eligibility rule behind the Category inference picker (platform status
    `NeedsCategorization` only — never knowledge_graph_status, never `n/a`);
  * `_prepare` source resolution: local Track 2B record first, matched platform
    document as the fallback, including the no-match and missing-text cases;
  * the proposal path of the skill: proposals stored on the app's own document
    node only — no assignments written and no status promoted anywhere;
  * the apply path: assignment edges plus the Ready promotion on both the local
    record and the matched platform document (the platform's C# equivalent).

Every store call and the chat call are replaced by typed fakes, so the test
never reaches a database or a model. Run from anywhere:

    python tests/categorization_unit_test.py      (exit code 0 = all pass)
"""

import asyncio
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT: Path = Path(__file__).resolve().parent.parent
SRC: Path = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

INDEX_HTML: Path = SRC / "app" / "static" / "index.html"

from app.operations import document_categorizer, document_store  # noqa: E402

ONY_CATEGORY: Dict[str, Any] = {"id": "cat-topic", "name": "Topic",
                                "description": "", "lov": [], "documents": 0}

SAMPLE: str = "council-decisions.pdf"
PLATFORM_ID: str = "docid-413"


def _local(status: str, text: str) -> Dict[str, Any]:
    """A successful read_document answer for a Track 2B record."""
    return {"ok": True,
            "document": {"id": document_store.document_node_id(SAMPLE), "name": SAMPLE,
                         "status": status, "text": text},
            "error": None}


def _local_missing() -> Dict[str, Any]:
    """A successful read_document answer with no Track 2B record."""
    return {"ok": False, "document": None,
            "error": {"type": "document_not_found",
                      "message": f"{SAMPLE} is not tracked in the Track 2B graph."}}


def _platform(status: Optional[str], text: Optional[str], found: bool = True,
              error: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A read_platform_document answer, in each of its three shapes."""
    if error is not None:
        return {"ok": False, "document": None, "error": error}
    document: Optional[Dict[str, Any]] = None
    if found:
        document = {"id": PLATFORM_ID, "status": status or "", "text": text or ""}

    return {"ok": True, "document": document, "error": None}


def _overview() -> Dict[str, Any]:
    """A category overview with the admissible-definitions contract satisfied."""
    return {"ok": True, "ontology_id": "Parlament", "categories": [ONY_CATEGORY], "error": None}


class _Stubs:
    """Install fakes into the categorizer module and restore it on exit."""

    def __init__(self) -> None:
        self.calls: List[str] = []
        self._originals: Dict[str, Any] = {}
        self._names: List[str] = ["read_document", "read_platform_document",
                                  "load_category_overview", "save_proposed_categories",
                                  "apply_categories", "promote_document_status",
                                  "promote_platform_document_status", "_ask_llm"]

    def __enter__(self) -> "_Stubs":
        for name in self._names:
            self._originals[name] = getattr(document_categorizer, name)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        for name, original in self._originals.items():
            setattr(document_categorizer, name, original)

        return False

    def install(self, read_document: Callable[[str], Dict[str, Any]],
                read_platform: Callable[[str, int], Dict[str, Any]],
                ask_llm: Optional[Callable[..., Any]] = None) -> None:
        """Wire one scenario's fakes, recording which ones were consulted."""
        def counted_local(name: str) -> Dict[str, Any]:
            self.calls.append("local")
            return read_document(name)

        def counted_platform(name: str, max_chars: int) -> Dict[str, Any]:
            self.calls.append("platform")
            return read_platform(name, max_chars)

        def saved(name: str, proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
            self.calls.append(f"save:{name}")
            return {"ok": True}

        def applied(name: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
            self.calls.append("apply")
            return {"ok": True, "applied": 0}

        def promoted(name: str, status: str) -> Dict[str, Any]:
            self.calls.append(f"promote:{status}")
            return {"ok": True}

        def platform_promoted(name: str, status: str) -> Dict[str, Any]:
            self.calls.append(f"promote_platform:{status}")
            return {"ok": True, "promoted": True, "platform_document_id": PLATFORM_ID, "error": None}

        async def canned_llm(document: Dict[str, Any],
                             definitions: List[Dict[str, Any]]) -> Dict[str, Any]:
            self.calls.append(f"llm:{len(str(document.get('text') or ''))}")
            return {"ok": True, "response": {"categories": [{"name": "topic", "value": None}]},
                    "error": None}

        document_categorizer.read_document = counted_local
        document_categorizer.read_platform_document = counted_platform
        document_categorizer.load_category_overview = _overview
        document_categorizer.save_proposed_categories = saved
        document_categorizer.apply_categories = applied
        document_categorizer.promote_document_status = promoted
        document_categorizer.promote_platform_document_status = platform_promoted
        document_categorizer._ask_llm = ask_llm or canned_llm


def test_platform_status_rule() -> None:
    assert document_store.platform_status_categorizable("NeedsCategorization") is True
    for wrong in ["n/a", "Ready", "needsCategorization", "NEEDS_CATEGORIZATION",
                  "Completed", "", None]:
        assert document_store.platform_status_categorizable(wrong) is False, wrong


def test_prepare_uses_local_source_when_eligible() -> None:
    with _Stubs() as stubs:
        stubs.install(lambda _name: _local("Ready", "local text"),
                      lambda _name, _chars: _platform("Ready", None, found=False))
        prepared: Dict[str, Any] = document_categorizer._prepare(SAMPLE)
        assert prepared["ok"] is True
        assert prepared["document"]["text"] == "local text"
        assert "platform" not in stubs.calls, "local text must not re-read the platform"


def test_prepare_falls_back_to_platform_document() -> None:
    with _Stubs() as stubs:
        stubs.install(lambda _name: _local("NotYetProcessed", ""),
                      lambda _name, _chars: _platform("NeedsCategorization", "platform text"))
        prepared = document_categorizer._prepare(SAMPLE)
        assert prepared["ok"] is True
        assert prepared["document"]["text"] == "platform text"
        assert prepared["document"]["id"] == PLATFORM_ID
        assert prepared["document"]["status"] == "NeedsCategorization"
        assert prepared["document"].get("source") == "platform"


def test_prepare_no_match_anywhere_is_document_not_found() -> None:
    with _Stubs() as stubs:
        stubs.install(lambda _name: _local_missing(),
                      lambda _name, _chars: _platform(None, None, found=False))
        prepared = document_categorizer._prepare(SAMPLE)
        assert prepared["ok"] is False
        assert prepared["error"]["type"] == "document_not_found"


def test_prepare_unmatched_keeps_local_ingest_reason() -> None:
    with _Stubs() as stubs:
        stubs.install(lambda _name: _local("NotYetProcessed", ""),
                      lambda _name, _chars: _platform(None, None, found=False))
        prepared = document_categorizer._prepare(SAMPLE)
        assert prepared["ok"] is False
        assert prepared["error"]["type"] == "not_categorizable"
        assert "ingest" in prepared["error"]["message"]


def test_prepare_platform_status_ineligible_names_the_status() -> None:
    with _Stubs() as stubs:
        stubs.install(lambda _name: _local("NotYetProcessed", ""),
                      lambda _name, _chars: _platform("Ready", "text"))
        prepared = document_categorizer._prepare(SAMPLE)
        assert prepared["ok"] is False
        assert prepared["error"]["type"] == "not_categorizable"
        assert "Ready" in prepared["error"]["message"]


def test_prepare_platform_document_without_text_is_refused() -> None:
    with _Stubs() as stubs:
        stubs.install(lambda _name: _local("NotYetProcessed", ""),
                      lambda _name, _chars: _platform("NeedsCategorization", "   "))
        prepared = document_categorizer._prepare(SAMPLE)
        assert prepared["ok"] is False
        assert prepared["error"]["type"] == "not_categorizable"
        assert "no extracted text" in prepared["error"]["message"].lower()


def test_prepare_store_failure_short_circuits_before_platform() -> None:
    with _Stubs() as stubs:
        stubs.install(lambda _name: {"ok": False, "document": None,
                                     "error": {"type": "auth_failed", "message": "no"}},
                      lambda _name, _chars: _platform("NeedsCategorization", "text"))
        prepared = document_categorizer._prepare(SAMPLE)
        assert prepared["ok"] is False
        assert prepared["error"]["type"] == "auth_failed"
        assert "platform" not in stubs.calls, "a failing store must not be retried per box"


def test_propose_stores_on_local_node_and_promotes_nothing() -> None:
    with _Stubs() as stubs:
        stubs.install(lambda _name: _local("NotYetProcessed", ""),
                      lambda _name, _chars: _platform("NeedsCategorization", "platform text"))
        result: Dict[str, Any] = asyncio.run(document_categorizer.categorize_document(SAMPLE, apply=False))
        assert result["ok"] is True, result["error"]
        assert result["mode"] == "propose"
        assert result["status"] == "NeedsCategorization"
        assert result["platform_document_id"] == PLATFORM_ID
        assert [item["name"] for item in result["categories"]] == ["Topic"]
        assert stubs.calls[0] == "local"
        assert f"save:{SAMPLE}" in stubs.calls
        assert "apply" not in stubs.calls
        assert not any(call.startswith("promote") for call in stubs.calls)
        assert any(call.startswith("llm:") for call in stubs.calls)


def test_apply_promotes_local_and_platform_status() -> None:
    with _Stubs() as stubs:
        stubs.install(lambda _name: _local("NeedsCategorization", "local text"),
                      lambda _name, _chars: _platform("NeedsCategorization", "platform text"))
        result: Dict[str, Any] = asyncio.run(document_categorizer.categorize_document(SAMPLE, apply=True))
        assert result["ok"] is True, result["error"]
        assert result["mode"] == "apply"
        assert result["status"] == "Ready"
        assert result["platform_promoted"] is True
        assert result["platform_status"] == "Ready"
        assert "apply" in stubs.calls
        assert "promote:Ready" in stubs.calls
        assert "promote_platform:Ready" in stubs.calls


def test_index_html_picker_contract() -> None:
    html: str = INDEX_HTML.read_text(encoding="utf-8")
    assert "function categoryInferenceEligible(documentInfo) {" in html
    assert 'return documentInfo.platform_status === "NeedsCategorization";' in html
    assert "documents.filter((documentInfo) => categoryInferenceEligible(documentInfo))" in html
    assert html.count("categoryInferenceEligible") == 2, "exactly one definition and one picker use"
    assert re.search(r"documentInfo\.status === \"NeedsCategorization\"", html) is None, \
        "the old app-local picker filter must be gone"
    body: str = html.split("function categoryInferenceEligible(documentInfo) {", 1)[1].split("\n}", 1)[0]
    code_lines: str = "\n".join(line for line in body.splitlines() if not line.strip().startswith("//"))
    assert "knowledge_graph_status" not in code_lines, "KG-extraction status must never gate the picker"


TESTS: List[Callable[[], None]] = [
    test_platform_status_rule,
    test_prepare_uses_local_source_when_eligible,
    test_prepare_falls_back_to_platform_document,
    test_prepare_no_match_anywhere_is_document_not_found,
    test_prepare_unmatched_keeps_local_ingest_reason,
    test_prepare_platform_status_ineligible_names_the_status,
    test_prepare_platform_document_without_text_is_refused,
    test_prepare_store_failure_short_circuits_before_platform,
    test_propose_stores_on_local_node_and_promotes_nothing,
    test_apply_promotes_local_and_platform_status,
    test_index_html_picker_contract,
]


def main() -> int:
    """Run every test, print one line each, and report the total."""
    failed: List[Tuple[str, str]] = []
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except (AssertionError, Exception) as error:
            failed.append((test.__name__, traceback.format_exc()))
            print(f"FAIL {test.__name__}: {error}")

    print(f"{len(TESTS) - len(failed)}/{len(TESTS)} passed")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
