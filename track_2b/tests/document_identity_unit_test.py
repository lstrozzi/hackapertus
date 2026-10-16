"""Focused unit tests for document identity — why one file must not get two nodes.

The synchroniser used to derive a Node key from the filename alone, so a sample the
platform had already uploaded (as `docid-413`) was written a second time as
`docid-track2b-<hash>`: one file, two Document nodes, and one of them an orphan. The
tests below pin the corrected behaviour without Neo4j and without an LLM:

  * a filename the graph already knows is addressed by that node, never by a new key;
  * the platform's numbered node wins over any other candidate;
  * a foreign corpus node that merely shares a filename is not adopted;
  * a brand new file is still created, exactly once, under the app's own key;
  * single-document writes/replies address the node by name, not only by the old key;
  * a failing identity read stops the sync instead of letting it create blind.

Every store call is a typed fake, so the test never reaches a database. Run from
anywhere:

    python tests/document_identity_unit_test.py      (exit code 0 = all pass)
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT: Path = Path(__file__).resolve().parent.parent
SRC: Path = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.operations import document_store  # noqa: E402

NEW_FILE: str = "brand-new.pdf"
TRACK2B_ID: str = document_store.document_node_id(NEW_FILE)

PLATFORM_ID: str = "docid-413"


class _Config:
    """A configured Neo4jConfig stand-in with an identity, for assertions."""

    def __init__(self) -> None:
        self.configured: bool = True


class _Store:
    """Fake store that answers the lookups the module is expected to make."""

    def __init__(self, ids_by_name: Optional[Dict[str, List[str]]] = None,
                 read_error: Optional[Dict[str, Any]] = None) -> None:
        self.ids_by_name: Dict[str, List[str]] = ids_by_name or {}
        self.read_error: Optional[Dict[str, Any]] = read_error
        self.reads: List[Tuple[str, Dict[str, Any]]] = []
        self.writes: List[Tuple[str, Dict[str, Any]]] = []

    def __enter__(self) -> "_Store":
        self._originals: Dict[str, Any] = {}
        for name in ("load_neo4j_config", "run_read", "run_write"):
            self._originals[name] = getattr(document_store, name)
        document_store.load_neo4j_config = lambda: _Config()  # type: ignore[assignment]
        document_store.run_read = self._read  # type: ignore[assignment]
        document_store.run_write = self._write  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        for name, original in self._originals.items():
            setattr(document_store, name, original)

        return False

    def _read(self, config: Any, statement: str,
              parameters: Optional[Dict[str, Any]] = None,
              timeout: float = 10.0) -> Dict[str, Any]:
        self.reads.append((statement, parameters or {}))
        if self.read_error is not None:
            return {"ok": False, "records": [], "error": self.read_error}
        if "collect(document)" in statement:
            names: List[str] = [str(item) for item in (parameters or {}).get("names", [])]
            records: List[Dict[str, Any]] = [{"name": name,
                                             "ids": self.ids_by_name.get(name, [])}
                                            for name in names]
            return {"ok": True, "records": records, "error": None}
        if "row.id" in statement:
            rows: List[Dict[str, str]] = [dict(row) for row in (parameters or {}).get("rows", [])]
            return {"ok": True,
                    "records": [{"name": row["name"], "status": "Ready", "proposed": None,
                                 "has_text": True, "assigned": 0} for row in rows],
                    "error": None}

        return {"ok": True, "records": [], "error": None}

    def _write(self, config: Any, statement: str,
               parameters: Optional[Dict[str, Any]] = None,
               timeout: float = 10.0) -> Dict[str, Any]:
        self.writes.append((statement, parameters or {}))

        return {"ok": True, "counters": {"nodes_created": 1,
                                        "relationships_created": 0,
                                        "properties_set": 1},
                "error": None}


def _synced_id(store: _Store, name: str = NEW_FILE) -> str:
    """Run a one-file sync and report the Node key the statement wrote to."""
    result: Dict[str, Any] = document_store.synchronize_documents(
        [{"name": name, "path": f"data/sample/{name}"}])
    assert result["ok"], f"sync failed: {result['error']}"
    assert len(store.writes) == 1, f"expected one write, got {len(store.writes)}"

    return str(store.writes[0][1]["documents"][0]["id"])


def test_new_file_created_once() -> bool:
    """An unknown filename is created, once, under the app's own key."""
    with _Store() as store:
        used: str = _synced_id(store)
        statement: str = store.writes[0][0]

    return used == TRACK2B_ID and "MERGE (document:Document {id: item.id})" in statement


def test_platform_node_adopted() -> bool:
    """A file the platform already has is reused, and its numbered id wins."""
    with _Store({NEW_FILE: ["docid-99", PLATFORM_ID, TRACK2B_ID]}) as store:
        used: str = _synced_id(store)
        status: str = str(document_store.update_document_status(
            NEW_FILE, "Processing").get("ok"))

    by_name: str = store.writes[-1][0]

    return (used == PLATFORM_ID
            and status == "True"
            and store.writes[-1][1]["id"] == TRACK2B_ID
            and "document.FileName = $name" in by_name
            and store.writes[-1][1]["name"] == NEW_FILE)


def test_only_node_reused_without_recreating() -> bool:
    """A node this app created earlier is found by name and never duplicated."""
    with _Store({NEW_FILE: [TRACK2B_ID]}) as store:
        used: str = _synced_id(store)

    return used == TRACK2B_ID


def test_foreign_node_not_adopted() -> bool:
    """Another corpus's node that shares a filename stays out of Track 2B's way."""
    with _Store({NEW_FILE: ["document_inno-151"]}) as store:
        used: str = _synced_id(store)

    return used == TRACK2B_ID


def test_sync_stops_on_lookup_error() -> bool:
    """A failing identity lookup must not fall through to a blind create."""
    failure: Dict[str, Any] = {"type": "auth", "message": "no access"}
    with _Store(read_error=failure) as store:
        result: Dict[str, Any] = document_store.synchronize_documents(
            [{"name": NEW_FILE, "path": f"data/sample/{NEW_FILE}"}])

    return result["ok"] is False and result["error"] == failure and not store.writes


def test_states_follow_resolved_ids() -> bool:
    """Statuses are read back through the resolved id and still keyed by filename."""
    with _Store({NEW_FILE: ["docid-9", PLATFORM_ID]}) as store:
        result: Dict[str, Any] = document_store.read_document_states(
            [NEW_FILE, "other.pdf"])

    rows: List[Dict[str, str]] = store.reads[-1][1]["rows"]

    return (result["ok"] and PLATFORM_ID in [row["id"] for row in rows]
            and NEW_FILE in result["states"] and "row.id" in store.reads[-1][0])


TESTS: List[Any] = [test_new_file_created_once,
                    test_platform_node_adopted,
                    test_only_node_reused_without_recreating,
                    test_foreign_node_not_adopted,
                    test_sync_stops_on_lookup_error,
                    test_states_follow_resolved_ids]


def main() -> int:
    """Run every test, print one line each, and fail the process on any failure."""
    failures: int = 0
    for test in TESTS:
        try:
            passed: bool = bool(test())
        except Exception:
            import traceback
            passed = False
            print(f"FAIL {test.__name__}\n{traceback.format_exc()}")

        if passed:
            print(f"PASS {test.__name__}")
        else:
            failures += 1

    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
