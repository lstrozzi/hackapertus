"""Unit tests for the T35 advanced-expansion engine — every store call faked.

Covers the parts the live Cypher validation could not reach:
  * the category ladder floor, the shown-ids clip and seed attribution;
  * the entity hub filter and the note it must leave behind;
  * the co-mention gate rejecting an unevidenced bridge (the docid-400 demo case);
  * a stored verdict skipping the judge entirely (ask must not be called);
  * a YES persisting a canonical-order RELATES edge with the full provenance set;
  * verdict parsing robustness, including the no-MATCH-line failure mode;
  * the no-seed and store-failure degradations, which must keep ok/honest notes.

Run from anywhere (plain python — the engine's own runtime imports are stdlib):

    python tests/chat_expansion_unit_test.py      (exit code 0 = all pass)
"""

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT: Path = Path(__file__).resolve().parent.parent
SRC: Path = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.operations import chat_expansion  # noqa: E402

FAILURES: List[str] = []


def check(label: str, condition: bool) -> None:
    if not condition:
        FAILURES.append(label)
        print(f"FAIL {label}")
    else:
        print(f"ok   {label}")


class FakeStore:
    """Scripted run_read/run_write: statements dispatch by their distinctive phrase."""

    def __init__(self, profiles: Optional[List[Dict[str, Any]]] = None,
                 ladder: Optional[List[Dict[str, Any]]] = None,
                 bridge: Optional[List[Tuple[str, str, str, Any]]] = None,
                 degrees: Optional[Dict[str, int]] = None,
                 evidenced: Optional[List[Tuple[str, str]]] = None,
                 cached: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
                 write_ok: bool = True,
                 fail: Optional[str] = None) -> None:
        self.profiles = profiles or []
        self.ladder = ladder or []
        self.bridge = bridge or []
        self.degrees = degrees or {}
        self.evidenced = set(evidenced or [])
        self.cached = {tuple(sorted(key)): value for key, value in (cached or {}).items()}
        self.write_ok = write_ok
        self.fail = fail
        self.reads: List[Tuple[str, Dict[str, Any]]] = []
        self.writes: List[Tuple[str, Dict[str, Any]]] = []

    def run_read(self, config: Any, statement: str, params: Optional[Dict[str, Any]] = None,
                 timeout: Optional[float] = None) -> Dict[str, Any]:
        params = params or {}
        self.reads.append((statement, params))
        if self.fail and self.fail in statement:
            return {"ok": False, "records": [], "counters": {},
                    "error": {"type": "fake", "message": "faked store outage"}}
        if "OPTIONAL MATCH (document)" in statement and "definition:KeywordCategory" in statement:
            wanted = set(params.get("ids") or [])
            return self._ok([p for p in self.profiles if p["id"] in wanted])
        if "shared_values" in statement and "HAS_CATEGORY" in statement:
            return self._ok([{"document_id": row["document_id"], "shared_values": row["shared_values"],
                              "shared_ids": list(row.get("shared_ids") or []),
                              "seed_lists": [list(row.get("via_seeds") or row.get("seed_lists") or [])]}
                             for row in self.ladder])
        if "entity.name AS raw_name" in statement:
            return self._ok([{"seed_id": seed, "document_id": doc, "entity_id": ent, "raw_name": name}
                            for (seed, doc, ent, name) in self.bridge])
        if "count(DISTINCT document.id) AS degree" in statement:
            return self._ok([{"entity_id": ent, "degree": degree}
                            for ent, degree in self.degrees.items()
                            if ent in set(params.get("entity_ids") or [])])
        if "CONTAINS ref.label" in statement:
            return self._ok([{"document_id": ref["document_id"], "entity_id": ref["entity_id"],
                              "label": ref["label"],
                              "evidenced": (ref["document_id"], ref["entity_id"], ref["label"]) in self.evidenced}
                            for ref in params.get("refs") or []])
        if "-[r:RELATES]-(b:Document" in statement:
            rows = []
            for pair in params.get("pairs") or []:
                hit = self.cached.get(tuple(sorted((pair["a"], pair["b"]))))
                if hit is not None:
                    rows.append({"a": pair["a"], "b": pair["b"], "method": hit.get("method"),
                                 "shared_values": hit.get("shared_values")})
            return self._ok(rows)
        raise AssertionError(f"unexpected read statement: {statement[:80]}")

    def run_write(self, config: Any, statement: str, params: Optional[Dict[str, Any]] = None,
                  timeout: Optional[float] = None) -> Dict[str, Any]:
        params = params or {}
        self.writes.append((statement, params))
        if not self.write_ok:
            return {"ok": False, "records": [], "counters": {},
                    "error": {"type": "fake", "message": "faked write outage"}}
        return {"ok": True, "records": [], "counters": {"relationship_created": 1}}

    @staticmethod
    def _ok(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"ok": True, "records": records, "counters": {}, "error": None}


class FakeAsk:
    """The injected judge: records prompts, replies per scripted candidate name."""

    def __init__(self, verdicts: Optional[List[str]] = None) -> None:
        self.verdicts = list(verdicts or [])
        self.prompts: List[Tuple[str, str]] = []

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append((system_prompt, user_prompt))
        if not self.verdicts:
            return "MATCH: no\nREASON: no scripted verdict"
        return self.verdicts.pop(0)


def profile(document_id: str, name: str = "", categories: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"id": document_id, "name": name or document_id, "head": "head text", "tail": "tail text",
            "categories": sorted(categories or [])}


def config_stub() -> Any:
    return object()


def run(name: str, cases: List[Any]) -> FakeAsk:
    print(f"\n== {name} ==")
    ask = FakeAsk(cases)
    return ask


def main() -> int:
    # 1 — no seeds: honest status, ok stays true, judge never runs.
    print("\n== no seeds ==")
    store = FakeStore()
    chat_expansion.run_read, chat_expansion.run_write = store.run_read, store.run_write
    ask = run("no seeds", [])
    result = chat_expansion.expand_seeds(config_stub(), ["Ready"], [], "q?", ask, "model-x")
    check("no_seeds keeps ok", result["ok"] is True and result["status"] == "no_seeds")
    check("no_seeds note present", any("Nothing to expand from" in note for note in result["notes"]))
    check("no seeds never read the store", not store.reads)

    # 2 — category ladder: floor at 3, ids clipped, seed attribution kept, judge YES -> confirmed + edge.
    print("\n== category ladder + YES verdict ==")
    store = FakeStore(
        profiles=[profile("docid-389", "Anfrage Capaul", ["cat-grp-cvp", "cat-ent-fraktionssekretariat"])],
        ladder=[{"document_id": "docid-402", "shared_values": 12,
                 "shared_ids": ["v1", "v2", "v3", "v4", "v5", "v6", "v7"],
                 "via_seeds": ["docid-389"]},
                {"document_id": "docid-400", "shared_values": 2,
                 "shared_ids": ["v8", "v9"], "via_seeds": ["docid-389"]}],
    )
    chat_expansion.run_read, chat_expansion.run_write = store.run_read, store.run_write
    ask = run("ladder", ["MATCH: yes\nREASON: Same Kleine Anfrage 20 24 quoted verbatim."])
    result = chat_expansion.expand_seeds(config_stub(), ["Ready"], ["docid-389"], "Was wurde aus 20 24?",
                                         ask, "model-x")
    by_id = {item["document_id"]: item for item in result["documents"]}
    check("ladder: 402 confirmed", "docid-402" in by_id)
    check("ladder: entered_by names category path",
          by_id.get("docid-402", {}).get("entered_by", "").startswith("category affinity: 12"))
    check("ladder: reason carried from judge", "verbatim" in by_id.get("docid-402", {}).get("reason", ""))
    check("ladder: shared ids clipped to 6", len(by_id.get("docid-402", {}).get("shared_categories", [])) == 6)
    check("ladder: floor rejects a 2-value candidate",
          any(item["document_id"] == "docid-400" for item in result["rejections"]) or
          "docid-400" not in by_id and not item_evidenced(store, "docid-400"))
    check("ladder: edge written once", len(store.writes) == 1)
    statement, params = store.writes[0]
    check("edge written in canonical order", params["left"] == "docid-389" and params["right"] == "docid-402")
    props = params["props"]
    for key in ("method", "matched", "match_reason", "judge_model", "entry", "seed_document", "shared_values",
                "shared_category_ids", "shared_entity_ids", "evidenced_labels", "created_at"):
        check(f"edge property {key} present", key in props)
    check("edge marks inferred method", props["method"] == "category-affinity+llm-match")

    # 2b — regression (found live 2026-10-16): the candidate sorts BEFORE the seed; verdicts must stay oriented.
    print("\n== candidate id sorts before seed id ==")
    store = FakeStore(
        profiles=[profile("docid-394", "Filing", ["cat-a"]), profile("docid-389", "Older", ["cat-a"]),
                  profile("docid-402", "Later", ["cat-a"])],
        ladder=[{"document_id": "docid-389", "shared_values": 9, "shared_ids": ["v1"], "via_seeds": ["docid-394"]},
                {"document_id": "docid-402", "shared_values": 8, "shared_ids": ["v1"], "via_seeds": ["docid-394"]}],
    )
    chat_expansion.run_read, chat_expansion.run_write = store.run_read, store.run_write
    ask = run("orientation", ["MATCH: no\nREASON: different year.", "MATCH: yes\nREASON: same affair."])
    result = chat_expansion.expand_seeds(config_stub(), ["Ready"], ["docid-394"], "q?", ask, "model-x")
    confirmed = {item["document_id"] for item in result["documents"]}
    rejected = {item["document_id"] for item in result["rejections"]}
    check("orientation: older candidate rejected under ITS id", "docid-389" in rejected)
    check("orientation: the seed is never reported as rejected", "docid-394" not in rejected)
    check("orientation: later candidate confirmed", confirmed == {"docid-402"})
    check("orientation: one reason per pair, none lost", len(rejected) + len(confirmed) == 2)

    # 3 — entity hop without co-mention evidence: the demo rejection, visible and phrased.
    print("\n== co-mention gate rejected (docid-400 demo case) ==")
    store = FakeStore(
        profiles=[profile("docid-389", "Anfrage")],
        bridge=[("docid-389", "docid-400", "ent-affair", "Kleine Anfrage 20 24 Capaul")],
        degrees={"ent-affair": 3},
        evidenced=set(),
    )
    chat_expansion.run_read, chat_expansion.run_write = store.run_read, store.run_write
    ask = run("gate", [])
    result = chat_expansion.expand_seeds(config_stub(), ["Ready"], ["docid-389"], "q?", ask, "model-x")
    rejection = next((item for item in result["rejections"] if item["document_id"] == "docid-400"), None)
    check("gate: 400 rejected", rejection is not None)
    check("gate: reason names co-mention", rejection is not None and "co-mention" in rejection["reason"])
    check("gate: judge never called for rejected bridge", len(ask.prompts) == 0)
    check("gate: nothing confirmed", result["status"] == "no_candidates" and not result["documents"])

    # 4 — entity hop WITH co-mention evidence: judged, YES confirmed with the honest phrase.
    print("\n== co-mention gate passed ==")
    store = FakeStore(
        profiles=[profile("docid-389", "Anfrage"), profile("docid-402", "Budget 2025")],
        bridge=[("docid-389", "docid-402", "ent-affair", "Kleine Anfrage 20 24")],
        degrees={"ent-affair": 3},
        evidenced={("docid-402", "ent-affair", "kleine anfrage 20 24"),
                   ("docid-389", "ent-affair", "kleine anfrage 20 24")},
    )
    chat_expansion.run_read, chat_expansion.run_write = store.run_read, store.run_write
    ask = run("gate-pass", ["MATCH: yes\nREASON: Budget post enumerates the same request number."])
    result = chat_expansion.expand_seeds(config_stub(), ["Ready"], ["docid-389"], "q?", ask, "model-x")
    entry = next((item for item in result["documents"] if item["document_id"] == "docid-402"), None)
    check("gate-pass: 402 confirmed", entry is not None)
    check("gate-pass: entered_by says co-mention verified",
          entry is not None and "co-mention verified" in entry["entered_by"])
    check("gate-pass: prompt carries the verbatim line",
          any("occurs verbatim in both bodies" in prompt for _, prompt in ask.prompts))

    # 5 — hub filter: abridging entity over the degree cap is skipped, with a note.
    print("\n== hub filter ==")
    store = FakeStore(
        profiles=[profile("docid-389", "Anfrage")],
        bridge=[("docid-389", "docid-405", "ent-kantonsrat", "kantonsrat")],
        degrees={"ent-kantonsrat": 16},
    )
    chat_expansion.run_read, chat_expansion.run_write = store.run_read, store.run_write
    ask = run("hub", [])
    result = chat_expansion.expand_seeds(config_stub(), ["Ready"], ["docid-389"], "q?", ask, "model-x")
    check("hub: nothing confirmed via the hub", not result["documents"])
    check("hub: note names the hub cap",
          any("institutional hubs" in note for note in result["notes"]))

    # 6 — cached verdict: the judge must not run at all.
    print("\n== cached verdict skips the judge ==")
    store = FakeStore(
        profiles=[profile("docid-389", "Anfrage", ["c1", "c2"]), profile("docid-402", "Antwort", ["c1", "c2"])],
        ladder=[{"document_id": "docid-402", "shared_values": 9, "shared_ids": ["c1", "c2"],
                 "via_seeds": ["docid-389"]}],
        cached={("docid-389", "docid-402"): {"method": "category-affinity+llm-match", "shared_values": 9}},
    )
    chat_expansion.run_read, chat_expansion.run_write = store.run_read, store.run_write
    ask = run("cached", [])
    result = chat_expansion.expand_seeds(config_stub(), ["Ready"], ["docid-389"], "q?", ask, "model-x")
    entry = next((item for item in result["documents"] if item["document_id"] == "docid-402"), None)
    check("cached: 402 confirmed from store", entry is not None and "cached" in entry["reason"])
    check("cached: judge never called", len(ask.prompts) == 0)
    check("cached: counted", result["cached"] == 1)

    # 7 — verdict parsing robustness.
    print("\n== verdict parsing ==")
    check("plain yes", chat_expansion._parse_verdict("MATCH: yes\nREASON: ok") == (True, "ok"))
    check("bolded verdict", chat_expansion._parse_verdict("**MATCH**: NO\nREASON: Institutions only.")[0] is False)
    check("no MATCH line is a NO", chat_expansion._parse_verdict("I think they match.")[0] is False)
    check("no MATCH line explains itself", "MATCH line" in chat_expansion._parse_verdict("I think they match.")[1])
    parsed_yes, parsed_reason = chat_expansion._parse_verdict("MATCH: yes")
    check("missing REASON still parses", parsed_yes and parsed_reason == "pair judge confirmed")

    # 8 — write failure downgrades the confirmation to a visible rejection.
    print("\n== write failure is not a silent pass ==")
    store = FakeStore(
        profiles=[profile("docid-389", "Anfrage", ["c1"]), profile("docid-402", "Antwort", ["c1"])],
        ladder=[{"document_id": "docid-402", "shared_values": 4, "shared_ids": ["c1"],
                 "via_seeds": ["docid-389"]}],
        write_ok=False,
    )
    chat_expansion.run_read, chat_expansion.run_write = store.run_read, store.run_write
    ask = run("write-fail", ["MATCH: yes\nREASON: Same affair."] * 3)
    result = chat_expansion.expand_seeds(config_stub(), ["Ready"], ["docid-389"], "q?", ask, "model-x")
    check("failed write produced no phantom expansion", not result["documents"])
    check("failed write visible as rejection",
          any("not persisted" in item["reason"] for item in result["rejections"]))

    # 9 — store failure: ok=False, status failed, honest note, no exception.
    print("\n== store failure degrades honestly ==")
    store = FakeStore(profiles=[profile("docid-389", "Anfrage")], fail="HAS_CATEGORY")
    chat_expansion.run_read, chat_expansion.run_write = store.run_read, store.run_write
    ask = run("store-down", [])
    try:
        result = chat_expansion.expand_seeds(config_stub(), ["Ready"], ["docid-389"], "q?", ask, "model-x")
        check("store failure kept ok contract", result["ok"] is False and result["status"] == "failed")
        check("store failure note is honest",
              any("answer is built from the seeds" in note for note in result["notes"]))
    except Exception as failure:  # noqa: BLE001 — the test itself proves this must not happen
        check(f"expand_seeds never raises ({failure})", False)

    # 10 — alias lists: an entity name stored as alias array normalises and gates on ANY alias.
    print("\n== alias list normalisation ==")
    store = FakeStore(
        profiles=[profile("docid-389", "Anfrage"), profile("docid-402", "Bericht")],
        bridge=[("docid-389", "docid-402", "ent-rr", ["Regierungsrat", "regirungsrat"])],
        degrees={"ent-rr": 3},
        evidenced={("docid-402", "ent-rr", "regirungsrat"), ("docid-389", "ent-rr", "regirungsrat")},
    )
    chat_expansion.run_read, chat_expansion.run_write = store.run_read, store.run_write
    ask = run("aliases", ["MATCH: no\nREASON: Only the institution is shared."])
    result = chat_expansion.expand_seeds(config_stub(), ["Ready"], ["docid-389"], "q?", ask, "model-x")
    gate_rejection = next((item for item in result["rejections"] if "co-mention" in item["reason"]), None)
    check("alias evidence reached the judge, not the gate", gate_rejection is None)
    check("alias bridge was judged and declined", any(item["document_id"] == "docid-402" and "pair check" in item["reason"]
                                                     for item in result["rejections"]))
    check("alias candidate not confirmed", not result["documents"])

    print(f"\n{len(FAILURES)} failure(s).")
    return 1 if FAILURES else 0


def item_evidenced(store: FakeStore, document_id: str) -> bool:
    """True if the fake ever saw a mention ref for this document (means it reached the gate)."""
    return any("CONTAINS ref.label" in statement and any(ref.get("document_id") == document_id
                                                         for ref in (params.get("refs") or []))
               for statement, params in store.reads)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as failure:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1)
