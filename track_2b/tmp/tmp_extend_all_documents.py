"""Run the ontology-extension skill on every tracked document, one by one.

Sequential on purpose: one LLM call in flight at a time against the shared
gateway, and one graph write per document. Every result line is flushed
immediately so a partial sweep is still readable from the log file. Nothing here
deletes or approves: proposals accumulate as flagged definitions, which is the
whole point of the feature.
"""

import httpx
import json
import pathlib
import sys
import time

BASE: str = "http://localhost:8000"
LOG: pathlib.Path = pathlib.Path(__file__).with_name("extend_sweep.log")
CALL_TIMEOUT: float = 300.0


def flagged(client: httpx.Client) -> dict:
    """Map of flagged category name -> description, read through the app's own API."""
    payload: dict = client.get(f"{BASE}/api/operations/categories", timeout=60.0).json()
    return {str(c.get("name")): str(c.get("description") or "")
            for c in payload.get("categories") or [] if c.get("proposed")}


def main() -> int:
    failed: int = 0
    log = LOG.open("w", encoding="utf-8")
    with httpx.Client() as client:
        listed: dict = client.get(f"{BASE}/api/operations/documents", timeout=60.0).json()
        names: list = [str(d.get("name")) for d in (listed.get("documents") or []) if d.get("name")]
        before: dict = flagged(client)
        write(log, f"documents={len(names)} flagged_before={len(before)}")
        created_total: int = 0
        errors: list = []
        for index, name in enumerate(names, start=1):
            started: float = time.perf_counter()
            note: str = ""
            try:
                response = client.post(f"{BASE}/api/operations/categories/extend",
                                       json={"document": name}, timeout=CALL_TIMEOUT)
                if response.status_code == 200:
                    body: dict = response.json()
                    created: int = int(body.get("created") or 0)
                    skipped: int = int(body.get("skipped") or 0)
                    rejected: list = body.get("rejected") or []
                    names_new: list = [str(p.get("text_en")) for p in (body.get("proposals") or [])]
                    created_total += created
                    note = (f"created={created} skipped={skipped} rejected={len(rejected)} "
                            f"new={','.join(names_new) if names_new else '-'}")
                else:
                    failed += 1
                    detail: str = str(response.json().get("detail"))[:120]
                    errors.append(f"{name[:12]}: {response.status_code} {detail}")
                    note = f"HTTP {response.status_code} {detail}"
            except Exception as error:
                failed += 1
                errors.append(f"{name[:12]}: {type(error).__name__}")
                note = f"{type(error).__name__}: {str(error)[:120]}"
            elapsed: float = time.perf_counter() - started
            write(log, f"{index:>3}/{len(names)} {name[:12]} {elapsed:6.1f}s {note}")
        after: dict = flagged(client)
        write(log, f"created_total={created_total} flagged_after={len(after)} errors={failed}")
        for new_name in sorted(set(after) - set(before)):
            write(log, f"NEW  {new_name}  |  {after[new_name][:150]}")
        for line in errors:
            write(log, f"ERR  {line}")
    log.close()
    return 0


def write(log, line: str) -> None:
    log.write(line + "\n")
    log.flush()
    print(line, flush=True)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
