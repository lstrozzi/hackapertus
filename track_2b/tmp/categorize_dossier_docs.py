"""
tmp/categorize_dossier_docs.py -- PREPARED, NOT RUN by the agent (owner instruction: prepare only).

Sweeps the documents behind the app's document panel ("Hackathon-temp :Dossier"
content) and categorizes them one by one via POST /api/operations/categories/apply.
Design notes (CLAUDE.md pattern 19):

  - documents enumerated FROM THE LIVE APP at run time (GET /api/operations/documents)
    -- hashes/ids are never hardcoded, the platform re-assigns them across re-imports
  - the apply endpoint takes only the document NAME; the app itself reads the
    platform node's text and re-attaches it, so no local text is needed
  - per-document result -> one flushed log line; failures never abort the sweep
  - resume: names already logged `ok` are skipped unless --force
  - optional --skip-ready passes over documents already Ready upstream (no LLM spend)

Run (from repo root):
  python Hackathon/hackapertus-submission-staging/track_2b/tmp/categorize_dossier_docs.py
  python ... --base-url http://localhost:8000 --limit 5 --interval 90 --force
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOG_PATH = BASE / "categorize_dossier_docs.log"


def _parse(raw: str):
    """Parse a JSON body, falling back to the first 300 raw characters."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw[:300]


def http_json(method: str, base_url: str, path: str, payload, timeout: int):
    """Call one endpoint; return (status, parsed body or text fragment). Never raises on HTTP errors."""
    url = base_url.rstrip("/") + path
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, method=method, data=body)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return resp.status, _parse(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return exc.code, _parse(raw)


def read_log_lines():
    """All sweep log lines, best effort (missing file = empty history)."""
    result = []
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            result = [line.strip() for line in f if line.strip()]

    return result


def previously_applied():
    """Document names already categorised with a genuine ok in a prior run."""
    applied = set()
    for line in read_log_lines():
        match = re.match(r"^\d{4}-\d{2}-\d{2}T\S+\s+\[\s*ok\s+\]\s+(\S+) ", line)
        if match:
            applied.add(match.group(1))

    return applied


def list_documents(base_url: str):
    """The live document list from the app (empty list when unreadable)."""
    status, body = http_json("GET", base_url, "/api/operations/documents", None, timeout=60)
    items = []
    if status == 200 and isinstance(body, dict):
        items = body.get("documents", [])

    return items


def log_line(line: str):
    """Print and append one timestamped line, flushed so the sweep is tail-able."""
    stamped = time.strftime("%Y-%m-%dT%H:%M:%S") + "  " + line
    print(stamped, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(stamped + "\n")


def main():
    parser = argparse.ArgumentParser(description="Categorize the Hackathon-temp :Dossier documents one by one.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N documents (0 = all).")
    parser.add_argument("--interval", type=int, default=0, help="Seconds to sleep between documents.")
    parser.add_argument("--force", action="store_true", help="Re-run documents already logged ok.")
    parser.add_argument("--skip-ready", action="store_true",
                        help="Skip documents whose platform status is already Ready (no LLM spend).")
    args = parser.parse_args()

    docs = list_documents(args.base_url)
    done = set() if args.force else previously_applied()
    pending = []
    for d in docs:
        name = str(d.get("name", "?"))
        if name in done:
            continue
        if args.skip_ready and d.get("platform_status") == "Ready":
            continue
        pending.append(d)
    if args.limit:
        pending = pending[: args.limit]

    ok = 0
    err = 0
    log_line(f"== dossier sweep: {len(docs)} docs, {len(done)} logged ok, {len(pending)} to process ==")
    for n, doc in enumerate(pending, start=1):
        name = str(doc.get("name", "?"))
        status, body = http_json("POST", args.base_url, "/api/operations/categories/apply",
                                 {"document": name}, timeout=900)
        applied = isinstance(body, dict) and body.get("ok") and body.get("status") == "Ready"
        if status == 200 and applied:
            ok += 1
            cats = ", ".join(str(c.get("name", "?")) for c in body.get("categories", [])) or "-"
            log_line(f"[  ok  ] {name} {n}/{len(pending)} cats({len(body.get('categories', []))}): {cats[:120]}")
        else:
            err += 1
            detail = body.get("detail", body.get("error", "no error field")) if isinstance(body, dict) else str(body)
            log_line(f"[ ERROR] {name} {n}/{len(pending)} status={status} {str(detail)[:160]}")
        if args.interval and n < len(pending):
            time.sleep(args.interval)

    log_line(f"== done: applied={ok} errors={err} (failures are per-document; the sweep never aborts) ==")
    print(f"log: {LOG_PATH}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
