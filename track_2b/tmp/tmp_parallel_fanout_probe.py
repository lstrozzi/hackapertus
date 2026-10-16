"""Offline probe: _run_declarations_in_parallel ordering, isolation, gate."""
import sys
import threading
import time

sys.path.insert(0, "src")
from app.operations import kg_extraction_runner as r


class FakeClient:
    def __init__(self):
        self.notes = []


client = FakeClient()
declarations = [{"name": "Person"}, {"name": "Affair"}, {"name": "Organisation"},
                {"name": "Canton"}, {"name": "Topic"}]

in_flight = 0
max_in_flight = 0
lock = threading.Lock()


def work(declaration, progress, base, span):
    global in_flight, max_in_flight
    name = str(declaration["name"])
    with lock:
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
    try:
        time.sleep(0.20)
        if name == "Affair":
            raise RuntimeError("simulated type failure")
        return {name: {"ok": True}}
    finally:
        with lock:
            in_flight -= 1


progress_lines = []


def progress(phase, percent):
    progress_lines.append((phase, percent))


results = r._run_declarations_in_parallel(client, declarations, work, progress,
                                          "entity types", 10, 60)
order = [label for label, _ in results]
failed = [label for label, payload in results if payload is None]
notes_ok = any("Affair: extraction failed" in note for note in client.notes)
order_ok = order == ["Person", "Affair", "Organisation", "Canton", "Topic"]
gate_ok = max_in_flight <= r.LLM_MAX_CONCURRENCY
pool_ok = max_in_flight == min(r.MAX_PARALLELISM, len(declarations))

print("order:", order)
print("failed:", failed)
print("notes ok:", notes_ok)
print("max in flight:", max_in_flight, "(pool size", min(r.MAX_PARALLELISM, len(declarations)), ")")
print("RESULT:", "PASS" if (order_ok and failed == ["Affair"] and notes_ok and gate_ok and pool_ok) else "FAIL")
