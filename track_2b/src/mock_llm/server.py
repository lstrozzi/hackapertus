"""Minimal dependency-free mock of the OpenAI-compatible surface the app uses.

Implements POST /v1/chat/completions, POST /v1/embeddings, POST /v1/rerank and
GET /v1/models so `make run MOCK=1` proves the whole pipeline end-to-end without
an API key and without GPU images. Serves on MOCK_PORT (default 9999); compose
points LLM_BASE_URL at http://mock-llm:9999/v1 and, under MOCK=1, the two
retrieval services at the same address (see the Makefile).

What "mock" means here, stated honestly per endpoint:

* chat -- one fixed reply. It proves wiring: route, JSON shape, latency. The
  wizard says so on the screen that shows it.
* embeddings -- a fixed-length vector per input, computed deterministically from
  the input's own character trigrams (crc32-bucketed, so it survives a restart).
  The probe checks lengths match, which is exactly what this answers correctly.
* rerank -- genuine lexical scoring: content-word overlap between query and
  document. A relevant document outranks a distractor because it shares words
  with the query, not because any index order was hard-coded. A cross-encoder
  would do this semantically; this does it lexically, and it is enough for the
  probe's question ("did the relevant document outscore the distractor"), not
  for any claim about retrieval quality.

No endpoint pretends to approximate model quality, and the wizard never calls a
mock pass a model pass.
"""

import json
import os
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EMBEDDING_DIMENSIONS: int = 64
MOCK_CHAT_MODEL: str = "mock-apertus-v1.5-8b"

# A deliberately small stop-word list: enough that "the/of/and" cannot carry a
# distractor to the top, which is the failure a lexical mock would hide.
_STOP_WORDS = frozenset("""a an and are as at be but by for from has have he her his i in is it
of on or our she so that the their them they this to was we were will with you your
""".split())


def _content_tokens(text: str) -> set:
    words = "".join(char if char.isalnum() or char.isspace() else " " for char in text.lower()).split()

    return {word for word in words if word not in _STOP_WORDS}


def _mock_embedding(text: str) -> list:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    padded = f"  {text.lower()}  "
    for index in range(len(padded) - 2):
        bucket = zlib.crc32(padded[index:index + 3].encode()) % EMBEDDING_DIMENSIONS
        vector[bucket] += 1.0
    norm = sum(value * value for value in vector) ** 0.5 or 1.0

    return [value / norm for value in vector]


def _lexical_score(query: str, document: str) -> float:
    query_tokens = _content_tokens(query)
    if not query_tokens:
        return 0.0

    return round(len(query_tokens & _content_tokens(document)) / len(query_tokens), 6)


class MockHandler(BaseHTTPRequestHandler):

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            self._respond(200, {"object": "list", "data": [{"id": "mock-apertus-v1.5-8b",
                                                            "object": "model"}]})
        else:
            self._respond(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        path: str = self.path.rstrip("/")
        length: int = int(self.headers.get("Content-Length", "0"))
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._respond(400, {"error": {"message": "invalid json"}})
            return
        if path.endswith("/chat/completions"):
            self._chat(request)
        elif path.endswith("/embeddings"):
            self._embeddings(request)
        elif path.endswith("/rerank"):
            self._rerank(request)
        else:
            self._respond(404, {"error": {"message": "not found"}})

    def _chat(self, request: dict) -> None:
        model: str = str(request.get("model", MOCK_CHAT_MODEL))
        reply: str = "Hello world — this is the bundled mock endpoint."
        completion: dict = {
            "id": f"mock-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 10, "total_tokens": 22},
        }
        self._respond(200, completion)

    def _embeddings(self, request: dict) -> None:
        model: str = str(request.get("model", "mock-embeddings"))
        raw = request.get("input", "")
        texts = raw if isinstance(raw, list) else [str(raw)]
        data = [{"object": "embedding", "index": position, "embedding": _mock_embedding(str(text))}
                for position, text in enumerate(texts)]
        self._respond(200, {"object": "list", "model": model, "data": data,
                            "usage": {"prompt_tokens": sum(len(str(text)) for text in texts),
                                      "total_tokens": sum(len(str(text)) for text in texts)}})

    def _rerank(self, request: dict) -> None:
        model: str = str(request.get("model", "mock-reranker"))
        query: str = str(request.get("query", ""))
        documents = request.get("documents") or []
        scored = [{"index": position, "relevance_score": _lexical_score(query, str(document))}
                  for position, document in enumerate(documents)]
        scored.sort(key=lambda item: (-item["relevance_score"], item["index"]))
        self._respond(200, {"model": model, "results": scored})

    def _respond(self, status: int, payload: dict) -> None:
        body: bytes = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


if __name__ == "__main__":
    port: int = int(os.environ.get("MOCK_PORT", "9999"))
    ThreadingHTTPServer(("0.0.0.0", port), MockHandler).serve_forever()
