"""Reranker and embeddings probes — wizard step 3.

These probes test the retrieval endpoints configured for the active plan. Plan Local
uses its Compose services; Plan SaaS calls the remote LiteLLM endpoints with the same
bearer key used for the chat probe.

The probes ask one small, defined question of each service — embedding returns a
vector of the right shape; rerank returns scores in a sensible order — and report
the typed outcome, in the same shape the chat probe uses.
"""

__all__ = [
    "ModelServicesConfig",
    "load_model_services_config",
    "test_embeddings",
    "test_reranker",
]

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

SERVICE_TIMEOUT_SECONDS = 120.0

# Defaults are the compose service names in docker-compose.plan-local.yml, reachable
# because the app joins that stack's network; a model still loading answers slowly,
# hence a timeout measured in minutes rather than the chat probe's seconds.
DEFAULT_EMBEDDINGS_URL = "http://embeddings:8354/v1"
DEFAULT_RERANKER_URL = "http://reranker:8364/v1"
DEFAULT_EMBEDDINGS_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

_EMBEDDING_INPUT = [
    "The applicant stated a political opinion in a police interview.",
    "Cook pasta in salted water for ten minutes, then drain.",
]

_RERANK_QUERY = "What did the applicant say at the police interview?"
_RERANK_DOCUMENTS = [
    "The applicant described a police interview in which they voiced a political opinion.",
    "Pasta should be cooked in salted water and drained before serving.",
    "The country-of-origin report lists several national newspapers.",
]


@dataclass(frozen=True)
class ModelServicesConfig:
    """Endpoint URLs and served model handles for both retrieval services."""

    embeddings_url: str
    embeddings_model: str
    embeddings_api_key: str
    reranker_url: str
    reranker_model: str
    reranker_api_key: str


def load_model_services_config(api_keys_enabled: bool = True) -> ModelServicesConfig:
    """Read the two service addresses from the environment."""
    fallback_api_key: str = os.environ.get("LLM_API_KEY", "").strip() if api_keys_enabled else ""
    config: ModelServicesConfig = ModelServicesConfig(
        embeddings_url=_base("EMBEDDINGS_URL", DEFAULT_EMBEDDINGS_URL),
        embeddings_model=os.environ.get("EMBEDDINGS_MODEL", DEFAULT_EMBEDDINGS_MODEL).strip() or DEFAULT_EMBEDDINGS_MODEL,
        embeddings_api_key=(os.environ.get("EMBEDDINGS_API_KEY", "").strip() or fallback_api_key),
        reranker_url=_base("RERANKER_URL", DEFAULT_RERANKER_URL),
        reranker_model=os.environ.get("RERANKER_MODEL", DEFAULT_RERANKER_MODEL).strip() or DEFAULT_RERANKER_MODEL,
        reranker_api_key=(os.environ.get("RERANKER_API_KEY", "").strip() or fallback_api_key),
    )

    return config


def _base(name: str, fallback: str) -> str:
    """Absolute service URL, without a trailing slash, so paths append cleanly."""
    return (os.environ.get(name, "").strip() or fallback).rstrip("/")


def _endpoint(base_url: str, route: str) -> str:
    """Accept either a versioned base URL or a URL that already includes the route."""
    endpoint: str = base_url.rstrip("/")
    if endpoint.rsplit("/", 1)[-1].lower() != route.lower():
        endpoint = f"{endpoint}/{route}"

    return endpoint


def test_embeddings(config: ModelServicesConfig) -> Dict[str, Any]:
    """One embedding request: two texts, one vector each, same dimension."""
    result: Dict[str, Any] = {
        "ok": False,
        "service": "embeddings",
        "endpoint": _endpoint(config.embeddings_url, "embeddings"),
        "model": config.embeddings_model,
        "latency_ms": None,
        "count": None,
        "dimensions": None,
        "error": None,
    }

    payload: Dict[str, Any] = {"model": config.embeddings_model, "input": _EMBEDDING_INPUT}
    body: Optional[Dict[str, Any]] = _post(result, payload, result["endpoint"], config.embeddings_api_key)

    if body is None:
        return result

    vectors = body.get("data")
    if not isinstance(vectors, list) or not vectors:
        result["error"] = {"type": "no_embedding_data", "message": "the response carried no 'data' array"}

        return result

    first = (vectors[0] or {}).get("embedding")
    if not isinstance(first, list) or not first:
        result["error"] = {"type": "malformed_embedding", "message": "the first vector is missing or empty"}

        return result

    result["count"] = len(vectors)
    result["dimensions"] = len(first)
    result["ok"] = all(isinstance(item, dict) and isinstance(item.get("embedding"), list) and len(item["embedding"]) == len(first)
                       for item in vectors)
    if not result["ok"]:
        result["error"] = {"type": "inconsistent_dimensions", "message": "the returned vectors differ in length"}

    return result


def test_reranker(config: ModelServicesConfig) -> Dict[str, Any]:
    """One rerank request: the relevant document must outscore the irrelevant one."""
    result: Dict[str, Any] = {
        "ok": False,
        "service": "reranker",
        "endpoint": _endpoint(config.reranker_url, "rerank"),
        "model": config.reranker_model,
        "latency_ms": None,
        "count": None,
        "top_relevance": None,
        "ranking_correct": None,
        "error": None,
    }

    payload: Dict[str, Any] = {"model": config.reranker_model, "query": _RERANK_QUERY, "documents": _RERANK_DOCUMENTS}
    body: Optional[Dict[str, Any]] = _post(result, payload, result["endpoint"], config.reranker_api_key)

    if body is None:
        return result

    results = body.get("results") or body.get("data")
    if not isinstance(results, list) or not results:
        result["error"] = {"type": "no_rerank_results", "message": "the response carried no 'results' array"}

        return result

    scored = [item for item in results if isinstance(item, dict)]
    if not scored:
        result["error"] = {"type": "malformed_rerank", "message": "no score entries could be read"}

        return result

    by_index = {item.get("index", item.get("document", {}).get("index")): _score_of(item) for item in scored}
    result["count"] = len(scored)
    result["top_relevance"] = max((s for s in by_index.values() if s is not None), default=None)
    relevant: Optional[float] = by_index.get(0)
    irrelevant: Optional[float] = by_index.get(1)
    result["ranking_correct"] = relevant is not None and irrelevant is not None and relevant > irrelevant
    result["ok"] = bool(result["ranking_correct"])

    if not result["ok"]:
        result["error"] = {
            "type": "unexpected_ranking",
            "message": f"the relevant document did not outscore the distractor (relevant={relevant}, distractor={irrelevant})",
        }

    return result


def _post(result: Dict[str, Any], payload: Dict[str, Any], url: str,
          api_key: str = "") -> Optional[Dict[str, Any]]:
    """POST one probe and return its JSON body, filling the result on any failure."""
    started: float = time.perf_counter()
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=SERVICE_TIMEOUT_SECONDS) as client:
            response: httpx.Response = client.post(url, json=payload, headers=headers)
        result["latency_ms"] = round((time.perf_counter() - started) * 1000)
        if response.status_code != 200:
            result["error"] = {"type": f"http_{response.status_code}", "message": (response.text or "")[:400]}

            return None
        return response.json()
    except httpx.TimeoutException:
        result["latency_ms"] = round((time.perf_counter() - started) * 1000)
        result["error"] = {"type": "timeout", "message": f"no answer within {SERVICE_TIMEOUT_SECONDS:.0f}s"}
    except (httpx.ConnectError, httpx.RequestError) as exc:
        result["latency_ms"] = round((time.perf_counter() - started) * 1000)
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    except ValueError as exc:
        result["error"] = {"type": "malformed_json", "message": str(exc)}

    return None


def _score_of(item: Dict[str, Any]) -> Optional[float]:
    """Read a score under either spelling, so both response shapes are understood."""
    score: Optional[float] = None
    for key in ("relevance_score", "score"):
        value: Any = item.get(key)
        if score is None and isinstance(value, (int, float)):
            score = float(value)

    return score
