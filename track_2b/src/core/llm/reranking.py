"""Centralized manager for external reranking model calls."""

from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

from typing import Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import requests

from src.config.constants import RERANKING_REQUEST_TIMEOUT_SECONDS
from src.core.llm.manager import LlmManager
from src.utils.log import log, INFO, DEBUG, WARNING, ERROR, PERF


RERANKING_AVAILABILITY_TIMEOUT_SECONDS = 5


class RerankingManager:
    """Singleton manager for configured reranking model requests."""

    _instance: Optional['RerankingManager'] = None
    _endpoints: List[str] = []
    _model: Optional[str] = None

    def __init__(self) -> None:
        """Initialize the configured reranking endpoints."""
        self.initialize()

    @classmethod
    def instance(cls) -> 'RerankingManager':
        """Get or create the singleton RerankingManager instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_settings(self) -> Optional[dict]:
        """Return the enabled reranking configuration, if valid."""
        settings = LlmManager.instance().get_reranking_settings()
        if not settings or not settings.get("enabled"):
            return None
        if not settings.get("model") or not settings.get("endpoint"):
            log(ERROR, "RerankingManager: Model or endpoint not configured")
            return None
        return settings

    def initialize(self) -> bool:
        """Load the configured reranking endpoint and model."""
        settings = self._get_settings()
        self._endpoints = [
            endpoint.strip()
            for endpoint in settings["endpoint"].split(",")
            if endpoint.strip()
        ] if settings else []
        self._model = settings["model"] if settings else None
        if not self._endpoints or not self._model:
            log(ERROR, "RerankingManager.initialize: Reranking endpoint or model is not configured")
            raise RuntimeError("Reranking endpoint or model is not configured")

        initialized:bool = self.is_available()
        if not initialized:
            log(ERROR, "RerankingManager.initialize: No available reranking endpoints")
            raise RuntimeError("No available reranking endpoints")

        log(INFO, f"RerankingManager.initialize: {len(self._endpoints)} available endpoint(s): {self._endpoints}")

        return initialized

    def _is_endpoint_available(self, endpoint: str) -> bool:
        """Check whether one reranking endpoint is reachable and has the model."""

        try:
            response = requests.head(
                f"{endpoint.rstrip('/')}/score",
                timeout=RERANKING_AVAILABILITY_TIMEOUT_SECONDS,
            )
            if response.status_code != 405:
                response.raise_for_status()
                return True
            log(DEBUG, f"RerankingManager._is_endpoint_available: HEAD {endpoint}/score returned 405, trying /models")

        except requests.exceptions.ConnectionError:
            log(WARNING, f"RerankingManager._is_endpoint_available: Cannot connect to reranker API at {endpoint}")
            return False
        
        except requests.exceptions.RequestException as error:
            log(DEBUG, f"RerankingManager._is_endpoint_available: HEAD check failed for {endpoint}: {error}; trying /models")

        try:
            models_url = f"{endpoint.rstrip('/')}/models"
            response = requests.get(
                models_url,
                timeout=RERANKING_AVAILABILITY_TIMEOUT_SECONDS,
            )
            response.raise_for_status()

            try:
                data = response.json()
            except Exception:
                return True

            if isinstance(data, list):
                models = data
            elif isinstance(data, dict) and isinstance(data.get("models"), list):
                models = data["models"]
            elif isinstance(data, dict) and isinstance(data.get("data"), list):
                models = data["data"]
            elif isinstance(data, dict):
                models = list(data.keys())
            else:
                models = []

            for model_entry in models:
                if isinstance(model_entry, str) and model_entry == self._model:
                    return True
                if isinstance(model_entry, dict) and self._model in (
                    model_entry.get("id"),
                    model_entry.get("name"),
                    model_entry.get("model"),
                ):
                    return True

            log(ERROR, f"RerankingManager._is_endpoint_available: Model '{self._model}' not listed at {models_url}")
            return False
        
        except requests.exceptions.RequestException as error:
            log(WARNING, f"RerankingManager._is_endpoint_available: /models fallback failed for {endpoint}: {error}")
            return False

    def is_available(self) -> bool:
        """Check all configured endpoints and remove unavailable endpoints."""
        if not self._endpoints:
            self.initialize()

        with ThreadPoolExecutor(max_workers=len(self._endpoints)) as executor:
            availability = executor.map(
                self._is_endpoint_available,
                self._endpoints,
            )
            available_endpoints = [
                endpoint
                for endpoint, is_available in zip(self._endpoints, availability)
                if is_available
            ]
        self._endpoints = available_endpoints
        return bool(self._endpoints)

    def _randomize_endpoints(self) -> List[str]:
        """Return the configured endpoints in a random order."""
        endpoints = self._endpoints.copy()
        random.shuffle(endpoints)
        return endpoints

    @staticmethod
    def _extract_score(result: Any) -> Optional[float]:
        """Extract a score from supported reranker response shapes."""
        score: Optional[float] = None
        if isinstance(result, dict):
            if result.get("data"):
                score = result["data"][0].get("score")
            elif isinstance(result.get("scores"), list) and result["scores"]:
                score = result["scores"][0]
            elif "score" in result:
                score = result.get("score")
        elif isinstance(result, list) and result and isinstance(result[0], dict):
            score = result[0].get("score")
        return score

    def score(self, query: str, text: str) -> Optional[float]:
        """Score one text against a query using the configured reranking model."""
        if not self._endpoints:
            self.initialize()

        endpoints: List[str] = self._randomize_endpoints()

        safe_query = query if query is not None else ""
        safe_text = text if text is not None else ""
        if len(safe_text) > 24000:
            safe_text = safe_text[:24000]

        payloads = [
            {"model": self._model, "text_1": safe_query, "text_2": safe_text},
            {"model": self._model, "text_1": [safe_query], "text_2": [safe_text]},
            {"model": self._model, "inputs": [{"text_1": safe_query, "text_2": safe_text}]},
            {"model": self._model, "inputs": [safe_query, safe_text]},
            {"model": self._model, "texts": [safe_query, safe_text]},
            {"model": self._model, "input": {"text_1": safe_query, "text_2": safe_text}},
            {"model": self._model, "instances": [{"source": safe_query, "target": safe_text}]},
        ]

        for index, payload in enumerate(payloads):
            try:
                response = requests.post(
                    f"{endpoints[0].rstrip('/')}/score",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=RERANKING_REQUEST_TIMEOUT_SECONDS,
                )
            except requests.exceptions.ConnectionError as error:
                log(ERROR, f"RerankingManager.score: Cannot connect to reranker API at {endpoints[0]}: {error}")
                return None
            except requests.exceptions.Timeout as error:
                log(ERROR, f"RerankingManager.score: Timeout connecting to reranker API at {endpoints[0]}: {error}")
                return None

            if response.status_code == 200:
                try:
                    score = self._extract_score(response.json())
                except Exception:
                    log(WARNING, "RerankingManager.score: Reranker returned non-JSON response")
                    return None
                if score is not None:
                    return score
                log(WARNING, f"RerankingManager.score: Unexpected response shape for payload #{index}: {response.text}")
                return None

            if response.status_code == 400:
                try:
                    body = response.json()
                except Exception:
                    body = response.text
                log(ERROR, f"RerankingManager.score: payload #{index} returned 400. Payload={json.dumps(payload)} Response={body}")
                continue

            try:
                response.raise_for_status()
            except requests.exceptions.RequestException as error:
                log(ERROR, f"RerankingManager.score: API error: {error}. Response: {response.text}")
                return None

        log(WARNING, "RerankingManager.score: All candidate payloads failed")
        return None

    @staticmethod
    def _extract_scores(result: Any) -> List[Optional[float]]:
        """Extract ordered scores from a batch reranker response."""
        scores: List[Optional[float]] = []
        if isinstance(result, dict) and isinstance(result.get("data"), list):
            scores = [
                item.get("score") if isinstance(item, dict) else item
                for item in result["data"]
            ]
        elif isinstance(result, dict) and isinstance(result.get("scores"), list):
            scores = result["scores"]
        elif isinstance(result, list):
            scores = [
                item.get("score") if isinstance(item, dict) else item
                for item in result
            ]
        return scores

    def _multi_score_for_endpoint(
        self,
        endpoint: str,
        query: str,
        texts: List[str],
    ) -> List[Optional[float]]:
        """Score multiple texts with one batch request against one endpoint."""
        scores: List[Optional[float]] = []

        if not texts:
            return scores

        if not endpoint:
            log(WARNING, "RerankingManager._multi_score_for_endpoint: Endpoint is empty")
            return scores

        if not self._model:
            self.initialize()

        safe_query = query if query is not None else ""
        safe_texts = [text if text is not None else "" for text in texts]
        safe_texts = [text[:24000] for text in safe_texts]
        payload = {
            "model": self._model,
            "text_1": [safe_query] * len(safe_texts),
            "text_2": safe_texts,
        }

        try:
            response = requests.post(
                f"{endpoint.rstrip('/')}/score",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=RERANKING_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            scores = self._extract_scores(response.json())
            if len(scores) != len(texts):
                log(WARNING, f"RerankingManager._multi_score_for_endpoint: Expected {len(texts)} scores, received {len(scores)}")
                scores = []
                
        except requests.exceptions.RequestException as error:
            log(ERROR, f"RerankingManager._multi_score_for_endpoint: Batch reranker request failed at {endpoint}: {error}")
        except Exception as error:
            log(ERROR, f"RerankingManager._multi_score_for_endpoint: Unexpected batch scoring error: {error}")

        return scores

    def multi_score(self, query: str, texts: List[str]) -> List[Optional[float]]:
        """Score texts concurrently by distributing them across configured endpoints."""
        scores: List[Optional[float]] = [None] * len(texts)
        perf_id: str = f"rer_multi_score_{id(texts)}"
        log(PERF, f"RerankingManager.multi_score: start, texts={len(texts)}, endpoints={len(self._endpoints)}", perf_id=f"{perf_id}_start")

        if texts:
            if not self._endpoints:
                self.initialize()

            endpoints: List[str] = self._randomize_endpoints()[:len(texts)]
            if endpoints:
                chunk_size: int = (len(texts) + len(endpoints) - 1) // len(endpoints)
                chunks = [
                    (endpoint, start, texts[start:start + chunk_size])
                    for index, endpoint in enumerate(endpoints)
                    for start in [index * chunk_size]
                    if start < len(texts)
                ]

                max_workers: int = min(32, len(chunks))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    pending = {
                        executor.submit(
                            self._multi_score_for_endpoint,
                            endpoint,
                            query,
                            chunk_texts,
                        ): start
                        for endpoint, start, chunk_texts in chunks
                    }
                    
                    for future in as_completed(pending):
                        start = pending[future]

                        try:
                            chunk_scores = future.result()
                            scores[start:start + len(chunk_scores)] = chunk_scores
                            
                        except Exception as error:
                            log(ERROR, f"RerankingManager.multi_score: Endpoint chunk failed: {error}")

        log(PERF, f"RerankingManager.multi_score: done, scored={sum(score is not None for score in scores)}/{len(texts)}", perf_id=f"{perf_id}_done", perf_since_id=f"{perf_id}_start")
        return scores


log_end_file(__file__)


if __name__ == "__main__":
    from src.utils.log import FATAL, INFO, log_exit, log_init
    from src.utils.unit_test import init_unit_test

    init_unit_test()
    log_init()

    try:
        manager = RerankingManager.instance()

        query = (
            "Assess whether the applicant faces a real risk of persecution or serious harm "
            "upon return to Kosovo because of threats from a former domestic partner, and "
            "whether the state authorities can provide effective protection."
        )
        single_text = (
            "The applicant faces a real risk of persecution and serious harm upon return to "
            "Kosovo because her former domestic partner repeatedly threatened to kill her. "
            "She reported the threats to the police and submitted threatening messages, but "
            "the authorities did not investigate the former partner or provide effective state "
            "protection. After moving to another municipality, the former partner found her "
            "again through relatives. This evidence supports the conclusion that the applicant "
            "cannot obtain effective protection from persecution or serious harm from her former "
            "partner in Kosovo."
        )
        batch_texts = [
            (
                "The applicant's former domestic partner threatened to kill her repeatedly after "
                "she ended the relationship. She sought help from the Kosovo police and provided "
                "the threatening messages, but the police did not investigate or issue an effective "
                "protection order. The former partner later located her after she moved. The "
                "applicant therefore faces a real risk of serious harm on return and cannot rely "
                "on effective state protection in Kosovo."
            ),
            (
                "The country information confirms that victims of domestic violence in Kosovo "
                "may report threats to the police and request protection orders. In practice, "
                "however, police investigations and court protection measures are inconsistently "
                "enforced. An abusive former partner who ignores warnings may continue to locate "
                "and threaten a victim, particularly where the victim lacks family support. "
                "These conditions are directly relevant to whether the applicant would receive "
                "effective state protection from persecution or serious harm."
            ),
        ]

        single_score = manager.score(query, single_text)
        if single_score is None:
            raise AssertionError(f"Unexpected single score: {single_score}")

        batch_scores = manager.multi_score(query, batch_texts)
        if len(batch_scores) != 2 or any(score is None for score in batch_scores):
            raise AssertionError(f"Unexpected batch scores: {batch_scores}")

        individual_batch_scores = [
            manager.score(query, text)
            for text in batch_texts
        ]
        if len(individual_batch_scores) != len(batch_texts) or any(
            score is None for score in individual_batch_scores
        ):
            raise AssertionError(f"Unexpected individual scores: {individual_batch_scores}")

        score_differences = [
            batch_score - individual_score
            for batch_score, individual_score in zip(batch_scores, individual_batch_scores)
        ]
        log(INFO, f"RerankingManager real-call tests passed: single={single_score}")
        log(INFO, f"Batch scores: {batch_scores}")
        log(INFO, f"Individual scores: {individual_batch_scores}")
        log(INFO, f"Batch minus individual differences: {score_differences}")

    except Exception as error:
        log(FATAL, f"RerankingManager unit tests failed: {error}")

    finally:
        log_exit()
