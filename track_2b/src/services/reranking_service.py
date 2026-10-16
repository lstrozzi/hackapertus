from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

from typing import List, Optional, Dict, Any
import traceback
import json
import time
import inspect
import requests

from src.interfaces.reranking_interface import RerankingInterface
from src.interfaces.documents_interface import DocumentsInterface
from src.config.constants import (
    MAX_RERANKING_OUTPUT_FRAG, MIN_RERANKING_OUTPUT_FRAG, MIN_RERANKING_SCORE_FRAG,  # Fragment-level operations
    MAX_RERANKING_OUTPUT_DOC, MIN_RERANKING_SCORE_DOC, MIN_RERANKING_OUTPUT_DOC,  # Document-level operations
    RERANKING_REQUEST_TIMEOUT_SECONDS
)
from src.core.llm import LlmManager
from src.core.llm.reranking import RerankingManager
from src.utils.log import log, INFO, DEBUG, WARNING, ERROR, PERF, find_goal_in_stack_frame


class RerankingService(RerankingInterface):
    """Rerank fragments and documents through the configured external API."""

    def __init__(self):
        self._log_prefix = 'RerankingService'
        self._documents_service: Optional[DocumentsInterface] = None

    def _get_documents_service(self) -> DocumentsInterface:
        """Get or initialize the documents service singleton."""
        if self._documents_service is None:
            from src.interfaces.documents_interface import DocumentsInterface
            from src.services.documents_service import get_documents_service
            self._documents_service: DocumentsInterface = get_documents_service()
        return self._documents_service

    @staticmethod
    def _track_agent_completion(completed: int, total: int) -> None:
        """Advance the calling agent's completion block when one is available."""
        if total <= 0:
            return
        try:
            goal = find_goal_in_stack_frame(inspect.stack())
            track_completion = getattr(getattr(goal, "goal_target_agent", None), "_completion_track", None)
            if track_completion is not None:
                track_completion(completed / total)
        except Exception as e:
            log(DEBUG, f"{RerankingService.__name__}._track_agent_completion: Unable to update agent completion: {e}")

    def _is_reranker_api_available(self, endpoint: str, model: Optional[str] = None) -> bool:
        """Check if the reranker API is reachable and available.

        This first attempts a lightweight HEAD to the `/score` path. If that
        returns a 405 (Method Not Allowed) or otherwise indicates the
        endpoint does not support the HEAD/score check, this method will try
        the `/models` endpoint as a fallback and (optionally) verify that the
        configured `model` is listed there.

        Args:
            endpoint: The reranker API endpoint URL (e.g. 'http://host:8364/v1')
            model: Optional model identifier to verify exists on the server

        Returns:
            True if the API looks available (and the model exists if provided),
            False otherwise
        """
        # First attempt a lightweight HEAD to the score path
        try:
            response = requests.head(f"{endpoint}/score", timeout=5)
            if response.status_code == 405:
                log(DEBUG, f"{self._log_prefix}._is_reranker_api_available: HEAD {endpoint}/score returned 405, will try /models fallback")
            else:
                response.raise_for_status()
                return True
        except requests.exceptions.ConnectionError:
            log(ERROR, f"{self._log_prefix}._is_reranker_api_available: Cannot connect to reranker API at {endpoint}")
            return False
        except requests.exceptions.RequestException as e:
            # If we get an HTTP error with 405 we fall through to models check,
            # otherwise log and continue to attempt /models as a best-effort.
            if getattr(e, 'response', None) is not None and e.response.status_code == 405:
                log(DEBUG, f"{self._log_prefix}._is_reranker_api_available: HEAD returned 405, trying /models: {e}")
            else:
                log(DEBUG, f"{self._log_prefix}._is_reranker_api_available: HEAD to /score failed: {e}. Trying /models as fallback.")

        # Fallback: try the /models endpoint to see if service is responding
        try:
            models_url = f"{endpoint.rstrip('/')}/models"
            r = requests.get(models_url, timeout=5)
            r.raise_for_status()
            # Optionally inspect returned models to ensure requested model exists
            if model:
                try:
                    data = r.json()
                except Exception:
                    log(DEBUG, f"{self._log_prefix}._is_reranker_api_available: /models returned non-json response")
                    return True

                # Normalize possible shapes: list, dict with 'models' or 'data'
                models_list = []
                if isinstance(data, list):
                    models_list = data
                elif isinstance(data, dict):
                    if 'models' in data and isinstance(data['models'], list):
                        models_list = data['models']
                    elif 'data' in data and isinstance(data['data'], list):
                        models_list = data['data']
                    else:
                        # Some servers may return a dict of model-name->meta
                        models_list = list(data.keys())

                # Try to match model identifier in returned entries
                model_found = False
                for m in models_list:
                    if isinstance(m, str) and m == model:
                        model_found = True
                        break
                    if isinstance(m, dict):
                        # common keys that may hold model id / name
                        if m.get('id') == model or m.get('name') == model or m.get('model') == model:
                            model_found = True
                            break

                if not model_found:
                    log(ERROR, f"{self._log_prefix}._is_reranker_api_available: Model '{model}' not listed at {models_url}")
                    return False

            return True
        
        except requests.exceptions.RequestException as e:
            log(ERROR, f"{self._log_prefix}._is_reranker_api_available: /models fallback failed for {endpoint}: {e}")
            return False

    def _call_reranker_api(self, query: str, text: str, endpoint: str, model: str) -> Optional[float]:
        """Call the reranker API and return the score."""
        url = f"{endpoint.rstrip('/')}/score"
        headers = {"Content-Type": "application/json"}

        # Ensure query/text are strings (some integrations send None)
        safe_query = query if query is not None else ""
        safe_text = text if text is not None else ""

        # Truncate text to avoid exceeding model context limits.
        # BAAI/bge-reranker-v2-m3 has a maximum of 8192 tokens; at ~4 chars/token
        # for mixed-language content this caps the text at roughly 6000 tokens,
        # leaving comfortable headroom for the query and special tokens.
        MAX_RERANKER_TEXT_CHARS = 24000
        if len(safe_text) > MAX_RERANKER_TEXT_CHARS:
            safe_text = safe_text[:MAX_RERANKER_TEXT_CHARS]

        # candidate payload shapes to try when the server returns Bad Request (400)
        candidate_payloads = [
            {"model": model, "text_1": safe_query, "text_2": safe_text},
            {"model": model, "text_1": [safe_query], "text_2": [safe_text]},
            {"model": model, "inputs": [{"text_1": safe_query, "text_2": safe_text}]},
            {"model": model, "inputs": [safe_query, safe_text]},
            {"model": model, "texts": [safe_query, safe_text]},
            {"model": model, "input": {"text_1": safe_query, "text_2": safe_text}},
            {"model": model, "instances": [{"source": safe_query, "target": safe_text}]}
        ]

        def _extract_score_from_result(result: Any) -> Optional[float]:
            if not result:
                return None
            if isinstance(result, dict):
                if "data" in result and result["data"]:
                    item = result["data"][0]
                    return item.get("score")
                # some runners return {'scores': [val]} or {'score': val}
                if "scores" in result and isinstance(result["scores"], list) and len(result["scores"])>0:
                    return result["scores"][0]
                if "score" in result:
                    return result.get("score")
            if isinstance(result, list) and len(result) > 0:
                first = result[0]
                if isinstance(first, dict) and "score" in first:
                    return first.get("score")
            return None

        try:
            # Try the primary payload first
            for idx, payload in enumerate(candidate_payloads):
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=RERANKING_REQUEST_TIMEOUT_SECONDS)
                    
                except requests.exceptions.ConnectionError as e:
                    log(ERROR, f"{self._log_prefix}._call_reranker_api: Cannot connect to reranker API at {endpoint}. Is the service running? Error: {e}")
                    return None
                except requests.exceptions.Timeout as e:
                    log(ERROR, f"{self._log_prefix}._call_reranker_api: Timeout connecting to reranker API at {endpoint}. Error: {e}")
                    return None

                # If successful, try to parse
                if response.status_code == 200:
                    try:
                        result = response.json()
                    except Exception:
                        log(WARNING, f"{self._log_prefix}._call_reranker_api: Non-JSON response from reranker API (status 200)")
                        return None

                    score = _extract_score_from_result(result)
                    if score is not None:
                        return score
                    else:
                        log(WARNING, f"{self._log_prefix}._call_reranker_api: Unexpected 200 response shape for payload #{idx}: {result}")
                        return None

                # If bad request, log response and try next payload
                if response.status_code == 400:
                    body = None
                    try:
                        body = response.json()
                    except Exception:
                        body = response.text
                    # Log full payload and response at ERROR to help diagnose request shape mismatches
                    try:
                        payload_str = json.dumps(payload)
                    except Exception:
                        payload_str = str(payload)
                    resp_str = str(body)
                    # Truncate very large bodies to avoid excessive logs
                    max_len = 8000
                    if len(payload_str) > max_len:
                        payload_log = payload_str[:max_len] + '...[truncated]'
                    else:
                        payload_log = payload_str
                    if len(resp_str) > max_len:
                        resp_log = resp_str[:max_len] + '...[truncated]'
                    else:
                        resp_log = resp_str
                    log(ERROR, f"{self._log_prefix}._call_reranker_api: payload #{idx} returned 400. Payload={payload_log} Response={resp_log}")
                    # try next candidate payload
                    continue

                # For other non-200 statuses, log and abort
                try:
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    # include response text for debugging
                    resp_text = None
                    try:
                        resp_text = response.text
                    except Exception:
                        resp_text = '<no response body>'
                    log(ERROR, f"{self._log_prefix}._call_reranker_api: API error: {e}. Response: {resp_text}")
                    return None

            # If we exhausted payloads without success
            log(WARNING, f"{self._log_prefix}._call_reranker_api: All candidate payloads failed or returned unexpected responses")
            return None

        except Exception as e:
            log(ERROR, f"{self._log_prefix}._call_reranker_api: Unexpected error: {e}. Traceback:\n{traceback.format_exc()}")
            return None

    def rerank(self, query: str, fragments: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Rerank fragments using external API.

        Args:
            query: Query text to compare against fragments.
            fragments: List of fragment dictionaries with the following fields:
                - "id": string, mandatory, same as fragment_id
                - "score": float, optional -- the RAG embeddings similarity score
                - "text": string, optional -- the entire fragment's text

        Returns:
            Ordered list of dicts with keys: id, score, ranking, text, or None if reranking fails.
        """
        if not fragments:
            log(WARNING, f"{self._log_prefix}.rerank: No fragments provided")
            return None
        
        _t_start = time.perf_counter()
        try:
            # Get reranking settings
            reranking_settings:dict = LlmManager.instance().get_reranking_settings()
            if not reranking_settings or not reranking_settings.get("enabled"):
                log(INFO, f"{self._log_prefix}.rerank: Reranking is disabled in settings")
                return None
            
            top_n:int = reranking_settings.get("top_n", None)
            log(INFO, f"{self._log_prefix}.rerank: query={query}, fragments={len(fragments)}")
            
            # Verify API is reachable before processing fragments
            if not RerankingManager.instance().is_available():
                log(ERROR, f"{self._log_prefix}.rerank: Reranker API is unavailable. Skipping reranking.")
                return None
            
            # Filter fragments that have text
            fragments_with_text = [f for f in fragments if f.get('text')]
            dropped_no_text = len(fragments) - len(fragments_with_text)
            if not fragments_with_text:
                log(INFO, f"{self._log_prefix}.rerank: All {len(fragments)} fragments dropped — none had text")
                return None
            if dropped_no_text > 0:
                log(INFO, f"{self._log_prefix}.rerank: {dropped_no_text}/{len(fragments)} fragments dropped (no text); {len(fragments_with_text)} sent to reranker")
            else:
                log(DEBUG, f"{self._log_prefix}.rerank: Reranking {len(fragments_with_text)} fragments with text...")
            
            batch_scores = RerankingManager.instance().multi_score(
                query,
                [frag['text'] for frag in fragments_with_text],
            )
            log(PERF, f"{self._log_prefix}.rerank: batch scores received, {len(batch_scores)} scores", perf_id=f"rerank_scores_{id(fragments)}")
            results = [
                {
                    'id': frag['id'],
                    'original_score': frag.get('score'),
                    'score': ranking,
                    'ranking': ranking,
                    'text': frag['text'],
                }
                for frag, ranking in zip(fragments_with_text, batch_scores)
                if ranking is not None
            ]
            self._track_agent_completion(len(results), len(fragments_with_text))
            
            if not results:
                log(WARNING, f"{self._log_prefix}.rerank: No fragments were successfully scored")
                return None
            
            # Sort by ranking descending
            results.sort(key=lambda x: x['ranking'], reverse=True)
            log(PERF, f"{self._log_prefix}.rerank: results sorted, {len(results)} results", perf_id=f"rerank_sorted_{id(fragments)}", perf_since_id=f"rerank_scores_{id(fragments)}")

            # Filter out fragments below configured minimum ranking (fragment-level filtering)
            min_ranking = MIN_RERANKING_SCORE_FRAG if MIN_RERANKING_SCORE_FRAG is not None else 0.0
            filtered = [r for r in results if r.get('ranking') is not None and r['ranking'] >= min_ranking]
            log(INFO, f"{self._log_prefix}.rerank: Score filter (min_ranking={min_ranking}): {len(filtered)}/{len(results)} fragments passed")
            if (not filtered or len(filtered) == 0) and MIN_RERANKING_SCORE_FRAG is None:
                log(WARNING, f"{self._log_prefix}.rerank: No fragments meet the minimum reranking score threshold, using MIN_RERANKING_OUTPUT_FRAG to ensure document diversity after aggregation. The top score is {results[0]['ranking']:.5f}")
                filtered = results[:MIN_RERANKING_OUTPUT_FRAG]  # Take top N fragments to ensure multiple distinct documents survive aggregation
            
            # DOCUMENT-LEVEL DIVERSITY: Ensure we keep enough fragments to cover MIN_RERANKING_OUTPUT distinct documents
            # This prevents all returned fragments from belonging to just 1-2 documents
            if len(filtered) > 0:
                # Extract document IDs from fragment IDs
                # Fragment format examples: "docid-XXXX-c1", "docid-XXXX-c1-p2", "docid-XXXX-p5", "docid-XXXX-k3"
                # We need to remove ALL suffixes to get the base document ID "docid-XXXX"
                def extract_doc_id(frag_id: str) -> Optional[str]:
                    """Extract base document ID from fragment ID by removing all suffixes."""
                    if not frag_id or not frag_id.startswith('docid-'):
                        return None
                    # Remove all "-X" suffixes where X starts with c, p, s, or k
                    parts = frag_id.split('-')
                    # parts[0] = 'docid', parts[1] = document_number, parts[2...] = suffixes
                    if len(parts) >= 3:
                        return f"{parts[0]}-{parts[1]}"  # docid-XXXX
                    return frag_id
                
                # Extract document IDs from filtered list
                doc_ids_in_filtered = set()
                for r in filtered:
                    frag_id = r.get('id', '')
                    doc_id = extract_doc_id(frag_id)
                    if doc_id:
                        doc_ids_in_filtered.add(doc_id)
                
                log(DEBUG, f"{self._log_prefix}.rerank: Initial filtered list has {len(doc_ids_in_filtered)} distinct documents: {sorted(doc_ids_in_filtered)}")
                
                # If we have fewer distinct documents than MIN_RERANKING_OUTPUT, expand filtered list
                if len(doc_ids_in_filtered) < MIN_RERANKING_OUTPUT_DOC and len(filtered) < len(results):
                    log(DEBUG, f"{self._log_prefix}.rerank: Only {len(doc_ids_in_filtered)} distinct documents in top {len(filtered)} fragments, expanding to ensure {MIN_RERANKING_OUTPUT_DOC} documents")
                    
                    # Add more fragments until we reach MIN_RERANKING_OUTPUT distinct documents
                    additional_needed = MIN_RERANKING_OUTPUT_DOC - len(doc_ids_in_filtered)
                    for r in results[len(filtered):]:
                        if additional_needed <= 0:
                            break
                        frag_id = r.get('id', '')
                        doc_id = extract_doc_id(frag_id)
                        if doc_id and doc_id not in doc_ids_in_filtered:
                            filtered.append(r)
                            doc_ids_in_filtered.add(doc_id)
                            additional_needed -= 1
                            log(DEBUG, f"{self._log_prefix}.rerank: Added fragment {frag_id} (doc: {doc_id}) to ensure diversity")
                    
                    log(INFO, f"{self._log_prefix}.rerank: Expanded to {len(filtered)} fragments covering {len(doc_ids_in_filtered)} distinct documents")

            # Determine output limit: prefer explicit top_n, otherwise use configured max (fragment-level limit)
            limit = top_n if top_n and top_n > 0 else MAX_RERANKING_OUTPUT_FRAG
            try:
                if limit and limit > 0:
                    filtered = filtered[:limit]
            except Exception:
                # if constants are malformed, fall back to returning all filtered
                pass

            log(INFO, f"{self._log_prefix}.rerank: Reranked {len(filtered)} fragments (min_ranking={min_ranking}, limit={limit})")
            log(PERF, f"{self._log_prefix}.rerank: post-processing complete, {len(filtered)} fragments", perf_id=f"rerank_done_{id(fragments)}", perf_since_id=f"rerank_scores_{id(fragments)}")
            _elapsed_ms = (time.perf_counter() - _t_start) * 1000.0
            _goal = find_goal_in_stack_frame(inspect.stack())
            if _goal is not None:
                _goal.add_perf_reranking_ms(_elapsed_ms)
            return filtered
            
        except Exception as e:
            log(ERROR, f"{self._log_prefix}.rerank: Error: {e}\n{traceback.format_exc()}")
            return None

    def rerank_multilang(self, query: str, fragments: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Rerank fragments using multi-language queries.
        
        Translates the query into multiple languages (EN, DE, FR, IT) and performs
        reranking against each translation. For each fragment, the highest score
        across all translations is used, ensuring cross-language documents get
        fair scoring.
        
        Args:
            query: Query text to compare against fragments.
            fragments: List of fragment dictionaries with "id", "score", "text" fields.
            
        Returns:
            Ordered list of reranked fragments, or None if reranking fails.
        """
        from src.config.constants import MULTILANG_SEARCH_ENABLED, MULTILANG_LANGUAGES
        from src.services.translation_service import get_translation_service
        
        if not fragments:
            log(WARNING, f"{self._log_prefix}.rerank_multilang: No fragments provided")
            return None
        
        _t_start = time.perf_counter()
        try:
            # If multi-language is disabled, fall back to single-language rerank
            if not MULTILANG_SEARCH_ENABLED:
                log(DEBUG, f"{self._log_prefix}.rerank_multilang: Multi-language reranking disabled")
                return self.rerank(query, fragments)
            
            log(INFO, f"{self._log_prefix}.rerank_multilang: Starting multi-language reranking with {len(fragments)} fragments in {MULTILANG_LANGUAGES}")
            
            # Get translations for all target languages
            translation_service = get_translation_service()
            translations = translation_service.translate_to_languages(query, MULTILANG_LANGUAGES)
            
            # Collect reranking results for each translation
            lang_results: Dict[str, Dict[str, float]] = {}  # lang -> {frag_id -> score}
            
            for lang, trans_query in translations.items():
                log(DEBUG, f"{self._log_prefix}.rerank_multilang: [{lang}] Reranking with '{trans_query[:60]}...'")
                lang_perf_id: str = f"rerank_multilang_{id(fragments)}_{lang}"
                log(PERF, f"{self._log_prefix}.rerank_multilang: [{lang}] start", perf_id=f"{lang_perf_id}_start")
                
                # Rerank with this translation
                result = self.rerank(trans_query, fragments)
                if result is not None:
                    lang_results[lang] = {r['id']: r.get('ranking', 0.0) for r in result}
                    log(DEBUG, f"{self._log_prefix}.rerank_multilang: [{lang}] Got {len(result)} results")
                else:
                    log(WARNING, f"{self._log_prefix}.rerank_multilang: [{lang}] Reranking returned None")
                log(PERF, f"{self._log_prefix}.rerank_multilang: [{lang}] done", perf_id=f"{lang_perf_id}_done", perf_since_id=f"{lang_perf_id}_start")
            
            if not lang_results:
                log(WARNING, f"{self._log_prefix}.rerank_multilang: No results from any language")
                return self.rerank(query, fragments)  # Fall back to original query
            
            # Combine results: for each fragment, use the max score across all translations
            combined_scores: Dict[str, float] = {}
            frag_by_id: Dict[str, Dict[str, Any]] = {f['id']: f for f in fragments}
            
            for frag in fragments:
                frag_id = frag['id']
                max_score = 0.0
                best_lang = None
                
                for lang, scores in lang_results.items():
                    if frag_id in scores and scores[frag_id] > max_score:
                        max_score = scores[frag_id]
                        best_lang = lang
                
                combined_scores[frag_id] = max_score
                if best_lang:
                    log(DEBUG, f"{self._log_prefix}.rerank_multilang: {frag_id} best score {max_score:.5f} from [{best_lang}]")
            
            # Build final results with combined scores
            results = []
            for frag in fragments:
                frag_id = frag['id']
                results.append({
                    'id': frag_id,
                    'original_score': frag.get('score'),
                    'score': combined_scores.get(frag_id, 0.0),
                    'ranking': combined_scores.get(frag_id, 0.0),
                    'text': frag.get('text')
                })
            
            # Sort by combined ranking
            results.sort(key=lambda x: x['ranking'], reverse=True)
            
            log(INFO, f"{self._log_prefix}.rerank_multilang: Combined {len(MULTILANG_LANGUAGES)} languages -> {len(results)} fragments")
            _elapsed_ms = (time.perf_counter() - _t_start) * 1000.0
            _goal = find_goal_in_stack_frame(inspect.stack())
            if _goal is not None:
                _goal.add_perf_reranking_ms(_elapsed_ms)
            return results
            
        except Exception as e:
            log(ERROR, f"{self._log_prefix}.rerank_multilang: Error: {e}\n{traceback.format_exc()}")
            # Fall back to single-language rerank on error
            return self.rerank(query, fragments)

    def rerank_documents(self, query: str, document_ids: List[str]) -> Optional[List[Dict[str, Any]]]:
        """Rerank documents by their chapters using parallel processing.

        This method fetches all chapters for the provided documents from the documents
        service, reranks them in parallel using ThreadPoolExecutor, and returns the
        MAX ranking score achieved by any chapter within each document.

        Args:
            query: Query text to compare against document chapters.
            document_ids: List of document IDs to rerank.

        Returns:
            Ordered list of dicts with keys: id (document id), ranking (max chapter ranking),
            or None if reranking cannot be performed.
        """
        if not document_ids:
            log(WARNING, f"{self._log_prefix}.rerank_documents: No document IDs provided")
            return None

        _t_start = time.perf_counter()
        try:
            # Get reranking settings
            reranking_settings: dict = LlmManager.instance().get_reranking_settings()
            if not reranking_settings or not reranking_settings.get("enabled"):
                log(INFO, f"{self._log_prefix}.rerank_documents: Reranking is disabled in settings")
                return None

            model: str = reranking_settings.get("model")
            endpoint: str = reranking_settings.get("endpoint")
            top_n: int = reranking_settings.get("top_n", None)

            if not model or not endpoint:
                log(ERROR, f"{self._log_prefix}.rerank_documents: Model or endpoint not configured")
                return None

            log(INFO, f"{self._log_prefix}.rerank_documents: query={query}, documents={len(document_ids)}, endpoint={endpoint}, model={model}")

            # Verify API is reachable before processing
            if not RerankingManager.instance().is_available():
                log(ERROR, f"{self._log_prefix}.rerank_documents: Reranker API is unavailable. Skipping reranking.")
                return None

            # Fetch all chapters for the provided documents
            docs_service = self._get_documents_service()
            chapters_by_doc = docs_service.get_chapters_for_documents(document_ids)
            if not chapters_by_doc:
                log(WARNING, f"{self._log_prefix}.rerank_documents: No chapters found for documents {document_ids}")
                return None

            log(DEBUG, f"{self._log_prefix}.rerank_documents: Fetched chapters for {len(chapters_by_doc)} documents")

            # Prepare all chapters for reranking
            all_chapters = []
            chapter_to_doc = {}  # Map chapter ID to document ID
            for doc_id, chapters in chapters_by_doc.items():
                for chapter in chapters:
                    all_chapters.append(chapter)
                    chapter_to_doc[chapter['id']] = doc_id

            if not all_chapters:
                log(WARNING, f"{self._log_prefix}.rerank_documents: No chapters with text to rerank")
                return None

            log(DEBUG, f"{self._log_prefix}.rerank_documents: Reranking {len(all_chapters)} chapters in parallel...")

            chapter_scores = RerankingManager.instance().multi_score(
                query,
                [chapter.get('text', '') for chapter in all_chapters],
            )
            chapter_rankings = {
                chapter['id']: ranking
                for chapter, ranking in zip(all_chapters, chapter_scores)
                if ranking is not None
            }

            if not chapter_rankings:
                log(WARNING, f"{self._log_prefix}.rerank_documents: No chapters were successfully scored")
                return None

            # Aggregate chapter rankings to document rankings (take MAX per document)
            doc_rankings = {}  # Map document ID to max ranking score
            for chapter_id, ranking in chapter_rankings.items():
                doc_id = chapter_to_doc.get(chapter_id)
                if doc_id:
                    if doc_id not in doc_rankings:
                        doc_rankings[doc_id] = ranking
                    else:
                        # Keep the maximum ranking for this document
                        doc_rankings[doc_id] = max(doc_rankings[doc_id], ranking)

            # Build result list with documents and their max chapter rankings
            results = []
            for doc_id, max_ranking in doc_rankings.items():
                results.append({
                    'id': doc_id,
                    'ranking': max_ranking
                })

            # Sort by ranking descending
            results.sort(key=lambda x: x['ranking'], reverse=True)

            # Filter out documents below minimum ranking threshold (document-level filtering)
            min_ranking = MIN_RERANKING_SCORE_DOC if MIN_RERANKING_SCORE_DOC is not None else 0.0
            filtered = [r for r in results if r.get('ranking') is not None and r['ranking'] >= min_ranking]
            if not filtered or len(filtered) == 0:
                log(WARNING, f"{self._log_prefix}.rerank_documents: No documents meet the minimum reranking score threshold, using MIN_RERANKING_OUTPUT_DOC. Top score is {results[0]['ranking']:.5f}")
                filtered = results[:MIN_RERANKING_OUTPUT_DOC]

            # Apply output limit
            limit = top_n if top_n and top_n > 0 else MAX_RERANKING_OUTPUT_DOC
            try:
                if limit and limit > 0:
                    filtered = filtered[:limit]
            except Exception:
                pass

            log(INFO, f"{self._log_prefix}.rerank_documents: Reranked {len(filtered)} documents (min_ranking={min_ranking}, limit={limit})")
            _elapsed_ms = (time.perf_counter() - _t_start) * 1000.0
            _goal = find_goal_in_stack_frame(inspect.stack())
            if _goal is not None:
                _goal.add_perf_reranking_ms(_elapsed_ms)
            return filtered

        except Exception as e:
            log(ERROR, f"{self._log_prefix}.rerank_documents: Error: {e}\n{traceback.format_exc()}")
            return None

    def rerank_strings(self, query: str, strings: List[str], must_return_top_n: bool = True) -> Optional[List[Dict[str, Any]]]:
        """Rerank provided strings for the given query.

        Args:
            query: Query text to compare against strings.
            strings: List of strings to rerank.
            must_return_top_n: When True (default), forces at least MIN_RERANKING_OUTPUT_DOC results
                even if no string meets the minimum score threshold — useful for RAG where some
                result is always required. When False, returns None if no string passes the
                threshold — appropriate for hints where irrelevant results should be suppressed.
        Returns:
            Ordered list of dicts consistent with other reranking methods. Each dict
            contains: 'id' (string), 'original_index' (int), 'score' (float),
            'ranking' (float), and 'text' (the original string). Returns None on failure.
        """
        if not strings:
            log(WARNING, f"{self._log_prefix}.rerank_strings: No strings provided")
            return None

        _t_start = time.perf_counter()
        try:
            # Get reranking settings
            reranking_settings: dict = LlmManager.instance().get_reranking_settings()
            if not reranking_settings or not reranking_settings.get("enabled"):
                log(INFO, f"{self._log_prefix}.rerank_strings: Reranking is disabled in settings")
                return None

            model: str = reranking_settings.get("model")
            endpoint: str = reranking_settings.get("endpoint")
            top_n: int = reranking_settings.get("top_n", None)

            if not model or not endpoint:
                log(ERROR, f"{self._log_prefix}.rerank_strings: Model or endpoint not configured")
                return None

            log(INFO, f"{self._log_prefix}.rerank_strings: query={query}, strings={len(strings)}, endpoint={endpoint}, model={model}")

            # Verify API is reachable before processing
            if not RerankingManager.instance().is_available():
                log(ERROR, f"{self._log_prefix}.rerank_strings: Reranker API is unavailable. Skipping reranking.")
                return None

            string_scores = RerankingManager.instance().multi_score(query, strings)
            results = [
                (text, ranking, index)
                for index, (text, ranking) in enumerate(zip(strings, string_scores))
                if ranking is not None
            ]

            if not results:
                log(WARNING, f"{self._log_prefix}.rerank_strings: No strings were successfully scored")
                return None

            # Sort by ranking descending
            results.sort(key=lambda x: x[1], reverse=True)

            # Filter out strings below minimum ranking threshold (document-level filtering for string comparison)
            min_ranking = MIN_RERANKING_SCORE_DOC if MIN_RERANKING_SCORE_DOC is not None else 0.0
            filtered = [r for r in results if r[1] is not None and r[1] >= min_ranking]
            if not filtered or len(filtered) == 0:
                if not must_return_top_n:
                    log(INFO, f"{self._log_prefix}.rerank_strings: No strings meet the minimum reranking score threshold and must_return_top_n=False. Top score is {results[0][1]:.5f}")
                    return None
                log(WARNING, f"{self._log_prefix}.rerank_strings: No strings meet the minimum reranking score threshold, using MIN_RERANKING_OUTPUT_DOC. Top score is {results[0][1]:.5f}")
                filtered = results[:MIN_RERANKING_OUTPUT_DOC]

            # Apply output limit
            limit = top_n if top_n and top_n > 0 else MAX_RERANKING_OUTPUT_DOC
            try:
                if limit and limit > 0:
                    filtered = filtered[:limit]
            except Exception:
                pass

            # Build dicts consistent with `rerank` output
            ordered = []
            for s, score, idx in filtered:
                ordered.append({
                    'idx': idx,
                    'ranking': score,
                    'text': s
                })

            log(INFO, f"{self._log_prefix}.rerank_strings: Reranked {len(ordered)} strings (min_ranking={min_ranking}, limit={limit})")
            _elapsed_ms = (time.perf_counter() - _t_start) * 1000.0
            _goal = find_goal_in_stack_frame(inspect.stack())
            if _goal is not None:
                _goal.add_perf_reranking_ms(_elapsed_ms)
            return ordered

        except Exception as e:
            log(ERROR, f"{self._log_prefix}.rerank_strings: Error: {e}\n{traceback.format_exc()}")
            return None

# module-level singleton
_reranking_service_instance: Optional[RerankingInterface] = None


def get_reranking_service() -> RerankingInterface:
    global _reranking_service_instance
    if _reranking_service_instance is None:
        _reranking_service_instance = RerankingService()
    return _reranking_service_instance


def set_reranking_service(service: RerankingInterface) -> None:
    global _reranking_service_instance
    _reranking_service_instance = service


log_end_file(__file__)


if __name__ == '__main__':
    from src.utils.unit_test import init_unit_test
    from src.utils.log import log_init, log_exit
    init_unit_test()
    log_init()

    try:
        query = (
            "Assess whether the applicant faces a real risk of persecution or serious harm "
            "upon return to Kosovo because of threats from a former domestic partner, and "
            "whether the state authorities can provide effective protection."
        )
        texts = [
            (
                "The applicant's former domestic partner threatened to kill her repeatedly. "
                "She reported the threats to the Kosovo police and submitted the threatening "
                "messages, but the police did not investigate or issue an effective protection "
                "order. The former partner later located her after she moved, creating a real "
                "risk of serious harm on return."
            ),
            (
                "Country information indicates that Kosovo has legal protection orders for "
                "victims of domestic violence, but police investigations and enforcement are "
                "inconsistent. Victims without family support may remain exposed when an "
                "aggressor ignores warnings and continues making threats."
            ),
            (
                "The applicant entered Switzerland in March 2024 and was registered at the "
                "reception centre. Her identity documents were retained for routine asylum "
                "processing, and she received an appointment for the initial interview."
            ),
            (
                "The medical report records that the applicant received treatment for a broken "
                "wrist after an assault. It recommends physiotherapy and a follow-up examination "
                "within six weeks, but does not assess the risk posed by the former partner."
            ),
            (
                "The appeal authority must determine whether the applicant's account is "
                "credible, whether the feared harm reaches the threshold for persecution or "
                "serious harm, and whether effective state protection would be available in "
                "the country of origin."
            ),
        ]

        service = RerankingService()
        results = service.rerank_strings(query, texts)
        if not results:
            raise RuntimeError("Reranking returned no results")

        log(INFO, f"RerankingService real-call test ranked {len(results)} of {len(texts)} texts")
        for result in results:
            log(INFO, f"RerankingService result: score={result['ranking']:.6f}, text={result['text']}")

    except Exception as error:
        log(ERROR, f"RerankingService real-call test failed: {error}\n{traceback.format_exc()}")

    finally:
        log_exit()
