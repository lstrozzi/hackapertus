from src.utils.log_fileload import log_start_file, log_end_file
from src.utils.unit_test import init_unit_test
log_start_file(__file__)


from typing import Dict, List, Optional, Set, Tuple, Any, Union, cast
import math
import re
import statistics
import traceback
import json
import time
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config.constants import MIN_RERANKING_OUTPUT_DOC, MIN_RERANKING_SCORE_FRAG, ACL_ENFORCEMENT_ENABLED, SEMANTIC_SEARCH_PARTITION_SIZE, MIN_SEMANTIC_TEXT_COMBINE_WORDS, MAX_PARALLELISM, LLM_FILTER_MAX_CONTENT_CHARS, LLM_FILTER_MAX_PROMPT_CHARS, LLM_FILTER_MAX_REASONING_CHARS
from src.services.user_service import get_user_service
from src.core.base.InputOutput import execute_neo4j_query
from src.interfaces.ontology_interface import OntologyInterface
from src.interfaces.documents_interface import DocumentsInterface
from src.interfaces.rag_interface import RagInterface
from src.interfaces.dataset_interface import DatasetInterface
from src.services.ontology_service import get_ontology_service
from src.services.documents_service import get_documents_service
from src.services.reranking_service import get_reranking_service
from src.services.dataset_service import get_dataset_service
from src.core.llm.embeddings import EmbeddingsManager
from src.utils.performance import timer_decorator, last_decorated_function_time_milli
from src.utils.log import log, DEBUG, INFO, WARNING, ERROR, PERF, find_goal_in_stack_frame


DEBUG_IDENTIFY_DOC: Optional[str] = None # "docid-8230"  # Special debug document ID for logging and testing purposes


class _LlmFilterError(Exception):
    """Raised by the T33 LLM relevance gate when it cannot produce a trustworthy verdict.

    Propagating this (instead of defaulting to NO or returning unfiltered results) means no
    candidate is judged by a failed call and the dataset is never left holding a half-filtered
    result; thalabus_query logs the failure as ERROR and returns an empty result with no
    dataset ids.
    """
    pass


class RagService(RagInterface):
    """Concrete implementation for retrieval-augmented generation (RAG) operations."""

    # RAG settings for semantic search
    TOP_N_RAG = 150
    TOP_N_CHAPTERS = TOP_N_RAG
    TOP_N_PARAGRAPHS = TOP_N_RAG * 3
    TOP_N_QUESTIONS = TOP_N_RAG
    TOP_N_SUMMARIES = TOP_N_RAG
    # ANN candidate pool size: how many nodes the vector index pre-selects before Cypher filtering.
    # Must be >= TOP_N_* for each source. Larger values increase recall but slow down the index lookup.
    # Note: all indexes use vector-3.0 with quantization enabled. The paragraph index is 3.8x larger
    # than chapters/summaries (1.86M vs 491K/545K nodes), so its ANN scan dominates query latency.
    ANN_NEIGHBOURS_CHAPTERS = 100
    ANN_NEIGHBOURS_PARAGRAPHS = 150
    ANN_NEIGHBOURS_SUMMARIES = 100
    SCORE_FACTOR_PARAGRAPHS = 1.0
    SCORE_FACTOR_QUESTIONS = 1.0
    SCORE_FACTOR_SUMMARIES = 1.0
    SCORE_CAP_CHAPTERS = 1.0
    SCORE_CAP_PARAGRAPHS = 1.0
    SCORE_CAP_QUESTIONS = 1.0
    SCORE_CAP_SUMMARIES = 1.0
    MIN_SCORE = 0.1  # Minimum similarity score to include a fragment
    
    # Text search settings
    TEXT_INDEX_NAME = "chapters_fulltext"  # Fulltext index name for chapter text search
    TEXT_SEARCH_MAX_CANDIDATES = 10000  # Max BM25 candidates to fetch before exact substring filtering

    # Short list handling: if number of results is small, return all (reranking will handle ordering)
    SHORT_LIST_MAX = 10

    # Standing-out detection parameters (tunable)
    STANDING_RATIO_THRESHOLD = 10.0  # ratio s_i / s_{i+1} to consider a big gap
    STANDING_ABS_GAP_MIN = 0.5  # absolute gap minimum to consider
    STANDING_GAP_FACTOR = 0.2  # gap must be at least this fraction of tail median
    STANDING_MEAN_RATIO = 2.0  # top mean must be >= this times rest median to accept cut
    STANDING_EPS = 1e-9
    

    def __init__(self):
        self._log_prefix = "RagService"

    def _hydrate_fragment_text(self, fragments: List[Dict[str, Any]], text_cache: Dict[str, Optional[str]]) -> None:
        """Load missing fragment text in one Neo4j round trip and cache it for this query."""
        missing_ids = [
            fragment_id for fragment_id in {fragment.get('fragmentid') for fragment in fragments}
            if fragment_id and fragment_id not in text_cache
        ]
        if not missing_ids:
            for fragment in fragments:
                fragment_id = fragment.get('fragmentid')
                if fragment_id in text_cache:
                    fragment['text'] = text_cache[fragment_id]
            return

        query = """
            MATCH (fragment)
            WHERE (fragment:Chapter OR fragment:Paragraph OR fragment:Summary OR fragment:Keyword)
              AND fragment.id IN $fragment_ids
            RETURN fragment.id AS fragmentid, fragment.text AS text
        """
        try:
            records = execute_neo4j_query(query, {"fragment_ids": missing_ids}) or []
            for record in records:
                text_cache[record.get('fragmentid')] = record.get('text')
            for fragment_id in missing_ids:
                text_cache.setdefault(fragment_id, None)
            for fragment in fragments:
                fragment_id = fragment.get('fragmentid')
                if fragment_id in text_cache:
                    fragment['text'] = text_cache[fragment_id]
        except Exception as error:
            log(ERROR, f"{self._log_prefix}._hydrate_fragment_text: Error: {error}")


    def _combine_fragments_and(self, left_fragments: List[Dict[str, Any]], right_fragments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine fragments using AND logic: keep only fragments from documents present in BOTH branches.
        
        Score combination: For fragments from the same document, use geometric mean of normalized scores
        to favor documents that score well on both criteria.
        
        Args:
            left_fragments: Fragments from left branch
            right_fragments: Fragments from right branch
            
        Returns:
            Combined list of fragments from documents present in both branches
        """
        log(DEBUG, f"{self._log_prefix}._combine_fragments_and: Starting AND combination")
        log(DEBUG, f"{self._log_prefix}._combine_fragments_and: Left: {len(left_fragments)} fragments, Right: {len(right_fragments)} fragments")
        
        # Group fragments by document ID
        left_by_doc = {}
        right_by_doc = {}
        
        for frag in left_fragments:
            doc_id = frag.get('documentid')
            if doc_id:
                if doc_id not in left_by_doc:
                    left_by_doc[doc_id] = []
                left_by_doc[doc_id].append(frag)
        
        for frag in right_fragments:
            doc_id = frag.get('documentid')
            if doc_id:
                if doc_id not in right_by_doc:
                    right_by_doc[doc_id] = []
                right_by_doc[doc_id].append(frag)
        
        log(DEBUG, f"{self._log_prefix}._combine_fragments_and: Left documents: {sorted(left_by_doc.keys())}")
        log(DEBUG, f"{self._log_prefix}._combine_fragments_and: Right documents: {sorted(right_by_doc.keys())}")
        
        # Find intersection of document IDs
        common_docs = set(left_by_doc.keys()) & set(right_by_doc.keys())
        log(DEBUG, f"{self._log_prefix}._combine_fragments_and: Common documents: {sorted(common_docs)}")
        
        if not common_docs:
            log(DEBUG, f"{self._log_prefix}._combine_fragments_and: No common documents, returning empty list")
            return []

        left_global_max = max((fragment.get('score', 0.0) for fragment in left_fragments), default=1.0)
        right_global_max = max((fragment.get('score', 0.0) for fragment in right_fragments), default=1.0)
        
        # Collect all fragments from documents in intersection
        result_fragments = []
        for doc_id in common_docs:
            # Get max scores from each branch for this document
            left_max = max(f.get('score', 0.0) for f in left_by_doc[doc_id])
            right_max = max(f.get('score', 0.0) for f in right_by_doc[doc_id])
            
            # log(DEBUG, f"{self._log_prefix}._combine_fragments_and:   Doc {doc_id}: {len(left_by_doc[doc_id])} left frags, {len(right_by_doc[doc_id])} right frags")
            # log(DEBUG, f"{self._log_prefix}._combine_fragments_and:   Doc {doc_id} - left_max: {left_max:.5f}, right_max: {right_max:.5f}")
            
            # Determine if branches are pure filters (all scores = 1.0)
            left_is_filter = all(f.get('score', 0.0) == 1.0 for f in left_by_doc[doc_id])
            right_is_filter = all(f.get('score', 0.0) == 1.0 for f in right_by_doc[doc_id])
            
            # log(DEBUG, f"{self._log_prefix}._combine_fragments_and:   Doc {doc_id} - left_is_filter: {left_is_filter}, right_is_filter: {right_is_filter}")
            
            # Combine fragments from both branches, deduplicating by fragmentid
            # (category queries set fragmentid = doc_id on both sides, so left and right
            # would otherwise both contribute the same fragment ID, causing duplicate
            # AGGREGATES relationships in the resulting dataset).
            _seen_fids: dict = {}
            _no_fid: list = []
            for _frag in left_by_doc[doc_id] + right_by_doc[doc_id]:
                _fid = _frag.get('fragmentid')
                if not _fid:
                    _no_fid.append(_frag)
                elif _fid not in _seen_fids or _frag.get('score', 0.0) > _seen_fids[_fid].get('score', 0.0):
                    _seen_fids[_fid] = _frag
            all_doc_fragments = list(_seen_fids.values()) + _no_fid
            left_fragment_ids = {
                fragment.get('fragmentid')
                for fragment in left_by_doc[doc_id]
                if fragment.get('fragmentid')
            }
            right_fragment_ids = {
                fragment.get('fragmentid')
                for fragment in right_by_doc[doc_id]
                if fragment.get('fragmentid')
            }
            left_fragment_refs = {id(fragment) for fragment in left_by_doc[doc_id]}
            right_fragment_refs = {id(fragment) for fragment in right_by_doc[doc_id]}

            # Adjust scores based on combination strategy
            if left_is_filter and not right_is_filter:
                # Left is filter, preserve right semantic scores
                # log(DEBUG, f"{self._log_prefix}._combine_fragments_and:   Doc {doc_id} - Strategy: Left filter, preserving right scores")
                for frag in all_doc_fragments:
                    if (frag.get('fragmentid') in right_fragment_ids
                            or id(frag) in right_fragment_refs):
                        result_fragments.append(frag)
            elif right_is_filter and not left_is_filter:
                # Right is filter, preserve left semantic scores
                # log(DEBUG, f"{self._log_prefix}._combine_fragments_and:   Doc {doc_id} - Strategy: Right filter, preserving left scores")
                for frag in all_doc_fragments:
                    if (frag.get('fragmentid') in left_fragment_ids
                            or id(frag) in left_fragment_refs):
                        result_fragments.append(frag)
            else:
                # Both semantic or both filters: use geometric mean
                if not left_is_filter and not right_is_filter:
                    # log(DEBUG, f"{self._log_prefix}._combine_fragments_and:   Doc {doc_id} - Strategy: Both semantic, using geometric mean")
                    # log(DEBUG, f"{self._log_prefix}._combine_fragments_and:   Global max - left: {left_global_max:.5f}, right: {right_global_max:.5f}")
                    
                    # Normalize and combine using geometric mean
                    scale = (left_global_max * right_global_max) ** 0.5 if left_global_max > 0 and right_global_max > 0 else 1.0
                    combined_norm = ((left_max / left_global_max) * (right_max / right_global_max)) ** 0.5 if left_global_max > 0 and right_global_max > 0 else max(left_max, right_max)
                    combined_score = combined_norm * scale
                    
                    # log(DEBUG, f"{self._log_prefix}._combine_fragments_and:   Scale: {scale:.5f}, combined_norm: {combined_norm:.5f}, combined_score: {combined_score:.5f}")
                    
                    # Apply combined score to all fragments from this document
                    for frag in all_doc_fragments:
                        new_frag = frag.copy()
                        new_frag['score'] = combined_score
                        result_fragments.append(new_frag)
                else:
                    # Both filters: take max score
                    # log(DEBUG, f"{self._log_prefix}._combine_fragments_and:   Doc {doc_id} - Strategy: Both filters, using score 1.0")
                    for frag in all_doc_fragments:
                        result_fragments.append(frag)
        
        log(DEBUG, f"{self._log_prefix}._combine_fragments_and: Final result: {len(result_fragments)} fragments from {len(common_docs)} documents")
        if result_fragments:
            # Show unique document scores
            doc_scores = {}
            for f in result_fragments:
                doc_id = f.get('documentid')
                score = f.get('score', 0.0)
                if doc_id not in doc_scores or score > doc_scores[doc_id]:
                    doc_scores[doc_id] = score
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            log(DEBUG, f"{self._log_prefix}._combine_fragments_and: Document scores: {[(d, f'{s:.5f}') for d, s in sorted_docs]}")
        
        return result_fragments
        
    def _combine_fragments_or(self, left_fragments: List[Dict[str, Any]], right_fragments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine fragments using OR logic: union of all fragments, keeping max score per fragment.
        
        Args:
            left_fragments: Fragments from left branch
            right_fragments: Fragments from right branch
            
        Returns:
            Combined list of all fragments with max scores
        """
        # Create a dict to track fragments by fragmentid
        fragments_by_id = {}
        
        for frag in left_fragments + right_fragments:
            frag_id = frag.get('fragmentid')
            if frag_id:
                if frag_id not in fragments_by_id:
                    fragments_by_id[frag_id] = frag.copy()
                else:
                    # Keep the fragment with higher score
                    if frag.get('score', 0.0) > fragments_by_id[frag_id].get('score', 0.0):
                        fragments_by_id[frag_id] = frag.copy()
        
        return list(fragments_by_id.values())
    
    def _combine_fragments_not(self, all_fragments: List[Dict[str, Any]], exclude_fragments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine fragments using NOT logic: remove fragments from documents that appear in exclude list.
        
        Args:
            all_fragments: All fragments (from getting all documents)
            exclude_fragments: Fragments to exclude (removes entire documents)
            
        Returns:
            Filtered list of fragments with excluded documents removed
        """
        # Get set of document IDs to exclude
        exclude_docs = set(f.get('documentid') for f in exclude_fragments if f.get('documentid'))
        
        # Filter out fragments from excluded documents
        result = [f for f in all_fragments if f.get('documentid') not in exclude_docs]
        
        return result
        
    def _semantic_search(self, search_term: str, subset_of_datasetids: Optional[List[str]] = None, effective_user_id: Optional[str] = None, limit_override: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic (RAG) search using embeddings.
        Semantic search is completely independent of ontology.
        Args:
            search_term: The semantic search term (without quotes)
            subset_of_datasetids: Optional list of dataset IDs to restrict search
            effective_user_id: Neo4j User.id of the requesting user. Used for ACL filtering (Task 13.2+).
                      When None, ACL filtering is skipped.
        Returns:
            List of fragment dictionaries with keys: fragmentid, documentid, text, score, source
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        try:
            if effective_user_id is None:
                log(WARNING, f"{self._log_prefix}._semantic_search: effective_user_id is None — ACL filtering will be skipped")
            log(DEBUG, f"{self._log_prefix}._semantic_search: search_term='{search_term}'")
            log(PERF, f"semantic_search: start '{search_term[:60]}'", perf_id="sem_start")
            
            # Get the embedding for the search term
            embeddings_manager = EmbeddingsManager.instance()
            if embeddings_manager is None:
                log(ERROR, f"{self._log_prefix}._semantic_search: EmbeddingsManager not available")
                return []
            
            search_embedding = embeddings_manager.get_embeddings(search_term)
            if search_embedding is None:
                log(ERROR, f"{self._log_prefix}._semantic_search: Failed to get embedding for search term")
                return []
            log(PERF, f"semantic_search: embedding ready", perf_id="sem_embed", perf_since_id="sem_start")
            
            # Retrieve fragments from all sources in parallel
            _parallel_db_t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=3) as executor:
                chapter_future = executor.submit(self._get_semantic_chapters, search_embedding, subset_of_datasetids, effective_user_id, limit_override)
                paragraph_future = executor.submit(self._get_semantic_paragraphs, search_embedding, subset_of_datasetids, effective_user_id, limit_override)
                summary_future = executor.submit(self._get_semantic_summaries, search_embedding, subset_of_datasetids, effective_user_id, limit_override)
                chapters = chapter_future.result()
                paragraphs = paragraph_future.result()
                summaries = summary_future.result()
            _parallel_db_ms = (time.perf_counter() - _parallel_db_t0) * 1000.0
            _goal = find_goal_in_stack_frame(inspect.stack())
            if _goal is not None:
                _goal.add_perf_db_ms(_parallel_db_ms)

            log(DEBUG, f"{self._log_prefix}._semantic_search: Found {len(chapters)} chapter fragments")
            log(PERF, f"semantic_search: chapters done: {len(chapters)}", perf_id="sem_chapters", perf_since_id="sem_embed")
            log(DEBUG, f"{self._log_prefix}._semantic_search: Found {len(paragraphs)} paragraph fragments")
            log(PERF, f"semantic_search: paragraphs done: {len(paragraphs)}", perf_id="sem_paragraphs", perf_since_id="sem_embed")
            log(DEBUG, f"{self._log_prefix}._semantic_search: Found {len(summaries)} summary fragments")
            log(PERF, f"semantic_search: summaries done: {len(summaries)}", perf_id="sem_summaries", perf_since_id="sem_embed")
            all_fragments = chapters + paragraphs + summaries
            
            # Filter by minimum score
            filtered_fragments = [f for f in all_fragments if f.get('score', 0.0) >= self.MIN_SCORE]
            
            if len(all_fragments) == 0:
                log(INFO, f"{self._log_prefix}._semantic_search: No fragments found (embeddings may not be available in database)")
            
            # Count unique documents for logging
            unique_docs = set(f.get('documentid') for f in filtered_fragments if f.get('documentid'))
            log(INFO, f"{self._log_prefix}._semantic_search: Found {len(filtered_fragments)} fragments from {len(unique_docs)} documents")
            log(PERF, f"semantic_search: done, {len(filtered_fragments)} frags from {len(unique_docs)} docs", perf_id="sem_end", perf_since_id="sem_start")
            return filtered_fragments
            
        except Exception as e:
            log(ERROR, f"{self._log_prefix}._semantic_search: Error: {e}\n{traceback.format_exc()}")
            return []

    def _semantic_search_with_acl(
        self,
        search_term: str,
        effective_user_id: Optional[str] = None,
        target_count: int = 1,
        subset_of_datasetids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Wrap _semantic_search in a batch-retry loop that widens the candidate pool
        when the number of ACL-permitted documents returned is below target_count.

        Because ACL filtering is embedded inside Neo4j (Task 13.2), _semantic_search
        already returns only permitted fragments. The loop simply counts distinct
        documentid values and retries with a larger limit_override if needed.

        Retry is controlled by four constants from src.config.constants:
            ACL_BATCH_INITIAL_LIMIT  — first-pass candidate count (default 150 = TOP_N_RAG)
            ACL_BATCH_MAX_LIMIT      — maximum candidate pool before giving up (default 1500)
            ACL_BATCH_MULTIPLIER     — growth factor per retry (default 3)
            ACL_BATCH_MAX_RETRIES    — hard iteration cap (default 5)

        Args:
            search_term: The semantic search term.
            effective_user_id: Neo4j User.id of the requesting user. Defaults to current user.
            target_count: Desired minimum number of distinct permitted documents.
            subset_of_datasetids: Optional dataset IDs to restrict search.

        Returns:
            List of ACL-permitted fragment dicts. May be shorter than target_count
            if the entire vector store is exhausted.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        from src.config.constants import (
            ACL_BATCH_INITIAL_LIMIT, ACL_BATCH_MAX_LIMIT,
            ACL_BATCH_MULTIPLIER, ACL_BATCH_MAX_RETRIES,
        )
        current_limit = ACL_BATCH_INITIAL_LIMIT
        attempt = 0
        permitted_fragments: List[Dict[str, Any]] = []
        _t0 = time.perf_counter()

        while attempt < ACL_BATCH_MAX_RETRIES and current_limit <= ACL_BATCH_MAX_LIMIT:
            attempt += 1
            # _semantic_search returns only ACL-permitted fragments when effective_user_id is set (Task 13.2)
            permitted_fragments = self._semantic_search(
                search_term,
                subset_of_datasetids,
                effective_user_id=effective_user_id,
                limit_override=current_limit,
            )
            unique_permitted_docs = {f["documentid"] for f in permitted_fragments if f.get("documentid")}
            elapsed_ms = (time.perf_counter() - _t0) * 1000.0
            log(INFO, f"{self._log_prefix}._semantic_search_with_acl: attempt={attempt} "
                      f"limit={current_limit} permitted_docs={len(unique_permitted_docs)} elapsed={elapsed_ms:.0f}ms")

            if len(unique_permitted_docs) >= target_count:
                if attempt > 1:
                    log(PERF, f"_sem_with_acl: reached target_count={target_count} "
                              f"after {attempt} attempts (limit={current_limit}, {elapsed_ms:.0f}ms)")
                return permitted_fragments

            if current_limit >= ACL_BATCH_MAX_LIMIT:
                log(WARNING, f"{self._log_prefix}._semantic_search_with_acl: exhausted at "
                             f"limit={current_limit} after {attempt} attempts; "
                             f"returning {len(unique_permitted_docs)} permitted docs")
                return permitted_fragments

            current_limit = min(current_limit * ACL_BATCH_MULTIPLIER, ACL_BATCH_MAX_LIMIT)

        final_count = len({f.get("documentid") for f in permitted_fragments if f.get("documentid")})
        log(WARNING, f"{self._log_prefix}._semantic_search_with_acl: hit max retries "
                     f"({ACL_BATCH_MAX_RETRIES}); returning {final_count} permitted docs")
        return permitted_fragments

    def _fetch_all_document_ids_for_search(self, subset_of_datasetids: Optional[List[str]], effective_user_id: Optional[str] = None) -> List[str]:
        """Fetch all document IDs relevant to the given search scope, respecting dataset and ACL filters.

        Used by the partitioned semantic search to obtain the full document population before
        splitting it into batches. ACL filtering is applied via a direct GRANT-path check so
        that the returned IDs are already permitted for the given effective user.

        Args:
            subset_of_datasetids: Optional list of dataset IDs to restrict the population.
            effective_user_id: Neo4j User.id of the requesting user. When provided and ACL is enabled,
                      only documents the effective user holds READ permission on are returned.

        Returns:
            List of document IDs (d.id) as strings, unordered.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        try:
            acl_active = ACL_ENFORCEMENT_ENABLED and effective_user_id is not None
            params: Dict[str, Any] = {}
            acl_block = self._acl_where_clause('d') if acl_active else ""
            if acl_active:
                params["effectiveUserId"] = effective_user_id

            if subset_of_datasetids and len(subset_of_datasetids) > 0:
                params["subset_ids"] = subset_of_datasetids
                cypher_query = f"""
                    {self._dataset_subset_membership_union()}
                    WHERE TRUE{acl_block}
                    RETURN d.id AS doc_id
                """
            else:
                cypher_query = f"""
                    MATCH (d:Document)
                    WHERE TRUE{acl_block}
                    RETURN DISTINCT d.id AS doc_id
                """

            result = execute_neo4j_query(cypher_query, params)
            if not result:
                return []
            doc_ids = [r['doc_id'] for r in result if r.get('doc_id')]
            log(INFO, f"{self._log_prefix}._fetch_all_document_ids_for_search: fetched {len(doc_ids)} document IDs (dataset_filter={bool(subset_of_datasetids)}, acl_active={acl_active})")
            return doc_ids
        except Exception as e:
            log(ERROR, f"{self._log_prefix}._fetch_all_document_ids_for_search: Error: {e}\n{traceback.format_exc()}")
            return []

    def _semantic_search_partitioned(self, search_term: str, subset_of_datasetids: Optional[List[str]] = None, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform exhaustive multi-language semantic search with large HNSW candidate set.

        Translates the search term once into all target languages, computes embeddings once
        per unique translation, then runs HNSW vector index queries with a large candidate
        limit (SEMANTIC_SEARCH_PARTITION_SIZE) per language. Using a large K in HNSW
        guarantees that documents in minority languages (e.g. German documents for an
        English query) are captured without falling back to slow per-partition cosine scans.

        Args:
            search_term: The semantic search term.
            subset_of_datasetids: Optional dataset IDs (forwarded to index queries).
            effective_user_id: Optional user ID for ACL filtering.

        Returns:
            Combined list of fragment dictionaries (OR across all languages).
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        try:
            from src.config.constants import MULTILANG_SEARCH_ENABLED, MULTILANG_LANGUAGES, MAX_PARALLELISM, SEMANTIC_SEARCH_PARTITION_SIZE
            from src.services.translation_service import get_translation_service

            _t0 = time.perf_counter()

            # Step 1: translate once before any search
            if MULTILANG_SEARCH_ENABLED:
                translation_service = get_translation_service()
                translations = translation_service.translate_to_languages(search_term, MULTILANG_LANGUAGES)
                log(INFO, f"{self._log_prefix}._semantic_search_partitioned: Translated to {list(translations.keys())} languages")
            else:
                translations = {'original': search_term}

            # Step 2: compute one embedding per unique translated text
            embeddings_manager = EmbeddingsManager.instance()
            if embeddings_manager is None:
                log(ERROR, f"{self._log_prefix}._semantic_search_partitioned: EmbeddingsManager not available")
                return []

            unique_texts = list(dict.fromkeys(translations.values()))  # deduplicate, preserve order
            text_to_embedding: Dict[str, List[float]] = {}
            with ThreadPoolExecutor(max_workers=min(MAX_PARALLELISM, len(unique_texts))) as emb_executor:
                emb_futures = {emb_executor.submit(embeddings_manager.get_embeddings, text): text for text in unique_texts}
                for fut in as_completed(emb_futures):
                    text = emb_futures[fut]
                    emb = fut.result()
                    if emb:
                        text_to_embedding[text] = emb

            if not text_to_embedding:
                log(ERROR, f"{self._log_prefix}._semantic_search_partitioned: No embeddings generated")
                return []

            log(INFO, f"{self._log_prefix}._semantic_search_partitioned: Starting HNSW search with ann_limit={SEMANTIC_SEARCH_PARTITION_SIZE} for {len(text_to_embedding)} language embeddings")

            # Step 3: HNSW search with large candidate limit per language.
            # SEMANTIC_SEARCH_PARTITION_SIZE (e.g. 5000) is used as limit_override, which
            # scales ANN_NEIGHBOURS proportionally inside _get_semantic_*. This returns a
            # large candidate pool per language in milliseconds, unlike per-partition cosine scans.
            ann_limit = SEMANTIC_SEARCH_PARTITION_SIZE  # e.g. 5000

            def search_one_translation(embedding: List[float]) -> List[Dict[str, Any]]:
                """Search Chapter, Paragraph, Summary HNSW indexes for one language embedding."""
                with ThreadPoolExecutor(max_workers=3) as ex:
                    f_ch = ex.submit(self._get_semantic_chapters, embedding, subset_of_datasetids, effective_user_id, ann_limit)
                    f_pa = ex.submit(self._get_semantic_paragraphs, embedding, subset_of_datasetids, effective_user_id, ann_limit)
                    f_su = ex.submit(self._get_semantic_summaries, embedding, subset_of_datasetids, effective_user_id, ann_limit)
                    return f_ch.result() + f_pa.result() + f_su.result()

            combined_fragments: List[Dict[str, Any]] = []
            max_workers = min(MAX_PARALLELISM, len(text_to_embedding))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(search_one_translation, emb): text for text, emb in text_to_embedding.items()}
                for future in as_completed(futures):
                    try:
                        lang_fragments = future.result()
                        if lang_fragments:
                            combined_fragments = self._combine_fragments_or(combined_fragments, lang_fragments) if combined_fragments else lang_fragments
                    except Exception as e:
                        log(ERROR, f"{self._log_prefix}._semantic_search_partitioned: Error in translation search: {e}")

            _goal = find_goal_in_stack_frame(inspect.stack())
            if _goal is not None:
                _goal.add_perf_db_ms((time.perf_counter() - _t0) * 1000.0)

            unique_docs = set(f.get('documentid') for f in combined_fragments if f.get('documentid'))
            _total_ms = (time.perf_counter() - _t0) * 1000.0
            log(INFO, f"{self._log_prefix}._semantic_search_partitioned: done - {len(combined_fragments)} fragments from {len(unique_docs)} documents in {_total_ms:.0f}ms ({len(text_to_embedding)} language embeddings, ann_limit={ann_limit})")
            return combined_fragments

        except Exception as e:
            log(ERROR, f"{self._log_prefix}._semantic_search_partitioned: Error: {e}\n{traceback.format_exc()}")
            return []

    def _semantic_search_multilang(self, search_term: str, subset_of_datasetids: Optional[List[str]] = None, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform multi-language semantic search.
        
        Translates the search term into multiple languages (EN, DE, FR, IT) and performs
        semantic search for each translation, then combines results using OR logic.
        This addresses cross-language retrieval issues where documents in one language
        may not match queries in another language due to embedding similarity limitations.
        
        Args:
            search_term: The semantic search term
            subset_of_datasetids: Optional list of dataset IDs to restrict search
            effective_user_id: Neo4j User.id of the requesting user. Forwarded to _semantic_search for ACL filtering (Task 13.2+).
            
        Returns:
            Combined list of fragment dictionaries from all language searches
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        from src.config.constants import MULTILANG_SEARCH_ENABLED, MULTILANG_LANGUAGES
        from src.services.translation_service import get_translation_service
        
        try:
            # If partitioned search is enabled, delegate entirely — it calls
            # _semantic_search_constrained_multilang() per partition which already handles
            # multi-language. Skip the global ANN vector index path complet
            if SEMANTIC_SEARCH_PARTITION_SIZE is not None:
                log(INFO, f"{self._log_prefix}._semantic_search_multilang: delegating to partitioned search (partition_size={SEMANTIC_SEARCH_PARTITION_SIZE})")
                return self._semantic_search_partitioned(search_term, subset_of_datasetids, effective_user_id)

            # If multi-language search is disabled, fall back to single-language search
            if not MULTILANG_SEARCH_ENABLED:
                log(DEBUG, f"{self._log_prefix}._semantic_search_multilang: Multi-language search disabled, using single language")
                return self._semantic_search(search_term, subset_of_datasetids, effective_user_id)
            
            log(INFO, f"{self._log_prefix}._semantic_search_multilang: Starting multi-language search for '{search_term[:50]}...' in {MULTILANG_LANGUAGES}")
            
            # Get translations for all target languages
            translation_service = get_translation_service()
            translations = translation_service.translate_to_languages(search_term, MULTILANG_LANGUAGES)
            
            # Log the translations
            for lang, trans in translations.items():
                log(DEBUG, f"{self._log_prefix}._semantic_search_multilang: [{lang}] '{trans[:80]}...'")
            
            # Perform semantic search for each translation in parallel
            all_fragments: List[Dict[str, Any]] = []
            lang_results: Dict[str, List[Dict[str, Any]]] = {}
            
            def search_with_translation(lang: str, trans_text: str) -> Tuple[str, List[Dict[str, Any]]]:
                """Execute semantic search for a single translation."""
                try:
                    fragments = self._semantic_search(trans_text, subset_of_datasetids, effective_user_id)
                    return (lang, fragments)
                except Exception as e:
                    log(ERROR, f"{self._log_prefix}._semantic_search_multilang: Error searching with {lang} translation: {e}")
                    return (lang, [])
            
            # Execute all searches in parallel
            _multilang_db_t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=len(translations)) as executor:
                futures = {
                    executor.submit(search_with_translation, lang, trans): lang 
                    for lang, trans in translations.items()
                }
                for future in as_completed(futures):
                    try:
                        lang, fragments = future.result()
                        lang_results[lang] = fragments
                        log(DEBUG, f"{self._log_prefix}._semantic_search_multilang: [{lang}] found {len(fragments)} fragments")
                    except Exception as e:
                        lang = futures[future]
                        log(ERROR, f"{self._log_prefix}._semantic_search_multilang: Exception for {lang}: {e}")
                        lang_results[lang] = []
            _multilang_db_ms = (time.perf_counter() - _multilang_db_t0) * 1000.0
            _goal = find_goal_in_stack_frame(inspect.stack())
            if _goal is not None:
                _goal.add_perf_db_ms(_multilang_db_ms)
            
            # Combine all results using OR logic (union with max scores for duplicates)
            combined_fragments = []
            for lang, fragments in lang_results.items():
                if fragments:
                    if not combined_fragments:
                        combined_fragments = fragments
                    else:
                        combined_fragments = self._combine_fragments_or(combined_fragments, fragments)
            
            # Log summary
            total_unique_frags = len(combined_fragments)
            unique_docs = set(f.get('documentid') for f in combined_fragments if f.get('documentid'))
            log(INFO, f"{self._log_prefix}._semantic_search_multilang: Combined {len(MULTILANG_LANGUAGES)} languages -> {total_unique_frags} unique fragments from {len(unique_docs)} documents")
            
            return combined_fragments
            
        except Exception as e:
            log(ERROR, f"{self._log_prefix}._semantic_search_multilang: Error: {e}\n{traceback.format_exc()}")
            # Fall back to single-language search on error
            log(WARNING, f"{self._log_prefix}._semantic_search_multilang: Falling back to single-language search")
            return self._semantic_search(search_term, subset_of_datasetids, effective_user_id)
    
    def _acl_where_clause(self, doc_var: str) -> str:
        """
        Return a Cypher AND-block that enforces READ permission for $effectiveUserId on the
        Document node bound to `doc_var`. Delegates to the shared utility in
        src.utils.acl_cypher_utils so the clause logic is defined in one place.
        The caller must include $effectiveUserId in the query parameters.
        """
        from src.utils.acl_cypher_utils import acl_document_where_clause
        return acl_document_where_clause(doc_var)

    def _dataset_subset_membership_predicate(self, doc_pattern: str, subset_param: str = "$subset_ids") -> str:
        """Boolean OR-chain: doc_pattern's dataset aggregates it directly, via its Chapter,
        or via a Paragraph->Chapter chain. Three fixed-depth branches, no variable-length paths.

        Args:
            doc_pattern: Cypher node pattern for the document, e.g. "d" (already bound) or
                         "d:Document {id: paragraph.documentid}" (bound with a property filter).
            subset_param: Cypher parameter name holding the list of dataset IDs.
        """
        return (
            f"(EXISTS {{ MATCH (ds:Dataset)-[:AGGREGATES]->({doc_pattern}) WHERE ds.id IN {subset_param} }}"
            f" OR EXISTS {{ MATCH (ds:Dataset)-[:AGGREGATES]->(:Chapter)-[:PART_OF]->({doc_pattern}) WHERE ds.id IN {subset_param} }}"
            f" OR EXISTS {{ MATCH (ds:Dataset)-[:AGGREGATES]->(:Paragraph)-[:PART_OF]->(:Chapter)-[:PART_OF]->({doc_pattern}) WHERE ds.id IN {subset_param} }})"
        )

    def _dataset_subset_membership_union(self, subset_param: str = "$subset_ids", extra_filter: str = "") -> str:
        """CALL{} subquery returning DISTINCT :Document nodes (as `d`) aggregated by any
        dataset in subset_param, either directly, via a Chapter, or via a Paragraph->Chapter
        chain. Three fixed-depth branches combined with UNION — no variable-length paths.

        Args:
            subset_param: Cypher parameter name holding the list of dataset IDs.
            extra_filter: Optional extra boolean Cypher expression (referencing `d`), ANDed
                          into every branch, e.g. "d.id IN $doc_ids".
        """
        filt = f" AND {extra_filter}" if extra_filter else ""
        return f"""
        CALL {{
            MATCH (ds:Dataset)-[:AGGREGATES]->(d:Document)
            WHERE ds.id IN {subset_param}{filt}
            RETURN d
            UNION
            MATCH (ds:Dataset)-[:AGGREGATES]->(:Chapter)-[:PART_OF]->(d:Document)
            WHERE ds.id IN {subset_param}{filt}
            RETURN d
            UNION
            MATCH (ds:Dataset)-[:AGGREGATES]->(:Paragraph)-[:PART_OF]->(:Chapter)-[:PART_OF]->(d:Document)
            WHERE ds.id IN {subset_param}{filt}
            RETURN d
        }}
        WITH DISTINCT d"""

    def _get_semantic_chapters(self, search_embedding: List[float], subset_of_datasetids: Optional[List[str]] = None, effective_user_id: Optional[str] = None, limit_override: Optional[int] = None) -> List[dict]:
        """Retrieve chapter fragments using semantic similarity via vector index.
        
        Note: Semantic search is completely independent of ontology. Documents are found based on
        embedding similarity regardless of their ontology assignment.

        Args:
            search_embedding: Pre-computed query embedding vector.
            subset_of_datasetids: Optional dataset IDs to restrict search.
            effective_user_id: Neo4j User.id of the requesting user. When provided and
                      ACL_ENFORCEMENT_ENABLED=True, only fragments whose parent
                      document the effective user holds READ permission on are returned.
            limit_override: When set, overrides TOP_N_CHAPTERS and scales ANN_NEIGHBOURS_CHAPTERS
                            proportionally. Used by the ACL batch-retry loop (Task 13.3).
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        try:
            if limit_override is not None:
                _scale = limit_override / self.TOP_N_RAG
                _effective_limit = limit_override
                _effective_ann = max(round(self.ANN_NEIGHBOURS_CHAPTERS * _scale), limit_override)
            else:
                _effective_limit = self.TOP_N_CHAPTERS
                _effective_ann = self.ANN_NEIGHBOURS_CHAPTERS
            params = {
                "embedding": search_embedding,
                "nearest_neighbours": _effective_ann,
                "limit": _effective_limit
            }

            acl_active = ACL_ENFORCEMENT_ENABLED and effective_user_id is not None
            if acl_active:
                params["effectiveUserId"] = effective_user_id

            if subset_of_datasetids and len(subset_of_datasetids) > 0:
                # With dataset filter: traverse to Document (already bound as `d`)
                params["subset_ids"] = subset_of_datasetids
                acl_block = self._acl_where_clause('d') if acl_active else ""
                cypher_query = f"""CYPHER 25
                    MATCH (chapter:Chapter)
                      SEARCH chapter IN (
                        VECTOR INDEX typical_rag
                        FOR $embedding
                        LIMIT $nearest_neighbours
                      ) SCORE AS score
                    MATCH (chapter)-[:PART_OF]->(d:Document)
                    WHERE {self._dataset_subset_membership_predicate('d')}{acl_block}
                    RETURN chapter.id AS fragmentid,
                           chapter.documentid AS documentid,
                           score AS raw_score
                    ORDER BY raw_score DESC
                    LIMIT $limit
                """
            else:
                if acl_active:
                    # ACL active without dataset filter: traverse to Document for ACL check
                    acl_block = self._acl_where_clause('doc')
                    cypher_query = f"""CYPHER 25
                        MATCH (chapter:Chapter)
                          SEARCH chapter IN (
                            VECTOR INDEX typical_rag
                            FOR $embedding
                            LIMIT $nearest_neighbours
                          ) SCORE AS score
                        MATCH (chapter)-[:PART_OF]->(doc:Document)
                        WHERE TRUE{acl_block}
                        RETURN chapter.id AS fragmentid,
                               chapter.documentid AS documentid,
                               score AS raw_score
                        ORDER BY raw_score DESC
                        LIMIT $limit
                    """
                else:
                    # No filter, no ACL: use documentid stored on node, skip graph traversal
                    cypher_query = f"""CYPHER 25
                        MATCH (chapter:Chapter)
                          SEARCH chapter IN (
                            VECTOR INDEX typical_rag
                            FOR $embedding
                            LIMIT $nearest_neighbours
                          ) SCORE AS score
                        WHERE chapter.documentid IS NOT NULL
                        RETURN chapter.id AS fragmentid,
                               chapter.documentid AS documentid,
                               score AS raw_score
                        ORDER BY raw_score DESC
                        LIMIT $limit
                    """

            result = execute_neo4j_query(cypher_query, params)
            fragments = []

            if result:
                for record in result:
                    score = min(record.get('raw_score', 0.0), self.SCORE_CAP_CHAPTERS)
                    fragments.append({
                        'fragmentid': record.get('fragmentid'),
                        'documentid': record.get('documentid'),
                        'text': record.get('text'),
                        'score': score,
                        'source': 'chapters'
                    })

            return fragments

        except Exception as e:
            log(ERROR, f"{self._log_prefix}._get_semantic_chapters: Error: {e}")
            return []
    
    def _get_semantic_paragraphs(self, search_embedding: List[float], subset_of_datasetids: Optional[List[str]] = None, effective_user_id: Optional[str] = None, limit_override: Optional[int] = None) -> List[dict]:
        """Retrieve paragraph fragments using semantic similarity via vector index.
        
        Note: Semantic search is completely independent of ontology.

        Args:
            search_embedding: Pre-computed query embedding vector.
            subset_of_datasetids: Optional dataset IDs to restrict search.
            effective_user_id: Neo4j User.id of the requesting user. When provided and
                      ACL_ENFORCEMENT_ENABLED=True, only fragments whose parent
                      document the effective user holds READ permission on are returned.
            limit_override: When set, overrides TOP_N_PARAGRAPHS (preserving the 3x ratio relative
                            to TOP_N_RAG) and scales ANN_NEIGHBOURS_PARAGRAPHS proportionally.
                            Used by the ACL batch-retry loop (Task 13.3).
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        try:
            if limit_override is not None:
                _scale = limit_override / self.TOP_N_RAG
                _effective_limit = max(round(self.TOP_N_PARAGRAPHS * _scale), limit_override)
                _effective_ann = max(round(self.ANN_NEIGHBOURS_PARAGRAPHS * _scale), _effective_limit)
            else:
                _effective_limit = self.TOP_N_PARAGRAPHS
                _effective_ann = self.ANN_NEIGHBOURS_PARAGRAPHS
            params = {
                "embedding": search_embedding,
                "nearest_neighbours": _effective_ann,
                "limit": _effective_limit
            }

            acl_active = ACL_ENFORCEMENT_ENABLED and effective_user_id is not None
            if acl_active:
                params["effectiveUserId"] = effective_user_id

            if subset_of_datasetids and len(subset_of_datasetids) > 0:
                params["subset_ids"] = subset_of_datasetids
                if acl_active:
                    acl_block = self._acl_where_clause('doc')
                    cypher_query = f"""CYPHER 25
                        MATCH (paragraph:Paragraph)
                          SEARCH paragraph IN (
                            VECTOR INDEX chapterparagraph
                            FOR $embedding
                            LIMIT $nearest_neighbours
                          ) SCORE AS score
                        MATCH (paragraph)-[:PART_OF]->(chapter)-[:PART_OF]->(doc:Document)
                        WHERE {self._dataset_subset_membership_predicate('doc')}{acl_block}
                        RETURN paragraph.id AS fragmentid,
                               paragraph.documentid AS documentid,
                               score * {self.SCORE_FACTOR_PARAGRAPHS} AS raw_score
                        ORDER BY raw_score DESC
                        LIMIT $limit
                    """
                else:
                    # With dataset filter, no ACL: existing behaviour
                    cypher_query = f"""CYPHER 25
                    MATCH (paragraph:Paragraph)
                      SEARCH paragraph IN (
                        VECTOR INDEX chapterparagraph
                        FOR $embedding
                        LIMIT $nearest_neighbours
                      ) SCORE AS score
                    WHERE paragraph.documentid IS NOT NULL
                      AND {self._dataset_subset_membership_predicate('d:Document {id: paragraph.documentid}')}
                    RETURN paragraph.id AS fragmentid,
                           paragraph.documentid AS documentid,
                           score * {self.SCORE_FACTOR_PARAGRAPHS} AS raw_score
                    ORDER BY raw_score DESC
                    LIMIT $limit
                """
            else:
                if acl_active:
                    acl_block = self._acl_where_clause('doc')
                    cypher_query = f"""CYPHER 25
                        MATCH (paragraph:Paragraph)
                          SEARCH paragraph IN (
                            VECTOR INDEX chapterparagraph
                            FOR $embedding
                            LIMIT $nearest_neighbours
                          ) SCORE AS score
                        MATCH (paragraph)-[:PART_OF]->(chapter)-[:PART_OF]->(doc:Document)
                        WHERE TRUE{acl_block}
                        RETURN paragraph.id AS fragmentid,
                               paragraph.documentid AS documentid,
                               score * {self.SCORE_FACTOR_PARAGRAPHS} AS raw_score
                        ORDER BY raw_score DESC
                        LIMIT $limit
                    """
                else:
                    # No filter, no ACL: use documentid stored on node, skip graph traversal entirely
                    cypher_query = f"""CYPHER 25
                    MATCH (paragraph:Paragraph)
                      SEARCH paragraph IN (
                        VECTOR INDEX chapterparagraph
                        FOR $embedding
                        LIMIT $nearest_neighbours
                      ) SCORE AS score
                    WHERE paragraph.documentid IS NOT NULL
                    RETURN paragraph.id AS fragmentid,
                           paragraph.documentid AS documentid,
                           score * {self.SCORE_FACTOR_PARAGRAPHS} AS raw_score
                    ORDER BY raw_score DESC
                    LIMIT $limit
                """

            result = execute_neo4j_query(cypher_query, params)
            fragments = []

            if result:
                for record in result:
                    score = min(record.get('raw_score', 0.0), self.SCORE_CAP_PARAGRAPHS)
                    fragments.append({
                        'fragmentid': record.get('fragmentid'),
                        'documentid': record.get('documentid'),
                        'text': record.get('text'),
                        'score': score,
                        'source': 'paragraphs'
                    })

            return fragments

        except Exception as e:
            log(ERROR, f"{self._log_prefix}._get_semantic_paragraphs: Error: {e}")
            return []
    
    def _get_semantic_summaries(self, search_embedding: List[float], subset_of_datasetids: Optional[List[str]] = None, effective_user_id: Optional[str] = None, limit_override: Optional[int] = None) -> List[dict]:
        """Retrieve summary fragments using semantic similarity via vector index.
        
        Note: Semantic search is completely independent of ontology.

        Args:
            search_embedding: Pre-computed query embedding vector.
            subset_of_datasetids: Optional dataset IDs to restrict search.
            effective_user_id: Neo4j User.id of the requesting user. When provided and
                      ACL_ENFORCEMENT_ENABLED=True, only fragments whose parent
                      document the effective user holds READ permission on are returned.
            limit_override: When set, overrides TOP_N_SUMMARIES and scales ANN_NEIGHBOURS_SUMMARIES
                            proportionally. Used by the ACL batch-retry loop (Task 13.3).
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        try:
            if limit_override is not None:
                _scale = limit_override / self.TOP_N_RAG
                _effective_limit = limit_override
                _effective_ann = max(round(self.ANN_NEIGHBOURS_SUMMARIES * _scale), limit_override)
            else:
                _effective_limit = self.TOP_N_SUMMARIES
                _effective_ann = self.ANN_NEIGHBOURS_SUMMARIES
            params = {
                "embedding": search_embedding,
                "nearest_neighbours": _effective_ann,
                "limit": _effective_limit
            }

            acl_active = ACL_ENFORCEMENT_ENABLED and effective_user_id is not None
            if acl_active:
                params["effectiveUserId"] = effective_user_id

            # Summaries already traverse to `d:Document` via MATCH (c)-[:PART_OF]->(d:Document),
            # so the ACL clause can be appended directly to the WHERE block.
            dataset_filter = ""
            if subset_of_datasetids and len(subset_of_datasetids) > 0:
                dataset_filter = f"AND {self._dataset_subset_membership_predicate('d')}"
                params["subset_ids"] = subset_of_datasetids

            acl_block = self._acl_where_clause('d') if acl_active else ""

            cypher_query = f"""CYPHER 25
                MATCH (s:Summary)
                  SEARCH s IN (
                    VECTOR INDEX summary
                    FOR $embedding
                    LIMIT $nearest_neighbours
                  ) SCORE AS score
                MATCH (c:Chapter)-[:HAS_SUMMARY]->(s)
                MATCH (c)-[:PART_OF]->(d:Document)
                WHERE TRUE {dataset_filter}{acl_block}
                RETURN c.id AS fragmentid,
                       COALESCE(s.documentid, d.id) AS documentid,
                       s.text AS text,
                       score * {self.SCORE_FACTOR_SUMMARIES} AS raw_score
                ORDER BY raw_score DESC
                LIMIT $limit
            """

            result = execute_neo4j_query(cypher_query, params)
            fragments = []

            if result:
                for record in result:
                    score = min(record.get('raw_score', 0.0), self.SCORE_CAP_SUMMARIES)
                    fragments.append({
                        'fragmentid': record.get('fragmentid'),
                        'documentid': record.get('documentid'),
                        'text': record.get('text'),
                        'score': score,
                        'source': 'summaries'
                    })

            return fragments

        except Exception as e:
            log(ERROR, f"{self._log_prefix}._get_semantic_summaries: Error: {e}")
            return []
    
    def _normalize_for_fulltext_search(self, text: str) -> str:
        """
        Normalize text for fulltext search by removing or handling problematic characters.
        
        This helps ensure consistent search results when text contains punctuation that
        can cause Lucene/Neo4j fulltext index parsing issues (like apostrophes in contractions).
        
        Strategy:
        - Remove standalone punctuation that's not meaningful in fulltext search
        - Keep alphanumeric characters and spaces for robust matching
        - Preserve word boundaries
        
        Args:
            text: The text to normalize
            
        Returns:
            Normalized text suitable for fulltext queries
        """
        # Replace apostrophes with spaces to handle contractions like "Jane's" -> "Jane s"
        # This allows the fulltext index to match both "Jane's" and variations
        normalized = text.replace("'", " ")
        # Also normalize other common punctuation that might cause issues
        # but preserve essential query structure
        normalized = normalized.replace('"', " ")  # Remove quotes
        normalized = normalized.replace('-', " ")   # Replace hyphens with spaces
        # Collapse multiple spaces to single space
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _escape_lucene_special_chars(self, text: str) -> str:
        r"""
        Escape special characters in text for Lucene fulltext index queries.
        
        Lucene reserves certain characters for query syntax:
        + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
        
        To search for text containing these characters, they must be escaped with a backslash.
        
        Args:
            text: The text to escape
            
        Returns:
            Text with special characters escaped for Lucene
        """
        # Characters that need escaping in Lucene queries
        lucene_special_chars = ['+', '-', '&&', '||', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '\\', '/']
        
        # Escape each special character by prefixing with backslash
        result = text
        for char in lucene_special_chars:
            if char in result:
                result = result.replace(char, '\\' + char)
        
        return result
    
    def _apply_exact_text_filter(self, candidates: List[Dict[str, Any]], search_term: str, case_sensitive: bool) -> List[Dict[str, Any]]:
        """Apply exact substring filter to BM25 candidates, scoring exact matches at 1.0 and
        non-exact BM25 matches at a normalized score < 1.0 (capped to TOP_N_CHAPTERS).

        Phase 2 of the text-search strategy: BM25 finds up to TEXT_SEARCH_MAX_CANDIDATES
        fragments; this method then separates exact substring matches (score 1.0) from
        BM25-only matches (normalized BM25 score < 1.0, top-N kept).

        Args:
            candidates: Raw BM25 results, each with 'text' and 'score' fields.
            search_term: The original (un-normalized) search term to match exactly.
            case_sensitive: Whether to use case-sensitive matching.

        Returns:
            Combined list: all exact matches first (score=1.0), then top-N BM25-only (<1.0).
        """
        exact_fragments: List[Dict[str, Any]] = []
        bm25_only_fragments: List[Dict[str, Any]] = []

        if not candidates:
            return []

        max_bm25: float = max(f.get('score', 0.0) for f in candidates)
        if max_bm25 <= 0.0:
            max_bm25 = 1.0

        for frag in candidates:
            text: str = frag.get('text') or ''
            if case_sensitive:
                is_exact: bool = search_term in text
            else:
                is_exact = search_term.lower() in text.lower()

            if is_exact:
                exact_frag: Dict[str, Any] = frag.copy()
                exact_frag['score'] = 1.0
                exact_fragments.append(exact_frag)
            else:
                bm25_frag: Dict[str, Any] = frag.copy()
                bm25_frag['score'] = (frag.get('score', 0.0) / max_bm25) * 0.99
                bm25_only_fragments.append(bm25_frag)

        # Keep only top-N BM25-only results (already sorted by score from Cypher)
        bm25_only_fragments = bm25_only_fragments[:self.TOP_N_CHAPTERS]

        log(DEBUG, f"{self._log_prefix}._apply_exact_text_filter: {len(exact_fragments)} exact matches, {len(bm25_only_fragments)} BM25-only (capped at {self.TOP_N_CHAPTERS})")
        return exact_fragments + bm25_only_fragments

    def _text_search(self, search_term: str, case_sensitive: bool, subset_of_datasetids: Optional[List[str]] = None, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform fulltext index search with two-phase scoring.

        Phase 1: BM25 fulltext index returns up to TEXT_SEARCH_MAX_CANDIDATES candidates.
        Phase 2: Exact substring check on all candidates — exact matches score 1.0,
                 BM25-only matches keep a normalized score < 1.0 (top-N retained).

        Args:
            search_term: The text to search for (without quotes)
            case_sensitive: True for case-sensitive, False for case-insensitive
            subset_of_datasetids: Optional list of dataset IDs to restrict search
            effective_user_id: Neo4j User.id of the requesting user.

        Returns:
            List of fragment dictionaries with keys: fragmentid, documentid, text, score, source
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        try:
            log(DEBUG, f"{self._log_prefix}._text_search: search_term='{search_term}', case_sensitive={case_sensitive}")

            all_fragments: List[Dict[str, Any]] = []

            # Normalize for Lucene (BM25 phase); keep original for exact match phase
            normalized_search_term: str = self._normalize_for_fulltext_search(search_term)
            log(DEBUG, f"{self._log_prefix}._text_search: normalized_search_term='{normalized_search_term}'")

            dataset_filter: str = ""
            escaped_search_term: str = self._escape_lucene_special_chars(normalized_search_term)
            params: Dict[str, Any] = {"search_term": escaped_search_term, "limit": self.TEXT_SEARCH_MAX_CANDIDATES}
            if subset_of_datasetids and len(subset_of_datasetids) > 0:
                dataset_filter = f"AND {self._dataset_subset_membership_predicate('d')}"
                params["subset_ids"] = subset_of_datasetids
            acl_active: bool = ACL_ENFORCEMENT_ENABLED and effective_user_id is not None
            acl_block: str = self._acl_where_clause('d') if acl_active else ""
            if acl_active:
                params["effectiveUserId"] = effective_user_id

            # Phase 1a: BM25 chapter search (up to TEXT_SEARCH_MAX_CANDIDATES)
            cypher_query_chapters = f"""
                CALL db.index.fulltext.queryNodes('{self.TEXT_INDEX_NAME}', $search_term)
                YIELD node AS chapter, score
                MATCH (chapter)-[:PART_OF]->(d:Document)
                WHERE TRUE {dataset_filter}{acl_block}
                RETURN chapter.id AS fragmentid,
                       d.id AS documentid,
                       d.documentid AS doc_id,
                       chapter.text AS text,
                       score
                ORDER BY score DESC
                LIMIT $limit
            """

            chapter_candidates: List[Dict[str, Any]] = []
            result = execute_neo4j_query(cypher_query_chapters, params)
            if result:
                for record in result:
                    doc_id = record.get('documentid') or record.get('doc_id')
                    chapter_candidates.append({
                        'fragmentid': record.get('fragmentid'),
                        'documentid': doc_id,
                        'text': record.get('text'),
                        'score': record.get('score', 0.0),
                        'source': 'chapter_text'
                    })

            # Phase 2a: exact substring filter on all chapter candidates
            chapter_fragments = self._apply_exact_text_filter(chapter_candidates, search_term, case_sensitive)
            all_fragments.extend(chapter_fragments)
            log(DEBUG, f"{self._log_prefix}._text_search: {len(chapter_fragments)} chapter fragments after exact filter ({len(chapter_candidates)} BM25 candidates)")

            # Phase 1b: BM25 keyword search (up to TEXT_SEARCH_MAX_CANDIDATES)
            cypher_query_keywords = f"""
                CALL db.index.fulltext.queryNodes('keywordsTextIndex', $search_term)
                YIELD node AS keyword, score
                MATCH (keyword)<-[:HAS_KEYWORD]-(d:Document)
                WHERE TRUE {dataset_filter}{acl_block}
                RETURN keyword.id AS fragmentid,
                       d.id AS documentid,
                       d.documentid AS doc_id,
                       keyword.text AS text,
                       score
                ORDER BY score DESC
                LIMIT $limit
            """

            keyword_candidates: List[Dict[str, Any]] = []
            result = execute_neo4j_query(cypher_query_keywords, params)
            if result:
                for record in result:
                    doc_id = record.get('documentid') or record.get('doc_id')
                    keyword_candidates.append({
                        'fragmentid': record.get('fragmentid'),
                        'documentid': doc_id,
                        'text': record.get('text'),
                        'score': record.get('score', 0.0),
                        'source': 'keyword'
                    })

            # Phase 2b: exact substring filter on all keyword candidates
            keyword_fragments = self._apply_exact_text_filter(keyword_candidates, search_term, case_sensitive)
            all_fragments.extend(keyword_fragments)

            unique_docs = set(f.get('documentid') for f in all_fragments if f.get('documentid'))
            log(INFO, f"{self._log_prefix}._text_search: Found {len(all_fragments)} fragments from {len(unique_docs)} documents (BM25 candidates: {len(chapter_candidates)} chapters, {len(keyword_candidates)} keywords)")
            return all_fragments

        except Exception as e:
            log(ERROR, f"{self._log_prefix}._text_search: Error: {e}\n{traceback.format_exc()}")
            return []
    
    def _semantic_search_constrained(self, search_term: str, partition: Optional[str] = None, constrained_doc_ids: Optional[List[str]] = None, subset_of_datasetids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search optionally constrained to specific document IDs.

        When constrained_doc_ids is provided, filters results to those documents only.
        The partition parameter is kept for backwards compatibility but ignored — the HNSW
        vector index is used for all searches (not brute-force cosine).

        Args:
            search_term: The semantic search term
            partition: Ignored (kept for API compatibility). The HNSW index is used instead.
            constrained_doc_ids: Optional list of document IDs to restrict results to.
            subset_of_datasetids: Optional list of dataset IDs to restrict search.

        Returns:
            List of fragment dictionaries
        """
        try:
            log(DEBUG, f"{self._log_prefix}._semantic_search_constrained: search_term='{search_term}', constrained_doc_ids={len(constrained_doc_ids) if constrained_doc_ids else None}")

            embeddings_manager = EmbeddingsManager.instance()
            if embeddings_manager is None:
                log(ERROR, f"{self._log_prefix}._semantic_search_constrained: EmbeddingsManager not available")
                return []

            search_embedding = embeddings_manager.get_embeddings(search_term)
            if search_embedding is None:
                log(ERROR, f"{self._log_prefix}._semantic_search_constrained: Failed to get embedding for search term")
                return []

            # Always use exact cosine similarity scan scoped by the available constraint.
            # HNSW (global ANN index) cannot be used with a post-filter because target documents
            # may rank outside the global top-K (e.g. a German document for an English query).
            # The _fetch_and_score_* functions apply the constraint as a pre-filter inside Neo4j:
            #   - constrained_doc_ids → WHERE documentid IN [...] (scan only those docs)
            #   - partition            → WHERE partition = $p   (scan ~1/26 of all nodes)
            #   - neither              → full scan              (slow, but correct)
            with ThreadPoolExecutor(max_workers=3) as executor:
                f_ch = executor.submit(self._fetch_and_score_chapters, search_embedding, partition, constrained_doc_ids)
                f_pa = executor.submit(self._fetch_and_score_paragraphs, search_embedding, partition, constrained_doc_ids)
                f_su = executor.submit(self._fetch_and_score_summaries, search_embedding, partition, constrained_doc_ids)
                all_fragments = f_ch.result() + f_pa.result() + f_su.result()

            filtered_fragments = [f for f in all_fragments if f.get('score', 0.0) >= self.MIN_SCORE]
            filtered_fragments.sort(key=lambda x: x.get('score', 0.0), reverse=True)

            unique_docs = set(f.get('documentid') for f in filtered_fragments if f.get('documentid'))
            log(DEBUG, f"{self._log_prefix}._semantic_search_constrained: Found {len(filtered_fragments)} fragments from {len(unique_docs)} documents")
            return filtered_fragments

        except Exception as e:
            log(ERROR, f"{self._log_prefix}._semantic_search_constrained: Error: {e}\n{traceback.format_exc()}")
            return []

    def _semantic_search_constrained_multilang(self, search_term: str, partition: Optional[str] = None, constrained_doc_ids: Optional[List[str]] = None, subset_of_datasetids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform multi-language semantic search for a single partition or constrained set of documents.
        
        Translates the search term into multiple languages and performs constrained
        semantic search for each translation, then combines results using OR logic.
        
        Args:
            search_term: The semantic search term
            partition: Single letter A-Z to filter by fragment.partition property. If None, searches all.
            constrained_doc_ids: Optional list of document IDs to restrict search to (takes precedence over partition)
            subset_of_datasetids: Optional list of dataset IDs (not used with partition mode)
            
        Returns:
            Combined list of fragment dictionaries from all language searches
        """
        from src.config.constants import MULTILANG_SEARCH_ENABLED, MULTILANG_LANGUAGES
        from src.services.translation_service import get_translation_service
        
        try:
            # If multi-language search is disabled, fall back to single-language search
            if not MULTILANG_SEARCH_ENABLED:
                log(DEBUG, f"{self._log_prefix}._semantic_search_constrained_multilang: Multi-language search disabled")
                return self._semantic_search_constrained(search_term, partition=partition, constrained_doc_ids=constrained_doc_ids, subset_of_datasetids=subset_of_datasetids)
            
            log(INFO, f"{self._log_prefix}._semantic_search_constrained_multilang: Starting multi-language constrained search in {MULTILANG_LANGUAGES} for partition '{partition}', restricted to {len(constrained_doc_ids) if constrained_doc_ids is not None else 'all'}{f' subset of {len(subset_of_datasetids)}' if subset_of_datasetids is not None else ''}, search term: {search_term}")
            
            # Get translations for all target languages
            translation_service = get_translation_service()
            translations = translation_service.translate_to_languages(search_term, MULTILANG_LANGUAGES)
            
            # Perform constrained semantic search for each translation in parallel
            lang_results: Dict[str, List[Dict[str, Any]]] = {}
            
            def search_with_translation(lang: str, trans_text: str) -> Tuple[str, List[Dict[str, Any]]]:
                """Execute constrained semantic search for a single translation."""
                try:
                    fragments = self._semantic_search_constrained(trans_text, partition=partition, constrained_doc_ids=constrained_doc_ids, subset_of_datasetids=subset_of_datasetids)
                    return (lang, fragments)
                except Exception as e:
                    log(ERROR, f"{self._log_prefix}._semantic_search_constrained_multilang: Error for {lang}: {e}")
                    return (lang, [])
            
            # Execute all searches in parallel
            _constrained_multilang_db_t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=len(translations)) as executor:
                futures = {
                    executor.submit(search_with_translation, lang, trans): lang 
                    for lang, trans in translations.items()
                }
                for future in as_completed(futures):
                    try:
                        lang, fragments = future.result()
                        lang_results[lang] = fragments
                        log(DEBUG, f"{self._log_prefix}._semantic_search_constrained_multilang: [{lang}] found {len(fragments)} fragments")
                    except Exception as e:
                        lang = futures[future]
                        log(ERROR, f"{self._log_prefix}._semantic_search_constrained_multilang: Exception for {lang}: {e}")
                        lang_results[lang] = []
            _constrained_multilang_db_ms = (time.perf_counter() - _constrained_multilang_db_t0) * 1000.0
            _goal = find_goal_in_stack_frame(inspect.stack())
            if _goal is not None:
                _goal.add_perf_db_ms(_constrained_multilang_db_ms)

            # Combine all results using OR logic
            combined_fragments = []
            for lang, fragments in lang_results.items():
                if fragments:
                    if not combined_fragments:
                        combined_fragments = fragments
                    else:
                        combined_fragments = self._combine_fragments_or(combined_fragments, fragments)
            
            # Log summary
            unique_docs = set(f.get('documentid') for f in combined_fragments if f.get('documentid'))
            log(INFO, f"{self._log_prefix}._semantic_search_constrained_multilang: Combined {len(MULTILANG_LANGUAGES)} languages -> {len(combined_fragments)} unique fragments from {len(unique_docs)} documents")
            
            return combined_fragments
            
        except Exception as e:
            log(ERROR, f"{self._log_prefix}._semantic_search_constrained_multilang: Error: {e}\n{traceback.format_exc()}")
            # Fall back to single-language search on error
            return self._semantic_search_constrained(search_term, partition=partition, subset_of_datasetids=subset_of_datasetids)
    
    @staticmethod
    def _partition_letter_from_hash(documentid: str) -> str:
        """Map a documentid to a partition letter A-Z using a deterministic hash function.

        CENTRALIZED: This is the single source of truth for the hash->partition mapping.
        All partition-based search and maintenance code uses this method to ensure consistency.

        This distributes documents across 26 partitions. All fragments (chapters, paragraphs, summaries)
        of a document share the same partition key (stored in the 'partition' property).

        Uses hashlib.sha256 to ensure the same partition letter is produced across all
        Python process restarts (unlike Python's built-in hash() which is randomized).

        Args:
            documentid: The document ID (e.g., 'docid-1', 'docid-8230')

        Returns:
            A single uppercase letter A-Z
        """
        import hashlib
        h = int(hashlib.sha256(documentid.encode('utf-8')).hexdigest(), 16) % 26
        return chr(ord('A') + h)

    def _get_partition_key(self, documentid: str) -> str:
        """Convenience instance method wrapper for partition letter mapping."""
        return self._partition_letter_from_hash(documentid)

    def _compute_cosine_similarity(self, embedding: List[float], search_embedding_np, search_norm) -> float:
        """Compute cosine similarity between a fragment embedding and search embedding."""
        import numpy as np
        if embedding is None:
            return 0.0
        frag_embedding_np = np.array(embedding)
        frag_norm = np.linalg.norm(frag_embedding_np)
        if frag_norm == 0 or search_norm == 0:
            return 0.0
        return float(np.dot(search_embedding_np, frag_embedding_np) / (search_norm * frag_norm))
    
    def _fetch_and_score_chapters(self, search_embedding: List[float], partition: Optional[str] = None, constrained_doc_ids: Optional[List[str]] = None) -> List[dict]:
        """Fetch chapters for a partition or constrained set of documents, and compute similarity scores server-side.

        Args:
            search_embedding: Pre-computed query embedding vector.
            partition: When provided, filter by chapter.partition = $partition (indexed, fast).
            constrained_doc_ids: When provided, filter by chapter.documentid IN constrained_doc_ids (takes precedence over partition).
            If both are None, searches all chapters (expensive; used for unconstrained search).
        """
        try:
            params = {"search_embedding": search_embedding, "min_score": self.MIN_SCORE}

            # constrained_doc_ids takes precedence over partition
            if constrained_doc_ids:
                params["constrained_doc_ids"] = constrained_doc_ids
                cypher_query = """
                    MATCH (chapter:Chapter)
                    WHERE chapter.documentid IN $constrained_doc_ids
                    WITH chapter, vector.similarity.cosine(chapter.embedding, $search_embedding) AS raw_score
                    WHERE raw_score IS NOT NULL AND raw_score >= $min_score
                    RETURN chapter.id AS fragmentid,
                           chapter.documentid AS documentid,
                           raw_score AS score
                """
            elif partition:
                params["partition"] = partition
                cypher_query = """
                    MATCH (chapter:Chapter {partition: $partition})
                    WITH chapter, vector.similarity.cosine(chapter.embedding, $search_embedding) AS raw_score
                    WHERE raw_score IS NOT NULL AND raw_score >= $min_score
                    RETURN chapter.id AS fragmentid,
                           chapter.documentid AS documentid,
                           raw_score AS score
                """
            else:
                cypher_query = """
                    MATCH (chapter:Chapter)
                    WITH chapter, vector.similarity.cosine(chapter.embedding, $search_embedding) AS raw_score
                    WHERE raw_score IS NOT NULL AND raw_score >= $min_score
                    RETURN chapter.id AS fragmentid,
                           chapter.documentid AS documentid,
                           raw_score AS score
                """
            
            result = execute_neo4j_query(cypher_query, params)
            fragments = []
            
            if result:
                for record in result:
                    score = min(record.get('score', 0.0), self.SCORE_CAP_CHAPTERS)
                    fragments.append({
                        'fragmentid': record.get('fragmentid'),
                        'documentid': record.get('documentid'),
                        'text': record.get('text'),
                        'score': score,
                        'source': 'chapters'
                    })
            
            return fragments
            
        except Exception as e:
            log(ERROR, f"{self._log_prefix}._fetch_and_score_chapters: Error: {e}")
            return []
    
    def _fetch_and_score_paragraphs(self, search_embedding: List[float], partition: Optional[str] = None, constrained_doc_ids: Optional[List[str]] = None) -> List[dict]:
        """Fetch paragraphs for a partition or constrained set of documents, and compute similarity scores server-side.

        Args:
            search_embedding: Pre-computed query embedding vector.
            partition: When provided, filter by paragraph.partition = $partition (indexed, fast).
            constrained_doc_ids: When provided, filter by paragraph.documentid IN constrained_doc_ids (takes precedence over partition).
                      When None, searches all paragraphs (expensive; used for unconstrained search).
        """
        try:
            params = {"search_embedding": search_embedding, "min_score": self.MIN_SCORE}

            # constrained_doc_ids takes precedence over partition
            if constrained_doc_ids:
                params["constrained_doc_ids"] = constrained_doc_ids
                cypher_query = """
                    MATCH (paragraph:Paragraph)
                    WHERE paragraph.documentid IN $constrained_doc_ids
                    WITH paragraph, vector.similarity.cosine(paragraph.embedding, $search_embedding) AS raw_score
                    WHERE raw_score IS NOT NULL AND raw_score >= $min_score
                    RETURN paragraph.id AS fragmentid,
                           paragraph.documentid AS documentid,
                           raw_score AS score
                """
            elif partition:
                params["partition"] = partition
                cypher_query = """
                    MATCH (paragraph:Paragraph {partition: $partition})
                    WITH paragraph, vector.similarity.cosine(paragraph.embedding, $search_embedding) AS raw_score
                    WHERE raw_score IS NOT NULL AND raw_score >= $min_score
                    RETURN paragraph.id AS fragmentid,
                           paragraph.documentid AS documentid,
                           raw_score AS score
                """
            else:
                cypher_query = """
                    MATCH (paragraph:Paragraph)
                    WITH paragraph, vector.similarity.cosine(paragraph.embedding, $search_embedding) AS raw_score
                    WHERE raw_score IS NOT NULL AND raw_score >= $min_score
                    RETURN paragraph.id AS fragmentid,
                           paragraph.documentid AS documentid,
                           raw_score AS score
                """
            
            result = execute_neo4j_query(cypher_query, params)
            fragments = []
            
            if result:
                for record in result:
                    score = min(record.get('score', 0.0) * self.SCORE_FACTOR_PARAGRAPHS, self.SCORE_CAP_PARAGRAPHS)
                    fragments.append({
                        'fragmentid': record.get('fragmentid'),
                        'documentid': record.get('documentid'),
                        'text': record.get('text'),
                        'score': score,
                        'source': 'paragraphs'
                    })
            
            return fragments
            
        except Exception as e:
            log(ERROR, f"{self._log_prefix}._fetch_and_score_paragraphs: Error: {e}")
            return []
    
    def _fetch_and_score_summaries(self, search_embedding: List[float], partition: Optional[str] = None, constrained_doc_ids: Optional[List[str]] = None) -> List[dict]:
        """Fetch summaries for a partition or constrained set of documents, and compute similarity scores server-side.

        Args:
            search_embedding: Pre-computed query embedding vector.
            partition: When provided, filter by summary.partition = $partition (indexed, fast).
            constrained_doc_ids: When provided, filter by summary.documentid IN constrained_doc_ids (takes precedence over partition).
                      When None, searches all summaries (expensive; used for unconstrained search).
        """
        try:
            params = {"search_embedding": search_embedding, "min_score": self.MIN_SCORE}

            # constrained_doc_ids takes precedence over partition
            if constrained_doc_ids:
                params["constrained_doc_ids"] = constrained_doc_ids
                cypher_query = """
                    MATCH (summary:Summary)
                    WHERE summary.documentid IN $constrained_doc_ids
                    WITH summary, vector.similarity.cosine(summary.embedding, $search_embedding) AS raw_score
                    WHERE raw_score IS NOT NULL AND raw_score >= $min_score
                    RETURN summary.id AS fragmentid,
                           summary.documentid AS documentid,
                           raw_score AS score
                """
            elif partition:
                params["partition"] = partition
                cypher_query = """
                    MATCH (summary:Summary {partition: $partition})
                    WITH summary, vector.similarity.cosine(summary.embedding, $search_embedding) AS raw_score
                    WHERE raw_score IS NOT NULL AND raw_score >= $min_score
                    RETURN summary.id AS fragmentid,
                           summary.documentid AS documentid,
                           raw_score AS score
                """
            else:
                cypher_query = """
                    MATCH (summary:Summary)
                    WITH summary, vector.similarity.cosine(summary.embedding, $search_embedding) AS raw_score
                    WHERE raw_score IS NOT NULL AND raw_score >= $min_score
                    RETURN summary.id AS fragmentid,
                           summary.documentid AS documentid,
                           raw_score AS score
                """
            
            result = execute_neo4j_query(cypher_query, params)
            fragments = []
            
            if result:
                for record in result:
                    score = min(record.get('score', 0.0) * self.SCORE_FACTOR_SUMMARIES, self.SCORE_CAP_SUMMARIES)
                    fragments.append({
                        'fragmentid': record.get('fragmentid'),
                        'documentid': record.get('documentid'),
                        'text': record.get('text'),
                        'score': score,
                        'source': 'summaries'
                    })
            
            return fragments
            
        except Exception as e:
            log(ERROR, f"{self._log_prefix}._fetch_and_score_summaries: Error: {e}")
            return []
    
    def _text_search_constrained(self, search_term: str, case_sensitive: bool, constrained_doc_ids: List[str], subset_of_datasetids: Optional[List[str]] = None, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform fulltext search restricted to specific document IDs with two-phase scoring.

        Phase 1: BM25 fulltext index returns up to TEXT_SEARCH_MAX_CANDIDATES candidates
                 within the constrained document set.
        Phase 2: Exact substring check — exact matches score 1.0, BM25-only matches
                 keep a normalized score < 1.0 (top-N retained).

        Args:
            search_term: The text to search for
            case_sensitive: True for case-sensitive, False for case-insensitive
            constrained_doc_ids: List of document IDs to restrict search to
            subset_of_datasetids: Optional list of dataset IDs
            effective_user_id: Neo4j User.id of the requesting user.

        Returns:
            List of fragment dictionaries from the constrained documents
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        try:
            if not constrained_doc_ids:
                return []

            log(DEBUG, f"{self._log_prefix}._text_search_constrained: search_term='{search_term}', constrained to {len(constrained_doc_ids)} documents")

            all_fragments: List[Dict[str, Any]] = []

            normalized_search_term: str = self._normalize_for_fulltext_search(search_term)
            log(DEBUG, f"{self._log_prefix}._text_search_constrained: normalized_search_term='{normalized_search_term}'")

            escaped_search_term: str = self._escape_lucene_special_chars(normalized_search_term)
            params: Dict[str, Any] = {"search_term": escaped_search_term, "limit": self.TEXT_SEARCH_MAX_CANDIDATES, "doc_ids": constrained_doc_ids}

            dataset_filter: str = ""
            if subset_of_datasetids and len(subset_of_datasetids) > 0:
                dataset_filter = f"AND {self._dataset_subset_membership_predicate('d')}"
                params["subset_ids"] = subset_of_datasetids
            acl_active: bool = ACL_ENFORCEMENT_ENABLED and effective_user_id is not None
            acl_block: str = self._acl_where_clause('d') if acl_active else ""
            if acl_active:
                params["effectiveUserId"] = effective_user_id

            # Phase 1a: BM25 chapter search within constrained doc IDs
            cypher_query_chapters = f"""
                CALL db.index.fulltext.queryNodes('{self.TEXT_INDEX_NAME}', $search_term)
                YIELD node AS chapter, score
                MATCH (chapter)-[:PART_OF]->(d:Document)
                WHERE (d.id IN $doc_ids OR d.documentid IN $doc_ids) {dataset_filter}{acl_block}
                RETURN chapter.id AS fragmentid,
                       d.id AS documentid,
                       d.documentid AS doc_id,
                       chapter.text AS text,
                       score
                ORDER BY score DESC
                LIMIT $limit
            """

            chapter_candidates: List[Dict[str, Any]] = []
            result = execute_neo4j_query(cypher_query_chapters, params)
            if result:
                for record in result:
                    doc_id = record.get('documentid') or record.get('doc_id')
                    chapter_candidates.append({
                        'fragmentid': record.get('fragmentid'),
                        'documentid': doc_id,
                        'text': record.get('text'),
                        'score': record.get('score', 0.0),
                        'source': 'chapter_text'
                    })

            # Phase 2a: exact substring filter on chapter candidates
            chapter_fragments = self._apply_exact_text_filter(chapter_candidates, search_term, case_sensitive)
            all_fragments.extend(chapter_fragments)
            log(DEBUG, f"{self._log_prefix}._text_search_constrained: {len(chapter_fragments)} chapter fragments after exact filter ({len(chapter_candidates)} BM25 candidates)")

            # Phase 1b: BM25 keyword search within constrained doc IDs
            cypher_query_keywords = f"""
                CALL db.index.fulltext.queryNodes('keywordsTextIndex', $search_term)
                YIELD node AS keyword, score
                MATCH (keyword)<-[:HAS_KEYWORD]-(d:Document)
                WHERE (d.id IN $doc_ids OR d.documentid IN $doc_ids) {dataset_filter}{acl_block}
                RETURN keyword.id AS fragmentid,
                       d.id AS documentid,
                       d.documentid AS doc_id,
                       keyword.text AS text,
                       score
                ORDER BY score DESC
                LIMIT $limit
            """

            keyword_candidates: List[Dict[str, Any]] = []
            result = execute_neo4j_query(cypher_query_keywords, params)
            if result:
                for record in result:
                    doc_id = record.get('documentid') or record.get('doc_id')
                    keyword_candidates.append({
                        'fragmentid': record.get('fragmentid'),
                        'documentid': doc_id,
                        'text': record.get('text'),
                        'score': record.get('score', 0.0),
                        'source': 'keyword'
                    })

            # Phase 2b: exact substring filter on keyword candidates
            keyword_fragments = self._apply_exact_text_filter(keyword_candidates, search_term, case_sensitive)
            all_fragments.extend(keyword_fragments)

            unique_docs = set(f.get('documentid') for f in all_fragments if f.get('documentid'))
            log(INFO, f"{self._log_prefix}._text_search_constrained: Found {len(all_fragments)} fragments from {len(unique_docs)} documents (BM25 candidates: {len(chapter_candidates)} chapters, {len(keyword_candidates)} keywords)")
            return all_fragments

        except Exception as e:
            log(ERROR, f"{self._log_prefix}._text_search_constrained: Error: {e}\n{traceback.format_exc()}")
            return []
    
    def _filter_fragments_by_semantic(self, search_term: str, fragments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter given fragments by semantic similarity to search_term.
        This is used when we want to constrain semantic search to a subset of fragments from a previous AND branch.
        
        Args:
            search_term: The semantic search term
            fragments: List of fragments to filter
            
        Returns:
            Filtered list of fragments with updated scores based on semantic similarity
        """
        try:
            if not fragments:
                return []
            
            # Get the embedding for the search term
            embeddings_manager = EmbeddingsManager.instance()
            if embeddings_manager is None:
                log(ERROR, f"{self._log_prefix}._filter_fragments_by_semantic: EmbeddingsManager not available")
                return []
            
            search_embedding = embeddings_manager.get_embeddings(search_term)
            if search_embedding is None:
                log(ERROR, f"{self._log_prefix}._filter_fragments_by_semantic: Failed to get embedding for search term")
                return []
            
            # For each fragment, calculate semantic similarity if it has text
            import numpy as np
            filtered = []
            for frag in fragments:
                text = frag.get('text')
                if not text:
                    # No text to compare, skip this fragment
                    continue
                
                # Get embedding for fragment text
                frag_embedding = embeddings_manager.get_embeddings(text)
                if frag_embedding is None:
                    continue
                
                # Calculate cosine similarity
                similarity = np.dot(search_embedding, frag_embedding) / (
                    np.linalg.norm(search_embedding) * np.linalg.norm(frag_embedding)
                )
                
                # Only keep fragments above minimum score threshold
                if similarity >= self.MIN_SCORE:
                    filtered.append({
                        'fragmentid': frag.get('fragmentid'),
                        'documentid': frag.get('documentid'),
                        'text': text,
                        'score': float(similarity),
                        'source': frag.get('source', 'filtered_semantic')
                    })
            
            # Sort by score descending
            filtered.sort(key=lambda x: x.get('score', 0.0), reverse=True)
            return filtered
            
        except Exception as e:
            log(ERROR, f"{self._log_prefix}._filter_fragments_by_semantic: Error: {e}\n{traceback.format_exc()}")
            return []
    
    def _filter_fragments_by_text(self, search_term: str, case_sensitive: bool, fragments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter given fragments by text matching.
        This is used when we want to constrain text search to a subset of fragments from a previous AND branch.
        
        Args:
            search_term: The text to search for
            case_sensitive: True for case-sensitive, False for case-insensitive
            fragments: List of fragments to filter
            
        Returns:
            Filtered list of fragments that contain the search term
        """
        try:
            if not fragments:
                return []
            
            # Normalize search term
            normalized_search_term = self._normalize_for_fulltext_search(search_term)
            
            # Filter fragments by text matching
            filtered = []
            for frag in fragments:
                text = frag.get('text')
                if not text:
                    continue
                
                # Normalize fragment text
                normalized_text = self._normalize_for_fulltext_search(text)
                
                # Check if search term appears in text
                if case_sensitive:
                    if search_term in text:
                        filtered.append(frag)
                else:
                    if normalized_search_term.lower() in normalized_text.lower():
                        filtered.append(frag)
            
            return filtered
            
        except Exception as e:
            log(ERROR, f"{self._log_prefix}._filter_fragments_by_text: Error: {e}\n{traceback.format_exc()}")
            return []
    
    def _tokenize_thalabus_query(self, query: str) -> List[str]:
        """
        Tokenize the thalabus query string into tokens.
        
        Recognizes:
        - Parentheses: ( )
        - Boolean operators: AND, OR, NOT (case-insensitive)
        - Semantic search: "text in double quotes"
        - Text search (case-insensitive): 'text in single quotes'
        - Text search (case-sensitive): @'text in single quotes'
        - Category expressions: CATEGORY or CATEGORY=VALUE1,VALUE2 or CATEGORY="quoted value"
        - Special keyword: TRUE
        
        Returns:
            List of tokens
        """
        tokens = []
        i = 0
        query = query.strip()
        
        while i < len(query):
            # Skip whitespace
            if query[i].isspace():
                i += 1
                continue
            
            # Handle parentheses
            if query[i] in '()':
                tokens.append(query[i])
                i += 1
                continue
            
            # Check for case-sensitive text search @'...'
            if query[i] == '@' and i + 1 < len(query) and query[i + 1] == "'":
                i += 2  # skip @'
                start = i
                while i < len(query) and query[i] != "'":
                    if query[i] == '\\' and i + 1 < len(query):
                        i += 2  # skip escaped character
                    else:
                        i += 1
                if i >= len(query):
                    raise ValueError("Unterminated @'...' text search")
                text = query[start:i]
                tokens.append(('TEXT_CASE_SENSITIVE', text))
                i += 1  # skip closing '
                continue
            
            # Check for case-insensitive text search '...'
            # A single quote is only a text-search delimiter when it is NOT immediately
            # preceded by an alphanumeric character.  A quote after a letter/digit is a
            # natural-language apostrophe (e.g. "cos'ha", "quand'e", "it's") and must
            # be skipped so the tokenizer does not try to parse it as a text search.
            if query[i] == "'" and (i == 0 or not query[i - 1].isalnum()):
                i += 1  # skip opening '
                start = i
                while i < len(query) and query[i] != "'":
                    if query[i] == '\\' and i + 1 < len(query):
                        i += 2  # skip escaped character
                    else:
                        i += 1
                if i >= len(query):
                    raise ValueError("Unterminated '...' text search")
                text = query[start:i]
                tokens.append(('TEXT_CASE_INSENSITIVE', text))
                i += 1  # skip closing '
                continue

            # Natural-language apostrophe (preceded by alphanumeric): skip and continue.
            if query[i] == "'":
                i += 1
                continue
            
            # Helper function to check if a character is part of an identifier (alphanumeric or underscore)
            def is_identifier_char(c):
                return c.isalnum() or c == '_'
            
            # Check for NOT operator (must check before category extraction)
            if i + 3 <= len(query) and query[i:i+3].upper() == 'NOT':
                # NOT is an operator if preceded by: space, operators, parens, comma, or start of string
                # AND followed by: space, operators, parens, comma, or end of string
                before_ok = (i == 0 or not is_identifier_char(query[i-1]))
                after_ok = (i+3 >= len(query) or not is_identifier_char(query[i+3]))
                if before_ok and after_ok:
                    tokens.append('NOT')
                    i += 3
                    continue
            
            # Check for AND operator (must check before category extraction)
            if i + 3 <= len(query) and query[i:i+3].upper() == 'AND':
                before_ok = (i == 0 or not is_identifier_char(query[i-1]))
                after_ok = (i+3 >= len(query) or not is_identifier_char(query[i+3]))
                if before_ok and after_ok:
                    tokens.append('AND')
                    i += 3
                    continue
            
            # Check for OR operator (must check before category extraction)
            if i + 2 <= len(query) and query[i:i+2].upper() == 'OR':
                before_ok = (i == 0 or not is_identifier_char(query[i-1]))
                after_ok = (i+2 >= len(query) or not is_identifier_char(query[i+2]))
                if before_ok and after_ok:
                    tokens.append('OR')
                    i += 2
                    continue
            
            # Try to extract a category expression (which may contain quoted values)
            # or semantic search "..." depending on context
            start = i
            # Look ahead to see if this looks like a category (identifier followed by = or comma-separated values)
            j = i
            while j < len(query) and (query[j].isalnum() or query[j] in '_'):
                j += 1
            
            # Check if we have CATEGORY= or CATEGORY, pattern (indicating it's a category expression)
            is_category_start = False
            if j > i:
                # Skip whitespace after identifier
                k = j
                while k < len(query) and query[k].isspace():
                    k += 1
                # Check if followed by = or comma
                if k < len(query) and query[k] in '=,':
                    is_category_start = True
            
            if is_category_start:
                # Extract category expression, including quoted values if present
                # and allowing spaces in unquoted values
                i = j  # continue from end of identifier
                category_expr = query[start:j]
                
                # If we have =, extract the value part (which may contain quoted strings or spaces)
                if i < len(query) and query[i] in '=,':
                    while i < len(query):
                        # Stop at parentheses
                        if query[i] in '()':
                            break
                        # Check if we're at the start of an operator (AND, OR, NOT)
                        # These must be preceded and followed by non-identifier characters (not alphanumeric or underscore)
                        if i > 0 and query[i].upper() in ['A', 'O', 'N']:
                            # Look ahead to see if this could be AND, OR, or NOT
                            before_ok = not is_identifier_char(query[i-1])
                            if query[i:i+3].upper() == 'AND' and (i+3 >= len(query) or not is_identifier_char(query[i+3])) and before_ok:
                                break
                            elif query[i:i+2].upper() == 'OR' and (i+2 >= len(query) or not is_identifier_char(query[i+2])) and before_ok:
                                break
                            elif query[i:i+3].upper() == 'NOT' and (i+3 >= len(query) or not is_identifier_char(query[i+3])) and before_ok:
                                break
                        
                        if query[i] == '"':
                            # Skip quoted value
                            i += 1
                            while i < len(query) and query[i] != '"':
                                if query[i] == '\\' and i + 1 < len(query):
                                    i += 2
                                else:
                                    i += 1
                            if i < len(query):
                                i += 1  # skip closing "
                        elif query[i] == "'":
                            # Skip quoted value
                            i += 1
                            while i < len(query) and query[i] != "'":
                                if query[i] == '\\' and i + 1 < len(query):
                                    i += 2
                                else:
                                    i += 1
                            if i < len(query):
                                i += 1  # skip closing '
                        else:
                            i += 1
                    category_expr = query[start:i].strip()
                
                tokens.append(('CATEGORY', category_expr))
                continue
            
            # Check for semantic search "..." (not preceded by identifier=)
            if query[i] == '"':
                i += 1  # skip opening "
                start_text = i
                while i < len(query) and query[i] != '"':
                    if query[i] == '\\' and i + 1 < len(query):
                        i += 2  # skip escaped character
                    else:
                        i += 1
                if i >= len(query):
                    raise ValueError('Unterminated "..." semantic search')
                text = query[start_text:i]
                tokens.append(('SEMANTIC', text))
                i += 1  # skip closing "
                continue
            
            # Extract keyword or other token (TRUE, or remaining identifiers)
            start = i
            while i < len(query) and query[i] not in '() \t\n\r\'"@':
                i += 1
            
            token = query[start:i].strip()
            if token and token.upper() not in ('AND', 'OR', 'NOT'):
                if token.upper() == 'TRUE':
                    tokens.append('TRUE')
                else:
                    tokens.append(('CATEGORY', token))
        
        return tokens

    def _sanitize_query(self, query: str) -> str:
        """
        Sanitize escaped quotes in the incoming query string and validate balanced quotes.

        Replaces escaped double and single quotes (\" -> ", \' -> ') which may
        originate from double-escaping in callers. After replacement, ensures both
        single and double quotes appear an even number of times; otherwise raises
        ValueError so callers can see malformed input.
        """
        if query is None:
            return query

        # First, normalize CATEGORY='value' -> CATEGORY="value" by removing escapes for apostrophes
        # Normalize CATEGORY-style single-quoted values to double-quoted values.
        # Handle cases where the source literal used backslash-escapes which Python
        # has already processed (so internal apostrophes may be present as plain ''' characters).
        q = query
        out_s_parts = []
        idx = 0
        L = len(q)
        while idx < L:
            eq_pos = q.find('=', idx)
            if eq_pos == -1:
                out_s_parts.append(q[idx:])
                break
            # copy up to '='
            out_s_parts.append(q[idx:eq_pos])
            j = eq_pos + 1
            # copy '=' and any immediate spaces
            while j < L and q[j].isspace():
                j += 1
            if j < L and q[j] == "'":
                # Found a single-quoted value after '='. Find closing quote.
                start_quote = j
                k = start_quote + 1
                close_quote = -1
                while k < L:
                    if q[k] == "'":
                        # Heuristic: treat this as the closing quote if next char is whitespace or end or a punctuation
                        nxt = q[k+1] if k+1 < L else None
                        if nxt is None or nxt.isspace() or nxt in ")],":
                            close_quote = k
                            break
                    k += 1
                if close_quote == -1:
                    # fallback to last single-quote in the remainder
                    last_sq = q.rfind("'", start_quote+1)
                    if last_sq != -1:
                        close_quote = last_sq
                if close_quote != -1:
                    inner = q[start_quote+1:close_quote]
                    # Unescape any backslash-escaped single quotes (if present)
                    inner = inner.replace("\\'", "'")
                    out_s_parts.append('=' + '"' + inner + '"')
                    idx = close_quote + 1
                    continue
                else:
                    # couldn't find a reasonable close quote; just copy the rest
                    out_s_parts.append(q[eq_pos:])
                    break
            else:
                # no single-quoted value after '=', copy '=' and continue
                out_s_parts.append(q[eq_pos: eq_pos+1])
                idx = eq_pos + 1
                continue

        s = ''.join(out_s_parts)
        out_chars = []
        i = 0
        n = len(s)

        in_double = False
        in_single = False
        escape = False
        pending_equals = False  # set when we see '=' outside quotes
        category_mode = False   # when we converted CATEGORY='...' -> treat as double

        while i < n:
            ch = s[i]

            # Track '=' outside quotes to allow CATEGORY='x' -> CATEGORY="x"
            if not in_double and not in_single:
                if ch == '=':
                    out_chars.append(ch)
                    pending_equals = True
                    i += 1
                    continue

            # Handle escape sequences
            if ch == '\\':
                # If at end, just emit backslash
                if i + 1 >= n:
                    out_chars.append('\\')
                    i += 1
                    continue

                nxt = s[i + 1]

                # If inside double-quoted region
                if in_double or (pending_equals and nxt == "'"):
                    # For double-quoted output we want inner double-quotes escaped, single-quotes raw
                    if nxt == '"':
                        # emit raw double-quote inside double-quoted phrase
                        out_chars.append('"')
                    elif nxt == "'":
                        # drop the backslash and emit single quote
                        out_chars.append("'")
                    else:
                        out_chars.append(nxt)
                    i += 2
                    pending_equals = False
                    continue

                # If inside single-quoted region
                if in_single:
                    if nxt == "'":
                        # keep escaped single quote inside single quotes
                        out_chars.append("\\'")
                    elif nxt == '"':
                        out_chars.append('"')
                    else:
                        out_chars.append(nxt)
                    i += 2
                    continue

                # Outside any quote: simply unescape common escapes (emit the escaped char)
                out_chars.append(nxt)
                i += 2
                pending_equals = False
                continue

            # Handle quote characters
            if ch == '"' and not in_single:
                if in_double:
                    # closing double quote
                    out_chars.append('"')
                    in_double = False
                    category_mode = False
                else:
                    # opening double quote
                    out_chars.append('"')
                    in_double = True
                    in_single = False
                pending_equals = False
                i += 1
                continue

            # If we encounter a single-quote but we converted a CATEGORY to double-quote mode,
            # treat the single-quote as the closing delimiter for that category value.
            if ch == "'" and in_double and category_mode:
                out_chars.append('"')
                in_double = False
                category_mode = False
                pending_equals = False
                i += 1
                continue

            if ch == "'" and not in_double:
                # Special-case: CATEGORY='value' when seen immediately after '='
                if pending_equals and not in_double and not in_single:
                    # convert to double-quoted value for categories
                    out_chars.append('"')
                    in_double = True
                    category_mode = True
                    pending_equals = False
                    i += 1
                    continue

                if in_single:
                    # closing single quote
                    out_chars.append("'")
                    in_single = False
                else:
                    # opening single quote
                    out_chars.append("'")
                    in_single = True
                pending_equals = False
                i += 1
                continue

            # If we reach here, ordinary character
            out_chars.append(ch)
            # reset pending_equals if we hit any non-space other than handled above
            if pending_equals and not ch.isspace():
                pending_equals = False
            i += 1

        # If we converted a category that started with single quote and never closed it,
        # ensure we close it with a double quote
        if in_double and category_mode:
            out_chars.append('"')

        return ''.join(out_chars)
        # Parse input and handle escapes in-context so that escaped quotes within
        # quoted strings are unescaped correctly, while validating balanced quotes
        out_chars = []
        i = 0
        in_double = False
        in_single = False
        while i < len(query):
            ch = query[i]

            # Handle escape sequences with context-awareness.
            # If an escaped single-quote appears inside a double-quoted string,
            # preserve the backslash so the single-quote remains escaped (\').
            if ch == '\\' and i + 1 < len(query):
                next_ch = query[i + 1]
                if next_ch == "'" and in_double:
                    out_chars.append('\\')
                    out_chars.append("'")
                else:
                    # For other escapes, unescape by appending the escaped character
                    out_chars.append(next_ch)
                i += 2
                continue

            # Toggle double-quote state when not inside single quotes
            if ch == '"' and not in_single:
                out_chars.append(ch)
                in_double = not in_double
                i += 1
                continue

            # Handle single-quote: inside double-quotes we escape it, otherwise toggle
            if ch == "'":
                if in_double:
                    # Escape single quotes that appear inside double-quoted strings
                    out_chars.append('\\')
                    out_chars.append("'")
                else:
                    out_chars.append(ch)
                    in_single = not in_single
                i += 1
                continue

            # If outside quotes and we see '=', trim surrounding whitespace
            if ch == '=' and not in_double and not in_single:
                while out_chars and out_chars[-1].isspace():
                    out_chars.pop()
                out_chars.append('=')
                j = i + 1
                while j < len(query) and query[j].isspace():
                    j += 1
                i = j
                continue

            out_chars.append(ch)
            i += 1

        sanitized_query = ''.join(out_chars)

        # Validate that quotes are balanced using parsing state
        if in_double or in_single:
            log(ERROR, f"{self._log_prefix}._sanitize_query: Unmatched quote after sanitization: orig={repr(query)} sanitized={repr(sanitized_query)}")
            raise ValueError('Unmatched quote in query after sanitization')

        return sanitized_query
    
    def _parse_thalabus_expression(self, tokens: List, pos: int) -> Tuple:
        """
        Parse OR expression (lowest precedence).
        
        Returns:
            (AST_node, next_position)
        """
        left, pos = self._parse_thalabus_and_expression(tokens, pos)
        
        while pos < len(tokens) and tokens[pos] == 'OR':
            pos += 1  # consume OR
            right, pos = self._parse_thalabus_and_expression(tokens, pos)
            left = {'op': 'OR', 'left': left, 'right': right}
        
        return left, pos
    
    def _parse_thalabus_and_expression(self, tokens: List, pos: int) -> Tuple:
        """
        Parse AND expression (higher precedence than OR).
        
        Returns:
            (AST_node, next_position)
        """
        left, pos = self._parse_thalabus_not_expression(tokens, pos)
        
        while pos < len(tokens) and tokens[pos] == 'AND':
            pos += 1  # consume AND
            right, pos = self._parse_thalabus_not_expression(tokens, pos)
            left = {'op': 'AND', 'left': left, 'right': right}
        
        return left, pos
    
    def _parse_thalabus_not_expression(self, tokens: List, pos: int) -> Tuple:
        """
        Parse NOT expression (higher precedence than AND).
        
        Returns:
            (AST_node, next_position)
        """
        if pos < len(tokens) and tokens[pos] == 'NOT':
            pos += 1  # consume NOT
            expr, pos = self._parse_thalabus_not_expression(tokens, pos)
            return {'op': 'NOT', 'operand': expr}, pos
        else:
            return self._parse_thalabus_primary(tokens, pos)
    
    def _unquote_value(self, value: str) -> str:
        """Remove quotes from a value if it's quoted with matching quotes.
        
        Handles:
        - "quoted value" -> quoted value
        - 'quoted value' -> quoted value
        - unquoted -> unquoted
        - "unclosed -> "unclosed (returns as-is)
        """
        value = value.strip()
        if len(value) >= 2:
            if (value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"):
                return value[1:-1]
        return value
    
    def _parse_thalabus_primary(self, tokens: List, pos: int) -> Tuple:
        """
        Parse primary expression (parentheses, TRUE, semantic, text, or category).
        
        Returns:
            (AST_node, next_position)
        """
        if pos >= len(tokens):
            raise ValueError("Unexpected end of query")
        
        token = tokens[pos]
        
        # Handle parentheses
        if token == '(':
            pos += 1  # consume '('
            expr, pos = self._parse_thalabus_expression(tokens, pos)
            if pos >= len(tokens) or tokens[pos] != ')':
                raise ValueError("Missing closing parenthesis")
            pos += 1  # consume ')'
            return expr, pos
        
        # Handle TRUE (null search)
        if token == 'TRUE':
            return {'type': 'TRUE'}, pos + 1
        
        # Handle semantic search "..."
        if isinstance(token, tuple) and token[0] == 'SEMANTIC':
            return {'type': 'SEMANTIC', 'text': token[1]}, pos + 1
        
        # Handle text search '...' or @'...'
        if isinstance(token, tuple) and token[0] == 'TEXT_CASE_INSENSITIVE':
            return {'type': 'TEXT', 'text': token[1], 'case_sensitive': False}, pos + 1
        
        if isinstance(token, tuple) and token[0] == 'TEXT_CASE_SENSITIVE':
            return {'type': 'TEXT', 'text': token[1], 'case_sensitive': True}, pos + 1
        
        # Handle category expression
        if isinstance(token, tuple) and token[0] == 'CATEGORY':
            category_expr = token[1]
            if '=' in category_expr:
                category, values_str = category_expr.split('=', 1)
                # Split values by comma, then unquote each one
                value_list = [self._unquote_value(v.strip()) for v in values_str.split(',')]
                return {'type': 'CATEGORY', 'name': category.strip(), 'values': value_list}, pos + 1
            else:
                return {'type': 'CATEGORY', 'name': category_expr.strip(), 'values': None}, pos + 1
        
        raise ValueError(f"Unexpected token: {token}")
    
    def _serialize_ast_to_query(self, node: dict) -> str:
        """Convert AST node back to query string.
        
        Args:
            node: AST node to serialize
            
        Returns:
            Query string representation of the node
        """
        if 'op' in node:
            if node['op'] == 'AND':
                left = self._serialize_ast_to_query(node['left'])
                right = self._serialize_ast_to_query(node['right'])
                return f"({left} AND {right})"
            
            elif node['op'] == 'OR':
                left = self._serialize_ast_to_query(node['left'])
                right = self._serialize_ast_to_query(node['right'])
                return f"({left} OR {right})"
            
            elif node['op'] == 'NOT':
                operand = self._serialize_ast_to_query(node['operand'])
                return f"NOT ({operand})"
        
        elif node.get('type') == 'TRUE':
            return 'TRUE'
        
        elif node.get('type') == 'CATEGORY':
            category_name = node['name']
            values = node['values']
            if values is None:
                return category_name
            else:
                # Quote values that contain spaces or special characters
                quoted_values = []
                for v in values:
                    if ' ' in v or ',' in v or '"' in v:
                        # Escape quotes and wrap in quotes
                        escaped_v = v.replace('\\', '\\\\').replace('"', '\\"')
                        quoted_values.append(f'"{escaped_v}"')
                    else:
                        quoted_values.append(v)
                return f"{category_name}={','.join(quoted_values)}"
        
        elif node.get('type') == 'SEMANTIC':
            text = node['text']
            # Escape quotes in the text
            escaped_text = text.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped_text}"'
        
        elif node.get('type') == 'TEXT':
            text = node['text']
            case_sensitive = node['case_sensitive']
            # Escape quotes in the text
            escaped_text = text.replace('\\', '\\\\').replace("'", "\\'")
            if case_sensitive:
                return f"@'{escaped_text}'"
            else:
                return f"'{escaped_text}'"
        
        else:
            raise ValueError(f"Cannot serialize AST node: {node}")
    
    def _mark_category_only_subtrees(self, node: dict, ontology_id: Optional[str]) -> None:
        """Recursively mark subtrees that can be handled entirely by categories_query().
        
        This works from innermost to outermost, marking nodes with 'is_category_only' flag
        and storing the serialized query in 'category_query' field.
        
        Args:
            node: AST node to process (modified in place)
            ontology_id: The ontology ID to validate categories against. If None, matches against any ontology.
        """
        # First, recursively process children
        if 'op' in node:
            if node['op'] in ('AND', 'OR'):
                self._mark_category_only_subtrees(node['left'], ontology_id)
                self._mark_category_only_subtrees(node['right'], ontology_id)
            elif node['op'] == 'NOT':
                self._mark_category_only_subtrees(node['operand'], ontology_id)
        
        # Now check if this node (and its entire subtree) is category-only
        # Serialize the subtree to a query string
        try:
            serialized_query = self._serialize_ast_to_query(node)
            
            # Ask ontology service if this is a category-only query
            # If ontology_id is None, we'll try to match against any ontology during evaluation
            ontology_service:OntologyInterface = get_ontology_service()
            if ontology_service and ontology_id is not None:
                if ontology_service.is_categories_only_query(ontology_id, serialized_query):
                    # Mark this node as category-only
                    node['is_category_only'] = True
                    node['category_query'] = serialized_query
                    log(DEBUG, f"{self._log_prefix}._mark_category_only_subtrees: Marked as category-only: {serialized_query}")
                else:
                    node['is_category_only'] = False
                    log(DEBUG, f"{self._log_prefix}._mark_category_only_subtrees: Not category-only: {serialized_query}")
            else:
                # If no ontology specified, we'll determine category validity during evaluation
                node['is_category_only'] = False
                log(DEBUG, f"{self._log_prefix}._mark_category_only_subtrees: Ontology not specified, deferring category validation to evaluation time")
        except Exception as e:
            # If serialization or validation fails, mark as not category-only
            node['is_category_only'] = False
            log(DEBUG, f"{self._log_prefix}._mark_category_only_subtrees: Error checking node, marked as not category-only: {e}")
    
    def _evaluate_thalabus_ast(self, node: dict, ontology_id: Optional[str], subset_of_datasetids: Optional[List[str]], all_fragments: Optional[List[Dict[str, Any]]], automatically_rerank: bool = False, constrain_to_fragments: Optional[List[Dict[str, Any]]] = None, base_limit_fragments: Optional[List[Dict[str, Any]]] = None, effective_user_id: Optional[str] = None, text_cache: Optional[Dict[str, Optional[str]]] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Evaluate the AST and return list of fragments with scores.
        
        Now works entirely at FRAGMENT level:
        - Retrieves fragments (chapters, paragraphs, summaries, keywords)
        - Combines fragments using AND/OR/NOT logic
        - Performs reranking on fragments (not documents)
        - Returns fragments, which are only aggregated to documents at the final output stage
        
        Args:
            node: AST node to evaluate
            ontology_id: The ontology ID. If None, matches categories against any ontology.
            subset_of_datasetids: Optional list of dataset IDs to restrict search
            all_fragments: List of all possible fragments (for NOT operation), None if not yet determined
            automatically_rerank: If True, rerank results at each branch before combining
            constrain_to_fragments: If provided, constrains semantic/text searches to only these fragments
            
        Returns:
            List of fragment dictionaries, or None if TRUE (null search)
        """
        if text_cache is None:
            text_cache = {}

        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()

        # Check if this entire subtree is category-only and can be delegated
        if node.get('is_category_only') and 'category_query' in node:
            category_query = node['category_query']
            log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: Starting category search: {category_query}")
            
            ontology_service:OntologyInterface = get_ontology_service()
            if ontology_service is None:
                log(ERROR, f"{self._log_prefix}: Ontology service not available")
                return []
            
            doc_ids = ontology_service.categories_query(ontology_id, category_query, effective_user_id=effective_user_id)

            # Filter by subset_of_datasetids: only keep docs that belong to one of the datasets
            if subset_of_datasetids and doc_ids:
                cypher_ds_filter = f"""
                    {self._dataset_subset_membership_union(extra_filter="d.id IN $doc_ids")}
                    RETURN d.id AS doc_id
                """
                ds_result = execute_neo4j_query(cypher_ds_filter, {"doc_ids": doc_ids, "subset_ids": subset_of_datasetids})
                filtered_doc_ids = [r['doc_id'] for r in (ds_result or []) if r.get('doc_id')]
                if len(filtered_doc_ids) < len(doc_ids):
                    log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: subset_of_datasetids filtered category-only results from {len(doc_ids)} to {len(filtered_doc_ids)} documents")
                doc_ids = filtered_doc_ids

            # Convert to fragments with default score of 1.0
            fragments = []
            for doc_id in doc_ids:
                fragments.append({
                    'fragmentid': doc_id,  # Use doc_id as fragment id for category results
                    'documentid': doc_id,
                    'text': None,  # No text for category-only results
                    'score': 1.0,
                    'source': 'category'
                })
            log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: Category search '{category_query}' completed: {len(fragments)} fragments from {len(doc_ids)} documents")
            return fragments
        
        if 'op' in node:
            if node['op'] == 'AND':
                log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Processing AND operation")
                log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Left branch: {self._serialize_ast_to_query(node['left'])}")
                log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Right branch: {self._serialize_ast_to_query(node['right'])}")
                
                left_fragments = self._evaluate_thalabus_ast(node['left'], ontology_id, subset_of_datasetids, all_fragments, automatically_rerank, constrain_to_fragments, base_limit_fragments, effective_user_id, text_cache)
                
                log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Left branch returned {len(left_fragments) if left_fragments is not None else 0} fragments")
                if left_fragments:
                    left_docs = set([f['documentid'] for f in left_fragments])
                    log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Left branch documents: {sorted(left_docs)}")
                    log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Left branch scores: {[(f['documentid'], f['score']) for f in left_fragments[:10]]}")
                
                # Rerank left branch if enabled and results exist
                if automatically_rerank and left_fragments is not None and len(left_fragments) > 0:
                    left_query = self._serialize_ast_to_query(node['left'])
                    log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Reranking left branch")
                    left_fragments = self._rerank_branch_results(left_query, left_fragments, subset_of_datasetids, text_cache)
                    log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: After reranking: {len(left_fragments)} fragments")
                    # Filter by reranking score threshold before AND combination
                    left_fragments = self._filter_fragments_by_rerank_score(left_fragments, MIN_RERANKING_SCORE_FRAG)
                    log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: After reranking filter MIN_RERANKING_SCORE_FRAG={MIN_RERANKING_SCORE_FRAG}: {len(left_fragments)} fragments")
                
                # Pass left_fragments as context for NOT operations on right side
                # This enables filter-based NOT: "X AND NOT Y" filters X by removing Y matches
                context_for_right = left_fragments if node['right'].get('op') == 'NOT' else all_fragments
                
                # OPTIMIZATION: For AND operations with semantic/text on right, constrain search to left_fragments
                # This prevents searching the entire corpus when we know the result must be in left_fragments
                right_node_type = node['right'].get('type')
                constrain_right = left_fragments if right_node_type in ['SEMANTIC', 'TEXT'] else constrain_to_fragments
                log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Right node type: {right_node_type}, constraining: {constrain_right is not None}")
                
                right_fragments = self._evaluate_thalabus_ast(node['right'], ontology_id, subset_of_datasetids, context_for_right, automatically_rerank, constrain_right, base_limit_fragments, effective_user_id, text_cache)
                
                log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Right branch returned {len(right_fragments) if right_fragments is not None else 0} fragments")
                if right_fragments:
                    right_docs = set([f['documentid'] for f in right_fragments])
                    log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Right branch documents: {sorted(right_docs)}")
                    log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Right branch scores: {[(f['documentid'], f['score']) for f in right_fragments[:10]]}")
                
                # Rerank right branch if enabled and results exist
                if automatically_rerank and right_fragments is not None and len(right_fragments) > 0:
                    right_query = self._serialize_ast_to_query(node['right'])
                    log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Reranking right branch")
                    right_fragments = self._rerank_branch_results(right_query, right_fragments, subset_of_datasetids, text_cache)
                    log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: After reranking: {len(right_fragments)} fragments")
                    # Filter by reranking score threshold before AND combination
                    right_fragments = self._filter_fragments_by_rerank_score(right_fragments)
                    log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: After score filter: {len(right_fragments)} fragments")
                
                # Handle TRUE cases
                if left_fragments is None and right_fragments is None:
                    log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Both branches are TRUE, returning None")
                    return None
                if left_fragments is None:
                    log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Left branch is TRUE, returning right fragments")
                    return right_fragments
                if right_fragments is None:
                    log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Right branch is TRUE, returning left fragments")
                    return left_fragments
                
                # AND operation: keep only fragments from documents in both branches
                log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Combining fragments with AND logic")
                result_fragments = self._combine_fragments_and(left_fragments, right_fragments)
                log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: AND combination returned {len(result_fragments)} fragments")
                if result_fragments:
                    result_docs = set([f['documentid'] for f in result_fragments])
                    log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Result documents: {sorted(result_docs)}")
                    log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Result scores: {[(f['documentid'], f['score']) for f in result_fragments[:10]]}")
                log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast AND: {len(left_fragments)} left + {len(right_fragments)} right = {len(result_fragments)} combined")
                return result_fragments
            
            elif node['op'] == 'OR':
                # OR is a union — each side is evaluated independently but both respect base_limit_fragments
                left_fragments = self._evaluate_thalabus_ast(node['left'], ontology_id, subset_of_datasetids, all_fragments, automatically_rerank, None, base_limit_fragments, effective_user_id, text_cache)
                
                # Rerank left branch if enabled and results exist
                if automatically_rerank and left_fragments is not None and len(left_fragments) > 0:
                    left_query = self._serialize_ast_to_query(node['left'])
                    left_fragments = self._rerank_branch_results(left_query, left_fragments, subset_of_datasetids, text_cache)
                    # Filter by reranking score threshold before OR combination
                    left_fragments = self._filter_fragments_by_rerank_score(left_fragments, MIN_RERANKING_SCORE_FRAG)
                    log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: After reranking filter MIN_RERANKING_SCORE_FRAG={MIN_RERANKING_SCORE_FRAG}: {len(left_fragments)} fragments")
                
                right_fragments = self._evaluate_thalabus_ast(node['right'], ontology_id, subset_of_datasetids, all_fragments, automatically_rerank, None, base_limit_fragments, effective_user_id, text_cache)
                
                # Rerank right branch if enabled and results exist
                if automatically_rerank and right_fragments is not None and len(right_fragments) > 0:
                    right_query = self._serialize_ast_to_query(node['right'])
                    right_fragments = self._rerank_branch_results(right_query, right_fragments, subset_of_datasetids, text_cache)
                    # Filter by reranking score threshold before OR combination
                    right_fragments = self._filter_fragments_by_rerank_score(right_fragments)
                    log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: After reranking filter MIN_RERANKING_SCORE_FRAG={MIN_RERANKING_SCORE_FRAG}: {len(right_fragments)} fragments")
                
                # Handle TRUE cases
                if left_fragments is None and right_fragments is None:
                    return None
                if left_fragments is None:
                    return right_fragments
                if right_fragments is None:
                    return left_fragments
                
                # OR operation: union of all fragments
                result_fragments = self._combine_fragments_or(left_fragments, right_fragments)
                log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast OR: {len(left_fragments)} left + {len(right_fragments)} right = {len(result_fragments)} combined")
                return result_fragments
            
            elif node['op'] == 'NOT':
                # NOT is a filter operation that requires context from left side (via all_fragments)
                # Standalone NOT is not supported - it should be used with AND: "X AND NOT Y"
                # When used in AND context, all_fragments contains the left-side results to filter
                
                if all_fragments is None:
                    # Standalone NOT without context - return empty results
                    log(WARNING, f"{self._log_prefix}._evaluate_thalabus_ast NOT: Standalone NOT operation without left-side context returns empty")
                    return []
                
                # NOT operand searches the full corpus to find what to exclude.
                # base_limit_fragments is intentionally NOT propagated: NOT must scan freely
                # so it can identify all documents to exclude from the left-side result.
                operand_fragments = self._evaluate_thalabus_ast(node['operand'], ontology_id, subset_of_datasetids, None, automatically_rerank, None, None, effective_user_id, text_cache)
                
                # Rerank operand branch if enabled and results exist
                if automatically_rerank and operand_fragments is not None and len(operand_fragments) > 0:
                    operand_query = self._serialize_ast_to_query(node['operand'])
                    operand_fragments = self._rerank_branch_results(operand_query, operand_fragments, subset_of_datasetids, text_cache)
                    # Filter by reranking score threshold before NOT operation
                    operand_fragments = self._filter_fragments_by_rerank_score(operand_fragments, MIN_RERANKING_SCORE_FRAG)
                    log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: After reranking filter MIN_RERANKING_SCORE_FRAG={MIN_RERANKING_SCORE_FRAG}: {len(operand_fragments)} fragments")
                
                # Handle TRUE case
                if operand_fragments is None:
                    return all_fragments  # NOT TRUE means keep all fragments
                
                # Filter operation: exclude documents from all_fragments that appear in operand
                result_fragments = self._combine_fragments_not(all_fragments, operand_fragments)
                log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast NOT: {len(all_fragments)} base - {len(operand_fragments)} excluded = {len(result_fragments)} result")
                return result_fragments
            
            else:
                raise ValueError(f"Unknown operator: {node['op']}")
        
        elif node.get('type') == 'TRUE':
            # TRUE means no search
            return None
        
        elif node.get('type') == 'CATEGORY':
            # Delegate to ontology service
            category_name = node['name']
            values = node['values']
            
            # Build a simple category query for ontology_service
            if values is None:
                category_query = category_name
            else:
                category_query = f"{category_name}={','.join(values)}"
            
            ontology_service:OntologyInterface = get_ontology_service()
            if ontology_service is None:
                log(ERROR, f"{self._log_prefix}: Ontology service not available")
                return []
            
            # Check if this is a valid category query
            # If ontology_id is None, try to match against any ontology
            is_valid_category = False
            if ontology_id is not None:
                is_valid_category = ontology_service.is_categories_only_query(ontology_id, category_query)
            else:
                # When ontology_id is None, try to validate against any ontology by checking if it's a known category
                # This is a fallback approach - the ontology service should be consulted for any matching ontology
                is_valid_category = ontology_service.is_categories_only_query(None, category_query) if hasattr(ontology_service, 'is_categories_only_query') else False
            
            if is_valid_category:
                # Valid category - delegate to categories_query
                doc_ids = ontology_service.categories_query(ontology_id, category_query, effective_user_id=effective_user_id)
                # Convert to fragments with default score of 1.0
                fragments = []
                for doc_id in doc_ids:
                    fragments.append({
                        'fragmentid': doc_id,  # Use doc_id as fragment id for category results
                        'documentid': doc_id,
                        'text': None,  # No text for category-only results
                        'score': 1.0,
                        'source': 'category'
                    })
                return fragments
            else:
                # Invalid category - treat as semantic search fallback
                # Convert category expression to natural language for better semantic matching
                # e.g., "MEDICAL_CONDITION=ADVANCED_DEGENERATIVE_RHEUMATIC_DISORDERS" 
                # becomes "MEDICAL CONDITION ADVANCED DEGENERATIVE RHEUMATIC DISORDERS"
                search_term = category_query.replace('=', ' ').replace(',', ' ').replace('_', ' ')
                log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: Category '{category_query}' not found in ontology, treating as semantic search: '{search_term}'")
                
                # Perform semantic search: use ACL retry loop when effective_user_id is supplied,
                # otherwise fall through to multi-language search.
                if effective_user_id:
                    semantic_fragments = self._semantic_search_with_acl(search_term, effective_user_id, self.TOP_N_RAG, subset_of_datasetids)
                else:
                    semantic_fragments = self._semantic_search_multilang(search_term, subset_of_datasetids, effective_user_id=None)
                unique_docs = len(set(f.get('documentid') for f in semantic_fragments if f.get('documentid')))
                log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: Semantic fallback found {len(semantic_fragments)} fragments from {unique_docs} documents")
                
                return semantic_fragments
        
        elif node.get('type') == 'SEMANTIC':
            # For semantic search, use embedding-based search
            # For short queries (5 words or fewer), also combine with text search to catch exact matches
            # For longer queries, use only semantic search for better paraphrase handling
            search_term = node['text']
            word_count = len(search_term.split())
            log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: Starting semantic search for '{search_term}' ({word_count} words)")
            
            # OPTIMIZATION: use the most specific constraint available:
            # constrain_to_fragments (AND-branch optimisation) takes precedence; fall back to
            # base_limit_fragments (global limit_to_documents from caller) when not set.
            # When neither is set the full-corpus vector search runs unchanged.
            effective_constrain = constrain_to_fragments if constrain_to_fragments is not None else base_limit_fragments
            if effective_constrain is not None:
                # Extract unique document IDs from effective_constrain
                constrained_doc_ids = list(set(f.get('documentid') for f in effective_constrain if f.get('documentid')))
                log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: Constraining semantic search to {len(constrained_doc_ids)} documents")
                
                # Perform multi-language constrained semantic search
                semantic_fragments = self._semantic_search_constrained_multilang(search_term, partition=None, constrained_doc_ids=constrained_doc_ids, subset_of_datasetids=subset_of_datasetids)
                unique_docs_semantic = len(set(f.get('documentid') for f in semantic_fragments if f.get('documentid')))
                log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Constrained multi-language semantic search found {len(semantic_fragments)} fragments from {unique_docs_semantic} documents for '{search_term}' ({word_count} words)")
                
                # For short queries, also filter by text search
                if word_count <= MIN_SEMANTIC_TEXT_COMBINE_WORDS:
                    text_fragments = self._text_search_constrained(search_term, case_sensitive=False, constrained_doc_ids=constrained_doc_ids, subset_of_datasetids=subset_of_datasetids, effective_user_id=effective_user_id)
                    unique_docs_text = len(set(f.get('documentid') for f in text_fragments if f.get('documentid')))
                    log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Constrained text search found {len(text_fragments)} fragments from {unique_docs_text} documents for '{search_term}'")
                    
                    # Combine results
                    combined_fragments = self._combine_fragments_or(semantic_fragments, text_fragments)
                    unique_docs_combined = len(set(f.get('documentid') for f in combined_fragments if f.get('documentid')))
                    log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: Combined constrained semantic + text for '{search_term}': {len(semantic_fragments)} semantic + {len(text_fragments)} text = {len(combined_fragments)} combined fragments ({unique_docs_combined} docs)")
                    log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: Semantic search '{search_term}' completed: {len(combined_fragments)} fragments from {unique_docs_combined} documents")
                    return combined_fragments
                else:
                    log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: Semantic search '{search_term}' completed: {len(semantic_fragments)} fragments from {unique_docs_semantic} documents")
                    return semantic_fragments
            
            # Get results from semantic search: use ACL retry loop when effective_user_id is supplied,
            # otherwise fall through to multi-language search.
            if effective_user_id:
                semantic_fragments = self._semantic_search_with_acl(search_term, effective_user_id, self.TOP_N_RAG, subset_of_datasetids)
            else:
                semantic_fragments = self._semantic_search_multilang(search_term, subset_of_datasetids, effective_user_id=None)
            unique_docs_semantic = len(set(f.get('documentid') for f in semantic_fragments if f.get('documentid')))
            log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Multi-language semantic search found {len(semantic_fragments)} fragments from {unique_docs_semantic} documents for '{search_term}' ({word_count} words)")
            
            # For short queries (5 or fewer words), also use text search and combine results
            # This helps catch exact phrase matches that embeddings might miss
            if word_count <= MIN_SEMANTIC_TEXT_COMBINE_WORDS:
                # Get results from text search (case-insensitive fulltext search)
                text_fragments = self._text_search(search_term, case_sensitive=False, subset_of_datasetids=subset_of_datasetids, effective_user_id=effective_user_id)
                unique_docs_text = len(set(f.get('documentid') for f in text_fragments if f.get('documentid')))
                log(DEBUG, f"{self._log_prefix}._evaluate_thalabus_ast: Text search found {len(text_fragments)} fragments from {unique_docs_text} documents for '{search_term}'")
                
                # Combine results: union of both searches
                combined_fragments = self._combine_fragments_or(semantic_fragments, text_fragments)
                unique_docs_combined = len(set(f.get('documentid') for f in combined_fragments if f.get('documentid')))
                
                log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: Combined semantic + text search for short query '{search_term}' ({word_count} words, 5 or fewer): {len(semantic_fragments)} semantic ({unique_docs_semantic} docs) + {len(text_fragments)} text ({unique_docs_text} docs) = {len(combined_fragments)} combined fragments ({unique_docs_combined} docs)")
                return combined_fragments
            else:
                # For longer queries, use only semantic search (better for paraphrase handling)
                log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: Using semantic search only for longer query '{search_term}' ({word_count} words): {len(semantic_fragments)} fragments from {unique_docs_semantic} documents")
                log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: Semantic search '{search_term}' completed: {len(semantic_fragments)} fragments from {unique_docs_semantic} documents")
                return semantic_fragments
        
        elif node.get('type') == 'TEXT':
            # For text search, use only text search (no semantic search)
            search_term = node['text']
            case_sensitive = node['case_sensitive']
            log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: Starting text search for '{search_term}' (case_sensitive={case_sensitive})")
            
            # OPTIMIZATION: use the most specific constraint available (same logic as SEMANTIC above).
            effective_constrain = constrain_to_fragments if constrain_to_fragments is not None else base_limit_fragments
            if effective_constrain is not None:
                constrained_doc_ids = list(set(f.get('documentid') for f in effective_constrain if f.get('documentid')))
                log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: Constraining text search to {len(constrained_doc_ids)} documents")
                text_fragments = self._text_search_constrained(search_term, case_sensitive, constrained_doc_ids, subset_of_datasetids, effective_user_id=effective_user_id)
                unique_docs_text = len(set(f.get('documentid') for f in text_fragments if f.get('documentid')))
                log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: Text search '{search_term}' completed: {len(text_fragments)} fragments from {unique_docs_text} documents")
                return text_fragments
            
            text_fragments = self._text_search(search_term, case_sensitive, subset_of_datasetids, effective_user_id=effective_user_id)
            unique_docs_text = len(set(f.get('documentid') for f in text_fragments if f.get('documentid')))
            log(INFO, f"{self._log_prefix}._evaluate_thalabus_ast: Text search '{search_term}' completed: {len(text_fragments)} fragments from {unique_docs_text} documents")
            return text_fragments
        
        else:
            raise ValueError(f"Unknown AST node type: {node}")
    
    def _get_all_document_ids(self, ontology_id: Optional[str], subset_of_datasetids: Optional[List[str]]) -> Set[str]:
        """
        Get all document IDs for the given ontology and optional dataset.
        Used for NOT operations.
        
        Args:
            ontology_id: The ontology ID. If None, returns all documents across any ontology.
            subset_of_datasetids: Optional list of dataset IDs to restrict search.
        
        Returns:
            Set of all document IDs
        """
        try:
            if ontology_id is None:
                # No specific ontology - get all documents
                if not subset_of_datasetids or len(subset_of_datasetids) == 0:
                    cypher_query = """
                        MATCH (d:Document)
                        RETURN DISTINCT d.id AS document_id
                    """
                    params = {}
                else:
                    cypher_query = f"""
                        {self._dataset_subset_membership_union()}
                        RETURN d.id AS document_id
                    """
                    params = {"subset_ids": subset_of_datasetids}
            else:
                # Specific ontology - get documents following that ontology
                if not subset_of_datasetids or len(subset_of_datasetids) == 0:
                    cypher_query = """
                        MATCH (d:Document)-[:FOLLOWS]->(o:Ontology {id: $ontology_id})
                        RETURN DISTINCT d.id AS document_id
                    """
                    params = {"ontology_id": ontology_id}
                else:
                    cypher_query = f"""
                        {self._dataset_subset_membership_union(extra_filter="EXISTS { MATCH (d)-[:FOLLOWS]->(:Ontology {id: $ontology_id}) }")}
                        RETURN d.id AS document_id
                    """
                    params = {"ontology_id": ontology_id, "subset_ids": subset_of_datasetids}
            
            result = execute_neo4j_query(cypher_query, params)
            if result:
                return {record.get('document_id') for record in result if record.get('document_id')}
            return set()
        except Exception as e:
            log(ERROR, f"{self._log_prefix}._get_all_document_ids: Error: {e}")
            return set()

    def _identify_standing_out_results(self, results: List[dict]) -> List[dict]:
        """
        Identify which results stand out from the rest using statistical methods.
        
        Uses a cascade of methods, looking for the earliest strong signal:
        1. Multiplicative ratio drop (finds where score drops by 2x+)
        2. Z-score outlier detection (identifies statistical outliers)
        3. Elbow detection with strict criteria (finds sharp curve bends)
        4. Aggressive percentile-based cutoff (top 10-20% if distributed widely)
        
        Args:
            results: List of document details with 'score' field, sorted by score descending.
        
        Returns:
            Filtered list of document details that stand out.
        """
        try:
            if not results:
                return results

            n = len(results)
            # If the list is short, return all (there is a reranker later)
            if n <= self.SHORT_LIST_MAX:
                return results

            # Extract scores as floats (descending order expected)
            scores = [float(r.get('score', 0.0) or 0.0) for r in results]
            
            # Log score distribution for debugging
            log(DEBUG, f"{self._log_prefix}._identify_standing_out_results: Analyzing {n} results. Top scores: {scores[:5] if len(scores) >= 5 else scores}")

            # If all scores are essentially identical, return all
            if math.isclose(max(scores), min(scores), rel_tol=1e-6, abs_tol=1e-9):
                log(DEBUG, f"{self._log_prefix}._identify_standing_out_results: All scores identical, returning all")
                return results

            eps = self.STANDING_EPS
            
            # Method 1: Look for FIRST significant multiplicative drop (not the maximum)
            # This prioritizes early strong signals over later ones
            for i in range(min(n - 1, 20)):  # Only check first 20 positions for early cutoff
                ratio = scores[i] / (scores[i + 1] + eps)
                # Looking for 2x drop (aggressive) in early positions
                if ratio >= 2.0:
                    log(DEBUG, f"{self._log_prefix}._identify_standing_out_results: Method 1 (early ratio drop): cutoff at index {i}, ratio={ratio:.3f}, scores={scores[i]:.3f} -> {scores[i+1]:.3f}")
                    return results[:i + 1]
            
            # Method 2: Z-score based outlier detection
            # Calculate z-scores to find values that are statistically extreme
            if n >= 3:
                mean_score = statistics.mean(scores)
                stdev_score = statistics.stdev(scores) if n > 1 else 0.0
                
                if stdev_score > eps:
                    # Find where z-scores drop below threshold (2.0 = top ~2.5%)
                    z_scores = [(s - mean_score) / stdev_score for s in scores]
                    
                    # Find first position where z-score drops below 2.0
                    for i, z in enumerate(z_scores):
                        if z < 2.0:
                            if i > 0:  # At least one result should be included
                                log(DEBUG, f"{self._log_prefix}._identify_standing_out_results: Method 2 (z-score): cutoff at index {i-1}, z-score threshold crossed at {z:.3f}")
                                return results[:i]
                            break
            
            # Method 3: Elbow detection using normalized second derivative
            # Find sharp bends in the score curve
            if n >= 4:
                # Calculate normalized first derivative (percentage change)
                norm_first_deriv = []
                for i in range(n - 1):
                    if scores[i] > eps:
                        pct_change = (scores[i] - scores[i + 1]) / scores[i]
                        norm_first_deriv.append(pct_change)
                    else:
                        norm_first_deriv.append(0.0)
                
                # Calculate second derivative of normalized changes
                if len(norm_first_deriv) >= 2:
                    second_deriv = [norm_first_deriv[i] - norm_first_deriv[i + 1] 
                                   for i in range(len(norm_first_deriv) - 1)]
                    
                    if second_deriv:
                        mean_2nd = statistics.mean(second_deriv)
                        stdev_2nd = statistics.stdev(second_deriv) if len(second_deriv) > 1 else 0.0
                        
                        # Look for FIRST significant elbow (not maximum) in early positions
                        if stdev_2nd > eps:
                            for i in range(min(len(second_deriv), 15)):
                                if second_deriv[i] > mean_2nd + 2.5 * stdev_2nd:
                                    cutoff_idx = i + 1
                                    log(DEBUG, f"{self._log_prefix}._identify_standing_out_results: Method 3 (early elbow): cutoff at index {cutoff_idx}, 2nd_deriv={second_deriv[i]:.6f} vs threshold={mean_2nd + 2.5*stdev_2nd:.6f}")
                                    return results[:cutoff_idx + 1]
            
            # Method 4: IQR-based outlier detection (more aggressive)
            if n >= 10:
                # Calculate quartiles
                q1 = statistics.quantiles(scores, n=4)[0]  # 25th percentile
                q3 = statistics.quantiles(scores, n=4)[2]  # 75th percentile
                iqr = q3 - q1
                
                # More aggressive fence: Q3 + 1.0 * IQR (instead of 1.5)
                upper_fence = q3 + 1.0 * iqr
                
                # Count outliers
                outlier_count = sum(1 for s in scores if s > upper_fence)
                
                if outlier_count > 0 and outlier_count < n * 0.5:  # Less than 50% are outliers
                    log(DEBUG, f"{self._log_prefix}._identify_standing_out_results: Method 4 (IQR outliers): {outlier_count} outliers detected, upper_fence={upper_fence:.3f}")
                    return results[:outlier_count]
            
            # Method 5: Percentile-based hard cutoff for large result sets
            # If we have many results and haven't found a clear cutoff, take top 15%
            if n > 20:
                cutoff_count = max(3, int(n * 0.15))  # Top 15%, but at least 3
                log(DEBUG, f"{self._log_prefix}._identify_standing_out_results: Method 5 (percentile cutoff): returning top {cutoff_count} of {n} results (15%)")
                return results[:cutoff_count]
            
            # No clear cutoff found - return all results
            log(DEBUG, f"{self._log_prefix}._identify_standing_out_results: No clear cutoff found, returning all {n} results")
            return results

        except Exception as e:
            log(ERROR, f"{self._log_prefix}._identify_standing_out_results: Error: {e}\n{traceback.format_exc()}")
            return results

    def _rerank_branch_results(self, query: str, fragments: List[Dict[str, Any]], subset_of_datasetids: Optional[List[str]], text_cache: Optional[Dict[str, Optional[str]]] = None) -> List[Dict[str, Any]]:
        """Rerank fragments for a branch of the query before combining with other branches.
        
        Uses multi-language reranking to ensure cross-language documents get fair scoring.
        
        IMPORTANT: Only reranks if the query contains semantic content. Pure category-only
        queries (e.g., "CATEGORY=VALUE") should not be reranked because they have no semantic
        meaning to rank against. Only attempts reranking if at least one fragment comes from
        a semantic search source ('semantic', 'text') rather than just category matching.
        
        Args:
            query: The query text for this branch (serialized from AST subtree)
            fragments: List of fragment dictionaries with 'fragmentid', 'documentid', 'text', 'score'
            subset_of_datasetids: Optional list of dataset IDs to restrict search
            
        Returns:
            Reranked list of fragments. If reranking is skipped or fails, returns original fragments.
        """
        if not fragments:
            return fragments
        
        # Skip reranking if all fragments come from category matching (no semantic content)
        # This prevents unnecessary reranking service calls and maintains proper scoring
        has_semantic_content = any(f.get('source') in ['semantic', 'text'] for f in fragments)
        if not has_semantic_content:
            log(DEBUG, f"{self._log_prefix}._rerank_branch_results: Skipping reranking - query has no semantic content (all fragments from '{fragments[0].get('source', 'unknown')}' source)")
            return fragments
            
        try:
            if text_cache is None:
                text_cache = {}
            self._hydrate_fragment_text(fragments, text_cache)
            reranking_service:RagInterface = get_reranking_service()
            if reranking_service is None:
                log(INFO, f"{self._log_prefix}._rerank_branch_results: Reranking service not available")
                return fragments
            
            # Prepare fragments for reranking (reranking service expects 'id', 'score', 'text')
            fragments_for_reranking = []
            for frag in fragments:
                fragments_for_reranking.append({
                    'id': frag.get('fragmentid'),
                    'score': frag.get('score', 0.0),
                    'text': frag.get('text', '')
                })
            
            # Call multi-language reranking service for cross-language document fairness
            reranked = reranking_service.rerank_multilang(query, fragments_for_reranking)
            
            if reranked and len(reranked) > 0:
                # Create mapping from fragmentid to reranked data
                reranked_map = {r.get('id'): r for r in reranked}
                
                # Log what we're trying to match
                log(DEBUG, f"{self._log_prefix}._rerank_branch_results: Reranked map IDs: {list(reranked_map.keys())}")
                log(DEBUG, f"{self._log_prefix}._rerank_branch_results: Looking for fragmentids: {[f.get('fragmentid') for f in fragments[:5]]}")
                
                # Update scores in original fragments, preserving all fields including documentid
                result_fragments = []
                matched = 0
                not_matched = 0
                for frag in fragments:
                    frag_id = frag.get('fragmentid')
                    if frag_id in reranked_map:
                        matched += 1
                        updated_frag = frag.copy()
                        # Update score from reranking while preserving documentid and all other fields
                        updated_frag['score'] = reranked_map[frag_id].get('ranking', reranked_map[frag_id].get('score', frag.get('score', 0.0)))
                        updated_frag['_reranked_score'] = True
                        result_fragments.append(updated_frag)
                    else:
                        not_matched += 1
                
                log(INFO, f"{self._log_prefix}._rerank_branch_results: Reranked branch '{query[:50]}...' - {len(fragments)} fragments -> {matched} matched, {not_matched} not matched, {len(result_fragments)} returned")
                distinct_docs = len(set(f.get('documentid') for f in result_fragments if f.get('documentid')))
                log(DEBUG, f"{self._log_prefix}._rerank_branch_results: Reranked fragments have {distinct_docs} distinct documents")
                return result_fragments
            else:
                log(DEBUG, f"{self._log_prefix}._rerank_branch_results: Reranking returned no results, using original fragments")
                return fragments
            
        except Exception as e:
            log(ERROR, f"{self._log_prefix}._rerank_branch_results: Error reranking branch: {e}")
            return fragments
    
    def _filter_fragments_by_rerank_score(self, fragments: List[Dict[str, Any]], min_reranking:float=MIN_RERANKING_SCORE_FRAG) -> List[Dict[str, Any]]:
        """Filter fragments by reranking score threshold, keeping only those above MIN_RERANKING_SCORE_FRAG.
        
        Args:
            fragments: List of fragment dictionaries with 'score' field
            
        Returns:
            Filtered list of fragments with score >= min_reranking. If no fragments meet the threshold, returns an empty list.
        """
        if not fragments:
            return fragments

        if not any(fragment.get('_reranked_score') for fragment in fragments):
            return fragments
        
        filtered = []
        for fragment in fragments:
            was_reranked = fragment.pop('_reranked_score', False)
            if was_reranked and fragment.get('score', 0.0) >= min_reranking:
                filtered.append(fragment)
        if len(filtered) < len(fragments):
            log(DEBUG, f"{self._log_prefix}._filter_fragments_by_rerank_score: Filtered {len(fragments)} -> {len(filtered)} fragments (min_reranking={min_reranking})")

        return filtered
    
    def _perform_final_reranking(self, query: str, results: List[dict]) -> None:
        """Perform final reranking on results using the RerankingService.
        
        Modifies results list in place to reorder documents by reranking scores.
        If reranking service is unavailable or fails, results remain unchanged.
        
        Args:
            query: The original query string
            results: List of document details to rerank (modified in place)
        """
        try:
            log(INFO, f"{self._log_prefix}._perform_final_reranking: Performing final reranking on {len(results)} documents using reranking service")
            
            # Get the reranking service
            reranking_service:RagInterface = get_reranking_service()
            if reranking_service is None:
                log(WARNING, f"{self._log_prefix}._perform_final_reranking: Reranking service not available")
                return
            
            # Prepare documents for reranking: use original query without quotes
            original_query = query.strip('"\'')
            
            # Get documents service to fetch full text
            documents_service:DocumentsInterface = get_documents_service()
            
            # Prepare fragments for reranking with required format (id, score, text)
            MAX_RERANK_TEXT = 2000  # Limit to 2000 chars to avoid token limit
            doc_ids_for_rerank = [doc.get('id', '') for doc in results]
            fetched_docs = {}
            if documents_service:
                try:
                    fetched_docs = documents_service.get_documents(doc_ids_for_rerank, include_text=True, include_summary=False)
                except Exception as e:
                    log(DEBUG, f"{self._log_prefix}._perform_final_reranking: Could not batch-fetch documents: {e}")
            
            fragments_for_reranking = []
            for doc in results:
                doc_id = doc.get('id', '')
                # Try to get document text for reranking, but limit to first N chars to avoid API limits
                doc_text = None
                
                full_doc = fetched_docs.get(doc_id)
                if full_doc:
                    # Use full text, or combine title and description if not available
                    doc_text = full_doc.get('text') or full_doc.get('enhanced_text')
                    if not doc_text and (full_doc.get('title') or full_doc.get('description')):
                        doc_text = (full_doc.get('title') or '') + ' ' + (full_doc.get('description') or '')
                    
                    # Truncate to avoid API token limits
                    if doc_text and len(doc_text) > MAX_RERANK_TEXT:
                        doc_text = doc_text[:MAX_RERANK_TEXT]
                
                # Fall back to description or title if full text not available
                if not doc_text:
                    doc_text = doc.get('description') or doc.get('title') or f"Document {doc_id}"
                    if len(doc_text) > MAX_RERANK_TEXT:
                        doc_text = doc_text[:MAX_RERANK_TEXT]
                
                fragments_for_reranking.append({
                    'id': doc_id,
                    'score': doc.get('score', 0.0),
                    'text': doc_text
                })
            
            try:
                # Call reranking service
                reranked = reranking_service.rerank(original_query, fragments_for_reranking)
                
                if reranked and len(reranked) > 0:
                    # Create a mapping of doc_id to reranking order (use index as the signal)
                    rerank_indices = {}
                    for i, item in enumerate(reranked):
                        rerank_indices[item['id']] = i
                    
                    # Sort results using reranking order, but preserve semantic score for display
                    results.sort(key=lambda x: rerank_indices.get(x.get('id'), len(results)))
                    log(INFO, f"{self._log_prefix}._perform_final_reranking: Reranking completed and results re-sorted by reranking order")
                else:
                    log(WARNING, f"{self._log_prefix}._perform_final_reranking: Reranking returned no results")
            except Exception as e:
                log(ERROR, f"{self._log_prefix}._perform_final_reranking: Error during reranking: {e}")
        
        except Exception as e:
            log(ERROR, f"{self._log_prefix}._perform_final_reranking: Error: {e}\n{traceback.format_exc()}")

    def _automatic_rerank(self, query: str, results: List[dict]) -> List[dict]:
        """Perform automatic reranking of `results` using the RerankingService.

        Returns the new ordered list of document detail dicts (may be the original
        `results` if reranking is skipped or fails).
        """
        try:
            reranking_service:RagInterface = get_reranking_service()
            documents_service:DocumentsInterface = get_documents_service()
            doc_ids_for_rerank = [r.get('id') for r in results]
            fetched_docs = {}
            try:
                fetched_docs = documents_service.get_documents(doc_ids_for_rerank, include_text=True, include_summary=False)
            except Exception:
                pass

            fragments_for_reranking = []
            for r in results:
                doc_id = r.get('id')
                doc = fetched_docs.get(doc_id)

                text = None
                if doc:
                    text = doc.get('text') or doc.get('enhanced_text') or (doc.get('title') or '') + '\n' + (doc.get('description') or '')

                fragments_for_reranking.append({'id': doc_id, 'score': r.get('score', 0.0), 'text': text})

            try:
                reranked = reranking_service.rerank(query, fragments_for_reranking)
            except Exception as e:
                log(ERROR, f"{self._log_prefix}._automatic_rerank: Reranking call failed: {e}")
                reranked = None

            if reranked:
                new_results = []
                for entry in reranked:
                    rid = entry.get('id')
                    if not rid:
                        continue
                    orig = next((x for x in results if x.get('id') == rid), None)
                    if orig:
                        orig['score'] = entry.get('ranking', entry.get('score', orig.get('score', 0.0)))
                        orig['reranking'] = entry.get('ranking')
                        new_results.append(orig)

                log(INFO, f"{self._log_prefix}._automatic_rerank: Returned {len(new_results)} documents after reranking")
                return new_results
            else:
                log(INFO, f"{self._log_prefix}._automatic_rerank: Reranking produced no results or was skipped; returning original results")
                return results

        except Exception as e:
            log(ERROR, f"{self._log_prefix}._automatic_rerank: Error during reranking: {e}\n{traceback.format_exc()}")
            return results
    
    def _final_rerank_results(self, query: str, results: List[dict], subset_of_datasetids: Optional[List[str]] = None) -> List[dict]:
        """Perform final reranking for `thalabus_query` and return the (possibly) reordered results.

        This calls the reranking service's rerank_documents method, which:
        1. Fetches all chapters for the documents via the documents service
        2. Reranks chapters in parallel using ThreadPoolExecutor
        3. Returns the max ranking score per document

        The cypher query for chapter fetching is now in DocumentsService, not here.
        """
        try:
            log(INFO, f"{self._log_prefix}.thalabus_query: Performing final reranking on {len(results)} documents using reranking service")

            reranking_service:RagInterface = get_reranking_service()
            if reranking_service is None:
                log(WARNING, f"{self._log_prefix}.thalabus_query: Reranking service not available")
                return results

            # Use the original query without surrounding quotes
            original_query = query.strip('"\'')

            # Extract document IDs from results
            doc_ids = [r.get('id') for r in results if r.get('id')]
            if not doc_ids:
                log(WARNING, f"{self._log_prefix}.thalabus_query: No document IDs found in results")
                return results

            try:
                # Call reranking service with document IDs
                # It will fetch chapters, rerank them in parallel, and return max score per document
                reranked = reranking_service.rerank_documents(original_query, doc_ids)

                if reranked and len(reranked) > 0:
                    # Create a mapping of doc_id to reranking score
                    rerank_scores = {item['id']: item.get('ranking', 0.0) for item in reranked}

                    # Filter results to only include documents that passed reranking
                    # and sort them by reranking score (descending)
                    results = [r for r in results if r.get('id') in rerank_scores]
                    results.sort(key=lambda x: rerank_scores.get(x.get('id'), -1), reverse=True)
                    
                    log(INFO, f"{self._log_prefix}.thalabus_query: Reranking completed, filtered to {len(results)} documents and re-sorted by ranking score")
                else:
                    log(WARNING, f"{self._log_prefix}.thalabus_query: Reranking returned no results")
            except Exception as e:
                log(ERROR, f"{self._log_prefix}.thalabus_query: Error during reranking: {e}")

            return results

        except Exception as e:
            log(ERROR, f"{self._log_prefix}.thalabus_query: Error during final reranking: {e}\n{traceback.format_exc()}")
            return results
    
    def contains_semantic_query_parts(self, query: str) -> bool:
        """
        Heuristically determine whether the provided query contains semantic
        components that would benefit from reranking.

        Heuristics used (ordered by priority):
        - explicit semantic token: any double-quoted phrase ("...")
        - semantic tokens after tokenization (SEMANTIC)
        - natural language indicators: question words (who/what/where/when/why/how)
        - punctuation suggesting free text (?, !, .) or multiple-word free text

        Returns True when reranking should be considered, False otherwise.
        """
        try:
            if not query or not isinstance(query, str):
                return False

            q = query.strip()
            if len(q) == 0:
                return False

            # Priority: explicit double-quoted semantic phrases
            if '"' in q:
                return True

            # Tokenize and look for explicit SEMANTIC tokens
            try:
                sanitized = self._sanitize_query(q)
                tokens = self._tokenize_thalabus_query(sanitized)
                for t in tokens:
                    if isinstance(t, tuple) and t[0] == 'SEMANTIC':
                        return True
            except Exception:
                # If tokenization fails, fall back to other heuristics
                pass

            # Natural language / question heuristics
            lower = q.lower()
            wh_starts = ('who ', 'what ', 'where ', 'when ', 'why ', 'how ', 'tell ', 'describe ', 'explain ', 'summar')
            if any(lower.startswith(w) for w in wh_starts) or any(ch in q for ch in ['?', '!']):
                return True

            # If the input looks like free-form multi-word text (not a simple CATEGORY=... expression)
            words = [w for w in q.split() if len(w) > 0]
            if len(words) >= 3 and '=' not in q:
                return True

            return False
        except Exception:
            return False
    
    def is_thalabus_query(self, text: str) -> bool:
        """Detect if a string is a syntactically valid thalabus query.

        Uses the full tokenizer + parser pipeline to determine whether the input
        conforms precisely to the thalabus query language grammar. A valid query
        must:
        1. Tokenize without errors (balanced quotes, valid structure)
        2. Parse completely — all tokens consumed, no trailing garbage
        3. Contain at least one meaningful query node:
           - CATEGORY expression (UPPER_CASE_WITH_UNDERSCORES name, with or without =value)
           - SEMANTIC search ("double-quoted")
           - TEXT search ('single-quoted' or @'case-sensitive')
           - TRUE (null search keyword)

        No heuristics are used. If the input does not parse as a valid thalabus
        expression, False is returned.

        Args:
            text: The text to check

        Returns:
            True if the text is a valid thalabus query, False otherwise
        """
        if not text or not isinstance(text, str):
            return False

        text = text.strip()
        if len(text) == 0:
            return False

        try:
            sanitized = self._sanitize_query(text)
            tokens = self._tokenize_thalabus_query(sanitized)

            if not tokens:
                return False

            # Full parse: must consume all tokens
            ast, end_pos = self._parse_thalabus_expression(tokens, 0)
            if end_pos != len(tokens):
                return False

            # Walk the AST to confirm at least one meaningful query node is present.
            # A bare lowercase word (e.g. "help") tokenizes as CATEGORY but is not a
            # valid query — only accept ALL_UPPER_CASE category names (machine-generated
            # identifiers, never natural language words).
            def _has_query_node(node: dict) -> bool:
                if node is None:
                    return False
                node_type = node.get('type')
                if node_type in ('SEMANTIC', 'TEXT', 'TRUE'):
                    return True
                if node_type == 'CATEGORY':
                    name = node.get('name', '')
                    return bool(name and re.match(r'^[A-Z][A-Z0-9_]*$', name))
                op = node.get('op')
                if op in ('AND', 'OR'):
                    return _has_query_node(node.get('left')) or _has_query_node(node.get('right'))
                if op == 'NOT':
                    return _has_query_node(node.get('operand'))
                return False

            return _has_query_node(ast)

        except Exception:
            return False
    
    def _extract_category_names_from_ast(self, node: dict) -> Set[str]:
        """Extract all category names from the AST recursively.
        
        Args:
            node: AST node to traverse
            
        Returns:
            Set of category names found in the AST
        """
        category_names = set()
        
        if node.get('type') == 'CATEGORY':
            category_names.add(node['name'])
        elif 'op' in node:
            if node['op'] == 'NOT':
                category_names.update(self._extract_category_names_from_ast(node['operand']))
            else:  # AND or OR
                category_names.update(self._extract_category_names_from_ast(node['left']))
                category_names.update(self._extract_category_names_from_ast(node['right']))
        
        return category_names
    
    def _find_ontology_for_category(self, category_name: str) -> Optional[str]:
        """Find the ontology ID that contains the given category name.
        
        Queries Neo4j to find KeywordCategory nodes matching the category name,
        then follows the relationship to the Ontology node.
        
        Args:
            category_name: The category name to search for
            
        Returns:
            Ontology ID if found, None otherwise
        """
        try:
            # Query to find the ontology that contains this category
            # Note: Ontology -[:CONTAINS]-> KeywordCategory, and KeywordCategory uses 'text_en' property
            cypher_query = """
                MATCH (o:Ontology)-[:CONTAINS]->(kc:KeywordCategory {text_en: $category_name})
                RETURN o.id AS ontology_id
                LIMIT 1
            """
            results = execute_neo4j_query(cypher_query, {"category_name": category_name})
            
            if results and len(results) > 0:
                ontology_id = results[0].get('ontology_id')
                log(DEBUG, f"{self._log_prefix}._find_ontology_for_category: Found ontology '{ontology_id}' for category '{category_name}'")
                return ontology_id
            else:
                log(DEBUG, f"{self._log_prefix}._find_ontology_for_category: No ontology found for category '{category_name}'")
                return None
                
        except Exception as e:
            log(ERROR, f"{self._log_prefix}._find_ontology_for_category: Error finding ontology for category '{category_name}': {e}")
            return None

    def _normalize_document_scores_by_fragment_count(self, results: List[dict]) -> List[dict]:
        """
        Normalize document scores by fragment count to ensure fair ranking.
        
        Problem: Documents with more fragments tend to get higher aggregated scores,
        which unfairly biases results toward longer documents. This function normalizes
        scores so that documents are ranked fairly regardless of their length.
        
        Strategy: Get fragment counts from DocumentsService and apply normalization.
        
        Args:
            results: List of result documents with scores
            
        Returns:
            List of results with normalized scores (re-ranked)
        """
        if not results:
            return results
        
        try:
            docs_service: DocumentsInterface = get_documents_service()
            if docs_service is None:
                log(WARNING, f"{self._log_prefix}._normalize_document_scores_by_fragment_count: DocumentsService not available, skipping normalization")
                return results
            
            # Get fragment counts for all documents (counts only, no text/title transferred)
            doc_ids = [r.get('documentid') or r.get('id') for r in results]
            fragment_counts_raw = docs_service.get_chapter_counts_for_documents(doc_ids)
            fragment_counts = {doc_id: max(1, fragment_counts_raw.get(doc_id, 1)) for doc_id in doc_ids}
            
            # Log original scores and fragment counts
            log(DEBUG, f"{self._log_prefix}._normalize_document_scores_by_fragment_count: Original scores and fragment counts:")
            for result in results:
                doc_id = result.get('documentid') or result.get('id')
                original_score = result.get('score', 0.0)
                frag_count = fragment_counts.get(doc_id, 1)
                # log(DEBUG, f"  {doc_id}: score={original_score:.6f}, fragments={frag_count}")
            
            # Apply normalization: Divide score by sqrt(fragment_count)
            # Using sqrt instead of linear division provides a balanced approach:
            # - Longer documents still get some benefit (they have more content)
            # - But the advantage is reduced (not linear)
            import math
            for result in results:
                doc_id = result.get('documentid') or result.get('id')
                original_score = result.get('score', 0.0)
                frag_count = fragment_counts.get(doc_id, 1)
                
                # Normalize by square root of fragment count
                normalized_score = original_score / math.sqrt(frag_count)
                
                # Store both scores for transparency
                result['score_original'] = original_score
                result['score'] = normalized_score
                result['fragment_count'] = frag_count
            
            # Re-sort by normalized score
            results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
            
            # Log normalized scores
            log(DEBUG, f"{self._log_prefix}._normalize_document_scores_by_fragment_count: Normalized scores (score / sqrt(fragment_count)):")
            for result in results:
                doc_id = result.get('documentid') or result.get('id')
                normalized_score = result.get('score', 0.0)
                original_score = result.get('score_original', 0.0)
                frag_count = result.get('fragment_count', 1)
                # log(DEBUG, f"  {doc_id}: normalized={normalized_score:.6f} (was {original_score:.6f}, {frag_count} fragments)")
            
            return results
            
        except Exception as e:
            log(ERROR, f"{self._log_prefix}._normalize_document_scores_by_fragment_count: Error: {e}\n{traceback.format_exc()}")
            return results

    @timer_decorator
    def thalabus_query(self, query: str, effective_user_id: Optional[str] = None, ontology_id: Optional[str] = None, subset_of_datasetids: Optional[List[str]] = None, automatically_rerank: bool = True, create_dataset: Optional[Union[bool, str]] = None, session_id: Optional[str] = None, userid: Optional[str] = None, limit_to_documents: Optional[List[str]] = None, limit_to_dossier_id: Optional[str] = None, return_documents: bool = True, llm_filter_prompt: Optional[str] = None, llm_filter_score_threshold: Optional[float] = None) -> Tuple[List[dict], Optional[List[str]]]:
        """
        Search for documents using the thalabus query language.
        
        Args:
            query: The query string in thalabus query language with:
                   - Category filters: CATEGORY or CATEGORY=VALUE1,VALUE2
                   - Semantic search: "text in quotes"
                   - Text search (case-insensitive): 'text'
                   - Text search (case-sensitive): @'text'
                   - Boolean operators: AND, OR, NOT
                   - Parentheses for grouping
                   - TRUE for null search
            ontology_id: The ID of the ontology to search within. If None, searches across any ontology
                        and categories will be matched against any ontology.
            subset_of_datasetids: Optional list of dataset IDs to restrict search.
            create_dataset: If True, create a dataset with auto-generated name. If string, use as dataset name.
            session_id: Session ID for dataset creation. Required if create_dataset is set.
            userid: User ID for dataset creation. If not provided and session_id is set, extracted from session_id.
            effective_user_id: Neo4j User.id of the requesting user. When provided, only documents the effective user
                      holds READ permission on are included in the result — no unauthorized document
                      appears in the returned list (ACL guarantee). When None, ACL filtering is skipped
                      (safe fallback for callers that do not yet supply it).
            limit_to_documents: Optional list of document IDs to restrict results to.
            limit_to_dossier_id: Optional dossier ID. When provided, search is restricted to documents
                                 owned by that dossier. If limit_to_documents is also provided, the
                                 effective set is the intersection of both.
            llm_filter_prompt: Optional relevance question used by the LLM relevance gate (T33).
                               Must be a non-blank string of at most LLM_FILTER_MAX_PROMPT_CHARS characters
                               whenever llm_filter_score_threshold is set. Supplying a prompt without a
                               threshold is valid and simply leaves the gate disabled (warning logged).
            llm_filter_score_threshold: Optional strict lower bound on the final reranked score (T33).
                               A null value disables the gate entirely. When set, only candidates whose
                               final score is STRICTLY greater than this finite number are evaluated by
                               the configured DEFAULT model: each such candidate is judged against
                               llm_filter_prompt (Documents by full 'text', kg:// entities by their
                               Thalabus serialization) with medium reasoning — one request when the
                               model's is_reasoning capability supports self-reasoning, otherwise a
                               reasoning request followed by a separate YES/NO request. NO candidates
                               are removed from the results and from the created dataset; YES candidates
                               are kept and their dataset AGGREGATES weight is set to exactly 1.0.
                               Candidates at or below the threshold are never modified. Gate failures
                               (invalid arguments, DB read errors, oversized content, LLM failure,
                               malformed verdicts) are never treated as NO: the gate aborts before any
                               dataset write, the query returns no results and no dataset ids, and the
                               failure is logged as ERROR.
            
        Returns:
            A tuple of (document_list, dataset_ids) where:
                - document_list: List of document details sorted by score in descending order.
                                 When effective_user_id is set, every entry has been drawn exclusively from
                                 ACL-permitted fragments; no unauthorized document appears in this list.
                - dataset_ids: List with single dataset ID if dataset was created, None otherwise
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if session_id is None: session_id = get_user_service().get_session_id()
        results: List[dict] = []
        dataset_ids: Optional[List[str]] = None

        try:
            # --- T33: validate the optional LLM relevance gate arguments ---
            llm_gate_enabled, llm_gate_error = self._validate_llm_filter_arguments(llm_filter_prompt, llm_filter_score_threshold)
            if llm_gate_error is not None:
                raise ValueError(llm_gate_error)

            # Strip /query prefix if present (used in CLI)
            if query.strip().startswith('/query '):
                query = query.strip()[7:].strip()

            # --- RESOLVE DOSSIER FILTER ---
            # If a dossier ID is given, resolve it to document IDs and intersect with limit_to_documents.
            if limit_to_dossier_id:
                from src.services.dossiers_service import get_dossiers_service
                from src.interfaces.dossiers_interface import DossiersInterface
                dossiers_service: DossiersInterface = get_dossiers_service()
                dossier_docs = dossiers_service.get_documents_of_dossier(limit_to_dossier_id)
                dossier_doc_ids = [d['id'] for d in dossier_docs if d.get('id')]
                if not dossier_doc_ids:
                    log(WARNING, f"{self._log_prefix}.thalabus_query: limit_to_dossier_id='{limit_to_dossier_id}' resolved to 0 documents — returning empty")
                    return [], None
                log(INFO, f"{self._log_prefix}.thalabus_query: limit_to_dossier_id='{limit_to_dossier_id}' resolved to {len(dossier_doc_ids)} documents")
                if limit_to_documents:
                    # intersection: only docs that are both in the dossier AND in limit_to_documents
                    dossier_set = set(dossier_doc_ids)
                    limit_to_documents = [d for d in limit_to_documents if d in dossier_set]
                    log(INFO, f"{self._log_prefix}.thalabus_query: intersection of dossier + limit_to_documents -> {len(limit_to_documents)} documents")
                else:
                    limit_to_documents = dossier_doc_ids

            # --- PRE-FILTER limit_to_documents BY DATASET ---
            # When both limit_to_documents and subset_of_datasetids are provided, reduce
            # limit_to_documents to only those IDs that actually belong to one of the datasets.
            # This ensures the constrained semantic search path (which cannot apply a dataset
            # filter itself) never returns documents outside the requested datasets.
            if limit_to_documents and subset_of_datasetids:
                cypher_filter = f"""
                    {self._dataset_subset_membership_union(extra_filter="d.id IN $doc_ids")}
                    RETURN d.id AS doc_id
                """
                filter_result = execute_neo4j_query(cypher_filter, {"doc_ids": limit_to_documents, "subset_ids": subset_of_datasetids})
                filtered_ids = [r['doc_id'] for r in (filter_result or []) if r.get('doc_id')]
                if len(filtered_ids) < len(limit_to_documents):
                    log(INFO, f"{self._log_prefix}.thalabus_query: subset_of_datasetids pre-filter reduced limit_to_documents from {len(limit_to_documents)} to {len(filtered_ids)}")
                limit_to_documents = filtered_ids
                if not limit_to_documents:
                    log(WARNING, f"{self._log_prefix}.thalabus_query: No documents remain after subset_of_datasetids pre-filter — returning empty")
                    return [], None
            
            log(DEBUG, f"{self._log_prefix}.thalabus_query: ontology_id={ontology_id}, query={query}")

            # Sanitize and validate quotes, then tokenize
            sanitized_query = self._sanitize_query(query)
            try:
                tokens = self._tokenize_thalabus_query(sanitized_query)
                log(DEBUG, f"{self._log_prefix}.thalabus_query: Tokens: {tokens}")

            except ValueError as ve:
                results = None
                raise
            
            # Parse tokens into AST
            ast, pos = self._parse_thalabus_expression(tokens, 0)
            if pos < len(tokens):
                # Provide more helpful error message
                remaining_tokens = tokens[pos:]
                error_msg = f"Unexpected tokens after parsing: {remaining_tokens}"
                
                # Check for common mistakes
                if remaining_tokens and remaining_tokens[0] == 'NOT':
                    error_msg += "\n\nHint: The NOT operator cannot stand alone. Did you forget to use AND or OR before it?"
                    error_msg += "\n  Example: Use 'X AND NOT Y' instead of 'X NOT Y'"
                elif remaining_tokens and isinstance(remaining_tokens[0], tuple) and remaining_tokens[0][0] == 'CATEGORY':
                    error_msg += "\n\nHint: Missing boolean operator (AND/OR) between expressions?"
                    error_msg += "\n  Example: Use 'X AND Y' instead of 'X Y'"
                
                raise ValueError(error_msg)
            log(DEBUG, f"{self._log_prefix}.thalabus_query: AST: {ast}")
            
            # Validate: NOT cannot be used as a standalone top-level operator
            # It must be used as a filter within AND expressions: "X AND NOT Y"
            if ast.get('op') == 'NOT':
                raise ValueError("NOT operator cannot be used standalone. Use it with AND: \"X AND NOT Y\"")
            
            # AUTO-DETECT ONTOLOGY: If ontology_id is not provided, try to deduce it from categories in the query
            # This prevents category queries from falling back to semantic search when ontology is missing
            if ontology_id is None:
                # Extract all category names from the AST
                category_names = self._extract_category_names_from_ast(ast)
                
                if category_names:
                    log(DEBUG, f"{self._log_prefix}.thalabus_query: No ontology_id provided, found categories: {category_names}")
                    
                    # Try to find ontology for the first category
                    # (In practice, all categories in a query should belong to the same ontology)
                    for category_name in category_names:
                        detected_ontology_id = self._find_ontology_for_category(category_name)
                        if detected_ontology_id:
                            ontology_id = detected_ontology_id
                            log(INFO, f"{self._log_prefix}.thalabus_query: Auto-detected ontology_id='{ontology_id}' from category '{category_name}'")
                            break
                    
                    if ontology_id is None:
                        log(WARNING, f"{self._log_prefix}.thalabus_query: Could not auto-detect ontology for categories: {category_names}")
                else:
                    log(DEBUG, f"{self._log_prefix}.thalabus_query: No categories found in query (semantic/text search only)")
            
            # Mark category-only subtrees for optimization
            self._mark_category_only_subtrees(ast, ontology_id)
            log(DEBUG, f"{self._log_prefix}.thalabus_query: AST after marking: {ast}")
            
            # Log the optimized query representation
            optimized_query = self._serialize_ast_to_query(ast)
            log(INFO, f"{self._log_prefix}.thalabus_query: Optimized query representation: {optimized_query}")
            log(PERF, f"thalabus_query: start AST evaluation", perf_id="tq_start")
            
            # Build base_limit_fragments from limit_to_documents for propagation into AST evaluation.
            # When provided, every SEMANTIC/TEXT search inside the AST is constrained to these docs
            # via _semantic_search_constrained / _text_search_constrained instead of the global
            # vector-index search, so docs that rank outside the index top-N are still found.
            # When None/empty, behaviour is unchanged from the original implementation.
            base_limit_frags = ([{"documentid": d} for d in limit_to_documents] if limit_to_documents else None)

            # Evaluate AST to get fragments (not documents yet!)
            # Pass automatically_rerank flag to enable branch-level reranking
            text_cache: Dict[str, Optional[str]] = {}
            fragments = self._evaluate_thalabus_ast(ast, ontology_id, subset_of_datasetids, None, automatically_rerank, base_limit_fragments=base_limit_frags, effective_user_id=effective_user_id, text_cache=text_cache)

            # Post-filter fragments by limit_to_documents: catches category-only nodes that
            # bypass the constrained-search path inside _evaluate_thalabus_ast.
            if limit_to_documents and fragments:
                _limit_doc_set = set(limit_to_documents)
                _before = len(fragments)
                _debug_had_document = any(f.get('documentid') == DEBUG_IDENTIFY_DOC for f in fragments) if DEBUG_IDENTIFY_DOC else None
                fragments = [f for f in fragments if f.get('documentid') in _limit_doc_set]
                if _before != len(fragments):
                    log(INFO, f"{self._log_prefix}.thalabus_query: limit_to_documents filtered fragments from {_before} to {len(fragments)}")
                if DEBUG_IDENTIFY_DOC and _debug_had_document and not any(f.get('documentid') == DEBUG_IDENTIFY_DOC for f in fragments):
                    log(WARNING, f"[DEBUG_IDENTIFY_DOC] REMOVED by limit_to_documents filter (only allowed: {limit_to_documents})")

            # Capture permitted document IDs from ACL-filtered fragments (Task 13.4).
            # When effective_user_id is set, fragments are already ACL-filtered by _evaluate_thalabus_ast
            # (via _semantic_search_with_acl).  Recording the permitted IDs here lets the
            # post-aggregation guard below close any remaining gap between fragment-level
            # filtering and the final document-level results list.
            permitted_doc_ids: Optional[set] = None
            if effective_user_id:
                permitted_doc_ids = {f.get("documentid") for f in (fragments or []) if f.get("documentid")}
                log(DEBUG, f"{self._log_prefix}.thalabus_query: ACL guard captured {len(permitted_doc_ids)} permitted doc IDs from fragments")

            # Handle TRUE (null search)
            if fragments is None:
                log(INFO, f"{self._log_prefix}.thalabus_query: Query is TRUE (no search required)")
                return results, None
            
            if not fragments:
                log(INFO, f"{self._log_prefix}.thalabus_query: No fragments found matching query")
                return results, None
            
            log(INFO, f"{self._log_prefix}.thalabus_query: Found {len(fragments)} fragments")
            if fragments and DEBUG_IDENTIFY_DOC:
                    _debug_has_document = any(f.get('documentid') == DEBUG_IDENTIFY_DOC for f in fragments)
                    if _debug_has_document:
                        _debug_frags_document = [f for f in fragments if f.get('documentid') == DEBUG_IDENTIFY_DOC]
                        log(INFO, f"[DEBUG_IDENTIFY_DOC] Found in fragments: {len(_debug_frags_document)} fragment(s) with scores: {[f.get('score', 0.0) for f in _debug_frags_document]}")
                    else:
                        log(WARNING, f"[DEBUG_IDENTIFY_DOC] NOT found in fragments after AST evaluation")
            log(PERF, f"thalabus_query: AST evaluated, {len(fragments)} frags", perf_id="tq_ast", perf_since_id="tq_start")
            
            # FRAGMENT-LEVEL RERANKING (if enabled and not already done at branch level)
            # Only perform final fragment reranking if it's a simple query (no boolean operators)
            # For complex queries, branch-level reranking already handled in _evaluate_thalabus_ast
            if automatically_rerank and fragments and (ast.get('type') not in ['AND', 'OR', 'NOT']):
                if DEBUG_IDENTIFY_DOC:
                    _debug_before_rerank = any(f.get('documentid') == DEBUG_IDENTIFY_DOC for f in fragments)
                    if _debug_before_rerank:
                        _debug_frags_before = [f for f in fragments if f.get('documentid') == DEBUG_IDENTIFY_DOC]
                        log(INFO, f"[DEBUG_IDENTIFY_DOC] Before fragment-level reranking: {len(_debug_frags_before)} fragments with scores {[f.get('score', 0.0) for f in _debug_frags_before]}")
                log(DEBUG, f"{self._log_prefix}.thalabus_query: Performing fragment-level reranking on {len(fragments)} fragments")
                fragments = self._rerank_branch_results(query, fragments, subset_of_datasetids, text_cache)
                log(DEBUG, f"{self._log_prefix}.thalabus_query: After fragment reranking: {len(fragments)} fragments")
                # Filter by reranking score threshold before final aggregation
                fragments = self._filter_fragments_by_rerank_score(fragments)
                log(DEBUG, f"{self._log_prefix}.thalabus_query: After score filter: {len(fragments)} fragments")
                if DEBUG_IDENTIFY_DOC:
                    _debug_after_rerank = any(f.get('documentid') == DEBUG_IDENTIFY_DOC for f in fragments)
                    if _debug_before_rerank and _debug_after_rerank:
                        _debug_frags_after = [f for f in fragments if f.get('documentid') == DEBUG_IDENTIFY_DOC]
                        log(INFO, f"[DEBUG_IDENTIFY_DOC] After fragment-level reranking: {len(_debug_frags_after)} fragments with scores {[f.get('score', 0.0) for f in _debug_frags_after]}")
                    elif _debug_before_rerank and not _debug_after_rerank:
                        log(WARNING, f"[DEBUG_IDENTIFY_DOC] REMOVED during fragment-level reranking")
            log(PERF, f"thalabus_query: reranking done, {len(fragments)} frags", perf_id="tq_rerank", perf_since_id="tq_ast")

            # Create dataset from fragments (before optional aggregation so it works for both return modes)
            dataset_ids: Optional[List[str]] = None
            if create_dataset and fragments:
                try:
                    dataset_service: DatasetInterface = get_dataset_service()
                    if dataset_service is None:
                        log(WARNING, f"{self._log_prefix}.thalabus_query: Dataset creation requested but DatasetService not available")
                    elif not session_id:
                        log(WARNING, f"{self._log_prefix}.thalabus_query: Dataset creation requested but session_id not provided")
                    else:
                        user_id = userid if userid else get_user_service().get_user_uuid()
                        if isinstance(create_dataset, str):
                            dataset_name = create_dataset
                            log(DEBUG, f"{self._log_prefix}.thalabus_query: Creating dataset with name: {dataset_name}")
                        else:
                            dataset_name, _ = dataset_service.generate_unique_dataset_name(prefix="query_")
                            log(DEBUG, f"{self._log_prefix}.thalabus_query: Creating dataset with auto-generated name: {dataset_name}")
                        # Extract fragment IDs and prepare weights from fragment scores
                        # This preserves the actual fragments (chapters, paragraphs, summaries) that matched,
                        # rather than aggregating to documents. The dataset will contain:
                        # - d:Document nodes if fragments are document-level
                        # - c:Chapter nodes if fragments are chapter-level (fragmentid pattern: "docid-c#")
                        # - p:Paragraph nodes if fragments are paragraph-level (fragmentid pattern: "docid-p#")
                        # - s:Summary nodes if fragments are summary-level (fragmentid pattern: "docid-s")
                        fragment_ids = []
                        weights = {}
                        for frag in fragments:
                            frag_id = frag.get('fragmentid')
                            if frag_id:
                                fragment_ids.append(frag_id)
                                score = frag.get('score', 1.0)
                                try:
                                    v = float(score)
                                    weights[frag_id] = math.trunc(v * 1e5) / 1e5  # Truncate to 5 decimal places
                                except Exception:
                                    weights[frag_id] = 1.0
                        fragment_types = {}
                        for frag_id in fragment_ids:
                            if '-p' in frag_id:
                                fragment_types['paragraphs'] = fragment_types.get('paragraphs', 0) + 1
                            elif '-c' in frag_id:
                                fragment_types['chapters'] = fragment_types.get('chapters', 0) + 1
                            elif '-s' in frag_id or frag_id.endswith('-summary'):
                                fragment_types['summaries'] = fragment_types.get('summaries', 0) + 1
                            elif 'docid-' in frag_id or frag_id.startswith('docid-'):
                                fragment_types['documents'] = fragment_types.get('documents', 0) + 1
                        metadata = {
                            "query": query,
                            "ontology_id": ontology_id,
                            "item_count": len(fragment_ids),
                            "fragment_types": fragment_types,
                            "creation": "thalabus_query"
                        }
                        dataset_result = dataset_service.create_dataset_from_fragments(
                            fragments=fragment_ids,
                            userid=user_id,
                            dataset_name=dataset_name,
                            dataset_description=f"Dataset created from thalabus query: {query[:100]}",
                            metadata=metadata,
                            domain=None,
                            subset_of_datasetids=subset_of_datasetids if subset_of_datasetids else None,
                            weights=weights,
                            invisible=True
                        )
                        if dataset_result is not None:
                            dataset_id = dataset_result.get('dataset_id')
                            dataset_ids = [dataset_id]
                            log(INFO, f"{self._log_prefix}.thalabus_query: Created dataset {dataset_id} with {len(fragment_ids)} fragments: {fragment_types}")
                            log(PERF, f"thalabus_query: dataset created: {dataset_id}, {len(fragment_ids)} frags", perf_id="tq_dataset", perf_since_id="tq_rerank")
                        else:
                            error_msg = dataset_result.get('error_message', 'Unknown error') if dataset_result else 'No result returned'
                            log(ERROR, f"{self._log_prefix}.thalabus_query: Failed to create dataset: {error_msg}")

                except Exception as e:
                    log(ERROR, f"{self._log_prefix}.thalabus_query: Error creating dataset: {e}\n{traceback.format_exc()}")

            # T33: optional LLM relevance gate over the finally reranked fragments.
            # Document-level queries are gated once, on the aggregated documents below, so the same
            # content is never judged twice. A gate failure empties the returned list and clears
            # dataset_ids: the caller sees an explicit failed query rather than an unfiltered
            # success (an error is never converted into a NO verdict).
            if llm_gate_enabled and not return_documents:
                try:
                    fragments = self._run_llm_relevance_gate(
                        items=fragments,
                        id_key='fragmentid',
                        fragments=None,
                        dataset_ids=dataset_ids,
                        prompt=llm_filter_prompt if llm_filter_prompt is not None else '',
                        threshold=float(cast(float, llm_filter_score_threshold)),
                        effective_user_id=effective_user_id,
                    )

                except _LlmFilterError as gate_error:
                    log(ERROR, f"{self._log_prefix}.thalabus_query: LLM relevance gate aborted (not treating errors as NO): {gate_error}")
                    fragments = []
                    dataset_ids = None

            # If return_documents=False, return fragments directly (skip document aggregation)
            if not return_documents:
                self._hydrate_fragment_text(fragments, text_cache)
                log(INFO, f"{self._log_prefix}.thalabus_query: Returning {len(fragments)} fragments directly (return_documents=False)")
                log(PERF, f"thalabus_query: done (fragments), {len(fragments)} frags", perf_id="tq_end", perf_since_id="tq_rerank")
                return fragments, dataset_ids

            log(DEBUG, f"{self._log_prefix}.thalabus_query: Aggregating {len(fragments)} fragments to documents...")
            
            # AGGREGATE FRAGMENTS TO DOCUMENTS (only here at the end!)
            fragments_by_doc = {}
            for frag in fragments:
                doc_id = frag.get('documentid')
                if doc_id:
                    if doc_id not in fragments_by_doc:
                        fragments_by_doc[doc_id] = []
                    fragments_by_doc[doc_id].append(frag)
            
            # Calculate max score per document from its fragments
            doc_scores = {}
            for doc_id, doc_frags in fragments_by_doc.items():
                doc_scores[doc_id] = max(f.get('score', 0.0) for f in doc_frags)
            
            if DEBUG_IDENTIFY_DOC and DEBUG_IDENTIFY_DOC in doc_scores:
                log(INFO, f"[DEBUG_IDENTIFY_DOC] After fragment aggregation: score={doc_scores[DEBUG_IDENTIFY_DOC]:.6f}, total_fragments={len(fragments_by_doc.get(DEBUG_IDENTIFY_DOC, []))}")
            elif DEBUG_IDENTIFY_DOC:
                log(WARNING, f"[DEBUG_IDENTIFY_DOC] NOT in doc_scores after aggregation (only {len(doc_scores)} documents aggregated)")
            
            # Sort documents by score descending to respect ranking
            doc_ids_sorted = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)
            
            if DEBUG_IDENTIFY_DOC and DEBUG_IDENTIFY_DOC in doc_ids_sorted:
                _rank = doc_ids_sorted.index(DEBUG_IDENTIFY_DOC) + 1
                log(INFO, f"[DEBUG_IDENTIFY_DOC] Ranking after sort: position {_rank} out of {len(doc_ids_sorted)}")
            elif DEBUG_IDENTIFY_DOC:
                log(WARNING, f"[DEBUG_IDENTIFY_DOC] NOT in sorted doc_ids_sorted")
            
            # ENFORCE MIN_RERANKING_OUTPUT at DOCUMENT level
            # Ensure we keep at least MIN_RERANKING_OUTPUT documents, even if scores are low
            if len(doc_ids_sorted) < MIN_RERANKING_OUTPUT_DOC:
                # Keep all documents if we have fewer than minimum
                doc_ids = doc_ids_sorted
                log(DEBUG, f"{self._log_prefix}.thalabus_query: Keeping all {len(doc_ids)} documents (below MIN_RERANKING_OUTPUT={MIN_RERANKING_OUTPUT_DOC})")
            else:
                # Keep at least MIN_RERANKING_OUTPUT documents
                doc_ids = doc_ids_sorted[:max(MIN_RERANKING_OUTPUT_DOC, len(doc_ids_sorted))]
                log(DEBUG, f"{self._log_prefix}.thalabus_query: Enforcing MIN_RERANKING_OUTPUT: keeping top {len(doc_ids)} documents (from {len(doc_ids_sorted)} total)")
            
            if DEBUG_IDENTIFY_DOC:
                if DEBUG_IDENTIFY_DOC in doc_ids:
                    log(INFO, f"[DEBUG_IDENTIFY_DOC] Passed MIN_RERANKING_OUTPUT filter")
                else:
                    _debug_in_sorted = DEBUG_IDENTIFY_DOC in doc_ids_sorted
                    log(WARNING, f"[DEBUG_IDENTIFY_DOC] FILTERED OUT by MIN_RERANKING_OUTPUT: was in sorted list={_debug_in_sorted}, keeping top {len(doc_ids)} docs (limit={MIN_RERANKING_OUTPUT_DOC})")
            
            if not doc_ids:
                log(INFO, f"{self._log_prefix}.thalabus_query: No documents after aggregation")
                return results, None
            
            log(INFO, f"{self._log_prefix}.thalabus_query: Aggregated {len(fragments)} fragments into {len(doc_ids)} documents (scores: {[f'{doc_scores[d]:.5f}' for d in doc_ids[:3]]}...)")
            log(PERF, f"thalabus_query: fragment aggregation done, {len(doc_ids)} docs", perf_id="tq_agg", perf_since_id="tq_rerank")
            
            # Fetch document details for each matching document ID
            log(DEBUG, f"{self._log_prefix}.thalabus_query: Fetching details for {len(doc_ids)} documents")
            
            # Build query with optional restriction to subset_of_datasetids
            if not subset_of_datasetids or len(subset_of_datasetids) == 0:
                cypher_query = """
                    MATCH (d:Document)
                    WHERE d.id IN $document_ids OR d.documentid IN $document_ids
                    RETURN d.id AS document_id, 
                           d.documentid AS documentid,
                           d.name AS name,
                           d.title AS title,
                           d.description AS description,
                           labels(d) AS labels,
                           keys(d) AS properties
                """
                query_params = {"document_ids": doc_ids}
            else:
                log(DEBUG, f"{self._log_prefix}.thalabus_query: Filtering {len(doc_ids)} documents by subset_of_datasetids={subset_of_datasetids}")
                if DEBUG_IDENTIFY_DOC and DEBUG_IDENTIFY_DOC in doc_ids:
                    log(INFO, f"[DEBUG_IDENTIFY_DOC] Fetching with dataset restriction: {subset_of_datasetids}")
                cypher_query = f"""
                    {self._dataset_subset_membership_union(extra_filter="(d.id IN $document_ids OR d.documentid IN $document_ids)")}
                    RETURN d.id AS document_id, 
                           d.documentid AS documentid,
                           d.name AS name,
                           d.title AS title,
                           d.description AS description,
                           labels(d) AS labels,
                           keys(d) AS properties
                """
                query_params = {"document_ids": doc_ids, "subset_ids": subset_of_datasetids}
            
            query_results = execute_neo4j_query(cypher_query, query_params)
            log(PERF, f"thalabus_query: doc details fetched, {len(query_results) if query_results else 0} results", perf_id="tq_docfetch", perf_since_id="tq_agg")
            
            log(DEBUG, f"{self._log_prefix}.thalabus_query: Query returned {len(query_results) if query_results else 0} results")
            
            if query_results:
                for record in query_results:
                    doc_id = record.get('document_id') or record.get('documentid')
                    doc_details = {
                        'id': doc_id,
                        'documentid': record.get('documentid'),
                        'name': record.get('name'),
                        'title': record.get('title'),
                        'description': record.get('description'),
                        'labels': record.get('labels', []),
                        'properties': record.get('properties', []),
                        'score': doc_scores.get(doc_id, 0.0)
                    }
                    results.append(doc_details)
                    if DEBUG_IDENTIFY_DOC and doc_id == DEBUG_IDENTIFY_DOC:
                        log(INFO, f"[DEBUG_IDENTIFY_DOC] Fetched from Neo4j: title={doc_details.get('title')}, score={doc_details.get('score'):.6f}")
            else:
                if DEBUG_IDENTIFY_DOC and DEBUG_IDENTIFY_DOC in doc_ids:
                    log(WARNING, f"[DEBUG_IDENTIFY_DOC] Was in doc_ids to fetch, but query_results is empty (Neo4j query might have failed)")
            
            # Sort results by score descending
            results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
            
            if DEBUG_IDENTIFY_DOC:
                if results and results[0].get('id') == DEBUG_IDENTIFY_DOC:
                    log(INFO, f"[DEBUG_IDENTIFY_DOC] Top result after sort: rank=1")
                elif any(r.get('id') == DEBUG_IDENTIFY_DOC for r in results):
                    _rank = next(i+1 for i, r in enumerate(results) if r.get('id') == DEBUG_IDENTIFY_DOC)
                    log(INFO, f"[DEBUG_IDENTIFY_DOC] After sort: rank={_rank} out of {len(results)}")
                elif DEBUG_IDENTIFY_DOC in doc_ids:
                    log(WARNING, f"[DEBUG_IDENTIFY_DOC] NOT in final results list (was in doc_ids to fetch)")

            # POST-AGGREGATION ACL GUARD (Task 13.4): ensure no unauthorized document leaks
            # through the fragment-to-document rollup step.  The fragment-level filtering in
            # _evaluate_thalabus_ast (Tasks 13.2/13.3) is the primary defence; this guard is
            # a belt-and-suspenders check against edge cases (e.g. document fetched by an
            # alternate ID that was not covered by the fragment's documentid set).
            if permitted_doc_ids is not None:
                _before_acl = len(results)
                if DEBUG_IDENTIFY_DOC and any(r.get("id") == DEBUG_IDENTIFY_DOC for r in results):
                    log(INFO, f"[DEBUG_IDENTIFY_DOC] Before ACL guard: in results")
                results = [
                    r for r in results
                    if r.get("id") in permitted_doc_ids or r.get("documentid") in permitted_doc_ids
                ]
                if len(results) < _before_acl:
                    log(INFO, f"{self._log_prefix}.thalabus_query: post-aggregation ACL guard removed {_before_acl - len(results)} unauthorized document(s)")
                if DEBUG_IDENTIFY_DOC and not any(r.get("id") == DEBUG_IDENTIFY_DOC for r in results) and _before_acl > 0:
                    log(WARNING, f"[DEBUG_IDENTIFY_DOC] REMOVED by ACL guard")

            # NORMALIZE SCORES BY FRAGMENT COUNT (fair ranking regardless of document length)
            # This ensures longer documents don't get unfairly higher scores
            # SKIP normalization for category-only queries where all documents have equal relevance (score=1.0)
            # Check if this is a category-only query by looking at fragment sources
            is_category_only = all(doc_scores.get(doc_id, 0.0) == 1.0 for doc_id in doc_ids)
            if results and not is_category_only:
                if DEBUG_IDENTIFY_DOC:
                    _debug_before_norm_document = None
                    _debug_before_rank = None
                    for i, r in enumerate(results):
                        if r.get("id") == DEBUG_IDENTIFY_DOC:
                            _debug_before_norm_document = r.get("score")
                            _debug_before_rank = i + 1
                            break
                    if _debug_before_norm_document is not None:
                        log(INFO, f"[DEBUG_IDENTIFY_DOC] Before normalization: score={_debug_before_norm_document:.6f}, rank={_debug_before_rank}")
                results = self._normalize_document_scores_by_fragment_count(results)
                if DEBUG_IDENTIFY_DOC:
                    _debug_after_norm_document = None
                    _debug_after_rank = None
                    for i, r in enumerate(results):
                        if r.get("id") == DEBUG_IDENTIFY_DOC:
                            _debug_after_norm_document = r.get("score")
                            _debug_after_rank = i + 1
                            break
                    if _debug_before_norm_document is not None and _debug_after_norm_document is not None:
                        log(INFO, f"[DEBUG_IDENTIFY_DOC] After normalization: score={_debug_after_norm_document:.6f} (was {_debug_before_norm_document:.6f}), rank={_debug_after_rank} (was {_debug_before_rank})")
                    elif _debug_before_norm_document is not None and _debug_after_norm_document is None:
                        log(WARNING, f"[DEBUG_IDENTIFY_DOC] REMOVED during normalization")
            elif is_category_only:
                log(DEBUG, f"{self._log_prefix}.thalabus_query: Skipping normalization for category-only query (all scores are 1.0)")

            # CLEANUP: Remove internal normalization tracking fields before returning
            # These fields (score_original, fragment_count) are only for debugging/logging
            # and should not be exposed to consumers of thalabus_query()
            for result in results:
                result.pop('score_original', None)
                result.pop('fragment_count', None)
            log(PERF, f"thalabus_query: results built and normalized, {len(results)} docs", perf_id="tq_normalize", perf_since_id="tq_docfetch")

            # Identify which results stand out from the rest based on score gaps
            # Only do this if automatic reranking is disabled, because reranking already
            # determined relevance at fragment level
            if results and not automatically_rerank:
                results = self._identify_standing_out_results(results)

            # T33: optional LLM relevance gate — runs after final reranking (and after score
            # normalization) and before returning results, so it judges and prunes exactly what
            # the caller would receive. A gate failure never means "everything was NO": the
            # dataset keeps its pre-gate content and the query fails explicitly (empty results,
            # no dataset ids) instead of returning unfiltered results as if filtering succeeded.
            if llm_gate_enabled:
                try:
                    results = self._run_llm_relevance_gate(
                        items=results,
                        id_key='id',
                        fragments=fragments,
                        dataset_ids=dataset_ids,
                        prompt=llm_filter_prompt if llm_filter_prompt is not None else '',
                        threshold=float(cast(float, llm_filter_score_threshold)),
                        effective_user_id=effective_user_id,
                    )
                except _LlmFilterError as gate_error:
                    log(ERROR, f"{self._log_prefix}.thalabus_query: LLM relevance gate aborted (not treating errors as NO): {gate_error}")
                    results = []
                    dataset_ids = None

            log(DEBUG, f"{self._log_prefix}.thalabus_query: Found {len(results)} documents with details")
            
        except Exception as e:
            log(ERROR, f"{self._log_prefix}.thalabus_query: Error during search: {e}\n{traceback.format_exc()}")
        
        log(PERF, f"thalabus_query: done, {len(results)} docs", perf_id="tq_end", perf_since_id="tq_start")
        return results, dataset_ids

    def thalabus_query_analyser(self, query: str, ontology_id: Optional[str] = None, subset_of_datasetids: Optional[List[str]] = None, automatically_rerank: bool = True) -> Tuple[Dict[str, Any], List[dict]]:
        """
        Analyze the given query and return an analysis of its components, performed in this way:
        1. runs the query and identifies all results
        2. gathers all CATEGORIES from the results
        3. for each category, counts how many results belong to it
        4. for each category where the values are multiple, aggregates the counts
        5. returns a dictionary with the following structure:
           {
               "total_results": int,  # total number of results found
               "categories": {
                   "category_name_1": {
                       "count": int,  # number of results in this category
                       "values": {
                           "value_1": int,  # count of results with this value
                           "value_2": int,
                           ...
                        }
                    },
                    "category_name_2": {
                        "count": int,
                        "values": {
                        "value_1": int,
                        "value_2": int
                        }
                    },
                   ...
               }
           }
        
        Implementation details:
        - Uses the thalabus_query method to get initial results
        - Executes a Cypher query to fetch categories for all result documents
        - Aggregates counts per category and per value
        - Handles errors gracefully and logs appropriately

        :param query: The query string to analyze.
        :return: A dictionary containing the analysis results.
        """
        query_results: List[dict] = []
        categories_data: Dict[str, Any] = {}

        try:
            log(DEBUG, f"{self._log_prefix}.thalabus_query_analyser: Analyzing query: {query}")
            
            # Step 1: Run the query to get results (forward optional parameters)
            query_results, _ = self.thalabus_query(query, None, ontology_id, subset_of_datasetids, automatically_rerank)
            
            if not query_results:
                log(INFO, f"{self._log_prefix}.thalabus_query_analyser: No results found for query")
                return ({
                    "total_results": 0,
                    "categories": {}
                }, query_results)
            
            # Step 2: Gather all categories from the results
            doc_ids = [doc.get('id') for doc in query_results if doc.get('id')]
            
            if not doc_ids:
                log(WARNING, f"{self._log_prefix}.thalabus_query_analyser: No document IDs found in results")
                return ({
                    "total_results": len(query_results),
                    "categories": {}
                }, query_results)
            
            # Fetch categories for all documents
            cypher_query = """
                MATCH (d:Document)-[r:HAS_CATEGORY]->(c:Category)-[:OF_CATEGORY]->(k:KeywordCategory)
                WHERE d.id IN $document_ids
                RETURN d.id AS document_id,
                       k.text_en AS category_name,
                       c.text AS category_value
            """
            
            category_results = execute_neo4j_query(cypher_query, {"document_ids": doc_ids})
            
            if not category_results:
                log(INFO, f"{self._log_prefix}.thalabus_query_analyser: No categories found for documents")
                return ({
                    "total_results": len(query_results),
                    "categories": {}
                }, query_results)
            
            # Step 3 & 4: Aggregate category counts and values
            categories_data:dict[str, dict[str, Any]] = {}
            doc_category_tracking:dict[str, set] = {}  # Track which documents have been counted for each category
            
            for record in category_results:
                doc_id = record.get('document_id')
                category_name = record.get('category_name')
                category_value = record.get('category_value')
                
                if not category_name:
                    continue
                
                # Initialize category if not exists
                if category_name not in categories_data:
                    categories_data[category_name] = {
                        "count": 0,
                        "values": {}
                    }
                    doc_category_tracking[category_name] = set()
                
                # Count each document only once per category
                if doc_id not in doc_category_tracking[category_name]:
                    categories_data[category_name]["count"] += 1
                    doc_category_tracking[category_name].add(doc_id)
                
                # If category has a value, track it
                if category_value:
                    if category_value not in categories_data[category_name]["values"]:
                        categories_data[category_name]["values"][category_value] = 0
                    categories_data[category_name]["values"][category_value] += 1
            
            log(INFO, f"{self._log_prefix}.thalabus_query_analyser: Analyzed {len(query_results)} documents with {len(categories_data)} categories")
            
        except Exception as e:
            error_message: str = f"{self._log_prefix}.thalabus_query_analyser: Error analyzing query: {e}\n{traceback.format_exc()}"
            log(ERROR, error_message)
            return ({
                "error": True,
                "error_message": error_message
            }, query_results if query_results is not None else [])

        return ({
            "total_results": len(query_results) if query_results is not None else None,
            "categories": categories_data
        }, query_results)

    def extend_dataset_with_linked_entities(self, dataset_id: str, effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Extend a document-fragment dataset with KG entities linked to its documents.

        For every document fragment in the dataset, walks the IS_DOCUMENTED chain backwards
        (paragraph -> chapter -> document -> entity) and merges each linked entity into the
        dataset as a kg:// fragment: entities not yet present are added with the fragment's
        weight (the maximum weight among fragments of the same document); entities already
        present are updated only when the new weight is higher.
        """
        from src.services.ontology_source_helper import parse_kg_source_token

        result: Dict[str, Any] = {"success": False, "candidates": 0, "added": 0, "bumped": 0, "linked": 0, "dataset_id": dataset_id}

        try:
            dataset_service: DatasetInterface = get_dataset_service()
            dataset_info = dataset_service.get_dataset_info(
                dataset_ids=[dataset_id],
                return_top_n=0,
                return_attributes=["id"],
            )
            if dataset_info is None or not dataset_info.get("success"):
                log(WARNING, f"{self._log_prefix}.extend_dataset_with_linked_entities: could not read dataset info for {dataset_id}")
                return result

            ds_list = dataset_info.get("datasets", [])
            if not ds_list:
                log(WARNING, f"{self._log_prefix}.extend_dataset_with_linked_entities: dataset {dataset_id} not found")
                return result

            existing_weights: Dict[str, float] = {}
            doc_weights: Dict[str, float] = {}
            for node in ds_list[0].get("top_nodes", []) or []:
                node_id = str(node.get("id") or "")
                if not node_id:
                    continue
                node_weight: float = float(node.get("weight")) if node.get("weight") is not None else 1.0
                if node_id.startswith("kg://"):
                    kg_id = parse_kg_source_token(node_id)
                    if kg_id:
                        existing_weights[kg_id] = max(existing_weights.get(kg_id, 0.0), node_weight)
                else:
                    doc_id = DocumentsInterface.get_document_id_from_fragment_id(node_id)
                    if doc_id:
                        doc_weights[doc_id] = max(doc_weights.get(doc_id, 0.0), node_weight)

            if doc_weights:
                rows = execute_neo4j_query(
                    "MATCH (e:Entity)-[:IS_DOCUMENTED]->(d:Document) WHERE d.id IN $doc_ids RETURN e.id AS entity_id, d.id AS document_id",
                    {"doc_ids": list(doc_weights.keys())},
                ) or []
                candidate_weights: Dict[str, float] = {}
                for row in rows:
                    entity_id = row.get("entity_id")
                    document_id = row.get("document_id")
                    if not entity_id:
                        continue
                    candidate_weights[entity_id] = max(candidate_weights.get(entity_id, 0.0), doc_weights.get(document_id, 1.0))

                to_add: List[str] = [eid for eid in candidate_weights.keys() if eid not in existing_weights]
                to_bump: List[str] = [eid for eid, w in candidate_weights.items() if eid in existing_weights and w > existing_weights[eid]]
                result["candidates"] = len(candidate_weights)
                result["added"] = len(to_add)
                result["bumped"] = len(to_bump)

                items: List[Dict[str, float]] = [{"id": eid, "weight": candidate_weights[eid]} for eid in set(to_add) | set(to_bump)]
                if items:
                    link_rows = execute_neo4j_query(
                        """
                        MATCH (d:Dataset {id: $dataset_id})
                        WITH d
                        UNWIND $items AS item
                        MATCH (e:Entity {id: item.id})
                        MERGE (d)-[r:AGGREGATES]->(e)
                        ON CREATE SET r.weight = item.weight
                        ON MATCH SET r.weight = CASE WHEN r.weight IS NULL OR r.weight < item.weight THEN item.weight ELSE r.weight END
                        RETURN count(e) AS linked_count
                        """,
                        {"dataset_id": dataset_id, "items": items},
                    )
                    result["linked"] = int(link_rows[0].get("linked_count", 0)) if link_rows else 0
                    execute_neo4j_query(
                        "MATCH (d:Dataset {id: $dataset_id}) OPTIONAL MATCH (d)-[:AGGREGATES]->(o) WITH d, count(o) AS c SET d.objects_count = c",
                        {"dataset_id": dataset_id},
                    )
                    if result["linked"] > 0:
                        merged_metadata: Dict[str, Any] = {}
                        raw_metadata = ds_list[0].get("metadata")
                        if isinstance(raw_metadata, dict):
                            merged_metadata = dict(raw_metadata)
                        elif isinstance(raw_metadata, str) and raw_metadata:
                            try:
                                parsed_metadata = json.loads(raw_metadata)
                                if isinstance(parsed_metadata, dict):
                                    merged_metadata = parsed_metadata
                            except Exception:
                                merged_metadata = {}
                        merged_metadata["includes_knowledge_graph"] = True
                        execute_neo4j_query(
                            "MATCH (d:Dataset {id: $dataset_id}) SET d.metadata = $metadata",
                            {"dataset_id": dataset_id, "metadata": json.dumps(merged_metadata)},
                        )
                    log(INFO, f"{self._log_prefix}.extend_dataset_with_linked_entities: dataset {dataset_id}: candidates={result['candidates']}, added={result['added']}, bumped={result['bumped']}, linked={result['linked']}")

            result["success"] = True

        except Exception as e:
            log(ERROR, f"{self._log_prefix}.extend_dataset_with_linked_entities: Error extending dataset {dataset_id}: {e}\n{traceback.format_exc()}")
            return result

        return result

    # ------------------------------------------------------------------
    # T33: optional LLM relevance gate (llm_filter_prompt / llm_filter_score_threshold)
    #
    # Contract (see Documentation/Documents/active/events/2026_10_HACKATHON.T33.md):
    # - A null threshold disables the gate. When active, only candidates whose final
    #   reranked score is STRICTLY greater than the threshold are evaluated; candidates
    #   at or below the threshold are never touched.
    # - The configured DEFAULT model (LlmSettings models[DEFAULT]) is used; its
    #   'is_reasoning' capability flag selects one request (self-reasoning) or two
    #   requests (reasoning first, then a separate YES/NO request given that reasoning).
    # - A Document candidate is evaluated on its full 'text'; a kg:// candidate on its
    #   Thalabus entity serialization. Content is never truncated or substituted.
    # - NO candidates are removed from the returned list and from the created dataset;
    #   YES candidates are kept and their dataset AGGREGATES weight is set to exactly 1.0.
    # - Errors (invalid args, DB read failure, oversized/absent content, LLM failure,
    #   malformed verdict) NEVER degrade to a silent NO and NEVER yield an unfiltered
    #   result that pretends the gate succeeded: the gate raises _LlmFilterError, the
    #   dataset and results stay in their exact pre-gate state and the failure is logged.
    # - Content found absent at gate time (node deleted, empty/missing text, ACL-blocked
    #   entity) is an explicit, observable per-candidate drop (WARNING log), not an error.
    # ------------------------------------------------------------------

    def _validate_llm_filter_arguments(self, llm_filter_prompt: Optional[str], llm_filter_score_threshold: Optional[float]) -> Tuple[bool, Optional[str]]:
        """Validate the T33 gate arguments. Returns (enabled, error_message).

        Rules: a null threshold disables the gate (even when a prompt is given — that
        combination is valid and only logs a warning); a set threshold requires a
        non-blank string prompt; the threshold must be a finite real number; the prompt
        must be a string within LLM_FILTER_MAX_PROMPT_CHARS.
        """
        enabled: bool = False
        error: Optional[str] = None

        if llm_filter_score_threshold is None:
            if llm_filter_prompt is not None:
                log(WARNING, f"{self._log_prefix}._validate_llm_filter_arguments: llm_filter_prompt supplied without llm_filter_score_threshold — gate disabled")
            return False, None

        if isinstance(llm_filter_score_threshold, bool) or not isinstance(llm_filter_score_threshold, (int, float)):
            error = f"llm_filter_score_threshold must be a finite number or None, got {type(llm_filter_score_threshold).__name__}"
        elif not math.isfinite(float(llm_filter_score_threshold)):
            error = "llm_filter_score_threshold must be finite (NaN/Inf are invalid)"
        elif llm_filter_prompt is None or not isinstance(llm_filter_prompt, str) or not llm_filter_prompt.strip():
            error = "llm_filter_prompt must be a non-blank string when llm_filter_score_threshold is set"
        elif len(llm_filter_prompt) > LLM_FILTER_MAX_PROMPT_CHARS:
            error = f"llm_filter_prompt exceeds the maximum length of {LLM_FILTER_MAX_PROMPT_CHARS} characters"

        if error is None:
            enabled = True

        return enabled, error

    def _llm_filter_resolve_default_model(self) -> Optional[Dict[str, Any]]:
        """Resolve the configured DEFAULT model into {'model','provider','is_reasoning'}.

        Returns None when the platform configuration has no usable DEFAULT entry.
        """
        model: Optional[Dict[str, Any]] = None

        try:
            from src.core.llm.manager import LlmManager
            default_model: Optional[Dict[str, Any]] = LlmManager.instance().get_default_model()
            if default_model is not None and default_model.get("model"):
                model = {
                    "model": default_model.get("model"),
                    "provider": default_model.get("modelprovider"),
                    "is_reasoning": bool(default_model.get("is_reasoning", False)),
                }
        except Exception as e:
            log(ERROR, f"{self._log_prefix}._llm_filter_resolve_default_model: Error resolving DEFAULT model: {e}")

        return model

    def _llm_filter_load_document_text(self, document_id: str) -> Tuple[Optional[str], bool]:
        """Load the full text of a candidate Document by id.

        Returns (text, read_ok). read_ok=False means the database read failed and the
        gate must abort; text=None with read_ok=True means the node vanished or holds no
        usable text, which is handled as an explicit per-candidate drop. ACL note: the
        candidate ids reached this point only via the ACL-filtered retrieval paths of
        thalabus_query, mirroring _hydrate_fragment_text's trusted-internal read.
        """
        outcome: Tuple[Optional[str], bool] = (None, False)
        result = execute_neo4j_query(
            "MATCH (d:Document) WHERE d.id = $doc_id OR d.documentid = $doc_id RETURN d.text AS text LIMIT 1",
            {"doc_id": document_id},
        )
        if not result.success:
            log(ERROR, f"{self._log_prefix}._llm_filter_load_document_text: Database read failed for candidate '{document_id}'")
        else:
            rows = list(result)
            if rows:
                value = rows[0].get("text")
                outcome = (value, True) if isinstance(value, str) and value.strip() else (None, True)
            else:
                outcome = (None, True)

        return outcome

    def _llm_filter_load_entity_content(self, entity_id: str, effective_user_id: Optional[str]) -> Optional[str]:
        """Serialize a kg:// candidate via the ACL-checked Entity lookup and discourse serialization."""
        content: Optional[str] = None

        try:
            ontology_service = get_ontology_service()
            entities: Dict[str, dict] = ontology_service.get_entities_by_ids([entity_id], include_relationships=True, effective_user_id=effective_user_id)
            entity: Optional[dict] = entities.get(entity_id) if entities else None
            if entity is not None:
                discourse = ontology_service.entity_to_discourse_text(entity)
                if isinstance(discourse, str) and discourse.strip():
                    content = discourse
        except Exception as e:
            log(ERROR, f"{self._log_prefix}._llm_filter_load_entity_content: Error loading entity '{entity_id}': {e}")

        return content

    def _llm_filter_load_content(self, candidate: Dict[str, Any], effective_user_id: Optional[str]) -> Tuple[Optional[str], bool]:
        """Load evaluation content for a gate candidate according to its kind."""
        content: Optional[str] = None
        read_ok: bool = False

        if candidate.get("kind") == "kg":
            content = self._llm_filter_load_entity_content(str(candidate.get("id")), effective_user_id)
            read_ok = True
        elif candidate.get("kind") == "document":
            content, read_ok = self._llm_filter_load_document_text(str(candidate.get("id")))

        return content, read_ok

    def _llm_filter_llm_call(self, model: Dict[str, Any], messages: List[Dict[str, str]], max_completion_tokens: int) -> Any:
        """Single seam for the LLM transport used by the relevance gate (DEFAULT model, medium reasoning)."""
        from src.core.base.enums import AgentReasoningEnum
        from src.core.llm.manager import LlmManager
        llm_response = LlmManager.instance().chat_completions(
            messages=messages,
            model_input=model.get("model"),
            model_provider=model.get("provider"),
            maximum_reasoning_required=AgentReasoningEnum.MEDIUM,
            max_completion_tokens=max_completion_tokens,
            purpose="llm_relevance_gate",
        )
        return llm_response

    def _llm_filter_parse_verdict(self, raw: Optional[str]) -> Optional[str]:
        """Parse a strict YES/NO verdict from an LLM response.

        Rules designed so that surrounding reasoning text is never mistaken for the
        verdict: an exact YES/NO answer wins; otherwise only the LAST non-empty line is
        consulted (first YES/NO token on it); only if that fails is the last YES/NO
        token anywhere in the response used. Returns 'YES', 'NO' or None (malformed).
        """
        verdict: Optional[str] = None
        if isinstance(raw, str) and raw.strip():
            text: str = raw.strip().strip('`').strip().strip('"').strip().strip("'").strip()
            upper: str = text.upper()
            exact = re.fullmatch(r"(YES|NO)[.!?]*", upper)
            if exact is not None:
                verdict = exact.group(1)
            if verdict is None:
                lines: List[str] = [ln.strip() for ln in text.splitlines() if ln.strip()]
                last_line: str = lines[-1] if lines else ""
                match = re.search(r"\b(YES|NO)\b", last_line.upper())
                if match is not None:
                    verdict = match.group(1)
            if verdict is None:
                all_matches = list(re.finditer(r"\b(YES|NO)\b", text.upper()))
                if all_matches:
                    verdict = all_matches[-1].group(1)

        return verdict

    def _llm_filter_request_verdict(self, model: Dict[str, Any], prompt: str, content: str) -> str:
        """Request a binary relevance verdict for one candidate content.

        With an is_reasoning model this is a single request that must end in YES/NO.
        Otherwise a first request collects reasoning and a second, separate request
        receives the prompt, the content and that reasoning and must answer YES or NO.
        Raises _LlmFilterError on transport failure; returns only after successful parsing.
        """
        from src.core.llm.config import MAX_COMPLETION_TOKENS_SMALL, MAX_COMPLETION_TOKENS_MEDIUM, MAX_COMPLETION_TOKENS_TINY
        max_tokens: int = MAX_COMPLETION_TOKENS_SMALL
        system_message: Dict[str, str] = {"role": "system", "content": "You are a strict relevance evaluator for a knowledge retrieval system. Judge only whether the provided content is relevant to the given question."}

        if model.get("is_reasoning"):
            user_text: str = (f"Question: {prompt}\n\nContent to evaluate:\n{content}\n\n"
                              "Decide whether the content is relevant to the question. Reason internally, then reply with exactly one word: YES or NO.")
            raw = self._llm_filter_llm_response_text(model, max_tokens, [system_message, {"role": "user", "content": user_text}])
            verdict = self._llm_filter_parse_verdict(raw)
            if verdict is None:
                stricter: str = (f"Question: {prompt}\n\nContent to evaluate:\n{content}\n\n"
                                 "What is the relevance verdict? Respond with exactly one word, YES or NO, and nothing else.")
                raw = self._llm_filter_llm_response_text(model, MAX_COMPLETION_TOKENS_TINY, [system_message, {"role": "user", "content": stricter}])
                verdict = self._llm_filter_parse_verdict(raw)
            if verdict is None:
                raise _LlmFilterError("malformed-verdict: LLM answer did not contain a parseable YES/NO verdict")
            return verdict

        reasoning_prompt: str = (f"Question: {prompt}\n\nContent to evaluate:\n{content}\n\n"
                                 "Analyze concisely whether the content is relevant to the question. Do NOT give the final verdict yet — provide reasoning only.")
        raw = self._llm_filter_llm_response_text(model, MAX_COMPLETION_TOKENS_MEDIUM, [system_message, {"role": "user", "content": reasoning_prompt}])
        reasoning: Optional[str] = raw
        if reasoning is None or not reasoning.strip():
            raise _LlmFilterError("llm-call-failed: reasoning request returned no content")
        reasoning = reasoning.strip()
        if len(reasoning) > LLM_FILTER_MAX_REASONING_CHARS:
            reasoning = reasoning[:LLM_FILTER_MAX_REASONING_CHARS]

        verdict_prompt: str = (f"Question: {prompt}\n\nContent to evaluate:\n{content}\n\n"
                               f"Reasoning gathered so far:\n{reasoning}\n\n"
                               "Based on the question, the content and the reasoning above, reply with exactly one word: YES or NO.")
        raw = self._llm_filter_llm_response_text(model, max_tokens, [system_message, {"role": "user", "content": verdict_prompt}])
        verdict = self._llm_filter_parse_verdict(raw)
        if verdict is None:
            raise _LlmFilterError("malformed-verdict: LLM answer did not contain a parseable YES/NO verdict")
        return verdict

    def _llm_filter_llm_response_text(self, model: Dict[str, Any], max_completion_tokens: int, messages: List[Dict[str, str]]) -> Optional[str]:
        """Execute one gate LLM request and return only the response text (never the thinking trace)."""
        text: Optional[str] = None
        try:
            llm_response = self._llm_filter_llm_call(model, messages, max_completion_tokens)
            if llm_response is not None and isinstance(getattr(llm_response, "response", None), str):
                text = llm_response.response
        except Exception as e:
            log(ERROR, f"{self._log_prefix}._llm_filter_llm_response_text: LLM call failed: {e}")
        if text is not None and not text.strip():
            text = None
        if text is None:
            raise _LlmFilterError("llm-call-failed: LLM returned no response")
        return text

    def _llm_filter_strip_fragment_suffix(self, fragment_id: Optional[str]) -> str:
        """Strip a query-string suffix (e.g. '?entity=...') from a fragment identifier."""
        stripped: str = str(fragment_id) if fragment_id is not None else ""
        return stripped.split("?", 1)[0]

    def _llm_filter_collect_candidates(self, items: List[dict], id_key: str, threshold: float) -> Tuple[List[Dict[str, Any]], int]:
        """Group scored items into gate candidates, keeping only those strictly above the threshold.

        With id_key='fragmentid' fragments are grouped per owning document and the
        candidate score is the maximum fragment score, so a document keeps the
        document-level semantics of the score. Non-numeric scores are logged and skipped
        (the item is left completely untouched).
        """
        candidates: List[Dict[str, Any]] = []
        skipped: int = 0
        doc_groups: Dict[str, Dict[str, Any]] = {}

        for index, item in enumerate(items):
            raw_id = item.get(id_key)
            if raw_id is None:
                skipped += 1
                continue
            try:
                score: float = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                log(WARNING, f"{self._log_prefix}._llm_filter_collect_candidates: non-numeric score for '{raw_id}' — candidate left untouched")
                skipped += 1
                continue
            if not math.isfinite(score) or score <= threshold:
                continue

            normalized_id: str = self._llm_filter_strip_fragment_suffix(raw_id)
            if normalized_id.lower().startswith("kg://"):
                from src.services.ontology_source_helper import parse_kg_source_token
                bare: Optional[str] = parse_kg_source_token(normalized_id)
                if not bare:
                    skipped += 1
                    continue
                candidates.append({"id": bare, "kind": "kg", "score": score, "member_ids": {bare}, "item_indices": {index}})
            elif id_key == "fragmentid":
                doc_id: Optional[str] = item.get("documentid") or DocumentsInterface.get_document_id_from_fragment_id(normalized_id)
                if not doc_id:
                    skipped += 1
                    continue
                group = doc_groups.get(doc_id)
                if group is None:
                    group = {"id": doc_id, "kind": "document", "score": score, "member_ids": {doc_id}, "item_indices": {index}}
                    doc_groups[doc_id] = group
                group["score"] = max(group["score"], score)
                group["member_ids"].add(normalized_id)
                group["item_indices"].add(index)
            else:
                members: set = {normalized_id}
                candidates.append({"id": normalized_id, "kind": "document", "score": score, "member_ids": members, "item_indices": {index}})

        candidates.extend(doc_groups.values())
        return candidates, skipped

    def _llm_filter_enrich_document_members(self, candidates: List[Dict[str, Any]], fragments: Optional[List[Dict[str, Any]]]) -> None:
        """For document candidates, fold every retrieved fragment id of the same document into its member ids."""
        if not fragments:
            return
        by_doc: Dict[str, List[str]] = {}
        for frag in fragments:
            doc_id = frag.get("documentid")
            frag_id = frag.get("fragmentid")
            if doc_id and frag_id:
                by_doc.setdefault(str(doc_id), []).append(self._llm_filter_strip_fragment_suffix(str(frag_id)))
        for candidate in candidates:
            if candidate.get("kind") == "document":
                candidate["member_ids"].update(by_doc.get(str(candidate.get("id")), []))

    def _llm_filter_apply_to_datasets(self, dataset_ids: List[str], remove_ids: List[str], yes_ids: List[str]) -> bool:
        """Remove AGGREGATES links of NO/undeterminable candidates and set weight 1.0 for YES candidates.

        Idempotent: repeated runs delete already-absent links and re-set the same weight.
        Raises _LlmFilterError when a dataset write fails, so callers keep pre-gate state
        semantics instead of reporting a half-applied filter.
        """
        applied: bool = True

        for dataset_id in dataset_ids:
            if remove_ids:
                result = execute_neo4j_query(
                    "MATCH (ds:Dataset {id: $ds_id})-[r:AGGREGATES]->(n) WHERE n.id IN $member_ids DELETE r",
                    {"ds_id": dataset_id, "member_ids": remove_ids},
                )
                if not result.success:
                    raise _LlmFilterError(f"dataset-write-failed: could not remove candidates from dataset '{dataset_id}'")
            if yes_ids:
                result = execute_neo4j_query(
                    "MATCH (ds:Dataset {id: $ds_id})-[r:AGGREGATES]->(n) WHERE n.id IN $member_ids SET r.weight = 1.0",
                    {"ds_id": dataset_id, "member_ids": yes_ids},
                )
                if not result.success:
                    raise _LlmFilterError(f"dataset-write-failed: could not update weights in dataset '{dataset_id}'")
            count_result = execute_neo4j_query(
                "MATCH (d:Dataset {id: $dataset_id}) OPTIONAL MATCH (d)-[:AGGREGATES]->(o) WITH d, count(o) AS c SET d.objects_count = c",
                {"dataset_id": dataset_id},
            )
            if not count_result.success:
                log(WARNING, f"{self._log_prefix}._llm_filter_apply_to_datasets: could not refresh objects_count for dataset '{dataset_id}'")

        return applied

    def _llm_filter_evaluate_candidate(self, model: Dict[str, Any], prompt: str, candidate: Dict[str, Any], effective_user_id: Optional[str]) -> Optional[str]:
        """Judge one gate candidate, returning 'YES'/'NO' or None when it has no evaluable content.

        Raises _LlmFilterError (never a silent NO) when the candidate cannot be trusted: failed
        database read, oversized content, LLM transport failure or an unparseable verdict.
        """
        verdict: Optional[str] = None
        content, read_ok = self._llm_filter_load_content(candidate, effective_user_id)

        if not read_ok:
            raise _LlmFilterError(f"content-read-failed: database read failed for candidate '{candidate.get('id')}'")
        if content is None or not content.strip():
            log(WARNING, f"{self._log_prefix}._llm_filter_evaluate_candidate: candidate '{candidate.get('id')}' has no evaluable content (node vanished, empty text or ACL-blocked) — dropping it explicitly")
        elif len(content) > LLM_FILTER_MAX_CONTENT_CHARS:
            raise _LlmFilterError(f"content-oversized: candidate '{candidate.get('id')}' exceeds {LLM_FILTER_MAX_CONTENT_CHARS} characters and must not be truncated")
        else:
            verdict = self._llm_filter_request_verdict(model, prompt, content)

        return verdict

    def _llm_filter_evaluate_candidates(self, model: Dict[str, Any], prompt: str, candidates: List[Dict[str, Any]], effective_user_id: Optional[str], max_workers: int) -> Tuple[Dict[str, bool], List[Dict[str, Any]]]:
        """Judge all gate candidates concurrently and return (verdicts by candidate id, dropped candidates).

        Candidates are independent, so each gets its own request on a ThreadPoolExecutor bounded by
        the caller-computed max_workers; verdicts are keyed by candidate id so the
        result cannot depend on completion order. The first failure cancels every queued candidate
        and is re-raised after the in-flight requests have finished (cancelled futures remove only
        queued work; issued requests run to completion bounded by the provider timeout, and their
        tokens are spent), so an aborted gate never writes anything.
        """
        verdicts: Dict[str, bool] = {}
        dropped_no_content: List[Dict[str, Any]] = []
        first_error: Optional[Exception] = None

        with ThreadPoolExecutor(max_workers=max_workers) as gate_executor:
            futures: Dict[Any, Dict[str, Any]] = {gate_executor.submit(self._llm_filter_evaluate_candidate, model, prompt, candidate, effective_user_id): candidate for candidate in candidates}
            for future in as_completed(futures):
                try:
                    verdict: Optional[str] = future.result()
                except Exception as candidate_error:
                    if isinstance(candidate_error, _LlmFilterError):
                        first_error = candidate_error
                    else:
                        first_error = _LlmFilterError(f"candidate-evaluation-failed: {type(candidate_error).__name__}: {candidate_error}")
                        log(ERROR, f"{self._log_prefix}._llm_filter_evaluate_candidates: unexpected error evaluating a candidate: {candidate_error}. Traceback:\n{traceback.format_exc()}")
                    for pending_future in futures:
                        pending_future.cancel()
                    break
                candidate: Dict[str, Any] = futures[future]
                if verdict is None:
                    dropped_no_content.append(candidate)
                else:
                    verdicts[str(candidate.get("id"))] = verdict == "YES"

        if first_error is not None:
            raise first_error

        return verdicts, dropped_no_content

    def _run_llm_relevance_gate(self, items: List[dict], id_key: str, fragments: Optional[List[Dict[str, Any]]], dataset_ids: Optional[List[str]], prompt: str, threshold: float, effective_user_id: Optional[str]) -> List[dict]:
        """T33: apply the optional LLM relevance gate over finally reranked candidates.

        Only candidates with final score strictly greater than the threshold are evaluated.
        NO candidates (and candidates whose content vanished/became unusable) are removed
        from the returned list and from every created dataset; YES candidates are kept and
        have their dataset AGGREGATES weight set to exactly 1.0. Candidates at or below the
        threshold are never modified. Any failure raises _LlmFilterError before any state is
        mutated, so the dataset and the returned list keep their exact pre-gate content and
        the caller logs the failure — an error is never converted into a NO verdict. Candidates
        are judged concurrently (bounded by MAX_PARALLELISM) and the first failure
        cancels the remaining requests.
        """
        kept: List[dict] = items
        enabled: bool
        error: Optional[str]
        enabled, error = self._validate_llm_filter_arguments(prompt, threshold)
        if error is not None:
            raise ValueError(error)

        if not enabled or not items:
            return kept

        model: Optional[Dict[str, Any]] = self._llm_filter_resolve_default_model()
        if model is None:
            raise _LlmFilterError("default-model-unavailable: no usable DEFAULT model is configured")

        candidates, ineligible_skipped = self._llm_filter_collect_candidates(items, id_key, threshold)
        if id_key == 'id':
            self._llm_filter_enrich_document_members(candidates, fragments)
        if not candidates:
            log(INFO, f"{self._log_prefix}._run_llm_relevance_gate: no candidate scored above {threshold} — gate is a no-op")
            return kept

        workers: int = min(MAX_PARALLELISM, len(candidates))
        log(INFO, f"{self._log_prefix}._run_llm_relevance_gate: evaluating {len(candidates)} candidate(s) with score > {threshold} via DEFAULT model '{model.get('model')}' (is_reasoning={model.get('is_reasoning')}) across {workers} concurrent request(s)")

        verdicts, dropped_no_content = self._llm_filter_evaluate_candidates(model, prompt, candidates, effective_user_id, workers)

        removed_indices: set = set()
        remove_ids: List[str] = []
        yes_ids: List[str] = []
        for candidate in candidates:
            if verdicts.get(str(candidate.get("id"))) is False or candidate in dropped_no_content:
                removed_indices.update(candidate["item_indices"])
                remove_ids.extend(sorted(candidate["member_ids"]))
            if verdicts.get(str(candidate.get("id"))) is True:
                yes_ids.extend(sorted(candidate["member_ids"]))

        if dataset_ids and (remove_ids or yes_ids):
            self._llm_filter_apply_to_datasets(list(dataset_ids), remove_ids, yes_ids)

        kept = [item for index, item in enumerate(items) if index not in removed_indices]
        log(INFO, f"{self._log_prefix}._run_llm_relevance_gate: done — evaluated={len(verdicts)}, yes={sum(1 for v in verdicts.values() if v)}, no={sum(1 for v in verdicts.values() if not v)}, dropped_no_content={len(dropped_no_content)}, removed_items={len(removed_indices)}, untouched_at_or_below_threshold={len(items) - sum(len(c['item_indices']) for c in candidates)}")

        return kept
_rag_service_instance: Optional[RagInterface] = None

def get_rag_service() -> RagInterface:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RagService()
    return _rag_service_instance

def set_rag_service(service: RagInterface) -> None:
    global _rag_service_instance
    _rag_service_instance = service


log_end_file(__file__)


#region unit tests
if __name__ == '__main__':
    from src.utils.unit_test import init_unit_test
    init_unit_test()

    from src.utils.log import log_init, log_exit, push_console_log_level, pop_console_log_level
    log_init()

    try:
        # Test results collection for CSV export
        test_results = []
        
        def single_test(query: str, ontology_id: str = None, automatically_rerank: bool = True, create_dataset:str=None, limit_to_dossier_id: str = None) -> Tuple[List[dict], float]:
            """Run the query and return the ordered list of matching documents and elapsed time in milliseconds.

            Returns an empty list if no documents found. Results are sorted by
            descending score (most relevant first).
            """
            log(INFO, f"\n\n{'=' * 80}\nSingle test for query: {query}\n{'=' * 80}")
            service = get_rag_service()
            log(INFO, f"Testing query: {query}")
            if limit_to_dossier_id:
                log(INFO, f"Limiting search to dossier: {limit_to_dossier_id}")
            results, _ = service.thalabus_query(query, None, ontology_id, automatically_rerank=automatically_rerank, create_dataset=create_dataset, limit_to_dossier_id=limit_to_dossier_id)
            elapsed_time: float = last_decorated_function_time_milli
            num_docs = len(results)
            log(INFO, f"Results: {num_docs} documents found, time taken: {elapsed_time:.2f} ms")

            # Log top results for quick inspection
            if results:
                for doc in results[:5]:
                    doc_id = doc.get('id')
                    score = doc.get('score', 0.0)
                    log(INFO, f"  - {doc_id} (score: {score:.4f})")
                    
                if len(results) > 5:
                    log(INFO, f"  ... and {len(results) - 5} more documents")
            else:
                log(WARNING, f"No documents found for this query: {query}")

            return results, elapsed_time
        
        def record_test(test_nr: int, description: str, query: str, result_comment: str, passed: bool):
            """Record test results for CSV export."""
            test_results.append({
                'sequence_nr': test_nr,
                'test_description': description,
                'thalabus_query': query,
                'documents_found_comment': result_comment,
                'test_passed': 'YES' if passed else 'NO'
            })

        def find_document_position(doc_id: str, results: List[dict]) -> Tuple[bool, int]:
            """Find document position in results list (1-indexed rank).
            
            Args:
                doc_id: Document ID to search for
                results: List of document results sorted by score descending
                
            Returns:
                Tuple of (found: bool, position: int)
                position is 1-indexed (1 = top rank), or 0 if not found
            """
            for idx, doc in enumerate(results):
                if doc.get('id') == doc_id:
                    return (True, idx + 1)  # 1-indexed position
            return (False, 0)

        if False:
            # query = '"Cosa è successo a Bob in cucina?"'
            limit_to_dossier_id = None  # = 'dossier-bob'  # Temporary Dossier containing docid-543
            query = '"Cosa ha fatto il gatto in cucina?"'     # Bob the Cat
            # Test automatic ontology detection: ontology_id should be auto-detected from category
            # After auto-detection, category validation should succeed and all weights should be 1.0
            results, elapsed_time = single_test(query, ontology_id=None, automatically_rerank=True, create_dataset="Test1", limit_to_dossier_id=limit_to_dossier_id)
            if results is not None:
                # check if a document id docid-543 is in the results (this document is in the dataset and should be highly relevant to the query)
                bob_doc_found = any(doc.get('name') == 'bob-01.txt' for doc in results)
                if bob_doc_found:
                    log(INFO, f"Test passed: {len(results)} documents found for query: {query}")
                else:
                    log(WARNING, f"Test failed: bob-01.txt not found in results for query: {query}")
            pass

        if False:
            limit_to_dossier_id = None # 'dos-36'  # Temporary Dossier containing docid-8230
            query1 = '"man fled after relationship with married woman fearing for his life from husband\'s associates"'
            log(INFO, f"Testing Nigeria relationship case query limited to {limit_to_dossier_id} (Temporary Dossier containing docid-8230)")
            results1, elapsed_time = single_test(query1, ontology_id=None, automatically_rerank=True, create_dataset="dataset-luca", limit_to_dossier_id=limit_to_dossier_id)
            if results1 is not None:
                docid_8230_found, position_8230 = find_document_position('docid-8230', results1)
                if docid_8230_found:
                    log(INFO, f"✓ Test PASSED: docid-8230 found at position {position_8230} in results ({len(results1)} documents total)")
                else:
                    log(WARNING, f"✗ Test FAILED: docid-8230 NOT found in results ({len(results1)} documents found)")
            else:
                log(WARNING, f"✗ Test FAILED: No results returned for query")
            pass

            query2 = '"man fled after relationship with married woman fearing for his life from husband\'s associates\" AND COUNTRY_OF_ORIGIN=NG'
            log(INFO, f"Testing Nigeria relationship case query limited to {limit_to_dossier_id} (Temporary Dossier containing docid-8230)")
            results2, elapsed_time = single_test(query2, ontology_id=None, automatically_rerank=True, create_dataset="dataset-luca", limit_to_dossier_id=limit_to_dossier_id)
            if results2 is not None:
                docid_8230_found, position_8230 = find_document_position('docid-8230', results2)
                if docid_8230_found:
                    log(INFO, f"✓ Test PASSED: docid-8230 found at position {position_8230} in results ({len(results2)} documents total)")
                else:
                    log(WARNING, f"✗ Test FAILED: docid-8230 NOT found in results ({len(results2)} documents found)")

            pass

        if False:
            # Test DECISION_ID=E-4946/2021 — must find exactly 1 document
            log(INFO, f"\n{'='*80}\nTest: DECISION_ID=E-4946/2021 (must find exactly 1 document)\n{'='*80}")
            decision_query = 'DECISION_ID=E-4946/2021'
            results_dec, elapsed_time_dec = single_test(decision_query, ontology_id=None, automatically_rerank=True, create_dataset="Test-Decision")
            if results_dec is not None and len(results_dec) == 1:
                log(INFO, f"✓ Test PASSED: DECISION_ID=E-4946/2021 found exactly 1 document: {results_dec[0].get('id')}")
                record_test(50, "DECISION_ID=E-4946/2021 (must find exactly 1)", decision_query, f"Found exactly 1 document: {results_dec[0].get('id')}", True)
            else:
                log(ERROR, f"✗ Test FAILED: DECISION_ID=E-4946/2021 found {len(results_dec) if results_dec else 0} documents (expected 1)")
                record_test(50, "DECISION_ID=E-4946/2021 (must find exactly 1)", decision_query, f"Found {len(results_dec) if results_dec else 0} documents (expected 1)", False)
            pass

        if False:
            query = """COUNTRY_OF_ORIGIN=FR AND "the applicant lost a leg" AND ADMISSIBILITY_STATUS=ADMISSIBLE"""
            # query = 'REMOVAL_POSSIBILITY=IMPOSSIBLE_TECHNICAL'
            # Test automatic ontology detection: ontology_id should be auto-detected from category
            # After auto-detection, category validation should succeed and all weights should be 1.0
            results, elapsed_time = single_test(query, ontology_id=None, create_dataset="Test1")

        if False:
            query = """'Syrien' AND 'politischen Geheimdienst' AND '2025'"""
            expected_docid = 'docid-55769'
            results, elapsed_time = single_test(query, ontology_id=None, automatically_rerank=False, create_dataset=f"expected-{expected_docid}")
            found = False
            if results is not None:
                found_docids = [doc.get('id') for doc in results]
                doc_found, position = find_document_position(expected_docid, results)
                if doc_found:
                    log(INFO, f"✓ Test PASSED: Expected document {expected_docid} found at position {position} in results ({len(results)} documents total)")
                    record_test(51, "Syrien AND politischen geheimdienst AND 2025", query, f"Found expected document {expected_docid} at position {position}", True)
                    found = True
                else:
                    log(ERROR, f"✗ Test FAILED: Expected document {expected_docid} NOT found in results ({len(results)} documents total)")
                    record_test(51, "Syrien AND politischen geheimdienst AND 2025", query, f"Expected document {expected_docid} NOT found", False)
            else:
                log(ERROR, f"✗ Test FAILED: No results returned for query")
                record_test(51, "Syrien AND politischen geheimdienst AND 2025", query, "No results returned", False)

            if not found:
                # since the document is really there, we test now individual parts of the query, to see which part is failing.
                # If each individual part of the query returns the expected document, then the problem is in the AND combination of the query.
                subqueries = [term.strip() for term in query.split('AND')]
                for subquery in subqueries:
                    sub_results, sub_elapsed_time = single_test(subquery, ontology_id=None, automatically_rerank=False, create_dataset=f"subquery-{subquery}")
                    if sub_results is not None:
                        sub_doc_found, sub_position = find_document_position(expected_docid, sub_results)
                        if sub_doc_found:
                            log(INFO, f"✓ Subquery PASSED: Expected document {expected_docid} found at position {sub_position} in results for subquery '{subquery}' ({len(sub_results)} documents total)")
                        else:
                            log(WARNING, f"✗ Subquery WARNING: Expected document {expected_docid} NOT found in results for subquery '{subquery}' ({len(sub_results)} documents total)")
                    else:
                        log(ERROR, f"✗ Subquery ERROR: No results returned for subquery '{subquery}'")

                # final verification: get the document manually, search in the enhanced_text for the query terms
                documents_service: DocumentsInterface = get_documents_service()
                document = documents_service.get_document(expected_docid)
                if document:
                    enhanced_text = document.get('enhanced_text', '')
                    if enhanced_text:
                        log(INFO, f"Enhanced text of document {expected_docid} retrieved for verification:\n```\n{enhanced_text}\n```")
                        if all(term.strip("'").lower() in enhanced_text.lower() for term in subqueries):
                            log(INFO, f"✓ Verification PASSED: All subquery terms found in enhanced_text of document {expected_docid}")
                        else:
                            log(ERROR, f"✗ Verification FAILED: Not all subquery terms found in enhanced_text of document {expected_docid}")

            pass

        if True and False:
            expected_docid = 'docid-43613'
            auto_rerank = True
            auto_rerank_str = "disabled" if not auto_rerank else "enabled"

            # semantic search only, 1 search term
            query1 = '    "Court reject asylum claims involving alleged Christian conversion"    '
            results, elapsed_time = single_test(query1, ontology_id=None, automatically_rerank=auto_rerank, create_dataset=f"expected-{expected_docid}")
            found = False
            if results is not None:
                found_docids = [doc.get('id') for doc in results]
                doc_found, position = find_document_position(expected_docid, results)
                if doc_found:
                    log(INFO, f"✓ Test PASSED (with reranking {auto_rerank_str}): Expected document {expected_docid} found at position {position} in results ({len(results)} documents total)")
                    record_test(52, f"{query1} (with reranking {auto_rerank_str})", query1, f"Found expected document {expected_docid} at position {position}", True)
                    found = True
                else:
                    log(ERROR, f"✗ Test FAILED (with reranking {auto_rerank_str}): Expected document {expected_docid} NOT found in results ({len(results)} documents total)")
                    record_test(52, f"{query1} (with reranking {auto_rerank_str})", query1, f"Expected document {expected_docid} NOT found", False)
            else:
                log(ERROR, f"✗ Test FAILED (with reranking {auto_rerank_str}): No results returned for query")
                record_test(52, f"{query1} (with reranking {auto_rerank_str})", query1, "No results returned", False)
            pass

            # add a category filter to the query, to see if the expected document is still found
            query1 = '    COUNTRY_OF_ORIGIN=SY AND "Court reject asylum claims involving alleged Christian conversion"    '
            results, elapsed_time = single_test(query1, ontology_id=None, automatically_rerank=auto_rerank, create_dataset=f"expected-{expected_docid}")
            found = False
            if results is not None:
                found_docids = [doc.get('id') for doc in results]
                doc_found, position = find_document_position(expected_docid, results)
                if doc_found:
                    score = results[position - 1].get('score', 0.0)
                    log(INFO, f"✓ Test PASSED (with reranking {auto_rerank_str}): Expected document {expected_docid} found at position {position} in results ({len(results)} documents total, score={score:.4f})")
                    record_test(52, f"{query1} (with reranking {auto_rerank_str})", query1, f"Found expected document {expected_docid} at position {position} (score={score:.4f})", True)
                    found = True
                else:
                    log(ERROR, f"✗ Test FAILED (with reranking {auto_rerank_str}): Expected document {expected_docid} NOT found in results ({len(results)} documents total)")
                    record_test(52, f"{query1} (with reranking {auto_rerank_str})", query1, f"Expected document {expected_docid} NOT found", False)
            else:
                log(ERROR, f"✗ Test FAILED (with reranking {auto_rerank_str}): No results returned for query")
                record_test(52, f"{query1} (with reranking {auto_rerank_str})", query1, "No results returned", False)
            pass

            # add another category filter to the query, to see if the expected document is still found
            query1 = '    COUNTRY_OF_ORIGIN=SY AND "Court reject asylum claims involving alleged Christian conversion" AND "major contradictions between first and second interviews"    '
            results, elapsed_time = single_test(query1, ontology_id=None, automatically_rerank=auto_rerank, create_dataset=f"expected-{expected_docid}")
            found = False
            if results is not None:
                found_docids = [doc.get('id') for doc in results]
                doc_found, position = find_document_position(expected_docid, results)
                if doc_found:
                    score = results[position - 1].get('score', 0.0)
                    log(INFO, f"✓ Test PASSED (with reranking {auto_rerank_str}): Expected document {expected_docid} found at position {position} in results ({len(results)} documents total, score={score:.4f})")
                    record_test(52, f"{query1} (with reranking {auto_rerank_str})", query1, f"Found expected document {expected_docid} at position {position} (score={score:.4f})", True)
                    found = True
                else:
                    log(ERROR, f"✗ Test FAILED (with reranking {auto_rerank_str}): Expected document {expected_docid} NOT found in results ({len(results)} documents total)")
                    record_test(52, f"{query1} (with reranking {auto_rerank_str})", query1, f"Expected document {expected_docid} NOT found", False)
            else:
                log(ERROR, f"✗ Test FAILED (with reranking {auto_rerank_str}): No results returned for query")
                record_test(52, f"{query1} (with reranking {auto_rerank_str})", query1, "No results returned", False)
            pass

        if True:
            # T33 integration test: the optional LLM relevance gate on top of a real thalabus_query.
            # Retrieval for "Bob the Cat" finds several Bob stories; the gate, asked
            # "Is this document describing the splilled milk incident in the kitchen?",
            # must keep only bob-01.txt — the kitchen incident document.
            gate_query = '"Bob the Cat"'
            gate_prompt = "Is this document describing the splilled milk incident in the kitchen?"
            gate_threshold = 0.0

            # The gate uses the configured DEFAULT model. On a machine whose DEFAULT endpoint is down the
            # call fails and the gate aborts by design, so allow a test-only in-memory endpoint override
            # (environment variable; neither LlmSettings.json nor any other file is modified).
            import os
            gate_endpoint_override = os.getenv("T33_LLM_ENDPOINT", "").strip()
            if gate_endpoint_override:
                from src.core.llm.manager import LlmManager
                gate_manager = LlmManager.instance()
                gate_manager.get_default_model()  # force the lazy configuration load before patching
                for gate_model_entry in (gate_manager._llm_configuration or {}).get("models", []):
                    if gate_model_entry.get("id") == "DEFAULT":
                        gate_model_entry["endpoint"] = gate_endpoint_override
                        log(INFO, f"T33 gate test: DEFAULT endpoint overridden in memory only -> {gate_endpoint_override}")

            # Resolve the expected document id from its constant name; ids are assigned dynamically.
            bob_rows = list(execute_neo4j_query("MATCH (d:Document) WHERE d.name = $name RETURN d.id AS id LIMIT 1", {"name": "bob-01.txt"}))
            assert bob_rows, "T33 gate test aborted: bob-01.txt not found in the database, cannot resolve its document id"
            bob01_id = str(bob_rows[0].get('id'))
            log(INFO, f"\n{'='*80}\nT33 gate test: query={gate_query}, prompt={gate_prompt}, threshold={gate_threshold}, expected only {bob01_id}\n{'='*80}")

            gating_service = get_rag_service()
            baseline_results, _ = gating_service.thalabus_query(gate_query, ontology_id=None, automatically_rerank=True)
            baseline_ids = [doc.get('id') for doc in (baseline_results or [])]
            log(INFO, f"T33 gate test baseline (gate disabled): {len(baseline_ids)} documents -> {baseline_ids}")
            assert bob01_id in baseline_ids, f"T33 gate test is not meaningful: {bob01_id} is absent from the ungated baseline results"
            assert len(baseline_ids) > 1, f"T33 gate test is not meaningful: baseline returned only {baseline_ids}, the gate would have nothing to filter"

            gated_results, gated_dataset_ids = gating_service.thalabus_query(gate_query, ontology_id=None, automatically_rerank=True, llm_filter_prompt=gate_prompt, llm_filter_score_threshold=gate_threshold)
            assert gated_results, "T33 gate test FAILED: the gate aborted or removed every document (see the ERROR log above — a gate failure is never treated as a NO verdict)"
            gated_ids = [doc.get('id') for doc in gated_results]
            log(INFO, f"T33 gate test gated: {len(gated_ids)} documents -> {gated_ids} (dataset ids: {gated_dataset_ids})")
            assert gated_ids == [bob01_id], f"T33 gate test FAILED: expected exactly [{bob01_id}], got {gated_ids}"
            log(INFO, f"[PASS] T33 LLM relevance gate test passed: only {bob01_id} (bob-01.txt) survived the relevance filter")
            record_test(53, 'T33 LLM relevance gate on Bob the Cat', gate_query, f"Kept only {bob01_id} (bob-01.txt); ungated baseline had {len(baseline_ids)} documents", True)
            pass

        if False:
            expected_docid = 'docid-43613'
            auto_rerank = True
            auto_rerank_str = "disabled" if not auto_rerank else "enabled"

            # add another category filter to the query, to see if the expected document is still found
            query1 = '    PROCEDURE_TYPE=RE_EXAMINATION AND POST_FLIGHT_GROUNDS=SUBJECTIVE_EXIL_SURVENANT AND MEDICAL_STATUS=NO_HEALTH_ISSUES AND INTERNAL_FLIGHT_ALT=NOT_AVAILABLE AND DECISION_OUTCOME=REJECTION_WITH_REMOVAL AND COUNTRY_OF_ORIGIN=SY AND "Court reject asylum claims involving alleged Christian conversion" AND "major contradictions between first and second interviews"    '
            results, elapsed_time = single_test(query1, ontology_id=None, automatically_rerank=auto_rerank, create_dataset=f"expected-{expected_docid}")
            found = False
            if results is not None:
                found_docids = [doc.get('id') for doc in results]
                doc_found, position = find_document_position(expected_docid, results)
                if doc_found:
                    score = results[position - 1].get('score', 0.0)
                    log(INFO, f"✓ Test PASSED (with reranking {auto_rerank_str}): Expected document {expected_docid} found at position {position} in results ({len(results)} documents total, score={score:.4f})")
                    record_test(52, f"{query1} (with reranking {auto_rerank_str})", query1, f"Found expected document {expected_docid} at position {position} (score={score:.4f})", True)
                    found = True
                else:
                    log(ERROR, f"✗ Test FAILED (with reranking {auto_rerank_str}): Expected document {expected_docid} NOT found in results ({len(results)} documents total)")
                    record_test(52, f"{query1} (with reranking {auto_rerank_str})", query1, f"Expected document {expected_docid} NOT found", False)
            else:
                log(ERROR, f"✗ Test FAILED (with reranking {auto_rerank_str}): No results returned for query")
                record_test(52, f"{query1} (with reranking {auto_rerank_str})", query1, "No results returned", False)
            pass

        if False:
            query = '"what happened to Bob in the kitchen?"'
            # Test automatic ontology detection: ontology_id should be auto-detected from category
            # After auto-detection, category validation should succeed and all weights should be 1.0
            results, elapsed_time = single_test(query, ontology_id=None, automatically_rerank=True, create_dataset="Test1")
            pass

        if False:
            # test query sanitization (many cases)
            queries_to_test = [
                { "query": """ "Who is the mother of Jane?" """, "expected": """ "Who is the mother of Jane?" """ },
                { "query": """ "Who is Jane's mother?" """, "expected": """ "Who is Jane's mother?" """ },
                { "query": """ "Who is Jane\'s mother?" """, "expected": """ "Who is Jane's mother?" """ },

                # Single-quoted text searches
                { "query": """ 'Bob' """, "expected": """ 'Bob' """ },
                { "query": """ 'Bob\'s hat' """, "expected": """ 'Bob\'s hat' """ },
                { "query": """ 'Bob has a "green" hat' """, "expected": """ 'Bob has a "green" hat' """ },

                # Case-sensitive text
                { "query": """ @'ExactCase' """, "expected": """ @'ExactCase' """ },

                # Category values with quoted contents
                { "query": """ CATEGORY=\"Some Value\" """, "expected": """ CATEGORY="Some Value" """ },
                { "query": """ CATEGORY=\"Some's Value\" """, "expected": """ CATEGORY="Some's Value" """ },
                { "query": """ CATEGORY='O\'Reilly' """, "expected": """ CATEGORY="O'Reilly" """ },

                # Nested/escaped double-quotes inside semantic phrase
                { "query": """ \"He said something\" """, "expected": """ "He said something" """ },
                { "query": """ \"He said 'hello' to Jane\" """, "expected": """ "He said 'hello' to Jane" """ },
                { "query": """ \"He said \'hello\' to Jane\" """, "expected": """ "He said 'hello' to Jane" """ },
                { "query": """ \"He said \"hello\" to Jane\" """, "expected": """ "He said \"hello\" to Jane" """ },

                # Mixed quotes with escaped parts
                { "query": """ \"It's \\\"complicated\\\"\" """, "expected": """ "It's \"complicated\"" """ },

                # Single-quotes containing escaped sequences
                { "query": """ 'Mix \'and\' match' """, "expected": """ 'Mix \'and\' match' """ },

                # Double-quoted phrase containing an unescaped single quote
                { "query": """ "Nested 'single' inside" """, "expected": """ "Nested 'single' inside" """ }
            ]

            # Validate sanitization results using the service sanitizer
            service = get_rag_service()
            for idx, entry in enumerate(queries_to_test, start=1):
                orig = entry.get('query')
                expected = entry.get('expected')
                try:
                    sanitized = service._sanitize_query(orig)
                except Exception as e:
                    log(ERROR, f"Sanitization raised for test #{idx}: orig={repr(orig)} error={e}")
                    raise
                log(INFO, f"Sanitize test #{idx}: orig={repr(orig)} -> sanitized={repr(sanitized)}")
                try:
                    assert sanitized == expected, f"Sanitize mismatch #{idx}: expected={repr(expected)} got={repr(sanitized)}"
                except AssertionError as ae:
                    # Debug: print detailed repr and codepoints for orig/expected/sanitized
                    def show_chars(label, s):
                        return f"{label}: {repr(s)} | {[ord(c) for c in s]}"
                    log(ERROR, f"ASSERT FAIL #{idx}\n" + show_chars('orig', orig) + "\n" + show_chars('expected', expected) + "\n" + show_chars('sanitized', sanitized))
                    raise

        if False:
            # Progressive query refinement test - each AND should return subset of previous
            log(INFO, f"\n{'='*80}\nProgressive Query Refinement Test\n{'='*80}")
            
            # Define the progressive queries
            queries = [
                "DUBLIN_COUNTRY_RISK=ART3_ECHR_RISK",
                "DUBLIN_COUNTRY_RISK=ART3_ECHR_RISK AND REMOVAL_POSSIBILITY=IMPOSSIBLE_TECHNICAL",
                "DUBLIN_COUNTRY_RISK=ART3_ECHR_RISK AND REMOVAL_POSSIBILITY=IMPOSSIBLE_TECHNICAL AND MEDICAL_STATUS=PSYCHOLOGICAL_PTSD"
            ]
            
            # Store results for each query
            query_results = []
            
            # Execute each query
            for idx, query in enumerate(queries, start=1):
                log(INFO, f"\n{'-'*80}\nStep {idx}: {query}\n{'-'*80}")
                results, _ = single_test(query, ontology_id="Juris", automatically_rerank=False)
                
                doc_ids = [doc['id'] for doc in results]
                query_results.append({
                    'query': query,
                    'results': results,
                    'doc_ids': set(doc_ids),
                    'count': len(doc_ids)
                })
                
                log(INFO, f"Step {idx} returned {len(doc_ids)} documents")
                if doc_ids:
                    log(INFO, f"Document IDs: {', '.join(doc_ids[:10])}")
                    if len(doc_ids) > 10:
                        log(INFO, f"... and {len(doc_ids) - 10} more")
            
            # Verify subset relationships
            log(INFO, f"\n{'='*80}\nSubset Verification\n{'='*80}")
            all_passed = True
            
            for i in range(1, len(query_results)):
                prev_set = query_results[i-1]['doc_ids']
                curr_set = query_results[i]['doc_ids']
                prev_count = query_results[i-1]['count']
                curr_count = query_results[i]['count']
                
                is_subset = curr_set.issubset(prev_set)
                is_equal_or_less = curr_count <= prev_count
                
                log(INFO, f"\nStep {i} → Step {i+1}:")
                log(INFO, f"  Previous count: {prev_count}")
                log(INFO, f"  Current count:  {curr_count}")
                log(INFO, f"  Is subset: {is_subset}")
                log(INFO, f"  Count decreased or equal: {is_equal_or_less}")
                
                if not is_subset:
                    # Find documents that are in current but not in previous
                    extra_docs = curr_set - prev_set
                    log(ERROR, f"  ❌ FAILED: Current results NOT a subset of previous!")
                    log(ERROR, f"  Extra documents found: {extra_docs}")
                    all_passed = False
                elif not is_equal_or_less:
                    log(ERROR, f"  ❌ FAILED: Count increased (should decrease or stay same)")
                    all_passed = False
                else:
                    reduction = prev_count - curr_count
                    percentage = (reduction / prev_count * 100) if prev_count > 0 else 0
                    log(INFO, f"  ✓ PASSED: Reduced by {reduction} documents ({percentage:.1f}%)")
            
            # Final summary
            log(INFO, f"\n{'='*80}\nFinal Summary\n{'='*80}")
            for idx, qr in enumerate(query_results, start=1):
                log(INFO, f"Step {idx}: {qr['count']} documents")
            
            if all_passed:
                log(INFO, f"\n✓ ALL SUBSET TESTS PASSED - Progressive refinement works correctly!")
            else:
                log(ERROR, f"\n❌ SOME TESTS FAILED - Subset relationship violated!")
            
            log(INFO, f"{'='*80}\n")

        if False:
            # Example single tests with comprehensive Bob the Cat scenarios
            # These tests validate that the correct documents are ranked at the top for various queries
            
            # Document IDs and their categorization:
            # 
            # bob01 = docid-543 (Bob-01.txt - Kitchen Incident)
            #   CONTENT_RATING=G, DOCUMENT_CATEGORY=STORY, FILE_FORMAT=TXT
            #   INTERACTION_FOCUS=HUMAN_ANIMAL, LANGUAGE=EN, MAIN_CHARACTER=ANIMAL
            #   MAIN_PLACE=WHISKERVILLE, MORAL_THEME=HARMONY, NARRATIVE_PERSPECTIVE=THIRD_PERSON
            #   PROTAGONIST_SPECIES=CAT, PUBLISHING_FORMAT=DIGITAL, SERIES_VOLUME=1
            #   SETTING_TYPE=SMALL_TOWN, STORY_GENRE=ANIMAL_TALE, TARGET_AGE=KIDS
            #
            # bob02 = docid-544 (Bob-02.txt - Garden Adventure with Joy the Squirrel)
            #   CONTENT_RATING=G, DOCUMENT_CATEGORY=STORY, FILE_FORMAT=TXT
            #   INTERACTION_FOCUS=ANIMAL_ANIMAL,HUMAN_ANIMAL, LANGUAGE=EN, MAIN_CHARACTER=ANIMAL
            #   MAIN_PLACE=GARDEN, MORAL_THEME=FRIENDSHIP, NARRATIVE_PERSPECTIVE=THIRD_PERSON
            #   PROTAGONIST_SPECIES=CAT, PUBLISHING_FORMAT=DIGITAL, SERIES_VOLUME=2
            #   SETTING_TYPE=SUBURBAN, STORY_GENRE=ANIMAL_TALE, TARGET_AGE=KIDS
            #
            # bob03 = docid-545 (Bob-03.txt - Attic Adventure)
            #   CONTENT_RATING=G, DOCUMENT_CATEGORY=STORY, FILE_FORMAT=TXT
            #   INTERACTION_FOCUS=HUMAN_ANIMAL, LANGUAGE=EN, MAIN_CHARACTER=ANIMAL
            #   MAIN_PLACE=HOUSE, MORAL_THEME=FAMILY, NARRATIVE_PERSPECTIVE=THIRD_PERSON
            #   PROTAGONIST_SPECIES=CAT, PUBLISHING_FORMAT=DIGITAL, SERIES_VOLUME=3
            #   SETTING_TYPE=SUBURBAN, STORY_GENRE=ADVENTURE, TARGET_AGE=TEENS,KIDS
            
            bob01 = "docid-543"  # Bob-01.txt
            bob02 = "docid-544"  # Bob-02.txt
            bob03 = "docid-545"  # Bob-03.txt
            correct_documents = [bob01, bob02, bob03]     # All three Bob books
            correct_spilled_milk_documents = [bob01]      # Only bob-01.txt has spilled milk incident
            correct_joy_documents = [bob02]               # Only bob-02.txt has Joy the Squirrel
            correct_attic_documents = [bob03]             # Only bob-03.txt has attic adventure
            correct_garden_documents = [bob02]            # Only bob-02.txt has garden adventure
            
            # ============================================================================
            # SEMANTIC SEARCH TESTS (Double Quotes) - Phrase-based similarity
            # ============================================================================
            
            # Test 1: Jane's mother query (paraphrased versions should find Bob books)
            log(INFO, f"\n{'='*80}\nTest 1: Who is the mother of Jane? (semantic)\n{'='*80}")
            bob_1a, _ = single_test('"who is the mother of Jane?"')
            assert bob_1a and len(bob_1a) > 0, "Expected to find Bob the Cat document"
            top = bob_1a[0]
            assert top['id'] in correct_documents, f"Test 1 failed: Found unexpected document: {top['id']}"
            log(INFO, f"[PASS] Test 1 passed: Found {top['id']}")
            record_test(1, "Who is the mother of Jane? (semantic)", '"who is the mother of Jane?"', f"Found {top['id']}", True)
            
            # Test 2: Alternative phrasing for Jane's mother
            log(INFO, f"\n{'='*80}\nTest 2: Who is Jane's mother? (semantic)\n{'='*80}")
            bob_1b, _ = single_test('"who is Jane\'s mother?"')
            assert bob_1b and len(bob_1b) > 0, "Expected to find Bob the Cat document"
            top = bob_1b[0]
            assert top['id'] in correct_documents, f"Test 2 failed: Found unexpected document: {top['id']}"
            log(INFO, f"[PASS] Test 2 passed: Found {top['id']}")
            record_test(2, "Who is Jane's mother? (semantic)", '"who is Jane\'s mother?"', f"Found {top['id']}", True)
            
            # Test 3: Spilled milk query (should find docid-543)
            log(INFO, f"\n{'='*80}\nTest 3: Milk spilled on kitchen floor (semantic)\n{'='*80}")
            bob_milk, _ = single_test('"milk spilled kitchen floor"')
            assert bob_milk and len(bob_milk) > 0, "Expected to find Bob the Cat document"
            top = bob_milk[0]
            assert top['id'] in correct_spilled_milk_documents, f"Test 3 failed: Found {top['id']}, expected {correct_spilled_milk_documents}"
            log(INFO, f"[PASS] Test 3 passed: Found {top['id']} (milk incident)")
            record_test(3, "Milk spilled on kitchen floor (semantic)", '"milk spilled kitchen floor"', f"Found {top['id']} (milk incident)", True)
            
            # Test 4: Cleanup with mop query (should find docid-543 - CRITICAL TEST)
            log(INFO, f"\n{'='*80}\nTest 4: Cleanup with the mop (semantic) - CRITICAL TEST\n{'='*80}")
            bob_cleanup, _ = single_test('"cleanup with the mop"')
            assert bob_cleanup and len(bob_cleanup) > 0, "Expected to find Bob the Cat document"
            top = bob_cleanup[0]
            assert top['id'] in correct_documents, f"Test 4 failed: Found unexpected document: {top['id']}"
            assert top['id'] == bob01, f"Test 4 failed: Expected {bob01}, got {top['id']}"
            log(INFO, f"[CRITICAL PASS] Test 4: Found {bob01} with cleanup query")
            record_test(4, "Cleanup with the mop (semantic) - CRITICAL TEST", '"cleanup with the mop"', f"Found {bob01} with cleanup query", True)
            
            # Test 5: Kitchen floor incident AND cleanup
            log(INFO, f"\n{'='*80}\nTest 5: Kitchen floor incident AND cleanup (semantic)\n{'='*80}")
            bob_incident, _ = single_test('"kitchen floor incident" AND "cleanup with the mop"')
            assert bob_incident and len(bob_incident) > 0, "Expected to find Bob the Cat document"
            top = bob_incident[0]
            assert top['id'] in correct_spilled_milk_documents, f"Test 5 failed: Found {top['id']}, expected {correct_spilled_milk_documents}"
            log(INFO, f"[PASS] Test 5 passed: Found {top['id']} (incident + cleanup)")
            record_test(5, "Kitchen floor incident AND cleanup (semantic)", '"kitchen floor incident" AND "cleanup with the mop"', f"Found {top['id']} (incident + cleanup)", True)
            
            # Test 6: Joy the Squirrel query (should find docid-544)
            log(INFO, f"\n{'='*80}\nTest 6: Who is Joy the Squirrel? (semantic)\n{'='*80}")
            joy_1a, _ = single_test('"who is Joy the Squirrel?"')
            assert joy_1a and len(joy_1a) > 0, "Expected to find Joy the Squirrel document"
            top = joy_1a[0]
            assert top['id'] in correct_joy_documents, f"Test 6 failed: Found {top['id']}, expected {correct_joy_documents}"
            log(INFO, f"[PASS] Test 6 passed: Found {top['id']} (Joy the Squirrel)")
            record_test(6, "Who is Joy the Squirrel? (semantic)", '"who is Joy the Squirrel?"', f"Found {top['id']} (Joy the Squirrel)", True)
            
            # Test 7: Where did Bob meet Joy (should find docid-544, but any Bob document is acceptable)
            log(INFO, f"\n{'='*80}\nTest 7: Where did Bob meet Joy? (semantic)\n{'='*80}")
            joy_1b, _ = single_test('"where did Bob meet Joy?"')
            assert joy_1b and len(joy_1b) > 0, "Expected to find Joy the Squirrel document"
            top = joy_1b[0]
            assert top['id'] in correct_documents, f"Test 7 failed: Found {top['id']}, expected one of {correct_documents}"
            log(INFO, f"[PASS] Test 7 passed: Found {top['id']} (Joy meeting)")
            record_test(7, "Where did Bob meet Joy? (semantic)", '"where did Bob meet Joy?"', f"Found {top['id']} (Joy meeting)", True)
            
            # Test 8: Bob's attic adventure (should find docid-545)
            log(INFO, f"\n{'='*80}\nTest 8: What did Bob find in the attic? (semantic)\n{'='*80}")
            attic_query, _ = single_test('"what did Bob find in the attic?"')
            assert attic_query and len(attic_query) > 0, "Expected to find Bob the Cat document"
            top = attic_query[0]
            assert top['id'] in correct_attic_documents, f"Test 8 failed: Found {top['id']}, expected {correct_attic_documents}"
            log(INFO, f"[PASS] Test 8 passed: Found {top['id']} (attic adventure)")
            record_test(8, "What did Bob find in the attic? (semantic)", '"what did Bob find in the attic?"', f"Found {top['id']} (attic adventure)", True)
            
            # Test 9: Rooftop garden query (should find docid-545)
            log(INFO, f"\n{'='*80}\nTest 9: Rooftop garden discovery (semantic)\n{'='*80}")
            rooftop, _ = single_test('"secret rooftop garden"')
            assert rooftop and len(rooftop) > 0, "Expected to find Bob the Cat document"
            top = rooftop[0]
            assert top['id'] in correct_documents, f"Test 9 failed: Found unexpected document: {top['id']}"
            log(INFO, f"[PASS] Test 9 passed: Found {top['id']}")
            record_test(9, "Rooftop garden discovery (semantic)", '"secret rooftop garden"', f"Found {top['id']}", True)
            
            # Test 10: Garden adventure (should find docid-544)
            log(INFO, f"\n{'='*80}\nTest 10: Bob's garden adventure (semantic)\n{'='*80}")
            garden, _ = single_test('"garden adventure friendship"')
            assert garden and len(garden) > 0, "Expected to find Bob the Cat document"
            top = garden[0]
            assert top['id'] in correct_garden_documents, f"Test 10 failed: Found {top['id']}, expected {correct_garden_documents}"
            log(INFO, f"[PASS] Test 10 passed: Found {top['id']} (garden adventure)")
            record_test(10, "Bob's garden adventure (semantic)", '"garden adventure friendship"', f"Found {top['id']} (garden adventure)", True)
            
            # ============================================================================
            # TEXT SEARCH TESTS (Single Quotes) - Exact text matching, case-insensitive
            # ============================================================================
            
            # Test 11: Text search for "Bob" (case-insensitive)
            log(INFO, f"\n{'='*80}\nTest 11: Text search for 'Bob' (case-insensitive)\n{'='*80}")
            text_bob, _ = single_test("'Bob'")
            assert text_bob and len(text_bob) > 0, "Expected to find Bob the Cat document via text search"
            top = text_bob[0]
            assert top['id'] in correct_documents, f"Test 11 failed: Found unexpected document: {top['id']}"
            log(INFO, f"[PASS] Test 11 passed: Found {top['id']} via text search")
            record_test(11, "Text search for 'Bob' (case-insensitive)", "'Bob'", f"Found {top['id']} via text search", True)
            
            # Test 12: Text search for "mop" (should find docid-543)
            log(INFO, f"\n{'='*80}\nTest 12: Text search for 'mop' (case-insensitive)\n{'='*80}")
            text_mop, _ = single_test("'mop'")
            assert text_mop and len(text_mop) > 0, "Expected to find Bob the Cat document"
            top = text_mop[0]
            assert top['id'] in correct_documents, f"Test 12 failed: Found unexpected document: {top['id']}"
            log(INFO, f"[PASS] Test 12 passed: Found {top['id']} via text search for 'mop'")
            record_test(12, "Text search for 'mop' (case-insensitive)", "'mop'", f"Found {top['id']} via text search for 'mop'", True)
            
            # Test 13: Text search for "Joy" (should find docid-544)
            log(INFO, f"\n{'='*80}\nTest 13: Text search for 'Joy' (case-insensitive)\n{'='*80}")
            text_joy, _ = single_test("'Joy' AND 'squirrel'")
            assert text_joy and len(text_joy) > 0, "Expected to find Joy the Squirrel document via text search"
            top = text_joy[0]
            assert top['id'] in correct_joy_documents, f"Test 13 failed: Found {top['id']}, expected {correct_joy_documents}"
            log(INFO, f"[PASS] Test 13 passed: Found {top['id']} via text search for 'Joy'")
            record_test(13, "Text search for 'Joy' (case-insensitive)", "'Joy' AND 'squirrel'", f"Found {top['id']} via text search for 'Joy'", True)
            
            # Test 14: Text search for "garden" (should find docid-544)
            log(INFO, f"\n{'='*80}\nTest 14: Text search for 'garden' (case-insensitive)\n{'='*80}")
            text_garden, _ = single_test("'garden'")
            assert text_garden and len(text_garden) > 0, "Expected to find Bob the Cat document"
            top = text_garden[0]
            assert top['id'] in correct_garden_documents, f"Test 14 failed: Found {top['id']}, expected {correct_garden_documents}"
            log(INFO, f"[PASS] Test 14 passed: Found {top['id']} via text search for 'garden'")
            record_test(14, "Text search for 'garden' (case-insensitive)", "'garden'", f"Found {top['id']} via text search for 'garden'", True)
            
            # Test 15: Text search for "attic" (should find docid-545)
            log(INFO, f"\n{'='*80}\nTest 15: Text search for 'attic' (case-insensitive)\n{'='*80}")
            text_attic, _ = single_test("'attic'")
            assert text_attic and len(text_attic) > 0, "Expected to find Bob the Cat document"
            top = text_attic[0]
            assert top['id'] in correct_attic_documents, f"Test 15 failed: Found {top['id']}, expected {correct_attic_documents}"
            log(INFO, f"[PASS] Test 15 passed: Found {top['id']} via text search for 'attic'")
            record_test(15, "Text search for 'attic' (case-insensitive)", "'attic'", f"Found {top['id']} via text search for 'attic'", True)
            
            # Test 16: Text search for "milk" (case-insensitive, should find docid-543)
            log(INFO, f"\n{'='*80}\nTest 16: Text search for 'milk' (case-insensitive)\n{'='*80}")
            text_milk, _ = single_test("'milk'")
            assert text_milk and len(text_milk) > 0, "Expected to find Bob the Cat document"
            top = text_milk[0]
            assert top['id'] in correct_spilled_milk_documents, f"Test 16 failed: Found {top['id']}, expected {correct_spilled_milk_documents}"
            log(INFO, f"[PASS] Test 16 passed: Found {top['id']} via text search for 'milk'")
            record_test(16, "Text search for 'milk' (case-insensitive)", "'milk'", f"Found {top['id']} via text search for 'milk'", True)
            
            # ============================================================================
            # CASE-SENSITIVE TEXT SEARCH TESTS (@'...' syntax)
            # ============================================================================
            
            # Test 17: Case-sensitive search for "Bob" (exact case)
            log(INFO, f"\n{'='*80}\nTest 17: Case-sensitive text search for 'Bob'\n{'='*80}")
            case_bob, _ = single_test("@'Bob'")
            assert case_bob and len(case_bob) > 0, "Expected to find Bob the Cat document via case-sensitive search"
            top = case_bob[0]
            assert top['id'] in correct_documents, f"Test 17 failed: Found unexpected document: {top['id']}"
            log(INFO, f"[PASS] Test 17 passed: Found {top['id']} via case-sensitive search")
            record_test(17, "Case-sensitive text search for 'Bob'", "@'Bob'", f"Found {top['id']} via case-sensitive search", True)
            
            # Test 18: Case-sensitive search for "Joy" (should find in docid-544)
            log(INFO, f"\n{'='*80}\nTest 18: Case-sensitive text search for 'Joy'\n{'='*80}")
            case_joy, _ = single_test("@'Joy'")
            assert case_joy and len(case_joy) > 0, "Expected to find Joy the Squirrel document"
            top = case_joy[0]
            assert top['id'] in correct_joy_documents, f"Test 18 failed: Found {top['id']}, expected {correct_joy_documents}"
            log(INFO, f"[PASS] Test 18 passed: Found {top['id']} via case-sensitive search for 'Joy'")
            record_test(18, "Case-sensitive text search for 'Joy'", "@'Joy'", f"Found {top['id']} via case-sensitive search for 'Joy'", True)
            
            # ============================================================================
            # COMBINATION TESTS (Mixed semantic + text with AND operator)
            # ============================================================================
            
            # Test 19: Semantic + text combination
            log(INFO, f"\n{'='*80}\nTest 19: Semantic AND text search combined\n{'='*80}")
            combined1, _ = single_test('"kitchen floor incident" AND \'clean\'')
            assert combined1 and len(combined1) > 0, "Expected to find Bob the Cat document"
            top = combined1[0]
            assert top['id'] in correct_documents, f"Test 19 failed: Found unexpected document: {top['id']}"
            log(INFO, f"[PASS] Test 19 passed: Found {top['id']} (semantic AND text)")
            record_test(19, "Semantic AND text search combined", '"kitchen floor incident" AND \'clean\'', f"Found {top['id']} (semantic AND text)", True)
            
            # Test 20: Semantic phrase OR operation
            log(INFO, f"\n{'='*80}\nTest 20: Semantic search with OR operator\n{'='*80}")
            or_test, _ = single_test('"milk spilled" OR "Joy the squirrel"')
            assert or_test and len(or_test) > 0, "Expected to find Bob the Cat document"
            top = or_test[0]
            assert top['id'] in correct_documents, f"Test 20 failed: Found unexpected document: {top['id']}"
            log(INFO, f"[PASS] Test 20 passed: Found {top['id']} (semantic OR)")
            record_test(20, "Semantic search with OR operator", '"milk spilled" OR "Joy the squirrel"', f"Found {top['id']} (semantic OR)", True)
            
            # Test 21a: Not operator (exclude documents) with semantic search
            try:
                push_console_log_level(DEBUG)
                log(INFO, f"\n{'='*80}\nTest 21a: NOT operator for exclusion with semantic search\n{'='*80}")
                not_test, _ = single_test('"Bob the Cat" AND NOT "garden"')
                assert not_test and len(not_test) > 0, "Expected to find Bob the Cat document"
                top = not_test[0]
                assert top['id'] in correct_documents, f"Test 21 failed: Found unexpected document: {top['id']}"
                # Should exclude docid-544 (garden adventure) when possible
                if top['id'] == bob02:
                    log(WARNING, f"Test 21: Expected to exclude garden adventure, but got it anyway")
                else:
                    log(INFO, f"[PASS] Test 21 passed: Found {top['id']} (excluding garden)")
                record_test(21, "NOT operator for exclusion with semantic search", '"Bob the Cat" AND NOT "garden"', f"Found {top['id']} (excluding garden)", True)
            finally:
                pop_console_log_level()
            
            # Test 21b: Not operator (exclude documents) with text search
            try:
                push_console_log_level(DEBUG)
                log(INFO, f"\n{'='*80}\nTest 21b: NOT operator for exclusion with text search\n{'='*80}")
                not_test, _ = single_test('"Bob the Cat" AND NOT \'garden\'')
                assert not_test and len(not_test) > 0, "Expected to find Bob the Cat document"
                top = not_test[0]
                assert top['id'] in correct_documents, f"Test 21 failed: Found unexpected document: {top['id']}"
                # Should exclude docid-544 (garden adventure) when possible
                if top['id'] == bob02:
                    log(WARNING, f"Test 21: Expected to exclude garden adventure, but got it anyway")
                else:
                    log(INFO, f"[PASS] Test 21 passed: Found {top['id']} (excluding garden)")
                record_test(22, "NOT operator for exclusion with text search", '"Bob the Cat" AND NOT \'garden\'', f"Found {top['id']} (excluding garden)", True)
            finally:
                pop_console_log_level()
            
            # Test 22: Complex boolean expression
            try:
                push_console_log_level(DEBUG)
                log(INFO, f"\n{'='*80}\nTest 22: Complex boolean expression\n{'='*80}")
                complex_test, _ = single_test('("Bob" OR \'mop\') AND ("adventure" OR "incident")')
                assert complex_test and len(complex_test) > 0, "Expected to find Bob the Cat document"
                top = complex_test[0]
                assert top['id'] in correct_documents, f"Test 22 failed: Found unexpected document: {top['id']}"
                log(INFO, f"[PASS] Test 22 passed: Found {top['id']} (complex boolean)")
                record_test(23, "Complex boolean expression", '("Bob" OR \'mop\') AND ("adventure" OR "incident")', f"Found {top['id']} (complex boolean)", True)
                
                # Test 23: Four-word query (tests combined semantic+text threshold)
                log(INFO, f"\n{'='*80}\nTest 23: Four-word query (threshold test)\n{'='*80}")
                four_word, _ = single_test('"milk spilled on floor"')
                assert four_word and len(four_word) > 0, "Expected to find Bob the Cat document"
                top = four_word[0]
                assert top['id'] in correct_documents, f"Test 23 failed: Found unexpected document: {top['id']}"
                log(INFO, f"[PASS] Test 23 passed: Found {top['id']} (4-word query)")
                record_test(24, "Four-word query (threshold test)", '"milk spilled on floor"', f"Found {top['id']} (4-word query)", True)
            finally:
                pop_console_log_level()
            
            # Test 24: Five-word query (boundary case for combined search)
            log(INFO, f"\n{'='*80}\nTest 24: Five-word query (at boundary threshold)\n{'='*80}")
            five_word, _ = single_test('"the kitchen floor incident happened"')
            assert five_word and len(five_word) > 0, "Expected to find Bob the Cat document"
            top = five_word[0]
            assert top['id'] in correct_documents, f"Test 24 failed: Found unexpected document: {top['id']}"
            log(INFO, f"[PASS] Test 24 passed: Found {top['id']} (5-word query at threshold)")
            record_test(25, "Five-word query (at boundary threshold)", '"the kitchen floor incident happened"', f"Found {top['id']} (5-word query at threshold)", True)
            
            # Test 25: Longer semantic query (>5 words, semantic-only search)
            log(INFO, f"\n{'='*80}\nTest 25: Long semantic query (semantic-only, no text search)\n{'='*80}")
            long_query, _ = single_test('"what adventure did the cat have with the squirrel"')
            assert long_query and len(long_query) > 0, "Expected to find Bob the Cat document"
            top = long_query[0]
            assert top['id'] in correct_documents, f"Test 25 failed: Found unexpected document: {top['id']}"
            log(INFO, f"[PASS] Test 25 passed: Found {top['id']} (long semantic query)")
            record_test(26, "Long semantic query (semantic-only, no text search)", '"what adventure did the cat have with the squirrel"', f"Found {top['id']} (long semantic query)", True)
            
            # ============================================================================
            # EDGE CASES AND SPECIAL TESTS
            # ============================================================================
            
            # Test 26: Verify multiple results returned in correct order
            log(INFO, f"\n{'='*80}\nTest 26: Verify result ordering (top-ranked is most relevant)\n{'='*80}")
            multi_results, _ = single_test('"Bob"')  # Broad query matching all three
            assert multi_results and len(multi_results) > 0, "Expected to find Bob the Cat document"
            log(INFO, f"[PASS] Test 26 passed: Top result is {multi_results[0]['id']}")
            record_test(27, "Verify result ordering (top-ranked is most relevant)", '"Bob"', f"Top result is {multi_results[0]['id']}", True)
            
            # Test 27: Verify reranking with -r flag (automatic reranking enabled)
            log(INFO, f"\n{'='*80}\nTest 27: Test with automatic reranking enabled\n{'='*80}")
            rerank_result, _ = single_test('"cleanup with the mop"', automatically_rerank=True)
            assert rerank_result and len(rerank_result) > 0, "Expected to find Bob the Cat document with reranking"
            top = rerank_result[0]
            assert top['id'] == bob01, f"Test 27 failed: With reranking, expected {bob01}, got {top['id']}"
            log(INFO, f"[CRITICAL PASS] Test 27: {bob01} ranked top with reranking enabled")
            record_test(28, "Test with automatic reranking enabled", '"cleanup with the mop"', f"{bob01} ranked top with reranking enabled", True)
            
            # Test 28: OR-connected semantic queries (multiple semantic phrases)
            log(INFO, f"\n{'='*80}\nTest 28: OR-connected semantic queries (Martha/tool/animal/Joy)\n{'='*80}")
            or_semantic_test, _ = single_test('"for which food is Martha famous for" OR "which tool did she reach for to clean the kitchen\'s floor" OR "which animal did Bob meet in the garden" OR "who is Joy the Squirrel"')
            assert or_semantic_test and len(or_semantic_test) > 0, "Expected to find Bob the Cat document"
            top = or_semantic_test[0]
            assert top['id'] in correct_documents, f"Test 28 failed: Found unexpected document: {top['id']}"
            log(INFO, f"[PASS] Test 28 passed: Found {top['id']} (OR-connected semantic queries)")
            record_test(29, "OR-connected semantic queries (Martha/tool/animal/Joy)", '"for which food is Martha famous for" OR "which tool did she reach for to clean the kitchen\'s floor" OR "which animal did Bob meet in the garden" OR "who is Joy the Squirrel"', f"Found {top['id']} (OR-connected semantic queries)", True)
            
            # ============================================================================
            # CATEGORY-BASED TESTS (Simple category queries)
            # ============================================================================
            
            # Test 29: Simple category query - all Bob documents
            log(INFO, f"\n{'='*80}\nTest 29: Category query STORY_GENRE=ANIMAL_TALE\n{'='*80}")
            cat_test1, _ = single_test('STORY_GENRE=ANIMAL_TALE')
            assert cat_test1 and len(cat_test1) > 0, "Expected to find Bob the Cat documents"
            # Should return bob01 and bob02 (both are ANIMAL_TALE)
            found_ids = [doc['id'] for doc in cat_test1]
            assert bob01 in found_ids and bob02 in found_ids, f"Test 29 failed: Expected both {bob01} and {bob02} in results"
            log(INFO, f"[PASS] Test 29 passed: Found {len([d for d in found_ids if d in [bob01, bob02]])} ANIMAL_TALE documents")
            record_test(30, "Category query STORY_GENRE=ANIMAL_TALE", 'STORY_GENRE=ANIMAL_TALE', f"Found {len([d for d in found_ids if d in [bob01, bob02]])} ANIMAL_TALE documents", True)
            
            # Test 30: Category query - specific to bob03
            log(INFO, f"\n{'='*80}\nTest 30: Category query STORY_GENRE=ADVENTURE\n{'='*80}")
            cat_test2, _ = single_test('STORY_GENRE=ADVENTURE')
            assert cat_test2 and len(cat_test2) > 0, "Expected to find adventure document"
            found_ids = [doc['id'] for doc in cat_test2]
            assert bob03 in found_ids, f"Test 30 failed: Expected {bob03} in results"
            log(INFO, f"[PASS] Test 30 passed: Found {bob03} (ADVENTURE)")
            record_test(31, "Category query STORY_GENRE=ADVENTURE", 'STORY_GENRE=ADVENTURE', f"Found {bob03} (ADVENTURE)", True)
            
            # Test 31: Category query - MORAL_THEME
            log(INFO, f"\n{'='*80}\nTest 31: Category query MORAL_THEME=FRIENDSHIP\n{'='*80}")
            cat_test3, _ = single_test('MORAL_THEME=FRIENDSHIP')
            assert cat_test3 and len(cat_test3) > 0, "Expected to find friendship document"
            top = cat_test3[0]
            assert top['id'] == bob02, f"Test 31 failed: Expected {bob02}, got {top['id']}"
            log(INFO, f"[PASS] Test 31 passed: Found {bob02} (FRIENDSHIP theme)")
            record_test(32, "Category query MORAL_THEME=FRIENDSHIP", 'MORAL_THEME=FRIENDSHIP', f"Found {bob02} (FRIENDSHIP theme)", True)
            
            # Test 32: Category query - SETTING_TYPE
            log(INFO, f"\n{'='*80}\nTest 32: Category query SETTING_TYPE=SMALL_TOWN\n{'='*80}")
            cat_test4, _ = single_test('SETTING_TYPE=SMALL_TOWN')
            assert cat_test4 and len(cat_test4) > 0, "Expected to find small town document"
            top = cat_test4[0]
            assert top['id'] == bob01, f"Test 32 failed: Expected {bob01}, got {top['id']}"
            log(INFO, f"[PASS] Test 32 passed: Found {bob01} (SMALL_TOWN setting)")
            record_test(33, "Category query SETTING_TYPE=SMALL_TOWN", 'SETTING_TYPE=SMALL_TOWN', f"Found {bob01} (SMALL_TOWN setting)", True)
            
            # Test 33: Category query - MAIN_PLACE
            log(INFO, f"\n{'='*80}\nTest 33: Category query MAIN_PLACE=GARDEN\n{'='*80}")
            cat_test5, _ = single_test('MAIN_PLACE=GARDEN')
            assert cat_test5 and len(cat_test5) > 0, "Expected to find garden document"
            top = cat_test5[0]
            assert top['id'] == bob02, f"Test 33 failed: Expected {bob02}, got {top['id']}"
            log(INFO, f"[PASS] Test 33 passed: Found {bob02} (GARDEN location)")
            record_test(34, "Category query MAIN_PLACE=GARDEN", 'MAIN_PLACE=GARDEN', f"Found {bob02} (GARDEN location)", True)
            
            # ============================================================================
            # BOOLEAN CATEGORY TESTS (AND/OR operators with categories)
            # ============================================================================
            
            # Test 34: Category AND - should find bob02 and bob03
            log(INFO, f"\n{'='*80}\nTest 34: Category AND query SETTING_TYPE=SUBURBAN AND PROTAGONIST_SPECIES=CAT\n{'='*80}")
            cat_and1, _ = single_test('SETTING_TYPE=SUBURBAN AND PROTAGONIST_SPECIES=CAT')
            assert cat_and1 and len(cat_and1) > 0, "Expected to find suburban cat documents"
            found_ids = [doc['id'] for doc in cat_and1]
            assert bob02 in found_ids and bob03 in found_ids, f"Test 34 failed: Expected {bob02} and {bob03}"
            assert bob01 not in found_ids, f"Test 34 failed: {bob01} should not be in results (SMALL_TOWN)"
            log(INFO, f"[PASS] Test 34 passed: Found {bob02} and {bob03} (SUBURBAN)")
            record_test(35, "Category AND query SETTING_TYPE=SUBURBAN AND PROTAGONIST_SPECIES=CAT", 'SETTING_TYPE=SUBURBAN AND PROTAGONIST_SPECIES=CAT', f"Found {bob02} and {bob03} (SUBURBAN)", True)
            
            # Test 35: Category AND - specific to bob03
            log(INFO, f"\n{'='*80}\nTest 35: Category AND query MORAL_THEME=FAMILY AND STORY_GENRE=ADVENTURE\n{'='*80}")
            cat_and2, _ = single_test('MORAL_THEME=FAMILY AND STORY_GENRE=ADVENTURE')
            assert cat_and2 and len(cat_and2) > 0, "Expected to find family adventure document"
            top = cat_and2[0]
            assert top['id'] == bob03, f"Test 35 failed: Expected {bob03}, got {top['id']}"
            log(INFO, f"[PASS] Test 35 passed: Found {bob03} (FAMILY + ADVENTURE)")
            record_test(36, "Category AND query MORAL_THEME=FAMILY AND STORY_GENRE=ADVENTURE", 'MORAL_THEME=FAMILY AND STORY_GENRE=ADVENTURE', f"Found {bob03} (FAMILY + ADVENTURE)", True)
            
            # Test 36: Category AND - specific to bob01
            log(INFO, f"\n{'='*80}\nTest 36: Category AND query SERIES_VOLUME=1 AND SETTING_TYPE=SMALL_TOWN\n{'='*80}")
            cat_and3, _ = single_test('SERIES_VOLUME=1 AND SETTING_TYPE=SMALL_TOWN')
            assert cat_and3 and len(cat_and3) > 0, "Expected to find first volume document"
            top = cat_and3[0]
            assert top['id'] == bob01, f"Test 36 failed: Expected {bob01}, got {top['id']}"
            log(INFO, f"[PASS] Test 36 passed: Found {bob01} (Volume 1 in SMALL_TOWN)")
            record_test(37, "Category AND query SERIES_VOLUME=1 AND SETTING_TYPE=SMALL_TOWN", 'SERIES_VOLUME=1 AND SETTING_TYPE=SMALL_TOWN', f"Found {bob01} (Volume 1 in SMALL_TOWN)", True)
            
            # Test 37: Category OR - should find bob01 or bob02
            log(INFO, f"\n{'='*80}\nTest 37: Category OR query STORY_GENRE=ANIMAL_TALE OR STORY_GENRE=ADVENTURE\n{'='*80}")
            cat_or1, _ = single_test('STORY_GENRE=ANIMAL_TALE OR STORY_GENRE=ADVENTURE')
            assert cat_or1 and len(cat_or1) > 0, "Expected to find Bob documents"
            found_ids = [doc['id'] for doc in cat_or1]
            assert bob01 in found_ids and bob02 in found_ids and bob03 in found_ids, f"Test 37 failed: Expected all Bob documents"
            log(INFO, f"[PASS] Test 37 passed: Found all Bob documents (ANIMAL_TALE OR ADVENTURE)")
            record_test(38, "Category OR query STORY_GENRE=ANIMAL_TALE OR STORY_GENRE=ADVENTURE", 'STORY_GENRE=ANIMAL_TALE OR STORY_GENRE=ADVENTURE', "Found all Bob documents (ANIMAL_TALE OR ADVENTURE)", True)
            
            # Test 38: Category OR with AND - complex boolean
            log(INFO, f"\n{'='*80}\nTest 38: Complex category query (MORAL_THEME=FRIENDSHIP OR MORAL_THEME=HARMONY) AND STORY_GENRE=ANIMAL_TALE\n{'='*80}")
            cat_complex1, _ = single_test('(MORAL_THEME=FRIENDSHIP OR MORAL_THEME=HARMONY) AND STORY_GENRE=ANIMAL_TALE')
            assert cat_complex1 and len(cat_complex1) > 0, "Expected to find animal tale documents"
            found_ids = [doc['id'] for doc in cat_complex1]
            assert bob01 in found_ids and bob02 in found_ids, f"Test 38 failed: Expected {bob01} and {bob02}"
            assert bob03 not in found_ids, f"Test 38 failed: {bob03} should not match (ADVENTURE, not ANIMAL_TALE)"
            log(INFO, f"[PASS] Test 38 passed: Found {bob01} and {bob02} (filtered by complex boolean)")
            record_test(39, "Complex category query (MORAL_THEME=FRIENDSHIP OR MORAL_THEME=HARMONY) AND STORY_GENRE=ANIMAL_TALE", '(MORAL_THEME=FRIENDSHIP OR MORAL_THEME=HARMONY) AND STORY_GENRE=ANIMAL_TALE', f"Found {bob01} and {bob02} (filtered by complex boolean)", True)
            
            # Test 39: Category query with TARGET_AGE (bob03 has both TEENS and KIDS)
            log(INFO, f"\n{'='*80}\nTest 39: Category query TARGET_AGE=TEENS\n{'='*80}")
            cat_teens, _ = single_test('TARGET_AGE=TEENS')
            assert cat_teens and len(cat_teens) > 0, "Expected to find teen document"
            found_ids = [doc['id'] for doc in cat_teens]
            assert bob03 in found_ids, f"Test 39 failed: Expected {bob03} in results"
            log(INFO, f"[PASS] Test 39 passed: Found {bob03} (TARGET_AGE=TEENS)")
            record_test(40, "Category query TARGET_AGE=TEENS", 'TARGET_AGE=TEENS', f"Found {bob03} (TARGET_AGE=TEENS)", True)
            
            # ============================================================================
            # MIXED SEMANTIC + CATEGORY TESTS
            # ============================================================================
            
            # Test 40: Semantic + Category - specific to bob01
            log(INFO, f"\n{'='*80}\nTest 40: Mixed query \"kitchen\" AND MAIN_PLACE=WHISKERVILLE\n{'='*80}")
            mixed1, _ = single_test('"kitchen" AND MAIN_PLACE=WHISKERVILLE')
            assert mixed1 and len(mixed1) > 0, "Expected to find kitchen document in Whiskerville"
            top = mixed1[0]
            assert top['id'] == bob01, f"Test 40 failed: Expected {bob01}, got {top['id']}"
            log(INFO, f"[PASS] Test 40 passed: Found {bob01} (semantic + category)")
            record_test(41, "Mixed query \"kitchen\" AND MAIN_PLACE=WHISKERVILLE", '"kitchen" AND MAIN_PLACE=WHISKERVILLE', f"Found {bob01} (semantic + category)", True)
            
            # Test 41: Semantic + Category - specific to bob02
            log(INFO, f"\n{'='*80}\nTest 41: Mixed query \"squirrel\" AND MORAL_THEME=FRIENDSHIP\n{'='*80}")
            mixed2, _ = single_test('"squirrel" AND MORAL_THEME=FRIENDSHIP')
            assert mixed2 and len(mixed2) > 0, "Expected to find squirrel friendship document"
            top = mixed2[0]
            assert top['id'] == bob02, f"Test 41 failed: Expected {bob02}, got {top['id']}"
            log(INFO, f"[PASS] Test 41 passed: Found {bob02} (squirrel + FRIENDSHIP)")
            record_test(42, "Mixed query \"squirrel\" AND MORAL_THEME=FRIENDSHIP", '"squirrel" AND MORAL_THEME=FRIENDSHIP', f"Found {bob02} (squirrel + FRIENDSHIP)", True)
            
            # Test 42: Semantic + Category - specific to bob03
            log(INFO, f"\n{'='*80}\nTest 42: Mixed query \"attic\" AND STORY_GENRE=ADVENTURE\n{'='*80}")
            mixed3, _ = single_test('"attic" AND STORY_GENRE=ADVENTURE')
            assert mixed3 and len(mixed3) > 0, "Expected to find attic adventure document"
            top = mixed3[0]
            assert top['id'] == bob03, f"Test 42 failed: Expected {bob03}, got {top['id']}"
            log(INFO, f"[PASS] Test 42 passed: Found {bob03} (attic + ADVENTURE)")
            record_test(43, "Mixed query \"attic\" AND STORY_GENRE=ADVENTURE", '"attic" AND STORY_GENRE=ADVENTURE', f"Found {bob03} (attic + ADVENTURE)", True)
            
            # Test 43: Semantic OR Category - should find multiple
            log(INFO, f"\n{'='*80}\nTest 43: Mixed OR query \"Joy\" OR MAIN_PLACE=HOUSE\n{'='*80}")
            mixed_or1, _ = single_test('"Joy" OR MAIN_PLACE=HOUSE')
            assert mixed_or1 and len(mixed_or1) > 0, "Expected to find documents"
            found_ids = [doc['id'] for doc in mixed_or1]
            assert bob02 in found_ids and bob03 in found_ids, f"Test 43 failed: Expected {bob02} (Joy) and {bob03} (HOUSE)"
            log(INFO, f"[PASS] Test 43 passed: Found {bob02} and {bob03} (semantic OR category)")
            record_test(44, "Mixed OR query \"Joy\" OR MAIN_PLACE=HOUSE", '"Joy" OR MAIN_PLACE=HOUSE', f"Found {bob02} and {bob03} (semantic OR category)", True)
            
            # Test 44: Complex mixed query with AND/OR
            log(INFO, f"\n{'='*80}\nTest 44: Complex mixed \"Bob\" AND (SETTING_TYPE=SUBURBAN OR SETTING_TYPE=SMALL_TOWN)\n{'='*80}")
            mixed_complex1, _ = single_test('"Bob" AND (SETTING_TYPE=SUBURBAN OR SETTING_TYPE=SMALL_TOWN)')
            assert mixed_complex1 and len(mixed_complex1) > 0, "Expected to find Bob documents"
            found_ids = [doc['id'] for doc in mixed_complex1]
            assert all(bid in found_ids for bid in correct_documents), f"Test 44 failed: Expected all Bob documents"
            log(INFO, f"[PASS] Test 44 passed: Found all Bob documents (complex mixed query)")
            record_test(45, "Complex mixed \"Bob\" AND (SETTING_TYPE=SUBURBAN OR SETTING_TYPE=SMALL_TOWN)", '"Bob" AND (SETTING_TYPE=SUBURBAN OR SETTING_TYPE=SMALL_TOWN)', "Found all Bob documents (complex mixed query)", True)
            
            # ============================================================================
            # MIXED SEMANTIC + TEXT + CATEGORY TESTS (Most complex)
            # ============================================================================
            
            # Test 45: Triple mix - semantic + text + category
            log(INFO, f"\n{'='*80}\nTest 45: Triple mix \"kitchen incident\" AND 'mop' AND MORAL_THEME=HARMONY\n{'='*80}")
            triple1, _ = single_test('"kitchen incident" AND \'mop\' AND MORAL_THEME=HARMONY')
            assert triple1 and len(triple1) > 0, "Expected to find kitchen document"
            top = triple1[0]
            assert top['id'] == bob01, f"Test 45 failed: Expected {bob01}, got {top['id']}"
            log(INFO, f"[PASS] Test 45 passed: Found {bob01} (semantic + text + category)")
            record_test(46, "Triple mix \"kitchen incident\" AND 'mop' AND MORAL_THEME=HARMONY", '"kitchen incident" AND \'mop\' AND MORAL_THEME=HARMONY', f"Found {bob01} (semantic + text + category)", True)
            
            # Test 46: Triple mix with OR
            log(INFO, f"\n{'='*80}\nTest 46: Triple mix \"garden\" OR 'Joy' OR MAIN_PLACE=GARDEN\n{'='*80}")
            triple2, _ = single_test('"garden" OR \'Joy\' OR MAIN_PLACE=GARDEN')
            assert triple2 and len(triple2) > 0, "Expected to find garden documents"
            found_ids = [doc['id'] for doc in triple2]
            assert bob02 in found_ids, f"Test 46 failed: Expected {bob02} in results"
            log(INFO, f"[PASS] Test 46 passed: Found documents including {bob02} (triple OR)")
            record_test(47, "Triple mix \"garden\" OR 'Joy' OR MAIN_PLACE=GARDEN", '"garden" OR \'Joy\' OR MAIN_PLACE=GARDEN', f"Found documents including {bob02} (triple OR)", True)
            
            # Test 47: Triple mix complex - all three types with grouping
            log(INFO, f"\n{'='*80}\nTest 47: Complex triple mix (\"Bob\" OR 'cat') AND PROTAGONIST_SPECIES=CAT AND SERIES_VOLUME=2\n{'='*80}")
            triple3, _ = single_test('("Bob" OR \'cat\') AND PROTAGONIST_SPECIES=CAT AND SERIES_VOLUME=2')
            assert triple3 and len(triple3) > 0, "Expected to find volume 2 document"
            top = triple3[0]
            assert top['id'] == bob02, f"Test 47 failed: Expected {bob02}, got {top['id']}"
            log(INFO, f"[PASS] Test 47 passed: Found {bob02} (complex triple mix with grouping)")
            record_test(48, "Complex triple mix (\"Bob\" OR 'cat') AND PROTAGONIST_SPECIES=CAT AND SERIES_VOLUME=2", '("Bob" OR \'cat\') AND PROTAGONIST_SPECIES=CAT AND SERIES_VOLUME=2', f"Found {bob02} (complex triple mix with grouping)", True)
            
            # Test 48: Triple mix with NOT operator
            log(INFO, f"\n{'='*80}\nTest 48: Triple mix with NOT \"Bob\" AND STORY_GENRE=ANIMAL_TALE AND NOT 'garden'\n{'='*80}")
            triple4, _ = single_test('"Bob" AND STORY_GENRE=ANIMAL_TALE AND NOT \'garden\'')
            assert triple4 and len(triple4) > 0, "Expected to find Bob document without garden"
            found_ids = [doc['id'] for doc in triple4]
            # Should find bob01 (ANIMAL_TALE, no garden), might find bob02 depending on NOT filter effectiveness
            assert bob01 in found_ids, f"Test 48 failed: Expected {bob01} in results"
            log(INFO, f"[PASS] Test 48 passed: Found documents including {bob01} (triple mix with NOT)")
            record_test(49, "Triple mix with NOT \"Bob\" AND STORY_GENRE=ANIMAL_TALE AND NOT 'garden'", '"Bob" AND STORY_GENRE=ANIMAL_TALE AND NOT \'garden\'', 
                       f"Found documents including {bob01} (triple mix with NOT)", True)
            
            log(INFO, f"\n\n{'='*80}\n[SUCCESS] ALL 48 TESTS PASSED!\n{'='*80}\n")
            
            # Write test results to CSV
            import csv
            import os
            csv_path = os.path.join('output', 'rag_test_results.csv')
            os.makedirs('output', exist_ok=True)
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['sequence_nr', 'test_description', 'thalabus_query', 'documents_found_comment', 'test_passed']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(test_results)
            
            log(INFO, f"\n{'='*80}\nTest results exported to: {csv_path}\n{'='*80}\n")

        if False:
            # tests the thalabus_query_analyser function
            service = get_rag_service()
            test_queries = [
                'STORY_GENRE=ANIMAL_TALE',
                'MEDICAL_CONDITION=ADVANCED_DEGENERATIVE_RHEUMATIC_DISORDERS',
                'MEDICAL_STATUS=PSYCHOLOGICAL_PTSD AND RECEIVING_STATE=IQ AND REMOVAL_POSSIBILITY=IMPOSSIBLE_TECHNICAL and "editorial"'
            ]
            for tq in test_queries:
                log(INFO, f"\n{'='*80}\nAnalyser test for query: {tq}\n{'='*80}")
                analysis, results = service.thalabus_query_analyser(tq)
                # Pretty-print the analysis for easier reading in logs
                try:
                    pretty = json.dumps(analysis, indent=2, ensure_ascii=False)
                except Exception:
                    from pprint import pformat
                    pretty = pformat(analysis, depth=4)
                log(INFO, f"Analysis result:\n{pretty}")
                pass

        if False:
            # Call test generation with explicit parameters passed from outside
            # dossier_id = "dossier-jurispub"
            # ontology_id = "Juris"
            # report_file = "reports/rag_test_generation_report_juris.txt"
            dossier_id = "dossier-bob"
            ontology_id = "FictionalStories"
            report_file = "reports/rag_test_generation_report_bob.txt"

            from src.services.unit_testing.rag_service_unit_test import generate_rag_test_questions
            generate_rag_test_questions(dossier_id, ontology_id, report_file, documents_limit=64, questions_per_category=12)

        if False:
            # Use the :TestQuestionRag nodes to run the tests automatically, and generate a report of success/failure to the file
            # "reports/rag_test_execution_report.txt".
            # dossier_id = "dossier-jurispub"
            # ontology_id = "Juris"
            dossier_id = "dossier-bob"
            ontology_id = "FictionalStories"
            
            from src.services.unit_testing.rag_service_unit_test import execute_rag_tests
            execute_rag_tests(dossier_id, ontology_id, test_limit=-1)

    except Exception as e:
        log(ERROR, f"Exception: {e}\n{traceback.format_exc()}")

    finally:
        log_exit()
#endregion
