from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import json
import os
import requests
from typing import Any, Dict, List, Optional
from src.config.constants import BLAZOR_HOSTNAME, BLAZOR_PORT, BLAZOR_PROTOCOL, USE_REAL_SSL_CERTIFICATE
from src.core.base.InputOutput import execute_neo4j_query
from src.core.base.enums import HintRecommendation
from src.core.base.hints import HintConsolidationPolicy, get_hint_consolidation_policy, get_injection_excluded_hint_recommendations
from src.interfaces.hints_interface import HintsInterface
from src.interfaces.reranking_interface import RerankingInterface
from src.core.llm.embeddings import EmbeddingsManager
from src.services.reranking_service import get_reranking_service
from src.utils.log import log, DEBUG, INFO, WARNING, ERROR, FATAL


class HintsService(HintsInterface):
    """Concrete implementation of HintsInterface for Neo4j database operations."""
    _warning_already_logged: bool = False

    def find_hints_without_embeddings(self) -> Dict[str, Dict[str, str]]:
        """Find all hints that don't have embeddings yet."""
        query = """
                MATCH (h:LlmHint)
                WHERE h.embedding IS null
                RETURN h.prompt AS prompt_text, h.agent_hints AS agent_hints
            """
        parameters = {}
        result = execute_neo4j_query(query, parameters)
        if result is None:
            return {}
        
        # Convert list of records to a dictionary: {prompt_text: deserialized_agent_hints}
        hints_dict = {}
        for record in result:
            prompt_text = record[0]
            agent_hints_json = record[1]
            try:
                agent_hints = json.loads(agent_hints_json) if agent_hints_json else {}
                hints_dict[prompt_text] = agent_hints
            except Exception as e:
                log(WARNING, f"Failed to parse agent_hints JSON for prompt '{prompt_text}': '{agent_hints_json}'. Exception: {e}")
                hints_dict[prompt_text] = {}
        
        return hints_dict

    def create_hints_vector_indexes(self, embeddings_dimension: int) -> None:
        """Create or update the vector indexes for hints in the database."""
        execute_neo4j_query("""
                CREATE VECTOR INDEX `hints` IF NOT EXISTS
                FOR (h: LlmHint) ON (h.embedding)
                OPTIONS { indexConfig: {
                    `vector.dimensions`: $dimension,
                    `vector.similarity_function`: 'cosine'
                } }
            """,
            {"dimension": embeddings_dimension},
        )

    def update_hint(self, prompt_text: str, agent_hints: Dict[str, str], embedding: List[float], suppress_debug_logs: bool = False) -> bool:
        """
        Update or create a user hint in the Neo4j database for the given prompt_text and agent_hints.
        Stores agent_hints as a JSON string along with the provided embedding.
        agent_hints is a Dict[str, str] where key=agent_name, value=hint_text. Use "DEFAULT" as the agent key for hints that apply to all agents.
        """
        if embedding is None:
            log(ERROR, f"The embeddings for the hint '{prompt_text}' are None.")
            return False # embeddings are None

        query = """
            MERGE (h:LlmHint {prompt: $prompt})
            SET h.embedding = $embeddings,
                h.agent_hints = $agent_hints_json
            RETURN h.prompt AS prompt
        """
        parameters = {
            "prompt": prompt_text,
            "embeddings": embedding,
            "agent_hints_json": json.dumps(agent_hints, ensure_ascii=False)
        }
        result = execute_neo4j_query(query, parameters)
        if result:
            if not suppress_debug_logs:
                log(DEBUG, f"The hint '{prompt_text}' has been created or updated.")
        else:
            log(ERROR, f"The hint '{prompt_text}' could not be created or updated.")
        # Note: Vector index creation should be handled externally with proper embeddings_dimension
        return True  # hint has been created or updated

    def delete_hint(self, prompt_text: str) -> bool:
        """Delete a hint by its prompt text."""
        query = """
            MATCH (h:LlmHint {prompt: $prompt_text})
            DETACH DELETE h
        """
        parameters = {"prompt_text": prompt_text}
        result = execute_neo4j_query(query, parameters)
        if result:
            log(INFO, f"The hint with prompt '{prompt_text}' has been deleted.")
            return True  # hint has been deleted
        else:
            log(ERROR, f"The hint with prompt '{prompt_text}' could not be deleted.")
            return False  # hint could not be deleted

    def get_best_hints_for_agent(self, agent_name: str, user_prompt: str, embedding: List[float], top_n: int, min_score: float, exclude: Optional[List[HintRecommendation]] = None) -> List[Dict[str, str]]:
        """
        Retrieve the most relevant hint from Neo4j based on the agent name and the prompt by vectorizing the user prompt.
        Only return hints where the similarity score is at least min_score (unless min_score is -1, which means no minimum).
        
        Args:
            agent_name: The name of the agent requesting the hint
            user_prompt: The user prompt to find hints for (for logging/debugging purposes)
            embedding: The embedding vector for the user prompt
            top_n: Maximum number of hints to retrieve from Neo4j
            min_score: Minimum similarity score required for a hint to be considered. 
                       Use -1 to disable minimum score requirement.
            exclude: List of analysis recommendations; matching hints are never returned.
                     None (default) applies the service's default exclusion (see get_default_excluded_hint_recommendations).
                     An empty list disables the exclusion and returns hints with any recommendation.
                     HintRecommendation.NONE matches hints that have no analysis recommendation.
            
        Returns:
            The best matching hints, ordered best first, each one a dict with the hint 'id'
            and the hint 'text' for the requesting agent, or None if no suitable hint is found
        """
        if embedding is None:
            return None

        # Check if the vector index exists, return None if it doesn't
        if not self._ensure_vector_index_exists():
            return None

        # When no explicit exclusion was requested, apply the centralized default exclusion
        if exclude is None:
            exclude = self.get_default_excluded_hint_recommendations()

        # Determine the analysis recommendations that must never be returned
        excluded_recommendations: set = set()
        exclude_unanalyzed_hints: bool = False
        if exclude:
            for recommendation in exclude:
                if recommendation.value is None:
                    exclude_unanalyzed_hints = True
                else:
                    excluded_recommendations.add(recommendation.value)

        # Query Neo4j for the most relevant hints
        query: str = """CYPHER 25
            MATCH (h:LlmHint)
              SEARCH h IN (
                VECTOR INDEX hints
                FOR $embeddings
                LIMIT $top_n
              ) SCORE AS score
            RETURN h.id as hint_id, h.prompt as prompt, h.agent_hints as agent_hints_json, h.analysis_recommendation as analysis_recommendation, score
            ORDER BY score DESC
            LIMIT $top_n
        """

        parameters: dict = {"embeddings": embedding, "top_n": top_n}
        result = execute_neo4j_query(query, parameters, ignore_exception=True)
        best_hints: List[Dict[str, str]] = []
        no_minimum_required: bool = (min_score == -1)

        if result:
            for record in result:
                hint_id: Optional[str] = record.get("hint_id")
                agent_hints_json: str = record.get("agent_hints_json")
                score: float = record.get("score", 0)
                analysis_recommendation: Optional[str] = record.get("analysis_recommendation")

                # Skip hints whose analysis recommendation is excluded
                if analysis_recommendation is None:
                    if exclude_unanalyzed_hints:
                        continue
                elif analysis_recommendation in excluded_recommendations:
                    continue

                # Check if score meets minimum requirement (or if no minimum is required)
                if agent_hints_json and (no_minimum_required or score >= min_score):
                    try:
                        agent_hints = json.loads(agent_hints_json)
                        # agent_hints is now a Dict[str, str] where key=agent_name, value=hint_text

                        # Prefer agent-specific hint over DEFAULT
                        hint_text: Optional[str] = None
                        if agent_name in agent_hints:
                            hint_text = agent_hints[agent_name]
                        elif "DEFAULT" in agent_hints:
                            hint_text = agent_hints["DEFAULT"]

                        hint_prompt: Optional[str] = record.get("prompt")
                        if hint_text is not None and hint_prompt is not None:
                            best_hints.append({"id": hint_id, "prompt": hint_prompt, "text": hint_text})
                        elif hint_text is not None:
                            log(ERROR, f"Hint '{hint_id}' has agent guidance but no prompt.")

                    except Exception as e:
                        log(ERROR, f"Failed to parse AgentHints JSON: {e}")

        if len(best_hints) > 1:
            try:
                reranking_service: RerankingInterface = get_reranking_service()
                if reranking_service is not None:
                    hint_texts: List[str] = [hint["text"] for hint in best_hints]
                    reranked = reranking_service.rerank_strings(user_prompt, hint_texts, must_return_top_n=False)
                    if reranked:
                        # Map the reranked texts back to their hints through the original index
                        reranked_hints: List[Dict[str, str]] = [best_hints[item['idx']] for item in reranked if item.get('idx') is not None and item['idx'] < len(best_hints)]
                        if reranked_hints:
                            best_hints = reranked_hints
            except Exception as e:
                log(WARNING, f"HintsService.get_best_hints_for_agent: Reranking error: {e}")

        return best_hints

    def get_default_excluded_hint_recommendations(self) -> List[HintRecommendation]:
        """Analysis recommendations excluded by get_best_hints_for_agent when no explicit exclude list is given: exactly those whose policy excludes injection."""
        return get_injection_excluded_hint_recommendations()

    def get_pending_consolidation_hints_for_agent(self, agent_name: str) -> List[str]:
        """Return applicable hints requiring consolidation, bypassing vector ranking and reranking."""
        query = """
            MATCH (h:LlmHint)
            WHERE h.needs_consolidation = true
            RETURN h.agent_hints AS agent_hints
        """
        records = execute_neo4j_query(query, {}, ignore_exception=True)
        pending_hints: List[str] = []

        if records:
            for record in records:
                agent_hints_value: Any = record.get("agent_hints")
                try:
                    agent_hints = json.loads(agent_hints_value) if isinstance(agent_hints_value, str) else agent_hints_value
                    if not isinstance(agent_hints, dict):
                        continue

                    hint = agent_hints.get(agent_name) or agent_hints.get("DEFAULT")
                    if isinstance(hint, str) and hint.strip():
                        pending_hints.append(hint)
                except (TypeError, ValueError) as ex:
                    log(WARNING, f"Failed to parse pending consolidation hint for agent '{agent_name}': {ex}")

        return pending_hints

    def get_hint_ids_by_recommendation(self, recommendation: HintRecommendation) -> List[str]:
        """Return Hint IDs matching the requested analysis recommendation."""
        if recommendation == HintRecommendation.NONE:
            query = """
                MATCH (h:LlmHint)
                WHERE h.needs_consolidation = true
                  AND h.analysis_recommendation IS NULL
                RETURN h.id AS hint_id
            """
            parameters = {}
        else:
            query = """
                MATCH (h:LlmHint)
                WHERE h.analysis_recommendation = $analysis_recommendation
                RETURN h.id AS hint_id
            """
            parameters = {"analysis_recommendation": recommendation.value}

        records = execute_neo4j_query(query, parameters, ignore_exception=True)
        hint_ids: List[str] = [
            record["hint_id"]
            for record in records or []
            if isinstance(record.get("hint_id"), str) and record["hint_id"]
        ]

        return hint_ids

    def get_hint_ids_requiring_merge(self) -> List[str]:
        """Return the IDs of Hints whose policy set the merge flag and which carry at least one merge candidate."""
        query = """
            MATCH (h:LlmHint)
            WHERE h.requires_hints_merge = true
              AND h.merge_candidates IS NOT NULL
              AND size(h.merge_candidates) > 0
            RETURN h.id AS hint_id
        """
        records = execute_neo4j_query(query, {}, ignore_exception=True)
        hint_ids: List[str] = [
            record["hint_id"]
            for record in records or []
            if isinstance(record.get("hint_id"), str) and record["hint_id"]
        ]

        return hint_ids

    def update_hint_state(
        self,
        hint_id: str,
        recommendation: HintRecommendation,
        needs_consolidation: Optional[bool],
        analysis_results: Optional[str] = None,
        merge_candidates: Optional[List[str]] = None
    ) -> bool:
        """Set one Hint's analysis recommendation, consolidation flag, analysis results, merge candidates, and the follow-up flags implied by the recommendation's policy."""
        policy: HintConsolidationPolicy = get_hint_consolidation_policy(recommendation)
        query = """
            MATCH (h:LlmHint {id: $hint_id})
            WITH h,
                 h.needs_consolidation AS previous_needs_consolidation,
                 h.analysis_recommendation AS previous_analysis_recommendation
            SET h.needs_consolidation = $needs_consolidation,
                h.analysis_recommendation = $analysis_recommendation,
                h.analysis_results = $analysis_results,
                h.merge_candidates = $merge_candidates,
                h.requires_human_engineer = $requires_human_engineer,
                h.requires_hints_merge = $requires_hints_merge
            RETURN h.id AS hint_id,
                   previous_needs_consolidation,
                   previous_analysis_recommendation
        """
        parameters = {
            "hint_id": hint_id,
            "needs_consolidation": needs_consolidation,
            "analysis_recommendation": recommendation.value,
            "analysis_results": analysis_results,
            "merge_candidates": merge_candidates if merge_candidates else None,
            "requires_human_engineer": policy.requires_human_engineer,
            "requires_hints_merge": policy.requires_hints_merge
        }
        result = execute_neo4j_query(query, parameters, ignore_exception=True)
        updated = bool(result)
        state_changed = updated and (
            result[0].get("previous_needs_consolidation") != needs_consolidation
            or result[0].get("previous_analysis_recommendation") != recommendation.value
        )
        if state_changed:
            self._notify_hint_state_changed()

        return updated

    def update_hint_agent_hints(self, hint_id: str, agent_hints: Dict[str, str]) -> bool:
        """Replace one Hint's agent hint texts, keeping its prompt, embedding, and analysis state untouched."""
        query = """
            MATCH (h:LlmHint {id: $hint_id})
            SET h.agent_hints = $agent_hints_json
            RETURN h.id AS hint_id
        """
        parameters = {
            "hint_id": hint_id,
            "agent_hints_json": json.dumps(agent_hints, ensure_ascii=False)
        }
        result = execute_neo4j_query(query, parameters, ignore_exception=True)
        updated = bool(result)
        if updated:
            # The injectable content of the Hint changed: ThalabusWeb must reload its hint cache.
            self._notify_hint_state_changed()

        return updated

    def _notify_hint_state_changed(self) -> None:
        """Notify ThalabusWeb after a successful state mutation."""
        token = os.getenv("PLATFORM_API_TOKEN") or os.getenv("THALABUS_TOKEN")
        if not token:
            log(WARNING, "Cannot notify ThalabusWeb about a hint update: the platform API token is not configured.")
            return

        api_url = f"{BLAZOR_PROTOCOL}://{BLAZOR_HOSTNAME}:{BLAZOR_PORT}/v1/hints/changed"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = requests.post(
                api_url,
                headers=headers,
                timeout=5,
                verify=USE_REAL_SSL_CERTIFICATE,
            )
            if response.status_code != 200:
                log(WARNING, f"ThalabusWeb hint-change notification returned HTTP {response.status_code}.")
        except requests.RequestException as ex:
            log(WARNING, f"Could not notify ThalabusWeb about a hint update: {ex}")

    def _ensure_vector_index_exists(self) -> bool:
        """Check if the vector index exists. Returns True if it exists, False otherwise."""
        try:
            # Check if index exists
            check_query = "SHOW INDEXES YIELD name WHERE name = 'hints'"
            result = execute_neo4j_query(check_query, {})
            
            if not result or len(result) == 0:
                if not self._warning_already_logged:
                    log(WARNING, "Vector index 'hints' not found. Index must be created before querying hints.")
                    self._warning_already_logged = True

                embeddings_dimension = EmbeddingsManager.instance().get_embeddings_dimension()
                self.create_hints_vector_indexes(embeddings_dimension=embeddings_dimension)
                return False
            
            return True
        except Exception as e:
            log(WARNING, f"Could not check vector index existence: {e}")
            return False
        

    def get_hint_details(self, hint_id: str) -> Optional[Dict[str, Any]]:
        """Return every stored property of one Hint, or None when the Hint does not exist."""
        query = """
            MATCH (h:LlmHint {id: $hint_id})
            RETURN h.id AS hint_id,
                   h.prompt AS prompt,
                   h.agent_hints AS agent_hints,
                   h.category AS category,
                   h.llm_in AS llm_in,
                   h.needs_consolidation AS needs_consolidation,
                   h.analysis_results AS analysis_results,
                   h.analysis_recommendation AS analysis_recommendation,
                   h.merge_candidates AS merge_candidates,
                   h.requires_human_engineer AS requires_human_engineer,
                   h.requires_hints_merge AS requires_hints_merge
        """
        parameters = {"hint_id": hint_id}
        records = execute_neo4j_query(query, parameters, ignore_exception=True)
        hint_details: Optional[Dict[str, Any]] = None

        if records:
            hint_details = self._map_hint_record(records[0])

        return hint_details

    def get_consolidated_hints(self, exclude_hint_id: str) -> List[Dict[str, Any]]:
        """
        Return every consolidated Hint with all of its stored properties.
        Hints that still require consolidation are excluded, as is the Hint currently being consolidated.
        """
        query = """
            MATCH (h:LlmHint)
            WHERE (h.needs_consolidation IS NULL OR h.needs_consolidation = false)
              AND h.id <> $exclude_hint_id
            RETURN h.id AS hint_id,
                   h.prompt AS prompt,
                   h.agent_hints AS agent_hints,
                   h.category AS category,
                   h.llm_in AS llm_in,
                   h.needs_consolidation AS needs_consolidation,
                   h.analysis_results AS analysis_results,
                   h.analysis_recommendation AS analysis_recommendation,
                   h.merge_candidates AS merge_candidates,
                   h.requires_human_engineer AS requires_human_engineer,
                   h.requires_hints_merge AS requires_hints_merge
        """
        parameters = {"exclude_hint_id": exclude_hint_id}
        records = execute_neo4j_query(query, parameters, ignore_exception=True)
        consolidated_hints: List[Dict[str, Any]] = [
            self._map_hint_record(record)
            for record in records or []
        ]

        return consolidated_hints

    def get_hint_embedding(self, hint_id: str) -> Optional[List[float]]:
        """Return the stored embedding vector of one Hint, or None when the Hint or its embedding does not exist."""
        query = """
            MATCH (h:LlmHint {id: $hint_id})
            RETURN h.embedding AS embedding
        """
        parameters = {"hint_id": hint_id}
        records = execute_neo4j_query(query, parameters, ignore_exception=True)
        embedding: Optional[List[float]] = None

        if records:
            stored_embedding: Any = records[0].get("embedding")
            if isinstance(stored_embedding, list) and stored_embedding:
                embedding = stored_embedding

        return embedding

    def get_hint_similarity_candidates(self, hint_id: str, embedding: List[float], top_n: int, min_score: float) -> List[Dict[str, Any]]:
        """
        Return Hints semantically similar to the given embedding, ordered by descending similarity, for the Hint-versus-Hint evaluation.
        Excludes the compared Hint itself, Hints whose analysis recommendation excludes injection (they can never share an agent context with the compared Hint),
        and Hints whose similarity is below min_score. Each returned Hint carries its 'similarity_score'.
        """
        if embedding is None:
            return []

        if not self._ensure_vector_index_exists():
            return []

        excluded_recommendations: List[str] = [
            recommendation.value
            for recommendation in self.get_default_excluded_hint_recommendations()
            if recommendation.value is not None
        ]

        query: str = """CYPHER 25
            MATCH (h:LlmHint)
              SEARCH h IN (
                VECTOR INDEX hints
                FOR $embeddings
                LIMIT $top_n
              ) SCORE AS score
            WHERE h.id <> $hint_id
              AND (h.analysis_recommendation IS NULL OR NOT h.analysis_recommendation IN $excluded_recommendations)
              AND score >= $min_score
            RETURN h.id AS hint_id,
                   h.prompt AS prompt,
                   h.agent_hints AS agent_hints,
                   h.category AS category,
                   h.llm_in AS llm_in,
                   h.needs_consolidation AS needs_consolidation,
                   h.analysis_results AS analysis_results,
                   h.analysis_recommendation AS analysis_recommendation,
                   h.merge_candidates AS merge_candidates,
                   h.requires_human_engineer AS requires_human_engineer,
                   h.requires_hints_merge AS requires_hints_merge,
                   score
            ORDER BY score DESC
            LIMIT $top_n
        """
        parameters: dict = {
            "embeddings": embedding,
            "top_n": top_n,
            "hint_id": hint_id,
            "excluded_recommendations": excluded_recommendations,
            "min_score": min_score,
        }
        records = execute_neo4j_query(query, parameters, ignore_exception=True)
        candidates: List[Dict[str, Any]] = []

        for record in records or []:
            candidate: Dict[str, Any] = self._map_hint_record(record)
            similarity_score: Any = record.get("score")
            if isinstance(similarity_score, float):
                candidate["similarity_score"] = similarity_score
            candidates.append(candidate)

        return candidates

    def _map_hint_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Convert one Hint database record into a plain dictionary, parsing agent hints."""
        agent_hints_value: Any = record.get("agent_hints")
        agent_hints: Any = {}

        try:
            agent_hints = json.loads(agent_hints_value) if isinstance(agent_hints_value, str) else agent_hints_value

        except (TypeError, ValueError) as ex:
            log(WARNING, f"Failed to parse the agent hints of Hint '{record.get('hint_id')}': {ex}")

        if not isinstance(agent_hints, dict):
            agent_hints = {}

        stored_merge_candidates: Any = record.get("merge_candidates")
        merge_candidates: List[str] = [
            str(candidate_id)
            for candidate_id in stored_merge_candidates
            if candidate_id is not None
        ] if isinstance(stored_merge_candidates, list) else []

        return {
            "id": record.get("hint_id"),
            "prompt": record.get("prompt"),
            "agent_hints": agent_hints,
            "category": record.get("category"),
            "llm_in": record.get("llm_in"),
            "needs_consolidation": record.get("needs_consolidation"),
            "analysis_results": record.get("analysis_results"),
            "analysis_recommendation": record.get("analysis_recommendation"),
            "merge_candidates": merge_candidates,
            "requires_human_engineer": record.get("requires_human_engineer"),
            "requires_hints_merge": record.get("requires_hints_merge"),
        }

log_end_file(__file__)
