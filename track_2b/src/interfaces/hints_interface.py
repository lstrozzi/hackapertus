from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from src.core.base.enums import HintRecommendation

class HintsInterface(ABC):
    """Abstract interface for hints repository operations."""

    @abstractmethod
    def find_hints_without_embeddings(self) -> Dict[str, Dict[str, str]]:
        """
        Find all hints that don't have embeddings yet.
        Returns:
            Dictionary mapping prompt_text to agent_hints dictionary.
        """
        pass

    @abstractmethod
    def create_hints_vector_indexes(self, embeddings_dimension: int) -> None:
        """
        Create or update the vector indexes for hints in the database.
        
        Args:
            embeddings_dimension: The dimension of the embeddings vector
        """
        pass

    @abstractmethod
    def update_hint(self, prompt_text: str, agent_hints: Dict[str, str], embedding: List[float], suppress_debug_logs: bool = False) -> bool:
        """
        Update or create a user hint for the given prompt_text and agent_hints.
        
        Args:
            prompt_text: The prompt text to associate with the hints
            agent_hints: Dictionary where key=agent_name, value=hint_text. Use "DEFAULT" as the agent key for hints that apply to all agents.
            embedding: The embedding vector for the prompt text
            suppress_debug_logs: Whether to suppress debug logging
            
        Returns:
            True if the hint was successfully created or updated, False otherwise.
        """
        pass

    @abstractmethod
    def delete_hint(self, prompt_text: str) -> bool:
        """
        Delete a hint by its prompt text.
        
        Args:
            prompt_text: The prompt text of the hint to delete
            
        Returns:
            True if the hint was successfully deleted, False otherwise.
        """
        pass

    @abstractmethod
    def get_best_hints_for_agent(self, agent_name: str, user_prompt: str, embedding: List[float], top_n: int, min_score: float, exclude: Optional[List[HintRecommendation]] = None) -> List[Dict[str, str]]:
        """
        Retrieve the most relevant hint based on the agent name and the prompt by vectorizing the user prompt.
        
        Args:
            agent_name: The name of the agent requesting the hint
            user_prompt: The user prompt to find hints for (for logging/debugging purposes)
            embedding: The embedding vector for the user prompt
            top_n: Maximum number of hints to retrieve
            min_score: Minimum similarity score required for a hint to be considered. Use -1 to disable minimum score requirement.
            exclude: List of analysis recommendations; matching hints are never returned.
                     None (default) applies the service's default exclusion (see get_default_excluded_hint_recommendations).
                     An empty list disables the exclusion and returns hints with any recommendation.
                     HintRecommendation.NONE matches hints that have no analysis recommendation.
            
        Returns:
            The best matching hints, ordered best first, each one a dict with the hint 'id'
            and the hint 'text' for the requesting agent, or None if no suitable hint is found.
        """
        pass

    @abstractmethod
    def get_default_excluded_hint_recommendations(self) -> List[HintRecommendation]:
        """Return the analysis recommendations that get_best_hints_for_agent excludes when called without an explicit exclude list.

        These are exactly the recommendations whose policy in HINT_CONSOLIDATION_POLICIES sets exclude_injection=True.
        """
        pass

    @abstractmethod
    def get_pending_consolidation_hints_for_agent(self, agent_name: str) -> List[str]:
        """Return all not-yet-consolidated hints applicable to the agent, without ranking."""
        pass

    @abstractmethod
    def get_hint_ids_by_recommendation(self, recommendation: HintRecommendation) -> List[str]:
        """Return Hint IDs matching the requested analysis recommendation."""
        pass

    @abstractmethod
    def get_hint_ids_requiring_merge(self) -> List[str]:
        """Return the IDs of Hints whose policy set requires_hints_merge and which carry at least one merge candidate.

        These are the Hints the stage-2 merge worker consolidates with their recorded merge candidates.
        """
        pass

    @abstractmethod
    def get_hint_embedding(self, hint_id: str) -> Optional[List[float]]:
        """Return the stored embedding vector of one Hint, or None when the Hint or its embedding does not exist."""
        pass

    @abstractmethod
    def get_hint_similarity_candidates(self, hint_id: str, embedding: List[float], top_n: int, min_score: float) -> List[Dict[str, Any]]:
        """Return Hints semantically similar to the embedding of the given Hint, for the Hint-versus-Hint evaluation.

        Excludes the compared Hint itself and Hints whose analysis recommendation excludes injection,
        because those can never share an agent context with the compared Hint.
        """
        pass

    @abstractmethod
    def update_hint_state(
        self,
        hint_id: str,
        recommendation: HintRecommendation,
        needs_consolidation: Optional[bool],
        analysis_results: Optional[str] = None,
        merge_candidates: Optional[List[str]] = None
    ) -> bool:
        """Set one Hint's analysis recommendation, consolidation flag, analysis results, and merge candidates.

        The follow-up flags requires_human_engineer and requires_hints_merge are derived from the
        recommendation's policy in HINT_CONSOLIDATION_POLICIES and persisted together with it.
        Merge candidates are the Hint IDs the Hint-versus-Hint comparison confirmed as mergeable,
        persisted so the merge worker and the Hints page can present them without re-comparing.

        Args:
            hint_id: The ID of the hint to update
            recommendation: The new analysis recommendation for the hint
            needs_consolidation: The new consolidation flag for the hint
            analysis_results: The new analysis results for the hint (optional)
            merge_candidates: The Hint IDs this Hint may be merged with (optional)

        Returns:
            True if the hint was successfully updated, False otherwise.
        """
        pass

    @abstractmethod
    def update_hint_agent_hints(self, hint_id: str, agent_hints: Dict[str, str]) -> bool:
        """Replace one Hint's agent hint texts, keeping its prompt, embedding, and analysis state untouched.

        Used by the stage-2 merge worker, which persists the merged guidance under the survivor Hint's ID
        before any source Hint's state is changed.

        Args:
            hint_id: The ID of the hint whose agent hint texts are replaced
            agent_hints: The new agent hint texts, keyed by agent name ("DEFAULT" applies to all agents)

        Returns:
            True if the hint was successfully updated, False otherwise.
        """
        pass