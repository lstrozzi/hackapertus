from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from datetime import datetime

class CacheEntryInterface(ABC):
    """Abstract interface for cache entry repository operations."""

    @abstractmethod
    def save_cache_entry(self, hash: str, key: str, value: Any, thinking_message: str = None, messages: List[Dict] = None, 
                        tools: list = None, response_format: Dict = None, 
                        structured_output: object = None, guid: str = None, 
                        agent: str = None, category: str = None, purpose: str = None, 
                        goal_id: str = None, date: datetime = None) -> None:
        """
        Save a cache entry with comprehensive metadata.
        
        Args:
            hash: The hash key for the cache entry
            key: The original key
            value: The value to cache
            thinking_message: Optional thinking/reasoning message from LLM
            messages: List of messages for training data
            tools: List of tools used
            response_format: Response format specification
            structured_output: Structured output object
            guid: Plan GUID
            agent: Agent name
            category: Agent cache category
            purpose: Purpose of the cache entry
            goal_id: Goal ID chain
            date: Creation/update date
        """
        pass

    @abstractmethod
    def get_cache_entry_by_hash(self, hash: str, key: str = None) -> Tuple[Any, str, str, str]:
        """
        Retrieve a cache entry by hash.
        
        Args:
            hash: The hash key to lookup
            key: Optional exact key to match
            
        Returns:
            Tuple of (value, hash, guid, thinking_message) or (None, hash, None, None) if not found
        """
        pass

    @abstractmethod
    def update_cache_entry_usage(self, hash: str, guid: str, date: datetime) -> None:
        """
        Update cache entry usage metadata.
        
        Args:
            hash: The hash key of the cache entry
            guid: The GUID of the current execution
            date: The usage date
        """
        pass

    @abstractmethod
    def delete_cache_entry(self, hash: str = None, guid: str = None) -> None:
        """
        Delete cache entry(ies).
        
        Args:
            hash: Optional hash key to delete specific entry
            guid: Optional GUID to delete entries by execution GUID
        """
        pass

    @abstractmethod
    def mark_cache_entries_as_sample(self, guid: str, is_good: bool, auto_generated: bool = False) -> None:
        """
        Mark cache entries as training samples.
        
        Args:
            guid: Plan GUID to mark entries for
            is_good: Whether the entries are good training samples
            auto_generated: Whether the samples were auto-generated
        """
        pass

    @abstractmethod
    def mark_plan_cache_entries_as_sample(self, guid: str, is_good: bool, auto_generated: bool = False) -> bool:
        """Mark cache entries created or executed by a plan as training samples."""
        pass

    @abstractmethod
    def create_feedback_message(self, guid: str, is_good: bool, message: str, plan: str, messages: str) -> bool:
        """Persist a plan feedback message and its captured conversation context."""
        pass

    @abstractmethod
    def delete_plan_cache_entries(self, guid: str) -> None:
        """Delete cache entries created or executed by a plan."""
        pass

    @abstractmethod
    def get_training_data_cache_entries(self, agent: str = None, purpose: str = None,
                                      is_good: bool = True, category: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve cache entries for training data export.
        
        Args:
            agent: Optional agent name to filter by
            purpose: Optional purpose to filter by
            is_good: Whether to get good or bad training samples
            category: Optional cache category to filter by
            
        Returns:
            List of cache entry dictionaries suitable for training data
        """
        pass

    @abstractmethod
    def get_distinct_agents_and_purposes(self, agent: str = None) -> List[Tuple[str, str]]:
        """
        Get distinct combinations of agents and purposes from cache entries.
        
        Returns:
            List of tuples containing (agent, purpose) combinations
        """
        pass