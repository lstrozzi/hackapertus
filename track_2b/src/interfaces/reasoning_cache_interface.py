from abc import ABC, abstractmethod
from typing import Optional, Any, Dict


class ReasoningCacheInterface(ABC):
    """Interface for reasoning cache operations."""

    @abstractmethod
    def cache_reasoning_result(self, key: str, value: Any) -> None:
        """
        Cache a reasoning result.
        
        Args:
            key: The cache key
            value: The value to cache
        """
        pass

    @abstractmethod
    def get_cached_reasoning_result(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached reasoning result.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value, or None if not found
        """
        pass

    @abstractmethod
    def invalidate_cache(self, key: Optional[str] = None) -> None:
        """
        Invalidate cache entries.
        
        Args:
            key: Optional specific key to invalidate. If None, clears entire cache.
        """
        pass
