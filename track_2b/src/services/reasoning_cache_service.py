from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

from typing import Optional, Any, Dict
import threading

from src.interfaces.reasoning_cache_interface import ReasoningCacheInterface
from src.utils.log import log, DEBUG, INFO, WARNING, ERROR


class ReasoningCacheService(ReasoningCacheInterface):
    """Concrete implementation for reasoning cache operations."""

    def __init__(self):
        self._log_prefix = "ReasoningCacheService"
        self._cache: Dict[str, Any] = {}
        # Lock to protect _cache for multithreaded access
        self._cache_lock = threading.RLock()

    def cache_reasoning_result(self, key: str, value: Any) -> None:
        """
        Cache a reasoning result.
        
        Args:
            key: The cache key
            value: The value to cache
        """
        try:
            if key is None or key.strip() == "":
                log(WARNING, f"{self._log_prefix}.cache_reasoning_result: Invalid key provided")
                return
            # Protect write with lock
            with self._cache_lock:
                self._cache[key] = value
            log(DEBUG, f"{self._log_prefix}.cache_reasoning_result: Cached result for key '{key}'")
            
        except Exception as e:
            log(ERROR, f"{self._log_prefix}.cache_reasoning_result: Error caching result: {e}")

    def get_cached_reasoning_result(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached reasoning result.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value, or None if not found
        """
        try:
            if key is None or key.strip() == "":
                log(WARNING, f"{self._log_prefix}.get_cached_reasoning_result: Invalid key provided")
                return None
            # Protect read with lock
            with self._cache_lock:
                result = self._cache.get(key)
            if result is not None:
                log(DEBUG, f"{self._log_prefix}.get_cached_reasoning_result: Retrieved cached result for key '{key}'")
            else:
                log(DEBUG, f"{self._log_prefix}.get_cached_reasoning_result: No cached result found for key '{key}'")
            
            return result
            
        except Exception as e:
            log(ERROR, f"{self._log_prefix}.get_cached_reasoning_result: Error retrieving cached result: {e}")
            return None

    def invalidate_cache(self, key: Optional[str] = None) -> None:
        """
        Invalidate cache entries.
        
        Args:
            key: Optional specific key to invalidate. If None, clears entire cache.
        """
        try:
            # Protect clear/delete operations with lock
            with self._cache_lock:
                if key is None:
                    self._cache.clear()
                    log(DEBUG, f"{self._log_prefix}.invalidate_cache: Cleared entire cache")
                else:
                    if key in self._cache:
                        del self._cache[key]
                        log(DEBUG, f"{self._log_prefix}.invalidate_cache: Invalidated cache entry for key '{key}'")
                    else:
                        log(DEBUG, f"{self._log_prefix}.invalidate_cache: Cache key '{key}' not found")

        except Exception as e:
            log(ERROR, f"{self._log_prefix}.invalidate_cache: Error invalidating cache: {e}")


# module level singleton accessors (same pattern as other services)
_reasoning_cache_service_instance: Optional[ReasoningCacheInterface] = None

def get_reasoning_cache_service() -> ReasoningCacheInterface:
    global _reasoning_cache_service_instance
    if _reasoning_cache_service_instance is None:
        _reasoning_cache_service_instance = ReasoningCacheService()
    return _reasoning_cache_service_instance

def set_reasoning_cache_service(service: ReasoningCacheInterface) -> None:
    global _reasoning_cache_service_instance
    _reasoning_cache_service_instance = service


log_end_file(__file__)
