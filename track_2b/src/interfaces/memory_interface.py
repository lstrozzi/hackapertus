from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryInterface(ABC):
    """Interface for user-owned memory node operations."""

    @abstractmethod
    def get_user_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Return memories connected to the specified Neo4j User node."""
        pass

    @abstractmethod
    def add_user_memory(
        self,
        user_id: str,
        classification: str,
        content: str,
        topic: str = "",
        subtopic: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Create a memory only when its owning User node exists."""
        pass

    @abstractmethod
    def update_user_memory(
        self,
        user_id: str,
        memory_id: str,
        classification: str,
        content: str,
        topic: str = "",
        subtopic: str = "",
    ) -> bool:
        """Update a memory only through the specified user's ownership edge."""
        pass