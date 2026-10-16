from abc import ABC, abstractmethod
from typing import Optional


class UserInterface(ABC):
    """Interface for user and session identity management."""

    @abstractmethod
    def set_session_id(self, session_id: str) -> None:
        """Set the session identity for explicit background or test context."""
        pass

    @abstractmethod
    def get_session_id(self) -> str:
        """Return the request-local session ID, or explicit background context."""
        pass

    @abstractmethod
    def set_user_id(self, user_id: Optional[str]) -> None:
        """Set the username for explicit background or test context."""
        pass

    @abstractmethod
    def get_user_id(self) -> Optional[str]:
        """Return the request-local username, or explicit background context."""
        pass

    @abstractmethod
    def set_user_uuid(self, user_uuid: Optional[str]) -> None:
        """Set the user UUID for explicit background or test context."""
        pass

    @abstractmethod
    def get_user_uuid(self) -> Optional[str]:
        """Return the request-local user UUID, or explicit background context."""
        pass

    @abstractmethod
    def has_capability(self, user_uuid: Optional[str], capability_id: str) -> bool:
        """Return whether the specified user has the given capability."""
        pass

    @abstractmethod
    def assume_identity_from_request(self) -> bool:
        """Copy the authenticated request identity into the process-global identity; True if assumed."""
        pass

    @abstractmethod
    def assume_identity_from_document_owner(self, document_id: Optional[str]) -> bool:
        """Assume the identity of the user owning the document's dossier; True if assumed."""
        pass

    @abstractmethod
    def assume_identity_from_background_job(self, document_id: Optional[str]) -> bool:
        """Assume the document owner's identity only when the current identity is absent or 'global'."""
        pass
