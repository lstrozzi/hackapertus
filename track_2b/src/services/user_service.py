import traceback
from typing import Any, Dict, List, Optional
from flask import g, has_request_context
from src.interfaces.user_interface import UserInterface
from src.utils.log import log, INFO, DEBUG, ERROR

GLOBAL_USER_ID = 'global'
USER_CAPABILITY_QUERY: str = """
    MATCH (u:User:SystemAcl {id: $user_uuid})
          -[:HAS_ROLE]->(:UserRole:SystemAcl)
          -[:HAS_CAPABILITY]->(:UserCapability:SystemAcl {id: $capability_id})
    RETURN count(u) > 0 AS has_capability
"""

# Mirrors Neo4jDocumentRepository.GetResponsibleUserName (C#): walk the document's dossier
# ownership chain to find the real user behind it; never the 'global' service account.
DOCUMENT_OWNER_QUERY: str = """
    MATCH (doc:Document {id: $document_id})<-[:CONTAINS]-(dossier:Dossier)
    WHERE dossier.type <> 'root' AND coalesce(dossier.isDeleted, false) = false
    MATCH (group:Group)-[:OWNS]->(dossier)
    MATCH (user:User)-[:MEMBER_OF]->(group)
    WHERE coalesce(user.id, '') <> 'global' AND coalesce(user.username, '') <> 'global'
    WITH user, group
    ORDER BY coalesce(group.isUserGroup, false) DESC, coalesce(user.username, '') ASC
    RETURN user.id AS user_id, user.username AS user_name
    LIMIT 1
"""

# Plain process-wide globals: each container/process serves a single session for its whole
# lifetime, so identity is process-global, not per-thread/per-context. A regular global (unlike
# contextvars.ContextVar or threading.local) is visible from every thread automatically, with no
# risk of silently losing the value across manually-spawned threads (ThreadPoolExecutor, etc.).
_session_id: str = GLOBAL_USER_ID
_user_id: Optional[str] = GLOBAL_USER_ID
_user_uuid: Optional[str] = GLOBAL_USER_ID


class UserService(UserInterface):
    """Manages user and session identity for the current context."""

    def __init__(self):
        pass

    def set_session_id(self, session_id: str) -> None:
        """Set the session identity for explicit background or test context."""
        try:
            global _session_id
            _session_id = session_id or GLOBAL_USER_ID
            log(DEBUG, f"=========== Setting SESSION_ID = {session_id} ===========")

        except Exception as e:
            log(ERROR, f"Error setting session ID: {e}. Traceback:\n{traceback.format_exc()}")

    def get_session_id(self) -> str:
        """Return the request-local session ID, or explicit background context."""
        result: str = _session_id
        try:
            if has_request_context() and hasattr(g, "session_id"):
                result = g.session_id or GLOBAL_USER_ID

        except Exception as e:
            log(ERROR, f"Error getting session ID: {e}. Traceback:\n{traceback.format_exc()}")

        return result

    def set_user_id(self, user_id: Optional[str]) -> None:
        """Set the username for explicit background or test context."""
        try:
            global _user_id
            _user_id = user_id or GLOBAL_USER_ID
            log(DEBUG, f"=========== Setting USER_ID = {user_id or GLOBAL_USER_ID} ===========")

        except Exception as e:
            log(ERROR, f"Error setting user ID: {e}. Traceback:\n{traceback.format_exc()}")

    def get_user_id(self) -> Optional[str]:  # noqa: ARG002
        """Return the request-local username, or explicit background context."""
        result: Optional[str] = _user_id
        try:
            if has_request_context() and hasattr(g, "username"):
                result = g.username

        except Exception as e:
            log(ERROR, f"Error getting user ID: {e}. Traceback:\n{traceback.format_exc()}")

        return result

    def set_user_uuid(self, user_uuid: Optional[str]) -> None:
        """Set the user UUID for explicit background or test context."""
        try:
            global _user_uuid
            _user_uuid = user_uuid or GLOBAL_USER_ID
            log(DEBUG, f"=========== Setting USER_UUID = {user_uuid or GLOBAL_USER_ID} ===========")

        except Exception as e:
            log(ERROR, f"Error setting user UUID: {e}. Traceback:\n{traceback.format_exc()}")

    def assume_identity_from_request(self) -> bool:
        """Copy the authenticated JWT identity of the current request into the process-global identity.

        Skill pipeline threads (e.g. ingestion worker threads) have no Flask request context and
        would otherwise fall back to 'global'; seeding the process globals makes them inherit the
        real job user. Returns True when a real identity was assumed.
        """
        result: bool = False

        try:
            user_uuid: Optional[str] = getattr(g, "user_uuid", None) if has_request_context() else None
            if user_uuid:
                username: Optional[str] = getattr(g, "username", None)
                session_id: Optional[str] = getattr(g, "session_id", None)
                self.set_user_uuid(user_uuid)
                self.set_user_id(username)
                if session_id:
                    self.set_session_id(session_id)
                result = True
                log(DEBUG, f"Assumed request identity as process identity: user '{username}' ({user_uuid}), session '{session_id}'")

        except Exception as e:
            log(ERROR, f"Error assuming request identity: {e}. Traceback:\n{traceback.format_exc()}")

        return result

    def assume_identity_from_document_owner(self, document_id: Optional[str]) -> bool:
        """Assume the effective identity of the user who owns the document's dossier.

        Used by background jobs (automatic document ingestion) so ACL-gated operations execute as
        the document's real owner instead of the 'global' service account. Returns True when the
        owner could be resolved and assumed.
        """
        result: bool = False

        try:
            if not document_id:
                log(DEBUG, "Cannot assume document owner identity: no document id provided")
            else:
                from src.core.base.InputOutput import execute_neo4j_query
                records: Optional[List[Dict[str, Any]]] = execute_neo4j_query(DOCUMENT_OWNER_QUERY, {"document_id": document_id})
                owner_id: Optional[str] = records[0].get("user_id") if records else None
                if owner_id:
                    owner_name: str = records[0].get("user_name") or owner_id
                    self.set_user_uuid(owner_id)
                    self.set_user_id(owner_name)
                    result = True
                    log(INFO, f"Background job assumed effective identity of document owner '{owner_name}' ({owner_id}) for document '{document_id}'")
                else:
                    log(ERROR, f"Could not resolve a document owner to assume for '{document_id}'; identity remains '{_user_id}'")

        except Exception as e:
            log(ERROR, f"Error assuming document owner identity for '{document_id}': {e}. Traceback:\n{traceback.format_exc()}")

        return result

    def assume_identity_from_background_job(self, document_id: Optional[str]) -> bool:
        """Assume the document owner's identity, but only when no real identity is in effect yet.

        Called by skill entry points: an interactive run already carries an authenticated identity
        (request JWT or seeded process globals) and must keep it, while an automated background run
        still holds 'global' and must adopt the document's real owner for ACL checks to pass.
        """
        result: bool = False

        try:
            effective_user_uuid: Optional[str] = self.get_user_uuid()
            if not effective_user_uuid or effective_user_uuid == GLOBAL_USER_ID:
                result = self.assume_identity_from_document_owner(document_id)

        except Exception as e:
            log(ERROR, f"Error assuming background job identity for '{document_id}': {e}. Traceback:\n{traceback.format_exc()}")

        return result

    def get_user_uuid(self) -> Optional[str]:  # noqa: ARG002
        """Return the request-local user UUID, or explicit background context."""
        result: Optional[str] = _user_uuid
        try:
            if has_request_context() and hasattr(g, "user_uuid"):
                result = g.user_uuid

        except Exception as e:
            log(ERROR, f"Error getting user UUID: {e}. Traceback:\n{traceback.format_exc()}")

        return result

    def has_capability(self, user_uuid: Optional[str], capability_id: str) -> bool:
        """Return whether the specified user has the given capability."""
        has_capability: bool = False

        if user_uuid and capability_id:
            from src.core.base.InputOutput import execute_neo4j_query

            capability_records: Optional[List[Dict[str, Any]]] = execute_neo4j_query(
                USER_CAPABILITY_QUERY,
                {"user_uuid": user_uuid, "capability_id": capability_id},
            )
            has_capability = bool(capability_records and capability_records[0].get("has_capability", False))

        return has_capability


# Singleton instance
_user_service: Optional[UserService] = None


def get_user_service() -> UserInterface:
    """Get or create the singleton UserService instance."""
    try:
        global _user_service
        if _user_service is None:
            _user_service = UserService()

    except Exception as e:
        log(ERROR, f"Error getting user service: {e}. Traceback:\n{traceback.format_exc()}")

    return _user_service
