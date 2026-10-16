from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import traceback
import re
import copy
import time
from typing import Dict, List, Optional
from contextlib import contextmanager

from src.core.base.InputOutput import execute_neo4j_query, send_refresh_document_repository_to_blazor
from src.interfaces.reasoning_cache_interface import ReasoningCacheInterface
from src.services.reasoning_cache_service import get_reasoning_cache_service
from src.utils.log import log, log_init, log_exit, INFO, DEBUG, WARNING, ERROR, FATAL
from src.utils.text_utils import get_elements_by_xpath

# Enumerations – also re-exported here for backward compatibility
from src.core.base.enums import (
    ReasoningStatus, TaskStatus, IssueStatus, IssueSeverity,
    DivisionScope, DivisionExecution, DivisionAction, DivisionVisibility,
)

# Shared XML helpers and constants – also re-exported for backward compatibility
from src.core.base.reasoning.xml_utils import (
    xml_escape, xml_unescape, xml_cdata,
    log_write_context_error, log_read_context_warning,
    LINE_SEPARATOR, REASONING_DOCUMENT_CONTENT_TYPE,
)

# Sub-classes – also re-exported for backward compatibility
from src.core.base.reasoning.ReasoningIssue import ReasoningIssue
from src.core.base.reasoning.ReasoningTask import ReasoningTask
from src.core.base.reasoning.ReasoningResult import ReasoningResult
from src.core.base.reasoning.ReasoningSummary import ReasoningSummary
from src.core.base.reasoning.ReasoningTitle import ReasoningTitle
from src.core.base.reasoning.ReasoningLanguage import ReasoningLanguage
from src.core.base.reasoning.ReasoningDivision import ReasoningDivision


class ReasoningDocument:
    """Document class for managing reasoning structures."""

    def __init__(self, title: str | None = None, id: str | None = None, placeholder_map: Dict[str, str] | None = None, is_template: bool = False) -> None:
        self._rdoc_title: str | None = title
        self._rdoc_id: str | None = id
        self._rdoc_is_template: bool = is_template  # Track if this is a DocumentTemplate node in Neo4j
        self._rdoc_templateid: str | None = None
        self._rdoc_divisions: List[ReasoningDivision] = []
        self._rdoc_associated_template: 'ReasoningDocument' | None = None
        self._rdoc_placeholder_map: Dict[str, str] = placeholder_map if placeholder_map is not None else {}

        # Lock state tracking for write safety (counter for nested contexts)
        self._rdoc_lock_depth: int = 0  # Incremented on entry, decremented on exit
        # Lock state tracking for read safety (ensures fresh data from Neo4j)
        self._rdoc_read_depth: int = 0  # Incremented on entry, decremented on exit
        self._rdoc_parent_document: 'ReasoningDocument' | None = None  # For division/task/issue parent reference

        # Timestamp tracking for optimized read_context (avoid unnecessary reloads)
        self._rdoc_last_update: float | None = None  # Unix timestamp of last Neo4j update

        # Bypass flag for read-only documents (templates) - disables read_context protection
        self._rdoc_bypass_read_context: bool = False

        # Language configuration from <languages> header block
        self._rdoc_languages: List[ReasoningLanguage] = []

        # Shared instruction snippets keyed by id, defined in <common> block
        self._rdoc_common_instructions: Dict[str, str] = {}

    def _check_write_allowed(self) -> bool:
        """Check if write operations are allowed (lock must be held)."""
        if self._rdoc_bypass_read_context:
            log(ERROR, f"Write operation attempted on read-only document (bypass enabled) with ID '{self._rdoc_id}'. This is a programming error.")
            return False
        return self._rdoc_lock_depth > 0

    def _check_read_allowed(self) -> bool:
        """Check if read operations are in a protected read context.

        Returns True if document is in read or write context (both protect against stale reads),
        or if bypass is enabled (for read-only templates).
        """
        return self._rdoc_bypass_read_context or self._rdoc_read_depth > 0 or self._rdoc_lock_depth > 0

    def _needs_reload_from_neo4j(self) -> bool:
        """Check if document needs to be reloaded from Neo4j based on cached or database timestamp.

        Uses cache-first strategy to avoid Neo4j queries when possible.

        Returns:
            True if reload is needed, False if current in-memory state is fresh
        """
        if self._rdoc_id is None or self._rdoc_id == "":
            return False  # Can't reload without ID

        if self._rdoc_last_update is None:
            # Never loaded, need to load
            log(DEBUG, f"Document {self._rdoc_id} has no timestamp, reload needed")
            return True

        try:
            # Check cache first (avoids Neo4j query)
            cached_doc = self._get_cached_document()
            if cached_doc is not None and cached_doc._rdoc_last_update is not None:
                # Compare with cached timestamp
                if cached_doc._rdoc_last_update > self._rdoc_last_update:
                    log(DEBUG, f"Document {self._rdoc_id} is stale compared to cache (cache: {cached_doc._rdoc_last_update}, local: {self._rdoc_last_update})")
                    return True
                else:
                    log(DEBUG, f"Document {self._rdoc_id} is fresh compared to cache (cache: {cached_doc._rdoc_last_update}, local: {self._rdoc_last_update})")
                    return False

            # Cache miss, query Neo4j for latest update timestamp
            log(DEBUG, f"Cache miss for document {self._rdoc_id}, checking Neo4j timestamp")
            # Use correct label and attribute based on whether this is a template
            if self._rdoc_is_template:
                query = """
                    MATCH (d:DocumentTemplate {templateid: $document_id})
                    RETURN COALESCE(d.last_update, 0) as last_update
                """
            else:
                query = """
                    MATCH (d:ReasoningDocument {id: $document_id})
                    RETURN COALESCE(d.last_update, 0) as last_update
                """
            result = execute_neo4j_query(query, {"document_id": self._rdoc_id})

            if not result or len(result) == 0:
                log(WARNING, f"Document {self._rdoc_id} not found in Neo4j during timestamp check")
                return False  # Document doesn't exist, can't reload

            neo4j_timestamp = result[0][0]
            if neo4j_timestamp == 0:
                log(DEBUG, f"Document {self._rdoc_id} has no Neo4j timestamp, assuming fresh")
                return False

            # Compare timestamps
            if neo4j_timestamp > self._rdoc_last_update:
                log(DEBUG, f"Document {self._rdoc_id} is stale (Neo4j: {neo4j_timestamp}, local: {self._rdoc_last_update}), reload needed")
                return True

            log(DEBUG, f"Document {self._rdoc_id} is fresh (Neo4j: {neo4j_timestamp}, local: {self._rdoc_last_update})")
            return False

        except Exception as e:
            log(ERROR, f"Error checking timestamp for document {self._rdoc_id}: {e}")
            return False  # On error, don't reload to avoid infinite loops

    def _is_in_initialization_phase(self) -> bool:
        """Check if document is in initialization phase (not yet persisted to Neo4j).

        Documents created fresh from XML templates but not yet saved don't need write_context().
        We detect this by checking if the document has been assigned an ID yet.
        """
        return self._rdoc_id is None or self._rdoc_id == ""

    def _get_cached_document(self) -> Optional['ReasoningDocument']:
        """Retrieve document from cache if available.

        Returns:
            Cached ReasoningDocument instance or None if not cached
        """
        if self._rdoc_id is None or self._rdoc_id == "":
            return None

        cache_service: ReasoningCacheInterface = get_reasoning_cache_service()
        cached_doc = cache_service.get_cached_reasoning_result(self._rdoc_id)
        return cached_doc

    def _cache_document(self, doc: 'ReasoningDocument') -> None:
        """Store document in cache.

        Args:
            doc: ReasoningDocument instance to cache
        """
        if doc._rdoc_id is None or doc._rdoc_id == "":
            return

        cache_service: ReasoningCacheInterface = get_reasoning_cache_service()
        cache_service.cache_reasoning_result(doc._rdoc_id, doc)
        log(DEBUG, f"Cached document {doc._rdoc_id} (timestamp: {doc._rdoc_last_update})")

    @contextmanager
    def write_context(self, external_lock):
        """Context manager for safe write operations with automatic reload and save.

        This is the ONLY way to perform write operations on a ReasoningDocument.

        Pattern:
            document_lock = get_lock("REASONING_DOCUMENT", doc_id)
            with reasoning_document.write_context(document_lock):
                # Fresh document is automatically loaded from Neo4j
                div = reasoning_document.get_division_by_id(division_id)
                div.set_status(new_status)
                # Automatically saved to Neo4j on context exit

        Args:
            external_lock: The external lock object (from ResourceMgmt.get_lock)

        Yields:
            None

        Process:
            1. Acquires external_lock
            2. Increments _rdoc_lock_depth (enables writes)
            3. Reloads fresh document from Neo4j (discards stale in-memory state) - only on outermost entry
            4. Yields control to caller (who can now perform writes)
            5. Saves updated document to Neo4j - only on outermost exit
            6. Decrements _rdoc_lock_depth (supports nested contexts)
            7. Releases external_lock
        """
        with external_lock:
            try:
                # Increment lock depth for nested context support
                self._rdoc_lock_depth += 1

                # Only reload on outermost context entry
                if self._rdoc_lock_depth == 1:
                    # CRITICAL: Reload fresh state (cache-first, then Neo4j) to avoid stale data
                    # This ensures concurrent modifications don't get overwritten
                    fresh_doc = None

                    # Try cache first
                    cached_doc = self._get_cached_document()
                    if cached_doc is not None:
                        # Check if cache is fresher than current in-memory state
                        if self._rdoc_last_update is None or cached_doc._rdoc_last_update is None or cached_doc._rdoc_last_update >= self._rdoc_last_update:
                            fresh_doc = cached_doc
                            log(DEBUG, f"Using cached document state for write_context on document {self._rdoc_id}")

                    # Cache miss or stale, reload from Neo4j
                    if fresh_doc is None:
                        log(DEBUG, f"Cache miss or stale for document {self._rdoc_id}, loading from Neo4j")
                        fresh_doc = ReasoningDocument.from_id(self._rdoc_id, is_template=self._rdoc_is_template, placeholder_map=self._rdoc_placeholder_map)
                        if fresh_doc is not None:
                            # Cache the freshly loaded document
                            self._cache_document(fresh_doc)

                    if fresh_doc is not None:
                        # Preserve template reference and programmatically-set templateid
                        # (neither is reliably round-tripped through Neo4j in all cases)
                        preserved_template = self._rdoc_associated_template
                        preserved_templateid = self._rdoc_templateid
                        # Replace in-memory state with fresh state (from cache or Neo4j)
                        self._rdoc_title = fresh_doc._rdoc_title
                        self._rdoc_templateid = fresh_doc._rdoc_templateid
                        # CRITICAL: Deep-copy divisions to prevent race conditions when multiple threads
                        # share cached document divisions. Without deep-copy, threads would share division
                        # objects and overwrite each other's _rd_parent_document references.
                        self._rdoc_divisions = copy.deepcopy(fresh_doc._rdoc_divisions)
                        self._rdoc_associated_template = preserved_template  # Restore template reference
                        # Restore programmatically-set templateid if fresh doc doesn't have one
                        # (set_template_id() is called outside write_context to fix missing templateids;
                        # without this, the fix would be silently discarded by the reload above)
                        if preserved_templateid and not self._rdoc_templateid:
                            self._rdoc_templateid = preserved_templateid
                        self._rdoc_last_update = fresh_doc._rdoc_last_update  # Preserve timestamp
                        # Preserve lock state and placeholder map
                        # Re-link all nested objects to this document for write checking
                        for div in self._rdoc_divisions:
                            div._rd_parent_document = self
                            for task in div._rd_tasks:
                                task._rt_parent_division = div
                            for result in div._rd_results:
                                result._rr_parent_division = div
                            for issue in div._rd_issues:
                                issue._ri_parent_division = div
                        log(DEBUG, f"Reloaded fresh document state for document {self._rdoc_id}")
                    else:
                        log(WARNING, f"Failed to reload document {self._rdoc_id} - proceeding with current state")
                else:
                    log(DEBUG, f"Nested write_context detected (depth={self._rdoc_lock_depth}), skipping reload")

                # Yield control to caller who can now perform writes
                yield

                # Only save on outermost context exit
                if self._rdoc_lock_depth == 1:
                    # CRITICAL: Save modified state back to Neo4j (write-through policy)
                    success = self.update_document_in_neo4j()
                    if success:
                        log(DEBUG, f"Successfully saved document {self._rdoc_id} to Neo4j after write operations")
                        # Write-through: Update cache with latest saved state
                        self._cache_document(self)
                        # Notify Blazor to refresh this specific document
                        send_refresh_document_repository_to_blazor([self._rdoc_id])
                    else:
                        log(ERROR, f"Failed to save document {self._rdoc_id} to Neo4j after write operations")
                else:
                    log(DEBUG, f"Nested write_context exit (depth={self._rdoc_lock_depth}), deferring save to outer context")

            finally:
                # Decrement lock depth (supports nested contexts)
                self._rdoc_lock_depth -= 1

    @contextmanager
    def read_context(self):
        """Context manager for safe read operations with optimized reload from Neo4j.

        This ensures getters always read the latest state from the database.
        Uses timestamp-based optimization to avoid unnecessary reloads.

        Pattern:
            with reasoning_document.read_context():
                # Fresh document is automatically loaded from Neo4j (only if stale)
                div = reasoning_document.get_division_by_id(division_id)
                status = div.status()  # Reads latest status
                # Document automatically released on context exit

        Yields:
            None

        Process:
            1. Increments _rdoc_read_depth (enables read protection checks)
            2. Checks timestamp - only reloads if Neo4j has newer data (optimization)
            3. Yields control to caller (who can now safely read)
            4. Decrements _rdoc_read_depth (supports nested contexts)
        """
        try:
            # Increment read depth for nested context support
            self._rdoc_read_depth += 1

            # Only check/reload on outermost context entry
            if self._rdoc_read_depth == 1:
                # OPTIMIZATION: Check if reload is needed (cache-first, then timestamp-based)
                log(DEBUG, f"read_context entry: checking if reload needed for {self._rdoc_id}. Current divisions: {len(self._rdoc_divisions)}, timestamp: {self._rdoc_last_update}")
                if self._needs_reload_from_neo4j():
                    fresh_doc = None

                    # Try cache first
                    cached_doc = self._get_cached_document()
                    if cached_doc is not None:
                        # Check if cache is fresher than current in-memory state
                        if self._rdoc_last_update is None or cached_doc._rdoc_last_update is None or cached_doc._rdoc_last_update >= self._rdoc_last_update:
                            fresh_doc = cached_doc
                            log(DEBUG, f"Using cached document state for read_context on document {self._rdoc_id}")

                    # Cache miss or stale, reload from Neo4j
                    if fresh_doc is None:
                        log(DEBUG, f"Cache miss or stale for document {self._rdoc_id}, loading from Neo4j")
                        fresh_doc = ReasoningDocument.from_id(self._rdoc_id, is_template=self._rdoc_is_template, placeholder_map=self._rdoc_placeholder_map)
                        if fresh_doc is not None:
                            # Cache the freshly loaded document
                            self._cache_document(fresh_doc)

                    if fresh_doc is not None:
                        # Preserve template reference (not stored in Neo4j, set programmatically)
                        preserved_template = self._rdoc_associated_template
                        # Replace in-memory state with fresh state (from cache or Neo4j)
                        self._rdoc_title = fresh_doc._rdoc_title
                        self._rdoc_templateid = fresh_doc._rdoc_templateid
                        # CRITICAL: Deep-copy divisions to prevent race conditions when multiple threads
                        # share cached document divisions. Without deep-copy, threads would share division
                        # objects and overwrite each other's _rd_parent_document references.
                        self._rdoc_divisions = copy.deepcopy(fresh_doc._rdoc_divisions)
                        self._rdoc_associated_template = preserved_template  # Restore template reference
                        self._rdoc_last_update = fresh_doc._rdoc_last_update  # Update timestamp
                        # Re-link all nested objects to this document for read checking
                        for div in self._rdoc_divisions:
                            div._rd_parent_document = self
                            for task in div._rd_tasks:
                                task._rt_parent_division = div
                            for result in div._rd_results:
                                result._rr_parent_division = div
                            for issue in div._rd_issues:
                                issue._ri_parent_division = div
                        log(DEBUG, f"Reloaded fresh document state for read operation on document {self._rdoc_id}")
                    else:
                        log(WARNING, f"Failed to reload document {self._rdoc_id} for read - proceeding with current state")
                else:
                    log(DEBUG, f"Document {self._rdoc_id} is up-to-date, skipping reload (cache/timestamp optimization)")
            else:
                log(DEBUG, f"Nested read_context detected (depth={self._rdoc_read_depth}), skipping reload")

            # Yield control to caller who can now safely read
            yield

        finally:
            # Decrement read depth (supports nested contexts)
            self._rdoc_read_depth -= 1

    def id(self) -> str:
        """Get the document ID (metadata, never changes)."""
        return self._rdoc_id

    def set_id(self, id: str) -> None:
        """Set the document ID."""
        self._rdoc_id = id

    def template_id(self) -> str:
        """Get the template ID if it exists (structural metadata)."""
        return self._rdoc_templateid

    def set_template_id(self, template_id: str) -> None:
        """Set the template ID for this document."""
        self._rdoc_templateid = template_id

    def languages(self) -> List[ReasoningLanguage]:
        """Get the list of language configuration entries from the <languages> header block."""
        return self._rdoc_languages

    def set_languages(self, languages: List[ReasoningLanguage]) -> None:
        """Replace the complete list of language entries."""
        self._rdoc_languages = list(languages) if languages is not None else []

    def get_language_master(self) -> 'ReasoningLanguage | None':
        """Return the master ReasoningLanguage entry, or None if no languages are configured."""
        for lang in self._rdoc_languages:
            if lang.master():
                return lang
        return self._rdoc_languages[0] if self._rdoc_languages else None

    def get_language_for_lang(self, lang: str) -> 'ReasoningLanguage | None':
        """Return the ReasoningLanguage entry for the given ISO code, or None."""
        lang_lower = (lang or "").lower().strip()
        for entry in self._rdoc_languages:
            if entry.lang() == lang_lower:
                return entry
        return None

    def common_instructions(self) -> Dict[str, str]:
        """Return the shared instruction snippets defined in the <common> block.

        Returns:
            Dict mapping instruction id to its text content.
        """
        return self._rdoc_common_instructions

    def set_common_instruction(self, id: str, text: str) -> None:
        """Add or replace a common instruction snippet.

        Args:
            id:   The unique identifier for the instruction snippet.
            text: The instruction text (may contain placeholders).
        """
        self._rdoc_common_instructions[id] = text

    def resolve_common_instruction_includes(self, text: str, instruction_data: dict) -> str:
        """Prepend any referenced <common> instruction texts to *text*.

        Looks up each id listed in instruction_data["include_common_instructions_ids"]
        (comma-separated) in this document's _rdoc_common_instructions dict, with a
        fallback to the associated template's common instructions when set.  Placeholder
        translations from _rdoc_placeholder_map are applied to the common snippets.

        Args:
            text:             Raw instruction text already retrieved for the status.
            instruction_data: The dict stored under the status key in _rd_instructions.

        Returns:
            The text with any common snippets prepended, separated by a blank line.
        """
        if not isinstance(instruction_data, dict):
            return text
        include_ids_str = instruction_data.get("include_common_instructions_ids", "")
        if not include_ids_str:
            return text
        common_parts = []
        for cid in [x.strip() for x in include_ids_str.split(",") if x.strip()]:
            common_text = self._rdoc_common_instructions.get(cid, "")
            # Fallback: look in the associated template document
            if not common_text and self._rdoc_associated_template is not None:
                common_text = self._rdoc_associated_template._rdoc_common_instructions.get(cid, "")
            if common_text:
                # Apply placeholder translations
                for ph, val in self._rdoc_placeholder_map.items():
                    common_text = common_text.replace(ph, val)
                common_parts.append(common_text)
        if common_parts:
            prefix = "\n\n".join(common_parts)
            return prefix + ("\n\n" + text if text else "")
        return text

    def title(self) -> str:
        """Get the document title."""
        return self._rdoc_title

    def set_title(self, title: str) -> None:
        """Set the document title."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDocument', 'set_title', self._rdoc_id, 'set title'))
            return
        # Replace all HTML <br> tags with XML-compliant <br/>
        if title:
            # Replace <br>, <br/>, <br > with <br/>
            title = re.sub(r"<br\s*/?>", LINE_SEPARATOR, title)
        self._rdoc_title = title

    def content_type(self) -> str:
        """Get the content type of the document (constant metadata)."""
        return REASONING_DOCUMENT_CONTENT_TYPE

    def divisions(self) -> List['ReasoningDivision']:
        """Get the list of divisions."""
        return self._rdoc_divisions

    def associated_template(self) -> 'ReasoningDocument':
        """Get the associated template if it exists"""
        return self._rdoc_associated_template

    def placeholder_map(self) -> Dict[str, str]:
        """Get the placeholder translation map (structural metadata)."""
        return self._rdoc_placeholder_map

    def set_placeholder_map(self, placeholder_map: Dict[str, str]) -> None:
        """Set the placeholder translation map.

        Args:
            placeholder_map: Dictionary mapping placeholder names (e.g., "%LANGUAGE%") to their target values (e.g., "Italian")
        """
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDocument', 'set_placeholder_map', self._rdoc_id, 'set placeholder_map'))
            return
        self._rdoc_placeholder_map = placeholder_map if placeholder_map is not None else {}

    def set_bypass_read_context(self, bypass: bool = True) -> None:
        """Enable or disable bypass of read_context protection.

        Use this for read-only documents (e.g., templates) that don't need
        Neo4j synchronization protection, improving performance.

        Args:
            bypass: True to bypass read_context checks, False to enforce them (default: True)
        """
        self._rdoc_bypass_read_context = bypass
        # Propagate to all divisions so they also bypass read checks
        for div in self._rdoc_divisions:
            div._rd_bypass_read_context = bypass

    def set_associate_template(self, template: 'ReasoningDocument') -> None:
        """Associate this template with another ReasoningTemplate"""
        self._rdoc_associated_template = template

    def add_division(self, division: ReasoningDivision) -> None:
        """Add a division to the template"""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDocument', 'add_division', self._rdoc_id, 'add division'))
            return
        division._rd_parent_document = self
        self._rdoc_divisions.append(division)

    def get_division_by_id(self, division_id: str) -> ReasoningDivision:
        """Get a specific division by its ID"""
        if not self._check_read_allowed():
            log(WARNING, log_read_context_warning('ReasoningDocument', 'get_division_by_id', self._rdoc_id))
        for division in self._rdoc_divisions:
            if division._rd_id == division_id:
                return division
        return None

    def get_next_division(self, current_division_id: str) -> ReasoningDivision:
        """Get the next division after the current one"""
        # Lazy read_context pattern: check if already protected, only wrap if needed
        if self._check_read_allowed():
            for i, division in enumerate(self._rdoc_divisions):
                if division._rd_id == current_division_id:
                    # Return the next division if it exists
                    if i + 1 < len(self._rdoc_divisions):
                        return self._rdoc_divisions[i + 1]
                    else:
                        return None
        else:
            with self.read_context():
                for i, division in enumerate(self._rdoc_divisions):
                    if division._rd_id == current_division_id:
                        # Return the next division if it exists
                        if i + 1 < len(self._rdoc_divisions):
                            return self._rdoc_divisions[i + 1]
                        else:
                            return None

    def get_incomplete_divisions(self) -> List[ReasoningDivision]:
        """Get all divisions that are not complete"""
        # Lazy read_context pattern: check if already protected, only wrap if needed
        if self._check_read_allowed():
            return [division for division in self._rdoc_divisions if not division.is_complete()]
        else:
            with self.read_context():
                return [division for division in self._rdoc_divisions if not division.is_complete()]

    def get_division_with_public_summary(self) -> ReasoningDivision:
        """Get the division with scope=Public in the document, or in alternative the last division if there isn't any """
        division_with_public_summary = None

        if self._rdoc_divisions:
            for division in reversed(self._rdoc_divisions):
                # Lazy read_context pattern: check if already protected, only wrap if needed
                if self._check_read_allowed():
                    div_scope = division.scope()
                else:
                    with self.read_context():
                        div_scope = division.scope()
                if div_scope == DivisionScope.PUBLIC:
                    division_with_public_summary = division
                    break

        if division_with_public_summary is None and self._rdoc_divisions:
            # No division with public scope found, return the last division as a fallback
            division_with_public_summary = self._rdoc_divisions[-1]

        return division_with_public_summary

    def get_first_incomplete_division(self) -> ReasoningDivision:
        """Get the first division that is not complete"""
        incomplete = self.get_incomplete_divisions()
        return incomplete[0] if incomplete else None

    def find_division_by_task_id(self, task_id: str) -> ReasoningDivision:
        """Find a division by a specific task ID"""
        # Lazy read_context pattern: check if already protected, only wrap if needed
        if self._check_read_allowed():
            division_id = ReasoningTask.strip_division_id_from_task_id(task_id)
            stripped_task_id = ReasoningTask.strip_id(task_id)
            for division in self._rdoc_divisions:
                if division_id is not None and division._rd_id != division_id:
                    # Not the right division, continue
                    continue

                if division._rd_tasks is not None:
                    # Check if the task ID matches any task in this division
                    if any(ReasoningTask.strip_id(task._rt_task_id) == stripped_task_id for task in division._rd_tasks):
                        return division
            return None
        else:
            with self.read_context():
                division_id = ReasoningTask.strip_division_id_from_task_id(task_id)
                stripped_task_id = ReasoningTask.strip_id(task_id)
                for division in self._rdoc_divisions:
                    if division_id is not None and division._rd_id != division_id:
                        # Not the right division, continue
                        continue

                    if division._rd_tasks is not None:
                        # Check if the task ID matches any task in this division
                        if any(ReasoningTask.strip_id(task._rt_task_id) == stripped_task_id for task in division._rd_tasks):
                            return division
                return None

    def strip_instructions(self) -> None:
        """Strip the 'instructions' attribute from all divisions"""
        # Lazy read_context pattern: check if already protected, only wrap if needed
        if self._check_read_allowed():
            for division in self._rdoc_divisions:
                if hasattr(division, '_rd_instructions'):
                    del division._rd_instructions
        else:
            with self.read_context():
                for division in self._rdoc_divisions:
                    if hasattr(division, '_rd_instructions'):
                        del division._rd_instructions

    def get_divisions_by_status(self, status: str) -> List[ReasoningDivision]:
        """Get all divisions with a specific status (str or ReasoningStatus)"""
        if isinstance(status, str):
            status_enum = ReasoningStatus(status)
        else:
            status_enum = status
        return [division for division in self._rdoc_divisions if division._rd_status == status_enum]

    def get_instructions_for_division(self, division_id: str) -> Dict[str, Dict[str, str]]:
        """Get all instructions for a specific division by its ID"""
        if not self._check_read_allowed():
            log(WARNING, log_read_context_warning('ReasoningDocument', 'get_instructions_for_division', self._rdoc_id))
        instructions = None
        # Ensure read protection for get_division_by_id call
        if self._check_read_allowed():
            # Already in a read context, call directly
            division = self.get_division_by_id(division_id)
        else:
            # Not in a read context, use read_context
            with self.read_context():
                division = self.get_division_by_id(division_id)
        if division and hasattr(division, '_rd_instructions'):
            instructions = division._rd_instructions
        if (instructions is None or len(instructions) == 0) and (self._rdoc_associated_template is not None):
            # Use read_context to safely read from template document
            with self._rdoc_associated_template.read_context():
                instructions = self._rdoc_associated_template.get_instructions_for_division(division_id)
        return instructions

    def get_instructions_for_division_current_status(self, division_id: str, include_common_instructions: bool = True) -> str:
        """Get instructions for a specific division's current status.

        Args:
            division_id: The ID of the division.
            include_common_instructions: When True (default), common instruction snippets
                referenced via include_common_instructions_ids are prepended to the returned
                text.  Pass False to retrieve the raw division-level instruction text only.
        """
        # Lazy read_context pattern: check if already protected, only wrap if needed
        if self._check_read_allowed():
            division = self.get_division_by_id(division_id)
        else:
            with self.read_context():
                division = self.get_division_by_id(division_id)

        if division and hasattr(division, '_rd_instructions'):
            text = division.get_instructions_for_current_status(self._rdoc_placeholder_map)
            if include_common_instructions:
                status_key = division._rd_status.value
                instruction_data = division._rd_instructions.get(status_key, {})
                return self.resolve_common_instruction_includes(text, instruction_data)
            return text
        return ""

    def get_recommended_tools_for_division(self, division_id: str, status: str = None) -> str:
        """Get recommended tools for a specific division and status"""
        # Lazy read_context pattern: check if already protected, only wrap if needed
        if self._check_read_allowed():
            division = self.get_division_by_id(division_id)
        else:
            with self.read_context():
                division = self.get_division_by_id(division_id)
        if division and hasattr(division, '_rd_instructions'):
            if status is None:
                return division.get_recommended_tools_for_current_status(self._rdoc_placeholder_map)
            else:
                return division.get_recommended_tools_for_status(status, self._rdoc_placeholder_map)
        return ""

    def get_recommended_plan_for_division(self, division_id: str, status: str = None) -> str:
        """Get recommended plan for a specific division and status"""
        # Lazy read_context pattern: check if already protected, only wrap if needed
        if self._check_read_allowed():
            division = self.get_division_by_id(division_id)
        else:
            with self.read_context():
                division = self.get_division_by_id(division_id)
        if division and hasattr(division, '_rd_instructions'):
            if status is None:
                return division.get_recommended_plan_for_current_status(self._rdoc_placeholder_map)
            else:
                return division.get_recommended_plan_for_status(status, self._rdoc_placeholder_map)
        return ""

    def get_action_for_division(self, division_id: str, status: str = None) -> DivisionAction:
        """
        Get the action for a specific division and status.

        Args:
            division_id: The ID of the division to get the action for
            status: Optional status to filter the action by, defaults to None (current status)

        Returns:
            DivisionAction enum value representing the action for the division
        """
        # Ensure read protection for get_division_by_id call
        if self._check_read_allowed():
            # Already in a read context, call directly
            division = self.get_division_by_id(division_id)
        else:
            # Not in a read context, use read_context
            with self.read_context():
                division = self.get_division_by_id(division_id)
        if division is None:
            return DivisionAction.NONE

        # Actions are template-defined behaviour (not stored as runtime state).
        # Always prefer the associated template when one is set, because write_context()
        # reloads the runtime document from Neo4j which may restore _rd_instructions
        # without the 'action' attribute, causing the template fallback to be skipped.
        if self._rdoc_associated_template is not None:
            with self._rdoc_associated_template.read_context():
                template_division = self._rdoc_associated_template.get_division_by_id(division_id)
            if template_division is not None and hasattr(template_division, '_rd_instructions'):
                effective_status = status
                if effective_status is None:
                    effective_status = division._rd_status.value
                instruction_data = template_division._rd_instructions.get(effective_status, {})
                action_value = instruction_data.get("action", DivisionAction.NONE.value)
                return DivisionAction(action_value)

        # Fallback: no template associated – read from the runtime document's own instructions
        if hasattr(division, '_rd_instructions'):
            if status is None:
                status = division._rd_status.value
            instruction_data = division._rd_instructions.get(status, {})
            action_value = instruction_data.get("action", DivisionAction.NONE.value)
            return DivisionAction(action_value)
        return DivisionAction.NONE

    def find_incomplete_tasks(self, division_id: str) -> List[ReasoningTask]:
        """
        Find all incomplete tasks within a specific division in the reasoning template.

        Args:
            division_id: The ID of the division to search within

        Returns:
            A list of ReasoningTask objects that are incomplete within the specified division
        """
        # Get the division by ID from the reasoning template
        # Ensure read protection for get_division_by_id call
        if self._check_read_allowed():
            # Already in a read context, call directly
            division = self.get_division_by_id(division_id)
        else:
            # Not in a read context, use read_context
            with self.read_context():
                division = self.get_division_by_id(division_id)

        if division is None:
            return []

        # Return all incomplete tasks from the division
        return division.get_incomplete_tasks()

    def find_tasks_by_status(self, division_id: str, status_list: List[str | TaskStatus]) -> List[ReasoningTask]:
        """
        Find all tasks within a specific division that match any of the provided statuses.
        Args:
            division_id: The ID of the division to search within
            status_list: List of status values (str or TaskStatus) to search for
        Returns:
            A list of ReasoningTask objects that match any of the specified statuses within the division
        """
        if not status_list:
            return []
        # Ensure read protection for get_division_by_id call
        if self._check_read_allowed():
            # Already in a read context, call directly
            division = self.get_division_by_id(division_id)
        else:
            # Not in a read context, use read_context
            with self.read_context():
                division = self.get_division_by_id(division_id)
        if division is None:
            return []
        matching_tasks = []
        for status in status_list:
            if isinstance(status, str):
                status_enum = TaskStatus(status)
            else:
                status_enum = status
            matching_tasks.extend(division.get_tasks_by_status(status_enum))

        # Remove duplicates while preserving order
        seen_task_ids = set()
        unique_tasks = []
        for task in matching_tasks:
            if task.id() not in seen_task_ids:
                unique_tasks.append(task)
                seen_task_ids.add(task.id())

        return unique_tasks

    def reset_reasoning_status(self) -> bool:
        # resets the reasoning status of all divisions to OPEN and returns True,
        # unless they are already OPEN in which case nothing is performed and False is returned
        changed = False

        for division in self._rdoc_divisions:
            # Lazy read_context pattern: check if already protected, only wrap if needed
            if self._check_read_allowed():
                div_status = division.status()
            else:
                with self.read_context():
                    div_status = division.status()
            if div_status != ReasoningStatus.RS_OPEN:
                division.set_status(ReasoningStatus.RS_OPEN)
                changed = True

        return changed

    def to_dict(self) -> dict:
        """Convert the template to a dictionary representation"""
        return {
            "title": self._rdoc_title,
            "template_id": self._rdoc_id,
            "divisions": [division.to_dict() for division in self._rdoc_divisions]
        }

    def to_xml(self, escape: bool = False, remove_information: Dict[str, bool] = None, full_information_for_division_id: str = None) -> str:
        """
        Convert the ReasoningDocument to XML format with conditional content removal.
        Args:
            escape (bool): Whether to escape XML special characters (&, <, >, ", '). Defaults to False.
            remove_information (Dict[str, bool], optional): Dictionary specifying which content types to exclude.
                Supported keys:
                - "remove_tasks": Exclude tasks sections
                - "remove_results": Exclude results sections
                - "remove_prompt": Exclude prompt sections
                - "remove_keywords": Exclude keywords/metadata sections
                - "remove_issues": Exclude issues sections
                - "remove_recommendations": Exclude recommendations sections
                - "remove_summary": Exclude summary sections
                - "remove_instructions": Exclude instructions sections
                Defaults to None (include all content).
        Returns:
            str: XML representation of the document wrapped in <root> tags for XML compliance.
        """
        html_parts = []
        html_parts.append('<root>')
        html_parts.append('<header>')
        html_parts.append(f'    <title>{xml_escape(self._rdoc_title, escape)}</title>')
        if self._rdoc_associated_template is not None:
            html_parts.append(f'    <template id="{xml_escape(self._rdoc_associated_template._rdoc_id, escape)}"></template>')
        elif hasattr(self, '_rdoc_templateid') and self._rdoc_templateid is not None and len(self._rdoc_templateid.strip()) > 0:
            html_parts.append(f'    <template id="{xml_escape(self._rdoc_templateid, escape)}"></template>')
        elif self._rdoc_is_template and self._rdoc_id is not None:
            # For template documents only, use the document ID as the template ID
            html_parts.append(f'    <template id="{xml_escape(self._rdoc_id, escape)}"></template>')
        # Note: For regular ReasoningDocuments without template info, we don't output a template tag
        # This prevents self-referencing bugs where document ID gets used as template ID
        if self._rdoc_languages:
            html_parts.append('    <languages>')
            for lang_entry in self._rdoc_languages:
                html_parts.append(lang_entry.to_xml(escape))
            html_parts.append('    </languages>')
        html_parts.append('</header>')
        html_parts.append('')

        # Emit <common> block if there are shared instruction snippets
        if self._rdoc_common_instructions:
            html_parts.append('<common>')
            for ci_id, ci_text in self._rdoc_common_instructions.items():
                html_parts.append(f'    <instructions id="{xml_escape(ci_id, escape)}">')
                html_parts.append(f'        {xml_cdata(ci_text)}')
                html_parts.append('    </instructions>')
            html_parts.append('</common>')
            html_parts.append('')

        for i, division in enumerate(self._rdoc_divisions, 1):
            # we remove information according to the input argument, but if the division is the one we want full information for, we include all
            remove_information_for_division = remove_information if division.id() != full_information_for_division_id else None
            html_parts.append(division.to_xml(escape, remove_information_for_division))
            if i < len(self._rdoc_divisions):
                html_parts.append('')

        html_parts.append('</root>')
        return '\n'.join(html_parts)

    def update_document_in_neo4j(self) -> bool:
        try:
            merge_query = ""
            template_id = None

            if self is None:
                log(WARNING, "Cannot update document: self is None")
                return False

            # Update timestamp for optimized read_context
            current_timestamp = time.time()
            self._rdoc_last_update = current_timestamp

            # Convert the reasoning template back to HTML
            updated_html = self.to_xml()
            if self.associated_template() is not None:
                template_id = self.associated_template().id()
                merge_query = """
                    WITH d
                    MATCH (t:DocumentTemplate {templateid: $template_id})
                    MERGE (d)-[:BASED_ON]->(t)
                """
            else:
                template_id = None

            # Update the document in Neo4j with timestamp
            update_query = f"""
                MERGE (d:Document:ReasoningDocument {{id: $document_id}})
                SET d.enhanced_text = $enhanced_text,
                    d.last_update = $last_update
                {merge_query}
                RETURN d.id AS document_id
            """
            parameters = {
                "document_id": self.id(),
                "template_id": template_id,
                "enhanced_text": updated_html,
                "last_update": current_timestamp
            }

            log(DEBUG, f"Updating ReasoningDocument '{self.id()}' in Neo4j...")
            result = execute_neo4j_query(update_query, parameters)

            if result is not None:
                log(DEBUG, f"Successfully updated ReasoningDocument '{self.id()}' in Neo4j")
                return True
            else:
                log(ERROR, f"Failed to update ReasoningDocument '{self.id()}' in Neo4j")
                return False

        except Exception as e:
            stack_trace = traceback.format_exc()
            log(ERROR, f"Error updating document '{self.id()}' in Neo4j: {e}\nStack trace:\n{stack_trace}")
            return False

    @classmethod
    def _extract_inner_html(cls, xml_content: str, xpath: str) -> str:
        """Extract the inner HTML content of an element, preserving nested tags and CDATA"""
        try:
            from lxml import html, etree
            from src.utils.text_utils import sanitize_xml_for_parsing

            # Fix eventually invalid xpath by adding <root> tag if necessary
            if not "<root>" in xml_content:
                content = f"<root>{xml_content}</root>"
            else:
                content = xml_content

            # For CDATA content, use a regex approach to extract the content directly
            # Convert XPath to a simple element finder
            element_name = xpath.split('//')[-1].split('[')[0]  # Get element name from xpath
            div_id = ""
            attribute_filter = ""
            position_index = None  # Track if a positional index like [1], [2], etc. is specified

            if "@id='" in xpath:
                div_id = xpath.split("@id='")[1].split("'")[0]

            # Extract positional index if present (e.g., [1], [2] in "recommendation[1]")
            if '[' in element_name or ('[' in xpath and not '@' in xpath.split('[')[-1].split(']')[0]):
                # Check for numeric positional index
                bracket_content = xpath.split('[')[-1].split(']')[0]
                if bracket_content.isdigit():
                    position_index = int(bracket_content) - 1  # Convert to 0-based index
                    # Remove the positional index from element name if it's there
                    if '[' in element_name:
                        element_name = element_name.split('[')[0]

            # Extract attribute filters from XPath (e.g., @for_task_id='t-001', @id='issue-1')
            if '[' in xpath and '@' in xpath:
                filter_part = xpath.split('[')[-1].split(']')[0]  # Get content between [ and ]
                if '@' in filter_part and '=' in filter_part:
                    attr_name = filter_part.split('@')[1].split('=')[0]
                    attr_value = filter_part.split("'")[1] if "'" in filter_part else ""
                    if attr_value:
                        # Escape special regex characters in the attribute value
                        escaped_attr_value = re.escape(attr_value)
                        attribute_filter = f'{attr_name}="{escaped_attr_value}"'

            # Build regex pattern to find the element and extract its CDATA content
            if div_id and attribute_filter:
                # Include both division id and attribute filter
                pattern = rf'<division[^>]*id="{re.escape(div_id)}"[^>]*>.*?<{element_name}[^>]*{attribute_filter}[^>]*>\s*<!\[CDATA\[(.*?)\]\]>\s*</{element_name}>'
            elif div_id:
                # Only division id, no attribute filter (for elements like summary, prompt)
                pattern = rf'<division[^>]*id="{re.escape(div_id)}"[^>]*>.*?<{element_name}[^>]*>\s*<!\[CDATA\[(.*?)\]\]>\s*</{element_name}>'
            elif attribute_filter:
                # Only attribute filter, no division id
                pattern = rf'<{element_name}[^>]*{attribute_filter}[^>]*>\s*<!\[CDATA\[(.*?)\]\]>\s*</{element_name}>'
            else:
                # No filters, just element name
                pattern = rf'<{element_name}[^>]*>\s*<!\[CDATA\[(.*?)\]\]>\s*</{element_name}>'

            # Use findall to get all matches when position_index is specified
            if position_index is not None:
                matches = re.findall(pattern, content, re.DOTALL)
                if matches and position_index < len(matches):
                    return matches[position_index]
            else:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    return match.group(1)

            # Fallback to the original XML parsing approach for non-CDATA content
            # Use our sanitizer to handle CDATA and other XML issues
            content = sanitize_xml_for_parsing(content)

            # Parse as XML (not HTML) to preserve CDATA content
            tree = etree.fromstring(content.encode('utf-8'))

            # Find elements using XPath
            elements = tree.xpath(xpath)

            if elements:
                element = elements[0]
                # Get the inner HTML by converting children to string
                inner_parts = []
                for child in element:
                    # Convert each child element to string
                    child_html = etree.tostring(child, encoding='unicode', method='xml')
                    inner_parts.append(child_html)

                # Also get any direct text content
                if element.text and element.text.strip():
                    inner_parts.insert(0, element.text.strip())

                # Handle tail text after each child
                for i, child in enumerate(element):
                    if child.tail and child.tail.strip():
                        if i + 1 < len(inner_parts):
                            inner_parts[i + 1] = inner_parts[i + 1] + child.tail
                        else:
                            inner_parts.append(child.tail.strip())

                return ''.join(inner_parts).strip()

            return ""

        except Exception as e:
            # Fallback to text content if extraction fails
            log(ERROR, f"Error extracting inner HTML for {xpath}: {e}")
            return ""

    @classmethod
    def _build_html_from_subelements(cls, subelements):
        """Build HTML string from subelements structure"""
        if isinstance(subelements, list):
            html_parts = []
            for element in subelements:
                if isinstance(element, dict):
                    tag = element.get('tag', '')
                    attributes = element.get('attributes', {})
                    text = element.get('text', '')
                    nested_subelements = element.get('subelements', [])

                    if tag:
                        # Build opening tag with attributes
                        attr_str = ''
                        if attributes:
                            attr_parts = []
                            for key, value in attributes.items():
                                if value is not None:
                                    attr_parts.append(f'{key}="{value}"')
                            if attr_parts:
                                attr_str = ' ' + ' '.join(attr_parts)

                        html_parts.append(f'<{tag}{attr_str}>')

                        # Add text content
                        if text:
                            html_parts.append(text)

                        # Recursively process nested subelements
                        if nested_subelements:
                            html_parts.append(cls._build_html_from_subelements(nested_subelements))

                        # Add closing tag
                        html_parts.append(f'</{tag}>')
                    else:
                        # Just text node
                        html_parts.append(text)
                else:
                    # Simple string element
                    html_parts.append(str(element))
            return ''.join(html_parts)
        else:
            return str(subelements)

    @classmethod
    def from_xml(cls, xml_content: str, id: str, escape: bool = False, placeholder_map: Dict[str, str] | None = None, is_template: bool = False) -> 'ReasoningDocument':
        """Create a ReasoningDocument from XML content.

        Args:
            xml_content: The XML content to parse
            id: The document ID
            escape: Whether the content is XML-escaped
            placeholder_map: Optional placeholder translation map (e.g., {"%LANGUAGE%": "Italian"})
            is_template: Whether this document is a template (DocumentTemplate node in Neo4j)

        Returns:
            A ReasoningDocument instance
        """
        document = cls(placeholder_map=placeholder_map, is_template=is_template)
        # Temporarily enable write access during parsing (new document construction)
        document._rdoc_lock_depth = 1  # Allow writes during XML parsing

        # Extract title and template id from HTML (respect tag case)
        try:
            # Get title from Header element
            title_elements = get_elements_by_xpath(xml_content, "//header//title", multiple=False)
            if title_elements:
                title_text = title_elements.get('text', '') if isinstance(title_elements, dict) else str(title_elements)
                document.set_title(xml_unescape(title_text, escape))
            # Extract <template id="..."> from Header
            template_elem = get_elements_by_xpath(xml_content, "//header//template", multiple=False)
            if template_elem and isinstance(template_elem, dict):
                document._rdoc_templateid = template_elem.get('attributes', {}).get('id', '')
            else:
                document._rdoc_templateid = None

            if id is not None:
                document._rdoc_id = id
            else:
                # Get template ID from Header element if not provided
                if template_elem and isinstance(template_elem, dict):
                    document._rdoc_id = template_elem.get('attributes', {}).get('id', '')
                else:
                    document._rdoc_id = None

            # Parse <languages> block from header (optional)
            document._rdoc_languages = []
            languages_block_match = re.search(
                r'<header[^>]*>.*?<languages>(.*?)</languages>',
                xml_content, re.DOTALL
            )
            if languages_block_match:
                languages_block = languages_block_match.group(1)
                for lang_match in re.finditer(
                    r'<language([^>]*?)\s*/?>',
                    languages_block, re.DOTALL
                ):
                    attrs_str = lang_match.group(1)
                    lang_m = re.search(r'lang="([^"]*)"', attrs_str)
                    lang_code = lang_m.group(1) if lang_m else "en"
                    master_m = re.search(r'master="([^"]*)"', attrs_str)
                    master = master_m.group(1).lower() == "true" if master_m else False
                    type_m = re.search(r'type="([^"]*)"', attrs_str)
                    lang_type = type_m.group(1).lower().strip() if type_m else None
                    # Skip guide-only placeholder values from master template
                    if lang_type and '|' in lang_type:
                        lang_type = None
                    document._rdoc_languages.append(ReasoningLanguage(lang=lang_code, master=master, type=lang_type))

            # Parse <common> block – shared instruction snippets
            document._rdoc_common_instructions = {}
            common_instr_elems = get_elements_by_xpath(xml_content, "//common//instructions", multiple=True)
            if common_instr_elems:
                for ci_elem in common_instr_elems:
                    if isinstance(ci_elem, dict):
                        ci_id = ci_elem.get('attributes', {}).get('id', '')
                        ci_text = xml_unescape(ci_elem.get('text', ''), escape)
                        if ci_id:
                            document._rdoc_common_instructions[ci_id] = ci_text

            # Get all Division elements
            division_elements = get_elements_by_xpath(xml_content, "//division", multiple=True)

            for div in division_elements:
                division = ReasoningDivision()
                division._rd_id = div.get('id') if isinstance(div, dict) else None
                # Parse cycle attribute, default to 0 if missing or invalid
                cycle_val = 0
                if isinstance(div, dict):
                    cycle_str = div.get('attributes', {}).get('cycle', None)
                    try:
                        cycle_val = int(cycle_str) if cycle_str is not None else 0
                    except Exception:
                        cycle_val = 0
                division._rd_cycle = cycle_val
                # Parse scope attribute, default to PRIVATE if missing or invalid
                scope_val = DivisionScope.PRIVATE
                if isinstance(div, dict):
                    scope_str = div.get('attributes', {}).get('scope', None)
                    if scope_str:
                        try:
                            scope_val = DivisionScope(scope_str.lower())
                        except Exception:
                            scope_val = DivisionScope.PRIVATE
                division._rd_scope = scope_val
                # Parse execution attribute
                execution_val = DivisionExecution.SEQUENTIAL   # default
                if isinstance(div, dict):
                    execution_str = div.get('attributes', {}).get('execution', None)
                    if execution_str:
                        try:
                            execution_val = DivisionExecution(execution_str.lower())
                        except Exception:
                            execution_val = DivisionExecution.SEQUENTIAL
                division._rd_execution = execution_val
                # Parse sequence attribute
                sequence_val = None
                if isinstance(div, dict):
                    sequence_val = div.get('attributes', {}).get('sequence', None)
                division._rd_sequence = sequence_val
                # Parse indentation attribute, default to 0
                indentation_val = 0
                if isinstance(div, dict):
                    indentation_str = div.get('attributes', {}).get('indentation', None)
                    try:
                        indentation_val = int(indentation_str) if indentation_str is not None else 0
                    except Exception:
                        indentation_val = 0
                division._rd_indentation = indentation_val
                # Parse visibility attribute, default to VISIBLE
                visibility_val = DivisionVisibility.VISIBLE
                if isinstance(div, dict):
                    visibility_str = div.get('attributes', {}).get('visibility', None)
                    if visibility_str:
                        try:
                            visibility_val = DivisionVisibility(visibility_str.lower())
                        except Exception:
                            visibility_val = DivisionVisibility.VISIBLE
                division._rd_visibility = visibility_val
                # Ensure _rd_instructions is properly initialized for this division
                division._rd_instructions = {}

                # Extract division elements using the original HTML content with proper XPaths
                status_elem = get_elements_by_xpath(xml_content, f"//division[@id='{division._rd_id}']//status", multiple=False)
                if status_elem and isinstance(status_elem, dict):
                    status_val = status_elem.get('text', '')
                    try:
                        division._rd_status = ReasoningStatus(status_val)
                    except Exception:
                        division._rd_status = ReasoningStatus.RS_OPEN

                # Parse <titles> block (multi-language); fall back to bare <title> for old XML
                division._rd_titles = []
                titles_block_match = re.search(
                    rf'<division[^>]*id="{re.escape(division._rd_id)}"[^>]*>.*?<titles>(.*?)</titles>',
                    xml_content, re.DOTALL
                )
                if titles_block_match:
                    titles_block = titles_block_match.group(1)
                    for title_match in re.finditer(
                        r'<title([^>]*)>\s*([^<]*)\s*</title>',
                        titles_block, re.DOTALL
                    ):
                        attrs_str = title_match.group(1)
                        text = xml_unescape(title_match.group(2).strip(), escape)
                        lang_m = re.search(r'lang="([^"]*)"', attrs_str)
                        lang = lang_m.group(1) if lang_m else "en"
                        master_m = re.search(r'master="([^"]*)"', attrs_str)
                        master = master_m.group(1).lower() == "true" if master_m else False
                        if text:
                            division._rd_titles.append(ReasoningTitle(text=text, lang=lang, master=master))
                else:
                    # Backward compat: single bare <title> element
                    title_elem = get_elements_by_xpath(xml_content, f"//division[@id='{division._rd_id}']//title", multiple=False)
                    if title_elem and isinstance(title_elem, dict):
                        raw_text = xml_unescape(title_elem.get('text', ''), escape)
                        if raw_text:
                            division._rd_titles = [ReasoningTitle(text=raw_text, lang="en", master=True)]

                # Parse <shortTitles> block (multi-language); fall back to bare <shortTitle> for old XML
                division._rd_short_titles = []
                short_titles_block_match = re.search(
                    rf'<division[^>]*id="{re.escape(division._rd_id)}"[^>]*>.*?<shortTitles>(.*?)</shortTitles>',
                    xml_content, re.DOTALL
                )
                if short_titles_block_match:
                    short_titles_block = short_titles_block_match.group(1)
                    for st_match in re.finditer(
                        r'<shortTitle([^>]*)>\s*([^<]*)\s*</shortTitle>',
                        short_titles_block, re.DOTALL
                    ):
                        attrs_str = st_match.group(1)
                        text = xml_unescape(st_match.group(2).strip(), escape)
                        lang_m = re.search(r'lang="([^"]*)"', attrs_str)
                        lang = lang_m.group(1) if lang_m else "en"
                        master_m = re.search(r'master="([^"]*)"', attrs_str)
                        master = master_m.group(1).lower() == "true" if master_m else False
                        if text:
                            division._rd_short_titles.append(ReasoningTitle(text=text, lang=lang, master=master))
                else:
                    # Backward compat: single bare <shortTitle> element
                    short_title_elem = get_elements_by_xpath(xml_content, f"//division[@id='{division._rd_id}']//shortTitle", multiple=False)
                    if short_title_elem and isinstance(short_title_elem, dict):
                        raw_text = xml_unescape(short_title_elem.get('text', ''), escape)
                        if raw_text:
                            division._rd_short_titles = [ReasoningTitle(text=raw_text, lang="en", master=True)]

                # Extract focusContent element
                focus_content_elem = get_elements_by_xpath(xml_content, f"//division[@id='{division._rd_id}']//focusContent", multiple=False)
                if focus_content_elem and isinstance(focus_content_elem, dict):
                    division._rd_focus_content = xml_unescape(focus_content_elem.get('text', ''), escape)
                    # Parse splitToChunks attribute
                    split_to_chunks_str = focus_content_elem.get('attributes', {}).get('splitToChunks', None)
                    if split_to_chunks_str is not None:
                        division._rd_focus_content_split_to_chunks = split_to_chunks_str.lower() == "true"
                    else:
                        division._rd_focus_content_split_to_chunks = None
                else:
                    division._rd_focus_content = None
                    division._rd_focus_content_split_to_chunks = None

                # Extract instructions elements - there can be multiple, one for each state
                instructions_elems = get_elements_by_xpath(xml_content, f"//division[@id='{division._rd_id}']//instructions", multiple=True)
                if instructions_elems:
                    for instr_elem in instructions_elems:
                        if isinstance(instr_elem, dict):
                            for_status = instr_elem.get('attributes', {}).get('for_status', '')
                            instruction_text = xml_unescape(instr_elem.get('text', ''), escape)
                            recommended_tools = instr_elem.get('attributes', {}).get('recommended_tools', '')
                            recommended_plan = instr_elem.get('attributes', {}).get('recommended_plan', '')
                            action = instr_elem.get('attributes', {}).get('action', DivisionAction.NONE.value)
                            instruction_id = instr_elem.get('attributes', {}).get('id', '')
                            include_ids = instr_elem.get('attributes', {}).get('include_common_instructions_ids', '')
                            if for_status:
                                # Handle multiple states separated by |
                                states = [state.strip() for state in for_status.split('|')]
                                for state in states:
                                    if state:  # Only add non-empty states
                                        division._rd_instructions[state] = {
                                            "text": instruction_text,
                                            "recommended_tools": recommended_tools,
                                            "recommended_plan": recommended_plan,
                                            "action": action if len(action) > 0 else DivisionAction.NONE.value,
                                            "id": instruction_id,
                                            "include_common_instructions_ids": include_ids,
                                        }

                # Extract summaries from <summaries> block (multiple language variants)
                division._rd_summaries = []
                summaries_block_match = re.search(
                    rf'<division[^>]*id="{re.escape(division._rd_id)}"[^>]*>.*?<summaries>(.*?)</summaries>',
                    xml_content, re.DOTALL
                )
                if summaries_block_match:
                    summaries_block = summaries_block_match.group(1)
                    for sum_match in re.finditer(
                        r'<summary([^>]*)>\s*(?:<!\[CDATA\[(.*?)\]\]>|([^<]*))\s*</summary>',
                        summaries_block, re.DOTALL
                    ):
                        attrs_str = sum_match.group(1)
                        cdata_text = sum_match.group(2)
                        plain_text = sum_match.group(3)
                        text = cdata_text if cdata_text is not None else (plain_text or "").strip()
                        lang_m = re.search(r'lang="([^"]*)"', attrs_str)
                        lang = lang_m.group(1) if lang_m else "en"
                        master_m = re.search(r'master="([^"]*)"', attrs_str)
                        master = master_m.group(1).lower() == "true" if master_m else False
                        version_m = re.search(r'version="([^"]*)"', attrs_str)
                        version = int(version_m.group(1)) if version_m else 0
                        if text:  # only add non-empty summaries
                            division._rd_summaries.append(ReasoningSummary(text=text, lang=lang, master=master, version=version))

                # Extract prompt element - preserve inner HTML
                prompt_elem = get_elements_by_xpath(xml_content, f"//division[@id='{division._rd_id}']//prompt", multiple=False)
                if prompt_elem and isinstance(prompt_elem, dict):
                    # Extract the inner HTML content directly from the original XML (CDATA preserves content as-is)
                    division._rd_prompt = cls._extract_inner_html(xml_content, f"//division[@id='{division._rd_id}']//prompt")

                keywords_elem = get_elements_by_xpath(xml_content, f"//division[@id='{division._rd_id}']//keywords", multiple=False)
                if keywords_elem and isinstance(keywords_elem, dict):
                    division._rd_keywords = xml_unescape(keywords_elem.get('text', ''), escape)

                references_elem = get_elements_by_xpath(xml_content, f"//division[@id='{division._rd_id}']//references", multiple=False)
                if references_elem and isinstance(references_elem, dict):
                    division._rd_references = xml_unescape(references_elem.get('text', ''), escape)

                # Extract results elements and parse each <result> child
                division._rd_results = []
                results_elems = get_elements_by_xpath(xml_content, f"//division[@id='{division._rd_id}']//result", multiple=True)
                if results_elems:
                    for result_elem in results_elems:
                        if isinstance(result_elem, dict):
                            result_elem_attr = result_elem.get('attributes', {})
                            for_task_id = result_elem_attr.get('for_task_id', '')
                            # extract inner HTML/text so CDATA is preserved (no unescaping needed)
                            result_text = cls._extract_inner_html(xml_content, f"//division[@id='{division._rd_id}']//result[@for_task_id='{for_task_id}']")
                            result_obj = ReasoningResult(for_task_id=for_task_id, result=result_text)
                            division._rd_results.append(result_obj)

                # Extract tasks
                task_elems = get_elements_by_xpath(xml_content, f"//division[@id='{division._rd_id}']//task", multiple=True)
                if task_elems:
                    for task_elem in task_elems:
                        if isinstance(task_elem, dict):
                            task_id = task_elem.get('attributes', {}).get('id', '')
                            task_status = task_elem.get('attributes', {}).get('status', TaskStatus.TS_OPEN.value)
                            if not isinstance(task_status, TaskStatus):
                                try:
                                    task_status = TaskStatus(task_status)
                                except Exception:
                                    task_status = TaskStatus.TS_OPEN
                            # Extract inner HTML/text for the task so CDATA content is preserved (handles raw '<' and '>')
                            task_description = cls._extract_inner_html(xml_content, f"//division[@id='{division._rd_id}']//task[@id='{task_id}']")
                            # Fallback to plain text if extraction failed
                            if task_description is None or task_description == "":
                                task_description = xml_unescape(task_elem.get('text', ''), escape)
                            recommended_plan = task_elem.get('attributes', {}).get('recommended_plan', '')
                            recommended_tools = task_elem.get('attributes', {}).get('recommended_tools', '')
                            task = ReasoningTask(
                                task_id=task_id,
                                status=task_status,
                                description=task_description,
                                recommended_plan=recommended_plan,
                                recommended_tools=recommended_tools
                            )
                            division.add_task(task)

                # Extract issues
                division._rd_issues = []
                issue_elems = get_elements_by_xpath(xml_content, f"//division[@id='{division._rd_id}']//issue", multiple=True)
                if issue_elems:
                    for issue_elem in issue_elems:
                        if isinstance(issue_elem, dict):
                            issue_id = issue_elem.get('attributes', {}).get('id', '')
                            issue_status = issue_elem.get('attributes', {}).get('status', IssueStatus.OPEN.value)
                            if not isinstance(issue_status, IssueStatus):
                                try:
                                    issue_status = IssueStatus(issue_status)
                                except Exception:
                                    issue_status = IssueStatus.OPEN
                            # Parse severity attribute, default to LOW
                            issue_severity = issue_elem.get('attributes', {}).get('severity', IssueSeverity.LOW.value)
                            if not isinstance(issue_severity, IssueSeverity):
                                try:
                                    issue_severity = IssueSeverity(issue_severity)
                                except Exception:
                                    issue_severity = IssueSeverity.LOW
                            issue_description = cls._extract_inner_html(xml_content, f"//division[@id='{division._rd_id}']//issue[@id='{issue_id}']")
                            issue = ReasoningIssue(
                                issue_id=issue_id,
                                status=issue_status,
                                description=issue_description,
                                severity=issue_severity
                            )
                            division.add_issue(issue)

                # Extract recommendations (multiple <recommendation> elements)
                division._rd_recommendations = []
                recommendation_elems = get_elements_by_xpath(xml_content, f"//division[@id='{division._rd_id}']//recommendation", multiple=True)
                if recommendation_elems:
                    for i, rec_elem in enumerate(recommendation_elems):
                        if isinstance(rec_elem, dict):
                            # Extract the inner HTML content directly from the original XML to preserve CDATA
                            # For each recommendation, build a unique XPath using its position within the recommendations container
                            rec_text = cls._extract_inner_html(xml_content, f"//division[@id='{division._rd_id}']//recommendations/recommendation[{i+1}]")
                            division._rd_recommendations.append(rec_text)

                document.add_division(division)

        except Exception as e:
            log(ERROR, f"Error parsing HTML content for ReasoningTemplate: {e}")
            log(ERROR, f"Traceback: {traceback.format_exc()}")
        finally:
            # Restore lock state after parsing (document should not be writable without explicit lock)
            document._rdoc_lock_depth = 0

        return document

    @classmethod
    def from_id(cls, document_id: str, is_template: bool = False, placeholder_map: Dict[str, str] | None = None) -> 'ReasoningDocument':
        document_html = None
        last_update = None
        try:
            if document_id is not None:
                id_attribute = 'templateid' if is_template else 'id'
                label_attribute = 'DocumentTemplate' if is_template else 'ReasoningDocument'
                # Use COALESCE to handle documents without last_update property (returns 0 for old documents)
                document_query = f"""
                    MATCH (d:{label_attribute})
                    WHERE d.{id_attribute} = $document_id
                    RETURN d.{id_attribute} as id, d.name, d.title, d.text, d.enhanced_text, d.enhanced_type, COALESCE(d.last_update, NULL) as last_update
                """
                document_result = execute_neo4j_query(document_query, {"document_id": document_id})
                if document_result:
                    for record in document_result:
                        document_html = record[4] if record[4] is not None else record[3]
                        last_update = record[6] if len(record) > 6 else None
                        break
        except Exception as e:
            log(ERROR, f"Error loading document '{document_id}': {e}")
            document_html = None

        document = cls.from_xml(document_html, id=document_id, placeholder_map=placeholder_map, is_template=is_template) if document_html is not None else None
        if document is not None:
            # Set timestamp (will be None for old documents without last_update property)
            document._rdoc_last_update = last_update
            # Preserve is_template flag (from_xml doesn't know about it)
            document._rdoc_is_template = is_template
        return document


log_end_file(__file__)


if __name__ == "__main__":
    log_init()

    try:
        if True:
            # Test: programmatically create a reasoning document, convert it to XML, then back from XML
            # This validates the new <summaries><summary> XML structure introduced for the summary field.
            log(INFO, "\n--- Test: round-trip ReasoningDocument with <summaries> wrapper ---")

            # 1. Build document programmatically (in-memory, no Neo4j needed)
            doc = ReasoningDocument(title="Test Document", id="test-summaries-roundtrip")
            doc._rdoc_lock_depth = 1  # allow writes without Neo4j (same as from_xml does during construction)

            div1 = ReasoningDivision(
                id="div-001",
                title="First Division",
                status=ReasoningStatus.RS_OPEN,
                summary="<p>Summary with <strong>HTML</strong> content and special chars: & < ></p>",
                scope=DivisionScope.PUBLIC,
            )
            div1.add_new_task("Task one", task_id="t-001", status=TaskStatus.TS_OPEN)
            div1.add_new_task("Task two", task_id="t-002", status=TaskStatus.TS_IN_PROGRESS)
            div1.add_recommendation("First recommendation")
            div1.add_recommendation("Second recommendation")
            # Add additional language summaries for div1
            div1.set_summary("<p>Zusammenfassung auf Deutsch</p>", lang="de", master=False)
            div1.set_summary("<p>Résumé en français</p>", lang="fr", master=False)
            doc.add_division(div1)

            div2 = ReasoningDivision(
                id="div-002",
                title="Second Division",
                status=ReasoningStatus.RS_COMPLETE,
                summary="Plain text summary without HTML.",
                scope=DivisionScope.PRIVATE,
            )
            issue = ReasoningIssue(issue_id="issue-001", description="A test issue", severity=IssueSeverity.MEDIUM)
            div2.add_issue(issue)
            doc.add_division(div2)

            div3_no_summary = ReasoningDivision(
                id="div-003",
                title="Division without summary",
                status=ReasoningStatus.RS_OPEN,
            )
            doc.add_division(div3_no_summary)

            doc._rdoc_lock_depth = 0  # restore lock state

            # 2. Serialize to XML and verify structure
            xml_output = doc.to_xml()
            log(INFO, "Serialized XML:\n" + xml_output)

            assert "<summaries>" in xml_output, "XML must contain <summaries> wrapper tag"
            assert "<summary" in xml_output, "XML must contain <summary> inner tag"
            assert "</summaries>" in xml_output, "XML must contain </summaries> closing tag"
            # Verify language attributes
            assert 'lang="en"' in xml_output, "XML must contain lang=\"en\" attribute"
            assert 'lang="de"' in xml_output, "XML must contain lang=\"de\" attribute"
            assert 'lang="fr"' in xml_output, "XML must contain lang=\"fr\" attribute"
            assert 'master="true"' in xml_output, "XML must contain master=\"true\" attribute"
            import re as _re
            # div-003 has no summary -> no <summaries> block should appear for it
            # Count occurrences: we have two divisions with summaries so two <summaries> blocks
            summaries_count = len(_re.findall(r'<summaries>', xml_output))
            assert summaries_count == 2, f"Expected 2 <summaries> blocks (one per division with summary), found {summaries_count}"
            # div1 has 3 summaries (EN master + DE + FR)
            div1_block_match = _re.search(r'<division id="div-001"[^>]*>.*?</division>', xml_output, _re.DOTALL)
            assert div1_block_match, "div-001 block must be present"
            div1_block = div1_block_match.group(0)
            assert div1_block.count('<summary') == 3, f"div-001 must have 3 <summary> elements, got {div1_block.count('<summary')}"
            log(INFO, "PASS: XML structure contains correct <summaries> wrappers and language attributes")

            # 3. Parse back from XML
            restored = ReasoningDocument.from_xml(xml_output, id="test-summaries-roundtrip")
            assert restored is not None, "from_xml must return a document"
            # Stamp the restored doc so read_context won't try to reload from Neo4j
            # (this is a pure serialisation test – no Neo4j involvement needed).
            restored._rdoc_last_update = time.time()

            with restored.read_context():
                assert restored.title() == "Test Document", f"Title mismatch: {restored.title()}"
                assert len(restored.divisions()) == 3, f"Expected 3 divisions, got {len(restored.divisions())}"

                rdiv1 = restored.get_division_by_id("div-001")
                assert rdiv1 is not None, "div-001 must exist"
                summary1 = rdiv1.summary()  # master (EN)
                assert summary1 == div1.summary(), f"Summary mismatch for div-001:\nExpected: {div1.summary()}\nGot:      {summary1}"
                assert rdiv1.summary(lang="de") == "<p>Zusammenfassung auf Deutsch</p>", f"DE summary mismatch: {rdiv1.summary(lang='de')}"
                assert rdiv1.summary(lang="fr") == "<p>Résumé en français</p>", f"FR summary mismatch: {rdiv1.summary(lang='fr')}"
                assert len(rdiv1.summaries()) == 3, f"Expected 3 summaries in div-001, got {len(rdiv1.summaries())}"
                master_summaries = [s for s in rdiv1.summaries() if s.master()]
                assert len(master_summaries) == 1, f"Expected exactly 1 master summary, got {len(master_summaries)}"
                assert master_summaries[0].lang() == "en", f"Master summary should be EN, got {master_summaries[0].lang()}"
                assert len(rdiv1.tasks()) == 2, f"Expected 2 tasks in div-001, got {len(rdiv1.tasks())}"
                assert len(rdiv1.recommendations()) == 2, f"Expected 2 recommendations in div-001, got {len(rdiv1.recommendations())}"
                assert rdiv1.scope() == DivisionScope.PUBLIC, "div-001 scope must be PUBLIC"
                log(INFO, f"PASS: div-001 summaries round-tripped correctly (EN master + DE + FR)")

                rdiv2 = restored.get_division_by_id("div-002")
                assert rdiv2 is not None, "div-002 must exist"
                summary2 = rdiv2.summary()
                assert summary2 == div2.summary(), f"Summary mismatch for div-002:\nExpected: {div2.summary()}\nGot:      {summary2}"
                assert len(rdiv2.summaries()) == 1, f"Expected 1 summary in div-002, got {len(rdiv2.summaries())}"
                assert len(rdiv2.issues()) == 1, f"Expected 1 issue in div-002, got {len(rdiv2.issues())}"
                log(INFO, f"PASS: div-002 summary round-tripped correctly: {summary2}")

                rdiv3 = restored.get_division_by_id("div-003")
                assert rdiv3 is not None, "div-003 must exist"
                assert not rdiv3.summary(), f"div-003 should have no summary, got: {rdiv3.summary()}"
                log(INFO, "PASS: div-003 correctly has no summary")

            log(INFO, "--- All <summaries> round-trip tests PASSED ---\n")

        if False:
            from src.core.base.ResourceMgmt import get_lock

            # load from neo4j the template "template-asylum" and print the division identifiers, along with available instructions
            template_id = "template-asylum"
            template = ReasoningDocument.from_id(template_id, is_template=True)
            if template is None:
                log(ERROR, "Template not found.")
            else:
                log(INFO, f"Loaded template: {template._rdoc_title} with ID {template._rdoc_id}")
                for division in template.divisions():
                    log(INFO, f"Division ID: {division.id()}")

            # now instantiate a ReasoningDocument from that template, and test the to_xml and from_xml methods
            if template is not None:
                # Convert template to XML
                template_xml = template.to_xml()
                # Create a new document from the template XML
                new_document = ReasoningDocument.from_xml(template_xml, id="test-doc-from-template")

                if new_document is None:
                    log(ERROR, "Failed to create document from template.")
                else:
                    log(INFO, f"\nCreated new document from template: {new_document._rdoc_title} with ID {new_document._rdoc_id}")
                    for division in new_document.divisions():
                        log(INFO, f"Division ID: {division.id()}")

            # manipulate the new document's division: add 3 tasks, 2 issues, 4 recommendations
            if new_document is not None and len(new_document.divisions()) > 0:
                # Set the document ID before writing
                new_document.set_id("docid-000")

                # Get a lock and use proper write_context
                doc_lock = get_lock("REASONING_DOCUMENT", "docid-000")
                with new_document.write_context(doc_lock):
                    division = new_document.divisions()[0]

                    # Add 3 tasks
                    task1 = division.add_new_task("Review asylum documentation", task_id="task-001", status=TaskStatus.TS_OPEN)
                    task2 = division.add_new_task("Verify applicant identity", task_id="task-002", status=TaskStatus.TS_IN_PROGRESS)
                    task3 = division.add_new_task("Assess credibility of claims", task_id="task-003", status=TaskStatus.TS_OPEN)

                    log(INFO, f"\nAdded 3 tasks to division '{division.id()}'")

                    # Add 2 issues (as ReasoningIssue objects or dicts)
                    issue1 = ReasoningIssue(issue_id="issue-001", description="Missing supporting documents from Kosovo authorities", severity=IssueSeverity.HIGH)
                    issue2 = ReasoningIssue(issue_id="issue-002", description="Inconsistencies in timeline of events", severity=IssueSeverity.MEDIUM)
                    division.add_issue(issue1)
                    division.add_issue(issue2)

                    log(INFO, "Added 2 issues to division")

                    # Add 4 recommendations
                    division.add_recommendation("Obtain certified copies of police reports from Kosovo")
                    division.add_recommendation("Request additional witness statements")
                    division.add_recommendation("Conduct follow-up interview with applicant")
                    division.add_recommendation("Consult with legal experts on jurisdiction")

                    log(INFO, "Added 4 recommendations to division")

                    # Save the document to Neo4j (happens automatically on context exit)

                log(INFO, "Document saved to Neo4j (automatic on write_context exit)")

                # Now load the document back from Neo4j to test caching/reading
                log(INFO, "\n--- Testing Neo4j Read/Caching ---")
                loaded_document = ReasoningDocument.from_id("docid-000", is_template=False)
                if loaded_document is None:
                    log(ERROR, "Failed to load document from Neo4j")
                else:
                    log(INFO, f"Successfully loaded document from Neo4j: {loaded_document.title()} (ID: {loaded_document.id()})")

            # now under read_context, print the document's divisions and other information contained in them
            if new_document is not None:
                with new_document.read_context():
                    log(INFO, f"\n--- Document Details (under read_context) ---")
                    log(INFO, f"Document Title: {new_document.title()}")
                    log(INFO, f"Document ID: {new_document.id()}")
                    log(INFO, f"Number of divisions: {len(new_document.divisions())}")

                    for i, div in enumerate(new_document.divisions(), 1):
                        log(INFO, f"\n  Division {i}: {div.id()}")
                        log(INFO, f"    Title: {div.title()}")
                        log(INFO, f"    Status: {div.status().value}")
                        log(INFO, f"    Scope: {div.scope().value}")

                        tasks = div.tasks()
                        log(INFO, f"    Tasks: {len(tasks)}")
                        for task in tasks:
                            log(INFO, f"      - {task.description()} [ID: {task.id()}, Status: {task.status().value}]")

                        issues = div.issues()
                        log(INFO, f"    Issues: {len(issues)}")
                        for issue in issues:
                            log(INFO, f"      - {issue}")

                        recommendations = div.recommendations()
                        log(INFO, f"    Recommendations: {len(recommendations)}")
                        for rec in recommendations:
                            log(INFO, f"      - {rec}")

        if False:
            # --- Test for recommendation duplication issue ---
            log(INFO, "\n--- Testing Recommendation Duplication Issue ---")
            test_doc = ReasoningDocument(title="Test Document", id="test-doc-1")
            test_div = ReasoningDivision(id="test-div-1", title="Test Division")

            # Add multiple distinct recommendations
            test_recommendations = [
                "Obtain and attach the original police reports, investigation numbers and any statements from Kosovo authorities.",
                "Verify the authenticity of all documents provided.",
                "Cross-reference dates with official records.",
                "Obtain statements from eyewitnesses.",
                "Review travel records and border crossings.",
                "Consult with legal experts on jurisdiction."
            ]

            for rec in test_recommendations:
                test_div.add_recommendation(rec)

            test_doc.add_division(test_div)

            log(INFO, f"Original recommendations ({len(test_div.recommendations())} total):")
            for i, rec in enumerate(test_div.recommendations(), 1):
                log(INFO, f"  {i}. {rec[:60]}...")

            # Convert to XML
            xml_output = test_doc.to_xml()
            log(INFO, "\nXML recommendations section:")
            rec_section = re.search(r'<recommendations>.*?</recommendations>', xml_output, re.DOTALL)
            if rec_section:
                log(INFO, rec_section.group(0)[:500] + "..." if len(rec_section.group(0)) > 500 else rec_section.group(0))

            # Convert back from XML
            test_doc_restored = ReasoningDocument.from_xml(xml_output, id="test-doc-1")
            restored_div = test_doc_restored.get_division_by_id("test-div-1")

            log(INFO, f"\nRestored recommendations ({len(restored_div.recommendations())} total):")
            for i, rec in enumerate(restored_div.recommendations(), 1):
                log(INFO, f"  {i}. {rec[:60]}...")

            # Check if duplication occurred
            if len(restored_div.recommendations()) > 0:
                first_rec = restored_div.recommendations()[0]
                all_same = all(rec == first_rec for rec in restored_div.recommendations())
                if all_same and len(restored_div.recommendations()) > 1:
                    log(ERROR, "ISSUE CONFIRMED: All recommendations are duplicates of the first one!")
                else:
                    log(INFO, "SUCCESS: Recommendations are preserved correctly!")

            # --- End test ---
            log(INFO, "\n--- Continuing with original tests ---")

            doc_id = "docid-93"
            template_id = "template-deepinternetsearch"
            template = ReasoningDocument.from_id(template_id, is_template=True)
            if template is None:
                log(ERROR, "Template not found.")
            else:
                log(INFO, f"Loaded template: {template._rdoc_title} with ID {template._rdoc_id}")

            assert template is not None, "Template should not be None"
            assert template.divisions() is not None, "Template divisions should not be None"
            assert len(template.divisions()) > 0, "Template should have at least one divisions"
            assert template.divisions()[0].instructions() is not None, "First division instructions should not be None"
            first_division_instruction_for_open = template.divisions()[1].instructions()['open'].get("text")
            assert "Read the user prompt" in first_division_instruction_for_open, f"First division instruction[0] text must be correct, instead it's: {first_division_instruction_for_open}"

            # let's add html representation in the prompt and in the summary
            prompt_html = "<p>Prompt with <strong>HTML</strong> content</p>"
            summary_html = "<p>Summary with <em>HTML</em> content</p>"
            first_division = template.divisions()[1]
            first_division._rd_prompt = prompt_html
            first_division.set_summary(summary_html, lang="en", master=True)
            first_division.set_cycle(2)
            # let's save it, and reload it, to test if it works
            template_html = template.to_xml()
            template_from_html = ReasoningDocument.from_xml(template_html, id=template._rdoc_id)
            assert template_from_html is not None, "Template from HTML should not be None"
            # Wrap test getter calls with read_context for protection
            with template_from_html.read_context():
                print(f"Prompt= {template_from_html.divisions()[1].prompt()}")
                assert template_from_html.divisions()[1].prompt() == prompt_html, "Prompt should match the original HTML content"
                print(f"Summary= {template_from_html.divisions()[1].summary()}")
                assert template_from_html.divisions()[1].summary() == summary_html, "Summary should match the original HTML content"
                assert template_from_html.divisions()[1].cycle() == 2, "Cycle should match the original value"

                public_division = template_from_html.get_division_with_public_summary()
                assert public_division is not None, "There should be a division with public summary"
                assert public_division.scope() == DivisionScope.PUBLIC, "The division with public summary should have PUBLIC scope"
                for div in template_from_html.divisions():
                    if div.id() != public_division.id():
                        assert div.scope() == DivisionScope.PRIVATE, "No other division should have PUBLIC scope"

            document = ReasoningDocument.from_id(doc_id)
            if document is None:
                log(INFO, f"Document id={doc_id} not found.")
            else:
                log(INFO, f"Loaded document: {document._rdoc_title} with ID {doc_id}")

            # --- Round-trip XML test ---
            log(INFO, "\n--- Round-trip XML test ---")
            template = ReasoningDocument.from_id(template_id, is_template=True)
            xml1 = template.to_xml()
            query = """MATCH (n:DocumentTemplate {templateid: $template_id}) RETURN n.text as text"""
            parameters = {"template_id": template_id}
            result = execute_neo4j_query(query, parameters)
            if result is None or len(result) == 0:
                log(ERROR, "Failed to retrieve template from database.")
            else:
                xml2 = result[0].get("text")
                if xml2 is None:
                    log(ERROR, "Failed to retrieve text from database.")
                elif xml1 == xml2:
                    log(INFO, "Round-trip XML test PASSED: XML output is identical after parse and re-serialize.")
                else:
                    import difflib
                    # Normalize lines by stripping whitespace for comparison
                    xml1_lines = [line.strip() for line in xml1.splitlines() if line.strip()]
                    xml2_lines = [line.strip() for line in xml2.splitlines() if line.strip()]
                    diff = difflib.unified_diff(xml1_lines, xml2_lines, fromfile='original', tofile='roundtrip', lineterm='')
                    log(ERROR, 'There are differences in the XML generation:\n' + '\n'.join(diff))

    except Exception as e:
        log(FATAL, f"Error in ReasoningDocument tests: {e}")

    finally:
        log_exit()
