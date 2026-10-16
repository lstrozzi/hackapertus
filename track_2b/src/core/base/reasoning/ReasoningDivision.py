from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import re
from typing import Dict, List, TYPE_CHECKING

from src.core.base.enums import ReasoningStatus, TaskStatus, DivisionScope, DivisionExecution, DivisionAction, DivisionVisibility
from src.core.base.reasoning.xml_utils import (xml_escape, xml_cdata, log_write_context_error, log_read_context_warning,
                                                LINE_SEPARATOR, DEFAULT_RECOMMENDED_PLAN, DEFAULT_RECOMMENDED_TOOLS)
from src.core.base.reasoning.ReasoningIssue import ReasoningIssue
from src.core.base.reasoning.ReasoningTask import ReasoningTask
from src.core.base.reasoning.ReasoningResult import ReasoningResult
from src.core.base.reasoning.ReasoningSummary import ReasoningSummary
from src.core.base.reasoning.ReasoningTitle import ReasoningTitle
from src.utils.log import log, DEBUG, WARNING, ERROR

if TYPE_CHECKING:
    from src.core.base.reasoning.ReasoningDocument import ReasoningDocument


class ReasoningDivision:
    """Represents a division in the reasoning template"""

    def __init__(self, id: str | None = None,
                 status: ReasoningStatus | str = ReasoningStatus.RS_OPEN,
                 scope: DivisionScope = DivisionScope.PRIVATE,
                 execution: DivisionExecution = DivisionExecution.SEQUENTIAL,
                 indentation: int = 0,
                 visibility: DivisionVisibility = DivisionVisibility.VISIBLE,
                 title: str | None = None,
                 titles: List[ReasoningTitle] | None = None,
                 short_title: str | None = None,
                 short_titles: List[ReasoningTitle] | None = None,
                 sequence: str | None = None,
                 focus_content: List[str] | str | None = None,
                 focus_content_split_to_chunks: bool | None = None,
                 instructions: dict | None = None,
                 summary: str | None = None,
                 summaries: List[ReasoningSummary] | None = None,
                 keywords: str | None = None,
                 references: str | None = None,
                 prompt: str | None = None,
                 results: List[ReasoningResult] | None = None,
                 recommendations: List[str] | None = None,
                 cycle: int = 0,
                 issues: List['ReasoningIssue'] | None = None,
                 placeholder_map: Dict[str, str] | None = None) -> None:
        # type checking

        # CRITICAL: Initialize parent reference FIRST - needed by set_status() and other methods
        self._rd_parent_document: 'ReasoningDocument' | None = None

        # Bypass flag for read-only divisions (templates) - disables read_context protection
        self._rd_bypass_read_context: bool = False
        self._rd_id: str | None = id
        self._rd_scope: DivisionScope = scope
        self._rd_execution: DivisionExecution = execution
        self._rd_indentation: int = indentation if indentation is not None else 0
        self._rd_visibility: DivisionVisibility = visibility if visibility is not None else DivisionVisibility.VISIBLE
        self._rd_status: ReasoningStatus
        self.set_status(status if isinstance(status, ReasoningStatus) else ReasoningStatus(status))
        # Build _rd_titles from the explicit list first, then fall back to the legacy
        # scalar `title` shorthand (creates a single EN master title).
        if titles is not None:
            self._rd_titles: List[ReasoningTitle] = list(titles)
        elif title is not None and str(title).strip():
            self._rd_titles = [ReasoningTitle(text=title, lang="en", master=True)]
        else:
            self._rd_titles = []
        # Build _rd_short_titles similarly.
        if short_titles is not None:
            self._rd_short_titles: List[ReasoningTitle] = list(short_titles)
        elif short_title is not None and str(short_title).strip():
            self._rd_short_titles = [ReasoningTitle(text=short_title, lang="en", master=True)]
        else:
            self._rd_short_titles = []
        self._rd_sequence: str | None = sequence
        # Normalize focus_content to a comma-separated string.
        # If a list is provided, join non-empty items with commas. If a string is provided, keep it.
        self._rd_focus_content: str | None
        if isinstance(focus_content, list):
            cleaned = [str(x).strip() for x in focus_content if x is not None and str(x).strip() != ""]
            self._rd_focus_content = ",".join(cleaned) if cleaned else None
        else:
            self._rd_focus_content = focus_content
        self._rd_focus_content_split_to_chunks: bool | None = focus_content_split_to_chunks
        self._rd_instructions: Dict[str, Dict[str, str]] = instructions if instructions is not None else {}
        # Build _rd_summaries from the explicit list first, then fall back to the legacy
        # scalar `summary` shorthand (creates a single EN master summary).
        if summaries is not None:
            self._rd_summaries: List[ReasoningSummary] = list(summaries)
        elif summary is not None and str(summary).strip():
            self._rd_summaries = [ReasoningSummary(text=summary, lang="en", master=True)]
        else:
            self._rd_summaries = []
        self._rd_keywords: str | None = keywords
        self._rd_references: str | None = references
        self._rd_prompt: str | None = prompt
        self._rd_results: List[ReasoningResult] = results if results is not None else []  # List of ReasoningResult
        self._rd_tasks: List[ReasoningTask] = []  # Initialize as an empty list
        self._rd_issues: List[ReasoningIssue] = issues if issues is not None else []  # List of ReasoningIssue
        self._rd_recommendations: List[str] = recommendations if recommendations is not None else []
        self._rd_cycle: int = cycle
        self._rd_placeholder_map: Dict[str, str] = placeholder_map if placeholder_map is not None else {}

    def _check_write_allowed(self) -> bool:
        """Check if write operations are allowed via parent document lock."""
        if self._rd_bypass_read_context:
            log(ERROR, f"Write operation attempted on read-only division (bypass enabled) with ID '{self._rd_id}'. This is a programming error.")
            return False
        if self._rd_parent_document is None:
            return True  # No parent = allow (e.g., during construction)
        return self._rd_parent_document._check_write_allowed()

    def _check_read_allowed(self) -> bool:
        """Check if read operations are protected via parent document read context."""
        if self._rd_parent_document is None:
            return True  # No parent = allow (e.g., during construction)
        # Check bypass flag first for performance
        if self._rd_bypass_read_context:
            return True
        return self._rd_parent_document._check_read_allowed()

    def _is_in_initialization_phase(self) -> bool:
        """Check if division is in initialization phase (parent document not yet persisted)."""
        if self._rd_parent_document is None:
            return True  # No parent = in initialization
        return self._rd_parent_document._is_in_initialization_phase()

    def id(self) -> str:
        """Get the division ID (metadata, never changes)."""
        return self._rd_id

    def status(self) -> ReasoningStatus:
        """Get the division status."""
        if not self._check_read_allowed():
            log(WARNING, log_read_context_warning('ReasoningDivision', 'status', self._rd_id))
        return self._rd_status

    def set_status(self, status: ReasoningStatus) -> None:
        """Set the status of this division"""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_status', self._rd_id, 'set status'))
            return
        if isinstance(status, str):
            status = ReasoningStatus(status)
        self._rd_status = status

    def scope(self) -> DivisionScope:
        """Get the division scope."""
        if not self._check_read_allowed():
            log(WARNING, log_read_context_warning('ReasoningDivision', 'scope', self._rd_id))
        return self._rd_scope

    def set_scope(self, scope: DivisionScope) -> None:
        """Set the scope of this division"""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_scope', self._rd_id, 'set scope'))
            return
        if isinstance(scope, str):
            scope = DivisionScope(scope)
        elif scope is None:
            scope = DivisionScope.PRIVATE       # default
        self._rd_scope = scope

    def execution(self) -> DivisionExecution:
        """Get the division execution."""
        if not self._check_read_allowed():
            log(WARNING, log_read_context_warning('ReasoningDivision', 'execution', self._rd_id))
        return self._rd_execution

    def set_execution(self, execution: DivisionExecution) -> None:
        """Set the execution of this division"""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_execution', self._rd_id, 'set execution'))
            return
        if isinstance(execution, str):
            execution = DivisionExecution(execution)
        elif execution is None:
            execution = DivisionExecution.SEQUENTIAL       # default
        self._rd_execution = execution

    def indentation(self) -> int:
        """Get the division indentation level (0–5)."""
        return self._rd_indentation if hasattr(self, '_rd_indentation') and self._rd_indentation is not None else 0

    def set_indentation(self, indentation: int) -> None:
        """Set the indentation level for this division (0–5)."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_indentation', self._rd_id, 'set indentation'))
            return
        self._rd_indentation = int(indentation) if indentation is not None else 0

    def visibility(self) -> DivisionVisibility:
        """Get the division visibility."""
        return self._rd_visibility if hasattr(self, '_rd_visibility') and self._rd_visibility is not None else DivisionVisibility.VISIBLE

    def set_visibility(self, visibility: DivisionVisibility) -> None:
        """Set the visibility of this division."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_visibility', self._rd_id, 'set visibility'))
            return
        if isinstance(visibility, str):
            visibility = DivisionVisibility(visibility)
        elif visibility is None:
            visibility = DivisionVisibility.VISIBLE       # default
        self._rd_visibility = visibility

    def cycle(self) -> int:
        """Get the division cycle."""
        if not self._check_read_allowed():
            log(WARNING, log_read_context_warning('ReasoningDivision', 'cycle', self._rd_id))
        return self._rd_cycle if self._rd_cycle is not None else 0

    def set_cycle(self, cycle: int) -> int:
        """Set the cycle of this division"""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_cycle', self._rd_id, 'set cycle'))
            return self._rd_cycle
        self._rd_cycle = cycle if cycle is not None else 0
        return self._rd_cycle

    def titles(self) -> List[ReasoningTitle]:
        """Get all language-specific titles for this division."""
        return self._rd_titles

    def set_titles(self, titles: List[ReasoningTitle]) -> None:
        """Replace the complete list of titles."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_titles', self._rd_id, 'set titles'))
            return
        self._rd_titles = list(titles) if titles is not None else []

    def add_title(self, t: ReasoningTitle) -> None:
        """Append a ReasoningTitle to the list."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'add_title', self._rd_id, 'add title'))
            return
        if self._rd_titles is None:
            self._rd_titles = []
        self._rd_titles.append(t)

    def clear_titles(self) -> None:
        """Remove all titles from this division."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'clear_titles', self._rd_id, 'clear titles'))
            return
        self._rd_titles = []

    def get_title_for_lang(self, lang: str) -> 'ReasoningTitle | None':
        """Return the ReasoningTitle for the given language code, or None."""
        lang_lower = (lang or "").lower().strip()
        for t in self._rd_titles:
            if t.lang().lower() == lang_lower:
                return t
        return None

    def get_title_master(self) -> 'ReasoningTitle | None':
        """Return the master ReasoningTitle object.

        Logic:
        1. If there's only one title, it's master by definition (regardless of attribute)
        2. If there's no title with master=True, default to English language title
        3. Otherwise return the title marked as master=True
        Returns None if no titles exist.
        """
        if not self._rd_titles:
            return None
        if len(self._rd_titles) == 1:
            return self._rd_titles[0]
        for t in self._rd_titles:
            if t.master():
                return t
        for t in self._rd_titles:
            if t.lang().lower() == "en":
                return t
        return self._rd_titles[0]

    def title(self, lang: str = None) -> str:
        """Get title text (backward-compatible convenience accessor).

        - If *lang* is specified, returns the text of the matching language
          variant (or None if not found).
        - Otherwise returns the text of the master title, falling back to
          the first available title, or None if there are none.
        """
        if not self._rd_titles:
            return None
        if lang is not None:
            t = self.get_title_for_lang(lang)
            return t.text() if t else None
        master = self.get_title_master()
        return master.text() if master else None

    def set_title(self, text: str, lang: str = "en", master: bool = True) -> None:
        """Set (add or replace) the title for a given language.

        If *master* is True the master flag is transferred to this entry,
        clearing it from any previously-flagged entry.  If a title for
        *lang* already exists it is updated in-place; otherwise a new
        ReasoningTitle is appended.

        Args:
            text: The title text.
            lang: Two-letter language code (default: "en").
            master: Whether this is the master title.
        """
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_title', self._rd_id, 'set title'))
            return
        if text:
            text = re.sub(r"<br\s*/?>", LINE_SEPARATOR, text)
        if master:
            for t in self._rd_titles:
                if t.master():
                    t.set_master(False)
        existing = self.get_title_for_lang(lang)
        if existing is not None:
            existing.set_text(text)
            if master:
                existing.set_master(True)
        else:
            self._rd_titles.append(ReasoningTitle(text=text, lang=lang, master=master))

    def short_titles(self) -> List[ReasoningTitle]:
        """Get all language-specific short titles for this division."""
        return self._rd_short_titles

    def set_short_titles(self, short_titles: List[ReasoningTitle]) -> None:
        """Replace the complete list of short titles."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_short_titles', self._rd_id, 'set short titles'))
            return
        self._rd_short_titles = list(short_titles) if short_titles is not None else []

    def add_short_title(self, t: ReasoningTitle) -> None:
        """Append a ReasoningTitle to the short titles list."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'add_short_title', self._rd_id, 'add short title'))
            return
        if self._rd_short_titles is None:
            self._rd_short_titles = []
        self._rd_short_titles.append(t)

    def clear_short_titles(self) -> None:
        """Remove all short titles from this division."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'clear_short_titles', self._rd_id, 'clear short titles'))
            return
        self._rd_short_titles = []

    def get_short_title_for_lang(self, lang: str) -> 'ReasoningTitle | None':
        """Return the short ReasoningTitle for the given language code, or None."""
        lang_lower = (lang or "").lower().strip()
        for t in self._rd_short_titles:
            if t.lang().lower() == lang_lower:
                return t
        return None

    def get_short_title_master(self) -> 'ReasoningTitle | None':
        """Return the master short ReasoningTitle object.

        Logic mirroring get_title_master().
        Returns None if no short titles exist.
        """
        if not self._rd_short_titles:
            return None
        if len(self._rd_short_titles) == 1:
            return self._rd_short_titles[0]
        for t in self._rd_short_titles:
            if t.master():
                return t
        for t in self._rd_short_titles:
            if t.lang().lower() == "en":
                return t
        return self._rd_short_titles[0]

    def short_title(self, lang: str = None) -> str:
        """Get short title text (backward-compatible convenience accessor).

        - If *lang* is specified, returns the text of the matching language
          variant (or None if not found).
        - Otherwise returns the text of the master short title, or None.
        """
        if not self._rd_short_titles:
            return None
        if lang is not None:
            t = self.get_short_title_for_lang(lang)
            return t.text() if t else None
        master = self.get_short_title_master()
        return master.text() if master else None

    def set_short_title(self, text: str, lang: str = "en", master: bool = True) -> None:
        """Set (add or replace) the short title for a given language.

        If *master* is True the master flag is transferred to this entry,
        clearing it from any previously-flagged entry.  If a short title for
        *lang* already exists it is updated in-place; otherwise a new
        ReasoningTitle is appended.

        Args:
            text: The short title text.
            lang: Two-letter language code (default: "en").
            master: Whether this is the master short title.
        """
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_short_title', self._rd_id, 'set short_title'))
            return
        if text:
            text = re.sub(r"<br\s*/?>", LINE_SEPARATOR, text)
        if master:
            for t in self._rd_short_titles:
                if t.master():
                    t.set_master(False)
        existing = self.get_short_title_for_lang(lang)
        if existing is not None:
            existing.set_text(text)
            if master:
                existing.set_master(True)
        else:
            self._rd_short_titles.append(ReasoningTitle(text=text, lang=lang, master=master))

    def sequence(self) -> str:
        """Get the division sequence (structural metadata, never changes after init)."""
        return self._rd_sequence

    def set_sequence(self, sequence: str) -> None:
        """Set the sequence text for this division"""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_sequence', self._rd_id, 'set sequence'))
            return
        self._rd_sequence = sequence

    def instructions(self) -> Dict[str, Dict[str, str]]:
        """Get the instructions dictionary."""
        return self._rd_instructions

    def set_instructions(self, instructions: Dict[str, Dict[str, str]]) -> None:
        """Set the instructions dictionary for this division."""
        self._rd_instructions = instructions

    def summaries(self) -> List[ReasoningSummary]:
        """Get all language-specific summaries for this division."""
        return self._rd_summaries

    def set_summaries(self, summaries: List[ReasoningSummary]) -> None:
        """Replace the complete list of summaries."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_summaries', self._rd_id, 'set summaries'))
            return
        self._rd_summaries = list(summaries) if summaries is not None else []

    def add_summary(self, s: ReasoningSummary) -> None:
        """Append a ReasoningSummary to the list."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'add_summary', self._rd_id, 'add summary'))
            return
        if self._rd_summaries is None:
            self._rd_summaries = []
        self._rd_summaries.append(s)

    def clear_summaries(self) -> None:
        """Remove all summaries from this division."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'clear_summaries', self._rd_id, 'clear summaries'))
            return
        self._rd_summaries = []

    def get_summary_for_lang(self, lang: str) -> 'ReasoningSummary | None':
        """Return the ReasoningSummary for the given language code, or None."""
        lang_lower = (lang or "").lower().strip()
        for s in self._rd_summaries:
            if s.lang().lower() == lang_lower:
                return s
        return None

    def get_summary_master(self) -> 'ReasoningSummary | None':
        """Return the master ReasoningSummary object.

        Logic:
        1. If there's only one summary, it's master by definition (regardless of attribute)
        2. If there's no summary with master=True, default to English language summary
        3. Otherwise return the summary marked as master=True
        Returns None if no summaries exist.
        """
        if not self._rd_summaries:
            return None

        # If only one summary, it's master by definition
        if len(self._rd_summaries) == 1:
            return self._rd_summaries[0]

        # Look for explicitly marked master
        for s in self._rd_summaries:
            if s.master():
                return s

        # No explicit master; default to English
        for s in self._rd_summaries:
            if s.lang().lower() == "en":
                return s

        # Fallback: return first summary if no English found
        return self._rd_summaries[0] if self._rd_summaries else None

    def summary(self, lang: str = None) -> str:
        """Get summary text (backward-compatible convenience accessor).

        - If *lang* is specified, returns the text of the matching language
          variant (or None if not found).
        - Otherwise returns the text of the master summary, falling back to
          the first available summary, or None if there are none.
        """
        if not self._check_read_allowed():
            log(WARNING, log_read_context_warning('ReasoningDivision', 'summary', self._rd_id))
        if not self._rd_summaries:
            return None
        if lang is not None:
            s = self.get_summary_for_lang(lang)
            return s.text() if s else None
        # Return master first, then first element
        for s in self._rd_summaries:
            if s.master():
                return s.text()
        return self._rd_summaries[0].text()

    def set_summary(self, text: str, lang: str = "en", master: bool = True, version: int | None = None) -> None:
        """Set (add or replace) the summary for a given language.

        If *master* is True the master flag is transferred to this entry,
        clearing it from any previously-flagged entry.  If a summary for
        *lang* already exists it is updated in-place; otherwise a new
        ReasoningSummary is appended.

        Args:
            text: The summary text.
            lang: Two-letter language code (default: "en").
            master: Whether this is the master summary.
            version: If provided, sets the version counter to this exact value
                     instead of auto-incrementing. Use this when updating a
                     translation to match the master's version number.
        """
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_summary', self._rd_id, 'set summary'))
            return
        if text:
            text = re.sub(r"<br\s*/?>", LINE_SEPARATOR, text)
        if master:
            for s in self._rd_summaries:
                if s.master():
                    s.set_master(False)
        existing = self.get_summary_for_lang(lang)
        if existing is not None:
            existing.set_text(text)
            if master:
                existing.set_master(True)
            if version is not None:
                existing.set_version(version)
            else:
                existing.increment_version()
        else:
            self._rd_summaries.append(ReasoningSummary(text=text, lang=lang, master=master, version=version if version is not None else 0))

    def issues(self) -> List['ReasoningIssue']:
        """Get the list of issues for this division."""
        if not self._check_read_allowed():
            log(WARNING, log_read_context_warning('ReasoningDivision', 'issues', self._rd_id))
        return self._rd_issues

    def set_issues(self, issues: List['ReasoningIssue | Dict']) -> None:
        """Set the list of issues for this division."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_issues', self._rd_id, 'set issues'))
            return
        new_issues = []
        for issue in issues:
            if isinstance(issue, ReasoningIssue):
                issue._ri_parent_division = self
                new_issues.append(issue)
            elif isinstance(issue, dict):
                new_issue = ReasoningIssue.from_dict(issue)
                new_issue._ri_parent_division = self
                new_issues.append(new_issue)

        self._rd_issues = new_issues

    def add_issue(self, issue: 'ReasoningIssue | Dict') -> None:
        """Add an issue to this division."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'add_issue', self._rd_id, 'add issue'))
            return
        if self._rd_issues is None:
            self._rd_issues = []

        if isinstance(issue, dict):
            issue = ReasoningIssue.from_dict(issue)

        issue._ri_parent_division = self
        self._rd_issues.append(issue)

    def get_issue_by_id(self, issue_id: str) -> 'ReasoningIssue':
        """Get a specific issue by its ID."""
        if not self._check_read_allowed():
            log(WARNING, log_read_context_warning('ReasoningDivision', 'get_issue_by_id', self._rd_id))
        for issue in self._rd_issues:
            if issue.id() == issue_id:
                return issue
        return None

    def recommendations(self) -> List[str]:
        """Get the list of recommendations for this division."""
        return self._rd_recommendations if self._rd_recommendations is not None else []

    def set_recommendations(self, recommendations: List[str]) -> None:
        """Set the list of recommendations for this division."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_recommendations', self._rd_id, 'set recommendations'))
            return
        self._rd_recommendations = recommendations if recommendations is not None else []

    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation to this division."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'add_recommendation', self._rd_id, 'add recommendation'))
            return
        if self._rd_recommendations is None:
            self._rd_recommendations = []
        self._rd_recommendations.append(recommendation)

    def remove_recommendation(self, recommendation: str) -> None:
        """Remove all occurrences of a recommendation from this division."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'remove_recommendation', self._rd_id, 'remove recommendation'))
            return
        if self._rd_recommendations is None:
            return
        self._rd_recommendations = [r for r in self._rd_recommendations if r != recommendation]

    def clear_recommendations(self) -> None:
        """Clear all recommendations for this division."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'clear_recommendations', self._rd_id, 'clear recommendations'))
            return
        self._rd_recommendations = []

    def keywords(self) -> str:
        """Get the division keywords."""
        return self._rd_keywords

    def set_keywords(self, keywords: str) -> None:
        """Set the keywords text for this division"""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_keywords', self._rd_id, 'set keywords'))
            return
        # Replace all HTML <br> tags with XML-compliant <br/>
        if keywords:
            # Replace <br>, <br/>, <br > with <br/>
            keywords = re.sub(r"<br\s*/?>", LINE_SEPARATOR, keywords)
        self._rd_keywords = keywords

    def references(self) -> str:
        """Get the division references."""
        return self._rd_references

    def set_references(self, references: str) -> None:
        """Set the references text for this division"""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_references', self._rd_id, 'set references'))
            return
        # Replace all HTML <br> tags with XML-compliant <br/>
        if references:
            # Replace <br>, <br/>, <br > with <br/>
            references = re.sub(r"<br\s*/?>", LINE_SEPARATOR, references)
        self._rd_references = references

    def prompt(self) -> str:
        """Get the division prompt."""
        if not self._check_read_allowed():
            log(WARNING, log_read_context_warning('ReasoningDivision', 'prompt', self._rd_id))
        return self._rd_prompt

    def set_prompt(self, prompt: str) -> None:
        """Set the prompt text for this division"""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_prompt', self._rd_id, 'set prompt'))
            return
        # Replace all HTML <br> tags with XML-compliant <br/>
        if prompt:
            # Replace <br>, <br/>, <br > with <br/>
            prompt = re.sub(r"<br\s*/?>", LINE_SEPARATOR, prompt)
        self._rd_prompt = prompt

    def focus_content(self) -> str:
        """Get the division focus content as a comma-separated string (structural metadata)."""
        return self._rd_focus_content

    def set_focus_content(self, focus_content: List[str] | str) -> None:
        """Set the focus content for this division.

        Accepts either a list of strings (will be joined into a comma-separated string)
        or a prebuilt comma-separated string.
        """
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_focus_content', self._rd_id, 'set focus_content'))
            return
        if isinstance(focus_content, list):
            cleaned = [str(x).strip() for x in focus_content if x is not None and str(x).strip() != ""]
            self._rd_focus_content = ",".join(cleaned) if cleaned else None
        else:
            self._rd_focus_content = focus_content

    def focus_content_split_to_chunks(self) -> bool:
        """Get the division focus content split to chunks flag (structural metadata)."""
        return self._rd_focus_content_split_to_chunks

    def set_focus_content_split_to_chunks(self, split_to_chunks: bool) -> None:
        """Set the focus content split to chunks flag for this division"""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_focus_content_split_to_chunks', self._rd_id, 'set focus_content_split_to_chunks'))
            return
        self._rd_focus_content_split_to_chunks = split_to_chunks

    def _get_placeholder_map(self) -> Dict[str, str]:
        """Get the placeholder map from the parent document.

        This is an internal method used by instruction getters to apply placeholder translations.
        The map is stored at the ReasoningDocument level, not at the division level.
        """
        # This will be overridden when accessed through a parent document context
        return {}

    def _apply_placeholder_translations(self, text: str, placeholder_map: Dict[str, str] | None = None) -> str:
        """Apply placeholder translations to text.

        Replaces placeholders (e.g., "%LANGUAGE%") with their corresponding values from the placeholder_map.
        If no placeholder_map is provided, uses the division's internal placeholder_map.

        Args:
            text: The text containing placeholders
            placeholder_map: Optional dict mapping placeholder names to values (e.g., {"%LANGUAGE%": "Italian"})

        Returns:
            The text with all placeholders replaced by their values
        """
        if not text:
            return text

        # Use provided placeholder_map, fallback to division's internal map
        map_to_use = placeholder_map if placeholder_map is not None else self._rd_placeholder_map

        # If no placeholder map is available, return text unchanged
        if not map_to_use:
            return text

        # Replace each placeholder with its value
        result = text
        for placeholder, value in map_to_use.items():
            result = result.replace(placeholder, value)

        return result

    def results(self) -> List['ReasoningResult']:
        """Get the list of results."""
        if not self._check_read_allowed():
            log(WARNING, log_read_context_warning('ReasoningDivision', 'results', self._rd_id))
        return self._rd_results

    def tasks(self) -> List['ReasoningTask']:
        """Get the list of tasks."""
        if not self._check_read_allowed():
            log(WARNING, log_read_context_warning('ReasoningDivision', 'tasks', self._rd_id))
        return self._rd_tasks

    def to_dict(self, master_language_only: bool = True) -> dict:
        """Convert division to dictionary.
        
        Args:
            master_language_only (bool): If True, include only master language titles and short titles.
                                        Defaults to True to exclude non-master language variants.
        """
        # Filter titles to master language only if requested
        if master_language_only:
            master_titles = [t.to_dict() for t in self._rd_titles if t.master()]
            titles_to_include = master_titles if master_titles else [t.to_dict() for t in self._rd_titles[:1]]
            
            master_short_titles = [t.to_dict() for t in self._rd_short_titles if t.master()]
            short_titles_to_include = master_short_titles if master_short_titles else [t.to_dict() for t in self._rd_short_titles[:1]]
        else:
            titles_to_include = [t.to_dict() for t in self._rd_titles]
            short_titles_to_include = [t.to_dict() for t in self._rd_short_titles]
        
        return {
            "division_id": self._rd_id,
            "status": self._rd_status.value,
            "titles": titles_to_include,
            "title": self.title(),  # backward-compat: master title text
            "short_titles": short_titles_to_include,
            "short_title": self.short_title(),  # backward-compat: master short title text
            "sequence": self._rd_sequence,
            "focus_content": self._rd_focus_content,
            "focus_content_split_to_chunks": self._rd_focus_content_split_to_chunks,
            "instructions": self._rd_instructions,
            "summaries": [s.to_dict() for s in self._rd_summaries],
            "summary": self._rd_summaries[0].text() if self._rd_summaries else None,  # backward-compat
            "keywords": self._rd_keywords,
            "references": self._rd_references,
            "prompt": self._rd_prompt,
            "results": [result.to_dict() for result in self._rd_results],
            "tasks": [task.to_dict() for task in self._rd_tasks]
        }

    def is_complete(self) -> bool:
        return self._rd_status == ReasoningStatus.RS_COMPLETE

    def is_open(self) -> bool:
        return self._rd_status == ReasoningStatus.RS_OPEN

    def add_task(self, task: ReasoningTask) -> None:
        """Add a task to this division"""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'add_task', self._rd_id, 'add task'))
            return
        task._rt_parent_division = self
        self._rd_tasks.append(task)

    def add_new_task(self, description: str, task_id: str = None, status: TaskStatus = TaskStatus.TS_OPEN,
                     recommended_plan: str = "", recommended_tools: str = "") -> ReasoningTask:
        """Add a new task with the given description"""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'add_new_task', self._rd_id, 'add new task'))
            return None
        if task_id is None:
            # Always generate sequential zero-padded 3-digit task IDs (t-001, t-002, ...)
            max_num = 0
            for task in self._rd_tasks:
                if task._rt_task_id:
                    m = re.search(r"t-(\d{1,})$", str(task._rt_task_id))
                    if m:
                        try:
                            num = int(m.group(1))
                            if num > max_num:
                                max_num = num
                        except Exception:
                            pass
            task_number = max_num + 1
            task_id = f"t-{task_number:03d}"
        new_task = ReasoningTask(task_id=task_id, status=status, description=description, recommended_plan=recommended_plan, recommended_tools=recommended_tools)
        self.add_task(new_task)
        return new_task

    def get_incomplete_tasks(self) -> List[ReasoningTask]:
        """Get all tasks that are not complete"""
        return [task for task in self._rd_tasks if not task.is_complete()]

    def get_task_by_id(self, task_id: str) -> ReasoningTask:
        """Get a specific task by its ID"""
        if not self._check_read_allowed():
            log(WARNING, log_read_context_warning('ReasoningDivision', 'get_task_by_id', self._rd_id))
        stripped_task_id = ReasoningTask.strip_id(task_id)
        for task in self._rd_tasks:
            if task._rt_task_id == stripped_task_id:
                return task
        return None

    def get_tasks_by_status(self, status: TaskStatus) -> List[ReasoningTask]:
        """Get all tasks with a specific status (TaskStatus or str)"""
        if isinstance(status, str):
            status = TaskStatus(status)
        return [task for task in self._rd_tasks if task._rt_status == status]

    def get_instructions_for_current_status(self, placeholder_map: Dict[str, str] | None = None) -> str:
        """Get instructions text for the current status.

        Args:
            placeholder_map: Optional dict mapping placeholder names to values.
            include_common_instructions: When True (default), common snippets referenced
                via include_common_instructions_ids are prepended by the parent document.
                Pass False to retrieve the raw division-level text only.
        """
        instruction_data = self._rd_instructions.get(self._rd_status.value, {})
        if isinstance(instruction_data, str):
            # Handle legacy format where instructions was just a string
            return self._apply_placeholder_translations(instruction_data, placeholder_map)
        
        text = instruction_data.get("text", "")
        return self._apply_placeholder_translations(text, placeholder_map)

    def get_instructions_for_status(self, status: str, placeholder_map: Dict[str, str] | None = None) -> str:
        """Get instructions text for a specific status"""
        instruction_data = self._rd_instructions.get(status, {})
        if isinstance(instruction_data, str):
            # Handle legacy format where instructions was just a string
            return self._apply_placeholder_translations(instruction_data, placeholder_map)
        text = instruction_data.get("text", "")
        return self._apply_placeholder_translations(text, placeholder_map)

    def set_instructions_for_status(self, status: str, instruction_text: str,
                                    recommended_tools: str = "", recommended_plan: str = "",
                                    action: DivisionAction = DivisionAction.NONE) -> None:
        """Set instructions for a specific status with optional recommended tools and plan"""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'set_instructions_for_status', self._rd_id, 'set instructions for status'))
            return
        self._rd_instructions[status] = {
            "text": instruction_text,
            "recommended_tools": recommended_tools,
            "recommended_plan": recommended_plan,
            "action": action.value if isinstance(action, DivisionAction) else action
        }

    def get_recommended_tools_for_status(self, status: str, placeholder_map: Dict[str, str] | None = None) -> str:
        """Get recommended tools for a specific status, defaulting to DEFAULT_RECOMMENDED_TOOLS if missing or empty"""
        instruction_data = self._rd_instructions.get(status, {})
        if isinstance(instruction_data, dict):
            tools = instruction_data.get("recommended_tools", DEFAULT_RECOMMENDED_TOOLS)
        else:
            tools = DEFAULT_RECOMMENDED_TOOLS
        return self._apply_placeholder_translations(tools, placeholder_map) if tools else tools

    def get_recommended_plan_for_status(self, status: str, placeholder_map: Dict[str, str] | None = None) -> str:
        """Get recommended plan for a specific status"""
        instruction_data = self._rd_instructions.get(status, {})
        if isinstance(instruction_data, dict):
            recommended_plan = instruction_data.get("recommended_plan", DEFAULT_RECOMMENDED_PLAN)
        else:
            recommended_plan = DEFAULT_RECOMMENDED_PLAN

        if recommended_plan is None or len(recommended_plan.strip()) == 0:
            recommended_plan = DEFAULT_RECOMMENDED_PLAN

        return self._apply_placeholder_translations(recommended_plan, placeholder_map)

    def get_recommended_tools_for_current_status(self) -> str:
        """Get recommended tools for the current status, defaulting to DEFAULT_RECOMMENDED_PLAN if missing or empty"""
        return self.get_recommended_tools_for_status(self._rd_status.value)

    def get_recommended_plan_for_current_status(self, placeholder_map: Dict[str, str] | None = None) -> str:
        """Get recommended plan for the current status"""
        return self.get_recommended_plan_for_status(self._rd_status.value, placeholder_map)

    def update_result(self, for_task_id: str, result_summary: str) -> None:
        """Update the results related to a specific task ID"""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningDivision', 'update_result', self._rd_id, 'update result'))
            return
        # Find existing result for the task, then update it; if not found, append a new one
        stripped_for_task_id = ReasoningTask.strip_id(for_task_id)
        found = False
        for result in self._rd_results:
            if ReasoningTask.strip_id(result._rr_for_task_id) == stripped_for_task_id:
                # Update the existing result
                if result.result() != result_summary:
                    log(WARNING, f"Overwriting existing result for task ID '{stripped_for_task_id}' in division '{self._rd_id}'.\nOld result:\n{result.result()}\nNew result:\n{result_summary}")

                result.set_result(result_summary)
                found = True
                log(DEBUG, f"Updated result for task ID '{stripped_for_task_id}' in division '{self._rd_id}':\n{result_summary}")
                break

        if not found:
            # If no existing result found, create a new one
            new_result = ReasoningResult(for_task_id=stripped_for_task_id, result=result_summary)
            new_result._rr_parent_division = self
            self._rd_results.append(new_result)
            log(DEBUG, f"Added new result for task ID '{stripped_for_task_id}' in division '{self._rd_id}':\n{result_summary}")

    def to_xml(self, escape: bool = False, remove_information: Dict[str, bool] = None, exclude_task_status: List[TaskStatus] = None) -> str:
        """
        Convert the ReasoningDivision to XML format with conditional content removal.
        Args:
            escape (bool): Whether to escape XML special characters (&, <, >, ", '). Defaults to False.
            remove_information (Dict[str, bool], optional): Dictionary specifying which content types to exclude.
                Supported keys:
                - "remove_tasks": Exclude tasks sections
                - "remove_results": Exclude results sections
                - "remove_prompt": Exclude prompt sections
                - "remove_focus_content": Exclude focus content sections
                - "remove_keywords": Exclude keywords/metadata sections
                - "remove_issues": Exclude issues sections
                - "remove_recommendations": Exclude recommendations sections
                - "remove_summary": Exclude summary sections
                - "remove_instructions": Exclude instructions sections
                - "master_language_only": Collapse multi-language blocks to a single flat master-language element
                Defaults to None (include all content).
            exclude_task_status (List[TaskStatus], optional): List of task statuses to exclude from tasks and results.
                Defaults to None (include all statuses).
        Returns:
            str: XML representation of the division.
        """
        remove_tasks = remove_information["remove_tasks"] if remove_information and "remove_tasks" in remove_information else False
        remove_results = remove_information["remove_results"] if remove_information and "remove_results" in remove_information else False
        remove_prompt = remove_information["remove_prompt"] if remove_information and "remove_prompt" in remove_information else False
        remove_focus_content = remove_information["remove_focus_content"] if remove_information and "remove_focus_content" in remove_information else False
        remove_keywords = remove_information["remove_keywords"] if remove_information and "remove_keywords" in remove_information else False
        remove_issues = remove_information["remove_issues"] if remove_information and "remove_issues" in remove_information else False
        remove_recommendations = remove_information["remove_recommendations"] if remove_information and "remove_recommendations" in remove_information else False
        remove_summary = remove_information["remove_summary"] if remove_information and "remove_summary" in remove_information else False
        remove_instructions = remove_information["remove_instructions"] if remove_information and "remove_instructions" in remove_information else False
        master_language_only = remove_information["master_language_only"] if remove_information and "master_language_only" in remove_information else False

        html_parts = []
        division_id = self._rd_id or "division"
        cycle_attr = f' cycle="{self._rd_cycle}"' if hasattr(self, "_rd_cycle") and self._rd_cycle != 0 else ""
        scope_attr = f' scope="{self._rd_scope.value.lower()}"' if hasattr(self, "_rd_scope") and self._rd_scope != DivisionScope.PRIVATE else ""
        execution_attr = f' execution="{self._rd_execution.value.lower()}"' if hasattr(self, "_rd_execution") and self._rd_execution != DivisionExecution.SEQUENTIAL else ""
        indentation_attr = f' indentation="{self._rd_indentation}"' if hasattr(self, "_rd_indentation") and self._rd_indentation != 0 else ""
        visibility_attr = f' visibility="{self._rd_visibility.value.lower()}"' if hasattr(self, "_rd_visibility") and self._rd_visibility != DivisionVisibility.VISIBLE else ""
        sequence_attr = f' sequence="{xml_escape(self._rd_sequence, escape)}"' if hasattr(self, "_rd_sequence") and self._rd_sequence is not None and self._rd_sequence.strip() else ""
        html_parts.append(f'<division id="{xml_escape(division_id, escape)}"{cycle_attr}{scope_attr}{execution_attr}{indentation_attr}{visibility_attr}{sequence_attr}>')
        html_parts.append(f'    <status>{xml_escape(self._rd_status.value, escape)}</status>')
        if self._rd_titles:
            non_empty_titles = [t for t in self._rd_titles if t.text() and t.text().strip()]
            if non_empty_titles:
                if master_language_only:
                    master_title = next((t for t in non_empty_titles if t.master()), non_empty_titles[0])
                    html_parts.append(f'    <title>{xml_escape(master_title.text(), escape)}</title>')
                else:
                    html_parts.append('    <titles>')
                    for t in non_empty_titles:
                        html_parts.append(t.to_xml(escape, element_name="title"))
                    html_parts.append('    </titles>')
        if self._rd_short_titles:
            non_empty_short_titles = [t for t in self._rd_short_titles if t.text() and t.text().strip()]
            if non_empty_short_titles:
                if master_language_only:
                    master_short_title = next((t for t in non_empty_short_titles if t.master()), non_empty_short_titles[0])
                    html_parts.append(f'    <shortTitle>{xml_escape(master_short_title.text(), escape)}</shortTitle>')
                else:
                    html_parts.append('    <shortTitles>')
                    for t in non_empty_short_titles:
                        html_parts.append(t.to_xml(escape, element_name="shortTitle"))
                    html_parts.append('    </shortTitles>')

        # Add focusContent tag if content exists and not removed
        if not remove_focus_content and self._rd_focus_content and self._rd_focus_content.strip():
            split_attr = ""
            if self._rd_focus_content_split_to_chunks is not None:
                split_value = "true" if self._rd_focus_content_split_to_chunks else "false"
                split_attr = f' splitToChunks="{split_value}"'
            html_parts.append(f'    <focusContent{split_attr}>{xml_escape(self._rd_focus_content, escape)}</focusContent>')

        if not remove_instructions and hasattr(self, '_rd_instructions') and self._rd_instructions is not None and len(self._rd_instructions) > 0:
            for state, instruction_data in self._rd_instructions.items():
                text = instruction_data.get("text", "")
                recommended_tools = instruction_data.get("recommended_tools", "")
                recommended_plan = instruction_data.get("recommended_plan", "")
                action = instruction_data.get("action", DivisionAction.NONE.value)
                instruction_id = instruction_data.get("id", "")
                include_ids = instruction_data.get("include_common_instructions_ids", "")
                attrs = [f'for_status="{xml_escape(state, escape)}"']
                if instruction_id:
                    attrs.append(f'id="{xml_escape(instruction_id, escape)}"')
                if include_ids:
                    attrs.append(f'include_common_instructions_ids="{xml_escape(include_ids, escape)}"')
                if recommended_tools:
                    attrs.append(f'recommended_tools="{xml_escape(recommended_tools, escape)}"')
                if recommended_plan:
                    attrs.append(f'recommended_plan="{xml_escape(recommended_plan, escape)}"')
                # Only include action if not NONE
                if action and action != DivisionAction.NONE.value:
                    attrs.append(f'action="{xml_escape(action, escape)}"')
                html_parts.append(f'    <instructions {" ".join(attrs)}>')
                # preserve arbitrary HTML/text by using CDATA (e.g., for content with embedded XML tags like <LOADED_DOCUMENTS>)
                html_parts.append(f'        {xml_cdata(text)}')
                html_parts.append('    </instructions>')

        if not remove_summary and self._rd_summaries:
            non_empty_summaries = [s for s in self._rd_summaries if s.text() and s.text().strip()]
            if non_empty_summaries:
                if master_language_only:
                    master_summary = next((s for s in non_empty_summaries if s.master()), non_empty_summaries[0])
                    html_parts.append(f'    <summary><![CDATA[{master_summary.text()}]]></summary>')
                else:
                    html_parts.append('    <summaries>')
                    for s in non_empty_summaries:
                        html_parts.append(s.to_xml(escape))
                    html_parts.append('    </summaries>')

        if not remove_keywords and ((self._rd_keywords and self._rd_keywords.strip()) or (self._rd_references and self._rd_references.strip())):
            html_parts.append('    <metadata>')
            if self._rd_keywords and self._rd_keywords.strip():
                html_parts.append(f'        <keywords>{xml_escape(self._rd_keywords, escape)}</keywords>')
            if self._rd_references and self._rd_references.strip():
                html_parts.append(f'        <references>{xml_escape(self._rd_references, escape)}</references>')
            html_parts.append('    </metadata>')

        if not remove_prompt and self._rd_prompt and self._rd_prompt.strip():
            html_parts.append(f'    <prompt>')
            # preserve arbitrary HTML/text by using CDATA
            html_parts.append(f'        {xml_cdata(self._rd_prompt)}')
            html_parts.append('    </prompt>')

        if not remove_tasks and self._rd_tasks:
            html_parts.append('    <tasks>')
            for task in self._rd_tasks:
                # Skip tasks with excluded statuses
                # Ensure read protection for task.status() call
                if exclude_task_status:
                    if self._rd_parent_document and self._rd_parent_document._check_read_allowed():
                        # Already in a read context, call directly
                        task_status = task.status()
                    else:
                        # Not in a read context, use read_context from parent document
                        if self._rd_parent_document:
                            with self._rd_parent_document.read_context():
                                task_status = task.status()
                        else:
                            task_status = task.status()
                    if task_status in exclude_task_status:
                        continue
                html_parts.append(f'        {task.to_xml(escape)}')
            html_parts.append('    </tasks>')

        if not remove_results and self._rd_results and len(self._rd_results) > 0:
            html_parts.append(f'    <results>')
            for result in self._rd_results:
                # Skip results for tasks with excluded statuses
                if exclude_task_status:
                    task = self.get_task_by_id(result._rr_for_task_id)
                    if task:
                        # Ensure read protection for task.status() call
                        if self._rd_parent_document and self._rd_parent_document._check_read_allowed():
                            # Already in a read context, call directly
                            task_status = task.status()
                        else:
                            # Not in a read context, use read_context from parent document
                            if self._rd_parent_document:
                                with self._rd_parent_document.read_context():
                                    task_status = task.status()
                            else:
                                task_status = task.status()
                        if task_status in exclude_task_status:
                            continue
                html_parts.append(f'        {result.to_xml(escape)}')
            html_parts.append('    </results>')

        if not remove_issues and self._rd_issues and len(self._rd_issues) > 0:
            html_parts.append(f'    <issues>')
            for issue in self._rd_issues:
                html_parts.append(f'        {issue.to_xml(escape)}')
            html_parts.append('    </issues>')

        if not remove_recommendations and hasattr(self, '_rd_recommendations') and self._rd_recommendations and len(self._rd_recommendations) > 0:
            html_parts.append('    <recommendations>')
            for rec in self._rd_recommendations:
                # preserve arbitrary HTML/text by using CDATA
                html_parts.append(f'        <recommendation>{xml_cdata(rec)}</recommendation>')
            html_parts.append('    </recommendations>')

        html_parts.append('</division>')
        return '\n'.join(html_parts)


log_end_file(__file__)
