from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import re
from typing import TYPE_CHECKING

from src.core.base.enums import IssueStatus, IssueSeverity
from src.core.base.reasoning.xml_utils import xml_escape, xml_cdata, log_write_context_error, log_read_context_warning, LINE_SEPARATOR
from src.utils.log import log, WARNING, ERROR

if TYPE_CHECKING:
    from src.core.base.reasoning.ReasoningDivision import ReasoningDivision


class ReasoningIssue:
    """Represents an issue to be reported to the user"""

    def __init__(self, issue_id: str | None = None, description: str = "", status: IssueStatus | str = IssueStatus.OPEN, severity: IssueSeverity | str = IssueSeverity.LOW) -> None:
        self._ri_issue_id: str | None = issue_id
        self._ri_description: str = description
        self._ri_status: IssueStatus = status if isinstance(status, IssueStatus) else IssueStatus(status)
        self._ri_severity: IssueSeverity = severity if isinstance(severity, IssueSeverity) else IssueSeverity(severity)
        self._ri_parent_division: 'ReasoningDivision' | None = None

    def _check_write_allowed(self) -> bool:
        """Check if write operations are allowed via parent document lock."""
        if self._ri_parent_division is None or self._ri_parent_division._rd_parent_document is None:
            return True  # No parent = allow (e.g., during construction)
        return self._ri_parent_division._rd_parent_document._check_write_allowed()

    def _check_read_allowed(self) -> bool:
        """Check if read operations are protected via parent document read context."""
        if self._ri_parent_division is None or self._ri_parent_division._rd_parent_document is None:
            return True  # No parent = allow (e.g., during construction)
        return self._ri_parent_division._rd_parent_document._check_read_allowed()

    @classmethod
    def from_dict(cls, data: dict) -> "ReasoningIssue":
        """
        Create a ReasoningIssue from a dictionary.
        Example input:
        {
            "issue_id": "i-123",
            "description": "Description of the issue",
            "severity": "low|medium|high",
            "status": "open|in progress|closed"
        }
        """
        issue_id = data.get("issue_id") or data.get("id")
        description = data.get("description", "")
        # Coerce status and severity to enums, fallback to defaults if invalid
        status_val = data.get("status", IssueStatus.OPEN.value)
        try:
            status = IssueStatus(status_val)
        except Exception:
            status = IssueStatus.OPEN
        severity_val = data.get("severity", IssueSeverity.LOW.value)
        try:
            severity = IssueSeverity(severity_val)
        except Exception:
            severity = IssueSeverity.LOW
        return cls(issue_id=issue_id, description=description, status=status, severity=severity)

    def id(self) -> str:
        """Get the issue ID (metadata, never changes)."""
        return self._ri_issue_id

    def description(self) -> str:
        """Get the issue description (metadata)."""
        return self._ri_description

    def set_description(self, description: str) -> None:
        """Set the issue description."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningIssue', 'set_description', self._ri_issue_id, 'set description'))
            return
        # Replace all HTML <br> tags with XML-compliant <br/>
        if description:
            # Replace <br>, <br/>, <br > with <br/>
            description = re.sub(r"<br\s*/?>", LINE_SEPARATOR, description)
        self._ri_description = description

    def status(self) -> IssueStatus:
        """Get the issue status."""
        if not self._check_read_allowed():
            log(WARNING, log_read_context_warning('ReasoningIssue', 'status', self._ri_issue_id))
        return self._ri_status

    def set_status(self, status: IssueStatus) -> None:
        """Set the status of this issue."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningIssue', 'set_status', self._ri_issue_id, 'set status'))
            return
        if isinstance(status, str):
            status = IssueStatus(status)
        self._ri_status = status

    def severity(self) -> IssueSeverity:
        """Get the issue severity."""
        if not self._check_read_allowed():
            log(WARNING, log_read_context_warning('ReasoningIssue', 'severity', self._ri_issue_id))
        return self._ri_severity

    def set_severity(self, severity: IssueSeverity) -> None:
        """Set the severity of this issue."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningIssue', 'set_severity', self._ri_issue_id, 'set severity'))
            return
        if isinstance(severity, str):
            severity = IssueSeverity(severity)
        self._ri_severity = severity

    def to_dict(self) -> dict:
        """Convert the issue to a dictionary representation."""
        return {
            "issue_id": self._ri_issue_id,
            "description": self._ri_description,
            "severity": self._ri_severity.value,
            "status": self._ri_status.value
        }

    def __str__(self) -> str:
        """String representation of the issue for debugging and logging."""
        return f"ReasoningIssue(issue_id='{self._ri_issue_id}', severity='{self._ri_severity.value}', status='{self._ri_status.value}', description='{self._ri_description[:50]}{'...' if len(self._ri_description) > 50 else ''}')"

    def __repr__(self) -> str:
        """Detailed representation of the issue."""
        return f"ReasoningIssue(issue_id='{self._ri_issue_id}', severity='{self._ri_severity.value}', status='{self._ri_status.value}', description='{self._ri_description}')"

    def to_xml(self, escape: bool = False) -> str:
        """Convert the issue to HTML format, using CDATA for description content"""
        html_parts = []
        if self._ri_issue_id:
            html_parts.append(f'<issue id="{xml_escape(self._ri_issue_id, escape)}"')
        else:
            html_parts.append('<issue')
        # Always include status attribute
        # Include severity attribute if present
        if self._ri_severity:
            html_parts.append(f' severity="{xml_escape(self._ri_severity.value, escape)}"')
        # Always include status attribute
        if self._ri_status:
            html_parts.append(f' status="{xml_escape(self._ri_status.value, escape)}"')
        html_parts.append(f'>')
        # preserve arbitrary HTML/text by using CDATA for the description
        if self._ri_description is None:
            html_parts.append('</issue>')
        else:
            html_parts.append(f'{xml_cdata(self._ri_description)}</issue>')
        return ''.join(html_parts)


log_end_file(__file__)
