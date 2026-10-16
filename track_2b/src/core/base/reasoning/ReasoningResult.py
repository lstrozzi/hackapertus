from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import re
from typing import TYPE_CHECKING

from src.core.base.reasoning.xml_utils import xml_escape, xml_cdata, log_write_context_error, log_read_context_warning, LINE_SEPARATOR
from src.utils.log import log, WARNING, ERROR

if TYPE_CHECKING:
    from src.core.base.reasoning.ReasoningDivision import ReasoningDivision


class ReasoningResult:
    """Represents a result in the reasoning template"""

    def __init__(self, for_task_id: str | None = None, result: str = "") -> None:
        self._rr_for_task_id: str | None = for_task_id
        self._rr_result: str = result
        self._rr_parent_division: 'ReasoningDivision' | None = None

    def _check_write_allowed(self) -> bool:
        """Check if write operations are allowed via parent document lock."""
        if self._rr_parent_division is None or self._rr_parent_division._rd_parent_document is None:
            return True  # No parent = allow (e.g., during construction)
        return self._rr_parent_division._rd_parent_document._check_write_allowed()

    def _check_read_allowed(self) -> bool:
        """Check if read operations are protected via parent document read context."""
        if self._rr_parent_division is None or self._rr_parent_division._rd_parent_document is None:
            return True  # No parent = allow (e.g., during construction)
        return self._rr_parent_division._rd_parent_document._check_read_allowed()

    def to_dict(self) -> dict:
        """Convert the result to a dictionary representation"""
        return {
            "for_task_id": self._rr_for_task_id,
            "result": self._rr_result
        }

    def result(self) -> str:
        """Get the result text."""
        if not self._check_read_allowed():
            log(WARNING, log_read_context_warning('ReasoningResult', 'result', self._rr_for_task_id))
        return self._rr_result

    def set_result(self, result: str) -> None:
        """Set the result text."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningResult', 'set_result', self._rr_for_task_id, 'set result'))
            return
        # Replace all HTML <br> tags with XML-compliant <br/>
        if result:
            # Replace <br>, <br/>, <br > with <br/>
            result = re.sub(r"<br\s*/?>", LINE_SEPARATOR, result)
        self._rr_result = result

    def __str__(self) -> str:
        """String representation of the result for debugging and logging"""
        return f"ReasoningResult(for_task_id='{self._rr_for_task_id}', result='{self._rr_result[:50]}{'...' if len(self._rr_result) > 50 else ''}')"

    def __repr__(self) -> str:
        """Detailed representation of the result"""
        return f"ReasoningResult(for_task_id='{self._rr_for_task_id}', result='{self._rr_result}')"

    def to_xml(self, escape: bool) -> str:
        """Convert the result to HTML format, escaping & for XML validity"""
        html_parts = []
        if self._rr_for_task_id:
            html_parts.append(f'<result for_task_id="{xml_escape(self._rr_for_task_id, escape)}">')
        else:
            html_parts.append('<result>')
        # preserve arbitrary HTML/text by using CDATA for the result content
        if self._rr_result is None:
            html_parts.append('\n</result>')
        else:
            html_parts.append('\n')
            html_parts.append(xml_cdata(self._rr_result))
            html_parts.append('</result>')
        return ''.join(html_parts)


log_end_file(__file__)
