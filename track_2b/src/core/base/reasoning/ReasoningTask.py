from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import re
from typing import TYPE_CHECKING

from src.core.base.enums import TaskStatus
from src.core.base.reasoning.xml_utils import (xml_escape, xml_cdata, log_write_context_error, log_read_context_warning,
                                                LINE_SEPARATOR, DEFAULT_RECOMMENDED_PLAN, DEFAULT_RECOMMENDED_TOOLS)
from src.utils.log import log, WARNING, ERROR

if TYPE_CHECKING:
    from src.core.base.reasoning.ReasoningDivision import ReasoningDivision


class ReasoningTask:
    """Represents a task in the reasoning template"""

    def __init__(self, task_id: str | None = None, status: TaskStatus | str = TaskStatus.TS_OPEN, description: str = "", recommended_plan: str = "", recommended_tools: str = "") -> None:
        self._rt_task_id: str | None = task_id
        self._rt_parent_division: 'ReasoningDivision' | None = None
        self._rt_status: TaskStatus = status if isinstance(status, TaskStatus) else TaskStatus(status)
        self._rt_description: str = description
        self._rt_recommended_plan: str = recommended_plan
        self._rt_recommended_tools: str = recommended_tools

    def _check_write_allowed(self) -> bool:
        """Check if write operations are allowed via parent document lock."""
        if self._rt_parent_division is None or self._rt_parent_division._rd_parent_document is None:
            return True  # No parent = allow (e.g., during construction)
        return self._rt_parent_division._rd_parent_document._check_write_allowed()

    def _check_read_allowed(self) -> bool:
        """Check if read operations are protected via parent document read context."""
        if self._rt_parent_division is None or self._rt_parent_division._rd_parent_document is None:
            return True  # No parent = allow (e.g., during construction)
        return self._rt_parent_division._rd_parent_document._check_read_allowed()

    def id(self) -> str:
        """Get the task ID (metadata, never changes)."""
        return self._rt_task_id

    def status(self) -> TaskStatus:
        """Get the task status."""
        if not self._check_read_allowed():
            log(WARNING, log_read_context_warning('ReasoningTask', 'status', self._rt_task_id))
        return self._rt_status

    def set_status(self, status: TaskStatus) -> None:
        """Set the status of this task"""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningTask', 'set_status', self._rt_task_id, 'set status'))
            return
        if isinstance(status, str):
            status = TaskStatus(status)
        elif not isinstance(status, TaskStatus):
            raise ValueError(f"Invalid status type: {type(status)}. Expected TaskStatus or str.")

        self._rt_status = status

    def description(self) -> str:
        """Get the task description (metadata)."""
        return self._rt_description

    def set_description(self, description: str) -> None:
        """Set the task description."""
        if not self._check_write_allowed():
            log(ERROR, log_write_context_error('ReasoningTask', 'set_description', self._rt_task_id, 'set description'))
            return
        # Replace all HTML <br> tags with XML-compliant <br/>
        if description:
            # Replace <br>, <br/>, <br > with <br/>
            description = re.sub(r"<br\s*/?>", LINE_SEPARATOR, description)
        self._rt_description = description

    def recommended_plan(self) -> str:
        """Get the recommended plan for this task (metadata)."""
        return self._rt_recommended_plan if self._rt_recommended_plan is not None else DEFAULT_RECOMMENDED_PLAN

    def recommended_tools(self) -> str:
        """Get the recommended tools for this task (metadata)."""
        return self._rt_recommended_tools if self._rt_recommended_tools is not None else DEFAULT_RECOMMENDED_TOOLS

    def to_dict(self) -> dict:
        """Convert the task to a dictionary representation"""
        recommended_plan = self._rt_recommended_plan if self._rt_recommended_plan else DEFAULT_RECOMMENDED_PLAN
        recommended_tools = self._rt_recommended_tools if self._rt_recommended_tools else DEFAULT_RECOMMENDED_TOOLS
        return {
            "task_id": self._rt_task_id,
            "status": self._rt_status.value,
            "description": self._rt_description,
            "recommended_plan": recommended_plan,
            "recommended_tools": recommended_tools
        }

    def is_complete(self) -> bool:
        """Check if the task is complete"""
        return self._rt_status == TaskStatus.TS_COMPLETE

    def __str__(self) -> str:
        """String representation of the task for debugging and logging"""
        return f"ReasoningTask(id='{self._rt_task_id}', status='{self._rt_status.value}', description='{self._rt_description[:50]}{'...' if len(self._rt_description) > 50 else ''}')"

    def __repr__(self) -> str:
        """Detailed representation of the task"""
        return f"ReasoningTask(task_id='{self._rt_task_id}', status='{self._rt_status.value}', description='{self._rt_description}', recommended_plan='{self._rt_recommended_plan}', recommended_tools='{self._rt_recommended_tools}')"

    def to_xml(self, escape: bool = False) -> str:
        """Convert the task to HTML format, escaping & for XML validity"""
        html_parts = []
        if self._rt_task_id:
            html_parts.append(f'<task id="{xml_escape(self._rt_task_id, escape)}"')
        else:
            html_parts.append('<task')
        # Always include status attribute
        if self._rt_status:
            html_parts.append(f' status="{xml_escape(self._rt_status.value, escape)}"')
        # Only include recommended_plan if not the default value
        if self._rt_recommended_plan is not None and self._rt_recommended_plan != DEFAULT_RECOMMENDED_PLAN:
            html_parts.append(f' recommended_plan="{xml_escape(self._rt_recommended_plan, escape)}"')
        if self._rt_recommended_tools is not None and self._rt_recommended_tools != DEFAULT_RECOMMENDED_TOOLS:
            html_parts.append(f' recommended_tools="{xml_escape(self._rt_recommended_tools, escape)}"')
        # preserve arbitrary HTML/text (including raw < and >) by using CDATA
        html_parts.append(f'>{xml_cdata(self._rt_description)}</task>')
        return ''.join(html_parts)

    @classmethod
    def strip_id(cls, task_id: str) -> str:
        """Strip the right of '#' prefix from the task ID if it exists. Example: input 'div-1#t-2' returns 't-2'"""
        if '#' in task_id:
            return task_id.split('#', 1)[1]
        return task_id

    @classmethod
    def strip_division_id_from_task_id(cls, task_id: str) -> str:
        """Strip the left of '#' prefix from the task ID if it exists. Example: input 'div-1#t-2' returns 'div-1'"""
        if '#' in task_id:
            return task_id.split('#', 1)[0]
        return None


log_end_file(__file__)
