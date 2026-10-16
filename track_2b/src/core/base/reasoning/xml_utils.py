from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import inspect
from src.core.base.enums import DocumentType
from src.utils.log import log, WARNING, ERROR

DEFAULT_RECOMMENDED_PLAN = "+plan_call_tool_and_update_reasoning_document"
DEFAULT_RECOMMENDED_TOOLS = None
LINE_SEPARATOR = "\n"
REASONING_DOCUMENT_CONTENT_TYPE = DocumentType.XML.value


def xml_escape(text, escape: bool):
    """Escape special characters for XML/HTML validity"""
    if not text or not escape:
        return text
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#x27;"))

def xml_unescape(text, escape: bool):
    """Unescape XML/HTML entities back to original characters"""
    if not text or not escape:
        return text
    return (text.replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", '"')
                .replace("&#x27;", "'"))

def xml_cdata(text: str) -> str:
    """Wrap text in a CDATA section safely by splitting any occurrence of ']]>' so it cannot close the CDATA block prematurely."""
    if text is None:
        return '<![CDATA[]]>'
    # Split any terminating sequence inside the content to avoid breaking the CDATA block
    safe = str(text).replace(']]>', ']]]]><![CDATA[>')
    return f'<![CDATA[{safe}]]>'


def log_write_context_error(class_name: str, method_name: str, object_id: str, operation: str) -> str:
    """
    Generate a detailed error message with file location, line number, method name, and class name
    for missing write context protection.

    Args:
        class_name: Name of the class (e.g., 'ReasoningDocument', 'ReasoningDivision')
        method_name: Name of the method being called (e.g., 'set_status')
        object_id: ID of the object being modified
        operation: The operation being attempted (e.g., 'set status to RS_COMPLETE')

    Returns:
        A formatted error message string with file, line, method, and class info
    """
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename.split('\\')[-1]  # Get just the filename
    line_number = frame.f_lineno

    # Build stack trace showing the sequence of calls that led to this write without lock
    stack_lines = []
    current_frame = frame
    depth = 0
    max_depth = 15  # Limit stack depth to avoid excessive output

    while current_frame is not None and depth < max_depth:
        code_filename = current_frame.f_code.co_filename.split('\\')[-1]
        code_line = current_frame.f_lineno
        function_name = current_frame.f_code.co_name

        # Skip internal logging and context manager frames
        if function_name not in ['log_read_context_warning', 'log_write_context_error', 'log', '<module>']:
            try:
                import linecache
                source_line = linecache.getline(current_frame.f_code.co_filename, code_line).strip()
                if source_line:
                    stack_lines.append(f"  [{depth}] {code_filename}:{code_line} in {function_name}()\n      → {source_line[:100]}")
                else:
                    stack_lines.append(f"  [{depth}] {code_filename}:{code_line} in {function_name}()")
            except:
                stack_lines.append(f"  [{depth}] {code_filename}:{code_line} in {function_name}()")

        current_frame = current_frame.f_back
        depth += 1

    stack_trace_str = "\n".join(reversed(stack_lines)) if stack_lines else "  [no stack frames available]"
    return (f"Attempted to {method_name}() on {class_name} '{object_id}' at {filename}:{line_number} without write context lock - missing write_context() protection.\n"
            f"Operation: {operation}\n"
            f"Call sequence:\n{stack_trace_str}")

def log_read_context_warning(class_name: str, method_name: str, object_id: str) -> str:
    """
    Generate a warning message with file location, line number, method name, and class name
    for getter calls outside of read context protection. Includes full stack trace.

    Args:
        class_name: Name of the class (e.g., 'ReasoningDocument', 'ReasoningDivision')
        method_name: Name of the getter method being called (e.g., 'status')
        object_id: ID of the object being read

    Returns:
        A formatted warning message string with file, line, method, and class info, plus stack trace
    """
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename.split('\\')[-1]  # Get just the filename
    line_number = frame.f_lineno

    # Build stack trace showing the sequence of calls
    stack_lines = []
    current_frame = frame
    depth = 0
    max_depth = 15  # Limit stack depth to avoid excessive output

    while current_frame is not None and depth < max_depth:
        code_filename = current_frame.f_code.co_filename.split('\\')[-1]  # Just the filename
        code_line = current_frame.f_lineno
        function_name = current_frame.f_code.co_name

        # Skip internal logging and context manager frames
        if function_name not in ['log_read_context_warning', 'log_write_context_error', 'log', '<module>']:
            # Get the source code line for context
            try:
                import linecache
                source_line = linecache.getline(current_frame.f_code.co_filename, code_line).strip()
                if source_line:
                    stack_lines.append(f"  [{depth}] {code_filename}:{code_line} in {function_name}()\n      → {source_line[:100]}")
                else:
                    stack_lines.append(f"  [{depth}] {code_filename}:{code_line} in {function_name}()")
            except:
                stack_lines.append(f"  [{depth}] {code_filename}:{code_line} in {function_name}()")

        current_frame = current_frame.f_back
        depth += 1

    # Build the final warning message with stack trace
    stack_trace_str = "\n".join(reversed(stack_lines)) if stack_lines else "  [no stack frames available]"
    return (f"Reading {class_name} '{object_id}' via {method_name}() at {filename}:{line_number} "
            f"without read_context() protection - may read stale data.\n"
            f"Call sequence:\n{stack_trace_str}")


log_end_file(__file__)
