"""Chat Workspace tab."""

from .chat_qa import render as render_chat_qa


def render() -> str:
    """Render the retrieval-grounded chat interface."""
    return render_chat_qa()