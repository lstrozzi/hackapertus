"""Documents Workspace tab."""

from .document_ingestion import render as render_document_ingestion
from .document_upload import render as render_document_upload


def render() -> str:
    """Render document upload and ingestion tools."""
    return render_document_upload() + render_document_ingestion()