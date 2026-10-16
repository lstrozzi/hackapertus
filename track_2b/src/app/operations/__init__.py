"""Independent tab implementations for the Operations wizard step."""

from .categories import render as render_categories
from .chat import render as render_chat
from .documents import render as render_documents
from .graph import render as render_graph
from .knowledge import render as render_knowledge

__all__ = ["render_operations_mockups"]


def render_operations_mockups() -> str:
    """Compose the five Workspace tabs from independent implementations."""
    stylesheets: list[str] = [
        "document_upload", "document_ingestion", "category_inference",
        "category_inspection", "categorization", "category_extension",
        "kg_metamodel_inspection", "kg_extraction", "kg_graph_view", "chat_qa",
    ]
    links: str = "".join(
        f'<link rel="stylesheet" href="/static/css/operations/{name}.css">'
        for name in stylesheets
    )
    pages: str = (
        '<div class="operation-workspace">'
        '<div class="operation-tabs" role="tablist" aria-label="Workspace pages">'
        '<button type="button" role="tab" data-operation-page="documents" aria-selected="true">Documents</button>'
        '<button type="button" role="tab" data-operation-page="categories" aria-selected="false" tabindex="-1">Categories</button>'
        '<button type="button" role="tab" data-operation-page="knowledge" aria-selected="false" tabindex="-1">Knowledge</button>'
        '<button type="button" role="tab" data-operation-page="graph" aria-selected="false" tabindex="-1">Graph</button>'
        '<button type="button" role="tab" data-operation-page="chat" aria-selected="false" tabindex="-1">Chat Q&amp;A</button>'
        '</div>'
        f'<section role="tabpanel" data-operation-panel="documents"><div class="operation-grid">{render_documents()}</div></section>'
        f'<section role="tabpanel" data-operation-panel="categories" hidden><div class="operation-grid">{render_categories()}</div></section>'
        '<section role="tabpanel" data-operation-panel="knowledge" hidden>'
        '<div class="operation-grid">'
        f'{render_knowledge()}'
        '</div></section>'
        f'<section role="tabpanel" data-operation-panel="graph" hidden>{render_graph()}</section>'
        '<section role="tabpanel" data-operation-panel="chat" hidden>'
        f'<div class="operation-grid">{render_chat()}</div>'
        '</section></div>'
    )

    return links + pages
