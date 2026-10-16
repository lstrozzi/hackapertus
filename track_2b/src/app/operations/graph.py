"""Graph Workspace tab."""

from .kg_graph_view import render as render_kg_graph_view


def render() -> str:
    """Render the knowledge graph viewer."""
    return render_kg_graph_view()