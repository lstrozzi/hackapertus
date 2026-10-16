"""Knowledge Workspace tab."""

from .kg_extraction import render as render_kg_extraction
from .kg_metamodel_inspection import render as render_kg_metamodel_inspection


def render() -> str:
    """Render knowledge-graph schema inspection and extraction tools."""
    return render_kg_metamodel_inspection() + render_kg_extraction()