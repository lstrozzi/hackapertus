"""Categories Workspace tab."""

from .category_extension import render as render_category_extension
from .category_inference import render as render_category_inference
from .category_inspection import render as render_category_inspection
from .categorization import render as render_categorization


def render() -> str:
    """Render category proposal, inspection, assignment, and ontology-extension tools."""
    return (
        render_category_inference()
        + render_category_inspection()
        + render_categorization()
        + render_category_extension()
    )