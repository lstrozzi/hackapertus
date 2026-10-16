from typing import Any, Dict


class BaseExtension:
    """Base class for type-specific rendering extensions."""

    def str(self, attributes: Dict[str, Any]) -> str:
        """Render a node's attributes as human-readable text."""
        return ""