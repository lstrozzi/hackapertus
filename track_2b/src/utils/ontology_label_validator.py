import re
from typing import Optional


MAX_ONTOLOGY_LABEL_LENGTH: int = 64
_VALID_ONTOLOGY_LABEL_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_]+$")


def is_valid_ontology_label(label: Optional[str]) -> bool:
    is_valid: bool = (
        isinstance(label, str)
        and bool(label)
        and len(label) <= MAX_ONTOLOGY_LABEL_LENGTH
        and _VALID_ONTOLOGY_LABEL_PATTERN.fullmatch(label) is not None
    )

    return is_valid


def validate_ontology_label(label: Optional[str]) -> None:
    if not is_valid_ontology_label(label):
        raise ValueError("Ontology labels must contain 1 to 64 ASCII letters, numbers, or underscores.")