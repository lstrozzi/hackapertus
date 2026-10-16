from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import re
from typing import List, Optional, Tuple

# Canonical source-id token formats, shared by every place that creates, merges, or parses
# an ontology attribute's "source_id" field (a comma-joined list of these tokens):
#   doc://<document_id>[?p=<paragraph_id>]   - a document/chapter fragment (paragraph_id is a raw, local <p id> anchor)
#   kg://<entity_id>                         - a raw knowledge-graph Entity node
_DOC_SOURCE_RE = re.compile(r'^doc://(?P<document_id>[^?]+)(?:\?p=(?P<paragraph_id>.+))?$')
_KG_SOURCE_RE = re.compile(r'^kg://(?P<entity_id>.+)$')


def build_doc_source_id(document_id: str, paragraph_id: Optional[str] = None) -> str:
    """Build a canonical 'doc://<document_id>[?p=<paragraph_id>]' source-id token."""
    source_id: str = f"doc://{document_id}"

    if paragraph_id:
        source_id = f"{source_id}?p={paragraph_id}"

    return source_id


def build_kg_source_id(entity_id: str, qualifier: Optional[str] = None) -> str:
    """Build a canonical 'kg://<entity_id>[/<qualifier>]' source-id token (qualifier e.g. '<source_label>.<source_attribute>')."""
    source_id: str = f"kg://{entity_id}"

    if qualifier:
        source_id = f"{source_id}/{qualifier}"

    return source_id


def parse_doc_source_token(token: Optional[str]) -> Optional[Tuple[str, Optional[str]]]:
    """Parse a 'doc://<document_id>[?p=<paragraph_id>]' token into (document_id, paragraph_id), or None if it doesn't match."""
    parsed: Optional[Tuple[str, Optional[str]]] = None

    if token:
        match = _DOC_SOURCE_RE.match(token.strip())
        if match:
            parsed = (match.group("document_id"), match.group("paragraph_id"))

    return parsed


def parse_kg_source_token(token: Optional[str]) -> Optional[str]:
    """Parse a KG source token and return its base entity ID.

    Input format:
        ``kg://<entity_id>``
        ``kg://<entity_id>/<source_label>.<source_attribute>``
        Either form may also contain a query suffix such as ``?p=...``.

    The qualifier identifies the source attribute that produced the reference;
    it is intentionally removed because callers need the Neo4j entity ID.
    Returns the entity ID for a valid KG token, otherwise ``None``.
    """
    entity_id: Optional[str] = None

    if token:
        match = _KG_SOURCE_RE.match(token.strip())
        if match:
            entity_id = match.group("entity_id").split("?", 1)[0].split("/", 1)[0]

    return entity_id


def split_source_id_list(source_id_field: Optional[str]) -> List[str]:
    """Split a comma-joined 'source_id' attribute value into its individual, non-empty tokens."""
    return [token for token in (source_id_field or "").split(",") if token]


def join_source_id_list(tokens: List[str]) -> str:
    """Join individual source-id tokens back into a comma-joined 'source_id' attribute value."""
    return ",".join(tokens)


def source_id_list_has_token(source_id_field: Optional[str], token: str) -> bool:
    """Check whether 'token' is present, as an exact match, among the tokens of a comma-joined 'source_id' field."""
    return token in split_source_id_list(source_id_field)


def add_token_to_source_id_list(source_id_field: Optional[str], token: str) -> Tuple[str, bool]:
    """Add 'token' to a comma-joined 'source_id' field if not already present. Returns (new_field, changed)."""
    tokens = split_source_id_list(source_id_field)
    changed = token not in tokens

    if changed:
        tokens.append(token)

    return join_source_id_list(tokens), changed


def token_references_document(token: str, document_id: str) -> bool:
    """Check whether a source-id token references the given document_id (doc:// scheme only), normalizing chapter/paragraph fragments to their root document."""
    from src.interfaces.documents_interface import DocumentsInterface

    parsed = parse_doc_source_token(token)
    return parsed is not None and DocumentsInterface.get_document_id_from_fragment_id(parsed[0]) == document_id


def remove_document_from_source_id_list(source_id_field: Optional[str], document_id: str) -> Tuple[str, bool]:
    """Remove every token referencing 'document_id' from a comma-joined 'source_id' field. Returns (new_field, changed)."""
    tokens = split_source_id_list(source_id_field)
    remaining_tokens = [token for token in tokens if not token_references_document(token, document_id)]
    changed = len(remaining_tokens) != len(tokens)

    return join_source_id_list(remaining_tokens), changed


log_end_file(__file__)
