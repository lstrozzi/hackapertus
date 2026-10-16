"""Shared utilities for parsing and resolving dataset reference URIs."""

import re
from typing import Optional, Tuple

# Matches a doc:// path (without scheme/query string) into optional chapter/paragraph segments.
# Examples: "docid-106" -> Document, "docid-106-c0" -> Chapter, "docid-106-c0-p2" -> Paragraph
_DOC_URI_PATH_RE = re.compile(r'^(?P<docid>docid-[^-\s]+)(?:-(?P<chapter>c[^-\s]+))?(?:-(?P<paragraph>p[^-\s]+))?$')

# Matches doc://, kg://, and inno:// source URIs embedded in free-text attribute/answer values.
EMBEDDED_REFERENCE_URI_RE = re.compile(r'\b(?:doc|kg|inno)://[^\s<>"\')]+')


def parse_reference_uri(uri: str) -> Optional[Tuple[str, str]]:
    """Parse a canonical source URI into (node_label, node_id).

    Supported schemes: doc:// (Document/Chapter/Paragraph), kg:// and inno:// (Entity).
    Note: inno:// references Innovator knowledge-graph nodes, which in Neo4j carry the dual
    label :Entity:Innovator (imported from the EAM/Innovator ontology) rather than plain
    :Entity; both schemes resolve to the same "Entity" node_label here since node id (not
    label) is what identifies the target for relationship matching.
    Returns None for schemes with no corresponding Neo4j node (http://, https://, skill://, ds://)
    or for malformed/unrecognized URIs.

    Examples:
        doc://docid-106-c0-p2?id=text-5  -> ("Paragraph", "docid-106-c0-p2")
        doc://docid-106-c0                -> ("Chapter", "docid-106-c0")
        doc://docid-106                   -> ("Document", "docid-106")
        kg://charly                       -> ("Entity", "charly")
        kg://charly/AttributeGroup.field  -> ("Entity", "charly")            # qualifier suffix stripped
        inno://inno-42                    -> ("Entity", "inno-42")  # node is :Entity:Innovator
    """
    result: Optional[Tuple[str, str]] = None

    if uri and "://" in uri:
        scheme, _, rest = uri.partition("://")
        path = rest.split("?", 1)[0].strip("/")
        if path:
            if scheme == "doc":
                match = _DOC_URI_PATH_RE.match(path)
                if match:
                    if match.group("paragraph"):
                        result = ("Paragraph", path)
                    elif match.group("chapter"):
                        result = ("Chapter", path)
                    else:
                        result = ("Document", path)
            elif scheme in ("kg", "inno"):
                # strip an optional '/<source_label>.<source_attribute>' qualifier suffix (see build_kg_source_id())
                entity_id = path.split("/", 1)[0]
                result = ("Entity", entity_id)

    return result


def build_reference_uri(node_label: str, node_id: str, qualifier: Optional[str] = None) -> Optional[str]:
    """Build a canonical source URI with optional document query or knowledge-graph path metadata.

    Inverse of parse_reference_uri() for graph-backed node types. Returns None for
    labels with no canonical URI scheme.
    """
    result: Optional[str] = None

    if node_label and node_id:
        if node_label in ("Document", "Chapter", "Paragraph"):
            result = f"doc://{node_id}"
            if qualifier:
                result = f"{result}?p={qualifier}"
        elif node_label == "Entity":
            result = f"kg://{node_id}"
            if qualifier:
                result = f"{result}/{qualifier}"

    return result


def resolve_source_reference(fragment_id_or_uri: str) -> Optional[Tuple[str, str]]:
    """Resolve a bare fragment ID or a canonical source URI into (node_label, node_id).

    Handles both formats transparently:
      - bare fragment ID (e.g. "docid-19-c7-p3", as produced by generate_source_id())
      - canonical URI (doc://, kg://, inno://, as embedded in free-text attribute values)

    Chapter/Paragraph targets are always normalized to their root Document, since
    SOURCE relationships only ever target Document or Entity nodes.
    Returns None if the reference cannot be resolved (e.g. http://, unrecognized scheme).
    """
    from src.interfaces.documents_interface import DocumentsInterface

    result: Optional[Tuple[str, str]] = None

    if fragment_id_or_uri:
        if "://" in fragment_id_or_uri:
            parsed_reference = parse_reference_uri(fragment_id_or_uri)
            if parsed_reference:
                node_label, node_id = parsed_reference
                if node_label in ("Chapter", "Paragraph"):
                    document_id = DocumentsInterface.get_document_id_from_fragment_id(node_id)
                    if document_id:
                        result = ("Document", document_id)
                else:
                    result = (node_label, node_id)
        else:
            # Bare fragment ID (plain doc/chapter/paragraph form) — always normalize to its root Document.
            document_id = DocumentsInterface.get_document_id_from_fragment_id(fragment_id_or_uri)
            if document_id:
                result = ("Document", document_id)

    return result
