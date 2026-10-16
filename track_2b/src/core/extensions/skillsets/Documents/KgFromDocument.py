from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import traceback
import concurrent.futures
import time
from enum import Enum
from typing import Annotated, Dict, List
from bs4 import BeautifulSoup

from src.core.base.InputOutput import completion_percentage, execute_neo4j_query
from src.core.extensions.skillsets.KnowledgeGraph.KnowledgeGraphEditor import (
    community_create,
    entity_create_or_update,
    relationship_create_or_update,
    tokenize_entity_id,
)
from src.services.ontology_service import (
    ATTRIBUTE_ENTITY_DESCRIPTION,
    ATTRIBUTE_ENTITY_SOURCE_LABEL,
    ATTRIBUTE_ENTITY_TARGET_LABEL,
    get_ontology_service,
)
from src.services.documents_service import get_documents_service
from src.services.user_service import get_user_service
from src.interfaces.user_interface import UserInterface
from src.config.constants import MAX_PARALLELISM
from src.core.llm.manager import LlmManager
from src.utils.text_utils import generate_text_from_html, strip_html_element_attributes
from src.utils.ontology_label_validator import validate_ontology_label
from src.utils.log import log, DEBUG, INFO, WARNING, ERROR, FATAL
from src.core.extensions.skillsets.Documents.KgFromDocument_llm import (
    _call_llm_extract_entities,
    _call_llm_extract_entity_attribute,
    _call_llm_extract_relationships,
    _call_llm_extract_relationship_attribute,
    _call_llm_deduplicate_entities,
)

# Parallelization: sequential (1 worker) when a debugger is attached so breakpoints work,
# otherwise use the configured pool size from constants.
MAX_PARALLELIZATION: int = MAX_PARALLELISM  # 1 if IS_DEBUG else MAX_PARALLELISM

# Keys that identify the ontology node itself rather than user-defined attributes.
# 'name' is excluded because Phase 1 already handles name variant extraction.
_ONTOLOGY_META_KEYS = frozenset({"ontology_id", "id", "label", ATTRIBUTE_ENTITY_DESCRIPTION, "name"})


class KGExtractionPhase(Enum):
    """Controls how far the KG extraction runs. Useful for debugging."""
    ENTITIES = 1                 # Phase 1 only: discover entity instances
    ENTITY_ATTRIBUTES = 2        # Phases 1-2: entities + their attributes
    RELATIONSHIPS = 3            # Phases 1-3: entities, attributes, relationships
    RELATIONSHIP_ATTRIBUTES = 4  # Full run (default): all phases


# Chunking constants for parallel block-based processing
CHUNK_MAX_CHARS = 3000   # max characters per entity-extraction chunk


def _get_existing_community_entities(community_id: str, entity_label: str) -> List[dict]:
    """Return [{entity_id, name}] for existing entities of *entity_label* in the community."""
    query = f"MATCH (c:Community {{id: $community_id}})-[:AGGREGATES]->(e:Entity:{entity_label}) RETURN e.id AS entity_id, e.name AS name"
    parameters = {"community_id": community_id}
    result = execute_neo4j_query(query, parameters)
    if not result:
        return []
    return [{"entity_id": row["entity_id"], "name": row.get("name") or row["entity_id"]} for row in result if row.get("entity_id")]


def _get_existing_entity_attribute_values(community_id: str, entity_label: str, entity_ids: List[str], attr_name: str) -> Dict[str, List[str]]:
    """Return {entity_id: [value, ...]} for *attr_name* across entities of *entity_label* in the community."""
    if not entity_ids:
        return {}
    query = (
        f"MATCH (c:Community {{id: $community_id}})-[:AGGREGATES]->(e:Entity:{entity_label}) "
        f"WHERE e.id IN $entity_ids AND e.`{attr_name}` IS NOT NULL "
        f"RETURN e.id AS entity_id, e.`{attr_name}` AS attr_value"
    )
    parameters = {"community_id": community_id, "entity_ids": entity_ids}
    result = execute_neo4j_query(query, parameters)
    existing: Dict[str, List[str]] = {}
    if result:
        for row in result:
            eid = row.get("entity_id")
            val = row.get("attr_value")
            if eid and val is not None:
                if isinstance(val, list):
                    existing[eid] = [str(v) for v in val if v]
                else:
                    existing[eid] = [str(val)]
    return existing


# ---------------------------------------------------------------------------
# Helpers – ontology
# ---------------------------------------------------------------------------

def _get_ontology_attributes(ontology_node: dict) -> Dict[str, str]:
    """Return {attribute_name: description} for an entity/relationship ontology node dict.

    Skips all ontology metadata keys so only user-defined attributes remain.
    """
    result = {}
    for key, value in ontology_node.items():
        if key not in _ONTOLOGY_META_KEYS:
            result[key] = value or ""
    return result


def _get_relationship_source_target_labels(ontology_id: str, rel_label: str):
    """Return (source_labels_str, target_labels_str) for a RelationshipOntology node.

    Returns ("--", "--") when no constraint is defined.
    """
    # Use parameterized checks for both node id and label membership to avoid injection
    query = (
        "MATCH (o:Ontology {id: $ontology_id})-[:CONTAINS]->(r:RelationshipOntology) "
        "WHERE r.id = $rel_label OR $rel_label IN labels(r) "
        f"RETURN r.{ATTRIBUTE_ENTITY_SOURCE_LABEL} AS source_labels, "
        f"r.{ATTRIBUTE_ENTITY_TARGET_LABEL} AS target_labels"
    )
    parameters = {"ontology_id": ontology_id, "rel_label": rel_label}
    result = execute_neo4j_query(query, parameters)
    if result and len(result) > 0:
        src = result[0].get("source_labels") or "--"
        tgt = result[0].get("target_labels") or "--"
        return src, tgt
    return "--", "--"


# ---------------------------------------------------------------------------
# Document chunking
# ---------------------------------------------------------------------------

def _chunk_document(document_text: str) -> List[str]:
    """Split document HTML into character-limited chunks for entity extraction.

    Accumulates top-level elements into chunks until adding the next element would
    exceed CHUNK_MAX_CHARS.  Elements that alone exceed the limit are handled as:
    - Leaf (no element children): included as-is regardless of size.
    - Container (has element children): replaced by its first-level element children,
      which are then accumulated with the same character budget.

    Falls back to treating the whole text as a single chunk if parsing fails.
    """
    try:
        soup = BeautifulSoup(document_text, "html.parser")
        container = soup.body or soup
        top_elements = [el for el in container.children if hasattr(el, "name") and el.name]
    except Exception as e:
        log(WARNING, f"KGFromDocument: failed to parse document as HTML for chunking, falling back to single chunk: {e}")
        return [document_text]

    if not top_elements:
        return [document_text]

    # Flatten oversized container elements into their first-level children.
    flat: List[str] = []
    for el in top_elements:
        html_str = strip_html_element_attributes(str(el), ["id"])
        if len(html_str) <= CHUNK_MAX_CHARS:
            flat.append(html_str)
        else:
            children = [c for c in el.children if hasattr(c, "name") and c.name]
            if not children:
                flat.append(html_str)  # leaf — include as-is
            else:
                for child in children:
                    flat.append(strip_html_element_attributes(str(child), ["id"]))

    if not flat:
        return [document_text]

    # Accumulate flat elements into character-budget chunks.
    chunks: List[str] = []
    current_parts: List[str] = []
    current_size = 0
    for html_str in flat:
        el_size = len(html_str)
        if current_parts and current_size + el_size > CHUNK_MAX_CHARS:
            chunks.append("\n".join(current_parts))
            current_parts = [html_str]
            current_size = el_size
        else:
            current_parts.append(html_str)
            current_size += el_size
    if current_parts:
        chunks.append("\n".join(current_parts))

    log(DEBUG, f"KGFromDocument: document split into {len(chunks)} chunk(s) (sizes: {[len(c) for c in chunks]} chars)")
    return chunks


# ---------------------------------------------------------------------------
# Entity and relationship extraction
# ---------------------------------------------------------------------------

def _extract_entities(
    document_text: str,
    entity_ontologies: list,
    community_id: str,
    document_id: str,
    stop_after: KGExtractionPhase = KGExtractionPhase.RELATIONSHIP_ATTRIBUTES,
    document_context: str = "",
    ontology_description: str = "",
) -> Dict[str, dict]:
    """Extract and persist all entity types. Returns all_entities: entity_id -> {id, name, label}."""
    all_entities: Dict[str, dict] = {}
    chunks = _chunk_document(document_text)

    # Distribute progress 0.50-0.80 across entity type processing
    entity_ontologies_count = len(entity_ontologies)
    progress_per_entity = 0.60 / max(entity_ontologies_count, 1)  # 60% / number of entity types
    entity_index = 0

    for entity_ontology in entity_ontologies:
        entity_label = entity_ontology.get("label") or entity_ontology.get("id")
        validate_ontology_label(entity_label)
        entity_description = entity_ontology.get(ATTRIBUTE_ENTITY_DESCRIPTION) or ""
        entity_attributes = _get_ontology_attributes(entity_ontology)

        log(INFO, f"KGFromDocument: extracting entities of type '{entity_label}' (ontology attributes: {list(entity_attributes.keys())})")

        existing_entities = _get_existing_community_entities(community_id, entity_label)
        log(INFO, f"KGFromDocument: found {len(existing_entities)} existing '{entity_label}' entities in community '{community_id}'")

        raw_entities = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLELIZATION) as executor:
            futures = {executor.submit(_call_llm_extract_entities, f"<TEXT_BLOCK_IN_FOCUS>{chunk}</TEXT_BLOCK_IN_FOCUS>", entity_label, entity_description, document_context, ontology_description, existing_entities): i for i, chunk in enumerate(chunks)}
            for future in concurrent.futures.as_completed(futures):
                chunk_index = futures[future]
                try:
                    chunk_entities = future.result()
                    log(DEBUG, f"KGFromDocument: chunk {chunk_index} returned {len(chunk_entities)} entities for type '{entity_label}': {[e.get('entity_id') for e in chunk_entities]}")
                    raw_entities.extend(chunk_entities)
                except Exception as e:
                    log(ERROR, f"KGFromDocument: chunk {chunk_index} failed to extract '{entity_label}' entities: {e}")
        if not raw_entities:
            log(INFO, f"KGFromDocument: no '{entity_label}' entities found in document")
            continue

        entity_name_variants: Dict[str, List[str]] = {}
        for ent in raw_entities:
            eid = tokenize_entity_id(ent.get("entity_id") or "")
            if not eid:
                continue
            name = str(ent.get("name") or eid).strip()
            if eid not in entity_name_variants:
                entity_name_variants[eid] = []
            if name and name not in entity_name_variants[eid]:
                entity_name_variants[eid].append(name)
        unique_entities: List[dict] = [
            {"entity_id": eid, "name": variants[0], "names": variants}
            for eid, variants in entity_name_variants.items()
        ]
        log(DEBUG, f"KGFromDocument: name variants per entity: { {eid: variants for eid, variants in entity_name_variants.items()} }")

        if not unique_entities:
            log(INFO, f"KGFromDocument: no unique entities after deduplication for type '{entity_label}'")
            continue

        log(INFO, f"KGFromDocument: {len(unique_entities)} unique '{entity_label}' entities to persist: {[e['entity_id'] for e in unique_entities]}")

        # Ask LLM to identify duplicate entities (e.g., aliases or alternative names) and wrong-type entities
        dedup_map, entities_to_remove = _call_llm_deduplicate_entities(entity_label, unique_entities, document_context)

        # Remove wrong-type entities flagged by the LLM
        if entities_to_remove:
            remove_set = set(entities_to_remove)
            unique_entities = [e for e in unique_entities if e["entity_id"] not in remove_set]
            log(INFO, f"KGFromDocument: removed {len(remove_set)} wrong-type '{entity_label}' entities: {entities_to_remove}")

        # Merge duplicate entities
        merged_entities = {}
        for ent in unique_entities:
            eid = ent["entity_id"]
            canonical_eid = dedup_map.get(eid, eid)
            if canonical_eid not in merged_entities:
                merged_entities[canonical_eid] = {"entity_id": canonical_eid, "name": ent["name"], "names": list(ent["names"])}
            else:
                for n in ent["names"]:
                    if n not in merged_entities[canonical_eid]["names"]:
                        merged_entities[canonical_eid]["names"].append(n)

        unique_entities = list(merged_entities.values())
        log(INFO, f"KGFromDocument: after LLM deduplication, {len(unique_entities)} '{entity_label}' entities remain: {[e['entity_id'] for e in unique_entities]}")

        entity_ids = [e["entity_id"] for e in unique_entities]
        entity_names = [e["name"] for e in unique_entities]

        # Phase 1 persistence: create/update the entity node with its name variants.
        for ent in unique_entities:
            eid = ent["entity_id"]
            log(DEBUG, f"KGFromDocument: persisting entity '{eid}' (label='{entity_label}', community_id='{community_id}')")
            persist_ok = entity_create_or_update(
                entity_id=eid,
                community_id=community_id,
                label=entity_label,
                attributes={"name": ent["names"]},
                document_or_chapter_id=document_id,
                paragraph_id=None,
            )
            if persist_ok:
                all_entities[eid] = {"id": eid, "name": ent["name"], "label": entity_label}
                log(DEBUG, f"KGFromDocument: saved entity '{eid}' ({entity_label})")
            else:
                log(WARNING, f"KGFromDocument: entity_create_or_update returned False for entity '{eid}' ({entity_label})")

        # Phase 2: extract all attributes across chunks in a single parallel sweep.
        # Chunk-based (same as Phase 1): the LLM emits source_id pointing to the specific
        # HTML element id within the chunk, preserving provenance without per-element calls.
        if stop_after.value >= KGExtractionPhase.ENTITY_ATTRIBUTES.value and entity_attributes:
            log(INFO, f"KGFromDocument: extracting {len(entity_attributes)} attribute(s) for '{entity_label}' across {len(chunks)} chunk(s) — {len(entity_attributes) * len(chunks)} parallel LLM call(s)")

            # Pre-fetch existing attribute values from Neo4j once per attribute (LLM context only).
            existing_attr_values_by_attr = {
                attr_name: _get_existing_entity_attribute_values(community_id, entity_label, entity_ids, attr_name)
                for attr_name in entity_attributes
            }

            entity_ids_set = set(entity_ids)
            persist_count = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLELIZATION) as executor:
                future_to_key: Dict = {}
                for attr_name, attr_description in entity_attributes.items():
                    existing = existing_attr_values_by_attr[attr_name]
                    for chunk_idx, chunk in enumerate(chunks):
                        f = executor.submit(_call_llm_extract_entity_attribute, f"<TEXT_BLOCK_IN_FOCUS>{chunk}</TEXT_BLOCK_IN_FOCUS>", entity_label, entity_ids, entity_names, attr_name, attr_description, document_context, existing)
                        future_to_key[f] = (attr_name, chunk_idx)

                for future in concurrent.futures.as_completed(future_to_key):
                    attr_name, chunk_idx = future_to_key[future]
                    try:
                        results = future.result()
                        for item in results:
                            eid = tokenize_entity_id(item.get("entity_id") or "")
                            value = str(item.get("value") or "").strip()
                            if eid in entity_ids_set and value:
                                item_source_id = str(item.get("source_id") or "").strip() or None
                                log(DEBUG, f"KGFromDocument: chunk {chunk_idx}, element '{item_source_id}': '{attr_name}'='{value}' for entity '{eid}'")
                                entity_create_or_update(entity_id=eid, community_id=community_id, label=entity_label, attributes={attr_name: value}, document_or_chapter_id=document_id, paragraph_id=item_source_id)
                                persist_count += 1
                    except Exception as e:
                        log(ERROR, f"KGFromDocument: chunk {chunk_idx} failed for attribute '{attr_name}': {e}")

            log(INFO, f"KGFromDocument: attribute extraction complete for '{entity_label}' — {persist_count} attribute value(s) persisted")

        # Update progress: each entity type gets progress_per_entity allocated
        completion_percentage[0] = 0.20 + (entity_index + 1) * progress_per_entity
        entity_index += 1

    log(INFO, f"KGFromDocument: entity extraction complete, {len(all_entities)} entities persisted: {list(all_entities.keys())}")
    return all_entities


def _extract_relationships(
    document_text: str,
    relationship_ontologies: list,
    ontology_id: str,
    all_entities: Dict[str, dict],
    document_id: str,
    stop_after: KGExtractionPhase = KGExtractionPhase.RELATIONSHIP_ATTRIBUTES,
    document_context: str = "",
    ontology_description: str = "",
) -> None:
    """Extract and persist all relationship types."""
    all_entity_ids = list(all_entities.keys())
    # Use chunks (like entity Phase 1): multiple elements per LLM call reduces total call count.
    # The LLM still emits a per-element source_id from the HTML element ids inside the chunk,
    # so provenance is preserved without needing one call per element.
    chunks = _chunk_document(document_text)

    # Distribute progress 0.80-0.95 across relationship type processing
    relationship_ontologies_count = len(relationship_ontologies)
    progress_per_relationship = 0.15 / max(relationship_ontologies_count, 1)  # 15% / number of relationship types
    relationship_index = 0

    for rel_ontology in relationship_ontologies:
        rel_label = rel_ontology.get("label") or rel_ontology.get("id")
        validate_ontology_label(rel_label)
        rel_description = rel_ontology.get(ATTRIBUTE_ENTITY_DESCRIPTION) or ""
        rel_attributes = _get_ontology_attributes(rel_ontology)

        src_labels_raw, tgt_labels_raw = _get_relationship_source_target_labels(ontology_id, rel_label)
        source_label_str = src_labels_raw if src_labels_raw != "--" else "any"
        target_label_str = tgt_labels_raw if tgt_labels_raw != "--" else "any"

        log(INFO, f"KGFromDocument: extracting relationships of type '{rel_label}' ({source_label_str} -> {target_label_str}) across {len(chunks)} chunk(s)")

        raw_rels = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLELIZATION) as executor:
            futures = {executor.submit(_call_llm_extract_relationships, f"<TEXT_BLOCK_IN_FOCUS>{chunk}</TEXT_BLOCK_IN_FOCUS>", rel_label, rel_description, source_label_str, target_label_str, all_entity_ids, document_context, ontology_description): i for i, chunk in enumerate(chunks)}
            for future in concurrent.futures.as_completed(futures):
                chunk_index = futures[future]
                try:
                    chunk_rels = future.result()
                    log(DEBUG, f"KGFromDocument: chunk {chunk_index} returned {len(chunk_rels)} relationships for type '{rel_label}'")
                    raw_rels.extend(chunk_rels)
                except Exception as e:
                    log(ERROR, f"KGFromDocument: chunk {chunk_index} failed to extract '{rel_label}' relationships: {e}")

        if not raw_rels:
            log(INFO, f"KGFromDocument: no '{rel_label}' relationships found")
            continue

        seen_pairs: set = set()
        valid_rels: List[dict] = []
        for rel in raw_rels:
            src = tokenize_entity_id(rel.get("source_entity_id") or "")
            tgt = tokenize_entity_id(rel.get("target_entity_id") or "")
            if src in all_entities and tgt in all_entities:
                pair = (src, tgt)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    item_source_id = str(rel.get("source_id") or "").strip() or None
                    valid_rels.append({"source_entity_id": src, "target_entity_id": tgt, "source_id": item_source_id})
            else:
                log(WARNING, f"KGFromDocument: skipping '{rel_label}' relationship with unknown entity ids src='{src}', tgt='{tgt}'")

        if not valid_rels:
            log(INFO, f"KGFromDocument: no valid '{rel_label}' relationships after entity validation")
            continue

        log(INFO, f"KGFromDocument: {len(valid_rels)} valid '{rel_label}' relationships to persist")

        rel_attribute_values: Dict[tuple, Dict[str, str]] = {
            (r["source_entity_id"], r["target_entity_id"]): {}
            for r in valid_rels
        }

        if stop_after.value >= KGExtractionPhase.RELATIONSHIP_ATTRIBUTES.value and rel_attributes:
            for attr_name, attr_description in rel_attributes.items():
                log(DEBUG, f"KGFromDocument: extracting attribute '{attr_name}' for '{rel_label}' relationships")
                attr_results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLELIZATION) as executor:
                    futures = {executor.submit(_call_llm_extract_relationship_attribute, f"<TEXT_BLOCK_IN_FOCUS>{chunk}</TEXT_BLOCK_IN_FOCUS>", rel_label, valid_rels, attr_name, attr_description, document_context, ontology_description): i for i, chunk in enumerate(chunks)}
                    for future in concurrent.futures.as_completed(futures):
                        chunk_index = futures[future]
                        try:
                            chunk_attr_results = future.result()
                            log(DEBUG, f"KGFromDocument: chunk {chunk_index} returned {len(chunk_attr_results)} attribute values for '{attr_name}' of '{rel_label}' relationships")
                            attr_results.extend(chunk_attr_results)
                        except Exception as e:
                            log(ERROR, f"KGFromDocument: chunk {chunk_index} failed to extract attribute '{attr_name}' for '{rel_label}': {e}")
                for item in attr_results:
                    src = tokenize_entity_id(item.get("source_entity_id") or "")
                    tgt = tokenize_entity_id(item.get("target_entity_id") or "")
                    value = item.get("value")
                    key = (src, tgt)
                    if key in rel_attribute_values and value:
                        rel_attribute_values[key][attr_name] = value

        for rel in valid_rels:
            src = rel["source_entity_id"]
            tgt = rel["target_entity_id"]
            item_source_id = rel.get("source_id")
            attrs = rel_attribute_values.get((src, tgt)) or None
            relationship_create_or_update(
                ontology_id=ontology_id,
                entity1_id=src,
                entity2_id=tgt,
                label=rel_label,
                attributes=attrs,
                document_or_chapter_id=document_id,
                paragraph_id=item_source_id,
            )
            log(DEBUG, f"KGFromDocument: element '{item_source_id}': saved relationship '{rel_label}': '{src}' -> '{tgt}' with {len(attrs) if attrs else 0} attribute(s)")

        # Update progress: each relationship type gets progress_per_relationship allocated
        completion_percentage[0] = 0.80 + (relationship_index + 1) * progress_per_relationship
        relationship_index += 1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def KgFromDocument(
    ontology_id: Annotated[str, "Identifier of the ontology to use for KG construction. The ontology defines :EntityOntology and :RelationshipOntology nodes with their attributes."],
    document_id: Annotated[str, "Identifier of the document to extract the knowledge graph from. The document's enhanced_text (or fallback text) is processed."],
    community_id: Annotated[str, "Optional identifier of the community (entity group) in which the extracted entities are stored. If omitted, a community is auto-generated from ontology_id and document_id."] = None,
    restrict_to_ontology_entity_ids: Annotated[list, "Optional debug parameter: if provided, only extract entities of these types (by label or id). For example: ['animal', 'person']"] = None,
    stop_after: KGExtractionPhase = KGExtractionPhase.RELATIONSHIP_ATTRIBUTES,
    goal=None,
) -> dict:
    """
    Extracts a Knowledge Graph from a document according to the given ontology.

    Algorithm:
    1. Load the document's enhanced text.
    2. Load entity types and relationship types from the ontology.
    3. Ensure the target community exists in Neo4j.
    4. For each EntityOntology:
       a. Phase 1  – LLM (per entity type × chunk): discover entity instances (id + name).
       b. Phase 1b – LLM (per entity type): deduplicate aliases and remove wrong-type entities.
       c. Persist deduplicated entities to Neo4j via entity_create_or_update.
       d. Phase 2  – LLM (per entity type × attribute × chunk): extract attribute values;
          persist inline; LLM emits source_id pointing to the HTML element id within the chunk.
    5. For each RelationshipOntology:
       a. Phase 3  – LLM (per relationship type × chunk): discover relationship instances;
          LLM emits source_id; Python deduplicates and validates entity ids.
       b. Phase 4  – LLM (per relationship type × attribute × chunk): extract attribute values.
       c. Persist relationships to Neo4j via relationship_create_or_update.
    """
    result = None

    try:
        # Automated background runs (DocGraphBuilderJob) have no authenticated identity: adopt the
        # document owner's identity so the ACL-gated document/ontology reads and entity writes pass.
        user_service: UserInterface = get_user_service()
        assumed_owner_identity: bool = user_service.assume_identity_from_background_job(document_id)
        if assumed_owner_identity:
            log(INFO, f"KgFromDocument: assumed effective identity of document owner '{user_service.get_user_id()}'")

        # 1. Load document text
        document = get_documents_service().get_document(document_id, include_text=True, include_summary=True)
        if document is None:
            return {"error": 1, "error_message": f"Document '{document_id}' not found."}

        document_text = document.get("enhanced_text") or document.get("text") or ""
        if not document_text:
            return {"error": 1, "error_message": f"Document '{document_id}' has no text content."}
        
        completion_percentage[0] = 0.0
        # Link the document to the ontology it was processed with
        get_documents_service().link_document_to_ontology(document_id, ontology_id)

        # Build document_context: use plain text directly when small enough, otherwise fall back to the summary.
        # This context is injected into every LLM call so the model can disambiguate entity types across chunks.
        doc_plain_text = generate_text_from_html(document.get("text") or "")
        max_loaded_document_length = LlmManager.instance().get_max_loaded_document_length()
        document_context = doc_plain_text if len(doc_plain_text) <= max_loaded_document_length else (document.get("summary") or "")

        log(INFO, f"KGFromDocument: processing document '{document_id}' ({len(document_text)} chars) with ontology '{ontology_id}'")
        completion_percentage[0] = 0.10

        # 2. Load ontology definitions
        ontology_svc = get_ontology_service()
        entity_ontologies = ontology_svc.get_ontology_entities(ontology_id)
        relationship_ontologies = ontology_svc.get_ontology_relationships(ontology_id)

        if not entity_ontologies and not relationship_ontologies:
            return {"error": 1, "error_message": f"Ontology '{ontology_id}' not found or contains no entity/relationship definitions."}

        log(INFO, f"KGFromDocument: ontology has {len(entity_ontologies)} entity type(s) and {len(relationship_ontologies)} relationship type(s)")
        completion_percentage[0] = 0.15

        # Build the full ontology description once — injected into every LLM call so the model
        # understands all entity/relationship types and avoids cross-type misclassification.
        ontology_description = ontology_svc._get_ontology_as_string(ontology_id, None, include_attributes=True)

        # Filter entity ontologies if restrict_to_ontology_entity_ids is specified
        if restrict_to_ontology_entity_ids:
            restrict_set = set(label.lower() for label in restrict_to_ontology_entity_ids)
            entity_ontologies = [eo for eo in entity_ontologies if (eo.get("label") or eo.get("id")).lower() in restrict_set]
            log(INFO, f"KGFromDocument: after filtering, {len(entity_ontologies)} entity type(s) remain: {[eo.get('label') or eo.get('id') for eo in entity_ontologies]}")

        # 3. Ensure the community exists
        if not community_id:
            existing_communities = ontology_svc.get_ontology_communities(ontology_id)
            if existing_communities:
                community_id = next((cid for cid in existing_communities if cid.upper() == "DEFAULT"), existing_communities[0])
                log(INFO, f"KGFromDocument: no community_id provided, resolved to '{community_id}' from ontology '{ontology_id}'")
            else:
                community_id = tokenize_entity_id(f"{ontology_id}_{document_id}")
                log(INFO, f"KGFromDocument: no communities found for ontology '{ontology_id}', using auto-generated community_id '{community_id}'")

        if not execute_neo4j_query("MATCH (c:Community {id: $community_id}) RETURN c.id AS id", {"community_id": community_id}):
            community_create(
                id=community_id,
                name=community_id,
                description=f"KG extracted from document '{document_id}' using ontology '{ontology_id}'",
                ontology_id=ontology_id,
            )
            log(INFO, f"KGFromDocument: created community '{community_id}'")
        completion_percentage[0] = 0.20

        # 4. Extract and persist entities
        all_entities = _extract_entities(document_text, entity_ontologies, community_id, document_id, stop_after, document_context, ontology_description)
        completion_percentage[0] = 0.80

        # 5. Extract and persist relationships
        if stop_after.value >= KGExtractionPhase.RELATIONSHIPS.value:
            _extract_relationships(document_text, relationship_ontologies, ontology_id, all_entities, document_id, stop_after, document_context, ontology_description)
        completion_percentage[0] = 0.95

        log(INFO, f"KGFromDocument: completed. {len(all_entities)} entities in community '{community_id}'")

        result = {
            "success": True,
            "community_id": community_id,
            "entities_count": len(all_entities),
            "entities": [
                {"id": v["id"], "label": v["label"], "name": v["name"]}
                for v in all_entities.values()
            ],
            "repeat_contextualizer": True,
        }
        completion_percentage[0] = 1.0

    except Exception as e:
        stack_trace = traceback.format_exc()
        log(ERROR, f"KGFromDocument: {stack_trace}")
        result = {"error": 1, "error_message": str(e)}

    return result


def _KgFromDocument(
    ontology_id: Annotated[str, "Identifier of the ontology to use for KG construction. The ontology defines :EntityOntology and :RelationshipOntology nodes with their attributes."],
    document_id: Annotated[str, "Identifier of the document to extract the knowledge graph from. The document's enhanced_text (or fallback text) is processed."],
    community_id: Annotated[str, "Optional identifier of the community (entity group) in which the extracted entities are stored. If omitted, a community is auto-generated from ontology_id and document_id."] = None,
    restrict_to_ontology_entity_ids: Annotated[list, "Optional debug parameter: if provided, only extract entities of these types (by label or id). For example: ['animal', 'person']"] = None,
    stop_after: KGExtractionPhase = KGExtractionPhase.RELATIONSHIP_ATTRIBUTES,
    goal=None,
) -> dict:
    """Mock version of KgFromDocument that simulates the workflow with 3-second delays between phases.
    
    Simulates:
    - Phase 1: Entity extraction (0% -> 25%)
    - Phase 2: Entity attributes (25% -> 50%)
    - Phase 3: Relationship extraction (50% -> 75%)
    - Phase 4: Relationship attributes (75% -> 100%)
    """
    result = None
    
    try:
        log(INFO, f"KgFromDocument_Mock: starting mock KG extraction for document '{document_id}' with ontology '{ontology_id}'")
        completion_percentage[0] = 0.0
        
        # Link the document to the ontology it was processed with
        get_documents_service().link_document_to_ontology(document_id, ontology_id)
        
        # Phase 1: Entity extraction
        log(INFO, f"KgFromDocument_Mock: Phase 1 - Entity extraction")
        time.sleep(3)
        completion_percentage[0] = 0.25
        
        # Phase 2: Entity attributes
        if stop_after.value >= KGExtractionPhase.ENTITY_ATTRIBUTES.value:
            log(INFO, f"KgFromDocument_Mock: Phase 2 - Entity attributes")
            time.sleep(3)
            completion_percentage[0] = 0.50
        
        # Phase 3: Relationship extraction
        if stop_after.value >= KGExtractionPhase.RELATIONSHIPS.value:
            log(INFO, f"KgFromDocument_Mock: Phase 3 - Relationship extraction")
            time.sleep(3)
            completion_percentage[0] = 0.75
        
        # Phase 4: Relationship attributes
        if stop_after.value >= KGExtractionPhase.RELATIONSHIP_ATTRIBUTES.value:
            log(INFO, f"KgFromDocument_Mock: Phase 4 - Relationship attributes")
            time.sleep(3)
            completion_percentage[0] = 1.0
        
        # Generate mock community ID if not provided
        if not community_id:
            community_id = tokenize_entity_id(f"{ontology_id}_{document_id}")

        log(INFO, f"KgFromDocument_Mock: completed. Mock entities in community '{community_id}'")        
        result = {
            "success": True,
            "community_id": community_id,
            "entities_count": 5,  # Mock count
            "entities": [
                {"id": "entity-1", "label": "Type1", "name": "Mock Entity 1"},
                {"id": "entity-2", "label": "Type2", "name": "Mock Entity 2"},
                {"id": "entity-3", "label": "Type1", "name": "Mock Entity 3"},
                {"id": "entity-4", "label": "Type3", "name": "Mock Entity 4"},
                {"id": "entity-5", "label": "Type2", "name": "Mock Entity 5"},
            ],
            "repeat_contextualizer": True,
        }
        
    except Exception as e:
        stack_trace = traceback.format_exc()
        log(ERROR, f"KgFromDocument_Mock: {stack_trace}")
        result = {"error": 1, "error_message": str(e)}
    
    return result


log_end_file(__file__)


if __name__ == "__main__":
    from src.utils.unit_test import init_unit_test
    from src.utils.log import log_init, log_exit

    init_unit_test()
    log_init()

    try:
        document_name = "AWS Certified Solutions Architect - Associate certificate.pdf"
        doc_rows = execute_neo4j_query("MATCH (d:Document {name: $name}) RETURN d.id AS id", {"name": document_name})
        
        if not doc_rows:
            log(FATAL, f"Unit test for {__file__} failed: document '{document_name}' not found in Neo4j")
        else:
            doc_id = doc_rows[0]["id"]
            log(INFO, f"Found document id '{doc_id}' for '{document_name}'")
            log(INFO, "=================================================================")
            log(INFO, f"   EXTRACTING ENTITIES ONLY (stop_after=KGExtractionPhase.ENTITIES)   ")
            test_result = KgFromDocument(
                ontology_id="FictionalStories",
                document_id=doc_id,
                restrict_to_ontology_entity_ids=None, # ["Person"],
                # stop_after=KGExtractionPhase.ENTITIES,
            )
            log(INFO, f"Test result: {test_result}")

            # log(INFO, "=================================================================")
            # log(INFO, "   EXTRACTING ENTITY ATTRIBUTES (stop_after=KGExtractionPhase.ENTITY_ATTRIBUTES)   ")
            # test_result_attrs = KgFromDocument(
            #     ontology_id="FictionalStories",
            #     document_id=doc_id,
            #     restrict_to_ontology_entity_ids=None, # ["Person"],
            #     stop_after=KGExtractionPhase.ENTITY_ATTRIBUTES,
            # )
            # log(INFO, f"Test result (attributes): {test_result_attrs}")

            # log(INFO, "=================================================================")
            # log(INFO, "   EXTRACTING RELATIONSHIPS (stop_after=KGExtractionPhase.RELATIONSHIP_ATTRIBUTES)   ")
            # test_result_rels = KgFromDocument(
            #     ontology_id="FictionalStories",
            #     document_id=doc_id,
            #     restrict_to_ontology_entity_ids=None,
            #     stop_after=KGExtractionPhase.RELATIONSHIP_ATTRIBUTES,
            # )
            # log(INFO, f"Test result (relationships): {test_result_rels}")

    except Exception as e:
        log(FATAL, f"Unit test for {__file__} failed with exception: {e}. Traceback:\n{traceback.format_exc()}")

    finally:
        log_exit()
