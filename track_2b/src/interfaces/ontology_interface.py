from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Set, Tuple, TYPE_CHECKING
from src.config.constants import MAX_RAG_MESSAGE_LENGTH_DEFAULT

if TYPE_CHECKING:
    from src.core.thalaplan.Plan import Plan


class OntologyInterface(ABC):
    """Abstract interface for ontology repository / service operations."""

    @abstractmethod
    def has_ontologies(self, document_ids: Optional[List[str]] = None, plan: Optional['Plan'] = None) -> bool:
        """Return True if any ontologies are available in the backing store.

        This is a minimal method used by agents to decide whether ontology
        related processing should run. Implementations may query a database
        or other storage to determine availability.
        
        Args:
            document_ids: Optional list of document IDs to check if any follow an ontology
            plan: Optional Plan object to check if plan_ontology_id has Entity nodes
            
        Returns:
            True if ontologies are available based on the provided parameters
        """
        raise NotImplementedError()

    @abstractmethod
    def is_not_reserved_and_unconsolidated_attribute(self, attr_name: str) -> bool:
        """Check if an attribute name is not reserved and not consolidated.

        Args:
            attr_name: The attribute name to check

        Returns:
            True if the attribute is not reserved and not consolidated, False otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def _is_not_reserved_and_consolidated_attribute(self, attr_name: str) -> bool:
        """Check if an attribute name is not reserved and is consolidated.

        Private method - not used externally in the codebase.

        Args:
            attr_name: The attribute name to check

        Returns:
            True if the attribute is not reserved and is consolidated, False otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def get_entity_raw_attributes(self, element_id: str) -> Dict[str, str]:
        """Return {attr_name: raw_json_value} for non-reserved, unconsolidated attributes of the Entity with the given elementId."""
        raise NotImplementedError()

    @abstractmethod
    def nodes_exist(self, node_references: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Optional[bool]]:
        """Check which (node_label, node_id) references correspond to existing Neo4j nodes."""
        raise NotImplementedError()

    @abstractmethod
    def get_primary_key_attribute_names(self, label: str) -> List[str]:
        """Return the ordered attribute names forming the primary key of the EntityOntology with the given label, or an empty list if none is defined."""
        raise NotImplementedError()

    @abstractmethod
    def find_conflicting_entity_by_primary_key(self, label: str, entity_id: str, attributes: Dict[str, Any]) -> Optional[str]:
        """Return the id of another Entity with the same label whose primary-key attribute values collide with the given ones, or None."""
        raise NotImplementedError()

    @abstractmethod
    def remove_obsolete_references_from_entity_attributes(self, element_id: str, obsolete_references: Set[Tuple[str, str]]) -> bool:
        """Prune source_id entries and embedded reference URIs pointing to no-longer-existing nodes from an entity's attributes."""
        raise NotImplementedError()

    @abstractmethod
    def create_source_relationship(self, entity_element_id: str, target_label: str, target_id: str) -> bool:
        """Create the SOURCE relationship from an Entity to a target node, if not already present."""
        raise NotImplementedError()

    @abstractmethod
    def _get_ontology_as_string(self, ontology_id: str, ontology_description: str, include_attributes: bool = True) -> str:
        """Get ontology as formatted string"""
        pass

    @abstractmethod
    def get_entities_as_string(self, community_id: str, entity_ids: list = None, include_relationships: bool = False, omit_headers: bool = False) -> str:
        """Returns a string representation of entities filtered by community_id and/or entity_ids.

        Args:
            omit_headers: When True, suppresses the '## ...' section header lines so the
                          returned string contains only entity content (useful for downstream
                          consumers such as rerankers that should not receive metadata headers).
        """
        pass

    @abstractmethod
    def get_entities(self, community_id: str, label_filter_regex: str = None, query_where_clause: str = "") -> list:
        """Get entities filtered by community_id and optional label regex"""
        pass

    @abstractmethod
    def get_entities_by_ids(self, entity_ids: List[str], include_relationships: bool = False, effective_user_id: Optional[str] = None) -> Dict[str, dict]:
        """Return a dict mapping entity_id -> entity dict for the given entity_ids.
        
        Re-uses the same consolidated-attribute logic as get_entities() and, if requested,
        fetches incoming/outgoing relationships in one additional query and attaches them
        as lists of dicts with keys: 'label' (node label), 'id' and 'name' (name_consolidated).
        
        Args:
            entity_ids: List of entity IDs to fetch
            include_relationships: Whether to include incoming/outgoing relationships
            effective_user_id: Neo4j User.id of the requesting user. When provided, only entities
                      whose parent Ontology (via DEFINES->Community->AGGREGATES path) the
                      effective user holds GRANT:READ on are returned. When None, ACL is skipped
                      (system/pipeline bypass).
            
        Returns:
            Dictionary mapping entity_id to entity dict
        """
        pass

    @abstractmethod
    def expand_entities_with_neighbors(
        self,
        entity_weights: Dict[str, float],
        min_rank: Optional[float] = None,
        neighbor_weight_factor: Optional[float] = None,
        effective_user_id: Optional[str] = None
    ) -> Dict[str, float]:
        """Broaden a kg entity_id -> weight mapping with adjacent (neighbour) entities.

        For every entity whose weight is strictly greater than min_rank, fetches its incoming/outgoing
        Entity neighbours and adds any not already present in entity_weights, using
        neighbor_weight_factor * source entity weight as the neighbour's weight. Existing entries are
        never overwritten; when a neighbour is reachable from several sources, the highest computed weight is kept.

        Args:
            entity_weights: Mapping of entity_id -> weight (e.g. from a KG RAG search) to expand.
            min_rank: Minimum weight for an entity to have its neighbours added (defaults to KG_NEIGHBOR_EXPANSION_MIN_RANK).
            neighbor_weight_factor: Multiplier applied to the source weight for added neighbours (defaults to KG_NEIGHBOR_WEIGHT_FACTOR).
            effective_user_id: Neo4j User.id used for ACL filtering; see get_entities_by_ids.

        Returns:
            A new dict containing the original entries plus any added neighbour entries.
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_relationships_as_string(self, community_id: str) -> str:
        """Get relationships for a community as formatted string.
        
        Private method - only used internally within Ontology.py.
        
        Args:
            community_id: The community identifier
            
        Returns:
            Formatted string representation of relationships
        """
        pass

    @abstractmethod
    def get_ontology_entities(self, ontology_id: str) -> List[dict]:
        """Get entity ontology definitions for the given ontology_id.
        
        Args:
            ontology_id: The ontology identifier (or None for all ontologies)
            
        Returns:
            List of entity ontology definition dicts
        """
        pass

    @abstractmethod
    def check_read_access_on_ontology(self, effective_user_id: Optional[str], ontology_id: str) -> bool:
        """Check whether effective_user_id has GRANT:READ on the given Ontology node.

        Args:
            effective_user_id: Neo4j User.id. When None, returns True (system/pipeline bypass).
            ontology_id: The id property of the target Ontology node.

        Returns:
            True if access is granted or effective_user_id is None, False otherwise.
        """
        raise NotImplementedError()

    @abstractmethod
    def check_read_access_on_community(self, effective_user_id: Optional[str], community_id: str) -> bool:
        """Check whether effective_user_id has GRANT:READ on the given Community node.

        Resolves the parent Ontology via [:DEFINES] and delegates to
        check_read_access_on_ontology (communities inherit permissions from their ontology).

        Args:
            effective_user_id: Neo4j User.id. When None, returns True (system/pipeline bypass).
            community_id: The id property of the target Community node.

        Returns:
            True if access is granted or effective_user_id is None, False otherwise.
        """
        raise NotImplementedError()

    @abstractmethod
    def check_has_any_read_access(self, effective_user_id: Optional[str]) -> bool:
        """Return True if the effective user has GRANT:READ on at least one Ontology.

        Used as a fast early-abort guard when no specific community is targeted and
        the full ACL enforcement will be handled post-search by get_entities_by_ids.

        Args:
            effective_user_id: Neo4j User.id. When None, returns True (system/pipeline bypass).

        Returns:
            True if the effective user is a tenant-admin or holds GRANT:READ on any Ontology.
        """
        raise NotImplementedError()

    @abstractmethod
    def ontology_exists(self, ontology_id: str) -> bool:
        """Return True if an Ontology node with the given id exists in the backing store."""
        raise NotImplementedError()

    @abstractmethod
    def get_ontology_relationships(self, ontology_id: str) -> List[dict]:
        """Get relationship ontology definitions for the given ontology_id.
        
        Args:
            ontology_id: The ontology identifier (or None for all ontologies)
            
        Returns:
            List of relationship ontology definition dicts
        """
        pass

    @abstractmethod
    def get_ontology_communities(self, ontology_id: str) -> List[str]:
        """Return community IDs linked to the given ontology, sorted alphabetically.
        
        Args:
            ontology_id: The ontology identifier
            
        Returns:
            Sorted list of community ID strings; empty list if none found
        """
        pass

    @abstractmethod
    def validate_community_ids(self, community_ids: Optional[List[str]], ontology_id: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """Return unique valid community IDs and IDs that are not available.

        When ontology_id is provided, validation is limited to communities defined
        by that ontology. Otherwise, all existing communities are considered.
        """
        pass

    @abstractmethod
    def get_ontology_and_entites_as_string(self, goal, ontology_id: str, community_ids: List[str], skip_details: bool, condense_categories: bool = False) -> str:
        """Get the rules for the ontology and all entities and relationships in the specified communities.
        
        Args:
            goal: The current goal (not used at present)
            ontology_id: The ID of the ontology. Special values:
                - ONTOLOGY_ID_ALL ('__ALL__'): Returns all ontologies (used by AgentOntology for selection)
                - None or the default ontology id (from get_default_ontology_id()): Returns None (no specific ontology/categories)
                - Specific ID: Returns that specific ontology
            community_ids: The IDs of the communities. If None, all communities linked to the ontology are considered
            skip_details: If True, only ontology and community descriptions are returned
            condense_categories: If True, output categories as comma-separated list only. If False, output verbose format with descriptions and values.
            
        Returns:
            Formatted string representation of ontology, entities, and relationships, or None
        """
        pass

    @abstractmethod
    def semantic_search_entity(self, prompt: str, community_ids: List[str] = None, limit: int = None, effective_user_id: Optional[str] = None) -> dict:
        """Searches the knowledge graph for entities that match the semantic search query.
        
        Args:
            prompt: The search query
            community_ids: List of community IDs to filter by (optional)
            limit: The maximum number of results to return
            
        Returns:
            Dictionary where keys are entity IDs and values are entity dicts with relationships
        """
        pass

    @abstractmethod
    def entity_to_discourse_text(self, entity: dict, linked_document_char_budget: int = MAX_RAG_MESSAGE_LENGTH_DEFAULT, excluded_document_ids: Optional[Set[str]] = None) -> str:
        """Convert an entity dict (from semantic_search_entity) to a natural-language discourse string.

        The entity dict is expected to contain 'name', 'label', 'description',
        'outgoing_relationships' and 'incoming_relationships'.

        Args:
            entity: Entity dictionary as returned by semantic_search_entity.
            linked_document_char_budget: Character budget for this entity's linked-document blocks.

        Returns:
            Natural-language prose describing the entity and its relationships, followed by linked
            document text in <LINKED_DOCUMENT id="..."> fenced blocks within the supplied budget.
        """
        pass

    @abstractmethod
    def get_linked_documents(self, entity_id: str, limit: int = 10, excluded_document_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """Fetch the Document nodes linked to an entity via IS_DOCUMENTED, including their full stored text.

        Args:
            entity_id: The id property of the entity node.
            limit: Maximum number of linked documents returned.

        Returns:
            List of dicts with 'id', 'title' and 'text' (the full stored Document.text) per linked document.
        """
        pass

    @abstractmethod
    def linked_documents_discourse(self, entity_id: str, limit: int = 10, max_chars: int = MAX_RAG_MESSAGE_LENGTH_DEFAULT, excluded_document_ids: Optional[Set[str]] = None) -> str:
        """Render linked-document blocks within the supplied character budget.

        Each linked document is rendered as:
            <LINKED_DOCUMENT id="..." title="...">
            ```
            Document.text, truncated with a marker when needed to fit max_chars
            ```
            </LINKED_DOCUMENT>

        Args:
            entity_id: The id property of the entity node.
            limit: Maximum number of linked documents included.
            max_chars: Maximum serialized characters for linked-document blocks.

        Returns:
            Concatenated <LINKED_DOCUMENT> blocks within max_chars, or an empty string if none fit.
        """
        pass

    @abstractmethod
    def create_semantic_search_index(self, ignore_warnings: bool = False) -> None:
        """Create the semantic search vector index for entities"""
        pass

    @abstractmethod
    def update_semantic_search(self, entity_id: str) -> None:
        """Update semantic search embedding for a single entity"""
        pass

    @abstractmethod
    def update_semantic_search_all(self) -> None:
        """Update semantic search embeddings for all entities"""
        pass

    @abstractmethod
    def consolidate_sentences(self, is_relationship: bool, attribute_name: str, label: str, ontology_id: str, values_to_consolidate: List[str]) -> str:
        """Consolidate multiple attribute values into one using LLM"""
        pass

    @abstractmethod
    def consolidate_entities(self, restrict_to_entity_ids: List[str] = None, suppress_debug_logs: bool = False) -> bool:
        """Consolidate entity attributes that require consolidation"""
        pass

    @abstractmethod
    def consolidate_relationships(self, suppress_debug_logs: bool = False) -> bool:
        """Consolidate relationship attributes that require consolidation"""
        pass

    @abstractmethod
    def identify_entities_to_consolidate(self, goal, community_id: str, label: str, entities: List[dict]) -> List[List[str]]:
        """Identify which entities should be merged together"""
        pass

    @abstractmethod
    def consolidate_knowledge_graph(self, force_create_semantic_search: bool = False, suppress_debug_logs: bool = False) -> bool:
        """Consolidate all entities and relationships in the knowledge graph"""
        pass

    @abstractmethod
    def remove_relationships_source(self, document_id: str) -> None:
        """Remove relationship information from a specific document source"""
        pass

    @abstractmethod
    def remove_entities_source(self, document_id: str) -> None:
        """Remove entity information from a specific document source"""
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its extracted information from the knowledge graph"""
        pass

    @abstractmethod
    def get_categories_count_for_ontology(self, ontology_id: str, exclude_proposed: bool = False) -> int:
        """Get count of categories for a given ontology."""
        pass

    @abstractmethod
    def get_categories_for_ontology(self, ontology_id: str, exclude_proposed: bool = False) -> List[dict]:
        """Get categories for a given ontology as a list of dictionaries.
        
        Args:
            ontology_id: The ontology identifier
            exclude_proposed: If True, exclude categories with proposed=True (only return confirmed categories)
            
        Returns:
            List of category dictionaries
        """
        pass

    @abstractmethod
    def save_proposed_categories(self, ontology_id: str, proposed_categories: List[Dict[str, Any]], effective_user_id: Optional[str] = None) -> bool:
        """Save proposed categories as KeywordCategory nodes with 'proposed'='true' attribute.

        Gated on GRANT:ADD on the target ontology. When effective_user_id is None the principal is
        resolved from the current request or background context (defaulting to 'global'), so the
        ACL check always runs against a real principal.
        
        Args:
            ontology_id: The ontology identifier to associate proposals with
            proposed_categories: List of category proposals, each with 'text_en', 'description', and optional 'lov'
            effective_user_id: Neo4j User.id of the requesting user
            
        Returns:
            True if all proposals were saved successfully, False otherwise
        """
        pass

    @abstractmethod
    def delete_keyword_categories(self, ontology_id: Optional[str], pattern: Optional[str], delete_all: bool = False) -> int:
        """Delete KeywordCategory nodes by ontology and pattern or delete all.

        Args:
            ontology_id: The ontology id to operate on (required).
            pattern: Optional regex pattern to match `text_en` of KeywordCategory nodes.
            delete_all: If True, delete all KeywordCategory nodes in the ontology.

        Returns:
            Number of deleted nodes.
        """
        raise NotImplementedError()

    @abstractmethod
    def is_categories_only_query(self, ontology_id: str, query: str) -> bool:
        """Check if a query expression contains only valid category searches.
        
        This method determines if a query can be delegated entirely to categories_query()
        for efficient Cypher-level processing. Returns True only if:
        1. The query contains only category expressions (no semantic/text searches)
        2. All mentioned categories exist in the ontology
        
        Args:
            ontology_id: The ontology identifier to validate categories against
            query: Query expression to check (e.g., "COUNTRY=IQ AND IS_DEAD")
            
        Returns:
            True if query contains only valid categories, False otherwise
        """
        pass

    @abstractmethod
    def get_ontology_for_query(self, query: str) -> Optional[str]:
        """Return an ontology id that best matches the provided thalabus query.

        Implementations should examine the query for category identifiers
        (for example: COUNTRY_OF_ORIGIN, IS_DEAD) and return the `id` of an
        Ontology that contains a matching `KeywordCategory` node, or `None` if
        no suitable ontology is found.
        """
        pass
    
    @abstractmethod
    def categories_query(self, ontology_id: str, query: str, effective_user_id: Optional[str] = None) -> List[str]:
        """Search for documents based on category assignments using boolean queries.
        
        Args:
            ontology_id: The ontology identifier to search within
            query: Boolean expression query (e.g., "COUNTRY=IQ AND IS_DEAD" or "DUBLIN_ARTICLE=ART_13,ART_25_2")
            effective_user_id: Neo4j User.id of the requesting user. When provided and ACL_ENFORCEMENT_ENABLED=True,
                      only documents the effective user holds GRANT:READ on are returned. When None, ACL is skipped.
            
        Returns:
            List of document IDs matching the query
        """
        pass

    @abstractmethod
    def get_ontology_id_for_category(self, category_text_en: str) -> Optional[str]:
        """Retrieve the ontology ID that contains the given category (by text_en).
        
        Args:
            category_text_en: The text_en value of the KeywordCategory (e.g., 'MAIN_EVENT').
        
        Returns:
            The ontology ID as a string, or None if not found.
        """
        pass

    @abstractmethod
    def get_ontology_keyword_categories_count(self, ontology_id: str) -> int:
        """Return the number of KeywordCategory nodes contained in the given ontology.

        Args:
            ontology_id: The ID of the ontology to check.

        Returns:
            The count of (Ontology {id})-[:CONTAINS]->(KeywordCategory) nodes (0 if none or on error).
        """
        raise NotImplementedError()

    @abstractmethod
    def find_duplicate_categories(self, document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Find duplicated Category nodes for documents.

        Returns a list of dicts with keys:
            - document_id
            - category_key (KeywordCategory.text_en)
            - category_value (Category.text)
            - count (number of Category nodes)
            - category_node_ids (list of internal node ids)

        If `document_ids` is provided, restrict results to those documents.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_lexicon_for_reasoning_document(self, document_id: str, target_language: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return Lexicon entries linked to the ontology(ies) associated with a ReasoningDocument.

        Follows the graph path:
            (ReasoningDocument {id: document_id}) -[:REASONS_WITH|OWNS]- (Dossier)
            -[:FOLLOWS]-> (Ontology) -[:HAS_LEXICON]-> (Lexicon)

        Args:
            document_id: The ``id`` property of the ReasoningDocument node.
            target_language: Optional ISO 639-1 language code (e.g. ``'it'``, ``'de'``).
                             When provided, only lexicon entries for that language are returned.

        Returns:
            List of dicts, each with keys:
                - ``ontology_id`` (str)  — id of the linked Ontology
                - ``lexicon_id``  (str)  — id of the Lexicon node
                - ``lang``        (str)  — language code of the entry
                - ``text``        (str)  — translation/hint text
            Returns an empty list when no lexicon entries exist or on error.
        """
        raise NotImplementedError()
