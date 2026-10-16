from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Union, Dict, Any

from src.core.base.enums import DocumentStatus, KnowledgeGraphStatus




class FragmentType(Enum):
    NotIdentified = "NotIdentified"
    Document = "Document"
    Chapter = "Chapter"
    Paragraph = "Paragraph"
    Internet = "Internet"
    KnowledgeGraphEntity = "KnowledgeGraphEntity"


class DocumentsInterface(ABC):
    """Abstract interface for document repository operations."""

    @abstractmethod
    def get_all_documents(self, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all documents.
        Args:
            effective_user_id: If provided, only documents the effective user has READ permission on are returned. Pass None to skip ACL.
        Returns:
            List of document dictionaries.
        """
        pass

    @abstractmethod
    def get_document(self, document_id: str, include_details: bool = False, include_category_translations: bool = True, include_text: bool = True, include_summary: bool = True, effective_user_id: Optional[str] = None) -> Union[Dict[str, Any], None]:
        """
        Retrieve a single document by its ID.
        Args:
            document_id: The ID of the document to retrieve.
            include_details: If True, include keywords and category metadata.
            include_category_translations: If False, return category entries containing only `text_en` (no other language fields).
            include_text: If True (default), include the text and enhanced_text fields. When False,
                          those fields are omitted from the query to avoid unnecessary Neo4j traffic.
            include_summary: If True (default), fetch the document summary via HAS_SUMMARY. When False,
                             the summary subquery is skipped entirely to reduce Neo4j traffic.
            effective_user_id: If provided, READ permission is verified. Returns None on denial. Pass None to skip ACL.
        Returns:
            Document dictionary or None if not found or access denied.
        """
        pass

    @abstractmethod
    def get_documents(self, document_ids: List[str], include_text: bool = False, include_summary: bool = True, include_details: bool = False, include_category_translations: bool = True, merge_clause: Optional[str] = None, limit: Optional[int] = None, effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve multiple documents by their IDs in a single batched query.
        Args:
            document_ids: List of document IDs to retrieve.
            include_text: If True, include the text and enhanced_text fields. When False (default),
                          those fields are omitted from the query to avoid unnecessary Neo4j traffic.
            include_summary: If True (default), fetch the document summary via HAS_SUMMARY. When False,
                             the summary subquery is skipped entirely to reduce Neo4j traffic.
            include_details: If True, include keywords and category metadata (same as get_document).
            include_category_translations: If False, return category entries containing only `text_en`
                                           (no other language fields). Only used when include_details=True.
            merge_clause: Optional Cypher WHERE clause to add additional filters (e.g., "d.extraction_quality > 0.8").
                         Should not include the 'WHERE' keyword - only the condition itself. Defaults to None.
            limit: Optional maximum number of documents to return. Defaults to None (no limit).
            effective_user_id: If provided, only documents the effective user has READ permission on are returned. Pass None to skip ACL.
        Returns:
            Dictionary mapping each document_id to its document dictionary (same shape as get_document),
            or to None if that document was not found or access denied.
        """
        pass

    @abstractmethod
    def generate_new_document_id(self) -> str:
        """
        Generate a new document ID following the naming convention 'docid-nnnn'.
        Returns:
            New document ID as a string.
        """
        pass

    def get_document_id_by(self, url: Optional[str]) -> Union[str, None]:
        """
        Retrieve a document ID by its URL.
        Args:
            url: The URL of the document.
        Returns:
            Document ID as a string or None if not found.
        """
        pass

    @abstractmethod
    def create_document(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new document.
        Args:
            document_data: Dictionary containing document properties.
        Returns:
            Created document dictionary or error message.
        """
        pass

    @abstractmethod
    def create_or_update_document_by_url(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or update a document node by its `url`.

        This operation should merge on the `url` property and either create
        a new document node or update the existing one. Implementations must
        ensure this is safe under concurrent calls (atomic upsert).
        Args:
            document_data: Dictionary containing document properties, including `url`.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def update_document(self, document_id: str, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing document.
        Args:
            document_id: The ID of the document to update.
            document_data: Dictionary containing updated document properties.
        Returns:
            Updated document dictionary or error message.
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document by its ID.
        Args:
            document_id: The ID of the document to delete.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def get_documents_by_status(self, status: DocumentStatus, merge_clause: Optional[str] = None, limit: Optional[int] = None, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents filtered by status.
        Args:
            status: The status to filter by as a DocumentStatus enum value.
            merge_clause: Optional Cypher WHERE clause to add additional filters (e.g., "d.extraction_quality > 0.8").
                         Should not include the 'WHERE' keyword - only the condition itself. Defaults to None.
            limit: Optional maximum number of documents to return. Defaults to None (no limit).
        Returns:
            List of document dictionaries matching the status and optional filters.
        """
        pass

    @abstractmethod
    def get_documents_by_type(self, document_type: str, merge_clause: Optional[str] = None, limit: Optional[int] = None, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents filtered by type.
        Args:
            document_type: The document type to filter by.
            merge_clause: Optional Cypher WHERE clause to add additional filters (e.g., "d.extraction_quality > 0.8").
                         Should not include the 'WHERE' keyword - only the condition itself. Defaults to None.
            limit: Optional maximum number of documents to return. Defaults to None (no limit).
            effective_user_id: If provided, only documents the effective user has READ permission on are returned. Pass None to skip ACL.
        Returns:
            List of document dictionaries matching the type and optional filters, and accessible to the effective user.
        """
        pass

    @abstractmethod
    def get_chapters_for_documents(self, document_ids: List[str], effective_user_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve all chapters for given documents.
        Args:
            document_ids: List of document IDs to fetch chapters for.
            effective_user_id: If provided, only chapters from documents the effective user has READ permission on are returned. Pass None to skip ACL.
        Returns:
            Dictionary mapping document_id to list of chapter dictionaries with 'id', 'text', and 'title'.
        """
        pass

    @abstractmethod
    def get_chapter_counts_for_documents(self, document_ids: List[str], effective_user_id: Optional[str] = None) -> Dict[str, int]:
        """
        Retrieve only the chapter count per document (no text or title transferred).
        Args:
            document_ids: List of document IDs to count chapters for.
            effective_user_id: If provided, only chapters from documents the effective user has READ permission on are counted. Pass None to skip ACL.
        Returns:
            Dictionary mapping document_id to chapter count.
        """
        pass

    @abstractmethod
    def get_category_search_content(self, document_ids: List[str], max_chars: int) -> Dict[str, Any]:
        """Fetch category-search chapters or summaries that fit within a character budget."""
        pass

    @abstractmethod
    def get_content_for_goal(self, content_id_attribute: str, content_ids: List[str], content_label: str, only_if_connected_to: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load goal context content and associated document metadata."""
        pass

    @abstractmethod
    def update_content_enhanced_text(self, content_id_attribute: str, content_id: str, content_label: str, enhanced_text: str, enhanced_type: Optional[str] = None) -> bool:
        """Persist enhanced content fields for a document or fragment node."""
        pass

    @abstractmethod
    def get_document_categories(self, document_id: str, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve categories assigned to a document.

        Each returned dict should contain at least:
            - 'category_id' : id of the KeywordCategory node (kc.id)
            - 'category_key' : kc.text_en (human-readable key)
            - 'category_value' : c.text (the optional value assigned on the Category node)

        Args:
            document_id: The document id (either `id` or `documentid`).
            effective_user_id: If provided, READ permission on the document is verified. Returns empty list on denial. Pass None to skip ACL.
        Returns:
            List of category dictionaries, or empty list if access denied.
        """
        pass

    @abstractmethod
    def document_has_any_categories(self, document_id: str, effective_user_id: Optional[str] = None) -> bool:
        """
        Check whether the given document has any :HAS_CATEGORY relationships to Category nodes.

        Args:
            document_id: The document id (either `id` or `documentid`).
            effective_user_id: If provided, READ permission on the document is verified. Returns False on denial. Pass None to skip ACL.

        Returns:
            True if the document has one or more :HAS_CATEGORY relationships and is accessible, False otherwise.
        """
        pass

    @abstractmethod
    def count_document_categories(self, document_id: str) -> int:
        """
        Return the number of Category nodes linked to the given document via :HAS_CATEGORY.

        Args:
            document_id: The document id (either `id` or `documentid`).

        Returns:
            Integer count of Category nodes linked to the document (0 if none or on error).
        """
        pass

    @abstractmethod
    def remove_document_category(self, document_id: str, category_text_en: str) -> Dict[str, Any]:
        """
        Remove a single category from a document.

        Deletes the Category node(s) linked to the specified KeywordCategory (identified by text_en)
        for the given document.

        Args:
            document_id: The document id (either `id` or `documentid`).
            category_text_en: The text_en value of the KeywordCategory to remove (e.g., 'MAIN_EVENT').
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def remove_all_document_categories(self, document_id: str) -> Dict[str, Any]:
        """
        Remove all categories from a document.

        Deletes all Category nodes linked to the document via HAS_CATEGORY relationships.

        Args:
            document_id: The document id (either `id` or `documentid`).
        Returns:
            Success/error message dictionary with count of removed categories.
        """
        pass

    @abstractmethod
    def upsert_document_category(self, document_id: str, category_text_en: str, category_value: Optional[str] = None) -> Dict[str, Any]:
        """
        Upsert (create or update) a category for a document.

        If the category value is provided, updates or creates the Category node with that value.
        If no value is provided, uses the category_text_en as the category text.

        Args:
            document_id: The document id (either `id` or `documentid`).
            category_text_en: The text_en value of the KeywordCategory (e.g., 'MAIN_EVENT').
            category_value: Optional value for the category. If None, category_text_en is used as the text.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def upsert_document_categories_batch(self, document_ids: List[str], category_text_en: str, category_value: Optional[str] = None, max_workers: int = 4) -> Dict[str, Any]:
        """
        Upsert (create or update) a category for multiple documents in parallel.

        Parallelizes upsert_document_category calls across multiple documents using ThreadPoolExecutor.

        Args:
            document_ids: List of document ids to update.
            category_text_en: The text_en value of the KeywordCategory (e.g., 'MAIN_EVENT').
            category_value: Optional value for the category. If None, category_text_en is used as the text.
            max_workers: Maximum number of parallel worker threads (default: 4).

        Returns:
            Dictionary containing:
                - 'success': bool indicating overall success
                - 'success_count': number of successfully updated documents
                - 'total_count': total number of documents processed
                - 'affected_total': total number of categories upserted
                - 'errors': List of (document_id, error_message) tuples for failed documents
        """
        pass

    @abstractmethod
    def validate_category_key(self, category_text_en: str) -> Dict[str, Any]:
        """
        Validate that a KeywordCategory with the given `text_en` exists.

        Returns a dictionary with at least the keys:
            - 'exists': bool
            - 'lov': Optional[list|str] (the stored LoV value, if any)
        """
        pass

    @abstractmethod
    def validate_category_value(self, category_text_en: str, category_value: str) -> Dict[str, Any]:
        """
        Validate that `category_value` is allowed for the KeywordCategory identified
        by `category_text_en`. If the KeywordCategory does not define a LoV, any
        value is considered valid.

        Returns a dictionary with keys:
            - 'valid': bool
            - 'allowed': Optional[list] (list of allowed values when LoV is present)
        """
        pass

    @abstractmethod
    def get_documents_count(self, dossier_ids: Optional[List[str]] = None, merge_clause: Optional[str] = None, limit: Optional[int] = None) -> int:
        """
        Return the total number of documents in the repository.

        Behavior:
          - If `dossier_ids` is None: count all documents.
          - If `dossier_ids` is an empty list: return 0.
          - If `dossier_ids` is a non-empty list: count documents owned by any
            of the dossiers whose ids are in the list (via
            `(:Dossier)-[:CONTAINS]->(:Document)`).

        Args:
            merge_clause: Optional Cypher WHERE clause to add additional filters (e.g., "d.status = 'active'").
                         Should not include the 'WHERE' keyword - only the condition itself. Defaults to None.
            limit: Optional maximum number of documents to count. Defaults to None (no limit).

        Returns:
            Total document count as an integer.
        """
        pass

    @abstractmethod
    def update_document_status(self, document_id: str, status: DocumentStatus) -> Dict[str, Any]:
        """
        Update the status of a document.
        Args:
            document_id: The ID of the document to update.
            status: New status as a DocumentStatus enum value.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def update_knowledge_graph_status(self, document_id: str, knowledge_graph_status: KnowledgeGraphStatus) -> Dict[str, Any]:
        """
        Update the knowledge graph status of a document.
        Args:
            document_id: The ID of the document to update.
            knowledge_graph_status: New knowledge graph status as a KnowledgeGraphStatus enum value.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def write_document_file(self, document_id: str, document_name: str, document_text: str) -> Optional[str]:
        """
        Write the physical file for a document to the document repository.
        Args:
            document_id: The ID of the document (used for logging).
            document_name: The filename to write (including extension, e.g. 'My Document.txt').
            document_text: The text content to write to the file.
        Returns:
            The full file path on success, or None on failure.
        """
        pass

    @abstractmethod
    def update_document_file(self, document_id: str) -> Optional[str]:
        """
        Overwrite the physical file for a document in the document repository using its current
        Neo4j content. Only supported for documents whose 'type' attribute is one of:
        'text'/'txt' (.txt), 'md'/'markdown' (.md), 'html' (.html), 'csv' (.csv).
        For html type, enhanced_text is written; for all others, plain text derived from
        enhanced_text (or text as fallback) is written.
        Args:
            document_id: The ID of the document to overwrite.
        Returns:
            The full file path on success, None if the type is not supported or on failure.
        """
        pass

    @abstractmethod
    def update_document_configuration(self, document_id: str, configuration_id: str) -> Dict[str, Any]:
        """
        Update the document configuration.
        Args:
            document_id: The ID of the document to update.
            configuration_id: New configuration ID for the document.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def update_document_categories(self, document_id: str, categories: Dict[str, Optional[str]], ontology_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Update the categories associated with a document.
        Args:
            document_id: The ID of the document to update.
            categories: Dictionary of category names to values.
            ontology_id: Optional. The ID of the ontology that contains these categories.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def get_category_ontology_id_for_document(self, document_id: str) -> Optional[str]:
        """
        Retrieve the default category ontology ID associated with a document.
        Args:
            document_id: The ID of the document.
        Returns:
            The ontology ID as a string, or None if not found or not specified.
        """
        pass

    @abstractmethod
    def create_or_update_chapter(self, document_id: str, chapter_id: str, chapter_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or update a chapter for a document with embeddings.
        Args:
            document_id: The ID of the document.
            chapter_id: The ID of the chapter.
            chapter_data: Dictionary containing chapter text, title, and embedding.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def create_or_update_image(self, document_id: str, chapter_id: str, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or update an image associated with a chapter.
        Args:
            document_id: The ID of the document.
            chapter_id: The ID of the chapter.
            image_data: Dictionary containing image ID and base64 data.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def create_vector_index(self, index_name: str, node_label: str, property_name: str, dimension: int) -> Dict[str, Any]:
        """
        Create a vector index for embeddings.
        Args:
            index_name: Name of the index to create.
            node_label: Neo4j node label (e.g., 'Chapter').
            property_name: Property name containing embeddings (e.g., 'embedding').
            dimension: Dimension of the vector embeddings.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def create_all_required_indexes(self) -> Dict[str, Any]:
        """
        Create all required vector and fulltext indexes for document ingestion.
        This should be called once at the start of ingestion to ensure all indexes exist.
        Creates indexes for: Chapter, Paragraph, Summary, Keyword, Question nodes.
        Returns:
            Success/error message dictionary with details of created indexes.
        """
        pass

    @abstractmethod
    def create_or_update_keyword(self, document_id: str, node_id: str, keyword: str, lang: str, embedding: List[float]) -> Dict[str, Any]:
        """
        Create or update a keyword with embedding.
        Args:
            document_id: The ID of the document.
            node_id: The ID of the node (chapter) to link the keyword to.
            keyword: The keyword text.
            lang: Language code for the keyword.
            embedding: Vector embedding for the keyword.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def create_or_update_summary(self, chapter_id: str, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or update a summary for a chapter.
        Args:
            chapter_id: The ID of the chapter.
            summary_data: Dictionary containing summary text, title, and embedding.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def create_or_update_document_summary(self, document_id: str, summary_text: str, title: str) -> Dict[str, Any]:
        """
        Create or update a document-level summary.
        Args:
            document_id: The ID of the document.
            summary_text: The summary text content.
            title: The title of the summary.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def create_or_update_paragraph(self, chapter_id: str, paragraph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or update a paragraph for a chapter.
        Args:
            chapter_id: The ID of the chapter.
            paragraph_data: Dictionary containing paragraph ID, text, and embedding.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def create_or_update_paragraphs_batch(self, chapter_id: str, paragraphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create or update multiple paragraphs for a chapter in a single operation.
        Args:
            chapter_id: The ID of the chapter.
            paragraphs: List of dictionaries containing paragraph data.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def create_or_update_questions_batch(self, chapter_id: str, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create or update multiple questions for a chapter in a single operation.
        Args:
            chapter_id: The ID of the chapter.
            questions: List of dictionaries containing question data.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def get_fragments_by_ids(self, fragment_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve fragments (documents, chapters, paragraphs, or summaries) by their IDs.

        Fragment ID formats:
        - Document: 'docid-xxx'
        - Chapter: 'docid-xxx-cyyy'
        - Paragraph: 'docid-xxx-cyyy-pzzz'

        Returns:
            List of dictionaries with keys: fragmentid, text, enhanced_text, type
        """
        pass

    @abstractmethod
    def get_nodes_from_fragments(self, fragment_ids: List[str]) -> List[Any]:
        """
        Retrieve Neo4j node records for the provided fragment ids.

        Fragment id formats:
        - Document: 'docid-xxx'
        - Chapter: 'docid-xxx-cyyy'
        - Paragraph: 'docid-xxx-cyyy-pzzz'

        Returns:
            List of Result rows (each row contains the matched node as first column).
        """
        pass

    @abstractmethod
    def get_fragment_type(self, fragment_id: str, default_type: FragmentType = FragmentType.NotIdentified) -> FragmentType:
        """
        Determine the fragment type for a single fragment id and return a `FragmentType`.
        Rules:
          - if the fragment contains '-p' -> Paragraph
          - elif contains '-c' -> Chapter
          - elif starts with 'docid-' -> Document
          - else -> default_type
        """
        pass

    @abstractmethod
    def create_fulltext_index(self, index_name: str, node_label: str, properties: List[str]) -> Dict[str, Any]:
        """
        Create a fulltext index.
        Args:
            index_name: Name of the index to create.
            node_label: Neo4j node label.
            properties: List of property names to index.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def delete_document_nodes_for_reingestion(self, document_id: str) -> Dict[str, Any]:
        """
        Delete document nodes in preparation for re-ingestion.
        This deletes:
        1. All nodes with documentid except the Document node itself
        2. All Keywords associated to the Document node
        3. Orphaned Keyword and Paragraph nodes (no relationships)
        4. Sets extraction_quality to NULL on the Document node
        Args:
            document_id: The ID of the document.
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def delete_all_indexes_and_clear_database(self) -> Dict[str, Any]:
        """
        Clear the graph database (except CacheEntry nodes) and delete all indexes.
        This is used when completely resetting the RAG database.
        Returns:
            Success/error message dictionary.
        """
        pass

    # ── Fragment-ID helpers ───────────────────────────────────────────────────
    # Fragment IDs come in two formats that must be handled transparently:
    #   plain :  docid-xxx[-cyyy[-pzzz]]
    #   URI   :  doc://docid-xxx[-cyyy[-pzzz]]
    #
    # Both formats follow the same suffix conventions:
    #   Chapter   ends with  -c{N}          (e.g. docid-42-c3)
    #   Paragraph ends with  -c{N}-p{M}     (e.g. docid-42-c3-p7)
    #   Summary   ends with  -s / -s{N}     (e.g. docid-42-s or docid-42-s2)
    #   Document  has no type suffix        (e.g. docid-42)

    @staticmethod
    def normalize_fragment_id(fragment_id: str) -> str:
        """Strip the 'doc://' URI scheme if present and return the bare fragment ID."""
        if fragment_id and fragment_id.startswith("doc://"):
            return fragment_id[len("doc://"):]
        return fragment_id

    @staticmethod
    def parse_fragment_id(fragment_id: str) -> Dict[str, Optional[str]]:
        """Parse a fragment ID (plain or URI form) into its component parts.

        Returns a dict with keys:
            raw          – original value as passed in
            bare         – fragment_id with 'doc://' prefix stripped
            document_id  – the document portion only (e.g. 'docid-42')
            chapter_id   – full chapter ID if applicable (e.g. 'docid-42-c3'), else None
            type         – 'document' | 'chapter' | 'paragraph' | 'summary' | 'kg_entity' | 'unknown'
        """
        if not fragment_id or not isinstance(fragment_id, str):
            return {"raw": fragment_id, "bare": fragment_id, "document_id": None, "chapter_id": None, "type": "unknown"}

        if fragment_id.startswith("kg://"):
            # kg:// entity IDs have no chapter/paragraph suffix convention; never treat them as a document.
            return {"raw": fragment_id, "bare": fragment_id, "document_id": None, "chapter_id": None, "type": "kg_entity"}

        bare = fragment_id[len("doc://"):] if fragment_id.startswith("doc://") else fragment_id
        parts = bare.rsplit("-", 1)
        suffix = parts[1] if len(parts) == 2 else ""

        if suffix.startswith("p") and suffix[1:].isdigit():
            # Paragraph: bare = "{doc_id}-c{N}-p{M}"
            parent = parts[0].rsplit("-", 1)
            chapter_id = parts[0] if (len(parent) == 2 and parent[1].startswith("c") and parent[1][1:].isdigit()) else None
            document_id = parent[0] if chapter_id else parts[0]
            return {"raw": fragment_id, "bare": bare, "document_id": document_id, "chapter_id": chapter_id, "type": "paragraph"}

        if suffix.startswith("c") and suffix[1:].isdigit():
            # Chapter: bare = "{doc_id}-c{N}"
            return {"raw": fragment_id, "bare": bare, "document_id": parts[0], "chapter_id": bare, "type": "chapter"}

        if suffix == "s" or (suffix.startswith("s") and suffix[1:].isdigit()):
            # Summary: bare = "{doc_id}-s" or "{doc_id}-s{N}"
            return {"raw": fragment_id, "bare": bare, "document_id": parts[0], "chapter_id": None, "type": "summary"}

        # Document (or unrecognised): no type suffix
        return {"raw": fragment_id, "bare": bare, "document_id": bare, "chapter_id": None, "type": "document"}

    @staticmethod
    def get_document_id_from_fragment_id(fragment_id: str) -> Optional[str]:
        """Return just the document_id portion of a fragment ID (plain or URI form)."""
        parsed = DocumentsInterface.parse_fragment_id(fragment_id)
        return parsed.get("document_id")

    @staticmethod
    def get_chapter_id_from_fragment_id(fragment_id: str) -> Optional[str]:
        """Return the chapter ID for a paragraph or chapter fragment, or None for other types."""
        parsed = DocumentsInterface.parse_fragment_id(fragment_id)
        return parsed.get("chapter_id")

    @abstractmethod
    def delete_orphan_document_nodes(self) -> Dict[str, Any]:
        """
        Delete orphan document nodes (Chapter, Paragraph etc. nodes without any relationships to a Document node).
        Returns:
            Success/error message dictionary.
        """
        pass

    @abstractmethod
    def get_documents_with_multiple_categories(self, dossier_id: str, ontology_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all documents in a dossier that have been assigned more than one category.
        
        Args:
            dossier_id: The ID of the dossier to filter documents by.
            ontology_id: Optional. If provided, only count categories from this ontology when determining "multiple categories".
                        If not provided, all categories are counted.
        
        Returns:
            List of document dictionaries with at least 'id', 'documentid', and 'categories' fields.
            The 'categories' field contains only the categories assigned to that document.
        """
        pass

    @abstractmethod
    def find_ontology_for_document(self, document_id: str) -> Optional[str]:
        """
        Find the ontology ID associated with a document via its owning dossier.

        Traverses the graph: (Dossier)-[:CONTAINS]->(Document) and (Dossier)-[:FOLLOWS]->(Ontology)
        to resolve the ontology that governs categorisation for the given document.

        Args:
            document_id: The ID of the document.

        Returns:
            The ontology ID string if the document belongs to a dossier that follows an ontology,
            None otherwise (document not in a dossier, or dossier has no ontology).
        """
        pass

    @abstractmethod
    def link_document_to_ontology(self, document_id: str, ontology_id: str) -> Dict[str, Any]:
        """
        Create a FOLLOWS relationship from a Document node to an Ontology node.

        Args:
            document_id: The ID of the document.
            ontology_id: The ID of the ontology.

        Returns:
            Dict with 'success' key on success, or 'error' and 'error_message' on failure.
        """
        pass

    @abstractmethod
    def get_document_ids_for_dossier(self, dossier_id: str, merge_clause: Optional[str] = None, limit: Optional[int] = None, effective_user_id: Optional[str] = None) -> List[str]:
        """
        Return the IDs of all documents contained by a given dossier.

        Args:
            dossier_id: The ID of the dossier.
            merge_clause: Optional Cypher WHERE clause to filter documents (e.g., "d.status = 'active'").
                         Should not include the 'WHERE' keyword - only the condition itself. Defaults to None.
            limit: Optional maximum number of document IDs to return. Defaults to None (no limit).
            effective_user_id: If provided, READ permission on the dossier is verified. Returns empty list on denial. Pass None to skip ACL.

        Returns:
            List of document ID strings matching the dossier and optional filter.
            Empty list if the dossier does not exist or owns no documents, or if access is denied.
        """
        pass

    @abstractmethod
    def search_documents_by_semantic(self, documents: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Filter a dict of document metadata by semantic relevance to a free-text query.

        Each document is represented to the reranker as "<name>: <summary>"
        and only those scoring at or above MIN_RERANKING_SCORE_DOC are kept.
        If the reranking service is unavailable or returns no results, the
        original dict is returned unchanged.

        Args:
            documents: Dict mapping document_id -> document dict with at least
                       'name' and 'summary' keys (as returned by get_all_documents
                       or get_documents).
            query: Natural-language description of the document(s) to find.

        Returns:
            Filtered dict with the same structure, containing only the documents
            that are semantically relevant to the query.
        """
        pass

    @abstractmethod
    def get_document_id_by_name(self, doc_name: str) -> Optional[str]:
        """
        Retrieve a document ID by its name property.
        Args:
            doc_name: The name of the document (d.name in Neo4j).
        Returns:
            Document ID string, or None if not found.
        """
        pass

    @abstractmethod
    def get_chapter_for_editing(self, chapter_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a chapter's editable content fields for in-memory HTML editing.
        Returns a dict with keys: text, enhanced_text, enhanced_type, doc_type.
        Returns None if not found, or an error dict on failure.
        """
        pass

    @abstractmethod
    def get_document_for_editing(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document's editable content fields for in-memory HTML editing.
        Returns a dict with keys: text, title, enhanced_text, enhanced_type, doc_type.
        Returns None if not found, or an error dict on failure.
        """
        pass

    @abstractmethod
    def get_chapter_ids_for_document(self, doc_id: str) -> List[str]:
        """
        Retrieve the IDs of all chapters belonging to a document.
        Args:
            doc_id: The document ID.
        Returns:
            List of chapter ID strings (may be empty if the document has no chapters).
        """
        pass

    @abstractmethod
    def save_enhanced_content(self, content_id: str, node_label: str, enhanced_text: str, enhanced_type: str, doc_type: str, original_text: str, tx: str) -> None:
        """
        Persist edited enhanced content to Neo4j, creating a version snapshot when the
        transaction ID has changed, and back-converting the plain-text field from the
        new enhanced_text based on doc_type.
        Args:
            content_id:    The ID of the Document or Chapter node.
            node_label:    'Document' or 'Chapter'.
            enhanced_text: The new HTML content to persist.
            enhanced_type: The content type identifier (e.g. 'html').
            doc_type:      The document type string (e.g. 'md', 'text', 'html', 'csv').
            original_text: The current plain-text field value (used as fallback).
            tx:            The current plan transaction / GUID string.
        """
        pass

    @abstractmethod
    def delete_content_versions(self, content_id: str, node_label: str) -> bool:
        """
        Permanently delete all version snapshots for a Document or Chapter node.
        Args:
            content_id:  The ID of the Document or Chapter.
            node_label:  'Document' or 'Chapter'.
        Returns:
            True on success.
        """
        pass

    @abstractmethod
    def delete_temporary_documents(self, sessionid: str) -> None:
        """Delete all temporary ReasoningDocument nodes for the given session that are not linked to any Dossier."""
        pass
