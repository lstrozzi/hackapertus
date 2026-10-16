from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any, Optional

class DossiersInterface(ABC):
    """Abstract interface for dossier repository operations."""

    @abstractmethod
    def get_all_dossiers(self, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all dossiers.
        Returns:
            List of dossier dictionaries.

        Each dossier dictionary may include the following keys in addition to basic
        `id`, `name`, and `description`:
            - `follows_ontology_id`: optional id of an `Ontology` node the dossier follows (via `[:FOLLOWS]`)
            - `follows_ontology_name`: optional human-readable name of the followed ontology
            - `template_id`: optional id of a `DocumentTemplate` the dossier is based on (via `[:BASED_ON]`)
            - `template_title`: optional title of that template
        """
        pass

    @abstractmethod
    def delete_dossier(self, dossier_id: str, effective_user_id: Optional[str] = None) -> bool:
        """
        Delete a dossier node (and its relationships) by id.

        Args:
            dossier_id: The ID of the dossier to delete.
            effective_user_id: If provided, DELETE permission is verified before deletion. Pass None to skip ACL.

        Returns:
            True on success, False on denial or failure.
        """
        pass
    @abstractmethod
    def get_dossier(self, dossier_id: str, effective_user_id: Optional[str] = None) -> Union[Dict[str, Any], None]:
        """
        Retrieve a single dossier by its ID.
        Args:
            dossier_id: The ID of the dossier to retrieve.
            effective_user_id: If provided, READ permission is verified. Returns None on denial. Pass None to skip ACL.
        Returns:
            Dossier dictionary, or None if not found or access denied.
        The returned dictionary includes:
            - `document_ids`: list of document ids linked via `[:CONTAINS]`
            - `template_ids`: list of template ids linked via `[:BASED_ON]`
            - `reasoning_ids`: list of reasoning document ids linked via `[:REASONS_WITH]`
            - `follows_ontology_id` / `follows_ontology_name`: optional ontology linked via `[:FOLLOWS]`
        """
        pass

    @abstractmethod
    def get_documents_of_dossier(self, dossier_id: str, include_details: bool = False, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all documents belonging to a specific dossier.

        Args:
            dossier_id: The ID of the dossier whose documents to retrieve.
            include_details: If True, return additional document attributes (text, enhanced_text,
                title, status, summary, keywords, categories, timestamps, etc.) similar to
                DocumentsService.get_document(include_details=True).
            effective_user_id: If provided, READ permission on the dossier is verified. Returns [] on denial. Pass None to skip ACL.

        Returns:
            List of document dictionaries belonging to the specified dossier. Each item will at
            minimum contain `id`, and may include extended fields when `include_details` is True.
        """
        pass

    @abstractmethod
    def get_ontology_id_for_dossier(self, dossier_id: str, effective_user_id: Optional[str] = None) -> Union[str, None]:
        """
        Return the ontology id that the dossier follows, if any.

        Args:
            dossier_id: The dossier identifier
            effective_user_id: If provided, READ permission on the dossier is verified. Returns None on denial. Pass None to skip ACL.

        Returns:
            Ontology id string if the dossier follows an ontology, otherwise None
        """
        pass
    @abstractmethod
    def link_document_to_dossier(self, dossier_id: str, document_ids: List[str], effective_user_id: Optional[str] = None) -> bool:
        """
        Link documents to a dossier by creating `[:CONTAINS]` relationships.
        Args:
            dossier_id: The ID of the dossier.
            document_ids: The IDs of the documents to link.
            effective_user_id: If provided, ADD permission on the dossier is verified. Returns False on denial. Pass None to skip ACL.
        Returns:
            True on success, False on denial or failure.
        """
        pass

    @abstractmethod
    def create_dossier(self, id: str, name: str, description: str) -> Union[Dict[str, Any], None]:
        """
        Create a new dossier.
        Args:
            id: The ID of the new dossier.
            name: The name of the new dossier.
            description: The description of the new dossier.
        Returns:
            The created dossier dictionary.
        """
        pass

    @abstractmethod
    def get_category_usage_stats_in_dossier(self, dossier_id: str, category_text_en: str, effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve usage statistics for a category in a dossier.
        
        Shows how many documents in the dossier have the specified category assigned.
        If the KeywordCategory defines a LoV (List of Values), also shows the count breakdown by value.
        
        Args:
            dossier_id: The ID of the dossier to analyze.
            category_text_en: The text_en value of the KeywordCategory (e.g., 'COUNTRY_OF_ORIGIN').
        
        Returns:
            Dictionary containing:
                - 'category': The category name (text_en)
                - 'ontology_id': The ontology that contains this category
                - 'total_count': Total number of documents with this category in the dossier
                - 'has_lov': Whether the KeywordCategory defines a List of Values
                - 'lov': The LoV string if defined, None otherwise
                - 'value_counts': Dict mapping each unique category value to its document count
                - 'error': Error message if operation failed (instead of above fields)
        """
        pass

    @abstractmethod
    def get_user_dossier_id(self, user_id: str) -> Union[str, None]:
        """
        Return the ID of the personal dossier owned by the user's group.

        Traverses: (u:User:SystemAcl {id: user_id})-[:MEMBER_OF]->(g:Group:SystemAcl {isUserGroup: true})-[:OWNS]->(d:Dossier)

        Args:
            user_id: The user identifier.

        Returns:
            Dossier ID string if found, otherwise None.
        """
        pass

    @abstractmethod
    def count_documents_in_dossier(self, dossier_id: str, effective_user_id: Optional[str] = None) -> int:
        """
        Count the number of documents linked to a given dossier via `[:CONTAINS]`.

        Args:
            dossier_id: The dossier identifier.
            effective_user_id: If provided, READ permission on the dossier is verified. Returns 0 on denial. Pass None to skip ACL.

        Returns:
            The number of documents linked to the dossier as an integer.
        """
        pass

    @abstractmethod
    def get_dossiers_for_document(self, document_id: str, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return all dossiers that contain the given document via `[:CONTAINS]`.

        Args:
            document_id: The document identifier.
            effective_user_id: If provided, only dossiers on which the effective user holds READ permission are returned. Pass None to skip ACL.

        Returns:
            List of dossier dictionaries, each containing at minimum 'id', 'name', 'description'.
            Returns an empty list if no accessible dossiers contain the document.
        """
        pass