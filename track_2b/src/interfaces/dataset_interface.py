from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union

from click import Option

# Re-export for backward compatibility; these are defined in dataset_helpers module.
from src.utils.dataset_helpers import parse_reference_uri, build_reference_uri, resolve_source_reference, EMBEDDED_REFERENCE_URI_RE


class DatasetInterface(ABC):
    """Interface for dataset operations in Neo4j database."""

    @abstractmethod
    def contains_kg_entities(self, dataset_name: str) -> bool:
        """Return whether the dataset contains at least one knowledge graph Entity node."""
        pass
    
    @abstractmethod
    def generate_unique_dataset_name(self, prefix: str = "dataset_", session_id: Optional[str] = None) -> Tuple[str, int]:
        """Generate a unique dataset name with the given prefix."""
        pass

    @abstractmethod
    def create_dataset(
        self,
        records: List[Union[Dict[str, Any], Any]],
        dataset_id: Optional[str] = None,
        userid: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
        subset_of_datasetids: Optional[List[str]] = None,
        # weights: mapping from node identifier (node property `id`) to numeric weight
        # Example: {"docid-1": 0.9, "docid-2": 0.5}
        # The keys must match the Neo4j node property used in Cypher as `n.id`.
        # Values should be floats (or ints) representing the weight for that node.
        weights: Optional[Dict[str, float]] = None,
        permanent: bool = False,
        invisible: Optional[bool] = None
    ) -> Dict[str, Union[bool, str, list]]:
        """Create a new dataset with the given records and metadata.
        
        Args:
            subset_of_datasetids: List of parent dataset IDs (currently only first one is used)
            permanent: If True, the dataset is NOT linked to the current Session node.
        """
        pass

    @abstractmethod
    def set_dataset_invisible(self, dataset_id: str, invisible: Optional[bool]) -> Dict[str, Union[bool, str]]:
        """Set a dataset's visibility. None removes the invisible property."""
        pass

    @abstractmethod
    def set_dataset_readonly(self, dataset_id: str, readonly: bool) -> Dict[str, Union[bool, str]]:
        """Set a permanent dataset's read-only status."""
        pass
    
    @abstractmethod
    def create_dataset_from_fragments(
        self,
        fragments: List[str],
        userid: str,
        dataset_name: str,
        dataset_description: str = None,
        metadata: Dict[str, Any] = None,
        dataset_id: Optional[str] = None,
        domain: Optional[str] = None,
        subset_of_datasetids: Optional[List[str]] = None,
        # weights: mapping from node identifier (node property `id`) to numeric weight
        # Example: {"docid-1": 0.9, "docid-2": 0.5}
        # The keys must match the Neo4j node property used in Cypher as `n.id`.
        # Values should be floats (or ints) representing the weight for that node.
        weights: Optional[Dict[str, float]] = None,
        invisible: Optional[bool] = None
    ) -> Dict[str, Union[bool, str, list]]:
        """Create a new dataset with the given fragments and metadata.
        
        Args:
            subset_of_datasetids: List of parent dataset IDs (currently only first one is used)
            fragments may include skill:// URIs (skill://SkillName/InstanceId), which are
            materialized as :DatasetObject:Skill nodes (skill, input, output) via get_parameters().
        """
        pass

    @abstractmethod
    def upsert_dataset_fragments(
        self,
        dataset_id: str,
        fragments: List[str],
        weight: float = 1.0,
        hidden: Optional[bool] = None,
    ) -> Dict[str, Union[bool, str, int]]:
        """Add or update graph fragments in a dataset without duplicating relationships.

        fragments may include skill:// URIs, materialized as :DatasetObject:Skill nodes.
        """
        pass
    
    @abstractmethod
    def get_fragments_from_dataset(
        self,
        dataset_id: str,
        include_weights: bool = True
    ) -> List[Dict[str, Any]]:
        """Get all Document, Chapter, and Paragraph nodes associated with a dataset.
        
        Returns a list of dictionaries with node properties and optional weights.
        The format matches what create_dataset_from_fragments() expects as input.
        
        Args:
            dataset_id: The dataset ID (with or without session prefix)
            include_weights: Whether to include the weight from AGGREGATES relationship
            
        Returns:
            List of dictionaries with node properties (id, fragmentid, etc.) and optional 'weight' key
        """
        pass
    
    @abstractmethod
    def delete_dataset(self, dataset_name: str) -> Dict[str, Union[bool, str]]:
        """Delete a dataset and all its associated objects."""
        pass
    
    @abstractmethod
    def find_datasets(self) -> List[str]:
        """Find all datasets for the current session."""
        pass
    
    @abstractmethod
    def get_dataset_info(
        self,
        dataset_ids: Optional[List[str]],
        return_top_n: int = 0,
        return_attributes: Optional[List[str]] = None,
        metadata_only: bool = False,
        return_content: bool = False,
    ) -> Dict[str, Union[bool, str, list]]:
        """Get information about one or more datasets.

        Args:
            dataset_ids: List of full dataset IDs to retrieve. Pass None to return
                all session datasets. IDs must be fully-qualified (e.g. 'session#name'
                for session-scoped or bare name for permanent). Use
                :meth:`identify_dataset_id` to resolve a name to its ID first.
            return_top_n: Maximum number of top nodes to return (0 = none, positive = limit).
            return_attributes: Optional list of node attributes to include in top_nodes.
            metadata_only: When True, skips the object-count and top-nodes queries and returns
                only the dataset header fields (id, name, description, metadata, subset_of,
                readonly).
                Use when you only need the dataset descriptor and do not need fragment data.
            return_content: When True and return_top_n > 0, enriches each node in top_nodes
                with readable content — meaningful only for :Document, :Chapter, and :Paragraph
                nodes associated with the dataset:
                - :Chapter nodes: adds ``chapter_id`` and ``chapter_text`` from the node's
                  ``id`` and ``text`` properties.
                - :Paragraph nodes: adds ``paragraph_id`` and ``paragraph_text`` from the
                  node's ``id`` and ``text`` properties.
                - :Document, :Chapter, :Paragraph nodes: adds ``document_summary`` from the
                  :Summary node connected to the parent :Document. Duplicates are suppressed —
                  if multiple nodes share the same document, the summary is included only once.
        """
        pass

    @abstractmethod
    def get_datasets(
        self,
        datasets: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Return dataset dictionaries with id, name, description, domain, metadata, objects_count, and readonly keys."""
        pass
    
    @abstractmethod
    def dataset_union(
        self,
        description_result: str,
        dataset_result: str,
        dataset_names: List[str],
        metadata: Dict[str, Union[str, int]]
    ) -> Dict[str, Union[bool, str, list]]:
        """Perform set union operation on multiple datasets."""
        pass
    
    @abstractmethod
    def dataset_intersection(
        self,
        description_result: str,
        dataset_result: str,
        dataset_names: List[str],
        metadata: Dict[str, Union[str, int]]
    ) -> Dict[str, Union[bool, str, list]]:
        """Perform set intersection operation on multiple datasets."""
        pass
    
    @abstractmethod
    def dataset_difference(
        self,
        description_result: str,
        dataset_result: str,
        dataset_names: List[str],
        metadata: Dict[str, Union[str, int]]
    ) -> Dict[str, Union[bool, str, list]]:
        """Perform set difference operation (dataset[0] - dataset[1])."""
        pass
    
    @abstractmethod
    def dataset_inclusion(
        self,
        dataset_names: List[str]
    ) -> Dict[str, Union[bool, str]]:
        """Check if second dataset is included in the first."""
        pass
    
    @abstractmethod
    def dataset_comparison(
        self,
        dataset_names: List[str]
    ) -> Dict[str, Union[bool, str]]:
        """Check if two datasets contain exactly the same elements."""
        pass

    @abstractmethod
    def duplicate_dataset(
        self,
        source_dataset_id: str,
        new_dataset_name: Optional[str] = None,
        new_dataset_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        subset_of_source: bool = True,
        minimum_ranking: Optional[float] = None,
        invisible: Optional[bool] = None,
        duplicate_hidden_relationship_statuses: bool = False,
        set_relationship_hidden: Optional[bool] = None,
    ) -> Dict[str, Union[bool, str, int]]:
        """Create a new dataset that is a server-side copy of source_dataset_id.

        All AGGREGATES relationships are duplicated in a single Cypher query —
        no fragment data is transferred over the network. This single method covers
        both "full duplication" (reranker fast-path) and "filtered clone" (assistant)
        use cases.

        Args:
            source_dataset_id: The source dataset ID (with or without session prefix).
            new_dataset_name: Name for the new dataset. Auto-generated if None.
            new_dataset_id: Full ID for the new dataset. Auto-generated if None.
            metadata: Optional metadata dict for the new dataset.
            subset_of_source: When True, creates a SUBSET_OF relationship to the source.
            minimum_ranking: When provided, only relationships with weight >= this value are copied.
            invisible: Passed through to the new Dataset node's `invisible` property.
            duplicate_hidden_relationship_statuses: When True, copies the hidden property from source relationships.
                When False (default), the hidden property is not copied (relationships will have no hidden property).
            set_relationship_hidden: If provided (True/False), explicitly sets the hidden property on all cloned relationships.
                This takes precedence over duplicate_hidden_relationship_statuses.

        Returns:
            Dict with 'dataset_name', 'dataset_id', 'object_count' on success,
            or 'error' / 'error_message' on failure.
        """
        pass

    @abstractmethod
    def set_dataset_relationships_hidden(
        self,
        dataset_id: str,
        node_ids: List[str],
        hidden: Optional[bool],
    ) -> Dict[str, Union[bool, str, int]]:
        """Set or clear only the 'hidden' property on AGGREGATES relationships from
        dataset_id to nodes whose 'id' property is in node_ids.

        Args:
            dataset_id: The dataset ID (with or without session prefix).
            node_ids: List of target node id values to match.
            hidden: True marks the matching relationships hidden; None removes the 'hidden'
                property (relationship becomes visible again).

        Returns:
            Dict with 'success' and 'updated_count', or 'error' / 'error_message' on failure.
        """
        pass

    @abstractmethod
    def hide_dataset_relationships_except(
        self,
        dataset_id: str,
        keep_node_ids: List[str],
    ) -> Dict[str, Union[bool, str, int]]:
        """Hide all AGGREGATES relationships in dataset_id whose target node id is NOT in
        keep_node_ids, and ensure relationships to keep_node_ids remain visible (hidden cleared).

        Returns:
            Dict with 'success', 'hidden_count', 'visible_count', or 'error' / 'error_message'.
        """
        pass

    @abstractmethod
    def set_relationship_focus(
        self,
        dataset_id: str,
        target: str,
        attribute: str,
        values: List[str],
    ) -> Dict[str, Union[bool, str, int]]:
        """Set a focus attribute on the dataset relationship to one target node."""
        pass

    @abstractmethod
    def add_dataset_elements(
        self,
        dataset_id: str,
        records: List[Dict[str, Any]],
        dataset_name: Optional[str] = None,
        dataset_description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        permanent: bool = False,
    ) -> Dict[str, Union[bool, str, int]]:
        """Add new DatasetObject nodes to a dataset without deleting existing ones.

        Creates the Dataset node (and session link) if it does not yet exist.
        Each record may carry an 'id' field; if absent a unique ID is generated.

        Returns:
            Dict with 'success', 'added_count', 'dataset_id', or 'error'/'error_message'.
        """
        pass

    @abstractmethod
    def edit_dataset_element(
        self,
        dataset_id: str,
        record: Dict[str, Any],
        replace_all: bool = False,
    ) -> Dict[str, Union[bool, str]]:
        """Update properties of an existing DatasetObject node within a dataset.

        The record must contain an 'id' field matching an existing node.

        Args:
            replace_all: If True, replaces ALL existing properties with those in record
                         (the 'id' field is always preserved).
                         If False (default), merges: adds/overwrites only supplied keys.

        Returns:
            Dict with 'success', or 'error'/'error_message'.
        """
        pass

    @abstractmethod
    def delete_dataset_elements(
        self,
        dataset_id: str,
        element_ids: List[str],
    ) -> Dict[str, Union[bool, str, int]]:
        """Remove specific DatasetObject nodes from a dataset by their 'id' property.

        Returns:
            Dict with 'success', 'deleted_count', or 'error'/'error_message'.
        """
        pass

    @abstractmethod
    def build_dataset_id(self, dataset_name: str, permanent: bool = False) -> str:
        """Build the full dataset ID.

        Args:
            dataset_name: The short name of the dataset (without session prefix).
            permanent: If False (default), prepends the current session ID:
                       '<session_id>#<dataset_name>'.
                       If True, returns the bare dataset name unchanged.

        Returns:
            The full dataset ID string.
        """
        pass

    @abstractmethod
    def is_dataset_permanent(self, dataset_name: Optional[str] = None, dataset_id: Optional[str] = None) -> bool:
        """Return True if the dataset is permanent (not session-scoped).

        Exactly one of *dataset_name* or *dataset_id* must be provided.
        If *dataset_name* is given, resolves it to its full id first via
        :meth:`identify_dataset_id`. A dataset is permanent when its full id
        contains no ``#`` separator (permanent ids are bare names; session-scoped
        ids have the form ``session_id#name``). Communities are permanent Datasets
        and therefore use bare IDs as well; not every permanent Dataset is a Community.
        """
        pass

    @abstractmethod
    def set_dataset_permanent(self, dataset_id: str, permanent: bool) -> Dict[str, Any]:
        """Change the permanency of an existing dataset.

        Renames the dataset's id according to :meth:`build_dataset_id` (using the
        bare dataset name extracted from *dataset_id*) and then connects or
        disconnects the Dataset node from the current Session node:

        - ``permanent=True``:  detach from Session (bare name id, no HAS_DATASET link).
        - ``permanent=False``: attach to Session   (session-scoped id, HAS_DATASET link).

        Returns a dict with keys ``success``, ``old_id``, ``new_id`` on success, or
        ``error`` / ``error_message`` on failure.
        """
        pass

    @abstractmethod
    def identify_dataset_id(self, dataset_name: str) -> str:
        """Resolve a dataset name to its full dataset ID for an existing dataset.

        Lookup order:
        1. Session-scoped:  '<session_id>#<dataset_name>'  — tried first.
        2. Permanent:       '<dataset_name>'               — fallback if not found.

        Use this method when reading/updating/deleting an existing dataset whose
        name was supplied by the caller (e.g. from a skill argument).  Use
        :meth:`build_dataset_id` when *creating* a new dataset.

        Returns:
            The full dataset ID of the first matching dataset, or the bare
            ``dataset_name`` if no matching dataset exists.
        """
        pass

    @abstractmethod
    def identify_dataset_ids(self, dataset_names: List[str]) -> List[str]:
        """Resolve a list of dataset names to their full dataset IDs in a single Neo4j query.

        For each name, lookup order is:
        1. Session-scoped:  '<session_id>#<dataset_name>'  — tried first.
        2. Permanent:       '<dataset_name>'               — fallback if not found in Neo4j.

        Use this instead of calling :meth:`identify_dataset_id` in a loop to avoid
        N round-trips to Neo4j.

        Returns:
            A list of fully-qualified dataset IDs in the same order as ``dataset_names``.
        """
        pass

    @abstractmethod
    def delete_session_datasets(self, sessionid: str) -> None:
        """Delete all datasets and their associated objects for the given session."""
        pass