"""
Dataset Service Module

This module provides functionality for managing datasets in the Neo4j database,
including creation, deletion, retrieval, and set operations (union, intersection, difference).
"""

from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)


import json
import traceback
from typing import Any, List, Dict, Tuple, Optional, Union
from src.interfaces.dataset_interface import DatasetInterface
from src.interfaces.documents_interface import DocumentsInterface
from src.services.user_service import UserInterface
from src.services.user_service import get_user_service
from src.services.dataset_exception import DatasetReadOnlyError
from src.core.base.InputOutput import execute_neo4j_query, get_parameters
from src.utils.dataset_helpers import parse_reference_uri
from src.services.ontology_source_helper import parse_kg_source_token
from src.utils.log import log, DEBUG, PERF, INFO, WARNING, ERROR, FATAL


# Configuration
MAX_RETURNED_DOCUMENT_IDS = 10


class DatasetService(DatasetInterface):
    """Service for dataset operations in Neo4j database."""

    def _ensure_dataset_writable(self, dataset_id: str, allow_missing: bool = False) -> None:
        """Raise when an existing dataset is marked read-only."""
        records = execute_neo4j_query(
            "MATCH (d:Dataset {id: $dataset_id}) RETURN d.readonly AS readonly",
            {"dataset_id": dataset_id},
        )
        if records:
            if bool(records[0].get("readonly") or False):
                raise DatasetReadOnlyError(
                    f"Dataset '{dataset_id}' is read-only and cannot be modified."
                )
        elif not allow_missing:
            return

    @staticmethod
    def _strip_dataset_uri(dataset_reference: str) -> str:
        """Remove the optional ds:// URI prefix from a Dataset reference."""
        if dataset_reference.startswith("ds://"):
            return dataset_reference[len("ds://"):]
        return dataset_reference
    
    def generate_unique_dataset_name(self, prefix: str = "dataset_") -> Tuple[str, int]:
        """Generate a unique dataset name with the given prefix."""
        existing_datasets = self.find_datasets()
        
        idx = 1
        while True:
            candidate_name = f"{prefix}{idx}"
            if candidate_name not in existing_datasets:
                return candidate_name, idx
            idx += 1

    def build_dataset_id(self, dataset_name: str, permanent: bool = False) -> str:
        """Build the full dataset ID.

        If permanent is False (default): '<session_id>#<dataset_name>'
        If permanent is True: '<dataset_name>'
        """
        dataset_name = self._strip_dataset_uri(dataset_name)
        if permanent:
            return dataset_name
        sessionid = get_user_service().get_session_id()
        return f"{sessionid}#{dataset_name}"

    def is_dataset_permanent(self, dataset_name: Optional[str] = None, dataset_id: Optional[str] = None) -> bool:
        """Return True if the dataset is permanent (not session-scoped).

        Exactly one of dataset_name or dataset_id must be provided.
        """
        if dataset_name is not None:
            dataset_id = self.identify_dataset_id(dataset_name)
        elif dataset_id is None:
            raise ValueError("is_dataset_permanent: either dataset_name or dataset_id must be provided")
        return "#" not in dataset_id

    def identify_dataset_id(self, dataset_name: str) -> str:
        """Resolve dataset_name to its full dataset ID for an existing dataset.

        Lookup order:
        1. Session-scoped: '<session_id>#<dataset_name>' — tried first.
        2. Permanent:      '<dataset_name>'              — fallback if not found.
        """
        dataset_name = self._strip_dataset_uri(dataset_name)
        session_id = get_user_service().get_session_id()
        candidate = f"{session_id}#{dataset_name}"
        records = execute_neo4j_query(
            "MATCH (d:Dataset {id: $id}) RETURN d.id LIMIT 1",
            {"id": candidate}
        )
        if records:
            return candidate
        
        # Not found as a session dataset — assume it is a permanent dataset
        return dataset_name

    def identify_dataset_ids(self, dataset_names: List[str]) -> List[str]:
        """Resolve a list of dataset names to their full dataset IDs in a single Neo4j query.

        Lookup order per name:
        1. Session-scoped: '<session_id>#<dataset_name>' — tried first.
        2. Permanent:      '<dataset_name>'              — fallback if not found in Neo4j.
        """
        if not dataset_names:
            return []
        
        dataset_names = [self._strip_dataset_uri(name) for name in dataset_names]
        session_id = get_user_service().get_session_id()
        # Build all session-scoped candidates up-front
        candidates = [f"{session_id}#{name}" for name in dataset_names]
        # Single query: find which session-scoped candidates actually exist
        found = execute_neo4j_query(
            "MATCH (d:Dataset) WHERE d.id IN $candidates RETURN d.id",
            {"candidates": candidates}
        )
        found_ids = {row[0] for row in found} if found else set()
        # For each name: use session-scoped ID if it exists, else fall back to bare name
        return [
            candidate if candidate in found_ids else name
            for name, candidate in zip(dataset_names, candidates)
        ]

    def contains_kg_entities(self, dataset_name: str) -> bool:
        """Return whether the named dataset contains at least one Entity node."""
        result: bool = False

        try:
            dataset_id: str = self.identify_dataset_id(dataset_name)
            records = execute_neo4j_query(
                """
                MATCH (d:Dataset {id: $dataset_id})-[:AGGREGATES]->(:Entity)
                RETURN count(*) > 0 AS contains_kg_entities
                """,
                {"dataset_id": dataset_id}
            )
            if records:
                result = bool(records[0].get("contains_kg_entities", False))
        except Exception as e:
            log(ERROR, f"DatasetService.contains_kg_entities error: {e}")

        return result

    def create_dataset(
        self,
        records: List[Union[Dict[str, Any], Any]],
        dataset_id: Optional[str] = None,
        userid: str = None,
        dataset_name: str = None,
        dataset_description: str = None,
        metadata: Dict[str, Any] = None,
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
        # Validate subset_of_datasetids parameter
        if subset_of_datasetids is not None:
            if not isinstance(subset_of_datasetids, list):
                log(ERROR, f"DatasetService.create_dataset(): subset_of_datasetids must be a list of strings or None, got {type(subset_of_datasetids).__name__}")
                subset_of_datasetids = None
            elif len(subset_of_datasetids) > 0:
                # Check if any element is a dict (incorrect format)
                if any(isinstance(item, dict) for item in subset_of_datasetids):
                    log(ERROR, f"DatasetService.create_dataset(): subset_of_datasetids contains dictionaries instead of strings. Extracting 'id' field from each dict.")
                    # Auto-fix: extract 'id' from each dict
                    subset_of_datasetids = [item.get('id') if isinstance(item, dict) else item for item in subset_of_datasetids]
                    # Filter out None values
                    subset_of_datasetids = [item for item in subset_of_datasetids if item is not None]
                # Check if all remaining elements are strings
                if not all(isinstance(item, str) for item in subset_of_datasetids):
                    log(ERROR, f"DatasetService.create_dataset(): subset_of_datasetids must contain only strings, got mixed types: {[type(item).__name__ for item in subset_of_datasetids]}")
                    # Filter to keep only strings
                    subset_of_datasetids = [item for item in subset_of_datasetids if isinstance(item, str)]
        
        # ensure dataset id exists: if not provided, build from sessionid and dataset_name
        sessionid = get_user_service().get_session_id()
        if userid is None:
            userid = get_user_service().get_user_uuid()
        if dataset_name is None:
            dataset_name, _ = self.generate_unique_dataset_name(prefix="dataset_")
        if dataset_id is None:
            dataset_id = self.build_dataset_id(dataset_name, permanent=permanent)

        # convert the metadata dictionary to a JSON string
        metadata_json = json.dumps(metadata) if metadata is not None else None
        objects_count = len(records) if records is not None else 0

        if permanent:
            # Permanent dataset: create/update the Dataset node without linking to any Session
            dataset_result = execute_neo4j_query("""
                MERGE (d:Dataset {id: $dataset_id})
                SET
                    d.name = $dataset_name,
                    d.description = $dataset_description,
                    d.metadata = $metadata,
                    d.objects_count = $objects_count,
                    d.created = timestamp(),
                    d.domain = $domain,
                    d.invisible = $invisible
                RETURN d, labels(d) AS dataset_labels
            """, parameters={
                "dataset_name": dataset_name,
                "dataset_id": dataset_id,
                "dataset_description": dataset_description,
                "metadata": metadata_json,
                "objects_count": objects_count,
                "domain": domain,
                "invisible": invisible
            })

            if not dataset_result:
                error_message = f"DatasetService.create_dataset(): Failed to create permanent dataset {dataset_id}"
                log(ERROR, error_message)
                return {"error": 3, "error_message": error_message}

            log(DEBUG, f"DatasetService.create_dataset(): Permanent dataset {dataset_id} created (not linked to session)")
        else:
            # Session-scoped dataset: ensure Session exists, then link to it
            session_result = execute_neo4j_query(
                "MATCH (s:Session {id: $sessionid}) RETURN s",
                {"sessionid": sessionid}
            )
            if not session_result:
                user_result = execute_neo4j_query(
                    "MATCH (u:User:SystemAcl {id: $userid}) RETURN u.id",
                    {"userid": userid}
                )
                if not user_result:
                    log(WARNING, f"DatasetService.create_dataset(): User {userid} not found: creating it implicitly.")
                    execute_neo4j_query("""
                        MERGE (u:User {id: $userid, username: $userid})
                        SET u:SystemAcl
                    """, {"userid": userid})

                log(INFO, f"DatasetService.create_dataset(): Session {sessionid} not found: creating it implicitly.")
                execute_neo4j_query("""
                    MATCH (u:User:SystemAcl {id: $userid})
                    MERGE (u)-[:HAS_SESSION]->(s:Session {id: $sessionid})
                """, {
                    "sessionid": sessionid,
                    "userid": userid}
                )

            dataset_result = execute_neo4j_query("""
                MATCH (s:Session {id: $sessionid})
                SET s.userid = $userid
                MERGE (d:Dataset {id: $dataset_id})
                SET
                    d.name = $dataset_name,
                    d.description = $dataset_description,
                    d.metadata = $metadata,
                    d.objects_count = $objects_count,
                    d.created = timestamp(),
                    d.domain = $domain,
                    d.invisible = $invisible
                MERGE (s)-[:HAS_DATASET]->(d)
                RETURN d, s
            """, parameters={
                "sessionid": sessionid,
                "userid": userid,
                "dataset_name": dataset_name,
                "dataset_id": dataset_id,
                "dataset_description": dataset_description,
                "metadata": metadata_json,
                "objects_count": objects_count,
                "domain": domain,
                "invisible": invisible
            })

            if not dataset_result:
                error_message = f"DatasetService.create_dataset(): Failed to create dataset or associate with session {sessionid}"
                log(ERROR, error_message)
                return {"error": 3, "error_message": error_message}

            log(DEBUG, f"DatasetService.create_dataset(): Dataset {dataset_id} associated with session {sessionid}")

        # delete all nodes eventually associated with the dataset, only if they are labeled DatasetObject
        log(PERF, f"create_dataset: starting cleanup for {dataset_id}", perf_id="cd_cleanup_start")
        execute_neo4j_query("""
            MATCH (d:Dataset {id: $dataset_id})-[:AGGREGATES]->(n:DatasetObject)
            DETACH DELETE n
        """, {"dataset_id": dataset_id})
        log(PERF, f"create_dataset: DatasetObject cleanup done", perf_id="cd_del1", perf_since_id="cd_cleanup_start")

        # de-associate all nodes from the dataset, except the session node
        execute_neo4j_query("""
            MATCH (d:Dataset {id: $dataset_id})-[r:AGGREGATES]->(n)
            WHERE NOT n:Session AND NOT n:Dataset
            DELETE r
        """, {"dataset_id": dataset_id})
        log(PERF, f"create_dataset: AGGREGATES cleanup done", perf_id="cd_del2", perf_since_id="cd_del1")

        # Connect the dataset node to the objects returned by the query
        document_ids = []
        if records is not None and len(records) > 0:
            # Determine node_ids and document_ids from records.
            # Three supported input formats:
            #   A) Lightweight dicts with 'eid' key (from get_nodes_from_fragments) — no embedding transferred
            #   B) Legacy Neo4j Node objects (record[0] is a Node)
            #   C) Arbitrary dict records — create new DatasetObject nodes
            node_ids = None  # None → fall through to DatasetObject path (Format C)
            document_ids_set = set()

            first = records[0]
            if hasattr(first, 'keys') and 'eid' in first.keys():
                # Format A: lightweight records {eid, id, documentid, labels}
                node_ids = []
                for record in records:
                    node_ids.append(record['eid'])
                    node_labels = record.get('labels') or []
                    if 'Document' in node_labels:
                        doc_id = record.get('id') or record.get('documentid')
                    elif 'Chapter' in node_labels or 'Paragraph' in node_labels:
                        doc_id = record.get('documentid')
                    else:
                        doc_id = None
                    if doc_id:
                        document_ids_set.add(doc_id)

            elif len(first) > 0 and not isinstance(first, dict) and first[0].__class__.__name__ == 'Node':
                # Format B: legacy Neo4j Node objects
                node_ids = []
                for record in records:
                    node = record[0]
                    node_ids.append(node.element_id)
                    if 'Document' in node.labels:
                        doc_id = node.get('id') or node.get('documentid')
                    elif 'Chapter' in node.labels or 'Paragraph' in node.labels:
                        doc_id = node.get('documentid')
                    else:
                        doc_id = None
                    if doc_id:
                        document_ids_set.add(doc_id)

            if node_ids is not None:
                # Formats A & B: link existing graph nodes to the dataset by element ID
                document_ids = list(document_ids_set)
                subset_ids = subset_of_datasetids if subset_of_datasetids else []

                log(DEBUG, f"DatasetService.create_dataset: Linking {len(node_ids)} nodes to dataset {dataset_id}")
                log(DEBUG, f"DatasetService.create_dataset: subset_of_datasetids = {subset_ids}")
                log(DEBUG, f"DatasetService.create_dataset: First few node element IDs: {node_ids[:3] if len(node_ids) > 3 else node_ids}")

                if weights is None:
                    link_result = execute_neo4j_query("""
                        MATCH (d:Dataset {id: $dataset_id})
                        WITH d
                        MATCH (n)
                        WHERE elementId(n) in $nodes
                          AND (
                            size($subset_of_datasetids) = 0 
                            OR EXISTS { 
                              MATCH (n)<-[:AGGREGATES]-(parent:Dataset) 
                              WHERE parent.id IN $subset_of_datasetids 
                            }
                          )
                        MERGE (d)-[:AGGREGATES]->(n)
                    """, {
                        "dataset_id": dataset_id,
                        "nodes": node_ids,
                        "subset_of_datasetids": subset_ids
                    })
                else:
                    link_result = execute_neo4j_query("""
                        MATCH (d:Dataset {id: $dataset_id})
                        WITH d
                        MATCH (n)
                        WHERE elementId(n) in $nodes
                          AND (
                            size($subset_of_datasetids) = 0 
                            OR EXISTS { 
                              MATCH (n)<-[:AGGREGATES]-(parent:Dataset) 
                              WHERE parent.id IN $subset_of_datasetids 
                            }
                          )
                        MERGE (d)-[r:AGGREGATES]->(n)
                        SET r.weight = $weights[n.id]
                    """, {
                        "dataset_id": dataset_id,
                        "nodes": node_ids,
                        "weights": weights,
                        "subset_of_datasetids": subset_ids
                    })

                if link_result is None:
                    error_message = "DatasetService: The dataset object linking failed."
                    return {"error": 1, "error_message": error_message, "error_data": link_result}

                log(DEBUG, f"DatasetService.create_dataset: Link query executed, link_result type: {type(link_result)}")

                if subset_ids and len(subset_ids) > 0:
                    for parent_id in subset_ids:
                        execute_neo4j_query("""
                            MATCH (parent:Dataset {id: $parent_id})
                            MATCH (child:Dataset {id: $child_id})
                            MERGE (child)-[:SUBSET_OF]->(parent)
                        """, {
                            "parent_id": parent_id,
                            "child_id": dataset_id
                        })
                        log(DEBUG, f"DatasetService.create_dataset: Created SUBSET_OF relationship: {dataset_id} -> {parent_id}")

            else:
                # Format C: arbitrary dict records — create new DatasetObject nodes
                idx = 0
                for record in records:
                    node_properties = {}
                    for key in record.keys():
                        if not key or '\x00' in key:
                            log(WARNING, f"DatasetService.create_dataset: skipping invalid property key {repr(key)}")
                            continue
                        value = record[key]
                        node_properties[key] = value

                    if 'labels' in node_properties and 'Document' in node_properties.get('labels', []):
                        doc_id = node_properties.get('id') or node_properties.get('documentid')
                        if doc_id:
                            document_ids.append(doc_id)

                    do_id = str(idx)
                    link_result = execute_neo4j_query("""
                        CREATE (o:DatasetObject {id: $do_id})
                        SET o += $properties
                        WITH o
                        MATCH (d:Dataset {id: $dataset_id})
                        WITH d,o
                        MERGE (d)-[:AGGREGATES]->(o)
                        RETURN o
                    """,
                    {
                        "dataset_id": dataset_id,
                        "do_id": do_id,
                        "properties": node_properties
                    })

                    if link_result is None:
                        error_message = "DatasetService: The dataset object linking failed."
                        log(ERROR, error_message)
                        return {"error": 2, "error_message": error_message, "error_data": link_result}

                    idx += 1

        result = {
            "dataset_name": dataset_name,
            "dataset_id": dataset_id,
            "object_count": objects_count
        }
        
        # Add document_ids as a comma-separated string if any documents were found
        if document_ids:
            if len(document_ids) > MAX_RETURNED_DOCUMENT_IDS:
                result["warning_document_ids_too_many"] = f"Too many document IDs to display: {len(document_ids)}."
            else:
                result["document_ids"] = document_ids
        
        return result
    
    def _link_skill_items_to_dataset(self, dataset_id: str, skill_items: List[Dict]) -> int:
        """Materialize skill:// fragments (skill://SkillName/InstanceId) as :DatasetObject:Skill nodes
        (skill, input, output from get_parameters()) and AGGREGATES them into the dataset.
        The dual label lets the C# side reconstruct a skill:// URI instead of the generic do:// form."""
        linked_count = 0

        if skill_items:
            nodes: List[Dict] = []
            for item in skill_items:
                skill_uri = item["id"]
                skill_name, _, instance_id = skill_uri[len("skill://"):].partition("/")
                if not skill_name or not instance_id:
                    log(WARNING, f"_link_skill_items_to_dataset: malformed skill URI '{skill_uri}', skipping.")
                    continue
                params = get_parameters(skill_name, instance_id, default_value={}) or {}
                nodes.append({
                    "id": f"{skill_name}/{instance_id}",  # matches the path expected after "skill://"
                    "skill": skill_uri,
                    "input": json.dumps(params.get("args")),
                    "output": json.dumps(params.get("output")),
                    "weight": item["weight"],
                })

            if nodes:
                execute_neo4j_query("""
                    MATCH (d:Dataset {id: $dataset_id})
                    WITH d
                    UNWIND $items AS item
                    MERGE (o:DatasetObject:Skill {id: item.id})
                    SET o.skill = item.skill, o.input = item.input, o.output = item.output
                    MERGE (d)-[r:AGGREGATES]->(o)
                    SET r.weight = item.weight
                """, {"dataset_id": dataset_id, "items": nodes})
                linked_count = len(nodes)
                log(PERF, f"_link_skill_items_to_dataset: linked {linked_count} DatasetObject:Skill node(s)", perf_id="cdf_skills")

        return linked_count

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
        # Validate subset_of_datasetids parameter
        if subset_of_datasetids is not None:
            if not isinstance(subset_of_datasetids, list):
                log(ERROR, f"DatasetService.create_dataset_from_fragments(): subset_of_datasetids must be a list of strings or None, got {type(subset_of_datasetids).__name__}")
                subset_of_datasetids = None
            elif len(subset_of_datasetids) > 0:
                # Check if any element is a dict (incorrect format)
                if any(isinstance(item, dict) for item in subset_of_datasetids):
                    log(ERROR, f"DatasetService.create_dataset_from_fragments(): subset_of_datasetids contains dictionaries instead of strings. Extracting 'id' field from each dict.")
                    # Auto-fix: extract 'id' from each dict
                    subset_of_datasetids = [item.get('id') if isinstance(item, dict) else item for item in subset_of_datasetids]
                    # Filter out None values
                    subset_of_datasetids = [item for item in subset_of_datasetids if item is not None]
                # Check if all remaining elements are strings
                if not all(isinstance(item, str) for item in subset_of_datasetids):
                    log(ERROR, f"DatasetService.create_dataset_from_fragments(): subset_of_datasetids must contain only strings, got mixed types: {[type(item).__name__ for item in subset_of_datasetids]}")
                    # Filter to keep only strings
                    subset_of_datasetids = [item for item in subset_of_datasetids if isinstance(item, str)]
        
        try:
            if not fragments:
                return {"success": False, "error": 1, "error_message": "create_dataset_from_fragments(): No fragments provided."}

            log(PERF, f"create_dataset_from_fragments: start, {len(fragments)} frags", perf_id="cdf_start")

            # Step 1: Set up Session, Dataset node and clean old associations.
            # Pass records=[] to skip the old elementId-based linking path.
            setup_result = self.create_dataset(
                records=[],
                dataset_id=dataset_id,
                userid=userid,
                dataset_name=dataset_name,
                dataset_description=dataset_description,
                metadata=metadata,
                domain=domain,
                subset_of_datasetids=None,  # SUBSET_OF relationships handled in step 5
                weights=None,
                invisible=invisible,
            )
            if 'error' in setup_result:
                return setup_result

            log(PERF, f"create_dataset_from_fragments: setup done", perf_id="cdf_setup", perf_since_id="cdf_start")

            actual_dataset_id = setup_result['dataset_id']
            actual_dataset_name = setup_result['dataset_name']
            subset_ids = subset_of_datasetids if subset_of_datasetids else []

            # Step 2: Classify fragment IDs by type in Python — no Neo4j lookup needed.
            chapter_items: List[Dict] = []
            paragraph_items: List[Dict] = []
            doc_items: List[Dict] = []
            kg_items: List[Dict] = []
            skill_items: List[Dict] = []
            document_ids_set: set = set()

            for fid in fragments:
                w = weights.get(fid, 1.0) if weights else 1.0
                if str(fid).lower().startswith("skill://"):
                    skill_items.append({"id": str(fid).split("?", 1)[0], "weight": w})
                    continue
                if str(fid).lower().startswith("kg://"):
                    kg_id = parse_kg_source_token(str(fid))
                    if kg_id:
                        kg_items.append({"id": kg_id, "weight": w})
                    continue
                parsed = DocumentsInterface.parse_fragment_id(fid)
                t = parsed.get("type", "unknown")
                bare = parsed.get("bare", fid)
                if t == "chapter" or t == "summary":
                    chapter_items.append({"id": bare, "weight": w})
                    document_ids_set.add(parsed.get("document_id") or bare)
                elif t == "paragraph":
                    paragraph_items.append({"id": bare, "weight": w})
                    document_ids_set.add(parsed.get("document_id") or bare)
                elif t == "document":
                    doc_items.append({"id": bare, "weight": w})
                    document_ids_set.add(bare)

            # Deduplicate items by bare id, keeping the highest weight for each.
            # Prevents duplicate AGGREGATES relationships when the same fragment id
            # appears more than once in the input (e.g. from AND-combined category queries).
            def _dedup_items(items: list) -> list:
                seen: dict = {}
                for item in items:
                    iid = item['id']
                    if iid not in seen or item['weight'] > seen[iid]['weight']:
                        seen[iid] = item
                return list(seen.values())
            chapter_items = _dedup_items(chapter_items)
            paragraph_items = _dedup_items(paragraph_items)
            doc_items = _dedup_items(doc_items)
            kg_items = _dedup_items(kg_items)
            skill_items = _dedup_items(skill_items)

            log(PERF, f"create_dataset_from_fragments: classify done, chapters={len(chapter_items)}, paras={len(paragraph_items)}, docs={len(doc_items)}, kg={len(kg_items)}, skills={len(skill_items)}", perf_id="cdf_classify", perf_since_id="cdf_setup")

            # Step 3: Link fragments directly using label-specific UNWIND + CREATE.
            # CREATE is safe because create_dataset() already cleaned up all old
            # AGGREGATES relationships. Label hints let Neo4j use the schema index
            # on Chapter/Paragraph/Document.id, avoiding an extra elementId round-trip.
            #
            # IMPORTANT: Do NOT include EXISTS subqueries inside the write transactions.
            # An EXISTS check inside UNWIND+CREATE causes O(N²) traversal of the
            # in-transaction relationship state, turning 200 items into ~54s instead of ~1s.
            # When subset filtering is needed, pre-filter with a separate READ query first.
            if subset_ids:
                # Pre-filter: keep only fragments already associated with a parent dataset.
                # This is a read-only query and avoids the O(N²) in-transaction EXISTS problem.
                if chapter_items:
                    rows = execute_neo4j_query("""
                        UNWIND $ids AS fid
                        MATCH (c:Chapter {id: fid})
                        WHERE EXISTS {
                            MATCH (c)<-[:AGGREGATES]-(parent:Dataset)
                            WHERE parent.id IN $subset_ids
                        }
                        RETURN fid
                    """, {"ids": [i["id"] for i in chapter_items], "subset_ids": subset_ids})
                    elg = {row["fid"] for row in (rows or [])}
                    chapter_items = [i for i in chapter_items if i["id"] in elg]
                if paragraph_items:
                    rows = execute_neo4j_query("""
                        UNWIND $ids AS fid
                        MATCH (p:Paragraph {id: fid})
                        WHERE EXISTS {
                            MATCH (p)<-[:AGGREGATES]-(parent:Dataset)
                            WHERE parent.id IN $subset_ids
                        }
                        RETURN fid
                    """, {"ids": [i["id"] for i in paragraph_items], "subset_ids": subset_ids})
                    elg = {row["fid"] for row in (rows or [])}
                    paragraph_items = [i for i in paragraph_items if i["id"] in elg]
                if doc_items:
                    rows = execute_neo4j_query("""
                        UNWIND $ids AS fid
                        MATCH (doc:Document {id: fid})
                        WHERE EXISTS {
                            MATCH (doc)<-[:AGGREGATES]-(parent:Dataset)
                            WHERE parent.id IN $subset_ids
                        }
                        RETURN fid
                    """, {"ids": [i["id"] for i in doc_items], "subset_ids": subset_ids})
                    elg = {row["fid"] for row in (rows or [])}
                    doc_items = [i for i in doc_items if i["id"] in elg]
                if kg_items:
                    rows = execute_neo4j_query("""
                        UNWIND $ids AS fid
                        MATCH (n:Entity {id: fid})
                        WHERE EXISTS {
                            MATCH (n)<-[:AGGREGATES]-(parent:Dataset)
                            WHERE parent.id IN $subset_ids
                        }
                        RETURN fid
                    """, {"ids": [i["id"] for i in kg_items], "subset_ids": subset_ids})
                    elg = {row["fid"] for row in (rows or [])}
                    kg_items = [i for i in kg_items if i["id"] in elg]

            if chapter_items:
                log(DEBUG, f"create_dataset_from_fragments: attempting to link {len(chapter_items)} chapters: {[item.get('id') for item in chapter_items[:3]]}")
                execute_neo4j_query("""
                    MATCH (d:Dataset {id: $dataset_id})
                    WITH d
                    UNWIND $items AS item
                    MATCH (c:Chapter {id: item.id})
                    CREATE (d)-[r:AGGREGATES]->(c)
                    SET r.weight = item.weight
                """, {"dataset_id": actual_dataset_id, "items": chapter_items})
                log(PERF, f"create_dataset_from_fragments: chapters linked ({len(chapter_items)})", perf_id="cdf_chapters", perf_since_id="cdf_classify")

            if paragraph_items:
                log(DEBUG, f"create_dataset_from_fragments: attempting to link {len(paragraph_items)} paragraphs: {[item.get('id') for item in paragraph_items[:3]]}")
                execute_neo4j_query("""
                    MATCH (d:Dataset {id: $dataset_id})
                    WITH d
                    UNWIND $items AS item
                    MATCH (p:Paragraph {id: item.id})
                    CREATE (d)-[r:AGGREGATES]->(p)
                    SET r.weight = item.weight
                """, {"dataset_id": actual_dataset_id, "items": paragraph_items})
                log(PERF, f"create_dataset_from_fragments: paragraphs linked ({len(paragraph_items)})", perf_id="cdf_paragraphs", perf_since_id="cdf_classify")

            if doc_items:
                log(DEBUG, f"create_dataset_from_fragments: attempting to link {len(doc_items)} documents: {[item.get('id') for item in doc_items[:3]]}")
                execute_neo4j_query("""
                    MATCH (d:Dataset {id: $dataset_id})
                    WITH d
                    UNWIND $items AS item
                    MATCH (doc:Document {id: item.id})
                    CREATE (d)-[r:AGGREGATES]->(doc)
                    SET r.weight = item.weight
                """, {"dataset_id": actual_dataset_id, "items": doc_items})
                log(PERF, f"create_dataset_from_fragments: docs linked ({len(doc_items)})", perf_id="cdf_docs", perf_since_id="cdf_classify")

            if kg_items:
                log(DEBUG, f"create_dataset_from_fragments: attempting to link {len(kg_items)} KG nodes: {[item.get('id') for item in kg_items[:3]]}")
                link_result = execute_neo4j_query("""
                    MATCH (d:Dataset {id: $dataset_id})
                    WITH d
                    UNWIND $items AS item
                    MATCH (n:Entity {id: item.id})
                    CREATE (d)-[r:AGGREGATES]->(n)
                    SET r.weight = item.weight
                    RETURN count(n) AS linked_count
                """, {"dataset_id": actual_dataset_id, "items": kg_items})
                linked_count = link_result[0].get("linked_count", 0) if link_result else 0
                log(PERF, f"create_dataset_from_fragments: KG nodes linked ({linked_count}/{len(kg_items)})", perf_id="cdf_kg", perf_since_id="cdf_classify")

            skill_linked_count = self._link_skill_items_to_dataset(actual_dataset_id, skill_items)

            # Step 4: Fix objects_count (was 0 because records=[] was passed to create_dataset).
            # CRITICAL: When subset_ids is provided, pre-filtering reduces chapter_items, paragraph_items, and doc_items.
            # We must count the ACTUAL linked items, not the original len(fragments) count.
            actual_linked_count = len(chapter_items) + len(paragraph_items) + len(doc_items) + len(kg_items) + skill_linked_count
            execute_neo4j_query(
                "MATCH (d:Dataset {id: $dataset_id}) SET d.objects_count = $count",
                {"dataset_id": actual_dataset_id, "count": actual_linked_count}
            )

            # Step 5: Create SUBSET_OF relationships if needed.
            if subset_ids:
                for parent_id in subset_ids:
                    execute_neo4j_query("""
                        MATCH (parent:Dataset {id: $parent_id})
                        MATCH (child:Dataset {id: $child_id})
                        MERGE (child)-[:SUBSET_OF]->(parent)
                    """, {"parent_id": parent_id, "child_id": actual_dataset_id})

            document_ids = list(document_ids_set)
            result = {
                "dataset_name": actual_dataset_name,
                "dataset_id": actual_dataset_id,
                "object_count": actual_linked_count,  # Use actual linked count, not original len(fragments)
            }
            if document_ids:
                if len(document_ids) > MAX_RETURNED_DOCUMENT_IDS:
                    result["warning_document_ids_too_many"] = f"Too many document IDs to display: {len(document_ids)}."
                else:
                    result["document_ids"] = document_ids
            return result

        except Exception as e:
            error_message = f"create_dataset_from_fragments(): {e}"
            log(ERROR, f"DatasetService.create_dataset_from_fragments ERROR: {error_message}. Traceback:\n{traceback.format_exc()}")
            return {"success": False, "error": 2, "error_message": error_message}

    def upsert_dataset_fragments(
        self,
        dataset_id: str,
        fragments: List[str],
        weight: float = 1.0,
        hidden: Optional[bool] = None,
    ) -> Dict[str, Union[bool, str, int]]:
        """Add or update document and entity fragments without duplicate relationships.

        fragments may include skill:// URIs (skill://SkillName/InstanceId), which are
        materialized as :DatasetObject:Skill nodes (skill, input, output) via get_parameters().
        """
        result: Dict[str, Union[bool, str, int]] = {"success": True, "updated_count": 0}
        normalized_fragments: Dict[Tuple[str, str], None] = {}
        skill_items: List[Dict] = []

        for fragment in fragments or []:
            if str(fragment).lower().startswith("skill://"):
                skill_items.append({"id": str(fragment).split("?", 1)[0], "weight": weight})
                continue
            parsed = parse_reference_uri(fragment)
            if parsed is not None:
                normalized_fragments[parsed] = None

        if normalized_fragments or skill_items:
            dataset_id = self.identify_dataset_id(dataset_id)
            self._ensure_dataset_writable(dataset_id)

            for node_label, node_id in normalized_fragments:
                records = execute_neo4j_query(
                    f"""
                    MATCH (d:Dataset {{id: $dataset_id}})
                    MATCH (n:{node_label} {{id: $node_id}})
                    MERGE (d)-[r:AGGREGATES]->(n)
                    SET r.weight = $weight, r.hidden = $hidden
                    RETURN count(r) AS updated_count
                    """,
                    {
                        "dataset_id": dataset_id,
                        "node_id": node_id,
                        "weight": weight,
                        "hidden": hidden,
                    },
                )
                result["updated_count"] = int(result["updated_count"]) + (records[0].get("updated_count", 0) if records else 0)

            result["updated_count"] = int(result["updated_count"]) + self._link_skill_items_to_dataset(dataset_id, skill_items)

            execute_neo4j_query(
                """
                MATCH (d:Dataset {id: $dataset_id})
                OPTIONAL MATCH (d)-[:AGGREGATES]->(n)
                WITH d, count(n) AS object_count
                SET d.objects_count = object_count
                """,
                {"dataset_id": dataset_id},
            )

        return result

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
        sessionid = get_user_service().get_session_id()
        
        # Normalize dataset ID: add session prefix if not present
        if "#" not in dataset_id:
            dataset_id = f"{sessionid}#{dataset_id}"
        
        try:
            log(DEBUG, f"get_fragments_from_dataset: dataset_id={dataset_id}, include_weights={include_weights}")
            
            # Query for document fragments and KG nodes associated with the dataset.
            # Return explicit properties (no embedding) to avoid transferring 1536-float arrays.
            query = """
                MATCH (d:Dataset {id: $dataset_id})-[r:AGGREGATES]->(n)
                WHERE n:Document OR n:Chapter OR n:Paragraph OR NOT (n:Dataset)
                RETURN
                    n.id AS id,
                    n.documentid AS documentid,
                    n.text AS text,
                    n.title AS title,
                    n.name AS name,
                    n.status AS status,
                    n.type AS type,
                    n.enhanced_text AS enhanced_text,
                    n.enhanced_type AS enhanced_type,
                    labels(n) AS labels,
                    r.weight AS weight
            """
            
            results = execute_neo4j_query(query, {"dataset_id": dataset_id})
            log(DEBUG, f"get_fragments_from_dataset: query returned {len(results) if results else 0} results for dataset_id={dataset_id}")
            
            if not results:
                log(DEBUG, f"get_fragments_from_dataset: No fragments found for dataset {dataset_id}")
                return []
            
            fragments = []
            for idx, record in enumerate(results):
                node_id = record.get('id')
                if not node_id:
                    log(WARNING, f"get_fragments_from_dataset: fragment[{idx}] has no 'id' property, skipping. labels={record.get('labels')}")
                    continue
                
                is_document_fragment = any(label in (record.get('labels') or []) for label in ['Document', 'Chapter', 'Paragraph'])
                fragment_dict = {k: v for k, v in {
                    'id': node_id if is_document_fragment else f"kg://{node_id}",
                    'documentid': record.get('documentid'),
                    'text': record.get('text'),
                    'title': record.get('title'),
                    'name': record.get('name'),
                    'status': record.get('status'),
                    'type': record.get('type'),
                    'enhanced_text': record.get('enhanced_text'),
                    'enhanced_type': record.get('enhanced_type'),
                    'labels': record.get('labels', []),
                }.items() if v is not None}

                weight = record.get('weight')
                if include_weights and weight is not None:
                    fragment_dict['weight'] = weight
                    log(DEBUG, f"get_fragments_from_dataset: fragment[{idx}] id={node_id}, weight={weight}, labels={fragment_dict.get('labels')}")
                else:
                    log(DEBUG, f"get_fragments_from_dataset: fragment[{idx}] id={node_id}, weight=None, labels={fragment_dict.get('labels')}")

                fragments.append(fragment_dict)
            
            log(DEBUG, f"get_fragments_from_dataset: Retrieved {len(fragments)} fragments from dataset {dataset_id}")
            return fragments
            
        except Exception as e:
            error_message = f"get_fragments_from_dataset(): {e}"
            log(ERROR, f"DatasetService.get_fragments_from_dataset ERROR: {error_message}. Traceback:\n{traceback.format_exc()}")
            return []

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
        no fragment data is transferred over the network.

        Args:
            subset_of_source: When True, creates a SUBSET_OF relationship to the source.
            minimum_ranking: When provided, only relationships with weight >= this value are copied.
            invisible: Passed through to the new Dataset node's `invisible` property.
            duplicate_hidden_relationship_statuses: When True, copies the hidden property from source relationships.
                When False (default), the hidden property is not copied (relationships will have no hidden property).
            set_relationship_hidden: If provided (True/False), explicitly sets the hidden property on all cloned relationships.
                This takes precedence over duplicate_hidden_relationship_statuses.
        """
        source_dataset_id = self.identify_dataset_id(source_dataset_id)

        # Fetch source descriptor to copy description and domain
        source_info = execute_neo4j_query(
            "MATCH (d:Dataset {id: $id}) RETURN d.description AS description, d.domain AS domain",
            {"id": source_dataset_id}
        )
        if not source_info:
            error_msg = f"duplicate_dataset(): Source dataset '{source_dataset_id}' not found."
            log(ERROR, f"DatasetService.{error_msg}")
            return {"error": 1, "error_message": error_msg}

        source_description = source_info[0].get("description")
        source_domain = source_info[0].get("domain")

        if new_dataset_name is None:
            new_dataset_name, _ = self.generate_unique_dataset_name(prefix="dataset_copy_")
        if new_dataset_id is None:
            new_dataset_id = self.build_dataset_id(new_dataset_name)

        log(PERF, f"duplicate_dataset: start — cloning {source_dataset_id} → {new_dataset_id}", perf_id="dup_start")

        # Step 1: Create the new Dataset node (handles session setup and old-rel cleanup)
        setup_result = self.create_dataset(
            records=[],
            dataset_id=new_dataset_id,
            userid=get_user_service().get_user_uuid(),
            dataset_name=new_dataset_name,
            dataset_description=source_description,
            metadata=metadata,
            domain=source_domain,
            subset_of_datasetids=None,
            invisible=invisible,
        )
        if 'error' in setup_result:
            return setup_result
        log(PERF, f"duplicate_dataset: Dataset node created", perf_id="dup_node_done", perf_since_id="dup_start")

        # Step 2: Copy AGGREGATES relationships server-side in one Cypher query,
        # optionally filtered by minimum_ranking.
        # Handle the hidden property based on parameters:
        # - set_relationship_hidden (if provided): explicitly set to True/False
        # - duplicate_hidden_relationship_statuses=True: copy from source
        # - default (both False/None): don't copy hidden (relationships will have no hidden property)
        if set_relationship_hidden is not None:
            relationship_props = "weight: source_relationship.weight, hidden: $hidden"
            extra_params: Dict[str, Any] = {"hidden": set_relationship_hidden}
        elif duplicate_hidden_relationship_statuses:
            relationship_props = "weight: source_relationship.weight, hidden: source_relationship.hidden"
            extra_params = {}
        else:
            relationship_props = "weight: source_relationship.weight"
            extra_params = {}

        copy_records = execute_neo4j_query(
            f"""
            MATCH (source:Dataset {{id: $source_id}})-[source_relationship:AGGREGATES]->(node)
            WHERE $minimum_ranking IS NULL OR coalesce(source_relationship.weight, 0.0) >= $minimum_ranking
            MATCH (dest:Dataset {{id: $dest_id}})
            CREATE (dest)-[clone_relationship:AGGREGATES]->(node)
            SET clone_relationship = {{{relationship_props}}}
            RETURN count(clone_relationship) AS object_count
            """,
            {
                "source_id": source_dataset_id,
                "dest_id": new_dataset_id,
                "minimum_ranking": minimum_ranking,
                **extra_params,
            },
        )
        object_count = copy_records[0].get("object_count", 0) if copy_records else 0
        log(PERF, f"duplicate_dataset: AGGREGATES relationships copied", perf_id="dup_rel_done", perf_since_id="dup_node_done")

        # Step 3: Fix objects_count (was 0 because records=[] was passed to create_dataset)
        execute_neo4j_query(
            "MATCH (d:Dataset {id: $dataset_id}) SET d.objects_count = $count",
            {"dataset_id": new_dataset_id, "count": object_count}
        )

        # Step 4: Optionally create SUBSET_OF relationship to source
        if subset_of_source:
            execute_neo4j_query("""
                MATCH (src:Dataset {id: $source_id})
                MATCH (dst:Dataset {id: $dest_id})
                MERGE (dst)-[:SUBSET_OF]->(src)
            """, {"source_id": source_dataset_id, "dest_id": new_dataset_id})

        log(PERF, f"duplicate_dataset: done", perf_id="dup_done", perf_since_id="dup_rel_done")
        return {
            "dataset_name": new_dataset_name,
            "dataset_id": new_dataset_id,
            "object_count": object_count,
        }

    def set_dataset_relationships_hidden(
        self,
        dataset_id: str,
        node_ids: List[str],
        hidden: Optional[bool],
    ) -> Dict[str, Union[bool, str, int]]:
        """Set or clear only AGGREGATES.hidden for matching nodes in one dataset."""
        result: Dict[str, Union[bool, str, int]] = {"success": True, "updated_count": 0}

        if node_ids:
            dataset_id = self.identify_dataset_id(dataset_id)
            self._ensure_dataset_writable(dataset_id)
            records = execute_neo4j_query(
                """
                MATCH (d:Dataset {id: $dataset_id})-[r:AGGREGATES]->(n)
                WHERE n.id IN $node_ids
                SET r.hidden = $hidden
                RETURN count(r) AS updated_count
                """,
                {"dataset_id": dataset_id, "node_ids": node_ids, "hidden": hidden},
            )
            result["updated_count"] = records[0]["updated_count"] if records else 0

        return result

    def hide_dataset_relationships_except(
        self,
        dataset_id: str,
        keep_node_ids: List[str],
    ) -> Dict[str, Union[bool, str, int]]:
        """Hide relationships to nodes not in keep_node_ids; keep listed nodes' relationships visible."""
        dataset_id = self.identify_dataset_id(dataset_id)
        self._ensure_dataset_writable(dataset_id)
        records = execute_neo4j_query(
            """
            MATCH (d:Dataset {id: $dataset_id})-[r:AGGREGATES]->(n)
            SET r.hidden = CASE WHEN n.id IN $keep_node_ids THEN null ELSE true END
            RETURN
                count(CASE WHEN n.id IN $keep_node_ids THEN 1 END) AS visible_count,
                count(CASE WHEN NOT n.id IN $keep_node_ids THEN 1 END) AS hidden_count
            """,
            {"dataset_id": dataset_id, "keep_node_ids": keep_node_ids or []},
        )
        if records:
            result = {
                "success": True,
                "visible_count": records[0].get("visible_count", 0),
                "hidden_count": records[0].get("hidden_count", 0),
            }
        else:
            result = {"success": False, "error": 1, "error_message": f"hide_dataset_relationships_except(): dataset '{dataset_id}' not found or has no AGGREGATES relationships."}

        return result

    def set_relationship_focus(
        self,
        dataset_id: str,
        target: str,
        attribute: str,
        values: List[str],
    ) -> Dict[str, Union[bool, str, int]]:
        """Set a list-valued focus attribute on one dataset relationship."""
        result: Dict[str, Union[bool, str, int]] = {"success": False, "updated_count": 0}
        parsed_target = parse_reference_uri(target)
        if parsed_target is None:
            result["error"] = 1
            result["error_message"] = f"set_relationship_focus(): invalid target '{target}'."
        elif attribute != "element_ids":
            result["error"] = 1
            result["error_message"] = f"set_relationship_focus(): invalid relationship attribute '{attribute}'."
        else:
            _, target_id = parsed_target
            dataset_id = self.identify_dataset_id(dataset_id)
            self._ensure_dataset_writable(dataset_id)
            records = execute_neo4j_query(
                f"""
                MATCH (d:Dataset {{id: $dataset_id}})-[r:AGGREGATES]->(n)
                WHERE n.id = $target_id
                SET r.`{attribute}` = $values
                RETURN count(r) AS updated_count
                """,
                {"dataset_id": dataset_id, "target_id": target_id, "values": values or []},
            )
            result["updated_count"] = records[0].get("updated_count", 0) if records else 0
            result["success"] = result["updated_count"] > 0
        return result

    def set_dataset_permanent(self, dataset_id: str, permanent: bool) -> Dict[str, Any]:
        """Change the permanency of an existing dataset.

        Renames the dataset id (using build_dataset_id) and connects or disconnects
        the Dataset node from the current Session node:
        - permanent=True:  new id is bare dataset_name, remove HAS_DATASET link to session.
        - permanent=False: new id is 'session_id#dataset_name', add HAS_DATASET link to session.
        """
        # Extract the bare dataset name from the current id
        if "#" in dataset_id:
            dataset_name = dataset_id.split("#", 1)[1]
        else:
            dataset_name = dataset_id

        new_id = self.build_dataset_id(dataset_name, permanent=permanent)

        try:
            if new_id != dataset_id:
                # Rename the dataset node id
                execute_neo4j_query(
                    "MATCH (d:Dataset {id: $old_id}) SET d.id = $new_id",
                    {"old_id": dataset_id, "new_id": new_id}
                )
                log(INFO, f"set_dataset_permanent: renamed dataset id '{dataset_id}' -> '{new_id}'")

            session_id = get_user_service().get_session_id()

            if permanent:
                # Detach from session: remove the HAS_DATASET relationship
                execute_neo4j_query(
                    "MATCH (s:Session {id: $session_id})-[r:HAS_DATASET]->(d:Dataset {id: $dataset_id}) DELETE r",
                    {"session_id": session_id, "dataset_id": new_id}
                )
                log(INFO, f"set_dataset_permanent: detached dataset '{new_id}' from session '{session_id}'")
            else:
                # Attach to session: ensure Session exists, then create HAS_DATASET link
                session_result = execute_neo4j_query(
                    "MATCH (s:Session {id: $session_id}) RETURN s",
                    {"session_id": session_id}
                )
                if not session_result:
                    userid = get_user_service().get_user_uuid()
                    execute_neo4j_query(
                        "MERGE (u:User {id: $userid, username: $userid}) SET u:SystemAcl",
                        {"userid": userid}
                    )
                    execute_neo4j_query(
                        "MATCH (u:User:SystemAcl {id: $userid}) MERGE (u)-[:HAS_SESSION]->(s:Session {id: $session_id})",
                        {"userid": userid, "session_id": session_id}
                    )
                execute_neo4j_query(
                    "MATCH (s:Session {id: $session_id}), (d:Dataset {id: $dataset_id}) MERGE (s)-[:HAS_DATASET]->(d)",
                    {"session_id": session_id, "dataset_id": new_id}
                )
                log(INFO, f"set_dataset_permanent: attached dataset '{new_id}' to session '{session_id}'")

            return {"success": True, "old_id": dataset_id, "new_id": new_id}

        except Exception as e:
            error_message = f"An error occurred while setting dataset permanency: {e}"
            log(ERROR, f"DatasetService.set_dataset_permanent ERROR: {error_message}. Traceback:\n{traceback.format_exc()}")
            return {"success": False, "error": 2, "error_message": error_message}

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
        """
        import uuid
        try:
            user_service: UserInterface = get_user_service()
            sessionid = user_service.get_session_id()
            userid = user_service.get_user_uuid()
            metadata_json = json.dumps(metadata) if metadata is not None else None

            self._ensure_dataset_writable(dataset_id, allow_missing=True)

            if permanent:
                execute_neo4j_query("""
                    MERGE (d:Dataset {id: $dataset_id})
                    SET d.name        = coalesce($dataset_name, d.name),
                        d.description = coalesce($dataset_description, d.description),
                        d.metadata    = coalesce($metadata, d.metadata)
                    RETURN d
                """, {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "dataset_description": dataset_description,
                    "metadata": metadata_json,
                })
            else:
                execute_neo4j_query("""
                    MERGE (u:User {id: $userid, username: $userid})
                    SET u:SystemAcl
                    MERGE (u)-[:HAS_SESSION]->(s:Session {id: $sessionid})
                    SET s.userid = $userid
                    WITH s
                    MERGE (d:Dataset {id: $dataset_id})
                    SET d.name        = coalesce($dataset_name, d.name),
                        d.description = coalesce($dataset_description, d.description),
                        d.metadata    = coalesce($metadata, d.metadata)
                    MERGE (s)-[:HAS_DATASET]->(d)
                    RETURN d
                """, {
                    "sessionid": sessionid,
                    "userid": userid,
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "dataset_description": dataset_description,
                    "metadata": metadata_json,
                })

            added = 0
            for record in records:
                props = {k: v for k, v in record.items() if k and '\x00' not in k}
                element_id = str(props.get("id") or uuid.uuid4())
                props["id"] = element_id
                result = execute_neo4j_query("""
                    MATCH (d:Dataset {id: $dataset_id})
                    CREATE (o:DatasetObject {id: $element_id})
                    SET o += $properties
                    MERGE (d)-[:AGGREGATES]->(o)
                    RETURN o
                """, {
                    "dataset_id": dataset_id,
                    "element_id": element_id,
                    "properties": props,
                })
                if result is None:
                    return {
                        "success": False,
                        "error": 1,
                        "error_message": f"Failed to add element '{element_id}' to dataset '{dataset_id}'",
                    }
                added += 1

            return {"success": True, "added_count": added, "dataset_id": dataset_id}

        except DatasetReadOnlyError:
            raise

        except Exception as e:
            log(ERROR, f"DatasetService.add_dataset_elements ERROR: {e}\n{traceback.format_exc()}")
            return {"success": False, "error": 2, "error_message": str(e)}

    def edit_dataset_element(
        self,
        dataset_id: str,
        record: Dict[str, Any],
        replace_all: bool = False,
    ) -> Dict[str, Union[bool, str]]:
        """Update properties of an existing DatasetObject node within a dataset."""
        element_id = record.get("id")
        if not element_id:
            return {"success": False, "error": 1, "error_message": "The 'id' field is mandatory in the record."}
        try:
            props = {k: v for k, v in record.items() if k and '\x00' not in k}
            self._ensure_dataset_writable(dataset_id)
            if replace_all:
                cypher = """
                    MATCH (d:Dataset {id: $dataset_id})-[:AGGREGATES]->(o:DatasetObject {id: $element_id})
                    SET o = $properties
                    RETURN o
                """
            else:
                cypher = """
                    MATCH (d:Dataset {id: $dataset_id})-[:AGGREGATES]->(o:DatasetObject {id: $element_id})
                    SET o += $properties
                    RETURN o
                """
            result = execute_neo4j_query(cypher, {
                "dataset_id": dataset_id,
                "element_id": str(element_id),
                "properties": props,
            })
            if not result:
                return {
                    "success": False,
                    "error": 2,
                    "error_message": f"Element '{element_id}' not found in dataset '{dataset_id}'.",
                }
            return {"success": True}

        except DatasetReadOnlyError:
            raise

        except Exception as e:
            log(ERROR, f"DatasetService.edit_dataset_element ERROR: {e}\n{traceback.format_exc()}")
            return {"success": False, "error": 3, "error_message": str(e)}

    def delete_dataset_elements(
        self,
        dataset_id: str,
        element_ids: List[str],
    ) -> Dict[str, Any]:
        """Remove specific DatasetObject nodes from a dataset by their 'id' property."""
        try:
            self._ensure_dataset_writable(dataset_id)
            match_result = execute_neo4j_query("""
                MATCH (d:Dataset {id: $dataset_id})-[:AGGREGATES]->(o:DatasetObject)
                WHERE o.id IN $element_ids
                RETURN collect(o.id) AS matched_ids
            """, {"dataset_id": dataset_id, "element_ids": element_ids})
            matched_ids = match_result[0].get("matched_ids", []) if match_result else []
            matched_ids = [str(element_id) for element_id in matched_ids]
            requested_ids = [str(element_id) for element_id in element_ids]
            matched_id_set = set(matched_ids)
            missing_ids = [element_id for element_id in requested_ids if element_id not in matched_id_set]
            deleted_count = len(matched_ids)

            if matched_ids:
                execute_neo4j_query("""
                    MATCH (d:Dataset {id: $dataset_id})-[:AGGREGATES]->(o:DatasetObject)
                    WHERE o.id IN $element_ids
                    DETACH DELETE o
                """, {"dataset_id": dataset_id, "element_ids": matched_ids})

            result: Dict[str, Any] = {
                "success": not missing_ids,
                "deleted_count": deleted_count,
            }
            if missing_ids:
                error_messages = [
                    f"Element '{element_id}' was not found in dataset '{dataset_id}'."
                    for element_id in missing_ids
                ]
                result["error"] = 2
                result["error_messages"] = error_messages
                result["error_message"] = " ".join(error_messages)

            return result

        except DatasetReadOnlyError:
            raise

        except Exception as e:
            log(ERROR, f"DatasetService.delete_dataset_elements ERROR: {e}\n{traceback.format_exc()}")
            return {"success": False, "error": 1, "error_message": str(e)}

    def delete_dataset(self, dataset_name: str) -> Dict[str, Union[bool, str]]:
        """Delete a dataset and all its associated objects."""
        # Accept either a bare name or a full id (containing '#')
        if "#" in dataset_name:
            dataset_id = dataset_name
        else:
            dataset_id = self.identify_dataset_id(dataset_name)
        
        try:
            # Delete the DatasetView node together with the owner and DatasetObject nodes.
            execute_neo4j_query("""
                MATCH (d:Dataset {id:$datasetid})
                OPTIONAL MATCH (d)-[:AGGREGATES]->(do:DatasetObject)
                OPTIONAL MATCH (d)-[:HAS_VIEW]->(view:DatasetView)
                DETACH DELETE do, view, d
            """, {"datasetid": dataset_id})
            log(INFO, "Deleted dataset: " + dataset_id)
            return {"success": True}
        
        except Exception as e:
            error_message = f"An error occurred: {e}"
            log(ERROR, f"DatasetService.delete_dataset ERROR: {error_message}. Traceback:\n{traceback.format_exc()}")
            return {"success": False, "error": 2, "error_message": error_message}

    def set_dataset_readonly(self, dataset_id: str, readonly: bool) -> Dict[str, Union[bool, str]]:
        """Set the read-only status of a permanent dataset."""
        result: Dict[str, Union[bool, str]] = {"success": False}

        try:
            dataset_id = self.identify_dataset_id(dataset_id)
            if readonly and "#" in dataset_id:
                result["error"] = 1
                result["error_message"] = "Session-scoped datasets cannot be marked read-only."
            else:
                records = execute_neo4j_query(
                    """
                    MATCH (d:Dataset {id: $dataset_id})
                    SET d.readonly = $readonly
                    RETURN d.id AS dataset_id, d.readonly AS readonly
                    """,
                    {"dataset_id": dataset_id, "readonly": bool(readonly)},
                )
                if records:
                    result = {
                        "success": True,
                        "dataset_id": dataset_id,
                        "readonly": bool(records[0].get("readonly") or False),
                    }
                else:
                    result["error"] = 2
                    result["error_message"] = f"Dataset '{dataset_id}' was not found."
        except Exception as e:
            result["error"] = 3
            result["error_message"] = str(e)
            log(ERROR, f"DatasetService.set_dataset_readonly ERROR: {e}\n{traceback.format_exc()}")

        return result

    def delete_session_datasets(self, sessionid: str) -> None:
        """Delete all datasets and their associated objects for the given session."""
        execute_neo4j_query("""
            MATCH (s:Session)-[:HAS_DATASET]->(d:Dataset)
            WHERE s.id = $sessionid
            OPTIONAL MATCH (d)-[:AGGREGATES]->(n:DatasetObject)
            OPTIONAL MATCH (d)-[:HAS_VIEW]->(view:DatasetView)
            DETACH DELETE d, n, view
        """, {"sessionid": sessionid})
    
    def find_datasets(self) -> List[str]:
        """Find all datasets for the current session."""
        datasets = execute_neo4j_query("MATCH (d:Dataset) WHERE d.id STARTS WITH $sessionid RETURN d.id", {"sessionid": get_user_service().get_session_id() + "#"})
        return [dataset["d.id"].split("#")[1] for dataset in datasets]
    
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
                only the dataset header fields (id, name, description, metadata, subset_of).
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

        When *metadata_only* is True the method skips the object-count and
        top-nodes queries and returns each dataset entry with ``object_count=None``
        and ``top_nodes=[]``.
        """
        sessionid = get_user_service().get_session_id()
        if not dataset_ids:
            # Fetch the user's session datasets and all permanent datasets, including Communities.
            dataset_records = execute_neo4j_query(
                """
                MATCH (d:Dataset)
                WHERE NOT d.id CONTAINS '#'
                   OR d.id STARTS WITH $session_prefix
                RETURN d, labels(d) AS dataset_labels
                """,
                {"session_prefix": sessionid + "#"}
            )
        else:
            # Callers must pass fully-qualified dataset IDs (e.g. 'session#name' or bare
            # permanent name). Use identify_dataset_id() to resolve a name to its ID first.
            dataset_records = execute_neo4j_query("MATCH (d:Dataset) WHERE d.id in $dataset_ids RETURN d, labels(d) AS dataset_labels", {"dataset_ids": dataset_ids})
        
        if dataset_records:
            result = { "datasets": [] }
            for dataset in dataset_records:
                dataset_id = dataset[0]["id"]
                dataset_name = dataset[0]["name"]
                dataset_description = dataset[0]["description"]
                metadata = dataset[0]["metadata"]
                is_community = "Community" in (dataset[1] or []) if len(dataset) > 1 else False

                if metadata_only:
                    # Skip the COUNT and top-nodes queries — objects_count is stored as a
                    # node property so we can read it directly without a relationship traversal.
                    stored_count = dataset[0].get("objects_count")
                    parent_datasets_meta = execute_neo4j_query(
                        "MATCH (d:Dataset {id: $dataset_id})-[:SUBSET_OF]->(parent:Dataset) RETURN parent.id",
                        {"dataset_id": dataset_id}
                    )
                    subset_of_ids_meta = [p[0] for p in parent_datasets_meta] if parent_datasets_meta else []
                    result["datasets"].append({
                        "id": dataset_id,
                        "name": dataset_name,
                        "description": dataset_description,
                        "object_count": stored_count,
                        "session_id": sessionid,
                        "user_id": get_user_service().get_user_uuid(),
                        "metadata": metadata,
                        "subset_of": subset_of_ids_meta if subset_of_ids_meta else None,
                        "readonly": bool(dataset[0].get("readonly") or False),
                        "permanent": self.is_dataset_permanent(dataset_id=dataset_id),
                        "is_community": is_community,
                        "top_nodes": [],
                    })
                    continue

                object_count = execute_neo4j_query("MATCH (d:Dataset {id: $dataset_id})-[:AGGREGATES]->(o) RETURN count(o)", {"dataset_id": dataset_id})
                
                # Query for parent datasets (SUBSET_OF relationships)
                parent_datasets = execute_neo4j_query(
                    "MATCH (d:Dataset {id: $dataset_id})-[:SUBSET_OF]->(parent:Dataset) RETURN parent.id",
                    {"dataset_id": dataset_id}
                )
                subset_of_ids = [p[0] for p in parent_datasets] if parent_datasets else []
                
                result["datasets"].append({
                    "id": dataset_id,
                    "name": dataset_name,
                    "description": dataset_description,
                    "object_count": object_count[0][0],
                    "session_id": sessionid,
                    "user_id": get_user_service().get_user_uuid(),
                    "metadata": metadata,
                    "subset_of": subset_of_ids if subset_of_ids else None,
                    "readonly": bool(dataset[0].get("readonly") or False),
                    "permanent": self.is_dataset_permanent(dataset_id=dataset_id),
                    "is_community": is_community,
                })

                if return_attributes:
                    # Use Cypher map projection to fetch only the requested fields — avoids
                    # transferring full node objects across all rows (text/embedding can be huge).
                    # Fetch Neo4j labels for classification and node's label property for output.
                    fields = ", ".join(f".{attr}" for attr in return_attributes)
                    query = f"MATCH (d:Dataset {{id: $dataset_id}})-[r:AGGREGATES]->(o) RETURN o{{{fields}}}, labels(o) as neo4j_labels, o.label as node_label, r.weight as weight"
                else:
                    query = "MATCH (d:Dataset {id: $dataset_id})-[r:AGGREGATES]->(o) RETURN o, labels(o) as neo4j_labels, o.label as node_label, r.weight as weight"
                if return_top_n > 0:
                    # positive number indicates the number of top nodes to return. If negative, return all nodes
                    query += " LIMIT $return_top_n"
                parameters = {"dataset_id": dataset_id, "return_top_n": return_top_n}
                top_nodes = execute_neo4j_query(query, parameters)
                result["datasets"][-1]["top_nodes"] = []
                for node in top_nodes:
                    node_obj = node[0]
                    neo4j_labels = node[1]
                    node_label = node[2]
                    weight = node[3] if len(node) > 3 else None
                    node_attributes = node_obj.keys()
                    if return_attributes:
                        node_attributes = { attr: node_obj[attr] for attr in return_attributes if attr in node_attributes }
                    else:
                        node_attributes = { attr: node_obj[attr] for attr in node_attributes if attr!="embedding" }
                    # Store Neo4j labels internally for classification (not in result)
                    node_attributes["_neo4j_labels"] = neo4j_labels
                    # Include actual node label if it exists, otherwise omit the label field
                    if node_label is not None:
                        node_attributes["label"] = node_label
                    # Add weight from relationship if available
                    if weight is not None:
                        node_attributes["weight"] = weight
                        # Also add as 'score' for backward compatibility
                        node_attributes["score"] = weight
                    if "id" in node_attributes and "Entity" in neo4j_labels:
                        # clearly label this as a KG node by prefixing the id with "kg://"
                        node_attributes["id"] = f"kg://{node_attributes['id']}"
                    result["datasets"][-1]["top_nodes"].append(node_attributes)

                if return_content and result["datasets"][-1]["top_nodes"]:
                    # Enrich :Chapter/:Paragraph nodes with text content, and all
                    # :Document/:Chapter/:Paragraph nodes with the parent document summary.
                    # Classify nodes by type based on Neo4j labels (stored internally).
                    # Run three optimized queries (0, 1, 2 hops) instead of one slow query with variable-length path.
                    doc_ids: List[str] = [n.get("id") for n in result["datasets"][-1]["top_nodes"] if "Document" in n.get("_neo4j_labels", [])]
                    chapter_ids: List[str] = [n.get("id") for n in result["datasets"][-1]["top_nodes"] if "Chapter" in n.get("_neo4j_labels", [])]
                    para_ids: List[str] = [n.get("id") for n in result["datasets"][-1]["top_nodes"] if "Paragraph" in n.get("_neo4j_labels", [])]

                    content_map: Dict[str, Dict] = {}

                    # Query 1: Documents (0 hops) — direct lookup with HAS_SUMMARY
                    if doc_ids:
                        doc_records = execute_neo4j_query("""
                            MATCH (n:Document) WHERE n.id IN $doc_ids
                            OPTIONAL MATCH (n)-[:HAS_SUMMARY]->(s:Summary)
                            RETURN n.id, labels(n), n.text, s.text
                        """, {"doc_ids": doc_ids})
                        for rec in (doc_records or []):
                            content_map[rec[0]] = {"labels": rec[1] or [], "text": rec[2], "doc_id": rec[0], "summary_text": rec[3]}

                    # Query 2: Chapters (1 hop) — Chapter -> Document -> Summary
                    if chapter_ids:
                        chapter_records = execute_neo4j_query("""
                            MATCH (n:Chapter) WHERE n.id IN $chapter_ids
                            OPTIONAL MATCH (n)-[:PART_OF]->(doc:Document)-[:HAS_SUMMARY]->(s:Summary)
                            RETURN n.id, labels(n), n.text, doc.id, s.text
                        """, {"chapter_ids": chapter_ids})
                        for rec in (chapter_records or []):
                            content_map[rec[0]] = {"labels": rec[1] or [], "text": rec[2], "doc_id": rec[3], "summary_text": rec[4]}

                    # Query 3: Paragraphs (2 hops) — Paragraph -> Chapter -> Document -> Summary
                    if para_ids:
                        para_records = execute_neo4j_query("""
                            MATCH (n:Paragraph) WHERE n.id IN $para_ids
                            OPTIONAL MATCH (n)-[:PART_OF]->(c:Chapter)-[:PART_OF]->(doc:Document)-[:HAS_SUMMARY]->(s:Summary)
                            RETURN n.id, labels(n), n.text, doc.id, s.text
                        """, {"para_ids": para_ids})
                        for rec in (para_records or []):
                            content_map[rec[0]] = {"labels": rec[1] or [], "text": rec[2], "doc_id": rec[3], "summary_text": rec[4]}

                    seen_doc_ids: set = set()
                    for node_attrs in result["datasets"][-1]["top_nodes"]:
                        nid = node_attrs.get("id")
                        info_entry = content_map.get(nid, {})
                        labels = info_entry.get("labels", [])
                        if "Chapter" in labels:
                            node_attrs["chapter_id"] = nid
                            node_attrs["chapter_text"] = info_entry.get("text")
                        elif "Paragraph" in labels:
                            node_attrs["paragraph_id"] = nid
                            node_attrs["paragraph_text"] = info_entry.get("text")
                        doc_id = info_entry.get("doc_id")
                        summary_text = info_entry.get("summary_text")
                        if doc_id and summary_text and doc_id not in seen_doc_ids:
                            seen_doc_ids.add(doc_id)
                            node_attrs["document_summary"] = summary_text

            # Strip internal fields before returning
            for ds in result.get("datasets", []):
                for node in ds.get("top_nodes", []):
                    node.pop("_neo4j_labels", None)

            result["success"] = True
        else:
            result = {"success": False, "error": 2, "error_message": "DatasetService: The datasets could not be retrieved."}

        return result

    def set_dataset_invisible(self, dataset_id: str, invisible: Optional[bool]) -> Dict[str, Union[bool, str]]:
        result: Dict[str, Union[bool, str]] = {"success": False}
        try:
            rows = execute_neo4j_query(
                """
                MATCH (d:Dataset {id: $dataset_id})
                SET d.invisible = $invisible
                RETURN d.id AS id
                """,
                {"dataset_id": dataset_id, "invisible": invisible}
            )
            if rows:
                result = {"success": True, "dataset_id": dataset_id}
            else:
                result = {"success": False, "error_message": f"Dataset not found: {dataset_id}"}
        except Exception as e:
            log(ERROR, f"DatasetService.set_dataset_invisible error: {e}")
            result = {"success": False, "error_message": str(e)}
        return result

    def get_datasets(self, datasets: Optional[List[str]], metadata_only: bool = False) -> List[Dict[str, Any]]:
        """Return a list of dataset dicts with normalized keys expected by Plan.fetch_datasets.

        Each dict will contain: id, name, description, domain, metadata (dict or None), objects_count (int).
        Always uses metadata_only=True internally since this method never returns top_nodes.
        """
        info = self.get_dataset_info(datasets, metadata_only=True)
        results: List[Dict[str, Any]] = []
        if not info or not info.get("success"):
            return results

        for ds in info.get("datasets", []):
            # metadata may already be a JSON string or dict
            metadata = ds.get("metadata")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = None

            objects_count = ds.get("object_count") if ds.get("object_count") is not None else ds.get("objects_count")
            subset_of = ds.get("subset_of")  # List of parent dataset IDs or None
            results.append({
                "id": ds.get("id"),
                "name": ds.get("name"),
                "description": ds.get("description"),
                "domain": ds.get("domain") if "domain" in ds else None,
                "metadata": metadata,
                "objects_count": objects_count,
                "subset_of": subset_of,
                "readonly": bool(ds.get("readonly") or False),
                "is_community": bool(ds.get("is_community") or False)
            })

        return results
    
    def dataset_union(
        self,
        description_result: str,
        dataset_result: str,
        dataset_names: List[str],
        metadata: Dict[str, Union[str, int]]
    ) -> Dict[str, Union[bool, str, list]]:
        """Perform set union operation on multiple datasets."""
        user_service: UserInterface = get_user_service()
        sessionid = user_service.get_session_id()
        
        if metadata is None:
            metadata = {}
        metadata["creation"] = f"UNION({','.join(dataset_names)})"

        userid = get_user_service().get_user_uuid()
        dataset_result_id = f"{sessionid}#{dataset_result}"
        dataset_result_name = dataset_result
        dataset_ids = [f"{sessionid}#{dataset_name}" for dataset_name in dataset_names]

        # perform set union on the datasets
        execute_neo4j_query("""
            MATCH (s:Session {id: $sessionid})
            SET s.userid = $userid
            WITH s
            MATCH (source:Dataset)-[source_association:AGGREGATES]->(sourceObject)
            WHERE source.id IN $dataset_ids
            MERGE (d:Dataset {id: $dataset_result_id})
            SET
                d.name = $dataset_result_name,
                d.description = $description_result,
                d.created = timestamp(),
                d.metadata = $metadata
            WITH s, d, sourceObject, source_association
            MERGE (s)-[:HAS_DATASET]->(d)
            WITH d, sourceObject, source_association
            MERGE (d)-[r:AGGREGATES]->(sourceObject)
            WITH d, source_association, r
            WITH d, source_association, r
            SET r.weight = CASE
                WHEN COALESCE(r.weight, 0) > source_association.weight THEN COALESCE(r.weight, 0)
                ELSE source_association.weight
            END
        """, parameters= {
            "sessionid": sessionid,
            "userid": userid,
            "dataset_ids": dataset_ids,
            "dataset_result_id": dataset_result_id,
            "dataset_result_name": dataset_result_name,
            "description_result": description_result,
            "metadata": json.dumps(metadata)
        })

        # set the object count of the resulting dataset
        result = execute_neo4j_query("""
            MATCH (d:Dataset {id: $dataset_result_id})-[r:AGGREGATES]->()
            WITH d, COUNT(r) as object_count
            SET d.object_count = object_count
            RETURN object_count
        """, parameters={ "dataset_result_id": dataset_result_id })

        if result:
            return { "success": True, "dataset_name": dataset_result, "object_count": result[0].get("object_count", 0) }
        else:
            return { "success": False, "error": 2, "error_message": "dataset_union(): The datasets could not be united." }
    
    def dataset_intersection(
        self,
        description_result: str,
        dataset_result: str,
        dataset_names: List[str],
        metadata: Dict[str, Union[str, int]]
    ) -> Dict[str, Union[bool, str, list]]:
        """Perform set intersection operation on multiple datasets."""
        user_service: UserInterface = get_user_service()
        sessionid = user_service.get_session_id()
        
        if len(dataset_names) < 2:
            return {"success": False, "error": 1, "error_message": "dataset_intersection(): At least two datasets are required."}

        if metadata is None:
            metadata = {}
        metadata["creation"] = f"INTERSECTION({','.join(dataset_names)})"

        userid = get_user_service().get_user_uuid()
        dataset_result_id = f"{sessionid}#{dataset_result}"

        where_clauses = []
        parameters = {
            "sessionid": sessionid,
            "userid": userid,
            "dataset_result_id": dataset_result_id,
            "dataset_result_name": dataset_result,
            "description_result": description_result,
            "metadata": json.dumps(metadata)
        }
        
        match_clauses = []
        weight_clauses = []
        with_clauses = []
        for i, dataset_name in enumerate(dataset_names):
            dataset_id = f"{sessionid}#{dataset_name}" if f"{sessionid}#" not in dataset_name else dataset_name
            match_clause = f"(dataset{i}:Dataset)-[x{i}:AGGREGATES]->(object{i})"
            if match_clause not in match_clauses:
                match_clauses.append(match_clause)
            where_clauses.append(f"dataset{i}.id IN $dataset_id{i}")
            parameters[f"dataset_id{i}"] = [dataset_id]
            weight_clauses.append(f"LOG(1.0 + COALESCE(x{i}.weight, 0))")
            with_clauses.append(f"x{i}")
        weight_clause = f"({' + '.join(weight_clauses)})"
        with_clause = ", ".join(with_clauses)
        
        match_clause = ", ".join(match_clauses)
        where_clause = " AND ".join(where_clauses)
        object_comparison_clause = " AND ".join([f"object{i} = object0" for i in range(1, len(dataset_names))])
        
        query = f"""
            MATCH (s:Session {{id: $sessionid}})
            SET s.userid = $userid
            MERGE (d:Dataset {{id: $dataset_result_id}})
            SET
                d.name = $dataset_result_name,
                d.description = $description_result,
                d.created = timestamp(),
                d.metadata = $metadata
            MERGE (s)-[:HAS_DATASET]->(d)
            WITH d
            MATCH {match_clause}
            WHERE {where_clause} AND {object_comparison_clause}
            MERGE (d)-[r:AGGREGATES]->(object0)
            WITH r, {with_clause}
            SET r.weight = ({weight_clause})"""
        
        result = execute_neo4j_query(query, parameters)
        if result is None:
            return { "success": False, "error": 2, "error_message": "dataset_intersection(): The datasets could not be intersected." }
        
        # Find the maximum weight
        max_weight_query = """
            MATCH (d:Dataset {id: $dataset_result_id})-[r:AGGREGATES]->()
            RETURN MAX(r.weight) as max_weight
        """
        max_weight_result = execute_neo4j_query(max_weight_query, parameters)
        max_weight = max_weight_result[0]['max_weight'] if max_weight_result and max_weight_result[0]['max_weight'] else 1.0
        
        # Normalize the weights and set the object count of the resulting dataset
        normalize_query = f"""
            MATCH (d:Dataset {{id: $dataset_result_id}})-[r:AGGREGATES]->()
            WITH d, r, {max_weight} as max_weight
            SET r.weight = r.weight / max_weight
            WITH d, COUNT(r) as object_count
            SET d.object_count = object_count
            RETURN object_count
        """
        result = execute_neo4j_query(normalize_query, parameters)

        if result:
            output = { "success": True, "dataset_name": dataset_result }
            if len(result) == 1:
                output["object_count"] = result[0].get("object_count", 0)
            return output
        else:
            return { "success": False, "error": 3, "error_message": "dataset_intersection(): The datasets could not be intersected." }
    
    def dataset_difference(
        self,
        description_result: str,
        dataset_result: str,
        dataset_names: List[str],
        metadata: Dict[str, Union[str, int]]
    ) -> Dict[str, Union[bool, str, list]]:
        """Perform set difference operation (dataset[0] - dataset[1])."""
        user_service: UserInterface = get_user_service()
        sessionid = user_service.get_session_id()
        
        if len(dataset_names) < 2:
            return {"success": False, "error": 4, "error_message": "dataset_difference(): At least two datasets are required."}
        if metadata is None:
            metadata = {}
        metadata["creation"] = f"DIFFERENCE({','.join(dataset_names)})"

        userid = get_user_service().get_user_uuid()

        match_clauses = []
        where_clauses = []
        parameters = {
            "dataset_result_id": f"{sessionid}#{dataset_result}",
            "dataset_result_name": dataset_result,
            "description_result": description_result,
            "metadata": json.dumps(metadata),
            "sessionid": sessionid,
            "userid": userid
        }

        for i, dataset_name in enumerate(dataset_names):
            dataset_id = [f"{sessionid}#{dataset_name}"]
            match_clauses.append(f"(dataset{i}:Dataset)-[x{i}:AGGREGATES]->(object{i})")
            where_clauses.append(f"dataset{i}.id IN $dataset_id{i}")
            parameters[f"dataset_id{i}"] = dataset_id

        match_clause = ", ".join(match_clauses)
        where_clause = " AND ".join(where_clauses)

        query = f"""
            MATCH (s:Session {{id: $sessionid}})
            SET s.userid = $userid
            MERGE (d:Dataset {{id: $dataset_result_id}})
            SET
                d.name = $dataset_result_name,
                d.description = $description_result,
                d.created = timestamp(),
                d.metadata = $metadata
            MERGE (s)-[:HAS_DATASET]->(d)
            WITH d
            MATCH {match_clause}
            WHERE {where_clause}
            AND NOT (dataset1)-[:AGGREGATES]->(object0)
            WITH d, object0, x0.weight AS weight
            MERGE (d)-[r:AGGREGATES]->(object0)
            SET r.weight = weight
            WITH d, count(DISTINCT r) as object_count
            SET d.object_count = object_count
            RETURN object_count    """

        result = execute_neo4j_query(query, parameters)

        if result:
            if len(result) == 0:
                object_count = 0
            else:
                object_count = result[0].get("object_count", 0)
            return { "success": True, "dataset_name": dataset_result, "object_count": object_count }
        else:
            return { "success": False, "error": 5, "error_message": "dataset_difference(): The datasets could not be differenced." }
    
    def dataset_inclusion(
        self,
        dataset_names: List[str]
    ) -> Dict[str, Union[bool, str]]:
        """Check if second dataset is included in the first."""
        user_service: UserInterface = get_user_service()
        sessionid = user_service.get_session_id()
        
        if len(dataset_names) != 2:
            return {"success": False, "error": 6, "error_message": "dataset_inclusion(): Exactly two datasets are required."}
        
        included = False
        dataset_ids = [f"{sessionid}#{dataset_name}" for dataset_name in dataset_names]
        
        try:
            result = execute_neo4j_query("""
                MATCH (d1:Dataset)-[:AGGREGATES]->(o1)
                WHERE d1.id = $dataset_ids[0]
                MATCH (d2:Dataset)-[:AGGREGATES]->(o2)
                WHERE d2.id = $dataset_ids[1]
                WITH collect(o1) AS dataset0, collect(o2) AS dataset1
                RETURN ALL(o2 IN dataset0 WHERE o2 IN dataset1) AS included
            """, parameters={"dataset_ids": dataset_ids})
        except Exception as e:
            log(ERROR, f"DatasetService.dataset_inclusion ERROR: {e}. Traceback:\n{traceback.format_exc()}")
            return {"success": False, "error": 7, "error_message": f"dataset_inclusion(): {str(e)}"}

        if result:
            included = result[0].get("included", False)
        return { "success": True, "included": included }
    
    def dataset_comparison(
        self,
        dataset_names: List[str]
    ) -> Dict[str, Union[bool, str]]:
        """Check if two datasets contain exactly the same elements."""
        user_service: UserInterface = get_user_service()
        sessionid = user_service.get_session_id()
        
        if len(dataset_names) != 2:
            return {"success": False, "error": 8, "error_message": "dataset_comparison(): Exactly two datasets are required."}
        
        equal = False
        dataset_ids = [f"{sessionid}#{dataset_name}" for dataset_name in dataset_names]

        try:
            result = execute_neo4j_query("""
                MATCH (d1:Dataset)-[:AGGREGATES]->(o1:DatasetObject)
                WHERE d1.id = $dataset_ids[0]
                MATCH (d2:Dataset)-[:AGGREGATES]->(o2:DatasetObject)
                WHERE d2.id = $dataset_ids[1]
                WITH collect(o1) AS dataset0, collect(o2) AS dataset1
                RETURN dataset0 = dataset1 AS equal
            """, parameters={"dataset_ids": dataset_ids})
        except Exception as e:
            log(ERROR, f"DatasetService.dataset_comparison ERROR: {e}. Traceback:\n{traceback.format_exc()}")
            return {"success": False, "error": 9, "error_message": f"dataset_comparison(): {str(e)}"}

        if result:
            equal = result[0].get("equal", False)
        return { "success": True, "equal": equal }


# module level singleton accessors (same pattern as other services)
_dataset_service_instance: Optional[DatasetInterface] = None

def get_dataset_service() -> DatasetInterface:
    global _dataset_service_instance
    if _dataset_service_instance is None:
        _dataset_service_instance = DatasetService()
    return _dataset_service_instance

def set_dataset_service(service: DatasetInterface) -> None:
    global _dataset_service_instance
    _dataset_service_instance = service


log_end_file(__file__)


if __name__ == "__main__":
    import json

    from src.utils.unit_test import init_unit_test
    init_unit_test()

    from src.services.user_service import get_user_service
    from src.utils.log import log_init, log_exit
    log_init()

    dataset_service = get_dataset_service()
    try:
        if False:
            # Test creating a dataset
            test_dataset_id = "testsession#testdataset"
            test_userid = get_user_service().get_user_uuid()
            test_dataset_name = "testdataset"
            test_dataset_description = "This is a test dataset."
            test_metadata = {"source": "unit_test", "type": "test_data"}
            fragment_ids = ["docid-543", "docid-543-c0"]

            create_result = dataset_service.create_dataset_from_fragments(
                fragments=fragment_ids,
                dataset_id=test_dataset_id,
                userid=test_userid,
                dataset_name=test_dataset_name,
                dataset_description=test_dataset_description,
                metadata=test_metadata
            )
            log(INFO, f"Create Dataset Result: {json.dumps(create_result, indent=4)}")

        if True:
            # test get dataset info with summaries for query_1 documents
            log(INFO, "========== Unit Test: DatasetInfo with summaries ==========")
            result = dataset_service.get_dataset_info(
                dataset_ids=['luca-s-1#query_1'],
                return_top_n=3,
                return_attributes=['id', 'name', 'title'],
                return_content=True
            )

            if result.get("success"):
                log(INFO, f"Test Result: SUCCESS - Retrieved {len(result.get('datasets', []))} dataset(s)")
                for ds in result.get("datasets", []):
                    log(INFO, f"Dataset: {ds['name']} ({ds['object_count']} objects)")
                    top_nodes = ds.get("top_nodes", [])
                    log(INFO, f"  Top nodes returned: {len(top_nodes)}")

                    for i, node in enumerate(top_nodes):
                        node_id = node.get("id")
                        has_summary = "document_summary" in node
                        summary_preview = (node.get("document_summary", "")[:100] + "...") if has_summary else "[NO SUMMARY]"
                        log(INFO, f"  Node {i+1}: {node_id} | Has Summary: {has_summary} | Preview: {summary_preview}")

                    # Verify all nodes have summaries
                    nodes_with_summary = [n for n in top_nodes if "document_summary" in n]
                    if len(nodes_with_summary) == len(top_nodes):
                        log(INFO, f"✓ PASS: All {len(top_nodes)} nodes have document_summary field")
                    else:
                        log(WARNING, f"✗ FAIL: Only {len(nodes_with_summary)}/{len(top_nodes)} nodes have document_summary field")

                    pass
            else:
                log(ERROR, f"Test Result: FAILED - {result.get('error_message', 'Unknown error')}")

    except Exception as e:
        log(FATAL, f"Unit test for {__file__} failed with exception: {e}. Traceback:\n{traceback.format_exc()}")

    finally:
        log_exit()
