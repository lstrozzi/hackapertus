from logging import FATAL

from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import json
import traceback
import re
from typing import Optional, List, Dict, Any, Set, Tuple, TYPE_CHECKING
from src.interfaces.ontology_interface import OntologyInterface
from src.core.base.InputOutput import execute_neo4j_query
from src.utils.log import log, DEBUG, INFO, WARNING, ERROR
from src.utils.acl_cypher_utils import acl_ontology_read_access_query, acl_ontology_any_read_access_query, acl_ontology_permission_check_query, acl_ontology_where_clause, acl_document_where_clause
from src.services.user_service import get_user_service
from src.interfaces.user_interface import UserInterface
from src.config.constants import MAX_RAG_MESSAGE_LENGTH_DEFAULT, KG_HYDRATION_LINKED_DOCUMENTS_ENABLED

if TYPE_CHECKING:
    from src.core.thalaplan.Plan import Plan

# Constants moved from Ontology module
ONTOLOGY_ID_ALL = '__ALL__'  # Special value to retrieve all ontologies (used by AgentOntology)
ONTOLOGY_DEFAULT_ID: str = "DEFAULT"
COMMUNITY_ID_DEFAULT = 'DefaultCommunity'
COMMUNITY_NAME_DEFAULT = "DefaultCommunity"

ATTRIBUTE_REQUIRES_CONSOLIDATION = "requires_consolidation"
ATTRIBUTE_ENTITY_DESCRIPTION = "entity_description"
ATTRIBUTE_ENTITY_SOURCE_LABEL = "entity_source_labels"
ATTRIBUTE_ENTITY_TARGET_LABEL = "entity_target_labels"
RESERVED_ONTOLOGY_LABELS = ['Ontology', 'EntityOntology', 'RelationshipOntology', 'Entity']
RESERVED_ONTOLOGY_ATTRIBUTES = ["id", "label", "source_id", "embedding", "isDefault", ATTRIBUTE_REQUIRES_CONSOLIDATION, ATTRIBUTE_ENTITY_DESCRIPTION, ATTRIBUTE_ENTITY_SOURCE_LABEL, ATTRIBUTE_ENTITY_TARGET_LABEL, "color_background", "color_foreground", "uri"]
RESERVED_ATTRIBUTE_STARTSWITH = ["citation_"]
CASE_SENSITIVE_ATTRIBUTES = ["name"]
CONSOLIDATION_SUFFIX = "_consolidated"
AGGREGATE_VALUES_ATTRIBUTE = "aggregate_values"


def linked_document_budget_per_entity(entity_count: int) -> int:
    """Divide the configured default aggregate RAG budget evenly across entities."""
    count: int = max(1, int(entity_count))
    budget: int = max(0, MAX_RAG_MESSAGE_LENGTH_DEFAULT // count)

    return budget


def get_default_ontology_id() -> Optional[str]:
    """Return the id of the Ontology node with isDefault=true, or None if not found."""
    try:
        result = execute_neo4j_query(
            "MATCH (o:Ontology {isDefault: true}) RETURN o.id AS id LIMIT 1", {}
        )
        if result and len(result) > 0:
            return result[0].get("id")
    except Exception as e:
        log(ERROR, f"get_default_ontology_id: {e}")
    return None


# Regex to detect and extract a leading "(type)" hint from ontology attribute values
# e.g. "(string) Strength of..." -> type="string", rest="Strength of..."
_re_type_hint = re.compile(r'^\(([^)]+)\)\s*')

def _extract_declared_type(attribute_definition: Any) -> Optional[str]:
    """Extract the datatype from an ontology attribute definition."""
    if not isinstance(attribute_definition, str):
        return None
    match = _re_type_hint.match(attribute_definition.strip())
    return match.group(1).strip().lower() if match else None


def _convert_value_to_declared_type(value: Any, declared_type: Optional[str]) -> Any:
    """Convert a value according to an ontology datatype, preserving text on failure."""
    # TODO: Convert declared dates to native Neo4j temporal values when the write path supports them consistently.
    if value is None or declared_type is None or declared_type in ("string", "date"):
        return value

    try:
        if declared_type in ("int", "integer"):
            return int(str(value).strip())
        if declared_type in ("float", "double", "number"):
            return float(str(value).strip())
        if declared_type in ("bool", "boolean"):
            normalized = str(value).strip().lower()
            if normalized in ("true", "1"):
                return True
            if normalized in ("false", "0"):
                return False
    except (TypeError, ValueError):
        return value

    return value


SYSTEM_PROMPT_CONSOLIDATE = f"""You are a helpful assistant, whose only job is to consolidate sentences.

# GOAL
You are given a list of sentences, and you need to consolidate them into a single sentence.
These sentences are related to one attribute of a knowledge graph entity or relationship.
The sentences have been extracted from different sources, such as documents where such entities and relationships are mentioned.
The goal of consolidation, is to create a single, coherent, and concise sentence that summarizes all the information contained in the input sentences
but without any redundancy, and without adding any new information. The consolidated sentence should be a summary of the input sentences
and it applies to the attribute that whose sentences need consolidation.

# GUIDELINES
## ATTRIBUTE DEFINITION
You are provided by a strict definition for the attribute that you need to consolidate.
This definition has been provided by the user and must be adhered to.

## LANGUAGE
The sentences may be written in different languages: in that case, you should output a consolidated sentence in English.
If the sentences are written all in the same language: you should output a consolidated sentence in the same language.

## ATTRIBUTE CLASSES
### Class "A" attributes
One class "A" of attributes requires just nouns or numbers or dates or time in the consolidated result.
Also belonging to class "A" are attributes that refer to book titles, names, labels, codes or other unique values.
Examples are: attribute "name", "age", "birthdate", "gender", which respectively admit values like "John", 20, "1990-01-01" etc.

### Class "B" attributes
Another class "B" of attributes may require a full sentence in the consolidated result.
Examples are: attribute "description", "title", "summary", which respectively admit values like "John is a nice person", "The Great Getsby", "A summary of the book" etc.
You have to decide which class the attribute belongs to, and consolidate the sentences accordingly.

## STYLE
The consolidated sentence should be concise and clear, without any redundancy.
The consolidated sentence should be a single sentence, or a very short paragraph if necessary to convey all information.
The consolidated sentence should be written in the positive form, as if you are describing something that is true.

## LENGTH
The consolidated sentence should be as short as possible, while still conveying all information.


# EXAMPLES
    - the attribute "name" of type "string" of an entity that represent an Animal.
      The attribues's definition is: "The full name of the animal as it appears in the sources. Use capital letters as usual for names."
      This attribute belongs to class "A" attributes because it is a noun.
      It may have multiple sentences that describe the name of the animal, such as:
        "The animal is called 'Pat the Snake'.",
        "The snake is named 'Pat'.",
        "The animal is known as 'pat'.
      In this example, the consolidated name must be: "Pat the Snake"

    - the attribute "age" of type "integer" of a Person entity.
      The attribues's definition is: "The age of the person in years (type: integer)."
      It belongs to class "A" attributes because it is a number.
      The sentences to consolidate can be like:
        "The person is 20 years old.",
        "The age of the person is 20.",
        "The person is twenty years old."
      In this example, the consolidated age must be: 20

    - the attribute "title" of a Document entity.
      The attribues's definition is: "The title of the document. It must be a unique, precise value as found in the sources."
      It belongs to class "A" attributes because it is the title of a book and has a unique value.
      The sentences to consolidate can be like:
        "The document is titled 'The Great Gatsby'.",
        "The title of the document is 'The Great Gatsby'.",
        "The document is named 'The Great Gatsby'."
      In this example, the consolidated title must be: "The Great Gatsby"

    - the attribute "required_resources" of a Task entity.
      The attribues's definition is: "A comma-separated list of resources needed to accomplish the task, including their amount. Omit articles and conjunctions."
      This attribute belongs to class "B" attributes because it doesn't have the characteristics to belong to class "A".
      The sentences to consolidate can be like:
        "A computer, a pencil, a piece of paper.",
        "To accomplish the task, a computer, and a paper notebook are needed.",
        "The task needs a pencil, a piece of paper and a laptop.".
      In this example, the consolidated required_resources must be: "1 computer, 1 pencil, 1 piece of paper"

    - the attribute "description" of a relationship between two entities (relationship source=Mary and destination=John).
      The attribues's definition is: "A succint description of the relationship between source and destination entities."
      This attribute belongs to class "B" attributes because it doesn't have the characteristics to belong to class "A".
      The sentences to consolidate can be like:
        "The relationship source entity is parent of the destination entity.",
        "Mary is the mother of John: the best mother in the world.",
        "John is the son of Mary. John loves his mother."
      In this example, the consolidated description must be: "Mary is the mother of John: she is a good and loved mother."
"""
SYSTEM_PROMPT_MERGE_ENTITIES = f"""You are a helpful assistant, whose only job is to identify if some entities from a list are actually the variations of the same entity, just with different id and slighlty different attributes.
The provided list of entities has been extracted by multiple sources and may contain duplicates or variations of the same entity. The extraction process ran in parallel, and the entities were not matched or merged.
Your task is to identify the entities that are actually the same entity, and merge them into a single entity.

# OUTPUT FORMAT
The output of your task is a Json structure that contains:
- "optional_internal_thought": a string that describes the reasoning that conducts you to formulate the correct answer.
- "entity_to_consolidate": a list of lists of entity ids. Each list of entity ids represents the entities that should be merged together.

# EXAMPLE
Let's consider the following list of entities labeled "Person", which share the following entity definition: "A person is a human being, from a fictional or real document source."
The list of entities is:
- Entity "John" with id "john123" and name="John Doe" and age=20, profession="John is a good developer".
- Entity "John" with id "john456" and name="John Doe" and age=20, profession="Computer programming is John's job, and he's very proficient at it".
- Entity "Mary" with id "mary789" and name="Mary Smith" and age=20, profession="Mary is a good developer".

In this example, the entities "john123" and "john456" are actually the same person, John Doe, and should be merged together. Mary is a different person and should not be merged with John.
The output of your task should be: [["john123", "john456"]]
"""
OUTPUT_FORMAT_MERGE_ENTITIES = {
    "optional_internal_thought": "The reasoning that conducts the assistant to formulate the correct answer",
    "entity_to_consolidate": [["john123", "john456"]]
}
OUTPUT_SCHEMA_MERGE_ENTITIES = {
    "name": "MergeEntitiesResponse",
    "schema": {
        "type": "object",
        "properties": {
            "optional_internal_thought": {
                "type": "string"
            },
            "entity_to_consolidate": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            }
        },
        "required": ["optional_internal_thought", "entity_to_consolidate"]
    },
    "strict": True
}


class OntologyService(OntologyInterface):
    """Concrete implementation of OntologyInterface.

    This implementation currently provides a placeholder `has_ontologies`
    method that returns False. Replace with a real implementation that
    queries the Neo4j store or other backing store when available.
    """

    def __init__(self):
        self._log_prefix = "OntologyService"

    def check_read_access_on_ontology(self, effective_user_id: Optional[str], ontology_id: str) -> bool:
        """Check whether effective_user_id has GRANT:READ permission on the given Ontology node.

        Mirrors the C# AclService.CheckAccess / AclQueryBuilder logic:
          1. Tenant-admin groups bypass all checks (always True).
          2. A direct GRANT:READ from any of the effective user's groups on the ontology node.
          3. An inherited GRANT:READ from any ancestor reachable via [:CONTAINS|DEFINES*]
             up to the first protected ancestor (inclusive stop).

        Args:
            effective_user_id: Neo4j User.id.  When None the call returns True (system/pipeline bypass,
                      matching the C# convention of effectiveUserId == null skipping the check).
            ontology_id: The id property of the target Ontology node.

        Returns:
            True if access is granted, False otherwise.
        """
        if effective_user_id is None:
            return True

        try:
            query = acl_ontology_read_access_query()
            result = execute_neo4j_query(query, {"effective_user_id": effective_user_id, "ontology_id": ontology_id})
            if result and len(result) > 0:
                return bool(result[0].get("has_access", False))
        except Exception as e:
            log(ERROR, f"{self._log_prefix}.check_read_access_on_ontology: Error for effective user='{effective_user_id}' ontology='{ontology_id}': {e}")
        return False

    def check_ontology_permission(self, effective_user_id: Optional[str], ontology_id: str, permission: str) -> bool:
        """Check a resource permission on an ontology, preserving system bypass semantics."""
        result = False

        if effective_user_id is None:
            result = True
        else:
            try:
                query = acl_ontology_permission_check_query(permission)
                query_result = execute_neo4j_query(query, {
                    "effective_user_id": effective_user_id,
                    "ontology_id": ontology_id
                })
                if query_result and len(query_result) > 0:
                    result = bool(query_result[0].get("has_access", False))
            except Exception as e:
                log(ERROR, f"{self._log_prefix}.check_ontology_permission: Error for effective user='{effective_user_id}' ontology='{ontology_id}' permission='{permission}': {e}")

        return result

    def check_read_access_on_community(self, effective_user_id: Optional[str], community_id: str) -> bool:
        """Check whether effective_user_id has GRANT:READ permission on the given Community node.

        Resolves the Ontology that defines the community (one-to-one [:DEFINES] relationship)
        and delegates to check_read_access_on_ontology.

        Args:
            effective_user_id: Neo4j User.id.  When None the call returns True (system/pipeline bypass).
            community_id: The id property of the target Community node.

        Returns:
            True if access is granted, False otherwise.
        """
        if effective_user_id is None:
            return True

        try:
            # First, try to find the community directly
            query = """
                MATCH (o:Ontology)-[:DEFINES]->(c:Community {id: $community_id})
                RETURN o.id AS ontology_id
                LIMIT 1
            """
            result = execute_neo4j_query(query, {"community_id": community_id})
            
            # If no community found, the ID might be an ontology ID — resolve to community
            if not result or len(result) == 0:
                log(DEBUG, f"check_read_access_on_community: No community found for '{community_id}', trying as ontology ID")
                community_query = """
                    MATCH (o:Ontology {id: $ontology_id})-[:DEFINES]->(c:Community)
                    RETURN c.id AS community_id, o.id AS ontology_id
                    LIMIT 1
                """
                result = execute_neo4j_query(community_query, {"ontology_id": community_id})
                if not result or len(result) == 0:
                    log(WARNING, f"{self._log_prefix}.check_read_access_on_community: No parent Ontology found for community='{community_id}'")
                    return False
                # Use the resolved community ID for subsequent checks
                community_id = result[0].get("community_id")
                ontology_id = result[0].get("ontology_id")
            else:
                ontology_id = result[0].get("ontology_id")
            
            if not ontology_id:
                log(WARNING, f"{self._log_prefix}.check_read_access_on_community: Parent Ontology has no id for community='{community_id}'")
                return False
            return self.check_read_access_on_ontology(effective_user_id, ontology_id)
        except Exception as e:
            log(ERROR, f"{self._log_prefix}.check_read_access_on_community: Error for effective user='{effective_user_id}' community='{community_id}': {e}")
        return False

    def check_has_any_read_access(self, effective_user_id: Optional[str]) -> bool:
        """Return True if the effective user has GRANT:READ on at least one Ontology.

        Used as a fast early-abort guard when no specific community is targeted and
        the full ACL enforcement will be handled post-search by get_entities_by_ids.

        Args:
            effective_user_id: Neo4j User.id. When None, returns True (system/pipeline bypass).

        Returns:
            True if the effective user is a tenant-admin or holds GRANT:READ on any Ontology.
        """
        if effective_user_id is None:
            return True

        try:
            query = acl_ontology_any_read_access_query()
            result = execute_neo4j_query(query, {"effective_user_id": effective_user_id})
            if result and len(result) > 0:
                return bool(result[0].get("has_any_access", False))
        except Exception as e:
            log(ERROR, f"{self._log_prefix}.check_has_any_read_access: Error for effective user='{effective_user_id}': {e}")
        return False

    def has_ontologies(self, document_ids: Optional[List[str]] = None, plan: Optional['Plan'] = None) -> bool:
        """Return True if any ontologies are available in the backing store.
        
        Args:
            document_ids: Optional list of document IDs to check if any follow an ontology
            plan: Optional Plan object to check if plan_ontology_id has Entity nodes
            
        Returns:
            True if ontologies are available based on the provided parameters
        """
        from src.core.base.InputOutput import execute_neo4j_query
        has_ontologies: bool = False

        try:
            # Case 1: Check if any document follows any ontology
            if document_ids is not None and len(document_ids) > 0:
                query: str = """
                    MATCH (d:Document)-[:FOLLOWS]->(o:Ontology)
                    WHERE d.id IN $document_ids OR d.documentid IN $document_ids
                    RETURN count(o) > 0 AS has_ontologies
                """
                parameters: Dict[str, List[str]] = {"document_ids": document_ids}
                result = execute_neo4j_query(query, parameters)
                if result and len(result) > 0:
                    has_ontologies = result[0].get("has_ontologies", False)
                    log(DEBUG, f"{self._log_prefix}: Documents {document_ids} follow ontology: {has_ontologies}")

            # Case 2: Check if plan's ontology has Entity or KeywordCategory nodes (only if document check not requested)
            elif plan is not None and hasattr(plan, 'get_ontology_id') and plan.get_ontology_id() is not None:
                ontology_id: str = plan.get_ontology_id()
                query: str = """
                    MATCH (o:Ontology {id: $ontology_id})
                    OPTIONAL MATCH (o)-[:CONTAINS]->(n)
                    WHERE n:Entity OR n:KeywordCategory
                    RETURN count(n) > 0 AS has_ontologies
                """
                parameters: Dict[str, str] = {"ontology_id": ontology_id}
                result = execute_neo4j_query(query, parameters)
                if result and len(result) > 0:
                    has_ontologies = result[0].get("has_ontologies", False)
                    log(DEBUG, f"{self._log_prefix}: Ontology {ontology_id} has entities: {has_ontologies}")

            # Case 3: Check if any Entity or KeywordCategory nodes exist (only if neither document nor plan provided)
            else:
                query: str = """
                    MATCH (n)
                    WHERE n:Entity OR n:KeywordCategory
                    RETURN count(n) > 0 AS has_ontologies
                    LIMIT 1
                """
                result = execute_neo4j_query(query, {})
                if result and len(result) > 0:
                    has_ontologies = result[0].get("has_ontologies", False)
                    log(DEBUG, f"{self._log_prefix}: Any entities or keyword categories exist: {has_ontologies}")
            
        except Exception as e:
            log(ERROR, f"{self._log_prefix}: Error in has_ontologies(): {e}")
            has_ontologies = False

        # return only once at the end of the function
        return has_ontologies

    def is_not_reserved_and_unconsolidated_attribute(self, attr_name: str) -> bool:
        """Check if an attribute name is not reserved and not consolidated.

        Args:
            attr_name: The attribute name to check

        Returns:
            True if the attribute is not reserved and not consolidated, False otherwise
        """
        # reject non-strings or empty names
        if not isinstance(attr_name, str):
            return False
        attr = attr_name.strip()
        if attr == "":
            return False

        # reject reserved attributes (use stripped name)
        if attr in RESERVED_ONTOLOGY_ATTRIBUTES:
            return False

        # reject consolidated attributes
        if attr.endswith(CONSOLIDATION_SUFFIX):
            return False

        # if we reached this point, the attribute is not reserved and not consolidated
        return True

    def _is_not_reserved_and_consolidated_attribute(self, attr_name: str) -> bool:
        """Check if an attribute name is not reserved and is consolidated.

        Private method - not used externally in the codebase.

        Args:
            attr_name: The attribute name to check

        Returns:
            True if the attribute is not reserved and is consolidated, False otherwise
        """
        # reject non-strings or empty names
        if not isinstance(attr_name, str):
            return False
        attr = attr_name.strip()
        if attr == "":
            return False

        # reject reserved attributes (use stripped name)
        if attr in RESERVED_ONTOLOGY_ATTRIBUTES:
            return False

        # accept only consolidated attributes
        if not attr.endswith(CONSOLIDATION_SUFFIX):
            return False

        # if we reached this point, the attribute is not reserved and is consolidated
        return True

    def _get_ontology_as_string(self, ontology_id: str, ontology_description: str, include_attributes: bool = True) -> str:
        """Get ontology as formatted string"""
        from src.core.base.InputOutput import execute_neo4j_query
        
        rules = ""

        # copy all the EntityOntology nodes, which represent the rules for creating Entity nodes
        rules += "\n## Ontology Entity Definitions:\n"
        query = f"""MATCH (o:Ontology {{id: $ontology_id}})-[:CONTAINS]->(e:EntityOntology) RETURN e"""
        parameters = {"ontology_id": ontology_id}
        result = execute_neo4j_query(query, parameters)
        if result is None:
            raise Exception("2: Query returned None")
        for record in result:
            labels = list(record['e'].labels)
            labels = [label for label in labels if label not in RESERVED_ONTOLOGY_LABELS]
            label = labels[0] if len(labels) > 0 else None
            # Fallback: use the node's id if no label found after filtering
            if label is None and 'id' in record['e']:
                label = record['e']['id']
            description = record['e'][ATTRIBUTE_ENTITY_DESCRIPTION]
            desc_escaped = (description or "").replace('"', "'")
            rules += f'\n<ENTITY id="{label}" description="{desc_escaped}">\n'
            if include_attributes:
                skip_keys = set(RESERVED_ONTOLOGY_ATTRIBUTES) | {'label', 'id'}
                for key in record['e'].keys():
                    if key in skip_keys:
                        continue
                    value = record['e'][key]
                    if value is None:
                        continue
                    value_str = str(value)
                    attr_type = None
                    type_match = _re_type_hint.match(value_str)
                    if type_match:
                        attr_type = type_match.group(1).strip()
                        value_str = value_str[type_match.end():].strip()
                    type_attr = f' type="{attr_type}"' if attr_type else ''
                    rules += f'  <ATTRIBUTE id="{key}"{type_attr}>{value_str}</ATTRIBUTE>\n'
            rules += '</ENTITY>\n'

        # copy all the RelationshipOntology nodes, which represent the rules for creating neo4j relationships
        rules += "\n## Ontology Relationship Definitions:\n"
        query = f"""MATCH (o:Ontology {{id: $ontology_id}})-[:CONTAINS]->(r:RelationshipOntology) RETURN r"""
        parameters = {"ontology_id": ontology_id}
        result = execute_neo4j_query(query, parameters)
        if result is None:
            raise Exception("3: Query returned None")
        for record in result:
            labels = list(record['r'].labels)
            labels = [label for label in labels if label not in RESERVED_ONTOLOGY_LABELS]
            label = labels[0] if len(labels) > 0 else None
            # Fallback: if no label found after filtering, use the relationship node's id
            if label is None and 'id' in record['r']:
                label = record['r']['id']
            description = record['r'][ATTRIBUTE_ENTITY_DESCRIPTION]
            entity_source_label = record['r'][ATTRIBUTE_ENTITY_SOURCE_LABEL]
            entity_target_label = record['r'][ATTRIBUTE_ENTITY_TARGET_LABEL]

            # Determine start/end entity labels for XML attributes
            if entity_source_label == "--" and entity_target_label == "--":
                start_entities = "*"
                end_entities = "*"
            elif entity_source_label == "--":
                start_entities = entity_target_label or "*"
                end_entities = entity_target_label or "*"
            elif entity_target_label == "--":
                start_entities = entity_source_label or "*"
                end_entities = entity_source_label or "*"
            else:
                start_entities = entity_source_label or "*"
                end_entities = entity_target_label or "*"

            # Escape description for XML attribute
            desc_escaped = (description or "").replace('"', "'")
            rules += f'\n<RELATIONSHIP id="{label}" start_entities="{start_entities}" end_entities="{end_entities}" description="{desc_escaped}">\n'

            if include_attributes:
                # Output each non-reserved attribute as an XML <ATTRIBUTE> element
                # Skip entity_description, entity_source_labels, entity_target_labels (already in the tag above)
                skip_keys = set(RESERVED_ONTOLOGY_ATTRIBUTES) | {'label', 'id'}
                for key in record['r'].keys():
                    if key in skip_keys:
                        continue
                    value = record['r'][key]
                    if value is None:
                        continue
                    value_str = str(value)
                    # Extract optional type hint from value if it starts with "(type)"
                    attr_type = None
                    type_match = _re_type_hint.match(value_str)
                    if type_match:
                        attr_type = type_match.group(1).strip()
                        value_str = value_str[type_match.end():].strip()
                    type_attr = f' type="{attr_type}"' if attr_type else ''
                    rules += f'  <ATTRIBUTE id="{key}"{type_attr}>{value_str}</ATTRIBUTE>\n'

            rules += '</RELATIONSHIP>\n'

        return rules

    def get_entities_as_string(self, community_id: str, entity_ids: list = None, include_relationships: bool = False, omit_headers: bool = False) -> str:
        """
        Returns a string representation of entities filtered by community_id and/or entity_ids.
        If entity_ids is provided and community_id is None, returns entities by id only.
        If both are provided, returns entities in the community whose id is in entity_ids.
        
        Args:
            community_id (str): ID of the community to filter entities by
            entity_ids (List[str], optional): Specific entity IDs to include
            include_relationships (bool, optional): Whether to include relationship information
            omit_headers (bool, optional): When True, suppresses the '## ...' section header
                lines so the returned string contains only entity content (useful for
                downstream consumers such as rerankers that should not receive metadata headers).
            
        Returns:
            str: Formatted string representation of entities and their relationships
        """
        import json
        from src.core.base.InputOutput import execute_neo4j_query
        
        def _build_entity_query():
            """Helper function to build the main entity query based on parameters"""
            if community_id is not None:
                if entity_ids is not None:
                    query = """MATCH (c:Community {id: $community_id})-[:AGGREGATES]->(e:Entity) 
                              WHERE e.id IN $entity_ids 
                              RETURN e ORDER BY e.id"""
                    parameters = {"community_id": community_id, "entity_ids": entity_ids}
                else:
                    query = """MATCH (c:Community {id: $community_id})-[:AGGREGATES]->(e:Entity) 
                              RETURN e ORDER BY e.id"""
                    parameters = {"community_id": community_id}
            elif entity_ids is not None:
                query = """MATCH (e:Entity) WHERE e.id IN $entity_ids 
                          RETURN e ORDER BY e.id"""
                parameters = {"entity_ids": entity_ids}
            else:
                return None, None
            return query, parameters

        def _get_all_relationships():
            """Helper function to get all relationships for the filtered entities in one query"""
            relationships_by_entity = {}
            
            if include_relationships:
                # Build relationship query based on the same filtering logic
                if community_id is not None:
                    if entity_ids is not None:
                        # Get relationships for specific entities within a community
                        # collect lists: [rel_label, node_id, node_name, labels(node)] so we can extract label and id in Python
                        rel_query = """
                            MATCH (c:Community {id: $community_id})-[:AGGREGATES]->(e:Entity)
                            WHERE e.id IN $entity_ids
                            OPTIONAL MATCH (e)-[r_out]->(target:Entity)
                            OPTIONAL MATCH (source:Entity)-[r_in]->(e)
                            WITH e.id as entity_id,
                                 collect(DISTINCT [type(r_out), target.id, COALESCE(target.name_consolidated, 'unnamed'), labels(target)]) as outgoing,
                                 collect(DISTINCT [type(r_in), source.id, COALESCE(source.name_consolidated, 'unnamed'), labels(source)]) as incoming
                            RETURN entity_id, outgoing, incoming
                            ORDER BY entity_id
                        """
                        parameters = {"community_id": community_id, "entity_ids": entity_ids}
                    else:
                        # Get relationships for all entities in a community
                        rel_query = """
                            MATCH (c:Community {id: $community_id})-[:AGGREGATES]->(e:Entity)
                            OPTIONAL MATCH (e)-[r_out]->(target:Entity)
                            OPTIONAL MATCH (source:Entity)-[r_in]->(e)
                            WITH e.id as entity_id,
                                 collect(DISTINCT [type(r_out), target.id, COALESCE(target.name_consolidated, 'unnamed'), labels(target)]) as outgoing,
                                 collect(DISTINCT [type(r_in), source.id, COALESCE(source.name_consolidated, 'unnamed'), labels(source)]) as incoming
                            RETURN entity_id, outgoing, incoming
                            ORDER BY entity_id
                        """
                        parameters = {"community_id": community_id}
                elif entity_ids is not None:
                    # Get relationships for specific entities (no community filter)
                    rel_query = """
                        MATCH (e:Entity) WHERE e.id IN $entity_ids
                        OPTIONAL MATCH (e)-[r_out]->(target:Entity)
                        OPTIONAL MATCH (source:Entity)-[r_in]->(e)
                        WITH e.id as entity_id,
                             collect(DISTINCT [type(r_out), target.id, COALESCE(target.name_consolidated, 'unnamed'), labels(target)]) as outgoing,
                             collect(DISTINCT [type(r_in), source.id, COALESCE(source.name_consolidated, 'unnamed'), labels(source)]) as incoming
                        RETURN entity_id, outgoing, incoming
                        ORDER BY entity_id
                    """
                    parameters = {"entity_ids": entity_ids}
                else:
                    return relationships_by_entity

                rel_result = execute_neo4j_query(rel_query, parameters)
                if rel_result:
                    def _filter_rel_list(rel_list):
                        """Normalize and filter a collected relationship list returned by Cypher.

                        Each element is expected to be a list/array with the relationship label
                        in position 0. We keep only entries whose label is not None.
                        """
                        if not rel_list:
                            return []
                        filtered = []
                        for rel in rel_list:
                            try:
                                # rel is expected to be a sequence: [rel_label, node_id, node_name, node_labels]
                                if rel and len(rel) > 0 and rel[0] is not None:
                                    filtered.append(rel)
                            except Exception:
                                # Be permissive: skip malformed entries
                                continue
                        return filtered

                    for record in rel_result:
                        entity_id = record["entity_id"]
                        relationships_by_entity[entity_id] = {
                            "outgoing": _filter_rel_list(record.get("outgoing", [])),
                            "incoming": _filter_rel_list(record.get("incoming", []))
                        }
            
            return relationships_by_entity

        # Set up the header message
        if community_id is not None:
            entities = "" if omit_headers else f"## Community Entities associated to community_id='{community_id}':\n"
        elif entity_ids is not None:
            entities = "" if omit_headers else f"## Entities with ids in {entity_ids}:\n"
        else:
            return "" if omit_headers else "## No community_id or entity_ids provided.\n"

        # Build and execute the main query
        query, parameters = _build_entity_query()
        if query is None:
            return entities

        # Get all relationships in one query if needed
        relationships_by_entity = _get_all_relationships()

        result = execute_neo4j_query(query, parameters)
        if result:
            for record in result:
                labels = list(record['e'].labels)
                labels = [label for label in labels if label not in RESERVED_ONTOLOGY_LABELS]
                label = labels[0] if len(labels) > 0 else None
                entity_id = record['e']['id']
                requires_consolidation = record['e'].get(ATTRIBUTE_REQUIRES_CONSOLIDATION, False)
                
                # Build entity attributes
                entities += f"Entity '{label}' {{id:'{entity_id}'}}: attributes=" + "{"
                
                # Process attributes
                attribute_pairs = []
                for key in record['e'].keys():
                    if not requires_consolidation:
                        # take all consolidated attributes
                        if key not in RESERVED_ONTOLOGY_ATTRIBUTES and key.endswith(CONSOLIDATION_SUFFIX):
                            clean_key = key.replace(CONSOLIDATION_SUFFIX, '')
                            value = record['e'][key]
                            # Normalize newlines in consolidated attribute values
                            normalized_value = str(value).replace('\n', ' ').replace('\r', ' ')
                            attribute_pairs.append(f"'{clean_key}'=`{normalized_value}`")
                    else:
                        # this entity requires consolidation: take all attributes that are not consolidated
                        if key not in RESERVED_ONTOLOGY_ATTRIBUTES and not key.endswith(CONSOLIDATION_SUFFIX):
                            try:
                                values_with_sources = json.loads(record['e'][key])
                                values = [value_dict['value'] for value_dict in values_with_sources]
                                # Replace newlines with spaces in each value to ensure clean single-line output
                                values = [str(v).replace('\n', ' ').replace('\r', ' ') for v in values]
                                attribute_pairs.append(f"'{key}'=`{values}`")
                            except (json.JSONDecodeError, TypeError, KeyError):
                                # Handle malformed data gracefully
                                attribute_pairs.append(f"'{key}'=`{record['e'][key]}`")
                
                entities += ", ".join(attribute_pairs) + "}"

                # Add relationship information if requested
                if include_relationships and entity_id in relationships_by_entity:
                    relationships_info = relationships_by_entity[entity_id]
                    
                    # Add outgoing relationships (multi-line)
                    if relationships_info["outgoing"]:
                        entities += "\noutgoing_relationships:\n"
                        for rel in relationships_info["outgoing"]:
                            # rel is [rel_label, node_id, node_name, node_labels]
                            rel_label = rel[0]
                            node_id = rel[1]
                            node_name = rel[2]
                            node_labels = rel[3] if len(rel) > 3 else []
                            # pick first non-reserved label
                            neighbor_label = None
                            for lab in node_labels:
                                if lab not in RESERVED_ONTOLOGY_LABELS:
                                    neighbor_label = lab
                                    break
                            neighbor_label = neighbor_label if neighbor_label is not None else (node_labels[0] if node_labels else 'Entity')
                            safe_node_name = str(node_name).replace('"', '\\"').replace('\n', ' ').replace('\r', ' ') if node_name is not None else ''
                            # Format as: (current)->[:rel_label]->(neighbor {id, name})
                            entities += f" - (e:{label} {{id:\"{entity_id}\"}})-[:{rel_label}]->(e:{neighbor_label} {{id:\"{node_id}\", name:\"{safe_node_name}\"}})\n"

                    # Add incoming relationships (multi-line)
                    if relationships_info["incoming"]:
                        entities += "incoming_relationships:\n"
                        for rel in relationships_info["incoming"]:
                            # rel is [rel_label, node_id, node_name, node_labels]
                            rel_label = rel[0]
                            node_id = rel[1]
                            node_name = rel[2]
                            node_labels = rel[3] if len(rel) > 3 else []
                            neighbor_label = None
                            for lab in node_labels:
                                if lab not in RESERVED_ONTOLOGY_LABELS:
                                    neighbor_label = lab
                                    break
                            neighbor_label = neighbor_label if neighbor_label is not None else (node_labels[0] if node_labels else 'Entity')
                            safe_node_name = str(node_name).replace('"', '\\"').replace('\n', ' ').replace('\r', ' ') if node_name is not None else ''
                            # Format as: (neighbor {id, name})-[:rel_label]->(current)
                            entities += f" - (e:{neighbor_label} {{id:\"{node_id}\", name:\"{safe_node_name}\"}})-[:{rel_label}]->(e:{label} {{id:\"{entity_id}\"}})\n"
                
                entities += ".\n"

        return entities

    def get_entities(self, community_id: str, label_filter_regex: str = None, query_where_clause: str = "") -> list:
        """Get entities filtered by community_id and optional label regex"""
        import re
        from src.core.base.InputOutput import execute_neo4j_query
        
        entities = []

        query = f"""MATCH (c:Community {{id: $community_id}})-[:AGGREGATES]->(e:Entity) {query_where_clause}
            WITH e, labels(e) AS e_labels,
                 [k IN keys(e) WHERE k ENDS WITH '_consolidated' AND NOT k IN $reserved_attrs] AS ck,
                 [k IN keys(e) WHERE k ENDS WITH '_consolidated' AND NOT k IN $reserved_attrs | e[k]] AS cv
            RETURN e.id AS eid, e_labels, ck, cv"""
        parameters = {"community_id": community_id, "reserved_attrs": RESERVED_ONTOLOGY_ATTRIBUTES}
        result = execute_neo4j_query(query, parameters)
        if result:
            for record in result:
                raw_labels = record['e_labels'] or []
                filtered_labels = [lb for lb in raw_labels if lb not in RESERVED_ONTOLOGY_LABELS]
                label = filtered_labels[0] if filtered_labels else None
                matches = True
                if label_filter_regex is not None and len(label_filter_regex) > 0:
                    if not re.match(label_filter_regex, label, re.IGNORECASE):
                        matches = False

                if matches:
                    entity = {"id": record['eid'], "label": label}
                    for key, value in zip(record['ck'] or [], record['cv'] or []):
                        entity[key.replace(CONSOLIDATION_SUFFIX, '')] = value
                    entities.append(entity)
        
        return entities

    def get_entities_by_ids(self, entity_ids: List[str], include_relationships: bool = False, effective_user_id: Optional[str] = None) -> Dict[str, dict]:
        """
        Return a dict mapping entity_id -> entity dict for the given entity_ids.
        Re-uses the same consolidated-attribute logic as get_entities() and, if requested,
        fetches incoming/outgoing relationships in one additional query and attaches them
        as lists of dicts with keys: 'label' (node label), 'id' and 'name' (name_consolidated).

        When effective_user_id is provided, only entities whose parent Ontology
        (Ontology -[:DEFINES]-> Community -[:AGGREGATES]-> Entity) the effective user holds
        GRANT:READ on are returned. Entities with no connected Ontology are included
        (no access-control boundary to enforce). When effective_user_id is None ACL is skipped.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        entities_by_id: Dict[str, dict] = {}

        if not entity_ids:
            return entities_by_id

        # 1) fetch entities — with optional ACL filter on the parent Ontology
        if effective_user_id is not None:
            acl_block = acl_ontology_where_clause("o")
            query = (
                "MATCH (e:Entity) WHERE e.id IN $entity_ids\n"
                "OPTIONAL MATCH (o:Ontology)-[:DEFINES]->(:Community)-[:AGGREGATES]->(e)\n"
                "WITH e, o\n"
                "WHERE o IS NULL\n"
                "   OR (o IS NOT NULL"
                + acl_block +
                "\n   )\nRETURN e ORDER BY e.id"
            )
            parameters: dict = {"entity_ids": entity_ids, "effectiveUserId": effective_user_id}
        else:
            query = "MATCH (e:Entity) WHERE e.id IN $entity_ids RETURN e ORDER BY e.id"
            parameters = {"entity_ids": entity_ids}

        result = execute_neo4j_query(query, parameters)
        if result is None:
            return entities_by_id

        for record in result:
            node = record['e']
            labels = list(node.labels)
            labels = [l for l in labels if l not in RESERVED_ONTOLOGY_LABELS]
            label = labels[0] if len(labels) > 0 else None
            entity_id = node['id']
            entity = {"id": entity_id, "label": label}

            # include consolidated attributes only (same behaviour as get_entities)
            for key in node.keys():
                if key not in RESERVED_ONTOLOGY_ATTRIBUTES and key.endswith(CONSOLIDATION_SUFFIX):
                    entity[key.replace(CONSOLIDATION_SUFFIX, '')] = node[key]

            # prepare empty relationships lists; will be filled if include_relationships==True
            entity["outgoing_relationships"] = []
            entity["incoming_relationships"] = []

            entities_by_id[entity_id] = entity

        # 2) fetch relationships for all the requested entities in one query (if requested)
        # Use only the ACL-permitted entity IDs collected in step 1.
        permitted_entity_ids = list(entities_by_id.keys())
        if include_relationships and permitted_entity_ids:
            rel_query = """
                MATCH (e:Entity) WHERE e.id IN $entity_ids
                OPTIONAL MATCH (e)-[r_out]->(target:Entity)
                OPTIONAL MATCH (source:Entity)-[r_in]->(e)
                WITH e.id as entity_id,
                     collect(DISTINCT [type(r_out), target.id, COALESCE(target.name_consolidated, null), labels(target), properties(r_out)]) as outgoing,
                     collect(DISTINCT [type(r_in), source.id, COALESCE(source.name_consolidated, null), labels(source), properties(r_in)]) as incoming
                RETURN entity_id, outgoing, incoming
                ORDER BY entity_id
            """
            parameters = {"entity_ids": permitted_entity_ids}
            rel_result = execute_neo4j_query(rel_query, parameters)

            def _filter_rel_list(rel_list):
                if not rel_list:
                    return []
                filtered = []
                for rel in rel_list:
                    try:
                        if rel and len(rel) > 0 and rel[0] is not None:
                            filtered.append(rel)
                    except Exception:
                        continue
                return filtered

            for record in rel_result or []:
                entity_id = record["entity_id"]
                outgoing = _filter_rel_list(record.get("outgoing", []))
                incoming = _filter_rel_list(record.get("incoming", []))

                # ensure entity exists in map (it should, but be permissive)
                if entity_id not in entities_by_id:
                    entities_by_id[entity_id] = {"id": entity_id, "label": None, "outgoing_relationships": [], "incoming_relationships": []}

                def _pick_neighbor_label(node_labels):
                    if not node_labels:
                        return "Entity"
                    for lab in node_labels:
                        if lab not in RESERVED_ONTOLOGY_LABELS:
                            return lab
                    return node_labels[0]

                # transform outgoing into list of dicts {label, id, name}
                out_list = []
                for rel in outgoing:
                    # rel is [rel_label, node_id, node_name, node_labels, rel_props]
                    relationship_label = rel[0] if len(rel) > 0 else None
                    node_id = rel[1] if len(rel) > 1 else None
                    node_name = rel[2] if len(rel) > 2 else None
                    # Normalize newlines in node name
                    if node_name:
                        node_name = str(node_name).replace('\n', ' ').replace('\r', ' ')
                    node_labels = rel[3] if len(rel) > 3 else []
                    rel_props_raw = dict(rel[4]) if len(rel) > 4 and rel[4] else {}
                    rel_attributes = {k.replace(CONSOLIDATION_SUFFIX, ""): v for k, v in rel_props_raw.items()
                                      if k not in RESERVED_ONTOLOGY_ATTRIBUTES and k.endswith(CONSOLIDATION_SUFFIX)}
                    neighbor_label = _pick_neighbor_label(node_labels)
                    out_list.append({"relationship_label": relationship_label, "entity_label": neighbor_label, "id": node_id, "entity_name": node_name, "rel_attributes": rel_attributes})
                entities_by_id[entity_id]["outgoing_relationships"] = out_list

                in_list = []
                for rel in incoming:
                    relationship_label = rel[0] if len(rel) > 0 else None
                    node_id = rel[1] if len(rel) > 1 else None
                    node_name = rel[2] if len(rel) > 2 else None
                    # Normalize newlines in node name
                    if node_name:
                        node_name = str(node_name).replace('\n', ' ').replace('\r', ' ')
                    node_labels = rel[3] if len(rel) > 3 else []
                    rel_props_raw = dict(rel[4]) if len(rel) > 4 and rel[4] else {}
                    rel_attributes = {k.replace(CONSOLIDATION_SUFFIX, ""): v for k, v in rel_props_raw.items()
                                      if k not in RESERVED_ONTOLOGY_ATTRIBUTES and k.endswith(CONSOLIDATION_SUFFIX)}
                    neighbor_label = _pick_neighbor_label(node_labels)
                    in_list.append({"relationship_label": relationship_label, "entity_label": neighbor_label, "id": node_id, "entity_name": node_name, "rel_attributes": rel_attributes})
                entities_by_id[entity_id]["incoming_relationships"] = in_list

        return entities_by_id

    def expand_entities_with_neighbors(
        self,
        entity_weights: Dict[str, float],
        min_rank: Optional[float] = None,
        neighbor_weight_factor: Optional[float] = None,
        effective_user_id: Optional[str] = None
    ) -> Dict[str, float]:
        """Broaden a kg entity_id -> weight mapping with adjacent (neighbour) entities."""
        from src.config.constants import KG_NEIGHBOR_EXPANSION_MIN_RANK, KG_NEIGHBOR_WEIGHT_FACTOR
        min_rank = KG_NEIGHBOR_EXPANSION_MIN_RANK if min_rank is None else min_rank
        neighbor_weight_factor = KG_NEIGHBOR_WEIGHT_FACTOR if neighbor_weight_factor is None else neighbor_weight_factor
        expanded_weights: Dict[str, float] = dict(entity_weights) if entity_weights else {}

        if neighbor_weight_factor > 0.0:
            source_entity_ids = [entity_id for entity_id, weight in expanded_weights.items() if weight > min_rank]
            if source_entity_ids:
                entities_by_id = self.get_entities_by_ids(source_entity_ids, include_relationships=True, effective_user_id=effective_user_id)

                for source_entity_id in source_entity_ids:
                    source_entity = entities_by_id.get(source_entity_id)
                    if not source_entity:
                        continue
                    source_weight = expanded_weights[source_entity_id]
                    neighbor_weight = source_weight * neighbor_weight_factor
                    outgoing_neighbors = source_entity.get("outgoing_relationships", [])
                    incoming_neighbors = source_entity.get("incoming_relationships", [])
                    neighbors = outgoing_neighbors + incoming_neighbors
                    for neighbor in neighbors:
                        neighbor_id = neighbor.get("id")
                        if not neighbor_id or neighbor_id in entity_weights:
                            continue
                        if neighbor_id not in expanded_weights or neighbor_weight > expanded_weights[neighbor_id]:
                            expanded_weights[neighbor_id] = neighbor_weight

        log(INFO, f"{self._log_prefix}.expand_entities_with_neighbors: original count={len(entity_weights)}, extended count={len(expanded_weights)}")
        return expanded_weights

    def _get_relationships_as_string(self, community_id: str) -> str:
        """Get relationships for a community as formatted string.
        
        Private method - only used internally within Ontology.py.
        """
        import json
        from src.core.base.InputOutput import execute_neo4j_query
        
        relationships = f"## Community Relationships that entitites associated to community_id='{community_id}' are using:\n"

        query = f"""MATCH (c:Community {{id: $community_id}})-[:AGGREGATES]->(e1:Entity)-[r]->(e2:Entity)
            RETURN type(r) AS r_type,
                   e1.id AS id_e1, labels(e1) AS labels_e1,
                   e2.id AS id_e2, labels(e2) AS labels_e2,
                   r[$req_consol] AS r_requires_consolidation,
                   [k IN keys(r) WHERE NOT k IN $reserved_attrs] AS rk,
                   [k IN keys(r) WHERE NOT k IN $reserved_attrs | r[k]] AS rv"""
        parameters = {"community_id": community_id,
                      "req_consol": ATTRIBUTE_REQUIRES_CONSOLIDATION,
                      "reserved_attrs": RESERVED_ONTOLOGY_ATTRIBUTES}
        result = execute_neo4j_query(query, parameters)
        if result:
            for record in result:
                label = record["r_type"]
                raw_labels_e1 = record['labels_e1'] or []
                labels_e1 = [lb for lb in raw_labels_e1 if lb not in ['Ontology', 'EntityOntology', 'RelationshipOntology', 'Entity']]
                label_e1 = labels_e1[0] if labels_e1 else None
                raw_labels_e2 = record['labels_e2'] or []
                labels_e2 = [lb for lb in raw_labels_e2 if lb not in ['Ontology', 'EntityOntology', 'RelationshipOntology', 'Entity']]
                label_e2 = labels_e2[0] if labels_e2 else None
                id_e1 = record['id_e1']
                id_e2 = record['id_e2']
                requires_consolidation = record['r_requires_consolidation'] or False
                # rebuild relationship props dict from parallel key/value lists
                r_props = dict(zip(record['rk'] or [], record['rv'] or []))

                relationships += f"Relationship ({id_e1})-[:'{label}']->({id_e2}). Relationship attributes=" + "{"
                for key, value in r_props.items():
                    if not requires_consolidation:
                        # take all consolidated attributes. For example:
                        # "description_consolidated": "John is a developer and an artist."
                        # "age_consolidated": 20
                        if key.endswith(CONSOLIDATION_SUFFIX):
                            # Normalize newlines in consolidated relationship attribute values
                            normalized_value = str(value).replace('\n', ' ').replace('\r', ' ')
                            relationships += f"'{key.replace(CONSOLIDATION_SUFFIX, '')}'=`{normalized_value}`, "
                    else:
                        # this entity requires consolidation: take all attributes that are not consolidated. For example:
                        # "description": [{"value": "John is a developer.", "source_id": "docid-19"}, {"value": "John is an artist.", "source_id": "docid-19"}]
                        # information to be added will exclude the sources, and for this example is:
                        # "description": ["John is a developer.", "John is an artist."]
                        if not key.endswith(CONSOLIDATION_SUFFIX):
                            values_with_sources = json.loads(value)
                            values = [value_dict['value'] for value_dict in values_with_sources]
                            # Replace newlines with spaces in each value to ensure clean single-line output
                            values = [str(v).replace('\n', ' ').replace('\r', ' ') for v in values]
                            relationships += f"'{key}'=`{values}`, "
                relationships = relationships[:-2] + "}.\n"
        
        return relationships

    def get_ontology_entities(self, ontology_id: str) -> List[dict]:
        """Get entity ontology definitions for the given ontology_id."""
        from src.core.base.InputOutput import execute_neo4j_query
        
        entities = []

        if ontology_id is None:
            query = """MATCH (o:Ontology)-[:CONTAINS]->(e:EntityOntology) RETURN o, e"""
            parameters = {}
        else:
            query = """MATCH (o:Ontology {id: $ontology_id})-[:CONTAINS]->(e:EntityOntology) RETURN o, e"""
            parameters = {"ontology_id": ontology_id}
        result = execute_neo4j_query(query, parameters)

        if result:
            for record in result:
                labels = list(record['e'].labels)
                labels = [label for label in labels if label not in RESERVED_ONTOLOGY_LABELS]
                label = labels[0] if len(labels) > 0 else None
                id = record['e']['id']
                entity = {
                    "ontology_id": record['o']['id'],
                    "id": id,
                    "label": label,
                    ATTRIBUTE_ENTITY_DESCRIPTION: record['e'][ATTRIBUTE_ENTITY_DESCRIPTION] if ATTRIBUTE_ENTITY_DESCRIPTION in record['e'] else None}
                for key in record['e'].keys():
                    if key not in RESERVED_ONTOLOGY_ATTRIBUTES and not key.endswith(CONSOLIDATION_SUFFIX):
                        entity[key] = record['e'][key]
                entities.append(entity)

        return entities

    def ontology_exists(self, ontology_id: str) -> bool:
        """Check whether an ontology with the given id exists."""
        from src.core.base.InputOutput import execute_neo4j_query
        try:
            query = "MATCH (o:Ontology {id: $id}) RETURN count(o) > 0 AS exists"
            params = {"id": ontology_id}
            res = execute_neo4j_query(query, params) or []
            if not res:
                return False
            return bool(res[0].get('exists', False))
        
        except Exception as e:
            log(ERROR, f"{self._log_prefix}: Error in ontology_exists('{ontology_id}'): {e}")
            return False

    def get_ontology_keyword_categories_count(self, ontology_id: str) -> int:
        """Return the number of KeywordCategory nodes contained in the given ontology."""
        from src.core.base.InputOutput import execute_neo4j_query
        try:
            query = """
                MATCH (o:Ontology {id: $ontology_id})-[:CONTAINS]->(kc:KeywordCategory)
                RETURN count(kc) AS cnt
            """
            records = execute_neo4j_query(query, {"ontology_id": ontology_id})
            if records and len(records) > 0:
                return int(records[0].get("cnt", 0))
            
        except Exception as e:
            log(WARNING, f"{self._log_prefix}: Error in get_ontology_keyword_categories_count('{ontology_id}'): {e}")
        return 0

    def get_ontology_relationships(self, ontology_id: str) -> List[dict]:
        """Get relationship ontology definitions for the given ontology_id."""
        from src.core.base.InputOutput import execute_neo4j_query
        
        relationships = []

        if ontology_id is None:
            query = """MATCH (o:Ontology)-[:CONTAINS]->(r:RelationshipOntology) RETURN o, r"""
            parameters = {}
        else:
            query = """MATCH (o:Ontology {id: $ontology_id})-[:CONTAINS]->(r:RelationshipOntology) RETURN o, r"""
            parameters = {"ontology_id": ontology_id}
        result = execute_neo4j_query(query, parameters)

        if result:
            for record in result:
                labels = list(record['r'].labels)
                labels = [label for label in labels if label not in RESERVED_ONTOLOGY_LABELS]
                label = labels[0] if len(labels) > 0 else None
                # Fallback: if no label found after filtering, use the relationship node's id
                if label is None and 'id' in record['r']:
                    label = record['r']['id']
                id = record['r']['id']
                relationship = {
                    "ontology_id": record['o']['id'],
                    "id": id,
                    "label": label,
                    ATTRIBUTE_ENTITY_DESCRIPTION: record['r'][ATTRIBUTE_ENTITY_DESCRIPTION] if ATTRIBUTE_ENTITY_DESCRIPTION in record['r'] else None}
                for key in record['r'].keys():
                    if key not in RESERVED_ONTOLOGY_ATTRIBUTES and key.endswith(CONSOLIDATION_SUFFIX):
                        relationship[key.replace(CONSOLIDATION_SUFFIX, '')] = record['r'][key]
                relationships.append(relationship)

        return relationships

    def get_ontology_communities(self, ontology_id: str) -> List[str]:
        """Return community IDs linked to the given ontology, sorted alphabetically."""
        from src.core.base.InputOutput import execute_neo4j_query
        try:
            query = "MATCH (o:Ontology {id: $ontology_id})-[:DEFINES]->(c:Community) RETURN c.id AS id ORDER BY c.id"
            rows = execute_neo4j_query(query, {"ontology_id": ontology_id}) or []
            return [row["id"] for row in rows if row.get("id")]
        except Exception as e:
            log(WARNING, f"{self._log_prefix}: Error in get_ontology_communities('{ontology_id}'): {e}")
            return []

    def validate_community_ids(self, community_ids: Optional[List[str]], ontology_id: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """Return unique valid community IDs and IDs that are not available."""
        valid_community_ids: List[str] = []
        invalid_community_ids: List[str] = []

        if community_ids:
            if ontology_id:
                available_community_ids = self.get_ontology_communities(ontology_id)
            else:
                query = "MATCH (c:Community) RETURN DISTINCT c.id AS id ORDER BY c.id"
                rows = execute_neo4j_query(query) or []
                available_community_ids = [row["id"] for row in rows if row.get("id")]

            for community_id in community_ids:
                if community_id in available_community_ids:
                    if community_id not in valid_community_ids:
                        valid_community_ids.append(community_id)
                elif community_id not in invalid_community_ids:
                    invalid_community_ids.append(community_id)

        return valid_community_ids, invalid_community_ids

    def get_ontology_and_entites_as_string(self, goal, ontology_id: str, community_ids: List[str], skip_details: bool, condense_categories: bool = False) -> str:
        """Get the rules for the ontology and all entities and relationships in the specified communities.

        Args:
            goal: The current goal (not used at present).
            ontology_id: The ID of the ontology. Special values:
                - ONTOLOGY_ID_ALL ('__ALL__'): Returns all ontologies (used by AgentOntology for selection)
                - None or the default ontology id (from get_default_ontology_id()): Returns None (no specific ontology/categories)
                - Specific ID: Returns that specific ontology
            community_ids: The IDs of the communities. If None, all communities linked to the ontology are considered.
            skip_details: If True, only ontology and community descriptions are returned (no entities/relationships).
            condense_categories: If True, output categories as comma-separated list only (e.g., <CATEGORIES>CATEGORY1,CATEGORY2</CATEGORIES>). If False, output verbose format with descriptions and values.

        Returns:
            Formatted string with ontology information, or None if no specific ontology is requested.
        """
        import traceback
        from src.core.base.InputOutput import execute_neo4j_query
        from src.utils.log import log, ERROR
        
        rules = None
        if skip_details is None:
            skip_details = community_ids is None

        try:
            # None or the default ontology means no specific ontology/categories - return None
            if ontology_id is None or ontology_id == get_default_ontology_id():
                return None
            
            if ontology_id == ONTOLOGY_ID_ALL:
                # ONTOLOGY_ID_ALL means "all ontologies" - used by AgentOntology for selection
                query = f"""MATCH (o:Ontology) RETURN o.id as id, o.name as name, o.description as description"""
            else:
                query = f"""MATCH (o:Ontology {{id: $ontology_id}}) RETURN o.id as id, o.name as name, o.description as description"""
            parameters = {"ontology_id": ontology_id}
            ontologies = execute_neo4j_query(query, parameters)
            if ontologies and len(ontologies) > 0:
                rules = ""
                for ontology in ontologies:
                    o_id = ontology['id']
                    o_desc = ontology['description']

                    if skip_details:
                        rules += f'<ONTOLOGY id="{o_id}" description="{o_desc}">\n'
                        
                        # include KeywordCategory nodes associated with this ontology
                        query = f"""
                            MATCH (o:Ontology {{id: $ontology_id}})-[:CONTAINS]->(k:KeywordCategory)
                            RETURN k.id as id,
                                   k.text_en as text_en,
                                   k.lov as lov,
                                   k.description as description
                            """
                        parameters = {"ontology_id": o_id}
                        keyword_categories = execute_neo4j_query(query, parameters)
                        if keyword_categories and len(keyword_categories) > 0:
                            if condense_categories:
                                category_names = [kc.get('text_en', 'unnamed') for kc in keyword_categories]
                                rules += f"\n<CATEGORIES>{','.join(category_names)}</CATEGORIES>\n"
                            else:
                                rules += f"\n<CATEGORIES>\nKeyword Categories in Ontology '{o_id}':\n"
                                for kc in keyword_categories:
                                    kc_text_en = kc.get('text_en', 'unnamed')
                                    kc_desc = kc.get('description', 'No description')
                                    kc_lov = kc.get('lov', None)
                                    if kc_lov is not None:
                                        rules += f"  - {kc_text_en}: {kc_desc} (List of Values: {kc_lov})\n"
                                    else:
                                        rules += f"  - {kc_text_en}: {kc_desc}\n"
                                rules += "</CATEGORIES>\n"
                        
                        # give a quick description of all communities in the ontology
                        query = f"""MATCH (o:Ontology {{id: $ontology_id}})-[:DEFINES]->(c:Community) RETURN c.id as id, c.description as description"""
                        parameters = {"ontology_id": o_id}
                        communities = execute_neo4j_query(query, parameters)
                        if communities and len(communities)>0:
                            rules += "\n<COMMUNITIES>\n"
                            for c in communities:
                                community_id = c['id']
                                community_name = community_id
                                community_description = c['description']
                                rules += f'<COMMUNITY id="{community_id}" name="{community_name}" description="{community_description}" />\n'
                            rules += "</COMMUNITIES>\n"
                    else:
                        rules += f'<ONTOLOGY id="{o_id}" description="{o_desc}">\n'
                        ontology = self._get_ontology_as_string(ontology_id=o_id, ontology_description=None, include_attributes=not skip_details)
                        rules += ontology
                        
                        # include KeywordCategory nodes associated with this ontology
                        query = f"""MATCH (o:Ontology {{id: $ontology_id}})-[:CONTAINS]->(k:KeywordCategory) RETURN k.text_en as text_en, k.description as description"""
                        parameters = {"ontology_id": o_id}
                        keyword_categories = execute_neo4j_query(query, parameters)
                        if keyword_categories and len(keyword_categories) > 0:
                            rules += f"\n<CATEGORIES>\n"
                            for kc in keyword_categories:
                                kc_text_en = kc.get('text_en', 'unnamed')
                                kc_desc = kc.get('description', None)
                                if kc_desc is not None:
                                    rules += f'<CATEGORY name:"{kc_text_en}">{kc_desc}</CATEGORY>\n'
                                else:
                                    rules += f'<CATEGORY name:"{kc_text_en}"></CATEGORY>\n'
                            rules += "</CATEGORIES>\n"

                    rules += "\n</ONTOLOGY>\n\n"

                if not skip_details:
                    if community_ids is not None:
                        rules += "<COMMUNITY_ENTITIES_AND_RELATIONSHIPS>\n"
                        for community_id in community_ids:
                            rules += f"\n\n<COMMUNITY>\n# Community '{community_id}':\n"
                            entities = self.get_entities_as_string(community_id)
                            relationships = self._get_relationships_as_string(community_id)

                            rules += f"{entities}\n{relationships}\n</COMMUNITY>\n"

                        rules += "</COMMUNITY_ENTITIES_AND_RELATIONSHIPS>\n"

        except Exception as e:
            log(ERROR, f"ERROR in get_ontology_rules: {e}\n{traceback.format_exc()}")
            ontologies = None

        return rules

    def semantic_search_entity(self, prompt: str, community_ids: List[str] = None, limit: int = None, effective_user_id: Optional[str] = None) -> dict:
        """Searches the knowledge graph for entities that match the semantic search query.

        Args:
            prompt: The search query.
            community_ids: List of community IDs to filter by (optional).
            limit: The maximum number of results to return.

        Returns:
            dict: A dictionary where the keys are entity IDs and the values are dictionaries containing:
                  - entity_id: The entity's ID
                  - elementId: Neo4j element ID
                  - label: The entity's primary label/type
                  - name: Consolidated entity name
                  - description: Consolidated entity description
                  - score: Semantic search relevance score
                  - attributes: All consolidated attributes of the entity
                  - outgoing_relationships: List of outgoing relationships with full details
                  - incoming_relationships: List of incoming relationships with full details
        """
        import traceback
        from src.config.constants import SEMANTIC_SEARCH_LIMIT, SEMANTIC_SEARCH_NEAREST_NEIGHBOURS, SEMANTIC_SEARCH_CANDIDATE_MULTIPLIER
        from src.core.base.InputOutput import execute_neo4j_query
        from src.core.llm import EmbeddingsManager
        from src.utils.log import log, DEBUG, WARNING, ERROR
        
        if limit is None:
            limit = SEMANTIC_SEARCH_LIMIT
        
        try:
            log(DEBUG, f"Starting semantic search for prompt: '{prompt}' with limit: {limit}")

            if effective_user_id is not None and community_ids:
                for community_id in community_ids:
                    if not self.check_read_access_on_community(effective_user_id, community_id):
                        log(ERROR, f"semantic_search_entity: user '{effective_user_id}' denied access to community '{community_id}'")
                        return {}
            
            embedding = EmbeddingsManager.instance().get_embeddings(prompt)
            if community_ids is not None and (len(community_ids) == 0 or (len(community_ids) == 1 and community_ids[0] is None)):
                community_ids = None

            # ensure we request at least `limit * SEMANTIC_SEARCH_CANDIDATE_MULTIPLIER` candidates
            # from the vector index to avoid the case where post-filtering/reranking would
            # cause fewer than `limit` results to be available. The multiplier is configurable
            # via SEMANTIC_SEARCH_CANDIDATE_MULTIPLIER.
            nearest_neighbours = max(SEMANTIC_SEARCH_NEAREST_NEIGHBOURS, limit * SEMANTIC_SEARCH_CANDIDATE_MULTIPLIER)

            community_match = "MATCH" if community_ids else "OPTIONAL MATCH"
            query = f"""CYPHER 25
                MATCH (e:Entity)
                  SEARCH e IN (
                    VECTOR INDEX ontology_entity
                    FOR $embedding
                    LIMIT $nearest_neighbours
                  ) SCORE AS score
                {community_match} (c:Community)-[:AGGREGATES]->(e)
                WHERE $community_ids IS NULL OR size($community_ids) = 0 OR c.id IN $community_ids
                RETURN e.id as id, elementId(e) AS elementId, e.name_consolidated as name, 
                       e.description_consolidated as description, score, e
                ORDER BY score DESC
                LIMIT $limit
                """
            parameters = { 
                "embedding": embedding,
                "nearest_neighbours": nearest_neighbours,
                "community_ids": community_ids,
                "limit": limit
            }
            communities_to_log = ", ".join(community_ids) if community_ids else "all communities"
            log(INFO, f"OntologyService semantic search query communities: {communities_to_log}")
            query_result = execute_neo4j_query(query, parameters)
            if query_result is None:
                log(WARNING, "the vector schema index 'ontology_entity' does not exist yet, we need to create it to avoid future errors")
                self.create_semantic_search_index()
                return {}

            elif len(query_result) == 0:
                log(DEBUG, "No entities found in semantic search")
                return {}

            entities = {}
            entity_ids = []
            
            # Process entities and collect their IDs
            for record in query_result:
                entity_id = record["id"]
                entity_ids.append(entity_id)
                entity_node = record["e"]
                
                # Extract entity label (consistent with other functions)
                labels = list(entity_node.labels) if hasattr(entity_node, 'labels') else []
                labels = [label for label in labels if label not in RESERVED_ONTOLOGY_LABELS]
                entity_label = labels[0] if len(labels) > 0 else None
                
                # Extract all consolidated attributes
                consolidated_attributes = {
                    key.replace(CONSOLIDATION_SUFFIX, ""): value
                    for key, value in entity_node.items()
                    if key.endswith(CONSOLIDATION_SUFFIX) and key not in RESERVED_ONTOLOGY_ATTRIBUTES
                }

                entity = {
                    "entity_id": entity_id,
                    "elementId": record["elementId"],
                    "label": entity_label,
                    "name": record["name"],
                    "description": record["description"],
                    "score": record["score"],
                    "attributes": consolidated_attributes,
                    "outgoing_relationships": [],
                    "incoming_relationships": []
                }
                entities[entity_id] = entity

            # Now fetch all relationships for these entities in a single query
            if entity_ids:
                rel_query = """
                    MATCH (e:Entity) WHERE e.id IN $entity_ids
                    OPTIONAL MATCH (e)-[r_out]->(target:Entity)
                    WITH e.id as entity_id,
                        collect(DISTINCT {
                            rel_type: type(r_out),
                            rel_props: r_out,
                            target_id: target.id,
                            target_name: COALESCE(target.name_consolidated, null),
                            target_labels: labels(target)
                        }) as outgoing
                    WITH entity_id, outgoing
                    OPTIONAL MATCH (source:Entity)-[r_in]->(e:Entity) WHERE e.id = entity_id
                    WITH entity_id, outgoing,
                        collect(DISTINCT {
                            rel_type: type(r_in),
                            rel_props: r_in,
                            source_id: source.id,
                            source_name: COALESCE(source.name_consolidated, null),
                            source_labels: labels(source)
                        }) as incoming
                    RETURN entity_id, outgoing, incoming
                    ORDER BY entity_id
                """
                rel_parameters = {"entity_ids": entity_ids}
                rel_result = execute_neo4j_query(rel_query, rel_parameters)

                def _filter_rel_list(rel_list):
                    """Filter out null relationships"""
                    if not rel_list:
                        return []
                    filtered = []
                    for rel in rel_list:
                        try:
                            if rel and rel.get("rel_type") is not None:
                                filtered.append(rel)
                        except Exception:
                            continue
                    return filtered

                def _pick_entity_label(node_labels):
                    """Extract the primary entity label, excluding reserved labels"""
                    if not node_labels:
                        return "Entity"
                    for lab in node_labels:
                        if lab not in RESERVED_ONTOLOGY_LABELS:
                            return lab
                    return node_labels[0] if node_labels else "Entity"

                def _extract_relationship_attributes(rel_props):
                    """Extract consolidated attributes from relationship properties"""
                    if not rel_props:
                        return {}
                    
                    rel_attributes = {}
                    for key, value in rel_props.items():
                        if key not in RESERVED_ONTOLOGY_ATTRIBUTES and key.endswith(CONSOLIDATION_SUFFIX):
                            rel_attributes[key.replace(CONSOLIDATION_SUFFIX, "")] = value
                    return rel_attributes

                # Process relationship results
                for record in rel_result or []:
                    entity_id = record["entity_id"]
                    if entity_id not in entities:
                        continue
                        
                    outgoing = _filter_rel_list(record.get("outgoing", []))
                    incoming = _filter_rel_list(record.get("incoming", []))

                    # Process outgoing relationships
                    for rel in outgoing:
                        # Normalize newlines in target entity name
                        target_name = rel["target_name"]
                        if target_name:
                            target_name = str(target_name).replace('\n', ' ').replace('\r', ' ')
                        rel_info = {
                            "relationship_label": rel["rel_type"],
                            "target_entity_id": rel["target_id"],
                            "target_entity_name": target_name,
                            "target_entity_label": _pick_entity_label(rel["target_labels"]),
                            "direction": "outgoing",
                            "attributes": _extract_relationship_attributes(rel["rel_props"])
                        }
                        entities[entity_id]["outgoing_relationships"].append(rel_info)

                    # Process incoming relationships
                    for rel in incoming:
                        # Normalize newlines in source entity name
                        source_name = rel["source_name"]
                        if source_name:
                            source_name = str(source_name).replace('\n', ' ').replace('\r', ' ')
                        rel_info = {
                            "relationship_label": rel["rel_type"],
                            "source_entity_id": rel["source_id"],
                            "source_entity_name": source_name,
                            "source_entity_label": _pick_entity_label(rel["source_labels"]),
                            "direction": "incoming",
                            "attributes": _extract_relationship_attributes(rel["rel_props"])
                        }
                        entities[entity_id]["incoming_relationships"].append(rel_info)

                log(DEBUG, f"Semantic search completed. Found {len(entities)} entities with relationships")
                return entities
        
        except Exception as e:
            log(ERROR, f"Error in semantic_search_entity: {e}\n{traceback.format_exc()}")
            return {}

    def get_linked_documents(self, entity_id: str, limit: int = 10, excluded_document_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """Fetch the Document nodes linked to an entity via IS_DOCUMENTED, including their full stored text.

        Args:
            entity_id: The id property of the entity node.
            limit: Maximum number of linked documents returned.
            excluded_document_ids: Optional set of Document ids already emitted elsewhere in the
                same context; linked documents with those ids are skipped to avoid duplication.

        Returns:
            List of dicts with 'id', 'title' and 'text' (the full stored Document.text) per linked document.
        """
        capped = int(limit) if limit and int(limit) > 0 else 10
        excluded: Set[str] = set(excluded_document_ids or ())
        query = f"""
            MATCH (e:Entity {{id: $entity_id}})-[:IS_DOCUMENTED]->(d:Document)
            RETURN d.id AS id, d.title AS title, d.text AS text
            LIMIT {capped}
        """

        result = execute_neo4j_query(query, {"entity_id": entity_id})

        docs: List[Dict[str, Any]] = []
        seen = set()
        for row in result or []:
            doc_id = row.get("id")
            text = row.get("text")
            if not doc_id or not text or doc_id in seen or doc_id in excluded:
                continue
            seen.add(doc_id)
            docs.append({"id": doc_id, "title": row.get("title") or "", "text": text})

        return docs

    def linked_documents_discourse(self, entity_id: str, limit: int = 10, max_chars: int = MAX_RAG_MESSAGE_LENGTH_DEFAULT, excluded_document_ids: Optional[Set[str]] = None) -> str:
        """Render linked-document texts within this entity's share of an aggregate budget.

        Each linked document is rendered as:
            <LINKED_DOCUMENT id="..." title="...">
            ```
            <full Document.text>
            ```
            </LINKED_DOCUMENT>

        Args:
            entity_id: The id property of the entity node.
            limit: Maximum number of linked documents included.
            max_chars: Maximum serialized characters for all linked-document blocks on this entity.
            excluded_document_ids: Optional set of Document ids already emitted elsewhere; their
                linked-document blocks are suppressed here.

        Returns:
            Concatenated, possibly truncated <LINKED_DOCUMENT> blocks within max_chars.
        """
        import html

        docs = self.get_linked_documents(entity_id, limit=limit, excluded_document_ids=excluded_document_ids)
        blocks: List[str] = []
        fence = "```"
        remaining_chars: int = max(0, int(max_chars))
        truncation_marker: str = "\n[... truncated ...]"
        for doc in docs:
            doc_id = html.escape(str(doc["id"]), quote=True)
            title = doc.get("title") or ""
            title_attr = f' title="{html.escape(str(title), quote=True)}"' if title else ""
            opening_tag: str = f'<LINKED_DOCUMENT id="{doc_id}"{title_attr}>\n{fence}\n'
            closing_tag: str = f'\n{fence}\n</LINKED_DOCUMENT>'
            separator: str = "\n" if blocks else ""
            available_text_chars: int = remaining_chars - len(separator) - len(opening_tag) - len(closing_tag)
            text: str = str(doc["text"])

            if available_text_chars < 0:
                continue

            if len(text) <= available_text_chars:
                block: str = f"{opening_tag}{text}{closing_tag}"
            elif available_text_chars >= len(truncation_marker):
                content_chars: int = available_text_chars - len(truncation_marker)
                block = f"{opening_tag}{text[:content_chars]}{truncation_marker}{closing_tag}"
            else:
                continue

            blocks.append(block)
            remaining_chars -= len(separator) + len(block)

        return "\n".join(blocks)

    def entity_to_discourse_text(self, entity: dict, linked_document_char_budget: int = MAX_RAG_MESSAGE_LENGTH_DEFAULT, excluded_document_ids: Optional[Set[str]] = None) -> str:
        """Convert an entity dict (from semantic_search_entity) to a natural-language discourse string.

        The entity dict is expected to contain 'name', 'label', 'description',
        'outgoing_relationships' and 'incoming_relationships'.

        Args:
            entity: Entity dictionary as returned by semantic_search_entity.
            linked_document_char_budget: This entity's share of the aggregate linked-document budget.
            excluded_document_ids: Optional set of Document ids already emitted as fragments in the
                same context; their <LINKED_DOCUMENT> blocks are omitted from this serialization.

        Returns:
            Natural-language prose describing the entity and its relationships, followed by linked
            document text in <LINKED_DOCUMENT id="..."> fenced blocks within its character budget.
        """
        if "Innovator" in (entity.get("labels") or []) or entity.get("source") == "Innovator" or "ea_element_id" in entity:
            try:
                from src.core.extensions.classes.Innovator import Innovator
                return Innovator().str(entity)
            
            except ImportError as e:
                log(ERROR, f"Error importing Innovator class: {e}")
            except Exception as e:
                log(ERROR, f"Error in Innovator.str(entity): {e}")

        from src.utils.text_utils import format_relationship_text
        
        def normalize_newlines(text):
            """Helper to replace newlines with spaces"""
            if isinstance(text, str):
                return text.replace('\n', ' ').replace('\r', ' ')
            return text
        
        _SKIP_KEYS = frozenset({
            "id", "entity_id", "elementId", "label", "name", "description",
            "score", "rank", "attributes",
            "outgoing_relationships", "incoming_relationships",
        })
        label = entity.get("label", "Entity")
        name = normalize_newlines(entity.get("name") or entity.get("entity_id") or entity.get("id", ""))
        description = normalize_newlines(entity.get("description", ""))

        parts = []
        parts.append(f"{name} is a {label}.")
        if description:
            parts.append(description)

        extra = {}
        nested = entity.get("attributes")
        if isinstance(nested, dict):
            extra.update(nested)
        for key, val in entity.items():
            if key not in _SKIP_KEYS and key not in extra:
                extra[key] = val
        for attr_name, attr_val in extra.items():
            if attr_val and attr_name not in ("name", "description"):
                readable = attr_name.replace("_", " ")
                normalized_val = normalize_newlines(str(attr_val)).rstrip().rstrip(".")
                parts.append(f"{name} {readable}: {normalized_val}.")

        for rel in entity.get("outgoing_relationships", []):
            rel_label = rel.get("relationship_label", "")
            target_name = normalize_newlines(
                rel.get("target_entity_name")
                or rel.get("entity_name")
                or rel.get("target_entity_id")
                or rel.get("id", "")
            )
            target_label = rel.get("target_entity_label") or rel.get("entity_label", "Entity")
            if rel_label and target_name:
                rel_text, prepend_is = format_relationship_text(rel_label)
                if prepend_is:
                    parts.append(f"{name} is {rel_text} {target_name} ({target_label}).")
                else:
                    parts.append(f"{name} {rel_text} {target_name} ({target_label}).")

        for rel in entity.get("incoming_relationships", []):
            rel_label = rel.get("relationship_label", "")
            source_name = normalize_newlines(
                rel.get("source_entity_name")
                or rel.get("entity_name")
                or rel.get("source_entity_id")
                or rel.get("id", "")
            )
            source_label = rel.get("source_entity_label") or rel.get("entity_label", "Entity")
            if rel_label and source_name:
                rel_text, prepend_is = format_relationship_text(rel_label)
                if prepend_is:
                    parts.append(f"{source_name} ({source_label}) is {rel_text} {name}.")
                else:
                    parts.append(f"{source_name} ({source_label}) {rel_text} {name}.")

        linked_entity_id = entity.get("id") or entity.get("entity_id")
        linked_block = ""
        if linked_entity_id and KG_HYDRATION_LINKED_DOCUMENTS_ENABLED:
            try:
                linked_block = self.linked_documents_discourse(
                    str(linked_entity_id),
                    max_chars=linked_document_char_budget,
                    excluded_document_ids=excluded_document_ids,
                )
            except Exception as exc:
                log(WARNING, f"entity_to_discourse_text: linked-document fetch failed for {linked_entity_id}: {exc}")

        if linked_block:
            parts.append(linked_block)

        return "\n".join(parts)

    def create_semantic_search_index(self, ignore_warnings: bool = True) -> None:
        """Create the semantic search vector index for entities"""
        import traceback
        from src.core.base.InputOutput import execute_neo4j_query
        from src.core.llm import EmbeddingsManager
        from src.utils.log import log, ERROR
        
        try:
            embeddings_dimension = EmbeddingsManager.instance().get_embeddings_dimension()

            # create if not existing the vector index for the entities
            execute_neo4j_query(
                """
                    CREATE VECTOR INDEX `ontology_entity` IF NOT EXISTS
                    FOR (e:Entity) ON (e.embedding)
                    OPTIONS { indexConfig: {
                        `vector.dimensions`: $dimension,
                        `vector.similarity_function`: 'cosine'
                    } }
                """,
                {"dimension": embeddings_dimension},
                ignore_warnings=ignore_warnings,
            )

        except Exception as e:
            log(ERROR, f"ERROR creating semantic search index: {e}\n{traceback.format_exc()}")

        return

    def update_semantic_search(self, entity_id: str) -> None:
        """Update semantic search embedding for a single entity"""
        from src.core.base.InputOutput import execute_neo4j_query
        from src.core.llm import EmbeddingsManager
        from src.utils.log import log, DEBUG, INFO
        
        self.create_semantic_search_index()

        # Fetch only _consolidated attributes — skip embedding to avoid transferring 1536 floats
        query = f"""
            MATCH (e:Entity) WHERE e.id = $entity_id
            WITH e, [k IN keys(e) WHERE k ENDS WITH '_consolidated'] AS ck
            RETURN ck AS consol_keys, [k IN ck | e[k]] AS consol_values
        """
        parameters = {"entity_id": entity_id}
        result = execute_neo4j_query(query, parameters)

        if result and len(result) > 0:
            log(DEBUG, f"Updating semantic search index for entity '{entity_id}'.")

            consol_keys = result[0]['consol_keys'] or []
            consol_values = result[0]['consol_values'] or []
            embedding_dict = {}
            for key, value in zip(consol_keys, consol_values):
                key_stripped = key.replace(CONSOLIDATION_SUFFIX, '')
                embedding_dict[key_stripped] = value

            if embedding_dict:
                embedding_text = json.dumps(embedding_dict)
                embedding = EmbeddingsManager.instance().get_embeddings(embedding_text)
                query = """
                    MATCH (e:Entity {id: $entity_id})
                    CALL db.create.setNodeVectorProperty(e, 'embedding', $embedding)
                """
                parameters = {"entity_id": entity_id, "embedding": embedding}
                execute_neo4j_query(query, parameters)
                log(INFO, f"Regenerated semantic search embedding for entity '{entity_id}'.")

        return

    def update_semantic_search_all(self) -> None:
        """Update semantic search embeddings for all entities"""
        from src.core.base.InputOutput import execute_neo4j_query
        
        self.create_semantic_search_index()

        query = """MATCH (e:Entity) RETURN e.id AS entity_id"""
        result = execute_neo4j_query(query)

        if result:
            for record in result:
                entity_id = record['entity_id']
                self.update_semantic_search(entity_id)

        return

    def consolidate_sentences(self, is_relationship: bool, attribute_name: str, label: str, ontology_id: str, values_to_consolidate: List[str]) -> str:
        """Consolidate multiple attribute values into one using LLM"""
        from src.core.base.InputOutput import execute_neo4j_query
        from src.core.llm import LlmManager, DEFAULT_TEMPERATURE_COLD, MAX_COMPLETION_TOKENS_SMALL
        from src.utils.log import log
        
        if not is_relationship:
            # identify the Ontology Element that defines the attribute
            query = f""" MATCH (o:Ontology {{id:$ontology_id}})-[:CONTAINS]->(e:EntityOntology:{label})
                         RETURN e.{attribute_name} as attribute_value
                     """
        else:
            # identify the Ontology Relationship that defines the attribute
            query = f""" MATCH (o:Ontology {{id:$ontology_id}})-[:CONTAINS]->(r:RelationshipOntology:{label})
                         RETURN r.{attribute_name} as attribute_value
                     """
        parameters = {"ontology_id": ontology_id}
        result = execute_neo4j_query(query, parameters)
        if result is None or len(result) == 0 or len(result[0]) == 0:
            attribute_value = None
        else:
            attribute_value = result[0]['attribute_value']

        # create a string with the list of values to be consolidated
        list_of_values = []
        for value in values_to_consolidate:
            list_of_values.append(f"`{value}`")
        values = ", ".join(list_of_values)

        # construct the instructions for the LLM
        message_content = f"""The following sentences are related to the attribute '{attribute_name}' of {"an entity" if is_relationship==False else "a relationship"}.\n"""
        if attribute_value is not None:
            message_content += f"""You must follow the attribute's definition in the Ontology: `{attribute_value}`.\n"""
        message_content += f"""\nHere are the sentences to consolidate. Please consolidate them according to the instructions above and following the attribute's ontology definition:\n```\n{values}\n```"""

        # ask the LLM to consolidate the sentences
        messages = [{
                "role": "system",
                "content": SYSTEM_PROMPT_CONSOLIDATE
            },{
                "role": "user",
                "content": message_content
            }
        ]
        llm_response = LlmManager.instance().chat_completions(messages=messages, max_completion_tokens=MAX_COMPLETION_TOKENS_SMALL, temperature=DEFAULT_TEMPERATURE_COLD)
        
        if llm_response is None:
            log(ERROR, "LLM response is None in consolidate_sentences - cannot consolidate values")
            return None
        
        consolidated_value = llm_response.response

        return consolidated_value

    def _get_ontology_attribute_types(self, ontology_id: str, label: str, is_relationship: bool) -> Dict[str, Optional[str]]:
        """Return declared attribute types for an ontology entity or relationship."""
        node_label = "RelationshipOntology" if is_relationship else "EntityOntology"
        query = f"""
            MATCH (o:Ontology {{id: $ontology_id}})-[:CONTAINS]->(n:{node_label})
            WHERE $label IN labels(n) OR n.id = $label
            RETURN properties(n) AS attributes
            LIMIT 1
        """
        result = execute_neo4j_query(query, {"ontology_id": ontology_id, "label": label}) or []
        if not result:
            return {}

        attributes = result[0].get("attributes") or {}
        return {
            key: _extract_declared_type(value)
            for key, value in attributes.items()
            if key not in RESERVED_ONTOLOGY_ATTRIBUTES
        }

    def _consolidate_entity_record(self, record: dict) -> bool:
        """Consolidate one entity record and return whether work was performed."""
        from src.core.base.InputOutput import execute_neo4j_query
        from src.utils.log import log, INFO, WARNING

        entity_id = record['entity_id']
        entity = dict(zip(record['ek'] or [], record['ev'] or []))

        query = f"""MATCH (e:Entity {{id: $entity_id}}) optional match (e)<-[:AGGREGATES]-(c:Community)<-[:DEFINES]-(o:Ontology) RETURN o.id as ontology_id"""
        parameters = {"entity_id": entity_id}
        result = execute_neo4j_query(query, parameters)
        if not result or result[0].get('ontology_id') is None:
            log(WARNING, f"consolidate_entities: skipping entity '{entity_id}' — no ontology found")
            return False

        raw_labels = record['e_labels'] or []
        label = next((lb for lb in raw_labels if lb not in RESERVED_ONTOLOGY_LABELS), None)
        ontology_id = result[0]['ontology_id']
        attribute_types = self._get_ontology_attribute_types(ontology_id, label, False)

        # remove all the previous consolidated attributes
        keys_to_remove = [f"e.`{key}`" for key in entity if key.endswith(CONSOLIDATION_SUFFIX)]
        if len(keys_to_remove) > 0:
            query = f"""MATCH (e:Entity {{id: $entity_id}}) REMOVE {', '.join(keys_to_remove)}"""
            parameters = {"entity_id": entity_id}
            execute_neo4j_query(query, parameters)

        # create new consolidated attributes, and the overall semantic search attribute
        values_to_add = {}
        for key in entity:
            if not key.endswith(CONSOLIDATION_SUFFIX):
                # each attribute is a list of dictionaries with the keys "value" and "source_id". Example for key="description":
                # "description": [{"value": "Luca is a developer.", "source_id": "docid-19"}, {"value": "Luca is an artist.", "source_id": "docid-19"}]
                value_list_of_dict = json.loads(entity[key])
                if len(value_list_of_dict) > 1:
                    # multiple values to consolidate
                    values_to_consolidate = [str(vd['value']) for vd in value_list_of_dict if vd.get('value') is not None]
                    consolidated_values = self.consolidate_sentences(label=label, values_to_consolidate=values_to_consolidate, attribute_name=key, is_relationship=False, ontology_id=ontology_id)
                    if consolidated_values is None:
                        continue
                elif len(value_list_of_dict) == 1:
                    raw_value = value_list_of_dict[0]['value']
                    raw_str = str(raw_value) if raw_value is not None else None
                    if raw_str is not None:
                        # A single source needs no semantic consolidation or LLM call.
                        consolidated_values = _convert_value_to_declared_type(raw_value, attribute_types.get(key))
                    else:
                        consolidated_values = None
                else:
                    # no values: nothing to consolidate
                    consolidated_values = None

                if consolidated_values is not None:
                    # add the consolidated value to the entity, with the suffix "_consolidated". For example: "description_consolidated"
                    typed_value = _convert_value_to_declared_type(consolidated_values, attribute_types.get(key))
                    values_to_add[f"{key}{CONSOLIDATION_SUFFIX}"] = typed_value

        some_work_done = False
        if len(values_to_add) > 0:
            query = """MATCH (e:Entity {id: $entity_id}) SET e += $consolidated_attributes"""
            parameters = {"entity_id": entity_id, "consolidated_attributes": values_to_add}
            log(INFO, f"Consolidating entity '{entity_id}' with the following query: {query}")
            execute_neo4j_query(query, parameters)
            some_work_done = True
            self.update_semantic_search(entity_id)

        query = f"""MATCH (e:Entity {{id: $entity_id}}) SET e.requires_consolidation = null"""
        parameters = {"entity_id": entity_id}
        execute_neo4j_query(query, parameters)

        return some_work_done

    def consolidate_entities(self, restrict_to_entity_ids: List[str] = None, suppress_debug_logs: bool = False) -> bool:
        """Consolidate entity attributes that require consolidation"""
        import traceback
        from src.core.base.InputOutput import execute_neo4j_query
        from src.utils.log import log, INFO, ERROR
        
        some_work_done = False

        try:
            if restrict_to_entity_ids:
                query = """ MATCH (e:Entity)
                            WHERE
                                coalesce(e.requires_consolidation, false) = true AND
                                e.id in $restrict_to_entity_ids
                            WITH e, labels(e) AS e_labels,
                                 [k IN keys(e) WHERE NOT k IN $reserved_attrs] AS ek,
                                 [k IN keys(e) WHERE NOT k IN $reserved_attrs | e[k]] AS ev
                            RETURN e.id AS entity_id, e_labels, ek, ev
                        """
            else:
                query = """ MATCH (e:Entity)
                            WHERE
                                coalesce(e.requires_consolidation, false) = true
                            WITH e, labels(e) AS e_labels,
                                 [k IN keys(e) WHERE NOT k IN $reserved_attrs] AS ek,
                                 [k IN keys(e) WHERE NOT k IN $reserved_attrs | e[k]] AS ev
                            RETURN e.id AS entity_id, e_labels, ek, ev
                        """
            parameters = {"restrict_to_entity_ids": restrict_to_entity_ids,
                          "reserved_attrs": RESERVED_ONTOLOGY_ATTRIBUTES}
            result_entity = execute_neo4j_query(query, parameters)
            if result_entity is None:
                raise Exception("1: Query returned None")

            from concurrent.futures import ThreadPoolExecutor, as_completed
            from src.config.constants import MAX_PARALLELISM

            if result_entity:
                max_workers = min(MAX_PARALLELISM, len(result_entity))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(self._consolidate_entity_record, record) for record in result_entity]
                    for future in as_completed(futures):
                        some_work_done |= future.result()

        except Exception as e:
            log(ERROR, f"Exception: {e}\n{traceback.format_exc()}")
            result = None

        return some_work_done

    def consolidate_relationships(self, suppress_debug_logs: bool = False) -> bool:
        """Consolidate relationship attributes that require consolidation"""
        import traceback
        from src.core.base.InputOutput import execute_neo4j_query
        from src.utils.log import log, INFO, ERROR
        
        some_work_done = False

        try:
            query = """ MATCH (e1:Entity)-[r]->(e2:Entity)
                        WHERE
                            coalesce(r.requires_consolidation, false) = true
                        WITH r
                        MATCH (e1:Entity)-[r]->(e2:Entity)
                        OPTIONAL MATCH (e1)<-[:AGGREGATES]-(c:Community)<-[:DEFINES]-(o:Ontology)
                        RETURN r, o.id as ontology_id
                    """
            result_relationship = execute_neo4j_query(query)
            if result_relationship is None:
                raise Exception("1: Query returned None")

            for record in result_relationship:
                relationship = record['r']
                relationship_id = relationship.id
                ontology_id = record['ontology_id']
                label = relationship.type
                attribute_types = self._get_ontology_attribute_types(ontology_id, label, True)

                # remove all the previous consolidated attributes
                keys_to_remove = [f"r.`{key}`" for key in relationship.keys() if key.endswith(CONSOLIDATION_SUFFIX)]
                if len(keys_to_remove) > 0:
                    query = f"""MATCH (r:Relationship {{id: $relationship_id}}) REMOVE {', '.join(keys_to_remove)}"""
                    parameters = {"relationship_id": relationship_id}
                    result = execute_neo4j_query(query, parameters)

                # create new consolidated attributes, and the overall semantic search attribute
                values_to_add = {}
                for key in relationship.keys():
                    if key not in RESERVED_ONTOLOGY_ATTRIBUTES and not key.endswith(CONSOLIDATION_SUFFIX):
                        # each attribute is a list of dictionaries with the keys "value" and "source_id". Example for key="description":
                        # "description": [{"value": "Luca is a developer.", "source_id": "docid-19"}, {"value": "Luca is an artist.", "source_id": "docid-19"}]
                        value_list_of_dict = json.loads(relationship[key])
                        if len(value_list_of_dict) > 1:
                            # multiple values to consolidate
                            values_to_consolidate = [str(vd['value']) for vd in value_list_of_dict if vd.get('value') is not None]
                            consolidated_values = self.consolidate_sentences(label=label, values_to_consolidate=values_to_consolidate, attribute_name=key, is_relationship=True, ontology_id=ontology_id)
                            if consolidated_values is None:
                                continue
                        elif len(value_list_of_dict) == 1:
                            raw_value = value_list_of_dict[0]['value']
                            raw_str = str(raw_value) if raw_value is not None else None
                            if raw_str is not None:
                                # A single source needs no semantic consolidation or LLM call.
                                consolidated_values = _convert_value_to_declared_type(raw_value, attribute_types.get(key))
                            else:
                                consolidated_values = None
                        else:
                            consolidated_values = None

                        if consolidated_values is not None:
                            typed_value = _convert_value_to_declared_type(consolidated_values, attribute_types.get(key))
                            values_to_add[f"{key}{CONSOLIDATION_SUFFIX}"] = typed_value
                if len(values_to_add) > 0:
                    query = """MATCH (:Entity)-[r]-(:Entity) WHERE id(r) = $relationship_id SET r += $consolidated_attributes"""
                    parameters = {"relationship_id": relationship_id, "consolidated_attributes": values_to_add}
                    log(INFO, f"Consolidating relationship '{relationship_id}' with the following query: {query}")

                    result = execute_neo4j_query(query, parameters)
                    some_work_done = True

                # remove the requires_consolidation flag
                query = f"""MATCH (:Entity)-[r]-(:Entity) WHERE id(r) = $relationship_id SET r.requires_consolidation = null"""
                parameters = {"relationship_id": relationship_id}
                result = execute_neo4j_query(query, parameters)

        except Exception as e:
            log(ERROR, f"ERROR in consolidate_knowledge_graph: {e}\n{traceback.format_exc()}")
            result = None

        return some_work_done

    def identify_entities_to_consolidate(self, goal, community_id: str, label: str, entities: List[dict]) -> List[List[str]]:
        """Identify which entities should be merged together"""
        from src.core.base.InputOutput import execute_neo4j_query
        from src.core.llm import LlmManager, DEFAULT_TEMPERATURE_COLD, MAX_COMPLETION_TOKENS_SMALL
        
        # use the LLM to identify the entities that need consolidation. Here's the proceedings:
        # 1. identify the entity's ontology using its label, get the ATTRIBUTE_ENTITY_DESCRIPTION
        # 2. prepare a message for the LLM with the list of entities to consolidate
        # 3. ask the LLM to consolidate the entities
        # 4. return the list of entities to consolidate

        # 1. identify the entity's ontology using its label, get the ATTRIBUTE_ENTITY_DESCRIPTION
        query = f"""MATCH (c:Community {{id:$community_id}})<-[:DEFINES]-(o:Ontology)-[:CONTAINS]->(e:EntityOntology) RETURN e.{ATTRIBUTE_ENTITY_DESCRIPTION} as {ATTRIBUTE_ENTITY_DESCRIPTION}"""
        parameters = {"community_id": community_id}
        result = execute_neo4j_query(query, parameters)
        if result is None:
            raise Exception("1: Query returned None")
        if len(result) > 0:
            entity_description = result[0][ATTRIBUTE_ENTITY_DESCRIPTION]
        else:
            entity_description = None

        # 2. prepare a message for the LLM with the list of entities to consolidate
        message_content = f"""The following entities are all labeled '{label}'"""
        if entity_description is not None:
            message_content += f""" and share the following entity definition: '{entity_description}'"""
        message_content += """\nSome of these entities may need consolidation. Please identify which of them need to be consolidated them according to the instructions:\n"""
        for entity in entities:
            message_content += f"""- Entity '{entity['id']}': attributes=`{entity}`\n"""

        # ask the LLM to consolidate the sentences
        messages = [{
                "role": "system",
                "content": SYSTEM_PROMPT_MERGE_ENTITIES
            },{
                "role": "user",
                "content": message_content
            }
        ]
        llm_response = LlmManager.instance().chat_completions(messages=messages, max_completion_tokens=MAX_COMPLETION_TOKENS_SMALL, temperature=DEFAULT_TEMPERATURE_COLD, response_format=OUTPUT_FORMAT_MERGE_ENTITIES, structured_output=OUTPUT_SCHEMA_MERGE_ENTITIES)
        consolidated_value = llm_response.response
        thinking_message = llm_response.thinking_message
        
        return consolidated_value["entity_to_consolidate"] if consolidated_value is not None and "entity_to_consolidate" in consolidated_value else []

    def consolidate_knowledge_graph(self, force_create_semantic_search: bool = False, suppress_debug_logs: bool = False) -> bool:
        """Consolidate all entities and relationships in the knowledge graph"""
        import traceback
        from src.utils.log import log, INFO, ERROR
        
        some_work_done = False

        try:
            pending_query = """
                MATCH (e:Entity)
                WHERE coalesce(e.requires_consolidation, false) = true
                RETURN count(e) AS count
                UNION ALL
                MATCH ()-[r]->()
                WHERE coalesce(r.requires_consolidation, false) = true
                RETURN count(r) AS count
            """
            pending_result = execute_neo4j_query(pending_query) or []
            pending_count = sum(record.get("count", 0) for record in pending_result)
            has_pending_consolidation = pending_count > 0
            from src.core.base.InputOutput import send_ontology_consolidation_status_to_blazor
            send_ontology_consolidation_status_to_blazor(has_pending_consolidation)

            some_work_done = self.consolidate_entities(suppress_debug_logs=suppress_debug_logs)
            some_work_done |= self.consolidate_relationships(suppress_debug_logs=suppress_debug_logs)
            pending_result = execute_neo4j_query(pending_query) or []
            pending_count = sum(record.get("count", 0) for record in pending_result)
            has_pending_consolidation = pending_count > 0
            send_ontology_consolidation_status_to_blazor(has_pending_consolidation)
            if some_work_done:
                log(INFO, f"Some entities and/or relationships have been consolidated.")

            if force_create_semantic_search:
                self.update_semantic_search_all()
                log(INFO, f"Semantic search index has been updated.")

        except Exception as e:
            log(ERROR, f"ERROR in consolidate_knowledge_graph: {e}\n{traceback.format_exc()}")

        return some_work_done

    def remove_relationships_source(self, document_id: str) -> None:
        """Remove relationship information from a specific document source"""
        import traceback
        from src.core.base.InputOutput import execute_neo4j_query
        from src.utils.log import log, DEBUG, INFO, ERROR
        
        log(DEBUG, f"Removing relationships from document '{document_id}'.")

        try:
            from src.services.ontology_source_helper import build_doc_source_id

            query = """MATCH (e:Entity)-[r]->(e2:Entity) WHERE $id in r.source_ids RETURN r"""
            parameters = {"id": build_doc_source_id(document_id)}
            result = execute_neo4j_query(query, parameters)

            for record in result:
                relationship = record['r']
                source_ids = json.loads(relationship['source_ids'])
                if document_id in source_ids:
                    source_ids.remove(document_id)
                    if len(source_ids) == 0:
                        log(DEBUG, f"Removing relationship: {record['r']}")

                        query = f"""MATCH ()-[r {{id: $relationship_id}}]-() DELETE r"""
                        parameters = {"relationship_id": relationship.id}
                        result = execute_neo4j_query(query, parameters)
                    else:
                        log(DEBUG, f"Updating relationship '{relationship.id}' with new sources: {source_ids}")

                        query = f"""MATCH ()-[r {{id: $relationship_id}}]-() SET r.source_ids = $source_ids"""
                        parameters = {"relationship_id": relationship.id, "source_ids": json.dumps(source_ids)}
                        result = execute_neo4j_query(query, parameters)

        except Exception as e:
            log(ERROR, f"ERROR in remove_relationships_source: {e}\n{traceback.format_exc()}")
            result = None

        log(INFO, f"[doc: {document_id}] Relationships removed.")
        return

    def remove_entities_source(self, document_id: str) -> None:
        """Remove entity information from a specific document source"""
        import traceback
        from src.core.base.InputOutput import execute_neo4j_query
        from src.utils.log import log, DEBUG, INFO, ERROR
        
        try:
            log(DEBUG, f"Removing entities from document '{document_id}'.")

            query = """MATCH (e:Entity)-[:SOURCE]->(d:Document {id:$document_id})
                WITH e, elementId(e) AS eid,
                     [k IN keys(e) WHERE NOT k IN $reserved_attrs] AS ek,
                     [k IN keys(e) WHERE NOT k IN $reserved_attrs | e[k]] AS ev
                RETURN e.id AS entity_id, eid, ek, ev"""
            parameters = {"document_id": document_id, "reserved_attrs": RESERVED_ONTOLOGY_ATTRIBUTES}
            result = execute_neo4j_query(query, parameters)

            for record in result:
                entity = dict(zip(record['ek'] or [], record['ev'] or []))
                entity_neo4j_eid = record['eid']
                residual_attributes = 0
                log(DEBUG, f"Removing entity: {record['entity_id']}")
                    
                for key in entity:
                    if not key.endswith(CONSOLIDATION_SUFFIX):
                        # remove the source_id from the entity
                        # format of the attribute: [{"value": "Martha", "source_id": "doc://docid-20?p=p-17,doc://docid-20?p=p-5,doc://docid-22"}, {"value": "Margareth", "source_id": "doc://docid-20?p=p-17"}]
                        if entity[key].startswith("["):
                            from src.services.ontology_source_helper import remove_document_from_source_id_list

                            attribute = json.loads(entity[key])
                            new_attribute_components = []

                            for attribute_component in attribute:
                                # we are processing a single attribute component, e.g. {"value": "Martha", "source_id": "doc://docid-20?p=p-17,doc://docid-20?p=p-5,doc://docid-22"}
                                new_attribute_sources, _ = remove_document_from_source_id_list(attribute_component["source_id"], document_id)
                                if new_attribute_sources:
                                    # this attribute has other sources: keep it
                                    new_attribute_components.append({"value": attribute_component["value"], "source_id": new_attribute_sources})

                            if len(new_attribute_components) > 0:
                                log(DEBUG, f"Updating attribute '{key}' with new sources: {new_attribute_components}")

                                attribute = json.dumps(new_attribute_components)
                                query = f"""MATCH (e:Entity) WHERE elementId(e)=$eid SET e.`{key}`=$attribute_value, e.`{key}{CONSOLIDATION_SUFFIX}`=null"""
                                parameters = {"eid": entity_neo4j_eid, "attribute_value": attribute}
                                result = execute_neo4j_query(query, parameters)
                                residual_attributes += 1
                            else:
                                # no more sources for this attribute: remove it
                                log(DEBUG, f"Removing attribute '{key}' from entity: {record['entity_id']}")

                                query = f"""MATCH (e:Entity) WHERE elementId(e)=$eid SET e.`{key}`=null, e.`{key}{CONSOLIDATION_SUFFIX}`=null"""
                                parameters = {"eid": entity_neo4j_eid}
                                result = execute_neo4j_query(query, parameters)
                
                if residual_attributes == 0:
                    # With no attributes left the entity is only worth keeping if a document still vouches
                    # for it. A curated IS_DOCUMENTED link is a human decision, and deleting the node here
                    # would undo it silently. The entity stays attribute-less and the integrity sweep
                    # re-appraises it later.
                    anchor_records = execute_neo4j_query(
                        """MATCH (e:Entity) WHERE elementId(e) = $eid AND (e)-[:IS_DOCUMENTED]-() RETURN e.id AS id""",
                        {"eid": entity_neo4j_eid}
                    )

                    if anchor_records:
                        log(DEBUG, f"Keeping attribute-less entity, still linked by IS_DOCUMENTED: {record['entity_id']}")
                    else:
                        # no more attributes and no remaining link: remove the entity
                        log(DEBUG, f"Removing entity: {record['entity_id']}")

                        query = """MATCH (e:Entity) WHERE elementId(e)=$eid DETACH DELETE e"""
                        parameters = {"eid": entity_neo4j_eid}
                        result = execute_neo4j_query(query, parameters)
                else:
                    # mark the entity as requiring consolidation
                    log(DEBUG, f"Marking entity as requiring consolidation: {record['entity_id']}")

                    query = """MATCH (e:Entity) WHERE elementId(e)=$eid SET e.requires_consolidation = true"""
                    parameters = {"eid": entity_neo4j_eid}
                    result = execute_neo4j_query(query, parameters)

        except Exception as e:
            log(ERROR, f"ERROR in remove_entities_source: {e}\n{traceback.format_exc()}")
            result = None

        log(INFO, f"[doc: {document_id}] Entities removed.")
        return

    def get_entity_raw_attributes(self, element_id: str) -> Dict[str, str]:
        """Return {attr_name: raw_json_value} for non-reserved, unconsolidated attributes of the Entity with the given elementId."""
        raw_attributes: Dict[str, str] = {}

        try:
            query = """
                MATCH (e:Entity)
                WHERE elementId(e) = $element_id
                RETURN keys(e) AS attribute_keys, [k IN keys(e) | e[k]] AS attribute_values
            """
            result = execute_neo4j_query(query, {"element_id": element_id})
            if result and len(result) > 0:
                attribute_keys = result[0].get("attribute_keys") or []
                attribute_values = result[0].get("attribute_values") or []
                for key, value in zip(attribute_keys, attribute_values):
                    if self.is_not_reserved_and_unconsolidated_attribute(key) and isinstance(value, str) and value.startswith("["):
                        raw_attributes[key] = value

        except Exception as e:
            log(ERROR, f"get_entity_raw_attributes(): Error reading raw attributes for entity elementId '{element_id}': {e}\n{traceback.format_exc()}")

        return raw_attributes

    def nodes_exist(self, node_references: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Optional[bool]]:
        """Check which (node_label, node_id) references correspond to existing Neo4j nodes."""
        existence_by_reference: Dict[Tuple[str, str], Optional[bool]] = {reference: None for reference in node_references}

        if node_references:
            try:
                node_ids_by_label: Dict[str, List[str]] = {}
                for node_label, node_id in node_references:
                    node_ids_by_label.setdefault(node_label, []).append(node_id)

                for node_label, node_ids in node_ids_by_label.items():
                    query = f"MATCH (n:{node_label}) WHERE n.id IN $node_ids RETURN n.id AS node_id"
                    result = execute_neo4j_query(query, {"node_ids": node_ids})
                    if result is None:
                        log(WARNING, f"nodes_exist(): Neo4j existence query failed for label '{node_label}'.")
                        continue

                    found_node_ids = {record["node_id"] for record in result if record.get("node_id")}
                    for node_id in node_ids:
                        existence_by_reference[(node_label, node_id)] = node_id in found_node_ids

            except Exception as e:
                log(ERROR, f"nodes_exist(): Error checking node existence: {e}\n{traceback.format_exc()}")

        return existence_by_reference

    def get_primary_key_attribute_names(self, label: str) -> List[str]:
        """Return the ordered attribute names forming the primary key of the EntityOntology with the given label, or an empty list if none is defined."""
        primary_key_attribute_names: List[str] = []

        if label:
            try:
                query = """
                    MATCH (n:EntityOntology)
                    WHERE $label IN labels(n)
                    MATCH (n)-[:HAS_ATTRIBUTE]->(a:OntologyAttribute)
                    WHERE a.primary_key IS NOT NULL
                    RETURN a.name AS name
                    ORDER BY a.primary_key
                """
                result = execute_neo4j_query(query, {"label": label}) or []
                primary_key_attribute_names = [record["name"] for record in result if record.get("name")]

            except Exception as e:
                log(ERROR, f"{self._log_prefix}.get_primary_key_attribute_names: Error for label='{label}': {e}")

        return primary_key_attribute_names

    def _extract_attribute_values(self, raw_value: Any) -> List[str]:
        """Extract the plain-text values out of a raw (possibly JSON-encoded, source-tagged) attribute value."""
        values: List[str] = []

        if isinstance(raw_value, str) and raw_value.startswith("[") and raw_value.endswith("]"):
            try:
                parsed_entries = json.loads(raw_value)
                values = [str(entry.get("value")) for entry in parsed_entries if isinstance(entry, dict) and entry.get("value") is not None]
            except (TypeError, ValueError):
                values = [raw_value]
        elif raw_value is not None:
            values = [str(raw_value)]

        return values

    def _primary_key_values_match(self, entity_attributes: Dict[str, Any], candidate_values: Dict[str, str]) -> bool:
        """Check whether an existing entity's stored attribute values already contain every candidate primary-key value."""
        all_match = True

        for name, candidate_value in candidate_values.items():
            stored_values = self._extract_attribute_values(entity_attributes.get(name))
            case_sensitive = name in CASE_SENSITIVE_ATTRIBUTES
            value_found = any(
                (stored_value == candidate_value) if case_sensitive else (stored_value.lower() == candidate_value.lower())
                for stored_value in stored_values
            )
            if not value_found:
                all_match = False
                break

        return all_match

    def find_conflicting_entity_by_primary_key(self, label: str, entity_id: str, attributes: Dict[str, Any]) -> Optional[str]:
        """Return the id of another Entity with the same label whose primary-key attribute values collide with the given ones, or None."""
        conflicting_entity_id: Optional[str] = None

        try:
            primary_key_attribute_names = self.get_primary_key_attribute_names(label)

            if primary_key_attribute_names and attributes:
                candidate_values: Dict[str, str] = {
                    name: str(attributes[name]).strip()
                    for name in primary_key_attribute_names
                    if attributes.get(name) is not None and str(attributes[name]).strip() != ""
                }

                # only enforce the check once every primary-key part is provided; a partial key cannot be compared
                if len(candidate_values) == len(primary_key_attribute_names):
                    query = """
                        MATCH (e:Entity)
                        WHERE $label IN labels(e) AND e.id <> $entity_id
                        RETURN e.id AS id, properties(e) AS attributes
                    """
                    result = execute_neo4j_query(query, {"label": label, "entity_id": entity_id}) or []

                    for record in result:
                        if self._primary_key_values_match(record.get("attributes") or {}, candidate_values):
                            conflicting_entity_id = record.get("id")
                            break

        except Exception as e:
            log(ERROR, f"{self._log_prefix}.find_conflicting_entity_by_primary_key: Error for label='{label}' entity_id='{entity_id}': {e}")

        return conflicting_entity_id

    def remove_obsolete_references_from_entity_attributes(self, element_id: str, obsolete_references: Set[Tuple[str, str]]) -> bool:
        """Prune source_id entries and embedded reference URIs pointing to no-longer-existing nodes from an entity's attributes."""
        from src.utils.dataset_helpers import EMBEDDED_REFERENCE_URI_RE, resolve_source_reference
        from src.services.ontology_source_helper import split_source_id_list, join_source_id_list

        success: bool = True

        if obsolete_references:
            try:
                raw_attributes = self.get_entity_raw_attributes(element_id)

                for key, raw_value in raw_attributes.items():
                    attribute_components = json.loads(raw_value)
                    new_attribute_components = []
                    attribute_changed = False

                    for attribute_component in attribute_components:
                        # format: {"value": "Martha", "source_id": "doc://docid-20?p=p-17,doc://docid-20?p=p-5,doc://docid-22"}
                        source_id = attribute_component.get("source_id")
                        attribute_sources_list = split_source_id_list(source_id)
                        new_components_list = []
                        for attribute_source in attribute_sources_list:
                            resolved_reference = resolve_source_reference(attribute_source)
                            if resolved_reference and resolved_reference in obsolete_references:
                                attribute_changed = True
                            else:
                                new_components_list.append(attribute_source)

                        if attribute_sources_list and len(new_components_list) == 0:
                            # no sources remain for this value: drop it entirely
                            continue

                        attribute_value_text = attribute_component.get("value")
                        if isinstance(attribute_value_text, str):
                            def _prune_embedded_reference(match):
                                nonlocal attribute_changed
                                resolved = resolve_source_reference(match.group(0))
                                if resolved and resolved in obsolete_references:
                                    attribute_changed = True
                                    return ""
                                return match.group(0)

                            attribute_value_text = EMBEDDED_REFERENCE_URI_RE.sub(_prune_embedded_reference, attribute_value_text)

                        new_attribute_component = dict(attribute_component)
                        new_attribute_component["value"] = attribute_value_text
                        if isinstance(source_id, str):
                            new_attribute_component["source_id"] = join_source_id_list(new_components_list)
                        new_attribute_components.append(new_attribute_component)

                    if attribute_changed:
                        if len(new_attribute_components) > 0:
                            log(DEBUG, f"remove_obsolete_references_from_entity_attributes(): Updating attribute '{key}' for entity elementId '{element_id}'.")
                            query = f"""MATCH (e:Entity) WHERE elementId(e) = $element_id SET e.`{key}` = $attribute_value, e.`{key}{CONSOLIDATION_SUFFIX}` = null"""
                            parameters = {"element_id": element_id, "attribute_value": json.dumps(new_attribute_components)}
                        else:
                            log(DEBUG, f"remove_obsolete_references_from_entity_attributes(): Removing attribute '{key}' for entity elementId '{element_id}' (no sources remain).")
                            query = f"""MATCH (e:Entity) WHERE elementId(e) = $element_id SET e.`{key}` = null, e.`{key}{CONSOLIDATION_SUFFIX}` = null"""
                            parameters = {"element_id": element_id}

                        result = execute_neo4j_query(query, parameters)
                        success = success and bool(result)

            except Exception as e:
                log(ERROR, f"remove_obsolete_references_from_entity_attributes(): Error pruning references for entity elementId '{element_id}': {e}\n{traceback.format_exc()}")
                success = False

        return success

    def create_source_relationship(self, entity_element_id: str, target_label: str, target_id: str) -> bool:
        """Create the SOURCE relationship from an Entity to a target node, if not already present."""
        success: bool = False

        try:
            query = f"""
                MATCH (e:Entity)
                WHERE elementId(e) = $entity_element_id
                WITH e
                MATCH (t:{target_label} {{id: $target_id}})
                MERGE (e)-[:SOURCE]->(t)
                RETURN e
            """
            parameters = {"entity_element_id": entity_element_id, "target_id": target_id}
            result = execute_neo4j_query(query, parameters)
            success = bool(result)
            if not success:
                log(WARNING, f"create_source_relationship(): No matching {target_label} node found for target_id '{target_id}'.")

        except Exception as e:
            log(ERROR, f"create_source_relationship(): Error creating SOURCE relationship for entity elementId '{entity_element_id}' to {target_label} '{target_id}': {e}\n{traceback.format_exc()}")

        return success

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its extracted information from the knowledge graph"""
        import traceback
        from src.core.base.InputOutput import execute_neo4j_query
        from src.utils.log import log, ERROR
        
        result = True

        try:
            # first, remove the information extracted from the document, from all entities and relationships
            ## relationships:
            self.remove_relationships_source(document_id)

            ## entities:
            self.remove_entities_source(document_id)

            # now that the extraction has been stripped, drop the links pointing at this document. This
            # runs here, after the entity scrub, so that scrub can see which entities are still vouched
            # for. It also runs before the node delete rather than relying on DETACH DELETE to cascade:
            # an edge that disappears as a side effect of a cascade is a silent data-loss path.
            query = """
                MATCH (:Entity)-[r:IS_DOCUMENTED]->(d:Document {id:$document_id})
                DELETE r
            """
            parameters = {"document_id": document_id}
            execute_neo4j_query(query, parameters)

            # now, find all Chapter nodes related to the document, along with all optionally associated Summary, Paragraph and Question nodes, and remove them.
            # the structure is:
            # (d:Document)<-[:PART_OF]-(c:Chapter)      0:n
            # (c:Chapter)-[:HAS_SUMMARY]->(s:Summary)   0:1
            # (c:Chapter)-[:HAS_QUESTION]->(q:Question) 0:n
            # (c:Chapter)<-[:PART_OF]-(p:Paragraph)     0:n
            query = """
                MATCH (d:Document {id:$document_id})<-[:PART_OF]-(c:Chapter)
                    OPTIONAL MATCH (c)-[:HAS_SUMMARY]->(s:Summary)
                    OPTIONAL MATCH (c)-[:HAS_QUESTION]->(q:Question)
                    OPTIONAL MATCH (c)<-[:PART_OF]-(p:Paragraph)
                    OPTIONAL MATCH (d)-[:HAS_PAGE]->(g:Page)
                    DETACH DELETE c, s, q, p, g"""
            parameters = {"document_id": document_id}
            execute_neo4j_query(query, parameters)

            # delete the Document node itself
            query = """MATCH (n {documentid:$id}) DETACH DELETE n"""
            parameters = {"id": document_id}
            execute_neo4j_query(query, parameters)

            # delete eventually isolated Keyword nodes, residual after removal
            query = """MATCH (k:Keyword) WHERE NOT (k)--() DELETE k"""
            execute_neo4j_query(query, parameters)
        except Exception as e:
            log(ERROR, f"delete_document(): ERROR: {e}\n{traceback.format_exc()}")
            result = False

        return result

    def get_categories_count_for_ontology(self, ontology_id: str, exclude_proposed: bool = False) -> int:
        """Get count of categories for a given ontology.
        """
        from src.core.base.InputOutput import execute_neo4j_query
        from src.utils.log import log, ERROR
        import traceback

        try:
            # None or the default ontology means no categories
            if ontology_id is None or ontology_id == get_default_ontology_id():
                return 0
            
            query = """
                MATCH (o:Ontology {id: $ontology_id})-[:CONTAINS]->(cat:KeywordCategory)
                WHERE ($exclude_proposed = false OR coalesce(cat.proposed, false) = false)
                RETURN count(cat) AS count
            """
            params = {"ontology_id": ontology_id, "exclude_proposed": exclude_proposed}

            result = execute_neo4j_query(query, params) or []
            if not result:
                return 0

            # result[0] expected to be a mapping with key 'count'
            try:
                cnt = result[0].get('count')
            except Exception:
                # fallback: take first value in mapping
                try:
                    cnt = list(result[0].values())[0]
                except Exception:
                    cnt = 0

            return int(cnt or 0)

        except Exception as e:
            log(ERROR, f"ERROR in get_categories_count_for_ontology: {e}\n{traceback.format_exc()}")
            return 0

    def get_categories_for_ontology(self, ontology_id: str, exclude_proposed: bool = False) -> List[dict]:
        """Get categories for a given ontology as a list of dictionaries.
        
        Args:
            ontology_id: The ontology identifier
            exclude_proposed: If True, exclude categories with proposed=True (only return confirmed categories)
            
        Returns:
            List of category dictionaries
        """
        from src.core.base.InputOutput import execute_neo4j_query

        categories: List[dict] = []

        try:
            if ontology_id is None:
                query = """MATCH (o:Ontology)-[:CONTAINS]->(cat:KeywordCategory) RETURN o, cat"""
                parameters = {}
            else:
                query = """MATCH (o:Ontology {id: $ontology_id})-[:CONTAINS]->(cat:KeywordCategory) RETURN o, cat"""
                parameters = {"ontology_id": ontology_id}

            result = execute_neo4j_query(query, parameters)
            if result is None:
                return categories

            for record in result:
                cat_node = record['cat']

                # support different driver/node shapes: py2neo/neo4j Node-like or dict with 'properties'
                props = {}
                identity = None
                labels = None
                element_id = None

                try:
                    if hasattr(cat_node, 'properties'):
                        props = dict(cat_node.properties)
                    elif isinstance(cat_node, dict) and 'properties' in cat_node:
                        props = dict(cat_node.get('properties') or {})
                    elif hasattr(cat_node, 'items'):
                        # some drivers return mapping-like nodes
                        try:
                            props = dict(cat_node.items())
                        except Exception:
                            props = {}

                    if hasattr(cat_node, 'identity'):
                        identity = cat_node.identity
                    elif isinstance(cat_node, dict) and 'identity' in cat_node:
                        identity = cat_node.get('identity')

                    if hasattr(cat_node, 'labels'):
                        labels = list(cat_node.labels)
                    elif isinstance(cat_node, dict) and 'labels' in cat_node:
                        labels = list(cat_node.get('labels') or [])

                    if hasattr(cat_node, 'elementId'):
                        element_id = cat_node.elementId
                    elif isinstance(cat_node, dict) and 'elementId' in cat_node:
                        element_id = cat_node.get('elementId')
                except Exception:
                    # defensive: ignore extraction errors and continue
                    props = {}

                # Filter out proposed categories if requested
                if exclude_proposed and props.get('proposed') is True:
                    continue

                cat = {"ontology_id": record['o']['id']}
                if identity is not None:
                    cat['identity'] = identity
                if labels is not None:
                    cat['labels'] = labels
                if element_id is not None:
                    cat['elementId'] = element_id

                # merge properties (id, text_en, text_it, etc.) into top-level
                for k, v in (props or {}).items():
                    cat[k] = v

                categories.append(cat)

        except Exception as e:
            log(ERROR, f"ERROR in get_categories_for_ontology: {e}\n{traceback.format_exc()}")

        return categories

    def find_duplicate_categories(self, document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Find duplicated Category nodes for documents.

        Returns a list of dicts with keys: document_id, category_key, category_value,
        count, category_node_ids.
        """
        try:
            if document_ids:
                query = """
                MATCH (d:Document)-[:HAS_CATEGORY]->(c:Category)-[:OF_CATEGORY]->(kc:KeywordCategory)
                WHERE d.id IN $document_ids OR d.documentid IN $document_ids
                WITH d.id AS document_id, kc.text_en AS category_key, c.text AS category_value,
                     collect(id(c)) AS category_node_ids, count(c) AS cnt
                WHERE cnt > 1
                RETURN document_id, category_key, category_value, cnt, category_node_ids
                ORDER BY cnt DESC
                """
                params = {"document_ids": document_ids}
            else:
                query = """
                MATCH (d:Document)-[:HAS_CATEGORY]->(c:Category)-[:OF_CATEGORY]->(kc:KeywordCategory)
                WITH d.id AS document_id, kc.text_en AS category_key, c.text AS category_value,
                     collect(id(c)) AS category_node_ids, count(c) AS cnt
                WHERE cnt > 1
                RETURN document_id, category_key, category_value, cnt, category_node_ids
                ORDER BY cnt DESC
                """
                params = {}

            rows = execute_neo4j_query(query, params) or []
            out: List[Dict[str, Any]] = []
            for r in rows:
                out.append({
                    "document_id": r.get("document_id"),
                    "category_key": r.get("category_key"),
                    "category_value": r.get("category_value"),
                    "count": int(r.get("cnt") or 0),
                    "category_node_ids": [int(x) for x in (r.get("category_node_ids") or [])]
                })
            return out
        except Exception as e:
            log(ERROR, f"{self._log_prefix}: Error in find_duplicate_categories(): {e}")
            return []

    def save_proposed_categories(self, ontology_id: str, proposed_categories: List[Dict[str, Any]], effective_user_id: Optional[str] = None) -> bool:
        """Save proposed categories as KeywordCategory nodes with 'proposed'='true' attribute.
        
        Skips categories that already exist in the ontology (by text_en match).

        ACL: this creates nodes under an :Ontology, so it is gated like every other write path -
        the caller needs GRANT:ADD on the target ontology (tenant-admin bypass included). The
        check cannot be skipped by omitting the principal: when `effective_user_id` is None it is
        resolved from the current request or background context, which defaults to 'global'.
        
        Args:
            ontology_id: The ontology identifier to associate proposals with
            proposed_categories: List of category proposals, each with 'text_en', 'description', 
                               and optional 'text_fr', 'text_de', 'text_it', 'lov'
            effective_user_id: Neo4j User.id of the requesting user. When None the principal is
                               resolved from the request/background context.
            
        Returns:
            True if all proposals were saved successfully, False otherwise
        """
        from src.core.base.InputOutput import execute_neo4j_query
        import uuid

        if not ontology_id or not proposed_categories:
            log(WARNING, "save_proposed_categories called with empty ontology_id or proposed_categories")
            return False

        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()

        if not self.check_ontology_permission(effective_user_id, ontology_id, "ADD"):
            log(ERROR, f"{self._log_prefix}.save_proposed_categories: ADD access denied for effective user '{effective_user_id}' on ontology '{ontology_id}', {len(proposed_categories)} proposal(s) not saved")
            return False

        try:
            # Get existing categories to avoid duplicates
            existing_categories = self.get_categories_for_ontology(ontology_id) or []
            existing_texts = {(c.get('text_en') or '').strip() for c in existing_categories if c.get('text_en')}
            
            skipped_count = 0
            saved_count = 0
            
            for proposal in proposed_categories:
                # Extract proposal data
                text_en: str = (proposal.get('text_en') or '').strip()
                description: str = (proposal.get('description') or '').strip()
                text_fr: Optional[str] = ((proposal.get('text_fr') or '').strip() or None)
                text_de: Optional[str] = ((proposal.get('text_de') or '').strip() or None)
                text_it: Optional[str] = ((proposal.get('text_it') or '').strip() or None)
                lov: Optional[str] = proposal.get('lov', None)
                category_class: Optional[str] = (proposal.get('class', None) or None)
                legal_source: Optional[str] = (proposal.get('legal_source', None) or None)
                
                if not text_en:
                    log(WARNING, f"Skipping proposal with empty text_en: {proposal}")
                    continue

                # Skip if category already exists
                if text_en in existing_texts:
                    log(DEBUG, f"Skipping existing category '{text_en}' for ontology '{ontology_id}'")
                    skipped_count += 1
                    continue

                # Generate unique ID for the proposed category
                proposal_id = f"keyword_cat_{uuid.uuid4().hex[:12]}"
                
                # Build properties dict for the new KeywordCategory node
                node_properties = {
                    "id": proposal_id,
                    "text_en": text_en,
                    "description": description,
                    "proposed": True
                }
                
                # Add translations if present
                if text_fr:
                    node_properties["text_fr"] = text_fr
                if text_de:
                    node_properties["text_de"] = text_de
                if text_it:
                    node_properties["text_it"] = text_it
                if lov:
                    node_properties["lov"] = lov

                # Add optional metadata fields
                if category_class:
                    node_properties["class"] = str(category_class)
                if legal_source:
                    node_properties["legal_source"] = str(legal_source)

                # Build the property list for CREATE
                property_assignments = [
                    "id: $id",
                    "text_en: $text_en",
                    "description: $description",
                    "proposed: $proposed"
                ]
                if text_fr:
                    property_assignments.append("text_fr: $text_fr")
                if text_de:
                    property_assignments.append("text_de: $text_de")
                if text_it:
                    property_assignments.append("text_it: $text_it")
                if lov:
                    property_assignments.append("lov: $lov")

                if category_class:
                    property_assignments.append("class: $class")
                if legal_source:
                    property_assignments.append("legal_source: $legal_source")

                # Create Cypher query to create the KeywordCategory node and link it to the ontology
                query = f"""MATCH (o:Ontology {{id: $ontology_id}})
CREATE (cat:KeywordCategory {{{', '.join(property_assignments)}}})
CREATE (o)-[:CONTAINS]->(cat)
RETURN cat.id as category_id"""
                
                params = {
                    "ontology_id": ontology_id,
                    "id": proposal_id,
                    "text_en": text_en,
                    "description": description,
                    "proposed": True
                }
                
                # Add translation parameters if present
                if text_fr:
                    params["text_fr"] = text_fr
                if text_de:
                    params["text_de"] = text_de
                if text_it:
                    params["text_it"] = text_it
                if lov:
                    params["lov"] = lov

                if category_class:
                    params["class"] = str(category_class)
                if legal_source:
                    params["legal_source"] = str(legal_source)

                result = execute_neo4j_query(query, params)
                if result is None or len(result) == 0:
                    log(WARNING, f"Failed to save proposed category '{text_en}' for ontology '{ontology_id}'")
                    return False
                    
                log(INFO, f"Saved proposed category '{text_en}' with ID '{proposal_id}' to ontology '{ontology_id}'")
                saved_count += 1

            if skipped_count > 0:
                log(INFO, f"Skipped {skipped_count} existing categories during import")
            if saved_count > 0:
                log(INFO, f"Saved {saved_count} new categories to ontology '{ontology_id}'")

            return True

        except Exception as e:
            log(ERROR, f"ERROR in save_proposed_categories: {e}\n{traceback.format_exc()}")
        return False

    def delete_keyword_categories(self, ontology_id: Optional[str], pattern: Optional[str], delete_all: bool = False) -> int:
        """Delete KeywordCategory nodes.

        If `delete_all` is True and `ontology_id` provided, deletes all KeywordCategory
        nodes in that ontology. If `pattern` is provided (a regex), deletes keyword
        categories whose `text_en` matches the regex. Returns the number of nodes deleted.
        """
        from src.core.base.InputOutput import execute_neo4j_query

        try:
            # collect matching node ids first
            if delete_all:
                if not ontology_id:
                    log(WARNING, "delete_keyword_categories: --all requested but no ontology_id provided")
                    return 0
                query = """
                    MATCH (o:Ontology {id: $ontology_id})-[:CONTAINS]->(kc:KeywordCategory)
                    RETURN kc.id as id
                """
                params = {"ontology_id": ontology_id}
            else:
                if not pattern:
                    log(WARNING, "delete_keyword_categories: no pattern provided and --all not set")
                    return 0
                # match by regex on text_en
                query = """
                    MATCH (o:Ontology {id: $ontology_id})-[:CONTAINS]->(kc:KeywordCategory)
                    WHERE kc.text_en =~ $regex
                    RETURN kc.id as id
                """
                params = {"ontology_id": ontology_id, "regex": pattern}

            result = execute_neo4j_query(query, params)
            if not result:
                return 0

            ids = [r.get('id') for r in result if r.get('id')]
            if not ids:
                return 0

            # delete by id list
            delete_query = """
                MATCH (kc:KeywordCategory) WHERE kc.id IN $ids
                DETACH DELETE kc
                RETURN count(kc) as deleted_count
            """
            del_params = {"ids": ids}
            del_result = execute_neo4j_query(delete_query, del_params)
            if del_result and isinstance(del_result, list) and len(del_result) > 0:
                try:
                    return int(del_result[0].get('deleted_count') or len(ids))
                except Exception:
                    return len(ids)
            return len(ids)
        except Exception as e:
            log(ERROR, f"ERROR in delete_keyword_categories: {e}\n{traceback.format_exc()}")
            return 0
            return False

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
        try:
            # Get all valid categories for this ontology
            # Note: Include proposed categories in validation - they should still be queryable
            valid_categories = self.get_categories_for_ontology(ontology_id, exclude_proposed=False)
            if not valid_categories:
                log(DEBUG, f"is_categories_only_query: No categories found for ontology '{ontology_id}'")
                return False
            
            # Build a set of valid category names (text_en values)
            valid_category_names = {cat.get('text_en') for cat in valid_categories if cat.get('text_en')}
            log(DEBUG, f"is_categories_only_query: Found {len(valid_category_names)} valid categories")
            
            # Tokenize the query using the existing tokenizer
            tokens = self._tokenize_query(query)
            log(DEBUG, f"is_categories_only_query: Tokens: {tokens}")
            
            # Check each token
            for token in tokens:
                # Skip operators and parentheses
                if token in ('AND', 'OR', 'NOT', '(', ')'):
                    continue
                
                # Extract category name from token
                # Token format is either "CATEGORY" or "CATEGORY=VALUE1,VALUE2"
                if '=' in token:
                    category_name = token.split('=', 1)[0].strip()
                else:
                    category_name = token.strip()
                
                # Check if this category exists in the ontology
                if category_name not in valid_category_names:
                    log(DEBUG, f"is_categories_only_query: Category '{category_name}' not found in ontology '{ontology_id}'")
                    return False
            
            log(DEBUG, f"is_categories_only_query: All categories valid for query '{query}'")
            return True
            
        except Exception as e:
            log(ERROR, f"ERROR in is_categories_only_query: {e}\n{traceback.format_exc()}")
            return False
    
    def categories_query(self, ontology_id: str, query: str, effective_user_id: Optional[str] = None) -> List[str]:
        """Search for documents based on category assignments using boolean queries.
        
        This method implements a boolean search algorithm for finding documents based on 
        their category assignments. It parses boolean expressions with AND/OR operators,
        parentheses, and category filters.
        
        Args:
            ontology_id: The ontology identifier to search within
            query: Boolean expression query. Examples:
                - Simple: "IS_DEAD"
                - With value: "COUNTRY=IQ"
                - Multiple values: "DUBLIN_ARTICLE=ART_13,ART_25_2"
                - Complex: "(COUNTRY=IQ OR COUNTRY=SY) AND IS_DEAD"
            effective_user_id: Neo4j User.id of the requesting user. When provided, only documents
                      the effective user holds GRANT:READ on are returned. When None, ACL is skipped.
        
        Returns:
            List of document IDs matching the query
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        from src.config.constants import ACL_ENFORCEMENT_ENABLED
        
        try:
            log(DEBUG, f"thalabus_query: ontology_id={ontology_id}, query={query}")
            
            # Parse the query into an AST
            follows_clause = ""
            tokens = self._tokenize_query(query)
            log(DEBUG, f"Tokens: {tokens}")
            
            ast, _ = self._parse_expression(tokens, 0)
            log(DEBUG, f"AST: {ast}")
            
            # Generate Cypher WHERE clause from AST
            where_clause, var_counter = self._generate_cypher_from_ast(ast, 1)
            log(DEBUG, f"WHERE clause: {where_clause}")

            if ontology_id is not None and len(ontology_id) > 0:
                follows_clause = "-[:FOLLOWS]->(o:Ontology {id: $ontology_id})"

            # ACL enforcement: append GRANT:READ check on the Document node
            acl_active = ACL_ENFORCEMENT_ENABLED and effective_user_id is not None
            if acl_active:
                acl_block = acl_document_where_clause("d")
                full_where = f"{where_clause}{acl_block}"
            else:
                full_where = where_clause

            # Build complete Cypher query
            cypher_query = f"""
                MATCH (d:Document){follows_clause}
                WHERE {full_where}
                RETURN DISTINCT d.id AS document_id
            """
            
            parameters = {"ontology_id": ontology_id}
            if acl_active:
                parameters["effectiveUserId"] = effective_user_id
            log(DEBUG, f"Executing Cypher: {cypher_query}")
            
            result = execute_neo4j_query(cypher_query, parameters)
            
            if result is None:
                log(INFO, "thalabus_query: No results found")
                return []
            
            doc_ids = [record.get('document_id') for record in result if record.get('document_id')]
            log(DEBUG, f"thalabus_query: Found {len(doc_ids)} documents")
            
            return doc_ids
            
        except Exception as e:
            log(ERROR, f"ERROR in thalabus_query: {e}\n{traceback.format_exc()}")
            return []

    def get_ontology_for_query(self, query: str) -> Optional[str]:
        """
        Heuristically determine an ontology id for a given thalabus query string.

        Strategy:
        - Extract candidate category names from the query (tokens that look like
          category identifiers e.g. COUNTRY_OF_ORIGIN, IS_DEAD, DUBLIN_ARTICLE)
        - For each candidate, search for a KeywordCategory node where
          `k.text_en == candidate` and return the `id` attribute of the
          parent Ontology `(o:Ontology)-[:CONTAINS]->(k)` if found.

        Returns the first matching ontology id or `None` if no mapping found.
        """
        if query is None:
            return None

        try:
            # Find candidate category names: uppercase words with optional underscores/numbers
            # that are likely used as category identifiers in thalabus queries.
            candidates = set()
            # Look for tokens before '=' first (e.g. COUNTRY_OF_ORIGIN=...)
            for m in re.finditer(r"\b([A-Z][A-Z0-9_]+)\s*(?==)", query):
                candidates.add(m.group(1))

            # If none found, also accept standalone uppercase tokens (e.g. IS_DEAD)
            if not candidates:
                for m in re.finditer(r"\b([A-Z][A-Z0-9_]+)\b", query):
                    candidates.add(m.group(1))

            if not candidates:
                return None

            for cat in candidates:
                cypher = """
                    MATCH (o:Ontology)-[:CONTAINS]->(k:KeywordCategory {text_en: $text_en})
                    RETURN o.id AS ontology_id LIMIT 1
                """
                params = {"text_en": cat}
                result = execute_neo4j_query(cypher, params)
                if result and len(result) > 0:
                    ontology_id = result[0].get("ontology_id")
                    if ontology_id:
                        log(DEBUG, f"{self._log_prefix}: get_ontology_for_query() found ontology '{ontology_id}' for category '{cat}'")
                        return ontology_id

        except Exception as e:
            log(ERROR, f"{self._log_prefix}: Error in get_ontology_for_query(): {e}")

        return None
    
    def get_ontology_id_for_category(self, category_text_en: str) -> Optional[str]:
        """Retrieve the ontology ID that contains the given category (by text_en).
        
        Args:
            category_text_en: The text_en value of the KeywordCategory (e.g., 'MAIN_EVENT').
        
        Returns:
            The ontology ID as a string, or None if not found.
        """
        try:
            query = """
                MATCH (kc:KeywordCategory {text_en: $category_text_en})
                MATCH (o:Ontology)-[:CONTAINS]->(kc)
                RETURN o.id AS ontology_id
                LIMIT 1
            """
            params = {"category_text_en": category_text_en}
            rows = execute_neo4j_query(query, params)
            
            if rows and len(rows) > 0:
                ontology_id = rows[0].get("ontology_id")
                log(DEBUG, f"{self._log_prefix}: Found ontology '{ontology_id}' for category '{category_text_en}'")
                return ontology_id
            else:
                log(WARNING, f"{self._log_prefix}: No ontology found for category '{category_text_en}'")
                return None
        
        except Exception as e:
            error_message = f"Error retrieving ontology for category: {e}"
            log(ERROR, f"{self._log_prefix}: {error_message}")
            return None
    
    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize the boolean query string.
        
        Returns list of tokens: parentheses, AND, OR, NOT, or category expressions
        """
        tokens = []
        i = 0
        query = query.strip()
        
        while i < len(query):
            # Skip whitespace
            if query[i].isspace():
                i += 1
                continue
            
            # Handle parentheses
            if query[i] in '()':
                tokens.append(query[i])
                i += 1
                continue
            
            # Check for NOT operator (must check before AND to avoid partial match)
            if i + 3 <= len(query) and query[i:i+3].upper() == 'NOT':
                # Treat underscores as part of identifiers, not word boundaries
                prev_is_ident = i > 0 and (query[i-1].isalnum() or query[i-1] == '_')
                next_is_ident = i+3 < len(query) and (query[i+3].isalnum() or query[i+3] == '_')
                if not prev_is_ident and not next_is_ident:
                    tokens.append('NOT')
                    i += 3
                    continue
            
            # Check for AND/OR operators
            if i + 3 <= len(query) and query[i:i+3].upper() == 'AND':
                # Treat underscores as part of identifiers, not word boundaries
                prev_is_ident = i > 0 and (query[i-1].isalnum() or query[i-1] == '_')
                next_is_ident = i+3 < len(query) and (query[i+3].isalnum() or query[i+3] == '_')
                if not prev_is_ident and not next_is_ident:
                    tokens.append('AND')
                    i += 3
                    continue
            
            if i + 2 <= len(query) and query[i:i+2].upper() == 'OR':
                # Treat underscores as part of identifiers, not word boundaries
                prev_is_ident = i > 0 and (query[i-1].isalnum() or query[i-1] == '_')
                next_is_ident = i+2 < len(query) and (query[i+2].isalnum() or query[i+2] == '_')
                if not prev_is_ident and not next_is_ident:
                    tokens.append('OR')
                    i += 2
                    continue
            
            # Extract category expression (e.g., CATEGORY or CATEGORY=VAL1,VAL2 or CATEGORY="VAL1, VAL2")
            # Handle quoted values that may contain spaces and commas
            start = i
            if query[i:i+1] not in '()':
                # Check if this looks like a category (identifier followed by optional =)
                j = i
                while j < len(query) and (query[j].isalnum() or query[j] in '_'):
                    j += 1
                
                # If we have = after identifier, extract the category=values part
                if j > i and j < len(query) and query[j] == '=':
                    # Extract identifier
                    i = j + 1  # skip the =
                    # Now extract the value part, which may contain quoted strings
                    while i < len(query):
                        # Stop at operators or parentheses
                        if query[i] in '()':
                            break
                        # Check for AND, OR, NOT operators (treat underscores as identifier chars)
                        if i > 0 and query[i] in 'AON' and query[i].isupper():
                            prev_is_ident = (query[i-1].isalnum() or query[i-1] == '_')
                            if query[i:i+3].upper() == 'AND':
                                next_is_ident = i+3 < len(query) and (query[i+3].isalnum() or query[i+3] == '_')
                                if not prev_is_ident and not next_is_ident:
                                    break
                            elif query[i:i+2].upper() == 'OR':
                                next_is_ident = i+2 < len(query) and (query[i+2].isalnum() or query[i+2] == '_')
                                if not prev_is_ident and not next_is_ident:
                                    break
                            elif query[i:i+3].upper() == 'NOT':
                                next_is_ident = i+3 < len(query) and (query[i+3].isalnum() or query[i+3] == '_')
                                if not prev_is_ident and not next_is_ident:
                                    break
                        
                        # Skip quoted strings (preserve them with quotes)
                        if query[i] == '"':
                            i += 1
                            while i < len(query) and query[i] != '"':
                                if query[i] == '\\' and i + 1 < len(query):
                                    i += 2
                                else:
                                    i += 1
                            if i < len(query):
                                i += 1  # skip closing "
                        else:
                            i += 1
                    
                    token = query[start:i].strip()
                else:
                    # No = sign, just extract the category name
                    while i < len(query) and query[i] not in '() \t\n\r':
                        i += 1
                    token = query[start:i].strip()
            else:
                token = ''
            
            if token and token.upper() not in ('AND', 'OR'):
                tokens.append(token)
        
        return tokens
    
    def _parse_expression(self, tokens: List[str], pos: int) -> tuple:
        """Parse OR expression (lowest precedence).
        
        Returns (AST_node, next_position)
        """
        left, pos = self._parse_and_expression(tokens, pos)
        
        while pos < len(tokens) and tokens[pos] == 'OR':
            pos += 1  # consume OR
            right, pos = self._parse_and_expression(tokens, pos)
            left = {'op': 'OR', 'left': left, 'right': right}
        
        return left, pos
    
    def _parse_and_expression(self, tokens: List[str], pos: int) -> tuple:
        """Parse AND expression (higher precedence than OR).
        
        Returns (AST_node, next_position)
        """
        left, pos = self._parse_not_expression(tokens, pos)
        
        while pos < len(tokens) and tokens[pos] == 'AND':
            pos += 1  # consume AND
            right, pos = self._parse_not_expression(tokens, pos)
            left = {'op': 'AND', 'left': left, 'right': right}
        
        return left, pos
    
    def _parse_not_expression(self, tokens: List[str], pos: int) -> tuple:
        """Parse NOT expression (highest precedence, applied before primary).
        
        Returns (AST_node, next_position)
        """
        if pos < len(tokens) and tokens[pos] == 'NOT':
            pos += 1  # consume NOT
            expr, pos = self._parse_not_expression(tokens, pos)  # Allow multiple NOTs
            return {'op': 'NOT', 'operand': expr}, pos
        else:
            return self._parse_primary(tokens, pos)
    
    def _parse_primary(self, tokens: List[str], pos: int) -> tuple:
        """Parse primary expression (parentheses or category).
        
        Returns (AST_node, next_position)
        """
        if pos >= len(tokens):
            raise ValueError("Unexpected end of query")
        
        token = tokens[pos]
        
        # Handle parentheses
        if token == '(':
            pos += 1  # consume '('
            expr, pos = self._parse_expression(tokens, pos)
            if pos >= len(tokens) or tokens[pos] != ')':
                raise ValueError("Missing closing parenthesis")
            pos += 1  # consume ')'
            return expr, pos
        
        # Handle category expression
        if '=' in token:
            category, values = token.split('=', 1)
            value_list = [v.strip() for v in values.split(',')]
            return {'type': 'category', 'name': category.strip(), 'values': value_list}, pos + 1
        else:
            return {'type': 'category', 'name': token.strip(), 'values': None}, pos + 1
    
    def _generate_cypher_from_ast(self, node: dict, var_counter: int) -> tuple:
        """Generate Cypher WHERE clause from AST.
        
        Returns (where_clause_string, next_var_counter)
        """
        if 'op' in node:
            if node['op'] == 'AND':
                left_clause, var_counter = self._generate_cypher_from_ast(node['left'], var_counter)
                right_clause, var_counter = self._generate_cypher_from_ast(node['right'], var_counter)
                return f"({left_clause} AND {right_clause})", var_counter
            
            elif node['op'] == 'OR':
                left_clause, var_counter = self._generate_cypher_from_ast(node['left'], var_counter)
                right_clause, var_counter = self._generate_cypher_from_ast(node['right'], var_counter)
                return f"({left_clause} OR {right_clause})", var_counter
            
            elif node['op'] == 'NOT':
                operand_clause, var_counter = self._generate_cypher_from_ast(node['operand'], var_counter)
                return f"(NOT {operand_clause})", var_counter
            
            else:
                raise ValueError(f"Unknown operator: {node['op']}")
        
        elif node.get('type') == 'category':
            category_name = node['name']
            values = node['values']
            
            if values is None:
                # Simple category check (any value)
                clause = f"EXISTS {{ MATCH (d)-[:HAS_CATEGORY]->(c{var_counter}:Category)-[:OF_CATEGORY]->(kc{var_counter}:KeywordCategory {{text_en: '{category_name}'}}) }}"
                return clause, var_counter + 1
            else:
                # Category with specific values (OR between them)
                subclauses = []
                for value in values:
                    clause = f"EXISTS {{ MATCH (d)-[:HAS_CATEGORY]->(c{var_counter}:Category {{text: '{value}'}})-[:OF_CATEGORY]->(kc{var_counter}:KeywordCategory {{text_en: '{category_name}'}}) }}"
                    subclauses.append(clause)
                    var_counter += 1
                
                combined = " OR ".join(subclauses)
                return f"({combined})", var_counter
        
        else:
            raise ValueError(f"Unknown AST node type: {node}")

    def get_lexicon_for_reasoning_document(self, document_id: str, target_language: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return Lexicon entries linked to the ontology(ies) associated with a ReasoningDocument.

        Follows the graph path:
            (ReasoningDocument {id: document_id}) -[:REASONS_WITH|OWNS]- (Dossier)
            -[:FOLLOWS]-> (Ontology) -[:HAS_LEXICON]-> (Lexicon)
        """
        try:
            query = """
                MATCH (rd:ReasoningDocument {id: $document_id})<-[:REASONS_WITH|OWNS]-(d:Dossier)
                      -[:FOLLOWS]->(o:Ontology)-[:HAS_LEXICON]->(l:Lexicon)
                WHERE $target_language IS NULL OR l.lang = $target_language
                RETURN DISTINCT o.id AS ontology_id, l.id AS lexicon_id, l.lang AS lang, l.text AS text
            """
            params = {
                "document_id": document_id,
                "target_language": target_language,
            }
            rows = execute_neo4j_query(query, params) or []
            return [
                {
                    "ontology_id": r.get("ontology_id"),
                    "lexicon_id":  r.get("lexicon_id"),
                    "lang":        r.get("lang"),
                    "text":        r.get("text"),
                }
                for r in rows
            ]
        except Exception as e:
            log(ERROR, f"{self._log_prefix}: Error in get_lexicon_for_reasoning_document('{document_id}'): {e}")
            return []


# module level singleton accessors (same pattern as other services)
_ontology_service_instance: Optional[OntologyInterface] = None


def get_ontology_service() -> OntologyInterface:
    global _ontology_service_instance
    if _ontology_service_instance is None:
        _ontology_service_instance = OntologyService()
    return _ontology_service_instance


def set_ontology_service(service: OntologyInterface) -> None:
    global _ontology_service_instance
    _ontology_service_instance = service


log_end_file(__file__)


if __name__ == "__main__":
    try:
        from src.utils.log import log_init, log_exit
        log_init()

        ontology_service = OntologyService()

        if True:
            # Test _get_ontology_as_string() with ontology "FictionalStories"
            for ontology_id in ['FictionalStories']:
                print(f"\n{'='*60}")
                print(f"Ontology: {ontology_id}")
                print('='*60)
                try:
                    result = ontology_service._get_ontology_as_string(
                        ontology_id=ontology_id,
                        ontology_description=None,
                        include_attributes=True
                    )
                    print(result)
                except Exception as ex:
                    log(ERROR, f"Error for ontology '{ontology_id}': {ex}")

        if False:
            # simple test
            test_query = '(COUNTRY_OF_ORIGIN=IQ OR COUNTRY_OF_ORIGIN=SY)'
            ontology_id = ontology_service.get_ontology_for_query(test_query)
            log(INFO, f"Ontology ID for query '{test_query}': {ontology_id}")

    except Exception as e:
        log(FATAL, f"Error in ontology_service main test: {e}\n{traceback.format_exc()}")

    finally:
        log_exit()
