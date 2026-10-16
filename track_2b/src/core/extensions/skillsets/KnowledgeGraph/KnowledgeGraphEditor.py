from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import json
import traceback
from typing import Annotated, Dict, Optional, Union, List, Tuple

from src.core.base.InputOutput import execute_neo4j_query, prepare_string_for_json
from src.interfaces.documents_interface import DocumentsInterface
from src.services.ontology_service import RESERVED_ONTOLOGY_LABELS, RESERVED_ONTOLOGY_ATTRIBUTES, CASE_SENSITIVE_ATTRIBUTES, COMMUNITY_ID_DEFAULT, COMMUNITY_NAME_DEFAULT, ATTRIBUTE_REQUIRES_CONSOLIDATION, CONSOLIDATION_SUFFIX, get_ontology_service
from src.services.ontology_source_helper import build_doc_source_id, add_token_to_source_id_list
from src.core.base.ResourceMgmt import get_lock
from src.utils.ontology_label_validator import validate_ontology_label
from src.utils.acl_cypher_utils import acl_document_permission_check_query
from src.utils.log import log, DEBUG, INFO, WARNING, ERROR
from datetime import datetime, timezone

IMMEDIATE_CONSOLIDATION = False

# Must stay byte-identical to the type registered in parameters.py and to the entry in pathrag_service._EXCLUDED_REL_TYPES.
IS_DOCUMENTED_RELATIONSHIP_TYPE = "IS_DOCUMENTED"
IS_DOCUMENTED_KINDS = ("curated", "extracted")


def _check_ontology_permission(ontology_id: Optional[str], effective_user_id: str, permission: str) -> bool:
    """Check one ontology permission without revealing whether the resource exists."""
    if not ontology_id:
        return False
    return get_ontology_service().check_ontology_permission(effective_user_id, ontology_id, permission)


def _get_entity_ontology_ids(entity_ids: List[str]) -> List[str]:
    """Resolve the parent ontologies for entities involved in a mutation."""
    if not entity_ids:
        return []
    query = """
        MATCH (o:Ontology)-[:DEFINES]->(:Community)-[:AGGREGATES]->(e:Entity)
        WHERE e.id IN $entity_ids
        RETURN DISTINCT o.id AS ontology_id
    """
    result = execute_neo4j_query(query, {"entity_ids": entity_ids})
    return [record["ontology_id"] for record in result or [] if record.get("ontology_id")]


def _check_entity_permissions(entity_ids: List[str], effective_user_id: str, permission: str) -> bool:
    """Require the requested permission on every entity's parent ontology."""
    ontology_ids = _get_entity_ontology_ids(entity_ids)
    return bool(ontology_ids) and all(
        _check_ontology_permission(ontology_id, effective_user_id, permission)
        for ontology_id in ontology_ids
    )


def _check_document_permission(document_id: str, effective_user_id: str, permission: str) -> bool:
    """Check one document permission without revealing whether the document exists."""
    has_access: bool = False

    if document_id and effective_user_id:
        try:
            query = acl_document_permission_check_query(permission)
            records = execute_neo4j_query(query, {"effective_user_id": effective_user_id, "document_id": document_id})
            if records:
                has_access = bool(records[0].get("has_access", False))
        except Exception as e:
            log(ERROR, f"KnowledgeGraphEditor._check_document_permission: error checking {permission} on '{document_id}' for '{effective_user_id}': {e}")

    return has_access

## Attributes storage
# The attributes are stored in the neo4j nodes as properties. The attributes are stored as key-value pairs.
# The key is the attribute name, and the value is the json structure (converted to string) of:
# - the attribute value
# - the document or chapter id, where the attribute value has been sourced from
# - the paragraph id, where the attribute value has been sourced from
#
## Attributes consolidation
# There are two reasons to consolidate the stored values of an attribute:
# - to merge the values of the same attribute from different sources
# - to embed the values of the same attribute in the search index
# The consolidation is done by merging the values of the same attribute from different sources,
# and storing the merged value into an additional attribute with the same name, but with the suffix "_consolidated".
# Upon adding or replacing the value of an attribute, the consolidation is triggered by removing the "_consolidated" attribute and setting the special attribute "requires_consolidation=true".
# A periodic task looks for unconsolidated attributes and consolidates them. This is located in the "Utils" container and runs approx. every 1 minute.
#
# Knowledge base language
# Unconsolidated attributes are stored in the information source's (i.e. document's) original language. Reasoning: better not to mess up with the original sources, we would risk to misrepresent the information.
# Consolidated attributes are stored in English. Reasoning: 1) different sources may be in different languages, we seek a common universal language here. 2) the search index is built from it, and works best in English.
# Other key attributes such as "aggregate_values", "accuracy_description", are stored in English. Reasoning: they are internal attributes, not visible to the user. We choose the best language understood by LLM and developers.
#
## Semantic search with embeddings
# During consolidation, an extra attribute is calculated: the "aggregate_values". It is a list of all values of the consolidated attributes.
# The search index is built from it, using the LLM embeddings.
# The search index is used for semantic search with embeddings.
#
## Information accuracy
# It is possible that multiple sources diverge regarding the same attribute of an entity.
# In this case, we add a new key/value pair to the attribute value: "accuracy" with a value between 0.0 and 1.0.
# The accuracy value is calculated by LLM.
# The default value for accuracy is 1.0. If the key is not present, the accuracy is assumed to be 1.0.
# An optional additional attribute is "accuracy_description" with a description of the accuracy value.
#
## EXAMPLE 1:
# Document docid-19 is parsed for first time, and the entities are extracted. The entities are stored in the neo4j database:
#   <p id='p-7'>Bummy is a nice-looking adorable bunny, owned by a 14-years old girl whose name is Jeanny</p>
# Extracted attributes:
#   Entity: 'bummy', label: 'Entity' / 'Animal'
#   - attribute "name": [{ value: "Bummy", "source_id": "doc://docid-19?p=p-7" }]
#   - attribute "name_consolidated": "Bummy"
#   - attribute "description": [{ value: "Bummy is a nice-looking adorable bunny, owned by Jeanny", "source_id": "doc://docid-19?p=p-7" }]
#   - attribute "description_consolidated": "Bummy is a nice bunny, owned by Jeanny"
#   Entity: 'jeanny', label: 'Person'
#   - attribute "name": { value: "Jeanny", "source_id": "doc://docid-19?p=p-7" }
#   - attribute "name_consolidated": "Jeanny"
#   - attribute "age": { value: "14", "source_id": "doc://docid-19?p=p-7" }
#   - attribute "age_consolidated": "14"
#
## EXAMPLE 2:
# New information is added based on Document docid-20:
#   <p id='p-8'>Bummy is really ugly, it has a wide scar on the face and it's scary to look</p>
# Extracted attributes (we describe only the changes for Bummy; its name and age are as in the previous example):
#   Entity: 'bummy', label: 'Entity' / 'Animal'
#   - attribute "description": [
#                               { value: "Bummy is a nice-looking adorable bunny, owned by Jeanny", "source_id": "doc://docid-19?p=p-7" },
#                               { value: "Bummy is really ugly, it has a wide scar on the face and it's scary to look", "source_id": "doc://docid-20?p=p-8" }
#                              ]
#
# Result after consolidation (all attributes are shown):
#   Entity: 'bummy', label: 'Entity' / 'Animal'
#   - attribute "name": [{ value: "Bummy", "source_id": "doc://docid-19?p=p-7" }]
#   - attribute "name_consolidated": "Bummy"
#   - attribute "description": [
#                               { value: "Bummy is a nice-looking adorable bunny, owned by Jeanny", "source_id": "doc://docid-19?p=p-7" },
#                               { value: "Bummy is really ugly, it has a wide scar on the face and it's scary to look", "source_id": "doc://docid-20?p=p-8" }
#                              ]
#   - attribute "description_consolidated": "Bummy is or was a nice bunny, owned by Jeanny. Bummy is or was really ugly, it has a wide scar on the face and it's scary to look. Its aspect is unclear, sources are conflicting."
#   - attribute "accuracy": 0.8
#   - attribute "accuracy_description": "All attributes are OK except 'description', which is not accurate, because it contains conflicting information: one source says that Bummy is nice, the other source says that Bummy is ugly."
#   - attribute "aggregate_values": [name:"Bummy", description:"Bummy is or was a nice bunny, owned by Jeanny. Bummy is or was really ugly, it has a wide scar on the face and it's scary to look. Its aspect is unclear, sources are conflicting."]


def community_create(id:str, name:str, description:str, ontology_id:str) -> bool:
    """
    Creates a new entity group (an entity group is named 'Community').
    """
    query = """ MATCH (o:Ontology {id: $ontology_id})
                MERGE (c:Community:Dataset {id: $id})
                SET 
                    c.description = $description
                MERGE (o)-[:DEFINES]->(c)
    """
    parameters = {"id": id, "name": name, "description": description, "ontology_id": ontology_id}
    result = execute_neo4j_query(query, parameters)
    if not result:
        log(ERROR, f"ERROR: Failed to create community '{name}' with id '{id}' for ontology '{ontology_id}'")
        return False
    return True

def community_get_entities(community_id: str) -> List[dict]:
    """
    Lists all entities in a community.
    """
    query = """MATCH (c:Community {id: $community_id})-[:AGGREGATES]->(e:Entity) RETURN
        e.id AS id,
        e.name AS name,
        e.description AS description
    """
    parameters = {"community_id": community_id}
    query_result = execute_neo4j_query(query, parameters)
    entities = []
    for record in query_result:
        entities.append({
            "id": record["id"],
            "name": record["name"],
            "description": record["description"]
        })

    return entities

def merge_attribute(original_attribute: dict, new_attribute_dict_list:List[Dict[str,str]], case_sensitive:bool) -> Tuple[List[dict], bool]:
    """ Adds or updates the value of an entity's attribute.
    With respect to the example at the top of this file, this function merges attributes of the same entity, where new information is provided.

    Returns:
    - the merged attribute, if successful
    - boolean to indicate a change happened (True) or not (False)

    Example 1: new value for an attribute
       before the merge:
         - attribute "description": { value: "Bummy is a nice-looking adorable bunny, owned by Jeanny", "source_id": "doc://docid-19?p=p-7" }
       
       new_attribute_dict: { "value": "Bummy is really ugly, it has a wide scar on the face and it's scary to look", "source_id": "doc://docid-20?p=p-8" }

       after the merge:
         - attribute "description": [
                               { value: "Bummy is a nice-looking adorable bunny, owned by Jeanny", "source_id": "doc://docid-19?p=p-7" },
                               { value: "Bummy is really ugly, it has a wide scar on the face and it's scary to look", "source_id": "doc://docid-20?p=p-8" }
                              ]
       Return value: True because the attribute value has changed

    Example 2: confirmation of same value for an attribute, from a different source
       before the merge:
         - attribute "name": { value: "Bummy", "source_id": "doc://docid-19?p=p-7" }
       
       new_attribute_value: { "value": "Bummy", "source_id": "doc://docid-20?p=p-8" }

       after the merge:
         - attribute : [{ value: "Bummy", "source_id": "doc://docid-19?p=p-7,doc://docid-20?p=p-8" }]

       Return value: False because the attribute value has not changed. The source_id changed but the value is the same.
    """
    new_attribute = None
    changed = False

    if original_attribute is None:
        # the attribute is new, therefore we create it and we signal a change
        new_attribute = new_attribute_dict_list
        changed = True
    else:
        # the attribute exists, we need to merge the new value with the existing value
        # first, search if the new value is already present in the attribute
        found = False
        for attribute in original_attribute:
            # compare the values in lowercase (not nice in case of case-sensitive values, such as names)
            if not isinstance(attribute["value"], str):
                # only strings can be compared in lowercase
                case_sensitive = True

            for new_attribute_dict in new_attribute_dict_list:
                new_value = new_attribute_dict["value"]
                new_source = new_attribute_dict["source_id"]
                if (case_sensitive == True and attribute["value"] == new_value) or (case_sensitive == False and attribute["value"].lower() == new_value.lower()):
                    attribute["source_id"], source_added = add_token_to_source_id_list(attribute["source_id"], new_source)
                    found = True
                    changed = source_added
                    break

        if not found:
            # the value is new, we add it to the attribute
            for new_attribute_dict in new_attribute_dict_list:
                original_attribute.append({"value": new_attribute_dict["value"], "source_id": new_attribute_dict["source_id"]})
            changed = bool(new_attribute_dict_list)

        new_attribute = original_attribute

    # Convert the merged attribute to a JSON string
    merged_attribute_json = json.dumps(new_attribute)
    return merged_attribute_json, changed

def merge_attributes(original_attributes: List[Dict[str, dict]], new_attributes: Dict[str, str], source:str) -> Tuple[Dict[str, dict], bool]:
    """ Adds or updates attributes of an entity.
    """
    result = {}
    changed = False

    # Sanitize keys of incoming attributes to ensure Neo4j compatibility
    new_attributes = {_sanitize_key(k): v for k, v in new_attributes.items()}

    # Iterate through the new attributes and merge them with the original attributes. Avoid reserved attributes, e.g. "id".
    for key, new_attribute_value in new_attributes.items():
        try:
            # if key not in RESERVED_ONTOLOGY_ATTRIBUTES and not key.endswith(CONSOLIDATION_SUFFIX):
            if get_ontology_service().is_not_reserved_and_unconsolidated_attribute(key):
                case_sensitive = key in CASE_SENSITIVE_ATTRIBUTES
                original_attribute = None

                if len(original_attributes) > 0:
                    for attribute_key, attribute_value in original_attributes.items():
                        if attribute_key == key:
                            try:
                                original_attribute = json.loads(attribute_value)
                                break
                            except:
                                pass

                if isinstance(new_attribute_value, str) and new_attribute_value.startswith("[") and new_attribute_value.endswith("]"):
                    new_attribute_dict_list = json.loads(new_attribute_value)
                elif isinstance(new_attribute_value, list):
                    new_attribute_dict_list = [{"value": item, "source_id": source} for item in new_attribute_value if item is not None]
                else:
                    new_attribute_dict_list = [{"value": new_attribute_value, "source_id": source}]
                merged_attribute, changed_attribute = merge_attribute(original_attribute, new_attribute_dict_list, case_sensitive=case_sensitive)
                result[key] = merged_attribute
                if changed_attribute:
                    changed = True
        except Exception as e:
            stack_trace = traceback.format_exc()
            log(ERROR, f"ERROR: An error occurred while merging '{key}' attribute, value '{new_attribute_value}': {e}\n{stack_trace}")

    return result, changed

def merge_relationships(entity_id: str, second_entity_id: str) -> None:
    """
    Merges relationships of two entities, where the relationship label is the same. In that case, all attributes are merged
    """
    query = """MATCH (e1:Entity {id: $entity_id})-[r]->(e2:Entity {id: $second_entity_id}) RETURN r"""
    parameters = {"entity_id": entity_id, "second_entity_id": second_entity_id}
    result = execute_neo4j_query(query, parameters)
    if result and len(result) > 0:
        # get a list of all relationships
        relationships = []
        for record in result:
            relationship = record["r"]
            relationship_id = relationship.element_id
            relationship_type = relationship.type
            relationship_properties = dict(relationship.items())
            relationships.append({"id": relationship_id, "type": relationship_type, "properties": relationship_properties})

        # identify duplicates in a dictionary where the key is the relationship type, and the value is a list of relationships
        duplicates = {}
        for relationship in relationships:
            relationship_id = relationship["id"]
            relationship_type = relationship["type"]
            relationship_properties = relationship["properties"]
            if relationship_type in duplicates:
                duplicates[relationship_type].append(relationship)
            else:
                duplicates[relationship_type] = [relationship]

        # merge the duplicates
        for relationship_type, relationship_list in duplicates.items():
            if len(relationship_list) > 1:
                # merge the relationships. Always take the first relationship as the base, and merge the other relationships into it.
                # we use the merge_attributes() function to merge the attributes of the relationship pairs, keeping the first relationship as the base.
                relationship_element_id_0 = relationship_list[0]["id"]
                relationship_properties_0 = relationship_list[0]["properties"]
                for i in range(1, len(relationship_list)):
                    relationship_element_id = relationship_list[i]["id"]
                    relationship_properties = relationship_list[i]["properties"]
                    new_attributes, _ = merge_attributes(relationship_properties_0, relationship_properties, source=None)
                    if new_attributes:
                        # update the relationship with the merged attributes
                        set_clause = ", ".join([f"r.`{key}` = $`{key}`" for key in new_attributes.keys()])
                        query = f"""MATCH (e1:Entity {{id: $entity_id}})-[r:{relationship_type}]->(e2:Entity {{id: $second_entity_id}})
                                    WHERE elementId(r) = $relationship_element_id
                                    SET {set_clause}"""
                        parameters = {"entity_id": entity_id, "second_entity_id": second_entity_id, "relationship_element_id": relationship_element_id_0, **new_attributes}
                        result = execute_neo4j_query(query, parameters)

                        # delete the merged relationship
                        query = f"""MATCH (e1:Entity {{id: $entity_id}})-[r:{relationship_type}]->(e2:Entity {{id: $second_entity_id}})
                                    WHERE elementId(r) = $relationship_element_id
                                    DELETE r"""
                        parameters = {"entity_id": entity_id, "second_entity_id": second_entity_id, "relationship_element_id": relationship_element_id}
                        result = execute_neo4j_query(query, parameters)


def entity_create_or_update(entity_id: str, community_id: str, label: str, attributes:Dict[str,str], document_or_chapter_id:str, paragraph_id:str) -> bool:
    """
    Creates a new entity or updates an existing entity.
    """
    success = True
    validate_ontology_label(label)
    source_id = build_doc_source_id(document_or_chapter_id, paragraph_id)

    if attributes is not None and len(attributes) > 0:
        # merge existing values in the attributes, with the new values. First,
        # we need to fetch the existing values; then, we merge them with the new values.
        query = """MATCH (e:Entity {id: $entity_id}) RETURN e AS entity"""
        parameters = {"entity_id": entity_id}
        result = execute_neo4j_query(query, parameters)
        original_attributes = []
        if result and len(result) > 0:
            entity = result[0]["entity"]
            original_attributes = dict(entity.items())

        # only a brand-new entity (no prior node under this id) can collide with another entity's primary key
        is_new_entity = not original_attributes
        conflicting_entity_id = get_ontology_service().find_conflicting_entity_by_primary_key(label, entity_id, sanitize_attribute_keys(attributes)) if is_new_entity and label else None

        if conflicting_entity_id:
            log(WARNING, f"KGFromDocument: entity_create_or_update: refusing to create entity '{entity_id}' ({label}) because its primary key value already exists on entity '{conflicting_entity_id}'.")
            query = None
            success = False
        else:
            merged_attributes, requires_consolidation = merge_attributes(original_attributes, attributes, source_id)

            if len(merged_attributes) > 0 or requires_consolidation:
                set_clause = ", ".join([f"e.`{key}` = $`{key}`" for key in merged_attributes.keys()])
                if requires_consolidation:
                    set_clause += f", e.{ATTRIBUTE_REQUIRES_CONSOLIDATION} = true"
                query = f"""MERGE (e:Entity:{label} {{id: $entity_id}}) SET {set_clause} RETURN e"""
                parameters = {"entity_id": entity_id, **merged_attributes}
            else:
                # all attributes were filtered out (reserved or duplicates) — still ensure the entity node exists
                query = f"""MERGE (e:Entity:{label} {{id: $entity_id}}) RETURN e"""
                parameters = {"entity_id": entity_id}

    else:
        query = f"""MERGE (e:Entity:{label} {{id: $entity_id}}) RETURN e""" if label else """MERGE (e:Entity {id: $entity_id}) RETURN e"""
        parameters = {"entity_id": entity_id}

    if query is not None:
        try:
            log(DEBUG, f"KGFromDocument: entity_create_or_update executing MERGE query: {query}")
            result = execute_neo4j_query(query, parameters)
            log(DEBUG, f"KGFromDocument: entity_create_or_update MERGE result: {result}")
            if not result:
                log(ERROR, f"KGFromDocument: entity_create_or_update MERGE entity '{entity_id}' returned empty result. Query: {query}, Parameters: {parameters}")
            success = success and bool(result)

        except Exception as e:
            log(ERROR, f"KGFromDocument: entity_create_or_update MERGE entity '{entity_id}' threw exception: {e}. Query: {query}, Parameters: {parameters}. Traceback: {traceback.format_exc()}")
            success = False

        try:
            query = "MATCH (c:Community {id: $community_id}) WITH c MATCH (e:Entity {id: $entity_id}) MERGE (c)-[:AGGREGATES]->(e) RETURN e"
            parameters = {"community_id": community_id, "entity_id": entity_id}
            log(DEBUG, f"KGFromDocument: entity_create_or_update executing AGGREGATES query for entity '{entity_id}' in community '{community_id}'")
            result = execute_neo4j_query(query, parameters)
            log(DEBUG, f"KGFromDocument: entity_create_or_update AGGREGATES result: {result}")
            if not result:
                log(ERROR, f"KGFromDocument: entity_create_or_update AGGREGATES relationship '{entity_id}' to community '{community_id}' returned empty result. Query: {query}, Parameters: {parameters}")
            success = success and bool(result)
            
        except Exception as e:
            log(ERROR, f"KGFromDocument: entity_create_or_update AGGREGATES relationship '{entity_id}' threw exception: {e}. Traceback: {traceback.format_exc()}")
            success = False

        if document_or_chapter_id is not None:
            # :SOURCE always targets the root Document, never a Chapter, regardless of the granularity passed in.
            root_document_id = DocumentsInterface.get_document_id_from_fragment_id(document_or_chapter_id)

            try:
                query = """
                    MATCH (d:Document {id:$docid})
                    WITH d
                    MATCH (e:Entity {id: $entity_id})
                    MERGE (e)-[:SOURCE]->(d)
                    RETURN e
                """
                parameters = {"entity_id": entity_id, "docid": root_document_id}
                log(DEBUG, f"KGFromDocument: entity_create_or_update executing SOURCE query for entity '{entity_id}' to Document '{root_document_id}'")
                result = execute_neo4j_query(query, parameters)
                log(DEBUG, f"KGFromDocument: entity_create_or_update SOURCE result: {result}")
                if not result:
                    log(ERROR, f"KGFromDocument: entity_create_or_update SOURCE relationship '{entity_id}' to Document '{root_document_id}' returned empty result. Query: {query}, Parameters: {parameters}")
                success = success and bool(result)

            except Exception as e:
                log(ERROR, f"KGFromDocument: entity_create_or_update SOURCE relationship '{entity_id}' threw exception: {e}. Traceback: {traceback.format_exc()}")
                success = False

    return success

def entity_merge(entity_id: str, second_entity_id: str) -> bool:
    """
    Merges two entities.
    First we walk through all attributes of the second entity and merge them with the attributes of the first entity.
    Then we delete the second entity.
    """
    success = True

    # fetch the attributes of the first entity
    query = """MATCH (e1:Entity {id: $entity_id1}) RETURN e1 AS entity"""
    parameters = {"entity_id1": entity_id}
    result = execute_neo4j_query(query, parameters)
    original_attributes = []
    if result and len(result) > 0:
        entity = result[0]["entity"]
        original_attributes = dict(entity.items())

    # fetch the attributes of the second entity
    query = """MATCH (e2:Entity {id: $entity_id2}) RETURN e2 AS entity"""
    parameters = {"entity_id2": second_entity_id}
    result = execute_neo4j_query(query, parameters)
    second_attributes = {}
    if result and len(result) > 0:
        entity = result[0]["entity"]
        second_attributes = dict(entity.items())

    # merge the attributes of the second entity with the attributes of the first entity
    merged_attributes, requires_consolidation = merge_attributes(original_attributes, second_attributes, source=None)

    if len(merged_attributes.keys()) > 0:
        # update the first entity with the merged attributes
        set_clause = ", ".join([f"e.`{key}` = $`{key}`" for key in merged_attributes.keys()])
        if requires_consolidation:
            set_clause += f", e.{ATTRIBUTE_REQUIRES_CONSOLIDATION} = true"
        query = f"""MATCH (e:Entity {{id: $entity_id1}}) SET {set_clause}"""
        parameters = {"entity_id1": entity_id, **merged_attributes}
        result = execute_neo4j_query(query, parameters)
        success = success and bool(result)

    # attach all relationships of the second entity to the first entity
    query = """MATCH (e1:Entity {id: $entity_id1}), (e2:Entity {id: $entity_id2})
            MATCH (e2)-[r]->(e3:Entity)
            CALL apoc.create.relationship(e1, type(r), properties(r), e3) YIELD rel
            DELETE r"""
    parameters = {"entity_id1": entity_id, "entity_id2": second_entity_id}
    result = execute_neo4j_query(query, parameters)

    # identify possibly duplicate relationships, and merge them
    merge_relationships(entity_id, second_entity_id)

    # delete the second entity
    query = """MATCH (e:Entity {id: $entity_id2}) DETACH DELETE e"""
    parameters = {"entity_id2": second_entity_id}
    result = execute_neo4j_query(query, parameters)
    success = success and bool(result)

    return success

def entity_delete(entity_id:str) -> bool:
    """
    Deletes an entity, by entity_id and label.
    If not found, returns True (idempotent — nothing to delete).
    """
    check_query = """MATCH (e:Entity {id: $entity_id}) RETURN e.id AS id"""
    check_result = execute_neo4j_query(check_query, {"entity_id": entity_id})
    if not check_result:
        log(INFO, f"INFO: Entity with id '{entity_id}' was not found, therefore there's no need to delete it. The result is successful.")
        return True
    
    query = """MATCH (e:Entity {id: $entity_id}) DETACH DELETE e"""
    execute_neo4j_query(query, {"entity_id": entity_id})
    return True

def _is_supported_link_kind(kind: str) -> bool:
    """True when `kind` is one of the two accepted IS_DOCUMENTED kinds.

    Both writers call this themselves rather than trusting their caller: the skill dispatcher already
    rejects anything else, but these functions are module-level and callable from anywhere, and an
    off-label kind is exactly the kind of damage the maintenance sweep exists to repair. Fixing it at
    the writer keeps the sweep free of self-inflicted work.
    """
    return kind in IS_DOCUMENTED_KINDS


def entity_document_link_create(entity_id: str, document_id: str, kind: str, created_by: str) -> bool:
    """Link an existing entity to an existing document with an IS_DOCUMENTED edge of the given kind.

    Both nodes must already exist, so this path can never mint a document from the knowledge-graph
    side. Re-running with the same kind changes nothing, which keeps the writer idempotent and
    single-writer. Audit properties stay scalar: string values in this graph get re-scanned for
    embedded reference URIs.
    """
    if not _is_supported_link_kind(kind):
        log(ERROR, f"KnowledgeGraphEditor: refusing to link entity '{entity_id}' to document '{document_id}': unsupported kind '{kind}', expected one of {', '.join(IS_DOCUMENTED_KINDS)}.")
        return False

    success: bool = False

    query = f"""
        MATCH (e:Entity {{id: $entity_id}})
        MATCH (d:Document {{id: $document_id}})
        MERGE (e)-[r:{IS_DOCUMENTED_RELATIONSHIP_TYPE} {{kind: $kind}}]->(d)
        ON CREATE SET r.created_by = $created_by, r.created_at = $created_at
        RETURN r
    """
    parameters = {
        "entity_id": entity_id,
        "document_id": document_id,
        "kind": kind,
        "created_by": created_by,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        result = execute_neo4j_query(query, parameters)
        if result:
            success = True
        else:
            # An empty result means a MATCH matched nothing: the entity or the document is not there.
            log(WARNING, f"KnowledgeGraphEditor: could not link entity '{entity_id}' to document '{document_id}': one of the two nodes does not exist.")

    except Exception as e:
        log(ERROR, f"KnowledgeGraphEditor: exception while linking entity '{entity_id}' to document '{document_id}': {e}")

    return success

def entity_document_link_delete(entity_id: str, document_id: str, kind: str) -> bool:
    """Remove the IS_DOCUMENTED edge of one explicit kind between an entity and a document.

    The kind filter is mandatory by design: an automated pass may clear its own 'extracted' edges
    without ever touching curated ones. Idempotent — finding no matching edge is still a success.
    """
    if not _is_supported_link_kind(kind):
        log(ERROR, f"KnowledgeGraphEditor: refusing to unlink entity '{entity_id}' from document '{document_id}': unsupported kind '{kind}'. A kind-less or misspelled delete could clear curated edges, so nothing was removed.")
        return False

    success: bool = False

    query = f"""
        MATCH (e:Entity {{id: $entity_id}})-[r:{IS_DOCUMENTED_RELATIONSHIP_TYPE} {{kind: $kind}}]->(d:Document {{id: $document_id}})
        WITH r
        DELETE r
        RETURN count(*) AS removed_count
    """
    parameters = {"entity_id": entity_id, "document_id": document_id, "kind": kind}

    try:
        result = execute_neo4j_query(query, parameters)
        removed_count = result[0].get("removed_count", 0) if result else 0
        log(INFO, f"KnowledgeGraphEditor: removed {removed_count} IS_DOCUMENTED '{kind}' edge(s) from entity '{entity_id}' to document '{document_id}'.")
        success = True

    except Exception as e:
        log(ERROR, f"KnowledgeGraphEditor: exception while unlinking entity '{entity_id}' from document '{document_id}': {e}")

    return success

def verify_relationship_legitimity(ontology_id:str, entity1_id: str, entity2_id: str, label: str, allow_any: bool=False) -> bool:
    success = True
    validate_ontology_label(label)
    entity1_id = entity1_id.lower().strip()
    entity2_id = entity2_id.lower().strip()

    if ontology_id is None or len(ontology_id) == 0:
        # the ontology_id is mandatory, we need to find it from the entities. Let's use the first entity
        query = """MATCH (e:Entity {id: $entity_id})<-[:AGGREGATES]-(c:Community)<-[:DEFINES]-(o:Ontology) RETURN o.id AS ontology_id"""
        parameters = {"entity_id": entity1_id}
        result = execute_neo4j_query(query, parameters)
        if result is None or len(result) == 0:
            if allow_any:
                log(WARNING, f"WARNING: The entity '{entity1_id}' is not found in the knowledge graph, therefore we cannot find the ontology. However, since 'allow_any' is True, we will allow the relationship anyway.")
                return True
            else:
                log(WARNING, f"WARNING: The relationship '{label}' is not defined in the ontology '{ontology_id}'.")
                return False
        ontology_id = result[0]["ontology_id"]

    # first we must check if the relationship has restrictions, or allows anything
    query = f"""MATCH (o:Ontology {{id:$ontology_id}})-[:CONTAINS]->(r:RelationshipOntology:{label}) RETURN r"""
    parameters = {"ontology_id": ontology_id}
    result = execute_neo4j_query(query, parameters)
    if result is None or len(result) == 0:
        log(WARNING, f"WARNING: The relationship '{label}' is not defined in the ontology '{ontology_id}'.")
        if allow_any:
            log(WARNING, f"WARNING: Since 'allow_any' is True, we will allow the relationship anyway.")
            return True
        else:
            return False
    
    # the entity_source_labels is a comma-separated list of labels, that are allowed for the source entity, or None if any label is allowed, or two dashes '--' if any entity is allowed but it must be the same as the target entity
    # the entity_target_labels is similarly regulated
    # Examples:
    #   entity_source_labels: 'Person,Animal'           (that is, the source entity must be labeled 'Person' or 'Animal')
    #   entity_target_labels: 'Person,Animal,Object'    (that is, the target entity must be labeled 'Person', 'Animal' or 'Object')
    #
    #   label: 'is_child_of'
    #   entity_source_labels: 'Person,Animal'           (that is, only Person and Animal can be children of another entity)
    #   entity_target_labels: '--'                      (that is, the target entity must be the same as the source entity, because Person is child of Person and Animal is child of Animal)
    relationship = result[0]["r"]
    entity_source_labels = relationship["entity_source_labels"]
    entity_target_labels = relationship["entity_target_labels"]
    entity_source_labels = entity_source_labels.split(",") if entity_source_labels is not None else None
    entity_target_labels = entity_target_labels.split(",") if entity_target_labels is not None else None
    entity_source_labels = [lbl.lower().strip() for lbl in entity_source_labels] if entity_source_labels is not None else None
    entity_target_labels = [lbl.lower().strip() for lbl in entity_target_labels] if entity_target_labels is not None else None

    # get information about the entities
    query = """
                MATCH (e1:Entity), (e2:Entity)
                MATCH (e1:Entity {id: $entity1_id})
                MATCH (e2:Entity {id: $entity2_id})
                RETURN e1, e2
            """
    parameters = {"entity1_id": entity1_id, "entity2_id": entity2_id}
    result = execute_neo4j_query(query, parameters)
    if len(result) == 0:
        log(WARNING, f"WARNING: The entities '{entity1_id}' and '{entity2_id}' are not found in the knowledge graph.")
        return False
    e1_labels = list(result[0]['e1'].labels)
    e1_labels = [label for label in e1_labels if label not in RESERVED_ONTOLOGY_LABELS]
    e2_labels = list(result[0]['e2'].labels)
    e2_labels = [label for label in e2_labels if label not in RESERVED_ONTOLOGY_LABELS]

    # Handle entities without domain-specific labels
    if not e1_labels:
        log(WARNING, f"WARNING: Entity '{entity1_id}' has no domain-specific labels (only reserved labels). Full labels: {list(result[0]['e1'].labels)}")
        return False
    if not e2_labels:
        log(WARNING, f"WARNING: Entity '{entity2_id}' has no domain-specific labels (only reserved labels). Full labels: {list(result[0]['e2'].labels)}")
        return False

    # Lowercase all labels for comparison against the (already lowercased) allowed lists.
    e1_labels_lower = [l.lower().strip() for l in e1_labels]
    e2_labels_lower = [l.lower().strip() for l in e2_labels]

    # Keep first label as representative for same-type check (-- sentinel).
    e1_label = e1_labels_lower[0]
    e2_label = e2_labels_lower[0]

    if (entity_source_labels and entity_source_labels[0] == '--') or (entity_target_labels and entity_target_labels[0] == '--'):
        # the source and target entities must be the same type, we check if they have the same label. If not, the relationship is not allowed.
        if e1_label != e2_label:
            log(WARNING, f"WARNING: The relationship '{label}' requires that the source and target entities must be the same, but the source entity '{entity1_id}' is labeled '{e1_label}' and the target entity '{entity2_id}' is labeled '{e2_label}'.")
            success = False

    if entity_source_labels is not None and entity_source_labels[0] != "--" and not any(l in entity_source_labels for l in e1_labels_lower):
        log(WARNING, f"WARNING: The relationship '{label}' is not allowed between the two entities '{entity1_id}' and '{entity2_id}'.")
        success = False
    if entity_target_labels is not None and entity_target_labels[0] != "--" and not any(l in entity_target_labels for l in e2_labels_lower):
        log(WARNING, f"WARNING: The relationship '{label}' is not allowed between the two entities '{entity1_id}' and '{entity2_id}'.")
        success = False

    return success

def relationship_create_or_update(ontology_id:str, entity1_id: str, entity2_id: str, label: str, attributes: Dict[str, str], document_or_chapter_id:str, paragraph_id:str) -> bool:
    """
    Creates a new relationship or updates an existing relationship.
    This is done only if the relationship is allowed between the two entities, based on their labels and the entity_source_labels / entity_target_labels of the relationship.
    For example:
        We want to create the relationship "likes" between the entities "john" and "mary", taken from the sentence "John likes Mary".
        We know that 'john' is an entity labeled 'Person', and same for 'mary'.
        The relationship 'likes' has the following restrictions:
        - entity_source_labels: ['Person,Animal']           (that is, the source entity must be labeled 'Person' or 'Animal')
        - entity_target_labels: ['Person,Animal,Object']    (that is, the target entity must be labeled 'Person', 'Animal' or 'Object')
        Which means, in general, that Persons or Animals may like other Persons, Animals or Objects.
        The request to create the relationship "likes" between 'john' and 'mary' will be successful.
        The request to create the relationship "likes" between 'john' and 'car' will be successful too, because 'car' is labeled 'Object'.
        But a request to create the relationship "likes" between 'car' and 'john' will fail, because a 'car' (Object) is not allowed for the relationship source label. It also makes sense, Objects cannot 'like' anything.
    """
    validate_ontology_label(label)
    success = verify_relationship_legitimity(ontology_id, entity1_id, entity2_id, label)
    if success is False:
        # the relationship is not allowed between the two entities, see example above "the car likes John"
        log(WARNING, f"relationship_create_or_update: relationship '{label}' not allowed between '{entity1_id}' and '{entity2_id}'")
        return False

    # Verify both entities exist before attempting to create the relationship
    check_query = """MATCH (e1:Entity {id: $entity1_id}) MATCH (e2:Entity {id: $entity2_id}) RETURN e1.id AS e1_id, e2.id AS e2_id"""
    check_result = execute_neo4j_query(check_query, {"entity1_id": entity1_id, "entity2_id": entity2_id})
    if check_result is None or len(check_result) == 0:
        # At least one entity doesn't exist - find out which one(s)
        e1_result = execute_neo4j_query("MATCH (e:Entity {id: $eid}) RETURN e.id", {"eid": entity1_id})
        e2_result = execute_neo4j_query("MATCH (e:Entity {id: $eid}) RETURN e.id", {"eid": entity2_id})
        missing = []
        if e1_result is None or len(e1_result) == 0:
            missing.append(f"'{entity1_id}'")
        if e2_result is None or len(e2_result) == 0:
            missing.append(f"'{entity2_id}'")
        log(ERROR, f"relationship_create_or_update: Cannot create relationship '{label}': entity/entities {', '.join(missing)} not found in the knowledge graph.")
        return False

    source_id = build_doc_source_id(document_or_chapter_id, paragraph_id)
    merged_attributes = None

    if attributes is not None and len(attributes) > 0:
        # merge existing values in the attributes, with the new values. First,
        # we need to fetch the existing values; then, we merge them with the new values.
        query = f"""MATCH (e1:Entity {{id: $entity1_id}})-[r:{label}]->(e2:Entity {{id: $entity2_id}}) RETURN r AS relationship"""
        parameters = {"entity1_id": entity1_id, "entity2_id": entity2_id}
        result = execute_neo4j_query(query, parameters)
        original_attributes = []
        if result and len(result) > 0:
            relationship = result[0]["relationship"]
            original_attributes = dict(relationship.items())
        merged_attributes, requires_consolidation = merge_attributes(original_attributes, attributes, source_id)
        
        set_clause = ", ".join([f"r.`{key}` = $`{key}`" for key in merged_attributes.keys()])
        if set_clause != "":
            set_clause += ", "
        if requires_consolidation:
            set_clause += f"r.{ATTRIBUTE_REQUIRES_CONSOLIDATION} = true, "
        query = f"""MATCH (e1:Entity {{id: $entity1_id}})
            WITH e1
            MATCH (e2:Entity {{id: $entity2_id}})
            MERGE (e1)-[r:{label}]->(e2)
            SET {set_clause}r.source_id = $source_id
            RETURN r"""
        parameters = {"entity1_id": entity1_id, "entity2_id": entity2_id, **merged_attributes, "source_id": source_id}
    else:
        query = f"""MATCH (e1:Entity {{id: $entity1_id}})
                    WITH e1
                    MATCH (e2:Entity {{id: $entity2_id}})
                    MERGE (e1)-[r:{label}]->(e2)
                    SET r.source_id = $source_id
                    RETURN r"""
        parameters = {"entity1_id": entity1_id, "entity2_id": entity2_id, "source_id": source_id}
    
    result = execute_neo4j_query(query, parameters)
    success = success and bool(result)
    
    return success

def relationship_delete(entity_id1: str, entity_id2: str, label: str) -> bool:
    """
    Deletes a relationship.
    """
    validate_ontology_label(label)
    query = f"""MATCH (e1:Entity {{id: $entity_id1}})-[r:{label}]-(e2:Entity {{id: $entity_id2}}) DELETE r"""
    parameters = {"entity_id1": entity_id1, "entity_id2": entity_id2}
    result = execute_neo4j_query(query, parameters)
    return bool(result)

def _sanitize_key(key: str) -> str:
    """Sanitize an attribute key so it is a valid Neo4j property name and Cypher parameter.
    Replaces whitespace, tabs and other special characters with underscores.
    Unlike tokenize_entity_id this does NOT lowercase the key.
    """
    result = key.replace(" ", "_").replace("\t", "_").replace(".", "_").replace(",", "_").replace(";", "_").replace(":", "_").replace("!", "_").replace("?", "_").replace("(", "_").replace(")", "_").replace("[", "_").replace("]", "_").replace("{", "_").replace("}", "_").replace("<", "_").replace(">", "_").replace("/", "_").replace("\\", "_").replace("|", "_").replace("'", "_").replace('"', "_").replace("`", "_").replace("~", "_").replace("@", "_").replace("#", "_").replace("$", "_").replace("%", "_").replace("^", "_").replace("&", "_").replace("*", "_").replace("+", "_").replace("=", "_")
    # collapse multiple consecutive underscores
    while "__" in result:
        result = result.replace("__", "_")
    # strip leading/trailing underscores
    result = result.strip("_")
    return result

def sanitize_attribute_keys(attributes: Dict[str, str]) -> Dict[str, str]:
    """Return a new dict with all attribute keys sanitized for Neo4j compatibility.
    Keys containing spaces or special characters are converted to valid identifiers.
    """
    if attributes is None:
        return attributes
    return {_sanitize_key(k): v for k, v in attributes.items()}

def _get_ontology_id_for_community(community_id: str) -> str:
    """Get the ontology_id for a given community. Returns None if not found."""
    if community_id is None:
        return None
    query = """MATCH (c:Community {id: $community_id})<-[:DEFINES]-(o:Ontology) RETURN o.id as ontology_id"""
    parameters = {"community_id": community_id}
    result = execute_neo4j_query(query, parameters)
    if result and len(result) > 0:
        return result[0]["ontology_id"]
    return None

def _get_allowed_ontology_attributes(ontology_id: str, label: str, is_relationship: bool = False) -> set:
    """Get the set of allowed attribute names from the ontology definition.
    Returns None if no ontology is defined, the label is not found, or the lookup fails.
    Returns a set of attribute names if found (may be empty if no custom attributes are defined).
    """
    if ontology_id is None or label is None:
        return None
    node_type = "RelationshipOntology" if is_relationship else "EntityOntology"
    query = f"""
        MATCH (o:Ontology {{id: $ontology_id}})-[:CONTAINS]->(n:{node_type}:{label})
        OPTIONAL MATCH (n)-[:HAS_ATTRIBUTE]->(a:OntologyAttribute)
        RETURN n, collect(DISTINCT a.name) AS attributeNames
    """
    parameters = {"ontology_id": ontology_id}
    result = execute_neo4j_query(query, parameters)
    if result is None or len(result) == 0:
        return None  # label not defined in ontology
    allowed = {name for name in (result[0].get("attributeNames") or []) if name and name not in RESERVED_ONTOLOGY_ATTRIBUTES}
    return allowed

def filter_attributes_by_ontology(attributes: Dict[str, str], ontology_id: str, label: str, is_relationship: bool = False) -> Dict[str, str]:
    """Filter attributes to only include those defined in the ontology.
    Logs a warning for each rejected attribute.
    If no ontology is defined or label not found in ontology, returns all attributes unchanged.
    """
    if attributes is None or len(attributes) == 0:
        return attributes
    allowed = _get_allowed_ontology_attributes(ontology_id, label, is_relationship)
    if allowed is None:
        return attributes  # no ontology constraint, pass all through
    filtered = {}
    rejected = []
    for key, value in attributes.items():
        if key in allowed:
            filtered[key] = value
        else:
            rejected.append(key)
    if rejected:
        entity_or_rel = "relationship" if is_relationship else "entity"
        log(WARNING, f"Ontology '{ontology_id}': attribute(s) {rejected} are not defined for {entity_or_rel} label '{label}' and were not added.")
    return filtered

def tokenize_entity_id(entity_id: str) -> str:
    result = entity_id.lower()

    # convert as well whitespaces, tabs, punctuation into undescored ("_")
    result = result.replace(" ", "_").replace("\t", "_").replace(".", "_").replace(",", "_").replace(";", "_").replace(":", "_").replace("!", "_").replace("?", "_").replace("(", "_").replace(")", "_").replace("[", "_").replace("]", "_").replace("{", "_").replace("}", "_").replace("<", "_").replace(">", "_").replace("/", "_").replace("\\", "_").replace("|", "_").replace("'", "_").replace('"', "_").replace("`", "_").replace("~", "_").replace("@", "_").replace("#", "_").replace("$", "_").replace("%", "_").replace("^", "_").replace("&", "_").replace("*", "_").replace("+", "_").replace("=", "_").replace(" ", "_")

    return result


def build_resource_lock(resource_type: str, *identifiers):
    """
    Build a consistent lock key and return the resource lock from get_lock.

    For resource_type == "RELATIONSHIP" the expected identifiers are:
        (label, entity_id_1, entity_id_2)
    The function will sort the two entity ids alphabetically to make the lock
    symmetric (so the order of start/end doesn't create different locks).

    For other resource types the identifiers are joined (or the single identifier
    is used) to create the lock key.
    """
    if resource_type == "RELATIONSHIP":
        # Expect label, entity1, entity2. Fall back to first identifier if not provided.
        if len(identifiers) >= 3:
            label, id1, id2 = identifiers[0], identifiers[1] or "", identifiers[2] or ""
            ordered = sorted([id1, id2])
            key = f"{label}__{ordered[0]}_{ordered[1]}"
        else:
            key = identifiers[0] if len(identifiers) > 0 else ""
    else:
        if len(identifiers) == 0:
            key = ""
        elif len(identifiers) == 1:
            key = identifiers[0]
        else:
            key = "_".join([str(i) for i in identifiers if i is not None and i != ""])

    return get_lock(resource_type, key)


def KnowledgeGraphEditor(
        operation: Annotated[str, "One of the following keywords: COMMUNITY_CREATE, ENTITY_CREATE_OR_UPDATE, ENTITY_MERGE, ENTITY_DELETE, RELATIONSHIP_CREATE_OR_UPDATE, RELATIONSHIP_DELETE, DOCUMENT_LINK_CREATE, DOCUMENT_LINK_DELETE."],
        community_id: Annotated[str, "The identifier of a community, that is of an entity group."]=COMMUNITY_ID_DEFAULT,
        community_name: Annotated[str, "The name of a community (mandatory for operation COMMUNITY_CREATE)."]=None,
        community_description: Annotated[str, "A brief description of a community (mandatory for operations COMMUNITY_CREATE)."]=None,
        ontology_id: Annotated[str, "The identifier of an ontology to be associated to the community to be created (mandatory in for operations COMMUNITY_CREATE and RELATIONSHIP_CREATE_OR_UPDATE)"]=None,
        entity_id: Annotated[str, "The identifier of an entity. Format: alphanumeric, no white spaces, camel case (mandatory for all operations ENTITY_CREATE_OR_UPDATE, ENTITY_MERGE, ENTITY_DELETE). For sentences like 'John likes Mary' for the relationship 'like' (assuming existing entities 'john' and 'mary'), you must use entity_id='john' and 'second_entity_id'='mary' because the relationships are mono-directional."]=None,
        second_entity_id_to_merge: Annotated[str, "The identifier of a second entity to merge with the first entity (mandatory for ENTITY_MERGE). If the same value for entity_id is given for second_entity_id_to_merge, an error is returned."]=None,
        label: Annotated[str, "The neo4j label of the entity or relationship. It must be a single word, no extra characters or white spaces except for the underscore (_). Mandatory for operations ENTITY_CREATE_OR_UPDATE, RELATIONSHIP_CREATE_OR_UPDATE, ENTITY_DELETE."]=None,
        attributes: Annotated[str, "A dictionary [str,str] of attribute name/value to update the entity. Do not use these reserved attributes: entity_description, id, label, embedding, entity_source_labels, entity_target_labels. Do not add attribute values that are already in the knowledge graph."]=None,
        relationship_start_entity_id: Annotated[str, "The identifier of the start entity of a relationship (mandatory for operation RELATIONSHIP_CREATE_OR_UPDATE)."]=None,
        relationship_end_entity_id: Annotated[str, "The identifier of the end entity of a relationship (mandatory for operation RELATIONSHIP_CREATE_OR_UPDATE)."]=None,
        document_or_chapter_id: Annotated[str, "The identifier of a document or chapter where the information has been sourced from (mandatory for operations ENTITY_CREATE_OR_UPDATE and RELATIONSHIP_CREATE_OR_UPDATE)."]=None,
        paragraph_id: Annotated[str, "The identifier of a paragraph where the information has been sourced from (mandatory for operations ENTITY_CREATE_OR_UPDATE and RELATIONSHIP_CREATE_OR_UPDATE). Example: from the html document <p id='p-17'> the value is: 'p-7'"]=None,
        linked_document_id: Annotated[str, "The identifier of an existing document to link the entity to, or to unlink the entity from (mandatory for operations DOCUMENT_LINK_CREATE and DOCUMENT_LINK_DELETE). The document must already exist."]=None,
        link_kind: Annotated[str, "Who decided the link exists: 'curated' when a person or an agent asserted it, 'extracted' when a program derived it. Mandatory for DOCUMENT_LINK_CREATE and DOCUMENT_LINK_DELETE; only edges of the given kind are ever created or deleted."]=None,
        *,
        effective_user_id,  # framework-injected trusted actor identity — deliberately unannotated so it is never exposed to the LLM tool schema
        goal=None,
        plan=None
        ) -> Annotated[Dict[str, Union[bool, str, list]], "A dictionary containing the results or an error message."]:
    """
    Allows operations of the Knowledge Graph entities, relationships, and entity-to-document links. The following operations are supported:
    - Create a new Community
    - Create a new entity
    - Update an existing entity
    - Merge two entities
    - Delete an entity
    - Create a new relationship
    - Update an existing relationship
    - Delete a relationship
    - Link an entity to an existing document
    - Unlink an entity from a document

    The knowledge graph is implemented in Neo4j, the entities are implemented as neo4j nodes, the relationships as neo4j relationships.
    """
    arguments = {"operation": operation, "community_id": community_id, "community_name": community_name, "community_description": community_description, "entity_id": entity_id, "second_entity_id_to_merge": second_entity_id_to_merge, "label": label, "attributes": attributes, "relationship_start_entity_id": relationship_start_entity_id, "relationship_end_entity_id": relationship_end_entity_id, "document_or_chapter_id": document_or_chapter_id, "paragraph_id": paragraph_id, "linked_document_id": linked_document_id, "link_kind": link_kind}
    result = {"success": True, "output": None}
    error_message = None
    is_warning = False
    operation_successful = True
    objects = None


    try:
        if not effective_user_id:
            error_message = "effective_user_id is required and must be an explicit user or system identity."
            raise ValueError(error_message)

        # adjust input parameters
        if isinstance(attributes, str):
            try:
                attributes = json.loads(attributes)
            except:
                # convert the attributes from a string to a dictionary
                # the original parameter may be a string with single quotes: "{'name':'Martha', 'description':'A meticulous homemaker with a "soft" spot for John\'s daughter Jane and her furry friend.'}",
                # which is not compatible with json.loads. We need to replace the single quotes with double quotes. However, we need to be careful with the single and double quotes inside the string.
                attributes_converted = prepare_string_for_json(attributes)
                attributes = json.loads(attributes_converted)
        if entity_id is not None:
            # convert the entity_id always to lowercase. We don't want LLM hallucinations to produce different entity IDs.
            entity_id = tokenize_entity_id(entity_id)

        if second_entity_id_to_merge is not None:
            # convert the second_entity_id_to_merge always to lowercase. We don't want LLM hallucinations to produce different entity IDs.
            second_entity_id_to_merge = tokenize_entity_id(second_entity_id_to_merge)

        if relationship_start_entity_id is not None:
            # convert the relationship_start_entity_id always to lowercase. We don't want LLM hallucinations to produce different entity IDs.
            relationship_start_entity_id = tokenize_entity_id(relationship_start_entity_id)

        if relationship_end_entity_id is not None:
            # convert the relationship_end_entity_id always to lowercase. We don't want LLM hallucinations to produce different entity IDs.
            relationship_end_entity_id = tokenize_entity_id(relationship_end_entity_id)

        if label is not None:
            # convert the label always to lowercase. We don't want LLM hallucinations to produce different labels.
            # also, convert spaces to "_" and remove any other special characters
            label = tokenize_entity_id(label)

        # sanitize attribute keys: replace spaces and special characters with underscores
        # this prevents Neo4j Cypher syntax errors like "e.personality traits = $personality traits"
        if isinstance(attributes, dict):
            attributes = sanitize_attribute_keys(attributes)

        # execute the operations
        if operation == "COMMUNITY_CREATE":
            if not community_name or not community_description:
                error_message = "Community name and description are required."
                raise ValueError(error_message)
            if not ontology_id:
                error_message = "Ontology ID is required."
                raise ValueError(error_message)
            if not _check_ontology_permission(ontology_id, effective_user_id, "ADD"):
                error_message = "Access denied for this knowledge graph operation."
                raise PermissionError(error_message)
            resource_lock = build_resource_lock("COMMUNITY", community_id)
            with resource_lock:
                operation_successful = community_create(community_id, community_name, community_description, ontology_id)
            pass

        elif operation == "ENTITY_CREATE_OR_UPDATE":
            if community_id is None:
                community_id = COMMUNITY_ID_DEFAULT
            if not entity_id or not community_id:
                error_message = "Entity ID, Community ID, Entity name, and Entity description are required."
                raise ValueError(error_message)
            entity_ontology_id = _get_ontology_id_for_community(community_id)
            entity_exists = bool(execute_neo4j_query(
                "MATCH (e:Entity {id: $entity_id}) RETURN e.id AS id",
                {"entity_id": entity_id}
            ))
            required_permission = "UPDATE" if entity_exists else "ADD"
            if not _check_ontology_permission(entity_ontology_id, effective_user_id, required_permission):
                error_message = "Access denied for this knowledge graph operation."
                raise PermissionError(error_message)
            # Filter attributes against the ontology definition (if an ontology is defined)
            if isinstance(attributes, dict) and len(attributes) > 0:
                effective_ontology_id = ontology_id or _get_ontology_id_for_community(community_id)
                if effective_ontology_id is not None:
                    attributes = filter_attributes_by_ontology(attributes, effective_ontology_id, label, is_relationship=False)
            resource_lock = build_resource_lock("ENTITY", entity_id)
            with resource_lock:
                operation_successful = entity_create_or_update(entity_id, community_id=community_id, label=label, attributes=attributes, document_or_chapter_id=document_or_chapter_id, paragraph_id=paragraph_id)
            pass

        elif operation == "ENTITY_MERGE":
            if not entity_id or not second_entity_id_to_merge:
                error_message = "Entity ID and Second Entity ID are required."
                raise ValueError(error_message)
            if entity_id == second_entity_id_to_merge:
                is_warning = True
                error_message = "Entity ID and Second Entity ID must be different."
                raise ValueError(error_message)
            if not _check_entity_permissions([entity_id, second_entity_id_to_merge], effective_user_id, "UPDATE"):
                error_message = "Access denied for this knowledge graph operation."
                raise PermissionError(error_message)
            resource_lock = build_resource_lock("ENTITY", entity_id)
            with resource_lock:
                operation_successful = entity_merge(entity_id, second_entity_id_to_merge)
            pass

        elif operation == "ENTITY_DELETE":
            if not entity_id:
                error_message = "Entity ID is required."
                raise ValueError(error_message)
            if not _check_entity_permissions([entity_id], effective_user_id, "DELETE"):
                error_message = "Access denied for this knowledge graph operation."
                raise PermissionError(error_message)
            resource_lock = build_resource_lock("ENTITY", entity_id)
            with resource_lock:
                operation_successful = entity_delete(entity_id)
            pass

        elif operation == "RELATIONSHIP_CREATE_OR_UPDATE":
            if not relationship_start_entity_id or not relationship_end_entity_id or not label:
                error_message = "Start Entity ID, End Entity ID and Label are required."
                raise ValueError(error_message)
            if not label or len(label) == 0:
                error_message = "Label is required."
                raise ValueError(error_message)
            relationship_entity_ids = [relationship_start_entity_id, relationship_end_entity_id]
            relationship_ontology_ids = _get_entity_ontology_ids(relationship_entity_ids)
            relationship_exists = bool(execute_neo4j_query(
                f"MATCH (e1:Entity {{id: $entity1_id}})-[:{label}]->(e2:Entity {{id: $entity2_id}}) RETURN e1.id AS id",
                {"entity1_id": relationship_start_entity_id, "entity2_id": relationship_end_entity_id}
            ))
            required_permission = "UPDATE" if relationship_exists else "ADD"
            if ontology_id and ontology_id not in relationship_ontology_ids:
                relationship_ontology_ids.append(ontology_id)
            if not relationship_ontology_ids or not all(
                _check_ontology_permission(parent_ontology_id, effective_user_id, required_permission)
                for parent_ontology_id in relationship_ontology_ids
            ):
                error_message = "Access denied for this knowledge graph operation."
                raise PermissionError(error_message)
            # Filter attributes against the ontology definition (if an ontology is defined)
            if isinstance(attributes, dict) and len(attributes) > 0:
                effective_ontology_id = ontology_id
                if effective_ontology_id is None:
                    # Try to resolve ontology_id from the start entity's community
                    ent_query = """MATCH (e:Entity {id: $eid})<-[:AGGREGATES]-(c:Community)<-[:DEFINES]-(o:Ontology) RETURN o.id AS ontology_id"""
                    ent_result = execute_neo4j_query(ent_query, {"eid": relationship_start_entity_id})
                    if ent_result and len(ent_result) > 0:
                        effective_ontology_id = ent_result[0]["ontology_id"]
                if effective_ontology_id is not None:
                    attributes = filter_attributes_by_ontology(attributes, effective_ontology_id, label, is_relationship=True)
            resource_lock = build_resource_lock("RELATIONSHIP", label, relationship_start_entity_id, relationship_end_entity_id)
            with resource_lock:
                operation_successful = relationship_create_or_update(ontology_id, relationship_start_entity_id, relationship_end_entity_id, label=label, attributes=attributes, document_or_chapter_id=document_or_chapter_id, paragraph_id=paragraph_id)
            pass

        elif operation == "RELATIONSHIP_DELETE":
            if not relationship_start_entity_id or not relationship_end_entity_id or not label:
                error_message = "Start Entity ID, End Entity ID, and Relationship name are required."
                raise ValueError(error_message)
            if not _check_entity_permissions(
                [relationship_start_entity_id, relationship_end_entity_id],
                effective_user_id,
                "DELETE"
            ):
                error_message = "Access denied for this knowledge graph operation."
                raise PermissionError(error_message)
            resource_lock = build_resource_lock("RELATIONSHIP", label, relationship_start_entity_id, relationship_end_entity_id)
            with resource_lock:
                operation_successful = relationship_delete(relationship_start_entity_id, relationship_end_entity_id, label=label)
            pass

        elif operation == "DOCUMENT_LINK_CREATE":
            if not entity_id or not linked_document_id:
                error_message = "Entity ID and Linked Document ID are required."
                raise ValueError(error_message)
            if link_kind not in IS_DOCUMENTED_KINDS:
                error_message = f"Link kind must be one of: {', '.join(IS_DOCUMENTED_KINDS)}."
                raise ValueError(error_message)
            # Both endpoints are gated: the entity through its ontology, the document through its own ACL.
            if not _check_entity_permissions([entity_id], effective_user_id, "UPDATE"):
                error_message = "Access denied for this knowledge graph operation."
                raise PermissionError(error_message)
            if not _check_document_permission(linked_document_id, effective_user_id, "READ"):
                error_message = "Access denied for this knowledge graph operation."
                raise PermissionError(error_message)
            resource_lock = build_resource_lock("ENTITY", entity_id)
            with resource_lock:
                operation_successful = entity_document_link_create(entity_id, linked_document_id, link_kind, effective_user_id)
            pass

        elif operation == "DOCUMENT_LINK_DELETE":
            if not entity_id or not linked_document_id:
                error_message = "Entity ID and Linked Document ID are required."
                raise ValueError(error_message)
            if link_kind not in IS_DOCUMENTED_KINDS:
                error_message = f"Link kind must be one of: {', '.join(IS_DOCUMENTED_KINDS)}."
                raise ValueError(error_message)
            if not _check_entity_permissions([entity_id], effective_user_id, "DELETE"):
                error_message = "Access denied for this knowledge graph operation."
                raise PermissionError(error_message)
            if not _check_document_permission(linked_document_id, effective_user_id, "READ"):
                error_message = "Access denied for this knowledge graph operation."
                raise PermissionError(error_message)
            resource_lock = build_resource_lock("ENTITY", entity_id)
            with resource_lock:
                operation_successful = entity_document_link_delete(entity_id, linked_document_id, link_kind)
            pass

        else:
            error_message = f"Operation '{operation}' is not supported."
            raise ValueError(error_message)

        arguments_not_null = {key: value for key, value in arguments.items() if value is not None}
        result["output"] = arguments_not_null
        result["output"]["returned_objects"] = objects
        result["repeat_contextualizer"] = True   # this forces the ontology to be reloaded, so that the next assistants can read it
        result["success"] = True
        if not operation_successful:
            result["error"] = 1
            if error_message is not None:
                result["error_message"] = error_message
            elif operation == "RELATIONSHIP_CREATE_OR_UPDATE":
                result["error_message"] = (f"Failed to create/update relationship '{label}' between "
                                           f"'{relationship_start_entity_id}' and '{relationship_end_entity_id}'. "
                                           f"Verify that both entities exist and the relationship is allowed in the ontology.")
            elif operation == "ENTITY_CREATE_OR_UPDATE":
                result["error_message"] = f"Failed to create/update entity '{entity_id}'. Check that the community '{community_id}' exists."
            else:
                result["error_message"] = f"Operation '{operation}' failed. Check the server logs for details."

    except Exception as e:
        stack_trace = traceback.format_exc()
        if error_message is None:
            error_message = f"An error occurred while executing the query: {e}"
        log(ERROR, f"ERROR in KnowledgeGraphEditor ({operation}): {error_message}\n{stack_trace}")
        result = {"success": False, "error": 2 if is_warning else 1, "error_message": error_message, "warning": is_warning, "operation": operation, "arguments": arguments}

    if IMMEDIATE_CONSOLIDATION:
        entities_to_consolidate = None
        try:
            if result["success"]:
                if operation in ["ENTITY_CREATE_OR_UPDATE", "ENTITY_MERGE"]:
                    resouce_lock = get_lock("ENTITY", entity_id)
                    with resouce_lock:
                        entities_to_consolidate = [entity_id]
                        get_ontology_service().consolidate_entities(entities_to_consolidate)
                elif operation in ["RELATIONSHIP_CREATE_OR_UPDATE"]:
                    # consolidate the relationship_start_entity_id and the relationship_end_entity_id
                    resouce_lock = get_lock("ENTITY", relationship_start_entity_id)
                    with resouce_lock:
                        entities_to_consolidate = [relationship_start_entity_id]
                        get_ontology_service().consolidate_entities(entities_to_consolidate)

                    resouce_lock = get_lock("ENTITY", relationship_end_entity_id)
                    with resouce_lock:
                        entities_to_consolidate = [relationship_end_entity_id]
                        get_ontology_service().consolidate_entities(entities_to_consolidate)
        except Exception as e:
            stack_trace = traceback.format_exc()
            log(ERROR, f"ERROR: An error occurred while consolidating the entities '{entities_to_consolidate}': {e}\n{stack_trace}")
    
    return result


def test_merge_relationships():
    community_id = COMMUNITY_ID_DEFAULT
    docid = "doc_456"

    # Create first entity "Alice"
    result = KnowledgeGraphEditor(
        "ENTITY_CREATE_OR_UPDATE",
        entity_id="Alice",
        community_id=community_id,
        label="Person",
        attributes={"name": "Alice", "description": "Alice is a developer."},
        document_or_chapter_id=docid,
        effective_user_id="global"
    )
    print(f"Create Alice: {result}")

    # Create second entity "Bob"
    result = KnowledgeGraphEditor(
        "ENTITY_CREATE_OR_UPDATE",
        entity_id="Bob",
        community_id=community_id,
        label="Person",
        attributes={"name": "Bob", "description": "Bob is a designer."},
        document_or_chapter_id=docid,
        effective_user_id="global"
    )
    print(f"Create Bob: {result}")

    # Create third entity "Charly"
    result = KnowledgeGraphEditor(
        "ENTITY_CREATE_OR_UPDATE",
        entity_id="Charly",
        community_id=community_id,
        label="Person",
        attributes={"name": "Charly", "description": "Charly is an artist."},
        document_or_chapter_id=docid,
        effective_user_id="global"
    )
    print(f"Create Charly: {result}")

    # Create multiple "FRIENDS_WITH" relationships between Alice and Bob
    result = KnowledgeGraphEditor(
        "RELATIONSHIP_CREATE_OR_UPDATE",
        relationship_start_entity_id="Alice",
        relationship_end_entity_id="Bob",
        label="FRIENDS_WITH",
        attributes={"description": "Alice is friends with Bob."},
        document_or_chapter_id=docid,
        effective_user_id="global"
    )
    print(f"Create FRIENDS_WITH Relationship 1: {result}")

    result = KnowledgeGraphEditor(
        "RELATIONSHIP_CREATE_OR_UPDATE",
        relationship_start_entity_id="Alice",
        relationship_end_entity_id="Charly",
        label="FRIENDS_WITH",
        attributes={"description": "Alice and Charly went to school together."},
        document_or_chapter_id=docid,
        effective_user_id="global"
    )
    print(f"Create FRIENDS_WITH Relationship: {result}")

    # Alter the label of the second relationship, in order to create a duplicate
    query = """
        MATCH (e1:Entity {id: $entity_id1})-[r:FRIENDS_WITH]->(e3:Entity {id: $entity_id3}),
            (e2:Entity {id: $entity_id2})
        CREATE (e1)-[newr:FRIENDS_WITH]->(e2)
        SET newr += r
        DELETE r
    """
    parameters = {"entity_id1": "Alice", "entity_id2": "Bob", "entity_id3": "Charly"}
    result = execute_neo4j_query(query, parameters)

    # Merge relationships between Alice and Bob
    result = KnowledgeGraphEditor(
        "ENTITY_MERGE",
        entity_id="Bob",
        second_entity_id_to_merge="Charly",
        effective_user_id="global"
    )
    print(f"Merge FRIENDS_WITH Relationships: {result}")

    # Verify the merge
    query = """
        MATCH (a:Entity {id: $entity_id})-[r:FRIENDS_WITH]->(b:Entity {id: $second_entity_id})
        RETURN r
    """
    parameters = {"entity_id": "Alice", "second_entity_id": "Bob"}
    merged_relationship = execute_neo4j_query(query, parameters)
    print(f"Merged Relationships: {merged_relationship}")

    # Delete the entities
    result = KnowledgeGraphEditor("ENTITY_DELETE", entity_id="Alice", effective_user_id="global")
    print(f"Delete Alice: {result}")
    result = KnowledgeGraphEditor("ENTITY_DELETE", entity_id="Bob", effective_user_id="global")
    print(f"Delete Bob: {result}")
    result = KnowledgeGraphEditor("ENTITY_DELETE", entity_id="Charly", effective_user_id="global")
    print(f"Delete Charly: {result}")



log_end_file(__file__)


if __name__ == '__main__':
    from src.utils.unit_test import init_unit_test
    init_unit_test()
    from src.utils.unit_test import init_unit_test
    from src.core.llm import llm_init
    from src.utils.log import log_init, log_exit
    init_unit_test()
    log_init(disable_screen_lock=True)
    llm_init()
    print("Test started.")

    if False:
        # ad-hoc tests
        operation = "SEMANTIC_SEARCH"
        semantic_search_query = "Luca"
        result = KnowledgeGraphEditor(operation, semantic_search_query=semantic_search_query, effective_user_id="global")
        print(f"Test result: {result}")

    if False:
        # Test merge_relationships()
        test_merge_relationships()

    if False:
        result = entity_delete("nora", "Animal")
        print(f"Test result: {result}")

    if False:
        # Test basic attribute functionality
        original_attribute = [{"value": "Bummy is a nice-looking adorable bunny, owned by Jeanny", "source_id": "doc://docid-19?p=p-7"}]
        new_attribute_value = "Bummy is really ugly, it has a wide scar on the face and it's scary to look"
        source = "doc://docid-20?p=p-8"
        new_attribute_dict = {"value": new_attribute_value, "source_id": source}
        merged_attribute, changed = merge_attribute(original_attribute, [new_attribute_dict], case_sensitive=False)
        assert changed == True
        print(f"Test result: {changed}, {merged_attribute}")

        original_attribute = [{"value": "Bummy", "source_id": "doc://docid-19?p=p-7"}]
        new_attribute_value = "Bummy"
        source = "doc://docid-20?p=p-8"
        new_attribute_dict = {"value": new_attribute_value, "source_id": source}
        merged_attribute, changed = merge_attribute(original_attribute, [new_attribute_dict], case_sensitive=False)
        assert changed == True
        print(f"Test result: {changed}, {merged_attribute}")

        # Test adding, updating, removing entities and relationships
        community_id = COMMUNITY_ID_DEFAULT
        docid = "docid-19"
        result = KnowledgeGraphEditor("COMMUNITY_LIST", effective_user_id="global")
        print(f"Test result: {result}")

        if len(result["output"]) == 0:
            # Create a new community
            result = KnowledgeGraphEditor("COMMUNITY_CREATE", community_id=community_id, community_name=COMMUNITY_NAME_DEFAULT, community_description="This is the default community, it contains entities and relationships.", ontology_id="FictionalStories", effective_user_id="global")
            print(f"Test result: {result}")

        # Create two new entities
        entity_id = "Luca"
        result = KnowledgeGraphEditor("ENTITY_CREATE_OR_UPDATE", entity_id=entity_id, community_id=community_id, label="Person", attributes={"name":entity_id, "description":"Luca is a developer."}, document_or_chapter_id=docid, effective_user_id="global")
        print(f"Test result: {result}")
        result = KnowledgeGraphEditor("ENTITY_CREATE_OR_UPDATE", entity_id=entity_id, community_id=community_id, label="Person", attributes={"name":entity_id, "description":"Luca is a developer."}, document_or_chapter_id=docid, effective_user_id="global")
        print(f"Test result: {result}")
        result = KnowledgeGraphEditor("ENTITY_CREATE_OR_UPDATE", entity_id=entity_id, community_id=community_id, label="Person", attributes={"name":entity_id, "description":"Luca is an artist."}, document_or_chapter_id=docid, effective_user_id="global")
        print(f"Test result: {result}")
        entity_id = "Mario"
        result = KnowledgeGraphEditor("ENTITY_CREATE_OR_UPDATE", entity_id=entity_id, community_id=community_id, label="Person", attributes={"name":entity_id, "description":"Mario is a designer."}, document_or_chapter_id=docid, effective_user_id="global")
        print(f"Test result: {result}")

        # Merge the two entities
        entity_id = "Luca2"
        result = KnowledgeGraphEditor("ENTITY_CREATE_OR_UPDATE", entity_id=entity_id, community_id=community_id, label="Person", attributes={"name":"Luca", "description":"Luca lives in New York."}, document_or_chapter_id=docid, effective_user_id="global")

        # Create a new relationship
        relationship_label = "WORKS_WITH"
        result = KnowledgeGraphEditor("RELATIONSHIP_CREATE_OR_UPDATE", relationship_start_entity_id="Luca", relationship_end_entity_id="Mario", label=relationship_label, attributes={"description":"Luca works with Mario at the thalabus project."}, document_or_chapter_id=docid, effective_user_id="global")
        print(f"Test result: {result}")

        # Merge the two entities
        result = KnowledgeGraphEditor("ENTITY_MERGE", entity_id="Luca", second_entity_id_to_merge="Luca2", effective_user_id="global")
        print(f"Test result: {result}")

        # Delete a relationship
        result = KnowledgeGraphEditor("RELATIONSHIP_DELETE", relationship_start_entity_id="Luca", relationship_end_entity_id="Mario", label=relationship_label, effective_user_id="global")
        print(f"Test result: {result}")

        # Delete the two entities
        result = KnowledgeGraphEditor("ENTITY_DELETE", entity_id="Luca", effective_user_id="global")
        print(f"Test result: {result}")
        result = KnowledgeGraphEditor("ENTITY_DELETE", entity_id="Mario", effective_user_id="global")
        print(f"Test result: {result}")

    if False:
        # Consolidated call: avoid creating an intermediary payload dict
        result = KnowledgeGraphEditor(
            operation="RELATIONSHIP_CREATE_OR_UPDATE",
            community_id="DefaultCommunity_FictionalStories",
            label="owns",
            attributes={"description": "Jane owns Bob, the fluffy orange cat, showing affection and companionship."},
            relationship_start_entity_id="jane",
            relationship_end_entity_id="bob",
            document_or_chapter_id="docid-338",
            paragraph_id="p-4",
            effective_user_id="global"
        )

        print(json.dumps(result, indent=2))

    if True:
        # Test: seed an entity 'jane' with an existing name attribute that already lists multiple sources,
        # then add a new source (doc://docid-673?p=p-2) for the same name value and print the result.
        # The existing attribute value (as requested) is:
        # [{"value": "Jane", "source_id": "doc://docid-673?p=p-1,doc://docid-689?p=p-1,doc://docid-675?p=p-1"}]

        # Create or update the entity 'jane' with the existing attribute list
        initial_attr = '[{"value":"Jane","source_id":"doc://docid-673?p=p-1,doc://docid-689?p=p-1,doc://docid-675?p=p-1"}]'
        result = KnowledgeGraphEditor(
            "ENTITY_CREATE_OR_UPDATE",
            entity_id="jane",
            community_id=COMMUNITY_ID_DEFAULT,
            label="Person",
            attributes={"name": initial_attr},
            document_or_chapter_id="docid-673",
            paragraph_id="p-1",
            effective_user_id="global"
        )
        print(f"Initial create Jane: {result}")

        # Now add a new source (doc://docid-673?p=p-2) for the same name value "Jane".
        result = KnowledgeGraphEditor(
            "ENTITY_CREATE_OR_UPDATE",
            entity_id="jane",
            community_id=COMMUNITY_ID_DEFAULT,
            label="Person",
            attributes={"name": "Jane"},
            document_or_chapter_id="docid-673",
            paragraph_id="p-2",
            effective_user_id="global"
        )
        print(f"Add new source for Jane: {result}")

        # Query the database to show the merged attribute for inspection
        try:
            query = """MATCH (e:Entity {id: $entity_id}) RETURN e.name AS name"""
            parameters = {"entity_id": "jane"}
            query_result = execute_neo4j_query(query, parameters)
            print(f"Entity 'jane' name attribute after merge: {query_result}")
        except Exception as e:
            print(f"Failed to fetch jane: {e}")

    print("Test finished. Closing log.")
    log_exit()
    print("Goodbye.")
