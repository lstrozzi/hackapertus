from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

from typing import Dict, List, Union, Optional, Any
from src.core.base.InputOutput import execute_neo4j_query
from src.utils.log import log, DEBUG, INFO, WARNING, ERROR, FATAL
from src.interfaces.dossiers_interface import DossiersInterface
from src.interfaces.user_interface import UserInterface
from src.services.user_service import get_user_service
from src.utils.acl_cypher_utils import acl_dossier_where_clause, acl_dossier_permission_check_query

class DossiersService(DossiersInterface):

    def _check_dossier_permission(self, effective_user_id: str, dossier_id: str, permission: str) -> bool:
        """Check whether effective_user_id holds `permission` on the given dossier.

        Traverses the full ACL chain: tenant-admin bypass → direct GRANT →
        inherited GRANT from ancestor dossiers (stopping at isProtected).
        Returns True if access is granted, False on denial or error.
        """
        result = False
        try:
            query = acl_dossier_permission_check_query(permission)
            records = execute_neo4j_query(query, {"effective_user_id": effective_user_id, "dossier_id": dossier_id})
            if records and len(records) > 0:
                result = bool(records[0].get("has_access", False))
        except Exception as e:
            log(ERROR, f"DossiersService._check_dossier_permission: error checking {permission} on '{dossier_id}' for '{effective_user_id}': {e}")
        return result

    def get_all_dossiers(self, effective_user_id: Optional[str] = None) -> List[dict]:
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        result = []

        acl_clause = acl_dossier_where_clause("d") if effective_user_id else ""
        query = f"""
            MATCH (d:Dossier)
            WHERE true{acl_clause}
            OPTIONAL MATCH (d)-[:FOLLOWS]->(o:Ontology)
            OPTIONAL MATCH (d)-[:BASED_ON]->(t:DocumentTemplate)
            RETURN
                d.id AS id,
                d.name AS name,
                d.description AS description,
                o.id AS follows_ontology_id,
                o.name AS follows_ontology_name,
                t.templateid AS based_on_template_id,
                t.name AS based_on_template_name
        """
        parameters = {"effectiveUserId": effective_user_id} if effective_user_id else None
        try:
            log(DEBUG, f"Executing query: {query}...")
            documents = execute_neo4j_query(query, parameters=parameters)
            if not documents:
                return []
            for doc in documents:
                dossier = {
                    "id": doc.get("id")
                }
                if doc.get("name") is not None:
                    dossier["name"] = doc.get("name")
                if doc.get("description") is not None:
                    dossier["description"] = doc.get("description")

                # include optional followed ontology info when present
                if doc.get("follows_ontology_id") is not None:
                    dossier["follows_ontology_id"] = doc.get("follows_ontology_id")
                if doc.get("follows_ontology_name") is not None:
                    dossier["follows_ontology_name"] = doc.get("follows_ontology_name")

                # include optional based-on template info when present
                if doc.get("based_on_template_id") is not None:
                    dossier["based_on_template_id"] = doc.get("based_on_template_id")
                if doc.get("based_on_template_name") is not None:
                    dossier["based_on_template_name"] = doc.get("based_on_template_name")

                result.append(dossier)
        except Exception as e:
            error_message = f"An error occurred: {e}"
            log(DEBUG, f"ERROR: {error_message}")
            result = [{"error": 2, "error_message": error_message}]

        return result

    def get_dossier(self, dossier_id: str, effective_user_id: Optional[str] = None) -> Union[dict, None]:
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_dossier_permission(effective_user_id, dossier_id, "READ"):
            log(WARNING, f"DossiersService.get_dossier: READ access denied for effective user '{effective_user_id}' on dossier '{dossier_id}'")
            return None
        result = None

        query = """
            MATCH (dossier:Dossier)
            WHERE dossier.id=$id
            OPTIONAL MATCH (dossier)-[:CONTAINS]->(document:Document)
            OPTIONAL MATCH (dossier)-[:BASED_ON]->(template:DocumentTemplate)
            OPTIONAL MATCH (dossier)-[:REASONS_WITH]->(reasoning:Document)
            OPTIONAL MATCH (dossier)-[:FOLLOWS]->(o:Ontology)
            RETURN dossier, document.id as document_id, template.templateid as template_id, reasoning.id as reasoning_id,
                   o.id AS follows_ontology_id, o.name AS follows_ontology_name
        """
        parameters = {"id": dossier_id}
        
        try:
            log(DEBUG, f"Executing query: {query}...")
            records = execute_neo4j_query(query, parameters)
            if not records:
                result = None
            else:
                # Collect all document, template, and reasoning IDs
                document_ids = []
                template_ids = []
                reasoning_ids = []
                dossier_node = None
                for record in records:
                    if dossier_node is None:
                        dossier_node = record.get("dossier")
                    doc_id = record.get("document_id")
                    if doc_id is not None:
                        document_ids.append(doc_id)
                    template_id = record.get("template_id")
                    if template_id is not None:
                        template_ids.append(template_id)
                    reasoning_id = record.get("reasoning_id")
                    if reasoning_id is not None:
                        reasoning_ids.append(reasoning_id)

                dossier_info = {
                    "success": True,
                    "dossier_id": dossier_id,
                    "document_ids": list(set(document_ids)),
                    "template_ids": list(set(template_ids)),
                    "reasoning_ids": list(set(reasoning_ids)),
                }
                if dossier_node:
                    dossier_info.update({
                        "name": dossier_node.get("name"),
                        "description": dossier_node.get("description"),
                        "created": dossier_node.get("created"),
                        "updated": dossier_node.get("updated"),
                        "status": dossier_node.get("status"),
                        "owner": dossier_node.get("owner"),
                        
                        # include followed ontology when available (from OPTIONAL MATCH)
                        "follows_ontology_id": records[0].get("follows_ontology_id") if records and len(records) > 0 else None,
                        "follows_ontology_name": records[0].get("follows_ontology_name") if records and len(records) > 0 else None,
                    })
                result = dossier_info

        except Exception as e:
            error_message = f"An error occurred: {e}"
            log(DEBUG, f"ERROR: {error_message}")
            result = {"error": 2, "error_message": error_message}

        return result

    def get_ontology_id_for_dossier(self, dossier_id: str, effective_user_id: Optional[str] = None) -> Optional[str]:
        """Get the ontology ID that this dossier follows, if any.
        
        Args:
            dossier_id: The ID of the dossier
            effective_user_id: If provided, READ permission on the dossier is verified. Pass None to skip ACL.
            
        Returns:
            The ontology ID if the dossier follows one, None otherwise
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_dossier_permission(effective_user_id, dossier_id, "READ"):
            log(WARNING, f"DossiersService.get_ontology_id_for_dossier: READ access denied for effective user '{effective_user_id}' on dossier '{dossier_id}'")
            return None
        try:
            query = """
                MATCH (dossier:Dossier)
                WHERE dossier.id=$id
                OPTIONAL MATCH (dossier)-[:FOLLOWS]->(o:Ontology)
                RETURN o.id AS ontology_id
            """
            parameters = {"id": dossier_id}
            log(DEBUG, f"Executing query to get ontology for dossier {dossier_id}...")
            records = execute_neo4j_query(query, parameters)
            
            if records and len(records) > 0:
                ontology_id = records[0].get("ontology_id")
                if ontology_id:
                    log(DEBUG, f"Found ontology '{ontology_id}' for dossier '{dossier_id}'")
                    return ontology_id
            
            log(DEBUG, f"No ontology found for dossier '{dossier_id}'")
            return None
        except Exception as e:
            error_message = f"Error getting ontology for dossier: {e}"
            log(ERROR, error_message)
            return None

    def get_documents_of_dossier(self, dossier_id: str, include_details: bool = False, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_dossier_permission(effective_user_id, dossier_id, "READ"):
            log(WARNING, f"DossiersService.get_documents_of_dossier: READ access denied for effective user '{effective_user_id}' on dossier '{dossier_id}'")
            return []
        result = []

        if include_details:
            query = """
                MATCH (dossier:Dossier)
                WHERE dossier.id=$id
                OPTIONAL MATCH (dossier)-[:CONTAINS]->(document:Document)
                OPTIONAL MATCH (document)-[:HAS_SUMMARY]->(s:Summary)
                OPTIONAL MATCH (document)-[:HAS_KEYWORD]->(k:Keyword)
                OPTIONAL MATCH (document)-[:HAS_CATEGORY]->(cat:Category)-[:OF_CATEGORY]->(kc:KeywordCategory)
                RETURN document.documentid AS document_id,
                       document.type AS document_type,
                       document.name AS document_name,
                       document.title AS title,
                       document.text AS text,
                       document.enhanced_text AS enhanced_text,
                       document.status AS status,
                       document.extraction_quality AS extraction_quality,
                       document.pages AS pages,
                       document.uploaded_datetime AS uploaded_datetime,
                       document.processed_datetime AS processed_datetime,
                       s.text AS summary,
                       collect(DISTINCT {text: k.text, lang: k.lang}) AS keywords,
                       collect(DISTINCT {text: cat.text, category_id: kc.id, category_text_en: kc.text_en}) AS categories
            """
        else:
            query = """
                MATCH (dossier:Dossier)
                WHERE dossier.id=$id
                OPTIONAL MATCH (dossier)-[:CONTAINS]->(document:Document)
                RETURN document.documentid as document_id, document.type as document_type, document.name as document_name
            """

        parameters = {"id": dossier_id}
        
        try:
            log(DEBUG, f"Executing query: {query}...")
            records = execute_neo4j_query(query, parameters)
            if records:
                # Collect all documents information
                documents = []
                for record in records:
                    doc_id:Optional[str] = record.get("document_id")
                    if doc_id is not None:
                        doc_info: Dict[str, Any] = {
                            "id": record.get("document_id"),
                            "type": record.get("document_type"),
                            "name": record.get("document_name"),
                        }

                        if include_details:
                            # Map extended fields when available
                            doc_info.update({
                                "title": record.get("title"),
                                "text": record.get("text"),
                                "enhanced_text": record.get("enhanced_text"),
                                "status": record.get("status"),
                                "extraction_quality": record.get("extraction_quality"),
                                "pages": record.get("pages"),
                                "uploaded_datetime": record.get("uploaded_datetime"),
                                "processed_datetime": record.get("processed_datetime"),
                                "summary": record.get("summary"),
                                "keywords": record.get("keywords"),
                                "categories": record.get("categories"),
                            })

                        documents.append(doc_info)

                result = documents

        except Exception as e:
            error_message = f"An error occurred: {e}"
            log(DEBUG, f"ERROR: {error_message}")
            result = {"error": 2, "error_message": error_message}

        return result

    def link_document_to_dossier(self, dossier_id: str, document_ids: List[str], effective_user_id: Optional[str] = None) -> bool:
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_dossier_permission(effective_user_id, dossier_id, "ADD"):
            log(WARNING, f"DossiersService.link_document_to_dossier: ADD access denied for effective user '{effective_user_id}' on dossier '{dossier_id}'")
            return False
        success = False

        query = """
            MATCH (dossier:Dossier {id: $dossier_id})
            MATCH (document:Document)
            WHERE document.documentid IN $document_ids
            MERGE (dossier)-[:CONTAINS]->(document)
        """
        parameters = {
            "dossier_id": dossier_id,
            "document_ids": document_ids
        }
        
        try:
            log(DEBUG, f"Executing query: {query}...")
            execute_neo4j_query(query, parameters)
            success = True

        except Exception as e:
            error_message = f"An error occurred: {e}"
            log(DEBUG, f"ERROR: {error_message}")
            success = False

        return success

    def get_user_dossier_id(self, user_id: str) -> Optional[str]:
        """Return the ID of the personal dossier owned by the user's group.

        Traverses: (u:User:SystemAcl {id: user_id})-[:MEMBER_OF]->(g:Group:SystemAcl {isUserGroup: true})-[:OWNS]->(d:Dossier)
        """
        try:
            query = """
                MATCH (u:User:SystemAcl {id: $user_id})-[:MEMBER_OF]->(g:Group:SystemAcl {isUserGroup: true})-[:OWNS]->(d:Dossier)
                RETURN d.id AS dossier_id
                LIMIT 1
            """
            records = execute_neo4j_query(query, {"user_id": user_id})
            if records and len(records) > 0:
                dossier_id = records[0].get("dossier_id")
                if dossier_id:
                    log(DEBUG, f"DossiersService.get_user_dossier_id: found user dossier '{dossier_id}' for user '{user_id}'")
                    return dossier_id
            log(DEBUG, f"DossiersService.get_user_dossier_id: no user dossier found for user '{user_id}'")
            return None
        except Exception as e:
            log(ERROR, f"DossiersService.get_user_dossier_id: error for user '{user_id}': {e}")
            return None

    def get_dossiers_for_document(self, document_id: str, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all dossiers that contain the given document."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        result: List[Dict[str, Any]] = []
        try:
            acl_clause = acl_dossier_where_clause("d") if effective_user_id else ""
            parameters: Dict[str, Any] = {"document_id": document_id}
            if effective_user_id:
                parameters["effectiveUserId"] = effective_user_id
            query = f"""
                MATCH (d:Dossier)-[:CONTAINS]->(doc:Document)
                WHERE doc.documentid = $document_id{acl_clause}
                RETURN d.id AS id, d.name AS name, d.description AS description
            """
            records = execute_neo4j_query(query, parameters)
            if records:
                for record in records:
                    dossier_id = record.get("id")
                    if dossier_id:
                        result.append({
                            "id": dossier_id,
                            "name": record.get("name"),
                            "description": record.get("description"),
                        })
            log(DEBUG, f"DossiersService.get_dossiers_for_document: found {len(result)} dossier(s) for document '{document_id}'")
        except Exception as e:
            log(ERROR, f"DossiersService.get_dossiers_for_document: error for document '{document_id}': {e}")
        return result

    def count_documents_in_dossier(self, dossier_id: str, effective_user_id: Optional[str] = None) -> int:
        """Count documents linked to the given dossier via [:CONTAINS]."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_dossier_permission(effective_user_id, dossier_id, "READ"):
            log(WARNING, f"DossiersService.count_documents_in_dossier: READ access denied for effective user '{effective_user_id}' on dossier '{dossier_id}'")
            return 0
        try:
            query = """
                MATCH (dossier:Dossier {id: $dossier_id})-[:CONTAINS]->(doc:Document)
                RETURN count(doc) AS doc_count
            """
            params = {"dossier_id": dossier_id}
            result = execute_neo4j_query(query, params)
            if result and isinstance(result, list) and len(result) > 0:
                count = result[0].get("doc_count")
                try:
                    return int(count) if count is not None else 0
                except Exception:
                    return 0
            return 0
        
        except Exception as e:
            error_message = f"An error occurred while counting documents in dossier {dossier_id}: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return 0
    
    def create_dossier(self, id: str, name: str, description: str, effective_user_id: Optional[str] = None) -> Union[dict, None]:
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()

        result = None

        query = """
            MERGE (dossier:Dossier {
                id: $id
            })
            SET dossier.name = $name,
                dossier.description = $description
            RETURN dossier
        """
        parameters = {
            "id": id,
            "name": name,
            "description": description
        }
        
        try:
            log(DEBUG, f"Executing query: {query}...")
            records = execute_neo4j_query(query, parameters)
            if records:
                dossier_node = records[0].get("dossier")
                result = {
                    "id": dossier_node.get("id"),
                    "name": dossier_node.get("name"),
                    "description": dossier_node.get("description")
                }
                
                # Grant Owner permissions to effective user's group
                if effective_user_id:
                    # Get the effective user's user group (isUserGroup=true) using effective_user_id directly
                    group_query = """
                        MATCH (u:User:SystemAcl {id: $effective_user_id})-[:MEMBER_OF]->(g:Group:SystemAcl {isUserGroup: true})
                        RETURN g.id AS group_id
                    """
                    group_records = execute_neo4j_query(group_query, {"effective_user_id": effective_user_id})
                    if group_records and len(group_records) > 0:
                        actor_group_id = group_records[0].get("group_id")
                        if actor_group_id:
                            # Use centralized ACL functions
                            from src.utils.acl_cypher_utils import (
                                acl_create_owns_query,
                                acl_create_grant_permissions_query
                            )
                            
                            # Create OWNS relationship
                            owns_query = acl_create_owns_query(actor_group_id, id, "Dossier")
                            execute_neo4j_query(owns_query, {"group_id": actor_group_id, "resource_id": id})
                            log(DEBUG, f"Created OWNS relationship from group '{actor_group_id}' to dossier '{id}'")
                            
                            # Grant all Owner permissions to effective user's group
                            grant_query = acl_create_grant_permissions_query(
                                actor_group_id, id, ["READ", "SHARING", "DELETE", "ADD", "UPDATE", "OWNERSHIP"], "Dossier"
                            )
                            grant_params = {
                                "group_id": actor_group_id,
                                "resource_id": id,
                                "permissions": ["READ", "SHARING", "DELETE", "ADD", "UPDATE", "OWNERSHIP"],
                                "grantedAt": "2026-07-10T00:00:00.000Z",
                                "grantedBy": effective_user_id
                            }
                            execute_neo4j_query(grant_query, grant_params)
                            log(DEBUG, f"Granted Owner permissions to group '{actor_group_id}' on dossier '{id}'")
                
                # Also grant Owner permissions to global user if effective user is not global
                # (global user is system/admin and should have access to all resources)
                if effective_user_id and effective_user_id != "global":
                    # Get the global user's group
                    global_group_query = """
                        MATCH (u:User:SystemAcl {id: 'global'})-[:MEMBER_OF]->(g:Group:SystemAcl {isUserGroup: true})
                        RETURN g.id AS group_id
                    """
                    global_group_records = execute_neo4j_query(global_group_query, {})
                    if global_group_records and len(global_group_records) > 0:
                        global_group_id = global_group_records[0].get("group_id")
                        if global_group_id:
                            # Use centralized ACL functions
                            from src.utils.acl_cypher_utils import acl_create_grant_permissions_query
                            
                            # Grant all Owner permissions to global group (system/admin access)
                            grant_query = acl_create_grant_permissions_query(
                                global_group_id, id, ["READ", "SHARING", "DELETE", "ADD", "UPDATE", "OWNERSHIP"], "Dossier"
                            )
                            grant_params = {
                                "group_id": global_group_id,
                                "resource_id": id,
                                "permissions": ["READ", "SHARING", "DELETE", "ADD", "UPDATE", "OWNERSHIP"],
                                "grantedAt": "2026-07-10T00:00:00.000Z",
                                "grantedBy": effective_user_id
                            }
                            execute_neo4j_query(grant_query, grant_params)
                            log(DEBUG, f"Granted Owner permissions to global group '{global_group_id}' on dossier '{id}'")

        except Exception as e:
            error_message = f"An error occurred: {e}"
            log(DEBUG, f"ERROR: {error_message}")
            result = {"error": 2, "error_message": error_message}

        return result

    def delete_dossier(self, dossier_id: str, effective_user_id: Optional[str] = None) -> bool:
        """
        Delete the dossier node (and all relationships) identified by `dossier_id`.

        This performs a `DETACH DELETE` so connected nodes are not removed,
        only the dossier node and its relationships are removed.
        Returns True on success, False on denial, failure, or if dossier not found.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_dossier_permission(effective_user_id, dossier_id, "DELETE"):
            log(WARNING, f"DossiersService.delete_dossier: DELETE access denied for effective user '{effective_user_id}' on dossier '{dossier_id}'")
            return False
        try:
            # Check existence
            check_q = """
                MATCH (d:Dossier {id: $id})
                RETURN d.id AS id
            """
            params = {"id": dossier_id}
            records = execute_neo4j_query(check_q, params)
            if not records:
                # dossier not found
                return False

            delete_q = """
                MATCH (d:Dossier {id: $id})
                DETACH DELETE d
            """
            execute_neo4j_query(delete_q, params)
            return True
        except Exception as e:
            error_message = f"An error occurred while deleting dossier {dossier_id}: {e}"
            log(DEBUG, f"ERROR: {error_message}")
            return False

    def get_category_usage_stats_in_dossier(self, dossier_id: str, category_text_en: str, effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve usage statistics for a category in a dossier.
        
        Shows how many documents in the dossier have the specified category assigned.
        If the KeywordCategory defines a LoV (List of Values), also shows the count breakdown by value.
        
        Args:
            dossier_id: The ID of the dossier to analyze.
            category_text_en: The text_en value of the KeywordCategory (e.g., 'COUNTRY_OF_ORIGIN').
            effective_user_id: If provided, READ permission on the dossier is verified. Pass None to skip ACL.
        
        Returns:
            Dictionary containing usage statistics for the category.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_dossier_permission(effective_user_id, dossier_id, "READ"):
            log(WARNING, f"DossiersService.get_category_usage_stats_in_dossier: READ access denied for effective user '{effective_user_id}' on dossier '{dossier_id}'")
            return {"error": "Access denied", "category": category_text_en}
        try:
            # First, get the KeywordCategory and its LoV
            query_kc = """
                MATCH (kc:KeywordCategory {text_en: $category_text_en})
                OPTIONAL MATCH (o:Ontology)-[:CONTAINS]->(kc)
                RETURN kc.id AS category_id, kc.text_en AS category_text_en, kc.lov AS lov, o.id AS ontology_id
                LIMIT 1
            """
            params_kc = {"category_text_en": category_text_en}
            kc_result = execute_neo4j_query(query_kc, params_kc)
            
            if not kc_result or len(kc_result) == 0:
                return {
                    "error": f"Category '{category_text_en}' not found",
                    "category": category_text_en
                }
            
            kc_data = kc_result[0]
            category_id = kc_data.get("category_id")
            ontology_id = kc_data.get("ontology_id")
            lov = kc_data.get("lov")  # The List of Values string, if defined
            
            # Get total count of documents with this category in the dossier
            query_total = """
                MATCH (dossier:Dossier {id: $dossier_id})-[:CONTAINS]->(d:Document)
                MATCH (d)-[:HAS_CATEGORY]->(c:Category)-[:OF_CATEGORY]->(kc:KeywordCategory {text_en: $category_text_en})
                RETURN count(DISTINCT d.id) AS total_count
            """
            params_total = {"dossier_id": dossier_id, "category_text_en": category_text_en}
            total_result = execute_neo4j_query(query_total, params_total)
            total_count = total_result[0].get("total_count", 0) if total_result else 0
            
            # Get breakdown by value (if needed)
            value_counts = {}
            if total_count > 0:
                query_values = """
                    MATCH (dossier:Dossier {id: $dossier_id})-[:CONTAINS]->(d:Document)
                    MATCH (d)-[:HAS_CATEGORY]->(c:Category)-[:OF_CATEGORY]->(kc:KeywordCategory {text_en: $category_text_en})
                    RETURN c.text AS value, count(DISTINCT d.id) AS count
                    ORDER BY count DESC, value ASC
                """
                params_values = {"dossier_id": dossier_id, "category_text_en": category_text_en}
                values_result = execute_neo4j_query(query_values, params_values)
                
                if values_result:
                    for row in values_result:
                        value = row.get("value") or "(no value)"
                        count = row.get("count", 0)
                        value_counts[value] = count
            
            log(INFO, f"[dossier: {dossier_id}] Category '{category_text_en}' usage: {total_count} documents")
            
            return {
                "category": category_text_en,
                "ontology_id": ontology_id,
                "total_count": total_count,
                "has_lov": lov is not None and len(lov.strip()) > 0,
                "lov": lov,
                "value_counts": value_counts
            }
        
        except Exception as e:
            error_message = f"Error retrieving category usage stats: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {
                "error": error_message,
                "category": category_text_en
            }

# Singleton instance for dependency injection
_dossiers_service_instance: Optional[DossiersInterface] = None

def get_dossiers_service() -> DossiersInterface:
    """Get the singleton instance of the dossiers service."""
    global _dossiers_service_instance
    if _dossiers_service_instance is None:
        _dossiers_service_instance = DossiersService()
    return _dossiers_service_instance

def set_dossiers_service(service: DossiersInterface) -> None:
    """Set a custom dossiers service implementation (useful for testing)."""
    global _dossiers_service_instance
    _dossiers_service_instance = service


log_end_file(__file__)

if __name__ == '__main__':
    from src.utils.unit_test import init_unit_test
    from src.utils.log import log_init, log_exit
    init_unit_test()
    log_init()

    dossiers_service = get_dossiers_service()
    all_dossiers = dossiers_service.get_all_dossiers()
    print(f"All Dossiers: {all_dossiers}")

    first_dossier = all_dossiers[0] if all_dossiers else None
    first_dossier = dossiers_service.get_dossier(first_dossier["id"] if first_dossier else "dossier-1")
    print(f"First Dossier: {first_dossier}")

    log_exit()
