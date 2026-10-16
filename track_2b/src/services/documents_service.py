from src.interfaces.dossiers_interface import DossiersInterface
from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import os
from typing import List, Union, Optional, Dict, Any
from src.core.base.InputOutput import execute_neo4j_query, get_document_path
from src.utils.log import log, DEBUG, INFO, WARNING, ERROR, FATAL
from src.core.base.enums import DocumentStatus, KnowledgeGraphStatus
from src.interfaces.user_interface import UserInterface
from src.interfaces.documents_interface import DocumentsInterface, FragmentType
from src.utils.document_partition import partition_letter_from_hash as _partition_letter_from_hash
from src.utils.acl_cypher_utils import acl_document_where_clause, acl_document_permission_check_query
from src.services.user_service import get_user_service
from src.services.dossiers_service import get_dossiers_service

class DocumentsService(DocumentsInterface):

    def _check_document_permission(self, effective_user_id: str, document_id: str, permission: str) -> bool:
        """Check whether effective_user_id holds `permission` on the given document.

        Traverses the full ACL chain: tenant-admin bypass → direct GRANT →
        inherited GRANT from ancestor dossiers (stopping at isProtected).
        Returns True if access is granted, False on denial or error.
        """
        result = False
        try:
            query = acl_document_permission_check_query(permission)
            records = execute_neo4j_query(query, {"effective_user_id": effective_user_id, "document_id": document_id})
            if records and len(records) > 0:
                result = bool(records[0].get("has_access", False))
        except Exception as e:
            log(ERROR, f"DocumentsService._check_document_permission: error checking {permission} on '{document_id}' for '{effective_user_id}': {e}")
        return result

    def get_all_documents(self, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all documents from Neo4j database. When effective_user_id is provided, only documents the effective user can READ are returned."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        result = []

        acl_clause = acl_document_where_clause("d") if effective_user_id else ""
        query = f"""
            MATCH (d:Document)
            WHERE true{acl_clause}
            OPTIONAL MATCH (d)-[:HAS_SUMMARY]->(s:Summary)
            RETURN
                d.id AS id,
                d.documentid AS documentid,
                d.name AS name,
                d.title AS title,
                d.label AS label,
                d.text AS text,
                d.enhanced_text AS enhanced_text,
                d.type AS type,
                d.status AS status,
                d.knowledge_graph_status AS knowledge_graph_status,
                d.scanned AS scanned,
                d.extraction_quality AS extraction_quality,
                d.pages AS pages,
                d.uploaded_datetime AS uploaded_datetime,
                d.processed_datetime AS processed_datetime,
                s.text AS summary
        """
        try:
            parameters = {"effectiveUserId": effective_user_id} if effective_user_id else None
            documents = execute_neo4j_query(query, parameters=parameters)
            if not documents:
                return []
            
            for doc in documents:
                document = {
                    "id": doc.get("id") or doc.get("documentid"),
                    "documentid": doc.get("documentid") or doc.get("id")
                }
                
                # Add optional fields if they exist
                optional_fields = ["name", "title", "label", "text", "enhanced_text", "type", "status", 
                                 "knowledge_graph_status", "scanned", "extraction_quality", "pages", "uploaded_datetime", 
                                 "processed_datetime", "summary"]
                
                for field in optional_fields:
                    if doc.get(field) is not None:
                        document[field] = doc.get(field)
                
                result.append(document)
                
        except Exception as e:
            error_message = f"An error occurred: {e}"
            log(ERROR, f"ERROR: {error_message}")
            result = [{"error": 2, "error_message": error_message}]

        return result

    def get_document(self, document_id: str, include_details: bool = False, include_category_translations: bool = True, include_text: bool = True, include_summary: bool = True, effective_user_id: Optional[str] = None) -> Union[Dict[str, Any], None]:
        """Retrieve a single document by its ID. Delegates to get_documents() for a single-element list."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        docs = self.get_documents(
            [document_id],
            include_text=include_text,
            include_summary=include_summary,
            include_details=include_details,
            include_category_translations=include_category_translations,
            effective_user_id=effective_user_id,
        )
        return docs.get(document_id)

    def get_documents(self, document_ids: List[str], include_text: bool = False, include_summary: bool = True, include_details: bool = False, include_category_translations: bool = True, merge_clause: Optional[str] = None, limit: Optional[int] = None, effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve multiple documents by their IDs in a single batched Neo4j query.

        text/enhanced_text are only fetched when include_text=True.
        The HAS_SUMMARY subquery is only issued when include_summary=True.
        When include_details=True, keyword and category subqueries are added and processed.
        merge_clause allows additional WHERE filtering if provided.
        limit allows restricting the number of results returned.
        Returns a dict mapping each requested document_id to its document dict, or to None if not found or access denied.
        When effective_user_id is provided, only documents the effective user has READ permission on are returned.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        result: Dict[str, Any] = {doc_id: None for doc_id in document_ids}
        if not document_ids:
            return result

        summary_subquery = """
                CALL (d) {
                    OPTIONAL MATCH (d)-[:HAS_SUMMARY]->(s:Summary)
                    RETURN s.text AS summary
                }
"""
        keywords_subquery = """
                CALL (d) {
                    OPTIONAL MATCH (d)-[:HAS_KEYWORD]->(k:Keyword)
                    RETURN collect(DISTINCT {text: k.text, lang: k.lang, category_id: null, category_text_en: null, category_text_fr: null, category_text_it: null, category_text_de: null, use_as_document_label: null}) AS keywords
                }

                CALL (d) {
                    OPTIONAL MATCH (d)-[:HAS_CATEGORY]->(cat:Category)-[:OF_CATEGORY]->(kc:KeywordCategory)
                    RETURN collect(DISTINCT {text: cat.text, lang: null, category_id: kc.id, category_text_en: kc.text_en, category_text_fr: kc.text_fr, category_text_it: kc.text_it, category_text_de: kc.text_de, use_as_document_label: kc.use_as_document_label}) AS categories
                }
"""
        text_fields = """
                       d.text AS text, d.enhanced_text AS enhanced_text,"""
        summary_field = """
                       summary,"""
        details_fields = """
                       keywords, categories,"""

        acl_clause = acl_document_where_clause("d") if effective_user_id else ""
        query = (
            "UNWIND $document_ids AS doc_id\n"
            "MATCH (d:Document)\n"
            "WHERE (d.id = doc_id OR d.documentid = doc_id)" + acl_clause + "\n"
        )
        if merge_clause:
            query += f"AND {merge_clause}\n"
        
        query += (
            (summary_subquery if include_summary else "")
            + """
                CALL (d) {
                    OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chapter)
                    RETURN collect(DISTINCT c.id) AS chapter_ids
                }

                CALL (d) {
                    OPTIONAL MATCH (d)<-[:CONTAINS]-(dossier:Dossier)
                    RETURN collect(DISTINCT dossier.id) AS dossier_ids
                }

"""
            + (keywords_subquery if include_details else "")
            + """                RETURN doc_id,
                       d.id AS id, d.documentid AS documentid,
                       d.name AS name, d.title AS title, d.label AS label,"""
            + (text_fields if include_text else "")
            + """
                       d.type AS type, d.status AS status, d.knowledge_graph_status AS knowledge_graph_status,
                       d.scanned AS scanned, d.extraction_quality AS extraction_quality,
                       d.pages AS pages, d.uploaded_datetime AS uploaded_datetime,
                       d.processed_datetime AS processed_datetime,"""
            + (summary_field if include_summary else "")
            + (details_fields if include_details else "")
            + """
                       chapter_ids, dossier_ids
            """
        )
        if limit is not None:
            query += f"LIMIT {limit}\n"

        try:
            parameters = {"document_ids": document_ids}
            if effective_user_id:
                parameters["effectiveUserId"] = effective_user_id
            records = execute_neo4j_query(query, parameters)
            for record in (records or []):
                doc_id_key = record.get("doc_id")
                doc_id_val = record.get("id") or record.get("documentid")
                lookup_key = doc_id_key or doc_id_val
                if lookup_key is None:
                    continue
                entry = {
                    "success": True,
                    "id": doc_id_val or doc_id_key,
                    "documentid": record.get("documentid") or doc_id_val or doc_id_key,
                    "name": record.get("name"),
                    "title": record.get("title"),
                    "label": record.get("label"),
                    "text": record.get("text") if include_text else None,
                    "enhanced_text": record.get("enhanced_text") if include_text else None,
                    "type": record.get("type"),
                    "status": record.get("status"),
                    "knowledge_graph_status": record.get("knowledge_graph_status"),
                    "scanned": record.get("scanned"),
                    "extraction_quality": record.get("extraction_quality"),
                    "pages": record.get("pages"),
                    "uploaded_datetime": record.get("uploaded_datetime"),
                    "processed_datetime": record.get("processed_datetime"),
                    "summary": record.get("summary") if include_summary else None,
                    "chapter_ids": [cid for cid in (record.get("chapter_ids") or []) if cid],
                    "dossier_ids": [did for did in (record.get("dossier_ids") or []) if did],
                }
                if include_details:
                    keywords_raw = record.get("keywords", [])
                    categories_raw = record.get("categories", [])
                    entry['keywords'] = {'*': [], 'DE': [], 'FR': [], 'IT': [], 'EN': []}
                    entry['categories'] = []
                    try:
                        kw_buckets = {'*': set(), 'DE': set(), 'FR': set(), 'IT': set(), 'EN': set()}
                        categories_dict: Dict[str, Dict[str, Any]] = {}
                        category_for_label: Optional[Dict[str, Any]] = None

                        for k in keywords_raw or []:
                            if isinstance(k, dict):
                                text = k.get('text')
                                lang = k.get('lang')
                            else:
                                text = k
                                lang = None

                            if not text:
                                continue

                            text_s = str(text).strip()
                            if not text_s:
                                continue

                            if lang is None or (isinstance(lang, str) and lang.strip() == '*'):
                                kw_buckets['*'].add(text_s)
                            else:
                                lang_norm = str(lang).strip().upper()
                                if lang_norm in ('DE', 'FR', 'IT', 'EN'):
                                    kw_buckets[lang_norm].add(text_s)
                                else:
                                    kw_buckets['*'].add(text_s)

                        for cat in categories_raw or []:
                            if isinstance(cat, dict):
                                category_id = cat.get('category_id')
                                cat_text = cat.get('text')
                                cat_text_en = cat.get('category_text_en')
                                cat_text_fr = cat.get('category_text_fr')
                                cat_text_it = cat.get('category_text_it')
                                cat_text_de = cat.get('category_text_de')
                                use_as_document_label = cat.get('use_as_document_label')

                                if category_id:
                                    cid = str(category_id)
                                    if cid not in categories_dict:
                                        cat_entry = {'id': cid, 'text_en': cat_text_en}
                                        if include_category_translations:
                                            cat_entry['text_fr'] = cat_text_fr
                                            cat_entry['text_it'] = cat_text_it
                                            cat_entry['text_de'] = cat_text_de
                                        categories_dict[cid] = cat_entry

                                        if use_as_document_label is True:
                                            category_for_label = {'id': cid, 'text_en': cat_text_en, 'value': cat_text}
                                            if include_category_translations:
                                                category_for_label['text_fr'] = cat_text_fr
                                                category_for_label['text_it'] = cat_text_it
                                                category_for_label['text_de'] = cat_text_de

                        for kk, vv in kw_buckets.items():
                            entry['keywords'][kk] = sorted(list(vv))
                        entry['categories'] = sorted(list(categories_dict.values()), key=lambda x: x.get('id'))
                        entry['category_for_document_label'] = category_for_label

                    except Exception:
                        pass

                result[lookup_key] = entry
                # also index by the requested ID in case the stored id differs slightly
                if doc_id_key and doc_id_key != lookup_key:
                    result[doc_id_key] = entry
        except Exception as e:
            error_message = f"An error occurred: {e}"
            log(ERROR, f"ERROR: {error_message}")

        return result

    def get_document_categories(self, document_id: str, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve categories assigned to a document.

        Returns a list of dicts with keys: category_id, category_key, category_value
        where `category_key` is typically `kc.text_en` and `category_value` is `c.text` (may be None).
        When effective_user_id is provided, READ permission on the document is verified.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id:
            doc = self.get_document(document_id, effective_user_id=effective_user_id)
            if doc is None:
                log(WARNING, f"DocumentsService.get_document_categories: READ access denied for effective user '{effective_user_id}' on document '{document_id}'")
                return []
        
        result: List[Dict[str, Any]] = []
        try:
            query = """
                MATCH (d:Document)
                WHERE d.id = $document_id OR d.documentid = $document_id
                OPTIONAL MATCH (d)-[:HAS_CATEGORY]->(c:Category)-[:OF_CATEGORY]->(kc:KeywordCategory)
                RETURN kc.id AS category_id, kc.text_en AS category_key, c.text AS category_value
                ORDER BY category_id
            """
            params = {"document_id": document_id}
            rows = execute_neo4j_query(query, params) or []
            for r in rows:
                cid = r.get('category_id') or r.get('category_key')
                # skip entries where no category definition is found
                if not cid:
                    continue
                result.append({
                    'category_id': r.get('category_id'),
                    'category_key': r.get('category_key'),
                    'category_value': r.get('category_value')
                })
        except Exception as e:
            error_message = f"An error occurred while retrieving categories: {e}"
            log(ERROR, f"ERROR: {error_message}")
        return result

    def document_has_any_categories(self, document_id: str, effective_user_id: Optional[str] = None) -> bool:
        """Return True when the given document has any :HAS_CATEGORY relationships.

        This is a simple existence check used to determine whether the document
        already has categories assigned and therefore should be skipped when
        only uncategorized documents are requested.
        When effective_user_id is provided, READ permission on the document is verified.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id:
            doc = self.get_document(document_id, effective_user_id=effective_user_id)
            if doc is None:
                return False
        
        try:
            cnt = self.count_document_categories(document_id)
            return int(cnt) > 0
        except Exception as e:
            log(DEBUG, f"document_has_any_categories(): count query failed: {e}")
            return False

    def count_document_categories(self, document_id: str, effective_user_id: Optional[str] = None) -> int:
        """Return the number of Category nodes linked to the given document via :HAS_CATEGORY.

        When effective_user_id is provided, READ permission on the document is verified.
        Returns 0 on error or access denied.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        # If effective_user_id provided, verify READ access to the document first
        if effective_user_id:
            doc = self.get_document(document_id, effective_user_id=effective_user_id)
            if doc is None:
                log(WARNING, f"DocumentsService.count_document_categories: READ access denied for effective user '{effective_user_id}' on document '{document_id}'")
                return 0

        try:
            query = """
                MATCH (d:Document)
                WHERE d.id = $document_id OR d.documentid = $document_id
                OPTIONAL MATCH (d)-[:HAS_CATEGORY]->(c:Category)
                RETURN count(c) AS cnt
            """
            params = {"document_id": document_id}
            rows = execute_neo4j_query(query, params) or []
            if rows and len(rows) > 0:
                return int(rows[0].get('cnt', 0) or 0)
        except Exception as e:
            log(DEBUG, f"count_document_categories(): query failed: {e}")
        return 0

    def remove_document_category(self, document_id: str, category_text_en: str, effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Remove a single category from a document.
        
        Deletes only the HAS_CATEGORY relationship, not the Category node itself.
        Orphaned Category nodes are cleaned up by the maintenance thread.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_document_permission(effective_user_id, document_id, "EDIT"):
            log(WARNING, f"DocumentsService.remove_document_category: EDIT access denied for effective user '{effective_user_id}' on document '{document_id}'")
            return {"error": 1, "error_message": "EDIT permission denied"}
        try:
            query = """
                MATCH (d:Document)
                WHERE d.id = $document_id OR d.documentid = $document_id
                MATCH (d)-[r:HAS_CATEGORY]->(c:Category)-[:OF_CATEGORY]->(kc:KeywordCategory)
                WHERE kc.text_en = $category_text_en
                DELETE r
                RETURN count(r) AS deleted_count
            """
            params = {"document_id": document_id, "category_text_en": category_text_en}
            result = execute_neo4j_query(query, params)
            
            if result:
                deleted_count = result[0].get("deleted_count", 0) if result else 0
                log(INFO, f"[doc: {document_id}] Removed category '{category_text_en}' (deleted {deleted_count} relationship(s))")
                return {"success": True, "document_id": document_id, "deleted_count": deleted_count}
            else:
                log(WARNING, f"Query result was empty for document {document_id}")
                return {"error": 3, "error_message": "Document not found or failed to remove category"}
        except Exception as e:
            error_message = f"An error occurred while removing category: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def remove_all_document_categories(self, document_id: str, effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Remove all categories from a document.
        
        Deletes only the HAS_CATEGORY relationships, not the Category nodes themselves.
        Orphaned Category nodes are cleaned up by the maintenance thread.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_document_permission(effective_user_id, document_id, "EDIT"):
            log(WARNING, f"DocumentsService.remove_all_document_categories: EDIT access denied for effective user '{effective_user_id}' on document '{document_id}'")
            return {"error": 1, "error_message": "EDIT permission denied"}
        try:
            query = """
                MATCH (d:Document)
                WHERE d.id = $document_id OR d.documentid = $document_id
                MATCH (d)-[r:HAS_CATEGORY]->(c:Category)
                DELETE r
                RETURN count(r) AS deleted_count
            """
            params = {"document_id": document_id}
            result = execute_neo4j_query(query, params)
            
            if result:
                deleted_count = result[0].get("deleted_count", 0) if result else 0
                log(INFO, f"[doc: {document_id}] Removed all {deleted_count} category relationship(s)")
                return {"success": True, "document_id": document_id, "deleted_count": deleted_count}
            else:
                log(WARNING, f"Query result was empty for document {document_id}")
                return {"error": 3, "error_message": "Document not found or failed to remove categories"}
        except Exception as e:
            error_message = f"An error occurred while removing all categories: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def upsert_document_category(self, document_id: str, category_text_en: str, category_value: Optional[str] = None, effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Upsert (create or update) a category for a document.
        
        If category_value is provided, uses that as the Category text.
        Otherwise uses category_text_en as the text.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_document_permission(effective_user_id, document_id, "EDIT"):
            log(WARNING, f"DocumentsService.upsert_document_category: EDIT access denied for effective user '{effective_user_id}' on document '{document_id}'")
            return {"error": 1, "error_message": "EDIT permission denied"}
        try:
            # Determine what text to use for the Category node
            category_text = (category_value.strip() if category_value else category_text_en.strip()).strip()

            # First verify that the KeywordCategory exists; if not, return a clear error
            try:
                check_query = """
                    MATCH (kc:KeywordCategory)
                    WHERE kc.text_en = $category_text_en
                    RETURN count(kc) AS cnt
                """
                check_params = {"category_text_en": category_text_en}
                check_res = execute_neo4j_query(check_query, check_params) or []
                if not check_res or int(check_res[0].get('cnt', 0)) == 0:
                    log(WARNING, f"KeywordCategory not found: {category_text_en}")
                    return {"error": 4, "error_message": f"KeywordCategory not found: {category_text_en}"}
            except Exception as e:
                log(ERROR, f"Error checking KeywordCategory existence: {e}")
                return {"error": 2, "error_message": f"Error checking KeywordCategory existence: {e}"}

            # If a category_value was provided and the KeywordCategory defines a List-of-Values (lov),
            # ensure the provided value is one of the allowed values.
            if category_value:
                # Validation is performed by separate service methods; call them here for enforcement
                check = self.validate_category_key(category_text_en)
                if not check.get('exists'):
                    log(WARNING, f"KeywordCategory not found: {category_text_en}")
                    return {"error": 4, "error_message": f"KeywordCategory not found: {category_text_en}"}

                if category_value:
                    val_check = self.validate_category_value(category_text_en, category_value)
                    if not val_check.get('valid'):
                        allowed = val_check.get('allowed') or []
                        log(WARNING, f"Invalid value '{category_value}' for category '{category_text_en}'. Allowed: {allowed}")
                        return {"error": 5, "error_message": f"Invalid value '{category_value}' for category '{category_text_en}'. Allowed: {', '.join(allowed)}"}
                # Before upserting, remove any existing Category nodes that are linked to
                # this document and to the same KeywordCategory. This prevents duplicate
                # Category nodes for the same (document, keywordcategory) pair.
                delete_query = """
                    MATCH (d:Document)
                    WHERE d.id = $document_id OR d.documentid = $document_id
                    MATCH (d)-[:HAS_CATEGORY]->(c:Category)-[:OF_CATEGORY]->(kc:KeywordCategory {text_en: $category_text_en})
                    DETACH DELETE c
                    RETURN count(c) AS deleted_count
                """
                delete_params = {"document_id": document_id, "category_text_en": category_text_en}
                try:
                    _ = execute_neo4j_query(delete_query, delete_params)
                except Exception:
                    # If deletion fails, continue to upsert to avoid leaving state inconsistent
                    log(WARNING, f"Failed to delete existing Category nodes for document {document_id} and category {category_text_en}")

                # perform the upsert (create Category node and link to document)
                query = """
                    MATCH (d:Document)
                    WHERE d.id = $document_id OR d.documentid = $document_id
                    MATCH (kc:KeywordCategory {text_en: $category_text_en})
                    MERGE (kc)<-[:OF_CATEGORY]-(c:Category {text: $category_text})
                    ON CREATE SET c.lang = null
                    MERGE (d)-[:HAS_CATEGORY]->(c)
                    RETURN count(c) AS affected_count
                """
                params = {
                    "document_id": document_id,
                    "category_text_en": category_text_en,
                    "category_text": category_text
                }
                result = execute_neo4j_query(query, params)

                affected_count = 0
                if result:
                    affected_count = result[0].get("affected_count", 0) if result else 0
                log(INFO, f"[doc: {document_id}] Upserted category '{category_text_en}={category_text}'")
                return {"success": True, "document_id": document_id, "affected_count": affected_count}
            else:
                log(WARNING, f"Query result was empty for document {document_id}")
                return {"error": 3, "error_message": "Document not found or failed to upsert category"}
        except Exception as e:
            error_message = f"An error occurred while upserting category: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def upsert_document_categories_batch(self, document_ids: List[str], category_text_en: str, category_value: Optional[str] = None, max_workers: int = 4, effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Upsert (create or update) a category for multiple documents in parallel.
        
        Uses ThreadPoolExecutor to parallelize upsert_document_category calls across documents.
        
        Args:
            document_ids: List of document ids to update.
            category_text_en: The text_en value of the KeywordCategory.
            category_value: Optional value for the category.
            max_workers: Maximum number of parallel worker threads (default: 4).
        
        Returns:
            Dictionary with aggregated results.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        
        if not document_ids:
            return {
                "success": True,
                "success_count": 0,
                "total_count": 0,
                "affected_total": 0,
                "errors": []
            }
        
        results = []
        errors = []
        affected_total = 0
        success_count = 0
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all upsert tasks
                futures = {
                    executor.submit(self.upsert_document_category, doc_id, category_text_en, category_value, effective_user_id): doc_id
                    for doc_id in document_ids
                }
                
                # Collect results as they complete
                for future in as_completed(futures):
                    doc_id = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result.get('success'):
                            success_count += 1
                            affected_total += result.get('affected_count', 0)
                        else:
                            error_msg = result.get('error_message', 'Unknown error')
                            errors.append((doc_id, error_msg))
                    except Exception as e:
                        errors.append((doc_id, str(e)))
            
            log(INFO, f"Batch upsert completed: {success_count}/{len(document_ids)} documents, {affected_total} categories upserted")
            
            return {
                "success": success_count == len(document_ids),
                "success_count": success_count,
                "total_count": len(document_ids),
                "affected_total": affected_total,
                "errors": errors
            }
        except Exception as e:
            error_message = f"Error in batch upsert: {e}"
            log(ERROR, error_message)
            return {
                "success": False,
                "success_count": success_count,
                "total_count": len(document_ids),
                "affected_total": affected_total,
                "errors": [(None, error_message)]
            }

    def generate_new_document_id(self) -> str:
        """Generate a new document ID following the naming convention 'docid-nnnn'."""
        document_id: str = None

        try:
            # Use atomic counter approach similar to C# implementation
            query = """
                MERGE (idx:DocumentIdx {id: 'document_index'})
                ON CREATE SET idx.value = 0
                ON MATCH SET idx.value = idx.value + 1
                RETURN idx.value AS newIndex"""
            
            params = {}
            result = execute_neo4j_query(query, params)
            if len(result) > 0:
                new_index = result[0]["newIndex"]
                document_id = f"docid-{new_index}"
                log(DEBUG, f"Generated new document ID: {document_id}")
            else:
                raise Exception("Failed to generate new document index")
        
        except Exception as e:
            error_message = f"Error generating document ID: {e}"
            log(ERROR, f"ERROR: {error_message}")
            raise Exception(error_message)
        
        return document_id

    def validate_category_key(self, category_text_en: str) -> Dict[str, Any]:
        """Check whether a KeywordCategory with `text_en` exists and return its LoV if present.

        Returns: { 'exists': bool, 'lov': Optional[Union[str,List]] }
        """
        try:
            query = """
                MATCH (kc:KeywordCategory)
                WHERE kc.text_en = $category_text_en
                RETURN kc.lov AS lov
            """
            res = execute_neo4j_query(query, {"category_text_en": category_text_en}) or []
            if not res:
                return {"exists": False}
            # neo4j driver returns Record objects which support .get(); handle generically
            try:
                lov = res[0].get('lov')
            except Exception:
                try:
                    lov = res[0][0]
                except Exception:
                    lov = None
            log(DEBUG, f"validate_category_key(): text_en={category_text_en}, lov={lov}")
            return {"exists": True, "lov": lov}
        except Exception as e:
            log(ERROR, f"validate_category_key(): Error querying KeywordCategory: {e}")
            return {"exists": False}

    def validate_category_value(self, category_text_en: str, category_value: str) -> Dict[str, Any]:
        """Validate a provided category value against KeywordCategory.lov.

        If the KeywordCategory defines no LoV, any value is valid.
        Returns: { 'valid': bool, 'allowed': Optional[list] }
        """
        try:
            check = self.validate_category_key(category_text_en)
            if not check.get('exists'):
                return {"valid": False, "allowed": []}

            lov_val = check.get('lov')
            log(DEBUG, f"validate_category_value(): text_en={category_text_en}, provided_value={category_value}, lov_val={lov_val}")
            if lov_val is None:
                return {"valid": True, "allowed": None}

            allowed: list = []
            if isinstance(lov_val, (list, tuple)):
                allowed = [str(v).strip() for v in lov_val if v is not None]
            else:
                allowed = [v.strip() for v in str(lov_val).split(',') if v.strip()]

            # DEBUG: print allowed values and membership
            # debug printing removed

            valid = (len(allowed) == 0) or (category_value in allowed)
            return {"valid": valid, "allowed": allowed}
        except Exception as e:
            log(ERROR, f"validate_category_value(): Error validating value: {e}")
            return {"valid": False, "allowed": []}

    def get_document_id_by(self, url: Optional[str]) -> Union[str, None]:
        """
        Retrieve a document ID by its URL.
        Args:
            url: The URL of the document.
        Returns:
            Document ID as a string or None if not found.
        """
        if url is None or len(url.strip()) == 0:
            return None

        

        document_id = None
        try:
            query = """
                MATCH (d:Document)
                WHERE d.url = $url
                RETURN d.id AS id, d.documentid AS documentid
            """
            parameters = {"url": url}
            result = execute_neo4j_query(query, parameters)

            if result and len(result) > 0:
                doc_record = result[0]
                document_id = doc_record.get("id") or doc_record.get("documentid")

                if len(result) == 1:
                    log(DEBUG, f"Found document ID by URL: {document_id}")
                else:
                    log(WARNING, f"Multiple documents found for URL: {url}. Returning the first one.")
            else:
                log(DEBUG, f"This document was not yet downloaded: {url}")

        except Exception as e:
            error_message = f"An error occurred while retrieving document ID by URL: {e}"
            log(ERROR, f"ERROR: {error_message}")

        return document_id

    def create_document(self, document_data: Dict[str, Any], effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new document."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        try:
            document_id = document_data.get("id", None) or document_data.get("documentid", None) or document_data.get("document_id", None)
            if not document_id:
                document_id = self.generate_new_document_id()

            partition = _partition_letter_from_hash(document_id)

            query = """
                MERGE (d:Document {id: $document_id})
                SET d.documentid = $document_id,
                    d.name = $name,
                    d.title = $title,
                    d.url = COALESCE($url, d.url),
                    d.text = $text,
                    d.enhanced_text = $enhanced_text,
                    d.type = $type,
                    d.status = COALESCE($status, $default_status),
                    d.knowledge_graph_status = COALESCE($knowledge_graph_status, $default_knowledge_graph_status),
                    d.scanned = COALESCE($scanned, false),
                    d.extraction_quality = $extraction_quality,
                    d.pages = $pages,
                    d.label = $label,
                    d.partition = $partition,
                    d.uploaded_datetime = COALESCE($uploaded_datetime, datetime()),
                    d.processed_datetime = $processed_datetime
                RETURN d.id AS id
            """
            
            parameters = {
                "document_id": document_id,
                "name": document_data.get("name", ""),
                "title": document_data.get("title"),
                "url": document_data.get("url"),
                "text": document_data.get("text"),
                "enhanced_text": document_data.get("enhanced_text"),
                "type": document_data.get("type"),
                "status": document_data.get("status"),
                "default_status": DocumentStatus.NOT_YET_PROCESSED.value,
                "knowledge_graph_status": document_data.get("knowledge_graph_status"),
                "default_knowledge_graph_status": KnowledgeGraphStatus.NOT_YET_PROCESSED.value,
                "scanned": document_data.get("scanned"),
                "extraction_quality": document_data.get("extraction_quality"),
                "pages": document_data.get("pages"),
                "label": document_data.get("label"),
                "partition": partition,
                "uploaded_datetime": document_data.get("uploaded_datetime"),
                "processed_datetime": document_data.get("processed_datetime")
            }

            result = execute_neo4j_query(query, parameters)

            if result:
                log(INFO, f"[doc: {document_id}] Created successfully")
                return {"success": True, "document_id": document_id}
            else:
                return {"error": 3, "error_message": "Failed to create document"}

        except Exception as e:
            error_message = f"An error occurred while creating document: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def create_or_update_document_by_url(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update a document node by `url` using MERGE on the url.

        This method ensures only one `Document` node exists per distinct `url`.
        If a document with the url exists, it will be updated with provided fields;
        otherwise a new node will be created. This prevents race-created duplicates
        when many workers attempt to ingest the same URL concurrently.
        """
        try:
            url = document_data.get("url")
            if not url:
                return {"error": 1, "error_message": "No URL provided for create_or_update_document_by_url"}

            document_id = document_data.get("id") or document_data.get("documentid") or self.generate_new_document_id()
            partition = _partition_letter_from_hash(document_id)

            query = """
                MERGE (d:Document {url: $url})
                SET d.documentid = COALESCE(d.documentid, $document_id),
                    d.id = COALESCE(d.id, $document_id),
                    d.name = $name,
                    d.title = $title,
                    d.text = $text,
                    d.enhanced_text = $enhanced_text,
                    d.type = $type,
                    d.status = COALESCE($status, d.status, $default_status),
                    d.knowledge_graph_status = COALESCE($knowledge_graph_status, d.knowledge_graph_status, $default_knowledge_graph_status),
                    d.scanned = COALESCE($scanned, d.scanned, false),
                    d.extraction_quality = $extraction_quality,
                    d.pages = $pages,
                    d.label = $label,
                    d.partition = COALESCE(d.partition, $partition),
                    d.uploaded_datetime = COALESCE($uploaded_datetime, d.uploaded_datetime, datetime()),
                    d.processed_datetime = $processed_datetime
                RETURN d.id AS id
            """

            parameters = {
                "document_id": document_id,
                "name": document_data.get("name", ""),
                "title": document_data.get("title"),
                "url": url,
                "text": document_data.get("text"),
                "enhanced_text": document_data.get("enhanced_text"),
                "type": document_data.get("type"),
                "status": document_data.get("status"),
                "default_status": DocumentStatus.NOT_YET_PROCESSED.value,
                "knowledge_graph_status": document_data.get("knowledge_graph_status"),
                "default_knowledge_graph_status": KnowledgeGraphStatus.NOT_YET_PROCESSED.value,
                "scanned": document_data.get("scanned"),
                "extraction_quality": document_data.get("extraction_quality"),
                "pages": document_data.get("pages"),
                "label": document_data.get("label"),
                "partition": partition,
                "uploaded_datetime": document_data.get("uploaded_datetime"),
                "processed_datetime": document_data.get("processed_datetime")
            }

            result = execute_neo4j_query(query, parameters)
            if result:
                return {"success": True, "document_id": document_id}
            else:
                return {"error": 3, "error_message": "Failed to create_or_update document by url"}

        except Exception as e:
            error_message = f"An error occurred while creating/updating document by url: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def update_document(self, document_id: str, document_data: Dict[str, Any], effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Update an existing document."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_document_permission(effective_user_id, document_id, "EDIT"):
            log(WARNING, f"DocumentsService.update_document: EDIT access denied for effective user '{effective_user_id}' on document '{document_id}'")
            return {"error": 1, "error_message": "EDIT permission denied"}
        try:
            # Build SET clause dynamically based on provided data
            set_clauses = []
            parameters = {"document_id": document_id}
            
            updatable_fields = ["name", "title", "text", "enhanced_text", "type", "status", 
                              "scanned", "extraction_quality", "pages", "processed_datetime", "url", "label"]
            
            for field in updatable_fields:
                if field in document_data:
                    set_clauses.append(f"d.{field} = ${field}")
                    parameters[field] = document_data[field]
            
            if not set_clauses:
                return {"error": 1, "error_message": "No valid fields to update"}

            query = f"""
                MATCH (d:Document)
                WHERE d.id = $document_id OR d.documentid = $document_id
                SET {', '.join(set_clauses)}
                RETURN d.id AS id
            """

            result = execute_neo4j_query(query, parameters)
            
            if result:
                log(INFO, f"[doc: {document_id}] Updated successfully")
                return {"success": True, "document_id": document_id}
            else:
                return {"error": 3, "error_message": "Document not found or update failed"}

        except Exception as e:
            error_message = f"An error occurred while updating document: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def delete_document(self, document_id: str, effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Delete a document and its related nodes (chapters, paragraphs, etc.)."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_document_permission(effective_user_id, document_id, "DELETE"):
            log(WARNING, f"DocumentsService.delete_document: DELETE access denied for effective user '{effective_user_id}' on document '{document_id}'")
            return {"error": 1, "error_message": "DELETE permission denied"}
        try:
            # Drop knowledge-graph links to this document before the cascade below. DETACH DELETE would
            # remove them as a side effect, but a curated link that vanishes unaccounted is a silent
            # data-loss path, so the deletion is made explicit.
            query = """
                MATCH (:Entity)-[r:IS_DOCUMENTED]->(d:Document)
                WHERE d.id = $document_id OR d.documentid = $document_id
                DELETE r
            """
            execute_neo4j_query(query, {"document_id": document_id})

            query = """
                MATCH (d:Document)
                WHERE d.id = $document_id OR d.documentid = $document_id
                OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chapter)
                OPTIONAL MATCH (c)<-[:PART_OF]-(p:Paragraph)
                OPTIONAL MATCH (c)<-[:HAS_IMAGE]-(img:Image)
                OPTIONAL MATCH (d)-[:HAS_SUMMARY]->(s:Summary)
                DETACH DELETE d, c, p, img, s
                RETURN count(d) AS deleted_count
            """
            
            parameters = {"document_id": document_id}
            result = execute_neo4j_query(query, parameters)
            
            if result and result[0].get("deleted_count", 0) > 0:
                log(INFO, f"[doc: {document_id}] Deleted successfully")
                return {"success": True, "document_id": document_id}
            else:
                return {"error": 3, "error_message": "Document not found"}

        except Exception as e:
            error_message = f"An error occurred while deleting document: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def get_documents_by_status(self, status: DocumentStatus, merge_clause: Optional[str] = None, limit: Optional[int] = None, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve documents filtered by status.
        
        Args:
            status: The DocumentStatus to filter by.
            merge_clause: Optional Cypher WHERE clause for additional filters. Defaults to None.
            limit: Optional maximum number of documents to return. Defaults to None (no limit).
            effective_user_id: If provided, only documents accessible by this effective user are returned. Defaults to None.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        result = []
        status = status.value

        acl_clause = acl_document_where_clause("d") if effective_user_id else ""
        query = f"""
            MATCH (d:Document)
            WHERE d.status = $status{acl_clause}
        """
        if merge_clause:
            query += f"AND {merge_clause}\n"
        
        query += """
            OPTIONAL MATCH (d)-[:HAS_SUMMARY]->(s:Summary)
            RETURN
                d.id AS id,
                d.documentid AS documentid,
                d.name AS name,
                d.title AS title,
                d.status AS status,
                d.type AS type,
                d.extraction_quality AS extraction_quality,
                d.uploaded_datetime AS uploaded_datetime,
                d.processed_datetime AS processed_datetime,
                s.text AS summary
        """
        if limit is not None:
            query += f"LIMIT {limit}\n"
        
        try:
            parameters = {"status": status}
            if effective_user_id:
                parameters["effectiveUserId"] = effective_user_id
            documents = execute_neo4j_query(query, parameters)
            if not documents:
                return []
            
            for doc in documents:
                document = {
                    "id": doc.get("id") or doc.get("documentid"),
                    "documentid": doc.get("documentid") or doc.get("id"),
                    "status": doc.get("status")
                }
                
                # Add optional fields if they exist
                optional_fields = ["name", "title", "type", "extraction_quality", 
                                 "uploaded_datetime", "processed_datetime", "summary"]
                
                for field in optional_fields:
                    if doc.get(field) is not None:
                        document[field] = doc.get(field)
                
                result.append(document)
                
        except Exception as e:
            error_message = f"An error occurred: {e}"
            log(ERROR, f"ERROR: {error_message}")
            result = [{"error": 2, "error_message": error_message}]

        return result

    def get_documents_by_type(self, document_type: str, merge_clause: Optional[str] = None, limit: Optional[int] = None, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve documents filtered by type. When effective_user_id is provided, only accessible documents are returned.
        
        Args:
            document_type: The document type to filter by.
            merge_clause: Optional Cypher WHERE clause for additional filters. Defaults to None.
            limit: Optional maximum number of documents to return. Defaults to None (no limit).
            effective_user_id: Optional ID of the effective user requesting the documents. Defaults to None.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        result = []

        acl_clause = acl_document_where_clause("d") if effective_user_id else ""
        query = f"""
            MATCH (d:Document)
            WHERE d.type = $document_type{acl_clause}
        """
        if merge_clause:
            query += f"AND {merge_clause}\n"
        
        query += """
            OPTIONAL MATCH (d)-[:HAS_SUMMARY]->(s:Summary)
            RETURN
                d.id AS id,
                d.documentid AS documentid,
                d.name AS name,
                d.title AS title,
                d.status AS status,
                d.type AS type,
                d.extraction_quality AS extraction_quality,
                d.uploaded_datetime AS uploaded_datetime,
                d.processed_datetime AS processed_datetime,
                s.text AS summary
        """
        if limit is not None:
            query += f"LIMIT {limit}\n"
        
        try:
            parameters = {"document_type": document_type}
            if effective_user_id:
                parameters["effectiveUserId"] = effective_user_id
            documents = execute_neo4j_query(query, parameters)
            if not documents:
                return []
            
            for doc in documents:
                document = {
                    "id": doc.get("id") or doc.get("documentid"),
                    "documentid": doc.get("documentid") or doc.get("id"),
                    "type": doc.get("type")
                }
                
                # Add optional fields if they exist
                optional_fields = ["name", "title", "status", "extraction_quality", 
                                 "uploaded_datetime", "processed_datetime", "summary"]
                
                for field in optional_fields:
                    if doc.get(field) is not None:
                        document[field] = doc.get(field)
                
                result.append(document)
                
        except Exception as e:
            error_message = f"An error occurred: {e}"
            log(ERROR, f"ERROR: {error_message}")
            result = [{"error": 2, "error_message": error_message}]

        return result
    
    def get_documents_count(self, dossier_ids: Optional[List[str]] = None, merge_clause: Optional[str] = None, limit: Optional[int] = None, effective_user_id: Optional[str] = None) -> int:
        """Return total number of Document nodes in the database.

        If `dossier_ids` is None, count all documents. If an empty list is
        provided, return 0. If a non-empty list is provided, count documents
        owned by any dossier whose `id` is in the list.
        
        When effective_user_id is provided, only documents the effective user has READ permission on are counted.
        
        Args:
            dossier_ids: Optional list of dossier IDs to filter by.
            merge_clause: Optional Cypher WHERE clause for additional filters. Defaults to None.
            limit: Optional maximum number of documents to count. Defaults to None (no limit).
            effective_user_id: Optional. If provided, applies ACL filtering to only count accessible documents.
        """
        documents_count: int = 0

        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()

        try:
            acl_clause = acl_document_where_clause("d") if effective_user_id else ""
            
            # None => all documents
            if dossier_ids is None:
                query = """
                    MATCH (d:Document)
                    WHERE true{acl_clause}
                """.replace("{acl_clause}", acl_clause)
                if merge_clause:
                    query += f"AND {merge_clause}\n"
                query += "RETURN count(d) AS count"
                if limit is not None:
                    query += f" LIMIT {limit}"
                params = {"effectiveUserId": effective_user_id} if effective_user_id else {}

            # Empty list => explicitly return 0
            elif isinstance(dossier_ids, list) and len(dossier_ids) == 0:
                return 0

            # Non-empty list => count documents owned by any of the dossiers
            else:
                query = """
                    MATCH (dos:Dossier)-[:CONTAINS]->(d:Document)
                    WHERE dos.id IN $dossier_ids{acl_clause}
                """.replace("{acl_clause}", acl_clause)
                if merge_clause:
                    query += f"AND {merge_clause}\n"
                query += "RETURN count(DISTINCT d) AS count"
                if limit is not None:
                    query += f" LIMIT {limit}"
                params = {"dossier_ids": dossier_ids}
                if effective_user_id:
                    params["effectiveUserId"] = effective_user_id

            result = execute_neo4j_query(query, params)
            if result and len(result) > 0:
                row = result[0]
                val = row.get("count")
                documents_count = int(val)

        except Exception as e:
            error_message = f"An error occurred while counting documents: {e}"
            log(ERROR, f"ERROR: {error_message}")

        return documents_count
 
    def update_document_status(self, document_id: str, status: DocumentStatus, effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Update the status of a document."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_document_permission(effective_user_id, document_id, "EDIT"):
            log(WARNING, f"DocumentsService.update_document_status: EDIT access denied for effective user '{effective_user_id}' on document '{document_id}'")
            return {"error": 1, "error_message": "EDIT permission denied"}
        try:
            status = status.value
            query = """
                MATCH (d:Document)
                WHERE d.id = $document_id OR d.documentid = $document_id
                SET d.status = $status,
                    d.partition = $partition,
                    d.processed_datetime = datetime()
                RETURN d.id AS id
            """
            
            parameters = {
                "document_id": document_id,
                "status": status,
                "partition": _partition_letter_from_hash(document_id)
            }
            result = execute_neo4j_query(query, parameters)
            
            if result:
                log(INFO, f"[doc: {document_id}] Status updated to: {status}")
                return {"success": True, "document_id": document_id, "status": status}
            else:
                return {"error": 3, "error_message": "Document not found"}

        except Exception as e:
            error_message = f"An error occurred while updating document status: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def update_knowledge_graph_status(self, document_id: str, knowledge_graph_status: KnowledgeGraphStatus) -> Dict[str, Any]:
        """Update the knowledge graph status of a document."""
        try:
            status = knowledge_graph_status.value
            query = """
                MATCH (d:Document)
                WHERE d.id = $document_id OR d.documentid = $document_id
                SET d.knowledge_graph_status = $knowledge_graph_status
                RETURN d.id AS id
            """

            parameters = {"document_id": document_id, "knowledge_graph_status": status}
            result = execute_neo4j_query(query, parameters)

            if result:
                log(INFO, f"[doc: {document_id}] Knowledge graph status updated to: {status}")
                return {"success": True, "document_id": document_id, "knowledge_graph_status": status}
            else:
                return {"error": 3, "error_message": "Document not found"}

        except Exception as e:
            error_message = f"An error occurred while updating document knowledge graph status: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def write_document_file(self, document_id: str, document_name: str, document_text: str, effective_user_id: Optional[str] = None) -> Optional[str]:
        """Write the physical file for a document to the document repository."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_document_permission(effective_user_id, document_id, "EDIT"):
            log(WARNING, f"DocumentsService.write_document_file: EDIT access denied for effective user '{effective_user_id}' on document '{document_id}'")
            return None
        try:
            repo_path = get_document_path()
            filename = document_name.strip()
            if not filename.endswith('.txt'):
                filename += '.txt'
            file_path = os.path.join(repo_path, filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(document_text or "")
            log(INFO, f"[doc: {document_id}] File written to: {file_path}")
            return file_path
        except Exception as e:
            log(ERROR, f"write_document_file(): failed to write file for document '{document_id}': {e}")
            return None

    # Maps document type attribute values to file extensions for filesystem-writable types.
    _WRITABLE_TYPE_TO_EXT = {
        "text": ".txt",
        "txt": ".txt",
        "md": ".md",
        "MD": ".md",
        "markdown": ".md",
        "html": ".html",
        "Html": ".html",
        "csv": ".csv",
        "Csv": ".csv",
    }

    def update_document_file(self, document_id: str, effective_user_id: Optional[str] = None) -> Optional[str]:
        """Overwrite the physical file for a document using its current Neo4j content."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_document_permission(effective_user_id, document_id, "EDIT"):
            log(WARNING, f"DocumentsService.update_document_file: EDIT access denied for effective user '{effective_user_id}' on document '{document_id}'")
            return None
        try:
            from src.utils.text_utils import generate_text_from_html
            query = "MATCH (d:Document) WHERE d.id = $document_id OR d.documentid = $document_id RETURN d.name AS name, d.type AS type, d.text AS text, d.enhanced_text AS enhanced_text LIMIT 1"
            records = execute_neo4j_query(query, {"document_id": document_id})
            if not records:
                log(ERROR, f"update_document_file(): document '{document_id}' not found.")
                return None
            record = records[0]
            doc_type = (record.get("type") or "").strip()
            ext = self._WRITABLE_TYPE_TO_EXT.get(doc_type) or self._WRITABLE_TYPE_TO_EXT.get(doc_type.lower())
            if ext is None:
                log(DEBUG, f"update_document_file(): document '{document_id}' has unsupported type '{doc_type}', skipping filesystem write.")
                return None
            doc_name = (record.get("name") or document_id).strip()
            # Ensure the filename has the correct extension
            name_lower = doc_name.lower()
            if not any(name_lower.endswith(e) for e in self._WRITABLE_TYPE_TO_EXT.values()):
                doc_name = doc_name + ext
            text = record.get("text")
            enhanced_text = record.get("enhanced_text")
            doc_type_lower = doc_type.lower()
            if doc_type_lower == "html":
                # For html: write enhanced_text (the HTML-edited content)
                content = enhanced_text or text or ""
            else:
                # For txt, md, csv: text is already back-converted to the correct format
                # after any edit via save_content; fall back to stripping HTML as safety net
                if text:
                    content = text
                elif enhanced_text:
                    content = generate_text_from_html(enhanced_text)
                else:
                    content = ""
            repo_path = get_document_path()
            file_path = os.path.join(repo_path, doc_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            log(INFO, f"[doc: {document_id}] File overwritten at: {file_path}")
            return file_path
        except Exception as e:
            log(ERROR, f"update_document_file(): failed to overwrite file for document '{document_id}': {e}")
            return None

    def get_document_id_by_name(self, doc_name: str, effective_user_id: Optional[str] = None) -> Optional[str]:
        """Retrieve a document ID by its name property.
        
        When effective_user_id is provided, READ permission on the document is verified.
        Returns None if document not found or access denied.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        try:
            records = execute_neo4j_query("MATCH (d:Document) WHERE d.name=$doc_name RETURN d.id AS id, d.documentid AS documentid", {"doc_name": doc_name})
            if not records:
                return None
            
            doc_id = records[0].get("id") or records[0].get("documentid")
            
            # If effective_user_id provided, verify READ access to the document
            if effective_user_id and doc_id:
                doc = self.get_document(doc_id, effective_user_id=effective_user_id)
                if doc is None:
                    log(WARNING, f"DocumentsService.get_document_id_by_name: READ access denied for effective user '{effective_user_id}' on document '{doc_id}'")
                    return None
            
            return doc_id
        except Exception as e:
            log(ERROR, f"get_document_id_by_name(): error: {e}")
            return None

    def get_chapter_for_editing(self, chapter_id: str, effective_user_id: Optional[str] = None) -> Optional[dict]:
        """Retrieve a chapter's editable content fields.
        
        When effective_user_id is provided, READ permission on the parent document is verified.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        try:
            # First retrieve the chapter to find its parent document
            chapter_query = "MATCH (c:Chapter {id: $chapter_id})-[:PART_OF]->(d:Document) RETURN c.id AS chapter_id, d.id AS document_id, d.documentid AS documentid"
            chapter_records = execute_neo4j_query(chapter_query, {"chapter_id": chapter_id})
            if not chapter_records:
                return None
            
            parent_doc_id = chapter_records[0].get("document_id") or chapter_records[0].get("documentid")
            
            # If effective_user_id provided, verify READ access to the parent document
            if effective_user_id and parent_doc_id:
                doc = self.get_document(parent_doc_id, effective_user_id=effective_user_id)
                if doc is None:
                    log(WARNING, f"DocumentsService.get_chapter_for_editing: READ access denied for effective user '{effective_user_id}' on document '{parent_doc_id}' (parent of chapter '{chapter_id}')")
                    return None
            
            records = execute_neo4j_query("MATCH (c:Chapter) WHERE c.id=$chapter_id OPTIONAL MATCH (c)-[:PART_OF]->(d:Document) RETURN c.text AS text, c.enhanced_text AS enhanced_text, c.enhanced_type AS enhanced_type, d.type AS doc_type", {"chapter_id": chapter_id})
            if not records:
                return None
            r = records[0]
            return {"text": r.get("text"), "enhanced_text": r.get("enhanced_text"), "enhanced_type": r.get("enhanced_type"), "doc_type": r.get("doc_type")}
        except Exception as e:
            log(ERROR, f"get_chapter_for_editing(): error: {e}")
            return {"error": 2, "error_message": str(e)}

    def get_document_for_editing(self, doc_id: str, effective_user_id: Optional[str] = None) -> Optional[dict]:
        """Retrieve a document's editable content fields.
        
        When effective_user_id is provided, READ permission on the document is verified.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        # If effective_user_id provided, verify READ access to the document first
        if effective_user_id:
            doc = self.get_document(doc_id, effective_user_id=effective_user_id)
            if doc is None:
                log(WARNING, f"DocumentsService.get_document_for_editing: READ access denied for effective user '{effective_user_id}' on document '{doc_id}'")
                return None

        try:
            records = execute_neo4j_query("MATCH (d:Document) WHERE d.id=$doc_id OR d.documentid=$doc_id RETURN d.text AS text, d.title AS title, d.enhanced_text AS enhanced_text, d.enhanced_type AS enhanced_type, d.type AS doc_type", {"doc_id": doc_id})
            if not records:
                return None
            r = records[0]
            return {"text": r.get("text"), "title": r.get("title"), "enhanced_text": r.get("enhanced_text"), "enhanced_type": r.get("enhanced_type"), "doc_type": r.get("doc_type")}
        except Exception as e:
            log(ERROR, f"get_document_for_editing(): error: {e}")
            return {"error": 2, "error_message": str(e)}

    def get_chapter_ids_for_document(self, doc_id: str, effective_user_id: Optional[str] = None) -> list:
        """Retrieve IDs of all chapters belonging to a document.
        
        When effective_user_id is provided, READ permission on the document is verified.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        # If effective_user_id provided, verify READ access to the document first
        if effective_user_id:
            doc = self.get_document(doc_id, effective_user_id=effective_user_id)
            if doc is None:
                log(WARNING, f"DocumentsService.get_chapter_ids_for_document: READ access denied for effective user '{effective_user_id}' on document '{doc_id}'")
                return []

        try:
            records = execute_neo4j_query("MATCH (c:Chapter)-[:PART_OF]->(d:Document) WHERE d.id=$doc_id OR d.documentid=$doc_id RETURN c.id AS chapter_id", {"doc_id": doc_id})
            return [r["chapter_id"] for r in records]
        except Exception as e:
            log(ERROR, f"get_chapter_ids_for_document(): error: {e}")
            return []

    def save_enhanced_content(self, content_id: str, node_label: str, enhanced_text: str, enhanced_type: str, doc_type: str, original_text: str, tx: str, effective_user_id: Optional[str] = None) -> None:
        """Persist edited enhanced content, creating a version snapshot when tx has changed."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        try:
            from src.utils.text_utils import generate_text_from_html, generate_markdown_from_html
            query_get = f"MATCH (n:{node_label} {{id: $content_id}}) RETURN n.enhanced_text AS enhanced_text, n.enhanced_type AS enhanced_type, n.enhanced_modified AS enhanced_modified, n.enhanced_tx AS enhanced_tx, n.documentid AS documentid"
            current = execute_neo4j_query(query_get, {"content_id": content_id})
            current_tx = current[0].get("enhanced_tx") if current else None
            if current_tx is not None and tx != current_tx:
                node_label_version = f"{node_label}Version"
                query_version = f"CREATE (n:{node_label_version}) SET n.id=$content_id, n.documentid=$document_id, n.text=$text, n.enhanced_text=$enhanced_text, n.enhanced_type=$enhanced_type, n.enhanced_modified=$enhanced_modified, n.enhanced_tx=$tx WITH n MATCH (m:{node_label} {{id: $content_id}}) MERGE (m)-[:HAS_VERSION]->(n)"
                log(DEBUG, f"save_enhanced_content(): saving version of {node_label} '{content_id}'...")
                execute_neo4j_query(query_version, {"content_id": content_id, "document_id": current[0].get("documentid"), "text": current[0].get("enhanced_text"), "enhanced_text": current[0].get("enhanced_text"), "enhanced_type": current[0].get("enhanced_type"), "enhanced_modified": current[0].get("enhanced_modified"), "tx": current_tx})
                log(DEBUG, f"...saved.")
            doc_type_lower = (doc_type or "").lower().strip()
            if doc_type_lower in ("html",):
                plain_text = enhanced_text or ""
            elif doc_type_lower in ("md", "markdown"):
                plain_text = generate_markdown_from_html(enhanced_text) if enhanced_text else (original_text or "")
            elif doc_type_lower in ("text", "txt", "csv"):
                plain_text = generate_text_from_html(enhanced_text) if enhanced_text else (original_text or "")
            else:
                plain_text = original_text
            query_save = f"MATCH (n:{node_label} {{id: $content_id}}) SET n.text=$text, n.enhanced_text=$enhanced_text, n.enhanced_type=$enhanced_type, n.enhanced_modified=datetime({{timezone: 'UTC'}}), n.enhanced_tx=$tx"
            log(DEBUG, f"save_enhanced_content(): saving {node_label} '{content_id}'...")
            execute_neo4j_query(query_save, {"content_id": content_id, "text": plain_text, "enhanced_text": enhanced_text, "enhanced_type": enhanced_type, "tx": tx})
            log(DEBUG, f"...saved.")
        except Exception as e:
            log(ERROR, f"save_enhanced_content(): error: {e}")

    def delete_content_versions(self, content_id: str, node_label: str, effective_user_id: Optional[str] = None) -> bool:
        """Permanently delete all version snapshots for a Document or Chapter node."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        try:
            log(INFO, f"delete_content_versions(): deleting all versions of {node_label} '{content_id}'...")
            execute_neo4j_query(f"MATCH (n:{node_label}Version) WHERE n.id=$content_id DETACH DELETE n", {"content_id": content_id})
            log(DEBUG, f"...deleted.")
            return True
        except Exception as e:
            log(ERROR, f"delete_content_versions(): error: {e}")
            return False

    def find_ontology_for_document(self, document_id: str, effective_user_id: Optional[str] = None) -> Optional[str]:
        """Find the ontology ID for a document by traversing its owning dossier.
        
        When effective_user_id is provided, READ permission on the document is verified.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        # If effective_user_id provided, verify READ access to the document first
        if effective_user_id:
            doc = self.get_document(document_id, effective_user_id=effective_user_id)
            if doc is None:
                log(WARNING, f"DocumentsService.find_ontology_for_document: READ access denied for effective user '{effective_user_id}' on document '{document_id}'")
                return None

        try:
            query = """
                MATCH (dos:Dossier)-[:CONTAINS]->(doc:Document)
                WHERE doc.id = $document_id OR doc.documentid = $document_id
                OPTIONAL MATCH (dos)-[:FOLLOWS]->(o:Ontology)
                RETURN o.id AS ontology_id
                LIMIT 1
            """
            records = execute_neo4j_query(query, {"document_id": document_id})
            if records and len(records) > 0:
                ontology_id = records[0].get("ontology_id")
                if ontology_id:
                    log(DEBUG, f"find_ontology_for_document(): document '{document_id}' -> ontology '{ontology_id}'")
                    return ontology_id
            log(DEBUG, f"find_ontology_for_document(): no ontology found for document '{document_id}'")
            return None
        
        except Exception as e:
            log(ERROR, f"find_ontology_for_document(): error: {e}")
            return None

    def link_document_to_ontology(self, document_id: str, ontology_id: str) -> Dict[str, Any]:
        """Create a FOLLOWS relationship from a Document node to an Ontology node."""
        result: Dict[str, Any] = {}

        try:
            query = """
                MATCH (d:Document)
                WHERE d.id = $document_id OR d.documentid = $document_id
                MERGE (o:Ontology {id: $ontology_id})
                MERGE (d)-[:FOLLOWS]->(o)
                RETURN d.id AS document_id
            """
            records = execute_neo4j_query(query, {"document_id": document_id, "ontology_id": ontology_id})
            if records:
                log(INFO, f"link_document_to_ontology(): document '{document_id}' now FOLLOWS ontology '{ontology_id}'")
                result = {"success": True, "document_id": document_id, "ontology_id": ontology_id}
            else:
                result = {"error": 3, "error_message": f"Document '{document_id}' not found"}

        except Exception as e:
            log(ERROR, f"link_document_to_ontology(): error: {e}")
            result = {"error": 2, "error_message": str(e)}

        return result

    def update_document_configuration(self, document_id: str, configuration_id: str, effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Update the document configuration."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_document_permission(effective_user_id, document_id, "EDIT"):
            log(WARNING, f"DocumentsService.update_document_configuration: EDIT access denied for effective user '{effective_user_id}' on document '{document_id}'")
            return {"error": 1, "error_message": "EDIT permission denied"}
        try:
            # First, remove any existing configuration relationship
            query_remove = """
                MATCH (d:Document)-[r:HAS_CONFIGURATION]->(config)
                WHERE d.id = $document_id OR d.documentid = $document_id
                DELETE r
            """
            execute_neo4j_query(query_remove, {"document_id": document_id})
            
            # Then, create new configuration relationship if configuration exists
            query_create = """
                MATCH (d:Document), (config:DocumentConfiguration)
                WHERE (d.id = $document_id OR d.documentid = $document_id) AND config.id = $configuration_id
                MERGE (d)-[:HAS_CONFIGURATION]->(config)
                RETURN d.id AS document_id, config.id AS configuration_id
            """
            
            parameters = {"document_id": document_id, "configuration_id": configuration_id}
            result = execute_neo4j_query(query_create, parameters)
            
            if result:
                log(INFO, f"[doc: {document_id}] Configuration updated to: {configuration_id}")
                return {"success": True, "document_id": document_id, "configuration_id": configuration_id}
            else:
                return {"error": 3, "error_message": "Document or configuration not found"}

        except Exception as e:
            error_message = f"An error occurred while updating document configuration: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def update_document_categories(self, document_id: str, categories: Dict[str, Optional[str]], ontology_id: Optional[str] = None, effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Update the categories associated with a document.
        Args:
            document_id: The ID of the document to update.
            categories: Dictionary of category names to values, e.g.:
                {
                    "COUNTRY": "IQ",    # Iraq
                    "ASYLUM": None      # Asylum category with no specific value
                }
            ontology_id: Optional. The ID of the ontology that contains these categories. If provided,
                creates a Document -[:FOLLOWS]-> Ontology -[:CONTAINS]-> KeywordCategory path.
        Preconditions:
            The categories must already exist as KeywordCategory nodes in the database. For example:
            MERGE (kc:KeywordCategory {id: 'COUNTRY'}) SET kc.text_en = 'COUNTRY', kc.description = 'Country category as per UN codes (2-letter codes)'
            MERGE (kc:KeywordCategory {id: 'ASYLUM'}) SET kc.text_en = 'ASYLUM', kc.description = 'Asylum category'
        Results:
            A new or updated Category node in neo4j for each category assignment, linked to the document and to its category definition. 
            Each Category is unique per (text, category) combination via the relationship pattern.
            Example:
                (d:Document)-[:HAS_CATEGORY]->(c:Category {text: 'IQ'})-[:OF_CATEGORY]->(kc:KeywordCategory {text_en: 'COUNTRY'})
                (d:Document)-[:HAS_CATEGORY]->(c:Category {text: 'ASYLUM'})-[:OF_CATEGORY]->(kc:KeywordCategory {text_en: 'ASYLUM'})
            For binary categories (case a), the category text equals the category name.
            For value-based categories (cases b & c), the category text contains the specific value.
            In addition (if ontology_id provided):
                (d:Document)-[:FOLLOWS]->(o:Ontology)-[:CONTAINS]->(kc:KeywordCategory)
        Returns:
            Success/error message dictionary.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_document_permission(effective_user_id, document_id, "EDIT"):
            log(WARNING, f"DocumentsService.update_document_categories: EDIT access denied for effective user '{effective_user_id}' on document '{document_id}'")
            return {"error": 1, "error_message": "EDIT permission denied"}
        result: Dict[str, Any] = None

        try:
            if not categories:
                log(INFO, f"[doc: {document_id}] No categories provided")
                return {"success": True, "document_id": document_id, "categories_count": 0}

            # Convert categories dict to a list for UNWIND and generate embeddings for each keyword
            category_items: List[Dict[str, Any]] = []
            
            for category_id, category_value in categories.items():
                # Use 'category: value' if value exists, else just 'category' for embedding
                keyword_text: str = str(category_value).strip() if category_value else str(category_id).strip()

                category_items.append({
                    "category_id": str(category_id).strip(),
                    "keyword_text": keyword_text
                })

            # Cypher query to create/update category nodes linked to category definitions
            # Optionally create Document -[:FOLLOWS]-> Ontology relationship if ontology_id provided
            query: str = """
                MATCH (d:Document)
                WHERE d.id = $document_id OR d.documentid = $document_id
                WITH d
            """
            
            if ontology_id:
                query += """
                MERGE (o:Ontology {id: $ontology_id})
                MERGE (d)-[:FOLLOWS]->(o)
                WITH d, o
                """
            
            query += """
                UNWIND $category_items AS item
                MATCH (kc:KeywordCategory {text_en: item.category_id})
                MERGE (kc)<-[:OF_CATEGORY]-(c:Category {text: item.keyword_text})
                ON CREATE SET c.lang = null
                ON MATCH SET c.lang = null
                WITH d, c
                MERGE (d)-[:HAS_CATEGORY]->(c)
                RETURN count(c) AS category_count
            """

            parameters: Dict[str, Any] = {
                "document_id": document_id,
                "category_items": category_items
            }
            
            if ontology_id:
                parameters["ontology_id"] = ontology_id

            result = execute_neo4j_query(query, parameters)

            if result:
                category_count: int = result[0].get("category_count", 0) if result else 0
                log(INFO, f"[doc: {document_id}] Updated {category_count} categories" + 
                    (f" linked to ontology {ontology_id}" if ontology_id else ""))
                result = {"success": True, "document_id": document_id, "categories_count": category_count}
            else:
                log(WARNING, f"Query result was empty for document {document_id}")
                result = {"error": 3, "error_message": "Document not found or failed to update categories"}

        except Exception as e:
            error_message: str = f"An error occurred while updating document categories: {e}"
            log(ERROR, f"ERROR: {error_message}")
            result = {"error": 2, "error_message": error_message}

        return result

    def get_category_ontology_id_for_document(self, document_id: str, effective_user_id: Optional[str] = None) -> Optional[str]:
        """
        Retrieve the default category ontology ID associated with a document.
        This can be stored as a property on the Document node or inferred from relationships.
        Args:
            document_id: The ID of the document.
            effective_user_id: Optional. If provided, verifies READ permission on the document before returning.
        Returns:
            The ontology ID as a string, or None if not found, not specified, or access denied.
        """
        if not document_id or (isinstance(document_id, str) and len(document_id.strip()) == 0):
            log(ERROR, "document_id cannot be null or empty string")
            return None
        
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        
        # If effective_user_id provided, verify READ access to the document first
        if effective_user_id:
            doc = self.get_document(document_id, effective_user_id=effective_user_id)
            if doc is None:
                log(WARNING, f"DocumentsService.get_category_ontology_id_for_document: READ access denied for effective user '{effective_user_id}' on document '{document_id}'")
                return None
        
        try:
            # Check if document has category_ontology_id property
            query: str = """
                MATCH (d:Document)
                WHERE d.id = $document_id OR d.documentid = $document_id
                RETURN d.category_ontology_id AS ontology_id
            """
            
            parameters: Dict[str, Any] = {"document_id": document_id}
            result = execute_neo4j_query(query, parameters)
            
            if not result or len(result) == 0:
                log(ERROR, f"Document with ID {document_id} not found")
                return None
            
            ontology_id: Optional[str] = result[0].get("ontology_id")
            if ontology_id:
                log(DEBUG, f"Found category ontology ID '{ontology_id}' for document {document_id}")
                return ontology_id
            
            log(DEBUG, f"Document {document_id} found but has no category ontology ID")
            return None
            
        except Exception as e:
            error_message: str = f"An error occurred while retrieving category ontology ID for document {document_id}: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return None

    def create_or_update_chapter(self, document_id: str, chapter_id: str, chapter_data: Dict[str, Any], effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create or update a chapter for a document."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_document_permission(effective_user_id, document_id, "EDIT"):
            log(WARNING, f"DocumentsService.create_or_update_chapter: EDIT access denied for effective user '{effective_user_id}' on document '{document_id}'")
            return {"error": 1, "error_message": "EDIT permission denied"}
        try:
            partition = _partition_letter_from_hash(document_id)

            query = """
                MERGE (chapter:Chapter {id: $chapter_id})
                SET chapter.text = $chapter_text, 
                    chapter.documentid = $document_id, 
                    chapter.title = $chapter_title,
                    chapter.enhanced_text = $chapter_enhanced_text,
                    chapter.enhanced_type = $chapter_enhanced_type,
                    chapter.partition = $partition
                WITH chapter
                CALL db.create.setNodeVectorProperty(chapter, 'embedding', $chapter_embedding)
                RETURN chapter.id AS id
            """
            
            parameters = {
                "document_id": document_id,
                "chapter_id": chapter_id,
                "chapter_text": chapter_data.get("text", ""),
                "chapter_title": chapter_data.get("title"),
                "chapter_embedding": chapter_data.get("embedding", []),
                "chapter_enhanced_text": chapter_data.get("enhanced_text", ""),
                "chapter_enhanced_type": chapter_data.get("enhanced_type", ""),
                "partition": partition,
            }

            result = execute_neo4j_query(query, parameters)
            
            if result:
                # Create relationship between document and chapter
                relationship_query = """
                    MATCH (doc:Document)
                    WHERE doc.id = $document_id OR doc.documentid = $document_id
                    WITH doc
                    MATCH (chapter:Chapter {id: $chapter_id})
                    MERGE (chapter)-[:PART_OF]->(doc)
                """
                execute_neo4j_query(relationship_query, {"document_id": document_id, "chapter_id": chapter_id})
                
                log(INFO, f"Chapter created/updated successfully: {chapter_id}")
                return {"success": True, "chapter_id": chapter_id}
            else:
                return {"error": 3, "error_message": "Failed to create/update chapter"}

        except Exception as e:
            error_message = f"An error occurred while creating/updating chapter: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def create_or_update_image(self, document_id: str, chapter_id: str, image_data: Dict[str, Any], effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create or update an image associated with a chapter."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_document_permission(effective_user_id, document_id, "EDIT"):
            log(WARNING, f"DocumentsService.create_or_update_image: EDIT access denied for effective user '{effective_user_id}' on document '{document_id}'")
            return {"error": 1, "error_message": "EDIT permission denied"}
        try:
            query = """
                MATCH (chapter:Chapter {id: $chapter_id})
                MERGE (img:Image {id: $image_id})
                SET img.image_base64 = $image_base64, img.documentid = $document_id
                MERGE (img)<-[:HAS_IMAGE]-(chapter)
                RETURN img.id AS id
            """
            
            parameters = {
                "document_id": document_id,
                "chapter_id": chapter_id,
                "image_id": image_data.get("id"),
                "image_base64": image_data.get("base64", "")
            }

            result = execute_neo4j_query(query, parameters)
            
            if result:
                log(INFO, f"Image created/updated successfully: {image_data.get('id')}")
                return {"success": True, "image_id": image_data.get("id")}
            else:
                return {"error": 3, "error_message": "Failed to create/update image"}

        except Exception as e:
            error_message = f"An error occurred while creating/updating image: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def create_vector_index(self, index_name: str, node_label: str, property_name: str, dimension: int) -> Dict[str, Any]:
        """Create a vector index for embeddings."""
        try:
            query = f"""
                CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
                FOR (n: {node_label}) ON (n.{property_name})
                OPTIONS {{ indexConfig: {{
                    `vector.dimensions`: $dimension,
                    `vector.similarity_function`: 'cosine'
                }} }}
            """
            
            parameters = {"dimension": dimension}
            execute_neo4j_query(query, parameters, ignore_warnings=True)
            
            log(DEBUG, f"Vector index '{index_name}' ensured to exist")
            return {"success": True, "index_name": index_name}

        except Exception as e:
            error_message = f"An error occurred while creating vector index: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def create_all_required_indexes(self) -> Dict[str, Any]:
        """Create or verify all Neo4j indexes required for document ingestion.

        Delegates to neo4j_indexes.ensure_all_indexes() which is the single
        source of truth for the complete index catalog.
        """
        try:
            from src.core.llm.embeddings import EmbeddingsManager
            from src.core.base.neo4j_indexes import ensure_all_indexes
            embeddings_dimension = EmbeddingsManager.instance().get_embeddings_dimension()
            ensure_all_indexes(embeddings_dimension)
            return {"success": True}
        except Exception as e:
            error_message = f"An error occurred while creating required indexes: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def update_document_content(self, document_id: str, document_name: str, document_text: str = None, effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Update document content and reset enhanced text fields."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_document_permission(effective_user_id, document_id, "EDIT"):
            log(WARNING, f"DocumentsService.update_document_content: EDIT access denied for effective user '{effective_user_id}' on document '{document_id}'")
            return {"error": 1, "error_message": "EDIT permission denied"}
        try:
            query = """
                MERGE (d:Document {id: $document_id})
                SET d.name = $document_name,
                    d.documentid = $document_id,
                    d.text = $document_text,
                    d.enhanced_text = null,
                    d.enhanced_tx = null,
                    d.enhanced_type = null,
                    d.enhanced_modified = null
                RETURN d.id AS id
            """
            
            parameters = {
                "document_id": document_id,
                "document_name": document_name,
                "document_text": document_text
            }

            result = execute_neo4j_query(query, parameters)
            
            if result:
                log(INFO, f"[doc: {document_id}] Content updated successfully")
                return {"success": True, "document_id": document_id}
            else:
                return {"error": 3, "error_message": "Failed to update document content"}

        except Exception as e:
            error_message = f"An error occurred while updating document content: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def create_or_update_keyword(self, document_id: str, node_id: str, keyword: str, lang: str, embedding: List[float], effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create or update a keyword with embedding."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_document_permission(effective_user_id, document_id, "EDIT"):
            log(WARNING, f"DocumentsService.create_or_update_keyword: EDIT access denied for effective user '{effective_user_id}' on document '{document_id}'")
            return {"error": 1, "error_message": "EDIT permission denied"}
        try:
            # Use an atomic Cypher update so concurrent updates cannot cause races.
            # Preserve existing lang when equal, keep '*' if already multiple, otherwise set to '*'.
            query = """
                MATCH (c {id: $node_id})
                MERGE (k:Keyword {text: $keyword})
                MERGE (c)-[:HAS_KEYWORD]->(k)
                SET k.embedding = $embedding,
                    k.lang = CASE
                        WHEN k.lang IS NULL THEN $lang
                        WHEN k.lang = $lang THEN k.lang
                        WHEN k.lang = '*' THEN '*'
                        ELSE '*'
                    END
                RETURN k.text AS keyword
            """

            # Normalize language to uppercase, but preserve '*' as-is
            lang_norm = None
            if lang is not None:
                lang_s = lang.strip()
                if lang_s == '*':
                    lang_norm = '*'
                elif len(lang_s) > 0:
                    lang_norm = lang_s.upper()

            parameters = {
                "document_id": document_id,
                "node_id": node_id,
                "keyword": keyword.strip(),
                "embedding": embedding,
                "lang": lang_norm
            }

            result = execute_neo4j_query(query, parameters)

            if result:
                log(INFO, f"Keyword created/updated successfully: {keyword}")
                return {"success": True, "keyword": keyword}
            else:
                return {"error": 3, "error_message": "Failed to create/update keyword"}

        except Exception as e:
            error_message = f"An error occurred while creating/updating keyword: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def create_or_update_summary(self, chapter_id: str, summary_data: Dict[str, Any], effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create or update a summary for a chapter."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        # Get document_id from summary_data for permission check
        document_id = summary_data.get("document_id", "")
        if effective_user_id and document_id and not self._check_document_permission(effective_user_id, document_id, "EDIT"):
            log(WARNING, f"DocumentsService.create_or_update_summary: EDIT access denied for effective user '{effective_user_id}' on document '{document_id}'")
            return {"error": 1, "error_message": "EDIT permission denied"}
        try:
            # Merge the Summary node by its id first, then ensure the chapter has a relationship
            # to that Summary. This avoids creating anonymous Summary nodes or accidental
            # concatenation of identifiers when the pattern-merge is used.
            document_id = summary_data.get("document_id", "")
            partition = _partition_letter_from_hash(document_id) if document_id else "A"

            query = """
                MATCH (chapter:Chapter {id: $chapter_id})
                SET chapter.title = $title
                WITH chapter
                MERGE (s:Summary {id: $summary_id})
                SET s.text = $summary, 
                    s.documentid = $document_id,
                    s.partition = $partition
                WITH chapter, s
                MERGE (chapter)-[:HAS_SUMMARY]->(s)
                WITH s
                CALL db.create.setNodeVectorProperty(s, 'embedding', $embedding)
                RETURN s.id AS summary_id
            """
            
            parameters = {
                "chapter_id": chapter_id,
                "summary_id": summary_data.get("summary_id"),
                "document_id": document_id,
                "summary": summary_data.get("text"),
                "title": summary_data.get("title"),
                "embedding": summary_data.get("embedding", []),
                "partition": partition
            }

            result = execute_neo4j_query(query, parameters)
            
            if result:
                log(INFO, f"Summary created/updated successfully: {summary_data.get('summary_id')}")
                return {"success": True, "summary_id": summary_data.get("summary_id")}
            else:
                return {"error": 3, "error_message": "Failed to create/update summary"}

        except Exception as e:
            error_message = f"An error occurred while creating/updating summary: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def create_or_update_document_summary(self, document_id: str, summary_text: str, title: str, effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create or update a document-level summary."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()

        try:
            # Generate embedding for the summary text
            from src.core.llm.embeddings import EmbeddingsManager
            embedding = EmbeddingsManager.instance().get_embeddings(summary_text) if summary_text else []
            
            # Ensure the document-level Summary node is merged by id first, then attach it.
            # Match by either `id` or `documentid` so we update the correct Document node,
            # and compute `summary_id` in Python to avoid expression evaluation issues in Cypher.
            partition = _partition_letter_from_hash(document_id)

            query1 = """
                MATCH (d:Document)
                WHERE d.id = $document_id OR d.documentid = $document_id
                SET d.title = $title
                RETURN d.id AS id, d.title AS title
            """
            query2 = """
                MATCH (d:Document)
                WHERE d.id = $document_id OR d.documentid = $document_id
                WITH d
                MERGE (s:Summary {id: $summary_id})
                SET s.text = $summary_text, 
                    s.documentid = $document_id,
                    s.partition = $partition
                WITH d, s
                CALL db.create.setNodeVectorProperty(s, 'embedding', $embedding)
                WITH d, s
                MERGE (d)-[:HAS_SUMMARY]->(s)
                RETURN count(*) AS count
            """

            parameters = {
                "document_id": document_id,
                "summary_text": summary_text,
                "title": title,
                "summary_id": f"{document_id}-summary",
                "partition": partition,
                "embedding": embedding
            }
            
            result1 = execute_neo4j_query(query1, parameters)
            result2 = execute_neo4j_query(query2, parameters)
            
            if result1 and result2:
                log(INFO, f"Executed create/update document summary for document ID: {document_id}. Results:\n1: {result1}\n2: {result2}")
                log(INFO, f"Document summary created/updated successfully: {document_id}-summary")
                return {"success": True, "summary_id": f"{document_id}-summary"}
            else:
                log(WARNING, f"Document summary create/update returned no results for document ID: {document_id}. Results:\n1: {result1}\n2: {result2}")
                return {"error": 3, "error_message": "Failed to create/update document summary"}

        except Exception as e:
            error_message = f"An error occurred while creating/updating document summary: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def create_or_update_paragraph(self, chapter_id: str, paragraph_data: Dict[str, Any], effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create or update a paragraph for a chapter."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        # Get document_id from paragraph_data for permission check
        document_id = paragraph_data.get("document_id", "")
        if effective_user_id and document_id and not self._check_document_permission(effective_user_id, document_id, "EDIT"):
            log(WARNING, f"DocumentsService.create_or_update_paragraph: EDIT access denied for effective user '{effective_user_id}' on document '{document_id}'")
            return {"error": 1, "error_message": "EDIT permission denied"}
        try:
            document_id = paragraph_data.get("document_id", "")
            partition = _partition_letter_from_hash(document_id) if document_id else "A"

            query = """
                MATCH (chapter:Chapter {id: $chapter_id})
                MERGE (par:Paragraph {id: $paragraph_id})
                SET par.text = $paragraph_text,
                    par.partition = $partition
                MERGE (par)-[:PART_OF]->(chapter)
                WITH par
                CALL db.create.setNodeVectorProperty(par, 'embedding', $paragraph_embedding)
                RETURN par.id AS id
            """
            
            parameters = {
                "chapter_id": chapter_id,
                "paragraph_id": paragraph_data.get("paragraph_id"),
                "paragraph_text": paragraph_data.get("paragraph_text"),
                "paragraph_embedding": paragraph_data.get("paragraph_embedding", []),
                "partition": partition,
            }

            result = execute_neo4j_query(query, parameters)
            
            if result:
                log(INFO, f"Paragraph created/updated successfully: {paragraph_data.get('paragraph_id')}")
                return {"success": True, "paragraph_id": paragraph_data.get("paragraph_id")}
            else:
                return {"error": 3, "error_message": "Failed to create/update paragraph"}

        except Exception as e:
            error_message = f"An error occurred while creating/updating paragraph: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def create_or_update_paragraphs_batch(self, chapter_id: str, paragraphs: List[Dict[str, Any]], effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create or update multiple paragraphs for a chapter in a single operation."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        # Check ACL on the document for the first paragraph (all should be from same document)
        if effective_user_id and paragraphs and len(paragraphs) > 0:
            first_doc_id = paragraphs[0].get("document_id", "")
            if first_doc_id and not self._check_document_permission(effective_user_id, first_doc_id, "EDIT"):
                log(WARNING, f"DocumentsService.create_or_update_paragraphs_batch: EDIT access denied for effective user '{effective_user_id}' on document '{first_doc_id}'")
                return {"error": 1, "error_message": "EDIT permission denied"}
        try:
            # Pre-compute partitions for each paragraph in Python
            paragraphs_with_partition = []
            for p in paragraphs:
                document_id = p.get("document_id", "")
                partition = _partition_letter_from_hash(document_id) if document_id else "A"
                p_copy = p.copy()
                p_copy["partition"] = partition
                paragraphs_with_partition.append(p_copy)

            query = """
                MATCH (chapter:Chapter {id: $chapter_id})-[:PART_OF]->(d:Document)
                UNWIND $paragraphs AS paragraph
                MERGE (par:Paragraph {id: paragraph.paragraph_id})
                SET par.text = paragraph.paragraph_text,
                    par.documentid = d.id,
                    par.partition = paragraph.partition
                MERGE (par)-[:PART_OF]->(chapter)
                WITH par, paragraph
                CALL db.create.setNodeVectorProperty(par, 'embedding', paragraph.paragraph_embedding)
            """
            
            parameters = {
                "chapter_id": chapter_id,
                "paragraphs": paragraphs_with_partition
            }

            result = execute_neo4j_query(query, parameters)
            log(INFO, f"Batch created/updated {len(paragraphs)} paragraphs for chapter: {chapter_id}")
            return {"success": True, "chapter_id": chapter_id, "paragraph_count": len(paragraphs)}

        except Exception as e:
            error_message = f"An error occurred while batch creating/updating paragraphs: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def create_or_update_questions_batch(self, chapter_id: str, questions: List[Dict[str, Any]], effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create or update multiple questions for a chapter in a single operation."""
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        # Check ACL on the document for the first question (all should be from same document)
        if effective_user_id and questions and len(questions) > 0:
            first_doc_id = questions[0].get("document_id", "")
            if first_doc_id and not self._check_document_permission(effective_user_id, first_doc_id, "EDIT"):
                log(WARNING, f"DocumentsService.create_or_update_questions_batch: EDIT access denied for effective user '{effective_user_id}' on document '{first_doc_id}'")
                return {"error": 1, "error_message": "EDIT permission denied"}
        try:
            query = """
                MERGE (p:Chapter {id: $chapter_id})
                WITH p
                UNWIND $questions AS question
                CREATE (q:Question {id: question.id})
                SET q.text = question.text, q.documentid = question.document_id
                MERGE (q)<-[:HAS_QUESTION]-(p)
                WITH q, question
                CALL db.create.setNodeVectorProperty(q, 'embedding', question.embedding)
            """
            
            parameters = {
                "chapter_id": chapter_id,
                "questions": questions
            }

            result = execute_neo4j_query(query, parameters)
            log(INFO, f"Batch created/updated {len(questions)} questions for chapter: {chapter_id}")
            return {"success": True, "chapter_id": chapter_id, "question_count": len(questions)}

        except Exception as e:
            error_message = f"An error occurred while batch creating/updating questions: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def create_fulltext_index(self, index_name: str, node_label: str, properties: List[str]) -> Dict[str, Any]:
        """Create a fulltext index."""
        try:
            properties_str = ", ".join([f"n.{prop}" for prop in properties])
            query = f"""
                CREATE FULLTEXT INDEX `{index_name}` IF NOT EXISTS FOR (n:{node_label}) ON EACH [{properties_str}]
                OPTIONS {{
                    indexConfig: {{
                        `fulltext.eventually_consistent`: true
                    }}
                }}
            """
            
            execute_neo4j_query(query, {}, ignore_warnings=True)
            
            log(INFO, f"Fulltext index '{index_name}' created/updated successfully")
            return {"success": True, "index_name": index_name}

        except Exception as e:
            error_message = f"An error occurred while creating fulltext index: {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def cleanup_orphan_nodes(self) -> Dict[str, Any]:
        """
        Clean up orphan nodes that are not connected to their parent nodes.
        This includes:
        - Summary nodes not connected to Document
        - Chapter nodes not connected to Document
        - Paragraph nodes not connected to Chapter
        
        Returns:
            Success/error message dictionary with details about cleanup.
        """
        errors = []
        
        # Step 1: Delete orphan Summaries (not connected to Document)
        try:
            summary_result = self.delete_orphan_nodes('Summary', 'Document')
            if not summary_result.get("success", False):
                log(WARNING, f"Warning deleting orphan Summary nodes: {summary_result.get('error_message')}")
        except Exception as e:
            error_msg = f"Exception during orphan Summary node cleanup: {str(e)}"
            log(ERROR, error_msg)
            errors.append(error_msg)
        
        # Step 2: Delete orphan Chapters (not connected to Document)
        try:
            chapter_result = self.delete_orphan_nodes('Chapter', 'Document')
            if not chapter_result.get("success", False):
                log(WARNING, f"Warning deleting orphan Chapter nodes: {chapter_result.get('error_message')}")
        except Exception as e:
            error_msg = f"Exception during orphan Chapter node cleanup: {str(e)}"
            log(ERROR, error_msg)
            errors.append(error_msg)
        
        # Step 3: Delete orphan Paragraphs (not connected to Chapter)
        try:
            paragraph_result = self.delete_orphan_nodes('Paragraph', 'Chapter')
            if not paragraph_result.get("success", False):
                log(WARNING, f"Warning deleting orphan Paragraph nodes: {paragraph_result.get('error_message')}")
        except Exception as e:
            error_msg = f"Exception during orphan Paragraph node cleanup: {str(e)}"
            log(ERROR, error_msg)
            errors.append(error_msg)

        if errors:
            return {
                "success": False,
                "errors": errors,
                "error_message": "; ".join(errors)
            }
        else:
            return {
                "success": True,
                "message": "Orphan nodes cleaned up successfully"
            }

    def delete_document_nodes_for_reingestion(self, document_id: str, effective_user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete document nodes in preparation for re-ingestion.
        This deletes:
        1. All nodes with documentid except the Document node itself
        2. All Keywords associated to the Document node
        3. Orphaned Keyword and Paragraph nodes (no relationships)
        4. Sets extraction_quality to NULL on the Document node
        5. Cleans up orphaned Summary, Chapter, and Paragraph nodes (not connected to their parents)
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        if effective_user_id and not self._check_document_permission(effective_user_id, document_id, "DELETE"):
            log(WARNING, f"DocumentsService.delete_document_nodes_for_reingestion: DELETE access denied for effective user '{effective_user_id}' on document '{document_id}'")
            return {"error": 1, "error_message": "DELETE permission denied"}
        errors = []
        
        # Step 1: Delete all nodes that carry documentid except the Document node itself
        try:
            query = """MATCH (n) WHERE n.documentid = $document_id AND NOT n:Document DETACH DELETE n"""
            execute_neo4j_query(query, {"document_id": document_id})
            log(INFO, f"Deleted nodes for Document {document_id}")
        except Exception as e:
            error_msg = f"Failed to delete nodes for Document {document_id}: {str(e)}"
            log(ERROR, error_msg)
            errors.append(error_msg)

        # Step 1b: Drop the links this kind of run derives automatically, so re-extraction starts clean.
        # Curated links survive on purpose: the Document node itself is kept across a re-ingestion, and
        # entity nodes never carry a documentid, so those edges stay valid through the churn.
        try:
            query = """MATCH (e:Entity)-[r:IS_DOCUMENTED {kind: 'extracted'}]->(d:Document) WHERE d.documentid = $document_id DELETE r"""
            execute_neo4j_query(query, {"document_id": document_id})
            log(INFO, f"Deleted extracted IS_DOCUMENTED links for Document {document_id}")
        except Exception as e:
            error_msg = f"Failed to delete extracted IS_DOCUMENTED links for Document {document_id}: {str(e)}"
            log(ERROR, error_msg)
            errors.append(error_msg)

        # Step 2: Delete all Keywords associated to the Document node
        try:
            query = """MATCH (d:Document)-[:HAS_KEYWORD]->(k:Keyword) WHERE d.documentid = $document_id DETACH DELETE k"""
            execute_neo4j_query(query, {"document_id": document_id})
            log(DEBUG, f"Deleted Keywords for Document {document_id}")
        except Exception as e:
            error_msg = f"Failed to delete Keywords for Document {document_id}: {str(e)}"
            log(ERROR, error_msg)
            errors.append(error_msg)

        # Step 3: Delete Keyword and Paragraph nodes that have no relationships (true orphans).
        try:
            query = """MATCH (k:Keyword) WHERE NOT ( (k)--() ) DELETE k"""
            execute_neo4j_query(query)
            query = """MATCH (p:Paragraph) WHERE NOT ( (p)--() ) DELETE p"""
            execute_neo4j_query(query)
            log(DEBUG, "Deleted orphan Keyword/Paragraph nodes")
        except Exception as e:
            error_msg = f"Failed to delete orphan Keyword/Paragraph nodes: {str(e)}"
            log(ERROR, error_msg)
            errors.append(error_msg)

        # Step 4: Set extraction_quality to NULL on the Document node for this document id
        try:
            query = """MATCH (d:Document) WHERE d.documentid = $document_id SET d.extraction_quality = NULL"""
            execute_neo4j_query(query, {"document_id": document_id})
            log(DEBUG, f"Set extraction_quality to NULL for Document {document_id}")
        except Exception as e:
            error_msg = f"Failed to set extraction_quality to NULL for Document {document_id}: {str(e)}"
            log(ERROR, error_msg)
            errors.append(error_msg)

        # Step 5: Clean up orphan Summary, Chapter, and Paragraph nodes
        try:
            cleanup_result = self.cleanup_orphan_nodes()
            if not cleanup_result.get("success", False):
                errors.extend(cleanup_result.get("errors", []))
        except Exception as e:
            error_msg = f"Exception during orphan node cleanup: {str(e)}"
            log(ERROR, error_msg)
            errors.append(error_msg)

        if errors:
            return {
                "success": False,
                "document_id": document_id,
                "errors": errors,
                "error_message": "; ".join(errors)
            }
        else:
            return {
                "success": True,
                "document_id": document_id,
                "message": "Document nodes deleted successfully for re-ingestion"
            }

    def delete_all_indexes_and_clear_database(self) -> Dict[str, Any]:
        """
        Clear the graph database (except CacheEntry nodes) and delete all indexes.
        This is used when completely resetting the RAG database.
        """
        errors = []
        
        # Step 1: Clear the graph database (except CacheEntry nodes).
        # Batched to avoid a single massive transaction on large databases.
        try:
            while True:
                batch = execute_neo4j_query(
                    "MATCH (n) WHERE NOT n:CacheEntry WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS c",
                    {}
                )
                if not batch or batch[0].get("c", 0) == 0:
                    break
            log(INFO, "Cleared graph database (preserved CacheEntry nodes)")
        except Exception as e:
            error_msg = f"Failed to clear graph database: {str(e)}"
            log(ERROR, error_msg)
            errors.append(error_msg)

        # Step 2: Delete each vector index
        try:
            index_names = execute_neo4j_query("SHOW VECTOR INDEXES YIELD name RETURN name AS index_name", {})
            for record in index_names:
                try:
                    index_name = record["index_name"]
                    execute_neo4j_query(f"DROP INDEX $index_name", {"index_name": index_name})
                    log(INFO, f"Deleted vector index: {index_name}")
                except Exception as e:
                    error_msg = f"Failed to delete index {index_name}: {str(e)}"
                    log(ERROR, error_msg)
                    errors.append(error_msg)
        except Exception as e:
            error_msg = f"Failed to retrieve vector indexes: {str(e)}"
            log(ERROR, error_msg)
            errors.append(error_msg)

        # Step 3: Delete full-text indexes
        try:
            execute_neo4j_query("DROP INDEX `keywordsTextIndex`", {})
            log(INFO, "Deleted full-text index: keywordsTextIndex")
        except Exception as e:
            # This might fail if the index doesn't exist, which is okay
            log(WARNING, f"Failed to delete full-text index 'keywordsTextIndex': {str(e)}")

        if errors:
            return {
                "success": False,
                "errors": errors,
                "error_message": "; ".join(errors)
            }
        else:
            return {
                "success": True,
                "message": "All indexes deleted and database cleared successfully"
            }

    def delete_orphan_nodes(self, child_node_label: str, parent_node_label: str) -> Dict[str, Any]:
        """
        Delete orphan nodes of a given type that have no relationship to a parent node type.
        
        Args:
            child_node_label: The label of the child node to check (e.g., 'Summary', 'Chapter', 'Paragraph')
            parent_node_label: The label of the parent node that should have a relationship to the child (e.g., 'Document', 'Chapter')
        
        Returns:
            Success/error message dictionary with count of deleted nodes.
            
        Examples:
            delete_orphan_nodes('Summary', 'Document')  # Delete summaries not connected to documents
            delete_orphan_nodes('Chapter', 'Document')  # Delete chapters not connected to documents
            delete_orphan_nodes('Paragraph', 'Chapter') # Delete paragraphs not connected to chapters
        """
        try:
            # Build the queries dynamically using the provided node labels.
            # Count first, then delete in batches to avoid OOM on large result sets.
            # Batched DELETE keeps each transaction small regardless of orphan count.
            BATCH_SIZE = 10_000
            count_query = f"""
                MATCH (child:{child_node_label})
                WHERE NOT (child)--(:{parent_node_label})
                RETURN count(child) AS deleted_count
            """
            delete_batch_query = f"""
                MATCH (child:{child_node_label})
                WHERE NOT (child)--(:{parent_node_label})
                WITH child LIMIT {BATCH_SIZE}
                DETACH DELETE child
                RETURN count(*) AS batch_deleted
            """

            count_result = execute_neo4j_query(count_query, {})
            deleted_count = count_result[0].get("deleted_count", 0) if count_result else 0

            # Delete in batches until no orphans remain
            while True:
                batch_result = execute_neo4j_query(delete_batch_query, {})
                batch_deleted = batch_result[0].get("batch_deleted", 0) if batch_result else 0
                if batch_deleted == 0:
                    break
            
            if deleted_count > 0:
                log(INFO, f"Deleted {deleted_count} orphan {child_node_label} nodes (not connected to {parent_node_label})")

            return {
                "success": True,
                "child_node_label": child_node_label,
                "parent_node_label": parent_node_label,
                "deleted_count": deleted_count,
                "message": f"Deleted {deleted_count} orphan {child_node_label} nodes successfully"
            }
        except Exception as e:
            error_message = f"An error occurred while deleting orphan {child_node_label} nodes (not connected to {parent_node_label}): {e}"
            log(ERROR, f"ERROR: {error_message}")
            return {"error": 2, "error_message": error_message}

    def delete_orphan_document_nodes(self) -> Dict[str, Any]:
        """
        Delete orphan document nodes (Chapter, Paragraph etc. nodes without any relationships to a Document node).
        Returns:
            Success/error message dictionary.

        Strategy:
            1. Identify orphan nodes by checking for the absence of relationships to Document nodes.
               We search for all nodes with the attribute "documentid" with no corresponding Document node.
            2. For each orphan document ID, call delete_document_nodes_for_reingestion() which handles comprehensive cleanup
               including orphaned Summary, Chapter, and Paragraph nodes.
            3. Clean up any remaining orphan nodes using cleanup_orphan_nodes().
        """
        errors = []
        
        # Step 1: Identify orphan document IDs
        try:
            # Lightweight existence check: a documentid is "orphan" when no Document
            # node carries that id.  Avoids the expensive variable-length path
            # expansion ([*1..3]) that caused OOM on large databases.
            query = """
                MATCH (n)
                WHERE n.documentid IS NOT NULL
                  AND NOT n:Document
                WITH DISTINCT n.documentid AS doc_id
                WHERE NOT EXISTS {
                    MATCH (d:Document {documentid: doc_id})
                }
                RETURN doc_id AS orphan_document_id
            """
            orphan_docs = execute_neo4j_query(query, {})
            orphan_document_ids = [record["orphan_document_id"] for record in orphan_docs]
            log(DEBUG, f"Identified {len(orphan_document_ids)} orphan document IDs")

        except Exception as e:
            error_msg = f"Failed to identify orphan document IDs: {str(e)}"
            log(ERROR, error_msg)
            return {"success": False, "error_message": error_msg}

        # Step 2: Delete nodes for each orphan document ID
        # delete_document_nodes_for_reingestion() handles all cleanup including orphaned nodes
        for doc_id in orphan_document_ids:
            result = self.delete_document_nodes_for_reingestion(doc_id)
            if not result.get("success", False):
                errors.append(f"Failed to delete nodes for orphan document ID {doc_id}: {result.get('error_message')}")

        # Step 3: Clean up any remaining orphan nodes
        try:
            cleanup_result = self.cleanup_orphan_nodes()
            if not cleanup_result.get("success", False):
                errors.extend(cleanup_result.get("errors", []))
        except Exception as e:
            error_msg = f"Exception during final orphan node cleanup: {str(e)}"
            log(ERROR, error_msg)
            errors.append(error_msg)

        if errors:
            return {
                "success": False,
                "errors": errors,
                "error_message": "; ".join(errors)
            }
        else:
            return {
                "success": True,
                "message": "All orphan document nodes deleted successfully"
            }

    def get_documents_with_multiple_categories(self, dossier_id: str, ontology_id: Optional[str] = None, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all documents in a dossier that have been assigned more than one category.
        
        Args:
            dossier_id: The ID of the dossier to filter documents by.
            ontology_id: Optional. If provided, only count categories from this ontology when determining "multiple categories".
                        If not provided, all categories are counted.
            effective_user_id: Optional. If provided, verifies READ permission on the dossier before returning documents.
        
        Returns:
            List of document dictionaries with at least 'id', 'documentid', and 'categories' fields.
            The 'categories' field contains only the categories assigned to that document.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        result = []
        
        # If effective_user_id provided, verify READ access to the dossier first
        if effective_user_id:
            dossiers_service: DossiersInterface = get_dossiers_service()
            dossier = dossiers_service.get_dossier(dossier_id, effective_user_id=effective_user_id)
            if dossier is None:
                log(WARNING, f"DocumentsService.get_documents_with_multiple_categories: READ access denied for effective user '{effective_user_id}' on dossier '{dossier_id}'")
                return []
        
        try:
            # Build query to get documents with multiple categories
            if ontology_id:
                query = """
                    MATCH (dossier:Dossier {id: $dossier_id})-[:CONTAINS]->(d:Document)
                    OPTIONAL MATCH (d)-[:HAS_CATEGORY]->(c:Category)-[:OF_CATEGORY]->(kc:KeywordCategory)
                    OPTIONAL MATCH (kc)-[:PART_OF]->(o:Ontology {id: $ontology_id})
                    WHERE o IS NOT NULL
                    WITH d, collect(DISTINCT {
                        category_id: kc.id,
                        category_text_en: kc.text_en,
                        category_value: c.text
                    }) AS categories
                    WHERE size(categories) > 1
                    RETURN d.id AS id, d.documentid AS documentid, d.name AS name, d.title AS title, categories
                    ORDER BY d.id
                """
                params = {"dossier_id": dossier_id, "ontology_id": ontology_id}
            else:
                query = """
                    MATCH (dossier:Dossier {id: $dossier_id})-[:CONTAINS]->(d:Document)
                    OPTIONAL MATCH (d)-[:HAS_CATEGORY]->(c:Category)-[:OF_CATEGORY]->(kc:KeywordCategory)
                    WITH d, collect(DISTINCT {
                        category_id: kc.id,
                        category_text_en: kc.text_en,
                        category_value: c.text
                    }) AS categories
                    WHERE size(categories) > 1
                    RETURN d.id AS id, d.documentid AS documentid, d.name AS name, d.title AS title, categories
                    ORDER BY d.id
                """
                params = {"dossier_id": dossier_id}
            
            rows = execute_neo4j_query(query, params)
            if rows is None:
                return result
            
            for row in rows:
                doc = {
                    "id": row.get("id"),
                    "documentid": row.get("documentid"),
                    "name": row.get("name"),
                    "title": row.get("title"),
                    "categories": row.get("categories", [])
                }
                result.append(doc)
            
            log(DEBUG, f"Found {len(result)} documents in dossier '{dossier_id}' with multiple categories")
        
        except Exception as e:
            error_message = f"Error retrieving documents with multiple categories: {e}"
            log(ERROR, f"ERROR: {error_message}")
        
        return result
    
    def get_documents_without_category(self, dossier_id: str, category_text_en: Optional[str] = None, ontology_id: Optional[str] = None, effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve documents in a dossier that either have no categories (if category_text_en is None) or don't have a specific category.
        
        Args:
            dossier_id: The ID of the dossier to filter documents by.
            category_text_en: Optional. If provided, return documents that don't have this specific category.
                             If not provided, return documents with no categories at all.
            ontology_id: Optional. Used when category_text_en is provided to filter by ontology.
            effective_user_id: Optional. If provided, verifies READ permission on the dossier before returning documents.
        
        Returns:
            List of document dictionaries with 'id' and 'documentid' fields.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        result = []
        
        # If effective_user_id provided, verify READ access to the dossier first
        if effective_user_id:
            dossiers_service: DossiersInterface = get_dossiers_service()
            dossier = dossiers_service.get_dossier(dossier_id, effective_user_id=effective_user_id)
            if dossier is None:
                log(WARNING, f"DocumentsService.get_documents_without_category: READ access denied for effective user '{effective_user_id}' on dossier '{dossier_id}'")
                return []
        
        try:
            if category_text_en and ontology_id:
                # Get documents that don't have the specific category
                query = """
                    MATCH (dossier:Dossier {id: $dossier_id})-[:CONTAINS]->(d:Document)
                    OPTIONAL MATCH (d)-[:HAS_CATEGORY]->(c:Category)-[:OF_CATEGORY]->(kc:KeywordCategory {text_en: $category_text_en})
                    OPTIONAL MATCH (kc)-[:PART_OF]->(o:Ontology {id: $ontology_id})
                    WHERE o IS NOT NULL
                    WITH d, count(c) AS cat_count
                    WHERE cat_count = 0
                    RETURN d.id AS id, d.documentid AS documentid, d.name AS name, d.title AS title
                    ORDER BY d.id
                """
                params = {"dossier_id": dossier_id, "category_text_en": category_text_en, "ontology_id": ontology_id}
            else:
                # Get documents with no categories at all
                query = """
                    MATCH (dossier:Dossier {id: $dossier_id})-[:CONTAINS]->(d:Document)
                    OPTIONAL MATCH (d)-[:HAS_CATEGORY]->(c:Category)
                    WITH d, count(c) AS cat_count
                    WHERE cat_count = 0
                    RETURN d.id AS id, d.documentid AS documentid, d.name AS name, d.title AS title
                    ORDER BY d.id
                """
                params = {"dossier_id": dossier_id}
            
            rows = execute_neo4j_query(query, params)
            if rows is None:
                return result
            
            for row in rows:
                doc = {
                    "id": row.get("id"),
                    "documentid": row.get("documentid"),
                    "name": row.get("name"),
                    "title": row.get("title")
                }
                result.append(doc)
            
            log(DEBUG, f"Found {len(result)} documents in dossier '{dossier_id}' without category")
        
        except Exception as e:
            error_message = f"Error retrieving documents without category: {e}"
            log(ERROR, f"ERROR: {error_message}")
        
        return result

    def get_document_ids_for_dossier(self, dossier_id: str, merge_clause: Optional[str] = None, limit: Optional[int] = None, effective_user_id: Optional[str] = None) -> List[str]:
        """Return the IDs of all documents owned by the given dossier.
        When effective_user_id is provided, verifies READ permission on the dossier before returning document IDs.
        
        Args:
            dossier_id: The ID of the dossier to retrieve documents for.
            merge_clause: Optional Cypher WHERE clause to filter documents (e.g., "d.status = 'active'").
                         Should not include the 'WHERE' keyword - only the condition itself. Defaults to None.
            limit: Optional maximum number of document IDs to return. Defaults to None (no limit).
        
        Returns:
            List of document IDs matching the dossier and optional filter.
        """
        if not dossier_id:
            return []
        
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        
        # If effective_user_id provided, verify READ access to the dossier first
        if effective_user_id:
            dossiers_service: DossiersInterface = get_dossiers_service()
            dossier = dossiers_service.get_dossier(dossier_id, effective_user_id=effective_user_id)
            if dossier is None:
                log(WARNING, f"DocumentsService.get_document_ids_for_dossier: READ access denied for effective user '{effective_user_id}' on dossier '{dossier_id}'")
                return []
        
        try:
            query = """
                MATCH (dos:Dossier {id: $dossier_id})-[:CONTAINS]->(d:Document)
                {merge_clause}
                RETURN d.id AS document_id
                {limit}
            """
            # Replace the placeholder with the actual merge clause if provided
            if merge_clause:
                query = query.replace("{merge_clause}", f"WHERE {merge_clause}")
            else:
                query = query.replace("{merge_clause}", "")
            if limit is not None:
                query = query.replace("{limit}", f"LIMIT {limit}")
            else:
                query = query.replace("{limit}", "")
            
            rows = execute_neo4j_query(query, {"dossier_id": dossier_id})
            return [row["document_id"] for row in rows if row.get("document_id")]
        
        except Exception as e:
            log(ERROR, f"Error retrieving document ids for dossier '{dossier_id}': {e}")
            return []

    def search_documents_by_semantic(self, documents: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Filter document metadata by semantic relevance to a free-text query.

        Documents are split into chunks of at most SEMANTIC_SEARCH_CHUNK_SIZE and ranked
        in parallel (one chunk per worker) using ThreadPoolExecutor with MAX_PARALLELISM
        workers, allowing multiple reranking models to be exercised concurrently.
        Only documents scoring at or above MIN_RERANKING_SCORE_DOC are returned.
        If the service is unavailable, the original dict is returned unchanged.
        """
        if not documents or not query:
            return documents

        SEMANTIC_SEARCH_CHUNK_SIZE = 100

        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from src.config.constants import MIN_RERANKING_SCORE_DOC, MAX_PARALLELISM
            from src.interfaces.reranking_interface import RerankingInterface
            from src.services.reranking_service import get_reranking_service

            reranking_service: RerankingInterface = get_reranking_service()
            if reranking_service is None:
                log(WARNING, "search_documents_by_semantic: reranking service not available, skipping filter")
                return documents

            fragments = []
            for doc_id, doc in documents.items():
                name = doc.get("name") or ""
                title = doc.get("title") or ""
                doc_type = doc.get("type") or ""
                summary = doc.get("summary") or ""
                text = f"Document type='{doc_type}' with title='{title}' and name '{name}' talks about '{summary}'"
                fragments.append({"id": doc_id, "text": text, "score": 1.0})

            # Split into chunks of max SEMANTIC_SEARCH_CHUNK_SIZE
            chunks = [fragments[i:i + SEMANTIC_SEARCH_CHUNK_SIZE] for i in range(0, len(fragments), SEMANTIC_SEARCH_CHUNK_SIZE)]
            num_chunks = len(chunks)
            max_workers = min(MAX_PARALLELISM, num_chunks) if MAX_PARALLELISM and MAX_PARALLELISM > 0 else num_chunks
            log(DEBUG, f"search_documents_by_semantic: reranking {len(fragments)} documents in {num_chunks} chunk(s) for query '{query}'")

            all_ranked: List[Dict[str, Any]] = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk = {executor.submit(reranking_service.rerank, query, chunk): chunk for chunk in chunks}
                for future in as_completed(future_to_chunk):
                    try:
                        result = future.result()
                        if result is not None:
                            all_ranked.extend(result)
                        else:
                            log(WARNING, "search_documents_by_semantic: a reranking chunk returned None, skipping")
                    except Exception as chunk_err:
                        log(ERROR, f"search_documents_by_semantic: chunk reranking failed: {chunk_err}")

            if not all_ranked:
                log(WARNING, "search_documents_by_semantic: no ranked results returned, returning all documents")
                return documents

            filtered = {}
            for item in all_ranked:
                doc_id = item.get("id") or item.get("fragmentid")
                score = item.get("score") or item.get("ranking") or 0.0
                if doc_id and score >= MIN_RERANKING_SCORE_DOC and doc_id in documents:
                    filtered[doc_id] = documents[doc_id]
                    log(DEBUG, f"search_documents_by_semantic: keeping '{doc_id}' (score={score:.3f})")
                else:
                    log(DEBUG, f"search_documents_by_semantic: dropping '{doc_id}' (score={score:.3f})")

            return filtered

        except Exception as e:
            log(ERROR, f"search_documents_by_semantic error: {e}")
            return documents

    def get_fragments_by_ids(self, fragment_ids: List[str], effective_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve fragments (documents, chapters, paragraphs, or summaries) by their IDs.
        
        Fragment ID formats:
        - Document: 'docid-xxx'
        - Chapter: 'docid-xxx-cyyy'
        - Paragraph: 'docid-xxx-cyyy-pzzz'
        
        When effective_user_id is provided, only fragments belonging to documents the effective user has READ permission on are returned.
        
        Returns:
            List of dictionaries with keys: fragmentid, text, enhanced_text, type
        """
        if not fragment_ids:
            return []
        
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        
        result = []
        
        try:
            # Build ACL clause for document filtering
            acl_clause = acl_document_where_clause("d") if effective_user_id else ""
            acl_clause_doc_parent = acl_document_where_clause("d") if effective_user_id else ""
            acl_params = {"effectiveUserId": effective_user_id} if effective_user_id else {}
            
            # Separate fragment IDs by type
            doc_ids = []
            chapter_ids = []
            paragraph_ids = []
            summary_ids = []
            
            for fid in fragment_ids:
                # Normalize input formats, e.g. 'doc://docid-1726-c4?rank=0.004...' -> 'docid-1726-c4'
                fid_clean = fid
                try:
                    if isinstance(fid_clean, dict) and 'fragmentid' in fid_clean:
                        fid_clean = fid_clean['fragmentid']
                    elif isinstance(fid_clean, str):
                        # remove optional scheme (doc://, http://, etc.)
                        if '://' in fid_clean:
                            fid_clean = fid_clean.split('://', 1)[1]
                        # strip query string if present
                        if '?' in fid_clean:
                            fid_clean = fid_clean.split('?', 1)[0]
                        # remove any trailing slashes and whitespace
                        fid_clean = fid_clean.strip().rstrip('/')
                    else:
                        fid_clean = str(fid_clean)
                except Exception:
                    fid_clean = str(fid)

                # Classify by the canonical fragment ID suffix.
                fragment_type = DocumentsInterface.parse_fragment_id(fid_clean).get("type")
                if fragment_type == "paragraph":
                    paragraph_ids.append(fid_clean)
                elif fragment_type == "chapter":
                    chapter_ids.append(fid_clean)
                elif fragment_type == "summary" or fid_clean.endswith("-summary"):
                    summary_ids.append(fid_clean)
                else:
                    doc_ids.append(fid_clean)
            
            # Query documents
            if doc_ids:
                query = """MATCH (d:Document)
                    WHERE d.id IN $ids{acl_clause}
                    RETURN d.id AS fragmentid, d.text AS text, d.enhanced_text AS enhanced_text, 'document' AS type""".replace("{acl_clause}", acl_clause)
                params = {"ids": doc_ids}
                params.update(acl_params)
                docs = execute_neo4j_query(query, params)
                if docs:
                    result.extend([{
                        'fragmentid': d.get('fragmentid'),
                        'text': d.get('text', ''),
                        'enhanced_text': d.get('enhanced_text'),
                        'type': d.get('type')
                    } for d in docs])
            
            # Query chapters (with parent document ACL check)
            if chapter_ids:
                query = """MATCH (c:Chapter)-[:PART_OF]->(d:Document)
                    WHERE c.id IN $ids{acl_clause_parent}
                    RETURN c.id AS fragmentid, c.text AS text, c.enhanced_text AS enhanced_text, 'chapter' AS type""".replace("{acl_clause_parent}", acl_clause_doc_parent)
                params = {"ids": chapter_ids}
                params.update(acl_params)
                chapters = execute_neo4j_query(query, params)
                if chapters:
                    result.extend([{
                        'fragmentid': c.get('fragmentid'),
                        'text': c.get('text', ''),
                        'enhanced_text': c.get('enhanced_text'),
                        'type': c.get('type')
                    } for c in chapters])
            
            # Query paragraphs (with parent chapter -> document ACL check)
            if paragraph_ids:
                query = """MATCH (p:Paragraph)-[:PART_OF]->(c:Chapter)-[:PART_OF]->(d:Document)
                    WHERE p.id IN $ids{acl_clause_ancestor}
                    RETURN p.id AS fragmentid, p.text AS text, p.enhanced_text AS enhanced_text, 'paragraph' AS type""".replace("{acl_clause_ancestor}", acl_clause_doc_parent)
                params = {"ids": paragraph_ids}
                params.update(acl_params)
                paragraphs = execute_neo4j_query(query, params)
                if paragraphs:
                    result.extend([{
                        'fragmentid': p.get('fragmentid'),
                        'text': p.get('text', ''),
                        'enhanced_text': p.get('enhanced_text'),
                        'type': p.get('type')
                    } for p in paragraphs])

            if summary_ids:
                query = """MATCH (s:Summary)
                    WHERE s.id IN $ids
                    OPTIONAL MATCH (d:Document)-[:HAS_SUMMARY]->(s)
                    OPTIONAL MATCH (c:Chapter)-[:HAS_SUMMARY]->(s)
                    OPTIONAL MATCH (c)-[:PART_OF]->(chapter_document:Document)
                    WITH s, coalesce(d, chapter_document) AS d
                    WHERE d IS NOT NULL{acl_clause}
                    RETURN s.id AS fragmentid, s.text AS text, s.enhanced_text AS enhanced_text, 'summary' AS type""".replace("{acl_clause}", acl_clause)
                params = {"ids": summary_ids}
                params.update(acl_params)
                summaries = execute_neo4j_query(query, params)
                if summaries:
                    result.extend([{
                        'fragmentid': summary.get('fragmentid'),
                        'text': summary.get('text', ''),
                        'enhanced_text': summary.get('enhanced_text'),
                        'type': summary.get('type')
                    } for summary in summaries])
            
            log(DEBUG, f"Retrieved {len(result)} fragments out of {len(fragment_ids)} requested")
            
        except Exception as e:
            error_message = f"An error occurred while fetching fragments: {e}"
            log(ERROR, f"ERROR: {error_message}")
        
        return result

    def get_nodes_from_fragments(self, fragment_ids: List[str], effective_user_id: Optional[str] = None) -> List[Any]:
        """
        Return Neo4j node records for the provided fragment ids.

        The returned value is the raw result of `execute_neo4j_query`, i.e. a
        list of Result rows where each row contains the matched node as first
        column (so callers can index row[0] to obtain the Node).
        
        When effective_user_id is provided, only fragments belonging to documents the effective user has READ permission on are returned.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        result = []
        if not fragment_ids:
            return result
        
        # Build ACL clause for document filtering
        acl_clause = acl_document_where_clause("d") if effective_user_id else ""
        acl_params = {"effectiveUserId": effective_user_id} if effective_user_id else {}

        # classify fragment ids by type
        docs = []
        chapters = []
        paragraphs = []
        for fid in fragment_ids:
            ftype = self.get_fragment_type(fid)
            if ftype == FragmentType.Document:
                docs.append(fid)
            elif ftype == FragmentType.Chapter:
                chapters.append(fid)
            elif ftype == FragmentType.Paragraph:
                paragraphs.append(fid)

        try:
            # fetch document nodes
            if docs:
                query = """MATCH (d:Document)
                    WHERE (d.id IN $ids OR d.documentid IN $ids){acl_clause}
                    RETURN elementId(d) AS eid, d.id AS id, d.documentid AS documentid, labels(d) AS labels""".replace("{acl_clause}", acl_clause)
                params = {"ids": docs}
                params.update(acl_params)
                rows = execute_neo4j_query(query, params) or []
                if rows:
                    result.extend(rows)

            # fetch chapter nodes (with parent document ACL check)
            if chapters:
                query = """MATCH (c:Chapter)-[:PART_OF]->(d:Document)
                    WHERE c.id IN $ids{acl_clause_parent}
                    RETURN elementId(c) AS eid, c.id AS id, c.documentid AS documentid, labels(c) AS labels""".replace("{acl_clause_parent}", acl_clause)
                params = {"ids": chapters}
                params.update(acl_params)
                rows = execute_neo4j_query(query, params) or []
                if rows:
                    result.extend(rows)

            # fetch paragraph nodes (with parent chapter -> document ACL check)
            if paragraphs:
                query = """MATCH (p:Paragraph)-[:PART_OF]->(c:Chapter)-[:PART_OF]->(d:Document)
                    WHERE p.id IN $ids{acl_clause_ancestor}
                    RETURN elementId(p) AS eid, p.id AS id, p.documentid AS documentid, labels(p) AS labels""".replace("{acl_clause_ancestor}", acl_clause)
                params = {"ids": paragraphs}
                params.update(acl_params)
                rows = execute_neo4j_query(query, params) or []
                if rows:
                    result.extend(rows)

        except Exception as e:
            log(ERROR, f"get_nodes_from_fragments(): Error fetching nodes: {e}")

        return result

    def get_fragment_type(self, fragment_id: str, default_type: FragmentType = FragmentType.NotIdentified) -> FragmentType:
        """Return the FragmentType for a single fragment id.

        Handles document fragments, Internet URI fragments, and Knowledge Graph entity URIs.
        Delegates to the centralised parse_fragment_id() helper on the interface.
        """
        if not fragment_id or not isinstance(fragment_id, str):
            return default_type

        if fragment_id.startswith(("http://", "https://")):
            return FragmentType.Internet
        if fragment_id.startswith("kg://"):
            return FragmentType.KnowledgeGraphEntity

        parsed = DocumentsInterface.parse_fragment_id(fragment_id)
        normalized_fragment_id = DocumentsInterface.normalize_fragment_id(fragment_id)
        type_str = parsed.get("type", "unknown")
        if not normalized_fragment_id.startswith("docid-"):
            return default_type
        if type_str == "paragraph":
            return FragmentType.Paragraph
        if type_str == "chapter":
            return FragmentType.Chapter
        if type_str == "summary":
            return FragmentType.Chapter   # summaries are chapter-level nodes
        if type_str == "document":
            return FragmentType.Document
        return default_type

    def get_chapters_for_documents(self, document_ids: List[str], effective_user_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve all chapters for given documents.
        Args:
            document_ids: List of document IDs to fetch chapters for.
            effective_user_id: If provided, only chapters from documents the effective user has READ permission on are returned. Pass None to skip ACL.
        Returns:
            Dictionary mapping document_id to list of chapter dictionaries with 'id', 'text', and 'title'.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        result = {}
        
        if not document_ids:
            return result
        
        try:
            acl_clause = acl_document_where_clause("d") if effective_user_id else ""
            cypher_query = f"""
                MATCH (d:Document)<-[:PART_OF]-(c:Chapter)
                WHERE d.id IN $document_ids{acl_clause}
                RETURN d.id AS doc_id, c.id AS chapter_id, c.text AS chapter_text, c.title AS chapter_title
                ORDER BY d.id, c.id
            """
            
            parameters = {"document_ids": document_ids}
            if effective_user_id:
                parameters["effectiveUserId"] = effective_user_id
            chapters = execute_neo4j_query(cypher_query, parameters)
            
            if chapters:
                for chapter in chapters:
                    doc_id = chapter.get('doc_id')
                    if doc_id not in result:
                        result[doc_id] = []
                    
                    result[doc_id].append({
                        'id': chapter.get('chapter_id'),
                        'text': chapter.get('chapter_text', ''),
                        'title': chapter.get('chapter_title', '')
                    })
            
            log(DEBUG, f"Retrieved chapters for {len(result)} documents out of {len(document_ids)} requested")
            
        except Exception as e:
            error_message = f"An error occurred while fetching chapters: {e}"
            log(ERROR, f"ERROR: {error_message}")
        
        return result

    def get_chapter_counts_for_documents(self, document_ids: List[str], effective_user_id: Optional[str] = None) -> Dict[str, int]:
        """
        Retrieve only the chapter count per document (no text or title transferred).
        Args:
            document_ids: List of document IDs to count chapters for.
            effective_user_id: If provided, only chapters from documents the effective user has READ permission on are counted. Pass None to skip ACL.
        Returns:
            Dictionary mapping document_id to chapter count.
        """
        if effective_user_id is None:
            user_service: UserInterface = get_user_service()
            effective_user_id = user_service.get_user_uuid()
        result = {}
        if not document_ids:
            return result
        try:
            acl_clause = acl_document_where_clause("d") if effective_user_id else ""
            cypher_query = f"""
                MATCH (d:Document)<-[:PART_OF]-(c:Chapter)
                WHERE d.id IN $document_ids{acl_clause}
                RETURN d.id AS doc_id, count(c) AS chapter_count
            """
            parameters = {"document_ids": document_ids}
            if effective_user_id:
                parameters["effectiveUserId"] = effective_user_id
            rows = execute_neo4j_query(cypher_query, parameters)
            if rows:
                for row in rows:
                    result[row.get('doc_id')] = row.get('chapter_count', 1)
        except Exception as e:
            log(ERROR, f"get_chapter_counts_for_documents(): {e}")
        return result

    def get_category_search_content(self, document_ids: List[str], max_chars: int) -> Dict[str, Any]:
        """Fetch chapters when they fit the budget, otherwise fetch summaries when those fit."""
        result: Dict[str, Any] = {
            "chapter_chars": max_chars + 1,
            "total_chapters": 0,
            "chapters": [],
            "summary_chars": max_chars + 1,
            "total_summaries": 0,
            "summaries": [],
            "summary_fetch_failed": False
        }

        if document_ids:
            effective_user_id: Optional[str] = get_user_service().get_user_uuid()
            acl_clause: str = acl_document_where_clause("d") if effective_user_id else ""
            parameters: Dict[str, Any] = {"doc_ids": document_ids}
            if effective_user_id:
                parameters["effectiveUserId"] = effective_user_id

            try:
                chapter_stats = execute_neo4j_query(
                    f"""
                    MATCH (c:Chapter)-[:PART_OF]->(d:Document)
                    WHERE d.id IN $doc_ids{acl_clause}
                    RETURN sum(size(coalesce(c.enhanced_text, c.text, ''))) AS total_chars,
                           count(c) AS total_chapters
                    """,
                    parameters,
                )
                if chapter_stats:
                    result["chapter_chars"] = int(chapter_stats[0].get("total_chars") or 0)
                    result["total_chapters"] = int(chapter_stats[0].get("total_chapters") or 0)
            except Exception as e:
                log(ERROR, f"DocumentsService.get_category_search_content: error counting chapter text: {e}")

            chapter_fetch_failed: bool = False
            if result["total_chapters"] > 0 and result["chapter_chars"] <= max_chars:
                try:
                    result["chapters"] = execute_neo4j_query(
                        f"""
                        MATCH (c:Chapter)-[:PART_OF]->(d:Document)
                        WHERE d.id IN $doc_ids{acl_clause}
                        RETURN c.id AS fragmentid, d.id AS documentid,
                               c.text AS text, c.enhanced_text AS enhanced_text
                        ORDER BY d.id, c.id
                        """,
                        parameters,
                    ) or []
                except Exception as e:
                    log(ERROR, f"DocumentsService.get_category_search_content: error fetching chapters: {e}")
                    chapter_fetch_failed = True

            if chapter_fetch_failed or result["total_chapters"] == 0 or result["chapter_chars"] > max_chars:
                try:
                    summary_stats = execute_neo4j_query(
                        f"""
                        MATCH (d:Document)-[:HAS_SUMMARY]->(s:Summary)
                        WHERE d.id IN $doc_ids{acl_clause}
                        RETURN sum(size(coalesce(s.enhanced_text, s.text, ''))) AS total_chars,
                               count(s) AS total_summaries
                        """,
                        parameters,
                    )
                    if summary_stats:
                        result["summary_chars"] = int(summary_stats[0].get("total_chars") or 0)
                        result["total_summaries"] = int(summary_stats[0].get("total_summaries") or 0)
                except Exception as e:
                    log(ERROR, f"DocumentsService.get_category_search_content: error counting summary text: {e}")

                if result["summary_chars"] <= max_chars:
                    try:
                        result["summaries"] = execute_neo4j_query(
                            f"""
                            MATCH (d:Document)-[:HAS_SUMMARY]->(s:Summary)
                            WHERE d.id IN $doc_ids{acl_clause}
                            RETURN s.id AS summaryid, d.id AS documentid, d.title AS title,
                                   s.text AS text, s.enhanced_text AS enhanced_text
                            ORDER BY d.id
                            """,
                            parameters,
                        ) or []
                    except Exception as e:
                        log(ERROR, f"DocumentsService.get_category_search_content: error fetching summaries: {e}")
                        result["summary_fetch_failed"] = True

        return result

    def get_content_for_goal(self, content_id_attribute: str, content_ids: List[str], content_label: str, only_if_connected_to: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load goal context content and associated document metadata."""
        result: List[Dict[str, Any]] = []
        allowed_labels = {"Document", "Chapter", "Paragraph", "Summary"}
        allowed_attributes = {"id", "documentid"}

        if content_ids and content_label in allowed_labels and content_id_attribute in allowed_attributes:
            query = f"""
                MATCH (d:{content_label})
                WHERE d.{content_id_attribute} IN $ids
                """
            if only_if_connected_to in allowed_labels:
                query += f"MATCH (d)-[]-(other:{only_if_connected_to})\n"
            query += """
                OPTIONAL MATCH (doss:Dossier)-[:CONTAINS]->(d)
                RETURN d.""" + content_id_attribute + """ AS id, d.name AS name, d.title AS title,
                       d.text AS text, d.enhanced_text AS enhanced_text, d.enhanced_type AS enhanced_type,
                       doss.id AS dossier_id
            """
            result = execute_neo4j_query(query, {"ids": content_ids}) or []

        return result

    def update_content_enhanced_text(self, content_id_attribute: str, content_id: str, content_label: str, enhanced_text: str, enhanced_type: Optional[str] = None) -> bool:
        """Persist enhanced content fields for a document or fragment node."""
        success: bool = False
        allowed_labels = {"Document", "Chapter", "Paragraph", "Summary"}
        allowed_attributes = {"id", "documentid"}

        if content_label in allowed_labels and content_id_attribute in allowed_attributes:
            if enhanced_type is None:
                query = f"""
                    MATCH (d:{content_label})
                    WHERE d.{content_id_attribute} = $id
                    SET d.enhanced_text = $enhanced_text
                    RETURN d.{content_id_attribute} AS id
                """
                parameters = {"id": content_id, "enhanced_text": enhanced_text}
            else:
                query = f"""
                    MATCH (d:{content_label})
                    WHERE d.{content_id_attribute} = $id
                    SET d.enhanced_text = $enhanced_text, d.enhanced_type = $enhanced_type
                    RETURN d.{content_id_attribute} AS id
                """
                parameters = {"id": content_id, "enhanced_text": enhanced_text, "enhanced_type": enhanced_type}

            records = execute_neo4j_query(query, parameters)
            success = bool(records)

        return success

    def delete_temporary_documents(self, sessionid: str) -> None:
        """Delete all temporary ReasoningDocument nodes for the given session that are not linked to any Dossier."""
        execute_neo4j_query("""
            MATCH (s:Session)-[:HAS_DOCUMENT]->(d:ReasoningDocument)
            WHERE s.id = $sessionid AND NOT (d)--(:Dossier)
            DETACH DELETE d
        """, {"sessionid": sessionid})

    def generate_new_document_id(self) -> str:
        """Generate a new unique document ID using atomic counter pattern. Returns ID in format 'docid-N' where N increments atomically."""
        new_id: str = ""

        try:
            query = """
                MERGE (idx:DocumentIdx {id: 'document_index'})
                ON CREATE SET idx.value = 0
                ON MATCH SET idx.value = idx.value + 1
                RETURN idx.value AS newIndex
            """
            records = execute_neo4j_query(query, {})

            if records and len(records) > 0:
                new_index: int = int(records[0].get("newIndex", 0))
                new_id = f"docid-{new_index}"

        except Exception as e:
            log(ERROR, f"DocumentsService.generate_new_document_id: error generating document ID: {e}")

        return new_id



# Singleton instance for dependency injection
_documents_service_instance: Optional[DocumentsInterface] = None

def get_documents_service() -> DocumentsInterface:
    """Get the singleton instance of the documents service."""
    global _documents_service_instance
    if _documents_service_instance is None:
        _documents_service_instance = DocumentsService()
    return _documents_service_instance

def set_documents_service(service: DocumentsInterface) -> None:
    """Set a custom documents service implementation (useful for testing)."""
    global _documents_service_instance
    _documents_service_instance = service


log_end_file(__file__)


if __name__ == '__main__':
    from src.utils.unit_test import init_unit_test
    import json
    from src.utils.log import log_init, log_exit
    init_unit_test()
    log_init()

    try:
        documents_service = get_documents_service()
        
        # Test getting all documents
        all_documents = documents_service.get_all_documents()
        log(INFO, f"All Documents: {len(all_documents) if isinstance(all_documents, list) else 'Error'}")
        
        # Test getting a specific document (if any exist)
        if all_documents and isinstance(all_documents, list) and len(all_documents) > 0:
            first_document_id = all_documents[0].get("id")
            if first_document_id:
                first_document = documents_service.get_document(first_document_id)
                print(f"First Document: {first_document}")
        
        # Test getting documents by status
        ready_documents = documents_service.get_documents_by_status(DocumentStatus.READY)
        log(INFO, f"Ready Documents: {len(ready_documents) if isinstance(ready_documents, list) else 'Error'}")

        # Test getting fragments by IDs
        test_fragment_ids = ["docid-543", "docid-543-c0"]
        fragments = documents_service.get_fragments_by_ids(test_fragment_ids)
        log(INFO, f"Fragments fetched: {json.dumps(fragments, indent=4)}")

        # Test search_documents_by_semantic
        log(INFO, "--- Testing search_documents_by_semantic ---")
        mock_documents = {
            "docid-15": {"docid": "docid-15", "name": "bob_01.txt", "title": "Life of Whiskers", "type": "Text", "summary": "A detailed account of a domestic cat named Whiskers and its daily adventures."},
            "docid-16": {"docid": "docid-16", "name": "bob_02.txt", "title": "Feline Behaviour", "type": "Text", "summary": "Scientific study of cat behaviour, including hunting instincts and social interactions among cats."},
            "docid-17": {"docid": "docid-17", "name": "bob_03.txt", "title": "Cats Through History", "type": "Text", "summary": "How cats have been kept as pets and worshipped throughout human history, from ancient Egypt to today."},
            "docid-18": {"docid": "docid-18", "name": "Quarterly Financial Report", "title": "Q3 Financial Report", "type": "Report", "summary": "Overview of the company's financial results for Q3, including revenue, expenses and EBITDA."},
            "docid-19": {"docid": "docid-19", "name": "Bridge Engineering Manual", "title": "Structural Design Guidelines", "type": "Manual", "summary": "Technical manual covering the structural design and maintenance of suspension bridges."},
            "docid-20": {"docid": "docid-20", "name": "Pasta Recipes", "title": "Italian Cuisine Collection", "type": "Cookbook", "summary": "A collection of traditional Italian pasta recipes with step-by-step cooking instructions."},
        }
        semantic_results = documents_service.search_documents_by_semantic(mock_documents, "documents about cats")
        log(INFO, f"Semantic search returned {len(semantic_results)} document(s): {list(semantic_results.keys())}")
        expected = {"docid-15", "docid-16", "docid-17"}
        found = set(semantic_results.keys())
        missing = expected - found
        unexpected = found - expected
        if missing:
            log(WARNING, f"FAIL — expected documents not returned: {missing}")
        else:
            log(INFO, "PASS — all expected cat documents were returned")
        if unexpected:
            log(WARNING, f"NOTE — additional documents also returned: {unexpected}")

    except Exception as e:
        log(FATAL, f"Error during DocumentsService test: {str(e)}")

    finally:
        log_exit()
