from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

"""Neo4j index catalog and bootstrap for Thalabus.

Single source of truth for all Neo4j index definitions.
Call ensure_all_indexes(dimension) to create or verify every index — it is
idempotent and safe to call on every startup.

Individual services (hints_service, ontology_service) retain their own
lazy-creation wrappers for on-demand scenarios; their index definitions
must match the catalog entries here.

Index creation sites (for reference):
  - Startup:       parameters._create_vector_indexes() -> ensure_all_indexes()
  - Maintenance:   parameters.perform_database_maintenance() uses RANGE_INDEXES
  - Doc ingestion: documents_service.create_all_required_indexes() -> ensure_all_indexes()
  - Coins:         CoinIngester.prepare_neo4j_database() -> ensure_all_indexes()
  - Lazy (hints):  hints_service.create_hints_vector_indexes()
  - Lazy (onto):   ontology_service.create_semantic_search_index()
"""

import traceback
from src.core.base.InputOutput import execute_neo4j_query
from src.utils.log import log, INFO, WARNING, ERROR

# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

# (index_name, node_label, property_name)
VECTOR_INDEXES = [
    ("hints",                  "LlmHint",   "embedding"),
    ("typical_rag",            "Chapter",   "embedding"),
    ("chapterparagraph",       "Paragraph", "embedding"),
    ("summary",                "Summary",   "embedding"),
    ("keywords",               "Keyword",   "embedding"),
    ("hypothetical_questions", "Question",  "embedding"),
    ("ontology_entity",        "Entity",    "embedding"),
]

# (index_name, node_label, properties_list)
FULLTEXT_INDEXES = [
    ("chapters_fulltext", "Chapter", ["text"]),
    ("keywordsTextIndex",  "Keyword", ["text"]),
]

# (node_label, property_name, index_name)
RANGE_INDEXES = [
    ("CacheEntry",        "hash",        "cache_entry_hash"),
    ("CacheEntry",        "guid",        "cache_entry_guid"),
    ("CacheEntry",        "guid_last_exec", "cache_entry_guid_last_exec"),
    ("Capability",        "id",          "capability_id_idx"),
    ("Category",          "text",        "category_text"),
    ("Category",          "id",          "category_id"),
    ("Chapter",           "id",          "chapter_id"),
    ("Chapter",           "documentid",  "chapter_documentid"),
    ("Chapter",           "partition",   "chapter_partition"),
    ("Community",         "id",          "community_id_idx"),
    ("Community",         "isProtected", "community_isProtected_idx"),
    ("Dataset",           "id",          "dataset_id"),
    ("Document",          "id",          "document_id"),
    ("Document",          "documentid",  "document_documentid"),
    ("Document",          "isDeleted",   "document_isDeleted_idx"),
    ("Document",          "isProtected", "document_isProtected_idx"),
    ("Document",          "partition",   "document_partition"),
    ("DocumentTemplate",  "id",          "document_template_id"),
    ("DocumentTemplate",  "templateid",  "document_template_templateid"),
    ("Dossier",           "id",          "dossier_id"),
    ("Dossier",           "isProtected", "dossier_isProtected_idx"),
    ("Entity",            "id",               "entity_id"),
    ("Entity",            "is_computed_view", "entity_is_computed_view_idx"),
    ("Entity",            "is_diagram",       "entity_is_diagram_idx"),
    ("Entity",            "is_relationship",  "entity_is_relationship_idx"),
    ("Entity",            "is_structure",     "entity_is_structure_idx"),
    ("Group",             "id",          "group_id_idx"),
    ("Image",             "id",          "image_id"),
    ("KeywordCategory",   "text_en",     "keyword_category_text_en"),
    ("KeywordCategory",   "id",          "keyword_category_id"),
    ("LlmHint",           "key",         "llm_hint_key"),
    ("LlmHint",           "prompt",      "llm_hint_prompt"),
    ("Ontology",          "isProtected", "ontology_isProtected_idx"),
    ("Paragraph",         "id",          "paragraph_id"),
    ("Paragraph",         "documentid",  "paragraph_documentid"),
    ("Paragraph",         "partition",   "paragraph_partition"),
    ("ReasoningDocument", "id",          "reasoning_document_id"),
    ("Session",           "id",          "session_id"),
    ("Summary",           "id",          "summary_id"),
    ("Summary",           "documentid",  "summary_documentid"),
    ("Summary",           "partition",   "summary_partition"),
    ("TranslationCache",  "key",         "translation_cache_key"),
    ("User",              "id",          "user_id"),
]

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def ensure_all_indexes(embeddings_dimension: int) -> None:
    """Create or verify all Neo4j indexes. Idempotent — safe to call on every startup."""
    try:
        _ensure_vector_indexes(embeddings_dimension)
        _ensure_fulltext_indexes()
        _ensure_range_indexes()
        log(INFO, "neo4j_indexes: all indexes verified.")
    except Exception as e:
        log(ERROR, f"neo4j_indexes: unexpected error in ensure_all_indexes: {e}\n{traceback.format_exc()}")


def _ensure_vector_indexes(dimension: int) -> None:
    for index_name, node_label, property_name in VECTOR_INDEXES:
        try:
            execute_neo4j_query(
                f"""
                CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
                FOR (n: {node_label}) ON (n.{property_name})
                OPTIONS {{ indexConfig: {{
                    `vector.dimensions`: $dimension,
                    `vector.similarity_function`: 'cosine'
                }} }}
                """,
                {"dimension": dimension},
                ignore_warnings=True,
            )
        except Exception as e:
            log(WARNING, f"neo4j_indexes: could not create vector index '{index_name}' on {node_label}.{property_name}: {e}")


def _ensure_fulltext_indexes() -> None:
    for index_name, node_label, properties in FULLTEXT_INDEXES:
        properties_str = ", ".join(f"n.{p}" for p in properties)
        try:
            execute_neo4j_query(
                f"""
                CREATE FULLTEXT INDEX `{index_name}` IF NOT EXISTS
                FOR (n:{node_label}) ON EACH [{properties_str}]
                OPTIONS {{ indexConfig: {{ `fulltext.eventually_consistent`: true }} }}
                """,
                {},
                ignore_warnings=True,
            )
        except Exception as e:
            log(WARNING, f"neo4j_indexes: could not create fulltext index '{index_name}' on {node_label}.{properties}: {e}")


def _ensure_range_indexes() -> None:
    for node_label, property_name, index_name in RANGE_INDEXES:
        try:
            execute_neo4j_query(
                f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{node_label}) ON (n.{property_name})",
                None,
                ignore_exception=True,
                ignore_warnings=True,
            )
        except Exception as e:
            log(WARNING, f"neo4j_indexes: could not create range index '{index_name}' on {node_label}.{property_name}: {e}")


def ensure_innovator_indexes() -> None:
    """Create range indexes used by the Innovator knowledge-graph import. Idempotent."""
    innovator_indexes = [
        entry for entry in RANGE_INDEXES
        if entry[0] == "Entity" and entry[1] in (
            "is_computed_view", "is_diagram", "is_relationship", "is_structure",
        )
    ]
    for node_label, property_name, index_name in innovator_indexes:
        try:
            execute_neo4j_query(
                f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{node_label}) ON (n.{property_name})",
                None,
                ignore_exception=True,
                ignore_warnings=True,
            )
            log(INFO, f"neo4j_indexes: ensured index '{index_name}' on {node_label}.{property_name}")
        except Exception as e:
            log(WARNING, f"neo4j_indexes: could not create index '{index_name}' on {node_label}.{property_name}: {e}")


log_end_file(__file__)
