import os
import sys

from typing import Optional
from src.utils.system import get_os_type

# flask config
PORT: int = 5010
ADDR: str = '0.0.0.0'
USE_REAL_SSL_CERTIFICATE: bool = False

# Users, Sessions
GLOBAL_SESSION_ID: str = 'global'
GLOBAL_USER_ID: str = 'global'

# ACL
ACL_MAX_INHERITANCE_DEPTH: int = 10  # max CONTAINS/DEFINES inheritance levels for every acl_* Cypher fragment; must match Constants.AclMaxInheritanceDepth in C#

# LLM config
AUTO_LANGUAGE: str = 'auto'
TOP_N_HINTS: int = 5
MIN_HINT_SCORE: float = -1    # minimum score for a hint to be considered useful
MAX_RAG_MESSAGE_LENGTH_DEFAULT: int = 50000  # maximum length (in characters) of the RAG message to avoid excessive token usage
MAX_LOADED_DOCUMENT_LENGTH_DEFAULT: int = 50000  # maximum length (in characters) of loaded documents to avoid excessive token usage
WARN_IF_TRUNCATING_THRESHOLD_DEFAULT: float = 0.5  # warn only when a truncated fragment has a ranking above this threshold

# Blazor config
DEFAULT_BLAZOR_HOSTNAME: str = 'host.docker.internal' if sys.platform != "win32" and sys.platform != "darwin" else '127.0.0.1'
BLAZOR_HOSTNAME: str = os.getenv('BLAZOR_HOSTNAME', DEFAULT_BLAZOR_HOSTNAME)
BLAZOR_PORT: int = int(os.getenv('BLAZOR_PORT', '5210'))
BLAZOR_PROTOCOL: str = os.getenv('BLAZOR_PROTOCOL', 'http')
CONTAINERUTILITY_HOSTNAME: str = os.getenv('CONTAINERUTILITY_HOSTNAME', '127.0.0.1')
CONTAINERUTILITY_PORT: int = int(os.getenv('CONTAINERUTILITY_PORT', '5100'))
CONTAINERUTILITY_INTERNAL_LISTEN_PORT: int = int(os.getenv('CONTAINERUTILITY_INTERNAL_LISTEN_PORT', str(CONTAINERUTILITY_PORT)))
CONTAINERUTILITY_PROTOCOL: str = os.getenv('CONTAINERUTILITY_PROTOCOL', 'http')
UTILS_BASE_URI: str = f"{CONTAINERUTILITY_PROTOCOL}://{CONTAINERUTILITY_HOSTNAME}:{CONTAINERUTILITY_PORT}"

# Service-internal JWT issued by ThalabusCore at container startup, used for this container's own calls to the Utils API
THALABUS_PYTHON_INTERNAL_SERVICE_TOKEN: str = os.getenv('THALABUS_PYTHON_INTERNAL_SERVICE_TOKEN', '')

# Skills config
BASE_PATH:str = "/app" if get_os_type() == "Linux" else "."

# Document repository
IMG_URI_PREFIX: str = "img://"
DIAG_URI_PREFIX: str = "diag://"
DIAGRAM_FOLDER_NAME: str = "diagrams"

# reasoning and deep reasoning
ENABLE_STRICT_TOOLS: bool = True
DEFAULT_REASONING_IN_DEBUG: str = "COMPLEX"
DEFAULT_FLAVOUR_IN_DEBUG: Optional[str] = None
MAX_DEEP_REASONING_REVIEW_CYCLES: int = 2
MAX_PARALLELISM: int = min(int(os.cpu_count() * 2.5), 12)
IS_DEBUG: bool = sys.gettrace() is not None

# agents
AGENT_ASSISTANT_REMINDER_ENABLE: bool = False
AGENT_ASSISTANT_PLACEHOLDER_ENABLE: bool = False  # set to False to disable placeholder-based long-generation; feature is configurable via assistant_subagents_enable setting

# document management
DOCUMENT_ID_PREFIX: str = "docid-"

# RAG
ENABLE_PATH_RAG_DEFAULT: bool = False
ENABLE_AGENT_REVIEWER: bool = False
ENABLE_RAG_QUERY_SPLITTER: bool = False
MAX_RAG_OUTPUT_FRAG: int = 1000
MAX_CHAPTER_FETCH: int = 1000           # Max chapters fetched per thalabus-query result (fed to reranker; higher = better recall for large docs)
MAX_FRAGMENTS_PER_DOC: int = 25         # Per-document fragment cap applied before global sort (prevents a single large doc from dominating)
MIN_RAG_SCORE_FRAG: float = 0.4

# Reranking - Fragment level (initial filtering and output limits)
RERANKER_DATASET_SUFFIX: str = "_reranker"
MAX_RERANKING_INPUT_FRAG: int = 100     # Max fragments processed in each reranking batch
MAX_RERANKING_PARALLELISM: int = 50     # Max concurrent HTTP calls to the reranker (I/O-bound; kept separate from general MAX_PARALLELISM)
RERANKING_REQUEST_TIMEOUT_SECONDS: int = 60  # Maximum time allowed for one reranker HTTP request
MAX_RERANKING_OUTPUT_FRAG: int = 100    # Max fragments to return from reranking
MIN_RERANKING_OUTPUT_FRAG: int = 50     # Min distinct fragments to ensure diversity
MIN_RERANKING_SCORE_FRAG: float = 0.1   # Min score for a fragment to pass reranking filter
MIN_DATASET_CLONE_RANKING: float = 0.1  # Minimum relationship weight retained in assistant dataset clones

# Reranking - Document level (diversity enforcement and final aggregation)
MAX_RERANKING_OUTPUT_DOC: int = 10      # Max documents in final aggregated results
MIN_RERANKING_OUTPUT_DOC: int = 5       # Min distinct documents to ensure diversity
MIN_RERANKING_SCORE_DOC: float = 0.05   # Min score for document-level operations
MIN_SEMANTIC_TEXT_COMBINE_WORDS: int = 3  # For queries with this many or fewer words, combine semantic and text search results

# Reranking - Knowledge-graph node level (kg:// fragments, reranked independently)
MAX_RERANKING_OUTPUT_KG: int = 10       # Max kg:// nodes to keep after reranking
MIN_RERANKING_OUTPUT_KG: int = 3        # Min kg:// nodes guaranteed in output (padded from top-score fallback)

# Knowledge-graph neighbour expansion (broadens AgentRagKB/AgentRagPathKb datasets with adjacent entities)
KG_NEIGHBOR_EXPANSION_MIN_RANK: float = 0.001   # Minimum rank/weight a kg entity must have for its neighbours to be added
KG_NEIGHBOR_WEIGHT_FACTOR: float = 0.8        # Weight multiplier applied to a neighbour when added (relative to its source entity's rank)

# KG hydration and RAG dataset KG augmentation
KG_HYDRATION_LINKED_DOCUMENTS_ENABLED: bool = False  # When False, entity discourse serialization omits <LINKED_DOCUMENT> blocks (ontology_service, pathrag_service, node_serialization_service)
RAG_EXTEND_DOCUMENT_DATASET_WITH_KG_ENTITIES: bool = True  # When True, AgentRAG extends its created dataset with entities linked via IS_DOCUMENTED to the result documents

# LLM relevance gate for thalabus_query (T33)
LLM_FILTER_MAX_CONTENT_CHARS: int = 400000   # Abort the gate (no silent truncation) when a candidate's evaluable content exceeds this many characters
LLM_FILTER_MAX_PROMPT_CHARS: int = 8000      # Reject the gate arguments as invalid when the relevance prompt exceeds this many characters
LLM_FILTER_MAX_REASONING_CHARS: int = 8000   # Clamp the reasoning text appended to the two-call YES/NO request to this many characters

# Multi-language search/reranking (addresses cross-language retrieval issues)
MULTILANG_SEARCH_ENABLED: bool = True    # Enable multi-language semantic search and reranking
MULTILANG_LANGUAGES: list = ['en', 'de', 'fr', 'it']  # Languages for multi-language search

# Partitioned semantic search (addresses recall failures in large corpora with >50k documents)
# When set, the global ANN vector index is replaced by exhaustive constrained search over batches
# of documents. This guarantees recall at the cost of higher latency. Set to None to disable.
SEMANTIC_SEARCH_PARTITION_SIZE: Optional[int] = 5000

# ACL
ACL_ENFORCEMENT_ENABLED: bool = True   # Set True once ACL graph (Tasks 3–10) is fully populated

# ACL batch-retry loop
ACL_BATCH_INITIAL_LIMIT: int = 150   # First-pass candidate count — matches TOP_N_RAG
ACL_BATCH_MAX_LIMIT: int    = 1500   # Maximum candidate pool before giving up
ACL_BATCH_MULTIPLIER: int   = 3      # Factor by which the limit grows each retry
ACL_BATCH_MAX_RETRIES: int  = 3      # Hard cap on loop iterations

# ontologies and knowledge-graph semantic search
SEMANTIC_SEARCH_LIMIT: int = 25
SEMANTIC_SEARCH_NEAREST_NEIGHBOURS: int = 50
SEMANTIC_SEARCH_CANDIDATE_MULTIPLIER: int = 2

# RAG chunking sizes
CHAPTER_TOKENS_SIZE: int = 1200
PARAGRAPH_TOKENS_SIZE: int = 300

# runpod reservations
RESERVATION_MINUTES: int = 15

# Web Searches
TAVILY_WEBSEARCH_SCORE_THRESHOLD: float = 0.20     # very low threshold to filter out only the least relevant results

# Training data commands that operate on the current plan_guid and must NOT trigger a new guid in AgentUser
# (they need to reference cache entries from the previous interaction)
COMMANDS_PRESERVE_GUID: list = ["/good", "/bad", "/restart"]