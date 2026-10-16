"""
LLM Core Module - Refactored from monolithic Llm.py

This module provides:
- Type definitions and enums (types.py)
- Configuration constants (config.py) 
- Shared utility functions (utils.py)
- Cache operations (cache.py)
- Core LLM classes (manager.py, client.py, interface.py, embeddings.py)
- Provider implementations (providers/)
- Initialization function (init.py)
"""

from .agent_rag_settings import AgentRagSettings, reasoning_levels, json_schema_tool_call_response
from .config import *
from .utils import (
    adjust_endpoint_to_docker_format,
    num_tokens_from_string,
    escape_characters_inside_json_strings,
    get_json_from_text,
    calculate_key,
    verify_formal_and_actual_dictionary
)
from .utils_yaml import get_yaml_from_text
from .cache import save_value_into_cache, get_value_from_cache

# Core LLM classes
from src.core.llm.manager import LlmManager
from .client import LlmClient, LlmClients
from .interface import LlmInterface
from .embeddings import EmbeddingsManager
from .reranking import RerankingManager
from .init import llm_init

# Provider implementations
from .providers import (
    LlmOpenAI,
    LlmGroq,
    LlmMistral,
    LlmEndpoint,
    LlmHttpRequest,
    LlmThalabus
)

# For backward compatibility, make key exports available at module level
__all__ = [
    # Types
    'AgentRagSettings',
    'reasoning_levels', 
    'json_schema_tool_call_response',
    
    # Config (imported via *)
    'ENABLE_MODEL_HIGH_INTELLIGENCE',
    'GROQ_API_KEY',
    'PLATFORM_API_TOKEN',
    'USE_LLM_CACHE',
    'DEFAULT_ACTIVE_LLM_CONFIGURATION_NAME',
    'MODEL_LOWINTELLIGENCE',
    'MODEL_HIGHINTELLIGENCE',
    'MODEL',
    'EMBEDDINGS_MODEL_OPENAI',
    'EMBEDDINGS_DIMENSION_OPENAI',
    'MAX_EMBEDDINGS_TOKENS',
    'MAX_COMPLETION_TOKENS',
    'MAX_COMPLETION_TOKENS_UNLIMITED',
    'MAX_COMPLETION_TOKENS_LARGE',
    'MAX_COMPLETION_TOKENS_MEDIUM',
    'MAX_COMPLETION_TOKENS_SMALL',
    'MAX_COMPLETION_TOKENS_TINY',
    'MAX_COMPLETION_TOKENS_PLANNING',
    'DEFAULT_TEMPERATURE',
    'DEFAULT_TEMPERATURE_COLD',
    'MODEL_CONFIG_DEFAULT',
    
    # Utils
    'adjust_endpoint_to_docker_format',
    'num_tokens_from_string',
    'escape_characters_inside_json_strings',
    'get_json_from_text',
    'get_yaml_from_text',
    'calculate_key',
    'verify_formal_and_actual_dictionary',
    
    # Cache
    'save_value_into_cache',
    'get_value_from_cache',
    
    # Core LLM classes
    'LlmManager',
    'LlmClient',
    'LlmClients',
    'LlmInterface',
    'EmbeddingsManager',
    'RerankingManager',
    'llm_init',
    
    # Provider implementations
    'LlmOpenAI',
    'LlmGroq',
    'LlmMistral',
    'LlmEndpoint',
    'LlmHttpRequest',
    'LlmThalabus'
]
