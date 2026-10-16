"""
LLM Configuration constants and settings.
"""

import os
import ast
from src.core.base.enums import AgentReasoningEnum


# Configuration flags
ENABLE_MODEL_HIGH_INTELLIGENCE = ast.literal_eval(os.getenv('ENABLE_MODEL_HIGH_INTELLIGENCE', 'False'))

# API Keys and tokens
GROQ_API_KEY = os.getenv('GROQ_API_KEY', None)
PLATFORM_API_TOKEN = os.getenv('PLATFORM_API_TOKEN')
if PLATFORM_API_TOKEN is None:
    print("*** FATAL: No API token found in the environment variables! ***")
    exit(1)

# Cache and plan settings
USE_LLM_CACHE = True
DEFAULT_ACTIVE_LLM_CONFIGURATION_NAME = None

# Model configurations
MODEL_LOWINTELLIGENCE = None    # Use default model for low intelligence tasks
MODEL_HIGHINTELLIGENCE = None   # Use default model for high intelligence tasks
MODEL = MODEL_HIGHINTELLIGENCE if ENABLE_MODEL_HIGH_INTELLIGENCE else MODEL_LOWINTELLIGENCE

# Embedding settings
EMBEDDINGS_MODEL_OPENAI = "text-embedding-3-large"
EMBEDDINGS_DIMENSION_OPENAI = 3072
MAX_EMBEDDINGS_TOKENS = 8190

# Token and call limits
MAX_COMPLETION_TOKENS = 32768  # 32k tokens

# Named token-budget constants for chat_completions() calls
MAX_COMPLETION_TOKENS_UNLIMITED = None   # use the model's full context window
MAX_COMPLETION_TOKENS_LARGE    = 30000   # deep-reasoning, large categorisation outputs
MAX_COMPLETION_TOKENS_MEDIUM   = 12000   # summaries, medium-length outputs
MAX_COMPLETION_TOKENS_SMALL    = 4000    # structured JSON, keywords, translation
MAX_COMPLETION_TOKENS_TINY     = 512     # single-value outputs (quality score, title)
MAX_COMPLETION_TOKENS_PLANNING = 20000   # planning agents (AgentPlanner, AgentMicroplanner, AgentToolPreparer)
LLM_REQUEST_TIMEOUT_SECONDS = 300        # maximum time allowed for one direct LLM HTTP request
LLM_PING_TIMEOUT_SECONDS = 5             # maximum time allowed for one LLM availability check
CLIENT_CALL_INITIAL_SLEEP_TIME_SEC = 5
MAX_CLIENT_CALLS_ATTEMPTS = 10
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_FREQUENCY_PENALTY = 0.3
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_REPETITION_PENALTY = 1.2
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3
DEFAULT_TEMPERATURE = 0.15
DEFAULT_TEMPERATURE_COLD = 0.1

# Thinking token budget per reasoning level (for Qwen and compatible models)
# Format: (enable_thinking: bool, budget: int | None, think_sentence: str | None)
#   - enable_thinking=False disables all thinking/reasoning
#   - budget=None means unlimited thinking when enabled
#   - think_sentence: optional guidance to inject as system message when thinking is budgeted
# NONE:    thinking disabled entirely
# MINIMAL: 256 tokens - suitable for quick schema extraction, validation
# FAST:    1024 tokens - short reasoning for simple tasks, categorization
# SIMPLE:  2048 tokens - standard reasoning for typical tasks
# MEDIUM:  4096 tokens - extended reasoning for complex analysis
# COMPLEX: unlimited - full reasoning capability, no budget constraint
# None:    defaults to COMPLEX behavior when no level is specified
THINKING_TOKEN_BUDGET: dict[AgentReasoningEnum | None, tuple[bool, int | None, str | None]] = {
    AgentReasoningEnum.NONE:     (False, None,  None),
    AgentReasoningEnum.MINIMAL:  (True,  256,   None),
    AgentReasoningEnum.FAST:     (True,  1024,  None),
    AgentReasoningEnum.SIMPLE:   (True,  2048,  None),
    AgentReasoningEnum.MEDIUM:   (True,  4096,  None),
    AgentReasoningEnum.COMPLEX:  (True,  None,  None),
    None:                        (True,  None,  None),
}

# When the active reasoning is SIMPLE, models whose "intelligence" (LlmSettings.json) is strictly above this
# threshold run with NONE reasoning. Adaptive-thinking models (e.g. Qwen 3.8) treat reasoning_effort and
# thinking_budget only as hints, so thinking can be disabled only by resolving the level to NONE.
SIMPLE_DISABLES_REASONING_ABOVE_INTELLIGENCE = 2.0

# Display labels for reasoning levels with thinking token budget information
REASONING_LEVEL_DISPLAY: dict[AgentReasoningEnum | None, str] = {
    AgentReasoningEnum.NONE:     "NONE (no thinking)",
    AgentReasoningEnum.MINIMAL:  "MINIMAL (256 tokens)",
    AgentReasoningEnum.FAST:     "FAST (1K tokens)",
    AgentReasoningEnum.SIMPLE:   "SIMPLE (2K tokens)",
    AgentReasoningEnum.MEDIUM:   "MEDIUM (4K tokens)",
    AgentReasoningEnum.COMPLEX:  "COMPLEX (unlimited)",
}

# Model configuration names
MODEL_CONFIG_DEFAULT = "DEFAULT"
# Dataset settings
HF_DATASET_NAME_DEFAULT = "lstrozzi/thalabus-training-data"

# Set default active configuration based on test flags
DEFAULT_ACTIVE_LLM_CONFIGURATION_NAME = MODEL_CONFIG_DEFAULT
