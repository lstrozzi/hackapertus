"""
LLM initialization functions.
Provides initialization logic for LLM and embeddings managers.
"""

from ...utils.log import log, DEBUG, INFO, FATAL


def llm_init() -> None:
    """
    Initialize LLM and Embeddings managers.
    
    This function initializes both the LlmManager and EmbeddingsManager singletons,
    and updates their configurations.
    """
    # Lazy imports to avoid circular dependencies
    from src.core.llm.manager import LlmManager
    from .embeddings import EmbeddingsManager
    from .reranking import RerankingManager
    
    if LlmManager._instance is not None:
        log(DEBUG, "LlmManager is already initialized. Skipping initialization.")
        return
    
    try:
        # Initialize LlmManager
        log(DEBUG, "Initializing LlmManager...")
        llmManager = LlmManager.instance()
        if llmManager is None:
            log(FATAL, "LlmManager.instance() is None. Please check the configuration and the environment.")
            raise RuntimeError("LlmManager.instance() is None. Please check the configuration and the environment.")
            
        log(DEBUG, "Initializing EmbeddingsManager...")
        embeddingsManager = EmbeddingsManager.instance()
        if embeddingsManager is None:
            log(FATAL, "EmbeddingsManager.instance() is None. Please check the configuration and the environment.")
            raise RuntimeError("EmbeddingsManager.instance() is None. Please check the configuration and the environment.")

        log(DEBUG, "Initializing RerankingManager...")
        rerankingManager = RerankingManager.instance()
        if rerankingManager is None:
            log(FATAL, "RerankingManager.instance() is None. Please check the configuration and the environment.")
            raise RuntimeError("RerankingManager.instance() is None. Please check the configuration and the environment.")
            
        log(DEBUG, "Updating LLM Configuration...")
        LlmManager.instance().update_llm_configuration()
        EmbeddingsManager.instance().update_embeddings_configuration()

        log(DEBUG, "LlmManager initialized.")

        # Initialize tools (commented out in original)
        # log(DEBUG, "Initializing tools...")
        # get_filtered_tools(plan=None)
        # log(INFO, "Tools initialized.")
    
    except Exception as e:
        log(FATAL, f"Exception during LlmManager initialization: {e}")

    return