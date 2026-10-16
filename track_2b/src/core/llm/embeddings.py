"""
EmbeddingsManager module for handling text embeddings.
Provides text embedding generation using various providers (OpenAI, custom endpoints).
"""

from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)


import time
import inspect
import threading
import traceback
import json
import requests
from typing import TYPE_CHECKING, List, Union

# Import necessary components from the llm module
from src.core.llm.config import (
    EMBEDDINGS_DIMENSION_OPENAI, EMBEDDINGS_MODEL_OPENAI,
    CLIENT_CALL_INITIAL_SLEEP_TIME_SEC, MAX_CLIENT_CALLS_ATTEMPTS, 
    MAX_EMBEDDINGS_TOKENS, USE_LLM_CACHE
)
from src.utils.log import FATAL, log, ERROR, WARNING, INFO, DEBUG, find_goal_in_stack_frame
from src.core.llm.cache import get_value_from_cache, save_value_into_cache
from src.core.llm.utils import num_tokens_from_string
from src.core.base.InputOutput import send_error_to_blazor

if TYPE_CHECKING:
    from src.core.agents.Agent import Agent


class EmbeddingsManager:
    """
    Singleton manager for text embeddings.
    Handles embedding generation using various providers including OpenAI and custom endpoints.
    """
    
    _instance: 'EmbeddingsManager' = None
    _embedding_provider: str
    _embeddings_dimension: int
    _embeddings_custom_urls: List[str] = None
    _custom_url_index: int = 0
    _url_lock: threading.Lock = threading.Lock()

    @classmethod
    def instance(cls) -> 'EmbeddingsManager':
        """Get the singleton instance of EmbeddingsManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the EmbeddingsManager instance."""
        self.update_embeddings_configuration()
    
    def _get_embeddings_openai(self, text: Union[str, List[str]], agent: 'Agent' = None, purpose: str = None) -> List[float]:
        """
        Get embeddings from OpenAI API.
        
        Args:
            text: Text to embed
            agent_name: Agent name for client management
            purpose: Purpose for client management
            
        Returns:
            List of embedding floats
        """
        # Lazy import to avoid circular dependencies
        from src.core.llm.manager import LlmManager
        
        client = LlmManager.instance().get_client("openai", endpoint=None, agent=agent, purpose=purpose, model=EMBEDDINGS_MODEL_OPENAI)
        if client is None:
            log(FATAL, "OpenAI client is None. Cannot get embeddings.")
            return None
        
        embedding = client.embeddings.create(input=[text], model=EMBEDDINGS_MODEL_OPENAI).data[0].embedding
        return embedding

    def _get_embeddings_custom(self, text: Union[str, List[str]], agent: 'Agent' = None, purpose: str = None) -> List[float]:
        """
        Get embeddings from custom API endpoint with round-robin load balancing.
        
        Args:
            text: Text to embed
            agent_name: Agent name (unused for custom)
            purpose: Purpose (unused for custom)
            
        Returns:
            List of embedding floats or None on error
        """
        if not self._embeddings_custom_urls:
            log(ERROR, "No custom embedding URLs configured.")
            return None

        data = None
        embedding = None
        if isinstance(text, list):
            input_texts = text
        else:
            input_texts = [text]

        # Round-robin: pick next URL in a thread-safe manner
        with self._url_lock:
            url_index = self._custom_url_index % len(self._embeddings_custom_urls)
            self._custom_url_index += 1
        primary_url = self._embeddings_custom_urls[url_index]

        # Try primary URL first, then failover to others
        urls_to_try = [primary_url] + [
            u for u in self._embeddings_custom_urls if u != primary_url
        ]

        last_exception: Exception = None
        for attempt_url in urls_to_try:
            try:
                response = requests.post(
                    attempt_url,
                    headers={"Content-Type": "application/json"},
                    json={"input": input_texts},
                    timeout=30
                )
                data = response.json()
                if response.status_code != 200 or not isinstance(data, dict) or len(data) == 0:
                    log(ERROR, f"Custom embeddings API ({attempt_url}) returned an error: {data}")
                    last_exception = None
                    continue
                if not "data" in data or not isinstance(data["data"], list) or len(data["data"]) == 0:
                    log(ERROR, f"Custom embeddings API ({attempt_url}) returned an invalid response: {data}")
                    send_error_to_blazor(f"Custom embeddings API ({attempt_url}) returned an invalid response: {data}")
                    last_exception = None
                    continue

                embedding = data["data"][0].get("embedding", None)
                if embedding:
                    return embedding
                
                last_exception = None
                continue

            except json.JSONDecodeError as e:
                stack = traceback.format_exc()
                log(ERROR, f"Custom embeddings API ({attempt_url}) returned an invalid JSON response: {e}\nStack trace:\n{stack}")
                last_exception = e
                continue

            except requests.exceptions.RequestException as e:
                stack = traceback.format_exc()
                log(WARNING, f"Custom embeddings API ({attempt_url}) request failed: {e}\nStack trace:\n{stack}")
                last_exception = e
                continue

        # All URLs failed
        if last_exception:
            log(ERROR, f"All custom embedding URLs failed. Last error: {last_exception}")
            send_error_to_blazor(f"All custom embedding URLs failed. Last error: {last_exception}")
        return None

    def _get_embeddings_no_cache(self, text: Union[str, List[str]], agent: 'Agent' = None, purpose: str = None) -> List[float]:
        """
        Get embeddings without cache, with retry logic.
        
        Args:
            text: Text to embed
            agent_name: Agent name for client management
            purpose: Purpose for client management
            
        Returns:
            List of embedding floats or None on error
        """
        error = True
        attempt = 0
        embedding = None
        sleep_time = CLIENT_CALL_INITIAL_SLEEP_TIME_SEC
        _t_start = time.perf_counter()

        while error and attempt < MAX_CLIENT_CALLS_ATTEMPTS:
            error = True
            try:
                if self._embedding_provider == "openai":
                    embedding = self._get_embeddings_openai(text, agent, purpose)
                elif self._embedding_provider == "custom":
                    embedding = self._get_embeddings_custom(text, agent, purpose)
                else:
                    log(ERROR, f"Unknown embedding provider: {self._embedding_provider}. Cannot get embeddings.")
                    raise ValueError(f"Unknown embedding provider: {self._embedding_provider}")
                error = False

            except Exception as e:
                stack = traceback.format_exc()
                log(ERROR, f"An unexpected error occurred: {e}\nStack trace:\n{stack}")
                send_error_to_blazor(f"OpenAI returned an exception (3): {e}\nStack trace:\n{stack}")
                error = True

            if error:
                if attempt >= MAX_CLIENT_CALLS_ATTEMPTS:
                    log(ERROR, f"get_embeddings(): Maximum number of attempts reached. Returning None.")
                    send_error_to_blazor(f"OpenAI returned an exception: Maximum number of attempts reached.")
                    return None
                log(WARNING, f"get_embeddings(): WARNING: Rate limit exceeded or another error. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                sleep_time += sleep_time
                attempt += 1
        
        _elapsed_ms = (time.perf_counter() - _t_start) * 1000.0
        _goal = find_goal_in_stack_frame(inspect.stack())
        if _goal is not None:
            _goal.add_perf_embeddings_ms(_elapsed_ms)
        return embedding

    def get_embeddings_dimension(self) -> int:
        """Get the dimension of embeddings for the current provider."""
        return self._embeddings_dimension

    def set_embedding_model(self, provider: str, custom_dimension: int = None, custom_url: str = None) -> None:
        """
        Set the embedding model and provider.
        
        Args:
            model: Provider name ("openai" or "custom")
            custom_dimension: Dimension for custom provider
            custom_url: Comma-separated URLs for custom provider (e.g., "url1,url2")
        """
        self._embedding_provider = provider.lower()
        if self._embedding_provider == "openai":
            self._embeddings_dimension = EMBEDDINGS_DIMENSION_OPENAI
            self._embedding_model = EMBEDDINGS_MODEL_OPENAI
            log(INFO, f"Using OpenAI embeddings model with standard dimension {self._embeddings_dimension}")
        elif self._embedding_provider == "custom":
            # Parse comma-separated URLs into a list
            if custom_url:
                self._embeddings_custom_urls = [
                    u.strip() for u in custom_url.split(",") if u.strip()
                ]
            else:
                self._embeddings_custom_urls = []
            self._embeddings_dimension = custom_dimension
            self._embedding_model = None  # No specific model for custom provider, it's automatically used based on the custom URL(s)
            if self._embeddings_custom_urls:
                log(DEBUG, f"Using custom embeddings model with dimension {self._embeddings_dimension} and {len(self._embeddings_custom_urls)} URL(s): {', '.join(self._embeddings_custom_urls)}")
            else:
                log(ERROR, "Custom embeddings model configured but no URLs provided.")
        else:
            log(ERROR, f"Unknown embedding model: {provider}")

    def update_embeddings_configuration(self) -> None:
        """Update embeddings configuration from LlmManager settings."""
        # Lazy import to avoid circular dependencies
        from src.core.llm.manager import LlmManager
        
        embeddings_settings = LlmManager.instance().get_embeddings_settings()
        if embeddings_settings:
            provider = embeddings_settings.get("provider", "openai").lower()
            embedding_size = embeddings_settings.get("embedding_size", None)
            custom_url = embeddings_settings.get("custom_url", None)
            self.set_embedding_model(provider, embedding_size, custom_url)
        else:
            self.set_embedding_model("openai")

    def get_embeddings_multi(self, texts: List[str], cache: bool = USE_LLM_CACHE) -> List[List[float]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            cache: Whether to use caching
            
        Returns:
            List of embedding lists
        """
        embeddings = []
        for text in texts:
            emb = self.get_embeddings(text, cache=cache)
            embeddings.append(emb)
        return embeddings
    
    def get_embeddings(self, text: Union[str, List[str]], cache: bool = USE_LLM_CACHE, agent_name: str = None, purpose: str = None) -> List[float]:
        """
        Get embeddings for text with caching support.
        
        Args:
            text: Text to embed (string or list)
            cache: Whether to use caching
            agent_name: Agent name for client management
            purpose: Purpose for client management
            
        Returns:
            List of embedding floats or None on error
        """
        if text is None:
            return None
        if isinstance(text, list):
            return self.get_embeddings_multi(text, cache)

        text = text.replace("\n", " ")

        while num_tokens_from_string(text) > MAX_EMBEDDINGS_TOKENS * 0.9:
            # The embedding model has a limit of MAX_EMBEDDINGS_TOKENS tokens, so we need to reduce the text length
            # We take a 10% margin to avoid the limit, because the tokenization may vary slightly between num_tokens_from_string and the actual tokenization
            text = text[:int(len(text) * 0.8)]

        if len(text) == 0:
            # If the text is empty, return a zero vector
            return [0.0] * self._embeddings_dimension

        embedding = None
        if cache:
            # Check if the embedding is already in the cache
            # The key is a combination of the model and the text, to avoid collisions with different models or texts
            if self._embedding_model is not None:
                key = f"EMBEDDINGS({self._embedding_model}): {text}"
            else:
                key = f"EMBEDDINGS: {text}"
            embedding, hash, guid, thinking_message = get_value_from_cache(key)
            if embedding:
                points = "..." if len(key) > 80 else ""
                log(DEBUG, f"using cache for key: {key[:80]}{points}.")
            else:
                points = "..." if len(key) > 80 else ""
                log(DEBUG, f"key not found in the cache. Generating response: {key[:80]}{points}.")
                embedding = self._get_embeddings_no_cache(text)
                save_value_into_cache(key, embedding, thinking_message=None, hash=hash)
        else:
            # No cache, just get the embeddings
            embedding = self._get_embeddings_no_cache(text)

        return embedding


log_end_file(__file__)

if __name__ == "__main__":
    from src.utils.log import log_init, log_exit
    log_init()

    try:
        text = "Hello, world!"
        em = EmbeddingsManager.instance()
        emb = em.get_embeddings(text)
        dim = em.get_embeddings_dimension()
        log(INFO, f"Standard embedding dimension: {dim}")
        log(INFO, f"Embedding for '{text}' dimension is: {len(emb)}")
        log(INFO, f"First 5 values: {emb[:5] if emb else None}")
        if len(emb) != dim:
            log(ERROR, f"Error: Embedding dimension {len(emb)} does not match expected {dim}")

    except Exception as e:
        log(FATAL, f"Fatal error in EmbeddingsManager test: {e}")
        
    finally:
        log_exit()
