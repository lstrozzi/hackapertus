"""
Endpoint LLM Provider Implementation

This provider connects to OpenAI-compatible endpoints.
"""

from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)


import os
from typing import TYPE_CHECKING, Any, Optional
from openai import OpenAI

from src.core.llm.interface import LlmInterface
from src.core.llm.client import LlmClient
from src.utils.log import log, ERROR, FATAL

if TYPE_CHECKING:
    from src.core.agents.Agent import Agent


class LlmEndpoint(LlmInterface):
    """OpenAI-compatible endpoint LLM provider implementation."""
    
    def __init__(self):
        super().__init__()
        self._model_provider = "endpoint"

    def _prepare_model_client(self, agent: Optional['Agent'] = None, purpose: Optional[str] = None, endpoint: Optional[str] = None) -> Any:
        """Prepare and return the Groq model client instance."""
        self._api_key = self.llmconfig.get("token") if self.llmconfig else None
            
        if not self._api_key:
            # try to get the API key from the environment variable
            self._api_key = os.getenv("OPENAI_API_KEY")
            if not self._api_key:
                log(FATAL, "FATAL ERROR: no OPENAI_API_KEY found")
                return None
        
        endpoint = self._get_endpoint(agent=agent, purpose=purpose, endpoint=endpoint)
        if endpoint is not None:
            openai_client = OpenAI(api_key=self._api_key, base_url=endpoint)
        else:
            openai_client = OpenAI(api_key=self._api_key)

        return openai_client

log_end_file(__file__)
