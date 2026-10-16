"""
LLM Client Management Classes

This module contains classes for managing LLM client connections and reservations.
"""

from typing import Dict, Any, Optional

from src.services.runpod.reservations import runpod_reserve_model
from src.utils.log import log, ERROR, DEBUG, WARNING


class LlmClient:
    """Represents a single LLM client instance with provider information."""
    
    def __init__(self, model_provider: Optional[str], endpoint: Optional[str], model_client: Any):
        self.model_provider: Optional[str] = model_provider
        self.endpoint: Optional[str] = endpoint
        self.model_client: Any = model_client

    def get_client(self) -> Any:
        """Get the underlying client instance."""
        if self.model_client is None:
            log(ERROR, f"Client for model provider {self.model_provider} is not initialized.")
            return None
        return self.model_client

    def renew_reservation(self, model_reservation: Optional[dict], agent, purpose: Optional[str]) -> Optional[dict]:
        """Renew the model reservation for this client."""
        # Import here to avoid circular imports
        from src.core.llm.manager import LlmManager
        
        if agent is not None:
            llmconfig = LlmManager.instance().get_llm_model_configuration_for_agent(agent, purpose)
            if llmconfig is None:
                log(ERROR, f"ERROR: No model configuration found for agent {agent.agent_name} and purpose {purpose}.")
                return None
        else:
            llmconfig = LlmManager.instance().get_default_model()
            if llmconfig is None:
                log(ERROR, f"FATAL: No default model found.")
                return None
        
        new_reservation = runpod_reserve_model(model_reservation, llmconfig)
        return new_reservation


class LlmClients:
    """Manages multiple LLM clients for different providers."""
    
    def __init__(self):
        self._clients: Dict[str, LlmClient] = {}        # the key is generated from model_provider and endpoint

    @staticmethod
    def __generate_key(model_provider: str, endpoint: Optional[str]) -> str:
        """Generate a unique key for the client based on provider and endpoint."""
        model_provider = model_provider.split("|")[0]  # strip any suffix after '|'
        if endpoint is not None:
            key = f"{model_provider}|{endpoint}"
        else:
            key = model_provider
        return key

    def add_llm_client(self, model_provider: str, client: LlmClient, endpoint: Optional[str]) -> None:
        """Add a new LLM client for the specified provider."""
        if not isinstance(client, LlmClient):
            raise TypeError(f"Expected LlmClient instance, got {type(client)}")
        
        key: str = LlmClients.__generate_key(model_provider, endpoint)
        self._clients[key] = client

    def get_llm_client(self, model_provider: str, endpoint: Optional[str]) -> Optional[LlmClient]:
        """Get the LLM client for the specified provider."""
        key: str = LlmClients.__generate_key(model_provider, endpoint)
        client = self._clients.get(key, None)
        return client
    
    def renew_client(self, model_provider: str, model_reservation: Optional[dict], agent, purpose: Optional[str], endpoint: Optional[str]) -> Optional[dict]:
        """Renew the reservation for the specified provider's client."""
        key: str = LlmClients.__generate_key(model_provider, endpoint)
        if key in self._clients:
            new_reservation = self._clients[key].renew_reservation(model_reservation, agent, purpose)
        else:
            new_reservation = None
            log(WARNING, f"Client for model provider {model_provider} not found. Cannot renew it.")
        
        return new_reservation

    def remove_client(self, model_provider: str, endpoint: Optional[str]) -> None:
        """Remove the client for the specified provider."""
        key: str = LlmClients.__generate_key(model_provider, endpoint)
        if key in self._clients:
            del self._clients[key]
            log(DEBUG, f"Client for model provider {model_provider}, endpoint {endpoint} has been removed.")
        else:
            log(WARNING, f"Client for model provider {model_provider}, endpoint {endpoint} not found. Cannot remove it.")
