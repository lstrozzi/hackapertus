"""
OpenAI LLM Provider Implementation
"""

from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)


import os
import time
import copy
from typing import TYPE_CHECKING, Any, List, Union, Dict, Tuple, Optional
from openai import OpenAI

from src.core.base.enums import AgentReasoningEnum
from src.core.base.llm_response import LlmResponse
from src.core.llm.config import MAX_COMPLETION_TOKENS
from src.core.llm.interface import LlmInterface
from src.core.llm.utils import adjust_endpoint_to_docker_format
from src.utils.log import log, ERROR, FATAL

if TYPE_CHECKING:
    from src.core.agents.Agent import Agent
    from src.core.thalaplan.Plan import Plan
    from src.core.thalaplan.goal.Goal import Goal


class LlmOpenAI(LlmInterface):
    """OpenAI LLM provider implementation."""
    
    def __init__(self):
        super().__init__()
        self._model_provider = "openai"

    def _get_endpoint(self, agent: 'Agent' = None, purpose: str = None, model: Optional[str] = None, endpoint: Optional[str] = None) -> str:
        """Get the endpoint URL for the given agent and purpose."""
        if endpoint is not None:
            return endpoint

        if self._endpoints is not None:
            # there's only one endpoint configured, return it directly
            return self._endpoints
        
        # Import here to avoid circular imports
        from src.core.llm.manager import LlmManager
        if agent is not None:
            llm_model_config = LlmManager.instance().get_llm_model_configuration_for_agent(agent, purpose)
        else:
            llm_model_config = LlmManager.instance().get_llm_model_configuration_by_model_name(model, self._model_provider)
        
        if "endpoint" in llm_model_config and llm_model_config["endpoint"] is not None and len(llm_model_config["endpoint"]) > 0:
            self._endpoints = llm_model_config["endpoint"]
    
        return self._endpoints

    def _prepare_model_client(self, agent: 'Agent' = None, purpose: str = None, endpoint: Optional[str] = None) -> Any:
        """Prepare and return the OpenAI model client instance."""
        # Get API key from environment if not in config
        self._api_key = self.llmconfig.get("token") if self.llmconfig else None
        if not self._api_key:
            # try to get the API key from the environment variable
            self._api_key = os.getenv("OPENAI_API_KEY")
            if not self._api_key:
                log(FATAL, "FATAL ERROR: no OPENAI_API_KEY found")
                return None
        
        # Create client with or without custom endpoint
        endpoint = self._get_endpoint(agent=agent, purpose=purpose, endpoint=endpoint)
        if endpoint:
            openai_client = OpenAI(api_key=self._api_key, base_url=endpoint)
        else:
            openai_client = OpenAI(api_key=self._api_key)
        
        return openai_client

    def _fix_response_format_for_provider(self, response_format: Dict) -> Dict:
        """
        Fix the response format for OpenAI peculiarities:
        - 'required' is required to be supplied and to be an array including every key in properties, including those prefixed as "optional_" which are logically optional.

        Input arguments:
        - response_format: original response format dictionary, in the format:
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "...",
                    "schema": { ... }
                }
            }

        """
        if response_format is None:
            return None
        
        def fix_required_fields(schema: Dict) -> None:
            """Recursively ensure 'required' contains all properties at every level.
            
            OpenAI's strict mode requires ALL properties to be in the 'required' array,
            even those prefixed with 'optional_' (which are only semantically optional
            for our code logic, but schema-wise must be marked as required for OpenAI).
            """
            if not isinstance(schema, dict):
                return
            
            # If this schema has properties, ensure required includes all of them
            if "properties" in schema and isinstance(schema["properties"], dict):
                all_property_keys = list(schema["properties"].keys())
                if all_property_keys:
                    # ALWAYS set required to include ALL properties for OpenAI strict mode
                    # This overrides any existing required array to ensure compliance
                    schema["required"] = all_property_keys
                
                # Recursively fix nested schemas
                for prop_value in schema["properties"].values():
                    if isinstance(prop_value, dict):
                        fix_required_fields(prop_value)
            
            # Handle other schema keywords that might contain nested schemas
            if "items" in schema and isinstance(schema["items"], dict):
                fix_required_fields(schema["items"])
            
            if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
                fix_required_fields(schema["additionalProperties"])
        
        fixed_response_format = copy.deepcopy(response_format)
        
        # Fix the schema inside json_schema
        if "json_schema" in fixed_response_format and "schema" in fixed_response_format["json_schema"]:
            fix_required_fields(fixed_response_format["json_schema"]["schema"])
        
        return fixed_response_format

    def _get_reasoning_prompt(self, llm_configuration:dict, reasoning_settings:AgentReasoningEnum) -> Optional[str]:
        reasoning_prompt = None

        if llm_configuration is not None:
            if "is_reasoning" in llm_configuration and llm_configuration["is_reasoning"]:
                if "model" in llm_configuration:
                    model_name = llm_configuration["model"]
                    if model_name.startswith("gpt-oss"):
                        # gpt-oss has specific reasoning prompts, for each setting
                        if reasoning_settings == AgentReasoningEnum.FAST:
                            reasoning_prompt = "Use quick and efficient reasoning effort to arrive at conclusions."
                        elif reasoning_settings == AgentReasoningEnum.SIMPLE:
                            reasoning_prompt = "Use medium reasoning effort to arrive at conclusions."
                        elif reasoning_settings == AgentReasoningEnum.COMPLEX:
                            reasoning_prompt = "Use high reasoning effort to arrive at conclusions."
                        else:
                            log(ERROR, f"Unknown reasoning settings: {reasoning_settings}")

        return reasoning_prompt

    # Override base class to provide simplified version (OpenAI doesn't modify messages)
    def _prepare_reasoning_effort(self, agent: 'Agent', purpose: str, maximum_reasoning_required: AgentReasoningEnum, messages: Optional[List[Dict]] = None) -> Tuple[Optional[str], Optional[AgentReasoningEnum]]:
        """Prepare reasoning effort if applicable.
        
        Args:
            agent: The agent making the request
            purpose: Purpose of the LLM call
            maximum_reasoning_required: Maximum reasoning level required
            messages: Not used by OpenAI provider
            
        Returns:
            Tuple of (reasoning_effort_string, effective_reasoning_level)
        """
        # Call base class but don't pass messages (OpenAI doesn't need to modify them)
        return super()._prepare_reasoning_effort(agent, purpose, maximum_reasoning_required, None)

    def _prepare_request_params(self, model: str, messages: List[Dict], max_completion_tokens: int, temperature: float) -> Dict:
        """Prepare the parameters for the OpenAI API request.
        
        Args:
            model: Model name to use
            messages: List of messages
            max_completion_tokens: Maximum tokens for completion
            temperature: Temperature setting
            
        Returns:
            Dictionary containing the request parameters
        """
        if temperature is None:
            temperature = TEMPERATURE_DEFAULT

        params = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature
        }
        return params

    # Inherited from base class: _extract_token_usage()
    # Extracts token usage from ChatCompletion objects using base implementation

    # Inherited from base class: _extract_thinking_message()
    # Returns None for OpenAI as thinking/reasoning is not disclosed by their API

    def _client_chat_completions(
        self,
        messages: List[Dict],
        agent: 'Agent' = None,
        plan: 'Plan' = None,
        goal: 'Goal' = None,
        purpose: str = None,
        tools: list = None,
        model: str = None,
        model_provider: str = None,
        endpoint: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        temperature: float = None,
        response_format: Dict = None,
        structured_output: object = None,
        maximum_reasoning_required: AgentReasoningEnum = None) -> Optional[LlmResponse]:
        """OpenAI chat completions implementation."""

        client = self._get_client(agent=agent, purpose=purpose, model=model, model_provider="openai", endpoint=endpoint)
        if client is None:
            return None

        # response format fixing, because OpenAI has some peculiarities
        response_format_fixed = self._fix_response_format_for_provider(response_format)

        # Prepare reasoning effort if applicable
        _, effective_reasoning = self._prepare_reasoning_effort(agent, purpose, maximum_reasoning_required)

        # Get model parameters
        model_parameters = self._get_model_parameters(model)
        model_name = model_parameters.get("model", model)

        start_ts = time.perf_counter()
        base_llm_response = super()._client_chat_completions(
            messages=messages,
            agent=agent,
            plan=plan,
            goal=goal,
            purpose=purpose,
            tools=tools,
            model=model_name,
            model_provider="openai",
            endpoint=endpoint,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            response_format=response_format_fixed,
            structured_output=structured_output,
            maximum_reasoning_required=effective_reasoning)
        end_ts = time.perf_counter()
        execution_time_msec = (end_ts - start_ts) * 1000.0
        
        # Track LLM performance in goal if provided
        if goal is not None:
            goal.add_perf_llm_ms(execution_time_msec)

        if base_llm_response is None:
            return None

        # base_llm_response is an LlmResponse; return it (it already contains tokens/thinking info)
        return base_llm_response


log_end_file(__file__)
