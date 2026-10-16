"""
LLM Interface Abstract Base Class

This module contains the abstract base class for all LLM provider implementations.
"""

from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import os
import re
import json
import random
import threading
import time
import copy
import traceback
import openai
import httpx
import concurrent.futures
from abc import ABC, abstractmethod
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Union, Tuple, Literal
from datetime import datetime

from src.core.base.enums import AgentReasoningEnum
from src.core.base.llm_response import LlmResponse, LlmParameters
from src.core.llm.client import LlmClients, LlmClient
from src.core.llm import (
    get_json_from_text, verify_formal_and_actual_dictionary, 
    calculate_key, save_value_into_cache, get_value_from_cache,
    adjust_endpoint_to_docker_format, MODEL, MODEL_LOWINTELLIGENCE, MODEL_HIGHINTELLIGENCE,
    DEFAULT_TEMPERATURE, CLIENT_CALL_INITIAL_SLEEP_TIME_SEC, DEFAULT_TOP_K, DEFAULT_TOP_P,
    LLM_PING_TIMEOUT_SECONDS,
    MAX_CLIENT_CALLS_ATTEMPTS, USE_LLM_CACHE, json_schema_tool_call_response
)
from src.core.llm.config import DEFAULT_FREQUENCY_PENALTY, DEFAULT_NO_REPEAT_NGRAM_SIZE, DEFAULT_REPETITION_PENALTY, DEFAULT_PRESENCE_PENALTY, SIMPLE_DISABLES_REASONING_ABOVE_INTELLIGENCE
from src.services.runpod.reservations import runpod_reserve_model, runpod_get_endpoint
from src.core.base.InputOutput import send_error_to_blazor
from src.services.cache_entry import get_cache_entry_service
from src.utils.log import log, log_llm_interaction, DEBUG, INFO, LLM_IN, LLM_OUT, WARNING, ERROR, FATAL

if TYPE_CHECKING:
    from src.core.agents.Agent import Agent
    from src.core.thalaplan.goal.Goal import Goal
    from src.core.thalaplan.Plan import Plan

VERIFY_ENDPOINT_SSL_CERTIFICATES = False


class LlmInterface(ABC):
    """Abstract base class for all LLM provider implementations."""
    
    def __init__(self):
        self._clients = LlmClients()
        self._model_provider: Optional[str] = None
        self._model_reservation: Optional[dict] = None
        self._api_key: Optional[str] = None

        # endpoints management for load balancing if multiple specified
        self._endpoints: Optional[list[str]] = None
        self._endpoints_lock = threading.Lock()
        self._endpoints_rotation_index = 0  # Round-robin rotation index (randomized at first init)
        self._endpoints_count = 0
        self._endpoints_in_use: list[int] = []

    @staticmethod
    def _resolve_api_key(token: Optional[str]) -> Optional[str]:
        """Resolve API key, expanding {{ENV_VAR}} placeholders via os.environ.
        
        Supports format: {{ENV_VAR_NAME}} to read environment variables.
        If token is empty or the environment variable is not defined/empty, returns None.
        """
        if not token or not token.strip():
            return None
        match = re.match(r'^\{\{([^}]+)\}\}$', token.strip())
        if match:
            resolved = os.environ.get(match.group(1))
            return resolved if resolved and resolved.strip() else None
        return token

    def _get_client(self, agent: 'Agent', purpose: str, model: Optional[str], model_provider: Optional[str], endpoint: str) -> Any:
        """Get the appropriate client for the given agent and purpose."""
        # Import here to avoid circular imports
        from src.core.llm.manager import LlmManager
        
        if agent is not None:
            agent_config_name = LlmManager.instance().get_agent_config_name(agent.agent_name, purpose)
            client = self._clients.get_llm_client(agent_config_name, endpoint)
        else:
            agent_config_name = model_provider if model_provider is not None else self._model_provider
            client = self._clients.get_llm_client(agent_config_name, endpoint)

        # if no client can be found, use the default client
        if client is None:
            # if no client is found, create a new one
            self._prepare_model(agent, purpose, model=model, model_provider=agent_config_name, endpoint=endpoint)
            client = self._clients.get_llm_client(agent_config_name, endpoint=endpoint)

        if client is None:
            log(FATAL, f"FATAL: No client found for model provider: {self._model_provider}, endpoint: {endpoint}")
            return None

        return client.get_client()
    
    def _get_endpoint(self, agent: Optional['Agent'] = None, purpose: Optional[str] = None, model: Optional[str] = None, endpoint: Optional[str] = None) -> Optional[str]:
        """Get the endpoint URL for the given agent and purpose."""
        if endpoint is not None:
            return endpoint
        
        # Check if endpoints are already initialized (fast path with read lock only for checking)
        if self._endpoints is not None and self._endpoints_count > 0:
            single_endpoint = self._get_endpoint_least_used(agent=agent, purpose=purpose, model=model, endpoint=endpoint)
            return single_endpoint
        
        # Import here to avoid circular imports
        endpoints = None
        from src.core.llm.manager import LlmManager
        if agent is not None:
            llm_model_config = LlmManager.instance().get_llm_model_configuration_for_agent(agent, purpose)
        else:
            llm_model_config = LlmManager.instance().get_llm_model_configuration(model)
        
        if llm_model_config is None:
            log(ERROR, f"No model configuration found for model '{model}' and purpose '{purpose}'. Cannot determine endpoint.")
            return None

        if "modelloader" in llm_model_config and llm_model_config["modelloader"] != "runpod.io":
            if "endpoint" in llm_model_config and llm_model_config["endpoint"] is not None and len(llm_model_config["endpoint"]) > 0:
                endpoints = llm_model_config["endpoint"]
        else:
            # runpod.io: register for usage of this model, and fetch the current endpoint from the runpod.io API
            self._model_reservation = runpod_reserve_model(self._model_reservation, llm_model_config)
            if self._model_reservation is not None:
                endpoints = runpod_get_endpoint(self._model_reservation)
            else:
                if "endpoint" in llm_model_config and len(llm_model_config["endpoint"]) > 0:
                    endpoints = llm_model_config["endpoint"]
        
        if endpoints is None or len(endpoints) == 0:
            # this model configuration does not have an endpoint, which is OK. For example: OpenAI models
            return None

        # Double-check locking pattern: only one thread should initialize endpoints
        with self._endpoints_lock:
            # Check again inside the lock - another thread may have initialized while we waited
            if self._endpoints is not None:
                single_endpoint = self._get_endpoint_least_used(agent=agent, purpose=purpose, model=model, endpoint=endpoint)
                return single_endpoint
            
            # adjust for Docker container, prepare endpoints configuration
            # Build the endpoint list first (no locks required)
            model = llm_model_config.get("model") if llm_model_config else None
            endpoints_str = adjust_endpoint_to_docker_format(endpoints)
            endpoint_list = [ep.strip() for ep in endpoints_str.split(",") if ep.strip()] if endpoints_str is not None else []

            # Resolve api_key for endpoint availability check
            raw_token = llm_model_config.get("token") if llm_model_config else None
            endpoint_check_api_key = self._resolve_api_key(raw_token)

            # Check endpoint availability in parallel (do NOT hold the lock during network I/O)
            unavailable_endpoints = []

            def check_endpoint(ep):
                try:
                    is_available = self._is_endpoint_available_with_model(ep, model, api_key=endpoint_check_api_key)
                    if not is_available:
                        unavailable_endpoints.append(ep)
                    return ep if is_available else None
                except Exception as exception:
                    unavailable_endpoints.append(f"{ep} ({type(exception).__name__}: {exception})")
                    return None

            # Release lock during I/O operations to avoid blocking other threads
            # But track that initialization is in progress
            pass  # Lock still held for now to prevent duplicate initialization
        
        # Perform network I/O outside the lock
        available_endpoints = []
        if endpoint_list:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(endpoint_list))) as executor:
                for res in executor.map(check_endpoint, endpoint_list):
                    if res:
                        available_endpoints.append(res)

        # Now update shared state under lock (only if still uninitialized)
        with self._endpoints_lock:
            # Final check - only initialize if still None (another thread might have won the race)
            if self._endpoints is None:
                self._endpoints = available_endpoints
                self._endpoints_count = len(self._endpoints)
                self._endpoints_rotation_index = random.randint(0, self._endpoints_count - 1) if self._endpoints_count > 0 else 0
                self._endpoints_in_use = [0] * self._endpoints_count
                log(INFO, f"Endpoint for model '{model}' -> initialized with {self._endpoints_count} available endpoints, starting at index {self._endpoints_rotation_index}")

                if self._endpoints_count == 0:
                    log(ERROR, f"No usable LLM endpoints: provider='{self._model_provider}', agent='{agent.agent_name if agent else None}', model='{model}', purpose='{purpose}', configured_endpoints={endpoint_list}, unavailable_endpoints={unavailable_endpoints}")
            
        # return the first endpoint if multiple are specified
        single_endpoint = self._get_endpoint_least_used(agent=agent, purpose=purpose, model=model, endpoint=endpoint)
        return single_endpoint
    
    def _get_endpoint_least_used(self, agent: Optional['Agent'] = None, purpose: Optional[str] = None, model: Optional[str] = None, endpoint:Optional[str] = None, increment_counter: bool = False) -> Optional[str]:
        """Get the next endpoint via round-robin selection (random start index at initialization).
        
        Args:
            agent: The agent making the request
            purpose: Purpose of the request
            model: Model name
            endpoint: Explicit endpoint to use (if provided, this is returned directly)
            increment_counter: If True, atomically increment the usage counter for the selected endpoint.
        
        Returns:
            The selected endpoint URL, or None if no endpoints available.
        """
        if endpoint is not None:
            if increment_counter:
                self._increment_endpoint_usage_counter(endpoint)
            return endpoint

        if self._endpoints is None or self._endpoints_count == 0:
            single_endpoint = self._get_endpoint(agent=agent, purpose=purpose, model=model)
            if single_endpoint is not None and increment_counter:
                self._increment_endpoint_usage_counter(single_endpoint)
            return single_endpoint  # Return the endpoint selected during initialization
        
        # Endpoints already initialized, proceed with load balancing
        with self._endpoints_lock:
            if self._endpoints_count == 0:
                log_level = ERROR if self._model_provider != "openai" else WARNING
                log(log_level, f"No endpoints available: provider='{self._model_provider}', agent='{agent.agent_name if agent else None}', model='{model}', purpose='{purpose}', requested_endpoint='{endpoint}', initialized_endpoints=0. This is an endpoint configuration or availability problem; no HTTP request was sent.")
                return None
            
            # Round-robin selection: use current index, then advance to next
            selected_index = self._endpoints_rotation_index
            self._endpoints_rotation_index = (self._endpoints_rotation_index + 1) % self._endpoints_count
            single_endpoint = self._endpoints[selected_index]
            
            if increment_counter:
                self._endpoints_in_use[selected_index] += 1
                log(DEBUG, f"Endpoint '{single_endpoint}' selected via round-robin (index={selected_index}, next={self._endpoints_rotation_index})")
        
        return single_endpoint
    
    def _decrement_endpoint_usage_counter(self, endpoint: str):
        """Decrement the usage counter for the given endpoint."""
        if self._endpoints is None or endpoint is None:
            return
        
        with self._endpoints_lock:
            try:
                index = self._endpoints.index(endpoint)
                if index != -1 and self._endpoints_in_use[index] > 0:
                    self._endpoints_in_use[index] -= 1
            except ValueError:
                pass  # endpoint not found, ignore

    def _increment_endpoint_usage_counter(self, endpoint: str):
        """Increment the usage counter for the given endpoint."""
        if self._endpoints is None or endpoint is None:
            return
        
        with self._endpoints_lock:
            try:
                index = self._endpoints.index(endpoint)
                if index != -1:
                    self._endpoints_in_use[index] += 1
            except ValueError:
                pass  # endpoint not found, ignore

    def _is_endpoint_available_with_model(self, endpoint: str, model:Optional[str], api_key: Optional[str] = None, timeout: float = LLM_PING_TIMEOUT_SECONDS, log_level:int = WARNING) -> bool:
        """Check if the given endpoint is reachable."""
        available = False
        verify_ssl_certificates:bool = VERIFY_ENDPOINT_SSL_CERTIFICATES
        if endpoint is not None and isinstance(endpoint, str) and "runpod" in endpoint.lower():
            verify_ssl_certificates = False  # runpod.io uses self-signed certificates

        try:
            # try to reach the endpoint with a simple GET request
            uri: str = endpoint.strip().rstrip("/") + "/models"
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
            response = httpx.get(uri, headers=headers, timeout=timeout, verify=verify_ssl_certificates)
            if response.status_code == 200:
                if model is not None and len(model) > 0:
                    models_info = response.json()
                    available_models = models_info.get("models", [])
                    # If 'models' is not present, try 'data' (as in the screenshot)
                    if not available_models and "data" in models_info:
                        available_models = models_info["data"]
                    # Extract model ids
                    available_model_ids = [m.get("id") for m in available_models if isinstance(m, dict) and "id" in m]
                    if model in available_model_ids:
                        available = True
                    else:
                        log(WARNING, f"Endpoint '{endpoint}' does not support model '{model}'. Available models: {available_model_ids}")
                else:
                    available = True

        except httpx.ConnectError as e:
            log(log_level, f"Endpoint '{endpoint}' not reachable: {e}")

        except httpx.ConnectTimeout as e:
            log(log_level, f"Endpoint '{endpoint}' connection timed out after {timeout} seconds: {e}")

        except Exception as e:
            log(ERROR, f"Exception while checking endpoint '{endpoint}': {e}")

        return available

    def ping_available_models(self, agent: Optional['Agent'], purpose: Optional[str], model: Optional[str], timeout: float, log_level: int = WARNING) -> bool:
        """Check whether a configured endpoint exposes the requested model."""
        available: bool = False
        from src.core.llm.manager import LlmManager

        llm_model_config = LlmManager.instance().get_llm_model_configuration_for_agent(agent, purpose)
        if llm_model_config is not None:
            endpoints = llm_model_config.get("endpoint")
            endpoints_str = adjust_endpoint_to_docker_format(endpoints)
            endpoint_list = [ep.strip() for ep in endpoints_str.split(",") if ep.strip()] if endpoints_str is not None else []
            api_key = self._resolve_api_key(llm_model_config.get("token"))

            for endpoint in endpoint_list:
                if self._is_endpoint_available_with_model(endpoint, model, api_key=api_key, timeout=timeout, log_level=log_level):
                    available = True
                    break

        return available
    
    def _is_supporting_system_message(self) -> bool:
        """Check if this provider supports system messages."""
        return True

    def _is_supporting_response_format(self) -> bool:
        """Check if this provider supports response format specifications."""
        return True
    
    def _parse_tool_call_from_content(self, message: Dict) -> Optional[Dict]:
        """Parse tool call from message content (JSON string containing tool call data).
        
        Handles various formats:
        - List of tool call dicts: [{"fname": ..., "arguments": ...}, ...]
        - Dict with tool_call key: {"tool_call": [...]}
        - Dict with tool_calls key: {"tool_calls": [...]}
        
        Returns:
            Dict with 'tool_call' and 'message' keys if tool calls found, None otherwise.
        """
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        
        content = content.strip()
        if content.lower() == "ok":
            return None
            
        try:
            content_json = json.loads(content)
            
            # Normalize different shapes into a raw list of tool-call dicts
            raw_calls = None
            
            if isinstance(content_json, list):
                raw_calls = content_json
            elif isinstance(content_json, dict):
                # support {"tool_call": [...]}, {"tool_calls": [...]}, or single dict
                tc = content_json.get("tool_call") or content_json.get("tool_calls")
                if isinstance(tc, list):
                    raw_calls = tc
                elif isinstance(tc, dict):
                    raw_calls = [tc]
            
            if (
                isinstance(raw_calls, list)
                and raw_calls
                and all(
                    isinstance(item, dict)
                    and ("fname" in item or "name" in item)
                    and "arguments" in item
                    for item in raw_calls
                )
            ):
                tool_calls_list = []
                for idx, tool_call in enumerate(raw_calls):
                    tool_calls_list.append({
                        "type": "function",
                        "fname": tool_call.get("fname") or tool_call.get("name"),
                        "arguments": tool_call.get("arguments"),
                        "id": tool_call.get("id") or f"tool_call_{idx+1}"
                    })
                
                tool_call_message = {
                    "role": message.get("role", "assistant"),
                    "tool_calls": tool_calls_list
                }
                return {
                    "tool_call": tool_calls_list,
                    "message": tool_call_message
                }
        except Exception:
            pass  # not valid JSON or not a tool call format
        
        return None

    def _parse_openai_tool_calls(self, message: Union[Dict, Any]) -> Optional[Dict]:
        """Parse OpenAI-style tool_calls from message (dict or ChatCompletion message).
        
        Returns:
            Dict with 'tool_call' and 'message' keys if tool calls found, None otherwise.
        """
        # Handle both dict and ChatCompletion message objects
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls")
            if not tool_calls or len(tool_calls) == 0:
                return None
            
            tool_calls_list = []
            for tool_call in tool_calls:
                function_info = tool_call.get("function", {})
                arguments = json.loads(function_info.get("arguments", "{}"))
                tool_calls_list.append({
                    "type": "function",
                    "fname": function_info.get("name"),
                    "arguments": arguments,
                    "id": tool_call.get("id")
                })
            
            tool_call_message = {
                "role": message.get("role", "assistant"),
                "refusal": message.get("refusal"),
                "tool_calls": tool_calls_list
            }
        else:
            # Handle ChatCompletion message object
            if not hasattr(message, 'tool_calls') or message.tool_calls is None or len(message.tool_calls) == 0:
                return None
            
            tool_calls_list = []
            for tool_call in message.tool_calls:
                arguments = json.loads(tool_call.function.arguments)
                tool_calls_list.append({
                    "type": "function",
                    "fname": tool_call.function.name,
                    "arguments": arguments,
                    "id": tool_call.id
                })
            
            tool_call_message = {
                "role": message.role,
                "refusal": getattr(message, "refusal", None),
                "tool_calls": tool_calls_list
            }
        
        return {
            "tool_call": tool_calls_list,
            "message": tool_call_message
        }

    def _parse_structured_json_response(self, response_dict:Optional[Union[dict, list]], response_text: Optional[str], response_format: Union[Dict, List], structured_output: Dict, iter: int, response_format_not_critical_default_value: object = None) -> Union[dict, list, str, None]:
        """Parse and validate structured JSON response against schema.
        
        Returns:
            - Dict: successfully parsed and validated JSON
            - "CONTINUE": parsing/validation failed, caller should retry
            - None: no response_format specified
        """
        if response_format is None or not (isinstance(response_format, Dict) or isinstance(response_format, List)):
            return None
        
        try:
            if response_dict is None and response_text is not None:
                response_dict = get_json_from_text(response_text, json_example=response_format, json_schema=structured_output["schema"] if structured_output is not None and "schema" in structured_output else None)

            if response_dict is None:
                log(WARNING, f"response not in the expected format, not a valid JSON. Trying again once more... (iter={iter}):\n{response_text}")
                return "CONTINUE"
            
            verify_ok, verify_errors = verify_formal_and_actual_dictionary(response_format, response_dict)
            if verify_ok:
                return response_dict  # successfully parsed the response as JSON
            elif response_format_not_critical_default_value is not None and (isinstance(response_format_not_critical_default_value, dict) or isinstance(response_format_not_critical_default_value, list)):
                log(WARNING, f"Response not in the expected format: missing some keys, but response_format_not_critical_default_value is specified. Using default value.\nresponse_text={response_text}\nresponse_dict={response_dict}\nErrors={verify_errors}\niter={iter}")
                deep_copy = copy.deepcopy(response_format_not_critical_default_value)
                return deep_copy
            else:
                log(WARNING, f"Response not in the expected format: missing some keys.\nresponse_text={response_text}\nresponse_dict={response_dict}\nErrors={verify_errors}\nTrying again once more... (iter={iter})")
                return "CONTINUE"
        
        except Exception as e:
            stack = traceback.format_exc()
            log(ERROR, f"Exception: {e}\n"
                      f"response not in the expected format. response_text={response_text}\n"
                      f"exception while parsing the LLM response as JSON. Trying again once more... (iter={iter})\nStack trace:\n{stack}")
            return "CONTINUE"

    def _extract_message_content(self, response: Union[ChatCompletion, dict, list], response_format=None) -> Tuple[Optional[Union[dict, list, ChatCompletionMessage]], Optional[str], bool]:
        """Extract content string from LLM response (ChatCompletion or dict).
        
        Returns:
            Tuple of (message, content, truncated) where truncated=True if finish_reason=='length'.
        """
        # Handle None response - no content to extract
        if response is None:
            return None, None, False
        
        _truncation_log = ERROR if response_format is not None else WARNING
        _truncation_msg = ("LLM output was TRUNCATED (finish_reason='length'). "
                           "The response is incomplete. The model may have reached its internal or endpoint limit.")

        # Case 1: OpenAI ChatCompletion object (native SDK response)
        if isinstance(response, ChatCompletion):
            # Check if choices array is empty - no response data available
            if not response.choices or len(response.choices) == 0:
                log(ERROR, f"response.choices is empty. Here's the response: {response}")
                return None, None, False
            choice = response.choices[0]
            finish_reason = getattr(choice, 'finish_reason', None)
            truncated = finish_reason == "length"
            if truncated:
                log(_truncation_log, _truncation_msg)
                return None, None, True  # Message must also be None if truncated
            message = choice.message
            content = message.content if hasattr(message, 'content') else None
            return message, content, truncated
        
        # Case 2: Dictionary response (raw HTTP/API response)
        elif isinstance(response, dict):
            truncated = False
            # Check if response has valid choices array
            if response is not None and "choices" in response and response["choices"] is not None:
                choice = response["choices"][0]
                finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
                truncated = finish_reason == "length"
                if truncated:
                    # Try to extract reasoning content before truncation for debugging
                    if response is not None and "choices" in response and response["choices"] is not None:
                        try:
                            reasoning: Optional[str] = response["choices"][0].get("message", {}).get("reasoning") if isinstance(response, dict) else None
                            if reasoning:
                                _truncation_msg += f"\n\nReasoning before truncation:\n<reasoning>\n{reasoning}\n</reasoning>"
                        except Exception:
                            log(ERROR, f"Exception while extracting reasoning from truncated response: {traceback.format_exc()}")
                            pass
                    log(LLM_OUT, _truncation_msg)
                    return None, None, True  # Message must also be None if truncated
                message = choice.get("message", {}) if isinstance(choice, dict) else {}
                content = message.get("content") if isinstance(message, dict) or isinstance(message, list) else None
            else:
                # No choices array - treat entire response as message
                message = response
                content = None
            return message, content, truncated

        # Case 3: List response (array of messages or choices)
        elif isinstance(response, list):
            truncated = False
            # Handle list responses: could be array of choices or array of messages
            if len(response) > 0:
                first_item = response[0]
                # If first item is a dict with 'choices', treat as choices array (similar to dict case)
                if isinstance(first_item, dict) and "choices" in first_item:
                    choices = first_item.get("choices", [])
                    if len(choices) > 0:
                        choice = choices[0]
                        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
                        truncated = finish_reason == "length"
                        if truncated:
                            log(_truncation_log, _truncation_msg)
                            return None, None, True
                        message = choice.get("message", {}) if isinstance(choice, dict) else {}
                        content = message.get("content") if isinstance(message, dict) or isinstance(message, list) else None
                    else:
                        message = None
                        content = None
                # If first item is a dict with 'message', treat as single message in list
                elif isinstance(first_item, dict) and "message" in first_item:
                    message = first_item.get("message")
                    content = message.get("content") if isinstance(message, dict) else None
                # Otherwise, treat the list itself as content (e.g., structured output as array)
                else:
                    message = response
                    content = None
            else:
                # Empty list - no content
                message = None
                content = None
            return message, content, truncated
        
        # Case 4: String response (plain text)
        elif isinstance(response, str):
            return None, response, False
        
        # Case 5: Unsupported type
        else:
            return None, None, False

    def _generate_final_response(self, response:Union[ChatCompletion, dict, list], response_format: Union[Dict, List], structured_output: Dict, tools: Optional[list], iter: int, response_format_not_critical_default_value: object = None) -> Optional[Union[str, Dict, list, Literal["CONTINUE"]]]:
        """Generate final response from LLM response (ChatCompletion object or raw dict).
        
        Handles three main response types:
        1. Tool calls - either from content (JSON string) or OpenAI-style tool_calls
        2. Structured JSON - validated against response_format schema
        3. Plain text - simple string response
        
        Returns:
            - None: when response is None, truncated, unsupported type, or null content
            - str: plain text response, or "CONTINUE" to signal retry needed
            - Dict: tool_call result or structured JSON response
            - list: message if it's a list type
        """
        if response is None:
            return None
        
        message:Optional[Union[ChatCompletion, dict, list]] = None
        response_text: Optional[str] = None
        
        # Normalize response to extract message and content
        message, response_text, _truncated = self._extract_message_content(response, response_format=response_format)
        if _truncated:
            log(WARNING, f"LLM response was truncated due to token limit. Returning CONTINUE to retry. (iter={iter})")
            return None  
        elif message is None and response_text is None:
            log(ERROR, f"Unsupported response type: {type(response)}")
            return None
    
        # 1. Try to parse tool calls from message content (JSON string with tool call data)
        if response_format is not None and tools is not None and len(tools) > 0:
            if isinstance(message, dict):
                tool_call_result = self._parse_tool_call_from_content(message)
                if tool_call_result is not None:
                    return tool_call_result
        
        # 2. Try to parse OpenAI-style tool_calls field (works for both dict and ChatCompletion)
        openai_tool_result = self._parse_openai_tool_calls(message)
        if openai_tool_result is not None:
            return openai_tool_result
        
        # 3. Try to parse structured JSON response (if response_format specified AND content looks like JSON)
        if response_format is not None:
            stripped_text = response_text.strip() if response_text is not None else None
            
            # If content appears to be JSON (starts with { or [), parse it
            if (stripped_text is not None and len(stripped_text) > 0 and stripped_text[0] in ('{', '[')) or (isinstance(message, dict) or isinstance(message, list)):
                structured_message = message if response_text is None else None
                json_result = self._parse_structured_json_response(structured_message, response_text, response_format, structured_output, iter, response_format_not_critical_default_value=response_format_not_critical_default_value)
                if json_result is not None:
                    return json_result
            
            # If we expected JSON but got plain text (model ignored response_format),
            # it might be reasoning/thinking text. Try to extract any JSON embedded in it.
            elif stripped_text:
                log(WARNING, f"Expected structured JSON but received plain text (iter={iter}). Model may have ignored response_format. Attempting to extract JSON from text...")
                
                # Try to find JSON in the text (look for { or [ followed by valid JSON)
                for start_char in ['{', '[']:
                    start_idx = stripped_text.find(start_char)
                    if start_idx != -1:
                        # Try to extract from first occurrence to end
                        potential_json = stripped_text[start_idx:]
                        try:
                            extracted = get_json_from_text(potential_json, json_example=response_format, json_schema=structured_output["schema"] if structured_output is not None and "schema" in structured_output else None)
                            if extracted is not None:
                                log(DEBUG, f"Successfully extracted JSON from reasoning text")
                                # Validate against schema
                                verify_ok, verify_errors = verify_formal_and_actual_dictionary(response_format, extracted)
                                if verify_ok:
                                    return extracted
                                elif response_format_not_critical_default_value is not None:
                                    log(WARNING, f"Extracted JSON doesn't match schema, but response_format_not_critical_default_value is specified. Using default value.\nresponse_text={response_text}\nErrors={verify_errors}")
                                    deep_copy: dict = copy.deepcopy(response_format_not_critical_default_value) # type: ignore
                                    return deep_copy
                                else:
                                    log(WARNING, f"Extracted JSON doesn't match schema: {verify_errors}")
                        except Exception:
                            pass
                
                # No valid JSON found in the text, request retry
                log(WARNING, f"Could not extract valid JSON from plain text response. Returning CONTINUE to retry.")
                return "CONTINUE"
        
        # 4. Return plain text response
        if response_text is not None:
            return response_text
        
        # Guard: only return message if it is a basic serializable type.
        # Raw ChatCompletionMessage objects must NOT be returned – they are Pydantic
        # models, not dicts, and propagating them causes the serialized message header
        # (role/content/refusal/annotations/…) to surface as the agent response instead
        # of parsed content.  This situation arises when the model returns content=null
        # with an empty tool_calls list (e.g. some reasoning model responses).
        if isinstance(message, (str, dict, list)):
            # If the model returned a dict with null content (e.g. a thinking model that
            # exhausted its budget without emitting a final answer), do not propagate the
            # raw message dict – it would expose the internal reasoning trace as the
            # visible response.  Return None to signal that no usable content was produced.
            if isinstance(message, dict) and message.get("content") is None:
                log(WARNING, f"_generate_final_response: model returned null content (iter={iter}). No usable response produced.")
                return None
            return message
        # If structured output was expected but the model returned null content, signal
        # a retry so chat_completions_uncached can try again (up to MAX_CLIENT_CALLS_ATTEMPTS).
        if response_format is not None:
            log(WARNING, f"LLM returned null content when structured output was expected (iter={iter}). Retrying…")
            return "CONTINUE"
        return None

    def _get_model_parameters(self, model: Optional[str]) -> Optional[Dict[str, Optional[Union[str, int, float]]]]:
        """Get model parameters with normalized values.
        
        Returns:
            Dictionary with model parameters. Values may be None if not configured
            or if the value is negative (treated as disabled). Zero values are
            replaced with DEFAULT_* constants.
        """
        if model is None:
            log(ERROR, "_get_model_parameters called with model=None")
            return None
        
        model_name = self.llmconfig.get("model", model) if self.llmconfig else model
        top_p = self.llmconfig.get("top_p", None) if self.llmconfig else None
        top_k = self.llmconfig.get("top_k", None) if self.llmconfig else None
        frequency_penalty = self.llmconfig.get("frequency_penalty", None) if self.llmconfig else None
        presence_penalty = self.llmconfig.get("presence_penalty", None) if self.llmconfig else None
        no_repeat_ngram_size = self.llmconfig.get("norepeatngramsize", None) if self.llmconfig else None
        repetition_penalty = self.llmconfig.get("repetitionpenalty", None) if self.llmconfig else None
        # Normalize parameters consistently:
        # - None -> None (absent)
        # - negative values -> None (treated as disabled)
        # - 0 -> use DEFAULT_* value
        # - positive -> keep the provided value (cast to appropriate type)

        def _normalize_float(value, default):
            if value is None:
                return None
            try:
                v = float(value)
            except Exception:
                return None
            if v < 0:
                return None
            if v == 0:
                return default
            return v

        def _normalize_int(value, default):
            if value is None:
                return None
            try:
                v = int(value)
            except Exception:
                return None
            if v < 0:
                return None
            if v == 0:
                return default
            return v

        top_p = _normalize_float(top_p, DEFAULT_TOP_P)
        top_k = _normalize_int(top_k, DEFAULT_TOP_K)
        frequency_penalty = _normalize_float(frequency_penalty, DEFAULT_FREQUENCY_PENALTY)
        presence_penalty = _normalize_float(presence_penalty, DEFAULT_PRESENCE_PENALTY)
        no_repeat_ngram_size = _normalize_int(no_repeat_ngram_size, DEFAULT_NO_REPEAT_NGRAM_SIZE)
        repetition_penalty = _normalize_float(repetition_penalty, DEFAULT_REPETITION_PENALTY)

        model_parameters = {
            "model": model_name,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repetition_penalty": repetition_penalty
        }
        return model_parameters

    def _prepare_model(self, agent: Optional['Agent'], purpose: Optional[str], model: Optional[str], model_provider: Optional[str], endpoint: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
        """Prepare the HTTP request configuration for the given agent and purpose.
        
        Returns:
            Tuple of (model_id, model_provider, llm_configuration).
            All values may be None if configuration cannot be found.
        """
        # Import here to avoid circular imports
        from src.core.llm.manager import LlmManager
        
        agent_config_name: Optional[str] = None
        llm_client: Optional[LlmClient] = None

        if agent is not None:
            self.llmconfig = LlmManager.instance().get_llm_model_configuration_for_agent(agent, purpose)
            if self.llmconfig is None:
                log(ERROR, f"No LLM configuration found for agent '{agent.agent_name}' and purpose '{purpose}'")
                return None, None, None
            agent_config_name = LlmManager.instance().get_agent_config_name(agent.agent_name, purpose)
            model_id = self.llmconfig["id"] if self.llmconfig and "id" in self.llmconfig else None
            model_provider = self.llmconfig["modelprovider"] if self.llmconfig and "modelprovider" in self.llmconfig else None
        else:
            model_id, model_provider, client = LlmManager.instance().get_model(agent, purpose, model_input=model, model_provider=model_provider, endpoint=endpoint)
            self.llmconfig = LlmManager.instance().get_llm_model_configuration_by_model_name(model_id, model_provider)
            if endpoint is None:
                endpoint = self.llmconfig["endpoint"] if self.llmconfig and "endpoint" in self.llmconfig and self.llmconfig["endpoint"] is not None and len(self.llmconfig["endpoint"]) > 0 else None
            agent_config_name = LlmManager.instance().build_interface_key(model_provider, endpoint)

        if agent_config_name is not None:
            llm_client = self._clients.get_llm_client(agent_config_name, endpoint)
            if llm_client is None:
                # Create a default client
                model_client = self._prepare_model_client(agent=agent, purpose=purpose, endpoint=endpoint)
                llm_client = LlmClient(model_provider=self._model_provider, endpoint=endpoint, model_client=model_client)
                self._clients.add_llm_client(agent_config_name, llm_client, endpoint)

        if self.llmconfig is not None:
            raw_token = self.llmconfig.get("token") if self.llmconfig and "token" in self.llmconfig and self.llmconfig["token"] is not None and len(self.llmconfig["token"]) > 0 else None
            self._api_key = self._resolve_api_key(raw_token)

        if agent_config_name is not None:
            # renew the reservation for this client
            self._model_reservation = self._clients.renew_client(agent_config_name, self._model_reservation, agent, purpose, endpoint)

        return model_id, model_provider, self.llmconfig

    @abstractmethod
    def _prepare_model_client(self, agent: Optional['Agent'] = None, purpose: Optional[str] = None, endpoint: Optional[str] = None) -> Any:
        """Prepare and return the model client instance.
        
        Args:
            agent: The agent making the request (optional)
            purpose: Purpose of the LLM call (optional)
            endpoint: Specific endpoint to use (optional)
            
        Returns:
            Model client instance (type varies by provider implementation)
        """
        pass

    def _prepare_de2generator_config(
        self,
        data: Dict,
        endpoint: str,
        agent: Optional['Agent'] = None,
        goal: Optional['Goal'] = None,
        cancellation_token: Optional[str] = None,
    ) -> str:
        """Prepare de²generator proxy configuration and return the final endpoint.
        
        If de²generator supervision is enabled for this model, mutates the request data dictionary
        by injecting a "de2generator" key with configuration metadata, then returns the de2generator URI.
        Otherwise returns the original endpoint unchanged without modifying data.
        
        Args:
            data: Request data dictionary to mutate. If de²generator is enabled, adds a "de2generator" key
                  containing: model_endpoint (the actual LLM endpoint), cancellation_token (if agent provided),
                  agent_name (if agent provided), and goal_id (if goal provided). This data is NOT forwarded to
                  the LLM backend.
            endpoint: The LLM endpoint to route through (or de2generator if enabled)
            agent: The agent making the request (optional); if provided, agent_name is added to de2generator config
            goal: The goal being executed (optional); if provided, goal_id is added to de2generator config
            
        Returns:
            The final endpoint URI to use (either original or de2generator URI)
        """
        EXTRA_CONFIGURATION_KEY: str = "de2generator"
        use_de2generator: bool = False
        de2generator_uri: Optional[str] = None
        if self.llmconfig:
            use_de2generator = self.llmconfig.get("useDE2Generator", False)
            de2generator_uri = self.llmconfig.get("de2GeneratorUri")
        
        if use_de2generator and de2generator_uri:
            # Add de²generator configuration to request (not forwarded to LLM)
            data[EXTRA_CONFIGURATION_KEY] = {
                "model_endpoint": f"{endpoint.strip('/')}/chat/completions",
                "cancellation_token": cancellation_token,
            }
            # Attach agent name for UI transaction card display
            if agent is not None and agent.agent_name is not None and len(agent.agent_name) > 0:
                data[EXTRA_CONFIGURATION_KEY]["agent_name"] = agent.agent_name
            # Attach goal ID chain for UI transaction card display
            if goal is not None:
                data[EXTRA_CONFIGURATION_KEY]["goal_id"] = goal.get_id_chain_as_str()

            # Route to de²generator instead of LLM backend
            final_uri = de2generator_uri.strip("/")
            log(INFO, f"LlmInterface: routing through de²generator at {final_uri}")
            return final_uri
        
        return endpoint

    def _replace_parameter_placeholders_in_messages(self, messages: List[Dict], parameters: Dict) -> List[Dict]:
        """Replace parameter placeholders in all messages."""
        if parameters is None or len(parameters) == 0:
            return messages
        
        messages_copy = []
        for message in messages:
            message_copy = message.copy()
            content = message_copy.get("content")
            if content is not None and isinstance(content, str):
                for key, value in parameters.items():
                    if value is None:
                        continue
                    content = content.replace(f"{{{key}}}", value)
                message_copy["content"] = content
            messages_copy.append(message_copy)
        
        return messages_copy

    def _sanitize_messages(self, messages: List[Dict]) -> List[Dict]:
        return messages  # default implementation does no sanitization, override in subclasses if needed

    def _prepare_response_format(self, response_format: Optional[Dict], structured_output: Optional[object], tools: Optional[list], llmconfig: Optional[Dict], agent: Optional['Agent'], purpose: Optional[str]) -> Optional[Dict]:
        """Prepare the response format argument for the LLM call.
        
        Args:
            response_format: User-specified response format (optional)
            structured_output: Structured output configuration (optional)
            tools: List of available tools (optional)
            llmconfig: LLM configuration dictionary (optional)
            agent: Agent making the request (optional)
            purpose: Purpose of the LLM call (optional)
            
        Returns:
            Response format argument for API call, or None if not applicable
        """
        # Import here to avoid circular imports
        from src.core.llm.manager import LlmManager
        
        response_format_arg: Optional[Dict] = None
        use_structured_output = llmconfig.get("usestructuredoutput", False) if llmconfig else False
        # log(DEBUG, f"use_structured_output={use_structured_output}, has_structured_output={structured_output is not None}, has_response_format={response_format is not None}")
        
        if use_structured_output and structured_output is not None:
            if isinstance(structured_output, dict):
                # All structured_output should now be in Format A: {"name": "...", "schema": {...}, "strict": ...}
                # Just wrap it in the OpenAI format
                response_format_arg = {"type": "json_schema", "json_schema": structured_output}
                log(DEBUG, f"Using structured output with schema: {structured_output.get('name', 'unnamed')}")
            else:
                response_format_arg = structured_output
        elif response_format is not None:
            if self._is_supporting_response_format():
                response_format_arg = {"type": "json_object"}
                log(DEBUG, f"Using JSON object mode (no strict schema enforcement)")
            else:
                response_format_arg = None
                log(DEBUG, f"Model doesn't support response_format, relying on prompt")
        
        if tools is not None and response_format_arg is None:
            requirestoolschema = llmconfig.get("requirestoolschema", False) if llmconfig else False
            if requirestoolschema:
                # json_schema_tool_call_response is now in Format A, just wrap it
                response_format_arg = {"type": "json_schema", "json_schema": json_schema_tool_call_response}
        
        # if response_format_arg:
        #     log(DEBUG, f"Final response_format_arg: {response_format_arg}")

        return response_format_arg

    def _validate_tools(self, tools: list) -> bool:
        """Validate the tools parameter structure."""
        if tools is None:
            return True
            
        for tool in tools:
            required_keys = ["type", "function"]
            function_required_keys = ["name", "description", "parameters"]
            param_required_keys = ["type", "properties", "required"]
            
            # Check top-level tool structure
            for key in required_keys:
                if key not in tool:
                    log(ERROR, f"tools parameter must contain a '{key}' key: {tool}")
                    return False
            
            # Check function structure
            for key in function_required_keys:
                if key not in tool["function"]:
                    log(ERROR, f"tools parameter must contain a '{key}' key in the 'function' key: {tool}")
                    return False
            
            # Check parameters structure
            for key in param_required_keys:
                if key not in tool["function"]["parameters"]:
                    log(ERROR, f"tools parameter must contain a '{key}' key in the 'parameters' key of the 'function' key: {tool}")
                    return False
            
            # Check properties structure
            for prop_name, prop_def in tool["function"]["parameters"]["properties"].items():
                if tool.get("origin") != "mcp" and "type" not in prop_def:
                    log(ERROR, f"tools parameter must contain a 'type' key in the 'properties' key: {tool}")
                    return False
                if "description" not in prop_def:
                    log(ERROR, f"tools parameter must contain a 'description' key in the 'properties' key: {tool}")
                    return False
        
        return True
    
    def _remove_incorrect_attributes_from_tool_calls_in_messages(self, tools: list) -> list:
        """Remove incorrect attributes from tool calls in messages."""
        # Remove attributes that are useful internally but not allowed by external LLM endpoints
        extra_top_level_keys = ["origin", "whitelist_if", "blacklist_if", "llm_label"]
        for tool in tools:
            # remove top-level extraneous keys
            for k in extra_top_level_keys:
                if k in tool:
                    try:
                        del tool[k]
                    except Exception:
                        pass

            # also remove extraneous keys from nested schema, if present
            try:
                schema = tool.get("schema") if isinstance(tool, dict) else None
                if isinstance(schema, dict):
                    for k in extra_top_level_keys:
                        if k in schema:
                            try:
                                del schema[k]
                            except Exception:
                                pass
            except Exception:
                # be defensive: ignore issues while sanitizing
                pass

        return tools
    
    def _is_reasoning_disabled_on_simple(self) -> bool:
        """True when the current model is intelligent enough that SIMPLE must mean "no reasoning"."""
        intelligence = self.llmconfig.get("intelligence") if isinstance(self.llmconfig, dict) else None
        disabled: bool = (
            isinstance(intelligence, (int, float))
            and intelligence >= SIMPLE_DISABLES_REASONING_ABOVE_INTELLIGENCE
        )

        return disabled

    def _compute_reasoning_effort(self, active_reasoning: AgentReasoningEnum, maximum_reasoning_required: AgentReasoningEnum | None) -> Tuple[Optional[str], AgentReasoningEnum]:
        """
        Compute the effective reasoning level and corresponding effort string.

        Concept: the effective reasoning is the minimum (i.e. the least costly)
        between the system `active_reasoning` and the agent's
        `maximum_reasoning_required`. For example, if active is SIMPLE but the
        agent requests FAST (a lower level), FAST takes effect. Conversely, if
        the agent requests COMPLEX but active limits to SIMPLE, SIMPLE takes
        effect.

        Returns (effort_string, effective_reasoning_enum).
        """
        # Numeric mapping: lower value = less effort. NONE (-1) disables reasoning entirely.
        reasoning_levels = {
            AgentReasoningEnum.NONE: -1,
            AgentReasoningEnum.MINIMAL: 0,
            AgentReasoningEnum.FAST: 1,
            AgentReasoningEnum.SIMPLE: 2,
            AgentReasoningEnum.MEDIUM: 3,
            AgentReasoningEnum.COMPLEX: 4
        }

        # SIMPLE on a high-intelligence model means no reasoning at all (NONE is the lowest level,
        # so it also wins over any per-agent cap in the min() below)
        if active_reasoning == AgentReasoningEnum.SIMPLE and self._is_reasoning_disabled_on_simple():
            active_reasoning = AgentReasoningEnum.NONE

        # Default to SIMPLE if unknown
        default_enum: AgentReasoningEnum = AgentReasoningEnum.SIMPLE
        effective_reasoning: Optional[AgentReasoningEnum] = None
        active_level:int = reasoning_levels.get(active_reasoning, reasoning_levels[default_enum])

        if maximum_reasoning_required is None:
            effective_level = active_level
            effective_reasoning = active_reasoning if active_reasoning is not None else default_enum
        else:
            max_level = reasoning_levels.get(maximum_reasoning_required, reasoning_levels[default_enum])
            effective_level = min(active_level, max_level)
            # reverse lookup
            effective_reasoning = next((k for k, v in reasoning_levels.items() if v == effective_level), default_enum)

        # Map effective reasoning enum to effort string understood by endpoints
        effort: Optional[str] = None
        if effective_reasoning == AgentReasoningEnum.NONE:
            effort = None
        elif effective_reasoning == AgentReasoningEnum.FAST:
            effort = "low"
        elif effective_reasoning == AgentReasoningEnum.SIMPLE:
            effort = "medium"
        elif effective_reasoning == AgentReasoningEnum.COMPLEX:
            effort = "high"
        else:
            effort = "minimal"

        return effort, effective_reasoning
    
    def _remove_incorrect_attributes_from_messages(self, messages: List[Dict]) -> List[Dict]:
        """Creates a copy of messages while removing incorrect attributes."""
        extra_keys = ["tool_call_id", "name", "tool_calls"]
        sanitized = []
        for message in messages:
            msg_copy = copy.deepcopy(message)
            for k in extra_keys:
                if k in msg_copy:
                    try:
                        del msg_copy[k]
                    except Exception:
                        pass
            sanitized.append(msg_copy)
        return sanitized

    def _get_llm_configuration(self, agent: Optional['Agent'], purpose: Optional[str], model: Optional[str], model_provider: Optional[str] = None, endpoint: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
        """Get LLM configuration for agent or model.
        
        Args:
            agent: The agent making the request (optional)
            purpose: Purpose of the LLM call (optional)
            model: Model name (optional)
            model_provider: Model provider (optional)
            endpoint: Specific endpoint (optional)
            
        Returns:
            Tuple of (llm_configuration, model, model_provider). All values may be None.
        """
        from src.core.llm.manager import LlmManager  # Import here to avoid circular imports
        
        llm_configuration = None
        model_provider = None
        
        if agent is not None:
            llm_configuration = LlmManager.instance().get_llm_model_configuration_for_agent(agent, purpose)
            model_provider = llm_configuration.get("modelprovider") if llm_configuration else None
        else:
            model_id, model_provider, client = LlmManager.instance().get_model(agent, purpose, model_input=model, model_provider=model_provider, endpoint=endpoint)
            llm_configuration = LlmManager.instance().get_llm_model_configuration_by_model_name(model, model_provider)
            model = llm_configuration.get("model") if llm_configuration else model_id
        
        return llm_configuration, model, model_provider

    def _prepare_base_request_params(self, model: Optional[str], messages: List[Dict], max_completion_tokens: Optional[int], temperature: Optional[float]) -> Dict:
        """Prepare the base parameters for the API request.
        
        Args:
            model: Model name to use
            messages: List of messages
            max_completion_tokens: Maximum tokens for completion (optional)
            temperature: Temperature setting (optional, defaults to DEFAULT_TEMPERATURE)
            
        Returns:
            Dictionary containing the request parameters
        """
        if temperature is None:
            temperature = DEFAULT_TEMPERATURE

        model_parameters = self._get_model_parameters(model)
        if model_parameters is None or not isinstance(model_parameters, dict):
            log(ERROR, f"Failed to get model parameters for model '{model}'. Check configuration.")
            return {}
        
        model = model_parameters.get("model")
        top_p = model_parameters.get("top_p")
        frequency_penalty = model_parameters.get("frequency_penalty")
        # top_k = model_parameters.get("top_k")
        # no_repeat_ngram_size = model_parameters.get("no_repeat_ngram_size")
        # repetition_penalty = model_parameters.get("repetition_penalty")

        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature
            # "repetition_penalty": repetition_penalty,
            # "top_k": top_k,
            # "no_repeat_ngram_size": no_repeat_ngram_size,
        }
        if max_completion_tokens is not None:
            params["max_completion_tokens"] = max_completion_tokens

        if top_p is not None:
            params["top_p"] = top_p
        if frequency_penalty is not None:
            params["frequency_penalty"] = frequency_penalty

        return params

    def _add_response_format_to_params(self, params: Dict, response_format: Optional[Dict]) -> None:
        """Add response format to request parameters.
        
        Args:
            params: Request parameters dictionary to update
            response_format: Response format specification (optional)
        """
        if response_format is not None:
            params["response_format"] = response_format

    def _add_tools_to_params(self, params: Dict, tools: Optional[list]) -> None:
        """Add tools to request parameters.
        
        Args:
            params: Request parameters dictionary to update
            tools: List of available tools (optional)
        """
        if tools is not None:
            params["tools"] = self._remove_incorrect_attributes_from_tool_calls_in_messages(tools)

    def _add_reasoning_effort_to_params(self, params: Dict, maximum_reasoning_required: Optional[AgentReasoningEnum]) -> None:
        """Add reasoning effort to request parameters.
        
        Args:
            params: Request parameters dictionary to update
            maximum_reasoning_required: Maximum reasoning level required (optional)
        """
        if maximum_reasoning_required is not None and maximum_reasoning_required != AgentReasoningEnum.NONE:
            reasoning_level = "minimal"
            if maximum_reasoning_required == AgentReasoningEnum.FAST:
                reasoning_level = "low"
            elif maximum_reasoning_required == AgentReasoningEnum.SIMPLE:
                reasoning_level = "medium"
            elif maximum_reasoning_required == AgentReasoningEnum.COMPLEX:
                reasoning_level = "high"

            params["reasoning_effort"] = reasoning_level

    def _extract_token_usage(self, response: Union[ChatCompletion, Dict]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Extract token usage from response (base implementation).
        
        Handles both ChatCompletion objects (OpenAI) and Dict responses (HTTP).
        
        Args:
            response: ChatCompletion or Dict response from the LLM
            
        Returns:
            Tuple of (prompt_tokens, completion_tokens, total_tokens) - all Optional
        """
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        
        if isinstance(response, ChatCompletion):
            # OpenAI ChatCompletion object
            if response.usage:
                total_tokens = response.usage.total_tokens if response.usage.total_tokens else None
                completion_tokens = response.usage.completion_tokens if response.usage.completion_tokens else None
                prompt_tokens = response.usage.prompt_tokens if response.usage.prompt_tokens else None
        elif isinstance(response, dict):
            # Dict response (HTTP provider)
            if "usage" in response:
                prompt_tokens = response["usage"].get("prompt_tokens")
                completion_tokens = response["usage"].get("completion_tokens")
                total_tokens = response["usage"].get("total_tokens")
        
        return prompt_tokens, completion_tokens, total_tokens

    def _extract_thinking_message(self, response: Union[ChatCompletion, Dict]) -> Optional[str]:
        """Extract the reasoning / internal thought from response (base implementation).
        
        Handles both ChatCompletion objects (OpenAI) and Dict responses (HTTP).
        Subclasses can override to provide provider-specific extraction logic.
        
        Args:
            response: ChatCompletion or Dict response from the LLM
            
        Returns:
            Thinking message if present, None otherwise
        """
        thinking_message = None
        
        if isinstance(response, ChatCompletion):
            # OpenAI ChatCompletion - not disclosed by API
            # Subclasses can override if the provider exposes this
            return None
        elif isinstance(response, dict):
            # Dict response (HTTP provider) - check common fields
            if "choices" in response and len(response["choices"]) > 0 and "message" in response["choices"][0]:
                message = response["choices"][0]["message"]
                if "reasoning" in message:
                    thinking_message = message["reasoning"]
                elif "reasoning_content" in message:
                    thinking_message = message["reasoning_content"]
        
        return thinking_message

    def _prepare_reasoning_effort(self, agent: Optional['Agent'], purpose: Optional[str], maximum_reasoning_required: Optional[AgentReasoningEnum], messages: Optional[List[Dict]] = None) -> Tuple[Optional[str], Optional[AgentReasoningEnum]]:
        """Prepare reasoning effort if applicable (base implementation).
        
        Args:
            agent: The agent making the request (optional)
            purpose: Purpose of the LLM call (optional)
            maximum_reasoning_required: Maximum reasoning level required (optional)
            messages: Optional list of messages to modify with reasoning instructions (optional)
            
        Returns:
            Tuple of (reasoning_effort_string, effective_reasoning_level). Both may be None.
        """
        from src.core.llm.manager import LlmManager
        
        reasoning_effort: Optional[str] = None
        effective_reasoning: Optional[AgentReasoningEnum] = None
        
        if agent is not None:
            llm_configuration = LlmManager.instance().get_llm_model_configuration_for_agent(agent, purpose)
            if llm_configuration is not None and "model" in llm_configuration:
                model_name = llm_configuration["model"]
                if model_name is not None and (model_name.startswith("gpt-5") or "gpt-oss" in model_name):
                    # gpt-5 / gpt-oss specific parameters
                    active_reasoning = LlmManager.instance().get_active_reasoning()
                    reasoning_effort, effective_reasoning = self._compute_reasoning_effort(active_reasoning, maximum_reasoning_required)
                    
                    # If messages list provided and reasoning effort determined, add instruction message
                    if messages is not None and reasoning_effort is not None:
                        log(DEBUG, f"Reasoning: agent={agent.agent_name if agent else None} effective={effective_reasoning} -> effort={reasoning_effort}")
                        reasoning_msg = {
                            "role": "system",
                            "name": "reasoning_effort_instructions",
                            "content": f"Reasoning: {reasoning_effort}"
                        }
                        messages.insert(0, reasoning_msg)  # always insert at the beginning
        else:
            # If no agent, we can still apply reasoning effort based on the active reasoning level and the maximum required
            active_reasoning = LlmManager.instance().get_active_reasoning()
            reasoning_effort, effective_reasoning = self._compute_reasoning_effort(active_reasoning, maximum_reasoning_required)
            if reasoning_effort is not None:
                log(DEBUG, f"Reasoning (no agent): effective={effective_reasoning} -> effort={reasoning_effort}")
        
        return reasoning_effort, effective_reasoning

    def _extract_finish_reason(self, response: Union[ChatCompletion, Dict]) -> Optional[str]:
        """Extract finish_reason from the first choice of a provider response.
        
        Args:
            response: ChatCompletion or Dict response from the LLM
            
        Returns:
            finish_reason string if present, None otherwise
        """
        if isinstance(response, ChatCompletion):
            if response.choices:
                return getattr(response.choices[0], 'finish_reason', None)
        elif isinstance(response, dict):
            choices = response.get("choices")
            if choices:
                first = choices[0]
                return first.get("finish_reason") if isinstance(first, dict) else None
        return None

    def _build_llm_response(
        self,
        response: Union[str, Dict, ChatCompletion, None],
        thinking_message: Optional[str],
        tokens: Tuple[Optional[int], Optional[int], Optional[int]],
        elapsed_time_msec: float,
        model: str,
        reasoning_level: Optional[AgentReasoningEnum] = None,
        endpoint: Optional[str] = None,
        finish_reason: Optional[str] = None,
        model_params: Optional[dict] = None) -> LlmResponse:
        """Build LlmResponse object with consistent structure.
        
        Args:
            response: The response content from the LLM (may be None)
            thinking_message: The reasoning/internal thought message (optional)
            tokens: Tuple of (prompt_tokens, completion_tokens, total_tokens) - all may be None
            elapsed_time_msec: Time taken for the request in milliseconds
            model: Model name used
            reasoning_level: Reasoning level applied (optional)
            endpoint: Endpoint used (optional)
            finish_reason: Finish reason from LLM (optional)
            model_params: Model parameters used (optional)
            
        Returns:
            LlmResponse object with all fields populated
        """
        prompt_tokens, completion_tokens, total_tokens = tokens
        finish_reason = finish_reason or self._extract_finish_reason(response)
        raw_params = model_params
        llm_parameters = LlmParameters(
            temperature=raw_params.get("temperature") if raw_params else None,
            top_p=raw_params.get("top_p") if raw_params else None,
            top_k=raw_params.get("top_k") if raw_params else None,
            repetition_penalty=raw_params.get("repetition_penalty") if raw_params else None,
            frequency_penalty=raw_params.get("frequency_penalty") if raw_params else None,
            presence_penalty=raw_params.get("presence_penalty") if raw_params else None,
            no_repeat_ngram_size=raw_params.get("no_repeat_ngram_size") if raw_params else None,
        ) if raw_params else None

        return LlmResponse(
            response,
            thinking_message,
            total_tokens=total_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            elapsed_time_msec=elapsed_time_msec,
            model=model,
            reasoning_level=reasoning_level,
            endpoint=endpoint,
            finish_reason=finish_reason,
            parameters=llm_parameters
        )

    def _separate_thinking_message_from_response(self, llm_response: LlmResponse) -> LlmResponse:
        try:
            if llm_response is None or llm_response.response is None or not isinstance(llm_response.response, str):
                return llm_response
            
            response_text: str = llm_response.response
            thinking_prefix: str = "<think>"
            thinking_suffix: str = "</think>"
            thinking_text: str = ""

            start_idx = response_text.find(thinking_prefix)
            end_idx = response_text.find(thinking_suffix)
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                thinking_text = response_text[start_idx + len(thinking_prefix):end_idx].strip()
                response_text = (response_text[:start_idx] + response_text[end_idx + len(thinking_suffix):]).strip()
                llm_response.response = response_text.strip()
                if llm_response.thinking_message is not None and len(llm_response.thinking_message) > 0:
                    # already has a thinking message, do not overwrite
                    log(WARNING, f"_separate_thinking_message_from_response: LlmResponse already has a thinking_message, not overwriting: '{llm_response.thinking_message}'\nWe are discarding the following: '{thinking_text}'")
                else:
                    llm_response.thinking_message = thinking_text

        except Exception as e:
            stack = traceback.format_exc()
            log(ERROR, f"Exception while separating thinking message from response: {e}\nStack trace:\n{stack}")

        return llm_response

    def _get_reasoning_prompt(self, llm_configuration:dict, reasoning_settings:AgentReasoningEnum) -> Optional[str]:
        reasoning_prompt = None
        return reasoning_prompt

    def chat_completions(
        self,
        messages: List[Dict],
        agent: 'Agent' = None,
        plan: 'Plan' = None,
        goal: 'Goal' = None,
        purpose: str = None,
        parameters: Dict = None,
        tools: list = None,
        model: str = None,
        max_completion_tokens: Optional[int] = None,
        temperature: float = None,
        response_format: Dict = None,
        structured_output: object = None,
        response_format_not_critical_default_value: object = None,
        maximum_reasoning_required: AgentReasoningEnum = None,
        cache: bool = USE_LLM_CACHE) -> LlmResponse:
        """Main chat completions method with caching support."""
        
        # get endpoint in round-robin if multiple are defined
        # NOTE: We do NOT increment counter here yet - we need to wait until we know if caching applies
        endpoint = self._get_endpoint_least_used(agent=agent, purpose=purpose, model=model, increment_counter=False)

        # prepare the model and the prompt engineering configuration
        model, self._model_provider, self.llmconfig = self._prepare_model(agent, purpose, model, model_provider=self._model_provider, endpoint=endpoint)

        # fix input parameters
        if model is None:
            model = MODEL
        elif model == "lowintelligence":
            model = MODEL_LOWINTELLIGENCE
        elif model == "highintelligence":
            model = MODEL_HIGHINTELLIGENCE
        if tools is not None and len(tools) == 0:
            # OpenAI would fail given an empty array of tools
            tools = None

        # Duplicate the input messages, manipulate it and send it to the LLM
        messages_copy = []
        for message in messages:
            content:str = message["content"] if "content" in message else None
            role:str = message["role"] if "role" in message else None
            if role == "user" and isinstance(content, str) and content.startswith("/"):
                # ignore commands, skip the message
                continue

            if content is not None and isinstance(content, str):
                # remove from the content the string 'PRESERVE_ENGLISH=[[...]]'
                content = re.sub(r'PRESERVE_ENGLISH=\[\[(.*?)\]\]', r'\1', content)

            # Start with role and content
            if "role" in message:
                message_copy = {"role": message["role"], "content": content}
            else:
                message_copy = message.copy()

            # Only include tool_call_id and name for allowed roles
            role = message.get("role", None)
            if role in ["tool", "function"] and "tool_call_id" in message:
                message_copy["tool_call_id"] = message["tool_call_id"]

            # Only include name for "tool" role
            if role == "tool" and "name" in message:
                message_copy["name"] = message["name"]
                
            # Always copy tool_calls if present
            if "tool_calls" in message:
                message_copy["tool_calls"] = message["tool_calls"].copy()

            messages_copy.append(message_copy)

        messages_copy = self._replace_parameter_placeholders_in_messages(messages_copy, parameters)

        log(LLM_IN, messages_copy)

        if cache:
            llm_response = self.chat_completions_cached(
                agent=agent, plan=plan, goal=goal, purpose=purpose, endpoint=endpoint,
                messages=messages_copy, tools=tools, model=model,
                max_completion_tokens=max_completion_tokens, temperature=temperature,
                response_format=response_format, structured_output=structured_output, response_format_not_critical_default_value=response_format_not_critical_default_value,
                maximum_reasoning_required=maximum_reasoning_required)
        else:
            # Atomically select the endpoint AND increment its usage counter in one operation
            # This prevents race conditions where multiple threads pick the same endpoint
            endpoint = self._get_endpoint_least_used(agent=agent, purpose=purpose, model=model, endpoint=endpoint, increment_counter=True)
            try:
                llm_response = self.chat_completions_uncached(
                    agent=agent, plan=plan, goal=goal, purpose=purpose,
                    messages=messages_copy, tools=tools, model=model, model_provider=self._model_provider, endpoint=endpoint,
                    max_completion_tokens=max_completion_tokens, temperature=temperature,
                    response_format=response_format, structured_output=structured_output,
                    response_format_not_critical_default_value=response_format_not_critical_default_value,
                    maximum_reasoning_required=maximum_reasoning_required)

            finally:
                self._decrement_endpoint_usage_counter(endpoint)

        if plan is not None:
            plan.add_tokens_used(llm_response)
            plan.set_token_limit(max_completion_tokens)         # specific to this call
            plan.set_token_limit_for_agent(agent, purpose)      # overall for this agent

        log(LLM_OUT, llm_response)
        log_llm_interaction(messages_copy, llm_response, goal=goal, agent=agent)
        return llm_response
    
    def chat_completions_uncached(
        self,
        messages: List[Dict],
        agent: Optional['Agent'] = None,
        plan: Optional['Plan'] = None,
        goal: Optional['Goal'] = None,
        purpose: Optional[str] = None,
        tools: Optional[list] = None,
        model: Optional[str] = None,
        model_provider: Optional[str] = None,
        endpoint: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict] = None,
        structured_output: Optional[object] = None,
        response_format_not_critical_default_value: Optional[object] = None,
        maximum_reasoning_required: Optional[AgentReasoningEnum] = None) -> LlmResponse:
        """Chat completions without caching."""
        
        # Import here to avoid circular imports
        from src.core.llm.manager import LlmManager
        
        # prepare parameters
        is_reasoning_llm = False
        is_reasoning_enabled_for_agent = False
        if agent is not None and agent.agent_force_model_temperature is not None and temperature is None:
            temperature = agent.agent_force_model_temperature
        if temperature is None:
            temperature = DEFAULT_TEMPERATURE

        # prepare the endpoint: openai.com or the custom endpoint
        if agent is not None:
            llmconfig = LlmManager.instance().get_llm_model_configuration_for_agent(agent, purpose)
        else:
            llmconfig = LlmManager.instance().get_llm_model_configuration_by_model_name(model, model_provider)
        # is_reasoning_enabled_for_agent = LlmManager.instance().is_reasoning_enabled_for_agent(agent)
        # is_reasoning_llm = llmconfig["is_reasoning"] if llmconfig is not None and "is_reasoning" in llmconfig else False

        if response_format is not None and (isinstance(response_format, Dict) or isinstance(response_format, List)):
            additional_system_message = f"""You MUST provide the response in JSON in the following json format.
Do NOT output any characters before or after the JSON object.
Note that parameters whose names start with 'optional_' may be omitted ONLY when they have no value (null or empty array).
If the parameter has a non-empty value, it MUST be included in the response.
If a parameter has this format: regexp(expression), it means that the value must match the regular expression indicated inside the parenthesis.
For example, if the parameter is: regexp(OK|FAIL), the value must be either 'OK' or 'FAIL'.
The JSON must be compact, with no unnecessary whitespace, newlines, or carriage returns.
The response MUST be in the following format:
'{response_format}'"""
            if self._is_supporting_system_message():
                messages.append({
                    "role": "system",
                    "content": additional_system_message
                })
            else:
                if isinstance(messages[-1]["content"], str):
                    # add the system message at the end of the content in the last message
                    messages[-1]["content"] += "\n" + additional_system_message
                elif isinstance(messages[-1]["content"], list):
                    # add the system message at the end of the content in the last message
                    messages[-1]["content"].append({"type": "text", "text": additional_system_message})

        # try maximum 3 times to get a correct response from the large language model
        iter_count = 1
        retry = True
        sleep_time = CLIENT_CALL_INITIAL_SLEEP_TIME_SEC
        
        # prepare the response_format parameter
        response_format_arg = self._prepare_response_format(response_format, structured_output, tools, llmconfig, agent, purpose)
        
        # validate tools parameter
        if not self._validate_tools(tools):
            return None, None

        while retry and iter_count <= MAX_CLIENT_CALLS_ATTEMPTS:
            try:
                model_parameters = self._get_model_parameters(model)
                model = model_parameters["model"]

                # invoke the LLM model with tools
                llm_response:LlmResponse = self._client_chat_completions(
                    messages=messages,
                    agent=agent,
                    plan=plan,
                    goal=goal,
                    purpose=purpose,
                    tools=tools,
                    model=model,
                    model_provider=model_provider,
                    endpoint=endpoint,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                    response_format=response_format_arg,
                    structured_output=structured_output,
                    maximum_reasoning_required=maximum_reasoning_required
                )
                
                # Check if llm_response is None (e.g., when client is not found)
                if llm_response is None:
                    log(ERROR, "LLM response is None - client may not be properly initialized")
                    return None
                
                # some LLM servers return a mixture of thinking and response text, it must be separated
                llm_response = self._separate_thinking_message_from_response(llm_response)

                # textual response from the LLM
                response = llm_response.response

                if (response_format_arg is not None and response is not None and isinstance(response, str) and 
                    (response.startswith("'{") or response.startswith("'[") or 
                     response.startswith("{") or response.startswith("[") or response.startswith("```"))):
                    # some LLMs may return a JSON string, so we need to parse it
                    try:
                        response = get_json_from_text(response, json_example=response_format, json_schema=structured_output["schema"] if structured_output is not None and "schema" in structured_output else None)
                    except Exception as e:
                        log(WARNING, f"WARNING: Could not parse the LLM response: {str(e)}")

                final_response = self._generate_final_response(response, response_format, structured_output, tools, iter_count) if response is not None else None
                if final_response is not None and isinstance(final_response, str) and "CONTINUE" in final_response:
                    iter_count += 1
                    continue

                llm_response.response = final_response
                return llm_response

            except openai.AuthenticationError as e:
                log(ERROR, f"OpenAI Error: The OPENAI_API_KEY is invalid, expired or not authenticated: {e}")
                send_error_to_blazor(f"OpenAI Error: The OPENAI_API_KEY is invalid, expired or not authenticated.")
                iter_count = MAX_CLIENT_CALLS_ATTEMPTS + 1
            except openai.RateLimitError as e:
                log(ERROR, f"RateLimitError: You exceeded your current quota, please check your plan and billing details: {e}")
                send_error_to_blazor(f"OpenAI Rate Limit exceeded: thalabus exceeded the current quota for OpenAI subscription... please check the plan and billing details.")
                iter_count = MAX_CLIENT_CALLS_ATTEMPTS + 1
            except Exception as e:
                stack = traceback.format_exc()
                log(ERROR, f"An unexpected error occurred: {e}.\nMESSAGES: {messages}\nStack trace:\n{stack}")
                send_error_to_blazor(f"OpenAI returned an exception (1): {e}")

            if iter_count >= MAX_CLIENT_CALLS_ATTEMPTS:
                log(ERROR, f"Maximum number of attempts reached. Returning None.")
                send_error_to_blazor(f"OpenAI returned an exception: Maximum number of attempts reached.")
            else:
                log(WARNING, f"Some error occurred in the large language model. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                sleep_time += sleep_time
            
            # something went wrong, retry
            iter_count += 1

        # if we reach here, we failed to get a response in the correct format after several tries
        log(ERROR, f"Error getting a response from OpenAI API compliant with the given format.\nExpected response_format: {response_format}")
        return LlmResponse(None, "")
    
    def chat_completions_cached(
        self, messages: List[Dict], tools: list, model: str, max_completion_tokens: int, 
        temperature: float, response_format: Dict, structured_output: object = None, response_format_not_critical_default_value: object = None,
        agent: 'Agent' = None, goal: 'Goal' = None, plan: 'Plan' = None, purpose: str = None, endpoint: Optional[str] = None,
        maximum_reasoning_required: AgentReasoningEnum = None) -> LlmResponse:
        """Chat completions with caching support."""
        
        key = calculate_key(messages, tools, model, max_completion_tokens, temperature, response_format, structured_output)
        response, hash_val, guid, thinking_message = get_value_from_cache(key)
        llm_response: LlmResponse = LlmResponse(response, thinking_message)
        
        # Treat empty strings, empty lists or empty dicts as empty (not valid cache values)
        is_empty = ((isinstance(response, str) or isinstance(response, list)) and len(response) == 0) or (isinstance(response, dict) and len(response) == 0)
        if response is not None and not is_empty:
            # key found in cache: return it immediately
            # log(DEBUG, "(^): using cached LLM response...")
            
            if plan is not None and plan.plan_guid is not None:
                guid = plan.plan_guid
                
                # update the cache entry
                if isinstance(response, dict):
                    response["guid"] = guid
                
                cache_service = get_cache_entry_service()
                cache_service.update_cache_entry_usage(hash_val, guid, datetime.now())

            llm_response = LlmResponse(response, thinking_message)

        else:
            try:
                # key not found in cache, generate the response
                log(DEBUG, "Generating response...")
                
                llm_response = self.chat_completions_uncached(
                    messages, agent=agent, purpose=purpose, endpoint=endpoint,
                    plan=plan, goal=goal,
                    tools=tools, model=model, max_completion_tokens=max_completion_tokens,
                    temperature=temperature, response_format=response_format,
                    structured_output=structured_output, response_format_not_critical_default_value=response_format_not_critical_default_value,
                    maximum_reasoning_required=maximum_reasoning_required)
                
                if llm_response is not None and llm_response.response is not None:
                    # Treat empty strings, empty lists or empty dicts as empty (not valid cache values)
                    is_empty = ((isinstance(llm_response.response, str) or isinstance(llm_response.response, list)) and len(llm_response.response) == 0) or (isinstance(llm_response.response, dict) and len(llm_response.response) == 0)
                    if is_empty:
                        log(DEBUG, "(*): ...response is empty (string/dict/list), not saving into cache.")
                    else:
                        save_value_into_cache(
                            key, llm_response.response, thinking_message=llm_response.thinking_message, hash=hash_val, agent=agent, goal=goal, plan=plan,
                            purpose=purpose, messages=messages, tools=tools,
                            response_format=response_format, structured_output=structured_output)
                        log(DEBUG, "(*): ...response is saved into cache.")

                if plan is not None:
                    plan.add_tokens_used(llm_response)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    log(ERROR, f"(*): RateLimitError: You exceeded your current quota, please check your plan and billing details: {e}")
                    send_error_to_blazor(f"RateLimitError: thalabus exceeded the current quota for OpenAI subscription... please check the plan and billing details")
                else:
                    log(ERROR, f"(*): An unexpected error occurred: {e}")
                    send_error_to_blazor(f"OpenAI returned an error: {e}")

            except Exception as e:
                stack = traceback.format_exc()
                log(ERROR, f"(*): An unexpected exception has occurred: {e}\nStack trace:\n{stack}")

        return llm_response

    def _client_chat_completions(
        self,
        messages: List[Dict],
        agent: Optional['Agent'] = None,
        plan: Optional['Plan'] = None,
        goal: Optional['Goal'] = None,
        purpose: Optional[str] = None,
        tools: Optional[list] = None,
        model: Optional[str] = None,
        model_provider: Optional[str] = None,
        endpoint: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict] = None,
        structured_output: Optional[object] = None,
        response_format_not_critical_default_value: Optional[object] = None,
        maximum_reasoning_required: Optional[AgentReasoningEnum] = None
        ) -> Optional[LlmResponse]:
        """Default implementation for client chat completions.

        Matches the parameter set used by `chat_completions_uncached` so callers
        can pass all available data. Returns an `LlmResponse` or `None` on
        failure.
        """

        client = self._get_client(agent=agent, purpose=purpose, model=model, model_provider=model_provider, endpoint=endpoint)
        if client is None:
            return None

        # Get LLM configuration for agent or model
        llm_configuration, model, model_provider = self._get_llm_configuration(agent, purpose, model)

        # Build the request parameters
        params = self._prepare_base_request_params(model, messages, max_completion_tokens, temperature)

        # Add optional parameters
        self._add_response_format_to_params(params, response_format)
        self._add_tools_to_params(params, tools)
        self._add_reasoning_effort_to_params(params, maximum_reasoning_required)

        # Call provider and wrap results into LlmResponse
        start = time.time()
        response = client.chat.completions.create(**params)
        elapsed_msec = (time.time() - start) * 1000.0

        # Extract tokens and thinking message if available
        tokens = self._extract_token_usage(response)
        thinking_message = self._extract_thinking_message(response)

        # Build and return consistent LlmResponse
        llm_response = self._build_llm_response(response=response, thinking_message=thinking_message, tokens=tokens, elapsed_time_msec=elapsed_msec, model=model, reasoning_level=maximum_reasoning_required, endpoint=endpoint)
        return llm_response


log_end_file(__file__)
