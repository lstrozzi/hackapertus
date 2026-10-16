"""
HTTP Request LLM Provider Implementation

This provider makes direct HTTP requests to LLM endpoints.
"""

from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)


import json
import requests
import traceback
import time
import uuid
from typing import TYPE_CHECKING, Any, List, Dict, Optional, Union, Tuple

from src.core.base.enums import AgentReasoningEnum
from src.core.base.llm_response import LlmResponse
from src.core.llm.interface import LlmInterface
from src.core.thalaplan.goal.exceptions import GoalCancelledException
from src.core.llm import (
    MAX_COMPLETION_TOKENS, USE_LLM_CACHE, LLM_REQUEST_TIMEOUT_SECONDS, json_schema_tool_call_response
)
from src.utils.log import log, DEBUG, INFO, WARNING, ERROR, FATAL

if TYPE_CHECKING:
    from src.core.agents.Agent import Agent
    from src.core.thalaplan.goal.Goal import Goal
    from src.core.thalaplan.Plan import Plan

USE_OLD_FUNCTION_NOTATION = False
RESPONSE_FORMAT_INSTRUCTIONS = """The response format must be strictly in accordance with the following specification.
Any dictionary key starting with "optional_" is considered optional and must be omitted from the response if the value is null.
Use strict JSON with double-quoted keys and double-quoted string values. Do not include any double quotes inside string values.
Single quotes inside strings are allowed.

Examples:
Correct: {"key": "value 'ABC' with single quotes inside string is correct"}
Wrong: {"key": "value "ABC" with double quotes inside string is wrong, \"escaping\" also not allowed, same as \'single quotes\' escaping is not allowed."}
Wrong: {'key': 'single quotes json is wrong'}

Specification of expected json structure:
"""


class LlmHttpRequest(LlmInterface):
    """HTTP Request LLM provider for direct endpoint communication."""
    
    def __init__(self):
        super().__init__()
        self._model_provider = "http"
        self.llmconfig: Optional[Dict] = None

    def _prepare_model_client(self, agent: Optional['Agent'] = None, purpose: Optional[str] = None, endpoint: Optional[str] = None) -> Any:
        """Prepare and return the model client instance."""
        return None

    def _prepare_response_format_arg(self, response_format: Dict, structured_output: object, tools: list) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Prepare the response format argument for the HTTP request.
        
        Returns:
            Tuple of (response_format_arg, response_format_message)
            - response_format_arg: dict to include in request body's response_format field
            - response_format_message: dict message to insert in messages list (or None)
        """
        response_format_arg: Optional[Dict] = None
        response_format_message: Optional[Dict] = None
        use_structured_output: bool = self.llmconfig.get("usestructuredoutput", False) if self.llmconfig else False
        
        if use_structured_output and structured_output is not None:
            if isinstance(structured_output, dict):
                # Check if this is already a proper response_format with type at top level
                # Valid values are: "text", "json_object", "json_schema"
                top_level_type = structured_output.get("type")
                if top_level_type in ("text", "json_object", "json_schema"):
                    # Already in the correct format for the endpoint
                    response_format_arg = structured_output
                else:
                    # All structured_output should now be in Format A: {"name": "...", "schema": {...}, "strict": ...}
                    # Just wrap it in the OpenAI format
                    response_format_arg = {"type": "json_schema", "json_schema": structured_output}
            else:
                response_format_arg = structured_output
        elif response_format is not None:
            response_format_arg = {"type": "json_object"}

        if (tools is not None and response_format_arg is None and self.llmconfig and self.llmconfig.get("requirestoolschema", False) == True):
            response_format_arg = {"type": "json_schema", "json_schema": json_schema_tool_call_response}
        
        # Build response format message for the LLM (includes schema and/or example)
        if response_format is not None or structured_output is not None:
            response_format_text = RESPONSE_FORMAT_INSTRUCTIONS
            if use_structured_output and isinstance(structured_output, dict):
                response_format_text += f"""
Schema:
{json.dumps(structured_output)}"""
            if isinstance(response_format, dict) or isinstance(response_format, list):
                response_format_text += f"""
Example:
{json.dumps(response_format)}"""
            elif isinstance(response_format, str):
                response_format_text += f"""
Example:
{response_format}"""
                
            response_format_message = {
                "role": "system",
                "name": "LlmHttpRequest:response_format_instructions",
                "content": response_format_text
            }

        return response_format_arg, response_format_message

    # Override base class to pass messages for modification (HTTP provider adds reasoning to messages)
    def _prepare_reasoning_effort(self, agent: Optional['Agent'], purpose: Optional[str], maximum_reasoning_required: Optional[AgentReasoningEnum], messages: Optional[List[Dict]] = None) -> Tuple[Optional[str], Optional[AgentReasoningEnum]]:
        """Prepare reasoning effort if applicable.
        
        Args:
            agent: The agent making the request
            purpose: Purpose (not used, only for signature compatibility)
            maximum_reasoning_required: Maximum reasoning level required
            messages: List of messages to potentially modify with reasoning instructions
            
        Returns:
            Tuple of (reasoning_effort, effective_reasoning)
        """
        # Call base class with messages so it can add reasoning instructions
        return super()._prepare_reasoning_effort(agent, purpose, maximum_reasoning_required, messages)

    def _prepare_request_data(self, model: str, messages_copy: List[Dict], max_completion_tokens: int, temperature: float) -> Dict:
        """Prepare the JSON data for the HTTP request.
        
        Args:
            model: Model name to use
            messages_copy: List of messages
            max_completion_tokens: Maximum tokens for completion
            temperature: Temperature setting
            
        Returns:
            Dictionary containing the request data
        """
        model_parameters = self._get_model_parameters(model)
        data = {
            "model": model_parameters["model"],
            "messages": messages_copy,
            "max_tokens": max_completion_tokens,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature
            # "top_k": model_parameters["top_k"],
            # "no_repeat_ngram_size": model_parameters["no_repeat_ngram_size"],
            # "repetition_penalty": model_parameters["repetition_penalty"]
        }
        if model_parameters["top_k"] is not None:
            top_k = model_parameters["top_k"]
            data["top_k"] = top_k

        if model_parameters["frequency_penalty"] is not None:
            frequency_penalty = model_parameters["frequency_penalty"]
            data["frequency_penalty"] = frequency_penalty

        if model_parameters["presence_penalty"] is not None:
            presence_penalty = model_parameters["presence_penalty"]
            data["presence_penalty"] = presence_penalty

        return data

    def _prepare_tools_information(self, tools: list, messages_copy: List[Dict], data: Dict, use_old_function_notation: bool=False) -> None:
        """Prepare tools information if applicable.
        
        Args:
            tools: List of available tools
            messages_copy: List of messages to potentially modify with tool instructions
            data: Request data dictionary to update with tools
        """
        if tools is None:
            return
            
        # sanitize tools to remove any internal-only attributes that LLM endpoints may reject
        try:
            key = "tools" if not use_old_function_notation else "functions"
            data[key] = self._remove_incorrect_attributes_from_tool_calls_in_messages(tools)
            data["tool_choice"] = "auto"

        except Exception:
            # fallback: send tools as-is if sanitization fails for some reason
            data["tools"] = tools

        # add instructions on the format of tool calls
        tool_instructions_msg = {
            "role": "system",
            "name": "LlmHttpRequest:tool_call_instructions",
            "content": """When you want to use a tool, respond with a tool call in the specified format, which is a list of tool call objects. The format is as follows:
{ "tool_call": [ {
    "fname": "WeatherInfo",
    "arguments": {
        ... tool arguments ...
    }
}, ...
] }
"""
        }
        if len(messages_copy) > 1:
            messages_copy.insert(1, tool_instructions_msg)     # insert after the initial system prompt
        else:
            messages_copy.append(tool_instructions_msg)        # append at the end if no system prompt exists
        data["messages"] = messages_copy
        return

    def _prepare_response_format_message(self, response_format_arg: Dict, response_format_message: Dict, messages_copy: List[Dict], data: Dict) -> None:
        """Prepare response format instructions message.
        
        Args:
            response_format_arg: Response format argument for the request
            response_format_message: Response format message to add to messages
            messages_copy: List of messages to potentially modify
            data: Request data dictionary to update
        """
        if response_format_message is None:
            return
            
        # Add response format instructions as a system message
        if "response_format" not in data and response_format_arg is not None:
            data["response_format"] = response_format_arg

        if len(messages_copy) > 1:
            messages_copy.insert(1, response_format_message)     # insert after the initial system prompt
        else:
            messages_copy.append(response_format_message)        # append at the end if no system prompt exists

    def _construct_uri(self, final_uri: str) -> str:
        """Construct the chat completions URI, avoiding duplicate /v1 path segments.
        
        Args:
            final_uri: Base URI that may or may not already include /v1
            
        Returns:
            Complete URI for /v1/chat/completions endpoint
        """
        uri_base: str = final_uri.rstrip("/")
        
        if uri_base.endswith("/v1"):
            uri_base = uri_base[:-3]

        return uri_base + "/v1/chat/completions"

    # Inherited from base class: _extract_token_usage()
    # Extracts token usage from Dict responses using base implementation

    # Inherited from base class: _extract_thinking_message()
    # Extracts reasoning/thinking from Dict responses by checking 'reasoning' and 'reasoning_content' fields

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
        """Make HTTP request to LLM endpoint."""

        # Validate endpoint
        endpoint = self._get_endpoint(agent=agent, purpose=purpose, endpoint=endpoint)
        if endpoint is None:
            log_level = ERROR if self._model_provider != "openai" else WARNING
            log(log_level, f"LlmHttpRequest skipped: provider='{self._model_provider}', agent='{agent.agent_name if agent else None}', model='{model}', purpose='{purpose}', requested_endpoint='{endpoint}'. No endpoint was selected, so no HTTP request was sent.")
            return LlmResponse(None, "", model=model)

        # Prepare the response_format parameter
        response_format_arg, response_format_message = self._prepare_response_format_arg(response_format, structured_output, tools)
        
        # Sanitize messages to remove any internal-only attributes that LLM endpoints may reject
        messages_copy = self._remove_incorrect_attributes_from_messages(messages)

        # Prepare reasoning effort if applicable
        reasoning_effort, effective_reasoning = self._prepare_reasoning_effort(agent, purpose, maximum_reasoning_required, messages_copy)

        # Prepare the JSON data for the HTTP request
        request_model_params = {**self._get_model_parameters(model), "temperature": temperature}
        data = self._prepare_request_data(model, messages_copy, max_completion_tokens, temperature)
        
        # Prepare tools information if applicable
        self._prepare_tools_information(tools, messages_copy, data, use_old_function_notation=USE_OLD_FUNCTION_NOTATION)

        # Handle response format (applies to both tool and non-tool cases)
        if response_format_arg is not None:
            use_structured_output = self.llmconfig.get("usestructuredoutput", False) if self.llmconfig else False
            requires_tools_schema = (tools is not None and len(tools) > 0 and self.llmconfig and self.llmconfig.get("requirestoolschema", False) == True)
            if use_structured_output and structured_output is not None:
                data["response_format"] = response_format_arg
            elif requires_tools_schema:
                # when tools are used, always enforce the tool call response schema
                data["response_format"] = {"type": "json_schema", "json_schema": json_schema_tool_call_response}

        # Prepare response format instructions message
        self._prepare_response_format_message(response_format_arg, response_format_message, messages_copy, data)

        # Sanitize messages
        data["messages"] = self._sanitize_messages(data["messages"])

        # ── de²generator proxy routing (Thalabus extension) ──
        # Centralized in LlmInterface._prepare_de2generator_config()
        cancellation_token = str(uuid.uuid4()) if agent is not None else None
        final_uri = self._prepare_de2generator_config(
            data,
            endpoint,
            agent=agent,
            goal=goal,
            cancellation_token=cancellation_token,
        )
        cancellation_registered = bool(
            agent is not None
            and cancellation_token is not None
            and "de2generator" in data
        )
        if cancellation_registered:
            agent._register_llm_request(cancellation_token, final_uri)
        
        # Make a REST POST call to the local LLM endpoint
        # NOTE: Use the endpoint parameter directly - it was already selected by the base class
        # with proper load balancing. Do NOT call _get_endpoint again as this would bypass
        # the load balancing and potentially select a different endpoint.
        try:
            uri: str = self._construct_uri(final_uri)
            headers: Optional[Dict[str, str]] = None
            if self._api_key and self._api_key.strip():
                headers = {"Authorization": f"Bearer {self._api_key.strip()}"}

            log(DEBUG, f"LlmHttpRequest: POST {uri}")
            # log(DEBUG, f"LlmHttpRequest: POST {uri} with data: {data}")

            # Measure execution time of the HTTP request for performance tracking
            start_ts = time.perf_counter()
            http_response = requests.post(uri, json=data, headers=headers, timeout=LLM_REQUEST_TIMEOUT_SECONDS)
            end_ts = time.perf_counter()
            execution_time_msec = (end_ts - start_ts) * 1000.0

            if http_response.status_code == 499:
                raise GoalCancelledException(
                    "The LLM request was cancelled by the client.",
                    cancellation_token=cancellation_token,
                )
            
            # Track LLM performance in goal if provided
            if goal is not None:
                goal.add_perf_llm_ms(execution_time_msec)
            
            response = None
            if http_response.status_code == 200:
                response = http_response.json()
                log(DEBUG, response)

                # Extract model from response if available, otherwise use the one from request
                response_model = response.get("model", model) if response else model

                # Extract token usage
                tokens = self._extract_token_usage(response)
    
                if response and "choices" in response and len(response["choices"]) > 0 and "message" in response["choices"][0]:
                    # Extract finish_reason from raw response before processing (it is lost after _generate_final_response)
                    raw_finish_reason = response["choices"][0].get("finish_reason") if isinstance(response["choices"][0], dict) else None

                    # Extract the reasoning / internal thought
                    thinking_message = self._extract_thinking_message(response)

                    final_response = self._generate_final_response(
                        response,
                        response_format,
                        structured_output,
                        tools,
                        1,
                        response_format_not_critical_default_value=response_format_not_critical_default_value)  # iter=1 for HTTP requests

                    if final_response == "CONTINUE":
                        log(ERROR, "ERROR: LlmHttpRequest: the response from the endpoint is not in the expected format after retry")
                        final_response = None  # fall through to allow retry-without-reasoning logic below

                    if final_response is None and thinking_message:
                        log(WARNING, f"LlmHttpRequest: model produced thinking but no response content (finish_reason={raw_finish_reason!r}). Thinking tokens may have exhausted the budget.")

                    llm_response: LlmResponse = self._build_llm_response(
                        final_response,
                        thinking_message,
                        tokens,
                        execution_time_msec,
                        response_model,
                        effective_reasoning,
                        endpoint=endpoint,
                        finish_reason=raw_finish_reason,
                        model_params=request_model_params
                    )

                    # some LLM servers return a mixture of thinking and response text, it must be separated
                    llm_response = self._separate_thinking_message_from_response(llm_response)

                    return llm_response
                else:
                    log(ERROR, f"ERROR: LlmHttpRequest: the response from the endpoint is not in the expected format.\nLLM_RAW: {response}")
                    return LlmResponse(None, "", model=response_model)
            else:
                error_message = f"ERROR: LLM request to endpoint '{endpoint}' failed with status code {http_response.status_code}: {http_response.text}"
                log(ERROR, f"{error_message}.\nLLM_RAW: {response}")
                return LlmResponse(None, "", model=model, error_message=error_message)
                
        except GoalCancelledException:
            raise
        
        except Exception as e:
            stack = traceback.format_exc()
            log(ERROR, f"ERROR: LlmHttpRequest: the HTTP request to the endpoint failed with exception: {e}\nStack trace:\n{stack}")
            error_message = f"ERROR: LLM request to endpoint failed with exception: {e}"
            return LlmResponse(None, "", model=model, error_message=error_message)
        finally:
            if cancellation_registered and agent is not None and cancellation_token is not None:
                agent._clear_llm_request(cancellation_token)

log_end_file(__file__)
