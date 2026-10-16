"""
Mistral LLM Provider Implementation
"""

from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)


import traceback
import time
import httpx
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple, Union, Any
try:
    from mistralai.client import Mistral  # mistralai v2.x
except ImportError:
    from mistralai import Mistral  # mistralai v1.x

from src.core.base.enums import AgentReasoningEnum
from src.core.base.llm_response import LlmResponse
from src.core.llm.interface import LlmInterface
from src.core.llm import (
    adjust_endpoint_to_docker_format,
    DEFAULT_TEMPERATURE, USE_LLM_CACHE
)
from src.core.llm.config import DEFAULT_FREQUENCY_PENALTY, DEFAULT_NO_REPEAT_NGRAM_SIZE, DEFAULT_REPETITION_PENALTY
from src.utils.log import log, DEBUG, INFO, WARNING, ERROR, FATAL

if TYPE_CHECKING:
    from src.core.agents.Agent import Agent
    from src.core.thalaplan.goal.Goal import Goal
    from src.core.thalaplan.Plan import Plan


MISTRAL_DEFAULT_FREQUENCY_PENALTY = 0.9
MISTRAL_DEFAULT_NO_REPEAT_NGRAM_SIZE = 3
MISTRAL_DEFAULT_REPETITION_PENALTY = 1.2

SYSTEM_PROMPT_MISTRAL_REASONING = """# HOW YOU SHOULD THINK AND ANSWER

First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.

Your thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response to the user.[/THINK]Here, provide a self-contained response.
"""

class LlmMistral(LlmInterface):
    """Mistral LLM provider implementation."""
    
    __FIX_MESSAGES_FOR_MISTRAL_COMPATIBILITY = True
    __PREVENT_MISTRAL_REPEAT_TOKENS = True

    def __init__(self):
        super().__init__()
        self._model_provider = "mistral"

    def _prepare_model_client(self, agent: 'Agent' = None, purpose: str = None, endpoint: Optional[str] = None) -> Any:
        """Prepare and return the Mistral model client instance."""
        self._api_key = self.llmconfig.get("token") if self.llmconfig else None
        endpoint = self._get_endpoint(agent=agent, purpose=purpose, endpoint=endpoint)
        endpoint = adjust_endpoint_to_docker_format(endpoint)
        endpoint = endpoint.rstrip('/').rstrip("/v1") if endpoint is not None else None
        mistral_client = Mistral(api_key=self._api_key if self._api_key else "", server_url=endpoint)
        return mistral_client

    def _get_model_parameters(self, model: str) -> dict:
        model_parameters = super()._get_model_parameters(model)

        if self.__PREVENT_MISTRAL_REPEAT_TOKENS:
            # Set Mistral-specific defaults if not already set
            if "frequency_penalty" not in model_parameters or model_parameters["frequency_penalty"] is None or model_parameters["frequency_penalty"] == DEFAULT_FREQUENCY_PENALTY:
                model_parameters["frequency_penalty"] = MISTRAL_DEFAULT_FREQUENCY_PENALTY
            if "no_repeat_ngram_size" not in model_parameters or model_parameters["no_repeat_ngram_size"] is None or model_parameters["no_repeat_ngram_size"] == DEFAULT_NO_REPEAT_NGRAM_SIZE:
                model_parameters["no_repeat_ngram_size"] = MISTRAL_DEFAULT_NO_REPEAT_NGRAM_SIZE
            if "repetition_penalty" not in model_parameters or model_parameters["repetition_penalty"] is None or model_parameters["repetition_penalty"] == DEFAULT_REPETITION_PENALTY:
                model_parameters["repetition_penalty"] = MISTRAL_DEFAULT_REPETITION_PENALTY

        return model_parameters

    def _is_supporting_system_message(self) -> bool:
        """Mistral does not support system messages in the same way as OpenAI."""
        return False

    def _is_supporting_response_format(self) -> bool:
        """Mistral does not support response format in the same way as OpenAI."""
        return False

    def _format_messages_for_mistral(self, model: str, messages_orig: List[Dict], parameters: Dict = None) -> List[Dict]:
        """Format messages according to Mistral's requirements."""
        messages_for_mistral = None

        messages = super()._remove_incorrect_attributes_from_messages(messages_orig)

        if self.__FIX_MESSAGES_FOR_MISTRAL_COMPATIBILITY:
            messages_for_mistral = []
            prev_role = None

            if model is not None and "-reasoning" in model.lower():
                log(DEBUG, f"LlmMistral: _format_messages_for_mistral: Added system message for mistral reasoning model '{model}'")
                messages_for_mistral.append({"role": "system", "content": SYSTEM_PROMPT_MISTRAL_REASONING})
                messages_for_mistral.append({"role": "user", "content": ""})    # Placeholder user message to follow system, else mistral parser may error out

            for i, message in enumerate(messages):
                content = message.get("content")

                # Replace any {key} placeholders in the message content with values from the parameters dict.
                if content is not None and parameters is not None:
                    for key, value in parameters.items():
                        if value is None:
                            continue
                        content = content.replace(f"{{{key}}}", value)

                role = message.get("role")

                # If a 'system' role would come after an 'assistant', insert a 'user' message with empty content in between.
                if prev_role == "assistant" and role == "system":
                    messages_for_mistral.append({"role": "user", "content": ""})

                # Build the message copy, preserving all fields except content (which may be replaced)
                message_copy = message.copy()
                message_copy["content"] = content

                # Copy 'tool_calls' deeply if present
                if "tool_calls" in message:
                    message_copy["tool_calls"] = message["tool_calls"].copy()

                messages_for_mistral.append(message_copy)
                prev_role = role

        return messages_for_mistral if messages_for_mistral is not None else messages
    
    def _extract_token_usage(self, response: Any) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Extract token usage from Mistral response."""
        total_tokens = response.usage.total_tokens if hasattr(response, 'usage') and response.usage and hasattr(response.usage, 'total_tokens') else None
        completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage and hasattr(response.usage, 'completion_tokens') else None
        prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage and hasattr(response.usage, 'prompt_tokens') else None

        return total_tokens, prompt_tokens, completion_tokens
    
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
        maximum_reasoning_required: 'AgentReasoningEnum' = None) -> Optional[LlmResponse]:
        """Mistral-specific chat completions implementation."""
        
        thinking_message: Optional[str] = None
        total_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        prompt_tokens: Optional[int] = None
        execution_time_msec: Optional[float] = None
        reasoning_level: Optional[AgentReasoningEnum] = None

        endpoint = self._get_endpoint_least_used(agent=agent, purpose=purpose, model=model, endpoint=endpoint)
        if endpoint is None:
            from src.core.llm.manager import LlmManager
            llm_settings = LlmManager.instance().get_llm_model_configuration_for_agent(agent, purpose)
            log(FATAL, f"LlmMistral: No endpoint specified for Mistral LLM provider.\nllm settings: {llm_settings}")
            return LlmResponse(None, None)
        client = self._get_client(agent=agent, purpose=purpose, model=model, model_provider="mistral", endpoint=endpoint)
        if client is None:
            return LlmResponse(None, None)

        if temperature is None:
            temperature = DEFAULT_TEMPERATURE

        from src.core.llm.manager import LlmManager
        llm_model_config = LlmManager.instance().get_llm_model_configuration_by_model_name(model, model_provider="mistral")
        model_id = llm_model_config.get("model", model) if llm_model_config else model
        model_parameters = self._get_model_parameters(model)

        # Cap max_tokens to model's limit (most Mistral models support 32k-128k context but lower completion limits)
        # Mistral Large 2: 128k context, but completion typically capped at 16k-32k
        # Adjust based on the model being used
        model_max_tokens = llm_model_config.get("token_limit_kilo", 32) * 1024 if llm_model_config else 32768
        if max_completion_tokens is not None and max_completion_tokens > model_max_tokens:
            log(WARNING, f"Requested max_tokens ({max_completion_tokens}) exceeds model limit ({model_max_tokens}). Capping to {model_max_tokens}")
            max_completion_tokens = model_max_tokens

        try:
            self._increment_endpoint_usage_counter(endpoint)

            # Build request parameters based on what's provided
            params = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature
            }
            if max_completion_tokens is not None:
                params["max_tokens"] = max_completion_tokens
            if model_parameters["top_k"] is not None:
                params["top_k"] = model_parameters["top_k"]
            if model_parameters["frequency_penalty"] is not None:
                params["frequency_penalty"] = model_parameters["frequency_penalty"]
            if model_parameters["presence_penalty"] is not None:
                params["presence_penalty"] = model_parameters["presence_penalty"]
            # NOT SUPPORTED IN MISTRAL API
            # if "repetition_penalty" in model_parameters:
            #     params["repetition_penalty"] = model_parameters["repetition_penalty"]
            # if "no_repeat_ngram_size" in model_parameters:
            #     params["no_repeat_ngram_size"] = model_parameters["no_repeat_ngram_size"]

            # Include tools if provided
            if tools is not None:
                params["tools"] = tools
            
            # Include response format if provided (Mistral uses 'response_format')
            if response_format is not None and self._is_supporting_response_format():
                params["response_format"] = response_format

            # Measure execution time to call the LLM model with the Mistral API
            # Retry on transient network errors (e.g., server disconnected)
            _max_network_retries = 3
            _network_retry_delay = 2.0
            for _network_attempt in range(1, _max_network_retries + 1):
                try:
                    start_ts = time.perf_counter()
                    response_llm = client.chat.complete(**params)       # Call to Mistral API
                    end_ts = time.perf_counter()
                    execution_time_msec = (end_ts - start_ts) * 1000.0
                    
                    # Track LLM performance in goal if provided (only count successful attempt)
                    if goal is not None:
                        goal.add_perf_llm_ms(execution_time_msec)
                    
                    break  # success
                except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError) as net_err:
                    if _network_attempt < _max_network_retries:
                        log(WARNING, f"LlmMistral: transient network error (attempt {_network_attempt}/{_max_network_retries}): {net_err}. Retrying in {_network_retry_delay}s...")
                        time.sleep(_network_retry_delay)
                        _network_retry_delay *= 2
                    else:
                        raise

            # Handle potential Unset values if present in response (for compatibility with different SDK versions)
            if hasattr(response_llm, 'choices') and response_llm.choices and len(response_llm.choices) > 0:
                message = response_llm.choices[0].message
                if hasattr(message, 'tool_calls') and message.tool_calls is not None:
                    try:
                        # Check if it's Unset type by checking its string representation
                        if str(type(message.tool_calls)) in ("<class 'mistralai.types.basemodel.Unset'>", "<class 'mistralai.client.types.basemodel.Unset'>"):
                            message.tool_calls = None
                    except:
                        pass
                if hasattr(message, 'content') and message.content is not None:
                    try:
                        if str(type(message.content)) in ("<class 'mistralai.types.basemodel.Unset'>", "<class 'mistralai.client.types.basemodel.Unset'>"):
                            message.content = None
                    except:
                        pass

            total_tokens, prompt_tokens, completion_tokens = self._extract_token_usage(response_llm)
            response = response_llm.choices[0].message.content if response_llm.choices and len(response_llm.choices) > 0 and response_llm.choices[0].message else None

        except Exception as e:
            stack = traceback.format_exc()
            log(ERROR, f"ERROR: LlmMistral: an exception occurred while calling the Mistral API: {e}\nStack trace:\n{stack}")
            raise e
        
        finally:
            self._decrement_endpoint_usage_counter(endpoint)

        return LlmResponse(response,
            thinking_message,
            total_tokens=total_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            elapsed_time_msec=execution_time_msec,
            model=model,
            reasoning_level=reasoning_level,
            endpoint=endpoint
        )
    
    def _extract_message_content(self, response: Union[Any, dict], response_format=None) -> Tuple[dict, Optional[str], bool]:
        """Extract content string from LLM response object.
        
        Returns:
            Tuple of (message, content, truncated) where truncated=True if finish_reason=='length'.
        """
        if hasattr(response, 'choices') and response.choices:
            if not response.choices or len(response.choices) == 0:
                log(ERROR, f"response.choices is empty. Here's the response: {response}")
                return None, None, False
            choice = response.choices[0]
            finish_reason = getattr(choice, 'finish_reason', None)
            truncated = finish_reason == "length"
            if truncated:
                _truncation_log = ERROR if response_format is not None else WARNING
                log(_truncation_log, ("LLM output was TRUNCATED (finish_reason='length'). "
                                      "The response is incomplete. The model may have reached its internal or endpoint limit."))
                return None, None, True  # Message must also be None if truncated
            message = choice.message
            content = message.content if hasattr(message, 'content') else None
        else:
            message, content, truncated = super()._extract_message_content(response, response_format=response_format)

        return message, content, truncated
        
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
        maximum_reasoning_required: 'AgentReasoningEnum' = None,
        cache: bool = USE_LLM_CACHE) -> LlmResponse:
        """
        Mistral-specific chat completions with message format adjustments.
        
        Mistral has specific requirements:
        1. Only one system message in the beginning
        2. Subsequent messages must alternate between 'user' and 'assistant' roles
        """

        # Reformat messages for Mistral's requirements
        messages_copy = self._format_messages_for_mistral(model, messages, parameters)

        # Call parent implementation with formatted messages
        llm_response = super().chat_completions(
            messages=messages_copy,
            agent=agent,
            plan=plan,
            goal=goal,
            purpose=purpose,
            parameters=parameters,
            tools=tools,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            response_format=response_format,
            structured_output=structured_output,
            response_format_not_critical_default_value=response_format_not_critical_default_value,
            maximum_reasoning_required=maximum_reasoning_required,
            cache=cache)
        
        return llm_response

log_end_file(__file__)
