"""
Qwen LLM Provider Implementation (Alibaba Cloud)
"""

from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import copy
import traceback
from typing import TYPE_CHECKING, Dict, List, Optional

from src.core.base.enums import AgentReasoningEnum
from src.core.base.llm_response import LlmResponse
from src.core.llm.providers.provider_http import LlmHttpRequest
from src.utils.log import log, DEBUG, INFO, WARNING, ERROR, FATAL, LLM_IN

if TYPE_CHECKING:
    from src.core.agents.Agent import Agent
    from src.core.thalaplan.goal.Goal import Goal
    from src.core.thalaplan.Plan import Plan

# Role used to replace "system" for messages after the first system message.
# Qwen only allows "system" at position 0; subsequent instructions are sent as
# "user" turns so the model still reads them without rejecting the request.
QWEN_SECONDARY_SYSTEM_ROLE = "user"


class LlmQwen(LlmHttpRequest):
    """Qwen LLM provider implementation inheriting from HTTP Request provider."""

    def __init__(self):
        super().__init__()
        self._model_provider = "qwen"
        self._current_reasoning: Optional[AgentReasoningEnum] = None

    def _prepare_model_client(self, agent: Optional['Agent'] = None, purpose: Optional[str] = None, endpoint=None):
        """Prepare and resolve the API token for Qwen.
        
        Handles token resolution from llmconfig, including environment variable expansion
        using the {{ENV_VAR}} format. If no valid token is found, _api_key remains None
        and no authorization header will be added.
        """
        self._api_key = None
        if self.llmconfig and "token" in self.llmconfig and self.llmconfig["token"]:
            token = self.llmconfig["token"]
            self._api_key = self._resolve_api_key(token)
        return None

    def _prepare_reasoning_effort(self, agent, purpose, maximum_reasoning_required, messages=None):
        """Cache the effective reasoning level so _prepare_request_data can use it.

        The base class only computes effective_reasoning for gpt-5/gpt-oss models, so for
        Qwen (non-gpt) the returned effective_reasoning is always None when called with an
        agent.  That caused two problems:
          1. The global active_reasoning cap (LlmManager._active_reasoning) was completely
             bypassed: a COMPLEX request would never be reduced to SIMPLE even if the user
             had set active_reasoning=SIMPLE.
          2. self._current_reasoning ended up as the raw maximum_reasoning_required (or None),
             so COMPLEX / None always produced budget=None in _prepare_request_data, which
             made de2generator display "✕ NO THINKING" instead of the correct "∞ THINKING".

        Fix: when the base class did not compute an effective reasoning level, do it here
        using the same _compute_reasoning_effort helper so the active_reasoning cap is
        respected for Qwen too.
        """
        reasoning_effort, effective_reasoning = super()._prepare_reasoning_effort(
            agent, purpose, maximum_reasoning_required, messages
        )
        if effective_reasoning is None:
            # Base class skipped the computation for non-gpt models with an agent.
            # Apply the global active_reasoning cap ourselves.
            from src.core.llm.manager import LlmManager
            active_reasoning = LlmManager.instance().get_active_reasoning() or AgentReasoningEnum.SIMPLE
            _, effective_reasoning = self._compute_reasoning_effort(active_reasoning, maximum_reasoning_required)

        from src.core.llm.manager import LlmManager
        log(
            DEBUG,
            f"LlmQwen reasoning: agent={agent.agent_name if agent is not None else None}, "
            f"purpose={purpose}, "
            f"active={LlmManager.instance().get_active_reasoning()}, "
            f"agent_cap={maximum_reasoning_required}, "
            f"effective={effective_reasoning}"
        )
        self._current_reasoning = effective_reasoning
        return reasoning_effort, effective_reasoning

    @staticmethod
    def _get_reasoning_effort(model: str, reasoning_level: Optional[AgentReasoningEnum]) -> Optional[str]:
        """Map an effective reasoning level to the endpoint's reasoning_effort value.

        Qwen 3.8 documents only "low", "medium" and "xhigh", and its chat template rejects
        any other value, so COMPLEX maps to "xhigh" there. Other Qwen models do not document
        "xhigh", so they keep the existing "high".
        """
        effort: Optional[str] = None

        if reasoning_level in (AgentReasoningEnum.MINIMAL, AgentReasoningEnum.FAST):
            effort = "low"
        elif reasoning_level in (AgentReasoningEnum.SIMPLE, AgentReasoningEnum.MEDIUM):
            effort = "medium"
        elif reasoning_level != AgentReasoningEnum.NONE:
            model_id: str = str(model).lower()
            if "qwen" in model_id and "3.8" in model_id:
                effort = "xhigh"
            else:
                effort = "high"

        return effort

    def _prepare_request_data(self, model, messages_copy, max_completion_tokens, temperature):
        """Inject chat_template_kwargs with enable_thinking and thinking_budget.

        Reasoning configuration is centralized in LlmManager.get_thinking_token_budget_for().
        Each level specifies (enable_thinking, budget, think_sentence) where:
          - enable_thinking=False disables all thinking/reasoning
          - budget=None means unlimited when thinking is enabled
          - think_sentence is injected as a system message to guide reasoning when not None
        """
        data = super()._prepare_request_data(model, messages_copy, max_completion_tokens, temperature)
        
        # Hard cap max_tokens at 32768 per Qwen/vLLM recommendation
        qwen_max_tokens = min(max_completion_tokens, 32768) if max_completion_tokens is not None else 32768
        data["max_tokens"] = qwen_max_tokens
        data["max_completion_tokens"] = qwen_max_tokens

        # Get reasoning config from LlmManager; default to (True, None, None) if not found
        from src.core.llm.manager import LlmManager
        enable_thinking, budget, think_sentence = LlmManager.instance().get_thinking_token_budget_for(self._current_reasoning)
        
        chat_template_kwargs: dict = {"enable_thinking": enable_thinking}
        if enable_thinking and budget is not None:
            chat_template_kwargs["thinking_budget"] = budget
        
        data["chat_template_kwargs"] = chat_template_kwargs

        reasoning_effort = self._get_reasoning_effort(model, self._current_reasoning)
        if reasoning_effort is not None:
            data["reasoning_effort"] = reasoning_effort
        
        # When reasoning is NONE, inject both /no_think directive and empty thinking block for maximum certainty
        if self._current_reasoning == AgentReasoningEnum.NONE and "messages" in data and len(data["messages"]) > 0:
            messages = data["messages"]
            no_think_system_message = {"role": "system", "content": "/no_think"}
            empty_thinking_message = {"role": "assistant", "content": "<think>\n\n</think>\n\n"}
            messages.insert(1, no_think_system_message)
            messages.insert(2, empty_thinking_message)
            log(DEBUG, f"LlmQwen: Injected /no_think directives for NONE reasoning (messages count: {len(messages)})")
        
        log(DEBUG, f"LlmQwen: enable_thinking={enable_thinking}, "
                   f"thinking_budget={chat_template_kwargs.get('thinking_budget')} "
                   f"(reasoning={self._current_reasoning})")
        
        return data

    def _prepare_response_format_arg(self, response_format, structured_output, tools):
        """Override to downgrade json_schema to json_object for Qwen/vLLM compatibility.

        Some vLLM deployments serving Qwen do not support the 'json_schema'
        response_format type and require 'structural_tag' instead.  Since the
        JSON schema is already injected into the message prompt by the base class,
        we simply replace 'json_schema' with 'json_object' in the returned
        response_format_arg so the endpoint accepts the request.
        """
        response_format_arg, response_format_message = super()._prepare_response_format_arg(
            response_format, structured_output, tools
        )
        if isinstance(response_format_arg, dict) and response_format_arg.get("type") == "json_schema":
            response_format_arg = {"type": "json_object"}
        return response_format_arg, response_format_message

    def _sanitize_messages(self, messages):
        """Override to enforce Qwen's constraint that 'system' only appears first.

        Called after all injections (tools, response format, reasoning) have been
        applied to the message list, so every mid-conversation system message is
        caught here.  The original list is not modified; a new list is returned
        with the role of any secondary system messages replaced by
        QWEN_SECONDARY_SYSTEM_ROLE.
        """
        result = []
        found_first_system = False

        messages_copy = super()._sanitize_messages(messages)  # Call parent method in case it does something

        for msg in messages_copy:
            if msg.get("role") == "system":
                if not found_first_system:
                    found_first_system = True
                    if self._current_reasoning == AgentReasoningEnum.NONE and isinstance(msg.get("content"), str):
                        msg = copy.deepcopy(msg)
                        msg["content"] = msg["content"] + "\n/no_think"
                    result.append(msg)
                    from src.core.llm.manager import LlmManager
                    _, _, think_sentence = LlmManager.instance().get_thinking_token_budget_for(self._current_reasoning)
                    if think_sentence is not None:
                        result.append({"role": QWEN_SECONDARY_SYSTEM_ROLE, "content": think_sentence})
                        log(DEBUG, f"LlmQwen: Injected think_sentence for {self._current_reasoning} reasoning")
                else:
                    replacement = copy.deepcopy(msg)
                    replacement["role"] = QWEN_SECONDARY_SYSTEM_ROLE
                    result.append(replacement)
            else:
                result.append(msg)

        log(DEBUG, "LlmQwen messages:\n" + "\n".join(
            f"  [{i}] role={msg.get('role')} content={str(msg.get('content', ''))[:120]}"
            for i, msg in enumerate(result)
        ))

        return result

    def _separate_thinking_message_from_response(self, llm_response: LlmResponse) -> LlmResponse:
        """
        Override to fix Qwen's malformed thinking tags before processing.
        
        Qwen sometimes outputs '/think>' instead of '</think>'.
        This method fixes the malformed tag and ensures proper opening tag.
        """
        if llm_response and llm_response.response and isinstance(llm_response.response, str):
            # Fix malformed closing tag '/think>\n\n' -> '</think>\n\n'
            if "/think>\n\n" in llm_response.response:
                llm_response.response = llm_response.response.replace("/think>\n\n", "</think>\n\n")
                
                # Ensure opening tag is present if we found a closing tag
                if not llm_response.response.startswith("<think>"):
                    llm_response.response = "<think>" + llm_response.response
        
        # Call parent class implementation to extract thinking
        return super()._separate_thinking_message_from_response(llm_response)

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
        """Override to retry without reasoning when Qwen exhausts its token budget mid-think."""
        llm_response = super().chat_completions_uncached(
            messages, agent=agent, plan=plan, goal=goal, purpose=purpose,
            tools=tools, model=model, model_provider=model_provider, endpoint=endpoint,
            max_completion_tokens=max_completion_tokens, temperature=temperature,
            response_format=response_format, structured_output=structured_output,
            response_format_not_critical_default_value=response_format_not_critical_default_value,
            maximum_reasoning_required=maximum_reasoning_required)
        if (llm_response is not None
                # and llm_response.finish_reason == "length"  # also retry on finish_reason='stop' (thinking budget exhausted)
                and (llm_response.response is None or (isinstance(llm_response.response, str) and len(llm_response.response.strip()) == 0))
                and llm_response.thinking_message is not None
                and len(llm_response.thinking_message) > 0):
            
            try:
                log(WARNING, f"LlmQwen: no response content after thinking (finish_reason={llm_response.finish_reason}, thinking_len={len(llm_response.thinking_message)}). Retrying without reasoning.")
                retry_messages = copy.deepcopy(messages)
                thinking_injection = (f"[PREVIOUS INTERRUPTED REASONING]\n{llm_response.thinking_message}\n\n"
                                      "[INSTRUCTION]\nYour previous reasoning was interrupted. "
                                      "Now generate ONLY the final response without thinking again. "
                                      "Do not repeat the reasoning above.")
                if retry_messages and retry_messages[0].get("role") == "system":
                    if isinstance(retry_messages[0].get("content"), str):
                        retry_messages[0]["content"] = retry_messages[0]["content"] + "\n\n" + thinking_injection
                    elif isinstance(retry_messages[0].get("content"), list):
                        retry_messages[0]["content"].append({"type": "text", "text": thinking_injection})
                else:
                    retry_messages.insert(0, {"role": "system", "content": thinking_injection})
                retry_max_tokens = (max_completion_tokens + 4000) if max_completion_tokens else 36768
                retry_response = super().chat_completions_uncached(
                    retry_messages, agent=agent, plan=plan, goal=goal, purpose=purpose,
                    tools=tools, model=model, model_provider=model_provider, endpoint=endpoint,
                    max_completion_tokens=retry_max_tokens, temperature=temperature,
                    response_format=response_format, structured_output=structured_output,
                    response_format_not_critical_default_value=response_format_not_critical_default_value,
                    maximum_reasoning_required=AgentReasoningEnum.NONE)
                if (retry_response is not None
                        and retry_response.response is not None
                        and (not isinstance(retry_response.response, str) or len(retry_response.response.strip()) > 0)):
                    log(INFO, f"LlmQwen: retry without reasoning succeeded.")
                    return retry_response
                log(ERROR, f"LlmQwen: retry without reasoning produced empty response (finish_reason={retry_response.finish_reason if retry_response else None}). Keeping original.")

            except Exception as e:
                log(ERROR, f"LlmQwen: exception during retry without reasoning: {e}\n{traceback.format_exc()}")

        return llm_response

log_end_file(__file__)


if __name__ == "__main__":
    from src.utils.log import log_init, log_exit
    from src.core.llm.manager import LlmManager
    from src.core.base.enums import AgentReasoningEnum
    
    log_init()

    model_name = "qwen3.5-35B-A3B-FP8"
    all_results = []
    USE_CACHE = False
    
    def _format_stat_row(name: str, response) -> str:
        """Format a single statistics row, handling both success and failure."""
        if response is None:
            return f"  {name:12s} | status=FAILED | no response data"
        
        # Handle cases where response exists but has None attributes
        prompt_tokens = response.prompt_tokens if response.prompt_tokens is not None else 0
        completion_tokens = response.completion_tokens if response.completion_tokens is not None else 0
        total_tokens = response.total_tokens if response.total_tokens is not None else 0
        
        return (f"  {name:12s} | status=OK | prompt={prompt_tokens:5} | completion={completion_tokens:5} | total={total_tokens:5}"
                f" | elapsed={round(response.elapsed_time_msec/1000, 1) if response.elapsed_time_msec else 0:5.1f}s"
                f" | thinking_len={(len(response.thinking_message) if response.thinking_message else 0):6d}")

    try:
        llm_manager = LlmManager.instance()
        config_for_model = llm_manager.get_llm_model_configuration_by_model_name(model_name, "qwen")
        log(INFO, f"Configuration for Qwen/{model_name}: {config_for_model}")
        if config_for_model is None:
            log(ERROR, f"No configuration found for model '{model_name}' with provider 'qwen'. Check LlmSettings.json.")
            raise ValueError(f"No configuration found for model '{model_name}' with provider 'qwen'")

        # Extract model name from config
        model_id = config_for_model.get("id", None)
        if not model_id:
            log(FATAL, f"Model ID not found in configuration for {model_name}")
            raise ValueError(f"Model ID not found in configuration for {model_name}")
        
        # Define level configurations for all tests (can be customized per test if needed)
        level_config = {
            AgentReasoningEnum.NONE: 256,       # at 50 tokens per second, this is about 5 seconds of thinking
            AgentReasoningEnum.MINIMAL: 256,    # about 5 seconds of thinking
            AgentReasoningEnum.FAST: 512,       # about 10 seconds of thinking
            AgentReasoningEnum.SIMPLE: 1024,    # about 20 seconds of thinking
            AgentReasoningEnum.MEDIUM: 1536,    # about 30 seconds of thinking
            AgentReasoningEnum.COMPLEX: 2048    # about 40 seconds of thinking
        }
        
        def _run_test_across_all_levels(
            test_name: str,
            test_func,
            level_config: dict,
            model_id: str,
            messages: Optional[list[dict]] = None
        ) -> list[tuple[str, "LlmResponse"]]:
            """
            Helper function to run a test function across all reasoning levels.
            
            Args:
                test_name: Name of the test (e.g., "T1", "T2", "T3")
                test_func: Callable that takes (reasoning_level, max_tokens) and returns LlmResponse
                level_config: Dict mapping AgentReasoningEnum to max_tokens for each level
                model_id: Model ID for logging
            
            Returns:
                List of tuples (test_label, response) for all tested levels
            """
            results = []
            for reasoning_level, max_tokens in level_config.items():
                level_name = reasoning_level.name if reasoning_level else "DEFAULT"
                label = f"{test_name}-{level_name}"
                log(LLM_IN, f"{label}: model={model_id}")
                try:
                    r = test_func(reasoning_level, max_tokens, messages) if messages is not None else test_func(reasoning_level, max_tokens)
                    results.append((label, r))
                    log(INFO, f"{label} response: {r.response if r else None}")
                except Exception as e:
                    log(ERROR, f"{label} failed: {e}")
                    results.append((label, None))
            return results
    
        if True:
            # TEST 1: Simple chat across all reasoning levels
            test_messages = [
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": "Say hello world in a concise manner." }
            ]
            test_messages_overthink = [
                { "role": "system", "content": "You are a helpful assistant, which over-thinks before giving an answer." },
                { "role": "user", "content": "Say hello world in a concise manner, but think about it for a long time." }
            ]
            
            def test0_func(reasoning_level, max_tokens, messages: list[dict] = test_messages):
                response: LlmResponse = LlmManager.instance().chat_completions(
                    messages,
                    model_input=model_id,
                    model_provider="qwen",
                    maximum_reasoning_required=reasoning_level,
                    max_completion_tokens=max_tokens,
                    cache=USE_CACHE
                )
                return response
            
            log(INFO, f"Running Test 0 (simple chat) across all reasoning levels for model {model_id}...")
            all_results.extend(_run_test_across_all_levels("T0", test0_func, { AgentReasoningEnum.NONE: 256 }, model_id))

            log(INFO, f"Running Test 0 (simple chat) with reasoning disabled (NONE) for model {model_id}...")
            all_results.extend(_run_test_across_all_levels("T0", test0_func, { AgentReasoningEnum.MINIMAL: 256 }, model_id))

            log(INFO, f"Running Test 0 (simple chat) with overthinking enabled for model {model_id}...")
            all_results.extend(_run_test_across_all_levels("T0", test0_func, { AgentReasoningEnum.COMPLEX: 256 }, model_id, messages=test_messages_overthink))
        
        if False:
            # TEST 1: Simple chat across all reasoning levels
            test_messages = [
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": "Say hello world in a concise manner." }
            ]
            
            def test1_func(reasoning_level, max_tokens, messages: list[dict] = test_messages):
                response: LlmResponse = LlmManager.instance().chat_completions(
                    messages,
                    model_input=model_id,
                    model_provider="qwen",
                    maximum_reasoning_required=reasoning_level,
                    max_completion_tokens=max_tokens,
                    cache=USE_CACHE
                )
                return response
            
            all_results.extend(_run_test_across_all_levels("T1", test1_func, level_config, model_id))
        
        if False:
            # TEST 2: JSON generation across all reasoning levels
            test_json_dict = {
                "greeting": "The greeting message to be generated",
                "number": "A number to be included in the greeting"
            }
            test_json_schema = {
                "type": "object",
                "properties": {
                    "greeting": {"type": "string"},
                    "number": {"type": "string"}
                },
                "required": ["greeting", "number"]
            }
            test_messages_json = [
                { "role": "system", "content": "You are a helpful assistant that outputs JSON." },
                { "role": "user", "content": "Generate a JSON object with a greeting message and a random number." }
            ]
            
            def test2_func(reasoning_level, max_tokens, messages: list[dict] = test_messages_json):
                return LlmManager.instance().chat_completions(
                    messages,
                    model_input=model_id,
                    model_provider="qwen",
                    response_format=test_json_dict,
                    structured_output=test_json_schema,
                    maximum_reasoning_required=reasoning_level,
                    max_completion_tokens=max_tokens,
                    cache=USE_CACHE
                )
            
            all_results.extend(_run_test_across_all_levels("T2", test2_func, level_config, model_id))
            
        if False:
            # TEST 3: Tool calling across all reasoning levels
            test_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "WeatherForecast",
                        "description": "Get the weather forecast for a given city and date.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "The name of the city to get the forecast for."
                                },
                                "date": {
                                    "type": "string",
                                    "description": "The date for the forecast in YYYY-MM-DD format."
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": "Temperature unit. Defaults to celsius."
                                }
                            },
                            "required": ["city", "date"]
                        }
                    }
                }
            ]
            test_messages_tools = [
                { "role": "system", "content": "You are a helpful assistant with access to weather tools." },
                { "role": "user", "content": "What will the weather be like in Paris on 2026-03-15?" }
            ]
            
            def test3_func(reasoning_level, max_tokens, messages: list[dict] = test_messages_tools):
                return LlmManager.instance().chat_completions(
                    messages,
                    model_input=model_id,
                    model_provider="qwen",
                    tools=test_tools,
                    maximum_reasoning_required=reasoning_level,
                    max_completion_tokens=max_tokens,
                    cache=USE_CACHE
                )
            
            all_results.extend(_run_test_across_all_levels("T3", test3_func, level_config, model_id))

        # Print unified statistics table for all tests (only if not using cache)
        if not USE_CACHE:
            log(INFO, "\n--- ALL TESTS STATISTICS ---\n" + "\n".join(
                _format_stat_row(name, r)
                for name, r in all_results
            ) + "\n----------------------------")
        else:
            log(INFO, "Tests completed with caching enabled (statistics omitted).")
    
    except Exception as e:
        log(ERROR, f"Error during Qwen LLM test: {e}")
        if all_results and not USE_CACHE:
            log(INFO, "\n--- ALL TESTS STATISTICS (partial) ---\n" + "\n".join(
                _format_stat_row(name, r)
                for name, r in all_results
            ) + "\n--------------------------------------")

    finally:
        log_exit()
