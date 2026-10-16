"""
Groq LLM Provider Implementation
"""

import os
import ast
import traceback
from typing import TYPE_CHECKING, Any, List, Dict, Optional, Tuple, Union
from groq import Groq

from src.core.base.enums import AgentReasoningEnum
from src.core.base.llm_response import LlmResponse
from src.core.llm.interface import LlmInterface
from src.core.llm.client import LlmClient
from src.core.llm import MAX_COMPLETION_TOKENS, USE_LLM_CACHE
from src.utils.log import log, FATAL, DEBUG, WARNING, ERROR

if TYPE_CHECKING:
    from src.core.agents.Agent import Agent
    from src.core.thalaplan.goal.Goal import Goal
    from src.core.thalaplan.Plan import Plan


class LlmGroq(LlmInterface):
    """Groq LLM provider implementation."""
    
    def __init__(self):
        super().__init__()
        self._model_provider = "groq"

    def _prepare_model_client(self, agent: 'Agent' = None, purpose: str = None, endpoint: Optional[str] = None) -> Any:
        """Prepare and return the Groq model client instance."""
        self._api_key = self.llmconfig.get("token") if self.llmconfig else None
            
        if not self._api_key:
            # try to get the API key from the environment variable
            self._api_key = os.getenv("GROQ_API_KEY")
            if not self._api_key:
                log(FATAL, "FATAL ERROR: no GROQ_API_KEY found")
                return None
            
        groq_client = Groq(api_key=self._api_key)
        return groq_client

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
        max_completion_tokens: int = MAX_COMPLETION_TOKENS,
        temperature: float = None,
        response_format: Dict = None,
        structured_output: object = None,
        response_format_not_critical_default_value: object = None,
        maximum_reasoning_required: 'AgentReasoningEnum' = None,
        cache: bool = USE_LLM_CACHE) -> LlmResponse:
        """Chat completions with Groq-specific post-processing."""

        # Import here to avoid circular imports  
        from src.core.llm.manager import LlmManager
        
        # prepare the model and the prompt engineering configuration
        if agent is not None:
            agent_config_name = LlmManager.instance().get_agent_config_name(agent.agent_name, purpose)
        self._prepare_model(agent, purpose, model, model_provider=self._model_provider, endpoint=None)
        
        log(DEBUG, f"LlmGroq: using the Groq API. Agent: {agent.agent_name if agent else None}, purpose: {purpose}, messages_count: {len(messages)}")
        
        llm_response = super().chat_completions(
            messages=messages,
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

        # Groq-specific tool call handling
        if tools is not None and llm_response.response != "OK" and isinstance(llm_response.response, str):
            response, thinking_message = self._process_groq_tool_response(llm_response.response, tools, response_format)
            llm_response = LlmResponse(response, thinking_message, total_tokens=llm_response.total_tokens,
                                       completion_tokens=llm_response.completion_tokens,
                                       prompt_tokens=llm_response.prompt_tokens,
                                       elapsed_time_msec=llm_response.elapsed_time_msec,
                                       model=llm_response.model,
                                       reasoning_level=llm_response.reasoning_level)

        log(DEBUG, f"response: {llm_response.response}")
        return llm_response

    def _process_groq_tool_response(self, response: str, tools: list, response_format: Dict) -> Union[Dict, Tuple[str, None]]:
        """Process Groq tool response format and convert to OpenAI-compatible format."""
        try:
            # remove any tags from the response
            response_stripped = response.replace("<tool_call>", "").replace("</tool_call>", "").strip()
            response_dict = ast.literal_eval(response_stripped)
            
            if "name" in response_dict:
                # the response may contain a tool call. Extract the tool name, and verify if it exists in the tools list
                tool_name = response_dict["name"]
                tool_args = response_dict["arguments"]
                tool_id = response_dict.get('id', 0)
                
                if tool_id == 0:
                    # groq just returns the number 0, which is not a valid tool id for OpenAI. Must convert it to a string
                    tool_id = str(tool_id)
                
                # Find the tool in the tools list
                tool_found = None
                for t in tools:
                    if t.get("type") == "function" and t.get("function", {}).get("name") == tool_name:
                        tool_found = t
                        break

                if tool_found is not None:
                    # the tool exists in the tools list. Create OpenAI-compatible response
                    return {
                        "tool_call": [{
                            "id": tool_id,
                            "type": "function",
                            "fname": tool_name,
                            "arguments": tool_args
                        }],
                        "message": {
                            "role": "assistant",
                            "refusal": None,
                            "tool_calls": [{
                                "type": "function",
                                "fname": tool_name,
                                "arguments": tool_args,
                                "id": tool_id
                            }]
                        }
                    }
                elif response_format is not None:
                    # we have a normal json structure, nothing to do with tools
                    log(WARNING, f"the tool '{tool_name}' is not in the tools list, assuming we have a normal json structure")
                else:
                    log(ERROR, f"the tool '{tool_name}' is not in the tools list, assuming we have a normal json structure")
            else:
                log(ERROR, f"the response does not contain a 'name' key, assuming we have a normal json structure")

        except Exception as e:
            stack = traceback.format_exc()
            log(ERROR, f"The response is not in the expected JSON format. Exception: {e}\nStack trace:\n{stack}")

        return response, None