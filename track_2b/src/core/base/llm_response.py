"""LLM Response Object Implementation"""
from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

from typing import Optional, Union, Dict
from src.core.base.enums import AgentReasoningEnum

class LlmParameters:
    """LLM call parameters used for a single inference request.

    All fields are Optional; None means the parameter was not configured or was
    explicitly disabled (e.g. a raw value of -1 in LlmSettings.json).
    """

    def __init__(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class LlmResponse:
    """Response object returned by LLM completion methods."""
    
    def __init__(self, response: Optional[Union[str, Dict]], thinking_message: Optional[str] = "", 
                 total_tokens: Optional[int] = None, completion_tokens: Optional[int] = None, prompt_tokens: Optional[int] = None,
                 elapsed_time_msec: Optional[float] = None, model: Optional[str] = None, reasoning_level: Optional[AgentReasoningEnum] = None,
                 endpoint: Optional[str] = None, finish_reason: Optional[str] = None,
                 parameters: Optional[LlmParameters] = None,
                 error_message: Optional[str] = None):
        self.response: Optional[Union[str, Dict]] = response
        self.thinking_message: Optional[str] = thinking_message
        self.prompt_tokens: Optional[int] = prompt_tokens
        self.total_tokens: Optional[int] = total_tokens  # Can be None if usage not available
        self.completion_tokens: Optional[int] = completion_tokens
        self.elapsed_time_msec: Optional[float] = elapsed_time_msec
        self.reasoning_level: Optional[AgentReasoningEnum] = reasoning_level
        self.model: Optional[str] = model.lstrip("/").rstrip() if model else None
        self.endpoint: Optional[str] = endpoint.rstrip().rstrip("/").rstrip("/v1") if endpoint else None
        self.finish_reason: Optional[str] = finish_reason  # 'stop' | 'length' | 'tool_calls' | None
        self.parameters: Optional[LlmParameters] = parameters
        self.error_message: Optional[str] = error_message

    
    @property
    def was_truncated(self) -> bool:
        """True when the model stopped because it hit the token limit."""
        return self.finish_reason == "length"

log_end_file(__file__)
