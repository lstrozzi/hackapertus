from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import json
import requests
import threading
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, Any

from src.core.base.enums import AgentReasoningEnum
from src.core.base.InputOutput import completion_percentage, is_stopped, send_plan_to_blazor
from src.core.thalaplan.goal.Goal import Goal
from src.core.thalaplan.goal.enums import GoalStatus
from src.core.thalaplan.goal.exceptions import GoalCancelledException
from src.utils.log import log, DEBUG, INFO, WARNING, ERROR, FATAL


if TYPE_CHECKING:
    from src.core.thalaplan.goal.Goal import Goal


class Agent(ABC):
    """Abstract base class for all agent implementations."""

    def __init__(self, name: str):
        self.agent_name: str = name.split('.')[-1]   # ("src.core.agents.Agent" will be saved as "Agent")
        self.agent_instance_id: Optional[str] = None
        self.agent_cancellation_token: str = str(uuid.uuid4())
        self._active_llm_requests: Dict[str, str] = {}
        self._active_llm_requests_lock = threading.Lock()
        self.agent_output: Optional[dict] = None
        self.agent_force_model: Optional[str] = None
        self.agent_force_model_provider: Optional[str] = None
        self.agent_force_model_endpoint: Optional[str] = None
        self.agent_force_model_temperature: Optional[float] = None
        self.agent_execution: List[str] = []
        self.agent_cache_category: str = "chatbot"
        self.agent_minimum_reasoning: AgentReasoningEnum = AgentReasoningEnum.FAST
        self.agent_overrule_silent_summary: bool = False
        self.agent_lock: threading.Lock = threading.Lock()
        
        # completion tracking attributes
        self.agent_completion_blocks_weights: List[float] = [1.0]
        self.agent_completion_blocks: List[float] = [0.0]
        self.agent_completion_block_titles: List[str] = [""]
        self.agent_completion_current_block: int = -1
        self.agent_progress_completed: Optional[int] = None
        self.agent_progress_max: Optional[int] = None

        # blazor progress notification thread (lazily started by _start_blazor_progress_updates)
        self._blazor_stop_event: Optional[threading.Event] = None
        self._blazor_thread: Optional[threading.Thread] = None
        self._blazor_last_completion: float = -1.0
        self._blazor_last_step: str = ""

    # methods
    @abstractmethod
    def __str__(self):
        return json.dumps(self.to_json())
    
    @abstractmethod
    def to_json(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "agent_instance_id": self.agent_instance_id,
            "agent_execution": self.agent_execution,
            "agent_minimum_reasoning": self.agent_minimum_reasoning.value,
            "completion": self._completion_compute(),
            "completion_step": self.agent_completion_block_titles[self.agent_completion_current_block] if self.agent_completion_current_block >= 0 else "",
            "progress_completed": self.agent_progress_completed,
            "progress_max": self.agent_progress_max
        }
    
    @abstractmethod
    def clone(self) -> 'Agent':
        cloned = Agent(self.agent_name)
        return cloned
    
    @abstractmethod
    def clone_to(self, target) -> None:
        target.agent_name = self.agent_name
        target.agent_cancellation_token = str(uuid.uuid4())
        target.agent_overrule_silent_summary = self.agent_overrule_silent_summary

    def raise_if_plan_stopped(self) -> None:
        if is_stopped():
            raise GoalCancelledException("Plan execution was stopped by the user.")
    
    @abstractmethod
    def run(self, goal:'Goal') -> bool:
        pass

    def rollback(self, goal: 'Goal') -> bool:
        return self._generic_rollback(goal)

    def _generic_rollback(self, goal: 'Goal') -> bool:
        """
        Generic rollback building block, shared by every trainable agent's rollback() override.
        Discards goal's speculative work (its whole subgoal subtree, like speculative execution being
        discarded), resets every goal that transitively depended on it (also those depending on its ancestors,
        e.g. the Verifier) and reopens its ancestors, so that the scheduler enters them again and the plan is
        ready to be re-run (see the /restart command). Async goals still running are cancelled first, and the
        rollback is refused (False) if one of them cannot be stopped.
        Idempotent: calling it again on an already-reset goal is a no-op.
        """
        goal_perf_counters: Dict[str, float] = goal.get_aggregated_perf_counters()
        already_reset: bool = (
            goal.goal_status == GoalStatus.NOT_STARTED
            and not goal.goal_subgoals
            and all(counter == 0.0 for counter in goal_perf_counters.values())
        )
        rolled_back: bool = True

        if not already_reset:
            dependents: List[Goal] = goal.get_rollback_dependents()
            ancestors: List[Goal] = goal.get_rollback_ancestors()

            for affected in [goal] + dependents:
                for running in affected.get_subtree_goals():
                    if not running.cancel_async_execution():
                        log(WARNING, f"Rollback of goal {goal.get_id_chain_as_str()} refused: async goal {running.get_id_chain_as_str()} '{running.goal_title}' did not stop in time")
                        rolled_back = False

            if rolled_back:
                for dependent in dependents:
                    dependent.reset_for_rollback()
                goal.reset_for_rollback()
                for ancestor in ancestors:
                    ancestor.reset_ancestor_for_rollback()
                send_plan_to_blazor(goal.goal_plan, goal.get_id_chain())

        return rolled_back

    def _register_llm_request(self, cancellation_token: str, de2generator_uri: str) -> None:
        with self._active_llm_requests_lock:
            self.agent_cancellation_token = cancellation_token
            self._active_llm_requests[cancellation_token] = de2generator_uri

    def _clear_llm_request(self, cancellation_token: str) -> None:
        with self._active_llm_requests_lock:
            self._active_llm_requests.pop(cancellation_token, None)

    def get_active_llm_cancellation_tokens(self) -> List[str]:
        with self._active_llm_requests_lock:
            return list(self._active_llm_requests)

    def cancel_llm_request(self, cancellation_token: Optional[str] = None, timeout: float = 5.0) -> bool:
        """Cancel one active LLM call, or all active calls when no token is supplied."""
        with self._active_llm_requests_lock:
            if cancellation_token is not None:
                endpoint = self._active_llm_requests.get(cancellation_token)
                targets = [(cancellation_token, endpoint)] if endpoint is not None else []
            else:
                targets = list(self._active_llm_requests.items())

        if not targets:
            return False

        all_cancelled = True
        for token, endpoint in targets:
            try:
                endpoint_base = endpoint.rstrip('/')
                if endpoint_base.endswith('/v1/chat/completions'):
                    endpoint_base = endpoint_base[:-len('/v1/chat/completions')]
                elif endpoint_base.endswith('/chat/completions'):
                    endpoint_base = endpoint_base[:-len('/chat/completions')]
                elif endpoint_base.endswith('/v1'):
                    endpoint_base = endpoint_base[:-3]
                response = requests.post(
                    f"{endpoint_base}/v1/cancel",
                    json={"cancellation_token": token},
                    timeout=timeout,
                )
                body = response.json() if response.content else {}
                if response.status_code != 200 or body.get("status") not in {"cancelled", "already_cancelled"}:
                    all_cancelled = False
                    log(ERROR, f"Agent: LLM cancellation failed for token {token}: HTTP {response.status_code} {body}")
            except Exception as exception:
                all_cancelled = False
                log(ERROR, f"Agent: LLM cancellation failed for token {token}: {exception}")

        return all_cancelled

    def _is_placeholder_feature_enabled(self) -> bool:
        """Check if placeholder-based long-generation is enabled globally. Returns True if enabled, False otherwise."""
        from src.config.constants import AGENT_ASSISTANT_PLACEHOLDER_ENABLE
        if AGENT_ASSISTANT_PLACEHOLDER_ENABLE is False:
            return False
        
        try:
            from src.core.llm import LlmManager
            quality_settings = LlmManager.instance().get_quality_settings()
            # If setting is explicitly present, use it; otherwise fall back to constant
            if "assistant_subagents_enable" in quality_settings:
                return quality_settings.get("assistant_subagents_enable", False)
            return AGENT_ASSISTANT_PLACEHOLDER_ENABLE
        except Exception:
            return AGENT_ASSISTANT_PLACEHOLDER_ENABLE

    # protected methods
    def _completion_set_blocks(self, blocks_weights:List[float]) -> None:
        # set the completion blocks. This will be used to track the completion percentage of the agent.
        self.agent_completion_blocks_weights = blocks_weights
        self.agent_completion_blocks = [0.0] * len(blocks_weights)
        self.agent_completion_block_titles = [""] * len(blocks_weights)
        self.agent_completion_current_block = -1
        
    def _completion_start_block(self, block_title:str=None) -> int:
        # start tracking completion percentage
        with self.agent_lock:
            self._completion_set(self._completion_compute())
            if self.agent_completion_current_block < len(self.agent_completion_blocks)-1:
                self.agent_completion_current_block = self.agent_completion_current_block + 1
            else:
                log(WARNING, f"$Agent: Completion block {self.agent_completion_current_block} is already set. Skipping to the next block.")

            self.agent_completion_block_titles[self.agent_completion_current_block] = block_title if block_title is not None else f"Block {self.agent_completion_current_block+1}"
            self.agent_completion_blocks[self.agent_completion_current_block] = 0.0
            return self.agent_completion_current_block
        
    def _completion_end_block(self) -> int:
        self._completion_track(1.0)
        log(INFO, f"Completion: {self._completion_compute()}")
        return self.agent_completion_current_block
    
    def _completion_end_all_blocks(self) -> None:
        # end all blocks and set the completion percentage to 100%
        with self.agent_lock:
            for i in range(len(self.agent_completion_blocks)):
                self.agent_completion_blocks_weights[i] = 1.0 / len(self.agent_completion_blocks)
                self.agent_completion_blocks[i] = 1.0
            self._completion_set(self._completion_compute())
            log(INFO, f"Completion: {self._completion_compute()}")

    def _completion_track(self, percentage:float) -> float:
        with self.agent_lock:
            if self.agent_completion_current_block < 0:
                # if the current block is not set, set it to 0
                self._completion_start_block()

            if percentage > self.agent_completion_blocks[self.agent_completion_current_block]:
                # set the completion percentage for the current block
                self.agent_completion_blocks[self.agent_completion_current_block] = percentage
                self._completion_set(self._completion_compute())
            
            return self.agent_completion_blocks[self.agent_completion_current_block]
        
    def _completion_compute(self) -> float:
        # calculate the completion percentage based on all the blocks
        percentage = 0.0
        for i in range(len(self.agent_completion_blocks)):
            if i < self.agent_completion_current_block:
                percentage += min(self.agent_completion_blocks_weights[i], 1.0)
            elif i == self.agent_completion_current_block:
                percentage += min(self.agent_completion_blocks_weights[i] * self.agent_completion_blocks[i], 1.0)

        return percentage

    def _completion_set(self, percentage:float) -> None:
        completion_percentage[0] = max(0.0, min(percentage, 1.0))

    def _start_blazor_progress_updates(self, goal: 'Goal', interval_seconds: float = 5.0) -> None:
        """Start a background thread that periodically pushes plan state to Blazor.
        Only sends if completion % or step title has changed since the last send.
        Call _stop_blazor_progress_updates() (e.g. in a finally block) to stop it.
        If a previous thread is still running it is stopped cleanly before starting a new one."""
        if goal is None:
            return
        # Stop any previously running timer to avoid orphaned threads.
        if self._blazor_stop_event is not None:
            self._blazor_stop_event.set()
        if self._blazor_thread is not None:
            self._blazor_thread.join(timeout=1.0)
            self._blazor_thread = None
        self._blazor_stop_event = threading.Event()
        self._blazor_last_completion = -1.0
        self._blazor_last_step = ""

        def _periodic_send():
            while not self._blazor_stop_event.wait(interval_seconds):
                with self.agent_lock:
                    current_completion = self._completion_compute()
                    current_step = (
                        self.agent_completion_block_titles[self.agent_completion_current_block]
                        if self.agent_completion_current_block >= 0 else ""
                    )
                if current_completion != self._blazor_last_completion or current_step != self._blazor_last_step:
                    send_plan_to_blazor(goal.goal_plan)
                    self._blazor_last_completion = current_completion
                    self._blazor_last_step = current_step

        self._blazor_thread = threading.Thread(
            target=_periodic_send,
            daemon=True,
            name=f"{self.agent_name}-BlazorTimer",
        )
        self._blazor_thread.start()

    def _stop_blazor_progress_updates(self, goal: 'Goal') -> None:
        """Stop the background Blazor progress thread and send one final update."""
        if self._blazor_stop_event is not None:
            self._blazor_stop_event.set()
        if self._blazor_thread is not None:
            self._blazor_thread.join(timeout=1.0)
            self._blazor_thread = None
        self._blazor_stop_event = None
        if goal is not None:
            send_plan_to_blazor(goal.goal_plan)

    def _get_last_user_prompt_or_goal_prompt(self, goal:Goal, ignore_augmented_prompt:bool=False, ignore_goal_prompt:bool=False) -> Tuple[Optional[Union[str,dict]], Optional[List[str]], Optional[Goal]]:
        """
                Retrieve the most recent prompt and optional keywords for a given goal.

                Return values:
                        tuple: (prompt, prompt_keywords)
                                - prompt (str | None): The consolidated prompt text. This can be:
                                        * the goal's explicit `goal.goal_prompt` (if set),
                                        * the most recent user prompt summary, or
                                        * an augmented prompt produced by a prompt-augmenter (may be a
                                            comma-separated string when assembled from a list).
                                          Note: ", " is used as the separator when joining list elements into a string.
                                - prompt_keywords (List[str] | None): Optional list of keywords
                                    extracted from an augmented prompt. If no keywords are available,
                                    this will be `None`.

                Notes:
                        - Both values can be `None` and callers should handle missing values.
                        - The function prefers augmented prompts when they are more recent
                            than the user/goal prompt (unless `ignore_augmented_prompt=True`).

                Examples:
                        # 1) Goal has an explicit prompt set
                        >>> # returns ("Summarize quarterly results", None)

                        # 2) No goal prompt, use latest user prompt summary
                        >>> # returns ("Find references about X and Y", None)

                        # 3) Augmented prompt provided as a dict with keywords
                        >>> augmented_prompt = {
                        ...     "augmented_prompt": ["Step 1: fetch data", "Step 2: summarize"],
                        ...     "thalabus_query": '"fetch", "summarize"'
                        ... }
                        >>> # returns ("Step 1: fetch data, Step 2: summarize", ["fetch", "summarize"])

                        # 4) Augmented prompt provided as a list
                        >>> augmented_prompt = ["a", "b", "c"]
                        >>> # returns ("a, b, c", None)

                Return type hints:
                        - `(str | None, List[str] | None)`
        """
        if goal is None:
            return prompt, prompt_keywords, None

        prompt_goal:Optional[Goal] = None
        prompt:Optional[Union[str, dict]] = None
        prompt_keywords:Optional[List[str]] = None
        
        if ignore_goal_prompt is not True:
            # identify goal's specific prompt, if available
            prompt = goal.goal_prompt if goal is not None else None

        # if no specific goal's prompt is found, get the latest user's prompt
        if prompt is None:
            prompt, prompt_goal = goal.get_last_summary(agent_name="AgentUser")

        if ignore_augmented_prompt is False:
            # get the latest augmented prompt
            augmented_prompt, augmented_prompt_goal = goal.get_last_summary(agent_name="AgentPromptAugmenter")

            augmented_prompt_is_more_recent:bool = True
            if prompt_goal is not None:
                # determine if the augmented prompt is more recent than the user prompt, and if so, use it
                augmented_prompt_is_more_recent = Goal.compare_id_chains(augmented_prompt_goal.get_id_chain(), prompt_goal.get_id_chain()) >= 0 if augmented_prompt_goal is not None and prompt_goal is not None else False

            if augmented_prompt_is_more_recent and augmented_prompt is not None:
                # augmented_prompt may be either:
                # - a dict with keys 'augmented_prompt' (array) and 'keywords' (comma-separated string)
                # - or directly an array/list of strings
                prompt_keywords_comma_separated = None
                if isinstance(augmented_prompt, dict):
                    if "augmented_prompt" in augmented_prompt:
                        candidate = augmented_prompt["augmented_prompt"]
                        if isinstance(candidate, list):
                            prompt = ", ".join([str(p) for p in candidate])
                        elif isinstance(candidate, dict):
                            prompt = candidate
                        else:
                            prompt = str(candidate)
                    prompt_keywords_comma_separated = augmented_prompt.get("thalabus_query") if "thalabus_query" in augmented_prompt else None
                elif isinstance(augmented_prompt, list):
                    prompt = ", ".join([str(p) for p in augmented_prompt])
                elif isinstance(augmented_prompt, str):
                    prompt = augmented_prompt

                if prompt_keywords_comma_separated is not None and len(prompt_keywords_comma_separated) > 0:
                    prompt_keywords = [kw.strip() for kw in prompt_keywords_comma_separated.split(",") if kw.strip() is not None and len(kw.strip()) > 0]

        if prompt is not None and len(prompt) == 0:
            prompt = None
        if prompt_keywords is not None and len(prompt_keywords) == 0:
            prompt_keywords = None

        return prompt, prompt_keywords, prompt_goal

    def _extract_prompt_components(self, prompt: Union[str, dict]) -> Tuple[Union[str, dict], Optional[List[str]], Optional[List[str]], Optional[str], Optional[str]]:
        """
        Extract prompt components from a prompt input that may be a dict.
        
        This method handles three extraction scenarios:
        1. When prompt contains "prompts" or "keywords" fields (traditional RAG)
        2. When prompt contains "thalabus_query" field (thalabus-based RAG)
        3. When prompt is a string (no extraction needed)
        
        Args:
            prompt: Input prompt - can be a string or dict with special fields
            
        Returns:
            Tuple of (prompt, prompt_chunks, prompt_keywords, thalabus_query, subset_of_dataset) where:
                - prompt: The normalized prompt string or dict
                - prompt_chunks: List of prompt chunks (if extracted from dict), None otherwise
                - prompt_keywords: List of keywords (if extracted from dict), None otherwise
                - thalabus_query: Thalabus query string (if extracted from dict), None otherwise
                - subset_of_dataset: Dataset id to restrict the search to (if extracted from dict), None otherwise
        """
        prompt_chunks: Optional[List[str]] = None
        prompt_keywords: Optional[List[str]] = None
        thalabus_query: Optional[str] = None
        subset_of_dataset: Optional[str] = None
        
        if isinstance(prompt, dict):
            # subset_of_dataset is independent of which branch below matches
            subset_of_dataset = prompt.get("subset_of_dataset", None)

            if "prompts" in prompt or "keywords" in prompt:
                # Extract traditional RAG components (prompts and keywords)
                prompt_chunks = prompt.get("prompts", None)
                prompt_keywords = prompt.get("keywords", None)
                
                if prompt_chunks is not None:
                    prompt = ", ".join(prompt_chunks)
                else:
                    prompt = prompt.get("prompt", None)
                    
            elif "thalabus_query" in prompt:
                # Extract thalabus query component
                thalabus_query = prompt.get("thalabus_query", None)
                # Keep original prompt as-is, will be handled by the caller
        
        return prompt, prompt_chunks, prompt_keywords, thalabus_query, subset_of_dataset

    def _get_and_extract_prompt_components(self, goal: 'Goal') -> Tuple[Union[str, dict, None], Optional[List[str]], Optional[List[str]], Optional[str], Optional[str]]:
        """
        Get prompt from goal and extract its components.
        
        This method follows a prioritized sequence:
        1. Retrieve the prompt via _get_last_user_prompt_or_goal_prompt(), which uses goal.goal_prompt
           as the base and then overrides it with an AgentPromptAugmenter augmented prompt when
           one is available and more recent (supplying the thalabus_query).
        2. Extract components (chunks, keywords, thalabus_query, subset_of_dataset) from the found prompt.
        
        Args:
            goal: The Goal object containing potential prompts
            
        Returns:
            Tuple of (prompt, prompt_chunks, prompt_keywords, thalabus_query, subset_of_dataset) where:
                - prompt: The normalized prompt string or dict (or None if not found)
                - prompt_chunks: List of prompt chunks (if extracted), None otherwise
                - prompt_keywords: List of keywords (if extracted or from last prompt), None otherwise
                - thalabus_query: Thalabus query string (if extracted), None otherwise
                - subset_of_dataset: Dataset id to restrict the search to (if extracted), None otherwise
        """
        prompt: Union[str, dict, None] = None
        prompt_keywords: Optional[List[str]] = None
        prompt_chunks: Optional[List[str]] = None
        thalabus_query: Optional[str] = None
        subset_of_dataset: Optional[str] = None
        
        # Get the prompt via _get_last_user_prompt_or_goal_prompt, which handles goal.goal_prompt
        # internally AND checks for an AgentPromptAugmenter augmented prompt (with thalabus_query).
        # Previously, step 1 set prompt = goal.goal_prompt directly and skipped step 2, which meant
        # AgentPromptAugmenter's thalabus_query was never found when goal_prompt was a plain string
        # propagated from the parent goal by AgentPlanner.
        if goal is not None:
            prompt, prompt_keywords, prompt_goal = self._get_last_user_prompt_or_goal_prompt(goal)
        
        # Extract components from the found prompt
        if prompt is not None:
            prompt, extracted_chunks, extracted_keywords, thalabus_query, subset_of_dataset = self._extract_prompt_components(prompt)
            
            # Use extracted values, preferring explicit extraction over fallback keywords
            if extracted_chunks is not None:
                prompt_chunks = extracted_chunks
            if extracted_keywords is not None:
                prompt_keywords = extracted_keywords
        
        return prompt, prompt_chunks, prompt_keywords, thalabus_query, subset_of_dataset

    def _find_dataset_name_for_output(self, goal: Optional[Goal]) -> Optional[str]:
        """Walk the goal dependency tree to find an AgentPromptAugmenter and extract its dataset_name."""
        if goal is None:
            return None
        try:
            for dep in (goal.get_dependent_goals() or []):
                for candidate in (dep.get_dependent_goals() if dep is not None else []):
                    if candidate is None:
                        continue
                    if not candidate.is_goal_target_agent_name("AgentPromptAugmenter"):
                        continue
                    try:
                        summary = candidate.get_summary()
                        if summary is None:
                            continue
                        name = self._extract_dataset_name_from_summary(summary)
                        if name:
                            return name
                    except Exception:
                        continue
        except Exception as ex:
            log(DEBUG, f"AgentReranker._find_dataset_name_for_output: tree walk error: {ex}")
        return None

    @staticmethod
    def _parse_key_value_response(response: str) -> dict:
        """Parse a plain-text 'key: value' response into a dict.

        - Strips leading/trailing blank lines and whitespace
        - Tolerates extra whitespace around the colon separator and values
        - Splits on the first colon only (values may contain colons, e.g. thalabus queries)
        - Lines without a colon are ignored
        """
        result = {}
        if not response:
            return result
        for line in response.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if ':' in line:
                key, _, value = line.partition(':')
                result[key.strip()] = value.strip()
        return result

    @staticmethod
    def _should_skip_goal_prompt_message(messages: List[dict], goal_prompt: Optional[str]) -> bool:
        """Check if goal_prompt is identical to any existing system message's content.
        
        Extracts the actual prompt text from within XML tags (<PLAN_SYSTEM_PROMPT>...</PLAN_SYSTEM_PROMPT>)
        if present, or compares directly if the exact goal_prompt already exists as a system message.
        
        Returns True if the goal_prompt should be skipped (already exists as a system message),
        False if it should be added.
        
        Args:
            messages: List of messages to check
            goal_prompt: The goal prompt to check for duplicates
            
        Returns:
            True if goal_prompt matches any system message content, False otherwise
        """
        if goal_prompt is None or not isinstance(goal_prompt, str):
            return True
        
        goal_prompt_clean = goal_prompt.strip()
        
        for msg in messages:
            if msg.get('role') == 'system':
                content = msg.get('content')
                if isinstance(content, str):
                    content_clean = content.strip()
                    
                    # Direct match: exact content equals goal_prompt
                    if content_clean == goal_prompt_clean:
                        return True
                    
                    # Extract text between <PLAN_SYSTEM_PROMPT> tags and compare
                    if '<PLAN_SYSTEM_PROMPT>' in content_clean:
                        import re
                        match = re.search(r'<PLAN_SYSTEM_PROMPT>\s*(.*?)\s*</PLAN_SYSTEM_PROMPT>', content_clean, re.DOTALL)
                        if match:
                            extracted_text = match.group(1).strip()
                            if extracted_text == goal_prompt_clean:
                                return True
        
        return False

    @staticmethod
    def _append_goal_prompt_message(messages: List[dict], goal_prompt: Optional[str], normalize_content: bool = False) -> None:
        """Append a goal_prompt message to the messages list if it doesn't already exist.
        
        Checks for duplicates before appending. If goal_prompt is identical to any existing system 
        message content, it will not be appended.
        
        Args:
            messages: List of messages to append to (modified in-place)
            goal_prompt: The goal prompt to add
            normalize_content: If True, applies normalize_hyphens() to the content before appending
        """
        if goal_prompt is None or Agent._should_skip_goal_prompt_message(messages, goal_prompt):
            return
        
        content = f"There are specific instructions about how to format the information and output it to the user, consider them: `{goal_prompt}`"
        
        if normalize_content:
            from src.utils.text_utils import normalize_hyphens
            content = normalize_hyphens(content)
        
        messages.append({
            "role": "system",
            "content": content
        })

log_end_file(__file__)
