from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import html
import json
import uuid
import concurrent.futures
import threading
import time
import traceback
import pytz
import tzlocal

from datetime import datetime
from typing import TYPE_CHECKING, List, Tuple, Union, Optional, Union, Dict, Any, Set

from src.config.app_config import app_config
from src.config.constants import MIN_HINT_SCORE, TOP_N_HINTS, MIN_RERANKING_SCORE_FRAG, MAX_RERANKING_OUTPUT_DOC
from src.utils.text_utils import strip_html_tag_ids, add_html_tag_ids
from src.core.base.enums import FocusIterationType, HintRecommendation
from src.core.base.hints import get_best_hints_for_agent, get_pending_consolidation_hints_for_agent
from src.core.base.reasoning.ReasoningDocument import ReasoningDocument
from src.core.base.InputOutput import is_paused, is_stopped, send_plan_to_blazor, encode_image, execute_neo4j_query
from src.core.base.ResourceMgmt import get_lock
from src.core.llm.manager import LlmManager
from src.interfaces.ontology_interface import OntologyInterface
from src.interfaces.dossiers_interface import DossiersInterface
from src.interfaces.documents_interface import DocumentsInterface
from src.interfaces.dataset_interface import DatasetInterface
from src.services.ontology_service import get_ontology_service, linked_document_budget_per_entity
from src.services.dossiers_service import get_dossiers_service
from src.services.documents_service import get_documents_service
from src.services.dataset_service import get_dataset_service
from src.services.user_service import get_user_service
from src.services.ontology_source_helper import parse_kg_source_token
from src.core.thalaplan.goal.enums import GoalStatus, GoalTarget, GoalValidation, GoalAsyncStatus
from src.core.thalaplan.goal.exceptions import GoalCancelledException
from src.utils.text_utils import strip_html_tag_ids, add_html_tag_ids, generate_html_from_text, add_html_tag_ids, eliminate_duplicate_newlines, clean_fragment_html, strip_linked_document_blocks
from src.utils.log import log, log_plan, INFO, DEBUG, PERF, WARNING, ERROR, FATAL

if TYPE_CHECKING:
    from src.core.agents.Agent import Agent
    from src.core.thalaplan.Plan import Plan


PARALLEL_GOALS_ENABLE = True        # enable parallel goals execution
PARALLEL_GOALS_MAX = 100            # maximum number of parallel goals to run
MAX_ERROR_REPEAT_COUNT = 3          # maximum number of times to repeat a goal on error
TRUNCATED_MESSAGE = """
...(truncated)...
</FRAGMENTS>

<INSTRUCTIONS_FOR_DEALING_WITH_TRUNCATED_MESSAGE>
The AgentAssistant **MUST** inform the user, that too many "information sources" were retrieved for this goal, and that the response may be incomplete or less accurate as a result.
The AgentAssistant must as well recommend the user to narrow down the focus of the goal, or to increase the relevance of the RAG retrieval (for example: by improving the search query, or by filtering the source documents).
To inform the user, the AgentAssistant must use an html <p class="alert alert-warning">...</p> block, in the conversation language.

Below you find a complete list of document ids that were retrieved for this goal.
</INSTRUCTIONS_FOR_DEALING_WITH_TRUNCATED_MESSAGE>
"""
MAX_TO_JSON_RECURSE_LEVEL = 16

# Cache of the current user's profile memories (:User node attributes) keyed by user identity.
# The cache has a short TTL so the lookup does not run on every LLM call of the same plan turn,
# yet profile edits made in the settings UI become visible to the chatbot without a restart.
_user_memories_cache: Dict[str, Tuple[float, Optional[Dict[str, str]]]] = {}
_user_memories_cache_lock = threading.Lock()
USER_MEMORIES_CACHE_TTL_SECONDS: float = 30.0

#region utilities
def instantiate_agent(module_name: str, class_name: str, agent_name: str):
    if class_name is None:
        return None
    module = __import__(module_name, fromlist=[class_name])
    class_ = getattr(module, class_name)
    agent = class_(agent_name)
    return agent

def get_status_color_for_status(status:GoalStatus) -> str:
    status_color =  "\033[1;32m" if status == GoalStatus.COMPLETED else \
                    "\033[1;37m" if status == GoalStatus.NOT_STARTED else \
                    "\033[1;34m" if status == GoalStatus.IN_PROGRESS or status == GoalStatus.CHILDREN_IN_PROGRESS else \
                    "\033[1;35m" if status == GoalStatus.RUN_EVERY_TIME else \
                    "\033[1;31m" if status == GoalStatus.FAILED else "\033[0m"
    return status_color
#endregion utilities


class Goal:
    """Goal class for managing individual goals within a plan."""

    def __init__(self, plan:'Plan', parent:'Goal', title:str, description:str, target:GoalTarget, agent:'Agent', dependent_upon:List[int], prompt:str=None, async_status:GoalAsyncStatus=None) -> None:
        # initialize goal object

        # global attributes
        self.goal_guid: str = str(uuid.uuid4())
        self.goal_plan: 'Plan' = plan
        self.goal_parent: 'Goal' = parent
        self.goal_title: str = title
        self.goal_description: str = description

        # attributes specifically micro-planning and tool-preparation
        self.goal_prompt: Optional[Union[str, dict]] = prompt
        self.goal_recommended_tools = None      # TODO: identify the type correctly
        self.goal_recommended_plan: Optional[str] = None
        self.goal_focus_docids: Optional[List[str]] = None
        self.goal_focus_iteration: FocusIterationType = None
        self.goal_focus_regex: Optional[str] = None
        self.goal_focus_instr: Optional[str] = None
        self.goal_focus_reasoning_division: Optional[str] = None

        # attributes specifically for images
        self.goal_images: Optional[str] = None

        # goal overriding attributes (for example: to override the default prompt, recommended tools or plan for a goal, based on user input or agent decisions)
        self.goal_override_role: Optional[str] = None
        self.goal_override_json_output: Optional[dict] = None
        self.goal_override_structured_output: Optional[dict] = None
        self.goal_override_recommended_plan: Optional[str] = None
        self.goal_override_recommended_tools: Optional[list] = None

        # attributes that define the goal's tree structure
        self.goal_sequence_id: Optional[str] = None
        self.goal_subgoals: List['Goal'] = []                           # array of Goal objects
        self.goal_parallel_subgoals: List['Goal'] = []                  # subgoals that can be executed in parallel, part of the runtime parallel handling
        self.goal_dependent_upon: Optional[List[int]] = dependent_upon  # optional array of integer, offset from the current goal_id()

        # attributes for status
        self.goal_status: GoalStatus = GoalStatus.NOT_STARTED
        self.goal_async_status: GoalAsyncStatus = async_status if async_status is not None else GoalAsyncStatus.SYNC
        self.goal_async_thread: Optional[threading.Thread] = None       # background thread of an async execution, kept so that it can be cancelled and awaited
        self.goal_error: Optional[str] = None
        self.goal_error_message: Optional[str] = None
        self.goal_error_repeat_count: int = 0

        # attributes that regulate which agent or pseudo-agent is responsible for the goal
        self.goal_target: GoalTarget = target
        _agent_class = agent if isinstance(agent, type) else type(agent)
        self.goal_target_agent: Optional['Agent'] = instantiate_agent(module_name=_agent_class.__module__, class_name=_agent_class.__name__, agent_name=_agent_class.__module__) if agent else None
        self.set_target_agent_name(agent.__module__ if agent else None)

        # attributes for goal's outputs
        self.goal_summary: Optional[str] = None
        self.goal_summary_pretty: Optional[str] = None # A "pretty" version of the goal summary, suitable for display in UIs or to be passed to the LLM for better comprehension (for example: with markdown or html formatting); this is separate from `goal_summary` to allow LLMs to consume a plain-text version if needed. This attribute is not copied by `clone()`, not serialized in `to_json()`, and is only set via `set_summary(..., pretty=...)`
        self.goal_agent_did_nothing: bool = False
        self.goal_planner_outputs: Optional[List[str]] = None
        self.goal_summary_suppress: bool = False  # When True, suppress/hide this goal's user-facing summary. get_summary() returns None for suppressed summaries; the flag is copied by `clone()`, emitted as `goal_summary_suppress` in `to_json()`, and is used (for example) when re-inserting/re-running goals after errors to avoid duplicating messages to the user. Note: the Consolidator reads goal_summary directly and is not affected by this flag.
        self.goal_silent_summary: bool = False  # When True, hide the user-facing summary (visible to LLM agents via `goal_internal_thought`); overruled by agent settings
        self.goal_repeat_on_error: bool = False
        self.goal_purpose: Optional[str] = None
        self.goal_html_summary: Optional[str] = None
        self.goal_tool_call_message: Optional[str] = None
        self.goal_proposed_followups: Optional[list] = None
        self.goal_blazor_component: Optional[str] = None
        self.goal_internal_thought: Optional[str] = None

        # quality attributes
        self.goal_validation: GoalValidation = GoalValidation.NOT_VALIDATED  # Validation status of the goal's summary, set by AgentVerifier

        # RAG and Dataset management
        self.goal_documents: Optional[str] = None
        self.goal_datasets: Optional[List[dict]] = None
        self.goal_rag_fragments: Optional[List[str]] = None
        self.goal_rag_docids: Optional[List[str]] = None
        self.goal_rag_toomanyfragments: Optional[bool] = None

        # attributes that track the plan execution
        self.goal_instantiated_plan: Optional[str] = None
        self.goal_completion_reason: Optional[str] = None

        # miscellaneous attributes
        self.goal_attributes: Optional[dict] = None

        # attributes for debugging
        self.goal_debug: Optional[dict] = None
        self.goal_last_update_datetime: datetime = datetime.now()

        # performance tracking attributes
        self.goal_perf_lock: threading.Lock = threading.Lock() # lock to protect concurrent += on perf counters
        self.goal_perf_run_duration_ms: float = 0.0 # total goal duration in milliseconds, set at the end of the goal execution
        self.goal_perf_db_duration_ms: float = 0.0 # total duration of database calls made during the goal execution
        self.goal_perf_llm_duration_ms: float = 0.0 # total duration of LLM calls made during the goal execution
        self.goal_perf_embeddings_duration_ms: float = 0.0 # total duration of embedding API calls made during the goal execution
        self.goal_perf_reranking_duration_ms: float = 0.0 # total duration of reranking calls made during the goal execution
        self.goal_perf_external_calls_duration_ms: float = 0.0 # total duration of external calls (e.g. REST API calls) made during the goal execution

        # update parent's subgoals
        if self.goal_parent and not self in self.goal_parent.goal_subgoals:
            self.goal_parent.goal_subgoals.append(self)

    def get_goal_planner_outputs(self) -> Optional[List[str]]:
        return self.goal_planner_outputs

    def set_goal_planner_outputs(self, value: Optional[List[str]]) -> None:
        self.goal_planner_outputs = value

    def clone(self) -> 'Goal':
        dependent_upon_copy = self.goal_dependent_upon.copy() if self.goal_dependent_upon else None
        clone = Goal(self.goal_plan, None, self.goal_title, self.goal_description, self.goal_target, self.goal_target_agent, dependent_upon_copy)
        clone.goal_guid = self.goal_guid
        clone.goal_subgoals = [subgoal.clone() for subgoal in self.goal_subgoals]
        clone.goal_status = self.goal_status
        clone.set_async_status(self.get_async_status())
        clone.goal_error = self.goal_error
        clone.goal_error_message = self.goal_error_message
        clone.goal_error_repeat_count = self.goal_error_repeat_count
        clone.goal_title = self.goal_title
        clone.goal_description = self.goal_description
        clone.set_target_agent_name(self.goal_target_agent_name)
        clone.goal_prompt = self.goal_prompt
        clone.goal_recommended_tools = self.goal_recommended_tools
        clone.goal_recommended_plan = self.goal_recommended_plan
        clone.goal_focus_docids = self.goal_focus_docids
        clone.goal_focus_iteration = self.goal_focus_iteration
        clone.goal_focus_regex = self.goal_focus_regex
        clone.goal_focus_instr = self.goal_focus_instr
        clone.goal_focus_reasoning_division = self.goal_focus_reasoning_division
        clone.goal_images = self.goal_images
        clone.goal_documents = self.goal_documents
        clone.goal_override_role = self.goal_override_role
        clone.goal_override_json_output = self.goal_override_json_output
        clone.goal_override_structured_output = self.goal_override_structured_output
        clone.goal_override_recommended_plan = self.goal_override_recommended_plan
        clone.goal_override_recommended_tools = self.goal_override_recommended_tools
        clone.goal_sequence_id = self.goal_sequence_id
        clone.goal_summary = self.goal_summary
        clone.goal_summary_pretty = self.goal_summary_pretty
        clone.goal_agent_did_nothing = self.goal_agent_did_nothing
        planner_outputs = self.get_goal_planner_outputs()
        clone.set_goal_planner_outputs(planner_outputs.copy() if planner_outputs else None)
        clone.goal_silent_summary = self.goal_silent_summary
        clone.goal_summary_suppress = self.goal_summary_suppress
        clone.goal_repeat_on_error = self.goal_repeat_on_error
        clone.goal_validation = self.goal_validation
        clone.goal_purpose = self.goal_purpose
        clone.goal_tool_call_message = self.goal_tool_call_message
        clone.goal_proposed_followups = self.goal_proposed_followups
        clone.goal_html_summary = self.goal_html_summary
        clone.goal_blazor_component = self.goal_blazor_component
        clone.goal_rag_fragments = self.goal_rag_fragments
        clone.goal_rag_docids = self.goal_rag_docids
        clone.goal_rag_toomanyfragments = self.goal_rag_toomanyfragments
        clone.goal_internal_thought = self.goal_internal_thought
        clone.goal_instantiated_plan = self.goal_instantiated_plan
        clone.goal_completion_reason = self.goal_completion_reason
        clone.goal_attributes = self.goal_attributes
        clone.goal_debug = self.goal_debug
        clone.goal_last_update_datetime = self.goal_last_update_datetime
        clone.set_perf_run_ms(self.goal_perf_run_duration_ms)
        clone.add_perf_db_ms(self.goal_perf_db_duration_ms)
        clone.add_perf_llm_ms(self.goal_perf_llm_duration_ms)
        clone.add_perf_embeddings_ms(self.goal_perf_embeddings_duration_ms)
        clone.add_perf_reranking_ms(self.goal_perf_reranking_duration_ms)
        clone.add_perf_external_calls_ms(self.goal_perf_external_calls_duration_ms)
        return clone
    
    def __str__(self):
        return self.full_str(recursive = True)
    
    def full_str(self, recursive:bool=True) -> str:
        status_color = self.get_status_color()
        s = f"""
\033[1;30mGoal ID:\033[0m {'.'.join(str(i) for i in self.get_id_chain())}
\033[1;30mGoal Title:\033[0m {self.goal_title}
\033[1;30mGoal Description:\033[0m \033[1;90m{self.goal_description}\033[0m
"""
        if self.goal_error:
            s += f"""\033[1;30mGoal Error:\033[0m {self.goal_error}\n"""
        if self.goal_error_message:
            s += f"""\033[1;30mGoal Error Message:\033[0m {self.goal_error_message}\n"""
        if self.goal_sequence_id:
            s += f"""\033[1;30mGoal Sequence ID:\033[0m \033[0;31m{self.goal_sequence_id}\033[0m\n"""
        if self.goal_prompt:
            s += f"""\033[1;30mGoal Prompt:\033[0m \033[0;31m{self.goal_prompt}\033[0m\n"""
        if self.goal_recommended_tools:
            s += f"""\033[1;30mGoal Recommended Tools: \033[0;31m{self.goal_recommended_tools}\033[0m\n"""
        if self.goal_recommended_plan:
            s += f"""\033[1;30mGoal Recommended Plan: \033[0;31m{self.goal_recommended_plan}\033[0m\n"""
        if self.goal_focus_docids:
            s += f"""\033[1;30mGoal Focus Document: \033[0;31m{self.goal_focus_docids}\033[0m\n"""
        if self.goal_focus_iteration:
            s += f"""\033[1;30mGoal Focus Iteration: \033[0;31m{self.goal_focus_iteration}\033[0m\n"""
        if self.goal_focus_regex:
            s += f"""\033[1;30mGoal Focus Regex: \033[0;31m{self.goal_focus_regex}\033[0m\n"""
        if self.goal_focus_instr:
            s += f"""\033[1;30mGoal Focus Instructions: \033[0;31m{self.goal_focus_instr}\033[0m\n"""
        if self.goal_focus_reasoning_division:
            s += f"""\033[1;30mGoal Focus Reasoning Division: \033[0;31m{self.goal_focus_reasoning_division}\033[0m\n"""
        if self.goal_override_role:
            s += f"""\033[1;30mGoal Override Role: \033[0;31m{self.goal_override_role}\033[0m\n"""
        if self.goal_override_json_output:
            s += f"""\033[1;30mGoal Override JSON Output: \033[0;31m{self.goal_override_json_output}\033[0m\n"""
        if self.goal_override_structured_output:
            s += f"""\033[1;30mGoal Override Structured Output: \033[0;31m{self.goal_override_structured_output}\033[0m\n"""
        if self.goal_override_recommended_plan:
            s += f"""\033[1;30mGoal Override Recommended Plan: \033[0;31m{self.goal_override_recommended_plan}\033[0m\n"""
        if self.goal_override_recommended_tools:
            s += f"""\033[1;30mGoal Override Recommended Tools: \033[0;31m{self.goal_override_recommended_tools}\033[0m\n"""
        if self.goal_dependent_upon:
            s += f"""\033[1;30mGoal Dependent upon:\033[0m {", ".join(self.get_id_chain_as_str(-1) + "." + str(goal + self.goal_id()) for goal in self.goal_dependent_upon) if self.goal_dependent_upon else "(-)"}{" (" + self.get_dependencies_status().value + ")" if self.goal_dependent_upon else ""}
"""
        s += f"""\033[1;30mGoal Status:\033[0m {status_color}{self.goal_status}{f" ({self.goal_completion_reason})" if self.goal_completion_reason else ""}\033[0m
\033[1;30mGoal Target:\033[0m \033[1;30m{self.goal_target}\033[0m
\033[1;30mGoal Target Agent:\033[0m \033[1;37m{self.goal_target_agent.agent_name if self.goal_target_agent else None}\033[0m
\033[1;30mGoal Last Update:\033[0m {self.goal_last_update_datetime}
"""
        if self.goal_rag_docids is not None:
            s += f"""\033[1;30mGoal RAG DocIDs:\033[0m {self.goal_rag_docids}\n"""
        if self.goal_internal_thought is not None:
            s += f"""\033[1;30mGoal Internal Thought:\033[0m \033[1;34m{self.goal_internal_thought}\033[0m\n"""
        if self.goal_instantiated_plan is not None:
            s += f"""\033[1;30mGoal Instantiated Plan:\033[0m \033[0;31m{self.goal_instantiated_plan}\033[0m\n"""
        datasets = self.get_datasets()
        if datasets is not None:
            s += f"""\033[1;30mGoal Datasets:\033[0m {datasets}\n"""
        if self.goal_rag_fragments is not None:
            if len(self.goal_rag_fragments) == 0:
                s += f"""\033[1;30mGoal RAG Fragments:\033[0m \033[1;34m[]\033[0m (zero fragments identified)\n"""
            else:
                s += f"""\033[1;30mGoal RAG Fragments:\033[0m [\n"""
                for fragment in self.goal_rag_fragments:
                    if isinstance(fragment, dict):
                        raw_text = fragment.get("text") or fragment.get("content") or fragment.get("body") or fragment.get("fragment_text") or ""
                        fragment_text = str(raw_text).replace("\n", "<br/>").replace("\r", "<br/>")
                        if len(fragment_text) > 80:
                            fragment_text = fragment_text[:80] + "..."
                        s += f"""{{
    "document_id": "\033[1;34m{fragment["documentid"] if "documentid" in fragment else "N/A"}\033[0m",
    "fragment_id": "\033[1;34m{fragment["fragmentid"] if "fragmentid" in fragment else "N/A"}\033[0m",
    "ranking": "\033[1;34m{fragment["ranking"] if "ranking" in fragment else "N/A"}\033[0m",
    "ranking_text": "\033[1;34m{fragment["ranking_text"] if "ranking_text" in fragment else "N/A"}\033[0m",
    "text": "\033[1;34m{format(fragment_text)}\033[0m"
}},
"""
                    else:
                        s += f"""    "fragment_id": "\033[1;34m{fragment}\033[0m"\n"""
                s += f"\033[0m]\n"
        if self.goal_rag_toomanyfragments:
            s += f"""\033[1;30mGoal RAG Too Many Fragments:\033[0m \033[1;31mTrue\033[0m\n"""
        if self.goal_attributes is not None:
            s += f"""\033[1;30mGoal Attributes:\033[0m {self.goal_attributes}\n"""
        if self.goal_summary is not None:
            if isinstance(self.goal_summary, str):
                s += f"""\033[1;30mGoal Summary:\033[0m \033[1;32m<<{self.goal_summary.strip()}>>"""
            else:
                s += f"""\033[1;30mGoal Summary:\033[0m \033[1;32m{self.goal_summary}"""
            if self.goal_silent_summary == True:
                s += f""" (***SILENT***)"""
            if self.goal_summary_suppress == True:
                s += f""" (***SUPPRESS***)"""
            s += "\033[0m\n"
        if self.goal_summary_pretty is not None:
            s += f"""\033[1;30mGoal Summary Pretty:\033[0m \033[1;32m<<{self.goal_summary_pretty.strip()}>>\033[0m\n"""
        if self.goal_repeat_on_error == True:
            s += f""" (***REPEAT ON ERROR***)"""
        if self.goal_error_repeat_count > 0:
            s += f"""\033[1;30mGoal Error Repeat Count:\033[0m \033[1;31m{self.goal_error_repeat_count}\033[0m\n"""
        if self.goal_purpose is not None:
            s += f"""\033[1;30mGoal Purpose:\033[0m \033[1;32m{self.goal_purpose}\033[0m\n"""
        if self.goal_images is not None:
            s += f"""\033[1;30mGoal Images:\033[0m \033[1;32m<<{self.goal_images}>>\033[0m\n"""
        if self.goal_documents is not None:
            s += f"""\033[1;30mGoal Documents:\033[0m \033[1;32m<<{self.goal_documents}>>\033[0m\n"""
        if self.goal_tool_call_message is not None:
            s += f"""\033[1;30mGoal Tool Call Reponse:\033[0m {self.goal_tool_call_message}\n"""
        if self.goal_proposed_followups is not None:
            s += f"""\033[1;30mGoal Proposed Followups:\033[0m {', '.join(self.goal_proposed_followups)}\n"""
        if self.goal_html_summary is not None:
            s += f"""\033[1;30mGoal HTML Summary:\033[0m \033[1;32m{self.goal_html_summary}\033[0m\n"""
        if self.goal_blazor_component is not None:
            s += f"""\033[1;30mGoal Blazor Component:\033[0m \033[1;32m{self.goal_blazor_component}\033[0m\n"""
        if self.goal_target_agent is not None and self.goal_target_agent.agent_execution is not None and len(self.goal_target_agent.agent_execution) > 0:
            s += f"""\033[1;30mGoal Execution:\033[0m \033[1;32m{self.goal_target_agent.agent_execution}\033[0m\n"""
        if self.goal_debug is not None:
            s += f"""\033[1;30mGoal Debug:\033[0m {self.goal_debug}\n"""

        if recursive and self.goal_subgoals:
            s += f"Goal Subgoals: {len(self.goal_subgoals)}"
            for subgoal in self.goal_subgoals:
                s += f"\n{subgoal.full_str(recursive=recursive)}"
        return s

    def get_aggregated_perf_counters(self) -> dict:
        """
        Get performance counters for this goal and all its descendants (recursively).
        If the goal has no subgoals, returns only its own counters.
        If the goal has subgoals, returns the sum of its own counters + all descendants.
        
        Returns:
            dict with keys: 'run_duration_ms', 'db_duration_ms', 'llm_duration_ms', 'external_calls_duration_ms'
        """
        run_duration = self.goal_perf_run_duration_ms
        db_duration = self.goal_perf_db_duration_ms
        llm_duration = self.goal_perf_llm_duration_ms
        embeddings_duration = self.goal_perf_embeddings_duration_ms
        reranking_duration = self.goal_perf_reranking_duration_ms
        external_calls_duration = self.goal_perf_external_calls_duration_ms
        
        # Recursively add all descendants' counters
        if self.goal_subgoals:
            for subgoal in self.goal_subgoals:
                subgoal_counters = subgoal.get_aggregated_perf_counters()
                run_duration += subgoal_counters['run_duration_ms']
                db_duration += subgoal_counters['db_duration_ms']
                llm_duration += subgoal_counters['llm_duration_ms']
                embeddings_duration += subgoal_counters['embeddings_duration_ms']
                reranking_duration += subgoal_counters['reranking_duration_ms']
                external_calls_duration += subgoal_counters['external_calls_duration_ms']
        
        return {
            'run_duration_ms': run_duration,
            'db_duration_ms': db_duration,
            'llm_duration_ms': llm_duration,
            'embeddings_duration_ms': embeddings_duration,
            'reranking_duration_ms': reranking_duration,
            'external_calls_duration_ms': external_calls_duration
        }

    def add_perf_db_ms(self, ms: float) -> None:
        with self.goal_perf_lock:
            self.goal_perf_db_duration_ms += ms

    def add_perf_llm_ms(self, ms: float) -> None:
        with self.goal_perf_lock:
            self.goal_perf_llm_duration_ms += ms

    def add_perf_embeddings_ms(self, ms: float) -> None:
        with self.goal_perf_lock:
            self.goal_perf_embeddings_duration_ms += ms

    def add_perf_reranking_ms(self, ms: float) -> None:
        with self.goal_perf_lock:
            self.goal_perf_reranking_duration_ms += ms

    def add_perf_external_calls_ms(self, ms: float) -> None:
        with self.goal_perf_lock:
            self.goal_perf_external_calls_duration_ms += ms

    def set_perf_run_ms(self, ms: float) -> None:
        with self.goal_perf_lock:
            self.goal_perf_run_duration_ms = ms

    def get_dependent_upon_as_str(self) -> str:
        dependent_upon = "(-)"
        try:
            # Build a human-readable string for dependencies (e.g. "1.2.3, 1.2.4")
            if self.goal_dependent_upon:
                # Convert offsets to absolute ids and sort them so the output is in sequence order
                parent_prefix = self.get_id_chain_as_str(-1)
                base_prefix = (parent_prefix + ".") if parent_prefix else ""
                abs_pairs: List[tuple[int, int]] = []  # list of (absolute_id, original_offset)
                for dep in self.goal_dependent_upon:
                    abs_id = int(dep) + self.goal_id()
                    if abs_id < 1:
                        continue  # or decide how to handle
                    abs_pairs.append((abs_id, dep))

                # sort by absolute id to ensure sequence order
                abs_pairs.sort(key=lambda p: p[0])

                dependent_parts: List[str] = [f"{base_prefix}{p[0]}" for p in abs_pairs]
                dependent_upon = ", ".join(dependent_parts)
        except Exception as e:
            log(ERROR, f"Error building dependent_upon string for goal {self.get_id_chain_as_str()}: {e}\n{traceback.format_exc()}")
        return dependent_upon
    
    def to_json(self, include_only_goal_id:List[int]=None, recurse_level:int=0) -> dict:
        isPartial = False

        if include_only_goal_id is not None and len(include_only_goal_id)<2:
            include_only_goal_id = None

        if self.goal_rag_fragments is not None:
            rag_fragments = []
            for fragment in self.goal_rag_fragments:
                if isinstance(fragment, dict):
                    # convert the fragment to a shorter version, eliminating the text
                    rag_fragments.append({
                        "documentid": fragment["documentid"] if "documentid" in fragment else None,
                        "fragmentid": fragment["fragmentid"] if "fragmentid" in fragment else None,
                        "ranking": fragment["ranking"] if "ranking" in fragment else None,
                        "ranking_text": fragment["ranking_text"] if "ranking_text" in fragment else None
                    })
                elif isinstance(fragment, str):
                    # fragment is a string with the fragmentid inside: keep it as is
                    rag_fragments.append({"fragmentid": fragment})
        else:
            rag_fragments = None

        if isinstance(self.goal_instantiated_plan, dict):
            instantiated_plan = json.dumps(self.goal_instantiated_plan)
        elif isinstance(self.goal_instantiated_plan, list):
            instantiated_plan = json.dumps(self.goal_instantiated_plan)
        else:
            instantiated_plan = self.goal_instantiated_plan

        if include_only_goal_id is None or len(self.get_id_chain()) > 2:
            subgoals = self.goal_subgoals
        else:
            subgoals = self.goal_subgoals

            # filter subgoals based on include_only_goal_id: only take the first two levels of the include_only_goal_id
            filter_to = include_only_goal_id[:2]
            if self.goal_parent is None and self.goal_id() != filter_to[0]:
                # 'self' is at top level: exclude the not-matching with filter_to[0]
                isPartial = True
                subgoals = None
            elif self.goal_parent is not None and self.goal_parent.goal_parent is None and self.goal_id() != filter_to[1]:
                # 'self' is at second level: exclude the not-matching with filter_to[1]
                if self.goal_subgoals is not None and len(self.goal_subgoals) > 0:
                    isPartial = True
                subgoals = None

        if self.goal_debug is not None:
            goal_debug = json.dumps(self.goal_debug)
        else:
            goal_debug = None

        summary = self.get_summary() if self.get_summary() is not None else None
        if summary is not None and (isinstance(summary, dict) or isinstance(summary, list)):
            summary = json.dumps(summary)

        if recurse_level >= MAX_TO_JSON_RECURSE_LEVEL:
            subgoals = None

        dependent_upon = self.get_dependent_upon_as_str()
        
        # Get aggregated performance counters (including descendants if any)
        aggregated_perf = self.get_aggregated_perf_counters()
        
        goal_json = {
            "goal_id": self.get_id_chain(),
            "goal_guid": self.goal_guid,
            "goal_parent": self.goal_parent.goal_id() if self.goal_parent else None,
            "goal_dependent_upon": dependent_upon,
            "goal_dependencies_status": self.get_dependencies_status().value,
            "goal_title": self.goal_title,
            "goal_description": self.goal_description,
            "goal_sequence_id": self.goal_sequence_id,
            "goal_prompt": str(self.goal_prompt) if self.goal_prompt is not None else None,
            "goal_recommended_tools": self.goal_recommended_tools,
            "goal_recommended_plan": self.goal_recommended_plan,
            "goal_focus_docids": self.goal_focus_docids,
            "goal_focus_iteration": self.goal_focus_iteration,
            "goal_focus_regex": self.goal_focus_regex,
            "goal_focus_instr": self.goal_focus_instr,
            "goal_focus_reasoning_division": self.goal_focus_reasoning_division,
            "goal_subgoals": [goal.to_json(include_only_goal_id, recurse_level=recurse_level+1) for goal in subgoals] if subgoals else None,
            "goal_status": self.goal_status.value,
            "goal_async_status": self.get_async_status().value if self.get_async_status() != GoalAsyncStatus.SYNC else None,
            "goal_error": self.goal_error if isinstance(self.goal_error, bool) else True if self.goal_error else False,
            "goal_error_message": self.goal_error_message,
            "goal_error_repeat_count": self.goal_error_repeat_count,
            "goal_completion_reason": self.goal_completion_reason,
            "goal_summary": summary,
            "goal_summary_pretty": self.goal_summary_pretty,
            "goal_agent_did_nothing": self.goal_agent_did_nothing,
            "goal_planner_outputs": self.get_goal_planner_outputs(),
            "goal_silent_summary": self.goal_silent_summary,
            "goal_summary_suppress": self.goal_summary_suppress,
            "goal_repeat_on_error": self.goal_repeat_on_error,
            "goal_validation": self.goal_validation.value,
            "goal_purpose": self.goal_purpose,
            "goal_images": self.goal_images,
            "goal_documents": self.goal_documents,
            "goal_override_role": self.goal_override_role,
            "goal_override_json_output": json.dumps(self.goal_override_json_output, ensure_ascii=False) if self.goal_override_json_output else None,
            "goal_override_structured_output": json.dumps(self.goal_override_structured_output) if self.goal_override_structured_output else None,
            "goal_override_recommended_plan": self.goal_override_recommended_plan,
            "goal_override_recommended_tools": self.goal_override_recommended_tools,
            "goal_proposed_followups": self.goal_proposed_followups,
            "goal_html_summary": self.goal_html_summary,
            "goal_blazor_component": self.goal_blazor_component,
            "goal_internal_thought": self.goal_internal_thought,
            "goal_instantiated_plan": instantiated_plan,
            "goal_rag_fragments": rag_fragments,
            "goal_datasets": self.get_datasets(),
            "goal_rag_docids": self.goal_rag_docids,
            "goal_rag_toomanyfragments": self.goal_rag_toomanyfragments,
            "goal_target": self.goal_target.value if self.goal_target else None,
            "goal_target_agent": self.goal_target_agent.to_json() if self.goal_target_agent else None,
            "goal_target_agent_name": self.goal_target_agent_name,
            "goal_attributes": self.goal_attributes,
            "goal_debug": goal_debug,
            "goal_last_update_datetime": self.goal_last_update_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "goal_perf_run_duration_ms": aggregated_perf['run_duration_ms'],
            "goal_perf_db_duration_ms": aggregated_perf['db_duration_ms'],
            "goal_perf_llm_duration_ms": aggregated_perf['llm_duration_ms'],
            "goal_perf_embeddings_duration_ms": aggregated_perf['embeddings_duration_ms'],
            "goal_perf_reranking_duration_ms": aggregated_perf['reranking_duration_ms'],
            "goal_perf_external_calls_duration_ms": aggregated_perf['external_calls_duration_ms'],
            "goal_is_partial": isPartial
        }
        planner_outputs = self.get_goal_planner_outputs()
        if planner_outputs is not None:
            goal_json["goal_planner_outputs"] = planner_outputs

        return goal_json

    def from_json(self, json_data: dict):
        self.goal_sequence_id = json_data.get("goal_id")[-1]
        self.goal_guid = json_data.get("goal_guid")
        if json_data.get("goal_dependent_upon") != "(-)":
            self.goal_dependent_upon = []
            dependent_upon = json_data.get("goal_dependent_upon").split(",")
            for dep in dependent_upon:
                dependency = int(dep.split(".")[-1]) - self.goal_sequence_id
                self.goal_dependent_upon.append(dependency)
        else:
            self.goal_dependent_upon = []
        self.goal_title = json_data.get("goal_title")
        self.goal_description = json_data.get("goal_description")
        self.goal_prompt = json_data.get("goal_prompt")
        # Ensure goal_recommended_tools is always a List[str] or None
        recommended_tools_data = json_data.get("goal_recommended_tools")
        if recommended_tools_data is not None:
            if isinstance(recommended_tools_data, str):
                self.goal_recommended_tools = [recommended_tools_data]
            elif isinstance(recommended_tools_data, list):
                self.goal_recommended_tools = recommended_tools_data
            else:
                self.goal_recommended_tools = None
        else:
            self.goal_recommended_tools = None
        self.goal_recommended_plan = json_data.get("goal_recommended_plan")
        self.goal_focus_docids = json_data.get("goal_focus_docids")
        self.goal_focus_iteration = json_data.get("goal_focus_iteration", "disabled")
        self.goal_focus_regex = json_data.get("goal_focus_regex")
        self.goal_focus_instr = json_data.get("goal_focus_instr")
        self.goal_focus_reasoning_division = json_data.get("goal_focus_reasoning_division")
        self.goal_status = GoalStatus(json_data.get("goal_status"))
        async_status_value = json_data.get("goal_async_status")
        self.set_async_status(GoalAsyncStatus(async_status_value) if async_status_value else GoalAsyncStatus.SYNC)
        self.goal_error = json_data.get("goal_error")
        self.goal_error_message = json_data.get("goal_error_message")
        self.goal_error_repeat_count = json_data.get("goal_error_repeat_count", 0)
        self.goal_completion_reason = json_data.get("goal_completion_reason")
        self.goal_summary = json_data.get("goal_summary")
        self.goal_summary_pretty = json_data.get("goal_summary_pretty")
        self.goal_agent_did_nothing = bool(json_data.get("goal_agent_did_nothing", False))
        planner_outputs = json_data.get("goal_planner_outputs")
        self.set_goal_planner_outputs(list(planner_outputs) if isinstance(planner_outputs, list) else None)
        self.goal_silent_summary = json_data.get("goal_silent_summary", False)
        self.goal_summary_suppress = json_data.get("goal_summary_suppress", False)
        self.goal_repeat_on_error = json_data.get("goal_repeat_on_error", False)
        self.goal_validation = GoalValidation(json_data.get("goal_validation", GoalValidation.NOT_VALIDATED.value))
        self.goal_purpose = json_data.get("goal_purpose")
        self.goal_images = json_data.get("goal_images", None)
        self.goal_documents = json_data.get("goal_documents", None)
        self.goal_override_role = json_data.get("goal_override_role")
        goal_override_json_output = json_data.get("goal_override_json_output")
        if goal_override_json_output:
            if isinstance(goal_override_json_output, bytes):
                goal_override_json_output = goal_override_json_output.decode("utf-8")
            self.goal_override_json_output = json.loads(goal_override_json_output)
        else:
            self.goal_override_json_output = None
        self.goal_override_structured_output = json.loads(json_data.get("goal_override_structured_output")) if json_data.get("goal_override_structured_output") else None
        self.goal_override_recommended_plan = json_data.get("goal_override_recommended_plan")
        # Ensure goal_override_recommended_tools is always a List[str] or None
        override_recommended_tools_data = json_data.get("goal_override_recommended_tools")
        if override_recommended_tools_data is not None:
            if isinstance(override_recommended_tools_data, str):
                self.goal_override_recommended_tools = [override_recommended_tools_data]
            elif isinstance(override_recommended_tools_data, list):
                self.goal_override_recommended_tools = override_recommended_tools_data
            else:
                self.goal_override_recommended_tools = None
        else:
            self.goal_override_recommended_tools = None
        self.goal_proposed_followups = json_data.get("goal_proposed_followups")
        self.goal_html_summary = json_data.get("goal_html_summary")
        self.goal_blazor_component = json_data.get("goal_blazor_component")
        self.goal_internal_thought = json_data.get("goal_internal_thought")
        # normalize goal_datasets on load
        datasets = json_data.get("goal_datasets", None)
        if datasets is not None:
            # set_datasets handles errors internally
            self.set_datasets(datasets)
        self.goal_rag_fragments = json_data.get("goal_rag_fragments")
        self.goal_rag_docids = json_data.get("goal_rag_docids")
        self.goal_rag_toomanyfragments = json_data.get("goal_rag_toomanyfragments")
        self.goal_target = GoalTarget(json_data.get("goal_target")) if json_data.get("goal_target") else None
        self.set_target_agent_name(json_data.get("goal_target_agent_name"))
        self.goal_attributes = json_data.get("goal_attributes")
        self.goal_debug = json.loads(json_data.get("goal_debug")) if json_data.get("goal_debug") else None
        self.goal_last_update_datetime = datetime.strptime(json_data.get("goal_last_update_datetime"), "%Y-%m-%d %H:%M:%S.%f")
        self.set_perf_run_ms(json_data.get("goal_perf_run_duration_ms", 0.0))
        self.add_perf_db_ms(json_data.get("goal_perf_db_duration_ms", 0.0))
        self.add_perf_llm_ms(json_data.get("goal_perf_llm_duration_ms", 0.0))
        self.add_perf_embeddings_ms(json_data.get("goal_perf_embeddings_duration_ms", 0.0))
        self.add_perf_reranking_ms(json_data.get("goal_perf_reranking_duration_ms", 0.0))
        self.add_perf_external_calls_ms(json_data.get("goal_perf_external_calls_duration_ms", 0.0))
        self.goal_is_partial = json_data.get("goal_is_partial")
        subgoals = json_data.get("goal_subgoals", [])
        if json_data.get("goal_instantiated_plan") is not None:
            instantiated_plan = json_data.get("goal_instantiated_plan")
            if isinstance(instantiated_plan, str):
                self.goal_instantiated_plan = instantiated_plan
            else:
                self.goal_instantiated_plan = json.loads(json_data.get("goal_instantiated_plan"))
        if subgoals is not None:
            self.goal_subgoals = []
            for goal_data in subgoals:
                subgoal = Goal(self.goal_plan, self, "", None, GoalTarget.UNDEFINED, None, [])
                subgoal.from_json(goal_data)
        else:
            self.goal_subgoals = None
        return self

    def get_status_color(self) -> str:
        status_color = get_status_color_for_status(self.goal_status)
        if self.goal_completion_reason and "<CANCELLED>" in self.goal_completion_reason:
            status_color = "\033[1;31m"  # dark red
        return status_color

    def get_active_goals(self) -> List['Goal']:
        active_goals: List['Goal'] = []

        if self.goal_status == GoalStatus.IN_PROGRESS:
            active_goals.append(self)

        if self.goal_subgoals is not None:
            for subgoal in self.goal_subgoals:
                active_goals.extend(subgoal.get_active_goals())

        return active_goals

    def cancel_llm_requests(self, timeout: float = 5.0) -> bool:
        cancelled: bool = False

        if self.goal_target_agent is not None:
            cancelled = self.goal_target_agent.cancel_llm_request(timeout=timeout)

        return cancelled

    def set_datasets(self, dataset_ids:Optional[Union[List[dict], List[str]]]) -> None:
        """Normalize and set `goal_datasets` to a list of dataset dicts.

        Accepts list of dicts or list of string ids. Uses plan.fetch_datasets for strings.
        Errors are handled internally: on failure `goal_datasets` is set to the raw input.
        The method no longer accepts or maps the legacy `datasetid` key — callers
        must provide canonical dicts with an `id` field or string ids.
        """
        try:
            if dataset_ids is None:
                self.goal_datasets = None
                return

            if isinstance(dataset_ids, list) and len(dataset_ids) == 0:
                self.goal_datasets = []
                return

            normalized: List[dict] = []
            # strings -> fetch
            if isinstance(dataset_ids, list) and isinstance(dataset_ids[0], str):
                try:
                    normalized = self.goal_plan.fetch_datasets(dataset_ids) if self.goal_plan is not None else [{"id": str(d)} for d in dataset_ids]
                    # fetch_datasets may return [] if the dataset isn't found in Neo4j yet; fall back to minimal id-only dicts
                    if not normalized:
                        normalized = [{"id": str(d)} for d in dataset_ids]
                except Exception:
                    normalized = [{"id": str(d)} for d in dataset_ids]
            else:
                for d in dataset_ids:
                    if isinstance(d, dict):
                        # Only accept canonical 'id' key; do not map legacy 'datasetid'
                        normalized.append(d)
                    else:
                        normalized.append({"id": str(d)})

            # deduplicate by id
            seen = set()
            deduped: List[dict] = []
            for ds in normalized:
                dsid = ds.get("id") if isinstance(ds, dict) else str(ds)
                if dsid is None:
                    key = json.dumps(ds, sort_keys=True) if isinstance(ds, dict) else str(ds)
                    if key not in seen:
                        deduped.append(ds)
                        seen.add(key)
                else:
                    if str(dsid) not in seen:
                        deduped.append(ds)
                        seen.add(str(dsid))

            self.goal_datasets = deduped
            # update plan datasets
            try:
                if self.goal_plan is not None:
                    # if original input were strings, pass them; else pass deduped dicts
                    self.goal_plan.add_datasets(dataset_ids if isinstance(dataset_ids, list) and isinstance(dataset_ids[0], str) else deduped)
            except Exception:
                # plan update is best-effort; swallow errors
                pass
        except Exception:
            # preserve raw input on failure to avoid losing data
            self.goal_datasets = dataset_ids

    def get_datasets(self, include_communities: bool = False) -> Optional[List[dict]]:
        """Return attached Datasets, excluding Community nodes by default."""
        if self.goal_datasets is None:
            return None
        if include_communities:
            return self.goal_datasets.copy()
        return [
            dataset
            for dataset in self.goal_datasets
            if not (
                dataset.get("is_community", False)
                or dataset.get("isCommunity", False)
                or "Community" in (dataset.get("labels") or [])
            )
        ]
    
    def get_dataset_ids(self) -> List[str]:
        """Return list of dataset IDs as strings."""
        if self.goal_datasets is None:
            return []
        return [ds.get("id") for ds in self.goal_datasets if ds.get("id") is not None]
    
    def get_dataset_names(self) -> List[str]:
        """Return list of dataset names as strings."""
        if self.goal_datasets is None:
            return []
        return [ds.get("name") for ds in self.goal_datasets if ds.get("name") is not None]
    
    def goal_id(self) -> int:
        if self.goal_parent == None:
            return 1
        else:
            try:
                return self.goal_parent.goal_subgoals.index(self) + 1
            except Exception as e:
                print(f"Error getting goal ID for {self.goal_title}: {e}")  # note: cannot use the log() because it may start infinite recursion
                return 0  # ignore

    def get_id_chain(self, truncate:int=0) -> List[int]:
        chain = []
        current = self
        while current:
            chain.append(current.goal_id())
            current = current.goal_parent
        chain.reverse()
        return chain[:- (-truncate)] if truncate else chain
    
    def get_id_chain_as_str(self, truncate:int=0) -> str:
        return ".".join(str(i) for i in self.get_id_chain(truncate))

    def get_previous_goal(self, delta_from_self:int=-1) -> 'Goal':
        if self.goal_plan is None:
            return None
        
        with self.goal_plan.plan_lock:
            if self.goal_parent == None:
                return None
            else:
                idx = self.goal_parent.goal_subgoals.index(self)
                if idx == 0:
                    return None
                else:
                    return self.goal_parent.goal_subgoals[idx + delta_from_self]

    def get_goal_by_id_chain(self, chain:List[int]) -> 'Goal':
        """
        Retrieve a goal by traversing a chain of goal IDs.

        This method searches for a goal by following a sequence of goal IDs provided in the `chain` list.
        The search starts from the current goal (`self`) and proceeds through its subgoals.

        Parameters:
        chain (List[int]): A list of goal IDs representing the path to the desired goal. The first element
                        in the list should match the ID of the current goal (`self`). Each subsequent
                        element should match the ID of a subgoal in the hierarchy.

        Returns:
        Goal: The goal object that matches the final ID in the chain, or None if no such goal is found.
        """
        if len(chain) == 0:
            return self
        if len(chain) == 1 and self.goal_id() == chain[0]:
            return self
        if self.goal_id() != chain[0]:
            return None
        
        child_id_to_find = chain[1]
        for goal in self.goal_subgoals:
            if goal.goal_id() == child_id_to_find:
                return goal.get_goal_by_id_chain(chain[1:])
        return None

    def get_sibling(self, target: GoalTarget, target_agent_name: str = None, position: Union[int, str] = 0) -> 'Goal':
        if self.goal_parent is None or not self.goal_parent.goal_subgoals:
            return None

        siblings = self.goal_parent.goal_subgoals
        # Filter siblings by target and agent name
        filtered = [
            sibling for sibling in siblings
            if sibling.goal_target == target and (target_agent_name is None or sibling.is_goal_target_agent_name(target_agent_name))
        ]

        if not filtered:
            return None

        if position == 'first':
            return filtered[0]
        elif position == 'last':
            return filtered[-1]
        elif isinstance(position, int):
            if 0 <= position < len(filtered):
                return filtered[position]
            else:
                return None
        else:
            return None

    def get_subgoal_by_target(self, target:GoalTarget, target_agent_name:str=None, fromIdx:int=0, direction:int=1) -> 'Goal':
        """
        Retrieve a subgoal by its target.

        This method searches for a subgoal within the current goal's subgoals that matches the specified target.

        Parameters:
        target (GoalTarget): The target of the subgoal to find.
        fromIdx (int): The starting index for the search (default: 0).
        direction (int): The direction of search. 1 for forward, -1 for backward (default: 1).

        Returns:
        Goal: The first subgoal that matches the specified target, or None if no such subgoal is found.
        """
        if self.goal_subgoals is None or len(self.goal_subgoals) == 0:
            return None
        
        if direction == 1:
            # Forward search
            for child in self.goal_subgoals[fromIdx:]:
                if child.goal_target is None:
                    continue
                if child.goal_target == target and (target_agent_name is None or child.is_goal_target_agent_name(target_agent_name)):
                    return child
        else:
            # Backward search
            for i in range(fromIdx, -1, -1):
                child = self.goal_subgoals[i]
                if child.goal_target is None:
                    continue
                if child.goal_target == target:
                    return child
        
        return None

    def get_next_goal(self, with_title:str=None, with_target:str=None, with_target_agent_name:str=None, not_before:'Goal'=None) -> 'Goal':
        if self.goal_subgoals == None or len(self.goal_subgoals) == 0:
            return None
        
        index_not_before = self.goal_subgoals.index(not_before) if not_before else -1

        for child in self.goal_subgoals:
            if not_before and self.goal_subgoals.index(child) <= index_not_before:
                continue

            if with_title and child.goal_title == with_title:
                return child
            if with_target and child.goal_target == with_target:
                return child
            if with_target_agent_name and child.is_goal_target_agent_name(with_target_agent_name):
                return child

        return None

    def get_last(self, statuses:List[GoalStatus]=None, agents:List[str]=None) -> 'Goal':
        if self.goal_subgoals == None or len(self.goal_subgoals) == 0:
            # no subgoals: return self if the status is in the list, otherwise return None
            if (statuses is None or self.goal_status in statuses) and (agents is None or self.goal_target_agent_name in agents):
                return self
            else:
                return None
        
        for child in reversed(self.goal_subgoals):
            if (statuses is None or child.goal_status in statuses) and (agents is None or child.goal_target_agent_name in agents):
                return child
            elif child.goal_status == GoalStatus.CHILDREN_IN_PROGRESS:
                found = child.get_last(statuses=statuses, agents=agents)
                if found:
                    return found

        # no subgoals with the specified status: return self if the status is in the list, otherwise return None
        if (statuses is None or self.goal_status in statuses) and (agents is None or self.goal_target_agent_name in agents):
            return self
        else:
            return None
    
    def get_last_by_target(self, targets:List[GoalTarget]) -> 'Goal':
        if self.goal_subgoals == None or len(self.goal_subgoals) == 0:
            # no subgoals: return self if the status is in the list, otherwise return None
            if targets is not None and self.goal_target in targets:
                return self
            else:
                return None
        
        for child in reversed(self.goal_subgoals):
            if targets is not None and child.goal_target in targets:
                return child
            elif child.goal_status == GoalStatus.CHILDREN_IN_PROGRESS:
                return child.get_last_by_target(targets)

        # no subgoals with the specified status: return self if the status is in the list, otherwise return None
        return None
        
    def get_last_recursive(self, statuses:List[GoalStatus]=None) -> 'Goal':
        if self.goal_status in statuses:
            return self
        if self.goal_subgoals == None or len(self.goal_subgoals) == 0:
            return None
        if self.goal_subgoals:
            for child in reversed(self.goal_subgoals):
                last = child.get_last_recursive(statuses=statuses)
                if last:
                    return last

    def get_last_user_goal(self) -> 'Goal':
        last_user_goal = self.goal_plan.plan_goal.get_last([GoalStatus.COMPLETED], ["AgentUser"]) if self.goal_plan is not None else None
        return last_user_goal

    def find_next_goal_with_agent(self, agent_name: str, status: GoalStatus, visited: set = None) -> 'Goal':
        """ Explores the plan recursively, starting at the current goal (self),
            traversing first the subgoals, then the next siblings and their subgoals,
            then the parent (starting from self) and its parent and so on.
            Will only return the first found goal with the requested status and goal_target_agent_name, or return None
        """
        try:
            # Initialize visited set if not provided
            if visited is None:
                visited = set()
            
            # Prevent infinite recursion by checking if we've already visited this goal
            if self.get_id_chain_as_str() in visited:
                return None
            visited.add(self.get_id_chain_as_str())

            # Check if current goal matches the criteria
            if self.goal_status == status and self.is_goal_target_agent_name(agent_name):
                return self
            
            # First, traverse subgoals recursively
            if self.goal_subgoals:
                for subgoal in self.goal_subgoals:
                    found_goal = subgoal.find_next_goal_with_agent(agent_name, status, visited)
                    if found_goal:
                        return found_goal
            
            # Then, traverse next siblings and their subgoals
            if self.goal_parent and self.goal_parent.goal_subgoals:
                current_index = self.goal_parent.goal_subgoals.index(self)
                # Check all siblings after the current goal
                for i in range(current_index + 1, len(self.goal_parent.goal_subgoals)):
                    sibling = self.goal_parent.goal_subgoals[i]
                    found_goal = sibling.find_next_goal_with_agent(agent_name, status, visited)
                    if found_goal:
                        return found_goal
            
            # Finally, traverse up to parent and continue the search from there
            # But skip the parent's own check since we're coming from it
            if self.goal_parent and self.goal_parent not in visited:
                return self.goal_parent._find_next_goal_with_agent_from_parent(agent_name, status, visited, self)
            
            # No matching goal found
            return None
        
        except Exception as e:
            log(ERROR, f"Error occurred while finding next goal with agent: {e}\n{traceback.format_exc()}")
            return None
    
    def _find_next_goal_with_agent_from_parent(self, agent_name: str, status: GoalStatus, visited: set, coming_from: 'Goal') -> 'Goal':
        """ Helper method for parent traversal to avoid infinite recursion """
        try:
            # Mark this goal as visited
            if self.get_id_chain_as_str() in visited:
                return None
            visited.add(self.get_id_chain_as_str())
            
            # Don't check the current goal since we're traversing upward
            # Go directly to next siblings and parent traversal
            
            # Check next siblings after the goal we came from
            if self.goal_parent and self.goal_parent.goal_subgoals:
                current_index = self.goal_parent.goal_subgoals.index(self)
                for i in range(current_index + 1, len(self.goal_parent.goal_subgoals)):
                    sibling = self.goal_parent.goal_subgoals[i]
                    found_goal = sibling.find_next_goal_with_agent(agent_name, status, visited)
                    if found_goal:
                        return found_goal
            
            # Continue traversing up to parent
            if self.goal_parent and self.goal_parent not in visited:
                return self.goal_parent._find_next_goal_with_agent_from_parent(agent_name, status, visited, self)
            
            return None
        
        except Exception as e:
            log(ERROR, f"Error in parent traversal for finding next goal with agent: {e}\n{traceback.format_exc()}")
            return None

    def get_summary(self) -> Optional[str]:
        if self.goal_summary_suppress == True:
            log(DEBUG, f"Goal {self.get_id_chain_as_str()} (Agent: {self.goal_target_agent_name}) summary is suppressed")
            return None
        
        summary = None
        target_agent_overrules_silent_summary = (self.goal_target_agent is not None and self.goal_target_agent.agent_overrule_silent_summary == True)
        if not self.goal_silent_summary or target_agent_overrules_silent_summary:
            if self.goal_summary_pretty is not None and len(self.goal_summary_pretty) > 0:
                summary = self.goal_summary_pretty
            else:
                summary = self.goal_summary
        elif self.goal_internal_thought is not None and len(self.goal_internal_thought) > 0 and self.goal_internal_thought.lower() != "null":
            log(DEBUG, f"Goal {self.get_id_chain_as_str()} (Agent: {self.goal_target_agent_name}) summary is silenced")
            silent_summary_text = "The following message can be seen by LLM Agents but not by the user; remember, the user doesn't see this message. If the information needs to be shared with the user, it needs to be repeated (in appropriate format and language) later on by AgentAssistant or AgentCustom"
            summary = f"{silent_summary_text}: `{self.goal_internal_thought}`"

        if summary is not None and not isinstance(summary, str):
            if isinstance(summary, list) or isinstance(summary, dict):
                summary = json.dumps(summary, indent=4)
            else:
                summary = str(summary)

        if summary is not None and len(summary) > 0 and self.goal_error_repeat_count > 0:
            summary = f"[THIS MESSAGE HAS BEEN REPEATED {self.goal_error_repeat_count} TIMES DUE TO ERROR. Ignore the previous messages and consider the following]\n{summary}"

        return summary

    def is_validated(self) -> GoalValidation:
        return self.goal_validation

    def set_validated(self, validation: GoalValidation = GoalValidation.VALIDATED_OK) -> None:
        self.goal_validation = validation

    def set_summary(self, 
                    summary:str,
                    pretty_summary:Optional[str]=None,
                    status:GoalStatus=None, 
                    completion_reason:str=None, 
                    override_role:str=None, 
                    override_json_output:str=None, 
                    override_structured_output:str=None,
                    recommended_plan:str=None,
                    recommended_tools:List[str]=None) -> None:
        self.goal_summary = summary
        self.goal_summary_pretty = pretty_summary
        if status:
            self.goal_status = status
        if completion_reason:
            self.goal_completion_reason = completion_reason
        if override_role:
            self.goal_override_role = override_role
        if override_json_output:
            self.goal_override_json_output = override_json_output
        if override_structured_output:
            self.goal_override_structured_output = override_structured_output
        if recommended_plan:
            self.goal_override_recommended_plan = recommended_plan
        if recommended_tools:
            self.goal_override_recommended_tools = recommended_tools

    def set_override_json_output(self, override_json_output:Union[List, Dict, None], override_structured_output:Union[Dict, None]) -> None:
        self.goal_override_json_output = override_json_output
        self.goal_override_structured_output = override_structured_output

    def set_recommended_tools(self, tools):
        """
        Set goal_recommended_tools ensuring it's always a List[str] or None.
        
        Args:
            tools: Can be None, str, or List[str]
        """
        if tools is None:
            self.goal_recommended_tools = None
        elif isinstance(tools, str):
            self.goal_recommended_tools = [tools]
        elif isinstance(tools, list):
            # Ensure all items in the list are strings
            if all(isinstance(item, str) for item in tools):
                self.goal_recommended_tools = tools
            else:
                # Convert non-string items to strings or filter them out
                self.goal_recommended_tools = [str(item) for item in tools if item is not None]
        else:
            # For any other type, convert to None
            self.goal_recommended_tools = None

    def get_last_checkpoint(self) -> 'Goal':
        return self.goal_plan.plan_goal.get_last_by_target([GoalTarget.NOOP])

    def set_images(self, images:List[str]):
        self.goal_images = images
    
    def set_documents(self, documents:List[str]):
        self.goal_documents = documents

    def get_async_status(self) -> GoalAsyncStatus:
        return self.goal_async_status

    def set_async_status(self, status: GoalAsyncStatus) -> None:
        self.goal_async_status = status

    def get_last_summary(self, agent_name:str=None, visited:set=None) -> Tuple[Optional[str], Optional['Goal']]:
        if visited is None:
            visited = set()
        if self in visited:
            return None, None
        visited.add(self)
        
        if self.goal_parent is None:
            summary = self.goal_summary if self.goal_summary_pretty is None else self.goal_summary_pretty
            return (summary, self) if summary is not None else (None, None)
        
        # identify the last completed goal in the current chain, going backwards from self
        current_chain = self.goal_parent.goal_subgoals
        idx = current_chain.index(self)
        for i in range(idx, -1, -1):
            current_goal = current_chain[i]
            if current_goal.goal_target == GoalTarget.ONLY_SUBGOALS and current_goal.goal_status == GoalStatus.COMPLETED and current_goal.goal_subgoals is not None and len(current_goal.goal_subgoals) > 0:
                subgoals_last_summary, node = current_goal.goal_subgoals[-1].get_last_summary(agent_name=agent_name, visited=visited)
                if subgoals_last_summary is not None:
                    return subgoals_last_summary, node
            if current_goal.goal_status == GoalStatus.COMPLETED:
                if agent_name is None or (current_goal.goal_target_agent is not None and current_goal.goal_target_agent_name == agent_name):
                    if current_goal.goal_summary is not None:
                        summary = current_goal.goal_summary if current_goal.goal_summary_pretty is None else current_goal.goal_summary_pretty
                        return summary, current_goal
        
        last_summary, last_goal = self.goal_parent.get_last_summary(visited=visited)
        return (last_summary, last_goal) if last_summary is not None else (None, None)
    
    def get_last_summaries_or_rag(self) -> List['Goal']:
        # in the current goal chain level (e.g. if we are in 1.3.5 then all the goals that start with 1.3),
        # retrieve from 'self' (1.3.5) in reverse order (from 1.3.5 to 1.3.1) the last summaries
        if self.goal_parent is None:
            return []
        
        summaries = []
        current_chain = self.goal_parent.goal_subgoals
        idx = current_chain.index(self)
        for i in range(idx, -1, -1):
            current_goal = current_chain[i]
            if current_goal.goal_summary is not None and len(current_goal.goal_summary) > 0:
                summaries.append(current_goal)
            elif current_goal.goal_rag_fragments is not None and len(current_goal.goal_rag_fragments) > 0:
                summaries.append(current_goal)
        return summaries

    def set_goal_status(self, status:GoalStatus, recursive:bool=False, completion_reason:str=None) -> None:
        """
        Set the status of the goal, and updates the last update datetime of the goal and its parent chain.
        If there are subgoals, their statuses is updated as well (unless they are already completed or failed).
        """
        status_updated: bool = False
        previous_status: GoalStatus = self.goal_status
        last_update_datetime = self.goal_last_update_datetime
        now = datetime.now()
        while last_update_datetime == now:
            time.sleep(0.001)
            now = datetime.now()

        if not recursive or (self.goal_status != GoalStatus.FAILED and self.goal_status != GoalStatus.COMPLETED and self.goal_status != GoalStatus.RUN_EVERY_TIME):
            self.goal_status = status
            self.set_last_update_datetime(now)
            status_updated = True

        if completion_reason is not None:
            self.goal_completion_reason = completion_reason

        if status_updated and status == GoalStatus.CANCELLED and previous_status != GoalStatus.CANCELLED:
            cancellation_reason = completion_reason or "No cancellation reason was provided."
            log(WARNING, f"Goal {self.get_id_chain_as_str()} '{self.goal_title}' was cancelled at the user's request. Reason: {cancellation_reason}")

        if status_updated and status == GoalStatus.CANCELLED:
            self.cancel_dependent_goals()

        if recursive and self.goal_subgoals is not None:
            for subgoal in self.goal_subgoals:
                subgoal.set_goal_status(status, recursive=recursive, completion_reason=completion_reason)

    def reset_for_rollback(self) -> None:
        """
        Discard this goal's produced work (subgoals and every derived/output field) and bring it back
        to NOT_STARTED, as if it had never run. Generic building block reused by Agent rollback().
        Input/planning fields (goal_prompt, goal_recommended_*, goal_override_*, goal_focus_*) are left untouched,
        since they describe what the goal is meant to do, not what it produced.
        """
        self.goal_subgoals = []
        self.goal_status = GoalStatus.NOT_STARTED
        self.goal_silent_summary = False
        if self.goal_async_status in (GoalAsyncStatus.ASYNC_IN_PROGRESS, GoalAsyncStatus.ASYNC_COMPLETE, GoalAsyncStatus.ASYNC_FAILED, GoalAsyncStatus.ASYNC_ABORTED):
            # the goal was created as async: run() must spawn its background thread again, instead of running it synchronously
            self.goal_async_status = GoalAsyncStatus.ASYNC_REQUESTED
            self.goal_async_thread = None
        self._clear_outputs_for_rollback()

    def reset_ancestor_for_rollback(self) -> None:
        """
        Reopen a container goal whose descendants were rolled back: it keeps its children (static structure and
        siblings that were not rolled back) but loses the outputs derived from them, and becomes IN_PROGRESS so
        that the scheduler enters it again and runs the reset children.
        """
        self.goal_status = GoalStatus.IN_PROGRESS
        self._clear_outputs_for_rollback()

    def _clear_outputs_for_rollback(self) -> None:
        self.goal_error = None
        self.goal_error_message = None
        self.goal_error_repeat_count = 0
        self.goal_summary = None
        self.goal_summary_pretty = None
        self.goal_agent_did_nothing = False
        self.goal_planner_outputs = None
        self.goal_summary_suppress = False
        self.goal_html_summary = None
        self.goal_tool_call_message = None
        self.goal_proposed_followups = None
        self.goal_blazor_component = None
        self.goal_internal_thought = None
        self.goal_validation = GoalValidation.NOT_VALIDATED
        self.goal_images = None
        self.goal_documents = None
        self.goal_datasets = None
        self.goal_rag_fragments = None
        self.goal_rag_docids = None
        self.goal_rag_toomanyfragments = None
        with self.goal_perf_lock:
            self.goal_perf_run_duration_ms = 0.0
            self.goal_perf_db_duration_ms = 0.0
            self.goal_perf_llm_duration_ms = 0.0
            self.goal_perf_embeddings_duration_ms = 0.0
            self.goal_perf_reranking_duration_ms = 0.0
            self.goal_perf_external_calls_duration_ms = 0.0
        self.goal_instantiated_plan = None
        self.goal_completion_reason = None
        self.goal_attributes = None
        self.goal_debug = None
        self.set_last_update_datetime(datetime.now())

    def get_transitive_forward_dependents(self) -> List['Goal']:
        """
        Return every later sibling goal that directly or transitively depends on self, in order.
        Unlike cancel_dependent_goals(), this does not skip goals that already reached a terminal status
        (COMPLETED/FAILED/CANCELLED) -- it is used to find goals whose work must be reset after a rollback,
        regardless of their current status.
        """
        dependents: List['Goal'] = []

        if self.goal_parent is None or self.goal_parent.goal_subgoals is None:
            return dependents

        siblings = self.goal_parent.goal_subgoals
        self_index = self.goal_id() - 1  # 0-based index, matches get_dependencies_status()'s resolution
        dependent_indices = {self_index}

        for sibling in siblings[self_index + 1:]:
            if not sibling.goal_dependent_upon:
                continue
            sibling_index = sibling.goal_id() - 1
            sibling_depends_on_known = any(
                (dependency + sibling_index) in dependent_indices
                for dependency in sibling.goal_dependent_upon
            )
            if sibling_depends_on_known:
                dependents.append(sibling)
                dependent_indices.add(sibling_index)

        return dependents

    def get_rollback_dependents(self) -> List['Goal']:
        """
        Goals that must be reset, besides self, when self is rolled back: the transitive forward dependents of self,
        plus those of each ancestor (below the root), e.g. an async Verifier that depends on the enclosing container.
        They are reset wholesale (their subtree is discarded), which suits leaf agents and the Consolidator.
        """
        dependents: List['Goal'] = []
        node: Optional['Goal'] = self

        while node is not None:
            dependents.extend(node.get_transitive_forward_dependents())
            node = node.goal_parent

        return dependents

    def get_rollback_ancestors(self) -> List['Goal']:
        """Ancestors of self, from the closest one up to (and excluding) the plan's root goal."""
        ancestors: List['Goal'] = []
        node: Optional['Goal'] = self.goal_parent

        while node is not None and node.goal_parent is not None:
            ancestors.append(node)
            node = node.goal_parent

        return ancestors

    def get_subtree_goals(self) -> List['Goal']:
        goals: List['Goal'] = [self]

        for subgoal in (self.goal_subgoals or []):
            goals.extend(subgoal.get_subtree_goals())

        return goals

    def cancel_async_execution(self, timeout: float = 10.0) -> bool:
        """
        Cancel an in-flight async execution (the agent's de2generator LLM request) and wait for its thread to end.
        Returns False when the thread is still running after the timeout.
        """
        stopped: bool = True

        if self.get_async_status() == GoalAsyncStatus.ASYNC_IN_PROGRESS:
            self.cancel_llm_requests()
            thread: Optional[threading.Thread] = self.goal_async_thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout)
                stopped = not thread.is_alive()

        return stopped

    def has_been_engaged(self) -> bool:
        """True if this goal or any of its descendants has ever left the NOT_STARTED state."""
        if self.goal_status != GoalStatus.NOT_STARTED:
            return True

        return any(subgoal.has_been_engaged() for subgoal in (self.goal_subgoals or []))

    def get_latest_goal_for_agent_type(self, agent_type: type) -> Optional['Goal']:
        """
        Find the goal of the given agent type belonging to the most recently engaged top-level
        fragment in self.goal_subgoals, skipping any trailing fragment(s) that never started --
        ChatBot2's add_plan_chat() always keeps one such fragment ready for the next user turn,
        so the plain last element is not necessarily the latest actual interaction.
        Intended to be called on the plan's root goal to resolve "the latest trainable goal of this agent type",
        used for identity (goal_guid) based rollback authorization -- never by scanning the whole plan history.
        """
        for fragment in reversed(self.goal_subgoals or []):
            if not fragment.has_been_engaged():
                continue
            found = fragment._find_goal_for_agent_type(agent_type)
            if found is not None:
                return found

        return None

    def _find_goal_for_agent_type(self, agent_type: type) -> Optional['Goal']:
        if isinstance(self.goal_target_agent, agent_type):
            return self

        if self.goal_subgoals:
            for subgoal in self.goal_subgoals:
                found = subgoal._find_goal_for_agent_type(agent_type)
                if found is not None:
                    return found


        return None

    def cancel_dependent_goals(self) -> None:
        if self.goal_parent is None or self.goal_parent.goal_subgoals is None:
            return

        sibling_goals = self.goal_parent.goal_subgoals
        current_index = sibling_goals.index(self)

        for sibling in sibling_goals[current_index + 1:]:
            if sibling.goal_dependent_upon is None:
                continue

            depends_on_self = any(
                dependency + sibling.goal_id() - 1 == self.goal_id() - 1  # 0-based index, matches get_dependencies_status()
                for dependency in sibling.goal_dependent_upon
            )
            if depends_on_self and sibling.goal_status == GoalStatus.NOT_STARTED:
                sibling.set_goal_status(GoalStatus.CANCELLED, completion_reason="Dependency cancelled")

        # if every sibling has now reached a terminal state and at least one of them was cancelled,
        # the parent can never complete its intended work as planned: cancel it too, so that whatever
        # depends on the parent at the next level up is cancelled in turn (recursively, via this same method)
        parent = self.goal_parent
        if parent.goal_status not in [GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.CANCELLED]:
            if parent.is_children_completed():
                any_child_cancelled = any(child.goal_status == GoalStatus.CANCELLED for child in (parent.goal_subgoals or []))
                if any_child_cancelled:
                    parent.set_goal_status(GoalStatus.CANCELLED, completion_reason="Child goal cancelled")

    def get_target_agent_name(self) -> str:
        return self.goal_target_agent_name

    def set_target_agent_name(self, name:str) -> None:
        # this method accepts a new name in these formats:
        #  - None
        #  - 'AgentClassName' (for example: 'AgentToolCaller')
        #  - 'src.core.agents.AgentClassName' (for example: 'src.core.agents.AgentToolCaller')
        # in all cases, only the class name (e.g. 'AgentToolCaller') is stored in the attribute goal_target_agent_name
        if name is None:
            self.goal_target_agent_name = None
        else:
            self.goal_target_agent_name = name.split('.')[-1]

    def is_goal_target_agent_name(self, name:str) -> bool:
        # this method checks if the given name matches the goal's target agent name
        # for the comparison with the input name, same rules as for set_goal_agent_target_name apply
        if name is None:
            return self.goal_target_agent_name is None
        else:
            return self.goal_target_agent_name == name.split('.')[-1]

    def set_last_update_datetime(self, datetime:datetime) -> None:
        self.goal_last_update_datetime = datetime
        if self.goal_parent is not None:
            self.goal_parent.set_last_update_datetime(datetime)

    def is_children_completed(self) -> bool:
        if self.goal_subgoals == None or len(self.goal_subgoals) == 0:
            return True
        
        for child in self.goal_subgoals:
            if child.goal_status not in [GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.CANCELLED]:
                return False
        return True
    
    def shift_dependencies(self, new_goals: List['Goal'], after_id:List[int]) -> None:
        # this method is called from within insert_goals() when inserting new goals in the middle of a level
        # Example:
        #   1.1.1 goal [no dependencies]            (completed)
        #   1.1.2 goal [depends on 1.1.1]           (completed)
        #   1.1.3 goal [depends on 1.1.1, 1.1.2]    (completed)
        #   1.1.4 goal [depends on 1.1.3]           (in progress, calling insert_goals())
        #   1.1.5 goal [depends on 1.1.3, 1.1.4]    (not started)
        #   1.1.6 goal [depends on 1.1.5]           (not started)
        #
        # we need to insert two new goals after 1.1.4:
        #   1.1.4 first new goal [no intrinsic dependencies]
        #   1.1.5 second new goal [depends on first new goal, which is in position -1 respect to itself]
        #
        # What we want as result is:
        #   1.1.1 goal [no dependencies]            (completed)
        #   1.1.2 goal [depends on 1.1.1]           (completed)
        #   1.1.3 goal [depends on 1.1.1, 1.1.2]    (completed)
        #   1.1.4 goal [depends on 1.1.3]           (in progress, calling insert_goals())
        #   1.1.5 first new goal [depends on 1.1.3] (not started)             --> dependency injection to the current goal (self)
        #   1.1.6 second new goal [depends on 1.1.3, 1.1.4]    (not started)  --> dependency injection to the current goal (self), plus the intrinsic dependencies
        #   1.1.7 goal [depends on 1.1.3, 1.1.4, 1.1.5, 1.1.6] (not started)  --> dependency on 1.1.3, 1.1.4 are maintained, plus new dependencies are injected
        #   1.1.8 goal [depends on 1.1.7]           (not started)             --> dependency on 1.1.5 is shifted to 1.1.7 (it is maintained, indexes are shifted)
        #
        # Now, let's rewrite it with the correct negative values of ths dependencies:
        # Start:
        #   1.1.1 goal [no dependencies]            (completed)
        #   1.1.2 goal [depends on -1]              (completed)
        #   1.1.3 goal [depends on -2, -1]          (completed)
        #   1.1.4 goal [depends on -1]              (in progress, calling insert_goals())
        #   1.1.5 goal [depends on -2, -1]          (not started)
        #   1.1.6 goal [depends on -1]              (not started)
        #
        # After inserting:
        #   1.1.1 goal [no dependencies]            (completed)                             --> unchanged
        #   1.1.2 goal [depends on -1]              (completed)                             --> unchanged
        #   1.1.3 goal [depends on -2, -1]          (completed)                             --> unchanged
        #   1.1.4 goal [depends on -1]              (in progress, calling insert_goals())   --> unchanged
        #   1.1.5 first new goal [depends on -1]    (not started)             --> dependency injection to the current goal (self)
        #   1.1.6 second new goal [depends on -2, -1] (not started)           --> dependency injection to the current goal (self) at -2, plus the intrinsic dependencies which are now -1
        #   1.1.7 goal [depends on -4, -3, -2, -1]  (not started)             --> dependency on 1.1.3, 1.1.4 are maintained + new dependencies are injected
        #   1.1.8 goal [depends on -1]              (not started)             --> unchanged
        #
        # In summary:
        #  a)  up to and inluding the current goal (self = point of insertion), the dependencies are left unchanged
        #  b)  the newly added goals must be dependent on the point of insertion (self), plus their own dependencies
        #  c)  any subsequent goal dependent on the point of insertion (self) will have:
        #  c1) existing dependencies to the goals from beginning up to the point of insertion (self) are shifted by -len(new_goals).
        #      For example 1.1.5 was originally dependent on [-2,-1] and is now 1.1.7 dependent on [-4,-3, ...]. Note that the new -4,-3 and the old -2,-1 are the same goals (e.g. 1.1.3, 1.1.4)
        #  c2) new dependencies are injected to the newly inserted goals, IF and only IF the goal was dependent on the point of insertion (self)
        #      For example 1.1.5 was originally dependent on [-2,-1] and is now 1.1.7 dependent on [..., -2, -1], where -2, -1 are the new goals (e.g. 1.1.5, 1.1.6)
        #      but 1.1.6 was originally dependent on [-1] and no dependency is injected to it, so it remains [-1]
        #
        # Algorithm:
        # 1. identify the goal chain, the current goal index (self) = the point of insertion after which the new goals are inserted
        # 2. for each goal after the point of insertion (self):
        #    b1) if the goal is dependent on goals in the range [1..point of insertion (self)] then shift those dependencies by -len(new_goals)
        #    b2) if the goal is dependent on the point of insertion (self), then inject the new dependencies
        #
        # REFERENCE TO UNIT TEST AT THE END OF THIS FILE for a live demonstration of this algorithm with this example data
        try:
            goal_chain = self.goal_parent.goal_subgoals if self.goal_parent else None
            if goal_chain is None:
                return

            if app_config.testing:
                log_plan(self.goal_plan)

            # b) add dependencies of the newly inserted goals to the point of insertion (self)
            for new_goal_idx in range(0, len(new_goals)):
                new_goal = new_goals[new_goal_idx]
                if new_goal.goal_dependent_upon is None:
                    new_goal.goal_dependent_upon = []
                additional_dependency = -(new_goal_idx+1)
                if additional_dependency not in new_goal.goal_dependent_upon:
                    new_goal.goal_dependent_upon.append(additional_dependency)          # the first is appended as -1, the second as -2, etc.

            if app_config.testing:
                log_plan(self.goal_plan)

            for idx in range(self.goal_id()+len(new_goals), len(goal_chain)):           # in the example, self.goal_id==4, there are 2 new goals, so the loop
                                                                                        # iterates between idx 6<=idx<8, which is idx=6 and 7, corresponding to goals 1.1.7, 1.1.8
                goal = goal_chain[idx]
                p = idx - len(new_goals) - self.goal_id() + 1                           # for 1.1.7 p = (6 - 2 - 4) = 1, for 1.1.8 p = (7 - 2 - 4) = 1

                if goal.goal_dependent_upon is None:
                    goal.goal_dependent_upon = []

                was_dependent_upon = -(idx - len(new_goals))
                dependencies_copy = goal.goal_dependent_upon.copy()
                for dependency in dependencies_copy:
                    if dependency <= was_dependent_upon:                                # in this example, goal 1.1.7 generates -(idx-self.goal_id) = -(6-4) = -2.
                                                                                        # c1) shift dependencies of the goals after the point of insertion (self)
                                                                                        # dependency is -2 (1.1.3) and -1 (1.1.4), so the condition is true for both dependencies.
                                                                                        # goal 1.1.7 has dependencies [-2,-1], both of which were originally to
                                                                                        # 1.1.3 and 1.1.4, both of which are now shifted to -4 and -3 respectively. In other words, 
                                                                                        # the dependencies must be shifted by -len(new_goals) = -2
                        shifted_dependency = dependency - len(new_goals)
                        if shifted_dependency < -idx:
                            log(WARNING, f"While shifting dependencies, a shifted dependency {shifted_dependency} less than the current maximum {-idx} was found. This should never happen.")
                        else:
                            goal.goal_dependent_upon[goal.goal_dependent_upon.index(dependency)] = shifted_dependency

                    if dependency == was_dependent_upon:                                # in this example, goal 1.1.7 generates -(idx-self.goal_id()) = -(6-4) = -2.
                        # c2) inject dependencies to the newly inserted goals if the goal was dependent on the point of insertion (self)
                                                                                        # in this example, goal 1.1.7 must be additionally dependent on the new goals 1.1.5 and 1.1.6,
                                                                                        # which are at -2, -1 respect to it. In general, -N-p..-1-p where N is the number of new goals
                                                                                        # and p is the original distance (before insertion) of the goal from the point of insertion (self)
                        additional_dependencies = range(-p, -p-len(new_goals), -1)      # for 1.1.7 this generates -1, -2, which is correctly pointing to the new goals 1.1.5 and 1.1.6
                        for additional_dependency in additional_dependencies:
                            if additional_dependency not in goal.goal_dependent_upon:
                                if additional_dependency < -idx:
                                    log(WARNING, f"While shifting dependencies, an additional dependency {additional_dependency} less than the current maximum {-idx} was found. This should never happen.")
                                else:
                                    goal.goal_dependent_upon.append(additional_dependency)
        
        except Exception as e:
            log(ERROR, f"Error inserting goals: {e}\n{traceback.format_exc()}")

        if app_config.testing:
            log_plan(self.goal_plan)

        return

    def remove_dependencies(self, goals_to_remove: List['Goal']) -> None:
        """
        Remove the indicated sibling goals from the dependencies of 'self'.
        
        Example:
         - self is 1.2.3.4, dependent on 1.2.3.1 and 1.2.3.3 (that is: goal_dependent_upon = [-3, -1])
         - goals_to_remove contains 1.2.3.1 (that is: the -3 must be removed from self.goal_dependent_upon)
        """
        if self.goal_parent is None or not self.goal_dependent_upon or not goals_to_remove:
            return

        self_idx = self.goal_id()  # 1-based position of self in parent

        for goal_to_remove in goals_to_remove:
            if goal_to_remove is None or goal_to_remove.goal_parent is not self.goal_parent:
                continue

            goal_to_remove_idx = goal_to_remove.goal_id()  # 1-based position of the goal to remove in parent
            if goal_to_remove_idx >= self_idx:
                continue

            delta_to_remove = goal_to_remove_idx - self_idx
            while delta_to_remove in self.goal_dependent_upon:
                self.goal_dependent_upon.remove(delta_to_remove)
                log(DEBUG, f"remove_dependencies: removed dependency {delta_to_remove} from goal '{self.get_id_chain_as_str()}' (was pointing to '{goal_to_remove.get_id_chain_as_str()}')")

    def remove_dependencies_to(self) -> None:
        """Remove from all subsequent sibling goals any dependency delta that resolves to self.

        After inserting self (e.g. a container goal) into the parent's subgoal list,
        shift_dependencies() may have added delta entries in subsequent siblings that point
        back to self so that they wait for it to complete.  Call this method on self to strip
        those injected entries, allowing those siblings to proceed in parallel with self.

        Delta resolution (same as get_dependencies_status):
            sibling at 1-based position j, delta d  →  points to 1-based position (d + j).
        Therefore the delta that points to self at position k from a sibling at position j is:
            d = k - j   (always negative since j > k).
        """
        if self.goal_parent is None:
            return

        siblings = self.goal_parent.goal_subgoals
        self_idx = self.goal_id()  # 1-based position of self in parent

        # Iterate over all siblings that come *after* self (0-based index >= k means 1-based >= k+1)
        for subsequent_siblings_idx in range(self_idx, len(siblings)):
            sibling = siblings[subsequent_siblings_idx]
            subsequent_siblings_idx_1 = subsequent_siblings_idx + 1  # 1-based position of this sibling
            delta_to_self = self_idx - subsequent_siblings_idx_1  # the negative delta that resolves to self
            if sibling.goal_dependent_upon and delta_to_self in sibling.goal_dependent_upon:
                sibling.goal_dependent_upon.remove(delta_to_self)
                log(DEBUG, f"remove_dependencies_to: removed dependency {delta_to_self} from goal '{sibling.get_id_chain_as_str()}' (was pointing to '{self.get_id_chain_as_str()}')")

    def insert_goals(self, new_goals:List['Goal'], after_id:Union[str, List[int]], instantiated_plan:Union[list,dict,str]) -> bool:
        was_added = False

        # add self as parent to new goals
        for goal in new_goals:
            goal.goal_parent = self
            goal.goal_instantiated_plan = instantiated_plan
            if self.goal_plan:
                goal.goal_plan = self.goal_plan
        
        # insert new goals
        if after_id is None:
            # replace all subgoals
            self.goal_subgoals = new_goals
            was_added = True
        elif isinstance(after_id, str) and after_id == "first":
            # insert new goals at the beginning of current subgoals
            self.goal_subgoals = new_goals + self.goal_subgoals
            was_added = True
        elif isinstance(after_id, str) and after_id == "last":
            # append new goals at the end of current subgoals
            if self.goal_subgoals == None:
                self.goal_subgoals = new_goals
            else:
                self.goal_subgoals += new_goals
            was_added = True
        else:
            # insert new goals after the specified goal
            pass
            for idx, goal in enumerate(self.goal_subgoals):
                goal_id = goal.get_id_chain()
                if goal_id == after_id:
                    # insert new goals after the specified goal
                    self.goal_subgoals = self.goal_subgoals[:idx+1] + new_goals + self.goal_subgoals[idx+1:]
                    point_of_insertion = self.goal_subgoals[idx]
                    point_of_insertion.shift_dependencies(new_goals, after_id)
                    was_added = True
                    break
                else:
                    # recurse down the chain, if the subgoal has subgoals
                    goal_id_len = len(goal_id)
                    after_id_truncated = after_id[:goal_id_len]
                    if goal_id == after_id_truncated:
                        was_added = goal.insert_goals(new_goals, after_id, instantiated_plan)
                        break
                        
            # update statuses up the parent chain
            node = self
            while node is not None:
                if node.goal_status == GoalStatus.COMPLETED:
                    node.goal_status = GoalStatus.IN_PROGRESS
                node = node.goal_parent

        return was_added

    def _insert_current_goal_and_dependencies_in_case_of_error(self) -> None:
        # identify the previous goals on the same level, that this goal is dependent upon
        # and insert them after the current goal
        if app_config.testing:
            log_plan(self.goal_plan)
        log(INFO, "Inserting current goal and dependencies in case of error")

        dependent_goals = []
        if self.goal_dependent_upon is not None and len(self.goal_dependent_upon) > 0:
            for dep in self.goal_dependent_upon:
                # find the goal by looking at the parent subgoals, index of this goal plus the negative integer in the dependent_upon list
                dependent_index = dep + self.goal_id() - 1
                if dependent_index >= 0:
                    dependent_sibling = self.goal_parent.goal_subgoals[dependent_index]
                    if dependent_sibling not in dependent_goals:
                        # create a clone of the dependent sibling goal, set its status to NOT_STARTED and append it to the list of dependent goals
                        cloned_sibling = dependent_sibling.clone()
                        cloned_sibling.goal_status = GoalStatus.NOT_STARTED
                        cloned_sibling.goal_error = None
                        cloned_sibling.goal_error_message = None
                        cloned_sibling.goal_error_repeat_count += 1
                        if cloned_sibling.goal_error_repeat_count >= MAX_ERROR_REPEAT_COUNT:
                            log(WARNING, f"***WARNING: Goal '{cloned_sibling.goal_title}' has been repeated {MAX_ERROR_REPEAT_COUNT} times. It will be marked as failed.")
                            return  # hard-cut to avoid infinite loop
                        else:
                            dependent_goals.append(cloned_sibling)

                        # make sure that the original goal's summary disappears from build_messages()
                        dependent_sibling.goal_summary_suppress = True
                        cloned_sibling.goal_summary_suppress = False

        # add self to the list of dependent goals
        cloned_self = self.clone()
        cloned_self.goal_status = GoalStatus.NOT_STARTED
        cloned_self.goal_error = None
        cloned_self.goal_error_message = None
        cloned_self.goal_error_repeat_count += 1
        if cloned_self.goal_error_repeat_count >= MAX_ERROR_REPEAT_COUNT:
            log(WARNING, f"***WARNING: Goal '{cloned_self.goal_title}' has been repeated {MAX_ERROR_REPEAT_COUNT} times. It will be marked as failed.")
        else:
            dependent_goals.append(cloned_self)
            self.goal_summary_suppress = True

        # insert the dependent goals after the current goal
        if len(dependent_goals) > 0:
            with self.goal_plan.plan_lock:
                self.goal_parent.insert_goals(dependent_goals, after_id=self.get_id_chain(), instantiated_plan=self.goal_instantiated_plan)

        if app_config.testing:
            log_plan(self.goal_plan)

        return
    
    def suppress_dependency_summaries(self) -> None:
        # suppress the summaries of the dependent goals
        if self.goal_dependent_upon is not None and len(self.goal_dependent_upon) > 0:
            for dep in self.goal_dependent_upon:
                # find the goal by looking at the parent subgoals, index of this goal plus the negative integer in the dependent_upon list
                dependent_index = dep + self.goal_id() - 1
                if dependent_index >= 0:
                    dependent_sibling = self.goal_parent.goal_subgoals[dependent_index]
                    dependent_sibling.goal_summary_suppress = True

    def cancel_subgoals(self, after:'Goal'=None, max_goals_to_cancel:int=None, target:'GoalTarget'=None, completion_reason:Optional[str]=None) -> None:
        if self.goal_subgoals is None or len(self.goal_subgoals) == 0:
            return
        
        if completion_reason is None:
            completion_reason = "<CANCELLED>" + (f" by order of '{after.goal_title}'" if after else "")

        idx = self.goal_subgoals.index(after) if after else -1
        cancel_count = 0
        for i in range(idx+1, len(self.goal_subgoals)):
            if max_goals_to_cancel is not None and cancel_count >= max_goals_to_cancel:
                break
            goal = self.goal_subgoals[i]
            if target is None or goal.goal_target == target:
                was_already_cancelled: bool = (
                    goal.goal_status == GoalStatus.CANCELLED
                    or (
                        goal.goal_status == GoalStatus.COMPLETED
                        and goal.goal_completion_reason is not None
                        and "<CANCELLED>" in goal.goal_completion_reason
                    )
                )
                goal.goal_status = GoalStatus.COMPLETED
                goal.goal_summary = None
                goal.goal_summary_pretty = None
                goal.goal_completion_reason = completion_reason
                if not was_already_cancelled:
                    log(WARNING, f"Goal {goal.get_id_chain_as_str()} '{goal.goal_title}' was cancelled at the user's request. Reason: {completion_reason}")
                cancel_count += 1

    def remove_goals(self, last_goal_targets_to_remove:List[GoalTarget], position:str, status:GoalStatus) -> bool:
        success = True

        if position == "last":
            # iterate in the last_goal_targets_to_remove, in reverse order, and remove the last subgoals with the same target
            for target in reversed(last_goal_targets_to_remove):
                for i in range(len(self.goal_subgoals)-1, -1, -1):
                    if self.goal_subgoals[i].goal_target == target and self.goal_subgoals[i].goal_status == status:
                        del self.goal_subgoals[i]
                        break
        else:
            # not implemented
            success = False
            log(ERROR, f"***ERROR: Removing goals at position '{position}' is not implemented.")

        return success

    def get_dependent_goals(self) -> List['Goal']:
        if self is None:
            return None
        
        dependent_goals = []
        if self.goal_dependent_upon is not None and len(self.goal_dependent_upon) > 0:
            for dep in self.goal_dependent_upon:
                # find the goal by looking at the parent subgoals, index of this goal plus the negative integer in the dependent_upon list
                dependent_index = dep + self.goal_id() - 1
                if dependent_index >= 0:
                    dependent_sibling = self.goal_parent.goal_subgoals[dependent_index]
                    dependent_goals.append(dependent_sibling)

        return dependent_goals

    def get_dependencies_status(self) -> GoalStatus:
        # returns NOT_STARTED if any of the dependent goals are not started
        # returns IN_PROGRESS if any of the dependent goals are in progress, or their children
        # returns CANCELLED if any of the dependent goals are cancelled
        # returns FAILED if any of the dependent goals are failed
        # returns COMPLETED if all dependent goals are completed
        dependency_status: GoalStatus = GoalStatus.COMPLETED
        
        if self.goal_dependent_upon is not None and len(self.goal_dependent_upon) > 0 and self.goal_id() > 1:
            for goal in self.goal_dependent_upon:
                # find the goal by looking at the parent subgoals, index of this goal plus the negative integer in the dependent_upon list
                dependent_index = goal + self.goal_id() - 1
                if dependent_index >= 0:
                    dependent_sibling = self.goal_parent.goal_subgoals[dependent_index]
                    if dependent_sibling.goal_status == GoalStatus.CANCELLED:
                        dependency_status = GoalStatus.CANCELLED
                        break
                    elif dependent_sibling.goal_status == GoalStatus.NOT_STARTED:
                        dependency_status = GoalStatus.NOT_STARTED
                    elif dependent_sibling.goal_status in [GoalStatus.IN_PROGRESS, GoalStatus.CHILDREN_IN_PROGRESS] and dependency_status == GoalStatus.COMPLETED:
                        dependency_status = GoalStatus.IN_PROGRESS

        return dependency_status
    
    def identify_goals_with_rag_fragments_suitable_for_self(self) -> List['Goal']:
        """Return the goals that `self` directly depends upon which contain RAG fragments.

        Behavior:
        - For each entry in `self.goal_dependent_upon` (which are relative/negative
          deltas pointing to previous goals on the same level), resolve the
          referenced Goal via `get_previous_goal(delta_from_self=...)`.
        - If the resolved Goal exists and has non-empty `goal_rag_fragments`,
          include it in the returned list.
        - Preserve the order of `goal_dependent_upon` and remove duplicates.
        """
        rag_goals: List['Goal'] = []

        if not self.goal_parent or not self.goal_dependent_upon:
            return rag_goals

        for dep in self.goal_dependent_upon:
            try:
                dependent_goal = self.get_previous_goal(delta_from_self=dep)
            except Exception:
                dependent_goal = None

            if dependent_goal is None:
                continue

            # include only if the dependent goal has RAG fragments or datasets
            if dependent_goal.goal_rag_fragments is not None and len(dependent_goal.goal_rag_fragments) > 0:
                rag_goals.append(dependent_goal)
            elif dependent_goal.goal_datasets is not None and len(dependent_goal.goal_datasets) > 0:
                rag_goals.append(dependent_goal)

        # remove duplicates while preserving order (use id chain as unique key)
        unique: List['Goal'] = []
        seen = set()
        for g in rag_goals:
            try:
                gid = tuple(g.get_id_chain())
            except Exception:
                gid = None
            if gid is None or gid in seen:
                continue
            seen.add(gid)
            unique.append(g)

        return unique

    def _collect_fragments_from_goal_rag_fragments(self, goal: 'Goal', minimum_rank: float, fragments_by_id: dict, current_char_count: int, enforce_budget: bool = True) -> tuple[int, bool]:
        """
        Private helper method to collect fragments from goal.goal_rag_fragments.
        
        Args:
            goal: The goal to collect fragments from
            minimum_rank: Minimum ranking threshold to keep a fragment
            fragments_by_id: Dictionary to accumulate fragments (modified in place)
            current_char_count: Current character count already collected
            
        Returns:
            Tuple of (characters_added, truncation_occurred)
        """
        chars_added: int = 0
        truncated: bool = False
        max_rag_message_length: int = LlmManager.instance().get_max_rag_message_length()
        warn_if_truncating_threshold: float = LlmManager.instance().get_warn_if_truncating_threshold()
        
        for fragment in goal.goal_rag_fragments or []:
            if isinstance(fragment, dict) and "ranking" in fragment:
                try:
                    ranking_float = float(fragment["ranking"] if fragment["ranking"] is not None else 0.0)
                    if ranking_float >= minimum_rank:
                        frag_id = fragment.get('fragmentid') or fragment.get('fragment_id') or fragment.get('id')
                        
                        # Calculate fragment text length for budget accounting.
                        # Stub fragments (text=None) use 2000 chars as a conservative
                        # estimate so the budget is applied correctly and not all stubs
                        # pass through to the renderer unchecked.
                        frag_text = fragment.get('text')
                        frag_text_len = len(str(frag_text)) if frag_text else 2000

                        # Check if adding this fragment would exceed the limit
                        if enforce_budget and current_char_count + chars_added + frag_text_len > max_rag_message_length:
                            truncated = ranking_float > warn_if_truncating_threshold
                            break
                        
                        if frag_id:
                            # Keep fragment with highest ranking
                            if frag_id not in fragments_by_id or ranking_float > float(fragments_by_id[frag_id].get('ranking', 0.0)):
                                # If replacing, subtract old fragment's length
                                if frag_id in fragments_by_id:
                                    old_text = fragments_by_id[frag_id].get('text', '')
                                    chars_added -= len(str(old_text)) if old_text else 0
                                frag_copy = fragment.copy()
                                frag_copy["_source_goal_id"] = tuple(goal.get_id_chain())
                                fragments_by_id[frag_id] = frag_copy
                                chars_added += frag_text_len
                        else:
                            # No ID, add to a list for fragments without IDs
                            if None not in fragments_by_id:
                                fragments_by_id[None] = []
                            frag_copy = fragment.copy() if isinstance(fragment, dict) else fragment
                            if isinstance(frag_copy, dict):
                                frag_copy["_source_goal_id"] = tuple(goal.get_id_chain())
                            fragments_by_id[None].append(frag_copy)
                            chars_added += frag_text_len
                except (ValueError, TypeError):
                    log(ERROR, f"WARNING: Conversion failed. '{fragment}' is not a valid float.")
                    # Add without deduplication if ranking conversion fails
                    frag_text = fragment.get('text', '')
                    frag_text_len = len(str(frag_text)) if frag_text else 0
                    
                    if enforce_budget and current_char_count + chars_added + frag_text_len > max_rag_message_length:
                        truncated = False
                        break
                    
                    if None not in fragments_by_id:
                        fragments_by_id[None] = []
                    frag_copy = fragment.copy() if isinstance(fragment, dict) else fragment
                    if isinstance(frag_copy, dict):
                        frag_copy["_source_goal_id"] = tuple(goal.get_id_chain())
                    fragments_by_id[None].append(frag_copy)
                    chars_added += frag_text_len
            else:
                # Fragment without ranking, add to list
                frag_text = fragment.get('text', '') if isinstance(fragment, dict) else ''
                frag_text_len = len(str(frag_text)) if frag_text else 0
                
                if enforce_budget and current_char_count + chars_added + frag_text_len > max_rag_message_length:
                    truncated = False
                    break
                
                if None not in fragments_by_id:
                    fragments_by_id[None] = []
                frag_copy = fragment.copy() if isinstance(fragment, dict) else fragment
                if isinstance(frag_copy, dict):
                    frag_copy["_source_goal_id"] = tuple(goal.get_id_chain())
                fragments_by_id[None].append(frag_copy)
                chars_added += frag_text_len
        
        return (chars_added, truncated)

    def _collect_fragments_from_goal_datasets(self, goal: 'Goal', minimum_rank: float, fragments_by_id: dict, current_char_count: int, enforce_budget: bool = True) -> tuple[int, bool]:
        """
        Private helper method to collect fragments from goal.goal_datasets.
        
        Args:
            goal: The goal to collect dataset fragments from
            minimum_rank: Minimum ranking threshold to keep a fragment
            fragments_by_id: Dictionary to accumulate fragments (modified in place)
            current_char_count: Current character count already collected
            
        Returns:
            Tuple of (characters_added, truncation_occurred)
        """
        chars_added: int = 0
        truncated: bool = False
        max_rag_message_length: int = LlmManager.instance().get_max_rag_message_length()
        warn_if_truncating_threshold: float = LlmManager.instance().get_warn_if_truncating_threshold()
        
        datasets = goal.get_datasets() or []
        log(DEBUG, f"_collect_fragments_from_goal_datasets: goal={goal.get_id_chain_as_str()}, datasets={len(datasets)}, minimum_rank={minimum_rank}")
        
        for dataset in datasets:
            dataset_id = dataset.get('id') if isinstance(dataset, dict) else dataset
            log(DEBUG, f"_collect_fragments_from_goal_datasets: processing dataset_id={dataset_id}, is_dict={isinstance(dataset, dict)}")
            
            if dataset_id:
                try:
                    dataset_service: DatasetInterface = get_dataset_service()
                    dataset_fragments = dataset_service.get_fragments_from_dataset(
                        dataset_id=dataset_id,
                        include_weights=True
                    )
                    log(DEBUG, f"_collect_fragments_from_goal_datasets: dataset_service returned {len(dataset_fragments) if dataset_fragments else 0} fragments")
                    # Convert to the format expected by consolidated_fragments
                    frag_count_processed = 0
                    frag_count_filtered_no_id = 0
                    frag_count_filtered_low_score = 0
                    frag_count_added = 0
                    
                    filtered_low_score_frags: list = []  # fragments below minimum_rank, kept for fill-up
                    for frag_dict in dataset_fragments or []:
                        if enforce_budget and truncated:
                            break
                            
                        frag_id = frag_dict.get('id')
                        if not frag_id:
                            frag_count_filtered_no_id += 1
                            log(DEBUG, f"_collect_fragments_from_goal_datasets: skipping fragment with no 'id': {frag_dict}")
                            continue
                        
                        # Calculate fragment text length
                        frag_text = frag_dict.get('text', '')
                        frag_text_len = len(str(frag_text)) if frag_text else 0
                        frag_count_processed += 1
                        
                        # Check if adding this fragment would exceed the limit
                        if enforce_budget and current_char_count + chars_added + frag_text_len > max_rag_message_length:
                            truncated = ranking_float > warn_if_truncating_threshold
                            break
                        
                        # Check if this fragment has a ranking/weight that meets minimum threshold
                        weight = frag_dict.get('weight', 1.0)
                        try:
                            ranking_float = float(weight)
                            if ranking_float >= minimum_rank:
                                # Add 'ranking' key for consistency with RAG fragments
                                frag_dict['ranking'] = ranking_float
                                frag_dict['fragmentid'] = frag_dict['id']
                                
                                # Keep fragment with highest ranking
                                if frag_id not in fragments_by_id or ranking_float > float(fragments_by_id[frag_id].get('ranking', 0.0)):
                                    # If replacing, subtract old fragment's length
                                    if frag_id in fragments_by_id:
                                        old_text = fragments_by_id[frag_id].get('text', '')
                                        chars_added -= len(str(old_text)) if old_text else 0
                                    frag_copy = frag_dict.copy()
                                    frag_copy['_source_goal_id'] = tuple(goal.get_id_chain())
                                    fragments_by_id[frag_id] = frag_copy
                                    chars_added += frag_text_len
                                    frag_count_added += 1
                            else:
                                frag_count_filtered_low_score += 1
                                log(DEBUG, f"_collect_fragments_from_goal_datasets: fragment {frag_id} filtered: weight={weight} < minimum_rank={minimum_rank}")
                                filtered_low_score_frags.append((ranking_float, frag_id, frag_dict))

                        except (ValueError, TypeError):
                            # If weight conversion fails, include it anyway with the raw weight
                            frag_dict['ranking'] = weight
                            frag_dict['fragmentid'] = frag_dict['id']
                            # Keep fragment with highest ranking (use raw weight)
                            try:
                                existing_ranking = float(fragments_by_id.get(frag_id, {}).get('ranking', 0.0))
                                if frag_id not in fragments_by_id or weight > existing_ranking:
                                    # If replacing, subtract old fragment's length
                                    if frag_id in fragments_by_id:
                                        old_text = fragments_by_id[frag_id].get('text', '')
                                        chars_added -= len(str(old_text)) if old_text else 0
                                    frag_copy = frag_dict.copy()
                                    frag_copy['_source_goal_id'] = tuple(goal.get_id_chain())
                                    fragments_by_id[frag_id] = frag_copy
                                    chars_added += frag_text_len
                            except:
                                # If comparison fails, just add it
                                if frag_id not in fragments_by_id:
                                    fragments_by_id[frag_id] = frag_dict
                                    chars_added += frag_text_len
                    
                    log(DEBUG, f"_collect_fragments_from_goal_datasets: dataset {dataset_id} summary: processed={frag_count_processed}, no_id={frag_count_filtered_no_id}, low_score={frag_count_filtered_low_score}, added={frag_count_added}")

                    # Fill-up: if fewer than MAX_RERANKING_OUTPUT_DOC fragments were collected,
                    # pad with the top-scoring below-threshold fragments (sorted descending by weight).
                    frag_count_before_fillup = len(fragments_by_id)
                    if frag_count_before_fillup < MAX_RERANKING_OUTPUT_DOC and filtered_low_score_frags and (not truncated or not enforce_budget):
                        needed = MAX_RERANKING_OUTPUT_DOC - frag_count_before_fillup
                        candidates = sorted(filtered_low_score_frags, key=lambda t: t[0], reverse=True)
                        log(INFO, f"_collect_fragments_from_goal_datasets: dataset {dataset_id} fill-up: only {frag_count_before_fillup} fragments above threshold, adding up to {needed} more from {len(candidates)} below-threshold candidates (MAX_RERANKING_OUTPUT_DOC={MAX_RERANKING_OUTPUT_DOC})")
                        frag_count_fillup = 0
                        for ranking_float, frag_id, frag_dict in candidates:
                            if frag_count_fillup >= needed:
                                break
                            if frag_id in fragments_by_id:
                                continue
                            frag_text = frag_dict.get('text', '')
                            frag_text_len = len(str(frag_text)) if frag_text else 0
                            if enforce_budget and current_char_count + chars_added + frag_text_len > max_rag_message_length:
                                truncated = False
                                break
                            frag_dict['ranking'] = ranking_float
                            frag_dict['fragmentid'] = frag_dict['id']
                            frag_copy = frag_dict.copy()
                            frag_copy['_source_goal_id'] = tuple(goal.get_id_chain())
                            fragments_by_id[frag_id] = frag_copy
                            chars_added += frag_text_len
                            frag_count_fillup += 1
                        log(DEBUG, f"_collect_fragments_from_goal_datasets: fill-up added {frag_count_fillup} fragment(s), total now {len(fragments_by_id)}")
                
                except Exception as e:
                    log(ERROR, f"Error fetching fragments from dataset {dataset_id}: {e}")
                    
                if enforce_budget and truncated:
                    break
        
        return (chars_added, truncated)

    def _build_category_only_rag_message_lines(self, consolidated_fragments: List[Union[str, dict]]) -> List[str]:
        """
        Size-aware renderer for category-only document stubs, plus any kg:// entities mixed in.

        This renderer is invoked (via `_build_rag_message_lines`) only when every fragment in
        `consolidated_fragments` is a bare document-level stub (a category-based search result
        with no chapter/paragraph suffix and no pre-fetched text), or a kg:// entity reference.
        It never receives paragraph/chapter/summary fragments that already carry ranked text -
        those are handled by the general `_build_rag_message_lines` renderer instead.

        Args:
            consolidated_fragments: List of fragment entries, where each entry is either:
                                - a dict with at least one of the keys 'fragmentid', 'fragment_id',
                                    'documentid', 'document_id', or 'id' holding the identifier. The actual
                                    fragment id is preferred over the owning document id. Two identifier shapes
                                    are recognised:
                    * a bare document id, e.g. "docid-42" (category-search document stub)
                    * a kg:// URI, e.g. "kg://inno-421" (knowledge-graph entity reference)
                  Other dict keys (e.g. 'ranking', 'text') are ignored by this renderer.
                - a plain string containing either identifier shape above.
              Duplicate identifiers (after normalisation) are collected only once.

        Returns:
            A list of strings, one per output line, that together form the full
            "<RAG_FRAGMENTS>...</RAG_FRAGMENTS>" block to be joined with '\n' and
            inserted into the LLM system message. The exact inner shape depends on
            which branch is taken (all branches still include any kg:// entity
            fragments, rendered by `_build_kg_entity_fragment_lines`, appended near the end):

          a) Count total chapter text chars in Neo4j for all matched documents.
          b) If it fits within MAX_RAG_MESSAGE_LENGTH_DEFAULT: fetch all chapters and render
             as standard <FRAGMENT> blocks.
          c) If chapters exceed the budget: count document summary text chars.
          c2) If summaries fit: fetch summaries and render as <SUMMARY> blocks.
          d.1) If even summaries exceed the budget: emit a structured guidance block
               that instructs the LLM to inform the user about refinement options
               (QueryExecutor, refined search, QueryAnalyser) without fabricating
               any document content.
        """
        max_chars: int = LlmManager.instance().get_max_rag_message_length()

        # ── Collect unique document IDs and kg:// entity IDs (preserve order) ─────
        # kg:// fragments are diverted here too because their fragment ID has no
        # chapter/paragraph suffix, so they parse as type='document'. They must be
        # split out and rendered separately — Document/Chapter/Summary queries below
        # never match an Entity node, which previously caused them to be silently dropped.
        doc_ids: List[str] = []
        kg_ids: List[str] = []
        kg_rankings: Dict[str, Any] = {}
        seen_ids: set = set()
        for f in consolidated_fragments:
            if isinstance(f, dict):
                doc_id = (f.get('fragmentid') or f.get('fragment_id')
                          or f.get('documentid') or f.get('document_id') or f.get('id'))
            else:
                doc_id = str(f)
            if not doc_id or doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            if doc_id.startswith("kg://"):
                kg_id = parse_kg_source_token(doc_id)
                if kg_id:
                    kg_ids.append(kg_id)
                if isinstance(f, dict):
                    ranking = f.get("ranking") if f.get("ranking") is not None else f.get("score")
                    if ranking is not None and kg_id not in kg_rankings:
                        kg_rankings[kg_id] = ranking
            else:
                # Strip a 'doc://' scheme and any trailing '?...' query string so the id
                # matches Neo4j's bare Document.id property, same as the kg_ids normalisation above.
                bare_doc_id = doc_id[len("doc://"):] if doc_id.startswith("doc://") else doc_id
                doc_ids.append(bare_doc_id.split("?", 1)[0])

        n_docs = len(doc_ids)
        # The renderer below emits those documents' chapters/summaries, so entity blocks must
        # not re-emit the same documents via IS_DOCUMENTED.
        kg_lines = self._build_kg_entity_fragment_lines(kg_ids, kg_rankings, excluded_document_ids=set(doc_ids) if doc_ids else None)
        if n_docs == 0:
            if kg_lines:
                return ["<RAG_FRAGMENTS>"] + kg_lines + ["</RAG_FRAGMENTS>"]
            return ["<RAG_FRAGMENTS>", "No documents found.", "</RAG_FRAGMENTS>"]

        category_content = get_documents_service().get_category_search_content(doc_ids, max_chars)
        total_chapter_chars: int = category_content["chapter_chars"]
        total_chapters: int = category_content["total_chapters"]
        chapter_rows = category_content["chapters"]
        total_summary_chars: int = category_content["summary_chars"]
        total_summaries: int = category_content["total_summaries"]
        summary_rows = category_content["summaries"]
        summary_fetch_failed: bool = category_content["summary_fetch_failed"]

        # ── Step b: Chapters fit → fetch and render ────────────────────────────────
        # total_chapters == 0 means these documents have no Chapter nodes at all (only a
        # Summary), not that their chapter text "fits" - fall through to the summary step instead.
        if total_chapters > 0 and total_chapter_chars <= max_chars:
            try:
                lines: List[str] = []
                lines.append("<RAG_FRAGMENTS>")
                lines.append(
                    f"The following fragments were retrieved from a category-based document search "
                    f"({n_docs} document(s), {total_chapters} chapter(s))."
                )
                lines.append(
                    "You can use them to answer the user's question or provide additional information."
                )
                lines.append("")
                for row in (chapter_rows or []):
                    fid  = row.get('fragmentid', '')
                    text = row.get('enhanced_text') or row.get('text') or ''
                    # Clean the fragment text to remove unnecessary attributes and newlines
                    cleaned_text = clean_fragment_html(str(text))
                    lines.append(f'<FRAGMENT id="{fid}">')
                    lines.append(cleaned_text)
                    lines.append("</FRAGMENT>")
                    lines.append("")
                lines.extend(kg_lines)
                lines.append("</RAG_FRAGMENTS>")
                return lines
            except Exception as e:
                log(ERROR, f"_build_category_only_rag_message_lines: error fetching chapters: {e}")
                # Fall through to summary attempt

        # ── Step c2: Summaries fit → fetch and render ──────────────────────────────
        if total_summary_chars <= max_chars and not summary_fetch_failed:
            try:
                lines = []
                lines.append("<RAG_FRAGMENTS>")
                lines.append(
                    f"The following are document summaries from a category-based search "
                    f"({n_docs} document(s) found). Full chapter content ({total_chapter_chars:,} chars) "
                    f"exceeded the context budget ({max_chars:,} chars); summaries are shown instead."
                )
                lines.append(
                    "You can use these summaries to answer the user's question."
                )
                lines.append("")
                for row in (summary_rows or []):
                    sid     = row.get('summaryid', '')
                    doc_id  = row.get('documentid', '')
                    title   = row.get('title') or ''
                    text    = row.get('enhanced_text') or row.get('text') or ''
                    title_safe = title.replace('"', '&quot;') if title else ''
                    title_attr = f' title="{title_safe}"' if title_safe else ''
                    lines.append(f'<SUMMARY id="{sid}" document="{doc_id}"{title_attr}>')
                    lines.append(str(text))
                    lines.append("</SUMMARY>")
                    lines.append("")
                lines.extend(kg_lines)
                lines.append("</RAG_FRAGMENTS>")
                return lines
            except Exception as e:
                log(ERROR, f"_build_category_only_rag_message_lines: error fetching summaries: {e}")
                # Fall through to guidance message

        # ── Step d.1: Nothing fits — structured guidance for the LLM ──────────────
        est_minutes = n_docs  # ~1 minute per document for QueryExecutor
        lines = []
        lines.append("<RAG_FRAGMENTS>")
        lines.append(
            f'<CATEGORY_SEARCH_RESULT doc_count="{n_docs}" '
            f'chapter_chars="{total_chapter_chars}" '
            f'summary_chars="{total_summary_chars}" '
            f'summary_fetch_failed="{str(summary_fetch_failed).lower()}" '
            f'limit="{max_chars}">'
        )
        if summary_fetch_failed:
            unavailable_reason: str = "document summaries could not be retrieved because a database error occurred"
            retrieval_reason: str = "a retrieval error"
        else:
            unavailable_reason = (
                f"both the full chapter content ({total_chapter_chars:,} chars) and document summaries "
                f"({total_summary_chars:,} chars) exceed the available context budget ({max_chars:,} chars)"
            )
            retrieval_reason = "context-size limits"

        lines.append(f"A category-based search found {n_docs} matching document(s), but {unavailable_reason}. No text content can be shown.")
        lines.append("")
        lines.append(
            "IMPORTANT — INSTRUCTIONS FOR THE LLM: You CANNOT provide fact-based answers about "
            "the content of these documents. Do NOT fabricate or infer document content. "
            f"Clearly inform the user that matching documents were found but their content could not "
            f"be retrieved due to {retrieval_reason}, and offer the following refinement options:"
        )
        lines.append("")
        lines.append(
            f"ref.1) DEEP ANALYSIS (QueryExecutor tool): Re-run this search with the QueryExecutor "
            f"tool, which performs parallel document-level analysis. "
            f"Estimated time: ~{est_minutes} minute(s) ({n_docs} document(s) × ~1 min each)."
        )
        lines.append(
            "ref.2) REFINED SEARCH: Add more category filters or combine with text/semantic search "
            "terms to reduce the number of matching documents to a manageable size."
        )
        lines.append(
            "ref.3) VISUAL ANALYSIS (QueryAnalyser tool): Request a graphical decomposition of the "
            "search results using the QueryAnalyser tool. This reveals the distribution of results "
            "and helps the user decide how to refine the search (ref.2) or target the analysis (ref.1)."
        )
        lines.append("</CATEGORY_SEARCH_RESULT>")
        lines.extend(kg_lines)
        lines.append("</RAG_FRAGMENTS>")
        return lines

    def _build_kg_entity_fragment_lines(self, kg_ids: List[str], rankings: Optional[Dict[str, Any]] = None, excluded_document_ids: Optional[Set[str]] = None) -> List[str]:
        """
        Fetch and render knowledge-graph entities as <FRAGMENT> blocks, independent of the document budget.

        Unlike document fragments, kg:// entities are never subject to the chapter/summary
        size budgeting performed by `_build_category_only_rag_message_lines`; every requested
        entity that exists in Neo4j is rendered in full. Entity lookup is delegated to
        `OntologyService.get_entities_by_ids()` (the same ACL-checked, consolidated-attribute
        method `AgentReranker` uses) instead of querying Neo4j directly here.

        Args:
            kg_ids: List of bare Neo4j entity node ids (the "kg://" scheme and any trailing
                "?..." query string already stripped by the caller), e.g. ["inno-421"].
                An empty or falsy list is a valid no-op input.

        Returns:
            A list of strings, one per output line, containing one
            '<FRAGMENT id="kg://<entity_id>">...</FRAGMENT>' block per entity id that was
            found (and ACL-permitted) in Neo4j (ids with no matching/permitted node are silently
            skipped), in the same order as `kg_ids`. The fragment body is the raw text from
            `OntologyService.entity_to_discourse_text()`; its newlines are preserved so
            serialized entity attributes remain readable. When supplied, `rankings` adds the
            reranker score to each matching FRAGMENT tag.
            Returns an empty list when `kg_ids` is empty.
        """
        if not kg_ids:
            return []

        kg_lookup = get_ontology_service().get_entities_by_ids(kg_ids, include_relationships=True)
        linked_document_char_budget: int = linked_document_budget_per_entity(len(kg_ids))

        lines: List[str] = []
        for entity_id in kg_ids:
            attributes = kg_lookup.get(entity_id)
            if attributes is None:
                continue
            fragment_text = get_ontology_service().entity_to_discourse_text(
                attributes,
                linked_document_char_budget=linked_document_char_budget,
                excluded_document_ids=excluded_document_ids,
            )
            rank_text = ""
            ranking = (rankings or {}).get(entity_id)
            if ranking is not None:
                try:
                    rank_text = f' rank="{float(ranking):.2f}"'
                except (TypeError, ValueError):
                    rank_text = f' rank="{ranking}"'
            lines.append(f'<FRAGMENT id="kg://{entity_id}"{rank_text}>')
            lines.append(str(fragment_text) if fragment_text else f"(Entity text for id 'kg://{entity_id}' not found.)")
            lines.append("</FRAGMENT>")
            lines.append("")
        return lines

    def _build_rag_message_lines(self, consolidated_fragments: List[Union[str, dict]]) -> List[str]:
        """
        Build the general-purpose "<RAG_FRAGMENTS>...</RAG_FRAGMENTS>" message lines for a
        consolidated, ranked, and budget-filtered list of fragments (as produced by
        `_generate_rag_messages`), rendering document paragraphs/chapters/summaries and
        kg:// knowledge-graph entities in ranked order.

        Before doing any rendering, this method first checks whether every entry in
        `consolidated_fragments` is a bare document-level stub (i.e.
        `DocumentsInterface.parse_fragment_id(...)` returns type == 'document', which is the
        case for category-based search results and for kg:// entities before the kg:// scheme
        is recognised). If so, it delegates entirely to the size-aware
        `_build_category_only_rag_message_lines` renderer instead of the logic below.

        Args:
            consolidated_fragments: List of fragment entries to render, each either:
                - a dict, optionally containing any of the following keys:
                    * 'paragraphid' / 'paragraph_id' / 'fragmentid' / 'fragment_id': the
                      fragment identifier (bare id or 'doc://'/'kg://' URI). Checked in this
                      priority order; the first non-empty value found is used.
                    * 'documentid' / 'document_id': the owning document id, used for display
                      only when the fragment id itself does not encode it.
                    * 'ranking' (float-like, optional): relevance score in [0, 1], rendered as
                      the FRAGMENT tag's `rank="..."` attribute when present.
                    * 'ranking_text' (optional): currently read but not rendered.
                    * 'text' / 'enhanced_text' (str, optional): fragment body already resolved
                      by the caller; when both are absent the text is looked up from Neo4j
                      (Chapter/Paragraph/Summary nodes for doc ids, Entity nodes for kg:// ids).
                    * 'duplicate' (bool, optional): currently read but not rendered.
                - a bare string containing a fragment id (plain, 'doc://', or 'kg://' form),
                  used as-is with its text always looked up from Neo4j.
              Paragraph-level fragment ids are promoted to display at their parent chapter's
              granularity (chapter id and chapter text are substituted automatically).

        Returns:
            A list of strings, one per output line, that together form the full
            "<RAG_FRAGMENTS>...</RAG_FRAGMENTS>" block (joined with '\n' by the caller) to be
            inserted into the LLM system message; each fragment becomes one
            '<FRAGMENT id="..." rank="...">...</FRAGMENT>' entry, in the same order as the
            input list. Returns None if `consolidated_fragments` is empty.
        """
        # ── Dispatch: category-only document/kg stubs get a size-aware renderer ──
        # A fragment is a document-level or kg:// stub when parse_fragment_id() returns
        # type='document' or type='kg_entity'. This happens for category-only Thalabus
        # queries (_rag_using_service skips the chapter fetch) and for reranked kg entities.
        # Both types must be classified together here: _build_category_only_rag_message_lines
        # already renders kg:// fragments alongside document stubs, so excluding kg_entity
        # from this predicate would wrongly divert mixed doc-stub+kg sets to the general
        # renderer below, which cannot resolve bare document-level ids (no Chapter/Paragraph/
        # Summary node shares that id).
        if consolidated_fragments:
            def _is_doc_or_kg_stub(f) -> bool:
                fid = (
                    (f.get('paragraphid') or f.get('paragraph_id') or f.get('fragmentid')
                     or f.get('fragment_id') or f.get('documentid') or f.get('document_id')
                     or f.get('id') or '')
                    if isinstance(f, dict) else str(f)
                )
                return DocumentsInterface.parse_fragment_id(fid).get('type') in ('document', 'kg_entity')
            
            if all(_is_doc_or_kg_stub(f) for f in consolidated_fragments):
                return self._build_category_only_rag_message_lines(consolidated_fragments)

        if not consolidated_fragments:
            return None
        
        overall_message_lines: List[str] = []
        overall_message_lines.append("<RAG_FRAGMENTS>")
        overall_message_lines.append("The following fragments were identified in the last RAG retrieval. You can use them to answer the user's question or to provide additional information.")
        overall_message_lines.append("The fragments are ranked by relevance (1=high, 0=very low). The ranking is based on the RAG model's scoring system.")
        overall_message_lines.append("")

        # ── Pre-fetch: batch-load all fragment texts in a single Neo4j query ─────
        # Paragraphs are promoted to their parent Chapter for display, so their
        # chapter IDs are pre-computed here.  Any fragment lacking both 'text' and
        # 'enhanced_text' is included in the batch.  This replaces the N+1 per-item
        # fallback that previously triggered one DB round-trip (~220 ms) per fragment.
        para_to_chapter: dict = {}   # original paragraph fragmentid → chapter_id
        ids_to_fetch: list = []      # unique IDs to look up in Neo4j
        kg_ids_to_fetch: list = []

        # ── Linked-document dedup: ids of documents already represented by an emitted fragment ──
        # Entity serialization re-emits whole source documents in <LINKED_DOCUMENT> blocks; when
        # the same document (or any of its fragments) is already rendered in this message, the
        # entity block is suppressed to avoid duplicate content.
        excluded_document_ids: Set[str] = set()
        for _ex_frag in consolidated_fragments:
            _ex_fid = (
                (_ex_frag.get('paragraphid') or _ex_frag.get('paragraph_id')
                 or _ex_frag.get('fragmentid') or _ex_frag.get('fragment_id')
                 or _ex_frag.get('documentid') or _ex_frag.get('document_id')
                 or _ex_frag.get('id') or '')
                if isinstance(_ex_frag, dict) else str(_ex_frag)
            )
            if not _ex_fid:
                continue
            _ex_doc = DocumentsInterface.parse_fragment_id(str(_ex_fid)).get("document_id")
            if _ex_doc:
                excluded_document_ids.add(_ex_doc)

        for _pf_frag in consolidated_fragments:
            if isinstance(_pf_frag, dict):
                _pf_fid = (_pf_frag.get('paragraphid') or _pf_frag.get('paragraph_id')
                           or _pf_frag.get('fragmentid') or _pf_frag.get('fragment_id') or '')
                # Use the centralised parser to handle both plain and 'doc://' IDs,
                # and to resolve paragraphs to their parent chapter ID for display.
                _pf_parsed = DocumentsInterface.parse_fragment_id(_pf_fid)
                if _pf_parsed["type"] == "paragraph" and _pf_parsed["chapter_id"]:
                    _pf_fetch_id = _pf_parsed["chapter_id"]
                    para_to_chapter[_pf_fid] = _pf_fetch_id
                else:
                    _pf_fetch_id = _pf_parsed["bare"] or _pf_fid
                _pf_needs = (_pf_fid in para_to_chapter
                             or (not _pf_frag.get('text')
                                 and not _pf_frag.get('enhanced_text')))
                if _pf_needs and _pf_fetch_id and _pf_fetch_id not in ids_to_fetch:
                    ids_to_fetch.append(_pf_fetch_id)

                if str(_pf_fid).startswith("kg://"):
                    kg_id = parse_kg_source_token(str(_pf_fid))
                    if kg_id and kg_id not in kg_ids_to_fetch:
                        kg_ids_to_fetch.append(kg_id)

            elif isinstance(_pf_frag, str) and _pf_frag:
                # Bare-string kg:// fragments need the same split as dict fragments, otherwise
                # they never reach kg_ids_to_fetch and the kg_lookup below always misses them.
                if _pf_frag.startswith("kg://"):
                    kg_id = parse_kg_source_token(_pf_frag)
                    if kg_id and kg_id not in kg_ids_to_fetch:
                        kg_ids_to_fetch.append(kg_id)
                elif _pf_frag not in ids_to_fetch:
                    ids_to_fetch.append(DocumentsInterface.normalize_fragment_id(_pf_frag))

        text_lookup: dict = {}   # id → {'text': ..., 'enhanced_text': ...}
        if ids_to_fetch:
            _pf_rows = get_documents_service().get_fragments_by_ids(ids_to_fetch)
            if _pf_rows:
                for _pf_row in _pf_rows:
                    text_lookup[_pf_row["fragmentid"]] = {"text": _pf_row.get("text"), "enhanced_text": _pf_row.get("enhanced_text")}

        kg_lookup: dict = get_ontology_service().get_entities_by_ids(kg_ids_to_fetch, include_relationships=True) if kg_ids_to_fetch else {}
        linked_document_char_budget: int = linked_document_budget_per_entity(len(kg_ids_to_fetch))

        for ranked_frag in consolidated_fragments:
            # If fragment is a dict, we expect documentid, fragmentid, ranking, ranking_text, text, duplicate
            if isinstance(ranked_frag, dict):
                fragment_id = ranked_frag.get('paragraphid') or ranked_frag.get('paragraph_id') or ranked_frag.get('fragmentid') or ranked_frag.get('fragment_id') or 'unknown-fragment-id'
                document_id = ranked_frag.get('documentid') or ranked_frag.get('document_id') or "-".join(fragment_id.split("-")[:2]) if "docid-" in fragment_id else 'unknown-document-id'
                ranking = ranked_frag.get('ranking')
                ranking_text = ranked_frag.get('ranking_text')
                text = ranked_frag.get('text')
                enhanced_text = ranked_frag.get('enhanced_text')
                duplicate = ranked_frag.get('duplicate', False)
                
                # Paragraph → Chapter promotion: use the centralised parser to handle
                # both 'doc://' URI and plain fragment IDs, resolve the chapter_id and
                # pull text/enhanced_text from the pre-fetched text_lookup.
                _fid_parsed = DocumentsInterface.parse_fragment_id(fragment_id)
                fragment_id = _fid_parsed["bare"] or fragment_id   # strip doc:// scheme
                if _fid_parsed["type"] == "paragraph" and _fid_parsed["chapter_id"]:
                    fragment_id = _fid_parsed["chapter_id"]
                    _fetched = text_lookup.get(fragment_id, {})
                    text = _fetched.get('text') or text
                    enhanced_text = _fetched.get('enhanced_text') or enhanced_text

                # Prepare ranking string for attribute
                ranking_str = ""
                if ranking is not None:
                    try:
                        ranking_float = float(ranking)
                        ranking_str = f"{ranking_float:.2f}"
                    except Exception:
                        ranking_str = str(ranking)
                
                # Dedup: cut <LINKED_DOCUMENT> blocks for documents already emitted as fragments
                if excluded_document_ids and fragment_id.startswith("kg://"):
                    if enhanced_text:
                        enhanced_text = strip_linked_document_blocks(str(enhanced_text), excluded_document_ids)
                    elif text:
                        text = strip_linked_document_blocks(str(text), excluded_document_ids)

                # Announce fragment with id and rank as attributes
                fragment_tag = f'<FRAGMENT id="{fragment_id}"'
                if ranking_str:
                    fragment_tag += f' rank="{ranking_str}"'
                fragment_tag += '>'
                overall_message_lines.append(fragment_tag)
                # Ensure text is a string
                if enhanced_text:
                    # Clean the fragment text to remove unnecessary attributes and newlines
                    overall_message_lines.append(clean_fragment_html(str(enhanced_text)))
                elif text:
                    # don't change the text content; just append it
                    overall_message_lines.append(clean_fragment_html(str(text)))
                else:
                    # No text in dict: look up from the pre-fetched batch (avoids per-item DB call)
                    if fragment_id.startswith("kg://"):
                        kg_id = parse_kg_source_token(fragment_id)
                        kg_attributes = kg_lookup.get(kg_id) if kg_id else None
                        fragment_text = get_ontology_service().entity_to_discourse_text(
                            kg_attributes,
                            linked_document_char_budget=linked_document_char_budget,
                            excluded_document_ids=excluded_document_ids or None,
                        ) if kg_attributes else None
                    else:
                        _fetched = text_lookup.get(fragment_id, {})
                        fragment_text = _fetched.get('enhanced_text') or _fetched.get('text')
                    if fragment_text is not None:
                        overall_message_lines.append(str(fragment_text))
                    else:
                        overall_message_lines.append(f"(Fragment text for id '{fragment_id}' not found.)")
                overall_message_lines.append("</FRAGMENT>")
                overall_message_lines.append("")

            elif isinstance(ranked_frag, str):
                # If it's just a fragment id string, present it and try fetching its text
                fragment_id = ranked_frag
                overall_message_lines.append(f'<FRAGMENT id="{fragment_id}">')
                # Use pre-fetched text_lookup (avoids per-item DB call)
                _fetched = text_lookup.get(fragment_id, {})
                if fragment_id.startswith("kg://"):
                    kg_id = parse_kg_source_token(fragment_id)
                    kg_attributes = kg_lookup.get(kg_id) if kg_id else None
                    fragment_text = get_ontology_service().entity_to_discourse_text(
                        kg_attributes,
                        linked_document_char_budget=linked_document_char_budget,
                        excluded_document_ids=excluded_document_ids or None,
                    ) if kg_attributes else None
                else:
                    fragment_text = _fetched.get('enhanced_text') or _fetched.get('text')
                if fragment_text is not None:
                    overall_message_lines.append(str(fragment_text))
                else:
                    overall_message_lines.append(f"(Fragment text for id '{fragment_id}' not found.)")

                overall_message_lines.append("</FRAGMENT>")
                overall_message_lines.append("")
            
            else:
                overall_message_lines.append(f"### ERROR: UNKNOWN DATA TYPE: '{ranked_frag}'")
                overall_message_lines.append("")

        overall_message_lines.append("</RAG_FRAGMENTS>")
        return overall_message_lines

    def _truncate_rag_message(self, overall_message: str, consolidated_fragments: List[Union[str, dict]]) -> str:
        """
        Private helper method to truncate RAG message if it exceeds maximum length.

        For semantic/text searches (d.2): when the rendered message still exceeds
        MAX_RAG_MESSAGE_LENGTH_DEFAULT, replace the overflow portion with a structured
        guidance block that instructs the LLM to inform the user and offer
        refinement options, instead of silently cutting off mid-text.
        
        Args:
            overall_message: The complete RAG message string
            consolidated_fragments: List of fragments to extract document IDs from
            
        Returns:
            Truncated message string with document ID list, or original message if no truncation needed
        """
        max_rag_message_length: int = LlmManager.instance().get_max_rag_message_length()

        # check for maximum length
        if len(overall_message) > max_rag_message_length:
            log(WARNING, f"Goal {self.get_id_chain_as_str()} RAG message length {len(overall_message)} exceeds maximum of {max_rag_message_length}. Truncating message.")
            list_of_document_ids = []
            for ranked_frag in consolidated_fragments:
                if isinstance(ranked_frag, dict):
                    document_id = ranked_frag.get('documentid') or ranked_frag.get('document_id') or 'unknown-document-id'
                    if document_id not in list_of_document_ids:
                        list_of_document_ids.append(document_id)
            n_docs = len(list_of_document_ids)
            # d.2: Build the suffix (truncation notice + document IDs + guidance).
            # Then truncate the content so content + suffix fits within the budget.
            doc_ids_block = (
                "<RAG_DOCUMENT_IDS>The following document ids were included in the original message:\n"
                + ",".join(list_of_document_ids) + "\n</RAG_DOCUMENT_IDS>\n"
            )
            guidance = (
                "\n<SEARCH_RESULT_OVERFLOW "
                f'doc_count="{n_docs}" limit="{max_rag_message_length}">\n'
                "The RAG fragments exceed the available context budget and could not all be included. "
                "You CANNOT provide complete fact-based answers. Clearly inform the user that not all "
                "retrieved content could be shown and offer the following refinement options:\n\n"
                "ref.1) DEEP ANALYSIS (QueryExecutor tool): Re-run this search with the QueryExecutor "
                f"tool for parallel document-level analysis (~{n_docs} minute(s) estimated).\n"
                "ref.2) REFINED SEARCH: Add more specific filters or terms to reduce the result set.\n"
                "ref.3) VISUAL ANALYSIS (QueryAnalyser tool): Request a graphical decomposition of "
                "results to identify how to best refine the search.\n"
                "</SEARCH_RESULT_OVERFLOW>\n"
            )
            suffix = TRUNCATED_MESSAGE + "\n\n" + doc_ids_block + guidance
            content_budget = max(0, max_rag_message_length - len(suffix))
            overall_message = overall_message[:content_budget] + suffix
        
        return overall_message

    def _generate_rag_messages(self, goals: Union['Goal', List['Goal']], minimum_rank: float = MIN_RERANKING_SCORE_FRAG) -> str:
        """
        Consolidate RAG fragments from one or many Goal instances and return a single overall message.

        Args:
            goals: A single Goal or a list of Goal instances to collect fragments from.
            minimum_rank: Minimum ranking threshold to keep a fragment (float-like).

        Returns:
            A consolidated overall_message string or None if no fragments found.
        """
        # normalize goals to a list
        if goals is None:
            return None
        if not isinstance(goals, list) and not isinstance(goals, tuple):
            goals_list = [goals]
        else:
            goals_list = list(goals)

        consolidated_fragments: List[Union[str, dict]] = []
        truncation_occurred = False
        all_collected_document_ids = []
        rag_stats: List[dict] = []  # Collect statistics for RAG fragments

        # create a list of fragments above minimum ranking based on:
        #  a) fragments from dependent goals, that may be in the goal_rag_fragments attribute
        #  b) fragments from the goal_datasets attribute, for the datasets that contain Document, Chapter or Paragraph nodes
        for goal in goals_list:
            if goal is None:
                continue

            # Use a dict to track best fragment by ID (keeping highest ranking)
            fragments_by_id = {}
            
            # a) collect all ranked candidates from goal.goal_rag_fragments (no budget cutoff here)
            _n_input_frags = len(goal.goal_rag_fragments) if goal.goal_rag_fragments else 0
            chars_added_rag, truncated_rag = self._collect_fragments_from_goal_rag_fragments(goal, minimum_rank, fragments_by_id, 0, enforce_budget=False)

            # b) collect all ranked candidates from goal.goal_datasets (no budget cutoff here)
            chars_added_ds, truncated_ds = self._collect_fragments_from_goal_datasets(goal, minimum_rank, fragments_by_id, 0, enforce_budget=False)
            
            _n_collected = len(fragments_by_id) - (1 if None in fragments_by_id else 0) + (len(fragments_by_id.get(None, [])) if None in fragments_by_id else 0)
            log(INFO, f"_generate_rag_messages: goal {goal.get_id_chain_as_str()}: input_frags={_n_input_frags}, collected={_n_collected} (budget applied after global sort)")

            # Convert dict to list, preserving fragments without IDs
            filtered = []
            for frag_id, frag in fragments_by_id.items():
                if frag_id is None:
                    # Add all fragments without IDs
                    filtered.extend(frag)
                else:
                    filtered.append(frag)
                    
                # Collect document IDs for potential truncation message
                if isinstance(frag, dict):
                    doc_id = frag.get('documentid') or frag.get('document_id')
                    if doc_id and doc_id not in all_collected_document_ids:
                        all_collected_document_ids.append(doc_id)
                elif isinstance(frag, list):
                    for f in frag:
                        if isinstance(f, dict):
                            doc_id = f.get('documentid') or f.get('document_id')
                            if doc_id and doc_id not in all_collected_document_ids:
                                all_collected_document_ids.append(doc_id)
            
            # extend consolidated list preserving order across goals
            consolidated_fragments.extend(filtered)
            
            # Collect statistics for this goal's RAG fragments
            stat = self._collect_rag_statistics(goal, fragments_by_id)
            rag_stats.append(stat)

        if len(consolidated_fragments) == 0:
            return None

        # Sort consolidated fragments by ranking (highest first), then by normalized fragment id
        def _sort_key(ranked_frag):
            # Primary sort: ranking (descending)
            ranking = 0.0
            if isinstance(ranked_frag, dict):
                try:
                    ranking = float(ranked_frag.get('ranking', 0.0))
                except (ValueError, TypeError):
                    ranking = 0.0
            
            # Secondary sort: normalized fragment id (ascending)
            fragment_id = ''
            if isinstance(ranked_frag, dict):
                fragment_id = ranked_frag.get('paragraphid') or ranked_frag.get('paragraph_id') or ranked_frag.get('fragmentid') or ranked_frag.get('fragment_id') or ''
                # Normalize paragraph ids to chapter ids similarly to the presentation logic
                if "-p" in fragment_id:
                    last = fragment_id.split("-")[-1]
                    if last.startswith("p") and last[1:].isdigit():
                        fragment_id = fragment_id.rsplit("-", 1)[0]
            else:
                fragment_id = str(ranked_frag)
            
            return (-ranking, str(fragment_id).lower())  # Negative ranking for descending order

        try:
            # sort fragments by ranking (descending) and then by normalized fragment id (ascending)
            consolidated_fragments = sorted(consolidated_fragments, key=_sort_key)

            # de-duplicate fragments: keep the highest ranked occurrence of each normalized fragment id
            seen_ids = set()
            deduped_fragments: List[Dict[str, Any]] = []
            for frag in consolidated_fragments:
                frag_id = ''
                if isinstance(frag, dict):
                    frag_id = frag.get('paragraphid') or frag.get('paragraph_id') or frag.get('fragmentid') or frag.get('fragment_id') or ''
                    # Normalize paragraph ids to chapter ids similarly to the presentation logic
                    if "-p" in frag_id:
                        last = frag_id.split("-")[-1]
                        if last.startswith("p") and last[1:].isdigit():
                            frag_id = frag_id.rsplit("-", 1)[0]
                else:
                    frag_id = str(frag)
                
                if frag_id not in seen_ids:
                    seen_ids.add(frag_id)
                    deduped_fragments.append(frag)
            
            consolidated_fragments = deduped_fragments

        except Exception as e:
            # If sorting fails for any reason, fall back to original order
            log(ERROR, f"Error sorting/deduplicating fragments for goal {self.get_id_chain_as_str()}. Falling back to original order. Exception: {e}\n{traceback.format_exc()}")
            pass

        # Apply the character budget only after all candidates have been merged,
        # ranked, and deduplicated. This prevents low-ranked early results from
        # consuming the budget before higher-ranked fragments are considered.
        max_rag_message_length = LlmManager.instance().get_max_rag_message_length()
        # Walk fragments in descending rank order (already sorted).  Stop as soon as a
        # fragment would exceed the budget — every fragment that fits outranks every
        # fragment that is excluded, satisfying the invariant: "higher-ranked fragments
        # are included; lower-ranked ones are excluded when the budget is full."
        selected_fragments: List[Union[str, dict]] = []
        selected_char_count = 0
        for fragment in consolidated_fragments:
            # Use enhanced_text over text: the renderer prefers enhanced_text and it is
            # typically larger, so it gives a more accurate pre-render budget estimate.
            # Fall back to text, then a conservative 2000-char stub estimate.
            if isinstance(fragment, dict):
                fragment_text = fragment.get('enhanced_text') or fragment.get('text')
            else:
                fragment_text = None
            fragment_length = len(str(fragment_text)) if fragment_text else 2000
            if selected_char_count + fragment_length > max_rag_message_length:
                ranking_value = fragment.get('ranking') if isinstance(fragment, dict) else None
                try:
                    ranking_float = float(ranking_value) if ranking_value is not None else 0.0
                except (ValueError, TypeError):
                    ranking_float = 0.0
                truncation_occurred = ranking_float > LlmManager.instance().get_warn_if_truncating_threshold()
                break  # all remaining fragments are lower-ranked; stop here
            selected_fragments.append(fragment)
            selected_char_count += fragment_length
        consolidated_fragments = selected_fragments
        
        # Update rag_stats with final selected fragment counts
        self._update_rag_statistics_with_selection(rag_stats, consolidated_fragments)
        
        # Build a markdown-friendly consolidated message
        overall_message_lines = self._build_rag_message_lines(consolidated_fragments)
        if overall_message_lines:
            overall_message = "\n".join(overall_message_lines)
        else:
            overall_message = ""

        # Add truncation notice if necessary
        if truncation_occurred:
            log(WARNING, f"Goal {self.get_id_chain_as_str()} RAG message was truncated during collection. Character limit: {max_rag_message_length}, selected: {selected_char_count} chars across {len(consolidated_fragments)} fragments")
            overall_message += "\n\n" + TRUNCATED_MESSAGE + "\n\n"
            if len(all_collected_document_ids) > 0:
                overall_message += "<RAG_DOCUMENT_IDS>The following document ids were included in the message:\n" + ",".join(all_collected_document_ids) + "\n</RAG_DOCUMENT_IDS>\n"

        # Post-render safety net: the pre-render estimates above may undercount
        # (e.g. stubs whose text is fetched from Neo4j by the renderer).  If the
        # rendered message still exceeds the budget, truncate it here.
        overall_message = self._truncate_rag_message(overall_message, consolidated_fragments)
        
        # Store statistics in goal_debug for the current goal
        if rag_stats:
            if self.goal_debug is None:
                self.goal_debug = {}
            self.goal_debug["rag_stats"] = rag_stats

        return overall_message

    def _collect_rag_statistics(self, goal: 'Goal', fragments_by_id: dict) -> dict:
        """
        Collect RAG statistics for a single goal.
        
        Args:
            goal: The goal to collect statistics for
            fragments_by_id: Dictionary of fragments collected from this goal
            
        Returns:
            Dictionary containing RAG statistics for this goal
        """
        # Determine source type and source dataset
        source_type = "dataset" if goal.goal_datasets and len(goal.get_datasets() or []) > 0 else "list"
        source_ds = None
        if source_type == "dataset":
            dataset_ids = goal.get_dataset_ids()
            if dataset_ids:
                source_ds = f"ds://{dataset_ids[0]}"  # Use first dataset ID
        
        # Calculate ranking statistics from fragments_by_id
        rag_rank_highest = 0.0
        rag_rank_lowest = float('inf')
        rag_fragments_used = 0
        for frag_id, frag in fragments_by_id.items():
            if frag_id is not None:
                if isinstance(frag, dict):
                    ranking = frag.get('ranking')
                    try:
                        ranking_float = float(ranking) if ranking is not None else 0.0
                        rag_rank_highest = max(rag_rank_highest, ranking_float)
                        rag_rank_lowest = min(rag_rank_lowest, ranking_float)
                        rag_fragments_used += 1
                    except (ValueError, TypeError):
                        pass
        
        # Handle case where no fragments were collected
        if rag_fragments_used == 0:
            rag_rank_lowest = 0.0
        elif rag_rank_lowest == float('inf'):
            rag_rank_lowest = rag_rank_highest
        
        # Check for too many fragments warning
        rag_too_many_fragments_warning = goal.goal_rag_toomanyfragments if hasattr(goal, 'goal_rag_toomanyfragments') and goal.goal_rag_toomanyfragments else False
        
        return {
            "source_type": source_type,
            "source_goal": goal.get_id_chain(),
            "source_ds": source_ds,
            "rag_fragments_used": rag_fragments_used,
            "rag_rank_highest": rag_rank_highest,
            "rag_rank_lowest": rag_rank_lowest,
            "rag_threshold_warning": LlmManager.instance().get_warn_if_truncating_threshold(),
            "rag_too_many_fragments_warning": rag_too_many_fragments_warning
        }

    def _update_rag_statistics_with_selection(self, rag_stats: List[dict], consolidated_fragments: List[Union[str, dict]]) -> None:
        """
        Update RAG statistics with final selected fragment counts after budget filtering.
        
        Args:
            rag_stats: List of statistics dictionaries to update
            consolidated_fragments: List of fragments that made it through the budget filter
        """
        for stat in rag_stats:
            # Count how many fragments from this source made it into selected_fragments
            source_goal_id = tuple(stat["source_goal"])
            selected_count = 0
            selected_highest = 0.0
            selected_lowest = float('inf')
            
            for frag in consolidated_fragments:
                if isinstance(frag, dict):
                    frag_goal_id = frag.get('_source_goal_id')
                    if frag_goal_id == source_goal_id:
                        selected_count += 1
                        ranking = frag.get('ranking')
                        try:
                            ranking_float = float(ranking) if ranking is not None else 0.0
                            selected_highest = max(selected_highest, ranking_float)
                            selected_lowest = min(selected_lowest, ranking_float)
                        except (ValueError, TypeError):
                            pass
            
            if selected_count > 0:
                if selected_lowest == float('inf'):
                    selected_lowest = selected_highest
                stat["rag_fragments_used"] = selected_count
                stat["rag_rank_highest"] = selected_highest
                stat["rag_rank_lowest"] = selected_lowest

    def _fetch_ontology_rules(self, ontology_id:str, community_ids:List[str]) -> str:
        if self.goal_plan.plan_community_ids is not None and len(self.goal_plan.plan_community_ids) > 0:
            # if there are communities in the plan, we skip details, such as entity, relationship attributes and a list of entities and relationships
            skip_details = False
        elif self.goal_plan.plan_loaded_documents is not None and len(self.goal_plan.plan_loaded_documents) > 0:
            # if there loaded documents in the plan, we skip details, such as entity, relationship attributes and a list of entities and relationships
            skip_details = False
        else:
            skip_details = True

        ontology_service: OntologyInterface = get_ontology_service()
        ontology_rules = ontology_service.get_ontology_and_entites_as_string(self, ontology_id=ontology_id, community_ids=community_ids, skip_details=skip_details)

        return ontology_rules
        
    def _is_this_the_last_user_message(self, g:'Goal', agent_name:str) -> bool:
        # returns True if the goal 'g' is the last user message (that is, produced by AgentUser)
        # in the conversation, otherwise False

        # find the current goal chain level, which is the parent goal's subgoals
        same_level_goals = self.goal_parent.goal_subgoals if self.goal_parent else []

        # find out the last user message in the same level goals. Scan in reverse order, so that the first one found is the last one
        last_user_message = None
        for goal in reversed(same_level_goals):
            if goal.goal_target_agent is not None and goal.goal_target_agent_name.startswith(agent_name) and goal.goal_status == GoalStatus.COMPLETED:
                last_user_message = goal
                break

        is_last = last_user_message == g
        return is_last
    
    def get_hints(self, agent_name: str = None, user_prompt: str = None, top_n: int = TOP_N_HINTS, min_score: float = MIN_HINT_SCORE, exclude: Optional[List[HintRecommendation]] = None) -> List[Dict[str, str]]:
        """
        Retrieve the most relevant hint from Neo4j based on the agent name and the prompt by vectorizing the user prompt.
        Only return hints where the similarity score is at least min_score (unless min_score is -1, which means no minimum).
        Hints whose analysis recommendation is in 'exclude' are never returned.
        When 'exclude' is None (default), the hints service's centralized default exclusion applies (Redundant and every CONFLICT_* recommendation); an empty list disables it.
        Each returned hint is a dict with the hint 'id' and the hint 'text' for the requesting agent.
        """
        best_hints: List[Dict[str, str]] = get_best_hints_for_agent(agent_name, user_prompt, top_n, min_score, exclude)
        return best_hints if best_hints else []

    def get_user_memories(self) -> Optional[Dict[str, str]]:
        """
        Retrieve the current user's profile memories (name, initials, email) from the :User node
        in Neo4j, cached per user identity for USER_MEMORIES_CACHE_TTL_SECONDS so a plan that calls
        the LLM several times does not query Neo4j on every build_messages() call, while profile
        edits made by the user still become visible quickly.
        Returns None when no :User node can be resolved for the current session identity.
        """
        memories: Optional[Dict[str, str]] = None

        try:
            user_service = get_user_service()
            user_uuid: Optional[str] = user_service.get_user_uuid()
            user_name: Optional[str] = user_service.get_user_id()
            cache_key: str = f"{user_uuid}|{user_name}"
            cached: bool = False
            now: float = time.time()

            with _user_memories_cache_lock:
                cache_entry: Optional[Tuple[float, Optional[Dict[str, str]]]] = _user_memories_cache.get(cache_key)
                if cache_entry is not None and (now - cache_entry[0]) < USER_MEMORIES_CACHE_TTL_SECONDS:
                    memories = cache_entry[1]
                    cached = True

            if cached == False and (user_uuid or user_name):
                user_records: Optional[List[Dict[str, str]]] = execute_neo4j_query(
                    """
                    MATCH (u:User)
                    WHERE u.id = $user_uuid OR u.username = $user_name
                    RETURN u.fullname AS name, u.initials AS initials, u.email AS email
                    LIMIT 1
                    """,
                    {"user_uuid": user_uuid, "user_name": user_name},
                )
                if user_records and len(user_records) > 0:
                    memories = user_records[0]

                with _user_memories_cache_lock:
                    _user_memories_cache[cache_key] = (time.time(), memories)

        except Exception as e:
            log(ERROR, f"Could not load user memories from the :User node: {e}. Traceback:\n{traceback.format_exc()}")

        return memories

    def _build_memories_message(self) -> Optional[Dict[str, str]]:
        """
        Build the system message exposing the current user's profile memories, formatted as:
            <MEMORIES>
            <MEMORY id="user_preferences">
            Name: ...
            Initials: ...
            Email: ...
            </MEMORY>
            </MEMORIES>
        Returns None when no user memories are available.
        """
        memory_message: Optional[Dict[str, str]] = None

        try:
            memories: Optional[Dict[str, str]] = self.get_user_memories()
            if memories is not None:
                memory_lines: List[str] = []
                name: Optional[str] = memories.get("name")
                initials: Optional[str] = memories.get("initials")
                email: Optional[str] = memories.get("email")
                if name:
                    memory_lines.append(f"Name: {name}")
                if initials:
                    memory_lines.append(f"Initials: {initials}")
                if email:
                    memory_lines.append(f"Email: {email}")

                if len(memory_lines) > 0:
                    memories_text: str = (
                        "<MEMORIES>\n"
                        + "<MEMORY id=\"user_preferences\">\n"
                        + "\n".join(memory_lines)
                        + "\n</MEMORY>\n</MEMORIES>\n"
                    )
                    memory_message = {"role": "system", "content": memories_text}

        except Exception as e:
            log(ERROR, f"Could not build the user memories message: {e}. Traceback:\n{traceback.format_exc()}")

        return memory_message
    
    def _get_model_reasoning_prompt(self, agent:'Agent'=None, purpose:str=None) -> str|None:
        reasoning_prompt = None

        llm_manager = LlmManager.instance()
        llmconfiguration = llm_manager.get_llm_model_configuration_for_agent(agent=agent, agent_purpose=purpose)
        if llmconfiguration is not None:
            if "is_reasoning" in llmconfiguration and llmconfiguration["is_reasoning"]:
                model_provider = llmconfiguration.get("modelprovider", "openai")
                endpoint = llmconfiguration.get("endpoint", None)
                llm_interface = llm_manager.get_interface(model_provider, endpoint)
                reasoning_settings = llm_manager.get_active_reasoning()
                reasoning_prompt = llm_interface._get_reasoning_prompt(llmconfiguration, reasoning_settings)

        return reasoning_prompt

    def load_documents_or_chapters(self, content_id_attribute:str, content_ids:str, content_label:str, load_enhanced_text:bool, only_if_connected_to:Optional[str]=None) -> dict:
        response = None

        try:
            log(DEBUG, f"loading from neo4j {content_label} '{content_ids}'...")

            # load the document
            content_ids_as_array = content_ids.split(",") if content_ids else []
            
            result = get_documents_service().get_content_for_goal(
                content_id_attribute,
                content_ids_as_array,
                content_label,
                only_if_connected_to,
            )
            if len(result) == 0:
                log(WARNING, f"Could not find {content_label} '{content_ids}' in the database.")
                return None
            elif len(result) > 1:
                log(WARNING, f"More than one {content_label} found for '{content_ids}'. Only the first one will be returned.")
                result = [result[0]]

            for record in result:
                content_id = record.get("id")
                doc_name = record.get("name")
                doc_title = record.get("title")
                doc_text = record.get("text")
                doc_enhanced_text = record.get("enhanced_text")
                doc_enhanced_type = record.get("enhanced_type") if load_enhanced_text and record.get("enhanced_type") is not None else "plaintext"
                dossier_id = record.get("dossier_id")
                if load_enhanced_text and (doc_enhanced_text is None or len(doc_enhanced_text) == 0):
                    # regenerate the tag ids, which might have been manipulated by previous operations
                    if doc_enhanced_text is None or len(doc_enhanced_text) == 0:
                        doc_enhanced_text = generate_html_from_text(doc_text)               # pass 1: convert text to html
                        doc_enhanced_text = add_html_tag_ids(doc_enhanced_text)             # pass 2: add unique ids to each html tag
                        doc_enhanced_text = eliminate_duplicate_newlines(doc_enhanced_text) # pass 3: eliminate duplicate newlines
                        self._save_document_or_chapter_enhanced_text(content_id_attribute, content_id, content_label, doc_enhanced_text, doc_enhanced_type)

                response = { "id": content_id, "text": doc_text, "enhanced_text": doc_enhanced_text, "name": doc_name, "title": doc_title, "content_type": doc_enhanced_type, "dossier_id": dossier_id }

            log(DEBUG, f"...loaded.")

        except Exception as e:
            log(ERROR, f"Contextualizer Load Document generated an exception: {e}\n{traceback.format_exc()}")
            response = None

        return response
    
    def _save_document_or_chapter_enhanced_text(self, content_id_attribute:str, content_id:str, content_label:str, enhanced_text:str, enhanced_type:str) -> None:
        try:
            log(DEBUG, f"Saving enhanced text for document/chapter '{content_id}'...")
            saved = get_documents_service().update_content_enhanced_text(
                content_id_attribute,
                content_id,
                content_label,
                enhanced_text,
                enhanced_type,
            )
            if saved:
                log(DEBUG, f"...saved.")
            else:
                log(WARNING, f"***WARNING***: Could not save enhanced text for document/chapter '{content_id}'.")
        except Exception as e:
            log(ERROR, f"Could not save enhanced text for document/chapter '{content_id}': {e}\n{traceback.format_exc()}")

        return

    def _get_dossier_information(self, dossier_id:Optional[str], include_summaries:bool=True, include_text:bool=False, is_session_dossier:bool=False) -> str:
        dossier_information = ""

        if dossier_id is not None:
            try:
                # Get the dossier service and retrieve dossier information
                dossiers_service: DossiersInterface = get_dossiers_service()
                dossier = dossiers_service.get_dossier(dossier_id)
                
                if dossier is not None and "success" in dossier and dossier["success"]:
                    session_attr = ' is_session_dossier="true"' if is_session_dossier else ""
                    dossier_information += f"""<DOSSIER id="{dossier_id}"{session_attr}>\n"""

                    # Add associated reasoning template information if available
                    reasoning_template_ids = dossier.get("reasoning_ids")
                    if reasoning_template_ids is not None and isinstance(reasoning_template_ids, list) and len(reasoning_template_ids) > 0:
                        dossier_information += f"""The dossier id='{dossier_id}' is associated with reasoning template_ids='{','.join(reasoning_template_ids)}'.\n"""

                    # Get the documents belonging to this dossier
                    documents = dossiers_service.get_documents_of_dossier(dossier_id)
                    
                    if documents and len(documents) > 0:
                        dossier_information += f"""The dossier id='{dossier_id}' contains the following documents:\n"""
                        
                        # Get the documents service to retrieve detailed information
                        documents_service = get_documents_service()

                        # Collect all document IDs and batch-fetch in a single query.
                        # text/enhanced_text are only requested from Neo4j when include_text is True.
                        # Summary is always fetched here (include_summaries controls whether it is rendered).
                        document_ids_ordered = [doc_info.get("id") for doc_info in documents if doc_info.get("id")]
                        fetched_map = documents_service.get_documents(document_ids_ordered, include_text=include_text, include_summary=include_summaries)
                        fetched_documents = [(doc_id, fetched_map.get(doc_id)) for doc_id in document_ids_ordered]

                        # Calculate total text length to decide whether to include text for all or none
                        total_text_length = 0
                        if include_text:
                            for _, document in fetched_documents:
                                if document is not None and document.get("success"):
                                    document_text = document.get("enhanced_text") or document.get("text") or None
                                    if document_text is not None:
                                        total_text_length += len(document_text)

                        # Include text for all documents only if the total fits within the limit
                        max_loaded_document_length = LlmManager.instance().get_max_loaded_document_length()
                        include_text_for_all = include_text and total_text_length <= max_loaded_document_length

                        for document_id, document in fetched_documents:
                            if document is not None and "success" in document and document["success"]:
                                document_name = document.get("name") or ""
                                document_title = document.get("title") or ""
                                dossier_information += f"""<DOSSIER_DOCUMENT_SUMMARY id="{document_id}" name="{document_name}" title="{document_title}">\n"""
                                
                                if include_summaries:
                                    document_summary = document.get("summary")
                                    if document_summary is not None and len(document_summary) > 0:
                                        dossier_information += f"""<SUMMARY>{document_summary}</SUMMARY>\n"""
                                
                                if include_text_for_all:
                                    document_text = document.get("enhanced_text") or document.get("text") or None
                                    if document_text is not None:
                                        dossier_information += f"""<FULL_TEXT>\n{document_text}\n</FULL_TEXT>\n"""

                                dossier_information += "</DOSSIER_DOCUMENT_SUMMARY>\n"

                        dossier_information += "</DOSSIER>\n"
                    elif "session-dossier-" not in dossier_id:
                        dossier_information += f"""
<DOSSIER>
The dossier with dossier_id '{dossier_id}' contains no documents.
</DOSSIER>
"""
                    else:
                        # Session dossier with no documents: contribute nothing
                        dossier_information = ""
                else:
                    log(WARNING, f"Dossier '{dossier_id}' not found or returned error")
            except Exception as e:
                log(ERROR, f"Error retrieving dossier information for '{dossier_id}': {e}\n{traceback.format_exc()}")
        
        return dossier_information

    def _get_document_summaries(self, document_ids:List[str]) -> str:
        """
        Generate a string with information about referenced documents and their summaries.
        
        Args:
            document_ids: List of document IDs to be processed.
                         Can be None, a list, or other types (will be handled appropriately).
        
        Returns:
            A string with document summaries information, or empty string if no documents.
        """
        result = ""
        
        # Add information about referenced documents and their summaries
        if document_ids is not None and len(document_ids) > 0:
            result += f"""\n\n<DOCUMENT_SUMMARIES>\nSome documents have been identified during the last RAG retrieval. Here is the summary for each of the documents:\n"""
            for doc_id in set(document_ids):
                loaded_content = self.load_documents_or_chapters("documentid", doc_id, "Summary", load_enhanced_text=False, only_if_connected_to="Document")
                if loaded_content is not None:
                    dossier_id = f", dossier_id:'{loaded_content['dossier_id']}'" if "dossier_id" in loaded_content and loaded_content['dossier_id'] else ""
                    result += f""" - document id:'{loaded_content["id"]}'{dossier_id}, title:`{loaded_content['title']}`, summary: `{loaded_content['text']}`\n"""
            result += f"""\n</DOCUMENT_SUMMARIES>\n\n"""
        
        return result
    
    def _get_loaded_documents_string(self, loaded_documents:List[str], max_loaded_documents_length: Optional[int] = None) -> str:
        """
        Generate a string with full text of loaded documents.
        
        If the total length exceeds max_loaded_documents_length, falls back to loading summaries.
        If summaries also exceed the limit, they are truncated.
        
        Args:
            loaded_documents: List of loaded document IDs to be processed.
                             Can be None, a list, or other types (will be handled appropriately).
            max_loaded_documents_length: Maximum total length allowed for loaded documents.
        
        Returns:
            A string with loaded documents content, or empty string if no documents.
        """
        result = ""
        if max_loaded_documents_length is None:
            max_loaded_documents_length = LlmManager.instance().get_max_loaded_document_length()
        
        # Add full text of loaded documents
        if loaded_documents is not None and isinstance(loaded_documents, list):
            # Handle both empty and non-empty lists
            if len(loaded_documents) > 0:
                # First, try to build the full documents string
                full_result = f"""\n\n<LOADED_DOCUMENTS>\nThese documents have been LOADED:"""
                document_contents = []
                
                for doc_or_chapter_id in set(loaded_documents):
                    doc_or_chapter_id_split = doc_or_chapter_id.split("-c")
                    chapter_id = doc_or_chapter_id_split[1] if len(doc_or_chapter_id_split) > 1 else None
                    node_label = "Document" if chapter_id is None else "Chapter"
                    loaded_content = self.load_documents_or_chapters("id", doc_or_chapter_id, node_label, load_enhanced_text=True)
                    if loaded_content is not None:
                        doc_string = f"""
<DOCUMENT_CONTENT id='{loaded_content["id"]}' title='{loaded_content['title']}'>
```{loaded_content['content_type']}
{loaded_content.get('enhanced_text') or loaded_content['text']}
```
</DOCUMENT_CONTENT>
"""
                        
                        # Add document categories if available
                        try:
                            documents_service = get_documents_service()
                            categories = documents_service.get_document_categories(loaded_content["id"])
                            if categories and len(categories) > 0:
                                doc_string += f"\n<DOCUMENT_CATEGORIES id='{loaded_content['id']}'>\n"
                                # Group categories by category_key
                                categories_by_key: Dict[str, List[str]] = {}
                                for cat in categories:
                                    key = cat.get('category_key')
                                    value = cat.get('category_value')
                                    if key:
                                        if key not in categories_by_key:
                                            categories_by_key[key] = []
                                        if value:
                                            categories_by_key[key].append(value)
                                
                                # Format categories as CATEGORY_NAME[=VALUE1[,VALUE2[,...]]]
                                for key in sorted(categories_by_key.keys()):
                                    values = categories_by_key[key]
                                    if values:
                                        category_line = f"{key}={','.join(values)}"
                                    else:
                                        category_line = key
                                    doc_string += f"{category_line}\n"
                                
                                doc_string += "</DOCUMENT_CATEGORIES>\n"

                        except Exception as e:
                            log(WARNING, f"Could not retrieve categories for document '{loaded_content['id']}': {e}")
                        
                        doc_string += "\n"
                        document_contents.append((loaded_content["id"], doc_string))
                
                # Check total length with full documents
                for _, doc_string in document_contents:
                    full_result += doc_string
                full_result += f"""\n</LOADED_DOCUMENTS>\n\n"""
                
                if len(full_result) <= max_loaded_documents_length:
                    # Full documents fit within limit
                    result = full_result
                else:
                    # Full documents exceed limit, fallback to summaries
                    log(WARNING, f"Loaded documents exceed maximum length ({len(full_result)} > {max_loaded_documents_length}). Falling back to summaries.")
                    
                    summary_result = f"""\n\n<LOADED_DOCUMENTS>\nThese documents have been LOADED (summaries only, full content exceeded {max_loaded_documents_length} characters):"""
                    document_ids = [doc_id for doc_id, _ in document_contents]
                    
                    for doc_id in document_ids:
                        # Extract base document ID (remove chapter suffix if present)
                        base_doc_id = doc_id.split("-c")[0]
                        loaded_summary = self.load_documents_or_chapters("documentid", base_doc_id, "Summary", load_enhanced_text=False, only_if_connected_to="Document")
                        if loaded_summary is not None:
                            dossier_id = f", dossier_id:'{loaded_summary['dossier_id']}'" if "dossier_id" in loaded_summary and loaded_summary['dossier_id'] else ""
                            summary_result += f"""\n - document id:'{loaded_summary["id"]}'{dossier_id}, title:`{loaded_summary['title']}`, summary: `{loaded_summary['text']}`"""
                    
                    summary_result += f"""\n</LOADED_DOCUMENTS>\n\n"""
                    
                    # Check if summaries also exceed the limit
                    if len(summary_result) <= max_loaded_documents_length:
                        result = summary_result
                    else:
                        # Summaries also exceed limit, truncate
                        log(WARNING, f"Document summaries also exceed maximum length ({len(summary_result)} > {max_loaded_documents_length}). Truncating summaries.")
                        truncated_result = summary_result[:max_loaded_documents_length]
                        truncated_result += f"\n{TRUNCATED_MESSAGE}\n</LOADED_DOCUMENTS>\n\n"
                        result = truncated_result
            else:
                # Empty list case: no documents to load
                result = f"""\n\n<LOADED_DOCUMENTS>\nNo documents have been loaded.\n</LOADED_DOCUMENTS>\n\n"""
        
        elif loaded_documents is not None:
            # loaded_documents is not None and not a list - this is an error
            log(WARNING, f"***WARNING***: loaded_documents is not a list: {loaded_documents}")
            result = f"""\n\n<LOADED_DOCUMENTS>\nNo documents have been loaded.\n</LOADED_DOCUMENTS>\n\n"""
        else:
            # loaded_documents is None
            result = f"""\n\n<LOADED_DOCUMENTS>\nNo documents have been loaded.\n</LOADED_DOCUMENTS>\n\n"""

        return result
    
    def _add_document_summaries_to_goal_summary(self, document_ids:List[str]) -> None:
        """
        Add information about referenced documents and their summaries to the goal summary.
        
        Args:
            document_ids: List of document IDs to be processed and added to the summary.
                         Can be None, a list, or other types (will be handled appropriately).
        """
        # Ensure goal_summary is initialized
        if self.goal_summary is None:
            self.goal_summary = ""
        
        # Add the generated string to goal_summary
        self.goal_summary += self._get_document_summaries(document_ids)

    def _add_loaded_documents_to_goal_summary(self, loaded_documents:List[str]) -> None:
        """
        Add full text of loaded documents to the goal summary.
        
        Args:
            loaded_documents: List of loaded document IDs to be processed and added to the summary.
                             Can be None, a list, or other types (will be handled appropriately).
        """
        if loaded_documents is None or not isinstance(loaded_documents, list) or len(loaded_documents) == 0:
            # No loaded documents to add
            return
        
        # Ensure goal_summary is initialized
        if self.goal_summary is None:
            self.goal_summary = ""
        
        # Add the generated string to goal_summary
        self.goal_summary += self._get_loaded_documents_string(loaded_documents)

    def get_reasoning_document(self, reasoning_document_id:str) -> str:
        """
        Generate a string with partial text of reasoning/scratchpad document. This method differs from get_loaded_documents_string(),
        in that the reasoning document (which is an XML file) is loaded from neo4j, then the XML is parsed into an object of class ReasoningDocument,
        then selected parts of it are extracted and generated back into an XML content.

        The selection happens as follows:
        all ReasoningDivisions are extracted, but their following elements are not extracted:
        - tasks are not extracted
        - task results are not extracted

        The reasons why they are not extracted are:
        1) task requests and their results have already been summarized in the "summary",
           no need of data duplication that fills the conversation buffer unnecessarily
        2) the presence of already processed data confuses the LLMs, which while processing other
           divisions, gets distracted from tasks and results of divisions not in focus, and
           mistakenly process the unnecessary information as well.
        
        Args:
            reasoning_document_id: ID of the reasoning document to be processed.
                                  Can be None (will be handled appropriately).
        
        Returns:
            A string with reasoning document content, or empty string if reasoning_document_id is None.
        """
        if reasoning_document_id is None:
            return ""
        
        # load reasoning document's content from neo4j into a ReasoningDocument object
        reasoning_document = ReasoningDocument.from_id(reasoning_document_id, is_template=False)
        if reasoning_document is None:
            log(WARNING, f"Failed to load reasoning document with ID: {reasoning_document_id}")
            return ""
        
        extracted_text = reasoning_document.to_xml(remove_information={
            "remove_tasks": True,
            "remove_results": True,
            "remove_prompt": True,
            "remove_keywords": True,
            "remove_instructions": True,
            "remove_issues": False,
            "remove_recommendations": False,
            "remove_summary": False,
            "master_language_only": True
        }, full_information_for_division_id=self.goal_focus_reasoning_division)

        # Build the result string
        result = f"""\n\n<REASONING_DOCUMENT>\nThis is the scratchpad reasoning document:
- ReasoningDocument id: '{reasoning_document.id()}', title: '{reasoning_document.title()}', content:
```{reasoning_document.content_type()}
{extracted_text}
```
</REASONING_DOCUMENT>

"""
        
        return result
    
    def _add_reasoning_document_to_goal_summary(self, reasoning_document_id:str) -> None:
        """
        Add partial text of reasoning/scratchpad document to the goal summary.
        
        Args:
            reasoning_document_id: ID of the reasoning document to be processed and added to the summary.
                                  Can be None (will be handled appropriately).
        """
        # Ensure goal_summary is initialized
        if self.goal_summary is None:
            self.goal_summary = ""
        
        # Add the generated string to goal_summary
        self.goal_summary += self.get_reasoning_document(reasoning_document_id)
        
        return

    def _build_canonical_rag_metadata_suffix(self, g: 'Goal', include_datasets: bool = False) -> str:
        """
        Build a format-only metadata suffix for a historical goal. Historical source references
        are intentionally omitted because reconstructing them from stored goal state can be
        incomplete or stale; the current response must still provide real references.

        Returns an empty string when the goal has neither RAG fragments nor datasets attached.
        """
        suffix: str = ""

        if g.goal_rag_fragments or g.get_dataset_ids():
            suffix = (
                "\noptional_rag_fragments_used: ..."
                "\noptional_proposed_dataset_name: ..."
            )

        if include_datasets and g.get_dataset_ids():
            dataset_ids = ";".join(
                str(dataset_id)
                for dataset_id in g.get_dataset_ids()
                if dataset_id is not None and str(dataset_id).strip()
            )
            if dataset_ids:
                suffix += f"\noptional_datasets_used_in_this_answer: {dataset_ids}"

        return suffix

    def build_messages(
        self,
        agent: Optional['Agent'] = None,
        purpose: str = None,
        system_prompt: str = None,
        include_focus: bool = False,
        include_rag: bool = False,
        exclude_tools: bool = False,
        include_images: bool = False,
        recurse: bool = False,
        debug_log: bool = False,
        system_prompt_label: Optional[str] = None,
    ) -> List[dict]:
        system_messages_and_focus: list = []
        hint_message: dict | None = None
        memory_message: dict | None = None
        _bm_perf: bool = not recurse  # only log PERF at the top-level call, not recursively
        if _bm_perf:
            _bm_agent = agent.agent_name if agent is not None else 'none'
            log(PERF, f"build_messages: start (goal={self.get_id_chain_as_str()}, agent={_bm_agent}, rag={include_rag})", perf_id="bm_start")
        
        # workaround for llama: the "tools" cannot appear in the messages, else it gives an error
        if agent is not None:
            # this is to avoid an OpenAI error, which is presented because they ALWAYS expect tool_calls to be immediately followed by a tool response
            exclude_tools = True
            
        if not recurse:
            # include system_prompt only once, at the top level
            if agent is not None and system_prompt is None:
                # prepare the system prompt, taking it from the agent's default system prompt and eventually appending an extention to it
                system_prompt = agent.agent_system_prompt

            if system_prompt is not None and len(system_prompt) > 0:
                system_message_tag = "SYSTEM_MESSAGE"
                if agent is not None:
                    prompt_label = system_prompt_label if system_prompt_label is not None else "DEFAULT"
                    prompt_id = f"{agent.agent_name}/{prompt_label}"
                    system_message_tag = f'SYSTEM_MESSAGE id="{html.escape(prompt_id, quote=True)}"'
                system_messages_and_focus = [
                    {"role": "system", "content": f"<{system_message_tag}>\n{system_prompt}\n</SYSTEM_MESSAGE>\n"},
                ]

            if agent is not None:
                model_reasoning_prompt = self._get_model_reasoning_prompt(agent=agent, purpose=purpose)
                if model_reasoning_prompt is not None:
                    model_reasoning_prompt = f"<REASONING>\n{model_reasoning_prompt}\n</REASONING>\n"
                    if system_messages_and_focus is not None and len(system_messages_and_focus) > 0:
                        system_messages_and_focus[0]["content"] = model_reasoning_prompt + "\n\n" + system_messages_and_focus[0]["content"]
                    else:
                        system_messages_and_focus = [ {"role": "system", "content": model_reasoning_prompt} ]

        if not recurse and agent is not None:
            pending_hints = get_pending_consolidation_hints_for_agent(agent.agent_name)
            if pending_hints:
                pending_hints_text = (
                    "<UNCONSOLIDATED_HINTS>\n"
                    f"{agent.agent_name} must follow all of these instructions until they are consolidated:\n"
                )
                for pending_hint in pending_hints:
                    pending_hints_text += f"<HINT>\n{pending_hint}\n</HINT>\n"
                pending_hints_text += "</UNCONSOLIDATED_HINTS>\n"
                system_messages_and_focus.append({"role": "system", "content": pending_hints_text})

        # determine which are the dependent goals on self
        other_dependent_goals = []
        if self.goal_dependent_upon is not None:
            for other_parallel_goal in self.goal_parallel_subgoals:
                if self != other_parallel_goal:
                    for goal_delta_index in other_parallel_goal.goal_dependent_upon:
                        previous_goal = other_parallel_goal.get_previous_goal(delta_from_self=goal_delta_index)
                        other_dependent_goals.append(previous_goal)

        # get messages from parent goals
        parent_messages = self.goal_parent.build_messages(
            agent=agent,
            purpose=purpose,
            system_prompt=system_prompt,
            include_focus=False,
            include_rag=include_rag,
            exclude_tools=exclude_tools,
            include_images=include_images,
            recurse=True,
            debug_log=debug_log,
            system_prompt_label=system_prompt_label,
        ) if self.goal_parent else []
        current_level_goals = self.goal_parent.goal_subgoals if self.goal_parent else []
        current_level_messages = []
        
        # process each goal in the current level
        for g in current_level_goals:
            is_cancelled_agent_user = (
                g.goal_status == GoalStatus.CANCELLED
                and g.goal_target_agent_name == "AgentUser"
            )
            if (g.goal_status != GoalStatus.COMPLETED) and (g.goal_status != GoalStatus.RUN_EVERY_TIME) and not is_cancelled_agent_user:
                if debug_log:
                    log(DEBUG, f"Goal {g.get_id_chain_as_str()} is not completed, skipping")
                continue

            # Skip debug goals (e.g., from /system_prompt, /lang commands)
            if g.goal_target == GoalTarget.DEBUG:
                if debug_log:
                    log(DEBUG, f"Goal {g.get_id_chain_as_str()} is a debug goal, skipping")
                continue

            if (self in self.goal_parallel_subgoals) and (g in self.goal_parallel_subgoals) and (g in other_dependent_goals) and (g != self):
                # skip over parallel subgoals, we don't know their status and we're not interested in them
                # because they are not dependent on the current goal and the current goal is not dependent on them
                if debug_log:
                    log(DEBUG, f"Goal {g.get_id_chain_as_str()} is a parallel subgoal, skipping. Details:\n - goal_parallel_subgoals: {[g.get_id_chain_as_str() for g in self.goal_parallel_subgoals]}\n - other_dependent_goals: {[dep.get_id_chain_as_str() for dep in other_dependent_goals]}\n - current_level_goals: {[g.get_id_chain_as_str() for g in current_level_goals]}\n - self: {self}\n - g: {g}")
                continue

            # get a message including the summary of the goal
            summary = g.get_summary()
            if summary is not None and (summary.lower() == 'none' or summary.lower() == 'null' or len(summary) == 0):
                summary = None
            if summary is not None:
                if (g.goal_target_agent is not None):
                    agent_module = g.goal_target_agent.__module__.split('.')[-1]
                    # AgentVerifier summaries (fact-check annotations) are internal; never expose them to other agents
                    if agent_module.startswith('AgentVerifier'):
                        continue
                    if agent_module.startswith('AgentAssistant'):
                        message = {"role":"assistant", "content":summary}
                    elif agent_module.startswith('AgentCustom'):
                        message = {"role":"assistant", "content":summary}
                    elif agent_module.startswith('AgentUser'):
                        user_role = "user" if g.goal_override_role is None else g.goal_override_role
                        user_content = summary
                        if user_content.strip().startswith("/use_plan "):
                            # the user has send a command such as: /use_plan plan-abc hello there please calculate 2+2 for me
                            # in this case, we want to extract only the part after the plan id, which is the actual user content,
                            # and ignore the command and the plan id, which are only meant to identify the plan
                            tokens = user_content.split(maxsplit=2)
                            user_content = tokens[2] if len(tokens) >= 3 else None
                            if not user_content:
                                continue
                        if is_cancelled_agent_user:
                            user_content = f"{user_content}\n\n[INFORMATION FROM THE SYSTEM: THIS PROMPT AND ITS EXECUTION WAS CANCELLED]"
                        if g.goal_override_json_output is not None:
                            user_content += f"""
NOTE: the AgentAssistant must use the following JSON structure for the output:
```
{g.goal_override_json_output}
```
You MUST enclose Json dictionary keys and values in double quotes, for example:
{{ "summary" : "{{ \"key\": \"value\" }}" }}
"""
                        if g.goal_override_structured_output is not None:
                            pass # TODO: not used
                        message = {"role":user_role, "content":user_content}
                        last_user_message = self._is_this_the_last_user_message(g, agent_name="AgentUser")
                        if last_user_message == True and agent is not None:
                            if _bm_perf:
                                log(PERF, f"build_messages: calling get_hints", perf_id="bm_hints_start", perf_since_id="bm_start")
                            hints:List[Dict[str, str]] = self.get_hints(agent_name=agent.agent_name, user_prompt=user_content, top_n=TOP_N_HINTS, min_score=MIN_HINT_SCORE)  # get the most relevant hints for the user prompt and the calling agent; redundant and conflictual hints are never injected, via the hints service default exclusion
                            if _bm_perf:
                                log(PERF, f"build_messages: get_hints done, {len(hints) if hints else 0} hints", perf_id="bm_hints_done", perf_since_id="bm_hints_start")
                            if hints and len(hints) > 0:
                                hints_text: str = f"<HINTS>\n{agent.agent_name} must follow any applicable of the following instructions:\n"
                                for hint in hints:
                                    hints_text += (
                                        f"<HINT id=\"{hint.get('id')}\">\n"
                                        f"<PROMPT>{hint.get('prompt')}</PROMPT>\n"
                                        f"<RECOMMENDATION>\n{hint.get('text')}\n"
                                        "</RECOMMENDATION>\n"
                                        "</HINT>\n"
                                    )
                                hints_text += "</HINTS>\n"
                                hint_message = {
                                    "role": "system",
                                    "content": hints_text
                                }
                                log(DEBUG, f"Injecting hint: {hint_message['content']}")
                            else:
                                hint_message = None 
                                log(DEBUG, "No hint to inject for the user message.")
                            # build the user memories message regardless of whether hints were found,
                            # so it is injected as its own system message right after the hints message
                            memory_message = self._build_memories_message()
                            if memory_message is not None:
                                log(DEBUG, f"Injecting user memories: {memory_message['content']}")
                            else:
                                log(DEBUG, "No user memories to inject for the user message.")
                    elif agent_module.startswith('AgentToolCaller'):
                        tool_role = "tool" if exclude_tools == False else "user"
                        message = {"role":tool_role, "content":summary, "tool_call_id":g.goal_target_agent.agent_tool_call_id}
                    elif agent_module.startswith('AgentRAG'):
                        message = {"role":"system", "content":summary}
                    elif agent_module.startswith('AgentPlanner'):
                        message = {"role":"system", "content":summary}
                    elif agent_module.startswith('AgentStrategist'):
                        message = {"role":"system", "content":summary}
                    elif agent_module.startswith('AgentToolPreparer'):
                        if g.goal_tool_call_message is not None and exclude_tools == False:
                            tool_calls_role: str = "system"
                            tool_calls = None

                            if isinstance(g.goal_tool_call_message, dict) and "tool_calls" in g.goal_tool_call_message:
                                tool_calls = []
                                tool_calls_results: list = g.goal_tool_call_message["tool_calls"]
                                tool_calls_role: str = g.goal_tool_call_message["role"] if "role" in g.goal_tool_call_message else "system"
                                
                                for tool_call in tool_calls_results:
                                    tool_calls.append({
                                        "id": tool_call["id"],
                                        "type": tool_call["type"],
                                        "function": {
                                            "name": tool_call["fname"],
                                            "arguments": json.dumps(tool_call["arguments"])
                                        }
                                    })
                            elif isinstance(g.goal_tool_call_message, list):
                                tool_calls = []
                                tool_calls_results: list = g.goal_tool_call_message
                                
                                for idx, tool_call in enumerate(tool_calls_results):
                                    tool_calls.append({
                                        "id": tool_call.get("id", f"call_{idx}"),
                                        "type": tool_call.get("type", "function"),
                                        "function": {
                                            "name": tool_call["fname"],
                                            "arguments": json.dumps(tool_call["arguments"])
                                        }
                                    })
                            elif isinstance(g.goal_tool_call_message, str):
                                tool_calls = str(g.goal_tool_call_message)
                            else:
                                log(WARNING, f"tool_call_message is not in expected format: {g.goal_tool_call_message}")
                                tool_calls = str(g.goal_tool_call_message)

                            if tool_calls is not None:
                                message = {
                                    "role": tool_calls_role,
                                    "tool_calls": tool_calls
                                }
                        else:
                            message = {"role":"assistant", "content":summary}
                    else:
                        message = {"role":"system", "content":summary}
                elif g.goal_target == GoalTarget.ONLY_SUBGOALS:
                    message = {"role":"assistant", "content":summary}
                else:
                    message = {"role":"system", "content":summary}

                if include_images and g.goal_images is not None and len(g.goal_images) > 0:
                    # prepare the message with the prompt and the images. Need to re-format the content to include the images
                    content = message["content"]
                    message["content"] = [
                        {"type": "text", "text": content}
                    ]
                    for image in g.goal_images:
                        base64_image = encode_image(image)
                        if base64_image is not None:
                            message["content"].append({
                                "type": "image_url", "image_url": { "url": f"data:image/png;base64,{base64_image}"
                            }})

                if g.goal_target_agent_name is not None:
                    if "role" in message and (message["role"] == "assistant" or message["role"] == "user"):
                        # if the agent is provided, we can add the agent's name to the message. This is only possible for the assistant or user role
                        message["name"] = g.goal_target_agent_name

                # Reconstruct history using the canonical wire format (fenced HTML + trailing
                # optional_ lines) instead of an ad hoc <MESSAGE_METADATA> tag, so historical
                # assistant answers look exactly like the format the LLM is asked to produce.
                # This prevents the model from imitating a legacy tag that isn't part of its
                # own output contract (see AgentAssistant.txt).
                is_assistant_role = message.get("role") == "assistant"
                rag_metadata_suffix = self._build_canonical_rag_metadata_suffix(g, is_assistant_role)
                if rag_metadata_suffix:
                    if isinstance(message["content"], str):
                        content = message["content"] or ""
                        if is_assistant_role:
                            content = f"```html\n{content}\n```"
                        message["content"] = content + rag_metadata_suffix

                    elif isinstance(message["content"], list) and len(message["content"]) > 0:
                        # Find the text item and append to it
                        for item in message["content"]:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text = item["text"] or ""
                                if is_assistant_role:
                                    text = f"```html\n{text}\n```"
                                item["text"] = text + rag_metadata_suffix
                                break

                    elif isinstance(message["content"], dict):
                        # Add metadata as a dictionary attribute
                        message["content"]["__metadata"] = {
                            "datasets": g.get_dataset_ids()
                        }

                # Inject the Hint message BEFORE the user message, so the LLM sees hints before the latest user prompt
                if hint_message is not None:
                    current_level_messages.append(hint_message)
                    hint_message = None  # reset: only inject once, immediately before the last user message

                # Inject the Memories message right after the hints message (or directly before the
                # last user message when there are no hints), as its own dedicated system message
                if memory_message is not None:
                    current_level_messages.append(memory_message)
                    memory_message = None  # reset: only inject once, immediately before the last user message

                current_level_messages.append(message)
            else:
                # summary is None
                if debug_log:
                    log(DEBUG, f"Goal {g.get_id_chain_as_str()} summary is None")

        # if there are goals with RAG fragments, produce a single consolidated message and insert it
        if include_rag:
            if _bm_perf:
                log(PERF, f"build_messages: calling identify_goals_with_rag_fragments", perf_id="bm_rag_start", perf_since_id="bm_start")
            rag_goals: List['Goal'] = self.identify_goals_with_rag_fragments_suitable_for_self() if include_rag else []
            if rag_goals is not None and len(rag_goals) > 0:
                log(DEBUG, f"build_messages: {self.get_id_chain_as_str()}: found {len(rag_goals)} rag goal(s): {[g.get_id_chain_as_str() + '(' + (g.goal_target_agent_name or '?') + ':' + str(len(g.goal_rag_fragments) if g.goal_rag_fragments else 0) + 'frags)' for g in rag_goals]}")
                
                # Initialize goal_debug if needed before generating RAG messages
                if self.goal_debug is None:
                    self.goal_debug = {}
                
                rag_message_text = self._generate_rag_messages(rag_goals)
                if rag_message_text is not None and len(rag_message_text) > 0:
                    rag_message = {"role": "system", "content": rag_message_text}
                    
                    # put consolidated RAG message before other current level messages
                    current_level_messages.insert(0, rag_message)
                else:
                    log(WARNING, f"build_messages: {self.get_id_chain_as_str()}: _generate_rag_messages returned empty/None despite {len(rag_goals)} rag goal(s) with fragments")
            else:
                # Log when no RAG goals were found but include_rag was True
                if not recurse:
                    _dep_upon = self.goal_dependent_upon
                    log(DEBUG, f"build_messages: {self.get_id_chain_as_str()}: no rag goals found (goal_dependent_upon={_dep_upon})")
            if _bm_perf:
                log(PERF, f"build_messages: RAG messages done, {len(rag_goals) if rag_goals else 0} rag goals", perf_id="bm_rag_done", perf_since_id="bm_rag_start")

        messages = system_messages_and_focus + parent_messages + current_level_messages
        
        # Inject plan system prompt if available
        if not recurse and self.goal_plan is not None:
            plan_system_prompt = self.goal_plan.get_system_prompt()
            if plan_system_prompt is not None and len(plan_system_prompt) > 0:
                plan_system_msg = {"role": "system", "content": f"<PLAN_SYSTEM_PROMPT>\n{plan_system_prompt}\n</PLAN_SYSTEM_PROMPT>\n"}
                # Find the index of the first user message (agent prompt)
                user_msg_idx = None
                for idx, msg in enumerate(messages):
                    if msg.get("role") == "user":
                        user_msg_idx = idx
                        break
                # Insert the plan system prompt after the user message, or at the beginning if no user message
                insert_idx = user_msg_idx + 1 if user_msg_idx is not None else 0
                messages.insert(insert_idx, plan_system_msg)
        
        if _bm_perf:
            log(PERF, f"build_messages: done, {len(messages)} messages total", perf_id="bm_done", perf_since_id="bm_start")
        return messages

    def count_messages_since_last_disambiguation(self) -> int:
        count = 99999

        # get the current goal at level one (the one with two ids)
        subgoals_level_two = self.goal_plan.plan_goal.goal_subgoals
        goal_id = self.get_id_chain()       # for example: [1, 4, 3, 1]
        goal_id_level_two = goal_id[:2]     # for example: [1, 4]

        if len(goal_id_level_two) < 2:
            return count
        
        # find the current goal at level two
        current_idx = goal_id_level_two[1]-1

        # walk backwards in the goals starting for example from [1, 3] until we find a disambiguation goal
        for i in range(current_idx, -1, -1):
            g = subgoals_level_two[i]
            if g.goal_proposed_followups is not None:
                count = current_idx - i
                break

        return count

    def fetch_datasets(self, dataset_ids: Optional[List[str]] = None) -> List[dict]:
        # Delegate to the plan-level implementation so dataset fetching is centralized
        if self.goal_plan is None:
            return []
        try:
            return self.goal_plan.fetch_datasets(dataset_ids)
        except Exception:
            return []

    def find_last_rag_fragments(self, last_goal:'Goal') -> 'Goal':
        """
        Resolve and hydrate RAG fragments stored on a goal into fully-populated dicts.

        Agent goals store their RAG results in `goal_rag_fragments`. The shape of each
        entry depends on the agent that produced it:

        - **Full fragment** (e.g. from AgentRAG): a dict that already contains at least
          ``fragmentid``, ``documentid``, and ``text``.  Used as-is.
        - **Stub fragment** (e.g. from AgentReranker): a dict (or bare string) that only
          carries a ``fragmentid`` with no ``documentid`` / ``text``.  AgentReranker
          trims the fragment list down to its top-N ranking results but only retains the
          IDs, relying on earlier goals to hold the full data.

        For stub fragments the method first walks the sibling/parent goal tree
        **once** to build a `{fragmentid → full_frag}` index, then resolves all
        stubs against that index in O(1) per stub.  Any stubs that remain unresolved
        after the tree search are resolved in a **batched Neo4j UNION query** that
        issues one sub-query per label (Chapter, Paragraph, Summary) so each
        sub-query can use its label index independently.

        The returned list always uses a normalised shape::

            {
                "documentid":    str,
                "fragmentid":    str,
                "text":          str,
                "ranking":       float | None,
                "ranking_text":  str | None,
            }

        Args:
            last_goal: The goal whose ``goal_rag_fragments`` should be resolved.
                       Typically a dependency of the current consolidator goal.

        Returns:
            A list of fully-hydrated fragment dicts ready for inclusion in the
            consolidator response.  Fragments that cannot be resolved (neither found
            in the goal tree nor in Neo4j) are silently dropped.

        Performance note:
            The backward tree walk is O(M) where M = total fragments in all sibling
            and ancestor goals.  Stubs not found in the tree have their documentid
            derived from the fragmentid naming convention — no Neo4j call at all.
            text is left as None and fetched lazily by _build_rag_message_lines for
            only the ~15-20 fragments that survive the downstream character-budget filter.
        """
        if last_goal.goal_rag_fragments is None:
            return []

        # ── Phase 1: separate full fragments from stubs ──────────────────────────
        # A "stub" is a fragment dict that has a fragmentid but no documentid/text.
        # AgentReranker produces stubs: it trims the fragment list to its top-N but
        # only keeps the IDs, expecting earlier goals to hold the full data.
        full_frags: dict = {}       # fragmentid → already-hydrated dict
        stub_ids: list  = []        # fragmentids still needing resolution
        stub_rankings: dict = {}    # fragmentid → (ranking, ranking_text) from the stub
        
        for frag in last_goal.goal_rag_fragments:
            frag_id = frag["fragmentid"] if isinstance(frag, dict) and "fragmentid" in frag else frag
            if isinstance(frag, dict) and "documentid" in frag:
                # full fragment — use as-is
                full_frags[frag_id] = frag
            else:
                # stub — record ranking metadata carried on the stub (if any)
                stub_ids.append(frag_id)
                if isinstance(frag, dict):
                    stub_rankings[frag_id] = (frag.get("ranking"), frag.get("ranking_text"))

        # ── Phase 2: resolve stubs via backward sibling-tree search ──────────────
        # Walk the sibling/parent goal tree ONCE to build a complete index of every
        # full fragment (one with a "documentid") reachable from the current node.
        # This is O(M) where M = total fragments in the tree, rather than O(N × M)
        # for the naive "re-walk the tree for each stub" approach that becomes very
        # slow in long conversations with many turns and many fragments per turn.
        tree_index: dict = {}       # fragmentid → first full-fragment dict found
        search_node = self
        while search_node is not None and search_node.goal_parent is not None:
            for sibling in reversed(search_node.goal_parent.goal_subgoals):
                if sibling.goal_rag_fragments is not None:
                    for sibling_frag in sibling.goal_rag_fragments:
                        if (isinstance(sibling_frag, dict)
                                and "documentid" in sibling_frag
                                and "fragmentid" in sibling_frag):
                            fid = sibling_frag["fragmentid"]
                            if fid not in tree_index:
                                tree_index[fid] = sibling_frag
            search_node = search_node.goal_parent

        resolved: dict = {}         # fragmentid → hydrated dict (from tree search)
        still_unresolved: list = [] # stub_ids not found in the tree

        if stub_ids:
            log(DEBUG, f"find_last_rag_fragments: resolving {len(stub_ids)} stubs; "
                       f"tree_index has {len(tree_index)} full frags. "
                       f"Sample stub id: {stub_ids[0]!r}")
        for frag_id in stub_ids:
            hit = tree_index.get(frag_id)
            if hit:
                resolved[frag_id] = hit
            else:
                still_unresolved.append(frag_id)

        # ── Phase 3: derive documentid from fragmentid for still-unresolved stubs ─
        # Rather than querying Neo4j for text that will mostly be discarded by the
        # downstream character-budget filter (AgentAssistant can consume only ~15-20
        # fragments within its token budget), documentid is derived in-memory from
        # the fragmentid naming convention via DocumentsInterface.parse_fragment_id(),
        # which handles both the plain and 'doc://' URI forms.
        # text is left as None; _build_rag_message_lines will batch-fetch it for
        # only those fragments that make it past the budget filter.
        for frag_id in still_unresolved:
            parsed = DocumentsInterface.parse_fragment_id(frag_id)
            derived_doc_id = parsed["document_id"] or frag_id
            resolved[frag_id] = {
                "fragmentid": frag_id,
                "documentid": derived_doc_id,
                "text":       None,   # fetched lazily by _build_rag_message_lines
            }

        # ── Phase 4: build the final normalised output list ───────────────────────
        # Iterate in original fragment order so ranking ordering is preserved.
        # text may be None for stubs resolved only by documentid derivation;
        # _build_rag_message_lines will batch-fetch those on demand.
        last_rag_fragments = []
        for frag in last_goal.goal_rag_fragments:
            frag_id = frag["fragmentid"] if isinstance(frag, dict) and "fragmentid" in frag else frag
            source = full_frags.get(frag_id) or resolved.get(frag_id)
            if source:
                ranking, ranking_text = stub_rankings.get(frag_id, (None, None))
                last_rag_fragments.append({
                    "documentid":   source["documentid"],
                    "fragmentid":   frag_id,
                    "text":         source.get("text"),   # may be None; see Phase 3 above
                    "ranking":      source.get("ranking") if source.get("ranking") is not None else ranking,
                    "ranking_text": source.get("ranking_text") if source.get("ranking_text") is not None else ranking_text,
                })

        return last_rag_fragments

    def pause(self) -> None:
        message_written = False
        while is_paused[0]:
            if not message_written:
                log(INFO, f"***PAUSED***: Pausing execution until resumed...")
            time.sleep(1)

        if message_written:
            log(INFO, f"***RESUMED***: Resuming execution...")

    def _run_parallel(self, subgoal:'Goal') -> dict:
        subgoal_response = None

        target_agent_name = subgoal.goal_target_agent.agent_name if subgoal.goal_target_agent else ""
        log(DEBUG, f"Running goal {subgoal.goal_target}{target_agent_name} in parallel...")

        try:
            subgoal_response = subgoal.run()

        except GoalCancelledException as exc:
            subgoal.set_goal_status(GoalStatus.CANCELLED, recursive=True, completion_reason=str(exc))
            subgoal_response = {
                "success": False,
                "cancelled": True,
                "error_message": str(exc),
            }

        except Exception as exc:
            log(ERROR, f"Goal {subgoal.goal_target}{target_agent_name} generated an exception: {exc}\n{traceback.format_exc()}")

        log(DEBUG, f"...completed running goal {subgoal.goal_target}{target_agent_name} in parallel.")

        if subgoal_response is None:
            log(ERROR, f"Goal {subgoal.goal_target} {target_agent_name} returned a None response.")

        return subgoal_response

    def run_async(self) -> None:
        """Execute this goal asynchronously in a background thread.

        Called from run() when goal_async_status == ASYNC_REQUESTED.
        Only Agent goals support async execution. Dispatches directly to run_agent()
        (bypassing run() to avoid re-entering the status-guard logic), then updates
        goal_async_status and calls send_plan_to_blazor() so the C# UI reflects the outcome.
        Status is already set to ASYNC_IN_PROGRESS by run() before spawning this thread.
        """
        _id = self.get_id_chain_as_str()
        log(INFO, f"Goal {_id} '{self.goal_title}': async execution starting")
        run_start_time = time.perf_counter()
        try:
            if self.goal_target != GoalTarget.AGENT:
                log(ERROR, f"Goal {_id}: async execution is only supported for AGENT goals, but target is {self.goal_target}")
                self.set_async_status(GoalAsyncStatus.ASYNC_FAILED)
            else:
                self.run_agent()
                self.set_perf_run_ms(round((time.perf_counter() - run_start_time) * 1000, 2))
                self.set_async_status(GoalAsyncStatus.ASYNC_COMPLETE)
                log(INFO, f"Goal {_id} '{self.goal_title}': async execution completed successfully in {self.goal_perf_run_duration_ms:.0f} ms")

        except Exception as exc:
            self.set_async_status(GoalAsyncStatus.ASYNC_FAILED)
            log(ERROR, f"Goal {_id} '{self.goal_title}': async execution failed with exception: {exc}\n{traceback.format_exc()}")

        finally:
            send_plan_to_blazor(self.goal_plan, self.get_id_chain())

    def run(self) -> dict:
        if is_stopped():
            return None
        
        fatal = False
        response = None
        subgoal_response = None
        current_goals_level_already_executed = []
        run_start_time: Optional[float] = None
        parallelism_level = PARALLEL_GOALS_MAX
        if PARALLEL_GOALS_ENABLE == False or app_config.testing:
            parallelism_level = 1

        if self.goal_status not in [GoalStatus.COMPLETED, GoalStatus.CANCELLED] and self.get_dependencies_status() == GoalStatus.COMPLETED:
            log(DEBUG, f"'{'.'.join(str(i) for i in self.get_id_chain())} {self.goal_title}' Goal is running...")
            run_start_time = time.perf_counter()

            if not self.is_children_completed() and not is_stopped():
                # run all subgoals
                log(DEBUG, f"{self.goal_status} --> \033[1;34mGoalStatus.CHILDREN_IN_PROGRESS\033[0m")

                self.set_goal_status(GoalStatus.CHILDREN_IN_PROGRESS)
                restart_current_goals_level = True
                repeat_contextualizer = False
                while restart_current_goals_level and not fatal and not is_stopped():
                    # eventually pause execution
                    self.pause()

                    # execute all subgoals, and repeat in case a restart is needed
                    children_completed = True

                    # gather all subgoals that that can be run in parallel
                    parallel_subgoals = []
                    for subgoal in self.goal_subgoals:
                        force_schedule = False
                        if repeat_contextualizer:
                            # we got the command to repeat the contextualizer goal, let's do it
                            self.repeat_contextualizer()
                            repeat_contextualizer = False

                        if (
                            subgoal.goal_status not in [GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.CANCELLED]
                            and subgoal.get_dependencies_status() == GoalStatus.CANCELLED
                        ):
                            # a goal whose dependency was cancelled can never become schedulable again;
                            # cancel it too, otherwise it stays NOT_STARTED forever and the plan never converges
                            subgoal.set_goal_status(GoalStatus.CANCELLED, completion_reason="Dependency cancelled")
                            continue

                        if force_schedule or (
                            subgoal.goal_status not in [GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.CANCELLED]
                            and subgoal.get_dependencies_status() == GoalStatus.COMPLETED
                            and (subgoal not in current_goals_level_already_executed or subgoal.goal_status == GoalStatus.CHILDREN_IN_PROGRESS)
                        ):
                            parallel_subgoals.append(subgoal)
                            current_goals_level_already_executed.append(subgoal)
                            if subgoal.goal_target == GoalTarget.CONTEXTUALIZER:
                                # run the contextualizer goal alone, break the loop
                                break
                            if len(parallel_subgoals) >= parallelism_level:
                                # reached the maximum number of parallel subgoals
                                break

                    # run all subgoals in parallel
                    if len(parallel_subgoals) > 0:
                        # mark down that each of the parallel subgoals will be executed in parallel, and is competing with the others
                        for subgoal in parallel_subgoals:
                            subgoal.goal_parallel_subgoals = parallel_subgoals

                        log(DEBUG, f"Identified {len(parallel_subgoals)} subgoals to be executed in parallel:")
                        for subgoal in parallel_subgoals:
                            log(DEBUG, f"--> {subgoal.goal_target} {subgoal.goal_target_agent.agent_name if subgoal.goal_target_agent else ''}")

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            # execute the identified parallelizable subgoals in parallel
                            future_to_subgoal = {executor.submit(self._run_parallel, subgoal): subgoal for subgoal in parallel_subgoals}

                            # wait for all subgoals to complete
                            results = []
                            for future in future_to_subgoal:
                                result = future.result()  # This will block until the future is done
                                results.append((future, result))

                            # examine the results in the same order as the subgoals
                            for future, result in results:
                                subgoal = future_to_subgoal[future]

                                try:
                                    subgoal_response = result
                                    if subgoal_response is not None:
                                        if "internal_thought" in subgoal_response and subgoal_response["internal_thought"] is not None and len(subgoal_response["internal_thought"]) > 0:
                                            self.goal_internal_thought = subgoal_response["internal_thought"]
                                        if "repeat_contextualizer" in subgoal_response and subgoal_response["repeat_contextualizer"]:
                                            # repeat the contextualizer goal, but only once
                                            repeat_contextualizer = True
                                        if "error" in subgoal_response and subgoal_response["error"]:
                                            if response is None:
                                                response = {}
                                            response["error"] = subgoal_response["error"]
                                            response["error_message"] = subgoal_response["error_message"] if "error_message" in subgoal_response else None
                                        if "fatal" in subgoal_response and subgoal_response["fatal"]:
                                            restart_current_goals_level = False
                                            children_completed = False
                                            fatal = True
                                            break
                                        if "success" in subgoal_response and subgoal_response["success"] == False:
                                            restart_current_goals_level = False
                                            children_completed = True
                                            break
                                        if subgoal.goal_status == GoalStatus.CHILDREN_IN_PROGRESS:
                                            # This AGENT goal spawned children and needs to be re-scheduled to run them.
                                            # Keep restart_current_goals_level=True so the while loop continues and
                                            # re-runs this goal (the scheduling guard allows CHILDREN_IN_PROGRESS goals
                                            # to bypass the current_goals_level_already_executed check).
                                            children_completed = False
                                        elif ((subgoal.goal_status != GoalStatus.COMPLETED) and (subgoal.goal_status != GoalStatus.RUN_EVERY_TIME)) or "abort" in subgoal_response and subgoal_response["abort"]:
                                            restart_current_goals_level = False
                                            children_completed = False
                                            break
                                        elif "restart_current_goals_level" in subgoal_response and subgoal_response["restart_current_goals_level"]:
                                            current_goals_level_already_executed.clear()
                                            children_completed = False
                                            break
                                    else:
                                        log(ERROR, f"Goal {subgoal.goal_title}: the subgoal_response is None.")
                                        restart_current_goals_level = False
                                        children_completed = False
                                        fatal = True
                                        break
                                except Exception as exc:
                                    log(ERROR, f"Goal {subgoal.goal_title} generated an exception: {exc}\n{traceback.format_exc()}")
                                    restart_current_goals_level = False
                                    children_completed = False
                                    fatal = True
                                    break
                    else:
                        log(DEBUG, f"No subgoals identified to be executed in parallel.")
                        restart_current_goals_level = False

                if children_completed:
                    # Propagate last child's summary for ONLY_SUBGOALS goals, and also for AGENT
                    # goals that dynamically injected children (long-generation pipeline).
                    agent_with_subgoals = (self.goal_target == GoalTarget.AGENT and self.goal_subgoals)
                    if self.goal_target != GoalTarget.ONLY_SUBGOALS and not agent_with_subgoals:
                        log(DEBUG, f"{self.goal_status} --> \033[1;34mGoalStatus.IN_PROGRESS\033[0m")
                    elif self.goal_subgoals is not None and len(self.goal_subgoals) > 0 and subgoal_response is not None:
                        # for goals with only subgoals: take over the last child's summaries into the parent's goal.
                        # For AGENT goals that injected children (long-generation), only update goal_summary when the
                        # child returns a real summary — fillers return summary=None intentionally, and we must not wipe
                        # the skeleton before Rewriter has a chance to read it and replace it with assembled text.
                        _child_summary = subgoal_response["summary"] if "summary" in subgoal_response else None
                        if not agent_with_subgoals or _child_summary is not None:
                            self.goal_summary = _child_summary
                        self.goal_summary_pretty = subgoal_response["pretty"] if "pretty" in subgoal_response else None
                        self.goal_summary_suppress = subgoal_response["summary_suppress"] if "summary_suppress" in subgoal_response else False
                        self.goal_rag_fragments = subgoal_response["rag_fragments"] if "rag_fragments" in subgoal_response else None
                        self.goal_rag_toomanyfragments = subgoal_response["rag_toomanyfragments"] if "rag_toomanyfragments" in subgoal_response else None
                        self.goal_rag_docids = subgoal_response["document_ids"] if "document_ids" in subgoal_response else None
                        self.goal_attributes = subgoal_response["attributes"] if "attributes" in subgoal_response else None
                        if "datasets" in subgoal_response:
                            # normalize and set goal datasets; this will also update the plan
                            self.set_datasets(subgoal_response["datasets"])
                        if "internal_thought" in subgoal_response and subgoal_response["internal_thought"] is not None and len(subgoal_response["internal_thought"]) > 0:
                            self.goal_internal_thought = subgoal_response["internal_thought"]
                        if "optional_disambiguating_answer_proposals" in subgoal_response:
                            self.goal_proposed_followups = subgoal_response["optional_disambiguating_answer_proposals"]
                        if "instantiated_plan" in subgoal_response and subgoal_response["instantiated_plan"] is not None:
                            self.goal_instantiated_plan = subgoal_response["instantiated_plan"]
                        if "html" in subgoal_response:
                            self.goal_html_summary = subgoal_response["html"]
                        if "blazor_component" in subgoal_response:
                            self.goal_blazor_component = subgoal_response["blazor_component"]
                        # If blazor_component was not forwarded by the last subgoal (e.g. the
                        # Consolidator doesn't echo it), scan all completed subgoals in reverse
                        # order (starting from the latest child).
                        if self.goal_blazor_component is None:
                            for sg in reversed(self.goal_subgoals):
                                if sg.goal_blazor_component:
                                    self.goal_blazor_component = sg.goal_blazor_component
                                    # Also carry that child's summary so C# has the data payload
                                    if not self.goal_summary and sg.goal_summary:
                                        self.goal_summary = sg.goal_summary
                                    break
                    self.set_goal_status(GoalStatus.IN_PROGRESS)
                    send_plan_to_blazor(self.goal_plan, self.get_id_chain())

            # update status
            if self.goal_status == GoalStatus.NOT_STARTED:
                log(DEBUG, f"{self.goal_status} --> \033[1;34mGoalStatus.IN_PROGRESS\033[0m")
                self.set_goal_status(GoalStatus.IN_PROGRESS)
            elif self.goal_target != GoalTarget.ONLY_SUBGOALS:
                status_color = self.get_status_color()
                log(DEBUG, f"Goal Status: {status_color}{self.goal_status}\033[0m")

            if not fatal:
                # check if goal requests async (fire-and-forget) execution
                if self.get_async_status() == GoalAsyncStatus.ASYNC_REQUESTED:
                    if self.goal_target != GoalTarget.AGENT:
                        log(WARNING, f"Goal {self.get_id_chain_as_str()}: ASYNC_REQUESTED set but goal target is {self.goal_target}; async is only supported for AGENT goals")
                        self.set_async_status(GoalAsyncStatus.SYNC)
                    else:
                        log(INFO, f"Goal {self.get_id_chain_as_str()} '{self.goal_title}': spawning async background thread")
                        self.set_async_status(GoalAsyncStatus.ASYNC_IN_PROGRESS)
                        self.set_goal_status(GoalStatus.COMPLETED)
                        async_thread = threading.Thread(target=self.run_async, daemon=True, name=f"goal-async-{self.get_id_chain_as_str()}")
                        self.goal_async_thread = async_thread
                        async_thread.start()
                # implement the goal (synchronous path)
                if self.goal_target == GoalTarget.AGENT:
                    if self.get_async_status() != GoalAsyncStatus.ASYNC_REQUESTED and self.get_async_status() != GoalAsyncStatus.ASYNC_IN_PROGRESS:
                        if not self.goal_subgoals:
                            # Normal case: agent has not yet run (no children). Run it now.
                            response = self.run_agent()
                        # else: agent already ran and injected children; they just completed via the
                        # children loop above. Summary was propagated in the children_completed block.
                        # Do not call run_agent() again.
                    # else: async goal running in background thread — skip synchronous execution
                elif self.goal_target == GoalTarget.CONTEXTUALIZER:
                    response = self.run_contextualizer()
                elif self.goal_target == GoalTarget.NOOP:
                    response = self._run_noop()
                elif self.goal_target == GoalTarget.CONSOLIDATOR:
                    response = self._run_consolidator()
                elif self.goal_target == GoalTarget.ONLY_SUBGOALS:
                    # only subgoals -- no action needed
                    pass
                elif self.goal_target == GoalTarget.DEBUG:
                    # DEBUG goal -- no action needed
                    pass
                else:
                    log(ERROR, f"\033[1;31m***ERROR*** UNDEFINED Goal Target: {self.goal_target} (async_status={self.get_async_status()}, title={self.goal_title})\033[0m")

                # update status
                if self.goal_status == GoalStatus.IN_PROGRESS:
                    if self.is_children_completed():
                        status_color = get_status_color_for_status(GoalStatus.COMPLETED)
                        log(DEBUG, f"{self.goal_status} --> {status_color}GoalStatus.COMPLETED\033[0m")
                        self.set_goal_status(GoalStatus.COMPLETED)

                log(DEBUG, f"...goal is completed: '{'.'.join(str(i) for i in self.get_id_chain())} {self.goal_title}'")
            else:
                log(FATAL, f"\033[1;31m*** FATAL ***: Fatal error occurred, stopping the plan execution\033[0m")

            if run_start_time is not None:
                if self.get_async_status() != GoalAsyncStatus.ASYNC_IN_PROGRESS:
                    self.set_perf_run_ms(round((time.perf_counter() - run_start_time) * 1000, 2))
                    log(DEBUG, f"Goal {self.get_id_chain_as_str()} run duration: {self.goal_perf_run_duration_ms} ms")
                # For async goals, timing is set by run_async() when the background thread completes

        if response is None:
            response = {}
        if ("success" not in response) and ((self.goal_status == GoalStatus.COMPLETED) or (self.goal_status == GoalStatus.RUN_EVERY_TIME)):
            response["success"] = True
        if fatal:
            response["fatal"] = fatal

        # Propagate goal_summary and blazor_component for ONLY_SUBGOALS goals, and also for AGENT
        # goals that ran and then executed dynamically injected children (long-generation pipeline).
        _propagate_as_subgoals = self.goal_target == GoalTarget.ONLY_SUBGOALS or (self.goal_target == GoalTarget.AGENT and self.goal_subgoals)
        if _propagate_as_subgoals and self.goal_summary and "summary" not in response:
            response["summary"] = self.goal_summary
        if _propagate_as_subgoals and self.goal_blazor_component and "blazor_component" not in response:
            response["blazor_component"] = self.goal_blazor_component

        return response
    
    def run_agent(self) -> dict:
        self.goal_target_agent.raise_if_plan_stopped()

        # send the plan to blazor, so that the status "in progress..." is displayed in the UI as a ThinkingMessage
        send_plan_to_blazor(self.goal_plan, self.get_id_chain())

        # run the agent
        _agent_name: str = self.goal_target_agent.agent_name or "UNNAMED_AGENT"
        response: Optional[dict] = None
        try:
            log(INFO, f"'{_agent_name}': Agent starting, goal: {self.get_id_chain_as_str()}")
            response = self.goal_target_agent.run(self)
            self.goal_target_agent.raise_if_plan_stopped()

        except GoalCancelledException:
            raise
        
        except Exception as _e:
            log(ERROR, f"'{_agent_name}': Agent raised an unhandled exception: {_e}\n{traceback.format_exc()}")
            response = { "success": False, "error": True, "error_message": str(_e) }

        finally:
            _frags_n: int = len(response.get('rag_fragments') or []) if response else 0
            _ds: Optional[list] = response.get('datasets') if response else None
            _extra: str = f", frags={_frags_n}" if _frags_n else ""
            _extra += f", datasets={_ds}" if _ds else ""
            if _agent_name == "AgentReranker" and _ds:
                log(DEBUG, f"'{_agent_name}': Detailed datasets info: {json.dumps(_ds, indent=2, default=str)[:500]}")
            log(INFO, f"'{_agent_name}': Agent done, success={response.get('success') if response else None}{_extra}")

        # if the agent didn't run, or responded with None, we assume success
        if response is None:
            log(DEBUG, f"Agent {self.goal_target_agent.agent_name} did responded with None: we assume success and continue.")
            response = { "success": True }

        # fix the summary, sometimes it's "None" and that's not good
        if response is not None and "summary" in response and response["summary"] is not None:
            if response["summary"] == "None" or len(response["summary"]) == 0:
                response["summary"] = None

        if response is not None:
            # update the goal's summary with the agent's response
            self.goal_summary = response["summary"] if "summary" in response else None
            self.goal_summary_pretty = response["pretty"] if "pretty" in response else self.goal_summary_pretty
            if "goal_planner_outputs" in response:
                self.set_goal_planner_outputs(response["goal_planner_outputs"])
            self.goal_agent_did_nothing = bool(response.get("agent_did_nothing", False))
            self.goal_summary_suppress = response["summary_suppress"] if "summary_suppress" in response else self.goal_summary_suppress
            self.goal_proposed_followups = response["optional_disambiguating_answer_proposals"] if "optional_disambiguating_answer_proposals" in response else None
            if "datasets" in response:
                # set_datasets handles errors internally
                self.set_datasets(response["datasets"])
            self.goal_rag_fragments = response["rag_fragments"] if "rag_fragments" in response else None
            self.goal_rag_toomanyfragments = response["rag_toomanyfragments"] if "rag_toomanyfragments" in response else None
            self.goal_rag_docids = response["document_ids"] if "document_ids" in response else None
            self.goal_attributes = response["attributes"] if "attributes" in response else None

            # update the goal's document RAG with the agent's response
            if "document_ids" in response and response["document_ids"] is not None and len(response["document_ids"]) > 0:
                # some document ids were identified in the response. Must add them to the plan's document focus
                self.goal_plan.add_documents(response["document_ids"])
            if "rag_toomanyfragments" in response:
                self.goal_rag_toomanyfragments = response["rag_toomanyfragments"]
            if "internal_thought" in response and response["internal_thought"] is not None:
                self.goal_internal_thought = response["internal_thought"]
            if "instantiated_plan" in response and response["instantiated_plan"] is not None:
                self.goal_instantiated_plan = response["instantiated_plan"]
            if 'html' in response:
                self.goal_html_summary = response['html']
            if 'blazor_component' in response:
                self.goal_blazor_component = response['blazor_component']
            if 'error' in response and response['error']:
                self.goal_error = response['error']
                self.goal_error_message = response['error_message'] if 'error_message' in response else None

        if (response is None) or (isinstance(response, dict) and "success" in response and response["success"]):
            # check if there are subgoals and if they are all completed
            if self.goal_subgoals is not None and len(self.goal_subgoals) > 0:
                all_subgoals_completed = True
                for subgoal in self.goal_subgoals:
                    if subgoal.goal_status != GoalStatus.COMPLETED:
                        all_subgoals_completed = False
                        break
                if all_subgoals_completed:
                    self.set_goal_status(GoalStatus.COMPLETED)
                    log(DEBUG, f"...\033[1;32mGoalStatus.COMPLETED\033[0m: {self.goal_title}")
                else:
                    # some subgoals are not completed yet; keep goal_target=AGENT so the scheduler
                    # re-runs this goal and the children loop executes the injected subgoals
                    self.set_goal_status(GoalStatus.CHILDREN_IN_PROGRESS)
            else:
                self.set_goal_status(GoalStatus.COMPLETED)
                log(DEBUG, f"...\033[1;32mGoalStatus.COMPLETED\033[0m: {self.goal_title}")
        else:
            response["success"] = False
            self.set_goal_status(GoalStatus.FAILED)
            log(WARNING, f"...\033[1;31mGoalStatus.FAILED\033[0m: {self.goal_title}. Error: {response.get('error_message', 'No error_message provided')}")
        
        if self.goal_repeat_on_error == True and "error" in response and response["error"] == True:
            # the agent goal failed, but it can be retried
            self._insert_current_goal_and_dependencies_in_case_of_error()
            response["restart_current_goals_level"] = True
            log(DEBUG, f"The 'restart_on_error'' condition is triggered: the goal and its dependencies will be injected and retried.")

        return response

    def repeat_contextualizer(self) -> dict:
        response = { "success": True }

        try:
            log(DEBUG, f"Repeating Contextualizer...")

            plan = self.goal_plan

            # identify the contextualizer goal. It's on first level of the plan's goal subgoals. There is only one in the whole plan
            contextualizer_goal = None
            for goal in plan.plan_goal.goal_subgoals:
                if goal.goal_target == GoalTarget.CONTEXTUALIZER:
                    contextualizer_goal = goal
                    break

            if contextualizer_goal is not None:
                # run the contextualizer goal
                response = contextualizer_goal.run()
            
            response = { "success": True }
        except Exception as e:
            log(ERROR, f"Contextualizer generated an exception: {e}\n{traceback.format_exc()}")
            response["success"] = False
        
        return response
    
    def run_contextualizer(self) -> dict:
        try:
            log(DEBUG, f"Running Contextualizer...")

            # sanity check
            if self is None or self.goal_plan is None:
                return { "success": False, "fatal": True, "error": "Goal or Plan is None" }

            # if datasets have not yet been retrieved, fetch them
            self.goal_plan.plan_datasets = self.fetch_datasets()

            response = { "success": True }
            # initialize the goal summary if not already done
            self.goal_summary = ""

            if self.goal_plan.plan_focus is not None:
                self.goal_summary += f"""# Conversation Context\nThe context of the conversation is:
```
{self.goal_plan.plan_focus}
```"""

            # add date and timezone information, optionally time if needed
            try:
                tz_name = tzlocal.get_localzone_name()
            except Exception:
                tz_name = "UTC"
            current_datetime = datetime.now(pytz.timezone(tz_name))
            if self.goal_plan.plan_include_time == True:
                # add date, time and timezone information. This is needed for time-sensitive tasks, and its regulated by the plan flag
                self.goal_summary += f"""
<DATE_TIME>
Current date: {current_datetime.strftime('%Y-%m-%d')}, current time: {current_datetime.strftime('%H:%M:%S')}, timezone: '{tz_name}'.
</DATE_TIME>
"""
            else:
                # add month and year (no day) to preserve LLM cache longevity. Cache stays valid for entire month instead of changing daily.
                month_year = current_datetime.strftime('%B %Y')  # e.g., "January 2026"
                self.goal_summary += f"""
<DATE_TIME>
Current month and year: {month_year}, timezone: '{tz_name}'.
</DATE_TIME>
"""

            # add information about language
            current_language = self.goal_plan.get_language_in_use()
            self.goal_summary += f"""<LANGUAGE>\nThe language in use: '{current_language}'.\n</LANGUAGE>\n\n"""

            dossier_id = self.goal_plan.get_dossier_id()
            session_dossier_id = self.goal_plan.get_session_dossier_id()
            # dossier information
            dossier_content = ""
            if dossier_id is not None:
                dossier_content += self._get_dossier_information(dossier_id, include_summaries=True, include_text=False)
            if session_dossier_id is not None:
                dossier_content += self._get_dossier_information(session_dossier_id, include_summaries=False, include_text=True, is_session_dossier=True)
            
            if dossier_content:
                self.goal_summary += f"""<DOSSIERS>
The following dossiers are associated with this plan.
IMPORTANT: When the user refers to "this dossier", they always mean the active plan dossier (the one WITHOUT is_session_dossier="true"). The session dossier (marked with is_session_dossier="true") is a technical implementation detail invisible to the user — NEVER interpret "this dossier" as the session dossier.
The documents contained in the dossier ARE NOT LOADED and are only presented with summaries.
If you need to access the full text of the documents, the AgentPlanner needs to explicitly use RAG (recommend an appropriate plan),
and they will be added to the context with their full text in the next steps.
"""
                self.goal_summary += dossier_content
                self.goal_summary += f"""\n</DOSSIERS>\n\n"""
            elif dossier_id is not None:
                self.goal_summary += f"""\n\n<DOSSIERS>\nNo dossier is associated with this plan.\n</DOSSIERS>\n\n"""
            
            log(DEBUG, f"Contextualizer initial goal summary: {self.goal_summary}")
            # add information about referenced documents and their summaries
            self._add_document_summaries_to_goal_summary(self.goal_plan.plan_document_ids)

            # add full text of loaded documents, templates and reasoning document (if available)
            # (exclude the reasoning document, which is added specifically in the next code lines)
            loaded_documents = self.goal_plan.plan_loaded_documents
            # filter out the reasoning document id (and any None entries) so we pass a clean list
            if isinstance(loaded_documents, list):
                loaded_documents = [d for d in loaded_documents if d is not None and d != self.goal_plan.plan_reasoning_document_id]
            else:
                loaded_documents = []

            # remove the dossier's documents from the loaded documents, else the context becomes far too long
            dossier_id_for_exclusion = self.goal_plan.get_dossier_id()
            if dossier_id_for_exclusion and loaded_documents:
                docs_service: DocumentsInterface = get_documents_service()
                dossier_doc_ids: set = set(docs_service.get_document_ids_for_dossier(dossier_id_for_exclusion))
                loaded_documents_exclusing_dossier: list = [d for d in loaded_documents if d not in dossier_doc_ids]
            else:
                loaded_documents_exclusing_dossier: list = loaded_documents
            self._add_loaded_documents_to_goal_summary(loaded_documents_exclusing_dossier)

            # add reasoning document
            if self.goal_plan.plan_reasoning_document_id is not None:
                self._add_reasoning_document_to_goal_summary(self.goal_plan.plan_reasoning_document_id)

            # add the focussed documents in full
            if self.goal_focus_docids is not None and len(self.goal_focus_docids)>0:
                focussed_documents = self.goal_focus_docids.split(",")
                self._add_loaded_documents_to_goal_summary(focussed_documents)

            # add the ontology rules to the goal summary (if currently active)
            ontology_rules = self._fetch_ontology_rules(self.goal_plan.get_ontology_id(), self.goal_plan.plan_community_ids)
            if ontology_rules is not None:
                self.goal_summary += "\n<ONTOLOGY_RULES>\n" + ontology_rules + "\n</ONTOLOGY_RULES>\n\n"

            # add the datasets to the goal summary
            document_datasets = [d for d in (self.goal_plan.plan_datasets or []) if not d.get("is_community", False)]
            if len(document_datasets) > 0:
                if self.goal_summary is None:
                    self.goal_summary = ""
                self.goal_summary += f"""
<DATASETS>
    Some datasets have been identified during the last RAG retrieval. Here is the summary for each of the datasets:"""
                for dataset in document_datasets:
                    dataset_id: str = str(dataset.get("id", dataset.get("name", "")))
                    dataset_reference = dataset_id if dataset_id.startswith("ds://") else f"ds://{dataset_id}"
                    # Build dataset opening tag with optional subset_of attribute
                    if "subset_of" in dataset and dataset["subset_of"] is not None and len(dataset["subset_of"]) > 0:
                        subset_list = ",".join(dataset["subset_of"])
                        self.goal_summary += f"""
    <DATASET id="{dataset_reference}" name="{dataset['name']}" subset_of="{subset_list}">"""
                    else:
                        self.goal_summary += f"""
    <DATASET id="{dataset_reference}" name="{dataset['name']}">"""
                    
                    # Always add description, domain, and metadata (regardless of subset_of)
                    self.goal_summary += f"""
        description: {dataset['description']}"""
                    if "domain" in dataset and dataset['domain'] is not None and len(str(dataset['domain'])) > 0:
                        self.goal_summary += f"""
        domain: {dataset['domain']}"""
                    if "metadata" in dataset and dataset['metadata'] is not None and dataset['metadata'] != "":
                        self.goal_summary += f"""
        metadata: {dataset['metadata']}"""
                    self.goal_summary += """
    </DATASET>"""

                self.goal_summary += """
</DATASETS>
"""

            # add any available hints to the goal summary
            if self.goal_plan.plan_hints is not None and len(self.goal_plan.plan_hints) > 0:
                if self.goal_summary is None:
                    self.goal_summary = ""
                self.goal_summary += f"""\n\n<HINTS>\nSome hints have been identified during the last RAG retrieval. Here is the summary for each of the hints:\n"""
                for hint in self.goal_plan.plan_hints:
                    self.goal_summary += f"""\n<HINT>\n{hint}\n</HINT>\n\n"""
                self.goal_summary += f"""\n</HINTS>\n"""
        
        except Exception as e:
            log(ERROR, f"Contextualizer generated an exception: {e}\n{traceback.format_exc()}")
            response = { "success": False }
            
        self.set_goal_status(GoalStatus.RUN_EVERY_TIME)
        return response
    
    def _run_noop(self) -> dict:
        log(DEBUG, f"Running Noop...")

        # the NOOP returns the goal prompt as the summary
        result = {
            "success": True,
            "summary": f"CHECKPOINT [Step {self.get_id_chain_as_str()}]"
        }
        self.set_summary(result["summary"])
        
        return result

    def _consolidator_deduplicate_fragments(self, current_goal:'Goal', already_seen_fragment_ids:List[str]) -> None:
        """
        Walk backwards through the goal chain and mark duplicate RAG fragments.

        This function traverses sibling goals (from the current goal backwards) and then
        moves up the parent chain recursively. For every RAG fragment (a dict with a
        "fragmentid" key) encountered, it checks whether the fragment id has already
        been seen (tracked in `already_seen_fragment_ids`). If a fragment id is already
        present, the fragment object is annotated with `frag["duplicate"] = True` so
        later processing can treat it as a duplicate. Otherwise the fragment id is
        appended to `already_seen_fragment_ids`.

        Args:
            current_goal (Goal): The goal to start scanning from.
            already_seen_fragment_ids (List[str]): Mutable list acting as a set of
                fragment ids discovered so far (kept across recursive calls).

        Side effects:
            - Mutates fragment dicts by adding a `duplicate` boolean flag when a
              duplicate is detected.
            - Mutates `already_seen_fragment_ids` by appending new fragment ids.

        Notes:
            - The traversal order ensures that the most recent fragments are treated
              as the canonical ones; older duplicates are marked.
        """
        # go backwards in the message chain, every time there's a duplicate RAG fragment, replace its text with an indication that it's a duplicate
        # to go backwards, start from the current goal and go up in the sibling chain, until go to the parent and repeat
        if current_goal.goal_parent is None:
            # identify eventual duplicates in the current goal's fragments
            fragments = current_goal.goal_rag_fragments
            if fragments is not None:
                for frag in fragments:
                    if isinstance(frag, dict) and "fragmentid" in frag:
                        if frag["fragmentid"] in already_seen_fragment_ids:
                            frag["duplicate"] = True
                        else:
                            already_seen_fragment_ids.append(frag["fragmentid"])
            return
        
        # go up in the sibling chain, start from the last sibling and go backwards
        idx = current_goal.goal_parent.goal_subgoals.index(current_goal)
        for i in range(idx-1, -1, -1):
            sibling = current_goal.goal_parent.goal_subgoals[i]
            fragments = sibling.goal_rag_fragments
            if fragments is not None:
                for frag in fragments:
                    if isinstance(frag, dict) and "fragmentid" in frag:
                        if frag["fragmentid"] in already_seen_fragment_ids:
                            frag["duplicate"] = True
                        else:
                            already_seen_fragment_ids.append(frag["fragmentid"])

        # go up in the parent chain
        self._consolidator_deduplicate_fragments(current_goal.goal_parent, already_seen_fragment_ids)

    def _consolidator_streamline_loaded_documents(self) -> None:
        """
        Streamline (normalize) enhanced HTML text of loaded documents/chapters.

        For each id in the plan's `plan_loaded_documents` this method loads the
        document/chapter with enhanced text, normalizes its HTML tag ids by stripping
        and re-adding tag ids (via `strip_html_tag_ids` and `add_html_tag_ids`) and
        writes the updated `enhanced_text` back to the database only if it changed.

        Behavior and constraints:
            - Only operates on items whose `content_type` is `html`; other types are
              ignored (the function returns early when a non-html content is found).
            - Uses `load_documents_or_chapters` to fetch current enhanced text and
              `execute_neo4j_query` to persist updates.
            - Exceptions are caught and logged; failures do not raise.

        Side effects:
            - May update `enhanced_text` in the database for documents that changed.
            - Logs debug messages when documents are streamlined.
        """
        if self.goal_plan is not None and self.goal_plan.plan_loaded_documents is not None and len(self.goal_plan.plan_loaded_documents) > 0:
            for content_id in self.goal_plan.plan_loaded_documents:
                # identify the content label
                id_split = content_id.split("-c")
                chapter_id = id_split[1] if len(id_split) > 1 else None
                content_label = "Document" if chapter_id is None else "Chapter"

                # determine the document id (strip the chapter suffix if present)
                doc_id = id_split[0]

                # load the document
                loaded_content = self.load_documents_or_chapters("id", content_id, content_label, load_enhanced_text=True)
                if loaded_content is not None:
                    enhanced_text_raw = loaded_content["text"]
                    content_type = loaded_content["content_type"]
                    if content_type != "html":
                        # only streamline html content is supported
                        continue
                    
                    try:
                        # replaces all html tags
                        streamlined_html = strip_html_tag_ids(enhanced_text_raw)
                        streamlined_html = add_html_tag_ids(streamlined_html)

                        if (streamlined_html != enhanced_text_raw):
                            # acquire per-document lock before writing to neo4j
                            lock = get_lock("DOCUMENT", doc_id)
                            with lock:
                                get_documents_service().update_content_enhanced_text(
                                    "id",
                                    content_id,
                                    content_label,
                                    streamlined_html,
                                )
                                log(DEBUG, f"Document id {content_id} was streamlined.")

                    except Exception as e:
                        log(ERROR, f"Contextualizer Streamline Document generated an exception: {e}\n{traceback.format_exc()}")

    def _run_consolidator_for_plan(self) -> dict:
        log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN] Starting for goal {self.get_id_chain_as_str()}")
        log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN] Initial state - goal_summary: {self.goal_summary}")
        log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN] Initial state - goal_summary_suppress: {self.goal_summary_suppress}")
        log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN] Initial state - goal_silent_summary: {self.goal_silent_summary}")
        log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN] Initial state - goal_rag_fragments count: {len(self.goal_rag_fragments) if self.goal_rag_fragments else 0}")
        log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN] Initial state - goal_rag_toomanyfragments: {self.goal_rag_toomanyfragments}")

        # grab information from all previous goals in the plan, on which this consolidator depends
        log(PERF, "cfp: start", perf_id="cfp_start")
        input_goals:List[Goal] = []
        if self.goal_dependent_upon:
            for goal_delta_index in self.goal_dependent_upon:
                previous_goal = self.get_previous_goal(delta_from_self=goal_delta_index)
                if previous_goal is not None:
                    input_goals.append(previous_goal)
                    log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN] Processing dependent goal {previous_goal.get_id_chain_as_str()}")
                    log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN]   - goal_summary: {previous_goal.goal_summary}")
                    log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN]   - goal_summary_suppress: {previous_goal.goal_summary_suppress}")
                    log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN]   - goal_silent_summary: {previous_goal.goal_silent_summary}")
                    log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN]   - goal_rag_fragments count: {len(previous_goal.goal_rag_fragments) if previous_goal.goal_rag_fragments else 0}")
                    log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN]   - goal_rag_toomanyfragments: {previous_goal.goal_rag_toomanyfragments}")

        log(PERF, f"cfp: {len(input_goals)} input goals gathered", perf_id="cfp_goals", perf_since_id="cfp_start")
        if len(input_goals) == 0:
            log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN] No dependent goals found, nothing to consolidate.")
            return { "success": True, "datasets": None }

        # Merge RAG fragments from all dependent goals into a single deduplicated list.
        #
        # Each input goal may carry fragments produced by a different agent:
        #   - AgentRAG stores full fragment dicts (fragmentid + documentid + text + ranking)
        #   - AgentReranker stores stub dicts (fragmentid only) after trimming to its top-N
        #
        # find_last_rag_fragments() resolves every stub to a full dict by walking backwards
        # through sibling goals (or, as a last resort, querying Neo4j directly).
        # After resolution, fragments are deduplicated by fragmentid across all goals so
        # that the same chapter/paragraph is not included twice in the consolidator output.
        # Document IDs are collected in parallel to update the plan's document focus.
        all_rag_fragments: List[dict] = []
        seen_fragment_ids: set = set()
        all_datasets: List[dict] = []
        doc_ids: List[str] = []
        optional_disambiguating_answer_proposals = None
        last_attributes = None
        for g in input_goals:
            g_id = g.get_id_chain_as_str()
            n_frags = len(g.goal_rag_fragments) if g.goal_rag_fragments else 0
            log(PERF, f"cfp loop goal {g_id}: {n_frags} raw frags, agent={g.goal_target_agent_name}", perf_id=f"cfp_g_{g_id}", perf_since_id="cfp_goals")
            # collect RAG fragments
            if g.goal_rag_fragments is not None and len(g.goal_rag_fragments) > 0:
                log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN] Goal {g_id} has {len(g.goal_rag_fragments)} fragments")
                frags = self.find_last_rag_fragments(g)
                log(PERF, f"cfp loop goal {g_id}: find_last_rag_fragments returned {len(frags)}", perf_since_id=f"cfp_g_{g_id}")
                added = 0
                for frag in frags:
                    frag_id = frag.get("fragmentid") if isinstance(frag, dict) else None
                    if frag_id is not None and frag_id in seen_fragment_ids:
                        continue   # skip fragment already collected from another goal
                    if frag_id is not None:
                        seen_fragment_ids.add(frag_id)
                    all_rag_fragments.append(frag)
                    added += 1
                    if isinstance(frag, dict) and "documentid" in frag and frag["documentid"] not in doc_ids:
                        doc_ids.append(frag["documentid"])
                log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN]   - Goal {g_id} contributed {added} fragments ({len(frags) - added} skipped as duplicates)")

            # collect structured datasets so names and metadata remain available to the UI
            datasets = g.get_datasets()
            if datasets and len(datasets) > 0:
                log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN] Goal {g_id} has {len(datasets)} datasets")
                for dataset in datasets:
                    dataset_id: str = dataset.get('id') if isinstance(dataset, dict) and dataset.get('id') else str(dataset)
                    if not any(existing.get('id') == dataset_id for existing in all_datasets):
                        all_datasets.append(dataset if isinstance(dataset, dict) else {"id": dataset_id})

            # collect attributes from the last goal that has them
            if g.goal_attributes is not None:
                last_attributes = g.goal_attributes

            # collect proposed followups from the first goal that has them
            if optional_disambiguating_answer_proposals is None and g.goal_proposed_followups is not None and len(g.goal_proposed_followups) > 0:
                optional_disambiguating_answer_proposals = g.goal_proposed_followups

        
        log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN] Total rag fragments collected: {len(all_rag_fragments)}.\nDatasets collected: {len(all_datasets) if len(all_datasets) > 0 else 'none'}.")
        log(PERF, f"cfp: {len(all_rag_fragments)} frags, {len(all_datasets)} datasets", perf_id="cfp_frags", perf_since_id="cfp_goals")

        # now grab all summaries from the input goals
        all_summaries = []
        for g in input_goals:
            if g.goal_summary_suppress:
                log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN] Skipping suppressed goal {g.get_id_chain_as_str()} summary")
                continue
            if isinstance(g.goal_summary, str):
                summary = g.goal_summary
            elif isinstance(g.goal_summary, list):
                summary = "\n".join(g.goal_summary)
            elif isinstance(g.goal_summary, dict):
                if "returned_objects" in g.goal_summary:
                    operation = f"The '{g.goal_summary['operation']}' operation returned: " if "operation" in g.goal_summary else ""
                    summary = f"{operation}{g.goal_summary['returned_objects']}"
                elif "results" in g.goal_summary:
                    operation = "This information was extracted with RAG: "
                    results = g.goal_summary['results']
                    if isinstance(results, list) and len(results) == 1:
                        if "results" in results[0]:
                            results = results[0]["results"]
                    summary = f"{operation}{results}"
                else:
                    summary = g.get_summary()
            else:
                summary = g.get_summary()

            if summary is not None and len(summary) > 0:
                if summary.strip().startswith("FUNCTION"):
                    summary = f"<HIDDEN_TO_USER>\n{summary}\n</HIDDEN_TO_USER>"
                all_summaries.append(summary)

        log(PERF, f"cfp: {len(all_summaries)} summaries", perf_id="cfp_summaries", perf_since_id="cfp_frags")

        # Build the consolidated summary. Command summaries (starting with '/') must be
        # propagated as-is — they trigger specific UI behaviours and must not be buried
        # inside a block of mixed text.  If the only non-empty summaries are commands,
        # return the first command alone so the consolidator goal carries a clean /command.
        command_summaries   = [s for s in all_summaries if s.strip().startswith("/")]
        content_summaries   = [s for s in all_summaries if not s.strip().startswith("/")]
        if content_summaries:
            # regular content present: keep it; append any command at the end so it is
            # still visible to the frontend without corrupting the readable text.
            consolidated_summary = "\n".join(content_summaries + command_summaries)
        elif command_summaries:
            # Only a command remains (for example /no_information_found); surface it alone.
            consolidated_summary = command_summaries[0]
        else:
            consolidated_summary = None
        log(INFO, f"[CONSOLIDATOR_FOR_PLAN] consolidated_summary='{consolidated_summary}' (content={len(content_summaries)}, cmds={len(command_summaries)})")

        # mark duplicate fragments in the broader goal tree (historical context across conversation turns)
        self._consolidator_deduplicate_fragments(self, [])

        # consolidate the eventually manipulated loaded documents in the background (fire-and-forget)
        threading.Thread(target=self._consolidator_streamline_loaded_documents, daemon=True).start()

        response = {
            "summary": consolidated_summary,
            "rag_fragments": all_rag_fragments if len(all_rag_fragments) > 0 else None,
            "rag_toomanyfragments": self.goal_rag_toomanyfragments,
            "document_ids": doc_ids if len(doc_ids) > 0 else None,
            "datasets": all_datasets if len(all_datasets) > 0 else None,
            "attributes": last_attributes,
            "optional_disambiguating_answer_proposals": optional_disambiguating_answer_proposals,
            "success": True
        }
        
        log(DEBUG, f"[CONSOLIDATOR_FOR_PLAN] Response - summary length: {len(response['summary']) if response['summary'] else 0}\n[CONSOLIDATOR_FOR_PLAN] Response - rag_fragments count: {len(response['rag_fragments']) if response['rag_fragments'] else 0}\n[CONSOLIDATOR_FOR_PLAN] Response - datasets count: {len(response['datasets']) if response['datasets'] else 0}")
        log(PERF, f"cfp: done, {len(all_summaries)} summaries, {len(all_rag_fragments)} frags", perf_id="cfp_end", perf_since_id="cfp_start")

        return response

    def _run_consolidator(self) -> dict:
        log(DEBUG, f"Running Consolidator for goal '{self.get_id_chain_as_str()}'...")
        log(PERF, "Consolidator start", perf_id="cons_start")

        try:
            response = self._run_consolidator_for_plan()

        except Exception as e:
            log(ERROR, f"Consolidator for goal {self.get_id_chain_as_str()} generated an exception: {e}\n{traceback.format_exc()}")
            response = { "success": False, "datasets": None }

        log(PERF, "Consolidator done", perf_id="cons_end", perf_since_id="cons_start")
        log(DEBUG, f"[CONSOLIDATOR] Final response for goal {self.get_id_chain_as_str()}: {response}")

        # Mirror what run_agent() does: store all relevant fields from the response back
        # onto this goal so that the frontend and downstream goals can read them.
        # This is especially important for command summaries (for example /no_information_found)
        # that must surface to the UI even though AgentAssistant is
        # bypassed by /query, and for rag_fragments so that any goal depending directly
        # on this Consolidator goal can find its fragments via goal_rag_fragments.
        if response is not None:
            _cons_summary = response.get("summary")
            if _cons_summary and len(_cons_summary.strip()) > 0:
                self.goal_summary = _cons_summary
                log(DEBUG, f"[CONSOLIDATOR] Set goal_summary='{self.goal_summary}'")
            self.goal_rag_fragments = response.get("rag_fragments")
            self.goal_rag_toomanyfragments = response.get("rag_toomanyfragments")
            self.goal_rag_docids = response.get("document_ids")
            if "datasets" in response:
                self.set_datasets(response["datasets"])
            log(DEBUG, f"[CONSOLIDATOR] Set goal_rag_fragments count={len(self.goal_rag_fragments) if self.goal_rag_fragments else 0}")

        send_plan_to_blazor(self.goal_plan, f"Consolidator completed for goal {self.get_id_chain_as_str()}")
        
        return response
    
    def __self_describe(self) -> str:
        description = None
        if self.goal_target == GoalTarget.UNDEFINED:
            description = "UNDEFINED"
        elif self.goal_target == GoalTarget.ONLY_SUBGOALS:
            description = "This goal is a container for sub-goals. It doesn't do anything in and by itself."
        elif self.goal_target == GoalTarget.CONTEXTUALIZER:
            description = "The Contextualizer fetches the plan's focus and RAG summaries and provides them to the conversation"
        elif self.goal_target == GoalTarget.DEBUG:
            description = "(debug message)"
        elif self.goal_target == GoalTarget.AGENT and self.goal_target_agent:
            if self.goal_target_agent_name.startswith('AgentAssistant'):
                description = "The Agent Assistant is an A.I. Agent which uses OpenAI's GPT to produce a message, based on the previous conversation context."
            elif self.goal_target_agent_name.startswith('AgentUser'):
                description = "The AgentUser awaits input from the user: it is the user's turn to speak. The user's input is submitted via the ChatBot and this goal makes it available to the conversation."
            elif self.goal_target_agent_name.startswith('AgentToolPreparer'):
                description = "The AgentToolPreparer is responsible to identify which python function to call, and which are the parameters. It uses the 'tools' of the LLM"
            elif self.goal_target_agent_name.startswith('AgentToolCaller'):
                description = "The AgentToolCaller responsible to execute a python function. Previously the Agent Assistant may have suggested to call a Skill, which case the AgentTool will execute the Skill. Othwerwise, the AgentTool will execute any python function provided at development time in a plan template."
            elif self.goal_target_agent_name.startswith('AgentFocusser'):
                description = "The AgentFocusser is responsible to summarize the conversation and update the plan_focus."
            elif self.goal_target_agent_name.startswith('AgentRAG'):
                description = "The AgentRAG is responsible to fetch document fragments from neo4j, sort them out, and make them available to the conversation."
            elif self.goal_target_agent_name.startswith('AgentPlanner'):
                description = "The AgentPlanner is responsible to instantiate a plan that will guide the next steps of the conversation."
            elif self.goal_target_agent_name.startswith('AgentReviewer'):
                description = "The AgentReviewer is responsible for quality: it checks previous generated answers and outputs either OK or a list of issues to be corrected."
            else:
                description = "UNKNOWN AGENT"
        else:
            description = "UNKNOWN TARGET"

        return description

    # Static methods
    @staticmethod
    def compare_id_chains(chain1: list, chain2: list) -> int:
        """
        Compare two int[] chains lexicographically.
        Returns 1 if chain1 > chain2, -1 if chain1 < chain2, 0 if equal.
        """
        min_len = min(len(chain1), len(chain2))
        for i in range(min_len):
            if chain1[i] > chain2[i]:
                return 1
            elif chain1[i] < chain2[i]:
                return -1
        if len(chain1) > len(chain2):
            return 1
        elif len(chain1) < len(chain2):
            return -1
        return 0


log_end_file(__file__)

#region unit tests



if __name__ == '__main__':
    from src.utils.unit_test import init_unit_test
    from src.utils.log import log_init, log_exit
    init_unit_test()
    log_init(limit_console_log_level=DEBUG)

    def _test_collect_fragments_from_rerank_dataset():
        """Test _collect_fragments_from_goal_datasets against the live rerank_1 dataset in Neo4j."""
        from src.config.constants import MIN_RERANKING_SCORE_FRAG
        from src.services.dataset_service import get_dataset_service
        from src.interfaces.dataset_interface import DatasetInterface

        DATASET_ID = "luca-s-1#rerank_1"

        # ── Step 1: Direct service call ─────────────────────────────────────────────
        log(DEBUG, "=== Step 1: dataset_service.get_fragments_from_dataset() ===")
        dataset_service: DatasetInterface = get_dataset_service()
        fragments = dataset_service.get_fragments_from_dataset(
            dataset_id=DATASET_ID,
            include_weights=True
        )
        log(DEBUG, f"get_fragments_from_dataset returned {len(fragments) if fragments else 0} fragments")
        for i, f in enumerate(fragments or []):
            log(DEBUG, f"  [{i}] id={f.get('id')}, weight={f.get('weight')}, labels={f.get('labels')}, text_len={len(str(f.get('text', '') or ''))}")

        # ── Step 2: Instantiate two minimal Goal objects ─────────────────────────────
        # 'caller_goal'  = goal calling build_messages (e.g. AgentAssistant 1.4.2.8)
        # 'dataset_goal' = goal whose datasets we read  (e.g. AgentReranker 1.4.2.6)
        log(DEBUG, "=== Step 2: Creating minimal Goal objects ===")

        import threading
        class _MockPlan:
            plan_lock = threading.Lock()
            plan_loaded_documents = None

        mock_plan = _MockPlan()

        parent_goal = Goal(mock_plan, None, "Parent (test)", None, GoalTarget.ONLY_SUBGOALS, None, None)
        filler = Goal(mock_plan, parent_goal, "filler (RAGQuerySplitter)", None, GoalTarget.AGENT, None, None)
        dataset_goal = Goal(mock_plan, parent_goal, "AgentReranker (test)", None, GoalTarget.AGENT, None, None)
        caller_goal = Goal(mock_plan, parent_goal, "AgentAssistant (test)", None, GoalTarget.AGENT, None, [-2, -1])
        parent_goal.goal_subgoals = [filler, dataset_goal, caller_goal]

        # Simulate what run_agent() stores after AgentReranker completes:
        dataset_goal.set_datasets([{"id": DATASET_ID, "name": "rerank_1"}])
        log(DEBUG, f"dataset_goal.goal_datasets = {dataset_goal.goal_datasets}")

        # ── Step 3: identify_goals_with_rag_fragments_suitable_for_self ──────────────
        log(DEBUG, "=== Step 3: identify_goals_with_rag_fragments_suitable_for_self ===")
        rag_goals = caller_goal.identify_goals_with_rag_fragments_suitable_for_self()
        log(DEBUG, f"returned {len(rag_goals)} goal(s)")
        for g in rag_goals:
            log(DEBUG, f"  rag goal: title={g.goal_title}, datasets={g.goal_datasets}")

        # ── Step 4: _collect_fragments_from_goal_datasets ────────────────────────────
        log(DEBUG, "=== Step 4: _collect_fragments_from_goal_datasets ===")
        log(DEBUG, f"minimum_rank (MIN_RERANKING_SCORE_FRAG) = {MIN_RERANKING_SCORE_FRAG}")

        fragments_by_id = {}
        chars_added, truncated = caller_goal._collect_fragments_from_goal_datasets(
            goal=dataset_goal,
            minimum_rank=MIN_RERANKING_SCORE_FRAG,
            fragments_by_id=fragments_by_id,
            current_char_count=0
        )
        log(DEBUG, f"Result: chars_added={chars_added}, truncated={truncated}, fragments_collected={len(fragments_by_id)}")
        for fid, frag in fragments_by_id.items():
            log(DEBUG, f"  collected: id={fid}, ranking={frag.get('ranking')}, text_len={len(str(frag.get('text', '') or ''))}")

        if len(fragments_by_id) == 0:
            log(WARNING, "*** NO FRAGMENTS COLLECTED — checking per-fragment weights vs minimum_rank:")
            for f in (fragments or []):
                w = f.get('weight')
                try:
                    passes = float(w) >= MIN_RERANKING_SCORE_FRAG
                except Exception:
                    passes = None
                log(WARNING, f"  id={f.get('id')}, weight={w}, passes_threshold={passes}")
        else:
            log(DEBUG, "*** Fragments collected OK ***")

        # ── Step 5: identify_goals_with_rag_fragments_suitable_for_self (with plan) ──
        log(DEBUG, "=== Step 5: identify_goals_with_rag_fragments_suitable_for_self (with plan) ===")
        rag_goals = caller_goal.identify_goals_with_rag_fragments_suitable_for_self()
        log(DEBUG, f"returned {len(rag_goals)} goal(s)")
        for g in rag_goals:
            log(DEBUG, f"  rag goal: title={g.goal_title}, datasets={g.goal_datasets}")

        # ── Step 6: _generate_rag_messages end-to-end ────────────────────────────────
        log(DEBUG, "=== Step 6: _generate_rag_messages end-to-end ===")
        rag_message = caller_goal._generate_rag_messages(rag_goals, minimum_rank=MIN_RERANKING_SCORE_FRAG)
        if rag_message:
            log(DEBUG, f"SUCCESS: _generate_rag_messages returned {len(rag_message)} chars")
            log(DEBUG, f"First 300 chars: {rag_message[:300]}")
        else:
            log(WARNING, "FAIL: _generate_rag_messages returned None or empty")

    try:

        if True:
            _test_collect_fragments_from_rerank_dataset()

        if False:
            # Example:
            #   1.1 goal [no dependencies]
            #   1.2 goal [depends on 1.1]
            #   1.3 goal [depends on 1.1, 1.2]
            #   1.4 goal [depends on 1.3]         (point of insertion after_id==1.4)
            #   1.5 goal [depends on 1.3, 1.4]
            #   1.6 goal [depends on 1.5]
            goal1 = Goal(None, None, "Goal 1", "Description of Goal 1", None, None, [])             # parent goal
            goal11 = Goal(None, goal1, "Goal 1.1", "Description of Goal 1.1", None, None, [])
            goal12 = Goal(None, goal1, "Goal 1.2", "Description of Goal 1.2", None, None, [-1])
            goal13 = Goal(None, goal1, "Goal 1.3", "Description of Goal 1.3", None, None, [-2, -1])
            goal14 = Goal(None, goal1, "Goal 1.4", "Description of Goal 1.4", None, None, [-1])     # point of insertion: after this one we add the new goals
            goal15 = Goal(None, goal1, "Goal 1.5", "Description of Goal 1.5", None, None, [-2, -1])
            goal16 = Goal(None, goal1, "Goal 1.6", "Description of Goal 1.6", None, None, [-1])
            print("===================================")
            print(f"{goal1}")

            new_goal1 = Goal(None, None, "New Goal 1", "Description of New Goal 1", None, None, [])
            new_goal2 = Goal(None, None, "New Goal 2", "Description of New Goal 2", None, None, [-1])
            new_goals = [new_goal1, new_goal2]

            # Add the new goals after goal 1.4
            goal1.insert_goals(new_goals, after_id=goal14.get_id_chain(), instantiated_plan=None)

            # Expected result after inserting:
            # [0]  1.1 goal [no dependencies]                --> unchanged
            # [1]  1.2 goal [depends on -1]                  --> unchanged
            # [2]  1.3 goal [depends on -2, -1]              --> unchanged
            # [3]  1.4 goal [depends on -1]                  --> unchanged
            # [4]  1.5 first new goal [depends on -1]        --> dependency injection to the current goal (self)
            # [5]  1.6 second new goal [depends on -2, -1]   --> dependency injection to the current goal (self) at -2, plus the intrinsic dependencies which are now -1
            # [6]  1.7 goal [depends on -4, -3, -2, -1]      --> dependency on 1.3, 1.4 are maintained + new dependencies are injected
            # [7]  1.8 goal [depends on -1]                  --> unchanged
            print("\n=================================== AFTER INSERTION:")
            print(f"{goal1}")
            assert set(goal1.goal_subgoals[0].goal_dependent_upon) == set([])
            assert set(goal1.goal_subgoals[1].goal_dependent_upon) == set([-1])
            assert set(goal1.goal_subgoals[2].goal_dependent_upon) == set([-2, -1])
            assert set(goal1.goal_subgoals[3].goal_dependent_upon) == set([-1])
            assert set(goal1.goal_subgoals[4].goal_dependent_upon) == set([-1])
            # Observed behavior after insertion:
            # index 4 -> New Goal 1 dependent upon [-1]
            # index 5 -> New Goal 2 dependent upon [-1, -2] (order may differ)
            # index 6 -> original 1.5 now dependent upon [-2, -1]
            # index 7 -> original 1.6 now dependent upon [-1]
            assert set(goal1.goal_subgoals[5].goal_dependent_upon) == set([-1, -2])
            assert set(goal1.goal_subgoals[6].goal_dependent_upon) == set([-2, -1])
            assert set(goal1.goal_subgoals[7].goal_dependent_upon) == set([-1])

            # ------------------------------------------------------------------
            # Unit test: generate_rag_messages formatting
            # ------------------------------------------------------------------
            print("\n=== RAG message formatting unit test ===")
            rag_goal = Goal(None, None, "RAG Goal", "RAG description", None, None, [])
            rag_goal.goal_rag_fragments = [
                {"documentid": "docid-1", "fragmentid": "docid-1-c0", "ranking": 1.0, "ranking_text": "very_high", "text": "<p>Example text</p>"},
                {"documentid": "docid-1", "fragmentid": "docid-1-c1", "ranking": 0.4, "ranking_text": "low", "text": "<p>Other text</p>"}
            ]

            rag_message = rag_goal._generate_rag_messages(rag_goal)
            print(rag_message)

            # Basic sanity checks for the new markdown formatting
            assert rag_message is not None, "generate_rag_messages returned None"
            assert "### Fragment: docid-1-c0" in rag_message, "missing fragment heading"
            assert "```docid-1-c0" in rag_message, "missing labeled code fence for fragment"
            assert "<p>Example text</p>" in rag_message, "fragment text missing"
            assert "RAG_fragments.txt" not in rag_message, "old wrapper still present"
            print("RAG formatting unit test passed.")

    except Exception as e:
        log(ERROR, f"Unit test for {__file__} failed with exception: {e}. Traceback:\n{traceback.format_exc()}")
    finally:
        log_exit()
#endregion
