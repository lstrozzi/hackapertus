from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import os
import json
import threading
import traceback
import uuid
import re
import concurrent.futures
from typing import TYPE_CHECKING, List, Optional, Union

from src.core.base.enums import AgentReasoningEnum
from src.core.base.InputOutput import get_language_name
from src.core.base.llm_response import LlmResponse
from src.core.llm.config import MAX_COMPLETION_TOKENS
from src.core.llm.agent_rag_settings import AgentRagSettings
from src.core.llm.manager import LlmManager
from src.core.thalaplan.goal.enums import GoalStatus, GoalTarget, GoalAsyncStatus
from src.core.thalaplan.goal.Goal import Goal
from src.core.extensions.skillset_management import import_class, import_function
from src.interfaces.dossiers_interface import DossiersInterface
from src.interfaces.dataset_interface import DatasetInterface
from src.services.ontology_service import ONTOLOGY_DEFAULT_ID
from src.services.cache_entry import get_cache_entry_service
from src.services.dossiers_service import get_dossiers_service
from src.services.dataset_service import get_dataset_service
from src.utils.log import log, log_plan, INFO, DEBUG, WARNING, ERROR, FATAL

if TYPE_CHECKING:
    from src.core.agents.Agent import Agent

#region global variables
initialized = False
global_plan = None
global_dossier_id = None
global_session_dossier_id = None
agent_assistant = None
agent_user = None
agent_focusser = None
agent_verifier = None
agent_ontology = None
plan_from_prompt_template = None
plan_chat_template = None
plan_search_template = None
#endregion global variables


def init():
    global initialized, agent_assistant, agent_presenter, agent_user, agent_focusser, agent_verifier, plan_from_prompt_template, plan_chat_template, plan_search_template
    if initialized:
        return
    initialized = True
    
    agent_assistant = import_class("src.core.agents.AgentAssistant", "AgentAssistant")
    agent_presenter = import_class("src.core.agents.AgentPresenter", "AgentPresenter")
    agent_user = import_class("src.core.agents.AgentUser", "AgentUser")
    agent_focusser = import_class("src.core.agents.AgentFocusser", "AgentFocusser")
    agent_verifier = import_class("src.core.agents.AgentVerifier", "AgentVerifier")
    # agent_ontologist = import_class("src.core.agents.AgentOntology", "AgentOntology")
    agent_strategist = import_class("src.core.agents.AgentStrategist", "AgentStrategist")
    agent_planner = import_class("src.core.agents.AgentPlanner", "AgentPlanner")
    agent_microplanner = import_class("src.core.agents.AgentMicroplanner", "AgentMicroplanner")

    plan_from_prompt_template = []
    plan_from_prompt_template.append(Goal(None, None, "Assistant-prompts", None, GoalTarget.AGENT, agent_assistant, None))
    plan_from_prompt_template.append(Goal(None, None, "Presenter-prompts", None, GoalTarget.AGENT, agent_presenter, [-1]))

    plan_chat_template = []
    plan_chat_template.append(Goal(None, None, "Focusser", None, GoalTarget.AGENT, agent_focusser, None, async_status=GoalAsyncStatus.ASYNC_REQUESTED))
    plan_chat_template.append(Goal(None, None, "Human-prompts", None, GoalTarget.AGENT, agent_user, []))
    composite_goal = Goal(None, None, "CompositeAssistant", None, GoalTarget.ONLY_SUBGOALS, None, [-1])
    composite_goal.insert_goals([
        # Goal(None, None, "AgentOntology", None, GoalTarget.AGENT, agent_ontologist, None), => moved to RAG pipeline
        Goal(None, None, "AgentStrategist", None, GoalTarget.AGENT, agent_strategist, None),
        Goal(None, None, "AgentPlanner", None, GoalTarget.AGENT, agent_planner, [-1]),
        Goal(None, None, "Consolidator", None, GoalTarget.CONSOLIDATOR, None, [-1])
    ], after_id=None, instantiated_plan=None)
    plan_chat_template.append(composite_goal)

    plan_search_template = []
    plan_search_template.append(Goal(None, None, "Search", None, GoalTarget.AGENT, agent_microplanner, [-1]))

    return

def instantiate_plan(plan:'Plan', template:List[Goal]) -> List[Goal]:
    goals = []
    for template_goal in template:
        goal_dependencies_copy = template_goal.goal_dependent_upon.copy() if template_goal.goal_dependent_upon is not None else []
        goal = Goal(plan, None, template_goal.goal_title, template_goal.goal_description, template_goal.goal_target, template_goal.goal_target_agent, goal_dependencies_copy)
        goal.goal_silent_summary = template_goal.goal_silent_summary
        goal.goal_recommended_plan = template_goal.goal_recommended_plan
        goal.goal_override_recommended_tools = template_goal.goal_override_recommended_tools
        goal.goal_repeat_on_error = template_goal.goal_repeat_on_error
        goal.goal_purpose = template_goal.goal_purpose
        goal.set_target_agent_name(template_goal.goal_target_agent_name)
        goal.goal_prompt = template_goal.goal_prompt
        goal.set_async_status(template_goal.get_async_status())
        if template_goal.goal_subgoals:
            goal.goal_subgoals = instantiate_plan(plan, template_goal.goal_subgoals)
            for subgoal in goal.goal_subgoals:
                subgoal.goal_parent = goal
        goals.append(goal)

    return goals

def plan_from_prompt(plan:'Plan', prompt:str) -> List[Goal]:
    init()
    template = plan_from_prompt_template
    goals = instantiate_plan(plan, template)
    return goals

def plan_chat(plan:'Plan') -> List[Goal]:
    init()
    template = plan_chat_template
    goals = instantiate_plan(plan, template)
    if LlmManager.instance().get_quality_settings().get("agent_verifier_enable", False):
        verifier_goal = Goal(plan, None, "Verifier", None, GoalTarget.AGENT, agent_verifier, [-1], async_status=GoalAsyncStatus.ASYNC_REQUESTED)
        goals.append(verifier_goal)
    return goals

def set_global_plan(plan:'Plan') -> None:
    global global_plan
    log(INFO, f"Setting global plan: {plan.plan_name}")
    global_plan = plan

def get_global_plan() -> 'Plan':
    global global_plan
    return global_plan

def set_global_dossier_id(dossier_id:Optional[str]) -> None:
    global global_dossier_id
    if dossier_id is not None:
        log(INFO, f"Memorizing global dossier ID: {dossier_id}")
    global_dossier_id = dossier_id
   
def set_global_session_dossier_id(dossier_id:str) -> None:
    global global_session_dossier_id
    if dossier_id is not None:
        log(INFO, f"Memorizing global dossier ID: {dossier_id}")
    global_session_dossier_id = dossier_id

def get_global_dossier_id() -> str:
    global global_dossier_id
    return global_dossier_id

def get_global_session_dossier_id() -> str:
    global global_session_dossier_id
    return global_session_dossier_id


class Plan:
    """Plan class for managing goals, documents, and execution state."""

    def __init__(self, name:str, description:str=None):
        # identification and description
        self.plan_guid: str = self.generate_guid()
        self.plan_name: str = name
        self.plan_description: str = description
        
        # plan root goal (will be set after Goal creation)
        self.plan_goal: Goal = None
        
        # plan version management
        self.plan_revision: int = 0
        
        # configuration
        self.plan_focus: str = None
        self.plan_domain: Optional[str] = None
        self.plan_flavour: str = None
        self.plan_active_reasoning: AgentReasoningEnum = AgentReasoningEnum.FAST
        
        # token statistics
        self.plan_total_completion_tokens: int = 0
        self.plan_maximum_tokens: int = 0
        self.plan_token_limit: int = None
        
        # whitelists and blacklists
        self.plan_template_whitelist: Optional[list] = None
        self.plan_template_blacklist: list = None
        
        # documents
        self.plan_dossier_id = None                # loaded dossier. A summary of its documents is included in the context
        self.plan_session_dossier_id = None        # uploaded documents for this session. Their text content is included in the context
        self.plan_document_ids: list = []          # all document that have been collected as RAG sources
        self.plan_loaded_documents: list = []      # documents explicitely loaded into the plan by DocumentLoader skill
        
        # deep reasoning
        self.plan_document_template_ids: list = []
        self.plan_reasoning_document_id: str = None
        
        # datasets (always a list of dataset dicts in fetch_datasets format)
        self.plan_datasets: Optional[List[dict]] = None
        
        # knowledge graph, ontology, community
        self.plan_ontology_id:  Optional[str] = None
        self.plan_community_ids: Optional[List[str]] = None
        
        # hints
        self.plan_hints: list = []

        # language and time
        self.plan_default_language: str = get_language_name()
        self.plan_language_in_use: Optional[str] = None
        self.plan_include_time: bool = False

        # custom extensions
        self.plan_extensions: dict = None

        # SDK extensions
        self.plan_is_remote_sdk: bool = False
        self.plan_use_only_remote_skill: bool = False
        self.plan_system_prompt : Optional[str] = None
        
        # debugging
        self.plan_disable_debug: bool = False
        
        # multi-threading lock
        self.plan_lock: threading.Lock = threading.Lock()
        
        # command handling
        self.plan_command_handler = None
        self.plan_translator_agent = None

        # add root goal
        self.plan_goal: Goal = Goal(self, None, "Root Goal", None, GoalTarget.ONLY_SUBGOALS, None, [])

        # add first goal(s)
        contextualizer: Goal = Goal(self, self.plan_goal, None, None, GoalTarget.CONTEXTUALIZER, None, [])
        contextualizer.set_goal_status(GoalStatus.RUN_EVERY_TIME)

    def __str__(self, include_rag:bool=False):
        return f"""
\033[90mPlan Name:\033[0m {self.plan_name}
\033[90mPlan Revision:\033[0m {self.plan_revision}
\033[90mPlan Description:\033[0m {self.plan_description}
\033[90mPlan Domain:\033[0m {self.plan_domain if self.plan_domain is not None else 'general'}
\033[90mPlan Domain:\033[0m {self.plan_flavour if self.plan_flavour is not None else 'NEUTRAL'}
\033[90mPlan Focus:\033[0m {self.plan_focus}
\033[90mPlan Dossier ID:\033[0m {self.plan_dossier_id}
\033[90mPlan Session Dossier ID:\033[0m {self.plan_session_dossier_id}
\033[90mPlan Document IDs:\033[0m {self.plan_document_ids}
\033[90mPlan Loaded Documents:\033[0m {self.plan_loaded_documents}
\033[90mPlan Datasets:\033[0m {self.plan_datasets}
\033[90mPlan GUID:\033[0m {self.plan_guid}
\033[90mPlan Hints:\033[0m {self.plan_hints}
\033[90mPlan Default Language:\033[0m {self.plan_default_language}
\033[90mPlan Language In Use:\033[0m {self.get_language_in_use()}
\033[90mPlan Goal:\033[0m {self.plan_goal.full_str(recursive=True)}
\033[90mPlan Revision: \033[0m{self.plan_revision}
    """
    
    def clone(self) -> 'Plan':
        plan = Plan(self.plan_name, self.plan_description)

        # Basic metadata
        plan.plan_domain = self.plan_domain
        plan.plan_flavour = self.plan_flavour
        plan.plan_focus = self.plan_focus
        plan.plan_default_language = self.plan_default_language
        plan.set_language(self.get_language_in_use())
        plan.plan_include_time = self.plan_include_time

        # Reasoning / tokens / templates
        plan.plan_active_reasoning = self.plan_active_reasoning
        plan.plan_total_completion_tokens = self.plan_total_completion_tokens
        plan.plan_maximum_tokens = self.plan_maximum_tokens
        plan.plan_template_whitelist = self.plan_template_whitelist.copy() if self.plan_template_whitelist is not None else None
        plan.plan_template_blacklist = self.plan_template_blacklist.copy() if self.plan_template_blacklist is not None else None

        # Dossier / documents
        plan.plan_dossier_id = self.plan_dossier_id
        plan.plan_session_dossier_id = self.plan_session_dossier_id
        if self.plan_document_ids:
            plan.add_documents(self.plan_document_ids.copy())
        if self.plan_document_template_ids:
            plan.add_document_templates(self.plan_document_template_ids.copy())
        if self.plan_loaded_documents:
            plan.add_loaded_documents(self.plan_loaded_documents.copy())
        if self.plan_reasoning_document_id:
            plan.set_reasoning_document_id(self.plan_reasoning_document_id)

        # Datasets, ontology, communities, hints, extensions
        plan.plan_datasets = self.plan_datasets.copy() if isinstance(self.plan_datasets, list) else self.plan_datasets
        plan.plan_ontology_id = self.plan_ontology_id
        plan.plan_community_ids = self.plan_community_ids.copy() if isinstance(self.plan_community_ids, list) else self.plan_community_ids
        plan.plan_hints = self.plan_hints.copy() if isinstance(self.plan_hints, list) else (self.plan_hints if self.plan_hints is not None else [])

        # SDK extensions
        plan.plan_is_remote_sdk = self.plan_is_remote_sdk
        plan.plan_use_only_remote_skill = self.plan_use_only_remote_skill

        # Debug, command and agents
        plan.plan_disable_debug = self.plan_disable_debug
        plan.plan_translator_agent = self.plan_translator_agent
        plan.plan_command_handler = self.plan_command_handler

        # Create a fresh lock for the clone
        plan.plan_lock = threading.Lock()

        # Clone the goal tree
        plan.plan_goal = self.plan_goal.clone()

        # Revision and identity
        plan.plan_revision = self.plan_revision
        plan.plan_guid = self.plan_guid

        return plan

    def to_json(self, include_only_goal_id:List[int]=None) -> dict:
        plan_goal_json = self.plan_goal.to_json(include_only_goal_id)
        plan_json = {
            "plan_name": self.plan_name,
            "plan_description": self.plan_description,
            "plan_domain": self.plan_domain,
            "plan_focus": self.plan_focus,
            "plan_total_completion_tokens": self.plan_total_completion_tokens,
            "plan_maximum_tokens": self.plan_maximum_tokens,
            "plan_token_limit": self.get_token_limit(),
            "plan_default_language": self.plan_default_language,
            "plan_language_in_use": self.get_language_in_use(),
            "plan_dossier_id": self.plan_dossier_id,
            "plan_session_dossier_id": self.plan_session_dossier_id,
            "plan_document_ids": self.plan_document_ids,
            "plan_document_template_ids": self.plan_document_template_ids,
            "plan_loaded_documents": self.plan_loaded_documents,
            "plan_reasoning_document_id": self.plan_reasoning_document_id,
            "plan_datasets": self.plan_datasets,
            "plan_ontology_id": self.plan_ontology_id,
            "plan_community_ids": self.plan_community_ids,
            "plan_hints": self.plan_hints,
            "plan_disable_debug": self.plan_disable_debug,
            "plan_goal": plan_goal_json,
            "plan_revision": self.plan_revision,
            "plan_guid": self.plan_guid
        }
        return plan_json

    def from_json(self, json_data: dict):
        self.plan_name = json_data.get("plan_name")
        self.plan_description = json_data.get("plan_description")
        self.plan_domain = json_data.get("plan_domain")
        self.plan_focus = json_data.get("plan_focus")

        # token statistics
        self.plan_total_completion_tokens = json_data.get("plan_total_completion_tokens", 0)
        self.plan_maximum_tokens = json_data.get("plan_maximum_tokens", 0)
        self.plan_default_language = json_data.get("plan_default_language")
        self.set_language(json_data.get("plan_language_in_use"))
        self.plan_dossier_id = json_data.get("plan_dossier_id")
        self.plan_session_dossier_id = json_data.get("plan_session_dossier_id")
        
        # Use the add methods to ensure proper initialization and deduplication
        document_ids = json_data.get("plan_document_ids")
        if document_ids:
            self.add_documents(document_ids)
        
        document_template_ids = json_data.get("plan_document_template_ids")
        if document_template_ids:
            self.add_document_templates(document_template_ids)
            
        loaded_documents = json_data.get("plan_loaded_documents")
        if loaded_documents:
            self.add_loaded_documents(loaded_documents)

        self.plan_reasoning_document_id = json_data.get("plan_reasoning_document_id")
        # Normalize datasets via setter so strings are converted to dicts when possible
        plan_datasets = json_data.get("plan_datasets")
        if plan_datasets is not None:
            self.set_datasets(plan_datasets)
        self.plan_ontology_id = json_data.get("plan_ontology_id")
        self.plan_community_ids = json_data.get("plan_community_ids")
        self.plan_hints = json_data.get("plan_hints", [])
        self.plan_disable_debug = json_data.get("plan_disable_debug")
        self.plan_revision = json_data.get("plan_revision")
        self.plan_goal.from_json(json_data.get("plan_goal"))
        return
    
    def set_command_handler(self, command_handler) -> None:
        self.plan_command_handler = command_handler

    def generate_guid(self) -> str:
        return str(uuid.uuid4())

    def get_rag_settings(self) -> AgentRagSettings:
        try:
            rag_settings = LlmManager.instance().get_active_rag_settings()
            
        except Exception as e:
            log(ERROR, f"Error retrieving RAG settings from LLM Manager: {e}\n{traceback.format_exc()}")
            rag_settings = AgentRagSettings()   # defaults
            
        return rag_settings

    def get_token_limit(self) -> int:
        if self.plan_token_limit is None:
            self.plan_token_limit = MAX_COMPLETION_TOKENS
        return self.plan_token_limit
    
    def set_token_limit(self, token_limit:int) -> None:
        # increase only
        if token_limit is None:
            return
        if self.plan_token_limit is None or token_limit > self.plan_token_limit*1024:
            self.plan_token_limit = token_limit
        return
    
    def set_token_limit_for_agent(self, agent:'Agent', purpose:Optional[str]=None) -> None:
        # increase only
        llm_config = LlmManager.instance().get_llm_model_configuration_for_agent(agent, purpose)
        if llm_config is None:
            return
        
        token_limit_kilo = llm_config["token_limit_kilo"]*1024 if "token_limit_kilo" in llm_config and llm_config["token_limit_kilo"] is not None else None
        if token_limit_kilo is not None:
            token_limit = token_limit_kilo * 1024
            if (self.plan_token_limit is None or token_limit > self.plan_token_limit):
                self.plan_token_limit = token_limit
        return

    def set_language(self, language: Optional[str]) -> None:
        """Set the conversation language for this plan.
        
        Args:
            language: 2-letter ISO language code (e.g., 'en', 'de', 'fr') or None to use default
        """
        if language is not None and language.strip():
            self.plan_language_in_use = language.strip().lower()
            log(INFO, f"Plan language set to: {self.plan_language_in_use}")
        else:
            self.plan_language_in_use = self.plan_default_language
            log(INFO, f"Plan language reset to default: {self.plan_default_language}")

    def get_language_in_use(self) -> Optional[str]:
        """Get the conversation language currently in use.
        
        Returns:
            2-letter ISO language code if set, otherwise default language
        """
        return self.plan_language_in_use if self.plan_language_in_use is not None else self.plan_default_language

    def set_system_prompt(self, system_prompt: Optional[str]) -> None:
        """Set the system prompt for this plan.
        
        Args:
            system_prompt: The system prompt string or None to clear it
        """
        if system_prompt is not None and system_prompt.strip():
            self.plan_system_prompt = system_prompt.strip()
            log(INFO, f"Plan system prompt set to: {self.plan_system_prompt[:50]}..." if len(self.plan_system_prompt) > 50 else f"Plan system prompt set to: {self.plan_system_prompt}")
        else:
            self.plan_system_prompt = None
            log(INFO, f"Plan system prompt cleared")

    def get_system_prompt(self) -> Optional[str]:
        """Get the system prompt currently in use.
        
        Returns:
            The system prompt string if set, otherwise None
        """
        return self.plan_system_prompt

    def set_is_remote_sdk(self, value: bool) -> None:
        """Mark this plan as originating from a remote SDK session.

        Args:
            value: True if the session was started from the remote SDK, False otherwise
        """
        self.plan_is_remote_sdk = bool(value)

    def get_is_remote_sdk(self) -> bool:
        """Return whether this plan was started from a remote SDK session.

        Returns:
            True if the session originates from the remote SDK, False otherwise
        """
        return self.plan_is_remote_sdk

    def set_use_only_remote_skill(self, value: bool) -> None:
        """When True, AgentToolPreparer will present only RemoteSkill for remote SDK sessions.

        Args:
            value: True to restrict tool selection to RemoteSkill only
        """
        self.plan_use_only_remote_skill = bool(value)

    def get_use_only_remote_skill(self) -> bool:
        """Return whether the tool selection is restricted to RemoteSkill only.

        Returns:
            True if only RemoteSkill should be offered to the LLM
        """
        return self.plan_use_only_remote_skill

    def declare_sample_as(self, good:bool, msg:str|None=None, auto_generated:bool=False, target_guid:str|None=None) -> None:
        guid_to_use = target_guid if target_guid is not None else self.plan_guid
        if (guid_to_use is None):
            log(WARNING, "declare_sample_as() called but plan_guid is None, skipping")
            return
        
        cache_entry_service = get_cache_entry_service()
        marked = cache_entry_service.mark_plan_cache_entries_as_sample(guid_to_use, good, auto_generated)
        if not marked:
            log(ERROR, f"Failed to mark CacheEntry nodes for plan_guid={guid_to_use}")
            return
        
        log(INFO, f"Marked CacheEntry nodes as good={good}, auto_generated={auto_generated} for plan_guid={guid_to_use}")

        # take note of optional message
        if msg is not None:
            last_goal = self.get_last_goal([GoalStatus.NOT_STARTED])
            messages = last_goal.build_messages(agent=None, purpose=None, include_focus=True, include_rag=True, include_images=False)

            created = cache_entry_service.create_feedback_message(
                guid_to_use,
                good,
                msg,
                self.plan_goal.full_str(recursive=True),
                json.dumps(messages),
            )
            if not created:
                log(ERROR, f"Failed to create FeedbackMessage for plan_guid={guid_to_use}")
                return
            
            log(INFO, f"Created FeedbackMessage for plan_guid={guid_to_use}")

    def rewind(self) -> None:
        """Remove the last two messages from the plan; erase the cache entries for the last plan guid."""
        if (self.plan_guid is None):
            return
        
        # delete the last two messages from the plan
        last_goals_to_remove = plan_chat_template
        last_goal_targets_to_remove = [g.goal_target for g in last_goals_to_remove]
        self.plan_goal.remove_goals(last_goal_targets_to_remove, position="last", status=GoalStatus.COMPLETED)

        # erase the cache entries for the last plan guid
        get_cache_entry_service().delete_cache_entry(guid=self.plan_guid)

    def erase_cache_entries(self, guid: Optional[str]) -> None:
        """Delete every cache entry created or last executed under the given plan GUID, so that a re-run cannot be replayed from the cache."""
        if guid is None:
            return

        try:
            get_cache_entry_service().delete_plan_cache_entries(guid)
            log(INFO, f"Erased the cache entries of plan_guid={guid}")

        except Exception as e:
            log(ERROR, f"Failed to erase the cache entries of plan_guid={guid}: {e}\n{traceback.format_exc()}")

    def discard_pending_turn(self) -> int:
        """
        Remove the root-level fragment whose AgentUser is currently awaiting input -- typically the one that just
        received the /restart command -- including its leading Focusser and every later goal of the fragment.
        Async goals of the removed fragment are cancelled. Returns the number of removed top-level goals.
        """
        removed_goals: List[Goal] = []

        with self.plan_lock:
            subgoals: List[Goal] = self.plan_goal.goal_subgoals
            user_index: Optional[int] = next(
                (
                    index for index in range(len(subgoals) - 1, -1, -1)
                    if subgoals[index].goal_target_agent_name == "AgentUser" and subgoals[index].goal_status == GoalStatus.IN_PROGRESS
                ),
                None
            )
            if user_index is not None:
                start_index: int = user_index
                if user_index > 0 and subgoals[user_index - 1].goal_target_agent_name == "AgentFocusser":
                    start_index = user_index - 1
                removed_goals = subgoals[start_index:]
                del subgoals[start_index:]

        for removed_goal in removed_goals:
            for goal in removed_goal.get_subtree_goals():
                if not goal.cancel_async_execution():
                    log(WARNING, f"discard_pending_turn: async goal '{goal.goal_title}' did not stop in time")

        return len(removed_goals)

    def export_training_data(self, hf_dataset_name: str, enable_translations: bool = False, agent_name: str = None) -> None:
        # 1. identify all the cache entries that are associated with agents and agent purposes
        # 2. export the training data to a file for each of the cache entries matching agents+purposes

        # identify agents and agent purposes
        cache_entry_service = get_cache_entry_service()
        agents_purposes = cache_entry_service.get_distinct_agents_and_purposes(agent_name)
        log(DEBUG, f"Agents & Purposes: {agents_purposes}")

        # create directory if it does not exist
        path = "training_data"
        os.makedirs(path, exist_ok=True)
        
        all_records = []
        
        for agent_purpose in agents_purposes:
            agent = str(agent_purpose[0]).replace(":", "_")
            purpose = agent_purpose[1]

            try:
                cache_entries = cache_entry_service.get_training_data_cache_entries(
                    agent=agent,
                    purpose=purpose,
                    is_good=True,
                    category="chatbot",
                )
                if not cache_entries:
                    log(DEBUG, f"{agent_purpose}: No records found, skipping...")
                    continue

                # with open(f"{path}/training_data_{agent}_{purpose}.jsonl", "w", encoding="utf-8") as file:
                for cache_entry in cache_entries:
                    cached_messages = cache_entry["messages"]
                    cached_response = cache_entry["response"]

                    if cached_messages.startswith("[") or cached_messages.startswith("{"):
                        messages = json.loads(cached_messages.encode('utf-8'))
                    else:
                        log(ERROR, f"{agent_purpose}: ERROR: messages not in JSON format")
                        continue

                    if cached_response.startswith("{"):
                        response = json.loads(cached_response.encode('utf-8'))
                    else:
                        response = cached_response

                    # retrieve messages and the final assistant message, called 'response'
                    if isinstance(response, dict) and "tool_call" in response:
                        response = {"tool_calls": response["tool_call"]}
                    elif "summary" in response:
                        response = {"content": response}
                    else:
                        response = {"content": response}
                    

                    # In message, delete examples between <EXAMPLES> and </EXAMPLES>, as well as the HINT=[[...]]
                    for message in messages:
                        if "content" in message and isinstance(message["content"], str):
                            message["content"] = re.sub(r'<EXAMPLES>.*?</EXAMPLES>', '', message["content"], flags=re.DOTALL)
                            message["content"] = re.sub(r'HINT=\[\[.*?\]\]', '', message["content"])
                        
                    # add the response_format output to the last line of the messages, if used
                    if "response_format" in cache_entry:
                        cached_response_format = cache_entry["response_format"]
                        last_assistant = messages[-1]
                        if last_assistant["role"] == "user":
                            last_assistant["content"] += f"""You MUST provide the response in JSON in the following json format.
    Do NOT output any characters before or after the JSON object.
    Note that if a parameter is labeled as optional, it means that it is not mandatory to include it in the response.
    If a parameter has this format: regexp(expression), it means that the value must match the regular expression indicated inside the parenthesis.
    For example, if the parameter is: regexp(OK|FAIL), the value must be either 'OK' or 'FAIL'.
    The JSON must be compact, with no unnecessary whitespace, newlines, or carriage returns.
    The response MUST be in the following format:
    '{cached_response_format}'"""

                    # add the final assistant message
                    response["role"] = "assistant"
                    messages.append(response)

                    # Build base training item
                    training_line = {"messages": messages}
                    if "tools" in cache_entry:
                        training_line["tools"] = json.loads(cache_entry["tools"])

                    # NOT IN USE:
                    # structured_output = cache_entry["structured_output"]
                    
                    # Handle translations
                    json_lines = []
                    if enable_translations and not cache_entry.get("auto_generated"):
                        translated = self.translate_line(training_line)
                        for lang in ["??", "en", "fr", "de", "it"]:
                            if lang in translated:
                                json_lines.append(translated[lang])
                    else:
                        json_lines = [training_line["messages"]]

                    # Create final records
                    for msg_chain in json_lines:
                        last_asst = next((m for m in reversed(msg_chain) if m["role"] == "assistant"), None)
                        if not last_asst:
                            continue  # Skip invalid entries

                        answers = last_asst.get("content") or last_asst.get("tool_calls")

                        all_records.append({
                            "agent": agent,
                            "purpose": purpose,
                            "messages": msg_chain,
                            "answers": last_asst.get("content") or last_asst.get("tool_calls")
                        })
                        #file.write(f"{json.dumps(final_line)}\n")
     
            except Exception as e:
                log(ERROR, f"{agent_purpose}: Error: {e}\n{traceback.format_exc()}")

        # push to hf hub 
        log(INFO, f"Pushing training data to Hugging Face Hub")
        try:
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            if hf_token is None:
                log(ERROR, "HF_TOKEN not found, please set it in the environment variables")

            log(INFO, f"Training data pushed to Hugging Face Hub")

            valid_records = []
            for record in all_records:
                # Convert any non-serializable objects
                if isinstance(record["answers"], dict):
                    answers = json.dumps(record["answers"])
                else:
                    answers = record["answers"]

                clean_record = {
                    "agent": str(record["agent"]),
                    "purpose": str(record["purpose"]),
                    "messages": json.dumps(record["messages"]),  # Serialize nested structures
                    "answers": json.dumps(record["answers"])
                    }
                valid_records.append(clean_record)
            
            from datasets import Dataset
            dataset = Dataset.from_list(valid_records)
            dataset.push_to_hub(hf_dataset_name, token=hf_token)

            
        except Exception as e:
            log(ERROR, f"Error pushing training data to Hugging Face Hub: {e}\n{traceback.format_exc()}")

        # save into one jsonl file
        output_path = os.path.join(path, "training_data_all.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for record in all_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")  

        return

    def translate_content(self, content):
        if content is None:
            return None
        return self.plan_translator_agent.translate([content])

    def translate_line(self, training_line:dict) -> dict:
        # instantiate the translator agent
        if self.plan_translator_agent is None:
            self.plan_translator_agent = import_class("src.core.agents.AgentTranslator", "AgentTranslator")("AgentTranslator")

        # translate the last two messages of the training line to the target language, then return the whole training line
        messages = training_line["messages"]

        # identify the last two messages, one is with role "assistant" and the other is with role "user"
        message_assistant = None
        message_user = None
        message_assistant_index = None
        message_user_index = None
        hint_user = None
        for index, message in enumerate(reversed(messages)):
            if message["role"] == "assistant":
                message_assistant = message
                message_assistant_index = len(messages) - index - 1

            elif message["role"] == "user":
                message_user = message
                message_user_index = len(messages) - index - 1

            if message_assistant is not None and message_user is not None:
                break

        if message_assistant is None or message_user is None:
            training_line_translated = { "??" : messages }
            return training_line_translated

        # remove the PRESERVE_ENGLISH from the message content. Format: PRESERVE_ENGLISH=[[hint]]
        message_assistant_content = message_assistant["content"] if "content" in message_assistant else None
        message_user_content = message_user["content"]
        import re

        if "PRESERVE_ENGLISH" in message_user_content:
            # save the PRESERVE_ENGLISH in original language, avoid translating it
            # remove the PRESERVE_ENGLISH from the message content
            match = re.search(r'PRESERVE_ENGLISH=\[\[(.*?)\]\]', message_user_content, re.DOTALL)
            if match:
                hint_user = match.group(1)
                # remove the hint from the message content
                message_user_content = message_user_content.replace(f"PRESERVE_ENGLISH=[[{hint_user}]]", "[[]]")
            else:
                hint_user = None  # or handle the case where no hint is found

        # call the LLM to perform the translation (in parallel to be faster)
        # message_assistant_translated = self.plan_translator_agent.translate([message_assistant_content])
        # message_user_translated = self.plan_translator_agent.translate([message_user_content])
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_assistant = executor.submit(self.translate_content, message_assistant_content)
            future_user = executor.submit(self.translate_content, message_user_content)
            
            message_assistant_translated = future_assistant.result()
            message_user_translated = future_user.result()

        # restore the hints
        if hint_user is not None:
            for translation in message_user_translated[0]["translations"]:
                translation["content"] = translation["content"].replace("[[]]", f"PRESERVE_ENGLISH=[[{hint_user}]]")

        # use the translated messages to create the training line
        translated_lines = {}
        for lang in ["en", "de", "fr", "it"]:
            # find the original message in the training line with flag "assistant" or "user"
            translated_lines[lang] = messages.copy()

            # substitute the content of the assistant message with the translated content
            translated_lines[lang][message_assistant_index] = messages[message_assistant_index].copy()
            # Iterate through the translations to find the one matching the language
            if message_assistant_translated is not None:
                for translation in message_assistant_translated[0]["translations"]:
                    if translation["language"] == lang:
                        translated_lines[lang][message_assistant_index]["content"] = translation["content"]
                        break

            # substitute the content of the user message with the translated content
            translated_lines[lang][message_user_index] = messages[message_user_index].copy()
            for translation in message_user_translated[0]["translations"]:
                if translation["language"] == lang:
                    translated_lines[lang][message_user_index]["content"] = translation["content"]
                    break
            # add seperate response for each language
            translated_lines[lang]

        return translated_lines

    def get_last_goal(self, statuses:List[GoalStatus]) -> Goal:
        return self.plan_goal.get_last(statuses=statuses)
        
    def get_last_failed(self) -> Goal:
        return self.plan_goal.get_last_recursive(statuses=[GoalStatus.FAILED])
        
    def add_plan_from_prompt(self, prompt:str, after_id:Optional[Union[str, List[int]]]=None):
        planfragment = plan_from_prompt(self, prompt)
        with self.plan_lock:
            inserted = self.plan_goal.insert_goals(planfragment, after_id, instantiated_plan=None)

        return planfragment if inserted else None

    def add_plan_from_template(self, plan_template_name:str, prompt:str, after_id:Union[str, List[int]]) -> Optional[List['Goal']]:
        from src.core.llm.manager import LlmManager
        templates = LlmManager.instance().get_agent_plans_template()
        if plan_template_name not in templates:
            log(WARNING, f"add_plan_from_template: unknown plan template '{plan_template_name}'")
            return None
        template = templates[plan_template_name]["template"]
        goals = instantiate_plan(self, template)
        for g in goals:
            if g.goal_prompt is None:
                g.goal_prompt = prompt
        # wrap the template goals in an ONLY_SUBGOALS container with a Consolidator as the last child.
        # the Consolidator depends only on the last goal if it is an AgentAssistant (pass-through),
        # or on all template goals otherwise (full aggregation). The container then inherits the
        # Consolidator's response summary, making the assistant's answer visible in subsequent rounds.
        # NOTE: all templates include a Consolidator; do not add a second one to avoid duplicated assistant messages.
        template_already_has_consolidator = (goals and goals[-1].goal_target == GoalTarget.CONSOLIDATOR)
        if not template_already_has_consolidator:
            # find the last non-Consolidator goal to determine assistant pass-through vs full aggregation
            last_non_consolidator = next((g for g in reversed(goals) if g.goal_target != GoalTarget.CONSOLIDATOR), None)
            last_is_assistant = (last_non_consolidator is not None and last_non_consolidator.goal_target_agent_name is not None and last_non_consolidator.goal_target_agent_name.startswith("AgentAssistant"))
            if last_is_assistant:
                consolidator_deps = [-1]
            else:
                consolidator_deps = [-(i + 1) for i in range(len(goals))]
            consolidator = Goal(self, None, "Consolidator", None, GoalTarget.CONSOLIDATOR, None, consolidator_deps)
            goals_with_consolidator = goals + [consolidator]
        else:
            goals_with_consolidator = goals
        container = Goal(self, None, plan_template_name, None, GoalTarget.ONLY_SUBGOALS, None, None)
        container.insert_goals(goals_with_consolidator, after_id=None, instantiated_plan=plan_template_name)
        with self.plan_lock:
            inserted = self.plan_goal.insert_goals([container], after_id, instantiated_plan=plan_template_name)
        return [container] if inserted else None

    def add_plan_chat(self, after_id:Union[str, List[int]]) -> List['Goal']:
        planfragment = plan_chat(self)

        # chain each goal to the one immediately before it within THIS fragment only;
        # the first goal (index 0) must stay dependency-free, otherwise it would reach back
        # into the previous turn's tail goal and inherit a permanent CANCELLED status from it,
        # causing every subsequent turn to be auto-cancelled forever by the dependency scheduler
        for index, goal in enumerate(planfragment):
            if len(goal.goal_dependent_upon) == 0 and index > 0:
                goal.goal_dependent_upon = [-1]

        # verify if the plan is already in the plan, not executed yet. We do as follows:
        # we loop in reverse order, and if we find a goal with the same target and status different than NOT_STARTED, then we must insert the new plan
        must_insert = False
        idx = -1
        for g in reversed(planfragment):
            goal = self.plan_goal.goal_subgoals[idx]
            if goal.goal_target != g.goal_target or goal.goal_target_agent_name != g.goal_target_agent_name or goal.goal_status != GoalStatus.NOT_STARTED:
                must_insert = True
                break
            idx -= 1
            if idx < -len(self.plan_goal.goal_subgoals):
                break

        if must_insert:
            with self.plan_lock:
                inserted = self.plan_goal.insert_goals(planfragment, after_id, instantiated_plan=None)

            if inserted:
                log_plan(self, include_rag=False)
                log(DEBUG, "Plan_chat was inserted")
                return planfragment
            else:
                log(DEBUG, "Plan_chat was NOT inserted")
                return None

        else:
            log(DEBUG, "Plan already in the plan, not inserting")
            return None

    def set_goal_status(self, goal_id:Union[str, List[int]], status:GoalStatus) -> None:
        goal = self.plan_goal.get_goal_by_id_chain(goal_id)
        if goal is not None:
            goal.set_goal_status(status, recursive=True)

    def set_goal_summary(self, goal_id:Union[str, List[int]], summary:str, consolidated:bool=False) -> None:
        goal = self.plan_goal.get_goal_by_id_chain(goal_id)
        if goal is not None:
            goal.set_summary(summary, consolidated=consolidated)

    def add_tokens_used(self, llm_response: LlmResponse) -> None:
        # Be tolerant: update whichever token fields are present on the response.
        if llm_response is None:
            return

        with self.plan_lock:
            # Sum completion tokens when available
            if llm_response.completion_tokens is not None:
                self.plan_total_completion_tokens += llm_response.completion_tokens

            # Track the maximum observed prompt tokens (if available)
            if llm_response.prompt_tokens is not None and llm_response.prompt_tokens > self.plan_maximum_tokens:
                self.plan_maximum_tokens = llm_response.prompt_tokens

    def get_dossier_id(self) -> str:
        if (self.plan_dossier_id is None and global_dossier_id is not None):
            self.set_dossier_id(global_dossier_id)
        return self.plan_dossier_id

    def set_dossier_id(self, dossier_id: Union[str, None]) -> bool:
        # the plan will be focussed on dossier_id, and will automatically load all its associated documents
        self.plan_dossier_id = dossier_id
        success = False

        if dossier_id is not None:
            log(INFO, f"Setting dossier for the plan to: '{dossier_id}'")
            dossier_reasoning_document_id = None
            dossier_reasoning_template_id = None

            dossiers_service = get_dossiers_service()
            dossier_details = dossiers_service.get_dossier(dossier_id)
            if dossier_details is None:
                log(ERROR, f"Plan.set_dossier_id: Dossier {dossier_id} not found.")
            else:
                dossier_owned_docids = dossier_details.get("document_ids", [])
                dossier_reasoning_template_ids = dossier_details.get("template_ids", [])
                if dossier_reasoning_template_ids is None or len(dossier_reasoning_template_ids) > 1:
                    log(ERROR, f"Plan.set_dossier_id: Dossier {dossier_id} does not have exactly one reasoning template, cannot set it.")
                elif len(dossier_reasoning_template_ids) == 0:
                    log(INFO, f"Plan.set_dossier_id: Dossier {dossier_id} does not have any reasoning template.")
                else:
                    dossier_reasoning_template_id = dossier_reasoning_template_ids[0]
                    log(DEBUG, f"Plan.set_dossier_id: Dossier {dossier_id} has exactly one reasoning template, setting it: {dossier_reasoning_template_id}")
                dossier_reasoning_document_ids = dossier_details.get("reasoning_ids", [])
                if dossier_reasoning_document_ids is None or len(dossier_reasoning_document_ids) > 1:
                    log(ERROR, f"Plan.set_dossier_id: Dossier {dossier_id} does not have exactly one reasoning document, cannot set it.")
                elif dossier_reasoning_document_ids is not None and len(dossier_reasoning_document_ids) == 1:
                    dossier_reasoning_document_id = dossier_reasoning_document_ids[0]

                # Set ontology if the dossier follows one
                dossier_ontology_id = dossier_details.get("follows_ontology_id")
                if dossier_ontology_id is not None:
                    log(INFO, f"Plan.set_dossier_id: Dossier {dossier_id} follows ontology '{dossier_ontology_id}', setting plan ontology.")
                    self.set_ontology_id(dossier_ontology_id)

                log(INFO, f"Setting dossier ID to {dossier_id}, reasoning template ID to {dossier_reasoning_template_id}, reasoning document ID to {dossier_reasoning_document_id} and owned document IDs to {dossier_owned_docids}")

                # Add reasoning document ID to loaded documents if present, preserving order and uniqueness
                if False:  # DISABLED: plan_loaded_documents should only track explicitly loaded documents, not all dossier documents
                    loaded_docs = dossier_owned_docids.copy()
                    if dossier_reasoning_document_id is not None and isinstance(dossier_reasoning_document_id, str) and len(dossier_reasoning_document_id) > 0:
                        if dossier_reasoning_document_id not in loaded_docs:
                            loaded_docs.append(dossier_reasoning_document_id)
                    # Remove duplicates while preserving order
                    seen = set()
                    self.plan_loaded_documents = [x for x in loaded_docs if not (x in seen or seen.add(x))]
                    
                # Ensure plan_document_template_ids is always a list
                if dossier_reasoning_template_id is not None:
                    self.plan_document_template_ids = [dossier_reasoning_template_id]
                else:
                    self.plan_document_template_ids = []
                self.plan_reasoning_document_id = dossier_reasoning_document_id

                success = True
        else:
            # remove the dossier id from the loaded documents
            self.plan_loaded_documents = [doc_id for doc_id in self.plan_loaded_documents if doc_id != dossier_id]
            # reset the reasoning document list
            self.plan_reasoning_document_id = None
            self.plan_document_template_ids = []
            success = True

        return success
    
    def get_session_dossier_id(self) -> str:
        if (self.plan_session_dossier_id is None and global_session_dossier_id is not None):
            self.set_session_dossier_id(global_session_dossier_id)
        return self.plan_session_dossier_id
    
    def set_session_dossier_id(self, dossier_id: str) -> None:
        self.plan_session_dossier_id = dossier_id

    def get_ontology_id(self) -> Optional[str]:
        """
        Returns the ontology ID for the current plan. If the ontology ID is set to "DEFAULT",
        it will be treated as None.
        """
        ontology_id: Optional[str] = self.plan_ontology_id
        if ontology_id == ONTOLOGY_DEFAULT_ID:
            ontology_id = None

        return ontology_id

    def set_ontology_id(self, ontology_id: Optional[str]) -> None:
        self.plan_ontology_id = ontology_id

    def get_datasets(self) -> Optional[List[dict]]:
        return self.plan_datasets
    
    def fetch_datasets(self, dataset_ids: Optional[List[str]] = None) -> List[dict]:
        # Delegate to dataset service which centralizes Neo4j access and normalization
        dataset_service: DatasetInterface = get_dataset_service()
        try:
            return dataset_service.get_datasets(dataset_ids)
        
        except Exception as e:
            log(ERROR, f"Plan.fetch_datasets: Error fetching datasets from service: {e}\n{traceback.format_exc()}")
            return []

    def add_datasets(self, dataset_ids:Optional[Union[List[dict],List[str]]]) -> None:
        if dataset_ids is None:
            return

        if self.plan_datasets is None:
            self.plan_datasets = []

        # Normalize input: if caller provided a list of strings (ids), fetch full dataset dicts
        to_add: List[Union[dict, str]] = dataset_ids
        if len(dataset_ids) > 0 and isinstance(dataset_ids[0], str):
            try:
                to_add = self.fetch_datasets(dataset_ids)
            except Exception:
                # fallback: keep original strings if fetch fails
                to_add = dataset_ids

        # Build set of existing ids for deduplication
        existing_ids = set()
        for existing in self.plan_datasets:
            if isinstance(existing, dict):
                if existing.get("id") is not None:
                    existing_ids.add(str(existing.get("id")))
            else:
                existing_ids.add(str(existing))

        for ds in to_add:
            if isinstance(ds, dict):
                dsid = ds.get("id")
                if dsid is None:
                    # if no id, append the dict as-is to avoid losing information
                    if ds not in self.plan_datasets:
                        self.plan_datasets.append(ds)
                else:
                    if str(dsid) not in existing_ids:
                        self.plan_datasets.append(ds)
                        existing_ids.add(str(dsid))
            else:
                dsid = str(ds)
                if dsid not in existing_ids:
                    # try to fetch full dict; if not possible, store the id string
                    try:
                        fetched = self.fetch_datasets([dsid])
                        if fetched and len(fetched) > 0:
                            self.plan_datasets.append(fetched[0])
                        else:
                            self.plan_datasets.append(dsid)
                        existing_ids.add(dsid)
                    except Exception:
                        self.plan_datasets.append(dsid)
                        existing_ids.add(dsid)

    def set_datasets(self, dataset_ids:Optional[Union[List[dict],List[str]]]) -> None:
        self.plan_datasets = None
        if dataset_ids is not None:
            self.add_datasets(dataset_ids)

    def add_documents(self, document_ids:List[str]) -> None:
        if self.plan_document_ids is None:
            self.plan_document_ids = []
        # Only add document IDs that are not already in the list
        for doc_id in document_ids:
            if doc_id not in self.plan_document_ids:
                self.plan_document_ids.append(doc_id)

    def add_document_templates(self, document_ids:List[str]) -> None:
        if self.plan_document_template_ids is None:
            self.plan_document_template_ids = []
        # Only add document IDs that are not already in the list
        for doc_id in document_ids:
            if doc_id not in self.plan_document_template_ids:
                self.plan_document_template_ids.append(doc_id)

    def add_loaded_documents(self, document_ids:List[str]) -> None:
        if self.plan_loaded_documents is None:
            self.plan_loaded_documents = []
        # Only add document IDs that are not already in the list
        for doc_id in document_ids:
            if doc_id not in self.plan_loaded_documents:
                self.plan_loaded_documents.append(doc_id)

    def clear_loaded_documents(self) -> None:
        """Clear all loaded documents from the plan."""
        self.plan_loaded_documents = []

    def remove_loaded_documents(self, document_ids: Union[str, list, None]) -> tuple:
        """Remove specific documents from plan_loaded_documents.
        
        Args:
            document_ids: document ID string or list of document IDs to remove
        
        Returns:
            Tuple of (list of removed doc IDs, list of warning messages for non-existent docs)
        """
        if document_ids is None:
            return [], []
        
        if isinstance(document_ids, str):
            document_ids = [document_ids]
        
        if self.plan_loaded_documents is None:
            self.plan_loaded_documents = []
        
        removed = []
        warnings = []
        for doc_id in document_ids:
            if doc_id in self.plan_loaded_documents:
                self.plan_loaded_documents.remove(doc_id)
                removed.append(doc_id)
            else:
                warnings.append(f"Warning: Document '{doc_id}' not in loaded documents")
        
        return removed, warnings

    def remove_documents(self, document_ids: List[str]) -> None:
        """Remove document IDs from the plan's document list"""
        if self.plan_document_ids is not None:
            self.plan_document_ids = [doc_id for doc_id in self.plan_document_ids if doc_id not in document_ids]

    def remove_document_templates(self, document_ids: List[str]) -> None:
        """Remove document template IDs from the plan's document template list"""
        if self.plan_document_template_ids is not None:
            self.plan_document_template_ids = [doc_id for doc_id in self.plan_document_template_ids if doc_id not in document_ids]

    def get_reasoning_document_id(self) -> str:
        return self.plan_reasoning_document_id

    def set_reasoning_document_id(self, document_id: str) -> None:
        self.plan_reasoning_document_id = document_id

    def get_document_ids(self) -> List[str]:
        """Get a copy of the document IDs list"""
        return self.plan_document_ids.copy() if self.plan_document_ids else []

    def get_document_template_ids(self) -> List[str]:
        return self.plan_document_template_ids if self.plan_document_template_ids else []

    def get_loaded_document_ids(self) -> List[str]:
        """Get a copy of the loaded documents list"""
        return self.plan_loaded_documents.copy() if self.plan_loaded_documents else []

    def has_document(self, document_id: str) -> bool:
        """Check if a document ID exists in the plan"""
        return self.plan_document_ids is not None and document_id in self.plan_document_ids

    def has_document_template(self, document_id: str) -> bool:
        """Check if a document template ID exists in the plan"""
        return self.plan_document_template_ids is not None and document_id in self.plan_document_template_ids

    def has_loaded_document(self, document_id: str) -> bool:
        """Check if a loaded document ID exists in the plan"""
        return self.plan_loaded_documents is not None and document_id in self.plan_loaded_documents

    def clear_documents(self) -> None:
        """Clear all document IDs"""
        self.plan_document_ids = []

    def clear_document_templates(self) -> None:
        """Clear all document template IDs"""
        self.plan_document_template_ids = []

    def add_plan_debug_message(self, messages: Optional[List], command: str=None, goal_target=GoalTarget.DEBUG, goal_target_agent_name: str=None) -> List['Goal']:
        inserted = False
        target_agent = None
        
        if goal_target_agent_name is not None:
            target_agent = import_class(f"src.core.agents.{goal_target_agent_name}", goal_target_agent_name)
        goal = Goal(self, None, "Debug Message", None, goal_target, target_agent, [])

        # html formatting
        goal.goal_summary = "<pre>\n"

        # add the initial command
        if command is not None:
            goal.goal_summary += "<p><strong>" + command + "</strong></p>"

        # add the debug messages
        if messages is not None:
            if len(messages) == 1:
                goal.goal_summary += messages[0]
            else:
                goal.goal_summary += json.dumps(messages, indent=4)
        
        # html formatting
        goal.goal_summary += "\n</pre>"
        
        last_goal = self.plan_goal.get_last(statuses=[GoalStatus.COMPLETED, GoalStatus.FAILED])
        if last_goal is not None:
            with self.plan_lock:
                inserted = self.plan_goal.insert_goals([goal], after_id=last_goal.get_id_chain(), instantiated_plan=None)

        return [goal] if inserted else None

    def update_focus(self, focus:str) -> None:
        self.plan_focus = focus

    def increment_revision(self) -> int:
        self.plan_revision += 1
        return self.plan_revision

    def query(self, query: Optional[str], provide_answer:bool=False, max_results: Optional[int]=5) -> bool:
        # perform a knowledge graph search for the given query, returning top_k results
        inserted:bool = False

        try:
            # prepare query parameters
            goal_parameters:dict = None
            plan_name:str = None

            if provide_answer:
                plan_name = "plan_rag_assistant_and_answer"
                goal_parameters = {
                    "prompts": [query],
                    "max_results": max_results
                }
            else:
                plan_name = "plan_rag_assistant_no_answer"
                goal_parameters = {
                    "thalabus_query": query,        # the query is provided directly; prompt augmentation will skip building the query
                    "max_results": max_results
                }

            # prepare query goal with the plan_query_template
            get_plan_function = import_function("src.core.agents.AgentPlanner", "get_plan")
            if get_plan_function is None:
                log(ERROR, "Error performing knowledge graph search: get_plan function not found")
                return False
            plan_query_template:List[Goal] = get_plan_function(plan_name=plan_name)

            if plan_query_template is None or len(plan_query_template) == 0:
                log(ERROR, "Error performing knowledge graph search: plan_rag_assistant_and_answer not found or empty")
                return False
            
            # instantiate the plan from the template, inject the query parameters
            plan_query:List[Goal] = instantiate_plan(self, plan_query_template)
            if provide_answer:
                # Only inject the prompt into the first goal (AgentPromptAugmenter).
                # Subsequent goals (AgentRAG, etc.) must walk the dependency chain to
                # pick up the thalabus_query produced by AgentPromptAugmenter.
                plan_query[0].goal_prompt = goal_parameters
            else:
                # thalabus_query is already structured; inject into every goal so each
                # agent can read it directly without walking the dependency chain.
                for goal in plan_query:
                    goal.goal_prompt = goal_parameters

            # prepare a single goal with the plan as subgoals
            plan_query_goal = Goal(self, None, "Command /query", "Issued directly with the command /query", GoalTarget.ONLY_SUBGOALS, None, [-2])
            plan_query_goal.goal_subgoals = plan_query
            for subgoal in plan_query_goal.goal_subgoals:
                subgoal.goal_parent = plan_query_goal

            with self.plan_lock:
                log_plan(self)
                inserted = self.plan_goal.insert_goals([plan_query_goal], after_id="last", instantiated_plan=plan_name)
                log_plan(self)

        except Exception as e:
            log(ERROR, f"Error performing knowledge graph search: {e}\n{traceback.format_exc()}")
        
        return inserted

    def analyse_query(self, query: Optional[str]) -> bool:
        """
        Analyze a thalabus query and return category breakdowns for matching documents.
        
        Args:
            query: The thalabus query string to analyze
            
        Returns:
            Formatted string with analysis results (pretty-printed JSON)
        """
        inserted:bool = False

        try:
            # prepare query parameters
            goal_parameters:dict = None
            plan_name:str = "plan_rag_analyse_query"
            goal_parameters = {
                "prompts": [query],
                "query_analyse": True
            }

            # prepare query goal with the plan_query_template
            get_plan_function = import_function("src.core.agents.AgentPlanner", "get_plan")
            if get_plan_function is None:
                log(ERROR, "Error performing knowledge graph search: get_plan function not found")
                return False
            plan_query_analyse_template:List[Goal] = get_plan_function(plan_name=plan_name)

            if plan_query_analyse_template is None or len(plan_query_analyse_template) == 0:
                log(ERROR, "Error performing knowledge graph search: plan_rag_assistant_and_answer not found or empty")
                return False
            
            # instantiate the plan from the template, inject the search parameters into each goal
            plan_query_analyse:List[Goal] = instantiate_plan(self, plan_query_analyse_template)
            for goal in plan_query_analyse:
                goal.goal_prompt = goal_parameters
                goal.goal_recommended_tools = ["+QueryAnalyser"]

            # prepare a single goal with the plan as subgoals
            plan_query_goal = Goal(self, None, "Command /query-analyse", "Issued directly with the command /query-analyse", GoalTarget.ONLY_SUBGOALS, None, [-2])
            plan_query_goal.goal_subgoals = plan_query_analyse
            for subgoal in plan_query_goal.goal_subgoals:
                subgoal.goal_parent = plan_query_goal

            with self.plan_lock:
                log_plan(self)
                inserted = self.plan_goal.insert_goals([plan_query_goal], after_id="last", instantiated_plan=plan_name)
                log_plan(self)
        
        except Exception as e:
            log(ERROR, f"Error in analyse_query: {e}\n{traceback.format_exc()}")
            return f"Error: Failed to analyze query: {str(e)}"
        
        return inserted

    def start_reasoning(self, template_id: Optional[str] = None) -> bool:
        inserted:bool = False

        if template_id is None:
            dossiers_service: DossiersInterface = get_dossiers_service()
            dossier_id = self.get_dossier_id()
            if dossier_id is not None:
                dossier_details = dossiers_service.get_dossier(dossier_id)
                if dossier_details is not None:
                    dossier_template_ids = dossier_details.get("template_ids", [])
                    if dossier_template_ids is not None and len(dossier_template_ids) == 1:
                        template_id = dossier_template_ids[0]
                        log(DEBUG, f"start_reasoning: Using template_id from dossier {dossier_id}: {template_id}")
                    elif dossier_template_ids is None or len(dossier_template_ids) == 0:
                        log(WARNING, f"start_reasoning: Dossier {dossier_id} has no reasoning template, cannot start reasoning.")
                    else:
                        log(WARNING, f"start_reasoning: Dossier {dossier_id} has multiple reasoning templates, cannot start reasoning.")
                else:
                    log(WARNING, f"start_reasoning: Dossier {dossier_id} not found, cannot start reasoning.")

        if template_id is None:
            return False

        try:
            # prepare reasoning goal with the plan_reasoning_template
            get_plan_function = import_function("src.core.agents.AgentPlanner", "get_plan")
            if get_plan_function is None:
                log(ERROR, "Error starting reasoning: get_plan function not found")
                return False
            plan_reasoning_template:List[Goal] = get_plan_function(plan_name="plan_deep_reasoning")

            if plan_reasoning_template is None or len(plan_reasoning_template) == 0:
                log(ERROR, "Error starting reasoning: plan_deep_reasoning not found or empty")
                return False
            
            # instantiate the plan from the template, inject the template_id parameter into each goal
            plan_reasoning:List[Goal] = instantiate_plan(self, plan_reasoning_template)
            for goal in plan_reasoning:
                goal.goal_prompt = {
                    "template_id": template_id
                }

            # prepare a single goal with the plan as subgoals
            plan_reasoning_goal = Goal(self, None, "Command /reason", "Issued directly with the command /reason", GoalTarget.ONLY_SUBGOALS, None, [-2])
            plan_reasoning_goal.goal_subgoals = plan_reasoning
            for subgoal in plan_reasoning_goal.goal_subgoals:
                subgoal.goal_parent = plan_reasoning_goal

            with self.plan_lock:
                log_plan(self)
                inserted = self.plan_goal.insert_goals([plan_reasoning_goal], after_id="last", instantiated_plan="plan_reasoning")
                log_plan(self)

        except Exception as e:
            log(ERROR, f"Error starting reasoning: {e}\n{traceback.format_exc()}")

        return inserted

    def run(self) -> dict:
        response = self.plan_goal.run()
        return response

    def cancel_llm_requests(self, timeout: float = 5.0) -> bool:
        cancelled: bool = False

        if self.plan_goal is not None:
            active_goals = self.plan_goal.get_active_goals()

            if active_goals:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_goals)) as executor:
                    cancellation_results = list(
                        executor.map(
                            lambda goal: goal.cancel_llm_requests(timeout=timeout),
                            active_goals
                        )
                    )
                cancelled = any(cancellation_results)

        return cancelled


log_end_file(__file__)

#region unit test
if __name__ == '__main__':
    from src.utils.unit_test import init_unit_test
    init_unit_test()
    plan = Plan("What is the capital of France?")
    print(plan)

    from src.core.llm import HF_DATASET_NAME_DEFAULT
    plan.export_training_data(HF_DATASET_NAME_DEFAULT)
#endregion unit test
