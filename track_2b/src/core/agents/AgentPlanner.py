import uuid
from src.core.base.llm_response import LlmResponse
from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import json
import html
import re
from typing import Dict, Optional, Tuple, Union, List

from src.config.constants import DOCUMENT_ID_PREFIX, ENABLE_AGENT_REVIEWER, ENABLE_PATH_RAG_DEFAULT, ENABLE_RAG_QUERY_SPLITTER
from src.config.app_config import app_config
from src.core.base.enums import AgentReasoningEnum
from src.core.extensions.skillset_management import get_filtered_tools, import_class
from src.core.agents.Agent import Agent
from src.core.thalaplan.goal.enums import GoalStatus, GoalTarget
from src.core.thalaplan.Plan import Plan
from src.core.thalaplan.goal.Goal import Goal
from src.core.thalaplan.plan_utils import get_goals_from_plan_template
from src.core.llm import LlmManager, MODEL_HIGHINTELLIGENCE, DEFAULT_TEMPERATURE_COLD, get_json_from_text, get_yaml_from_text, MAX_COMPLETION_TOKENS_PLANNING
from src.utils.log import log, INFO, DEBUG, WARNING, ERROR, FATAL, log_plan


#region globals
MODEL_PLANNER = MODEL_HIGHINTELLIGENCE
AGENTPLANNER_FINETUNE_ENABLE = True
MIN_INTELLIGENCE_LEVEL_FOR_BACKUP_PLAN = 0.5
MINIMUM_REASONING_AGENT_PLANNER = AgentReasoningEnum.SIMPLE
MAX_PLANNER_ATTEMPTS = 2

# fine-tuning configuration for the AgentPlanner
FINETUNE_PURPOSE = "Finetune"
DEFAULT_SINGLE_GOAL_SEQUENCE_ID = "Step-1"
DEFAULT_SINGLE_GOAL_PROMPT = "Provide an answer to the latest user prompt"

# YAML format example shown to LLM via the system prompt (AgentPlanner.txt).
# No structured_output / response_format is passed to the provider so
# the LLM is free to produce plain-text YAML instead of constrained JSON.
SYSTEM_PROMPT_FORMAT_COMPLEX_YAML = """optional_prebuilt_complex_plan_id: "pre-built-plan-xxx"
optional_complex_plan:
  optional_doc_ids: "docid-xxxxxx,docid-yyyyyy"
  goals:
    - goal_sequence_id: "Step-1"
      goal: "example of textual description of goal 1"
      dependent_on_sequence_ids: []
      recommended_plan: ""
      optional_recommended_tools: []
    - goal_sequence_id: "Step-2"
      goal: "..."
      dependent_on_sequence_ids:
        - "Step-1"
      recommended_plan: ""
      optional_recommended_tools: []
optional_require_exact_time: false
"""
#endregion globals

#region plans configuration
def get_path_rag_enabled() -> bool:
    llm_settings = LlmManager.instance().get_llm_model_configurations()
    features = llm_settings.get("features", {}) if isinstance(llm_settings, dict) else {}
    enabled = bool(features.get("enable_path_rag", ENABLE_PATH_RAG_DEFAULT))

    return enabled

def init_agent_planner() -> None:
    # Initialize all agent classes required for the planner
    path_rag_enabled:bool = get_path_rag_enabled()
    log(INFO, f"AgentPlanner path RAG setting is {path_rag_enabled}")

    agent_internet = import_class("src.core.agents.AgentInternet", "AgentInternet")
    agent_prompt_augmenter = import_class("src.core.agents.AgentPromptAugmenter", "AgentPromptAugmenter")
    agent_assistant = import_class("src.core.agents.AgentAssistant", "AgentAssistant")
    agent_presenter = import_class("src.core.agents.AgentPresenter", "AgentPresenter")
    agent_imager = import_class("src.core.agents.AgentImager", "AgentImager")
    agent_rag = import_class("src.core.agents.AgentRAG", "AgentRAG")
    agent_rag_query_splitter = import_class("src.core.agents.AgentRAGQuerySplitter", "AgentRAGQuerySplitter")
    agent_internet = import_class("src.core.agents.AgentInternet", "AgentInternet")
    if path_rag_enabled:
        agent_rag_kb = import_class("src.core.agents.AgentRagPathKb", "AgentRagPathKb")
    else:
        agent_rag_kb = import_class("src.core.agents.AgentRagKB", "AgentRagKB")
    agent_reranker = import_class("src.core.agents.AgentReranker", "AgentReranker")
    agent_toolpreparer = import_class("src.core.agents.AgentToolPreparer", "AgentToolPreparer")
    agent_toolcaller = import_class("src.core.agents.AgentToolCaller", "AgentToolCaller")
    agent_reviewer = import_class("src.core.agents.AgentReviewer", "AgentReviewer")
    agent_deep_reasoning = import_class("src.core.agents.AgentDeepReasoning", "AgentDeepReasoning")
    agent_deep_reasoning_continuum = import_class("src.core.agents.AgentDeepReasoningContinuum", "AgentDeepReasoningContinuum")
    agent_deepdocumentupdater = import_class("src.core.agents.AgentDeepDocumentUpdater", "AgentDeepDocumentUpdater")
    agent_ontologist = import_class("src.core.agents.AgentOntology", "AgentOntology")
    agent_user = import_class("src.core.agents.AgentUser", "AgentUser")
    agent_mira = import_class("src.core.agents.AgentMira", "AgentMira")
    agent_mira_iterator = import_class("src.core.agents.AgentMiraIterator", "AgentMiraIterator")

    agent_plan_ask_user_then_continue_reasoning = []
    agent_plan_ask_user_goal = Goal(None, None, "Assistant-user", None, GoalTarget.AGENT, agent_user, [])
    agent_plan_ask_user_goal.goal_silent_summary = True
    agent_plan_ask_user_then_continue_reasoning.append(agent_plan_ask_user_goal)
    agent_plan_ask_user_then_continue_reasoning.append(Goal(None, None, "Assistant-user", None, GoalTarget.AGENT, agent_deep_reasoning_continuum, [-1]))
    LlmManager.instance().add_or_update_plan_fragment("plan_ask_user_then_continue_reasoning", {
        "id": "plan_ask_user_then_continue_reasoning",
        "name": "Ask user for input, then continue reasoning",
        "template": agent_plan_ask_user_then_continue_reasoning,
        "description": """This plan asks the user for input, then the deep reasoning is resumed. Typically used during a Deep Reasoning process.""",
        "blacklisted": True
        })

    agent_plan_call_tool_and_answer_template = []
    agent_plan_call_tool_and_answer_template.append(Goal(None, None, "Assistant-toolpreparer", None, GoalTarget.AGENT, agent_toolpreparer, []))
    agent_plan_call_tool_and_answer_template.append(Goal(None, None, "Assistant-answer", None, GoalTarget.AGENT, agent_assistant, [-1]))
    agent_plan_call_tool_and_answer_template.append(Goal(None, None, "Presenter-answer", None, GoalTarget.AGENT, agent_presenter, [-1]))
    agent_plan_call_tool_and_answer_template.append(Goal(None, None, "Consolidator", None, GoalTarget.CONSOLIDATOR, None, [-1]))
    LlmManager.instance().add_or_update_plan_fragment("plan_call_tool_and_answer", {
        "id": "plan_call_tool_and_answer",
        "name": "Call skill(s) and then provide an answer to the user",
        "template": agent_plan_call_tool_and_answer_template,
        "description": """This plan executes one or many tool invocations with already prepared parameters, and then directly calls the Assistant to answer the user."""
        })

    agent_plan_call_tool_but_no_answer_template = []
    agent_plan_call_tool_but_no_answer_template.append(Goal(None, None, "Assistant-toolpreparer", None, GoalTarget.AGENT, agent_toolpreparer, []))
    # agent_plan_call_tool_but_no_answer_template.append(Goal(None, None, "AgentIterator", None, GoalTarget.AGENT, agent_iterator_toolpreparer, [-1]))      
    agent_plan_call_tool_but_no_answer_template.append(Goal(None, None, "Consolidator", None, GoalTarget.CONSOLIDATOR, None, [-1]))
    LlmManager.instance().add_or_update_plan_fragment("plan_call_tool_but_no_answer", {
        "id": "plan_call_tool_but_no_answer",
        "name": "Call skill(s) but do not provide an answer to the user",
        "template": agent_plan_call_tool_but_no_answer_template,
        # "corrective_template": ...,   # this is a plan with only the agent_toolpreparer + agent_iterator_toolpreparer
        "description": """This plan executes the tool invocation with already prepared parameters; in this plan there is no answer to the user.""",
        "assistant_answer": False,
        "dont_replace_for_domains": ["coins"]
        })

    agent_plan_simple_answer_template = []
    agent_plan_simple_answer_template.append(Goal(None, None, "Assistant-answer", None, GoalTarget.AGENT, agent_assistant, []))
    agent_plan_simple_answer_template.append(Goal(None, None, "Presenter-answer", None, GoalTarget.AGENT, agent_presenter, [-1]))
    agent_plan_simple_answer_template.append(Goal(None, None, "Consolidator", None, GoalTarget.CONSOLIDATOR, None, [-1]))
    LlmManager.instance().add_or_update_plan_fragment("plan_simple_answer", {
        "id": "plan_simple_answer",
        "name": "Provide an answer the user",
        "template": agent_plan_simple_answer_template,
        "description": """When the user prompt does not require a tool invocation, and does not require retrieval augmented generation because
either the information to answer the user's prompt is already available, or because there is no need to retrieve additional information
from the avaialble documents in the document repository, this plan calls the Prompt Answering Assistant to directly answer the user."""
        })

    agent_plan_call_tool_and_update_reasoning_document = []
    agent_plan_call_tool_and_update_reasoning_document.append(Goal(None, None, "Assistant-toolpreparer", None, GoalTarget.AGENT, agent_toolpreparer, []))
    agent_plan_call_tool_and_update_reasoning_document.append(Goal(None, None, "DeepReasoning-DocumentUpdater", None, GoalTarget.AGENT, agent_deepdocumentupdater, [-1]))
    agent_plan_call_tool_and_update_reasoning_document.append(Goal(None, None, "Consolidator", None, GoalTarget.CONSOLIDATOR, None, [-1]))
    LlmManager.instance().add_or_update_plan_fragment("plan_call_tool_and_update_reasoning_document", {
        "id": "plan_call_tool_and_update_reasoning_document",
        "name": "Calls skill(s), does not provide an answer to the user, and updates the Reasoning Document",
        "template": agent_plan_call_tool_and_update_reasoning_document,
        "description": """This plan executes the tool invocation; in this plan there is no answer to the user. The reasoning document is updated with the results of the tool invocation.""",
        "assistant_answer": False,
        "blacklisted": True
        })

    agent_plan_internal_answer_template = []
    goal_assistant_silent_answer = Goal(None, None, "Silent-answer", None, GoalTarget.AGENT, agent_assistant, [])
    goal_assistant_silent_answer.goal_silent_summary = True
    agent_plan_internal_answer_template.append(goal_assistant_silent_answer)
    LlmManager.instance().add_or_update_plan_fragment("plan_internal_answer", {
        "id": "plan_internal_answer",
        "name": "Provide an answer to other agents, but not visible to the user",
        "template": agent_plan_internal_answer_template,
        "description": """Generates an internal answer that is later on used by the plan_simple_answer.""",
        "blacklisted": True
        })

    agent_plan_rag_assistant_template = []
    agent_plan_rag_assistant_template.append(Goal(None, None, "Prompt Augmenter", None, GoalTarget.AGENT, agent_prompt_augmenter, []))
    agent_plan_rag_assistant_template.append(Goal(None, None, "Assistant-RAG", None, GoalTarget.AGENT, agent_rag, [-1]))
    agent_plan_rag_assistant_template.append(Goal(None, None, "Assistant-Internet", None, GoalTarget.AGENT, agent_internet, [-2]))
    agent_plan_rag_assistant_template.append(Goal(None, None, "AgentOntology", None, GoalTarget.AGENT, agent_ontologist, None))
    agent_plan_rag_assistant_template.append(Goal(None, None, "Assistant-RAG-PathKB", None, GoalTarget.AGENT, agent_rag_kb, [-1, -4]))
    agent_plan_rag_assistant_template.append(Goal(None, None, "Assistant-Reranker", None, GoalTarget.AGENT, agent_reranker, [-1, -3, -4]))
    agent_plan_rag_assistant_template.append(Goal(None, None, "Assistant-Mira", None, GoalTarget.AGENT, agent_mira, []))
    if ENABLE_RAG_QUERY_SPLITTER:
        agent_plan_rag_assistant_template.append(Goal(None, None, "Assistant-RAGQuerySplitter", None, GoalTarget.AGENT, agent_rag_query_splitter, [-1]))
        agent_plan_rag_assistant_template.append(Goal(None, None, "Assistant-answer", None, GoalTarget.AGENT, agent_assistant, [-1, -2, -3]))
    else:
        agent_plan_rag_assistant_template.append(Goal(None, None, "Assistant-answer", None, GoalTarget.AGENT, agent_assistant, [-1, -2]))
    agent_plan_rag_assistant_template.append(Goal(None, None, "Presenter-answer", None, GoalTarget.AGENT, agent_presenter, [-1]))
    agent_plan_rag_assistant_template.append(Goal(None, None, "Consolidator", None, GoalTarget.CONSOLIDATOR, None, [-1]))
    LlmManager.instance().add_or_update_plan_fragment("plan_rag_assistant_and_answer", {
        "id": "plan_rag_assistant_and_answer",
        "name": "RAG: retrieve information from various sources of information, such as documents, Internet and Knowledge Base, and provide it to the following agents (the information is not directly visible to the user)",
        "template": agent_plan_rag_assistant_template,
        "description": """When the user prompt does not require a tool invocation, but may require retrieval augmented generation because the information to answer the user's prompt is not readily available, this plan calls performs the following searches in parallel:
- optionally scans the document repository and other local sources in order to find relevant text fragments (Retrieval-Augmented Generation, RAG);
- optionally searches in the Internet for relevant sources;
- optionally searches in the Knowledge Base for relevant information.
The results are then consolidated to answer the user."""
        })

    agent_plan_rag_assistant_no_answer_template = []
    agent_plan_rag_assistant_no_answer_template.append(Goal(None, None, "Prompt Augmenter", None, GoalTarget.AGENT, agent_prompt_augmenter, []))
    agent_plan_rag_assistant_no_answer_template.append(Goal(None, None, "Assistant-RAG", None, GoalTarget.AGENT, agent_rag, [-1]))
    agent_plan_rag_assistant_no_answer_template.append(Goal(None, None, "Assistant-Internet", None, GoalTarget.AGENT, agent_internet, [-2]))
    agent_plan_rag_assistant_no_answer_template.append(Goal(None, None, "AgentOntology", None, GoalTarget.AGENT, agent_ontologist, None))
    agent_plan_rag_assistant_no_answer_template.append(Goal(None, None, "Assistant-RAG-PathKB", None, GoalTarget.AGENT, agent_rag_kb, [-1, -4]))
    agent_plan_rag_assistant_no_answer_template.append(Goal(None, None, "Assistant-Reranker", None, GoalTarget.AGENT, agent_reranker, [-1, -3, -4]))
    agent_plan_rag_assistant_no_answer_template.append(Goal(None, None, "Assistant-Mira", None, GoalTarget.AGENT, agent_mira, [-6]))
    if ENABLE_RAG_QUERY_SPLITTER:
        agent_plan_rag_assistant_no_answer_template.append(Goal(None, None, "Assistant-RAGQuerySplitter", None, GoalTarget.AGENT, agent_rag_query_splitter, [-2]))
        agent_plan_rag_assistant_no_answer_template.append(Goal(None, None, "Consolidator", None, GoalTarget.CONSOLIDATOR, None, [-1, -2, -3, -5]))
    else:
        agent_plan_rag_assistant_no_answer_template.append(Goal(None, None, "Consolidator", None, GoalTarget.CONSOLIDATOR, None, [-1, -2, -4]))
    LlmManager.instance().add_or_update_plan_fragment("plan_rag_assistant_no_answer", {
        "id": "plan_rag_assistant_no_answer",
        "name": "RAG: retrieve information from various sources of information, such as documents, Internet and Knowledge Base, and provide it to the following agents (the information is not directly visible to the user)",
        "template": agent_plan_rag_assistant_no_answer_template,
        "description": """When the user prompt does not require a tool invocation, but may require retrieval augmented generation because the information to answer the user's prompt is not readily available, this plan calls performs the following searches in parallel:
- optionally scans the document repository and other local sources in order to find relevant text fragments (Retrieval-Augmented Generation, RAG);
- optionally searches in the Internet for relevant sources;
- optionally searches in the Knowledge Base for relevant information.
The results are available to a further plan_simple_answer that must be scheduled in a separate goal (depending on the goal with this plan_rag_assistant_no_answer)."""
        })

    # plan_mira: orchestrates multi-step MIRA reasoning with AgentMiraIterator
    # Blacklisted - not directly invoked by agents, only called programmatically from AgentMira
    agent_plan_mira_template = []
    agent_plan_mira_template.append(Goal(None, None, "AgentMiraIterator", None, GoalTarget.AGENT, agent_mira_iterator, []))
    agent_plan_mira_template.append(Goal(None, None, "Consolidator", None, GoalTarget.CONSOLIDATOR, None, [-1]))
    LlmManager.instance().add_or_update_plan_fragment("plan_mira", {
        "id": "plan_mira",
        "name": "MIRA: Multi-step reasoning with MIRA tool invocations",
        "template": agent_plan_mira_template,
        "description": """Orchestrates multi-step reasoning by delegating to AgentMiraIterator, which analyzes requests, prepares tool invocations, executes MIRA MCP tools, and consolidates results. Called programmatically from AgentMira when tool invocation is needed. Blacklisted to prevent direct agent invocation.""",
        "blacklisted": True
    })

    # plan_mira_iterator: continuation goal for AgentMiraIterator's iteration loop
    # 1-goal plan containing only AgentMiraIterator; used by _inject_continuation_goal()
    # Blacklisted - programmatically injected as continuation when iteration is needed
    agent_plan_mira_iterator_template = []
    agent_plan_mira_iterator_template.append(Goal(None, None, "AgentMiraIterator-Continuation", None, GoalTarget.AGENT, agent_mira_iterator, []))
    LlmManager.instance().add_or_update_plan_fragment("plan_mira_iterator", {
        "id": "plan_mira_iterator",
        "name": "MIRA Iterator Continuation: continue multi-step reasoning loop",
        "template": agent_plan_mira_iterator_template,
        "description": """1-goal plan containing AgentMiraIterator for continuation in multi-step reasoning loops. Programmatically injected when iteration is needed. Blacklisted to prevent direct agent invocation.""",
        "blacklisted": True
    })

    agent_plan_rag_analyse_query_template = []
    agent_plan_rag_analyse_query_template.append(Goal(None, None, "Prompt Augmenter", None, GoalTarget.AGENT, agent_prompt_augmenter, []))
    goal_rag_analyse_toolpreparer = Goal(None, None, "Assistant-toolpreparer", None, GoalTarget.AGENT, agent_toolpreparer, [-1])
    goal_rag_analyse_toolpreparer.goal_recommended_plan = "+plan_call_tool_but_no_answer"  # mandatory: no AgentAssistant answer after the tool call
    agent_plan_rag_analyse_query_template.append(goal_rag_analyse_toolpreparer)
    # agent_plan_rag_analyse_query_template.append(Goal(None, None, "Assistant-answer", None, GoalTarget.AGENT, agent_assistant, [-1]))  # not needed because the analysis results are directly given to the user with the blazor_component "QueryAnalysisResults" in the ThalabusWebPlugin C# project
    agent_plan_rag_analyse_query_template.append(Goal(None, None, "Consolidator", None, GoalTarget.CONSOLIDATOR, None, [-1]))
    LlmManager.instance().add_or_update_plan_fragment("plan_rag_analyse_query", {
        "id": "plan_rag_analyse_query",
        "name": "Analyse the query and provide an answer to the user",
        "template": agent_plan_rag_analyse_query_template,
        "description": """This plan analyses the query using the Query Analyser skill, then provides the analysis to the user.""",
        })

    agent_plan_deep_reasoning_template = []
    agent_plan_deep_reasoning_template.append(Goal(None, None, "Assistant-DeepReasoningContinuum", None, GoalTarget.AGENT, agent_deep_reasoning_continuum, []))
    LlmManager.instance().add_or_update_plan_fragment("plan_deep_reasoning", {
        "id": "plan_deep_reasoning",
        "name": "Deep Reasoning: Analyse the input and provide a detailed answer",
        "template": agent_plan_deep_reasoning_template,
        "description": """This plan must be used if the user's prompt requires deep reasoning and analysis of the input.""",
        "blacklisted": True
        })

    agent_plan_deep_reasoning_continuum_template = []
    agent_plan_deep_reasoning_continuum_template.append(Goal(None, None, "Assistant-DeepReasoning", None, GoalTarget.AGENT, agent_deep_reasoning, []))
    agent_plan_deep_reasoning_continuum_template.append(Goal(None, None, "Assistant-DeepReasoningContinuum", None, GoalTarget.AGENT, agent_deep_reasoning_continuum, [-1]))
    LlmManager.instance().add_or_update_plan_fragment("plan_deep_reasoning_continuum", {
        "id": "plan_deep_reasoning_continuum",
        "name": "Deep Reasoning Continuum: Analyse the input and provide a detailed answer",
        "template": agent_plan_deep_reasoning_continuum_template,
        "description": """This plan must be used if the user's prompt requires deep reasoning and analysis of the input.""",
        "blacklisted": True
        })

    agent_plan_analyse_image_template = []
    agent_plan_analyse_image_template.append(Goal(None, None, "Assistant-Imager", None, GoalTarget.AGENT, agent_imager, []))
    agent_plan_analyse_image_template.append(Goal(None, None, "Consolidator", None, GoalTarget.CONSOLIDATOR, None, [-1]))
    LlmManager.instance().add_or_update_plan_fragment("plan_analyse_image_template", {
        "id": "plan_analyse_image_template",
        "name": "Analyse an image and provide an answer to the user",
        "template": agent_plan_analyse_image_template,
        "description": """This plan must be used if an image is present as input and the user's prompt requires the analysis of the image.""",
        "blacklisted": True
        })

    # this plan is blacklisted and agents cannot use it directly. This plan is instantiated programmatically when required in some Python functions
    agent_plan_execute_tool_template = []
    agent_plan_execute_tool_template.append(Goal(None, None, "Assistant-toolcaller", None, GoalTarget.AGENT, agent_toolcaller, []))
    agent_plan_execute_tool_template.append(Goal(None, None, "Assistant-answer", None, GoalTarget.AGENT, agent_assistant, [-1]))
    agent_plan_execute_tool_template.append(Goal(None, None, "Consolidator", None, GoalTarget.CONSOLIDATOR, None, [-1]))
    LlmManager.instance().add_or_update_plan_fragment("plan_execute_tool", {
        "id": "plan_execute_tool",
        "name": "Execute the Skill(s) and provide an answer to the user",
        "template": agent_plan_execute_tool_template,
        "description": """This plan executes the tool invocation with already prepared parameters, and answers to the user.""",
        "blacklisted": True
        })

    # this plan is used when a review cycle is needed
    agent_plan_execute_tool_and_review_cycle_template = []
    agent_plan_execute_tool_and_review_cycle_template.append(Goal(None, None, "No/operation", None, GoalTarget.NOOP, None, []))
    agent_plan_execute_tool_and_review_cycle_template.append(Goal(None, None, "Assistant-toolpreparer", None, GoalTarget.AGENT, agent_toolpreparer, [-1]))
    if ENABLE_AGENT_REVIEWER:
        agent_plan_execute_tool_and_review_cycle_template.append(Goal(None, None, "Assistant-reviewer", None, GoalTarget.AGENT, agent_reviewer, [-1]))
    agent_plan_execute_tool_and_review_cycle_template.append(Goal(None, None, "Consolidator", None, GoalTarget.CONSOLIDATOR, None, [-1]))
    LlmManager.instance().add_or_update_plan_fragment("plan_execute_tool_and_review_cycle", {
        "id": "plan_execute_tool_and_review_cycle",
        "name": "Execute the Skill(s) and provide an answer to the user, with review cycle",
        "template": agent_plan_execute_tool_and_review_cycle_template,
        "description": """This plan checks if there's anything to do with regards of the current microplan goal, eventually executes tools, then verifies that the job was done; if not, cycles are carried through until the AgentReviewer is satisfied. No answer to the user is provided.""",
        "blacklisted": True
        })

    # this plan is blacklisted and agents cannot use it directly. This plan is instantiated programmatically when required in some Python functions
    agent_plan_execute_tool_no_answer_template = []
    agent_plan_execute_tool_no_answer_template.append(Goal(None, None, "Assistant-toolcaller", None, GoalTarget.AGENT, agent_toolcaller, []))
    LlmManager.instance().add_or_update_plan_fragment("plan_execute_tool_no_answer", {
        "id": "plan_execute_tool_no_answer",
        "name": "Execute the Skill(s) but do not provide a direct answer to the user; the information from the Skill(s) is given to the next agents",
        "template": agent_plan_execute_tool_no_answer_template,
        "description": """This plan executes the tool invocation with already prepared parameters. No answer to the user is given (presumably it will be given later, as a consolidation of multiple tool invocations).""",
        "blacklisted": True
        })

    return

def replace_plan_with_answering_plan(plan_name:str) -> str:
    if plan_name == "plan_call_tool_but_no_answer":
        return "plan_call_tool_and_answer"
    elif plan_name == "plan_execute_tool_no_answer":
        return "plan_execute_tool"
    elif plan_name == "plan_rag_assistant_no_answer":
        return "plan_rag_assistant_and_answer"
    else:
        return None

def get_plan(plan_name:str) -> List[Goal]:
    templates = LlmManager.instance().get_agent_plans_template()
    if plan_name is None or plan_name not in templates:
        log(ERROR, f"ERROR: AgentPlanner has failed to find the plan '{plan_name}' in the templates.\nThe available templates are: {list(templates.keys())}")
        return None
    
    plan_tool_template = templates[plan_name]["template"]
    return plan_tool_template
#endregion plans configuration


class AgentPlanner(Agent):
    def __init__(self, name: str):
        super().__init__(name)
        self.agent_disable_plan_corrections = False
        self.agent_force_model_temperature = DEFAULT_TEMPERATURE_COLD
        self.agent_minimum_reasoning = MINIMUM_REASONING_AGENT_PLANNER
        self._last_planner_error = None

    def clone(self) -> 'AgentPlanner':
        clone = AgentPlanner(self.agent_name)
        super().clone_to(clone)
        return clone

    def clone_to(self, target) -> None:
        super().clone_to(target)

    @staticmethod
    def _record_planner_output(goal: Goal, raw_response: object) -> None:
        if isinstance(raw_response, str) and raw_response.strip():
            planner_outputs = goal.get_goal_planner_outputs()
            if planner_outputs is None:
                planner_outputs = []
            planner_outputs.append(raw_response)
            goal.set_goal_planner_outputs(planner_outputs)

    def __str__(self):
        return json.dumps(self.to_json())
    
    def to_json(self) -> dict:
        json = super().to_json()
        return json

    def _validate_planner_response(self, response: object) -> Tuple[bool, str]:
        """Validate the parsed planner response and explain invalid structures."""
        if response is None:
            return False, "The LLM response was empty or could not be parsed as YAML."

        if not isinstance(response, dict):
            return False, f"The parsed YAML root must be a mapping, but was {type(response).__name__}."

        if "tool_call" in response:
            return True, ""

        if AGENTPLANNER_FINETUNE_ENABLE and response.get("optional_other") is True:
            return True, ""

        complex_plan = response.get("optional_complex_plan")
        if complex_plan is None:
            if "goals" in response:
                return False, "The 'goals' field is at the YAML root; it must be indented under 'optional_complex_plan'."
            return False, "The response does not contain 'optional_complex_plan' or a tool call."

        if not isinstance(complex_plan, dict):
            return False, "The 'optional_complex_plan' field must be a mapping."

        goals = complex_plan.get("goals")
        if not isinstance(goals, list) or len(goals) == 0:
            return False, "The 'optional_complex_plan.goals' field must be a non-empty list."

        return True, ""

    def _ask_finetuned_planner(self, goal:Goal, messages:List[dict], plan:'Plan') -> Optional[dict]:
        self.raise_if_plan_stopped()
        finetune_configuration = LlmManager.instance().get_llm_model_configuration_for_agent(agent=self, agent_purpose="Finetune")
        if finetune_configuration is None or finetune_configuration.get("id") == "DEFAULT":
            log(INFO, f"{self.agent_name} fine-tuned planner is disabled because the model configuration is set to DEFAULT.")
            return None
        else:
            log(INFO, f"{self.agent_name} fine-tuned planner is enabled with model configuration: {finetune_configuration['id'] if 'id' in finetune_configuration else 'UNKNOWN'}.")

        if not LlmManager.instance().is_agent_model_available(self, FINETUNE_PURPOSE, log_level=WARNING):
            return None

        finetune_prompt = LlmManager.instance().get_system_prompt(agent=self, purpose="Finetune")
        finetune_response:Optional[dict] = None

        if finetune_configuration is not None and finetune_prompt is not None:
            finetune_messages = [dict(message) for message in messages]
            if len(finetune_messages) > 0 and finetune_messages[0].get("role") == "system":
                finetune_messages[0]["content"] = f"<SYSTEM_MESSAGE>\n{finetune_prompt}\n</SYSTEM_MESSAGE>\n"

            try:
                llm_response: LlmResponse = LlmManager.instance().chat_completions(
                    agent=self,
                    purpose=FINETUNE_PURPOSE,
                    plan=goal.goal_plan if goal.goal_plan is not None else plan,
                    goal=goal,
                    messages=finetune_messages,
                    max_completion_tokens=MAX_COMPLETION_TOKENS_PLANNING,
                    cache=False)
                self.raise_if_plan_stopped()
                raw_response = llm_response.response if llm_response else None
                self._record_planner_output(goal, raw_response)
                parsed_response = get_yaml_from_text(raw_response) if isinstance(raw_response, str) else raw_response
                is_valid, validation_error = self._validate_planner_response(parsed_response)
                if is_valid and isinstance(parsed_response, dict) and parsed_response.get("optional_other") is not True:
                    finetune_response = parsed_response
                elif not is_valid:
                    log(WARNING, f"{self.agent_name} fine-tuned planner output was invalid: {validation_error}. LLM response: {raw_response}")

            except Exception as e:
                log(WARNING, f"{self.agent_name} fine-tuned planner call failed: {e}")

        return finetune_response

    def _ask_standard_planner(
        self,
        goal: Goal,
        messages: List[dict],
        tools: Optional[List[Dict]],
        plan: 'Plan',
        model: str
    ) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
        response = None
        raw_response = None
        response_error = None

        for attempt in range(MAX_PLANNER_ATTEMPTS):
            self.raise_if_plan_stopped()
            attempt_messages = list(messages)
            if attempt > 0:
                attempt_messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was invalid and could not be used to create a plan. "
                        f"Validation error: {response_error}\n"
                        "Return the complete corrected YAML now. Preserve all requested values, "
                        "and place goals under optional_complex_plan.goals."
                    )
                })

            try:
                llm_response = LlmManager.instance().chat_completions(
                    agent=self,
                    plan=goal.goal_plan if goal.goal_plan is not None else plan,
                    goal=goal,
                    messages=attempt_messages,
                    tools=tools,
                    response_format=None,
                    structured_output=None,
                    model_input=model,
                    max_completion_tokens=MAX_COMPLETION_TOKENS_PLANNING)
                self.raise_if_plan_stopped()
                raw_response = llm_response.response if llm_response else None
                response = get_yaml_from_text(raw_response) if isinstance(raw_response, str) else raw_response
                is_valid, response_error = self._validate_planner_response(response)
                if is_valid:
                    break

            except Exception as e:
                response = None
                response_error = f"YAML parsing failed: {e}"

            raw_text = raw_response if raw_response is not None else "<no response>"
            response_error = f"{response_error} Raw LLM response: {raw_text}"
            log(WARNING, f"'{self.agent_name}': planner attempt {attempt + 1} failed: {response_error}")

        self.raise_if_plan_stopped()
        return response, raw_response, response_error

    def _finalize_standard_plan_response(
        self,
        goal: Goal,
        response: dict,
        workflow_plan: Optional[str],
        filtered_tools_info: Optional[List[dict]],
        workflows: List[dict],
        backup_plan_default: str
    ) -> Tuple[dict, Optional[List[dict]]]:
        templates = LlmManager.instance().get_agent_plans_template()

        if isinstance(response, dict) and "tool_call" in response:
            result = response["tool_call"]
        elif "optional_complex_plan" in response and response["optional_complex_plan"] is not None and "goals" in response["optional_complex_plan"] and len(response["optional_complex_plan"]["goals"]) > 0:
            if len(response["optional_complex_plan"]["goals"]) == 1:
                if workflow_plan is None \
                and "recommended_plan" in response["optional_complex_plan"]["goals"][0] \
                and response["optional_complex_plan"]["goals"][0]["recommended_plan"] is not None \
                and len(response["optional_complex_plan"]["goals"][0]["recommended_plan"]) > 0 \
                and response["optional_complex_plan"]["goals"][0]["recommended_plan"] in templates \
                and (not "blacklisted" in response["optional_complex_plan"]["goals"][0]["recommended_plan"] or response["optional_complex_plan"]["goals"][0]["recommended_plan"]["blacklisted"]==False):
                    result = {
                        "plan": response["optional_complex_plan"]["goals"][0]["recommended_plan"],
                        "recommended_tools": response["optional_complex_plan"]["goals"][0].get("optional_recommended_tools", None)
                    }

                    if not self.agent_disable_plan_corrections and result["plan"].endswith("_no_answer"):
                        replacement_plan = replace_plan_with_answering_plan(result["plan"])

                        if goal.goal_plan is not None and goal.goal_plan.plan_domain is not None:
                            if templates[result["plan"]]["dont_replace_for_domains"] is not None and goal.goal_plan.plan_domain in templates[result["plan"]]["dont_replace_for_domains"]:
                                replacement_plan = None

                        if replacement_plan is not None:
                            log(WARNING, f"LLM has proposed a simple plan with no answer ('{result['plan']}'), but only a single goal. We replace it with '{replacement_plan}', else the user would not get an answer!")
                            result["plan"] = replacement_plan
                else:
                    optional_doc_ids = response["optional_complex_plan"]["optional_doc_ids"] if "optional_doc_ids" in response["optional_complex_plan"] else None
                    if optional_doc_ids is not None and not optional_doc_ids.startswith(DOCUMENT_ID_PREFIX):
                        log(WARNING, f"***WARNING: {self.agent_name} LLM has found an invalid document ID '{optional_doc_ids}' in the complex plan. It has been removed")
                        optional_doc_ids = None

                    result = {
                        "optional_complex_plan": response["optional_complex_plan"],
                        "workflow_plan": workflow_plan,
                        "optional_doc_ids": optional_doc_ids
                    }
            else:
                for i in range(len(response["optional_complex_plan"]["goals"])-1):
                    if "recommended_plan" in response["optional_complex_plan"]["goals"][i] and response["optional_complex_plan"]["goals"][i]["recommended_plan"] == "plan_call_tool_and_answer":
                        response["optional_complex_plan"]["goals"][i]["recommended_plan"] = "plan_call_tool_but_no_answer"

                if not self.agent_disable_plan_corrections and "recommended_plan" in response["optional_complex_plan"]["goals"][-1] and response["optional_complex_plan"]["goals"][-1]["recommended_plan"] == "plan_call_tool_but_no_answer":
                    replacement_plan = "plan_call_tool_and_answer"
                    if goal.goal_plan is not None and goal.goal_plan.plan_domain is not None:
                        recommended_plan = response["optional_complex_plan"]["goals"][-1]["recommended_plan"]
                        if "dont_replace_for_domains" in templates[recommended_plan] and templates[recommended_plan]["dont_replace_for_domains"] is not None and goal.goal_plan.plan_domain in templates[recommended_plan]["dont_replace_for_domains"]:
                            replacement_plan = None
                    if replacement_plan is not None:
                        response["optional_complex_plan"]["goals"][-1]["recommended_plan"] = replacement_plan

                optional_doc_ids = response["optional_complex_plan"]["optional_doc_ids"] if "optional_doc_ids" in response["optional_complex_plan"] else None
                if optional_doc_ids is not None and not optional_doc_ids.startswith(DOCUMENT_ID_PREFIX):
                    log(WARNING, f"***WARNING: {self.agent_name} LLM has found an invalid document ID '{optional_doc_ids}' in the complex plan. It has been removed")
                    optional_doc_ids = None

                result = {
                    "optional_complex_plan": response["optional_complex_plan"],
                    "workflow_plan": workflow_plan,
                    "optional_doc_ids": optional_doc_ids
                }
        else:
            warning = True
            plan = None
            if isinstance(response, str) and len(response) > 0:
                plan = response
                warning = False
            elif isinstance(response, dict):
                if "plan" in response and response["plan"] is not None and len(response["plan"]) > 0:
                    plan = response["plan"]
                    warning = False

            if warning or plan is None:
                log(WARNING, f"LLM has failed to calculate a new plan. We must use a backup plan {backup_plan_default}")
                plan = backup_plan_default

            result = {"plan": plan}
            filtered_tools_info = None

        return result, filtered_tools_info
    
    def get_filtered_workflows(self, plan:'Plan') -> Tuple[List[dict], List[dict]]:
        filtered = []
        blacklist = []

        workflows = LlmManager.instance().get_workflows()
        active_reasoning = LlmManager.instance().get_active_reasoning()
        if active_reasoning is not None:
            active_reasoning = active_reasoning.value
        else:
            active_reasoning = AgentReasoningEnum.FAST.value

        for wf in workflows:
            # check if the tool has a "WhiteList_IF" or a "BlackList_IF" condition
            blacklist_if = wf["blacklist_if"] if "blacklist_if" in wf else None

            if blacklist_if is not None and len(blacklist_if) > 0:
                # evaluate the condition by executing it. It is in the form:
                #   "(plan) -> { return plan.plan_domain == "coins"; }"
                # The plan is passed as an argument to the lambda function
                try:
                    blacklisted = eval(blacklist_if, {"llm_manager":LlmManager.instance(), "reasoning":active_reasoning}, {"plan":plan})
                    if callable(blacklisted):
                        blacklisted = blacklisted(plan)
                        if blacklisted:
                            blacklist.append(wf)
                            continue
                except Exception as e:
                    log(ERROR, f"'ERROR: {self.agent_name}' Agent has failed to evaluate the BlackList_IF condition for the workflow '{wf['id']}': {e}")

            filtered.append(wf)

        # if there are no whitelisted tools, return the filtered list which excludes the blacklisted tools
        return filtered, blacklist

    def get_goals_instances_from_complex_plan(self, plan:'Plan', current_goal:Goal, complex_plan:dict, output_recommended_plan:bool, output_recommended_tools:bool) -> List[Goal]:
        # generate the goals instances from the complex plan
        # remember that the dependencies are a difference from the current goal sequence nr.
        goals = complex_plan["optional_complex_plan"]["goals"]

        goals_instances = []

        idx = 0
        for goal in goals:
            goal_sequence_id = goal.get("goal_sequence_id", DEFAULT_SINGLE_GOAL_SEQUENCE_ID)
            goal_prompt = goal.get("goal")
            if goal_prompt is None:
                goal_prompt = DEFAULT_SINGLE_GOAL_PROMPT
            goal_description = ": " + goal["name"] if "name" in goal else ": " + goal["description"] if "description" in goal else ""
            goal_title = goal_sequence_id
            dependent_on_sequence_ids = goal.get("dependent_on_sequence_ids", [])
            if dependent_on_sequence_ids is not None:
                # convert eventual strings into negative integers
                dependent_on_sequence_ids_int:List[int] = []
                for dep_seq_id in dependent_on_sequence_ids:
                    if isinstance(dep_seq_id, int):
                        if dep_seq_id >= 0 or dep_seq_id < -len(goals):
                            log(ERROR, f"Invalid dependency sequence ID: {dep_seq_id}")
                        dependent_on_sequence_ids_int.append(dep_seq_id)
                    elif isinstance(dep_seq_id, str):
                        # identify the previous step with the same goal_sequence_id as this dep_seq_id string, and measure its distance from the current goal
                        # example: if we are on Step-2 (second of the goals array), and we have a dependency on "Step-1", we want to add -1 to the list
                        # iterate on the goals from the current position - 1, to the beginning, until we find the dependency. If not found, no dependency is appended
                        for i in range(idx - 1, -1, -1):
                            if goals[i]["goal_sequence_id"] == dep_seq_id:
                                # we found the dependency, we must add it as negative (e.g. Step-2 depends on Step-1, then: -1)
                                dep = i - idx
                                if dep >= 0 or dep < -len(goals):
                                    log(ERROR, f"Invalid dependency sequence ID: {dep_seq_id} converted to int: {dep}")
                                dependent_on_sequence_ids_int.append(dep)
                                break

                dependent_on_sequence_ids = dependent_on_sequence_ids_int

            # Handle recommended tools and plan for each individual goal
            recommended_tools = None
            recommended_plan = None

            if goal_prompt is not None:
                if output_recommended_plan:
                    # add a recommended plan to the goal's prompt
                    recommended_plan = goal["recommended_plan"] if "recommended_plan" in goal else None
                    if recommended_plan is not None and len(recommended_plan) > 0:
                        goal_prompt += f" (recommended plan: {recommended_plan})"

                if output_recommended_tools:
                    # add optional recommended tools to the goal's prompt
                    optional_recommended_tools = goal["optional_recommended_tools"] if "optional_recommended_tools" in goal else None
                    if optional_recommended_tools is not None and len(optional_recommended_tools) > 0:
                        # Ensure recommended_tools is always a List[str]
                        if isinstance(optional_recommended_tools, list):
                            recommended_tools = optional_recommended_tools
                        elif isinstance(optional_recommended_tools, str):
                            recommended_tools = [optional_recommended_tools]
                        else:
                            recommended_tools = None
                        
                        if recommended_tools is not None and len(recommended_tools) > 0:
                            if len(recommended_tools) > 1:
                                goal_prompt += f" (recommended tools: {','.join(recommended_tools)})"
                            else:
                                goal_prompt += f" (recommended tool: {recommended_tools[0]})"

            # instantiate the microplanner
            agent_microplanner = import_class("src.core.agents.AgentMicroplanner", "AgentMicroplanner")
            goal_instance = Goal(plan, current_goal, goal_title, goal_description, GoalTarget.AGENT, agent_microplanner, dependent_on_sequence_ids, goal_prompt)
            goal_instance.goal_sequence_id = goal_sequence_id
            goal_instance.goal_recommended_tools = recommended_tools
            goal_instance.goal_recommended_plan = recommended_plan if recommended_plan is not None and len(recommended_plan)>0 else None
            goal_instance.goal_title = goal["name"] if "name" in goal else None
            goal_instance.goal_focus_docids = complex_plan["optional_doc_ids"] if "optional_doc_ids" in complex_plan else None
            if goal_instance.goal_focus_docids is not None and not goal_instance.goal_focus_docids.startswith(DOCUMENT_ID_PREFIX):
                log(WARNING, f"***WARNING: {self.agent_name} LLM has found an invalid document ID '{goal_instance.goal_focus_docids}' in the complex plan. It has been removed")
                goal_instance.goal_focus_docids = None
            goal_instance.goal_focus_iteration = goal["focus_iteration"] if "focus_iteration" in goal else None
            goal_instance.goal_focus_regex = goal["focus_regex"] if "focus_regex" in goal else None
            goal_instance.goal_focus_instr = goal["focus_instr"] if "focus_instr" in goal else None
            goal_instance.goal_focus_reasoning_division = goal["focus_reasoning_division"] if "focus_reasoning_division" in goal else None
            goals_instances.append(goal_instance)
            idx += 1

        return goals_instances

    def get_plans_for_goal(self, goal:Goal) -> List[str]:
        plan = goal.goal_plan
        plans = []
        templates = LlmManager.instance().get_agent_plans_template()
        for plan_name in templates.keys():
            templates = LlmManager.instance().get_agent_plans_template()
            p = templates[plan_name]
            plan_description = p["description"]
            add = False

            # if there's a whitelist and the plan is included in it, or there's no blacklist or the plan is not included in it, or there are no restrictions
            if (plan.plan_template_whitelist is not None and plan_name in plan.plan_template_whitelist) or (plan.plan_template_blacklist is None or plan_name not in plan.plan_template_blacklist):
                if "blacklisted" not in p or p["blacklisted"]==False:
                    if goal.goal_recommended_plan is None or len(goal.goal_recommended_plan) == 0:
                        # no particular goal recommendation: add all plans
                        add = True
                    elif goal.goal_recommended_plan[0]!='+':
                        # the goal has a weak recommendation, add all plans
                        add = True
            if goal.goal_recommended_plan is not None and len(goal.goal_recommended_plan)>0 and goal.goal_recommended_plan[0]=='+':
                recommended_plan = goal.goal_recommended_plan[1:]
                if recommended_plan == plan_name:
                    # the goal has a specific, mandatory recommendation. Add only this plan
                    add = True

            if add:
                plans.append(f'- {plan_name}: {plan_description}\n')

        return plans
    
    def get_tools_for_goal(self, goal:Goal, tools_info:list) -> list:
        tools_list = []
        for tool in tools_info:
            tool_name = tool["schema"]["function"]["name"]
            tool_description = tool["schema"]["function"]["description"]
            tools_list.append(f'"{tool_name}": {tool_description}\n')
        return tools_list
    
    def get_tools_prompt_for_goal(self, goal:Goal, tools_info:list) -> Tuple[int,str]:
        """
            Determines if each tool's instructions need to be included in the system prompt and adds them if necessary.
            There are three ways to include a tool's instructions in the system prompt:
            1. if the current agent (self) has a specific LLM label for the tool
               (ref. to the thalabus page "Agents" under "Platform settings", go to an Agent and then see the section "Dictionary with other information: Label")
            2. if the tool's .skill configuration file contains a "LLM_Label" key, with a string matching the current agent's labels (as in the previous point)
            3. if the tool's "LLM_Label" key (see previous point) matches the current plan's plan_domain value

            Args:
                goal (Goal): The goal object.
                tools_info (list): A list containing information about the tools.

            Returns:
                Tuple[int, str]: A tuple containing an integer and a string representing the system prompt.
        """
        log(DEBUG, f"{self.agent_name} LLM is gathering tools instructions for the system prompt...")

        # for each tool, add the tool's eventual "other information" to the system prompt
        llm_instructions = []
        tools_prompt = ""
        tools_added = 0
        domain = goal.goal_plan.plan_domain if goal is not None and goal.goal_plan is not None else None

        for tool in tools_info:
            skillid = tool['schema']['function']['name']
            skill_llm_instructions = LlmManager.instance().get_system_prompt(agent=self, label=skillid)
            skill_llm_label = tool['llm_label']
            if skill_llm_instructions is not None:
                tools_prompt += f"# Instructions for the skill {skillid}: {skill_llm_instructions}\n"
                tools_added += 1
                log(DEBUG, f"... {self.agent_name} LLM has added instructions for the {skillid} to the system prompt: reason is that there is a label matching the tool name '{skillid}' for this agent.")
            if skill_llm_label is not None:
                for _label in [l.strip() for l in skill_llm_label.split(',') if l.strip()]:
                    if _label not in llm_instructions:
                        llm_instructions.append(_label)
                        log(DEBUG, f"... {self.agent_name} {skillid} requires instructions with the LLM_Label '{_label}' to be included.")

        if len(llm_instructions) > 0:
            tools_prompt += f"<TOOLS_INSTRUCTIONS>\n"
            
            for inst in llm_instructions:
                skill_llm_label = LlmManager.instance().get_system_prompt(agent=self, label=inst)
                if skill_llm_label is not None:
                    tools_added += 1
                    instruction_id = html.escape(f"{self.agent_name}/{inst}", quote=True)
                    tools_prompt += f"<INSTRUCTIONS id=\"{instruction_id}\">{skill_llm_label}</INSTRUCTIONS>\n\n"
            
            tools_prompt += f"</TOOLS_INSTRUCTIONS>\n"

        if domain is not None:
            # add the domain-specific instructions
            domain_instructions = LlmManager.instance().get_system_prompt(agent=self, label=domain)
            if domain_instructions is not None:
                tools_added += 1
                tools_prompt += f"# Domain-specific instructions for domain '{domain}': {domain_instructions}\n"
                log(DEBUG, f"... {self.agent_name} LLM has added domain-specific instructions to the system prompt: reason is that there is a label matching the domain '{domain}' for this agent.")

        log(DEBUG, f"... {self.agent_name} LLM has completed gathering tools instructions for the system prompt.")

        return tools_added, tools_prompt

    def _add_ontology_hints(self, messages:list) -> list:
        if messages is None or len(messages) == 0:
            return messages
        
        for i in range(0, len(messages)):
            if "role" in messages[i] and "content" in messages[i] and messages[i]["content"] is not None and messages[i]["role"] == "system" and "</ONTOLOGY_RULES>" in messages[i]["content"]:
                # Insert a hint immediately before the closing </ONTOLOGY_RULES> tag
                ontology_hint = LlmManager.instance().get_system_prompt(agent=self, label="KeywordsAndCategoriesHint")
                if ontology_hint is not None:
                    messages[i]["content"] = re.sub(r'(</ONTOLOGY_RULES>)', ontology_hint + r'\1', messages[i]["content"], count=1)
                break

        return messages

    def _consolidate_leading_system_messages(self, messages: list) -> list:
        """
        Merge all leading contiguous system messages into a single system message.

        Only applies to Qwen-family models: on Qwen only the very first system
        message is honoured; every subsequent one is demoted to a user-role turn
        by _sanitize_messages.  Merging them here ensures the full instruction
        set (main prompt, ontology hints, <PLANS>, <TOOLS>, <PRE-BUILT-PLANS>, …)
        lands in the single privileged system slot as one coherent block.

        For all other providers the messages list is returned unchanged.

        System messages that appear *after* a non-system message (e.g. the
        <SYSTEM_MESSAGE_REMINDER> appended at the very end) are intentionally
        left untouched: Qwen will demote them to user-role turns, which makes
        them the last instruction read before the model replies — the intended
        behaviour for the reminder.
        """
        if not messages:
            return messages

        # Only consolidate for Qwen providers
        llm_configuration = LlmManager.instance().get_llm_model_configuration_for_agent(agent=self, agent_purpose=None)
        model_provider = llm_configuration.get("modelprovider") if llm_configuration else None
        if model_provider is None or model_provider.lower() != "qwen":
            return messages

        leading_contents: list = []
        cut: int = 0
        for msg in messages:
            if msg.get("role") == "system":
                leading_contents.append(msg.get("content", ""))
                cut += 1
            else:
                break

        if len(leading_contents) <= 1:
            return messages  # nothing to consolidate

        merged = {"role": "system", "content": "\n\n".join(leading_contents)}
        log(DEBUG, f"_consolidate_leading_system_messages: merged {cut} system messages into one.")
        return [merged] + messages[cut:]

    def _ask_LLM_about_macro_plan_internal_thought(self, goal:Goal, user_prompt:str=None, model:str=MODEL_PLANNER) -> str:
        """
        Generate internal thought analysis to guide plan creation.
        Returns plain text analysis, not JSON.
        """
        # get the system prompt for internal thought generation
        system_prompt_internal_thought = LlmManager.instance().get_system_prompt(agent=self, label="InternalThought")
        if system_prompt_internal_thought is None:
            log(WARNING, f"WARNING: {self.agent_name} LLM has failed to provide a system prompt for internal thought generation.")
            return None
        
        # build the conversation messages for the LLM as input
        messages = goal.build_messages(
            agent=self,
            system_prompt=system_prompt_internal_thought,
            include_focus=True,
            include_rag=False,
            include_images=True,
            system_prompt_label="InternalThought",
        )
        
        # find the message with the <ONTOLOGY>...</ONTOLOGY> tag and inject a hint about the internal thought
        messages = self._add_ontology_hints(messages)
        
        # add the available plans to the system prompt
        plans = self.get_plans_for_goal(goal)
        messages.insert(1, {"role":"system", "content": f"<PLANS>\nThese plans are available: {''.join(plans)}\n</PLANS>"})
        
        # get filtered tools info
        filtered_tools_info, whitelisted_tools, blacklisted_tools = get_filtered_tools(goal.goal_plan, goal)
        
        # add tools information
        tools_list = self.get_tools_for_goal(goal, filtered_tools_info)
        if len(tools_list) > 0:
            tools_text = "<TOOLS>\nThese skills (=tools) are available:\n"
            for tool_text in tools_list:
                tools_text += tool_text
            tools_text += "</TOOLS>\n"
            messages.insert(2, {"role":"system", "content": tools_text})
        else:
            messages.insert(2, {"role":"system", "content": "<TOOLS>\nNo skills (=tools) are available.\n</TOOLS>\n"})
        
        # for each tool, add the tool's eventual "other information" to the system prompt
        tools_added, tools_prompt = self.get_tools_prompt_for_goal(goal, filtered_tools_info)
        if tools_added > 0 and tools_prompt is not None and len(tools_prompt) > 0:
            messages.insert(2, {"role":"system", "content": tools_prompt})
        
        # get all available pre-built complex plans
        plan = goal.goal_plan
        workflows, blacklisted_workflows = self.get_filtered_workflows(plan)
        if len(workflows) > 0:
            workflows_prompt = "<PRE-BUILT-PLANS>\nThese pre-built complex plans are available:\n"
            workflows_prompt += f"PRE-BUILT-PLAN-IDS=[{','.join('pre-built-plan-'+wf['id'] for wf in workflows)}]\n"
            for wf in workflows:
                if "llm_instructions" in wf:
                    wf_inst = wf['llm_instructions']
                    workflows_prompt += f"'pre-built-plan-{wf['id']}': {wf_inst}\n"
            workflows_prompt += "</PRE-BUILT-PLANS>\n"
            messages.insert(3, {"role":"system", "content": workflows_prompt})
        
        if user_prompt is not None and len(user_prompt) > 0:
            # add the additional prompt to the messages prompt
            messages.append({"role":"user", "content": user_prompt})

        # Merge the leading system messages into one so that single-system-message
        # providers (e.g. Qwen) receive all instructions in the privileged slot.
        messages = self._consolidate_leading_system_messages(messages)

        # get the response from the LLM
        try:
            llm_response = LlmManager.instance().chat_completions(
                agent=self,
                plan=goal.goal_plan if goal.goal_plan is not None else plan,
                goal=goal,
                messages=messages,
                tools=None,
                response_format=None,
                structured_output=None,
                model_input=model,
                max_completion_tokens=MAX_COMPLETION_TOKENS_PLANNING)
            
            internal_thought = llm_response.response
            if internal_thought is not None and isinstance(internal_thought, str):
                log(DEBUG, f"{self.agent_name} generated internal thought: {internal_thought[:200]}...")
                return internal_thought
            else:
                log(WARNING, f"WARNING: {self.agent_name} failed to generate valid internal thought")
                return None
            
        except Exception as e:
            log(ERROR, f"ERROR: {self.agent_name} failed to generate internal thought: {str(e)}")
            return None

    def ask_LLM_about_macro_plan(self, goal:Goal, user_prompt:str=None, model:str=MODEL_PLANNER) -> Tuple[Union[List[dict], dict], dict, bool]:
        self.raise_if_plan_stopped()
        result = None
        plan = goal.goal_plan
        extended_system_prompt_append = None
        workflow_plan = None
        override_plan_name = None
        filtered_tools_info = None
        backup_plan_default = "plan_rag_assistant_and_answer"
        # set to True when optional_require_exact_time newly switches the plan into exact-time mode,
        # so the Contextualizer can be re-run with the exact date before the next subgoal executes
        repeat_contextualizer = False

        if plan is not None and plan.plan_goal is not None:
            previous_goal = plan.plan_goal.get_last(statuses=[GoalStatus.COMPLETED], agents=["AgentUser"])
            if previous_goal is not None and previous_goal.goal_override_recommended_plan is not None:
                override_plan_name = previous_goal.goal_override_recommended_plan
                if override_plan_name.startswith('+'):
                    override_plan_name = override_plan_name[1:]
                if override_plan_name is None:
                    log(ERROR, f"***ERROR: {self.agent_name} LLM has failed to find the override plan '{previous_goal.goal_override_recommended_plan}' in the templates.")
                else:
                    log(DEBUG, f"{self.agent_name} has found an override of the recommended plan in goal {previous_goal.get_id_chain_as_str()}: '{override_plan_name}'")

        if override_plan_name is None:
            # check if skills need to be called
            filtered_tools_info, whitelisted_tools, blacklisted_tools = get_filtered_tools(goal.goal_plan, goal)

            # prepare system prompt with the skills
            # add the system prompt for the complex plans
            system_prompt_complex = LlmManager.instance().get_system_prompt(agent=self, purpose=None)
            if system_prompt_complex is None:
                log(ERROR, f"ERROR: {self.agent_name} LLM has failed to provide a system prompt for the agent planner.")
                return None, None, repeat_contextualizer
            system_prompt = system_prompt_complex
            # YAML output: no structured_output or response_format constraints
            # The system prompt (AgentPlanner.txt) contains YAML format instructions and examples

            # determine if reasoning is enabled
            llm_configuration = LlmManager.instance().get_llm_model_configuration_for_agent(agent=self, agent_purpose=None)
            is_reasoning_llm = llm_configuration["is_reasoning"] if llm_configuration is not None and "is_reasoning" in llm_configuration else False
            internal_thought_text:Optional[str] = None
            
            if not LlmManager.instance().is_reasoning_enabled_for_agent(self) or is_reasoning_llm == True:
                # no, reasoning is not enabled. Remove all reasoning-related fields from the system prompt
                # (no format structures to modify for YAML output)

                # now remove the reasoning-related instructions from the system_prompt, which are enclosed in "INTERNAL_THOUGHT=[[...]]" sequences. Use regular expressions
                system_prompt = re.sub(r"INTERNAL_THOUGHT=\[\[.*?\]\]", "", system_prompt, flags=re.DOTALL)
            else:
                # yes, we must use reasoning. Generate internal thought first as separate call
                log(DEBUG, f"{self.agent_name} is generating internal thought analysis...")
                internal_thought_text = self._ask_LLM_about_macro_plan_internal_thought(goal, user_prompt, model)
                self.raise_if_plan_stopped()
                
                # No format structures to modify for YAML output
                
                # Inject the internal thought into the system prompt where INTERNAL_THOUGHT markers are
                if internal_thought_text is not None and len(internal_thought_text) > 0:
                    # Replace INTERNAL_THOUGHT=[[...]] with the actual internal thought text
                    replacement_text = f"The following internal thought analysis has been generated to guide your plan creation:\n\n{internal_thought_text}\n\nUse this analysis to construct your YAML output."
                    system_prompt = re.sub(r"INTERNAL_THOUGHT=\[\[.*?\]\]", replacement_text, system_prompt, flags=re.DOTALL)
                else:
                    # If internal thought generation failed, just remove the markers
                    log(WARNING, f"WARNING: Internal thought generation failed, removing INTERNAL_THOUGHT markers")
                    system_prompt = re.sub(r"INTERNAL_THOUGHT=\[\[.*?\]\]", "", system_prompt, flags=re.DOTALL)
                
            if extended_system_prompt_append is not None:
                system_prompt += f"\n{extended_system_prompt_append}"

            # build the conversation messages for the LLM as input
            self.agent_system_prompt = system_prompt
            messages = goal.build_messages(agent=self, include_focus=True, include_rag=False, include_images=True)

            # find the message with the <ONTOLOGY>...</ONTOLOGY> tag and inject a hint about the complex plan response
            messages = self._add_ontology_hints(messages)

            # add the available plans to the system prompt
            plans = self.get_plans_for_goal(goal)
            messages.insert(1, {"role":"system", "content": f"<PLANS>\nThese plans are available for your yaml response structure:\n{''.join(plans)}\n</PLANS>"})

            # at complex plan level, that is at macro-planning, no direct tools invocation is allowed.
            # we describe the available tools to the system prompt, because we don't add them as "tools" to be called
            tools: Optional[List[Dict]] = None
            tools_list = self.get_tools_for_goal(goal, filtered_tools_info)
            if len(tools_list) > 0:
                tools_text = "<TOOLS>\nThese skills (=tools) are available for your yaml response structure:\n"
                for tool_text in tools_list:
                    tools_text += tool_text
                tools_text += "</TOOLS>\n"
                messages.insert(2, {"role":"system", "content": tools_text})
            else:
                # tools stays None — AgentPlanner never invokes tools via function-calling
                messages.insert(2, {"role":"system", "content": "<TOOLS>\nNo skills (=tools) are available for your yaml response structure.\n</TOOLS>\n"})

            # for each tool, add the tool's eventual "other information" to the system prompt
            tools_added, tools_prompt = self.get_tools_prompt_for_goal(goal, filtered_tools_info)
            if tools_added > 0 and tools_prompt is not None and len(tools_prompt) > 0:
                messages.insert(2, {"role":"system", "content": tools_prompt})

            # get all available pre-built complex plans, which are called 'workflows'
            workflows_prompt = ""
            workflows, blacklisted_workflows = self.get_filtered_workflows(plan)
            if len(workflows) > 0:
                workflows_prompt += "<PRE-BUILT-PLANS>\nThese pre-built complex plans are available for your yaml response structure:\n"
                workflows_prompt += f"PRE-BUILT-PLAN-IDS=[{','.join('pre-built-plan-'+wf['id'] for wf in workflows)}]\n"
                for wf in workflows:
                    if "llm_instructions" in wf:
                        wf_inst = wf['llm_instructions']
                        workflows_prompt += f"'pre-built-plan-{wf['id']}': {wf_inst}\n"
                workflows_prompt += "</PRE-BUILT-PLANS>\n"
                messages.insert(3, {"role":"system", "content": workflows_prompt})

            # add a reminder that strengthens YAML-only output discipline.
            # Note: on Qwen-family models this message is demoted to a user-role turn by _sanitize_messages,
            # which makes it the last instruction the model sees before it replies — ideal for a final nudge.
            messages.append({"role":"system", "content": (
                "<SYSTEM_MESSAGE_REMINDER>"
                "IMPORTANT OUTPUT RULES — comply with ALL of the following:\n"
                "1. Your entire response must be valid YAML only. Do not add any text, explanation, commentary, or reasoning before or after the YAML block.\n"
                "2. Do NOT append a 'Reasoning:', 'Note:', 'Explanation:', or any prose section after your YAML output. The YAML must be the last thing you write.\n"
                "3. Do NOT reproduce or echo back any of the context blocks you received (<PLANS>, <TOOLS>, <PRE-BUILT-PLANS>, etc.). Those are inputs, not outputs.\n"
                "4. Do NOT wrap your YAML in markdown code fences (no ```yaml ... ```).\n"
                "5. Follow exactly the YAML structure and field names shown in the system prompt."
                "</SYSTEM_MESSAGE_REMINDER>"
            )})

            # add the user prompt at the end of the messages, so that it doesn't dilute the system prompt information and instructions, which are crucial for the LLM to generate a valid response
            if user_prompt is not None and len(user_prompt) > 0:
                # add the additional prompt to the messages prompt
                messages.append({"role":"user", "content": user_prompt})

            # Merge the leading system messages into one so that single-system-message
            # providers (e.g. Qwen) receive all instructions in the privileged slot.
            # The <SYSTEM_MESSAGE_REMINDER> (appended above, after the non-system history)
            # is intentionally outside the leading block and will be demoted to a
            # user-role turn by _sanitize_messages — making it the last instruction
            # the model reads before it replies.
            messages = self._consolidate_leading_system_messages(messages)

            response = self._ask_finetuned_planner(goal, messages, plan) if AGENTPLANNER_FINETUNE_ENABLE else None
            self.raise_if_plan_stopped()
            response_error = None
            raw_response = None
            if response is None:
                response, raw_response, response_error = self._ask_standard_planner(
                    goal,
                    messages,
                    tools,
                    plan,
                    model)
                self._record_planner_output(goal, raw_response)

            is_valid, validation_error = self._validate_planner_response(response)
            if not is_valid:
                self._last_planner_error = (
                    f"AgentPlanner failed after {MAX_PLANNER_ATTEMPTS} attempts: {validation_error}. "
                    f"Last raw LLM response: {raw_response if raw_response is not None else '<no response>'}"
                )
            else:
                self._last_planner_error = None

            # save the response in the goal's debug log
            if goal.goal_debug is None:
                goal.goal_debug = {}
            goal.goal_debug["llm_output"] = response
            
            if self._last_planner_error is not None:
                log(ERROR, f"'{self.agent_name}': {self._last_planner_error}")
                return None, None, repeat_contextualizer

            backup_plan = False
            if response is None:
                backup_plan = True
            else:
                if "optional_prebuilt_complex_plan_id" in response and response["optional_prebuilt_complex_plan_id"] is not None and isinstance(response["optional_prebuilt_complex_plan_id"], str) and len(response["optional_prebuilt_complex_plan_id"]) > 0:
                    # check whether to instantiate a pre-built complex plan (i.e. a workflow)
                    pre_built_plan_id = response["optional_prebuilt_complex_plan_id"]
                    if "pre-built-plan-" in pre_built_plan_id:
                        pre_built_plan_id = pre_built_plan_id.replace("pre-built-plan-", "")

                    if pre_built_plan_id in [wf["id"] for wf in workflows]:
                        # instantiate the workflow
                        response["optional_complex_plan"] = LlmManager.instance().get_workflow_by_id(pre_built_plan_id)
                        workflow_plan = pre_built_plan_id

                if "optional_complex_plan" not in response or response["optional_complex_plan"] is None or ("goals" in response["optional_complex_plan"] and len(response["optional_complex_plan"]["goals"]) < 1):
                    llm_configuration = LlmManager.instance().get_llm_model_configuration_for_agent(agent=self)
                    if "intelligence" in llm_configuration and llm_configuration["intelligence"] is not None and llm_configuration["intelligence"] < MIN_INTELLIGENCE_LEVEL_FOR_BACKUP_PLAN:
                        log(WARNING, f"WARNING: {self.agent_name} LLM has failed to calculate a new plan. The intelligence level is too low. We must use a backup plan.")
                        backup_plan = True
                        if "optional_complex_plan" in response and "goals" in response["optional_complex_plan"] and len(response["optional_complex_plan"]["goals"]) > 0 and "goal" in response["optional_complex_plan"]["goals"][0] and "research" in response["optional_complex_plan"]["goals"][0]["goal"].lower():
                            backup_plan_default = "plan_rag_assistant_and_answer"

                if "optional_require_exact_time" in response:
                    require_exact_time = response["optional_require_exact_time"]
                    if require_exact_time == True and goal.goal_plan.plan_include_time == False:
                        # the Contextualizer already ran earlier in this round without the exact date;
                        # it must be re-run so the exact <DATE_TIME> is available to later subgoals
                        repeat_contextualizer = True
                    goal.goal_plan.plan_include_time = require_exact_time
                    log(DEBUG, f"Updated plan_include_time to: {require_exact_time}")
            
            if backup_plan:
                # the model has been too stupid to generate a plan, so we need to use a backup plan
                log(WARNING, f"LLM has failed to calculate a new plan. Using a backup plan: {backup_plan_default}")
                plan = backup_plan_default
                result = { "plan": plan }
                filtered_tools_info = None
            else:
                result, filtered_tools_info = self._finalize_standard_plan_response(
                    goal,
                    response,
                    workflow_plan,
                    filtered_tools_info,
                    workflows,
                    backup_plan_default)

        else:
            # override_plan_name is not None: the SDK provided a mandatory plan to be used. We must use it
            log(INFO, f"'{self.agent_name}' has been instructed to use the mandatory plan '{override_plan_name}'")
            result = { "plan": override_plan_name }
            filtered_tools_info = None

        return filtered_tools_info, result, repeat_contextualizer

    def insert_simple_plan(self, goal:Goal, proposed_plan_name:str, dependent_on_previous_goal:bool, output_recommended_plan:bool, output_recommended_tools:bool, after_id:str=None, duplicate_agent_name:Optional[str]=None, duplicate_division_ids:Optional[List[str]]=None) -> dict:
        # simple plan: instantiate the new plan proposed by the LLM
        log(DEBUG, f"'{self.agent_name}' has proposed a new plan to be added: \033[31m{proposed_plan_name}\033[0m")
        
        with goal.goal_plan.plan_lock:
            plan_fragment = get_goals_from_plan_template(goal.goal_plan, proposed_plan_name)
            if plan_fragment is None:
                log(ERROR, f"'ERROR: {self.agent_name}' LLM has returned an inexistent plan. Planning failed!")
                response = {
                    "success": False,
                    "internal_thought": None
                }
                return response

            if dependent_on_previous_goal:
                # add the dependencies to the new plan
                idx = -1
                for dep_goal in plan_fragment:
                    if dep_goal.goal_dependent_upon is None:
                        dep_goal.goal_dependent_upon = []
                    else:
                        dep_goal.goal_dependent_upon = dep_goal.goal_dependent_upon.copy()
                    if dep_goal.goal_parent is not None and dep_goal.goal_parent.goal_subgoals is not None and idx < -dep_goal.goal_parent.goal_subgoals.index(dep_goal):
                        log(WARNING, "Add an invalid dependency... this shouldn't happen")
                    dep_goal.goal_dependent_upon.append(idx)       # add the dependency to the previous goal
                    dep_goal.goal_sequence_id = goal.goal_sequence_id
                    idx -= 1

            # optionally duplicate one of the agent name and fix it into the new focus
            # (both arguments must be provided)
            if duplicate_agent_name is not None and duplicate_division_ids is not None:
                goal_to_duplicate: Optional[Goal] = None
                for g in plan_fragment:
                    if g.get_target_agent_name() == duplicate_agent_name:
                        goal_to_duplicate = g
                        break
                if goal_to_duplicate is not None:
                    # duplicate the goal, as many times as the length of duplicate_division_ids, and 
                    # set each time the new focus to the corresponding division id
                    div_idx: int = 0
                    for division_id in duplicate_division_ids:
                        duplicated_goal: Goal = Goal.clone(goal_to_duplicate)
                        duplicated_goal.goal_dependent_upon = [- (1 + div_idx)]  # depend upon the original duplicated goal
                        duplicated_goal.goal_focus_reasoning_division = division_id  # set the new focus

                        # insert the duplicated goal right after the original duplicated goal
                        plan_fragment.insert(plan_fragment.index(goal_to_duplicate) + 1 + div_idx, duplicated_goal)
                        log(DEBUG, f"Duplicated goal for agent '{duplicate_agent_name}' with new focus: {division_id}")
                        div_idx += 1

                    # finally, remove the original goal to duplicate
                    plan_fragment.remove(goal_to_duplicate)

                    # the last goal must be dependent on all previous duplicated goals
                    last_goal: Goal = plan_fragment[-1]
                    last_goal.goal_dependent_upon = list(range(-div_idx, 0))
                    
            # insert the prompt into each goal
            # Normalize goal prompt to a single string if it's a list
            new_goal_prompt = goal.goal_prompt
            new_recommended_tools = goal.goal_recommended_tools
            new_recommended_plan = goal.goal_recommended_plan

            # Convert to string if it's not already (handle dict, list, etc.)
            if new_goal_prompt is not None and not isinstance(new_goal_prompt, str):
                new_goal_prompt = str(new_goal_prompt)

            if output_recommended_plan is False and new_goal_prompt is not None:
                # remove the '(recommended plan: ...)' from the new_goal
                new_goal_prompt = re.sub(r'\s*\(recommended plan:.*?\)', '', new_goal_prompt)

            if output_recommended_tools is False and new_goal_prompt is not None:
                # remove the '(recommended tool: ...)' from the new_goal
                new_goal_prompt = re.sub(r'\s*\(recommended tool:.*?\)', '', new_goal_prompt)

            for g in plan_fragment:
                g.goal_prompt = new_goal_prompt if g.goal_prompt is None else g.goal_prompt
                g.goal_recommended_tools = new_recommended_tools
                g.goal_recommended_plan = new_recommended_plan
                g.goal_focus_docids = goal.goal_focus_docids
                g.goal_focus_iteration = goal.goal_focus_iteration
                g.goal_focus_regex = goal.goal_focus_regex
                g.goal_focus_instr = goal.goal_focus_instr
                if g.goal_focus_reasoning_division is None:
                    g.goal_focus_reasoning_division = goal.goal_focus_reasoning_division

            # insert the plan after the current goal
            if after_id is not None:
                # instantiate as siblings after the specified goal
                inserted = goal.goal_parent.insert_goals(plan_fragment, after_id=after_id, instantiated_plan=proposed_plan_name)
            else:
                if app_config.testing:
                    log_plan(goal.goal_plan)

                # instantiate as sub-goals of the current goal
                inserted = goal.insert_goals(plan_fragment, after_id=None, instantiated_plan=proposed_plan_name)
                
            if inserted == False:
                log(WARNING, f"'WARNING: {self.agent_name}' Agent has failed to insert the new plan after the current goal.")
            elif app_config.testing:
                log_plan(goal.goal_plan)

        response = {
            "restart_current_goals_level": True,
            "instantiated_plan": proposed_plan_name,
            "success": True,
            "internal_thought": None
        }
        return response
    
    def insert_complex_plan(self, current_goal:Goal, internal_thought:str, complex_plan:dict, output_recommended_plan:bool, output_recommended_tools:bool, add_consolidator:bool=True, after_id:str=None) -> dict:

        log(DEBUG, f"Adding new complex plan:\n{complex_plan}")
        goal_instances = self.get_goals_instances_from_complex_plan(current_goal.goal_plan, current_goal, complex_plan, output_recommended_plan, output_recommended_tools)

        # insert the new instances as subgoals of the current goal
        with current_goal.goal_plan.plan_lock:
            inserted = current_goal.insert_goals(goal_instances, after_id=after_id, instantiated_plan=None)

            # add the CONSOLIDATOR as a last goal of the subgoals, if not already the last goal
            if add_consolidator and current_goal.goal_subgoals is not None and len(current_goal.goal_subgoals) > 0 and current_goal.goal_subgoals[-1].goal_target != GoalTarget.CONSOLIDATOR:
                # generate a list of dependencies, which is an array of negative integers. The previous goal is at index -1, the one before that at index -2, etc.
                all_previous_goal_ids = list(range(-1, -len(goal_instances)-1, -1))
                consolidator = Goal(None, None, "Consolidator", "The consolidator goal is the last goal of the complex plan.", GoalTarget.CONSOLIDATOR, None, all_previous_goal_ids)
                current_goal.insert_goals([consolidator], after_id="last", instantiated_plan=None)

            if app_config.testing and current_goal.goal_plan is not None:
                log_plan(current_goal.goal_plan)

        result = {
            "restart_current_goals_level": True,
            "internal_thought": internal_thought,
            "success": True
        }
        return result

    def prepare_tool_goals(self, tools_call:dict, goal:Goal, tools_info:List[dict]) -> dict:
        # determine which plan to insert: if the plan forbids to answer the user, then use the plan_execute_tool_no_answer
        plan_name = "plan_execute_tool"     # default: execute the tool and answer the user
        assistant_needed = True
        templates = LlmManager.instance().get_agent_plans_template()
        if goal.goal_instantiated_plan is not None and goal.goal_instantiated_plan in templates:
            plan_template = templates[goal.goal_instantiated_plan]
            if "assistant_answer" in plan_template and plan_template["assistant_answer"]==False:
                assistant_needed = False
        if assistant_needed and goal.goal_recommended_plan is not None and len(goal.goal_recommended_plan) > 0:
            recommended_plan_name = goal.goal_recommended_plan.lstrip('+')
            if recommended_plan_name in templates:
                plan_template = templates[recommended_plan_name]
                if "assistant_answer" in plan_template and plan_template["assistant_answer"] == False:
                    assistant_needed = False

        # check if the next goal is an assistant already. If so, do not insert another assistant
        next_goal = goal.goal_parent.get_next_goal(with_target=GoalTarget.AGENT, with_target_agent_name="AgentAssistant", not_before=goal)
        if next_goal is not None:
            assistant_needed = False

        # avoid inserting an unwanted or redundant assistant
        if not assistant_needed:
            plan_name = "plan_execute_tool_no_answer"
        
        plan_fragment = get_goals_from_plan_template(goal.goal_plan, plan_name, len(tools_call))
        idx = 0
        summaries = []
        for tool in tools_call:
            # convert the tool call from string to json structure if necessary
            if isinstance(tool, str):
                tool = get_json_from_text(tool)
            
            # guard against malformed tool descriptions coming from LLMs
            tool_type = tool.get("type", "function") if isinstance(tool, dict) else "function"
            if tool_type == "function":
                fname = tool.get("fname")
                if fname is None:
                    fname = tool.get("name")
                args = tool.get("arguments")
                tool_call_id = tool.get("id") if isinstance(tool, dict) else None
                # if fname or args are missing, coerce to safe defaults
                if fname is None:
                    fname = ""
                if args is None:
                    args = {}
                if fname.startswith("functions."):
                    # workaround for the function name: remove the "functions." prefix, which starting with OpenAI gpt-4.1 comes around at random
                    fname = fname[len("functions."):]

                plan_fragment[idx].goal_status = GoalStatus.NOT_STARTED
                plan_fragment[idx].goal_recommended_tools = goal.goal_recommended_tools
                plan_fragment[idx].goal_target_agent.agent_fname = fname
                plan_fragment[idx].goal_target_agent.agent_args = args
                plan_fragment[idx].goal_target_agent.agent_description = f"Calling the function: '{fname}' with arguments: {args}"
                plan_fragment[idx].goal_target_agent.agent_tool_call_id = tool_call_id
                plan_fragment[idx].goal_sequence_id = goal.goal_sequence_id
                message = f"FUNCTION CALL: '{fname}' Args: {args}"
                summaries.append(message)
            else:
                log(WARNING, f"'WARNING: {self.agent_name}' Agent received an invalid tool call description from the LLM: {tool}")
                plan_fragment[idx].goal_status = GoalStatus.COMPLETED
            idx += 1
        summary = "\n".join(summaries)

        with goal.goal_plan.plan_lock:
            # insert the plan after the current goal
            inserted = goal.goal_parent.insert_goals(plan_fragment, after_id=goal.get_id_chain(), instantiated_plan=plan_name)

        response = {
            "restart_current_goals_level": True,
            "summary": summary,
            "success": True,
            "internal_thought": None
        }
        return response
    
    def insert_plan(self, goal:Goal, tools_info:List[dict], proposed_plan:Union[list,dict], output_recommended_plan:bool, output_recommended_tools:bool)->dict:
        if isinstance(proposed_plan, list):
            # the proposed plan is a list of tools to be called
            tool_list = [tool["fname"] for tool in proposed_plan]
            log(DEBUG, f"'{self.agent_name}' has proposed a list of tools to be called: {tool_list}")

            # prepare invocation of the tools
            response = self.prepare_tool_goals(proposed_plan, goal, tools_info)
        elif isinstance(proposed_plan, dict):
            # the proposed plan is a simple or complex plan
            internal_thought = proposed_plan["optional_internal_thought"] if "optional_internal_thought" in proposed_plan else None
            internal_thought = f"PRESERVE_ENGLISH=[[{internal_thought}]]" if internal_thought is not None else None
            if "plan" in proposed_plan:
                # simple plan: instantiate the new plan proposed by the LLM as a string that indicates which templates to instantiate
                recommended_tools = proposed_plan.get("recommended_tools", None)
                if recommended_tools is None:
                    recommended_tools = goal.goal_recommended_tools
                # Ensure recommended_tools is always a List[str]
                if recommended_tools is not None:
                    if isinstance(recommended_tools, str):
                        recommended_tools = [recommended_tools]
                    elif not isinstance(recommended_tools, list):
                        recommended_tools = None
                goal.goal_recommended_tools = recommended_tools
                response = self.insert_simple_plan(goal, proposed_plan["plan"], dependent_on_previous_goal=False, output_recommended_plan=output_recommended_plan, output_recommended_tools=output_recommended_tools)
            elif "optional_complex_plan" in proposed_plan:
                # complex plan: instantiate the new plan proposed by the LLM
                response = self.insert_complex_plan(goal, internal_thought, proposed_plan, output_recommended_plan, output_recommended_tools)
            else:
                # error: the proposed plan is neither a simple nor a complex plan. This should not happen
                log(ERROR, f"'ERROR: {self.agent_name}' Agent has failed to calculate a new plan.")
                response = {
                    "success": False,
                    "error": True,
                    "error_message": f"The proposed plan is neither a simple nor a complex plan: {proposed_plan}",
                    "internal_thought": None
                }
                return response
        else:
            # error: the proposed plan is neither a list nor a dictionary. This should not happen
            log(ERROR, f"'ERROR: {self.agent_name}' Agent has failed to calculate a new plan.")
            response = {
                "success": False,
                "error": True,
                "error_message": f"The proposed plan is neither a list nor a dictionary: {proposed_plan}",
                "internal_thought": None
            }

        return response

    def rollback(self, goal: Goal) -> bool:
        return super().rollback(goal)

    def run(self, goal:Goal) -> dict:
        self.raise_if_plan_stopped()
        log(DEBUG, f"'{self.agent_name}' Agent is running...")
        plan_fragment = None
        response = None

        # calculates the new plan. The return values are:
        # - an array of tools (=functions) to be called, or
        # - a dictionary containing details about the new plan to be added after the current goal
        tools_info, proposed_plan, repeat_contextualizer = self.ask_LLM_about_macro_plan(goal)
        self.raise_if_plan_stopped()

        if proposed_plan is not None:
            # insert the new plan
            response = self.insert_plan(goal, tools_info, proposed_plan, output_recommended_plan=True, output_recommended_tools=True)

            if repeat_contextualizer:
                # the exact <DATE_TIME> context was just enabled: ask Goal.py to re-run the
                # Contextualizer before the next subgoal (e.g. AgentToolPreparer) builds its messages
                response["repeat_contextualizer"] = True

            if plan_fragment is not None:
                log(DEBUG, f"'{self.agent_name}' has proposed a new plan. Here's the revised plan:\n" + \
                           f"{goal.goal_plan}")
        else:
            log(WARNING, f"'{self.agent_name}' has failed to calculate a new plan.")
            response = {
                "success": False,
                "error": True,
                "error_message": self._last_planner_error or "AgentPlanner failed to calculate a new plan.",
                "internal_thought": None
            }

        return response


log_end_file(__file__)


if __name__ == '__main__':
    from src.utils.unit_test import init_unit_test
    from src.utils.log import log_init, log_exit, FATAL
    init_unit_test()
    log_init()

    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        global passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))
            failed += 1

    try:
        # ── Test 1: Single-goal RAG plan ──────────────────────────────────────
        yaml_plan = """optional_complex_plan:
  goals:
    - goal_sequence_id: "Step-1"
      goal: "Fetch information about Charly the Dog"
      dependent_on_sequence_ids: []
      recommended_plan: "plan_rag_assistant_and_answer"
optional_require_exact_time: false
"""
        parsed = get_yaml_from_text(yaml_plan)
        check("Single-goal plan parsed to dict", isinstance(parsed, dict), str(parsed))
        goals = parsed.get("optional_complex_plan", {}).get("goals", [])
        check("Single-goal: one goal", len(goals) == 1)
        check("Single-goal: correct plan", goals[0].get("recommended_plan") == "plan_rag_assistant_and_answer")

        # ── Test 2: Multi-goal consolidated plan ──────────────────────────────
        yaml_multi = """optional_complex_plan:
  goals:
    - goal_sequence_id: "Step-1"
      goal: "Retrieve info about Charly from documents"
      dependent_on_sequence_ids: []
      recommended_plan: "plan_rag_assistant_no_answer"
    - goal_sequence_id: "Step-2"
      goal: "Fetch current temperature for Paris"
      dependent_on_sequence_ids: []
      recommended_plan: "plan_call_tool_but_no_answer"
      optional_recommended_tools:
        - "WeatherInfo"
    - goal_sequence_id: "Step-3"
      goal: "Consolidated answer about Charly and Paris temperature"
      dependent_on_sequence_ids:
        - "Step-1"
        - "Step-2"
      recommended_plan: "plan_simple_answer"
optional_require_exact_time: true
"""
        parsed = get_yaml_from_text(yaml_multi)
        check("Multi-goal plan parsed", isinstance(parsed, dict), str(parsed))
        goals = parsed.get("optional_complex_plan", {}).get("goals", [])
        check("Multi-goal: three goals", len(goals) == 3)
        check("Multi-goal: step-2 has tool", goals[1].get("optional_recommended_tools") == ["WeatherInfo"])
        check("Multi-goal: step-3 deps correct",
              goals[2].get("dependent_on_sequence_ids") == ["Step-1", "Step-2"])
        check("Multi-goal: require_exact_time=True", parsed.get("optional_require_exact_time") == True)

        # ── Test 3: Pre-built plan (no optional_complex_plan) ─────────────────
        yaml_prebuilt = """optional_prebuilt_complex_plan_id: "pre-built-plan-deep-rag"
"""
        parsed = get_yaml_from_text(yaml_prebuilt)
        check("Pre-built plan parsed", isinstance(parsed, dict), str(parsed))
        check("Pre-built plan id correct",
              parsed.get("optional_prebuilt_complex_plan_id") == "pre-built-plan-deep-rag")
        check("Pre-built: no optional_complex_plan emitted",
              "optional_complex_plan" not in parsed)

        # ── Test 4: INTERNAL_THOUGHT marker removal (system-prompt handling) ──
        prompt_with_marker = """INTERNAL_THOUGHT=[[
  optional_internal_thought: some reasoning
]]
optional_prebuilt_complex_plan_id: pre-built-plan-xxx
"""
        excluded = re.sub(r"INTERNAL_THOUGHT=\[\[.*?\]\]", "", prompt_with_marker, flags=re.DOTALL)
        check("INTERNAL_THOUGHT marker removed", "INTERNAL_THOUGHT" not in excluded)
        check("INTERNAL_THOUGHT: rest of prompt intact",
              "optional_prebuilt_complex_plan_id" in excluded)

    except Exception as e:
        log(FATAL, f"An error occurred during testing: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        print(f"\nResults: {passed} passed, {failed} failed")
        log_exit()
