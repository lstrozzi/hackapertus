"""
LLM Manager - Core Orchestration Class

This module contains the LlmManager singleton class that orchestrates all LLM operations,
configuration management, and provider coordination.
"""

from torch import log_
from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import os
import json
import threading
import traceback
import hashlib
import requests
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Union, Tuple
from jsonschema import Draft7Validator

from src.core.llm import (
    AgentRagSettings, reasoning_levels, 
    DEFAULT_ACTIVE_LLM_CONFIGURATION_NAME, PLATFORM_API_TOKEN, LLM_PING_TIMEOUT_SECONDS,
    DEFAULT_TEMPERATURE, USE_LLM_CACHE, MAX_COMPLETION_TOKENS, MAX_COMPLETION_TOKENS_UNLIMITED
)
from src.core.base.enums import AgentReasoningEnum
from src.core.base.llm_response import LlmResponse
from src.config.constants import USE_REAL_SSL_CERTIFICATE, BLAZOR_HOSTNAME, BLAZOR_PORT, BLAZOR_PROTOCOL, DEFAULT_FLAVOUR_IN_DEBUG, GLOBAL_SESSION_ID, MAX_RAG_MESSAGE_LENGTH_DEFAULT, MAX_LOADED_DOCUMENT_LENGTH_DEFAULT, WARN_IF_TRUNCATING_THRESHOLD_DEFAULT
from src.services.user_service import get_user_service
from src.core.extensions.skillset_management import import_class
from src.core.llm.interface import LlmInterface
from src.utils.log import log, DEBUG, INFO, LLM_IN, WARNING, ERROR, FATAL
from src.utils.system import get_os_type

if TYPE_CHECKING:
    from src.core.agents.Agent import Agent
    from src.core.thalaplan.goal.Goal import Goal
    from src.core.thalaplan.Plan import Plan


class LlmManager:
    """
    Singleton class that manages LLM configuration, providers, and orchestrates all LLM operations.
    
    This is the central coordination point for:
    - LLM configuration management
    - Provider selection and instantiation
    - Agent-to-model mappings
    - Reasoning and RAG settings
    - Plan template management
    """
    
    _instance: Optional['LlmManager'] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self):
        # LLM configuration
        self._llm_configuration: Optional[Dict[str, Any]] = None
        self._llm_interfaces: Dict[str, Optional[LlmInterface]] = {}          # key = model_provider [+ endpoint], value = LlmInterface instance

        # LLM active configuration
        self._llm_active_configuration_name: str = DEFAULT_ACTIVE_LLM_CONFIGURATION_NAME

        # Active settings
        self._active_flavour: str = "NEUTRAL"
        self._active_reasoning: AgentReasoningEnum = AgentReasoningEnum.COMPLEX
        self._active_rag_settings: AgentRagSettings = AgentRagSettings()

        # JSON schema cache
        self._json_schema_cache: Dict[str, dict] = {}

        # Plan fragments
        self._agent_plans_template = {}
        self._agent_plans_template_lock = threading.Lock()
        self._builtin_plans_initialized: bool = False
    
    @classmethod
    def instance(cls) -> 'LlmManager':
        """Get or create the singleton LlmManager instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    log(DEBUG, "Creating a new instance of LlmManager...")
                    cls._instance = LlmManager()
        return cls._instance
    
    def build_interface_key(self, model_provider: Optional[str], endpoint: Optional[str] = None) -> Optional[str]:
        """Build a unique key for LLM interface based on provider and endpoint."""
        if model_provider is None:
            return None
        if endpoint is not None and len(endpoint) > 0:
            key = f"{model_provider}|{endpoint}"
        else:
            key = model_provider
        return key

    def _read_llm_configuration(self) -> bool:
        """Read the LLM configuration from Blazor server or local JSON file."""
        running_in_docker = get_os_type() == "Linux" or os.getenv("LOCAL_AI_API_ENDPOINT") is not None  
        # log(DEBUG, f"_read_llm_configuration(): running_in_docker={running_in_docker}, because os_type={get_os_type()}")

        if running_in_docker:
            extended_configuration = self._fetch_configuration_from_blazor()
            self._update_flavour_and_rag(extended_configuration)
        
        elif self._llm_configuration is None:
            log(DEBUG, "fetching configuration from local file...")
            self._fetch_configuration_from_file(running_in_docker)
            if self._llm_configuration is None:
                log(FATAL, "No LLM configuration could be loaded.")
                return False
            self._update_flavour_and_rag(None)

        if not self._llm_configuration:
            log(FATAL, "_read_llm_configuration(): Could not fetch configuration from Blazor while running in Docker.")
            return False
        
        return True
    
    def _update_flavour_and_rag(self, extended_configuration: Optional[dict]) -> None:
        if extended_configuration is not None:
            active_flavour = extended_configuration.get("flavour", DEFAULT_FLAVOUR_IN_DEBUG)
            active_rag = extended_configuration.get("ragsettings", {})

            # Apply settings
            self.update_flavour_details(active_flavour)
            self.update_rag_settings(active_rag)

        return

    def _fetch_configuration_from_blazor(self) -> Optional[Dict]:
        """Fetch LLM configuration from Blazor server."""
        try:
            if get_user_service().get_session_id() is None or get_user_service().get_session_id() == GLOBAL_SESSION_ID:
                url = f"{BLAZOR_PROTOCOL}://{BLAZOR_HOSTNAME}:{BLAZOR_PORT}/v1/api/platform/config/llm"
            else:
                url = f"{BLAZOR_PROTOCOL}://{BLAZOR_HOSTNAME}:{BLAZOR_PORT}/v1/api/platform/config/llm/{get_user_service().get_session_id()}"
            response = requests.get(url, headers={"Authorization": f"Bearer {PLATFORM_API_TOKEN}"}, verify=USE_REAL_SSL_CERTIFICATE)
            
            if response.status_code == 200:
                configuration = response.json()
                if configuration and "llmconfiguration" in configuration:
                    self._llm_configuration = configuration["llmconfiguration"]
                    if "activeconfiguration" in configuration:
                        self._llm_active_configuration_name = configuration["activeconfiguration"]
                    return configuration
                
                else:
                    log(FATAL, f"""ERROR: The LLM Configuration could not be fetched from Blazor. Response: {response}
Reason: {response.reason}
Content: {response.content}""")
            else:
                log(FATAL, f"""wouldn't fetch the LLM Configuration from Blazor:\nUrl: {url}\nResponse: {response}
Reason: {response.reason}
Content: {response.content}""")

        except Exception as e:
            stack_trace = traceback.format_exc()
            log(FATAL, f"FATAL: Couldn't fetch the LLM Configuration from blazor server. Exception: {e}\nStack trace:\n{stack_trace}")
        
        return None

    def _fetch_configuration_from_file(self, running_in_docker: bool) -> Optional[Dict]:
        """Fetch LLM configuration from local JSON file."""
        if running_in_docker:
            log(WARNING, "LLM Configuration is None, it will be fetched later on if possible.")
            return None

        repository_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
        file_paths = [
            os.path.join(repository_root, "ThalabusWeb", "Configuration", "LlmSettings.json"),
            os.path.abspath("../ThalabusWeb/Configuration/LlmSettings.json"),
            os.path.abspath("ThalabusWeb/Configuration/LlmSettings.json")
        ]
        
        for path in file_paths:
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8-sig") as file:
                        self._llm_configuration = json.load(file)
                    self._llm_active_configuration_name = "DEFAULT"
                    log(DEBUG, f"Successfully loaded LLM configuration from {path}")
                    return self._llm_configuration
                
                except Exception as e:
                    stack = traceback.format_exc()
                    log(ERROR, f"ERROR loading {path}: {e}\nStack trace:\n{stack}")
            else:
                log(WARNING, f"LLM Configuration file not found at {path}")
        
        log(ERROR, f"LlmSettings.json not found in any expected location: {file_paths}")
        return None

    def get_json_schema(self, json_sample: str, agent_name: str, purpose: str) -> Optional[dict]:
        """Generate a JSON schema from a JSON sample."""
        def infer_schema(data):
            if isinstance(data, dict):
                return {
                    "type": "object",
                    "description": "This is a JSON object.",
                    "properties": {key: infer_schema(value) for key, value in data.items()},
                    "required": list(data.keys()),
                    "additionalProperties": False
                }
            elif isinstance(data, list):
                return {
                    "type": "array",
                    "items": infer_schema(data[0]) if data else {}
                }
            elif isinstance(data, bool):
                return {"type": "boolean", "description": "This is a boolean value."}
            elif isinstance(data, str):
                return {"type": "string", "description": data}
            elif isinstance(data, int):
                return {"type": "integer", "description": "This is an integer value."}
            elif isinstance(data, float):
                return {"type": "number", "description": "This is a float value."}
            elif data is None:
                return {"type": "null", "description": "This is a null value."}
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

        def generate_cache_key(json_sample: str, agent_name: str, purpose: str) -> str:
            return hashlib.md5(f"{json_sample}_{agent_name}_{purpose}".encode()).hexdigest()

        try:
            # Check if the schema is already cached
            cache_key = generate_cache_key(json_sample, agent_name, purpose)
            if cache_key in self._json_schema_cache:
                log(DEBUG, "Using cached JSON schema.")
                return self._json_schema_cache[cache_key]

            # Parse the JSON sample
            if isinstance(json_sample, str):
                json_data = json.loads(json_sample)
            else:
                json_data = json_sample

            # Generate the schema
            json_schema = infer_schema(json_data)

            # Validate the schema
            Draft7Validator.check_schema(json_schema)

            # Format for OpenAI interface
            json_schema = {
                "name": f"{agent_name}_{purpose}",
                "schema": json_schema,
                "strict": True
            }

            # Cache the schema
            self._json_schema_cache[cache_key] = json_schema
            log(DEBUG, f"Generated and cached JSON schema: {json.dumps(json_schema, indent=2, ensure_ascii=False)}")

            return json_schema
    
        except Exception as e:
            stack = traceback.format_exc()
            log(ERROR, f"ERROR generating JSON schema: {e}\nStack trace:\n{stack}")

        return None
    
    def get_agent_config_name(self, agent_name: str, agent_purpose: Optional[str]) -> str:
        """Generate configuration name for agent and purpose combination."""
        config_code = agent_name + "_" + agent_purpose if agent_purpose is not None and len(agent_purpose) > 0 else agent_name
        return config_code

    def get_agent_reasoning(self, agent_name: str, purpose: Optional[str] = None) -> Optional[AgentReasoningEnum]:
        """Get configured maximum reasoning for an agent from settings.
        
        Args:
            agent_name: Name of the agent
            purpose: Purpose of the LLM call (optional)
            
        Returns:
            AgentReasoningEnum if configured, None otherwise (uses active_reasoning)
        """
        if self._llm_configuration is None:
            return None
        
        agents = self._llm_configuration.get("agents", {})
        config_code = self.get_agent_config_name(agent_name, purpose)
        agent_config = agents.get(config_code, agents.get("DEFAULT", {}))
        
        reasoning_str = agent_config.get("reasoning")
        if reasoning_str is None or reasoning_str == "":
            return None
            
        try:
            return AgentReasoningEnum[reasoning_str.upper()]
        except (KeyError, AttributeError):
            log(WARNING, f"Invalid reasoning value '{reasoning_str}' for agent '{config_code}'")
            return None

    def get_thinking_token_budget_for(self, reasoning_level: Optional[AgentReasoningEnum]) -> tuple[bool, Optional[int], Optional[str]]:
        """Get thinking token budget configuration for a given reasoning level.
        
        Args:
            reasoning_level: AgentReasoningEnum level or None
            
        Returns:
            Tuple of (enable_thinking: bool, budget: int | None, think_sentence: str | None)
        """
        from src.core.llm.config import THINKING_TOKEN_BUDGET
        return THINKING_TOKEN_BUDGET.get(reasoning_level, (True, None, None))

    def get_reasoning_level_display(self, reasoning_level: Optional[AgentReasoningEnum]) -> str:
        """Get user-friendly display label for a reasoning level with token budget info.
        
        Args:
            reasoning_level: AgentReasoningEnum level or None
            
        Returns:
            Display string like "MEDIUM (4K tokens)"
        """
        from src.core.llm.config import REASONING_LEVEL_DISPLAY
        if reasoning_level is None:
            return "(default - use active reasoning)"
        return REASONING_LEVEL_DISPLAY.get(reasoning_level, str(reasoning_level))

    def _init_builtin_plans(self) -> dict:
        """Initialize built-in plan fragments."""
        if self._builtin_plans_initialized:
            return self._agent_plans_template

        self._builtin_plans_initialized = True
        log(DEBUG, "initializing the built-in plan fragments...")

        with self._agent_plans_template_lock:
            from src.core.agents.AgentPlanner import init_agent_planner
            init_agent_planner()

        return self._agent_plans_template

    def _log_plan_fragments(self) -> None:
        """Log debug information about plan fragments."""
        log(DEBUG, f"'{__name__}' has initialized the plan fragments:")
        templates = self.get_agent_plans_template()
        
        for plan_key, plan_data in templates.items():
            plan_name = plan_data.get("name", "Unknown")
            plan_description = plan_data.get("description", "").replace('\n', '  ') if plan_data.get("description") else ""
            plan_template_agents = plan_data.get("template_agents")
            plan_agents = plan_data.get("template")
            plan_agents_count = (len(plan_template_agents) if plan_template_agents 
                               else len(plan_agents) if plan_agents else 0)
            is_custom_plan = " [CUSTOM_PLAN]" if plan_data.get("customplan", False) else ""
            
            log(DEBUG, f"\n- {plan_key}= {plan_name} [#{plan_agents_count} agent(s)]{is_custom_plan}: '{plan_description}'")

    def _reload_custom_plans(self) -> dict:
        """Reload custom plans from LLM configuration."""
        llmconfigs = self.get_llm_model_configurations()
        if (llmconfigs and "plan_fragments" in llmconfigs and 
            len(llmconfigs["plan_fragments"]) > 0):
            
            with self._agent_plans_template_lock:
                plan_fragments = llmconfigs["plan_fragments"]
                
                for plan_key, plan_data in plan_fragments.items():
                    if plan_data.get("customplan", False):
                        log(DEBUG, f"Adding custom plan '{plan_key}' to the agent plans template...")
                        self.add_or_update_plan_fragment(plan_key, plan_data)

        return self._agent_plans_template

    def get_agent_plans_template(self) -> dict:
        """Get agent plans template, initializing if necessary."""
        if not self._agent_plans_template:
            self._init_builtin_plans()
            self._reload_custom_plans()
            
        return self._agent_plans_template

    def add_or_update_plan_fragment(self, plan_key: str, plan: dict) -> None:
        """Add or update a plan fragment in the template."""
        # Convert template_agents to template format if needed
        if ("template" not in plan and plan.get("customplan", False) and 
            "template_agents" in plan):
            
            self._convert_template_agents_to_template(plan)

        self._agent_plans_template[plan_key] = plan
        log(DEBUG, f"the custom plan '{plan_key}' was added.")

    def _convert_template_agents_to_template(self, plan: dict) -> None:
        """Convert template_agents format to template format."""
        try:
            goal_class = import_class("src.core.thalaplan.goal.Goal", "Goal")
            goal_target = import_class("src.core.thalaplan.goal.Goal", "GoalTarget")
            llm_config = self.get_llm_model_configurations()
            llm_agents = llm_config.get("agents", {}) if llm_config else {}
            template_agents = plan["template_agents"]
            template = []

            for template_agent in template_agents:
                agent_name = template_agent["name"]
                agent_dependencies = template_agent.get("dependencies", [])
                agent_definition = llm_agents.get(agent_name)
                
                if agent_definition is None:
                    log(ERROR, f"ERROR: the agent '{agent_name}' was not found in the LLM configuration.")
                    continue
                    
                agent_description = agent_definition.get("description")
                
                # Determine agent class
                if (agent_name in llm_agents and llm_agents[agent_name] and 
                    not llm_agents[agent_name].get("customagent", False)):
                    agent_class_name = agent_name
                else:
                    agent_class_name = "AgentCustom"
                
                agent_class = import_class(f"src.core.agents.{agent_class_name}", agent_class_name)
                
                # Create goal
                goal = goal_class(None, None, agent_name, agent_description, 
                                goal_target.AGENT, agent_class, agent_dependencies, None)
                goal.set_target_agent_name(agent_name)
                goal.goal_prompt = template_agent.get("prompt") if "prompt" in template_agent and template_agent.get("prompt") is not None else None
                goal.goal_silent_summary = template_agent.get("silent_summary", False)
                goal.goal_repeat_on_error = template_agent.get("repeat_on_error", False)
                goal.goal_purpose = template_agent.get("purpose")
                goal.goal_override_recommended_tools = template_agent.get("override_recommended_tools")
                
                if goal.goal_purpose is not None and len(goal.goal_purpose) == 0:
                    goal.goal_purpose = None
                    
                template.append(goal)

            plan["template"] = template
            
        except Exception as e:
            log(ERROR, f"Error converting template_agents to template: {e}")

    def update_llm_configuration(self) -> None:
        """Update LLM configuration and reset interfaces."""
        self._llm_interfaces = {}  # reset all interfaces

        log(DEBUG, "Reloading LLM configuration...")
        self._read_llm_configuration()

        log(DEBUG, "Reloading Agent plans template...")
        self._init_builtin_plans()

        log(DEBUG, "Reloading custom plans...")
        self._reload_custom_plans()

        log(DEBUG, "Updating active configuration...")
        self._log_plan_fragments()

        log(DEBUG, "Llm configuration has been reloaded, and Agent plans template has been reset.")

    def convert_workflow_to_json(self, workflow: Dict) -> Dict:
        """Convert workflow format to JSON format."""
        goals = []
        for objective in workflow.get("objectives", []):
            objective_enabled = objective.get("enable", True)
            goal = {
                "goal_sequence_id": objective["id"],
                "goal": objective["prompt"] if objective_enabled else None,
                "dependent_on_sequence_ids": objective.get("dependent_on", []),
                "recommended_plan": objective.get("recommended_plan"),
                "optional_recommended_tools": objective.get("optional_recommended_tools", []),
                "focus_iteration": objective.get("focus_iteration"),
                "focus_on_document": objective.get("focus_on_document"),
                "focus_reasoning_division": objective.get("focus_reasoning_division"),
                "focus_regex": objective.get("focus_regex"),
                "focus_instr": objective.get("focus_instr")
            }
            goals.append(goal)
        
        return {"goals": goals}

    def get_workflows(self) -> Optional[Dict]:
        """Get available workflows from configuration."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        return self._llm_configuration.get("workflows") if self._llm_configuration else None
    
    def get_workflow_by_id(self, workflow_id: str) -> Optional[Dict]:
        """Get a specific workflow by ID."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        workflows = self.get_workflows()
        if workflows:
            for workflow in workflows:
                if workflow.get("id") == workflow_id:
                    return self.convert_workflow_to_json(workflow)
            
        return None

    def get_mcp_settings(self) -> Optional[Dict]:
        """Get MCP (Model Context Protocol) settings."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if self._llm_configuration is None:
            log(FATAL, "No LLM configuration found.")
            return None

        return self._llm_configuration.get("mcp")

    def get_cache_settings(self) -> Optional[Dict]:
        """Get cache configuration settings."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if self._llm_configuration is None:
            log(FATAL, "No LLM configuration found.")
            return None

        cache_configuration = self._llm_configuration.get("cache_configuration")
        if cache_configuration is None:
            return {"enable_cache": True}

        return {
            "enable_cache": cache_configuration.get("enable_cache", True)
        }

    def get_quality_settings(self) -> Dict:
        """Get quality configuration settings."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if self._llm_configuration is None:
            return {}

        return self._llm_configuration.get("quality_configuration") or {}

    def get_embeddings_settings(self) -> Optional[Dict]:
        """Get embeddings configuration settings."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if not self._llm_configuration or "embeddings" not in self._llm_configuration:
            log(ERROR, "get_embeddings_settings(): No embeddings settings found in the LLM configuration.")
            return None
        
        return self._llm_configuration["embeddings"]
    
    def get_reranking_settings(self) -> Optional[Dict]:
        """Get reranking configuration settings."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if not self._llm_configuration or "reranking" not in self._llm_configuration:
            log(ERROR, "get_reranking_settings(): No reranking settings found in the LLM configuration.")
            return None

        reranking = self._llm_configuration.get("reranking")
        if reranking is None:
            log(ERROR, "get_reranking_settings(): Reranking settings are None in the LLM configuration.")
            return None
        
        reranking_settings: dict = {
            "enabled": reranking.get("use_reranking_model", False),
            "model": reranking.get("model", None),
            "endpoint": reranking.get("endpoint", None)
        }
        
        return reranking_settings

    def get_max_rag_message_length(self) -> int:
        """Get the configured maximum RAG message length, or its default."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        configured_value = self._llm_configuration.get("max_rag_message_length") if self._llm_configuration else None
        return configured_value if isinstance(configured_value, int) and configured_value > 0 else MAX_RAG_MESSAGE_LENGTH_DEFAULT

    def get_max_loaded_document_length(self) -> int:
        """Get the configured maximum loaded document length, or its default."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        configured_value = self._llm_configuration.get("max_loaded_document_length") if self._llm_configuration else None
        return configured_value if isinstance(configured_value, int) and configured_value > 0 else MAX_LOADED_DOCUMENT_LENGTH_DEFAULT

    def get_warn_if_truncating_threshold(self) -> float:
        """Get the ranking threshold for warnings about truncated RAG fragments."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        configured_value = self._llm_configuration.get("warn_if_truncating_threshold") if self._llm_configuration else None
        return configured_value if isinstance(configured_value, (int, float)) and configured_value >= 0 else WARN_IF_TRUNCATING_THRESHOLD_DEFAULT

    def get_crawler_confluence_settings(self) -> Optional[Dict]:
        """Get Confluence crawler settings."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if not self._llm_configuration:
            return None
            
        crawler = self._llm_configuration.get("content_crawler")
        return crawler.get("confluence") if crawler else None

    def get_crawler_web_settings(self) -> Optional[List[Dict]]:
        """Get web crawler settings (content_crawler.web): one entry per configured web site."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if not self._llm_configuration:
            return None

        crawler = self._llm_configuration.get("content_crawler")
        return crawler.get("web") if crawler else None

    def get_default_model(self) -> Optional[Dict]:
        """Get the default model configuration."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if self._llm_configuration is None:
            log(FATAL, "No LLM configuration found.")
            return None

        for model in self._llm_configuration.get("models", []):
            if model.get("id") == "DEFAULT":
                return model

        return None

    def get_default_model_for_skills(self) -> Optional[str]:
        """Get the default model configuration for skills."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if self._llm_configuration is None:
            log(FATAL, "No LLM configuration found.")
            return None

        for skill in self._llm_configuration.get("skill_configurations", []):
            if skill.get("name") == "DEFAULT":
                return skill.get("configured_model")

        return None
        
    def get_model_by_skill_id(self, skill_id: str) -> Optional[str]:
        """Get model configuration by skill ID."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if self._llm_configuration is None:
            log(FATAL, "No LLM configuration found.")
            return None

        skills = self._llm_configuration.get("skill_configurations", [])
        for skill in skills:
            if skill.get("skill_id") == skill_id:
                model = skill.get("configured_model")
                if not model or model == "DEFAULT":
                    default_skill = self.get_default_model_for_skills()
                    if default_skill == "DEFAULT" or not default_skill:
                        default_model = self.get_default_model()
                        model = default_model.get("model") if default_model else None
                    else:
                        model = default_skill
                return model

        return None

    def get_active_configuration(self) -> Optional[Dict]:
        """Get the currently active LLM configuration."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if self._llm_configuration is None:
            log(FATAL, "No LLM configuration found.")
            return None

        configurations = self._llm_configuration.get("configurations", [])
        default_configuration = None
        active_configuration = None

        for config in configurations:
            if config.get("name") == self._llm_active_configuration_name:
                active_configuration = config
            if config.get("name") == "DEFAULT":
                default_configuration = config

        return active_configuration or default_configuration
    
    def set_active_configuration_name(self, configuration_name: str) -> bool:
        """Set the active configuration name."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if self._llm_configuration is None:
            log(FATAL, "No LLM configuration found.")
            return False

        # Verify validity of the new configuration name
        configurations = self._llm_configuration.get("configurations", [])
        configuration_found = any(config.get("name") == configuration_name for config in configurations)
        
        if not configuration_found:
            return False

        self._llm_interfaces = {}  # reset all interfaces
        self._llm_active_configuration_name = configuration_name
        return True

    def get_agent_model_mapping(self, agent_name: str, agent_purpose: Optional[str]) -> Optional[str]:
        """Get the model mapping for a specific agent and purpose."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if self._llm_configuration is None:
            log(FATAL, "No LLM configuration found.")
            return None

        active_configuration = self.get_active_configuration()
        if not active_configuration:
            return None
            
        # Get agent model mapping
        mapping_key = "agentmodelmapping" if "agentmodelmapping" in active_configuration else "agentModelMapping"
        agent_model_mapping = active_configuration.get(mapping_key, {})
        
        config_code = self.get_agent_config_name(agent_name, agent_purpose)
        mapping = agent_model_mapping.get(config_code)
        
        if mapping is None:
            mapping = agent_model_mapping.get("DEFAULT")

        return mapping

    def get_llm_model_configurations(self) -> Optional[Dict[str, Any]]:
        """Get all LLM model configurations."""
        if self._llm_configuration is None:
            self._read_llm_configuration()
            
        return self._llm_configuration

    def get_llm_model_configuration(self, model_config_id: Optional[str] = DEFAULT_ACTIVE_LLM_CONFIGURATION_NAME) -> Optional[Dict]:
        """Get a specific LLM model configuration by ID."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if self._llm_configuration is None:
            log(FATAL, "No LLM configuration found.")
            return None

        models = self._llm_configuration.get("models", [])
        model_configuration = None
        model_configuration_default = None
        
        for config in models:
            if config.get("id") == model_config_id:
                model_configuration = config
            if config.get("id") == "DEFAULT":
                model_configuration_default = config
        
        return model_configuration or model_configuration_default
    
    def get_llm_model_configuration_by_model_name(self, model_name: Optional[str], model_provider: Optional[str]) -> Optional[Dict]:
        """Get LLM model configuration by model name and provider."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if self._llm_configuration is None:
            log(FATAL, "No LLM configuration found.")
            return None

        models = self._llm_configuration.get("models", [])
        for config in models:
            config_model_id = config["id"].lower() if "id" in config and isinstance(config["id"], str) else None
            config_model_name = config["model"].lower() if "model" in config and isinstance(config["model"], str) else None
            config_model_provider = config["modelprovider"].lower() if "modelprovider" in config and isinstance(config["modelprovider"], str) else None
            if ((model_name is None or config_model_name == model_name.lower() or config_model_id == model_name.lower()) and (model_provider is None or config_model_provider == model_provider.lower())):
                return config

        log(WARNING, f"No model configuration found for model: {model_name}, provider: {model_provider}")
        return None
    
    def get_llm_model_configuration_for_agent(self, agent: 'Agent', agent_purpose: Optional[str] = None) -> Optional[Dict]:
        """Get LLM model configuration for a specific agent."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if self._llm_configuration is None:
            log(FATAL, "No LLM configuration found.")
            return None

        if agent is not None and agent.agent_force_model is None:
            model_configuration_id = self.get_agent_model_mapping(agent.agent_name, agent_purpose)
            model_configuration = self.get_llm_model_configuration(model_config_id=model_configuration_id)
        else:
            force_model = agent.agent_force_model if agent is not None else None
            model_provider = agent.agent_force_model_provider if agent is not None else None
            model_configuration = self.get_llm_model_configuration_by_model_name(model_name=force_model, model_provider=model_provider)

        return model_configuration

    def get_agent_prompt_engineering(self, agent_name: str, agent_purpose: Optional[str]) -> Optional[Dict]:
        """Get prompt engineering configuration for an agent."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if self._llm_configuration is None:
            log(FATAL, "No LLM configuration found.")
            return None

        agents = self._llm_configuration.get("agents", {})
        config_code = self.get_agent_config_name(agent_name, agent_purpose)
        agent_model = self.get_agent_model_mapping(agent_name, agent_purpose)

        # Find prompt engineering configurations
        default_prompt_engineering = agents.get("DEFAULT", {}).get("promptengineering", {})
        agent_prompt_engineering = agents.get(config_code, {}).get("promptengineering", default_prompt_engineering)

        # Find specific configuration for the model
        return agent_prompt_engineering.get(agent_model, agent_prompt_engineering.get("DEFAULT", {}))

    def get_interface(self, model_provider: str, endpoint: str):
        """Get LLM client for the specified provider."""
        key = self.build_interface_key(model_provider, endpoint)
        if key is None:
            log(ERROR, "get_interface(): model_provider is None, cannot build interface key.")
            return None
        
        if key not in self._llm_interfaces:
            self._llm_interfaces[key] = self._create_llm_interface(model_provider)

        interface = self._llm_interfaces[key]
        return interface

    def get_client(self, model_provider: str, endpoint: str, agent: 'Agent', purpose: str, model:Optional[str]):
        """Get LLM client for the specified provider."""
        key = self.build_interface_key(model_provider, endpoint)

        if key is None:
            log(ERROR, "get_client(): model_provider is None, cannot build interface key.")
            return None
        
        if key not in self._llm_interfaces:
            self._llm_interfaces[key] = self._create_llm_interface(model_provider)

        interface = self._llm_interfaces[key]
        client = interface._get_client(agent, purpose, model=model, model_provider=model_provider, endpoint=endpoint) if interface is not None else None
        return client

    def _create_llm_interface(self, model_provider: Optional[str]) -> Optional['LlmInterface']:
        """Create LLM interface for the specified provider."""
        if model_provider == "openai":
            from src.core.llm.providers.provider_openai import LlmOpenAI
            return LlmOpenAI()
        
        elif model_provider == "groq":
            from src.core.llm.providers.provider_groq import LlmGroq
            return LlmGroq()
        
        elif model_provider == "mistral":
            from src.core.llm.providers import LlmMistral
            if LlmMistral is None:
                log(ERROR, "Mistral provider is disabled (mistralai package could not be imported)")
                return None
            return LlmMistral()
        
        elif model_provider == "qwen":
            from src.core.llm.providers.provider_qwen import LlmQwen
            return LlmQwen()
        
        elif model_provider == "apertus":
            from src.core.llm.providers.provider_apertus import LlmApertus
            return LlmApertus()
        
        elif model_provider == "endpoint":
            from src.core.llm.providers.provider_endpoint import LlmEndpoint
            return LlmEndpoint()
        
        elif model_provider == "http":

            from src.core.llm.providers.provider_http import LlmHttpRequest
            return LlmHttpRequest()
        
        elif model_provider == "thalabus":
            from src.core.llm.providers.provider_thalabus import LlmThalabus
            return LlmThalabus()
        
        else:
            raise Exception(f"Unsupported API type: {model_provider}")

    def get_system_prompt(self, agent: 'Agent', agent_name: Optional[str] = None, purpose: Optional[str] = None, label: Optional[str] = None) -> Optional[str]:
        """Get system prompt for an agent."""
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if self._llm_configuration is None:
            log(FATAL, "No LLM configuration found.")
            return None

        if agent_name is None and agent is not None:
            agent_name = agent.agent_name
        if agent_name is None:
            log(ERROR, "get_system_prompt(): No agent name provided.")
            return None
        if "::" in agent_name:
            agent_name = agent_name.split("::")[-1]
        if "." in agent_name:
            agent_name = agent_name.split(".")[-1]

        agent_config_name = self.get_agent_config_name(agent_name, purpose)
        agents = self._llm_configuration.get("agents", {})
        
        if agent_config_name not in agents:
            log(ERROR, f"get_system_prompt(): No agent configuration found for '{agent_config_name}'.")
            return None
            
        agent_prompts = agents[agent_config_name]
        prompt_engineering = agent_prompts.get("promptengineering", {})
        
        active_configuration = self.get_active_configuration()
        active_configuration_name = active_configuration.get("name") if active_configuration else None
        
        if not prompt_engineering or not active_configuration_name:
            log(ERROR, f"get_system_prompt(): No prompt engineering found for agent '{agent_config_name}' or no active configuration.")
            return None
            
        active_prompt_engineering = (prompt_engineering.get(active_configuration_name) or prompt_engineering.get("DEFAULT"))
        if not active_prompt_engineering:
            log(ERROR, f"get_system_prompt(): No active prompt engineering found for agent '{agent_config_name}' and configuration '{active_configuration_name}'.")
            return None

        if label is None:
            return active_prompt_engineering.get("systemprompt")
        else:
            other_prompts = active_prompt_engineering.get("other", [])
            for item in other_prompts:
                if item.get("key").lower() == label.lower():
                    return item.get("value")
        
        return None

    def get_model(self, agent: Optional['Agent'], purpose: Optional[str] = None, model_input: Optional[str] = None, model_provider: Optional[str] = None, endpoint: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[Any]]:
        """Get model, provider, and interface for an agent or explicit model."""

        if agent is not None:
            if agent.agent_force_model is None:
                model_mapping = self.get_agent_model_mapping(agent.agent_name, purpose)
                llm_configuration = self.get_llm_model_configuration(model_mapping)
                if llm_configuration is None:
                    log(ERROR, f"No LLM configuration found for the agent '{agent.agent_name}' with purpose '{purpose}' using model mapping '{model_mapping}'.")
                    return None, None, None
                model = llm_configuration["model"]
                model_provider = llm_configuration.get("modelprovider") or llm_configuration.get("modelProvider")
                endpoint = llm_configuration.get("endpoint", None)
            else:
                model = agent.agent_force_model
                model_provider = agent.agent_force_model_provider
                endpoint = agent.agent_force_model_endpoint

            execution = f"{model_provider}:{model}"
            if execution not in agent.agent_execution:
                agent.agent_execution.append(execution)
        elif model_input is None:
            # Determine which model to use
            default_model = self.get_default_model()
            if default_model is None:
                log(FATAL, "No default model configuration found.")
                return None, None, None
            model = default_model["model"]
            model_provider = default_model["modelprovider"]
            endpoint = default_model.get("endpoint", None)
            log(DEBUG, f"Requested to use the default model: {model}")
        else:
            # Hard-coded, no agent name or purpose
            model = model_input
            if model_provider is None:
                # search for the provider given the model id
                llm_configuration = self.get_llm_model_configuration_by_model_name(model_name=model, model_provider=model_provider)
                if llm_configuration is None:
                    log(FATAL, f"No LLM configuration found for the model: {model}")
                    return None, None, None
                
                model_provider = llm_configuration.get("modelprovider") if llm_configuration else None
                endpoint = llm_configuration.get("endpoint") if llm_configuration else None
            log(DEBUG, f"Using hard-coded model: {model}")

        # Get or create LLM interface
        key = self.build_interface_key(model_provider, endpoint)
        if key is None:
            log(ERROR, "get_model(): model_provider is None, cannot build interface key.")
            return model, model_provider, None
        
        if key not in self._llm_interfaces:
            self._llm_interfaces[key] = self._create_llm_interface(model_provider)
            
        return model, model_provider, self._llm_interfaces[key]

    def is_agent_model_available(self, agent: 'Agent', purpose: Optional[str], log_level: int = WARNING) -> bool:
        """Check whether an agent's configured model is available before a finetune call."""
        available: bool = False

        model, model_provider, llm_interface = self.get_model(agent, purpose)
        if model is not None and model_provider is not None and llm_interface is not None:
            available = llm_interface.ping_available_models(
                agent=agent,
                purpose=purpose,
                model=model,
                timeout=LLM_PING_TIMEOUT_SECONDS,
                log_level=log_level)

            if not available:
                log(log_level, f"Model availability ping failed: provider='{model_provider}', agent='{agent.agent_name if agent else None}', model='{model}', purpose='{purpose}'.")

        return available

    def update_flavour_details(self, flavour: str) -> bool:
        """Update active flavour settings."""
        if not flavour:
            if self._active_flavour != flavour:
                log(INFO, "No flavour provided, using default (setting flavour to None).")
                self._active_flavour = flavour
            return False
        
        if self._llm_configuration is None:
            self._read_llm_configuration()

        if self._llm_configuration is None:
            log(FATAL, "No LLM configuration found.")
            return False

        flavours = self._llm_configuration.get("flavours", {})
        matched_flavour = self._find_matching_flavour(flavour, flavours)
        
        if matched_flavour:
            if self._active_flavour != matched_flavour:
                log(INFO, f"Selected flavour: '{matched_flavour}' for input: '{flavour}'")
                self._active_flavour = matched_flavour
            return True
        else:
            if get_user_service().get_session_id() != GLOBAL_SESSION_ID:
                log(WARNING, f"No flavour found matching: '{flavour}'. Available flavours: {list(flavours.keys())}")
            return False

    def _find_matching_flavour(self, flavour: str, flavours: Dict) -> Optional[str]:
        """Find matching flavour with case-insensitive and partial matching."""
        # Try exact match first (case-sensitive)
        if flavour in flavours:
            return flavour
        
        # Try case-insensitive exact match
        for existing_flavour in flavours.keys():
            if existing_flavour.lower() == flavour.lower():
                return existing_flavour
        
        # Try case-insensitive partial match from the beginning
        partial_matches = []
        flavour_lower = flavour.lower()
        
        for existing_flavour in flavours.keys():
            if existing_flavour.lower().startswith(flavour_lower):
                partial_matches.append(existing_flavour)
        
        # Ensure unique match
        if len(partial_matches) == 1:
            return partial_matches[0]
        elif len(partial_matches) > 1:
            log(WARNING, f"Ambiguous flavour '{flavour}' matches multiple options: {partial_matches}")
        
        return None
    
    def get_active_flavour(self) -> Tuple[Optional[str], Optional[str]]:
        """Get the active flavour and its configuration."""
        if self._active_flavour is None:
            return None, None

        flavour_config = None
        if self._llm_configuration and "flavours" in self._llm_configuration:
            flavour_config = self._llm_configuration["flavours"].get(self._active_flavour)
            
        return self._active_flavour, flavour_config
    
    def set_active_reasoning(self, reasoning_str: Optional[str]) -> Tuple[bool, Optional[AgentReasoningEnum]]:
        """Update active reasoning settings."""
        log(DEBUG, f"update_reasoning_details(): current_reasoning={self._active_reasoning}, reasoning_str='{reasoning_str}'")
        if reasoning_str is None:
            if self._active_reasoning != AgentReasoningEnum.SIMPLE:
                if get_user_service().get_session_id() != GLOBAL_SESSION_ID:
                    log(INFO, "No reasoning provided, using default (setting reasoning to AgentReasoningEnum.SIMPLE).")
            reasoning_str = AgentReasoningEnum.SIMPLE.name
        
        try:
            reasoning_enum = AgentReasoningEnum(reasoning_str.upper())
            log(DEBUG, f"Converted reasoning string '{reasoning_str}' to enum: {reasoning_enum}")
            if self._active_reasoning != reasoning_enum:
                log(INFO if get_user_service().get_session_id() != GLOBAL_SESSION_ID else DEBUG, f"Updating reasoning details: {reasoning_enum}")
                self._active_reasoning = reasoning_enum
            return True, reasoning_enum
            
        except ValueError:
            stack = traceback.format_exc()
            log(ERROR, f"'{reasoning_str}' cannot be converted to AgentReasoning.\nStack trace:\n{stack}")
            return False, None
        
    def get_active_reasoning(self) -> Optional[AgentReasoningEnum]:
        """Get the active reasoning level."""
        return self._active_reasoning
    
    def update_rag_settings(self, rag_settings_input: Union[dict, AgentRagSettings]) -> bool:
        """Update active RAG settings."""
        try:
            if isinstance(rag_settings_input, AgentRagSettings):
                rag_settings = rag_settings_input
            else:
                rag_settings = AgentRagSettings.from_dict(rag_settings_input)
            self._active_rag_settings = rag_settings
            log(DEBUG, f"RAG settings updated to: {self._active_rag_settings}")
            return True
            
        except Exception as e:
            stack = traceback.format_exc()
            log(ERROR, f"'{rag_settings_input}' cannot be converted to AgentRagSettings.\nStack trace:\n{stack}")
            return False

    def get_active_rag_settings(self) -> AgentRagSettings:
        """Get the active RAG settings."""
        return self._active_rag_settings

    def is_reasoning_enabled(self, reasoning_threshold: AgentReasoningEnum) -> bool:
        """Check if reasoning is enabled for the given threshold."""
        current_reasoning = self.get_active_reasoning()
        if current_reasoning is None:
            return False
        
        return reasoning_levels[current_reasoning] >= reasoning_levels[reasoning_threshold]

    def is_reasoning_enabled_for_agent(self, agent: 'Agent') -> bool:
        """Check if reasoning is enabled for a specific agent."""
        if agent is None:
            return True
            
        active_reasoning = self.get_active_reasoning()
        if active_reasoning is None:
            return False
        
        minimum_reasoning_level = reasoning_levels[agent.agent_minimum_reasoning]
        current_reasoning_level = reasoning_levels[active_reasoning]
        
        return current_reasoning_level >= minimum_reasoning_level

    def chat_completions(
        self,
        messages: List[Dict],
        agent: Optional['Agent'] = None,
        purpose: Optional[str] = None,
        plan: Optional['Plan'] = None,
        goal: Optional['Goal'] = None,
        parameters: Optional[Dict] = None,
        tools: Optional[list] = None,
        model_input: Optional[str] = None,
        model_provider: Optional[str] = None,
        max_completion_tokens: Optional[int] = MAX_COMPLETION_TOKENS_UNLIMITED,
        temperature: Optional[float] = None,
        response_format: Optional[Dict] = None,
        structured_output: Optional[object] = None,
        response_format_not_critical_default_value: Optional[object] = None,
        maximum_reasoning_required: Optional[AgentReasoningEnum] = None,
        cache: bool = USE_LLM_CACHE) -> LlmResponse:
        """Main chat completions entry point."""
        
        # NEW: If no explicit reasoning provided, check agent configuration
        if maximum_reasoning_required is None and agent is not None:
            maximum_reasoning_required = self.get_agent_reasoning(agent.agent_name, purpose)
        
        # Get model configuration
        model_supports_temperature:bool = True
        model_id, model_provider, llm_interface = self.get_model(agent, purpose, model_input=model_input, model_provider=model_provider)

        # Determine temperature
        if temperature is None:
            llm_configuration = self.get_llm_model_configuration_by_model_name(model_id, model_provider)
            if llm_configuration is None:
                log(ERROR, f"Cannot find this model: {model_id}, provider: {model_provider}")
                return LlmResponse(None, "")

            if "temperature" in llm_configuration:
                configured_temperature = llm_configuration["temperature"]
                if configured_temperature < 0.0:
                    # some models (e.g. gpt-5) do not accept temperature any longer, therefore disable it
                    model_supports_temperature = False
                    temperature = 1.0

            if model_supports_temperature:
                if agent is not None and agent.agent_force_model_temperature is not None and temperature == None:
                    temperature = agent.agent_force_model_temperature
                elif "temperature" in llm_configuration:
                    temperature = llm_configuration["temperature"]
                else:
                    temperature = DEFAULT_TEMPERATURE

        llm_response = llm_interface.chat_completions(
            messages=messages,
            agent=agent,
            plan=plan,
            goal=goal,
            purpose=purpose,
            parameters=parameters,
            tools=tools,
            model=model_id,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            response_format=response_format,
            structured_output=structured_output,
            response_format_not_critical_default_value=response_format_not_critical_default_value,
            maximum_reasoning_required=maximum_reasoning_required,
            cache=cache
        ) if llm_interface is not None else LlmResponse(None, None)

        return llm_response
    

log_end_file(__file__)

if __name__ == "__main__":
    from src.utils.log import log_init, log_exit
    log_init()

    try:
        llm_manager = LlmManager.instance()
        default_model = llm_manager.get_default_model()
        print(f"Default model: {default_model}")

        config_for_model = llm_manager.get_llm_model_configuration_by_model_name("gpt-4.1-mini", "openai")
        print(f"Configuration for gpt-4.1-mini: {config_for_model}")

        # quick test LLM
        test_messages = [
            { "role": "system", "content": "You are a helpful assistant." },
            { "role": "user", "content": "Say hello world in a concise manner." }
        ]
        log(LLM_IN, f"Test LLM call: messages={test_messages}")

        response_format={"response": "Concise assistant reply to the prompt."}
        response_schema = {
            "name": "TestResponse",
            "schema": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "Concise assistant reply to the prompt"
                    }
                },
                "required": ["response"],
                "additionalProperties": False
            },
            "strict": True
        }

        response = LlmManager.instance().chat_completions(
            test_messages,
            maximum_reasoning_required=AgentReasoningEnum.MINIMAL,
            response_format=response_format,
            structured_output=response_schema,
            
        )
        print(f"LLM response: {response}")

    except Exception as e:
        log(ERROR, f"Error in LLM Manager test: {e}")

    finally:
        log_exit()
