from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import sys
import os
import json
import importlib
import inspect
import asyncio
from typing import TYPE_CHECKING, List, Callable, Dict, Any, Union, Tuple, Optional

from src.config.constants import BASE_PATH, ENABLE_STRICT_TOOLS
from src.core.base.enums import AgentReasoningEnum
from src.core.extensions.mcp_client.mcp_client_global import McpClient, asyncio_main_loop
from src.utils.log import log, INFO, DEBUG, WARNING, ERROR, FATAL, print_lock


if TYPE_CHECKING:
    from src.core.thalaplan.Plan import Plan
    from src.core.thalaplan.goal.Goal import Goal


#region skills registration
# Global variables
agent_all_available_tools = None

# Import management
def import_class(module_path: str, class_name: str):
    mod = __import__(module_path, fromlist=[''])
    # for comp in class_name:
    mod = getattr(mod, class_name)
    return mod

def import_function(module_path: str, function_name: str):
    mod = __import__(module_path, fromlist=[''])
    func = getattr(mod, function_name)
    return func

def addSkillsToPath():
    # Modify the PYTHONPATH to include all Skills from the Skillsets subfolders
    for root, dirs, files in os.walk('{base_path}/src/core/extensions/skillsets'):
        for name in dirs:
            if "_" not in name:
                composite_path = os.path.join(root, name)
                log(DEBUG, f"Adding '{composite_path}' to the PYTHONPATH")
                sys.path.insert(0, composite_path)
    log(INFO, f"PYTHONPATH: {sys.path}")

# Load the skill configurations from the Skillsets folder
def load_skills_configs():
    skills_configs = {}
    for skillset in os.listdir(f'{BASE_PATH}/src/core/extensions/skillsets'):
        if os.path.isdir(f'{BASE_PATH}/src/core/extensions/skillsets/{skillset}'):
            skills_configs[skillset] = {}
            for skill in os.listdir(f'{BASE_PATH}/src/core/extensions/skillsets/{skillset}'):
                if skill.endswith('.skill'):
                    try:
                        log(DEBUG, f"Loading skill file {skill}")
                        with open(f'{BASE_PATH}/src/core/extensions/skillsets/{skillset}/{skill}', 'r', encoding='utf-8-sig') as file:
                            skills_configs[skillset][os.path.splitext(skill)[0]] = json.load(file)
                    except Exception as e:
                        log(ERROR, f"ERROR loading skill file {skill}: {e}, path= '{BASE_PATH}/src/core/extensions/skillsets/{skillset}/{skill}'")
    return skills_configs

def get_all_skills_as_tools() -> List[dict]:
    from src.core.base.OpenaiUtilities import get_function_schema
    
    global asyncio_main_loop, McpClient
    # Register all the skills
    tools = []
    mcp_client = McpClient[0]
    if mcp_client is None:
        raise RuntimeError("McpClient is not initialized. Please start the MCP client thread first.")
    mcp_tools = mcp_client.tools if mcp_client is not None else []
    for t in mcp_tools:
        try:
            schema = get_function_schema(t.function, name=t.name, description=t.description)
            schema["origin"] = "mcp"  # Add origin to the schema
            schema["function"]["parameters"] = t.function_schema
            tools.append({
                "function": t.function,
                "schema":  schema,
                "whitelist_if": t.whitelist_if,
                "blacklist_if": t.blacklist_if,
                "llm_label": None,
                "origin": "mcp"
            })
        except Exception as e:
            log(ERROR, f"ERROR: get_all_skills_as_tools() mcp tool '{t.name}': {e}")

    skills_configs = load_skills_configs()
    for skillset, skills in skills_configs.items():
        for skillid, skill_config in skills.items():
            try:
                # Dynamical import statement equivalent to: "from Skillsets.{skillset} import {skillid} as skill"
                sys.path.insert(0, f'{BASE_PATH}/src/core/extensions/skillsets/{skillset}')
                loaded_skill_module = importlib.import_module(f"src.core.extensions.skillsets.{skillset}.{skillid}")
                skill_function = getattr(loaded_skill_module, skillid)
                skill_schema = get_function_schema(
                    skill_function,
                    name=skillid,
                    description=skill_config['Description'],
                    excluded_parameters={"agent", "goal", "plan", "effective_user_id"},
                )
                skill_schema["origin"] = "thalabus_skill"  # Add origin to the schema
                # Register the skill
                log(DEBUG, f"Getting information from skill '{skillid}'")
                tools.append({
                    "function": skill_function,
                    "schema": skill_schema,
                    "whitelist_if": skill_config['WhiteList_IF'] if 'WhiteList_IF' in skill_config else None,
                    "blacklist_if": skill_config['BlackList_IF'] if 'BlackList_IF' in skill_config else None,
                    "llm_label": skill_config['LLM_Label'] if 'LLM_Label' in skill_config else None,
                    "origin": "thalabus_skill"
                })
            except Exception as e:
                log(ERROR, f"ERROR: get_all_skills_as_tools() skill '{skillid}': {e}")

    # returns number of loaded skills
    return tools

def is_async_function(func):
    return inspect.iscoroutinefunction(func)

# This function extracts the annotated arguments of the imported function (=skill) and calls it with the arguments from the input dictionary
def skill_wrapper(func: Callable[..., Any], args_dict: Dict[str, Any]) -> Callable[..., Any]:
    global asyncio_main_loop
    """
    This function takes a function and a dictionary of arguments, and returns a new function that 
    when called, calls the original function with the arguments from the dictionary.
    """
    sig = inspect.signature(func)

    def wrapped_func():
        # Bind the arguments to the function signature
        skillid = func.__name__
        # If the function accepts **kwargs, pass all non-special args through (not just named ones)
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_var_keyword:
            filtered_args = {
                k: v
                for k, v in args_dict.items()
                if k not in ["agent", "goal", "plan", "effective_user_id"]
            }
        else:
            filtered_args = {k: v for k, v in args_dict.items() if k in sig.parameters}
        bound_args = sig.bind_partial(**filtered_args)

        # Call the function with the bound arguments
        if is_async_function(func):
            try:
                # If the function is asynchronous, run it in the asyncio event loop
                if asyncio_main_loop[0] is None:
                    log(ERROR, "ERROR: asyncio_main_loop is not initialized. Cannot run the asynchronous function.")
                    return None
                if has_var_keyword:
                    args = {
                        k: v
                        for k, v in args_dict.items()
                        if k not in ["agent", "goal", "plan", "effective_user_id"]
                    }
                else:
                    args = {k: v for k, v in args_dict.items() if k in sig.parameters}
                if "kwargs" in sig.parameters and "kwargs" in args:
                    kwargs = args.pop("kwargs")
                    future = asyncio.run_coroutine_threadsafe(
                        func(**args, **kwargs), asyncio_main_loop[0]
                    )
                else:
                    future = asyncio.run_coroutine_threadsafe(
                        func(**args), asyncio_main_loop[0]
                    )
                response = future.result()
            except Exception as e:
                log(ERROR, f"ERROR: wrapped_func() async function '{skillid}': {e}")
                return None
        else:
            response = func(*bound_args.args, **bound_args.kwargs)

        return response

    wrapped_func.__name__ = func.__name__
    wrapped_func.__doc__ = f"Wrapper for the function {func.__name__}"

    return wrapped_func

def get_plan_whitelist_tools(goal: 'Goal') -> Optional[List[dict]]:
    """Return the subset of available tools that the instantiated plan's whitelist_tools restricts to,
    or None if the plan defines no tool whitelist."""
    global agent_all_available_tools
    if goal is None or agent_all_available_tools is None:
        return None
    plan_name = goal.goal_instantiated_plan
    if plan_name is None or not isinstance(plan_name, str):
        return None
    # Use sys.modules to avoid a circular import: manager.py imports this module
    manager_module = sys.modules.get('src.core.llm.manager')
    if manager_module is None:
        return None
    templates = manager_module.LlmManager.instance().get_agent_plans_template()
    plan_template = templates.get(plan_name)
    if plan_template is None:
        return None
    whitelist_tool_names: List[str] = plan_template.get("whitelist_tools")
    if not whitelist_tool_names:
        return None
    plan_tools = [
        tool for tool in agent_all_available_tools
        if tool["schema"]["function"]["name"] in whitelist_tool_names
    ]
    return plan_tools if plan_tools else None

def tools_init() -> None:
    global agent_all_available_tools

    if agent_all_available_tools is None:
        # Initialize the tools by loading all skills as tools
        agent_all_available_tools = get_all_skills_as_tools()
        log(INFO, f"Initialized {len(agent_all_available_tools)} tools from skills.")
    
def get_filtered_tools(plan:'Plan', goal:Optional['Goal']=None) -> Tuple[List[dict], List[dict], List[dict]]:
    global agent_all_available_tools

    filtered = []
    whitelist = []
    blacklist = []

    active_reasoning:AgentReasoningEnum = plan.plan_active_reasoning if plan is not None else AgentReasoningEnum.SIMPLE

    recommended_tools:Optional[List[str]] = get_mandatory_tools(goal.goal_recommended_tools) if goal is not None and goal.goal_recommended_tools is not None and len(goal.goal_recommended_tools) > 0 else None
    if goal is not None and goal.goal_override_recommended_tools is not None:
        recommended_tools = goal.goal_override_recommended_tools

    # TODO: remove old commented code
    # if agent_all_available_tools is None:
    #     agent_all_available_tools = get_all_skills_as_tools()
    if agent_all_available_tools is None:
        log(FATAL, "ERROR: get_filtered_tools() agent_all_available_tools is None. Please initialize the tools first by calling init_tools().")
        return [], [], []

    # if the instantiated plan defines a tool whitelist, restrict to those tools only
    plan_tools = get_plan_whitelist_tools(goal)
    if plan_tools is not None:
        return plan_tools, None, None

    for tool in agent_all_available_tools:
        # check if the tool has a "WhiteList_IF" or a "BlackList_IF" condition
        whitelist_if = tool["whitelist_if"] if "whitelist_if" in tool else None
        blacklist_if = tool["blacklist_if"] if "blacklist_if" in tool else None

        if plan is not None and whitelist_if is not None:
            # evaluate the condition by executing it. It is in the form:
            #   "(plan) -> { return plan.plan_domain == "coins"; }"
            # The plan is passed as an argument to the lambda function
            try:
                whitelisted = eval(whitelist_if, {"reasoning": active_reasoning.value}, {"plan": plan})
                if callable(whitelisted):
                    whitelisted = whitelisted(plan)
                    if whitelisted:
                        whitelist.append(tool)
            except Exception as e:
                log(ERROR, f"'ERROR: get_filtered_tools() has failed to evaluate the WhiteList_IF condition for the tool '{tool['schema']['function']['name']}': {e}")

        if recommended_tools is not None and "+"+tool["schema"]["function"]["name"] in recommended_tools:
            # the tool has been strongly recommended, such as "+DocumentEditor" in the Workflow recommendations. Therefore, it is whitelisted
            whitelist.append(tool)

        if plan is not None and blacklist_if is not None:
            # evaluate the condition by executing it. It is in the form:
            #   "(plan) -> { return plan.plan_domain == "coins"; }"
            # The plan is passed as an argument to the lambda function
            try:
                blacklisted = eval(blacklist_if, {"reasoning": active_reasoning.value}, {"plan": plan})
                if callable(blacklisted):
                    blacklisted = blacklisted(plan)
                    if blacklisted:
                        blacklist.append(tool)
                        continue
            except Exception as e:
                log(ERROR, f"'ERROR: get_filtered_tools has failed to evaluate the BlackList_IF condition for the tool '{tool['schema']['function']['name']}': {e}")

        filtered.append(tool)

    if len(whitelist) > 0:
        # if there are whitelisted tools, return them. All other tools are excluded
        return whitelist, whitelist, []
    
    # if there are no whitelisted tools, return the filtered list which excludes the blacklisted tools
    return filtered, whitelist, blacklist

def get_mandatory_tools(tools:Union[str, List[str]]) -> List[str]:
    if ENABLE_STRICT_TOOLS == True:
        # transforms the tools list in the format ["tool1", "tool2"] to ["+tool1", "+tool2"]
        if isinstance(tools, str):
            if len(tools) == 0 or tools.startswith("+"):
                return [tools]
            else:
                return ["+" + tools]
        elif isinstance(tools, list):
            result = []
            for tool in tools:
                if tool.startswith("+"):
                    # tool is already mandatory, so we add it to the result
                    result.append(tool)
                else:
                    # tool is not mandatory, so we make it mandatory by adding a "+" prefix
                    result.append("+" + tool)
            return result
        else:
            log(ERROR, f"ERROR: get_mandatory_tools() tools parameter is not a string or a list: {tools}")
            return None

    # Ensure we always return a List[str] even when ENABLE_STRICT_TOOLS is False
    if isinstance(tools, str):
        return [tools]
    elif isinstance(tools, list):
        return tools
    else:
        return None
#endregion


log_end_file(__file__)