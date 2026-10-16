from src.core.thalaplan.Plan import Plan
from src.core.thalaplan.goal.Goal import Goal
from src.core.llm import LlmManager
from src.utils.log import log, DEBUG, INFO, WARNING, ERROR
from typing import List, Optional


def get_goals_from_plan_template(plan: Plan, plan_name: str, tool_nr: int = None) -> Optional[List[Goal]]:
    """
    Get goal instances from a plan template by name.
    
    This is a standalone utility function that can be used by any agent that needs
    to instantiate plan templates without inheriting from AgentPlanner.
    
    Args:
        plan: The parent Plan object
        plan_name: The name of the plan template (e.g., "plan_mira")
        tool_nr: Optional number of times to duplicate the first goal (for multi-tool calls)
    
    Returns:
        List of Goal instances, or None if plan template not found
    """
    # import dynamically the instantiate_plan function from src.core.base.Plan.py
    module_name = 'Plan'
    function_name = 'instantiate_plan'
    module = __import__(f"src.core.thalaplan.{module_name}", fromlist=[function_name])
    instantiate_plan = getattr(module, function_name)

    # instantiate the plan_tool_template
    templates = LlmManager.instance().get_agent_plans_template()
    if plan_name not in templates:
        log(ERROR, f"ERROR: Agent has failed to find the plan '{plan_name}' in the templates. The available templates are: {list(templates.keys())}")
        return None
    
    plan_tool_template = templates[plan_name]["template"]
    goals = instantiate_plan(plan, plan_tool_template)
    
    if tool_nr is not None and tool_nr > 1:
        # duplicate the first goal tool_nr-1 times
        goal = goals[0]
        for i in range(1, tool_nr):
            cloned_goal = goal.clone()
            goals.insert(0, cloned_goal)
    
    return goals
