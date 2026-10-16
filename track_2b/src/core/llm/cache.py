"""
Cache utilities for LLM operations.
"""

from typing import TYPE_CHECKING, List, Dict, Any, Tuple
from datetime import datetime

from src.utils.text_utils import hash_string
from src.services.cache_entry import get_cache_entry_service

if TYPE_CHECKING:
    from src.core.agents.Agent import Agent
    from src.core.thalaplan.goal.Goal import Goal
    from src.core.thalaplan.Plan import Plan


def save_value_into_cache(key: str, value: Any, thinking_message: str = None, hash: str = None, agent: 'Agent' = None, plan: 'Plan' = None, goal: 'Goal' = None, purpose: str = None, messages: List[Dict] = None, tools: list = None, response_format: Dict = None, structured_output: object = None) -> None:
    """
    Save a value to the cache with comprehensive metadata.
    """
    if hash is None:
        hash = hash_string(key)
        
    cache_service = get_cache_entry_service()
    cache_service.save_cache_entry(
        hash=hash,
        key=key,
        value=value,
        thinking_message=thinking_message,
        messages=messages,
        tools=tools,
        response_format=response_format,
        structured_output=structured_output,
        guid=plan.plan_guid if plan is not None else None,
        agent=agent.agent_name if agent is not None else None,
        category=agent.agent_cache_category if agent is not None else None,
        purpose=purpose,
        goal_id=goal.get_id_chain_as_str() if goal is not None else None,
        date=datetime.now()
    )


def get_value_from_cache(key: str) -> Tuple[Any, str, str, str]:
    """
    Retrieve a value from the cache by key.
    
    Returns:
        Tuple of (value, hash, guid, thinking_message) or (None, hash, None, None) if not found.
    """
    hash = hash_string(key)
    cache_service = get_cache_entry_service()
    return cache_service.get_cache_entry_by_hash(hash, key)

