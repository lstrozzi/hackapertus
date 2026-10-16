from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import json
from typing import List, Dict, Any, Tuple
from datetime import datetime

from src.utils.log import log, INFO, WARNING, ERROR
from src.core.base.InputOutput import execute_neo4j_query
from src.interfaces.cache_entry_interface import CacheEntryInterface


class CacheEntryService(CacheEntryInterface):
    """Neo4j implementation of cache entry repository operations."""

    def _is_cache_enabled(self) -> bool:
        """Check whether the cache is enabled via LLM configuration."""
        from src.core.llm.manager import LlmManager  # lazy import to avoid circular dependency
        cache_settings = LlmManager.instance().get_cache_settings()
        if cache_settings is None:
            return True
        return cache_settings.get("enable_cache", True)

    def save_cache_entry(self, hash: str, key: str, value: Any, thinking_message: str = "", messages: List[Dict] = None, 
                        tools: list = None, response_format: Dict = None, 
                        structured_output: object = None, guid: str = "", 
                        agent: str = None, category: str = None, purpose: str = None, 
                        goal_id: str = None, date: datetime = None) -> None:
        """Save a cache entry with comprehensive metadata."""

        if not self._is_cache_enabled():
            return

        if date is None:
            date = datetime.now()
            
        query = """MERGE (c:CacheEntry {hash: $hash})
                   ON CREATE SET
                       c.guid = $guid,
                       c.date_created = $date
                   SET c.key = $key,
                       c.messages = $messages,
                       c.tools = $tools,
                       c.response_format = $response_format,
                       c.structured_output = $structured_output,
                       c.response = $value,
                       c.thinking_message = $thinking_message,
                       c.guid_last_exec = $guid,
                       c.agent = $agent,
                       c.category = $category,
                       c.purpose = $purpose,
                       c.goal_id = $goal_id,
                       c.date_lastused = $date"""

        if value is None:
            # Delete the cache node
            value_to_save = None
            query = "MATCH (c:CacheEntry {hash: $hash}) DETACH DELETE c"
        elif isinstance(value, str):
            value_to_save = "'" + value + "'"
        elif isinstance(value, dict):
            value_to_save = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, list):
            value_to_save = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            value_to_save = str(value).lower()
        elif isinstance(value, int):
            value_to_save = str(value)
        elif isinstance(value, float):
            value_to_save = str(value)
        else:
            log(ERROR, f"ERROR: save_cache_entry(): unsupported type: {type(value)}")
            return

        # duplicate the messages and add instruction needed by training data (model fine-tuning)
        if messages is not None:
            messages_copy = messages.copy()
            for message in messages_copy:
                # add the instruction for the training data, to ignore all messages that are not needed for the model fine-tuning
                # this can be done by setting the parameter "weight" to 0
                if message["role"] == "assistant" and "tool_calls" in message:
                    message["weight"] = 0
                elif message["role"] == "assistant" and "content" in message and "FUNCTION CALL" in message["content"]:
                    message["weight"] = 0
        else:
            messages_copy = None

        parameters = {
            "hash": hash,
            "key": key,
            "messages": json.dumps(messages_copy, ensure_ascii=False) if messages_copy is not None else None,
            "tools": json.dumps(tools, ensure_ascii=False) if tools is not None else None,
            "response_format": json.dumps(response_format, ensure_ascii=False) if response_format is not None else None,
            "structured_output": json.dumps(structured_output, ensure_ascii=False) if structured_output is not None else None,
            "value": value_to_save,
            "thinking_message": thinking_message if thinking_message is not None else "",
            "guid": guid if guid is not None else "",
            "agent": agent,
            "category": category,
            "purpose": purpose,
            "goal_id": goal_id,
            "date": date
        }
        result = execute_neo4j_query(query, parameters)
        if not result.success:
            log(WARNING, "save_cache_entry(): Failed to save cache entry.")

    def get_cache_entry_by_hash(self, hash: str, key: str = None) -> Tuple[Any, str, str, str]:
        """Retrieve a cache entry by hash."""
        if not self._is_cache_enabled():
            return None, hash, None, None

        query = """
            MATCH (c:CacheEntry {hash: $hash})
            RETURN 
            c.key AS key, 
            c.response AS value, 
            c.guid AS guid, 
            c.thinking_message AS thinking_message
        """
        parameters = {"hash": hash}
        result_list = execute_neo4j_query(query, parameters)

        if result_list and len(result_list) == 1:
            # unique cache entry found
            result = result_list[0]
        elif result_list and len(result_list) > 1:
            # search for a cache entry with the exact key
            if key is not None:
                matching_results = [x for x in result_list if x["key"] == key]
                result = matching_results[0] if matching_results else None
            else:
                result = result_list[0]
        else:
            result = None
            
        if result is not None and len(result) == 4:
            value = result[1]
            guid = result[2]
            thinking_message = result[3]
            if thinking_message is not None and len(thinking_message) == 0:
                thinking_message = None
            if guid is not None and len(guid) == 0:
                guid = None

            if value is None:
                return None, hash, guid, thinking_message
            if value.startswith("'{") and value.endswith("}'"):
                try:
                    json_result = json.loads(value[1:-1])
                    guid_result = guid
                except json.JSONDecodeError as e:
                    log(ERROR, f"get_cache_entry_by_hash(): JSONDecodeError: {e}. The value is:\n{value}")
                    json_result = None
                    guid_result = None
                return json_result, hash, guid_result, thinking_message
            if value.startswith("'") and value.endswith("'"):
                return value[1:-1], hash, guid, thinking_message
            if value.startswith("[") and value.endswith("]"):
                return json.loads(value), hash, guid, thinking_message
            if value.startswith("{") and value.endswith("}"):
                return json.loads(value), hash, guid, thinking_message
            if value.lower() == "true":
                return True, hash, guid, thinking_message
            if value.lower() == "false":
                return False, hash, guid, thinking_message
            if value.isdigit():
                return int(value), hash, guid, thinking_message
            if "." in value:
                try:
                    return float(value), hash, guid, thinking_message
                except ValueError:
                    return value, hash, guid, thinking_message
            return value, hash, guid, thinking_message
        else:
            return None, hash, None, None

    def update_cache_entry_usage(self, hash: str, guid: str, date: datetime) -> None:
        """Update cache entry usage metadata."""
        if not self._is_cache_enabled():
            return

        if date is None:
            date = datetime.now()
            
        query = """MATCH (c:CacheEntry {hash: $hash}) 
                   SET c.guid_last_exec = $guid, 
                       c.date_lastused = $date"""
        parameters = {
            "hash": hash,
            "guid": guid,
            "date": date
        }
        execute_neo4j_query(query, parameters)

    def delete_cache_entry(self, hash: str = None, guid: str = None) -> None:
        """Delete cache entry(ies)."""
        if hash is not None:
            query = "MATCH (c:CacheEntry {hash: $hash}) DETACH DELETE c"
            parameters = {"hash": hash}
        elif guid is not None:
            query = "MATCH (c:CacheEntry) WHERE c.guid_last_exec = $guid DETACH DELETE c"
            parameters = {"guid": guid}
        else:
            log(ERROR, "delete_cache_entry(): Either hash or guid must be provided")
            return
            
        execute_neo4j_query(query, parameters)

    def mark_cache_entries_as_sample(self, guid: str, is_good: bool, auto_generated: bool = False) -> None:
        """Mark cache entries as training samples."""
        query = "MATCH (c:CacheEntry) WHERE c.guid = $guid SET c.good = $good, c.auto_generated = $auto_generated"
        parameters = {
            "guid": guid,
            "good": is_good, 
            "auto_generated": auto_generated
        }
        execute_neo4j_query(query, parameters)

    def mark_plan_cache_entries_as_sample(self, guid: str, is_good: bool, auto_generated: bool = False) -> bool:
        """Mark cache entries created or executed by a plan as training samples."""
        query = "MATCH (c:CacheEntry) WHERE c.guid = $guid OR c.guid_last_exec = $guid SET c.good = $good, c.auto_generated = $auto_generated, c.feedback_guid = $guid"
        parameters = {
            "guid": guid,
            "good": is_good,
            "auto_generated": auto_generated
        }
        result = execute_neo4j_query(query, parameters)
        success = bool(result and result.success)
        return success

    def create_feedback_message(self, guid: str, is_good: bool, message: str, plan: str, messages: str) -> bool:
        """Persist a plan feedback message and its captured conversation context."""
        query = """CREATE (f:FeedbackMessage)
            SET f.guid = $guid,
                f.good = $good,
                f.message = $message,
                f.plan = $plan,
                f.messages = $messages,
                f.timestamp = datetime()"""
        parameters = {
            "guid": guid,
            "good": is_good,
            "message": message,
            "plan": plan,
            "messages": messages
        }
        result = execute_neo4j_query(query, parameters)
        success = bool(result and result.success)
        return success

    def delete_plan_cache_entries(self, guid: str) -> None:
        """Delete cache entries created or executed by a plan."""
        query = "MATCH (c:CacheEntry) WHERE c.guid = $guid OR c.guid_last_exec = $guid DETACH DELETE c"
        execute_neo4j_query(query, {"guid": guid})

    def get_training_data_cache_entries(self, agent: str = None, purpose: str = None,
                                      is_good: bool = True, category: str = None) -> List[Dict[str, Any]]:
        """Retrieve cache entries for training data export."""
        # Build query based on filters
        where_conditions = ["c.good = $good"]
        parameters = {"good": is_good}
        
        if agent is not None:
            where_conditions.append("c.agent = $agent")
            parameters["agent"] = agent
            
        if purpose is not None:
            where_conditions.append("c.purpose = $purpose")
            parameters["purpose"] = purpose

        if category is not None:
            where_conditions.append("c.category = $category")
            parameters["category"] = category
            
        where_clause = " AND ".join(where_conditions)
        
        query = f"""MATCH (c:CacheEntry) 
                    WHERE {where_clause} 
                    RETURN c.key AS key, 
                           c.messages AS messages, 
                           c.tools AS tools, 
                           c.response_format AS response_format, 
                           c.structured_output AS structured_output, 
                           c.response AS response, 
                           c.agent AS agent, 
                           c.purpose AS purpose, 
                           c.guid AS guid,
                           c.goal_id AS goal_id,
                           c.date_created AS date_created,
                           c.auto_generated AS auto_generated"""
        
        result_list = execute_neo4j_query(query, parameters)
        return result_list.results if result_list else []

    def get_distinct_agents_and_purposes(self, agent: str = None) -> List[Tuple[str, str]]:
        """Get distinct combinations of agents and purposes from cache entries."""
        query = "MATCH (c:CacheEntry) WHERE $agent IS NULL OR c.agent = $agent RETURN DISTINCT c.agent, c.purpose"
        query_response = execute_neo4j_query(query, {"agent": agent})
        
        if not query_response:
            return []
            
        agents_purposes = [(record["c.agent"], record["c.purpose"]) for record in query_response]
        return agents_purposes


# Global service instance
_cache_entry_service = None

def get_cache_entry_service() -> CacheEntryService:
    """Get the global cache entry service instance."""
    global _cache_entry_service
    if _cache_entry_service is None:
        _cache_entry_service = CacheEntryService()
    return _cache_entry_service

log_end_file(__file__)