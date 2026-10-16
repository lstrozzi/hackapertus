import threading
import platform
import re
import time
import inspect
import copy
import json
import os
import sys
import traceback

from typing import Optional, List, Dict, Any
from datetime import datetime

# Note: elasticsearch is imported lazily in log_set_es_client() to defer startup overhead

from src.core.base.llm_response import LlmResponse
from src.utils.system import get_os_type

LOG_FILENAME = "thalabus.log"
LOG_PLAN_FILENAME = "plan.log"
LOG_ALL_TO_LOGFILE = False
PERFORMANCE_LOG_ACTIVE: bool = False
ENABLE_LOG_SCREEN_LOCK = True
MAX_DEBUG_MESSAGES = 12
MAX_ESCLIENT_RETRIES = 5

DEBUG    = 0
PERF     = 1
INFO     = 2
LLM_IN   = 3
LLM_OUT  = 4
FUNCTION = 5
PLAN     = 6
WARNING  = 7
ERROR    = 8
FATAL    = 9
CLEAR    = 10
Level_Strings = ["DEBUG", "PERF", "INFO", "LLM_IN", "LLM_OUT", "FUNCTION", "PLAN", "WARNING", "ERROR", "FATAL", "CLEAR"]

# Log globals
LogFile = None
LogESClient = None
ESClientLoggingMessage = False
ESClientDisabled = False
ESClientRetryCount = 0
LogLock = threading.Lock()
LogThread = None
LogShutdown = False
Debug_messages = []
LogSkillParameters = {"skillset": None, "skillid": None, "sessionid": None, "instance": None}
LogThreadEvent = threading.Event()
ExcludedLogLevels = []  # List of log levels to exclude from output
PerfTimestamps: dict = {}
PerfTimestampsLock = threading.Lock()
PerfSeqNo: int = 0
PerfSeqNoLock = threading.Lock()

# ANSI color codes for console output
RESET = "\033[0m"
GRAY = "\033[90m"
WHITE = "\033[97m"
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
LIGHT_GRAY = "\033[37m"
DARK_GRAY = "\033[90m"
LIGHT_RED = "\033[91m"
LIGHT_GREEN = "\033[92m"
LIGHT_YELLOW = "\033[93m"
LIGHT_BLUE = "\033[94m"
LIGHT_MAGENTA = "\033[95m"
LIGHT_CYAN = "\033[96m"
C_NEUTRAL = GRAY
C_DEBUG = GRAY
C_LLM_IN = MAGENTA
C_LLM_OUT = BLUE
C_FUNCTION = GREEN
C_INFO = WHITE
C_PLAN = BLACK
C_WARNING = YELLOW
C_ERROR = RED
C_FATAL = LIGHT_RED
C_PERF = CYAN
C_CLEAR = LIGHT_CYAN
C_LEVEL_CODES = [C_DEBUG, C_PERF, C_INFO, C_LLM_IN, C_LLM_OUT, C_FUNCTION, C_PLAN, C_WARNING, C_ERROR, C_FATAL, C_CLEAR]
LOG_THREAD_SLEEP_SECONDS = 5

# Multithreading locks to enable thread-safe execution
print_lock = threading.Lock()

# PERF report helpers – parse the log file and print a hierarchical summary
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
_PERF_ELAPSED_RE = re.compile(r"^\['([^']+)' \+ ([0-9.]+)s\] ")
_PERF_ID_RE = re.compile(r"^\[#(\d+)(?: '([^']+)')?\] ")
_PERF_CHILD_OF_RE = re.compile(r"^\[child_of:'([^']+)'\] ")

# LLM interaction dump files
_LlmInteractionSeqNo: int = 0
_LlmInteractionSeqNoLock = threading.Lock()
_LLM_INTERACTION_PREFIX: str = "log_llm_interaction_"
console_log_level: Optional[int] = None


def _print_to_console(text: str) -> None:
    """Write text to stdout line by line and flush to prevent truncation in terminals and debuggers."""
    for line in text.split('\n'):
        sys.stdout.write(line + '\n')
    sys.stdout.flush()

def log_thread():
    global LogFile
    while LogShutdown is False:
        # every LOG_THREAD_SLEEP_SECONDS seconds, flush the log file
        with LogLock:
            if LogFile:
                try:
                    LogFile.flush()
                except Exception as e:
                    # ignore flush errors (file may have been closed concurrently)
                    tb = traceback.format_exc()
                    try:
                        # best-effort logging of the flush failure
                        if LogFile:
                            LogFile.write(f"***flush error*** {e}\n{tb}\n")
                            LogFile.flush()
                    except Exception:
                        # if even writing to LogFile fails, fallback to printing
                        print(f"***flush error*** {e}\n{tb}")

        # wait but allow immediate wake-up via LogThreadEvent
        if not LogShutdown:
            try:
                LogThreadEvent.wait(LOG_THREAD_SLEEP_SECONDS)
            finally:
                # clear the event so it can be reused
                LogThreadEvent.clear()

    return

def log_set_es_client():
    """Set up the Elasticsearch client for logging."""
    global ESClientDisabled, LogESClient

    es_password = os.getenv("ELASTICSEARCH_PASSWORD")
    if not es_password or len(es_password) == 0: # if no password is provided, do not set up the ES client
        return None
    es_endpoint = os.getenv("ELASTICSEARCH_ENDPOINT", "http://localhost:9200")
    if not es_endpoint or len(es_endpoint) == 0: # if no endpoint is provided, do not set up the ES client
        return None
    
    # Lazy import Elasticsearch only when actually needed
    from elasticsearch import Elasticsearch
    
    es_client = Elasticsearch(es_endpoint, http_auth=('elastic', es_password))
    if es_client.ping() is False:
        ESClientDisabled = True
        LogESClient = None
        print("WARNING: Elasticsearch logging is disabled due to repeated errors. Please check your Elasticsearch configuration.")
        return None

    return es_client

def _get_llm_interaction_log_dir() -> str:
    """Return the directory where LLM interaction dump files are written."""
    os_type = get_os_type()
    return "/app/logs" if os_type == "Linux" else "./logs"

def log_llm_interaction(messages: list, response: 'LlmResponse', goal=None, agent=None) -> None:
    """Write one LLM interaction (messages + response) to a dedicated numbered file.

    Files are named log_llm_interaction_XXXX[_AgentName][_GoalId].txt.
    The sequence number is generated atomically so concurrent threads never collide.
    Each file is written fresh — never appended to.

    Args:
        messages: The LLM_IN messages array sent to the model.
        response: The LlmResponse object returned by the model.
        goal: Optional Goal object — used to extract goal_id and agent name fallback.
        agent: Optional Agent object — used to extract agent_name.
    """
    global _LlmInteractionSeqNo
    with _LlmInteractionSeqNoLock:
        _LlmInteractionSeqNo += 1
        seq = _LlmInteractionSeqNo

    # Resolve agent name: explicit agent arg > goal.goal_target_agent.agent_name > goal.goal_target_agent_name
    agent_name = ""
    if agent is not None and hasattr(agent, 'agent_name'):
        agent_name = agent.agent_name or ""
    elif goal is not None:
        if hasattr(goal, 'goal_target_agent') and goal.goal_target_agent is not None and hasattr(goal.goal_target_agent, 'agent_name'):
            agent_name = goal.goal_target_agent.agent_name or ""
        elif hasattr(goal, 'goal_target_agent_name') and goal.goal_target_agent_name:
            agent_name = goal.goal_target_agent_name

    # Resolve goal id
    goal_id = ""
    if goal is not None and hasattr(goal, 'get_id_chain_as_str'):
        try:
            goal_id = goal.get_id_chain_as_str() or ""
        except Exception:
            pass

    # Build filename suffix from available metadata
    suffix_parts = [f"{seq:04d}"]
    if agent_name:
        suffix_parts.append(agent_name)
    if goal_id:
        suffix_parts.append(goal_id)
    filename_base = _LLM_INTERACTION_PREFIX + "_".join(suffix_parts) + ".txt"

    log_dir = _get_llm_interaction_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, filename_base)

    try:
        # Format LLM_IN
        messages_clean = remove_images(copy.deepcopy(messages)) if messages else []
        messages_str = json.dumps(process_newlines(messages_clean), indent=4, separators=(',', ': '), ensure_ascii=False)

        # Format LLM_OUT
        reasoning_from_response = ""
        thinking = ""
        tokens_line = ""
        elapsed_line = ""
        finish_reason = ""
        llm_parameters = None
        if response is None:
            response_str = "(no response)"
        elif hasattr(response, 'response'):
            r = response.response
            if isinstance(r, dict):
                r = dict(r)
                r.pop("__reasoning__", None)
                if "role" in r:
                    # Raw API message dict — extract only the content field
                    reasoning_from_response = r.get("reasoning") or ""
                    content = r.get("content")
                    response_str = json.dumps(content, indent=4, separators=(',', ': '), ensure_ascii=False) if content is not None else "(null)"
                else:
                    response_str = json.dumps(r, indent=4, separators=(',', ': '), ensure_ascii=False)
            else:
                response_str = str(r) if r is not None else "(no response)"
            thinking = getattr(response, 'thinking_message', None) or reasoning_from_response
            model = getattr(response, 'model', None) or ""
            total_tokens = getattr(response, 'total_tokens', None)
            prompt_tokens = getattr(response, 'prompt_tokens', None)
            completion_tokens = getattr(response, 'completion_tokens', None)
            elapsed_msec = getattr(response, 'elapsed_time_msec', None)
            finish_reason = getattr(response, 'finish_reason', None) or ""
            llm_parameters = getattr(response, 'parameters', None)
            tokens_line = f"Model: {model}, Total: {total_tokens}, Prompt: {prompt_tokens}, Completion: {completion_tokens}" if total_tokens else "(no token info)"
            elapsed_line = f"Elapsed: {round(elapsed_msec / 1000, 1)}s" if elapsed_msec else ""
        else:
            response_str = str(response)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"=== LLM INTERACTION #{seq:04d} -- {timestamp} ===")
            if agent_name:
                f.write(f"  Agent: {agent_name}")
            if goal_id:
                f.write(f"  Goal: {goal_id}")
            f.write("\n\n")
            f.write("--- LLM_OUT ---\n")
            f.write(response_str)
            if thinking:
                f.write(f"\n\n--- REASONING ---\n{thinking}")
            f.write("\n\n--- LLM_IN ---\n")
            f.write(messages_str)
            if tokens_line:
                f.write(f"\n\n--- TOKENS ---\n{tokens_line}")
            if elapsed_line:
                f.write(f"  |  {elapsed_line}")
            if finish_reason:
                f.write(f"\n\n--- FINISH_REASON ---\n{finish_reason}")
            if llm_parameters is not None:
                params_dict = llm_parameters.to_dict()
                if params_dict:
                    params_str = ", ".join(f"{k}: {v}" for k, v in params_dict.items())
                    f.write(f"\n\n--- PARAMETERS ---\n{params_str}")
            f.write("\n")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"log_llm_interaction: failed to write {filename}: {e}\n{tb}")

def log_init(log_filename:str=LOG_FILENAME, disable_screen_lock:bool=False, limit_console_log_level:int=None, performance_log_active:Optional[bool]=None):
    global LogFile, LogThread, console_log_level, LogESClient, _LlmInteractionSeqNo, PERFORMANCE_LOG_ACTIVE
    if LogFile:
        return

    if performance_log_active is not None:
        PERFORMANCE_LOG_ACTIVE = performance_log_active
    
    # Reset interaction counter and remove any leftover interaction dump files
    _LlmInteractionSeqNo = 0
    _llm_interaction_log_dir = _get_llm_interaction_log_dir()
    if os.path.isdir(_llm_interaction_log_dir):
        for _f in os.listdir(_llm_interaction_log_dir):
            if _f.startswith(_LLM_INTERACTION_PREFIX) and _f.endswith(".txt"):
                try:
                    os.remove(os.path.join(_llm_interaction_log_dir, _f))
                except Exception:
                    pass

    os_type = get_os_type()
    if os_type == "Linux":
        base_path = "/app/"
    else:
        base_path = "./"

    if disable_screen_lock:
        global ENABLE_LOG_SCREEN_LOCK
        ENABLE_LOG_SCREEN_LOCK = False
    
    if limit_console_log_level is not None:
        console_log_level = limit_console_log_level

    if LogFile:
        LogFile.close()
    LogFile = open(base_path + log_filename, "w", encoding="utf-8")

    if LogThread is None:
        # ensure the event is cleared before starting
        LogThreadEvent.clear()
        LogThread = threading.Thread(target=log_thread, daemon=True)
        LogThread.start()
    return

def push_console_log_level(level: int) -> None:
    """Push a new console log level, returning visibility to at least `level`.

    Use `pop_console_log_level()` to restore the previous level.
    """
    global console_log_level
    if not hasattr(push_console_log_level, '_stack'):
        push_console_log_level._stack = []
    # Save current level (may be None)
    push_console_log_level._stack.append(console_log_level)
    console_log_level = level

def pop_console_log_level() -> None:
    """Restore the previous console log level pushed by `push_console_log_level()`.

    If nothing was pushed, this sets console log level to None.
    """
    global console_log_level
    if not hasattr(push_console_log_level, '_stack') or len(push_console_log_level._stack) == 0:
        console_log_level = None
        return
    console_log_level = push_console_log_level._stack.pop()

def set_excluded_log_levels(levels: list) -> None:
    """Set which log levels should be excluded from output.
    
    Args:
        levels: List of log level integers to exclude (e.g., [LLM_IN, LLM_OUT])
    """
    global ExcludedLogLevels
    ExcludedLogLevels = levels if levels else []

def log_exit():
    from src.utils.log_fileload import log_print
    global LogFile, LogThread, LogShutdown

    with LogLock:
        LogShutdown = True
        log_print("Log thread is shutting down...")

        # wake the log thread immediately so it can exit without waiting
        LogThreadEvent.set()

        if LogThread:
            # stop the log thread
            LogThread.join()
            LogThread = None

        if LogFile:
            # close the log file
            LogFile.flush()
            LogFile.close()
            LogFile = None
    return

def log_set_skill(skillset:str, skillid:str, sessionId:str, instanceId:str) -> None:
    global LogSkillParameters
    LogSkillParameters["skillset"] = skillset
    LogSkillParameters["skillid"] = skillid
    LogSkillParameters["sessionid"] = sessionId
    LogSkillParameters["instance"] = instanceId
    return

def find_class_in_stack_frame(stack):
    for frame in stack:
        if "self" in frame.frame.f_locals:
            return frame.frame.f_locals["self"]
    return None

def find_goal_in_stack_frame(stack):
    for frame in stack:
        if "goal" in frame.frame.f_locals:
            return frame.frame.f_locals["goal"]
    c = find_class_in_stack_frame(stack)
    if c and c.__class__.__name__ == "Goal":
        return c
    return None

def process_newlines(obj):
    if isinstance(obj, dict):
        return {k: process_newlines(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [process_newlines(elem) for elem in obj]
    elif isinstance(obj, str):
        replaced = obj.replace('\\n', '\n')
        return replaced
    else:
        return obj

def remove_images(msg_list):
    if not isinstance(msg_list, list):
        return msg_list
    
    for i in range(len(msg_list)):
        msg = msg_list[i]
        if isinstance(msg, dict):
            # Check if the dictionary contains an "image_url" key
            if "image_url" in msg:
                msg["image_url"] = "..."
            # Recursively process nested dictionaries or lists
            for key, value in msg.items():
                if isinstance(value, (list, dict)):
                    msg[key] = remove_images(value)
        elif isinstance(msg, list):
            # Recursively process nested lists
            msg_list[i] = remove_images(msg)
    
    return msg_list

def log(level:int, msg:str|LlmResponse, perf_id:Optional[str]=None, perf_since_id:Optional[str]=None) -> None:
    global LogFile, Level_Strings, Debug_messages, console_log_level, LogESClient, ESClientLoggingMessage, ESClientDisabled, ESClientRetryCount, ExcludedLogLevels

    # Check if this log level is excluded
    if level in ExcludedLogLevels:
        return

    # PERF logs are suppressed unless explicitly enabled
    if level == PERF and not PERFORMANCE_LOG_ACTIVE:
        return

    # Get the calling frame
    frame = inspect.currentframe()
    stack = inspect.stack()

    # Option to set breakpoints here
    if level >= WARNING:
        pass
    if level >= ERROR:
        pass
    if level == LLM_OUT:
        pass

    # Get the calling function name
    os_type = platform.system()
    separator = "\\" if os_type == "Windows" else "/"
    calling_function_name = {msg.split("'")[1]} if level==FUNCTION and "'" in msg else ""
    calling_function = stack[1].function + f"({calling_function_name})"
    calling_module = stack[1].filename.split(separator)[-1]
    calling_class = find_class_in_stack_frame(stack)
    calling_class_name = calling_class.__class__.__name__ if calling_class else ""
    calling_goal = find_goal_in_stack_frame(stack)
    if calling_goal and hasattr(calling_goal, "get_id_chain_as_str"):
        calling_goal_id = str(calling_goal.get_id_chain_as_str())
    else:
        calling_goal_id = ""

    # PERF level: timestamp bookkeeping and elapsed computation
    elapsed_perf_str = ""
    perf_seq_str = ""
    if level == PERF:
        global PerfSeqNo
        with PerfSeqNoLock:
            PerfSeqNo += 1
            perf_seq_no = PerfSeqNo
        perf_key_prefix = calling_goal_id
        now_ts = time.perf_counter()
        # Capture caller frame IDs (skip log() frame at index 0) for automatic nesting detection
        current_frame_ids = frozenset(id(f.frame) for f in stack[1:])
        if perf_since_id is not None:
            since_key = f"{perf_key_prefix}:{perf_since_id}"
            with PerfTimestampsLock:
                since_entry = PerfTimestamps.get(since_key)
            if since_entry is not None:
                since_ts = since_entry['ts'] if isinstance(since_entry, dict) else since_entry
                elapsed_s = (now_ts - since_ts)
                elapsed_perf_str = f"['{perf_since_id}' + {elapsed_s:.3f}s] "
        if perf_id is not None:
            perf_key = f"{perf_key_prefix}:{perf_id}"
            # Detect parent: the registered perf_id whose frame_ids is the largest
            # proper subset of the current frame_ids (logged from an outer call frame)
            parent_pid = None
            best_depth = 0
            with PerfTimestampsLock:
                for k, v in PerfTimestamps.items():
                    if not isinstance(v, dict):
                        continue
                    fids = v.get('frame_ids')
                    if fids and fids < current_frame_ids and len(fids) > best_depth:
                        candidate = k.rsplit(':', 1)[-1]
                        if candidate != perf_id:
                            parent_pid = candidate
                            best_depth = len(fids)
                PerfTimestamps[perf_key] = {'ts': now_ts, 'frame_ids': current_frame_ids}
            child_str = f"[child_of:'{parent_pid}'] " if parent_pid else ""
            perf_seq_str = f"[#{perf_seq_no} '{perf_id}'] {child_str}"
        else:
            perf_seq_str = f"[#{perf_seq_no}] "

    # Append msg2 to msg if provided
    if isinstance(msg, LlmResponse):
        # special case for LLM_OUT, create a nice structured output
        if isinstance(msg.response, dict):
            msg.thinking_message = msg.response.pop("__reasoning__") if "__reasoning__" in msg.response else msg.thinking_message
            response_str = json.dumps(msg.response, indent=4, separators=(',', ': '), ensure_ascii=False)
        else:
            response_str = msg.response

        # Convert elapsed time from milliseconds to seconds and round to 1 decimal
        if msg.elapsed_time_msec is not None and msg.elapsed_time_msec > 0:
            elapsed_seconds = msg.elapsed_time_msec / 1000.0
            elapsed_seconds_rounded = round(elapsed_seconds, 1)
            # Calculate tokens per second using total tokens over elapsed seconds
            tokens_per_second = msg.completion_tokens / elapsed_seconds if elapsed_seconds > 0 else None
            tokens_per_second_rounded = round(tokens_per_second, 1) if tokens_per_second is not None else None
            stats = f", Elapsed: {elapsed_seconds_rounded} sec, T/sec: {tokens_per_second_rounded}"
        else:
            tokens_per_second = None
            stats = ""

        # Include reasoning level if available
        reasoning_level = msg.reasoning_level
        if reasoning_level is not None:
            stats = f", Reasoning: {reasoning_level.value}{stats}"

        # Include endpoint if available
        endpoint_str = f"Endpoint: {msg.endpoint}, " if msg.endpoint is not None else ""

        model_str = f"Model: {msg.model}, " if msg.model is not None else ""
        tokens_str = f"{model_str}Total Tokens: {msg.total_tokens}, Prompt: {msg.prompt_tokens}, Completion: {msg.completion_tokens}{stats}" if msg.total_tokens is not None else "Cache was used, no tokens counted"
        msg = f"""{response_str}\n<__reasoning__>
{msg.thinking_message}
</__reasoning__>
<__tokens__>
{endpoint_str}{tokens_str}
</__tokens__>"""
        pass

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Safe check for log level
    if not isinstance(level, int) or level < 0 or level >= len(C_LEVEL_CODES):
        # Log the invalid level as DEBUG, but do not recurse if already DEBUG
        safe_level = DEBUG
        msg_color = C_LEVEL_CODES[safe_level]
    else:
        msg_color = C_LEVEL_CODES[level]

    # format the message content
    msg_copy = None
    if isinstance(msg, dict) or isinstance(msg, list):
        # pretty print the dictionary or list, with indentation, spaces, and newlines
        msg_copy = copy.deepcopy(msg)
        msg_copy = process_newlines(msg_copy)
        msg_copy = remove_images(msg_copy)
        msg_formatted = json.dumps(msg_copy, indent=4, separators=(',', ': '), ensure_ascii=False)
        msg_formatted = msg_formatted.replace('\\n', '\n')
    elif level == PLAN:
        # pretty print the plan
        msg_formatted = msg.__str__()
    else:
        msg_formatted = msg

    # prepend elapsed and sequence number for PERF messages
    if perf_seq_str:
        msg_formatted = elapsed_perf_str + perf_seq_str + str(msg_formatted)

    # format the log message
    calling_module = calling_module[:20]
    calling_class_name = calling_class_name[:20]
    calling_goal_id = calling_goal_id[:15]
    calling_function = calling_function[:30]
    color_class = C_NEUTRAL if level not in [LLM_IN] else msg_color
    color_function = C_NEUTRAL if level not in [FUNCTION] else msg_color
    log_message_file = f"{C_NEUTRAL}{timestamp}|{calling_module:<20}|{color_class}{calling_class_name:<20}{C_NEUTRAL}|{calling_goal_id:<15}|{color_function}{calling_function:<30}{RESET}|{msg_color}{Level_Strings[level]:<8}{RESET}|{msg_color}{msg_formatted}{RESET}"
    if os_type == "Linux":
        if isinstance(msg, dict) or isinstance(msg, list):
            # no formatting with Blazor debug pages
            if msg_copy is not None:
                msg_console = json.dumps(msg_copy)
            else:
                msg_console = json.dumps(msg)
        elif msg_formatted is not None:
            # formatting without ANSI terminal colors when debugging on Linux
            msg_console = msg_formatted.replace('\n', '\\n')
        else:
            msg_console = msg

        log_message_console = f"{timestamp}|{calling_module:<20}|{calling_class_name:<20}|{calling_goal_id:<15}|{calling_function:<30}|{Level_Strings[level]:<8}|{msg_console}"
    else:
        # formatting for ANSI terminal colors when debugging on Windows or Macintosh
        log_message_console = log_message_file

    # print to console, which requires a lock to prevent interleaving among multiple threads
    # PERF messages are treated like INFO for console visibility: they show whenever INFO shows,
    # but are still suppressable if console_log_level is set above INFO (e.g., WARNING).
    effective_console_level = console_log_level if console_log_level is not None else INFO
    effective_msg_level = INFO if level == PERF else level
    if effective_msg_level >= effective_console_level:
        with print_lock:
            if ENABLE_LOG_SCREEN_LOCK and level == DEBUG:
                log_message_console = log_message_console.replace("\n", "  |||  ")
                if len(Debug_messages) < MAX_DEBUG_MESSAGES:
                    # Add the debug message to the list of debug messages
                    Debug_messages.append(log_message_console)
                    _print_to_console(log_message_console)
                else:
                    # Move the cursor up 10 rows + clear the screen from the cursor down + reprint the debug messages
                    Debug_messages.pop(0)
                    Debug_messages.append(log_message_console)
                    print(f"\033[{MAX_DEBUG_MESSAGES}A\033[J" + "\n".join(Debug_messages), flush=True)
            else:
                if os_type != "Linux":
                    if len(Debug_messages) > 0:
                        # Clear all debug messages and print the current message
                        print(f"\033[{len(Debug_messages)}A\033[J" + log_message_console, flush=True)
                        Debug_messages = []
                    else:
                        _print_to_console(log_message_console)
                else:
                    # in docker container, print the message without ANSI colors, and add $skillid to the beginning of the message, if the log_level is at least INFO
                    skillid = LogSkillParameters["skillid"]
                    if level < INFO or skillid is None:
                        _print_to_console(log_message_console)
                    else:
                        _print_to_console(f"${skillid} {log_message_console}")

    # Write and flush each log message so container tailers see it immediately.
    if level > DEBUG or LOG_ALL_TO_LOGFILE:
        with LogLock:
            if not LogFile:
                log_init()

            LogFile.write(f"{log_message_file}\n")
            try:
                LogFile.flush()
            except Exception as e:
                tb = traceback.format_exc()
                try:
                    if LogFile:
                        LogFile.write(f"***flush error*** {e}\n{tb}\n")
                        LogFile.flush()
                except Exception:
                    print(f"***flush error*** {e}\n{tb}")
    
    # write the log message to Elasticsearch, if > DEBUG
    if level > DEBUG:
        if os_type == "Linux" and not ESClientDisabled:
            # write the log message to Elasticsearch. Only if running in a Docker container
            if LogESClient is None:
                LogESClient = log_set_es_client()
            if LogESClient:
                log_entry = {
                    "timestamp": timestamp,
                    "module": calling_module,
                    "class": calling_class_name,
                    "goal_id": calling_goal_id,
                    "function": calling_function,
                    "level": Level_Strings[level],
                    "message": msg_formatted,
                    "session_id": LogSkillParameters["sessionid"],
                }
                try:
                    LogESClient.index(index="thalabus_logs", body=log_entry)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"*** Error writing log to Elasticsearch: {e}\n{tb}")
                    ESClientRetryCount += 1
                    if ESClientRetryCount > MAX_ESCLIENT_RETRIES:
                        ESClientDisabled = True
                        LogESClient = None
                        log(WARNING, "Elasticsearch logging is disabled due to repeated errors. Please check your Elasticsearch configuration.")
        else:
            if not ESClientLoggingMessage:
                ESClientLoggingMessage = True
                log(DEBUG, "Elasticsearch logging is only supported in Linux containers. Skipping Elasticsearch logging.")

    return

def check_dependencies(goal) -> bool:
    correct = True

    if goal.goal_parent is not None:
        # check the dependencies of this goal, they must be negative integers, not greater than the index of the goal with respect to the parent's chain
        goal_index = goal.goal_parent.goal_subgoals.index(goal)
        if goal.goal_dependent_upon is not None:
            for dep in goal.goal_dependent_upon:
                if dep >= 0:
                    log(WARNING, f"Goal {goal.get_id_chain_as_str()} has a non-negative dependency {dep}. This is incorrect.")
                    correct = False
                elif dep < -goal_index:
                    log(WARNING, f"Goal {goal.get_id_chain_as_str()} has a dependency {dep} that is greater than its index {goal_index}. This is incorrect.")
                    correct = False

    # recursively check subgoals
    if goal.goal_subgoals is not None:
        for subgoal in goal.goal_subgoals:
            subgoal_correct = check_dependencies(subgoal)
            if not subgoal_correct:
                correct = False

    return correct

def log_plan(plan, include_rag:bool=True) -> None:
    if plan is None:
        return
    
    try:
        # perform a sanity check on the dependencies
        if plan.plan_goal is not None:
            check_dependencies(plan.plan_goal)

        # update the plan log
        plan_str = plan.__str__()
        with open(LOG_PLAN_FILENAME, "w", encoding="utf-8") as f:
            f.write(f"Plan log: {datetime.now()}\n\n")
            f.write(plan_str)
            f.write("\n")
    except Exception as e:
        tb = traceback.format_exc()
        log(ERROR, f"Error writing plan log: {e}\n{tb}")

    return

def get_log_file_path() -> str:
    """Return the path to the active thalabus log file."""
    try:
        os_type = get_os_type()
        base_path = "/app/" if os_type == "Linux" else "./"
        return os.path.join(base_path, LOG_FILENAME)
    except Exception:
        return "./thalabus.log"

def log_file_offset(log_path: str) -> int:
    """Return current byte size of the log file (used as a start offset before a query)."""
    try:
        if os.path.exists(log_path):
            return os.path.getsize(log_path)
    except Exception:
        pass
    return 0

def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub('', text)

def parse_perf_lines(log_path: str, start_offset: int) -> List[Dict[str, Any]]:
    """Read log file from *start_offset* and return parsed PERF entries."""
    entries: List[Dict[str, Any]] = []
    try:
        with open(log_path, 'rb') as fh:
            fh.seek(start_offset)
            raw = fh.read().decode('utf-8', errors='replace')
    except Exception:
        return entries

    for line in raw.splitlines():
        clean = _strip_ansi(line)
        parts = clean.split('|')
        if len(parts) < 7:
            continue
        level_field = parts[5].strip()
        if level_field != 'PERF':
            continue

        timestamp = parts[0].strip()
        module = parts[1].strip()
        message = '|'.join(parts[6:]).strip()

        elapsed_since: Optional[float] = None
        since_id: Optional[str] = None
        seq_no: Optional[int] = None
        perf_id: Optional[str] = None

        m = _PERF_ELAPSED_RE.match(message)
        if m:
            since_id = m.group(1)
            elapsed_since = float(m.group(2))
            message = message[m.end():]

        m = _PERF_ID_RE.match(message)
        if m:
            seq_no = int(m.group(1))
            perf_id = m.group(2)
            message = message[m.end():]

        parent_id: Optional[str] = None
        m = _PERF_CHILD_OF_RE.match(message)
        if m:
            parent_id = m.group(1)
            message = message[m.end():]

        entries.append({
            'timestamp': timestamp,
            'module': module,
            'seq_no': seq_no,
            'perf_id': perf_id,
            'parent_id': parent_id,
            'since_id': since_id,
            'elapsed_since': elapsed_since,
            'message': message,
        })

    return entries

def print_perf_summary(log_path: str, start_offset: int) -> None:
    """Parse PERF entries from the log and print a hierarchical summary."""
    entries = parse_perf_lines(log_path, start_offset)
    if not entries:
        print('\n\033[93m[--perf] No PERF log entries found.\033[0m')
        return

    # Build tree of named entries using parent_id links detected by log()
    by_id: Dict[str, Any] = {}
    for e in entries:
        if e['perf_id']:
            e['_children'] = []
            by_id[e['perf_id']] = e

    roots: List[Any] = []
    for e in entries:
        if not e['perf_id']:
            continue
        pid = e.get('parent_id')
        if pid and pid in by_id:
            by_id[pid]['_children'].append(e)
        else:
            roots.append(e)

    # DFS to produce flat ordered list with depth for each named entry
    flat: List[Any] = []  # list of (depth, entry)

    def _dfs(node: Any, depth: int) -> None:
        flat.append((depth, node))
        for child in node['_children']:
            _dfs(child, depth + 1)

    for root in roots:
        _dfs(root, 0)

    # Insert unnamed entries (no perf_id) at the right depth by seq_no proximity
    unnamed = [e for e in entries if not e['perf_id']]
    for ue in unnamed:
        if ue['seq_no'] is None:
            flat.append((0, ue))
            continue
        insert_pos = 0
        for i, (d, e) in enumerate(flat):
            sn = e.get('seq_no')
            if sn is not None and sn < ue['seq_no']:
                insert_pos = i + 1
        depth = flat[insert_pos - 1][0] if insert_pos > 0 else 0
        flat.insert(insert_pos, (depth, ue))

    # Render
    ID_W = 30
    EL_W = 10
    print('\n\033[96m' + '─' * 90 + '\033[0m')
    print('\033[96m PERF SUMMARY\033[0m')
    print('\033[96m' + '─' * 90 + '\033[0m')
    print(f"  \033[90m{'Seq':>4}  {'Tree / ID':<{ID_W}}  {'Elapsed':>{EL_W}}  Message\033[0m")
    print(f"  \033[90m{'─'*4}  {'─'*ID_W}  {'─'*EL_W}  {'─'*30}\033[0m")

    for depth, e in flat:
        indent = '  ' * depth
        seq_str = f"{e['seq_no']:4d}" if e['seq_no'] is not None else '   ?'
        id_part = e['perf_id'] or ''
        id_display = (indent + id_part)[:ID_W]
        elapsed_str = f"{e['elapsed_since']:.3f}s" if e['elapsed_since'] is not None else ''
        msg = e['message']
        print(f"  {seq_str}  {id_display:<{ID_W}}  {elapsed_str:>{EL_W}}  {msg}")

    # Slowest named steps
    timed = [(e['perf_id'], e['elapsed_since']) for _, e in flat if e.get('perf_id') and e['elapsed_since'] is not None]
    if timed:
        timed_sorted = sorted(timed, key=lambda x: x[1], reverse=True)
        print()
        print('\033[96m Slowest steps:\033[0m')
        for pid, el in timed_sorted[:5]:
            bar = '█' * min(int(el / timed_sorted[0][1] * 20), 20)
            print(f"  {pid:<30}  {el:>8.3f}s  \033[93m{bar}\033[0m")

    print('\033[96m' + '─' * 90 + '\033[0m')


if __name__ == "__main__":
    log_init()
    log(INFO, "Start of normal logging test")
    log(DEBUG, "Debug message")
    log(LLM_IN, "LLM messages")
    log(LLM_OUT, "LLM responses")
    log(FUNCTION, "Function message")
    log(INFO, "Info message")
    log(WARNING, "Warning message")
    log(ERROR, "Error message")
    log(FATAL, "Fatal message")

    log(INFO, "Start of ANSI SCREEN LOCK feature test")
    for i in range(5):
        log(DEBUG, f"Message {i}")
        time.sleep(0.25)

    log(INFO, "Start of second ANSI SCREEN LOCK feature test\nThis message is very long to test the screen lock feature\nIt is so long that it will be split into multiple lines")

    for i in range(5):
        log(DEBUG, f"Message {i}")
        time.sleep(0.25)

    log(INFO, "Start of third ANSI SCREEN LOCK feature test")
    for i in range(20):
        log(DEBUG, f"Message {i}")
        time.sleep(0.25)
        if i==5:
            log(DEBUG, "VERY LONG DEBUGGING MESSAGE\nIN MULTIPLE LINES\nTO TEST THE SCREEN LOCK FEATURE")
            time.sleep(1)
    log(INFO, "End of ANSI SCREEN LOCK feature test")

    log_exit()
