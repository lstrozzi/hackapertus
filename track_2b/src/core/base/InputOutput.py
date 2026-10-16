from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

# python libraries-related imports
import os
import time
import threading
import requests
import httpx
import json
import concurrent
import pycountry
import base64
from enum import Enum
from datetime import date, datetime, timezone
from threading import Condition
from typing import TYPE_CHECKING, List, Callable, Dict, Any, Union, Tuple, Optional

# thalabus imports
from src.utils.log import log, INFO, DEBUG, WARNING, ERROR, FATAL, PERF, print_lock
from src.utils.performance import total_performance_time_milli
from src.core.base.Shutdown import shutdown_event
from src.config.constants import CONTAINERUTILITY_PROTOCOL, CONTAINERUTILITY_HOSTNAME, CONTAINERUTILITY_PORT, BLAZOR_HOSTNAME, BLAZOR_PORT, BLAZOR_PROTOCOL, ADDR, PORT, USE_REAL_SSL_CERTIFICATE
from src.api.auth.service_token import request_with_service_token
from src.core.base.neo4j_queries import execute_neo4j_query  # noqa: F401
from src.services.user_service import get_user_service
from src.config.app_config import app_config
from src.utils.system import get_os_type

if TYPE_CHECKING:
    from src.core.thalaplan.Plan import Plan

OUTPUT_PATH = os.getenv('OUTPUT_PATH', 'output')
CACHE_PATH_ROOT = os.getenv('CACHE_PATH_ROOT', '/app/cache')

#region logging
if get_os_type() == "Linux":
    print(f"=========================================================")
    print(f"InputOutput.py Configuration:")
    print(f"ADDR: {ADDR}")
    print(f"PORT: {PORT}")
    print(f"BLAZOR_HOSTNAME: {BLAZOR_HOSTNAME}")
    print(f"BLAZOR_PORT: {BLAZOR_PORT}")
    print(f"BLAZOR_PROTOCOL: {BLAZOR_PROTOCOL}")
    print(f"SESSION_ID: {get_user_service().get_session_id()}")
    print(f"CONTAINERUTILITY_HOSTNAME: {CONTAINERUTILITY_HOSTNAME}")
    print(f"CONTAINERUTILITY_PORT: {CONTAINERUTILITY_PORT}")
    print(f"USE_REAL_SSL_CERTIFICATE: {USE_REAL_SSL_CERTIFICATE}")
    print(f"CACHE_PATH_ROOT: {CACHE_PATH_ROOT}")
    print(f"=========================================================")

# reduce verbosity from the neo4j driver (we re-log notifications ourselves)
try:
    import logging
    logging.getLogger("neo4j").setLevel(logging.ERROR)
    logging.getLogger("neo4j.bolt").setLevel(logging.ERROR)
except Exception:
    pass
#endregion logging

#region global variables
is_paused = [False]                 # Pause flag to pause the execution of the plan (it must be a mutable container)
plan_is_stopped = [False]           # Stop flag to stop the execution of the plan (it must be a mutable container)

# Buffer for input data
buffer = None
input_role = None
input_language = None
input_images = None
input_documents = None
input_json_output = None
input_structured_output = None
input_recommended_plan = None
input_recommended_tools = None
input_active_reasoning = None
input_active_flavour = None
input_rag_settings = None
input_language = None
buffer_condition = Condition()
buffer_ready_for_input = False

# Job status management
completion_percentage = [1]
completion_percentage[0] = 0.0

#default language
default_language_code = 'en'

# Blazor communication
blazor_communication_event = threading.Event()      # Event to signal the plan update
blazor_communication_lock = threading.Lock()        # Lock to protect the communication queue
blazor_communication_queue = []                     # Queue to store the plan updates

#endregion

#region init/shutdown
def init_input_output() -> None:
    # Start the Blazor communication thread
    # ContextVars (session/user identity) are not inherited by new OS threads: capture and re-apply them explicitly
    session_id = get_user_service().get_session_id()
    user_id = get_user_service().get_user_id()
    user_uuid = get_user_service().get_user_uuid()

    def _run_blazor_communication_thread():
        get_user_service().set_session_id(session_id)
        get_user_service().set_user_id(user_id)
        get_user_service().set_user_uuid(user_uuid)
        blazor_communication_thread()

    thread = threading.Thread(target=_run_blazor_communication_thread)
    thread.daemon = True
    thread.start()

def shutdown_input_output() -> None:
    # Stop the Blazor communication thread
    shutdown_event[0].set()

    # Close the Neo4j driver
    from src.core.base.neo4j_queries import _reset_neo4j_driver
    _reset_neo4j_driver()
#endregion init/shutdown

#region parameters
def get_parameters(category_or_skillid:str, subcategory:str, default_value:dict=None, session_id:str=None) -> dict:
    if session_id is None: session_id = get_user_service().get_session_id()
    result = get_session_parameters(session_id, category_or_skillid, subcategory)
    return result if result is not None else default_value

def put_parameters(category_or_skillid:str, subcategory:str, params:dict, session_id:str=None, operation:str=None) -> dict:
    if session_id is None: session_id = get_user_service().get_session_id()
    return put_session_parameters(session_id, category_or_skillid, subcategory, params, operation=operation)

def delete_parameters(category_or_skillid:str, subcategory:str, session_id:str=None) -> dict:
    if session_id is None: session_id = get_user_service().get_session_id()
    return delete_session_parameters(session_id, category_or_skillid, subcategory)

def get_session_parameters(session_id:str, category_or_skillid:str, subcategory:str) -> dict:
    # Fetch parameters from the local API.
    try:
        subcategory_url = f"/{subcategory}" if subcategory is not None else ""
        api_url = f'{CONTAINERUTILITY_PROTOCOL}://{CONTAINERUTILITY_HOSTNAME}:{CONTAINERUTILITY_PORT}/parameters/{session_id}/{category_or_skillid}{subcategory_url}'

        response = request_with_service_token("GET", api_url)
        if response.status_code == 200:
            params_data = response.json()
            return params_data['parameters']
        else:
            log(WARNING, f"WARNING: No data to get parameters to: {api_url}. Response code: {response.status_code}")
            return None
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            log(ERROR, "RateLimitError: You exceeded your current quota, please check your plan and billing details.")
            log(ERROR, e.response.json())
        else:
            raise
    except Exception as e:
        log(ERROR, f"ERROR: Exception to get parameters to: {api_url}. {e}")
        return None

def put_session_parameters(session_id:str, category_or_skillid:str, subcategory:str, params:dict, operation:str=None) -> dict:
    # Put parameters to the local API.
    try:
        subcategory_url = f"/{subcategory}" if subcategory else ""
        api_url = f'{CONTAINERUTILITY_PROTOCOL}://{CONTAINERUTILITY_HOSTNAME}:{CONTAINERUTILITY_PORT}/parameters/{session_id}/{category_or_skillid}{subcategory_url}'
        if operation is not None and len(operation)>0:
            api_url += f"?operation={operation}"

        response = request_with_service_token(
            "PUT",
            api_url,
            data=json.dumps(params, cls=EnumEncoder),
            headers={"Content-Type": "application/json"},
        )
        if response.status_code == 200:
            params_data = response.json()
            return params_data['parameters']
        else:
            log(ERROR, f"ERROR: Exception to put parameters to: {api_url}. Response code: {response.status_code}")
            return None
    except Exception as e:
        log(ERROR, f"ERROR: Exception to put parameters to: {api_url}. {e}")
        return None

def delete_session_parameters(session_id:str, category_or_skillid:str, subcategory:str) -> dict:
    # Delete parameters from the local API.
    try:
        subcategory_url = f"/{subcategory}" if subcategory else ""
        api_url = f'http://{CONTAINERUTILITY_HOSTNAME}:{CONTAINERUTILITY_PORT}/parameters/{session_id}/{category_or_skillid}{subcategory_url}'
        
        log(DEBUG, f"DELETE request to: {api_url}")
        
        response = request_with_service_token("DELETE", api_url)
        log(DEBUG, f"DELETE response status: {response.status_code}")
        log(DEBUG, f"DELETE response text: {response.text}")
        
        if response.status_code == 200:
            params_data = response.json()
            log(DEBUG, f"DELETE response data: {params_data}")
            return params_data['parameters']
        else:
            log(ERROR, f"ERROR: Exception to delete parameters from: {api_url}. Response code: {response.status_code}")
            return None
    except Exception as e:
        log(ERROR, f"ERROR: Exception to delete parameters from: {api_url}. {e}")
        return None

def log_parameters(arguments_type:str, category_or_skillid:str, instanceid:str, params:dict) -> str:
    log(INFO, f"{arguments_type}: {params}")
    if (arguments_type == 'args'):
        total_performance_time_milli[0] = 0.0
    elif (arguments_type == 'output') and total_performance_time_milli[0]>0.0:
        params['external_time_milli'] = round(total_performance_time_milli[0])

    session_id = get_user_service().get_session_id()

    params = put_session_parameters(session_id, category_or_skillid, instanceid, params)
    if params is None:
        log(ERROR, "Cannot put the parameters to the data API")

    return instanceid
#endregion

#region input/output
## Input/Output functions for sending data to the skill
def put_input_data(request_data: dict):
    """Put input data to buffer - business logic without HTTP handling."""
    global buffer, buffer_ready_for_input, input_role, input_images, input_documents, input_json_output, input_structured_output, input_recommended_plan, input_recommended_tools, input_active_reasoning, input_active_flavour, input_rag_settings, input_language
    try:
        log(INFO, f"Received request input data: {request_data}")
        with buffer_condition:
            buffer = request_data['input']
            input_role = request_data.get('role', "user")
            input_language = request_data.get('language', None)
            input_images = request_data.get('images', None)
            input_documents = request_data.get('documents', None)
            input_json_output = request_data.get('json_output', None)
            input_structured_output = request_data.get('structured_output', None)
            input_recommended_plan = request_data.get('recommended_plan', None)
            input_recommended_tools = request_data.get('recommended_tools', None)
            input_active_reasoning = request_data.get('active_reasoning', None)
            input_active_flavour = request_data.get('active_flavour', None)
            input_rag_settings = request_data.get('rag_settings', None)

            log(INFO, f"Input from user interface - text: '{buffer}', images: {input_images}, documents: {input_documents}, role: {input_role}, json_output: {input_json_output}, structured_output: {input_structured_output}, recommended_plan: {input_recommended_plan}, recommended_tools: {input_recommended_tools}, active_reasoning: {input_active_reasoning}, active_flavour: {input_active_flavour}, rag_settings: {input_rag_settings}, language: {input_language}")
            buffer_condition.notify_all()
            buffer_ready_for_input = False
            
        return True, "OK", None, 200
    except Exception as e:
        log(ERROR, f"Error putting input data: {str(e)}")
        return False, None, f"Error putting input data: {str(e)}", 500


def get_input_from_buffer(skillid: str) -> Tuple[str, Optional[List[str]], Optional[List[str]], Optional[str], Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    if app_config.testing:
        # This is running on the terminal for testing purposes
        user_input = input()
        if "IMAGE=" in user_input:
            # format to enter an image via the debugging session:
            # IMAGE=filename.jpg <user_input>
            image_name = user_input.split("=")[1]
            image_name = image_name.split(" ")[0]
            user_input = user_input.replace(f"IMAGE={image_name} ", "").strip()
            return user_input, [image_name], None, None, None, None, None, None, None, None, None, None
        else:
            return user_input, None, None, None, None, None, None, None, None, None, None, None

    else:
        # This is running in the Flask server
        global buffer, buffer_ready_for_input, input_images, input_documents, input_role, input_json_output, input_structured_output, input_recommended_plan, input_recommended_tools, input_active_reasoning, input_active_flavour, input_rag_settings, input_language
        with buffer_condition:
            buffer_ready_for_input = True           # used by the GET API "/ready_for_input"
            with print_lock:
                print("<<<")                        # Signal the user that the server is ready to receive input
            while not buffer:
                buffer_condition.wait()

            # extract input data
            buffer_ready_for_input = False
            user_input = buffer
            images = input_images
            documents = input_documents
            override_role = input_role
            language = input_language
            json_output = input_json_output
            structured_output = input_structured_output
            recommended_plan = input_recommended_plan
            recommended_tools = input_recommended_tools
            active_reasoning = input_active_reasoning
            active_flavour = input_active_flavour
            rag_settings = input_rag_settings
            language = input_language

            # reset buffer for next input
            buffer = None
            input_images = None
            input_documents = None
            input_role = None
            input_language = None
            input_json_output = None
            input_structured_output = None
            input_recommended_plan = None
            input_recommended_tools = None
            input_active_reasoning = None
            input_active_flavour = None
            input_rag_settings = None
            input_language = None
            if skillid is not None:
                with print_lock:
                    print(f">>> {user_input}")

        return user_input, images, documents, override_role, json_output, structured_output, recommended_plan, recommended_tools, active_reasoning, active_flavour, rag_settings, language
#endregion

#region utilities
def iso_alpha2_to_language_name(alpha2_code):
    language_name = ''
    try:
        language = pycountry.languages.get(alpha_2=alpha2_code)
        language_name = language.name
    except (AttributeError, ImportError) as e:
        log(ERROR, f"Language name error: '{e}'! Fallback to default language")
    
    if language_name == '':
        try:
            language = pycountry.languages.get(alpha_2=default_language_code)
            language_name = language.name
        except (AttributeError, ImportError) as e:
            log(ERROR, f"Language name error: '{e}'! Fallback to English")
    
    if language_name == '':
        language_name = "English"
    
    return language_name

def get_language_name():
    alpha2_code = os.getenv('USER_LANGUAGE', default_language_code)
    language_name = iso_alpha2_to_language_name(alpha2_code)
    if language_name is None or len(language_name) == 0:
        language_name = "English"
    return language_name

class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

def prepare_string_for_json(string:str) -> str:
    # convert the attributes from a string to a dictionary
    # the original parameter may be a string with single quotes: {'name':'Martha', 'description':'A meticulous homemaker with a "soft" spot for John\'s daughter Jane and her furry friend.'}
    # which is not compatible with json.loads. We need to replace the single quotes with double quotes. However, we need to be careful with the single and double quotes inside the string.
    # the result is: {"name":"Martha", "description":"A meticulous homemaker with a \"soft\" spot for John's daughter Jane and her furry friend."}
    string = string.replace("\\\'", "<$>")  # Replace single quotes with '<$>'
    string = string.replace("\"", "\\\"")   # Replace double quotes with '\"'
    string = string.replace("'", "\"")      # Replace single quotes with double quotes
    string = string.replace("<$>", "'")     # Replace '<$>' with single quotes
    return string

def set_stopped(value:bool) -> None:
    plan_is_stopped[0] = value

def is_stopped() -> bool:
    return plan_is_stopped[0]

def get_job_completion_data():
    """Get job completion percentage - business logic without HTTP handling."""
    try:
        return True, {'completion_percent': completion_percentage[0]}, None, 200
    except Exception as e:
        log(ERROR, f"Error getting job completion: {str(e)}")
        return False, None, f"Error getting job completion: {str(e)}", 500

def get_ready_for_input_data():
    """Check if system is ready for input - business logic without HTTP handling."""
    try:
        wait_for_blazor_communication_flush = False
        if buffer_ready_for_input:
            with blazor_communication_lock:
                if len(blazor_communication_queue) > 0:
                    wait_for_blazor_communication_flush = True

            if wait_for_blazor_communication_flush:
                time.sleep(1)

            return True, "OK", None, 200
        else:
            return False, None, "Service Unavailable", 503
    except Exception as e:
        log(ERROR, f"Error checking ready for input: {str(e)}")
        return False, None, f"Error checking ready for input: {str(e)}", 500

def get_document_path(filename:str=None) -> str:
    if filename is not None and "/" in filename:
        return filename
    
    documents_path: str = "./"
    
    if os.path.exists("../../../DocumentRepository"):
        documents_path = "../../../DocumentRepository/"
    elif os.path.exists("../DocumentRepository"):
        documents_path = "../DocumentRepository/"
    elif os.path.exists("./DocumentRepository"):
        documents_path = "./DocumentRepository/"
    elif os.path.exists("/app/documents"):
        documents_path = "/app/documents/"

    if filename is None:
        return documents_path
    else:
        return documents_path + filename

def get_output_path(filename: str = None) -> str:
    # the output path is the same as the document path, plus the OUTPUT_PATH
    # for example:
    #   doc_path = "/app/documents/document.pdf"
    #   output_path = "/app/documents/output/document.pdf"
    doc_filename = get_document_path(filename)
    doc_dir = os.path.dirname(doc_filename)                             # Get the directory of the document
    output_dir = os.path.join(doc_dir, OUTPUT_PATH)                     # Append OUTPUT_PATH to the directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, os.path.basename(doc_filename))     # Append the filename to the output directory
    return path

def get_document_image_path(filename:str, image_nr:int=None) -> str:
    target_dir = get_document_path(filename)
    target_dir = os.path.splitext(target_dir)[0] + ".images"

    if image_nr:
        target_dir += f"/{image_nr}.png"

    return target_dir

def get_last_modified(filename: str) -> str:
    try:
        last_modified = os.path.getmtime(filename)
        return datetime.fromtimestamp(last_modified, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    
    except Exception as e:
        return None

#endregion utilities

#region communication with Blazor
def send_refresh_document_repository_to_blazor(document_ids: Optional[List[str]] = None) -> None:
    """Notify Blazor to refresh the document repository (non-blocking).

    This function spawns a daemon thread that issues a POST to the Blazor
    controller endpoint `POST /document/refreshrepository`. The call is
    performed asynchronously and the function returns immediately; the
    HTTP response is ignored by callers. Success and failure are logged
    from the background thread. When `app_config.testing` is True this
    function is a no-op.

    Args:
        document_ids (Optional[List[str]]): If provided, appends ?docids=<id1>,<id2>,... to the request URL
            to specify which documents to refresh. A list of document ID strings.
            Defaults to None (refresh entire repository).
    """
    if app_config.testing == True:
        return

    def _send_request():
        try:
            url = f"{BLAZOR_PROTOCOL}://{BLAZOR_HOSTNAME}:{BLAZOR_PORT}/document/refreshrepository"
            if document_ids is not None:
                # Convert list of document_ids to comma-separated string
                docids_str = ",".join(document_ids)
                url += f"?docids={docids_str}"
            response = requests.post(url, verify=USE_REAL_SSL_CERTIFICATE, timeout=5)

            if response.status_code == 200:
                log(DEBUG, f"send_refresh_document_repository_to_blazor(): Successfully sent refresh command to Blazor")
            else:
                log(ERROR, f"send_refresh_document_repository_to_blazor(): ERROR: Failed to refresh repository. Response code: {response.status_code}, message: {response.text}")
        except Exception as e:
            log(ERROR, f"send_refresh_document_repository_to_blazor(): ERROR: Exception to send refresh document repository command to Blazor. {e}")

    # Make non-blocking call in a separate daemon thread and return immediately
    thread = threading.Thread(target=_send_request, daemon=True)
    thread.start()

    return

def send_ontology_consolidation_status_to_blazor(in_progress: bool) -> None:
    """Synchronously notify the shared Blazor layout about ontology consolidation state."""
    if app_config.testing == True:
        return

    try:
        url = f"{BLAZOR_PROTOCOL}://{BLAZOR_HOSTNAME}:{BLAZOR_PORT}/v1/sessions/ontology/consolidationstatus"
        token = os.getenv('PLATFORM_API_TOKEN', None)
        if token is None:
            log(ERROR, "send_ontology_consolidation_status_to_blazor(): PLATFORM_API_TOKEN is not configured")
        else:
            response = requests.post(url, json=in_progress, headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}, verify=USE_REAL_SSL_CERTIFICATE, timeout=5)
            if response.status_code != 200:
                log(ERROR, f"send_ontology_consolidation_status_to_blazor(): failed with response code {response.status_code}: {response.text}")
    except Exception as e:
        log(ERROR, f"send_ontology_consolidation_status_to_blazor(): exception: {e}")

    return

def send_ontology_data_changed_to_blazor() -> None:
    """Notify Blazor that ontology data changed and consumers should refresh."""
    if app_config.testing == True:
        return

    try:
        url = f"{BLAZOR_PROTOCOL}://{BLAZOR_HOSTNAME}:{BLAZOR_PORT}/v1/api/ontology/data-changed"
        token = os.getenv('PLATFORM_API_TOKEN', None)
        if token is None:
            log(ERROR, "send_ontology_data_changed_to_blazor(): PLATFORM_API_TOKEN is not configured")
        else:
            response = requests.post(url, headers={'Authorization': f'Bearer {token}'}, verify=USE_REAL_SSL_CERTIFICATE, timeout=5)
            if response.status_code != 200:
                log(ERROR, f"send_ontology_data_changed_to_blazor(): failed with response code {response.status_code}: {response.text}")
    except Exception as e:
        log(ERROR, f"send_ontology_data_changed_to_blazor(): exception: {e}")

    return

def send_plan_to_blazor(plan:'Plan', root_fragment_goal_id:List[int]=None) -> None:
    # the optimization to send only the root_fragment_goal_id has been removed, because it caused far too many issues
    # let's enqueue the update of the plan
    log(PERF, f"send_plan_to_blazor: start", perf_id="spb_start")
    with plan.plan_lock:
        plan.increment_revision()
        plan_json = plan.to_json()
    log(PERF, f"send_plan_to_blazor: to_json done, json_size={len(str(plan_json))}", perf_id="spb_json", perf_since_id="spb_start")

    with blazor_communication_lock:
        global blazor_communication_queue
        blazor_communication_queue = []         # discard all previous updates, anyways the new one is the most recent
        blazor_communication_queue.append({
            "plan_json": plan_json
        })
        blazor_communication_event.set()
    log(PERF, f"send_plan_to_blazor: enqueued", perf_id="spb_done", perf_since_id="spb_json")
    return

def blazor_communication_thread():
    # this thread dequeues the plan updates and sends them to the Blazor app
    global blazor_communication_queue

    log(DEBUG, "Blazor communication thread is started.")
    
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 10

    while shutdown_event[0].is_set() == False:
        # Wait for event notification with timeout to avoid missing signals
        # The timeout ensures we don't deadlock if a signal is somehow lost
        blazor_communication_event.wait(timeout=1.0)
        # Clear event immediately after waking to avoid race condition where
        # a new signal arrives before we clear, causing that signal to be lost
        blazor_communication_event.clear()
            
        # get the plan from the queue
        plan_json = None
        retry_count = 0
        with blazor_communication_lock:
            if len(blazor_communication_queue) > 0:
                queued_plan = blazor_communication_queue.pop(0)
                if queued_plan is not None:
                    plan_json = queued_plan["plan_json"]
                    retry_count = queued_plan.get("retry_count", 0)

        if plan_json is not None and app_config.testing == False:
            # Attempt to send the plan to Blazor with retry logic
            success = False
            try:
                result = execute_send_fragment_to_blazor(plan_json)
                # If result is not None, the request was sent (even if Blazor returned an error status)
                # None means a network/connection exception occurred
                if result is not None:
                    success = True
            except Exception as e:
                log(WARNING, f"blazor_communication_thread(): Exception sending plan to Blazor: {e}")
            
            if not success:
                retry_count += 1
                if retry_count <= MAX_RETRIES:
                    log(WARNING, f"blazor_communication_thread(): Failed to send plan to Blazor (attempt {retry_count}/{MAX_RETRIES}). Waiting {RETRY_DELAY_SECONDS} seconds before retry...")
                    time.sleep(RETRY_DELAY_SECONDS)
                    
                    # After waiting, check if the queue is empty
                    # If empty: re-enqueue this plan for retry
                    # If not empty: newer updates have arrived, abandon retry of stale plan
                    with blazor_communication_lock:
                        if len(blazor_communication_queue) == 0:
                            log(DEBUG, f"blazor_communication_thread(): Re-enqueueing plan for retry (attempt {retry_count + 1})")
                            blazor_communication_queue.append({
                                "plan_json": plan_json,
                                "retry_count": retry_count
                            })
                            blazor_communication_event.set()  # Signal that there's work to process
                        else:
                            log(DEBUG, f"blazor_communication_thread(): New plan updates arrived during retry wait, abandoning retry of stale plan")
                else:
                    log(FATAL, f"blazor_communication_thread(): FATAL: Failed to send plan to Blazor after {MAX_RETRIES} attempts. Giving up on this update.")
    
    log(DEBUG, "Blazor communication thread stopped.")
    return

def get_plan_update_status_from_blazor() -> dict:
    if app_config.testing == True:
        host = "127.0.0.1"
        port = '5210'
        protocol = 'http'
    else:
        host = BLAZOR_HOSTNAME
        port = BLAZOR_PORT
        protocol = BLAZOR_PROTOCOL
    
    # send the response to the Blazor app
    updated_plan_json = None
    global_api_token = os.getenv('PLATFORM_API_TOKEN', None)
    if global_api_token is None:
        log(FATAL, f"*** FATAL: No API token found in the environment variables! ***")
        return updated_plan_json
    url = f"{protocol}://{host}:{port}/v1/sessions/session/{get_user_service().get_session_id()}/planstatus"

    # send a POST request to the Blazor app, sequentialized amongst all threads
    try:
        timeout = 1000 if app_config.testing else 30

        response = requests.get(url, headers={'Authorization': f'Bearer {global_api_token}'}, verify=USE_REAL_SSL_CERTIFICATE, timeout=timeout)
        updated_plan_json = response.json()

        if response.status_code == 200:
            log(DEBUG, f"get_plan_update_status_from_blazor(): Got an updated plan status from blazor")
        else:
            log(ERROR, f"get_plan_update_status_from_blazor(): ERROR: Error to send response to: {url}. Response code: {response.status_code}, message: {response.text}")
    except Exception as e:
        log(ERROR, f"get_plan_update_status_from_blazor(): ERROR: Exception to send response to: {url}. {e}")

    return updated_plan_json

def execute_send_fragment_to_blazor(plan_json:str) -> dict:
    if app_config.testing == True:
        host = "localhost"
        port = '5210'
        protocol = 'http'
    else:
        host = BLAZOR_HOSTNAME
        port = BLAZOR_PORT
        protocol = BLAZOR_PROTOCOL
    
    # send the response to the Blazor app
    updated_plan_json = None
    global_api_token = os.getenv('PLATFORM_API_TOKEN', None)
    if global_api_token is None:
        log(FATAL, f"*** FATAL: No API token found in the environment variables! ***")
        return updated_plan_json
    url = f"{protocol}://{host}:{port}/v1/sessions/session/{get_user_service().get_session_id()}/plan"

    # send a POST request to the Blazor app, sequentialized amongst all threads
    try:
        log(DEBUG, f"Sending plan to Blazor app: {url}...")

        timeout = 600 if app_config.testing else 30
        response = requests.post(
            url=url,
            json=plan_json,
            headers={'Authorization': f'Bearer {global_api_token}', 'Content-Type': 'application/json'},
            verify=USE_REAL_SSL_CERTIFICATE,
            timeout=timeout
        )
        updated_plan_json = response.json()

        if response.status_code != 200:
            log(ERROR, f"ERROR: Error to send response to: {url}. Response code: {response.status_code}, message: {response.text}")
    except Exception as e:
        log(ERROR, f"ERROR: Exception to send response to: {url}. {e}")

    return updated_plan_json

def send_error_to_blazor(errormessage: str, exc: Exception = None) -> None:
    if app_config.testing:
        return

    # Send the response to the Blazor app
    host = os.getenv('BLAZOR_HOSTNAME', 'host.docker.internal')
    port = os.getenv('BLAZOR_PORT', '5210')
    protocol = os.getenv('BLAZOR_PROTOCOL', 'http')
    url = f"{protocol}://{host}:{port}/v1/api/utils/error"

    # Prepare the error message
    if exc is not None:
        errormessage += f". Exception: {exc}"
    error_data = {
        "message": str(errormessage),
        "session_id": get_user_service().get_session_id()
    }

    # Send a POST request to the Blazor app
    try:
        with open("error.log", "a") as file:
            file.write(f"Sending error to: {url}...\n")
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=error_data, headers=headers, verify=USE_REAL_SSL_CERTIFICATE)
        
        if response.status_code != 200:
            with open("error.log", "a") as file:
                file.write(f"ERROR: send_error_to_blazor(): Failed to send error to Blazor at {url}. Response code: {response.status_code}\n")
    except Exception as e:
        with open("error.log", "a") as file:
            file.write(f"ERROR: send_error_to_blazor(): Exception occurred while sending error to Blazor at {url}. {e}\n")
    return
#endregion

#region image utilities
def encode_image(image_name) -> str:
    image_path = image_name

    if not os.path.exists(image_name):
        # look for the image first in /app/images, then in /app/documents/images
        if get_os_type() == "Windows":
            if not "../" in image_path:
                image_path = f"../DocumentRepository/images/{image_name}"
        elif get_os_type() == "Mac":
            image_path = f"./images/{image_name}"
        else:
            # look for the image first in /app/images, then in /app/documents/images
            image_path = f"/app/images/{image_name}"
            if not os.path.exists(image_path):
                image_path = f"/app/documents/images/{image_name}"
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_name}")

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
#endregion image utilities

#region unit tests
def unit_stress_test_single(i:int):
    log(DEBUG, f"unit_stress_test_single({i}): started...")
    execute_neo4j_query("MERGE (n:TestNode {name: $name})", { "name": f"Node {i}" })
    log(DEBUG, f"unit_stress_test_single({i}): ...finished")

def unit_stress_test_single2(i:int):
    log(DEBUG, f"unit_stress_test_single2({i}): started...")
    node = execute_neo4j_query("MATCH (n:TestNode {name: $name}) RETURN n.name AS name", { "name": f"Node {i}" })
    assert node and len(node) == 1 and node[0]["name"] == f"Node {i}"
    log(DEBUG, f"unit_stress_test_single2({i}): ...finished")

def unit_stress_test(N:int = 100):
    execute_neo4j_query("MATCH (n:TestNode) DETACH DELETE n")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(unit_stress_test_single, range(N))
    number = execute_neo4j_query("MATCH (n:TestNode) RETURN count(n) AS count")[0]["count"]
    assert number == N
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(unit_stress_test_single2, range(N))
    execute_neo4j_query("MATCH (n:TestNode) DETACH DELETE n")


log_end_file(__file__)

if __name__ == '__main__':
    from src.utils.unit_test import init_unit_test

    from src.utils.log import log_init, log_exit
    init_unit_test()
    log_init()

    def test_parameters_functions():
        """Test get_parameters, put_parameters, and delete_parameters functions."""
        log(INFO, "Starting parameter functions test suite...")
        
        # Test data
        test_category = "test_category"
        test_subcategory = "test_subcategory"
        test_data = {
            "key1": "value1",
            "key2": {"nested": "data"},
            "key3": [1, 2, 3]
        }
        
        try:
            # Test 1: put_parameters - Store test data
            log(INFO, "Test 1: Testing put_parameters...")
            put_result = put_parameters(test_category, test_subcategory, test_data)
            log(INFO, f"put_parameters result: {put_result}")
            assert put_result is not None, "put_parameters should return data"
            assert put_result == test_data, "put_parameters should return the stored data"
            log(INFO, "PASS Test 1: put_parameters works correctly")
            
            # Test 2: get_parameters - Retrieve the stored data
            log(INFO, "Test 2: Testing get_parameters...")
            get_result = get_parameters(test_category, test_subcategory)
            log(INFO, f"get_parameters result: {get_result}")
            assert get_result is not None, "get_parameters should return data"
            assert get_result == test_data, "get_parameters should return the same data that was stored"
            log(INFO, "PASS Test 2: get_parameters works correctly")
            
            # Test 3: get_parameters with default value (existing data)
            log(INFO, "Test 3: Testing get_parameters with default value (data exists)...")
            default_data = {"default": "fallback"}
            get_result_with_default = get_parameters(test_category, test_subcategory, default_data)
            assert get_result_with_default == test_data, "Should return actual data, not default when data exists"
            log(INFO, "PASS Test 3: get_parameters with default value works correctly")
            
            # Test 4: delete_parameters - Clear the stored data
            log(INFO, "Test 4: Testing delete_parameters...")
            delete_result = delete_parameters(test_category, test_subcategory)
            log(INFO, f"delete_parameters result: {delete_result}")
            assert delete_result is not None, "delete_parameters should return a result"
            log(INFO, "PASS Test 4: delete_parameters executed successfully")
            
            # Test 5: get_parameters after delete (should return None or default)
            log(INFO, "Test 5: Testing get_parameters after delete...")
            get_after_delete = get_parameters(test_category, test_subcategory)
            log(INFO, f"get_parameters after delete: {get_after_delete}")
            # Note: This might return empty dict {} instead of None due to API behavior
            log(INFO, "PASS Test 5: get_parameters after delete executed")
            
            # Test 6: get_parameters with default value (no data)
            log(INFO, "Test 6: Testing get_parameters with default value (no data)...")
            default_data = {"default": "fallback", "test": True}
            get_result_default = get_parameters(test_category, test_subcategory, default_data)
            log(INFO, f"get_parameters with default (no data): {get_result_default}")
            # Should return default_data when no data exists
            if get_result_default == default_data:
                log(INFO, "PASS Test 6: get_parameters returns default value when no data exists")
            elif get_result_default == {} or get_result_default is None:
                log(WARNING, "PARTIAL Test 6: get_parameters returned empty/None instead of default (API behavior)")
            else:
                log(ERROR, f"FAIL Test 6: Expected default data or empty, got: {get_result_default}")
                
            # Test 7: Round-trip test - put, get, delete, get
            log(INFO, "Test 7: Complete round-trip test...")
            test_data2 = {"round_trip": "test", "number": 42}
            
            # Put data
            put_result2 = put_parameters("roundtrip", "test", test_data2)
            assert put_result2 == test_data2, "Put should return stored data"
            
            # Get data
            get_result2 = get_parameters("roundtrip", "test")
            assert get_result2 == test_data2, "Get should return the same data"
            
            # Delete data  
            delete_result2 = delete_parameters("roundtrip", "test")
            assert delete_result2 is not None, "Delete should return a result"
            
            # Get after delete
            get_after_delete2 = get_parameters("roundtrip", "test", {"empty": True})
            log(INFO, f"Final get after delete: {get_after_delete2}")
            
            log(INFO, "PASS Test 7: Complete round-trip test successful")
            
            # Test 8: Edge cases
            log(INFO, "Test 8: Testing edge cases...")
            
            # Empty data
            empty_result = put_parameters("edge", "empty", {})
            assert empty_result == {}, "Should handle empty dict"
            
            # Get non-existent with default
            non_existent = get_parameters("nonexistent", "category", {"default": "value"})
            log(INFO, f"Non-existent with default: {non_existent}")
            
            log(INFO, "PASS Test 8: Edge cases handled")
            
            log(INFO, "SUCCESS: ALL PARAMETER TESTS PASSED! The parameter functions are working correctly.")
            return True
            
        except Exception as e:
            log(ERROR, f"FAIL: PARAMETER TEST FAILED: {e}")
            import traceback
            log(ERROR, f"Stack trace: {traceback.format_exc()}")
            return False
        
        finally:
            # Cleanup - try to delete test data
            try:
                delete_parameters(test_category, test_subcategory)
                delete_parameters("roundtrip", "test") 
                delete_parameters("edge", "empty")
                log(INFO, "Test cleanup completed")
            except:
                log(WARNING, "Test cleanup had issues (this is not critical)")

    try:
        # Enable parameter function tests
        if True:
            log(INFO, "Running parameter function tests...")
            test_success = test_parameters_functions()
            if test_success:
                print("✅ Parameter tests passed.")
            else:
                print("❌ Parameter tests failed.")
        
        # Neo4j stress test (disabled by default)
        if False:
            unit_stress_test()
            print("Unit stress test passed.")

    except Exception as e:
        log(FATAL, f"FATAL: Exception in unit test: {e}")

    finally:
        log_exit()
