import time
import requests
from typing import Optional, Tuple

from src.utils.log import log, DEBUG, INFO, WARNING, ERROR
from src.config.constants import UTILS_BASE_URI, RESERVATION_MINUTES
from src.services.user_service import get_user_service


#region proxy to the Utility container
def reserve_pod(pod_id:str, session_id:str, minutes:int=60, llm_model_config:Optional[dict]=None) -> bool:
    """
    Proxy function to reserve a pod via the REST API.
    :param pod_id: The ID of the pod to reserve.
    :param session_id: The session ID for the reservation.
    :param minutes: The reservation duration in minutes (default is 60).
    :return: True if the reservation was successful, False otherwise.
    """
    url = f"{UTILS_BASE_URI}/runpod/pods/{pod_id}/reserve"
    payload = {
        "sessionid": session_id,
        "minutes": minutes,
        "llm_model_config": llm_model_config
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            log(DEBUG, f"Successfully reserved pod '{pod_id}' for session '{session_id}' for {minutes} minutes.")
            return True
        else:
            log(ERROR, f"Failed to reserve pod '{pod_id}'. Response: {response.status_code}, {response.text}")
            return False
    except Exception as e:
        log(ERROR, f"Error while reserving pod '{pod_id}': {e}")
        return False

def unreserve_pod(pod_id:str, session_id:str) -> bool:
    """
    Proxy function to unreserve a pod via the REST API.
    :param pod_id: The ID of the pod to unreserve.
    :param session_id: The session ID for the unreservation.
    :return: True if the unreservation was successful, False otherwise.
    """
    url = f"{UTILS_BASE_URI}/runpod/pods/{pod_id}/unreserve"
    payload = {
        "sessionid": session_id
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            log(DEBUG, f"Successfully unreserved pod '{pod_id}' for session '{session_id}'.")
            return True
        else:
            log(ERROR, f"Failed to unreserve pod '{pod_id}'. Response: {response.status_code}, {response.text}")
            return False
    except Exception as e:
        log(ERROR, f"Error while unreserving pod '{pod_id}': {e}")
        return False

def unreserve_all_pods_for_session(sessionid:str) -> bool:
    success = False

    # This function unreserves any running pods eventually reserved to this session
    try:
        log(DEBUG, f"Unreserving pods for session {sessionid}...")

        # Call the container utility service to unreserve pods: /runpod/sessions/<sessionid>/unreserve', method='POST'
        uri = f"{UTILS_BASE_URI}/runpod/sessions/{sessionid}/unreserve"
        response = requests.post(uri)
        if response.status_code == 204:
            # success but nothing was done (no pods reserved)
            log(DEBUG, f"Attempted to unreserve pods for session {sessionid}, but no pods were reserved.")
            success = True
        elif response.status_code == 200:
            log(INFO, f"Successfully unreserved pods for session {sessionid}")
            success = True
        else:
            log(WARNING, f"Failed to unreserve pods: {response.text}")
            success = False

    except Exception as e:
        log(ERROR, f"Error unreserving pods: {str(e)}")
        success = False

    return success
#endregion proxy to the Utility container


#region model reservation for agents
def runpod_reserve_model(model_reservation: Optional[dict], llm_model_config: Optional[dict]) -> Optional[dict]:
    if llm_model_config is None:
        raise ValueError("Model configuration is None.")
    
    if "modelloader" not in llm_model_config or llm_model_config["modelloader"] != "runpod.io":
        # only runpod models can be reserved
        return None
    
    new_reservation = model_reservation
    reservation_success = None

    if model_reservation is None or model_reservation["expiration_time"] <= time.time():
        # Create a new model reservation
        log(DEBUG, f"Creating new reservation for model '{llm_model_config['id']}'")

        reservation_success = reserve_pod(None, get_user_service().get_session_id(), RESERVATION_MINUTES, llm_model_config)
        if reservation_success:
            new_reservation = {
                "model_id": llm_model_config["id"],
                "pod_id": None,
                "create_time": time.time(),
                "expiration_time": time.time() + (RESERVATION_MINUTES * 60),
                "endpoint": llm_model_config["endpoint"]
            }

    # renew reservation if the time is not expired, the model is the same and the expiration time is approaching by 5 minutes
    elif model_reservation["model_id"] == llm_model_config["id"] and model_reservation["expiration_time"] - time.time() < 5 * 60:
        log(DEBUG, f"Renewing reservation for model '{llm_model_config['id']}'")

        # renew and update the reservation
        reservation_success = reserve_pod(model_reservation["pod_id"], get_user_service().get_session_id(), RESERVATION_MINUTES)
        new_reservation["expiration_time"] = time.time() + (RESERVATION_MINUTES * 60)
        new_reservation["endpoint"] = llm_model_config["endpoint"]

    if reservation_success == False:
        log(ERROR, f"Failed to reserve model '{llm_model_config['id']}'")
        return None

    return new_reservation

def runpod_get_endpoint(model_reservation:dict) -> str:
    if model_reservation is None:
        return
    if "endpoint" not in model_reservation:
        raise ValueError("Model reservation is None or endpoint is missing.")

    endpoint = model_reservation["endpoint"]
    return endpoint

#endregion model reservation for agents
