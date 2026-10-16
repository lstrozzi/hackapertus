import os
import time
import http.client
import json
from typing import Tuple, Dict, Optional
from enum import Enum
from src.utils.log import log, DEBUG, INFO, WARNING, ERROR

RUNPOD_KEY = [os.environ.get("RUNPOD_KEY", None)]
ALLOWED_GPU_TYPES = ['NVIDIA GeForce RTX 4090', 'NVIDIA RTX A5000', 'NVIDIA A40', 'NVIDIA GeForce RTX 3090', 'NVIDIA RTX A6000', 'NVIDIA RTX A4000', 'NVIDIA RTX A4500', 'NVIDIA A100 80GB PCIe', 'NVIDIA L4', 'NVIDIA RTX 4000 Ada Generation', 'NVIDIA A100-SXM4-80GB', 'NVIDIA RTX 6000 Ada Generation', 'NVIDIA H100 80GB HBM3', 'NVIDIA L40S', 'NVIDIA L40', 'NVIDIA H100 PCIe', 'NVIDIA RTX 2000 Ada Generation', 'NVIDIA GeForce RTX 3080 Ti', 'NVIDIA H100 NVL', 'NVIDIA GeForce RTX 3080', 'NVIDIA GeForce RTX 3070', 'NVIDIA RTX A2000', 'NVIDIA A30', 'NVIDIA H200', 'NVIDIA GeForce RTX 4080', 'Tesla V100-SXM2-32GB', 'NVIDIA GeForce RTX 4070 Ti', 'AMD Instinct MI300X OAM', 'Tesla V100-SXM2-16GB', 'Tesla V100-PCIE-16GB', 'NVIDIA RTX 5000 Ada Generation', 'NVIDIA GeForce RTX 3090 Ti', 'NVIDIA RTX 4000 SFF Ada Generation', 'Tesla V100-FHHL-16GB', 'NVIDIA GeForce RTX 4080 SUPER', 'NVIDIA A100-SXM4-40GB', 'NVIDIA GeForce RTX 5080', 'NVIDIA GeForce RTX 5090']
PODNAMEPREFIX = "managed-"

class PodManagedStatus(Enum):
    MANAGED = "managed"
    UNMANAGED = "unmanaged"

#region runpod APIs
def get_pods(pod_id: Optional[str] = None) -> Tuple[bool, str]:
    conn = http.client.HTTPSConnection("rest.runpod.io")
    try:
        headers = {'Authorization': f"Bearer {RUNPOD_KEY[0]}"}
        uri = f"/v1/pods/{pod_id}" if pod_id else "/v1/pods"
        conn.request("GET", uri, headers=headers)
        res = conn.getresponse()
        data = res.read()
        if res.status == 200:
            parsed_data = json.loads(data.decode("utf-8"))
            return True, parsed_data
        else:
            decoded_data = data.decode("utf-8")
            return False, decoded_data
    finally:
        conn.close()

def create_pod(name:str, gpuType:str, dockerStartCommand:str, volumeSizeGb:int) -> Tuple[bool, str]:
    conn = http.client.HTTPSConnection("rest.runpod.io")
    try:
        dockerStartCommandArray = dockerStartCommand.split(" ")

        payload = {
            #"allowedCudaVersions": ["12.7"],
            "cloudType":"SECURE",
            "computeType": "GPU",
            "containerDiskInGb": 8,
            #"countryCodes": [],
            #"cpuFlavorPriority": "availability",
            #"dataCenterIds": ["EU-RO-1", "CA-MTL-1"],
            #"dataCenterPriority": "availability",
            #"dockerEntrypoint": [],
            "dockerStartCmd": dockerStartCommandArray,
            "env": {
                "HF_HOME": "/workspace/hf_home",
                "HF_TOKEN": "{{ RUNPOD_SECRET_HF_TOKEN }}",
                "PUBLIC_KEY": "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDe19FVg1GqyP35CwazPx4QxiZ/9vSaEFyOvUP/rZ6L959UoVt13V2bNOisEOUBOjOcfZX80XFLJ+T4nJfeafQ0cncC++pgQtZl4cmlUc2Q1Sn9bZx3JqmEkaSoEowbEk1bXaLPtCQ9WrJJo31ZOG2Cvv8zsCmNNhc3Lqetg9ewz312JUl++UiorR1mcPvqJEawBtqh3jWVqK3MwmuD9cohx1aXfAguJFTGi9n4MiX/9VvUQkYRCGk1ft1e83MkjyNfVzVBuuoud+G7+0xq+d+rbZ3Dkk2ddYfgi0GmuH/WHUlzkAWrwPvi0YUskcuIT1On5BV3fndl9PXE7a5KAjf5T6aAZpm7RAWMnzTzb2OrY/k1uKfnOXzj9jH91bE4KtJsvXFwUy2ydN8O5PPN7NA6CPkxhq/mgddWtnUM4TlkJt/Z/Ol4KZcElyJW/HQ2RXlcy/AxYyeFmxzzwpKS5Ic7IUIJX/iT5blz9IPWYHgKlxaZcKNRSzIkBZe95nZfhcE="
            },
            "gpuCount": 1,
            "gpuTypeIds": [ gpuType ],
            #"gpuTypePriority": "availability",
            "imageName": "vllm/vllm-openai:latest",
            "interruptible": False,
            "name": name,
            "ports": ["8000/http", "22/tcp"],
            "supportPublicIp": True,
            "volumeInGb": volumeSizeGb,
            "volumeMountPath": "/workspace"
        }
        headers = {
            'Content-Type': "application/json",
            'Authorization': f"Bearer {RUNPOD_KEY[0]}"
        }
        conn.request("POST", "/v1/pods", json.dumps(payload), headers)

        res = conn.getresponse()
        data = res.read()

        return res.status >= 200 and res.status <= 201, json.loads(data.decode("utf-8"))
    finally:
        conn.close()

def stop_pod(pod_id: str) -> Tuple[bool, str]:
    conn = http.client.HTTPSConnection("rest.runpod.io")
    try:
        headers = {'Authorization': f"Bearer {RUNPOD_KEY[0]}"}
        conn.request("POST", f"/v1/pods/{pod_id}/stop", headers=headers)
        res = conn.getresponse()
        data = res.read()
        return res.status == 200, json.loads(data.decode("utf-8")) if res.status == 200 else data.decode("utf-8")
    finally:
        conn.close()

def restart_pod(pod_id: str) -> Tuple[bool, str]:
    conn = http.client.HTTPSConnection("rest.runpod.io")
    try:
        headers = {'Authorization': f"Bearer {RUNPOD_KEY[0]}"}
        conn.request("POST", f"/v1/pods/{pod_id}/start", headers=headers)       # restart is the same as start. The /restart endpoint has another semantic meaning
        res = conn.getresponse()
        data = res.read()
        return res.status == 200, json.loads(data.decode("utf-8")) if res.status == 200 else data.decode("utf-8")
    finally:
        conn.close()

def delete_pod(pod_id: str) -> Tuple[bool, str]:
    conn = http.client.HTTPSConnection("rest.runpod.io")
    try:
        headers = {'Authorization': f"Bearer {RUNPOD_KEY[0]}"}
        conn.request("DELETE", f"/v1/pods/{pod_id}", headers=headers)
        res = conn.getresponse()
        data = res.read()
        return res.status == 200, str(data)
    finally:
        conn.close()
#endregion runpod APIs

#region managed pods
managed_pods:Dict[str, dict] = {}

def get_managed_pods(pod_id:str | None = None, sync:bool=True) -> Dict[str, dict]:
    if sync:
        sync_managed_pods()

    return managed_pods

def manage_pod(pod_id: str, managed_status: PodManagedStatus) -> bool:
    success = False

    sync_managed_pods()
    if pod_id in managed_pods:
        managed_pods[pod_id]["managed_status"] = managed_status.value

        if managed_status == PodManagedStatus.UNMANAGED:
            # remove all reservations for the pod
            if "reservations" in managed_pods[pod_id]:
                del managed_pods[pod_id]["reservations"]
                log(INFO, f"Removed all reservations for pod {pod_id}.")
            
            log(INFO, f"Pod {pod_id} is now unmanaged.")
        else:
            # add pod to managed pods
            managed_pods[pod_id]["managed_status"] = PodManagedStatus.MANAGED.value
            log(INFO, f"Pod {pod_id} is now managed.")

        success = True

    return success

def reserve_pod(pod_id: str | None, sessionid:str | None, minutes:int, sync:bool=True) -> bool:
    """
    Reserve a pod for a specific session and duration (minutes).
    """
    if sync:
        sync_managed_pods()

    if pod_id not in managed_pods:
        log(ERROR, f"Pod {pod_id} is not existing.")
        return False

    if managed_pods[pod_id]["managed_status"] != PodManagedStatus.MANAGED.value:
        log(ERROR, f"Pod {pod_id} is not managed.")
        return True
    
    expiration_time = time.time() + (minutes * 60)
    if "reservations" not in managed_pods[pod_id]:
        managed_pods[pod_id]["reservations"] = {}
    if sessionid not in managed_pods[pod_id]["reservations"]:
        managed_pods[pod_id]["reservations"][sessionid] = {}

    managed_pods[pod_id]["reservations"][sessionid]["expiration_time"] = expiration_time
    managed_pods[pod_id]["reservations"][sessionid]["expiration_time_UTC"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(expiration_time))
    managed_pods[pod_id]["reservations"][sessionid]["expiration_time_local"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(expiration_time))
    log(INFO, f"Reserved pod {pod_id} for session {sessionid} for {minutes} minutes. Expiration time: {managed_pods[pod_id]['reservations'][sessionid]['expiration_time_local']} (local time)")
    return True

def unreserve_pod(pod_id:str | None, sessionid:str | None, sync:bool=True) -> bool:
    """
    Unreserve a pod for a specific session.
    Returns True if unreserved successfully, False otherwise.
    """
    if sync:
        sync_managed_pods()

    if pod_id not in managed_pods:
        log(WARNING, f"Pod {pod_id} is not existing.")
        return False

    if managed_pods[pod_id]["managed_status"] != PodManagedStatus.MANAGED.value:
        log(ERROR, f"Pod {pod_id} is not managed.")
        return True
    
    if "reservations" in managed_pods[pod_id] and sessionid in managed_pods[pod_id]["reservations"]:
        del managed_pods[pod_id]["reservations"][sessionid]
        if len(managed_pods[pod_id]["reservations"]) == 0:
            del managed_pods[pod_id]["reservations"]
        log(INFO, f"Unreserved pod {pod_id} for session {sessionid}.")
        return True

    log(DEBUG, f"Session {sessionid} is not reserved for pod {pod_id}.")
    return False

def unreserve_session(sessionid:str) -> Tuple[bool, int]:
    """
    Unreserve all pods for a specific session.
    """
    success = True
    unreserved_sessions_count = 0
    sync_managed_pods()

    for pod_id in managed_pods.keys():
        if managed_pods[pod_id]["managed_status"] == PodManagedStatus.MANAGED.value:
            unreserved = unreserve_pod(pod_id, sessionid, sync=False)
            if unreserved:
                unreserved_sessions_count += 1
            success = success and unreserve_pod
            
    if not success:
        log(ERROR, f"Failed to unreserve session {sessionid} from all pods.")

    return success, unreserved_sessions_count

def sync_managed_pods() -> None:
    try:
        global managed_pods
        success, pods = get_pods()
        if not success:
            raise ValueError("Failed to fetch managed pods.")
        if not isinstance(pods, list):
            raise ValueError("Invalid response format for managed pods.")

        # Create a set of current pod IDs from the fetched pods
        current_pod_ids = {pod.get("id") for pod in pods if pod.get("id") is not None}

        # Remove non-existent pods from managed_pods
        for pod_id in list(managed_pods.keys()):
            if pod_id not in current_pod_ids:
                log(INFO, f"Removing non-existent pod {pod_id} from managed_pods.")
                del managed_pods[pod_id]

        # Update or add pods to managed_pods
        for pod in pods:
            pod_id = pod.get("id")
            if pod_id is None:
                raise ValueError("Pod ID is missing in the response.")
            if pod_id in managed_pods:
                # refresh pod details
                managed_pods[pod_id]["details"] = pod
            else:
                if PODNAMEPREFIX in pod["name"]:
                    # add managed pod
                    status = PodManagedStatus.MANAGED
                else:
                    # add un-managed pod
                    status = PodManagedStatus.UNMANAGED

                managed_pods[pod_id] = {
                    "managed_status": status.value,
                    "details": pod
                }

    except Exception as e:
        log(ERROR, f"Error syncing managed pods: {e}")
#endregion managed pods

if __name__ == "__main__":
    # Example usage
    CREATE = False
    DELETE = False

    if RUNPOD_KEY[0] is None:
        raise ValueError("RUNPOD_KEY environment variable not set.")
    
    success, pods = get_pods()
    if not success:
        raise ValueError("Failed to fetch pods.")
    print(" --- ALL PODS ---")
    print(json.dumps(pods, indent=4))

    if len(pods) > 0:
        pod_id = pods[0]["id"]
        success, pod = get_pods(pod_id=pod_id)
        if not success:
            raise ValueError("Failed to fetch pod details.")
        print("\n --- POD 0 ---")
        print(json.dumps(pod, indent=4))

        if DELETE:
            # Stop the pod
            pod_id = pod["id"]
            print(f"Stopping pod with ID: {pod_id}")
            success, stop_response = stop_pod(pod_id=pod_id)
            if not success:
                raise ValueError("Failed to stop pod.")
            print("\n --- STOP POD ---")
            print(f"Stop response: {stop_response}")

            # wait until status is "desiredStatus" is "STOPPED"
            is_running = True
            while is_running:
                success, pod = get_pods(pod_id=pod_id)
                if not success:
                    raise ValueError("Failed to fetch pods.")
                if pod["desiredStatus"] == "EXITED":
                    is_running = False
                    break
                else:
                    print(f"Pod {pod_id} is still running. Current status: {pod['status']}")
                    time.sleep(5)

            print(f"Deleting pod with ID: {pod_id}")
            success, delete_response = delete_pod(pod_id=pod_id)
            if not success:
                raise ValueError("Failed to delete pod.")
            print("\n --- DELETE POD ---")
            print(f"Delete response: {success}")

    else:
        print("No pods found.")

        if CREATE:
            dockerStartCommand = "--host 0.0.0.0 --port 8000 --model meta-llama/Meta-Llama-3.1-8B-Instruct --dtype bfloat16 --enforce-eager --gpu-memory-utilization 0.95 --api-key sk-IrR7Bwxtin0haWagUnPrBgq5PurnUz86 --max-model-len 8128"
            gpuType = "NVIDIA GeForce RTX 3090"

            # Create a new pod
            print("\nCreating a new pod...")
            success, create_response = create_pod("my pod", gpuType, dockerStartCommand, 20)
            print(json.dumps(create_response, indent=4))
            if not success:
                raise ValueError("Failed to create pod.")
            print("\n --- CREATED POD ---")
            print(f"Create response: {create_response}")
