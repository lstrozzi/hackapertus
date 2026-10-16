import threading

# Dictionary to hold locks for resources
resource_locks = {}
resource_locks_lock = threading.Lock()

def get_lock(resource_class:str, resource_id:str) -> threading.Lock:
    if resource_class is None:
        raise ValueError("resource_class and resource_id cannot be None")
    if not isinstance(resource_class, str) or (resource_id is not None and not isinstance(resource_id, str)):
        raise ValueError("resource_class and resource_id must be strings")
    
    if resource_id is None or resource_id == "":
        id = f"{resource_class}__GLOBAL"
    else:
        id = f"{resource_class}__{resource_id}"

    with resource_locks_lock:
        if id not in resource_locks:
            resource_locks[id] = threading.Lock()
        return resource_locks[id]
