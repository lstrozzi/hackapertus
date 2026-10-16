import threading

LOG_FILE_LOADING = False
file_loading_lock = threading.Lock()

def log_start_file(source_file:str) -> None:
    if LOG_FILE_LOADING:
        with file_loading_lock:
            print(f"--- LOADING FILE START: {source_file} ---")
    return

def log_end_file(source_file:str) -> None:
    if LOG_FILE_LOADING:
        with file_loading_lock:
            print(f"--- LOADING FILE END:   {source_file} ---")
    return

def log_print(message:str) -> None:
    if LOG_FILE_LOADING:
        with file_loading_lock:
            print(message)
    return
