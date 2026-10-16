from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)


import time
from src.utils.log import log, DEBUG, INFO, WARNING, ERROR

# Performance counters
total_performance_time_milli: list[float] = [1]      # It must be a mutable container, else it will be treated as a local variable
total_performance_time_milli[0] = 0.0
last_decorated_function_time_milli: float = 0.0


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        global total_performance_time_milli, last_decorated_function_time_milli

        if func.__name__ is not None:
            log(DEBUG, f"timer_decorator: function '{func.__name__}' started...")
        else:
            log(DEBUG, "timer_decorator started...")

        # Measure the time taken by the function
        start_time: float = time.perf_counter()
        result = func(*args, **kwargs)
        end_time: float = time.perf_counter()

        # Calculate elapsed time in milliseconds
        last_decorated_function_time_milli = (end_time - start_time) * 1000  # convert to milliseconds
        log(DEBUG, f"...elapsed time: {last_decorated_function_time_milli} ms")

        # Update the total external time counter
        total_performance_time_milli[0] += last_decorated_function_time_milli
        return result
    
    return wrapper

log_end_file(__file__)
