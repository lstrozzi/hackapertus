# Shutdown.py

import threading
from typing import List

shutdown_event: List[threading.Event] = []
shutdown_event.append(threading.Event())
