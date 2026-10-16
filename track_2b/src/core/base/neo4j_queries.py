from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import os
import time
import inspect
import hashlib
import threading
from typing import Optional

from neo4j import GraphDatabase, Driver, Result
from neo4j.exceptions import SessionExpired as Neo4jSessionExpired, ServiceUnavailable as Neo4jServiceUnavailable

from src.utils.log import log, INFO, WARNING, ERROR, PERF, find_goal_in_stack_frame
from src.utils.system import get_os_type

# Track which deprecation warnings have been shown for which queries to avoid duplicates
_shown_deprecation_warnings: dict = {}
_deprecation_warning_lock = threading.Lock()


class Neo4jQueryResult:
    """Return value from execute_neo4j_query().

    Wraps the result rows with additional metadata (elapsed time, caller info).
    Supports list-style iteration, indexing and length checks via delegation to
    `results` so that call-sites that treat the object as a list continue to
    work without change.
    """

    def __init__(
        self,
        results: Optional[list],
        success: bool,
        elapsed_s: float = 0.0,
        caller: str = "",
        goal_id: str = "",
    ):
        self.results: Optional[list] = results
        self.success: bool = success
        self.elapsed_s: float = elapsed_s
        self.caller: str = caller
        self.goal_id: str = goal_id

    # ------------------------------------------------------------------
    # List-style delegation for backward compatibility
    # ------------------------------------------------------------------

    def __bool__(self) -> bool:
        """True when the query succeeded and returned at least one row."""
        return self.results is not None and len(self.results) > 0

    def __iter__(self):
        return iter(self.results if self.results is not None else [])

    def __len__(self) -> int:
        return len(self.results) if self.results is not None else 0

    def __getitem__(self, index):
        return self.results[index]

    def __repr__(self) -> str:  # pragma: no cover
        n = len(self.results) if self.results is not None else "err"
        return f"Neo4jQueryResult(rows={n}, success={self.success}, elapsed={self.elapsed_s:.3f}s)"


# ---------------------------------------------------------------------------
# Driver management
# ---------------------------------------------------------------------------

neo4jdriver: Optional[Driver] = None


def get_neo4j_driver() -> Driver:
    global neo4jdriver

    if neo4jdriver is not None:
        return neo4jdriver

    os_type = get_os_type()
    neo4juri_default = 'bolt://localhost:7687' if os_type != "Linux" else 'bolt://neo4j:7687'
    neo4juri = os.getenv('NEO4J_URI', neo4juri_default)
    if os_type == "Linux":
        neo4juri = neo4juri.replace('127.0.0.1', 'host.docker.internal').replace('localhost', 'host.docker.internal')
    neo4jusername = os.getenv('NEO4J_USERNAME', 'neo4j')
    neo4jpassword = os.getenv('NEO4J_PASSWORD')
    if not neo4jpassword:
        raise RuntimeError("NEO4J_PASSWORD must be set before connecting to Neo4j")
    neo4jdriver = GraphDatabase.driver(neo4juri, auth=(neo4jusername, neo4jpassword))

    log(INFO, f"Connected to Neo4j at {neo4juri} as user '{neo4jusername}'")
    return neo4jdriver


def _reset_neo4j_driver() -> None:
    """Close and discard the cached driver so the next call to get_neo4j_driver() creates a fresh one."""
    global neo4jdriver
    old = neo4jdriver
    neo4jdriver = None
    if old is not None:
        try:
            old.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def execute_neo4j_query(
    query: str,
    parameters: dict = None,
    ignore_exception: bool = False,
    ignore_warnings: bool = False,
) -> Neo4jQueryResult:

    def _sanitize_parameters(obj):
        """Recursively sanitize parameters — replace 'embedding' values with a placeholder to avoid logging huge vectors."""
        if isinstance(obj, dict):
            new = {}
            for k, v in obj.items():
                if 'embedding' in k.lower():
                    new[k] = '[truncated embedding]'
                else:
                    new[k] = _sanitize_parameters(v)
            return new
        elif isinstance(obj, list):
            return [_sanitize_parameters(x) for x in obj]
        else:
            return obj

    def _run_query() -> list:
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run(query, parameters)
            result_list = list(result)      # collect rows eagerly

            # consume summary and surface any server notifications via our logger
            if not ignore_warnings:
                try:
                    summary = result.consume()
                    clean_parameters = _sanitize_parameters(parameters)
                    notifications = getattr(summary, 'notifications', None) or []
                    
                    # Create a hash of the query to track deprecation warnings per query
                    query_hash = hashlib.md5(query.encode()).hexdigest()
                    
                    for n in notifications:
                        try:
                            code = getattr(n, 'code', None)
                            title = getattr(n, 'title', None)
                            desc = getattr(n, 'description', None)
                            sev = getattr(n, 'severity', None)
                            
                            # Check if this is a deprecation warning and if we've already shown it for this query
                            if code == 'Neo.ClientNotification.Statement.FeatureDeprecationWarning':
                                with _deprecation_warning_lock:
                                    if query_hash not in _shown_deprecation_warnings:
                                        _shown_deprecation_warnings[query_hash] = True
                                        log(WARNING, f"Neo4j notification: code={code}, title={title}, severity={sev}, description={desc} - Query: {query}\nParameters: {clean_parameters}")
                            else:
                                # For non-deprecation warnings, always log them
                                if code or title or desc:
                                    log(WARNING, f"Neo4j notification: code={code}, title={title}, severity={sev}, description={desc} - Query: {query}\nParameters: {clean_parameters}")
                                else:
                                    log(WARNING, f"Neo4j notification: {n}\nQuery: {query}\nParameters: {clean_parameters}")
                        except Exception:
                            log(WARNING, f"Neo4j notification: {n}\nQuery: {query}\nParameters: {clean_parameters}")
                except Exception:
                    pass

            return result_list

    # Capture caller info once (first frame outside this file)
    _stack: list = inspect.stack()
    _caller_frame = next((f for f in _stack[1:] if 'neo4j_queries' not in f.filename), _stack[1])
    _caller: str = f"{_caller_frame.function} ({os.path.basename(_caller_frame.filename)})"
    _goal = find_goal_in_stack_frame(_stack)
    _goal_id: str = _goal.get_id_chain_as_str() if _goal and hasattr(_goal, 'get_id_chain_as_str') else ""
    _caller_info: str = f"caller={_caller}" + (f", goal={_goal_id}" if _goal_id else "")
    _t0: float = 0.0
    _t1: float = 0.0

    _rows: Optional[list] = None
    _success: bool = False
    try:
        _max_retries = 2  # retry up to 2 more times = 3 total attempts
        _attempt = 0
        while True:
            try:
                _t0 = time.perf_counter()
                _rows = _run_query()
                _success = True
                break  # success — exit retry loop

            except Exception as conn_err:
                _is_retryable = isinstance(conn_err, (Neo4jSessionExpired, Neo4jServiceUnavailable)) or \
                                "Unable to retrieve routing information" in str(conn_err)
                if not _is_retryable:
                    raise
                _attempt += 1
                if _attempt > _max_retries:
                    raise
                _delay = 0.5 * (2 ** (_attempt - 1))  # 0.5 s, then 1.0 s
                log(WARNING, f"execute_neo4j_query: Neo4j connection lost ({conn_err}), "
                             f"resetting driver and retrying in {_delay:.1f}s "
                             f"(attempt {_attempt}/{_max_retries})...")
                _reset_neo4j_driver()
                time.sleep(_delay)

            finally:
                _t1 = time.perf_counter()
                _elapsed = _t1 - _t0

    except Exception as e:
        try:
            e_as_string = str(e)
            clean_parameters = _sanitize_parameters(parameters)
        except Exception:
            clean_parameters = '[unavailable]'

        uri = os.getenv('NEO4J_URI', None)
        _is_routing_error = "Unable to retrieve routing information" in e_as_string
        if not ignore_exception and ("EquivalentSchemaRuleAlreadyExistsException" not in e_as_string and "There is no such vector schema index" not in e_as_string) and not _is_routing_error:
            log(ERROR, f"execute_neo4j_query caused an exception: {e} from uri {uri}. The query was: {query} and parameters: {clean_parameters}")
        else:
            if "There is no such vector schema index" in e_as_string:
                log(WARNING, f"execute_neo4j_query: The vector schema index does not exist yet. The query was: {query}")
            elif _is_routing_error:
                log(WARNING, f"execute_neo4j_query caused an exception: {e} from uri {uri}. The query was: {query} and parameters: {clean_parameters}")
            elif ignore_exception:
                log(WARNING, f"execute_neo4j_query caused an exception: {e} from uri {uri}. The query was: {query} and parameters: {clean_parameters}")
            else:
                log(ERROR, f"execute_neo4j_query caused an exception: {e} from uri {uri}. The query was: {query} and parameters: {clean_parameters}")

        _rows = None
        _success = False

    finally:
        _elapsed: float = _t1 - _t0
        _rows_str: str = str(len(_rows)) if _rows is not None else "err"
        log(PERF, f"neo4j: {_elapsed:.3f}s, rows={_rows_str}, {_caller_info}")

        if _goal is not None:
            _goal.add_perf_db_ms(_elapsed * 1000)  # accumulate DB duration into the goal's performance tracking

    return Neo4jQueryResult(
        results=_rows,
        success=_success,
        elapsed_s=_elapsed,  # use the measured elapsed time
        caller=_caller,
        goal_id=_goal_id,
    )


log_end_file(__file__)
