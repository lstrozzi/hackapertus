from enum import Enum

class GoalStatus(Enum):
    NOT_STARTED = "Not Started"
    RUN_EVERY_TIME = "Run Every Time"
    IN_PROGRESS = "In Progress"
    CHILDREN_IN_PROGRESS = "Children in Progress"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"

class GoalAsyncStatus(Enum):
    SYNC = "SYNC"
    ASYNC_REQUESTED = "ASYNC_REQUESTED"
    ASYNC_IN_PROGRESS = "ASYNC_IN_PROGRESS"
    ASYNC_COMPLETE = "ASYNC_COMPLETE"
    ASYNC_FAILED = "ASYNC_FAILED"
    ASYNC_ABORTED = "ASYNC_ABORTED"

class GoalTarget(Enum):
    UNDEFINED = "Undefined"
    ONLY_SUBGOALS = "Only subgoals"
    CONTEXTUALIZER = "Contextualizer"
    AGENT = "Agent"
    CONSOLIDATOR = "Consolidator"
    NOOP = "No Operation"
    DEBUG = "Debug"

class GoalValidation(Enum):
    NOT_VALIDATED = "NOT_VALIDATED"
    VALIDATED_OK = "VALIDATED_OK"
    VALIDATED_ISSUES = "VALIDATED_ISSUES"
