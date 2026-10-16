from enum import Enum

class DocumentStatus(Enum):
    READY = "Ready"
    PROCESSING = "Processing"
    OPTIMIZING = "Optimizing"
    NOT_YET_PROCESSED = "NotYetProcessed"
    FAILED = "Failed"
    NEEDS_CATEGORIZATION = "NeedsCategorization"

class KnowledgeGraphStatus(Enum):
    NOT_YET_PROCESSED = "NotYetProcessed"
    FAILED = "Failed"
    READY = "Ready"
    PROCESSING = "Processing"

class DocumentType(Enum):
    PDF = "PDF"
    TEXT = "Text"
    IMAGE = "Image"
    DOCX = "Docx"
    XML = "Xml"
    HTML = "Html"
    CSV = "Csv"
    MD = "MD"
    OTHER = "Other"

class AgentReasoningEnum(Enum):
    NONE = "NONE"           # no internal_thoughts overall, in any circumstances
    MINIMAL = "MINIMAL"     # no internal_thoughts overall, in most circumstances
    FAST = "FAST"           # no internal_thoughts overall, in any circumstances
    SIMPLE = "SIMPLE"       # internal_thoughts are enabled in some circumstances: for AgentPlanner
    MEDIUM = "MEDIUM"       # internal_thoughts are enabled in more circumstances: for AgentPlanner and AgentExecutor
    COMPLEX = "COMPLEX"     # internal_thoughts are enabled overall

class FocusIterationType(Enum):
    DISABLED = "disabled"
    REGEX = "regex"
    CHAPTERS = "chapters"
    ONTOLOGY = "ontology"
    RELATIONSHIP = "ontology-relationship"
    ENTITY = "entity"
    COMMUNITY = "community"
    DATASET = "dataset"

class ReasoningStatus(Enum):
    """Enumeration for division statuses"""
    RS_OPEN = "open"                                # no tasks have been added yet. The instructions must specify how tasks are created.
    RS_IN_PROGRESS = "in progress"                  # tasks have been added, but not all tasks are complete. No instructions expected.
    RS_COMPILING_SUMMARY = "compiling summary"      # all tasks are complete and the results are finalized. The instructions must specify how to create the summary.
    RS_COMPILING_ISSUES = "compiling issues"        # all tasks are complete and the results are finalized. The instructions must specify how to create issues based on the results.
    RS_COMPILING_RECOMMENDATIONS = "compiling recommendations"  # all tasks are complete and the results are finalized. The instructions must specify how to create recommendations based on the results.
    RS_IN_REVIEW = "in review"                      # all tasks are complete, but the results are not yet finalized. The instructions must specify how to review the results, and may specify additional tasks to perform.
    RS_COMPLETE = "complete"                        # nothing more to do

    @staticmethod
    def next_status(current_status):
        """Get the next status in the reasoning workflow"""
        if current_status == ReasoningStatus.RS_OPEN:
            return ReasoningStatus.RS_IN_PROGRESS
        elif current_status == ReasoningStatus.RS_IN_PROGRESS:
            return ReasoningStatus.RS_COMPILING_SUMMARY
        elif current_status == ReasoningStatus.RS_COMPILING_SUMMARY:
            return ReasoningStatus.RS_COMPILING_ISSUES
        elif current_status == ReasoningStatus.RS_COMPILING_ISSUES:
            return ReasoningStatus.RS_COMPILING_RECOMMENDATIONS
        elif current_status == ReasoningStatus.RS_COMPILING_RECOMMENDATIONS:
            return ReasoningStatus.RS_IN_REVIEW
        elif current_status == ReasoningStatus.RS_IN_REVIEW:
            return ReasoningStatus.RS_COMPLETE
        else:
            return current_status  # If already COMPLETE or unknown, stay the same

class TaskStatus(Enum):
    """Enumeration for reasoning task"""
    TS_OPEN = "open"                                # task has been added, but not yet started.
    TS_IN_PROGRESS = "in progress"                  # task has been added, and being executed.
    TS_READY = "ready"                              # task has completed, but the results are not yet used to generate the summary.
    TS_COMPLETE = "complete"                        # nothing more to do, the task results have been used to generate the summary. It's considered as archived and the results won't be used further, to spare conversation buffer tokens.

class IssueStatus(Enum):
    """Enumeration for issue statuses"""
    OPEN = "open"
    IN_PROGRESS = "in progress"
    CLOSED = "closed"

class IssueSeverity(Enum):
    """Enumeration for issue severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class DivisionScope(Enum):
    """Enumeration for division scope levels"""
    PRIVATE = "private"     # default
    PUBLIC = "public"

class DivisionExecution(Enum):
    """Enumeration for division execution levels"""
    SEQUENTIAL = "sequential"     # default
    PARALLEL = "parallel"

class DivisionAction(Enum):
    """Enumeration for division action attribute (LoV)"""
    NONE = "none"
    SET_PROMPT_FOR_NEXT_DIVISION = "set prompt for next division"
    SET_FOCUS_CONTENT_FOR_NEXT_DIVISION = "set focus content for next division"

class DivisionVisibility(Enum):
    """Enumeration for division visibility"""
    VISIBLE = "visible"     # default
    HIDDEN = "hidden"

class HintRecommendation(Enum):
    """Analysis outcome of a trainer Hint. What an outcome implies for injection, escalation, and merging is defined in the policy table in src/core/base/hints.py."""
    NONE = None
    REDUNDANT = "Redundant"
    TRAINER_INSPECTION = "TrainerInspection"
    CONFLICT_WITH_AGENT_INSTRUCTIONS = "ConflictWithAgentInstructions"
    CONFLICT_UNRECOVERABLE_WITH_HINT = "ConflictUnrecoverableWithHint"
    PARTIAL_OVERLAP_WITH_HINT = "PartialOverlapWithHint"
    NEW_VALID_HINT_NOT_OVERLAPPING = "NewValidHintNotOverlapping"
    PROMOTE_TO_AGENT_INSTRUCTION = "PromoteToAgentInstruction"
    REJECTED_POLICY = "RejectedPolicy"
    INVALID = "Invalid"
