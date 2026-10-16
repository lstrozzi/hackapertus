from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import json
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config.constants import MAX_PARALLELISM
from src.core.base.enums import AgentReasoningEnum, HintRecommendation
from src.utils.log import log, log_init, log_exit, DEBUG, INFO, WARNING, ERROR, FATAL
from src.core.base.Shutdown import shutdown_event

# Global controls
ENABLE_HINT_REDUNDANCY_CHECK: bool = True


HINT_CONSOLIDATION_SKILL_ID: str = "HintConsolidation"
HINT_CONSOLIDATION_REASONING_LEVEL: AgentReasoningEnum = AgentReasoningEnum.MEDIUM

# The Hint-versus-Hint comparison receives only the most similar candidates, selected by the stored Hint embeddings,
# so that the prompt stays bounded no matter how large the Hint store grows.
HINT_SIMILARITY_CANDIDATE_LIMIT: int = 5
HINT_SIMILARITY_MIN_SCORE: float = 0.75

# Maps the relationship value answered by the Hint-versus-Hint comparison onto the analysis recommendation it produces.
HINT_RELATIONSHIP_RECOMMENDATIONS: Dict[str, HintRecommendation] = {
    "unrelated": HintRecommendation.NONE,
    "partial_overlap": HintRecommendation.PARTIAL_OVERLAP_WITH_HINT,
    "unrecoverable_conflict": HintRecommendation.CONFLICT_UNRECOVERABLE_WITH_HINT,
}

if TYPE_CHECKING:
    from src.services.hints_service import HintsService
    from src.core.llm.manager import LlmManager


@dataclass(frozen=True)
class HintRedundancyResult:
    """Outcome of the redundancy check: the two model verdicts and the explanation to persist.
    Iterating yields (is_redundant, analysis_results), so callers may unpack the historical two-tuple."""
    is_redundant: bool
    adds_new_information: bool
    analysis_results: Optional[str]

    def __iter__(self):
        return iter((self.is_redundant, self.analysis_results))

# System prompt for the redundancy check performed when consolidating a trainer Hint.
HINT_REDUNDANCY_SYSTEM_PROMPT: str = """You are the consolidation analyst for the Thalabus trainer Hints system.

# Redundancy check instructions for the Thalabus trainer Hints system.
Trainer Hints are corrections and instructions that users give to improve how this system's AI agents behave. A Hint may contain rules, descriptions, and/or examples.
Your only task is to evaluate redundancy: decide whether a newly proposed Hint is already FULLY described by one or more of the existing consolidated Hints.

A new Hint is redundant ONLY IF all of its rules, descriptions, and examples are already fully covered by the existing Hints AND the new Hint adds no relevant new information. Even when a similar Hint exists, the new Hint is NOT redundant as soon as it adds any relevant information, such as an additional rule, a nuance, a restriction, or an example. Consider the combination of the existing Hints as a whole: together they may already fully cover the new Hint.

You will receive: the list of existing consolidated Hints in a system message, then a user message that describes the new Hint to evaluate.

Answer 'adds_new_information': true ONLY when the new Hint adds at least one genuine rule, description, or example that no existing Hint covers; answer it false when the contribution is absent or unclear, so that a trainer can inspect the Hint.

Respond with a JSON object containing exactly three fields: 'redundant', a boolean that is true ONLY when the new Hint is fully redundant; 'adds_new_information', a boolean that is true ONLY when the new Hint genuinely adds new information as described above; and 'reason', a concise explanation of your evaluation. They must appear in this order:
1. reason
2. redundant
3. adds_new_information

Format 'reason' as readable Markdown with exactly these headings in this order: '### Existing coverage', '### Comparison', and '### Conclusion'. Keep each section concise. In the conclusion, state whether the new Hint is redundant or what relevant information it adds.
Because 'reason' is inside a JSON string, encode each line break as the two-character JSON escape sequence \\n (and blank lines as \\n\\n). Do not put literal line breaks inside the JSON string.

# Example
{"reason":"### Existing coverage\\nHint 5 already requires plan_rag_assistant_and_answer for factual questions about content in an existing dataset.\\n\\n### Comparison\\nHint 5 also covers using the matching dataset when available and unrestricted RAG when none matches; the new Hint adds no further rule.\\n\\n### Conclusion\\nThe new Hint is fully covered by Hint 5 and is redundant.","redundant":true,"adds_new_information":false}
"""

# One-shot example for every possible value of the analysis recommendation, so that the model has seen
# each outcome it can produce, and so that the prompt stays aligned with the enumeration in enums.py.
HINT_REDUNDANCY_ONE_SHOT_EXAMPLES: Dict[HintRecommendation, str] = {
    HintRecommendation.NONE:
        "An unanalyzed Hint arrives with 'Needs consolidation: True' and 'Analysis recommendation: None'. "
        "This is the state of the Hint you must evaluate, never an outcome: your answer always settles it. "
        "When the information you were given is not sufficient to evaluate redundancy, answer 'redundant': false so that a trainer can inspect the Hint.",
    HintRecommendation.REDUNDANT:
        "Existing Hint: 'When you answer about a case, always cite the paragraph of the document you relied on.' "
        "New Hint: 'Cite the paragraph you relied on when answering a question about a case.' "
        "The new Hint restates the same rule with no addition, so the answer is: "
        "{\"reason\": \"### Existing coverage\\nThe existing Hint requires citing the relied-upon paragraph when answering about a case.\\n\\n### Comparison\\nThe new Hint restates the same rule without adding a nuance or example.\\n\\n### Conclusion\\nThe new Hint is fully covered and is redundant.\", \"redundant\": true}",
    HintRecommendation.TRAINER_INSPECTION:
        "This is the fallback when the evaluation is not conclusive. "
        "Existing Hint: 'When you answer about a case, always cite the paragraph of the document you relied on.' "
        "New Hint: 'When quoting more than one sentence from a document, cite each quoted sentence separately.' "
        "It is not redundant and might add a per-sentence attribution rule, but whether that rule is genuinely new or only an aspect of the existing citation requirement is unclear to you, so the answer is: "
        "{\"reason\": \"### Existing coverage\\nThe existing Hint requires citing the relied-upon paragraph.\\n\\n### Comparison\\nWhether per-sentence attribution for quotations is already covered is not clear from the existing Hints.\\n\\n### Conclusion\\nThe new Hint is not redundant, but its contribution is unclear and should be inspected by a trainer.\", \"redundant\": false, \"adds_new_information\": false}",
    HintRecommendation.CONFLICT_WITH_AGENT_INSTRUCTIONS:
        "A conflict with the agent's own instructions is a different evaluation, performed by its own check, never by this redundancy check. "
        "Never answer in a way that leads to this recommendation here: judge redundancy only. "
        "When the new Hint may contradict an agent instruction, still answer this redundancy check on its own terms and let the separate conflict check report the conflict.",
    HintRecommendation.CONFLICT_UNRECOVERABLE_WITH_HINT:
        "An unrecoverable conflict with another Hint is a different evaluation, performed when Hint-versus-Hint candidates are compared and no merge can reconcile them, never by this redundancy check. "
        "Never answer in a way that leads to this recommendation here: judge redundancy only. "
        "When the new Hint appears to contradict or overlap another Hint, still answer this redundancy check on its own terms and let the Hint-versus-Hint comparison report the conflict.",
    HintRecommendation.PARTIAL_OVERLAP_WITH_HINT:
        "A partial overlap with another Hint is a different evaluation, performed when Hint-versus-Hint candidates are compared for mergeability, never by this redundancy check. "
        "Never answer in a way that leads to this recommendation here: judge redundancy only. "
        "An overlapping Hint is by definition not fully covered, so answering 'redundant': false is consistent with this outcome being decided elsewhere.",
    HintRecommendation.NEW_VALID_HINT_NOT_OVERLAPPING:
        "When the new Hint is not redundant and genuinely adds at least one new rule, description, or example that no existing Hint covers, answer: "
        "{\"reason\": \"### Existing coverage\\nNo existing Hint covers per-paragraph summaries of long hearing transcripts.\\n\\n### Comparison\\nThe new Hint requires such summaries and adds a rule none of the existing Hints contains.\\n\\n### Conclusion\\nThe new Hint is valid, adds new information, and is not redundant.\", \"redundant\": false, \"adds_new_information\": true}",
    HintRecommendation.PROMOTE_TO_AGENT_INSTRUCTION:
        "Promoting a Hint to a permanent agent instruction is a different evaluation, performed when a non-conflicting Hint reveals a broader prompt issue, never by this redundancy check. "
        "Never answer in a way that leads to this recommendation here: judge redundancy only.",
    HintRecommendation.REJECTED_POLICY:
        "Rejecting a Hint for a policy or safety violation is a different evaluation, performed by its own guard check before this one, never by this redundancy check. "
        "Never answer in a way that leads to this recommendation here: judge redundancy only.",
    HintRecommendation.INVALID:
        "Declaring a Hint structurally invalid is a code-level validation, performed before any LLM check, never by this redundancy check. "
        "Never answer in a way that leads to this recommendation here: judge redundancy only.",
}

# System prompt for evaluating conflicts between trainer Hints and agent instructions
HINT_CONFLICT_SYSTEM_PROMPT: str = """You are the agent-instruction conflict analyst for the Thalabus trainer Hints system.

Your only task is to determine whether the new Hint conflicts with the permanent agent instructions supplied for the Hint's target agent or agents.

A conflict exists when the Hint requires something an agent instruction forbids, forbids something an instruction requires, or otherwise makes the supplied instructions impossible to follow together. Compare the Hint ONLY with the supplied agent instructions identified by their XML IDs: <SYSTEM_MESSAGE id="AgentAssistant/DEFAULT"> for the main agent system prompt, <INSTRUCTIONS id="AgentPlanner/DocumentManagement"> for the named instructions the agent has configured for itself, and <SKILL id="DataAccess/DatabaseReader"> for the permanent descriptions of the skills (=tools) an agent can use. A skill description states what the skill does and how it must be used: a Hint conflicts with a skill when it demands behavior the description forbids, or instructs to use the skill in a way the description rules out.

Do NOT compare the Hint with other Hints, consolidated or otherwise. Do NOT infer instructions that were not supplied. Similar subject matter, added detail, or a different recommendation is not by itself a conflict. Report a conflict only when the two directions cannot both be followed.

Respond with a JSON object containing exactly two fields, in this order: 'reason' (a concise Markdown explanation) and 'conflict' (a boolean). The reason must identify the conflicting instruction by its supplied XML ID, quote or accurately summarize both directions, and conclude whether they can be followed together. If there is no conflict, explain why the directions are compatible.

# Example
{"reason":"### Agent instruction\n<SYSTEM_MESSAGE id=\"AgentAssistant/DEFAULT\"> says AgentAssistant must never call tools directly.\n\n### Hint comparison\nThe Hint requires AgentAssistant to call DocumentFetcher directly, which the instruction forbids.\n\n### Conclusion\nThese directions cannot both be followed; this is a conflict with agent instructions.","conflict":true}
"""

# Example JSON object for a conflict between a Hint and agent instructions
HINT_CONFLICT_JSON_EXAMPLE: Dict[str, Any] = {
    "reason": "### Agent instruction\n<SYSTEM_MESSAGE id=\"AgentAssistant/DEFAULT\"> says AgentAssistant must never call tools directly.\n\n### Hint comparison\nThe Hint requires AgentAssistant to call DocumentFetcher directly, which the instruction forbids.\n\n### Conclusion\nThese directions cannot both be followed; this is a conflict with agent instructions.",
    "conflict": True,
}

# System prompt for the usage-policy guard performed before any other analysis of a trainer Hint
HINT_POLICY_SYSTEM_PROMPT: str = """You are the usage-policy guard for the Thalabus trainer Hints system.

Trainer Hints are corrections and instructions that users give to improve how this system's AI agents behave. Your only task is to decide whether following the Hint would make an agent break a rule it must never break. Rejecting is not about quality, wording, or taste.

Reject the Hint ONLY when, followed as written, it would make an agent do at least one of the following:
- reveal, exfiltrate, or access another user's or another tenant's data beyond what access control grants it;
- bypass, disable, or instruct it to ignore authentication, authorization, or access-control rules;
- fabricate evidence or citations, or intentionally mislead a court, an authority, or an authorized reviewer;
- hide, delete, or alter evidence or records so that the truth of what happened can no longer be established;
- produce content that attacks, demeans, or dehumanizes a person or a group.

Improving relevance, planning, formatting, tone, or how a tool gets used is never a policy violation, even when you disagree with the approach. When you are unsure, answer 'rejected': false: a Hint is lost when it is wrongly rejected, and a trainer can always inspect one that passes.

Do NOT compare the Hint with other Hints or with agent instructions; separate checks own those evaluations.

Respond with a JSON object containing exactly two fields, in this order: 'reason' (a concise Markdown explanation) and 'rejected' (a boolean). The reason must name the rule the Hint would break and quote or accurately summarize the part of the Hint that breaks it, or state which permitted form of improvement the Hint is.

# Example
{"reason":"### Requested behavior\\nThe Hint instructs the agent to include the full content of every dataset of the other tenants in its answer.\\n\\n### Policy comparison\\nFollowing it would disclose another tenant's data, which access control does not grant and the Hint does not authorize.\\n\\n### Conclusion\\nThe Hint would make the agent break a rule it must never break, so it is rejected.","rejected":true}
"""

# JSON schema for the usage-policy guard
HINT_POLICY_JSON_SCHEMA: Dict[str, Any] = {
    "name": "hint_policy_violation_response",
    "schema": {
        "type": "object",
        "description": "Evaluation of whether following a trainer Hint would make an agent break a rule it must never break.",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Concise Markdown explanation with the headings '### Requested behavior', '### Policy comparison', and '### Conclusion'. Name the rule the Hint would break, or state which permitted form of improvement the Hint is.",
            },
            "rejected": {
                "type": "boolean",
                "description": "True only when following the Hint would make an agent break a rule it must never break.",
            },
        },
        "required": ["reason", "rejected"],
        "additionalProperties": False,
    },
    "strict": True,
}

# Example JSON object for a Hint rejected by the usage-policy guard
HINT_POLICY_JSON_EXAMPLE: Dict[str, Any] = {
    "reason": "### Requested behavior\\nThe Hint instructs the agent to include the full content of every dataset of the other tenants in its answer.\\n\\n### Policy comparison\\nFollowing it would disclose another tenant's data, which access control does not grant and the Hint does not authorize.\\n\\n### Conclusion\\nThe Hint would make the agent break a rule it must never break, so it is rejected.",
    "rejected": True,
}

def get_hint_structural_error(hint: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return why a Hint cannot be analyzed at all, or None when it has the shape every later step relies on.
    Only the shape is judged: a validly shaped Hint can still be judged redundant, conflicting, or against policy."""
    error: Optional[str] = None

    if hint is None:
        error = "The Hint could not be found in the database."
    else:
        problems: List[str] = []

        prompt: Any = hint.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            problems.append("it has no prompt text holding the requested correction")

        category: Any = hint.get("category")
        if not isinstance(category, str) or not category.strip():
            problems.append("it has no category")

        agent_hints: Any = hint.get("agent_hints")
        has_target_agent: bool = isinstance(agent_hints, dict) and any(
            isinstance(agent_name, str)
            and agent_name.strip()
            and isinstance(hint_text, str)
            and hint_text.strip()
            for agent_name, hint_text in agent_hints.items()
        )
        if not has_target_agent:
            problems.append("it names no target agent with a hint text for it")

        if problems:
            error = "The Hint is unusable because " + "; ".join(problems) + "."

    return error

# JSON schema for the conflict response between a Hint and agent instructions
HINT_CONFLICT_JSON_SCHEMA: Dict[str, Any] = {
    "name": "hint_agent_instruction_conflict_response",
    "schema": {
        "type": "object",
        "description": "Evaluation of whether a trainer Hint contradicts the permanent instructions of its target agent.",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Concise Markdown explanation with the headings '### Agent instruction', '### Hint comparison', and '### Conclusion'. Identify the conflicting instruction by its supplied XML ID and explain whether both directions can be followed together.",
            },
            "conflict": {
                "type": "boolean",
                "description": "True only when the Hint and at least one supplied agent instruction cannot both be followed.",
            },
        },
        "required": ["reason", "conflict"],
        "additionalProperties": False,
    },
    "strict": True,
}

# System prompt for the Hint-versus-Hint comparison performed when consolidating a trainer Hint.
HINT_VS_HINT_SYSTEM_PROMPT: str = """You are the Hint-versus-Hint analyst for the Thalabus trainer Hints system.

Trainer Hints are corrections and instructions that users give to improve how this system's AI agents behave. Hints are dynamic guidance, injected on demand into an agent's context when semantically relevant. Your only task is to evaluate the relationship between one new Hint and a small selection of the most similar existing Hints, given to you as candidates.

Evaluate the relationship between the new Hint and the candidates, considered one by one and in combination:
- 'unrecoverable_conflict': the new Hint and at least one candidate require contradictory behavior for an agent that could receive both — same or overlapping scope, and one forbids or requires what the other requires or forbids — such that no merge of their content can reconcile them. Two Hints only conflict when an agent could realistically be steered by both at once: Hints targeting entirely different decision situations with no overlapping scope are unrelated, even when their wording shares a topic.
- 'partial_overlap': the new Hint and at least one candidate cover the same case and share some rules, descriptions, or examples, but each also contributes useful content the other lacks, and a merge of their content into one consolidated Hint is possible and desirable.
- 'unrelated': no candidate conflicts with the new Hint and none shares a mergeable overlap with it.

Similar subject matter alone is not overlap or conflict. Do NOT compare the new Hint with permanent agent instructions: a separate check already evaluated that; judge only relationships between Hints. When the information given is not sufficient to decide, answer 'unrelated' so that no state is changed on uncertain evidence.

Respond with a JSON object containing exactly three fields, in this order: 'reason' (a concise Markdown explanation), 'relationship' (one of 'unrelated', 'partial_overlap', 'unrecoverable_conflict'), and 'candidate_hint_ids' (the IDs of the candidates that drove the answer; empty when unrelated).

Format 'reason' as readable Markdown with exactly these headings in this order: '### New Hint', '### Candidate comparison', and '### Conclusion'. Name each relevant candidate by its Hint ID, state the shared ground or the contradiction, and conclude the relationship.
Because 'reason' is inside a JSON string, encode each line break as the two-character JSON escape sequence \\n (and blank lines as \\n\\n). Do not put literal line breaks inside the JSON string.
List in 'candidate_hint_ids' only IDs of candidates supplied to you, exactly as given.

# Example
{"reason":"### New Hint\\nThe new Hint requires answering in the user's language and citing document IDs for every RAG answer.\\n\\n### Candidate comparison\\nCandidate hints-014 already requires RAG answers to cite document IDs, but says nothing about the answer language; the new Hint adds that rule and repeats the citation rule.\\n\\n### Conclusion\\nThe two Hints cover the same case with shared and complementary content: a partial overlap that a merge can reconcile.","relationship":"partial_overlap","candidate_hint_ids":["hints-014"]}
"""

# One-shot example given to the LLM for the Hint-versus-Hint comparison, as `response_format`.
HINT_VS_HINT_JSON_EXAMPLE: Dict[str, Any] = {
    "reason": "### New Hint\nThe new Hint requires summarizing long hearing transcripts per paragraph.\n\n### Candidate comparison\nNo supplied candidate covers per-paragraph summaries of hearing transcripts, and none contradicts the new requirement.\n\n### Conclusion\nThe new Hint is unrelated to the supplied candidates.",
    "relationship": "unrelated",
    "candidate_hint_ids": [],
}

# JSON schema for the Hint-versus-Hint comparison response.
HINT_VS_HINT_JSON_SCHEMA: Dict[str, Any] = {
    "name": "hint_vs_hint_response",
    "schema": {
        "type": "object",
        "description": "The relationship of a new trainer Hint to the supplied similar Hint candidates.",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Concise Markdown explanation with the headings '### New Hint', '### Candidate comparison', and '### Conclusion'. Name each relevant candidate by its Hint ID and explain the relationship. Since this is a JSON string, encode line breaks with the JSON \\n escape.",
            },
            "relationship": {
                "type": "string",
                "enum": ["unrelated", "partial_overlap", "unrecoverable_conflict"],
                "description": "'unrecoverable_conflict' only when no merge can reconcile the contradiction; 'partial_overlap' when a merge is possible and desirable; otherwise 'unrelated'.",
            },
            "candidate_hint_ids": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "The ID of one supplied candidate that drove the answer.",
                },
                "description": "The IDs of the candidates that drove the relationship; an empty array when the answer is 'unrelated'.",
            },
        },
        "required": ["reason", "relationship", "candidate_hint_ids"],
        "additionalProperties": False,
    },
    "strict": True,
}

@dataclass(frozen=True)
class HintRelationshipResult:
    """Outcome of the Hint-versus-Hint comparison: the relationship the model confirmed, the candidates it named, and the explanation to persist."""
    recommendation: HintRecommendation
    candidate_hint_ids: List[str]
    analysis_results: Optional[str]

# System prompt for the stage-2 merge performed on a Hint whose Hint-versus-Hint comparison confirmed a partial overlap.
HINT_MERGE_SYSTEM_PROMPT: str = """You are the Hint consolidation writer for the Thalabus trainer Hints system.

Trainer Hints are corrections and instructions that users give to improve how this system's AI agents behave. A Hint stores, per target agent, the guidance text injected into that agent when the Hint is selected. The Hint-versus-Hint comparison already decided that the Hint to consolidate and one or more candidate Hints partially overlap: they cover the same case, share some content, and each contributes useful content the others lack. Your only task is to write the consolidated guidance.

You will receive: the Hint to consolidate and its candidate Hints in a system message, each showing its per-agent guidance texts, then a user message asking for the merge.

Merge the guidance of all supplied Hints into one consolidated set of guidance texts:
- Produce exactly one merged guidance text for every agent key that any supplied Hint has a guidance text for, so that no injected guidance is lost when the candidate Hints stop being injected.
- Keep every rule, description, nuance, and example that any source Hint contributes: a merge may reword, reorder, and combine, but must never drop a distinct piece of guidance, weaken a requirement, or add guidance none of the sources contains.
- Express duplicated content once, in the clearest shared wording; resolve duplicate coverage without changing its meaning.
- Keep the guidance addressed to the same agent it was written for; use the agent key exactly as supplied.
- Keep each merged text in the language of the source texts for that agent.

Respond with a JSON object containing exactly two fields, in this order: 'reason' (a concise Markdown explanation) and 'merged_hints' (an array with one entry per agent key covered). Each entry contains exactly three fields, in this order: 'agent' (the agent key exactly as supplied), 'merged_hint' (the consolidated guidance text for that agent), and 'source_hint_ids' (the IDs of the supplied Hints this entry's content comes from, including the Hint being consolidated itself).

Format 'reason' as readable Markdown with exactly these headings in this order: '### Sources', '### Merge', and '### Conclusion'. Name the consolidated Hint and every candidate by its Hint ID, state how shared content was unified and which contributions were kept where, and conclude that no guidance was lost.
Because 'reason' is inside a JSON string, encode each line break as the two-character JSON escape sequence \\n (and blank lines as \\n\\n). Do not put literal line breaks inside the JSON string.

# Example
{"reason":"### Sources\\nHint hints-013 (the Hint to consolidate) and candidate hints-014 both require citing document IDs in RAG answers.\\n\\n### Merge\\nThe citation requirement is stated once in combined wording; the answer-language rule from hints-013 and the paragraph-level detail from hints-014 are kept.\\n\\n### Conclusion\\nThe merged guidance covers both agents fully and no guidance was lost.","merged_hints":[{"agent":"AgentAssistant","merged_hint":"When answering from a dataset, cite the document IDs you relied on, cite each paragraph separately, and answer in the user's language.","source_hint_ids":["hints-013","hints-014"]}]}
"""

# One-shot example given to the LLM for the stage-2 merge, as `response_format`.
HINT_MERGE_JSON_EXAMPLE: Dict[str, Any] = {
    "reason": "### Sources\nHint hints-013 (the Hint to consolidate) and candidate hints-014 both require citing sources in RAG answers.\n\n### Merge\nThe citation requirement is stated once in combined wording, and the language rule and paragraph detail are kept.\n\n### Conclusion\nThe merged guidance covers both agents fully and no guidance was lost.",
    "merged_hints": [
        {
            "agent": "AgentAssistant",
            "merged_hint": "When answering from a dataset, cite the document IDs you relied on, cite each paragraph separately, and answer in the user's language.",
            "source_hint_ids": ["hints-013", "hints-014"],
        }
    ],
}

# JSON schema for the stage-2 merge response.
HINT_MERGE_JSON_SCHEMA: Dict[str, Any] = {
    "name": "hint_merge_response",
    "schema": {
        "type": "object",
        "description": "The consolidated guidance writing of a Hint merge, one entry per agent key covered.",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Concise Markdown explanation with the headings '### Sources', '### Merge', and '### Conclusion'. Name the supplied Hints by ID and conclude that no guidance was lost. Since this is a JSON string, encode line breaks with the JSON \\n escape.",
            },
            "merged_hints": {
                "type": "array",
                "description": "One entry for every agent key that any supplied Hint has guidance for.",
                "items": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "The agent key exactly as supplied to you.",
                        },
                        "merged_hint": {
                            "type": "string",
                            "description": "The consolidated guidance text for this agent, keeping every rule, nuance, and example the supplied Hints contribute for it.",
                        },
                        "source_hint_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "The ID of one supplied Hint this entry's content comes from.",
                            },
                            "description": "The IDs of the supplied Hints this entry's content comes from, including the Hint being consolidated itself.",
                        },
                    },
                    "required": ["agent", "merged_hint", "source_hint_ids"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["reason", "merged_hints"],
        "additionalProperties": False,
    },
    "strict": True,
}

@dataclass(frozen=True)
class HintConsolidationPolicy:
    """What one analysis recommendation implies: may the Hint be injected, does a human engineer have to follow up, does the Hint have to be merged."""
    exclude_injection: bool
    requires_human_engineer: bool
    requires_hints_merge: bool

# Single source of truth: every HintRecommendation maps to exactly one policy.
# Access it through get_hint_consolidation_policy, never by indexing this table directly.
HINT_CONSOLIDATION_POLICIES: Dict[HintRecommendation, HintConsolidationPolicy] = {
    HintRecommendation.NONE: HintConsolidationPolicy(exclude_injection=False, requires_human_engineer=False, requires_hints_merge=False),
    HintRecommendation.REDUNDANT: HintConsolidationPolicy(exclude_injection=True, requires_human_engineer=False, requires_hints_merge=False),
    HintRecommendation.TRAINER_INSPECTION: HintConsolidationPolicy(exclude_injection=False, requires_human_engineer=False, requires_hints_merge=False),
    HintRecommendation.CONFLICT_WITH_AGENT_INSTRUCTIONS: HintConsolidationPolicy(exclude_injection=True, requires_human_engineer=True, requires_hints_merge=False),
    HintRecommendation.CONFLICT_UNRECOVERABLE_WITH_HINT: HintConsolidationPolicy(exclude_injection=True, requires_human_engineer=False, requires_hints_merge=False),
    HintRecommendation.PARTIAL_OVERLAP_WITH_HINT: HintConsolidationPolicy(exclude_injection=False, requires_human_engineer=False, requires_hints_merge=True),
    HintRecommendation.NEW_VALID_HINT_NOT_OVERLAPPING: HintConsolidationPolicy(exclude_injection=False, requires_human_engineer=False, requires_hints_merge=False),
    HintRecommendation.PROMOTE_TO_AGENT_INSTRUCTION: HintConsolidationPolicy(exclude_injection=False, requires_human_engineer=True, requires_hints_merge=False),
    HintRecommendation.REJECTED_POLICY: HintConsolidationPolicy(exclude_injection=True, requires_human_engineer=True, requires_hints_merge=False),
    HintRecommendation.INVALID: HintConsolidationPolicy(exclude_injection=True, requires_human_engineer=True, requires_hints_merge=False),
}

def get_hint_consolidation_policy(recommendation: HintRecommendation) -> HintConsolidationPolicy:
    """Return the policy of one analysis recommendation; a recommendation missing from the table is a programming error, never a silent default."""
    policy: HintConsolidationPolicy = HINT_CONSOLIDATION_POLICIES[recommendation]

    return policy

def get_injection_excluded_hint_recommendations() -> List[HintRecommendation]:
    """Return every analysis recommendation whose Hints must never be injected into an agent's context."""
    excluded: List[HintRecommendation] = [
        recommendation
        for recommendation, policy in HINT_CONSOLIDATION_POLICIES.items()
        if policy.exclude_injection
    ]

    return excluded


def _build_hint_redundancy_examples() -> str:
    """Render the one-shot examples section of the redundancy system prompt, one example per analysis recommendation."""
    section: str = "\nThe analysis recommendation that your answer leads to, with one example for each possible value:\n\n"
    for recommendation, example in HINT_REDUNDANCY_ONE_SHOT_EXAMPLES.items():
        recommendation_name: str = recommendation.value if recommendation.value is not None else "None"
        section += f"### Example for the analysis recommendation '{recommendation_name}'\n{example}\n\n"

    return section.rstrip()

HINT_REDUNDANCY_SYSTEM_PROMPT += _build_hint_redundancy_examples()

# One-shot example given to the LLM for the redundancy check, as `response_format`.
# This serves two purposes: it shows the model the exact expected shape (a concrete example is
# the strongest mitigation against malformed JSON), and it demonstrates the required key order
# (`reason` before `redundant`), which a JSON schema cannot express.
HINT_REDUNDANCY_JSON_EXAMPLE: Dict[str, Any] = {
    "reason": "### Existing coverage\nHint 'hints-001' requires the agent to answer in the user's language and cite document IDs.\n\n### Comparison\nThe new Hint adds no rule, nuance, or example beyond those requirements.\n\n### Conclusion\nThe new Hint is fully covered and is redundant.",
    "redundant": True,
    "adds_new_information": False,
}

# Structured output requested from the LLM for the redundancy check, in the "Format A" shape
# expected by LlmInterface._prepare_response_format: {"name": ..., "schema": ..., "strict": ...}.
# Every property carries a description, and `required` lists every property without exception,
# as strict JSON schema mode demands.
HINT_REDUNDANCY_JSON_SCHEMA: Dict[str, Any] = {
    "name": "hint_redundancy_response",
    "schema": {
        "type": "object",
        "description": "The evaluation of whether a newly proposed trainer Hint is already fully described by the existing consolidated Hints.",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Concise Markdown explanation using the headings '### Existing coverage', '### Comparison', and '### Conclusion' in that order. Since this is a JSON string, encode line breaks with the JSON \\n escape. Name the covering Hints or state which relevant rule, nuance, restriction, or example the new Hint adds.",
            },
            "redundant": {
                "type": "boolean",
                "description": "True only if the existing Hints, taken together, already contain all the rules, descriptions, and examples of the new Hint and the new Hint adds no relevant new information.",
            },
            "adds_new_information": {
                "type": "boolean",
                "description": "True only if the new Hint adds at least one genuine rule, description, or example that no existing Hint covers; false when it adds nothing or when its contribution is unclear.",
            },
        },
        "required": ["reason", "redundant", "adds_new_information"],
        "additionalProperties": False,
    },
    "strict": True,
}

# Global service instance
_hints_service: Optional["HintsService"] = None

# Cache of the skillset configurations (the skillsets are static for the lifetime of the server process).
_skill_configs_cache: Optional[Dict[str, Dict[str, Any]]] = None

def _load_skill_configs() -> Dict[str, Dict[str, Any]]:
    """Load the metadata of every configured skill at most once for this process."""
    global _skill_configs_cache
    if _skill_configs_cache is None:
        from src.core.extensions.skillset_management import load_skills_configs

        _skill_configs_cache = load_skills_configs()

    return _skill_configs_cache

def _render_agent_skill_instructions(agent_name: str, llm_manager: 'LlmManager', skip_labels: Optional[Set[str]] = None) -> List[str]:
    """Render the static skill catalog as <SKILL> elements, plus the agent's instructions for each skill exactly as the agent runtime injects them:
    instructions keyed by the skill ID itself first, then instructions for every label in the skill's LLM_Label, each as an <INSTRUCTIONS> element.
    MCP tools are runtime connections without static configuration and are therefore not rendered."""
    rendered: List[str] = []
    seen_labels: Set[str] = set(skip_labels or set())

    for skillset, skills in _load_skill_configs().items():
        for skillid, skill_config in skills.items():
            if not isinstance(skill_config, dict):
                continue
            description: Any = skill_config.get("Description")
            if isinstance(description, str) and description.strip():
                rendered.append(f'<SKILL id="{skillset}/{skillid}">{description.strip()}</SKILL>')

            labels: List[str] = [skillid]
            label: Any = skill_config.get("LLM_Label")
            if isinstance(label, str):
                labels.extend(entry.strip() for entry in label.split(",") if entry.strip())

            for part in labels:
                if part in seen_labels:
                    continue

                seen_labels.add(part)
                instruction_text: Any = llm_manager.get_system_prompt(agent=None, agent_name=agent_name, label=part)
                if isinstance(instruction_text, str) and instruction_text.strip():
                    rendered.append(f'<INSTRUCTIONS id="{agent_name}/{part}">\n{instruction_text.strip()}\n</INSTRUCTIONS>')

    return rendered

def _get_hints_service() -> "HintsService":
    """Get or create the hints service singleton."""
    from src.services.hints_service import HintsService

    global _hints_service
    if _hints_service is None:
        _hints_service = HintsService()
    return _hints_service

def find_hints_without_embeddings() -> Dict[str, Dict[str, str]]:
    """Find all hints that don't have embeddings yet."""
    return _get_hints_service().find_hints_without_embeddings()

def create_hints_vector_indexes() -> None:
    """Create or update the vector indexes for hints in the database."""
    from src.core.llm import EmbeddingsManager

    embeddings_dimension = EmbeddingsManager.instance().get_embeddings_dimension()
    _get_hints_service().create_hints_vector_indexes(embeddings_dimension)

def update_hint(prompt_text: str, agent_hints: Dict[str, str], suppress_debug_logs: bool = False) -> bool:
    """
    Update or create a user hint in the Neo4j database for the given prompt_text and agent_hints.
    Prepares embeddings for the prompt_text and stores agent_hints as a JSON string.
    agent_hints is a Dict[str, str] where key=agent_name, value=hint_text. Use "DEFAULT" as the agent key for hints that apply to all agents.
    """
    from src.core.llm import EmbeddingsManager

    embedding_for_prompt = EmbeddingsManager.instance().get_embeddings(prompt_text)
    if embedding_for_prompt is None:
        log(ERROR, f"The embeddings for the hint '{prompt_text}' could not be prepared.")
        return False
    
    result = _get_hints_service().update_hint(prompt_text, agent_hints, embedding_for_prompt, suppress_debug_logs)
    
    # Create vector indexes after successful update
    if result:
        create_hints_vector_indexes()
    
    return result

def delete_hint(prompt_text: str) -> bool:
    """Delete a hint by its prompt text."""
    return _get_hints_service().delete_hint(prompt_text)

def get_best_hints_for_agent(agent_name: str, user_prompt: str, top_n: int, min_score: float, exclude: Optional[List[HintRecommendation]] = None) -> Optional[List[Dict[str, str]]]:
    """
    Retrieve the most relevant hint from Neo4j based on the agent name and the prompt by vectorizing the user prompt.
    Only return hints where the similarity score is at least min_score (unless min_score is -1, which means no minimum).
    
    Args:
        agent_name: The name of the agent requesting the hint
        user_prompt: The user prompt to find hints for
        top_n: Maximum number of hints to retrieve from Neo4j
        min_score: Minimum similarity score required for a hint to be considered. 
                   Use -1 to disable minimum score requirement.
        exclude: List of analysis recommendations; matching hints are never returned.
                 None (default) applies the hints service's default exclusion (see get_default_excluded_hint_recommendations).
                 An empty list disables the exclusion and returns hints with any recommendation.
                 HintRecommendation.NONE matches hints that have no analysis recommendation.
        
    Returns:
        The best matching hints, ordered best first, each one a dict with the hint 'id'
        and the hint 'text' for the requesting agent, or None if no suitable hint is found
    """
    if user_prompt is None:
        return None

    from src.core.llm import EmbeddingsManager

    embedding = EmbeddingsManager.instance().get_embeddings(user_prompt)
    if embedding is None:
        return None
    
    best_hints: List[Dict[str, str]] = _get_hints_service().get_best_hints_for_agent(agent_name, user_prompt, embedding, top_n, min_score, exclude)

    if best_hints is None:
        return None

    return best_hints

def get_default_excluded_hint_recommendations() -> List[HintRecommendation]:
    """Return the analysis recommendations excluded by default when hints are requested without an explicit exclude list."""
    return _get_hints_service().get_default_excluded_hint_recommendations()

def get_pending_consolidation_hints_for_agent(agent_name: str) -> List[str]:
    """Return all pending hints applicable to this agent without ranking them."""
    if not agent_name:
        return []

    return _get_hints_service().get_pending_consolidation_hints_for_agent(agent_name)

def get_hint_ids_by_recommendation(recommendation: HintRecommendation) -> List[str]:
    """Return Hint IDs matching the requested analysis recommendation."""
    return _get_hints_service().get_hint_ids_by_recommendation(recommendation)

def process_hints_requiring_consolidation(hint_ids: List[str]) -> None:
    """Process hints marked for consolidation in parallel."""
    log(DEBUG, f"Trainer Hints requiring consolidation: {', '.join(hint_ids)}")

    if not hint_ids:
        return

    max_workers: int = min(len(hint_ids), MAX_PARALLELISM)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_hint_id = {
            executor.submit(consolidate_hint, hint_id): hint_id
            for hint_id in hint_ids
        }
        for future in as_completed(future_to_hint_id):
            hint_id: str = future_to_hint_id[future]
            try:
                future.result()

            except Exception as e:
                log(ERROR, f"Failed to process trainer Hint '{hint_id}': {e}")

    return

def _render_hint_for_redundancy_check(hint: Dict[str, Any]) -> str:
    """Render one Hint with all of its stored information, for the redundancy check."""
    rendered: str = f"- Hint ID: `{hint.get('id')}`\n"
    rendered += f"  Prompt: `{hint.get('prompt')}`\n"
    rendered += f"  Category: `{hint.get('category')}`\n"
    rendered += f"  Needs consolidation: {hint.get('needs_consolidation')}\n"
    rendered += f"  Analysis recommendation: `{hint.get('analysis_recommendation')}`\n"

    analysis_results: Optional[str] = hint.get("analysis_results")
    if analysis_results:
        rendered += f"  Analysis results: `{analysis_results}`\n"

    agent_hints: Dict[str, Any] = hint.get("agent_hints") or {}
    if agent_hints:
        rendered += "  Agent hints:\n"
        for agent_name, hint_text in agent_hints.items():
            rendered += f"    - Agent: `{agent_name}`: `{hint_text}`\n"

    llm_in: Optional[str] = hint.get("llm_in")
    if llm_in:
        rendered += "  LLM input context:\n```\n"
        rendered += llm_in
        rendered += "\n```\n"

    return rendered

def consolidate_hint_check_for_redundancy(hint_id: str) -> HintRedundancyResult:
    """
    Ask the LLM synchronously whether the given Hint is already fully described by any of the
    already-consolidated Hints.
    Returns a HintRedundancyResult: redundant only when the model confirms full redundancy,
    adds_new_information only when the model confirms a genuinely new contribution, and the
    explanation of the evaluation to be persisted as the analysis results of the Hint.
    """
    is_redundant: bool = False
    adds_new_information: bool = False
    analysis_results: Optional[str] = None
    hints_service: HintsService = _get_hints_service()
    new_hint: Optional[Dict[str, Any]] = hints_service.get_hint_details(hint_id)

    if new_hint is None:
        log(ERROR, f"Cannot check the redundancy of trainer Hint '{hint_id}': the Hint could not be found.")
        analysis_results = "Redundancy evaluation could not be performed: this Hint is no longer in the database."

    else:
        existing_hints: List[Dict[str, Any]] = hints_service.get_consolidated_hints(exclude_hint_id=hint_id)

        if not existing_hints:
            log(DEBUG, f"Trainer Hint '{hint_id}' is not redundant: there are no other consolidated Hints to compare it with.")
            analysis_results = "Not redundant: there are no other consolidated Hints to compare this Hint with."

        else:
            from src.core.llm import DEFAULT_TEMPERATURE_COLD, LlmManager, MAX_COMPLETION_TOKENS_SMALL

            existing_hints_message: str = "These are all the existing consolidated Hints, each with all of the information stored about it:\n\n"
            for existing_hint in existing_hints:
                existing_hints_message += _render_hint_for_redundancy_check(existing_hint)
                existing_hints_message += "\n"

            new_hint_message: str = (
                "The user just gave the feedback that produced the following new Hint. "
                "Evaluate whether this new Hint is already fully described by one or more of the existing consolidated Hints above.\n\n"
                f"{_render_hint_for_redundancy_check(new_hint)}\n"
                "Remember: the new Hint is redundant ONLY IF the existing Hints already contain all of its rules, descriptions, and examples, "
                "and the new Hint adds no relevant new information. Reply with the requested JSON object."
            )

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": HINT_REDUNDANCY_SYSTEM_PROMPT},
                {"role": "system", "content": existing_hints_message},
                {"role": "user", "content": new_hint_message},
            ]

            llm_manager: LlmManager = LlmManager.instance()
            model_configuration_id: Optional[str] = llm_manager.get_model_by_skill_id(HINT_CONSOLIDATION_SKILL_ID)
            llm_response = llm_manager.chat_completions(
                messages=messages,
                model_input=model_configuration_id,
                max_completion_tokens=MAX_COMPLETION_TOKENS_SMALL,
                temperature=DEFAULT_TEMPERATURE_COLD,
                response_format=HINT_REDUNDANCY_JSON_EXAMPLE,
                structured_output=HINT_REDUNDANCY_JSON_SCHEMA,
                maximum_reasoning_required=HINT_CONSOLIDATION_REASONING_LEVEL
            )

            redundancy_evaluation: Optional[Dict[str, Any]] = None
            if llm_response and llm_response.response:
                if isinstance(llm_response.response, dict):
                    redundancy_evaluation = llm_response.response
                elif isinstance(llm_response.response, str):
                    try:
                        redundancy_evaluation = json.loads(llm_response.response)

                    except json.JSONDecodeError:
                        log(WARNING, f"The LLM redundancy check for trainer Hint '{hint_id}' returned text that could not be parsed as JSON.")

            if redundancy_evaluation is None:
                log(WARNING, f"The LLM redundancy check for trainer Hint '{hint_id}' returned no usable answer: the Hint is kept for trainer inspection.")
                analysis_results = "Redundancy evaluation did not produce a usable answer: the Hint is kept for trainer inspection."

            else:
                reason: Optional[str] = redundancy_evaluation.get("reason")
                is_redundant = redundancy_evaluation.get("redundant") is True
                adds_new_information = redundancy_evaluation.get("adds_new_information") is True
                if reason:
                    log(DEBUG, f"Redundancy evaluation for trainer Hint '{hint_id}': {reason}")
                    analysis_results = str(reason)

    return HintRedundancyResult(is_redundant=is_redundant, adds_new_information=adds_new_information, analysis_results=analysis_results)

def consolidate_hint_check_for_conflict_with_agent_instructions(hint_id: str) -> Tuple[Optional[bool], Optional[str]]:
    """Compare a trainer Hint with the active prompt engineering AND the skills of its target agents."""
    has_conflict: Optional[bool] = None
    analysis_results: Optional[str] = None
    hint_details: Optional[Dict[str, Any]] = _get_hints_service().get_hint_details(hint_id)

    if hint_details is None:
        analysis_results = "Agent-instruction conflict evaluation could not be performed: this Hint is no longer in the database."
    else:
        from src.core.llm import DEFAULT_TEMPERATURE_COLD, LlmManager, MAX_COMPLETION_TOKENS_SMALL

        agent_hints: Dict[str, Any] = hint_details.get("agent_hints") or {}
        llm_manager: LlmManager = LlmManager.instance()
        rendered_instructions: List[str] = []

        for agent_name, hint_text in agent_hints.items():
            if not isinstance(agent_name, str) or not isinstance(hint_text, str) or not hint_text.strip():
                continue

            prompt_engineering: Optional[Dict[str, Any]] = llm_manager.get_agent_prompt_engineering(agent_name, None)
            if not isinstance(prompt_engineering, dict):
                continue

            system_prompt: Optional[str] = prompt_engineering.get("systemprompt")
            if isinstance(system_prompt, str) and system_prompt.strip():
                rendered_instructions.append(
                    f'<SYSTEM_MESSAGE id="{agent_name}/DEFAULT">\n{system_prompt}\n</SYSTEM_MESSAGE>'
                )

            other_instructions: Any = prompt_engineering.get("other", [])
            rendered_other_labels: Set[str] = set()
            if isinstance(other_instructions, list):
                for instruction in other_instructions:
                    if not isinstance(instruction, dict):
                        continue
                    label: Any = instruction.get("key")
                    instruction_text: Any = instruction.get("value")
                    if isinstance(label, str) and isinstance(instruction_text, str) and instruction_text.strip():
                        rendered_other_labels.add(label)
                        rendered_instructions.append(
                            f'<INSTRUCTIONS id="{agent_name}/{label}">\n{instruction_text}\n</INSTRUCTIONS>'
                        )

            rendered_instructions.extend(_render_agent_skill_instructions(agent_name, llm_manager, skip_labels=rendered_other_labels))

        if not rendered_instructions:
            analysis_results = "Agent-instruction conflict evaluation could not be performed: no active instructions were found for the Hint's target agents. The Hint is kept for trainer inspection."
        else:
            instructions_message: str = (
                "These are the active permanent agent instructions for the Hint's target agents: system messages, named agent instructions, and skill descriptions. "
                "Compare the Hint only with these instructions; their XML IDs identify the source:\n\n"
                + "\n\n".join(rendered_instructions)
            )
            hint_message: str = (
                "Evaluate whether this trainer Hint conflicts with any supplied agent instruction. "
                "Do not compare it with other Hints.\n\n"
                + _render_hint_for_redundancy_check(hint_details)
            )
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": HINT_CONFLICT_SYSTEM_PROMPT},
                {"role": "system", "content": instructions_message},
                {"role": "user", "content": hint_message},
            ]

            model_configuration_id: Optional[str] = llm_manager.get_model_by_skill_id(HINT_CONSOLIDATION_SKILL_ID)
            llm_response = llm_manager.chat_completions(
                messages=messages,
                model_input=model_configuration_id,
                max_completion_tokens=MAX_COMPLETION_TOKENS_SMALL,
                temperature=DEFAULT_TEMPERATURE_COLD,
                response_format=HINT_CONFLICT_JSON_EXAMPLE,
                structured_output=HINT_CONFLICT_JSON_SCHEMA,
                maximum_reasoning_required=HINT_CONSOLIDATION_REASONING_LEVEL,
            )

            conflict_evaluation: Optional[Dict[str, Any]] = None
            if llm_response and llm_response.response:
                if isinstance(llm_response.response, dict):
                    conflict_evaluation = llm_response.response
                elif isinstance(llm_response.response, str):
                    try:
                        conflict_evaluation = json.loads(llm_response.response)

                    except json.JSONDecodeError:
                        log(WARNING, f"The LLM agent-instruction conflict check for trainer Hint '{hint_id}' returned text that could not be parsed as JSON.")

            if conflict_evaluation is not None and isinstance(conflict_evaluation.get("conflict"), bool):
                has_conflict = conflict_evaluation["conflict"]
                reason: Any = conflict_evaluation.get("reason")
                if isinstance(reason, str) and reason.strip():
                    analysis_results = reason
                else:
                    analysis_results = "The agent-instruction conflict check completed without a reason from the LLM."
            else:
                analysis_results = "Agent-instruction conflict evaluation did not produce a usable answer. The Hint is kept for trainer inspection."

    return has_conflict, analysis_results

def _render_hint_similarity_candidates(candidates: List[Dict[str, Any]]) -> str:
    """Render the similarity-selected candidate Hints of the Hint-versus-Hint comparison, with their stored information and similarity."""
    rendered: str = "These are the existing Hints most similar to the Hint to evaluate, each with all of the information stored about it:\n\n"
    for candidate in candidates:
        similarity_score: Any = candidate.get("similarity_score")
        if isinstance(similarity_score, float):
            rendered += f"- Similarity to the new Hint: {similarity_score:.3f}\n"

        rendered += _render_hint_for_redundancy_check(candidate)
        rendered += "\n"

    return rendered

def consolidate_hint_compare_with_similar_hints(hint_id: str) -> HintRelationshipResult:
    """
    Compare a trainer Hint with the existing Hints that are most similar to it, selected by the Hint's stored embedding.
    Returns the analysis recommendation the confirmed relationship implies (`NONE` when the candidates are unrelated or the
    evidence is not sufficient, so that no state is changed on uncertain evidence), the candidate IDs the model named,
    and the explanation of the evaluation to be persisted as the analysis results of the Hint.
    """
    recommendation: HintRecommendation = HintRecommendation.NONE
    candidate_hint_ids: List[str] = []
    analysis_results: Optional[str] = None
    hints_service: HintsService = _get_hints_service()
    new_hint: Optional[Dict[str, Any]] = hints_service.get_hint_details(hint_id)

    if new_hint is None:
        log(ERROR, f"Cannot compare trainer Hint '{hint_id}' with similar Hints: the Hint could not be found.")
        analysis_results = "Hint-versus-Hint evaluation could not be performed: this Hint is no longer in the database."

    else:
        embedding: Optional[List[float]] = hints_service.get_hint_embedding(hint_id)
        if embedding is None:
            log(WARNING, f"Trainer Hint '{hint_id}' has no stored embedding: similar Hints could not be selected.")
            analysis_results = "Hint-versus-Hint evaluation could not be performed: this Hint has no stored embedding."

        else:
            candidates: List[Dict[str, Any]] = hints_service.get_hint_similarity_candidates(
                hint_id=hint_id,
                embedding=embedding,
                top_n=HINT_SIMILARITY_CANDIDATE_LIMIT,
                min_score=HINT_SIMILARITY_MIN_SCORE,
            )

            if not candidates:
                log(DEBUG, f"Trainer Hint '{hint_id}' is unrelated to other Hints: no similar Hint candidate passed the similarity threshold.")
                analysis_results = "Not related to another Hint: no other Hint is similar enough to be worth comparing."

            else:
                from src.core.llm import DEFAULT_TEMPERATURE_COLD, LlmManager, MAX_COMPLETION_TOKENS_SMALL

                candidates_message: str = _render_hint_similarity_candidates(candidates)
                new_hint_message: str = (
                    "Evaluate the relationship between the following new Hint and the supplied candidate Hints. "
                    "Judge only the relationship between Hints; the comparison with permanent agent instructions is a separate check.\n\n"
                    f"{_render_hint_for_redundancy_check(new_hint)}\n"
                    "Respond with the requested JSON object, naming the candidates that drove your answer."
                )
                messages: List[Dict[str, str]] = [
                    {"role": "system", "content": HINT_VS_HINT_SYSTEM_PROMPT},
                    {"role": "system", "content": candidates_message},
                    {"role": "user", "content": new_hint_message},
                ]

                llm_manager: LlmManager = LlmManager.instance()
                model_configuration_id: Optional[str] = llm_manager.get_model_by_skill_id(HINT_CONSOLIDATION_SKILL_ID)
                llm_response = llm_manager.chat_completions(
                    messages=messages,
                    model_input=model_configuration_id,
                    max_completion_tokens=MAX_COMPLETION_TOKENS_SMALL,
                    temperature=DEFAULT_TEMPERATURE_COLD,
                    response_format=HINT_VS_HINT_JSON_EXAMPLE,
                    structured_output=HINT_VS_HINT_JSON_SCHEMA,
                    maximum_reasoning_required=HINT_CONSOLIDATION_REASONING_LEVEL,
                )

                relationship_evaluation: Optional[Dict[str, Any]] = None
                if llm_response and llm_response.response:
                    if isinstance(llm_response.response, dict):
                        relationship_evaluation = llm_response.response
                    elif isinstance(llm_response.response, str):
                        try:
                            relationship_evaluation = json.loads(llm_response.response)

                        except json.JSONDecodeError:
                            log(WARNING, f"The LLM Hint-versus-Hint comparison for trainer Hint '{hint_id}' returned text that could not be parsed as JSON.")

                if relationship_evaluation is None:
                    log(WARNING, f"The LLM Hint-versus-Hint comparison for trainer Hint '{hint_id}' returned no usable answer: the Hint keeps the outcome of its earlier checks.")
                    analysis_results = "Hint-versus-Hint evaluation did not produce a usable answer: the Hint keeps the outcome of its earlier checks."

                else:
                    relationship: Any = relationship_evaluation.get("relationship")
                    reason: Any = relationship_evaluation.get("reason")
                    named_ids: Any = relationship_evaluation.get("candidate_hint_ids")
                    supplied_candidate_ids: List[str] = [str(candidate.get("id")) for candidate in candidates]
                    if isinstance(named_ids, list):
                        candidate_hint_ids = [str(candidate_id) for candidate_id in named_ids if str(candidate_id) in supplied_candidate_ids]

                    if not isinstance(relationship, str) or relationship not in HINT_RELATIONSHIP_RECOMMENDATIONS:
                        analysis_results = "Hint-versus-Hint evaluation returned a relationship outside the supported set: the Hint keeps the outcome of its earlier checks."

                    elif relationship != "unrelated" and not candidate_hint_ids:
                        log(WARNING, f"The LLM Hint-versus-Hint comparison for trainer Hint '{hint_id}' reported '{relationship}' without naming a supplied candidate: the Hint keeps the outcome of its earlier checks.")
                        analysis_results = "Hint-versus-Hint evaluation reported a relationship without naming the candidate Hints it concerns: the Hint keeps the outcome of its earlier checks."

                    else:
                        recommendation = HINT_RELATIONSHIP_RECOMMENDATIONS[relationship]
                        if reason:
                            log(DEBUG, f"Hint-versus-Hint evaluation for trainer Hint '{hint_id}': {reason}")
                            analysis_results = str(reason)

    return HintRelationshipResult(recommendation=recommendation, candidate_hint_ids=candidate_hint_ids, analysis_results=analysis_results)

def _append_analysis_results(existing_results: Optional[str], additional_results: Optional[str]) -> Optional[str]:
    """Combine the analysis explanation of one check with that of the next, keeping every evaluation visible."""
    combined: Optional[str] = existing_results
    if additional_results:
        combined = f"{existing_results}\n\n---\n\n{additional_results}" if existing_results else additional_results

    return combined

def consolidate_hint_check_policy(hint_id: str) -> Tuple[Optional[bool], Optional[str]]:
    """Ask the usage-policy guard whether following this Hint would make an agent break a rule it must never break."""
    is_rejected: Optional[bool] = None
    analysis_results: Optional[str] = None
    hint_details: Optional[Dict[str, Any]] = _get_hints_service().get_hint_details(hint_id)

    if hint_details is None:
        analysis_results = "Usage-policy evaluation could not be performed: this Hint is no longer in the database."
    else:
        from src.core.llm import DEFAULT_TEMPERATURE_COLD, LlmManager, MAX_COMPLETION_TOKENS_SMALL

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": HINT_POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": "Evaluate this trainer Hint against the usage policy.\n\n" + _render_hint_for_redundancy_check(hint_details)},
        ]

        llm_manager: LlmManager = LlmManager.instance()
        model_configuration_id: Optional[str] = llm_manager.get_model_by_skill_id(HINT_CONSOLIDATION_SKILL_ID)
        llm_response = llm_manager.chat_completions(
            messages=messages,
            model_input=model_configuration_id,
            max_completion_tokens=MAX_COMPLETION_TOKENS_SMALL,
            temperature=DEFAULT_TEMPERATURE_COLD,
            response_format=HINT_POLICY_JSON_EXAMPLE,
            structured_output=HINT_POLICY_JSON_SCHEMA,
            maximum_reasoning_required=HINT_CONSOLIDATION_REASONING_LEVEL,
        )

        policy_evaluation: Optional[Dict[str, Any]] = None
        if llm_response and llm_response.response:
            if isinstance(llm_response.response, dict):
                policy_evaluation = llm_response.response
            elif isinstance(llm_response.response, str):
                try:
                    policy_evaluation = json.loads(llm_response.response)

                except json.JSONDecodeError:
                    log(WARNING, f"The usage-policy guard for trainer Hint '{hint_id}' returned text that could not be parsed as JSON.")

        if policy_evaluation is not None and isinstance(policy_evaluation.get("rejected"), bool):
            is_rejected = policy_evaluation["rejected"]
            reason: Any = policy_evaluation.get("reason")
            if isinstance(reason, str) and reason.strip():
                analysis_results = reason
            else:
                analysis_results = "The usage-policy guard completed without a reason from the LLM."
        else:
            analysis_results = "Usage-policy evaluation did not produce a usable answer. The Hint is kept for trainer inspection."

    return is_rejected, analysis_results

def _render_hint_merge_participants(hint_id: str, survivor: Dict[str, Any], participants: List[Dict[str, Any]]) -> str:
    """Render the Hint to consolidate and its loaded merge candidates, each labelled with its role in the merge."""
    rendered: str = (
        f"These are the Hints to merge. The Hint with ID `{hint_id}` is the Hint being consolidated: "
        "the consolidated guidance is stored under its ID and the merged candidates stop being injected. "
        "Every supplied Hint contributes content the merged guidance must keep:\n\n"
    )
    rendered += "Role: Hint being consolidated\n"
    rendered += _render_hint_for_redundancy_check(survivor)
    rendered += "\n"
    for candidate in participants:
        rendered += "Role: merged candidate\n"
        rendered += _render_hint_for_redundancy_check(candidate)
        rendered += "\n"

    return rendered

@dataclass(frozen=True)
class HintMergeEvaluation:
    """Outcome of evaluating the merge answer: whether it is complete enough to persist, the merged texts, and the explanation."""
    is_complete: bool
    merged_agent_hints: Dict[str, str]
    analysis_results: str

def _evaluate_hint_merge_answer(hint_id: str, survivor: Dict[str, Any], participants: List[Dict[str, Any]]) -> HintMergeEvaluation:
    """Ask the LLM for the consolidated guidance and check it strictly: every agent any source covers must come back covered."""
    evaluation: HintMergeEvaluation = HintMergeEvaluation(is_complete=False, merged_agent_hints={}, analysis_results="Hint merge did not produce a usable answer: the Hint stays queued for the next merge attempt.")
    required_agent_keys: List[str] = []

    for participant in [survivor] + participants:
        for agent_name, agent_hint_text in (participant.get("agent_hints") or {}).items():
            if isinstance(agent_name, str) and agent_name.strip() and isinstance(agent_hint_text, str) and agent_hint_text.strip() and agent_name not in required_agent_keys:
                required_agent_keys.append(agent_name)

    from src.core.llm import DEFAULT_TEMPERATURE_COLD, LlmManager, MAX_COMPLETION_TOKENS

    merge_messages: List[Dict[str, str]] = [
        {"role": "system", "content": HINT_MERGE_SYSTEM_PROMPT},
        {"role": "system", "content": _render_hint_merge_participants(hint_id, survivor, participants)},
        {
            "role": "user",
            "content": (
                "Consolidate the Hint being consolidated and its merge candidates into one Hint. "
                "Respond with the requested JSON object, covering every agent key that any supplied Hint addresses."
            ),
        },
    ]

    llm_manager: LlmManager = LlmManager.instance()
    model_configuration_id: Optional[str] = llm_manager.get_model_by_skill_id(HINT_CONSOLIDATION_SKILL_ID)
    llm_response = llm_manager.chat_completions(
        messages=merge_messages,
        model_input=model_configuration_id,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        temperature=DEFAULT_TEMPERATURE_COLD,
        response_format=HINT_MERGE_JSON_EXAMPLE,
        structured_output=HINT_MERGE_JSON_SCHEMA,
        maximum_reasoning_required=HINT_CONSOLIDATION_REASONING_LEVEL,
    )

    merge_evaluation: Optional[Dict[str, Any]] = None
    if llm_response and llm_response.response:
        if isinstance(llm_response.response, dict):
            merge_evaluation = llm_response.response
        elif isinstance(llm_response.response, str):
            try:
                merge_evaluation = json.loads(llm_response.response)

            except json.JSONDecodeError:
                log(WARNING, f"The LLM merge for trainer Hint '{hint_id}' returned text that could not be parsed as JSON.")

    if merge_evaluation is not None:
        reason: Any = merge_evaluation.get("reason")
        merged_entries: Any = merge_evaluation.get("merged_hints")
        participant_ids: List[str] = [hint_id] + [str(participant.get("id")) for participant in participants]
        supplied_ids: Set[str] = set(participant_ids)
        merged_agent_hints: Dict[str, str] = {}

        if isinstance(merged_entries, list):
            for entry in merged_entries:
                if not isinstance(entry, dict):
                    continue

                agent_name: Any = entry.get("agent")
                merged_hint: Any = entry.get("merged_hint")
                sources: Any = entry.get("source_hint_ids")
                named_sources: List[str] = [
                    str(source_id)
                    for source_id in (sources if isinstance(sources, list) else [])
                    if str(source_id) in supplied_ids
                ]
                if isinstance(agent_name, str) and agent_name in required_agent_keys and isinstance(merged_hint, str) and merged_hint.strip() and named_sources:
                    merged_agent_hints[agent_name] = merged_hint.strip()

        uncovered_agent_keys: List[str] = [agent_name for agent_name in required_agent_keys if agent_name not in merged_agent_hints]
        if not isinstance(reason, str) or not reason.strip():
            evaluation = HintMergeEvaluation(is_complete=False, merged_agent_hints={}, analysis_results="Hint merge answer carried no explanation: the Hint stays queued for the next merge attempt.")
        elif uncovered_agent_keys:
            log(WARNING, f"The LLM merge for trainer Hint '{hint_id}' left agents uncovered: {', '.join(uncovered_agent_keys)}.")
            evaluation = HintMergeEvaluation(is_complete=False, merged_agent_hints={}, analysis_results=f"Hint merge answer left the agents {', '.join(uncovered_agent_keys)} uncovered: the Hint stays queued for the next merge attempt.")
        else:
            log(DEBUG, f"Hint merge evaluation for trainer Hint '{hint_id}': {reason}")
            evaluation = HintMergeEvaluation(is_complete=True, merged_agent_hints=merged_agent_hints, analysis_results=reason)

    return evaluation

def _perform_hint_merge(
    hint_id: str,
    survivor: Dict[str, Any],
    participants: List[Dict[str, Any]],
    missing_candidate_ids: List[str],
) -> bool:
    """Persist a complete merge answer in the order that can never lose content: merged texts first, candidate retirements after."""
    completed: bool = False
    evaluation: HintMergeEvaluation = _evaluate_hint_merge_answer(hint_id, survivor, participants)

    if not evaluation.is_complete:
        log(WARNING, f"The merge of trainer Hint '{hint_id}' was not persisted: {evaluation.analysis_results}")

    else:
        content_updated: bool = _get_hints_service().update_hint_agent_hints(hint_id, evaluation.merged_agent_hints)

        if not content_updated:
            log(ERROR, f"The merged guidance for trainer Hint '{hint_id}' could not be persisted: the Hint stays queued for the next merge attempt.")

        else:
            candidate_analysis_results: str = f"Absorbed by the merge into Hint '{hint_id}'.\n\n{evaluation.analysis_results}"
            failed_candidate_ids: List[str] = []

            for participant in participants:
                marked: bool = _get_hints_service().update_hint_state(
                    hint_id=str(participant.get("id")),
                    recommendation=HintRecommendation.REDUNDANT,
                    needs_consolidation=False,
                    analysis_results=candidate_analysis_results,
                )
                if not marked:
                    failed_candidate_ids.append(str(participant.get("id")))

            merge_note: str = evaluation.analysis_results
            if missing_candidate_ids:
                merge_note = f"{merge_note}\n\nMerge candidates no longer present and therefore skipped: {', '.join(missing_candidate_ids)}."

            if failed_candidate_ids:
                # The merged content is already persisted, so re-queuing the merge flag only schedules
                # the retirement of the remaining candidates: repeating the merge converges to the same result.
                _get_hints_service().update_hint_state(
                    hint_id=hint_id,
                    recommendation=HintRecommendation.PARTIAL_OVERLAP_WITH_HINT,
                    needs_consolidation=survivor.get("needs_consolidation"),
                    analysis_results=_append_analysis_results(survivor.get("analysis_results"), f"{merge_note}\n\nThese candidates could not be marked Redundant; the merge is re-queued for them: {', '.join(failed_candidate_ids)}."),
                    merge_candidates=failed_candidate_ids,
                )
                log(WARNING, f"The merge of trainer Hint '{hint_id}' could not retire the candidates {', '.join(failed_candidate_ids)}: the merge is re-queued for them.")

            else:
                completed = _get_hints_service().update_hint_state(
                    hint_id=hint_id,
                    recommendation=HintRecommendation.NEW_VALID_HINT_NOT_OVERLAPPING,
                    needs_consolidation=False,
                    analysis_results=_append_analysis_results(survivor.get("analysis_results"), merge_note),
                )
                if completed:
                    log(INFO, f"Trainer Hint '{hint_id}' absorbed {', '.join(str(participant.get('id')) for participant in participants)}: the candidates are now Redundant and are no longer injected on their own.")
                else:
                    log(ERROR, f"The consolidated state of trainer Hint '{hint_id}' could not be persisted: the merge is retried on the next pass.")

    return completed

def consolidate_hint_merge_with_candidates(hint_id: str) -> bool:
    """
    Consolidate a trainer Hint with the merge candidates the Hint-versus-Hint comparison recorded for it,
    as the stage-2 merge of the two-stage consolidation. The LLM writes the consolidated guidance, which is
    persisted under this Hint's ID before any candidate's state is changed, so the content of a candidate
    can never be lost even when a later step fails. On success this Hint becomes a regular non-overlapping
    Hint, which releases its merge requirement, and every merged candidate becomes Redundant, which removes
    it from injection. Returns True only when the whole state transition was persisted; otherwise this Hint
    stays in the merge queue and the next worker pass retries it, converging to the same consolidated result.
    """
    log(DEBUG, f"Consolidating trainer Hint '{hint_id}' with its recorded merge candidates.")
    completed: bool = False
    hints_service: HintsService = _get_hints_service()
    survivor: Optional[Dict[str, Any]] = hints_service.get_hint_details(hint_id)

    if survivor is None:
        log(WARNING, f"Cannot merge trainer Hint '{hint_id}': the Hint is no longer in the database.")

    elif survivor.get("analysis_recommendation") == HintRecommendation.REDUNDANT.value:
        # A previous pass already absorbed this Hint into another one and only its stale merge state is
        # left behind: re-marking it Redundant clears the queue entry without touching any content.
        completed = hints_service.update_hint_state(
            hint_id=hint_id,
            recommendation=HintRecommendation.REDUNDANT,
            needs_consolidation=False,
            analysis_results=survivor.get("analysis_results"),
        )
        log(INFO, f"Trainer Hint '{hint_id}' had already been merged into another Hint: only its stale merge state was cleared.")

    else:
        stored_candidates: Any = survivor.get("merge_candidates")
        candidate_ids: List[str] = [
            candidate_id
            for candidate_id in (stored_candidates if isinstance(stored_candidates, list) else [])
            if isinstance(candidate_id, str) and candidate_id.strip() and candidate_id != hint_id
        ]
        participants: List[Dict[str, Any]] = []
        missing_candidate_ids: List[str] = []

        for candidate_id in candidate_ids:
            candidate_details: Optional[Dict[str, Any]] = hints_service.get_hint_details(candidate_id)
            if candidate_details is None:
                log(WARNING, f"Merge candidate '{candidate_id}' of trainer Hint '{hint_id}' is no longer in the database: it is skipped.")
                missing_candidate_ids.append(candidate_id)
            else:
                participants.append(candidate_details)

        if not candidate_ids:
            # The queue entry asks for a merge without naming a candidate: leave it untouched so the
            # next Hint-versus-Hint comparison can name one instead of inventing a merge here.
            log(WARNING, f"Trainer Hint '{hint_id}' is queued for a merge without any usable merge candidate: it is left untouched.")

        elif not participants:
            # Every recorded candidate has disappeared: no merge is possible anymore, so release the
            # merge requirement instead of leaving the Hint in the queue forever (the prompt itself is unchanged).
            completed = hints_service.update_hint_state(
                hint_id=hint_id,
                recommendation=HintRecommendation.NEW_VALID_HINT_NOT_OVERLAPPING,
                needs_consolidation=survivor.get("needs_consolidation"),
                analysis_results=_append_analysis_results(
                    survivor.get("analysis_results"),
                    "Hint merge cancelled: none of the recorded merge candidates is in the database anymore.",
                ),
            )
            log(INFO, f"The merge of trainer Hint '{hint_id}' was cancelled because all its recorded merge candidates are gone.")

        else:
            completed = _perform_hint_merge(hint_id, survivor, participants, missing_candidate_ids)

    return completed

def process_hints_requiring_merge(hint_ids: List[str]) -> None:
    """Consolidate the queued Hint merges in parallel."""
    log(DEBUG, f"Trainer Hints requiring a merge: {', '.join(hint_ids)}")

    if not hint_ids:
        return

    max_workers: int = min(len(hint_ids), MAX_PARALLELISM)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_hint_id = {
            executor.submit(consolidate_hint_merge_with_candidates, hint_id): hint_id
            for hint_id in hint_ids
        }
        for future in as_completed(future_to_hint_id):
            hint_id: str = future_to_hint_id[future]
            try:
                merged: bool = future.result()
                if merged:
                    log(INFO, f"Trainer Hint '{hint_id}' finished its merge.")

            except Exception as e:
                log(ERROR, f"Failed to merge trainer Hint '{hint_id}': {e}")

    return

def consolidate_hint(hint_id: str) -> None:
    """Analyze a pending Hint and move it to the recommendation its content deserves."""
    log(DEBUG, f"Consolidating trainer Hint '{hint_id}'.")

    needs_consolidation: Optional[bool] = None
    merge_candidates: Optional[List[str]] = None

    recommendation: HintRecommendation = HintRecommendation.TRAINER_INSPECTION
    analysis_results: Optional[str] = None
    is_rejected: Optional[bool] = None
    has_conflict: Optional[bool] = None

    hint_details: Optional[Dict[str, Any]] = _get_hints_service().get_hint_details(hint_id)
    structural_error: Optional[str] = get_hint_structural_error(hint_details) if hint_details is not None else None

    if structural_error is not None:
        recommendation = HintRecommendation.INVALID
        needs_consolidation = False
        analysis_results = structural_error
        log(WARNING, f"Trainer Hint '{hint_id}' is structurally invalid: {structural_error}")

    else:
        is_rejected, analysis_results = consolidate_hint_check_policy(hint_id)

        if is_rejected is True:
            recommendation = HintRecommendation.REJECTED_POLICY
            needs_consolidation = False
        else:
            has_conflict, analysis_results = consolidate_hint_check_for_conflict_with_agent_instructions(hint_id)

            if has_conflict is True:
                recommendation = HintRecommendation.CONFLICT_WITH_AGENT_INSTRUCTIONS
                needs_consolidation = False
            elif has_conflict is False:
                redundancy_result = consolidate_hint_check_for_redundancy(hint_id)
                is_redundant: bool
                redundancy_results: Optional[str]
                is_redundant, redundancy_results = redundancy_result
                analysis_results = redundancy_results
                adds_new_information: bool = bool(getattr(redundancy_result, "adds_new_information", False))

                if is_redundant:
                    recommendation = HintRecommendation.REDUNDANT
                    needs_consolidation = False
                else:
                    if adds_new_information:
                        recommendation = HintRecommendation.NEW_VALID_HINT_NOT_OVERLAPPING
                        needs_consolidation = False

                    relationship_result: HintRelationshipResult = consolidate_hint_compare_with_similar_hints(hint_id)
                    if relationship_result.recommendation is not HintRecommendation.NONE:
                        recommendation = relationship_result.recommendation
                        needs_consolidation = False
                        if relationship_result.recommendation is HintRecommendation.PARTIAL_OVERLAP_WITH_HINT:
                            merge_candidates = relationship_result.candidate_hint_ids if relationship_result.candidate_hint_ids else None

                    analysis_results = _append_analysis_results(analysis_results, relationship_result.analysis_results)

    updated: bool = _get_hints_service().update_hint_state(
        hint_id,
        recommendation=recommendation,
        needs_consolidation=needs_consolidation,
        analysis_results=analysis_results,
        merge_candidates=merge_candidates,
    )
    if updated:
        log(INFO, f"Successfully marked trainer Hint '{hint_id}' as '{recommendation.value}'.")
    else:
        log(ERROR, f"Could not mark trainer Hint '{hint_id}' as '{recommendation.value}'.")

    return

def trainer_hint_consolidation_thread() -> None:
    if not ENABLE_HINT_REDUNDANCY_CHECK:
        log(INFO, "Hint redundancy check is disabled. Quitting trainer_hint_consolidation_thread.")
        return
    
    try:
        log(INFO, "Starting trainer_hint_consolidation_thread...")

        while not shutdown_event[0].is_set():
            try:
                hint_ids: List[str] = get_hint_ids_by_recommendation(HintRecommendation.NONE)
                if hint_ids:
                    process_hints_requiring_consolidation(hint_ids)
                else:
                    log(DEBUG, "Trainer Hint consolidation thread is alive and running.")

            except Exception as e:
                log(ERROR, f"Error while checking trainer Hints for consolidation: {e}")

            try:
                merge_ids: List[str] = _get_hints_service().get_hint_ids_requiring_merge()
                if merge_ids:
                    process_hints_requiring_merge(merge_ids)

            except Exception as e:
                log(ERROR, f"Error while merging trainer Hints: {e}")

            shutdown_event[0].wait(5)

    except Exception as e:
        log(FATAL, f"FATAL ERROR in trainer_hint_consolidation_thread: {e}")

    finally:
        log(INFO, "Exiting trainer_hint_consolidation_thread...")

    return


log_end_file(__file__)


if __name__ == "__main__":
    RESTORE_HINT_FOR_TRAINING_INSPECTION = False

    # This block is for testing purposes only: it consolidates, one at a time, the Hints that are
    # currently waiting for a trainer, then puts each of them back into the state it had before.
    def recommendation_from_stored_value(stored_value: Optional[str]) -> HintRecommendation:
        """Convert one stored analysis recommendation back into its enumeration member."""
        for recommendation in HintRecommendation:
            if recommendation.value == stored_value:
                return recommendation

        return HintRecommendation.NONE

    def get_hints_requiring_trainer_inspection() -> List[Dict[str, Any]]:
        """Snapshot every Hint waiting for a trainer, together with the state needed to restore it later."""
        hints_service: HintsService = _get_hints_service()
        snapshots: List[Dict[str, Any]] = []

        hints_for_trainer_inspection: List[str] = get_hint_ids_by_recommendation(HintRecommendation.TRAINER_INSPECTION)
        for hint_id in hints_for_trainer_inspection:
            hint_details: Optional[Dict[str, Any]] = hints_service.get_hint_details(hint_id)
            if hint_details is not None:
                snapshots.append(hint_details)

        if snapshots:
            log(INFO, f"Selected for testing: {', '.join(str(snapshot.get('id')) for snapshot in snapshots)}")

        return snapshots

    def reset_hints_for_testing(snapshots: List[Dict[str, Any]]) -> None:
        """Put every selected Hint back into the pending state, as if it had never been analyzed."""
        hints_service: HintsService = _get_hints_service()
        reset_hint_ids: List[str] = []

        for snapshot in snapshots:
            hint_id: str = str(snapshot.get("id"))
            update_succeeded: bool = hints_service.update_hint_state(hint_id=hint_id, recommendation=HintRecommendation.NONE, needs_consolidation=True)
            if update_succeeded:
                reset_hint_ids.append(hint_id)

        if reset_hint_ids:
            log(INFO, f"Reset for testing: {', '.join(reset_hint_ids)}")

        return

    def restore_hints_after_testing(snapshots: List[Dict[str, Any]]) -> None:
        """Return every selected Hint to the exact state it had before testing."""
        hints_service: HintsService = _get_hints_service()
        restored_hint_ids: List[str] = []

        for snapshot in snapshots:
            hint_id: str = str(snapshot.get("id"))
            recommendation: HintRecommendation = recommendation_from_stored_value(snapshot.get("analysis_recommendation"))
            needs_consolidation: Optional[bool] = snapshot.get("needs_consolidation")
            analysis_results: Optional[str] = snapshot.get("analysis_results")
            update_succeeded: bool = hints_service.update_hint_state(hint_id=hint_id, recommendation=recommendation, needs_consolidation=needs_consolidation, analysis_results=analysis_results)
            if update_succeeded:
                restored_hint_ids.append(hint_id)

        if restored_hint_ids:
            log(INFO, f"Restored to their original state: {', '.join(restored_hint_ids)}")

        return

    def display_hint_after_consolidation(hint_id: str) -> None:
        """Print the full state of one Hint, so the outcome of its consolidation can be inspected."""
        def print_robust(text: str) -> None:
            """Print text, replacing characters the console encoding cannot represent."""
            console_encoding: str = getattr(sys.stdout, "encoding", None) or "utf-8"
            print(text.encode(console_encoding, errors="replace").decode(console_encoding, errors="replace"))

        hint_details: Optional[Dict[str, Any]] = _get_hints_service().get_hint_details(hint_id)
        print_robust("\n" + "-" * 100)

        if hint_details is None:
            print_robust(f"Hint '{hint_id}' could not be displayed: it is no longer in the database.")
            return

        print_robust(f"After consolidation, trainer Hint '{hint_id}' is:")
        print_robust(_render_hint_for_redundancy_check(hint_details))

        return

    log_init()
    tested_hints: List[Dict[str, Any]] = []

    try:
        tested_hints = get_hints_requiring_trainer_inspection()
        reset_hints_for_testing(tested_hints)

        for hint_snapshot in tested_hints:
            hint_id: str = str(hint_snapshot.get("id"))
            try:
                consolidate_hint(hint_id)
                display_hint_after_consolidation(hint_id)
                pass # breakpoint here to inspect the state after each hint is consolidated

            except Exception as e:
                log(ERROR, f"Failed to consolidate trainer Hint '{hint_id}' while testing: {e}")

    finally:
        if RESTORE_HINT_FOR_TRAINING_INSPECTION:
            restore_hints_after_testing(tested_hints)

        log_exit()
