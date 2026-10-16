from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import json
from typing import Dict, List, Optional, Union

from src.utils.log import log, DEBUG, INFO, WARNING, ERROR


def _repair_unquoted_colon_values(text: str) -> str:
    """
    Heuristic repair for a common LLM YAML mistake: a plain scalar value that
    contains a colon, which YAML may interpret as a mapping key indicator.

    For every mapping line whose unquoted value contains any colon character,
    the value is wrapped in double-quoted style (with internal backslashes and
    double-quotes escaped).  Lines that are already quoted, block-scalar,
    flow-collection, anchors, or list-item lines are left untouched.

    Using any colon (not just colon-space) as the trigger is intentional:
    both `: ` and `:` at end-of-value can confuse a YAML parser in block
    context, and for LLM-generated YAML plain-scalar values a bare colon is
    always free text that must be quoted.
    """
    import re
    _SKIP_PREFIXES = ('"', "'", '|', '>', '{', '[', '&', '*', '!')
    _LINE_RE = re.compile(r'^(\s*)([A-Za-z_][A-Za-z0-9_-]*):\s+(.+)$')

    repaired = []
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if m:
            indent, key, value = m.group(1), m.group(2), m.group(3)
            value_stripped = value.strip()
            if (':' in value_stripped
                    and not value_stripped.startswith(_SKIP_PREFIXES)):
                escaped = value_stripped.replace('\\', '\\\\').replace('"', '\\"')
                line = f'{indent}{key}: "{escaped}"'
        repaired.append(line)
    return '\n'.join(repaired)


def _strip_prose_trailer(text: str) -> str:
    """
    Remove trailing non-YAML prose that LLMs occasionally append after their
    structured YAML output (e.g. a ``Reasoning:`` block or numbered-list
    explanation).

    Two patterns at indent-0 are treated as the start of prose:
      1. A single Title-Case word followed by a bare colon, e.g. ``Reasoning:``
         or ``Note:``.  (Snake-case YAML keys such as ``optional_complex_plan:``
         are NOT matched because they contain underscores.)
      2. A numbered-list item such as ``1. text`` or ``2) text``.

    When either pattern is found the text is truncated just before that line
    (trailing blank lines are also removed).
    """
    import re
    # Single CamelCase/Title word + colon, nothing after — "Reasoning:", "Note:"
    # Deliberately excludes snake_case keys (contains underscore) and hyphenated keys.
    _PROSE_HEADER = re.compile(r'^[A-Z][A-Za-z]*:\s*$')
    # Numbered list marker: "1. ", "2) ", etc.
    _NUMBERED_LIST = re.compile(r'^\s*\d+[.)]\s')

    lines = text.splitlines()
    cutoff = len(lines)

    for i, line in enumerate(lines):
        if _PROSE_HEADER.match(line) or _NUMBERED_LIST.match(line):
            cutoff = i
            break

    if cutoff < len(lines):
        # Trim any blank lines that precede the cutoff
        while cutoff > 0 and not lines[cutoff - 1].strip():
            cutoff -= 1
        if cutoff > 0:
            log(DEBUG, f'_strip_prose_trailer: stripped {len(lines) - cutoff} trailing line(s)')
            return '\n'.join(lines[:cutoff])
    return text


def _normalize_block_empty_collections(text: str) -> str:
    """Collapse block-style empty collections that confuse YAML parsers.

    LLMs sometimes write:
        values:
        []
    instead of the valid inline form:
        values: []
    This function detects the pattern (a mapping key line immediately followed
    by a line containing only `[]` or `{}`) and collapses it to inline form.
    """
    import re
    return re.sub(r'(^\s*[A-Za-z_][A-Za-z0-9_-]*:)\s*\n(\s*)(\[\]|\{\})', r'\1 \3', text, flags=re.MULTILINE)


def get_yaml_from_text(yaml_text: str) -> Optional[Union[Dict, List]]:
    """
    Extract and parse YAML from LLM text output, with robust handling
    of markdown code fences and minor formatting issues.

    Args:
        yaml_text: The text containing YAML to extract

    Returns:
        Parsed YAML as dict or list, or None if parsing fails
    """
    import yaml

    if yaml_text is None:
        return None

    text = yaml_text.strip()
    if len(text) == 0:
        return {}

    # Strip markdown code fences
    if text.startswith("```yaml"):
        text = text[len("```yaml"):].lstrip()
    elif text.startswith("```yml"):
        text = text[len("```yml"):].lstrip()
    elif text.startswith("```"):
        text = text[3:].lstrip()

    if text.endswith("```"):
        text = text[:-3].rstrip()

    text = text.strip()

    # Normalize block-style empty collections (e.g. 'values:\n[]') to inline form before
    # any other processing — this is a common LLM formatting mistake that causes parse errors.
    text = _normalize_block_empty_collections(text)

    try:
        result = yaml.safe_load(text)
        if isinstance(result, (dict, list)):
            return result
        if isinstance(result, str):
            # YAML parsed a plain scalar string — not structured output
            log(WARNING, f"get_yaml_from_text(): YAML parsed to a plain string, not a dict/list")
            return None
        return None
    except yaml.YAMLError as e:
        # Keep the raw error for a potential final warning; don't log yet — recovery may succeed.
        first_error = e
        # Attempt JSON fallback — some models may still emit JSON despite YAML instructions
        try:
            fallback = json.loads(text)
            if isinstance(fallback, (dict, list)):
                log(INFO, f"get_yaml_from_text(): JSON fallback succeeded")
                return fallback
        except Exception:
            pass
        # Attempt colon-repair: re-quote plain scalar values that contain a bare colon.
        repaired = _repair_unquoted_colon_values(text)
        if repaired != text:
            try:
                result = yaml.safe_load(repaired)
                if isinstance(result, (dict, list)):
                    log(INFO, f"get_yaml_from_text(): colon-repair succeeded")
                    return result
            except yaml.YAMLError:
                pass
        # Attempt prose-strip: remove any trailing LLM reasoning/commentary block
        # (e.g. "Reasoning:\n..."), then re-apply colon-repair on the shorter text.
        stripped = _strip_prose_trailer(text)
        if stripped != text:
            stripped_repaired = _repair_unquoted_colon_values(stripped)
            try:
                result = yaml.safe_load(stripped_repaired)
                if isinstance(result, (dict, list)):
                    log(INFO, f"get_yaml_from_text(): prose-strip succeeded")
                    return result
            except yaml.YAMLError:
                pass
        # All recovery attempts exhausted — now log the original error as a warning.
        log(WARNING, f"get_yaml_from_text(): YAML parsing failed (all recovery attempts exhausted): {first_error}")
        return None


log_end_file(__file__)


if __name__ == '__main__':
    from src.utils.unit_test import init_unit_test
    from src.utils.log import log_init, log_exit
    init_unit_test()
    log_init()

    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        global passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))
            failed += 1

    try:
        # ── Test 1: None input ───────────────────────────────────────────────
        r = get_yaml_from_text(None)
        check("None input returns None", r is None)

        # ── Test 2: Empty string ─────────────────────────────────────────────
        r = get_yaml_from_text("")
        check("Empty string returns {}", r == {})

        # ── Test 3: Basic key-value pairs ────────────────────────────────────
        r = get_yaml_from_text("optional_language_used_by_user: en\noptional_require_exact_time: false")
        check("Basic YAML parsed", isinstance(r, dict), str(r))
        check("Language field correct", r.get("optional_language_used_by_user") == "en")
        check("Bool field correct (false→False)", r.get("optional_require_exact_time") == False)

        # ── Test 4: Markdown code fence stripping ────────────────────────────
        fenced = "```yaml\noptional_language_used_by_user: fr\n```"
        r = get_yaml_from_text(fenced)
        check("Markdown code fence stripped", isinstance(r, dict), str(r))
        check("Fenced value correct", r.get("optional_language_used_by_user") == "fr")

        # ── Test 5: Single-goal complex plan ─────────────────────────────────
        single_goal_yaml = """optional_complex_plan:
  goals:
    - goal_sequence_id: "Step-1"
      goal: "Fetch information about Charly the Dog"
      dependent_on_sequence_ids: []
      recommended_plan: "plan_rag_assistant_and_answer"
optional_language_used_by_user: "en"
optional_require_exact_time: false
"""
        r = get_yaml_from_text(single_goal_yaml)
        check("Single-goal plan parsed", isinstance(r, dict), str(r))
        goals = r.get("optional_complex_plan", {}).get("goals", [])
        check("Single-goal: one goal in list", len(goals) == 1)
        check("Single-goal: correct sequence id", goals[0].get("goal_sequence_id") == "Step-1")
        check("Single-goal: correct plan", goals[0].get("recommended_plan") == "plan_rag_assistant_and_answer")
        check("Single-goal: empty deps", goals[0].get("dependent_on_sequence_ids") == [])

        # ── Test 6: Multi-goal complex plan with dependencies ────────────────
        multi_goal_yaml = """optional_complex_plan:
  goals:
    - goal_sequence_id: "Step-1"
      goal: "Retrieve information about Charly from available documents"
      dependent_on_sequence_ids: []
      recommended_plan: "plan_rag_assistant_no_answer"
    - goal_sequence_id: "Step-2"
      goal: "Fetch the current temperature for Paris"
      dependent_on_sequence_ids: []
      recommended_plan: "plan_call_tool_but_no_answer"
      optional_recommended_tools:
        - "WeatherInfo"
    - goal_sequence_id: "Step-3"
      goal: "Provide a consolidated answer about Charly and the Paris temperature"
      dependent_on_sequence_ids:
        - "Step-1"
        - "Step-2"
      recommended_plan: "plan_simple_answer"
optional_language_used_by_user: "en"
optional_require_exact_time: true
"""
        r = get_yaml_from_text(multi_goal_yaml)
        check("Multi-goal plan parsed", isinstance(r, dict), str(r))
        goals = r.get("optional_complex_plan", {}).get("goals", [])
        check("Multi-goal: three goals", len(goals) == 3)
        check("Multi-goal: step 2 has tool", goals[1].get("optional_recommended_tools") == ["WeatherInfo"])
        check("Multi-goal: step 3 dep on step-1 and step-2",
              goals[2].get("dependent_on_sequence_ids") == ["Step-1", "Step-2"])
        check("Multi-goal: require_exact_time=true", r.get("optional_require_exact_time") == True)

        # ── Test 7: Pre-built plan id (no optional_complex_plan) ─────────────
        prebuilt_yaml = """optional_prebuilt_complex_plan_id: "pre-built-plan-deep-rag"
optional_language_used_by_user: "de"
"""
        r = get_yaml_from_text(prebuilt_yaml)
        check("Pre-built plan parsed", isinstance(r, dict), str(r))
        check("Pre-built plan id correct",
              r.get("optional_prebuilt_complex_plan_id") == "pre-built-plan-deep-rag")
        check("Pre-built: no optional_complex_plan key",
              "optional_complex_plan" not in r)

        # ── Test 8: JSON fallback ─────────────────────────────────────────────
        json_str = '{"optional_language_used_by_user": "it", "optional_require_exact_time": false}'
        r = get_yaml_from_text(json_str)
        check("JSON fallback parsed", isinstance(r, dict), str(r))
        check("JSON fallback value correct", r.get("optional_language_used_by_user") == "it")

        # ── Test 9: colon-repair — unquoted value with embedded colon-space ───
        colon_in_value_yaml = """optional_complex_plan:
  goals:
    - goal_sequence_id: Step-1
      goal: Process the two documents: "Cognome e Nome - Correzione 2 - Righe Errate.docx" (incorrect lines) and "Cognome e Nome - Correzione 2 - Righe Mancanti.docx" (missing lines)
      dependent_on_sequence_ids: []
      recommended_plan: plan_simple_answer
optional_language_used_by_user: it
"""
        r = get_yaml_from_text(colon_in_value_yaml)
        check("Colon-repair: parsed successfully", isinstance(r, dict), str(r))
        if isinstance(r, dict):
            goals = r.get("optional_complex_plan", {}).get("goals", [])
            check("Colon-repair: one goal", len(goals) == 1)
            check("Colon-repair: goal text preserved",
                  "Cognome e Nome" in (goals[0].get("goal") or ""))
            check("Colon-repair: language correct", r.get("optional_language_used_by_user") == "it")

        # ── Test 10: optional_doc_ids in complex plan ─────────────────────────
        docids_yaml = """optional_complex_plan:
  optional_doc_ids: "docid-abc123,docid-def456"
  goals:
    - goal_sequence_id: "Step-1"
      goal: "Analyse the specified documents"
      dependent_on_sequence_ids: []
      recommended_plan: "plan_rag_assistant_and_answer"
optional_language_used_by_user: "en"
"""
        r = get_yaml_from_text(docids_yaml)
        check("optional_doc_ids parsed", isinstance(r, dict), str(r))
        check("optional_doc_ids value correct",
              r.get("optional_complex_plan", {}).get("optional_doc_ids") == "docid-abc123,docid-def456")

        # ── Test 11: colon-repair — colon NOT followed by space (e.g. "(Epic):") ─
        # Reproduces the real-world failure: instructions_to_report containing
        # "ABC-123 (Epic): ...summary..." where the colon after (Epic) is NOT
        # followed by a space when the value ends there.
        jira_colon_yaml = """optional_complex_plan:
  goals:
  - goal_sequence_id: Step-1
    goal: Query the Epics dataset and summarise each item in English
    dependent_on_sequence_ids: []
    recommended_plan: plan_call_tool_and_answer
    optional_recommended_tools:
    - DatasetCreateReport
    optional_recommended_parameters:
      instructions_to_report: Summarise this item in a few sentences in English. You must start with the JIRA KEY (JIRA TYPE) and a ':'. For example: ABC-123 (Epic): ...summary...
      report_excludes_content: true
optional_language_used_by_user: en
optional_require_exact_time: false
"""
        r = get_yaml_from_text(jira_colon_yaml)
        check("JIRA colon-repair: parsed successfully", isinstance(r, dict), str(r))
        if isinstance(r, dict):
            goals = r.get("optional_complex_plan", {}).get("goals", [])
            check("JIRA colon-repair: one goal", len(goals) == 1)
            params = goals[0].get("optional_recommended_parameters") or {}
            instr = params.get("instructions_to_report") or ""
            check("JIRA colon-repair: instructions_to_report preserved",
                  "ABC-123 (Epic)" in instr)
            check("JIRA colon-repair: language correct", r.get("optional_language_used_by_user") == "en")

        # ── Test 12: prose-strip — LLM appends a Reasoning: block after YAML ─
        prose_trailer_yaml = """optional_complex_plan:
  goals:
  - goal_sequence_id: Step-1
    goal: Query the Epics dataset
    dependent_on_sequence_ids: []
    recommended_plan: plan_call_tool_and_answer
optional_language_used_by_user: en
optional_require_exact_time: false
Reasoning:
The user wants to summarize items in a dataset called Epics. According to the tool instructions:
- DatasetCreateReport should be used for any question about dataset content
- You must create a complex plan
Since the dataset is called Epics, I need to create a plan that:
1. Uses DatasetCreateReport to query the Epics dataset
2. Asks for a summary in English
"""
        r = get_yaml_from_text(prose_trailer_yaml)
        check("Prose-strip: parsed successfully", isinstance(r, dict), str(r))
        if isinstance(r, dict):
            check("Prose-strip: optional_complex_plan present",
                  "optional_complex_plan" in r)
            check("Prose-strip: Reasoning key NOT present", "Reasoning" not in r)
            check("Prose-strip: language correct", r.get("optional_language_used_by_user") == "en")

        # ── Test 13: combined — colon in value AND Reasoning: trailing prose ──
        # Reproduces the exact real-world failure from 2026-04-15.
        combined_yaml = """optional_complex_plan:
  goals:
  - goal_sequence_id: Step-1
    goal: Query the Epics dataset and summarise each item in English
    dependent_on_sequence_ids: []
    recommended_plan: plan_call_tool_and_answer
    optional_recommended_tools:
    - DatasetCreateReport
    optional_recommended_parameters:
      instructions_to_report: Summarise this item in a few sentences in English. You must start with the JIRA KEY (JIRA TYPE) and a ':'. For example: ABC-123 (Epic): ...summary...
      report_excludes_content: true
optional_language_used_by_user: en
optional_require_exact_time: false
Reasoning:
The user wants to summarize items in a dataset called Epics. According to the tool instructions for DatasetCreateReport:
- DatasetCreateReport should be used when the user has ANY QUESTION about the dataset content
- You must create a plan with a single step
Since the dataset is called Epics, I need to:
1. Query the Epics dataset
2. Return a summary per item
"""
        r = get_yaml_from_text(combined_yaml)
        check("Combined colon+prose: parsed successfully", isinstance(r, dict), str(r))
        if isinstance(r, dict):
            goals = r.get("optional_complex_plan", {}).get("goals", [])
            check("Combined colon+prose: one goal", len(goals) == 1)
            params = goals[0].get("optional_recommended_parameters") or {}
            instr = params.get("instructions_to_report") or ""
            check("Combined colon+prose: instructions_to_report preserved",
                  "ABC-123 (Epic)" in instr)
            check("Combined colon+prose: Reasoning key NOT present", "Reasoning" not in r)
            check("Combined colon+prose: language correct", r.get("optional_language_used_by_user") == "en")

        # ── Test 14: goal value with embedded double-quoted inline string ────
        # Reproduces the real-world failure from 2026-04-15: the goal plain scalar
        # contains "instructions_to_report: \"...\"" — an embedded key:value that
        # the YAML parser mistakes for a nested mapping key.
        embedded_quoted_yaml = """optional_complex_plan:
  goals:
  - goal_sequence_id: Step-1
    goal: Query dataset demo-test1-A and for each item determine if there is any animal in column 2. Use instructions_to_report: "For each item, answer the question 'is there any animal in column 2?' in one sentence. You must start with the item identifier and a ':'. For example: item-1: Yes, there is an animal in column 2."
    dependent_on_sequence_ids: []
    recommended_plan: plan_call_tool_and_answer
    optional_recommended_tools:
    - DatasetCreateReport
optional_language_used_by_user: en
optional_require_exact_time: false
"""
        r = get_yaml_from_text(embedded_quoted_yaml)
        check("Embedded-quotes: parsed successfully", isinstance(r, dict), str(r))
        if isinstance(r, dict):
            goals = r.get("optional_complex_plan", {}).get("goals", [])
            check("Embedded-quotes: one goal", len(goals) == 1)
            goal_text = goals[0].get("goal") or ""
            check("Embedded-quotes: goal contains instructions_to_report",
                  "instructions_to_report" in goal_text)
            check("Embedded-quotes: goal preserves example text", "item-1:" in goal_text)
            check("Embedded-quotes: language correct",
                  r.get("optional_language_used_by_user") == "en")

    except Exception as e:
        log(FATAL, f"Unexpected error during tests: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print(f"\nResults: {passed} passed, {failed} failed")
        log_exit()
