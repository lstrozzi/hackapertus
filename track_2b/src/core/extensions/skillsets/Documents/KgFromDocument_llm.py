from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

from typing import Any, List, Optional

from src.config.constants import IS_DEBUG
from src.core.base.enums import AgentReasoningEnum
from src.core.llm.config import MAX_COMPLETION_TOKENS_MEDIUM, MAX_COMPLETION_TOKENS_SMALL
from src.core.llm.manager import LlmManager
from src.core.llm import get_yaml_from_text
from src.utils.log import log, DEBUG, INFO, WARNING


SKILL_ID: str = "KgFromDocument"


_REASONING_LEVEL_EXTRACTION = AgentReasoningEnum.FAST   # pure pattern matching
_REASONING_LEVEL_REASONING  = AgentReasoningEnum.SIMPLE # dedup / relationships: light thinking, some reasoning beyond surface-level patterns, but still focused on extraction rather than deep inference
_CACHE = True # False if IS_DEBUG else True
if _CACHE == False:
    log(WARNING, "KGFromDocument: LLM caching is DISABLED (debug mode); expect higher latency and more API calls to the LLM provider")


def _get_model() -> Optional[str]:
    """Return the model configured for this skill, resolved dynamically and falling back to DEFAULT."""
    return LlmManager.instance().get_model_by_skill_id(SKILL_ID)

def _parse_llm_response(llm_response: Any, top_key: str) -> list:
    if llm_response is None or llm_response.response is None:
        return []
    data = llm_response.response
    if isinstance(data, str):
        data = get_yaml_from_text(data)
    if isinstance(data, dict):
        result = data.get(top_key)
        return result if isinstance(result, list) else []
    return []


def _extract_entity_name(name) -> str:
    """Return a plain string from an entity name field that may be a Neo4j value list."""
    if isinstance(name, list) and name:
        first = name[0]
        if isinstance(first, dict):
            return str(first.get("value", first))
        return str(first)
    if isinstance(name, dict):
        return str(name.get("value", name))
    return str(name) if name is not None else ""


def _call_llm_extract_entities(
    document_text: str,
    entity_label: str,
    entity_description: str,
    document_context: str = "",
    ontology_description: str = "",
    existing_entities: List[dict] = [],
) -> List[dict]:
    """Ask the LLM to list all instances of *entity_label* found in the document.

    Returns a list of dicts: [{entity_id, name}].
    """
    system_prompt = f"""You are a knowledge graph extraction assistant.
Task: extract ALL "{entity_label}" entities from <TEXT_BLOCK_IN_FOCUS>.
Entity definition: {entity_description or entity_label}

Output YAML:
entities:
  - entity_id: <lowercase_with_underscores>
    name: <exact name from TEXT_BLOCK_IN_FOCUS>

RULES:
- Source: extract ONLY from <TEXT_BLOCK_IN_FOCUS>. Do NOT include entities that appear only in <DOCUMENT_CONTEXT>.
- <DOCUMENT_CONTEXT> is provided only to help you identify wrong-type entities to exclude.
- Output your YAML immediately once you have identified all qualifying entities. Do not re-examine the same evidence multiple times.

EXAMPLE 1 — Extract "Flower" entities
TEXT_BLOCK_IN_FOCUS: "Charly barked loudly as Tom handed the letter to Sarah. There was a vase of roses on the table, and some margheritas were visible out of the window."
DOCUMENT_CONTEXT: "Charly is a dog owned by Tom and Sarah, who are a couple. They live in a house with a garden where they grow various flowers."
→ Roses and Margheritas are flowers → include. Tom and Sarah are persons (not flowers) → exclude. Charly is a dog (not a flower) → exclude.
entities:
  - entity_id: roses
    name: Roses
  - entity_id: margheritas
    name: Margheritas

EXAMPLE 2 — Extract "Person" entities
TEXT_BLOCK_IN_FOCUS: "Charly knocked over his water bowl."
DOCUMENT_CONTEXT: "Charly is a dog owned by Tom and Sarah, who are a couple. They live in a house with a garden where they grow various flowers."
→ Charly is a dog (not a person) → exclude. No persons appear in this block.
entities: []
"""

    user_prompt = document_text
    messages = [{"role": "system", "content": system_prompt}]
    if existing_entities:
        existing_lines = "\n".join(f"- entity_id: {e['entity_id']}, name: {_extract_entity_name(e['name'])}" for e in existing_entities)
        messages.append({"role": "system", "content": f"EXISTING {entity_label.upper()} ENTITIES ALREADY IN THIS KNOWLEDGE GRAPH COMMUNITY:\n{existing_lines}\nIf the same entity appears in <TEXT_BLOCK_IN_FOCUS>, reuse its entity_id exactly."})
    if document_context:
        messages.append({"role": "system", "content": f"<DOCUMENT_CONTEXT>\n{document_context}\n</DOCUMENT_CONTEXT>"})
    messages.append({"role": "user", "content": user_prompt})
    llm_response = LlmManager.instance().chat_completions(
        messages=messages,
        model_input=_get_model(),
        max_completion_tokens=MAX_COMPLETION_TOKENS_MEDIUM,
        maximum_reasoning_required=_REASONING_LEVEL_EXTRACTION,
        cache=_CACHE
    )
    entities = _parse_llm_response(llm_response, "entities")
    log(INFO, f"KGFromDocument: LLM found {len(entities)} '{entity_label}' entity instances")
    return entities


def _call_llm_extract_entity_attribute(
    document_text: str,
    entity_label: str,
    entity_ids: List[str],
    entity_names: List[str],
    attribute_name: str,
    attribute_description: str,
    document_context: str = "",
    existing_attribute_values: dict = None,
) -> List[dict]:
    """Ask the LLM to extract a single *attribute_name* value for each entity instance from the text block.

    Returns a list of dicts: [{entity_id, value, source_id}].
    """
    entity_lines = "\n".join(
        f"- entity_id: {eid}, name: {name}"
        for eid, name in zip(entity_ids, entity_names)
    )
    system_prompt = f"""You are a knowledge graph extraction assistant.
Task: extract the "{attribute_name}" attribute ({attribute_description or attribute_name}) for each listed "{entity_label}" entity from <TEXT_BLOCK_IN_FOCUS>.

Output YAML:
values:
  - entity_id: <entity_id_from_list>
    value: <extracted value>
    source_id: <id_attribute_of_the_html_element_where_the_value_was_found>

RULES:
- Source: extract ONLY from <TEXT_BLOCK_IN_FOCUS>. Do NOT use <DOCUMENT_CONTEXT> as a source.
- <DOCUMENT_CONTEXT> is provided only to help identify entities and disambiguate context.
- Skip an entity entirely if "{attribute_name}" is not stated in <TEXT_BLOCK_IN_FOCUS>.
- Use exact entity_id values from the provided entity list.
- Output values in their canonical type: for numeric attributes (e.g. age, count, year), output the digit form (e.g. `12` not `twelve years old`); for dates use YYYY-MM-DD; for booleans use `true`/`false`.
- source_id: copy the `id` attribute of the HTML element in <TEXT_BLOCK_IN_FOCUS> that contains the evidence. Use the most specific (innermost) element id available. Omit source_id if the element has no id.
- If already-known values are provided for an entity, extract ADDITIONAL or NEW *facts* only. Do NOT extract information whose meaning is already captured by an already-known value — even if the source text uses different wording. When in doubt, skip.
- For each (entity_id, source_id) pair output at most ONE entry. If multiple pieces of information from the same HTML element are new, consolidate them into a single value for that element.
- Do NOT explain your reasoning. Output ONLY the YAML.
- When no values are found, output exactly: values: []  (inline — NOT values: on one line then [] on the next)

EXAMPLE 1 — attribute "physical_description" for Person, entity charly
TEXT_BLOCK_IN_FOCUS:
<p id="para-3">Charly, a big fluffy dog with a brown coat, wagged his tail.</p>
<p id="para-4">Tom handed the letter to Sarah and sat down.</p>
→ "big fluffy dog with a brown coat" is in para-3 → include with source_id para-3. No description for Tom or Sarah in this block → skip them.
values:
  - entity_id: charly
    value: big fluffy dog with a brown coat
    source_id: para-3

EXAMPLE 2 — attribute "physical_description" for Person, entity charly; already-known: "big fluffy dog"
TEXT_BLOCK_IN_FOCUS: <p id="para-7">Charly's brown eyes caught the light as he ran.</p>
→ "brown eyes" is NEW (not in already-known) → include.
values:
  - entity_id: charly
    value: brown eyes
    source_id: para-7

EXAMPLE 3 — attribute "role" for Person, entity charly; block does not mention his role
TEXT_BLOCK_IN_FOCUS: <p id="para-9">Charly knocked over his water bowl.</p>
→ No role information in this block → skip charly entirely.
values: []

EXAMPLE 4 — attribute "what_happened" for Event, entity the_incident; already-known: "Bob leaped onto the counter and knocked over the milk with his tail"
TEXT_BLOCK_IN_FOCUS: <p id="para-7">Bob leaped onto the kitchen counter with a mischievous twinkle in his eye. In his excitement, his tail swept across the counter, knocking over a cup of milk.</p>
→ "mischievous twinkle in his eye" and "tail swept the counter, knocking over milk" describe the same scene already captured by the already-known value. Different wording, same facts → no new information.
values: []

EXAMPLE 5 — attribute "what_happened" for Event, entity the_incident; multiple new facts in one element
TEXT_BLOCK_IN_FOCUS: <p id="para-10">Tom dropped the vase. It shattered on the floor. Sarah gasped and rushed over.</p>
→ All three facts are new and come from para-10 → output ONE consolidated entry, not three separate ones.
values:
  - entity_id: the_incident
    value: Tom dropped the vase, which shattered on the floor; Sarah gasped and rushed over.
    source_id: para-10"""
    user_prompt = (
        f'Entities:\n{entity_lines}\n\n'
        f'{document_text}'
    )
    messages = [{"role": "system", "content": system_prompt}]
    if existing_attribute_values:
        lines = []
        for eid, vals in existing_attribute_values.items():
            if vals:
                known = "; ".join(vals) if isinstance(vals, list) else str(vals)
                lines.append(f"- entity_id: {eid}, already-known {attribute_name}: {known}")
        if lines:
            messages.append({"role": "system", "content": f"ALREADY-KNOWN VALUES FOR '{attribute_name.upper()}' (extract ADDITIONAL information only; do not repeat):\n" + "\n".join(lines)})
    if document_context:
        messages.append({"role": "system", "content": f"<DOCUMENT_CONTEXT>\n{document_context}\n</DOCUMENT_CONTEXT>"})
    messages.append({"role": "user", "content": user_prompt})
    llm_response = LlmManager.instance().chat_completions(
        messages=messages,
        model_input=_get_model(),
        max_completion_tokens=MAX_COMPLETION_TOKENS_MEDIUM,  # chunk may yield many entities with verbose values
        maximum_reasoning_required=_REASONING_LEVEL_EXTRACTION,
        cache=_CACHE
    )
    values = _parse_llm_response(llm_response, "values")
    log(DEBUG, f"KGFromDocument: LLM extracted {len(values)} values for attribute '{attribute_name}' of '{entity_label}'")
    return values


def _call_llm_extract_relationships(
    document_text: str,
    rel_label: str,
    rel_description: str,
    source_label: str,
    target_label: str,
    existing_entity_ids: List[str],
    document_context: str = "",
    ontology_description: str = "",
) -> List[dict]:
    """Ask the LLM to list all relationship instances of *rel_label* in the document.

    Returns a list of dicts: [{source_entity_id, target_entity_id, source_id}].
    """
    entity_ids_str = ", ".join(existing_entity_ids) if existing_entity_ids else "(none)"
    system_prompt = f"""You are a knowledge graph extraction assistant.
Task: extract ALL "{rel_label}" relationships ({rel_description or rel_label}) from <TEXT_BLOCK_IN_FOCUS>.
Direction: {source_label} → {target_label}
Known entity IDs (use ONLY these for source and target): {entity_ids_str}

Output YAML:
relationships:
  - source_entity_id: <entity_id_from_known_list>
    target_entity_id: <entity_id_from_known_list>
    source_id: <id_attribute_of_the_html_element_where_the_evidence_was_found>

RULES:
- Source: extract ONLY from <TEXT_BLOCK_IN_FOCUS>. Do NOT use <DOCUMENT_CONTEXT> as a source.
- <DOCUMENT_CONTEXT> is provided only to help disambiguate entities.
- Use ONLY entity IDs from the known list for both source and target.
- source_id: copy the `id` attribute of the HTML element in <TEXT_BLOCK_IN_FOCUS> that contains the evidence. Use the most specific (innermost) element id available. Omit source_id if the element has no id.
- Do NOT explain your reasoning. Output ONLY the YAML.
- When no relationships are found, output exactly: relationships: []  (inline — NOT relationships: on one line then [] on the next)

EXAMPLE 1 — Extract "OWNED_BY" relationships (Animal → Person)
TEXT_BLOCK_IN_FOCUS: <p id="para-5">Charly wagged his tail as Tom, his devoted owner, clipped on the leash.</p>
→ "OWNED_BY": charly (Animal) → tom (Person), evidence in para-5
relationships:
  - source_entity_id: charly
    target_entity_id: tom
    source_id: para-5

EXAMPLE 2 — Extract "OWNED_BY" relationships; block does not mention ownership
TEXT_BLOCK_IN_FOCUS: <p id="para-9">The park was busy. Tom threw the ball and Charly ran after it.</p>
→ No ownership relationship explicitly stated in this block.
relationships: []"""
    user_prompt = document_text
    messages = [{"role": "system", "content": system_prompt}]
    if ontology_description:
        messages.append({"role": "system", "content": f"FULL ONTOLOGY SCHEMA:\n{ontology_description}"})
    if document_context:
        messages.append({"role": "system", "content": f"<DOCUMENT_CONTEXT>\n{document_context}\n</DOCUMENT_CONTEXT>"})
    messages.append({"role": "user", "content": user_prompt})
    llm_response = LlmManager.instance().chat_completions(
        messages=messages,
        model_input=_get_model(),
        max_completion_tokens=MAX_COMPLETION_TOKENS_SMALL,
        maximum_reasoning_required=_REASONING_LEVEL_REASONING,
        cache=_CACHE
    )
    relationships = _parse_llm_response(llm_response, "relationships")
    log(INFO, f"KGFromDocument: LLM found {len(relationships)} '{rel_label}' relationship instances")
    return relationships


def _call_llm_extract_relationship_attribute(
    document_text: str,
    rel_label: str,
    relationships: List[dict],
    attribute_name: str,
    attribute_description: str,
    document_context: str = "",
    ontology_description: str = "",
) -> List[dict]:
    """Ask the LLM to extract a single *attribute_name* value for each relationship instance.

    Returns a list of dicts: [{source_entity_id, target_entity_id, value}].
    """
    rel_lines = "\n".join(
        f"- {r['source_entity_id']} -> {r['target_entity_id']}"
        for r in relationships
    )
    system_prompt = f"""Extract attribute "{attribute_name}" ({attribute_description or attribute_name}) for each listed "{rel_label}" relationship.
Source: <TEXT_BLOCK_IN_FOCUS> only. Use exact source/target IDs from list. Skip if value not explicitly stated. <DOCUMENT_CONTEXT> is context only.
Example — attribute "since" for OWNED_BY (charly → tom): Tom had adopted Charly three years ago → value: three years ago
Output YAML:
values:
  - source_entity_id: charly
    target_entity_id: tom
    value: three years ago"""
    user_prompt = (
        f'Relationships:\n{rel_lines}\n\n'
        f'{document_text}'
    )
    messages = [{"role": "system", "content": system_prompt}]
    if ontology_description:
        messages.append({"role": "system", "content": f"FULL ONTOLOGY SCHEMA:\n{ontology_description}"})
    if document_context:
        messages.append({"role": "system", "content": f"<DOCUMENT_CONTEXT>\n{document_context}\n</DOCUMENT_CONTEXT>"})
    messages.append({"role": "user", "content": user_prompt})
    llm_response = LlmManager.instance().chat_completions(
        messages=messages,
        model_input=_get_model(),
        max_completion_tokens=MAX_COMPLETION_TOKENS_SMALL,
        maximum_reasoning_required=_REASONING_LEVEL_REASONING,
        cache=_CACHE
    )
    values = _parse_llm_response(llm_response, "values")
    log(DEBUG, f"KGFromDocument: LLM extracted {len(values)} values for attribute '{attribute_name}' of relationship '{rel_label}'")
    return values


def _call_llm_deduplicate_entities(
    entity_label: str,
    unique_entities: List[dict],
    document_context: str = "",
) -> tuple:
    """Ask the LLM to identify duplicate entities and wrong-type entities.

    Input: List of dicts [{entity_id, name}] already deduplicated by Python.
    Returns: (dedup_map, elements_to_remove) where dedup_map maps entity_id -> canonical_entity_id
             and elements_to_remove is a list of entity_ids that do not belong to entity_label.
    """
    if len(unique_entities) <= 1:
        return {ent["entity_id"]: ent["entity_id"] for ent in unique_entities}, []

    entity_lines = "\n".join(
        f"- entity_id: {e['entity_id']}, name: {e['name']}"
        for e in unique_entities
    )
    system_prompt = f"""Deduplicate "{entity_label}" entities using <DOCUMENT_CONTEXT>.
Merge only if clearly the same real-world entity; remove only if clearly wrong type. When in doubt, do nothing.
Examples:
  bob (a cat) among Persons → elements_to_remove: [bob]
  charly + charly_the_poodle (alias) → duplicates: [{{primary_entity_id: charly, duplicate_entity_ids: [charly_the_poodle]}}]
  no issues → duplicates: [], elements_to_remove: []
Output YAML:
duplicates: []
elements_to_remove: []"""

    user_prompt = (
        f'Entities:\n{entity_lines}\n\n'
        f'Identify duplicates and flag any entity that is NOT a {entity_label}.'
    )
    messages = [{"role": "system", "content": system_prompt}]
    if document_context:
        messages.append({"role": "system", "content": f"<DOCUMENT_CONTEXT>\n{document_context}\n</DOCUMENT_CONTEXT>"})
    messages.append({"role": "user", "content": user_prompt})
    llm_response = LlmManager.instance().chat_completions(
        messages=messages,
        model_input=_get_model(),
        max_completion_tokens=MAX_COMPLETION_TOKENS_SMALL,
        maximum_reasoning_required=_REASONING_LEVEL_REASONING,
        cache=_CACHE
    )
    duplicates = _parse_llm_response(llm_response, "duplicates")
    elements_to_remove: List[str] = _parse_llm_response(llm_response, "elements_to_remove")

    # Build deduplication mapping: entity_id -> canonical_entity_id
    dedup_map = {ent["entity_id"]: ent["entity_id"] for ent in unique_entities}
    for dup_group in duplicates:
        primary = dup_group.get("primary_entity_id")
        duplicates_list = dup_group.get("duplicate_entity_ids", [])
        if primary:
            for dup_id in duplicates_list:
                if dup_id in dedup_map:
                    dedup_map[dup_id] = primary
                    log(INFO, f"KGFromDocument: LLM deduplicated '{dup_id}' -> '{primary}'")

    log(INFO, f"KGFromDocument: LLM deduplication found {len([d for d in duplicates if d.get('duplicate_entity_ids')])} groups with duplicates, {len(elements_to_remove)} wrong-type entities to remove")
    return dedup_map, elements_to_remove


log_end_file(__file__)