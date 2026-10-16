from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import traceback
from typing import Any, Dict, List

from src.core.extensions.classes.BaseExtension import BaseExtension


class Innovator(BaseExtension):
    """Render imported Innovator entities without exposing internal identifiers."""

    _HIDDEN_KEYS = {"embedding", "id", "ea_element_id", "elementId"}

    def _value(self, value: Any) -> Any:
        if isinstance(value, list) and value:
            if isinstance(value[0], dict) and "value" in value[0]:
                values = [str(item.get("value")) for item in value if item.get("value") not in (None, "")]
                return ", ".join(values)
            values = [str(item) for item in value if item not in (None, "")]
            return ", ".join(values)
        if isinstance(value, dict) and "value" in value:
            return value.get("value")
        return value

    def _display_name(self, attributes: Dict[str, Any]) -> str:
        name = self._value(attributes.get("name"))
        if name:
            return str(name)
        return str(
            self._value(attributes.get("id"))
            or self._value(attributes.get("entity_id"))
            or self._value(attributes.get("OriginUUID"))
            or "Innovator entity"
        )

    def _display_related_name(self, relationship: Dict[str, Any], include_ids: bool) -> str:
        related_name = self._value(
            relationship.get("related_name")
            or relationship.get("target_entity_name")
            or relationship.get("source_entity_name")
            or relationship.get("entity_name")
        )
        related_id = self._value(
            relationship.get("related_id")
            or relationship.get("target_entity_id")
            or relationship.get("source_entity_id")
            or relationship.get("entity_id")
            or relationship.get("id")
        )
        display_name = str(related_name or related_id or "")
        if include_ids and related_id and related_name:
            display_name += f" ({related_id})"
        return display_name

    def _format_relationships(self, attributes: Dict[str, Any], direction: str, name: str, include_ids: bool) -> List[str]:
        relationships = attributes.get(direction, [])
        if not isinstance(relationships, list):
            return []

        result: List[str] = []
        for relationship in relationships:
            if not isinstance(relationship, dict):
                continue
            relation = relationship.get("relationship") or relationship.get("relationship_label")
            related_name = self._display_related_name(relationship, include_ids)
            if relation and related_name:
                if direction == "outgoing_relationships":
                    result.append(f"{name} {relation} {related_name}.")
                else:
                    result.append(f"{related_name} {relation} {name}.")
        return result

    def str(self, attributes: Dict[str, Any], include_ids: bool = False) -> str:
        name = self._display_name(attributes).strip()
        entity_id = self._value(attributes.get("id") or attributes.get("entity_id"))
        display_name = name
        if include_ids and entity_id and self._value(attributes.get("name")):
            display_name += f" ({entity_id})"
        labels = attributes.get("labels") or []
        entity_type = attributes.get("label") or attributes.get("type")
        if not entity_type and isinstance(labels, list):
            entity_type = next((label for label in labels if label not in {"Entity", "Innovator"}), None)
        entity_type = entity_type or "Innovator entity"
        parts: List[str] = [f"{display_name} is a {entity_type}."]

        for key in ("description", "namespace", "OriginUUID"):
            value = self._value(attributes.get(key))
            if value:
                label = "Origin UUID" if key == "OriginUUID" else key.replace("_", " ")
                parts.append(f"{label}: {value}.")

        relationship_keys = {"outgoing_relationships", "incoming_relationships"}
        for key, value in attributes.items():
            if key in self._HIDDEN_KEYS or key in {"name", "label", "labels", "type", "description", "namespace", "OriginUUID"} | relationship_keys:
                continue
            value = self._value(value)
            if value and not isinstance(value, (dict, list)):
                parts.append(f"{key.replace('_', ' ')}: {value}.")

        parts.extend(self._format_relationships(attributes, "outgoing_relationships", display_name, include_ids))
        parts.extend(self._format_relationships(attributes, "incoming_relationships", display_name, include_ids))
        return "\n".join(parts)


log_end_file(__file__)


if __name__ == "__main__":
    from src.utils.log import FATAL, log, log_exit, log_init

    test_entities: List[Dict[str, Any]] = [
        {
            "id": "5a2ee9db_a24d_540d_a0dc_291dd3e97946",
            "labels": ["Diagram", "Entity", "Innovator"],
            "name": [{"value": "Fahrzeug zuruecknehmen"}],
            "namespace": [{"value": "Fahrzeug zuruecknehmen"}],
            "OriginUUID": [{"value": "5a2ee9db-a24d-540d-a0dc-291dd3e97946"}],
            "outgoing_relationships": [
                {
                    "relationship_label": "depicts",
                    "target_entity_name": [{"value": "Mietvertrag abrechnen"}],
                    "target_entity_id": "8c0950a8_583f_b816_362c_4f704878cda4",
                },
                {
                    "relationship_label": "depicts",
                    "target_entity_name": [{"value": "Schaeden erfassen"}],
                    "target_entity_id": "da072bab_1fb1_e97b_1d15_6d371410d7b8",
                },
            ],
        },
        {
            "id": "8c0950a8_583f_b816_362c_4f704878cda4",
            "labels": ["Diagram", "Entity", "Innovator"],
            "name": [{"value": "Mietvertrag abrechnen"}],
            "OriginUUID": [{"value": "8c0950a8-583f-b816-362c-4f704878cda4"}],
        },
        {
            "id": "da072bab_1fb1_e97b_1d15_6d371410d7b8",
            "labels": ["Diagram", "Entity", "Innovator"],
            "name": [{"value": "Schaeden erfassen"}],
            "OriginUUID": [{"value": "da072bab-1fb1-e97b-1d15-6d371410d7b8"}],
        },
        {
            "id": "b1b8b1cc_359e_66da_834c_0ef6bdfba89b",
            "labels": ["Viewpoints", "Entity", "Innovator"],
            "name": [{"value": "Views"}],
            "namespace": [{"value": "Unternehmensarchitekturmodell mit ArchiMate"}],
            "OriginUUID": [{"value": "b1b8b1cc-359e-66da-834c-0ef6bdfba89b"}],
        },
        {
            "id": "59779f74_e7b8_2f7c_5c06_a20833d405c3",
            "labels": ["Diagram", "Entity", "Innovator"],
            "name": [{"value": "Geschaeftsprozesssicht Rueckgabe"}],
            "namespace": [{"value": "Views"}],
            "OriginUUID": [{"value": "59779f74-e7b8-2f7c-5c06-a20833d405c3"}],
        },
        {
            "id": "1bcf48aa_e518_8cab_2ecd_505c1c385908",
            "labels": ["Diagram", "Entity", "Innovator"],
            "name": [{"value": "Innovator Enterprise Suite"}],
            "namespace": [{"value": "Mietwagenunternehmen"}],
            "OriginUUID": [{"value": "1bcf48aa-e518-8cab-2ecd-505c1c385908"}],
        },
    ]

    log_init()
    try:
        renderer = Innovator()
        print("\nInnovator rendering test\n" + "=" * 80)
        for entity in test_entities:
            print(renderer.str(entity))
        print("\nWith IDs enabled\n" + "=" * 80)
        print(renderer.str(test_entities[0], include_ids=True))

    except Exception as e:
        log(FATAL, f"Innovator unit test failed with exception: {e}. Traceback:\n{traceback.format_exc()}")
        
    finally:
        log_exit()


