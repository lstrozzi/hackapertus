"""
LLM Type definitions, enums, and data classes.
"""

import json
from typing import Optional, Union
from src.core.base.enums import AgentReasoningEnum


# Reasoning levels mapping
reasoning_levels = {
    AgentReasoningEnum.NONE:    -1,
    AgentReasoningEnum.MINIMAL: 0,
    AgentReasoningEnum.FAST:    1,
    AgentReasoningEnum.SIMPLE:  2,
    AgentReasoningEnum.MEDIUM:  3,
    AgentReasoningEnum.COMPLEX: 4
}


class AgentRagSettings:
    """
    RAG settings with individual boolean switches for each setting type.
    Replaces the old AgentRagSettingsEnum that combined multiple settings.
    """
    internet: bool
    dossier: bool
    documents: bool
    knowledge_graph: bool
    mira: bool
    innovator_model: Optional[str]  # "repository/modelUuid" selected in the UI
    knowledge_graph_community_id: Optional[str]
    community_ids: Optional[list[str]]
    
    def __init__(self, internet: bool = True, dossier: bool = False, documents: bool = True, knowledge_graph: bool = True, mira: bool = False, innovator_model: Optional[str] = None, knowledge_graph_community_id: Optional[str] = None, community_ids: Optional[list[str]] = None):
        self.internet = internet
        self.dossier = dossier
        self.documents = documents
        self.knowledge_graph = knowledge_graph
        self.mira = mira
        self.innovator_model = innovator_model
        self.knowledge_graph_community_id = knowledge_graph_community_id
        self.community_ids = community_ids
    
    @classmethod
    def from_dict(cls, data: Union[dict, str, object]) -> 'AgentRagSettings':
        """Create AgentRagSettings from:
        - a dict (snake_case or camelCase keys)
        - a JSON string representing the dict
        - a legacy enum-style string (e.g. 'documentsinternet')
        - an object with attributes (e.g. coming from a deserialized C# payload)

        If data is None, returns defaults (internet=True, dossier=False, documents=True, knowledge_graph=True).
        """
        # Default: all enabled except dossier
        if data is None:
            return cls()

        # If it's already an AgentRagSettings instance, return it
        if isinstance(data, cls):
            return data

        # If it's a plain string, try JSON decode first, then legacy names
        if isinstance(data, str):
            s = data.strip()
            # try parse JSON
            if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
                try:
                    parsed = json.loads(s)
                    data = parsed
                except Exception:
                    # fallthrough to legacy parsing
                    pass

        # If after attempting JSON decode it's still a string, interpret legacy enum-like values
        if isinstance(data, str):
            key = data.strip().lower()
            mapping = {
                'none': cls(False, False, False, False),
                'internet': cls(True, False, False, False),
                'dossier': cls(False, True, False, False),
                'documents': cls(False, False, True, False),
                'documentsinternet': cls(True, False, True, False),
                'documentskg': cls(False, False, True, True),
                'documentsinternetkg': cls(True, False, True, True),
            }
            if key in mapping:
                return mapping[key]

            # best-effort: check for keywords
            return cls(
                internet=('internet' in key),
                dossier=('dossier' in key),
                documents=('documents' in key),
                knowledge_graph=('kg' in key or 'knowledge' in key or 'knowledge_graph' in key)
            )

        # If it's an object with attributes (e.g. deserialized C# object), try to read attributes
        if not isinstance(data, dict):
            # support both PascalCase and camelCase attribute names
            def attr_get(obj, *names, default=None):
                for n in names:
                    if hasattr(obj, n):
                        return getattr(obj, n)
                return default
            internet_val = attr_get(data, 'internet', 'Internet')
            dossier_val = attr_get(data, 'dossier', 'Dossier')
            documents_val = attr_get(data, 'documents', 'Documents')
            kg_val = attr_get(data, 'knowledge_graph', 'knowledgeGraph', 'KnowledgeGraph')

            # helper to coerce common string booleans to actual booleans
            def to_bool(v):
                if v is None:
                    return None
                if isinstance(v, bool):
                    return v
                if isinstance(v, str):
                    s = v.strip().lower()
                    if s in ('1', 'true', 'yes', 'on'):
                        return True
                    if s in ('0', 'false', 'no', 'off'):
                        return False
                    # fallthrough: non-empty string -> True
                    return bool(s)
                try:
                    return bool(v)
                except Exception:
                    return None

            # if any attribute found, use those (fall back to True when None)
            mira_val = attr_get(data, 'mira', 'Mira')
            innovator_model_val = attr_get(data, 'innovator_model', 'innovatorModel', 'InnovatorModel')
            knowledge_graph_community_id_val = attr_get(data, 'knowledge_graph_community_id', 'knowledgeGraphCommunityId', 'KnowledgeGraphCommunityId')
            community_ids_val = attr_get(data, 'community_ids', 'knowledge_graph_community_ids', 'knowledgeGraphCommunityIds', 'KnowledgeGraphCommunityIds')
            if internet_val is not None or dossier_val is not None or documents_val is not None or kg_val is not None or mira_val is not None:
                i = to_bool(internet_val)
                d = to_bool(dossier_val)
                doc = to_bool(documents_val)
                kg = to_bool(kg_val)
                m = to_bool(mira_val)
                return cls(
                    internet=i if i is not None else True,
                    dossier=d if d is not None else False,
                    documents=doc if doc is not None else True,
                    knowledge_graph=kg if kg is not None else True,
                    mira=m if m is not None else True,
                    innovator_model=str(innovator_model_val) if innovator_model_val else None,
                    knowledge_graph_community_id=str(knowledge_graph_community_id_val) if knowledge_graph_community_id_val else None,
                    community_ids=list(community_ids_val) if community_ids_val else None
                )

            # no recognized attributes -> fall back to init defaults
            return cls()

        # If we reach here, data is expected to be dict-like. If not, return defaults.
        if not isinstance(data, dict):
            return cls()

        return cls(
            internet=data.get('internet', True),
            dossier=data.get('dossier', False),  # Default False for dossier
            documents=data.get('documents', True),
            knowledge_graph=data.get('knowledge_graph', data.get('knowledgeGraph', True)),
            mira=data.get('mira', data.get('Mira', True)),
            innovator_model=data.get('innovator_model', data.get('innovatorModel', data.get('InnovatorModel'))) or None,
            knowledge_graph_community_id=data.get('knowledge_graph_community_id', data.get('knowledgeGraphCommunityId', data.get('KnowledgeGraphCommunityId'))) or None,
            community_ids=data.get('community_ids', data.get('knowledge_graph_community_ids', data.get('knowledgeGraphCommunityIds'))) or None
        )
    
    @classmethod
    def from_string(cls, rag_string: str) -> 'AgentRagSettings':
        """Parse RAG settings from string representation."""
        if not rag_string or rag_string.strip() == "":
            return cls()
            
        # Parse comma-separated list of settings
        enabled_settings = {setting.strip().lower() for setting in rag_string.split(',') if setting.strip()}
        
        return cls(
            internet='internet' in enabled_settings,
            dossier='dossier' in enabled_settings,
            documents='documents' in enabled_settings,
            knowledge_graph='knowledge_graph' in enabled_settings or 'kg' in enabled_settings
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        result: dict = {
            'internet': self.internet,
            'dossier': self.dossier,
            'documents': self.documents,
            'knowledge_graph': self.knowledge_graph,
            'mira': self.mira,
        }
        if self.innovator_model:
            result['innovator_model'] = self.innovator_model
        if self.knowledge_graph_community_id:
            result['knowledge_graph_community_id'] = self.knowledge_graph_community_id
        if self.community_ids:
            result['community_ids'] = self.community_ids
        return result
    
    def __str__(self) -> str:
        enabled = []
        if self.internet:
            enabled.append("internet")
        if self.dossier:
            enabled.append("dossier")
        if self.documents:
            enabled.append("documents")
        if self.knowledge_graph:
            enabled.append("knowledge_graph")
        if self.mira:
            enabled.append("mira")
        return f"AgentRagSettings: [{', '.join(enabled) if enabled else 'none'}]"


# JSON Schema definitions
json_schema_tool_call_response = {
    "name": "tool_call_response",
    "schema": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "role": { "type": "string" },
                "content": { "type": "string" },
                "function_call": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "arguments": { "type": "string" }
                    },
                    "required": ["name", "arguments"]
                }
            },
            "required": ["role", "content", "function_call"]
        }
    },
    "strict": True
}