"""
Apertus LLM Provider Implementation (Apertus 8B / 70B served via an OpenAI-compatible endpoint)
"""

from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import copy
from typing import Dict, List

from src.core.llm.providers.provider_http import LlmHttpRequest
from src.utils.log import log, DEBUG

# The Apertus chat template accepts "system" only as the first message; later ones are sent as "user".
APERTUS_SECONDARY_SYSTEM_ROLE = "user"


class LlmApertus(LlmHttpRequest):
    """Apertus provider: HTTP provider that keeps "system" only at position 0."""

    def __init__(self):
        super().__init__()
        self._model_provider = "apertus"

    def _prepare_model_client(self, agent=None, purpose=None, endpoint=None):
        """Resolve the optional API token from llmconfig ({{ENV_VAR}} expansion supported)."""
        self._api_key = None
        if self.llmconfig and "token" in self.llmconfig and self.llmconfig["token"]:
            self._api_key = self._resolve_api_key(self.llmconfig["token"])
        return None

    def _sanitize_messages(self, messages: List[Dict]) -> List[Dict]:
        """Replace the role of every system message after the first one, without mutating the input."""
        result: List[Dict] = []
        found_first_system: bool = False

        for msg in super()._sanitize_messages(messages):
            if msg.get("role") == "system":
                if not found_first_system:
                    found_first_system = True
                    result.append(msg)
                else:
                    replacement = copy.deepcopy(msg)
                    replacement["role"] = APERTUS_SECONDARY_SYSTEM_ROLE
                    result.append(replacement)
            else:
                result.append(msg)

        log(DEBUG, "LlmApertus messages:\n" + "\n".join(
            f"  [{i}] role={m.get('role')} content={str(m.get('content', ''))[:120]}"
            for i, m in enumerate(result)
        ))

        return result


log_end_file(__file__)
