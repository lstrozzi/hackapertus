from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import re
from src.core.base.reasoning.xml_utils import xml_escape, LINE_SEPARATOR


class ReasoningTitle:
    """Represents a single language-specific title (or short title) for a ReasoningDivision.

    A division may hold multiple ReasoningTitle instances – one per supported
    language (EN, DE, FR, IT, …).  Exactly one should be flagged as master=True
    to designate the canonical/authoritative version.

    Used for both the <titles>/<title> and <shortTitles>/<shortTitle> constructs.
    """

    def __init__(self, text: str = "", lang: str = "en", master: bool = False) -> None:
        self._rt_text: str = text if text is not None else ""
        self._rt_lang: str = (lang or "en").lower().strip()
        self._rt_master: bool = bool(master)

    # ------------------------------------------------------------------
    # Getters / setters
    # ------------------------------------------------------------------

    def text(self) -> str:
        """Get the title text."""
        return self._rt_text

    def set_text(self, text: str) -> None:
        """Set the title text, normalising HTML line-breaks."""
        if text:
            text = re.sub(r"<br\s*/?>", LINE_SEPARATOR, text)
        self._rt_text = text if text is not None else ""

    def lang(self) -> str:
        """Get the ISO 639-1 language code (e.g. 'en', 'de', 'fr', 'it')."""
        return self._rt_lang

    def set_lang(self, lang: str) -> None:
        """Set the language code."""
        self._rt_lang = (lang or "en").lower().strip()

    def master(self) -> bool:
        """Return True if this is the master (primary) title."""
        return self._rt_master

    def set_master(self, master: bool) -> None:
        """Set the master flag."""
        self._rt_master = bool(master)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "text": self._rt_text,
            "lang": self._rt_lang,
            "master": self._rt_master,
        }

    def to_xml(self, escape: bool = False, element_name: str = "title") -> str:
        """Render as an indented title element (caller wraps in <titles> or <shortTitles>)."""
        lang_attr = f' lang="{self._rt_lang}"' if self._rt_lang else ""
        master_attr = ' master="true"' if self._rt_master else ""
        text_escaped = xml_escape(self._rt_text, escape)
        lines = [
            f'        <{element_name}{lang_attr}{master_attr}>',
            f'            {text_escaped}',
            f'        </{element_name}>',
        ]
        return '\n'.join(lines)

    @classmethod
    def from_dict(cls, d: dict) -> 'ReasoningTitle':
        return cls(
            text=d.get("text", ""),
            lang=d.get("lang", "en"),
            master=bool(d.get("master", False)),
        )

    def __repr__(self) -> str:
        return (
            f"ReasoningTitle(lang={self._rt_lang!r}, master={self._rt_master}, "
            f"text={self._rt_text[:40]!r})"
        )


log_end_file(__file__)
