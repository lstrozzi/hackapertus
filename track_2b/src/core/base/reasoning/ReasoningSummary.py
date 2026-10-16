from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import re
from src.core.base.reasoning.xml_utils import xml_cdata, LINE_SEPARATOR


class ReasoningSummary:
    """Represents a single language-specific summary for a ReasoningDivision.

    A division may hold multiple ReasoningSummary instances – one per supported
    language (EN, DE, FR, IT, …).  Exactly one should be flagged as master=True
    to designate the canonical/authoritative version.
    """

    def __init__(self, text: str = "", lang: str = "en", master: bool = False, version: int = 0) -> None:
        self._rs_text: str = text if text is not None else ""
        self._rs_lang: str = (lang or "en").lower().strip()
        self._rs_master: bool = bool(master)
        self._rs_version: int = max(0, int(version)) if version is not None else 0

    # ------------------------------------------------------------------
    # Getters / setters
    # ------------------------------------------------------------------

    def text(self) -> str:
        """Get the summary text."""
        return self._rs_text

    def set_text(self, text: str) -> None:
        """Set the summary text, normalising HTML line-breaks."""
        if text:
            text = re.sub(r"<br\s*/?>", LINE_SEPARATOR, text)
        self._rs_text = text if text is not None else ""

    def lang(self) -> str:
        """Get the ISO 639-1 language code (e.g. 'en', 'de', 'fr', 'it')."""
        return self._rs_lang

    def set_lang(self, lang: str) -> None:
        """Set the language code."""
        self._rs_lang = (lang or "en").lower().strip()

    def master(self) -> bool:
        """Return True if this is the master (primary) summary."""
        return self._rs_master

    def set_master(self, master: bool) -> None:
        """Set the master flag."""
        self._rs_master = bool(master)

    def version(self) -> int:
        """Get the version counter (auto-incremented on each overwrite)."""
        return self._rs_version

    def set_version(self, version: int) -> None:
        """Set the version counter (must be a non-negative integer)."""
        self._rs_version = max(0, int(version)) if version is not None else 0

    def increment_version(self) -> None:
        """Increment the version counter by 1."""
        self._rs_version += 1

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "text": self._rs_text,
            "lang": self._rs_lang,
            "master": self._rs_master,
            "version": self._rs_version,
        }

    def to_xml(self, escape: bool = False) -> str:
        """Render as an indented <summary> element (caller wraps in <summaries>)."""
        lang_attr = f' lang="{self._rs_lang}"' if self._rs_lang else ""
        master_attr = ' master="true"' if self._rs_master else ""
        version_attr = f' version="{self._rs_version}"' if self._rs_version else ""
        lines = [
            f'        <summary{lang_attr}{master_attr}{version_attr}>',
            f'            {xml_cdata(self._rs_text)}',
            '        </summary>',
        ]
        return '\n'.join(lines)

    @classmethod
    def from_dict(cls, d: dict) -> 'ReasoningSummary':
        return cls(
            text=d.get("text", ""),
            lang=d.get("lang", "en"),
            master=bool(d.get("master", False)),
            version=int(d.get("version", 0)),
        )

    @classmethod
    def are_translations_up_to_date(cls, summaries: list['ReasoningSummary']) -> bool:
        """Check if all non-master summaries have version >= master summary version.

        Args:
            summaries: List of ReasoningSummary objects to check

        Returns:
            True if:
            - No master summary exists (no baseline to compare against), OR
            - All non-master summaries have version >= master version
            False otherwise.
        """
        if not summaries:
            return True  # Empty list is considered "up to date"

        # Find the master summary
        master_summary = None
        for s in summaries:
            if s.master():
                master_summary = s
                break

        # If no master summary, consider all translations up to date
        if master_summary is None:
            return True

        master_version = master_summary.version()

        # Check all non-master summaries
        for s in summaries:
            if not s.master():
                if s.version() < master_version:
                    return False

        return True

    def __repr__(self) -> str:
        return (
            f"ReasoningSummary(lang={self._rs_lang!r}, master={self._rs_master}, "
            f"version={self._rs_version}, text={self._rs_text[:40]!r})"
        )


log_end_file(__file__)
