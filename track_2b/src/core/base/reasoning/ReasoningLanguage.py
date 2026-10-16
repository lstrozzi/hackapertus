from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)


class ReasoningLanguage:
    """Represents a single language entry in a ReasoningDocument's <languages> header block.

    Each entry declares one supported language for the document with an optional
    translation type.  Exactly one entry should have master=True.

    Attributes:
        lang:   ISO 639-1 language code, e.g. "en", "de", "fr", "it" (stored lowercase).
        master: True when this is the source / primary language of the document.
        type:   For non-master languages: "auto" (auto-translate) or "disabled"
                (language declared but translation suppressed).  None for the master entry.
    """

    VALID_TYPES = {"auto", "disabled"}

    def __init__(self, lang: str = "en", master: bool = False, type: str | None = None) -> None:
        self._rl_lang: str = (lang or "en").lower().strip()
        self._rl_master: bool = bool(master)
        self._rl_type: str | None = type.lower().strip() if type and type.strip() else None

    # ------------------------------------------------------------------
    # Getters / setters
    # ------------------------------------------------------------------

    def lang(self) -> str:
        """Get the ISO 639-1 language code."""
        return self._rl_lang

    def set_lang(self, lang: str) -> None:
        """Set the language code."""
        self._rl_lang = (lang or "en").lower().strip()

    def master(self) -> bool:
        """Return True if this is the master (source) language."""
        return self._rl_master

    def set_master(self, master: bool) -> None:
        """Set the master flag."""
        self._rl_master = bool(master)

    def type(self) -> str | None:
        """Get the translation type ('auto', 'disabled', or None for master)."""
        return self._rl_type

    def set_type(self, type: str | None) -> None:
        """Set the translation type."""
        self._rl_type = type.lower().strip() if type and type.strip() else None

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "lang": self._rl_lang,
            "master": self._rl_master,
        }
        if self._rl_type is not None:
            d["type"] = self._rl_type
        return d

    def to_xml(self, escape: bool = False) -> str:
        """Render as a self-closing <language> element (caller wraps in <languages>)."""
        lang_attr = f' lang="{self._rl_lang}"'
        master_attr = f' master="{"true" if self._rl_master else "false"}"'
        type_attr = f' type="{self._rl_type}"' if self._rl_type is not None else ""
        return f'        <language{lang_attr}{master_attr}{type_attr} />'

    @classmethod
    def from_dict(cls, d: dict) -> 'ReasoningLanguage':
        return cls(
            lang=d.get("lang", "en"),
            master=bool(d.get("master", False)),
            type=d.get("type", None),
        )

    def __repr__(self) -> str:
        return f"ReasoningLanguage(lang={self._rl_lang!r}, master={self._rl_master}, type={self._rl_type!r})"


log_end_file(__file__)
