from abc import ABC, abstractmethod
from typing import Dict, Any, List
from src.config.constants import AUTO_LANGUAGE


class TranslationInterface(ABC):
    """Abstract interface for translation service operations."""

    @abstractmethod
    def translate(self, text: str, target_language: str, source_language: str = AUTO_LANGUAGE, disable_cache: bool = False, lexicon_hints: List[str] = None, protected_keywords: List[str] = None) -> Dict[str, Any]:
        """
        Translate text to a target language.
        
        Args:
            text: The text to translate.
            target_language: Target language code (e.g., 'de', 'fr', 'es').
            source_language: Source language code. Defaults to 'auto' for auto-detection.
            disable_cache: If True, bypass cache.
            lexicon_hints: Optional list of lexicon rule strings to inject into the LLM
                           prompt so the translation respects terminology constraints
                           (e.g. "You MUST always use 'Bobby' for 'Bob'").
            protected_keywords: Optional list of keyword strings with explanations that
                                 must NEVER be translated (e.g. ["YES, NO (boolean response keywords)",
                                 "OK (confirmation keyword)"]).
        
        Returns:
            Dictionary with translation result containing:
                - translated_text: The translated text
                - target_language: The target language code
                - source_language: The detected or specified source language
        """
        pass

    @abstractmethod
    def translate_to_languages(self, text: str, target_languages: List[str], source_language: str = AUTO_LANGUAGE, disable_cache: bool = False) -> Dict[str, str]:
        """
        Translate text to multiple target languages.
        
        Args:
            text: The text to translate.
            target_languages: List of target language codes (e.g., ['en', 'de', 'fr', 'it']).
            source_language: Source language code. Defaults to 'auto' for auto-detection.
            disable_cache: If True, bypass cache and always translate.
        
        Returns:
            Dictionary mapping language code to translated text.
            Example: {'en': 'Hello', 'de': 'Hallo', 'fr': 'Bonjour', 'it': 'Ciao'}
            If the source language matches a target language, returns the original text for that language.
        """
        pass
