from pyexpat import model
from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)

import hashlib
import re
from typing import Dict, Any, Tuple, List

from src.core.base.llm_response import LlmResponse
from src.core.llm.manager import LlmManager
from src.core.base.InputOutput import execute_neo4j_query
from src.core.base.enums import AgentReasoningEnum
from src.utils.log import log, DEBUG, INFO, WARNING, ERROR, FATAL, LLM_IN, LLM_OUT
from src.interfaces.translation_interface import TranslationInterface
from src.config.constants import AUTO_LANGUAGE


class TranslationService(TranslationInterface):
    """Service for handling text translation operations."""

    def _compute_cache_key(self, text: str, target_language: str) -> str:
        """Compute a cache key based on target language and a hash of the input text."""
        try:
            # Use sha256 of the full text plus its length to reduce collision risk
            h = hashlib.sha256(text.encode('utf-8')).hexdigest()
            key = f"{target_language}:{h}:{len(text)}"
            return key
        
        except Exception as e:
            log(ERROR, f"Error computing cache key: {e}")
            # Fallback to a simple but safe key
            return f"{target_language}:{hashlib.sha256(text[:256].encode('utf-8')).hexdigest()}"

    def _cache_read(self, key: str, target_language: str) -> Dict[str, Any] | None:
        """Read a translation from the Neo4j cache.

        Returns the translated text or None if not found.
        """
        try:
            cypher = (
                "MATCH (t:TranslationCache {key:$key, target_language:$target_language}) "
                "RETURN t.translated_text AS translated_text, t.source_language AS source_language LIMIT 1"
            )
            params = {"key": key, "target_language": target_language}
            result = execute_neo4j_query(cypher, params)
            if result and len(result) > 0:
                cached = result[0]
                translated_text = cached.get("translated_text")
                if translated_text is None:
                    return None
                return {
                    "translated_text": self._normalize_swiss_orthography(translated_text, target_language),
                    "source_language": self._normalize_source_language(cached.get("source_language")),
                }
            return None
        
        except Exception as e:
            log(ERROR, f"Error reading translation cache: {e}")
            return None

    def _cache_write(self, key: str, target_language: str, translated_text: str, source_language: str | None) -> None:
        """Write a translation into the Neo4j cache.

        Creates or updates a node labeled `TranslationCache` with properties `key`,
        `target_language` and `translated_text`.
        """
        try:
            cypher = (
                "MERGE (t:TranslationCache {key:$key}) "
                "SET t.target_language = $target_language, t.translated_text = $translated_text, t.source_language = $source_language "
                "RETURN t"
            )
            params = {"key": key, "target_language": target_language, "translated_text": translated_text, "source_language": source_language}
            execute_neo4j_query(cypher, params)

        except Exception as e:
            log(ERROR, f"Error writing translation cache: {e}")
    
    def _get_translation_model_and_provider(self) -> Tuple[str, str]:
        """Get the translation model name and provider from LLM configuration.
        
        Reads from the 'translation' section of LLM settings and retrieves the
        corresponding model configuration.
        
        Returns:
            Tuple of (model_name, model_provider)
        
        Raises:
            Exception if translation configuration or model is not found
        """
        try:
            # Get full LLM configuration
            llm_config_full = LlmManager.instance().get_llm_model_configurations()
            if not llm_config_full:
                raise Exception("LLM configuration is not available")
            
            # Extract translation settings
            translation_config = llm_config_full.get("translation", {})
            if not translation_config:
                raise Exception("Translation configuration section not found in LLM settings")
            
            # Get the model identifier (e.g., "DEFAULT")
            translator_model_id = translation_config.get("model", "DEFAULT")
            
            # Get the full model configuration
            model_config = LlmManager.instance().get_llm_model_configuration(translator_model_id)
            if not model_config:
                raise Exception(f"Model configuration not found for model ID: {translator_model_id}")
            
            # Extract model name and provider
            model_id = model_config.get("id")
            model_provider = model_config.get("modelprovider")
            
            if not model_id or not model_provider:
                raise Exception(f"Model name or provider missing in configuration for {translator_model_id}")
            
            log(DEBUG, f"Translation model: {model_id}, provider: {model_provider}")
            return (model_id, model_provider)
            
        except Exception as e:
            log(ERROR, f"Error getting translation model and provider: {e}")
            raise

    def _map_language_code_to_name(self, language_code: str) -> str | None:
        """Map a language code to its full language name.

        Returns None if the code is not recognized.
        """
        language_map = {
            "en": "English",
            "de": "German",
            "de-ch": "German (Swiss orthography)",
            "gsw": "Swiss German dialect",
            "fr": "French",
            "es": "Spanish",
            "it": "Italian",
            "it-ch": "Swiss Italian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ru": "Russian",
            "pt": "Portuguese",
            "ar": "Arabic",
            "hi": "Hindi",
            "la": "Latin"
            # Add more mappings as needed
        }
        return language_map.get(language_code.lower())

    def _normalize_source_language(self, source_language: Any) -> str | None:
        """Return a stable lowercase two-letter language code or None."""
        normalized = str(source_language or "").strip().lower()
        result = normalized if re.fullmatch(r"[a-z]{2}", normalized) else None
        return result

    def _normalize_swiss_orthography(self, translated_text: str, target_language: str) -> str:
        """Enforce ss instead of ß for de-CH, applied to both fresh and cached results."""
        result = translated_text
        if target_language.lower() == "de-ch":
            result = result.replace("ß", "ss")
        return result

    def translate(self, text: str, target_language: str, source_language: str = AUTO_LANGUAGE, disable_cache: bool = False, lexicon_hints: List[str] = None, protected_keywords: List[str] = None) -> Dict[str, Any]:
        """
        Translate text to a target language.
        
        Args:
            text: The text to translate.
            target_language: Target language code (e.g., 'de', 'fr', 'es').
            source_language: Source language code. Defaults to 'auto' for auto-detection.
            lexicon_hints: Optional list of lexicon rule strings injected into the LLM
                           prompt so the translation respects terminology constraints.
            protected_keywords: Optional list of keyword strings with explanations that
                                 must NEVER be translated (e.g. ["YES, NO (boolean response keywords)",
                                 "OK (confirmation keyword)"]).
        
        Returns:
            Dictionary with translation result.
        """
        try:
            log(DEBUG, f"Translation request: text length={len(text)}, target_language={target_language}, source_language={source_language}")
            
            if not disable_cache:
                # Cache key incorporates lexicon hints and protected keywords so entries
                # with different constraints are never mixed.
                cache_key = self._compute_cache_key(text, target_language)
                if lexicon_hints:
                    hints_blob = "|".join(lexicon_hints)
                    hints_hash = hashlib.sha256(hints_blob.encode("utf-8")).hexdigest()[:16]
                    cache_key = f"{cache_key}:lex={hints_hash}"
                if protected_keywords:
                    kw_blob = "|".join(protected_keywords)
                    kw_hash = hashlib.sha256(kw_blob.encode("utf-8")).hexdigest()[:16]
                    cache_key = f"{cache_key}:kw={kw_hash}"
                cached = self._cache_read(cache_key, target_language)
                if cached is not None:
                    log(DEBUG, f"Translation cache hit for key {cache_key}")
                    return {
                        "translated_text": cached["translated_text"],
                        "target_language": target_language,
                        "source_language": cached["source_language"]
                    }

            # Map language code to full language name, else the LLM doesn't fully understand
            target_language_ext:str = self._map_language_code_to_name(target_language)
            if target_language_ext is None:
                log(WARNING, f"Unknown target language code '{target_language}', proceeding with code as is.")
                target_language_ext = target_language

            swiss_orthography_instruction: str = ""
            if target_language.lower() == "de-ch":
                swiss_orthography_instruction = """
SWISS ORTHOGRAPHY: This is standard High German used in Switzerland, not Swiss German dialect.
Use standard German grammar and vocabulary. Never use the sharp S character (ß); always write ss instead.
Do not use dialect forms such as "isch" or "Strooss" unless they are present in the source text.
"""
            elif target_language.lower() == "gsw":
                swiss_orthography_instruction = """
SWISS GERMAN DIALECT: Translate into natural Swiss German dialect (Schweizerdeutsch), not standard High German.
Use appropriate Swiss German dialect vocabulary and grammar. This is distinct from standard German with Swiss orthography.
"""

            # Use the default LLM configuration via LlmManager to perform translation
            system_prompt:str = f"""You are a helpful translator.
Translate the user's text to this language: '{target_language}'. 
{swiss_orthography_instruction}
Return the normal translated text, preceded on the first line only by:
original_language: XX
Replace XX with the mostly-used language in the input as a lowercase two-letter ISO code.
If the input contains multiple languages, choose the language used most often, not merely
the first language encountered. If no language can be determined, use an empty value.
The first line must contain only this attribute and must not be translated or repeated.

Examples:
Task: translate to german
Input: "Hello, how are you?"
Output:
original_language: en
Hallo, wie geht es dir?

Task: translate to german
Input: "Bonjour tout le monde. Hello everyone. Bonjour encore."
Output:
original_language: fr
Hallo zusammen. Hallo allerseits. Noch einmal hallo.

CRITICAL: DO NOT TRANSLATE TEXT WITHIN PLACEHOLDERS
Placeholders are marked with %...% (e.g., %CONTENT%, %LANGUAGE%, %PAGE_NUMBER%). 
These are programmatic token substitution markers and MUST REMAIN EXACTLY AS-IS.
Examples of what to preserve:
  ✓ Correct: ... into the language %LANGUAGE%: %CONTENT%
  ✗ Wrong:   ... into the language Español: %CONTENIDO%
Always preserve the exact placeholder text including the surrounding % symbols.

CRITICAL: DO NOT TRANSLATE JSON ATTRIBUTE KEYS
JSON keys are technical identifiers and MUST REMAIN IN ENGLISH.
When translating text that contains JSON, preserve all JSON keys exactly as-is.
ONLY translate the string values inside the JSON (the "..." parts).
Examples of fixed English keys that MUST NEVER be translated: "title", "page_number", "date", "header", "footer", "location", "salutation"
  ✓ Correct JSON: {{"title": "El titulo", "page_number": "1", "date": "15 de enero"}}
  ✗ Wrong JSON:   {{"título": "El titulo", "número_de_página": "1", "fecha": "15 de enero"}}

CRITICAL: DO NOT TRANSLATE TECHNICAL KEYWORDS
The following are English technical keywords used as command responses or markers. They MUST NEVER be translated:
  - YES, NO (boolean response keywords)
  - OK (confirmation keyword)
  - INSERT:, REPLACE: (command prefixes)
These must ALWAYS remain in English, even when translating instructions for non-English languages.

PRESERVE STRUCTURAL ELEMENTS
There may be HTML tags or markdown formatting in the text—preserve these in the translation.
For example, the following must be preserved: <b>bold</b>, *italic*, **bold**, `code`, ```code blocks```.
For cases like: <span class="info-source" id="...">CONTENT</span>, preserve the entire tag structure around CONTENT
but translate CONTENT itself.

Do not wrap the translated text in quotes or code blocks unless the original text had them.

Example:
Context: Text to be translated:
```
Ciao mondo! Questa è una <span class="info-source" id="p-1">prova</span> del servizio di traduzione.
Campo della lingua: %LANGUAGE%, Contenuto: %CONTENT%.
Questa è una tabella markdown, che deve essere tradotta ma preservare la struttura della tabella:
| Italiano | Inglese |
|----------|---------|
| Ciao     | Hello   |
```
User message: You **MUST** provide the translation into the target language: 'EN'.

Response:
Hello world! This is a <span class="info-source" id="p-1">test</span> of the translation service.
Language field: %LANGUAGE%, Content: %CONTENT%.
This is a markdown table, that must be translated but preserve the table structure:
| Italian  | English |
|----------|---------|
| Ciao     | Hello   |

Note that placeholders (%LANGUAGE%, %CONTENT%) were preserved exactly, the tag structure remained intact,
and you did not output any extra commentary or code block wrapping.
"""
            # Append dynamic protected keywords block if provided
            if protected_keywords:
                kw_lines = "\n".join(f"  - {kw}" for kw in protected_keywords)
                system_prompt += f"""\n\nCRITICAL: DO NOT TRANSLATE ADDITIONAL PROTECTED KEYWORDS
The following keywords MUST NEVER be translated — always keep them exactly as written in English:
{kw_lines}
"""
            contect_prompt:str = f"Text to be translated:\n\n```{text}```"
            user_prompt:str = f"You **MUST** provide the translation into the target language: '{target_language_ext}'."

            messages = [
                {"role": "system", "content": system_prompt},
            ]
            if lexicon_hints:
                lexicon_preamble = "LEXICON RULES — you MUST follow these terminology constraints when translating:"
                lexicon_blocks = "\n\n".join(f"```\n{h}\n```" for h in lexicon_hints)
                messages.append({"role": "system", "content": f"{lexicon_preamble}\n\n{lexicon_blocks}"})
            messages.append({"role": "system", "content": contect_prompt})
            messages.append({"role": "user", "content": user_prompt})

            log(LLM_IN, f"translate [{target_language}]: {text}")

            llm_error_reason: str | None = None

            # Resolve the model configuration BEFORE the LLM call: a missing or invalid
            # model definition must surface as an error to the caller, not be swallowed
            # by the LLM-call error handling below.
            model_id, model_provider = self._get_translation_model_and_provider()

            try:
                # Use the configured translation model and provider
                llm_response:LlmResponse = LlmManager.instance().chat_completions(messages=messages, model_input=model_id, model_provider=model_provider, cache=not disable_cache, maximum_reasoning_required=AgentReasoningEnum.NONE)

            except Exception as e:
                log(ERROR, f"LLM translation call failed: {e}")
                llm_error_reason = str(e)
                llm_response = None

            translated: str = None
            detected_source_language: str | None = None
            if llm_response is not None and llm_response.response is not None:
                response_text = str(llm_response.response).strip()
                language_match = re.match(r"^original_language:\s*([a-zA-Z]{2})?\s*(?:\r?\n|$)", response_text, re.IGNORECASE)
                if language_match is not None:
                    detected_source_language = self._normalize_source_language(language_match.group(1))
                    translated = response_text[language_match.end():].strip()
                else:
                    translated = response_text
                translated = self._normalize_swiss_orthography(translated, target_language)
                log(INFO, f"translate [{target_language}]: {translated}")

            if translated is None or len(translated.strip()) == 0:
                # fallback to explicit failure text, never to the original text
                reason_suffix = f" Reason: {llm_error_reason}" if llm_error_reason else ""
                translated = f"[TRANSLATION ERROR] Translation failed for target language '{target_language}'. Check the translation model configuration in Settings > Special Models.{reason_suffix}"
                log(ERROR, translated)
                disable_cache = True  # do not cache errors

            if not disable_cache:
                # write to cache
                try:
                    self._cache_write(cache_key, target_language, translated, detected_source_language)
                
                except Exception as e:
                    log(ERROR, f"Failed to write translation cache: {e}")

            result = {
                "translated_text": translated,
                "target_language": target_language,
                "source_language": detected_source_language
            }

            log(DEBUG, f"Translation completed for language '{target_language}'")
            return result
            
        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            log(ERROR, error_msg)
            raise

    def translate_to_languages(self, text: str, target_languages: List[str], source_language: str = "auto", disable_cache: bool = False) -> Dict[str, str]:
        """
        Translate text to multiple target languages in parallel.
        
        Args:
            text: The text to translate.
            target_languages: List of target language codes (e.g., ['en', 'de', 'fr', 'it']).
            source_language: Source language code. Defaults to 'auto' for auto-detection.
            disable_cache: If True, bypass cache and always translate.
        
        Returns:
            Dictionary mapping language code to translated text.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        result: Dict[str, str] = {}
        
        if not text or not text.strip():
            log(WARNING, "translate_to_languages: Empty text provided")
            return {lang: text for lang in target_languages}
        
        log(INFO, f"translate_to_languages: Translating to {len(target_languages)} languages: {target_languages}")
        
        def translate_single(lang: str) -> tuple:
            """Translate to a single language and return (lang, translated_text)."""
            try:
                trans_result = self.translate(text, lang, source_language, disable_cache)
                return (lang, trans_result.get("translated_text", text))
            except Exception as e:
                log(ERROR, f"translate_to_languages: Failed for language '{lang}': {e}")
                return (lang, text)  # Return original text on failure
        
        # Execute translations in parallel
        with ThreadPoolExecutor(max_workers=len(target_languages)) as executor:
            futures = {executor.submit(translate_single, lang): lang for lang in target_languages}
            for future in as_completed(futures):
                try:
                    lang, translated = future.result()
                    result[lang] = translated
                except Exception as e:
                    lang = futures[future]
                    log(ERROR, f"translate_to_languages: Exception for '{lang}': {e}")
                    result[lang] = text  # Fallback to original
        
        log(INFO, f"translate_to_languages: Completed translations for {len(result)} languages")
        return result


# module level singleton accessors
_translation_service_instance = None


def get_translation_service() -> TranslationInterface:
    global _translation_service_instance
    if _translation_service_instance is None:
        _translation_service_instance = TranslationService()
    return _translation_service_instance

def set_translation_service(service: TranslationInterface) -> None:
    global _translation_service_instance
    _translation_service_instance = service


    # cache helper methods are attached below as methods of the class via monkeypatch placeholder


log_end_file(__file__)


if __name__ == "__main__":
    try:
        from src.utils.log import log_init, log_exit
        log_init()

        svc = get_translation_service()
        sample_text = "Hello, world! This is a test of the translation service."
        result = svc.translate(text=sample_text, target_language="it", disable_cache=True)
        log(INFO, f"Translation result:\n{result}")

    except Exception as exc:
        log(ERROR, f"Translation service test failed: {exc}")

    finally:
        log_exit()
