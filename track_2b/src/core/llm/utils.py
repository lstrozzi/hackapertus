"""
Shared utility functions for LLM operations.
"""

from src.utils.log_fileload import log_start_file, log_end_file
log_start_file(__file__)


import re
import json
import ast
from numpy import isin
import tiktoken
import traceback
from typing import List, Dict, Optional, Tuple, Union

from src.config.constants import BLAZOR_HOSTNAME
from src.utils.system import get_os_type
from src.utils.json_utils import convert_malformed_json
from src.utils.text_utils import is_valid_regex
from src.utils.log import log, WARNING, ERROR, DEBUG
from src.core.llm.config import MODEL, MODEL_HIGHINTELLIGENCE


def adjust_endpoint_to_docker_format(endpoint: str) -> str:
    """
    Adjust endpoint URL for different environments (Docker vs native).
    """
    if endpoint is None or len(endpoint) == 0:
        return endpoint
    
    if get_os_type() == "Linux":
        # we can't use 'localhost', we need to use 'host.docker.internal' or different location to access the host machine
        # this is available in the environment variable BLAZOR_HOSTNAME
        endpoint = endpoint.replace("localhost", BLAZOR_HOSTNAME)
        endpoint = endpoint.replace("127.0.0.1", BLAZOR_HOSTNAME)

    return endpoint


def num_tokens_from_string(string: str) -> int:
    """
    Returns the number of tokens in a text string.
    """
    try:
        embedding_model = "o200k_base" if MODEL == MODEL_HIGHINTELLIGENCE else "cl100k_base"
        encoding = tiktoken.get_encoding(embedding_model)

        string = string.replace("<|endoftext|>", "")        # remove special tokens that tiktoken counts
        num_tokens = len(encoding.encode(string))
    except Exception as e:
        log(WARNING, f"num_tokens_from_string(): {e}. Returning 0 tokens for string length {len(string)}.")
        num_tokens = 0
    return num_tokens


def escape_characters_inside_json_strings(json_text: str) -> str:
    """
    Removes all newlines and escapes illegal characters inside JSON strings.
    """
    new_json_text = []
    lines = json_text.split("\n")

    for line in lines:
        line_trimmed = line.strip()
        if line_trimmed.startswith('"') and ":" in line_trimmed:
            # This is probably a key-value pair line
            key, value = line_trimmed.split(":", 1)
            key = key.strip()
            value = value.strip()

            if value.startswith('"') and value.endswith(('"', '",', '"}', '"]', '" }', '" ]')):
                value_content = value[1:-1]  # Remove surrounding quotes
                # Reset escaped characters
                value_content = value_content.replace('\\"', '"').replace("\\\\", "\\")
                # Escape illegal characters
                value_content = (
                    value_content.replace("\\", "\\\\")
                    .replace('"', '\\"')
                    .replace("\n", "\\n")
                    .replace("\t", "\\t")
                    .replace("'", "\"")
                )
                value = f'"{value_content}"'
            new_json_text.append(f"{key}: {value}")
        else:
            new_json_text.append(line_trimmed)

    return " ".join(new_json_text)


def _auto_escape_unescaped_double_quotes(s: str) -> str:
    """
    Heuristic fixer that escapes double quotes that appear inside JSON
    string values but are not escaped. It scans characters and when inside
    a double-quoted string, if we encounter a double quote whose following
    non-space character is not a JSON delimiter (comma, ], }, :) or end
    of string, we treat that double quote as an internal quote and escape it.
    This is conservative and intended to correct common LLM mistakes like
    '"Search for "GPT-OSS-20B..."'.
    """
    if s is None:
        return s

    out = []
    in_string = False
    escape = False
    i = 0
    L = len(s)

    while i < L:
        ch = s[i]

        if escape:
            out.append(ch)
            escape = False
            i += 1
            continue

        if ch == "\\":
            # start of escape sequence
            out.append(ch)
            escape = True
            i += 1
            continue

        if ch == '"':
            if in_string:
                # lookahead to decide if this quote really closes the string
                j = i + 1
                # skip whitespace
                while j < L and s[j].isspace():
                    j += 1

                next_ch = s[j] if j < L else None
                # valid closers after a string: comma, ], }, :, '=', or end
                # include '=' here as a safe-guard so we don't escape a closing
                # quote when the caller intends to repair '=' -> ':' later.
                if next_ch in [',', ']', '}', ':', '=', None]:
                    out.append(ch)
                    in_string = False
                else:
                    # probably an unescaped inner quote -> escape it
                    out.append('\\"')
                i += 1
                continue
            else:
                # entering a string
                in_string = True
                out.append(ch)
                i += 1
                continue

        out.append(ch)
        i += 1

    return ''.join(out)


def _convert_equals_to_colon(s: str) -> str:
    """
    Convert occurrences of quoted-key assignment using '=' into ':'
    for common malformed JSON like `{"key"="value"}` or `{'key'='value'}`.
    This is a conservative, regex-based transformation that only replaces
    an equals sign when it appears between a quoted key and the next token.
    """
    if s is None:
        return s

    # Replace patterns like "key" =  or 'key' =  with "key":
    s2 = re.sub(r'(["\'][^"\']+["\'])\s*=\s*', r"\1:", s)
    return s2


def _normalize_unicode_and_smart_quotes(s: str) -> str:
    """
    Normalize special Unicode characters and typographic quotes to ASCII equivalents.
    Handles:
    - Smart/curly quotes (U+201C/U+201D/U+2018/U+2019) -> straight quotes
    - Non-breaking spaces (U+00A0, U+202F) -> regular space
    - Various dashes (en-dash U+2013, em-dash U+2014, hyphen U+2011, etc.) -> regular hyphen
    - Other typographic characters that may cause encoding issues
    This helps when JSON contains characters that can't be encoded in cp1252 (Windows console)
    or when LLM output uses curly quotes which break JSON parsing.
    """
    if s is None:
        return s
    
    # Smart/typographic quotes
    s = s.replace('\u201c', '"')  # Left double quotation mark
    s = s.replace('\u201d', '"')  # Right double quotation mark
    s = s.replace('\u2018', "'")  # Left single quotation mark
    s = s.replace('\u2019', "'")  # Right single quotation mark
    
    # Non-breaking spaces
    s = s.replace('\u00a0', ' ')  # Non-breaking space
    s = s.replace('\u202f', ' ')  # Narrow non-breaking space
    s = s.replace('\u2009', ' ')  # Thin space
    s = s.replace('\u200b', '')   # Zero-width space
    
    # Various dashes
    s = s.replace('\u2011', '-')  # Non-breaking hyphen
    s = s.replace('\u2012', '-')  # Figure dash
    s = s.replace('\u2013', '-')  # En dash
    s = s.replace('\u2014', '-')  # Em dash
    s = s.replace('\u2015', '-')  # Horizontal bar
    s = s.replace('\u2043', '-')  # Hyphen bullet
    s = s.replace('\u2010', '-')  # Hyphen
    
    # Other typographic characters
    s = s.replace('\u2047', '?')  # Double question mark
    s = s.replace('\u2048', '?')  # Question exclamation mark
    s = s.replace('\u2049', '!')  # Exclamation question mark
    
    return s


def _compact_multiline_json(s: str) -> str:
    """
    Convert multiline JSON to single line while preserving string content.
    Removes newlines and extra spaces outside of string values to help
    with parsing complex nested structures.
    """
    if s is None:
        return s
    
    result = []
    in_string = False
    escape = False
    
    for ch in s:
        if escape:
            result.append(ch)
            escape = False
            continue
            
        if ch == '\\':
            result.append(ch)
            escape = True
            continue
            
        if ch == '"' and not in_string:
            in_string = True
            result.append(ch)
            continue
            
        if ch == '"' and in_string:
            in_string = False
            result.append(ch)
            continue
            
        if in_string:
            # Inside string: preserve everything including newlines
            result.append(ch)
        else:
            # Outside string: collapse whitespace
            if ch in '\r\n\t':
                # Skip newlines and tabs outside strings
                continue
            elif ch == ' ':
                # Keep single space, skip consecutive spaces
                if result and result[-1] != ' ':
                    result.append(ch)
            else:
                result.append(ch)
    
    return ''.join(result).strip()


def _quote_unquoted_keys(s: str) -> str:
    """
    Quote unquoted object keys. Transforms `{a: 1, b: 2}` -> `{"a": 1, "b": 2}`.
    This is a conservative transformation that only targets JS-like unquoted
    keys appearing at object key positions (start of object or after a comma).
    """
    if s is None:
        return s

    # This regex finds unquoted keys consisting of letters, digits, underscore or hyphen
    # that appear before a colon and are not already quoted. It's intentionally simple
    # and may not cover all edge-cases, but is safe for common LLM outputs.
    def _repl(m):
        key = m.group(1)
        return '"' + key + '":'

    # Use a positive lookbehind to ensure we're at the start of an object or after a comma/whitespace
    pattern = r'(?:(?<=\{|,)|(?<=\s))([A-Za-z_][A-Za-z0-9_\-]*)\s*:\s*'
    try:
        return re.sub(pattern, _repl, s)
    except Exception:
        return s


def _insert_missing_commas(s: str) -> str:
    """
    Insert missing commas between adjacent value/key pairs like
    `{"a": 1 "b": 2}` -> `{"a": 1, "b": 2}`.
    It looks for a value (string, number, true/false/null, object/array) followed
    immediately by a quote that likely starts the next key and inserts a comma.
    """
    if s is None:
        return s

    # We'll apply two conservative substitutions to insert commas where they're clearly missing:
    # 1) numeric/boolean/null/closing-brace-or-bracket values followed by another value token
    #    e.g. `1 true` -> `1, true`
    # 2) quoted string values followed by another quoted token without a comma
    #    e.g. `"v" "k"` -> `"v", "k"`
    try:
        # pattern for numbers, booleans, null, or closing braces/brackets
        pattern_val = re.compile(r'((?:\btrue\b|\bfalse\b|\bnull\b)|(?:\d+(?:\.\d+)?)|(?:\}|\]))(?=\s*(?:"|\d|\{|\[|\-|[A-Za-z]))', flags=re.IGNORECASE)
        s2 = pattern_val.sub(lambda m: m.group(1) + ',', s)

        # pattern for quoted string values followed directly by another quoted token
        pattern_str = re.compile(r'("(?:\\.|[^"\\])*")(?=\s*")')
        s3 = pattern_str.sub(lambda m: m.group(1) + ',', s2)

        return s3
    except Exception:
        return s


def _quote_unquoted_array_elements(s: str) -> str:
    """
    Quote unquoted array elements so `[a, b]` -> `["a", "b"]`.
    This targets identifiers that are not numbers, booleans, null, objects, arrays or quoted strings.
    """
    if s is None:
        return s

    try:
        pattern = re.compile(r'(\[|,)\s*([A-Za-z_][A-Za-z0-9_\-]*)\s*(?=,|\])')
        return pattern.sub(lambda m: m.group(1) + ' "' + m.group(2) + '"', s)
    except Exception:
        return s


def _insert_commas_between_unquoted_array_tokens(s: str) -> str:
    """
    Insert commas between adjacent unquoted identifier-like tokens inside arrays
    when commas are completely missing, e.g. `[foo-bar baz qux-1]` -> `[foo-bar, baz, qux-1]`.
    This is conservative: it only acts on array segments that contain no commas and
    where all whitespace-separated tokens look like identifiers (letters, digits, underscore, hyphen).
    """
    if s is None:
        return s

    def _process_array_inner(m):
        inner = m.group(1)
        # If the array inner already has commas or quotes or braces/colons, skip
        if ',' in inner or '"' in inner or "'" in inner or ':' in inner or '{' in inner or '}' in inner:
            return '[' + inner + ']'

        tokens = re.split(r'\s+', inner.strip())
        if not tokens:
            return '[' + inner + ']'

        # Only transform if every token matches an identifier-like pattern
        if all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_\-]*', t) for t in tokens):
            return '[' + ', '.join(tokens) + ']'

        return '[' + inner + ']'

    try:
        return re.sub(r'\[([^\]]+)\]', _process_array_inner, s)
    except Exception:
        return s


def _escape_inner_quotes_using_structure(s: str, expected_keys: Optional[List[str]] = None) -> str:
    """
    Structure-aware quote escaper: uses expected JSON keys to identify where
    string values begin and end, allowing us to definitively escape all quotes
    between the opening and closing quotes of a value.
    
    Strategy: For each expected key, find its value's opening quote, scan BACKWARDS from the
    expected next key (or end of object) to find the TRUE closing quote, then escape ALL quotes in between.
    """
    if s is None or not expected_keys:
        return s
    
    result = s
    
    for idx, key in enumerate(expected_keys):
        # Find the key in the current result string
        key_pattern = f'"{key}"'
        key_pos = result.find(key_pattern)
        if key_pos == -1:
            continue
        
        # Find the colon after the key
        colon_pos = key_pos + len(key_pattern)
        while colon_pos < len(result) and result[colon_pos] in ' \t\n\r':
            colon_pos += 1
        
        if colon_pos >= len(result) or result[colon_pos] != ':':
            continue
        
        # Find the opening quote of the value
        value_start = colon_pos + 1
        while value_start < len(result) and result[value_start] in ' \t\n\r':
            value_start += 1
        
        if value_start >= len(result) or result[value_start] != '"':
            continue
        
        # Find the search end position: either the next key or a closing brace/bracket
        search_end = len(result)
        if idx + 1 < len(expected_keys):
            # Look for the next key
            next_key_pattern = f'"{expected_keys[idx + 1]}"'
            next_key_pos = result.find(next_key_pattern, value_start)
            if next_key_pos != -1:
                search_end = next_key_pos
        
        # Scan BACKWARDS from search_end to find the TRUE closing quote
        # This is the LAST quote followed by a comma, ], }, or whitespace+comma/]/}
        closing_quote_pos = None
        i = search_end - 1
        
        while i > value_start:
            if result[i] == '"':
                # Check if this quote is escaped
                # Count preceding backslashes
                num_backslashes = 0
                j = i - 1
                while j >= 0 and result[j] == '\\':
                    num_backslashes += 1
                    j -= 1
                
                # If even number of backslashes (or zero), the quote is not escaped
                if num_backslashes % 2 == 0:
                    # Check what follows this quote
                    k = i + 1
                    while k < len(result) and result[k] in ' \t\n\r':
                        k += 1
                    
                    next_ch = result[k] if k < len(result) else None
                    
                    if next_ch in [',', ']', '}', None]:
                        # This is the TRUE closing quote
                        closing_quote_pos = i
                        break
            
            i -= 1
        
        if closing_quote_pos is None or closing_quote_pos <= value_start:
            # No valid closing quote found, skip this key
            continue
        
        # Now escape all UNescaped quotes between value_start and closing_quote_pos
        # Build the new string in parts: before_value + escaped_value + after_value
        before_value = result[:value_start + 1]  # Include opening quote
        value_content = result[value_start + 1:closing_quote_pos]  # Content between quotes
        after_value = result[closing_quote_pos:]  # From closing quote onward
        
        # Escape all unescaped quotes in value_content
        escaped_content = []
        i = 0
        
        while i < len(value_content):
            if value_content[i] == '\\':
                # Copy the backslash and the next character
                escaped_content.append(value_content[i])
                if i + 1 < len(value_content):
                    escaped_content.append(value_content[i + 1])
                    i += 2
                else:
                    i += 1
                continue
            
            if value_content[i] == '"':
                # Escape this quote
                escaped_content.append('\\')
                escaped_content.append('"')
                i += 1
                continue
            
            escaped_content.append(value_content[i])
            i += 1
        
        result = before_value + ''.join(escaped_content) + after_value
    
    return result


def _escape_inner_quotes_fallback(s: str) -> str:
    """
    Fallback escaper when structure is not available. Uses simple heuristics:
    - Track when inside a string
    - A quote closes a string if followed by: comma, colon, closing bracket/brace, or end
    - Otherwise, escape it as an inner quote
    """
    if s is None:
        return s
    
    out = []
    i = 0
    L = len(s)
    in_string = False
    escape_next = False
    
    while i < L:
        ch = s[i]
        
        if escape_next:
            out.append(ch)
            escape_next = False
            i += 1
            continue
        
        if ch == '\\':
            out.append(ch)
            escape_next = True
            i += 1
            continue
        
        if ch == '"':
            if not in_string:
                out.append(ch)
                in_string = True
                i += 1
            else:
                # Check if this closes the string
                j = i + 1
                while j < L and s[j] in ' \t\n\r':
                    j += 1
                next_ch = s[j] if j < L else None
                
                if next_ch in [',', ':', ']', '}', None]:
                    # Closing quote
                    out.append(ch)
                    in_string = False
                    i += 1
                else:
                    # Inner quote - escape it
                    out.append('\\')
                    out.append(ch)
                    i += 1
        else:
            out.append(ch)
            i += 1
    
    return ''.join(out)


def _fix_missing_closing_quote_on_keys(s: str) -> str:
    """
    Fix missing closing double quotes on JSON dictionary keys.
    E.g., "optional_require_exact_time:false -> "optional_require_exact_time":false
    
    This scans for patterns where an opening quote is followed by valid key characters,
    then directly by a colon without a closing quote.
    """
    if s is None:
        return s
    
    # Pattern: opening quote, one or more valid key chars (letters, digits, underscore, underscore),
    # directly followed by a colon (no closing quote in between)
    # Example: "optional_require_exact_time:false
    # We want to insert a closing quote before the colon
    
    # Regex: "([A-Za-z_][A-Za-z0-9_]*):
    # Replace with: "\1":
    try:
        pattern = r'"([A-Za-z_][A-Za-z0-9_]*)\s*:'
        result = re.sub(pattern, r'"\1":', s)
        if result != s:
            log(DEBUG, f"get_json_from_text(): fixed missing closing quotes on keys\nBefore:\n{s}\nAfter:\n{result}")
        return result
    except Exception:
        return s



def get_json_from_text(json_text: str, json_example: Optional[Union[Dict, List]] = None, json_schema: Optional[Dict] = None) -> Optional[Union[Dict, List]]:
    """
    Extract and parse JSON from text, with robust error handling.
    
    Args:
        json_text: The text containing JSON to extract
        json_example: Optional example JSON dict or list showing expected structure
        json_schema: Optional JSON schema dict defining expected structure
    
    Returns:
        Parsed JSON dict or list, or None if parsing fails
    """
    if json_text is None:
        return None
    if len(json_text.strip()) == 0:
        return {}
    
    # Extract expected keys from schema or example for structure-aware escaping
    expected_keys = None
    if json_schema and 'properties' in json_schema:
        expected_keys = list(json_schema['properties'].keys())
    elif isinstance(json_example, dict):
        expected_keys = list(json_example.keys())
    
    result = None
    json_adjusted = None

    try:
        if json_text.startswith("```") or json_text.endswith("```"):
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
        if (json_text.startswith("'{") and json_text.endswith("}'")) or (json_text.startswith("'[") and json_text.endswith("]'")):
            return get_json_from_text(json_text[1:-1])

        # Remove leading 'json' marker (like "json\n" or "JSON\r\n").
        # This also handles fenced code blocks that were like ```json\n...``` where the initial ``` removal left a leading 'json\n'.
        json_text = re.sub(r'(?i)\Ajson\r?\n', '', json_text)

        # Strip any initial/final newlines (CR/LF) after the above removals so parsing isn't confused by extra blank lines
        json_text = json_text.strip('\r\n')

        # Normalize smart/curly quotes and special Unicode characters to ASCII equivalents
        json_text = _normalize_unicode_and_smart_quotes(json_text)
        
        # Compact multiline JSON to single line (for deeply nested structures)
        # This helps with parsing complex JSONs that span many lines
        json_text = _compact_multiline_json(json_text)
        
        # Remove trailing dots before closing braces/brackets (common LLM mistake)
        # e.g., "[].}" -> "[]}" or "{.}" -> "{}"
        json_text = re.sub(r'\.([\]}])', r'\1', json_text)
        
        # Extract the most likely JSON substring by scanning for a balanced
        # opening ('{' or '[') and its corresponding matching closing
        # bracket while correctly handling quoted strings and escapes.
        # find first occurrences (or -1 if not present)
        first_curly = json_text.find('{')
        first_square = json_text.find('[')
        candidates = [i for i in (first_curly, first_square) if i >= 0]
        if not candidates:
            # No JSON opening bracket found
            return None

        first_idx = min(candidates)

        opening_char = json_text[first_idx]
        closing_char = ']' if opening_char == '[' else '}'

        # Scan forward from the opening bracket to find the matching
        # closing bracket. Ignore brackets that appear inside string
        # literals and respect escape characters.
        depth = 0
        in_string = False
        string_char = None
        escape = False
        last_idx = None

        for i in range(first_idx, len(json_text)):
            ch = json_text[i]

            if escape:
                escape = False
                continue

            if ch == "\\":
                # next character is escaped
                escape = True
                continue

            if in_string:
                if ch == string_char:
                    in_string = False
                    string_char = None
                continue

            # not in a string
            if ch == '"' or ch == "'":
                in_string = True
                string_char = ch
                continue

            if ch == opening_char:
                depth += 1
            elif ch == closing_char:
                depth -= 1
                if depth == 0:
                    last_idx = i
                    break

        if last_idx is None:
            # matching closing bracket not found - use entire text and close any unclosed brackets
            json_adjusted = json_text[first_idx:]
            # Add missing closing brackets based on depth, but only if we're not inside a string
            # (if in_string is True, the depth might be wrong due to malformed quotes)
            if depth > 0 and not in_string:
                # We have unclosed brackets - add the appropriate number of closing brackets
                json_adjusted += closing_char * depth
        else:
            json_adjusted = json_text[first_idx:last_idx + 1]
        

        # Now fix mistakes in the json string, that Mistral typically creates
        json_adjusted = json_adjusted.replace("""\\\n""", """\\n""")

        # Convert the string to a dictionary using json.loads
        result = json.loads(json_adjusted)
        
    except Exception as e:
        stack = traceback.format_exc()
        log(WARNING, f"get_json_from_text()::json.loads(): {e}.\nStack trace:\n{stack}\n\nThe text is:\n{json_text}\n\nWe tried the adjusted text:\n{json_adjusted}\nNow let's manipulate the json to make it more valid.")
        try:
            try:
                # let's try to manipulate the json string to make it valid
                json_adjusted = escape_characters_inside_json_strings(json_adjusted)
                result = json.loads(json_adjusted)
            except Exception as e:
                log(WARNING, f"get_json_from_text()::json.loads(): {e}.\n\nThe text is:\n{json_text}\n\nWe tried the adjusted text:\n{json_adjusted}\nWe will try auto-escape repair and other fallbacks.")

                # As an additional conservative repair, first convert quoted-key equals
                # assignments into proper JSON colons: {"k"="v"} -> {"k":"v"}
                try:
                    repaired_eq = _convert_equals_to_colon(json_adjusted)
                    if repaired_eq != json_adjusted:
                        log(DEBUG, f"get_json_from_text(): converted equals to colon\nBefore:\n{json_adjusted}\nAfter:\n{repaired_eq}")
                        json_adjusted = repaired_eq
                        result = json.loads(json_adjusted)
                except Exception:
                    # ignore and continue to other fallbacks
                    pass

                # Attempt to fix missing closing quotes on dictionary keys
                # E.g., "optional_require_exact_time:false -> "optional_require_exact_time":false
                if result is None:
                    try:
                        repaired_missing_quote = _fix_missing_closing_quote_on_keys(json_adjusted)
                        if repaired_missing_quote != json_adjusted:
                            log(DEBUG, f"get_json_from_text(): fixed missing closing quotes on keys\nBefore:\n{json_adjusted}\nAfter:\n{repaired_missing_quote}")
                            json_adjusted = repaired_missing_quote
                            result = json.loads(json_adjusted)
                    except Exception:
                        # ignore and continue to other fallbacks
                        pass

                # Attempt structure-aware quote escaping if we have expected keys
                # Track whether this repair was applied to skip comma/quote repairs that could break it
                deterministic_escape_applied = False
                if result is None and expected_keys:
                    try:
                        repaired_det = _escape_inner_quotes_using_structure(json_adjusted, expected_keys)
                        if repaired_det != json_adjusted:
                            log(DEBUG, f"get_json_from_text(): applied structure-aware quote escape\nBefore:\n{json_adjusted}\nAfter:\n{repaired_det}")
                            json_adjusted = repaired_det
                            deterministic_escape_applied = True
                            result = json.loads(json_adjusted)
                    except Exception:
                        # ignore and continue with other fallbacks
                        pass
                
                # Fallback: try simple heuristic escaper if structure-aware failed or unavailable
                # Continue even if structure-aware was applied but didn't succeed
                if result is None:
                    try:
                        repaired_det = _escape_inner_quotes_fallback(json_adjusted)
                        if repaired_det != json_adjusted:
                            log(DEBUG, f"get_json_from_text(): applied fallback quote escape\nBefore:\n{json_adjusted}\nAfter:\n{repaired_det}")
                            json_adjusted = repaired_det
                            deterministic_escape_applied = True
                            result = json.loads(json_adjusted)
                    except Exception:
                        # ignore and continue with other fallbacks
                        pass

                # Attempt conservative auto-escape repair for unescaped inner quotes.
                # Apply repeatedly until stable (some LLM outputs require multiple passes)
                if result is None:
                    try:
                        repaired = json_adjusted
                        for _pass in range(5):
                            new = _auto_escape_unescaped_double_quotes(repaired)
                            if new == repaired:
                                break
                            repaired = new

                        if repaired != json_adjusted:
                            log(DEBUG, f"get_json_from_text(): applied auto-escape repair\nBefore:\n{json_adjusted}\nAfter:\n{repaired}")
                            json_adjusted = repaired
                            result = json.loads(json_adjusted)
                            # if successful, continue
                    except Exception:
                        # ignore and continue with other fallbacks
                        pass

                # Attempt to quote unquoted keys like {a:1, b:2} -> {"a":1, "b":2}
                if result is None:
                    try:
                        repaired_q = _quote_unquoted_keys(json_adjusted)
                        if repaired_q != json_adjusted:
                            log(DEBUG, f"get_json_from_text(): quoted unquoted keys\nBefore:\n{json_adjusted}\nAfter:\n{repaired_q}")
                            json_adjusted = repaired_q
                            result = json.loads(json_adjusted)
                    except Exception:
                        pass

                # Attempt to insert missing commas between adjacent pairs
                # Skip this if deterministic escape was applied, as comma insertion can break escaped strings
                if result is None and not deterministic_escape_applied:
                    try:
                        repaired_c = _insert_missing_commas(json_adjusted)
                        if repaired_c != json_adjusted:
                            log(DEBUG, f"get_json_from_text(): inserted missing commas\nBefore:\n{json_adjusted}\nAfter:\n{repaired_c}")
                            json_adjusted = repaired_c
                            result = json.loads(json_adjusted)
                    except Exception:
                        pass

                # Attempt to quote unquoted array elements like [a, b] -> ["a", "b"]
                # Skip this if deterministic escape was applied, as these repairs can break escaped strings
                if result is None and not deterministic_escape_applied:
                    try:
                        # First, insert commas between adjacent unquoted identifier tokens when commas are entirely missing
                        repaired_ac = _insert_commas_between_unquoted_array_tokens(json_adjusted)
                        if repaired_ac != json_adjusted:
                            log(DEBUG, f"get_json_from_text(): inserted commas between unquoted array tokens\nBefore:\n{json_adjusted}\nAfter:\n{repaired_ac}")
                            json_adjusted = repaired_ac
                            result = json.loads(json_adjusted)
                    except Exception:
                        pass

                    try:
                        repaired_ae = _quote_unquoted_array_elements(json_adjusted)
                        if repaired_ae != json_adjusted:
                            log(DEBUG, f"get_json_from_text(): quoted unquoted array elements\nBefore:\n{json_adjusted}\nAfter:\n{repaired_ae}")
                            json_adjusted = repaired_ae
                            result = json.loads(json_adjusted)
                    except Exception:
                        pass

                if result is None:
                    log(WARNING, f"get_json_from_text()::json.loads(): {e}.\n\nThe text is:\n{json_text}\n\nWe tried the adjusted text:\n{json_adjusted}\nWe re-try with convert_malformed_json()")
                    try:
                        json_adjusted = convert_malformed_json(json_adjusted)
                        log(DEBUG, f"get_json_from_text()::json.loads(): {e}.\n\nThe converted text is:\n{json_adjusted}\nWe re-try with json.loads()")
                        result = json.loads(json_adjusted)
                    except Exception as e:
                        log(WARNING, f"get_json_from_text()::json.loads(): {e}.\n\nThe text is:\n{json_text}\n\nWe tried the correctly formatted text:\n{json_adjusted}\nWe re-try with ast.literal_eval()")
                        result = ast.literal_eval(json_adjusted)
        except Exception as e:
            stack = traceback.format_exc()
            if not "optional_internal_thought" in json_adjusted:
                log(ERROR, f"get_json_from_text()::ast.literal_eval(): {e}.\nStack trace:\n{stack}\nThere's totally nothing we can do with this json.")
            else:
                log(WARNING, f"get_json_from_text()::ast.literal_eval(): {e}.\nStack trace:\n{stack}\nThere's nothing we can do with this json, as it looks. Last hope: remove the optional_internal_thought and try again.")
                try:
                    # let's remove starting with "optional_internal_thought": until "optional_complex_plan": and try again. Use a regex
                    # we must remove the optional_internal_thought, because it's not a valid json
                    pattern = r"\"optional_internal_thought\":.*?\"optional_complex_plan\":"
                    json_adjusted = re.sub(pattern, "", json_adjusted)
                    result = json.loads(json_adjusted)
                except Exception as e:
                    stack = traceback.format_exc()
                    log(ERROR, f"get_json_from_text()::ast.literal_eval(): {e}.\nStack trace:\n{stack}\nThere's totally nothing we can do with this json.")
                    return None                

    return result


def calculate_key(messages: List[Dict], tools: List[Dict], model: str, max_tokens: int, temperature: float, response_format: Dict, structured_output: object) -> str:
    """
    Calculate cache key for LLM requests.
    """
    functions_str = ""
    if tools is not None:
        function_names = [tool['function']['name'] for tool in tools if 'function' in tool and 'name' in tool['function']]
        functions_str = ",".join(function_names)

    key = f"CHAT_COMPLETIONS({model}): messages<" + str(messages) + "> ||| max_tokens<" + str(max_tokens) + "> ||| temperature<" + str(temperature) + "> ||| response_format<" + str(response_format) + "> ||| functions<" + functions_str + "> ||| structured_output<" + str(structured_output) + ">"
    return key


def verify_formal_and_actual_dictionary(formal: Union[dict, List], actual: Union[list, dict]) -> Tuple[bool, List[str]]:
    """
    Verifies if the actual dictionary contains all the keys from the formal dictionary.

    Returns a tuple (is_valid, errors) where errors is a list of human-readable
    error messages describing any mismatches found.
    """
    errors: List[str] = []
    result = True
    # If 'actual' is empty or None, allow it only when ALL formal
    # keys are optional (convention: keys start with 'optional_'). In that
    # case the absence of values is valid and we return success.
    if actual is None or len(actual) == 0:
        # Determine if `formal` defines only optional entries. Use a
        # safe check for mapping-like objects.
        try:
            if isinstance(formal, Dict):
                formal_keys = formal.keys() if hasattr(formal, 'keys') else []
                all_optional = all(str(k).lower().startswith("optional_") for k in formal_keys)
            elif isinstance(formal, List):
                all_optional = all(isinstance(item, str) and item.lower().startswith("optional_") for item in formal)
            else:
                all_optional = False

        except Exception:
            all_optional = False

        if all_optional:
            return True, []

        errors.append("'actual' input is None or empty, and some required keys are missing.")
        return False, errors

    try:
        if isinstance(formal, Dict) and not isinstance(actual, Dict):
            msg = f"'formal' input is a dictionary but 'actual' input is of type {type(actual).__name__}"
            log(ERROR, f"ERROR: verify_formal_and_actual_dictionary(): {msg}")
            errors.append(msg)
            return False, errors
        elif isinstance(formal, List) and not isinstance(actual, List):
            msg = f"'formal' input is a list but 'actual' input is of type {type(actual).__name__}"
            log(ERROR, f"ERROR: verify_formal_and_actual_dictionary(): {msg}")
            errors.append(msg)
            return False, errors
        
        if isinstance(actual, Dict):
            for formal_key in formal.keys():
                if formal_key not in actual.keys():
                    if "optional" not in formal_key:
                        errors.append(f"Missing required key: {formal_key}")
                        result = False
                        break
                    else:
                        continue

                actual_item = actual[formal_key]
                # nested dict -> recurse
                if isinstance(formal[formal_key], Dict):
                    if "optional" in formal_key and (actual[formal_key] is None or (hasattr(actual[formal_key], '__len__') and len(actual[formal_key]) == 0)):
                        continue
                    sub_ok, sub_errors = verify_formal_and_actual_dictionary(formal[formal_key], actual_item)
                    if not sub_ok:
                        errors.extend([f"{formal_key}.{e}" for e in sub_errors])
                        result = False
                        break
                
                # list type
                if isinstance(formal[formal_key], list):
                    if not isinstance(actual_item, list) and not ("optional" in formal_key and actual_item is None):
                        errors.append(f"Type mismatch for key '{formal_key}': expected list, got {type(actual_item).__name__}")
                        result = False
                        break

                # string type and regexp checks
                if isinstance(formal[formal_key], str):
                    if not isinstance(actual_item, str) and not ("optional" in formal_key and actual_item is None):
                        errors.append(f"Type mismatch for key '{formal_key}': expected str, got {type(actual_item).__name__}")
                        result = False
                        break
                    
                    if formal[formal_key].startswith("regexp("):
                        pattern = formal[formal_key][7:-1]
                        if is_valid_regex(pattern):
                            try:
                                if not re.match(pattern, actual_item):
                                    msg = f"regex mismatch for key {formal_key}: pattern {pattern} != value {actual_item}"
                                    log(ERROR, f"ERROR: verify_formal_and_actual_dictionary(): {msg}")
                                    errors.append(msg)
                                    result = False
                                    break
                            except Exception as e:
                                stack = traceback.format_exc()
                                msg = f"invalid regex check for key {formal_key}: {e}"
                                log(ERROR, f"ERROR: verify_formal_and_actual_dictionary(): {msg}\nStack trace:\n{stack}")
                                errors.append(msg)
                                result = False
                                break

        elif isinstance(actual, List):
            for idx, actual_item in enumerate(actual):
                if isinstance(formal, List):
                    formal_item = formal[idx] if idx < len(formal) else None
                    if formal_item is None:
                        msg = f"'formal' input list has no item at index {idx} to match 'actual' item"
                        log(ERROR, f"ERROR: verify_formal_and_actual_dictionary(): {msg}")
                        errors.append(msg)
                        result = False
                        break

                    sub_ok, sub_errors = verify_formal_and_actual_dictionary(formal_item, actual_item)
                    if not sub_ok:
                        errors.extend([f"[{idx}].{e}" for e in sub_errors])
                        result = False
                        break

                else:
                    msg = f"'formal' input is not a list but 'actual' is a list at index {idx}"
                    log(ERROR, f"ERROR: verify_formal_and_actual_dictionary(): {msg}")
                    errors.append(msg)
                    result = False
                    break
                    
        else:
            msg = f"'actual' input is of unsupported type: {type(actual).__name__}"
            log(ERROR, f"ERROR: verify_formal_and_actual_dictionary(): {msg}")
            errors.append(msg)
            result = False

    except Exception as e:
        stack = traceback.format_exc()
        msg = f"Exception during verification: {e}."
        log(ERROR, f"ERROR: verify_formal_and_actual_dictionary(): {msg}\nStack trace:\n{stack}")
        errors.append(msg)
        result = False
    
    return result, errors


if __name__ == "__main__":
    from src.utils.log import log_init, log_exit
    
    try:
        log_init()

        # Counters and lists for final summary
        passed = 0
        failed = 0
        failed_tests: List[str] = []

        # Simple runnable tests for quick local verification (no pytest required).
        def _run_case(name: str, text: str, json_example:Union[list,dict], json_schema:dict, expect_type:type):
            print(f"--- Test: {name} ---")
            parsed = None
            # update counters declared in outer scope
            global passed, failed
            try:
                parsed = get_json_from_text(text, json_example=json_example, json_schema=json_schema)
            except Exception as e:
                print(f"Exception while parsing: {e}")
                parsed = None

            print("Input:")
            try:
                print(text)
            except UnicodeEncodeError:
                # If the text contains characters that can't be printed to the console,
                # use sys.stdout.buffer to write bytes with 'replace' error handling
                import sys
                sys.stdout.buffer.write(text.encode('utf-8', errors='replace'))
                sys.stdout.buffer.write(b'\n')
            print("Parsed:", parsed)
            ok = (parsed is not None and isinstance(parsed, expect_type)) if expect_type is not None else (parsed is None)
            if ok:
                passed += 1
            else:
                failed += 1
                failed_tests.append(name)
            print("Result:", "PASS" if ok else "FAIL")
            print()

        # 1) Basic valid JSON
        _run_case("valid_json", '''
        {
            "name": "John Doe",
            "age": 30,
            "address": {
                "street": "123 Main St",
                "city": "Anytown"
            },
            "hobbies": ["reading", "traveling", "swimming"]
        }
        ''', json_example={"name": "Example", "age": 0, "address": {"street": "Example St", "city": "Example City"}, "hobbies": ["reading", "traveling", "swimming"]},
        json_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"}
                    },
                    "required": ["street", "city"],
                    "additionalProperties": True
                },
                "hobbies": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["name", "age", "address", "hobbies"],
            "additionalProperties": True
        }, expect_type=dict)

        # 2) Fenced code block and JSON marker
        _run_case(
            "fenced_json",
            "```json\n{\"ok\": true, \"items\": [1,2,3]}\n```",
            json_example={"ok": True, "items": [1, 2, 3]},
            json_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}, "items": {"type": "array", "items": {"type": "integer"}}},
                "required": ["ok", "items"],
            },
            expect_type=dict,
        )

        # 3) Embedded JSON with trailing explanation text
        _run_case(
            "embedded_trailing_text",
            "Result: {\"a\": 1, \"b\": \"x\"} Explanation follows.",
            json_example={"a": 1, "b": "x"},
            json_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "string"}}, "required": ["a", "b"]},
            expect_type=dict,
        )

        # 4) Malformed JSON with unescaped inner quotes - we now attempt auto-repair
        _run_case(
            "malformed_unescaped_quotes",
            '{"optional_new_tasks":["Search for "GPT-OSS-20B total parameters" using WebSearch"]}',
            json_example={"optional_new_tasks": ["Search for GPT-OSS-20B total parameters using WebSearch"]},
            json_schema={"type": "object", "properties": {"optional_new_tasks": {"type": "array", "items": {"type": "string"}}}, "required": ["optional_new_tasks"]},
            expect_type=dict,
        )

        # 5) JSON wrapped in single quotes
        _run_case(
            "single_quoted_wrapper",
            "'{\"k\": \"v\"}'",
            json_example={"k": "v"},
            json_schema={"type": "object", "properties": {"k": {"type": "string"}}, "required": ["k"]},
            expect_type=dict,
        )

        # Additional malformed cases (common LLM outputs)
        # 6) Single quotes for keys and values
        _run_case(
            "single_quotes_keys_values",
            "{'a': 'b', 'c': 'd'}",
            json_example={"a": "b", "c": "d"},
            json_schema={"type": "object", "properties": {"a": {"type": "string"}, "c": {"type": "string"}}, "required": ["a", "c"]},
            expect_type=dict,
        )

        # 7) Missing comma between items
        # Expect a fixed dictionary — we now try to insert missing commas
        _run_case(
            "missing_comma",
            '{"a": 1 "b": 2}',
            json_example={"a": 1, "b": 2},
            json_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]},
            expect_type=dict,
        )

        # 8) Trailing comma
        # Our conservative repair now fixes trailing commas (e.g. '{"a": 1,}' -> '{"a": 1}'),
        # so expect a parsed dict rather than None.
        _run_case(
            "trailing_comma",
            '{"a": 1,}',
            json_example={"a": 1},
            json_schema={"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
            expect_type=dict,
        )

        # 9) Unquoted keys (javascript-like)
        # We attempt to quote unquoted keys so this should parse to a dict
        _run_case(
            "unquoted_keys",
            '{a: 1, b: 2}',
            json_example={"a": 1, "b": 2},
            json_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]},
            expect_type=dict,
        )

        # 10) Smart quotes (typographic) from copy-paste — now normalized to straight quotes
        _run_case(
            "smart_quotes",
            '{“a”: “b”}',
            json_example={"a": "b"},
            json_schema={"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
            expect_type=dict,
        )

        # 11) Equals instead of colon with single quotes (user-specified)
        _run_case(
            "equals_single_quotes",
            "{'attribute'='value'}",
            json_example={"attribute": "value"},
            json_schema={"type": "object", "properties": {"attribute": {"type": "string"}}, "required": ["attribute"]},
            expect_type=dict,
        )

        # 12) Equals instead of colon with double quotes and inner single quote (Italian examples)
        _run_case(
            "equals_double_quotes_italian_1",
            '{"attribute"="La citta\' si chiama Roma."}',
            json_example={"attribute": "La citta' si chiama Roma."},
            json_schema={"type": "object", "properties": {"attribute": {"type": "string"}}, "required": ["attribute"]},
            expect_type=dict,
        )
        _run_case(
            "equals_double_quotes_italian_2",
            '{"attribute"="La citta\' e\' chiamata Roma."}',
            json_example={"attribute": "La citta' e' chiamata Roma."},
            json_schema={"type": "object", "properties": {"attribute": {"type": "string"}}, "required": ["attribute"]},
            expect_type=dict,
        )

        # --- List / array cases (new) ---
        # 13) Basic valid JSON array
        _run_case(
            "valid_array",
            '[1, 2, 3, 4]',
            json_example=[1, 2, 3, 4],
            json_schema={"type": "array", "items": {"type": "integer"}},
            expect_type=list,
        )

        # 14) Fenced array code block
        _run_case(
            "fenced_array",
            "```json\n[\"a\", \"b\", \"c\"]\n```",
            json_example=["a", "b", "c"],
            json_schema={"type": "array", "items": {"type": "string"}},
            expect_type=list,
        )

        # 15) Array with missing commas between elements -> should be fixed
        _run_case(
            "array_missing_comma",
            '[1 2 3]',
            json_example=[1, 2, 3],
            json_schema={"type": "array", "items": {"type": "integer"}},
            expect_type=list,
        )

        # 16) Array with trailing comma -> expect list after conservative repair
        _run_case(
            "array_trailing_comma",
            '["x", "y", ]',
            json_example=["x", "y"],
            json_schema={"type": "array", "items": {"type": "string"}},
            expect_type=list,
        )

        # 17) Array with unescaped inner quotes in a string element
        _run_case(
            "array_unescaped_quotes",
            '["Search for "GPT-OSS-20B"", "another"]',
            json_example=["Search for GPT-OSS-20B", "another"],
            json_schema={"type": "array", "items": {"type": "string"}},
            expect_type=list,
        )

        # Additional list cases mirroring dictionary tests
        # 18) Embedded array with trailing text
        _run_case(
            "embedded_trailing_array_text",
            "Result: [1, 2, 3] Explanation follows.",
            json_example=[1, 2, 3],
            json_schema={"type": "array", "items": {"type": "integer"}},
            expect_type=list,
        )

        # 19) Single-quoted wrapper around an array
        _run_case(
            "single_quoted_array_wrapper",
            "'[1, 2, 3]'",
            json_example=[1, 2, 3],
            json_schema={"type": "array", "items": {"type": "integer"}},
            expect_type=list,
        )

        # 20) Single quotes for elements
        _run_case(
            "single_quotes_elements",
            "['a', 'b', 'c']",
            json_example=["a", "b", "c"],
            json_schema={"type": "array", "items": {"type": "string"}},
            expect_type=list,
        )

        # 21) Unquoted elements (JS-like) -> attempt to quote and parse
        _run_case(
            "unquoted_elements",
            '[a, b, c]',
            json_example=["a", "b", "c"],
            json_schema={"type": "array", "items": {"type": "string"}},
            expect_type=list,
        )

        # 22) Smart quotes in array elements
        _run_case(
            "smart_quotes_elements",
            '[“a”, “b”, “c”]',
            json_example=["a", "b", "c"],
            json_schema={"type": "array", "items": {"type": "string"}},
            expect_type=list,
        )

        # --- Additional edge-case array tests ---
        # 23) Nested arrays with a missing comma in the inner array
        _run_case(
            "nested_arrays",
            '[1, [2 3], 4]',
            json_example=[1, [2, 3], 4],
            json_schema={"type": "array"},
            expect_type=list,
        )

        # 24) Mixed-type array with numbers, booleans, null, objects and inner arrays (with some missing commas)
        _run_case(
            "mixed_type_elements",
            '["a", 1 true, null, {"k": "v"}, [5 6]]',
            json_example=["a", 1, None, {"k": "v"}, [5, 6]],
            json_schema={"type": "array"},
            expect_type=list,
        )

        # 25) Unquoted hyphenated tokens in array
        _run_case(
            "unquoted_hyphenated_elements",
            '[foo-bar, baz]',
            json_example=["foo-bar", "baz"],
            json_schema={"type": "array", "items": {"type": "string"}},
            expect_type=list,
        )

        # 26) Unquoted hyphenated tokens without commas
        _run_case(
            "unquoted_hyphenated_missing_commas",
            '[foo-bar baz qux-1]',
            json_example=["foo-bar", "baz", "qux-1"],
            json_schema={"type": "array", "items": {"type": "string"}},
            expect_type=list,
        )

        # --- Real-world malformed examples reproduced from logs ---
        # 27) Summary contains unescaped inner double-quotes (e.g. a diagram label "550 min. 550")
        schema_real_world = {
            "properties": {
                "Title": {"type": "string"},
                "Summary": {"type": "string"},
                "Keywords_EN": {"type": "string"}
            }
        }
        _run_case(
            "real_world_summary_unescaped_quotes",
            '''{"Title":"Dishwasher Safety and Installation Instructions","Summary":"These pages provide safety warnings and installation instructions for the AEG/Electrolux GA60SLVC 60 cm sliding-door dishwasher. The installation section references a vent diagram labeled "550 min. 550", links to YouTube tutorials and directs users to safety chapters.","Keywords_EN":"Dishwasher, AEG, Electrolux"}''',
            json_example={"Title": "Dishwasher Safety and Installation Instructions", "Summary": "These pages provide safety warnings...", "Keywords_EN": "Dishwasher, AEG, Electrolux"},
            expect_type=dict,
            json_schema=schema_real_world,
        )

        # 28) Keywords value contains many comma-separated tokens and unescaped inner quotes
        schema_keywords = {
            "properties": {
                "Title": {"type": "string"},
                "Keywords_EN": {"type": "string"}
            }
        }
        _run_case(
            "keywords_unescaped_inner_quotes",
            '''{"Title":"KW test","Keywords_EN":"Dishwasher, "AEG", "Electrolux", GA60SLVC, 60 cm sliding door, www.youtube.com/electrolux"}''',
            json_example={"Title": "KW test", "Keywords_EN": "Dishwasher, AEG, Electrolux, GA60SLVC"},
            expect_type=dict,
            json_schema=schema_keywords,
        )

        # 29) Locale-style commas inside tokens and stray commas around identifiers (e.g. 60, cm or 55,0)
        schema_locale = {
            "properties": {
                "Title": {"type": "string"},
                "Summary": {"type": "string"}
            }
        }
        _run_case(
            "locale_commas_and_malformed_numbers",
            '''{"Title":"Locale commas","Summary":"Model GA60,SLVC 60, cm sliding-door. See label "550, min. 550," for venting."}''',
            json_example={"Title": "Locale commas", "Summary": "Model GA60 SLVC 60 cm sliding-door. See label 550 min. 550 for venting."},
            expect_type=dict,
            json_schema=schema_locale,
        )
        # After running all cases, print a minimal final summary and failed tests list
        # 30) Real-world malformed query example (note the '.' near the end)
        _run_case(
            "real_world_query_jane_martha",
            '''{"messages":{"optional_replace":"<ul><li>Query about identities of individuals called Jane and Martha</li></ul>" },"optional_unnecessary_document_ids":null,"optionail_ontology_id":null,"optioal_communtyids":[].}''',
            json_example={
                "messages": {"optional_replace": "<ul><li>Query about identities of individuals called Jane and Martha</li></ul>"},
                "optional_unnecessary_document_ids": None,
                "optionail_ontology_id": None,
                "optioal_communtyids": [],
            },
            json_schema={
                "type": "object",
                "properties": {
                    "messages": {"type": "object", "properties": {"optional_replace": {"type": "string"}}},
                    "optional_unnecessary_document_ids": {"type": ["array", "null"]},
                    "optionail_ontology_id": {"type": ["string", "null"]},
                    "optioal_communtyids": {"type": "array"},
                },
            },
            expect_type=dict,
        )

        # 31) Inner quote in contracted word like kitchen"s
        schema_augmented = {
            "properties": {
                "augmented_prompt": {"type": "array", "items": {"type": "string"}},
                "thalabus_query": {"type": "string"},
                "guid": {"type": "string"}
            }
        }
        _run_case(
            "inner_quote_in_contraction",
            '''{"augmented_prompt": ["For which food is Martha famous?", "Which tool did she reach for to clean the kitchen"s floor?", "What animal did Bob meet in the garden?", "Who is Joy the Squirrel?"], "thalabus_query": "\"Martha\" or \"Bob\" or \"Joy the Squirrel\"", "guid": "b859ee18-d3fc-4e4d-89dc-60f3e843abe3"}''',
            json_example={
                "augmented_prompt": ["For which food is Martha famous?", "Which tool did she reach for to clean the kitchen's floor?", "What animal did Bob meet in the garden?", "Who is Joy the Squirrel?"],
                "thalabus_query": "\"Martha\" or \"Bob\" or \"Joy the Squirrel\"",
                "guid": "b859ee18-d3fc-4e4d-89dc-60f3e843abe3"
            },
            json_schema=schema_augmented,
            expect_type=dict,
        )

        # 32) Valid apostrophe in contraction (well-formed JSON)
        schema_bob = {
            "properties": {
                "augmented_prompt": {"type": "array", "items": {"type": "string"}},
                "thalabus_query": {"type": "string"}
            }
        }
        _run_case(
            "valid_apostrophe_in_contraction",
            '''{"augmented_prompt": ["Who is Bob? Please provide information about Bob's identity, background, or any relevant details available."], "thalabus_query": "\"Bob\""}''',
            json_example={
                "augmented_prompt": ["Who is Bob? Please provide information about Bob's identity, background, or any relevant details available."],
                "thalabus_query": "\"Bob\""
            },
            json_schema=schema_bob,
            expect_type=dict,
        )

        # 33) Unclosed brace at end - Bob and Italy temperature query
        schema_messages = {
            "properties": {
                "messages": {"type": "object", "properties": {"optional_replace": {"type": "string"}}}
            }
        }
        _run_case(
            "unclosed_brace_bob_italy",
            '''{"messages":{"optional_replace":"<ul><li>User wants information on 'Bob' and the current temperature in Italy's top three largest cities.</li></ul>"}''',
            json_example={
                "messages": {"optional_replace": "<ul><li>User wants information on 'Bob' and the current temperature in Italy's top three largest cities.</li></ul>"}
            },
            json_schema=schema_messages,
            expect_type=dict,
        )

        # 34) User-provided rank case — explanation contains unescaped inner quotes (e.g. "Bob")
        # Issue: the `explanation` string contains unescaped double quotes around "Bob", making
        # the JSON invalid. get_json_from_text() should attempt to repair this by escaping inner quotes.
        _run_case(
            "user_rank_unescaped_inner_quotes_bob",
            '''{"rank":[{"fragmentid":"docid-98-c0-p2","explanation":"Forse l'argomento di interesse "Bob" con i suoi tratti fisici e il suo ruolo nella serie.","rank":0.99}]}''',
            json_example={
                "rank": [{
                    "fragmentid": "docid-98-c0-p2",
                    "explanation": "Forse l'argomento di interesse Bob con i suoi tratti fisici e il suo ruolo nella serie.",
                    "rank": 0.99
                }]
            },
            json_schema={"type": "object"},
            expect_type=dict,
        )

        # 35) Missing closing double quote in dictionary key (LLM mistake)
        # Issue: "optional_require_exact_time:false should be "optional_require_exact_time":false
        # The key is missing its closing double quote before the colon
        _run_case(
            "missing_closing_quote_dict_key",
            '''{"optional_complex_plan": {"goals": [{"goal_sequence_id": "Step-1", "goal": "Analyze the query to identify all relevant categories related to chronic physical issues in asylum applicants from North Africa.", "dependent_on_sequence_ids": [], "recommended_plan": "plan_call_tool_but_no_answer", "optional_recommended_tools": ["QueryAnalyser"]}, {"goal_sequence_id": "Step-2", "goal": "Retrieve and analyze all cases involving chronic physical issues (MEDICAL_STATUS: CHRONIC_PHYSICAL) for asylum applicants from North Africa (COUNTRY_OF_ORIGIN: ISO 3166-1 alpha-2 codes for North African countries).", "dependent_on_sequence_ids": ["Step-1"], "recommended_plan": "plan_rag_assistant_and_answer"}]},"optional_require_exact_time":false}''',
            json_example={
                "optional_complex_plan": {"goals": [{"goal_sequence_id": "Step-1"}]},
                "optional_require_exact_time": False
            },
            json_schema={"type": "object"},
            expect_type=dict,
        )

        # 36) Complex asylum case with nested arrays and objects (malformed with stray characters)
        schema_asylum_case = {
            "type": "object",
            "properties": {
                "optional_issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue_id": {"type": "string"},
                            "description": {"type": "string"},
                            "severity": {"type": "string"},
                            "status": {"type": "string"}
                        }
                    }
                },
                "optional_recommendations": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "optional_focus_content_id": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "optional_split_to_chunks": {"type": "boolean"}
            }
        }
        _run_case(
            "complex_asylum_case_nested",
            '''{
"optional_issues": [
{
"issue_id": "i-1",
"description": "Does the situation of the applicants constitute persecution within the meaning of Article 3 of the Swiss Asylum Act? This question is legally significant because establishing persecution is the core prerequisite for granting refugee status; without it the claim for protection under Art. 3 fails.",
"severity": "high",
"status": "open"
},
{
"issue_id": "i-2",
"description": "Can the harmful acts be attributed to the state, or are they solely the conduct of private individuals? Attribution determines whether the state is responsible for the persecution; if the state is not the perpetrator, the analysis must focus on the availability of effective state protection, which can still satisfy the refugee definition under Swiss case law.",
"severity": "high",
"status": "open"
},
{
"issue_id": "i-3",
"description": "Is effective state protection unavailable or insufficient in Kosovo for the applicants? Demonstrating the lack of effective protection satisfies the "lack of state protection" element required when persecution is committed by non‑state actors, and is therefore decisive for refugee status eligibility.",
"severity": "high",
"status": "open"
},
{
"issue_id": "i-4",
"description": "Is internal relocation within Kosovo a reasonable and viable alternative for the applicants? Under Art. 31a LAsi the applicant must have exhausted internal‑flight possibilities unless such relocation is unreasonable; a finding of unreasonableness preserves admissibility of the claim.",
"severity": "high",
"status": "open"
},
{
"issue_id": "i-5",
"description": "Are the applicants' claims credible, and are there evidentiary gaps that could undermine credibility? Credibility assessment (Art. 7 LAsi) can lead to partial or total rejection of the claim; missing evidence such as police reports or medical certificates may weaken the narrative.",
"severity": "high",
"status": "open"
},
{
"issue_id": "i-6",
"description": "Is removal of the applicants enforceable in light of their serious medical conditions and the risk of inhuman or degrading treatment? This issue is crucial because, even if refugee status is denied, removal may be barred under non‑refoulement (Art. 5 LAsi), Article 83 LEtr (reasonableness, possibility, legality), and ECHR Art. 3, leading to provisional admission or suspension of removal.",
"severity": "high",
"status": "open"
}
],
"optional_recommendations": [
"Obtain full police investigation reports and any case numbers from Kosovo authorities, UNMIK and KFOR",
"Secure detailed medical certificates from treating physicians, including prognosis, treatment plan and necessity of continuous care",
"Gather documentary evidence of any additional protection requests made to local authorities (e.g., written complaints, protection orders)",
"Conduct a feasibility assessment of internal relocation options within Kosovo, considering medical mobility constraints",
"Collect information on the capacity of Kosovo health facilities to provide equivalent treatment for liver cirrhosis and psychiatric care"
],
"optional_focus_content_id": [
"docid-5185"
],
"optional_split_to_chunks": false
}''',
            json_example={
                "optional_issues": [
                    {
                        "issue_id": "i-1",
                        "description": "Does the situation of the applicants constitute persecution...",
                        "severity": "high",
                        "status": "open"
                    }
                ],
                "optional_recommendations": ["Obtain full police investigation reports..."],
                "optional_focus_content_id": ["docid-5185"],
                "optional_split_to_chunks": False
            },
            json_schema=schema_asylum_case,
            expect_type=dict,
        )

        print(f"Tests passed: {passed}")
        print(f"Tests failed: {failed}")
        if failed_tests:
            print("Failed tests:")
            for n in failed_tests:
                print("-", n)

    finally:
        log_exit()


log_end_file(__file__)