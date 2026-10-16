import re
import ast
import json

def convert_malformed_json(input_str: str) -> str:
    """
    Convert a malformed JSON-like string that uses single quotes for object keys and string
    values, and Python-style literals (null, true, false), into a properly formatted JSON string
    with double quotes.

    Args:
        input_str: The malformed JSON string.

    Returns:
        A valid JSON-formatted string with double quotes and Unicode preserved.

    Raises:
        ValueError: If the input cannot be parsed or serialized.
    """
    # Replace Python-style literals null/true/false following a colon
    def _replace_literal(match):  # pragma: no cover
        full = match.group(0)
        lit = match.group(1)
        low = lit.lower()
        if low == 'null':
            rep = 'None'
        elif low == 'true':
            rep = 'True'
        elif low == 'false':
            rep = 'False'
        else:
            rep = lit
        # preserve original spacing/prefix from colon up to the literal
        return full[:-len(lit)] + rep

    # Match ':' then optional whitespace, then one of the literals
    literal_pattern = re.compile(r':\s*(null|true|false)\b', flags=re.IGNORECASE)
    fixed = literal_pattern.sub(_replace_literal, input_str)

    try:
        # Safely evaluate the Python-like literal structure
        data = ast.literal_eval(fixed)
    except Exception as e:
        raise ValueError(f"Invalid input for conversion: {e}")

    try:
        # Dump to JSON with double quotes; preserve non-ASCII characters
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        raise ValueError(f"Error serializing to JSON: {e}")

def dict_to_html_table(data: dict) -> str:
    """Convert a dictionary to an HTML table string."""
    if not isinstance(data, dict):
        return str(data)
    html = "<table border='1'><tbody>"
    for key, value in data.items():
        html += f"<tr><th>{key}</th><td>{value if value is not None else ''}</td></tr>"
    html += "</tbody></table>"
    return html
