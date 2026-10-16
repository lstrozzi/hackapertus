from __future__ import annotations

import re
import hashlib
from typing import List
from bs4 import BeautifulSoup
import markdown  # Added import for markdown library
from lxml import html, etree
import unicodedata
from src.utils.log import log, ERROR


def strip_outer_code_fences(text: str) -> str:
    """Remove transport fences at the text boundaries while preserving inner fences."""
    if not text or not isinstance(text, str):
        return text

    lines: List[str] = text.strip().splitlines()
    fence_pattern = re.compile(r"^\s*`{3,}(?:[A-Za-z0-9_-]+)?\s*$")

    while lines and fence_pattern.fullmatch(lines[0]):
        lines.pop(0)
    while lines and fence_pattern.fullmatch(lines[-1]):
        lines.pop()

    return "\n".join(lines).strip()


def generate_html_from_text(text: str) -> str:
    """
    Generate HTML from text using the markdown library. The text will be converted with the following rules:
    - Markdown headers are converted to h1, h2, etc.
    - Lists, tables, and other Markdown structures are preserved.
    - Newlines at the end of non-empty lines are converted to <br> tags.
    - Existing HTML structures are preserved as-is.
    - HTML comments are preserved.
    :param text: Text
    :return: HTML
    """
    if not text:
        return ""

    # Convert Markdown to HTML
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "nl2br"])

    # Post-process the HTML to ensure compatibility with the expected output
    soup = BeautifulSoup(html, "html.parser")

    return soup.prettify()


def strip_html_element_attributes(html_text: str, keep_attributes: List[str] = None) -> str:
    """Remove all attributes from HTML elements, keeping only those listed in keep_attributes.

    :param html_text: HTML string to process.
    :param keep_attributes: attribute names to preserve (e.g. ["id"]). If None, all attributes are removed.
    :return: HTML string with unwanted attributes stripped.
    """
    keep = set(keep_attributes or [])
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup.find_all(True):
        tag.attrs = {k: v for k, v in tag.attrs.items() if k in keep}
    return str(soup)


def generate_text_from_html(html:str) -> str:
    """
    Generate text from HTML. The HTML will be converted with the following rules:
    - All HTML tags (including <p>, <p id="...">, etc.) will be removed
    - <br> tags will be replaced with newlines
    :param html: HTML
    :return: Text
    """
    # Remove all HTML tags using regex
    # This handles both simple tags (<p>) and tags with attributes (<p id="text-1">)
    text = re.sub(r'<[^>]+>', '', html)
    
    # Clean up the remaining content
    # Replace multiple newlines with single newlines
    text = re.sub(r'\n+', '\n', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text

def generate_markdown_from_html(html: str) -> str:
    """
    Convert HTML (as produced by enhanced_text) back to Markdown.
    Handles the common structural elements used in Thalabus documents:
    headings (h1-h6), paragraphs, line breaks, bold/italic, ordered and
    unordered lists, block-quotes, code blocks, horizontal rules, and links.
    HTML element IDs (e.g. id="p-3") are stripped from the output.
    :param html: HTML string
    :return: Markdown string
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    def _convert_node(node) -> str:
        from bs4 import NavigableString, Tag
        if isinstance(node, NavigableString):
            return str(node)

        tag = node.name.lower() if node.name else ""
        inner = "".join(_convert_node(c) for c in node.children)

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            return f"\n{'#' * level} {inner.strip()}\n"
        elif tag == "p":
            text = inner.strip()
            return f"\n{text}\n" if text else ""
        elif tag == "br":
            return "\n"
        elif tag in ("strong", "b"):
            text = inner.strip()
            return f"**{text}**" if text else ""
        elif tag in ("em", "i"):
            text = inner.strip()
            return f"*{text}*" if text else ""
        elif tag == "code":
            return f"`{inner}`"
        elif tag == "pre":
            return f"\n```\n{inner.strip()}\n```\n"
        elif tag == "blockquote":
            lines = inner.strip().splitlines()
            return "\n" + "\n".join(f"> {l}" for l in lines) + "\n"
        elif tag == "hr":
            return "\n---\n"
        elif tag == "a":
            href = node.get("href", "")
            text = inner.strip()
            return f"[{text}]({href})" if href else text
        elif tag == "ul":
            items = [c for c in node.children if getattr(c, "name", None) == "li"]
            lines = ["- " + "".join(_convert_node(ch) for ch in item.children).strip() for item in items]
            return "\n" + "\n".join(lines) + "\n"
        elif tag == "ol":
            items = [c for c in node.children if getattr(c, "name", None) == "li"]
            lines = [f"{i+1}. " + "".join(_convert_node(ch) for ch in item.children).strip() for i, item in enumerate(items)]
            return "\n" + "\n".join(lines) + "\n"
        elif tag == "li":
            return inner  # handled by parent ul/ol
        elif tag in ("div", "section", "article", "main", "body", "html", "root"):
            return inner
        elif tag in ("span",):
            return inner
        elif tag in ("table", "thead", "tbody", "tfoot", "tr", "th", "td"):
            return inner  # basic pass-through; full table rendering is out of scope
        else:
            return inner  # unknown tags: return inner content

    md = _convert_node(soup)

    # Collapse 3+ consecutive blank lines to 2
    md = re.sub(r'\n{3,}', '\n\n', md)
    return md.strip()

def deduplicate_classes(css_classes: list) -> list:
    """
    Deduplicates and preserves order of CSS classes.
    :param css_classes: List of CSS class names
    :return: List with duplicates removed, preserving order
    """
    seen = set()
    result = []
    for cls in css_classes:
        if cls not in seen:
            seen.add(cls)
            result.append(cls)
    return result

def strip_html_tag_ids(html: str) -> str:
    """
    Strips IDs from HTML tags. Any id attributes will be removed, regardless of the format. 
    Example: <p id="test-1">Hello World</p> will be converted to <p>Hello World</p>
    :param html: HTML
    :return: HTML with IDs removed
    """
    return re.sub(r'(<\w+)\s+id="[^"]*"', r'\1', html)

def add_html_tag_ids(html: str) -> str:
    """
    Add unique IDs to all HTML tags. This is useful for testing purposes.
    Each HTML tag will be modified to include an id attribute with a unique value.
    The id attribute must be in the format: id="<tagname>-<number>" where <tagname> is the tag name and <number> is a unique number.
    :param html: HTML
    :return: HTML with unique IDs
    """
    soup = BeautifulSoup(html, "html.parser")
    id_counter = 1

    for tag in soup.find_all(True):  # True finds all tags
        if 'id' not in tag.attrs:
            if tag.name == 'br':
                continue
            tag['id'] = f"{tag.name}-{id_counter}"
            id_counter += 1

    return str(soup)

def strip_ids_from_html(html: str) -> str:
    """
    Strip IDs from HTML tags.
    :param html: HTML
    :return: HTML with IDs removed
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(True):  # True finds all tags
        if 'id' in tag.attrs:
            del tag.attrs['id']
    return str(soup)

def replace_text_in_element(enhanced_text: str, element_id: str, text_to_be_replaced: str, new_text: str, new_tag: str = None) -> str:
    # this function replaces a substring 'text_to_be_replaced' within an enhanced_text, with new_text.
    # if the new_text is None or empty string, the element is removed altogether.
    # if an optional new_tag is present, the text is wrapped in the new_tag. Else it's simply replaced
    # if the element does not exist, the enhanced_text is returned as is.

    soup = BeautifulSoup(enhanced_text, "html.parser")
    
    element = soup.find(id=element_id)
    if element:
        if text_to_be_replaced is None or text_to_be_replaced == "":
            text_to_be_replaced = element.text

        if not new_text or new_text == "":
            element.decompose()
        elif new_tag:
            new_element = soup.new_tag(new_tag, **{"class": "thalabus-edit"})
            new_element.string = new_text
            element_text = element.text
            start_index = element_text.find(text_to_be_replaced)
            if start_index != -1:
                end_index = start_index + len(text_to_be_replaced)
                element_text = element_text[:start_index] + str(new_element) + element_text[end_index:]
                element.clear()
                element.append(BeautifulSoup(element_text, "html.parser"))
        else:
            new_content = element.text.replace(text_to_be_replaced, new_text)
            element.clear()
            element.append(BeautifulSoup(new_content, "html.parser"))
            css_classes = deduplicate_classes(element.get("class", []))
            if "thalabus-edit" not in css_classes:
                element["class"] = css_classes + ["thalabus-edit"]
    
    return str(soup)

def replace_text_globally(enhanced_text: str, text_to_be_replaced: str, new_text: str) -> str:
    # Replaces all occurrences of text_to_be_replaced globally in HTML.
    #
    # Three cases are handled:
    # 1. The text is the complete text of a single leaf element:
    #    → replace its content and add class="thalabus-edit" to that element
    # 2. The text appears only inside HTML attribute values (e.g. style="color: darkgreen;"):
    #    → replace in every matching attribute value and add class="thalabus-edit" to those elements
    # 3. The text spans (part of) one or more text nodes:
    #    → wrap each replaced occurrence in <span class="thalabus-edit">new_text</span>
    #
    # new_text can contain HTML markup which will be parsed and inserted.

    from bs4 import NavigableString, Tag

    soup = BeautifulSoup(enhanced_text, "html.parser")

    # ── Case 1: entire text content of a single leaf element ──────────────────
    for element in soup.find_all(True):
        if element.get_text() == text_to_be_replaced and not element.find_all():
            css_classes = deduplicate_classes(element.get("class", []))
            if "thalabus-edit" not in css_classes:
                element["class"] = css_classes + ["thalabus-edit"]
            element.clear()
            if new_text:
                for child in BeautifulSoup(new_text, "html.parser").contents:
                    element.append(child)
            return str(soup)

    # ── Case 2: text occurs in attribute values (e.g. style attributes) ───────
    # Check whether there is at least one text node containing the search text.
    def _has_text_node_match(s):
        for node in s.find_all(string=True):
            if isinstance(node, NavigableString) and not isinstance(node, Tag):
                if text_to_be_replaced in str(node):
                    return True
        return False

    if not _has_text_node_match(soup):
        # No text node matches → look in attribute values
        attr_matched = False
        for element in soup.find_all(True):
            for attr_name, attr_value in list(element.attrs.items()):
                if attr_name.startswith("previous-"):
                    continue
                if isinstance(attr_value, str) and text_to_be_replaced in attr_value:
                    # Store original value for rollback
                    element[f"previous-{attr_name}"] = attr_value
                    element[attr_name] = attr_value.replace(text_to_be_replaced, new_text or "")
                    css_classes = deduplicate_classes(element.get("class", []))
                    if "thalabus-edit" not in css_classes:
                        element["class"] = css_classes + ["thalabus-edit"]
                    attr_matched = True
        if attr_matched:
            return str(soup)
        # Nothing found anywhere — return unchanged
        return enhanced_text

    # ── Case 3: text appears in text nodes → wrap in <span class="thalabus-edit"> ──
    def replace_in_element(elem):
        new_children = []
        modified = False
        for child in list(elem.children):
            if isinstance(child, NavigableString) and not isinstance(child, Tag):
                text_str = str(child)
                if text_to_be_replaced in text_str:
                    parts = text_str.split(text_to_be_replaced)
                    for part in parts[:-1]:
                        if part:
                            new_children.append(NavigableString(part))
                        new_span = soup.new_tag("span", **{"class": "thalabus-edit"})
                        if new_text:
                            for span_child in BeautifulSoup(new_text, "html.parser").contents:
                                new_span.append(span_child)
                        new_children.append(new_span)
                    if parts[-1]:
                        new_children.append(NavigableString(parts[-1]))
                    modified = True
                else:
                    new_children.append(child)
            elif isinstance(child, Tag):
                replace_in_element(child)
                new_children.append(child)
            else:
                new_children.append(child)
        if modified:
            elem.clear()
            for child in new_children:
                elem.append(child)

    replace_in_element(soup)
    return str(soup)

def append_text_in_element(enhanced_text: str, element_id: str, anchor_text: str, text_to_insert: str, before: bool) -> str:
    # Inserts text_to_insert immediately before or after the first occurrence of anchor_text inside the element.
    # before=True  → result: "...text_to_insert anchor_text..."
    # before=False → result: "...anchor_text text_to_insert..."
    # If anchor_text is not found in the element, enhanced_text is returned unchanged.
    soup = BeautifulSoup(enhanced_text, "html.parser")
    element = soup.find(id=element_id)
    if element and anchor_text:
        replacement = (text_to_insert + anchor_text) if before else (anchor_text + text_to_insert)
        current_text = element.get_text()
        if anchor_text in current_text:
            new_text = current_text.replace(anchor_text, replacement, 1)
            element.string = new_text
            css_classes = deduplicate_classes(element.get("class", []))
            if "thalabus-edit" not in css_classes:
                element["class"] = css_classes + ["thalabus-edit"]
    return str(soup)

def merge_2_tags(enhanced_text: str, element_id1: str, element_id2: str, content_separator: str, consolidate_hypenated_words: bool) -> str:
    # Identify element1
    pattern1 = rf'(<(\w+)([^>]*) id="{element_id1}"([^>]*)>)(.*?)(</\2>)'
    element1 = re.search(pattern1, enhanced_text, flags=re.DOTALL)
    if element1 is None:
        return enhanced_text

    # Identify element2
    pattern2 = rf'(<(\w+)([^>]*) id="{element_id2}"([^>]*)>)(.*?)(</\2>)'
    element2 = re.search(pattern2, enhanced_text, flags=re.DOTALL)
    if element2 is None:
        return enhanced_text

    # Merge content of element2 into element1
    content1 = element1.group(5)
    content2 = element2.group(5)
    if consolidate_hypenated_words:
        # Consolidate hyphenated words at the end of the first element and the beginning of the second element
        hyphen_pattern = rf'(\w+)-\s*$'
        match = re.search(hyphen_pattern, content1)
        if match:
            content1 = content1[:match.start(1)] + match.group(1)
            content2 = content2.lstrip()
            content_separator = ''  # Remove separator if consolidating hyphenated words
    
    replacement = rf'\1{content1}{content_separator}{content2}\6'
    enhanced_text = re.sub(pattern1, replacement, enhanced_text, flags=re.DOTALL)

    # Remove element2 from enhanced_text
    enhanced_text = re.sub(pattern2, '', enhanced_text, flags=re.DOTALL)
    return (enhanced_text or "").strip()

def merge_tags(enhanced_text: str, element_ids: List[str], content_separator: str, consolidate_hypenated_words:bool=False) -> str:
    # merges all tags in element_ids into the first tag
    # this function uses merge_2_tags to merge 2 tags at a time
    if len(element_ids) < 2:
        return enhanced_text
    
    element_id1 = element_ids[0]
    for idx in range(1, len(element_ids)):
        element_id2 = element_ids[idx]
        enhanced_text = merge_2_tags(enhanced_text, element_id1, element_id2, content_separator, consolidate_hypenated_words)

    return enhanced_text

def eliminate_duplicate_newlines(text: str) -> str:
    # first, replace all lines with white spaces to empty lines
    text = re.sub(r"\n\s*\n", "\n\n", text)

    # then, replace all multiple newlines with a single newline
    return re.sub(r"\n+", "\n", text)

def add_html_block_newlines(html: str, indent: str = "  ") -> str:
    """
    Pretty-prints HTML with proper indentation for block-level elements.
    Inline elements stay on the same line as their surrounding content.
    Mixed content (text + inline tags alongside block children) is handled by
    grouping consecutive inline/text nodes on a single indented line.
    :param html: HTML string
    :param indent: Indentation string per level (default: two spaces)
    :return: Indented HTML string
    """
    from bs4 import NavigableString, Tag

    BLOCK_TAGS = {
        'div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li', 'table', 'thead', 'tbody', 'tfoot', 'tr', 'td', 'th',
        'section', 'article', 'header', 'footer', 'blockquote', 'pre',
        'details', 'summary', 'figure', 'figcaption',
    }
    VOID_TAGS = {'br', 'hr', 'img', 'input', 'meta', 'link', 'area', 'base', 'col', 'embed', 'param', 'source', 'track', 'wbr'}

    def _render_inline(node) -> str:
        if isinstance(node, NavigableString):
            return str(node)
        tag = node.name
        if tag is None:
            return "".join(_render_inline(c) for c in node.children)
        attrs_str = _attrs(node)
        if tag.lower() in VOID_TAGS:
            return f"<{tag}{attrs_str}>"
        inner = "".join(_render_inline(c) for c in node.children)
        return f"<{tag}{attrs_str}>{inner}</{tag}>"

    def _attrs(node) -> str:
        result = ""
        for k, v in node.attrs.items():
            v_str = " ".join(v) if isinstance(v, list) else str(v)
            result += f' {k}="{v_str}"'
        return result

    def _render(node, level: int) -> str:
        if isinstance(node, NavigableString):
            return str(node)
        tag = node.name
        if tag is None:
            return "".join(_render(c, level) for c in node.children)
        tag_lower = tag.lower()
        is_block = tag_lower in BLOCK_TAGS
        is_void = tag_lower in VOID_TAGS
        pad = indent * level
        attrs_str = _attrs(node)
        if is_void:
            return f"{pad}<{tag}{attrs_str}>" if is_block else f"<{tag}{attrs_str}>"
        open_tag = f"<{tag}{attrs_str}>"
        close_tag = f"</{tag}>"
        children = list(node.children)
        has_block_children = any(isinstance(c, Tag) and c.name and c.name.lower() in BLOCK_TAGS for c in children)
        if is_block and has_block_children:
            parts = []
            inline_buf = []
            for c in children:
                if isinstance(c, Tag) and c.name and c.name.lower() in BLOCK_TAGS:
                    if inline_buf:
                        joined = "".join(inline_buf).strip()
                        if joined:
                            parts.append(f"{indent * (level + 1)}{joined}")
                        inline_buf = []
                    parts.append(_render(c, level + 1))
                else:
                    inline_buf.append(_render_inline(c))
            if inline_buf:
                joined = "".join(inline_buf).strip()
                if joined:
                    parts.append(f"{indent * (level + 1)}{joined}")
            return f"{pad}{open_tag}\n" + "\n".join(parts) + f"\n{pad}{close_tag}"
        elif is_block:
            inner = "".join(_render_inline(c) for c in children)
            return f"{pad}{open_tag}{inner.strip()}{close_tag}"
        else:
            inner = "".join(_render_inline(c) for c in children)
            return f"{open_tag}{inner}{close_tag}"

    soup = BeautifulSoup(html, "html.parser")
    parts = [_render(child, 0) for child in soup.children if not isinstance(child, NavigableString) or str(child).strip()]
    return "\n".join(parts)

def consolidate_unnumbered_list(html_text: str, list_parameters: List[List[str]]) -> str:
    # creates an unnumbered list with existing element ids
    # Example:
    #   html_text = <h3 id="h3-16">System Context Diagram</h3>
    #               <p id="p-17">The system architecture of the ABC CRM application consists of several key components:</p>
    #               <p id="p-18">User Interface (UI):</p>
    #               <p id="p-19">The User Interface is built with React.js and includes various UI components to provide a responsive and intuitive user experience. Users interact with the system through this interface.</p>
    #               <p id="p-20">API Server:</p>
    #               <p id="p-21">The API Server is developed using Node.js and Express. It handles the business logic and data processing tasks. It includes subcomponents for business logic, authentication, and authorization, ensuring secure and efficient handling of user requests and data.</p>
    #               <p id="p-22">Database:</p>
    #               <p id="p-23">The Database is a MySQL database that stores critical information related to customer data, sales data, marketing data, and support data. It ensures the structured and reliable storage of all CRM-related information.</p>
    #               <p id="p-24">Third-Party Integrations:</p>
    #               <p id="p-25">The system integrates with various third-party services such as email providers, social media platforms, and payment gateways. This integration layer allows the CRM to extend its functionality and interact with external applications seamlessly.</p>
    #               <p id="p-26">This architecture ensures that the ABC CRM application is scalable, secure, and capable of handling complex customer relationship management tasks efficiently</p>
    #               <p id="p-27">The diagram shows integration with the BCD and CDE application, along with REST API usage.</p>
    #   list_parameters=[["p-17", "p-18"], ["p-20", "p-21"]], ["p-22", "p-23"]], ["p-24", "p-25"]]
    # Output:
    #               <h3 id="h3-16">System Context Diagram</h3>
    #               <p id="p-17">The system architecture of the ABC CRM application consists of several key components:</p>
    #               <ul id="ul-1">
    #                   <li id="li-18">
    #                       <span id="p-18">User Interface (UI):</span>
    #                       <span id="p-19">The User Interface is built with React.js and includes various UI components to provide a responsive and intuitive user experience. Users interact with the system through this interface.</span>
    #                   </li>
    #                   <li id="li-20">
    #                       <span id="p-20">API Server:</span>
    #                       <span id="p-21">The API Server is developed using Node.js and Express. It handles the business logic and data processing tasks. It includes subcomponents for business logic, authentication, and authorization, ensuring secure and efficient handling of user requests and data.</span>
    #                   </li>
    #                   <li id="li-22">
    #                       <span id="p-22">Database:</span>
    #                       <span id="p-23">The Database is a MySQL database that stores critical information related to customer data, sales data, marketing data, and support data. It ensures the structured and reliable storage of all CRM-related information.</span>
    #                   </li>
    #                   <li id="li-24">
    #                       <span id="p-24">Third-Party Integrations:</span>
    #                       <span id="p-25">The system integrates with various third-party services such as email providers, social media platforms, and payment gateways. This integration layer allows the CRM to extend its functionality and interact with external applications seamlessly.</span>
    #                   </li>
    #               </ul>
    #               <p id="p-26">This architecture ensures that the ABC CRM application is scalable, secure, and capable of handling complex customer relationship management tasks efficiently</p>
    #               <p id="p-27">The diagram shows integration with the BCD and CDE application, along with REST API usage.</p>

# Plan for the algorithm:
    # 1. Parse the enhanced_text into a BeautifulSoup object
    # 2. Create a new <ul> element with a unique id
    # 3. For each list in list_parameters:
    #    a. For each element in the list:
    #       i. Find the element with the given id
    #       ii. Create a new <li> element with a unique id
    #       iii. If the element is found, move it inside the <li> element
    #       iv. Append the <li> element to the <ul> element
    # 4. Remove empty <p>
    # 5. Convert the BeautifulSoup object back to a string

    # 1. Parse the enhanced_text into a BeautifulSoup object
    soup = BeautifulSoup(f"<root>{html_text}</root>", "html.parser")

    # 2. Create a new <ul> element with a unique id
    ul = soup.new_tag("ul", id="ul-unique")
    first_element = soup.find(id=list_parameters[0][0])
    if first_element is None:
        log(ERROR, f"consolidate_unnumbered_list: element '{list_parameters[0][0]}' not found in html — cannot create list. Check that all ids in list_parameters exist in the document.")
        return html_text  # return unchanged
    first_element.insert_before(ul)

    # 3. For each list in list_parameters:
    for list_parameter in list_parameters:
        # a. Create a new <li> element with a unique id
        li = soup.new_tag("li", id=f"li-{list_parameter[0].split('-')[1]}")
        
        # b. For each element in the list_parameter pair:
        for element_id in list_parameter:
            # i. Find the element with the given id
            element = soup.find(id=element_id)
            if element is None:
                continue

            # ii. Create a new <span> element with the same id
            span = soup.new_tag("span", id=element_id)

            # iii. Move the content into the <span> element
            for child in element.children:
                span.append(child.extract())

            # iv. Append the <span> to the <li>
            li.append(span)
        
        # v. Append the <li> to the <ul> element
        ul.append(li)

    # 4. Remove empty <p> tags
    for p in soup.find_all('p'):
        if not (p.text or "").strip():
            p.decompose()

    # Ensure the <root> tag exists
    if not soup.find("root"):
        root = soup.new_tag("root")
        root.append(soup)
        soup = BeautifulSoup(str(root), "html.parser")

    print(soup.prettify())

    # 5. Convert the BeautifulSoup object back to a string
    result = str(soup.find("root")).replace("<root>", "").replace("</root>", "")
    return result

def append_element_after(enhanced_text: str, element_ids: List[str], text: str, new_tag: str) -> str:
    soup = BeautifulSoup(enhanced_text, "html.parser")

    new_tag_initial = new_tag[0]

    def set_content(element, content: str):
        # Parse content as HTML so embedded markup (with single-quoted attributes) is preserved;
        # BeautifulSoup normalises single-quote attributes to double-quote on render.
        parsed = BeautifulSoup(content, "html.parser")
        for child in list(parsed.contents):
            element.append(child)

    if not element_ids:
        # No anchor element specified: append at the end of the document
        seq = 1
        while soup.find(id=f"end-{new_tag_initial}-{seq}"):
            seq += 1
        new_element_id = f"end-{new_tag_initial}-{seq}"
        new_element = soup.new_tag(new_tag, id=new_element_id, **{"class": "thalabus-edit"})
        set_content(new_element, text)
        soup.append(new_element)
        return str(soup)

    # find the sequence number of the new tag
    seq = 1
    while soup.find(id=f"{element_ids[0]}-{new_tag_initial}-{seq}"):
        seq += 1

    for element_id in element_ids:
        element = soup.find(id=element_id)
        if element:
            new_element_id = f"{element_id}-{new_tag_initial}-{seq}"
            new_element = soup.new_tag(new_tag, id=new_element_id, **{"class": "thalabus-edit"})
            set_content(new_element, text)
            element.insert_after(new_element)

    return str(soup)

def append_element_before(enhanced_text: str, element_ids: List[str], text: str, new_tag: str) -> str:
    soup = BeautifulSoup(enhanced_text, "html.parser")

    new_tag_initial = new_tag[0]

    def set_content(element, content: str):
        # Parse content as HTML so embedded markup (with single-quoted attributes) is preserved;
        # BeautifulSoup normalises single-quote attributes to double-quote on render.
        parsed = BeautifulSoup(content, "html.parser")
        for child in list(parsed.contents):
            element.append(child)

    if not element_ids:
        # No anchor element specified: prepend at the beginning of the document
        seq = 1
        while soup.find(id=f"start-{new_tag_initial}-{seq}"):
            seq += 1
        new_element_id = f"start-{new_tag_initial}-{seq}"
        new_element = soup.new_tag(new_tag, id=new_element_id, **{"class": "thalabus-edit"})
        set_content(new_element, text)
        soup.insert(0, new_element)
        return str(soup)

    # find the sequence number of the new tag
    seq = 1
    while soup.find(id=f"{element_ids[0]}-{new_tag_initial}-{seq}"):
        seq += 1

    for element_id in element_ids:
        element = soup.find(id=element_id)
        if element:
            new_element_id = f"{element_id}-{new_tag_initial}-{seq}"
            new_element = soup.new_tag(new_tag, id=new_element_id, **{"class": "thalabus-edit"})
            set_content(new_element, text)
            element.insert_before(new_element)

    return str(soup)

def _assign_element_id_end(soup, new_element, prefix: str) -> None:
    """Assign a unique id of the form '{prefix}-{tag_initial}-{seq}' to new_element."""
    tag_initial = new_element.name[0]
    seq = 1
    while soup.find(id=f"{prefix}-{tag_initial}-{seq}"):
        seq += 1
    new_element["id"] = f"{prefix}-{tag_initial}-{seq}"

def insert_html_element_after(enhanced_text: str, element_ids: List[str], html: str) -> str:
    # Inserts a complete HTML element (provided as raw html string) after the elements identified by element_ids.
    # All top-level elements in html are inserted in order; each receives a unique id and thalabus-edit class.
    soup = BeautifulSoup(enhanced_text, "html.parser")

    parsed = BeautifulSoup(html, "html.parser")
    top_elements = [el for el in parsed.contents if hasattr(el, 'name') and el.name]
    if not top_elements:
        return enhanced_text

    if not element_ids:
        fresh = BeautifulSoup(html, "html.parser")
        fresh_elements = [el for el in fresh.contents if hasattr(el, 'name') and el.name]
        for new_element in fresh_elements:
            _assign_element_id_end(soup, new_element, "end")
            css_classes = deduplicate_classes(new_element.get("class", []))
            if "thalabus-edit" not in css_classes:
                new_element["class"] = css_classes + ["thalabus-edit"]
            soup.append(new_element)
        return str(soup)

    for element_id in element_ids:
        element = soup.find(id=element_id)
        if element:
            fresh = BeautifulSoup(html, "html.parser")
            fresh_elements = [el for el in fresh.contents if hasattr(el, 'name') and el.name]
            prev_el = element
            for new_element in fresh_elements:
                _assign_element_id_end(soup, new_element, element_id)
                css_classes = deduplicate_classes(new_element.get("class", []))
                if "thalabus-edit" not in css_classes:
                    new_element["class"] = css_classes + ["thalabus-edit"]
                prev_el.insert_after(new_element)
                prev_el = new_element

    return str(soup)

def delete_html_elements(enhanced_text: str, element_ids: List[str]) -> str:
    # Marks the elements identified by element_ids as pending deletion by adding 'thalabus-edit' and 'thalabus-delete' classes.
    # The actual removal happens in accept_changes_util when accept=True; on reject, the classes are simply removed.
    soup = BeautifulSoup(enhanced_text, "html.parser")
    for element_id in element_ids:
        element = soup.find(id=element_id.strip())
        if element:
            css_classes = deduplicate_classes(element.get("class", []))
            if "thalabus-edit" not in css_classes:
                css_classes.append("thalabus-edit")
            if "thalabus-delete" not in css_classes:
                css_classes.append("thalabus-delete")
            element["class"] = css_classes
    return str(soup)

def insert_html_element_before(enhanced_text: str, element_ids: List[str], html: str) -> str:
    # Inserts a complete HTML element (provided as raw html string) before the elements identified by element_ids.
    # All top-level elements in html are inserted in order; each receives a unique id and thalabus-edit class.
    soup = BeautifulSoup(enhanced_text, "html.parser")

    parsed = BeautifulSoup(html, "html.parser")
    top_elements = [el for el in parsed.contents if hasattr(el, 'name') and el.name]
    if not top_elements:
        return enhanced_text

    if not element_ids:
        fresh = BeautifulSoup(html, "html.parser")
        fresh_elements = [el for el in fresh.contents if hasattr(el, 'name') and el.name]
        for i, new_element in enumerate(fresh_elements):
            _assign_element_id_end(soup, new_element, "start")
            css_classes = deduplicate_classes(new_element.get("class", []))
            if "thalabus-edit" not in css_classes:
                new_element["class"] = css_classes + ["thalabus-edit"]
            soup.insert(i, new_element)
        return str(soup)

    for element_id in element_ids:
        element = soup.find(id=element_id)
        if element:
            fresh = BeautifulSoup(html, "html.parser")
            fresh_elements = [el for el in fresh.contents if hasattr(el, 'name') and el.name]
            for new_element in fresh_elements:
                _assign_element_id_end(soup, new_element, element_id)
                css_classes = deduplicate_classes(new_element.get("class", []))
                if "thalabus-edit" not in css_classes:
                    new_element["class"] = css_classes + ["thalabus-edit"]
                element.insert_before(new_element)

    return str(soup)

def create_html_list(enhanced_text: str, element_ids: List[str], list_tag: str = "ul") -> str:
    # Converts a set of elements into a <ul> or <ol> list.
    # The text content of each specified element becomes a <li> item.
    # The new list is inserted before the first specified element.
    # All specified elements are marked as pending deletion with thalabus-edit/thalabus-delete classes.
    soup = BeautifulSoup(enhanced_text, "html.parser")

    elements = []
    for eid in element_ids:
        el = soup.find(id=eid.strip())
        if el:
            elements.append(el)

    if not elements:
        return enhanced_text

    # Assign a unique id to the new list element
    seq = 1
    while soup.find(id=f"{list_tag}-{seq}"):
        seq += 1
    list_id = f"{list_tag}-{seq}"

    # Build the new list element
    list_el = soup.new_tag(list_tag, id=list_id)
    list_el["class"] = ["thalabus-edit"]

    for el in elements:
        li = soup.new_tag("li")
        li.string = el.get_text(strip=True)
        list_el.append(li)

    # Insert the new list before the first selected element
    elements[0].insert_before(list_el)

    # Mark original elements as pending deletion
    for el in elements:
        css_classes = deduplicate_classes(el.get("class", []))
        if "thalabus-edit" not in css_classes:
            css_classes.append("thalabus-edit")
        if "thalabus-delete" not in css_classes:
            css_classes.append("thalabus-delete")
        el["class"] = css_classes

    return str(soup)

def dissolve_html_list(enhanced_text: str, element_ids: List[str], new_tag: str = "p") -> str:
    # Dissolves a <ul> or <ol> list into individual block elements.
    # Each <li> child becomes a new element with new_tag, inserted after the list in order.
    # The list element is marked as pending deletion with thalabus-edit/thalabus-delete classes.
    soup = BeautifulSoup(enhanced_text, "html.parser")

    for element_id in element_ids:
        list_el = soup.find(id=element_id.strip())
        if list_el is None or list_el.name not in ("ul", "ol"):
            continue

        li_items = [li for li in list_el.children if hasattr(li, "name") and li.name == "li"]
        if not li_items:
            continue

        prev_el = list_el
        for li in li_items:
            new_el = soup.new_tag(new_tag)
            new_el.string = li.get_text(strip=True)
            new_el["class"] = ["thalabus-edit"]
            prev_el.insert_after(new_el)
            prev_el = new_el

        # Mark the list as pending deletion
        css_classes = deduplicate_classes(list_el.get("class", []))
        if "thalabus-edit" not in css_classes:
            css_classes.append("thalabus-edit")
        if "thalabus-delete" not in css_classes:
            css_classes.append("thalabus-delete")
        list_el["class"] = css_classes

    return str(soup)

def move_html_elements(enhanced_text: str, element_ids: List[str], target_element_id: str, position: str) -> str:
    # Moves elements identified by element_ids to a new position relative to target_element_id.
    # The elements are cloned at the destination with class="thalabus-edit";
    # the originals are marked with class="thalabus-edit thalabus-delete" (pending deletion).
    # Positions: BEFORE, AFTER, INSIDE_FIRST, INSIDE_LAST.
    soup = BeautifulSoup(enhanced_text, "html.parser")

    target = soup.find(id=target_element_id.strip()) if target_element_id else None
    if target is None:
        return enhanced_text

    elements = []
    for eid in element_ids:
        el = soup.find(id=eid.strip())
        if el:
            elements.append(el)

    if not elements:
        return enhanced_text

    def make_clone(el):
        # Serialise + re-parse to produce a detached copy independent of the original tree.
        fresh = BeautifulSoup(str(el), "html.parser")
        clone = next((c for c in fresh.contents if hasattr(c, "name") and c.name), None)
        if clone is not None:
            css_classes = deduplicate_classes(clone.get("class", []))
            if "thalabus-edit" not in css_classes:
                clone["class"] = css_classes + ["thalabus-edit"]
        return clone

    if position == "BEFORE":
        for el in elements:
            clone = make_clone(el)
            if clone is not None:
                target.insert_before(clone)
    elif position == "AFTER":
        prev = target
        for el in elements:
            clone = make_clone(el)
            if clone is not None:
                prev.insert_after(clone)
                prev = clone
    elif position == "INSIDE_FIRST":
        for el in reversed(elements):   # reversed so document order is preserved inside the target
            clone = make_clone(el)
            if clone is not None:
                target.insert(0, clone)
    elif position == "INSIDE_LAST":
        for el in elements:
            clone = make_clone(el)
            if clone is not None:
                target.append(clone)

    # Mark originals as pending deletion
    for el in elements:
        css_classes = deduplicate_classes(el.get("class", []))
        if "thalabus-edit" not in css_classes:
            css_classes.append("thalabus-edit")
        if "thalabus-delete" not in css_classes:
            css_classes.append("thalabus-delete")
        el["class"] = css_classes

    return str(soup)

def accept_changes_util(enhanced_text: str, accept: bool) -> str:
    # Removes the "thalabus-edit" class from HTML elements.
    # If accept is True:  removes the class and all "previous-*" rollback attributes.
    # If accept is False: removes the class, restores "previous-content" text and/or
    #                     any "previous-{attr}" attribute values (e.g. previous-style).
    #
    # <span class="thalabus-edit"> that was NEWLY ADDED as a wrapper (case 3, no previous-* attrs):
    #   → unwrapped entirely (tag removed, content kept).
    # Any other element with "thalabus-edit" (including existing spans with previous-* attrs, case 2):
    #   → class removed, previous-* attrs handled, tag kept.

    soup = BeautifulSoup(enhanced_text, "html.parser")
    for element in list(soup.find_all(class_="thalabus-edit")):
        if element.attrs is None:  # element was destroyed (decomposed) as a child of an earlier decompose() call
            continue
        has_previous_attrs = any(k.startswith("previous-") for k in (element.attrs or {}))
        is_new_wrapper_span = element.name == "span" and not has_previous_attrs
        is_deletion = "thalabus-delete" in (element.get("class") or [])

        if is_deletion:
            if accept:
                element.decompose()
            else:
                css_classes = [c for c in element.get("class", []) if c not in ("thalabus-edit", "thalabus-delete")]
                if css_classes:
                    element["class"] = css_classes
                else:
                    del element["class"]
        elif is_new_wrapper_span:
            # Newly added span wrapper (case 3) — unwrap it
            if not accept:
                previous_content = element.attrs.pop("previous-content", None)
                if previous_content:
                    element.string = previous_content
            element.unwrap()
        else:
            # Existing element with thalabus-edit class — just clean up the class and rollback attrs
            element["class"].remove("thalabus-edit")
            if not element["class"]:
                del element["class"]
            if accept:
                # Remove all previous-* rollback attrs
                for attr_key in [k for k in list((element.attrs or {}).keys()) if k.startswith("previous-")]:
                    element.attrs.pop(attr_key, None)
            else:
                previous_content = (element.attrs or {}).pop("previous-content", None)
                if previous_content:
                    element.string = previous_content
                # Restore any previous-{attr} values (e.g. previous-style → style)
                for attr_key in [k for k in list((element.attrs or {}).keys()) if k.startswith("previous-")]:
                    original_attr = attr_key[len("previous-"):]
                    element[original_attr] = (element.attrs or {}).pop(attr_key)
    return str(soup)

def compare_html(html1: str, html2: str) -> bool:
    soup1 = BeautifulSoup(html1, "html.parser")
    soup2 = BeautifulSoup(html2, "html.parser")
    return soup1.prettify() == soup2.prettify()

def get_chapter_headers(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    headers = []
    for header in soup.find_all(re.compile(r"h[1-6]")):
        header_id = header.get('id')
        if header_id:
            headers.append(header_id)
    return headers

def get_element_by_id(html:str, element_id:str) -> dict:
    """
    Get the element with the given id from the HTML.
    :param html: HTML
    :param element_id: ID of the element to find
    :return: Dictionary with the element's tag name and attributes, or None if not found
    """
    soup = BeautifulSoup(html, "html.parser")
    element = soup.find(id=element_id)
    if element:
        return {
            "tag": element.name,
            "attributes": dict(element.attrs),
            "text": element.get_text(strip=True)
        }
    return None

def get_element_ids_by_xpath(html_content: str, xpath: str, multiple: bool = False) -> str | List[str]:
    """
    Get the ID(s) of the element(s) that match the given XPath expression.
    :param html_content: HTML content
    :param xpath: XPath expression to find the element(s)
    :param multiple: If True, return a list of all matching element IDs. If False, return only the first match.
    :return: ID of the element (or list of IDs if multiple=True), or None/empty list if not found
    """
    try:
        # Parse HTML with lxml for XPath support
        tree = html.fromstring(html_content)
        
        # Find elements using XPath
        elements = tree.xpath(xpath)
        
        if elements:
            if multiple:
                # Return list of all IDs
                ids = []
                for element in elements:
                    element_id = element.get('id')
                    if element_id:
                        ids.append(element_id)
                return ids
            else:
                # Return first ID
                element_id = elements[0].get('id')
                return element_id
        
        return [] if multiple else None
        
    except Exception as e:
        # Fallback to BeautifulSoup CSS selector if lxml fails or XPath is invalid
        # Convert simple XPath to CSS selector for basic cases
        try:
            css_selector = xpath
            # Convert basic XPath patterns to CSS selectors
            if xpath.startswith('//'):
                css_selector = xpath[2:]  # Remove leading //
            if '[' in css_selector and ']' in css_selector:
                # Handle basic attribute selectors
                css_selector = css_selector.replace('[', ':contains("').replace(']', '")')
            
            soup = BeautifulSoup(html_content, "html.parser")
            elements = soup.select(css_selector)
            
            if elements:
                if multiple:
                    ids = []
                    for element in elements:
                        element_id = element.get('id')
                        if element_id:
                            ids.append(element_id)
                    return ids
                else:
                    return elements[0].get('id')
        except:
            pass
        
        return [] if multiple else None

def sanitize_xml_for_parsing(xml_content: str) -> str:
    """
    Prepare a string for parsing with lxml.etree.fromstring by fixing a couple
    of common XML pitfalls in user content:
    - normalize ampersands so that lone '&' become '&amp;'
    - escape any literal CDATA end sequence ']]>' ONLY when it appears outside CDATA sections.

    This function is safe to call repeatedly and is intended to be used right
    before calling `etree.fromstring` on user-provided XML/HTML fragments.
    """
    # Remove illegal XML control characters (these commonly break XML parsers).
    # Keep allowed whitespace characters: tab (\x09), LF (\x0A), CR (\x0D).
    # XML 1.0 disallows: 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, 0x7F.
    illegal_xml_chars_re = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
    xml_content = illegal_xml_chars_re.sub("", xml_content)

    # normalize existing escaped ampersands then escape raw ampersands
    xml_content = xml_content.replace("&amp;", "&")
    xml_content = xml_content.replace("&", "&amp;")

    # Handle literal ']]>' sequences more carefully - only escape when they appear
    # outside of CDATA sections. Inside CDATA sections, ']]>' should terminate the CDATA.
    if ']]>' in xml_content:
        # Simple approach: only escape ']]>' that is NOT immediately followed by '</something>'
        # This catches cases where ']]>' appears in content but not as CDATA terminators
        # Replace ']]>' with ']]&gt;' only when NOT followed by a closing tag or end of CDATA
        # Keep CDATA terminators intact: ']]></element>' should stay as-is
        xml_content = re.sub(r'\]\]>(?![<\s])', ']]&gt;', xml_content)

    return xml_content

def get_elements_by_xpath(html_content: str, xpath: str, multiple: bool = False) -> dict | List[dict]:
    """
    Get detailed information about element(s) that match the given XPath expression.
    :param html_content: HTML content
    :param xpath: XPath expression to find the element(s)
    :param multiple: If True, return a list of all matching elements. If False, return only the first match.
    :return: Dictionary with element details (or list of dictionaries if multiple=True), or None/empty list if not found
    
    Each dictionary contains:
    - tag: HTML tag name
    - id: Element ID (if available)
    - attributes: All element attributes as a dictionary
    - text: Trimmed text content (including text from child elements)
    - subelements: List of direct child elements with same structure (except subelements)
    """
    def _element_to_dict(element) -> dict:
        """Convert an lxml element to a dictionary with essential details. Skips non-element children (e.g., comments)."""
        result = {
            "tag": element.tag,
            "id": element.get('id'),
            "attributes": dict(element.attrib),
            "text": element.text_content().strip() if hasattr(element, 'text_content') else (element.text or "").strip(),
            "subelements": []
        }

        # Process child elements recursively, skip non-element nodes (e.g., comments, processing instructions)
        for child in element:
            # Only process if child is an lxml Element (not Comment, etc.)
            if not hasattr(child, 'tag') or not isinstance(child.tag, str):
                continue
            child_dict = {
                "tag": child.tag,
                "id": child.get('id'),
                "attributes": dict(child.attrib),
                "text": child.text_content().strip() if hasattr(child, 'text_content') else (child.text or "").strip()
            }
            result["subelements"].append(child_dict)

        return result
    
    try:
        # Fix eventually invalid xpath by adding <root> tag if necessary
        if not html_content.startswith("<root>"):
            html_content = f"<root>{html_content}</root>"

        # Prepare content for parsing
        html_content = sanitize_xml_for_parsing(html_content)

        # Parse as XML to preserve case sensitivity instead of HTML
        tree = etree.fromstring(html_content.encode('utf-8'))

        # Find elements using XPath
        elements = tree.xpath(xpath)

        if elements:
            if multiple:
                # Return list of all element dictionaries
                return [_element_to_dict(element) for element in elements]
            else:
                # Return first element dictionary
                return _element_to_dict(elements[0])

        return [] if multiple else None

    except Exception as e:
        # Fallback to BeautifulSoup if lxml fails
        try:
            # Convert simple XPath to CSS selector for basic cases
            css_selector = xpath
            if xpath.startswith('//'):
                css_selector = xpath[2:]  # Remove leading //
            if '[' in css_selector and ']' in css_selector:
                # Handle basic attribute selectors
                css_selector = css_selector.replace('[', ':contains("').replace(']', '")')
            
            soup = BeautifulSoup(html_content, "html.parser")
            elements = soup.select(css_selector)
            
            def _bs_element_to_dict(element) -> dict:
                """Convert a BeautifulSoup element to a dictionary."""
                result = {
                    "tag": element.name,
                    "id": element.get('id'),
                    "attributes": dict(element.attrs),
                    "text": element.get_text(strip=True),
                    "subelements": []
                }
                
                # Process direct children
                for child in element.children:
                    if hasattr(child, 'name') and child.name:  # Skip text nodes
                        child_dict = {
                            "tag": child.name,
                            "id": child.get('id'),
                            "attributes": dict(child.attrs),
                            "text": child.get_text(strip=True)
                        }
                        result["subelements"].append(child_dict)
                
                return result
            
            if elements:
                if multiple:
                    return [_bs_element_to_dict(element) for element in elements]
                else:
                    return _bs_element_to_dict(elements[0])
        except:
            pass
        
        return [] if multiple else None

def normalize_hyphens(input_str: str) -> str:
    """
    Replace common Unicode hyphen/dash characters with ASCII '-'.
    """
    if not input_str:
        return input_str
    if not isinstance(input_str, str):
        return input_str

    normalised_string: str = (
        input_str
        .replace('\u2010', '-')  # hyphen
        .replace('\u2011', '-')  # non-breaking hyphen
        .replace('\u2012', '-')  # figure dash
        .replace('\u2013', '-')  # en dash
        .replace('\u2014', '-')  # em dash
        .replace('\u2015', '-')  # horizontal bar
        .replace('\u2212', '-')  # minus sign
    )
    return normalised_string

def is_valid_regex(pattern: str) -> bool:
    """
    Check if a regex pattern is valid.
    """
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False

def hash_string(input_string: str) -> str:
    """
    Generate SHA-256 hash of input string.
    """
    return hashlib.sha256(input_string.encode()).hexdigest()

def clean_unicode_text(text: str) -> str:
    """Clean unicode text by replacing problematic characters with ASCII equivalents.

    This is used by CLI utilities (e.g. `doc save`) to export more portable text.

    Strategy:
    - NFD normalize and remove combining marks (diacritics)
    - Replace common punctuation variants with ASCII equivalents
    - Drop zero-width and other problematic characters
    - Keep ASCII; keep non-ASCII letters as a last resort
    """

    if not text:
        return text

    nfd_text = unicodedata.normalize("NFD", text)
    cleaned = "".join(c for c in nfd_text if unicodedata.category(c) != "Mn")

    replacements = {
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus sign
        "\u00ad": "",  # soft hyphen
        "\u202f": " ",  # narrow no-break space
        "\u00a0": " ",  # nbsp
        "\ufeff": "",  # zero-width no-break space
        "\u200b": "",  # zero-width space
        "\u200c": "",  # zero-width non-joiner
        "\u200d": "",  # zero-width joiner
        "\u2026": "...",  # ellipsis
        "\u2022": "-",  # bullet
    }

    for unicode_char, ascii_char in replacements.items():
        cleaned = cleaned.replace(unicode_char, ascii_char)

    result_chars: list[str] = []
    for c in cleaned:
        if ord(c) < 128:
            result_chars.append(c)
            continue

        # Keep non-ASCII letters (rare after diacritic removal)
        if unicodedata.category(c).startswith("L"):
            result_chars.append(c)
            continue

        # Replace other non-ASCII with a space to avoid gluing words
        if c.isspace():
            result_chars.append(" ")

    # Collapse repeated spaces created by replacements
    result = "".join(result_chars)
    result = " ".join(result.split())
    return result


def clean_fragment_html(html: str) -> str:
    """Remove layout attributes from HTML fragments (page, top, bottom).
    
    This utility function cleans up RAG fragment HTML by removing attributes that
    are not needed for reading and consume unnecessary tokens. The attributes
    'page', 'top', and 'bottom' are removed from all HTML elements. Element IDs
    are preserved because they are required for precise source citations.
    
    Additionally, newlines within text content are replaced with spaces to ensure
    clean, readable output for knowledge graph entities and other serialized data.
    
    Example:
        Input:  <p id="text-1" page="1" top="0" bottom="0">Bob the Cat</p>
        Output: <p id="text-1">Bob the Cat</p>
        
        Input:  <p>Bob is lives in\nWhiskerville (Place).</p>
        Output: <p>Bob is lives in Whiskerville (Place).</p>
    
    :param html: HTML string with fragment attributes
    :return: HTML string with unnecessary attributes removed and newlines normalized
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove layout metadata while preserving element IDs for source citations.
        attributes_to_remove = {"page", "top", "bottom"}
        
        for tag in soup.find_all(True):
            # Remove unnecessary attributes
            for attr in attributes_to_remove:
                if attr in tag.attrs:
                    del tag.attrs[attr]
            
            # Replace newlines in text content with spaces
            if hasattr(tag, 'string') and tag.string:
                tag.string.replace_with(tag.string.replace('\n', ' '))
            for string in tag.find_all(string=True):
                if hasattr(string, 'replace_with'):
                    string.replace_with(string.replace('\n', ' '))
        
        result = str(soup)
        # Remove extra spaces that BeautifulSoup adds after opening tags (e.g., "<p>  text" -> "<p>text")
        result = re.sub(r'(>\s+)(?=[^ ])', '>', result)
        return result
    except Exception:
        # On any error, return the input unchanged
        return html


def format_relationship_text(rel_label: str) -> tuple:
    """Format a relationship label for natural language output.
    
    Converts snake_case relationship labels to lowercase with spaces,
    and determines whether to prepend 'is' based on the relationship type.
    
    State-of-being relationships (starting with 'is_') get 'is' prepended.
    Action verbs (like 'lives_in') do not get 'is' prepended.
    
    Args:
        rel_label: The relationship label in snake_case (e.g., 'lives_in', 'is_friend_of')
    
    Returns:
        tuple: (rel_text, prepend_is) where:
            - rel_text: The formatted relationship text in lowercase with spaces
            - prepend_is: Boolean indicating if 'is' should be prepended
    
    Examples:
        format_relationship_text('lives_in') -> ('lives in', False)
            Results in: "Bob lives in Whiskerville"
        format_relationship_text('is_friend_of') -> ('is friend of', True)
            Results in: "Bob is friend of Jane" (after conditional "is" prepending)
    """
    rel_text = rel_label.lower().replace("_", " ")
    prepend_is = rel_label.lower().startswith("is_")
    return rel_text, prepend_is


_LINKED_DOCUMENT_BLOCK_RE = re.compile(
    r'<LINKED_DOCUMENT id="([^"]*)"[^>]*>\n```\n.*?\n```\n</LINKED_DOCUMENT>\n?',
    re.DOTALL,
)


def strip_linked_document_blocks(text: str, excluded_document_ids: set) -> str:
    """Remove <LINKED_DOCUMENT id="..."> blocks whose document id is in excluded_document_ids.

    Used when an entity's discourse text was serialized before the render site knew that the
    linked source document (or one of its fragments) is already present elsewhere in the same
    RAG message; the duplicated block is cut out instead of being re-serialized.

    Args:
        text: Serialized entity discourse text, possibly containing <LINKED_DOCUMENT> blocks.
        excluded_document_ids: Set of Document ids whose blocks should be removed.

    Returns:
        The input text with matching <LINKED_DOCUMENT> blocks removed, unchanged when the
        exclusion set is empty or no block matches.
    """
    from html import unescape

    if not text or not excluded_document_ids:
        return text

    excluded = set(excluded_document_ids)

    def _remove_excluded(match: "re.Match[str]") -> str:
        doc_id = unescape(match.group(1))
        return "" if doc_id in excluded else match.group(0)

    return _LINKED_DOCUMENT_BLOCK_RE.sub(_remove_excluded, text)


#region unit tests
if __name__ == "__main__":
    text = """
# Title: Hello World

This is a test to convert Markdown into HTML.

## Chapter 1: Introduction

Here is some text divided into multiple lines,  
where some part is in one line and another  
part is on the next.

### Subsection: Features

- Feature 1: Easy to use
- Feature 2: Lightweight
- Feature 3: Supports **Markdown** and _HTML_

### Subsection: Table Example

| Feature       | Description           |
|---------------|-----------------------|
| Easy to use   | Simple and intuitive  |
| Lightweight   | Minimal dependencies  |
| Markdown/HTML | Flexible input/output |

## Chapter 2: Conclusion

This is a new paragraph.

Another paragraph with a [link](https://example.com) and **bold text**.

- Nested List Example:
  - Item 1
    - Sub-item 1.1
    - Sub-item 1.2
  - Item 2
    - Sub-item 2.1
    - Sub-item 2.2
"""
    expected_html = """
<h1>
 Title: Hello World
</h1>
<p>
 This is a test to convert Markdown into HTML.
</p>
<h2>
 Chapter 1: Introduction
</h2>
<p>
 Here is some text divided into multiple lines,
 <br/>
 where some part is in one line and another
 <br/>
 part is on the next.
</p>
<h3>
 Subsection: Features
</h3>
<ul>
 <li>
  Feature 1: Easy to use
 </li>
 <li>
  Feature 2: Lightweight
 </li>
 <li>
  Feature 3: Supports
  <strong>
   Markdown
  </strong>
  and
  <em>
   HTML
  </em>
 </li>
</ul>
<h3>
 Subsection: Table Example
</h3>
<table>
 <thead>
  <tr>
   <th>
    Feature
   </th>
   <th>
    Description
   </th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>
    Easy to use
   </td>
   <td>
    Simple and intuitive
   </td>
  </tr>
  <tr>
   <td>
    Lightweight
   </td>
   <td>
    Minimal dependencies
   </td>
  </tr>
  <tr>
   <td>
    Markdown/HTML
   </td>
   <td>
    Flexible input/output
   </td>
  </tr>
 </tbody>
</table>
<h2>
 Chapter 2: Conclusion
</h2>
<p>
 This is a new paragraph.
</p>
<p>
 Another paragraph with a
 <a href="https://example.com">
  link
 </a>
 and
 <strong>
  bold text
 </strong>
 .
</p>
<ul>
 <li>
  Nested List Example:
 </li>
 <li>
  Item 1
  <ul>
   <li>
    Sub-item 1.1
   </li>
   <li>
    Sub-item 1.2
   </li>
  </ul>
 </li>
 <li>
  Item 2
  <ul>
   <li>
    Sub-item 2.1
   </li>
   <li>
    Sub-item 2.2
   </li>
  </ul>
 </li>
</ul>
"""
    print(f"```text\n{text}\n```")
    html = generate_html_from_text(text)
    print(f"```html\n{html}\n```")
    assert compare_html(html, expected_html), "HTML does not match!"
    
    text = "Hello World\n\nThis is a test\n\nHere is <b>bold</b>, <b>very bold</b> text"
    print(f"```text\n{text}\n```")
    html = generate_html_from_text(text)
    print(f"```html\n{html}\n```")
    print()
    expected_output = '<p>Hello World</p>\n<p>This is a test</p>\n<p>Here is <b>bold</b>, <b>very bold</b> text</p>\n'
    assert compare_html(html, expected_output), "HTML does not match!"
    
    text = "Here is <b>bold,</b>\n <b>very bold</b> text"
    print(f"```text\n{text}\n```")
    html = generate_html_from_text(text)
    print(f"```html\n{html}\n```")
    print()
    expected_output = '<p>Here is <b>bold,</b><br/> <b>very bold</b> text</p>\n'
    assert compare_html(html, expected_output), "HTML does not match!"
    
    text = "Here is <b>bold,</b>\n <b>very bold</b> text\n<!-- This is a comment -->\nAnother line"
    print(f"```text\n{text}\n```")
    html = generate_html_from_text(text)
    print(f"```html\n{html}\n```")
    print()
    expected_output = '<p>Here is <b>bold,</b><br/> <b>very bold</b> text</p><!-- This is a comment --><p>Another line</p>\n'
    assert compare_html(html, expected_output), "HTML does not match!"

    html2 = add_html_tag_ids(html)
    print(f"```html\n{html2}\n```")
    assert html != html2

    text3 = strip_ids_from_html(html2)
    print(f"```html\n{text3}\n```")

    enhanced_text = '<p>first paragraph</p> <p id="test-id">Hello xxx World</p> <p>last paragraph</p>'
    element_id = 'test-id'
    text = 'xxx'
    new_tag = 'span'
    new_element_id = 'span-test-id'
    expected_output = '<p>first paragraph</p> <p id="test-id">Hello <span class="thalabus-edit">xxx</span> World</p> <p>last paragraph</p>'
    result = replace_text_in_element(enhanced_text, element_id, text, text, new_tag)
    print(f"```html\n{result}\n```")
    assert result == expected_output

    enhanced_text = '<p id="1">first paragraph</p> <p id="x">Hello World</p> <p id="2">last paragraph</p>'
    element_id1 = '1'
    element_id2 = '2'
    expected_output = '<p id="1">first paragraph last paragraph</p> <p id="x">Hello World</p>'
    result = merge_2_tags(enhanced_text, element_id1, element_id2, ' ', consolidate_hypenated_words=False)
    print(f"```html\n{result}\n```")
    assert result == expected_output

    result = merge_tags(enhanced_text, ['1', 'x', '2'], ' ')
    expected_output = '<p id="1">first paragraph Hello World last paragraph</p>'
    print(f"```html\n{result}\n```")
    assert result == expected_output

    enhanced_text = '<p id="1">first paragra-</p> <p id="2">ph last paragraph</p>'
    result = merge_tags(enhanced_text, ['1', '2'], ' ', consolidate_hypenated_words=True)
    expected_output = '<p id="1">first paragraph last paragraph</p>'
    print(f"```html\n{result}\n```")
    assert result == expected_output

    # Test append_element_after
    enhanced_text = '<p id="1">first paragraph</p> <p id="2">second paragraph</p>'
    element_ids = ['1']
    text = 'appended text'
    new_tag = 'h1'
    expected_output = '<p id="1">first paragraph</p><h1 class="thalabus-edit" id="1-h-1">appended text</h1> <p id="2">second paragraph</p>'
    result = append_element_after(enhanced_text, element_ids, text, new_tag)
    print(f"```html\n{result}\n```")
    assert result == expected_output

    # Test accept_changes_util
    enhanced_text = '<p id="1" class="thalabus-edit" previous-content="first paragraph">modified paragraph</p> <p id="2">second paragraph</p>'
    expected_output = '<p id="1">modified paragraph</p> <p id="2">second paragraph</p>'
    result = accept_changes_util(enhanced_text, accept=True)
    print(f"```html\n{result}\n```")
    assert result == expected_output

    # Test reject_changes_util
    enhanced_text = '<p id="1" class="thalabus-edit" previous-content="first paragraph">modified paragraph</p> <p id="2">second paragraph</p>'
    expected_output = '<p id="1">first paragraph</p> <p id="2">second paragraph</p>'
    result = accept_changes_util(enhanced_text, accept=False)
    print(f"```html\n{result}\n```")
    assert result == expected_output

    # Test get_elements_by_xpath
    test_html = '''
    <div id="container">
        <h1 id="title" class="header">Main Title</h1>
        <p id="intro" data-type="introduction">Introduction paragraph with <strong>bold text</strong>.</p>
        <ul id="list">
            <li id="item1">First item</li>
            <li id="item2">Second item</li>
        </ul>
    </div>
    '''
    
    # Test single element
    result = get_elements_by_xpath(test_html, '//h1[@id="title"]')
    print(f"Single element result: {result}")
    assert result is not None
    assert result['tag'] == 'h1'
    assert result['id'] == 'title'
    assert 'header' in result['attributes']['class']
    
    # Test multiple elements
    results = get_elements_by_xpath(test_html, '//li', multiple=True)
    print(f"Multiple elements result: {results}")
    assert len(results) == 2
    assert results[0]['id'] == 'item1'
    assert results[1]['id'] == 'item2'
    
    # Test element with subelements
    result = get_elements_by_xpath(test_html, '//p[@id="intro"]')
    print(f"Element with subelements: {result}")
    assert result is not None
    assert len(result['subelements']) == 1  # The <strong> tag
    assert result['subelements'][0]['tag'] == 'strong'
    assert 'tail' not in result  # Ensure redundant fields are removed
    assert 'content' not in result
    assert 'inner_text' not in result
    assert result['text'] == "Introduction paragraph with bold text."  # Trimmed text
    
    # Test replace_text_globally (Case 2: style attribute replacement)
    html_style_attr = (
        '<p id="p-1">A heartwarming tale featuring '
        '<span style="color: darkgreen; font-weight: bold;">Bob</span>'
        ' (the mischievous orange cat), '
        '<span style="color: darkgreen; font-weight: bold;">Jane</span>'
        ' (his human companion).</p>'
    )
    result = replace_text_globally(html_style_attr, "color: darkgreen;", "color: darkblue;")
    assert 'color: darkblue;' in result, "FAIL replace_text_globally case2: replacement not found"
    assert 'thalabus-edit' in result, "FAIL replace_text_globally case2: thalabus-edit class missing"
    assert 'previous-style=' in result, "FAIL replace_text_globally case2: previous-style rollback attr missing"
    # Verify class is on the existing span element (style updated), NOT a new wrapper span
    from bs4 import BeautifulSoup as _BS_test
    _s_test = _BS_test(result, "html.parser")
    for edited in _s_test.find_all(class_="thalabus-edit"):
        assert edited.get("style") is not None, "FAIL replace_text_globally case2: thalabus-edit element has no style attr"
        assert "color: darkblue;" in edited["style"], "FAIL replace_text_globally case2: style attr not updated"
    
    # Case COMMIT on style-attr edit: class removed, previous-style removed, style stays updated
    committed_style = accept_changes_util(result, accept=True)
    assert "color: darkblue;" in committed_style, "FAIL replace_text_globally case2 commit: new style gone"
    assert "thalabus-edit" not in committed_style, "FAIL replace_text_globally case2 commit: class not removed"
    assert "previous-style" not in committed_style, "FAIL replace_text_globally case2 commit: previous-style not cleaned"
    
    # Case REJECT on style-attr edit: style restored to original, class removed
    rejected_style = accept_changes_util(result, accept=False)
    assert "color: darkgreen;" in rejected_style, "FAIL replace_text_globally case2 reject: original style not restored"
    assert "color: darkblue;" not in rejected_style, "FAIL replace_text_globally case2 reject: new style still present"
    assert "thalabus-edit" not in rejected_style, "FAIL replace_text_globally case2 reject: class not removed"
    assert "previous-style" not in rejected_style, "FAIL replace_text_globally case2 reject: previous-style attr not removed"
    
    # Test replace_text_globally (Case 3: text node replacement)
    html_text_node = '<p id="p-2">The date is January 2023 and again January 2023 here.</p>'
    result3 = replace_text_globally(html_text_node, "January 2023", "March 2024")
    assert result3.count('<span class="thalabus-edit">March 2024</span>') == 2, "FAIL replace_text_globally case3: expected 2 spans"
    assert "January 2023" not in result3, "FAIL replace_text_globally case3: original still present"
    
    # Test replace_text_globally (Case 1: leaf element)
    html_leaf = '<p id="p-3"><span>darkgreen</span></p>'
    result1 = replace_text_globally(html_leaf, "darkgreen", "darkblue")
    assert 'class="thalabus-edit"' in result1, "FAIL replace_text_globally case1: thalabus-edit class missing"
    assert "darkblue" in result1, "FAIL replace_text_globally case1: replacement not found"
    assert '<span class="thalabus-edit">' not in result1 or result1.count("thalabus-edit") == 1, "FAIL replace_text_globally case1: unexpected extra span"
    
    # Test accept_changes_util with text node spans (COMMIT)
    committed = accept_changes_util(result3, accept=True)
    assert "March 2024" in committed, "FAIL accept_changes_util COMMIT: replacement text gone"
    assert "thalabus-edit" not in committed, "FAIL accept_changes_util COMMIT: class not removed"
    assert "<span" not in committed, "FAIL accept_changes_util COMMIT: span still present"
    
    # Test accept_changes_util with text node spans (REJECT with no previous-content)
    rejected_no_prev = accept_changes_util(result3, accept=False)
    assert "thalabus-edit" not in rejected_no_prev, "FAIL accept_changes_util REJECT: class not removed"
    
    print("All tests passed!")
#endregion
