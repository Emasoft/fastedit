"""Split a file into per-chunk parts, and join those parts back into one.

Format-aware where the format allows a clean, lossless cut (a JSON array's
elements, a Markdown file's headings, CSV/TSV rows); a plain line-count
chunker otherwise. Every split writes a manifest.json into the output
directory so `join` can put the pieces back together without re-parsing
or re-serializing content -- reconstruction is byte-for-byte for every
mode except XML/HTML element splitting, which is lossy in general (see
below) and therefore read-only: split only, no join.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MANIFEST_NAME = "manifest.json"

_MARKDOWN_EXTS = {".md", ".markdown", ".mdx"}
_CSV_EXTS = {".csv"}
_TSV_EXTS = {".tsv"}
_JSON_EXTS = {".json"}
_JSONL_EXTS = {".jsonl", ".ndjson"}
_XML_EXTS = {".xml"}
_HTML_EXTS = {".html", ".htm"}


class SplitJoinError(ValueError):
    """A requested mode doesn't fit the file (wrong --by, --rows on non-csv, ...)."""


def detect_format(path: Path) -> str:
    """Guess a split/join format from the file extension, defaulting to plain text."""
    ext = path.suffix.lower()
    if ext in _JSON_EXTS:
        return "json"
    if ext in _JSONL_EXTS:
        return "jsonl"
    if ext in _XML_EXTS:
        return "xml"
    if ext in _HTML_EXTS:
        return "html"
    if ext in _MARKDOWN_EXTS:
        return "markdown"
    if ext in _CSV_EXTS:
        return "csv"
    if ext in _TSV_EXTS:
        return "tsv"
    return "text"


# --- line-count chunking (works on any text file) ---


def split_by_lines(text: str, n: int) -> list[str]:
    """Chunk raw text into groups of exactly n lines each (last group may be short)."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return [""]
    return ["".join(lines[i : i + n]) for i in range(0, len(lines), n)]


# --- CSV/TSV row chunking ---


def split_csv_rows(text: str, n: int) -> list[str]:
    """Chunk CSV/TSV data rows into groups of n, repeating the header row in each.

    Naive and line-based: a data row is one physical line. A field value
    containing an embedded newline (legal, quoted, RFC 4180 CSV) would be
    split incorrectly -- a real CSV parser would be needed to handle that,
    out of scope for this pass.
    """
    lines = text.splitlines(keepends=True)
    if not lines:
        return [""]
    header, rows = lines[0], lines[1:]
    if not rows:
        return [header]
    return ["".join([header, *rows[i : i + n]]) for i in range(0, len(rows), n)]


def join_csv_chunks(chunks: list[bytes]) -> bytes:
    """Undo split_csv_rows: keep the first chunk whole, strip the repeated header from the rest."""
    if not chunks:
        return b""
    out = [chunks[0]]
    header = chunks[0].splitlines(keepends=True)[0] if chunks[0] else b""
    for chunk in chunks[1:]:
        lines = chunk.splitlines(keepends=True)
        if lines and lines[0] == header:
            lines = lines[1:]
        out.append(b"".join(lines))
    return b"".join(out)


# --- Markdown heading chunking ---


def split_markdown_headings(text: str, level: int) -> list[str]:
    """Cut markdown text right before each heading of the given level (## for level 2).

    Each chunk is a contiguous, unmodified slice of the original text, so
    concatenating the chunks in order reproduces the file exactly.
    """
    import itertools

    marker = "#" * level + " "
    lines = text.splitlines(keepends=True)
    boundaries = [i for i, line in enumerate(lines) if line.startswith(marker)]
    if not boundaries:
        raise SplitJoinError(f"no level-{level} headings ('{marker.strip()} ...') found")
    if boundaries[0] != 0:
        boundaries = [0, *boundaries]
    boundaries.append(len(lines))
    chunks = []
    for start, end in itertools.pairwise(boundaries):
        if start == end:
            continue
        chunks.append("".join(lines[start:end]))
    return chunks


# --- JSON array element splitting ---


def split_json_array(text: str) -> tuple[str, list[str], list[str], str]:
    """Split a top-level JSON array into (prefix, elements, separators, suffix).

    Elements and separators are exact substrings of *text* -- concatenating
    prefix + elements[0] + separators[0] + elements[1] + ... + elements[-1]
    + suffix reproduces *text* byte-for-byte, regardless of the source's
    original indentation or spacing.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise SplitJoinError(f"not valid JSON: {e}") from e
    if not isinstance(parsed, list):
        kind = "object" if isinstance(parsed, dict) else type(parsed).__name__
        raise SplitJoinError(f"--by element needs a JSON array at the top level, got a JSON {kind}")

    i = 0
    while text[i] in " \t\r\n":
        i += 1
    # json.loads already proved the top-level value is an array, so this is its '['.
    pos = i + 1
    while pos < len(text) and text[pos] in " \t\r\n":
        pos += 1
    if not parsed:
        return text[: pos + 1] if text[pos] == "]" else text, [], [], ""

    decoder = json.JSONDecoder()
    elements: list[str] = []
    separators: list[str] = []
    prefix_end = pos
    while True:
        _, end = decoder.raw_decode(text, pos)
        elements.append(text[pos:end])
        k = end
        while k < len(text) and text[k] in " \t\r\n":
            k += 1
        if k < len(text) and text[k] == ",":
            m = k + 1
            while m < len(text) and text[m] in " \t\r\n":
                m += 1
            separators.append(text[end:m])
            pos = m
            continue
        if k < len(text) and text[k] == "]":
            return text[:prefix_end], elements, separators, text[end:]
        raise SplitJoinError("malformed JSON array (expected ',' or ']')")


def join_json_array(prefix: str, elements: list[bytes], separators: list[str], suffix: str) -> bytes:
    """Undo split_json_array."""
    body = []
    for idx, element in enumerate(elements):
        body.append(element)
        if idx < len(separators):
            body.append(separators[idx].encode("utf-8"))
    return prefix.encode("utf-8") + b"".join(body) + suffix.encode("utf-8")


# --- XML/HTML top-level-child splitting (read-only: split, never join) ---

_TAG_RE = re.compile(
    r"<!--.*?-->"  # comments
    r"|<!\[CDATA\[.*?\]\]>"  # CDATA
    r"|<\?.*?\?>"  # processing instructions / xml declaration
    r"|<!DOCTYPE[^>]*>"  # doctype
    r"|<(?P<close>/?)(?P<name>[a-zA-Z][a-zA-Z0-9:_.-]*)(?P<attrs>[^>]*?)(?P<selfclose>/?)>",
    re.DOTALL,
)

_VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


def split_markup_top_level_children(text: str) -> list[str]:
    """Best-effort split of an XML/HTML document into its root element's direct children.

    This is a naive tag-depth scanner, not a real (X)HTML parser: it does
    not understand script/style content with embedded angle brackets, or
    exotic markup. It is used only for `split`'s read-only element mode --
    there is deliberately no matching join, because a plucked-out fragment
    loses ancestor context (namespace declarations, xml:base, inherited
    attributes) that a real round-trip would need. Good enough for typical
    hand-written or generated markup; anything exotic should use --lines.
    """
    root_start = None
    root_name = None
    depth = 0
    children: list[str] = []
    child_start = None
    for m in _TAG_RE.finditer(text):
        tag = m.group(0)
        if tag.startswith(("<!--", "<![CDATA[", "<?", "<!DOCTYPE")):
            continue
        name = m.group("name")
        is_close = bool(m.group("close"))
        is_self_closing = bool(m.group("selfclose")) or name.lower() in _VOID_ELEMENTS

        if root_start is None:
            if is_close:
                continue
            root_start = m.start()
            root_name = name
            depth = 1
            if is_self_closing:
                return []  # a self-closing root has no children to split out
            continue

        if depth == 1 and not is_close:
            child_start = m.start()

        if is_close:
            depth -= 1
            if depth == 1 and child_start is not None:
                children.append(text[child_start : m.end()])
                child_start = None
            if depth == 0 and name == root_name:
                break
        elif not is_self_closing:
            depth += 1
        elif depth == 1 and child_start is not None:
            children.append(text[child_start : m.end()])
            child_start = None

    return children
