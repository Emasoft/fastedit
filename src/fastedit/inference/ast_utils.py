"""AST data types and retrieval via tldr structure/extract.

Provides the ASTNode dataclass and functions to get AST maps from files
using tldr's structure and extract commands.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ASTNode:
    """A function, method, or class from the AST map."""
    name: str
    kind: str  # "function", "method", "class", "interface", etc.
    line_start: int
    line_end: int
    signature: str
    parent: str | None = None  # parent class name (from tldr extract)


@dataclass
class ChunkRegion:
    """A region of the file to extract and merge."""
    start_line: int  # 1-indexed, inclusive
    end_line: int     # 1-indexed, inclusive
    matched_nodes: list[str]  # names of AST nodes in this region


@dataclass
class ChunkedMergeResult:
    """Result of a chunked merge operation."""
    merged_code: str
    parse_valid: bool
    chunks_used: int
    chunk_regions: list[tuple[int, int]]  # (start, end) of each chunk
    model_tokens: int
    latency_ms: float
    chunks_rejected: int = 0  # chunks rejected due to hallucination
    # Step A2: validation-retry attempts consumed by the unified
    # retry-until-valid loop (merge attempts beyond the first per merge
    # site — whole-file counts as one site, each chunk as one). 0 for
    # deterministic paths and first-attempt successes.
    retries: int = 0


@dataclass
class BatchEdit:
    """A single edit operation in a batch. Extensible — add any number to a list."""
    snippet: str
    after: str | None = None
    replace: str | None = None
    preserve_siblings: bool = False
    """When True with `replace=ClassName`: carry over any named sibling
    members (methods, nested classes) that exist in the original class
    but aren't mentioned in the snippet. Lets you edit a subset of a
    class's members without enumerating the rest. No-op without `replace`."""


@dataclass
class DeleteResult:
    """Result of a deterministic symbol deletion."""
    merged_code: str
    parse_valid: bool
    deleted_symbol: str
    deleted_kind: str  # "function", "method", "class", etc.
    deleted_lines: tuple[int, int]  # (start, end) that were removed
    lines_removed: int


@dataclass
class MoveResult:
    """Result of a deterministic symbol move."""
    merged_code: str
    parse_valid: bool
    moved_symbol: str
    moved_kind: str
    from_lines: tuple[int, int]  # original position
    after_symbol: str
    new_lines: tuple[int, int]   # new position after move


def get_ast_map_from_source(
    source_code: str,
    file_path: str,
    language: str | None = None,
) -> list[ASTNode]:
    """In-memory AST map using tree-sitter directly. No disk read, no daemon.

    Unlike :func:`get_ast_map`, which shells out to ``tldr structure`` and
    is subject to the daemon's salsa-cache invalidation race (stale line
    numbers after a recent write), this function parses ``source_code`` in
    process via tree-sitter — the authoritative view for any edit
    pipeline that already has the source text in hand.

    The returned list mirrors what ``tldr structure`` produces: top-level
    functions, classes (with methods nested as separate nodes whose
    ``parent`` points to the enclosing class), and top-level constants
    (Rust ``const``, TS/JS ``const``/``let``, etc — whatever a
    ``replace=`` or ``after=`` edit might legitimately target).

    B3: data/config/markup formats get symbols too, from the declarative
    per-format spec table ``_FORMAT_SYMBOL_SPECS`` (html elements with an
    id, XML elements, markdown sections, JSON/YAML keys, CSS rule sets,
    TOML tables, SQL CREATE objects, Dockerfile stages, bash functions).
    The per-format semantics are documented on that table.

    Args:
        source_code: Full source text to parse. Takes precedence over disk.
        file_path: Used only for extension-based language detection.
        language: B3 explicit language hint, consulted when the file's
            suffix does not resolve. The all-grammars extra languages
            (lua, scala, graphql, ...) are deliberately extension-unwired,
            so a caller that KNOWS the language (the pipeline threads the
            caller's ``language=`` through) can still get symbol anchoring
            — the same "resolve when requested explicitly" contract the
            grammar resolver honors. Wins over suffix detection when both
            resolve (the caller's explicit statement is authoritative).

    Returns:
        A list of :class:`ASTNode` matching :func:`get_ast_map`'s shape.
        Empty list if the language is unsupported or parsing fails.
    """
    from ..data_gen.ast_analyzer import (
        canonical_language_name,
        detect_language,
        get_parser,
    )

    if language is not None:
        language = canonical_language_name(language)
    else:
        language = detect_language(file_path)
    if language is None:
        return []

    try:
        parser = get_parser(language)
        tree = parser.parse(source_code.encode("utf-8"))
    except (ValueError, RuntimeError, ImportError):
        return []

    source_bytes = source_code.encode("utf-8")
    root = tree.root_node

    # Step D2 stress: the split-lines view the trailing-blank trim below
    # scans (one O(lines) pass, shared by every trimmed symbol span).
    trim_lines: list[str] | None = None

    nodes: list[ASTNode] = []
    seen_keys: set[tuple[int, int, str]] = set()

    def _identifier_text(node) -> str:
        """Extract the name identifier for a definition node."""
        # Tree-sitter conventional `name` field — works for most.
        name_child = node.child_by_field_name("name")
        if name_child is not None:
            return source_bytes[name_child.start_byte:name_child.end_byte].decode(
                "utf-8", errors="replace",
            )
        # Fallback: scan for common identifier node types.
        for child in node.children:
            if child.type in (
                "identifier", "type_identifier", "property_identifier",
                "field_identifier", "constant", "scoped_identifier",
            ):
                return source_bytes[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace",
                )
        # B3: C/C++ `function_definition` carries the name inside a
        # `function_declarator` (the `declarator` field), not as a direct
        # identifier child — descend into it so C/C++ functions surface in
        # the map instead of being silently dropped as anonymous.
        for child in node.children:
            if child.type == "function_declarator":
                identifier = _first_descendant_of_type(
                    child, ("identifier", "field_identifier"),
                )
                if identifier is not None:
                    return source_bytes[
                        identifier.start_byte:identifier.end_byte
                    ].decode("utf-8", errors="replace")
        return ""

    def _add(name: str, kind: str, start: int, end: int, parent: str | None) -> None:
        if not name or name == "<anonymous>":
            return
        key = (start, end, name)
        if key in seen_keys:
            return
        seen_keys.add(key)
        nodes.append(ASTNode(
            name=name, kind=kind,
            line_start=start, line_end=end,
            signature="", parent=parent,
        ))

    def _line_end(node) -> int:
        """1-indexed last line of ``node`` (B3 end-point rule).

        tree-sitter's ``end_point`` sits AFTER the last byte. A node whose
        last byte is a newline — markdown sections, TOML tables and YAML
        block values swallow the blank line(s) before the next construct —
        lands its end_point at column 0 of the NEXT row, so the naive
        ``end_point[0] + 1`` reports one line PAST the node and a delete or
        insert-after spliced from that span eats (or lands inside) the
        following block. When the end column is 0 the node's real last line
        is the row before the end point; otherwise the last byte sits on
        the end row itself.
        """
        end_row, end_col = node.end_point
        return end_row if end_col == 0 else end_row + 1

    func_types = _FUNCTION_LIKE_NODE_TYPES.get(language, set())
    class_types = _CLASS_LIKE_NODE_TYPES.get(language, set())
    const_types = _CONST_LIKE_NODE_TYPES.get(language, set())
    format_spec = _FORMAT_SYMBOL_SPECS.get(language)

    def _walk(node, parent_class: str | None) -> None:
        nonlocal trim_lines
        nt = node.type

        # B3: data/config/markup formats — one declarative spec per
        # language (see _FORMAT_SYMBOL_SPECS for the per-format symbol
        # semantics). A matched node is added under its spec's kind and,
        # when the spec nests symbols, its children are walked with the
        # symbol as parent so dotted qualification can disambiguate
        # repeats.
        if format_spec is not None and nt in format_spec.node_types:
            name = format_spec.extract(node, language, source_bytes)
            start = node.start_point[0] + 1
            end = _line_end(node)
            if format_spec.trim_trailing_blank_lines:
                if trim_lines is None:
                    trim_lines = source_code.splitlines()
                while end > start and not trim_lines[end - 1].strip():
                    end -= 1
            _add(
                name,
                format_spec.kind_by_node.get(nt, format_spec.kind),
                start, end, parent_class,
            )
            if format_spec.recurse:
                for child in node.children:
                    _walk(child, name if name else parent_class)
            return

        # Python: decorated_definition wraps the real definition. Use its
        # span (which covers the decorators) and the inner definition's name.
        if language == "python" and nt == "decorated_definition":
            inner = None
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    inner = child
                    break
            if inner is not None:
                name = _identifier_text(inner)
                start = node.start_point[0] + 1
                end = _line_end(node)
                if inner.type == "class_definition":
                    _add(name, "class", start, end, parent_class)
                    # Recurse inside the class body looking for methods.
                    for child in inner.children:
                        _walk(child, name)
                else:
                    kind = "method" if parent_class else "function"
                    _add(name, kind, start, end, parent_class)
                return

        # Elixir: def/defp/defmacro live on `call` nodes — defer to the
        # existing elixir helpers for accurate detection.
        if language == "elixir" and nt == "call":
            from ..data_gen.ast_analyzer import (
                _elixir_definition_name,
                _is_elixir_function_node,
                _is_elixir_module_node,
            )
            if _is_elixir_module_node(node, source_bytes):
                name = _elixir_definition_name(node, source_bytes)
                start = node.start_point[0] + 1
                end = _line_end(node)
                _add(name, "class", start, end, parent_class)
                for child in node.children:
                    _walk(child, name)
                return
            if _is_elixir_function_node(node, source_bytes):
                name = _elixir_definition_name(node, source_bytes)
                start = node.start_point[0] + 1
                end = _line_end(node)
                kind = "method" if parent_class else "function"
                _add(name, kind, start, end, parent_class)
                return
            # Other `call` (e.g. import) — don't descend.
            return

        if nt in class_types:
            name = _identifier_text(node)
            start = node.start_point[0] + 1
            end = _line_end(node)
            _add(name, "class", start, end, parent_class)
            # Recurse into this class's body so methods/nested classes
            # get populated with parent=<class name>.
            for child in node.children:
                _walk(child, name)
            return

        if nt in func_types:
            name = _identifier_text(node)
            start = node.start_point[0] + 1
            end = _line_end(node)
            kind = "method" if parent_class else "function"
            _add(name, kind, start, end, parent_class)
            # Don't recurse into function bodies looking for more defs
            # (rare; top-level entity enumeration is what callers want).
            return

        if nt in const_types:
            name = _const_name(node, language, source_bytes)
            if name:
                start = node.start_point[0] + 1
                end = _line_end(node)
                _add(name, "constant", start, end, parent_class)
            return

        # Recurse structurally — needed for wrappers like TS
        # `export_statement` / `lexical_declaration` that contain the
        # actual definition one level down.
        for child in node.children:
            _walk(child, parent_class)

    for top_child in root.children:
        _walk(top_child, None)

    nodes.sort(key=lambda n: n.line_start)
    return nodes


# Function-like tree-sitter node types per language. Kept local to avoid
# import-time coupling with snippet_analysis (which has its own map).
_FUNCTION_LIKE_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "decorated_definition"},
    "javascript": {
        "function_declaration", "method_definition",
        "generator_function_declaration",
    },
    "typescript": {
        "function_declaration", "method_definition", "method_signature",
    },
    "tsx": {
        "function_declaration", "method_definition", "method_signature",
    },
    "rust": {"function_item"},
    "go": {"function_declaration", "method_declaration"},
    "java": {"method_declaration", "constructor_declaration"},
    "c": {"function_definition"},
    "cpp": {"function_definition"},
    "ruby": {"method", "singleton_method"},
    "swift": {"function_declaration", "initializer_declaration"},
    "kotlin": {"function_declaration"},
    "c_sharp": {
        "method_declaration", "constructor_declaration",
        # B3: a bare C# method (as the direct-swap snippet parse sees it)
        # is a top-level local function in this grammar. Real local
        # functions inside method bodies are never reached — the walker
        # stops at the enclosing method — so this extra kind only serves
        # top-level functions and snippet parses, which is exactly what
        # the direct-swap gate needs.
        "local_function_statement",
    },
    "php": {"function_definition", "method_declaration"},
    "elixir": {"call"},
    # B3: shell functions are the addressable symbols of a bash script
    # (the `name` field is conventional). Variable assignments are
    # deliberately not symbols — see _FORMAT_SYMBOL_SPECS notes.
    "bash": {"function_definition"},
    # B3: all-grammars extra languages sampled by the golden matrix. The
    # grammar resolver serves them on every install that has the extra;
    # these rows give their functions symbol semantics so `after=` /
    # `replace=` anchor on real definitions.
    "lua": {"function_declaration", "function_definition"},
    "scala": {"function_definition"},
}

# Class-like tree-sitter node types per language.
_CLASS_LIKE_NODE_TYPES: dict[str, set[str]] = {
    "python": {"class_definition"},
    "javascript": {"class_declaration"},
    "typescript": {
        "class_declaration", "interface_declaration", "type_alias_declaration",
    },
    "tsx": {
        "class_declaration", "interface_declaration", "type_alias_declaration",
    },
    "rust": {"struct_item", "enum_item", "trait_item", "impl_item", "union_item"},
    "go": {"type_declaration"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration"},
    "c": {"struct_specifier", "enum_specifier"},
    "cpp": {"class_specifier", "struct_specifier"},
    "ruby": {"class", "module"},
    "swift": {"class_declaration", "struct_declaration", "protocol_declaration"},
    "kotlin": {
        "class_declaration", "object_declaration",
    },
    "c_sharp": {
        "class_declaration", "interface_declaration", "struct_declaration",
    },
    "php": {
        "class_declaration", "interface_declaration", "trait_declaration",
    },
    "elixir": {"call"},
    # B3: scala objects/classes (all-grammars extra, sampled by goldens).
    "scala": {"object_definition", "class_definition", "trait_definition"},
}

# Constant/variable declaration node types — anything a caller might name
# as `replace=X` or `after=X` at the top level (e.g. Rust `const`, TS
# `const`/`let`, Go `var`/`const`).
_CONST_LIKE_NODE_TYPES: dict[str, set[str]] = {
    "rust": {"const_item", "static_item"},
    "typescript": {"lexical_declaration", "variable_declaration"},
    "tsx": {"lexical_declaration", "variable_declaration"},
    "javascript": {"lexical_declaration", "variable_declaration"},
    "go": {"var_declaration", "const_declaration"},
    "java": {"field_declaration"},
    "c": {"declaration"},
    "cpp": {"declaration"},
    "kotlin": {"property_declaration"},
    "swift": {"property_declaration"},
    "c_sharp": {"field_declaration"},
    "php": {"const_declaration"},
    # Python top-level assignments are exposed as `expression_statement`
    # containing an `assignment`; we omit them here because tldr only
    # surfaces module-level UPPER_SNAKE as a "constant" heuristically,
    # and this path rarely targets Python constants by name.
}


# ---------------------------------------------------------------------------
# B3: name extractors for the per-format symbol specs below. Each is a pure
# structural function (node, source_bytes) -> name, registered in a spec —
# never selected by an if-chain.
# ---------------------------------------------------------------------------

def _first_child_of_type(node, types: tuple[str, ...]):
    """First direct child whose type is in ``types``, or None."""
    for child in node.children:
        if child.type in types:
            return child
    return None


def _first_descendant_of_type(node, types: tuple[str, ...]):
    """First descendant (depth-first) whose type is in ``types``, or None."""
    for child in node.children:
        if child.type in types:
            return child
        found = _first_descendant_of_type(child, types)
        if found is not None:
            return found
    return None


def _node_text(node, source_bytes: bytes) -> str:
    if node is None:
        return ""
    return source_bytes[node.start_byte:node.end_byte].decode(
        "utf-8", errors="replace",
    )


def _strip_quote_pair(text: str) -> str:
    """Remove one wrapping pair of matching single/double quotes."""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


def _html_element_name(node, language: str, source_bytes: bytes) -> str:
    """Name of an HTML symbol element: the value of its ``id`` attribute.

    Both the open tag and a self-closing tag carry attributes. An element
    without an ``id`` gets no name (its node is simply not a symbol).
    """
    for container in node.children:
        if container.type not in ("start_tag", "self_closing_tag"):
            continue
        for attr in container.children:
            if attr.type != "attribute":
                continue
            name_node = _first_child_of_type(attr, ("attribute_name",))
            if _node_text(name_node, source_bytes) != "id":
                continue
            value_node = _first_descendant_of_type(attr, ("attribute_value",))
            return _node_text(value_node, source_bytes)
    return ""


def _xml_element_name(node, language: str, source_bytes: bytes) -> str:
    """Name of an XML element: its start tag's ``Name`` token."""
    stag = _first_child_of_type(node, ("STag",))
    if stag is None:
        return ""
    return _node_text(_first_child_of_type(stag, ("Name",)), source_bytes)


def _markdown_section_name(node, language: str, source_bytes: bytes) -> str:
    """Name of a markdown section: its heading text (markers stripped).

    Works for ATX (``## Title``) and setext (``Title`` + underline) shapes:
    both carry the heading text on their first line.
    """
    heading = _first_child_of_type(node, ("atx_heading", "setext_heading"))
    if heading is None:
        return ""
    first_line = _node_text(heading, source_bytes).split("\n", 1)[0]
    return first_line.lstrip("#").strip()


def _json_pair_name(node, language: str, source_bytes: bytes) -> str:
    """Name of a JSON pair: its key, unquoted."""
    key = node.child_by_field_name("key")
    return _strip_quote_pair(_node_text(key, source_bytes))


def _yaml_key_name(node, language: str, source_bytes: bytes) -> str:
    """Name of a YAML block-mapping pair: its key, unquoted."""
    key = node.child_by_field_name("key")
    return _strip_quote_pair(_node_text(key, source_bytes))


def _css_rule_name(node, language: str, source_bytes: bytes) -> str:
    """Name of a CSS rule set: its full selector text (``.header, .nav``)."""
    return _node_text(
        _first_child_of_type(node, ("selectors",)), source_bytes,
    )


_TOML_KEY_TYPES = ("bare_key", "dotted_key", "quoted_key")


def _toml_table_name(node, language: str, source_bytes: bytes) -> str:
    """Name of a TOML table: its (possibly dotted) key text."""
    key = _first_child_of_type(node, _TOML_KEY_TYPES)
    return _strip_quote_pair(_node_text(key, source_bytes))


def _sql_object_name(node, language: str, source_bytes: bytes) -> str:
    """Name of a SQL CREATE statement's object.

    CREATE INDEX names the index with a direct ``identifier`` (its
    ``object_reference`` child is the ON target); CREATE TABLE/VIEW name
    the object through an ``object_reference``. Direct identifier first,
    object_reference fallback — order does the disambiguation.
    """
    direct = _first_child_of_type(node, ("identifier",))
    if direct is not None:
        return _node_text(direct, source_bytes)
    ref = _first_descendant_of_type(node, ("object_reference",))
    if ref is not None:
        return _node_text(
            _first_child_of_type(ref, ("identifier",)), source_bytes,
        )
    return ""


def _dockerfile_stage_name(node, language: str, source_bytes: bytes) -> str:
    """Name of a Dockerfile build stage: its ``AS`` alias, else its image."""
    alias = _first_child_of_type(node, ("image_alias",))
    if alias is not None:
        return _node_text(alias, source_bytes)
    return _node_text(
        _first_child_of_type(node, ("image_spec",)), source_bytes,
    )


def _name_field_name(node, language: str, source_bytes: bytes) -> str:
    """Name of a definition node that uses the conventional ``name`` field.

    Falls back to a direct ``name``-typed child for grammars that leave the
    field untagged (graphql's ``object_type_definition`` carries a bare
    ``name`` child).
    """
    name_node = node.child_by_field_name("name")
    if name_node is None:
        name_node = _first_child_of_type(node, ("name",))
    return _node_text(name_node, source_bytes)


@dataclass(frozen=True)
class _FormatSymbolSpec:
    """Declarative symbol semantics for one data/config/markup format.

    Attributes:
        node_types: tree-sitter node kinds that ARE addressable symbols.
        kind: the ASTNode.kind label reported for those symbols.
        extract: name extractor — ``(node, language, source_bytes) -> str``;
            an empty name means the node is not a symbol after all.
        recurse: walk the node's children with the symbol as parent so
            nested symbols qualify (html/xml sections, json/yaml keys).
        kind_by_node: per-node-kind kind-label overrides (SQL's
            table/view/index family).
        trim_trailing_blank_lines: pull the symbol span's END up onto the
            last CONTENT line. Grammars whose node byte-range swallows the
            blank separator before the next construct (markdown's
            ``section`` extends to the next heading) would otherwise make
            every replace/delete splice eat that separator — a byte of
            layout the op never declared (Step D2 stress finding: a
            section replace at 100MB dropped the blank between sections).
            The trim is content-preserving: blank lines carry no symbol
            content, and the following construct's own leading layout is
            untouched.
    """

    node_types: tuple[str, ...]
    kind: str
    extract: Callable[..., str]
    recurse: bool = False
    kind_by_node: dict[str, str] = field(default_factory=dict)
    trim_trailing_blank_lines: bool = False


# ---------------------------------------------------------------------------
# B3: per-format symbol semantics for data/config/markup languages.
#
# The function/class/const tables above describe CODE languages. Data and
# markup formats anchor their edits on DIFFERENT constructs, so each format
# declares its own symbol semantics here: which tree-sitter node kinds count
# as an addressable symbol, what the symbol is named, and whether symbols
# nest (nested symbols get `parent=<enclosing symbol>` so the dotted
# qualification in _resolve_symbol can disambiguate repeats).
#
# Adding a format = one table row (declarative, no walker branches). The
# chosen semantics per format:
#
#   html       an element carrying an `id` attribute; named by the
#              attribute VALUE. `id` is HTML's naming mechanism — tag names
#              repeat and address nothing. Elements without an id are
#              content (still reachable via content anchors). Symbols nest.
#   xml        every `element`, named by its tag name. XML has no other
#              addressable construct; sibling repeats are refused as
#              ambiguous by _resolve_symbol (fail loud, never first-match).
#              Symbols nest (parent = enclosing element name).
#   markdown   `section` (heading + its body), named by the heading text.
#              Sections nest; a section's span covers its subsections, so
#              replace=/delete= of a parent section takes them with it —
#              target leaf sections for member edits.
#   json       `pair` (key: value), named by the key. Nested objects'
#              pairs become nested symbols (parent = enclosing key).
#   yaml       `block_mapping_pair`, named by the key (quotes stripped).
#              Nested mappings' pairs become nested symbols.
#   css        `rule_set`, named by its full selector text (`.header`,
#              `body`, `a:hover`). Rules inside @media blocks are found by
#              structural descent; @media itself is a container, not a
#              symbol.
#   toml       `table`/`table_array_element`, named by the key text
#              (dotted keys stay dotted, e.g. `tool.pytest`). Top-level
#              bare pairs are deliberately NOT symbols: the table is the
#              TOML addressing unit.
#   sql        CREATE statements that define a schema object
#              (create_table/view/materialized_view/index), named by the
#              object name; DML (SELECT/INSERT/UPDATE) is an operation, not
#              a definition, and is never a symbol.
#   dockerfile `from_instruction` (a build stage), named by its `AS` alias
#              when present, else by its image spec. Non-FROM instructions
#              are steps, not symbols.
#   graphql    (all-grammars extra, sampled by the goldens) schema type
#              definitions (object/interface/enum/union/input/scalar),
#              named by their `name` field.
#   bash       (in _FUNCTION_LIKE_NODE_TYPES) `function_definition`.
#              Shell variable assignments are deliberately not symbols.
# ---------------------------------------------------------------------------
_FORMAT_SYMBOL_SPECS: dict[str, _FormatSymbolSpec] = {
    "html": _FormatSymbolSpec(
        node_types=("element",),
        kind="element",
        extract=_html_element_name,
        recurse=True,
    ),
    "xml": _FormatSymbolSpec(
        node_types=("element",),
        kind="element",
        extract=_xml_element_name,
        recurse=True,
    ),
    "markdown": _FormatSymbolSpec(
        node_types=("section",),
        kind="section",
        extract=_markdown_section_name,
        recurse=True,
        trim_trailing_blank_lines=True,
    ),
    "json": _FormatSymbolSpec(
        node_types=("pair",),
        kind="key",
        extract=_json_pair_name,
        recurse=True,
    ),
    "yaml": _FormatSymbolSpec(
        node_types=("block_mapping_pair",),
        kind="key",
        extract=_yaml_key_name,
        recurse=True,
    ),
    "css": _FormatSymbolSpec(
        node_types=("rule_set",),
        kind="rule",
        extract=_css_rule_name,
    ),
    "toml": _FormatSymbolSpec(
        node_types=("table", "table_array_element"),
        kind="table",
        extract=_toml_table_name,
    ),
    "sql": _FormatSymbolSpec(
        node_types=(
            "create_table", "create_view", "create_materialized_view",
            "create_index",
        ),
        kind="table",
        kind_by_node={
            "create_table": "table",
            "create_view": "view",
            "create_materialized_view": "view",
            "create_index": "index",
        },
        extract=_sql_object_name,
    ),
    "dockerfile": _FormatSymbolSpec(
        node_types=("from_instruction",),
        kind="stage",
        extract=_dockerfile_stage_name,
    ),
    # B3: all-grammars extra, sampled by the golden matrix. Schema type
    # definitions are the addressable symbols of a GraphQL document.
    "graphql": _FormatSymbolSpec(
        node_types=(
            "object_type_definition", "interface_type_definition",
            "enum_type_definition", "union_type_definition",
            "input_object_type_definition", "scalar_type_definition",
        ),
        kind="type",
        extract=_name_field_name,
    ),
}


def _const_name(node, language: str, source_bytes: bytes) -> str:
    """Extract the declared name from a constant/variable declaration node.

    Returns the FIRST declared name — if the declaration binds multiple
    identifiers (e.g. ``const A = 1, B = 2;``), subsequent names are not
    surfaced individually (matches tldr's common behavior).
    """
    # Rust: const_item / static_item use a `name` field.
    # Java field_declaration / C(++) declaration / PHP const_declaration:
    # find the first identifier child.
    name_field = node.child_by_field_name("name")
    if name_field is not None:
        return source_bytes[name_field.start_byte:name_field.end_byte].decode(
            "utf-8", errors="replace",
        )

    # TS/JS lexical_declaration: variable_declarator -> identifier.
    # B3: c_sharp field_declaration wraps its declarator one level deeper
    # (variable_declaration -> variable_declarator -> identifier), so the
    # declarator scan descends through that wrapper too.
    for child in node.children:
        declarators = (
            [child] if child.type in ("variable_declarator", "init_declarator")
            else (
                [gc for gc in child.children if gc.type == "variable_declarator"]
                if child.type == "variable_declaration" else []
            )
        )
        for declarator in declarators:
            for gc in declarator.children:
                if gc.type in ("identifier", "property_identifier"):
                    return source_bytes[gc.start_byte:gc.end_byte].decode(
                        "utf-8", errors="replace",
                    )

    # Kotlin property_declaration: has a `variable_declaration` child
    # which contains a `simple_identifier` (older grammars) or a bare
    # `identifier` (current tree-sitter-kotlin).
    if language == "kotlin":
        for child in node.children:
            if child.type == "variable_declaration":
                for gc in child.children:
                    if gc.type in ("simple_identifier", "identifier"):
                        return source_bytes[gc.start_byte:gc.end_byte].decode(
                            "utf-8", errors="replace",
                        )

    # PHP const_declaration: `const ELEMENT = 10;` — the name sits in a
    # `const_element` child (its `name` field is untagged in this grammar,
    # so match the `name`-typed child node).
    if language == "php":
        for child in node.children:
            if child.type == "const_element":
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    name_node = _first_child_of_type(child, ("name",))
                if name_node is not None:
                    return source_bytes[
                        name_node.start_byte:name_node.end_byte
                    ].decode("utf-8", errors="replace")

    # Go var_declaration / const_declaration: descend through var_spec /
    # const_spec to find the identifier.
    if language == "go":
        for child in node.children:
            if child.type in ("var_spec", "const_spec"):
                for gc in child.children:
                    if gc.type == "identifier":
                        return source_bytes[gc.start_byte:gc.end_byte].decode(
                            "utf-8", errors="replace",
                        )

    # Generic fallback: first identifier-ish descendant.
    for child in node.children:
        if child.type in ("identifier", "property_identifier", "field_identifier"):
            return source_bytes[child.start_byte:child.end_byte].decode(
                "utf-8", errors="replace",
            )

    return ""


def get_ast_map(file_path: str, total_lines: int = 0) -> list[ASTNode]:
    """Get AST definitions with line ranges and parent class info.

    Primary: tldr structure (has line_start + line_end for most languages),
    enriched with parent info from tldr extract (class->method hierarchy).
    Fallback: tldr extract only (has line_number + parent, end lines computed).

    .. note::
       This reads from disk and delegates to the tldr daemon, which
       has a well-known file-watcher invalidation race. For callers that
       already hold the source text, prefer get_ast_map_from_source
       -- it parses in-memory and is race-free.
    """
    import contextlib
    import logging
    import os
    import re
    import tempfile
    from pathlib import Path

    _log = logging.getLogger("fastedit.chunked_merge")

    # tldr (like tree-sitter) counts rows by scanning for "\n" -- a bare CR
    # (classic-Mac line ending) is invisible to it, collapsing every line of
    # a CR-only or CR-mixed file into one, which silently corrupts the
    # returned line_start/line_end for every symbol. Point tldr at an
    # LF-normalized temp copy instead when a bare CR is present: swapping a
    # "\r" not followed by "\n" for "\n" is a same-length, same-position
    # substitution, so the line numbers it returns stay valid indices into
    # the real file's own line list -- callers always slice their own
    # in-memory original_lines, never this temp file's content.
    query_path = file_path
    tmp_path = None
    try:
        raw = Path(file_path).read_bytes()
        if re.search(rb"\r(?!\n)", raw):
            normalized = re.sub(rb"\r(?!\n)", b"\n", raw)
            fd, tmp_path = tempfile.mkstemp(suffix=Path(file_path).suffix)
            os.write(fd, normalized)
            os.close(fd)
            query_path = tmp_path
    except OSError:
        pass

    try:
        nodes = _get_ast_via_structure(query_path)
        if nodes:
            _enrich_parents_from_extract(nodes, query_path)
            _log.info("get_ast_map: %d nodes via structure for %s", len(nodes), file_path)
            return nodes
        _log.debug("get_ast_map: structure returned 0 nodes, trying extract for %s", file_path)
        nodes = _get_ast_via_extract(query_path, total_lines)
        _log.info("get_ast_map: %d nodes via extract for %s", len(nodes), file_path)
        return nodes
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


def _enrich_parents_from_extract(nodes: list[ASTNode], file_path: str) -> None:
    """Set parent class names on method nodes using tldr extract hierarchy.

    tldr structure returns a flat list (no parent info). tldr extract nests
    methods under their class. We call extract, build a line→parent map,
    and apply it to the structure nodes.
    """
    try:
        result = subprocess.run(
            ["tldr", "extract", file_path, "--format", "json"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode != 0:
            return
        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return

    # Build line_number → parent_class_name map from extract hierarchy
    line_to_parent: dict[int, str] = {}
    for cls in data.get("classes", []):
        cls_name = cls.get("name", "")
        if not cls_name:
            continue
        for method in cls.get("methods", []):
            line = method.get("line_number")
            if line:
                line_to_parent[line] = cls_name

    # Apply to nodes: match by line_start
    for node in nodes:
        if node.kind in ("method", "function") and node.line_start in line_to_parent:
            node.parent = line_to_parent[node.line_start]


def _get_ast_via_structure(file_path: str) -> list[ASTNode]:
    """Primary: tldr structure --format compact → definitions with line ranges."""
    try:
        result = subprocess.run(
            ["tldr", "structure", file_path, "--format", "compact"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode != 0:
            return []

        data = json.loads(result.stdout)
        files = data.get("files", [])
        if not files:
            return []
        definitions = files[0].get("definitions", [])

        nodes = []
        for defn in definitions:
            if defn.get("line_start") and defn.get("line_end"):
                nodes.append(ASTNode(
                    name=defn["name"],
                    kind=defn.get("kind", "unknown"),
                    line_start=defn["line_start"],
                    line_end=defn["line_end"],
                    signature=defn.get("signature", ""),
                ))
        return nodes
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return []


def _get_ast_via_extract(file_path: str, total_lines: int = 0) -> list[ASTNode]:
    """Fallback: tldr extract → line_number only, compute end lines."""
    try:
        result = subprocess.run(
            ["tldr", "extract", file_path, "--format", "json"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode != 0:
            return []

        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return []

    # Collect all entries with line numbers
    # (name, kind, line, signature, parent_class_name_or_None)
    raw: list[tuple[str, str, int, str, str | None]] = []

    for fn in data.get("functions", []):
        if fn.get("name") and fn.get("line_number"):
            raw.append((fn["name"], "function", fn["line_number"], "", None))

    for cls in data.get("classes", []):
        cls_name = cls.get("name", "")
        cls_line = cls.get("line_number", 0)
        if cls_name and cls_line:
            raw.append((cls_name, "class", cls_line, "", None))
        for method in cls.get("methods", []):
            if method.get("name") and method.get("line_number"):
                raw.append((method["name"], "method", method["line_number"], "", cls_name))

    if not raw:
        return []

    raw.sort(key=lambda x: x[2])

    # Compute end lines from consecutive entries
    if not total_lines:
        try:
            with open(file_path) as f:
                total_lines = sum(1 for _ in f)
        except OSError:
            total_lines = raw[-1][2] + 50  # rough estimate

    # B36: the old `next_entry.line_start - 1` rule truncated a class to the
    # line above its first method (the method is nested INSIDE the class, but
    # the rule clamped to it anyway) and let a method's end swallow following
    # lines tldr extract never reported (module constants, comments, the next
    # symbol's header). Use real end positions from the in-memory tree-sitter
    # map where the grammar is available; entries it cannot match fall back
    # to an order-aware clamp: an entry ends just before the next entry that
    # is NOT nested inside it, so a class spans its body and a method cannot
    # cross a sibling's header.
    true_ends = _extract_true_end_lines(file_path, raw)

    nodes = []
    for i, (name, kind, line_start, sig, parent) in enumerate(raw):
        line_end = true_ends.get((name, parent, line_start))
        if line_end is None:
            line_end = _clamped_extract_end(i, raw, total_lines)
        nodes.append(ASTNode(
            name=name, kind=kind,
            line_start=line_start, line_end=line_end,
            signature=sig, parent=parent,
        ))
    return nodes


def _extract_true_end_lines(
    file_path: str,
    raw: list[tuple[str, str, int, str, str | None]],
) -> dict[tuple[str, str | None, int], int]:
    """Map ``(name, parent, line_start)`` to the tree-sitter ``line_end``.

    tldr extract only reports start lines. Reading the file and parsing it
    in-memory yields exact end positions for every entry the grammar covers;
    the result is keyed so each extract entry can adopt its own node's end.
    Returns an empty dict when the file cannot be read or the language has
    no tree-sitter grammar — callers then fall back to the order-aware
    clamp. Line numbers stay valid against the original file either way:
    a bare CR is replaced with LF by a same-length, same-position
    substitution before parsing.
    """
    from pathlib import Path

    from ..split_join import normalize_bare_cr_for_ast

    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return {}

    reference = get_ast_map_from_source(normalize_bare_cr_for_ast(source), file_path)
    if not reference:
        return {}

    ends: dict[tuple[str, str | None, int], int] = {}
    for node in reference:
        ends.setdefault((node.name, node.parent, node.line_start), node.line_end)
    return ends


def _clamped_extract_end(
    index: int,
    raw: list[tuple[str, str, int, str, str | None]],
    total_lines: int,
) -> int:
    """Order-aware end line for extract entry ``index`` (B36 fallback).

    The entry ends immediately before the next entry that is NOT nested
    inside it — its own methods and inner definitions are skipped, so a
    class spans its body instead of stopping at its first member. When every
    following entry is nested inside it, the entry extends to EOF.
    """
    for j in range(index + 1, len(raw)):
        if not _extract_entry_nested_in(j, index, raw):
            return raw[j][2] - 1
    return total_lines


def _extract_entry_nested_in(
    entry_idx: int,
    ancestor_idx: int,
    raw: list[tuple[str, str, int, str, str | None]],
) -> bool:
    """True when ``raw[entry_idx]`` is (per its parent chain) inside
    ``raw[ancestor_idx]``.

    Walks the parent-name chain upward, at each step choosing the closest
    preceding entry with that name — the best available evidence when only
    start lines and immediate parent names are known. The ``seen`` set guards
    against parent-name cycles.
    """
    child = raw[entry_idx]
    ancestor = raw[ancestor_idx]
    if entry_idx == ancestor_idx or child[2] <= ancestor[2]:
        return False
    parent_name = child[4]
    cursor_line = child[2]
    seen: set[str] = set()
    while parent_name and parent_name not in seen:
        seen.add(parent_name)
        if parent_name == ancestor[0]:
            return True
        step = None
        for mid in raw:
            if (mid[0] == parent_name and mid[2] < cursor_line
                    and (step is None or mid[2] > step[2])):
                step = mid
        if step is None:
            return False
        parent_name = step[4]
        cursor_line = step[2]
    return False


def _contains_span(outer: ASTNode, inner: ASTNode) -> bool:
    """True when ``outer``'s line span strictly encloses ``inner``'s.

    Equal spans do not count as containment: two entries covering the same
    lines are duplicates of one symbol, not ancestor and descendant.
    """
    if (outer.line_start, outer.line_end) == (inner.line_start, inner.line_end):
        return False
    return outer.line_start <= inner.line_start and outer.line_end >= inner.line_end


def _qualified_name(node: ASTNode, ast_nodes: list[ASTNode]) -> str:
    """Full dotted path to ``node`` (e.g. ``Outer.Inner.save``).

    Walks the ``parent`` chain outward, verifying each step by line
    containment so two same-named ancestors in different scopes cannot be
    confused. Falls back to the raw parent name when the enclosing node is
    not present in the map (tldr occasionally lists methods without their
    class). Each step moves to a strictly larger span, so the walk cannot
    cycle.
    """
    labels = [node.name]
    current = node
    while current.parent:
        container = next(
            (n for n in ast_nodes
             if n is not current
             and n.name == current.parent
             and _contains_span(n, current)),
            None,
        )
        if container is None:
            labels.append(current.parent)
            break
        labels.append(container.name)
        current = container
    return ".".join(reversed(labels))


def _has_ancestor_chain(
    node: ASTNode,
    parts: list[str],
    ast_nodes: list[ASTNode],
) -> bool:
    """True when ``node``'s ancestry matches ``parts`` (outermost first).

    The immediate parent must equal the innermost segment (the ``parent``
    field, populated by both the tldr extract hierarchy and the in-memory
    tree-sitter walker). Every outer segment is verified by walking to the
    enclosing node — line containment disambiguates same-named ancestors.
    """
    if not parts:
        return True
    inner_name = parts[-1]
    if node.parent != inner_name:
        return False
    containers = [
        n for n in ast_nodes
        if n is not node
        and n.name == inner_name
        and _contains_span(n, node)
    ]
    if not containers:
        # The parent name matches but no enclosing node is present in the
        # map — accept on the name evidence available; outer segments cannot
        # be verified and are treated as satisfied.
        return True
    return any(_has_ancestor_chain(c, parts[:-1], ast_nodes) for c in containers)


def _resolve_symbol(name: str, ast_nodes: list[ASTNode]) -> ASTNode | None:
    """Find an AST node by name, supporting dotted qualification.

    Grammar:
      - bare identifier: ``save``
      - dotted path, any depth: ``Class.method``, ``Class.Inner.method``

    A dotted path matches the node whose name is the last segment and whose
    ancestry matches the remaining segments (see
    :func:`_has_ancestor_chain`). A bare name that matches more than one
    DISTINCT node — two classes each defining ``save`` — raises ValueError
    listing every match's qualified name (B25): first-match-wins silently
    edited the wrong symbol. A dotted path that still matches several nodes
    (e.g. two same-named nested classes) is ambiguous too.

    A match fully contained inside another match (a constructor sharing its
    class's name) is not a competitor: the outermost match wins and the
    member stays reachable via its qualified path.

    B3: a node whose FULL name equals the query resolves first. Data and
    config formats name their symbols with dots baked in (CSS selectors,
    TOML dotted keys, Elixir dotted aliases), which the dotted-qualification
    grammar below would mis-split into ancestor segments; an exact literal
    name is the most specific reading of the query and wins whenever a
    single distinct symbol carries it. Multiple distinct literal matches are
    ambiguous (same B25 refusal as bare duplicates).

    Returns None when nothing matches — callers keep their existing
    not-found refusal behaviour.
    """
    # B3 literal pass: one distinct symbol named exactly `name` is it.
    literal: dict[tuple[str | None, int, int], ASTNode] = {}
    for node in ast_nodes:
        if node.name == name:
            literal.setdefault((node.parent, node.line_start, node.line_end), node)
    if literal:
        literal_matches = [
            n for n in literal.values()
            if not any(
                other is not n and _contains_span(other, n)
                for other in literal.values()
            )
        ]
        if len(literal_matches) == 1:
            return literal_matches[0]
        qualified = sorted({_qualified_name(n, ast_nodes) for n in literal_matches})
        raise ValueError(
            f"Symbol '{name}' is ambiguous: {len(literal_matches)} definitions "
            f"match. Qualify it as one of: {', '.join(qualified)}"
        )

    parts = name.split(".")
    leaf = parts[-1]

    # Collapse map entries describing the SAME symbol twice (same name,
    # parent, and span) so duplicated maps cannot manufacture a false
    # ambiguity. Distinct spans mean distinct symbols — e.g. two overloaded
    # method signatures stay distinct and ambiguous.
    unique: dict[tuple[str | None, int, int], ASTNode] = {}
    for node in ast_nodes:
        if node.name == leaf:
            unique.setdefault((node.parent, node.line_start, node.line_end), node)
    matches = list(unique.values())

    if len(parts) > 1:
        matches = [
            n for n in matches
            if _has_ancestor_chain(n, parts[:-1], ast_nodes)
        ]

    # A match fully contained inside another match is a MEMBER of it — a
    # Java/Kotlin constructor sharing its class's name, a nested class with
    # the same name. For the bare name the outermost match is the target the
    # caller means (legacy first-match behaviour agreed: the class header
    # sorts first); the member stays reachable via its qualified path
    # (``Store.Store``). Only genuinely competing matches — siblings or
    # unrelated definitions — are ambiguous (B25).
    matches = [
        n for n in matches
        if not any(other is not n and _contains_span(other, n) for other in matches)
    ]

    if not matches:
        return None
    if len(matches) > 1:
        qualified = sorted({_qualified_name(n, ast_nodes) for n in matches})
        raise ValueError(
            f"Symbol '{name}' is ambiguous: {len(matches)} definitions "
            f"match. Qualify it as one of: {', '.join(qualified)}"
        )
    return matches[0]


def _qualified_symbol_names(ast_nodes: list[ASTNode]) -> list[str]:
    """Build qualified symbol names for error messages.

    Methods with a parent class become 'Class.method'.
    If a name appears only once, use the bare name.
    Duplicate names get qualified to help Claude disambiguate.
    """
    # Count occurrences of each name
    name_counts: dict[str, int] = {}
    for node in ast_nodes:
        name_counts[node.name] = name_counts.get(node.name, 0) + 1

    result = []
    for node in ast_nodes:
        if node.name == "__init__":
            # Always qualify __init__ — it's always a duplicate
            if node.parent:
                result.append(f"{node.parent}.{node.name}")
            continue
        if name_counts[node.name] > 1 and node.parent:
            result.append(f"{node.parent}.{node.name}")
        else:
            result.append(node.name)
    return result
