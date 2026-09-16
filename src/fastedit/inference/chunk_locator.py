"""Chunk location: finds which regions of a file to extract and merge.

Given a snippet and original file, uses AST analysis to determine the
minimal chunk(s) that need to be sent to the model.
"""

from __future__ import annotations

from ..split_join import normalize_bare_cr_for_ast
from .ast_utils import (
    ASTNode,
    ChunkRegion,
    _resolve_symbol,
    get_ast_map_from_source,
)
from .markers import is_marker_line
from .snippet_analysis import (
    _extract_snippet_names,
    _find_import_region,
    _find_insertion_region,
    _find_matching_nodes,
    _has_import_changes,
    _merge_overlapping_regions,
)

_MAX_BLOCK_LINES = 100  # Narrow functions larger than this to sub-blocks

# B27: minimum sliding-window content score for trusting a narrow. The old
# 0.2 accepted windows where four of five lines were NOT in the original —
# the window locked onto spurious stripped-line matches and was then cut
# ±padding RAW lines around it, handing the model a mid-block fragment to
# "repair". 0.6 means the snippet's leading window lines must mostly exist,
# contiguously, in the node before the narrow is trusted. Tunable: lower it
# and sloppy snippets narrow again (mid-block cuts return); raise it and
# large-node edits fall through to the model path more often.
_MIN_NARROW_SCORE = 0.6

# tree-sitter node types that represent block structures
_BLOCK_TYPES = {
    "for_statement", "while_statement", "if_statement",
    "with_statement", "try_statement", "match_statement",
    # JS/TS/Go/Rust/Java/etc.
    "for_in_statement", "for_of_statement", "switch_statement",
    "do_statement", "loop_expression", "match_expression",
}


def locate_chunks(
    snippet: str,
    original_code: str,
    file_path: str,
    padding: int = 30,
    language: str | None = None,
    after: str | None = None,
    replace: str | None = None,
) -> list[ChunkRegion]:
    """Locate the chunk region(s) in the original file that the snippet edits.

    Language-agnostic: uses tldr AST analysis (16 languages) with regex fallback.
    Handles single-site edits, multi-site edits, import changes, and
    targeted insertion via the ``after`` parameter.

    Args:
        snippet: The edit snippet (with or without markers).
        original_code: The full original file content.
        file_path: Path to the file (for tldr AST extraction).
        padding: Lines of context to include above and below the edit region.
        language: Optional language hint (e.g. "python", "typescript").
        after: Optional symbol name — insert new code after this function/class.
            Uses tldr to find the symbol's line range and creates a tight chunk.
        replace: Optional symbol name — replace this function/class/method entirely
            with the snippet. Uses AST to find the symbol's exact line range.

    Returns:
        List of ChunkRegion(s) to extract and merge.
    """
    original_lines = original_code.splitlines()
    total_lines = len(original_lines)

    # B26: parse the IN-MEMORY `original_code`, not the disk-cached tldr map.
    # get_ast_map consults the tldr daemon, whose salsa cache can hold
    # pre-write line numbers for a file that was just rewritten — chunks
    # spliced from stale coordinates land in the wrong region. The fast
    # paths in chunked_merge already parse `original_code` in-memory via
    # get_ast_map_from_source; locate_chunks must agree with them. A bare CR
    # is swapped for LF by a same-length, same-position substitution first
    # (tree-sitter counts rows by scanning for "\n"), keeping every returned
    # line number valid against original_lines — the same protection
    # get_ast_map's LF-normalized temp-file path used to provide.
    ast_nodes = get_ast_map_from_source(
        normalize_bare_cr_for_ast(original_code), file_path,
    )

    if not ast_nodes:
        # No AST available — use the whole file
        return [ChunkRegion(1, total_lines, ["<whole file>"])]

    # If `replace` is specified, find the named symbol's exact line range
    # and use that as the chunk region.  Supports 'Class.method' to
    # disambiguate duplicate names (e.g. two __init__ methods).
    if replace:
        target_node = _resolve_symbol(replace, ast_nodes)
        if target_node:
            # If the function is large and the snippet uses ellipsis markers,
            # narrow to just the edited sub-region within the function.
            node_size = target_node.line_end - target_node.line_start + 1
            has_ellipsis = "... existing code ..." in snippet or "# ..." in snippet
            if node_size > _MAX_BLOCK_LINES and has_ellipsis:
                narrowed = _narrow_large_node(
                    target_node, snippet, original_lines,
                    original_code=original_code, language=language,
                    max_lines=_MAX_BLOCK_LINES, padding=padding,
                )
                return [ChunkRegion(
                    narrowed[0], narrowed[1],
                    [replace],
                )]
            return [ChunkRegion(
                target_node.line_start, target_node.line_end,
                [replace],
            )]

    # If `after` is specified, find the named symbol and the next one,
    # create a chunk spanning the gap between them.  Supports 'Class.method'.
    if after:
        anchor_node = _resolve_symbol(after, ast_nodes)
        if anchor_node:
            # Find the next AST node after the anchor
            next_node = None
            for node in ast_nodes:
                if node.line_start > anchor_node.line_end and (next_node is None or node.line_start < next_node.line_start):
                    next_node = node
            if next_node:
                return [ChunkRegion(
                    anchor_node.line_start, next_node.line_end,
                    [after, next_node.name],
                )]
            else:
                # anchor is the last symbol — chunk from it to EOF
                return [ChunkRegion(
                    anchor_node.line_start, total_lines,
                    [after],
                )]

    # Find which AST nodes the snippet modifies
    matched_nodes = _find_matching_nodes(snippet, original_lines, ast_nodes, language)

    # Check for import changes
    import_region = None
    if _has_import_changes(snippet, original_code, language):
        import_region = _find_import_region(original_code, language, ast_nodes)

    # Check for new code insertion (snippet has definitions not in file)
    snippet_names = _extract_snippet_names(snippet, language)
    existing_names = {n.name for n in ast_nodes}
    has_new_defs = any(n not in existing_names for n in snippet_names)

    insertion = None
    if has_new_defs:
        insertion = _find_insertion_region(
            snippet, original_lines, ast_nodes, total_lines, language,
        )

    # If nothing matched at all, fall back to whole file
    if not matched_nodes and not import_region and not insertion:
        return [ChunkRegion(1, total_lines, ["<unmatched>"])]

    # Build regions from matches
    raw_regions: list[tuple[int, int]] = []
    region_names: dict[tuple[int, int], list[str]] = {}

    # Import region (if detected)
    if import_region:
        raw_regions.append(import_region)
        region_names[import_region] = ["<imports>"]

    # Insertion region for new definitions
    if insertion:
        ins_region = (insertion.start_line, insertion.end_line)
        raw_regions.append(ins_region)
        region_names[ins_region] = insertion.matched_nodes

    # Function/class regions — snap to enclosing class boundaries,
    # then narrow large nodes to sub-blocks
    for node in matched_nodes:
        parent = _find_enclosing_parent(node, ast_nodes)
        if parent:
            region = (parent.line_start, parent.line_end)
        else:
            region = _narrow_large_node(
                node, snippet, original_lines,
                original_code=original_code, language=language,
            )

        raw_regions.append(region)
        region_names.setdefault(region, []).append(node.name)

    # Merge overlapping/close regions
    merged_regions = _merge_overlapping_regions(raw_regions)

    chunks = []
    for start, end in merged_regions:
        # Collect names from all raw regions within this merged region
        names: list[str] = []
        for raw_region, raw_names in region_names.items():
            if raw_region[0] >= start and raw_region[1] <= end:
                names.extend(raw_names)
        chunks.append(ChunkRegion(start, end, names))

    return chunks


def _find_enclosing_block(
    source: str,
    language: str,
    target_line: int,
    function_start: int,
    function_end: int,
) -> tuple[int, int] | None:
    """Use tree-sitter to find the smallest block containing target_line.

    Walks the AST to find the tightest for/if/while/try/with block that
    encloses the target line within the given function range.

    Returns (start_line, end_line) 1-indexed, or None if no block found.
    """
    try:
        from ..data_gen.ast_analyzer import parse_code
        tree = parse_code(source, language)
    except Exception:  # noqa: BLE001 -- deliberate: parse of arbitrary source (unknown language/malformed) means "no enclosing block", never propagates
        return None

    # Collect ALL enclosing blocks, pick the best-sized one
    candidates: list[tuple[int, int]] = []

    def walk(node):
        node_start = node.start_point[0] + 1  # 1-indexed
        node_end = node.end_point[0] + 1

        # Skip nodes outside the function
        if node_end < function_start or node_start > function_end:
            return
        # Skip nodes that don't contain the target line
        if node_start > target_line or node_end < target_line:
            return

        if node.type in _BLOCK_TYPES:
            candidates.append((node_start, node_end))

        for child in node.children:
            walk(child)

    walk(tree.root_node)

    if not candidates:
        return None

    # Prefer blocks that are 30-100 lines. If none in that range,
    # pick the smallest block that's >= 20 lines.
    _MIN_BLOCK = 20
    _IDEAL_MIN = 30
    _IDEAL_MAX = 100

    ideal = [c for c in candidates if _IDEAL_MIN <= (c[1] - c[0]) <= _IDEAL_MAX]
    if ideal:
        return min(ideal, key=lambda c: c[1] - c[0])

    adequate = [c for c in candidates if (c[1] - c[0]) >= _MIN_BLOCK]
    if adequate:
        return min(adequate, key=lambda c: c[1] - c[0])

    # All blocks are tiny — return the largest one
    return max(candidates, key=lambda c: c[1] - c[0])


def _narrow_large_node(
    node: ASTNode,
    snippet: str,
    original_lines: list[str],
    original_code: str = "",
    language: str | None = None,
    max_lines: int = _MAX_BLOCK_LINES,
    padding: int = 10,
) -> tuple[int, int]:
    """Narrow a large function to the sub-block relevant to the snippet.

    Returns ``(start_line, end_line)`` (1-indexed, inclusive), always
    constrained to the node's own span: either the sub-block the snippet
    clearly targets, or the FULL node range when the narrow cannot be
    trusted (callers treat the full range as "send the whole node").

    B27 contract (Step 17): the narrow is trusted only when the snippet's
    non-marker lines match a contiguous window of the node with score
    ≥ ``_MIN_NARROW_SCORE`` AND tree-sitter can name the enclosing block
    around that window. The cut boundaries are then that block's AST node
    edges — never a ±``padding`` raw-line window, which cut mid-block and
    handed the model a broken fragment to "repair". Without a confident
    window match, without ``original_code``/``language`` to parse, or
    without an enclosing block, the narrow is REJECTED and the full node
    range is returned.

    ``padding`` is retained for API compatibility and is now a no-op:
    raw line-count padding was exactly the mid-block cut this function no
    longer performs (same deprecation pattern as ``deterministic_edit``'s
    ``max_drop_gap``).
    """
    node_size = node.line_end - node.line_start + 1
    if node_size <= max_lines:
        return (node.line_start, node.line_end)

    # Step 1: Find which line the snippet targets via line matching
    node_lines = original_lines[node.line_start - 1:node.line_end]
    # Filter out ellipsis markers — they don't represent real code and
    # would tank the match score when most snippet lines are markers.
    # Uses the shared LINE-ANCHORED predicate (single source of truth,
    # B15): a marker phrase embedded mid-line is real code, not a marker.
    snippet_lines = [
        line.rstrip() for line in snippet.splitlines()
        if line.strip() and not is_marker_line(line)
    ]

    if not snippet_lines:
        return (node.line_start, node.line_end)

    # Sliding window to find best match position
    best_score = 0.0
    best_offset = 0
    window = min(len(snippet_lines), 10)

    for offset in range(len(node_lines) - window + 1):
        region = [line.rstrip() for line in node_lines[offset:offset + window]]
        matches = sum(
            1 for s, r in zip(snippet_lines[:window], region, strict=False)
            if s.strip() == r.strip()
        )
        score = matches / window
        if score > best_score:
            best_score = score
            best_offset = offset

    if best_score < _MIN_NARROW_SCORE:
        # B27: a window the snippet barely overlaps is not evidence for
        # ANY location — no confident narrow, keep the whole node.
        return (node.line_start, node.line_end)

    target_line = node.line_start + best_offset

    # Step 2: cut at the enclosing block's AST node edges (B27). Without
    # something parsed there is no node edge to cut on — a raw-line window
    # here is precisely the mid-block cut B27 removes.
    if not (original_code and language):
        return (node.line_start, node.line_end)

    block = _find_enclosing_block(
        original_code, language, target_line,
        node.line_start, node.line_end,
    )
    if not block:
        # The matched window sits outside any for/if/while/try block; no
        # AST edge bounds a smaller cut, so keep the whole node.
        return (node.line_start, node.line_end)

    return (
        max(node.line_start, block[0]),
        min(node.line_end, block[1]),
    )


def _find_enclosing_parent(
    node: ASTNode,
    ast_nodes: list[ASTNode],
) -> ASTNode | None:
    """Find the smallest enclosing class/interface that contains the node.

    For large classes (>150 lines), returns None to keep method edits at
    method granularity instead of snapping to the whole class.
    """
    best = None
    for candidate in ast_nodes:
        if candidate is node:
            continue
        if candidate.kind not in ("class", "interface"):
            continue
        if (candidate.line_start <= node.line_start
                and candidate.line_end >= node.line_end
                and (best is None or (candidate.line_end - candidate.line_start <
                                      best.line_end - best.line_start))):
            # Pick the smallest enclosing parent
            best = candidate
    # Don't snap to large classes — keep method edits at method granularity
    if best is not None and (best.line_end - best.line_start) > 150:
        return None
    return best
