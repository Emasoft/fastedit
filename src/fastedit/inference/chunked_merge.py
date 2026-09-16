"""Chunked merge: AST-guided windowed editing for large files.

Instead of asking the model to rewrite an entire 1000+ line file,
we use tree-sitter AST analysis (via tldr) to locate the edit region,
extract a small chunk around it, merge just that chunk, and splice back.

The model only ever sees ~100-300 lines, which is its sweet spot.
Works across all 16 languages supported by tldr.
"""

from __future__ import annotations

import itertools
import re

from ..split_join import detect_line_ending, normalize_line_endings

# --- Re-export all public types and functions for backward compatibility ---
# All existing `from .inference.chunked_merge import X` imports continue to work.
from .ast_utils import (  # noqa: F401
    ASTNode,
    BatchEdit,
    ChunkedMergeResult,
    ChunkRegion,
    DeleteResult,
    MoveResult,
    _qualified_symbol_names,
    _resolve_symbol,
    get_ast_map,
    get_ast_map_from_source,
)
from .chunk_locator import (  # noqa: F401
    _find_enclosing_block,
    _find_enclosing_parent,
    _narrow_large_node,
    locate_chunks,
)
from .indent import (
    _align_snippet_indent,
    _escape_tags,
    _new_tag_nonce,
    _realign_output,
    _unescape_tags,
)
from .markers import (  # noqa: F401  -- re-exported for backward compat
    _MARKER_PHRASES,
    is_marker_line,
    normalize_markers,
)
from .snippet_analysis import (  # noqa: F401
    _DEFINITION_PATTERNS,
    _MARKER_RE,
    _extract_identifiers,
    _extract_snippet_names,
    _find_import_region,
    _find_insertion_region,
    _find_matching_nodes,
    _get_import_line_set,
    _get_snippet_definitions,
    _has_import_changes,
    _merge_overlapping_regions,
    _regex_extract_names,
    _split_snippet,
    _top_level_extras,
    _try_tldr_snippet_parse,
)
from .symbols import (  # noqa: F401
    batch_chunked_merge,
    delete_symbol,
    move_symbol,
)
from .text_match import (  # noqa: F401 -- deterministic_edit re-exported
    _indent_width,
    _replacement_key,
    deterministic_edit,
)

# Marker detection is unified in :mod:`fastedit.inference.markers`
# (single source of truth, B15). ``_MARKER_PHRASES`` is re-exported for
# backward compatibility; ``_is_marker_line``/``_normalize_markers`` alias
# the shared LINE-ANCHORED predicate/normalizer — marker phrases embedded
# mid-line are real content, not markers.
_is_marker_line = is_marker_line
_normalize_markers = normalize_markers


def _snippet_has_target_signature(snippet: str, target_name: str) -> bool:
    """Check if the snippet contains a definition line naming ``target_name``.

    A ``replace=<name>`` edit normally expects the snippet to start with the
    target's signature (so ``deterministic_edit`` can align lines). But the
    signature is redundant information — the ``replace=`` kwarg already
    identifies the target structurally. This helper detects the common case
    where the caller omitted the signature, so the caller can prepend it
    from the AST and keep the existing line-matching logic intact.

    Matches against all languages FastEdit supports via the shared
    ``_DEFINITION_PATTERNS`` regexes (def/fn/func/fun/class/struct/...).
    A match requires both the definition keyword AND the captured name
    equalling ``target_name`` — otherwise a snippet that references another
    `def` in a comment or docstring would spuriously match.

    Falls back to a broader class-keyword check that tolerates language
    modifiers the shared regex does not enumerate (Java/C#/TS ``public``,
    ``private``, ``final``, etc.). Comment-prefixed lines are skipped to
    avoid false positives.
    """
    for line in snippet.splitlines():
        stripped = line.lstrip()
        is_comment = stripped.startswith(("//", "#", "*", "--"))
        for pattern in _DEFINITION_PATTERNS:
            m = pattern.search(line)
            if m and m.group(1) == target_name:
                return True
        if not is_comment:
            m = re.search(
                r"\b(?:class|struct|enum|trait|interface|protocol|"
                r"module|object|impl)\s+(\w+)",
                line,
            )
            if m and m.group(1) == target_name:
                return True
    return False


def _check_hallucinations(
    original_chunk: str,
    merged_chunk: str,
    snippet: str,
    tokens: list[tuple[str, int | None, str | None]] | None = None,
) -> float:
    """Score merge quality: 1.0 = clean, 0.0 = hallucinated.

    The validator follows ``deterministic_edit``'s content-level marker
    semantics, but intentionally does not reproduce that editor's anchor
    selection heuristics. ``deterministic_edit`` uses ambiguous-line and raw
    indentation checks to synthesize one output; this validator operates on
    normalized line content and accepts any model output satisfying the
    invariants below. Marker syntax legitimately permits both preservation
    and a locally justified replacement.

    The snippet is forward-scanned into context anchors (lines matching an
    as-yet-unconsumed original line) and new lines, delimited by
    preservation markers. The anchors partition the original — and the
    anchors located in the merge partition the merge — into aligned
    segments; each segment is validated independently.

    ``tokens`` optionally supplies a PRE-COMPUTED snippet classification of
    the exact shape :func:`_classify_snippet` returns. The deterministic-
    path gate (:func:`_deterministic_result_is_faithful`) passes the
    editor's OWN binding here so the invariants are evaluated against the
    partition the editor actually used; a naive re-scan binds ambiguous
    structural lines (a mid-snippet lone ``}``) at a different original
    depth than the editor's ambiguous-anchor rule, producing phantom
    anchors that reject faithful editor output. ``None`` (the model path)
    classifies with :func:`_classify_snippet` exactly as before.

    A merge is rejected on any of:

      * **marker leakage** — a snippet placeholder echoed into the merge.
      * **anchor loss / reorder** — a declared context anchor missing from
        the merge or appearing out of order.
      * **invention / omission / duplication / new-line reorder** — the
        merge's new lines in a segment must equal the snippet's declared
        new lines for that segment exactly (order and multiplicity).
      * **unjustified deletion** — an original removed without a local
        one-to-one justification: a unique shared ``_replacement_key``
        identity, or — in a marker-bearing segment only — positional
        adjacency for an identity-free line on the marker-adjacent side.
        A keyed original may only be replaced by a same-key new line,
        never by positional proximity; ambiguous keys fail closed, and
        without a marker every identity-free deletion fails closed.
      * **preserved-line reorder / loss** — surviving originals must keep
        their relative order and multiplicity.
      * **indent unfaithfulness (B14)** — consecutive surviving originals
        must keep the indent delta they have in the original, measured on
        the raw indent-bearing lines the stripped comparison discards; a
        uniform shift of a whole group is legitimate, selective
        re-indentation of one survivor is not.
      * **marker side-order violation** — a snippet-new line declared
        before the first marker must precede every surviving original in
        its segment, and one declared after the last marker must follow
        every survivor. A new line teleported to the wrong side of the
        preserved gap is rejected even when nothing is deleted. New lines
        declared between two markers are position-ambiguous and impose no
        side constraint.

    ``_replacement_key`` (imported from :mod:`text_match`) is the single
    shared identity heuristic — the validator maintains no second
    language model. Returns ``1.0`` for a clean merge, ``0.0`` otherwise.
    """
    # Marker leak guard. Scan the raw merged chunk first so a leaked
    # marker cannot be hidden by the stripped-line comparison below.
    for raw_line in merged_chunk.splitlines():
        if raw_line.strip() and _is_marker_line(raw_line):
            return 0.0

    orig, orig_raw = _raw_content_lines(original_chunk)
    merged, merged_raw = _raw_content_lines(merged_chunk)
    if tokens is None:
        tokens = _classify_snippet(snippet, orig)

    return 1.0 if _merge_is_faithful(
        orig, merged, tokens, orig_raw, merged_raw,
    ) else 0.0


def _deterministic_result_is_faithful(
    original_func: str,
    edited: str,
    snippet: str,
    prepend_signature_lines: int = 0,
) -> bool:
    """Gate the deterministic splice on the shared content validator (B3).

    ``deterministic_edit`` output used to be spliced with ONLY a parse
    check, and a parse cannot see content corruption: a dropped original
    line (B1/B2), a leaked keep-marker or a selectively re-indented
    survivor all parse fine. This helper runs the same
    :func:`_check_hallucinations` validator the model path is held to and
    requires a fully clean score — the deterministic path must never
    bypass the validator.

    The inputs are span-local, exactly the values available at the splice
    site: the original target span (``original_func``), the editor's
    edited span (``edited``), the snippet that declared the edit
    (including any auto-prepended signature lines, so the validator sees
    the same snippet ``deterministic_edit`` consumed) and the number of
    prepended signature lines.

    The snippet is classified with the EDITOR'S OWN classifier
    (:func:`text_match._classify_edit_lines`) rather than the validator's
    simpler forward scan, and the resulting binding is handed to the
    validator via its ``tokens`` parameter. Classification asymmetry note
    (Step 6 triage): the editor's ambiguous-anchor rule treats a lone
    structural line mid-snippet (``}``, ``)``, ``end``, ...) as new
    content, while the validator's naive re-scan binds it to the first
    unconsumed original occurrence — often a deeper closer. That phantom
    anchor re-partitioned the original and made the preserved body lines
    look like unjustified deletions, rejecting FAITHFUL editor output
    (Rust guard-additions in tests/test_chained_edits_stale_ast.py and
    tests/test_replace_without_signature.py). Content faithfulness —
    deletions, inventions, reorders, marker leakage, indent deltas — is
    still enforced in full; only the ANCHOR SELECTION is taken from the
    editor, which is the component that owns that decision.

    Returns ``True`` only when the score is clean (``1.0``); anything else
    fails closed and the caller discards the deterministic result to the
    model path.
    """
    from .text_match import _AmbiguousAnchorBinding, _classify_edit_lines

    raw_lines = original_func.splitlines()
    try:
        classified = _classify_edit_lines(
            raw_lines, snippet.splitlines(), prepend_signature_lines,
        )
    except _AmbiguousAnchorBinding:
        # B24: the editor's classifier raises when two candidate anchor
        # bindings remain equally valid after sequence+indent
        # disambiguation. The gate fails closed for the same inputs, so a
        # deterministic result built on a guessed binding never reaches
        # the file. (In practice unreachable — the editor already declined
        # on this classification — but the gate must never guess either.)
        return False

    # Translate the editor's raw-line indices into the validator's
    # content-line index space (blanks and marker lines are skipped
    # there). A context anchor bound to a literal marker line in the
    # original is invisible to the validator's content view, so it is
    # skipped here too — mirroring what _classify_snippet could ever
    # bind.
    content_idx: dict[int, int] = {}
    ci = 0
    for ri, ln in enumerate(raw_lines):
        if not ln.strip() or _is_marker_line(ln):
            continue
        content_idx[ri] = ci
        ci += 1

    tokens: list[tuple[str, int | None, str | None]] = []
    for kind, _si, orig_idx, line in classified:
        if kind == "blank":
            continue
        if kind == "marker":
            tokens.append(("marker", None, None))
        elif kind == "context" and orig_idx in content_idx:
            tokens.append(("context", content_idx[orig_idx], line.strip()))
        else:
            tokens.append(("new", None, line.strip()))

    return _check_hallucinations(
        original_func, edited, snippet, tokens=tokens,
    ) == 1.0


def _classify_snippet(
    snippet: str,
    orig: list[str],
) -> list[tuple[str, int | None, str | None]]:
    """Forward-scan the snippet into ordered tokens.

    Each token is ``("context", orig_idx, line)``, ``("new", None, line)``
    or ``("marker", None, None)``. A non-marker snippet line is a *context*
    anchor when its normalized content matches an as-yet-unconsumed original
    line (scanning forward, so anchors are strictly increasing in
    ``orig_idx``); otherwise it is a *new* line.

    This is deliberately simpler than ``deterministic_edit``'s classifier.
    The editor has synthesis-specific rules for ambiguous structural lines
    and indentation deltas; the validator receives stripped content after
    output realignment and must not copy those rules blindly. Its job is to
    enforce content, multiplicity, and ordering invariants, not reconstruct
    the editor's single chosen output.
    """
    tokens: list[tuple[str, int | None, str | None]] = []
    cursor = 0
    for raw in snippet.splitlines():
        s = raw.strip()
        if not s:
            continue
        if _is_marker_line(raw):
            tokens.append(("marker", None, None))
            continue
        match_idx = None
        for i in range(cursor, len(orig)):
            if orig[i] == s:
                match_idx = i
                break
        if match_idx is not None:
            tokens.append(("context", match_idx, s))
            cursor = match_idx + 1
        else:
            tokens.append(("new", None, s))
    return tokens


def _merge_is_faithful(
    orig: list[str],
    merged: list[str],
    tokens: list[tuple[str, int | None, str | None]],
    orig_raw: list[str],
    merged_raw: list[str],
) -> bool:
    """Check the merge against the per-segment invariants.

    Context anchors must appear in the merge in order (else declared
    context was dropped or shuffled). The anchors partition the original
    and the merge into aligned leading, internal and trailing segments,
    each validated by :func:`_segment_is_faithful` together with the raw,
    indent-bearing lines behind its stripped content (B14).

    Anchors are located in the merge through the GLOBAL survivor alignment
    (:func:`_lcs_pair_map`, the same LCS the per-segment survivor check
    uses) rather than a first-occurrence scan. When an anchor's value
    occurs more than once on either side — a lone ``}`` is both the guard
    closer a deterministic edit emitted AND a preserved closer — the
    first-occurrence scan binds the anchor to the wrong occurrence and
    mis-partitions both sides, pushing real survivors into the wrong
    segment where they look like unjustified deletions. The maximal
    alignment pairs every surviving original with its true merge partner;
    an anchor the alignment did not pair (its line did not survive) falls
    back to the scan, which rejects when the line is genuinely missing.
    """
    anchors = [t[1] for t in tokens if t[0] == "context"]

    lcs_pairs = _lcs_pair_map(orig, merged)
    merged_pos: list[int] = []
    mcursor = 0
    for a in anchors:
        # Prefer the global survivor pairing; fall back to the first
        # occurrence at or after the previous anchor for a non-survivor.
        found = lcs_pairs.get(a)
        if found is None:
            val = orig[a]
            for j in range(mcursor, len(merged)):
                if merged[j] == val:
                    found = j
                    break
        if found is None:
            return False
        merged_pos.append(found)
        mcursor = found + 1

    for (
        orig_seg, merged_seg, seg_tokens, orig_seg_raw, merged_seg_raw,
    ) in _build_segments(orig, merged, tokens, anchors, merged_pos, orig_raw, merged_raw):
        if not _segment_is_faithful(
            orig_seg, merged_seg, seg_tokens, orig_seg_raw, merged_seg_raw,
        ):
            return False
    return True


def _build_segments(
    orig: list[str],
    merged: list[str],
    tokens: list[tuple[str, int | None, str | None]],
    anchors: list[int],
    merged_pos: list[int],
    orig_raw: list[str],
    merged_raw: list[str],
) -> list[
    tuple[
        list[str],
        list[str],
        list[tuple[str, int | None, str | None]],
        list[str],
        list[str],
    ]
]:
    """Slice orig, merged and the snippet tokens into aligned segments.

    Returns ``(orig_seg, merged_seg, seg_tokens, orig_seg_raw,
    merged_seg_raw)`` per segment, where ``seg_tokens`` are the non-context
    tokens (new lines and markers) declared in that segment and the raw
    lists carry the indent-bearing source lines behind the segment's
    stripped content, index-aligned with it (B14). The leading segment
    precedes the first anchor, internal segments run between consecutive
    anchors and the trailing segment follows the last anchor; with no
    anchors there is one whole-span segment.
    """
    # Group non-context tokens by the anchor they follow: rank -1 means
    # "before the first anchor", rank k means "after anchor k".
    groups: dict[int, list[tuple[str, int | None, str | None]]] = {}
    anchor_rank = -1
    for t in tokens:
        if t[0] == "context":
            anchor_rank += 1
        else:
            groups.setdefault(anchor_rank, []).append(t)

    if not anchors:
        return [(orig, merged, groups.get(-1, []), orig_raw, merged_raw)]

    segments: list[
        tuple[
            list[str],
            list[str],
            list[tuple[str, int | None, str | None]],
            list[str],
            list[str],
        ]
    ] = []
    segments.append((
        orig[:anchors[0]],
        merged[:merged_pos[0]],
        groups.get(-1, []),
        orig_raw[:anchors[0]],
        merged_raw[:merged_pos[0]],
    ))
    for k in range(len(anchors) - 1):
        segments.append((
            orig[anchors[k] + 1:anchors[k + 1]],
            merged[merged_pos[k] + 1:merged_pos[k + 1]],
            groups.get(k, []),
            orig_raw[anchors[k] + 1:anchors[k + 1]],
            merged_raw[merged_pos[k] + 1:merged_pos[k + 1]],
        ))
    segments.append((
        orig[anchors[-1] + 1:],
        merged[merged_pos[-1] + 1:],
        groups.get(len(anchors) - 1, []),
        orig_raw[anchors[-1] + 1:],
        merged_raw[merged_pos[-1] + 1:],
    ))
    return segments


def _segment_is_faithful(
    orig_seg: list[str],
    merged_seg: list[str],
    seg_tokens: list[tuple[str, int | None, str | None]],
    orig_seg_raw: list[str],
    merged_seg_raw: list[str],
) -> bool:
    """Validate one segment against the preserve-by-default invariants.

    EVERY segment is protected (B4): a segment is an insertion zone, never
    a replacement zone. Whether it leads the span, trails it, fills a
    marker-less mid gap or a marker-bearing one, the originals it brackets
    survive by default and the snippet's declared new lines are INSERTIONS
    into them — the merge may not overwrite the segment with its new
    lines.

    A segment is faithful when all of the following hold:

      * **invention / omission / duplication / new-line reorder** — the
        merge's unmatched lines equal the declared new lines for the
        segment exactly (order and multiplicity);
      * **indentation faithfulness (B14)** — consecutive surviving
        originals keep the indent DELTA they have in the original,
        computed on the raw indent-bearing lines (see
        :func:`_indent_deltas_preserved`); a uniform shift of a whole
        group (block wrapping) is accepted, selective re-indentation of a
        single survivor is not;
      * **marker side-order** (marker-bearing segments) — a declared new
        line keeps the side of the preserved gap the snippet gave it (see
        :func:`_new_side_order_ok`);
      * **justified deletion** — every original the merge drops is
        justified (see :func:`_deletions_justified`): a unique shared
        ``_replacement_key`` identity, or — only when the segment actually
        carries a marker — marker-adjacent positional adjacency for an
        identity-free line. With no marker, identity-free deletions fail
        closed.
    """
    new_all = [t[2] for t in seg_tokens if t[0] == "new"]
    has_marker = any(t[0] == "marker" for t in seg_tokens)

    # Survivors are the originals kept (longest common subsequence on
    # stripped content); the remaining merged lines are new and must equal
    # the declared new lines exactly.
    kept_orig, kept_merged = _lcs_matched(orig_seg, merged_seg)
    new_in_merged = [
        merged_seg[j] for j in range(len(merged_seg)) if j not in kept_merged
    ]
    if new_in_merged != new_all:
        return False

    # B14: surviving originals keep their relative indentation. Checked on
    # the raw lines — the stripped comparison above cannot see indent
    # corruption.
    if not _indent_deltas_preserved(
        orig_seg_raw, kept_orig, merged_seg_raw, kept_merged,
    ):
        return False

    # Marker side-order invariant: a snippet-new line declared before the
    # first marker must precede every surviving original in the merge, and a
    # new line declared after the last marker must follow every survivor.
    # Checked before the deletion justification below because a wrong-side
    # new line can violate the invariant with no deletion at all (the
    # survivor multiset is intact — only the relative placement is corrupt).
    if has_marker and not _new_side_order_ok(seg_tokens, merged_seg, kept_merged):
        return False

    deleted = [i for i in range(len(orig_seg)) if i not in kept_orig]
    if not deleted:
        return True

    new_before: list[str] = []
    new_after: list[str] = []
    if has_marker:
        marker_positions = [i for i, t in enumerate(seg_tokens) if t[0] == "marker"]
        first_marker, last_marker = marker_positions[0], marker_positions[-1]
        new_before = [
            t[2] for i, t in enumerate(seg_tokens)
            if t[0] == "new" and i < first_marker
        ]
        new_after = [
            t[2] for i, t in enumerate(seg_tokens)
            if t[0] == "new" and i > last_marker
        ]
    return _deletions_justified(
        orig_seg, deleted, new_all, new_before, new_after,
        allow_positional=has_marker,
    )


def _indent_deltas_preserved(
    orig_seg_raw: list[str],
    kept_orig: set[int],
    merged_seg_raw: list[str],
    kept_merged: set[int],
) -> bool:
    """B14: consecutive surviving originals keep their relative indentation.

    The LCS pairs survivors on STRIPPED content, so a merge can keep every
    line's content while selectively re-indenting it — flattening one
    line's nesting under an ``if`` it no longer sits inside, say. For each
    pair of CONSECUTIVE surviving originals the indent delta between their
    raw lines must equal the delta between their merge counterparts.
    Widths come from ``_indent_width`` (tab-aware, ``expandtabs(4)``),
    reused from :mod:`text_match` — no second indent model.

    Blank/whitespace-only lines never enter the content view, so a
    survivor's neighbour here is its nearest non-blank line. A UNIFORM
    shift of a whole group changes every pairwise delta by the same
    amount and is accepted; moving a single line is not.

    The k-th original survivor pairs with the k-th merge survivor: the LCS
    traceback yields strictly increasing index pairs on both sides, so
    sorting each set and zipping reproduces the alignment.
    """
    pairs = zip(sorted(kept_orig), sorted(kept_merged))
    for (i1, j1), (i2, j2) in itertools.pairwise(pairs):
        orig_delta = (
            _indent_width(orig_seg_raw[i2]) - _indent_width(orig_seg_raw[i1])
        )
        merged_delta = (
            _indent_width(merged_seg_raw[j2]) - _indent_width(merged_seg_raw[j1])
        )
        if orig_delta != merged_delta:
            return False
    return True


def _deletions_justified(
    orig_seg: list[str],
    deleted: list[int],
    new_all: list[str],
    new_before: list[str],
    new_after: list[str],
    allow_positional: bool = False,
) -> bool:
    """Decide whether every deleted protected original is justified.

    A *keyed* original (``_replacement_key`` is not ``None``) may be deleted
    only when exactly one original and exactly one declared new line share
    its key — a unique shared identity. Any other keyed deletion fails
    closed. An *identity-free* original may be deleted by positional
    fallback only, when it sits contiguously against a marker boundary that
    carries an identity-free new line (front for lines declared before the
    marker, back for lines declared after) — never an arbitrary bystander.

    The positional fallback requires ``allow_positional``: it exists only
    where a marker defines the boundary it leans on. A segment without a
    marker has no such boundary, so EVERY identity-free deletion in it
    fails closed regardless of the declared new lines (defaults to False
    — fail closed).
    """
    n = len(orig_seg)
    deleted_set = set(deleted)

    orig_keys = [_replacement_key(line) for line in orig_seg]
    orig_key_count: dict[str, int] = {}
    for k in orig_keys:
        if k is not None:
            orig_key_count[k] = orig_key_count.get(k, 0) + 1
    new_key_count: dict[str, int] = {}
    for line in new_all:
        k = _replacement_key(line)
        if k is not None:
            new_key_count[k] = new_key_count.get(k, 0) + 1

    positional: list[int] = []
    for i in deleted:
        k = orig_keys[i]
        if k is not None:
            # Keyed deletion: require a unique shared identity on both sides.
            if orig_key_count.get(k, 0) == 1 and new_key_count.get(k, 0) == 1:
                continue
            return False
        positional.append(i)

    if not positional:
        return True

    if not allow_positional:
        # No marker in this segment — no marker-adjacent boundary exists,
        # so an identity-free deletion has no local justification.
        return False

    # Positional fallback capacity comes from identity-free new lines on
    # each side; deletions must form a contiguous run against that boundary.
    front_cap = sum(1 for line in new_before if _replacement_key(line) is None)
    back_cap = sum(1 for line in new_after if _replacement_key(line) is None)

    front_run = 0
    while front_run < n and front_run in deleted_set:
        front_run += 1
    back_run = 0
    while back_run < n and (n - 1 - back_run) in deleted_set:
        back_run += 1
    front_cover = min(front_run, front_cap)
    back_cover = min(back_run, back_cap)

    for i in positional:
        if i < front_cover or i >= n - back_cover:
            continue
        return False
    return True


def _new_side_order_ok(
    seg_tokens: list[tuple[str, int | None, str | None]],
    merged_seg: list[str],
    kept_merged: set[int],
) -> bool:
    """Check the marker side-order invariant for one protected segment.

    A snippet-new line declared *before the first marker* is a top-of-gap
    insertion — it must occur before every surviving original in the merge.
    A new line declared *after the last marker* is a bottom-of-gap
    insertion — it must occur after every survivor. A new line declared
    *between* two markers is position-ambiguous and imposes no constraint;
    the caller's invention/omission/deletion checks still apply to it.

    The k-th unmatched merged occurrence (in merge order) corresponds to the
    k-th declared new line, because the caller has already verified the
    merged new lines equal the declared new lines in order and multiplicity.
    Enforcing the constraint on the concrete merged indices from the LCS
    alignment — not on line values or global counts — keeps repeated
    survivors and repeated new lines correctly positioned, so an LCS tie
    cannot hide a wrong-side new occurrence.

    Returns ``True`` when the segment carries no marker or no surviving
    original (the constraint is then vacuous — deletion justification still
    governs acceptance upstream).
    """
    marker_positions = [i for i, t in enumerate(seg_tokens) if t[0] == "marker"]
    if not marker_positions:
        return True
    first_marker, last_marker = marker_positions[0], marker_positions[-1]

    # Side of each declared new line, in declared order. This list is
    # parallel to the unmatched merged occurrences (same length, same order).
    new_sides: list[str] = []
    for i, t in enumerate(seg_tokens):
        if t[0] != "new":
            continue
        if i < first_marker:
            new_sides.append("before")
        elif i > last_marker:
            new_sides.append("after")
        else:
            new_sides.append("between")

    survivors = sorted(kept_merged)
    if not survivors:
        return True
    first_survivor, last_survivor = survivors[0], survivors[-1]

    unmatched = [j for j in range(len(merged_seg)) if j not in kept_merged]
    for side, j in zip(new_sides, unmatched, strict=True):
        if side == "before" and j > first_survivor:
            return False
        if side == "after" and j < last_survivor:
            return False
    return True


def _lcs_pair_map(
    a: list[str],
    b: list[str],
) -> dict[int, int]:
    """Longest-common-subsequence pairing of two line lists.

    Returns ``{a_index: b_index}`` for one maximal alignment — strictly
    increasing on both sides, so the k-th surviving original maps to the
    k-th surviving merge line. This is the canonical survivor alignment:
    :func:`_merge_is_faithful` uses it to locate context anchors in the
    merge, :func:`_lcs_matched` derives its survivor sets from it.
    """
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return {}
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la - 1, -1, -1):
        for j in range(lb - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    pairs: dict[int, int] = {}
    i = j = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            pairs[i] = j
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _lcs_matched(
    a: list[str],
    b: list[str],
) -> tuple[set[int], set[int]]:
    """Longest-common-subsequence alignment of two line lists.

    Returns ``(matched_a, matched_b)`` — the index sets paired by an LCS.
    ``matched_a`` identifies the originals that survived; the unmatched
    ``b`` indices are the merge's new lines. Order and multiplicity of
    repeated lines are respected (this is a subsequence match, not a set).
    """
    pairs = _lcs_pair_map(a, b)
    return set(pairs), set(pairs.values())


def _raw_content_lines(s: str) -> tuple[list[str], list[str]]:
    """Content lines paired with the raw, indent-bearing lines behind them.

    Returns ``(content, raw)`` with ``content[i]`` the stripped form of
    ``raw[i]``. Blank/whitespace-only lines and keep-marker lines carry no
    comparable content and are skipped from BOTH views, keeping the lists
    index-aligned. The raw view exists for the B14 indentation check,
    which must see the leading whitespace the stripped content view
    deliberately discards.
    """
    content: list[str] = []
    raw: list[str] = []
    for line in s.splitlines():
        if not line.strip() or _is_marker_line(line):
            continue
        content.append(line.strip())
        raw.append(line)
    return content, raw


def _real_lines(s: str) -> list[str]:
    """Stripped, non-blank, non-marker lines — the lines that carry
    real content for the diff."""
    return _raw_content_lines(s)[0]


def _whole_file_rejection_reason(
    original_code: str,
    merged_code: str,
    snippet: str,
    language: str | None,
    result_truncated: bool,
) -> str | None:
    """Decide whether a whole-file merge output is usable (B13/B29, Step 11).

    Returns ``None`` when the output may be returned as the merge, else a
    human-readable reason string. The whole-file branch hands the ENTIRE
    file to the model, so its output must pass three gates before it may
    be returned — in order:

      1. the engine's truncation flag (B12) — a length-capped response is
         a partial file even when its payload looks complete;
      2. parse validity — required whenever the language is known, exactly
         like every other return path;
      3. the shared content validator (:func:`_check_hallucinations`) run
         over the FULL original vs the FULL merge output with the full
         snippet — the same preserve-by-default contract the per-chunk
         path is already held to (a dropped unmentioned line, an
         invention, a reorder, an unjustified deletion or a leaked
         preservation marker all fail).
    """
    if result_truncated:
        return "truncated (model hit the token cap)"
    if language:
        from ..data_gen.ast_analyzer import validate_parse
        if not validate_parse(merged_code, language):
            return f"merged output does not parse as {language}"
    if _check_hallucinations(original_code, merged_code, snippet) != 1.0:
        return (
            "merged output failed the content-faithfulness check "
            "(preserve-by-default violation: original lines dropped, "
            "invented, reordered or a marker leaked)"
        )
    return None


def _append_corrective_note(snippet: str, reason: str) -> str:
    """Append a corrective note to the whole-file retry prompt (Step 11).

    ``merge_fn`` takes only ``(original, snippet, language)``, so the
    snippet is the one channel a corrective note can ride on; the prompt
    template embeds it inside ``<update>``, where the model reads it as
    instruction. The note is plain prose — no marker syntax (a marker line
    would change the snippet's segment structure and could itself be
    echoed into a leak) and no language-specific comment prefix (no
    per-language branches). The retry is validated against the ORIGINAL
    snippet, so the note can never contaminate the gate: an attempt that
    echoes the note into code simply fails the validator.
    """
    note = (
        "NOTE: the previous merge attempt was rejected — "
        f"{reason}. Return the COMPLETE file: preserve every original "
        "line the snippet does not explicitly change, in order, and do "
        "not drop, reorder, summarize or invent code."
    )
    return f"{snippet}\n\n{note}"


# ---------------------------------------------------------------------------
# Core merge function — the only logic that remains in this file
# ---------------------------------------------------------------------------

def _original_line_endings_are_uniform(original_code: str) -> bool:
    """True when every line ending in ``original_code`` is the same style.

    Counted directly (never via ``splitlines``, which also breaks on other
    Unicode characters). A file is uniform when:

      * it carries CRLF pairs and NOTHING else — every ``\\r`` and every
        ``\\n`` belongs to a pair; or
      * it carries exactly one of lone-CR or bare-LF (or no endings at all).

    A MIXED file (``\\r\\n`` alongside bare ``\\n`` or lone ``\\r``) has no
    single convention to enforce, so the central normalizer must NOT rewrite
    its endings wholesale: the untouched regions' endings are the user's
    bytes, and converting them to the dominant style would corrupt content
    the edit never came near. For such files only the trailing-newline state
    is enforced (see :func:`_normalize_merged_eol`).
    """
    crlf = original_code.count("\r\n")
    cr = original_code.count("\r")
    lf = original_code.count("\n")
    if crlf:
        return crlf == cr and crlf == lf
    return not (cr and lf)


def _normalize_merged_eol(merged_code: str, original_code: str) -> str:
    """Central EOL + trailing-newline funnel for every merge return path.

    Step 14 (B19, B20 remainder, B31, B41). Reuses the shared
    :func:`split_join.detect_line_ending` / ``normalize_line_endings``
    machinery — this helper owns the POLICY, not a second EOL model:

      * **Line-ending convention (B19/B20).** When the original has one
        line-ending convention (:func:`_original_line_endings_are_uniform`),
        every bare-LF piece a splice or model path produced is converted to
        the original's ending — untouched regions already carry that
        ending, so the rewrite is a no-op on them and a repair on the
        produced pieces. A produced CRLF piece inside an LF file is
        converted back the same way (the mirror direction of the same bug).
      * **Mixed-ending originals.** No convention to enforce: endings pass
        through untouched (byte-exactness beats guessing). Only the trailing
        rule below applies.
      * **Trailing-newline state (B31).** Taken from the ORIGINAL, never
        from a ``"\\n"`` constant at a splice site: an original without a
        trailing terminator yields a merged file without one (any appended
        terminator is stripped), and an original with one yields a merged
        file that ends with exactly one (a produced span that lost the
        file's terminator at EOF gets it restored, in the original's own
        ending).

    ``parse_valid`` MUST be computed on this function's output — callers
    funnel BEFORE validating so every parse/validate gate sees the final
    bytes that would be written.

    Passing ``merged_code == original_code`` (the rejection convention:
    the original file is kept as merged_code) returns the original
    verbatim — the funnel is an exact no-op on its own input.
    """
    if merged_code == original_code:
        return merged_code
    line_ending = detect_line_ending(original_code)
    if _original_line_endings_are_uniform(original_code):
        merged_code = normalize_line_endings(merged_code, line_ending)
    if original_code.endswith(("\n", "\r")):
        if merged_code and not merged_code.endswith(("\n", "\r")):
            merged_code += line_ending
    elif merged_code.endswith(("\n", "\r")):
        merged_code = merged_code.rstrip("\r\n")
    return merged_code


def _merge_preserve_siblings(
    original_code: str,
    snippet: str,
    file_path: str,
    replace: str,
    language: str | None,
) -> ChunkedMergeResult:
    """Replace `replace` (a class) with `snippet`, carrying over any named
    children of the original class that don't appear in the snippet.

    Pure-AST, zero-token operation. Used when the caller passes
    `preserve_siblings=True` on a `replace=` edit.

    The merged output contains:
      - the snippet's class shell and any members it defines
      - followed by every sibling (method, nested class, ...) present in
        the original class but missing from the snippet, spliced verbatim
        before the closing brace of the snippet.

    Sibling boundaries come from tldr's structure/extract output. Field
    declarations aren't exposed as separate nodes by tldr in Java/Kotlin/
    Swift/TS, which is why the field change lives in the snippet itself.
    """
    import logging

    from ..split_join import detect_line_ending, normalize_line_endings

    _log = logging.getLogger("fastedit.chunked_merge")

    original_lines = original_code.splitlines(keepends=True)
    total_lines = len(original_lines)

    # The file's own prevailing ending must win over the snippet's --
    # inserted/edited spans are normalized to it, untouched original
    # lines are never touched (TRDD-CMRMA2YG line-ending fix).
    line_ending = detect_line_ending(original_code)

    # Race-free in-memory AST parse. `original_code` may differ from what
    # the tldr daemon has cached on disk; parsing in-memory is the only
    # way to guarantee correct line coordinates under chained edits.
    ast_nodes = get_ast_map_from_source(original_code, file_path) or []
    target = _resolve_symbol(replace, ast_nodes)
    if target is None:
        available = _qualified_symbol_names(ast_nodes)
        raise ValueError(
            f"Symbol '{replace}' not found in {file_path}. "
            f"Available: {available}"
        )

    class_start = target.line_start  # 1-indexed
    class_end = target.line_end      # 1-indexed inclusive

    # Collect named children of the class: any AST node strictly within
    # the class's line span, excluding the class itself.
    child_nodes = [
        n for n in ast_nodes
        if n.name != replace
        and n.line_start >= class_start
        and n.line_end <= class_end
    ]

    # Parse the snippet to discover which children it names.
    snippet_child_names: set[str] = set()
    snippet_nodes = _get_snippet_definitions(snippet, language)
    for n in snippet_nodes:
        if n.name != replace:
            snippet_child_names.add(n.name)

    # Missing = children in original but not mentioned by the snippet.
    missing = [n for n in child_nodes if n.name not in snippet_child_names]

    # Re-indent the snippet so its class header matches the original's
    # class header indent level.
    class_first_line = original_lines[class_start - 1] if class_start - 1 < total_lines else ""
    indent = class_first_line[: len(class_first_line) - len(class_first_line.lstrip())]

    snippet_lines = normalize_line_endings(snippet, line_ending).splitlines(keepends=True)
    snippet_first_nonblank = next(
        (ln for ln in snippet_lines if ln.strip()), class_first_line
    )
    snippet_indent = snippet_first_nonblank[
        : len(snippet_first_nonblank) - len(snippet_first_nonblank.lstrip())
    ]
    if snippet_indent != indent:
        new_snippet_lines: list[str] = []
        for line in snippet_lines:
            if line.strip() and line.startswith(snippet_indent):
                new_snippet_lines.append(indent + line[len(snippet_indent):])
            else:
                new_snippet_lines.append(line)
        snippet_lines = new_snippet_lines

    # Strip ellipsis marker lines from the snippet — preserve_siblings
    # subsumes their role.
    snippet_lines = [ln for ln in snippet_lines if not _MARKER_RE.match(ln)]

    # Build preserved blocks from the original (in original source order).
    preserved_blocks: list[str] = []
    for child in sorted(missing, key=lambda n: n.line_start):
        cstart = child.line_start - 1  # 0-indexed
        cend = child.line_end          # exclusive
        preserved_blocks.append("".join(original_lines[cstart:cend]))

    # Locate the closing brace line of the snippet's class body. We
    # assume languages with `{ ... }` class syntax (Java/Kotlin/Swift/TS
    # all qualify). Scan from the end for the first line that is exactly
    # `}` (after whitespace).
    close_idx: int | None = None
    for i in range(len(snippet_lines) - 1, -1, -1):
        if snippet_lines[i].strip() == "}":
            close_idx = i
            break

    if close_idx is None:
        raise ValueError(
            "preserve_siblings=True requires a class body with a `}` closing "
            "brace in the snippet — no such line found. This path currently "
            "supports Java/Kotlin/Swift/TypeScript (and similar brace-delimited "
            "languages)."
        )

    assembled: list[str] = []
    assembled.extend(snippet_lines[:close_idx])
    # Blank-line separator before preserved siblings if the snippet doesn't
    # already end with a blank line. Uses the file's own line ending so the
    # splice does not leave a mixed-ending seam (TRDD-CMRMA2YG).
    if preserved_blocks and assembled and assembled[-1].strip() != "":
        assembled.append(line_ending)
    for i, block in enumerate(preserved_blocks):
        assembled.append(block)
        # Blank-line separator between preserved siblings (not after last).
        if i < len(preserved_blocks) - 1 and not block.endswith(("\n\n", "\r\n\r\n")):
            assembled.append(line_ending)
    assembled.extend(snippet_lines[close_idx:])

    # Splice the assembled class body back into the original file.
    result_lines = list(original_lines)
    result_lines[class_start - 1:class_end] = assembled
    merged = "".join(result_lines)
    # Step 14 (B19/B31): the snippet's re-indented lines and the preserved
    # blocks are funnelled through the central normalizer so the emitted
    # endings follow the original's convention and the file's trailing-
    # newline state is the original's, not the snippet's.
    merged = _normalize_merged_eol(merged, original_code)

    parse_valid = True
    if language:
        from ..data_gen.ast_analyzer import validate_parse
        parse_valid = validate_parse(merged, language)
        if not parse_valid:
            _log.warning(
                "preserve_siblings produced parse-invalid output for "
                "replace='%s' in %s (language=%s)",
                replace, file_path, language,
            )

    _log.info(
        "preserve_siblings for replace='%s' (L%d-L%d): "
        "%d preserved sibling(s), 0 model tokens",
        replace, class_start, class_end, len(preserved_blocks),
    )
    return ChunkedMergeResult(
        merged_code=merged,
        parse_valid=parse_valid,
        chunks_used=0,
        chunk_regions=[],
        model_tokens=0,
        latency_ms=0.0,
        chunks_rejected=0,
    )



def _extract_signature_via_ast(
    source: str,
    language: str | None,
    target_start_line: int,
    target_end_line: int,
    fallback_line: str,
) -> str:
    """Return the full signature of a function/method/class, handling
    multi-line parameter lists.

    Uses tree-sitter's ``body`` field to find where the body begins; the
    signature is everything from the node's start byte up to (not including)
    the body's start byte. This correctly captures multi-line ``def foo(\n    a,\n    b,\n):``
    shapes that the prior heuristic (``original_lines[func_start]``) truncated to
    ``def foo(``, silently producing unclosed parens in the merged output.

    When the grammar places the body-opening delimiter at the start of the
    body node, the span is extended past it so the extracted signature ends
    AFTER the opener (B9). tree-sitter materializes such an opener as an
    anonymous child token at the body's own start byte — ``{`` for the
    C-family/Rust/TS/Go/Swift/Java class of grammars — so detecting it needs
    no language keyword list and no per-language branch: the check is purely
    structural (first body child, anonymous, starts exactly at the body,
    text is a single opening delimiter). Grammars whose opener is part of
    the signature line itself (Python's ``:``, a direct child of the
    function node) or that have no opener token (Ruby) need no extension:
    the colon is already inside the byte span, and prepending Ruby's
    brace-less signature is correct. A grammar that wraps the opener in an
    extra named node (Kotlin's ``function_body``) is a known limit — its
    span still ends before the brace, exactly as before this fix.

    Falls back to ``fallback_line`` when:
    - language is None or unsupported
    - tree-sitter parsing fails
    - no function-like node matches the target byte range
    - the matching node has no ``body`` field (unusual grammar)
    """
    if not language:
        return fallback_line
    try:
        from ..data_gen.ast_analyzer import parse_code
        tree = parse_code(source, language)
    except Exception:  # noqa: BLE001 -- deliberate: docstring lists "tree-sitter parsing fails" as a fallback trigger; degrade to fallback_line, never propagate
        return fallback_line

    src_bytes = source.encode("utf-8")
    # tree-sitter rows are 0-indexed; ASTNode.line_start is 1-indexed.
    target_row = target_start_line - 1

    def walk(node):
        # Match any function/method/class-like node that starts on the
        # target row AND has a body field with a later start byte.
        if node.start_point[0] == target_row:
            # Most grammars expose the body via a named field called "body"
            # (Python, Rust, Go, JS/TS, Java, C/C++, Ruby, Swift, PHP, C#).
            body = node.child_by_field_name("body")
            if body is None:
                # Kotlin uses an unnamed "function_body" child; Elixir parses
                # `def foo(a) do ... end` as a call with a "do_block" child.
                for child in node.children:
                    if child.type in ("function_body", "do_block"):
                        body = child
                        break
            if body and body.start_byte > node.start_byte:
                end_byte = body.start_byte
                # B9: include the body-OPENING delimiter so the prepended
                # signature never yields a brace-less splice. The grammar
                # places the opener as an anonymous child token starting
                # exactly at the body's start byte (``{`` for C-family/
                # Rust/TS/Go/Swift/Java); structural detection only — no
                # language keyword lists. A named first child at the same
                # byte (Python's block starts at the first statement) or an
                # opener the grammar folds into the signature line (Python's
                # ``:``, already inside the span) needs no extension.
                if body.child_count:
                    first = body.children[0]
                    if (
                        first.start_byte == body.start_byte
                        and not first.is_named
                        and src_bytes[first.start_byte:first.end_byte]
                        in (b"{", b"(", b"[")
                    ):
                        end_byte = first.end_byte
                return src_bytes[node.start_byte:end_byte].decode(
                    "utf-8", errors="replace",
                )
        for child in node.children:
            # Skip branches that can't contain the target
            if child.start_point[0] > target_end_line - 1:
                break
            if child.end_point[0] < target_row:
                continue
            result = walk(child)
            if result is not None:
                return result
        return None

    sig = walk(tree.root_node)
    if sig is None:
        return fallback_line
    # Trim trailing whitespace/newlines, then re-terminate with a single \n
    # so the prepended signature forms exactly one well-formed line prefix.
    return sig.rstrip() + "\n"


def chunked_merge(
    original_code: str,
    snippet: str,
    file_path: str,
    merge_fn,
    language: str | None = None,
    padding: int = 30,
    after: str | None = None,
    replace: str | None = None,
    preserve_siblings: bool = False,
) -> ChunkedMergeResult:
    """Merge a snippet into a large file using chunked extraction.

    Args:
        original_code: Full original file content.
        snippet: The edit snippet.
        file_path: Path to the file (for AST extraction).
        merge_fn: Callable(original_chunk, snippet, language) -> MergeResult.
        language: Optional language for validation.
        padding: Lines of context padding around edit regions.
        after: Optional symbol name — insert new code after this function/class.
        replace: Optional symbol name — replace this function/class/method entirely.
        preserve_siblings: When True alongside `replace=ClassName`, carry over
            any named sibling members (methods, nested classes) that exist in
            the original class but aren't mentioned in the snippet. Lets you
            edit a subset of a class's members without enumerating the rest.
            Only valid with `replace=`; raises ValueError otherwise.

    Returns:
        ChunkedMergeResult with the fully merged file.
    """
    # preserve_siblings is only meaningful on a `replace=` edit. Fail
    # early and loudly so callers don't silently no-op.
    if preserve_siblings and not replace:
        raise ValueError(
            "preserve_siblings=True requires replace=ClassName. "
            "The flag controls how `replace=` behaves when the snippet "
            "describes only a subset of the class's members."
        )

    # Normalize short / Unicode marker forms (v0.2.4) → canonical long
    # form. All downstream code (chunk_locator, text_match, model paths,
    # snippet_analysis) continues to see ``# ... existing code ...`` /
    # ``// ... existing code ...``; no other module needs to know about
    # the short forms. See ``normalize_markers`` docstring. Imported from
    # the shared markers module (single source of truth, B15).
    snippet = _normalize_markers(snippet)

    # Step 14 (B19/B20/B31): the file's own line-ending convention is
    # detected ONCE, here, and reused by every splice site below; the file's
    # trailing-newline state is enforced ONCE per return path by
    # _normalize_merged_eol, so no splice site decides it with a hardcoded
    # "\n".
    line_ending = detect_line_ending(original_code)

    if preserve_siblings and replace:
        return _merge_preserve_siblings(
            original_code=original_code,
            snippet=snippet,
            file_path=file_path,
            replace=replace,
            language=language,
        )

    original_lines = original_code.splitlines(keepends=True)
    total_lines = len(original_lines)

    import logging
    _log = logging.getLogger("fastedit.chunked_merge")

    # Fast path: `after` means pure text insertion — no model needed.
    # The snippet IS the new code; just splice it after the anchor symbol.
    if after:
        # B16: this path splices the snippet VERBATIM, so a preservation
        # marker inside it would be written into the file as a literal
        # comment line — silent corruption (it parses as a comment, so no
        # downstream check ever objects). Markers are merge directives,
        # never content: rewrite short/Unicode forms to the canonical long
        # form (idempotent — chunked_merge() already normalized the snippet
        # above; repeated here so this fast path stays correct on its own)
        # and DROP the marker-only lines BEFORE any indent arithmetic, so a
        # leading marker line cannot become the alignment base.
        snippet_text = _normalize_markers(snippet.rstrip("\n") + "\n")
        raw_snippet_lines = snippet_text.splitlines(keepends=True)
        kept_snippet_lines = [
            ln for ln in raw_snippet_lines if not _is_marker_line(ln)
        ]
        dropped_markers = len(raw_snippet_lines) - len(kept_snippet_lines)
        if dropped_markers:
            _log.warning(
                "after='%s': dropped %d preservation marker line(s) from the "
                "snippet — markers are directives to the merge pipeline, "
                "never content",
                after, dropped_markers,
            )
        if not any(ln.strip() for ln in kept_snippet_lines):
            # Fail loudly (repo convention): silently splicing an empty
            # piece would return a parse-valid, zero-token result
            # indistinguishable from a successful insert.
            raise ValueError(
                f"after='{after}' snippet carries no insertable code: every "
                f"line is a preservation marker or blank. Pass the new code "
                f"to insert after '{after}'."
            )
        snippet_text = "".join(kept_snippet_lines)

        # Parse the in-memory `original_code` (authoritative) instead of
        # shelling out to `tldr structure`, which consults a daemon cache
        # that can return stale line numbers after a recent write. See
        # `get_ast_map_from_source` for rationale.
        ast_nodes = get_ast_map_from_source(original_code, file_path)
        anchor_node = _resolve_symbol(after, ast_nodes or [])
        if anchor_node is None:
            available = _qualified_symbol_names(ast_nodes or [])
            raise ValueError(
                f"Symbol '{after}' not found in {file_path}. "
                f"Available: {available}"
            )

        # Insert after the anchor's last line
        anchor_end = anchor_node.line_end  # 1-indexed
        before = original_lines[:anchor_end]
        after_lines = original_lines[anchor_end:]

        # Align snippet indent to match the anchor's indent level.
        anchor_start_idx = anchor_node.line_start - 1
        anchor_first_line = original_lines[anchor_start_idx] if anchor_start_idx < total_lines else ""
        snippet_text = _align_snippet_indent(snippet_text, anchor_first_line)

        # B20 (splice half): the inserted piece must carry the ORIGINAL's
        # line-ending convention — a bare-LF piece inside a CRLF file is a
        # mixed-ending seam. Step 14 funnels the assembled merge through
        # _normalize_merged_eol below; normalizing the piece here as well
        # keeps the piece and its blank-line separators in one convention
        # before the (idempotent) funnel re-check.
        snippet_text = normalize_line_endings(snippet_text, line_ending)
        snippet_parts = snippet_text.splitlines(keepends=True)

        # Ensure blank line separator before and after the new code —
        # spelled with the file's own ending, never a hardcoded "\n".
        separator = [line_ending] if before and before[-1].strip() != "" else []
        trailing = [line_ending] if after_lines and after_lines[0].strip() != "" else []

        result_lines = before + separator + snippet_parts + trailing + after_lines
        merged = "".join(result_lines)
        # Step 14: funnel — the after= path already normalizes its own piece,
        # so this is a no-op here; every return path routes through it so the
        # trailing-newline state is enforced in exactly one place.
        merged = _normalize_merged_eol(merged, original_code)

        parse_valid = True
        if language:
            from ..data_gen.ast_analyzer import validate_parse
            parse_valid = validate_parse(merged, language)

        _log.info(
            "Fast-path insert after '%s' (L%d): %d snippet lines, 0 model tokens",
            after, anchor_end, len(snippet_parts),
        )
        return ChunkedMergeResult(
            merged_code=merged,
            parse_valid=parse_valid,
            chunks_used=0,
            chunk_regions=[],
            model_tokens=0,
            latency_ms=0.0,
        )

    # Guard: `replace=X` means "replace X with the snippet". The snippet
    # must therefore define at most X itself — not X plus additional new
    # symbols. Multi-symbol snippets under `replace=` silently force the
    # model to extend the chunk with code it has no draft for, which
    # breaks speculative decoding and burns minutes of AR generation.
    # If you want to replace X AND add Y, use fast_batch_edit with two
    # entries: {replace: 'X', ...} and {after: 'X', snippet: 'def Y...'}.
    if replace:
        extras = _top_level_extras(snippet, language, replace)
        if extras:
            raise ValueError(
                f"replace='{replace}' snippet defines additional symbol(s) "
                f"{extras}. One fast_edit call targets one symbol. "
                f"Use fast_batch_edit to replace '{replace}' and add "
                f"{extras} in a single round-trip: "
                f"[{{'replace': '{replace}', 'snippet': '...'}}, "
                f"{{'after': '{replace}', 'snippet': 'def {extras[0]}...'}}]"
            )

        # Fast path: deterministic text-match — 0 model tokens, instant.
        # Classifies snippet lines as context (matches original) vs new (the edit),
        # then splices new lines between context anchors. Falls back to model
        # if <2 context anchors or unsafe gap detected.
        from .text_match import _bracket_balance, deterministic_edit

        # In-memory parse (race-free). See comment above the `after:` fast
        # path for why we do not consult the tldr daemon here.
        ast_nodes = get_ast_map_from_source(original_code, file_path)
        target_node = _resolve_symbol(replace, ast_nodes or [])
        if target_node:
            func_start = target_node.line_start - 1  # 0-indexed
            func_end = target_node.line_end  # 1-indexed inclusive
            original_func = "".join(original_lines[func_start:func_end])
            # B9: how many leading snippet lines ARE the auto-prepended
            # signature span (0 when no prepend happens). Threaded into
            # deterministic_edit so the span can be pinned as fixed context.
            prepend_signature_lines = 0

            # Auto-preserve signature: when the caller passes replace=<name>
            # and the snippet doesn't contain the target's def/fn/func line,
            # prepend it from the AST. The replace= kwarg already identifies
            # the target structurally — requiring the signature in the snippet
            # is redundant. Without this guard the deterministic_edit path
            # (and the direct-swap fallback) treat the snippet as the full
            # new body and silently strip the signature. See regression test
            # ``test_replace_with_body_only_snippet_preserves_signature``.
            #
            # Scope: function/method/class-like symbols only. Constants,
            # fields, and other value-like targets are intentionally
            # excluded — their "body" is a single expression and callers
            # always include the full declaration. Auto-prepending on a
            # const turns a valid full-replacement snippet into a
            # two-declaration blob that breaks deterministic_edit (see
            # ``test_direct_swap_extend_literal_rust``).
            _SIGNATURE_KINDS = {
                "function", "method", "class", "interface",
                "struct", "enum", "trait", "impl", "module",
                "object", "protocol",
            }
            if (
                target_node.kind in _SIGNATURE_KINDS
                and func_start < len(original_lines)
                and not _snippet_has_target_signature(snippet, replace)
            ):
                # Multi-line signatures (def foo(\n    a,\n    b,\n):) require
                # walking the AST to find where the body begins. The old
                # approach grabbed only line[func_start] — the ``def foo(``
                # line — and dropped the continuation, producing unclosed
                # parens. _extract_signature_via_ast returns the full span
                # up to and including the body-opening delimiter (B9) and
                # falls back to the single-line behavior for unusual
                # grammars.
                fallback = original_lines[func_start]
                if not fallback.endswith("\n"):
                    fallback += "\n"
                target_signature_line = _extract_signature_via_ast(
                    original_code, language,
                    target_node.line_start, target_node.line_end,
                    fallback,
                )
                snippet = target_signature_line + snippet
                prepend_signature_lines = target_signature_line.count("\n")
                _log.info(
                    "replace='%s': snippet missing signature, auto-prepended "
                    "from AST (line %d, %d lines)",
                    replace, func_start + 1, prepend_signature_lines,
                )

            edited = deterministic_edit(
                original_func, snippet,
                prepend_signature_lines=prepend_signature_lines,
            )
            if edited is not None and _deterministic_result_is_faithful(
                original_func, edited, snippet, prepend_signature_lines,
            ):
                edited_lines = edited.splitlines(keepends=True)
                result_lines = list(original_lines)
                result_lines[func_start:func_end] = edited_lines
                merged = "".join(result_lines)
                # B19/B31 (Step 14): the editor re-emits the replaced span
                # LF-only and nothing here may decide the file's trailing-
                # newline state with a constant. The central funnel converts
                # the span to the original's ending (a mid-file span keeps
                # the terminator deterministic_edit's own finisher gave it)
                # and enforces the original's trailing state at EOF.
                merged = _normalize_merged_eol(merged, original_code)

                parse_valid = True
                if language:
                    from ..data_gen.ast_analyzer import validate_parse
                    parse_valid = validate_parse(merged, language)

                if not parse_valid:
                    # The splice is content-faithful (nothing deleted) but
                    # parse-invalid — e.g. a brace-language full-function
                    # snippet whose new body line cannot coexist with the
                    # kept original body line. Hold the editor's output to
                    # the same structural standard it applies to partial
                    # snippets (see the direct-swap gate below): discard it
                    # and let the qualified direct-swap — or the validated
                    # model path — produce a writable merge, instead of
                    # returning output the tool gates would only refuse.
                    _log.warning(
                        "Deterministic text-match for replace='%s' produced "
                        "a parse-invalid merge; discarding it and falling "
                        "through to direct-swap/model",
                        replace,
                    )
                else:
                    _log.info(
                        "Deterministic text-match for replace='%s': "
                        "0 model tokens, %d context anchors",
                        replace, sum(1 for _ in edited.splitlines()),
                    )
                    return ChunkedMergeResult(
                        merged_code=merged,
                        parse_valid=parse_valid,
                        chunks_used=0,
                        chunk_regions=[],
                        model_tokens=0,
                        latency_ms=0.0,
                    )
            elif edited is not None:
                # B3: the editor's own decline rules cover structure, but
                # a content-corrupt result (dropped original line, leaked
                # marker, selective re-indent) parses fine. Defense in
                # depth: discard it and take the ordinary direct-swap /
                # model route below — the deterministic path never
                # bypasses the validator.
                _log.warning(
                    "Deterministic result for replace='%s' failed the "
                    "content-faithfulness check; discarding it and "
                    "falling through to the model path",
                    replace,
                )
            # Direct-swap fast-path: when deterministic_edit can't anchor
            # (every body line changed), but the snippet is a complete
            # re-definition of the target symbol, we can do a pure AST
            # boundary replacement — zero model tokens. Covers patterns
            # like change_signature, extend_literal, and full-function
            # wrap_block.
            #
            # Conditions:
            #   1. Snippet contains no marker lines — markers imply
            #      "preserve original chunks" which is inherently
            #      incompatible with whole-symbol swap.
            #   2. Snippet parses cleanly (tldr structure) and reports
            #      exactly one top-level definition.
            #   3. That definition's name equals `replace` (already
            #      verified by the extras-check above, but re-verify
            #      here against the parsed AST to guard against
            #      regex-only name extraction false positives).
            #   4. The snippet is a COMPLETE re-definition: its bracket
            #      balance equals the replaced span's. A partial snippet
            #      (e.g. signature prepended, body given, closer still
            #      unspoken) would splice into an unbalanced, parse-invalid
            #      file — the same structural gate the text-match editor
            #      applies to its own output. Decline to the model instead.
            if not any(_is_marker_line(ln) for ln in snippet.splitlines()):
                from pathlib import Path as _Path
                snippet_parse = _try_tldr_snippet_parse(
                    snippet, _Path(file_path).suffix,
                )
                snippet_balance = _bracket_balance(snippet)
                span_balance = _bracket_balance(original_func)
                snippet_is_complete = snippet_balance == span_balance
                if snippet_parse and not snippet_is_complete:
                    _log.info(
                        "Direct-swap declined for replace='%s': snippet "
                        "parses but is not a complete re-definition "
                        "(bracket balance %d vs %d) — falling through",
                        replace, snippet_balance, span_balance,
                    )
                if (
                    snippet_parse
                    and snippet_is_complete
                    and len(snippet_parse) == 1
                    and snippet_parse[0] == replace
                ):
                    # Align snippet indent to match the target function's
                    # indent in the original.
                    anchor_first_line = (
                        original_lines[func_start]
                        if func_start < total_lines
                        else ""
                    )
                    # B31 (Step 14): the terminator appended here is only the
                    # SPLICE SEPARATOR a mid-file replacement needs (an
                    # unterminated last line would concatenate onto the next
                    # original line). It is spelled with the file's own
                    # ending; whether the FILE ends with a terminator is
                    # decided by _normalize_merged_eol below, from the
                    # original — never by this constant.
                    snippet_text = snippet.rstrip("\r\n") + line_ending
                    snippet_text = _align_snippet_indent(
                        snippet_text, anchor_first_line,
                    )
                    snippet_lines = snippet_text.splitlines(keepends=True)

                    result_lines = list(original_lines)
                    result_lines[func_start:func_end] = snippet_lines
                    merged = "".join(result_lines)
                    # Step 14 (B19): a bare-LF snippet swapped into a CRLF
                    # file is converted to the file's convention here — the
                    # splice that leaked CR bytes on the batch/multi paths.
                    merged = _normalize_merged_eol(merged, original_code)

                    parse_valid = True
                    if language:
                        from ..data_gen.ast_analyzer import validate_parse
                        parse_valid = validate_parse(merged, language)

                    _log.info(
                        "Direct-swap for replace='%s' (L%d-L%d): "
                        "0 model tokens, %d snippet lines",
                        replace, func_start + 1, func_end, len(snippet_lines),
                    )
                    return ChunkedMergeResult(
                        merged_code=merged,
                        parse_valid=parse_valid,
                        chunks_used=0,
                        chunk_regions=[],
                        model_tokens=0,
                        latency_ms=0.0,
                    )

            _log.info(
                "Deterministic text-match and direct-swap failed for "
                "replace='%s', falling back to model",
                replace,
            )

    chunks = locate_chunks(
        snippet, original_code, file_path, padding, language,
        replace=replace,
    )

    _log.info(
        "locate_chunks returned %d chunk(s) for %d-line file: %s",
        len(chunks), total_lines,
        [(c.start_line, c.end_line, c.matched_nodes) for c in chunks],
    )

    # Reject whole-file merge on large files — model will truncate
    _MAX_WHOLE_FILE_LINES = 150
    is_whole_file = (
        len(chunks) == 1
        and chunks[0].start_line == 1
        and chunks[0].end_line == total_lines
    )
    if is_whole_file and total_lines > _MAX_WHOLE_FILE_LINES:
        # Build a helpful list of available symbols
        ast_nodes = get_ast_map(file_path, total_lines)
        available = _qualified_symbol_names(ast_nodes or [])
        sym_hint = ""
        if available:
            preview = available[:8]
            sym_hint = (
                f" Available symbols: {preview}"
                + (f" (+{len(available) - 8} more)" if len(available) > 8 else "")
            )
        raise ValueError(
            f"Whole-file merge rejected: {total_lines} lines exceeds "
            f"{_MAX_WHOLE_FILE_LINES}-line safety limit. "
            f"Use `after='symbol'` to insert new code, or "
            f"`replace='symbol'` to modify lines inside an existing function "
            f"(context markers work — model sees only the function, not the whole file)."
            f"{sym_hint}"
        )

    # If only one chunk covering the whole file, just do a normal merge
    if is_whole_file:
        # B33: one random nonce per merge attempt — the placeholders sent to
        # the model cannot collide with user text (e.g. a file that literally
        # contains the old fixed placeholder string).
        tag_nonce = _new_tag_nonce()
        safe_code = _escape_tags(original_code, tag_nonce)
        safe_snippet = _escape_tags(snippet, tag_nonce)
        result = merge_fn(safe_code, safe_snippet, language)
        merged = _unescape_tags(result.merged_code, tag_nonce)
        # Step 14 (B19/B31): the model's payload is funnelled through the
        # central normalizer BEFORE the rejection gates below, so the parse
        # and content validators see the exact bytes that would be written.
        merged = _normalize_merged_eol(merged, original_code)

        # Step 11 (B13/B29): the whole-file path hands the ENTIRE file to
        # the model, so its output is validated before it may be returned:
        # the truncation flag (B12), parse validity when the language is
        # known, and the shared content validator over the FULL original
        # vs the FULL merge output — the same preserve-by-default contract
        # the per-chunk path is held to. A merge that silently drops an
        # unmentioned original line, invents code or leaks a marker parses
        # fine; only the content validator can see it.
        reason = _whole_file_rejection_reason(
            original_code, merged, snippet, language,
            getattr(result, "truncated", False),
        )

        if reason is not None:
            # Single corrective retry — the same one-shot budget the chunk
            # loop's retry machinery grants. The corrective note rides on
            # the snippet (merge_fn's only prompt channel); the retry is
            # validated against the ORIGINAL snippet, so the note itself
            # can never contaminate the gate.
            _log.warning(
                "Whole-file merge %s, retrying once with a corrective note",
                reason,
            )
            retry = merge_fn(
                safe_code, _append_corrective_note(safe_snippet, reason),
                language,
            )
            retry_merged = _unescape_tags(retry.merged_code, tag_nonce)
            retry_merged = _normalize_merged_eol(retry_merged, original_code)
            retry_reason = _whole_file_rejection_reason(
                original_code, retry_merged, snippet, language,
                getattr(retry, "truncated", False),
            )
            if retry_reason is None:
                result = retry
                merged = retry_merged
            else:
                # Both attempts unusable: reject the whole-file merge — no
                # write. Rejection convention = the chunk loop's: the
                # original file is kept as merged_code (never the
                # corrupted payload), parse_valid is forced False (the
                # universal "do not persist this" signal), and the
                # chunks_rejected/chunks_used accounting makes the
                # existing MCP/CLI gates refuse the write naturally.
                _log.error(
                    "Whole-file merge rejected after retry (%s) — keeping "
                    "the original file",
                    retry_reason,
                )
                return ChunkedMergeResult(
                    merged_code=original_code,
                    parse_valid=False,
                    chunks_used=1,
                    chunk_regions=[(1, total_lines)],
                    model_tokens=result.tokens_generated + retry.tokens_generated,
                    latency_ms=result.latency_ms + retry.latency_ms,
                    chunks_rejected=1,
                )

        return ChunkedMergeResult(
            merged_code=merged,
            parse_valid=result.parse_valid,
            chunks_used=1,
            chunk_regions=[(1, total_lines)],
            model_tokens=result.tokens_generated,
            latency_ms=result.latency_ms,
            chunks_rejected=0,
        )

    # For multi-chunk edits, split the snippet so each chunk only sees
    # its relevant portion (prevents import chunk from getting code, etc.)
    import_snippet = snippet
    code_snippet = snippet
    if len(chunks) > 1:
        has_import_chunk = any("<imports>" in c.matched_nodes for c in chunks)
        if has_import_chunk:
            import_snippet, code_snippet = _split_snippet(snippet, language)
            # If splitting produced empty parts, fall back to full snippet
            if not import_snippet.strip():
                import_snippet = snippet
            if not code_snippet.strip():
                code_snippet = snippet

    # Process each chunk independently and splice back.
    # Work backwards so line numbers don't shift.
    result_lines = list(original_lines)
    total_tokens = 0
    total_latency = 0.0
    rejected_chunks = 0
    chunk_regions = []

    for chunk in reversed(chunks):
        start_idx = chunk.start_line - 1  # 0-indexed
        end_idx = chunk.end_line           # exclusive

        # B33: a fresh nonce per chunk attempt — the chunk's escaped text is
        # the ONLY text carrying this nonce, so even a user file containing
        # placeholder-looking strings cannot collide with it.
        tag_nonce = _new_tag_nonce()
        chunk_text = "".join(original_lines[start_idx:end_idx])
        chunk_text = _escape_tags(chunk_text, tag_nonce)

        # Use the appropriate snippet portion for this chunk
        if len(chunks) > 1 and "<imports>" in chunk.matched_nodes:
            chunk_snippet = import_snippet
        elif len(chunks) > 1:
            chunk_snippet = code_snippet
        else:
            chunk_snippet = snippet

        safe_chunk_snippet = _escape_tags(chunk_snippet, tag_nonce)
        safe_chunk_snippet = _align_snippet_indent(safe_chunk_snippet, chunk_text)
        result = merge_fn(chunk_text, safe_chunk_snippet, language)

        # Retry once on parse failure or a truncated (length-capped)
        # response (B12) — the same single-retry machinery for both. A
        # truncated response is error-shaped even when its payload looks
        # complete; the retry gives the model one clean shot.
        merged_chunk_code = _unescape_tags(result.merged_code, tag_nonce)
        result_truncated = getattr(result, "truncated", False)
        parse_failed = False
        if language:
            from ..data_gen.ast_analyzer import validate_parse
            parse_failed = not validate_parse(merged_chunk_code, language)
        if result_truncated or parse_failed:
            _log.warning(
                "Chunk %d-%d %s, retrying once",
                chunk.start_line, chunk.end_line,
                "truncated (model hit the token cap)"
                if result_truncated else "parse invalid",
            )
            retry = merge_fn(chunk_text, safe_chunk_snippet, language)
            retry_code = _unescape_tags(retry.merged_code, tag_nonce)
            retry_clean = not getattr(retry, "truncated", False)
            if language:
                retry_clean = retry_clean and validate_parse(retry_code, language)
            if retry_clean:
                result = retry
                merged_chunk_code = retry_code

        # B12: a truncated result must NEVER be spliced — the model ran
        # out of tokens mid-payload, so the best-effort text is a partial
        # file. After the single retry above, mark the chunk rejected
        # (existing rejection bookkeeping) and keep the original chunk.
        if getattr(result, "truncated", False):
            _log.error(
                "Chunk %d-%d truncated after retry (model hit the token "
                "cap) — rejecting edit (keeping original)",
                chunk.start_line, chunk.end_line,
            )
            rejected_chunks += 1
            total_tokens += result.tokens_generated
            total_latency += result.latency_ms
            chunk_regions.append((chunk.start_line, chunk.end_line))
            continue

        # Re-align output indent to match the original chunk.
        merged_chunk_code = _realign_output(merged_chunk_code, chunk_text)

        # Check for hallucinations: verify anchor lines survived.
        # If the model dropped too many original lines, retry once.
        raw_chunk = _unescape_tags(chunk_text, tag_nonce)
        anchor_score = _check_hallucinations(raw_chunk, merged_chunk_code, chunk_snippet)
        if anchor_score < 0.85:
            _log.warning(
                "Chunk %d-%d anchor score %.1f%% — retrying (hallucination likely)",
                chunk.start_line, chunk.end_line, anchor_score * 100,
            )
            retry = merge_fn(chunk_text, safe_chunk_snippet, language)
            retry_code = _unescape_tags(retry.merged_code, tag_nonce)
            retry_code = _realign_output(retry_code, chunk_text)
            retry_score = _check_hallucinations(raw_chunk, retry_code, chunk_snippet)
            if retry_score > anchor_score:
                merged_chunk_code = retry_code
                result = retry
                anchor_score = retry_score
                _log.info("Retry improved anchor score: %.1f%% → %.1f%%", anchor_score * 100, retry_score * 100)
            else:
                _log.info("Retry didn't improve (%.1f%%), keeping original", retry_score * 100)

        # If both attempts failed badly, keep the original chunk unchanged
        # rather than writing corrupted code to disk.
        if anchor_score < 0.5:
            _log.error(
                "Chunk %d-%d anchor score %.1f%% — rejecting edit (keeping original)",
                chunk.start_line, chunk.end_line, anchor_score * 100,
            )
            rejected_chunks += 1
            total_tokens += result.tokens_generated
            total_latency += result.latency_ms
            chunk_regions.append((chunk.start_line, chunk.end_line))
            continue

        total_tokens += result.tokens_generated
        total_latency += result.latency_ms

        # A mid-file chunk whose last line lacks a terminator would
        # concatenate onto the following original line at splice time, so
        # the chunk is ALWAYS well-formed here; WHICH ending it carries —
        # and whether the FILE ends with a terminator at all — is decided by
        # _normalize_merged_eol at the return below (B31: the file's
        # trailing-newline state comes from the original, never from this
        # separator).
        merged_chunk_lines = merged_chunk_code.splitlines(keepends=True)
        if merged_chunk_lines and not merged_chunk_lines[-1].endswith(("\n", "\r")):
            merged_chunk_lines[-1] += line_ending

        result_lines[start_idx:end_idx] = merged_chunk_lines
        chunk_regions.append((chunk.start_line, chunk.end_line))

    merged_code = "".join(result_lines)
    # Step 14 (B19/B31): the assembled file is funnelled through the central
    # normalizer — produced chunks carrying the model's endings are converted
    # to the original's convention, and the file's trailing-newline state is
    # the original's.
    merged_code = _normalize_merged_eol(merged_code, original_code)

    parse_valid = True
    if language:
        from ..data_gen.ast_analyzer import validate_parse
        parse_valid = validate_parse(merged_code, language)

    return ChunkedMergeResult(
        merged_code=merged_code,
        parse_valid=parse_valid,
        chunks_used=len(chunks),
        chunk_regions=list(reversed(chunk_regions)),
        model_tokens=total_tokens,
        latency_ms=total_latency,
        chunks_rejected=rejected_chunks,
    )
