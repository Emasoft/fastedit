"""Step 17 — anchor disambiguation regressions (B24, B27, B28).

B24 — text_match bound each snippet context line to the FIRST forward
  original occurrence whose indent delta stayed within ±2 of the previous
  anchor's delta. A confidently-wrong-but-consistent repeat (``x = 1`` in
  two functions, a repeated ``return``) dragged the whole edit to the
  wrong region. The binding must consider ALL forward occurrences and
  prefer the one that keeps the snippet's anchor SEQUENCE contiguous with
  its neighbors AND matches the snippet line's indent; when two bindings
  remain equally valid the editor must DECLINE (return None) instead of
  silently picking the first.

B27 — ``_narrow_large_node`` accepted a sliding-window match at a stripped
  score as low as 0.2 and then cut ±padding RAW lines around it — mid-block
  cuts hand the model broken code to "repair". The acceptance threshold is
  a named tunable ≥ 0.6, and cuts happen at AST node edges (the enclosing
  block from the tree-sitter walk); without AST evidence the narrow is
  rejected (full node returned).

B28 — ``_find_insertion_region`` matched each context line to its FIRST
  occurrence and took ``max()`` as the anchor, so a common line steered
  the insertion anywhere. The anchor must be the RAREST context line
  (least frequent, then longest normalized content), and ALL context
  lines must land inside one consistent region — otherwise no region is
  returned and the caller's fallback runs.
"""

from __future__ import annotations

import pytest

from fastedit.inference import chunk_locator, snippet_analysis
from fastedit.inference.ast_utils import ASTNode
from fastedit.inference.chunk_locator import _narrow_large_node, locate_chunks
from fastedit.inference.text_match import deterministic_edit

# ===========================================================================
# B24 — deterministic_edit anchor disambiguation
# ===========================================================================


def test_repeated_context_line_binds_to_contiguous_sequence():
    """A repeated context line binds where the anchor SEQUENCE is contiguous.

    ``x = 1`` occurs in both alpha and beta; ``run(x)`` occurs only in beta,
    directly after beta's ``x = 1``. The old first-forward binding chose
    (alpha's x=1, beta's run(x)) — a binding spanning both functions — and
    the marker section then inserted the new ``telemetry(x)`` line at the
    TOP of that wrong gap, i.e. into alpha. The contiguous binding
    (5, 6) keeps both anchors together in beta.
    """
    original = (
        "def alpha():\n"
        "    x = 1\n"
        "    return x\n"
        "\n"
        "def beta():\n"
        "    x = 1\n"
        "    run(x)\n"
        "    return x\n"
    )
    snippet = (
        "    x = 1\n"
        "    telemetry(x)\n"
        "    # ... existing code ...\n"
        "    run(x)\n"
    )

    result = deterministic_edit(original, snippet)

    assert result is not None, (
        "contiguous binding exists — the editor must apply the edit, "
        "not decline"
    )
    lines = result.splitlines()
    # The edit landed in beta: telemetry sits between beta's two anchors.
    assert lines[5] == "    x = 1"
    assert lines[6] == "    telemetry(x)"
    assert lines[7] == "    run(x)"
    # Alpha is untouched — the old binding spliced telemetry in here.
    assert lines[1] == "    x = 1"
    assert lines[2] == "    return x"
    assert "telemetry" not in "\n".join(lines[:5])


def test_equally_valid_bindings_decline_to_model():
    """Two equally valid contiguous bindings → decline, never pick the first.

    alpha and beta have byte-identical bodies; the snippet's anchors match
    both regions with identical contiguity and indent. The old code
    silently picked alpha (first forward occurrence).
    """
    original = (
        "def alpha():\n"
        "    x = 1\n"
        "    run(x)\n"
        "\n"
        "def beta():\n"
        "    x = 1\n"
        "    run(x)\n"
    )
    snippet = (
        "    x = 1\n"
        "    probe()\n"
        "    run(x)\n"
    )

    assert deterministic_edit(original, snippet) is None, (
        "two equally valid anchor bindings must decline to the model path"
    )


def test_gate_fails_closed_on_ambiguous_binding():
    """The Step-6 gate consumes the same classifier — ambiguity fails closed.

    ``_deterministic_result_is_faithful`` must never ratify output whose
    anchor partition the editor itself would decline.
    """
    from fastedit.inference.chunked_merge import (
        _deterministic_result_is_faithful,
    )

    original = (
        "def alpha():\n"
        "    x = 1\n"
        "    run(x)\n"
        "\n"
        "def beta():\n"
        "    x = 1\n"
        "    run(x)\n"
    )
    snippet = (
        "    x = 1\n"
        "    probe()\n"
        "    run(x)\n"
    )

    assert _deterministic_result_is_faithful(
        original, original, snippet,
    ) is False


def test_identical_indent_occurrence_wins_without_declining():
    """Same content at two indents: the indent-matching occurrence wins.

    ``x = 1`` / ``run(x)`` exist at indent 4 (alpha) and indent 8 (beta's
    guarded block). The snippet is written at indent 8 — both bindings are
    equally contiguous, so the identical-indent preference must resolve the
    tie toward beta WITHOUT declining (the old code bound alpha, the first
    forward occurrence).
    """
    original = (
        "def alpha():\n"
        "    x = 1\n"
        "    run(x)\n"
        "\n"
        "def beta():\n"
        "    if guard:\n"
        "        x = 1\n"
        "        run(x)\n"
    )
    snippet = (
        "        x = 1\n"
        "        audit(x)\n"
        "        run(x)\n"
    )

    result = deterministic_edit(original, snippet)

    assert result is not None, (
        "identical-indent preference must resolve the tie, not decline"
    )
    lines = result.splitlines()
    # The edit landed inside beta's guarded block.
    assert lines[6] == "        x = 1"
    assert lines[7] == "        audit(x)"
    assert lines[8] == "        run(x)"
    # Alpha (the shallow-indent occurrence) is untouched.
    assert lines[1] == "    x = 1"
    assert lines[2] == "    run(x)"
    assert "audit" not in "\n".join(lines[:6])


# ===========================================================================
# B27 — _narrow_large_node threshold + AST-edge cuts
# ===========================================================================


def _large_python_source() -> list[str]:
    """A 208-line function: two 100-line for-blocks around scalars.

    Lines (1-indexed): def=1, total=2, phase_one=3, for i=4, step=5..104,
    phase_two=105, for j=106, step2=107..206, audit=207, return=208.
    """
    lines = ["def big():", "    total = 0", "    phase_one()"]
    lines.append("    for i in range(100):")
    lines += [f"        step({i})" for i in range(100)]
    lines.append("    phase_two()")
    lines.append("    for j in range(100):")
    lines += [f"        step2({j})" for j in range(100)]
    lines.append("    audit(total)")
    lines.append("    return total")
    return lines


def _large_node() -> ASTNode:
    return ASTNode(
        name="big", kind="function",
        line_start=1, line_end=208, signature="def big():",
    )


def test_narrow_acceptance_threshold_is_named_and_at_least_0_6():
    """The B27 acceptance threshold is a module-level tunable ≥ 0.6."""
    assert chunk_locator._MIN_NARROW_SCORE >= 0.6


def test_narrow_low_overlap_rejected_returns_full_node():
    """A 0.5-score window must NOT narrow into a mid-block cut.

    The snippet matches only half its window inside the second for-block.
    Pre-fix the 0.2 threshold accepted it and cut (96, 208) — a window
    slicing into surrounding code. Post-fix the narrow is rejected and the
    full node range is returned (the caller keeps the whole function).
    """
    lines = _large_python_source()
    snippet = (
        "        step2(40)\n"
        "        BRAND_NEW_ALPHA = 1\n"
        "        BRAND_NEW_BETA = 2\n"
        "        step2(43)\n"
    )

    result = _narrow_large_node(
        _large_node(), snippet, lines,
        original_code="\n".join(lines) + "\n", language="python",
    )

    assert result == (1, 208), (
        f"low-overlap window narrowed to {result}; the 0.2-threshold "
        f"mid-block cut must be rejected (full node returned)"
    )


def test_narrow_high_overlap_cuts_at_block_edges():
    """Positive control: a ≥0.6 window still narrows — at AST node edges.

    The snippet matches the second for-block exactly, so the cut boundaries
    are the block's own edges (106..206), not block±padding raw lines.
    """
    lines = _large_python_source()
    snippet = (
        "        step2(40)\n"
        "        step2(41)\n"
        "        step2(42)\n"
        "        step2(43)\n"
    )

    result = _narrow_large_node(
        _large_node(), snippet, lines,
        original_code="\n".join(lines) + "\n", language="python",
    )

    assert result == (106, 206), (
        f"high-overlap window narrowed to {result}; expected the "
        f"enclosing block's AST edges (106, 206)"
    )


def test_narrow_without_enclosing_block_rejected():
    """No enclosing block → no raw-line fallback window; full node returned.

    The snippet matches the function's top-level statements (lines 2..3),
    which no for/if/while/try encloses. Pre-fix this produced the ±padding
    raw-line window (1, 14), slicing into the first for-block. Without AST
    evidence the narrow must be rejected.
    """
    lines = _large_python_source()
    snippet = (
        "    total = 0\n"
        "    phase_one()\n"
    )

    result = _narrow_large_node(
        _large_node(), snippet, lines,
        original_code="\n".join(lines) + "\n", language="python",
    )

    assert result == (1, 208), (
        f"window with no enclosing block narrowed to {result}; the "
        f"raw-line fallback cut must be rejected (full node returned)"
    )


# ===========================================================================
# B28 — _find_insertion_region rarest-context anchoring
# ===========================================================================


def _patch_snippet_defs(monkeypatch, node: ASTNode) -> None:
    monkeypatch.setattr(
        snippet_analysis, "_get_snippet_definitions",
        lambda _snippet, _language=None: [node],
    )


def test_insertion_region_anchors_on_rarest_context_line(monkeypatch):
    """A rare context line outranks a common one for the insertion anchor.

    ``rare_beta_hook()`` occurs once (line 2, the true neighborhood);
    ``setup()`` occurs three times, first at line 6. Pre-fix the anchor was
    max(first occurrences) = 6, steering the region to (5, 11). The rare
    line anchors (1, 6).
    """
    original_lines = [
        "def early():",
        "    rare_beta_hook()",
        "    compute_extras()",
        "",
        "def middle():",
        "    setup()",
        "",
        "def late():",
        "    setup()",
        "    setup()",
        "    return",
    ]
    ast_nodes = [
        ASTNode(name="early", kind="function",
                line_start=1, line_end=3, signature="def early():"),
        ASTNode(name="middle", kind="function",
                line_start=5, line_end=6, signature="def middle():"),
        ASTNode(name="late", kind="function",
                line_start=8, line_end=11, signature="def late():"),
    ]
    _patch_snippet_defs(
        monkeypatch,
        ASTNode(name="gamma", kind="function",
                line_start=3, line_end=4, signature="def gamma():"),
    )
    snippet = "    rare_beta_hook()\n    setup()\ndef gamma():\n    pass\n"

    region = snippet_analysis._find_insertion_region(
        snippet, original_lines, ast_nodes, 11,
    )

    assert region is not None
    assert (region.start_line, region.end_line) == (1, 6), (
        f"common first occurrence steered the insertion region to "
        f"{(region.start_line, region.end_line)}; the rare context line "
        f"must anchor (1, 6)"
    )


def test_insertion_region_conflicting_context_returns_no_region(monkeypatch):
    """Context lines from two disjoint regions → no region at all.

    ``hook_one()`` lives at line 2, ``hook_two()`` at line 8; no single
    bracketed region contains both. Pre-fix max(first occurrences) = 8
    returned (7, 11) — a guess. Post-fix the function returns None so the
    caller's fallback runs.
    """
    original_lines = [
        "def top_a():",
        "    hook_one()",
        "",
        "def top_b():",
        "    filler()",
        "",
        "def bottom_a():",
        "    hook_two()",
        "",
        "def bottom_b():",
        "    filler()",
    ]
    ast_nodes = [
        ASTNode(name="top_a", kind="function",
                line_start=1, line_end=2, signature="def top_a():"),
        ASTNode(name="top_b", kind="function",
                line_start=4, line_end=5, signature="def top_b():"),
        ASTNode(name="bottom_a", kind="function",
                line_start=7, line_end=8, signature="def bottom_a():"),
        ASTNode(name="bottom_b", kind="function",
                line_start=10, line_end=11, signature="def bottom_b():"),
    ]
    _patch_snippet_defs(
        monkeypatch,
        ASTNode(name="gamma", kind="function",
                line_start=3, line_end=4, signature="def gamma():"),
    )
    snippet = "    hook_one()\n    hook_two()\ndef gamma():\n    pass\n"

    region = snippet_analysis._find_insertion_region(
        snippet, original_lines, ast_nodes, 11,
    )

    assert region is None, (
        f"conflicting context regions produced {region}; expected no "
        f"region so the caller's fallback runs"
    )


def test_locate_chunks_conflicting_insertion_falls_back_to_whole_file(
    tmp_path, monkeypatch,
):
    """The caller's fallback for a declined insertion region is safe.

    locate_chunks must degrade to the conservative whole-file region when
    _find_insertion_region declines — never to a guessed sub-region.
    """
    src = """\
def top_a():
    hook_one()

def top_b():
    filler()

def bottom_a():
    hook_two()

def bottom_b():
    filler()
"""
    target = tmp_path / "svc.py"
    target.write_text(src)
    _patch_snippet_defs(
        monkeypatch,
        ASTNode(name="gamma", kind="function",
                line_start=3, line_end=4, signature="def gamma():"),
    )
    snippet = "    hook_one()\n    hook_two()\ndef gamma():\n    pass\n"

    chunks = locate_chunks(snippet, src, str(target), language="python")

    assert len(chunks) == 1, chunks
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 11), (
        f"declined insertion region must fall back to the whole file, "
        f"got {(chunks[0].start_line, chunks[0].end_line)}"
    )
    assert chunks[0].matched_nodes == ["<unmatched>"]


# ===========================================================================
# Keep the Step-6 gate + editor partition identical (B24 critical note):
# the shared classifier means every decline above is observed by both
# consumers; test_gate_fails_closed_on_ambiguous_binding pins the gate side.
# ===========================================================================


_CONTIGUOUS_ORIGINAL = (
    "def alpha():\n"
    "    x = 1\n"
    "    return x\n"
    "\n"
    "def beta():\n"
    "    x = 1\n"
    "    run(x)\n"
    "    return x\n"
)
_CONTIGUOUS_SNIPPET = (
    "    x = 1\n"
    "    telemetry(x)\n"
    "    run(x)\n"
)


@pytest.mark.parametrize(
    "original,snippet",
    # Contiguous unique binding: gate sees the partition the editor used.
    [(_CONTIGUOUS_ORIGINAL, _CONTIGUOUS_SNIPPET)],
)
def test_classifier_is_single_source_for_editor_and_gate(original, snippet):
    """The classifier raises for ambiguity and binds uniquely otherwise.

    Calling it directly must yield exactly the binding the editor applied:
    the contiguous (beta) occurrence, never the first forward one.
    """
    from fastedit.inference.text_match import _classify_edit_lines

    classified = _classify_edit_lines(
        original.splitlines(), snippet.splitlines(),
    )
    contexts = [c for c in classified if c[0] == "context"]
    assert [(c[1], c[2]) for c in contexts] == [(0, 5), (2, 6)], (
        f"classifier bound the repeated anchor at the wrong occurrence: "
        f"{[(c[1], c[2]) for c in contexts]}"
    )
