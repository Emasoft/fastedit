"""Regression tests for TRDD-CMRMA2YG (fastedit edit --replace silent truncation).

`_try_deterministic_replace`'s direct-replacement fallback used to assume "the
snippet IS the new symbol" and splice it verbatim over the symbol's line range.
A snippet containing a keep-marker (`#...`, `//...`, ...) is NOT a complete
symbol -- the marker stands in for lines the author deliberately omitted -- so
splicing it verbatim silently deleted those lines while still exiting 0. These
tests pin the fix: `snippet_has_keep_marker` refuses the unsafe case with a
clean CLI error and leaves the file byte-for-byte unchanged, while every
legitimate case (trailing marker via the deterministic merge path, real code
that merely contains marker-like text, non-Python languages, and a bare `...`
Python stub body) keeps working.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from fastedit.inference.text_match import snippet_has_keep_marker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI_MODULE = [sys.executable, "-m", "fastedit"]


def run_cli(*args: str, input_text: str | None = None):
    """Run `python -m fastedit <args>` and return CompletedProcess."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [*CLI_MODULE, *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


class TestSnippetHasKeepMarker:
    """Unit tests for the predicate itself, independent of the CLI."""

    def test_short_hash_marker_is_detected(self):
        """A bare `#...` line is a keep-marker."""
        assert snippet_has_keep_marker("#...") is True

    def test_indented_short_hash_marker_is_detected(self):
        """An indented `#...` line is still a keep-marker."""
        assert snippet_has_keep_marker("  #...") is True

    def test_short_slash_marker_is_detected(self):
        """A bare `//...` line is a keep-marker."""
        assert snippet_has_keep_marker("//...") is True

    def test_canonical_hash_marker_is_detected(self):
        """The canonical long-form hash marker is a keep-marker."""
        assert snippet_has_keep_marker("# ... existing code ...") is True

    def test_canonical_slash_marker_is_detected(self):
        """The canonical long-form slash marker is a keep-marker."""
        assert snippet_has_keep_marker("// ... existing code ...") is True

    def test_bare_short_ellipsis_phrase_is_not_a_marker(self):
        """A bare `# ...` (no trailing `existing code`) is real code, not a marker."""
        assert snippet_has_keep_marker("# ...") is False

    def test_bare_short_slash_ellipsis_phrase_is_not_a_marker(self):
        """A bare `// ...` (no trailing `existing code`) is real code, not a marker."""
        assert snippet_has_keep_marker("// ...") is False

    def test_real_trailing_comment_is_not_a_marker(self):
        """A real code line whose comment happens to contain `# ...` is not a marker."""
        line = "x = 1  # ... and then some real trailing prose here"
        assert snippet_has_keep_marker(line) is False

    def test_ordinary_code_line_is_not_a_marker(self):
        """A plain code line is not a marker."""
        assert snippet_has_keep_marker("y = 2") is False

    def test_string_literal_ellipsis_is_not_a_marker(self):
        """A string literal containing three dots is not a marker."""
        assert snippet_has_keep_marker("return '...'") is False

    def test_bare_ellipsis_stub_body_is_not_a_marker(self):
        """A bare `...` line (Python Ellipsis stub body) is real code, not a marker."""
        assert snippet_has_keep_marker("...") is False

    def test_protocol_stub_method_is_not_a_marker(self):
        """A full stub method body (`def f(): ...`) is not a marker."""
        snippet = "def f(self) -> int:\n    ...\n"
        assert snippet_has_keep_marker(snippet) is False


class TestMidSnippetMarkerRefusal:
    """The exact TRDD-CMRMA2YG repro: a mid-snippet marker must refuse, not truncate."""

    def test_mid_snippet_marker_refuses_with_nonzero_exit(self, tmp_path: Path):
        """`--replace` with a mid-snippet `#...` marker must exit non-zero."""
        target = tmp_path / "mid.py"
        target.write_text("def g():\n    x = 1\n    y = 2\n    return x + y\n", encoding="utf-8")
        result = run_cli(
            "edit", str(target), "--replace", "g",
            "--snippet", "def g():\n#...\n    return 999\n",
        )
        assert result.returncode != 0

    def test_mid_snippet_marker_refusal_names_the_symbol(self, tmp_path: Path):
        """The refusal message must name the symbol so the user can act on it."""
        target = tmp_path / "mid.py"
        target.write_text("def g():\n    x = 1\n    y = 2\n    return x + y\n", encoding="utf-8")
        result = run_cli(
            "edit", str(target), "--replace", "g",
            "--snippet", "def g():\n#...\n    return 999\n",
        )
        assert "g" in result.stderr
        assert "keep-marker" in result.stderr

    def test_mid_snippet_marker_refusal_prints_no_traceback(self, tmp_path: Path):
        """The refusal must be a clean CLI diagnostic, never a raw Python traceback."""
        target = tmp_path / "mid.py"
        target.write_text("def g():\n    x = 1\n    y = 2\n    return x + y\n", encoding="utf-8")
        result = run_cli(
            "edit", str(target), "--replace", "g",
            "--snippet", "def g():\n#...\n    return 999\n",
        )
        assert "Traceback" not in result.stderr

    def test_mid_snippet_marker_refusal_leaves_file_unchanged(self, tmp_path: Path):
        """The file on disk must be byte-for-byte unchanged after a refused edit."""
        target = tmp_path / "mid.py"
        original = "def g():\n    x = 1\n    y = 2\n    return x + y\n"
        target.write_text(original, encoding="utf-8")
        run_cli(
            "edit", str(target), "--replace", "g",
            "--snippet", "def g():\n#...\n    return 999\n",
        )
        assert target.read_text(encoding="utf-8") == original


class TestTrailingMarkerStillWorks:
    """The trailing-marker case (never reaches the guard) must not regress."""

    def test_trailing_marker_preserves_untouched_tail(self, tmp_path: Path):
        """A trailing `#...` merges via deterministic_edit and preserves b, c, return."""
        target = tmp_path / "tail.py"
        target.write_text(
            "def f():\n    a = 1\n    b = 2\n    c = 3\n    return a + b + c\n",
            encoding="utf-8",
        )
        result = run_cli(
            "edit", str(target), "--replace", "f",
            "--snippet", "def f():\n    a = 99\n#...\n",
        )
        assert result.returncode == 0
        merged = target.read_text(encoding="utf-8")
        assert "b = 2" in merged
        assert "c = 3" in merged
        assert "return a + b + c" in merged


class TestRealCodeWithMarkerLikeTextIsNotRefused:
    """A snippet containing real code that merely resembles a marker must apply."""

    def test_snippet_with_real_hash_ellipsis_comment_is_applied(self, tmp_path: Path):
        """A full replacement body containing `# ...` in a real comment is not refused."""
        target = tmp_path / "h.py"
        target.write_text("def h():\n    return 1\n", encoding="utf-8")
        result = run_cli(
            "edit", str(target), "--replace", "h",
            "--snippet",
            "def h():\n    x = 1  # ... and then some real trailing prose here\n    return x\n",
        )
        assert result.returncode == 0
        assert "return x" in target.read_text(encoding="utf-8")

    def test_ellipsis_stub_body_replacement_is_applied(self, tmp_path: Path):
        """A `--replace` snippet whose full body is a bare `...` stub is not refused."""
        target = tmp_path / "stub.py"
        target.write_text("def f(self) -> int:\n    ...\n", encoding="utf-8")
        result = run_cli(
            "edit", str(target), "--replace", "f",
            "--snippet", "def f(self) -> int:\n    ...\n",
        )
        assert result.returncode == 0


class TestNonPythonLanguage:
    """The guard also refuses the analogous unsafe case in a non-Python language."""

    def test_js_direct_replace_marker_only_body_refuses(self, tmp_path: Path):
        """A JS `--replace` snippet consisting only of a keep-marker must refuse.

        This targets the direct-replacement fallback path specifically (a
        marker-only snippet with no other content cannot be merged by
        `deterministic_edit`, which always declines it), independent of any
        per-language merge-quality behavior of `deterministic_edit` itself.
        """
        target = tmp_path / "mid.js"
        target.write_text(
            "function g() {\n  var x = 1;\n  var y = 2;\n  return x + y;\n}\n",
            encoding="utf-8",
        )
        result = run_cli(
            "edit", str(target), "--replace", "g",
            "--snippet", "//...",
        )
        assert result.returncode != 0
        assert "keep-marker" in result.stderr
        assert target.read_text(encoding="utf-8") == (
            "function g() {\n  var x = 1;\n  var y = 2;\n  return x + y;\n}\n"
        )
