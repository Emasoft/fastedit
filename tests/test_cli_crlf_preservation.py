from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI_MODULE = [sys.executable, "-m", "fastedit"]


def run_cli(*args: str, input_text: str | None = None, env_extra: dict | None = None):
    """Invoke the real fastedit CLI as a subprocess, mirroring test_cli.py's helper."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [*CLI_MODULE, *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


class TestCLIEditPreservesLineEndings:
    """fastedit edit --replace (the deterministic fast path) is byte-exact
    outside the replaced symbol and normalizes the substituted text to the
    file's own prevailing line ending."""

    def test_crlf_file_replace_stays_all_crlf(self, tmp_path: Path) -> None:
        """--replace on a CRLF file keeps untouched lines CRLF and normalizes an LF snippet to CRLF too."""
        original = b"def f():\r\n    return 1\r\n\r\ndef g():\r\n    return 2\r\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli(
            "edit", str(target), "--replace", "g",
            "--snippet", "def g():\n    return 3\n",
        )
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert after.count(b"\r\n") == after.count(b"\n"), (
            "result has a mix of CRLF and bare LF lines: " + repr(after)
        )
        assert after.startswith(b"def f():\r\n    return 1\r\n\r\ndef g():\r\n")
        assert b"return 3" in after
        assert b"return 2" not in after

    def test_mixed_endings_file_replace_keeps_untouched_region_mixed(self, tmp_path: Path) -> None:
        """A file that was already mixed before the edit stays mixed outside the replaced span."""
        original = b"def f():\r\n    return 1\n\ndef g():\r\n    return 2\r\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli(
            "edit", str(target), "--replace", "f",
            "--snippet", "def f():\n    return 99\n",
        )
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        # The untouched tail (def g and its body) must be byte-identical to
        # the original -- the edit's blast radius is the replaced symbol only.
        assert after.endswith(b"\ndef g():\r\n    return 2\r\n")
        assert b"return 99" in after

    def test_bom_is_preserved_when_the_edit_does_not_touch_the_first_line(self, tmp_path: Path) -> None:
        """A UTF-8 BOM on an untouched first line survives an edit byte-exact."""
        original = b"\xef\xbb\xbf# header comment\ndef f():\n    return 1\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli(
            "edit", str(target), "--replace", "f",
            "--snippet", "def f():\n    return 2\n",
        )
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert after.startswith(b"\xef\xbb\xbf# header comment\n"), "BOM was stripped or corrupted by the edit"
        assert b"return 2" in after

    def test_non_ascii_content_survives_a_replace_edit_byte_exact(self, tmp_path: Path) -> None:
        """Non-ASCII UTF-8 content outside the replaced span is untouched byte-for-byte."""
        original = "def greet():\r\n    return \"café ☃ 日本語\"\r\n\r\ndef f():\r\n    return 1\r\n".encode("utf-8")
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli(
            "edit", str(target), "--replace", "f",
            "--snippet", "def f():\n    return 2\n",
        )
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert "café ☃ 日本語".encode("utf-8") in after
        assert b"\r\n" in after
        assert b"return 2" in after

    def test_after_insert_leaves_untouched_crlf_regions_byte_exact(self, tmp_path: Path) -> None:
        """--after (the model-merge path) never rewrites the CRLF/CR endings of lines it does not touch."""
        original = b"def f():\r\n    return 1\r\n\r\ndef g():\r\n    return 2\r\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli(
            "edit", str(target), "--after", "f",
            "--snippet", "def h():\n    return 3\n",
        )
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert after.startswith(b"def f():\r\n    return 1\r\n")
        assert after.endswith(b"def g():\r\n    return 2\r\n")
        assert b"def h" in after


class TestCLIDeletePreservesLineEndings:
    """fastedit delete only removes lines; every remaining line must be byte-exact."""

    def test_crlf_file_delete_keeps_remaining_lines_crlf(self, tmp_path: Path) -> None:
        """Deleting a symbol from a CRLF file leaves the surviving lines CRLF, untranslated."""
        original = b"def f():\r\n    return 1\r\n\r\ndef g():\r\n    return 2\r\n\r\ndef h():\r\n    return 3\r\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli("delete", str(target), "g")
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert b"\n" not in after.replace(b"\r\n", b""), (
            "a bare LF leaked into a CRLF file: " + repr(after)
        )
        assert after.startswith(b"def f():\r\n    return 1\r\n")
        assert after.rstrip(b"\r\n").endswith(b"def h():\r\n    return 3")
        assert b"return 2" not in after

    def test_bom_survives_a_delete(self, tmp_path: Path) -> None:
        """A UTF-8 BOM survives a delete that touches a later symbol."""
        original = b"\xef\xbb\xbfdef f():\n    return 1\n\n\ndef g():\n    return 2\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli("delete", str(target), "g")
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert after.startswith(b"\xef\xbb\xbfdef f():\n")


class TestCLIMovePreservesLineEndings:
    """fastedit move relocates a symbol within one file."""

    def test_crlf_file_move_keeps_untouched_and_moved_lines_crlf(self, tmp_path: Path) -> None:
        """Moving a symbol in a CRLF file never introduces a bare-LF line."""
        original = b"def f():\r\n    return 1\r\n\r\ndef g():\r\n    return 2\r\n\r\ndef h():\r\n    return 3\r\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli("move", str(target), "g", "--after", "h")
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert b"\n" not in after.replace(b"\r\n", b"")
        assert after.count(b"\r\n") == after.count(b"\n")
        assert after.index(b"def h") < after.index(b"def g")

    def test_cr_only_file_move_inserts_cr_separator_not_lf(self, tmp_path: Path) -> None:
        """Moving a symbol in a CR-only file uses a CR separator, not a hardcoded LF."""
        original = b"def f():\r    return 1\r\rdef g():\r    return 2\r\rdef h():\r    return 3\r"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli("move", str(target), "g", "--after", "h")
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert b"\n" not in after, "move_symbol used a hardcoded LF separator on a CR-only file"


class TestCLIRenamePreservesLineEndings:
    """fastedit rename (single file, AST-verified references)."""

    def test_crlf_file_rename_stays_byte_exact_outside_renamed_identifiers(self, tmp_path: Path) -> None:
        """Renaming references in a CRLF file changes only the identifier text, never the line endings."""
        original = b"def f():\r\n    x = 1\r\n    return x\r\n"
        target = tmp_path / "m.py"
        target.write_bytes(original)

        result = run_cli("rename", str(target), "x", "y")
        assert result.returncode == 0, result.stderr

        after = target.read_bytes()
        assert after == b"def f():\r\n    y = 1\r\n    return y\r\n"


class TestCLIRenameAllPreservesLineEndings:
    """fastedit rename-all (AST-verified, across a directory tree)."""

    def test_crlf_files_rename_all_stays_byte_exact_outside_renamed_identifiers(self, tmp_path: Path) -> None:
        """rename-all across two CRLF files never collapses their line endings."""
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        a.write_bytes(b"def use():\r\n    return shared_name(1)\r\n")
        b.write_bytes(b"def shared_name(n):\r\n    return n\r\n")

        result = run_cli("rename-all", str(tmp_path), "shared_name", "renamed_fn")
        assert result.returncode == 0, result.stderr

        assert a.read_bytes() == b"def use():\r\n    return renamed_fn(1)\r\n"
        assert b.read_bytes() == b"def renamed_fn(n):\r\n    return n\r\n"


class TestCLIMoveToFilePreservesLineEndings:
    """fastedit move-to-file (the one verb that genuinely crosses files)."""

    def test_moved_block_is_normalized_to_destination_crlf_convention(self, tmp_path: Path) -> None:
        """An LF-sourced symbol moved into a CRLF destination comes out CRLF, not mixed."""
        src = tmp_path / "src.py"
        dst = tmp_path / "dst.py"
        src.write_bytes(b"def moved():\n    return 1\n")
        dst.write_bytes(b"def existing():\r\n    return 0\r\n")

        result = run_cli("move-to-file", "moved", str(src), str(dst))
        assert result.returncode == 0, result.stderr

        after = dst.read_bytes()
        assert b"\n" not in after.replace(b"\r\n", b""), (
            "moved block kept its source LF endings instead of the destination's CRLF: " + repr(after)
        )
        assert after.startswith(b"def existing():\r\n    return 0\r\n")
        assert b"def moved" in after

    def test_source_file_remaining_content_stays_crlf_byte_exact(self, tmp_path: Path) -> None:
        """The source file's untouched remainder keeps its own CRLF endings after the move."""
        src = tmp_path / "src.py"
        dst = tmp_path / "dst.py"
        src.write_bytes(b"def keep():\r\n    return 0\r\n\r\ndef moved():\r\n    return 1\r\n")
        dst.write_bytes(b"def existing():\n    return 0\n")

        result = run_cli("move-to-file", "moved", str(src), str(dst))
        assert result.returncode == 0, result.stderr

        after = src.read_bytes()
        assert b"\n" not in after.replace(b"\r\n", b""), (
            "a bare LF leaked into the source file's remaining CRLF content: " + repr(after)
        )
        assert after.rstrip(b"\r\n") == b"def keep():\r\n    return 0"
