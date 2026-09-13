"""End-to-end CLI tests for `fastedit create` and `fastedit duplicate`.

Runs the real CLI via subprocess (same convention as test_cli.py's
run_cli), never calling cmd_create/cmd_duplicate directly -- these two
subcommands were built but "never exercised end to end" per the phase-1
task, so the whole point is to actually run the binary.
"""

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


class TestCLICreate:
    def test_create_writes_content_from_flag(self, tmp_path: Path) -> None:
        """`create --content` writes exactly that content to a new file."""
        new_file = tmp_path / "hello.py"
        result = run_cli("create", str(new_file), "--content", "x = 1\n")
        assert result.returncode == 0
        assert new_file.read_text(encoding="utf-8") == "x = 1\n"
        assert "Created" in result.stdout

    def test_create_refuses_existing_file_without_force(self, tmp_path: Path) -> None:
        """An existing file is refused with a clear message and exit code 2, no --force."""
        new_file = tmp_path / "hello.py"
        new_file.write_text("old\n", encoding="utf-8")
        result = run_cli("create", str(new_file), "--content", "new\n")
        assert result.returncode == 2
        assert "already exists" in result.stderr
        assert new_file.read_text(encoding="utf-8") == "old\n"

    def test_create_overwrites_existing_file_with_force(self, tmp_path: Path) -> None:
        """--force lets create overwrite an existing file."""
        new_file = tmp_path / "hello.py"
        new_file.write_text("old\n", encoding="utf-8")
        result = run_cli("create", str(new_file), "--content", "new\n", "--force")
        assert result.returncode == 0
        assert new_file.read_text(encoding="utf-8") == "new\n"

    def test_create_refuses_missing_parent_without_parents(self, tmp_path: Path) -> None:
        """A missing parent directory is refused with exit code 1, no --parents."""
        new_file = tmp_path / "sub" / "dir" / "hello.py"
        result = run_cli("create", str(new_file), "--content", "x = 1\n")
        assert result.returncode == 1
        assert "parent directory not found" in result.stderr
        assert not new_file.exists()

    def test_create_makes_missing_parents_with_parents(self, tmp_path: Path) -> None:
        """--parents creates the missing directory chain before writing."""
        new_file = tmp_path / "sub" / "dir" / "hello.py"
        result = run_cli("create", str(new_file), "--content", "x = 1\n", "--parents")
        assert result.returncode == 0
        assert new_file.read_text(encoding="utf-8") == "x = 1\n"

    def test_create_refuses_content_and_content_file_together(self, tmp_path: Path) -> None:
        """--content and --content-file are mutually exclusive."""
        new_file = tmp_path / "hello.py"
        content_file = tmp_path / "content.txt"
        content_file.write_text("x = 1\n", encoding="utf-8")
        result = run_cli(
            "create", str(new_file), "--content", "y = 2\n", "--content-file", str(content_file),
        )
        assert result.returncode == 1
        assert "mutually exclusive" in result.stderr
        assert not new_file.exists()

    def test_create_reads_content_from_file(self, tmp_path: Path) -> None:
        """--content-file <path> reads the file's bytes as the new content."""
        new_file = tmp_path / "hello.py"
        content_file = tmp_path / "content.txt"
        content_file.write_text("x = 1\n", encoding="utf-8")
        result = run_cli("create", str(new_file), "--content-file", str(content_file))
        assert result.returncode == 0
        assert new_file.read_text(encoding="utf-8") == "x = 1\n"

    def test_create_reads_content_file_dash_from_stdin(self, tmp_path: Path) -> None:
        """--content-file - reads content from stdin."""
        new_file = tmp_path / "hello.py"
        result = run_cli("create", str(new_file), "--content-file", "-", input_text="x = 1\n")
        assert result.returncode == 0
        assert new_file.read_text(encoding="utf-8") == "x = 1\n"

    def test_create_reads_bare_stdin_when_no_content_flag_given(self, tmp_path: Path) -> None:
        """With neither --content nor --content-file, create reads stdin directly."""
        new_file = tmp_path / "hello.py"
        result = run_cli("create", str(new_file), input_text="x = 1\n")
        assert result.returncode == 0
        assert new_file.read_text(encoding="utf-8") == "x = 1\n"

    def test_create_refuses_binary_content_file(self, tmp_path: Path) -> None:
        """A NUL-bearing --content-file is refused as binary, with the reason in the message."""
        new_file = tmp_path / "hello.py"
        content_file = tmp_path / "payload.dat"
        content_file.write_bytes(b"abc\x00def")
        result = run_cli("create", str(new_file), "--content-file", str(content_file))
        assert result.returncode == 1
        assert "binary" in result.stderr
        assert "NUL" in result.stderr
        assert not new_file.exists()

    def test_create_reports_symbols_for_a_parseable_language(self, tmp_path: Path) -> None:
        """A supported language's new file gets a symbol report, not just a line count."""
        new_file = tmp_path / "hello.py"
        result = run_cli("create", str(new_file), "--content", "def greet():\n    return 'hi'\n")
        assert result.returncode == 0
        assert "greet" in result.stdout

    def test_create_notes_unsupported_language(self, tmp_path: Path) -> None:
        """An extension fastedit/tldr doesn't parse still creates the file, with a note."""
        new_file = tmp_path / "notes.unsupportedext"
        result = run_cli("create", str(new_file), "--content", "hello\n")
        assert result.returncode == 0
        assert new_file.read_text(encoding="utf-8") == "hello\n"
        assert "no fastedit/tldr language support" in result.stdout


class TestCLIDuplicate:
    def test_duplicate_copies_content_exactly(self, tmp_path: Path) -> None:
        """duplicate writes the source's exact bytes to the destination."""
        source = tmp_path / "a.py"
        source.write_text("def f():\n    return 1\n", encoding="utf-8")
        dest = tmp_path / "b.py"
        result = run_cli("duplicate", str(source), str(dest))
        assert result.returncode == 0
        assert dest.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
        assert "Duplicated" in result.stdout

    def test_duplicate_refuses_missing_source(self, tmp_path: Path) -> None:
        """A source that doesn't exist is refused with a clear message."""
        source = tmp_path / "missing.py"
        dest = tmp_path / "b.py"
        result = run_cli("duplicate", str(source), str(dest))
        assert result.returncode == 1
        assert "source file not found" in result.stderr
        assert not dest.exists()

    def test_duplicate_refuses_binary_source(self, tmp_path: Path) -> None:
        """A binary source (NUL-bearing) is refused before any write happens."""
        source = tmp_path / "payload.dat"
        source.write_bytes(b"abc\x00def")
        dest = tmp_path / "b.dat"
        result = run_cli("duplicate", str(source), str(dest))
        assert result.returncode == 1
        assert "binary" in result.stderr
        assert not dest.exists()

    def test_duplicate_refuses_existing_destination_without_force(self, tmp_path: Path) -> None:
        """An existing destination is refused with exit code 2, no --force."""
        source = tmp_path / "a.py"
        source.write_text("x = 1\n", encoding="utf-8")
        dest = tmp_path / "b.py"
        dest.write_text("old\n", encoding="utf-8")
        result = run_cli("duplicate", str(source), str(dest))
        assert result.returncode == 2
        assert "already exists" in result.stderr
        assert dest.read_text(encoding="utf-8") == "old\n"

    def test_duplicate_overwrites_destination_with_force(self, tmp_path: Path) -> None:
        """--force lets duplicate overwrite an existing destination."""
        source = tmp_path / "a.py"
        source.write_text("x = 1\n", encoding="utf-8")
        dest = tmp_path / "b.py"
        dest.write_text("old\n", encoding="utf-8")
        result = run_cli("duplicate", str(source), str(dest), "--force")
        assert result.returncode == 0
        assert dest.read_text(encoding="utf-8") == "x = 1\n"

    def test_duplicate_refuses_missing_destination_parent_without_parents(self, tmp_path: Path) -> None:
        """A missing destination directory is refused with exit code 1, no --parents."""
        source = tmp_path / "a.py"
        source.write_text("x = 1\n", encoding="utf-8")
        dest = tmp_path / "sub" / "dir" / "b.py"
        result = run_cli("duplicate", str(source), str(dest))
        assert result.returncode == 1
        assert "parent directory not found" in result.stderr
        assert not dest.exists()

    def test_duplicate_makes_missing_destination_parents_with_parents(self, tmp_path: Path) -> None:
        """--parents creates the missing destination directory chain."""
        source = tmp_path / "a.py"
        source.write_text("x = 1\n", encoding="utf-8")
        dest = tmp_path / "sub" / "dir" / "b.py"
        result = run_cli("duplicate", str(source), str(dest), "--parents")
        assert result.returncode == 0
        assert dest.read_text(encoding="utf-8") == "x = 1\n"

    def test_duplicate_reports_symbols_for_a_parseable_language(self, tmp_path: Path) -> None:
        """The duplicated file gets its own symbol report (path-based, not the source's)."""
        source = tmp_path / "a.py"
        source.write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
        dest = tmp_path / "b.py"
        result = run_cli("duplicate", str(source), str(dest))
        assert result.returncode == 0
        assert "greet" in result.stdout
