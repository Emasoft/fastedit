"""Accuracy and conciseness tests for the fastedit agent skill.

Contract (skills/fastedit/SKILL.md):
  - The file exists and its YAML frontmatter parses: name == 'fastedit' and a
    description that is present, mentions fastedit, and fits the skills-CLI
    limit of 1024 characters.
  - Every `fastedit ...` command line inside the skill's fenced code blocks is
    accepted by the REAL argparse parser built by fastedit.cli.main() — the
    same no-invented-flags guarantee tests/test_cli_help.py gives the --help
    guide. Parse-only: nothing is executed or written.
  - The file stays <= 200 lines so the skill stays concise enough for an agent
    to load whole.
"""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILL_PATH = PROJECT_ROOT / "skills" / "fastedit" / "SKILL.md"
SKILL_NAME = "fastedit"
MAX_DESCRIPTION_CHARS = 1024
MAX_SKILL_LINES = 200


class _StopParsing(Exception):
    """Sentinel that stops cli.main() right after the parser is built."""


@pytest.fixture(scope="module")
def real_parser():
    """The exact parser object fastedit.cli.main() builds.

    Same capture trick as tests/test_cli_help.py: replace
    argparse.ArgumentParser.parse_args, record `self`, and stop main() before
    any dispatch. The real method is restored afterwards.
    """
    from fastedit import cli

    captured: dict = {}
    original = argparse.ArgumentParser.parse_args

    def _capture(self, *args, **kwargs):
        captured["parser"] = self
        raise _StopParsing

    argparse.ArgumentParser.parse_args = _capture
    try:
        with pytest.raises(_StopParsing):
            cli.main()
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured["parser"]


@pytest.fixture(scope="module")
def skill_text() -> str:
    assert SKILL_PATH.is_file(), f"missing agent skill: {SKILL_PATH}"
    return SKILL_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Minimal frontmatter reader — enough for name/description, no PyYAML.

    Supports `key: value` scalars and folded (`>` / `>-`) or literal (`|` / `|-`)
    multi-line scalars, whose indented continuation lines are joined with spaces
    (folded) — matching how a YAML loader resolves our description block.
    """
    match = re.match(r"\A---\n(.*?)\n---(?:\n|\Z)", text, re.DOTALL)
    assert match, "SKILL.md must open with a --- frontmatter block"
    lines = match.group(1).splitlines()
    fields: dict[str, str] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        key, sep, value = line.partition(":")
        assert sep, f"malformed frontmatter line: {line!r}"
        key, value = key.strip(), value.strip()
        if value in {">", ">-", "|", "|-"}:
            folded: list[str] = []
            index += 1
            while index < len(lines) and lines[index].startswith((" ", "\t")):
                folded.append(lines[index].strip())
                index += 1
            fields[key] = " ".join(folded)
            continue
        fields[key] = value.strip("'\"")
        index += 1
    return fields


def test_frontmatter_name_and_description(skill_text):
    fields = _parse_frontmatter(skill_text)
    assert fields.get("name") == SKILL_NAME
    description = fields.get("description", "")
    assert description, "description frontmatter field is required by the skills CLI"
    assert SKILL_NAME in description
    assert len(description) <= MAX_DESCRIPTION_CHARS, (
        f"description is {len(description)} chars (skills-CLI limit {MAX_DESCRIPTION_CHARS})"
    )


def test_skill_stays_concise(skill_text):
    line_count = len(skill_text.splitlines())
    assert line_count <= MAX_SKILL_LINES, (
        f"SKILL.md is {line_count} lines (conciseness guard: {MAX_SKILL_LINES})"
    )


# ---------------------------------------------------------------------------
# Fenced `fastedit ...` commands vs the real parser
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```")

# Path-shaped tokens are swapped for tmp dummies before parse_args.
# Parse-only either way: nothing is ever executed or written.
_PATH_TOKEN = re.compile(
    r"^(?:src/|tests/|docs/|data/|scripts/|parts/)|\.(?:py|md|json|csv|tsv|toml|txt|rs)$"
)


def _to_tmp_dummy(token: str, tmp_path: Path) -> str:
    if token.startswith("-") or not _PATH_TOKEN.search(token):
        return token
    return str(tmp_path / Path(token).name)


def _extract_fastedit_commands(skill_text: str) -> list[tuple[int, list[str]]]:
    """Every `fastedit ...` command inside the skill's fenced code blocks.

    Shell continuations (a quoted snippet spanning lines) accumulate until
    shlex can split the buffer cleanly; comment and prose lines between
    commands are skipped. Returns (start_line, argv) pairs.
    """
    commands: list[tuple[int, list[str]]] = []
    in_fence = False
    buffer: list[str] = []
    start_line = 0
    for lineno, raw in enumerate(skill_text.splitlines(), 1):
        if _FENCE.match(raw):
            assert not buffer, f"unclosed command continuation at line {lineno}"
            in_fence = not in_fence
            continue
        if not in_fence:
            continue
        stripped = raw.strip()
        if not buffer:
            if not stripped.startswith("fastedit "):
                continue
            buffer = [stripped]
            start_line = lineno
        else:
            buffer.append(raw)
        try:
            argv = shlex.split(" ".join(buffer))
        except ValueError:
            continue  # still inside a multi-line quoted snippet
        assert argv[0] == "fastedit"
        commands.append((start_line, argv))
        buffer = []
    assert not buffer, "unterminated command at the end of a fenced block"
    return commands


def test_skill_commands_match_the_real_parser(real_parser, skill_text, tmp_path):
    """Every fenced `fastedit ...` command must parse against the REAL parser."""
    commands = _extract_fastedit_commands(skill_text)
    assert commands, "no fastedit commands found in fenced blocks — check the skill"
    failures = []
    for line_no, argv in commands:
        parse_argv = [_to_tmp_dummy(token, tmp_path) for token in argv[1:]]
        try:
            real_parser.parse_args(parse_argv)
        except SystemExit as exc:
            failures.append(
                f"line {line_no}: {' '.join(argv)!r} rejected (exit {exc.code})"
            )
    assert not failures, "commands the real parser rejects:\n" + "\n".join(failures)
