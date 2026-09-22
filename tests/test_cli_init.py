"""Behavior tests for `fastedit init` — the one-shot agent-skill installer.

Contract (src/fastedit/cli.py::cmd_init):
  - Runs the Vercel skills CLI via npx with exactly this argv:
      npx --yes skills add Emasoft/fastedit --skill fastedit -g -a claude-code -y
    The `Emasoft/fastedit` shorthand resolves the fork's DEFAULT branch, not a
    pinned tag (re-running refreshes the skill). The subprocess is bounded
    (timeout=300) and un-checked (check=False) so cmd_init owns the failure
    reporting.
  - --skill-agent <agent> (default: claude-code) targets another agent; the
    only argv change is the -a value.
  - npx missing: exit 1 with a clean message carrying the manual command —
    an init that did nothing must say so.
  - skills-CLI failure (nonzero exit or timeout): exit 1 with the tool's own
    output shown, plus the manual command. Never a silent no-op.
  - Success: prints the skills-CLI output tail, then
      "Agent skill installed (global, Claude Code). Next: fastedit pull
       --model mlx-8bit (Apple Silicon) · fastedit doctor"
    and points at `fastedit mcp-install` for the optional MCP entry without
    running it.

subprocess.run and shutil.which are stubbed (monkeypatch) throughout — no
test touches npx, the network, or the user's global agent config.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastedit import cli as cli_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_ARGV = [
    "npx", "--yes", "skills", "add", "Emasoft/fastedit",
    "--skill", "fastedit", "-g", "-a", "claude-code", "-y",
]
MANUAL_COMMAND = " ".join(EXPECTED_ARGV)

SKILLS_STDOUT = (
    "sSkills: resolving Emasoft/fastedit (default branch)\n"
    "  installed fastedit (skill) -> ~/.claude/skills\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_which(monkeypatch, result="/usr/local/bin/npx") -> list[str]:
    """Stub shutil.which; records every binary name looked up."""
    looked_up: list[str] = []

    def fake_which(name):
        looked_up.append(name)
        return result

    monkeypatch.setattr(shutil, "which", fake_which)
    return looked_up


def _stub_run(
    monkeypatch,
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> list[tuple[list[str], dict]]:
    """Stub subprocess.run globally (same seam test_fast_edit_impact uses)."""
    calls: list[tuple[list[str], dict]] = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _run_init_inprocess(monkeypatch, *extra_args: str) -> None:
    """Run `fastedit init ...` through cli.main() with the update check off."""
    monkeypatch.setenv("FASTEDIT_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(sys, "argv", ["fastedit", "init", *extra_args])
    cli_module.main()


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run `python -m fastedit <args>` (for --help rendering checks only)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "fastedit", *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )


class _StopParsing(Exception):
    """Sentinel that stops cli.main() right after the parser is built."""


@pytest.fixture(scope="module")
def real_parser():
    """The exact parser object fastedit.cli.main() builds (test_cli_help trick)."""
    captured: dict = {}
    original = argparse.ArgumentParser.parse_args

    def _capture(self, *args, **kwargs):
        captured["parser"] = self
        raise _StopParsing

    argparse.ArgumentParser.parse_args = _capture
    try:
        with pytest.raises(_StopParsing):
            cli_module.main()
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured["parser"]


# ---------------------------------------------------------------------------
# Argparse registration
# ---------------------------------------------------------------------------

class TestInitArgparse:
    def test_init_subcommand_registered(self, real_parser):
        for action in real_parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                assert "init" in action.choices
                break
        else:
            raise AssertionError("main parser has no subparsers action")

    def test_init_defaults_to_claude_code(self, real_parser):
        args = real_parser.parse_args(["init"])
        assert args.skill_agent == "claude-code"

    def test_init_help_names_the_skill_agent_flag(self):
        result = run_cli("init", "--help")
        assert result.returncode == 0
        assert "--skill-agent" in result.stdout
        assert "claude-code" in result.stdout

    def test_init_in_top_level_help(self):
        result = run_cli("--help")
        assert result.returncode == 0
        assert "init" in result.stdout


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestInitInstallsTheSkill:
    def test_runs_the_skills_cli_with_the_documented_argv(self, monkeypatch, capsys):
        looked_up = _stub_which(monkeypatch)
        calls = _stub_run(monkeypatch, stdout=SKILLS_STDOUT)

        _run_init_inprocess(monkeypatch)

        assert "npx" in looked_up
        assert len(calls) == 1
        argv, kwargs = calls[0]
        assert argv == EXPECTED_ARGV
        assert "-g" in argv and "-y" in argv
        assert "--skill" in argv and "fastedit" in argv
        assert argv[argv.index("-a") + 1] == "claude-code"
        # Bounded, un-checked: cmd_init owns the failure reporting.
        assert kwargs["timeout"] == 300
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True

    def test_success_prints_output_tail_and_next_steps(self, monkeypatch, capsys):
        _stub_which(monkeypatch)
        calls = _stub_run(monkeypatch, stdout=SKILLS_STDOUT)

        _run_init_inprocess(monkeypatch)  # no SystemExit -> exit 0

        out = capsys.readouterr().out
        assert "installed fastedit (skill)" in out, out  # skills-CLI output tail
        assert "Agent skill installed (global, Claude Code)" in out, out
        assert "Next: fastedit pull --model mlx-8bit (Apple Silicon)" in out, out
        assert "fastedit doctor" in out, out
        # MCP is pointed at in the guidance text, never run by init.
        assert "fastedit mcp-install" in out, out
        assert len(calls) == 1

    def test_skill_agent_flag_only_changes_the_agent_value(self, monkeypatch, capsys):
        _stub_which(monkeypatch)
        calls = _stub_run(monkeypatch, stdout=SKILLS_STDOUT)

        _run_init_inprocess(monkeypatch, "--skill-agent", "codex")

        out = capsys.readouterr().out
        argv, _kwargs = calls[0]
        assert argv == [
            "npx", "--yes", "skills", "add", "Emasoft/fastedit",
            "--skill", "fastedit", "-g", "-a", "codex", "-y",
        ]
        assert "Agent skill installed (global, codex)" in out, out


# ---------------------------------------------------------------------------
# Failure paths — both fail loud with exit 1
# ---------------------------------------------------------------------------

class TestInitFailurePaths:
    def test_npx_missing_exits_1_with_the_manual_command(self, monkeypatch, capsys):
        _stub_which(monkeypatch, result=None)
        calls = _stub_run(monkeypatch)

        with pytest.raises(SystemExit) as excinfo:
            _run_init_inprocess(monkeypatch)

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "npx" in err
        assert "NOT installed" in err
        assert MANUAL_COMMAND in err
        assert calls == []  # nothing was executed

    def test_skills_cli_failure_exits_1_and_shows_the_output(self, monkeypatch, capsys):
        _stub_which(monkeypatch)
        _stub_run(
            monkeypatch,
            returncode=1,
            stdout="npm ERR! missing peer dependency",
            stderr="npm ERR! code ELIFECYCLE",
        )

        with pytest.raises(SystemExit) as excinfo:
            _run_init_inprocess(monkeypatch)

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "npm ERR! missing peer dependency" in err, err  # output shown
        assert "npm ERR! code ELIFECYCLE" in err, err
        assert "NOT installed" in err
        assert MANUAL_COMMAND in err

    def test_skills_cli_timeout_exits_1_with_guidance(self, monkeypatch, capsys):
        _stub_which(monkeypatch)

        def fake_run(argv, **kwargs):
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs["timeout"])

        monkeypatch.setattr(subprocess, "run", fake_run)

        with pytest.raises(SystemExit) as excinfo:
            _run_init_inprocess(monkeypatch)

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "timed out" in err
        assert "NOT installed" in err
        assert MANUAL_COMMAND in err
