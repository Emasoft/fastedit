"""Contract tests for scripts/install-dev.sh — the one installer.

Covers both forward modes (fork install pinned to a git ref, --dev editable
install from the working tree), the interactive source menu, branch
autodetect, the model-cache preflight (VALID kept / STALE removed), the
all-grammars prompt/flag, the agent-skill install (Vercel skills CLI, pinned
to the same branch as the package), the revert path, and the dry-run
transcript, all through the script's real CLI.
"""

from __future__ import annotations

import os
import shlex
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "install-dev.sh"
REPO_ROOT = SCRIPT.parent.parent
FORK_URL = "https://github.com/Emasoft/fastedit"


def run(*args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess[str]:
    # stdin=DEVNULL pins the installer to a non-TTY stdin so neither the
    # source menu nor the all-grammars prompt can ever block a test waiting
    # on a human, no matter how pytest itself was launched. The non-TTY path
    # is itself under test: the menu is skipped, all-grammars defaults to yes
    # with a one-line notice, and a stale model cache is removed without
    # asking.
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [str(SCRIPT), *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
        check=False,
        env=env,
    )


def _autodetected_branch() -> str:
    """The branch install-dev.sh autodetects for THIS clone.

    Mirrors the script's `git -C <repo> rev-parse --abbrev-ref HEAD`: a fork
    run from inside a clone with no explicit --ref installs THAT clone's
    current branch. A detached HEAD returns "HEAD" and the script falls back
    to its built-in default; these tests need a real branch.
    """
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    branch = result.stdout.strip()
    assert branch and branch != "HEAD", (
        f"expected the repo to be on a real branch, got {branch!r} ({result.stderr})"
    )
    return branch


def spec_argv(stdout: str, prefix: str) -> str:
    """Find the "+ <prefix><spec>" dry-run line and return the shell-unquoted spec.

    Asserts the spec is copy-pasteable as ONE argv token (shlex.split
    backslash-unescapes it back to the original string) rather than
    checking for a literal quoting style, since printf %q's escaping
    convention can vary. The spec is the LAST token on the line in both
    modes -- `--force --editable` precedes it only in dev mode.
    """
    for line in stdout.splitlines():
        if line.startswith(f"+ {prefix}"):
            tokens = shlex.split(line)
            return tokens[-1]
    raise AssertionError(f"no dry-run line starting with +{prefix!r} in:\n{stdout}")


def _npx_lines(stdout: str) -> list[list[str]]:
    """Every "+ npx ..." transcript line, shell-split into tokens.

    The skill add/remove lines are printed through the same quote_argv as the
    uv lines, so the same shlex round-trip applies: the skill tree URL must
    survive as ONE token (it holds no spaces or quotes, so quote_argv leaves
    it bare).
    """
    lines = []
    for line in stdout.splitlines():
        if line.startswith("+ npx "):
            lines.append(shlex.split(line))
    return lines


def _platform_extras() -> str:
    """The bare platform extras install-dev.sh auto-selects for THIS platform.

    Mirrors detect_backend_extra in the script, which installs EVERY extra this
    platform can actually install (owner directive: a default install must not
    be crippled). "All extras" cannot be literal -- mlx ships no Linux wheels
    and vllm no macOS wheels, so asking for both would fail to resolve on every
    platform. Hence: mlx+mcp on Apple Silicon, vllm+mcp on Linux with an NVIDIA
    driver, mcp alone elsewhere. mcp is unconditional: pure Python, and it is
    the MCP server other tools launch.

    Imports are local and complete on purpose: the Linux branch never executes
    on the machine this is developed on, so a missing module-level import would
    be a NameError that only ever fires on Linux and is invisible here.
    """
    import platform
    import shutil

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx,mcp"
    if platform.system() == "Linux" and shutil.which("nvidia-smi"):
        return "vllm,mcp"
    return "mcp"


def _with_all_grammars(extras: str) -> str:
    """Mirror the script's extras_with_all_grammars: append, dedupe, empty -> all-grammars."""
    if extras == "all-grammars" or extras.endswith(",all-grammars"):
        return extras
    return f"{extras},all-grammars" if extras else "all-grammars"


def _fork_spec(extras: str, ref: str | None = None) -> str:
    """The fork-mode package spec for an extras list (possibly empty).

    ref=None means the AUTODETECTED branch -- what the script installs when it
    runs from inside this clone with no explicit --ref.
    """
    if ref is None:
        ref = _autodetected_branch()
    if extras:
        return f"fastedits[{extras}] @ git+{FORK_URL}@{ref}"
    return f"fastedits @ git+{FORK_URL}@{ref}"


def _run_with_pty_stdin(feed: bytes, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the installer with a pty as stdin, write `feed` as the answers.

    These are REAL full-installer runs (no --dry-run) with a hard 30s
    subprocess budget, so they opt out of the agent-skill axis (--no-skill):
    two live npx calls measure ~7s+ each on this machine and would blow the
    budget for tests whose subject is the source menu and the grammar prompt.
    The skill axis is covered hermetically by TestAgentSkillInstall instead.

    Extra args go through to the script. On a TTY the script asks up to two
    questions IN ORDER -- first the source menu ("Install from: [1/2]", only
    when a fastedit clone sits next to the script), then the all-grammars
    [Y/n] prompt -- so `feed` must carry one line per question to answer
    (e.g. b"2\\nn\\n" = fork install, grammars no). The master side stays open
    until the child exits: closing it early makes the slave's read fail
    instead of delivering buffered input, which would silently flip every
    answer to the Enter default.
    """
    pty = pytest.importorskip("pty")
    master, slave = pty.openpty()
    try:
        proc = subprocess.Popen(
            [str(SCRIPT), "--no-skill", *args],
            stdin=slave,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except BaseException:
        os.close(master)
        os.close(slave)
        raise
    os.close(slave)
    # The pty line discipline buffers this input until the script's read
    # consumes it, so there is no write-before-read race to sleep away.
    os.write(master, feed)
    try:
        out, err = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise
    finally:
        os.close(master)
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def _stub_toolchain(
    tmp_path: Path, with_npx: bool = True, npx_add_fails: bool = False
) -> dict[str, str]:
    """A fake uv/pipx/pip3/fastedit/npx PATH so a REAL (non-dry-run) installer
    run can be exercised hermetically.

    Every mutating tool is a no-op stub and every reporting tool reports
    nothing installed, so a real run touches nothing outside the HOME the test
    hands it (where the model-cache dirs live). The fastedit stub prints the
    fork subcommand names on --help so the postflight verification passes. The
    npx stub answers `skills add` with success and `skills list -g` with an
    installed fastedit skill, so the agent-skill axis is hermetic too;
    with_npx=False drops it entirely to test the npx-missing path (PATH keeps
    only /usr/bin:/bin beyond the stubs, where no node lives on this machine),
    and npx_add_fails makes `skills add` exit non-zero with `skills list -g`
    reporting nothing, to test the failure-tolerant path. PATH is restricted
    to the stubs plus /usr/bin:/bin so the real uv (and the venv's real
    fastedit) on this machine can never be reached.
    """
    fake = tmp_path / "stub-bin"
    fake.mkdir()
    (fake / "uv").write_text(
        "#!/bin/bash\n"
        '[[ "$1" == "--version" ]] && { echo "uv 0.12.12 (stub)"; exit 0; }\n'
        "exit 0\n"
    )
    (fake / "pipx").write_text("#!/bin/bash\nexit 0\n")
    (fake / "pip3").write_text(
        "#!/bin/bash\n"
        '[[ "$1" == "show" ]] && exit 1\n'
        "exit 0\n"
    )
    (fake / "fastedit").write_text(
        "#!/bin/bash\n"
        'if [[ "$1" == "--help" ]]; then\n'
        '  echo "usage: fastedit [command]"\n'
        '  echo "commands:"\n'
        '  echo "  create"\n'
        '  echo "  duplicate"\n'
        '  echo "  split"\n'
        '  echo "  join"\n'
        "  exit 0\n"
        "fi\n"
        "exit 0\n"
    )
    if with_npx:
        # npx args reach the stub as `npx --yes skills <add|list> ...`, so the
        # subcommand is $3, not $1 -- the same --yes the installer passes.
        if npx_add_fails:
            add_body = "  echo 'error: skills add failed (stub)' >&2\n  exit 1\n"
            list_body = "  exit 0\n"
        else:
            add_body = "  exit 0\n"
            list_body = '  echo "  fastedit   (agent skill, global)"\n  exit 0\n'
        (fake / "npx").write_text(
            "#!/bin/bash\n"
            'if [[ "$2" == "skills" && "$3" == "add" ]]; then\n'
            + add_body +
            "fi\n"
            'if [[ "$2" == "skills" && "$3" == "list" ]]; then\n'
            + list_body +
            "fi\n"
            "exit 0\n"
        )
    for stub in fake.iterdir():
        stub.chmod(0o755)
    return {"PATH": f"{fake}:/usr/bin:/bin"}


class TestUvVersionFloor:
    """The preflight warns on a uv too old for bracket extras -- and only then."""

    def _run_with_fake_uv(self, tmp_path: Path, version_line: str):
        """Run the installer with a stub uv that prints `version_line`."""
        fake = tmp_path / "bin"
        fake.mkdir(exist_ok=True)
        stub = fake / "uv"
        stub.write_text(
            "#!/bin/bash\n"
            '[[ "$1" == "--version" ]] && { echo "' + version_line + '"; exit 0; }\n'
            "exit 0\n"
        )
        stub.chmod(0o755)
        env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ["PATH"])
        return subprocess.run(
            ["bash", str(SCRIPT), "--dry-run"],
            capture_output=True, text=True, env=env, timeout=60, check=False,
            stdin=subprocess.DEVNULL,
        )

    def test_old_uv_warns_about_silently_dropped_extras(self, tmp_path: Path) -> None:
        """A sub-0.5 uv must warn: it drops bracket extras SILENTLY, so nothing else would tell you."""
        r = self._run_with_fake_uv(tmp_path, "uv 0.4.30 (x 2024-01-01)")
        assert "older than 0.5" in r.stdout + r.stderr

    def test_modern_uv_is_silent(self, tmp_path: Path) -> None:
        """A current uv must produce no version warning at all."""
        r = self._run_with_fake_uv(tmp_path, "uv 0.12.12 (Homebrew)")
        assert "older than 0.5" not in r.stdout + r.stderr

    def test_unparseable_version_does_not_cry_wolf(self, tmp_path: Path) -> None:
        """A shifted or absent version field must NOT warn.

        MEASURED regression guard. Before the MAJOR.MINOR shape check, a
        `uv --version` whose second field is not the version -- a wrapper or a
        localized build printing "uv version 0.12.12" -- made the comparison
        operate on the literal word, and bash arithmetic treats an unset name as
        0, so both tests passed and a MODERN uv was warned at. A warning that
        cries wolf is worse than none: it teaches the reader to scroll past the
        one line that would have mattered.
        """
        for line in ("uv version 0.12.12", "uv"):
            r = self._run_with_fake_uv(tmp_path, line)
            assert "older than 0.5" not in r.stdout + r.stderr, line


class TestInstallDevScriptExists:
    def test_script_exists_and_is_executable(self) -> None:
        """The installer ships in the repo and is directly runnable."""
        assert SCRIPT.is_file()
        mode = SCRIPT.stat().st_mode
        assert mode & stat.S_IXUSR, "install-dev.sh must be executable"


class TestInstallDevDryRun:
    def test_dry_run_exits_zero_and_shows_default_install(self) -> None:
        """--dry-run succeeds and prints the uninstall sweep plus a platform-matched install spec."""
        result = run("--dry-run")
        assert result.returncode == 0, result.stderr
        assert "uv tool uninstall fastedits" in result.stdout
        # stdin is DEVNULL (non-TTY): the grammar prompt defaults to yes.
        assert spec_argv(result.stdout, "uv tool install ") == _fork_spec(
            _with_all_grammars(_platform_extras())
        )

    def test_dry_run_prints_the_fork_mode_line(self) -> None:
        """Fork mode names itself and the ref it pins to (the autodetected branch)."""
        result = run("--dry-run")
        assert result.returncode == 0, result.stderr
        assert f"fork install (pinned to {_autodetected_branch()})" in result.stdout

    def test_dry_run_sweeps_every_install_method(self) -> None:
        """Uninstall-first sweeps uv tool, pipx and pip so no leftover install collides."""
        result = run("--dry-run")
        assert result.returncode == 0, result.stderr
        assert "uv tool uninstall fastedits" in result.stdout
        assert "pipx uninstall fastedits" in result.stdout
        assert "uninstall -y fastedits" in result.stdout

    def test_dry_run_shows_model_pull_for_this_platform(self) -> None:
        """The forward install picks a model command for this platform in dry-run too."""
        result = run("--dry-run")
        assert result.returncode == 0, result.stderr
        assert "fastedit pull --model" in result.stdout

    def test_no_model_flag_skips_the_model_pull_line(self) -> None:
        """--no-model omits the pull command entirely."""
        result = run("--no-model", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "fastedit pull --model" not in result.stdout

    def test_ref_flag_overrides_default_branch(self) -> None:
        """--ref pins the install to an explicit branch/tag/sha instead of the autodetected one."""
        result = run("--ref", "v1.2.3", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert spec_argv(result.stdout, "uv tool install ") == _fork_spec(
            _with_all_grammars(_platform_extras()), ref="v1.2.3"
        )

    def test_extras_flag_adds_bracketed_extras_to_package_spec(self) -> None:
        """--extras mlx,mcp installs fastedits[mlx,mcp,...] rather than the bare package."""
        result = run("--extras", "mlx,mcp", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert spec_argv(result.stdout, "uv tool install ") == _fork_spec(
            _with_all_grammars("mlx,mcp")
        )

    def test_extras_all_grammars_is_not_duplicated(self) -> None:
        """--extras all-grammars plus a yes answer must not yield all-grammars twice."""
        result = run("--extras", "all-grammars", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert spec_argv(result.stdout, "uv tool install ") == _fork_spec("all-grammars")

    def test_help_lists_the_new_flags(self) -> None:
        """-h documents --dev and --all-grammars."""
        result = run("-h")
        assert result.returncode == 0, result.stderr
        assert "--dev" in result.stdout
        assert "--all-grammars" in result.stdout


class TestBranchAutodetect:
    """Fork mode, no explicit --ref, run from inside a clone -> THAT clone's branch."""

    def test_non_tty_dry_run_autodetects_the_local_branch(self) -> None:
        """Non-TTY default: the menu is skipped and the spec pins the AUTODETECTED branch."""
        branch = _autodetected_branch()
        result = run("--dry-run")
        assert result.returncode == 0, result.stderr
        assert f"branch: installing the local clone's current branch '{branch}' from the fork" in result.stdout
        # No terminal on stdin -> no source menu, no menu prompt anywhere.
        assert "Install from:" not in result.stderr
        # The remote verification prints what it WOULD check; the network call
        # itself is skipped in dry-run so the preview stays hermetic.
        assert f"+ git ls-remote --heads {FORK_URL} {branch}" in result.stdout
        assert (
            "note: dry-run: the fork-remote branch check is skipped (no network); "
            "the real run verifies before installing."
        ) in result.stdout
        assert spec_argv(result.stdout, "uv tool install ") == _fork_spec(
            _with_all_grammars(_platform_extras())
        )

    def test_explicit_ref_skips_autodetect_and_the_remote_check(self) -> None:
        """--ref is the user's pin (may be a tag or sha): no autodetect, no ls-remote check."""
        result = run("--ref", "v1.2.3", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "branch: installing the local clone's current branch" not in result.stdout
        assert "git ls-remote" not in result.stdout
        assert spec_argv(result.stdout, "uv tool install ") == _fork_spec(
            _with_all_grammars(_platform_extras()), ref="v1.2.3"
        )

    def test_dev_skips_autodetect(self) -> None:
        """--dev installs the working tree; there is no branch to detect."""
        result = run("--dev", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "branch: installing the local clone's current branch" not in result.stdout


class TestAllGrammarsFlag:
    def test_all_grammars_no_omits_the_extra(self) -> None:
        """--all-grammars no installs exactly the platform extras -- no grammar pack, no notice."""
        result = run("--all-grammars", "no", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert spec_argv(result.stdout, "uv tool install ") == _fork_spec(_platform_extras())
        assert "all-grammars" not in result.stdout + result.stderr
        assert "non-interactive" not in result.stdout + result.stderr

    def test_all_grammars_yes_includes_the_extra(self) -> None:
        """--all-grammars yes appends the pack to the platform extras without asking."""
        result = run("--all-grammars", "yes", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert spec_argv(result.stdout, "uv tool install ") == _fork_spec(
            _with_all_grammars(_platform_extras())
        )
        assert "non-interactive" not in result.stdout + result.stderr

    def test_non_tty_default_enables_all_grammars_with_a_notice(self) -> None:
        """Piped stdin: no prompt is asked; all grammars default ON, with a notice saying so."""
        result = run("--dry-run")
        assert result.returncode == 0, result.stderr
        assert "non-interactive: all grammars enabled by default" in result.stderr
        assert "pass --all-grammars no to skip" in result.stderr
        assert spec_argv(result.stdout, "uv tool install ") == _fork_spec(
            _with_all_grammars(_platform_extras())
        )

    def test_all_grammars_no_skips_the_model_pull_with_empty_extras(self) -> None:
        """--extras "" --all-grammars no: bare spec, model pull skipped because no backend."""
        result = run("--extras", "", "--all-grammars", "no", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert spec_argv(result.stdout, "uv tool install ") == _fork_spec("")
        assert "fastedit pull --model" not in result.stdout

    def test_all_grammars_rejects_an_invalid_value(self) -> None:
        """--all-grammars only accepts yes|no (y/n aliases included)."""
        result = run("--all-grammars", "maybe")
        assert result.returncode != 0
        assert "yes" in result.stderr and "no" in result.stderr

    def test_all_grammars_rejects_a_missing_value(self) -> None:
        """A valueless --all-grammars is an argument error, not a silently ignored flag."""
        result = run("--all-grammars")
        assert result.returncode != 0


class TestAllGrammarsPrompt:
    """The interactive [Y/n] prompt — needs a real TTY, so driven through a pty.

    On a TTY the source menu is asked FIRST (this repo is a clone), so every
    feed below answers the menu with [2] (fork) before reaching the grammars
    question.
    """

    def test_enter_defaults_to_yes(self) -> None:
        """A bare Enter at the grammars prompt means yes: the grammar pack joins the install."""
        r = _run_with_pty_stdin(b"2\n\n")  # menu -> fork; grammars -> Enter (yes)
        assert r.returncode == 0, r.stderr
        assert "Install from:" in r.stderr
        assert "Install all grammars?" in r.stderr
        assert spec_argv(r.stdout, "uv tool install ") == _fork_spec(
            _with_all_grammars(_platform_extras())
        )

    def test_no_answers_no(self) -> None:
        """Answering n skips the pack."""
        r = _run_with_pty_stdin(b"2\nn\n")  # menu -> fork; grammars -> n
        assert r.returncode == 0, r.stderr
        assert "Install all grammars?" in r.stderr
        assert spec_argv(r.stdout, "uv tool install ") == _fork_spec(_platform_extras())

    def test_explicit_flag_suppresses_the_prompt(self) -> None:
        """--all-grammars yes must never ask, even on a TTY."""
        # The feed answers only the source menu; the grammars question is
        # pre-answered by the flag, so nothing else is read.
        r = _run_with_pty_stdin(b"2\n", "--all-grammars", "yes")
        assert r.returncode == 0, r.stderr
        assert "Install all grammars?" not in r.stdout + r.stderr
        assert spec_argv(r.stdout, "uv tool install ") == _fork_spec(
            _with_all_grammars(_platform_extras())
        )


class TestSourceMenuPrompt:
    """The interactive source menu — [1] local editable tree vs [2] remote fork."""

    def test_menu_pick_1_installs_editable_local_tree(self) -> None:
        """Menu pick 1 -> the editable local-tree spec, exactly like --dev."""
        r = _run_with_pty_stdin(b"1\n", "--all-grammars", "no")
        assert r.returncode == 0, r.stderr
        assert "Install from: [1] local working tree" in r.stderr
        assert "dev install (editable, tracks your working tree)" in r.stdout
        assert spec_argv(r.stdout, "uv tool install --force --editable ") == (
            f"{REPO_ROOT}[{_platform_extras()}]"
        )

    def test_menu_bare_enter_defaults_to_the_local_tree(self) -> None:
        """A bare Enter at the menu takes the detected default: the local tree ([1])."""
        r = _run_with_pty_stdin(b"\n", "--all-grammars", "no")
        assert r.returncode == 0, r.stderr
        assert "dev install (editable, tracks your working tree)" in r.stdout
        assert spec_argv(r.stdout, "uv tool install --force --editable ") == (
            f"{REPO_ROOT}[{_platform_extras()}]"
        )

    def test_menu_pick_2_pins_the_autodetected_branch(self) -> None:
        """Menu pick 2 -> the fork spec, pinned to THIS clone's current branch."""
        branch = _autodetected_branch()
        r = _run_with_pty_stdin(b"2\n", "--all-grammars", "no")
        assert r.returncode == 0, r.stderr
        assert f"branch: installing the local clone's current branch '{branch}' from the fork" in r.stdout
        assert spec_argv(r.stdout, "uv tool install ") == _fork_spec(
            _platform_extras(), ref=branch
        )


class TestModelCachePreflight:
    """Preflight cache detection: VALID caches are kept, STALE (partial) ones removed."""

    def _home_with_cache(self, tmp_path: Path, home_name: str, files: dict[str, str] | None) -> Path:
        """A HOME whose model cache holds one model dir with the given files (None = dir only)."""
        home = tmp_path / home_name
        model_dir = home / ".cache" / "fastedit" / "models" / "some-model"
        model_dir.mkdir(parents=True)
        for fname, content in (files or {}).items():
            (model_dir / fname).write_text(content, encoding="utf-8")
        return home

    def test_dry_run_reports_preflight_headers_and_an_absent_cache(self, tmp_path: Path) -> None:
        """The preflight report prints on a dry run too, and says so when nothing is cached."""
        home = tmp_path / "empty-home"
        home.mkdir()
        result = run("--dry-run", env_extra={"HOME": str(home)})
        assert result.returncode == 0, result.stderr
        assert "preflight: currently installed fastedits (uv tool / pipx / pip):" in result.stdout
        assert f"preflight: model caches under {home}/.cache/fastedit/models:" in result.stdout
        assert "(no cache directory — nothing cached yet)" in result.stdout

    def test_dry_run_labels_stale_and_valid_and_never_removes(self, tmp_path: Path) -> None:
        """A partial download is STALE (would remove), a safetensors-bearing one is VALID (kept)."""
        home = tmp_path / "cache-home"
        models = home / ".cache" / "fastedit" / "models"
        stale = models / "stale-model"
        stale.mkdir(parents=True)
        (stale / "config.json").write_text("{}", encoding="utf-8")
        valid = models / "valid-model"
        valid.mkdir(parents=True)
        (valid / "model.safetensors").write_text("weights", encoding="utf-8")

        result = run("--dry-run", env_extra={"HOME": str(home)})
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "stale-model" in out and "STALE (partial, no *.safetensors)" in out
        assert "valid-model" in out and "VALID" in out
        assert "would remove stale model cache stale-model" in out
        assert "would remove stale model cache valid-model" not in out
        assert "model cache valid-model: kept — shared cache" in out
        # A dry run removes nothing.
        assert stale.exists() and valid.exists()

    def test_default_run_removes_a_stale_cache_without_asking(self, tmp_path: Path) -> None:
        """Non-TTY default: a partial cache is removed WITHOUT a prompt, so the next pull reinstalls cleanly.

        A REAL run (no --dry-run) against the stubbed toolchain: the install
        steps are no-ops, and the explicit --ref keeps the run off the network
        (it skips branch autodetect and the fork-remote verification).
        """
        home = self._home_with_cache(tmp_path, "stale-home", {"config.json": "{}"})
        env = {**_stub_toolchain(tmp_path), "HOME": str(home)}
        result = run("--ref", "feat/create-file", "--no-model", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert "note: non-interactive: removing stale model cache 'some-model'" in result.stderr
        assert "removed stale model cache 'some-model'" in result.stdout
        assert not (home / ".cache" / "fastedit" / "models" / "some-model").exists()

    def test_default_run_keeps_a_valid_cache(self, tmp_path: Path) -> None:
        """Non-TTY default: a cache holding *.safetensors is kept — it is shared with upstream."""
        home = self._home_with_cache(tmp_path, "valid-home", {"model.safetensors": "weights"})
        env = {**_stub_toolchain(tmp_path), "HOME": str(home)}
        result = run("--ref", "feat/create-file", "--no-model", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert "model cache some-model: kept — shared cache" in result.stdout
        assert "removed stale model cache" not in result.stdout
        kept = home / ".cache" / "fastedit" / "models" / "some-model" / "model.safetensors"
        assert kept.exists()

    def test_revert_keeps_every_cache(self, tmp_path: Path) -> None:
        """--revert never touches model weights — not even stale ones."""
        home = self._home_with_cache(tmp_path, "revert-home", {"config.json": "{}"})
        result = run("--revert", "--dry-run", env_extra={"HOME": str(home)})
        assert result.returncode == 0, result.stderr
        assert (
            "model caches: kept — --revert leaves downloaded weights in place (shared with upstream)"
        ) in result.stdout
        assert "would remove" not in result.stdout
        assert (home / ".cache" / "fastedit" / "models" / "some-model").exists()


class TestDevInstall:
    def test_dev_dry_run_spec_is_the_local_repo_path(self) -> None:
        """--dev installs editable from the working tree: path[extras], no git URL."""
        result = run("--dev", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "dev install (editable, tracks your working tree)" in result.stdout
        assert "git+" not in result.stdout
        assert spec_argv(
            result.stdout, "uv tool install --force --editable "
        ) == f"{REPO_ROOT}[{_with_all_grammars(_platform_extras())}]"

    def test_dev_dry_run_uses_force_and_editable(self) -> None:
        """The dev install line carries --force --editable so re-runs refresh in place."""
        result = run("--dev", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "+ uv tool install --force --editable " in result.stdout

    def test_dev_dry_run_with_all_grammars_no(self) -> None:
        """--dev --all-grammars no keeps the platform extras only."""
        result = run("--dev", "--all-grammars", "no", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert spec_argv(
            result.stdout, "uv tool install --force --editable "
        ) == f"{REPO_ROOT}[{_platform_extras()}]"

    def test_dev_with_explicit_extras(self) -> None:
        """--dev --extras honours the explicit list exactly."""
        result = run("--dev", "--extras", "mcp", "--all-grammars", "no", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert spec_argv(
            result.stdout, "uv tool install --force --editable "
        ) == f"{REPO_ROOT}[mcp]"

    def test_dev_dry_run_still_sweeps_and_shows_model_pull(self) -> None:
        """Dev mode keeps the sweep and the model logic — only the spec shape changes."""
        result = run("--dev", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "uv tool uninstall fastedits" in result.stdout
        assert "pipx uninstall fastedits" in result.stdout
        assert "fastedit pull --model" in result.stdout

    def test_dev_ref_combination_warns_and_uses_the_local_path(self) -> None:
        """--ref pins the git spec; in dev mode it has no effect, and the script says so."""
        result = run("--dev", "--ref", "v1.2.3", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "--ref has no effect" in result.stderr
        assert "git+" not in result.stdout
        assert "@v1.2.3" not in result.stdout

    def test_dev_revert_combination_is_rejected(self) -> None:
        """--dev --revert contradicts itself: editable working tree vs upstream PyPI."""
        result = run("--dev", "--revert")
        assert result.returncode != 0


class TestInstallDevRevert:
    def test_revert_dry_run_shows_uninstall_then_pypi_install(self) -> None:
        """--revert undoes the swap: sweep-uninstall the fork, reinstall plain fastedits."""
        result = run("--revert", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "uv tool uninstall fastedits" in result.stdout
        assert spec_argv(result.stdout, "uv tool install ") == "fastedits"

    def test_revert_dry_run_never_mentions_a_git_url(self) -> None:
        """--revert must install from PyPI, never reference the fork's git URL."""
        result = run("--revert", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "git+" not in result.stdout
        assert "git+" not in result.stderr

    def test_revert_dry_run_never_pulls_a_model(self) -> None:
        """--revert never touches model weights — they're shared with the fork."""
        result = run("--revert", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "fastedit pull" not in result.stdout

    def test_revert_never_asks_and_never_adds_grammars(self) -> None:
        """--revert restores upstream: no prompt, no grammar pack."""
        result = run("--revert", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "Install all grammars?" not in result.stdout + result.stderr
        assert "all-grammars" not in result.stdout + result.stderr
        assert "non-interactive" not in result.stdout + result.stderr


class TestAgentSkillInstall:
    """The agent-skill axis: installed with the Vercel skills CLI, pinned to
    the SAME branch as the package, and failure-tolerant exactly like the
    optional backend install -- a skill failure must never leave this machine
    without fastedit, and the postflight reports what actually happened.
    """

    SKILL_TREE_URL = "https://github.com/Emasoft/fastedit/tree/{ref}/skills/fastedit"

    def _add_tokens(self, result: subprocess.CompletedProcess[str]) -> list[str]:
        add = [t for t in _npx_lines(result.stdout) if "add" in t]
        assert len(add) == 1, f"expected exactly one skills-add line in:\n{result.stdout}"
        return add[0]

    def test_dry_run_prints_the_add_command_with_the_autodetected_branch(self) -> None:
        """Default dry run: the skill add names the AUTODETECTED branch in the tree URL,
        global and non-interactive with the Claude Code target."""
        branch = _autodetected_branch()
        result = run("--dry-run")
        assert result.returncode == 0, result.stderr
        tokens = self._add_tokens(result)
        assert tokens[1:6] == [
            "npx", "--yes", "skills", "add", self.SKILL_TREE_URL.format(ref=branch),
        ]
        assert tokens[6:] == ["--skill", "fastedit", "-g", "-a", "claude-code", "-y"]

    def test_skill_url_branch_follows_an_explicit_ref(self) -> None:
        """--ref pins BOTH the package spec and the skill source URL to the same branch."""
        result = run("--ref", "v1.2.3", "--dry-run")
        assert result.returncode == 0, result.stderr
        tokens = self._add_tokens(result)
        assert tokens[5] == self.SKILL_TREE_URL.format(ref="v1.2.3")
        assert spec_argv(result.stdout, "uv tool install ") == _fork_spec(
            _with_all_grammars(_platform_extras()), ref="v1.2.3"
        )

    def test_no_skill_flag_omits_every_skill_command(self) -> None:
        """--no-skill skips the whole axis: no npx invocation is printed at all."""
        result = run("--no-skill", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "npx" not in result.stdout

    def test_forward_run_never_prints_the_remove_command(self) -> None:
        """The remove command belongs to --revert only."""
        result = run("--dry-run")
        assert result.returncode == 0, result.stderr
        assert "skills remove" not in result.stdout + result.stderr

    def test_revert_prints_the_skill_remove_command(self) -> None:
        """--revert additionally removes the globally installed skill, best-effort shape."""
        result = run("--revert", "--dry-run")
        assert result.returncode == 0, result.stderr
        remove = [t for t in _npx_lines(result.stdout) if "remove" in t]
        assert len(remove) == 1, f"expected exactly one skills-remove line in:\n{result.stdout}"
        assert remove[0][1:] == ["npx", "--yes", "skills", "remove", "fastedit", "-g", "-y"]
        assert "skills add" not in result.stdout

    def test_revert_with_no_skill_skips_the_remove(self) -> None:
        """--no-skill skips the whole axis in revert mode too."""
        result = run("--revert", "--no-skill", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "npx" not in result.stdout

    def test_help_documents_no_skill(self) -> None:
        """-h documents --no-skill and the automatic skill install."""
        result = run("-h")
        assert result.returncode == 0, result.stderr
        assert "--no-skill" in result.stdout
        assert "agent skill" in result.stdout

    def test_real_run_installs_and_reports_the_skill(self, tmp_path: Path) -> None:
        """Real run with a stubbed toolchain: the add runs, the postflight reports installed."""
        home = tmp_path / "skill-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path), "HOME": str(home)}
        result = run("--ref", "feat/create-file", "--no-model", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert "installed: agent skill" in result.stdout
        assert "agent skill: installed (Claude Code, global)" in result.stdout

    def test_npx_missing_skips_the_skill_and_says_so(self, tmp_path: Path) -> None:
        """No npx on PATH: the install prints the manual-install note, the postflight
        says skipped -- and the run still exits 0 with the package installed."""
        home = tmp_path / "nonpx-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path, with_npx=False), "HOME": str(home)}
        result = run("--ref", "feat/create-file", "--no-model", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert (
            "npx not found — skipping the agent skill; install manually: "
            "npx skills add Emasoft/fastedit --skill fastedit -g -a claude-code -y"
        ) in result.stdout
        assert "agent skill: skipped (npx not found)" in result.stdout
        assert "verified: fastedit at" in result.stdout

    def test_failed_skill_install_warns_and_the_install_survives(self, tmp_path: Path) -> None:
        """A failed skill install must never fail the installer, and the postflight
        must report NOT FOUND rather than pretend the skill is there."""
        home = tmp_path / "failskill-home"
        home.mkdir()
        env = {**_stub_toolchain(tmp_path, npx_add_fails=True), "HOME": str(home)}
        result = run("--ref", "feat/create-file", "--no-model", env_extra=env)
        assert result.returncode == 0, result.stderr
        assert "warning: the agent skill could not be installed" in result.stderr
        assert "agent skill: NOT FOUND (see warnings above)" in result.stdout
        # The package itself still landed: the fork postflight passed.
        assert "verified: fastedit at" in result.stdout


class TestInstallDevArgumentValidation:
    def test_unknown_flag_exits_non_zero(self) -> None:
        """An unrecognized flag is rejected instead of silently ignored."""
        result = run("--this-flag-does-not-exist")
        assert result.returncode != 0
