from __future__ import annotations

import stat
import subprocess
from pathlib import Path

import shlex

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "install-fork.sh"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def spec_argv(stdout: str, prefix: str) -> str:
    """Find the "+ <prefix><spec>" dry-run line and return the shell-unquoted spec.

    Asserts the spec is copy-pasteable as ONE argv token (shlex.split
    backslash-unescapes it back to the original string) rather than
    checking for a literal quoting style, since printf %q's escaping
    convention can vary.
    """
    for line in stdout.splitlines():
        if line.startswith(f"+ {prefix}"):
            tokens = shlex.split(line)
            return tokens[-1]
    raise AssertionError(f"no dry-run line starting with +{prefix!r} in:\n{stdout}")



def _default_extras() -> str:
    """The bracketed extras install-fork.sh auto-selects for THIS platform, or "" if none.

    Mirrors detect_backend_extra in the script. The default install carries the
    backend extra matching the model the script will pull, because an
    extras-less install left the 1.7 GB model unloadable -- the first
    model-merge edit died with ModuleNotFoundError. Only Darwin/arm64 -> mlx is
    mapped; every other platform deliberately gets a bare spec, since which
    runtime serves the Linux bf16 model was never verified.
    """
    import platform

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "[mlx]"
    return ""


class TestInstallForkScriptExists:
    def test_script_exists_and_is_executable(self) -> None:
        """The fork-swap script ships in the repo and is directly runnable."""
        assert SCRIPT.is_file()
        mode = SCRIPT.stat().st_mode
        assert mode & stat.S_IXUSR, "install-fork.sh must be executable"


class TestInstallForkDryRun:
    def test_dry_run_exits_zero_and_shows_default_install(self) -> None:
        """--dry-run succeeds and prints the uninstall sweep plus a platform-matched install spec."""
        result = run("--dry-run")
        assert result.returncode == 0, result.stderr
        assert "uv tool uninstall fastedits" in result.stdout
        assert spec_argv(result.stdout, "uv tool install ") == (
            f"fastedits{_default_extras()} @ git+https://github.com/Emasoft/fastedit@feat/create-file"
        )

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
        """--ref pins the install to an explicit branch/tag/sha instead of the default."""
        result = run("--ref", "v1.2.3", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert spec_argv(result.stdout, "uv tool install ") == (
            f"fastedits{_default_extras()} @ git+https://github.com/Emasoft/fastedit@v1.2.3"
        )

    def test_extras_flag_adds_bracketed_extras_to_package_spec(self) -> None:
        """--extras mlx,mcp installs fastedits[mlx,mcp] rather than the bare package."""
        result = run("--extras", "mlx,mcp", "--dry-run")
        assert result.returncode == 0, result.stderr
        assert spec_argv(result.stdout, "uv tool install ") == (
            "fastedits[mlx,mcp] @ git+https://github.com/Emasoft/fastedit@feat/create-file"
        )


class TestInstallForkRevert:
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


class TestInstallForkArgumentValidation:
    def test_unknown_flag_exits_non_zero(self) -> None:
        """An unrecognized flag is rejected instead of silently ignored."""
        result = run("--this-flag-does-not-exist")
        assert result.returncode != 0
