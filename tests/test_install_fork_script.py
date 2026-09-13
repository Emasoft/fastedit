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
    """The bracketed extras install-fork.sh auto-selects for THIS platform.

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
        return "[mlx,mcp]"
    if platform.system() == "Linux" and shutil.which("nvidia-smi"):
        return "[vllm,mcp]"
    return "[mcp]"

class TestUvVersionFloor:
    """The preflight warns on a uv too old for bracket extras -- and only then."""

    def _run_with_fake_uv(self, tmp_path: Path, version_line: str):
        """Run the installer with a stub uv that prints `version_line`."""
        import os

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
            capture_output=True, text=True, env=env, timeout=60,
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
