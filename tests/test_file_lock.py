"""Cross-process file locking: two fastedit instances must never co-edit a file.

The bug this closes (audit gap): fastedit's write guards were all IN-PROCESS.
``_atomic_write``'s expected-stat guard (B37) and the MCP ``file_locks``
asyncio locks serialize writers inside one process, but a CLI run and an MCP
server — or two CLIs — editing the same file interleave whole
read-merge-write cycles and the second silently destroys the first's change.

Contract locked down here:

1. Cross-process exclusion: while one PROCESS holds ``acquire_edit_lock``,
   another process's non-blocking acquire raises ``FileLockedError`` naming
   the holder pid, how long it has held the lock, and the target path, and
   ends with the "file unchanged" assurance.
2. Crash safety: the lock is a kernel flock — a holder killed with SIGKILL
   leaves no stale lock; the next acquire succeeds. This is WHY flock beats
   O_EXCL lockfiles (whose creator's death would strand the marker forever).
3. Reentrancy: the SAME process re-acquiring the same path is a no-op
   returning the existing record (multi-edit phases, MCP + CLI in one
   process) — flock between two fds of one process would otherwise
   self-conflict. An inner release must not unlock; only the outermost does.
4. ``FASTEDIT_LOCK_DIR`` relocates the central lock directory (mirroring
   ``FASTEDIT_BACKUP_DIR``); the session conftest fixture points the whole
   suite at a tmp dir so tests never touch the real ``~/.fastedit/locks``.
5. Garbage/stale lock-file content never crashes an acquisition attempt —
   the refusal message just falls back to not naming the holder, and a
   lock file with no holder is acquirable (content is a courtesy stamp, the
   flock is the lock).
6. Batch dead-lock-freedom: two sequential batches over the same two files
   in REVERSE order both complete (per-file non-blocking acquisition in a
   fixed per-batch order cannot deadlock).
7. MCP surface: a locked file yields a clean ``Error: ...`` refusal from the
   fake-mcp tool harness, file unchanged (fast_edit; and per-target refusal
   with partial-batch semantics for fast_multi_edit).
8. CLI surface: ``cmd_edit`` on a locked file exits 1 printing the same
   message on stderr, file unchanged. ``--force`` does not bypass the lock
   (it is a parse-gate opt-out, unrelated).

The concurrent-modification stat guard (B37) is a DIFFERENT failure —
content changed vs lock held — and stays.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pytest

from fastedit.file_lock import (
    FileLockedError,
    acquire_edit_lock,
    lock_file_for,
)
from fastedit.inference.ast_utils import ChunkedMergeResult
from fastedit.mcp import tools_edit
from fastedit.mcp.backup import BackupStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PY_ORIGINAL = "def existing():\n    return 1\n"
PY_MERGED = "def existing():\n    return 2\n"


# ---------------------------------------------------------------------------
# Two-process helpers
# ---------------------------------------------------------------------------

# A holder process: acquires the lock, signals "ready", then holds until a
# release-signal file appears (or 60s — the parent kills it in tests that
# simulate a crash). The release poll is what makes the hold observable.
_HOLDER_SCRIPT = """
import os, sys, time
from fastedit.file_lock import acquire_edit_lock

target, ready, release = sys.argv[1], sys.argv[2], sys.argv[3]
with acquire_edit_lock(target):
    with open(ready, "w") as fh:
        fh.write("ready")
    deadline = time.time() + 60
    while time.time() < deadline and not os.path.exists(release):
        time.sleep(0.02)
"""

_PROBE_SCRIPT = (
    "import sys\n"
    "from fastedit.file_lock import acquire_edit_lock\n"
    "try:\n"
    "    with acquire_edit_lock(sys.argv[1]):\n"
    "        print('acquired')\n"
    "except Exception:\n"
    "    print('refused')\n"
)


def _subprocess_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return env


def _spawn_holder(target: Path, tmp_path: Path) -> subprocess.Popen:
    """Start a real second process holding *target*'s edit lock."""
    ready = tmp_path / "holder.ready"
    release = tmp_path / "holder.release"
    proc = subprocess.Popen(
        [
            sys.executable, "-c", _HOLDER_SCRIPT,
            str(target), str(ready), str(release),
        ],
        cwd=str(PROJECT_ROOT),
        env=_subprocess_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 15
    while not ready.exists() and time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError("holder exited before signalling ready")
        time.sleep(0.02)
    assert ready.exists(), "holder never signalled ready"
    return proc


def _release_holder(proc: subprocess.Popen, tmp_path: Path) -> None:
    (tmp_path / "holder.release").write_text("release", encoding="utf-8")
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:  # pragma: no cover — keeps suite clean
        proc.kill()
        proc.wait(timeout=15)


def _probe_acquire_elsewhere(target: Path) -> bool:
    """True when a FRESH process can acquire *target*'s lock right now."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT, str(target)],
        cwd=str(PROJECT_ROOT),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return "acquired" in result.stdout


# ---------------------------------------------------------------------------
# (a) two-process contention
# ---------------------------------------------------------------------------


class TestCrossProcessContention:
    def test_second_process_refused_naming_the_holder_pid(self, tmp_path):
        target = tmp_path / "contended.py"
        target.write_text(PY_ORIGINAL, encoding="utf-8")

        proc = _spawn_holder(target, tmp_path)
        try:
            with (
                pytest.raises(FileLockedError) as excinfo,
                acquire_edit_lock(target),
            ):
                pass  # pragma: no cover — never reached on conflict
        finally:
            _release_holder(proc, tmp_path)

        message = str(excinfo.value)
        assert f"pid {proc.pid}" in message, message
        assert "running" in message, message  # the holder's age segment
        assert str(target) in message, message
        assert "file unchanged" in message, message
        assert not message.startswith("Traceback"), message

    def test_the_lock_file_records_the_holder(self, tmp_path):
        """The visible stamp (pid/started) is what the refusal message reads."""
        target = tmp_path / "stamped.py"
        target.write_text("x = 1\n", encoding="utf-8")

        with acquire_edit_lock(target) as lock:
            content = lock.lock_file.read_text(encoding="utf-8")
            assert content.startswith(f"pid={os.getpid()}\n"), content
            assert "started=" in content, content


# ---------------------------------------------------------------------------
# (b) release-on-death: the crash-safety property
# ---------------------------------------------------------------------------


class TestReleaseOnDeath:
    def test_sigkilled_holder_leaves_no_stale_lock(self, tmp_path):
        target = tmp_path / "crashed.py"
        target.write_text("x = 1\n", encoding="utf-8")

        proc = _spawn_holder(target, tmp_path)
        proc.kill()  # SIGKILL: no cleanup handlers, no atexit, nothing
        proc.wait(timeout=15)

        # The kernel dropped the flock with the dead process: acquiring
        # succeeds, and the new holder re-stamps the courtesy record.
        with acquire_edit_lock(target) as lock:
            content = lock.lock_file.read_text(encoding="utf-8")
            assert content.startswith(f"pid={os.getpid()}\n"), content


# ---------------------------------------------------------------------------
# (c) same-process reentrancy
# ---------------------------------------------------------------------------


class TestSameProcessReentrancy:
    def test_nested_acquire_of_the_same_path_is_a_no_op(self, tmp_path):
        """Same process, same path: the second acquire reuses the record
        (flock between two fds of one process would self-conflict)."""
        target = tmp_path / "reentrant.py"
        target.write_text("x = 1\n", encoding="utf-8")

        with acquire_edit_lock(target) as outer:
            with acquire_edit_lock(target) as inner:
                assert inner is outer
            # The inner release must NOT have unlocked the file.
            assert _probe_acquire_elsewhere(target) is False
        # Only the outermost release unlocks.
        assert _probe_acquire_elsewhere(target) is True

    def test_released_lock_can_be_reacquired_in_the_same_process(self, tmp_path):
        target = tmp_path / "reacquire.py"
        target.write_text("x = 1\n", encoding="utf-8")

        with acquire_edit_lock(target):
            pass
        with acquire_edit_lock(target) as again:
            assert lock_file_for(target) == again.lock_file


# ---------------------------------------------------------------------------
# (d) FASTEDIT_LOCK_DIR isolation (+ session conftest fixture)
# ---------------------------------------------------------------------------


class TestLockDirOverride:
    def test_lock_files_land_under_the_override_dir(self, tmp_path, monkeypatch):
        lock_root = tmp_path / "locks"
        monkeypatch.setenv("FASTEDIT_LOCK_DIR", str(lock_root))
        target = tmp_path / "isolated.py"
        target.write_text("x = 1\n", encoding="utf-8")

        with acquire_edit_lock(target):
            lock_files = list(lock_root.glob("*.lock"))
            assert len(lock_files) == 1
            assert "started=" in lock_files[0].read_text(encoding="utf-8")
        assert lock_file_for(target).parent == lock_root

    def test_relative_override_is_refused_like_the_backup_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FASTEDIT_LOCK_DIR", "relative/path")
        with pytest.raises(ValueError, match="FASTEDIT_LOCK_DIR"):
            lock_file_for(tmp_path / "whatever.py")

    def test_session_fixture_isolates_the_real_lock_dir(self):
        """conftest points the whole suite at a session tmp dir — the real
        ~/.fastedit/locks is never touched by tests (mirrors the
        FASTEDIT_BACKUP_DIR fixture)."""
        override = os.environ.get("FASTEDIT_LOCK_DIR")
        assert override, "session fixture must set FASTEDIT_LOCK_DIR"
        assert "fastedit-lockdir" in override, override


# ---------------------------------------------------------------------------
# (e) garbage / stale lock-file content
# ---------------------------------------------------------------------------


class TestGarbageLockContent:
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="msvcrt byte-range locks block cross-process writes to the "
        "locked region; the fallback path is POSIX-verifiable here",
    )
    def test_unreadable_holder_stamp_falls_back_without_pid(self, tmp_path):
        """A holder exists but its stamp is garbage: the refusal degrades to
        not naming the holder instead of crashing the attempt."""
        target = tmp_path / "garbage.py"
        target.write_text("x = 1\n", encoding="utf-8")

        proc = _spawn_holder(target, tmp_path)
        try:
            # POSIX: the flock guards acquisition, not plain writes, so the
            # test can simulate a corrupted stamp under a live holder.
            lock_file_for(target).write_bytes(
                b"\xff\xfe not a pid line\n\x00garbage"
            )
            with (
                pytest.raises(FileLockedError) as excinfo,
                acquire_edit_lock(target),
            ):
                pass  # pragma: no cover
        finally:
            _release_holder(proc, tmp_path)

        message = str(excinfo.value)
        assert "another fastedit instance is editing" in message, message
        assert "pid" not in message, message
        assert str(target) in message, message

    def test_stale_garbage_content_does_not_block_acquisition(self, tmp_path):
        """No holder holds the lock: leftover content is irrelevant — the
        acquire succeeds and re-stamps the record."""
        target = tmp_path / "stale.py"
        target.write_text("x = 1\n", encoding="utf-8")
        lock_file = lock_file_for(target)
        lock_file.write_bytes(b"leftover garbage from a pre-flock era\n")

        with acquire_edit_lock(target):
            assert lock_file.read_text(encoding="utf-8").startswith(
                f"pid={os.getpid()}"
            )


# ---------------------------------------------------------------------------
# (f) batch dead-lock-freedom (in-process simulation of multi-edit)
# ---------------------------------------------------------------------------


class TestBatchDeadlockFreedom:
    def test_reverse_order_batches_over_the_same_files_both_complete(
        self, tmp_path,
    ):
        """Per-file locks, non-blocking, acquired in each batch's own fixed
        order: two sequential batches over the same two files in REVERSE
        order both complete — and the multi-edit write phase's per-target
        re-acquire rides the reentrancy registry."""
        first = tmp_path / "a.py"
        first.write_text("x = 1\n", encoding="utf-8")
        second = tmp_path / "b.py"
        second.write_text("x = 2\n", encoding="utf-8")

        for order in ([first, second], [second, first]):
            with contextlib.ExitStack() as batch:
                for path in order:
                    batch.enter_context(acquire_edit_lock(path))
                # The write phase re-acquires each target (no-op via the
                # registry) and releases it again without unlocking.
                for path in order:
                    with acquire_edit_lock(path):
                        pass
                    assert _probe_acquire_elsewhere(path) is False


# ---------------------------------------------------------------------------
# (g) MCP surface: clean refusal via the fake-mcp harness
# ---------------------------------------------------------------------------


class _FakeRequestContext:
    def __init__(self, lifespan_context):
        self.lifespan_context = lifespan_context


class _FakeClientContext:
    def __init__(self, lifespan_context):
        self.request_context = _FakeRequestContext(lifespan_context)


class _FakeEngine:
    def merge_auto(self, *a, **kw):  # pragma: no cover - assertion guard
        raise AssertionError("merge_fn must not be called; the merge is stubbed")


class _FakeBackend:
    @contextlib.asynccontextmanager
    async def acquire(self):
        yield _FakeEngine()


class _FakeMcp:
    def __init__(self, lifespan_context):
        self._lifespan_context = lifespan_context

    def get_context(self):
        return _FakeClientContext(self._lifespan_context)


def _install_fake_mcp(monkeypatch) -> dict:
    """Point ``tools_edit.mcp`` at a fake context; disable the update check."""
    lifespan_context = {
        "backend_kind": "mlx",
        "backend": _FakeBackend(),
        "snapshots": {},
        "backups": BackupStore(),
        "file_locks": defaultdict(asyncio.Lock),
    }
    monkeypatch.setattr(tools_edit, "mcp", _FakeMcp(lifespan_context))
    monkeypatch.setenv("FASTEDIT_NO_UPDATE_CHECK", "1")
    return lifespan_context


def _result(merged_code: str) -> ChunkedMergeResult:
    return ChunkedMergeResult(
        merged_code=merged_code,
        parse_valid=True,
        chunks_used=1,
        chunk_regions=[],
        model_tokens=0,
        latency_ms=0.0,
    )


class TestMcpLockedFileRefusal:
    def test_fast_edit_refuses_cleanly_and_writes_nothing(
        self, tmp_path, monkeypatch,
    ):
        _install_fake_mcp(monkeypatch)
        target = tmp_path / "mcp_locked.py"
        target.write_text(PY_ORIGINAL, encoding="utf-8")

        merge_calls: list = []

        def _must_not_merge(*a, **kw):
            merge_calls.append(kw)
            return _result(PY_MERGED)

        monkeypatch.setattr(tools_edit, "chunked_merge", _must_not_merge)

        proc = _spawn_holder(target, tmp_path)
        try:
            message = asyncio.run(tools_edit.fast_edit(
                file_path=str(target), edit_snippet="x", after="existing",
            ))
        finally:
            _release_holder(proc, tmp_path)

        assert merge_calls == [], "the merge must not run for a locked file"
        assert message.startswith("Error"), message
        assert "another fastedit instance" in message, message
        assert f"pid {proc.pid}" in message, message
        assert "file unchanged" in message, message
        assert target.read_text(encoding="utf-8") == PY_ORIGINAL

    def test_fast_multi_edit_refuses_only_the_locked_target(
        self, tmp_path, monkeypatch,
    ):
        """Partial-batch semantics: the locked target is refused, the free
        target still writes, and the summary says a file was not written."""
        _install_fake_mcp(monkeypatch)
        locked = tmp_path / "locked_target.py"
        locked.write_text(PY_ORIGINAL, encoding="utf-8")
        free = tmp_path / "free_target.py"
        free.write_text(PY_ORIGINAL, encoding="utf-8")

        monkeypatch.setattr(
            tools_edit, "batch_chunked_merge", lambda *a, **kw: _result(PY_MERGED),
        )

        proc = _spawn_holder(locked, tmp_path)
        try:
            message = asyncio.run(tools_edit.fast_multi_edit(
                file_edits=(
                    f'[{{"file_path": "{locked}", "edits": [{{"snippet": "x"}}]}}, '
                    f'{{"file_path": "{free}", "edits": [{{"snippet": "y"}}]}}]'
                ),
            ))
        finally:
            _release_holder(proc, tmp_path)

        assert f"{locked}: another fastedit instance" in message, message
        assert "not written" in message, message
        assert locked.read_text(encoding="utf-8") == PY_ORIGINAL
        assert free.read_text(encoding="utf-8") == PY_MERGED


# ---------------------------------------------------------------------------
# (h) CLI surface: exit 1 with the message
# ---------------------------------------------------------------------------


class TestCliLockedFileExitPath:
    def test_cmd_edit_exits_1_naming_the_holder(self, tmp_path, capsys):
        from fastedit.cli import cmd_edit

        target = tmp_path / "cli_locked.py"
        target.write_text(PY_ORIGINAL, encoding="utf-8")

        proc = _spawn_holder(target, tmp_path)
        try:
            args = argparse.Namespace(
                file=str(target),
                snippet="def existing():\n    return 2\n",
                replace="", after="existing",
                backend=None, model_path=None, api_base=None, api_model=None,
            )
            with pytest.raises(SystemExit) as excinfo:
                cmd_edit(args)
        finally:
            _release_holder(proc, tmp_path)

        assert excinfo.value.code == 1
        stderr = capsys.readouterr().err
        assert "Error: another fastedit instance" in stderr, stderr
        assert f"pid {proc.pid}" in stderr, stderr
        assert "file unchanged" in stderr, stderr
        assert "Traceback" not in stderr, stderr
        assert target.read_text(encoding="utf-8") == PY_ORIGINAL
