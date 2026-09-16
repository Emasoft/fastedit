"""Tests for BackupStore: prune race handling, test-session isolation, and
the Step 19 persistence contract (B38 N-deep + B22 raw bytes).

Step 19 additions lock down:

* **B38** — the old store kept ONE backup per file, so a second edit
  overwrote the first and a corrupted second edit could only be undone one
  step. The store now keeps ``_MAX_BACKUPS_PER_FILE`` timestamped backups
  per file; ``pop`` returns the newest and older steps stay restorable.
* **B22** — the store used to decode/encode text at ``__setitem__``/``pop``.
  It now stores RAW BYTES with no codec anywhere in the store, so a
  latin-1 file's backup round-trips byte-exact.
* **B39** — ``_atomic_write`` fsyncs the file before close and the directory
  after ``os.replace`` (asserted by counting ``os.fsync`` calls, not by
  mocking the OS deeply).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from fastedit.mcp.backup import _MAX_BACKUPS_PER_FILE, BackupStore, _atomic_write


def test_prune_old_survives_a_backup_removed_between_listing_and_unlink(tmp_path, monkeypatch):
    """_prune_old must not raise when a stale backup vanishes mid-loop (concurrent process)."""
    # Bypass __init__ (which resolves the real per-user backup dir unless
    # overridden) so this test exercises the race in _prune_old alone, fully
    # decoupled from where BackupStore's directory normally comes from.
    store = BackupStore.__new__(BackupStore)
    store._dir = tmp_path

    # Two stale backups: one is removed by a "concurrent process" right
    # before we get to it, the other must still be pruned normally.
    raced = tmp_path / "raced.bak"
    raced.write_text("old", encoding="utf-8")
    survivor = tmp_path / "survivor.bak"
    survivor.write_text("old", encoding="utf-8")
    stale_mtime = time.time() - BackupStore._MAX_AGE_SECS - 10
    os.utime(raced, (stale_mtime, stale_mtime))
    os.utime(survivor, (stale_mtime, stale_mtime))

    real_glob = Path.glob

    def racy_glob(self, pattern):
        results = list(real_glob(self, pattern))
        if self == tmp_path and pattern == "*.bak":
            # Simulate another process deleting `raced` right after our
            # listing snapshot was taken, before we get to stat/unlink it.
            raced.unlink()
        return results

    monkeypatch.setattr(Path, "glob", racy_glob)

    store._prune_old()  # must not raise FileNotFoundError

    assert not survivor.exists()


def test_backup_store_root_is_under_tmp_not_the_real_home_store():
    """The test session's BackupStore root must live under a tmp path, never the real per-user store."""
    real_home_store = Path.home() / ".fastedit" / "backups"
    store = BackupStore()
    assert store._dir != real_home_store
    assert "FASTEDIT_BACKUP_DIR" in os.environ
    assert str(store._dir) == os.environ["FASTEDIT_BACKUP_DIR"]


def test_backup_store_rejects_a_relative_backup_dir(monkeypatch):
    """BackupStore.__init__ raises ValueError when FASTEDIT_BACKUP_DIR is not an absolute path."""
    monkeypatch.setenv("FASTEDIT_BACKUP_DIR", "relative/backups")
    with pytest.raises(ValueError, match="absolute path"):
        BackupStore()


# ---------------------------------------------------------------------------
# Step 19 (B38): N-deep timestamped backups
# ---------------------------------------------------------------------------


def test_two_consecutive_stores_are_both_restorable(tmp_path):
    """B38: a second edit must not overwrite the first backup — pop returns
    the newest first, then the previous one, then nothing."""
    store = BackupStore()
    target = tmp_path / "f.py"
    store[str(target)] = b"state one\n"
    store[str(target)] = b"state two\n"

    assert store.pop(str(target)) == b"state two\n"  # newest first
    assert store.pop(str(target)) == b"state one\n"  # the older step survived
    with pytest.raises(KeyError):
        store.pop(str(target))


def test_five_edits_keep_exactly_max_backups_per_file(tmp_path):
    """B38: per-file depth is capped at _MAX_BACKUPS_PER_FILE; older backups
    are pruned on write and newest-first pop order is preserved."""
    store = BackupStore()
    target = tmp_path / "f.py"
    for i in range(8):
        store[str(target)] = f"state {i}\n".encode()

    # Exactly _MAX_BACKUPS_PER_FILE .bak files kept for this one file (the
    # session backup dir is shared, so scope the count to this file).
    assert len(store._key_paths(str(target))) == _MAX_BACKUPS_PER_FILE

    popped = [store.pop(str(target)) for _ in range(_MAX_BACKUPS_PER_FILE)]
    assert popped == [
        b"state 7\n", b"state 6\n", b"state 5\n", b"state 4\n", b"state 3\n",
    ]
    assert str(target) not in store  # undo history fully consumed


def test_peek_returns_the_newest_backup_without_removing_it(tmp_path):
    """The diff commands must look at the newest backup without consuming
    the undo history (peek is non-destructive; pop still sees it)."""
    store = BackupStore()
    target = tmp_path / "f.py"
    store[str(target)] = b"older\n"
    store[str(target)] = b"newest\n"

    assert store.peek(str(target)) == b"newest\n"
    assert store.peek(str(target)) == b"newest\n"  # still there
    assert str(target) in store
    assert store.pop(str(target)) == b"newest\n"


def test_meta_records_the_original_path_and_goes_with_the_last_backup(tmp_path):
    """One <hash>.meta per file (per-backup metas would duplicate the same
    original path N times and need extra prune bookkeeping); it is removed
    only when the file's LAST backup is consumed."""
    store = BackupStore()
    target = tmp_path / "f.py"
    store[str(target)] = b"one\n"
    store[str(target)] = b"two\n"

    meta = store._meta_path(str(target))
    assert meta.read_text(encoding="utf-8") == str(target)

    store.pop(str(target))
    assert meta.exists()  # a backup remains -> undo history exists
    store.pop(str(target))
    assert not meta.exists()  # last one gone -> no undo history at all


# ---------------------------------------------------------------------------
# Step 19 (B22): raw bytes — no decode/encode anywhere in the store
# ---------------------------------------------------------------------------


def test_latin1_backup_round_trips_byte_exact_no_decode_anywhere(tmp_path):
    """B22: a latin-1 file's 0xE9 byte survives the backup round-trip
    byte-exact. A UTF-8 decode would raise on it; errors="replace" would
    corrupt it into EF BF BD; neither may happen in the store."""
    store = BackupStore()
    target = tmp_path / "f.py"
    raw = b"# caf\xe9\nx = 1\n"  # 0xE9 is not valid UTF-8
    store[str(target)] = raw

    assert store.peek(str(target)) == raw
    assert store.pop(str(target)) == raw


def test_atomic_write_stores_the_current_file_as_raw_bytes(tmp_path):
    """B22: _atomic_write's backup of the CURRENT file is the file's raw
    bytes — byte-exact for a latin-1 file, independent of any codec."""
    store = BackupStore()
    target = tmp_path / "f.py"
    raw = b"# caf\xe9\nold = 1\n"
    target.write_bytes(raw)

    _atomic_write(target, "new = 2\n", backups=store, encoding="latin-1")

    assert store.pop(str(target)) == raw


# ---------------------------------------------------------------------------
# Step 19 (B39): fsync before close + best-effort directory fsync
# ---------------------------------------------------------------------------


def test_atomic_write_fsyncs_the_file_and_the_directory(tmp_path, monkeypatch):
    """B39: the written file is fsync'd before close and the directory is
    fsync'd after os.replace. Counted via os.fsync; the directory fd is
    identified by intercepting os.open rather than mocking the OS deeply."""
    real_open = os.open
    real_fsync = os.fsync
    opened_dirs: set[int] = set()
    fsync_targets: list[str] = []

    def fake_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if os.path.isdir(path):
            opened_dirs.add(fd)
        return fd

    def counting_fsync(fd):
        fsync_targets.append("dir" if fd in opened_dirs else "file")
        return real_fsync(fd)

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "fsync", counting_fsync)

    target = tmp_path / "f.py"
    target.write_text("old\n", encoding="utf-8")
    _atomic_write(target, "new\n", expected_stat=None)

    assert fsync_targets.count("file") == 1  # content durability before close
    assert fsync_targets.count("dir") == 1  # rename durability after replace
    assert target.read_text(encoding="utf-8") == "new\n"


def test_directory_fsync_failure_does_not_break_the_write(tmp_path, monkeypatch):
    """B39: the directory fsync is best-effort — a filesystem that refuses
    it must not fail an otherwise good write (content fsync already done)."""
    real_open = os.open
    real_fsync = os.fsync
    opened_dirs: set[int] = set()

    def fake_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if os.path.isdir(path):
            opened_dirs.add(fd)
        return fd

    def refusing_dir_fsync(fd):
        if fd in opened_dirs:
            raise OSError(45, "Operation not supported")  # e.g. some filesystems
        return real_fsync(fd)

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "fsync", refusing_dir_fsync)

    target = tmp_path / "f.py"
    target.write_text("old\n", encoding="utf-8")
    _atomic_write(target, "new\n")  # must not raise

    assert target.read_text(encoding="utf-8") == "new\n"
