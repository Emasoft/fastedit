"""Standalone backup and atomic write utilities -- no MCP dependency.

BackupStore and _atomic_write are pure filesystem utilities used by both
the MCP server and the CLI. Extracted here so the CLI can import them
without pulling in the `mcp` package.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path

from ..io_utils import UnsupportedEncodingError

logger = logging.getLogger("fastedit.backup")

# B38: how many backups are kept per file. Every write stores a NEW
# timestamped backup and prunes the oldest beyond this depth, so a second
# edit never destroys the first backup and a corrupted edit can be undone
# step by step.
_MAX_BACKUPS_PER_FILE = 5


class ConcurrentModificationError(RuntimeError):
    """B37: the destination changed on disk between the caller's read and
    this write; the write was refused before the destination was touched.

    RuntimeError rather than ValueError/OSError: the content was never bad
    (not a value problem) and nothing failed at the OS level -- this is
    fastedit's own refusal, and callers catch it EXPLICITLY to turn it into
    a clean "re-read and retry" message. Nothing was written; the file on
    disk is exactly as the external writer left it.
    """


class BackupStore:
    """Disk-backed N-deep undo store (B38: ``_MAX_BACKUPS_PER_FILE`` per file).

    Survives server restarts and Escape cancels. Backups live in
    ~/.fastedit/backups/ named ``<sha256-16>-<timestamp>.bak``: a hash of the
    original path plus a nanosecond timestamp, so consecutive edits stack
    instead of overwriting each other (the old 1-deep store reused a single
    name per file and a second edit destroyed the first backup). One
    ``<sha256-16>.meta`` file per ORIGINAL path records that path for
    display; one meta per file (not per backup) is enough because the hash
    is stable per original path -- per-backup metas would duplicate the same
    string N times and need extra bookkeeping to stay in sync with pruning.

    Backups store RAW BYTES (B22): there is no decode or encode anywhere in
    the store, so any file's backup -- latin-1, BOM'd, whatever -- round-trips
    byte-exact. On startup, backups older than 24h are pruned; on every
    write, only the newest ``_MAX_BACKUPS_PER_FILE`` per file are kept.
    """

    _MAX_AGE_SECS = 86400  # 24 hours

    def __init__(self):
        override = os.environ.get("FASTEDIT_BACKUP_DIR")
        if override:
            path = Path(override).expanduser()
            if not path.is_absolute():
                raise ValueError(
                    "FASTEDIT_BACKUP_DIR must be an absolute path "
                    f"(a leading ~ is expanded); got: {override!r}"
                )
            self._dir = path
        else:
            self._dir = Path.home() / ".fastedit" / "backups"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._prune_old()

    def _hash(self, file_path: str) -> str:
        """Stable hash prefix from the original file path."""
        return hashlib.sha256(file_path.encode()).hexdigest()[:16]

    def _key_paths(self, file_path: str) -> list[Path]:
        """All backup files for *file_path*, NEWEST first. Timestamps are
        zero-padded to a fixed width, so lexicographic order is numeric."""
        prefix = self._hash(file_path)
        return sorted(
            self._dir.glob(f"{prefix}-*.bak"),
            key=lambda p: p.name,
            reverse=True,
        )

    def _meta_path(self, file_path: str) -> Path:
        return self._dir / f"{self._hash(file_path)}.meta"

    def __setitem__(self, file_path: str, data: bytes) -> None:
        """Store *data* as a NEW timestamped backup for *file_path*.

        *data* is RAW BYTES and is stored exactly as given (B22): callers
        pass what they read off disk or hold in memory; the store never
        decodes or encodes. Only the newest ``_MAX_BACKUPS_PER_FILE``
        backups per file are kept (B38).
        """
        ts = time.time_ns()
        target = self._dir / f"{self._hash(file_path)}-{ts:020d}.bak"
        while target.exists():
            # Same-nanosecond collision (clock adjustment, two processes):
            # bump the timestamp instead of silently overwriting an existing
            # backup -- that overwrite was the B38 failure mode.
            ts += 1
            target = self._dir / f"{self._hash(file_path)}-{ts:020d}.bak"
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            os.write(fd, data)
            os.close(fd)
            os.replace(tmp, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        # Metadata (original path) so undo/diff can display it.
        self._meta_path(file_path).write_text(file_path, encoding="utf-8")
        self._prune_per_file(file_path)

    def __contains__(self, file_path: str) -> bool:
        return bool(self._key_paths(file_path))

    def peek(self, file_path: str) -> bytes:
        """Return the NEWEST backup for *file_path* WITHOUT removing it.

        Used by the diff commands, which must look at the last backup
        without consuming the undo history. Raises KeyError when no backup
        exists.
        """
        paths = self._key_paths(file_path)
        if not paths:
            raise KeyError(file_path)
        return paths[0].read_bytes()

    def pop(self, file_path: str) -> bytes:
        """Return and remove the NEWEST backup for *file_path* (undo
        semantics: one pop = one step back), as the RAW BYTES stored (B22).

        Raises KeyError when no backup exists.
        """
        paths = self._key_paths(file_path)
        if not paths:
            raise KeyError(file_path)
        newest = paths[0]
        data = newest.read_bytes()
        newest.unlink()
        if len(paths) == 1:
            # Last backup consumed: the file has no undo history left, so
            # its meta goes too (matching __contains__ going False).
            self._meta_path(file_path).unlink(missing_ok=True)
        return data

    def _prune_per_file(self, file_path: str) -> None:
        """Keep only the newest _MAX_BACKUPS_PER_FILE backups (B38)."""
        for stale in self._key_paths(file_path)[_MAX_BACKUPS_PER_FILE:]:
            with contextlib.suppress(OSError):
                stale.unlink()

    def _prune_old(self) -> None:
        """Delete backups older than _MAX_AGE_SECS, and each affected
        file's meta once its last backup is gone."""
        now = time.time()
        pruned = 0
        stale_hashes: set[str] = set()
        for p in self._dir.glob("*.bak"):
            try:
                if now - p.stat().st_mtime > self._MAX_AGE_SECS:
                    p.unlink()
                    # Backup names are "<hash>-<timestamp>.bak": the hash
                    # prefix groups a file's backups for meta cleanup.
                    stale_hashes.add(p.name.split("-")[0])
                    pruned += 1
            except FileNotFoundError:
                # A concurrent fastedit run or parallel test already removed
                # this .bak between our glob() listing and stat()/unlink().
                continue
        for h in stale_hashes:
            if not any(self._dir.glob(f"{h}-*.bak")):
                (self._dir / f"{h}.meta").unlink(missing_ok=True)
        if pruned:
            logger.info("Pruned %d stale backup(s) older than 24h", pruned)


def _refuse_if_changed_on_disk(
    path: Path, expected_stat: os.stat_result | None,
) -> None:
    """B37: raise when *path* no longer matches the stat captured at read
    time. Called immediately before ``os.replace`` -- the narrowest window
    available without kernel-level locking. Size or ``st_mtime_ns`` differing
    means something else wrote the file in the read-to-write window; the
    documented residual TOCTOU is a same-size write landing between this
    stat and the rename itself."""
    if expected_stat is None:
        return
    try:
        current = os.stat(path)
    except FileNotFoundError as e:
        raise ConcurrentModificationError(
            f"{path}: file changed on disk since it was read (it no longer "
            f"exists); re-read and retry. Nothing was written."
        ) from e
    if (current.st_size, current.st_mtime_ns) != (
        expected_stat.st_size, expected_stat.st_mtime_ns,
    ):
        raise ConcurrentModificationError(
            f"{path}: file changed on disk since it was read (size or mtime "
            f"differ); re-read and retry. Nothing was written."
        )


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory fsync after ``os.replace`` (B39): makes the
    rename itself durable, not just the file content. Some filesystems
    refuse directory fsync and Windows cannot open one this way, so any
    OSError is swallowed -- the content is already fsync'd; this only
    hardens the rename."""
    try:
        dfd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


def _atomic_write(
    path: Path,
    content: str | bytes,
    backups=None,
    encoding: str = "utf-8",
    expected_stat: os.stat_result | None = None,
) -> None:
    """Write content to a file atomically via temp file + rename.

    Prevents corrupt files if the process is interrupted mid-write.

    If *backups* is provided and the file already exists, the current file's
    RAW BYTES are stored as a NEW timestamped backup before overwriting
    (B22: no decode anywhere in the backup path, so a latin-1 file's backup
    is byte-exact regardless of codec; B38: consecutive edits stack instead
    of overwriting each other). This enables N-deep undo via ``fast_undo`` /
    ``fastedit undo``. Backups persist to disk (~/.fastedit/backups/).

    *content* may be str (every caller except duplicate/undo) or bytes (used
    by ``fastedit duplicate`` for a byte-for-byte copy that never decodes
    the source, and by undo, which restores the stored raw backup bytes
    verbatim). str content is encoded with *encoding* (B23) -- the SAME
    codec that decoded the file, passed down by every edit verb from its
    ``read_source`` call, so untouched bytes round-trip exactly. The default
    remains "utf-8" for callers that have no read-time codec. bytes content
    is written as-is: no BOM restore, no re-encoding.

    A UTF-8 BOM is a file-level marker, not line content: every write verb
    funnels its final write through here, so restoring a BOM the on-disk
    file already had -- when *content* is str and did not already keep
    it -- is done once, here, rather than in every caller. This matters
    because a merge/replace/delete/move touching the symbol at the very
    top of the file naturally rewrites line 1 as ordinary text and has no
    reason to know a BOM was riding on it; str content is the only case
    this applies to (bytes content is never touched).
    A "-sig" *encoding* supplies its own BOM at encode time; a leading
    U+FEFF still present in the text would encode as a SECOND BOM, so it
    is stripped first (the codec restores the marker).

    *expected_stat* (B37 lost-update guard) is the ``os.stat_result``
    captured from the SAME open the caller read the file through -- see
    ``io_utils.read_source(..., return_stat=True)``. When provided, the
    destination is re-stat'ed immediately BEFORE ``os.replace``: if its size
    or ``st_mtime_ns`` no longer match -- something else wrote the file in
    the read-to-write window -- the write is refused with
    ``ConcurrentModificationError`` and the destination is untouched. A
    destination that no longer exists is a concurrent modification too
    (this write would otherwise resurrect it). ``None`` (the default) skips
    the guard, preserving the behavior for callers with no read-time stat
    (new files, undo).

    Raises:
        UnsupportedEncodingError: str content that *encoding* cannot
            represent. Raised BEFORE any temp file is created and before
            anything is written; the file on disk is untouched.
        ConcurrentModificationError: *expected_stat* was provided and the
            destination changed on disk since it was captured. The temp
            file is cleaned up; the destination is untouched.
    """
    if backups is not None and path.exists():
        # B22: raw bytes -- the store never decodes, so the backup is
        # exactly what is on disk, whatever the codec.
        backups[str(path)] = path.read_bytes()
    if isinstance(content, bytes):
        data = content
    else:
        if encoding.endswith("-sig") and content.startswith("\ufeff"):
            content = content[1:]  # the codec adds the BOM; don't double it
        try:
            data = content.encode(encoding)
        except UnicodeEncodeError as e:
            raise UnsupportedEncodingError(
                f"{path}: content cannot be encoded with {encoding!r} ({e}); "
                f"the file was not modified."
            ) from e
    fd, tmp = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    closed = False
    try:
        if not isinstance(content, bytes) and path.exists():
            try:
                had_bom = path.read_bytes()[:3] == b"\xef\xbb\xbf"
            except OSError:
                had_bom = False
            if had_bom and not data.startswith(b"\xef\xbb\xbf"):
                data = b"\xef\xbb\xbf" + data
        os.write(fd, data)
        # B39: the content must survive a power cut, not just the rename.
        os.fsync(fd)
        closed = True
        os.close(fd)
        # B37: checked here, immediately before the rename, so the
        # read-to-write window is closed as tightly as userspace allows.
        _refuse_if_changed_on_disk(path, expected_stat)
        os.replace(tmp, path)  # atomic on POSIX
        _fsync_directory(path.parent)  # B39: durable rename (best-effort)
    except BaseException:
        if not closed:
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
