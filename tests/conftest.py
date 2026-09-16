"""Shared test fixtures.

Isolates BackupStore's on-disk backup directory during the test session so
the suite never writes into (or prunes) the real per-user backup store.

Deliberately NOT done via monkeypatching HOME: several model-dependent
tests locate the local FastEdit model relative to the home directory
(see fastedit.model_download), and changing HOME mid-session can trigger
a ~1.7 GB model re-download. FASTEDIT_BACKUP_DIR is a dedicated override
consumed by BackupStore.__init__ instead, so only the backup store moves.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolated_backup_store(tmp_path_factory):
    """Point every BackupStore at a session-scoped tmp dir, not the real one."""
    # The suite sets FASTEDIT_BACKUP_DIR so tests never write into or prune
    # the real per-user backup store at ~/.fastedit/backups.
    backup_root = tmp_path_factory.mktemp("fastedit-backups")
    previous = os.environ.get("FASTEDIT_BACKUP_DIR")
    os.environ["FASTEDIT_BACKUP_DIR"] = str(backup_root)
    yield backup_root
    if previous is None:
        del os.environ["FASTEDIT_BACKUP_DIR"]
    else:
        os.environ["FASTEDIT_BACKUP_DIR"] = previous
