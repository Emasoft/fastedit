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

# Fixture re-exports: the real-LLM engine fixture LIVES in llm_fixtures.py
# (implementation plan Phase A) and the MCP full-stack harness fixture lives
# in test_stress_100mb_llm.py (Step C2, reused by C3's seams suite). Importing
# them here puts them in conftest's fixture namespace so every llm-tier test
# can request `real_engine`/`mcp_harness` by name without re-importing them
# (a per-module import would shadow-collide with the test functions' fixture
# parameters — ruff F811, and the same collision the C2 module hit).
from llm_fixtures import real_engine  # noqa: F401 -- fixture discovery re-export
from test_stress_100mb_llm import (
    mcp_harness,  # noqa: F401 -- fixture discovery re-export
)


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
