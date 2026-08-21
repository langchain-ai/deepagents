"""Tests for the per-project `.env` skip store (`dotenv_skip.py`)."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from deepagents_code.dotenv_skip import (
    is_project_dotenv_skipped,
    skip_project_dotenv,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_skip_persists_under_canonical_key(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    store = tmp_path / "state" / "dotenv_skip.json"

    assert skip_project_dotenv(root / ".", store_path=store)
    assert is_project_dotenv_skipped(root, store_path=store)
    assert not is_project_dotenv_skipped(tmp_path / "other", store_path=store)
    if os.name != "nt":
        assert (store.stat().st_mode & 0o777) == 0o600


def test_subdirectory_of_skipped_root_is_not_independently_keyed(
    tmp_path: Path,
) -> None:
    """The store keys the canonical root; a subdirectory is a different key.

    `_load_dotenv` resolves the project root from the cwd before consulting the
    store, so a root skip covers subdirectories through that resolution — not
    through the store matching a subdirectory key directly.
    """
    root = tmp_path / "project"
    sub = root / "sub" / "dir"
    sub.mkdir(parents=True)
    store = tmp_path / "dotenv_skip.json"

    assert skip_project_dotenv(root, store_path=store)
    assert is_project_dotenv_skipped(root, store_path=store)
    # The bare subdirectory is not itself a stored key.
    assert not is_project_dotenv_skipped(sub, store_path=store)


def test_corrupt_store_fails_closed_without_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    store = tmp_path / "dotenv_skip.json"
    store.write_text("{ not json", encoding="utf-8")

    # Read path tolerates the corrupt store (treated as not skipped).
    assert not is_project_dotenv_skipped(root, store_path=store)
    # Write path refuses to overwrite a store it could not parse.
    assert not skip_project_dotenv(root, store_path=store)
    assert store.read_text(encoding="utf-8") == "{ not json"


def test_unsupported_version_is_ignored_on_read(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    store = tmp_path / "dotenv_skip.json"
    store.write_text(
        json.dumps({"version": 99, "projects": {str(root): {"skipped_at": "x"}}}),
        encoding="utf-8",
    )

    assert not is_project_dotenv_skipped(root, store_path=store)


def test_missing_store_reads_as_empty(tmp_path: Path) -> None:
    assert not is_project_dotenv_skipped(
        tmp_path / "any", store_path=tmp_path / "absent" / "dotenv_skip.json"
    )
