"""Tests for project extension trust persistence."""

import json
import os
from pathlib import Path

import pytest

from deepagents_code.extensions import (
    is_project_extensions_trusted,
    trust_project_extensions,
)


def test_trust_is_canonical_and_persistent(tmp_path: Path) -> None:
    """Trust should survive aliases of the same canonical project root."""
    project = tmp_path / "project"
    project.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(project, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    store = tmp_path / "state" / "extensions.json"

    assert not is_project_extensions_trusted(project, store_path=store)
    assert trust_project_extensions(alias, store_path=store)
    assert is_project_extensions_trusted(project, store_path=store)

    payload = json.loads(store.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert str(project.resolve()) in payload["projects"]
    if os.name != "nt":
        assert store.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        '{"version": 2, "projects": {}}',
        '{"version": 1, "projects": []}',
    ],
)
def test_malformed_store_fails_closed(tmp_path: Path, payload: str) -> None:
    """Unreadable or unsupported trust data must never grant execution."""
    store = tmp_path / "extensions.json"
    store.write_text(payload, encoding="utf-8")

    assert not is_project_extensions_trusted(tmp_path, store_path=store)
    assert not trust_project_extensions(tmp_path, store_path=store)
    assert store.read_text(encoding="utf-8") == payload


def test_invalid_entry_is_not_trusted(tmp_path: Path) -> None:
    """A project entry without trust metadata should not grant execution."""
    store = tmp_path / "extensions.json"
    store.write_text(
        json.dumps(
            {
                "version": 1,
                "projects": {str(tmp_path.resolve()): {"trusted_at": None}},
            }
        ),
        encoding="utf-8",
    )

    assert not is_project_extensions_trusted(tmp_path, store_path=store)
