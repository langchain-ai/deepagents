"""Tests for the anonymous per-install ID."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from deepagents_code import install_id
from deepagents_code.install_id import INSTALL_ID_FILENAME, get_install_id

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:
    """Reset the per-process cache so each test reads from disk."""
    install_id._cache.clear()
    yield
    install_id._cache.clear()


def test_creates_and_persists_random_id(tmp_path: Path) -> None:
    """A fresh state directory gets a random hex ID written to disk."""
    value = get_install_id(tmp_path)

    assert re.fullmatch(r"[0-9a-f]{32}", value)
    assert (tmp_path / INSTALL_ID_FILENAME).read_text(encoding="utf-8").strip() == (
        value
    )


def test_stable_across_processes(tmp_path: Path) -> None:
    """A new process (empty cache) reads back the same persisted ID."""
    first = get_install_id(tmp_path)
    install_id._cache.clear()

    assert get_install_id(tmp_path) == first


def test_malformed_file_is_replaced(tmp_path: Path) -> None:
    """A corrupt ID file is regenerated instead of sent to the provider."""
    (tmp_path / INSTALL_ID_FILENAME).write_text("not an id\n", encoding="utf-8")

    value = get_install_id(tmp_path)

    assert re.fullmatch(r"[0-9a-f]{32}", value)
    assert (tmp_path / INSTALL_ID_FILENAME).read_text(encoding="utf-8").strip() == (
        value
    )


def test_unwritable_state_dir_falls_back_to_process_id(tmp_path: Path) -> None:
    """If the ID cannot be persisted, one ID is still used for the process."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("blocked", encoding="utf-8")
    state_dir = blocker / ".state"

    value = get_install_id(state_dir)

    assert re.fullmatch(r"[0-9a-f]{32}", value)
    assert get_install_id(state_dir) == value
