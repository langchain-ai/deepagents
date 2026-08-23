"""Tests for the per-project `.env` skip store (`dotenv_skip.py`)."""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest

from deepagents_code.dotenv_skip import (
    is_project_dotenv_skipped,
    reset_session_skips,
    skip_project_dotenv,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clean_session_skips() -> Iterator[None]:
    """Keep the process-global session skip set from leaking between tests."""
    reset_session_skips()
    yield
    reset_session_skips()


def test_skip_persists_under_canonical_key(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    store = tmp_path / "state" / "dotenv_skip.json"

    assert skip_project_dotenv(root / ".", store_path=store)
    assert is_project_dotenv_skipped(root, store_path=store)
    assert not is_project_dotenv_skipped(tmp_path / "other", store_path=store)
    if os.name != "nt":
        assert (store.stat().st_mode & 0o777) == 0o600


def test_skip_key_follows_the_discovered_env_file(tmp_path: Path) -> None:
    """The skip key is the discovered `.env`'s parent, not the launch dir.

    A non-Git project has no `project_root`, so the key must come from walking
    up to the `.env` file. Keying on the launch directory would miss an
    ancestor `.env` when launching from the project root or a sibling.
    """
    from deepagents_code.dotenv_skip import skip_key_for_start_path

    root = tmp_path / "plain"  # no .git — project_root is None
    sub = root / "sub" / "dir"
    sub.mkdir(parents=True)
    (root / ".env").write_text("KEY=val\n", encoding="utf-8")

    # From the file's own directory and from a nested subdirectory, the key is
    # the `.env`'s parent.
    assert skip_key_for_start_path(root) == str(root.resolve())
    assert skip_key_for_start_path(sub) == str(root.resolve())


def test_skip_key_is_none_without_a_project_env(tmp_path: Path) -> None:
    from deepagents_code.dotenv_skip import skip_key_for_start_path

    empty = tmp_path / "empty"
    empty.mkdir()
    assert skip_key_for_start_path(empty) is None


def test_corrupt_store_is_reported_and_never_overwritten(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corrupt store loses skips, so say so on stderr and keep the file.

    The read path deliberately fails *open* — an unreadable store must not
    disable `.env` loading everywhere — so the dropped decision is only
    detectable if the warning is visible. `logger.warning` alone is not:
    the package logger has an in-memory handler and this runs pre-TUI.
    """
    root = tmp_path / "project"
    root.mkdir()
    store = tmp_path / "dotenv_skip.json"
    store.write_text("{ not json", encoding="utf-8")

    # Read path tolerates the corrupt store (treated as not skipped)...
    assert not is_project_dotenv_skipped(root, store_path=store)
    # ...but says so where a pre-TUI user can actually see it.
    assert "Remembered project .env skips are not applied" in capsys.readouterr().err
    # Write path refuses to overwrite a store it could not parse.
    assert not skip_project_dotenv(root, store_path=store)
    assert store.read_text(encoding="utf-8") == "{ not json"


def test_valid_entries_survive_one_invalid_entry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One unusable entry must not discard the rest of the store."""
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    store = tmp_path / "dotenv_skip.json"
    store.write_text(
        json.dumps(
            {
                "version": 1,
                "projects": {
                    str(good.resolve()): {"skipped_at": "2026-01-01T00:00:00+00:00"},
                    str(bad.resolve()): {},
                },
            }
        ),
        encoding="utf-8",
    )

    assert is_project_dotenv_skipped(good, store_path=store)
    assert not is_project_dotenv_skipped(bad, store_path=store)
    warning = capsys.readouterr().err
    assert "is invalid" in warning
    assert "Only this entry is ignored" in warning
    assert "other valid remembered project .env skips are still applied" in warning
    assert "Remembered project .env skips are not applied" not in warning


def test_concurrent_writes_preserve_every_entry(tmp_path: Path) -> None:
    """Read-merge-write under the lock must not drop a competing entry.

    Mirrors `hooks/test_trust.py::test_concurrent_writes_across_stores_preserve
    _every_entry` — the lock is the part of this module most likely to have been
    transcribed wrong from its template.
    """
    store = tmp_path / "state" / "dotenv_skip.json"
    roots = [tmp_path / f"project{index}" for index in range(8)]
    barrier = threading.Barrier(len(roots))

    def _write(root: Path) -> bool:
        barrier.wait()
        return skip_project_dotenv(root, store_path=store)

    with ThreadPoolExecutor(max_workers=len(roots)) as pool:
        assert all(pool.map(_write, roots))

    assert all(is_project_dotenv_skipped(root, store_path=store) for root in roots)


def test_session_skip_is_scoped_to_one_project(tmp_path: Path) -> None:
    """An in-process skip must not leak into a sibling project."""
    from deepagents_code.dotenv_skip import (
        is_project_dotenv_skipped_for_session,
        skip_project_dotenv_for_session,
    )

    skipped = tmp_path / "skipped"
    other = tmp_path / "other"
    skip_project_dotenv_for_session(skipped)

    assert is_project_dotenv_skipped_for_session(skipped)
    # Accepts an equivalent non-canonical path for the same directory.
    assert is_project_dotenv_skipped_for_session(skipped / ".")
    assert not is_project_dotenv_skipped_for_session(other)


def test_unsupported_version_is_ignored_on_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    store = tmp_path / "dotenv_skip.json"
    store.write_text(
        json.dumps({"version": 99, "projects": {str(root): {"skipped_at": "x"}}}),
        encoding="utf-8",
    )

    assert not is_project_dotenv_skipped(root, store_path=store)
    # A version bump silently dropping every skip would be undetectable.
    assert "version 99 is not supported" in capsys.readouterr().err


def test_missing_store_reads_as_empty(tmp_path: Path) -> None:
    assert not is_project_dotenv_skipped(
        tmp_path / "any", store_path=tmp_path / "absent" / "dotenv_skip.json"
    )


def test_session_skip_is_exported_for_child_processes(tmp_path: Path) -> None:
    """A session decision must reach the processes this one starts.

    The server runs in its own interpreter and reloads settings itself, so a
    skip that lived only in client memory would let the server load the very
    `.env` the user just refused.
    """
    from deepagents_code._env_vars import SERVER_DOTENV_SESSION_SKIPS
    from deepagents_code.dotenv_skip import skip_project_dotenv_for_session

    first = tmp_path / "first"
    second = tmp_path / "second"
    skip_project_dotenv_for_session(first)
    skip_project_dotenv_for_session(second)

    exported = json.loads(os.environ[SERVER_DOTENV_SESSION_SKIPS])
    assert exported == sorted([str(first.resolve()), str(second.resolve())])


def test_relayed_session_skip_applies_without_a_local_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relay alone must skip the file, with no in-process prompt answer.

    This is the server's situation: it never ran the prompt, so the env var is
    the only record of the user's answer.
    """
    from deepagents_code._env_vars import SERVER_DOTENV_SESSION_SKIPS
    from deepagents_code.dotenv_skip import is_project_dotenv_skipped_for_session

    skipped = tmp_path / "skipped"
    monkeypatch.setenv(
        SERVER_DOTENV_SESSION_SKIPS, json.dumps([str(skipped.resolve())])
    )

    assert is_project_dotenv_skipped_for_session(skipped)
    assert not is_project_dotenv_skipped_for_session(tmp_path / "other")


def test_relayed_session_skip_reaches_the_dotenv_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relayed key must suppress the real loader, not just the predicate."""
    from deepagents_code._env_vars import SERVER_DOTENV_SESSION_SKIPS
    from deepagents_code.config import _load_dotenv

    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("RELAYED_SKIP_CANARY=leaked\n", encoding="utf-8")
    monkeypatch.delenv("RELAYED_SKIP_CANARY", raising=False)
    monkeypatch.setenv(
        SERVER_DOTENV_SESSION_SKIPS, json.dumps([str(project.resolve())])
    )

    _load_dotenv(start_path=project, refresh_loaded=True)

    assert "RELAYED_SKIP_CANARY" not in os.environ


def test_unusable_relayed_skips_are_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dropping a relayed decision loads the file, so it cannot be silent."""
    from deepagents_code._env_vars import SERVER_DOTENV_SESSION_SKIPS
    from deepagents_code.dotenv_skip import is_project_dotenv_skipped_for_session

    monkeypatch.setenv(SERVER_DOTENV_SESSION_SKIPS, "{not json")

    assert not is_project_dotenv_skipped_for_session("/tmp")
    assert "is not valid JSON" in capsys.readouterr().err


def test_relayed_skips_of_the_wrong_shape_are_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from deepagents_code._env_vars import SERVER_DOTENV_SESSION_SKIPS
    from deepagents_code.dotenv_skip import is_project_dotenv_skipped_for_session

    monkeypatch.setenv(SERVER_DOTENV_SESSION_SKIPS, json.dumps({"a": 1}))

    assert not is_project_dotenv_skipped_for_session("/tmp")
    assert "not a JSON array of strings" in capsys.readouterr().err


def test_project_dotenv_cannot_write_the_session_skip_carrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only this process may write the carrier var (see `_INHERITED_PYTHONPATH_ENV`)."""
    from deepagents_code._env_vars import SERVER_DOTENV_SESSION_SKIPS
    from deepagents_code.config import _load_dotenv

    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(
        f'{SERVER_DOTENV_SESSION_SKIPS}=["/injected"]\n', encoding="utf-8"
    )
    monkeypatch.delenv(SERVER_DOTENV_SESSION_SKIPS, raising=False)

    _load_dotenv(start_path=project, refresh_loaded=True)

    assert SERVER_DOTENV_SESSION_SKIPS not in os.environ
