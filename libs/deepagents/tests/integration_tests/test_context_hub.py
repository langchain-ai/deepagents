"""Integration tests for ContextHubBackend against a real LangSmith Hub.

Skipped unless ``LANGSMITH_API_KEY`` is set. Each test fixture creates a
uniquely-named throwaway agent repo and deletes it on teardown, so these
tests are safe to run against a real tenant.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.skipif(
    not os.environ.get("LANGSMITH_API_KEY"),
    reason="LANGSMITH_API_KEY not set; skipping Context Hub integration tests.",
)

logger = logging.getLogger(__name__)


@pytest.fixture
def identifier() -> str:
    """Unique throwaway agent-repo handle under the current tenant."""
    return f"-/deepagents-ctx-hub-test-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def backend(identifier: str) -> Iterator:
    """Build a ContextHubBackend and delete the underlying repo on teardown."""
    from langsmith import Client  # noqa: PLC0415

    from deepagents.backends import ContextHubBackend  # noqa: PLC0415

    client = Client()
    yield ContextHubBackend(identifier, client=client)

    try:
        client.delete_agent(identifier)
    except Exception:  # noqa: BLE001 — best-effort cleanup
        logger.warning("Failed to delete test repo %r", identifier, exc_info=True)


def test_lazy_create_on_first_write(backend) -> None:
    """Pulling a non-existent repo returns empty; first write lazily creates it."""
    missing = backend.read("/notes.md")
    assert missing.error == "File '/notes.md' not found"

    write = backend.write("/notes.md", "# hi")
    assert write.error is None
    assert write.path == "/notes.md"

    read = backend.read("/notes.md")
    assert read.error is None
    assert read.file_data is not None
    assert read.file_data["content"] == "# hi"


def test_root_agents_md_is_read_only(backend) -> None:
    """The root /AGENTS.md file is not runtime-editable via ContextHubBackend."""
    write = backend.write("/AGENTS.md", "# attempt to overwrite config")
    assert write.error is not None
    assert "read-only" in write.error

    edit = backend.edit("/AGENTS.md", "foo", "bar")
    assert edit.error is not None
    assert "read-only" in edit.error


def test_round_trip_with_ls_grep_glob_edit(backend) -> None:
    assert backend.write("/a.md", "hello\nworld").error is None
    assert backend.write("/b.md", "hello again").error is None
    assert backend.write("/notes/day1.md", "first note").error is None

    ls_root = backend.ls("/")
    assert ls_root.entries is not None
    root_paths = {e["path"] for e in ls_root.entries}
    assert {"/a.md", "/b.md", "/notes"} <= root_paths

    ls_nested = backend.ls("/notes")
    assert ls_nested.entries is not None
    assert {e["path"] for e in ls_nested.entries} == {"/notes/day1.md"}

    grep = backend.grep("hello")
    assert grep.matches is not None
    assert {m["path"] for m in grep.matches} == {"/a.md", "/b.md"}

    glob = backend.glob("*.md")
    assert glob.matches is not None
    assert {m["path"] for m in glob.matches} >= {"/a.md", "/b.md"}

    edit = backend.edit("/a.md", "world", "earth")
    assert edit.error is None
    assert edit.occurrences == 1

    updated = backend.read("/a.md")
    assert updated.error is None
    assert updated.file_data is not None
    assert "earth" in updated.file_data["content"]


def test_persists_across_backend_instances(backend, identifier) -> None:
    """A fresh ContextHubBackend on the same identifier sees prior writes."""
    from langsmith import Client  # noqa: PLC0415

    from deepagents.backends import ContextHubBackend  # noqa: PLC0415

    assert backend.write("/persist.md", "original").error is None

    second = ContextHubBackend(identifier, client=Client())
    result = second.read("/persist.md")
    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["content"] == "original"


def test_parent_commit_conflict_surfaces_error(backend, identifier) -> None:
    """Concurrent writes against a stale parent_commit should be rejected."""
    from langsmith import Client  # noqa: PLC0415

    from deepagents.backends import ContextHubBackend  # noqa: PLC0415

    assert backend.write("/shared.md", "v0").error is None

    stale = ContextHubBackend(identifier, client=Client())
    stale.read("/shared.md")  # prime stale's commit_hash with current state

    # `backend` advances the repo.
    assert backend.write("/shared.md", "v1").error is None

    # `stale` now has an outdated parent_commit; server rejects.
    result = stale.write("/other.md", "should-fail")
    assert result.error is not None
    assert "Hub unavailable" in result.error


def test_download_files_round_trip(backend) -> None:
    assert backend.write("/blob.txt", "payload").error is None

    responses = backend.download_files(["/blob.txt", "/missing.txt"])
    assert len(responses) == 2
    assert responses[0].content == b"payload"
    assert responses[0].error is None
    assert responses[1].error == "file_not_found"


def test_upload_files_round_trip(backend) -> None:
    responses = backend.upload_files([("/u1.md", b"one"), ("/u2.md", b"two"), ("/bad.bin", b"\x80\xff")])
    assert responses[0].error is None
    assert responses[1].error is None
    assert responses[2].error == "invalid_path"

    assert backend.read("/u1.md").file_data["content"] == "one"
    assert backend.read("/u2.md").file_data["content"] == "two"
