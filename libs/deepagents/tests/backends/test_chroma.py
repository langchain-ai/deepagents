"""Tests for :class:`deepagents.backends.chroma.ChromaVectorBackend`.

These tests run entirely in-memory using ``chromadb.EphemeralClient`` so
they do not touch disk and do not require an external service.

Two fixtures are provided:

* ``backend``       -- a :class:`ChromaVectorBackend` pre-populated with
                       three documents covering distinct topics.
* ``empty_backend`` -- a :class:`ChromaVectorBackend` over an empty
                       collection.

Every Chroma collection is given a unique name per test (via :mod:`uuid`)
so that tests cannot bleed state into one another.
"""

from __future__ import annotations

import uuid

import pytest

chromadb = pytest.importorskip("chromadb")

from deepagents.backends.chroma import ChromaVectorBackend  # noqa: E402
from deepagents.backends.protocol import (  # noqa: E402
    EditResult,
    FileDownloadResponse,
    FileUploadResponse,
    WriteResult,
)

SEED_DOCUMENTS: list[tuple[str, str]] = [
    (
        "log_db_error",
        "ERROR: database connection refused on host db-primary at 03:14 UTC",
    ),
    (
        "log_user_login",
        "INFO: user alice logged in successfully from 10.0.0.4 with MFA",
    ),
    (
        "log_payment",
        "WARN: payment gateway latency exceeded 2000ms for order 9981",
    ),
]


def _unique_collection_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def backend() -> ChromaVectorBackend:
    """A backend pre-populated with three distinct log documents."""
    client = chromadb.EphemeralClient()
    b = ChromaVectorBackend(
        client=client,
        collection=_unique_collection_name("logs_populated"),
        n_results=5,
    )
    for doc_id, content in SEED_DOCUMENTS:
        b.write(f"/logs/id: {doc_id}", content)
    return b


@pytest.fixture
def empty_backend() -> ChromaVectorBackend:
    """A backend over a freshly created, empty collection."""
    client = chromadb.EphemeralClient()
    return ChromaVectorBackend(
        client=client,
        collection=_unique_collection_name("logs_empty"),
        n_results=5,
    )


def test_read_semantic_query(backend: ChromaVectorBackend) -> None:
    """``query:`` paths return formatted, numbered semantic-search hits."""
    output = backend.read("/logs/query: database connection problem", 0, 0)

    assert isinstance(output, str)
    assert output, "expected non-empty query output"
    first_line = output.splitlines()[0]
    assert first_line.startswith("[1] [score=")
    assert "id=" in first_line
    # The DB-error document should be the top hit for a DB-related query.
    assert "database" in output.lower()


def test_read_by_id(backend: ChromaVectorBackend) -> None:
    """``id:`` paths return the single matching document with score 1.00."""
    output = backend.read("/logs/id: log_user_login", 0, 0)

    assert "alice" in output
    assert "score=1.00" in output
    assert "id=log_user_login" in output


def test_read_missing_id(backend: ChromaVectorBackend) -> None:
    """An unknown id returns an error *string*, never raises."""
    output = backend.read("/logs/id: does_not_exist", 0, 0)

    assert isinstance(output, str)
    assert "not found" in output.lower()
    assert "does_not_exist" in output


def test_read_empty_query(backend: ChromaVectorBackend) -> None:
    """An empty ``query:`` payload returns a helpful error message."""
    output = backend.read("/logs/query: ", 0, 0)

    assert isinstance(output, str)
    assert "empty query" in output.lower()


def test_write_then_read(empty_backend: ChromaVectorBackend) -> None:
    """A doc written via ``write`` is retrievable via ``read('/logs/id: ...')``."""
    result = empty_backend.write(
        "/logs/id: cache_miss",
        "INFO: cache miss for key user:42",
    )

    assert isinstance(result, WriteResult)
    assert result.path == "/logs/id: cache_miss"

    fetched = empty_backend.read("/logs/id: cache_miss", 0, 0)
    assert "cache miss" in fetched
    assert "id=cache_miss" in fetched


def test_write_files_update_none(empty_backend: ChromaVectorBackend) -> None:
    """:class:`WriteResult` must always carry ``files_update=None``."""
    result = empty_backend.write("/logs/id: any_doc", "payload")

    assert isinstance(result, WriteResult)
    assert result.files_update is None


def test_ls_populated(backend: ChromaVectorBackend) -> None:
    """``ls_info`` returns one :class:`FileInfo` per stored document."""
    infos = backend.ls_info("/logs/")

    assert isinstance(infos, list)
    assert len(infos) == len(SEED_DOCUMENTS)
    # FileInfo is a TypedDict -- check structural shape, not isinstance.
    assert all(isinstance(info, dict) and "path" in info for info in infos)
    paths = {info["path"] for info in infos}
    for doc_id, _ in SEED_DOCUMENTS:
        assert f"/logs/id: {doc_id}" in paths


def test_ls_empty(empty_backend: ChromaVectorBackend) -> None:
    """``ls_info`` on an empty collection returns an empty list (not error)."""
    assert empty_backend.ls_info("/logs/") == []


def test_grep_raw_returns_matches(backend: ChromaVectorBackend) -> None:
    """``grep_raw`` returns a *list* of :class:`GrepMatch` for a valid pattern."""
    matches = backend.grep_raw("payment latency", path=None, glob=None)

    assert isinstance(matches, list)
    assert matches, "expected at least one semantic match"
    # GrepMatch is a TypedDict -- check structural shape, not isinstance.
    assert all(isinstance(m, dict) and "path" in m and "line" in m for m in matches)
    # Every match must carry the virtual /logs/id: <id> path convention.
    assert all(m["path"].startswith("/logs/id: ") for m in matches)


def test_edit_replaces_once(empty_backend: ChromaVectorBackend) -> None:
    """``replace_all=False`` replaces only the first occurrence."""
    empty_backend.write("/logs/id: doc1", "foo bar foo bar foo")

    result = empty_backend.edit(
        "/logs/id: doc1", "foo", "QUX", replace_all=False
    )
    assert isinstance(result, EditResult)
    assert result.occurrences == 1
    assert result.files_update is None

    fetched = empty_backend.read("/logs/id: doc1", 0, 0)
    # First 'foo' replaced; remaining two 'foo' tokens survive.
    assert fetched.count("QUX") == 1
    assert fetched.count("foo") == 2

def test_edit_replace_all(empty_backend: ChromaVectorBackend) -> None:
    """``replace_all=True`` replaces every occurrence and reports the count."""
    empty_backend.write("/logs/id: doc2", "foo bar foo bar foo")

    result = empty_backend.edit(
        "/logs/id: doc2", "foo", "QUX", replace_all=True
    )
    assert isinstance(result, EditResult)
    assert result.occurrences == 3
    assert result.files_update is None

    fetched = empty_backend.read("/logs/id: doc2", 0, 0)
    assert fetched.count("QUX") == 3
    assert "foo" not in fetched


def test_edit_missing_doc(empty_backend: ChromaVectorBackend) -> None:
    """Editing a missing doc returns ``occurrences=0`` -- never raises."""
    result = empty_backend.edit(
        "/logs/id: nonexistent", "foo", "bar", replace_all=False
    )

    assert isinstance(result, EditResult)
    assert result.occurrences == 0
    assert result.files_update is None


def test_glob_filters_paths(backend: ChromaVectorBackend) -> None:
    """``glob_info`` filters ``ls_info`` results by an :mod:`fnmatch` pattern."""
    all_infos = backend.glob_info("*", path="/logs/")
    assert len(all_infos) == len(SEED_DOCUMENTS)

    only_log = backend.glob_info("*log_*", path="/logs/")
    assert all(isinstance(info, dict) and "path" in info for info in only_log)
    assert len(only_log) == len(SEED_DOCUMENTS)
    paths = {info["path"] for info in only_log}
    assert "/logs/id: log_db_error" in paths

    only_payment = backend.glob_info("*log_payment*", path="/logs/")
    assert len(only_payment) == 1
    assert only_payment[0]["path"] == "/logs/id: log_payment"

    none_match = backend.glob_info("*does_not_exist*", path="/logs/")
    assert none_match == []


def test_upload_files(empty_backend: ChromaVectorBackend) -> None:
    """``upload_files`` writes each blob; ``ls_info`` reflects the new docs."""
    payload: list[tuple[str, bytes]] = [
        ("/logs/id: upl_one", b"first uploaded document"),
        ("/logs/id: upl_two", b"second uploaded document"),
    ]

    responses = empty_backend.upload_files(payload)

    assert isinstance(responses, list)
    assert len(responses) == 2
    assert all(isinstance(r, FileUploadResponse) for r in responses)
    assert all(r.error is None for r in responses)
    assert {r.path for r in responses} == {
        "/logs/id: upl_one",
        "/logs/id: upl_two",
    }

    paths = {info["path"] for info in empty_backend.ls_info("/logs/")}
    assert "/logs/id: upl_one" in paths
    assert "/logs/id: upl_two" in paths


def test_download_files(backend: ChromaVectorBackend) -> None:
    """``download_files`` returns content for known ids and an error for missing."""
    responses = backend.download_files(
        [
            "/logs/id: log_user_login",
            "/logs/id: missing_doc",
        ]
    )

    assert isinstance(responses, list)
    assert len(responses) == 2
    assert all(isinstance(r, FileDownloadResponse) for r in responses)

    found, missing = responses[0], responses[1]

    assert found.path == "/logs/id: log_user_login"
    assert found.error is None
    assert found.content is not None
    assert "alice" in found.content

    assert missing.path == "/logs/id: missing_doc"
    assert missing.content is None
    assert missing.error == "file_not_found"
