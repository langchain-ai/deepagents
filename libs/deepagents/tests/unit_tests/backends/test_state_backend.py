"""Unit tests for StateBackend."""

from typing import Any

import pytest

from deepagents.backends.state import StateBackend


def test_state_backend_raises_outside_graph_context():
    """StateBackend operations outside a graph context should raise RuntimeError."""
    be = StateBackend()
    with pytest.raises(RuntimeError, match="inside a LangGraph graph execution"):
        be.read("/anything.txt")


def test_upload_files_raises_outside_graph_context():
    """upload_files outside a graph context should raise RuntimeError."""
    be = StateBackend()
    with pytest.raises(RuntimeError, match="inside a LangGraph graph execution"):
        be.upload_files([("/hello.txt", b"hello")])


@pytest.mark.parametrize(("offset", "limit"), [(0, 0), (0, -3), (-1, 0)])
def test_state_backend_read_non_positive_limit_returns_empty_read(monkeypatch: pytest.MonkeyPatch, offset: int, limit: int) -> None:
    """`StateBackend.read` inherits the shared clamp rather than raising."""
    backend = StateBackend()
    files = {"/notes.txt": {"content": "one\ntwo\nthree", "encoding": "utf-8"}}
    monkeypatch.setattr(backend, "_read_files", lambda: files)

    result = backend.read("/notes.txt", offset=offset, limit=limit)

    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["content"] == ""
    assert result.start_line is None


def test_state_backend_read_negative_offset_starts_at_first_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """A negative offset is clamped to the start of the file instead of erroring."""
    backend = StateBackend()
    files = {"/notes.txt": {"content": "one\ntwo\nthree", "encoding": "utf-8"}}
    monkeypatch.setattr(backend, "_read_files", lambda: files)

    result = backend.read("/notes.txt", offset=-1, limit=2)

    assert result.error is None
    assert result.start_line == 1
    assert result.end_line == 2


def test_state_backend_reads_legacy_list_content(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = StateBackend()
    legacy_content = ["hello", "world", ""]
    files = {
        "/legacy.txt": {
            "content": legacy_content,
            "created_at": "2025-01-01T00:00:00+00:00",
            "modified_at": "2025-01-02T00:00:00+00:00",
        }
    }
    monkeypatch.setattr(backend, "_read_files", lambda: files)

    read_result = backend.read("/legacy.txt")
    assert read_result.file_data is not None
    assert read_result.file_data["content"] == "hello\nworld\n"
    assert read_result.file_data["encoding"] == "utf-8"

    listing = backend.ls("/").entries
    assert listing is not None
    assert listing[0]["size"] == len("hello\nworld\n")

    matches = backend.grep("world", path="/").matches
    assert matches is not None
    assert matches[0]["text"] == "world"

    glob_matches = backend.glob("*.txt", path="/").matches
    assert glob_matches is not None
    assert glob_matches[0]["size"] == len("hello\nworld\n")

    download = backend.download_files(["/legacy.txt"])[0]
    assert download.content == b"hello\nworld\n"
    assert files["/legacy.txt"]["content"] == legacy_content


def test_state_backend_reads_legacy_list_content_for_non_text_path(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = StateBackend()
    legacy_content = ["aGVsbG8="]
    files = {"/legacy.png": {"content": legacy_content}}
    monkeypatch.setattr(backend, "_read_files", lambda: files)

    read_result = backend.read("/legacy.png")

    assert read_result.file_data is not None
    assert read_result.file_data["content"] == "aGVsbG8="
    assert read_result.file_data["encoding"] == "utf-8"
    assert files["/legacy.png"]["content"] == legacy_content


def test_state_backend_edit_migrates_legacy_list_content(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = StateBackend()
    files = {"/legacy.txt": {"content": ["hello", "world"]}}
    updates: list[dict[str, Any]] = []
    monkeypatch.setattr(backend, "_read_files", lambda: files)
    monkeypatch.setattr(backend, "_send_files_update", updates.append)

    result = backend.edit("/legacy.txt", "world", "there")

    assert result.error is None
    assert updates[0]["/legacy.txt"]["content"] == "hello\nthere"
    assert updates[0]["/legacy.txt"]["encoding"] == "utf-8"
