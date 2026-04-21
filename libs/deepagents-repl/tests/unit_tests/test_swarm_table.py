"""Tests for ``create_table`` — materialising a JSONL table from sources."""

from __future__ import annotations

import json
from typing import Any

import pytest
from deepagents.backends.protocol import BackendProtocol, GlobResult

from deepagents_repl._swarm.table import create_table
from deepagents_repl._swarm.types import CreateTableSource


class _StubBackend(BackendProtocol):
    """Backend that only implements aglob (enough for create_table tests)."""

    def __init__(self, files: dict[str, list[str]] | None = None) -> None:
        self._files = files or {}
        self._errors: dict[str, str] = {}

    def set_glob_error(self, pattern: str, error: str) -> None:
        self._errors[pattern] = error

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        if pattern in self._errors:
            return GlobResult(error=self._errors[pattern])
        matches = self._files.get(pattern, [])
        return GlobResult(matches=[{"path": p} for p in matches])

    # Abstract methods not exercised here.
    def ls(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def read(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def write(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def edit(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def grep(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def glob(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def upload_files(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def download_files(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def _captured_write() -> tuple[dict[str, str], Any]:
    captured: dict[str, str] = {}

    def write(path: str, content: str) -> None:
        captured[path] = content

    return captured, write


async def test_creates_table_from_glob() -> None:
    backend = _StubBackend(files={"data/*.txt": ["data/a.txt", "data/b.txt"]})
    captured, write = _captured_write()
    await create_table(
        "/out.jsonl",
        CreateTableSource(glob="data/*.txt"),
        backend,
        write,
    )
    rows = [json.loads(line) for line in captured["/out.jsonl"].strip().split("\n")]
    assert rows == [
        {"id": "a.txt", "file": "data/a.txt"},
        {"id": "b.txt", "file": "data/b.txt"},
    ]


async def test_creates_table_from_file_paths() -> None:
    captured, write = _captured_write()
    await create_table(
        "/out.jsonl",
        CreateTableSource(file_paths=["alpha.md", "beta.md"]),
        _StubBackend(),
        write,
    )
    rows = [json.loads(line) for line in captured["/out.jsonl"].strip().split("\n")]
    assert rows == [
        {"id": "alpha.md", "file": "alpha.md"},
        {"id": "beta.md", "file": "beta.md"},
    ]


async def test_disambiguates_basename_collisions() -> None:
    captured, write = _captured_write()
    await create_table(
        "/out.jsonl",
        CreateTableSource(file_paths=["en/readme.md", "fr/readme.md"]),
        _StubBackend(),
        write,
    )
    rows = [json.loads(line) for line in captured["/out.jsonl"].strip().split("\n")]
    ids = {r["id"] for r in rows}
    assert ids == {"en-readme.md", "fr-readme.md"}


async def test_passes_through_inline_tasks() -> None:
    captured, write = _captured_write()
    tasks = [
        {"id": "one", "text": "first", "score": 1},
        {"id": "two", "text": "second", "score": 2},
    ]
    await create_table(
        "/out.jsonl",
        CreateTableSource(tasks=tasks),
        _StubBackend(),
        write,
    )
    rows = [json.loads(line) for line in captured["/out.jsonl"].strip().split("\n")]
    assert rows == tasks


async def test_rejects_tasks_missing_id() -> None:
    bad_tasks = [{"id": "ok"}, {"text": "no id"}, {"id": 42}]
    _, write = _captured_write()
    with pytest.raises(ValueError, match="tasks at index 1, 2"):
        await create_table(
            "/out.jsonl",
            CreateTableSource(tasks=bad_tasks),
            _StubBackend(),
            write,
        )


async def test_raises_when_source_empty() -> None:
    _, write = _captured_write()
    with pytest.raises(ValueError, match="at least one of"):
        await create_table("/out.jsonl", CreateTableSource(), _StubBackend(), write)


async def test_raises_when_no_matches() -> None:
    _, write = _captured_write()
    with pytest.raises(ValueError, match="No files matched"):
        await create_table(
            "/out.jsonl",
            CreateTableSource(glob="nothing/*.txt"),
            _StubBackend(),
            write,
        )


async def test_glob_error_propagates() -> None:
    backend = _StubBackend()
    backend.set_glob_error("secret/*", "permission denied")
    _, write = _captured_write()
    with pytest.raises(ValueError, match="permission denied"):
        await create_table(
            "/out.jsonl",
            CreateTableSource(glob="secret/*"),
            backend,
            write,
        )


async def test_merges_glob_and_file_paths() -> None:
    backend = _StubBackend(files={"feedback/*.txt": ["feedback/001.txt"]})
    captured, write = _captured_write()
    await create_table(
        "/out.jsonl",
        CreateTableSource(
            glob="feedback/*.txt",
            file_paths=["extra/manual.txt"],
        ),
        backend,
        write,
    )
    rows = [json.loads(line) for line in captured["/out.jsonl"].strip().split("\n")]
    paths = {r["file"] for r in rows}
    assert paths == {"feedback/001.txt", "extra/manual.txt"}
