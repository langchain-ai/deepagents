"""Tests for the ``readFile`` / ``writeFile`` REPL globals.

- ``readFile(path)`` goes directly to the backend — it does *not* consult
  the pending-writes buffer. A ``writeFile`` + ``readFile`` on the same
  path in one eval will not see the staged content.
- ``writeFile(path, content)`` stages into pending writes.
"""

from __future__ import annotations

from typing import Any

import pytest
from deepagents.backends.protocol import (
    BackendProtocol,
    GlobResult,
    ReadResult,
    WriteResult,
)
from quickjs_rs import Runtime

from deepagents_repl._repl import _ThreadREPL


class _InMemoryBackend(BackendProtocol):
    def __init__(self, files: dict[str, str] | None = None) -> None:
        self._files: dict[str, str] = dict(files or {})

    async def aread(
        self, file_path: str, offset: int = 0, limit: int = 2000
    ) -> ReadResult:
        if file_path not in self._files:
            return ReadResult(error="file_not_found")
        return ReadResult(
            file_data={"content": self._files[file_path], "encoding": "utf-8"}
        )

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        self._files[file_path] = content
        return WriteResult(path=file_path)

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        return GlobResult(matches=[])

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


@pytest.fixture
def runtime() -> Runtime:
    rt = Runtime()
    try:
        yield rt
    finally:
        rt.close()


async def test_read_file_returns_backend_content(runtime: Runtime) -> None:
    backend = _InMemoryBackend({"/data.json": '{"n": 42}'})
    repl = _ThreadREPL(runtime, timeout=5.0, capture_console=True, backend=backend)
    outcome = await repl.eval_async('await readFile("/data.json")')
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == '{"n": 42}'


async def test_read_file_missing_path_rejects_with_enoent(runtime: Runtime) -> None:
    backend = _InMemoryBackend()
    repl = _ThreadREPL(runtime, timeout=5.0, capture_console=True, backend=backend)
    outcome = await repl.eval_async(
        """
        try {
            await readFile("/nope");
            "unexpected"
        } catch (e) {
            e.message
        }
        """
    )
    assert outcome.error_type is None
    assert "ENOENT" in (outcome.result or "")
    assert "/nope" in (outcome.result or "")


async def test_write_file_stages_into_pending_buffer(runtime: Runtime) -> None:
    backend = _InMemoryBackend()
    repl = _ThreadREPL(runtime, timeout=5.0, capture_console=True, backend=backend)
    outcome = await repl.eval_async(
        """
        await writeFile("/out.txt", "hello");
        "ok"
        """
    )
    assert outcome.error_type is None, outcome.error_message
    assert repl._pending_writes == {"/out.txt": "hello"}
    # Backend is untouched until the middleware flushes pending writes.
    assert "/out.txt" not in backend._files


async def test_read_file_does_not_see_pending_writes(runtime: Runtime) -> None:
    """Top-level readFile goes straight to the backend and does not
    observe writes staged in the same eval."""
    backend = _InMemoryBackend()
    repl = _ThreadREPL(runtime, timeout=5.0, capture_console=True, backend=backend)
    outcome = await repl.eval_async(
        """
        await writeFile("/live.txt", "staged");
        try {
            await readFile("/live.txt");
            "unexpected"
        } catch (e) {
            e.message
        }
        """
    )
    assert outcome.error_type is None
    assert "ENOENT" in (outcome.result or "")


async def test_write_file_flushes_to_backend_after_drain(runtime: Runtime) -> None:
    """The middleware flushes pending writes to the backend after each eval;
    simulate that by draining manually."""
    backend = _InMemoryBackend()
    repl = _ThreadREPL(runtime, timeout=5.0, capture_console=True, backend=backend)
    await repl.eval_async('await writeFile("/out.txt", "flushed")')
    # Simulate middleware's post-eval flush.
    for path, content in repl._drain_pending_writes():
        await backend.awrite(path, content)
    assert backend._files["/out.txt"] == "flushed"


async def test_vfs_not_installed_without_backend(runtime: Runtime) -> None:
    repl = _ThreadREPL(runtime, timeout=5.0, capture_console=True)
    outcome = await repl.eval_async("typeof readFile + '/' + typeof writeFile")
    assert outcome.result == "undefined/undefined"
