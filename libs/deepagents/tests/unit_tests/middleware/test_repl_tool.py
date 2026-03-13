from __future__ import annotations

from dataclasses import dataclass

from langchain.tools import ToolRuntime

from deepagents.backends.protocol import BackendProtocol, ReplBackendProtocol, ReplResponse
from deepagents.middleware.filesystem import FilesystemMiddleware


@dataclass
class _DummyRuntime:
    state: dict


def _runtime() -> ToolRuntime[None, None]:
    return ToolRuntime(state=_DummyRuntime(state={}), context=None, config={}, stream_writer=None, tool_call_id="x", store=None)  # type: ignore[arg-type]


class _NoopBackend(BackendProtocol):
    def ls_info(self, path: str) -> list[dict]:
        return []


class _DummyReplBackend(_NoopBackend, ReplBackendProtocol):
    @property
    def id(self) -> str:
        return "dummy"

    def repl(self, code: str, *, timeout: int | None = None) -> ReplResponse:
        if timeout is not None and timeout == 123:
            msg = "boom"
            raise ValueError(msg)
        if code == "err":
            return ReplResponse(output="out", error="bad")
        return ReplResponse(output=f"echo:{code}")


def test_repl_tool_unsupported_backend_returns_error() -> None:
    mw = FilesystemMiddleware(backend=_NoopBackend())
    tool = next(t for t in mw.tools if t.name == "repl")
    runtime = _runtime()
    result = tool.func(code="x", runtime=runtime)
    assert "Error: REPL evaluation not available" in result


def test_repl_tool_timeout_validation() -> None:
    mw = FilesystemMiddleware(backend=_DummyReplBackend())
    tool = next(t for t in mw.tools if t.name == "repl")
    runtime = _runtime()

    assert tool.func(code="x", runtime=runtime, timeout=0) == "Error: timeout must be positive, got 0."
    assert "exceeds maximum allowed" in tool.func(code="x", runtime=runtime, timeout=999999)


def test_repl_tool_formats_output_and_error() -> None:
    mw = FilesystemMiddleware(backend=_DummyReplBackend())
    tool = next(t for t in mw.tools if t.name == "repl")
    runtime = _runtime()

    result = tool.func(code="err", runtime=runtime)
    assert result == "out\n[Error]\nbad"
