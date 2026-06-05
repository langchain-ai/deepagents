"""Unit tests for the empty `ls`/`glob` result normalization middleware."""

from typing import Any, cast

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage

from deepagents_code.filesystem_empty_result import _FilesystemEmptyResultMiddleware


def test_filesystem_empty_result_middleware_rewrites_empty_ls_and_glob() -> None:
    """Empty `ls` and `glob` outputs should be readable to the agent."""
    middleware = _FilesystemEmptyResultMiddleware()

    for tool_name in ("ls", "glob"):
        result = middleware.wrap_tool_call(
            cast("Any", None),
            lambda _request, name=tool_name: ToolMessage(
                content="[]",
                name=name,
                tool_call_id=f"call-{name}",
                status="success",
            ),
        )

        assert isinstance(result, ToolMessage)
        assert result.content == "No files found"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            ToolMessage(
                content="[]", name="grep", tool_call_id="call-1", status="success"
            ),
            "[]",
        ),
        (
            ToolMessage(content="[]", name="ls", tool_call_id="call-2", status="error"),
            "[]",
        ),
        (
            ToolMessage(
                content="['/tmp/a.py']",
                name="glob",
                tool_call_id="call-3",
                status="success",
            ),
            "['/tmp/a.py']",
        ),
        # Near-miss content: the match is intentionally exact (`== "[]"`), so a
        # whitespace-padded variant must pass through untouched.
        (
            ToolMessage(
                content="[ ]", name="ls", tool_call_id="call-4", status="success"
            ),
            "[ ]",
        ),
    ],
)
def test_filesystem_empty_result_middleware_preserves_other_results(
    message: ToolMessage, expected: str
) -> None:
    """Only successful empty `ls` and `glob` list outputs are rewritten."""
    middleware = _FilesystemEmptyResultMiddleware()

    result = middleware.wrap_tool_call(cast("Any", None), lambda _request: message)

    assert isinstance(result, ToolMessage)
    assert result.content == expected


async def test_filesystem_empty_result_middleware_rewrites_async() -> None:
    """The async wrapper (the path the REPL actually runs) rewrites empty output."""
    middleware = _FilesystemEmptyResultMiddleware()

    # `awrap_tool_call` awaits the handler, so it must be a coroutine function.
    async def handler(_request: object) -> ToolMessage:  # noqa: RUF029
        return ToolMessage(
            content="[]", name="ls", tool_call_id="call-async", status="success"
        )

    result = await middleware.awrap_tool_call(cast("Any", None), handler)

    assert isinstance(result, ToolMessage)
    assert result.content == "No files found"


async def test_filesystem_empty_result_middleware_async_preserves_error() -> None:
    """The async wrapper leaves non-success empty output untouched."""
    middleware = _FilesystemEmptyResultMiddleware()

    # `awrap_tool_call` awaits the handler, so it must be a coroutine function.
    async def handler(_request: object) -> ToolMessage:  # noqa: RUF029
        return ToolMessage(
            content="[]", name="ls", tool_call_id="call-async-err", status="error"
        )

    result = await middleware.awrap_tool_call(cast("Any", None), handler)

    assert isinstance(result, ToolMessage)
    assert result.content == "[]"


def test_filesystem_empty_result_middleware_passes_through_command() -> None:
    """Non-`ToolMessage` results (e.g. `Command`) pass through unchanged."""
    from langgraph.types import Command

    middleware = _FilesystemEmptyResultMiddleware()
    command: Command[Any] = Command(update={"messages": []})

    result = middleware.wrap_tool_call(cast("Any", None), lambda _request: command)

    assert result is command


def test_sdk_still_returns_bracket_for_empty_listing() -> None:
    """Canary: fail loudly when the SDK stops emitting "[]" for empty listings.

    `_FilesystemEmptyResultMiddleware` exists *only* to rewrite the literal
    "[]" that the SDK's `ls`/`glob` tools currently return for an empty result.
    Upstream is expected to start returning useful empty-result content
    directly (e.g. "No files found"). When that lands, the real tools below
    will no longer return "[]", this assertion fails, and the failure is the
    signal to delete `deepagents_code.filesystem_empty_result` and its wiring
    in `create_cli_agent` — the middleware will have become redundant.
    """
    from deepagents.backends import StoreBackend
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from langgraph.store.memory import InMemoryStore

    # `StoreBackend` works standalone (no LangGraph config required), so the
    # empty store gives us a genuinely empty listing through the real tools.
    backend = StoreBackend(store=InMemoryStore(), namespace=lambda _rt: ("filesystem",))
    middleware = FilesystemMiddleware(backend=backend)
    runtime = ToolRuntime(
        state={},
        context=None,
        tool_call_id="canary",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )

    ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
    glob_tool = next(tool for tool in middleware.tools if tool.name == "glob")

    ls_result = ls_tool.invoke({"runtime": runtime, "path": "/"})
    glob_result = glob_tool.invoke({"runtime": runtime, "pattern": "*", "path": "/"})

    assert ls_result.content == "[]", (
        "SDK `ls` no longer returns '[]' for an empty directory — "
        "_FilesystemEmptyResultMiddleware is likely redundant and can be removed."
    )
    assert glob_result.content == "[]", (
        "SDK `glob` no longer returns '[]' for an empty match — "
        "_FilesystemEmptyResultMiddleware is likely redundant and can be removed."
    )
