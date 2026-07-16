"""LangChain middleware exposing a gated, secure browser toolset."""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, NoReturn

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import BaseTool, ToolRuntime  # noqa: TC002  # inspected at runtime
from langchain_core.tools import StructuredTool
from langgraph.config import get_config
from pydantic import ConfigDict

from deepagents_browser.errors import BrowserAccessError, BrowserErrorCode, BrowserRuntimeError
from deepagents_browser.runtime import BrowserLimits, BrowserRuntimeManager, BrowserSession
from deepagents_browser.schemas import (
    ActInput,
    BrowserAction,
    NavigateInput,
    ScreenshotInput,
    SnapshotInput,
    TabsInput,
)
from deepagents_browser.state import BrowserState

_BROWSER_TOOL_NAMES = frozenset(
    {
        "browser_navigate",
        "browser_snapshot",
        "browser_act",
        "browser_screenshot",
        "browser_tabs",
    }
)
_BROWSER_CONSEQUENTIAL_TOOL_NAMES = frozenset({"browser_navigate", "browser_act", "browser_tabs"})
_RECOVERABLE_ACTION_ERROR_CODES = frozenset(
    {
        BrowserErrorCode.STALE_ELEMENT_REFERENCE.value,
        BrowserErrorCode.ELEMENT_IDENTITY_UNAVAILABLE.value,
        BrowserErrorCode.ELEMENT_CHANGED.value,
        BrowserErrorCode.PAGE_NAVIGATED.value,
        BrowserErrorCode.NAVIGATION_INVALIDATED_REFERENCE.value,
        BrowserErrorCode.ELEMENT_DISABLED.value,
        BrowserErrorCode.ELEMENT_NOT_EDITABLE.value,
        BrowserErrorCode.ACTION_TARGET_MISMATCH.value,
        BrowserErrorCode.ACTION_TIMEOUT.value,
        BrowserErrorCode.SCROLL_FAILED.value,
        BrowserErrorCode.ACTION_FAILED.value,
    }
)
_ACTION_REFRESH_INSTRUCTION = "Call browser_snapshot before the next browser_act."
_ACTION_RECOVERY_INSTRUCTION = "Call browser_snapshot, then retry with a new element reference."
_SNAPSHOT_RECOVERY_INSTRUCTION = "Retry browser_snapshot after the page finishes loading."
_SNAPSHOT_ATTEMPTS = 2


class _NavigateToolInput(NavigateInput):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runtime: ToolRuntime[Any, Any]


class _SnapshotToolInput(SnapshotInput):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runtime: ToolRuntime[Any, Any]


class _ActToolInput(ActInput):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runtime: ToolRuntime[Any, Any]


class _ScreenshotToolInput(ScreenshotInput):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runtime: ToolRuntime[Any, Any]


class _TabsToolInput(TabsInput):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runtime: ToolRuntime[Any, Any]


def _is_active(state: Mapping[str, Any] | None) -> bool:
    return state is not None and state.get("_browser_enabled") is True


def _require_active(runtime: ToolRuntime[Any, Any]) -> None:
    state = runtime.state
    if not isinstance(state, Mapping) or not _is_active(state):
        msg = "Browser access is disabled; `_browser_enabled` must be exactly true"
        raise BrowserAccessError(msg, code=BrowserErrorCode.ACCESS_DISABLED)


def _tool_name(tool: BaseTool | dict[str, Any]) -> str | None:
    name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
    return name if isinstance(name, str) else None


def _resolve_thread_id(fallback: str) -> str:
    try:
        config = get_config()
    except RuntimeError:
        return fallback
    thread_id = config.get("configurable", {}).get("thread_id") if config else None
    return fallback if thread_id is None else str(thread_id)


def _async_only() -> NoReturn:
    msg = "Browser tools require asynchronous invocation"
    raise BrowserRuntimeError(msg, code=BrowserErrorCode.ASYNC_REQUIRED)


class BrowserMiddleware(AgentMiddleware[BrowserState, ContextT, ResponseT]):
    """Expose five activation-gated browser tools to a Deep Agent.

    Construction is side-effect free: Playwright is not imported, initialized, or
    launched until an activated tool first needs a browser context. Each LangGraph
    thread receives an isolated, bounded browser context.

    Args:
        limits: Resource bounds for browser operations.
        runtime_manager: Internal dependency injection point for tests and advanced hosts.
    """

    state_schema = BrowserState

    def __init__(
        self,
        *,
        limits: BrowserLimits | None = None,
        runtime_manager: BrowserRuntimeManager | None = None,
    ) -> None:
        """Create exactly five tools without accessing a browser runtime."""
        super().__init__()
        if limits is not None and runtime_manager is not None:
            msg = "Pass either limits or runtime_manager, not both"
            raise ValueError(msg)
        self._runtime = runtime_manager or BrowserRuntimeManager(limits=limits)
        self._fallback_thread_id = f"browser_{uuid.uuid4().hex}"
        self.tools: list[BaseTool] = [
            self._build_navigate_tool(),
            self._build_snapshot_tool(),
            self._build_act_tool(),
            self._build_screenshot_tool(),
            self._build_tabs_tool(),
        ]
        names = tuple(tool.name for tool in self.tools)
        if len(names) != len(set(names)) or any(not name for name in names):
            msg = "Browser tool names must be non-empty and unique"
            raise ValueError(msg)
        self._tool_names = frozenset(names)
        if self._tool_names != _BROWSER_TOOL_NAMES:
            msg = "Browser middleware tool surface does not match the public contract"
            raise ValueError(msg)

    @property
    def tool_names(self) -> frozenset[str]:
        """Return the exact browser tool names owned by this middleware."""
        return self._tool_names

    @property
    def consequential_tool_names(self) -> frozenset[str]:
        """Return browser tools that should require human approval."""
        return _BROWSER_CONSEQUENTIAL_TOOL_NAMES

    @asynccontextmanager
    async def _session(self, runtime: ToolRuntime[Any, Any]) -> AsyncIterator[BrowserSession]:
        _require_active(runtime)
        thread_id = _resolve_thread_id(self._fallback_thread_id)
        async with self._runtime.lease_session(thread_id) as session:
            yield session

    def _build_navigate_tool(self) -> BaseTool:
        middleware = self

        def navigate_sync(
            url: str,
            runtime: ToolRuntime[Any, Any],
            page_ref: str | None = None,
        ) -> NoReturn:
            _require_active(runtime)
            _ = (url, page_ref)
            _async_only()

        async def navigate_async(
            url: str,
            runtime: ToolRuntime[Any, Any],
            page_ref: str | None = None,
        ) -> str:
            _require_active(runtime)
            await middleware._runtime.validate_url(url)
            async with middleware._session(runtime) as session:
                return await session.navigate(url, page_ref)

        return StructuredTool.from_function(
            name="browser_navigate",
            description="Navigate a browser tab to a validated HTTP(S) URL.",
            func=navigate_sync,
            coroutine=navigate_async,
            infer_schema=False,
            args_schema=_NavigateToolInput,
        )

    def _build_snapshot_tool(self) -> BaseTool:
        middleware = self

        def snapshot_sync(
            runtime: ToolRuntime[Any, Any],
            page_ref: str | None = None,
        ) -> NoReturn:
            _require_active(runtime)
            _ = page_ref
            _async_only()

        async def snapshot_async(
            runtime: ToolRuntime[Any, Any],
            page_ref: str | None = None,
        ) -> str:
            _require_active(runtime)
            async with middleware._session(runtime) as session:
                for attempt in range(_SNAPSHOT_ATTEMPTS):
                    try:
                        return await session.snapshot(page_ref)
                    except BrowserRuntimeError as exc:
                        if exc.code != BrowserErrorCode.PAGE_NAVIGATED.value:
                            raise
                        if attempt == _SNAPSHOT_ATTEMPTS - 1:
                            return json.dumps(
                                {
                                    "ok": False,
                                    "error": exc.as_dict(),
                                    "next": _SNAPSHOT_RECOVERY_INSTRUCTION,
                                },
                                separators=(",", ":"),
                            )
            msg = "Snapshot retry loop exited unexpectedly"
            raise BrowserRuntimeError(msg)

        return StructuredTool.from_function(
            name="browser_snapshot",
            description="Capture a bounded snapshot and fresh opaque actionable references.",
            func=snapshot_sync,
            coroutine=snapshot_async,
            infer_schema=False,
            args_schema=_SnapshotToolInput,
        )

    def _build_act_tool(self) -> BaseTool:
        middleware = self

        def act_sync(action: BrowserAction, runtime: ToolRuntime[Any, Any]) -> NoReturn:
            _require_active(runtime)
            _ = action
            _async_only()

        async def act_async(action: BrowserAction, runtime: ToolRuntime[Any, Any]) -> str:
            _require_active(runtime)
            async with middleware._session(runtime) as session:
                try:
                    result = json.loads(await session.act(action))
                except BrowserRuntimeError as exc:
                    if exc.code not in _RECOVERABLE_ACTION_ERROR_CODES:
                        raise
                    return json.dumps(
                        {
                            "ok": False,
                            "error": exc.as_dict(),
                            "next": _ACTION_RECOVERY_INSTRUCTION,
                        },
                        separators=(",", ":"),
                    )
                result["next"] = _ACTION_REFRESH_INSTRUCTION
                return json.dumps(result, separators=(",", ":"))

        return StructuredTool.from_function(
            name="browser_act",
            description=(
                "Perform one allowlisted action: constrained viewport-relative page scrolling, or "
                "an exact-element action using a fresh, single-use opaque reference. Raw "
                "selectors, coordinates, deltas, and JavaScript are not accepted. Call "
                "browser_snapshot after "
                "every successful action before calling browser_act again."
            ),
            func=act_sync,
            coroutine=act_async,
            infer_schema=False,
            args_schema=_ActToolInput,
        )

    def _build_screenshot_tool(self) -> BaseTool:
        middleware = self

        def screenshot_sync(
            runtime: ToolRuntime[Any, Any],
            page_ref: str | None = None,
        ) -> NoReturn:
            _require_active(runtime)
            _ = page_ref
            _async_only()

        async def screenshot_async(
            runtime: ToolRuntime[Any, Any],
            page_ref: str | None = None,
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            _require_active(runtime)
            async with middleware._session(runtime) as session:
                screenshot = await session.screenshot(page_ref)
            metadata = screenshot.metadata()
            image = {
                "type": "image",
                "base64": base64.b64encode(screenshot.data).decode("ascii"),
                "mime_type": screenshot.media_type,
            }
            content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": json.dumps(metadata, separators=(",", ":")),
                },
                image,
            ]
            artifact = {**metadata, "image": image}
            return content, artifact

        return StructuredTool.from_function(
            name="browser_screenshot",
            description="Capture a bounded fixed-viewport PNG screenshot.",
            func=screenshot_sync,
            coroutine=screenshot_async,
            infer_schema=False,
            args_schema=_ScreenshotToolInput,
            response_format="content_and_artifact",
        )

    def _build_tabs_tool(self) -> BaseTool:
        middleware = self

        def tabs_sync(
            runtime: ToolRuntime[Any, Any],
            operation: str = "list",
            page_ref: str | None = None,
        ) -> NoReturn:
            _require_active(runtime)
            _ = (operation, page_ref)
            _async_only()

        async def tabs_async(
            runtime: ToolRuntime[Any, Any],
            operation: str = "list",
            page_ref: str | None = None,
        ) -> str:
            _require_active(runtime)
            async with middleware._session(runtime) as session:
                return await session.tabs(operation, page_ref)

        return StructuredTool.from_function(
            name="browser_tabs",
            description="List, create, select, or close bounded browser tabs.",
            func=tabs_sync,
            coroutine=tabs_async,
            infer_schema=False,
            args_schema=_TabsToolInput,
        )

    def _filtered_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        if _is_active(request.state):
            return request
        filtered = [tool for tool in request.tools if _tool_name(tool) not in self._tool_names]
        return request.override(tools=filtered)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Hide exact-name browser tools from synchronous inactive model calls."""
        return handler(self._filtered_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Hide exact-name browser tools from asynchronous inactive model calls."""
        return await handler(self._filtered_request(request))

    async def aclose(self) -> None:
        """Idempotently close all browser resources owned by this middleware."""
        await self._runtime.aclose()
