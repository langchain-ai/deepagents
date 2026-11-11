"""Keep the Anthropic prompt cache warm while a slow tool runs.

Anthropic's default prompt-cache entry lives five minutes from the start of the
last request that wrote or read it. When a tool runs longer than that (a build,
a test suite, a subagent), the next model call misses the cache and re-writes
the whole conversation at the cache-write price. This middleware re-sends the
previous model request with `max_tokens=0` before the entry expires: the API
reads the cached prefix, refreshes its timer, and returns no output, so the
ping costs one cache read.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, ModelResponse
from langchain_core.messages import AIMessage

from deepagents.middleware._utils import is_anthropic_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelCallResult, ModelRequest
    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_SECONDS = 240.0
"""Ping one minute before Anthropic's five-minute cache entry would expire."""

_DEFAULT_MAX_PINGS = 6
"""Bounds the cost of a hung tool to a handful of cache reads."""

_MAX_TRACKED_CALLS = 64
"""Model calls whose tool calls never ran (rejected, interrupted) are dropped past this."""


@dataclass
class _PendingCall:
    """The last model request whose tool calls are now running."""

    request: ModelRequest
    started_at: float
    handler: Callable[[ModelRequest], ModelResponse] | None = None
    ahandler: Callable[[ModelRequest], Awaitable[ModelResponse]] | None = None
    active_tools: int = 0
    pings: int = 0
    stop: threading.Event = field(default_factory=threading.Event)
    task: asyncio.Task[None] | None = None


def _tool_call_ids(response: ModelCallResult) -> list[str]:
    """Return the ids of the tool calls in a model response."""
    messages = response.result if isinstance(response, ModelResponse) else [response]
    ids = (call.get("id") for message in messages if isinstance(message, AIMessage) for call in message.tool_calls)
    return [i for i in ids if i]


def _ping_request(request: ModelRequest) -> ModelRequest:
    """Return `request` asking for no output and no streaming.

    The Anthropic API rejects `max_tokens=0` on a streaming request, and the
    prefix it reads must match the original request byte for byte, so only
    these two settings change.
    """
    return request.override(model_settings={**request.model_settings, "max_tokens": 0, "stream": False})


class AnthropicCacheKeepaliveMiddleware(AgentMiddleware[Any, Any, Any]):
    """Re-read the Anthropic prompt cache while a long tool call runs.

    Pings start only for tool calls that follow an Anthropic model call, stop as
    soon as the last of those tool calls finishes, and stop for good after
    `max_pings` pings or the first ping error, so a hung tool costs at most
    `max_pings` cache reads. Pings appear in traces as model calls with no
    output tokens.
    """

    def __init__(self, *, interval: float = _DEFAULT_INTERVAL_SECONDS, max_pings: int = _DEFAULT_MAX_PINGS) -> None:
        """Initialize the middleware.

        Args:
            interval: Seconds between the start of the last model request and
                the next ping. Keep it under the cache entry's five-minute TTL.
            max_pings: Most pings sent while one batch of tool calls runs.
        """
        super().__init__()
        self.interval = interval
        self.max_pings = max_pings
        self._pending: OrderedDict[str, _PendingCall] = OrderedDict()
        self._lock = threading.Lock()

    def _register(self, response: ModelCallResult, call: _PendingCall) -> None:
        """Remember `call` for each tool call in `response`."""
        with self._lock:
            for tool_call_id in _tool_call_ids(response):
                self._pending[tool_call_id] = call
            while len(self._pending) > _MAX_TRACKED_CALLS:
                self._pending.popitem(last=False)

    def _claim(self, request: ToolCallRequest) -> _PendingCall | None:
        """Take the pending model call for this tool call and count it as running."""
        with self._lock:
            call = self._pending.pop(request.tool_call.get("id") or "", None)
            if call is not None:
                call.active_tools += 1
            return call

    def _release(self, call: _PendingCall) -> bool:
        """Count one tool call as finished; return whether none are left running."""
        with self._lock:
            call.active_tools -= 1
            return call.active_tools == 0

    def _next_delay(self, call: _PendingCall) -> float:
        """Seconds until the cache entry is due for a refresh."""
        return max(call.started_at + self.interval - time.monotonic(), 0.0)

    def _log_ping(self, call: _PendingCall, response: ModelResponse) -> None:
        """Log one ping and its cache usage."""
        usage = next((m.usage_metadata for m in response.result if isinstance(m, AIMessage)), None) or {}
        details = usage.get("input_token_details") or {}
        logger.info(
            "Prompt-cache keep-alive ping %d/%d: cache_read=%s cache_creation=%s",
            call.pings,
            self.max_pings,
            details.get("cache_read"),
            details.get("cache_creation"),
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Remember an Anthropic request so its tool calls can keep its cache warm."""
        started_at = time.monotonic()
        response = handler(request)
        if is_anthropic_chat_model(request.model):
            self._register(response, _PendingCall(request=request, started_at=started_at, handler=handler))
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Remember an Anthropic request so its tool calls can keep its cache warm (async version)."""
        started_at = time.monotonic()
        response = await handler(request)
        if is_anthropic_chat_model(request.model):
            self._register(response, _PendingCall(request=request, started_at=started_at, ahandler=handler))
        return response

    def _ping_loop(self, call: _PendingCall) -> None:
        """Ping until the tool calls finish, `max_pings` is reached, or a ping fails."""
        while call.pings < self.max_pings and not call.stop.wait(self._next_delay(call)):
            call.started_at = time.monotonic()
            call.pings += 1
            try:
                response = call.handler(_ping_request(call.request))  # ty: ignore[call-non-callable]
            except Exception:  # noqa: BLE001  # A best-effort ping must never fail the running tool call.
                logger.warning("Prompt-cache keep-alive ping failed; stopping pings.", exc_info=True)
                return
            self._log_ping(call, response)

    async def _aping_loop(self, call: _PendingCall) -> None:
        """Ping until cancelled, `max_pings` is reached, or a ping fails (async version)."""
        while call.pings < self.max_pings:
            await asyncio.sleep(self._next_delay(call))
            call.started_at = time.monotonic()
            call.pings += 1
            try:
                response = await call.ahandler(_ping_request(call.request))  # ty: ignore[call-non-callable]
            except Exception:  # noqa: BLE001  # A best-effort ping must never fail the running tool call.
                logger.warning("Prompt-cache keep-alive ping failed; stopping pings.", exc_info=True)
                return
            self._log_ping(call, response)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Keep the cache warm in a background thread while the tool runs."""
        call = self._claim(request)
        if call is None or call.handler is None:
            return handler(request)
        if call.active_tools == 1:
            call.stop = threading.Event()
            context = contextvars.copy_context()
            threading.Thread(target=context.run, args=(self._ping_loop, call), daemon=True).start()
        try:
            return handler(request)
        finally:
            if self._release(call):
                call.stop.set()

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Keep the cache warm in a background task while the tool runs (async version)."""
        call = self._claim(request)
        if call is None or call.ahandler is None:
            return await handler(request)
        if call.active_tools == 1:
            call.task = asyncio.create_task(self._aping_loop(call))
        try:
            return await handler(request)
        finally:
            if self._release(call) and call.task is not None:
                call.task.cancel()
