"""Server-owned LangSmith tracing client and ingestion diagnostics."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from queue import Full
from typing import TYPE_CHECKING, Literal, Protocol, override

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langchain.agents.middleware.types import AgentMiddleware, AgentState
    from langgraph.runtime import Runtime
    from langsmith import Client
    from requests import Response

logger = logging.getLogger(__name__)


class _TracingClient(Protocol):
    def flush(self, timeout: float) -> None: ...


@dataclass(frozen=True)
class TracingDiagnostics:
    """Counters for the active server tracing session."""

    queued: int
    retried: int
    dropped: int


class _Counters:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.queued = 0
        self.retried = 0
        self.dropped = 0

    def increment(self, field: str, amount: int = 1) -> None:
        with self._lock:
            setattr(self, field, getattr(self, field) + amount)

    def snapshot(self) -> TracingDiagnostics:
        with self._lock:
            return TracingDiagnostics(self.queued, self.retried, self.dropped)


_active_client: _TracingClient | None = None
_active_counters: _Counters | None = None


def create_server_tracing_client(
    *,
    api_key: str | None,
    api_url: str | None,
    anonymizer: Callable[[dict[str, object]], dict[str, object]] | None = None,
) -> Client:
    """Create the process-lifetime tracing client used by server graphs."""  # noqa: DOC201  # Keep the new helper's docstring to one line.
    from langsmith import Client

    counters = _Counters()

    class ServerTracingClient(Client):
        @override
        def _put_tracing_queue(self, item: object) -> None:
            if self.tracing_queue is None:
                return
            try:
                self.tracing_queue.put_nowait(item)
            except Full:
                operation = getattr(item, "item", item)
                operation_name = getattr(operation, "operation", "unknown")
                run_id = getattr(operation, "id", "unknown")
                payload_size = _payload_size(operation)
                counters.increment("dropped")
                logger.warning(
                    "Dropped LangSmith run %s %s at tracing queue boundary "
                    "(payload_size=%d)",
                    operation_name,
                    run_id,
                    payload_size,
                )
            else:
                counters.increment("queued")

        @override
        def request_with_retries(
            self,
            /,
            method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"],
            pathname: str,
            *,
            request_kwargs: Mapping[str, object] | None = None,
            stop_after_attempt: int = 1,
            retry_on: Sequence[type[BaseException]] | None = None,
            to_ignore: Sequence[type[BaseException]] | None = None,
            handle_response: Callable[[Response, int], object] | None = None,
            _context: str = "",
            **kwargs: object,
        ) -> Response:
            if _context and "/runs/" in pathname:
                counters.increment("retried", max(stop_after_attempt - 1, 0))
            return super().request_with_retries(
                method,
                pathname,
                request_kwargs=request_kwargs,
                stop_after_attempt=stop_after_attempt,
                retry_on=retry_on,
                to_ignore=to_ignore,
                handle_response=handle_response,
                _context=_context,
                **kwargs,
            )

    global _active_client, _active_counters  # noqa: PLW0603
    _active_client = ServerTracingClient(
        api_key=api_key,
        api_url=api_url,
        anonymizer=anonymizer,
    )
    _active_counters = counters
    return _active_client


def _payload_size(item: object) -> int:
    calculate_size = getattr(item, "calculate_serialized_size", None)
    if not isinstance(calculate_size, Callable):
        return 0
    try:
        return int(calculate_size())
    except (TypeError, ValueError):
        return 0


def active_server_tracing_client() -> _TracingClient | None:
    """Return the active server client, if one exists."""
    return _active_client


def active_tracing_diagnostics() -> TracingDiagnostics | None:
    """Return active server tracing counters, if a session exists."""
    return _active_counters.snapshot() if _active_counters is not None else None


def server_tracing_is_active() -> bool:
    """Return whether this process owns a server tracing client."""
    return _active_client is not None


async def flush_server_tracing(wait_seconds: float = 2.0) -> None:
    """Flush active server tracing before the next graph turn."""
    client = active_server_tracing_client()
    if client is None:
        return
    try:
        async with asyncio.timeout(wait_seconds):
            await asyncio.to_thread(client.flush, wait_seconds)
    except Exception:  # tracing must not change agent behavior
        logger.warning("LangSmith tracing flush failed", exc_info=True)


def server_tracing_flush_middleware() -> AgentMiddleware:
    """Return middleware that flushes tracing at each graph turn boundary."""
    from langchain.agents.middleware.types import AgentMiddleware

    class TracingFlushMiddleware(AgentMiddleware):
        @override
        async def aafter_agent(
            self, state: AgentState, runtime: Runtime[None]
        ) -> dict[str, object] | None:
            await flush_server_tracing()
            return None

    return TracingFlushMiddleware()
