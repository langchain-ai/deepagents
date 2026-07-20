"""Turn-boundary guard for `langgraph dev`'s blocking-call detector.

Under `langgraph dev`, BlockBuster instruments the event loop and raises
`BlockingError` when synchronous filesystem I/O (`os.readlink`,
`pathlib.Path.resolve`, `os.stat`, symlink checks) runs on the loop from
inside the async middleware chain. Because that exception unwinds every
`awrap_model_call` wrapper (Filesystem, Memory, HITL, and peers), a turn
aborts with an error status and empty output even after the agent has
finished its tool work — the completed results already appended to state are
discarded with it.

`BlockingCallGuardMiddleware` sits at the outermost position of the model-call
wrapper stack and catches `BlockingError` at the turn boundary. It ends the
model call with a recoverable assistant message instead of re-raising, so the
graph loop terminates cleanly and any tool results already in state survive.
The real fix is to keep sync filesystem I/O off the loop; this guard is the
safety net that stops a stray blocking call from silently destroying work.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, ModelResponse
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest

logger = logging.getLogger(__name__)

_RECOVERABLE_MESSAGE = (
    "This turn was interrupted by a blocking filesystem call detected on the "
    "event loop (langgraph dev BlockingError). Any completed tool results have "
    "been preserved; retry the request to continue."
)


def _is_blocking_error(exc: BaseException) -> bool:
    """Return whether an exception is BlockBuster's `BlockingError`.

    Matched by class name so the guard has no import-time dependency on
    `blockbuster`, which is only present under `langgraph dev`.
    """
    return type(exc).__name__ == "BlockingError"


def _recoverable_response() -> ModelResponse:
    """Return the assistant message that ends the turn without discarding work.

    Returns:
        A response carrying a single tool-call-free `AIMessage`, which ends the
            model loop cleanly and preserves tool results already in state.
    """
    return ModelResponse(result=[AIMessage(content=_RECOVERABLE_MESSAGE)])


class BlockingCallGuardMiddleware(AgentMiddleware[Any, Any]):
    """Catch `BlockingError` at the model-call boundary and recover the turn."""

    def wrap_model_call(  # noqa: PLR6301  # middleware override hook
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Run the sync handler, converting a `BlockingError` into recovery.

        Returns:
            The downstream response, or a recoverable assistant message when a
                blocking filesystem call is detected on the event loop.
        """
        try:
            return handler(request)
        except Exception as exc:
            if not _is_blocking_error(exc):
                raise
            logger.warning(
                "Recovered turn from blocking filesystem call on the event loop",
                exc_info=True,
            )
            return _recoverable_response()

    async def awrap_model_call(  # noqa: PLR6301  # middleware override hook
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Run the async handler, converting a `BlockingError` into recovery.

        Returns:
            The downstream response, or a recoverable assistant message when a
                blocking filesystem call is detected on the event loop.
        """
        try:
            return await handler(request)
        except Exception as exc:
            if not _is_blocking_error(exc):
                raise
            logger.warning(
                "Recovered turn from blocking filesystem call on the event loop",
                exc_info=True,
            )
            return _recoverable_response()


__all__ = ["BlockingCallGuardMiddleware"]
