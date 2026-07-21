"""Rubric middleware retries for transient grader transport failures."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import httpx
from deepagents.middleware.rubric import GraderResponse, RubricMiddleware, RubricState

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

logger = logging.getLogger(__name__)


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield an exception, its explicit/implicit causes, and group members once.

    Descends into `BaseExceptionGroup` members as well as `__cause__` and
    `__context__`, so a transient transport error wrapped in an async task group
    is still discovered. Each exception is yielded at most once.
    """
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif current.__context__ is not None:
            pending.append(current.__context__)


def _is_transient_grader_transport_error(exc: BaseException) -> bool:
    """Return whether a grader failure is a retryable transport/read error.

    Matches response-read faults (`httpx`/`httpcore` `ReadError`) and
    response-framing faults (`RemoteProtocolError`, aiohttp
    `TransferEncodingError`). Connect/timeout errors are intentionally excluded
    so only mid-response transport failures trigger the retry.
    """
    for current in _exception_chain(exc):
        if isinstance(current, (httpx.ReadError, httpx.RemoteProtocolError)):
            return True
        error_type = type(current)
        if error_type.__module__.startswith("httpcore") and error_type.__name__ in {
            "ReadError",
            "RemoteProtocolError",
        }:
            return True
        if (
            error_type.__module__ == "aiohttp.http_exceptions"
            and error_type.__name__ == "TransferEncodingError"
            and "Not enough data to satisfy transfer length header" in str(current)
        ):
            return True
    return False


class ReliableRubricMiddleware(RubricMiddleware):
    """Retry one transient grader transport failure without rerunning agent work.

    The retry re-invokes only the grader sub-agent, never the task agent, so it
    relies on the grader's own tools being read-only/idempotent. A second
    failure — transient or not — propagates to the base middleware, which
    surfaces it as a `grader_error` result rather than a silent success.
    """

    def _grade(
        self,
        state: RubricState,
        iteration: int,
        *,
        context: object | None = None,
    ) -> GraderResponse:
        try:
            return super()._grade(state, iteration, context=cast("Any", context))
        except Exception as exc:
            if not _is_transient_grader_transport_error(exc):
                raise
            logger.warning(
                "Rubric grader transport failed; retrying grading once",
                exc_info=True,
            )
        return super()._grade(state, iteration, context=cast("Any", context))

    async def _agrade(
        self,
        state: RubricState,
        iteration: int,
        *,
        context: object | None = None,
    ) -> GraderResponse:
        try:
            return await super()._agrade(state, iteration, context=cast("Any", context))
        except Exception as exc:
            if not _is_transient_grader_transport_error(exc):
                raise
            logger.warning(
                "Rubric grader transport failed; retrying grading once",
                exc_info=True,
            )
        return await super()._agrade(state, iteration, context=cast("Any", context))
