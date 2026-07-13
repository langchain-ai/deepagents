"""Rubric middleware retries for transient grader transport failures."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx
from deepagents.middleware.rubric import GraderResponse, RubricMiddleware, RubricState

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield an exception and its explicit or implicit causes once."""
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
    """Return whether a grader failure is a retryable response-read error."""
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
    """Retry one transient grader read failure without rerunning agent work."""

    def _grade(self, state: RubricState, iteration: int) -> GraderResponse:
        try:
            return super()._grade(state, iteration)
        except Exception as exc:
            if not _is_transient_grader_transport_error(exc):
                raise
            logger.warning(
                "Rubric grader transport failed; retrying grading once",
                exc_info=True,
            )
        return super()._grade(state, iteration)

    async def _agrade(self, state: RubricState, iteration: int) -> GraderResponse:
        try:
            return await super()._agrade(state, iteration)
        except Exception as exc:
            if not _is_transient_grader_transport_error(exc):
                raise
            logger.warning(
                "Rubric grader transport failed; retrying grading once",
                exc_info=True,
            )
        return await super()._agrade(state, iteration)
