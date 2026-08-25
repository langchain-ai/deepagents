"""Rubric middleware retries for transient grader transport failures."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NotRequired, cast

import httpx
from deepagents.middleware.rubric import (
    RUBRIC_GRADER_MESSAGE_SOURCE,
    GraderResponse,
    RubricMiddleware,
    RubricState,
)
from langchain.agents.middleware.types import AgentMiddleware, AgentState

from deepagents_code.goal_state_notice import is_conversation_control_message

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from deepagents.middleware.rubric import RubricEvaluation
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AnyMessage
    from langchain_core.tools import BaseTool

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


def _without_internal_control_messages(state: RubricState) -> RubricState:
    """Remove dcode control turns before the SDK builds grader evidence.

    Returns:
        Original state when unchanged, otherwise a shallow copy with filtered
        messages.
    """
    messages = state.get("messages", [])
    if not isinstance(messages, list):
        return state
    filtered: list[AnyMessage] = [
        message for message in messages if not is_conversation_control_message(message)
    ]
    if len(filtered) == len(messages):
        return state
    updated = dict(state)
    updated["messages"] = filtered
    return cast("RubricState", updated)


class RubricGraderState(AgentState[GraderResponse]):
    """Nested-grader state used to scope verification-tool budgets."""

    rubric_grading_operation_id: NotRequired[str]


class ReliableRubricMiddleware(RubricMiddleware):
    """Run a context-aware nested grader and retry transient transport failures.

    The nested grader receives Deep Agents Code's verification middleware and
    runtime context without requiring those application-specific capabilities in
    the SDK's `RubricMiddleware`. A transport retry re-invokes only the grader,
    never the task agent, so grader tools must be read-only or idempotent.

    The transport retry wraps `_invoke_grader` (one grader call) rather than
    `_grade` (the whole grading pass), so the SDK's coverage retry still runs on
    top of it: a transport fault is retried within a call, and a grader that
    under-reports its criteria is retried across calls.
    """

    def __init__(  # noqa: D107
        self,
        *,
        model: str | BaseChatModel,
        system_prompt: str | None = None,
        tools: Sequence[BaseTool] | None = None,
        grader_middleware: Sequence[AgentMiddleware[Any, Any]] | None = None,
        grader_context_schema: type[Any] | None = None,
        max_iterations: int = 3,
        on_evaluation: Callable[[RubricEvaluation], None] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            max_iterations=max_iterations,
            on_evaluation=on_evaluation,
        )
        self._grader_middleware = list(grader_middleware or ())
        self._grader_context_schema = grader_context_schema

    def _ensure_grader(self) -> Any:  # noqa: ANN401
        if self._grader is not None:
            return self._grader

        from deepagents._models import (  # noqa: PLC2701
            resolve_model,
        )
        from langchain.agents import create_agent

        resolved_model = resolve_model(self._model)
        self._resolved_model = resolved_model
        self._grader = create_agent(
            model=resolved_model,
            system_prompt=self._system_prompt,
            tools=self._tools,
            middleware=self._grader_middleware,
            name=RUBRIC_GRADER_MESSAGE_SOURCE,
            response_format=GraderResponse,
            state_schema=RubricGraderState,
            context_schema=self._grader_context_schema,
        )
        return self._grader

    def _grader_input(
        self,
        state: RubricState,
        iteration: int,
        correction: str | None = None,
    ) -> dict[str, Any]:
        """Build nested-grader input with a stable verification-operation ID.

        Drops dcode control turns before delegating to the SDK, which applies
        the delimiter sanitization for the untrusted transcript.

        Args:
            state: Agent state, read for the rubric and transcript.
            iteration: Zero-based grading iteration.
            correction: Feedback about a previous unusable response, if any.

        Returns:
            The nested grader's input state.
        """
        grading_run_id = state.get("_current_grading_run_id") or "untracked"
        grader_state = _without_internal_control_messages(state)
        grader_input = super()._grader_input(grader_state, iteration, correction)
        grader_input["rubric_grading_operation_id"] = f"{grading_run_id}:{iteration}"
        return grader_input

    def _invoke_grader(
        self,
        state: RubricState,
        iteration: int,
        correction: str | None = None,
        *,
        context: object | None = None,
    ) -> GraderResponse:
        """Run one grader call, retrying once on a transient transport failure.

        Returns:
            The grader's structured verdict.
        """
        try:
            return super()._invoke_grader(state, iteration, correction, context=context)
        except Exception as exc:
            if not _is_transient_grader_transport_error(exc):
                raise
            logger.warning(
                "Rubric grader transport failed; retrying grading once",
                exc_info=True,
            )
        return super()._invoke_grader(state, iteration, correction, context=context)

    async def _ainvoke_grader(
        self,
        state: RubricState,
        iteration: int,
        correction: str | None = None,
        *,
        context: object | None = None,
    ) -> GraderResponse:
        """Async variant of `_invoke_grader`. See that method for details.

        Returns:
            The grader's structured verdict.
        """
        try:
            return await super()._ainvoke_grader(
                state, iteration, correction, context=context
            )
        except Exception as exc:
            if not _is_transient_grader_transport_error(exc):
                raise
            logger.warning(
                "Rubric grader transport failed; retrying grading once",
                exc_info=True,
            )
        return await super()._ainvoke_grader(
            state, iteration, correction, context=context
        )
