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
    _strategy_from_result,  # noqa: PLC2701
)
from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphBubbleUp

from deepagents_code.goal_state_notice import is_conversation_control_message

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from deepagents.middleware.rubric import RubricEvaluation
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AnyMessage
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime

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

    @hook_config(can_jump_to=["model"])
    def after_agent(
        self,
        state: RubricState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Grade synchronously while preserving nested graph interrupts.

        Returns:
            The rubric state update, or `None` when no rubric is active.

        Raises:
            GraphBubbleUp: If the nested grader pauses or otherwise bubbles control.
        """
        prep = self._prepare_evaluation(state, runtime)
        if prep is None:
            return None
        grading_run_id, iteration = prep

        try:
            graded = self._grade(
                state,
                iteration,
                context=getattr(runtime, "context", None),
            )
        except GraphBubbleUp:
            raise
        except Exception as exc:  # noqa: BLE001
            return self._handle_grader_exception(
                runtime,
                state,
                grading_run_id,
                iteration,
                exc,
            )

        return self._finalize_evaluation(
            graded,
            state,
            runtime,
            grading_run_id,
            iteration,
        )

    async def aafter_agent(
        self,
        state: RubricState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Grade asynchronously while preserving nested graph interrupts.

        Returns:
            The rubric state update, or `None` when no rubric is active.

        Raises:
            GraphBubbleUp: If the nested grader pauses or otherwise bubbles control.
        """
        prep = self._prepare_evaluation(state, runtime)
        if prep is None:
            return None
        grading_run_id, iteration = prep

        try:
            graded = await self._agrade(
                state,
                iteration,
                context=getattr(runtime, "context", None),
            )
        except GraphBubbleUp:
            raise
        except Exception as exc:  # noqa: BLE001
            return self._handle_grader_exception(
                runtime,
                state,
                grading_run_id,
                iteration,
                exc,
            )

        return self._finalize_evaluation(
            graded,
            state,
            runtime,
            grading_run_id,
            iteration,
        )

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
    ) -> dict[str, Any]:
        """Build nested-grader input with a stable verification-operation ID.

        Returns:
            The nested grader's input state.
        """
        grading_run_id = state.get("_current_grading_run_id") or "untracked"
        grader_state = _without_internal_control_messages(state)
        payload = self._build_grader_payload(grader_state, iteration)
        return {
            "messages": [HumanMessage(content=payload)],
            "rubric_grading_operation_id": f"{grading_run_id}:{iteration}",
        }

    def _grade_once(
        self,
        state: RubricState,
        iteration: int,
        *,
        context: object | None,
    ) -> GraderResponse:
        grader = self._ensure_grader()
        metadata = self._grader_trace_metadata()
        self._record_grader_trace_metadata(metadata)
        result = grader.invoke(
            self._grader_input(state, iteration),
            config=self._grader_invocation_config(metadata),
            context=context,
        )
        self._record_grader_trace_metadata(
            self._grader_trace_metadata(
                effective_strategy=_strategy_from_result(result),
            )
        )
        return self._extract_graded(result)

    async def _agrade_once(
        self,
        state: RubricState,
        iteration: int,
        *,
        context: object | None,
    ) -> GraderResponse:
        grader = self._ensure_grader()
        metadata = self._grader_trace_metadata()
        self._record_grader_trace_metadata(metadata)
        result = await grader.ainvoke(
            self._grader_input(state, iteration),
            config=self._grader_invocation_config(metadata),
            context=context,
        )
        self._record_grader_trace_metadata(
            self._grader_trace_metadata(
                effective_strategy=_strategy_from_result(result),
            )
        )
        return self._extract_graded(result)

    def _grade(
        self,
        state: RubricState,
        iteration: int,
        *,
        context: object | None = None,
    ) -> GraderResponse:
        try:
            return self._grade_once(state, iteration, context=context)
        except Exception as exc:
            if not _is_transient_grader_transport_error(exc):
                raise
            logger.warning(
                "Rubric grader transport failed; retrying grading once",
                exc_info=True,
            )
        return self._grade_once(state, iteration, context=context)

    async def _agrade(
        self,
        state: RubricState,
        iteration: int,
        *,
        context: object | None = None,
    ) -> GraderResponse:
        try:
            return await self._agrade_once(state, iteration, context=context)
        except Exception as exc:
            if not _is_transient_grader_transport_error(exc):
                raise
            logger.warning(
                "Rubric grader transport failed; retrying grading once",
                exc_info=True,
            )
        return await self._agrade_once(state, iteration, context=context)
