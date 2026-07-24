"""Rubric middleware with dcode retry policies for the nested grader."""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
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


def _model_params_mapping(context: object | None) -> Mapping[str, Any] | None:
    """Return a context's model-params mapping when present."""
    if context is None:
        return None
    if isinstance(context, Mapping):
        params = context.get("model_params")
    else:
        params = getattr(context, "model_params", None)
    if isinstance(params, Mapping):
        return cast("Mapping[str, Any]", params)
    return None


def _with_model_params(
    context: object | None,
    params: Mapping[str, Any],
) -> object:
    """Return a context carrying `params` without mutating the original object.

    Prefers the caller's context type when it exposes replaceable `model_params`:
    mappings are shallow-copied, while dataclass-like / pydantic-style objects are
    rebuilt through their constructor. Falls back to a plain mapping when no
    context is provided.
    """
    if context is None:
        return {"model_params": dict(params)}
    if isinstance(context, Mapping):
        updated = dict(context)
        updated["model_params"] = dict(params)
        return updated
    try:
        return type(context)(model_params=dict(params))
    except TypeError:
        return {"model_params": dict(params)}


class RubricGraderState(AgentState[GraderResponse]):
    """Nested-grader state used to scope verification-tool budgets."""

    rubric_grading_operation_id: NotRequired[str]


class ReliableRubricMiddleware(RubricMiddleware):
    """Run a context-aware nested grader with dcode retry policies.

    The nested grader receives Deep Agents Code's verification middleware and
    runtime context without requiring those application-specific capabilities in
    the SDK's `RubricMiddleware`. Model-node retries are applied exclusively to
    the nested grader agent. A separate whole-grader transport retry re-invokes
    only the grader on mid-response transport failures so grader tools must stay
    read-only or idempotent.
    """

    def __init__(
        self,
        *,
        model: str | BaseChatModel,
        system_prompt: str | None = None,
        tools: Sequence[BaseTool] | None = None,
        grader_middleware: Sequence[AgentMiddleware[Any, Any]] | None = None,
        grader_context_schema: type[Any] | None = None,
        max_iterations: int = 3,
        on_evaluation: Callable[[RubricEvaluation], None] | None = None,
        model_retry_override: int | None = None,
        model_retry_fallback: int | None = None,
    ) -> None:
        """Initialize rubric grading with nested middleware and retry budgets.

        Args:
            model: Model or model identifier used by the grader.
            system_prompt: Custom grading instructions.
            tools: Read-only tools available to the grader.
            grader_middleware: Extra middleware installed on the nested grader
                (budgets, HITL, etc.).
            grader_context_schema: Context schema for the nested grader agent.
            max_iterations: Maximum rubric iterations.
            on_evaluation: Optional callback for completed evaluations.
            model_retry_override: Explicit `--max-retries` value inherited from
                the current process, or `None` to use request/checkpoint metadata.
            model_retry_fallback: Caller-provided budget for models without
                attached metadata.

        Raises:
            TypeError: If a retry value is not an integer.
            ValueError: If a retry value is negative.
        """
        retry_values = {
            "model_retry_override": model_retry_override,
            "model_retry_fallback": model_retry_fallback,
        }
        for name, value in retry_values.items():
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                msg = f"`{name}` must be an int or None"
                raise TypeError(msg)
            if value is not None and value < 0:
                msg = f"`{name}` must be non-negative"
                raise ValueError(msg)
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            max_iterations=max_iterations,
            on_evaluation=on_evaluation,
        )
        self._grader_middleware = list(grader_middleware or ())
        self._grader_context_schema = grader_context_schema
        self._model_retry_override = model_retry_override
        self._model_retry_fallback = model_retry_fallback

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
        """Create the grader with model-node retries and nested middleware.

        Returns:
            The cached or newly constructed grader agent.
        """
        if self._grader is not None:
            return self._grader

        from langchain.agents import create_agent

        from deepagents_code.config import (
            CLI_MAX_RETRIES_KEY,
            DEFAULT_MODEL_RETRIES,
            create_model,
            set_model_retry_metadata,
        )
        from deepagents_code.model_retry import CodeModelRetryMiddleware

        retry_fallback = (
            self._model_retry_override
            if self._model_retry_override is not None
            else self._model_retry_fallback
            if self._model_retry_fallback is not None
            else DEFAULT_MODEL_RETRIES
        )
        if isinstance(self._model, str):
            retry_kwargs = (
                {CLI_MAX_RETRIES_KEY: self._model_retry_override}
                if self._model_retry_override is not None
                else None
            )
            grader_model = create_model(
                self._model,
                extra_kwargs=retry_kwargs,
            ).model
            if (
                self._model_retry_override is not None
                or self._model_retry_fallback is not None
            ):
                set_model_retry_metadata(
                    grader_model,
                    retries=retry_fallback,
                    cli_override=self._model_retry_override,
                )
        else:
            grader_model = self._model

        self._resolved_model = grader_model
        middleware: list[AgentMiddleware[Any, Any]] = [
            CodeModelRetryMiddleware(max_retries=retry_fallback),
            *self._grader_middleware,
        ]
        self._grader = create_agent(
            model=grader_model,
            system_prompt=self._system_prompt,
            tools=self._tools,
            middleware=middleware,
            name=RUBRIC_GRADER_MESSAGE_SOURCE,
            response_format=GraderResponse,
            state_schema=RubricGraderState,
            context_schema=self._grader_context_schema,
        )
        return self._grader

    def _grader_context(
        self,
        state: RubricState,
        context: object | None,
    ) -> object | None:
        """Merge request-local retry metadata into the nested grader context.

        Prefer the parent runtime context (including approval mode and any
        existing `model_params`) and only rewrite it when this middleware owns an
        explicit override or the checkpoint/state carrier must supply one.

        Returns:
            Context passed to the nested grader, or `None` when no context is
            available and no retry override needs to be constructed.
        """
        from deepagents_code.config import CLI_MAX_RETRIES_KEY, is_valid_retry_count

        retry_override = self._model_retry_override
        if retry_override is None:
            existing = _model_params_mapping(context)
            if existing is not None:
                raw = existing.get(CLI_MAX_RETRIES_KEY)
                if is_valid_retry_count(raw):
                    return context
            model_params = cast("dict[str, Any]", state).get("_model_params")
            if isinstance(model_params, Mapping):
                raw = model_params.get(CLI_MAX_RETRIES_KEY)
                if is_valid_retry_count(raw):
                    retry_override = raw

        if retry_override is None:
            return context

        existing_params = _model_params_mapping(context)
        base_params: MutableMapping[str, Any] = (
            dict(existing_params) if existing_params is not None else {}
        )
        base_params[CLI_MAX_RETRIES_KEY] = retry_override
        return _with_model_params(context, base_params)

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
            context=self._grader_context(state, context),
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
            context=self._grader_context(state, context),
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
