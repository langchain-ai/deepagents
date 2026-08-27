"""CLI-specific rubric middleware customizations."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING, Annotated, Any, Literal, NotRequired, cast

from deepagents.middleware.rubric import (
    RUBRIC_GRADER_MESSAGE_SOURCE,
    GraderResponse,
    RubricMiddleware,
    RubricState,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    PrivateStateAttr,
)

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code.goal_state_notice import is_conversation_control_message
from deepagents_code.resume_state import (
    INHERIT_RUBRIC_MODEL,
    coerce_model_spec,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from deepagents.middleware.rubric import RubricEvaluation
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AnyMessage
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime


logger = logging.getLogger(__name__)


def _coerce_main_model_params(value: object) -> dict[str, Any]:
    """Copy checkpoint model params, or fail closed for malformed metadata.

    Args:
        value: Raw checkpoint value.

    Returns:
        A copied string-keyed mapping, or an empty mapping when malformed.
    """
    if not isinstance(value, Mapping):
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _model_specs_match(actual: str, requested: str) -> bool:
    """Compare canonical specs while tolerating one bare model name.

    Returns:
        Whether both values select the same model.
    """
    if (":" in actual) == (":" in requested):
        # Both canonical or both bare: only a literal match selects one model.
        return actual == requested
    # A mixed pair compares bare names. `split` mirrors `ModelSpec.parse`, so a
    # model id that itself contains a colon stays intact.
    return actual.split(":", 1)[-1] == requested.split(":", 1)[-1]


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


class ReliableRubricState(RubricState):
    """Rubric state carrying dcode's private runtime model selections.

    These three channels are declared a second time here, because this
    middleware's `state_schema` must list every channel it reads. Each
    annotation has to stay identical to its original in
    `resume_state.GoalRubricChannels` / `resume_state.ResumeState`: dropping a
    `PrivateStateAttr` marker leaks the field into the public graph input and
    output schema.
    """

    _model_spec: Annotated[NotRequired[str], PrivateStateAttr]
    """Active main model, written by `ConfigurableModelMiddleware`."""

    _model_params: Annotated[NotRequired[dict[str, Any] | None], PrivateStateAttr]
    """Params that belong to `_model_spec`, written alongside it."""

    _rubric_model_spec: Annotated[NotRequired[str], PrivateStateAttr]
    """Thread-scoped grader selection written by the TUI. Tri-state: absent,
    `resume_state.INHERIT_RUBRIC_MODEL`, or a model spec."""


class RubricGraderState(AgentState[GraderResponse]):
    """Nested-grader state used to scope verification-tool budgets."""

    rubric_grading_operation_id: NotRequired[str]


class ReliableRubricMiddleware(RubricMiddleware):
    """Run a context-aware nested grader with CLI verification middleware.

    The nested grader receives Deep Agents Code's verification middleware and
    runtime context without requiring those application-specific capabilities in
    the SDK's `RubricMiddleware`. The grader middleware stack owns model retries,
    so transient failures follow the same budget and taxonomy as every other
    dcode model call without replaying completed grader tools.

    The CLI configures the grader's `CodeModelRetryMiddleware` with hidden
    stream output. Grader messages use a nested namespace that both clients
    filter before rendering, so a dropped read or truncated body can retry the
    failed model node without duplicating visible output or replaying completed
    grader tools. Other model retry middleware instances keep the streamed-output
    guard enabled.

    The grader model is selected per request from thread state rather than
    fixed at construction. `inherit_main_model` supplies the default for a
    thread that has recorded no selection of its own.
    """

    # Widens the SDK's `RubricState` so LangGraph passes dcode's private
    # channels to this middleware. Without it `_grader_context` reads `None`
    # for every one of them and silently grades with the construction-time
    # model -- no error, no log, and the type checker stays happy.
    state_schema = ReliableRubricState

    def __init__(  # noqa: D107
        self,
        *,
        model: str | BaseChatModel,
        system_prompt: str | None = None,
        tools: Sequence[BaseTool] | None = None,
        grader_middleware: Sequence[AgentMiddleware[Any, Any]] | None = None,
        grader_context_schema: type[Any] | None = None,
        runtime_bootstrap_model: str | BaseChatModel | None = None,
        inherit_main_model: bool = False,
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
        self._runtime_bootstrap_model = runtime_bootstrap_model
        self._runtime_grader: Any = None
        self._inherit_main_model = inherit_main_model
        self._runtime_grader_model: ContextVar[str | None] = ContextVar(
            "runtime_grader_model",
            default=None,
        )

    @contextmanager
    def _runtime_grader_trace(self, model: str | None) -> Iterator[None]:
        """Scope trace diagnostics to one request's selected grader model."""
        token = self._runtime_grader_model.set(model)
        try:
            yield
        finally:
            self._runtime_grader_model.reset(token)

    def _grader_trace_metadata(
        self,
        *,
        effective_strategy: Literal["ProviderStrategy", "ToolStrategy"] | None = None,
    ) -> dict[str, str]:
        """Build diagnostics for the request-local grader selection.

        Returns:
            The configured model label and effective structured-output strategy.
        """
        runtime_model = self._runtime_grader_model.get()
        if runtime_model is not None and effective_strategy is None:
            # A runtime string is resolved inside the nested graph, so the
            # construction-time model cannot predict its output strategy;
            # deriving one from `self._model` would only be discarded.
            return {
                "rubric_grader_configured_model": runtime_model,
                "rubric_grader_effective_strategy": "unknown",
            }
        metadata = super()._grader_trace_metadata(
            effective_strategy=effective_strategy,
        )
        if runtime_model is not None:
            metadata["rubric_grader_configured_model"] = runtime_model
        return metadata

    @staticmethod
    def _context(context: object | None) -> CLIContextSchema:
        """Copy the parent runtime context into the nested grader schema.

        Returns:
            An independent context safe to customize for one grader call.
        """
        resolved = CLIContextSchema.from_payload(context)
        if resolved is None:
            if context is not None:
                # Defaults silently drop `approval_mode`/`auto_approve`, so a
                # yolo session would become approval-gated inside the nested
                # grader. That is a wiring bug, not a normal state.
                logger.warning(
                    "Unrecognized grader context type %s; using defaults",
                    type(context).__name__,
                )
            return CLIContextSchema()
        # The parent context is shared across concurrent grader calls; copy the
        # mutable containers so one call cannot mutate another's.
        return replace(
            resolved,
            model_params=dict(resolved.model_params),
            profile_overrides=dict(resolved.profile_overrides),
            hooks_server_events=list(resolved.hooks_server_events),
        )

    def _grader_context(
        self, state: ReliableRubricState, context: object | None
    ) -> CLIContextSchema:
        """Select the effective grader model without mutating shared middleware.

        `_rubric_model_spec` is a tri-state, so there are three outcomes:

        - the inheritance sentinel, or no selection while `inherit_main_model`
          is set, grades with the active main model;
        - a recorded spec grades with that dedicated model;
        - no selection while a construction-time grader model was configured
          leaves `model` unset, which selects that model.

        Returns:
            Request-local grader context carrying the selected model and params.
        """
        grader_context = self._context(context)
        selected = coerce_model_spec(state.get("_rubric_model_spec"))
        inherit = selected == INHERIT_RUBRIC_MODEL or (
            selected is None and self._inherit_main_model
        )
        if inherit:
            # Model and params are resolved as a unit. `_model_spec` is written
            # only after a main-model call, so on a thread's first grading pass
            # the channel is absent and the parent context still holds the live
            # `/model` override -- along with the params that belong to it.
            main_model = coerce_model_spec(state.get("_model_spec"))
            if main_model is not None:
                requested_model = coerce_model_spec(grader_context.model)
                grader_context.model = main_model
                grader_context.model_params = (
                    _coerce_main_model_params(state.get("_model_params"))
                    if requested_model is None
                    or _model_specs_match(main_model, requested_model)
                    else {}
                )
        else:
            # `selected` is either a recorded spec or `None`; `None` means "no
            # runtime override", which selects the model the grader was built
            # with. Clearing the parent's model is deliberate.
            grader_context.model = selected
            grader_context.model_params = {}
        return grader_context

    def _build_grader(self, model: str | BaseChatModel) -> tuple[Any, BaseChatModel]:
        """Build a nested grader around a resolved base model.

        Returns:
            The grader graph and its resolved base model.
        """
        from deepagents._models import (  # noqa: PLC2701
            resolve_model,
        )
        from langchain.agents import create_agent

        resolved_model = resolve_model(model)
        grader = create_agent(
            model=resolved_model,
            system_prompt=self._system_prompt,
            tools=self._tools,
            middleware=self._grader_middleware,
            name=RUBRIC_GRADER_MESSAGE_SOURCE,
            response_format=GraderResponse,
            state_schema=RubricGraderState,
            context_schema=self._grader_context_schema,
        )
        return grader, resolved_model

    def _ensure_grader(self) -> Any:  # noqa: ANN401
        if self._grader is not None:
            return self._grader
        runtime_model = self._runtime_grader_model.get()
        if runtime_model is not None and self._runtime_bootstrap_model is not None:
            if self._runtime_grader is None:
                self._runtime_grader, _ = self._build_grader(
                    self._runtime_bootstrap_model
                )
            return self._runtime_grader
        self._grader, self._resolved_model = self._build_grader(self._model)
        return self._grader

    def _invoke_grader(
        self,
        state: RubricState,
        iteration: int,
        correction: str | None = None,
        *,
        context: object | None = None,
    ) -> GraderResponse:
        """Invoke the nested grader with the state-selected model context.

        Returns:
            Parsed grader response.
        """
        reliable_state = cast("ReliableRubricState", state)
        grader_context = self._grader_context(reliable_state, context)
        with self._runtime_grader_trace(grader_context.model):
            return super()._invoke_grader(
                state,
                iteration,
                correction,
                context=grader_context,
            )

    async def _ainvoke_grader(
        self,
        state: RubricState,
        iteration: int,
        correction: str | None = None,
        *,
        context: object | None = None,
    ) -> GraderResponse:
        """Invoke the nested grader asynchronously with state-selected context.

        Returns:
            Parsed grader response.
        """
        reliable_state = cast("ReliableRubricState", state)
        grader_context = self._grader_context(reliable_state, context)
        with self._runtime_grader_trace(grader_context.model):
            return await super()._ainvoke_grader(
                state,
                iteration,
                correction,
                context=grader_context,
            )

    def _handle_grader_exception(
        self,
        runtime: Runtime[Any],
        state: RubricState,
        grading_run_id: str,
        iteration: int,
        exc: Exception,
    ) -> dict[str, Any]:
        reliable_state = cast("ReliableRubricState", state)
        context = self._grader_context(
            reliable_state,
            getattr(runtime, "context", None),
        )
        with self._runtime_grader_trace(context.model):
            return super()._handle_grader_exception(
                runtime,
                state,
                grading_run_id,
                iteration,
                exc,
            )

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
