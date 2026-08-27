"""CLI-specific rubric middleware customizations."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, cast

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

from deepagents_code._cli_context import CLIContext, CLIContextSchema
from deepagents_code.goal_state_notice import is_conversation_control_message
from deepagents_code.resume_state import (
    INHERIT_RUBRIC_MODEL,
    coerce_rubric_model_spec,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from deepagents.middleware.rubric import RubricEvaluation
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AnyMessage
    from langchain_core.tools import BaseTool


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
    """Rubric state carrying dcode's private runtime model selections."""

    _model_spec: Annotated[NotRequired[str], PrivateStateAttr]
    _model_params: Annotated[NotRequired[dict[str, Any] | None], PrivateStateAttr]
    _rubric_model_spec: Annotated[NotRequired[str], PrivateStateAttr]


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
    """

    state_schema = ReliableRubricState

    def __init__(  # noqa: D107
        self,
        *,
        model: str | BaseChatModel,
        system_prompt: str | None = None,
        tools: Sequence[BaseTool] | None = None,
        grader_middleware: Sequence[AgentMiddleware[Any, Any]] | None = None,
        grader_context_schema: type[Any] | None = None,
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
        self._inherit_main_model = inherit_main_model

    @staticmethod
    def _context(context: object | None) -> CLIContextSchema:
        """Copy the parent runtime context into the nested grader schema.

        Returns:
            An independent context safe to customize for one grader call.
        """
        if isinstance(context, CLIContextSchema):
            return replace(
                context,
                model_params=dict(context.model_params),
                profile_overrides=dict(context.profile_overrides),
                hooks_server_events=list(context.hooks_server_events),
            )
        if isinstance(context, dict):
            payload = cast("CLIContext", context)
            return CLIContextSchema(
                model=payload.get("model"),
                model_params=dict(payload.get("model_params") or {}),
                profile_overrides=dict(payload.get("profile_overrides") or {}),
                model_context_limit=payload.get("model_context_limit"),
                classifier_model=payload.get("classifier_model"),
                approval_mode=payload.get("approval_mode") or "manual",
                auto_approve=bool(payload.get("auto_approve", False)),
                approval_mode_key=payload.get("approval_mode_key"),
                thread_id=payload.get("thread_id"),
                turn_id=payload.get("turn_id"),
                hooks_snapshot_id=payload.get("hooks_snapshot_id"),
                hooks_server_events=list(payload.get("hooks_server_events") or []),
                prompt_id=payload.get("prompt_id"),
            )
        return CLIContextSchema()

    def _grader_context(
        self, state: ReliableRubricState, context: object | None
    ) -> CLIContextSchema:
        """Select the effective grader model without mutating shared middleware.

        Returns:
            Request-local grader context carrying the selected model and params.
        """
        grader_context = self._context(context)
        selected = coerce_rubric_model_spec(state.get("_rubric_model_spec"))
        inherit = selected == INHERIT_RUBRIC_MODEL or (
            selected is None and self._inherit_main_model
        )
        if inherit:
            grader_context.model = state.get("_model_spec") or grader_context.model
            grader_context.model_params = dict(state.get("_model_params") or {})
        elif selected:
            grader_context.model = selected
            grader_context.model_params = {}
        else:
            grader_context.model = None
            grader_context.model_params = {}
        return grader_context

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
        return super()._invoke_grader(
            state,
            iteration,
            correction,
            context=self._grader_context(reliable_state, context),
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
        return await super()._ainvoke_grader(
            state,
            iteration,
            correction,
            context=self._grader_context(reliable_state, context),
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
