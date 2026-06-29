"""Goal tools exposed to the agent for persisted TUI goals."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    NotRequired,
    TypedDict,
    TypeVar,
    cast,
)

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import override

# Runtime (not TYPE_CHECKING) import: LangChain resolves the state-schema
# annotations via `get_type_hints(include_extras=True)` to find the
# `PrivateStateAttr` markers, so `GoalStatus` must be importable at runtime or
# schema construction raises `NameError`. `coerce_goal_status` is used at
# runtime by `_goal_snapshot`.
from deepagents_code.resume_state import GoalStatus, coerce_goal_status

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

RubricSource = Literal["goal", "sticky", "invocation"]
"""Where the active rubric criteria came from, as reported to the model."""

GOAL_TOOLS_SYSTEM_PROMPT = (
    "The user may set a persistent goal with `/goal` and acceptance criteria "
    "with `/rubric` or `/criteria`. Use `get_rubric` to inspect the current "
    "criteria. When a goal is active, use `get_goal` to inspect the objective, "
    "accepted criteria, and status. Use `update_goal` only when you have "
    "evidence that the goal is complete or blocked. Do not pause, resume, clear, "
    "replace, or create goals yourself; those are user-controlled lifecycle "
    "actions."
)

ResponseT = TypeVar("ResponseT")


def _runtime_blocked_goal_retry_context(ctx: object) -> str | None:
    """Return blocked-goal retry context from LangGraph runtime context."""
    if isinstance(ctx, dict):
        value = ctx.get("blocked_goal_retry_context")
    else:
        value = getattr(ctx, "blocked_goal_retry_context", None)
    return value if isinstance(value, str) and value else None


class RubricSnapshot(TypedDict):
    """Read-only rubric view returned by the `get_rubric` tool to the model."""

    active: bool
    """Whether acceptance criteria are currently available."""

    criteria: str | None
    """Current acceptance criteria, or `None` when no rubric is set."""

    source: RubricSource | None
    """Where the criteria came from: `goal`, `sticky`, `invocation`, or `None`."""

    grading_status: str | None
    """Latest `RubricMiddleware` grading status for the in-progress or
    just-completed graded turn, or `None`.

    The middleware clears this at the start of the next graded turn, so
    a `None` does not imply grading never ran.
    """


class GoalSnapshot(TypedDict):
    """Read-only goal view returned by the `get_goal` tool to the model.

    A fixed-shape projection of goal state. Both construction branches in
    `_goal_snapshot` must populate every key, so the type checker catches a
    drift between them.
    """

    active: bool
    """Whether the goal is unfinished (derived as `status != "complete"`)."""

    objective: str | None
    """Active goal objective, or `None` when no goal is set."""

    status: GoalStatus | None
    """Lifecycle status, or `None` when no goal is set.

    A set-but-unlabeled or unrecognized persisted value is normalized to
    `"active"` by `coerce_goal_status`, so this is always a known `GoalStatus`
    when a goal is set.
    """

    criteria: str | None
    """Accepted criteria (from the shared rubric snapshot)."""

    note: str | None
    """Latest evidence or blocker note recorded by `update_goal`."""


class GoalToolState(AgentState):
    """State fields used by goal tools.

    The `_goal_*` channels mirror `ResumeState` and must carry `PrivateStateAttr`
    so they stay out of the public graph input/output schema. Middleware state
    schemas are merged with later entries winning, so bare (non-private)
    re-declarations here would override `ResumeState`'s private annotations and
    leak the fields into the public schema.
    """

    _goal_objective: Annotated[NotRequired[str | None], PrivateStateAttr]
    _goal_status: Annotated[NotRequired[GoalStatus | None], PrivateStateAttr]
    _goal_rubric: Annotated[NotRequired[str | None], PrivateStateAttr]
    _goal_status_note: Annotated[NotRequired[str | None], PrivateStateAttr]
    _sticky_rubric: Annotated[NotRequired[str | None], PrivateStateAttr]
    rubric: NotRequired[str | None]


def _clean_state_text(state: dict[str, Any], key: str) -> str | None:
    """Return a non-empty string from state, or `None`."""
    value = state.get(key)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _rubric_snapshot(state: dict[str, Any]) -> RubricSnapshot:
    """Build the `get_rubric` response from graph state.

    Args:
        state: Current graph state injected by LangGraph.

    Returns:
        Rubric snapshot visible to the model.
    """
    criteria = _clean_state_text(state, "rubric")
    goal_rubric = _clean_state_text(state, "_goal_rubric")
    sticky_rubric = _clean_state_text(state, "_sticky_rubric")
    objective = _clean_state_text(state, "_goal_objective")

    source: RubricSource | None = None
    if criteria is not None:
        if objective is not None and goal_rubric == criteria:
            source = "goal"
        elif sticky_rubric == criteria:
            source = "sticky"
        else:
            source = "invocation"
    # Fallback branches below run only when there is no public `rubric` input,
    # so `invocation` is unreachable here by construction — the criteria can
    # only be attributed to a `goal` or a `sticky` rubric.
    elif objective is not None and goal_rubric is not None:
        criteria = goal_rubric
        source = "goal"
    elif sticky_rubric is not None:
        criteria = sticky_rubric
        source = "sticky"

    # `_rubric_status` is owned by the SDK's `RubricMiddleware`, co-composed into
    # this agent's graph; see the `grading_status` field docstring above.
    grading_status = _clean_state_text(state, "_rubric_status")
    return {
        "active": criteria is not None,
        "criteria": criteria,
        "source": source,
        "grading_status": grading_status,
    }


def _goal_snapshot(state: dict[str, Any]) -> GoalSnapshot:
    """Build the `get_goal` response from graph state.

    Args:
        state: Current graph state injected by LangGraph.

    Returns:
        Goal snapshot visible to the model.
    """
    objective = _clean_state_text(state, "_goal_objective")
    rubric = _rubric_snapshot(state)
    if objective is None:
        return {
            "active": False,
            "objective": None,
            "status": None,
            "criteria": rubric["criteria"],
            "note": None,
        }
    # A set-but-unlabeled or unrecognized status defaults to "active"; an
    # unknown persisted value never leaks to the model as a bogus status.
    status: GoalStatus = coerce_goal_status(state.get("_goal_status")) or "active"
    note = _clean_state_text(state, "_goal_status_note")
    return {
        # A goal is active until it is complete; `blocked` is still unfinished.
        # Derive `active` from `status` so the two never disagree.
        "active": status != "complete",
        "objective": objective,
        "status": status,
        "criteria": rubric["criteria"],
        "note": note,
    }


def _update_goal_command(
    *,
    status: Literal["complete", "blocked"],
    note: str,
    tool_call_id: str,
    state: dict[str, Any],
) -> Command[Any]:
    """Build the constrained `update_goal` command.

    Args:
        status: Goal status the model is reporting (`complete` or `blocked`).
        note: Evidence the goal is complete, or the specific blocker. Required;
            the status is not committed without it.
        tool_call_id: Tool call ID for the returned `ToolMessage`.
        state: Current graph state injected by LangGraph.

    Returns:
        Command updating goal metadata and returning a tool response. When no
        goal is set or `note` is empty, no status is committed and the
        `ToolMessage` explains what the model must do instead.
    """
    # Enforced preconditions are only: an active goal exists and `note` is
    # non-empty. Criteria satisfaction is intentionally NOT verified here — the
    # rubric grader (`RubricMiddleware`) owns that. The tool docstring's "use
    # complete only when the criteria are satisfied" is model guidance, not an
    # enforced contract; do not add a criteria check here.
    objective = state.get("_goal_objective")
    if not isinstance(objective, str) or not objective:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No active goal is set.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
    clean_note = note.strip()
    if not clean_note:
        # Evidence is required: refuse to commit a status with no justification
        # rather than silently storing an empty note.
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"Provide a note with evidence before marking the "
                            f"goal {status}."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
    return Command(
        update={
            "_goal_status": status,
            "_goal_status_note": clean_note,
            "messages": [
                ToolMessage(
                    content=f"Goal marked {status}. {clean_note}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


class GoalToolsMiddleware(AgentMiddleware[GoalToolState, ContextT]):
    """Expose constrained goal tools to the main agent."""

    state_schema = GoalToolState

    def __init__(self) -> None:
        """Initialize goal tools."""
        super().__init__()

        @tool
        def get_rubric(
            state: Annotated[dict[str, Any], InjectedState],
        ) -> RubricSnapshot:
            """Read the current acceptance criteria used to evaluate completion.

            Call this to inspect the active rubric, whether it came from a goal,
            a sticky rubric, or the current invocation, and the latest grading
            status if a graded turn has already run.

            Returns:
                Rubric snapshot with `active`, `criteria`, `source`, and
                `grading_status` keys.
            """
            return _rubric_snapshot(state)

        @tool
        def get_goal(
            state: Annotated[dict[str, Any], InjectedState],
        ) -> GoalSnapshot:
            """Read the current persistent goal and acceptance criteria.

            Call this before deciding whether work is done to see the objective,
            the current acceptance criteria (which may come from the goal or a
            standalone rubric), the current status, and any prior note.

            Returns:
                Goal snapshot with `active`, `objective`, `status`, `criteria`,
                and `note` keys.
            """
            return _goal_snapshot(state)

        @tool
        def update_goal(
            status: Literal["complete", "blocked"],
            note: str,
            tool_call_id: Annotated[str, InjectedToolCallId],
            state: Annotated[dict[str, Any], InjectedState],
        ) -> Command[Any]:
            """Mark the current goal complete or blocked with evidence.

            Use `complete` only when the accepted criteria are satisfied; use
            `blocked` when you cannot proceed without user input. Do not create,
            pause, resume, clear, or replace goals — those are user-controlled.

            Args:
                status: `complete` when the criteria are met, `blocked` when you
                    are stuck and need the user.
                note: Evidence the criteria are satisfied, or the specific
                    blocker. Required — the status is not recorded without it.
                tool_call_id: Injected tool call ID for the tool response.
                state: Injected graph state holding the current goal.

            Returns:
                Command that updates goal status and returns a tool message.
            """
            return _update_goal_command(
                status=status,
                note=note,
                tool_call_id=tool_call_id,
                state=state,
            )

        self.tools = [get_rubric, get_goal, update_goal]

    @staticmethod
    def _request_with_goal_system_context(
        request: ModelRequest[ContextT],
    ) -> ModelRequest[ContextT]:
        """Inject goal guidance and any one-turn retry context.

        Returns:
            Model request with goal context appended to the system prompt.
        """
        retry_context = _runtime_blocked_goal_retry_context(request.runtime.context)
        prompt_parts = [GOAL_TOOLS_SYSTEM_PROMPT]
        if retry_context is not None:
            prompt_parts.append(retry_context)
        prompt = "\n\n".join(prompt_parts)

        if request.system_message is not None:
            content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{prompt}"},
            ]
        else:
            content = [{"type": "text", "text": prompt}]
        return request.override(
            system_message=SystemMessage(
                content=cast("list[str | dict[str, str]]", content)
            )
        )

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Inject goal-tool guidance into each model request.

        Returns:
            Model response from the wrapped handler.
        """
        return handler(self._request_with_goal_system_context(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        """Inject goal-tool guidance into each async model request.

        Returns:
            Model response from the wrapped handler.
        """
        return await handler(self._request_with_goal_system_context(request))
