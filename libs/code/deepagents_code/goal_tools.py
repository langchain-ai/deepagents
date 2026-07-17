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
    ContextT,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import override

# Runtime (not TYPE_CHECKING) imports. `GoalRubricChannels` supplies the shared
# `PrivateStateAttr`-marked goal/rubric channels that `GoalToolState` extends, so
# the markers are declared once (see that class). `coerce_goal_status` is used at
# runtime by `_goal_snapshot`; `GoalStatus` types its result and snapshot fields.
from deepagents_code.resume_state import (
    GoalRubricChannels,
    GoalStatus,
    coerce_goal_status,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

RubricSource = Literal["goal", "sticky", "invocation"]
"""Where the active rubric criteria came from, as reported to the model."""

GOAL_TOOLS_SYSTEM_PROMPT = """## Goal and Rubric Tools

Use `get_rubric` to inspect active acceptance criteria before deciding whether work is
complete.
When a goal is active, use `get_goal` to inspect the objective and current status.
A paused goal is persisted for later but must not drive work until the user resumes it.
A goal is marked complete automatically when its current grading turn satisfies the
accepted criteria. Use `update_goal` to report a blocker; `status="complete"` remains
available for optional completion evidence but is not required."""
"""Model-visible guidance injected before each request by `GoalToolsMiddleware`."""

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
    """Whether the goal is actionable (should drive work).

    Derived from `status`: `active` and `blocked` goals are actionable, while
    `paused` and `complete` goals are not. Note a `paused` goal is unfinished
    yet reports `active=False`. `False` when no goal is set (the
    `objective is None` branch), where `status` is also `None`.
    """

    objective: str | None
    """Active goal objective, or `None` when no goal is set."""

    status: GoalStatus | None
    """Lifecycle status, or `None` when no goal is set.

    A set-but-unlabeled or unrecognized persisted value is normalized to
    `"active"` by `coerce_goal_status`, so this is always a known `GoalStatus`
    when a goal is set.
    """

    criteria: str | None
    """Persisted goal criteria, or shared rubric criteria when no goal rubric exists."""

    note: str | None
    """Persisted completion evidence or blocker note for the goal."""


class GoalToolState(GoalRubricChannels):
    """State fields used by goal tools.

    Inherits the shared `_goal_*`/`_sticky_rubric` channels (with their
    `PrivateStateAttr` markers) from `GoalRubricChannels`, so the goal tools and
    `ResumeState` cannot drift apart. Adds only the public `rubric` graph input,
    which is intentionally non-private — it is the `RubricMiddleware` input.
    """

    rubric: NotRequired[str | None]
    """Public `RubricMiddleware` graph input (intentionally non-private).

    Distinct from the TUI-owned `_sticky_rubric`: this is the per-invocation
    rubric passed in via the graph schema, not checkpointed TUI state.
    """


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
    status = coerce_goal_status(state.get("_goal_status")) or "active"
    goal_is_actionable = objective is not None and status in {"active", "blocked"}
    sticky_is_goal_rubric = objective is not None and sticky_rubric == goal_rubric

    source: RubricSource | None = None
    if criteria is not None:
        if goal_is_actionable and goal_rubric == criteria:
            source = "goal"
        elif sticky_rubric == criteria and not sticky_is_goal_rubric:
            source = "sticky"
        else:
            source = "invocation"
    # Fallback branches below run only when there is no public `rubric` input,
    # so `invocation` is unreachable here by construction — the criteria can
    # only be attributed to an actionable `goal` or a standalone `sticky` rubric.
    elif goal_is_actionable and goal_rubric is not None:
        criteria = goal_rubric
        source = "goal"
    elif sticky_rubric is not None and not sticky_is_goal_rubric:
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
    criteria = _clean_state_text(state, "_goal_rubric") or rubric["criteria"]
    note = _clean_state_text(state, "_goal_status_note")
    return {
        # Blocked goals remain actionable, while paused and complete goals do not
        # drive work until the user changes their state.
        "active": status in {"active", "blocked"},
        "objective": objective,
        "status": status,
        "criteria": criteria,
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
        Command updating goal metadata and returning a tool response.
            A `complete` request stages `_pending_goal_completion_note` for
            the TUI to resolve once the rubric verdict lands, rather than
            committing the status directly; `blocked` commits immediately.

            When no goal is set or `note` is empty, nothing is committed
            and the `ToolMessage` explains what the model must do instead.
    """
    # Enforced preconditions here are only: an active goal exists and `note` is
    # non-empty. Completion is staged because `RubricMiddleware` records its
    # final verdict after the model stops making tool calls; the TUI resolves
    # the staged request during post-turn checkpoint sync.
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
    goal_status = coerce_goal_status(state.get("_goal_status")) or "active"
    if goal_status in {"paused", "complete"}:
        if goal_status == "paused":
            message = (
                "The goal is paused. The user must run `/goal resume` before its "
                "status can be updated."
            )
        else:
            message = "The goal is already complete and cannot be updated."
        return Command(
            update={
                "messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]
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
    if status == "complete":
        return Command(
            update={
                "_pending_goal_completion_note": clean_note,
                "messages": [
                    ToolMessage(
                        content=(
                            "Goal completion requested. It will be recorded if "
                            "the accepted rubric is satisfied."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    return Command(
        update={
            "_goal_status": status,
            "_goal_status_note": clean_note,
            "_pending_goal_completion_note": None,
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
            """Report a blocked goal or attach optional completion evidence.

            Use `blocked` when you cannot proceed without user input. Goals complete
            automatically after a satisfied goal-backed grading turn, so `complete`
            is optional and only stages its evidence for that result. Do not create,
            pause, resume, clear, or replace goals — those are user-controlled.

            Args:
                status: `complete` to attach completion evidence, or `blocked` when
                    you are stuck and need the user.
                note: Evidence the criteria are satisfied, or the specific
                    blocker. Required when calling this tool.
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
