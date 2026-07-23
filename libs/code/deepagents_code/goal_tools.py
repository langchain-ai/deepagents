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
)
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import override

# Runtime (not TYPE_CHECKING) imports. `GoalRubricChannels` supplies the shared
# `PrivateStateAttr`-marked goal/rubric channels that `GoalToolState` extends, so
# the markers are declared once (see that class). `coerce_goal_status` is used at
# runtime by `_goal_snapshot`; `GoalStatus` types its result and snapshot fields.
from deepagents_code.goal_state_notice import (
    build_goal_state_notice,
    goal_state_fingerprint,
    has_goal_or_rubric_state,
    latest_goal_state_message_index,
    latest_goal_state_notice,
    latest_human_is_unsaved_goal_continuation,
)
from deepagents_code.resume_state import (
    GoalRubricChannels,
    GoalStatus,
    coerce_goal_status,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langgraph.runtime import Runtime

RubricSource = Literal["goal", "sticky", "invocation"]
"""Where the active rubric criteria came from, as reported to the model."""

GOAL_TOOLS_SYSTEM_PROMPT = """## Goal and Rubric Tools

Consult the latest goal/rubric state notice in conversation history before using
these tools. Later notices supersede earlier notices. If no notice exists, assume
there is no actionable goal or rubric and do not call these tools.

When the latest notice says a rubric is active, use `get_rubric` when its exact
acceptance criteria are needed. When it says a goal is actionable, use `get_goal`
when its objective or current status is needed. Paused and completed goals must not
drive work. Use `update_goal` only for an actionable goal: report a blocker with it;
`status="complete"` remains available for optional completion evidence but is not
required. Private checkpoint state and the tools remain authoritative for details."""
"""Static model-visible guidance injected by `GoalToolsMiddleware`."""

GOAL_TOOL_NAMES = frozenset({"get_goal", "get_rubric", "update_goal"})
"""Tool names used by behavioral absence gates and middleware contract tests."""


def _goal_state_notice_for(
    state: dict[str, Any],
    messages: Sequence[object],
) -> HumanMessage | None:
    """Build a notice when effective history lacks current goal/rubric state.

    Args:
        state: Authoritative middleware state.
        messages: Messages visible at the next model boundary.

    Returns:
        Current notice to append, or `None` when history is already authoritative.
    """
    if latest_human_is_unsaved_goal_continuation(messages):
        return None
    latest = latest_goal_state_notice(messages)
    latest_candidate = latest_goal_state_message_index(messages)
    fingerprint = goal_state_fingerprint(state)
    if (
        latest is not None
        and latest[0] == latest_candidate
        and latest[1]["state_fingerprint"] == fingerprint
    ):
        return None
    if latest_candidate is None and not has_goal_or_rubric_state(state):
        return None
    return build_goal_state_notice(state)


ResponseT = TypeVar("ResponseT")


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
    update = {
        "_goal_status": status,
        "_goal_status_note": clean_note,
        "_pending_goal_completion_note": None,
    }
    return Command(
        update={
            **update,
            "messages": [
                ToolMessage(
                    content=f"Goal marked {status}. {clean_note}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


class GoalToolsMiddleware(AgentMiddleware[GoalToolState, ContextT]):
    """Expose constrained goal tools and maintain the goal-state notice.

    Besides registering `get_goal`/`get_rubric`/`update_goal`, this middleware
    keeps the model oriented at each model boundary: `before_model` persists a
    fresh goal-state notice into checkpointed history when the latest one no
    longer matches authoritative state, and `wrap_model_call` both appends the
    static goal guidance to the system prompt and re-pins the notice into the
    (post-summarization) request when the persisted one is out of view.
    """

    state_schema = GoalToolState

    def __init__(self) -> None:
        """Initialize goal tools."""
        super().__init__()

        @tool
        def get_rubric(
            state: Annotated[dict[str, Any], InjectedState],
        ) -> RubricSnapshot:
            """Read criteria when the latest state notice says a rubric is active.

            Use this only when the latest goal/rubric state notice reports an active
            rubric. It returns whether the criteria came from a goal, a sticky
            rubric, or the current invocation, plus the latest grading status.

            Returns:
                Rubric snapshot with `active`, `criteria`, `source`, and
                `grading_status` keys.
            """
            return _rubric_snapshot(state)

        @tool
        def get_goal(
            state: Annotated[dict[str, Any], InjectedState],
        ) -> GoalSnapshot:
            """Read a goal when the latest state notice says it is actionable.

            Use this only when the latest goal/rubric state notice reports an
            actionable goal. It returns the objective, criteria, lifecycle status,
            and any prior note from authoritative checkpoint state.

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
            """Update a goal only when the latest state notice says it is actionable.

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
    def _notice_update(state: AgentState[Any]) -> dict[str, Any] | None:
        """Compute the checkpointed notice update for a `before_model` boundary.

        Returns:
            A `messages` update carrying a fresh notice, or `None` when history
            already reflects current goal/rubric state.
        """
        values = cast("dict[str, Any]", state)
        raw_messages = values.get("messages", [])
        messages = list(raw_messages) if isinstance(raw_messages, list) else []
        notice = _goal_state_notice_for(values, messages)
        return {"messages": [notice]} if notice is not None else None

    @override
    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Persist a current goal-state notice into checkpointed history.

        This is the durable half of the notice mechanism; the transient
        counterpart in `wrap_model_call` re-pins the notice into a request whose
        persisted notice has scrolled out of the model-visible window.

        Returns:
            Message update containing a current notice, or `None` when unchanged.
        """
        del runtime
        return self._notice_update(state)

    @override
    async def abefore_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Persist a current goal-state notice at an async model boundary.

        Async twin of `before_model`; see it for the persisted-vs-transient split.

        Returns:
            Message update containing a current notice, or `None` when unchanged.
        """
        del runtime
        return self._notice_update(state)

    @staticmethod
    def _request_with_goal_system_prompt(
        request: ModelRequest[ContextT],
    ) -> ModelRequest[ContextT]:
        """Append static goal guidance and re-pin the notice into a request.

        The system prompt gains the static goal-tool guidance, and — when
        checkpointed history no longer surfaces a current notice — a transient
        goal-state notice is appended to the request messages only (not
        persisted; `before_model` owns the durable write).

        Returns:
            Model request with goal guidance in the system prompt and, when
            needed, a current goal-state notice appended to its messages.
        """
        if request.system_message is not None:
            content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{GOAL_TOOLS_SYSTEM_PROMPT}"},
            ]
        else:
            content = [{"type": "text", "text": GOAL_TOOLS_SYSTEM_PROMPT}]
        values = cast("dict[str, Any]", request.state)
        notice = _goal_state_notice_for(values, request.messages)
        messages = (
            [*request.messages, notice] if notice is not None else request.messages
        )
        return request.override(
            messages=messages,
            system_message=SystemMessage(
                content=cast("list[str | dict[str, str]]", content)
            ),
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
        return handler(self._request_with_goal_system_prompt(request))

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
        return await handler(self._request_with_goal_system_prompt(request))
