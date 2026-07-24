"""Goal tools exposed to the agent for persisted TUI goals."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    NotRequired,
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
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import Field
from typing_extensions import override

# Runtime (not TYPE_CHECKING) imports. `GoalRubricChannels` supplies the shared
# `PrivateStateAttr`-marked goal/rubric channels that `GoalToolState` extends, so
# the markers are declared once (see that class). `coerce_goal_status` is used at
# runtime by `_update_goal_command` to reject writes against a paused/complete goal.
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
    coerce_goal_status,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langgraph.runtime import Runtime

GOAL_TOOL_NAMES = frozenset({"update_goal"})
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
    """Expose the constrained `update_goal` tool and maintain the goal-state notice.

    The model reads goal awareness from the injected goal-state notice rather
    than a read tool: `before_model` persists a fresh notice into checkpointed
    history when the latest one no longer matches authoritative state, and
    `wrap_model_call` re-pins the notice into the (post-summarization) request
    when the persisted one is out of view. The notice carries the objective and
    acceptance criteria when actionable, so no `get_goal`/`get_rubric` lookup is
    needed. Only the write-side `update_goal` tool is registered.
    """

    state_schema = GoalToolState

    def __init__(self) -> None:
        """Initialize goal tools."""
        super().__init__()

        @tool
        def update_goal(
            status: Annotated[
                Literal["complete", "blocked"],
                Field(
                    description=(
                        "`complete` to attach completion evidence, or `blocked` "
                        "when you are stuck and need the user."
                    )
                ),
            ],
            note: Annotated[
                str,
                Field(
                    description=(
                        "Evidence the criteria are satisfied, or the specific "
                        "blocker. Required when calling this tool."
                    )
                ),
            ],
            tool_call_id: Annotated[str, InjectedToolCallId],
            state: Annotated[dict[str, Any], InjectedState],
        ) -> Command[Any]:
            """Update a goal only when the latest state notice says it is actionable.

            Read the current objective and acceptance criteria from the latest
            goal/rubric state notice in context; there is no read tool for them.
            Use `blocked` when you cannot proceed without user input. Goals complete
            automatically after a satisfied goal-backed grading turn, so `complete`
            is optional and only stages its evidence for that result. Do not create,
            pause, resume, clear, or replace goals — those are user-controlled.

            Returns:
                Command that updates goal status and returns a tool message.
            """
            return _update_goal_command(
                status=status,
                note=note,
                tool_call_id=tool_call_id,
                state=state,
            )

        self.tools = [update_goal]

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
    def _request_with_goal_notice(
        request: ModelRequest[ContextT],
    ) -> ModelRequest[ContextT]:
        """Re-pin the current goal-state notice into a model request when needed.

        When checkpointed history no longer surfaces a current notice, a
        transient goal-state notice is appended to the request messages only
        (not persisted; `before_model` owns the durable write). The system
        prompt is left unchanged.

        Returns:
            The original request when no notice is needed, otherwise a request
            with a current goal-state notice appended to its messages.
        """
        values = cast("dict[str, Any]", request.state)
        notice = _goal_state_notice_for(values, request.messages)
        if notice is None:
            return request
        return request.override(messages=[*request.messages, notice])

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Re-pin the goal-state notice into each model request when needed.

        Returns:
            Model response from the wrapped handler.
        """
        return handler(self._request_with_goal_notice(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        """Re-pin the goal-state notice into each async model request when needed.

        Returns:
            Model response from the wrapped handler.
        """
        return await handler(self._request_with_goal_notice(request))
