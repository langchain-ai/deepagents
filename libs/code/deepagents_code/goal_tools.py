"""Goal tools exposed to the agent for persisted TUI goals."""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    NotRequired,
    cast,
    override,
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

from deepagents_code.goal_state_limits import (
    GOAL_STATUS_NOTE_CHAR_LIMIT,
    GoalStateSizeError,
    validate_goal_status_note,
)
from deepagents_code.goal_state_notice import (
    build_goal_state_notice,
    goal_notice_size_error,
    goal_state_fingerprint,
    has_goal_or_rubric_state,
    is_goal_state_message,
    latest_goal_state_message_index,
    latest_goal_state_notice,
    latest_human_is_unsaved_goal_continuation,
    log_malformed_summarization_event,
    superseded_goal_state_placeholder,
    validated_summarization_cutoff,
)

# Runtime (not TYPE_CHECKING) import. `GoalRubricChannels` looks type-only but is
# a base class of `GoalToolState`, supplying the shared `PrivateStateAttr`-marked
# goal/rubric channels so the markers are declared once (see that class).
from deepagents_code.resume_state import (
    GoalRubricChannels,
    coerce_goal_status,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


GOAL_TOOL_NAMES = frozenset({"update_goal"})
"""Tool names used by behavioral absence gates and middleware contract tests."""


def _goal_state_notice_for(
    state: dict[str, Any],
    messages: Sequence[object],
    *,
    cutoff: int,
) -> HumanMessage | None:
    """Build a notice when effective history lacks current goal/rubric state.

    Args:
        state: Authoritative middleware state.
        messages: Messages visible at the next model boundary.
        cutoff: Summarization cutoff index that `messages` is measured against.
            Required rather than defaulted: both callers pass the full persisted
            list, where a notice below the cutoff is present but invisible to the
            model, and this middleware wraps the summarizer so a trimmed window
            never reaches here. A default of `0` would silently treat such a
            notice as visible.

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
        and latest[0] >= cutoff
    ):
        return None
    if latest_candidate is None and not has_goal_or_rubric_state(state):
        return None
    return build_goal_state_notice(state)


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

            Nothing is committed in five cases, and the `ToolMessage` explains
            what the model must do instead: no goal is set, saved state is too
            large to render as a notice, the goal is paused or already complete,
            `note` is empty, or `note` exceeds `GOAL_STATUS_NOTE_CHAR_LIMIT`.
            The `note` size is also gated by the tool schema's `max_length`, so
            the runtime check catches only calls that bypass it — and measures
            the stripped note.
    """
    # Enforced preconditions here are: an objective exists, its state fits the
    # notice budget, its status is neither paused nor complete, and `note` is
    # non-empty and fits `GOAL_STATUS_NOTE_CHAR_LIMIT`. Note the objective check alone
    # does not imply actionability — a paused goal has an objective too, so the
    # status check is separate. Completion is staged because `RubricMiddleware`
    # records its final verdict after the model stops making tool calls; the TUI
    # resolves the staged request during post-turn checkpoint sync.
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
    # Oversized state takes precedence over every other precondition: the notice
    # has already told the model the goal is unavailable and not to work toward
    # it or grade against it. Since the read tools were removed, `update_goal` is
    # the only goal surface the model has, so refusing here is what keeps that
    # instruction from resting on prose alone. Project exactly as the renderer
    # does, or the check passes against text the notice does not carry, which is
    # why this goes through the shared helper rather than projecting again here.
    #
    # The refusal covers an oversized `_goal_status_note` too, which the model
    # itself wrote on an earlier turn. It cannot replace that note with a shorter
    # one, because this is the call that would do it. Recovery is deliberately
    # user-only (`/goal clear`): the alternative is letting the model rewrite
    # state the notice has already told it is unavailable.
    exc = goal_notice_size_error(state)
    if exc is not None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"Saved goal/rubric state is too large to use, so its "
                            f"status cannot be updated. Ask the user to clear and "
                            f"recreate the goal. Validation detail: {exc}"
                        ),
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
    try:
        validate_goal_status_note(clean_note)
    except GoalStateSizeError as exc:
        return Command(
            update={
                "messages": [ToolMessage(content=str(exc), tool_call_id=tool_call_id)]
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
    history when the latest one no longer matches authoritative state (or has
    scrolled below the summarization cutoff), and `wrap_model_call` re-pins the
    notice into the request when the persisted one is out of view. This
    middleware wraps the summarizer rather than running after it, so the re-pin
    sees untrimmed history and discounts it against the same cutoff
    `before_model` uses. The notice carries the objective and status note for an
    actionable goal and the acceptance criteria for an active rubric, so no
    separate read tool is needed. Only the write-side `update_goal` tool is
    registered.
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
                    max_length=GOAL_STATUS_NOTE_CHAR_LIMIT,
                    description=(
                        "Evidence the criteria are satisfied, or the specific "
                        "blocker. Required when calling this tool."
                    ),
                ),
            ],
            tool_call_id: Annotated[str, InjectedToolCallId],
            state: Annotated[dict[str, Any], InjectedState],
        ) -> Command[Any]:
            """Update a goal only when the latest state notice says it is actionable.

            Read the current objective and any acceptance criteria from the latest
            goal/rubric state notice in context, or — right after a goal whose save
            failed — from the objective and criteria quoted in the accompanying
            goal continuation message. There is no read tool for them. Use
            `blocked` when you cannot proceed without user input. Goals complete
            automatically after a satisfied goal-backed grading turn, so
            `complete` is optional and only stages its evidence for that result.
            Do not create, pause, resume, clear, or replace goals — those are
            user-controlled.

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
            A state update with any of three independent parts, or `None` when
            none apply: a `messages` entry carrying a fresh notice; a
            `_summarization_event` reset discarding a malformed event; and a
            `rubric` entry set to `None` when saved goal/rubric state exceeds
            the notice budget, which clears the public per-invocation rubric so
            grading cannot re-inject oversized text. The oversized case can
            return a `rubric` clear with no `messages` key at all.
        """
        values = cast("dict[str, Any]", state)
        raw_messages = values.get("messages", [])
        messages = list(raw_messages) if isinstance(raw_messages, list) else []
        # `state["messages"]` is the full persisted list, so the cutoff rule
        # applies — see `validated_summarization_cutoff`. Discounting matters here
        # because it is what makes the durable write happen, instead of leaving
        # the transient re-pin in `wrap_model_call` to carry the objective on every
        # subsequent turn. For a valid event this matches the client-side
        # predicate in `app.py`. A malformed event diverges deliberately: `app.py`
        # collapses it to `0`, while here it forces a fresh notice and clears the
        # event.
        event = values.get("_summarization_event")
        cutoff = validated_summarization_cutoff(
            event,
            message_count=len(messages),
        )
        malformed_event = event is not None and cutoff is None
        notice = _goal_state_notice_for(
            values,
            messages,
            # Force a fresh notice when discarding an event so a summarization
            # regenerated on this boundary retains the canonical goal state.
            cutoff=len(messages) if malformed_event else (cutoff or 0),
        )
        update: dict[str, Any] = {}
        if malformed_event:
            log_malformed_summarization_event(event, len(messages))
            update["_summarization_event"] = None
        if notice is not None:
            update["messages"] = [notice]
        exc = goal_notice_size_error(values)
        # Keep authoritative saved state recoverable, but clear the public
        # per-invocation rubric so grading cannot re-inject oversized text.
        if exc is not None and values.get("rubric") is not None:
            logger.warning(
                "Goal/rubric state exceeds the notice budget; clearing the "
                "per-invocation rubric so this turn is not graded: %s",
                exc,
            )
            update["rubric"] = None
        return update or None

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
        """Ensure a current goal-state notice is visible in a model request.

        When checkpointed history no longer surfaces a current notice, a
        transient goal-state notice is appended to the request messages only
        (not persisted; `before_model` owns the durable write). Every superseded
        notice is replaced in place by a fixed-size stand-in, so a legacy
        oversized value cannot remain model-visible beside its bounded
        replacement. Replacement keeps the list length and every index stable,
        which the inner summarizer's persisted absolute cutoff depends on — see
        `superseded_goal_state_placeholder`. The system prompt is left unchanged.

        This middleware wraps the summarizer, so `request.messages` is the full
        persisted list rather than a trimmed window. The summarization cutoff is
        therefore passed through to `_goal_state_notice_for`, matching
        `before_model`: without it a notice sitting below the cutoff counts as
        visible here, the re-pin declines, and the inner summarizer then drops
        the only copy the model had.

        Returns:
            The request unchanged apart from any malformed-event state override,
            when no notice is needed and no superseded notice was replaced.
            Otherwise a request carrying any of: a current goal-state notice
            appended to its messages, superseded notices replaced in place within
            them, and — for a malformed `_summarization_event` — a state override
            nulling that event.
        """
        values = cast("dict[str, Any]", request.state)
        event = values.get("_summarization_event")
        cutoff = validated_summarization_cutoff(
            event,
            message_count=len(request.messages),
        )
        malformed_event = event is not None and cutoff is None
        if malformed_event:
            # Disable an invalid restored event in the request passed inward so
            # its Python slice cannot remove the only model-visible copy of the
            # goal state.
            log_malformed_summarization_event(event, len(request.messages))
            values = {**values, "_summarization_event": None}
            request = request.override(state=cast("AgentState[Any]", values))
        notice = _goal_state_notice_for(
            values,
            request.messages,
            # Force a fresh notice when discarding an event, matching
            # `_notice_update`, so a summarization regenerated on this boundary
            # retains the canonical goal state.
            cutoff=len(request.messages) if malformed_event else (cutoff or 0),
        )
        messages = list(request.messages)
        if notice is not None:
            messages.append(notice)
        latest_index = latest_goal_state_message_index(messages)
        # Replace superseded notices in place rather than removing them, so a
        # legacy oversized value stops being model-visible beside its bounded
        # replacement without moving any index. The inner summarizer picks its
        # cutoff from this list and persists it as an *absolute* index into
        # `state["messages"]`, which is never filtered, so a removal would make
        # the persisted cutoff slice the checkpointed list too early — silently
        # dropping live turns, and orphaning a `ToolMessage` whose `AIMessage`
        # was summarized away (which the provider rejects). See
        # `superseded_goal_state_placeholder`.
        #
        # Every superseded notice is replaced, including ones below the cutoff:
        # the stand-in is the same length-preserving shape either way, so there is
        # no alignment reason to treat them differently, and doing so keeps the
        # request free of stale goal text no matter where the cutoff lands.
        #
        # `is_goal_state_message` also matches on the `SYSTEM_MESSAGE_PREFIX`
        # text, not just the metadata source, so this can in principle replace a
        # human turn that opens with that exact sentence. The prefix arm is kept
        # because legacy notices predate the metadata and are the reason the
        # filter exists; the residual risk is a user pasting that sentence
        # verbatim as the start of a message, which only affects the transient
        # request window and never the checkpoint.
        replaced = False
        projected: list[Any] = []
        for index, message in enumerate(messages):
            if index != latest_index and is_goal_state_message(message):
                projected.append(superseded_goal_state_placeholder(message))
                replaced = True
            else:
                projected.append(message)
        if notice is None and not replaced:
            return request
        return request.override(messages=projected)

    @override
    def wrap_model_call[ResponseT](
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
    async def awrap_model_call[ResponseT](
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
