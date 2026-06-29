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
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

GOAL_TOOLS_SYSTEM_PROMPT = (
    "The user may set a persistent goal with `/goal`. When a goal is active, "
    "use `get_goal` to inspect the objective, accepted criteria, and status. "
    "Use `update_goal` only when you have evidence that the goal is complete "
    "or blocked. Do not pause, resume, clear, replace, or create goals yourself; "
    "those are user-controlled lifecycle actions."
)

ResponseT = TypeVar("ResponseT")


class GoalToolState(AgentState):
    """State fields used by goal tools."""

    _goal_objective: NotRequired[str | None]
    _goal_status: NotRequired[str | None]
    _goal_rubric: NotRequired[str | None]
    _goal_status_note: NotRequired[str | None]
    rubric: NotRequired[str | None]


def _goal_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Build the `get_goal` response from graph state.

    Args:
        state: Current graph state injected by LangGraph.

    Returns:
        Goal snapshot visible to the model.
    """
    objective = state.get("_goal_objective")
    rubric = state.get("_goal_rubric") or state.get("rubric")
    if not isinstance(objective, str) or not objective:
        return {
            "active": False,
            "objective": None,
            "status": None,
            "criteria": rubric if isinstance(rubric, str) else None,
            "note": None,
        }
    status = state.get("_goal_status")
    note = state.get("_goal_status_note")
    return {
        "active": status not in {"complete", None},
        "objective": objective,
        "status": status if isinstance(status, str) else "active",
        "criteria": rubric if isinstance(rubric, str) else None,
        "note": note if isinstance(note, str) else None,
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
        status: Terminal goal status requested by the model.
        note: Evidence or blocker explanation.
        tool_call_id: Tool call ID for the returned `ToolMessage`.
        state: Current graph state injected by LangGraph.

    Returns:
        Command updating goal metadata and returning a tool response.
    """
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
    content = f"Goal marked {status}."
    if clean_note:
        content += f" {clean_note}"
    return Command(
        update={
            "_goal_status": status,
            "_goal_status_note": clean_note or None,
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )


class GoalToolsMiddleware(AgentMiddleware[GoalToolState, ContextT]):
    """Expose constrained goal tools to the main agent."""

    state_schema = GoalToolState

    def __init__(self) -> None:
        """Initialize goal tools."""
        super().__init__()

        @tool
        def get_goal(
            state: Annotated[dict[str, Any], InjectedState],
        ) -> dict[str, Any]:
            """Read the current persistent goal and accepted criteria.

            Returns:
                Current goal snapshot.
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

            Returns:
                Command that updates goal status and returns a tool message.
            """
            return _update_goal_command(
                status=status,
                note=note,
                tool_call_id=tool_call_id,
                state=state,
            )

        self.tools = [get_goal, update_goal]

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
        if request.system_message is not None:
            content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{GOAL_TOOLS_SYSTEM_PROMPT}"},
            ]
        else:
            content = [{"type": "text", "text": GOAL_TOOLS_SYSTEM_PROMPT}]
        return handler(
            request.override(
                system_message=SystemMessage(
                    content=cast("list[str | dict[Any, Any]]", content)
                )
            )
        )

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
        if request.system_message is not None:
            content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{GOAL_TOOLS_SYSTEM_PROMPT}"},
            ]
        else:
            content = [{"type": "text", "text": GOAL_TOOLS_SYSTEM_PROMPT}]
        return await handler(
            request.override(
                system_message=SystemMessage(
                    content=cast("list[str | dict[Any, Any]]", content)
                )
            )
        )
