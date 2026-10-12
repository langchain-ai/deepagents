"""Fork goal visibility without parent grading or cache-prefix churn."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast
from unittest.mock import MagicMock

import pytest
from deepagents import create_deep_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatResult
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from deepagents_code.goal_state_notice import goal_state_notice_info
from deepagents_code.goal_tools import ForkGoalToolsMiddleware, GoalToolsMiddleware
from deepagents_code.reliable_rubric import ReliableRubricMiddleware

if TYPE_CHECKING:
    from deepagents.middleware.subagents import SubAgent
    from langchain.agents.middleware.types import AgentMiddleware, AgentState
    from langchain_core.runnables import RunnableConfig


def _grader_call(
    *, result: str, explanation: str, criteria: list[dict[str, object]]
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "GraderResponse",
                "args": {
                    "result": result,
                    "explanation": explanation,
                    "criteria": criteria,
                },
                "id": "grade",
            }
        ],
    )


class RecordingModel(GenericFakeChatModel):
    """Record effective model inputs while retaining real graph execution."""

    requests: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        **kwargs: Any,
    ) -> "RecordingModel":
        del tools, kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@tool
def inspect_piece() -> str:
    """Inspect one delegated piece."""
    return "piece checked"


@pytest.mark.filterwarnings(
    "default:The feature `forked subagents` is in beta:"
    "langchain_core._api.LangChainBetaWarning"
)
@pytest.mark.parametrize("mode", ["fork", "isolated"])
@pytest.mark.parametrize("saved_goal", [False, True])
def test_declarative_child_preserves_goal_visibility_and_parent_grading(
    mode: Literal["fork", "isolated"],
    saved_goal: bool,
) -> None:
    child = RecordingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "inspect_piece", "args": {}, "id": "inspect"}],
                ),
                AIMessage(content="delegated piece done"),
            ]
        )
    )
    parent = RecordingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "inspect piece",
                                "subagent_type": "worker",
                            },
                            "id": "delegate",
                        }
                    ],
                ),
                AIMessage(content="first draft"),
                AIMessage(content="revised whole goal"),
            ]
        )
    )
    grader_model = RecordingModel(
        messages=iter(
            [
                _grader_call(
                    result="needs_revision",
                    explanation="finish whole goal",
                    criteria=[
                        {
                            "name": "whole goal marker",
                            "passed": False,
                            "gap": "finish whole goal",
                        }
                    ],
                ),
                _grader_call(
                    result="satisfied",
                    explanation="whole goal done",
                    criteria=[{"name": "whole goal marker", "passed": True}],
                ),
            ]
        )
    )
    rubric = ReliableRubricMiddleware(model=grader_model, max_iterations=3)
    graph = create_deep_agent(
        model=parent,
        checkpointer=InMemorySaver(),
        system_prompt="stable parent instructions",
        tools=[inspect_piece],
        middleware=cast("list[AgentMiddleware]", [GoalToolsMiddleware(), rubric]),
        subagents=[
            cast(
                "SubAgent",
                {
                    "name": "worker",
                    "description": "inspect one piece",
                    "mode": mode,
                    "model": child,
                    "system_prompt": "worker instructions",
                    "middleware": [ForkGoalToolsMiddleware()] if mode == "fork" else [],
                },
            )
        ],
    )
    config: RunnableConfig = {"configurable": {"thread_id": "fork-goal-test"}}
    if saved_goal:
        graph.update_state(
            config,
            {
                "_goal_objective": "whole objective",
                "_goal_status": "active",
                "_goal_rubric": "whole goal marker",
            },
            as_node="GoalToolsMiddleware.before_model",
        )
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="finish the work")],
            "rubric": "whole goal marker",
        },
        config=config,
    )
    assert result["messages"][-1].content == "revised whole goal"
    assert len(grader_model.requests) == 2
    assert "first draft" in str(grader_model.requests[0])
    assert "revised whole goal" in str(grader_model.requests[1])
    assert len(parent.requests) == 3
    assert len(child.requests) == 2
    notices = [
        [message for message in request if goal_state_notice_info(message)]
        for request in child.requests
    ]
    if mode == "fork":
        assert [len(items) for items in notices] == [1, 1]
        assert "whole goal marker" in str(notices[0][0].content)
        if saved_goal:
            assert "whole objective" in str(notices[0][0].content)
        assert notices[0][0] == notices[1][0]
        assert child.requests[0][0] == child.requests[1][0]
        assert str(child.requests[0][0].content).startswith(
            str(parent.requests[0][0].content)
        )
        inherited = parent.requests[0][1:]
        assert child.requests[0][1 : 1 + len(inherited)] == inherited
        assert child.requests[1][: len(child.requests[0])] == child.requests[0]
    else:
        assert notices == [[], []]
        assert all(
            "whole goal marker" not in str(message.content)
            for request in child.requests
            for message in request
        )


def test_notice_snapshot_clears_when_invocation_rubric_is_removed() -> None:
    middleware = GoalToolsMiddleware()
    state = {"messages": [], "rubric": "criteria"}
    update = middleware.before_model(cast("AgentState", state), MagicMock())
    assert update is not None
    assert update["_fork_notice_rubric"] == "criteria"
    state.update(update)
    state["rubric"] = None
    cleared = middleware.before_model(cast("AgentState", state), MagicMock())
    assert cleared is not None
    assert cleared["_fork_notice_rubric"] is None
