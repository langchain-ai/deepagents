"""Exercise handoffs between the orchestrator's planner and executor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from deepagents_evals.routing.orchestrator import ESCALATE, HANDOFF, OrchestratorMiddleware

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.tools import BaseTool


class ScriptedModel(FakeMessagesListChatModel):
    model_name: str
    log: Any = (
        None  # `Any` keeps one list shared by both models; a typed list is copied on validation.
    )
    bound: list[str] = Field(default_factory=list)

    def bind_tools(self, tools: object, **kwargs: object) -> Self:
        return self.model_copy(update={"bound": [t.name for t in cast("list[BaseTool]", tools)]})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        self.log.append((self.model_name, self.bound))
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


@pytest.fixture(autouse=True)
def _untraced(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
        monkeypatch.setenv(name, "false")


@tool
def write_file(path: str) -> str:
    """Write a file."""
    return f"wrote {path}"


@tool
def run(command: str) -> str:
    """Run a shell command."""
    return f"{command}\n[Command failed with exit code 1]"


def _call(name: str, **args: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": f"{name}-{id(args)}"}]
    )


def _agent(
    planner: list[BaseMessage], executor: list[BaseMessage], **kwargs: int
) -> tuple[Any, list]:
    log: list[tuple[str, list[str]]] = []
    middleware = OrchestratorMiddleware(
        ScriptedModel(model_name="planner", responses=planner, log=log),
        ScriptedModel(model_name="executor", responses=executor, log=log),
        **kwargs,
    )
    return create_agent(
        model=middleware.executor, tools=[write_file, run], middleware=[middleware]
    ), log


def _transitions(result: dict[str, Any]) -> list[tuple[str, str]]:
    return [(r["to_phase"], r["trigger"]) for r in result["decision_log"]]


def test_planner_and_executor_trade_control_with_phase_tools() -> None:
    agent, log = _agent(
        planner=[_call(HANDOFF, plan="1. write"), _call(HANDOFF, plan="2. write elsewhere")],
        executor=[_call(ESCALATE, reason="path is read-only"), AIMessage(content="Done")],
    )
    result = agent.invoke({"messages": [HumanMessage(content="Fix it")]})

    assert [name for name, _ in log] == ["planner", "executor", "planner", "executor"]
    planner_tools, executor_tools = set(log[0][1]), set(log[1][1])
    assert HANDOFF in planner_tools
    assert not planner_tools & {"write_file", ESCALATE}
    assert {"write_file", ESCALATE} <= executor_tools
    assert HANDOFF not in executor_tools
    assert _transitions(result) == [("execute", "tool"), ("plan", "tool"), ("execute", "tool")]


def test_failure_streak_forces_the_planner_back() -> None:
    agent, log = _agent(
        planner=[_call(HANDOFF, plan="run it"), AIMessage(content="Gave up")],
        executor=[_call("run", command="make")] * 2,
        error_streak=2,
    )
    result = agent.invoke({"messages": [HumanMessage(content="Build it")]})

    assert [name for name, _ in log] == ["planner", "executor", "executor", "planner"]
    assert _transitions(result) == [("execute", "tool"), ("plan", "errors")]


def test_exhausted_escalations_leave_the_executor_in_control() -> None:
    agent, log = _agent(
        planner=[_call(HANDOFF, plan="run it")],
        executor=[*[_call("run", command="make")] * 3, AIMessage(content="Done")],
        max_escalations=0,
        error_streak=1,
    )
    result = agent.invoke({"messages": [HumanMessage(content="Build it")]})

    assert [name for name, _ in log] == ["planner", *["executor"] * 4]
    assert ESCALATE not in log[1][1]
    assert _transitions(result) == [("execute", "tool")]
