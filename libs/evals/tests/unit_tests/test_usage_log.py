"""Check per-call usage lines written for Harbor runs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from deepagents_evals.routing.orchestrator import HANDOFF, OrchestratorMiddleware
from deepagents_harbor.langgraph_project import langgraph_agent
from tests.unit_tests.test_routing_orchestrator import ScriptedModel, _call

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _untraced(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
        monkeypatch.setenv(name, "false")


def _used(message: AIMessage, tokens: int) -> AIMessage:
    usage = {"input_tokens": tokens, "output_tokens": 1, "total_tokens": tokens + 1}
    return message.model_copy(update={"usage_metadata": usage})


def test_each_call_logs_usage_and_orchestrator_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "usage.jsonl"
    monkeypatch.setattr(langgraph_agent, "USAGE_LOG", log)
    calls: list[tuple[str, list[str]]] = []
    planner_turns: list[BaseMessage] = [_used(_call(HANDOFF, plan="do it"), 100)]
    executor_turns: list[BaseMessage] = [_used(AIMessage(content="Done"), 7)]
    planner = ScriptedModel(model_name="planner", responses=planner_turns, log=calls)
    executor = ScriptedModel(model_name="executor", responses=executor_turns, log=calls)
    middleware = OrchestratorMiddleware(
        langgraph_agent._with_usage_log(planner), langgraph_agent._with_usage_log(executor)
    )
    agent = create_agent(model=middleware.executor, middleware=[middleware])

    agent.invoke({"messages": [HumanMessage(content="Fix it")]})

    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(line["orchestrator_phase"], line["usage"]["input_tokens"]) for line in lines] == [
        ("plan", 100),
        ("execute", 7),
    ]
