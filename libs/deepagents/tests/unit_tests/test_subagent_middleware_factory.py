"""Child middleware factories preserve native harness construction."""

from collections.abc import Awaitable, Callable, Sequence

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, create_deep_agent, register_harness_profile
from deepagents.middleware.subagents import SubAgent
from deepagents.profiles.harness.harness_profiles import _HARNESS_PROFILES
from tests.unit_tests.chat_model import GenericFakeChatModel


class _ChildTools(AgentMiddleware):
    def __init__(self, names: list[list[str]]) -> None:
        self.names = names

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        self.names.append([entry.name for entry in request.tools if hasattr(entry, "name")])
        return handler(request)

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
        self.names.append([entry.name for entry in request.tools if hasattr(entry, "name")])
        return await handler(request)


@tool
def child_probe() -> str:
    """Return a child-only result."""
    return "child result"


class _InjectedTools(_ChildTools):
    def __init__(self, names: list[list[str]]) -> None:
        super().__init__(names)
        self.tools = [child_probe]


@pytest.mark.parametrize("child", ["general-purpose", "researcher"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_child_factory_adds_executable_tools_without_parent_lifecycle(child: str, *, asynchronous: bool) -> None:
    names: list[list[str]] = []
    calls: list[str] = []
    child_model = GenericFakeChatModel(
        messages=iter([AIMessage(content="", tool_calls=[{"id": "probe", "name": "child_probe", "args": {}}]), AIMessage(content="child done")])
    )
    lead = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[{"id": "task", "name": "task", "args": {"description": "probe", "subagent_type": child}}]),
                AIMessage(content="done"),
            ]
        )
    )
    if child == "general-purpose":
        lead.messages = iter(
            [
                AIMessage(content="", tool_calls=[{"id": "task", "name": "task", "args": {"description": "probe", "subagent_type": child}}]),
                AIMessage(content="", tool_calls=[{"id": "probe", "name": "child_probe", "args": {}}]),
                AIMessage(content="child done"),
                AIMessage(content="done"),
            ]
        )

    def factory(spec: SubAgent) -> Sequence[AgentMiddleware]:
        calls.append(spec["name"])
        return [_InjectedTools(names)]

    agent = create_deep_agent(
        model=lead,
        subagents=[] if child == "general-purpose" else [{"name": child, "description": "Research.", "model": child_model}],
        subagent_middleware=factory,
    )
    inputs = {"messages": [{"role": "user", "content": "go"}]}
    result = await agent.ainvoke(inputs) if asynchronous else agent.invoke(inputs)
    assert result["messages"][-1].content == "done"
    history = lead.call_history if child == "general-purpose" else child_model.call_history
    assert any(isinstance(message, ToolMessage) and message.content == "child result" for call in history for message in call["messages"])
    assert names and all("child_probe" in tools for tools in names)
    assert child in calls
    assert "child_probe" not in [getattr(entry, "name", None) for entry in lead.tools]


@pytest.mark.parametrize("enabled", [True, False])
def test_factory_keeps_native_general_purpose_profile(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> None:
    monkeypatch.setattr("deepagents.profiles.harness.harness_profiles._HARNESS_PROFILES", dict(_HARNESS_PROFILES))

    class Excluded(AgentMiddleware):
        def before_model(self, state: object, runtime: object) -> None:
            pytest.fail("profile-excluded middleware must not run")

    register_harness_profile(
        "childfactory",
        HarnessProfile(
            system_prompt_suffix="Suffix.",
            excluded_tools=frozenset({"child_probe"}),
            extra_middleware=[Excluded()],
            excluded_middleware=frozenset({"Excluded"}),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=enabled, description="Profile helper.", system_prompt="Profile prompt."),
        ),
    )
    lead = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="", tool_calls=[{"id": "task", "name": "task", "args": {"description": "probe", "subagent_type": "general-purpose"}}]
                ),
                AIMessage(content="child done"),
                AIMessage(content="done"),
            ]
        )
        if enabled
        else iter([AIMessage(content="done")])
    )
    monkeypatch.setattr("deepagents.graph.resolve_model", lambda _: lead)
    names: list[list[str]] = []
    calls: list[str] = []

    def factory(spec: SubAgent) -> Sequence[AgentMiddleware]:
        calls.append(spec["name"])
        return [_InjectedTools(names)]

    agent = create_deep_agent(model="childfactory:test", subagent_middleware=factory)
    result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    assert result["messages"][-1].content == "done"
    assert calls == (["general-purpose"] if enabled else [])
    if enabled:
        assert names
        assert "child_probe" not in [getattr(entry, "name", None) for entry in lead.call_history[1]["tools"]]
        assert lead.call_history[1]["messages"][0].content == "Profile prompt.\n\nSuffix."
        task = next(entry for entry in lead.tools if getattr(entry, "name", None) == "task")
        assert "Profile helper." in task.description
    else:
        assert "task" not in [getattr(entry, "name", None) for entry in lead.tools]


def test_factory_leaves_compiled_and_async_children_opaque() -> None:
    lead = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    calls: list[str] = []

    def factory(spec: SubAgent) -> Sequence[AgentMiddleware]:
        calls.append(spec["name"])
        return []

    compiled = create_deep_agent(model=GenericFakeChatModel(messages=iter([AIMessage(content="opaque")])))
    create_deep_agent(
        model=lead,
        subagents=[
            {"name": "opaque", "description": "Opaque.", "runnable": compiled},
            {"name": "remote", "description": "Remote.", "graph_id": "remote"},
        ],
        subagent_middleware=factory,
    )
    assert calls == ["general-purpose"]
