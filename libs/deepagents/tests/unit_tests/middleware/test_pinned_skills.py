"""Deterministic skill loading through invocation state."""

from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Overwrite

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.graph import create_deep_agent
from deepagents.middleware.skills import SkillsMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "skills" / "example"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\nname: example\ndescription: An example skill\nmetadata:\n  include_tools: greet\n---\nSECRET SKILL INSTRUCTIONS\n"
    )
    return directory.parent


@tool
def greet() -> str:
    """Greet the user."""
    return "hello"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_pins_append_clear_and_disclose_tools(skills_dir: Path, *, asynchronous: bool) -> None:
    directory = skills_dir / "another"
    directory.mkdir()
    (directory / "SKILL.md").write_text("---\nname: another\ndescription: Another skill\n---\nANOTHER SKILL INSTRUCTIONS\n")
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")] * 5))
    middleware = SkillsMiddleware(backend=FilesystemBackend(virtual_mode=False), sources=[str(skills_dir)], tools=[greet], system_prompt=None)
    agent = create_agent(model, middleware=[middleware], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "pins"}}
    inputs = [
        {"messages": [HumanMessage(content="first")], "pinned_skills": ["example", "example"]},
        {"messages": [HumanMessage(content="second")], "pinned_skills": ["another"]},
        {"messages": [HumanMessage(content="third")], "pinned_skills": []},
        {"messages": [HumanMessage(content="fourth")]},
        {"messages": [HumanMessage(content="fifth")], "pinned_skills": Overwrite([])},
    ]
    for index, payload in enumerate(inputs):
        result = await agent.ainvoke(payload, config) if asynchronous else agent.invoke(payload, config)
        sent = model.call_history[-1]["messages"]
        pinned = [message for message in sent if "SECRET SKILL INSTRUCTIONS" in str(message.content)]
        assert len(pinned) == (1 if index < 4 else 0)
        assert all(isinstance(message, HumanMessage) for message in pinned)
        another = [message for message in sent if "ANOTHER SKILL INSTRUCTIONS" in str(message.content)]
        assert len(another) == (1 if 0 < index < 4 else 0)
        if index < 4:
            assert [entry.name for entry in model.call_history[-1]["tools"]] == ["greet"]
        assert not any("SKILL INSTRUCTIONS" in str(message.content) for message in result["messages"])
        expected_pins = ["example", "example"] + (["another"] if index > 0 else []) if index < 4 else []
        assert result["pinned_skills"] == expected_pins
        snapshot = await agent.aget_state(config) if asynchronous else agent.get_state(config)
        assert snapshot.values["_skill_tools_disclosed"] == ({"greet": "greet"} if index < 4 else {})


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_pinned_tool_can_execute(skills_dir: Path, *, asynchronous: bool) -> None:
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="", tool_calls=[{"name": "greet", "args": {}, "id": "greeting"}]), AIMessage(content="done")])
    )
    agent = create_agent(
        model,
        middleware=[SkillsMiddleware(backend=FilesystemBackend(virtual_mode=False), sources=[str(skills_dir)], tools=[greet])],
    )
    payload = {"messages": [HumanMessage(content="hello")], "pinned_skills": ["example"]}
    result = await agent.ainvoke(payload) if asynchronous else agent.invoke(payload)
    assert any(message.type == "tool" and message.content == "hello" for message in result["messages"])
    assert len(model.call_history) == 2
    assert all(any("SECRET SKILL INSTRUCTIONS" in str(message.content) for message in call["messages"]) for call in model.call_history)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_deep_agent_pinned_skill(skills_dir: Path, *, asynchronous: bool) -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    agent = create_deep_agent(model=model, backend=FilesystemBackend(virtual_mode=False), skills=[str(skills_dir)])
    payload = {"messages": [HumanMessage(content="use example")], "pinned_skills": ["example"]}
    result = await agent.ainvoke(payload) if asynchronous else agent.invoke(payload)
    assert result["pinned_skills"] == ["example"]
    assert any("SECRET SKILL INSTRUCTIONS" in str(message.content) for message in model.call_history[0]["messages"])


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("pins", [["missing"], "example", [1]])
async def test_invalid_pins_fail_before_model(skills_dir: Path, *, asynchronous: bool, pins: object) -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    agent = create_agent(model, middleware=[SkillsMiddleware(backend=FilesystemBackend(virtual_mode=False), sources=[str(skills_dir)])])
    payload = {"messages": [HumanMessage(content="hello")], "pinned_skills": pins}
    if asynchronous:
        with pytest.raises((TypeError, ValueError), match=r"pinned skill|pinned_skills|concatenate list"):
            await agent.ainvoke(payload)
    else:
        with pytest.raises((TypeError, ValueError), match=r"pinned skill|pinned_skills|concatenate list"):
            agent.invoke(payload)
    assert not model.call_history


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_unreadable_pin_fails_before_model(skills_dir: Path, *, asynchronous: bool) -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")] * 2))
    agent = create_agent(
        model,
        middleware=[SkillsMiddleware(backend=FilesystemBackend(virtual_mode=False), sources=[str(skills_dir)])],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "deleted"}}
    payload = {"messages": [HumanMessage(content="hello")], "pinned_skills": ["example"]}
    if asynchronous:
        await agent.ainvoke(payload, config)
    else:
        agent.invoke(payload, config)
    (skills_dir / "example" / "SKILL.md").unlink()
    if asynchronous:
        with pytest.raises(ValueError, match="Could not load pinned skill"):
            await agent.ainvoke({"messages": [HumanMessage(content="again")]}, config)
    else:
        with pytest.raises(ValueError, match="Could not load pinned skill"):
            agent.invoke({"messages": [HumanMessage(content="again")]}, config)
    assert len(model.call_history) == 1
