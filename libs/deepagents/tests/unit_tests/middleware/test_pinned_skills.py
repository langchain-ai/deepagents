"""Skill pinning through `create_deep_agent`, observed at the model's call history and the thread's state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain.tools import ToolRuntime  # noqa: TC002  # `@tool` resolves the injected `runtime` annotation at runtime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deepagents.graph import create_deep_agent
from deepagents.middleware.skills import MAX_SKILL_FILE_SIZE, SkillsMiddleware, SkillsState
from deepagents.middleware.subagents import CompiledSubAgent
from deepagents.middleware.summarization import SummarizationMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel
from tests.unit_tests.middleware.skill_tools_support import (
    SKILLS_SOURCE,
    ai,
    bound_tool_names,
    call,
    invoke,
    skills_agent,
    skills_backend,
    tool_messages,
    write_skill,
)

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.runtime import Runtime

CONFIG: RunnableConfig = {"configurable": {"thread_id": "pinning"}}


def _skill_md(name: str, description: str, body: str) -> str:
    """Return a `SKILL.md` with frontmatter and `body`."""
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"


CRM = _skill_md("crm", "Manage customer requests", "File the request in the CRM.")
CRM_PINNED = '<skill name="crm" path="/skills/crm/SKILL.md">\nFile the request in the CRM.\n</skill>'
HOUSE_STYLE = _skill_md("house-style", "Follow house style", "Use snake_case.")
HOUSE_STYLE_PINNED = '<skill name="house-style" path="/skills/house-style/SKILL.md">\nUse snake_case.\n</skill>'


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> str:
    """Run a test through both `invoke` and `ainvoke`."""
    return request.param


def _model(*turns: AIMessage) -> GenericFakeChatModel:
    """Return a fake model that plays `turns`, then answers "done" for every later call."""
    # Distinct objects: `add_messages` gives each an ID, so a reused one would replace its earlier copy.
    return GenericFakeChatModel(messages=iter([*turns, *(AIMessage(content="done") for _ in range(10))]))


def _agent(root: Path, model: GenericFakeChatModel, **kwargs: Any) -> CompiledStateGraph:
    """Build a deep agent over the skills under `root`, with an in-memory checkpointer."""
    return create_deep_agent(model=model, backend=skills_backend(root), skills=[SKILLS_SOURCE], checkpointer=InMemorySaver(), **kwargs)


def _pinned_messages(messages: list[Any]) -> list[str]:
    """Return the content of every pinned skill message in `messages`."""
    return [m.content for m in messages if m.additional_kwargs.get("lc_source") == "pinned_skill"]


def test_named_skills_follow_the_users_message_in_list_order(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "write-tests", content=_skill_md("write-tests", "Write tests first", "Start with a failing test."))
    write_skill(tmp_path, "house-style", content=HOUSE_STYLE)
    model = _model()

    invoke(
        _agent(tmp_path, model),
        {"messages": [HumanMessage("/write-tests /house-style for auth.py")], "pinned_skills": ["write-tests", "house-style"]},
        mode,
        CONFIG,
    )

    _system, user, *skills = model.call_history[0]["messages"]
    assert (user.type, user.content) == ("human", "/write-tests /house-style for auth.py")
    assert [(m.type, m.content, m.additional_kwargs) for m in skills] == [
        (
            "human",
            '<skill name="write-tests" path="/skills/write-tests/SKILL.md">\nStart with a failing test.\n</skill>',
            {
                "lc_source": "pinned_skill",
                "skill": {"name": "write-tests", "path": "/skills/write-tests/SKILL.md", "description": "Write tests first"},
            },
        ),
        (
            "human",
            HOUSE_STYLE_PINNED,
            {
                "lc_source": "pinned_skill",
                "skill": {"name": "house-style", "path": "/skills/house-style/SKILL.md", "description": "Follow house style"},
            },
        ),
    ]


def test_pin_is_stored_once_and_left_out_of_the_result(tmp_path: Path) -> None:
    write_skill(tmp_path, "crm", content=CRM)
    agent = _agent(tmp_path, _model())

    result = agent.invoke({"messages": [HumanMessage("turn 1")], "pinned_skills": ["crm"]}, CONFIG)
    agent.invoke({"messages": [HumanMessage("turn 2")]}, CONFIG)

    assert "pinned_skills" not in result
    stored = agent.get_state(CONFIG).values["messages"]
    assert [m.content for m in stored] == ["turn 1", CRM_PINNED, "done", "turn 2", "done"]


def test_pin_is_a_snapshot_and_pinning_again_appends_the_current_text(tmp_path: Path) -> None:
    write_skill(tmp_path, "crm", content=_skill_md("crm", "Manage customer requests", "Version one."))
    model = _model()
    agent = _agent(tmp_path, model)

    agent.invoke({"messages": [HumanMessage("turn 1")], "pinned_skills": ["crm"]}, CONFIG)
    write_skill(tmp_path, "crm", content=_skill_md("crm", "Manage customer requests", "Version two."))
    agent.invoke({"messages": [HumanMessage("turn 2")]}, CONFIG)
    turn_2 = _pinned_messages(model.call_history[-1]["messages"])
    agent.invoke({"messages": [HumanMessage("turn 3")], "pinned_skills": ["crm"]}, CONFIG)

    v1 = '<skill name="crm" path="/skills/crm/SKILL.md">\nVersion one.\n</skill>'
    v2 = '<skill name="crm" path="/skills/crm/SKILL.md">\nVersion two.\n</skill>'
    assert turn_2 == [v1]
    assert _pinned_messages(model.call_history[-1]["messages"]) == [v1, v2]


def test_repeated_names_pin_once(tmp_path: Path) -> None:
    write_skill(tmp_path, "crm", content=CRM)
    write_skill(tmp_path, "house-style", content=HOUSE_STYLE)
    model = _model()

    _agent(tmp_path, model).invoke({"messages": [HumanMessage("go")], "pinned_skills": ["crm", "crm", "house-style"]}, CONFIG)

    assert _pinned_messages(model.call_history[0]["messages"]) == [CRM_PINNED, HOUSE_STYLE_PINNED]


def test_unknown_names_are_skipped_and_cleared(tmp_path: Path) -> None:
    write_skill(tmp_path, "crm", content=CRM)
    model = _model()
    agent = _agent(tmp_path, model)

    agent.invoke({"messages": [HumanMessage("turn 1")], "pinned_skills": ["no-such-skill", "crm"]}, CONFIG)
    agent.invoke({"messages": [HumanMessage("turn 2")]}, CONFIG)

    assert _pinned_messages(model.call_history[0]["messages"]) == [CRM_PINNED]
    assert _pinned_messages(model.call_history[1]["messages"]) == [CRM_PINNED]


def test_all_unknown_names_pin_nothing_and_are_cleared(tmp_path: Path) -> None:
    write_skill(tmp_path, "crm", content=CRM)
    model = _model()
    agent = _agent(tmp_path, model)

    result = agent.invoke({"messages": [HumanMessage("go")], "pinned_skills": ["no-such-skill"]}, CONFIG)

    assert result["messages"][-1].content == "done"
    assert _pinned_messages(model.call_history[0]["messages"]) == []
    assert not agent.get_state(CONFIG).values.get("pinned_skills")


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(None, id="deleted"),
        pytest.param(b"", id="empty"),
        pytest.param(b"\xff\xfe not utf-8", id="not-utf-8"),
        pytest.param(b"x" * (MAX_SKILL_FILE_SIZE + 1), id="too-large"),
    ],
)
def test_skill_unreadable_since_it_was_loaded_is_skipped(tmp_path: Path, content: bytes | None) -> None:
    write_skill(tmp_path, "crm", content=CRM)
    write_skill(tmp_path, "house-style", content=HOUSE_STYLE)
    model = _model()
    agent = _agent(tmp_path, model)
    agent.invoke({"messages": [HumanMessage("load skills")]}, CONFIG)
    skill_md = tmp_path / "skills" / "crm" / "SKILL.md"
    if content is None:
        skill_md.unlink()
    else:
        skill_md.write_bytes(content)

    result = agent.invoke({"messages": [HumanMessage("go")], "pinned_skills": ["crm", "house-style"]}, CONFIG)

    assert result["messages"][-1].content == "done"
    assert _pinned_messages(model.call_history[-1]["messages"]) == [HOUSE_STYLE_PINNED]


@tool
def pin_skill(skill: str, runtime: ToolRuntime) -> Command:
    """Pin `skill`."""
    return Command(update={"pinned_skills": [skill], "messages": [ToolMessage(f"pinning {skill}", tool_call_id=runtime.tool_call_id)]})


def test_parallel_tool_calls_each_pin_a_skill(tmp_path: Path) -> None:
    write_skill(tmp_path, "crm", content=CRM)
    write_skill(tmp_path, "house-style", content=HOUSE_STYLE)
    model = _model(ai(call("pin_skill", "a1", skill="crm"), call("pin_skill", "a2", skill="house-style")))

    _agent(tmp_path, model, tools=[pin_skill]).invoke({"messages": [HumanMessage("go")]}, CONFIG)

    _system, _user, _calls, *rest = model.call_history[1]["messages"]
    assert [m.content for m in rest] == ["pinning crm", "pinning house-style", CRM_PINNED, HOUSE_STYLE_PINNED]


def test_pinned_skill_discloses_its_tools_until_summarized_away(tmp_path: Path) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    model = _model(
        ai(call("create_customer_request", "c1", title="a")),
        # Six messages: summarization keeps only the c2 exchange, dropping the pinned skill.
        ai(call("create_customer_request", "c2", title="b")),
    )
    summary_model = GenericFakeChatModel(messages=iter(["summary"] * 10))
    summarization = SummarizationMiddleware(model=summary_model, backend=skills_backend(tmp_path), trigger=("messages", 5), keep=("messages", 2))
    agent = skills_agent(tmp_path, model, middleware=[summarization], checkpointer=InMemorySaver())

    result = agent.invoke({"messages": [HumanMessage("file two requests")], "pinned_skills": ["crm"]}, CONFIG)

    assert [m.content for m in tool_messages(result, "create_customer_request")] == ["created a (c1)", "created b (c2)"]
    assert "create_customer_request" in bound_tool_names(model.call_history[1])
    assert "create_customer_request" not in bound_tool_names(model.call_history[2])


def test_subagent_result_never_pins_skills_in_the_parent(tmp_path: Path) -> None:
    write_skill(tmp_path, "crm", content=CRM)
    worker = RunnableLambda(lambda _state: {"messages": [AIMessage("worker done")], "pinned_skills": ["crm"]})
    subagent = CompiledSubAgent(name="worker", description="Does work.", runnable=worker)
    model = _model(ai(call("task", "t1", description="do it", subagent_type="worker")))

    _agent(tmp_path, model, subagents=[subagent]).invoke({"messages": [HumanMessage("delegate")]}, CONFIG)

    assert model.call_history[1]["messages"][-1].content == "worker done"
    assert _pinned_messages(model.call_history[1]["messages"]) == []


class _SyncOnlyBeforeModel(SkillsMiddleware):
    """Overrides only the sync `before_model`, with `AgentMiddleware`'s signature."""

    def before_model(self, state: SkillsState, runtime: Runtime) -> None:
        return None


def test_subclass_overriding_only_sync_before_model_still_runs_async(tmp_path: Path) -> None:
    write_skill(tmp_path, "crm", content=CRM)
    skills = _SyncOnlyBeforeModel(backend=skills_backend(tmp_path), sources=[SKILLS_SOURCE])
    agent = create_deep_agent(model=_model(), backend=skills_backend(tmp_path), middleware=[skills])

    result = invoke(agent, {"messages": [HumanMessage("go")]}, "async")

    assert result["messages"][-1].content == "done"
