"""Server-side skill pinning resolves, trust-checks, and reads the skill."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import SkillsState
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command

from deepagents_code.app import DeepAgentsApp
from deepagents_code.goal_state_notice import build_goal_continuation
from deepagents_code.plugins.adapters import skills_middleware
from deepagents_code.plugins.adapters.skills_middleware import PluginSkillsMiddleware
from deepagents_code.skills import trust
from deepagents_code.skills.invocation import build_skill_invocation_envelope
from deepagents_code.tui.widgets.message_store import MessageType

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


def _write_skill(
    path: Path, body: str = "Follow these instructions.", *, name: str = "review"
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: Review code\n---\n{body}\n", encoding="utf-8"
    )
    return path


def _state(name: str = "review") -> SkillsState:
    envelope = build_skill_invocation_envelope(name)
    return {
        "messages": [HumanMessage(envelope.prompt, **envelope.message_kwargs)],
        "pinned_skills": [name],
    }


def _deny_trust(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Answer every trust interrupt with a deny, recording the requests."""
    requests: list[object] = []

    def deny(value: object) -> dict[str, bool]:
        requests.append(value)
        return {"approved": False}

    monkeypatch.setattr(skills_middleware, "interrupt", deny)
    return requests


async def _pin(
    middleware: PluginSkillsMiddleware, state: SkillsState, mode: str
) -> dict[str, object] | None:
    runtime = Runtime()
    if mode == "async":
        update = await middleware.abefore_agent(state, runtime, {})
        if update:
            state.update(update)
        return await middleware.abefore_model(state, runtime)
    update = middleware.before_agent(state, runtime, {})
    if update:
        state.update(update)
    return middleware.before_model(state, runtime)


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> str:
    return request.param


async def test_pinning_asks_to_trust_an_outside_directory(
    tmp_path: Path, mode: str
) -> None:
    """An untrusted skill interrupts, and an approval is saved and pinned."""
    root = tmp_path / "skills"
    root.mkdir()
    outside = _write_skill(tmp_path / "outside" / "review" / "SKILL.md")
    (root / "review").symlink_to(outside.parent, target_is_directory=True)
    agent = create_agent(
        GenericFakeChatModel(messages=iter([AIMessage(content="Done")])),
        middleware=[
            PluginSkillsMiddleware(
                backend=FilesystemBackend(virtual_mode=False),
                sources=[(str(root), "User")],
            )
        ],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "trust"}}
    if mode == "async":
        paused = await agent.ainvoke(_state(), config)
    else:
        paused = agent.invoke(_state(), config)
    [request] = paused["__interrupt__"]
    directory = str(outside.parent.resolve())
    assert request.value == {
        "type": "skill_trust",
        "skill": "review",
        "directory": directory,
    }
    resume = Command(resume={request.id: {"approved": True}})
    if mode == "async":
        result = await agent.ainvoke(resume, config)
    else:
        result = agent.invoke(resume, config)
    assert result["messages"][1].additional_kwargs["lc_source"] == "pinned_skill"
    assert result["messages"][-1].content == "Done"
    assert trust.is_skill_dir_trusted(directory)


async def test_pinning_refuses_a_denied_directory(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deny fails the pin, and a saved approval later skips the prompt."""
    root = tmp_path / "skills"
    root.mkdir()
    outside = _write_skill(tmp_path / "outside" / "review" / "SKILL.md")
    (root / "review").symlink_to(outside.parent, target_is_directory=True)
    middleware = PluginSkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False), sources=[(str(root), "User")]
    )
    requests = _deny_trust(monkeypatch)
    with pytest.raises(ValueError, match="outside all allowed"):
        await _pin(middleware, _state(), mode)
    assert len(requests) == 1
    assert trust.trust_skill_dir(str(outside.parent.resolve()))
    result = await _pin(middleware, _state(), mode)
    assert result is not None
    assert result["messages"]
    assert len(requests) == 1


async def test_pinning_refuses_an_approval_it_cannot_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Later turns read trust from the store, so an unsaved approval fails."""
    root = tmp_path / "skills"
    root.mkdir()
    outside = _write_skill(tmp_path / "outside" / "review" / "SKILL.md")
    (root / "review").symlink_to(outside.parent, target_is_directory=True)
    middleware = PluginSkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False), sources=[(str(root), "User")]
    )
    monkeypatch.setattr(skills_middleware, "interrupt", lambda _: {"approved": True})
    monkeypatch.setattr(skills_middleware, "trust_skill_dir", lambda _: False)
    with pytest.raises(ValueError, match="Could not save trust"):
        await _pin(middleware, _state(), "sync")


async def test_graph_pins_once_and_preserves_snapshot(
    tmp_path: Path, mode: str
) -> None:
    path = _write_skill(tmp_path / "review" / "SKILL.md", "Original instructions")
    middleware = PluginSkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False),
        sources=[(str(tmp_path), "User")],
    )
    model = GenericFakeChatModel(
        messages=iter(
            [AIMessage(content="First answer"), AIMessage(content="Second answer")]
        )
    )
    agent = create_agent(model, middleware=[middleware], checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "pin-snapshot"}}
    if mode == "async":
        first = await agent.ainvoke(_state(), config)
    else:
        first = agent.invoke(_state(), config)
    assert "pinned_skills" not in first
    assert len(first["messages"]) == 3
    pinned = first["messages"][1]
    assert isinstance(pinned, HumanMessage)
    assert pinned.content == (
        f'<skill name="review" path="{path}">\nOriginal instructions\n</skill>'
    )
    assert pinned.additional_kwargs == {
        "lc_source": "pinned_skill",
        "skill": {"name": "review", "path": str(path), "description": "Review code"},
    }
    _write_skill(path, "Changed instructions")
    next_turn = {"messages": [HumanMessage("Continue")]}
    if mode == "async":
        second = await agent.ainvoke(next_turn, config)
    else:
        second = agent.invoke(next_turn, config)
    assert [message.content for message in second["messages"]] == [
        "I'm invoking the skill `review`.",
        pinned.content,
        "First answer",
        "Continue",
        "Second answer",
    ]
    assert agent.get_state(config).values["pinned_skills"] == []

    rows = DeepAgentsApp._convert_messages_to_data(second["messages"])
    assert [row.type for row in rows] == [
        MessageType.SKILL,
        MessageType.ASSISTANT,
        MessageType.USER,
        MessageType.ASSISTANT,
    ]
    assert rows[0].skill_body == "Original instructions"


@pytest.mark.parametrize("followup", ["ordinary", "goal"])
async def test_graph_recovers_from_failed_pin(
    tmp_path: Path, mode: str, followup: str
) -> None:
    """A pin that fails on a missing skill does not leak into later turns."""
    root = tmp_path / "skills"
    root.mkdir()
    middleware = PluginSkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False), sources=[(str(root), "User")]
    )
    agent = create_agent(
        GenericFakeChatModel(messages=iter([AIMessage(content="Recovered")])),
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "failed-pin"}}
    if mode == "async":
        with pytest.raises(ValueError, match="was not found"):
            await agent.ainvoke(_state(), config)
    else:
        with pytest.raises(ValueError, match="was not found"):
            agent.invoke(_state(), config)
    assert agent.get_state(config).values["pinned_skills"] == ["review"]
    request: SkillsState = {"messages": [HumanMessage("Continue without a skill")]}
    if followup == "goal":
        request = {"messages": [build_goal_continuation("resumed")]}
    if mode == "async":
        result = await agent.ainvoke(request, config)
    else:
        result = agent.invoke(request, config)
    assert result["messages"][-1].content == "Recovered"
    assert not any(
        message.additional_kwargs.get("lc_source") == "pinned_skill"
        for message in result["messages"]
    )
    assert agent.get_state(config).values["pinned_skills"] == []


async def test_new_process_rescans_cached_skills(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new server process rescans a thread's skills and keeps its history.

    Turns in one process keep the cached list, so skills added, edited, or
    removed appear only after `/reload` restarts the server.
    """
    root = tmp_path / "skills"
    root.mkdir()
    agent = create_agent(
        GenericFakeChatModel(
            messages=iter(AIMessage(content="Done", id=f"reply-{i}") for i in range(5))
        ),
        middleware=[
            PluginSkillsMiddleware(
                backend=FilesystemBackend(virtual_mode=False),
                sources=[(str(root), "User")],
            )
        ],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "rescan"}}

    async def turn() -> dict[str, str]:
        request: SkillsState = {"messages": [HumanMessage("Continue")]}
        if mode == "async":
            await agent.ainvoke(request, config)
        else:
            agent.invoke(request, config)
        skills = agent.get_state(config).values["skills_metadata"]
        return {skill["name"]: skill["description"] for skill in skills}

    def restart(process: str) -> None:
        monkeypatch.setattr(skills_middleware, "_PROCESS_ID", process)

    assert await turn() == {}
    _write_skill(root / "edited" / "SKILL.md", name="edited")
    _write_skill(root / "removed" / "SKILL.md", name="removed")
    assert await turn() == {}
    restart("second")
    assert await turn() == {
        "edited": "Review code",
        "removed": "Review code",
    }
    (root / "removed" / "SKILL.md").unlink()
    (root / "edited" / "SKILL.md").write_text(
        "---\nname: edited\ndescription: Edited\n---\nBody\n", encoding="utf-8"
    )
    _write_skill(root / "added" / "SKILL.md", name="added")
    assert set(await turn()) == {"edited", "removed"}
    restart("third")
    assert await turn() == {
        "edited": "Edited",
        "added": "Review code",
    }
    messages = agent.get_state(config).values["messages"]
    assert [m.id for m in messages if isinstance(m, AIMessage)] == [
        f"reply-{i}" for i in range(5)
    ]


@pytest.mark.parametrize("configured_base", [True, False])
async def test_pinning_resolves_extra_dirs_against_path_base(
    tmp_path: Path,
    configured_base: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relative `extra_allowed_dirs` resolve against the user's working dir."""
    project = tmp_path / "project"
    user_cwd = project / "subdir"
    target = _write_skill(user_cwd / "shared" / "review" / "SKILL.md")
    root = tmp_path / "skills"
    root.mkdir()
    (root / "review").symlink_to(target.parent, target_is_directory=True)
    (tmp_path / "config.toml").write_text(
        '[skills]\nextra_allowed_dirs = ["shared"]\n', encoding="utf-8"
    )
    _deny_trust(monkeypatch)
    middleware = PluginSkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False),
        sources=[(str(root), "User")],
        path_base=user_cwd if configured_base else project,
    )
    if configured_base:
        result = await _pin(middleware, _state(), "sync")
        assert result is not None
        assert result["messages"]
    else:
        with pytest.raises(ValueError, match="outside all allowed"):
            await _pin(middleware, _state(), "sync")


def test_pinning_checks_every_requested_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duplicates pin once, and a name other than the selection is trust-checked.

    Several names reach `before_model` only when a tool pins mid-turn, after
    `before_agent` has run, so the model hook is called directly.
    """
    root = tmp_path / "skills"
    _write_skill(root / "review" / "SKILL.md")
    _write_skill(root / "lint" / "SKILL.md", "Lint instructions", name="lint")
    outside = _write_skill(tmp_path / "outside" / "evil" / "SKILL.md", name="evil")
    (root / "evil").symlink_to(outside.parent, target_is_directory=True)
    _deny_trust(monkeypatch)
    middleware = PluginSkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False), sources=[(str(root), "User")]
    )
    state = _state()
    metadata = middleware.before_agent(state, Runtime(), {})
    assert metadata is not None
    state["skills_metadata"] = metadata["skills_metadata"]

    def pin(names: list[str]) -> dict[str, object] | None:
        state["pinned_skills"] = names
        return middleware.before_model(state, Runtime())

    result = pin(["review", "review", "lint"])
    assert result is not None
    messages = result["messages"]
    assert isinstance(messages, list)
    assert all(isinstance(m, HumanMessage) for m in messages)
    names = [
        cast("HumanMessage", m).additional_kwargs["skill"]["name"] for m in messages
    ]
    assert names == ["review", "lint"]
    with pytest.raises(ValueError, match="outside all allowed"):
        pin(["review", "evil"])
