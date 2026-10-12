"""Server-side skill pinning preserves selected paths and directory trust."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import SkillsState
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command

from deepagents_code.app import DeepAgentsApp
from deepagents_code.local_context import LocalContextMiddleware
from deepagents_code.plugins.adapters.skills_middleware import PluginSkillsMiddleware
from deepagents_code.skills import trust
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


def _state(path: Path, name: str = "review") -> SkillsState:
    return {
        "messages": [
            HumanMessage(
                "Review",
                additional_kwargs={
                    "__skill": {"name": name, "path": str(path.resolve())}
                },
            )
        ],
        "pinned_skills": [name],
    }


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


async def test_pinning_refreshes_cached_selection(tmp_path: Path, mode: str) -> None:
    user = tmp_path / "user"
    project = tmp_path / "project"
    path = _write_skill(user / "review" / "SKILL.md", "Old instructions")
    middleware = PluginSkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False),
        sources=[(str(user), "User"), (str(project), "Project")],
    )
    state = _state(path)
    first = await _pin(middleware, state, mode)
    assert first is not None
    selected = _write_skill(project / "review" / "SKILL.md", "New instructions")
    state["messages"] = _state(selected)["messages"]
    result = await _pin(middleware, state, mode)
    assert result is not None
    messages = result["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert "New instructions" in messages[0].content
    assert "Old instructions" not in messages[0].content
    assert messages[0].additional_kwargs["lc_source"] == "pinned_skill"


@pytest.mark.parametrize("refresh_context", [False, True])
async def test_pinning_rejects_changed_selection(
    tmp_path: Path, mode: str, refresh_context: bool
) -> None:
    root = tmp_path / "skills"
    path = _write_skill(root / "review" / "SKILL.md")
    middleware = PluginSkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False), sources=[(str(root), "User")]
    )
    state = _state(tmp_path / "different" / "SKILL.md")
    if refresh_context:
        refresh = LocalContextMiddleware._refresh_update(
            {"messages": [], "_local_context": "Old context"}, "New context", 4
        )
        state["messages"].extend(refresh["messages"])
    with pytest.raises(ValueError, match="changed location"):
        await _pin(middleware, state, mode)
    assert path.exists()


async def test_pinning_rechecks_directory_trust(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    outside = _write_skill(tmp_path / "outside" / "review" / "SKILL.md")
    (root / "review").symlink_to(outside.parent, target_is_directory=True)
    middleware = PluginSkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False), sources=[(str(root), "User")]
    )
    monkeypatch.setattr(trust, "load_trusted_skill_dirs", list)
    with pytest.raises(ValueError, match="outside all allowed"):
        await _pin(middleware, _state(outside), mode)
    monkeypatch.setattr(trust, "load_trusted_skill_dirs", lambda: [outside.parent])
    result = await _pin(middleware, _state(outside), mode)
    assert result is not None
    assert result["messages"]
    monkeypatch.setattr(trust, "load_trusted_skill_dirs", list)
    with pytest.raises(ValueError, match="outside all allowed"):
        await _pin(middleware, _state(outside), mode)


async def test_pinning_unknown_skill_fails(tmp_path: Path, mode: str) -> None:
    middleware = PluginSkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False), sources=[(str(tmp_path), "User")]
    )
    with pytest.raises(ValueError, match="Run /reload"):
        await _pin(middleware, _state(tmp_path / "missing" / "SKILL.md"), mode)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("delete", "Could not read"),
        ("empty", "Could not read"),
        ("invalid-utf8", "Could not read"),
        ("frontmatter-only", "no instructions after its frontmatter"),
        ("redirect", "changed location"),
    ],
)
async def test_pinning_revalidates_actual_read(
    tmp_path: Path, mode: str, change: str, error: str
) -> None:
    root = tmp_path / "skills"
    path = _write_skill(root / "review" / "SKILL.md")
    middleware = PluginSkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False), sources=[(str(root), "User")]
    )
    state = _state(path)
    metadata = middleware.before_agent(state, Runtime(), {})
    assert metadata is not None
    state.update(metadata)
    if change == "delete":
        path.unlink()
    elif change == "empty":
        path.write_bytes(b"")
    elif change == "invalid-utf8":
        path.write_bytes(b"\xff")
    elif change == "frontmatter-only":
        _write_skill(path, "")
    else:
        other = _write_skill(root / "other" / "SKILL.md")
        path.unlink()
        path.symlink_to(other)
    if mode == "async":
        with pytest.raises(ValueError, match=error):
            await middleware.abefore_model(state, Runtime())
    else:
        with pytest.raises(ValueError, match=error):
            middleware.before_model(state, Runtime())


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
        first = await agent.ainvoke(_state(path), config)
    else:
        first = agent.invoke(_state(path), config)
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
        "Review",
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


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        ("missing", "Run /reload"),
        ("unreadable", "no instructions"),
        ("untrusted", "outside all allowed"),
    ],
)
@pytest.mark.parametrize("followup", ["ordinary", "fixed", "different"])
async def test_graph_recovers_from_failed_pin(
    tmp_path: Path,
    mode: str,
    failure: str,
    error: str,
    followup: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    path = root / "review" / "SKILL.md"
    if failure == "unreadable":
        _write_skill(path, "")
    elif failure == "untrusted":
        outside = _write_skill(tmp_path / "outside" / "SKILL.md")
        (root / "review").symlink_to(outside.parent, target_is_directory=True)
    monkeypatch.setattr(trust, "load_trusted_skill_dirs", list)
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
        with pytest.raises(ValueError, match=error):
            await agent.ainvoke(_state(path), config)
    else:
        with pytest.raises(ValueError, match=error):
            agent.invoke(_state(path), config)
    assert agent.get_state(config).values["pinned_skills"] == ["review"]
    request: SkillsState = {"messages": [HumanMessage("Continue without a skill")]}
    if followup == "fixed":
        if failure == "untrusted":
            (root / "review").unlink()
        _write_skill(path, "Restored instructions")
    elif followup == "different":
        other = _write_skill(root / "other" / "SKILL.md", name="other")
        request = _state(other, "other")
    if mode == "async":
        result = await agent.ainvoke(request, config)
    else:
        result = agent.invoke(request, config)
    assert result["messages"][-1].content == "Recovered"
    pinned = [
        message.additional_kwargs["skill"]["name"]
        for message in result["messages"]
        if message.additional_kwargs.get("lc_source") == "pinned_skill"
    ]
    assert pinned == (["other"] if followup == "different" else [])
    assert agent.get_state(config).values["pinned_skills"] == []


async def test_graph_tool_can_pin_during_ordinary_turn(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_skill(tmp_path / "review" / "SKILL.md")

    @tool
    def pin_review() -> Command:
        """Pin the review instructions."""
        return Command(
            update={
                "pinned_skills": ["review"],
                "messages": [ToolMessage("Pinned review", tool_call_id="pin-1")],
            }
        )

    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "pin_review", "args": {}, "id": "pin-1"}],
                ),
                AIMessage(content="Done"),
            ]
        )
    )
    monkeypatch.setattr(
        GenericFakeChatModel, "bind_tools", lambda *_args, **_kwargs: model
    )
    middleware = PluginSkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False), sources=[(str(tmp_path), "User")]
    )
    agent = create_agent(model, tools=[pin_review], middleware=[middleware])
    request = {"messages": [HumanMessage("Review this")]}
    result = await agent.ainvoke(request) if mode == "async" else agent.invoke(request)
    pinned = [
        message
        for message in result["messages"]
        if message.additional_kwargs.get("lc_source") == "pinned_skill"
    ]
    assert len(pinned) == 1
    assert pinned[0].additional_kwargs["skill"]["name"] == "review"


async def test_pinning_plugin_name(tmp_path: Path, mode: str) -> None:
    path = _write_skill(tmp_path / "nested" / "review" / "SKILL.md")
    middleware = PluginSkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False),
        sources=[(str(tmp_path), "Plugin", "example")],
    )
    result = await _pin(middleware, _state(path, "example:nested:review"), mode)
    assert result is not None
    messages = result["messages"]
    assert isinstance(messages, list)
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].additional_kwargs["skill"]["name"] == "example:nested:review"


@pytest.mark.parametrize("configured_base", [True, False])
async def test_pinning_resolves_extra_dirs_against_path_base(
    tmp_path: Path,
    mode: str,
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
    monkeypatch.setattr(trust, "load_trusted_skill_dirs", list)
    middleware = PluginSkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False),
        sources=[(str(root), "User")],
        path_base=user_cwd if configured_base else project,
    )
    if configured_base:
        result = await _pin(middleware, _state(target), mode)
        assert result is not None
        assert result["messages"]
    else:
        with pytest.raises(ValueError, match="outside all allowed"):
            await _pin(middleware, _state(target), mode)


async def test_pinning_checks_every_requested_name(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duplicates pin once, and a name other than the selection is trust-checked.

    Several names reach `before_model` only when a tool pins mid-turn, after
    `before_agent` has run, so the model hook is called directly.
    """
    root = tmp_path / "skills"
    path = _write_skill(root / "review" / "SKILL.md")
    _write_skill(root / "lint" / "SKILL.md", "Lint instructions", name="lint")
    outside = _write_skill(tmp_path / "outside" / "evil" / "SKILL.md", name="evil")
    (root / "evil").symlink_to(outside.parent, target_is_directory=True)
    monkeypatch.setattr(trust, "load_trusted_skill_dirs", list)
    middleware = PluginSkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False), sources=[(str(root), "User")]
    )
    state = _state(path)
    metadata = middleware.before_agent(state, Runtime(), {})
    assert metadata is not None
    state["skills_metadata"] = metadata["skills_metadata"]

    async def pin(names: list[str]) -> dict[str, object] | None:
        state["pinned_skills"] = names
        if mode == "async":
            return await middleware.abefore_model(state, Runtime())
        return middleware.before_model(state, Runtime())

    result = await pin(["review", "review", "lint"])
    assert result is not None
    messages = result["messages"]
    assert isinstance(messages, list)
    assert all(isinstance(m, HumanMessage) for m in messages)
    names = [
        cast("HumanMessage", m).additional_kwargs["skill"]["name"] for m in messages
    ]
    assert names == ["review", "lint"]
    with pytest.raises(ValueError, match="outside all allowed"):
        await pin(["review", "evil"])
