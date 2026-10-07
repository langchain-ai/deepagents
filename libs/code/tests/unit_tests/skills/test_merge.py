"""Unit tests for skill-name collision (override) debug logging."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from deepagents.backends.filesystem import FilesystemBackend
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepagents_code.plugins.adapters.skills_middleware import PluginSkillsMiddleware

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents.middleware.skills import SkillsState
    from langchain_core.runnables import RunnableConfig


_MERGE_LOGGER = "deepagents_code.skills.merge"


def _create_skill(skill_dir: Path, name: str, description: str) -> None:
    """Create a minimal skill directory with a valid `SKILL.md`."""
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: {description}
---
Content
""")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("namespace", [None, "plugin"])
async def test_invalidated_catalog_rediscovers_skills(
    tmp_path: Path, *, asynchronous: bool, namespace: str | None
) -> None:
    """Reload adds, removes, and edits cached skills without losing history."""
    source = (
        (str(tmp_path), "Skills")
        if namespace is None
        else (str(tmp_path), "Skills", namespace)
    )
    middleware = PluginSkillsMiddleware(
        backend=FilesystemBackend(virtual_mode=False), sources=[source]
    )
    agent = create_agent(
        FakeMessagesListChatModel(
            responses=[AIMessage(content="done", id=f"reply-{i}") for i in range(5)]
        ),
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "skills"}}

    async def turn(*, refresh: bool = False) -> dict:
        payload: SkillsState = {"messages": [HumanMessage(content="continue")]}
        if refresh:
            payload["skills_metadata"] = None
        if asynchronous:
            return await agent.ainvoke(payload, config)
        return agent.invoke(payload, config)

    await turn()
    _create_skill(tmp_path / "removed", "removed", "Old skill")
    _create_skill(tmp_path / "edited", "edited", "Old description")
    await turn()
    assert agent.get_state(config).values["skills_metadata"] == []
    await turn(refresh=True)
    (tmp_path / "removed" / "SKILL.md").unlink()
    _create_skill(tmp_path / "edited", "edited", "New description")
    _create_skill(tmp_path / "added", "added", "Added skill")
    await turn()
    cached = agent.get_state(config).values["skills_metadata"]
    assert {skill["description"] for skill in cached} == {
        "Old skill",
        "Old description",
    }
    result = await turn(refresh=True)
    prefix = "plugin:" if namespace else ""
    assert {
        skill["name"]: skill["description"]
        for skill in agent.get_state(config).values["skills_metadata"]
    } == {f"{prefix}edited": "New description", f"{prefix}added": "Added skill"}
    assert len(result["messages"]) == 10


class TestMergeSkillHelper:
    """Directly exercise `merge_skill`."""


class TestListSkillsCollisionLogging:
    """Exercise collision logging through the CLI `list_skills` discovery path."""


class TestMiddlewareCollisionLogging:
    """Exercise collision logging through `PluginSkillsMiddleware` (sync + async).

    These lock in the new three-way `zip(self.sources, self.source_labels,
    self._namespaces, ...)` wiring: both entry points must merge through
    `merge_skill` and log overrides identically.
    """

    @staticmethod
    def _middleware(user_dir: Path, project_dir: Path) -> PluginSkillsMiddleware:
        """Build a middleware over two colliding, non-namespaced sources."""
        _create_skill(user_dir / "review", "review", "User review")
        _create_skill(project_dir / "review", "review", "Project review")
        return PluginSkillsMiddleware(
            backend=FilesystemBackend(virtual_mode=False),
            sources=[(str(user_dir), "User"), (str(project_dir), "Project")],
            system_prompt=None,
        )

    @staticmethod
    def _namespaced_middleware(dir_a: Path, dir_b: Path) -> PluginSkillsMiddleware:
        """Build a middleware over two colliding plugin (namespaced) sources.

        Both sources share the `myplugin` namespace and a `review` skill, so
        both qualify to `myplugin:review` and drive the namespaced (`else`)
        loop branch through `load_namespaced_skills` — the branch a plain-source
        collision never reaches.
        """
        _create_skill(dir_a / "review", "review", "Plugin A review")
        _create_skill(dir_b / "review", "review", "Plugin B review")
        return PluginSkillsMiddleware(
            backend=FilesystemBackend(virtual_mode=False),
            sources=[
                (str(dir_a), "Plugin A", "myplugin"),
                (str(dir_b), "Plugin B", "myplugin"),
            ],
            system_prompt=None,
        )
