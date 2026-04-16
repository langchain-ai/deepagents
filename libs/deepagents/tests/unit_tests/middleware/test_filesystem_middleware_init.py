"""Unit tests for FilesystemMiddleware initialization and configuration."""

from typing import Any

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langgraph.store.memory import InMemoryStore

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.middleware.filesystem import (
    APPLY_PATCH_TOOL_DESCRIPTION,
    WRITE_FILE_TOOL_DESCRIPTION,
    FilesystemMiddleware,
)


def build_composite_state_backend(*, routes: dict[str, Any]) -> CompositeBackend:
    return CompositeBackend(default=StateBackend(), routes=routes)


class TestFilesystemMiddlewareInit:
    """Tests for FilesystemMiddleware initialization that don't require LLM invocation."""

    def test_filesystem_tool_prompt_override(self) -> None:
        """Test that custom tool descriptions can be set via FilesystemMiddleware."""
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=StateBackend(),
                    custom_tool_descriptions={
                        "ls": "Charmander",
                        "read_file": "Bulbasaur",
                        "edit_file": "Squirtle",
                    },
                )
            ],
        )
        tools = agent.nodes["tools"].bound._tools_by_name
        assert "ls" in tools
        assert tools["ls"].description == "Charmander"
        assert "read_file" in tools
        assert tools["read_file"].description == "Bulbasaur"
        assert "write_file" in tools
        assert tools["write_file"].description == WRITE_FILE_TOOL_DESCRIPTION.rstrip()
        assert "edit_file" in tools
        assert tools["edit_file"].description == "Squirtle"

    def test_apply_patch_tool_absent_by_default(self) -> None:
        mw = FilesystemMiddleware(backend=StateBackend())
        names = {t.name for t in mw.tools}
        assert "apply_patch" not in names

    def test_apply_patch_tool_present_when_enabled(self) -> None:
        mw = FilesystemMiddleware(backend=StateBackend(), include_apply_patch=True)
        names = {t.name for t in mw.tools}
        assert "apply_patch" in names

    def test_apply_patch_custom_description(self) -> None:
        mw = FilesystemMiddleware(
            backend=StateBackend(),
            include_apply_patch=True,
            custom_tool_descriptions={"apply_patch": "Custom desc"},
        )
        tool = next(t for t in mw.tools if t.name == "apply_patch")
        assert tool.description == "Custom desc"

    def test_apply_patch_default_description(self) -> None:
        mw = FilesystemMiddleware(backend=StateBackend(), include_apply_patch=True)
        tool = next(t for t in mw.tools if t.name == "apply_patch")
        assert tool.description == APPLY_PATCH_TOOL_DESCRIPTION.rstrip()

    def test_filesystem_tool_prompt_override_with_longterm_memory(self) -> None:
        """Test that custom tool descriptions work with composite backends and longterm memory."""
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=build_composite_state_backend(routes={"/memories/": StoreBackend()}),
                    custom_tool_descriptions={
                        "ls": "Charmander",
                        "read_file": "Bulbasaur",
                        "edit_file": "Squirtle",
                    },
                )
            ],
            store=InMemoryStore(),
        )
        tools = agent.nodes["tools"].bound._tools_by_name
        assert "ls" in tools
        assert tools["ls"].description == "Charmander"
        assert "read_file" in tools
        assert tools["read_file"].description == "Bulbasaur"
        assert "write_file" in tools
        assert tools["write_file"].description == WRITE_FILE_TOOL_DESCRIPTION.rstrip()
        assert "edit_file" in tools
        assert tools["edit_file"].description == "Squirtle"
