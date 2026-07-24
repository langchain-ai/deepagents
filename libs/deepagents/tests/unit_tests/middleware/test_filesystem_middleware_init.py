"""Unit tests for FilesystemMiddleware initialization and configuration."""

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langgraph.store.memory import InMemoryStore

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.middleware.filesystem import (
    GREP_TOOL_DESCRIPTION,
    READ_FILE_TOOL_DESCRIPTION,
    WRITE_FILE_TOOL_DESCRIPTION,
    FilesystemMiddleware,
)


def build_composite_state_backend(*, routes: dict[str, Any]) -> CompositeBackend:
    return CompositeBackend(default=StateBackend(), routes=routes)


class TestLargeToolResultGuidanceInToolDescriptions:
    """Large-tool-result offload guidance lives in the tool descriptions.

    It used to be in the (now-trimmed) filesystem system prompt, so it is
    migrated into the always-visible `read_file` / `grep` descriptions.
    """

    def test_read_file_describes_offloaded_results(self) -> None:
        # read_file points at the exact path from the tool message (no hardcoded
        # directory, which would be wrong for a non-root artifacts root).
        assert "offloaded" in READ_FILE_TOOL_DESCRIPTION.lower()

    def test_grep_describes_searching_offloaded_results(self) -> None:
        assert "large_tool_results/" in GREP_TOOL_DESCRIPTION
        # Must not imply the root-only path; it is under the artifacts root.
        assert "artifacts root" in GREP_TOOL_DESCRIPTION


class TestFilesystemMiddlewareInit:
    """Tests for FilesystemMiddleware initialization that don't require LLM invocation."""

    def test_backend_class_is_rejected(self) -> None:
        """Backend factories were removed in 0.7; callers must pass instances."""
        with pytest.raises(TypeError, match=r"Backend factories were removed in deepagents 0\.7"):
            FilesystemMiddleware(backend=StateBackend)  # type: ignore[arg-type]

    def test_backend_factory_is_rejected(self) -> None:
        """Backend factories were removed in 0.7; callers must pass instances."""
        with pytest.raises(TypeError, match=r"Backend factories were removed in deepagents 0\.7"):
            FilesystemMiddleware(backend=lambda _rt: StateBackend())  # type: ignore[arg-type]

    def test_callable_backend_instance_is_accepted(self) -> None:
        """A callable initialized backend remains a backend instance."""

        class CallableStateBackend(StateBackend):
            def __call__(self) -> None:
                return None

        backend = CallableStateBackend()
        middleware = FilesystemMiddleware(backend=backend)

        assert middleware.backend is backend

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

    def test_filesystem_tool_prompt_override_with_longterm_memory(self) -> None:
        """Test that custom tool descriptions work with composite backends and longterm memory."""
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-6"),
            middleware=[
                FilesystemMiddleware(
                    backend=build_composite_state_backend(routes={"/memories/": StoreBackend(namespace=lambda _rt: ("filesystem",))}),
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
