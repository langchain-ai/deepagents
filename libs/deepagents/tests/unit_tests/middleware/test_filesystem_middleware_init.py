"""Unit tests for FilesystemMiddleware initialization and configuration."""

from typing import Any

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langgraph.store.memory import InMemoryStore

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.composite import Route, RoutePolicy
from deepagents.middleware.filesystem import (
    WRITE_FILE_TOOL_DESCRIPTION,
    FilesystemMiddleware,
    _build_policy_prompt,
)


class TestBuildPolicyPrompt:
    """Tests for _build_policy_prompt."""

    def test_returns_none_for_non_composite_backend(self) -> None:
        assert _build_policy_prompt(StateBackend()) is None

    def test_returns_none_when_no_policies(self) -> None:
        backend = CompositeBackend(default=StateBackend(), routes={})
        assert _build_policy_prompt(backend) is None

    def test_default_policy_shows_tool_names(self) -> None:
        backend = CompositeBackend(
            default=StateBackend(),
            routes={},
            default_policy=RoutePolicy(allowed_methods={"ls", "read", "glob", "grep"}),
        )
        prompt = _build_policy_prompt(backend)
        assert prompt is not None
        assert "glob" in prompt
        assert "grep" in prompt
        assert "ls" in prompt
        assert "read_file" in prompt
        assert "read" not in prompt.replace("read_file", "")

    def test_route_policy_shows_tool_names(self) -> None:
        backend = CompositeBackend(
            default=StateBackend(),
            routes={
                "/docs/": Route(
                    backend=StateBackend(),
                    policy=RoutePolicy(allowed_methods={"ls", "read", "write", "glob", "grep"}),
                ),
            },
        )
        prompt = _build_policy_prompt(backend)
        assert prompt is not None
        assert "`/docs/`" in prompt
        assert "write_file" in prompt
        assert "read_file" in prompt
        assert "edit_file" not in prompt

    def test_non_tool_methods_are_excluded(self) -> None:
        backend = CompositeBackend(
            default=StateBackend(),
            routes={},
            default_policy=RoutePolicy(allowed_methods={"read", "upload_files", "download_files"}),
        )
        prompt = _build_policy_prompt(backend)
        assert prompt is not None
        assert "upload_files" not in prompt
        assert "download_files" not in prompt
        assert "read_file" in prompt

    def test_multiple_routes_sorted(self) -> None:
        backend = CompositeBackend(
            default=StateBackend(),
            routes={
                "/z/": Route(backend=StateBackend(), policy=RoutePolicy(allowed_methods={"read"})),
                "/a/": Route(backend=StateBackend(), policy=RoutePolicy(allowed_methods={"ls"})),
            },
        )
        prompt = _build_policy_prompt(backend)
        assert prompt is not None
        assert prompt.index("/a/") < prompt.index("/z/")


def build_composite_state_backend(*, routes: dict[str, Any]) -> CompositeBackend:
    return CompositeBackend(default=StateBackend(), routes=routes)


class TestFilesystemMiddlewareInit:
    """Tests for FilesystemMiddleware initialization that don't require LLM invocation."""

    def test_filesystem_tool_prompt_override(self) -> None:
        """Test that custom tool descriptions can be set via FilesystemMiddleware."""
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
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
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
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
