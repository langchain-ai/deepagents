"""Unit tests for FilesystemMiddleware initialization and configuration."""

from typing import Any, cast

import pytest
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.store.memory import InMemoryStore

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from deepagents.middleware.filesystem import (
    _EXECUTE_TOOL_DESCRIPTION_WITH_GLOB_ONLY,
    _EXECUTE_TOOL_DESCRIPTION_WITH_GREP_ONLY,
    _GREP_TOOL_DESCRIPTION_WITHOUT_EXECUTE,
    GREP_TOOL_DESCRIPTION,
    READ_FILE_TOOL_DESCRIPTION,
    WRITE_FILE_TOOL_DESCRIPTION,
    FilesystemMiddleware,
)
from tests.unit_tests.chat_model import GenericFakeChatModel


def build_composite_state_backend(*, routes: dict[str, Any]) -> CompositeBackend:
    return CompositeBackend(default=StateBackend(), routes=routes)


class _SandboxBackend(SandboxBackendProtocol, StateBackend):
    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return ExecuteResponse(output="", exit_code=0)


def _ls_then_done(runs: int) -> GenericFakeChatModel:
    turn = [AIMessage(content="", tool_calls=[{"name": "ls", "args": {"path": "/"}, "id": "ls"}]), AIMessage(content="done")]
    return GenericFakeChatModel(messages=iter(turn * runs))


def _grep_tools_sent(model: GenericFakeChatModel) -> list[BaseTool]:
    return [next(tool for tool in call["tools"] if tool.name == "grep") for call in model.call_history]


class TestRequestToolDescriptions:
    """The grep and execute descriptions are rewritten per request without rebuilding the tool each call."""

    @pytest.mark.parametrize(
        ("backend", "expected_description"),
        [
            pytest.param(StateBackend(), _GREP_TOOL_DESCRIPTION_WITHOUT_EXECUTE, id="without-execute"),
            pytest.param(_SandboxBackend(), GREP_TOOL_DESCRIPTION, id="with-execute"),
        ],
    )
    async def test_model_gets_one_grep_tool_across_calls(self, backend: StateBackend, expected_description: str) -> None:
        model = _ls_then_done(runs=2)
        agent = create_agent(model=model, middleware=[FilesystemMiddleware(backend=backend)])

        agent.invoke({"messages": [HumanMessage(content="go")]})
        await agent.ainvoke({"messages": [HumanMessage(content="go")]})

        grep_tools = _grep_tools_sent(model)
        assert len(grep_tools) == 4
        assert all(tool is grep_tools[0] for tool in grep_tools)
        assert grep_tools[0].description == expected_description

    def test_each_description_gets_its_own_copy(self) -> None:
        middleware = FilesystemMiddleware(backend=_SandboxBackend())
        execute = next(tool for tool in middleware.tools if tool.name == "execute")

        def described(visible: set[str]) -> BaseTool:
            (tool,) = middleware._with_filtered_execute_description([execute], visible_search_tools=visible)
            return cast("BaseTool", tool)

        grep_only, glob_only, grep_only_again = described({"grep"}), described({"glob"}), described({"grep"})

        assert grep_only.description == _EXECUTE_TOOL_DESCRIPTION_WITH_GREP_ONLY
        assert glob_only.description == _EXECUTE_TOOL_DESCRIPTION_WITH_GLOB_ONLY
        assert grep_only is grep_only_again
        assert glob_only is not grep_only

    def test_each_source_tool_gets_its_own_copy(self) -> None:
        middleware = FilesystemMiddleware(backend=StateBackend())
        grep = next(tool for tool in middleware.tools if tool.name == "grep")
        other_grep = grep.model_copy(update={"func": lambda **_: "other"})

        (from_grep,) = middleware._with_filtered_grep_description([grep], include_execution=False)
        (from_other,) = middleware._with_filtered_grep_description([other_grep], include_execution=False)

        assert from_other is not from_grep
        assert cast("StructuredTool", from_other).func is cast("StructuredTool", other_grep).func
        assert cast("BaseTool", from_other).description == _GREP_TOOL_DESCRIPTION_WITHOUT_EXECUTE


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

    def test_state_backend_adds_files_state(self) -> None:
        middleware = FilesystemMiddleware(backend=StateBackend())
        agent = create_agent(model=ChatAnthropic(model="claude-sonnet-4-6"), middleware=[middleware])

        assert "files" in middleware.state_schema.__annotations__
        assert "files" in agent.channels

    def test_default_backend_adds_files_state(self) -> None:
        middleware = FilesystemMiddleware()

        assert "files" in middleware.state_schema.__annotations__

    def test_non_state_backend_uses_base_state(self) -> None:
        middleware = FilesystemMiddleware(backend=StoreBackend(namespace=lambda _rt: ("filesystem",)))
        agent = create_agent(model=ChatAnthropic(model="claude-sonnet-4-6"), middleware=[middleware])

        assert "files" not in middleware.state_schema.__annotations__
        assert "files" not in agent.channels

    def test_composite_state_backend_adds_files_state(self) -> None:
        store_backend = StoreBackend(namespace=lambda _rt: ("filesystem",))
        nested_backend = CompositeBackend(default=store_backend, routes={"/ephemeral/": StateBackend()})
        backend = CompositeBackend(default=store_backend, routes={"/nested/": nested_backend})
        middleware = FilesystemMiddleware(backend=backend)

        assert "files" in middleware.state_schema.__annotations__

    def test_composite_without_state_backend_uses_base_state(self) -> None:
        store_backend = StoreBackend(namespace=lambda _rt: ("filesystem",))
        backend = CompositeBackend(default=store_backend, routes={"/persistent/": store_backend})
        middleware = FilesystemMiddleware(backend=backend)

        assert "files" not in middleware.state_schema.__annotations__

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
