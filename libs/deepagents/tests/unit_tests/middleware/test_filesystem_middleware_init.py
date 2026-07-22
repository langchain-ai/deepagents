"""Unit tests for FilesystemMiddleware initialization and configuration."""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

from deepagents.backends import CompositeBackend, LocalShellBackend, StateBackend, StoreBackend
from deepagents.middleware.filesystem import (
    EXECUTION_SYSTEM_PROMPT,
    WRITE_FILE_TOOL_DESCRIPTION,
    FilesystemMiddleware,
)
from tests.unit_tests.chat_model import GenericFakeChatModel


def build_composite_state_backend(*, routes: dict[str, Any]) -> CompositeBackend:
    return CompositeBackend(default=StateBackend(), routes=routes)


class TestDynamicSystemPromptCache:
    """`_build_dynamic_system_prompt` caches per `include_execution` flag."""

    def test_returns_identical_cached_object(self) -> None:
        mw = FilesystemMiddleware(backend=StateBackend())
        first = mw._build_dynamic_system_prompt(include_execution=False)
        second = mw._build_dynamic_system_prompt(include_execution=False)
        assert first is second

    def test_execution_flag_changes_output(self) -> None:
        mw = FilesystemMiddleware(backend=StateBackend())
        without = mw._build_dynamic_system_prompt(include_execution=False)
        with_exec = mw._build_dynamic_system_prompt(include_execution=True)
        assert without != with_exec
        assert EXECUTION_SYSTEM_PROMPT not in without
        assert EXECUTION_SYSTEM_PROMPT in with_exec

    async def test_awrap_model_call_emits_dynamic_prompt(self) -> None:
        """`awrap_model_call` appends the same memoized prompt as the sync path.

        The cache call site is duplicated across `wrap_model_call` and
        `awrap_model_call`; this guards the async path against drift.
        """
        mw = FilesystemMiddleware(backend=StateBackend())
        # StateBackend has no execution support, so the execute tool (if any)
        # is filtered out and `include_execution` resolves to False.
        expected = mw._build_dynamic_system_prompt(include_execution=False)

        captured: list[ModelRequest] = []

        async def handler(request: ModelRequest) -> ModelResponse:
            captured.append(request)
            return ModelResponse(result=[AIMessage(content="ok")])

        request = ModelRequest(
            model=GenericFakeChatModel(messages=iter([AIMessage(content="ok")])),
            messages=[HumanMessage(content="hi")],
            tools=list(mw.tools),
        )

        await mw.awrap_model_call(request, handler)

        assert len(captured) == 1
        assert captured[0].system_prompt == expected


class TestLargeToolResultsPrompt:
    """Search guidance reflects the filesystem tools visible to the model."""

    def test_read_file_only_omits_search_guidance(self) -> None:
        middleware = FilesystemMiddleware(backend=StateBackend(), tools=["read_file"])

        prompt = middleware._build_dynamic_system_prompt(include_execution=False)

        assert (
            "In those cases, use `read_file` to inspect the saved result in chunks. "
            "Offloaded tool results are stored under `/large_tool_results/<tool_call_id>`."
        ) in prompt
        assert "`grep`" not in prompt
        assert "`execute`" not in prompt

    def test_execute_uses_shell_grep_guidance(self) -> None:
        middleware = FilesystemMiddleware(
            backend=LocalShellBackend(virtual_mode=True),
            tools=["read_file", "execute"],
        )

        prompt = middleware._build_dynamic_system_prompt(include_execution=True)

        assert (
            "or try `execute` with `grep -r <pattern> /large_tool_results/` if you need to search "
            "across offloaded tool results and do not know the exact file path"
        ) in prompt
        assert "or use `grep` within" not in prompt

    def test_backend_filtered_execute_omits_search_guidance(self) -> None:
        middleware = FilesystemMiddleware(
            backend=StateBackend(),
            tools=["read_file", "execute"],
        )

        prompt = middleware._build_dynamic_system_prompt(include_execution=False)

        assert "grep -r" not in prompt
        assert "or use `grep` within" not in prompt

    def test_grep_keeps_existing_search_guidance(self) -> None:
        middleware = FilesystemMiddleware(
            backend=StateBackend(),
            tools=["read_file", "grep"],
        )

        prompt = middleware._build_dynamic_system_prompt(include_execution=False)

        assert (
            "or use `grep` within `/large_tool_results/` if you need to search across offloaded tool results and do not know the exact file path"
        ) in prompt

    def test_default_tools_keep_existing_search_guidance(self) -> None:
        middleware = FilesystemMiddleware(backend=StateBackend())

        prompt = middleware._build_dynamic_system_prompt(include_execution=False)

        assert (
            "or use `grep` within `/large_tool_results/` if you need to search across offloaded tool results and do not know the exact file path"
        ) in prompt


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
