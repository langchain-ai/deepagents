"""Unit tests for FilesystemMiddleware initialization and configuration."""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend, StoreBackend
from deepagents.middleware.filesystem import (
    EXECUTION_SYSTEM_PROMPT,
    WRITE_FILE_TOOL_DESCRIPTION,
    FilesystemMiddleware,
)
from tests.unit_tests.chat_model import GenericFakeChatModel


def build_composite_state_backend(*, routes: dict[str, Any]) -> CompositeBackend:
    return CompositeBackend(default=StateBackend(), routes=routes)


def _runtime(tool_call_id: str = "read-file-test") -> ToolRuntime:
    return ToolRuntime(state={}, context=None, tool_call_id=tool_call_id, store=None, stream_writer=lambda _: None, config={})


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
            tools=[],
        )

        await mw.awrap_model_call(request, handler)

        assert len(captured) == 1
        assert captured[0].system_prompt == expected


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


class TestReadFilePaginationNotice:
    def test_partial_read_includes_remaining_line_notice(self) -> None:
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/notes.txt",
            {
                "content": "one\ntwo\nthree\nfour\n",
                "encoding": "utf-8",
                "created_at": "",
                "modified_at": "",
            },
        )
        middleware = FilesystemMiddleware(backend=StoreBackend(store=store, namespace=lambda _ctx: ("filesystem",)))
        read_file = next(tool for tool in middleware.tools if tool.name == "read_file")

        result = read_file.invoke({"runtime": _runtime(), "file_path": "/notes.txt", "offset": 0, "limit": 2})

        assert result.content.endswith("[Read 2 lines (lines 1-2 of 4 total). 2 lines remaining from offset 2.]")

    def test_read_reaching_end_omits_remaining_line_notice(self) -> None:
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/notes.txt",
            {
                "content": "one\ntwo\nthree\nfour\n",
                "encoding": "utf-8",
                "created_at": "",
                "modified_at": "",
            },
        )
        middleware = FilesystemMiddleware(backend=StoreBackend(store=store, namespace=lambda _ctx: ("filesystem",)))
        read_file = next(tool for tool in middleware.tools if tool.name == "read_file")

        result = read_file.invoke({"runtime": _runtime(), "file_path": "/notes.txt", "offset": 2, "limit": 2})

        assert "lines remaining" not in result.content

    def test_offset_read_reports_remaining_lines_from_next_offset(self) -> None:
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/notes.txt",
            {
                "content": "one\ntwo\nthree\nfour\nfive\n",
                "encoding": "utf-8",
                "created_at": "",
                "modified_at": "",
            },
        )
        middleware = FilesystemMiddleware(backend=StoreBackend(store=store, namespace=lambda _ctx: ("filesystem",)))
        read_file = next(tool for tool in middleware.tools if tool.name == "read_file")

        result = read_file.invoke({"runtime": _runtime(), "file_path": "/notes.txt", "offset": 2, "limit": 2})

        assert "lines 3-4 of 5 total" in result.content
        assert "1 line remaining from offset 4" in result.content

    def test_filesystem_backend_partial_read_includes_remaining_line_notice(self, tmp_path) -> None:
        (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
        middleware = FilesystemMiddleware(backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True))
        read_file = next(tool for tool in middleware.tools if tool.name == "read_file")

        result = read_file.invoke({"runtime": _runtime(), "file_path": "/notes.txt", "offset": 0, "limit": 2})

        assert result.content.endswith("[Read 2 lines (lines 1-2 of 3 total). 1 line remaining from offset 2.]")

    def test_blank_line_file_partial_read_includes_remaining_line_notice(self) -> None:
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/blank.txt",
            {
                "content": "\n\n\n",
                "encoding": "utf-8",
                "created_at": "",
                "modified_at": "",
            },
        )
        middleware = FilesystemMiddleware(backend=StoreBackend(store=store, namespace=lambda _ctx: ("filesystem",)))
        read_file = next(tool for tool in middleware.tools if tool.name == "read_file")

        result = read_file.invoke({"runtime": _runtime(), "file_path": "/blank.txt", "offset": 0, "limit": 2})

        assert result.content.endswith("[Read 2 lines (lines 1-2 of 3 total). 1 line remaining from offset 2.]")
