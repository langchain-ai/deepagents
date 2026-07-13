"""Unit tests for FilesystemMiddleware initialization and configuration."""

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
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
