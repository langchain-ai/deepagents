"""Unit tests for FilesystemMiddleware initialization and configuration."""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.middleware.filesystem import (
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


def _build_model_request() -> ModelRequest:
    """Construct a minimal `ModelRequest` for direct `wrap_model_call` tests."""
    return ModelRequest(
        model=ChatAnthropic(model_name="claude-sonnet-4-6", anthropic_api_key="fake"),  # type: ignore[call-arg]
        messages=[HumanMessage(content="hi")],
        system_message=None,
        tools=[],
        state={"messages": [], "files": {}},  # type: ignore[typeddict-unknown-key]
    )


def _captured_system_prompt(middleware: FilesystemMiddleware) -> str:
    """Run `wrap_model_call` with a capturing handler and return the injected prompt."""
    captured: dict[str, Any] = {}

    def _handler(request: ModelRequest):  # noqa: ANN202
        captured["request"] = request
        return None  # wrap_model_call doesn't inspect the return value here

    middleware.wrap_model_call(_build_model_request(), _handler)
    request = captured["request"]
    if request.system_message is None:
        return ""
    content = request.system_message.content
    if isinstance(content, str):
        return content
    # Block-form content: concatenate the text from text blocks.
    return "\n".join(
        block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text"
    )


class TestScratchpadPromptInjection:
    """`FilesystemMiddleware` appends a scratchpad prompt iff the backend declares one."""

    def test_prompt_absent_when_backend_has_no_scratchpad(self) -> None:
        mw = FilesystemMiddleware(backend=StateBackend())
        prompt = _captured_system_prompt(mw)
        assert "## Scratchpad" not in prompt

    def test_prompt_absent_when_composite_has_no_scratchpad_prefix(self) -> None:
        mw = FilesystemMiddleware(
            backend=CompositeBackend(default=StateBackend(), routes={}),
        )
        prompt = _captured_system_prompt(mw)
        assert "## Scratchpad" not in prompt

    def test_prompt_present_when_composite_declares_scratchpad_prefix(self) -> None:
        mw = FilesystemMiddleware(
            backend=CompositeBackend(
                default=StateBackend(),
                routes={},
                scratchpad_prefix="/scratchpad/",
            ),
        )
        prompt = _captured_system_prompt(mw)
        assert "## Scratchpad `/scratchpad/`" in prompt
        assert "ephemeral" in prompt
        assert "/memories/" in prompt
