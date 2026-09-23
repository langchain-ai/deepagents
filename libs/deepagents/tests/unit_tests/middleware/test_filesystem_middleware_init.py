"""Unit tests for FilesystemMiddleware initialization and configuration."""

import base64
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentState, ModelRequest, ModelResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import ModelInvalidRequestError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.store.memory import InMemoryStore

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.middleware.filesystem import (
    GREP_TOOL_DESCRIPTION,
    READ_FILE_TOOL_DESCRIPTION,
    WRITE_FILE_TOOL_DESCRIPTION,
    FilesystemMiddleware,
    FilesystemState,
    _file_block_supported,
    _scrub_unsupported_multimodal_content,
)

if TYPE_CHECKING:
    from langchain_core.messages.content import ContentBlock


class MaskedChatOpenAI(ChatOpenAI):
    @property
    def _llm_type(self) -> str:
        return "langchain-chat"


class MaskedAzureChatOpenAI(AzureChatOpenAI):
    @property
    def _llm_type(self) -> str:
        return "langchain-chat"


_DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def build_composite_state_backend(*, routes: dict[str, Any]) -> CompositeBackend:
    return CompositeBackend(default=StateBackend(), routes=routes)


@pytest.mark.parametrize("model_type", [MaskedChatOpenAI, MaskedAzureChatOpenAI])
@pytest.mark.parametrize(("use_responses_api", "supported"), [(False, False), (None, False), (True, True)])
def test_openai_docx_support_uses_provider_class_and_responses_api(
    model_type: type[ChatOpenAI] | type[AzureChatOpenAI],
    *,
    use_responses_api: bool | None,
    supported: bool,
) -> None:
    model = model_type.model_construct(use_responses_api=use_responses_api)
    block: ContentBlock = {"type": "file", "base64": "UEsDBA==", "mime_type": _DOCX_MIME_TYPE}

    assert model._llm_type == "langchain-chat"
    assert _file_block_supported(block, model=model, profile={}, in_tool_message=True) is supported


@pytest.mark.parametrize(
    ("model", "mime_type", "supported"),
    [
        (ChatOpenAI.model_construct(use_responses_api=True), "application/zip", False),
        (ChatGoogleGenerativeAI.model_construct(), _DOCX_MIME_TYPE, False),
    ],
)
def test_binary_file_allowlist(model: BaseChatModel, mime_type: str, *, supported: bool) -> None:
    block: ContentBlock = {"type": "file", "base64": "UEsDBA==", "mime_type": mime_type}

    assert _file_block_supported(block, model=model, profile={}, in_tool_message=False) is supported


@pytest.mark.parametrize("mime_type", ["text/csv", "text/markdown"])
def test_openai_responses_preserves_non_utf8_text_files(mime_type: str) -> None:
    block: ContentBlock = {
        "type": "file",
        "base64": base64.b64encode(b"value\n\xff\n").decode(),
        "mime_type": mime_type,
    }
    message = HumanMessage(content=[block])

    accepted = _scrub_unsupported_multimodal_content([message], ChatOpenAI.model_construct(use_responses_api=True))
    rejected = _scrub_unsupported_multimodal_content([message], ChatOpenAI.model_construct(use_responses_api=False))

    assert accepted == [message]
    assert accepted[0] is message
    assert rejected[0].content_blocks[0]["type"] == "text"


@pytest.mark.parametrize("reference", [{"file_id": "file_1"}, {"url": "https://example.com/archive.zip"}])
def test_file_references_bypass_allowlist(reference: dict[str, str]) -> None:
    block: ContentBlock = {"type": "file", **reference}

    assert _file_block_supported(block, model=None, profile={}, in_tool_message=False)


def test_pdf_tool_message_profile_is_enforced() -> None:
    block: ContentBlock = {"type": "file", "base64": "JVBERi0=", "mime_type": "application/pdf"}

    assert not _file_block_supported(
        block,
        model=None,
        profile={"pdf_inputs": True, "pdf_tool_message": False},
        in_tool_message=True,
    )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("retry_fails", [False, True])
async def test_rejected_read_file_media_falls_back_once(*, async_mode: bool, retry_fails: bool) -> None:
    media = ToolMessage(
        content=[{"type": "image", "base64": "invalid", "mime_type": "image/png"}],
        name="read_file",
        tool_call_id="read-1",
        id="message-1",
    )
    text = ToolMessage(content="file text", name="read_file", tool_call_id="read-2")
    human = HumanMessage(content=media.content)
    other_tool = media.model_copy(update={"name": "other_tool", "tool_call_id": "other"})
    request = ModelRequest(model=ChatOpenAI.model_construct(profile={}), messages=[human, media, text, other_tool])
    response = ModelResponse(result=[AIMessage(content="Recovered")])
    error = ModelInvalidRequestError("Invalid image")
    handler = (AsyncMock if async_mode else Mock)(side_effect=[error, error if retry_fails else response])
    middleware = FilesystemMiddleware(backend=StateBackend())

    async def invoke() -> object:
        if async_mode:
            return await middleware.awrap_model_call(request, handler)
        return middleware.wrap_model_call(request, handler)

    if retry_fails:
        with pytest.raises(ModelInvalidRequestError, match="Invalid image"):
            await invoke()
    else:
        assert await invoke() is response
    assert handler.call_count == 2
    retried = handler.call_args.args[0].messages
    assert retried[1].content == "Unsupported content. The file may be invalid, too large, or of an unsupported mime-type."
    assert retried[1].tool_call_id == media.tool_call_id
    assert retried[1].id == media.id
    assert retried[0] is human
    assert retried[2] is text
    assert retried[3] is other_tool
    assert request.messages[1] is media
    assert isinstance(media.content, list)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("error_type", [ModelInvalidRequestError, RuntimeError])
@pytest.mark.parametrize("name", ["read_file", "other_tool"])
async def test_invalid_requests_without_eligible_media_propagate(*, async_mode: bool, error_type: type[Exception], name: str) -> None:
    message = ToolMessage(content="text only", name=name, tool_call_id="tool-1")
    if name == "other_tool":
        message = message.model_copy(update={"content": [{"type": "image", "base64": "invalid", "mime_type": "image/png"}]})
    request = ModelRequest(model=ChatOpenAI.model_construct(profile={}), messages=[message])
    handler = (AsyncMock if async_mode else Mock)(side_effect=error_type("Rejected"))
    middleware = FilesystemMiddleware(backend=StateBackend())

    if async_mode:
        with pytest.raises(error_type, match="Rejected"):
            await middleware.awrap_model_call(request, handler)
    else:
        with pytest.raises(error_type, match="Rejected"):
            middleware.wrap_model_call(request, handler)
    assert handler.call_count == 1


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

        assert middleware.state_schema is FilesystemState
        assert "files" in agent.channels

    def test_default_backend_adds_files_state(self) -> None:
        middleware = FilesystemMiddleware()

        assert middleware.state_schema is FilesystemState

    def test_non_state_backend_uses_base_state(self) -> None:
        middleware = FilesystemMiddleware(backend=StoreBackend(namespace=lambda _rt: ("filesystem",)))
        agent = create_agent(model=ChatAnthropic(model="claude-sonnet-4-6"), middleware=[middleware])

        assert middleware.state_schema is AgentState
        assert "files" not in agent.channels

    def test_composite_state_backend_adds_files_state(self) -> None:
        store_backend = StoreBackend(namespace=lambda _rt: ("filesystem",))
        nested_backend = CompositeBackend(default=store_backend, routes={"/ephemeral/": StateBackend()})
        backend = CompositeBackend(default=store_backend, routes={"/nested/": nested_backend})
        middleware = FilesystemMiddleware(backend=backend)

        assert middleware.state_schema is FilesystemState

    def test_composite_without_state_backend_uses_base_state(self) -> None:
        store_backend = StoreBackend(namespace=lambda _rt: ("filesystem",))
        backend = CompositeBackend(default=store_backend, routes={"/persistent/": store_backend})
        middleware = FilesystemMiddleware(backend=backend)

        assert middleware.state_schema is AgentState

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
