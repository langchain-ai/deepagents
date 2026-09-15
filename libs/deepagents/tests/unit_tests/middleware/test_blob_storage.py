import base64
import hashlib
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from deepagents.middleware import BlobStorageMiddleware


def _tool_request(store: InMemoryStore, name: str = "read_file") -> ToolCallRequest:
    runtime = ToolRuntime(state={}, context=None, tool_call_id="call-1", store=store, stream_writer=lambda _: None, config={})
    return ToolCallRequest(tool_call={"id": "call-1", "name": name, "args": {}}, tool=None, state={}, runtime=runtime)


def _binary_message(payload: bytes = b"image bytes") -> ToolMessage:
    return ToolMessage(
        content=[{"type": "image", "base64": base64.b64encode(payload).decode(), "mime_type": "image/png"}],
        name="read_file",
        tool_call_id="call-1",
    )


def test_binary_read_is_replaced_by_content_addressed_reference() -> None:
    store = InMemoryStore()
    middleware = BlobStorageMiddleware()

    result = middleware.wrap_tool_call(_tool_request(store), lambda _request: _binary_message())

    assert isinstance(result, ToolMessage)
    block = result.content[0]
    digest = hashlib.sha256(b"image bytes").hexdigest()
    assert block == {"type": "image", "mime_type": "image/png", "blob_ref": digest}
    assert "image bytes" not in str(result)
    assert store.get(("deepagents", "blobs"), digest).value == {"base64": base64.b64encode(b"image bytes").decode()}


def test_duplicate_binary_reads_share_one_blob() -> None:
    store = MagicMock(wraps=InMemoryStore())
    middleware = BlobStorageMiddleware(store=store)
    request = _tool_request(InMemoryStore())

    middleware.wrap_tool_call(request, lambda _request: _binary_message())
    middleware.wrap_tool_call(request, lambda _request: _binary_message())

    assert store.put.call_count == 1


def test_video_command_offloads_synthetic_media_message() -> None:
    store = InMemoryStore()
    middleware = BlobStorageMiddleware(store=store)
    media = HumanMessage(content=[{"type": "image", "base64": base64.b64encode(b"frame").decode(), "mime_type": "image/jpeg"}])
    command = Command(update={"messages": [ToolMessage(content="frames", tool_call_id="call-1"), media]})

    result = middleware.wrap_tool_call(_tool_request(store), lambda _request: command)

    assert isinstance(result, Command)
    assert "base64" not in result.update["messages"][1].content[0]
    assert "blob_ref" in result.update["messages"][1].content[0]


def test_non_read_file_result_is_unchanged_without_store() -> None:
    middleware = BlobStorageMiddleware()
    request = _tool_request(InMemoryStore(), name="other")
    message = _binary_message()

    assert middleware.wrap_tool_call(request, lambda _request: message) is message


def test_invalid_base64_is_rejected() -> None:
    middleware = BlobStorageMiddleware(store=InMemoryStore())
    message = _binary_message().model_copy(update={"content": [{"type": "file", "base64": "%%%"}]})

    with pytest.raises(ValueError, match="not valid base64"):
        middleware.wrap_tool_call(_tool_request(InMemoryStore()), lambda _request: message)


def test_model_request_is_hydrated_without_mutating_history() -> None:
    store = InMemoryStore()
    middleware = BlobStorageMiddleware(store=store)
    offloaded = middleware.wrap_tool_call(_tool_request(store), lambda _request: _binary_message())
    assert isinstance(offloaded, ToolMessage)
    captured: list[ToolMessage] = []
    request = ModelRequest(model=MagicMock(), messages=[offloaded], tools=[], state={}, runtime=MagicMock(store=store))

    def handler(model_request: ModelRequest) -> ModelResponse:
        captured.extend(model_request.messages)
        return ModelResponse(result=[AIMessage(content="ok")])

    middleware.wrap_model_call(request, handler)

    assert "base64" not in offloaded.content[0]
    assert captured[0].content[0]["base64"] == base64.b64encode(b"image bytes").decode()
    assert "blob_ref" not in captured[0].content[0]


async def test_async_offload_and_hydration() -> None:
    store = InMemoryStore()
    middleware = BlobStorageMiddleware(store=store)
    request = _tool_request(store)

    async def tool_handler(_request: ToolCallRequest) -> ToolMessage:
        return _binary_message()

    offloaded = await middleware.awrap_tool_call(request, tool_handler)
    assert isinstance(offloaded, ToolMessage)
    model_request = ModelRequest(model=MagicMock(), messages=[offloaded], tools=[], state={}, runtime=MagicMock(store=store))

    async def model_handler(request: ModelRequest) -> ModelResponse:
        assert request.messages[0].content[0]["base64"] == base64.b64encode(b"image bytes").decode()
        return ModelResponse(result=[AIMessage(content="ok")])

    await middleware.awrap_model_call(model_request, model_handler)
