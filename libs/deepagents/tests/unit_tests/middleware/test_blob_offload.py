import base64
import hashlib
from pathlib import Path
from typing import Any

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from deepagents.backends import FilesystemBackend
from deepagents.middleware._blob_offload import _BLOB_REF_KEY, _MISSING_BLOB_TEXT, _BlobCache
from deepagents.middleware.filesystem import FilesystemMiddleware

PNG = b"\x89PNG\r\n\x1a\n fake image bytes"
PNG_B64 = base64.b64encode(PNG).decode("ascii")
PNG_DIGEST = hashlib.sha256(PNG).hexdigest()


def _request(tool_name: str = "read_file") -> ToolCallRequest:
    runtime = ToolRuntime(state={}, context=None, tool_call_id="call_1", store=None, stream_writer=lambda _: None, config={})
    return ToolCallRequest(runtime=runtime, tool_call={"id": "call_1", "name": tool_name, "args": {}}, state={}, tool=None)


def _media_message() -> ToolMessage:
    return ToolMessage(
        content_blocks=[{"type": "image", "base64": PNG_B64, "mime_type": "image/png"}],
        name="read_file",
        tool_call_id="call_1",
    )


def _capture_model_call(middleware: FilesystemMiddleware, messages: list[Any]) -> list[Any]:
    captured: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        captured.append(request)
        return ModelResponse(result=[AIMessage(content="ok")])

    middleware.wrap_model_call(ModelRequest(model=None, messages=messages, tools=[]), handler)
    return list(captured[0].messages)


def test_other_tools_are_not_offloaded(tmp_path: Path) -> None:
    middleware = FilesystemMiddleware(backend=FilesystemBackend(root_dir=tmp_path), offload_binary_reads=True)

    result = middleware.wrap_tool_call(_request("custom_tool"), lambda _: _media_message())

    assert isinstance(result, ToolMessage)
    assert result.content[0]["base64"] == PNG_B64


def test_command_results_offload_every_message(tmp_path: Path) -> None:
    middleware = FilesystemMiddleware(backend=FilesystemBackend(root_dir=tmp_path), offload_binary_reads=True)
    frames = HumanMessage(
        content=[
            {"type": "text", "text": "Frame at t=0"},
            {"type": "image", "base64": PNG_B64, "mime_type": "image/jpeg"},
        ],
        additional_kwargs={"read_file_media_result": True},
    )
    command = Command(update={"messages": [ToolMessage(content="sampled", name="read_file", tool_call_id="call_1"), frames]})

    result = middleware.wrap_tool_call(_request(), lambda _: command)

    assert isinstance(result, Command)
    stubbed = result.update["messages"][1]
    assert stubbed.content[1] == {"type": "image", "mime_type": "image/jpeg", _BLOB_REF_KEY: PNG_DIGEST}
    assert stubbed.additional_kwargs["read_file_media_result"] is True


def test_tampered_blob_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "blobs").mkdir()
    (tmp_path / "blobs" / PNG_DIGEST).write_bytes(b"not the original bytes")
    middleware = FilesystemMiddleware(backend=FilesystemBackend(root_dir=tmp_path), offload_binary_reads=True)
    stubbed = ToolMessage(
        content=[{"type": "image", "mime_type": "image/png", _BLOB_REF_KEY: PNG_DIGEST}],
        name="read_file",
        tool_call_id="call_1",
    )

    sent = _capture_model_call(middleware, [stubbed])

    assert sent[0].content == [{"type": "text", "text": _MISSING_BLOB_TEXT}]


def test_malformed_ref_never_reaches_backend(tmp_path: Path) -> None:
    middleware = FilesystemMiddleware(backend=FilesystemBackend(root_dir=tmp_path), offload_binary_reads=True)
    stubbed = ToolMessage(
        content=[{"type": "image", "mime_type": "image/png", _BLOB_REF_KEY: "../../etc/passwd"}],
        name="read_file",
        tool_call_id="call_1",
    )

    sent = _capture_model_call(middleware, [stubbed])

    assert sent[0].content == [{"type": "text", "text": _MISSING_BLOB_TEXT}]


def test_upload_failure_keeps_payload_inline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = FilesystemBackend(root_dir=tmp_path)

    def fail(_files: list[tuple[str, bytes]]) -> list[Any]:
        msg = "sandbox unavailable"
        raise RuntimeError(msg)

    monkeypatch.setattr(backend, "upload_files", fail)
    middleware = FilesystemMiddleware(backend=backend, offload_binary_reads=True)

    result = middleware.wrap_tool_call(_request(), lambda _: _media_message())

    assert isinstance(result, ToolMessage)
    assert result.content[0]["base64"] == PNG_B64


def test_cache_serves_without_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = FilesystemBackend(root_dir=tmp_path)
    middleware = FilesystemMiddleware(backend=backend, offload_binary_reads=True)
    result = middleware.wrap_tool_call(_request(), lambda _: _media_message())

    def fail(_paths: list[str]) -> list[Any]:
        msg = "cache miss"
        raise AssertionError(msg)

    monkeypatch.setattr(backend, "download_files", fail)
    sent = _capture_model_call(middleware, [result])

    assert sent[0].content[0]["base64"] == PNG_B64


def test_blob_cache_evicts_least_recently_used() -> None:
    cache = _BlobCache(max_bytes=10)
    cache.put("a", "12345")
    cache.put("b", "12345")
    assert cache.get("a") == "12345"

    cache.put("c", "12345")

    assert cache.get("b") is None
    assert cache.get("a") == "12345"
    assert cache.get("c") == "12345"
