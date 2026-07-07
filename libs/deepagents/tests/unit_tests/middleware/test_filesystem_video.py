import base64
from typing import cast

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langgraph.types import Command

import deepagents.middleware.filesystem as filesystem_middleware
from deepagents.backends import StateBackend
from deepagents.backends.protocol import ReadResult
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemState


@pytest.fixture(autouse=True)
def _enable_video_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise frame-extraction routing independently of the local test env."""
    monkeypatch.setattr(filesystem_middleware, "video_dependencies_available", lambda: True)


def test_read_file_video_routes_to_frame_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Video reads route through the frame extractor, not generic base64 media."""
    raw_bytes = b"\x00\x01\x02 fake video bytes"
    sentinel = [
        {"type": "text", "text": "Frame at t=00:00:00.000"},
        {"type": "image", "base64": "AAAA", "mime_type": "image/jpeg"},
        {"type": "text", "text": "Frame at t=00:00:02.000"},
        {"type": "image", "base64": "BBBB", "mime_type": "image/jpeg"},
    ]
    calls: list[dict[str, float | int]] = []

    def fake_extract(
        content: bytes,
        *,
        offset_seconds: float,
        duration_seconds: float,
        sampling_rate: float,
    ) -> list[dict[str, str]]:
        calls.append(
            {
                "len": len(content),
                "offset_seconds": offset_seconds,
                "duration_seconds": duration_seconds,
                "sampling_rate": sampling_rate,
            }
        )
        return list(sentinel)

    monkeypatch.setattr(filesystem_middleware, "extract_video_frames", fake_extract)

    middleware = FilesystemMiddleware(backend=_video_backend(raw_bytes))
    state = FilesystemState(messages=[], files={})
    runtime = _build_runtime(state, "video-read-1")
    read_file_tool = next(t for t in middleware.tools if t.name == "read_file")

    result = read_file_tool.invoke(
        {
            "file_path": "/clips/intro.mkv",
            "offset": 0,
            "limit": 30,
            "runtime": runtime,
        }
    )

    assert isinstance(result, Command)
    assert isinstance(result.update, dict)
    messages = result.update["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 2
    tool_message, media_message = messages
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.status == "success"
    assert tool_message.content == "Read video /clips/intro.mkv: sampled 2 frames. The sampled frames are attached in the following message."
    assert tool_message.additional_kwargs["read_file_path"] == "/clips/intro.mkv"
    assert tool_message.additional_kwargs["read_file_frame_count"] == 2
    assert isinstance(media_message, HumanMessage)
    assert media_message.additional_kwargs["read_file_media_result"] is True
    assert media_message.additional_kwargs["read_file_path"] == "/clips/intro.mkv"
    assert media_message.additional_kwargs["read_file_tool_call_id"] == "video-read-1"
    assert media_message.content == [
        {"type": "text", "text": "Reading first 30s of /clips/intro.mkv at 0.5 fps."},
        *sentinel,
    ]
    assert b"\x00\x01\x02" not in str(media_message.content).encode()

    openai_messages = convert_to_openai_messages(
        [
            HumanMessage(content="read video"),
            AIMessage(content="", tool_calls=[{"id": "video-read-1", "name": "read_file", "args": {"file_path": "/clips/intro.mkv"}}]),
            *messages,
        ]
    )
    openai_tool_message = next(message for message in openai_messages if message["role"] == "tool")
    assert isinstance(openai_tool_message["content"], str)
    assert all(
        block.get("type") != "video"
        for message in openai_messages
        for block in (message["content"] if isinstance(message.get("content"), list) else [])
    )

    assert calls == [
        {
            "len": len(raw_bytes),
            "offset_seconds": 0,
            "duration_seconds": 30,
            "sampling_rate": 0.5,
        }
    ]


@pytest.mark.parametrize(
    ("tool_input", "expected"),
    [
        (
            {"offset": 12, "limit": 90},
            {"offset_seconds": 12.0, "duration_seconds": 90.0, "sampling_rate": 0.5},
        ),
        (
            {},
            {"offset_seconds": 0.0, "duration_seconds": 100.0, "sampling_rate": 0.5},
        ),
    ],
    ids=["explicit-window", "default-window"],
)
def test_read_file_video_window_forwards_seconds(
    monkeypatch: pytest.MonkeyPatch,
    tool_input: dict[str, int],
    expected: dict[str, float],
) -> None:
    """`offset`/`limit` become seconds, with the default window preserved."""
    captured: dict[str, float] = {}

    def fake_extract(
        _content: bytes,
        *,
        offset_seconds: float,
        duration_seconds: float,
        sampling_rate: float,
    ) -> list[dict[str, str]]:
        captured.update(offset_seconds=offset_seconds, duration_seconds=duration_seconds, sampling_rate=sampling_rate)
        return [{"type": "text", "text": "ok"}]

    monkeypatch.setattr(filesystem_middleware, "extract_video_frames", fake_extract)
    middleware = FilesystemMiddleware(backend=_video_backend())
    state = FilesystemState(messages=[], files={})
    runtime = _build_runtime(state, "video-read-window")
    read_file_tool = next(t for t in middleware.tools if t.name == "read_file")

    read_file_tool.invoke({"file_path": "/c.mp4", "runtime": runtime, **tool_input})

    assert captured == expected


def test_read_file_video_media_result_ordered_after_parallel_tool_results() -> None:
    """Video frame attachments do not split a provider-required tool result batch."""
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_video", "name": "read_file", "args": {"file_path": "/c.mp4"}},
            {"id": "call_ls", "name": "ls", "args": {"path": "/"}},
        ],
    )
    video_tool = ToolMessage(content="sampled frames", name="read_file", tool_call_id="call_video")
    video_media = HumanMessage(
        content=[
            {"type": "text", "text": "Reading first 100s of /c.mp4 at 0.5 fps."},
            {"type": "image", "base64": "AAAA", "mime_type": "image/jpeg"},
        ],
        additional_kwargs={"read_file_media_result": True},
    )
    ls_tool = ToolMessage(content="[]", name="ls", tool_call_id="call_ls")

    user_message = HumanMessage(content="read and list")
    reordered = filesystem_middleware._move_media_results_after_tool_results([user_message, ai_message, video_tool, video_media, ls_tool])

    assert reordered == [user_message, ai_message, video_tool, ls_tool, video_media]


def test_read_file_video_extraction_error_surfaces_as_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """PyAV failures and missing-dep errors render as a tool error, not an exception."""

    def fake_extract(*_args: object, **_kwargs: object) -> list[dict[str, str]]:
        msg = "av not installed"
        raise filesystem_middleware.VideoExtractionError(msg)

    monkeypatch.setattr(filesystem_middleware, "extract_video_frames", fake_extract)
    middleware = FilesystemMiddleware(backend=_video_backend(b"corrupt"))
    state = FilesystemState(messages=[], files={})
    runtime = _build_runtime(state, "video-read-err")
    read_file_tool = next(t for t in middleware.tools if t.name == "read_file")

    result = read_file_tool.invoke({"file_path": "/c.mp4", "runtime": runtime})

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "av not installed" in result.content


def test_read_file_without_video_extra_keeps_generic_video_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing optional deps preserve the previous generic video read behavior."""
    monkeypatch.setattr(filesystem_middleware, "video_dependencies_available", lambda: False)
    monkeypatch.setattr(
        filesystem_middleware,
        "extract_video_frames",
        lambda *_args, **_kwargs: pytest.fail("video extraction should be gated"),
    )
    middleware = FilesystemMiddleware(backend=_video_backend(b"video bytes"))
    state = FilesystemState(messages=[], files={})
    runtime = _build_runtime(state, "video-read-no-extra")
    read_file_tool = next(t for t in middleware.tools if t.name == "read_file")

    result = read_file_tool.invoke({"file_path": "/c.mp4", "runtime": runtime})

    assert isinstance(result, ToolMessage)
    assert result.status == "success"
    assert isinstance(result.content, list)
    assert result.content[0]["type"] == "video"
    assert result.content[0]["base64"] == base64.b64encode(b"video bytes").decode("ascii")


def test_read_file_without_video_extra_keeps_mkv_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    """The new `.mkv` classification only applies with the video extra."""
    monkeypatch.setattr(filesystem_middleware, "video_dependencies_available", lambda: False)
    middleware = FilesystemMiddleware(backend=_video_backend(b"mkv bytes"))
    state = FilesystemState(messages=[], files={})
    runtime = _build_runtime(state, "mkv-read-no-extra")
    read_file_tool = next(t for t in middleware.tools if t.name == "read_file")

    result = read_file_tool.invoke({"file_path": "/c.mkv", "runtime": runtime})

    assert isinstance(result, ToolMessage)
    assert result.status == "success"
    assert isinstance(result.content, list)
    assert result.content[0]["type"] == "file"
    assert result.content[0]["base64"] == base64.b64encode(b"mkv bytes").decode("ascii")


def test_read_file_without_video_extra_uses_text_schema_and_description(monkeypatch: pytest.MonkeyPatch) -> None:
    """Video-specific read guidance is hidden when frame extraction is unavailable."""
    monkeypatch.setattr(filesystem_middleware, "video_dependencies_available", lambda: False)
    middleware = FilesystemMiddleware(backend=StateBackend())
    read_file_tool = next(t for t in middleware.tools if t.name == "read_file")

    assert "For videos" not in read_file_tool.description
    schema = read_file_tool.args_schema.model_json_schema()
    assert "For videos" not in schema["properties"]["offset"]["description"]
    assert "For videos" not in schema["properties"]["limit"]["description"]


def test_read_file_with_video_extra_uses_video_schema_and_description() -> None:
    """Video-specific read guidance is exposed when frame extraction is available."""
    # The autouse `_enable_video_extra` fixture forces the extra on for this test.
    middleware = FilesystemMiddleware(backend=StateBackend())
    read_file_tool = next(t for t in middleware.tools if t.name == "read_file")

    assert "For videos" in read_file_tool.description
    schema = read_file_tool.args_schema.model_json_schema()
    assert "For videos" in schema["properties"]["offset"]["description"]
    assert "For videos" in schema["properties"]["limit"]["description"]


def test_read_file_video_unexpected_value_error_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected extractor bugs should remain visible to operators."""

    def fake_extract(*_args: object, **_kwargs: object) -> list[dict[str, str]]:
        msg = "unexpected formatter failure"
        raise ValueError(msg)

    monkeypatch.setattr(filesystem_middleware, "extract_video_frames", fake_extract)
    middleware = FilesystemMiddleware(backend=_video_backend())
    state = FilesystemState(messages=[], files={})
    runtime = _build_runtime(state, "video-read-unexpected-value-error")
    read_file_tool = next(t for t in middleware.tools if t.name == "read_file")

    with pytest.raises(ValueError, match="unexpected formatter failure"):
        read_file_tool.invoke({"file_path": "/c.mp4", "runtime": runtime})


def test_read_file_video_non_positive_limit_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-positive `limit` is rejected as a tool error, not silently clamped."""
    called = {"count": 0}

    def fake_extract(*_args: object, **_kwargs: object) -> list[dict[str, str]]:
        called["count"] += 1
        return [{"type": "text", "text": "ok"}]

    monkeypatch.setattr(filesystem_middleware, "extract_video_frames", fake_extract)
    middleware = FilesystemMiddleware(backend=_video_backend())
    state = FilesystemState(messages=[], files={})
    runtime = _build_runtime(state, "video-read-bad-limit")
    read_file_tool = next(t for t in middleware.tools if t.name == "read_file")

    result = read_file_tool.invoke({"file_path": "/c.mp4", "limit": 0, "runtime": runtime})

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "limit must be > 0" in result.content
    assert called["count"] == 0


def test_read_file_video_payload_size_cap_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """State/store/custom video payloads are capped after base64 decoding."""
    called = {"count": 0}

    def fake_extract(*_args: object, **_kwargs: object) -> list[dict[str, str]]:
        called["count"] += 1
        return [{"type": "text", "text": "ok"}]

    monkeypatch.setattr(filesystem_middleware, "MAX_VIDEO_INPUT_BYTES", 3)
    monkeypatch.setattr(filesystem_middleware, "extract_video_frames", fake_extract)
    middleware = FilesystemMiddleware(backend=_video_backend(b"abcd"))
    state = FilesystemState(messages=[], files={})
    runtime = _build_runtime(state, "video-read-big-payload")
    read_file_tool = next(t for t in middleware.tools if t.name == "read_file")

    result = read_file_tool.invoke({"file_path": "/c.mp4", "runtime": runtime})

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "video payload exceeds maximum input size of 3 bytes" in result.content
    assert called["count"] == 0


def _build_runtime(state: FilesystemState, tool_call_id: str) -> ToolRuntime[None, FilesystemState]:
    """Build a `ToolRuntime` for video `read_file` tests."""
    return ToolRuntime(
        state=state,
        context=None,
        tool_call_id=tool_call_id,
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def _video_backend(raw: bytes = b"raw") -> StateBackend:
    """Backend stub that returns `raw` base64-encoded for any video read."""
    payload = base64.b64encode(raw).decode("ascii")

    class VideoBackend(StateBackend):
        def read(self, file_path: str, offset: int = 0, limit: int = 100) -> ReadResult:
            return ReadResult(file_data={"content": payload, "encoding": "base64"})

    return VideoBackend()


def test_wrap_model_call_reorders_media_after_parallel_tool_results() -> None:
    """`wrap_model_call` moves video media behind the full tool-result batch.

    With a parallel batch (video read_file + ls), the provider requires all
    `ToolMessage`s before any non-tool message. Deleting the reorder call
    from `wrap_model_call` would fail this test because the media
    `HumanMessage` would arrive between the two tool results.
    """
    mw = FilesystemMiddleware(backend=StateBackend())

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_video", "name": "read_file", "args": {"file_path": "/c.mp4"}},
            {"id": "call_ls", "name": "ls", "args": {"path": "/"}},
        ],
    )
    video_tool = ToolMessage(content="sampled frames", name="read_file", tool_call_id="call_video")
    video_media = HumanMessage(
        content=[{"type": "text", "text": "Reading first 100s of /c.mp4 at 0.5 fps."}],
        additional_kwargs={"read_file_media_result": True},
    )
    ls_tool = ToolMessage(content="[]", name="ls", tool_call_id="call_ls")
    user_msg = HumanMessage(content="read and list")

    captured: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        captured.append(request)
        return ModelResponse(result=[AIMessage(content="ok")])

    request = ModelRequest(
        model=None,
        messages=[user_msg, ai_msg, video_tool, video_media, ls_tool],
        tools=[],
    )

    mw.wrap_model_call(request, handler)

    assert len(captured) == 1
    reordered = captured[0].messages
    # All ToolMessages must come before the media HumanMessage.
    tool_indices = [i for i, m in enumerate(reordered) if isinstance(m, ToolMessage)]
    media_index = next(
        i for i, m in enumerate(reordered) if isinstance(m, HumanMessage) and m.additional_kwargs.get("read_file_media_result") is True
    )
    assert all(i < media_index for i in tool_indices)
    # Specifically: video_tool, ls_tool, then video_media.
    assert reordered[-3] == video_tool
    assert reordered[-2] == ls_tool
    assert reordered[-1] == video_media


async def test_awrap_model_call_reorders_media_after_parallel_tool_results() -> None:
    """`awrap_model_call` moves video media behind the full tool-result batch (async path)."""
    mw = FilesystemMiddleware(backend=StateBackend())

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_video", "name": "read_file", "args": {"file_path": "/c.mp4"}},
            {"id": "call_ls", "name": "ls", "args": {"path": "/"}},
        ],
    )
    video_tool = ToolMessage(content="sampled frames", name="read_file", tool_call_id="call_video")
    video_media = HumanMessage(
        content=[{"type": "text", "text": "Reading first 100s of /c.mp4 at 0.5 fps."}],
        additional_kwargs={"read_file_media_result": True},
    )
    ls_tool = ToolMessage(content="[]", name="ls", tool_call_id="call_ls")
    user_msg = HumanMessage(content="read and list")

    captured: list[ModelRequest] = []

    async def handler(request: ModelRequest) -> ModelResponse:
        captured.append(request)
        return ModelResponse(result=[AIMessage(content="ok")])

    request = ModelRequest(
        model=None,
        messages=[user_msg, ai_msg, video_tool, video_media, ls_tool],
        tools=[],
    )

    await mw.awrap_model_call(request, handler)

    assert len(captured) == 1
    reordered = captured[0].messages
    tool_indices = [i for i, m in enumerate(reordered) if isinstance(m, ToolMessage)]
    media_index = next(
        i for i, m in enumerate(reordered) if isinstance(m, HumanMessage) and m.additional_kwargs.get("read_file_media_result") is True
    )
    assert all(i < media_index for i in tool_indices)
    assert reordered[-3] == video_tool
    assert reordered[-2] == ls_tool
    assert reordered[-1] == video_media


# ---------------------------------------------------------------------------
# Async read_file video path (issue #5b)
# ---------------------------------------------------------------------------


async def test_async_read_file_video_routes_to_frame_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The async `read_file` coroutine produces the same Command as the sync path."""
    sentinel = [
        {"type": "text", "text": "Frame at t=00:00:00.000"},
        {"type": "image", "base64": "AAAA", "mime_type": "image/jpeg"},
    ]
    calls: list[dict[str, float | int]] = []

    def fake_extract(
        _content: bytes,
        *,
        offset_seconds: float,
        duration_seconds: float,
        sampling_rate: float,
    ) -> list[dict[str, str]]:
        calls.append(
            {
                "offset_seconds": offset_seconds,
                "duration_seconds": duration_seconds,
                "sampling_rate": sampling_rate,
            }
        )
        return list(sentinel)

    monkeypatch.setattr(filesystem_middleware, "extract_video_frames", fake_extract)

    middleware = FilesystemMiddleware(backend=_video_backend(b"fake video"))
    state = FilesystemState(messages=[], files={})
    runtime = _build_runtime(state, "video-async-1")
    read_file_tool = next(t for t in middleware.tools if t.name == "read_file")

    result = await read_file_tool.ainvoke(
        {
            "file_path": "/clips/intro.mkv",
            "offset": 0,
            "limit": 30,
            "runtime": runtime,
        }
    )

    assert isinstance(result, Command)
    messages = result.update["messages"]
    assert len(messages) == 2
    tool_message, media_message = messages
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.status == "success"
    assert tool_message.additional_kwargs["read_file_frame_count"] == 1
    assert isinstance(media_message, HumanMessage)
    assert media_message.additional_kwargs["read_file_media_result"] is True
    assert calls == [{"offset_seconds": 0.0, "duration_seconds": 30.0, "sampling_rate": 0.5}]


def test_read_file_video_base64_decode_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid base64 content surfaces as a tool error, not an exception."""
    called = {"count": 0}

    def fake_extract(*_args: object, **_kwargs: object) -> list[dict[str, str]]:
        called["count"] += 1
        return [{"type": "text", "text": "ok"}]

    monkeypatch.setattr(filesystem_middleware, "extract_video_frames", fake_extract)

    class BadBase64Backend(StateBackend):
        def read(self, file_path: str, offset: int = 0, limit: int = 100) -> ReadResult:
            # Not valid base64 — contains characters outside the base64 alphabet.
            return ReadResult(file_data={"content": "!!!not-base64!!!", "encoding": "base64"})

    middleware = FilesystemMiddleware(backend=BadBase64Backend())
    state = FilesystemState(messages=[], files={})
    runtime = _build_runtime(state, "video-bad-b64")
    read_file_tool = next(t for t in middleware.tools if t.name == "read_file")

    result = read_file_tool.invoke({"file_path": "/c.mp4", "runtime": runtime})

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "not valid base64" in result.content
    assert called["count"] == 0


def test_read_file_video_offset_header_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero offset produces the `[x.xxxs, y.yyys)` header format."""
    monkeypatch.setattr(
        filesystem_middleware,
        "extract_video_frames",
        lambda *_a, **_k: [{"type": "text", "text": "ok"}],
    )
    middleware = FilesystemMiddleware(backend=_video_backend())
    state = FilesystemState(messages=[], files={})
    runtime = _build_runtime(state, "video-offset")
    read_file_tool = next(t for t in middleware.tools if t.name == "read_file")

    result = cast("Command", read_file_tool.invoke({"file_path": "/c.mp4", "offset": 12, "limit": 90, "runtime": runtime}))
    messages = result.update["messages"]
    media_message = messages[1]
    assert isinstance(media_message, HumanMessage)
    header_block = media_message.content[0]
    assert isinstance(header_block, dict)
    text = header_block["text"]
    assert text.startswith("Reading [12.000s, 102.000s)")
    assert "0.5 fps" in text


def test_read_file_video_singular_frame_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single frame uses '1 frame' (singular), not '1 frames'."""
    sentinel = [
        {"type": "text", "text": "Frame at t=00:00:00.000"},
        {"type": "image", "base64": "AAAA", "mime_type": "image/jpeg"},
    ]
    monkeypatch.setattr(
        filesystem_middleware,
        "extract_video_frames",
        lambda *_a, **_k: list(sentinel),
    )
    middleware = FilesystemMiddleware(backend=_video_backend())
    state = FilesystemState(messages=[], files={})
    runtime = _build_runtime(state, "video-singular")
    read_file_tool = next(t for t in middleware.tools if t.name == "read_file")

    result = cast("Command", read_file_tool.invoke({"file_path": "/c.mp4", "runtime": runtime}))
    tool_message = result.update["messages"][0]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Read video /c.mp4: sampled 1 frame. The sampled frames are attached in the following message."


def test_read_file_video_truncation_hint_appended_on_byte_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the byte budget truncates output, a coverage hint is appended."""
    # Two frames that together exceed a tiny byte budget.
    sentinel = [
        {"type": "text", "text": "Frame at t=00:00:00.000"},
        {"type": "image", "base64": "A" * 100, "mime_type": "image/jpeg"},
        {"type": "text", "text": "Frame at t=00:00:02.000"},
        {"type": "image", "base64": "B" * 100, "mime_type": "image/jpeg"},
        {
            "type": "text",
            "text": (
                "Coverage truncated at t=00:00:00.000: the output or frame cap was reached "
                "before the full window was decoded. Continue from "
                "offset=0.000 to see the remaining frames."
            ),
        },
    ]
    monkeypatch.setattr(
        filesystem_middleware,
        "extract_video_frames",
        lambda *_a, **_k: list(sentinel),
    )
    middleware = FilesystemMiddleware(backend=_video_backend())
    state = FilesystemState(messages=[], files={})
    runtime = _build_runtime(state, "video-trunc")
    read_file_tool = next(t for t in middleware.tools if t.name == "read_file")

    result = cast("Command", read_file_tool.invoke({"file_path": "/c.mp4", "runtime": runtime}))
    media_message = result.update["messages"][1]
    assert isinstance(media_message, HumanMessage)
    blocks = media_message.content
    # The last block should be a truncation hint.
    last_block = blocks[-1]
    assert isinstance(last_block, dict)
    assert last_block["type"] == "text"
    assert "Coverage truncated" in last_block["text"]
    assert "Continue from offset=" in last_block["text"]
