"""Tests for VideoFrameExtractionMiddleware."""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from deepagents.middleware import _ffmpeg, video_frames
from deepagents.middleware._ffmpeg import (
    ExtractedFrame,
    ExtractionFailedError,
    FFmpegMissingError,
    FileCorruptError,
    NoVideoStreamError,
)
from deepagents.middleware.video_frames import (
    VideoFrameExtractionMiddleware,
    _filename_for_video_block,
)


def _video_block(data: bytes = b"fake", mime: str = "video/mp4", filename: str = "clip.mp4") -> dict[str, Any]:
    return {
        "type": "video",
        "source_type": "base64",
        "mime_type": mime,
        "base64": base64.b64encode(data).decode(),
        "source_metadata": {"filename": filename},
    }


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _make_request(model: MagicMock, messages: list[Any]) -> MagicMock:
    req = MagicMock()
    req.model = model
    req.messages = messages

    def _override(**kw: Any) -> MagicMock:
        new = MagicMock()
        new.model = model
        new.messages = kw.get("messages", messages)
        return new

    req.override.side_effect = _override
    return req


def _model_with_provider(provider: str, model_name: str) -> MagicMock:
    model = MagicMock()
    model._get_ls_params.return_value = {"ls_provider": provider}
    model.model_dump.return_value = {"model_name": model_name}
    return model


class TestInit:
    def test_defaults(self) -> None:
        mw = VideoFrameExtractionMiddleware()
        assert mw.max_frames == 200
        assert mw.scene_threshold == 0.30
        assert mw.max_width == 512
        assert mw.jpeg_quality == 5
        assert mw.video_capable_override is None
        assert mw.cache_size == 256

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"max_frames": 0}, "max_frames"),
            ({"max_frames": 10001}, "max_frames"),
            ({"scene_threshold": -0.1}, "scene_threshold"),
            ({"scene_threshold": 1.1}, "scene_threshold"),
            ({"max_width": 32}, "max_width"),
            ({"max_width": 8192}, "max_width"),
            ({"jpeg_quality": 1}, "jpeg_quality"),
            ({"jpeg_quality": 40}, "jpeg_quality"),
            ({"cache_size": -1}, "cache_size"),
            ({"cache_size": 100001}, "cache_size"),
        ],
    )
    def test_validation_rejects_out_of_range(self, kwargs: dict[str, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            VideoFrameExtractionMiddleware(**kwargs)  # type: ignore[arg-type]


class TestFilenameForVideoBlock:
    def test_prefers_source_metadata_filename(self) -> None:
        block = {
            "type": "video",
            "source_type": "base64",
            "mime_type": "video/mp4",
            "data": "AAA=",
            "source_metadata": {"filename": "trip.mp4"},
        }
        assert _filename_for_video_block(block) == "trip.mp4"

    def test_falls_back_to_source_metadata_name(self) -> None:
        block = {
            "type": "video",
            "mime_type": "video/mp4",
            "data": "AAA=",
            "source_metadata": {"name": "clip.mov"},
        }
        assert _filename_for_video_block(block) == "clip.mov"

    def test_falls_back_to_placeholder(self) -> None:
        block = {
            "type": "video",
            "mime_type": "video/mp4",
            "data": "AAA=",
            "extras": {"placeholder": "[video 1]"},
        }
        assert _filename_for_video_block(block) == "[video 1]"

    def test_ultimate_fallback(self) -> None:
        block = {"type": "video", "mime_type": "video/mp4", "data": "AAA="}
        assert _filename_for_video_block(block) == "video"


class TestWrapModelCallPassThrough:
    def test_gemini_passes_through_unchanged(self) -> None:
        mw = VideoFrameExtractionMiddleware()
        model = _model_with_provider("google_genai", "gemini-2.0-flash")
        messages = [HumanMessage(content=[_video_block()])]
        request = _make_request(model, messages)
        handler = MagicMock(return_value="RESPONSE")

        result = mw.wrap_model_call(request, handler)

        assert result == "RESPONSE"
        handler.assert_called_once_with(request)
        request.override.assert_not_called()

    def test_no_video_blocks_passes_through(self) -> None:
        mw = VideoFrameExtractionMiddleware()
        model = _model_with_provider("anthropic", "claude-sonnet-4-6")
        messages = [HumanMessage(content=[_text_block("hello")])]
        request = _make_request(model, messages)
        handler = MagicMock(return_value="RESPONSE")

        result = mw.wrap_model_call(request, handler)

        assert result == "RESPONSE"
        handler.assert_called_once_with(request)


class TestWrapModelCallHappyPath:
    def test_single_video_replaced_with_frame_sequence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_frames = [
            ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0a", timestamp_s=0.0),
            ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0b", timestamp_s=1.0),
        ]
        monkeypatch.setattr(video_frames, "_run_extraction", lambda *_a, **_kw: (fake_frames, 5.0))
        monkeypatch.setattr(_ffmpeg, "check_ffmpeg_available", lambda: True)

        mw = VideoFrameExtractionMiddleware()
        model = _model_with_provider("anthropic", "claude-sonnet-4-6")
        messages = [HumanMessage(content=[_video_block(filename="meeting.mp4")])]
        request = _make_request(model, messages)

        captured: dict[str, Any] = {}

        def handler(req: Any) -> str:  # noqa: ANN401
            captured["messages"] = req.messages
            return "OK"

        result = mw.wrap_model_call(request, handler)  # type: ignore[arg-type]  # handler returns str as a mock ModelResponse
        assert result == "OK"

        new_messages = captured["messages"]
        assert len(new_messages) == 1
        content = new_messages[0].content
        assert isinstance(content, list)

        # preamble + (ts, img) * 2 = 5 blocks total
        assert len(content) == 5
        assert content[0]["type"] == "text"
        assert "meeting.mp4" in content[0]["text"]
        assert content[1] == {"type": "text", "text": "Frame 1 at t=0.0s"}
        assert content[2]["type"] == "image"
        assert content[2]["mime_type"] == "image/jpeg"
        assert base64.b64decode(content[2]["data"]) == b"\xff\xd8\xff\xe0a"
        assert content[2]["source_metadata"]["extracted_from"] == "meeting.mp4"
        assert content[2]["source_metadata"]["frame_index"] == 0
        assert content[2]["source_metadata"]["timestamp_s"] == 0.0
        assert content[3] == {"type": "text", "text": "Frame 2 at t=1.0s"}
        assert content[4]["type"] == "image"
        assert content[4]["source_metadata"]["frame_index"] == 1

        # Original message state was NOT mutated.
        orig_content = messages[0].content
        assert isinstance(orig_content, list)
        first_block = orig_content[0]
        assert isinstance(first_block, dict)
        assert first_block["type"] == "video"

    @pytest.mark.asyncio
    async def test_awrap_model_call_offloads_to_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """awrap_model_call must not call _transform_messages on the event-loop thread."""
        import threading

        fake_frames = [ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0a", timestamp_s=0.0)]
        event_loop_thread = threading.current_thread()
        extraction_threads: list[threading.Thread] = []

        def fake_extraction(*_a: object, **_kw: object) -> tuple[list[ExtractedFrame], float]:
            extraction_threads.append(threading.current_thread())
            return fake_frames, 5.0

        monkeypatch.setattr(video_frames, "_run_extraction", fake_extraction)

        mw = VideoFrameExtractionMiddleware()
        model = _model_with_provider("anthropic", "claude-sonnet-4-6")
        messages = [HumanMessage(content=[_video_block(filename="async.mp4")])]
        request = _make_request(model, messages)

        async def handler(req: Any) -> str:  # noqa: ANN401
            return "OK"

        result = await mw.awrap_model_call(request, handler)  # type: ignore[arg-type]
        assert result == "OK"
        assert len(extraction_threads) == 1
        assert extraction_threads[0] is not event_loop_thread

    def test_stringified_tool_message_content_is_expanded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LangGraph's ToolNode JSON-stringifies tool output when a block type isn't
        whitelisted (video isn't). The middleware must recover the list and extract frames
        so raw video bytes don't ship to the model.
        """
        import json as _json

        from langchain_core.messages import ToolMessage

        fake_frames = [ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0a", timestamp_s=0.0)]
        monkeypatch.setattr(video_frames, "_run_extraction", lambda *_a, **_kw: (fake_frames, 5.0))
        monkeypatch.setattr(_ffmpeg, "check_ffmpeg_available", lambda: True)

        stringified = _json.dumps(
            [
                {
                    "type": "video",
                    "base64": base64.b64encode(b"fakevideo").decode(),
                    "mime_type": "video/mp4",
                    "source_metadata": {"filename": "tool.mp4"},
                }
            ]
        )
        tool_msg = ToolMessage(content=stringified, tool_call_id="call_1", name="read_file")

        mw = VideoFrameExtractionMiddleware()
        model = _model_with_provider("anthropic", "claude-sonnet-4-6")
        request = _make_request(model, [tool_msg])

        captured: dict[str, Any] = {}

        def handler(req: Any) -> str:  # noqa: ANN401
            captured["messages"] = req.messages
            return "OK"

        mw.wrap_model_call(request, handler)  # type: ignore[arg-type]
        content = captured["messages"][0].content
        # The raw base64 string must be gone; only extracted frames remain.
        assert isinstance(content, list)
        assert not any(isinstance(b, str) and "base64" in b and "video" in b for b in content)
        assert any(isinstance(b, dict) and b.get("type") == "image" for b in content)

    def test_ordering_preserved_with_interleaved_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_frames = [ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0", timestamp_s=0.0)]
        monkeypatch.setattr(video_frames, "_run_extraction", lambda *_a, **_kw: (fake_frames, 5.0))

        mw = VideoFrameExtractionMiddleware()
        model = _model_with_provider("anthropic", "claude-sonnet-4-6")
        messages = [
            HumanMessage(
                content=[
                    _text_block("before"),
                    _video_block(filename="a.mp4"),
                    _text_block("middle"),
                    _video_block(filename="b.mp4"),
                    _text_block("after"),
                ]
            )
        ]
        request = _make_request(model, messages)

        captured: dict[str, Any] = {}

        def handler(req: Any) -> str:  # noqa: ANN401
            captured["messages"] = req.messages
            return "OK"

        mw.wrap_model_call(request, handler)  # type: ignore[arg-type]
        content = captured["messages"][0].content

        # Text blocks retain their relative position.
        assert content[0] == _text_block("before")
        # Next items are the expanded video a.mp4 (3 items: preamble + ts + img).
        assert content[1]["type"] == "text" and "a.mp4" in content[1]["text"]
        assert content[2] == {"type": "text", "text": "Frame 1 at t=0.0s"}
        assert content[3]["type"] == "image"
        assert content[4] == _text_block("middle")
        # Then expanded b.mp4.
        assert content[5]["type"] == "text" and "b.mp4" in content[5]["text"]
        assert content[6] == {"type": "text", "text": "Frame 1 at t=0.0s"}
        assert content[7]["type"] == "image"
        assert content[8] == _text_block("after")


class TestProviderFrameCap:
    """`max_frames` is clamped per-provider so configs that exceed a provider's
    image cap (Anthropic 100, OpenAI 1500) don't trigger API errors."""

    def test_anthropic_clamps_max_frames_to_100(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_params: dict[str, Any] = {}

        def fake(_video_bytes: bytes, _mime: str | None, params: Any) -> tuple[list[ExtractedFrame], float]:
            captured_params["params"] = params
            return [ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0", timestamp_s=0.0)], 5.0

        monkeypatch.setattr(video_frames, "_run_extraction", fake)
        mw = VideoFrameExtractionMiddleware(max_frames=200)
        model = _model_with_provider("anthropic", "claude-sonnet-4-6")
        request = _make_request(model, [HumanMessage(content=[_video_block()])])
        mw.wrap_model_call(request, lambda _r: "OK")  # type: ignore[arg-type]
        assert captured_params["params"].max_frames == 100

    def test_openai_uses_full_max_frames(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_params: dict[str, Any] = {}

        def fake(_video_bytes: bytes, _mime: str | None, params: Any) -> tuple[list[ExtractedFrame], float]:
            captured_params["params"] = params
            return [ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0", timestamp_s=0.0)], 5.0

        monkeypatch.setattr(video_frames, "_run_extraction", fake)
        mw = VideoFrameExtractionMiddleware(max_frames=200)
        model = _model_with_provider("openai", "gpt-4o")
        request = _make_request(model, [HumanMessage(content=[_video_block()])])
        mw.wrap_model_call(request, lambda _r: "OK")  # type: ignore[arg-type]
        assert captured_params["params"].max_frames == 200


class TestErrorPaths:
    def _run_with_failure(self, monkeypatch: pytest.MonkeyPatch, failure: Exception) -> list[Any]:
        def fail(*_a: object, **_kw: object) -> list[Any]:
            raise failure

        monkeypatch.setattr(video_frames, "_run_extraction", fail)

        mw = VideoFrameExtractionMiddleware()
        model = _model_with_provider("anthropic", "claude-sonnet-4-6")
        messages = [HumanMessage(content=[_video_block(filename="bad.mp4")])]
        request = _make_request(model, messages)

        captured: dict[str, Any] = {}

        def handler(req: Any) -> str:  # noqa: ANN401
            captured["messages"] = req.messages
            return "OK"

        mw.wrap_model_call(request, handler)  # type: ignore[arg-type]
        return captured["messages"][0].content

    def test_ffmpeg_missing_produces_error_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        content = self._run_with_failure(monkeypatch, FFmpegMissingError("missing"))
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert "ffmpeg is not installed" in content[0]["text"]
        assert "'bad.mp4'" in content[0]["text"]

    def test_no_video_stream_produces_error_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        content = self._run_with_failure(monkeypatch, NoVideoStreamError("no stream"))
        assert "no video stream" in content[0]["text"]

    def test_file_corrupt_produces_error_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        content = self._run_with_failure(monkeypatch, FileCorruptError("corrupt"))
        assert "corrupt or unreadable" in content[0]["text"]

    def test_extraction_failed_produces_error_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        content = self._run_with_failure(monkeypatch, ExtractionFailedError("kaboom"))
        assert "frame extraction failed" in content[0]["text"]


class TestOverride:
    def test_override_false_forces_extraction_on_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_frames = [ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0", timestamp_s=0.0)]
        monkeypatch.setattr(video_frames, "_run_extraction", lambda *_a, **_kw: (fake_frames, 5.0))

        mw = VideoFrameExtractionMiddleware(video_capable_override=False)
        model = _model_with_provider("google_genai", "gemini-2.0-flash")
        messages = [HumanMessage(content=[_video_block()])]
        request = _make_request(model, messages)
        captured: dict[str, Any] = {}

        def handler(req: Any) -> str:  # noqa: ANN401
            captured["messages"] = req.messages
            return "OK"

        mw.wrap_model_call(request, handler)  # type: ignore[arg-type]
        content = captured["messages"][0].content
        assert any(b.get("type") == "image" for b in content)


class TestCache:
    def test_repeat_video_hits_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = {"n": 0}

        def fake(*_a: object, **_kw: object) -> tuple[list[ExtractedFrame], float]:
            call_count["n"] += 1
            return [ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0", timestamp_s=0.0)], 5.0

        monkeypatch.setattr(video_frames, "_run_extraction", fake)

        mw = VideoFrameExtractionMiddleware()
        model = _model_with_provider("anthropic", "claude-sonnet-4-6")
        video = _video_block(data=b"same-bytes-every-time", filename="repeat.mp4")

        for _ in range(3):
            messages = [HumanMessage(content=[video])]
            request = _make_request(model, messages)
            mw.wrap_model_call(request, lambda _r: "OK")

        assert call_count["n"] == 1

    def test_different_params_skip_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = {"n": 0}

        def fake(*_a: object, **_kw: object) -> tuple[list[ExtractedFrame], float]:
            call_count["n"] += 1
            return [ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0", timestamp_s=0.0)], 5.0

        monkeypatch.setattr(video_frames, "_run_extraction", fake)

        model = _model_with_provider("anthropic", "claude-sonnet-4-6")
        video = _video_block(data=b"bytes", filename="a.mp4")

        mw1 = VideoFrameExtractionMiddleware(scene_threshold=0.3)
        mw2 = VideoFrameExtractionMiddleware(scene_threshold=0.5)

        request = _make_request(model, [HumanMessage(content=[video])])
        mw1.wrap_model_call(request, lambda _r: "OK")
        request = _make_request(model, [HumanMessage(content=[video])])
        mw2.wrap_model_call(request, lambda _r: "OK")

        # Two different middlewares with different params -> two extractions.
        assert call_count["n"] == 2

    def test_lru_eviction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = {"n": 0}

        def fake(*_a: object, **_kw: object) -> tuple[list[ExtractedFrame], float]:
            call_count["n"] += 1
            return [ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0", timestamp_s=0.0)], 5.0

        monkeypatch.setattr(video_frames, "_run_extraction", fake)

        mw = VideoFrameExtractionMiddleware(cache_size=2)
        model = _model_with_provider("anthropic", "claude-sonnet-4-6")

        def run(data: bytes) -> None:
            mw.wrap_model_call(
                _make_request(model, [HumanMessage(content=[_video_block(data=data)])]),
                lambda _r: "OK",
            )

        run(b"a")
        run(b"b")
        run(b"c")  # Evicts 'a'.
        run(b"a")  # Re-extract (evicted).
        assert call_count["n"] == 4

    def test_cache_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = {"n": 0}

        def fake(*_a: object, **_kw: object) -> tuple[list[ExtractedFrame], float]:
            call_count["n"] += 1
            return [ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0", timestamp_s=0.0)], 5.0

        monkeypatch.setattr(video_frames, "_run_extraction", fake)

        mw = VideoFrameExtractionMiddleware(cache_size=0)
        model = _model_with_provider("anthropic", "claude-sonnet-4-6")
        video = _video_block(data=b"same", filename="a.mp4")

        for _ in range(3):
            mw.wrap_model_call(
                _make_request(model, [HumanMessage(content=[video])]),
                lambda _r: "OK",
            )
        assert call_count["n"] == 3


class TestAwrapModelCall:
    async def test_async_pass_through_on_gemini(self) -> None:
        mw = VideoFrameExtractionMiddleware()
        model = _model_with_provider("google_genai", "gemini-2.0-flash")
        messages = [HumanMessage(content=[_video_block()])]
        request = _make_request(model, messages)

        async def handler(_req: object) -> str:
            return "ASYNC"

        result = await mw.awrap_model_call(request, handler)
        assert result == "ASYNC"

    async def test_async_transforms_on_claude(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_frames = [ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0", timestamp_s=0.0)]
        monkeypatch.setattr(video_frames, "_run_extraction", lambda *_a, **_kw: (fake_frames, 5.0))

        mw = VideoFrameExtractionMiddleware()
        model = _model_with_provider("anthropic", "claude-sonnet-4-6")
        messages = [HumanMessage(content=[_video_block(filename="x.mp4")])]
        request = _make_request(model, messages)

        captured: dict[str, Any] = {}

        async def handler(req: Any) -> str:  # noqa: ANN401
            captured["messages"] = req.messages
            return "OK"

        result = await mw.awrap_model_call(request, handler)
        assert result == "OK"
        content = captured["messages"][0].content
        assert any(b.get("type") == "image" for b in content)


def test_exported_from_package() -> None:
    from deepagents.middleware import VideoFrameExtractionMiddleware as Exported  # noqa: PLC0415

    assert Exported is VideoFrameExtractionMiddleware


class TestRealDurationInPreamble:
    """Regression guard for Issue 2: preamble must report real video duration.

    The duration must not be derived from frame timestamps, which can be
    clustered when scene detection fires.
    """

    def test_preamble_uses_real_duration_not_frame_gap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clustered frames (tiny timestamp window) must not corrupt the preamble duration."""
        # All three frames are within 0.1s of each other, but the video is 60s long.
        fake_frames = [
            ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0a", timestamp_s=0.00),
            ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0b", timestamp_s=0.05),
            ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0c", timestamp_s=0.10),
        ]
        real_duration = 60.0
        monkeypatch.setattr(video_frames, "_run_extraction", lambda *_a, **_kw: (fake_frames, real_duration))

        mw = VideoFrameExtractionMiddleware()
        model = _model_with_provider("anthropic", "claude-sonnet-4-6")
        messages = [HumanMessage(content=[_video_block(filename="long.mp4")])]
        request = _make_request(model, messages)

        captured: dict[str, Any] = {}

        def handler(req: Any) -> str:  # noqa: ANN401
            captured["messages"] = req.messages
            return "OK"

        mw.wrap_model_call(request, handler)  # type: ignore[arg-type]
        content = captured["messages"][0].content
        preamble_text = content[0]["text"]

        # Must report the real 60s duration, not ~0.1s derived from frame timestamps.
        assert "duration 60.0s" in preamble_text, (
            f"Expected 'duration 60.0s' in preamble, got: {preamble_text!r}"
        )

    def test_preamble_reports_duration_from_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy-path: the preamble duration equals the duration returned by extraction."""
        fake_frames = [
            ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0a", timestamp_s=0.0),
            ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0b", timestamp_s=1.0),
        ]
        monkeypatch.setattr(video_frames, "_run_extraction", lambda *_a, **_kw: (fake_frames, 5.0))

        mw = VideoFrameExtractionMiddleware()
        model = _model_with_provider("anthropic", "claude-sonnet-4-6")
        messages = [HumanMessage(content=[_video_block(filename="clip.mp4")])]
        request = _make_request(model, messages)

        captured: dict[str, Any] = {}

        def handler(req: Any) -> str:  # noqa: ANN401
            captured["messages"] = req.messages
            return "OK"

        mw.wrap_model_call(request, handler)  # type: ignore[arg-type]
        content = captured["messages"][0].content
        preamble_text = content[0]["text"]

        assert "duration 5.0s" in preamble_text, (
            f"Expected 'duration 5.0s' in preamble, got: {preamble_text!r}"
        )
