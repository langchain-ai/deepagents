"""Tests for VideoFrameExtractionMiddleware."""

from __future__ import annotations

import pytest

from deepagents.middleware.video_frames import (
    VideoFrameExtractionMiddleware,
    _filename_for_video_block,
)


class TestInit:
    def test_defaults(self) -> None:
        mw = VideoFrameExtractionMiddleware()
        assert mw.max_frames == 95
        assert mw.scene_threshold == 0.30
        assert mw.max_width == 1024
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


import base64
from typing import Any, Callable
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage


def _video_block(data: bytes = b"fake", mime: str = "video/mp4", filename: str = "clip.mp4") -> dict[str, Any]:
    return {
        "type": "video",
        "source_type": "base64",
        "mime_type": mime,
        "data": base64.b64encode(data).decode(),
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
    def test_single_video_replaced_with_frame_sequence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deepagents.middleware import _ffmpeg, video_frames
        from deepagents.middleware._ffmpeg import ExtractedFrame

        fake_frames = [
            ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0a", timestamp_s=0.0),
            ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0b", timestamp_s=1.0),
        ]
        monkeypatch.setattr(
            video_frames, "_run_extraction", lambda *_a, **_kw: fake_frames
        )
        monkeypatch.setattr(_ffmpeg, "check_ffmpeg_available", lambda: True)

        mw = VideoFrameExtractionMiddleware()
        model = _model_with_provider("anthropic", "claude-sonnet-4-6")
        messages = [HumanMessage(content=[_video_block(filename="meeting.mp4")])]
        request = _make_request(model, messages)

        captured: dict[str, Any] = {}
        def handler(req: Any) -> str:
            captured["messages"] = req.messages
            return "OK"

        result = mw.wrap_model_call(request, handler)
        assert result == "OK"

        new_messages = captured["messages"]
        assert len(new_messages) == 1
        content = new_messages[0].content
        assert isinstance(content, list)

        # Expect: [preamble, ts1, img1, ts2, img2]
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
        assert messages[0].content[0]["type"] == "video"

    def test_ordering_preserved_with_interleaved_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deepagents.middleware import video_frames
        from deepagents.middleware._ffmpeg import ExtractedFrame

        fake_frames = [ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff\xe0", timestamp_s=0.0)]
        monkeypatch.setattr(
            video_frames, "_run_extraction", lambda *_a, **_kw: fake_frames
        )

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
        def handler(req: Any) -> str:
            captured["messages"] = req.messages
            return "OK"

        mw.wrap_model_call(request, handler)
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
