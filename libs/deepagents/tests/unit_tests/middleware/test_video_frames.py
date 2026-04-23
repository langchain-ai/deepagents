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
