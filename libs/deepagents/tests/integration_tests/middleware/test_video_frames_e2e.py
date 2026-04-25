"""End-to-end: feed a generated video into VideoFrameExtractionMiddleware.

Uses a fake BaseChatModel so no API key is required. Verifies that, for a
Claude-ish model, video blocks are replaced by image blocks before reaching
the model.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path  # noqa: TC003
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from deepagents.middleware.video_frames import VideoFrameExtractionMiddleware

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
        reason="ffmpeg/ffprobe not available",
    ),
]


@pytest.fixture
def sample_video_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "e2e.mp4"
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=320x240:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path.read_bytes()


def test_e2e_video_replaced_with_images_for_claude(sample_video_bytes: bytes) -> None:
    block = {
        "type": "video",
        "source_type": "base64",
        "mime_type": "video/mp4",
        "base64": base64.b64encode(sample_video_bytes).decode(),
        "source_metadata": {"filename": "e2e.mp4"},
    }

    model = MagicMock()
    model._get_ls_params.return_value = {"ls_provider": "anthropic"}
    model.model_dump.return_value = {"model_name": "claude-sonnet-4-6"}

    request = MagicMock()
    request.model = model
    request.messages = [HumanMessage(content=[block])]

    def _override(**kw: Any) -> MagicMock:
        new = MagicMock()
        new.model = model
        new.messages = kw.get("messages", request.messages)
        return new

    request.override.side_effect = _override

    captured: dict[str, Any] = {}

    def handler(req: Any) -> str:  # noqa: ANN401
        captured["messages"] = req.messages
        return "OK"

    mw = VideoFrameExtractionMiddleware()
    result = mw.wrap_model_call(request, handler)  # type: ignore[arg-type]  # handler returns str as mock ModelResponse
    assert result == "OK"

    content = captured["messages"][0].content
    image_blocks = [b for b in content if b.get("type") == "image"]
    text_blocks = [b for b in content if b.get("type") == "text"]
    assert len(image_blocks) >= 3  # 3s @ 1fps baseline + possible scene frames
    # First text block is the preamble; others are "Frame N at t=..." labels.
    assert "e2e.mp4" in text_blocks[0]["text"]
    for i, tb in enumerate(text_blocks[1:]):
        assert tb["text"].startswith(f"Frame {i + 1} at t=")
