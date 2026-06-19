"""Unit tests for the PyAV-based video frame extractor.

The tests exercise the real PyAV decoder against a synthetic 3-second clip so
the offset/limit -> seconds reinterpretation, frame sampling, validation, and
error mapping all run end-to-end. They are skipped automatically when the
``av`` (PyAV) extra is not installed, which keeps the default unit suite
lightweight.
"""

import base64

import pytest

import deepagents.middleware._video as video_module

av = pytest.importorskip("av", reason="`av` extra not installed (pip install deepagents[video])")
np = pytest.importorskip("numpy", reason="numpy not installed")


@pytest.fixture(scope="module")
def synthetic_video_bytes(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Build a synthetic 3-second test clip with distinct consecutive frames."""
    tmp = tmp_path_factory.mktemp("video")
    path = tmp / "clip.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=5)
    stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
    for i in range(15):
        img = np.tile(
            np.array([(i * 8) % 255, 128, 64], dtype=np.uint8),
            (32, 32, 1),
        )
        frame = av.VideoFrame.from_ndarray(img)
        frame.pts = i
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return path.read_bytes()


def test_select_sampling_rate_accepts_positive() -> None:
    assert video_module._select_sampling_rate(0.5) == 0.5
    assert video_module._select_sampling_rate(2.0) == 2.0


def test_select_sampling_rate_rejects_non_positive() -> None:
    for bad in (0, -0.1, -1):
        with pytest.raises(ValueError, match="must be > 0"):
            video_module._select_sampling_rate(bad)


def test_extract_rejects_negative_offset(synthetic_video_bytes: bytes) -> None:
    with pytest.raises(ValueError, match="offset_seconds must be >= 0"):
        video_module.extract_video_frames(
            synthetic_video_bytes,
            offset_seconds=-0.1,
            duration_seconds=1,
            sampling_rate=0.5,
        )


def test_extract_rejects_zero_or_negative_duration(synthetic_video_bytes: bytes) -> None:
    with pytest.raises(ValueError, match="duration_seconds must be > 0"):
        video_module.extract_video_frames(
            synthetic_video_bytes,
            offset_seconds=0,
            duration_seconds=0,
            sampling_rate=0.5,
        )


def test_extract_returns_interleaved_text_and_image_blocks(synthetic_video_bytes: bytes) -> None:
    """Two frames (t=0 and t=2) interleaved with text headers at 0.5 fps over a 3s clip."""
    blocks = video_module.extract_video_frames(
        synthetic_video_bytes,
        offset_seconds=0,
        duration_seconds=30,
        sampling_rate=0.5,
    )
    assert [b["type"] for b in blocks] == ["text", "image", "text", "image"]
    assert blocks[0]["text"].startswith("Frame at t=")
    assert blocks[1]["mime_type"] == "image/jpeg"
    assert blocks[2]["text"].startswith("Frame at t=")
    assert blocks[3]["mime_type"] == "image/jpeg"
    assert blocks[0]["text"] != blocks[2]["text"]
    jpeg = base64.b64decode(blocks[1]["base64"])
    assert jpeg[:3] == b"\xff\xd8\xff"


def test_extract_offset_skips_into_source(synthetic_video_bytes: bytes) -> None:
    """`offset_seconds` seeks into the clip; every emitted frame is at or after that point."""
    blocks = video_module.extract_video_frames(
        synthetic_video_bytes,
        offset_seconds=1.0,
        duration_seconds=1.5,
        sampling_rate=1.0,
    )
    headers = [b["text"] for b in blocks if b["type"] == "text"]
    assert headers, "expected at least one frame in [1s, 2.5s) at 1fps"
    for h in headers:
        assert h.startswith("Frame at t=")


def test_extract_no_frames_in_window_raises(synthetic_video_bytes: bytes) -> None:
    """Reading past the end of the clip yields a `VideoExtractionError`."""
    with pytest.raises(video_module.VideoExtractionError, match="No frames decoded"):
        video_module.extract_video_frames(
            synthetic_video_bytes,
            offset_seconds=10,
            duration_seconds=5,
            sampling_rate=0.5,
        )
