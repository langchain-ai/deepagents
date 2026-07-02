"""Unit tests for the PyAV-based video frame extractor.

The tests exercise the real PyAV decoder against a synthetic 3-second clip so
the offset/limit -> seconds reinterpretation, frame sampling, validation, and
error mapping all run end-to-end. They are skipped automatically when the
`av` (PyAV) extra is not installed, which keeps the default unit suite
lightweight.
"""

import base64
import io
from collections.abc import Iterator
from typing import cast

import pytest

import deepagents.middleware._video as video_module

av = pytest.importorskip("av", reason="`av` extra not installed (pip install deepagents[video])")
Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")


class _FakeFrame:
    def __init__(self, pts: int, *, width: int = 32, height: int = 32) -> None:
        self.pts = pts
        self.width = width
        self.height = height


@pytest.fixture(scope="module")
def synthetic_video_bytes(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Build a synthetic 3-second test clip with distinct consecutive frames."""
    tmp = tmp_path_factory.mktemp("video")
    path = tmp / "clip.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=5)
    stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
    for i in range(15):
        img = Image.new("RGB", (32, 32), ((i * 8) % 255, 128, 64))
        frame = av.VideoFrame.from_image(img)
        frame.pts = i
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return path.read_bytes()


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


def test_extract_rejects_zero_or_negative_sampling_rate(synthetic_video_bytes: bytes) -> None:
    with pytest.raises(ValueError, match="sampling_rate must be > 0"):
        video_module.extract_video_frames(
            synthetic_video_bytes,
            offset_seconds=0,
            duration_seconds=1,
            sampling_rate=0,
        )


def test_extract_returns_interleaved_text_and_image_blocks(synthetic_video_bytes: bytes) -> None:
    """Two frames (t=0 and t=2) interleaved with text headers at 0.5 fps over a 3s clip."""
    blocks = cast(
        "list[dict[str, str]]",
        video_module.extract_video_frames(
            synthetic_video_bytes,
            offset_seconds=0,
            duration_seconds=30,
            sampling_rate=0.5,
        ),
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
    blocks = cast(
        "list[dict[str, str]]",
        video_module.extract_video_frames(
            synthetic_video_bytes,
            offset_seconds=1.0,
            duration_seconds=1.5,
            sampling_rate=1.0,
        ),
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


def test_sample_frames_wraps_decode_iteration_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """PyAV decode errors raised after iteration starts surface as `VideoExtractionError`."""
    monkeypatch.setattr(video_module, "_encode_jpeg", lambda _frame: b"jpeg")

    def decoded_frames() -> Iterator[_FakeFrame]:
        yield _FakeFrame(0)
        raise av.error.InvalidDataError(1094995529, "invalid data")

    with pytest.raises(video_module.VideoExtractionError, match="Failed to decode video frames") as exc_info:
        video_module._sample_frames_in_window(
            decoded_frames(),
            offset_seconds=0,
            duration_seconds=3,
            sampling_rate=1,
            time_base=1,
            decode_error_types=video_module._video_backend_error_types(av),
        )

    assert isinstance(exc_info.value.__cause__, av.error.InvalidDataError)


def test_extract_wraps_seek_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seek failures surface as `VideoExtractionError` instead of escaping."""

    class FakeStream:
        type = "video"
        time_base = 1
        start_time = 0

    class FakeContainer:
        def __init__(self) -> None:
            self.closed = False
            self.streams = [FakeStream()]

        def seek(self, *_args: object, **_kwargs: object) -> None:
            msg = "seek failed"
            raise OSError(msg)

        def decode(self, *_args: object) -> Iterator[object]:
            msg = "decode should not run after seek fails"
            raise AssertionError(msg)

        def close(self) -> None:
            self.closed = True

    class FakeAv:
        error = type("ErrorNamespace", (), {})

        @staticmethod
        def open(_payload: object) -> FakeContainer:
            return container

    container = FakeContainer()
    monkeypatch.setattr(video_module, "_import_av", lambda: FakeAv)

    with pytest.raises(video_module.VideoExtractionError, match="Failed to decode video frames") as exc_info:
        video_module.extract_video_frames(
            b"fake",
            offset_seconds=1,
            duration_seconds=1,
            sampling_rate=1,
        )

    assert isinstance(exc_info.value.__cause__, OSError)
    assert container.closed


def test_sample_frames_normalizes_non_zero_stream_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Frame timestamps are measured from the video start, not raw container PTS."""
    monkeypatch.setattr(video_module, "_encode_jpeg", lambda _frame: b"jpeg")

    blocks = cast(
        "list[dict[str, str]]",
        video_module._sample_frames_in_window(
            [_FakeFrame(3600), _FakeFrame(3601), _FakeFrame(3602)],
            offset_seconds=0,
            duration_seconds=3,
            sampling_rate=1,
            time_base=1,
            stream_start_seconds=3600,
        ),
    )

    headers = [block["text"] for block in blocks if block["type"] == "text"]
    assert headers == [
        "Frame at t=00:00:00.000",
        "Frame at t=00:00:01.000",
        "Frame at t=00:00:02.000",
    ]


def test_sample_frames_keeps_next_target_after_late_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A late frame should not skip the next sampling target."""
    monkeypatch.setattr(video_module, "_encode_jpeg", lambda _frame: b"jpeg")

    blocks = cast(
        "list[dict[str, str]]",
        video_module._sample_frames_in_window(
            [_FakeFrame(1), _FakeFrame(2), _FakeFrame(3)],
            offset_seconds=0.4,
            duration_seconds=4,
            sampling_rate=1,
            time_base=1,
        ),
    )

    headers = [block["text"] for block in blocks if block["type"] == "text"]
    assert headers == [
        "Frame at t=00:00:01.000",
        "Frame at t=00:00:02.000",
        "Frame at t=00:00:03.000",
    ]


def test_sample_frames_does_not_cap_requested_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Huge requested windows are passed through unchanged.

    Output is still bounded by the layered output caps (`MAX_VIDEO_SAMPLED_FRAMES`,
    etc.); the agent's `limit` is the only authority on how much of the source is
    sampled.
    """
    monkeypatch.setattr(video_module, "_encode_jpeg", lambda _frame: b"jpeg")
    monkeypatch.setattr(video_module, "MAX_VIDEO_SAMPLED_FRAMES", 2)

    blocks = cast(
        "list[dict[str, str]]",
        video_module._sample_frames_in_window(
            [_FakeFrame(0), _FakeFrame(1), _FakeFrame(2), _FakeFrame(3)],
            offset_seconds=0,
            duration_seconds=999,
            sampling_rate=1,
            time_base=1,
        ),
    )

    headers = [block["text"] for block in blocks if block["type"] == "text"]
    # Three frames fit inside the 999 s window without being silently clamped;
    # only the frame-count cap (`MAX_VIDEO_SAMPLED_FRAMES = 2`) trims the output.
    assert headers == ["Frame at t=00:00:00.000", "Frame at t=00:00:01.000"]


def test_sample_frames_caps_frame_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sampling stops at the fixed frame-count budget."""
    monkeypatch.setattr(video_module, "_encode_jpeg", lambda _frame: b"jpeg")
    monkeypatch.setattr(video_module, "MAX_VIDEO_SAMPLED_FRAMES", 2)

    blocks = cast(
        "list[dict[str, str]]",
        video_module._sample_frames_in_window(
            [_FakeFrame(0), _FakeFrame(1), _FakeFrame(2)],
            offset_seconds=0,
            duration_seconds=3,
            sampling_rate=1,
            time_base=1,
        ),
    )

    headers = [block["text"] for block in blocks if block["type"] == "text"]
    assert headers == ["Frame at t=00:00:00.000", "Frame at t=00:00:01.000"]


def test_sample_frames_rejects_oversized_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extraction fails before retaining an oversized first image block."""
    monkeypatch.setattr(video_module, "_encode_jpeg", lambda _frame: b"jpeg")
    monkeypatch.setattr(video_module, "MAX_VIDEO_EMITTED_BYTES", 1)

    with pytest.raises(video_module.VideoExtractionError, match="safety budget"):
        video_module._sample_frames_in_window(
            [_FakeFrame(0)],
            offset_seconds=0,
            duration_seconds=1,
            sampling_rate=1,
            time_base=1,
        )


def test_encode_jpeg_rejects_oversized_dimensions_before_conversion() -> None:
    """Oversized frames are rejected before Pillow conversion retains them."""

    class OversizedFrame(_FakeFrame):
        def __init__(self) -> None:
            super().__init__(0, width=video_module.MAX_VIDEO_FRAME_SIDE + 1, height=1)
            self.converted = False

        def to_image(self):  # type: ignore[no-untyped-def]
            self.converted = True
            msg = "oversized frame should not be converted"
            raise AssertionError(msg)

    frame = OversizedFrame()

    with pytest.raises(video_module.VideoExtractionError, match="dimensions"):
        video_module._encode_jpeg(frame)
    assert not frame.converted


def test_encode_jpeg_downscales_high_resolution_frames() -> None:
    """Normal high-resolution video frames are resized instead of rejected."""

    class HighResolutionFrame(_FakeFrame):
        def __init__(self) -> None:
            super().__init__(0, width=2218, height=1440)

        def to_image(self):  # type: ignore[no-untyped-def]
            return Image.new("RGB", (2218, 1440), "black")

    jpeg = video_module._encode_jpeg(HighResolutionFrame())
    with Image.open(io.BytesIO(jpeg)) as img:
        width, height = img.size

    assert width <= video_module.MAX_VIDEO_OUTPUT_WIDTH
    assert height <= video_module.MAX_VIDEO_OUTPUT_HEIGHT
    assert width * height <= video_module.MAX_VIDEO_FRAME_PIXELS


def test_sample_frames_enforces_decode_deadline() -> None:
    """Best-effort decode timeout raises instead of continuing indefinitely."""
    with pytest.raises(video_module.VideoExtractionError, match="decoding exceeded"):
        video_module._sample_frames_in_window(
            [_FakeFrame(0)],
            offset_seconds=0,
            duration_seconds=1,
            sampling_rate=1,
            time_base=1,
            deadline_seconds=0,
        )
