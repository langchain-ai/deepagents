"""Integration tests exercising real ffmpeg/ffprobe binaries.

Skipped automatically when either binary is missing. Marked `integration` so
they can be filtered.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from deepagents.middleware._ffmpeg import (
    ExtractionParams,
    FileCorruptError,
    NoVideoStreamError,
    check_ffmpeg_available,
    extract_frames,
    probe_duration,
    probe_has_video_stream,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
        reason="ffmpeg/ffprobe not available",
    ),
]


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 5-second 320x240 10fps colour bars video."""
    path = tmp_path_factory.mktemp("video") / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi",
            "-i", "testsrc=duration=5:size=320x240:rate=10",
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def audio_only_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An audio-only file with .mp4 extension to test NoVideoStreamError."""
    path = tmp_path_factory.mktemp("audio") / "audio.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi",
            "-i", "sine=frequency=440:duration=2",
            "-c:a", "aac",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture
def corrupt_video(tmp_path: Path, sample_video: Path) -> Path:
    """A truncated video file."""
    path = tmp_path / "corrupt.mp4"
    data = sample_video.read_bytes()
    path.write_bytes(data[:64])
    return path


def test_check_ffmpeg_available_returns_true() -> None:
    check_ffmpeg_available.cache_clear()
    assert check_ffmpeg_available() is True


def test_probe_duration_matches_source(sample_video: Path) -> None:
    duration = probe_duration(sample_video)
    assert duration is not None
    assert duration == pytest.approx(5.0, abs=0.2)


def test_probe_has_video_stream_true(sample_video: Path) -> None:
    assert probe_has_video_stream(sample_video) is True


def test_probe_has_video_stream_false_for_audio_only(audio_only_file: Path) -> None:
    assert probe_has_video_stream(audio_only_file) is False


def test_extract_frames_short_video(sample_video: Path) -> None:
    params = ExtractionParams(
        max_frames=95, scene_threshold=0.3, max_width=1024, jpeg_quality=5
    )
    frames = extract_frames(sample_video, params)

    # baseline_interval = 1.0 → at least ~5 frames, plus possible scene frames.
    assert 3 <= len(frames) <= params.max_frames
    # Timestamps are monotonically non-decreasing and within the source duration.
    timestamps = [f.timestamp_s for f in frames]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == pytest.approx(0.0, abs=0.2)
    assert timestamps[-1] <= 5.1
    # All frames are valid JPEG (SOI/EOI markers).
    for f in frames:
        assert f.jpeg_bytes.startswith(b"\xff\xd8\xff")
        assert f.jpeg_bytes.endswith(b"\xff\xd9")


def test_extract_frames_respects_max_frames(sample_video: Path) -> None:
    params = ExtractionParams(
        max_frames=2, scene_threshold=0.3, max_width=1024, jpeg_quality=5
    )
    frames = extract_frames(sample_video, params)
    assert len(frames) <= 2


def test_extract_frames_respects_max_width(sample_video: Path) -> None:
    pytest.importorskip("PIL")
    from io import BytesIO  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    params = ExtractionParams(
        max_frames=95, scene_threshold=0.3, max_width=128, jpeg_quality=5
    )
    frames = extract_frames(sample_video, params)
    for f in frames:
        with Image.open(BytesIO(f.jpeg_bytes)) as im:
            assert im.width <= 128


def test_extract_frames_raises_no_video_stream(audio_only_file: Path) -> None:
    params = ExtractionParams(
        max_frames=95, scene_threshold=0.3, max_width=1024, jpeg_quality=5
    )
    with pytest.raises(NoVideoStreamError):
        extract_frames(audio_only_file, params)


def test_extract_frames_raises_file_corrupt(corrupt_video: Path) -> None:
    params = ExtractionParams(
        max_frames=95, scene_threshold=0.3, max_width=1024, jpeg_quality=5
    )
    with pytest.raises((FileCorruptError, NoVideoStreamError)):
        # A sufficiently corrupt file may also fail the video-stream probe first.
        extract_frames(corrupt_video, params)
