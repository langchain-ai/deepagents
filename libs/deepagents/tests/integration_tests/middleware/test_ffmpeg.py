"""Integration tests exercising real ffmpeg/ffprobe binaries.

Skipped automatically when either binary is missing. Marked `integration` so
they can be filtered.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path  # noqa: TC003

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


def _params(**overrides: object) -> ExtractionParams:
    base = {
        "sampling_rate": 1.0,
        "time_offset": 0.0,
        "duration": 5.0,
        "max_width": 512,
        "jpeg_quality": 5,
    }
    base.update(overrides)
    return ExtractionParams(**base)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 5-second 320x240 10fps colour bars video."""
    path = tmp_path_factory.mktemp("video") / "sample.mp4"
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
            "testsrc=duration=5:size=320x240:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def audio_only_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An audio-only file with .mp4 extension to test NoVideoStreamError."""
    path = tmp_path_factory.mktemp("audio") / "audio.mp4"
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
            "sine=frequency=440:duration=2",
            "-c:a",
            "aac",
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


def test_extract_frames_default_window(sample_video: Path) -> None:
    # 1 fps over 5s should yield ~5 frames.
    frames, duration = extract_frames(sample_video, _params(sampling_rate=1.0, duration=5.0))
    assert duration == pytest.approx(5.0, abs=0.2)
    assert 4 <= len(frames) <= 6
    timestamps = [f.timestamp_s for f in frames]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == pytest.approx(0.0, abs=0.2)
    for f in frames:
        assert f.jpeg_bytes.startswith(b"\xff\xd8\xff")
        assert f.jpeg_bytes.endswith(b"\xff\xd9")


def test_extract_frames_with_time_offset(sample_video: Path) -> None:
    # Start at 2s for 2s at 1fps → expect ~2 frames whose nominal timestamps
    # start at 2.0s (offset + idx / sampling_rate).
    frames, _duration = extract_frames(sample_video, _params(sampling_rate=1.0, time_offset=2.0, duration=2.0))
    assert 1 <= len(frames) <= 3
    assert frames[0].timestamp_s == pytest.approx(2.0, abs=0.5)


def test_extract_frames_respects_sampling_rate(sample_video: Path) -> None:
    # 2 fps over 5s should yield ~10 frames.
    frames, _duration = extract_frames(sample_video, _params(sampling_rate=2.0, duration=5.0))
    assert 8 <= len(frames) <= 12


def test_extract_frames_respects_max_width(sample_video: Path) -> None:
    pytest.importorskip("PIL")
    from io import BytesIO  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415  # type: ignore[import-unresolved]

    frames, _duration = extract_frames(sample_video, _params(max_width=128))
    for f in frames:
        with Image.open(BytesIO(f.jpeg_bytes)) as im:
            assert im.width <= 128


def test_extract_frames_raises_no_video_stream(audio_only_file: Path) -> None:
    with pytest.raises(NoVideoStreamError):
        extract_frames(audio_only_file, _params())


def test_extract_frames_raises_file_corrupt(corrupt_video: Path) -> None:
    with pytest.raises((FileCorruptError, NoVideoStreamError)):
        extract_frames(corrupt_video, _params())
