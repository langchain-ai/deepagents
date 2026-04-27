"""Pure-subprocess FFmpeg wrapper used by the read_file video extraction path.

No LangChain imports. No `deepagents` imports beyond stdlib. This isolation lets
the read_file tests mock this module wholesale and lets these functions be
tested against a real ffmpeg binary without pulling in the rest of deepagents.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


class ExtractionError(Exception):
    """Base class for errors raised by the FFmpeg wrapper."""


class FFmpegMissingError(ExtractionError):
    """Raised when ffmpeg or ffprobe is not found on the host."""


class NoVideoStreamError(ExtractionError):
    """Raised when a file has no decodable video stream."""


class FileCorruptError(ExtractionError):
    """Raised when ffprobe cannot parse the file or reports zero duration."""


class ExtractionFailedError(ExtractionError):
    """Raised when ffmpeg exits non-zero for any other reason (incl. timeout)."""


@dataclass(frozen=True)
class ExtractionParams:
    """Parameters controlling frame selection and JPEG encoding."""

    sampling_rate: float
    time_offset: float
    duration: float
    max_width: int
    jpeg_quality: int


@dataclass(frozen=True)
class ExtractedFrame:
    """A single extracted JPEG frame paired with its source timestamp."""

    jpeg_bytes: bytes
    timestamp_s: float


@lru_cache(maxsize=1)
def check_ffmpeg_available() -> bool:
    """Return True iff both `ffmpeg` and `ffprobe` are on PATH.

    Result is cached for the lifetime of the process; `cache_clear()` is
    available for tests.
    """
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


_FFPROBE_TIMEOUT_S = 15


def _run_ffprobe(video_path: Path) -> dict | None:
    """Run ffprobe and return parsed JSON, or None on any failure."""
    try:
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=_FFPROBE_TIMEOUT_S,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def probe_duration(video_path: Path) -> float | None:
    """Return the duration of a video in seconds, or None if unavailable."""
    data = _run_ffprobe(video_path)
    if not data:
        return None
    duration_str = data.get("format", {}).get("duration")
    if not isinstance(duration_str, str):
        return None
    try:
        return float(duration_str)
    except ValueError:
        return None


def probe_has_video_stream(video_path: Path) -> bool:
    """Return True iff the file has at least one decodable video stream."""
    data = _run_ffprobe(video_path)
    if not data:
        return False
    streams = data.get("streams")
    if not isinstance(streams, list):
        return False
    return any(s.get("codec_type") == "video" for s in streams if isinstance(s, dict))


def extract_frames(
    video_path: Path,
    params: ExtractionParams,
) -> tuple[list[ExtractedFrame], float]:
    """Extract a deterministic, fps-paced sequence of JPEG frames from a video.

    Frames are sampled at `params.sampling_rate` fps, starting at
    `params.time_offset` seconds into the source, for at most
    `params.duration` seconds of source material.

    Args:
        video_path: Path to a readable video file.
        params: Extraction parameters.

    Returns:
        A tuple `(frames, duration_s)`. `frames` is a list of
        `ExtractedFrame` ordered by source timestamp. `duration_s` is the
        source video duration in seconds as reported by ffprobe (the full
        clip length, not the requested window).

    Raises:
        FFmpegMissingError: If `ffmpeg` / `ffprobe` are not on PATH.
        NoVideoStreamError: If the file has no video stream.
        FileCorruptError: If ffprobe can't determine duration.
        ExtractionFailedError: If ffmpeg exits non-zero or times out.
    """
    if not check_ffmpeg_available():
        msg = "ffmpeg or ffprobe not found on PATH"
        raise FFmpegMissingError(msg)

    if not probe_has_video_stream(video_path):
        msg = f"no video stream in {video_path}"
        raise NoVideoStreamError(msg)

    duration_s = probe_duration(video_path)
    if duration_s is None or duration_s <= 0:
        msg = f"ffprobe reported no duration for {video_path}"
        raise FileCorruptError(msg)

    with tempfile.TemporaryDirectory(prefix="vfe_") as tmpdir_s:
        tmpdir = Path(tmpdir_s)
        output_pattern = tmpdir / "frame_%05d.jpg"

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{params.time_offset}",
            "-t",
            f"{params.duration}",
            "-i",
            str(video_path),
            "-vf",
            f"fps={params.sampling_rate},scale='min({params.max_width}\\,iw)':-2",
            "-fps_mode",
            "vfr",
            "-q:v",
            str(params.jpeg_quality),
            str(output_pattern),
        ]

        timeout_s = max(30.0, params.duration * 2.0)
        try:
            subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout_s,
            )
        except subprocess.CalledProcessError as exc:
            raise ExtractionFailedError(exc.stderr or "ffmpeg non-zero exit") from exc
        except subprocess.TimeoutExpired as exc:
            msg = f"ffmpeg timed out after {timeout_s}s"
            raise ExtractionFailedError(msg) from exc
        except FileNotFoundError as exc:
            raise FFmpegMissingError(str(exc)) from exc

        frame_paths = sorted(tmpdir.glob("frame_*.jpg"))
        # `fps=N` produces one frame every 1/N seconds of source time, starting
        # at -ss. The k-th output frame's source timestamp is therefore
        # time_offset + k / sampling_rate.
        interval = 1.0 / params.sampling_rate if params.sampling_rate > 0 else 0.0
        return (
            [
                ExtractedFrame(
                    jpeg_bytes=path.read_bytes(),
                    timestamp_s=params.time_offset + idx * interval,
                )
                for idx, path in enumerate(frame_paths)
            ],
            duration_s,
        )
