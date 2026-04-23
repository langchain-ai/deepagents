"""Pure-subprocess FFmpeg wrapper used by the video frame extraction middleware.

No LangChain imports. No `deepagents` imports beyond stdlib. This isolation lets
the middleware tests mock this module wholesale and lets these functions be
tested against a real ffmpeg binary without pulling in the rest of deepagents.
"""

from __future__ import annotations

import json
import re
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
    """Parameters controlling frame selection and JPEG encoding.

    All fields are immutable so that instances can participate in the
    middleware's content-hash cache key.
    """

    max_frames: int
    scene_threshold: float
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
    """Return the duration of a video in seconds, or None if unavailable.

    Args:
        video_path: Path to a video file on disk.

    Returns:
        Duration in seconds as a float, or None if ffprobe fails or the
        duration field is absent / unparseable.
    """
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


# ffmpeg's `metadata=print` filter emits lines like
#   frame:N    pts:P      pts_time:T
# followed by zero-or-more `key=value` metadata lines. Match frame markers
# tolerantly — any non-newline characters may appear between `frame:N` and
# `pts_time:T` (e.g. `pts:P`), and trailing content on the line is ignored.
_TS_LINE_RE = re.compile(r"^frame:(\d+)[^\n]*?pts_time:([0-9]+(?:\.[0-9]+)?)", re.MULTILINE)


def _parse_timestamps(metadata_file: Path) -> dict[int, float]:
    """Parse `metadata=print` output into a frame_index → timestamp_s map."""
    text = metadata_file.read_text() if metadata_file.exists() else ""
    out: dict[int, float] = {}
    for match in _TS_LINE_RE.finditer(text):
        out[int(match.group(1))] = float(match.group(2))
    return out


def _build_filter_string(baseline_interval: float, params: ExtractionParams, metadata_path: Path) -> str:
    r"""Build the `-vf` filter graph string.

    Commas separating filter-arg expressions inside `select=` / `scale=` must be
    escaped as `\\,` (a literal backslash + comma in the string passed to ffmpeg).
    """
    select = f"select='isnan(prev_selected_t) + gte(t-prev_selected_t\\,{baseline_interval}) + gt(scene\\,{params.scene_threshold})'"
    scale = f"scale='min({params.max_width}\\,iw)':-2"
    meta = f"metadata=print:file={metadata_path}"
    return f"{select}, {scale}, {meta}"


def extract_frames(
    video_path: Path,
    params: ExtractionParams,
) -> tuple[list[ExtractedFrame], float]:
    """Extract a capped, scene-aware sequence of JPEG frames from a video.

    Args:
        video_path: Path to a readable video file.
        params: Extraction parameters.

    Returns:
        A tuple `(frames, duration_s)`. `frames` is a list of
        `ExtractedFrame` ordered by source timestamp, with at most
        `params.max_frames` entries. `duration_s` is the source video
        duration in seconds as reported by ffprobe.

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

    baseline_interval = max(1.0, duration_s / params.max_frames)

    with tempfile.TemporaryDirectory(prefix="vfe_") as tmpdir_s:
        tmpdir = Path(tmpdir_s)
        metadata_file = tmpdir / "ts.txt"
        output_pattern = tmpdir / "frame_%05d.jpg"
        filter_str = _build_filter_string(baseline_interval, params, metadata_file)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            filter_str,
            "-fps_mode",
            "vfr",
            "-q:v",
            str(params.jpeg_quality),
            str(output_pattern),
        ]

        timeout_s = max(30.0, duration_s * 2.0)
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

        timestamps = _parse_timestamps(metadata_file)
        frame_paths = sorted(tmpdir.glob("frame_*.jpg"))
        frames: list[ExtractedFrame] = []
        for idx, path in enumerate(frame_paths):
            ts = timestamps.get(idx, float(idx) * baseline_interval)
            frames.append(ExtractedFrame(jpeg_bytes=path.read_bytes(), timestamp_s=ts))

        # Belt-and-suspenders hard-trim: scene detection can overshoot.
        return frames[: params.max_frames], duration_s
