"""Pure-subprocess FFmpeg wrapper used by the video frame extraction middleware.

No LangChain imports. No `deepagents` imports beyond stdlib. This isolation lets
the middleware tests mock this module wholesale and lets these functions be
tested against a real ffmpeg binary without pulling in the rest of deepagents.
"""

from __future__ import annotations

import shutil
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


# Forward declarations so the module exports a stable public surface even
# before the remaining functions are implemented in later tasks.
def probe_duration(video_path: Path) -> float | None:  # pragma: no cover - stub
    raise NotImplementedError


def probe_has_video_stream(video_path: Path) -> bool:  # pragma: no cover - stub
    raise NotImplementedError


def extract_frames(
    video_path: Path,
    params: ExtractionParams,
) -> list[ExtractedFrame]:  # pragma: no cover - stub
    raise NotImplementedError
