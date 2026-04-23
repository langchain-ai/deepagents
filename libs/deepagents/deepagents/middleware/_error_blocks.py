"""Error text-block builders for the video frame extraction middleware.

Exact wording is snapshot-tested; changes require updating
`tests/unit_tests/middleware/test_error_blocks.py`.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorReason(StrEnum):
    """Reason a video could not be transcoded to frames."""

    FFMPEG_MISSING = "ffmpeg_missing"
    NO_VIDEO_STREAM = "no_video_stream"
    FILE_CORRUPT = "file_corrupt"
    EXTRACTION_FAILED = "extraction_failed"


_REASON_PHRASES: dict[ErrorReason, str] = {
    ErrorReason.FFMPEG_MISSING: "ffmpeg is not installed on this host",
    ErrorReason.NO_VIDEO_STREAM: "the file contains no video stream",
    ErrorReason.FILE_CORRUPT: "the file is corrupt or unreadable",
    ErrorReason.EXTRACTION_FAILED: "frame extraction failed",
}


def build_error_text(filename: str, reason: ErrorReason) -> str:
    """Build the user-facing error text block for a failed video extraction.

    Args:
        filename: The filename or placeholder to mention in the message.
        reason: Why extraction failed.

    Returns:
        A single-line error string intended to be wrapped in a text content
        block and spliced into the conversation in place of the video.

    Raises:
        ValueError: If `reason` is not a known `ErrorReason`.
    """
    phrase = _REASON_PHRASES.get(reason)
    if phrase is None:
        msg = f"unknown error reason: {reason!r}"
        raise ValueError(msg)
    return f"[Video '{filename}' could not be processed: {phrase}. Please ask the user to describe it or retry.]"
