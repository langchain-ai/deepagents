"""Snapshot tests for video-extraction error text strings.

These tests intentionally assert exact wording so UX regressions are caught —
the error strings appear in conversation history and are read by end users.
"""

from __future__ import annotations

import pytest

from deepagents.middleware._error_blocks import (
    ErrorReason,
    build_error_text,
)


def test_ffmpeg_missing() -> None:
    assert build_error_text("video_1.mp4", ErrorReason.FFMPEG_MISSING) == (
        "[Video 'video_1.mp4' could not be processed: ffmpeg is not installed "
        "on this host. Please ask the user to describe it or retry.]"
    )


def test_no_video_stream() -> None:
    assert build_error_text("clip.mov", ErrorReason.NO_VIDEO_STREAM) == (
        "[Video 'clip.mov' could not be processed: the file contains no video "
        "stream. Please ask the user to describe it or retry.]"
    )


def test_file_corrupt() -> None:
    assert build_error_text("bad.mp4", ErrorReason.FILE_CORRUPT) == (
        "[Video 'bad.mp4' could not be processed: the file is corrupt or "
        "unreadable. Please ask the user to describe it or retry.]"
    )


def test_extraction_failed() -> None:
    assert build_error_text("[video 1]", ErrorReason.EXTRACTION_FAILED) == (
        "[Video '[video 1]' could not be processed: frame extraction failed. "
        "Please ask the user to describe it or retry.]"
    )


def test_all_reasons_are_covered() -> None:
    # Defensive: every ErrorReason member must produce a string.
    for reason in ErrorReason:
        text = build_error_text("x", reason)
        assert text.startswith("[Video 'x' could not be processed:")
        assert text.endswith("Please ask the user to describe it or retry.]")


def test_unknown_reason_raises() -> None:
    with pytest.raises(ValueError, match="unknown error reason"):
        build_error_text("x", "not-a-reason")  # type: ignore[arg-type]
