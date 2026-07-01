"""Video frame extraction for filesystem and request media handling.

This module is the boundary between Deep Agents middleware and the optional
video backend (PyAV). It imports PyAV lazily so a `deepagents` install without
the ``[video]`` extra stays lightweight; the import only fires when the agent
actually tries to read or transform a video.

For each video read the module decodes a contiguous slice of the source
(``offset``-seconds skip, ``limit``-seconds window) and emits sampled
frames at the configured sampling rate. The output is a list of interleaved
text+image content blocks so the model sees per-frame timestamps alongside
the JPEGs.

Sampling rate is intentionally fixed by the middleware. Agents control the
window they inspect through `read_file`'s existing `offset`/`limit` arguments,
which are interpreted as seconds for video reads.
"""

import base64
import io
import math
import time
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from langchain_core.messages.content import ContentBlock
else:
    ContentBlock = dict  # only used at runtime by the agent runtime


MISSING_VIDEO_HINT = "Reading video files requires the optional video dependencies. Install them with `uv add 'deepagents[video]'`."
MAX_VIDEO_SAMPLED_FRAMES: Final = 64
MAX_VIDEO_FRAME_PIXELS: Final = 1920 * 1080
MAX_VIDEO_FRAME_SIDE: Final = 4096
MAX_VIDEO_OUTPUT_WIDTH: Final = 1920
MAX_VIDEO_OUTPUT_HEIGHT: Final = 1080
MAX_VIDEO_EMITTED_BYTES: Final = 4 * 1024 * 1024
MAX_VIDEO_DECODE_SECONDS: Final = 10.0
_JPEG_QUALITY: Final = 85


class VideoExtractionError(RuntimeError):
    """Raised when PyAV cannot produce frames for the requested window."""


def _import_av() -> Any:  # noqa: ANN401
    """Import PyAV lazily so the dep stays optional.

    Returns:
        The imported `av` module.

    Raises:
        VideoExtractionError: If PyAV is not installed, with installation
            guidance in the message.
    """
    try:
        import av  # noqa: PLC0415 - lazy import keeps the extra optional
    except ImportError as exc:  # pragma: no cover - exercised only when `av` is absent
        msg = f"{MISSING_VIDEO_HINT} (underlying error: {exc})"
        raise VideoExtractionError(msg) from exc
    return av


def _format_timestamp(seconds: float) -> str:
    """Format a frame timestamp as `HH:MM:SS.mmm` for the text header block."""
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def extract_video_frames(
    content: bytes,
    *,
    offset_seconds: float,
    duration_seconds: float,
    sampling_rate: float,
) -> list[ContentBlock]:
    """Decode sampled frames from a video byte payload.

    Args:
        content: Raw bytes of the video file (as returned by the backend).
        offset_seconds: Seconds into the source to start sampling. Must be
            non-negative.
        duration_seconds: Seconds of source to sample. Must be > 0.
        sampling_rate: Frames per second to emit. > 0.

    Returns:
        Interleaved content blocks: a text header introducing each frame
        followed by a JPEG `image` content block. Raw video bytes are never
        returned.

    Raises:
        VideoExtractionError: If PyAV cannot open the payload or the
            requested window yields no decodable frames.
        ValueError: If argument validation fails before opening the file.
    """
    if offset_seconds < 0:
        msg = f"offset_seconds must be >= 0, got {offset_seconds!r}"
        raise ValueError(msg)
    if sampling_rate <= 0:
        msg = f"sampling_rate must be > 0, got {sampling_rate!r}"
        raise ValueError(msg)
    if duration_seconds <= 0:
        msg = f"duration_seconds must be > 0, got {duration_seconds!r}"
        raise ValueError(msg)
    rate = float(sampling_rate)
    duration = float(duration_seconds)

    av = _import_av()
    container = _open_video_container(av, content)
    backend_error_types = _video_backend_error_types(av)
    try:
        try:
            video_stream = _find_video_stream(container)
            time_base = float(video_stream.time_base)
            stream_start_seconds = _stream_start_seconds(video_stream, time_base)
            if offset_seconds > 0:
                # `seek` keeps the math correct across containers that already sit
                # at a non-zero timeline (e.g. trimmed clips).
                start_pts = _stream_start_pts(video_stream) + int(offset_seconds / time_base)
                container.seek(start_pts, any_frame=False, backward=True, stream=video_stream)

            blocks = list(
                _sample_frames_in_window(
                    container.decode(video_stream),
                    offset_seconds=offset_seconds,
                    duration_seconds=float(duration),
                    sampling_rate=rate,
                    time_base=time_base,
                    stream_start_seconds=stream_start_seconds,
                    deadline_seconds=time.monotonic() + MAX_VIDEO_DECODE_SECONDS,
                    decode_error_types=backend_error_types,
                )
            )
        except backend_error_types as exc:
            msg = f"Failed to decode video frames: {exc}"
            raise VideoExtractionError(msg) from exc
    finally:
        container.close()

    if not blocks:
        end_seconds = offset_seconds + duration
        msg = f"No frames decoded for window [{offset_seconds:.3f}s, {end_seconds:.3f}s)"
        raise VideoExtractionError(msg)
    return blocks


def _open_video_container(av: Any, content: bytes) -> Any:  # noqa: ANN401
    """Open a video byte payload, normalizing PyAV's failure modes.

    PyAV typically raises `av.error.InvalidDataError` for malformed inputs,
    but it falls back to `OSError` when the system ffmpeg library is missing
    or incompatible. Both surface to callers as `VideoExtractionError` so
    the middleware does not have to distinguish between them.
    """
    try:
        return av.open(io.BytesIO(content))
    except _video_backend_error_types(av) as exc:  # pragma: no cover - depends on host/input
        msg = f"Failed to open video payload: {exc}"
        raise VideoExtractionError(msg) from exc


def _video_backend_error_types(av: Any) -> tuple[type[BaseException], ...]:  # noqa: ANN401
    """Return backend failures that should surface as `VideoExtractionError`."""
    errors: list[type[BaseException]] = [OSError]
    av_error = getattr(av, "error", None)
    for name in ("FFmpegError", "InvalidDataError"):
        error_type = getattr(av_error, name, None)
        if (
            isinstance(error_type, type)
            and issubclass(error_type, BaseException)
            and not any(issubclass(error_type, existing) for existing in errors)
        ):
            errors.append(error_type)
    return tuple(errors)


def _find_video_stream(container: Any) -> Any:  # noqa: ANN401
    """Return the first video stream in `container` or raise."""
    video_stream = next((s for s in container.streams if s.type == "video"), None)
    if video_stream is None:
        msg = "Video payload contains no video stream"
        raise VideoExtractionError(msg)
    return video_stream


def _stream_start_pts(video_stream: Any) -> int:  # noqa: ANN401
    """Return the stream start timestamp in stream time-base units."""
    start_time = getattr(video_stream, "start_time", None)
    return int(start_time) if start_time is not None else 0


def _stream_start_seconds(video_stream: Any, time_base: float) -> float:  # noqa: ANN401
    """Return the stream start timestamp in seconds."""
    return _stream_start_pts(video_stream) * time_base


def _frame_seconds(frame: Any, *, time_base: float, stream_start_seconds: float) -> float | None:  # noqa: ANN401
    """Return a frame timestamp normalized to seconds from the video start."""
    pts = getattr(frame, "pts", None)
    if pts is not None:
        return max(0.0, float(pts) * time_base - stream_start_seconds)
    frame_time = getattr(frame, "time", None)
    if frame_time is not None:
        return max(0.0, float(frame_time) - stream_start_seconds)
    return None


def _frame_dimensions(frame: Any) -> tuple[int, int] | None:  # noqa: ANN401
    """Return frame dimensions when the decoder exposes them."""
    width = getattr(frame, "width", None)
    height = getattr(frame, "height", None)
    if width is None or height is None:
        return None
    return int(width), int(height)


def _validate_dimensions(width: int, height: int) -> None:
    """Reject frame dimensions that are too large to safely convert."""
    if width <= 0 or height <= 0:
        return
    if width > MAX_VIDEO_FRAME_SIDE or height > MAX_VIDEO_FRAME_SIDE:
        msg = f"Video frame dimensions {width}x{height} exceed the maximum {MAX_VIDEO_FRAME_SIDE}px side"
        raise VideoExtractionError(msg)


def _check_decode_deadline(deadline_seconds: float | None) -> None:
    """Raise when best-effort video decoding has exceeded its time budget."""
    if deadline_seconds is not None and time.monotonic() > deadline_seconds:
        msg = f"Video decoding exceeded the {MAX_VIDEO_DECODE_SECONDS:.1f}s safety budget"
        raise VideoExtractionError(msg)


def _sample_frames_in_window(
    decoded_frames: Any,  # noqa: ANN401
    *,
    offset_seconds: float,
    duration_seconds: float,
    sampling_rate: float,
    time_base: float,
    stream_start_seconds: float = 0.0,
    deadline_seconds: float | None = None,
    decode_error_types: tuple[type[BaseException], ...] = (),
) -> list[ContentBlock]:
    """Pick JPEG+timestamp content blocks for frames inside the requested window."""
    frame_interval_seconds = 1.0 / sampling_rate
    end_seconds = offset_seconds + float(duration_seconds)
    next_emit_seconds = offset_seconds
    blocks: list[ContentBlock] = []
    emitted_frames = 0
    emitted_bytes = 0
    try:
        for frame in decoded_frames:
            _check_decode_deadline(deadline_seconds)
            frame_seconds = _frame_seconds(frame, time_base=time_base, stream_start_seconds=stream_start_seconds)
            if frame_seconds is None:
                continue
            if frame_seconds >= end_seconds:
                break
            if frame_seconds + 1e-6 < next_emit_seconds:
                continue

            jpeg_bytes = _encode_jpeg(frame)
            image_base64 = base64.b64encode(jpeg_bytes)
            ts = _format_timestamp(frame_seconds)
            text = f"Frame at t={ts}"
            next_block_bytes = len(text.encode()) + len(image_base64)
            if emitted_bytes + next_block_bytes > MAX_VIDEO_EMITTED_BYTES:
                if emitted_frames == 0:
                    msg = f"Video frame output exceeded the {MAX_VIDEO_EMITTED_BYTES} byte safety budget before emitting a frame"
                    raise VideoExtractionError(msg)
                break

            blocks.append({"type": "text", "text": f"Frame at t={ts}"})
            blocks.append(
                {
                    "type": "image",
                    "base64": image_base64.decode("ascii"),
                    "mime_type": "image/jpeg",
                }
            )
            emitted_frames += 1
            emitted_bytes += next_block_bytes
            if emitted_frames >= MAX_VIDEO_SAMPLED_FRAMES:
                break
            emitted_index = math.floor((frame_seconds - offset_seconds) / frame_interval_seconds) + 1
            next_emit_seconds = max(
                next_emit_seconds + frame_interval_seconds,
                offset_seconds + frame_interval_seconds * emitted_index,
            )
    except decode_error_types as exc:
        msg = f"Failed to decode video frames: {exc}"
        raise VideoExtractionError(msg) from exc
    return blocks


def _encode_jpeg(frame: Any) -> bytes:  # noqa: ANN401
    """Encode a decoded PyAV frame as JPEG bytes via Pillow.

    Pillow is part of the `video` extra, and the import stays lazy so module
    load is independent of optional deps.
    """
    try:
        from PIL import Image  # noqa: PLC0415 - lazy import keeps the extra optional
    except ImportError as exc:  # pragma: no cover - exercised only when Pillow is absent
        msg = f"{MISSING_VIDEO_HINT} (underlying error: {exc})"
        raise VideoExtractionError(msg) from exc

    if dimensions := _frame_dimensions(frame):
        _validate_dimensions(*dimensions)

    img = frame.to_image() if hasattr(frame, "to_image") else Image.fromarray(frame.to_ndarray(format="rgb24"))
    _validate_dimensions(*img.size)
    width, height = img.size
    if width * height > MAX_VIDEO_FRAME_PIXELS or width > MAX_VIDEO_OUTPUT_WIDTH or height > MAX_VIDEO_OUTPUT_HEIGHT:
        img = img.copy()
        img.thumbnail((MAX_VIDEO_OUTPUT_WIDTH, MAX_VIDEO_OUTPUT_HEIGHT), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    return buf.getvalue()
