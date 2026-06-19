"""Video frame extraction for the `read_file` tool.

This module is the boundary between `read_file` and the optional video
backend (PyAV). It is imported lazily so a `deepagents` install without the
``[video]`` extra stays lightweight; the import only fires when the agent
actually tries to read a video file.

For each video read the module decodes a contiguous slice of the source
(``offset``-seconds skip, ``limit``-seconds window) and emits sampled
frames at the configured sampling rate. The output is a list of interleaved
text+image content blocks so the model sees per-frame timestamps alongside
the JPEGs.

Sampling rate is a deploy-time knob on `FilesystemMiddleware`, not an
agent-facing argument — the public `read_file` surface stays narrow and
operators trade frame density for token cost without prompting changes.
"""

import base64
import io
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.messages.content import ContentBlock
else:
    ContentBlock = dict  # only used at runtime by the agent runtime


MISSING_VIDEO_HINT = "Reading video files requires the optional video dependencies. Install them with `pip install 'deepagents[video]'`."


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


def _select_sampling_rate(value: float) -> float:
    """Validate and normalize the constructor-supplied sampling rate."""
    if value <= 0:
        msg = f"video_sampling_rate must be > 0, got {value!r}"
        raise ValueError(msg)
    return float(value)


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
    if duration_seconds <= 0:
        msg = f"duration_seconds must be > 0, got {duration_seconds!r}"
        raise ValueError(msg)
    rate = _select_sampling_rate(sampling_rate)

    av = _import_av()
    container = _open_video_container(av, content)
    try:
        video_stream = _find_video_stream(container)
        time_base = float(video_stream.time_base)
        if offset_seconds > 0:
            # `seek` keeps the math correct across containers that already sit
            # at a non-zero timeline (e.g. trimmed clips).
            start_pts = int(offset_seconds / time_base)
            container.seek(start_pts, any_frame=False, backward=True, stream=video_stream)

        blocks = list(
            _sample_frames_in_window(
                container.decode(video_stream),
                offset_seconds=offset_seconds,
                duration_seconds=duration_seconds,
                sampling_rate=rate,
                time_base=time_base,
            )
        )
    finally:
        container.close()

    if not blocks:
        end_seconds = offset_seconds + duration_seconds
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
    except av.error.InvalidDataError as exc:  # pragma: no cover - depends on the input
        msg = f"Failed to open video payload: {exc}"
        raise VideoExtractionError(msg) from exc
    except OSError as exc:  # pragma: no cover - dependency ffmpeg missing on host
        msg = f"Failed to open video payload: {exc}"
        raise VideoExtractionError(msg) from exc


def _find_video_stream(container: Any) -> Any:  # noqa: ANN401
    """Return the first video stream in `container` or raise."""
    video_stream = next((s for s in container.streams if s.type == "video"), None)
    if video_stream is None:
        msg = "Video payload contains no video stream"
        raise VideoExtractionError(msg)
    return video_stream


def _sample_frames_in_window(
    decoded_frames: Any,  # noqa: ANN401
    *,
    offset_seconds: float,
    duration_seconds: float,
    sampling_rate: float,
    time_base: float,
) -> list[ContentBlock]:
    """Pick JPEG+timestamp content blocks for frames inside the requested window."""
    frame_interval_seconds = 1.0 / sampling_rate
    end_seconds = offset_seconds + duration_seconds
    next_emit_seconds = offset_seconds
    blocks: list[ContentBlock] = []
    for frame in decoded_frames:
        frame_seconds = float(frame.pts) * time_base
        if frame_seconds >= end_seconds:
            break
        if frame_seconds + 1e-6 < next_emit_seconds:
            continue
        jpeg_bytes = _encode_jpeg(frame)
        ts = _format_timestamp(frame_seconds)
        blocks.append({"type": "text", "text": f"Frame at t={ts}"})
        blocks.append(
            {
                "type": "image",
                "base64": base64.b64encode(jpeg_bytes).decode("ascii"),
                "mime_type": "image/jpeg",
            }
        )
        emitted_index = round((frame_seconds - offset_seconds) / frame_interval_seconds) + 1
        next_emit_seconds = offset_seconds + frame_interval_seconds * emitted_index
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

    img = frame.to_image() if hasattr(frame, "to_image") else Image.fromarray(frame.to_ndarray(format="rgb24"))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
