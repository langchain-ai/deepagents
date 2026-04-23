"""Middleware that transcodes VideoContentBlocks into text-interleaved image sequences.

The middleware fires as a `wrap_model_call` hook. For non-video-capable providers
(everything except Gemini by default), it:

1. Deep-copies `request.messages`.
2. Replaces every `VideoContentBlock` with a `[preamble, (ts, image) × N]`
   sequence by shelling out to ffmpeg.
3. Calls `handler(request.override(messages=transformed))`.

Persisted state is never mutated. A per-instance LRU cache keyed by
`(sha256(video_bytes), extraction_params)` ensures a given video is extracted
at most once per process.
"""

from __future__ import annotations

import base64
import binascii
import copy
import logging
from collections import OrderedDict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, AnyMessage

from deepagents._models import get_model_identifier, get_model_provider
from deepagents.middleware import _ffmpeg
from deepagents.middleware._error_blocks import ErrorReason, build_error_text
from deepagents.middleware._ffmpeg import (
    ExtractedFrame,
    ExtractionError,
    ExtractionFailedError,
    ExtractionParams,
    FFmpegMissingError,
    FileCorruptError,
    NoVideoStreamError,
)
from deepagents.middleware._video_capability import is_video_capable

logger = logging.getLogger(__name__)

_MIME_TO_EXT: dict[str, str] = {
    "video/mp4": "mp4",
    "video/quicktime": "mov",
    "video/x-msvideo": "avi",
    "video/webm": "webm",
    "video/x-m4v": "m4v",
    "video/x-ms-wmv": "wmv",
}


def _filename_for_video_block(block: dict[str, Any]) -> str:
    """Resolve a best-effort filename for a video content block.

    Priority: source_metadata.filename → source_metadata.name →
    extras.placeholder → literal "video".
    """
    source_metadata = block.get("source_metadata") or {}
    if isinstance(source_metadata, dict):
        for key in ("filename", "name"):
            value = source_metadata.get(key)
            if isinstance(value, str) and value:
                return value
    extras = block.get("extras") or {}
    if isinstance(extras, dict):
        placeholder = extras.get("placeholder")
        if isinstance(placeholder, str) and placeholder:
            return placeholder
    return "video"


def _is_video_block(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") == "video"


def _decode_video_bytes(block: dict[str, Any]) -> bytes:
    data = block.get("data")
    if not isinstance(data, str):
        msg = "video block missing 'data'"
        raise ExtractionFailedError(msg)
    try:
        return base64.b64decode(data, validate=True)
    except binascii.Error as exc:
        raise ExtractionFailedError("video block 'data' is not valid base64") from exc


def _ext_for_mime(mime: str | None) -> str:
    if mime and mime in _MIME_TO_EXT:
        return _MIME_TO_EXT[mime]
    return "mp4"


def _run_extraction(
    video_bytes: bytes, mime: str | None, params: ExtractionParams
) -> list[ExtractedFrame]:
    """Write the bytes to a temp file and invoke `_ffmpeg.extract_frames`.

    Seam for monkeypatching in middleware tests.
    """
    ext = _ext_for_mime(mime)
    with NamedTemporaryFile(suffix=f".{ext}", delete=True) as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        return _ffmpeg.extract_frames(Path(tmp.name), params)


def _build_preamble(
    filename: str, num_frames: int, duration_s: float, baseline_interval: float
) -> dict[str, Any]:
    fps = 1.0 / baseline_interval if baseline_interval > 0 else 0.0
    return {
        "type": "text",
        "text": (
            f"The following {num_frames} images are frames extracted from the "
            f"video '{filename}' (duration {duration_s:.1f}s, extracted at "
            f"~{fps:.2f} fps + scene changes). Each frame is preceded by a "
            f"timestamp."
        ),
    }


def _frames_to_blocks(
    frames: list[ExtractedFrame], filename: str, duration_s: float, baseline_interval: float
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        _build_preamble(filename, len(frames), duration_s, baseline_interval)
    ]
    for i, frame in enumerate(frames):
        blocks.append(
            {"type": "text", "text": f"Frame {i + 1} at t={frame.timestamp_s:.1f}s"}
        )
        blocks.append(
            {
                "type": "image",
                "source_type": "base64",
                "mime_type": "image/jpeg",
                "data": base64.b64encode(frame.jpeg_bytes).decode(),
                "source_metadata": {
                    "extracted_from": filename,
                    "original_duration_s": duration_s,
                    "frame_index": i,
                    "timestamp_s": frame.timestamp_s,
                },
            }
        )
    return blocks


def _error_block(filename: str, reason: ErrorReason) -> dict[str, Any]:
    return {"type": "text", "text": build_error_text(filename, reason)}


class VideoFrameExtractionMiddleware(AgentMiddleware):
    """Middleware that transcodes video blocks to frame sequences for non-video-capable models.

    See `docs/superpowers/specs/2026-04-23-video-frame-extraction-design.md`.
    """

    def __init__(
        self,
        *,
        max_frames: int = 95,
        scene_threshold: float = 0.30,
        max_width: int = 1024,
        jpeg_quality: int = 5,
        video_capable_override: bool | None = None,
        cache_size: int = 256,
    ) -> None:
        if not 1 <= max_frames <= 1000:
            msg = f"max_frames must be in [1, 1000]; got {max_frames}"
            raise ValueError(msg)
        if not 0.0 <= scene_threshold <= 1.0:
            msg = f"scene_threshold must be in [0.0, 1.0]; got {scene_threshold}"
            raise ValueError(msg)
        if not 64 <= max_width <= 4096:
            msg = f"max_width must be in [64, 4096]; got {max_width}"
            raise ValueError(msg)
        if not 2 <= jpeg_quality <= 31:
            msg = f"jpeg_quality must be in [2, 31]; got {jpeg_quality}"
            raise ValueError(msg)
        if not 0 <= cache_size <= 10000:
            msg = f"cache_size must be in [0, 10000]; got {cache_size}"
            raise ValueError(msg)

        super().__init__()
        self.max_frames = max_frames
        self.scene_threshold = scene_threshold
        self.max_width = max_width
        self.jpeg_quality = jpeg_quality
        self.video_capable_override = video_capable_override
        self.cache_size = cache_size

        self._cache: OrderedDict[
            tuple[str, tuple[int, float, int, int]], list[ExtractedFrame]
        ] = OrderedDict()

    @property
    def _params(self) -> ExtractionParams:
        return ExtractionParams(
            max_frames=self.max_frames,
            scene_threshold=self.scene_threshold,
            max_width=self.max_width,
            jpeg_quality=self.jpeg_quality,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        provider = get_model_provider(request.model)
        model_name = get_model_identifier(request.model) or ""
        if is_video_capable(provider, model_name, override=self.video_capable_override):
            return handler(request)

        transformed, mutated = self._transform_messages(request.messages)
        if not mutated:
            return handler(request)
        return handler(request.override(messages=transformed))

    def _transform_messages(
        self, messages: list[AnyMessage]
    ) -> tuple[list[AnyMessage], bool]:
        """Return (new_messages, whether_any_video_was_found)."""
        mutated = False
        new_messages = copy.deepcopy(list(messages))
        for message in new_messages:
            content = getattr(message, "content", None)
            if not isinstance(content, list):
                continue
            new_content: list[Any] = []
            for block in content:
                if _is_video_block(block):
                    mutated = True
                    new_content.extend(self._expand_video_block(block))
                else:
                    new_content.append(block)
            message.content = new_content  # type: ignore[attr-defined]
        return new_messages, mutated

    def _expand_video_block(self, block: dict[str, Any]) -> list[dict[str, Any]]:
        filename = _filename_for_video_block(block)
        try:
            video_bytes = _decode_video_bytes(block)
            frames = _run_extraction(video_bytes, block.get("mime_type"), self._params)
            if len(frames) >= 2:
                duration_s = frames[-1].timestamp_s - frames[0].timestamp_s
                baseline_interval = (
                    duration_s / (len(frames) - 1) if len(frames) > 1 else 1.0
                )
            else:
                duration_s = frames[0].timestamp_s if frames else 0.0
                baseline_interval = 1.0
            return _frames_to_blocks(frames, filename, duration_s, baseline_interval)
        except FFmpegMissingError as exc:
            logger.warning("ffmpeg missing; replacing video %r: %s", filename, exc)
            return [_error_block(filename, ErrorReason.FFMPEG_MISSING)]
        except NoVideoStreamError as exc:
            logger.warning("no video stream in %r: %s", filename, exc)
            return [_error_block(filename, ErrorReason.NO_VIDEO_STREAM)]
        except FileCorruptError as exc:
            logger.warning("corrupt video %r: %s", filename, exc)
            return [_error_block(filename, ErrorReason.FILE_CORRUPT)]
        except ExtractionFailedError as exc:
            logger.warning("extraction failed for %r: %s", filename, exc)
            return [_error_block(filename, ErrorReason.EXTRACTION_FAILED)]
        except ExtractionError as exc:
            # Any other ExtractionError subclass we forgot to enumerate.
            logger.warning("unexpected extraction error for %r: %s", filename, exc)
            return [_error_block(filename, ErrorReason.EXTRACTION_FAILED)]
