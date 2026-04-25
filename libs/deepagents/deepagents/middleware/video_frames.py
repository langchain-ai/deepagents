"""Middleware that transcodes VideoContentBlocks into text-interleaved image sequences.

The middleware fires as a `wrap_model_call` hook. For non-video-capable providers
(everything except Gemini by default), it:

1. Deep-copies `request.messages`.
2. Replaces every `VideoContentBlock` with a `[preamble, (ts, image) x N]`
   sequence by shelling out to ffmpeg.
3. Calls `handler(request.override(messages=transformed))`.

Persisted state is never mutated. A per-instance LRU cache keyed by
`(sha256(video_bytes), extraction_params)` ensures a given video is extracted
at most once per process.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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

# Valid range constants for VideoFrameExtractionMiddleware constructor.
_MAX_FRAMES_MIN = 1
_MAX_FRAMES_MAX = 1000
_MAX_WIDTH_MIN = 64
_MAX_WIDTH_MAX = 4096
_JPEG_QUALITY_MIN = 2
_JPEG_QUALITY_MAX = 31
_CACHE_SIZE_MAX = 10000
# Minimum frame count that enables duration/interval estimation.
_MIN_FRAMES_FOR_DURATION = 2

# Per-provider images-per-request hard caps, sourced from docs as of 2026-04:
# - Anthropic (200K-context models like Claude Sonnet): 100/request.
#   https://platform.claude.com/docs/en/docs/build-with-claude/vision
# - OpenAI: 1500/request.
#   https://developers.openai.com/api/docs/guides/images-vision
# Effective `max_frames` is clamped to the provider's cap at request time.
_PROVIDER_FRAME_CAPS: dict[str, int] = {
    "anthropic": 100,
    "openai": 1500,
}
# Conservative fallback when the provider is unknown.
_DEFAULT_PROVIDER_FRAME_CAP = 100

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


def _is_video_block(block: object) -> bool:
    if not isinstance(block, dict):
        return False
    # ty narrows `dict` to `dict[Unknown, Unknown]`; `.get(str)` triggers
    # invalid-argument-type because the key is typed as `Never`.
    return block.get("type") == "video"  # type: ignore[arg-type]


def _maybe_parse_stringified_blocks(content: str) -> list[Any] | str:
    """Recover a content-block list that was JSON-stringified by LangGraph's ToolNode.

    Why: `langgraph.prebuilt.tool_node.msg_content_output` routes tool output
    through `json.dumps` when any block's `type` isn't in
    `TOOL_MESSAGE_BLOCK_TYPES` (notably `"video"` and `"audio"`). That turns
    `read_file` video output into a giant string that silently ships the raw
    base64 to the model. We parse it back so the normal transformation runs.

    Returns the parsed list on success; otherwise the original string.
    """
    # Cheap prefilter so we don't JSON-parse unrelated strings.
    if not content.startswith("[") or '"type": "video"' not in content:
        return content
    try:
        parsed = json.loads(content)
    except ValueError:
        return content
    if isinstance(parsed, list) and all(isinstance(b, dict) for b in parsed):
        return parsed
    return content


def _decode_video_bytes(block: dict[str, Any]) -> bytes:
    data = block.get("base64")
    if not isinstance(data, str):
        msg = "video block missing 'base64'"
        raise ExtractionFailedError(msg)
    try:
        return base64.b64decode(data, validate=True)
    except binascii.Error as exc:
        msg = "video block 'base64' is not valid base64"
        raise ExtractionFailedError(msg) from exc


def _ext_for_mime(mime: str | None) -> str:
    if mime and mime in _MIME_TO_EXT:
        return _MIME_TO_EXT[mime]
    return "mp4"


def _run_extraction(video_bytes: bytes, mime: str | None, params: ExtractionParams) -> tuple[list[ExtractedFrame], float]:
    """Write the bytes to a temp file and invoke `_ffmpeg.extract_frames`.

    Seam for monkeypatching in middleware tests.

    Returns:
        A tuple `(frames, duration_s)` as returned by `_ffmpeg.extract_frames`.
    """
    ext = _ext_for_mime(mime)
    with NamedTemporaryFile(suffix=f".{ext}", delete=True) as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        return _ffmpeg.extract_frames(Path(tmp.name), params)


def _cache_key(video_bytes: bytes, params: ExtractionParams) -> tuple[str, tuple[int, float, int, int]]:
    digest = hashlib.sha256(video_bytes).hexdigest()
    params_tuple = (
        params.max_frames,
        params.scene_threshold,
        params.max_width,
        params.jpeg_quality,
    )
    return digest, params_tuple


def _build_preamble(filename: str, num_frames: int, duration_s: float, baseline_interval: float) -> dict[str, Any]:
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


def _frames_to_blocks(frames: list[ExtractedFrame], filename: str, duration_s: float, baseline_interval: float) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [_build_preamble(filename, len(frames), duration_s, baseline_interval)]
    for i, frame in enumerate(frames):
        blocks.append({"type": "text", "text": f"Frame {i + 1} at t={frame.timestamp_s:.1f}s"})
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
        # User-facing upper bound. Effective value is clamped per-provider at
        # request time using `_PROVIDER_FRAME_CAPS` (Anthropic 100, OpenAI 1500
        # as of 2026-04), so 200 gives more frames on OpenAI while still being
        # safe on Anthropic (where it clamps to 100).
        max_frames: int = 200,
        scene_threshold: float = 0.30,
        max_width: int = 512,
        jpeg_quality: int = 5,
        video_capable_override: bool | None = None,
        cache_size: int = 256,
    ) -> None:
        """Initialise the middleware with extraction and caching parameters.

        Args:
            max_frames: Maximum number of frames to extract per video (1-1000).
            scene_threshold: Scene-change sensitivity in [0.0, 1.0].
            max_width: Maximum frame width in pixels (64-4096).
            jpeg_quality: JPEG quality factor passed to ffmpeg's `-q:v` (2-31;
                lower is higher quality).
            video_capable_override: Force video-capable detection on (True), off
                (False), or use the registry (None).
            cache_size: Maximum number of videos to keep in the extraction cache.
                0 disables caching.

        Raises:
            ValueError: If any parameter is outside its valid range.
        """
        if not _MAX_FRAMES_MIN <= max_frames <= _MAX_FRAMES_MAX:
            msg = f"max_frames must be in [{_MAX_FRAMES_MIN}, {_MAX_FRAMES_MAX}]; got {max_frames}"
            raise ValueError(msg)
        if not 0.0 <= scene_threshold <= 1.0:
            msg = f"scene_threshold must be in [0.0, 1.0]; got {scene_threshold}"
            raise ValueError(msg)
        if not _MAX_WIDTH_MIN <= max_width <= _MAX_WIDTH_MAX:
            msg = f"max_width must be in [{_MAX_WIDTH_MIN}, {_MAX_WIDTH_MAX}]; got {max_width}"
            raise ValueError(msg)
        if not _JPEG_QUALITY_MIN <= jpeg_quality <= _JPEG_QUALITY_MAX:
            msg = f"jpeg_quality must be in [{_JPEG_QUALITY_MIN}, {_JPEG_QUALITY_MAX}]; got {jpeg_quality}"
            raise ValueError(msg)
        if not 0 <= cache_size <= _CACHE_SIZE_MAX:
            msg = f"cache_size must be in [0, {_CACHE_SIZE_MAX}]; got {cache_size}"
            raise ValueError(msg)

        super().__init__()
        self.max_frames = max_frames
        self.scene_threshold = scene_threshold
        self.max_width = max_width
        self.jpeg_quality = jpeg_quality
        self.video_capable_override = video_capable_override
        self.cache_size = cache_size

        self._cache: OrderedDict[tuple[str, tuple[int, float, int, int]], tuple[list[ExtractedFrame], float]] = OrderedDict()

    @property
    def _params(self) -> ExtractionParams:
        """Configured params (no provider clamp). Use `_params_for_provider` for live calls."""
        return ExtractionParams(
            max_frames=self.max_frames,
            scene_threshold=self.scene_threshold,
            max_width=self.max_width,
            jpeg_quality=self.jpeg_quality,
        )

    def _params_for_provider(self, provider: str | None) -> ExtractionParams:
        """Return params with `max_frames` clamped to the provider's documented cap."""
        cap = _PROVIDER_FRAME_CAPS.get((provider or "").lower(), _DEFAULT_PROVIDER_FRAME_CAP)
        return ExtractionParams(
            max_frames=min(self.max_frames, cap),
            scene_threshold=self.scene_threshold,
            max_width=self.max_width,
            jpeg_quality=self.jpeg_quality,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        """Intercept a model call and expand video blocks for non-video-capable models."""
        provider = get_model_provider(request.model)
        model_name = get_model_identifier(request.model) or ""
        if is_video_capable(provider, model_name, override=self.video_capable_override):
            return handler(request)

        params = self._params_for_provider(provider)
        transformed, mutated = self._transform_messages(request.messages, params)
        if not mutated:
            return handler(request)
        return handler(request.override(messages=transformed))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> ModelResponse | AIMessage:
        """Async variant of `wrap_model_call`; reuses the sync `_transform_messages` helper."""
        provider = get_model_provider(request.model)
        model_name = get_model_identifier(request.model) or ""
        if is_video_capable(provider, model_name, override=self.video_capable_override):
            return await handler(request)

        params = self._params_for_provider(provider)
        transformed, mutated = await asyncio.to_thread(self._transform_messages, request.messages, params)
        if not mutated:
            return await handler(request)
        return await handler(request.override(messages=transformed))

    def _transform_messages(
        self,
        messages: list[AnyMessage],
        params: ExtractionParams | None = None,
    ) -> tuple[list[AnyMessage], bool]:
        """Return (new_messages, whether_any_video_was_found).

        `params` lets callers override `self._params` per request (used to
        clamp `max_frames` to the active provider's cap). When `None`, the
        configured params are used; this preserves the legacy call signature
        for tests and external callers.
        """
        effective_params = params if params is not None else self._params
        mutated = False
        new_messages = copy.deepcopy(list(messages))
        for message in new_messages:
            content = getattr(message, "content", None)
            # LangGraph's ToolNode JSON-stringifies tool output when any content
            # block type isn't in `TOOL_MESSAGE_BLOCK_TYPES` ("video" isn't).
            # Recover the original list so we can still find and expand videos.
            if isinstance(content, str):
                content = _maybe_parse_stringified_blocks(content)
            if not isinstance(content, list):
                continue
            message_mutated = False
            new_content: list[Any] = []
            for block in content:
                if _is_video_block(block):
                    mutated = True
                    message_mutated = True
                    new_content.extend(self._expand_video_block(block, effective_params))
                else:
                    new_content.append(block)
            if message_mutated:
                message.content = new_content
        return new_messages, mutated

    def _expand_video_block(
        self,
        block: dict[str, Any],
        params: ExtractionParams,
    ) -> list[dict[str, Any]]:
        filename = _filename_for_video_block(block)
        try:
            video_bytes = _decode_video_bytes(block)
            frames, duration_s = self._extract_with_cache(video_bytes, block.get("mime_type"), params)
            baseline_interval = max(1.0, duration_s / params.max_frames)
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

    def _extract_with_cache(
        self,
        video_bytes: bytes,
        mime: str | None,
        params: ExtractionParams | None = None,
    ) -> tuple[list[ExtractedFrame], float]:
        effective_params = params if params is not None else self._params
        if self.cache_size == 0:
            return _run_extraction(video_bytes, mime, effective_params)

        key = _cache_key(video_bytes, effective_params)
        cached = self._cache.get(key)
        if cached is not None:
            # LRU: move to MRU end.
            self._cache.move_to_end(key)
            return cached

        result = _run_extraction(video_bytes, mime, effective_params)
        self._cache[key] = result
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)  # Evict LRU.
        return result
