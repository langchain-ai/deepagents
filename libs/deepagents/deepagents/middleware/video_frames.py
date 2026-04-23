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

This module currently provides only the class initializer and a filename helper;
`wrap_model_call` and the extraction pipeline are implemented in follow-up tasks.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from deepagents.middleware._ffmpeg import ExtractedFrame, ExtractionParams


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


class VideoFrameExtractionMiddleware:
    """Middleware that transcodes video blocks to frame sequences for non-video-capable models.

    See `docs/superpowers/specs/2026-04-23-video-frame-extraction-design.md` for
    the full design.

    Args:
        max_frames: Hard cap on frames per video. Default 95 (Claude 100-image
            headroom).
        scene_threshold: FFmpeg `scene` threshold for adaptive frame selection.
        max_width: Longest image edge in pixels. Aspect ratio preserved.
        jpeg_quality: FFmpeg `-q:v` (2 = best, 31 = worst).
        video_capable_override: `None` to auto-detect via registry, `True` to
            always skip extraction, `False` to always extract.
        cache_size: Maximum distinct videos kept in the instance LRU cache.
            Zero disables caching.

    Raises:
        ValueError: If any parameter is out of its allowed range.
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
