"""Middleware that strips non-text media blocks before the model call.

Why: Terminal-Bench tasks such as `code-from-image` hand the agent an image
file (e.g. `/app/code.png`). When the agent calls the filesystem `read_file`
tool on it, the tool returns the file as an *image content block*
(`{"type": "image", "base64": ..., "mime_type": "image/png"}` — see
`deepagents/middleware/filesystem.py`). A text-only model (e.g. Nemotron-3-Ultra
on Baseten) then rejects the request with:

    openai.BadRequestError: 400 - This model does not support multimodal
    (image/video/audio) inputs.

…which crashes the whole trial. A system-prompt rule telling the model not to
read images is unreliable (the model ignored it). This middleware fixes it
deterministically: it intercepts the messages on their way to the model and
replaces any image/audio/video content block with a short text placeholder that
also nudges the agent to inspect such files programmatically.

This is model-agnostic and source-agnostic: it catches media from `read_file`
or any other origin, and is a no-op when there are no media blocks (so it is
safe to leave enabled for multimodal models too — though you'd typically only
attach it for text-only models).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

# Content-block `type` values that denote non-text media.
_MEDIA_BLOCK_TYPES = frozenset(
    {"image", "image_url", "audio", "input_audio", "video", "media"}
)
# `mime_type` top-level categories we treat as non-text.
_MEDIA_MIME_PREFIXES = frozenset({"image", "audio", "video"})


def _is_media_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    if block.get("type") in _MEDIA_BLOCK_TYPES:
        return True
    mime = block.get("mime_type")
    return isinstance(mime, str) and mime.split("/", 1)[0] in _MEDIA_MIME_PREFIXES


def _placeholder(block: dict[str, Any]) -> dict[str, Any]:
    mime = block.get("mime_type") or block.get("type") or "binary"
    return {
        "type": "text",
        "text": (
            f"[non-text content omitted ({mime}): this model is text-only and cannot "
            "view images/audio/video. Inspect such files programmatically instead — "
            "e.g. Python (PIL/OpenCV/numpy), ffmpeg, tesseract/OCR, or `file`/`xxd` — "
            "and reason from the program, data, or rendered output that produced them.]"
        ),
    }


def _sanitize_content(content: Any) -> tuple[Any, bool]:
    """Return (possibly-rewritten content, changed?)."""
    if not isinstance(content, list):
        return content, False
    changed = False
    out: list[Any] = []
    for block in content:
        if _is_media_block(block):
            out.append(_placeholder(block))
            changed = True
        else:
            out.append(block)
    return out, changed


def _sanitize_messages(messages: list[Any]) -> list[Any]:
    new_messages: list[Any] = []
    for msg in messages:
        new_content, changed = _sanitize_content(getattr(msg, "content", None))
        if not changed:
            new_messages.append(msg)
            continue
        # Prefer a non-mutating copy (pydantic v2 BaseMessage); fall back to mutate.
        try:
            new_messages.append(msg.model_copy(update={"content": new_content}))
        except Exception:  # noqa: BLE001 - defensive; never let sanitization crash the run
            try:
                msg.content = new_content
            except Exception:  # noqa: BLE001
                pass
            new_messages.append(msg)
    return new_messages


class TextOnlyMediaMiddleware(AgentMiddleware):
    """Replace image/audio/video content blocks with text for text-only models."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        sanitized = _sanitize_messages(list(request.messages))
        return handler(request.override(messages=sanitized))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        sanitized = _sanitize_messages(list(request.messages))
        return await handler(request.override(messages=sanitized))
