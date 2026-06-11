"""Reusable channel policy and formatting helpers.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import fnmatch
import mimetypes
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from deepagents_talon.interfaces import ChannelMedia, ChannelMessage
from deepagents_talon.media import resolve_bounded_media_path

if TYPE_CHECKING:
    from pathlib import Path

MAX_TEXT_CHARS = 4096
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_VIDEO_BYTES = 64 * 1024 * 1024

_LINK_PATTERN = re.compile(r"\[([^\]]+)]\(([^)]+)\)")
_HEADING_PATTERN = re.compile(r"^#{1,6}\s+", flags=re.MULTILINE)
_BOLD_PATTERN = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")
_ITALIC_PATTERN = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)|_([^_\n]+)_")


class ExposureMode(StrEnum):
    """Who may trigger a channel-backed agent."""

    SELF = "self"
    ALLOWLIST = "allowlist"
    OPEN = "open"


class ChannelMediaError(ValueError):
    """Raised when outbound media cannot be sent safely."""


@dataclass(frozen=True, slots=True)
class ChannelExposure:
    """Inbound exposure policy shared by channel adapters.

    Args:
        mode: Trigger policy for inbound messages.
        operator_id: Channel-specific id for the operator's own account.
        conversations: Conversation ids allowed in allowlist mode.
        mention_patterns: Glob-style patterns that may allow a message by text.
    """

    mode: ExposureMode = ExposureMode.SELF
    operator_id: str | None = None
    conversations: frozenset[str] = field(default_factory=frozenset)
    mention_patterns: tuple[str, ...] = ()

    def allows(self, message: ChannelMessage) -> bool:
        """Return whether an inbound message may trigger the agent.

        Args:
            message: Inbound message from a channel adapter.

        Returns:
            `True` when the message passes this exposure policy.
        """
        if self.mode == ExposureMode.OPEN:
            return True
        if self.mode == ExposureMode.SELF:
            return _is_self_message(message, self.operator_id)
        return message.conversation_id in self.conversations or _matches_text(
            message.text,
            self.mention_patterns,
        )


def format_markdown_for_channel(text: str) -> str:
    """Convert common Markdown into conservative WhatsApp-compatible text.

    Args:
        text: Markdown text returned by the agent.

    Returns:
        Text with common Markdown constructs mapped to WhatsApp formatting.
    """
    value = _HEADING_PATTERN.sub("", text)
    value = _LINK_PATTERN.sub(r"\1 (\2)", value)
    value = _ITALIC_PATTERN.sub(lambda match: f"_{match.group(1) or match.group(2)}_", value)
    return _BOLD_PATTERN.sub(lambda match: f"*{match.group(1) or match.group(2)}*", value)


def chunk_text(text: str, *, limit: int = MAX_TEXT_CHARS) -> list[str]:
    """Split outbound text into channel-sized chunks.

    Args:
        text: Text to split.
        limit: Maximum characters per returned chunk.

    Returns:
        Non-empty chunks no longer than `limit`.

    Raises:
        ValueError: If `limit` is not positive.
    """
    if limit < 1:
        msg = "chunk limit must be positive"
        raise ValueError(msg)

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split = _split_index(remaining, limit)
        chunk = remaining[:split].rstrip()
        chunks.append(chunk or remaining[:limit])
        remaining = remaining[split:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def validate_media(media: ChannelMedia, *, root: Path | None = None) -> ChannelMedia:
    """Validate outbound media path, type, and size.

    Args:
        media: Media payload to validate.
        root: Optional directory that must contain the media after symlink
            resolution.

    Returns:
        The validated media payload.

    Raises:
        ChannelMediaError: If the file is missing, unsupported, or too large.
    """
    try:
        path = (
            resolve_bounded_media_path(media.path, root, require_relative=False)
            if root is not None
            else media.path.expanduser()
        )
    except ValueError as exc:
        msg = str(exc)
        raise ChannelMediaError(msg) from exc
    if not path.is_file():
        msg = f"media file does not exist: {path}"
        raise ChannelMediaError(msg)

    detected = _media_type(path)
    if detected != media.media_type:
        msg = f"media file type {detected!r} does not match requested type {media.media_type!r}"
        raise ChannelMediaError(msg)

    limit = MAX_IMAGE_BYTES if media.media_type == "image" else MAX_VIDEO_BYTES
    size = path.stat().st_size
    if size > limit:
        msg = f"{media.media_type} media is too large: {size} bytes exceeds {limit}"
        raise ChannelMediaError(msg)

    return ChannelMedia(path=path, media_type=media.media_type, caption=media.caption)


def _is_self_message(message: ChannelMessage, operator_id: str | None) -> bool:
    if message.metadata.get("from_self") is True:
        return True
    return operator_id is not None and message.sender_id == operator_id


def _matches_text(text: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(text, pattern) for pattern in patterns)


def _split_index(text: str, limit: int) -> int:
    window = text[:limit]
    for delimiter in ("\n\n", "\n", " "):
        index = window.rfind(delimiter)
        if index > 0:
            return index + len(delimiter)
    return limit


def _media_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        msg = f"unsupported media file type: {path}"
        raise ChannelMediaError(msg)
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("video/"):
        return "video"
    msg = f"unsupported media mime type: {mime}"
    raise ChannelMediaError(msg)
