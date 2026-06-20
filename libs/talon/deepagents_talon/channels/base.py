"""Reusable channel policy and formatting helpers.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import fnmatch
import mimetypes
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_talon.interfaces import ChannelMedia, ChannelMessage
from deepagents_talon.media import resolve_bounded_media_path

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MAX_TEXT_CHARS = 4096
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_VIDEO_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_MEDIA_BYTES = 1024 * 1024 * 1024
MAX_MEDIA_BYTES_ENV = "DEEPAGENTS_TALON_MAX_MEDIA_BYTES"
OPEN_EXPOSURE_ACK_VALUE = "allow-arbitrary-senders"

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
    """Raised when channel media cannot be handled safely."""


@dataclass(frozen=True, slots=True)
class ChannelExposureEnv:
    """Environment variable names used to build channel exposure policy.

    Args:
        provider: Human-readable provider name used in error messages.
        exposure: Environment variable containing the exposure mode.
        allowlist_chats: Environment variable containing allowlisted chats.
        mention_patterns: Environment variable containing allowlist text patterns.
        operator_id: Environment variable containing trusted operator ids.
        open_ack: Environment variable acknowledging open-exposure risk.
        open_ack_value: Required acknowledgement value for open exposure.
        require_self_operator: Whether `self` exposure requires an operator id.
    """

    provider: str
    exposure: str
    allowlist_chats: str
    mention_patterns: str
    operator_id: str
    open_ack: str
    open_ack_value: str = OPEN_EXPOSURE_ACK_VALUE
    require_self_operator: bool = False


@dataclass(frozen=True, slots=True)
class FilteredMediaPaths:
    """Inbound media paths retained after applying local size caps.

    Args:
        paths: String paths that remain usable.
        mime_types: MIME types aligned with `paths`.
        changed: Whether any input path was dropped or ignored.
        errors: Human-readable drop reasons for logs or metadata.
    """

    paths: tuple[str, ...]
    mime_types: tuple[str, ...]
    changed: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChannelExposure:
    """Inbound exposure policy shared by channel adapters.

    Args:
        mode: Trigger policy for inbound messages.
        operator_id: Channel-specific id for the operator's own account. Preserved
            for callers that configure a single operator.
        conversations: Conversation ids allowed in allowlist mode.
        mention_patterns: Glob-style patterns that may allow a message by text.
        operator_ids: Channel-specific ids for operator accounts that may trigger
            `self` exposure.
    """

    mode: ExposureMode = ExposureMode.SELF
    operator_id: str | None = None
    conversations: frozenset[str] = field(default_factory=frozenset)
    mention_patterns: tuple[str, ...] = ()
    operator_ids: frozenset[str] = field(default_factory=frozenset, kw_only=True)

    def __post_init__(self) -> None:
        """Normalize the legacy single-operator field into the operator id set."""
        operator_ids = set(self.operator_ids)
        if self.operator_id is not None:
            operator_ids.add(self.operator_id)
        object.__setattr__(self, "operator_ids", frozenset(operator_ids))

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
            return _is_self_message(message, self.operator_ids)
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


def channel_exposure_from_env(
    env: Mapping[str, str],
    config: ChannelExposureEnv,
) -> ChannelExposure:
    """Build shared channel exposure policy from provider-specific env names.

    Args:
        env: Environment variable mapping.
        config: Provider-specific exposure environment mapping.

    Returns:
        Parsed exposure policy.

    Raises:
        ValueError: If the exposure mode is invalid or risk acknowledgement is missing.
    """
    mode = _exposure_mode(
        env.get(config.exposure, ExposureMode.SELF.value),
        provider=config.provider,
    )
    operator_ids = tuple(split_csv(env.get(config.operator_id, "")))
    operator_id = operator_ids[0] if operator_ids else None
    if mode == ExposureMode.SELF and config.require_self_operator and not operator_ids:
        msg = (
            f"{config.provider} self exposure requires {config.operator_id}; "
            f"set {config.exposure}=allowlist or open for other modes"
        )
        raise ValueError(msg)
    if mode == ExposureMode.OPEN:
        _require_open_acknowledgement(env, config)
    return ChannelExposure(
        mode=mode,
        operator_id=operator_id,
        conversations=frozenset(split_csv(env.get(config.allowlist_chats, ""))),
        mention_patterns=tuple(split_csv(env.get(config.mention_patterns, ""))),
        operator_ids=frozenset(operator_ids),
    )


def validate_media(
    media: ChannelMedia,
    *,
    root: Path | None = None,
    max_bytes: int | None = None,
) -> ChannelMedia:
    """Validate outbound media path, type, and size.

    Args:
        media: Media payload to validate.
        root: Optional directory that must contain the media after symlink
            resolution.
        max_bytes: Optional global media size cap.

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

    return _validate_media_size(media, path=path, max_bytes=max_bytes)


def replace_message_metadata(
    message: ChannelMessage,
    metadata: Mapping[str, object],
) -> ChannelMessage:
    """Return `message` with replaced metadata.

    Args:
        message: Original channel message.
        metadata: Replacement metadata.

    Returns:
        Channel message with the same identity and text, plus new metadata.
    """
    return ChannelMessage(
        conversation_id=message.conversation_id,
        text=message.text,
        sender_id=message.sender_id,
        message_id=message.message_id,
        metadata=metadata,
    )


def message_with_media_paths(
    message: ChannelMessage,
    *,
    media_paths: Sequence[str],
    mime_types: Sequence[str] = (),
) -> ChannelMessage:
    """Return `message` with normalized inbound-media path metadata.

    Args:
        message: Original channel message.
        media_paths: Local media paths associated with the message.
        mime_types: MIME types aligned with `media_paths`.

    Returns:
        Channel message with standard media path metadata.
    """
    paths = list(media_paths)
    types = list(mime_types)
    metadata = dict(message.metadata)
    metadata["media_paths"] = paths
    metadata["media_path"] = paths[0] if paths else None
    metadata["media_mime_types"] = types
    metadata["media_types"] = types
    metadata["voice_path"] = paths[0] if paths and metadata.get("media_type") == "voice" else None
    metadata["has_media"] = bool(paths)
    return replace_message_metadata(message, metadata)


def filter_capped_media_paths(
    media_paths: Sequence[object],
    mime_types: Sequence[object],
    *,
    max_bytes: int,
) -> FilteredMediaPaths:
    """Filter inbound local media paths against a configured size cap.

    Args:
        media_paths: Candidate local media paths from a channel payload.
        mime_types: Candidate MIME types aligned with `media_paths`.
        max_bytes: Maximum allowed file size.

    Returns:
        Filter result containing kept paths and dropped-path reasons.
    """
    kept: list[str] = []
    kept_mime_types: list[str] = []
    errors: list[str] = []
    changed = False
    for index, raw_path in enumerate(media_paths):
        if not isinstance(raw_path, str):
            changed = True
            continue
        try:
            validate_media_size(Path(raw_path), max_bytes=max_bytes)
        except FileNotFoundError:
            kept.append(raw_path)
        except ChannelMediaError as error:
            changed = True
            errors.append(str(error))
            continue
        else:
            kept.append(raw_path)
        if index < len(mime_types):
            mime_type = mime_types[index]
            if isinstance(mime_type, str):
                kept_mime_types.append(mime_type)
    return FilteredMediaPaths(
        paths=tuple(kept),
        mime_types=tuple(kept_mime_types),
        changed=changed,
        errors=tuple(errors),
    )


def validate_media_size(path: Path, *, max_bytes: int) -> None:
    """Validate a local media file against the configured global cap.

    Args:
        path: Local media file to inspect.
        max_bytes: Maximum allowed media file size.

    Raises:
        ChannelMediaError: If the file exceeds `max_bytes`.
    """
    size = path.stat().st_size
    if size > max_bytes:
        msg = f"media file is too large: {size} bytes exceeds {max_bytes}"
        raise ChannelMediaError(msg)


def max_media_bytes_from_env(env: Mapping[str, str]) -> int:
    """Return the configured global media cap.

    Args:
        env: Environment variable mapping.

    Returns:
        Maximum media bytes allowed for channel media.

    Raises:
        ValueError: If the configured value is not a positive integer.
    """
    value = env.get(MAX_MEDIA_BYTES_ENV)
    if value is None:
        return DEFAULT_MAX_MEDIA_BYTES
    try:
        parsed = int(value)
    except ValueError as error:
        msg = f"{MAX_MEDIA_BYTES_ENV} must be a positive integer byte count"
        raise ValueError(msg) from error
    if parsed < 1:
        msg = f"{MAX_MEDIA_BYTES_ENV} must be a positive integer byte count"
        raise ValueError(msg)
    return parsed


def parse_float(value: str | None, default: float) -> float:
    """Parse an optional float value with a default.

    Args:
        value: Raw environment value.
        default: Value returned when `value` is missing.

    Returns:
        Parsed float.

    Raises:
        ValueError: If `value` is not a float.
    """
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as error:
        msg = f"expected float value, got {value!r}"
        raise ValueError(msg) from error


def parse_int(value: str | None, default: int) -> int:
    """Parse an optional integer value with a default.

    Args:
        value: Raw environment value.
        default: Value returned when `value` is missing.

    Returns:
        Parsed integer.

    Raises:
        ValueError: If `value` is not an integer.
    """
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        msg = f"expected integer value, got {value!r}"
        raise ValueError(msg) from error


def split_csv(value: str) -> list[str]:
    """Split a comma-separated environment value.

    Args:
        value: Raw comma-separated value.

    Returns:
        Non-empty, stripped items.
    """
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_media_size(
    media: ChannelMedia,
    *,
    path: Path,
    max_bytes: int | None = None,
) -> ChannelMedia:
    limit = MAX_IMAGE_BYTES if media.media_type == "image" else MAX_VIDEO_BYTES
    if max_bytes is not None:
        limit = min(limit, max_bytes)
    size = path.stat().st_size
    if size > limit:
        msg = f"{media.media_type} media is too large: {size} bytes exceeds {limit}"
        raise ChannelMediaError(msg)

    return ChannelMedia(path=path, media_type=media.media_type, caption=media.caption)


def _is_self_message(message: ChannelMessage, operator_ids: frozenset[str]) -> bool:
    if message.metadata.get("from_self") is True:
        return True
    return message.sender_id is not None and message.sender_id in operator_ids


def _matches_text(text: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(text, pattern) for pattern in patterns)


def _exposure_mode(value: str, *, provider: str) -> ExposureMode:
    try:
        return ExposureMode(value)
    except ValueError as error:
        modes = ", ".join(mode.value for mode in ExposureMode)
        msg = f"invalid {provider} exposure mode {value!r}; expected one of: {modes}"
        raise ValueError(msg) from error


def _require_open_acknowledgement(
    env: Mapping[str, str],
    config: ChannelExposureEnv,
) -> None:
    if env.get(config.open_ack) == config.open_ack_value:
        return
    msg = (
        f"{config.provider} exposure mode 'open' allows arbitrary senders to trigger the "
        "agent with operator credentials and local host access; set "
        f"{config.open_ack}={config.open_ack_value} to acknowledge this risk"
    )
    raise ValueError(msg)


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
