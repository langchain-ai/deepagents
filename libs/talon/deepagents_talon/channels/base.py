"""Reusable channel policy and formatting helpers.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import mimetypes
import re
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from deepagents_talon.interfaces import ChannelMedia, ChannelMessage, MessageHandler, SendResult
from deepagents_talon.media import resolve_bounded_media_path

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

_T = TypeVar("_T")

MAX_TEXT_CHARS = 4096
DEFAULT_MAX_MEDIA_BYTES = 1024 * 1024 * 1024
MAX_MEDIA_BYTES_ENV = "DEEPAGENTS_TALON_MAX_MEDIA_BYTES"
OPEN_EXPOSURE_ACK_VALUE = "allow-arbitrary-senders"
OUTBOUND_MEDIA_DIR_ENV = "DEEPAGENTS_TALON_OUTBOUND_MEDIA_DIR"
WORKSPACE_ENV = "DEEPAGENTS_TALON_WORKSPACE"

ASR_ELIGIBLE_MEDIA_TYPES = frozenset({"voice", "video"})
"""Media types that may contain audio eligible for ASR transcription."""

_LINK_PATTERN = re.compile(r"\[([^\]]+)]\(([^)]+)\)")
_HEADING_PATTERN = re.compile(r"^#{1,6}\s+", flags=re.MULTILINE)
_BOLD_PATTERN = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")
_ITALIC_PATTERN = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)|_([^_\n]+)_")

logger = logging.getLogger(__name__)


class ExposureMode(StrEnum):
    """Who may trigger a channel-backed agent."""

    SELF = "self"
    ALLOWLIST = "allowlist"
    OPEN = "open"


class ChannelMediaError(ValueError):
    """Raised when channel media cannot be handled safely."""


@dataclass(frozen=True, slots=True)
class ChannelExposureEnv:
    """Environment variable prefix and options for channel exposure policy."""

    provider: str
    """Human-readable provider name used in error messages."""

    env_prefix: str
    """Prefix for exposure env vars (e.g. `DEEPAGENTS_TALON_TELEGRAM`)."""

    open_ack: str
    """Environment variable acknowledging open-exposure risk."""

    open_ack_value: str = OPEN_EXPOSURE_ACK_VALUE
    """Required acknowledgement value for open exposure."""

    require_self_operator: bool = False
    """Whether `self` exposure requires an operator id."""


@dataclass(frozen=True, slots=True)
class ChannelExposure:
    """Inbound exposure policy shared by channel adapters."""

    mode: ExposureMode = ExposureMode.SELF
    """Trigger policy for inbound messages."""

    conversations: frozenset[str] = field(default_factory=frozenset)
    """Conversation ids allowed in allowlist mode."""

    mention_patterns: tuple[str, ...] = ()
    """Glob-style patterns that may allow a message by text."""

    operator_ids: frozenset[str] = field(default_factory=frozenset)
    """Channel-specific ids for operator accounts that may trigger `self` exposure."""

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


async def dispatch_message(
    handler: MessageHandler | None,
    message: ChannelMessage,
    *,
    provider: str,
) -> None:
    """Dispatch an inbound message to the registered handler.

    Args:
        handler: Host callback for inbound messages, or ``None``.
        message: Channel message to dispatch.
        provider: Provider name for log messages.

    Raises:
        AssertionError: If no handler is registered (internal programming error).
    """
    if handler is None:
        logger.warning("Dropping %s message because no handler is registered", provider)
        return
    await handler(message)


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
    """Build shared channel exposure policy from provider-specific env prefix.

    Args:
        env: Environment variable mapping.
        config: Provider-specific exposure environment mapping.

    Returns:
        Parsed exposure policy.

    Raises:
        ValueError: If the exposure mode is invalid or risk acknowledgement is missing.
    """
    prefix = config.env_prefix
    exposure_var = f"{prefix}_EXPOSURE"
    operator_var = f"{prefix}_OPERATOR_ID"
    mode = _exposure_mode(
        env.get(exposure_var, ExposureMode.SELF.value),
        provider=config.provider,
    )
    operator_ids = frozenset(split_csv(env.get(operator_var, "")))
    if mode == ExposureMode.SELF and config.require_self_operator and not operator_ids:
        msg = (
            f"{config.provider} self exposure requires {operator_var}; "
            f"set {exposure_var}=allowlist or open for other modes"
        )
        raise ValueError(msg)
    if mode == ExposureMode.OPEN:
        _require_open_acknowledgement(env, config)
        logger.warning(
            "%s open exposure enabled; arbitrary senders can trigger the agent with "
            "operator credentials and local host access",
            config.provider,
        )
    return ChannelExposure(
        mode=mode,
        conversations=frozenset(split_csv(env.get(f"{prefix}_ALLOWLIST_CHATS", ""))),
        mention_patterns=tuple(split_csv(env.get(f"{prefix}_MENTION_PATTERNS", ""))),
        operator_ids=operator_ids,
    )


def outbound_media_root_from_env(env: Mapping[str, str]) -> Path:
    """Return the trusted outbound media root for channel attachments.

    Args:
        env: Environment variable mapping.

    Returns:
        Configured outbound media directory, workspace directory, or cwd.
    """
    raw = env.get(OUTBOUND_MEDIA_DIR_ENV) or env.get(WORKSPACE_ENV)
    if raw:
        return Path(raw).expanduser()
    return Path.cwd()


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


def message_with_media_paths(
    message: ChannelMessage,
    *,
    media_paths: Sequence[str],
    mime_types: Sequence[str] = (),
    has_media: bool | None = None,
) -> ChannelMessage:
    """Return `message` with normalized inbound-media path metadata.

    Args:
        message: Original channel message.
        media_paths: Local media paths associated with the message.
        mime_types: MIME types aligned with `media_paths`.
        has_media: Optional provider-reported media presence. When omitted, this
            is inferred from `media_paths`.

    Returns:
        Channel message with standard media path metadata.
    """
    paths = list(media_paths)
    types = list(mime_types)
    metadata = dict(message.metadata)
    has_media_value = bool(paths) if has_media is None else has_media
    metadata["has_media"] = has_media_value
    if paths:
        metadata["media_paths"] = paths
        metadata["media_path"] = paths[0]
        metadata["media_mime_types"] = types
        if metadata.get("media_type") == "voice":
            metadata["voice_path"] = paths[0]
        else:
            metadata.pop("voice_path", None)
    return replace(message, metadata=metadata)


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
    msg = f"{MAX_MEDIA_BYTES_ENV} must be a positive integer byte count"
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(msg) from error
    if parsed < 1:
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
    return _parse_number(float, value, default, label="float")


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
    return _parse_number(int, value, default, label="integer")


def _parse_number(
    convert: Callable[[str], _T],
    value: str | None,
    default: _T,
    *,
    label: str,
) -> _T:
    if value is None:
        return default
    try:
        return convert(value)  # type: ignore[return-value]
    except ValueError as error:
        msg = f"expected {label} value, got {value!r}"
        raise ValueError(msg) from error


def split_csv(value: str) -> list[str]:
    """Split a comma-separated environment value.

    Args:
        value: Raw comma-separated value.

    Returns:
        Non-empty, stripped items.
    """
    return [item.strip() for item in value.split(",") if item.strip()]


def optional_str(value: object) -> str | None:
    """Return a non-empty string value, or ``None``.

    Args:
        value: Raw value that may be a string.

    Returns:
        The string if it is a non-empty ``str``, otherwise ``None``.
    """
    return value if isinstance(value, str) and value else None


def _validate_media_size(
    media: ChannelMedia,
    *,
    path: Path,
    max_bytes: int | None = None,
) -> ChannelMedia:
    if max_bytes is None:
        return ChannelMedia(path=path, media_type=media.media_type, caption=media.caption)
    size = path.stat().st_size
    if size > max_bytes:
        msg = f"{media.media_type} media is too large: {size} bytes exceeds {max_bytes}"
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


_RETRYABLE_ERROR_FRAGMENTS = frozenset(
    {
        "connectionerror",
        "connectionreset",
        "connectionrefused",
        "connecttimeout",
        "timeout",
        "broken pipe",
        "remotedisconnected",
        "eoferror",
        "network",
        "transient",
    }
)


def _is_retryable_error(error: str | None) -> bool:
    """Return whether an error message looks like a transient network failure.

    Args:
        error: Error string from a failed send operation.

    Returns:
        `True` when the error text matches a known transient-network pattern.
    """
    if error is None:
        return False
    lowered = error.lower()
    return any(fragment in lowered for fragment in _RETRYABLE_ERROR_FRAGMENTS)


async def send_with_retry(
    send_fn: Callable[[], Awaitable[SendResult | None]],
    *,
    max_retries: int = 2,
    base_delay: float = 2.0,
) -> SendResult:
    """Call a channel send function with automatic retry on transient errors.

    Exceptions raised by ``send_fn`` are caught and converted to failed
    `SendResult` objects so that transport-level failures remain non-fatal
    to the caller.  Retryable exceptions (matching the same transient-network
    patterns used for `SendResult.error`) are retried like any other
    retryable failure.

    Args:
        send_fn: Zero-argument coroutine that performs the actual send.
        max_retries: Maximum number of retry attempts after the first failure.
        base_delay: Base delay in seconds for exponential backoff.

    Returns:
        The final `SendResult` from the send function.
    """
    result = _normalize_send_result(await _safe_send(send_fn))
    if result.success:
        return result
    if not (result.retryable or _is_retryable_error(result.error)):
        return result
    for attempt in range(1, max_retries + 1):
        delay = base_delay * (2 ** (attempt - 1))
        await asyncio.sleep(delay)
        result = _normalize_send_result(await _safe_send(send_fn))
        if result.success:
            return result
        if not (result.retryable or _is_retryable_error(result.error)):
            return result
    return result


async def _safe_send(send_fn: Callable[[], Awaitable[SendResult | None]]) -> SendResult | None:
    """Call ``send_fn`` and convert exceptions to failed `SendResult` objects.

    Args:
        send_fn: Zero-argument coroutine that performs the actual send.

    Returns:
        The `SendResult` from ``send_fn``, or a failed `SendResult` when an
        exception was raised.
    """
    try:
        return await send_fn()
    except Exception as exc:  # noqa: BLE001  # transport errors must not crash the host loop
        return SendResult(success=False, error=str(exc) or repr(exc), retryable=True)


def _normalize_send_result(result: SendResult | None) -> SendResult:
    """Normalize a send result, treating ``None`` as success for legacy adapters.

    Args:
        result: Return value from a channel send method, or ``None``.

    Returns:
        The original result, or a success result when the adapter returned ``None``.
    """
    if result is None:
        return SendResult(success=True)
    return result
