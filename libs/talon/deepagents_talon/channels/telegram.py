"""Telegram channel adapter backed by the Bot API over urllib.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from deepagents_talon.channels.base import (
    MAX_TEXT_CHARS,
    ChannelExposure,
    ChannelMediaError,
    ExposureMode,
    chunk_text,
    validate_media,
)
from deepagents_talon.interfaces import ChannelMedia, ChannelMessage, ChannelStatus, MessageHandler

if TYPE_CHECKING:
    from collections.abc import Mapping

    from deepagents_talon.config import TalonConfig

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.telegram.org"
DEFAULT_POLL_TIMEOUT_SECONDS = 30.0
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 35.0
MAX_PHOTO_BYTES = 10 * 1024 * 1024
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
OPEN_EXPOSURE_ACK_ENV = "DEEPAGENTS_TALON_TELEGRAM_OPEN_ACK"
OPEN_EXPOSURE_ACK_VALUE = "allow-arbitrary-senders"
_OFFSET_FILENAME = "telegram_offset.json"

# MarkdownV2 special characters that must be escaped.
_MARKDOWNV2_SPECIAL = re.compile(r"([_*\[\]()~`>#=+\-|{}.!\\])")


class TelegramError(RuntimeError):
    """Raised when the Telegram Bot API reports or causes a transport error."""


@dataclass(frozen=True, slots=True)
class TelegramChannelConfig:
    """Configuration for the Telegram channel adapter.

    Args:
        bot_token: Telegram Bot API authentication token.
        session_dir: Directory for Telegram session state (offset file).
        inbound_media_dir: Directory where inbound media is downloaded.
        outbound_media_dir: Optional root that outbound media must remain under.
        api_base: Telegram Bot API base URL.
        exposure: Inbound trigger policy.
        poll_timeout_seconds: Long-polling timeout passed to getUpdates.
        poll_interval_seconds: Delay between getUpdates calls.
        request_timeout_seconds: Per-request HTTP timeout for Bot API calls.
        operator_id: Telegram user ID for the operator (self exposure mode).
    """

    bot_token: str = field(repr=False)
    session_dir: Path
    inbound_media_dir: Path | None = None
    outbound_media_dir: Path | None = None
    api_base: str = DEFAULT_API_BASE
    exposure: ChannelExposure = field(default_factory=ChannelExposure)
    poll_timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    operator_id: str | None = None

    @classmethod
    def from_talon_config(cls, config: TalonConfig) -> TelegramChannelConfig:
        """Build Telegram channel configuration from Talon environment values.

        Args:
            config: Talon process configuration.

        Returns:
            Telegram channel configuration.

        Raises:
            ValueError: If the bot token is missing or exposure config is invalid.
        """
        env = config.env
        token = env.get("DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN") or env.get("TELEGRAM_BOT_TOKEN")
        if not token:
            msg = (
                "Telegram bot token is required "
                "(DEEPAGENTS_TALON_TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN)"
            )
            raise ValueError(msg)
        session = Path(
            env.get("DEEPAGENTS_TALON_TELEGRAM_SESSION_DIR", str(config.channel_dir / "telegram")),
        )
        inbound_media_dir = Path(
            env.get(
                "DEEPAGENTS_TALON_TELEGRAM_MEDIA_DIR",
                str(config.inbound_media_dir / "telegram"),
            ),
        )
        outbound_media_dir = Path(
            env.get("DEEPAGENTS_TALON_OUTBOUND_MEDIA_DIR")
            or env.get("DEEPAGENTS_TALON_WORKSPACE")
            or "/workspace",
        )
        exposure = _exposure_from_env(env)
        operator_id = env.get("DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID")
        return cls(
            bot_token=token,
            session_dir=session,
            inbound_media_dir=inbound_media_dir,
            outbound_media_dir=outbound_media_dir,
            api_base=env.get("DEEPAGENTS_TALON_TELEGRAM_API_BASE", DEFAULT_API_BASE),
            exposure=exposure,
            poll_timeout_seconds=_parse_float(
                env.get("DEEPAGENTS_TALON_TELEGRAM_POLL_TIMEOUT_SECONDS"),
                DEFAULT_POLL_TIMEOUT_SECONDS,
            ),
            poll_interval_seconds=_parse_float(
                env.get("DEEPAGENTS_TALON_TELEGRAM_POLL_INTERVAL_SECONDS"),
                DEFAULT_POLL_INTERVAL_SECONDS,
            ),
            request_timeout_seconds=_parse_float(
                env.get("DEEPAGENTS_TALON_TELEGRAM_REQUEST_TIMEOUT_SECONDS"),
                DEFAULT_REQUEST_TIMEOUT_SECONDS,
            ),
            operator_id=operator_id,
        )

    @property
    def offset_file(self) -> Path:
        """Path to the persisted getUpdates offset file."""
        return self.session_dir / _OFFSET_FILENAME


class TelegramTransport:
    """Small JSON HTTP client for the Telegram Bot API."""

    def __init__(self, *, api_base: str, token: str, timeout: float) -> None:
        """Initialize the transport.

        Args:
            api_base: Telegram Bot API base URL.
            token: Bot API authentication token.
            timeout: Request timeout in seconds.
        """
        self.api_base = api_base.rstrip("/")
        self.token = token
        self.timeout = timeout

    async def call(self, method: str, **params: object) -> object:
        """Call a Bot API method and return the decoded response.

        Args:
            method: Bot API method name (e.g. ``getUpdates``).
            **params: Request parameters passed as JSON body.

        Returns:
            JSON-decoded response body.

        Raises:
            TelegramError: If the request fails or the API returns an error.
        """
        return await asyncio.to_thread(self._request, method, params)

    def _request(self, method: str, params: dict[str, object]) -> object:
        url = f"{self.api_base}/bot{self.token}/{method}"
        body = json.dumps(params).encode()
        request = urllib.request.Request(  # noqa: S310  # URL is constructed from config.
            url,
            data=body,
            method="POST",
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(  # noqa: S310  # Bot API URL from config.
                request,
                timeout=self.timeout,
            ) as response:
                payload = json.loads(response.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            msg = f"Telegram Bot API request failed: {method}"
            raise TelegramError(msg) from error
        if isinstance(payload, dict) and not cast("Mapping[str, object]", payload).get("ok", True):
            description = cast("Mapping[str, object]", payload).get("description", "unknown error")
            msg = f"Telegram Bot API error in {method}: {description}"
            raise TelegramError(msg)
        return payload


class TelegramChannel:
    """Channel adapter for Telegram via the Bot API with long polling."""

    def __init__(
        self,
        config: TelegramChannelConfig,
        *,
        transport: TelegramTransport | None = None,
    ) -> None:
        """Initialize the Telegram channel without starting it.

        Args:
            config: Telegram channel configuration.
            transport: Optional test transport implementing the Bot API.
        """
        self.config = config
        self._transport = transport or TelegramTransport(
            api_base=config.api_base,
            token=config.bot_token,
            timeout=config.request_timeout_seconds,
        )
        self._handler: MessageHandler | None = None
        self._poll: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()
        self._status = ChannelStatus(provider="telegram", connected=False, detail="disconnected")
        self._bot_username: str | None = None
        self._operator_id = config.operator_id
        self._offset = 0

    def set_message_handler(self, handler: MessageHandler) -> None:
        """Register the host callback for inbound messages.

        Args:
            handler: Coroutine callback invoked for accepted inbound messages.
        """
        self._handler = handler

    async def start(self) -> None:
        """Load persisted offset, call getMe, and start the long-polling loop."""
        self.config.session_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.config.session_dir.chmod(0o700)
        if self.config.inbound_media_dir is not None:
            self.config.inbound_media_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.config.inbound_media_dir.chmod(0o700)
        self._stopped.clear()
        self._offset = _load_offset(self.config.offset_file)
        await self._identify_bot()
        self._poll = asyncio.create_task(self._poll_updates(), name="talon:telegram:poll")

    async def stop(self) -> None:
        """Stop the polling task and mark the channel as disconnected."""
        self._stopped.set()
        if self._poll is not None:
            self._poll.cancel()
            await asyncio.gather(self._poll, return_exceptions=True)
            self._poll = None
        self._status = ChannelStatus(provider="telegram", connected=False, detail="disconnected")

    async def send_message(self, conversation_id: str, text: str) -> None:
        """Send chunked text with MarkdownV2 formatting and plain-text fallback.

        Args:
            conversation_id: Telegram chat id.
            text: Message content to send.
        """
        for chunk in chunk_text(text, limit=MAX_TEXT_CHARS):
            escaped = _escape_markdown_v2(chunk)
            try:
                await self._transport.call(
                    "sendMessage",
                    chat_id=conversation_id,
                    text=escaped,
                    parse_mode="MarkdownV2",
                )
            except TelegramError:
                await self._transport.call(
                    "sendMessage",
                    chat_id=conversation_id,
                    text=chunk,
                )

    async def send_media(self, conversation_id: str, media: ChannelMedia) -> None:
        """Send validated image or document media to a Telegram chat.

        Args:
            conversation_id: Telegram chat id.
            media: Media payload to send.

        Raises:
            ChannelMediaError: If the media is too large or invalid.
        """
        checked = validate_media(media, root=self.config.outbound_media_dir)
        _check_telegram_size(checked)
        caption = _escape_markdown_v2(checked.caption) if checked.caption else None
        path = str(checked.path)
        if checked.media_type == "image":
            params: dict[str, object] = {"chat_id": conversation_id, "photo": path}
            if caption:
                params["caption"] = caption
                params["parse_mode"] = "MarkdownV2"
            await self._transport.call("sendPhoto", **params)
        else:
            params = {"chat_id": conversation_id, "document": path}
            if caption:
                params["caption"] = caption
                params["parse_mode"] = "MarkdownV2"
            await self._transport.call("sendDocument", **params)

    async def send_typing(self, conversation_id: str) -> None:
        """Send a Telegram typing indicator.

        Args:
            conversation_id: Telegram chat id.
        """
        try:
            await self._transport.call(
                "sendChatAction",
                chat_id=conversation_id,
                action="typing",
            )
        except TelegramError:
            logger.debug("Could not send Telegram typing indicator", exc_info=True)

    async def edit_message(self, conversation_id: str, message_id: str, text: str) -> None:
        """Edit a previously sent Telegram message.

        Args:
            conversation_id: Telegram chat id.
            message_id: Telegram message id.
            text: Replacement message content.
        """
        escaped = _escape_markdown_v2(text)
        try:
            await self._transport.call(
                "editMessageText",
                chat_id=conversation_id,
                message_id=int(message_id),
                text=escaped,
                parse_mode="MarkdownV2",
            )
        except TelegramError:
            await self._transport.call(
                "editMessageText",
                chat_id=conversation_id,
                message_id=int(message_id),
                text=text,
            )

    async def status(self) -> ChannelStatus:
        """Report the most recent Telegram Bot API connection status."""
        return self._status

    async def _identify_bot(self) -> None:
        try:
            payload = await self._transport.call("getMe")
        except TelegramError:
            logger.exception("Telegram getMe failed during startup")
            self._status = ChannelStatus(
                provider="telegram",
                connected=False,
                detail="getMe failed",
            )
            return
        result = _extract_result(payload)
        username = result.get("username") if isinstance(result, dict) else None
        if isinstance(username, str):
            self._bot_username = username
            logger.info("Telegram bot connected as @%s", username)
        self._status = ChannelStatus(
            provider="telegram",
            connected=True,
            detail=f"connected as @{username}" if isinstance(username, str) else "connected",
        )

    async def _poll_updates(self) -> None:
        while not self._stopped.is_set():
            try:
                payload = await self._transport.call(
                    "getUpdates",
                    offset=self._offset,
                    timeout=int(self.config.poll_timeout_seconds),
                    allowed_updates='["message"]',
                )
                updates = _extract_updates(payload)
                if updates:
                    self._offset = cast("int", updates[-1]["update_id"]) + 1
                    _save_offset(self.config.offset_file, self._offset)
                self._status = ChannelStatus(
                    provider="telegram",
                    connected=True,
                    detail="polling",
                )
                for update in updates:
                    message = _parse_update(update)
                    if message is None:
                        continue
                    if not self._is_operator(message):
                        self._operator_id = message.sender_id
                    if self.config.exposure.allows(message):
                        await self._dispatch(message)
                    else:
                        logger.debug(
                            "Dropping Telegram message %s from %s due to exposure policy",
                            message.message_id,
                            message.conversation_id,
                        )
            except (TelegramError, urllib.error.URLError, TimeoutError):
                logger.exception("Telegram long-polling error; retrying after interval")
                self._status = ChannelStatus(
                    provider="telegram",
                    connected=False,
                    detail="polling error",
                )
            except asyncio.CancelledError:
                raise
            await asyncio.sleep(self.config.poll_interval_seconds)

    def _is_operator(self, message: ChannelMessage) -> bool:
        """Auto-capture operator ID from first inbound message if not set."""
        if self._operator_id is not None:
            return message.sender_id == self._operator_id
        if message.sender_id is not None:
            self._operator_id = message.sender_id
            return True
        return False

    async def _dispatch(self, message: ChannelMessage) -> None:
        if self._handler is None:
            logger.warning("Dropping Telegram message because no handler is registered")
            return
        await self._handler(message)

    async def _download_inbound_media(self, file_id: str, destination: Path) -> None:
        """Download a file from the Telegram Bot API.

        Args:
            file_id: Telegram file identifier.
            destination: Local path to save the downloaded file.
        """
        payload = await self._transport.call("getFile", file_id=file_id)
        result = _extract_result(payload)
        file_path = result.get("file_path") if isinstance(result, dict) else None
        if not isinstance(file_path, str):
            msg = "Telegram getFile response missing file_path"
            raise TelegramError(msg)
        download_url = f"{self.config.api_base}/file/bot{self.config.bot_token}/{file_path}"
        await asyncio.to_thread(_download_file, download_url, destination, self.timeout)

    @property
    def timeout(self) -> float:
        """Request timeout for downloads."""
        return self.config.request_timeout_seconds


def _escape_markdown_v2(text: str) -> str:
    """Escape MarkdownV2 special characters.

    Args:
        text: Plain text to escape.

    Returns:
        Text with all MarkdownV2 special characters escaped.
    """
    return _MARKDOWNV2_SPECIAL.sub(r"\\\1", text)


def _check_telegram_size(media: ChannelMedia) -> None:
    """Validate media size against Telegram-specific limits.

    Args:
        media: Validated media payload.

    Raises:
        ChannelMediaError: If the file exceeds Telegram's size limit.
    """
    size = media.path.stat().st_size
    if media.media_type == "image" and size > MAX_PHOTO_BYTES:
        msg = f"photo media is too large for Telegram: {size} bytes exceeds {MAX_PHOTO_BYTES}"
        raise ChannelMediaError(msg)
    if media.media_type != "image" and size > MAX_DOCUMENT_BYTES:
        msg = f"document media is too large for Telegram: {size} bytes exceeds {MAX_DOCUMENT_BYTES}"
        raise ChannelMediaError(msg)


def _extract_result(payload: object) -> dict[str, object]:
    """Extract the ``result`` field from a Bot API response.

    Args:
        payload: Full Bot API response.

    Returns:
        The ``result`` object.

    Raises:
        TelegramError: If the response is malformed.
    """
    if not isinstance(payload, dict):
        msg = "Telegram Bot API response must be an object"
        raise TelegramError(msg)
    values = cast("Mapping[str, object]", payload)
    if not values.get("ok", True):
        description = values.get("description", "unknown error")
        msg = f"Telegram Bot API error: {description}"
        raise TelegramError(msg)
    result = values.get("result")
    if not isinstance(result, dict):
        msg = "Telegram Bot API response missing result"
        raise TelegramError(msg)
    return cast("dict[str, object]", result)


def _extract_updates(payload: object) -> list[dict[str, object]]:
    """Extract the list of updates from a getUpdates response.

    Args:
        payload: Full Bot API response.

    Returns:
        List of update objects.

    Raises:
        TelegramError: If the response is malformed.
    """
    if not isinstance(payload, dict):
        msg = "Telegram getUpdates response must be an object"
        raise TelegramError(msg)
    values = cast("Mapping[str, object]", payload)
    if not values.get("ok", True):
        description = values.get("description", "unknown error")
        msg = f"Telegram getUpdates error: {description}"
        raise TelegramError(msg)
    result = values.get("result")
    if not isinstance(result, list):
        msg = "Telegram getUpdates result must be a list"
        raise TelegramError(msg)
    return [cast("dict[str, object]", item) for item in result if isinstance(item, dict)]


def _parse_update(update: Mapping[str, object]) -> ChannelMessage | None:
    """Parse a single Telegram update into a ChannelMessage.

    Args:
        update: Raw update object from getUpdates.

    Returns:
        Parsed channel message, or ``None`` if the update should be skipped.
    """
    message = update.get("message")
    if not isinstance(message, dict):
        return None
    msg = cast("Mapping[str, object]", message)
    chat = msg.get("chat")
    if not isinstance(chat, dict):
        return None
    chat_values = cast("Mapping[str, object]", chat)
    chat_type = chat_values.get("type")
    if chat_type != "private":
        return None
    chat_id = chat_values.get("id")
    if not isinstance(chat_id, int):
        return None
    message_id = msg.get("message_id")
    if not isinstance(message_id, int):
        return None
    sender = msg.get("from")
    sender_id: str | None = None
    if isinstance(sender, dict):
        sender_id_raw = cast("Mapping[str, object]", sender).get("id")
        if isinstance(sender_id_raw, int):
            sender_id = str(sender_id_raw)

    text = msg.get("text")
    if not isinstance(text, str):
        text = ""

    metadata: dict[str, object] = {
        "provider": "telegram",
        "chat_type": "private",
        "from_self": sender_id is not None and sender_id == str(chat_id),
    }

    media_info = _extract_media_info(msg)
    if media_info is not None:
        metadata["media_type"] = media_info[0]
        metadata["file_id"] = media_info[1]

    return ChannelMessage(
        conversation_id=str(chat_id),
        text=text,
        sender_id=sender_id,
        message_id=str(message_id),
        metadata=metadata,
    )


def _extract_media_info(msg: Mapping[str, object]) -> tuple[str, str] | None:
    """Extract media type and file_id from a Telegram message.

    Args:
        msg: Telegram message object.

    Returns:
        Tuple of (media_type, file_id), or ``None`` if the message has no media.
    """
    photo = msg.get("photo")
    voice = msg.get("voice") or msg.get("audio")
    document = msg.get("document")
    if isinstance(photo, list) and photo:
        file_id = _largest_photo_file_id(photo)
        if file_id is not None:
            return "image", file_id
    if isinstance(voice, dict):
        file_id = cast("Mapping[str, object]", voice).get("file_id")
        if isinstance(file_id, str):
            return "voice", file_id
    if isinstance(document, dict):
        file_id = cast("Mapping[str, object]", document).get("file_id")
        if isinstance(file_id, str):
            return "document", file_id
    return None


def _largest_photo_file_id(photo_sizes: object) -> str | None:
    """Extract the file_id of the largest photo size from a photo array.

    Args:
        photo_sizes: List of photo size objects from a Telegram message.

    Returns:
        File id of the largest photo, or ``None``.
    """
    if not isinstance(photo_sizes, list):
        return None
    best: dict[str, object] | None = None
    for size in photo_sizes:
        if not isinstance(size, dict):
            continue
        current = best
        if current is None:
            best = cast("dict[str, object]", size)
        else:
            current_size = current.get("file_size", 0)
            new_size = cast("Mapping[str, object]", size).get("file_size", 0)
            if (
                isinstance(new_size, int)
                and isinstance(current_size, int)
                and new_size > current_size
            ):
                best = cast("dict[str, object]", size)
    if best is None:
        return None
    file_id = best.get("file_id")
    return file_id if isinstance(file_id, str) else None


def _download_file(url: str, destination: Path, timeout: float) -> None:
    """Download a file from a URL to a local path.

    Args:
        url: Source URL.
        destination: Destination file path.
        timeout: Download timeout in seconds.
    """
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    request = urllib.request.Request(url)  # noqa: S310  # URL constructed from Bot API config.
    with urllib.request.urlopen(  # noqa: S310  # download URL from Telegram API.
        request,
        timeout=timeout,
    ) as response:
        destination.write_bytes(response.read())
    destination.chmod(0o600)


def _exposure_from_env(env: Mapping[str, str]) -> ChannelExposure:
    """Build ChannelExposure from Telegram-specific env vars.

    Args:
        env: Environment variable mapping.

    Returns:
        Exposure policy for the Telegram channel.

    Raises:
        ValueError: If the exposure mode is invalid or open mode is not acknowledged.
    """
    mode = _exposure_mode(env.get("DEEPAGENTS_TALON_TELEGRAM_EXPOSURE", ExposureMode.SELF.value))
    if mode == ExposureMode.OPEN:
        _require_open_acknowledgement(env)
        logger.warning(
            "Telegram open exposure enabled; arbitrary senders can trigger the agent with "
            "operator credentials and local host access",
        )
    conversations = _split_csv(env.get("DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_CHATS", ""))
    mentions = tuple(_split_csv(env.get("DEEPAGENTS_TALON_TELEGRAM_MENTION_PATTERNS", "")))
    return ChannelExposure(
        mode=mode,
        operator_id=env.get("DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID"),
        conversations=frozenset(conversations),
        mention_patterns=mentions,
    )


def _exposure_mode(value: str) -> ExposureMode:
    try:
        return ExposureMode(value)
    except ValueError as error:
        modes = ", ".join(mode.value for mode in ExposureMode)
        msg = f"invalid Telegram exposure mode {value!r}; expected one of: {modes}"
        raise ValueError(msg) from error


def _require_open_acknowledgement(env: Mapping[str, str]) -> None:
    if env.get(OPEN_EXPOSURE_ACK_ENV) == OPEN_EXPOSURE_ACK_VALUE:
        return
    msg = (
        "Telegram exposure mode 'open' allows arbitrary senders to trigger the agent with "
        "operator credentials and local host access; set "
        f"{OPEN_EXPOSURE_ACK_ENV}={OPEN_EXPOSURE_ACK_VALUE} to acknowledge this risk"
    )
    raise ValueError(msg)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as error:
        msg = f"expected float value, got {value!r}"
        raise ValueError(msg) from error


# --- Offset persistence (ticket 23) ---


def _load_offset(offset_file: Path) -> int:
    """Load the persisted getUpdates offset from disk.

    Args:
        offset_file: Path to the offset state file.

    Returns:
        Persisted offset value, or ``0`` if the file is missing or corrupt.
    """
    if not offset_file.is_file():
        return 0
    try:
        data = json.loads(offset_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Telegram offset file is corrupt or unreadable; starting with offset=0")
        return 0
    offset = data.get("offset") if isinstance(data, dict) else None
    if not isinstance(offset, int) or offset < 0:
        logger.warning("Telegram offset file contains invalid offset; starting with offset=0")
        return 0
    return offset


def _save_offset(offset_file: Path, offset: int) -> None:
    """Atomically persist the getUpdates offset to disk.

    Args:
        offset_file: Path to the offset state file.
        offset: Current offset value to persist.
    """
    offset_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    content = json.dumps({"offset": offset})
    tmp = offset_file.with_suffix(".json.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(offset_file)
