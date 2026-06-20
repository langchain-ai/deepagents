"""Telegram channel adapter backed by the Bot API over urllib.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
import secrets
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from deepagents_talon.channels.base import (
    DEFAULT_MAX_MEDIA_BYTES,
    MAX_TEXT_CHARS,
    ChannelExposure,
    ChannelExposureEnv,
    ChannelMediaError,
    ExposureMode,
    channel_exposure_from_env,
    chunk_text,
    max_media_bytes_from_env,
    message_with_media_paths,
    parse_float,
    replace_message_metadata,
    split_csv,
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
MAX_CAPTION_CHARS = 1024
OPEN_EXPOSURE_ACK_ENV = "DEEPAGENTS_TALON_TELEGRAM_OPEN_ACK"
_OFFSET_FILENAME = "telegram_offset.json"
_ALLOWED_UPDATES = ["message", "channel_post"]


class TelegramError(RuntimeError):
    """Raised when the Telegram Bot API reports or causes a transport error."""


@dataclass(frozen=True, slots=True)
class _TelegramMediaInfo:
    """Media metadata extracted from an inbound Telegram message."""

    media_type: str
    file_id: str
    file_name: str | None = None
    mime_type: str | None = None


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
        max_media_bytes: Maximum media bytes allowed for inbound downloads and
            outbound local files before provider-specific limits are applied.
        operator_id: First Telegram user ID for the operator, preserved for
            compatibility with single-operator callers in self exposure mode.
        allowed_user_ids: Telegram user IDs allowed to trigger private chats in
            allowlist exposure mode.
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
    max_media_bytes: int = DEFAULT_MAX_MEDIA_BYTES
    operator_id: str | None = None
    allowed_user_ids: frozenset[str] = field(default_factory=frozenset)

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
        return cls(
            bot_token=token,
            session_dir=session,
            inbound_media_dir=inbound_media_dir,
            outbound_media_dir=outbound_media_dir,
            api_base=env.get("DEEPAGENTS_TALON_TELEGRAM_API_BASE", DEFAULT_API_BASE),
            exposure=exposure,
            poll_timeout_seconds=parse_float(
                env.get("DEEPAGENTS_TALON_TELEGRAM_POLL_TIMEOUT_SECONDS"),
                DEFAULT_POLL_TIMEOUT_SECONDS,
            ),
            poll_interval_seconds=parse_float(
                env.get("DEEPAGENTS_TALON_TELEGRAM_POLL_INTERVAL_SECONDS"),
                DEFAULT_POLL_INTERVAL_SECONDS,
            ),
            request_timeout_seconds=parse_float(
                env.get("DEEPAGENTS_TALON_TELEGRAM_REQUEST_TIMEOUT_SECONDS"),
                DEFAULT_REQUEST_TIMEOUT_SECONDS,
            ),
            max_media_bytes=max_media_bytes_from_env(env),
            operator_id=exposure.operator_id,
            allowed_user_ids=frozenset(
                split_csv(env.get("DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_USERS", "")),
            ),
        )

    @property
    def offset_file(self) -> Path:
        """Path to the persisted getUpdates offset file."""
        return self.session_dir / _OFFSET_FILENAME


class TelegramTransport:
    """Small HTTP client for the Telegram Bot API."""

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
            method: Bot API method name (e.g. `getUpdates`).
            **params: Request parameters passed as JSON body.

        Returns:
            JSON-decoded response body.

        Raises:
            TelegramError: If the request fails or the API returns an error.
        """
        return await asyncio.to_thread(self._request, method, params)

    async def upload(
        self,
        method: str,
        *,
        file_field: str,
        file_path: Path,
        **params: object,
    ) -> object:
        """Call a Bot API method with one local file as multipart form data.

        Args:
            method: Bot API method name (e.g. `sendPhoto`).
            file_field: Multipart field name for the file parameter.
            file_path: Local file path to upload.
            **params: Additional form fields.

        Returns:
            JSON-decoded response body.

        Raises:
            TelegramError: If the request fails or the API returns an error.
        """
        return await asyncio.to_thread(self._upload, method, file_field, file_path, params)

    def _request(self, method: str, params: dict[str, object]) -> object:
        url = f"{self.api_base}/bot{self.token}/{method}"
        body = json.dumps(params).encode()
        request = urllib.request.Request(  # noqa: S310  # URL is constructed from config.
            url,
            data=body,
            method="POST",
            headers={"content-type": "application/json"},
        )
        return self._send_request(method, request)

    def _upload(
        self,
        method: str,
        file_field: str,
        file_path: Path,
        params: dict[str, object],
    ) -> object:
        url = f"{self.api_base}/bot{self.token}/{method}"
        boundary = f"deepagents-talon-{secrets.token_hex(16)}"
        body = _encode_multipart_form(
            params,
            file_field=file_field,
            file_path=file_path,
            boundary=boundary,
        )
        request = urllib.request.Request(  # noqa: S310  # URL is constructed from config.
            url,
            data=body,
            method="POST",
            headers={
                "content-type": f"multipart/form-data; boundary={boundary}",
                "content-length": str(len(body)),
            },
        )
        return self._send_request(method, request)

    def _send_request(self, method: str, request: urllib.request.Request) -> object:
        try:
            with urllib.request.urlopen(  # noqa: S310  # Bot API URL from config.
                request,
                timeout=self.timeout,
            ) as response:
                payload = json.loads(response.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            msg = f"Telegram Bot API request failed: {method}"
            raise TelegramError(msg) from error
        return _raise_for_api_error(method, payload)


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
        self._exposure = _effective_exposure(config.exposure, config.operator_id)
        self._bot_id: str | None = None
        self._bot_username: str | None = None
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
        """Send chunked plain text.

        Args:
            conversation_id: Telegram chat id.
            text: Message content to send.
        """
        for chunk in chunk_text(text, limit=MAX_TEXT_CHARS):
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
        checked = validate_media(
            media,
            root=self.config.outbound_media_dir,
            max_bytes=self.config.max_media_bytes,
        )
        _check_telegram_size(checked)
        caption = await self._media_caption(conversation_id, checked.caption)
        if checked.media_type == "image":
            params: dict[str, object] = {"chat_id": conversation_id}
            if caption:
                params["caption"] = caption
            await self._transport.upload(
                "sendPhoto",
                file_field="photo",
                file_path=checked.path,
                **params,
            )
        else:
            params = {"chat_id": conversation_id}
            if caption:
                params["caption"] = caption
            await self._transport.upload(
                "sendDocument",
                file_field="document",
                file_path=checked.path,
                **params,
            )

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
        bot_id = result.get("id") if isinstance(result, dict) else None
        if isinstance(bot_id, int):
            self._bot_id = str(bot_id)
        username = result.get("username") if isinstance(result, dict) else None
        if isinstance(username, str):
            self._bot_username = username
            logger.info("Telegram bot connected as @%s", username)
        self._status = ChannelStatus(
            provider="telegram",
            connected=True,
            detail=f"connected as @{username}" if isinstance(username, str) else "connected",
        )

    async def _media_caption(self, conversation_id: str, caption: str | None) -> str | None:
        if not caption:
            return None
        if len(caption) <= MAX_CAPTION_CHARS:
            return caption
        await self.send_message(conversation_id, caption)
        return None

    async def _poll_updates(self) -> None:
        while not self._stopped.is_set():
            try:
                payload = await self._transport.call(
                    "getUpdates",
                    offset=self._offset,
                    timeout=int(self.config.poll_timeout_seconds),
                    allowed_updates=_ALLOWED_UPDATES,
                )
                updates = _extract_updates(payload)
                self._status = ChannelStatus(
                    provider="telegram",
                    connected=True,
                    detail="polling",
                )
                for update in updates:
                    message = _parse_update(update)
                    if message is None:
                        self._commit_update_offset(update)
                        continue
                    message = _with_from_self(message, self._bot_id)
                    if not _allows_telegram_message(
                        self._exposure,
                        self.config.allowed_user_ids,
                        message,
                    ):
                        logger.debug(
                            "Dropping Telegram message %s from %s due to exposure policy",
                            message.message_id,
                            message.conversation_id,
                        )
                        self._commit_update_offset(update)
                        continue
                    message = await self._prepare_inbound_media(message)
                    await self._dispatch(message)
                    self._commit_update_offset(update)
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

    def _commit_update_offset(self, update: Mapping[str, object]) -> None:
        update_id = update.get("update_id")
        if not isinstance(update_id, int):
            return
        next_offset = update_id + 1
        if next_offset <= self._offset:
            return
        self._offset = next_offset
        _save_offset(self.config.offset_file, self._offset)

    async def _dispatch(self, message: ChannelMessage) -> None:
        if self._handler is None:
            logger.warning("Dropping Telegram message because no handler is registered")
            return
        await self._handler(message)

    async def _prepare_inbound_media(self, message: ChannelMessage) -> ChannelMessage:
        media_type = message.metadata.get("media_type")
        file_id = message.metadata.get("file_id")
        if not isinstance(media_type, str) or not isinstance(file_id, str):
            return message
        if self.config.inbound_media_dir is None:
            return message

        try:
            destination = await self._download_inbound_media(
                file_id,
                media_type=media_type,
                message_id=message.message_id,
            )
        except ChannelMediaError as error:
            logger.warning(
                "Skipping Telegram inbound media for message %s: %s",
                message.message_id,
                error,
            )
            metadata = dict(message.metadata)
            metadata["has_media"] = False
            metadata["media_error"] = str(error)
            return replace_message_metadata(message, metadata)
        metadata = dict(message.metadata)
        path = str(destination)
        mime_type = _downloaded_mime_type(destination, metadata)
        message = replace_message_metadata(message, metadata)
        return message_with_media_paths(
            message,
            media_paths=[path],
            mime_types=[mime_type] if mime_type is not None else [],
        )

    async def _download_inbound_media(
        self,
        file_id: str,
        *,
        media_type: str,
        message_id: str | None,
    ) -> Path:
        """Download a file from the Telegram Bot API.

        Args:
            file_id: Telegram file identifier.
            media_type: Normalized media category.
            message_id: Telegram message identifier used to name the file.

        Returns:
            Local path to the downloaded file.
        """
        payload = await self._transport.call("getFile", file_id=file_id)
        result = _extract_result(payload)
        file_path = result.get("file_path") if isinstance(result, dict) else None
        if not isinstance(file_path, str):
            msg = "Telegram getFile response missing file_path"
            raise TelegramError(msg)
        file_size = result.get("file_size") if isinstance(result, dict) else None
        if isinstance(file_size, int) and file_size > self.config.max_media_bytes:
            msg = (
                "Telegram media is too large: "
                f"{file_size} bytes exceeds {self.config.max_media_bytes}"
            )
            raise ChannelMediaError(msg)
        if self.config.inbound_media_dir is None:
            msg = "Telegram inbound media directory is not configured"
            raise TelegramError(msg)
        suffix = _safe_suffix(file_path, media_type)
        destination = self.config.inbound_media_dir / _inbound_media_filename(
            message_id=message_id,
            file_id=file_id,
            suffix=suffix,
        )
        download_url = f"{self.config.api_base}/file/bot{self.config.bot_token}/{file_path}"
        await asyncio.to_thread(
            _download_file,
            download_url,
            destination,
            self.timeout,
            self.config.max_media_bytes,
        )
        return destination

    @property
    def timeout(self) -> float:
        """Request timeout for downloads."""
        return self.config.request_timeout_seconds


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


def _raise_for_api_error(method: str, payload: object) -> object:
    if isinstance(payload, dict) and not cast("Mapping[str, object]", payload).get("ok", True):
        description = cast("Mapping[str, object]", payload).get("description", "unknown error")
        msg = f"Telegram Bot API error in {method}: {description}"
        raise TelegramError(msg)
    return payload


def _encode_multipart_form(
    params: Mapping[str, object],
    *,
    file_field: str,
    file_path: Path,
    boundary: str,
) -> bytes:
    """Encode request parameters and one file as multipart form data.

    Args:
        params: Form fields to include before the file field.
        file_field: Multipart field name for the file parameter.
        file_path: Local file path to upload.
        boundary: Multipart boundary string.

    Returns:
        Multipart request body.
    """
    chunks: list[bytes] = []
    for key, value in params.items():
        if value is None:
            continue
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{_form_header_value(key)}"\r\n\r\n'
                ).encode(),
                _form_field_value(value).encode(),
                b"\r\n",
            ],
        )

    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                "Content-Disposition: form-data; "
                f'name="{_form_header_value(file_field)}"; '
                f'filename="{_form_header_value(file_path.name)}"\r\n'
            ).encode(),
            f"Content-Type: {mime_type}\r\n\r\n".encode(),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ],
    )
    return b"".join(chunks)


def _form_header_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "")


def _form_field_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _effective_exposure(
    exposure: ChannelExposure,
    operator_id: str | None,
) -> ChannelExposure:
    if (
        exposure.mode != ExposureMode.SELF
        or exposure.operator_id is not None
        or exposure.operator_ids
        or operator_id is None
    ):
        return exposure
    return replace(exposure, operator_id=operator_id)


def _with_from_self(message: ChannelMessage, bot_id: str | None) -> ChannelMessage:
    if bot_id is None or message.sender_id != bot_id:
        return message
    metadata = dict(message.metadata)
    metadata["from_self"] = True
    return replace_message_metadata(message, metadata)


def _allows_telegram_message(
    exposure: ChannelExposure,
    allowed_user_ids: frozenset[str],
    message: ChannelMessage,
) -> bool:
    if (
        exposure.mode == ExposureMode.ALLOWLIST
        and message.metadata.get("chat_type") == "private"
        and message.sender_id in allowed_user_ids
    ):
        return True
    return exposure.allows(message)


def _downloaded_mime_type(path: Path, metadata: dict[str, object]) -> str | None:
    raw = metadata.get("mime_type")
    if isinstance(raw, str) and "/" in raw:
        return raw
    raw_many = metadata.get("media_mime_types")
    if isinstance(raw_many, list):
        for item in raw_many:
            if isinstance(item, str) and "/" in item:
                return item
    guessed, _ = mimetypes.guess_type(path)
    return guessed


def _safe_suffix(file_path: str, media_type: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if re.fullmatch(r"\.[a-z0-9]{1,16}", suffix):
        return suffix
    return _default_suffix(media_type)


def _default_suffix(media_type: str) -> str:
    if media_type == "image":
        return ".jpg"
    if media_type == "voice":
        return ".ogg"
    return ".bin"


def _inbound_media_filename(
    *,
    message_id: str | None,
    file_id: str,
    suffix: str,
) -> str:
    message = _safe_filename_part(message_id or "message")
    token = _safe_filename_part(file_id)[-24:] or "file"
    return f"{message}_{token}{suffix}"


def _safe_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "file"


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
    values = _message_values(update)
    if values is None:
        return None
    msg, chat_id, message_id, chat_type = values
    return ChannelMessage(
        conversation_id=str(chat_id),
        text=_message_text(msg),
        sender_id=_sender_id(msg),
        message_id=str(message_id),
        metadata=_message_metadata(msg, chat_type=chat_type),
    )


def _message_values(
    update: Mapping[str, object],
) -> tuple[Mapping[str, object], int, int, str] | None:
    message = update.get("message")
    expected_chat_type = "private"
    if not isinstance(message, dict):
        message = update.get("channel_post")
        expected_chat_type = "channel"
    if not isinstance(message, dict):
        return None
    msg = cast("Mapping[str, object]", message)
    chat = msg.get("chat")
    if not isinstance(chat, dict):
        return None
    chat_values = cast("Mapping[str, object]", chat)
    chat_type = chat_values.get("type")
    if chat_type != expected_chat_type:
        return None
    chat_id = chat_values.get("id")
    if not isinstance(chat_id, int):
        return None
    message_id = msg.get("message_id")
    if not isinstance(message_id, int):
        return None
    return msg, chat_id, message_id, expected_chat_type


def _sender_id(msg: Mapping[str, object]) -> str | None:
    sender = msg.get("from")
    if isinstance(sender, dict):
        sender_id_raw = cast("Mapping[str, object]", sender).get("id")
        if isinstance(sender_id_raw, int):
            return str(sender_id_raw)
    return None


def _message_text(msg: Mapping[str, object]) -> str:
    text = msg.get("text")
    if not isinstance(text, str):
        text = msg.get("caption")
    if not isinstance(text, str):
        return ""
    return text


def _message_metadata(msg: Mapping[str, object], *, chat_type: str) -> dict[str, object]:
    metadata: dict[str, object] = {
        "provider": "telegram",
        "chat_type": chat_type,
        "from_self": False,
    }

    media_info = _extract_media_info(msg)
    if media_info is not None:
        metadata["media_type"] = media_info.media_type
        metadata["file_id"] = media_info.file_id
        if media_info.file_name is not None:
            metadata["file_name"] = media_info.file_name
        if media_info.mime_type is not None:
            metadata["mime_type"] = media_info.mime_type
            metadata["media_mime_types"] = [media_info.mime_type]

    return metadata


def _extract_media_info(msg: Mapping[str, object]) -> _TelegramMediaInfo | None:
    """Extract media type and file_id from a Telegram message.

    Args:
        msg: Telegram message object.

    Returns:
        Media info, or `None` if the message has no media.
    """
    photo = msg.get("photo")
    voice = msg.get("voice") or msg.get("audio")
    document = msg.get("document")
    if isinstance(photo, list) and photo:
        file_id = _largest_photo_file_id(photo)
        if file_id is not None:
            return _TelegramMediaInfo(media_type="image", file_id=file_id)
    if isinstance(voice, dict):
        values = cast("Mapping[str, object]", voice)
        file_id = values.get("file_id")
        if isinstance(file_id, str):
            return _TelegramMediaInfo(
                media_type="voice",
                file_id=file_id,
                mime_type=_optional_str(values.get("mime_type")),
            )
    if isinstance(document, dict):
        values = cast("Mapping[str, object]", document)
        file_id = values.get("file_id")
        if isinstance(file_id, str):
            return _TelegramMediaInfo(
                media_type="document",
                file_id=file_id,
                file_name=_optional_str(values.get("file_name")),
                mime_type=_optional_str(values.get("mime_type")),
            )
    return None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


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


def _download_file(url: str, destination: Path, timeout: float, max_bytes: int) -> None:
    """Download a file from a URL to a local path.

    Args:
        url: Source URL.
        destination: Destination file path.
        timeout: Download timeout in seconds.
        max_bytes: Maximum bytes to write before aborting.

    Raises:
        ChannelMediaError: If the remote file exceeds `max_bytes`.
    """
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    request = urllib.request.Request(url)  # noqa: S310  # URL constructed from Bot API config.
    with urllib.request.urlopen(  # noqa: S310  # download URL from Telegram API.
        request,
        timeout=timeout,
    ) as response:
        length = response.headers.get("content-length")
        if length is not None:
            try:
                expected = int(length)
            except ValueError:
                expected = None
            if expected is not None and expected > max_bytes:
                msg = f"media file is too large: {expected} bytes exceeds {max_bytes}"
                raise ChannelMediaError(msg)

        total = 0
        with destination.open("wb") as file:
            while chunk := response.read(64 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    file.close()
                    destination.unlink(missing_ok=True)
                    msg = f"media file is too large: {total} bytes exceeds {max_bytes}"
                    raise ChannelMediaError(msg)
                file.write(chunk)
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
    exposure = channel_exposure_from_env(
        env,
        ChannelExposureEnv(
            provider="Telegram",
            exposure="DEEPAGENTS_TALON_TELEGRAM_EXPOSURE",
            allowlist_chats="DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_CHATS",
            mention_patterns="DEEPAGENTS_TALON_TELEGRAM_MENTION_PATTERNS",
            operator_id="DEEPAGENTS_TALON_TELEGRAM_OPERATOR_ID",
            open_ack=OPEN_EXPOSURE_ACK_ENV,
            require_self_operator=True,
        ),
    )
    if exposure.mode == ExposureMode.OPEN:
        logger.warning(
            "Telegram open exposure enabled; arbitrary senders can trigger the agent with "
            "operator credentials and local host access",
        )
    return exposure


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
