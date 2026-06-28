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
from typing import TYPE_CHECKING, NamedTuple, cast

from deepagents_talon.channels.base import (
    DEFAULT_MAX_MEDIA_BYTES,
    MAX_TEXT_CHARS,
    ChannelExposure,
    ChannelExposureEnv,
    ChannelMediaError,
    ExposureMode,
    channel_exposure_from_env,
    chunk_text,
    dispatch_message,
    max_media_bytes_from_env,
    message_with_media_paths,
    optional_str,
    outbound_media_root_from_env,
    parse_float,
    split_csv,
    validate_media,
)
from deepagents_talon.interfaces import (
    ChannelMedia,
    ChannelMessage,
    ChannelReaction,
    ChannelStatus,
    MessageHandler,
    ReactionHandler,
    SendResult,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from deepagents_talon.config import TalonConfig

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.telegram.org"
DEFAULT_POLL_TIMEOUT_SECONDS = 30.0
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 35.0
MAX_CAPTION_CHARS = 1024
OPEN_EXPOSURE_ACK_ENV = "DEEPAGENTS_TALON_TELEGRAM_OPEN_ACK"
_OFFSET_FILENAME = "telegram_offset.json"
_ALLOWED_UPDATES = ["message", "channel_post", "message_reaction"]


class _TelegramError(RuntimeError):
    """Raised when the Telegram Bot API reports or causes a transport error.

    Args:
        message: Error description.
        retry_after: Seconds to wait before retrying, as reported by a 429 response.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        """Initialize the Telegram error."""
        super().__init__(message)
        self.retry_after = retry_after


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
        outbound_media_dir = outbound_media_root_from_env(env)
        exposure = channel_exposure_from_env(
            env,
            ChannelExposureEnv(
                provider="Telegram",
                env_prefix="DEEPAGENTS_TALON_TELEGRAM",
                open_ack=OPEN_EXPOSURE_ACK_ENV,
                require_self_operator=True,
            ),
        )
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
            allowed_user_ids=frozenset(
                split_csv(env.get("DEEPAGENTS_TALON_TELEGRAM_ALLOWLIST_USERS", "")),
            ),
        )

    @property
    def offset_file(self) -> Path:
        """Path to the persisted getUpdates offset file."""
        return self.session_dir / _OFFSET_FILENAME


class _TelegramTransport:
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
            _TelegramError: If the request fails or the API returns an error.
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
            _TelegramError: If the request fails or the API returns an error.
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
            raise _TelegramError(msg) from error
        return _validate_response(payload)


class TelegramChannel:
    """Channel adapter for Telegram via the Bot API with long polling.

    Only private chats and channel posts are processed. Messages from group
    and supergroup chats are silently dropped during parsing.
    """

    def __init__(
        self,
        config: TelegramChannelConfig,
        *,
        transport: _TelegramTransport | None = None,
    ) -> None:
        """Initialize the Telegram channel without starting it.

        Args:
            config: Telegram channel configuration.
            transport: Optional test transport implementing the Bot API.
        """
        self.config = config
        self._transport = transport or _TelegramTransport(
            api_base=config.api_base,
            token=config.bot_token,
            timeout=config.request_timeout_seconds,
        )
        self._handler: MessageHandler | None = None
        self._reaction_handler: ReactionHandler | None = None
        self._poll: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()
        self._status = ChannelStatus(provider="telegram", connected=False, detail="disconnected")
        self._exposure = config.exposure
        self._bot_id: str | None = None
        self._bot_username: str | None = None
        self._offset = 0

    def set_message_handler(self, handler: MessageHandler) -> None:
        """Register the host callback for inbound messages.

        Args:
            handler: Coroutine callback invoked for accepted inbound messages.
        """
        self._handler = handler

    def set_reaction_handler(self, handler: ReactionHandler) -> None:
        """Register the host callback for inbound reactions.

        Args:
            handler: Coroutine callback invoked for accepted inbound reactions.
        """
        self._reaction_handler = handler

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

    async def send_message(self, conversation_id: str, text: str) -> SendResult:
        """Send chunked plain text.

        Args:
            conversation_id: Telegram chat id.
            text: Message content to send.

        Returns:
            Result indicating whether the send succeeded.
        """
        for chunk in chunk_text(text, limit=MAX_TEXT_CHARS):
            payload = await self._transport.call(
                "sendMessage",
                chat_id=conversation_id,
                text=chunk,
            )
        return SendResult(success=True, message_id=_extract_telegram_message_id(payload))

    async def send_media(self, conversation_id: str, media: ChannelMedia) -> SendResult:
        """Send validated image, video, or document media to a Telegram chat.

        Args:
            conversation_id: Telegram chat id.
            media: Media payload to send.

        Returns:
            Result indicating whether the send succeeded.

        Raises:
            ChannelMediaError: If the media is too large or invalid.
        """
        checked = validate_media(
            media,
            root=self.config.outbound_media_dir,
            max_bytes=self.config.max_media_bytes,
        )
        caption = await self._media_caption(conversation_id, checked.caption)
        method, file_field = _telegram_send_method(checked.media_type)
        params: dict[str, object] = {"chat_id": conversation_id}
        if caption:
            params["caption"] = caption
        payload = await self._transport.upload(
            method,
            file_field=file_field,
            file_path=checked.path,
            **params,
        )
        return SendResult(success=True, message_id=_extract_telegram_message_id(payload))

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
        except _TelegramError:
            logger.debug("Could not send Telegram typing indicator", exc_info=True)

    async def edit_message(self, conversation_id: str, message_id: str, text: str) -> SendResult:
        """Edit a previously sent Telegram message.

        Args:
            conversation_id: Telegram chat id.
            message_id: Telegram message id.
            text: Replacement message content.

        Returns:
            Result indicating whether the edit succeeded.
        """
        await self._transport.call(
            "editMessageText",
            chat_id=conversation_id,
            message_id=int(message_id),
            text=text,
        )
        return SendResult(success=True, message_id=message_id)

    async def status(self) -> ChannelStatus:
        """Report the most recent Telegram Bot API connection status."""
        return self._status

    async def _identify_bot(self) -> None:
        try:
            payload = await self._transport.call("getMe")
        except _TelegramError:
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
                    await self._process_update(update)
                if updates:
                    self._advance_offset(updates)
            except _TelegramError as error:
                delay = (
                    error.retry_after
                    if error.retry_after is not None
                    else self.config.poll_interval_seconds
                )
                logger.warning(
                    "Telegram long-polling error; retrying after %.1fs: %s",
                    delay,
                    error,
                )
                self._status = ChannelStatus(
                    provider="telegram",
                    connected=False,
                    detail="polling error",
                )
                await asyncio.sleep(delay)
                continue
            except (urllib.error.URLError, TimeoutError):
                logger.exception("Telegram long-polling error; retrying after interval")
                self._status = ChannelStatus(
                    provider="telegram",
                    connected=False,
                    detail="polling error",
                )
            except asyncio.CancelledError:
                raise
            await asyncio.sleep(self.config.poll_interval_seconds)

    def _advance_offset(self, updates: list[dict[str, object]]) -> None:
        """Advance the persisted offset past the last update in the batch.

        Args:
            updates: Updates from a single getUpdates response.
        """
        max_id = -1
        for update in updates:
            update_id = update.get("update_id")
            if isinstance(update_id, int) and update_id >= max_id:
                max_id = update_id
        if max_id < 0:
            return
        next_offset = max_id + 1
        if next_offset <= self._offset:
            return
        self._offset = next_offset
        _save_offset(self.config.offset_file, self._offset)

    async def _process_update(self, update: dict[str, object]) -> None:
        reaction = _parse_reaction_update(update)
        if reaction is not None:
            await self._process_reaction(reaction)
            return

        message = _parse_update(update)
        if message is None:
            return
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
            return
        message = await self._prepare_inbound_media(message)
        await dispatch_message(self._handler, message, provider="Telegram")

    async def _process_reaction(self, reaction: ChannelReaction) -> None:
        if not _allows_telegram_reaction(
            self._exposure,
            self.config.allowed_user_ids,
            reaction,
        ):
            logger.debug(
                "Dropping Telegram reaction %s on %s due to exposure policy",
                reaction.message_id,
                reaction.conversation_id,
            )
            return
        await self._dispatch_reaction(reaction)

    async def _dispatch_reaction(self, reaction: ChannelReaction) -> None:
        if self._reaction_handler is None:
            logger.warning("Dropping Telegram reaction because no handler is registered")
            return
        await self._reaction_handler(reaction)

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
        except (ChannelMediaError, _TelegramError, urllib.error.URLError, TimeoutError) as error:
            logger.warning(
                "Skipping Telegram inbound media for message %s: %s",
                message.message_id,
                error,
            )
            metadata = dict(message.metadata)
            metadata["has_media"] = False
            metadata["media_error"] = str(error)
            return replace(message, metadata=metadata)
        path = str(destination)
        mime_type = _downloaded_mime_type(destination, dict(message.metadata))
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
            raise _TelegramError(msg)
        file_size = result.get("file_size") if isinstance(result, dict) else None
        if isinstance(file_size, int) and file_size > self.config.max_media_bytes:
            msg = (
                "Telegram media is too large: "
                f"{file_size} bytes exceeds {self.config.max_media_bytes}"
            )
            raise ChannelMediaError(msg)
        if self.config.inbound_media_dir is None:
            msg = "Telegram inbound media directory is not configured"
            raise _TelegramError(msg)
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
            self.config.request_timeout_seconds,
            self.config.max_media_bytes,
        )
        return destination


def _telegram_send_method(media_type: str) -> tuple[str, str]:
    """Return the Bot API method and file field for a media type."""
    if media_type == "image":
        return ("sendPhoto", "photo")
    if media_type == "video":
        return ("sendVideo", "video")
    return ("sendDocument", "document")


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


_C0_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _form_header_value(value: str) -> str:
    """Escape a string for safe inclusion in a multipart form-data header.

    Strips all C0 control characters and DEL, then escapes backslashes and
    double quotes to prevent header injection.
    """
    return _C0_CONTROL_RE.sub("", value).replace("\\", "\\\\").replace('"', '\\"')


def _form_field_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _with_from_self(message: ChannelMessage, bot_id: str | None) -> ChannelMessage:
    if bot_id is None or message.sender_id != bot_id:
        return message
    metadata = dict(message.metadata)
    metadata["from_self"] = True
    return replace(message, metadata=metadata)


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


def _allows_telegram_reaction(
    exposure: ChannelExposure,
    allowed_user_ids: frozenset[str],
    reaction: ChannelReaction,
) -> bool:
    if reaction.sender_id is None:
        return False
    return reaction.sender_id in exposure.operator_ids or reaction.sender_id in allowed_user_ids


_MIME_RE = re.compile(r"^[a-z]+/[a-z0-9.+\-]+$")


def _downloaded_mime_type(path: Path, metadata: dict[str, object]) -> str | None:
    """Determine MIME type for a downloaded media file.

    Checks metadata-provided MIME types first, falling back to file extension
    guessing. Metadata values are validated against the standard MIME format
    (``type/subtype``) rather than naively checking for a slash.
    """
    raw = metadata.get("mime_type")
    if isinstance(raw, str) and _MIME_RE.match(raw):
        return raw
    raw_many = metadata.get("media_mime_types")
    if isinstance(raw_many, list):
        for item in raw_many:
            if isinstance(item, str) and _MIME_RE.match(item):
                return item
    guessed, _ = mimetypes.guess_type(path)
    return guessed


def _safe_suffix(file_path: str, media_type: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if media_type == "voice" and suffix in {".oga", ".opus"}:
        return ".ogg"
    if re.fullmatch(r"\.[a-z0-9]{1,16}", suffix):
        return suffix
    if media_type == "image":
        return ".jpg"
    if media_type == "voice":
        return ".ogg"
    if media_type == "video":
        return ".mp4"
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


def _validate_response(payload: object) -> object:
    """Validate a Bot API response envelope and return the raw ``result`` value.

    Args:
        payload: Full Bot API response.

    Returns:
        The ``result`` field from the response.

    Raises:
        _TelegramError: If the response is not an object or reports an error.
    """
    if not isinstance(payload, dict):
        msg = "Telegram Bot API response must be an object"
        raise _TelegramError(msg)
    values = cast("Mapping[str, object]", payload)
    if not values.get("ok", True):
        description = values.get("description", "unknown error")
        msg = f"Telegram Bot API error: {description}"
        retry_after_raw = values.get("retry_after")
        retry_after = float(retry_after_raw) if isinstance(retry_after_raw, (int, float)) else None
        raise _TelegramError(msg, retry_after=retry_after)
    return values.get("result")


def _extract_result(result: object) -> dict[str, object]:
    """Extract a dict result from a validated Bot API response.

    Args:
        result: Validated ``result`` field from a Bot API response.

    Returns:
        The result object as a dict.

    Raises:
        _TelegramError: If the result is not a dict.
    """
    if not isinstance(result, dict):
        msg = "Telegram Bot API response missing result"
        raise _TelegramError(msg)
    return cast("dict[str, object]", result)


def _extract_telegram_message_id(payload: object) -> str | None:
    """Extract a message id from a Bot API send response payload."""
    result = _extract_result(payload)
    message_id = result.get("message_id")
    if isinstance(message_id, int):
        return str(message_id)
    return None


def _extract_updates(result: object) -> list[dict[str, object]]:
    """Extract the list of updates from a getUpdates response result.

    Args:
        result: Validated ``result`` field from a getUpdates response.

    Returns:
        List of update objects.

    Raises:
        _TelegramError: If the result is not a list.
    """
    if not isinstance(result, list):
        msg = "Telegram getUpdates result must be a list"
        raise _TelegramError(msg)
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
    return ChannelMessage(
        conversation_id=str(values.chat_id),
        text=_message_text(values.msg),
        sender_id=_sender_id(values.msg),
        message_id=str(values.message_id),
        metadata=_message_metadata(values.msg, chat_type=values.chat_type),
    )


def _parse_reaction_update(update: Mapping[str, object]) -> ChannelReaction | None:
    """Parse a Telegram message_reaction update into a ChannelReaction.

    Args:
        update: Raw update object from getUpdates.

    Returns:
        Parsed channel reaction, or ``None`` if required fields are missing.
    """
    values = _reaction_values(update)
    if values is None:
        return None
    return ChannelReaction(
        conversation_id=str(values.chat_id),
        message_id=str(values.message_id),
        emoji=values.emoji,
        sender_id=values.sender_id,
        metadata={
            "provider": "telegram",
            "chat_type": values.chat_type,
        },
    )


class _ReactionValues(NamedTuple):
    """Parsed core fields from a Telegram reaction update payload."""

    chat_id: int
    message_id: int
    chat_type: str
    sender_id: str
    emoji: str


class _MessageValues(NamedTuple):
    """Parsed core fields from a Telegram update payload."""

    msg: Mapping[str, object]
    chat_id: int
    message_id: int
    chat_type: str


def _message_values(
    update: Mapping[str, object],
) -> _MessageValues | None:
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
        logger.debug(
            "Skipping Telegram update with chat type %r (expected %r)",
            chat_type,
            expected_chat_type,
        )
        return None
    chat_id = chat_values.get("id")
    if not isinstance(chat_id, int):
        return None
    message_id = msg.get("message_id")
    if not isinstance(message_id, int):
        return None
    return _MessageValues(
        msg=msg,
        chat_id=chat_id,
        message_id=message_id,
        chat_type=expected_chat_type,
    )


def _reaction_values(update: Mapping[str, object]) -> _ReactionValues | None:
    raw = update.get("message_reaction")
    if not isinstance(raw, dict):
        return None
    reaction = cast("Mapping[str, object]", raw)
    chat = _reaction_chat(reaction)
    message_id = reaction.get("message_id")
    sender_id = _reaction_sender_id(reaction)
    emoji = _reaction_emoji(reaction.get("new_reaction"))
    if chat is None or not isinstance(message_id, int) or sender_id is None or emoji is None:
        return None
    return _ReactionValues(
        chat_id=chat.chat_id,
        message_id=message_id,
        chat_type=chat.chat_type,
        sender_id=sender_id,
        emoji=emoji,
    )


class _ReactionChat(NamedTuple):
    """Parsed chat fields from a Telegram reaction payload."""

    chat_id: int
    chat_type: str


def _reaction_chat(reaction: Mapping[str, object]) -> _ReactionChat | None:
    chat = reaction.get("chat")
    if not isinstance(chat, dict):
        return None
    chat_values = cast("Mapping[str, object]", chat)
    chat_id = chat_values.get("id")
    chat_type = chat_values.get("type")
    if not isinstance(chat_id, int) or not isinstance(chat_type, str):
        return None
    return _ReactionChat(chat_id=chat_id, chat_type=chat_type)


def _sender_id(msg: Mapping[str, object]) -> str | None:
    sender = msg.get("from")
    if isinstance(sender, dict):
        sender_id_raw = cast("Mapping[str, object]", sender).get("id")
        if isinstance(sender_id_raw, int):
            return str(sender_id_raw)
    return None


def _reaction_sender_id(reaction: Mapping[str, object]) -> str | None:
    sender = reaction.get("user")
    if not isinstance(sender, dict):
        return None
    sender_id_raw = cast("Mapping[str, object]", sender).get("id")
    if isinstance(sender_id_raw, int):
        return str(sender_id_raw)
    return None


def _reaction_emoji(value: object) -> str | None:
    if not isinstance(value, list):
        return None
    for item in value:
        if not isinstance(item, dict):
            continue
        reaction = cast("Mapping[str, object]", item)
        if reaction.get("type") != "emoji":
            continue
        emoji = reaction.get("emoji")
        if isinstance(emoji, str) and emoji:
            return emoji
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
    video = msg.get("video") or msg.get("video_note")
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
                mime_type=optional_str(values.get("mime_type")),
            )
    if isinstance(video, dict):
        values = cast("Mapping[str, object]", video)
        file_id = values.get("file_id")
        if isinstance(file_id, str):
            return _TelegramMediaInfo(
                media_type="video",
                file_id=file_id,
                file_name=optional_str(values.get("file_name")),
                mime_type=optional_str(values.get("mime_type")),
            )
    if isinstance(document, dict):
        values = cast("Mapping[str, object]", document)
        file_id = values.get("file_id")
        if isinstance(file_id, str):
            file_name = optional_str(values.get("file_name"))
            mime_type = optional_str(values.get("mime_type"))
            return _TelegramMediaInfo(
                media_type=_document_media_type(file_name=file_name, mime_type=mime_type),
                file_id=file_id,
                file_name=file_name,
                mime_type=mime_type,
            )
    return None


def _document_media_type(*, file_name: str | None, mime_type: str | None) -> str:
    guessed = mimetypes.guess_type(file_name)[0] if file_name else None
    for candidate in (mime_type, guessed):
        if candidate is not None:
            if candidate.startswith("audio/"):
                return "voice"
            if candidate.startswith("video/"):
                return "video"
    return "document"


def _photo_size(size: dict[str, object]) -> int:
    """Extract a numeric file_size from a Telegram photo size object."""
    raw = size.get("file_size")
    return raw if isinstance(raw, int) else 0


def _largest_photo_file_id(photo_sizes: object) -> str | None:
    """Extract the file_id of the largest photo size from a photo array.

    Args:
        photo_sizes: List of photo size objects from a Telegram message.

    Returns:
        File id of the largest photo, or ``None``.
    """
    if not isinstance(photo_sizes, list):
        return None
    sizes = [cast("dict[str, object]", s) for s in photo_sizes if isinstance(s, dict)]
    if not sizes:
        return None
    best = max(sizes, key=_photo_size)
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
    tmp = offset_file.parent / f".{offset_file.name}.{secrets.token_hex(8)}.tmp"
    tmp.write_text(content, encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(offset_file)
