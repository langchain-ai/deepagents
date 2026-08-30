"""Discord channel adapter backed by the `discord.py` Gateway client.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import discord

from deepagents_talon.channels.base import (
    ChannelExposure,
    ChannelExposureEnv,
    ChannelMediaError,
    channel_exposure_from_env,
    chunk_text,
    dispatch_message,
    max_media_bytes_from_env,
    message_with_media_paths,
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
from deepagents_talon.observability import log_debug_event

if TYPE_CHECKING:
    from deepagents_talon.config import TalonConfig

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 2000
"""Discord rejects any single message with more than 2000 characters."""

DEFAULT_MAX_MEDIA_BYTES = 1024 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT_SECONDS = 35.0
OPEN_EXPOSURE_ACK_ENV = "DEEPAGENTS_TALON_DISCORD_OPEN_ACK"

_SAFE_SUFFIX_PATTERN = re.compile(r"\.[a-z0-9]{1,16}")


@dataclass(frozen=True, slots=True)
class DiscordChannelConfig:
    """Configuration for the Discord channel adapter.

    Args:
        bot_token: Discord bot token used to authenticate the Gateway connection.
        inbound_media_dir: Directory where downloaded inbound attachments are stored.
        outbound_media_dir: Optional root that outbound media must remain under
            before it is attached to a message.
        exposure: Inbound trigger policy.
        allowed_user_ids: Discord user ids always allowed to DM the bot, regardless
            of exposure mode.
        max_media_bytes: Maximum media bytes allowed for inbound downloads and
            outbound local files.
        request_timeout_seconds: Timeout for Gateway connect and attachment
            downloads.
    """

    bot_token: str = field(repr=False)
    inbound_media_dir: Path | None = None
    outbound_media_dir: Path | None = None
    exposure: ChannelExposure = field(default_factory=ChannelExposure)
    allowed_user_ids: frozenset[str] = field(default_factory=frozenset)
    max_media_bytes: int = DEFAULT_MAX_MEDIA_BYTES
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS

    @classmethod
    def from_talon_config(cls, config: TalonConfig) -> DiscordChannelConfig:
        """Build Discord channel configuration from Talon environment values.

        Args:
            config: Talon process configuration.

        Returns:
            Discord channel configuration.

        Raises:
            ValueError: If the bot token is missing or exposure configuration is invalid.
        """
        env = config.env
        token = env.get("DEEPAGENTS_TALON_DISCORD_BOT_TOKEN")
        if not token:
            msg = "Discord bot token is required (DEEPAGENTS_TALON_DISCORD_BOT_TOKEN)"
            raise ValueError(msg)
        inbound_media_dir = Path(
            env.get(
                "DEEPAGENTS_TALON_DISCORD_MEDIA_DIR",
                str(config.inbound_media_dir / "discord"),
            ),
        )
        exposure = channel_exposure_from_env(
            env,
            ChannelExposureEnv(
                provider="Discord",
                env_prefix="DEEPAGENTS_TALON_DISCORD",
                open_ack=OPEN_EXPOSURE_ACK_ENV,
                require_self_operator=True,
            ),
        )
        return cls(
            bot_token=token,
            inbound_media_dir=inbound_media_dir,
            outbound_media_dir=outbound_media_root_from_env(env),
            exposure=exposure,
            allowed_user_ids=frozenset(
                split_csv(env.get("DEEPAGENTS_TALON_DISCORD_ALLOWLIST_USERS", "")),
            ),
            max_media_bytes=max_media_bytes_from_env(env),
            request_timeout_seconds=parse_float(
                env.get("DEEPAGENTS_TALON_DISCORD_REQUEST_TIMEOUT_SECONDS"),
                DEFAULT_REQUEST_TIMEOUT_SECONDS,
            ),
        )


@dataclass(frozen=True, slots=True)
class _DiscordAttachment:
    """Metadata for one inbound Discord attachment."""

    url: str
    filename: str
    content_type: str | None
    size: int


@dataclass(frozen=True, slots=True)
class _DiscordInboundMessage:
    """Provider-neutral view of a Gateway `on_message` event."""

    channel_id: str
    message_id: str
    sender_id: str | None
    text: str
    is_dm: bool
    from_self: bool
    attachments: tuple[_DiscordAttachment, ...] = ()


@dataclass(frozen=True, slots=True)
class _DiscordInboundReaction:
    """Provider-neutral view of a Gateway `on_raw_reaction_add` event."""

    channel_id: str
    message_id: str
    sender_id: str | None
    emoji: str


InboundMessageCallback = Callable[[_DiscordInboundMessage], Awaitable[None]]
InboundReactionCallback = Callable[[_DiscordInboundReaction], Awaitable[None]]


class _DiscordGateway(Protocol):
    """Narrow surface `DiscordChannel` needs from a Discord client implementation.

    Production code implements this with a real `discord.py` `Client`; tests
    inject a fake so unit tests never open a real Gateway connection.
    """

    @property
    def bot_id(self) -> str | None:
        """Authenticated bot user id once the Gateway connection is ready."""

    async def start(
        self,
        *,
        handle_message: InboundMessageCallback,
        handle_reaction: InboundReactionCallback,
    ) -> None:
        """Connect to the Gateway and begin dispatching inbound events."""

    async def stop(self) -> None:
        """Disconnect from the Gateway and release resources."""

    async def send_message(self, channel_id: str, text: str) -> str:
        """Send a text-only message and return the new message id."""

    async def send_file(
        self,
        channel_id: str,
        file_path: Path,
        *,
        content: str | None,
    ) -> str:
        """Send a file attachment with optional message content."""

    async def edit_message(self, channel_id: str, message_id: str, text: str) -> None:
        """Edit a previously sent message's content."""

    async def trigger_typing(self, channel_id: str) -> None:
        """Send a one-shot typing indicator."""


class _DiscordPyGateway:
    """Gateway implementation backed by the `discord.py` library."""

    def __init__(self, *, token: str, connect_timeout_seconds: float) -> None:
        self._token = token
        self._connect_timeout_seconds = connect_timeout_seconds
        self._client: discord.Client | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def bot_id(self) -> str | None:
        if self._client is None or self._client.user is None:
            return None
        return str(self._client.user.id)

    async def start(
        self,
        *,
        handle_message: InboundMessageCallback,
        handle_reaction: InboundReactionCallback,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)

        @client.event
        async def on_message(message: discord.Message) -> None:
            await handle_message(_convert_message(message, bot_id=self.bot_id))

        @client.event
        async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
            reaction = _convert_reaction(payload)
            if reaction is not None:
                await handle_reaction(reaction)

        self._client = client
        task = asyncio.create_task(client.start(self._token), name="talon:discord:gateway")
        self._task = task
        ready_task = asyncio.create_task(client.wait_until_ready())
        done, pending = await asyncio.wait(
            {task, ready_task},
            timeout=self._connect_timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if ready_task in done:
            if task in pending:
                # Gateway connected; leave it running for the adapter's lifetime.
                return
            await task
            return
        if task in done:
            ready_task.cancel()
            await task
            return
        task.cancel()
        ready_task.cancel()
        msg = "Timed out connecting to the Discord Gateway"
        raise TimeoutError(msg)

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.close()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
        self._client = None
        self._task = None

    async def send_message(self, channel_id: str, text: str) -> str:
        channel = await self._resolve_channel(channel_id)
        message = await channel.send(content=text)
        return str(message.id)

    async def send_file(
        self,
        channel_id: str,
        file_path: Path,
        *,
        content: str | None,
    ) -> str:
        channel = await self._resolve_channel(channel_id)
        message = await channel.send(content=content, file=discord.File(file_path))
        return str(message.id)

    async def edit_message(self, channel_id: str, message_id: str, text: str) -> None:
        channel = await self._resolve_channel(channel_id)
        message = await channel.fetch_message(int(message_id))
        await message.edit(content=text)

    async def trigger_typing(self, channel_id: str) -> None:
        channel = await self._resolve_channel(channel_id)
        async with channel.typing():
            pass

    async def _resolve_channel(self, channel_id: str) -> discord.abc.Messageable:
        if self._client is None:
            msg = "Discord gateway is not started"
            raise RuntimeError(msg)
        channel = self._client.get_channel(int(channel_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(channel_id))
        # Configured channel ids are DMs or guild text channels, which are
        # Messageable; forum/category channels are not valid send targets here.
        return cast("discord.abc.Messageable", channel)


class DiscordChannel:
    """Channel adapter for Discord via the `discord.py` Gateway client.

    Both DM channels and guild text channels are processed. Guild channels are
    subject to the same exposure policy as DMs, scoped by channel id through
    `DEEPAGENTS_TALON_DISCORD_ALLOWLIST_CHATS`.
    """

    def __init__(
        self,
        config: DiscordChannelConfig,
        *,
        gateway: _DiscordGateway | None = None,
    ) -> None:
        """Initialize the channel.

        Args:
            config: Discord channel configuration.
            gateway: Optional injectable gateway, used to avoid real Gateway
                connections in tests. Defaults to a real `discord.py`-backed
                gateway.
        """
        self.config = config
        self._gateway = gateway or _DiscordPyGateway(
            token=config.bot_token,
            connect_timeout_seconds=config.request_timeout_seconds,
        )
        self._handler: MessageHandler | None = None
        self._reaction_handler: ReactionHandler | None = None
        self._exposure = config.exposure
        self._status = ChannelStatus(provider="discord", connected=False, detail="disconnected")

    def set_message_handler(self, handler: MessageHandler) -> None:
        """Register the host callback for inbound messages.

        Args:
            handler: Coroutine callback invoked for each inbound channel message.
        """
        self._handler = handler

    def set_reaction_handler(self, handler: ReactionHandler) -> None:
        """Register the host callback for inbound reactions.

        Args:
            handler: Coroutine callback invoked for each inbound channel reaction.
        """
        self._reaction_handler = handler

    async def start(self) -> None:
        """Connect to the Discord Gateway and begin receiving events."""
        log_debug_event(
            logger,
            "discord.channel.starting",
            exposure=self._exposure.mode.value,
            inbound_media_enabled=self.config.inbound_media_dir is not None,
        )
        if self.config.inbound_media_dir is not None:
            self.config.inbound_media_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.config.inbound_media_dir.chmod(0o700)
        await self._gateway.start(
            handle_message=self._process_message,
            handle_reaction=self._process_reaction,
        )
        self._status = ChannelStatus(provider="discord", connected=True, detail="connected")
        log_debug_event(logger, "discord.channel.started", connected=True)

    async def stop(self) -> None:
        """Disconnect from the Discord Gateway and release resources."""
        log_debug_event(logger, "discord.channel.stopping")
        await self._gateway.stop()
        self._status = ChannelStatus(provider="discord", connected=False, detail="disconnected")
        log_debug_event(logger, "discord.channel.stopped")

    async def send_message(self, conversation_id: str, text: str) -> SendResult:
        """Send a message, splitting text over 2000 characters across multiple sends.

        Args:
            conversation_id: Discord channel id.
            text: Message content to send.

        Returns:
            Result indicating whether the last chunk send succeeded.
        """
        chunks = chunk_text(text, limit=MAX_TEXT_CHARS)
        log_debug_event(
            logger,
            "discord.outbound.text.started",
            chunk_count=len(chunks),
            text_chars=len(text),
        )
        message_id: str | None = None
        for chunk in chunks:
            message_id = await self._gateway.send_message(conversation_id, chunk)
        log_debug_event(
            logger,
            "discord.outbound.text.completed",
            chunk_count=len(chunks),
            message_id_present=message_id is not None,
        )
        return SendResult(success=True, message_id=message_id)

    async def send_media(self, conversation_id: str, media: ChannelMedia) -> SendResult:
        """Send media as a file attachment with an optional caption.

        Args:
            conversation_id: Discord channel id.
            media: Media payload to deliver.

        Returns:
            Result indicating whether the send succeeded.
        """
        checked = validate_media(
            media,
            root=self.config.outbound_media_dir,
            max_bytes=self.config.max_media_bytes,
        )
        content = await self._media_content(conversation_id, checked.caption)
        log_debug_event(
            logger,
            "discord.outbound.media.started",
            caption_present=content is not None,
            media_type=checked.media_type,
        )
        message_id = await self._gateway.send_file(conversation_id, checked.path, content=content)
        log_debug_event(
            logger,
            "discord.outbound.media.completed",
            media_type=checked.media_type,
            message_id_present=message_id is not None,
        )
        return SendResult(success=True, message_id=message_id)

    async def edit_message(self, conversation_id: str, message_id: str, text: str) -> SendResult:
        """Edit a previously sent message.

        Args:
            conversation_id: Discord channel id.
            message_id: Discord message id.
            text: Replacement message content.

        Returns:
            Result indicating whether the edit succeeded.
        """
        await self._gateway.edit_message(conversation_id, message_id, text)
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, conversation_id: str) -> None:
        """Send a one-shot typing indicator.

        Args:
            conversation_id: Discord channel id.
        """
        try:
            await self._gateway.trigger_typing(conversation_id)
        except Exception as error:  # noqa: BLE001  # transport errors must not crash the host loop
            log_debug_event(
                logger,
                "discord.outbound.typing.failed",
                error_type=type(error).__name__,
            )

    async def status(self) -> ChannelStatus:
        """Report the channel connection status."""
        return self._status

    async def _media_content(self, conversation_id: str, caption: str | None) -> str | None:
        if not caption:
            return None
        if len(caption) <= MAX_TEXT_CHARS:
            return caption
        await self.send_message(conversation_id, caption)
        return None

    async def _process_message(self, inbound: _DiscordInboundMessage) -> None:
        if inbound.from_self:
            # Discord's Gateway re-delivers the bot's own outbound messages through
            # on_message. Unlike Telegram/WhatsApp, the bot identity here is the
            # transport itself, not an operator account, so self-authored events
            # must never reach exposure checks -- admitting them would redispatch
            # every reply as a new prompt, looping forever.
            return
        message = ChannelMessage(
            conversation_id=inbound.channel_id,
            text=inbound.text,
            sender_id=inbound.sender_id,
            message_id=inbound.message_id,
            metadata=_message_metadata(inbound),
        )
        if not _allows_discord_message(self._exposure, self.config.allowed_user_ids, message):
            log_debug_event(
                logger,
                "discord.inbound.message.rejected",
                exposure=self._exposure.mode.value,
                has_media=bool(inbound.attachments),
            )
            return
        message = await self._prepare_inbound_media(message, inbound.attachments)
        log_debug_event(
            logger,
            "discord.inbound.message.dispatching",
            has_media=bool(message.metadata.get("has_media")),
        )
        await dispatch_message(self._handler, message, provider="Discord")
        log_debug_event(logger, "discord.inbound.message.dispatched")

    async def _process_reaction(self, inbound: _DiscordInboundReaction) -> None:
        reaction = ChannelReaction(
            conversation_id=inbound.channel_id,
            message_id=inbound.message_id,
            emoji=inbound.emoji,
            sender_id=inbound.sender_id,
            metadata={"provider": "discord"},
        )
        if not _allows_discord_reaction(self._exposure, self.config.allowed_user_ids, reaction):
            log_debug_event(
                logger,
                "discord.inbound.reaction.rejected",
                exposure=self._exposure.mode.value,
            )
            return
        if self._reaction_handler is None:
            logger.warning("Dropping Discord reaction because no handler is registered")
            return
        log_debug_event(logger, "discord.inbound.reaction.dispatching")
        await self._reaction_handler(reaction)
        log_debug_event(logger, "discord.inbound.reaction.dispatched")

    async def _prepare_inbound_media(
        self,
        message: ChannelMessage,
        attachments: tuple[_DiscordAttachment, ...],
    ) -> ChannelMessage:
        if not attachments or self.config.inbound_media_dir is None:
            return message
        attachment = attachments[0]
        if attachment.size > self.config.max_media_bytes:
            logger.warning("Skipping Discord inbound media because it exceeds the size cap")
            return _with_media_error(
                message,
                f"media file is too large: {attachment.size} bytes "
                f"exceeds {self.config.max_media_bytes}",
            )
        try:
            destination = await self._download_attachment(attachment, message_id=message.message_id)
        except (ChannelMediaError, OSError, urllib.error.URLError, TimeoutError) as error:
            logger.warning("Skipping Discord inbound media after download failure")
            return _with_media_error(message, str(error))
        mime_type = attachment.content_type or mimetypes.guess_type(destination.name)[0]
        return message_with_media_paths(
            message,
            media_paths=[str(destination)],
            mime_types=[mime_type] if mime_type else [],
        )

    async def _download_attachment(
        self,
        attachment: _DiscordAttachment,
        *,
        message_id: str | None,
    ) -> Path:
        if self.config.inbound_media_dir is None:
            msg = "Discord inbound media directory is not configured"
            raise ChannelMediaError(msg)
        suffix = _safe_suffix(attachment.filename, attachment.content_type)
        destination = self.config.inbound_media_dir / _inbound_media_filename(
            message_id=message_id,
            attachment_url=attachment.url,
            suffix=suffix,
        )
        await asyncio.to_thread(
            _download_attachment_file,
            attachment.url,
            destination,
            self.config.request_timeout_seconds,
            self.config.max_media_bytes,
        )
        return destination


def _with_media_error(message: ChannelMessage, error: str) -> ChannelMessage:
    metadata = dict(message.metadata)
    metadata["has_media"] = False
    metadata["media_error"] = error
    return replace(message, metadata=metadata)


def _convert_message(message: discord.Message, *, bot_id: str | None) -> _DiscordInboundMessage:
    attachments = tuple(
        _DiscordAttachment(
            url=attachment.url,
            filename=attachment.filename,
            content_type=attachment.content_type,
            size=attachment.size,
        )
        for attachment in message.attachments
    )
    sender_id = str(message.author.id)
    return _DiscordInboundMessage(
        channel_id=str(message.channel.id),
        message_id=str(message.id),
        sender_id=sender_id,
        text=message.content,
        is_dm=message.guild is None,
        from_self=bot_id is not None and sender_id == bot_id,
        attachments=attachments,
    )


def _convert_reaction(payload: discord.RawReactionActionEvent) -> _DiscordInboundReaction | None:
    emoji = payload.emoji.name if payload.emoji is not None else None
    if not emoji:
        return None
    user_id = payload.user_id
    return _DiscordInboundReaction(
        channel_id=str(payload.channel_id),
        message_id=str(payload.message_id),
        sender_id=str(user_id) if user_id is not None else None,
        emoji=emoji,
    )


def _message_metadata(inbound: _DiscordInboundMessage) -> dict[str, object]:
    metadata: dict[str, object] = {
        "provider": "discord",
        "is_dm": inbound.is_dm,
        "from_self": inbound.from_self,
    }
    if inbound.attachments:
        attachment = inbound.attachments[0]
        metadata["media_type"] = _attachment_media_type(attachment)
        if attachment.content_type:
            metadata["mime_type"] = attachment.content_type
    return metadata


def _attachment_media_type(attachment: _DiscordAttachment) -> str:
    content_type = (attachment.content_type or "").lower()
    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("video/"):
        return "video"
    if content_type.startswith("audio/"):
        return "voice" if _looks_like_voice_message(attachment.filename) else "audio"
    return "document"


def _looks_like_voice_message(filename: str) -> bool:
    return Path(filename).stem.startswith("voice-message")


def _allows_discord_message(
    exposure: ChannelExposure,
    allowed_user_ids: frozenset[str],
    message: ChannelMessage,
) -> bool:
    if (
        exposure.mode.value == "allowlist"
        and message.metadata.get("is_dm") is True
        and message.sender_id in allowed_user_ids
    ):
        return True
    return exposure.allows(message)


def _allows_discord_reaction(
    exposure: ChannelExposure,
    allowed_user_ids: frozenset[str],
    reaction: ChannelReaction,
) -> bool:
    if reaction.sender_id is None:
        return False
    return reaction.sender_id in exposure.operator_ids or reaction.sender_id in allowed_user_ids


def _safe_suffix(filename: str, content_type: str | None) -> str:
    suffix = Path(filename).suffix.lower()
    if _SAFE_SUFFIX_PATTERN.fullmatch(suffix):
        return suffix
    if content_type:
        guessed = mimetypes.guess_extension(content_type)
        if guessed:
            return guessed
    return ".bin"


def _inbound_media_filename(*, message_id: str | None, attachment_url: str, suffix: str) -> str:
    message = _safe_filename_part(message_id or "message")
    token = _safe_filename_part(attachment_url)[-24:] or "file"
    return f"{message}_{token}{suffix}"


def _safe_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "file"


def _download_attachment_file(url: str, destination: Path, timeout: float, max_bytes: int) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    request = urllib.request.Request(url)  # noqa: S310  # Discord attachment CDN URL
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        length = response.headers.get("content-length")
        if length is not None:
            expected = _parse_content_length(length)
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


def _parse_content_length(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None
