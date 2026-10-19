"""Discord channel adapter backed by the `discord.py` Gateway client.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import mimetypes
import re
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import discord
from discord import app_commands

from deepagents_talon.channels.base import (
    ChannelExposure,
    ChannelExposureEnv,
    ChannelMediaError,
    ExposureMode,
    channel_exposure_from_env,
    chunk_text,
    dispatch_message,
    max_media_bytes_from_env,
    message_with_media_paths,
    outbound_media_root_from_env,
    parse_content_length,
    parse_float,
    safe_filename_part,
    safe_suffix,
    split_csv,
    validate_media,
    with_media_error,
)
from deepagents_talon.commands import COMMANDS_BY_NAME, ChatCommand, visible_commands
from deepagents_talon.interfaces import (
    ChannelMedia,
    ChannelMessage,
    ChannelReaction,
    ChannelStatus,
    MessageHandler,
    ReactionHandler,
    SendResult,
)
from deepagents_talon.mcp_auth import extract_loopback_oauth_callback_url
from deepagents_talon.observability import log_debug_event
from deepagents_talon.pairing import SenderPairing, pairing_from_env

if TYPE_CHECKING:
    from deepagents_talon.config import TalonConfig

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 2000
"""Discord rejects any single message with more than 2000 characters."""

DEFAULT_MAX_MEDIA_BYTES = 1024 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT_SECONDS = 35.0
OPEN_EXPOSURE_ACK_ENV = "DEEPAGENTS_TALON_DISCORD_OPEN_ACK"
SLASH_COMMANDS_ENV = "DEEPAGENTS_TALON_DISCORD_SLASH_COMMANDS"
COMMAND_GUILD_ID_ENV = "DEEPAGENTS_TALON_DISCORD_COMMAND_GUILD_ID"
INCLUDE_OTHER_THREAD_PARTICIPANTS_ENV = "DEEPAGENTS_TALON_DISCORD_INCLUDE_OTHER_THREAD_PARTICIPANTS"
MENTION_OPTIONAL_CHATS_ENV = "DEEPAGENTS_TALON_DISCORD_MENTION_OPTIONAL_CHATS"
ALL_CHATS = "*"
"""Entry in `mention_optional_chats` that makes mentions optional in every guild channel."""
_ENV_PREFIX = "DEEPAGENTS_TALON_DISCORD"

_COMMAND_UNAVAILABLE_MESSAGE = "That command is not available here."
_UNAUTHORIZED_MESSAGE = "This assistant does not accept commands from you."
_COMMAND_NO_REPLY_MESSAGE = "Done."
_COMMAND_FAILED_MESSAGE = "Something went wrong running that command. Check Talon logs."
_THREAD_NAME_CHARS = 100
"""Discord rejects a thread name longer than 100 characters."""

_DEFAULT_THREAD_NAME = "Talon"
_THREAD_CONTEXT_MESSAGES = 40
_THREAD_CONTEXT_MESSAGE_CHARS = 1000
_THREAD_CONTEXT_CHARS = 12000
_THREAD_CONTEXT_UNAVAILABLE = "[Discord thread history unavailable; ask for context before acting.]"
_CONTEXT_MESSAGE_TYPES = frozenset({discord.MessageType.default, discord.MessageType.reply})
"""Message types that carry user-authored text, as opposed to system notices."""

_OAUTH_CANDIDATE_PATTERN = re.compile(r"http://[^\s<>\"'`]+", flags=re.IGNORECASE)
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSY_ENV_VALUES = frozenset({"0", "false", "no", "off"})


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
        slash_commands_enabled: Whether to register Talon's commands as Discord
            application commands so they appear in the native `/` picker.
        command_guild_id: Optional guild id to scope command registration to.
            A guild-scoped registration applies immediately, which is useful while
            developing, but by construction it never reaches direct messages.
        pairing: Optional sender pairing policy that admits approved DM senders
            and issues codes to unknown ones.
        include_other_thread_participants: Include thread history from
            non-allowlisted senders, accepting their potential influence on
            privileged tools.
        mention_optional_chats: Guild channel ids where a human's message is a
            turn without mentioning the bot, in the channel and its threads;
            `*` means every guild channel. Only the mention requirement is
            lifted: the exposure policy still decides who is admitted.
    """

    bot_token: str = field(repr=False)
    inbound_media_dir: Path | None = None
    outbound_media_dir: Path | None = None
    exposure: ChannelExposure = field(default_factory=ChannelExposure)
    allowed_user_ids: frozenset[str] = field(default_factory=frozenset)
    max_media_bytes: int = DEFAULT_MAX_MEDIA_BYTES
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    slash_commands_enabled: bool = True
    command_guild_id: str | None = None
    pairing: SenderPairing | None = None
    include_other_thread_participants: bool = field(default=False, kw_only=True)
    mention_optional_chats: frozenset[str] = field(default_factory=frozenset, kw_only=True)

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
        include_participants = env.get(INCLUDE_OTHER_THREAD_PARTICIPANTS_ENV, "0")
        if include_participants not in {"0", "1"}:
            msg = f"{INCLUDE_OTHER_THREAD_PARTICIPANTS_ENV} must be 0 or 1"
            raise ValueError(msg)
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
                env_prefix=_ENV_PREFIX,
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
            slash_commands_enabled=_parse_flag(env.get(SLASH_COMMANDS_ENV), default=True),
            command_guild_id=_parse_guild_id(env.get(COMMAND_GUILD_ID_ENV)),
            pairing=pairing_from_env(
                env,
                provider="discord",
                env_prefix=_ENV_PREFIX,
                open_exposure=exposure.mode == ExposureMode.OPEN,
                home=config.home,
            ),
            include_other_thread_participants=include_participants == "1",
            mention_optional_chats=_parse_mention_optional_chats(
                env.get(MENTION_OPTIONAL_CHATS_ENV, "")
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
    """Provider-neutral view of a Gateway `on_message` event.

    Args:
        channel_id: Channel or thread the message was posted in.
        message_id: Discord's id for the message.
        sender_id: Discord user id of the author.
        text: Message text with the bot's own mentions removed.
        is_dm: Whether the message arrived outside a guild.
        from_self: Whether this bot authored the message.
        attachments: Files attached to the message.
        history_chat: Parent channel whose archive a public thread shares.
        mentioned: Whether the message mentions the bot, directly or through its
            integration role.
        parent_id: Parent channel of the thread the message was posted in, for
            public and private threads alike; `None` outside a thread.
        from_bot: Whether a bot account, this one or another, authored the message.
    """

    channel_id: str
    message_id: str
    sender_id: str | None
    text: str
    is_dm: bool
    from_self: bool
    attachments: tuple[_DiscordAttachment, ...] = ()
    history_chat: str | None = None
    mentioned: bool = False
    parent_id: str | None = None
    from_bot: bool = False


@dataclass(frozen=True, slots=True)
class _DiscordContextMessage:
    """One earlier thread message offered to the agent as context."""

    sender_id: str
    text: str
    from_bot: bool


@dataclass(frozen=True, slots=True)
class _DiscordInboundReaction:
    """Provider-neutral view of a Gateway `on_raw_reaction_add` event."""

    channel_id: str
    message_id: str
    sender_id: str | None
    emoji: str


@dataclass(frozen=True, slots=True)
class _DiscordConnectionState:
    """Provider-neutral view of a Gateway connection lifecycle event."""

    connected: bool
    detail: str


class _InteractionResponder(Protocol):
    """Reply surface for one Discord application command invocation.

    Discord requires a response to every interaction, and only the first response
    may be immediate: anything sent after a deferral is a followup. `reject`
    answers without deferring, while `defer` plus `send` covers work that may
    outlast Discord's three-second initial-response budget.
    """

    async def reject(self, text: str) -> None:
        """Answer immediately and privately, without deferring."""

    async def defer(self) -> None:
        """Acknowledge the interaction so a reply can follow later."""

    async def send(self, text: str) -> str | None:
        """Send a followup reply and return its message id when one is reported."""


@dataclass(frozen=True, slots=True)
class _DiscordInboundInteraction:
    """Provider-neutral view of one inbound application command invocation.

    Args:
        command: Invoked command's bare name, without a leading slash.
        channel_id: Channel the command was invoked in, when Discord reports one.
        sender_id: Discord user id that invoked the command.
        interaction_id: Discord's id for this invocation.
        is_dm: Whether the command was invoked outside a guild.
        responder: Reply surface bound to this invocation.
        argument: Text of the command's optional argument, when one was given.
        history_chat: Parent channel whose archive a public thread shares.
        parent_id: Parent channel of the thread the command was invoked in.
    """

    command: str
    channel_id: str | None
    sender_id: str | None
    interaction_id: str
    is_dm: bool
    responder: _InteractionResponder
    argument: str | None = None
    history_chat: str | None = None
    parent_id: str | None = None


@dataclass(slots=True)
class _InteractionSink:
    """Routes one command's reply back to the interaction that asked for it.

    Args:
        conversation_id: Channel whose replies belong to this interaction.
        responder: Reply surface for the invocation.
        used: Whether a reply has been routed, so the caller knows if it still
            owes Discord a followup.
    """

    conversation_id: str
    responder: _InteractionResponder
    used: bool = False


_INTERACTION_SINK: ContextVar[_InteractionSink | None] = ContextVar(
    "talon_discord_interaction_sink",
    default=None,
)
"""Reply sink for the application command being handled on this task, if any.

A context variable rather than a table keyed by conversation: `discord.py` runs
every interaction in its own task, so each invocation sees only its own sink and
no other task -- a cron delivery, a background result, a progress message aimed
at the same channel -- can be captured by one.
"""

InboundMessageCallback = Callable[[_DiscordInboundMessage], Awaitable[None]]
InboundReactionCallback = Callable[[_DiscordInboundReaction], Awaitable[None]]
InboundConnectionCallback = Callable[[_DiscordConnectionState], Awaitable[None]]
InboundInteractionCallback = Callable[[_DiscordInboundInteraction], Awaitable[None]]

_CONNECTED_STATE = _DiscordConnectionState(connected=True, detail="connected")
_RECONNECTING_STATE = _DiscordConnectionState(connected=True, detail="reconnecting")


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
        handle_connection: InboundConnectionCallback,
        handle_interaction: InboundInteractionCallback,
    ) -> None:
        """Connect to the Gateway and begin dispatching inbound and connection events."""

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

    async def create_thread(self, channel_id: str, message_id: str, name: str) -> str:
        """Start a public thread from a message and return the thread id."""

    async def thread_context(
        self, thread_id: str, before_message_id: str
    ) -> list[_DiscordContextMessage]:
        """Read bounded thread messages preceding a message, oldest first."""


class _DiscordPyGateway:
    """Gateway implementation backed by the `discord.py` library."""

    def __init__(
        self,
        *,
        token: str,
        connect_timeout_seconds: float,
        commands_enabled: bool = True,
        command_guild_id: str | None = None,
    ) -> None:
        self._token = token
        self._connect_timeout_seconds = connect_timeout_seconds
        self._commands_enabled = commands_enabled
        self._command_guild_id = command_guild_id
        self._client: discord.Client | None = None
        self._task: asyncio.Task[None] | None = None
        self._watch: asyncio.Task[None] | None = None
        self._tree: app_commands.CommandTree[discord.Client] | None = None
        self._commands_synced = False

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
        handle_connection: InboundConnectionCallback,
        handle_interaction: InboundInteractionCallback,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)
        self._register_commands(client, handle_interaction)
        self._register_events(
            client,
            handle_message=handle_message,
            handle_reaction=handle_reaction,
            handle_connection=handle_connection,
        )
        self._client = client
        task = asyncio.create_task(
            client.start(self._token, reconnect=True),
            name="talon:discord:gateway",
        )
        self._task = task
        ready_task = asyncio.create_task(client.wait_until_ready())
        done, pending = await asyncio.wait(
            {task, ready_task},
            timeout=self._connect_timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if ready_task in done:
            if task in pending:
                # Gateway connected; watch the task so a later failure is not silent.
                self._watch = asyncio.create_task(
                    self._watch_gateway(task, handle_connection),
                    name="talon:discord:gateway-watch",
                )
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

    def _register_commands(
        self,
        client: discord.Client,
        handle_interaction: InboundInteractionCallback,
    ) -> None:
        """Build the application command tree for every advertised command.

        Registration with Discord happens later, in `_sync_commands`, because the
        sync call needs an application id that only exists once the Gateway is ready.

        Args:
            client: Client the command tree is attached to.
            handle_interaction: Callback invoked for each command invocation.
        """
        if not self._commands_enabled:
            return
        self._tree = app_commands.CommandTree(
            client,
            # Set once on the tree: `discord.py` merges these into every command
            # payload. `dm_channel` is what surfaces the commands in an operator's
            # direct messages, which is the only place the default `self` exposure
            # accepts anything. Guild install only -- a user install is rejected
            # unless it is also enabled in the Discord developer portal.
            allowed_contexts=app_commands.AppCommandContext(
                guild=True,
                dm_channel=True,
                private_channel=True,
            ),
            allowed_installs=app_commands.AppInstallationType(guild=True),
        )
        for command in visible_commands():
            self._tree.add_command(_build_app_command(command, handle_interaction))

    async def _sync_commands(self) -> None:
        """Register this process's commands with Discord exactly once."""
        if self._tree is None or self._commands_synced:
            return
        # Claimed before the first await: `on_ready` fires again after every
        # reconnect, and `discord.py` dispatches each one as its own task, so two
        # of them can reach this point concurrently.
        self._commands_synced = True
        guild = (
            discord.Object(id=int(self._command_guild_id))
            if self._command_guild_id is not None
            else None
        )
        try:
            await self._tree.sync(guild=guild)
        except Exception:  # noqa: BLE001  # A failed registration must not stop the channel.
            logger.warning("Could not register Discord application commands", exc_info=True)
            return
        log_debug_event(
            logger,
            "discord.commands.registered",
            command_count=len(visible_commands()),
            guild_scoped=guild is not None,
        )

    def _register_events(
        self,
        client: discord.Client,
        *,
        handle_message: InboundMessageCallback,
        handle_reaction: InboundReactionCallback,
        handle_connection: InboundConnectionCallback,
    ) -> None:
        """Register the Gateway event handlers `DiscordChannel` depends on.

        `discord.py` keys handlers off the callback name, so each function must keep
        the name of the event it serves.

        Args:
            client: Client whose Gateway events are being subscribed to.
            handle_message: Callback invoked for each inbound message.
            handle_reaction: Callback invoked for each inbound reaction.
            handle_connection: Callback invoked when the connection state changes.
        """

        @client.event
        async def on_message(message: discord.Message) -> None:
            await handle_message(_convert_message(message, bot_id=self.bot_id))

        @client.event
        async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
            reaction = _convert_reaction(payload)
            if reaction is not None:
                await handle_reaction(reaction)

        @client.event
        async def on_ready() -> None:
            # Connection state first: registering commands is a network round trip,
            # and a channel that reports itself connected is not waiting on it.
            await handle_connection(_CONNECTED_STATE)
            await self._sync_commands()

        @client.event
        async def on_resumed() -> None:
            await handle_connection(_CONNECTED_STATE)

        @client.event
        async def on_disconnect() -> None:
            # `discord.py` reconnects on its own for every disconnect that reaches
            # here, so this is a detail-only transition; `connected` stays true
            # until the gateway task itself terminates.
            await handle_connection(_RECONNECTING_STATE)

    async def _watch_gateway(
        self,
        task: asyncio.Task[None],
        handle_connection: InboundConnectionCallback,
    ) -> None:
        """Report the gateway task's terminal outcome instead of discarding it.

        Awaiting the task retrieves its exception, so an unrecoverable Gateway
        failure -- a revoked token closing with 4004, or 4014 once privileged
        intents are withdrawn -- is logged and reflected in the channel status
        rather than silently ending inbound delivery.

        Args:
            task: The running `client.start` task.
            handle_connection: Callback invoked with the terminal state.
        """
        try:
            await task
        except asyncio.CancelledError:
            raise
        # A narrower catch would restore the silent-death bug this watcher exists
        # to fix: any type that escaped here would end inbound delivery unreported.
        except Exception as error:
            logger.exception("Discord Gateway connection ended unexpectedly")
            detail = f"gateway stopped: {type(error).__name__}"
        else:
            detail = "gateway stopped"
        await handle_connection(_DiscordConnectionState(connected=False, detail=detail))

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.close()
        if self._watch is not None:
            self._watch.cancel()
            await asyncio.gather(self._watch, return_exceptions=True)
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
        self._client = None
        self._task = None
        self._watch = None

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

    async def create_thread(self, channel_id: str, message_id: str, name: str) -> str:
        channel = await self._fetch_channel(channel_id)
        if not isinstance(channel, discord.TextChannel):
            # A voice channel's text chat, for one, has no threads.
            msg = "Discord channel does not support threads"
            raise discord.ClientException(msg)
        thread = await channel.get_partial_message(int(message_id)).create_thread(name=name)
        return str(thread.id)

    async def thread_context(
        self, thread_id: str, before_message_id: str
    ) -> list[_DiscordContextMessage]:
        thread = await self._fetch_channel(thread_id)
        if not isinstance(thread, discord.Thread):
            return []
        history = [
            message
            async for message in thread.history(
                limit=_THREAD_CONTEXT_MESSAGES,
                before=discord.Object(id=int(before_message_id)),
            )
        ]
        history.reverse()
        if not any(message.id == thread.id for message in history):
            # A thread started from a channel message keeps that message in the
            # parent channel, not in its own history.
            starter = await _thread_starter(thread)
            if starter is not None:
                history.insert(0, starter)
        return [
            _DiscordContextMessage(
                sender_id=str(message.author.id),
                text=message.content,
                from_bot=message.author.bot,
            )
            for message in history
            if message.type in _CONTEXT_MESSAGE_TYPES and message.content
        ]

    async def _resolve_channel(self, channel_id: str) -> discord.abc.Messageable:
        # Configured channel ids are DMs, guild text channels, or threads, which
        # are Messageable; forum/category channels are not valid send targets here.
        return cast("discord.abc.Messageable", await self._fetch_channel(channel_id))

    async def _fetch_channel(self, channel_id: str) -> object:
        if self._client is None:
            msg = "Discord gateway is not started"
            raise RuntimeError(msg)
        channel = self._client.get_channel(int(channel_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(channel_id))
        return channel


async def _thread_starter(thread: discord.Thread) -> discord.Message | None:
    """Return the channel message a thread was started from, when there is one."""
    if thread.starter_message is not None:
        return thread.starter_message
    if not isinstance(thread.parent, discord.TextChannel):
        return None
    try:
        return await thread.parent.fetch_message(thread.id)
    except discord.NotFound, discord.Forbidden:
        # A thread not started from a message, a starter since deleted, or a
        # parent channel the bot cannot read; the thread's own history still counts.
        return None


class DiscordChannel:
    """Channel adapter for Discord via the `discord.py` Gateway client.

    A direct message with the bot is one conversation. In a guild the bot only
    answers when mentioned: a mention in a channel opens a thread on the
    mentioning message and replies there, and a mention inside an existing thread
    replies in that thread. Each thread is its own conversation, so follow-ups
    mention the bot in-thread. Guild messages are subject to the same exposure
    policy as DMs; an allowlisted channel also admits every thread in it.
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
            commands_enabled=config.slash_commands_enabled,
            command_guild_id=config.command_guild_id,
        )
        self._handler: MessageHandler | None = None
        self._reaction_handler: ReactionHandler | None = None
        self._exposure = config.exposure
        self._status = ChannelStatus(provider="discord", connected=False, detail="disconnected")
        self._stopping = False

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
        self._stopping = False
        await self._gateway.start(
            handle_message=self._process_message,
            handle_reaction=self._process_reaction,
            handle_connection=self._process_connection,
            handle_interaction=self._process_interaction,
        )
        self._status = ChannelStatus(provider="discord", connected=True, detail="connected")
        log_debug_event(logger, "discord.channel.started", connected=True)

    async def stop(self) -> None:
        """Disconnect from the Discord Gateway and release resources."""
        log_debug_event(logger, "discord.channel.stopping")
        self._stopping = True
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
        sink = _INTERACTION_SINK.get()
        if sink is not None and sink.conversation_id == conversation_id:
            return await self._send_interaction_reply(sink, text)
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

    async def _send_interaction_reply(self, sink: _InteractionSink, text: str) -> SendResult:
        """Deliver a command reply as the invoking interaction's response.

        Args:
            sink: Reply sink for the interaction being handled.
            text: Reply content, split if it exceeds Discord's per-message limit.

        Returns:
            Result indicating whether the last followup succeeded.
        """
        chunks = chunk_text(text, limit=MAX_TEXT_CHARS)
        log_debug_event(
            logger,
            "discord.outbound.interaction.started",
            chunk_count=len(chunks),
            text_chars=len(text),
        )
        message_id: str | None = None
        for chunk in chunks:
            # Marked before the send so a partial failure still counts as answered:
            # the caller must not add a fallback followup on top of a real reply.
            sink.used = True
            message_id = await sink.responder.send(chunk)
        log_debug_event(
            logger,
            "discord.outbound.interaction.completed",
            chunk_count=len(chunks),
            message_id_present=message_id is not None,
        )
        return SendResult(success=True, message_id=message_id)

    async def _process_interaction(self, inbound: _DiscordInboundInteraction) -> None:
        """Handle one application command invocation as its text equivalent.

        The invocation is turned into the message a typed command would have
        produced, so the host parses and dispatches it through exactly one code
        path and this adapter adds no second notion of what a command means.

        Args:
            inbound: Provider-neutral view of the invocation.
        """
        command = COMMANDS_BY_NAME.get(inbound.command)
        if command is None or inbound.channel_id is None:
            # Without a channel there is no conversation to act on, and guessing
            # one would let a command reset a thread the user never named.
            await inbound.responder.reject(_COMMAND_UNAVAILABLE_MESSAGE)
            return
        metadata: dict[str, object] = {
            "provider": "discord",
            "is_dm": inbound.is_dm,
            "from_self": False,
        }
        if inbound.history_chat is not None:
            metadata["history_chat"] = inbound.history_chat
        message = ChannelMessage(
            conversation_id=inbound.channel_id,
            text=f"{command.text} {inbound.argument}" if inbound.argument else command.text,
            sender_id=inbound.sender_id,
            message_id=inbound.interaction_id,
            metadata=metadata,
        )
        if not self._admits(message, parent_id=inbound.parent_id):
            log_debug_event(
                logger,
                "discord.inbound.interaction.rejected",
                exposure=self._exposure.mode.value,
            )
            # Unlike a typed command, which is simply dropped, Discord requires an
            # answer to every interaction, so a refusal is visible to its sender.
            await inbound.responder.reject(_UNAUTHORIZED_MESSAGE)
            return
        log_debug_event(logger, "discord.inbound.interaction.dispatching")
        await inbound.responder.defer()
        sink = _InteractionSink(inbound.channel_id, inbound.responder)
        token = _INTERACTION_SINK.set(sink)
        failed = False
        try:
            await dispatch_message(self._handler, message, provider="Discord")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001  # Report the failure through the interaction.
            logger.warning("Discord command %s failed", inbound.command, exc_info=True)
            failed = True
        finally:
            _INTERACTION_SINK.reset(token)
        if not sink.used:
            # A deferred interaction shows "thinking" until something follows up,
            # so answer even when the command produced no reply of its own.
            with contextlib.suppress(Exception):
                await inbound.responder.send(
                    _COMMAND_FAILED_MESSAGE if failed else _COMMAND_NO_REPLY_MESSAGE,
                )
        log_debug_event(logger, "discord.inbound.interaction.dispatched", failed=failed)

    def _admits(self, message: ChannelMessage, *, parent_id: str | None = None) -> bool:
        if _allows_discord_message(
            self._exposure, self.config.allowed_user_ids, message, parent_id=parent_id
        ):
            return True
        return self.config.pairing is not None and self.config.pairing.admits(message)

    async def _process_message(self, inbound: _DiscordInboundMessage) -> None:
        if inbound.from_self:
            # Discord's Gateway re-delivers the bot's own outbound messages through
            # on_message. Unlike Telegram/WhatsApp, the bot identity here is the
            # transport itself, not an operator account, so self-authored events
            # must never reach exposure checks -- admitting them would redispatch
            # every reply as a new prompt, looping forever.
            return
        if not inbound.is_dm and not self._is_guild_turn(inbound):
            return
        message = ChannelMessage(
            conversation_id=inbound.channel_id,
            text=inbound.text,
            sender_id=inbound.sender_id,
            message_id=inbound.message_id,
            metadata=_message_metadata(inbound),
        )
        if not self._admits(message, parent_id=inbound.parent_id):
            log_debug_event(
                logger,
                "discord.inbound.message.rejected",
                exposure=self._exposure.mode.value,
                has_media=bool(inbound.attachments),
            )
            if self.config.pairing is not None:
                await self.config.pairing.offer(message, self.send_message)
            return
        if not inbound.is_dm:
            # Only after admission, so a refused mention leaves no thread behind.
            message = await self._route_guild_message(message, inbound)
        message = await self._prepare_inbound_media(message, inbound.attachments)
        log_debug_event(
            logger,
            "discord.inbound.message.dispatching",
            has_media=bool(message.metadata.get("has_media")),
        )
        await dispatch_message(self._handler, message, provider="Discord")
        log_debug_event(logger, "discord.inbound.message.dispatched")

    def _is_guild_turn(self, inbound: _DiscordInboundMessage) -> bool:
        """Return whether a guild message addresses the bot.

        The Gateway delivers every guild message the bot can read; like a Slack
        channel, only a mention is a turn, except from a human in a channel where
        mentions are optional. Another bot must always mention this one, so two
        bots that each answer everything cannot reply to each other forever.

        Args:
            inbound: Gateway view of the guild message.

        Returns:
            `True` when the message should go on to the exposure policy.
        """
        if inbound.mentioned:
            return True
        optional = self.config.mention_optional_chats
        return not inbound.from_bot and (
            ALL_CHATS in optional
            or inbound.channel_id in optional
            or (inbound.parent_id is not None and inbound.parent_id in optional)
        )

    async def _route_guild_message(
        self, message: ChannelMessage, inbound: _DiscordInboundMessage
    ) -> ChannelMessage:
        """Move a channel mention into its own thread, or add a thread's context.

        Args:
            message: Admitted message, addressed to the channel it was posted in.
            inbound: Gateway view of the same message.

        Returns:
            The message, addressed to its thread and carrying any thread context.
        """
        if inbound.parent_id is None:
            return await self._open_thread(message, inbound)
        context = await self._thread_context(inbound)
        if not context:
            return message
        return replace(message, metadata={**message.metadata, "discord_thread_context": context})

    async def _open_thread(
        self, message: ChannelMessage, inbound: _DiscordInboundMessage
    ) -> ChannelMessage:
        try:
            thread_id = await self._gateway.create_thread(
                inbound.channel_id, inbound.message_id, _thread_name(inbound.text)
            )
        except discord.DiscordException, OSError, TimeoutError:
            # Most often a missing Create Public Threads permission. Answering in
            # the channel beats not answering at all.
            logger.warning(
                "Could not open a Discord thread; replying in the channel", exc_info=True
            )
            return message
        log_debug_event(logger, "discord.inbound.thread.opened")
        # A thread opened here is public, so it shares its parent's archive the
        # way any other public thread does.
        return replace(
            message,
            conversation_id=thread_id,
            metadata={**message.metadata, "history_chat": inbound.channel_id},
        )

    async def _thread_context(self, inbound: _DiscordInboundMessage) -> str:
        try:
            messages = await self._gateway.thread_context(inbound.channel_id, inbound.message_id)
        except discord.DiscordException, OSError, TimeoutError, ValueError:
            logger.warning("Could not read Discord thread context", exc_info=True)
            return _THREAD_CONTEXT_UNAVAILABLE
        senders = (
            None
            if self.config.include_other_thread_participants
            else self._exposure.operator_ids | self.config.allowed_user_ids
        )
        return _format_thread_context(messages, bot_id=self._gateway.bot_id, senders=senders)

    async def _process_reaction(self, inbound: _DiscordInboundReaction) -> None:
        reaction = ChannelReaction(
            conversation_id=inbound.channel_id,
            message_id=inbound.message_id,
            emoji=inbound.emoji,
            sender_id=inbound.sender_id,
            metadata={"provider": "discord"},
        )
        if not _allows_discord_reaction(
            self._exposure, self.config.allowed_user_ids, reaction
        ) and not (
            self.config.pairing is not None and self.config.pairing.admits_reaction(reaction)
        ):
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

    async def _process_connection(self, state: _DiscordConnectionState) -> None:
        if self._stopping:
            # `discord.py` dispatches `on_disconnect` as its own task while the
            # client closes, so a late event must not overwrite the final status.
            return
        previous = self._status
        self._status = ChannelStatus(
            provider="discord",
            connected=state.connected,
            detail=state.detail,
        )
        if self._status != previous:
            log_debug_event(
                logger,
                "discord.connection.changed",
                connected=state.connected,
                detail=state.detail,
            )

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
            return with_media_error(
                message,
                f"media file is too large: {attachment.size} bytes "
                f"exceeds {self.config.max_media_bytes}",
            )
        try:
            destination = await self._download_attachment(attachment, message_id=message.message_id)
        except (ChannelMediaError, OSError, urllib.error.URLError, TimeoutError) as error:
            logger.warning("Skipping Discord inbound media after download failure")
            return with_media_error(message, str(error))
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
        suffix = safe_suffix(attachment.filename, attachment.content_type)
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


def _public_thread_parent(channel: object) -> str | None:
    """Return the parent of a public Discord thread, if available."""
    if isinstance(channel, discord.Thread) and not channel.is_private():
        return _thread_parent(channel)
    return None


def _thread_parent(channel: object) -> str | None:
    """Return the parent of any Discord thread, public or private, if available."""
    if isinstance(channel, discord.Thread) and channel.parent_id is not None:
        return str(channel.parent_id)
    return None


def _bot_role_ids(message: discord.Message, bot_id: str) -> tuple[int, ...]:
    """Return ids of mentioned roles that Discord manages for this bot.

    Discord's mention autocomplete offers a bot's integration role next to the
    bot itself, and the two look alike, so mentioning either addresses the bot.
    """
    return tuple(
        role.id
        for role in message.role_mentions
        if role.tags is not None and str(role.tags.bot_id) == bot_id
    )


def _strip_bot_mentions(text: str, *, bot_id: str, role_ids: tuple[int, ...]) -> str:
    for token in (f"<@{bot_id}>", f"<@!{bot_id}>", *(f"<@&{role}>" for role in role_ids)):
        text = text.replace(token, "")
    return text.strip()


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
    text = message.content
    mentioned = False
    if bot_id is not None:
        role_ids = _bot_role_ids(message, bot_id)
        # `mentions` also holds the author of a replied-to message when the reply
        # pings them, so replying to the bot addresses it too.
        mentioned = bool(role_ids) or any(str(user.id) == bot_id for user in message.mentions)
        if mentioned:
            text = _strip_bot_mentions(text, bot_id=bot_id, role_ids=role_ids)
    return _DiscordInboundMessage(
        channel_id=str(message.channel.id),
        message_id=str(message.id),
        sender_id=sender_id,
        text=text,
        is_dm=message.guild is None,
        from_self=bot_id is not None and sender_id == bot_id,
        attachments=attachments,
        history_chat=_public_thread_parent(message.channel),
        mentioned=mentioned,
        parent_id=_thread_parent(message.channel),
        from_bot=message.author.bot,
    )


def _build_app_command(
    command: ChatCommand,
    handle_interaction: InboundInteractionCallback,
) -> app_commands.Command[Any, ..., None]:
    """Build one Discord application command for a Talon chat command.

    Args:
        command: Registry entry to expose.
        handle_interaction: Callback invoked when the command is used.

    Returns:
        Application command ready to add to a command tree.
    """

    # Left unnamed on purpose. `discord.py` takes the command name from the
    # argument below, and reassigning `__name__` here would make it disagree with
    # `__qualname__`, which `discord.py` reads as "this is a method" and then
    # rejects for having too few parameters.
    async def callback(interaction: discord.Interaction) -> None:
        await handle_interaction(_convert_interaction(interaction, command.name))

    # `discord.py` derives options from the callback's signature, so a command
    # with an argument needs a callback that declares one.
    async def callback_with_argument(
        interaction: discord.Interaction, argument: str | None = None
    ) -> None:
        await handle_interaction(_convert_interaction(interaction, command.name, argument))

    if command.argument is not None:
        return app_commands.Command(
            name=command.name,
            description=command.summary,
            callback=app_commands.describe(argument=command.argument)(callback_with_argument),
        )
    return app_commands.Command(
        name=command.name,
        description=command.summary,
        callback=callback,
    )


def _convert_interaction(
    interaction: discord.Interaction,
    command: str,
    argument: str | None = None,
) -> _DiscordInboundInteraction:
    """Convert a `discord.py` interaction into a provider-neutral value.

    Args:
        interaction: Interaction reported by the Gateway.
        command: Bare name of the invoked command.
        argument: Value of the command's optional argument, when given.

    Returns:
        Provider-neutral view of the invocation.
    """
    return _DiscordInboundInteraction(
        command=command,
        channel_id=str(interaction.channel_id) if interaction.channel_id is not None else None,
        sender_id=str(interaction.user.id),
        interaction_id=str(interaction.id),
        is_dm=interaction.guild_id is None,
        responder=_DiscordPyResponder(interaction),
        argument=argument,
        history_chat=_public_thread_parent(interaction.channel),
        parent_id=_thread_parent(interaction.channel),
    )


@dataclass(frozen=True, slots=True)
class _DiscordPyResponder:
    """Interaction reply surface backed by a `discord.py` interaction."""

    interaction: discord.Interaction

    async def reject(self, text: str) -> None:
        """Answer immediately and privately, without deferring.

        Args:
            text: Refusal to show the invoking user.
        """
        await self.interaction.response.send_message(text, ephemeral=True)

    async def defer(self) -> None:
        """Acknowledge the interaction so a reply can follow later."""
        await self.interaction.response.defer(thinking=True)

    async def send(self, text: str) -> str | None:
        """Send a followup reply.

        Args:
            text: Reply content, already within Discord's per-message limit.

        Returns:
            The followup's message id, when Discord reports one.
        """
        message = await self.interaction.followup.send(text, wait=True)
        return None if message is None else str(message.id)


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
    if inbound.history_chat is not None:
        metadata["history_chat"] = inbound.history_chat
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


def _parse_flag(value: str | None, *, default: bool) -> bool:
    """Parse a boolean environment value.

    Args:
        value: Raw environment value, or `None` when unset.
        default: Value to use when unset or empty.

    Returns:
        The parsed flag.

    Raises:
        ValueError: If the value is set but is not a recognized boolean.
    """
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in _TRUTHY_ENV_VALUES:
        return True
    if normalized in _FALSY_ENV_VALUES:
        return False
    msg = f"{SLASH_COMMANDS_ENV} must be a boolean"
    raise ValueError(msg)


def _parse_guild_id(value: str | None) -> str | None:
    """Parse an optional Discord guild id.

    Args:
        value: Raw environment value, or `None` when unset.

    Returns:
        The guild id, or `None` when unset.

    Raises:
        ValueError: If the value is set but is not a positive integer.
    """
    if value is None or not value.strip():
        return None
    guild_id = value.strip()
    if not guild_id.isdigit():
        msg = f"{COMMAND_GUILD_ID_ENV} must be a Discord guild id"
        raise ValueError(msg)
    return guild_id


def _allows_discord_message(
    exposure: ChannelExposure,
    allowed_user_ids: frozenset[str],
    message: ChannelMessage,
    *,
    parent_id: str | None = None,
) -> bool:
    if exposure.mode == ExposureMode.ALLOWLIST:
        if message.metadata.get("is_dm") is True and message.sender_id in allowed_user_ids:
            return True
        # An allowlisted channel admits every thread in it.
        if parent_id is not None and parent_id in exposure.conversations:
            return True
    return exposure.allows(message)


def _parse_mention_optional_chats(value: str) -> frozenset[str]:
    """Parse the channels where mentions are optional.

    Args:
        value: Comma-separated guild channel ids, or `*` for every guild channel.

    Returns:
        The parsed entries.

    Raises:
        ValueError: If an entry is neither a channel id nor `*`.
    """
    chats = frozenset(split_csv(value))
    if any(chat != ALL_CHATS and not chat.isdigit() for chat in chats):
        msg = f"{MENTION_OPTIONAL_CHATS_ENV} must be Discord channel ids or {ALL_CHATS}"
        raise ValueError(msg)
    return chats


def _thread_name(text: str) -> str:
    """Name a new thread after the first line of the message that opened it."""
    for line in text.splitlines():
        name = " ".join(line.split())
        if name:
            return name[:_THREAD_NAME_CHARS]
    return _DEFAULT_THREAD_NAME


def _format_thread_context(
    messages: list[_DiscordContextMessage],
    *,
    bot_id: str | None,
    senders: frozenset[str] | None,
) -> str:
    """Render earlier thread messages as escaped, bounded context lines.

    Only messages after the bot's last reply are kept. Everything before that
    reply is already in the agent's history, either as an earlier turn or as the
    context that turn was given, and in a thread where mentions are optional
    every trusted message is a turn.

    Args:
        messages: Earlier thread messages, oldest first.
        bot_id: This bot's user id.
        senders: Senders whose messages may be included, or `None` for anyone.

    Returns:
        One `sender: text` line per included message, trimmed from the oldest
            end to fit the context budget.
    """
    replies = [index for index, message in enumerate(messages) if message.sender_id == bot_id]
    unseen = messages[replies[-1] + 1 :] if replies else messages
    lines = [
        f"{message.sender_id}: {_context_text(message)}"
        for message in unseen
        if (senders is None or message.sender_id in senders)
        and not _contains_oauth_callback(message.text)
    ]
    while lines and sum(len(line) + 1 for line in lines) > _THREAD_CONTEXT_CHARS:
        lines.pop(0)
    return "\n".join(lines)


def _context_text(message: _DiscordContextMessage) -> str:
    text = escape(message.text[:_THREAD_CONTEXT_MESSAGE_CHARS], quote=False)
    if not message.from_bot:
        return text
    return (
        f'<input-message sender="discord:{escape(message.sender_id)}" surface="discord" '
        f'kind="message" sender_type="bot">\n{text}\n</input-message>'
    )


def _contains_oauth_callback(text: str) -> bool:
    """Exclude a whole message before truncation could hide a pasted credential."""
    return any(
        extract_loopback_oauth_callback_url(candidate) is not None
        for candidate in _OAUTH_CANDIDATE_PATTERN.findall(text)
    )


def _allows_discord_reaction(
    exposure: ChannelExposure,
    allowed_user_ids: frozenset[str],
    reaction: ChannelReaction,
) -> bool:
    if reaction.sender_id is None:
        return False
    return reaction.sender_id in exposure.operator_ids or reaction.sender_id in allowed_user_ids


def _inbound_media_filename(*, message_id: str | None, attachment_url: str, suffix: str) -> str:
    message = safe_filename_part(message_id or "message")
    token = safe_filename_part(attachment_url)[-24:] or "file"
    return f"{message}_{token}{suffix}"


def _download_attachment_file(url: str, destination: Path, timeout: float, max_bytes: int) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    request = urllib.request.Request(url)  # noqa: S310  # Discord attachment CDN URL
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        length = response.headers.get("content-length")
        if length is not None:
            expected = parse_content_length(length)
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
