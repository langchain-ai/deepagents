"""Slack channel adapter backed by `slack_sdk` Socket Mode.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import mimetypes
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, NoReturn, Protocol

import aiohttp
from slack_sdk.errors import SlackApiError
from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.webhook.async_client import AsyncWebhookClient

from deepagents_talon.channels.base import (
    ChannelExposure,
    ChannelExposureEnv,
    ChannelMediaError,
    ExposureMode,
    channel_exposure_from_env,
    chunk_text,
    dispatch_message,
    format_markdown_for_channel,
    max_media_bytes_from_env,
    message_with_media_paths,
    optional_str,
    outbound_media_root_from_env,
    parse_content_length,
    parse_float,
    safe_filename_part,
    safe_suffix,
    split_csv,
    validate_media,
    with_media_error,
)
from deepagents_talon.commands import COMMANDS_BY_NAME
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
    from slack_sdk.socket_mode.async_client import AsyncBaseSocketModeClient
    from slack_sdk.socket_mode.request import SocketModeRequest

    from deepagents_talon.config import TalonConfig

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 4000
"""Slack truncates `chat.postMessage` text well past this, but renders 4000 reliably."""

DEFAULT_MAX_MEDIA_BYTES = 1024 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT_SECONDS = 35.0
OPEN_EXPOSURE_ACK_ENV = "DEEPAGENTS_TALON_SLACK_OPEN_ACK"
_ENV_PREFIX = "DEEPAGENTS_TALON_SLACK"

_FILE_HOST = "files.slack.com"
"""The only host the bot token is ever sent to when downloading inbound files."""

_SEEN_EVENT_LIMIT = 1024
_SENT_MESSAGE_LIMIT = 1024
_THREAD_CONTEXT_PAGES = 20
_THREAD_CONTEXT_MESSAGES = 40
_THREAD_CONTEXT_CHARS = 12000

_COMMAND_UNAVAILABLE_MESSAGE = "That command is not available here."
_COMMAND_DM_ONLY_MESSAGE = (
    "Slash commands work only in a direct message with the bot. In a channel thread, "
    "mention the bot "
    "followed by the command instead, for example `@Talon /new`."
)
_UNAUTHORIZED_MESSAGE = "This assistant does not accept commands from you."
_COMMAND_NO_REPLY_MESSAGE = "Done."
_COMMAND_FAILED_MESSAGE = "Something went wrong running that command. Check Talon logs."

_CONVERSATION_PATTERN = re.compile(r"(?P<channel>[CDG][A-Z0-9]+)(?::(?P<thread>\d+\.\d+))?")
_CODE_SPAN_PATTERN = re.compile(r"```.*?```|`[^`\n]+`", flags=re.DOTALL)
_ESCAPED_MENTION_PATTERN = re.compile(r"&lt;@([UW][A-Z0-9]+)&gt;")
_WEB_LINK_PATTERN = re.compile(r"\[([^\]\n]+)]\((https?://[^)\s|]+)\)")
_NON_WEB_LINK_PATTERN = re.compile(r"(\[[^\]\n]+]\()([^)]+)(\))")
_SLACK_LINK_PATTERN = re.compile(r"<https?://[^>]+>")
_SKIN_TONE_PATTERN = re.compile(r"::skin-tone-\d$")
_INBOUND_LINK_PATTERN = re.compile(r"<((?:https?|mailto):[^<>|]+)(?:\|([^<>]*))?>")

_REACTION_EMOJI = {
    "+1": "\U0001f44d",
    "thumbsup": "\U0001f44d",
    "thumbsup_all": "\U0001f44d",
    "-1": "\U0001f44e",
    "thumbsdown": "\U0001f44e",
}
"""Slack reaction names Talon understands, mapped to the Unicode the host compares."""


@dataclass(frozen=True, slots=True)
class SlackChannelConfig:
    """Configuration for the Slack channel adapter.

    Args:
        bot_token: Bot user OAuth token (`xoxb-`) used for Web API calls.
        app_token: App-level token (`xapp-`) with `connections:write`, used to
            open the Socket Mode connection.
        inbound_media_dir: Directory where downloaded inbound files are stored.
        outbound_media_dir: Optional root that outbound media must remain under
            before it is uploaded.
        exposure: Inbound trigger policy.
        allowed_user_ids: Slack user ids always allowed to DM the bot, regardless
            of exposure mode.
        mention_allowlist_user_ids: Optional outbound mention restriction; `None`
            allows all valid user mentions, while an empty set allows none.
        max_media_bytes: Maximum media bytes allowed for inbound downloads and
            outbound local files.
        request_timeout_seconds: Timeout for connecting and for file downloads.
        pairing: Optional sender pairing policy that admits approved DM senders
            and issues codes to unknown ones.
    """

    bot_token: str = field(repr=False)
    app_token: str = field(repr=False)
    inbound_media_dir: Path | None = None
    outbound_media_dir: Path | None = None
    exposure: ChannelExposure = field(default_factory=ChannelExposure)
    allowed_user_ids: frozenset[str] = field(default_factory=frozenset)
    mention_allowlist_user_ids: frozenset[str] | None = None
    max_media_bytes: int = DEFAULT_MAX_MEDIA_BYTES
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    pairing: SenderPairing | None = None

    @classmethod
    def from_talon_config(cls, config: TalonConfig) -> SlackChannelConfig:
        """Build Slack channel configuration from Talon environment values.

        Args:
            config: Talon process configuration.

        Returns:
            Slack channel configuration.

        Raises:
            ValueError: If a token is missing or exposure configuration is invalid.
        """
        env = config.env
        bot_token = env.get("DEEPAGENTS_TALON_SLACK_BOT_TOKEN")
        if not bot_token:
            msg = "Slack bot token is required (DEEPAGENTS_TALON_SLACK_BOT_TOKEN)"
            raise ValueError(msg)
        app_token = env.get("DEEPAGENTS_TALON_SLACK_APP_TOKEN")
        if not app_token:
            msg = "Slack app-level token is required (DEEPAGENTS_TALON_SLACK_APP_TOKEN)"
            raise ValueError(msg)
        exposure = channel_exposure_from_env(
            env,
            ChannelExposureEnv(
                provider="Slack",
                env_prefix=_ENV_PREFIX,
                open_ack=OPEN_EXPOSURE_ACK_ENV,
                require_self_operator=True,
            ),
        )
        return cls(
            bot_token=bot_token,
            app_token=app_token,
            inbound_media_dir=Path(
                env.get(
                    "DEEPAGENTS_TALON_SLACK_MEDIA_DIR",
                    str(config.inbound_media_dir / "slack"),
                ),
            ),
            outbound_media_dir=outbound_media_root_from_env(env),
            exposure=exposure,
            allowed_user_ids=frozenset(
                split_csv(env.get("DEEPAGENTS_TALON_SLACK_ALLOWLIST_USERS", "")),
            ),
            mention_allowlist_user_ids=(
                frozenset(split_csv(env["DEEPAGENTS_TALON_SLACK_MENTION_ALLOWLIST_USERS"]))
                if "DEEPAGENTS_TALON_SLACK_MENTION_ALLOWLIST_USERS" in env
                else None
            ),
            max_media_bytes=max_media_bytes_from_env(env),
            request_timeout_seconds=parse_float(
                env.get("DEEPAGENTS_TALON_SLACK_REQUEST_TIMEOUT_SECONDS"),
                DEFAULT_REQUEST_TIMEOUT_SECONDS,
            ),
            pairing=pairing_from_env(
                env,
                provider="slack",
                env_prefix=_ENV_PREFIX,
                open_exposure=exposure.mode == ExposureMode.OPEN,
                home=config.home,
            ),
        )


@dataclass(frozen=True, slots=True)
class _SlackFile:
    """Metadata for one inbound Slack file."""

    file_id: str
    url: str
    filename: str
    mimetype: str | None
    size: int


@dataclass(frozen=True, slots=True)
class _SlackInboundMessage:
    """Provider-neutral view of a Slack `message` or `app_mention` event.

    Args:
        channel_id: Channel the message was posted in.
        ts: Slack timestamp of the message, which is its id.
        thread_ts: Timestamp of the thread root, when the message is a reply.
        sender_id: Slack user id of the author.
        text: Message text with the bot's own mention removed.
        is_dm: Whether the message arrived in a direct message with the bot.
        files: Files attached to the message.
    """

    channel_id: str
    ts: str
    thread_ts: str | None
    sender_id: str
    text: str
    is_dm: bool
    files: tuple[_SlackFile, ...] = ()

    @property
    def conversation_id(self) -> str:
        """Talon conversation: the DM itself, or the channel thread it belongs to."""
        if self.is_dm:
            return self.channel_id
        return f"{self.channel_id}:{self.thread_ts or self.ts}"


@dataclass(frozen=True, slots=True)
class _SlackInboundReaction:
    """Provider-neutral view of a Slack `reaction_added` event."""

    channel_id: str
    message_ts: str
    sender_id: str
    reaction: str


@dataclass(frozen=True, slots=True)
class _SlackConnectionState:
    """Provider-neutral view of a Socket Mode connection lifecycle event."""

    connected: bool
    detail: str


class _CommandResponder(Protocol):
    """Reply surface for one `/talon` invocation, backed by its `response_url`."""

    async def reject(self, text: str) -> None:
        """Answer privately to the invoking user."""

    async def send(self, text: str) -> str | None:
        """Send a visible reply and return its message id when one is reported."""


@dataclass(frozen=True, slots=True)
class _SlackInboundCommand:
    """Provider-neutral view of one `/talon` slash command invocation.

    Args:
        command: Requested Talon command name, without a leading slash.
        channel_id: Channel the command was invoked in.
        sender_id: Slack user id that invoked the command.
        trigger_id: Slack's id for this invocation.
        responder: Reply surface bound to this invocation.
        argument: Text after the command name, when any was given.
    """

    command: str
    channel_id: str
    sender_id: str
    trigger_id: str
    responder: _CommandResponder
    argument: str | None = None

    @property
    def is_dm(self) -> bool:
        """Whether the command was invoked in a direct message with the bot."""
        return self.channel_id.startswith("D")


@dataclass(slots=True)
class _CommandSink:
    """Routes one command's reply back to the invocation that asked for it.

    Args:
        conversation_id: Conversation whose replies belong to this invocation.
        responder: Reply surface for the invocation.
        used: Whether a reply has been routed, so the caller knows if it still
            owes the user an answer.
    """

    conversation_id: str
    responder: _CommandResponder
    used: bool = False


_COMMAND_SINK: ContextVar[_CommandSink | None] = ContextVar(
    "talon_slack_command_sink",
    default=None,
)
_TOOL_APPROVAL_PROMPT: ContextVar[bool] = ContextVar(
    "talon_slack_tool_approval_prompt", default=False
)
"""Reply sink for the slash command being handled on this task, if any.

`slack_sdk` runs each Socket Mode envelope in its own task, so each invocation
sees only its own sink, and unrelated sends to the same DM are never captured.
"""

InboundMessageCallback = Callable[[_SlackInboundMessage], Awaitable[None]]
InboundReactionCallback = Callable[[_SlackInboundReaction], Awaitable[None]]
InboundConnectionCallback = Callable[[_SlackConnectionState], Awaitable[None]]
InboundCommandCallback = Callable[[_SlackInboundCommand], Awaitable[None]]

_CONNECTED_STATE = _SlackConnectionState(connected=True, detail="connected")
_RECONNECTING_STATE = _SlackConnectionState(connected=True, detail="reconnecting")


class _SlackGateway(Protocol):
    """Narrow surface `SlackChannel` needs from a Slack client implementation.

    Production code implements this with `slack_sdk`; tests inject a fake so unit
    tests never open a real Socket Mode connection.
    """

    @property
    def bot_id(self) -> str | None:
        """Bot user id once authenticated."""

    async def start(
        self,
        *,
        handle_message: InboundMessageCallback,
        handle_reaction: InboundReactionCallback,
        handle_connection: InboundConnectionCallback,
        handle_command: InboundCommandCallback,
    ) -> None:
        """Connect and begin dispatching inbound and connection events."""

    async def stop(self) -> None:
        """Disconnect and release resources."""

    async def post_message(self, channel_id: str, text: str, *, thread_ts: str | None) -> str:
        """Post a message and return its timestamp."""

    async def thread_context(
        self, channel_id: str, thread_ts: str, before_ts: str
    ) -> list[tuple[str, str]]:
        """Read bounded preceding replies from a channel thread."""

    async def open_dm(self, user_id: str) -> str:
        """Open a DM with a user and return its channel id."""

    async def upload_file(
        self,
        channel_id: str,
        file_path: Path,
        *,
        thread_ts: str | None,
        comment: str | None,
    ) -> str | None:
        """Upload a file with an optional comment."""

    async def update_message(self, channel_id: str, ts: str, text: str) -> None:
        """Replace a previously posted message's text."""


class _SlackSdkGateway:
    """Gateway implementation backed by `slack_sdk` Socket Mode."""

    def __init__(self, *, bot_token: str, app_token: str, timeout_seconds: float) -> None:
        self._bot_token = bot_token
        self._app_token = app_token
        self._timeout_seconds = timeout_seconds
        self._web: AsyncWebClient | None = None
        self._socket: SocketModeClient | None = None
        self._bot_id: str | None = None
        self._seen_events: OrderedDict[str, None] = OrderedDict()

    @property
    def bot_id(self) -> str | None:
        return self._bot_id

    async def start(
        self,
        *,
        handle_message: InboundMessageCallback,
        handle_reaction: InboundReactionCallback,
        handle_connection: InboundConnectionCallback,
        handle_command: InboundCommandCallback,
    ) -> None:
        web = AsyncWebClient(token=self._bot_token, timeout=int(self._timeout_seconds))
        auth = await asyncio.wait_for(web.auth_test(), self._timeout_seconds)
        self._bot_id = optional_str(auth.get("user_id"))
        self._web = web
        # Built here, not in `__init__`: the client opens an aiohttp session and
        # schedules its message processor, both of which need a running loop.
        socket = SocketModeClient(app_token=self._app_token, web_client=web)
        self._socket = socket

        async def on_raw(
            _client: AsyncBaseSocketModeClient, message: dict, _raw: str | None
        ) -> None:
            if message.get("type") == "hello":
                await handle_connection(_CONNECTED_STATE)

        async def on_close(_message: object) -> None:
            # `slack_sdk` reconnects on its own, so this is a detail-only change.
            await handle_connection(_RECONNECTING_STATE)

        async def on_request(client: AsyncBaseSocketModeClient, request: SocketModeRequest) -> None:
            # Ack first: Slack redelivers any envelope not acked within 3 seconds,
            # and an agent turn takes far longer than that.
            await client.send_socket_mode_response(
                SocketModeResponse(envelope_id=request.envelope_id),
            )
            if request.type == "events_api":
                await self._dispatch_event(request.payload, handle_message, handle_reaction)
            elif request.type == "slash_commands":
                command = _convert_command(request.payload)
                if command is not None:
                    await handle_command(command)

        socket.message_listeners.append(on_raw)
        socket.on_close_listeners.append(on_close)
        socket.socket_mode_request_listeners.append(on_request)
        try:
            await asyncio.wait_for(socket.connect(), self._timeout_seconds)
        except TimeoutError:
            await self.stop()
            msg = "Timed out connecting to Slack Socket Mode"
            raise TimeoutError(msg) from None

    async def _dispatch_event(
        self,
        payload: dict,
        handle_message: InboundMessageCallback,
        handle_reaction: InboundReactionCallback,
    ) -> None:
        if self._already_seen(payload.get("event_id")):
            return
        event = payload.get("event")
        if not isinstance(event, dict):
            return
        if event.get("type") == "reaction_added":
            reaction = _convert_reaction(event, bot_id=self._bot_id)
            if reaction is not None:
                await handle_reaction(reaction)
            return
        message = _convert_event(event, bot_id=self._bot_id)
        if message is not None:
            await handle_message(message)

    def _already_seen(self, event_id: object) -> bool:
        """Record `event_id`, reporting whether Slack already delivered it.

        Slack redelivers an event whose ack it did not receive, for example across
        a reconnect, and a duplicate would run the same prompt twice.
        """
        if not isinstance(event_id, str):
            return False
        if event_id in self._seen_events:
            return True
        self._seen_events[event_id] = None
        if len(self._seen_events) > _SEEN_EVENT_LIMIT:
            self._seen_events.popitem(last=False)
        return False

    async def stop(self) -> None:
        if self._socket is not None:
            await self._socket.close()
        self._socket = None
        self._web = None

    async def post_message(self, channel_id: str, text: str, *, thread_ts: str | None) -> str:
        response = await self._client().chat_postMessage(
            channel=channel_id,
            text=text,
            thread_ts=thread_ts,
        )
        return str(response["ts"])

    async def thread_context(
        self, channel_id: str, thread_ts: str, before_ts: str
    ) -> list[tuple[str, str]]:
        messages: list[tuple[str, str]] = []
        cursor: str | None = None
        more = False
        for _ in range(_THREAD_CONTEXT_PAGES):
            response = await self._client().conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                latest=before_ts,
                limit=100,
                cursor=cursor,
            )
            for item in response.get("messages", []):
                if not isinstance(item, dict) or item.get("ts") == before_ts:
                    continue
                sender = optional_str(item.get("user"))
                raw_text = str(item.get("text") or "")
                if _contains_oauth_callback(raw_text):
                    continue
                text = _decode_mrkdwn(raw_text)
                if sender and text and not item.get("bot_id"):
                    messages.append((sender, text[:1000]))
            messages = messages[-_THREAD_CONTEXT_MESSAGES:]
            cursor = optional_str(response.get("response_metadata", {}).get("next_cursor"))
            more = bool(response.get("has_more") or cursor)
            if not more or not cursor:
                break
        if more:
            msg = "Slack thread history exceeds the retrieval limit"
            raise ValueError(msg)
        while (
            messages
            and sum(len(sender) + len(text) + 3 for sender, text in messages)
            > _THREAD_CONTEXT_CHARS
        ):
            messages.pop(0)
        return messages

    async def open_dm(self, user_id: str) -> str:
        response = await self._client().conversations_open(users=user_id)
        channel = response.get("channel")
        channel_id = optional_str(channel.get("id")) if isinstance(channel, dict) else None
        if (
            channel_id is None
            or not channel_id.startswith("D")
            or not _CONVERSATION_PATTERN.fullmatch(channel_id)
        ):
            msg = "Slack did not return a DM channel"
            raise ValueError(msg)
        return channel_id

    async def upload_file(
        self,
        channel_id: str,
        file_path: Path,
        *,
        thread_ts: str | None,
        comment: str | None,
    ) -> str | None:
        # Not `files_upload_v2`: it reads the whole file into memory, on the event
        # loop, before its first await. The same three steps run here instead, with
        # the body streamed from disk.
        client = self._client()
        stat = await asyncio.to_thread(file_path.stat)
        ticket = await client.files_getUploadURLExternal(
            filename=file_path.name,
            length=stat.st_size,
        )
        await self._stream_upload(str(ticket["upload_url"]), file_path)
        await client.files_completeUploadExternal(
            files=[{"id": str(ticket["file_id"]), "title": file_path.name}],
            channel_id=channel_id,
            initial_comment=comment,
            thread_ts=thread_ts,
        )
        # The completed upload does not report the timestamp of the message that
        # will carry the file.
        return None

    async def _stream_upload(self, upload_url: str, file_path: Path) -> None:
        if not _is_file_host_url(upload_url):
            msg = "refusing to upload a Slack file to an unexpected host"
            raise ChannelMediaError(msg)
        # Only the idle timeouts are bounded: a large file may take longer in total.
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=self._timeout_seconds,
            sock_read=self._timeout_seconds,
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            body = await asyncio.to_thread(file_path.open, "rb")
            with body:
                # aiohttp reads a file object in an executor, chunk by chunk.
                async with session.post(upload_url, data=body) as response:
                    if response.status != 200:  # noqa: PLR2004  # Slack's documented success
                        msg = f"Slack file upload failed with HTTP {response.status}"
                        raise ChannelMediaError(msg)

    async def update_message(self, channel_id: str, ts: str, text: str) -> None:
        await self._client().chat_update(channel=channel_id, ts=ts, text=text)

    def _client(self) -> AsyncWebClient:
        if self._web is None:
            msg = "Slack gateway is not started"
            raise RuntimeError(msg)
        return self._web


class SlackChannel:
    """Channel adapter for Slack via Socket Mode.

    A direct message with the bot is one conversation. In channels the bot only
    answers when mentioned, and replies in a thread under the mentioning message;
    each thread is its own conversation, so follow-ups mention the bot in-thread.
    """

    def __init__(self, config: SlackChannelConfig, *, gateway: _SlackGateway | None = None) -> None:
        """Initialize the channel.

        Args:
            config: Slack channel configuration.
            gateway: Optional injectable gateway, used to avoid real connections in
                tests. Defaults to a `slack_sdk`-backed gateway.
        """
        self.config = config
        self._gateway = gateway or _SlackSdkGateway(
            bot_token=config.bot_token,
            app_token=config.app_token,
            timeout_seconds=config.request_timeout_seconds,
        )
        self._handler: MessageHandler | None = None
        self._reaction_handler: ReactionHandler | None = None
        self._exposure = config.exposure
        self._status = ChannelStatus(provider="slack", connected=False, detail="disconnected")
        self._stopping = False
        # Slack reports a reaction by channel and message timestamp only, so the
        # thread a bot message belongs to is remembered here to route approvals.
        self._sent_threads: OrderedDict[tuple[str, str], str] = OrderedDict()

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
        """Open the Socket Mode connection and begin receiving events."""
        log_debug_event(
            logger,
            "slack.channel.starting",
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
            handle_command=self._process_command,
        )
        self._status = ChannelStatus(provider="slack", connected=True, detail="connected")
        log_debug_event(logger, "slack.channel.started", connected=True)

    async def stop(self) -> None:
        """Close the Socket Mode connection and release resources."""
        log_debug_event(logger, "slack.channel.stopping")
        self._stopping = True
        await self._gateway.stop()
        self._status = ChannelStatus(provider="slack", connected=False, detail="disconnected")
        log_debug_event(logger, "slack.channel.stopped")

    async def send_message(self, conversation_id: str, text: str) -> SendResult:
        """Send a message, splitting long text across multiple posts.

        Args:
            conversation_id: DM channel id, or `channel:thread_ts` for a thread.
            text: Markdown message content to send.

        Returns:
            Result carrying the timestamp of the last posted chunk.
        """
        sink = _COMMAND_SINK.get()
        if sink is not None and sink.conversation_id == conversation_id:
            return await self._send_command_reply(sink, text)
        channel_id, thread_ts = _parse_conversation_id(conversation_id)
        chunks = chunk_text(
            format_markdown_for_slack(
                text,
                mention_allowlist=(
                    frozenset()
                    if _TOOL_APPROVAL_PROMPT.get()
                    else self.config.mention_allowlist_user_ids
                ),
            ),
            limit=MAX_TEXT_CHARS,
        )
        log_debug_event(
            logger,
            "slack.outbound.text.started",
            chunk_count=len(chunks),
            text_chars=len(text),
        )
        message_id: str | None = None
        for chunk in chunks:
            message_id = await self._gateway.post_message(channel_id, chunk, thread_ts=thread_ts)
            self._remember_sent(channel_id, message_id, conversation_id)
        log_debug_event(logger, "slack.outbound.text.completed", chunk_count=len(chunks))
        return SendResult(success=True, message_id=message_id)

    async def send_tool_approval_prompt(self, conversation_id: str, text: str) -> SendResult:
        """Send a tool approval preview without activating user mentions."""
        token = _TOOL_APPROVAL_PROMPT.set(True)
        try:
            return await self.send_message(conversation_id, text)
        finally:
            _TOOL_APPROVAL_PROMPT.reset(token)

    async def send_media(self, conversation_id: str, media: ChannelMedia) -> SendResult:
        """Upload media as a file with an optional caption.

        Args:
            conversation_id: DM channel id, or `channel:thread_ts` for a thread.
            media: Media payload to deliver.

        Returns:
            Result indicating whether the upload succeeded.
        """
        channel_id, thread_ts = _parse_conversation_id(conversation_id)
        checked = validate_media(
            media,
            root=self.config.outbound_media_dir,
            max_bytes=self.config.max_media_bytes,
        )
        comment = await self._media_comment(conversation_id, checked.caption)
        log_debug_event(
            logger,
            "slack.outbound.media.started",
            caption_present=comment is not None,
            media_type=checked.media_type,
        )
        message_id = await self._gateway.upload_file(
            channel_id,
            checked.path,
            thread_ts=thread_ts,
            comment=comment,
        )
        log_debug_event(logger, "slack.outbound.media.completed", media_type=checked.media_type)
        return SendResult(success=True, message_id=message_id)

    def top_level_conversation_id(self, conversation_id: str) -> str:
        """Return the conversation that posts to a thread's channel, not the thread.

        Args:
            conversation_id: DM channel id, or `channel:thread_ts` for a thread.

        Returns:
            The channel id on its own, which Slack posts to the channel's top level.
        """
        channel_id, _ = _parse_conversation_id(conversation_id)
        return channel_id

    async def edit_message(self, conversation_id: str, message_id: str, text: str) -> SendResult:
        """Edit a previously posted message.

        Args:
            conversation_id: DM channel id, or `channel:thread_ts` for a thread.
            message_id: Slack timestamp of the message to edit.
            text: Replacement Markdown content.

        Returns:
            Result indicating whether the edit succeeded.
        """
        channel_id, _ = _parse_conversation_id(conversation_id)
        formatted = format_markdown_for_slack(
            text, mention_allowlist=self.config.mention_allowlist_user_ids
        )
        await self._gateway.update_message(channel_id, message_id, formatted[:MAX_TEXT_CHARS])
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, conversation_id: str) -> None:
        """Do nothing: Slack has no typing indicator for bots outside assistant threads.

        Args:
            conversation_id: Unused.
        """
        del conversation_id

    async def status(self) -> ChannelStatus:
        """Report the channel connection status."""
        return self._status

    def _remember_sent(self, channel_id: str, ts: str, conversation_id: str) -> None:
        self._sent_threads[channel_id, ts] = conversation_id
        if len(self._sent_threads) > _SENT_MESSAGE_LIMIT:
            self._sent_threads.popitem(last=False)

    async def _media_comment(self, conversation_id: str, caption: str | None) -> str | None:
        if not caption:
            return None
        comment = format_markdown_for_slack(
            caption, mention_allowlist=self.config.mention_allowlist_user_ids
        )
        if len(comment) <= MAX_TEXT_CHARS:
            return comment
        await self.send_message(conversation_id, caption)
        return None

    async def _send_command_reply(self, sink: _CommandSink, text: str) -> SendResult:
        chunks = chunk_text(
            format_markdown_for_slack(
                text,
                mention_allowlist=(
                    frozenset()
                    if _TOOL_APPROVAL_PROMPT.get()
                    else self.config.mention_allowlist_user_ids
                ),
            ),
            limit=MAX_TEXT_CHARS,
        )
        message_id: str | None = None
        for chunk in chunks:
            # Marked before the send so a partial failure still counts as answered:
            # the caller must not add a fallback reply on top of a real one.
            sink.used = True
            message_id = await sink.responder.send(chunk)
        log_debug_event(logger, "slack.outbound.command.completed", chunk_count=len(chunks))
        return SendResult(success=True, message_id=message_id)

    async def _process_command(self, inbound: _SlackInboundCommand) -> None:
        """Handle one `/talon` invocation as its typed-command equivalent.

        Args:
            inbound: Provider-neutral view of the invocation.
        """
        command = COMMANDS_BY_NAME.get(inbound.command)
        text = ""
        if command is not None:
            text = f"{command.text} {inbound.argument}" if inbound.argument else command.text
        message = ChannelMessage(
            conversation_id=inbound.channel_id,
            text=text,
            sender_id=inbound.sender_id,
            message_id=inbound.trigger_id,
            metadata={"provider": "slack", "is_dm": inbound.is_dm, "from_self": False},
        )
        rejection = self._command_rejection(inbound, message, known=command is not None)
        if rejection is not None:
            await inbound.responder.reject(rejection)
            return
        await self._dispatch_command(inbound, message)

    def _command_rejection(
        self,
        inbound: _SlackInboundCommand,
        message: ChannelMessage,
        *,
        known: bool,
    ) -> str | None:
        # Authorization first, so an unauthorized user learns nothing about which
        # commands exist or where they work.
        if not self._admits(message):
            log_debug_event(logger, "slack.inbound.command.rejected", reason="exposure")
            return _UNAUTHORIZED_MESSAGE
        if not known:
            return _COMMAND_UNAVAILABLE_MESSAGE
        if not inbound.is_dm:
            # A slash command carries no thread, so in a channel it would act on a
            # conversation that never holds an agent thread.
            return _COMMAND_DM_ONLY_MESSAGE
        return None

    async def _dispatch_command(
        self, inbound: _SlackInboundCommand, message: ChannelMessage
    ) -> None:
        log_debug_event(logger, "slack.inbound.command.dispatching")
        sink = _CommandSink(message.conversation_id, inbound.responder)
        token = _COMMAND_SINK.set(sink)
        failed = False
        try:
            await dispatch_message(self._handler, message, provider="Slack")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001  # Report the failure through the command reply.
            logger.warning("Slack command %s failed", inbound.command, exc_info=True)
            failed = True
        finally:
            _COMMAND_SINK.reset(token)
        if not sink.used:
            with contextlib.suppress(Exception):
                await inbound.responder.send(
                    _COMMAND_FAILED_MESSAGE if failed else _COMMAND_NO_REPLY_MESSAGE,
                )
        log_debug_event(logger, "slack.inbound.command.dispatched", failed=failed)

    def _admits(self, message: ChannelMessage) -> bool:
        if _allows_slack_message(self._exposure, self.config.allowed_user_ids, message):
            return True
        return self.config.pairing is not None and self.config.pairing.admits(message)

    async def _process_message(self, inbound: _SlackInboundMessage) -> None:
        message = ChannelMessage(
            conversation_id=inbound.conversation_id,
            text=inbound.text,
            sender_id=inbound.sender_id,
            message_id=inbound.ts,
            metadata=_message_metadata(inbound),
        )
        if not self._admits(message):
            log_debug_event(
                logger,
                "slack.inbound.message.rejected",
                exposure=self._exposure.mode.value,
                has_media=bool(inbound.files),
            )
            if self.config.pairing is not None:
                await self._offer_pairing(message, inbound)
            return
        if not inbound.is_dm and inbound.thread_ts:
            try:
                replies = await self._gateway.thread_context(
                    inbound.channel_id, inbound.thread_ts, inbound.ts
                )
                senders = self._exposure.operator_ids | self.config.allowed_user_ids
                context = "\n".join(
                    f"{sender}: {text}"
                    for sender, text in replies
                    if sender in senders and not _contains_oauth_callback(text)
                )
            except (SlackApiError, OSError, TimeoutError, ValueError):
                logger.warning("Could not read Slack thread context", exc_info=True)
                context = "[Slack thread history unavailable; ask for context before acting.]"
            if context:
                message = ChannelMessage(
                    conversation_id=message.conversation_id,
                    text=message.text,
                    sender_id=message.sender_id,
                    message_id=message.message_id,
                    metadata={**message.metadata, "slack_thread_context": context},
                )
        message = await self._prepare_inbound_media(message, inbound.files)
        log_debug_event(
            logger,
            "slack.inbound.message.dispatching",
            has_media=bool(message.metadata.get("has_media")),
        )
        await dispatch_message(self._handler, message, provider="Slack")
        log_debug_event(logger, "slack.inbound.message.dispatched")

    async def _offer_pairing(self, message: ChannelMessage, inbound: _SlackInboundMessage) -> None:
        pairing = self.config.pairing
        if pairing is None:
            return
        if not inbound.is_dm:
            if inbound.sender_id in pairing.env_sender_ids or pairing.store.is_paired(
                pairing.provider, inbound.sender_id
            ):
                return
            try:
                dm_id = await self._gateway.open_dm(inbound.sender_id)
            except (SlackApiError, OSError, TimeoutError, ValueError):
                logger.warning("Could not open Slack DM for sender pairing", exc_info=True)
                return
            message = ChannelMessage(
                conversation_id=dm_id,
                text=inbound.text,
                sender_id=inbound.sender_id,
                metadata={"provider": "slack", "is_dm": True},
            )
        await pairing.offer(message, self.send_message)

    async def _process_reaction(self, inbound: _SlackInboundReaction) -> None:
        conversation_id = self._sent_threads.get(
            (inbound.channel_id, inbound.message_ts),
        ) or _reaction_conversation_id(inbound)
        reaction = ChannelReaction(
            conversation_id=conversation_id,
            message_id=inbound.message_ts,
            emoji=_reaction_emoji(inbound.reaction),
            sender_id=inbound.sender_id,
            metadata={"provider": "slack"},
        )
        sender = inbound.sender_id
        if (
            sender not in self._exposure.operator_ids
            and sender not in self.config.allowed_user_ids
            and not (
                self.config.pairing is not None and self.config.pairing.admits_reaction(reaction)
            )
        ):
            log_debug_event(logger, "slack.inbound.reaction.rejected")
            return
        if self._reaction_handler is None:
            logger.warning("Dropping Slack reaction because no handler is registered")
            return
        log_debug_event(logger, "slack.inbound.reaction.dispatching")
        await self._reaction_handler(reaction)

    async def _process_connection(self, state: _SlackConnectionState) -> None:
        if self._stopping:
            # A close event raced with `stop`; it must not overwrite the final status.
            return
        previous = self._status
        self._status = ChannelStatus(
            provider="slack",
            connected=state.connected,
            detail=state.detail,
        )
        if self._status != previous:
            log_debug_event(
                logger,
                "slack.connection.changed",
                connected=state.connected,
                detail=state.detail,
            )

    async def _prepare_inbound_media(
        self,
        message: ChannelMessage,
        files: tuple[_SlackFile, ...],
    ) -> ChannelMessage:
        if not files or self.config.inbound_media_dir is None:
            return message
        file = files[0]
        if file.size > self.config.max_media_bytes:
            logger.warning("Skipping Slack inbound media because it exceeds the size cap")
            return with_media_error(
                message,
                f"media file is too large: {file.size} bytes exceeds {self.config.max_media_bytes}",
            )
        destination = self.config.inbound_media_dir / _inbound_media_filename(
            message_id=message.message_id,
            file=file,
        )
        try:
            await asyncio.to_thread(
                download_slack_file,
                file.url,
                destination,
                token=self.config.bot_token,
                timeout=self.config.request_timeout_seconds,
                max_bytes=self.config.max_media_bytes,
            )
        except (ChannelMediaError, OSError, urllib.error.URLError, TimeoutError) as error:
            logger.warning("Skipping Slack inbound media after download failure")
            return with_media_error(message, str(error))
        mime_type = file.mimetype or mimetypes.guess_type(destination.name)[0]
        return message_with_media_paths(
            message,
            media_paths=[str(destination)],
            mime_types=[mime_type] if mime_type else [],
        )


def format_markdown_for_slack(text: str, *, mention_allowlist: frozenset[str] | None = None) -> str:
    """Convert Markdown to Slack `mrkdwn`, allowing configured user mentions.

    Args:
        text: Markdown text returned by the agent.
        mention_allowlist: User IDs allowed to be mentioned, or `None` for all.

    Returns:
        Text safe to post as Slack `mrkdwn`.
    """
    parts: list[str] = []
    last = 0
    for match in _CODE_SPAN_PATTERN.finditer(text):
        parts.append(_format_prose(text[last : match.start()], mention_allowlist))
        parts.append(_escape_mrkdwn(match.group(0)))
        last = match.end()
    parts.append(_format_prose(text[last:], mention_allowlist))
    return "".join(parts)


def _format_prose(text: str, mention_allowlist: frozenset[str] | None) -> str:
    escaped = _escape_mrkdwn(text)
    linked = _WEB_LINK_PATTERN.sub(
        lambda match: f"<{match.group(2)}|{match.group(1)}>",
        escaped,
    )
    linked = _NON_WEB_LINK_PATTERN.sub(
        lambda match: (
            f"{match.group(1)}{match.group(2).replace('&lt;@', '&amp;lt;@')}{match.group(3)}"
        ),
        linked,
    )
    formatted = format_markdown_for_channel(linked)
    parts: list[str] = []
    last = 0
    for link in _SLACK_LINK_PATTERN.finditer(formatted):
        parts.append(_restore_mentions(formatted[last : link.start()], mention_allowlist))
        parts.append(link.group(0))
        last = link.end()
    parts.append(_restore_mentions(formatted[last:], mention_allowlist))
    return "".join(parts)


def _restore_mentions(text: str, allowlist: frozenset[str] | None) -> str:
    return _ESCAPED_MENTION_PATTERN.sub(
        lambda match: (
            f"<@{match.group(1)}>"
            if allowlist is None or match.group(1) in allowlist
            else match.group(0)
        ),
        text,
    )


def _escape_mrkdwn(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def download_slack_file(
    url: str,
    destination: Path,
    *,
    token: str,
    timeout: float,
    max_bytes: int,
) -> None:
    """Download a private Slack file with the bot token, within a size cap.

    The token is sent only to `https://files.slack.com`, and redirects are refused
    so the `Authorization` header can never be replayed to another host.

    Args:
        url: The file's `url_private_download` from the event payload.
        destination: Local path to write, created with owner-only permissions.
        token: Bot token authorizing the download.
        timeout: Request timeout in seconds.
        max_bytes: Maximum bytes to accept.

    Raises:
        ChannelMediaError: If the URL is not a Slack file URL, or the file is too
            large or arrives incomplete.
    """
    if not _is_file_host_url(url):
        msg = "refusing to download a Slack file from an unexpected host"
        raise ChannelMediaError(msg)
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})  # noqa: S310  # scheme checked above
    with _build_opener().open(request, timeout=timeout) as response:
        length = response.headers.get("content-length")
        expected = parse_content_length(length) if length is not None else None
        if expected is not None and expected > max_bytes:
            msg = f"media file is too large: {expected} bytes exceeds {max_bytes}"
            raise ChannelMediaError(msg)
        _write_capped(response, destination, max_bytes=max_bytes, expected=expected)


def _is_file_host_url(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    return parsed.scheme == "https" and parsed.hostname == _FILE_HOST


def _build_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_RefuseRedirects)


def _write_capped(
    response: IO[bytes],
    destination: Path,
    *,
    max_bytes: int,
    expected: int | None,
) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # `O_NOFOLLOW` stops a planted symlink from redirecting the write elsewhere.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as file:
            _copy_capped(response, file, max_bytes=max_bytes, expected=expected)
    except BaseException:
        # Any failure, including a timeout or reset mid-body, must not leave a
        # partial private file behind.
        destination.unlink(missing_ok=True)
        raise


def _copy_capped(
    response: IO[bytes],
    file: IO[bytes],
    *,
    max_bytes: int,
    expected: int | None,
) -> None:
    total = 0
    while chunk := response.read(64 * 1024):
        total += len(chunk)
        if total > max_bytes:
            msg = f"media file is too large: more than {max_bytes} bytes"
            raise ChannelMediaError(msg)
        file.write(chunk)
    # `HTTPResponse.read(amt)` returns `b""` on a premature EOF rather than
    # raising, so a cut-short body would otherwise pass as a whole file.
    if expected is not None and total != expected:
        msg = f"media download was incomplete: {total} of {expected} bytes"
        raise ChannelMediaError(msg)


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Fail instead of following a redirect, which would resend the bot token."""

    def redirect_request(  # noqa: PLR0913  # stdlib override signature
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> NoReturn:
        """Refuse every redirect.

        Raises:
            ChannelMediaError: Always.
        """
        del req, fp, code, msg, headers, newurl
        error = "Slack file download redirected; refusing to follow"
        raise ChannelMediaError(error)


def _parse_conversation_id(conversation_id: str) -> tuple[str, str | None]:
    """Split a Talon conversation id into a Slack channel and thread.

    Args:
        conversation_id: DM channel id, or `channel:thread_ts`.

    Returns:
        The channel id and, for a thread, its root timestamp.

    Raises:
        ValueError: If the id is not a Slack conversation id.
    """
    match = _CONVERSATION_PATTERN.fullmatch(conversation_id)
    if match is None:
        msg = "not a Slack conversation id"
        raise ValueError(msg)
    return match.group("channel"), match.group("thread")


def _reaction_conversation_id(inbound: _SlackInboundReaction) -> str:
    if inbound.channel_id.startswith("D"):
        return inbound.channel_id
    # Not a message this process posted, so the best guess is a thread root.
    return f"{inbound.channel_id}:{inbound.message_ts}"


def _reaction_emoji(name: str) -> str:
    base = _SKIN_TONE_PATTERN.sub("", name)
    return _REACTION_EMOJI.get(base, name)


def _convert_event(event: dict, *, bot_id: str | None) -> _SlackInboundMessage | None:
    """Convert a Slack message event, or drop it.

    Drops anything posted by a bot, including this one, and every edit, deletion,
    and other message subtype except a file share. Without this, each reply the
    bot posts would come back as a new prompt. A channel message is only taken
    from `app_mention`, because Slack also sends the same mention as a `message`.

    Args:
        event: The `event` object of an Events API payload.
        bot_id: This bot's user id.

    Returns:
        The message, or `None` when it must not reach the agent.
    """
    sender = optional_str(event.get("user"))
    if event.get("bot_id") or sender is None or sender == bot_id:
        return None
    if event.get("subtype") not in {None, "file_share"}:
        return None
    event_type = event.get("type")
    is_dm = event.get("channel_type") == "im"
    if not ((event_type == "message" and is_dm) or event_type == "app_mention"):
        return None
    channel = optional_str(event.get("channel"))
    ts = optional_str(event.get("ts"))
    if channel is None or ts is None:
        return None
    text = str(event.get("text") or "")
    if bot_id is not None:
        text = text.replace(f"<@{bot_id}>", "")
    text = _decode_mrkdwn(text)
    return _SlackInboundMessage(
        channel_id=channel,
        ts=ts,
        thread_ts=optional_str(event.get("thread_ts")),
        sender_id=sender,
        text=text.strip(),
        is_dm=is_dm,
        files=_convert_files(event.get("files")),
    )


def _contains_oauth_callback(text: str) -> bool:
    """Exclude whole historical messages before truncation can hide credentials.

    Scan URLs within prose and Slack mentions using the interception recognizer.
    This intentionally does not validate or complete a pending authorization.
    """
    return any(
        extract_loopback_oauth_callback_url(candidate) is not None
        for candidate in re.findall(r"http://[^\s<>\"'`]+", _decode_mrkdwn(text), re.IGNORECASE)
    )


def _decode_mrkdwn(text: str) -> str:
    """Turn Slack's inbound message encoding back into the text the user typed.

    Slack wraps a pasted URL as `<url>` or `<url|label>` and escapes `&`, `<`, and
    `>`. Left encoded, a pasted OAuth callback reads as `...&amp;state=...` and is
    not recognized. User, channel, and broadcast references such as `<@U123>` are
    kept as Slack sent them.

    Args:
        text: Message text as delivered in the event.

    Returns:
        Text with links unwrapped and entities unescaped.
    """

    def unwrap(match: re.Match[str]) -> str:
        url, label = match.group(1), match.group(2)
        if (
            not label
            or url in {label, f"mailto:{label}"}
            or extract_loopback_oauth_callback_url(url.replace("&amp;", "&"))
        ):
            return url
        return f"{label} ({url})"

    unwrapped = _INBOUND_LINK_PATTERN.sub(unwrap, text)
    # `&amp;` last, so an escaped `&lt;` stays the literal text `&lt;`.
    return unwrapped.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def _convert_files(value: object) -> tuple[_SlackFile, ...]:
    if not isinstance(value, list):
        return ()
    files: list[_SlackFile] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        file_id = optional_str(item.get("id"))
        url = optional_str(item.get("url_private_download")) or optional_str(
            item.get("url_private"),
        )
        size = item.get("size")
        if file_id is None or url is None or not isinstance(size, int):
            continue
        files.append(
            _SlackFile(
                file_id=file_id,
                url=url,
                filename=optional_str(item.get("name")) or file_id,
                mimetype=optional_str(item.get("mimetype")),
                size=size,
            ),
        )
    return tuple(files)


def _convert_reaction(event: dict, *, bot_id: str | None) -> _SlackInboundReaction | None:
    sender = optional_str(event.get("user"))
    reaction = optional_str(event.get("reaction"))
    item = event.get("item")
    if sender is None or sender == bot_id or reaction is None or not isinstance(item, dict):
        return None
    if item.get("type") != "message":
        return None
    channel = optional_str(item.get("channel"))
    ts = optional_str(item.get("ts"))
    if channel is None or ts is None:
        return None
    return _SlackInboundReaction(
        channel_id=channel,
        message_ts=ts,
        sender_id=sender,
        reaction=reaction,
    )


def _convert_command(payload: dict) -> _SlackInboundCommand | None:
    """Convert a slash command payload, or drop it.

    The command's own name is not checked. Socket Mode delivers only the slash
    commands of the app that owns the connection, so any name the operator gives
    it in the manifest, such as `/talon-dev`, reaches Talon the same way.

    Args:
        payload: The Socket Mode `slash_commands` payload.

    Returns:
        The invocation, or `None` when the payload is malformed.
    """
    channel = optional_str(payload.get("channel_id"))
    sender = optional_str(payload.get("user_id"))
    response_url = optional_str(payload.get("response_url"))
    if channel is None or sender is None or response_url is None:
        return None
    if not _is_slack_response_url(response_url):
        logger.warning("Dropping Slack command with an unexpected response URL")
        return None
    words = str(payload.get("text") or "").split(maxsplit=1)
    command = words[0].lower().removeprefix("/") if words else "help"
    return _SlackInboundCommand(
        command=command,
        channel_id=channel,
        sender_id=sender,
        trigger_id=optional_str(payload.get("trigger_id")) or "",
        responder=_WebhookResponder(response_url),
        # Passed through as typed, so `/talon pair approve <code>` reaches the host
        # exactly as `/pair approve <code>` would.
        argument=words[1].strip() if len(words) > 1 else None,
    )


def _is_slack_response_url(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ""
    return parsed.scheme == "https" and (host == "slack.com" or host.endswith(".slack.com"))


@dataclass(frozen=True, slots=True)
class _WebhookResponder:
    """Command reply surface backed by a slash command's `response_url`."""

    response_url: str = field(repr=False)

    async def reject(self, text: str) -> None:
        """Answer privately to the invoking user.

        Args:
            text: Refusal to show the invoking user.
        """
        await AsyncWebhookClient(self.response_url).send(text=text, response_type="ephemeral")

    async def send(self, text: str) -> str | None:
        """Send a visible reply.

        Args:
            text: Reply content, already within Slack's per-message limit.

        Returns:
            Always `None`: a `response_url` reply reports no message timestamp.
        """
        await AsyncWebhookClient(self.response_url).send(text=text, response_type="in_channel")
        return None


def _message_metadata(inbound: _SlackInboundMessage) -> dict[str, object]:
    metadata: dict[str, object] = {
        "provider": "slack",
        "is_dm": inbound.is_dm,
        "from_self": False,
    }
    if not inbound.is_dm:
        metadata["history_chat"] = inbound.channel_id
    if inbound.files:
        file = inbound.files[0]
        metadata["media_type"] = _file_media_type(file)
        if file.mimetype:
            metadata["mime_type"] = file.mimetype
    return metadata


def _file_media_type(file: _SlackFile) -> str:
    mimetype = (file.mimetype or "").lower()
    if mimetype.startswith("image/"):
        return "image"
    if mimetype.startswith("video/"):
        return "video"
    if mimetype.startswith("audio/"):
        return "audio"
    return "document"


def _inbound_media_filename(*, message_id: str | None, file: _SlackFile) -> str:
    message = safe_filename_part(message_id or "message")
    token = safe_filename_part(file.file_id)[-24:]
    return f"{message}_{token}{safe_suffix(file.filename, file.mimetype)}"


def _allows_slack_message(
    exposure: ChannelExposure,
    allowed_user_ids: frozenset[str],
    message: ChannelMessage,
) -> bool:
    if exposure.mode == ExposureMode.ALLOWLIST:
        if message.metadata.get("is_dm") is True and message.sender_id in allowed_user_ids:
            return True
        # An allowlisted channel admits every thread in it.
        channel_id = message.conversation_id.split(":", maxsplit=1)[0]
        if channel_id in exposure.conversations:
            return True
    return exposure.allows(message)
