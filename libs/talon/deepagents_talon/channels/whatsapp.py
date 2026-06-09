"""WhatsApp channel adapter backed by a loopback Node bridge.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shlex
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, cast

from deepagents_talon.channels.base import (
    MAX_TEXT_CHARS,
    ChannelExposure,
    ExposureMode,
    chunk_text,
    format_markdown_for_channel,
    validate_media,
)
from deepagents_talon.interfaces import ChannelMedia, ChannelMessage, ChannelStatus, MessageHandler

if TYPE_CHECKING:
    from collections.abc import Mapping

    from deepagents_talon.config import TalonConfig

logger = logging.getLogger(__name__)

DEFAULT_BRIDGE_HOST = "127.0.0.1"
DEFAULT_BRIDGE_PORT = 3000
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_HEALTH_INTERVAL_SECONDS = 5.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 10.0
DEFAULT_BRIDGE_START_TIMEOUT_SECONDS = 10.0
DEFAULT_BOT_HEADER = "deepagents bot"
DEFAULT_BRIDGE_TOKEN_BYTES = 32
_FAILED_HEALTH_RESTART_THRESHOLD = 3
OPEN_EXPOSURE_ACK_ENV = "DEEPAGENTS_TALON_WHATSAPP_OPEN_ACK"
OPEN_EXPOSURE_ACK_VALUE = "allow-arbitrary-senders"


class WhatsAppBridgeError(RuntimeError):
    """Raised when the WhatsApp bridge reports or causes a transport error."""


@dataclass(frozen=True, slots=True)
class WhatsAppChannelConfig:
    """Configuration for the WhatsApp channel adapter.

    Args:
        session_dir: Directory for bridge authentication and Chromium profile state.
        inbound_media_dir: Directory where the bridge stores downloaded inbound media.
        outbound_media_dir: Optional root that outbound media must remain under
            before it is staged for the bridge.
        host: Loopback host where the bridge listens.
        port: Loopback port where the bridge listens.
        exposure: Inbound trigger policy.
        bot_header: Header prepended to outbound messages so self-message chats
            can distinguish bot replies from operator-authored messages.
        bridge_command: Optional command used to start the Node bridge subprocess.
        chrome_path: Optional Chrome or Chromium executable path for Puppeteer.
        web_version_cache_url: Optional pinned WhatsApp Web HTML cache URL.
        bridge_token: Bearer token shared with the loopback bridge process.
        poll_interval_seconds: Interval for draining inbound bridge messages.
        health_interval_seconds: Interval for bridge health checks.
        request_timeout_seconds: Per-request timeout for loopback bridge calls.
    """

    session_dir: Path
    inbound_media_dir: Path | None = None
    outbound_media_dir: Path | None = None
    host: str = DEFAULT_BRIDGE_HOST
    port: int = DEFAULT_BRIDGE_PORT
    exposure: ChannelExposure = field(default_factory=ChannelExposure)
    bot_header: str = DEFAULT_BOT_HEADER
    bridge_command: tuple[str, ...] | None = None
    chrome_path: str | None = None
    web_version_cache_url: str | None = None
    bridge_token: str = field(
        default_factory=lambda: secrets.token_hex(DEFAULT_BRIDGE_TOKEN_BYTES),
        repr=False,
    )
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    health_interval_seconds: float = DEFAULT_HEALTH_INTERVAL_SECONDS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS

    @classmethod
    def from_talon_config(cls, config: TalonConfig) -> WhatsAppChannelConfig:
        """Build WhatsApp channel configuration from Talon environment values.

        Args:
            config: Talon process configuration.

        Returns:
            WhatsApp channel configuration.

        Raises:
            ValueError: If exposure or port environment values are invalid.
        """
        env = config.env
        host = env.get("DEEPAGENTS_TALON_WHATSAPP_BRIDGE_HOST", DEFAULT_BRIDGE_HOST)
        port = _parse_int(env.get("DEEPAGENTS_TALON_WHATSAPP_BRIDGE_PORT"), DEFAULT_BRIDGE_PORT)
        session = Path(
            env.get("DEEPAGENTS_TALON_WHATSAPP_SESSION_DIR", str(config.channel_dir / "whatsapp")),
        )
        inbound_media_dir = Path(
            env.get(
                "DEEPAGENTS_TALON_WHATSAPP_MEDIA_DIR",
                str(config.inbound_media_dir / "whatsapp"),
            ),
        )
        outbound_media_dir = Path(
            env.get("DEEPAGENTS_TALON_OUTBOUND_MEDIA_DIR")
            or env.get("DEEPAGENTS_TALON_WORKSPACE")
            or "/workspace",
        )
        command = _bridge_command(env)
        return cls(
            session_dir=session,
            inbound_media_dir=inbound_media_dir,
            outbound_media_dir=outbound_media_dir,
            host=host,
            port=port,
            exposure=_exposure_from_env(env),
            bot_header=env.get("DEEPAGENTS_TALON_WHATSAPP_BOT_HEADER", DEFAULT_BOT_HEADER),
            bridge_command=command,
            chrome_path=env.get("DEEPAGENTS_TALON_WHATSAPP_CHROME_PATH"),
            web_version_cache_url=env.get("DEEPAGENTS_TALON_WHATSAPP_WEB_VERSION_CACHE_URL"),
            bridge_token=env.get("DEEPAGENTS_TALON_WHATSAPP_BRIDGE_TOKEN")
            or secrets.token_hex(DEFAULT_BRIDGE_TOKEN_BYTES),
            poll_interval_seconds=_parse_float(
                env.get("DEEPAGENTS_TALON_WHATSAPP_POLL_SECONDS"),
                DEFAULT_POLL_INTERVAL_SECONDS,
            ),
            health_interval_seconds=_parse_float(
                env.get("DEEPAGENTS_TALON_WHATSAPP_HEALTH_SECONDS"),
                DEFAULT_HEALTH_INTERVAL_SECONDS,
            ),
            request_timeout_seconds=_parse_float(
                env.get("DEEPAGENTS_TALON_WHATSAPP_REQUEST_TIMEOUT_SECONDS"),
                DEFAULT_REQUEST_TIMEOUT_SECONDS,
            ),
        )

    @property
    def base_url(self) -> str:
        """Loopback bridge base URL."""
        return f"http://{self.host}:{self.port}"


class BridgeTransport:
    """Small JSON HTTP client for the loopback bridge."""

    def __init__(self, *, base_url: str, timeout: float, token: str | None = None) -> None:
        """Initialize the transport.

        Args:
            base_url: Bridge base URL.
            timeout: Request timeout in seconds.
            token: Optional bearer token for bridge authentication.
        """
        self.base_url = _validate_loopback_url(base_url.rstrip("/"))
        self.timeout = timeout
        self.token = token

    async def get(self, path: str) -> object:
        """Send a GET request and decode JSON.

        Args:
            path: Absolute bridge endpoint path.

        Returns:
            JSON-decoded response body.
        """
        return await asyncio.to_thread(self._request, "GET", path, None)

    async def post(self, path: str, payload: Mapping[str, object]) -> object:
        """Send a POST request with a JSON body and decode JSON.

        Args:
            path: Absolute bridge endpoint path.
            payload: JSON-serializable request body.

        Returns:
            JSON-decoded response body.
        """
        return await asyncio.to_thread(self._request, "POST", path, payload)

    def _request(self, method: str, path: str, payload: Mapping[str, object] | None) -> object:
        body = None if payload is None else json.dumps(payload).encode()
        headers = {"content-type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(  # noqa: S310  # base URL is validated as HTTP loopback.
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(  # noqa: S310  # bridge transport is loopback-only.
                request,
                timeout=self.timeout,
            ) as response:
                return json.loads(response.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            msg = f"WhatsApp bridge request failed: {method} {path}"
            raise WhatsAppBridgeError(msg) from error


class WhatsAppChannel:
    """Channel adapter for WhatsApp via a local Node bridge."""

    def __init__(
        self,
        config: WhatsAppChannelConfig,
        *,
        transport: BridgeTransport | None = None,
    ) -> None:
        """Initialize the WhatsApp channel without starting it.

        Args:
            config: WhatsApp channel configuration.
            transport: Optional test transport implementing the bridge API.
        """
        self.config = config
        self._transport = transport or BridgeTransport(
            base_url=config.base_url,
            timeout=config.request_timeout_seconds,
            token=config.bridge_token,
        )
        self._handler: MessageHandler | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._bridge_stdout: asyncio.Task[None] | None = None
        self._bridge_stderr: asyncio.Task[None] | None = None
        self._poll: asyncio.Task[None] | None = None
        self._health: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()
        self._status = ChannelStatus(provider="whatsapp", connected=False, detail="disconnected")
        self._failed_health_checks = 0

    def set_message_handler(self, handler: MessageHandler) -> None:
        """Register the host callback for inbound messages.

        Args:
            handler: Coroutine callback invoked for accepted inbound messages.
        """
        self._handler = handler

    async def start(self) -> None:
        """Start the bridge subprocess and background polling tasks."""
        self.config.session_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.config.session_dir.chmod(0o700)
        media_dir = _bridge_media_dir(self.config)
        media_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        media_dir.chmod(0o700)
        self._stopped.clear()
        await self._start_bridge()
        self._poll = asyncio.create_task(self._poll_messages(), name="talon:whatsapp:poll")
        self._health = asyncio.create_task(self._watch_health(), name="talon:whatsapp:health")

    async def stop(self) -> None:
        """Stop polling tasks and terminate the bridge subprocess."""
        self._stopped.set()
        tasks = [task for task in (self._poll, self._health) if task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._poll = None
        self._health = None
        await self._stop_bridge()
        self._status = ChannelStatus(provider="whatsapp", connected=False, detail="disconnected")

    async def send_message(self, conversation_id: str, text: str) -> None:
        """Send chunked, formatted text to a WhatsApp chat.

        Args:
            conversation_id: WhatsApp chat id.
            text: Message content to send.
        """
        for chunk in _chunk_with_bot_header(text, bot_header=self.config.bot_header):
            await self._post_result("/send", {"chat_id": conversation_id, "text": chunk})

    async def send_media(self, conversation_id: str, media: ChannelMedia) -> None:
        """Send validated image or video media to a WhatsApp chat.

        Args:
            conversation_id: WhatsApp chat id.
            media: Media payload to send.
        """
        checked = validate_media(media, root=self.config.outbound_media_dir)
        staged = await asyncio.to_thread(_stage_bridge_media, checked.path, self.config)
        payload: dict[str, object] = {
            "chat_id": conversation_id,
            "chatId": conversation_id,
            "path": str(staged),
            "filePath": str(staged),
            "mediaType": checked.media_type,
        }
        if checked.caption is not None:
            payload["caption"] = _with_bot_header(
                checked.caption, bot_header=self.config.bot_header
            )
        else:
            payload["caption"] = _bot_header(self.config.bot_header)
        await self._post_result("/send-media", payload)

    async def send_typing(self, conversation_id: str) -> None:
        """Send a WhatsApp typing indicator when the bridge supports it.

        Args:
            conversation_id: WhatsApp chat id.
        """
        await self._post_result("/typing", {"chat_id": conversation_id, "chatId": conversation_id})

    async def edit_message(self, conversation_id: str, message_id: str, text: str) -> None:
        """Edit a previously sent WhatsApp message.

        Args:
            conversation_id: WhatsApp chat id.
            message_id: Bridge message id.
            text: Replacement content.
        """
        await self._post_result(
            "/edit",
            {
                "chat_id": conversation_id,
                "chatId": conversation_id,
                "message_id": message_id,
                "messageId": message_id,
                "content": _with_bot_header(text, bot_header=self.config.bot_header),
                "message": _with_bot_header(text, bot_header=self.config.bot_header),
            },
        )

    async def status(self) -> ChannelStatus:
        """Report the most recent bridge connection status."""
        return self._status

    async def _start_bridge(self) -> None:
        if self.config.bridge_command is None or self._process is not None:
            return
        env = {
            **os.environ,
            "WHATSAPP_BRIDGE_HOST": self.config.host,
            "WHATSAPP_BRIDGE_PORT": str(self.config.port),
            "WHATSAPP_SESSION_DIR": str(self.config.session_dir),
            "WHATSAPP_BOT_HEADER": self.config.bot_header,
            "WHATSAPP_BRIDGE_TOKEN": self.config.bridge_token,
            "WHATSAPP_MEDIA_DIR": str(_bridge_media_dir(self.config)),
        }
        if self.config.chrome_path:
            env["WHATSAPP_CHROME_PATH"] = self.config.chrome_path
        if self.config.web_version_cache_url:
            env["WHATSAPP_WEB_VERSION_CACHE_URL"] = self.config.web_version_cache_url
        self._process = await asyncio.create_subprocess_exec(
            *self.config.bridge_command,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if self._process.stdout is not None:
            self._bridge_stdout = asyncio.create_task(
                self._forward_bridge_output(self._process.stdout, logging.INFO),
                name="talon:whatsapp:bridge:stdout",
            )
        if self._process.stderr is not None:
            self._bridge_stderr = asyncio.create_task(
                self._forward_bridge_output(self._process.stderr, logging.ERROR),
                name="talon:whatsapp:bridge:stderr",
            )
        await self._wait_for_bridge()

    async def _stop_bridge(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5)
        except TimeoutError:
            self._process.kill()
            await self._process.wait()
        await self._stop_bridge_output_tasks()
        self._process = None

    async def _restart_bridge(self) -> None:
        logger.warning("Restarting WhatsApp bridge after failed health checks")
        await self._stop_bridge()
        await self._start_bridge()
        self._failed_health_checks = 0

    async def _poll_messages(self) -> None:
        while not self._stopped.is_set():
            try:
                payload = await self._transport.get("/messages")
                for message in _parse_messages(payload):
                    if self.config.exposure.allows(message):
                        await self._dispatch(message)
                    else:
                        logger.debug(
                            "Dropping WhatsApp message %s from %s due to exposure policy",
                            message.message_id,
                            message.conversation_id,
                        )
            except WhatsAppBridgeError:
                logger.exception("Failed to poll WhatsApp bridge messages")
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _watch_health(self) -> None:
        while not self._stopped.is_set():
            try:
                payload = await self._transport.get("/health")
                self._status = _parse_status(payload)
                self._failed_health_checks = 0
            except WhatsAppBridgeError:
                self._failed_health_checks += 1
                self._status = ChannelStatus(
                    provider="whatsapp",
                    connected=False,
                    detail="disconnected",
                )
                if self._failed_health_checks >= _FAILED_HEALTH_RESTART_THRESHOLD:
                    await self._restart_bridge()
            await asyncio.sleep(self.config.health_interval_seconds)

    async def _wait_for_bridge(self) -> None:
        deadline = asyncio.get_running_loop().time() + DEFAULT_BRIDGE_START_TIMEOUT_SECONDS
        last_error: WhatsAppBridgeError | None = None
        while not self._stopped.is_set():
            if self._process is not None and self._process.returncode is not None:
                msg = f"WhatsApp bridge exited during startup with code {self._process.returncode}"
                raise WhatsAppBridgeError(msg) from last_error
            try:
                payload = await self._transport.get("/health")
                status = _parse_status(payload)
            except WhatsAppBridgeError as error:
                last_error = error
                if asyncio.get_running_loop().time() >= deadline:
                    msg = "WhatsApp bridge did not become ready before startup timeout"
                    raise WhatsAppBridgeError(msg) from error
                await asyncio.sleep(0.2)
            else:
                self._status = status
                return

    async def _forward_bridge_output(
        self,
        stream: asyncio.StreamReader,
        level: int,
    ) -> None:
        async for raw in stream:
            line = raw.decode(errors="replace").rstrip()
            if line:
                logger.log(level, "WhatsApp bridge: %s", line)

    async def _stop_bridge_output_tasks(self) -> None:
        tasks = [
            task
            for task in (self._bridge_stdout, self._bridge_stderr)
            if task is not None and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._bridge_stdout = None
        self._bridge_stderr = None

    async def _dispatch(self, message: ChannelMessage) -> None:
        if self._handler is None:
            logger.warning("Dropping WhatsApp message because no handler is registered")
            return
        await self._handler(message)

    async def _post_result(self, path: str, payload: Mapping[str, object]) -> object:
        response = await self._transport.post(path, payload)
        if isinstance(response, dict):
            result = cast("Mapping[str, object]", response)
            if result.get("success") is not False:
                return response
            msg = str(result.get("error") or "WhatsApp bridge returned an error")
            raise WhatsAppBridgeError(msg)
        return response


def _bot_header(value: str) -> str:
    return format_markdown_for_channel(f"**{value}**")


def _with_bot_header(text: str, *, bot_header: str) -> str:
    header = _bot_header(bot_header)
    if not text:
        return header
    return f"{header}\n{format_markdown_for_channel(text)}"


def _chunk_with_bot_header(text: str, *, bot_header: str) -> list[str]:
    header = _bot_header(bot_header)
    limit = MAX_TEXT_CHARS - len(header) - 1
    chunks = chunk_text(format_markdown_for_channel(text), limit=limit)
    return [f"{header}\n{chunk}" for chunk in chunks]


def bridge_script_path() -> Path:
    """Return the packaged Node bridge script path."""
    return Path(str(files("deepagents_talon.channels.whatsapp_bridge").joinpath("bridge.js")))


def _bridge_media_dir(config: WhatsAppChannelConfig) -> Path:
    return config.inbound_media_dir or config.session_dir.parent / "media"


def _stage_bridge_media(path: Path, config: WhatsAppChannelConfig) -> Path:
    media_dir = _bridge_media_dir(config).expanduser().resolve()
    source = path.expanduser().resolve()
    if source.is_relative_to(media_dir):
        return source

    media_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = media_dir / f"outbound_{secrets.token_hex(12)}{source.suffix}"
    shutil.copyfile(source, destination)
    destination.chmod(0o600)
    return destination


def _validate_loopback_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        msg = "WhatsApp bridge URL must use HTTP loopback"
        raise WhatsAppBridgeError(msg)
    return value


def _parse_messages(payload: object) -> list[ChannelMessage]:
    if not isinstance(payload, list):
        msg = "WhatsApp bridge /messages response must be a list"
        raise WhatsAppBridgeError(msg)
    return [_parse_message(item) for item in payload]


def _parse_message(payload: object) -> ChannelMessage:
    if not isinstance(payload, dict):
        msg = "WhatsApp bridge message must be an object"
        raise WhatsAppBridgeError(msg)
    values = cast("Mapping[str, object]", payload)
    media_paths = _str_list(
        values.get("media_paths")
        or values.get("mediaPaths")
        or values.get("mediaUrls")
        or values.get("media_urls"),
    )
    media_mime_types = _str_list(
        values.get("media_mime_types")
        or values.get("mediaMimeTypes")
        or values.get("mimeTypes")
        or values.get("media_types"),
    )
    message_type = _optional_str(
        values.get("message_type") or values.get("messageType") or values.get("mediaType"),
    )
    media_type = _message_media_type(values, message_type, media_mime_types)
    text = values.get("text")
    if not isinstance(text, str):
        text = values.get("body")
    return ChannelMessage(
        conversation_id=_required_str_any(values, ("chat_id", "chatId")),
        text=text if isinstance(text, str) else "",
        sender_id=_optional_str(values.get("user_id") or values.get("senderId")),
        message_id=_optional_str(values.get("message_id") or values.get("messageId")),
        metadata={
            "provider": "whatsapp",
            "message_type": message_type,
            "media_type": media_type,
            "chat_name": values.get("chat_name") or values.get("chatName"),
            "chat_type": values.get("chat_type") or values.get("chatType"),
            "chat_id_from": values.get("chat_id_from") or values.get("chatIdFrom"),
            "user_name": values.get("user_name") or values.get("senderName"),
            "media_paths": media_paths,
            "media_path": media_paths[0] if media_paths else None,
            "media_mime_types": media_mime_types,
            "media_types": media_mime_types,
            "voice_path": media_paths[0] if media_paths and media_type == "voice" else None,
            "has_media": bool(values.get("has_media") or values.get("hasMedia") or media_paths),
            "raw_message": values.get("raw_message") or {},
            "from_self": bool(values.get("from_self") or values.get("fromSelf")),
        },
    )


def _parse_status(payload: object) -> ChannelStatus:
    if not isinstance(payload, dict):
        msg = "WhatsApp bridge /health response must be an object"
        raise WhatsAppBridgeError(msg)
    values = cast("Mapping[str, object]", payload)
    detail = _required_str(values, "status")
    return ChannelStatus(
        provider="whatsapp",
        connected=detail == "connected",
        detail=detail,
    )


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        msg = f"WhatsApp bridge payload missing string field: {key}"
        raise WhatsAppBridgeError(msg)
    return value


def _required_str_any(payload: Mapping[str, object], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    names = ", ".join(keys)
    msg = f"WhatsApp bridge payload missing string field: {names}"
    raise WhatsAppBridgeError(msg)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _message_media_type(
    values: Mapping[str, object],
    message_type: str | None,
    media_mime_types: list[str],
) -> str | None:
    raw = _optional_str(values.get("media_type") or values.get("mediaType"))
    candidates = [raw, message_type, *media_mime_types]
    if any(_is_voice_type(candidate) for candidate in candidates):
        return "voice"
    if any(_is_image_type(candidate) for candidate in candidates):
        return "image"
    if any(_is_video_type(candidate) for candidate in candidates):
        return "video"
    return raw or message_type


def _is_voice_type(value: str | None) -> bool:
    if value is None:
        return False
    lowered = value.lower()
    return "audio" in lowered or lowered in {"voice", "ptt"}


def _is_image_type(value: str | None) -> bool:
    if value is None:
        return False
    lowered = value.lower()
    return "image" in lowered or lowered in {"photo", "sticker"}


def _is_video_type(value: str | None) -> bool:
    return isinstance(value, str) and "video" in value.lower()


def _exposure_from_env(env: Mapping[str, str]) -> ChannelExposure:
    mode = _exposure_mode(env.get("DEEPAGENTS_TALON_WHATSAPP_EXPOSURE", ExposureMode.SELF.value))
    if mode == ExposureMode.OPEN:
        _require_open_acknowledgement(env)
        logger.warning(
            "WhatsApp open exposure enabled; arbitrary senders can trigger the agent with "
            "operator credentials and local host access"
        )
    conversations = _split_csv(env.get("DEEPAGENTS_TALON_WHATSAPP_ALLOWLIST_CHATS", ""))
    mentions = tuple(_split_csv(env.get("DEEPAGENTS_TALON_WHATSAPP_MENTION_PATTERNS", "")))
    return ChannelExposure(
        mode=mode,
        operator_id=env.get("DEEPAGENTS_TALON_WHATSAPP_OPERATOR_ID"),
        conversations=frozenset(conversations),
        mention_patterns=mentions,
    )


def _exposure_mode(value: str) -> ExposureMode:
    try:
        return ExposureMode(value)
    except ValueError as error:
        modes = ", ".join(mode.value for mode in ExposureMode)
        msg = f"invalid WhatsApp exposure mode {value!r}; expected one of: {modes}"
        raise ValueError(msg) from error


def _require_open_acknowledgement(env: Mapping[str, str]) -> None:
    if env.get(OPEN_EXPOSURE_ACK_ENV) == OPEN_EXPOSURE_ACK_VALUE:
        return
    msg = (
        "WhatsApp exposure mode 'open' allows arbitrary senders to trigger the agent with "
        "operator credentials and local host access; set "
        f"{OPEN_EXPOSURE_ACK_ENV}={OPEN_EXPOSURE_ACK_VALUE} to acknowledge this risk"
    )
    raise ValueError(msg)


def _bridge_command(env: Mapping[str, str]) -> tuple[str, ...] | None:
    value = env.get("DEEPAGENTS_TALON_WHATSAPP_BRIDGE_COMMAND")
    if value:
        return tuple(shlex.split(value))
    if env.get("DEEPAGENTS_TALON_WHATSAPP_START_BRIDGE", "").lower() in {"1", "true", "yes"}:
        return ("node", str(bridge_script_path()))
    return None


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        msg = f"expected integer value, got {value!r}"
        raise ValueError(msg) from error


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as error:
        msg = f"expected float value, got {value!r}"
        raise ValueError(msg) from error
