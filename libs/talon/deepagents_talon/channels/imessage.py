"""Native macOS iMessage adapter using the externally installed `imsg` CLI."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_talon.channels.base import (
    DEFAULT_MAX_MEDIA_BYTES,
    ChannelExposure,
    ChannelExposureEnv,
    channel_exposure_from_env,
    chunk_text,
    dispatch_message,
    max_media_bytes_from_env,
    outbound_media_root_from_env,
    validate_media,
)
from deepagents_talon.interfaces import (
    ChannelMedia,
    ChannelMessage,
    ChannelStatus,
    MessageHandler,
    SendResult,
)

if TYPE_CHECKING:
    from deepagents_talon.config import TalonConfig

_ENV_PREFIX = "DEEPAGENTS_TALON_IMESSAGE"
_MAX_OUTPUT_BYTES = 64 * 1024
_SEND_TIMEOUT_SECONDS = 30.0
_STOP_TIMEOUT_SECONDS = 2.0
_MAX_CHAT_ID = 2**63 - 1
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IMessageChannelConfig:
    """Configure native Messages access.

    Args:
        cli_path: Installed `imsg` executable, not a shell command.
        exposure: Admission policy; operator ids are inbound phone or email handles.
        outbound_media_dir: Trusted root for outbound attachments.
        max_media_bytes: Maximum outbound attachment size.
    """

    cli_path: str = "imsg"
    exposure: ChannelExposure = field(default_factory=ChannelExposure)
    outbound_media_dir: Path = field(default_factory=Path.cwd)
    max_media_bytes: int = DEFAULT_MAX_MEDIA_BYTES

    @classmethod
    def from_talon_config(cls, config: TalonConfig) -> IMessageChannelConfig:
        """Read iMessage policy and media limits from the Talon environment.

        Args:
            config: Talon process configuration.
        """
        return cls(
            cli_path=config.env.get(f"{_ENV_PREFIX}_CLI_PATH", "imsg"),
            exposure=channel_exposure_from_env(
                config.env,
                ChannelExposureEnv(
                    provider="iMessage",
                    env_prefix=_ENV_PREFIX,
                    open_ack=f"{_ENV_PREFIX}_OPEN_ACK",
                    require_self_operator=True,
                ),
            ),
            outbound_media_dir=outbound_media_root_from_env(config.env),
            max_media_bytes=max_media_bytes_from_env(config.env),
        )


class IMessageChannel:
    """Receive new text messages and send through native macOS Messages chats."""

    def __init__(self, config: IMessageChannelConfig) -> None:
        """Initialize an adapter without launching the CLI.

        Args:
            config: Native channel configuration.
        """
        self.config = config
        self._handler: MessageHandler | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._watch: asyncio.Task[None] | None = None
        self._status = ChannelStatus(provider="imessage", connected=False, detail="disconnected")

    def set_message_handler(self, handler: MessageHandler) -> None:
        """Register the host callback for admitted inbound messages.

        Args:
            handler: Callback invoked after shared exposure checks.
        """
        self._handler = handler

    async def start(self) -> None:
        """Start watching the current tail; messages during downtime are not replayed."""
        if self._watch is not None and not self._watch.done():
            return
        if sys.platform != "darwin":
            msg = "iMessage requires macOS with Messages signed in and the imsg CLI installed"
            raise RuntimeError(msg)
        if not shutil.which(self.config.cli_path):
            msg = "Install steipete/imsg and set DEEPAGENTS_TALON_IMESSAGE_CLI_PATH if needed"
            raise RuntimeError(msg)
        try:
            self._process = await self._spawn("watch", "--json")
        except OSError:
            msg = "Unable to launch imsg; check CLI_PATH, executable permissions, and macOS access"
            raise RuntimeError(msg) from None
        self._status = ChannelStatus(provider="imessage", connected=True)
        self._watch = asyncio.create_task(self._read_messages(self._process))

    async def stop(self) -> None:
        """Cancel the watcher and terminate and reap its subprocess."""
        if self._watch is not None:
            self._watch.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch
        if self._process is not None:
            await _reap(self._process)
        self._watch = None
        self._process = None
        self._status = ChannelStatus(provider="imessage", connected=False, detail="disconnected")

    async def status(self) -> ChannelStatus:
        """Return current watcher connectivity."""
        return self._status

    async def send_message(self, conversation_id: str, text: str) -> SendResult:
        """Send chunked plain text to an existing numeric chat without retries.

        Args:
            conversation_id: Existing positive numeric chat id.
            text: Plain text to split and send.
        """
        if not _valid_conversation_id(conversation_id):
            return SendResult(success=False, error="iMessage requires a positive numeric chat id")
        for chunk in chunk_text(text):
            result = await self._send(conversation_id, "--text", chunk)
            if not result.success:
                return result
        return SendResult(success=True)

    async def send_media(self, conversation_id: str, media: ChannelMedia) -> SendResult:
        """Send a validated local file within the trusted outbound media root.

        Args:
            conversation_id: Existing positive numeric chat id.
            media: Local file and optional caption.
        """
        if not _valid_conversation_id(conversation_id):
            return SendResult(success=False, error="iMessage requires a positive numeric chat id")
        try:
            validated = validate_media(
                media, root=self.config.outbound_media_dir, max_bytes=self.config.max_media_bytes
            )
        except ValueError, OSError:
            return SendResult(
                success=False, error="iMessage media failed path, type, or size validation"
            )
        args = ["--file", str(validated.path.resolve())]
        if validated.caption:
            args.extend(("--text", validated.caption))
        return await self._send(conversation_id, *args)

    async def edit_message(self, conversation_id: str, message_id: str, text: str) -> SendResult:
        """Report unsupported message editing.

        Args:
            conversation_id: Target chat id.
            message_id: Original message id.
            text: Requested replacement text.
        """
        del conversation_id, message_id, text
        return SendResult(success=False, error="iMessage editing is not supported")

    async def send_typing(self, conversation_id: str) -> None:
        """Leave typing unchanged because the CLI does not support it.

        Args:
            conversation_id: Target chat id.
        """
        del conversation_id

    async def _spawn(self, *args: str) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            self.config.cli_path,
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=_MAX_OUTPUT_BYTES,
        )

    async def _read_messages(self, process: asyncio.subprocess.Process) -> None:
        try:
            if process.stdout is None:
                return
            while line := await process.stdout.readline():
                message = _parse_message(line)
                if message is not None and self.config.exposure.allows(message):
                    await dispatch_message(self._handler, message, provider="imessage")
        except OSError, ValueError, RuntimeError:
            logger.warning(
                "imsg watch failed; check Messages, Full Disk Access, and CLI permissions"
            )
        finally:
            self._status = ChannelStatus(
                provider="imessage",
                connected=False,
                detail="imsg watch disconnected; check Messages and macOS permissions",
            )
            await _reap(process)

    async def _send(self, conversation_id: str, *args: str) -> SendResult:
        process: asyncio.subprocess.Process | None = None
        try:
            async with asyncio.timeout(_SEND_TIMEOUT_SECONDS):
                process = await self._spawn("send", "--chat-id", conversation_id, *args, "--json")
                output = await _read_output(process)
            if process.returncode != 0:
                return SendResult(success=False, error="imsg send failed; delivery is unknown")
            payload = json.loads(output)
            if isinstance(payload, dict) and payload.get("status") == "sent":
                return SendResult(success=True)
        except OSError, ValueError, TimeoutError:
            return SendResult(success=False, error="imsg send failed; delivery is unknown")
        finally:
            if process is not None:
                await _reap(process)
        return SendResult(success=False, error="imsg did not confirm delivery")


def _valid_conversation_id(value: str) -> bool:
    return (
        0 < len(value) <= len(str(_MAX_CHAT_ID))
        and value.isascii()
        and value.isdecimal()
        and 0 < int(value) <= _MAX_CHAT_ID
    )


def _positive_id(value: object) -> bool:
    return type(value) is int and value > 0


def _parse_message(line: bytes) -> ChannelMessage | None:
    try:
        payload = json.loads(line)
    except ValueError, UnicodeDecodeError:
        return None
    if not isinstance(payload, dict) or not _valid_origin(payload):
        return None
    if not _positive_id(payload.get("id")) or not _positive_id(payload.get("chat_id")):
        return None
    sender, text = payload.get("sender"), payload.get("text")
    if (
        not isinstance(sender, str)
        or not sender.strip()
        or not isinstance(text, str)
        or not text.strip()
    ):
        return None
    return ChannelMessage(
        conversation_id=str(payload["chat_id"]),
        message_id=str(payload["id"]),
        sender_id=sender,
        text=text,
        metadata={
            "provider": "imessage",
            "chat_type": "group" if payload["is_group"] else "private",
            "from_self": False,
        },
    )


def _valid_origin(payload: dict[str, object]) -> bool:
    guid = payload.get("chat_guid")
    return (
        payload.get("is_from_me") is False
        and type(payload.get("is_group")) is bool
        and isinstance(guid, str)
        and guid.startswith(("iMessage;-;", "iMessage;+;"))
        and bool(guid.split(";", 2)[-1].strip())
        and "type" not in payload
        and not payload.get("is_reaction")
        and not payload.get("poll")
    )


async def _read_output(process: asyncio.subprocess.Process) -> bytes:
    if process.stdout is None:
        msg = "imsg stdout unavailable"
        raise ValueError(msg)
    output = bytearray()
    while chunk := await process.stdout.read(min(4096, _MAX_OUTPUT_BYTES + 1 - len(output))):
        output.extend(chunk)
        if len(output) > _MAX_OUTPUT_BYTES:
            msg = "imsg output exceeded limit"
            raise ValueError(msg)
    await process.wait()
    return bytes(output)


async def _reap(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
    try:
        await asyncio.wait_for(_finish(process), timeout=_STOP_TIMEOUT_SECONDS)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await asyncio.wait_for(_finish(process), timeout=_STOP_TIMEOUT_SECONDS)


async def _finish(process: asyncio.subprocess.Process) -> None:
    if process.stdout is not None:
        while await process.stdout.read(4096):
            pass
    await process.wait()
