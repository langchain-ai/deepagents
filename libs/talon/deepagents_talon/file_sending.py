"""Send files from the agent's filesystem to the chat that started the turn.

The tool reads through the agent's backend, so it reaches files in a local
workspace and in a remote sandbox alike. The host stages the bytes in a private
directory only for as long as the channel upload takes.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import unicodedata
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, tool

from deepagents_talon.background import _IN_SUBAGENT
from deepagents_talon.channels.base import (
    media_type_for_path,
    send_with_retry,
)
from deepagents_talon.interfaces import ChannelMedia, SendResult

if TYPE_CHECKING:
    from collections.abc import Generator

    from deepagents.backends.protocol import BackendProtocol

    from deepagents_talon.interfaces import ChannelAdapter, FileSendHandler

FILE_HANDLER: ContextVar[FileSendHandler | None] = ContextVar("talon_file_handler", default=None)

_MAX_NAME_BYTES = 200
"""Leaves room under the common 255-byte file name limit for any adapter renaming."""


@contextmanager
def file_scope(handler: FileSendHandler | None) -> Generator[None]:
    """Bind the turn's file sender for the duration of one invocation."""
    token = FILE_HANDLER.set(handler)
    try:
        yield
    finally:
        FILE_HANDLER.reset(token)


def build_send_file_tool(backend: BackendProtocol, *, max_bytes: int) -> BaseTool:
    """Build the `send_file` tool over the agent's filesystem backend.

    Args:
        backend: Backend the agent's file tools use, so paths mean the same thing.
        max_bytes: Largest file the tool will read and send.

    Returns:
        A tool that sends one file to the chat that started the turn.
    """

    @tool
    async def send_file(path: str, caption: str | None = None) -> str:
        """Send a file to the user in the chat that started this turn.

        Use this to deliver a file you created or found, such as a report, chart,
        image, or data export. Any file type works. Your reply text is still sent
        separately, so do not repeat the file's contents there.

        Args:
            path: Path of the file, as you would pass it to `read_file`.
            caption: Optional short text shown with the file.
        """
        return await _send(backend, path, caption, max_bytes)

    return send_file


async def _send(backend: BackendProtocol, path: str, caption: str | None, max_bytes: int) -> str:
    if _IN_SUBAGENT.get():
        return "Files cannot be sent from a subagent; return the path in your report instead."
    handler = FILE_HANDLER.get()
    if handler is None:
        return "No chat can receive files in this run, so nothing was sent."
    data = await _read(backend, path, max_bytes)
    if isinstance(data, str):
        return data
    name = display_filename(PurePosixPath(path).name)
    result = await handler(data, name, caption or None)
    if result.success:
        return f"Sent {name}."
    # Channel errors can carry transport details, so they go to the logs, not the model.
    return f"Could not send {name} to the chat."


def display_filename(name: str) -> str:
    """Return the name a recipient sees for a sent file.

    Spaces, punctuation, and non-ASCII letters are kept, since each file is staged
    alone in a fresh private directory. Only what could leave that directory or
    misrepresent the name is replaced: path separators, names made only of dots,
    and control or formatting characters such as right-to-left overrides.

    Args:
        name: Untrusted file name from the agent.

    Returns:
        A non-empty single path component of at most 200 UTF-8 bytes.
    """
    kept = "".join(
        "_" if char in "/\\" or unicodedata.category(char).startswith("C") else char
        for char in name
    )
    cleaned = kept.strip()
    if not cleaned.strip("."):
        return "file"
    return _truncate_name(cleaned)


def _truncate_name(name: str) -> str:
    if len(name.encode()) <= _MAX_NAME_BYTES:
        return name
    path = PurePosixPath(name)
    suffix = path.suffix if len(path.suffix.encode()) <= _MAX_NAME_BYTES // 4 else ""
    budget = _MAX_NAME_BYTES - len(suffix.encode())
    stem = name[: len(name) - len(suffix)].encode()[:budget].decode(errors="ignore")
    return stem + suffix


async def _read(backend: BackendProtocol, path: str, max_bytes: int) -> bytes | str:
    try:
        responses = await backend.adownload_files([path])
    except Exception as error:  # noqa: BLE001  # any backend failure must come back as tool text
        return f"Could not read {path}: {error}"
    response = responses[0] if len(responses) == 1 else None
    if response is None or response.error is not None or response.content is None:
        reason = response.error if response is not None else "no result"
        return f"Could not read {path}: {reason}."
    if len(response.content) > max_bytes:
        size = len(response.content)
        return f"{path} is too large to send: {size} bytes exceeds the {max_bytes}-byte limit."
    return response.content


async def send_file_to_channel(
    channel: ChannelAdapter,
    conversation_id: str,
    file: tuple[bytes, str, str | None],
    *,
    staging_root: Path,
) -> SendResult:
    """Stage file bytes privately, upload them to a chat, and remove the copy.

    Args:
        channel: Adapter for the chat that started the turn.
        conversation_id: Chat to send to.
        file: File bytes, display name, and optional caption.
        staging_root: Directory the adapter accepts outbound files from.

    Returns:
        The channel's send result, or a failed result when staging fails.
    """
    data, filename, caption = file
    try:
        directory = await asyncio.to_thread(_staging_directory, staging_root)
    except OSError as error:
        return SendResult(success=False, error=f"could not stage the file: {error}")
    try:
        return await _stage_and_send(channel, conversation_id, directory, data, filename, caption)
    finally:
        await asyncio.to_thread(shutil.rmtree, directory, ignore_errors=True)


async def _stage_and_send(  # noqa: PLR0913  # one upload's full description
    channel: ChannelAdapter,
    conversation_id: str,
    directory: Path,
    data: bytes,
    filename: str,
    caption: str | None,
) -> SendResult:
    try:
        path = await asyncio.to_thread(_write_private, directory / display_filename(filename), data)
    except OSError as error:
        return SendResult(success=False, error=f"could not stage the file: {error}")
    media = ChannelMedia(path=path, media_type=media_type_for_path(path), caption=caption)
    return await send_with_retry(lambda: channel.send_media(conversation_id, media))


def _staging_directory(root: Path) -> Path:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    # A fresh directory per send keeps the user-facing file name without collisions.
    return Path(tempfile.mkdtemp(prefix="send-", dir=root))


def _write_private(path: Path, data: bytes) -> Path:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with os.fdopen(os.open(path, flags, 0o600), "wb") as stream:
        stream.write(data)
    return path
