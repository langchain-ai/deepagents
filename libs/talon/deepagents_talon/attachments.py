"""Prepare downloaded documents for remote file tools without prompt injection."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import stat
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from uuid import uuid4

from deepagents_talon.channels.base import max_media_bytes_from_env
from deepagents_talon.media import (
    _document_paths,
    _media_mime_types,
    _media_paths,
    resolve_bounded_media_path,
)

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

    from deepagents_talon.config import TalonConfig
    from deepagents_talon.interfaces import ChannelMessage

logger = logging.getLogger(__name__)


async def stage_documents(
    message: ChannelMessage, *, backend: BackendProtocol, working_dir: str, config: TalonConfig
) -> ChannelMessage:
    """Copy document bytes into the active sandbox and replace host path aliases."""
    metadata = dict(message.metadata)
    paths = _media_paths(metadata)
    documents = _document_paths(metadata)
    mimes = _media_mime_types(metadata)
    staged: list[str] = []
    staged_mimes: list[str] = []
    changed = False
    for index, path in enumerate(paths):
        mime = mimes[index] if len(mimes) == len(paths) else "application/octet-stream"
        if path not in documents:
            staged.append(str(path))
            staged_mimes.append(mime)
            continue
        changed = True
        destination = await _stage_document(path, backend, working_dir, config, metadata)
        if destination is None:
            metadata["media_error"] = "attachment unavailable in sandbox"
        else:
            staged.append(destination)
            staged_mimes.append(mime)
    if not changed:
        return message
    for key in ("media_path", "media_paths", "media_urls", "voice_path"):
        metadata.pop(key, None)
    metadata.update(media_paths=staged, media_mime_types=staged_mimes, has_media=bool(staged))
    return replace(message, metadata=metadata)


async def _stage_document(
    path: Path,
    backend: BackendProtocol,
    working_dir: str,
    config: TalonConfig,
    metadata: dict[str, object],
) -> str | None:
    suffix = path.suffix if re.fullmatch(r"\.[a-zA-Z0-9]{1,10}", path.suffix) else ".bin"
    destination = str(
        PurePosixPath(working_dir) / ".talon" / "attachments" / (uuid4().hex + suffix)
    )
    try:
        data = await asyncio.to_thread(_document_bytes, path, config, metadata)
        results = await backend.aupload_files([(destination, data)])
    except Exception:
        logger.exception("Could not stage inbound document in sandbox")
        return None
    if len(results) != 1 or results[0].error is not None or results[0].path != destination:
        logger.warning("Sandbox rejected inbound document upload")
        return None
    return destination


def _document_bytes(path: Path, config: TalonConfig, metadata: dict[str, object]) -> bytes:
    roots = [config.inbound_media_dir]
    provider = metadata.get("provider")
    if provider in {"slack", "telegram", "whatsapp", "discord"}:
        override = config.env.get(f"DEEPAGENTS_TALON_{str(provider).upper()}_MEDIA_DIR")
        if override:
            roots.insert(0, Path(override))
    resolved = None
    for root in roots:
        try:
            resolved = resolve_bounded_media_path(path.absolute(), root)
            break
        except ValueError:
            continue
    if resolved is None:
        msg = "attachment is outside the inbound media roots"
        raise ValueError(msg)
    limit = max_media_bytes_from_env(config.env)
    with os.fdopen(os.open(resolved, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            msg = "attachment is not a bounded regular file"
            raise ValueError(msg)
        data = stream.read(limit + 1)
    if len(data) > limit:
        msg = "attachment exceeds the size cap"
        raise ValueError(msg)
    return data
