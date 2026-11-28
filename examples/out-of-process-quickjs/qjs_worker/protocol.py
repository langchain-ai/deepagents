"""Length-prefixed JSON frames: 4-byte big-endian length, then UTF-8 JSON."""

from __future__ import annotations

import asyncio
import json
import struct
import sys
from typing import IO

_LEN = struct.Struct(">I")
MAX_FRAME = 16 * 1024 * 1024


def encode(msg: dict) -> bytes:
    body = json.dumps(msg, separators=(",", ":")).encode()
    return _LEN.pack(len(body)) + body


def write_sync(out: IO[bytes], msg: dict) -> None:
    out.write(encode(msg))
    out.flush()


async def read(reader: asyncio.StreamReader) -> dict | None:
    """Next frame, or None at EOF."""
    try:
        header = await reader.readexactly(_LEN.size)
    except asyncio.IncompleteReadError:
        return None
    (size,) = _LEN.unpack(header)
    if size > MAX_FRAME:
        raise ValueError(f"frame of {size} bytes exceeds {MAX_FRAME}")
    return json.loads(await reader.readexactly(size))


async def stdin_reader() -> asyncio.StreamReader:
    reader = asyncio.StreamReader(limit=MAX_FRAME)
    loop = asyncio.get_running_loop()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    return reader
