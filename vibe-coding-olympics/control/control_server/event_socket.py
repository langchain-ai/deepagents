"""Shared helpers for Deep Agents CLI external-event sockets."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

SOCKET_TIMEOUT_SECS = 2.0


async def send_socket_event(
    socket_path: Path,
    *,
    kind: str,
    payload: str,
    correlation_prefix: str,
) -> None:
    """Send one JSON-lines external event to a player CLI socket.

    Args:
        socket_path: Unix-domain socket path exported by the player CLI.
        kind: External event kind accepted by the CLI event bus.
        payload: External event payload.
        correlation_prefix: Prefix used for the ACK correlation id.

    Raises:
        OSError: If the socket cannot be reached.
        TimeoutError: If the CLI does not acknowledge the event in time.
        RuntimeError: If the CLI returns a negative acknowledgement.
        json.JSONDecodeError: If the CLI returns malformed JSON.
    """
    correlation_id = f"{correlation_prefix}-{uuid4().hex}"
    envelope = {
        "kind": kind,
        "payload": payload,
        "correlation_id": correlation_id,
    }
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(str(socket_path)),
        timeout=SOCKET_TIMEOUT_SECS,
    )
    try:
        writer.write(json.dumps(envelope).encode("utf-8") + b"\n")
        await asyncio.wait_for(writer.drain(), timeout=SOCKET_TIMEOUT_SECS)
        line = await asyncio.wait_for(reader.readline(), timeout=SOCKET_TIMEOUT_SECS)
    finally:
        writer.close()
        await writer.wait_closed()

    if not line:
        msg = f"External event socket {socket_path} closed without an ACK"
        raise RuntimeError(msg)
    response = json.loads(line)
    if not isinstance(response, dict):
        msg = f"External event socket {socket_path} returned a non-object ACK"
        raise RuntimeError(msg)
    if response.get("ok") is not True:
        error = response.get("error", "unknown error")
        msg = f"External event socket {socket_path} rejected {kind}: {error}"
        raise RuntimeError(msg)
