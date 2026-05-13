"""Best-effort browser preview refresh helpers."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)

OPEN_ON_CLEAR_ENV = "VIBE_OPEN_ON_CLEAR"
PORT_ENV = "VIBE_PORT"
_READY_PROBES = 2
_READY_PROBE_INTERVAL_SECS = 0.2
_READY_TIMEOUT_SECS = 4.0
_REFRESH_DELAY_SECS = 0.8


def _enabled() -> bool:
    """Return whether clear should reopen the local browser preview."""
    return os.environ.get(OPEN_ON_CLEAR_ENV, "").strip() == "1"


def _preview_url(port: str) -> str | None:
    """Return the localhost Vite URL for a port string, if valid."""
    try:
        parsed = int(port)
    except ValueError:
        return None
    if not 0 < parsed < 65536:
        return None
    return f"http://localhost:{parsed}"


async def _probe_port(port: int) -> bool:
    """Return whether a TCP connection can be opened to localhost port."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port),
            timeout=0.5,
        )
    except OSError:
        return False
    except TimeoutError:
        return False
    writer.close()
    await writer.wait_closed()
    return True


async def _wait_for_ready_port(port: int) -> bool:
    """Wait until the local Vite port is accepting stable connections."""
    deadline = asyncio.get_running_loop().time() + _READY_TIMEOUT_SECS
    consecutive = 0
    while asyncio.get_running_loop().time() < deadline:
        if await _probe_port(port):
            consecutive += 1
            if consecutive >= _READY_PROBES:
                return True
        else:
            consecutive = 0
        await asyncio.sleep(_READY_PROBE_INTERVAL_SECS)
    return False


async def _open_url(url: str) -> None:
    """Ask macOS to open or refresh the browser tab for a URL."""
    opener = shutil.which("open")
    if opener is None:
        logger.debug("Skipping browser refresh because `open` is unavailable.")
        return
    proc = await asyncio.create_subprocess_exec(
        opener,
        url,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    await asyncio.wait_for(proc.wait(), timeout=2.0)


async def refresh_preview_when_ready(port: str) -> bool:
    """Reopen the local Vite preview once the server is reachable.

    Args:
        port: Vite development server port.

    Returns:
        `True` when a browser refresh was attempted, otherwise `False`.
    """
    url = _preview_url(port)
    if url is None:
        return False
    await asyncio.sleep(_REFRESH_DELAY_SECS)
    parsed = int(port)
    if not await _wait_for_ready_port(parsed):
        logger.warning("Vite preview on port %s was not reachable after clear.", port)
        return False
    try:
        await _open_url(url)
    except (OSError, TimeoutError) as exc:
        logger.warning("Could not refresh browser preview %s: %s", url, exc)
        return False
    return True


def schedule_preview_refresh(port: str | None) -> None:
    """Schedule a best-effort local browser refresh after clearing a round."""
    if not _enabled():
        return
    if port is None:
        return
    parsed = port.strip()
    if not parsed:
        return
    asyncio.create_task(refresh_preview_when_ready(parsed))


def schedule_preview_refresh_from_env() -> None:
    """Schedule a local browser refresh for the configured Vite port."""
    schedule_preview_refresh(os.environ.get(PORT_ENV, ""))
