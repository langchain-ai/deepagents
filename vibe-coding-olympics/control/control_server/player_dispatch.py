"""Dispatch player commands over LAN relays with local iTerm2 fallback."""

from __future__ import annotations

import logging
import os

import httpx

from control_server import iterm_ctrl

logger = logging.getLogger(__name__)

RELAY_ENV_PREFIX = "VIBE_PLAYER_"
RELAY_ENV_SUFFIX = "_RELAY"
PLAYER_TOKEN_ENV = "VIBE_PLAYER_TOKEN"


def configured_relays() -> dict[str, str]:
    """Return player-port to relay-URL mappings from environment variables."""
    relays: dict[str, str] = {}
    for key, value in os.environ.items():
        if not key.startswith(RELAY_ENV_PREFIX) or not key.endswith(RELAY_ENV_SUFFIX):
            continue
        port = key[len(RELAY_ENV_PREFIX) : -len(RELAY_ENV_SUFFIX)]
        url = value.strip().rstrip("/")
        if port and url:
            relays[port] = url
    return relays


def _target_relay_ports(
    ports: list[str] | None,
    relays: dict[str, str],
) -> list[str]:
    """Return target ports that should use LAN relays."""
    if ports is None:
        return sorted(relays)
    return [port for port in ports if port in relays]


async def _local_fallback_ports(
    ports: list[str] | None,
    relayed: list[str],
) -> list[str] | None:
    """Return local iTerm2 target ports after relay targets are removed."""
    if ports is not None:
        local = [port for port in ports if port not in relayed]
        return local or []
    if not relayed:
        return None
    local_players = await iterm_ctrl.list_players()
    return [port for port in local_players if port not in relayed]


async def _send_relay_event(
    port: str,
    relay_url: str,
    *,
    kind: str,
    payload: str,
) -> bool:
    """POST one command to a player relay."""
    token = os.environ.get(PLAYER_TOKEN_ENV, "").strip()
    if not token:
        logger.warning(
            "Skipping LAN relay for player %s because %s is not configured.",
            port,
            PLAYER_TOKEN_ENV,
        )
        return False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{relay_url}/command",
                json={"kind": kind, "payload": payload},
                headers={"authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("LAN relay command failed for player %s: %s", port, exc)
        return False
    return True


async def _send_to_relays(
    ports: list[str],
    relays: dict[str, str],
    *,
    kind: str,
    payload: str,
) -> list[str]:
    """Send a command to every selected relay and return successful ports."""
    sent: list[str] = []
    for port in ports:
        if await _send_relay_event(port, relays[port], kind=kind, payload=payload):
            sent.append(port)
    return sent


async def send_prompt_to_players(ports: list[str] | None, prompt: str) -> list[str]:
    """Inject a prompt into targeted player CLIs."""
    relays = configured_relays()
    relay_ports = _target_relay_ports(ports, relays)
    message = f"/skill:web-vibe Prompt: {prompt}"
    sent = await _send_to_relays(
        relay_ports,
        relays,
        kind="command",
        payload=message,
    )
    local_ports = await _local_fallback_ports(ports, relay_ports)
    if local_ports is None or local_ports:
        sent.extend(await iterm_ctrl.send_prompt_to_players(local_ports, prompt))
    return sent


async def times_up_players(ports: list[str] | None) -> list[str]:
    """Send a `times-up` signal to targeted player CLIs."""
    relays = configured_relays()
    relay_ports = _target_relay_ports(ports, relays)
    sent = await _send_to_relays(
        relay_ports,
        relays,
        kind="signal",
        payload="times-up",
    )
    local_ports = await _local_fallback_ports(ports, relay_ports)
    if local_ports is None or local_ports:
        sent.extend(await iterm_ctrl.times_up_players(local_ports))
    return sent


async def clear_players(ports: list[str] | None) -> list[str]:
    """Send a `force-clear` signal to targeted player CLIs."""
    relays = configured_relays()
    relay_ports = _target_relay_ports(ports, relays)
    cleared = await _send_to_relays(
        relay_ports,
        relays,
        kind="signal",
        payload="force-clear",
    )
    local_ports = await _local_fallback_ports(ports, relay_ports)
    if local_ports is None or local_ports:
        cleared.extend(await iterm_ctrl.clear_players(local_ports))
    return cleared


async def players_ready(ports: list[str] | None) -> list[str]:
    """Notify targeted player CLIs that both players are ready to start."""
    relays = configured_relays()
    relay_ports = _target_relay_ports(ports, relays)
    sent = await _send_to_relays(
        relay_ports,
        relays,
        kind="signal",
        payload="players-ready",
    )
    local_ports = await _local_fallback_ports(ports, relay_ports)
    if local_ports is None or local_ports:
        sent.extend(await iterm_ctrl.players_ready(local_ports))
    return sent


async def reset_players(ports: list[str] | None) -> list[str]:
    """Quit and relaunch local player CLIs.

    LAN relays intentionally do not own the iTerm2 process, so hard reset keeps
    using the local iTerm2 path. Use `clear_players` for remote round resets.
    """
    relays = configured_relays()
    relay_ports = _target_relay_ports(ports, relays)
    for port in relay_ports:
        logger.warning("Skipping hard reset for LAN relay player %s.", port)
    local_ports = await _local_fallback_ports(ports, relay_ports)
    if local_ports is None or local_ports:
        return await iterm_ctrl.reset_players(local_ports)
    return []
