"""Derive each player's deployed-site URL from operator-provided env config.

Convention: every player's vibe-coded site is served on the same port that
`play.sh` was invoked with (3001/3002 by default) on the player laptop's LAN
IP. The LAN IP is already known to the controller because the operator has
to configure `VIBE_PLAYER_<PORT>_RELAY` (e.g. `http://192.168.1.21:9771`)
for player dispatch to work at all. We reuse that host.

An explicit override is supported via `VIBE_PLAYER_<PORT>_SITE_URL` for
the rare case where a player deploys to a remote URL instead of serving
locally.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

RELAY_ENV_PREFIX = "VIBE_PLAYER_"
RELAY_ENV_SUFFIX = "_RELAY"
SITE_URL_ENV_SUFFIX = "_SITE_URL"


def _override_for(port: str) -> str | None:
    """Return an explicit site URL override for `port`, if configured."""
    raw = os.environ.get(f"{RELAY_ENV_PREFIX}{port}{SITE_URL_ENV_SUFFIX}", "")
    url = raw.strip().rstrip("/")
    return url or None


def _host_from_relay(port: str) -> str | None:
    """Extract the LAN host portion of the configured relay URL for `port`."""
    raw = os.environ.get(f"{RELAY_ENV_PREFIX}{port}{RELAY_ENV_SUFFIX}", "")
    relay = raw.strip()
    if not relay:
        return None
    parsed = urlparse(relay)
    host = parsed.hostname
    if not host:
        logger.warning(
            "Could not parse host from relay URL %r for port %s", relay, port
        )
        return None
    return host


def site_url_for(port: str) -> str | None:
    """Return the controller-reachable site URL for one player port.

    Lookup order:
        1. `VIBE_PLAYER_<PORT>_SITE_URL` — explicit override, used as-is.
        2. `VIBE_PLAYER_<PORT>_RELAY` host + the player port itself.

    Args:
        port: The `play.sh` port assigned to the player (e.g. `"3001"`).

    Returns:
        Site URL with no trailing slash, or `None` if neither env var
        is configured for this port.
    """
    override = _override_for(port)
    if override:
        return override
    host = _host_from_relay(port)
    if host is None:
        return None
    return f"http://{host}:{port}"


def site_urls(ports: list[str]) -> dict[str, str]:
    """Return `port -> site URL` for every port that has one configured."""
    out: dict[str, str] = {}
    for port in ports:
        url = site_url_for(port)
        if url:
            out[port] = url
    return out
