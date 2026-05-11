"""CLI hook adapter for reporting player readiness to `vibe-control`."""

from __future__ import annotations

import json
import logging
import os
import sys

import httpx

logger = logging.getLogger(__name__)


def main() -> None:
    """Read a Deep Agents hook payload from stdin and POST it to control."""
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        logger.warning("Ignoring malformed player hook payload: %s", exc)
        return

    port = os.environ.get("VIBE_PORT")
    if not port:
        logger.warning("Ignoring player hook payload without VIBE_PORT")
        return

    api = os.environ.get("VIBE_CONTROL_API", "http://localhost:8766").rstrip("/")
    event = payload.get("event")
    if event == "user.name.set":
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            return
        path = "/api/players/ready"
        body = {"port": port, "name": name.strip()}
    elif event == "competition.player.ready":
        path = "/api/players/model-ready"
        body = {"port": port}
    else:
        return

    try:
        response = httpx.post(
            f"{api}{path}",
            json=body,
            timeout=2.0,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        logger.warning("Failed to POST player hook event %s to %s", event, api)
        return


if __name__ == "__main__":
    main()
