"""CLI hook adapter for reporting player readiness to `vibe-control`."""

from __future__ import annotations

import json
import os
import sys

import httpx


def main() -> None:
    """Read a Deep Agents hook payload from stdin and POST it to control."""
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return

    port = os.environ.get("VIBE_PORT")
    if not port:
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
        httpx.post(
            f"{api}{path}",
            json=body,
            timeout=2.0,
        )
    except httpx.HTTPError:
        return


if __name__ == "__main__":
    main()
