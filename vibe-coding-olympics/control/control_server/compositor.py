"""Async HTTP compositor: drives the OBS runner over LAN.

The FSM in `state_machine` calls `set_scene` / `set_text` on entry to
each phase. In this control plane those calls fan out as one-shot HTTP
POSTs to the OBS runner, which is now a thin shim over obs-websocket.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx


class CompositorProtocol(Protocol):
    """Minimum compositor surface the state machine depends on."""

    async def set_scene(self, name: str) -> None:
        """Set the currently visible program scene."""

    async def set_text(self, source: str, value: str) -> None:
        """Update the text content of an OBS text input."""


class RemoteCompositor:
    """HTTP client that drives an OBS runner over the LAN."""

    def __init__(self, base_url: str, *, timeout_secs: float = 5.0) -> None:
        """Store the runner base URL; do not open a socket yet.

        Args:
            base_url: Root URL of the OBS runner, e.g. `http://host:8765`.
            timeout_secs: Per-request HTTP timeout.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout_secs = timeout_secs

    async def set_scene(self, name: str) -> None:
        """Switch the OBS program scene via the runner."""
        await self._post("/scene", {"name": name})

    async def set_text(self, source: str, value: str) -> None:
        """Update an OBS text-source value via the runner."""
        await self._post("/text", {"source": source, "value": value})

    async def _post(self, path: str, body: dict[str, Any]) -> None:
        """POST `body` to `{base_url}{path}`, raising `ConnectionError` on failure."""
        url = f"{self._base_url}{path}"
        async with httpx.AsyncClient(timeout=self._timeout_secs) as client:
            try:
                response = await client.post(url, json=body)
            except httpx.HTTPError as exc:
                msg = f"OBS runner at {self._base_url} unreachable: {exc}"
                raise ConnectionError(msg) from exc
        if response.status_code >= 400:
            try:
                detail: Any = response.json()
                if isinstance(detail, dict) and "detail" in detail:
                    detail = detail["detail"]
            except ValueError:
                detail = response.text
            msg = f"OBS runner rejected {path}: {detail}"
            raise ConnectionError(msg)
