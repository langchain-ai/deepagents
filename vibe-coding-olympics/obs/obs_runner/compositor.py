"""Thin wrapper around obs-websocket v5 for the MVP compositor surface.

Only two verbs are exposed: `set_scene` and `set_text`. Every other OBS
capability (browser-source URL, recording, source visibility) is
intentionally out of scope until the state machine + API are proven.

Missing scenes/sources are logged and skipped rather than raised. A
round in progress should not fail because the operator hasn't finished
building the OBS layout — the server log names every missing resource
so the gaps are visible and the transition still completes.
"""

from __future__ import annotations

import logging
from typing import Protocol

import obsws_python as obsws
from obsws_python.error import OBSSDKRequestError

logger = logging.getLogger(__name__)


class CompositorProtocol(Protocol):
    """Minimum compositor surface the state machine depends on.

    Kept as a `Protocol` so tests can pass a recording fake and the FSM
    can be exercised without a running OBS.
    """

    def set_scene(self, name: str) -> None:
        """Set the currently visible program scene."""

    def set_text(self, source: str, value: str) -> None:
        """Update the text content of an OBS text input."""


class ObsCompositor:
    """Connects to obs-websocket and exposes the MVP compositor verbs.

    Connection is lazy: `connect()` must be called before any command
    method. `close()` is idempotent. Instances are not thread-safe;
    FastAPI serializes writes through a single event loop.
    """

    def __init__(self, host: str, port: int, password: str) -> None:
        """Store connection params; do not open a socket yet.

        Args:
            host: obs-websocket hostname.
            port: obs-websocket port.
            password: obs-websocket password. Empty string disables auth.
        """
        self._host = host
        self._port = port
        self._password = password
        self._client: obsws.ReqClient | None = None

    def connect(self) -> None:
        """Open the obs-websocket connection.

        Raises:
            ConnectionError: If the socket cannot be established.
        """
        if self._client is not None:
            return
        try:
            self._client = obsws.ReqClient(
                host=self._host,
                port=self._port,
                password=self._password or None,
                timeout=5,
            )
        except Exception as exc:
            msg = f"OBS connect failed ({self._host}:{self._port}): {exc}"
            raise ConnectionError(msg) from exc

    def close(self) -> None:
        """Disconnect from obs-websocket. Safe to call multiple times."""
        if self._client is None:
            return
        try:
            self._client.disconnect()
        except Exception:
            pass
        self._client = None

    def set_scene(self, name: str) -> None:
        """Set the currently visible program scene.

        A missing scene is logged and skipped — the transition continues.

        Args:
            name: Exact OBS scene name.

        Raises:
            RuntimeError: If the compositor has not been connected.
        """
        client = self._require_client()
        try:
            client.set_current_program_scene(name)
        except OBSSDKRequestError as exc:
            logger.warning(
                "OBS rejected set_scene(%r): code=%s %s",
                name, exc.code, exc,
            )

    def set_text(self, source: str, value: str) -> None:
        """Update the text content of an OBS text input.

        Works for both `text_gdiplus_v2` (Windows) and `text_ft2_source_v2`
        (macOS/Linux) — both accept a `text` field in their input settings.
        A missing input is logged and skipped — the transition continues.

        Args:
            source: Exact OBS input name of the text source.
            value: New text to render.

        Raises:
            RuntimeError: If the compositor has not been connected.
        """
        client = self._require_client()
        try:
            client.set_input_settings(
                name=source,
                settings={"text": value},
                overlay=True,
            )
        except OBSSDKRequestError as exc:
            logger.warning(
                "OBS rejected set_text(%r): code=%s %s",
                source, exc.code, exc,
            )

    def _require_client(self) -> obsws.ReqClient:
        """Return the live client or raise if `connect()` was not called."""
        if self._client is None:
            msg = "ObsCompositor not connected; call connect() first."
            raise RuntimeError(msg)
        return self._client
