from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from control_server import iterm_ctrl


class FakeSession:
    def __init__(self, variables: dict[str, str]) -> None:
        self.variables = variables

    async def async_get_variable(self, name: str) -> str | None:
        return self.variables.get(name)


class TestSocketClear(unittest.IsolatedAsyncioTestCase):
    async def test_send_force_clear_writes_signal_envelope(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        path = Path(tmp_dir.name) / "events.sock"
        received: list[dict[str, Any]] = []

        async def handle_client(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            received.append(json.loads(await reader.readline()))
            writer.write(b'{"ok":true}\n')
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_unix_server(handle_client, path=str(path))
        try:
            await iterm_ctrl._send_force_clear(path)
        finally:
            server.close()
            await server.wait_closed()
            tmp_dir.cleanup()

        self.assertEqual(received[0]["kind"], "signal")
        self.assertEqual(received[0]["payload"], "force-clear")
        self.assertTrue(received[0]["correlation_id"].startswith("vibe-clear-"))

    async def test_clear_players_uses_session_socket_variable(self) -> None:
        tmp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        path = Path(tmp_dir.name) / "events.sock"
        received: list[dict[str, Any]] = []

        async def handle_client(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            received.append(json.loads(await reader.readline()))
            writer.write(b'{"ok":true}\n')
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        async def matching_sessions(
            ports: list[str] | None,
        ) -> list[tuple[str, FakeSession]]:
            self.assertEqual(ports, ["3001"])
            session = FakeSession({iterm_ctrl.SOCKET_VARIABLE: str(path)})
            return [("3001", session)]

        original = iterm_ctrl.matching_sessions
        iterm_ctrl.matching_sessions = matching_sessions  # type: ignore[assignment]
        server = await asyncio.start_unix_server(handle_client, path=str(path))
        try:
            cleared = await iterm_ctrl.clear_players(["3001"])
        finally:
            iterm_ctrl.matching_sessions = original
            server.close()
            await server.wait_closed()
            tmp_dir.cleanup()

        self.assertEqual(cleared, ["3001"])
        self.assertEqual(received[0]["payload"], "force-clear")


if __name__ == "__main__":
    unittest.main()
