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


class FakeTab:
    def __init__(self, sessions: list[FakeSession]) -> None:
        self.sessions = sessions


class FakeWindow:
    def __init__(self, tabs: list[FakeTab]) -> None:
        self.tabs = tabs


class FakeApp:
    def __init__(self, windows: list[FakeWindow]) -> None:
        self.windows = windows


class TestSocketClear(unittest.IsolatedAsyncioTestCase):
    def tearDown(self) -> None:
        iterm_ctrl._connection = None
        iterm_ctrl._connection_lock = None

    async def test_matching_sessions_reuses_iterm_connection(self) -> None:
        created: list[object] = []
        connection = object()
        app = FakeApp(
            [
                FakeWindow(
                    [
                        FakeTab(
                            [
                                FakeSession(
                                    {"user.vibe_player": "3001"}
                                )
                            ]
                        )
                    ]
                )
            ]
        )

        async def async_create() -> object:
            created.append(connection)
            return connection

        async def async_get_app(conn: object) -> FakeApp:
            self.assertIs(conn, connection)
            return app

        original_create = iterm_ctrl.iterm2.Connection.async_create
        original_get_app = iterm_ctrl.iterm2.async_get_app
        iterm_ctrl.iterm2.Connection.async_create = async_create  # type: ignore[assignment]
        iterm_ctrl.iterm2.async_get_app = async_get_app  # type: ignore[assignment]
        try:
            first = await iterm_ctrl.matching_sessions(None)
            second = await iterm_ctrl.matching_sessions(None)
        finally:
            iterm_ctrl.iterm2.Connection.async_create = original_create  # type: ignore[assignment]
            iterm_ctrl.iterm2.async_get_app = original_get_app  # type: ignore[assignment]

        self.assertEqual([port for port, _ in first], ["3001"])
        self.assertEqual([port for port, _ in second], ["3001"])
        self.assertEqual(created, [connection])

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

    async def test_send_prompt_invokes_web_vibe_skill(self) -> None:
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
            self.assertIsNone(ports)
            session = FakeSession({iterm_ctrl.SOCKET_VARIABLE: str(path)})
            return [("3001", session)]

        original = iterm_ctrl.matching_sessions
        iterm_ctrl.matching_sessions = matching_sessions  # type: ignore[assignment]
        server = await asyncio.start_unix_server(handle_client, path=str(path))
        try:
            sent = await iterm_ctrl.send_prompt_to_players(
                None, "a website for a taco truck"
            )
        finally:
            iterm_ctrl.matching_sessions = original
            server.close()
            await server.wait_closed()
            tmp_dir.cleanup()

        self.assertEqual(sent, ["3001"])
        self.assertEqual(received[0]["kind"], "command")
        self.assertEqual(
            received[0]["payload"],
            "/skill:web-vibe Prompt: a website for a taco truck",
        )
        self.assertTrue(received[0]["correlation_id"].startswith("vibe-prompt-"))


if __name__ == "__main__":
    unittest.main()
