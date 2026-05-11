from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

from control_server import player_relay


class TestPlayerRelay(unittest.IsolatedAsyncioTestCase):
    async def test_command_forwards_to_event_socket(self) -> None:
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
        app = player_relay.create_app()
        transport = httpx.ASGITransport(app=app)
        try:
            with patch.dict(
                "os.environ",
                {
                    "VIBE_EVENT_SOCKET": str(path),
                    "VIBE_PLAYER_TOKEN": "test-token",
                },
                clear=True,
            ), patch("control_server.deepagents_config.clear_recent_model") as clear:
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://relay",
                ) as client:
                    response = await client.post(
                        "/command",
                        json={"kind": "signal", "payload": "times-up"},
                        headers={"authorization": "Bearer test-token"},
                    )
        finally:
            server.close()
            await server.wait_closed()
            tmp_dir.cleanup()

        self.assertEqual(response.status_code, 200)
        clear.assert_not_called()
        self.assertEqual(response.json(), {"ok": True})
        self.assertEqual(received[0]["kind"], "signal")
        self.assertEqual(received[0]["payload"], "times-up")
        self.assertTrue(received[0]["correlation_id"].startswith("vibe-lan-"))

    async def test_force_clear_clears_recent_model_before_forwarding(self) -> None:
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
        app = player_relay.create_app()
        transport = httpx.ASGITransport(app=app)
        try:
            with (
                patch.dict(
                    "os.environ",
                    {
                        "VIBE_EVENT_SOCKET": str(path),
                        "VIBE_PLAYER_TOKEN": "test-token",
                    },
                    clear=True,
                ),
                patch(
                    "control_server.deepagents_config.clear_recent_model",
                    return_value=True,
                ) as clear,
            ):
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://relay",
                ) as client:
                    response = await client.post(
                        "/command",
                        json={"kind": "signal", "payload": "force-clear"},
                        headers={"authorization": "Bearer test-token"},
                    )
        finally:
            server.close()
            await server.wait_closed()
            tmp_dir.cleanup()

        self.assertEqual(response.status_code, 200)
        clear.assert_called_once_with()
        self.assertEqual(received[0]["payload"], "force-clear")

    async def test_command_requires_valid_bearer_token(self) -> None:
        app = player_relay.create_app()
        transport = httpx.ASGITransport(app=app)

        with patch.dict(
            "os.environ",
            {
                "VIBE_EVENT_SOCKET": "/tmp/deepagents-vibe-3001.sock",
                "VIBE_PLAYER_TOKEN": "test-token",
            },
            clear=True,
        ):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://relay",
            ) as client:
                response = await client.post(
                    "/command",
                    json={"kind": "signal", "payload": "times-up"},
                    headers={"authorization": "Bearer wrong-token"},
                )

        self.assertEqual(response.status_code, 401)

    async def test_command_fails_closed_when_token_is_not_configured(self) -> None:
        app = player_relay.create_app()
        transport = httpx.ASGITransport(app=app)

        with patch.dict(
            "os.environ",
            {"VIBE_EVENT_SOCKET": "/tmp/deepagents-vibe-3001.sock"},
            clear=True,
        ):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://relay",
            ) as client:
                response = await client.post(
                    "/command",
                    json={"kind": "signal", "payload": "times-up"},
                    headers={"authorization": "Bearer test-token"},
                )

        self.assertEqual(response.status_code, 503)

    async def test_missing_socket_returns_503(self) -> None:
        app = player_relay.create_app()
        transport = httpx.ASGITransport(app=app)

        with patch.dict(
            "os.environ",
            {
                "VIBE_EVENT_SOCKET": "/tmp/does-not-exist.sock",
                "VIBE_PLAYER_TOKEN": "test-token",
            },
            clear=True,
        ):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://relay",
            ) as client:
                response = await client.post(
                    "/command",
                    json={"kind": "signal", "payload": "times-up"},
                    headers={"authorization": "Bearer test-token"},
                )

        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
