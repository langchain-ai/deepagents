from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from control_server import player_dispatch


class TestPlayerDispatch(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_prefers_relay_and_falls_back_to_local(self) -> None:
        relay = AsyncMock(return_value=True)
        local = AsyncMock(return_value=["3002"])

        with (
            patch.dict(
                "os.environ",
                {
                    "VIBE_PLAYER_3001_RELAY": "http://player-1:9771",
                    "VIBE_PLAYER_TOKEN": "test-token",
                },
                clear=True,
            ),
            patch("control_server.player_dispatch._send_relay_event", relay),
            patch("control_server.iterm_ctrl.send_prompt_to_players", local),
        ):
            sent = await player_dispatch.send_prompt_to_players(
                ["3001", "3002"],
                "taco truck",
            )

        self.assertEqual(sent, ["3001", "3002"])
        relay.assert_awaited_once_with(
            "3001",
            "http://player-1:9771",
            kind="command",
            payload="/skill:web-vibe Prompt: taco truck",
        )
        local.assert_awaited_once_with(["3002"], "taco truck")

    async def test_all_targets_relays_and_unmapped_local_sessions(self) -> None:
        relay = AsyncMock(return_value=True)
        local = AsyncMock(return_value=["3002"])

        with (
            patch.dict(
                "os.environ",
                {
                    "VIBE_PLAYER_3001_RELAY": "http://player-1:9771",
                    "VIBE_PLAYER_TOKEN": "test-token",
                },
                clear=True,
            ),
            patch("control_server.player_dispatch._send_relay_event", relay),
            patch(
                "control_server.iterm_ctrl.list_players",
                new=AsyncMock(return_value=["3001", "3002"]),
            ),
            patch("control_server.iterm_ctrl.times_up_players", local),
        ):
            sent = await player_dispatch.times_up_players(None)

        self.assertEqual(sent, ["3001", "3002"])
        relay.assert_awaited_once_with(
            "3001",
            "http://player-1:9771",
            kind="signal",
            payload="times-up",
        )
        local.assert_awaited_once_with(["3002"])

    async def test_no_relay_mapping_uses_existing_iterm_all_path(self) -> None:
        local = AsyncMock(return_value=["3001", "3002"])

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("control_server.iterm_ctrl.clear_players", local),
        ):
            cleared = await player_dispatch.clear_players(None)

        self.assertEqual(cleared, ["3001", "3002"])
        local.assert_awaited_once_with(None)

    async def test_relay_without_token_is_not_reported_sent(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {"VIBE_PLAYER_3001_RELAY": "http://player-1:9771"},
                clear=True,
            ),
            patch(
                "control_server.iterm_ctrl.times_up_players",
                new=AsyncMock(),
            ) as local,
            self.assertLogs("control_server.player_dispatch", level="WARNING"),
        ):
            sent = await player_dispatch.times_up_players(["3001"])

        self.assertEqual(sent, [])
        local.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
