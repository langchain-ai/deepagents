from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from control_server import hook


class TestPlayerHook(unittest.TestCase):
    def test_user_name_set_posts_ready_name(self) -> None:
        payload = {"event": "user.name.set", "name": "Alice"}

        with (
            patch.dict(
                "os.environ",
                {"VIBE_PORT": "3001", "VIBE_CONTROL_API": "http://control"},
            ),
            patch("sys.stdin", io.StringIO(json.dumps(payload))),
            patch("control_server.hook.httpx.post") as post,
        ):
            hook.main()

        post.assert_called_once_with(
            "http://control/api/players/ready",
            json={"port": "3001", "name": "Alice"},
            timeout=2.0,
        )

    def test_competition_player_ready_posts_model_ready(self) -> None:
        payload = {"event": "competition.player.ready"}

        with (
            patch.dict(
                "os.environ",
                {"VIBE_PORT": "3001", "VIBE_CONTROL_API": "http://control"},
            ),
            patch("sys.stdin", io.StringIO(json.dumps(payload))),
            patch("control_server.hook.httpx.post") as post,
        ):
            hook.main()

        post.assert_called_once_with(
            "http://control/api/players/model-ready",
            json={"port": "3001"},
            timeout=2.0,
        )


if __name__ == "__main__":
    unittest.main()
