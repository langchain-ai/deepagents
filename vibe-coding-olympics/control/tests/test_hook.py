from __future__ import annotations

import io
import json
import unittest
from unittest.mock import Mock, patch

import httpx

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
            post.return_value = Mock()
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
            post.return_value = Mock()
            hook.main()

        post.assert_called_once_with(
            "http://control/api/players/model-ready",
            json={"port": "3001"},
            timeout=2.0,
        )

    def test_malformed_json_logs_warning(self) -> None:
        with (
            patch("sys.stdin", io.StringIO("{bad")),
            self.assertLogs("control_server.hook", level="WARNING") as logs,
        ):
            hook.main()

        self.assertIn("Ignoring malformed player hook payload", logs.output[0])

    def test_missing_port_logs_warning(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.stdin", io.StringIO(json.dumps({"event": "user.name.set"}))),
            self.assertLogs("control_server.hook", level="WARNING") as logs,
        ):
            hook.main()

        self.assertIn("without VIBE_PORT", logs.output[0])

    def test_post_failure_logs_warning(self) -> None:
        payload = {"event": "competition.player.ready"}

        with (
            patch.dict(
                "os.environ",
                {"VIBE_PORT": "3001", "VIBE_CONTROL_API": "http://control"},
            ),
            patch("sys.stdin", io.StringIO(json.dumps(payload))),
            patch(
                "control_server.hook.httpx.post",
                side_effect=httpx.ConnectError("offline"),
            ),
            self.assertLogs("control_server.hook", level="WARNING") as logs,
        ):
            hook.main()

        self.assertIn("Failed to POST player hook event", logs.output[0])

    def test_error_response_logs_warning(self) -> None:
        payload = {"event": "competition.player.ready"}
        response = httpx.Response(500, request=httpx.Request("POST", "http://control"))

        with (
            patch.dict(
                "os.environ",
                {"VIBE_PORT": "3001", "VIBE_CONTROL_API": "http://control"},
            ),
            patch("sys.stdin", io.StringIO(json.dumps(payload))),
            patch("control_server.hook.httpx.post", return_value=response),
            self.assertLogs("control_server.hook", level="WARNING") as logs,
        ):
            hook.main()

        self.assertIn("Failed to POST player hook event", logs.output[0])


if __name__ == "__main__":
    unittest.main()
