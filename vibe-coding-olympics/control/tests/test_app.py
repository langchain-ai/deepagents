from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from control_server import app as app_mod


class TestReadyPlayers(unittest.TestCase):
    def setUp(self) -> None:
        app_mod._ready_players.clear()
        app_mod._model_ready_ports.clear()

    def tearDown(self) -> None:
        app_mod._ready_players.clear()
        app_mod._model_ready_ports.clear()

    def test_ready_players_forward_to_obs_in_submission_order(self) -> None:
        client = TestClient(app_mod.create_app())
        forward = AsyncMock(return_value={"phase": "idle"})

        with patch("control_server.app._forward", forward):
            first = client.post(
                "/api/players/ready",
                json={"port": "3002", "name": "Bob"},
            )
            second = client.post(
                "/api/players/ready",
                json={"port": "3001", "name": "Alice"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["contestants"], ["Bob", "Alice"])
        forward.assert_any_await("ready", {"contestants": ["Bob"]})
        forward.assert_any_await("ready", {"contestants": ["Bob", "Alice"]})

    def test_round_start_uses_ready_names_when_contestants_omitted(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._ready_players.update({"3002": "Bob", "3001": "Alice"})
        app_mod._model_ready_ports.update({"3002", "3001"})
        forward = AsyncMock(return_value={"phase": "coding"})

        with (
            patch("control_server.app._forward", forward),
            patch(
                "control_server.iterm_ctrl.send_prompt_to_players",
                new=AsyncMock(return_value=["3001", "3002"]),
            ),
        ):
            response = client.post(
                "/api/round/start",
                json={"prompt": "build a taco truck"},
            )

        self.assertEqual(response.status_code, 200)
        forward.assert_awaited_once_with(
            "start",
            {"prompt": "build a taco truck", "contestants": ["Bob", "Alice"]},
        )

    def test_round_start_rejects_before_both_models_ready(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._ready_players.update({"3001": "Alice", "3002": "Bob"})
        app_mod._model_ready_ports.add("3001")

        response = client.post(
            "/api/round/start",
            json={"prompt": "build a taco truck", "contestants": ["Alice", "Bob"]},
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("Waiting for model selection from: Bob (3002)", response.text)

    def test_round_start_rejects_before_two_named_players(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._ready_players.update({"3001": "Alice"})
        app_mod._model_ready_ports.add("3001")

        response = client.post(
            "/api/round/start",
            json={"prompt": "build a taco truck", "contestants": ["Alice"]},
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn(
            "Two players must enter names and select models before start.",
            response.text,
        )

    def test_round_start_ignores_stale_extra_named_players(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._ready_players.update(
            {"3001": "Alice", "3002": "Bob", "3999": "Old Player"}
        )
        app_mod._model_ready_ports.update({"3001", "3002"})
        forward = AsyncMock(return_value={"phase": "coding"})

        with (
            patch("control_server.app._forward", forward),
            patch(
                "control_server.iterm_ctrl.send_prompt_to_players",
                new=AsyncMock(return_value=["3001", "3002"]),
            ),
        ):
            response = client.post(
                "/api/round/start",
                json={"prompt": "build a taco truck"},
            )

        self.assertEqual(response.status_code, 200)
        forward.assert_awaited_once_with(
            "start",
            {"prompt": "build a taco truck", "contestants": ["Alice", "Bob"]},
        )

    def test_round_start_rejects_empty_prompt(self) -> None:
        client = TestClient(app_mod.create_app())

        response = client.post(
            "/api/round/start",
            json={"prompt": "   ", "contestants": ["Alice"]},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("Prompt must not be empty.", response.text)

    def test_players_prompt_rejects_empty_prompt(self) -> None:
        client = TestClient(app_mod.create_app())

        response = client.post(
            "/api/players/prompt",
            json={"all": True, "prompt": ""},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("Prompt must not be empty.", response.text)

    def test_round_reset_clears_ready_names(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._ready_players.update({"3001": "Alice"})

        with patch("control_server.app._forward", new=AsyncMock(return_value={})):
            response = client.post("/api/round/reset", json={})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(app_mod._ready_players, {})
        self.assertEqual(app_mod._model_ready_ports, set())

    def test_model_ready_tracks_port_separately_from_name_ready(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._ready_players.update({"3001": "Alice", "3002": "Bob"})

        with patch(
            "control_server.iterm_ctrl.players_ready",
            new=AsyncMock(return_value=[]),
        ) as players_ready:
            response = client.post("/api/players/model-ready", json={"port": "3001"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ready"], {"3001": "Alice", "3002": "Bob"})
        self.assertEqual(response.json()["model_ready"], ["3001"])
        players_ready.assert_not_awaited()

    def test_second_model_ready_notifies_players_waiting_to_start(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._ready_players.update({"3001": "Alice", "3002": "Bob"})
        app_mod._model_ready_ports.add("3001")

        with patch(
            "control_server.iterm_ctrl.players_ready",
            new=AsyncMock(return_value=["3001", "3002"]),
        ) as players_ready:
            response = client.post("/api/players/model-ready", json={"port": "3002"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_ready"], ["3001", "3002"])
        self.assertEqual(response.json()["players_ready_sent"], ["3001", "3002"])
        players_ready.assert_awaited_once_with(None)

    def test_players_list_includes_model_ready_ports(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._ready_players.update({"3001": "Alice"})
        app_mod._model_ready_ports.add("3001")

        response = client.get("/api/players")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_ready"], ["3001"])

    def test_players_clear_resets_readiness_without_relaunching(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._ready_players.update({"3001": "Alice", "3002": "Bob"})
        app_mod._model_ready_ports.update({"3001", "3002"})
        forward = AsyncMock(return_value={"phase": "idle"})

        with (
            patch("control_server.app._forward", forward),
            patch(
                "control_server.iterm_ctrl.clear_players",
                new=AsyncMock(return_value=["3001", "3002"]),
            ) as clear_players,
        ):
            response = client.post("/api/players/clear", json={"all": True})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["cleared"], ["3001", "3002"])
        self.assertEqual(response.json()["ready"], {})
        self.assertEqual(response.json()["model_ready"], [])
        self.assertEqual(app_mod._ready_players, {})
        self.assertEqual(app_mod._model_ready_ports, set())
        clear_players.assert_awaited_once_with(None)
        forward.assert_awaited_once_with("reset", {})

    def test_control_ui_shows_start_error_when_models_are_missing(self) -> None:
        client = TestClient(app_mod.create_app())

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('id="start-error"', response.text)
        self.assertIn(
            "Both players must select a model before the round can start.",
            response.text,
        )
        self.assertIn('aria-disabled="true">Start</button>', response.text)


if __name__ == "__main__":
    unittest.main()
