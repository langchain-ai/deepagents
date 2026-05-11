from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from control_server import app as app_mod


class TestReadyPlayers(unittest.TestCase):
    def setUp(self) -> None:
        app_mod._connected_ports.clear()
        app_mod._ready_players.clear()
        app_mod._model_ready_ports.clear()
        app_mod._prompt_pool.clear()
        app_mod._prompt_pool.update(enumerate(app_mod.DEFAULT_PROMPTS, start=1))
        app_mod._next_prompt_id = len(app_mod._prompt_pool) + 1

    def tearDown(self) -> None:
        app_mod._connected_ports.clear()
        app_mod._ready_players.clear()
        app_mod._model_ready_ports.clear()
        app_mod._prompt_pool.clear()
        app_mod._prompt_pool.update(enumerate(app_mod.DEFAULT_PROMPTS, start=1))
        app_mod._next_prompt_id = len(app_mod._prompt_pool) + 1

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
        send_prompt = AsyncMock(return_value=["3002", "3001"])

        with (
            patch("control_server.app._forward", forward),
            patch(
                "control_server.iterm_ctrl.send_prompt_to_players",
                new=send_prompt,
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
        send_prompt.assert_awaited_once_with(["3002", "3001"], "build a taco truck")

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
        send_prompt = AsyncMock(return_value=["3001", "3002"])

        with (
            patch("control_server.app._forward", forward),
            patch(
                "control_server.iterm_ctrl.send_prompt_to_players",
                new=send_prompt,
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
        send_prompt.assert_awaited_once_with(["3001", "3002"], "build a taco truck")

    def test_round_start_draws_prompt_when_prompt_is_empty(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._ready_players.update({"3001": "Alice", "3002": "Bob"})
        app_mod._model_ready_ports.update({"3001", "3002"})
        app_mod._prompt_pool.clear()
        app_mod._prompt_pool[1] = "taco truck"
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
                json={"prompt": "   ", "contestants": ["Alice", "Bob"]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prompt"], "taco truck")
        forward.assert_awaited_once_with(
            "start",
            {
                "prompt": "taco truck",
                "contestants": ["Alice", "Bob"],
            },
        )

    def test_round_start_rejects_empty_prompt_pool(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._ready_players.update({"3001": "Alice", "3002": "Bob"})
        app_mod._model_ready_ports.update({"3001", "3002"})
        app_mod._prompt_pool.clear()

        response = client.post(
            "/api/round/start",
            json={"contestants": ["Alice", "Bob"]},
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("Prompt pool is empty.", response.text)

    def test_round_end_early_signals_players_then_ends_round(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._ready_players.update({"3001": "Alice", "3002": "Bob"})
        calls: list[str] = []

        async def times_up(ports: list[str] | None) -> list[str]:
            calls.append("times-up")
            self.assertEqual(ports, ["3001", "3002"])
            return ["3001", "3002"]

        async def forward(event: str, payload: dict[str, object]) -> dict[str, str]:
            calls.append(event)
            self.assertEqual(payload, {"scores": {"Alice": 8.2, "Bob": 7.5}})
            return {"phase": "scoreboard"}

        with (
            patch(
                "control_server.app._get_obs_state",
                new=AsyncMock(return_value={"phase": "coding"}),
            ),
            patch("control_server.app._forward", forward),
            patch("control_server.iterm_ctrl.times_up_players", times_up),
        ):
            response = client.post(
                "/api/round/end-early",
                json={"scores": {"Alice": 8.2, "Bob": 7.5}},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["times_up_sent"], ["3001", "3002"])
        self.assertEqual(response.json()["state"], {"phase": "scoreboard"})
        self.assertEqual(calls, ["times-up", "end"])

    def test_round_end_early_rejects_when_not_coding(self) -> None:
        client = TestClient(app_mod.create_app())

        with (
            patch(
                "control_server.app._get_obs_state",
                new=AsyncMock(return_value={"phase": "idle"}),
            ),
            patch("control_server.app._forward", new=AsyncMock()) as forward,
            patch(
                "control_server.iterm_ctrl.times_up_players",
                new=AsyncMock(),
            ) as times_up,
        ):
            response = client.post("/api/round/end-early", json={"scores": {}})

        self.assertEqual(response.status_code, 409)
        self.assertIn("Cannot end early while OBS is in phase `idle`.", response.text)
        times_up.assert_not_awaited()
        forward.assert_not_awaited()

    def test_prompt_crud_and_draw_endpoints(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._prompt_pool.clear()
        app_mod._next_prompt_id = 1

        created = client.post(
            "/api/prompts",
            json={"prompt": "  moon laundromat  "},
        )
        listed = client.get("/api/prompts")
        updated = client.patch(
            "/api/prompts/1",
            json={"prompt": "moon greenhouse"},
        )
        drawn = client.get("/api/prompts/draw")
        deleted = client.delete("/api/prompts/1")

        self.assertEqual(created.status_code, 200)
        self.assertEqual(
            created.json(),
            {"id": 1, "prompt": "moon laundromat"},
        )
        self.assertEqual(
            listed.json()["prompts"],
            [{"id": 1, "prompt": "moon laundromat"}],
        )
        self.assertEqual(
            updated.json(),
            {"id": 1, "prompt": "moon greenhouse"},
        )
        self.assertEqual(drawn.json(), {"prompt": "moon greenhouse"})
        self.assertEqual(
            deleted.json(),
            {"id": 1, "prompt": "moon greenhouse"},
        )
        self.assertEqual(client.get("/api/prompts").json()["prompts"], [])

    def test_prompt_list_places_new_entries_before_defaults(self) -> None:
        client = TestClient(app_mod.create_app())

        client.post("/api/prompts", json={"prompt": "moon greenhouse"})
        client.post("/api/prompts", json={"prompt": "neon museum"})

        prompts = client.get("/api/prompts").json()["prompts"]

        self.assertEqual(prompts[0]["prompt"], "neon museum")
        self.assertEqual(prompts[1]["prompt"], "moon greenhouse")
        self.assertEqual(prompts[2]["prompt"], "taco truck")

    def test_players_prompt_rejects_empty_prompt(self) -> None:
        client = TestClient(app_mod.create_app())

        response = client.post(
            "/api/players/prompt",
            json={"all": True, "prompt": ""},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("Prompt must not be empty.", response.text)

    def test_players_connect_tracks_connected_port(self) -> None:
        client = TestClient(app_mod.create_app())

        response = client.post("/api/players/connect", json={"port": "3001"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["connected"], ["3001"])
        self.assertIn("3001", app_mod._connected_ports)

    def test_players_heartbeat_refreshes_connected_port(self) -> None:
        client = TestClient(app_mod.create_app())

        with patch("control_server.app.time.monotonic", return_value=100.0):
            response = client.post("/api/players/heartbeat", json={"port": "3001"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["connected"], ["3001"])
        self.assertEqual(app_mod._connected_ports["3001"], 100.0)

    def test_players_list_expires_stale_connected_port(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._connected_ports["3001"] = 1.0
        app_mod._ready_players["3001"] = "Alice"
        app_mod._model_ready_ports.add("3001")

        with patch("control_server.app.time.monotonic", return_value=10.0):
            response = client.get("/api/players")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["connected"], [])
        self.assertEqual(response.json()["ready"], {})
        self.assertEqual(response.json()["model_ready"], [])
        self.assertEqual(app_mod._connected_ports, {})

    def test_round_reset_clears_ready_names(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._connected_ports["3001"] = app_mod.time.monotonic()
        app_mod._ready_players.update({"3001": "Alice"})

        with patch("control_server.app._forward", new=AsyncMock(return_value={})):
            response = client.post("/api/round/reset", json={})

        self.assertEqual(response.status_code, 200)
        self.assertIn("3001", app_mod._connected_ports)
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
        self.assertEqual(response.json()["connected"], ["3001"])
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
        players_ready.assert_awaited_once_with(["3001", "3002"])

    def test_players_list_includes_model_ready_ports(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._connected_ports["3001"] = app_mod.time.monotonic()
        app_mod._ready_players.update({"3001": "Alice"})
        app_mod._model_ready_ports.add("3001")

        with patch(
            "control_server.iterm_ctrl.list_players",
            new=AsyncMock(return_value=[]),
        ):
            response = client.get("/api/players")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["connected"], ["3001"])
        self.assertEqual(response.json()["model_ready"], ["3001"])

    def test_players_clear_resets_readiness_without_relaunching(self) -> None:
        client = TestClient(app_mod.create_app())
        seen_at = app_mod.time.monotonic()
        app_mod._connected_ports.update({"3001": seen_at, "3002": seen_at})
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
        self.assertEqual(response.json()["connected"], ["3001", "3002"])
        self.assertEqual(response.json()["ready"], {})
        self.assertEqual(response.json()["model_ready"], [])
        self.assertEqual(set(app_mod._connected_ports), {"3001", "3002"})
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
        self.assertIn(
            'id="btn-end-early" aria-disabled="true">End early</button>',
            response.text,
        )
        self.assertIn('id="end-error"', response.text)
        self.assertIn('id="obs-phase">unknown</span>', response.text)
        self.assertIn('id="state-summary"', response.text)
        self.assertIn('id="state-prompt">none</dd>', response.text)
        self.assertIn('id="state-contestants">none</dd>', response.text)
        self.assertIn('id="state-scores">none</dd>', response.text)
        self.assertIn("Not connected", response.text)
        self.assertIn('id="connected-players">none</span>', response.text)
        self.assertIn("renderPromptEditor", response.text)
        self.assertIn("edit.textContent = 'Edit'", response.text)
        self.assertIn("save.textContent = 'Save'", response.text)
        self.assertNotIn("Send prompt to all", response.text)


if __name__ == "__main__":
    unittest.main()
