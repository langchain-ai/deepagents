from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from control_server import app as app_mod
from control_server import site_urls as site_urls_mod
from control_server.round_timer import RoundTimer, TimerSnapshot


def _reset_module_globals() -> None:
    """Reset every mutable module-level singleton `control_server.app` owns.

    Without this, `_round_counter`/`_round_timer` leak across tests:
    counter assertions flake after the first run and pending timer
    tasks leak into the next test's event loop.
    """
    app_mod._connected_ports.clear()
    app_mod._ready_players.clear()
    app_mod._model_ready_ports.clear()
    app_mod._prompt_pool.clear()
    app_mod._prompt_pool.update(enumerate(app_mod.DEFAULT_PROMPTS, start=1))
    app_mod._next_prompt_id = len(app_mod._prompt_pool) + 1
    app_mod._round_context.clear()
    app_mod._last_eval_results.clear()
    app_mod._overlay_smoke_state = None
    app_mod._round_counter = 0
    # Swap in a fresh RoundTimer rather than awaiting cancel() across
    # the previous test's event loop, which TestClient has already
    # closed.
    app_mod._round_timer = RoundTimer()
    app_mod._eval_lock = None


class TestReadyPlayers(unittest.TestCase):
    def setUp(self) -> None:
        _reset_module_globals()

    def tearDown(self) -> None:
        _reset_module_globals()

    def test_overlay_route_serves_obs_browser_source_page(self) -> None:
        client = TestClient(app_mod.create_app())

        response = client.get("/overlay")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Vibe Olympics Overlay", response.text)
        self.assertIn("background: transparent !important", response.text)
        self.assertIn("/static/fonts/aeonik-mono/aeonikmono-regular.woff2", response.text)
        self.assertIn("defaultOverlayMode = params.get('mode')", response.text)
        self.assertIn("function syncOverlayMode", response.text)
        self.assertIn("fetch('/api/state'", response.text)
        self.assertIn("Player: Player 1", response.text)
        self.assertIn("function playerName", response.text)
        self.assertIn("transform: translateX(-50%)", response.text)
        self.assertIn(".preview-name.right", response.text)
        self.assertIn("right: 2.2%", response.text)
        self.assertGreaterEqual(response.text.count("Deep Agents: PVP Speedrun"), 3)
        self.assertNotIn("Vibe Coding Olympics", response.text)
        self.assertNotIn("split-prompt", response.text)
        self.assertIn("var(--blue-b) 20%", response.text)
        self.assertIn("var(--pink-b) 20%", response.text)
        self.assertIn("var(--paper) 50%", response.text)
        self.assertIn("background-image: var(--blue-gradient)", response.text)
        self.assertIn("background-image: var(--pink-gradient)", response.text)
        self.assertIn("var(--blue-b) 58%", response.text)
        self.assertIn("linear-gradient(180deg, var(--paper) 0%, var(--blue-b) 58%", response.text)
        self.assertIn("linear-gradient(180deg, var(--paper) 0%, var(--pink-b) 58%", response.text)
        self.assertIn("background-image: var(--blue-name-gradient)", response.text)
        self.assertIn("background-image: var(--pink-name-gradient)", response.text)
        self.assertIn("height: 11%;", response.text)
        self.assertIn("border-bottom: var(--line) solid var(--ink)", response.text)
        self.assertIn("line-height: 1;", response.text)
        self.assertNotIn('class="player-name', response.text)
        self.assertIn('id="focus-preview-label">Live Preview</div>', response.text)
        self.assertIn("white-space: nowrap", response.text)
        self.assertNotIn("P1 Live", response.text)
        self.assertNotIn("P2 Live", response.text)
        self.assertNotIn("Live<br>Preview", response.text)
        self.assertIn("split-gap-line", response.text)
        self.assertIn("split-gap-dot", response.text)
        self.assertIn("split-bg preview-terminal-gap", response.text)
        self.assertIn("split-bg bottom-gap", response.text)
        self.assertNotIn("P1 Terminal", response.text)
        self.assertNotIn("P2 Terminal", response.text)
        self.assertNotIn("terminal-title", response.text)
        self.assertNotIn("terminal-input", response.text)
        self.assertNotIn("terminal-arrow", response.text)
        self.assertIn("background-attachment: fixed", response.text)
        self.assertIn("isolation: isolate", response.text)
        self.assertIn("timer-warning", response.text)
        self.assertIn("function syncTimerWarning", response.text)
        self.assertIn("threshold_secs", response.text)
        self.assertIn('<div class="pane focus-terminal"></div>', response.text)
        self.assertIn(".focus-prompt", response.text)
        self.assertIn("focus-bg middle-gap", response.text)
        self.assertIn("focus-bg bottom-website", response.text)
        self.assertIn('#coding-view[data-focus="2"] .focus-bg', response.text)
        self.assertIn("bottom: 3%;", response.text)
        self.assertNotIn("focus-terminal-title", response.text)
        self.assertNotIn("focus-caption", response.text)

    def test_index_links_to_overlay_route(self) -> None:
        client = TestClient(app_mod.create_app())

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Vibe Olympics Control", response.text)
        self.assertIn('href="/overlay"', response.text)
        self.assertIn('id="smoke-modal"', response.text)
        self.assertIn('id="btn-open-smoke"', response.text)
        self.assertIn("/api/overlay-smoke", response.text)
        self.assertIn("Run transition tour", response.text)
        self.assertIn("#smoke-modal button", response.text)
        self.assertIn("smoke-command-actions", response.text)
        self.assertIn("btn-smoke-layout-split", response.text)
        self.assertIn("btn-smoke-layout-p1", response.text)
        self.assertNotIn("smoke-overlay-link", response.text)

    def test_static_fonts_are_served(self) -> None:
        client = TestClient(app_mod.create_app())

        response = client.get("/static/fonts/aeonik-mono/aeonikmono-regular.woff2")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "font/woff2")
        self.assertGreater(len(response.content), 0)

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
                "control_server.player_dispatch.send_prompt_to_players",
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
                "control_server.player_dispatch.send_prompt_to_players",
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
                "control_server.player_dispatch.send_prompt_to_players",
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

    def test_round_start_arms_the_server_authoritative_timer(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._ready_players.update({"3001": "Alice", "3002": "Bob"})
        app_mod._model_ready_ports.update({"3001", "3002"})
        forward = AsyncMock(return_value={"phase": "coding"})
        timer_start = AsyncMock()

        env = {"VIBE_ROUND_SECONDS": "120"}
        with (
            patch("control_server.app._forward", forward),
            patch(
                "control_server.player_dispatch.send_prompt_to_players",
                new=AsyncMock(return_value=["3001", "3002"]),
            ),
            patch.dict("os.environ", env, clear=False),
            patch.object(app_mod._round_timer, "start", new=timer_start),
        ):
            response = client.post(
                "/api/round/start",
                json={"prompt": "build something", "contestants": ["Alice", "Bob"]},
            )

        self.assertEqual(response.status_code, 200)
        # TestClient closes its per-request event loop on return, so we
        # can't observe the live timer task. Asserting `start` was
        # awaited with the configured duration is the durable contract.
        timer_start.assert_awaited_once()
        duration_arg = timer_start.await_args.args[0]
        self.assertEqual(duration_arg, 120.0)
        self.assertEqual(
            timer_start.await_args.kwargs["start_delay_secs"],
            app_mod._PLAYER_LAUNCH_COUNTDOWN_SECS,
        )
        self.assertEqual(response.json()["duration_secs"], 120.0)
        self.assertEqual(response.json()["round_num"], 1)

    def test_round_end_early_runs_judge_and_forwards_scores(self) -> None:
        from control_server import eval_runner

        client = TestClient(app_mod.create_app())
        app_mod._ready_players.update({"3001": "Alice", "3002": "Bob"})
        app_mod._round_context.update(
            {
                "prompt": "build a taco truck",
                "round_num": 1,
                "contestants": ["Alice", "Bob"],
            }
        )

        async def fake_run_eval(
            *, url: str, site_name: str, prompt: str, round_num: int, work_dir: Any
        ) -> eval_runner.EvalResult:
            del url, prompt, round_num, work_dir
            axes: dict[str, float | None] = dict.fromkeys(eval_runner.LLM_AXES, 0.5)
            return eval_runner.EvalResult.success(
                site_name=site_name,
                url="http://example/x",
                prompt="p",
                round_num=1,
                axes=axes,
            )

        forward = AsyncMock(return_value={"phase": "scoreboard"})
        times_up = AsyncMock(return_value=["3001", "3002"])

        with (
            patch(
                "control_server.app._get_obs_state",
                new=AsyncMock(return_value={"phase": "coding"}),
            ),
            patch("control_server.app._forward", forward),
            patch(
                "control_server.app.site_urls.resolve",
                lambda port: site_urls_mod.SiteUrlResult(
                    url=f"http://192.168.1.{port[-1]}:{port}", reason=None
                ),
            ),
            patch("control_server.player_dispatch.times_up_players", times_up),
            patch("control_server.eval_runner.run_eval", new=fake_run_eval),
        ):
            response = client.post("/api/round/end-early", json={})

        self.assertEqual(response.status_code, 200)
        forward.assert_awaited_once_with(
            "end",
            {"scores": {"Alice": 5.0, "Bob": 5.0}},
        )
        body = response.json()
        names = sorted(entry["name"] for entry in body["results"])
        self.assertEqual(names, ["Alice", "Bob"])
        for entry in body["results"]:
            self.assertEqual(entry["overall"], 0.5)
            self.assertEqual(entry["obs_score"], 5.0)
            self.assertFalse(entry["fallback"])
        self.assertIsNone(body["obs_error"])

    def test_round_end_early_rejects_when_not_coding(self) -> None:
        client = TestClient(app_mod.create_app())

        with (
            patch(
                "control_server.app._get_obs_state",
                new=AsyncMock(return_value={"phase": "idle"}),
            ),
            patch("control_server.app._forward", new=AsyncMock()) as forward,
            patch(
                "control_server.player_dispatch.times_up_players",
                new=AsyncMock(),
            ) as times_up,
        ):
            response = client.post("/api/round/end-early", json={})

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
            "control_server.player_dispatch.players_ready",
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
            "control_server.player_dispatch.players_ready",
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
                "control_server.player_dispatch.clear_players",
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
            'id="btn-end-early" aria-disabled="true">End early (trigger judge)</button>',
            response.text,
        )
        self.assertIn('id="btn-open-override"', response.text)
        self.assertIn('id="override-modal"', response.text)
        self.assertIn('id="timer-clock"', response.text)
        self.assertIn('id="eval-results"', response.text)
        self.assertIn('id="end-error"', response.text)
        self.assertIn('id="obs-phase">unknown</span>', response.text)
        self.assertIn('id="state-summary"', response.text)
        self.assertIn('id="state-prompt">none</dd>', response.text)
        self.assertIn('id="state-contestants">none</dd>', response.text)
        self.assertIn('id="state-scores">none</dd>', response.text)
        self.assertIn("Not connected", response.text)
        self.assertIn('id="ready-players">none</span>', response.text)
        self.assertNotIn("Connected players:", response.text)
        self.assertIn("renderPromptEditor", response.text)
        self.assertIn("edit.textContent = 'Edit'", response.text)
        self.assertIn("save.textContent = 'Save'", response.text)
        self.assertIn("function clearPromptInput()", response.text)
        self.assertGreaterEqual(response.text.count("clearPromptInput();"), 2)
        self.assertNotIn("Send prompt to all", response.text)


class TestRoundOverrideEnd(unittest.TestCase):
    def setUp(self) -> None:
        _reset_module_globals()

    def tearDown(self) -> None:
        _reset_module_globals()

    def test_override_end_forwards_supplied_scores_without_judge(self) -> None:
        client = TestClient(app_mod.create_app())
        forward = AsyncMock(return_value={"phase": "scoreboard"})
        run_eval = AsyncMock()

        with (
            patch("control_server.app._forward", forward),
            patch("control_server.eval_runner.run_eval", new=run_eval),
        ):
            response = client.post(
                "/api/round/override-end",
                json={"scores": {"Alice": 9.5, "Bob": 4.2}},
            )

        self.assertEqual(response.status_code, 200)
        forward.assert_awaited_once_with(
            "end",
            {"scores": {"Alice": 9.5, "Bob": 4.2}},
        )
        run_eval.assert_not_awaited()

    def test_override_end_rejects_out_of_range_scores(self) -> None:
        client = TestClient(app_mod.create_app())

        response = client.post(
            "/api/round/override-end",
            json={"scores": {"Alice": 11.0, "Bob": -1.0}},
        )

        self.assertEqual(response.status_code, 422)

    def test_override_end_cancels_the_round_timer(self) -> None:
        client = TestClient(app_mod.create_app())
        timer_cancel = AsyncMock()

        with (
            patch(
                "control_server.app._forward",
                new=AsyncMock(return_value={"phase": "scoreboard"}),
            ),
            patch.object(app_mod._round_timer, "cancel", new=timer_cancel),
        ):
            response = client.post(
                "/api/round/override-end",
                json={"scores": {"Alice": 5.0, "Bob": 5.0}},
            )

        self.assertEqual(response.status_code, 200)
        timer_cancel.assert_awaited_once()


class TestRoundEvalFallback(unittest.TestCase):
    def setUp(self) -> None:
        _reset_module_globals()

    def tearDown(self) -> None:
        _reset_module_globals()

    def test_missing_site_url_yields_fallback_results(self) -> None:
        client = TestClient(app_mod.create_app())
        app_mod._ready_players.update({"3001": "Alice", "3002": "Bob"})
        app_mod._round_context.update(
            {
                "prompt": "build something",
                "round_num": 1,
                "contestants": ["Alice", "Bob"],
            }
        )

        forward = AsyncMock(return_value={"phase": "scoreboard"})
        times_up = AsyncMock(return_value=[])

        async def never_called(
            *, url: str, site_name: str, prompt: str, round_num: int, work_dir: Any
        ) -> object:
            raise AssertionError(
                "run_eval should not be called when site URL is missing"
            )

        with (
            patch(
                "control_server.app._get_obs_state",
                new=AsyncMock(return_value={"phase": "coding"}),
            ),
            patch("control_server.app._forward", forward),
            patch(
                "control_server.app.site_urls.resolve",
                lambda port: site_urls_mod.SiteUrlResult(
                    url=None, reason=f"unset for {port}"
                ),
            ),
            patch("control_server.player_dispatch.times_up_players", times_up),
            patch("control_server.eval_runner.run_eval", new=never_called),
        ):
            response = client.post("/api/round/end-early", json={})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["results"]), 2)
        for entry in body["results"]:
            self.assertTrue(entry["fallback"])
            # Reason should reflect the resolver's diagnostic, not a
            # generic "no site URL configured" string.
            self.assertTrue(entry["fallback_reason"].startswith("unset for"))
            self.assertEqual(entry["url"], "")
        forward.assert_awaited_once()
        forwarded_event, forwarded_payload = forward.await_args.args
        self.assertEqual(forwarded_event, "end")
        self.assertEqual(set(forwarded_payload["scores"].keys()), {"Alice", "Bob"})

    def test_crashing_eval_does_not_abort_other_players(self) -> None:
        from control_server import eval_runner

        client = TestClient(app_mod.create_app())
        app_mod._ready_players.update({"3001": "Alice", "3002": "Bob"})
        app_mod._round_context.update(
            {"prompt": "p", "round_num": 1, "contestants": ["Alice", "Bob"]}
        )

        async def half_crashing(
            *, url: str, site_name: str, prompt: str, round_num: int, work_dir: Any
        ) -> eval_runner.EvalResult:
            del url, prompt, round_num, work_dir
            if site_name == "Alice":
                msg = "synthetic blowup"
                raise RuntimeError(msg)
            return eval_runner.EvalResult.success(
                site_name=site_name,
                url="http://example/x",
                prompt="p",
                round_num=1,
                axes=dict.fromkeys(eval_runner.LLM_AXES, 0.6),
            )

        with (
            patch(
                "control_server.app._get_obs_state",
                new=AsyncMock(return_value={"phase": "coding"}),
            ),
            patch(
                "control_server.app._forward",
                new=AsyncMock(return_value={"phase": "scoreboard"}),
            ),
            patch(
                "control_server.app.site_urls.resolve",
                lambda port: site_urls_mod.SiteUrlResult(
                    url=f"http://x:{port}", reason=None
                ),
            ),
            patch(
                "control_server.player_dispatch.times_up_players",
                new=AsyncMock(return_value=["3001", "3002"]),
            ),
            patch("control_server.eval_runner.run_eval", new=half_crashing),
        ):
            response = client.post("/api/round/end-early", json={})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        # Both players must still appear; one as a synthesized fallback
        # carrying the crash reason, the other with the real score.
        names_by_fallback = {
            entry["name"]: entry["fallback"] for entry in body["results"]
        }
        self.assertEqual(names_by_fallback, {"Alice": True, "Bob": False})
        alice = next(e for e in body["results"] if e["name"] == "Alice")
        self.assertIn("eval crashed", alice["fallback_reason"])


class TestStateExposesTimerAndEval(unittest.TestCase):
    def setUp(self) -> None:
        _reset_module_globals()

    def tearDown(self) -> None:
        _reset_module_globals()

    def test_state_includes_timer_round_and_namespaced_obs_error(self) -> None:
        client = TestClient(app_mod.create_app())
        with patch(
            "control_server.app._get_obs_state",
            new=AsyncMock(return_value={"phase": "idle"}),
        ):
            response = client.get("/api/state")
        body = response.json()
        self.assertEqual(body["timer"]["running"], False)
        self.assertEqual(body["timer"]["duration_secs"], 0.0)
        self.assertIsNone(body["timer"]["started_at"])
        self.assertIsNone(body["timer"]["warning"])
        self.assertEqual(body["eval"]["results"], [])
        self.assertIsNone(body["obs_error"])
        self.assertEqual(body["phase"], "idle")
        self.assertIn("round", body)
        self.assertIsNone(body["round"]["prompt"])
        self.assertEqual(body["round"]["contestants"], [])
        self.assertIsNone(body["round"]["duration_warning"])

    def test_state_namespaces_obs_runner_errors(self) -> None:
        from fastapi import HTTPException

        client = TestClient(app_mod.create_app())
        with patch(
            "control_server.app._get_obs_state",
            new=AsyncMock(side_effect=HTTPException(status_code=502, detail="down")),
        ):
            response = client.get("/api/state")
        body = response.json()
        self.assertEqual(body["obs_error"], {"detail": "down", "status": 502})
        # Critically: real OBS keys must NOT have been overwritten by the
        # diagnostic envelope (defense in depth — empty `obs` payload here).
        self.assertNotIn("status", body)

    def test_state_surfaces_invalid_round_duration_env(self) -> None:
        client = TestClient(app_mod.create_app())
        env = {"VIBE_ROUND_SECONDS": "not-a-number"}
        with (
            patch("control_server.app._get_obs_state", new=AsyncMock(return_value={})),
            patch.dict("os.environ", env, clear=False),
        ):
            response = client.get("/api/state")
        body = response.json()
        self.assertIsNotNone(body["round"]["duration_warning"])
        self.assertIn("not-a-number", body["round"]["duration_warning"])

    def test_state_includes_timer_warning_threshold(self) -> None:
        client = TestClient(app_mod.create_app())
        snapshot = TimerSnapshot.active(
            duration_secs=300.0,
            remaining_secs=59.0,
            started_at=123.0,
        )
        with (
            patch(
                "control_server.app._get_obs_state",
                new=AsyncMock(return_value={"phase": "coding"}),
            ),
            patch.object(app_mod._round_timer, "snapshot", return_value=snapshot),
        ):
            response = client.get("/api/state")
        body = response.json()
        self.assertEqual(body["timer"]["started_at"], 123.0)
        self.assertEqual(
            body["timer"]["warning"],
            {"threshold_secs": 60, "message": "1 minute left"},
        )

    def test_overlay_smoke_state_overrides_obs_state(self) -> None:
        client = TestClient(app_mod.create_app())

        response = client.post(
            "/api/overlay-smoke",
            json={
                "phase": "coding",
                "prompt": "build a neon scoreboard",
                "contestants": ["Alice", "Bob"],
                "duration_secs": 300,
                "remaining_secs": 60,
                "mode": "focus",
                "focus_player": 2,
            },
        )

        self.assertEqual(response.status_code, 200)
        state = response.json()["state"]
        self.assertEqual(state["phase"], "coding")
        self.assertEqual(state["prompt"], "build a neon scoreboard")
        self.assertEqual(state["contestants"], ["Alice", "Bob"])
        self.assertEqual(state["timer"]["warning"]["threshold_secs"], 60)
        self.assertTrue(state["overlay_smoke"]["active"])
        self.assertEqual(state["overlay_smoke"]["mode"], "focus")
        self.assertEqual(state["overlay_smoke"]["focus_player"], 2)

        with patch(
            "control_server.app._get_obs_state",
            new=AsyncMock(side_effect=AssertionError("OBS should not be queried")),
        ):
            smoke = client.get("/api/state")

        body = smoke.json()
        self.assertEqual(body["phase"], "coding")
        self.assertTrue(body["overlay_smoke"]["active"])
        self.assertEqual(body["overlay_smoke"]["mode"], "focus")
        self.assertEqual(body["overlay_smoke"]["focus_player"], 2)

    def test_overlay_smoke_scoreboard_defaults_scores(self) -> None:
        client = TestClient(app_mod.create_app())

        response = client.post(
            "/api/overlay-smoke",
            json={"phase": "scoreboard", "contestants": ["Alice", "Bob"]},
        )

        self.assertEqual(response.status_code, 200)
        state = response.json()["state"]
        self.assertEqual(state["phase"], "scoreboard")
        self.assertEqual(state["scores"], {"Alice": 8.6, "Bob": 7.8})
        self.assertFalse(state["timer"]["running"])

    def test_overlay_smoke_clear_restores_live_state(self) -> None:
        client = TestClient(app_mod.create_app())
        client.post("/api/overlay-smoke", json={"phase": "idle"})

        with patch(
            "control_server.app._get_obs_state",
            new=AsyncMock(return_value={"phase": "idle"}),
        ):
            response = client.delete("/api/overlay-smoke")

        state = response.json()["state"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(state["phase"], "idle")
        self.assertFalse(state["overlay_smoke"]["active"])
        self.assertIsNone(app_mod._overlay_smoke_state)


if __name__ == "__main__":
    unittest.main()
