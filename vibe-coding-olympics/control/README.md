# `control/` — Vibe Olympics control plane

Two entry points backed by the same iTerm2 discovery logic:

- `vibe-control` — FastAPI web UI at `http://localhost:8766`. Dispatches game-state events (proxied to the OBS runner) and player commands.
- `vibe-players` — CLI for scripting / SSH: `vibe-players list|prompt|times-up|clear|reset`.
- `vibe-player-hook` — Deep Agents hook adapter. Reports player names from `user.name.set` into the control panel.

```
browser ──POST /api/…──▶ vibe-control ──POST /transition──▶ obs runner ──▶ OBS
                              │
                              ├──iterm2 API──▶ player CLIs
                              │
shell ──vibe-players ─────────┘   (CLI path reuses the same iterm_ctrl helpers)
```

## Install

```bash
cd control
uv sync
```

## Run

Bring the OBS runner up first (it owns game state):

```bash
# terminal 1
cd ../obs && uv run vibe-obs                      # http://localhost:8765
```

Then the control panel:

```bash
# terminal 2
cd control && uv run vibe-control                  # http://localhost:8766
```

Open `http://localhost:8766` in a browser.

### CLI

Same venv, different entry point. Runs one-shot and exits:

```bash
cd control
uv run vibe-players list
uv run vibe-players prompt "a website for a taco truck" --all
uv run vibe-players times-up --all
uv run vibe-players clear --port 3001
uv run vibe-players clear --all
uv run vibe-players reset --port 3001
uv run vibe-players reset --all
```

In normal event flow, run `../play.sh <port>` once per player computer at the
start of the day. It starts the Vite server, opens the browser preview once,
and leaves the CLI waiting for controller prompts. Use `clear` between rounds;
it resets the CLI thread/readiness state without restarting Vite or reopening
the browser.

## Endpoints

| Path | Method | Body | Does |
| --- | --- | --- | --- |
| `/` | GET | — | Serves the HTML control panel |
| `/api/state` | GET | — | Proxies `GET /state` on the OBS runner |
| `/api/round/start` | POST | `{prompt, contestants[]}` | Fires `start` on the FSM |
| `/api/round/end` | POST | `{scores: {name: float}}` | Fires `end` on the FSM |
| `/api/round/reset` | POST | `{}` | Fires `reset` on the FSM |
| `/api/players` | GET | — | Lists active player ports |
| `/api/players/ready` | POST | `{port: str, name: str}` | Records a player name reported by the CLI hook and forwards ready names to OBS |
| `/api/players/prompt` | POST | `{prompt: str, port?: str, all?: bool}` | Sends `/skill:web-vibe Prompt: ...` to player CLI(s) |
| `/api/players/times-up` | POST | `{port?: str, all?: bool}` | Sends a `times-up` signal to player CLI(s) |
| `/api/players/clear` | POST | `{port?: str, all?: bool}` | Sends a socket `force-clear` signal and clears controller readiness for the targeted player CLI(s) |
| `/api/players/reset` | POST | `{port?: str, all?: bool}` | Quits + relaunches player CLI(s) |

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `VIBE_OBS_API` | `http://localhost:8765` | URL of the OBS runner |
| `VIBE_CONTROL_HOST` | `127.0.0.1` | Bind host for the control panel |
| `VIBE_CONTROL_PORT` | `8766` | Bind port for the control panel |
| `VIBE_CONTROL_API` | `http://localhost:8766` | URL used by `vibe-player-hook` from player machines |

## Player readiness hook

Deep Agents CLI already emits `user.name.set` after the player submits their
name during launch setup. `../play.sh` now writes a round-local hook config
and points the launched CLI at it automatically, so normal event laptops do
not need to edit `~/.deepagents/hooks.json`.

For manual launches, configure the hook on each player laptop:

```json
{
  "hooks": [
    {
      "events": ["user.name.set"],
      "command": [
        "uv",
        "run",
        "--project",
        "/Users/mdrxy/oss/deepagents/vibe-coding-olympics/control",
        "vibe-player-hook"
      ]
    }
  ]
}
```

Write that to `~/.deepagents/hooks.json` and set `VIBE_CONTROL_API` before
launching if the control panel is not on localhost from the player's point of
view. When using `play.sh`, set `VIBE_CONTROL_API` before running the script;
the value is exported into the player terminal.

## Session-discovery contract

`play.sh` tags each new iTerm2 session it creates with a player user-variable (`user.vibe_player=<port>`), a socket user-variable (`user.vibe_event_socket=<path>`), and a session name (`vibe-player-<port>`). `control_server/iterm_ctrl.py` is the single source of truth that reads those tags; both the web panel and `vibe-players` CLI delegate to it. If you ever change the tagging convention, update `play.sh` and `iterm_ctrl.py` together.
