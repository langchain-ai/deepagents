# `control/` — Vibe Olympics control plane

Two entry points backed by the same iTerm2 discovery logic:

- `vibe-control` — FastAPI web UI at `http://localhost:8766`. Dispatches game-state events (proxied to the OBS runner) and player commands.
- `vibe-players` — CLI for scripting / SSH: `vibe-players list|clear|reset`.

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
cd ../obs && OBS_PASSWORD='...' uv run vibe-obs   # http://localhost:8765
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
uv run vibe-players clear --port 3001
uv run vibe-players clear --all
uv run vibe-players reset --port 3001
uv run vibe-players reset --all
```

## Endpoints

| Path | Method | Body | Does |
| --- | --- | --- | --- |
| `/` | GET | — | Serves the HTML control panel |
| `/api/state` | GET | — | Proxies `GET /state` on the OBS runner |
| `/api/round/start` | POST | `{prompt, contestants[]}` | Fires `start` on the FSM |
| `/api/round/end` | POST | `{scores: {name: float}}` | Fires `end` on the FSM |
| `/api/round/reset` | POST | `{}` | Fires `reset` on the FSM |
| `/api/players` | GET | — | Lists active player ports |
| `/api/players/clear` | POST | `{port?: str, all?: bool}` | Sends `/clear` to player CLI(s) |
| `/api/players/reset` | POST | `{port?: str, all?: bool}` | Quits + relaunches player CLI(s) |

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `VIBE_OBS_API` | `http://localhost:8765` | URL of the OBS runner |
| `VIBE_CONTROL_HOST` | `127.0.0.1` | Bind host for the control panel |
| `VIBE_CONTROL_PORT` | `8766` | Bind port for the control panel |

## Session-discovery contract

`play.sh` tags each new iTerm2 session it creates with both a user-variable (`user.vibe_player=<port>`) and a session name (`vibe-player-<port>`). `control_server/iterm_ctrl.py` is the single source of truth that reads those tags; both the web panel and `vibe-players` CLI delegate to it. If you ever change the tagging convention, update `play.sh` and `iterm_ctrl.py` together.
