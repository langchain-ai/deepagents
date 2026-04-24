# `control/` — Vibe Olympics web control panel

Minimal FastAPI web UI that dispatches two kinds of commands from one page:

- **Game state** — start / end / reset the round. Proxied to the OBS runner's `POST /transition` endpoint.
- **Players** — list, `/clear`, or full `/quit`-and-relaunch the `play.sh`-spawned CLI sessions via the iTerm2 Python API.

```
browser ──POST /api/…──▶ control_server ──POST /transition──▶ obs runner ──▶ OBS
                              │
                              └──iterm2 API──▶ player CLIs
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

## Endpoints

| Path | Method | Body | Does |
| --- | --- | --- | --- |
| `/` | GET | — | Serves the HTML control panel |
| `/api/state` | GET | — | Proxies `GET /state` on the OBS runner |
| `/api/round/start` | POST | `{prompt, contestants[], round_num?}` | Fires `start` on the FSM |
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

## Relationship to `../control.py`

`control.py` is the standalone CLI (`list`/`clear`/`reset`). The web panel reuses the same tagging convention (`user.vibe_player` / `vibe-player-<port>` session name) via `control_server/iterm_ctrl.py`. Both are valid entry points; the CLI is handier over SSH or in scripts, the web UI for a producer driving a live round.
