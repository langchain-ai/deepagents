# Vibe Olympics control plane

Operator commands go through one control surface:

- `vibe-control` — FastAPI web UI at `http://localhost:8766`. Dispatches game-state events (proxied to the OBS runner) and player commands.
- `vibe-player-hook` — Deep Agents hook adapter that runs on each player laptop. It reports player names and model-ready status back to `vibe-control`.

Player command dispatch uses LAN relays when `VIBE_PLAYER_<port>_RELAY` is
configured, with local iTerm2 session discovery as a same-machine fallback. See
`LAN_COMMAND_CHANNEL.md` for the relay details.

```txt
browser ──POST /api/…──▶ vibe-control ──POST /transition──▶ obs runner ──▶ OBS
                              │
                              ├──HTTP──▶ player relays ──Unix socket──▶ player CLIs
                              └──iterm2 API──▶ local player CLIs
```

## Install

```bash
cd control
uv sync
```

## Run

The live setup assumes the controller and player laptops are on the same LAN.

The controller machine runs both the OBS runner and this control panel.

Bring the OBS runner up first because it owns game state:

```bash
# terminal 1
cd vibe-coding-olympics/obs
uv run vibe-obs  # http://localhost:8765
```

Then the control panel:

```bash
# terminal 2
cd vibe-coding-olympics/control
export VIBE_PLAYER_TOKEN=<shared-token>
export VIBE_PLAYER_3001_RELAY=http://<player-1-static-ip>:9771
export VIBE_PLAYER_3002_RELAY=http://<player-2-static-ip>:9771
VIBE_CONTROL_HOST=0.0.0.0 uv run vibe-control  # http://localhost:8766
```

Open `http://localhost:8766` in a browser on the controller machine, or
`http://<controller-static-ip>:8766` from another machine on the event LAN.

On player computers:

```bash
cd vibe-coding-olympics
export VIBE_PLAYER_TOKEN=<shared-token>
export VIBE_CONTROL_API=http://<controller-static-ip>:8766
./play.sh 3001                                  # player 1
./play.sh 3002                                  # player 2, on the other computer
```

`play.sh` starts a heartbeat loop and launches `vibe-player-relay` in a
separate iTerm tab. Set `VIBE_LAUNCH_RELAY=0` before running `play.sh` to skip
that tab, or override `VIBE_RELAY_HOST` / `VIBE_RELAY_PORT` if the default
`0.0.0.0:9771` does not work on a player laptop.

If port `8766` is unavailable, override the control server bind port and use the same port in `VIBE_CONTROL_API` on every player laptop:

```bash
cd vibe-coding-olympics/control
VIBE_CONTROL_HOST=0.0.0.0 VIBE_CONTROL_PORT=8876 uv run vibe-control
```

```bash
cd vibe-coding-olympics
export VIBE_CONTROL_API=http://<controller-static-ip>:8876
./play.sh 3001
```

If port `8765` is unavailable for the OBS runner, override the runner bind port and tell the control server where to reach it:

```bash
# terminal 1
cd vibe-coding-olympics/obs
VIBE_OBS_API_PORT=8875 uv run vibe-obs

# terminal 2
cd vibe-coding-olympics/control
VIBE_OBS_API=http://localhost:8875 VIBE_CONTROL_HOST=0.0.0.0 uv run vibe-control
```

In normal event flow, run `../play.sh <port>` once per player computer at the start of the day. It starts the Vite server, opens the browser preview once, and leaves the CLI waiting for controller prompts. Use the web UI's

**Reset round all** button between rounds; it resets CLI thread/readiness state without restarting Vite or reopening the browser.

## Endpoints

| Path | Method | Body | Does |
| --- | --- | --- | --- |
| `/` | GET | — | Serves the HTML control panel |
| `/api/state` | GET | — | Proxies `GET /state` on the OBS runner |
| `/api/round/start` | POST | `{prompt?, contestants[]}` | Fires `start` on the FSM, drawing from the prompt pool when `prompt` is blank or omitted |
| `/api/round/end` | POST | `{scores: {name: float}}` | Fires `end` on the FSM |
| `/api/round/end-early` | POST | `{scores: {name: float}}` | Requires OBS `coding`, sends `times-up` to current player CLI(s), then fires `end` on the FSM |
| `/api/round/reset` | POST | `{}` | Fires `reset` on the FSM |
| `/api/prompts` | GET | — | Lists prompt pool entries |
| `/api/prompts` | POST | `{prompt: str}` | Adds a prompt pool entry |
| `/api/prompts/{id}` | PATCH | `{prompt: str}` | Updates a prompt pool entry |
| `/api/prompts/{id}` | DELETE | — | Deletes a prompt pool entry |
| `/api/prompts/draw` | GET | — | Draws a random prompt pool entry |
| `/api/players` | GET | — | Lists active player ports |
| `/api/players/connect` | POST | `{port: str}` | Marks a player launcher as connected |
| `/api/players/heartbeat` | POST | `{port: str}` | Refreshes player connection state; stale ports expire after 6 seconds |
| `/api/players/ready` | POST | `{port: str, name: str}` | Records a player name reported by the CLI hook and forwards ready names to OBS |
| `/api/players/prompt` | POST | `{prompt: str, port?: str, all?: bool}` | Sends `/skill:web-vibe Prompt: ...` to player CLI(s) |
| `/api/players/times-up` | POST | `{port?: str, all?: bool}` | Sends a `times-up` signal to player CLI(s) |
| `/api/players/clear` | POST | `{port?: str, all?: bool}` | Sends a socket `force-clear` signal and clears controller readiness for the targeted player CLI(s) |

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `VIBE_OBS_API` | `http://localhost:8765` | URL of the OBS runner |
| `VIBE_CONTROL_HOST` | `127.0.0.1` | Bind host for the control panel |
| `VIBE_CONTROL_PORT` | `8766` | Bind port for the control panel |
| `VIBE_CONTROL_API` | `http://localhost:8766` | URL used by `vibe-player-hook` from player machines |
| `VIBE_PLAYER_<port>_RELAY` | _(unset)_ | Controller-side URL for a player laptop relay, e.g. `VIBE_PLAYER_3001_RELAY` |
| `VIBE_PLAYER_TOKEN` | _(unset)_ | Shared bearer token used by controller-to-relay commands |
| `VIBE_LAUNCH_RELAY` | `1` | Whether `play.sh` should launch the player relay tab |
| `VIBE_RELAY_HOST` | `0.0.0.0` | Bind host for the player relay launched by `play.sh` |
| `VIBE_RELAY_PORT` | `9771` | Bind port for the player relay launched by `play.sh` |
| `VIBE_DEEPAGENTS_CONFIG_PATH` | `~/.deepagents/config.toml` | Deep Agents CLI config mutated by player reset cleanup |

## Player readiness hook

Deep Agents CLI emits hook events when the player enters their name and when model selection is ready. `../play.sh` writes a temporary hook config for the launched player process, so normal event laptops do not need to edit `~/.deepagents/hooks.json`.

The hook command runs on the player laptop:

```txt
Deep Agents CLI -> vibe-player-hook -> POST to VIBE_CONTROL_API
```

It reports:

- `user.name.set` to `/api/players/ready`
- `competition.player.ready` to `/api/players/model-ready`

For manual launches, configure the hook on each player laptop:

```json
{
  "hooks": [
    {
      "events": ["competition.player.ready", "user.name.set"],
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

Write that to `~/.deepagents/hooks.json` and set `VIBE_CONTROL_API` before launching if the control panel is not on localhost from the player's point of view. When using `play.sh`, set `VIBE_CONTROL_API` before running the script; the value is exported into the player terminal.

## Session-discovery contract

`play.sh` tags each new iTerm2 session it creates with a player user-variable (`user.vibe_player=<port>`), a socket user-variable (`user.vibe_event_socket=<path>`), and a session name (`vibe-player-<port>`). `control_server/iterm_ctrl.py` is the single source of truth that reads those tags for the web panel. If you ever change the tagging convention, update `play.sh` and `iterm_ctrl.py` together.
