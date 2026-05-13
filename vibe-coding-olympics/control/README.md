# Vibe Olympics control plane

Operator commands go through one control surface. This service also owns the
round state machine (`IDLE → CODING → SCOREBOARD → IDLE`); the OBS runner is a
downstream renderer.

- `vibe-control` — FastAPI web UI at `http://localhost:8766`. Owns the FSM,
  drives the OBS runner over HTTP for scene/text writes, and fans out player
  commands.
- `vibe-player-hook` — Deep Agents hook adapter that runs on each player laptop. It reports player names and model-ready status back to `vibe-control`.

Player command dispatch uses LAN relays when `VIBE_PLAYER_<port>_RELAY` is
configured, with local iTerm2 session discovery as a same-machine fallback. See
`LAN_COMMAND_CHANNEL.md` for the relay details.

```txt
browser ──POST /api/…──▶ vibe-control ──┬──POST /scene──▶ obs runner ──▶ OBS
                              │         └──POST /text───▶
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

Bring the OBS runner up first so the control panel has a compositor to drive:

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

Click **Prepare next round** between rounds; it resets CLI thread/readiness state and blanks the Vite project without restarting Vite or reopening the browser.

## Endpoints

| Path | Method | Body | Does |
| --- | --- | --- | --- |
| `/` | GET | — | Serves the HTML control panel |
| `/overlay` | GET | — | Serves the transparent OBS Browser Source graphics overlay for the LED panel. Add `?mode=focus&p=1` or `?mode=focus&p=2` for a single-player focus layout. Live video feeds are expected to be composed in OBS by default. |
| `/api/state` | GET | — | Returns the in-process FSM snapshot plus `timer`, `round`, and `eval` fields |
| `/api/state/events` | GET | — | Server-sent event stream that pushes full `/api/state` payloads to the overlay; `/overlay` keeps slow polling as a fallback |
| `/api/eval/last` | GET | — | Returns the latest per-player judge results plus pending/published score maps |
| `/api/eval/publish` | POST | `{scores?: {name: float}}` | Publishes host-approved scores to the scoreboard. Empty `scores` accepts the pending judge scores. |
| `/api/overlay-smoke` | POST/DELETE | `{phase, prompt?, contestants?, scores?, duration_secs?, remaining_secs?, mode?, focus_player?}` | Enables or clears controller-only overlay smoke state without starting a real round |
| `/api/obs/scene` | POST | `{scene: str}` | Asks the OBS runner to switch directly to an OBS scene without changing game-state phase |
| `/api/round/start` | POST | `{prompt?, contestants[], duration_secs?}` | Fires `start` on the FSM, draws from the prompt pool when `prompt` is blank, sends the prompt to player CLIs, and arms the server-authoritative round timer after the CLI launch countdown |
| `/api/round/end` | POST | `{}` | Cancels the timer, sends `times-up` to player CLI(s), runs the LLM judge, and stores results on the control server for host approval |
| `/api/round/end-early` | POST | `{}` | Requires OBS `coding`, sends `times-up` to player CLI(s), runs the judge, and stores results on the control server for host approval |
| `/api/round/override-end` | POST | `{scores: {name: float}}` | Smoke-test bypass: cancels the timer and stores the supplied scores without invoking the judge |
| `/api/round/reset` | POST | `{}` | Fires `reset` on the FSM and cancels the timer |
| `/api/prompts` | GET | — | Lists prompt pool entries |
| `/api/prompts` | POST | `{prompt: str}` | Adds a prompt pool entry |
| `/api/prompts/{id}` | PATCH | `{prompt: str}` | Updates a prompt pool entry |
| `/api/prompts/{id}` | DELETE | — | Deletes a prompt pool entry |
| `/api/prompts/draw` | GET | — | Draws a random prompt pool entry |
| `/api/players` | GET | — | Lists active player ports |
| `/api/players/connect` | POST | `{port: str}` | Marks a player launcher as connected |
| `/api/players/heartbeat` | POST | `{port: str}` | Refreshes player connection state; stale ports expire after 6 seconds |
| `/api/players/ready` | POST | `{port: str, name: str}` | Records a player name reported by the CLI hook and fires `ready` on the FSM to render names in OBS |
| `/api/players/model-ready` | POST | `{port: str}` | Records that one player has selected a model and can wait for round start |
| `/api/players/model-unready` | POST | `{port: str}` | Clears model-ready status for one player after a local CLI round reset |
| `/api/players/prompt` | POST | `{prompt: str, port?: str, all?: bool}` | Sends `/skill:web-vibe Prompt: ...` to player CLI(s) |
| `/api/players/times-up` | POST | `{port?: str, all?: bool}` | Sends a `times-up` signal to player CLI(s) |
| `/api/players/clear` | POST | `{port?: str, all?: bool}` | Blanks the player Vite project, sends a socket `force-clear` signal, and clears controller readiness for the targeted player CLI(s) |

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `VIBE_OBS_API` | `http://localhost:8765` | URL of the OBS runner (compositor target) |
| `OBS_SCENE_IDLE` | `coding` | OBS scene to switch to on `IDLE` entry |
| `OBS_SCENE_CODING` | `coding` | OBS scene to switch to on `CODING` entry |
| `OBS_SCENE_SCOREBOARD` | `coding` | OBS scene to switch to on `SCOREBOARD` entry |
| `OBS_TEXT_PROMPT` | _(unset)_ | Optional OBS text source written with the round prompt |
| `OBS_TEXT_CONTESTANT_NAME_FMT` | _(unset)_ | Optional `{n}`-template for per-slot name sources |
| `OBS_TEXT_CONTESTANT_SCORE_FMT` | _(unset)_ | Optional `{n}`-template for per-slot score sources |
| `VIBE_CONTROL_HOST` | `127.0.0.1` | Bind host for the control panel |
| `VIBE_CONTROL_PORT` | `8766` | Bind port for the control panel |
| `VIBE_CONTROL_API` | `http://localhost:8766` | URL used by `vibe-player-hook` from player machines |
| `VIBE_PLAYER_<port>_RELAY` | _(unset)_ | Controller-side URL for a player laptop relay, e.g. `VIBE_PLAYER_3001_RELAY`. Its host is reused to build the player's site URL for the judge. |
| `VIBE_PLAYER_<port>_SITE_URL` | _(unset)_ | Optional explicit override for the player's site URL (e.g. a remote deploy). Wins over the relay-derived URL. |
| `VIBE_PLAYER_TOKEN` | _(unset)_ | Shared bearer token used by controller-to-relay commands |
| `VIBE_LAUNCH_RELAY` | `1` | Whether `play.sh` should launch the player relay tab |
| `VIBE_RELAY_HOST` | `0.0.0.0` | Bind host for the player relay launched by `play.sh` |
| `VIBE_RELAY_PORT` | `9771` | Bind port for the player relay launched by `play.sh` |
| `VIBE_ROUND_SECONDS` | `300` | Fallback player coding time when `/api/round/start` omits `duration_secs`; starts after the CLI launch countdown and triggers the judge on expiry |
| `VIBE_OVERLAY_INLINE_FEEDS` | _(unset)_ | Set to `1`, `true`, `yes`, or `on` to restore the legacy `/overlay` mode that embeds local NDI bridge iframes from `127.0.0.1:8889`. Leave unset for the production graphics-only overlay. |
| `VIBE_EVAL_DIR` | _(bundled `vibe-coding-olympics/eval`)_ | Override path to the eval workspace, useful for local development |
| `VIBE_EVAL_RESULTS_DIR` | system temp dir | Parent directory for `round-N-<name>.json` files written by the judge |
| `VIBE_DEEPAGENTS_CONFIG_PATH` | `~/.deepagents/config.toml` | Deep Agents CLI config mutated by player reset cleanup |

## Judge integration

The control server is the only thing that runs the LLM judge. When the round timer expires (or the operator clicks **End early**), the server:

1. Sends `times-up` to both player CLIs.
2. Derives each player's site URL — `http://<relay-host>:<player-port>` by default, or `VIBE_PLAYER_<port>_SITE_URL` if set.
3. Runs `vibe-coding-olympics/eval/judge.py` per site concurrently via `uv run`.
4. Aggregates per-axis scores into a single `[0, 1]` overall and scales them to `0..10` for the control website.
5. Holds the scores in the control panel as pending results. The host can accept them or edit manual overrides, then publish to the OBS scoreboard.
6. Stores per-axis results on the control server.

If the judge subprocess fails (15-second timeout, non-zero exit, missing/malformed JSON, or no usable LLM scores) the controller substitutes randomized axis scores and tags the result with `fallback=true` plus a `fallback_reason` so post-event analysis can audit which sites were judged versus filled in. No DQs are ever issued — the show goes on.

The **Override scores** modal on the control website calls `/api/round/override-end` and is intended only for smoke tests; it skips the judge and stores the entered scores without publishing to OBS.

## Player readiness hook

Deep Agents CLI emits hook events when the player enters their name and when model selection is ready. `../play.sh` writes a temporary hook config for the launched player process, so normal event laptops do not need to edit `~/.deepagents/hooks.json`.

The hook command runs on the player laptop:

```txt
Deep Agents CLI -> vibe-player-hook -> POST to VIBE_CONTROL_API
```

It reports:

- `user.name.set` to `/api/players/ready`
- `competition.player.ready` to `/api/players/model-ready`
- `competition.player.reset` to `/api/players/model-unready`

For manual launches, configure the hook on each player laptop:

```json
{
  "hooks": [
    {
      "events": [
        "competition.player.ready",
        "competition.player.reset",
        "user.name.set"
      ],
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
