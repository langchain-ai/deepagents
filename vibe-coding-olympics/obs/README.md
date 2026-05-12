# `obs/` — Vibe Coding Olympics game state machine + OBS compositor

MVP runner. Three-phase FSM (`IDLE` → `CODING` → `SCOREBOARD` → `IDLE`) plus a minimal compositor interface to OBS (scene switch + text updates). One-shot commands over HTTP. By default all FSM phases target the same OBS scene, `coding`; the browser overlay handles the idle, coding, and scoreboard visual states.

For live event startup, use `../README.md`. This file documents OBS setup and
the runner API.

```
producer  ──POST /transition──▶  FastAPI  ──obs-websocket──▶  OBS
                                    │
                                    └── StateMachine (in-proc)
```

## Install

```bash
cd obs
uv sync
```

## OBS prerequisites

1. OBS 28+ (obs-websocket is built-in).
2. Tools → WebSocket Server Settings → enable, note port (default `4455`). Leave authentication disabled for the default local setup.
3. Create three scenes:
   - `coding`
   - `p1 focus`
   - `p2 focus`
4. In `coding`, add these sources:
   - `Browser` — OBS Browser Source pointed at `http://localhost:8766/overlay`
   - `P1 Browser` — NDI player 1 browser feed
   - `P2 Browser` — NDI player 2 browser feed
   - `P1 CLI` — NDI player 1 CLI feed
   - `P2 CLI` — NDI player 2 CLI feed

The browser overlay renders prompts, names, scores, and idle/coding/scoreboard
state, so OBS text sources are not required by default.

For `p1 focus` and `p2 focus`, reuse the same NDI sources where possible and set
crop/position on the scene items in each scene. OBS scene-item transforms are
per scene item, so the same NDI source can usually have different layouts in
`coding` and focus scenes. Avoid source-level Crop/Pad filters for this layout
unless you deliberately want that crop shared everywhere. If you need different
NDI input settings for the same feed, create a second NDI source pointed at the
same upstream source.

## Run

```bash
uv run vibe-obs              # starts FastAPI on 127.0.0.1:8765
```

If you enable OBS WebSocket authentication, set `OBS_PASSWORD` before launching.

On startup the runner connects to OBS and primes the configured `IDLE` phase scene (default `coding`). If OBS is unreachable, the server still starts and `/healthz` reports `obs_connected: false`.

## API

### `POST /transition`

```bash
# Player names submitted before the round
curl -sS -X POST localhost:8765/transition \
  -H 'content-type: application/json' \
  -d '{"event":"ready","payload":{"contestants":["Alice","Bob"]}}'

# Start round
curl -sS -X POST localhost:8765/transition \
  -H 'content-type: application/json' \
  -d '{"event":"start","payload":{"prompt":"build a cat shrine","contestants":["Alice","Bob"]}}'

# End round → scoreboard
curl -sS -X POST localhost:8765/transition \
  -H 'content-type: application/json' \
  -d '{"event":"end","payload":{"scores":{"Alice":8.2,"Bob":7.5}}}'

# Reset → idle
curl -sS -X POST localhost:8765/transition \
  -H 'content-type: application/json' \
  -d '{"event":"reset","payload":{}}'
```

Invalid transitions return `409`. An unreachable OBS returns `503`.

### `POST /scene`

Switches OBS directly to a scene without changing the FSM snapshot. This is for
operator-controlled camera/layout cuts such as focus scenes.

```bash
curl -sS -X POST localhost:8765/scene \
  -H 'content-type: application/json' \
  -d '{"name":"p1 focus"}'
```

An unreachable OBS returns `503`.

### `GET /state`

Returns the current snapshot (phase, round, prompt, contestants, scores).

### `GET /healthz`

Returns `{ obs_connected, phase }`. Use from a producer to gate startup.

## Configuration

All env-driven. Defaults shown.

| Var | Default | Purpose |
| --- | --- | --- |
| `OBS_HOST` | `localhost` | obs-websocket host |
| `OBS_PORT` | `4455` | obs-websocket port |
| `OBS_PASSWORD` | _(empty)_ | Optional obs-websocket password, only needed when OBS WebSocket authentication is enabled |
| `OBS_SCENE_IDLE` | `coding` | Scene for `IDLE` phase |
| `OBS_SCENE_CODING` | `coding` | Scene for `CODING` phase |
| `OBS_SCENE_SCOREBOARD` | `coding` | Scene for `SCOREBOARD` phase |
| `OBS_TEXT_PROMPT` | _(unset)_ | Optional text input for the round prompt |
| `OBS_TEXT_CONTESTANT_NAME_FMT` | _(unset)_ | Optional `{n}`-template for per-slot name sources |
| `OBS_TEXT_CONTESTANT_SCORE_FMT` | _(unset)_ | Optional `{n}`-template for per-slot score sources |
| `VIBE_OBS_API_HOST` | `127.0.0.1` | FastAPI bind host |
| `VIBE_OBS_API_PORT` | `8765` | FastAPI bind port |

## Transitions

```
IDLE  ──ready──▶  IDLE        # writes: scene=coding
IDLE  ──start──▶  CODING      # writes: scene=coding
CODING ──end──▶  SCOREBOARD   # writes: scene=coding; scores stay in API state
SCOREBOARD ──reset──▶  IDLE   # writes: scene=coding
```

Any other `(phase, event)` pair is rejected. The FSM does not auto-advance; the 5-minute timer lives in `timer/index.html` and the producer (or a later pub/sub layer) decides when to fire `end`.

Slot ordering is stable in API state across a round. The overlay reads that
state directly from the control server.

## Out of scope for this MVP

- Pub/sub fanout beyond the control panel's one-shot player commands
- Browser-source URL rotation (e.g. swapping a live dev-server preview)
- Recording/streaming control
- LangSmith / event-log persistence
- Multi-round session bookkeeping
