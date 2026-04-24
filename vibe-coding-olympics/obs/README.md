# `obs/` — Vibe Coding Olympics game state machine + OBS compositor

MVP runner. Three-phase FSM (`IDLE` → `CODING` → `SCOREBOARD` → `IDLE`) plus a minimal compositor interface to OBS (scene switch + text updates). One-shot commands over HTTP. Pub/sub and per-contestant CLI client hooks are deferred.

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
2. Tools → WebSocket Server Settings → enable, note port (default `4455`) + password.
3. Create three scenes (names overridable via env):
   - `Idle` — standby card
   - `Coding` — contestant view; contains the text sources below
   - `Scoreboard` — results view; contains `Scores`
4. Create text sources (any backend works — `text_gdiplus_v2` on Windows, `text_ft2_source_v2` elsewhere):
   - Singular, on the `Coding` scene: `PromptText`
   - Two contestant slots — name + score each — on both `Coding` and `Scoreboard`:
     `Contestant1Name`, `Contestant1Score`, `Contestant2Name`, `Contestant2Score`.
   - Slot count is hardcoded to 2. Source-name templates are env-overridable — see below.

## Run

```bash
export OBS_PASSWORD=...      # only if OBS requires one
uv run vibe-obs              # starts FastAPI on 127.0.0.1:8765
```

On startup the runner connects to OBS, primes the idle scene, and clears all four text sources. If OBS is unreachable, the server still starts and `/healthz` reports `obs_connected: false`.

## API

### `POST /transition`

```bash
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
| `OBS_PASSWORD` | _(empty)_ | obs-websocket password |
| `OBS_SCENE_IDLE` | `Idle` | Scene for `IDLE` phase |
| `OBS_SCENE_CODING` | `Coding` | Scene for `CODING` phase |
| `OBS_SCENE_SCOREBOARD` | `Scoreboard` | Scene for `SCOREBOARD` phase |
| `OBS_TEXT_PROMPT` | `PromptText` | Text input for the round prompt |
| `OBS_TEXT_CONTESTANT_NAME_FMT` | `Contestant{n}Name` | `{n}`-template for per-slot name sources |
| `OBS_TEXT_CONTESTANT_SCORE_FMT` | `Contestant{n}Score` | `{n}`-template for per-slot score sources |
| `VIBE_OBS_API_HOST` | `127.0.0.1` | FastAPI bind host |
| `VIBE_OBS_API_PORT` | `8765` | FastAPI bind port |

## Transitions

```
IDLE  ──start──▶  CODING      # writes: scene=Coding, PromptText,
                              #         Contestant{n}Name per slot, clears unused
CODING ──end──▶  SCOREBOARD   # writes: scene=Scoreboard, Contestant{n}Score per
                              #         slot (mapped to each contestant's CODING slot)
SCOREBOARD ──reset──▶  IDLE   # writes: scene=Idle, clears all text sources
```

Any other `(phase, event)` pair is rejected. The FSM does not auto-advance; the 5-minute timer lives in `timer/index.html` and the producer (or a later pub/sub layer) decides when to fire `end`.

Slot ordering is stable across a round: contestant `i` from the `start` payload stays in `Contestant{i+1}Name` / `Contestant{i+1}Score` through to the scoreboard. Only two slots exist; contestants beyond the second are dropped.

## Out of scope for this MVP

- Pub/sub fanout to per-contestant CLI agents (separate PR)
- Browser-source URL rotation (e.g. swapping a live dev-server preview)
- Recording/streaming control
- LangSmith / event-log persistence
- Multi-round session bookkeeping
