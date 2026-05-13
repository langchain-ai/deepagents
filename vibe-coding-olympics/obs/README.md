# `obs/` — Vibe Coding Olympics OBS compositor

Thin HTTP shim over obs-websocket. Two verbs: switch scene, update text source.
No game state — the round FSM lives in the control plane
(`control_server.state_machine`). This runner just renders what control tells
it.

For live event startup, use `../README.md`. This file documents OBS setup and
the runner API.

```
control  ──POST /scene──▶  FastAPI  ──obs-websocket──▶  OBS
         ──POST /text──▶
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

On startup the runner connects to OBS. If OBS is unreachable, the server still
starts and `/healthz` reports `obs_connected: false`. The runner is stateless
between requests — the control plane is responsible for writing scene/text on
every phase entry.

## API

### `POST /scene`

Switch the current OBS program scene.

```bash
curl -sS -X POST localhost:8765/scene \
  -H 'content-type: application/json' \
  -d '{"name":"coding"}'
```

An unreachable OBS returns `503`.

### `POST /text`

Update the text content of an OBS text input.

```bash
curl -sS -X POST localhost:8765/text \
  -H 'content-type: application/json' \
  -d '{"source":"PromptText","value":"build a cat shrine"}'
```

An unreachable OBS returns `503`.

### `GET /healthz`

Returns `{ obs_connected }`. Use from a producer to gate startup.

## Configuration

All env-driven. Defaults shown.

| Var | Default | Purpose |
| --- | --- | --- |
| `OBS_HOST` | `localhost` | obs-websocket host |
| `OBS_PORT` | `4455` | obs-websocket port |
| `OBS_PASSWORD` | _(empty)_ | Optional obs-websocket password, only needed when OBS WebSocket authentication is enabled |
| `VIBE_OBS_API_HOST` | `127.0.0.1` | FastAPI bind host |
| `VIBE_OBS_API_PORT` | `8765` | FastAPI bind port |

Scene names per phase and OBS text-source mappings are configured in the
control plane — see `control_server.state_config`.

## Out of scope for this MVP

- Pub/sub fanout beyond the control panel's one-shot player commands
- Browser-source URL rotation (e.g. swapping a live dev-server preview)
- Recording/streaming control
- LangSmith / event-log persistence
- Multi-round session bookkeeping
