# Smoke test plan

Copy-paste-able. Put OBS on one half of the screen, this terminal on the other. Each step has expected OBS state below.

## 0. Prereqs

OBS ≥ 28 (obs-websocket is built-in, no plugin needed). Verify:

```bash
defaults read /Applications/OBS.app/Contents/Info CFBundleShortVersionString
```

## 1. OBS one-time setup

1. OBS → Tools → WebSocket Server Settings → enable, note port (`4455`). Leave authentication disabled for the default local setup.
2. Create scenes: `coding`, `p1 focus`, `p2 focus`, and `fallback`.
3. In `coding`, add:
   - `Browser` — Browser Source at `http://localhost:8766/overlay`
   - `P1 Browser` — NDI player 1 browser feed
   - `P2 Browser` — NDI player 2 browser feed
   - `P1 CLI` — NDI player 1 CLI feed
   - `P2 CLI` — NDI player 2 CLI feed
4. In `p1 focus` and `p2 focus`, reuse the same NDI sources where possible and
   set crop/position per scene item for the focus layout.
5. In `fallback`, add the looping fallback video source.
6. Select `coding` as the active scene.

## 2. Launch runner

```bash
cd /Users/mdrxy/oss/deepagents/vibe-coding-olympics/obs
uv run vibe-obs
```

If you enable OBS WebSocket authentication, set `OBS_PASSWORD` before launching.

Leave it running. Use a second terminal for everything below.

## 3. Connectivity check

```bash
curl -s localhost:8765/healthz
```

Expect: `{"obs_connected":true,"phase":"idle"}`. OBS should be on `coding`.

## 4. Start round → `coding`

```bash
curl -s -X POST localhost:8765/transition \
  -H 'content-type: application/json' \
  -d '{"event":"start","payload":{"prompt":"build a cat shrine","contestants":["Alice","Bob"]}}'
```

OBS switches to `coding`. The `/overlay` Browser Source renders prompt,
contestant names, and timer from control-server state.

## 5. Illegal transition (negative)

```bash
curl -s -X POST localhost:8765/transition \
  -H 'content-type: application/json' \
  -d '{"event":"start","payload":{}}' \
  -w '\nHTTP %{http_code}\n'
```

Expect `HTTP 409` with detail `event 'start' not valid from phase 'coding'`. OBS must not change.

## 6. Read current state mid-round

```bash
curl -s localhost:8765/state
```

Expect phase `coding`, round `1`, prompt, contestants.

## 7. End round → `scoreboard` state

```bash
curl -s -X POST localhost:8765/transition \
  -H 'content-type: application/json' \
  -d '{"event":"end","payload":{"scores":{"Alice":8.2,"Bob":7.5}}}'
```

OBS stays on `coding`; the browser overlay renders the scoreboard state. Scores are available in `/state` for the overlay:

- `scores.Alice` → `8.2`
- `scores.Bob` → `7.5`
- contestant order remains `Alice`, `Bob`

## 8. Reset → `idle` state

```bash
curl -s -X POST localhost:8765/transition \
  -H 'content-type: application/json' \
  -d '{"event":"reset","payload":{}}'
```

OBS stays on `coding`, and the browser overlay renders the idle state.

## 9. Auth-failure check (optional)

Stop the runner, relaunch with a wrong password:

```bash
OBS_PASSWORD=wrong uv run vibe-obs
```

Runner log prints `OBS unreachable at startup: …`. Then:

```bash
curl -s localhost:8765/healthz
curl -s -X POST localhost:8765/transition \
  -H 'content-type: application/json' \
  -d '{"event":"start","payload":{"prompt":"x"}}' \
  -w '\nHTTP %{http_code}\n'
```

Expect `obs_connected:false` and `HTTP 503`. OBS does not change.

## One-liner: run steps 3–8 back-to-back

```bash
curl -s localhost:8765/healthz; echo
sleep 2
curl -s -X POST localhost:8765/transition -H 'content-type: application/json' \
  -d '{"event":"start","payload":{"prompt":"build a cat shrine","contestants":["Alice","Bob"]}}'; echo
sleep 3
curl -s -X POST localhost:8765/transition -H 'content-type: application/json' \
  -d '{"event":"end","payload":{"scores":{"Alice":8.2,"Bob":7.5}}}'; echo
sleep 3
curl -s -X POST localhost:8765/transition -H 'content-type: application/json' \
  -d '{"event":"reset","payload":{}}'; echo
```

## Failure-mode cheatsheet

| Symptom | Fix |
| --- | --- |
| `obs_connected: false` on start | Wrong port/password, or WebSocket server not enabled |
| `409` on first `start` | FSM not in `idle` — `POST /transition reset` (from `scoreboard`) or restart runner |
| Scene switch fails with code 600 | Scene name mismatch. Confirm scenes are named exactly `coding`, `p1 focus`, `p2 focus`, and `fallback`, or set the `OBS_SCENE_*` env vars |
| Overlay content is stale | Confirm `Browser` points at `http://localhost:8766/overlay` and the control server can reach the OBS runner |
| A slot you expected to fill is blank | Contestant order in the `start` payload does not match the slot you're eyeballing; only the first two contestants are rendered |
