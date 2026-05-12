# Smoke test plan

Copy-paste-able. Put OBS on one half of the screen, this terminal on the other. Each step has expected OBS state below.

## 0. Prereqs

OBS ≥ 28 (obs-websocket is built-in, no plugin needed). Verify:

```bash
defaults read /Applications/OBS.app/Contents/Info CFBundleShortVersionString
```

## 1. OBS one-time setup

1. OBS → Tools → WebSocket Server Settings → enable, note port (`4455`). Leave authentication disabled for the default local setup.
2. Create the main scene: `coding`.
3. Add `Text (FreeType 2)` sources to `coding` with these exact names and some placeholder text:
   - `PromptText`, `Contestant1Name`, `Contestant2Name`
4. Select `coding` as the active scene.

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

Expect: `{"obs_connected":true,"phase":"idle"}`. OBS should be on `coding` with all text sources cleared (runner primes on startup).

## 4. Start round → `coding`

```bash
curl -s -X POST localhost:8765/transition \
  -H 'content-type: application/json' \
  -d '{"event":"start","payload":{"prompt":"build a cat shrine","contestants":["Alice","Bob"]}}'
```

OBS switches to `coding`. Text sources show:

- `PromptText` → `build a cat shrine`
- `Contestant1Name` → `Alice`
- `Contestant2Name` → `Bob`

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
- `Contestant1Name` / `Contestant2Name` remain `Alice` / `Bob`

## 8. Reset → `idle` state

```bash
curl -s -X POST localhost:8765/transition \
  -H 'content-type: application/json' \
  -d '{"event":"reset","payload":{}}'
```

OBS stays on `coding`. `PromptText` + every `Contestant{n}Name` source are cleared, and the browser overlay renders the idle state.

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
| Scene switches but text stays stale | Source name mismatch (case-sensitive). Check OBS names equal `PromptText` / `Contestant{n}Name`, or set `OBS_TEXT_*` / `OBS_TEXT_CONTESTANT_*_FMT` env vars to match your layout |
| Text source blanks but never fills | Source name mismatch, or the source is not present in the active `coding` scene |
| A slot you expected to fill is blank | Contestant order in the `start` payload does not match the slot you're eyeballing; only the first two contestants are rendered |
