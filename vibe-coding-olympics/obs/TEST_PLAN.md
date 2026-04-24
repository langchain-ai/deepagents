# Smoke test plan

Copy-paste-able. Put OBS on one half of the screen, this terminal on the other. Each step has expected OBS state below.

## 0. Prereqs

OBS ≥ 28 (obs-websocket is built-in, no plugin needed). Verify:

```bash
defaults read /Applications/OBS.app/Contents/Info CFBundleShortVersionString
```

## 1. OBS one-time setup

1. OBS → Tools → WebSocket Server Settings → enable, note port (`4455`) + password.
2. Create scenes: `Idle`, `Coding`, `Scoreboard`.
3. Add `Text (FreeType 2)` sources with these exact names and some placeholder text:
   - `Coding` scene: `PromptText`, `Contestant1Name`, `Contestant2Name`
   - `Scoreboard` scene: `Contestant1Name`, `Contestant1Score`, `Contestant2Name`, `Contestant2Score`
   - Tip: add the `Contestant{n}Name` sources in `Coding` first, then in `Scoreboard` right-click → *Add Existing* → pick the source so both scenes share a single instance (the runner writes once, both scenes reflect it).
4. Select `Idle` as the active scene.

## 2. Launch runner

```bash
cd /Users/mdrxy/oss/deepagents/vibe-coding-olympics/obs
export OBS_PASSWORD='paste-from-obs'   # skip if auth disabled
uv run vibe-obs
```

Leave it running. Use a second terminal for everything below.

## 3. Connectivity check

```bash
curl -s localhost:8765/healthz
```

Expect: `{"obs_connected":true,"phase":"idle"}`. OBS should be on `Idle` with all text sources cleared (runner primes on startup).

## 4. Start round → `Coding`

```bash
curl -s -X POST localhost:8765/transition \
  -H 'content-type: application/json' \
  -d '{"event":"start","payload":{"prompt":"build a cat shrine","round_num":1,"contestants":["Alice","Bob"]}}'
```

OBS switches to `Coding`. Text sources show:

- `PromptText` → `build a cat shrine`
- `Contestant1Name` → `Alice`
- `Contestant2Name` → `Bob`
- `Contestant1Score`, `Contestant2Score` → blank

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

## 7. End round → `Scoreboard`

```bash
curl -s -X POST localhost:8765/transition \
  -H 'content-type: application/json' \
  -d '{"event":"end","payload":{"scores":{"Alice":8.2,"Bob":7.5}}}'
```

OBS switches to `Scoreboard`. Each score lands in the same slot its contestant held in `Coding`:

- `Contestant1Score` → `8.20` (Alice's slot)
- `Contestant2Score` → `7.50` (Bob's slot)
- `Contestant1Name` / `Contestant2Name` remain `Alice` / `Bob`

## 8. Reset → `Idle`

```bash
curl -s -X POST localhost:8765/transition \
  -H 'content-type: application/json' \
  -d '{"event":"reset","payload":{}}'
```

OBS switches to `Idle`. `PromptText` + every `Contestant{n}Name` / `Contestant{n}Score` are cleared.

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
  -d '{"event":"start","payload":{"prompt":"x","round_num":1}}' \
  -w '\nHTTP %{http_code}\n'
```

Expect `obs_connected:false` and `HTTP 503`. OBS does not change.

## One-liner: run steps 3–8 back-to-back

```bash
curl -s localhost:8765/healthz; echo
sleep 2
curl -s -X POST localhost:8765/transition -H 'content-type: application/json' \
  -d '{"event":"start","payload":{"prompt":"build a cat shrine","round_num":1,"contestants":["Alice","Bob"]}}'; echo
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
| Scene switches but text stays stale | Source name mismatch (case-sensitive). Check OBS names equal `PromptText` / `Contestant{n}Name` / `Contestant{n}Score`, or set `OBS_TEXT_*` / `OBS_TEXT_CONTESTANT_*_FMT` env vars to match your layout |
| Text source blanks but never fills | Source exists in one scene only; confirm by viewing the target scene. Shared across scenes requires OBS *Add Existing* (single instance), not a duplicate |
| A slot you expected to fill is blank | Contestant order in the `start` payload does not match the slot you're eyeballing; only the first two contestants are rendered |
