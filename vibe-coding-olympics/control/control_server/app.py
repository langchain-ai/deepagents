"""FastAPI control panel: HTML page + JSON endpoints.

Fans one-shot commands to:

- The OBS runner (`POST /transition` at `VIBE_OBS_API`, default
  `http://localhost:8765`) for game-state events.
- iTerm2 player sessions via the helpers in `iterm_ctrl`.

No auth, no persistence, no websockets. Localhost MVP.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from control_server import iterm_ctrl

logger = logging.getLogger(__name__)

VIBE_OBS_API = os.environ.get("VIBE_OBS_API", "http://localhost:8765").rstrip("/")


class StartRequest(BaseModel):
    prompt: str
    contestants: list[str] = Field(default_factory=list)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Reject empty or whitespace-only prompts."""
        prompt = value.strip()
        if not prompt:
            msg = "Prompt must not be empty."
            raise ValueError(msg)
        return prompt


class EndRequest(BaseModel):
    scores: dict[str, float] = Field(default_factory=dict)


class PlayerTarget(BaseModel):
    """Target a single player by port or every active player."""

    port: str | None = None
    all: bool = False


class PlayerPromptRequest(PlayerTarget):
    """Send a selected prompt to one player or every active player."""

    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Reject empty or whitespace-only prompts."""
        prompt = value.strip()
        if not prompt:
            msg = "Prompt must not be empty."
            raise ValueError(msg)
        return prompt


class PlayerReadyRequest(BaseModel):
    """Player identity reported by the CLI onboarding hook."""

    port: str
    name: str


class PlayerModelReadyRequest(BaseModel):
    """Player model-selection readiness reported by the CLI hook."""

    port: str


_ready_players: dict[str, str] = {}
_model_ready_ports: set[str] = set()


def _ready_contestants() -> list[str]:
    """Return ready player names in submission order."""
    return list(_ready_players.values())


def _round_player_ports() -> list[str]:
    """Return the two player ports assigned to the current round."""
    return list(_ready_players)[:2]


def _all_named_players_model_ready() -> bool:
    """Return whether both named players have reached the waiting state."""
    ports = _round_player_ports()
    return len(ports) == 2 and all(port in _model_ready_ports for port in ports)


def _start_blocked_message() -> str:
    """Describe why the round cannot start yet."""
    ports = _round_player_ports()
    missing = [
        f"{_ready_players.get(port, port)} ({port})"
        for port in ports
        if port not in _model_ready_ports
    ]
    if len(ports) < 2:
        return "Two players must enter names and select models before start."
    if missing:
        return "Waiting for model selection from: " + ", ".join(missing)
    return "Both players must select a model before the round can start."


async def _forward(event: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Relay an FSM event to the OBS runner's `/transition` endpoint."""
    url = f"{VIBE_OBS_API}/transition"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(
                url, json={"event": event, "payload": payload}
            )
        except httpx.HTTPError as exc:
            msg = f"OBS runner at {VIBE_OBS_API} unreachable: {exc}"
            raise HTTPException(status_code=502, detail=msg) from exc
    if response.status_code >= 400:
        # Unwrap the upstream JSON detail so the UI sees one error
        # layer, not `{"detail": "{\"detail\": ...}"}`.
        try:
            body = response.json()
            detail = body.get("detail", body) if isinstance(body, dict) else body
        except ValueError:
            detail = response.text
        raise HTTPException(status_code=response.status_code, detail=detail)
    return response.json()


def _resolve_ports(target: PlayerTarget) -> list[str] | None:
    """Translate a `PlayerTarget` into an arg for `iterm_ctrl`."""
    if target.all:
        return None
    if not target.port:
        msg = "Provide `port` or set `all` to true."
        raise HTTPException(status_code=400, detail=msg)
    return [target.port]


def _clear_player_readiness(ports: list[str] | None) -> None:
    """Forget player readiness for targeted ports, or all players."""
    if ports is None:
        _ready_players.clear()
        _model_ready_ports.clear()
        return
    for port in ports:
        _ready_players.pop(port, None)
        _model_ready_ports.discard(port)


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Vibe Olympics Control</title>
<style>
  :root { color-scheme: dark; }
  body {
    font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
    max-width: 820px;
    margin: 2rem auto;
    padding: 0 1rem;
    background: #0a0a0a;
    color: #e5e5e5;
  }
  h1 { margin-top: 0; }
  section {
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    background: #141414;
  }
  h2 { margin-top: 0; font-size: 1rem; color: #a0a0a0; text-transform: uppercase; letter-spacing: 0.05em; }
  label { display: block; margin-bottom: 0.75rem; font-size: 0.9rem; color: #b0b0b0; }
  input[type=text], input[type=number] {
    width: 100%;
    padding: 0.55rem 0.6rem;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    background: #0a0a0a;
    color: #e5e5e5;
    font-family: inherit;
    font-size: 0.95rem;
    margin-top: 0.25rem;
    box-sizing: border-box;
  }
  input:focus { outline: none; border-color: #555; }
  .player-slot {
    width: 100%;
    min-height: 2.35rem;
    padding: 0.55rem 0.6rem;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    background: #0a0a0a;
    color: #e5e5e5;
    font-size: 0.95rem;
    margin-top: 0.25rem;
    box-sizing: border-box;
  }
  .player-slot.empty { color: #707070; }
  button {
    padding: 0.55rem 1.1rem;
    border: none;
    border-radius: 6px;
    background: #2563eb;
    color: #fff;
    font-family: inherit;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    margin-right: 0.4rem;
    margin-top: 0.25rem;
  }
  button:hover { filter: brightness(1.15); }
  button:disabled, button[aria-disabled="true"] { cursor: not-allowed; filter: grayscale(0.7) brightness(0.75); opacity: 0.55; }
  button.danger { background: #dc2626; }
  button.secondary { background: #525252; }
  .muted { color: #888; font-size: 0.85rem; margin-top: 0.65rem; }
  .inline-error { color: #fca5a5; font-size: 0.85rem; }
  .inline-error:empty { display: none; }
  #ready-players { color: #d4d4d4; }
  .label-line { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }
  .ready-badge {
    display: none;
    border-radius: 999px;
    padding: 0.08rem 0.45rem;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .ready-badge.visible { display: inline-block; }
  .ready-badge.waiting {
    color: #fde68a;
    border: 1px solid #a16207;
    background: #422006;
  }
  .ready-badge.ready {
    color: #86efac;
    border: 1px solid #166534;
    background: #052e16;
  }
  .action-line { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; }
  .row { display: flex; gap: 0.75rem; }
  .row > label { flex: 1; }
  #log {
    background: #000;
    color: #6ee7b7;
    padding: 0.75rem;
    border-radius: 6px;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 0.78rem;
    height: 200px;
    overflow-y: auto;
    white-space: pre-wrap;
    border: 1px solid #2a2a2a;
  }
  #log .err { color: #fca5a5; }
</style>
</head>
<body>
<h1>Vibe Olympics Control</h1>

<section>
  <h2>Start round</h2>
  <label>Prompt
    <input id="prompt" type="text" placeholder="build a cat shrine" required>
  </label>
  <div class="row">
    <label><span class="label-line"><span>Player 1</span><span class="ready-badge" id="c1-ready">Ready</span></span>
      <div class="player-slot empty" id="c1">Waiting for CLI player</div>
    </label>
    <label><span class="label-line"><span>Player 2</span><span class="ready-badge" id="c2-ready">Ready</span></span>
      <div class="player-slot empty" id="c2">Waiting for CLI player</div>
    </label>
  </div>
  <div class="action-line">
    <button id="btn-start" aria-disabled="true">Start</button>
    <span class="ready-badge ready" id="round-started">Round started</span>
    <span class="inline-error" id="start-error" role="alert"></span>
  </div>
</section>

<section>
  <h2>End round</h2>
  <div class="row">
    <label>Score 1
      <input id="s1" type="number" step="0.01" value="8.2">
    </label>
    <label>Score 2
      <input id="s2" type="number" step="0.01" value="7.5">
    </label>
  </div>
  <button id="btn-end">End</button>
</section>

<section>
  <h2>Game state</h2>
  <button class="secondary" id="btn-state">Get state</button>
  <button class="danger" id="btn-reset">Reset to Idle</button>
  <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid #2a2a2a;">
    <button id="btn-full-round">Run full round</button>
    <div style="color:#707070;font-size:0.8rem;margin-top:0.5rem;">
      <strong>Demo only.</strong> Fires <code>start</code> &rarr; wait 2s &rarr;
      <code>end</code> &rarr; wait 2s &rarr; <code>reset</code>, using the inputs
      above. Not intended for a live round.
    </div>
  </div>
</section>

<section>
  <h2>Players</h2>
  <button class="secondary" id="btn-list">List</button>
  <button id="btn-send-prompt">Send prompt to all</button>
  <button id="btn-times-up">Times up all</button>
  <button class="secondary" id="btn-clear">Reset round all</button>
  <div class="muted">Ready players: <span id="ready-players">none</span></div>
</section>

<section>
  <h2>Log</h2>
  <div id="log"></div>
</section>

<script>
const logEl = document.getElementById('log');
function log(msg, isErr) {
  const line = document.createElement('div');
  if (isErr) line.className = 'err';
  const ts = new Date().toISOString().slice(11, 19);
  line.textContent = `[${ts}] ${msg}`;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

async function api(path, body, options = {}) {
  try {
    const opts = { headers: { 'content-type': 'application/json' } };
    if (body !== undefined) { opts.method = 'POST'; opts.body = JSON.stringify(body); }
    const res = await fetch(path, opts);
    const text = await res.text();
    let json = null;
    try { json = JSON.parse(text); } catch (_) {}
    if (!options.quiet) log(`${res.status} ${path} → ${text}`, !res.ok);
    return { ok: res.ok, text, json };
  } catch (e) {
    log(`ERR ${path} → ${e}`, true);
    return { ok: false, text: String(e), json: null };
  }
}

function val(id) { return document.getElementById(id).value.trim(); }
function num(id) { return parseFloat(document.getElementById(id).value); }
function playerName(id) {
  const element = document.getElementById(id);
  return element.dataset.name || '';
}
function promptValue() {
  const input = document.getElementById('prompt');
  const prompt = input.value.trim();
  input.setCustomValidity(prompt ? '' : 'Prompt must not be empty.');
  if (!prompt) {
    input.reportValidity();
    log('prompt is required', true);
    return null;
  }
  return prompt;
}
document.getElementById('prompt').addEventListener('input', (event) => {
  if (event.target.value.trim()) event.target.setCustomValidity('');
});

let lastReadyNames = [];
let roundStartedTimer = null;
let canStartRound = false;
function setStartError(message) {
  document.getElementById('start-error').textContent = message;
}
function showRoundStarted() {
  const badge = document.getElementById('round-started');
  badge.classList.add('visible');
  setStartError('');
  if (roundStartedTimer !== null) clearTimeout(roundStartedTimer);
  roundStartedTimer = setTimeout(() => {
    badge.classList.remove('visible');
    roundStartedTimer = null;
  }, 3500);
}
function hideRoundStarted() {
  const badge = document.getElementById('round-started');
  badge.classList.remove('visible');
  if (roundStartedTimer !== null) {
    clearTimeout(roundStartedTimer);
    roundStartedTimer = null;
  }
}
function orderedReadyEntries(ready) {
  if (!ready) return [];
  return Object.entries(ready).map(([port, name]) => ({ port, name }));
}
function renderReady(ready, modelReady) {
  const entries = orderedReadyEntries(ready);
  const names = entries.map((entry) => entry.name);
  const modelReadyPorts = new Set(modelReady || []);
  document.getElementById('ready-players').textContent =
    names.length ? names.join(', ') : 'none';
  ['c1', 'c2'].forEach((id, index) => {
    const slot = document.getElementById(id);
    const next = names[index] || '';
    slot.dataset.name = next;
    slot.textContent = next || 'Waiting for CLI player';
    slot.classList.toggle('empty', !next);
    const badge = document.getElementById(`${id}-ready`);
    const entry = entries[index];
    const isReady = Boolean(entry && modelReadyPorts.has(entry.port));
    badge.textContent = isReady ? 'Ready' : 'Waiting for model';
    badge.classList.toggle('visible', Boolean(entry));
    badge.classList.toggle('ready', isReady);
    badge.classList.toggle('waiting', Boolean(entry && !isReady));
  });
  const roundPorts = entries.slice(0, 2).map((entry) => entry.port);
  canStartRound = roundPorts.length === 2
    && roundPorts.every((port) => modelReadyPorts.has(port));
  document.getElementById('btn-start').setAttribute(
    'aria-disabled',
    String(!canStartRound),
  );
  if (canStartRound) setStartError('');
  lastReadyNames = names;
}
async function refreshPlayers() {
  const result = await api('/api/players', undefined, { quiet: true });
  if (result.ok && result.json) renderReady(result.json.ready, result.json.model_ready);
}

document.getElementById('btn-start').onclick = async () => {
  hideRoundStarted();
  const prompt = promptValue();
  if (prompt === null) return;
  await refreshPlayers();
  const c1 = playerName('c1');
  const c2 = playerName('c2');
  const contestants = [c1, c2].filter(Boolean);
  if (!canStartRound) {
    const message = 'Both players must select a model before the round can start.';
    setStartError(message);
    log(message, true);
    return;
  }
  if (contestants.length === 0) { log('at least one player is required', true); return; }
  const result = await api('/api/round/start', { prompt, contestants });
  if (result.ok) {
    showRoundStarted();
    return;
  }
  if (result.json && result.json.detail) {
    setStartError(String(result.json.detail));
  }
};

document.getElementById('btn-end').onclick = () => {
  const c1 = playerName('c1');
  const c2 = playerName('c2');
  const scores = {};
  if (c1) scores[c1] = num('s1');
  if (c2) scores[c2] = num('s2');
  api('/api/round/end', { scores });
};

document.getElementById('btn-state').onclick = () => api('/api/state');
document.getElementById('btn-reset').onclick = () => api('/api/round/reset', {});

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

document.getElementById('btn-full-round').onclick = async () => {
  const btn = document.getElementById('btn-full-round');
  btn.disabled = true;
  try {
    const prompt = promptValue();
    if (prompt === null) return;
    const c1 = playerName('c1');
    const c2 = playerName('c2');
    const contestants = [c1, c2].filter(Boolean);
    if (contestants.length === 0) {
      log('at least one player is required', true);
      return;
    }
    const scores = {};
    if (c1) scores[c1] = num('s1');
    if (c2) scores[c2] = num('s2');

    log('full round: start');
    let r = await api('/api/round/start', { prompt, contestants });
    if (!r.ok) return;
    await sleep(2000);

    log('full round: end');
    r = await api('/api/round/end', { scores });
    if (!r.ok) return;
    await sleep(2000);

    log('full round: reset');
    await api('/api/round/reset', {});
  } finally {
    btn.disabled = false;
  }
};

document.getElementById('btn-list').onclick = async () => {
  const result = await api('/api/players');
  if (result.ok && result.json) renderReady(result.json.ready, result.json.model_ready);
};
document.getElementById('btn-send-prompt').onclick = () => {
  const prompt = promptValue();
  if (prompt === null) return;
  api('/api/players/prompt', { all: true, prompt });
};
document.getElementById('btn-times-up').onclick = () => api('/api/players/times-up', { all: true });
document.getElementById('btn-clear').onclick = () => api('/api/players/clear', { all: true });
refreshPlayers();
setInterval(refreshPlayers, 2000);
</script>
</body>
</html>
"""


def create_app() -> FastAPI:
    """Build the control-panel FastAPI app."""
    app = FastAPI(title="Vibe Olympics Control Panel")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _INDEX_HTML

    @app.get("/api/state")
    async def get_state() -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{VIBE_OBS_API}/state")
            except httpx.HTTPError as exc:
                msg = f"OBS runner at {VIBE_OBS_API} unreachable: {exc}"
                raise HTTPException(status_code=502, detail=msg) from exc
        return response.json()

    @app.post("/api/round/start")
    async def round_start(req: StartRequest) -> dict[str, Any]:
        if not _all_named_players_model_ready():
            raise HTTPException(status_code=409, detail=_start_blocked_message())
        contestants = req.contestants or _ready_contestants()[:2]
        state = await _forward(
            "start",
            {"prompt": req.prompt, "contestants": contestants},
        )
        sent = await iterm_ctrl.send_prompt_to_players(None, req.prompt)
        return {"state": state, "prompt_sent": sent}

    @app.post("/api/round/end")
    async def round_end(req: EndRequest) -> dict[str, Any]:
        return await _forward("end", {"scores": req.scores})

    @app.post("/api/round/reset")
    async def round_reset() -> dict[str, Any]:
        _ready_players.clear()
        _model_ready_ports.clear()
        return await _forward("reset", {})

    @app.get("/api/players")
    async def players_list() -> dict[str, Any]:
        return {
            "players": await iterm_ctrl.list_players(),
            "ready": dict(_ready_players),
            "model_ready": sorted(_model_ready_ports),
        }

    @app.post("/api/players/ready")
    async def players_ready(req: PlayerReadyRequest) -> dict[str, Any]:
        _ready_players[req.port] = req.name
        obs_state: dict[str, Any] | None = None
        obs_error: str | None = None
        try:
            obs_state = await _forward(
                "ready",
                {"contestants": _ready_contestants()},
            )
        except HTTPException as exc:
            obs_error = str(exc.detail)
            logger.warning("Could not forward ready players to OBS: %s", exc.detail)
        return {
            "ready": dict(_ready_players),
            "model_ready": sorted(_model_ready_ports),
            "contestants": _ready_contestants(),
            "obs": obs_state,
            "obs_error": obs_error,
        }

    @app.post("/api/players/model-ready")
    async def players_model_ready(req: PlayerModelReadyRequest) -> dict[str, Any]:
        _model_ready_ports.add(req.port)
        players_ready_sent: list[str] = []
        if _all_named_players_model_ready():
            players_ready_sent = await iterm_ctrl.players_ready(None)
        return {
            "ready": dict(_ready_players),
            "model_ready": sorted(_model_ready_ports),
            "players_ready_sent": players_ready_sent,
        }

    @app.post("/api/players/prompt")
    async def players_prompt(req: PlayerPromptRequest) -> dict[str, list[str]]:
        ports = _resolve_ports(req)
        return {"sent": await iterm_ctrl.send_prompt_to_players(ports, req.prompt)}

    @app.post("/api/players/times-up")
    async def players_times_up(target: PlayerTarget) -> dict[str, list[str]]:
        ports = _resolve_ports(target)
        return {"sent": await iterm_ctrl.times_up_players(ports)}

    @app.post("/api/players/clear")
    async def players_clear(target: PlayerTarget) -> dict[str, Any]:
        ports = _resolve_ports(target)
        cleared = await iterm_ctrl.clear_players(ports)
        _clear_player_readiness(ports)
        obs_state: dict[str, Any] | None = None
        obs_error: str | None = None
        try:
            obs_state = await _forward("reset", {})
        except HTTPException as exc:
            obs_error = str(exc.detail)
            logger.warning("Could not forward cleared players to OBS: %s", exc.detail)
        return {
            "cleared": cleared,
            "ready": dict(_ready_players),
            "model_ready": sorted(_model_ready_ports),
            "obs": obs_state,
            "obs_error": obs_error,
        }

    @app.post("/api/players/reset")
    async def players_reset(target: PlayerTarget) -> dict[str, list[str]]:
        ports = _resolve_ports(target)
        _clear_player_readiness(ports)
        return {"reset": await iterm_ctrl.reset_players(ports)}

    return app


app = create_app()
