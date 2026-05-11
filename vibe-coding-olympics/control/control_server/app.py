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
import random
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from control_server import iterm_ctrl, player_dispatch

logger = logging.getLogger(__name__)

VIBE_OBS_API = os.environ.get("VIBE_OBS_API", "http://localhost:8765").rstrip("/")
PLAYER_HEARTBEAT_TIMEOUT_SECS = 6.0

DEFAULT_PROMPTS = (
    "taco truck",
    "haunted house",
    "space tourism company",
    "cat café",
    "time traveler",
    "secret society",
    "deep sea explorer",
    "robot therapist",
    "cloud factory",
    "pirate radio station",
    "noodle shop on Mars",
    "wizard's bookstore",
    "retro arcade",
    "penguin sanctuary",
    "midnight bakery",
    "volcano observatory",
    "time capsule service",
    "dream interpreter",
    "underwater hotel",
    "dragon daycare",
    "haunted food truck",
    "cursed bed and breakfast",
    "fluorescent funeral parlor",
    "gravity-defying yoga studio",
    "gentrified horse ranch",
    "roller disco library",
    "underground napping league",
    "whisper-only podcast network",
    "digital pet adoption drive",
    "slightly illegal lemonade stand",
    "constellation naming rights bureau",
    "backwards-only driving school",
    "museum of stolen socks",
    "glow-in-the-dark botanist",
    "subscription imaginary friend",
    "clown-run call center",
    "lullaby record label",
    "intergalactic lost and found",
    "trampoline-based public transit",
    "powdered water delivery service",
    "retired superhero sidekick agency",
    "smoke-machine weather service",
    "unlicensed astral projection parlor",
    "roped-off dry cleaner lounge",
    "dinosaur Renaissance fair",
    "waterproof firepit installation",
    "yodeling etiquette academy",
    "mobile plastic surgery clinic",
    "multilingual pigeon messaging app",
    "zero-gravity buffet challenge",
    "surrealist tax prep firm",
    "galactic passport photo studio",
    "hand-drawn GPS navigation",
    "philosophical escape room",
    "after-hours postal service",
    "matchmaking service for ghosts",
    "rust-eating robot disposal",
    "tutu-wearing security firm",
    "mood-ring political polling",
    "butter-sculpture exchange",
    "memory-erasing nail salon",
    "tap-dancing chiropractor clinic",
    "sarcasm-translation hotline service",
    "chainmail fashion week",
    "helium-voiced legal defense squad",
    "fern-farming wellness retreat",
    "dew-collecting water sommelier",
    "off-grid time-share resort",
    "silent disco funeral procession",
    "maze-based mattress showroom",
    "tactical pillow fort architect",
    "crystal memory backup service",
    "pneumatic tube messaging company",
    "custom artificial volcano builder",
    "synchronized swimming accountant team",
)


class StartRequest(BaseModel):
    prompt: str | None = None
    contestants: list[str] = Field(default_factory=list)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str | None) -> str | None:
        """Normalize optional prompts supplied by the controller."""
        if value is None:
            return None
        prompt = value.strip()
        return prompt or None


class EndRequest(BaseModel):
    scores: dict[str, float] = Field(default_factory=dict)


class PromptRequest(BaseModel):
    """Prompt pool entry created or edited by the controller."""

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


class PlayerConnectedRequest(BaseModel):
    """Player process reported by the launcher."""

    port: str


class PlayerModelReadyRequest(BaseModel):
    """Player model-selection readiness reported by the CLI hook."""

    port: str


_connected_ports: dict[str, float] = {}
_ready_players: dict[str, str] = {}
_model_ready_ports: set[str] = set()
_prompt_pool: dict[int, str] = dict(enumerate(DEFAULT_PROMPTS, start=1))
_next_prompt_id = len(_prompt_pool) + 1


def _ready_contestants() -> list[str]:
    """Return ready player names in submission order."""
    return list(_ready_players.values())


def _round_player_ports() -> list[str]:
    """Return the two player ports assigned to the current round."""
    return list(_ready_players)[:2]


def _mark_player_connected(port: str) -> None:
    """Record a player heartbeat timestamp."""
    _connected_ports[port] = time.monotonic()


def _connected_player_ports() -> list[str]:
    """Return connected ports, expiring stale player state first."""
    now = time.monotonic()
    stale = [
        port
        for port, seen_at in _connected_ports.items()
        if now - seen_at > PLAYER_HEARTBEAT_TIMEOUT_SECS
    ]
    for port in stale:
        _connected_ports.pop(port, None)
        _ready_players.pop(port, None)
        _model_ready_ports.discard(port)
    return sorted(_connected_ports)


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


async def _get_obs_state() -> dict[str, Any]:
    """Return the OBS runner's current state snapshot."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{VIBE_OBS_API}/state")
        except httpx.HTTPError as exc:
            msg = f"OBS runner at {VIBE_OBS_API} unreachable: {exc}"
            raise HTTPException(status_code=502, detail=msg) from exc
    if response.status_code >= 400:
        try:
            body = response.json()
            detail = body.get("detail", body) if isinstance(body, dict) else body
        except ValueError:
            detail = response.text
        raise HTTPException(status_code=response.status_code, detail=detail)
    return response.json()


async def _end_round_early(scores: dict[str, float]) -> dict[str, Any]:
    """Signal active players that time is up, then end a live round."""
    state = await _get_obs_state()
    phase = state.get("phase")
    if phase != "coding":
        msg = f"Cannot end early while OBS is in phase `{phase}`."
        raise HTTPException(status_code=409, detail=msg)

    ports = _round_player_ports()
    times_up_sent = await player_dispatch.times_up_players(ports or None)
    ended = await _forward("end", {"scores": scores})
    return {
        "state": ended,
        "times_up_sent": times_up_sent,
    }


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


def _prompt_entries() -> list[dict[str, Any]]:
    """Return prompt pool entries in display order."""
    default_count = len(DEFAULT_PROMPTS)

    def sort_key(item: tuple[int, str]) -> tuple[int, int]:
        prompt_id, _prompt = item
        if prompt_id > default_count:
            return (0, -prompt_id)
        return (1, prompt_id)

    return [
        {"id": key, "prompt": value}
        for key, value in sorted(_prompt_pool.items(), key=sort_key)
    ]


def _draw_prompt() -> str:
    """Select one prompt from the pool."""
    if not _prompt_pool:
        msg = "Prompt pool is empty."
        raise HTTPException(status_code=409, detail=msg)
    return random.choice(list(_prompt_pool.values()))


def _round_prompt(req: StartRequest) -> str:
    """Return the supplied prompt or draw one from the pool."""
    return req.prompt or _draw_prompt()


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
  #prompt::placeholder { color: #707070; opacity: 1; }
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
  .port-note { color: #707070; font-size: 0.8rem; }
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
  .ready-badge.offline {
    color: #cbd5e1;
    border: 1px solid #475569;
    background: #111827;
  }
  .ready-badge.connected {
    color: #bfdbfe;
    border: 1px solid #1d4ed8;
    background: #172554;
  }
  .ready-badge.ready {
    color: #86efac;
    border: 1px solid #166534;
    background: #052e16;
  }
  .action-line { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; }
  .row { display: flex; gap: 0.75rem; }
  .row > label { flex: 1; }
  .prompt-pool {
    display: grid;
    gap: 0.5rem;
    margin-top: 0.75rem;
  }
  .prompt-entry {
    display: grid;
    grid-template-columns: 1fr auto auto;
    gap: 0.4rem;
    align-items: center;
    min-height: 2.35rem;
  }
  .prompt-entry.editing { grid-template-columns: 1fr auto auto auto; }
  .prompt-entry input { margin-top: 0; }
  .prompt-text {
    min-width: 0;
    overflow-wrap: anywhere;
    color: #e5e5e5;
  }
  .state-summary {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.35rem 0.75rem;
    margin-top: 0.75rem;
    padding: 0.75rem;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    background: #0a0a0a;
    font-size: 0.88rem;
  }
  .state-summary dt {
    color: #888;
    margin: 0;
  }
  .state-summary dd {
    color: #e5e5e5;
    margin: 0;
    overflow-wrap: anywhere;
  }
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
    <input id="prompt" type="text" placeholder="blank draws from prompt pool">
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
    <button class="secondary" id="btn-draw-prompt">Draw prompt</button>
    <span class="ready-badge ready" id="round-started">Round started</span>
    <span class="inline-error" id="start-error" role="alert"></span>
  </div>
</section>

<section>
  <h2>Prompt pool</h2>
  <div class="action-line">
    <input id="new-prompt" type="text" placeholder="Add a new website prompt">
    <button id="btn-add-prompt">Add</button>
  </div>
  <div class="muted">Leave the round prompt blank to draw randomly from this pool.</div>
  <div class="prompt-pool" id="prompt-pool"></div>
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
  <button class="danger" id="btn-end-early" aria-disabled="true">End early</button>
  <span class="inline-error" id="end-error" role="alert"></span>
</section>

<section>
  <h2>Game state</h2>
  <button class="secondary" id="btn-state">Get state</button>
  <button class="danger" id="btn-reset">Reset to Idle</button>
  <div class="muted">OBS phase: <span id="obs-phase">unknown</span></div>
  <dl class="state-summary" id="state-summary">
    <dt>Prompt</dt><dd id="state-prompt">none</dd>
    <dt>Contestants</dt><dd id="state-contestants">none</dd>
    <dt>Scores</dt><dd id="state-scores">none</dd>
  </dl>
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
  <button id="btn-times-up">Times up all</button>
  <button class="secondary" id="btn-clear">Reset round all</button>
  <div class="muted">Connected players: <span id="connected-players">none</span></div>
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
    if (options.method) opts.method = options.method;
    if (body !== undefined) {
      opts.method = opts.method || 'POST';
      opts.body = JSON.stringify(body);
    }
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
function promptValue(options = {}) {
  const input = document.getElementById('prompt');
  const prompt = input.value.trim();
  if (options.allowBlank && !prompt) {
    input.setCustomValidity('');
    return null;
  }
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
let currentPhase = 'unknown';
function setStartError(message) {
  document.getElementById('start-error').textContent = message;
}
function setEndError(message) {
  document.getElementById('end-error').textContent = message;
}
function renderPhase(phase) {
  currentPhase = phase || 'unknown';
  document.getElementById('obs-phase').textContent = currentPhase;
  document.getElementById('btn-end-early').setAttribute(
    'aria-disabled',
    String(currentPhase !== 'coding'),
  );
  if (currentPhase === 'coding') setEndError('');
}
function renderState(state) {
  if (!state) return;
  if (state.phase) renderPhase(state.phase);
  document.getElementById('state-prompt').textContent = state.prompt || 'none';
  const contestants = state.contestants || [];
  document.getElementById('state-contestants').textContent =
    contestants.length ? contestants.join(', ') : 'none';
  const scores = state.scores || {};
  const scoreEntries = Object.entries(scores);
  document.getElementById('state-scores').textContent = scoreEntries.length
    ? scoreEntries.map(([name, score]) => `${name}: ${score}`).join(', ')
    : 'none';
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
function orderedPlayerEntries(ready, connected) {
  const entries = [];
  const seen = new Set();
  for (const port of connected || []) {
    seen.add(port);
    entries.push({ port, name: ready && ready[port] ? ready[port] : '' });
  }
  for (const [port, name] of Object.entries(ready || {})) {
    if (seen.has(port)) continue;
    entries.push({ port, name });
  }
  return entries.slice(0, 2);
}
function renderReady(ready, modelReady, connected) {
  const entries = orderedPlayerEntries(ready, connected);
  const names = entries.map((entry) => entry.name).filter(Boolean);
  const modelReadyPorts = new Set(modelReady || []);
  const connectedPorts = new Set(connected || []);
  document.getElementById('connected-players').textContent =
    connectedPorts.size ? Array.from(connectedPorts).join(', ') : 'none';
  document.getElementById('ready-players').textContent =
    names.length ? names.join(', ') : 'none';
  ['c1', 'c2'].forEach((id, index) => {
    const slot = document.getElementById(id);
    const entry = entries[index];
    const next = entry ? entry.name : '';
    const isConnected = Boolean(entry && connectedPorts.has(entry.port));
    slot.dataset.name = next;
    slot.textContent = next || (isConnected ? `Connected (${entry.port})` : 'Not connected');
    slot.classList.toggle('empty', !next);
    const badge = document.getElementById(`${id}-ready`);
    const isReady = Boolean(entry && modelReadyPorts.has(entry.port));
    badge.textContent = isReady
      ? 'Ready'
      : isConnected
        ? (next ? 'Waiting for model' : 'Connected')
        : 'Not connected';
    badge.classList.toggle('visible', true);
    badge.classList.toggle('ready', isReady);
    badge.classList.toggle('waiting', Boolean(entry && isConnected && next && !isReady));
    badge.classList.toggle('connected', Boolean(entry && isConnected && !next));
    badge.classList.toggle('offline', !isConnected);
  });
  const roundPorts = entries
    .filter((entry) => entry.name)
    .slice(0, 2)
    .map((entry) => entry.port);
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
  if (result.ok && result.json) {
    renderReady(
      result.json.ready,
      result.json.model_ready,
      result.json.connected,
    );
  }
}
async function refreshState(options = {}) {
  const result = await api('/api/state', undefined, options);
  if (result.ok && result.json) renderState(result.json);
  return result;
}
async function loadPromptPool() {
  const result = await api('/api/prompts', undefined, { quiet: true });
  if (!result.ok || !result.json) return;
  const container = document.getElementById('prompt-pool');
  container.replaceChildren();
  for (const entry of result.json.prompts) {
    const row = document.createElement('div');
    row.className = 'prompt-entry';

    const text = document.createElement('div');
    text.className = 'prompt-text';
    text.textContent = entry.prompt;

    const edit = document.createElement('button');
    edit.className = 'secondary';
    edit.textContent = 'Edit';
    edit.onclick = () => renderPromptEditor(row, entry);

    const remove = document.createElement('button');
    remove.className = 'danger';
    remove.textContent = 'Delete';
    remove.onclick = async () => {
      const response = await api(
        `/api/prompts/${entry.id}`,
        undefined,
        { method: 'DELETE' },
      );
      if (response.ok) loadPromptPool();
    };

    row.append(text, edit, remove);
    container.appendChild(row);
  }
}
function renderPromptEditor(row, entry) {
  row.classList.add('editing');
  row.replaceChildren();

  const input = document.createElement('input');
  input.type = 'text';
  input.value = entry.prompt;

  const save = document.createElement('button');
  save.textContent = 'Save';
  save.onclick = async () => {
    const prompt = input.value.trim();
    if (!prompt) {
      log('prompt is required', true);
      return;
    }
    const response = await api(
      `/api/prompts/${entry.id}`,
      { prompt },
      { method: 'PATCH' },
    );
    if (response.ok) loadPromptPool();
  };

  const cancel = document.createElement('button');
  cancel.className = 'secondary';
  cancel.textContent = 'Cancel';
  cancel.onclick = loadPromptPool;

  const remove = document.createElement('button');
  remove.className = 'danger';
  remove.textContent = 'Delete';
  remove.onclick = async () => {
    const response = await api(
      `/api/prompts/${entry.id}`,
      undefined,
      { method: 'DELETE' },
    );
    if (response.ok) loadPromptPool();
  };

  row.append(input, save, cancel, remove);
  input.focus();
  input.select();
}
async function drawPrompt() {
  const result = await api('/api/prompts/draw');
  if (result.ok && result.json) {
    document.getElementById('prompt').value = result.json.prompt;
    document.getElementById('prompt').setCustomValidity('');
  }
  return result;
}

document.getElementById('btn-start').onclick = async () => {
  hideRoundStarted();
  const prompt = promptValue({ allowBlank: true });
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
  const body = { contestants };
  if (prompt !== null) body.prompt = prompt;
  const result = await api('/api/round/start', body);
  if (result.ok) {
    if (result.json && result.json.state) renderState(result.json.state);
    if (result.json && result.json.prompt) {
      document.getElementById('prompt').value = result.json.prompt;
    }
    showRoundStarted();
    return;
  }
  if (result.json && result.json.detail) {
    setStartError(String(result.json.detail));
  }
};

document.getElementById('btn-end').onclick = () => {
  api('/api/round/end', { scores: roundScores() }).then((result) => {
    if (result.ok && result.json) renderState(result.json);
  });
};

document.getElementById('btn-end-early').onclick = () => {
  if (currentPhase !== 'coding') {
    const message = `No live round to end early. OBS is ${currentPhase}.`;
    setEndError(message);
    log(message, true);
    return;
  }
  api('/api/round/end-early', { scores: roundScores() }).then((result) => {
    if (result.ok && result.json && result.json.state) {
      renderState(result.json.state);
    }
    if (!result.ok && result.json && result.json.detail) {
      setEndError(String(result.json.detail));
    }
  });
};

function roundScores() {
  const c1 = playerName('c1');
  const c2 = playerName('c2');
  const scores = {};
  if (c1) scores[c1] = num('s1');
  if (c2) scores[c2] = num('s2');
  return scores;
}

document.getElementById('btn-state').onclick = () => refreshState();
document.getElementById('btn-reset').onclick = () => {
  api('/api/round/reset', {}).then((result) => {
    if (result.ok && result.json) renderState(result.json);
  });
};
document.getElementById('btn-draw-prompt').onclick = drawPrompt;

async function addPrompt() {
  const input = document.getElementById('new-prompt');
  const prompt = input.value.trim();
  if (!prompt) {
    log('prompt is required', true);
    return;
  }
  const result = await api('/api/prompts', { prompt });
  if (result.ok) {
    input.value = '';
    loadPromptPool();
  }
}
document.getElementById('btn-add-prompt').onclick = addPrompt;
document.getElementById('new-prompt').addEventListener('keydown', (event) => {
  if (event.key !== 'Enter') return;
  event.preventDefault();
  addPrompt();
});

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

document.getElementById('btn-full-round').onclick = async () => {
  const btn = document.getElementById('btn-full-round');
  btn.disabled = true;
  try {
    const prompt = promptValue({ allowBlank: true });
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
    const body = { contestants };
    if (prompt !== null) body.prompt = prompt;
    let r = await api('/api/round/start', body);
    if (!r.ok) return;
    if (r.json && r.json.prompt) {
      document.getElementById('prompt').value = r.json.prompt;
    }
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
  if (result.ok && result.json) {
    renderReady(
      result.json.ready,
      result.json.model_ready,
      result.json.connected,
    );
  }
};
document.getElementById('btn-times-up').onclick = () => api('/api/players/times-up', { all: true });
document.getElementById('btn-clear').onclick = () => {
  api('/api/players/clear', { all: true }).then((result) => {
    if (result.ok && result.json && result.json.obs) renderState(result.json.obs);
  });
};
refreshPlayers();
refreshState({ quiet: true });
loadPromptPool();
setInterval(refreshPlayers, 2000);
setInterval(() => refreshState({ quiet: true }), 2000);
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
        return await _get_obs_state()

    @app.get("/api/prompts")
    async def prompts_list() -> dict[str, list[dict[str, Any]]]:
        return {"prompts": _prompt_entries()}

    @app.post("/api/prompts")
    async def prompts_create(req: PromptRequest) -> dict[str, Any]:
        global _next_prompt_id

        prompt_id = _next_prompt_id
        _next_prompt_id += 1
        _prompt_pool[prompt_id] = req.prompt
        return {"id": prompt_id, "prompt": req.prompt}

    @app.patch("/api/prompts/{prompt_id}")
    async def prompts_update(prompt_id: int, req: PromptRequest) -> dict[str, Any]:
        if prompt_id not in _prompt_pool:
            msg = "Prompt not found."
            raise HTTPException(status_code=404, detail=msg)
        _prompt_pool[prompt_id] = req.prompt
        return {"id": prompt_id, "prompt": req.prompt}

    @app.delete("/api/prompts/{prompt_id}")
    async def prompts_delete(prompt_id: int) -> dict[str, Any]:
        if prompt_id not in _prompt_pool:
            msg = "Prompt not found."
            raise HTTPException(status_code=404, detail=msg)
        prompt = _prompt_pool.pop(prompt_id)
        return {"id": prompt_id, "prompt": prompt}

    @app.get("/api/prompts/draw")
    async def prompts_draw() -> dict[str, str]:
        return {"prompt": _draw_prompt()}

    @app.post("/api/round/start")
    async def round_start(req: StartRequest) -> dict[str, Any]:
        if not _all_named_players_model_ready():
            raise HTTPException(status_code=409, detail=_start_blocked_message())
        prompt = _round_prompt(req)
        contestants = req.contestants or _ready_contestants()[:2]
        state = await _forward(
            "start",
            {"prompt": prompt, "contestants": contestants},
        )
        sent = await player_dispatch.send_prompt_to_players(_round_player_ports(), prompt)
        return {"state": state, "prompt": prompt, "prompt_sent": sent}

    @app.post("/api/round/end")
    async def round_end(req: EndRequest) -> dict[str, Any]:
        return await _forward("end", {"scores": req.scores})

    @app.post("/api/round/end-early")
    async def round_end_early(req: EndRequest) -> dict[str, Any]:
        return await _end_round_early(req.scores)

    @app.post("/api/round/reset")
    async def round_reset() -> dict[str, Any]:
        _ready_players.clear()
        _model_ready_ports.clear()
        return await _forward("reset", {})

    @app.get("/api/players")
    async def players_list() -> dict[str, Any]:
        return {
            "players": await iterm_ctrl.list_players(),
            "connected": _connected_player_ports(),
            "ready": dict(_ready_players),
            "model_ready": sorted(_model_ready_ports),
        }

    @app.post("/api/players/connect")
    async def players_connect(req: PlayerConnectedRequest) -> dict[str, Any]:
        _mark_player_connected(req.port)
        return {
            "connected": _connected_player_ports(),
            "ready": dict(_ready_players),
            "model_ready": sorted(_model_ready_ports),
        }

    @app.post("/api/players/heartbeat")
    async def players_heartbeat(req: PlayerConnectedRequest) -> dict[str, Any]:
        _mark_player_connected(req.port)
        return {
            "connected": _connected_player_ports(),
            "ready": dict(_ready_players),
            "model_ready": sorted(_model_ready_ports),
        }

    @app.post("/api/players/ready")
    async def players_ready(req: PlayerReadyRequest) -> dict[str, Any]:
        _mark_player_connected(req.port)
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
            "connected": _connected_player_ports(),
            "ready": dict(_ready_players),
            "model_ready": sorted(_model_ready_ports),
            "contestants": _ready_contestants(),
            "obs": obs_state,
            "obs_error": obs_error,
        }

    @app.post("/api/players/model-ready")
    async def players_model_ready(req: PlayerModelReadyRequest) -> dict[str, Any]:
        _mark_player_connected(req.port)
        _model_ready_ports.add(req.port)
        players_ready_sent: list[str] = []
        if _all_named_players_model_ready():
            players_ready_sent = await player_dispatch.players_ready(
                _round_player_ports()
            )
        return {
            "connected": _connected_player_ports(),
            "ready": dict(_ready_players),
            "model_ready": sorted(_model_ready_ports),
            "players_ready_sent": players_ready_sent,
        }

    @app.post("/api/players/prompt")
    async def players_prompt(req: PlayerPromptRequest) -> dict[str, list[str]]:
        ports = _resolve_ports(req)
        return {"sent": await player_dispatch.send_prompt_to_players(ports, req.prompt)}

    @app.post("/api/players/times-up")
    async def players_times_up(target: PlayerTarget) -> dict[str, list[str]]:
        ports = _resolve_ports(target)
        return {"sent": await player_dispatch.times_up_players(ports)}

    @app.post("/api/players/clear")
    async def players_clear(target: PlayerTarget) -> dict[str, Any]:
        ports = _resolve_ports(target)
        cleared = await player_dispatch.clear_players(ports)
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
            "connected": _connected_player_ports(),
            "ready": dict(_ready_players),
            "model_ready": sorted(_model_ready_ports),
            "obs": obs_state,
            "obs_error": obs_error,
        }

    return app


app = create_app()
