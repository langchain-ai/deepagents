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
from pydantic import BaseModel, Field

from control_server import iterm_ctrl

logger = logging.getLogger(__name__)

VIBE_OBS_API = os.environ.get("VIBE_OBS_API", "http://localhost:8765").rstrip("/")


class StartRequest(BaseModel):
    prompt: str
    contestants: list[str] = Field(default_factory=list)


class EndRequest(BaseModel):
    scores: dict[str, float] = Field(default_factory=dict)


class PlayerTarget(BaseModel):
    """Target a single player by port or every active player."""

    port: str | None = None
    all: bool = False


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
  button.danger { background: #dc2626; }
  button.secondary { background: #525252; }
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
    <input id="prompt" type="text" placeholder="build a cat shrine">
  </label>
  <div class="row">
    <label>Contestant 1
      <input id="c1" type="text" value="Alice">
    </label>
    <label>Contestant 2
      <input id="c2" type="text" value="Bob">
    </label>
  </div>
  <button id="btn-start">Start</button>
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
  <button class="secondary" id="btn-clear">Clear all</button>
  <button class="danger" id="btn-reset-players">Reset all</button>
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

async function api(path, body) {
  try {
    const opts = { headers: { 'content-type': 'application/json' } };
    if (body !== undefined) { opts.method = 'POST'; opts.body = JSON.stringify(body); }
    const res = await fetch(path, opts);
    const text = await res.text();
    log(`${res.status} ${path} → ${text}`, !res.ok);
    return { ok: res.ok, text };
  } catch (e) {
    log(`ERR ${path} → ${e}`, true);
    return { ok: false, text: String(e) };
  }
}

function val(id) { return document.getElementById(id).value.trim(); }
function num(id) { return parseFloat(document.getElementById(id).value); }

document.getElementById('btn-start').onclick = () => {
  const prompt = val('prompt');
  const c1 = val('c1');
  const c2 = val('c2');
  const contestants = [c1, c2].filter(Boolean);
  if (!prompt) { log('prompt is required', true); return; }
  if (contestants.length === 0) { log('at least one contestant is required', true); return; }
  api('/api/round/start', { prompt, contestants });
};

document.getElementById('btn-end').onclick = () => {
  const c1 = val('c1');
  const c2 = val('c2');
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
    const prompt = val('prompt');
    const c1 = val('c1');
    const c2 = val('c2');
    const contestants = [c1, c2].filter(Boolean);
    if (!prompt || contestants.length === 0) {
      log('prompt and at least one contestant are required', true);
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

document.getElementById('btn-list').onclick = () => api('/api/players');
document.getElementById('btn-clear').onclick = () => api('/api/players/clear', { all: true });
document.getElementById('btn-reset-players').onclick = () => api('/api/players/reset', { all: true });
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
        return await _forward(
            "start",
            {"prompt": req.prompt, "contestants": req.contestants},
        )

    @app.post("/api/round/end")
    async def round_end(req: EndRequest) -> dict[str, Any]:
        return await _forward("end", {"scores": req.scores})

    @app.post("/api/round/reset")
    async def round_reset() -> dict[str, Any]:
        return await _forward("reset", {})

    @app.get("/api/players")
    async def players_list() -> dict[str, list[str]]:
        return {"players": await iterm_ctrl.list_players()}

    @app.post("/api/players/clear")
    async def players_clear(target: PlayerTarget) -> dict[str, list[str]]:
        ports = _resolve_ports(target)
        return {"cleared": await iterm_ctrl.clear_players(ports)}

    @app.post("/api/players/reset")
    async def players_reset(target: PlayerTarget) -> dict[str, list[str]]:
        ports = _resolve_ports(target)
        return {"reset": await iterm_ctrl.reset_players(ports)}

    return app


app = create_app()
