"""FastAPI control panel: HTML page + JSON endpoints.

Fans one-shot commands to:

- The OBS runner (`POST /transition` at `VIBE_OBS_API`, default
  `http://localhost:8765`) for game-state events.
- iTerm2 player sessions via the helpers in `iterm_ctrl`.

No auth, no persistence, no websockets. Localhost MVP.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from control_server import eval_runner, iterm_ctrl, player_dispatch, site_urls
from control_server.round_timer import (
    RoundTimer,
    TimerWarning,
    timer_warning_for_remaining,
)

logger = logging.getLogger(__name__)

VIBE_OBS_API = os.environ.get("VIBE_OBS_API", "http://localhost:8765").rstrip("/")
STATIC_DIR = Path(__file__).with_name("static")
PLAYER_HEARTBEAT_TIMEOUT_SECS = 6.0
_DEFAULT_ROUND_DURATION_SECS = 300.0
_PLAYER_LAUNCH_COUNTDOWN_SECS = 5.0
"""Seconds the CLI blocks on `LaunchCountdownScreen` after controller start."""


def _round_duration_config() -> tuple[float, str | None]:
    """Return `(duration_secs, warning)` parsed from `VIBE_ROUND_SECONDS`.

    `warning` is non-None when the env var is set but unusable, so the
    operator UI can surface "configured value was rejected, using
    default" rather than silently running a 5-minute round.
    """
    raw = os.environ.get("VIBE_ROUND_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_ROUND_DURATION_SECS, None
    try:
        value = float(raw)
    except ValueError:
        return (
            _DEFAULT_ROUND_DURATION_SECS,
            f"Invalid VIBE_ROUND_SECONDS={raw!r}; using "
            f"{_DEFAULT_ROUND_DURATION_SECS:g}s.",
        )
    if value <= 0:
        return (
            _DEFAULT_ROUND_DURATION_SECS,
            f"VIBE_ROUND_SECONDS={raw!r} must be positive; using "
            f"{_DEFAULT_ROUND_DURATION_SECS:g}s.",
        )
    return value, None


def _round_duration_secs() -> float:
    """Round length in seconds. Override with `VIBE_ROUND_SECONDS`."""
    duration, _ = _round_duration_config()
    return duration


def _timer_warning_payload(warning: TimerWarning | None) -> dict[str, Any] | None:
    """Return JSON metadata for the current timer warning threshold."""
    if warning is None:
        return None
    return {
        "threshold_secs": warning.threshold_secs,
        "message": warning.message,
    }


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


_round_timer = RoundTimer()
_round_context: dict[str, Any] = {}
_round_counter = 0
_last_eval_results: list[dict[str, Any]] = []
_overlay_smoke_state: dict[str, Any] | None = None
_eval_lock: asyncio.Lock | None = None


def _get_eval_lock() -> asyncio.Lock:
    """Return the eval-orchestration lock, lazy-bound to the running loop."""
    global _eval_lock
    if _eval_lock is None:
        _eval_lock = asyncio.Lock()
    return _eval_lock


def _eval_workdir() -> Path:
    """Return the per-process eval results directory, resolved lazily.

    Read on each call (rather than at import time) so tests and
    operators can override `VIBE_EVAL_RESULTS_DIR` after the module has
    already loaded.
    """
    base = os.environ.get("VIBE_EVAL_RESULTS_DIR", "").strip() or tempfile.gettempdir()
    return Path(base) / "vibe-eval-results"


class OverrideScoresRequest(BaseModel):
    """Manual smoke-test override: bypass the judge and force OBS scores.

    Scores are constrained to the OBS scoreboard's 0..10 scale; `ge`/`le`
    bounds also reject `NaN` and `inf` payloads, so a malformed POST is
    a 422 rather than a poisoned scoreboard.
    """

    scores: dict[str, Annotated[float, Field(ge=0, le=10)]] = Field(
        default_factory=dict,
    )


class OverlaySmokeRequest(BaseModel):
    """Controller-only overlay preview state for OBS smoke tests."""

    phase: str
    prompt: str | None = None
    contestants: list[str] = Field(default_factory=list)
    scores: dict[str, Annotated[float, Field(ge=0, le=10)]] = Field(
        default_factory=dict,
    )
    duration_secs: Annotated[float, Field(ge=0)] = 300.0
    remaining_secs: Annotated[float | None, Field(ge=0)] = None
    mode: str = "split"
    focus_player: Annotated[int, Field(ge=1, le=2)] = 1

    @field_validator("phase")
    @classmethod
    def validate_phase(cls, value: str) -> str:
        """Allow only overlay phases the browser source can render."""
        phase = value.strip().lower()
        if phase not in {"idle", "coding", "scoreboard"}:
            msg = "phase must be one of: idle, coding, scoreboard"
            raise ValueError(msg)
        return phase

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        """Allow only overlay layout modes the browser source can render."""
        mode = value.strip().lower()
        if mode not in {"split", "focus"}:
            msg = "mode must be one of: split, focus"
            raise ValueError(msg)
        return mode

    @field_validator("prompt")
    @classmethod
    def validate_smoke_prompt(cls, value: str | None) -> str | None:
        """Normalize optional smoke-test prompts."""
        if value is None:
            return None
        prompt = value.strip()
        return prompt or None

    @field_validator("contestants")
    @classmethod
    def validate_smoke_contestants(cls, value: list[str]) -> list[str]:
        """Keep at most two non-empty smoke-test player names."""
        return [name.strip() for name in value if name.strip()][:2]


def _reset_round_state() -> None:
    """Forget the in-flight round context and the last eval snapshot."""
    _round_context.clear()
    _last_eval_results.clear()


def _ready_contestants() -> list[str]:
    """Return ready player names in submission order."""
    return list(_ready_players.values())


def _clear_overlay_smoke() -> None:
    """Disable the controller-only overlay smoke state."""
    global _overlay_smoke_state
    _overlay_smoke_state = None


def _smoke_contestants(contestants: list[str]) -> list[str]:
    """Return two player names for overlay smoke tests."""
    if contestants:
        return contestants[:2]
    ready = _ready_contestants()[:2]
    if ready:
        return ready
    return ["Ada Lovelace", "Grace Hopper"]


def _smoke_scores(
    contestants: list[str],
    scores: dict[str, float],
) -> dict[str, float]:
    """Return scoreboard scores keyed by smoke-test player names."""
    defaults = [8.6, 7.8]
    return {
        name: float(scores.get(name, defaults[index]))
        for index, name in enumerate(contestants[:2])
    }


def _set_overlay_smoke(req: OverlaySmokeRequest) -> dict[str, Any]:
    """Store and return a fake overlay state for smoke tests."""
    global _overlay_smoke_state

    contestants = _smoke_contestants(req.contestants)
    duration_secs = float(req.duration_secs)
    remaining_secs = (
        float(req.remaining_secs)
        if req.remaining_secs is not None
        else duration_secs
    )
    remaining_secs = min(remaining_secs, duration_secs) if duration_secs else 0.0
    now = time.monotonic()
    _overlay_smoke_state = {
        "phase": req.phase,
        "mode": req.mode,
        "focus_player": req.focus_player,
        "prompt": req.prompt
        or "Build a bold event landing page for a time-traveling taco truck.",
        "contestants": contestants,
        "scores": _smoke_scores(contestants, req.scores),
        "duration_secs": duration_secs,
        "timer_anchor_monotonic": now,
        "timer_anchor_remaining_secs": remaining_secs,
        "timer_started_at": now - max(0.0, duration_secs - remaining_secs),
        "created_at": time.time(),
    }
    state = _overlay_smoke_api_state()
    if state is None:
        msg = "Failed to create overlay smoke state."
        raise HTTPException(status_code=500, detail=msg)
    return state


def _overlay_smoke_api_state() -> dict[str, Any] | None:
    """Return the active fake `/api/state` payload for overlay smoke tests."""
    smoke = _overlay_smoke_state
    if smoke is None:
        return None

    phase = str(smoke["phase"])
    duration_secs = float(smoke["duration_secs"])
    remaining_secs = 0.0
    running = False
    warning: TimerWarning | None = None
    if phase == "coding":
        elapsed = max(0.0, time.monotonic() - float(smoke["timer_anchor_monotonic"]))
        remaining_secs = max(
            0.0,
            float(smoke["timer_anchor_remaining_secs"]) - elapsed,
        )
        running = remaining_secs > 0
        warning = timer_warning_for_remaining(
            duration_secs=duration_secs,
            remaining_secs=remaining_secs,
        )

    scores = dict(smoke["scores"]) if phase == "scoreboard" else {}
    timer = {
        "running": running,
        "duration_secs": duration_secs,
        "remaining_secs": remaining_secs,
        "started_at": smoke["timer_started_at"] if phase == "coding" else None,
        "warning": _timer_warning_payload(warning),
    }
    return {
        "phase": phase,
        "prompt": smoke["prompt"],
        "contestants": list(smoke["contestants"]),
        "scores": scores,
        "obs_error": None,
        "timer": timer,
        "round": {
            "prompt": smoke["prompt"],
            "round_num": None,
            "contestants": list(smoke["contestants"]),
            "started_at": smoke["created_at"],
            "completed_at": None,
            "last_reason": "overlay_smoke",
            "obs_error": None,
            "times_up_error": None,
            "duration_warning": None,
        },
        "eval": {
            "results": [],
        },
        "overlay_smoke": {
            "active": True,
            "mode": smoke["mode"],
            "focus_player": smoke["focus_player"],
        },
    }


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
            response = await client.post(url, json={"event": event, "payload": payload})
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


def _round_player_targets() -> list[dict[str, str]]:
    """Return per-player eval targets captured from current state.

    Each entry has `port`, `name`, and `url`. Players missing a site
    URL (no relay configured, or a malformed relay) are still returned
    so the caller can log a fallback for them. When the resolver
    surfaced a diagnostic reason, it is carried in `url_reason` so the
    eventual fallback can name the actual cause rather than "no URL".
    """
    targets: list[dict[str, str]] = []
    for port in _round_player_ports():
        name = _ready_players.get(port, port)
        resolved = site_urls.resolve(port)
        target: dict[str, str] = {
            "port": port,
            "name": name,
            "url": resolved.url or "",
        }
        if resolved.reason:
            target["url_reason"] = resolved.reason
        targets.append(target)
    return targets


def _eval_to_payload(result: eval_runner.EvalResult, *, port: str) -> dict[str, Any]:
    """Coerce an `EvalResult` to the JSON shape the OBS UI consumes."""
    return {
        "port": port,
        "name": result.site_name,
        "url": result.url,
        "axes": result.axes,
        "overall": result.overall,
        "obs_score": (
            eval_runner.to_obs_score(result.overall)
            if result.overall is not None
            else 0.0
        ),
        "fallback": result.fallback,
        "fallback_reason": result.fallback_reason,
    }


async def _evaluate_one(
    target: dict[str, str],
    *,
    prompt: str,
    round_num: int,
    work_dir: Path,
) -> dict[str, Any]:
    """Run the judge for a single player; coerce to a JSON-friendly dict."""
    port = target["port"]
    name = target["name"]
    url = target["url"]
    if not url:
        reason = target.get("url_reason") or "no site URL configured for port"
        result = eval_runner.EvalResult.fallback_for(
            site_name=name,
            url="",
            prompt=prompt,
            round_num=round_num,
            reason=reason,
        )
    else:
        result = await eval_runner.run_eval(
            url=url,
            site_name=name,
            prompt=prompt,
            round_num=round_num,
            work_dir=work_dir,
        )
    if result.fallback:
        logger.warning(
            "Eval fallback for %s (port %s, url %s): %s",
            name,
            port,
            url or "<none>",
            result.fallback_reason,
        )
    return _eval_to_payload(result, port=port)


async def _run_round_eval(*, reason: str) -> list[dict[str, Any]]:
    """Score the current round's players and forward the end event to OBS.

    Args:
        reason: Diagnostic tag, either `"timer"` or `"end_early"`.

    Returns:
        Per-player eval results (also stored in `_last_eval_results`).
    """
    async with _get_eval_lock():
        if not _round_context:
            logger.info("Skipping eval (%s): no round context.", reason)
            return []
        prompt = str(_round_context.get("prompt") or "")
        round_num = int(_round_context.get("round_num") or 0)
        targets = _round_player_targets()
        if not targets:
            logger.info("Skipping eval (%s): no round players.", reason)
            return []

        workdir = _eval_workdir()
        workdir.mkdir(parents=True, exist_ok=True)
        round_dir = workdir / f"round-{round_num}"

        ports = [t["port"] for t in targets]
        try:
            await player_dispatch.times_up_players(ports or None)
        except Exception as exc:
            logger.exception("Failed to send times-up to players before eval.")
            _round_context["times_up_error"] = repr(exc)
        else:
            _round_context.pop("times_up_error", None)

        raw_results = await asyncio.gather(
            *[
                _evaluate_one(
                    target,
                    prompt=prompt,
                    round_num=round_num,
                    # Isolate per-player so two players with the same
                    # sanitized site name cannot overwrite each other's
                    # judge JSON within a single round.
                    work_dir=round_dir / target["port"],
                )
                for target in targets
            ],
            return_exceptions=True,
        )

        per_site: list[dict[str, Any]] = []
        for target, item in zip(targets, raw_results, strict=True):
            if isinstance(item, BaseException):
                logger.exception(
                    "Eval task crashed for %s (port %s)",
                    target["name"],
                    target["port"],
                    exc_info=item,
                )
                synthetic = eval_runner.EvalResult.fallback_for(
                    site_name=target["name"],
                    url=target["url"],
                    prompt=prompt,
                    round_num=round_num,
                    reason=f"eval crashed: {item!r}",
                )
                per_site.append(_eval_to_payload(synthetic, port=target["port"]))
            else:
                per_site.append(item)

        obs_scores = {entry["name"]: entry["obs_score"] for entry in per_site}

        obs_state: dict[str, Any] | None = None
        obs_error: str | None = None
        try:
            obs_state = await _forward("end", {"scores": obs_scores})
        except HTTPException as exc:
            obs_error = str(exc.detail)
            logger.warning("Could not forward eval scores to OBS: %s", exc.detail)

        _last_eval_results.clear()
        _last_eval_results.extend(per_site)
        _round_context["last_reason"] = reason
        _round_context["obs_state"] = obs_state
        _round_context["obs_error"] = obs_error
        _round_context["completed_at"] = time.time()
        return per_site


async def _start_round_timer(prompt: str, contestants: list[str]) -> None:
    """Arm the server-authoritative timer for a freshly started round."""
    global _round_counter
    _round_counter += 1
    _round_context.clear()
    _last_eval_results.clear()
    _round_context.update(
        {
            "prompt": prompt,
            "contestants": list(contestants),
            "round_num": _round_counter,
            "ports": _round_player_ports(),
            "started_at": time.time(),
            "duration_secs": _round_duration_secs(),
            "start_delay_secs": _PLAYER_LAUNCH_COUNTDOWN_SECS,
        }
    )

    async def _on_expire() -> None:
        logger.info("Round timer expired; auto-evaluating.")
        try:
            await _run_round_eval(reason="timer")
        except Exception:
            logger.exception("Auto-eval on timer expiry failed.")

    await _round_timer.start(
        _round_duration_secs(),
        _on_expire,
        start_delay_secs=_PLAYER_LAUNCH_COUNTDOWN_SECS,
    )


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
  a { color: #93c5fd; }
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
  .timer-display {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
  }
  #timer-clock {
    font-size: 2.4rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: 0.04em;
  }
  .timer-meta {
    color: #94a3b8;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  #timer-bar {
    width: 100%;
    height: 6px;
    appearance: none;
    margin-top: 0.5rem;
    border-radius: 999px;
    overflow: hidden;
    background: #1f2937;
    border: 0;
  }
  #timer-bar::-webkit-progress-bar { background: #1f2937; }
  #timer-bar::-webkit-progress-value { background: #38bdf8; }
  #timer-bar::-moz-progress-bar { background: #38bdf8; }
  .eval-card {
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 0.6rem 0.75rem;
    margin-top: 0.75rem;
    background: #0a0a0a;
  }
  .eval-card.fallback { border-color: #7f1d1d; }
  .eval-card h3 {
    margin: 0 0 0.4rem 0;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
  }
  .eval-axes {
    display: grid;
    grid-template-columns: 9rem 1fr 3rem;
    gap: 0.25rem 0.6rem;
    align-items: center;
    font-size: 0.82rem;
  }
  .eval-axes .axis-bar {
    height: 6px;
    border-radius: 999px;
    background: #1f2937;
    overflow: hidden;
  }
  .eval-axes .axis-fill {
    display: block;
    height: 100%;
    background: #22c55e;
  }
  .eval-card.fallback .axis-fill { background: #f97316; }
  .eval-card .fallback-tag {
    color: #fca5a5;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .eval-overall {
    margin-top: 0.4rem;
    color: #cbd5e1;
    font-size: 0.85rem;
  }
  #override-modal,
  #smoke-modal {
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    background: #141414;
    color: #e5e5e5;
    padding: 1rem 1.25rem;
    max-width: 560px;
    width: 90%;
  }
  #override-modal::backdrop,
  #smoke-modal::backdrop { background: rgba(0, 0, 0, 0.55); }
  .smoke-actions {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.4rem;
    margin-top: 0.65rem;
  }
  #smoke-modal button {
    min-height: 2.2rem;
    margin: 0;
    padding: 0.4rem 0.55rem;
    border-radius: 6px;
    font-size: 0.82rem;
    line-height: 1.15;
  }
  .smoke-actions button {
    width: 100%;
    white-space: nowrap;
  }
  .smoke-command-actions {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) auto;
    gap: 0.4rem;
    align-items: center;
    margin-top: 0.5rem;
  }
  .smoke-command-actions button {
    white-space: nowrap;
  }
  #btn-smoke-cancel {
    min-width: 4.8rem;
  }
  .smoke-layout-actions {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.4rem;
    margin-top: 0.5rem;
  }
  @media (max-width: 560px) {
    .smoke-actions,
    .smoke-command-actions,
    .smoke-layout-actions {
      grid-template-columns: 1fr;
    }
  }
</style>
</head>
<body>
<h1>Vibe Olympics Control</h1>

<section>
  <h2>OBS overlay</h2>
  <div class="muted">
    Add <a id="overlay-link" href="/overlay" target="_blank" rel="noreferrer">/overlay</a>
    as an OBS Browser Source on top of the player feeds.
  </div>
  <div class="action-line">
    <button class="secondary" id="btn-open-smoke">Smoke test overlay…</button>
    <span class="ready-badge waiting" id="smoke-active">Smoke active</span>
  </div>
</section>

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
  <h2>Round timer</h2>
  <div class="timer-display">
    <span id="timer-clock">--:--</span>
    <span class="timer-meta" id="timer-meta">idle</span>
  </div>
  <progress id="timer-bar" max="1" value="0"></progress>
  <div class="muted">When the timer expires, the LLM judge scores both player websites automatically.</div>
</section>

<section>
  <h2>End round</h2>
  <p class="muted">
    Scores are computed automatically when the timer expires. End early to
    stop the round and trigger judging now. Use <em>Override scores</em>
    for smoke tests when you need to bypass the judge.
  </p>
  <button class="danger" id="btn-end-early" aria-disabled="true">End early (trigger judge)</button>
  <button class="secondary" id="btn-open-override">Override scores…</button>
  <span class="inline-error" id="end-error" role="alert"></span>
</section>

<section>
  <h2>Judge results</h2>
  <div class="muted">Latest per-axis evaluation. Fallback rows used randomized scores.</div>
  <div id="eval-results">No results yet.</div>
</section>

<dialog id="override-modal">
  <form method="dialog" id="override-form">
    <h2 style="margin-top:0">Override scores</h2>
    <p class="muted" style="margin-top:0">
      Bypass the judge entirely. Use for smoke tests only — the LLM judge is
      authoritative during live rounds.
    </p>
    <div class="row">
      <label>Score 1
        <input id="s1" type="number" step="0.01" min="0" max="10" value="8.2">
      </label>
      <label>Score 2
        <input id="s2" type="number" step="0.01" min="0" max="10" value="7.5">
      </label>
    </div>
    <div class="action-line">
      <button type="button" id="btn-override-submit">Submit override</button>
      <button type="button" class="secondary" id="btn-override-cancel">Cancel</button>
    </div>
    <span class="inline-error" id="override-error" role="alert"></span>
  </form>
</dialog>

<dialog id="smoke-modal">
  <form method="dialog" id="smoke-form">
    <h2 style="margin-top:0">Overlay smoke test</h2>
    <div class="row">
      <label>Player 1
        <input id="smoke-p1" type="text" value="Ada Lovelace">
      </label>
      <label>Player 2
        <input id="smoke-p2" type="text" value="Grace Hopper">
      </label>
    </div>
    <label>Prompt
      <input id="smoke-prompt" type="text" value="Build a bold event landing page for a time-traveling taco truck">
    </label>
    <div class="row">
      <label>Score 1
        <input id="smoke-s1" type="number" step="0.01" min="0" max="10" value="8.6">
      </label>
      <label>Score 2
        <input id="smoke-s2" type="number" step="0.01" min="0" max="10" value="7.8">
      </label>
    </div>
    <div class="smoke-actions">
      <button type="button" id="btn-smoke-idle">Idle</button>
      <button type="button" id="btn-smoke-coding">Coding</button>
      <button type="button" id="btn-smoke-scoreboard">Scores</button>
      <button type="button" id="btn-smoke-warning-150">2:30 flash</button>
      <button type="button" id="btn-smoke-warning-60">1:00 flash</button>
      <button type="button" id="btn-smoke-warning-30">0:30 flash</button>
    </div>
    <div class="smoke-command-actions">
      <button type="button" class="secondary" id="btn-smoke-tour">Run transition tour</button>
      <button type="button" class="danger" id="btn-smoke-clear">Clear smoke mode</button>
      <button type="button" class="secondary" id="btn-smoke-cancel">Close</button>
    </div>
    <div class="smoke-layout-actions">
      <button type="button" class="secondary" id="btn-smoke-layout-split">Split</button>
      <button type="button" class="secondary" id="btn-smoke-layout-p1">P1 focus</button>
      <button type="button" class="secondary" id="btn-smoke-layout-p2">P2 focus</button>
    </div>
    <span class="inline-error" id="smoke-error" role="alert"></span>
  </form>
</dialog>

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
function clearPromptInput() {
  const input = document.getElementById('prompt');
  input.value = '';
  input.setCustomValidity('');
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
function renderSmokeActive(active) {
  document.getElementById('smoke-active').classList.toggle('visible', Boolean(active));
}
function renderState(state) {
  if (!state) return;
  if (state.phase) renderPhase(state.phase);
  const smokeActive = Boolean(state.overlay_smoke && state.overlay_smoke.active);
  renderSmokeActive(smokeActive);
  if (smokeActive) {
    document.getElementById('btn-end-early').setAttribute('aria-disabled', 'true');
  }
  document.getElementById('state-prompt').textContent = state.prompt || 'none';
  const contestants = state.contestants || [];
  document.getElementById('state-contestants').textContent =
    contestants.length ? contestants.join(', ') : 'none';
  const scores = state.scores || {};
  const scoreEntries = Object.entries(scores);
  document.getElementById('state-scores').textContent = scoreEntries.length
    ? scoreEntries.map(([name, score]) => `${name}: ${score}`).join(', ')
    : 'none';
  renderTimer(state.timer);
  renderEval((state.eval && state.eval.results) || []);
}
function formatClock(seconds) {
  if (!isFinite(seconds) || seconds < 0) return '--:--';
  const total = Math.max(0, Math.round(seconds));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}
function renderTimer(timer) {
  const clock = document.getElementById('timer-clock');
  const meta = document.getElementById('timer-meta');
  const bar = document.getElementById('timer-bar');
  if (!timer || !timer.running) {
    clock.textContent = timer && timer.duration_secs
      ? formatClock(timer.duration_secs)
      : '--:--';
    meta.textContent = 'idle';
    bar.value = 0;
    bar.max = 1;
    return;
  }
  clock.textContent = formatClock(timer.remaining_secs);
  meta.textContent = `of ${formatClock(timer.duration_secs)}`;
  bar.max = timer.duration_secs || 1;
  bar.value = Math.max(0, (timer.duration_secs || 0) - (timer.remaining_secs || 0));
}
function renderEval(results) {
  const container = document.getElementById('eval-results');
  container.replaceChildren();
  if (!results || results.length === 0) {
    const placeholder = document.createElement('div');
    placeholder.className = 'muted';
    placeholder.textContent = 'No results yet.';
    container.appendChild(placeholder);
    return;
  }
  for (const entry of results) {
    const card = document.createElement('div');
    card.className = 'eval-card' + (entry.fallback ? ' fallback' : '');

    const header = document.createElement('h3');
    const label = document.createElement('span');
    label.textContent = `${entry.name || '?'} — ${(entry.obs_score || 0).toFixed(2)} / 10`;
    header.appendChild(label);
    if (entry.fallback) {
      const tag = document.createElement('span');
      tag.className = 'fallback-tag';
      tag.textContent = `fallback: ${entry.fallback_reason || 'unknown'}`;
      header.appendChild(tag);
    }
    card.appendChild(header);

    const axes = document.createElement('div');
    axes.className = 'eval-axes';
    const ordered = [
      'color', 'typography', 'layout', 'content_completeness',
      'creativity', 'interpretation_quality', 'accessibility',
    ];
    for (const axis of ordered) {
      const value = entry.axes ? entry.axes[axis] : null;
      const display = value === null || value === undefined ? 'n/a' : (value * 10).toFixed(1);
      const name = document.createElement('div');
      name.textContent = axis.replace(/_/g, ' ');
      const barWrap = document.createElement('div');
      barWrap.className = 'axis-bar';
      const fill = document.createElement('span');
      fill.className = 'axis-fill';
      fill.style.width = `${Math.round(Math.max(0, Math.min(1, value || 0)) * 100)}%`;
      barWrap.appendChild(fill);
      const num = document.createElement('div');
      num.textContent = display;
      num.style.textAlign = 'right';
      axes.append(name, barWrap, num);
    }
    card.appendChild(axes);

    if (entry.url) {
      const overall = document.createElement('div');
      overall.className = 'eval-overall';
      overall.textContent = entry.url;
      card.appendChild(overall);
    }

    container.appendChild(card);
  }
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
function playerSummary(entries) {
  const labels = entries.map((entry) => (
    entry.name ? `${entry.name} (${entry.port})` : `Connected (${entry.port})`
  ));
  return labels.length ? labels.join(', ') : 'none';
}
function renderReady(ready, modelReady, connected) {
  const entries = orderedPlayerEntries(ready, connected);
  const names = entries.map((entry) => entry.name).filter(Boolean);
  const modelReadyPorts = new Set(modelReady || []);
  const connectedPorts = new Set(connected || []);
  document.getElementById('ready-players').textContent = playerSummary(entries);
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

document.getElementById('btn-end-early').onclick = () => {
  if (currentPhase !== 'coding') {
    const message = `No live round to end early. OBS is ${currentPhase}.`;
    setEndError(message);
    log(message, true);
    return;
  }
  api('/api/round/end-early', {}).then((result) => {
    if (result.ok && result.json && result.json.state) {
      renderState(result.json.state);
      clearPromptInput();
    }
    if (!result.ok && result.json && result.json.detail) {
      setEndError(String(result.json.detail));
    }
  });
};

const overrideModal = document.getElementById('override-modal');
function openOverrideModal() {
  document.getElementById('override-error').textContent = '';
  if (typeof overrideModal.showModal === 'function') overrideModal.showModal();
  else overrideModal.setAttribute('open', '');
}
function closeOverrideModal() {
  if (typeof overrideModal.close === 'function') overrideModal.close();
  else overrideModal.removeAttribute('open');
}
document.getElementById('btn-open-override').onclick = openOverrideModal;
document.getElementById('btn-override-cancel').onclick = closeOverrideModal;
document.getElementById('btn-override-submit').onclick = async () => {
  const c1 = playerName('c1');
  const c2 = playerName('c2');
  const scores = {};
  if (c1) scores[c1] = num('s1');
  if (c2) scores[c2] = num('s2');
  const result = await api('/api/round/override-end', { scores });
  if (result.ok) {
    closeOverrideModal();
    refreshState({ quiet: true });
    return;
  }
  if (result.json && result.json.detail) {
    document.getElementById('override-error').textContent = String(result.json.detail);
  }
};

const smokeModal = document.getElementById('smoke-modal');
let smokeMode = 'split';
let smokeFocusPlayer = 1;
function setSmokeError(message) {
  document.getElementById('smoke-error').textContent = message;
}
function smokeInput(id) {
  return document.getElementById(id).value.trim();
}
function smokeNumber(id, fallback) {
  const value = num(id);
  return Number.isFinite(value) ? value : fallback;
}
function smokeContestants() {
  const p1 = smokeInput('smoke-p1') || playerName('c1') || 'Ada Lovelace';
  const p2 = smokeInput('smoke-p2') || playerName('c2') || 'Grace Hopper';
  return [p1, p2];
}
function smokePrompt() {
  return smokeInput('smoke-prompt')
    || val('prompt')
    || 'Build a bold event landing page for a time-traveling taco truck';
}
function smokeBody(phase, options = {}) {
  const contestants = smokeContestants();
  const scores = {};
  const mode = options.mode || smokeMode;
  const focusPlayer = options.focus_player || smokeFocusPlayer;
  scores[contestants[0]] = smokeNumber('smoke-s1', 8.6);
  scores[contestants[1]] = smokeNumber('smoke-s2', 7.8);
  return {
    phase,
    prompt: smokePrompt(),
    contestants,
    scores,
    duration_secs: 300,
    mode,
    focus_player: focusPlayer,
    ...options,
  };
}
function hydrateSmokeDefaults() {
  const p1 = playerName('c1');
  const p2 = playerName('c2');
  if (p1) document.getElementById('smoke-p1').value = p1;
  if (p2) document.getElementById('smoke-p2').value = p2;
  const prompt = val('prompt');
  if (prompt) document.getElementById('smoke-prompt').value = prompt;
}
function openSmokeModal() {
  hydrateSmokeDefaults();
  setSmokeError('');
  if (typeof smokeModal.showModal === 'function') smokeModal.showModal();
  else smokeModal.setAttribute('open', '');
}
function closeSmokeModal() {
  if (typeof smokeModal.close === 'function') smokeModal.close();
  else smokeModal.removeAttribute('open');
}
async function setOverlaySmoke(phase, options = {}) {
  setSmokeError('');
  if (options.mode) smokeMode = options.mode;
  if (options.focus_player) smokeFocusPlayer = options.focus_player;
  const result = await api('/api/overlay-smoke', smokeBody(phase, options));
  if (result.ok && result.json && result.json.state) {
    renderState(result.json.state);
    return true;
  }
  const message = result.json && result.json.detail
    ? String(result.json.detail)
    : result.text;
  setSmokeError(message);
  return false;
}
async function clearOverlaySmoke() {
  const result = await api('/api/overlay-smoke', undefined, { method: 'DELETE' });
  if (result.ok && result.json && result.json.state) {
    renderState(result.json.state);
    return true;
  }
  return false;
}
async function runSmokeTour() {
  const button = document.getElementById('btn-smoke-tour');
  button.disabled = true;
  try {
    const steps = [
      ['idle', {}],
      ['coding', { remaining_secs: 300 }],
      ['coding', { remaining_secs: 150 }],
      ['coding', { remaining_secs: 60 }],
      ['coding', { remaining_secs: 30 }],
      ['scoreboard', {}],
    ];
    for (const [phase, options] of steps) {
      const ok = await setOverlaySmoke(phase, options);
      if (!ok) return;
      await sleep(1300);
    }
  } finally {
    button.disabled = false;
  }
}
document.getElementById('btn-open-smoke').onclick = openSmokeModal;
document.getElementById('btn-smoke-cancel').onclick = closeSmokeModal;
document.getElementById('btn-smoke-idle').onclick = () => setOverlaySmoke('idle');
document.getElementById('btn-smoke-coding').onclick = () => (
  setOverlaySmoke('coding', { remaining_secs: 300 })
);
document.getElementById('btn-smoke-scoreboard').onclick = () => setOverlaySmoke('scoreboard');
document.getElementById('btn-smoke-warning-150').onclick = () => (
  setOverlaySmoke('coding', { remaining_secs: 150 })
);
document.getElementById('btn-smoke-warning-60').onclick = () => (
  setOverlaySmoke('coding', { remaining_secs: 60 })
);
document.getElementById('btn-smoke-warning-30').onclick = () => (
  setOverlaySmoke('coding', { remaining_secs: 30 })
);
document.getElementById('btn-smoke-layout-split').onclick = () => (
  setOverlaySmoke('coding', { mode: 'split', focus_player: 1, remaining_secs: 300 })
);
document.getElementById('btn-smoke-layout-p1').onclick = () => (
  setOverlaySmoke('coding', { mode: 'focus', focus_player: 1, remaining_secs: 300 })
);
document.getElementById('btn-smoke-layout-p2').onclick = () => (
  setOverlaySmoke('coding', { mode: 'focus', focus_player: 2, remaining_secs: 300 })
);
document.getElementById('btn-smoke-clear').onclick = clearOverlaySmoke;
document.getElementById('btn-smoke-tour').onclick = runSmokeTour;

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

    log('full round: end (override, skipping judge)');
    r = await api('/api/round/override-end', { scores });
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
    if (result.ok && result.json && result.json.obs) {
      renderState(result.json.obs);
      clearPromptInput();
    }
  });
};
refreshPlayers();
refreshState({ quiet: true });
loadPromptPool();
document.getElementById('overlay-link').href = `${window.location.origin}/overlay`;
setInterval(refreshPlayers, 2000);
setInterval(() => refreshState({ quiet: true }), 2000);
</script>
</body>
</html>
"""


_OVERLAY_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vibe Olympics Overlay</title>
<style>
  @font-face {
    font-family: "Aeonik Mono";
    src: url("/static/fonts/aeonik-mono/aeonikmono-regular.woff2") format("woff2");
    font-style: normal;
    font-weight: 400;
    font-display: block;
  }
  @font-face {
    font-family: "Aeonik Mono";
    src: url("/static/fonts/aeonik-mono/aeonikmono-medium.woff2") format("woff2");
    font-style: normal;
    font-weight: 500;
    font-display: block;
  }
  @font-face {
    font-family: "Aeonik Mono";
    src: url("/static/fonts/aeonik-mono/aeonikmono-semibold.woff2") format("woff2");
    font-style: normal;
    font-weight: 700;
    font-display: block;
  }
  @font-face {
    font-family: "IBM Plex Mono";
    src: url("/static/fonts/ibm-plex-mono/IBMPlexMono-Regular.ttf") format("truetype");
    font-style: normal;
    font-weight: 400;
    font-display: block;
  }
  @font-face {
    font-family: "IBM Plex Mono";
    src: url("/static/fonts/ibm-plex-mono/IBMPlexMono-Bold.ttf") format("truetype");
    font-style: normal;
    font-weight: 700;
    font-display: block;
  }
  :root {
    color-scheme: dark;
    --ink: #000;
    --paper: #fff;
    --blue-a: #868cfe;
    --blue-b: #82c9fe;
    --pink-a: #ed92ff;
    --pink-b: #d5c3f7;
    --soft-blue: #d8efff;
    --soft-pink: #f5d8ff;
    --line: max(2px, 0.11vw);
    --status: #ef4444;
  }
  * { box-sizing: border-box; }
  html,
  body {
    width: 100%;
    height: 100%;
    margin: 0;
    overflow: hidden;
    background: transparent !important;
    color: var(--ink);
    font-family: "Aeonik Mono", "IBM Plex Mono", ui-monospace, monospace;
  }
  #app {
    position: fixed;
    inset: 0;
    pointer-events: none;
  }
  .view {
    position: absolute;
    inset: 0;
    display: none;
  }
  .view.active { display: block; }
  .stage {
    position: absolute;
    inset: 0;
    overflow: hidden;
  }
  .split-shell,
  .focus-shell {
    position: absolute;
    inset: 0;
  }
  .split-shell {
    isolation: isolate;
  }
  .focus-shell { display: none; }
  #coding-view.focus .split-shell { display: none; }
  #coding-view.focus .focus-shell { display: block; }
  .gradient-left {
    background: linear-gradient(180deg, var(--blue-a) 0%, var(--blue-b) 42%, var(--paper) 100%);
  }
  .gradient-right {
    background: linear-gradient(180deg, var(--pink-a) 0%, var(--pink-b) 45%, var(--paper) 100%);
  }
  .top-band {
    position: absolute;
    top: 0;
    height: 10.5%;
    border-bottom: var(--line) solid var(--ink);
  }
  .top-band.left { left: 0; width: 50%; }
  .top-band.right { right: 0; width: 50%; }
  .bottom-band {
    position: absolute;
    bottom: 0;
    top: 10.5%;
    display: none;
  }
  .bottom-band.left { left: 0; width: 50%; }
  .bottom-band.right { right: 0; width: 50%; }
  .split-bg {
    position: absolute;
    background-attachment: fixed;
    background-size: 100vw 100vh;
    pointer-events: none;
    z-index: 1;
  }
  .split-bg.left {
    background-image: linear-gradient(180deg, var(--blue-a) 0%, var(--blue-b) 42%, var(--paper) 100%);
  }
  .split-bg.right {
    background-image: linear-gradient(180deg, var(--pink-a) 0%, var(--pink-b) 45%, var(--paper) 100%);
  }
  .split-bg.header-gap {
    top: 10.5%;
    height: 8.3%;
    width: 50%;
  }
  .split-bg.header-gap.left { left: 0; }
  .split-bg.header-gap.right { right: 0; }
  .split-bg.outer-left,
  .split-bg.inner-left,
  .split-bg.inner-right,
  .split-bg.outer-right {
    top: 18.8%;
    bottom: 0;
  }
  .split-bg.outer-left { left: 0; width: 2.2%; }
  .split-bg.inner-left { left: 48.4%; width: 1.6%; }
  .split-bg.inner-right { left: 50%; width: 1.6%; }
  .split-bg.outer-right { right: 0; width: 2.2%; }
  .split-bg.preview-terminal-gap {
    top: 61.8%;
    height: 5.5%;
    width: 46.2%;
  }
  .split-bg.name-preview-gap {
    top: 17.2%;
    height: 1.6%;
    width: 46.2%;
  }
  .split-bg.bottom-gap {
    top: 97%;
    bottom: 0;
    width: 46.2%;
  }
  .split-bg.column-left { left: 2.2%; }
  .split-bg.column-right { right: 2.2%; }
  .divider {
    position: absolute;
    left: 50%;
    top: 0;
    bottom: 0;
    width: var(--line);
    background: var(--ink);
    z-index: 3;
  }
  .chip {
    position: absolute;
    min-width: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 1.2vw;
    overflow: hidden;
    background: var(--ink);
    color: var(--paper);
    clip-path: polygon(5% 0, 95% 0, 100% 50%, 95% 100%, 5% 100%, 0 50%);
    font-weight: 700;
    text-transform: uppercase;
    white-space: nowrap;
    z-index: 4;
  }
  .event-chip {
    left: 50%;
    top: 2.25%;
    width: 30%;
    height: 5.5%;
    transform: translateX(-50%);
    font-size: min(1.65vw, 2.92vh);
    letter-spacing: 0.02em;
  }
  .timer-chip {
    right: 2.2%;
    top: 2.25%;
    width: 7.2%;
    height: 5.5%;
    font-size: min(1.55vw, 2.75vh);
  }
  .timer-chip.timer-warning {
    animation: timer-warning-flash 900ms ease-in-out 3;
  }
  @keyframes timer-warning-flash {
    0%,
    100% {
      background: var(--ink);
      color: var(--paper);
    }
    18%,
    72% {
      background: var(--status);
      color: var(--paper);
    }
  }
  .player-name {
    position: absolute;
    top: 12.3%;
    height: 4.9%;
    display: flex;
    align-items: center;
    padding: 0 1vw;
    background: rgba(255, 255, 255, 0.94);
    border: var(--line) solid var(--ink);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: min(1.35vw, 2.4vh);
    font-weight: 500;
    text-transform: uppercase;
    z-index: 4;
  }
  .player-name.left {
    left: 2.2%;
    width: 46.2%;
  }
  .player-name.right {
    right: 2.2%;
    width: 46.2%;
  }
  .split-event-chip {
    left: 2.2%;
    top: 2.85%;
    width: 27%;
    height: auto;
    justify-content: flex-start;
    padding: 0;
    background: transparent;
    color: var(--ink);
    clip-path: none;
    transform: none;
    font-size: min(1.45vw, 2.58vh);
  }
  .prompt-strip {
    position: absolute;
    left: 2.2%;
    right: 2.2%;
    top: 12.1%;
    min-height: 5.8%;
    display: flex;
    align-items: center;
    gap: 1.25vw;
    padding: 0.65vh 1vw;
    background: rgba(255, 255, 255, 0.94);
    border: var(--line) solid var(--ink);
    font-size: min(1.16vw, 2.06vh);
    font-weight: 500;
    z-index: 4;
  }
  .split-prompt-strip {
    left: 34%;
    right: 14%;
    top: 2.25%;
    min-height: 5.5%;
    justify-content: center;
    padding: 0.65vh 1.2vw;
  }
  .split-prompt-strip span:first-child {
    display: none;
  }
  .prompt-strip span:first-child {
    flex: 0 0 auto;
    font-weight: 700;
    text-transform: uppercase;
  }
  .prompt-text {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .pane {
    position: absolute;
    border: var(--line) solid var(--ink);
    background: transparent;
  }
  .pane-label {
    position: absolute;
    left: 0;
    top: 0;
    padding: 0.62vh 0.72vw;
    background: var(--paper);
    border-right: var(--line) solid var(--ink);
    border-bottom: var(--line) solid var(--ink);
    font-size: min(1.05vw, 1.86vh);
    font-weight: 500;
    line-height: 1.18;
    text-transform: uppercase;
  }
  .split-player.left .pane-label,
  #coding-view[data-focus="1"] .pane-label {
    background: linear-gradient(180deg, var(--paper), var(--soft-blue));
  }
  .split-player.right .pane-label,
  #coding-view[data-focus="2"] .pane-label {
    background: linear-gradient(180deg, var(--paper), var(--soft-pink));
  }
  .terminal-title {
    position: absolute;
    left: 0;
    right: 0;
    top: 0;
    height: 16%;
    display: flex;
    align-items: center;
    padding: 0 1vw;
    background: var(--ink);
    color: var(--paper);
    font-size: min(1.05vw, 1.86vh);
    font-weight: 500;
    text-transform: uppercase;
  }
  .terminal-input {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 16%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-left: 1vw;
    background: rgba(255, 255, 255, 0.92);
    border-top: var(--line) solid var(--ink);
    font-size: min(1.05vw, 1.86vh);
    font-weight: 500;
    text-transform: uppercase;
  }
  .terminal-arrow {
    align-self: stretch;
    width: 8%;
    display: grid;
    place-items: center;
    background: var(--ink);
    color: var(--paper);
    font-size: min(2vw, 3.55vh);
  }
  .split-player {
    position: absolute;
    top: 18.8%;
    bottom: 3%;
    width: 46.2%;
    z-index: 2;
  }
  .split-player.left { left: 2.2%; }
  .split-player.right { right: 2.2%; }
  .split-player .preview {
    left: 0;
    right: 0;
    top: 0;
    height: 55%;
  }
  .split-player .terminal {
    left: 0;
    right: 0;
    bottom: 0;
    height: 38%;
  }
  .focus-top {
    position: absolute;
    left: 0;
    right: 0;
    top: 0;
    height: 10.5%;
    border-bottom: var(--line) solid var(--ink);
  }
  #coding-view[data-focus="1"] .focus-top { background: linear-gradient(90deg, var(--blue-a), var(--blue-b), var(--paper)); }
  #coding-view[data-focus="2"] .focus-top { background: linear-gradient(90deg, var(--pink-a), var(--pink-b), var(--paper)); }
  .focus-name {
    position: absolute;
    left: 2.2%;
    top: 2.85%;
    max-width: 35%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: min(1.9vw, 3.38vh);
    font-weight: 500;
    text-transform: uppercase;
  }
  .focus-website {
    left: 2.2%;
    top: 18.8%;
    bottom: 3%;
    width: 60.5%;
  }
  .focus-terminal {
    right: 2.2%;
    top: 18.8%;
    bottom: 3%;
    width: 33%;
  }
  .focus-prompt {
    left: 2.2%;
    right: 2.2%;
    top: 12.1%;
  }
  .status {
    position: absolute;
    right: 2.2%;
    bottom: 1.3%;
    color: var(--status);
    font-size: min(1vw, 1.78vh);
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0;
  }
  .status.visible { opacity: 1; }
  .full-backdrop {
    position: absolute;
    inset: 0;
    background:
      linear-gradient(90deg, rgba(134, 140, 254, 0.98) 0 50%, rgba(237, 146, 255, 0.98) 50% 100%);
  }
  .idle-card {
    position: absolute;
    left: 8%;
    right: 8%;
    top: 25%;
    min-height: 36%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 2.4vh;
    padding: 5%;
    background: var(--ink);
    color: var(--paper);
    clip-path: polygon(2.5% 0, 97.5% 0, 100% 50%, 97.5% 100%, 2.5% 100%, 0 50%);
  }
  .idle-title {
    font-size: min(5.8vw, 10.31vh);
    line-height: 0.92;
    font-weight: 700;
    text-transform: uppercase;
  }
  .idle-subhead {
    font-size: min(1.85vw, 3.29vh);
    font-weight: 500;
  }
  .score-card {
    position: absolute;
    top: 24%;
    width: 38%;
    height: 48%;
    padding: 3%;
    background: var(--paper);
    border: var(--line) solid var(--ink);
  }
  .score-card.left { left: 7%; }
  .score-card.right { right: 7%; }
  .score-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: min(3.1vw, 5.51vh);
    font-weight: 700;
    text-transform: uppercase;
  }
  .score-num {
    margin-top: 8%;
    font-family: "IBM Plex Mono", "Aeonik Mono", ui-monospace, monospace;
    font-size: min(7.5vw, 13.33vh);
    line-height: 0.95;
    font-weight: 700;
  }
  .score-track {
    height: 6%;
    margin-top: 8%;
    border: var(--line) solid var(--ink);
    background: transparent;
  }
  .score-fill {
    height: 100%;
    width: 0%;
    background: var(--ink);
    transition: width 900ms ease;
  }
  .winner {
    margin-top: 5%;
    font-size: min(1.45vw, 2.58vh);
    font-weight: 700;
    text-transform: uppercase;
    opacity: 0;
  }
  .winner.visible { opacity: 1; }
</style>
</head>
<body>
<div id="app">
  <section class="view active" id="idle-view">
    <div class="stage">
      <div class="full-backdrop"></div>
      <div class="idle-card">
        <div class="idle-title">Vibe Coding Olympics</div>
        <div class="idle-subhead" id="idle-subhead">Waiting for players</div>
      </div>
    </div>
  </section>

  <section class="view split" id="coding-view">
    <div class="stage split-shell">
      <div class="top-band left gradient-left"></div>
      <div class="top-band right gradient-right"></div>
      <div class="bottom-band left gradient-left"></div>
      <div class="bottom-band right gradient-right"></div>
      <div class="split-bg header-gap left"></div>
      <div class="split-bg header-gap right"></div>
      <div class="split-bg outer-left left"></div>
      <div class="split-bg inner-left left"></div>
      <div class="split-bg inner-right right"></div>
      <div class="split-bg outer-right right"></div>
      <div class="split-bg name-preview-gap column-left left"></div>
      <div class="split-bg name-preview-gap column-right right"></div>
      <div class="split-bg preview-terminal-gap column-left left"></div>
      <div class="split-bg preview-terminal-gap column-right right"></div>
      <div class="split-bg bottom-gap column-left left"></div>
      <div class="split-bg bottom-gap column-right right"></div>
      <div class="divider"></div>
      <div class="chip timer-chip" id="split-clock">--:--</div>
      <div class="chip event-chip split-event-chip">Deep Agents PVP</div>
      <div class="prompt-strip split-prompt-strip">
        <span>Prompt</span>
        <span class="prompt-text" id="split-prompt">Waiting for prompt</span>
      </div>
      <div class="player-name left" id="split-p1-name">Player: Player 1</div>
      <div class="player-name right" id="split-p2-name">Player: Player 2</div>
      <div class="split-player left">
        <div class="pane preview">
          <div class="pane-label">P1 Live<br>Preview</div>
        </div>
        <div class="pane terminal">
          <div class="terminal-title">P1 Terminal</div>
          <div class="terminal-input"><span>Input</span><span class="terminal-arrow">&gt;</span></div>
        </div>
      </div>
      <div class="split-player right">
        <div class="pane preview">
          <div class="pane-label">P2 Live<br>Preview</div>
        </div>
        <div class="pane terminal">
          <div class="terminal-title">P2 Terminal</div>
          <div class="terminal-input"><span>Input</span><span class="terminal-arrow">&gt;</span></div>
        </div>
      </div>
    </div>

    <div class="stage focus-shell">
      <div class="focus-top"></div>
      <div class="focus-name" id="focus-name">Player: Player 1</div>
      <div class="chip event-chip">Deep Agents: PVP Speedrun</div>
      <div class="chip timer-chip" id="focus-clock">--:--</div>
      <div class="prompt-strip focus-prompt">
        <span>Prompt</span>
        <span class="prompt-text" id="focus-prompt">Waiting for prompt</span>
      </div>
      <div class="pane focus-website">
        <div class="pane-label" id="focus-preview-label">P1 Live<br>Preview</div>
      </div>
      <div class="pane focus-terminal"></div>
    </div>
  </section>

  <section class="view" id="scoreboard-view">
    <div class="stage">
      <div class="full-backdrop"></div>
      <div class="chip event-chip">Final Scores</div>
      <div id="score-wrap"></div>
    </div>
  </section>

  <div class="status" id="status">Disconnected</div>
</div>

<script>
const state = {
  phase: 'idle',
  prompt: '',
  contestants: [],
  scores: {},
  timer: null,
  lastTimerWarningId: '',
  lastFetch: 0,
  connected: true,
};

const params = new URLSearchParams(window.location.search);
const defaultOverlayMode = params.get('mode') === 'focus' ? 'focus' : 'split';
const defaultFocusIndex = params.get('p') === '2' ? 1 : 0;
let overlayMode = defaultOverlayMode;
let focusIndex = defaultFocusIndex;

const els = {
  idleView: document.getElementById('idle-view'),
  codingView: document.getElementById('coding-view'),
  scoreboardView: document.getElementById('scoreboard-view'),
  idleSubhead: document.getElementById('idle-subhead'),
  splitP1Name: document.getElementById('split-p1-name'),
  splitP2Name: document.getElementById('split-p2-name'),
  splitClock: document.getElementById('split-clock'),
  splitPrompt: document.getElementById('split-prompt'),
  focusName: document.getElementById('focus-name'),
  focusClock: document.getElementById('focus-clock'),
  focusPrompt: document.getElementById('focus-prompt'),
  focusPreviewLabel: document.getElementById('focus-preview-label'),
  scoreWrap: document.getElementById('score-wrap'),
  status: document.getElementById('status'),
};

function active(view) {
  els.idleView.classList.toggle('active', view === 'idle');
  els.codingView.classList.toggle('active', view === 'coding');
  els.scoreboardView.classList.toggle('active', view === 'scoreboard');
}

function text(value, fallback) {
  if (typeof value !== 'string') return fallback;
  const trimmed = value.trim();
  return trimmed || fallback;
}

function playerName(value, fallback) {
  return `Player: ${text(value, fallback)}`;
}

function syncOverlayMode(payload) {
  const smoke = payload.overlay_smoke;
  if (smoke && smoke.active) {
    overlayMode = smoke.mode === 'focus' ? 'focus' : 'split';
    focusIndex = Number(smoke.focus_player) === 2 ? 1 : 0;
    return;
  }
  overlayMode = defaultOverlayMode;
  focusIndex = defaultFocusIndex;
}

function formatClock(seconds) {
  if (!Number.isFinite(seconds)) return '--:--';
  const total = Math.max(0, Math.ceil(seconds));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function currentRemaining() {
  const timer = state.timer;
  if (!timer || !timer.running) return timer ? timer.remaining_secs : NaN;
  const elapsed = (Date.now() - state.lastFetch) / 1000;
  return Math.max(0, Number(timer.remaining_secs || 0) - elapsed);
}

let timerWarningClearHandle = null;

function flashTimerWarning() {
  const chips = [els.splitClock, els.focusClock];
  for (const chip of chips) {
    chip.classList.remove('timer-warning');
    void chip.offsetWidth;
    chip.classList.add('timer-warning');
  }
  if (timerWarningClearHandle !== null) {
    window.clearTimeout(timerWarningClearHandle);
  }
  timerWarningClearHandle = window.setTimeout(() => {
    for (const chip of chips) {
      chip.classList.remove('timer-warning');
    }
    timerWarningClearHandle = null;
  }, 3000);
}

function syncTimerWarning() {
  if (state.phase !== 'coding' || !state.timer || !state.timer.warning) return;
  const threshold = Number(state.timer.warning.threshold_secs);
  if (!Number.isFinite(threshold)) return;
  const startedAt = Number(state.timer.started_at);
  const timerId = Number.isFinite(startedAt) ? startedAt.toFixed(3) : 'unknown';
  const warningId = `${timerId}:${threshold}`;
  if (state.lastTimerWarningId === warningId) return;
  state.lastTimerWarningId = warningId;
  flashTimerWarning();
}

function renderIdle() {
  active('idle');
  const names = state.contestants.filter(Boolean);
  els.idleSubhead.textContent = names.length
    ? `Ready: ${names.join(' vs ')}`
    : 'Waiting for players';
}

function renderCoding() {
  active('coding');
  els.codingView.classList.toggle('focus', overlayMode === 'focus');
  els.codingView.classList.toggle('split', overlayMode !== 'focus');
  els.codingView.dataset.focus = String(focusIndex + 1);

  const p1 = text(state.contestants[0], 'Player 1');
  const p2 = text(state.contestants[1], 'Player 2');
  const prompt = text(state.prompt, 'Waiting for prompt');
  const clock = formatClock(currentRemaining());
  const focused = focusIndex === 0 ? p1 : p2;
  const focusedLabel = `P${focusIndex + 1}`;

  els.splitP1Name.textContent = playerName(p1, 'Player 1');
  els.splitP2Name.textContent = playerName(p2, 'Player 2');
  els.splitPrompt.textContent = prompt;
  els.splitClock.textContent = clock;

  els.focusName.textContent = playerName(focused, `Player ${focusIndex + 1}`);
  els.focusPrompt.textContent = prompt;
  els.focusClock.textContent = clock;
  els.focusPreviewLabel.innerHTML = `${focusedLabel} Live<br>Preview`;
}

function scoreEntries() {
  const names = state.contestants.length
    ? state.contestants
    : Object.keys(state.scores);
  return names.slice(0, 2).map((name) => ({
    name,
    score: Number(state.scores[name] || 0),
  }));
}

function renderScoreboard() {
  active('scoreboard');
  const entries = scoreEntries();
  const winnerScore = Math.max(...entries.map((entry) => entry.score), -1);
  els.scoreWrap.replaceChildren();
  entries.forEach((entry, index) => {
    const card = document.createElement('div');
    card.className = `score-card ${index === 0 ? 'left' : 'right'}`;
    const name = document.createElement('div');
    name.className = 'score-name';
    name.textContent = text(entry.name, 'Player');
    const score = document.createElement('div');
    score.className = 'score-num';
    score.textContent = entry.score.toFixed(2);
    const track = document.createElement('div');
    track.className = 'score-track';
    const fill = document.createElement('div');
    fill.className = 'score-fill';
    track.appendChild(fill);
    const winner = document.createElement('div');
    winner.className = 'winner';
    winner.classList.toggle('visible', entry.score === winnerScore && winnerScore > 0);
    winner.textContent = 'Winner';
    card.append(name, score, track, winner);
    els.scoreWrap.appendChild(card);
    requestAnimationFrame(() => {
      fill.style.width = `${Math.max(0, Math.min(100, entry.score * 10))}%`;
    });
  });
  if (entries.length === 0) {
    const card = document.createElement('div');
    card.className = 'score-card left';
    const name = document.createElement('div');
    name.className = 'score-name';
    name.textContent = 'Waiting for scores';
    card.appendChild(name);
    els.scoreWrap.appendChild(card);
  }
}

function render() {
  els.status.classList.toggle('visible', !state.connected);
  if (state.phase === 'coding') {
    renderCoding();
  } else if (state.phase === 'scoreboard') {
    renderScoreboard();
  } else {
    renderIdle();
  }
}

async function refreshState() {
  try {
    const response = await fetch('/api/state', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    syncOverlayMode(payload);
    state.phase = payload.phase || 'idle';
    state.prompt = payload.prompt || (payload.round && payload.round.prompt) || '';
    state.contestants = Array.isArray(payload.contestants) && payload.contestants.length
      ? payload.contestants
      : ((payload.round && payload.round.contestants) || []);
    state.scores = payload.scores || {};
    state.timer = payload.timer || null;
    state.lastFetch = Date.now();
    state.connected = !payload.obs_error;
    syncTimerWarning();
  } catch (error) {
    state.connected = false;
  }
  render();
}

refreshState();
setInterval(refreshState, 1000);
setInterval(() => {
  if (state.phase === 'coding') renderCoding();
}, 100);
</script>
</body>
</html>
"""


def create_app() -> FastAPI:
    """Build the control-panel FastAPI app."""
    app = FastAPI(title="Vibe Olympics Control Panel")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _INDEX_HTML

    @app.get("/overlay", response_class=HTMLResponse)
    async def overlay() -> str:
        return _OVERLAY_HTML

    @app.get("/api/state")
    async def get_state() -> dict[str, Any]:
        smoke_state = _overlay_smoke_api_state()
        if smoke_state is not None:
            return smoke_state

        obs: dict[str, Any] = {}
        obs_error: dict[str, Any] | None = None
        try:
            obs = await _get_obs_state()
        except HTTPException as exc:
            # Namespaced so a real OBS payload key like "error" or
            # "status" cannot silently shadow the diagnostic.
            obs_error = {"detail": str(exc.detail), "status": exc.status_code}
        snapshot = _round_timer.snapshot()
        _, duration_warning = _round_duration_config()
        return {
            **obs,
            "obs_error": obs_error,
            "timer": {
                "running": snapshot.running,
                "duration_secs": snapshot.duration_secs,
                "remaining_secs": snapshot.remaining_secs,
                "started_at": snapshot.started_at,
                "warning": _timer_warning_payload(snapshot.warning),
            },
            "round": {
                "prompt": _round_context.get("prompt"),
                "round_num": _round_context.get("round_num"),
                "contestants": list(_round_context.get("contestants") or []),
                "started_at": _round_context.get("started_at"),
                "completed_at": _round_context.get("completed_at"),
                "last_reason": _round_context.get("last_reason"),
                "obs_error": _round_context.get("obs_error"),
                "times_up_error": _round_context.get("times_up_error"),
                "duration_warning": duration_warning,
            },
            "eval": {
                "results": list(_last_eval_results),
            },
            "overlay_smoke": {
                "active": False,
            },
        }

    @app.get("/api/eval/last")
    async def get_last_eval() -> dict[str, Any]:
        return {"results": list(_last_eval_results)}

    @app.post("/api/overlay-smoke")
    async def overlay_smoke(req: OverlaySmokeRequest) -> dict[str, Any]:
        return {"state": _set_overlay_smoke(req)}

    @app.delete("/api/overlay-smoke")
    async def overlay_smoke_clear() -> dict[str, Any]:
        _clear_overlay_smoke()
        return {"state": await get_state()}

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
        _clear_overlay_smoke()
        prompt = _round_prompt(req)
        contestants = req.contestants or _ready_contestants()[:2]
        state = await _forward(
            "start",
            {"prompt": prompt, "contestants": contestants},
        )
        sent = await player_dispatch.send_prompt_to_players(
            _round_player_ports(), prompt
        )
        await _start_round_timer(prompt, contestants)
        return {
            "state": state,
            "prompt": prompt,
            "prompt_sent": sent,
            "round_num": _round_context.get("round_num"),
            "duration_secs": _round_context.get("duration_secs"),
        }

    @app.post("/api/round/end")
    async def round_end() -> dict[str, Any]:
        _clear_overlay_smoke()
        await _round_timer.cancel()
        results = await _run_round_eval(reason="end_now")
        return {
            "state": _round_context.get("obs_state"),
            "obs_error": _round_context.get("obs_error"),
            "results": results,
        }

    @app.post("/api/round/end-early")
    async def round_end_early() -> dict[str, Any]:
        _clear_overlay_smoke()
        state = await _get_obs_state()
        phase = state.get("phase")
        if phase != "coding":
            msg = f"Cannot end early while OBS is in phase `{phase}`."
            raise HTTPException(status_code=409, detail=msg)
        await _round_timer.cancel()
        results = await _run_round_eval(reason="end_early")
        return {
            "state": _round_context.get("obs_state"),
            "obs_error": _round_context.get("obs_error"),
            "results": results,
        }

    @app.post("/api/round/override-end")
    async def round_override_end(req: OverrideScoresRequest) -> dict[str, Any]:
        _clear_overlay_smoke()
        await _round_timer.cancel()
        obs_state = await _forward("end", {"scores": req.scores})
        _round_context["last_reason"] = "override"
        _round_context["obs_state"] = obs_state
        _round_context["obs_error"] = None
        _round_context["completed_at"] = time.time()
        return {"state": obs_state, "scores": req.scores}

    @app.post("/api/round/reset")
    async def round_reset() -> dict[str, Any]:
        _clear_overlay_smoke()
        _ready_players.clear()
        _model_ready_ports.clear()
        await _round_timer.cancel()
        _reset_round_state()
        return await _forward("reset", {})

    @app.get("/api/players")
    async def players_list() -> dict[str, Any]:
        connected = _connected_player_ports()
        return {
            "players": await iterm_ctrl.list_players(),
            "connected": connected,
            "ready": dict(_ready_players),
            "model_ready": sorted(_model_ready_ports),
            "site_urls": site_urls.site_urls(_round_player_ports() or connected),
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
        _clear_overlay_smoke()
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
