"""Environment-driven configuration for the OBS runner.

All OBS scene/source names ship with sensible defaults so a fresh OBS
profile can be wired up by importing the included scene collection and
naming things to match. Every name is overridable via env var so a
producer with an existing OBS layout does not need to rename anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from obs_runner.state_machine import Phase


def _env(name: str, default: str) -> str:
    """Return env var `name`, stripped — falls back to `default`."""
    value = os.environ.get(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Config:
    """Runtime configuration resolved from the process environment.

    Attributes:
        obs_host: Host running obs-websocket.
        obs_port: obs-websocket port (OBS 28+ default is 4455).
        obs_password: obs-websocket password; empty string disables auth.
        scenes: Phase to OBS scene-name mapping.
        text_prompt: Text input that receives the round prompt.
        contestant_name_template: `{n}`-placeholder pattern for name sources.
        contestant_score_template: `{n}`-placeholder pattern for score sources.
        api_host: Bind host for the FastAPI control server.
        api_port: Bind port for the FastAPI control server.
    """

    obs_host: str = "localhost"
    obs_port: int = 4455
    obs_password: str = ""
    scenes: dict[Phase, str] = field(default_factory=dict)
    text_prompt: str = "PromptText"
    contestant_name_template: str = "Contestant{n}Name"
    contestant_score_template: str = "Contestant{n}Score"
    api_host: str = "127.0.0.1"
    api_port: int = 8765

    def name_source(self, slot: int) -> str:
        """Return the OBS text-source name for slot `n` (1-indexed)."""
        return self.contestant_name_template.format(n=slot)

    def score_source(self, slot: int) -> str:
        """Return the OBS text-source name for slot `n`'s score."""
        return self.contestant_score_template.format(n=slot)


def load_config() -> Config:
    """Read env vars and return an immutable `Config`.

    Env vars:
        `OBS_HOST`, `OBS_PORT`, `OBS_PASSWORD` — connection to obs-websocket.
        `OBS_SCENE_IDLE`, `OBS_SCENE_CODING`, `OBS_SCENE_SCOREBOARD` — scene
            names for each phase.
        `OBS_TEXT_PROMPT` — singular text input name for the prompt.
        `OBS_TEXT_CONTESTANT_NAME_FMT`, `OBS_TEXT_CONTESTANT_SCORE_FMT` —
            `{n}`-placeholder patterns (default `Contestant{n}Name` /
            `Contestant{n}Score`).
        `VIBE_OBS_API_HOST`, `VIBE_OBS_API_PORT` — FastAPI bind.

    Returns:
        Config populated from the environment with defaults applied.
    """
    scenes = {
        Phase.IDLE: _env("OBS_SCENE_IDLE", "Idle"),
        Phase.CODING: _env("OBS_SCENE_CODING", "Coding"),
        Phase.SCOREBOARD: _env("OBS_SCENE_SCOREBOARD", "Scoreboard"),
    }
    return Config(
        obs_host=_env("OBS_HOST", "localhost"),
        obs_port=int(_env("OBS_PORT", "4455")),
        obs_password=os.environ.get("OBS_PASSWORD", ""),
        scenes=scenes,
        text_prompt=_env("OBS_TEXT_PROMPT", "PromptText"),
        contestant_name_template=_env(
            "OBS_TEXT_CONTESTANT_NAME_FMT", "Contestant{n}Name"
        ),
        contestant_score_template=_env(
            "OBS_TEXT_CONTESTANT_SCORE_FMT", "Contestant{n}Score"
        ),
        api_host=_env("VIBE_OBS_API_HOST", "127.0.0.1"),
        api_port=int(_env("VIBE_OBS_API_PORT", "8765")),
    )
