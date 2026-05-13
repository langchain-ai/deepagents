"""Scene/text-source configuration consumed by the state machine.

These are *domain* concerns of the FSM (which scene/text source each
phase paints), so they live in the control plane next to the machine.
OBS-websocket transport config (host/port/password) stays in the OBS
runner package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from control_server.state_machine import Phase


def _env(name: str, default: str) -> str:
    """Return env var `name`, stripped — falls back to `default`."""
    value = os.environ.get(name, "").strip()
    return value or default


def _optional_env(name: str) -> str | None:
    """Return a stripped env var, or `None` when unset or blank."""
    value = os.environ.get(name, "").strip()
    return value or None


@dataclass(frozen=True)
class StateConfig:
    """Scene names and optional text-source templates.

    Attributes:
        scenes: Phase to OBS scene-name mapping.
        text_prompt: Optional text input that receives the round prompt.
            `None` disables OBS prompt text writes.
        contestant_name_template: Optional `{n}`-placeholder pattern for
            name sources. `None` disables contestant-name text writes.
        contestant_score_template: Optional `{n}`-placeholder pattern for
            score sources. `None` disables score text writes.
    """

    scenes: dict[Phase, str] = field(default_factory=dict)
    text_prompt: str | None = None
    contestant_name_template: str | None = None
    contestant_score_template: str | None = None

    def name_source(self, slot: int) -> str | None:
        """Return the OBS text-source name for slot `n`, if enabled."""
        if self.contestant_name_template is None:
            return None
        return self.contestant_name_template.format(n=slot)

    def score_source(self, slot: int) -> str | None:
        """Return the OBS text-source name for slot `n`'s score, if enabled."""
        if self.contestant_score_template is None:
            return None
        return self.contestant_score_template.format(n=slot)


def load_state_config() -> StateConfig:
    """Read env vars and return an immutable `StateConfig`.

    Env vars:
        `OBS_SCENE_IDLE`, `OBS_SCENE_CODING`, `OBS_SCENE_SCOREBOARD` —
            scene names for each phase. All default to `coding` because
            the browser overlay handles idle/coding/scoreboard visuals
            inside one OBS scene.
        `OBS_TEXT_PROMPT` — optional singular text input for the prompt.
        `OBS_TEXT_CONTESTANT_NAME_FMT`, `OBS_TEXT_CONTESTANT_SCORE_FMT` —
            optional `{n}`-placeholder patterns. OBS text-source writes
            are disabled unless these env vars are set.
    """
    scenes = {
        Phase.IDLE: _env("OBS_SCENE_IDLE", "coding"),
        Phase.CODING: _env("OBS_SCENE_CODING", "coding"),
        Phase.SCOREBOARD: _env("OBS_SCENE_SCOREBOARD", "coding"),
    }
    return StateConfig(
        scenes=scenes,
        text_prompt=_optional_env("OBS_TEXT_PROMPT"),
        contestant_name_template=_optional_env("OBS_TEXT_CONTESTANT_NAME_FMT"),
        contestant_score_template=_optional_env("OBS_TEXT_CONTESTANT_SCORE_FMT"),
    )
