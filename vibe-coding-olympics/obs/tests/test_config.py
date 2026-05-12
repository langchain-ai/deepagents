from __future__ import annotations

from unittest.mock import patch

from obs_runner.config import load_config
from obs_runner.state_machine import Phase


def test_load_config_defaults_all_phases_to_coding_scene() -> None:
    env = {
        "OBS_SCENE_IDLE": "",
        "OBS_SCENE_CODING": "",
        "OBS_SCENE_SCOREBOARD": "",
    }

    with patch.dict("os.environ", env, clear=False):
        config = load_config()

    assert config.scenes == {
        Phase.IDLE: "coding",
        Phase.CODING: "coding",
        Phase.SCOREBOARD: "coding",
    }
    assert config.contestant_score_template is None


def test_load_config_allows_phase_scene_overrides() -> None:
    env = {
        "OBS_SCENE_IDLE": "Idle",
        "OBS_SCENE_CODING": "Coding",
        "OBS_SCENE_SCOREBOARD": "Scoreboard",
        "OBS_TEXT_CONTESTANT_SCORE_FMT": "Contestant{n}Score",
    }

    with patch.dict("os.environ", env, clear=False):
        config = load_config()

    assert config.scenes == {
        Phase.IDLE: "Idle",
        Phase.CODING: "Coding",
        Phase.SCOREBOARD: "Scoreboard",
    }
    assert config.contestant_score_template == "Contestant{n}Score"
