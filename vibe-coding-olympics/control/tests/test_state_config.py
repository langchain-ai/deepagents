"""Tests for `control_server.state_config.load_state_config`."""

from __future__ import annotations

from unittest.mock import patch

from control_server.state_config import load_state_config
from control_server.state_machine import Phase


def test_load_config_defaults_all_phases_to_coding_scene() -> None:
    env = {
        "OBS_SCENE_IDLE": "",
        "OBS_SCENE_CODING": "",
        "OBS_SCENE_SCOREBOARD": "",
        "OBS_TEXT_PROMPT": "",
        "OBS_TEXT_CONTESTANT_NAME_FMT": "",
        "OBS_TEXT_CONTESTANT_SCORE_FMT": "",
    }

    with patch.dict("os.environ", env, clear=False):
        config = load_state_config()

    assert config.scenes == {
        Phase.IDLE: "coding",
        Phase.CODING: "coding",
        Phase.SCOREBOARD: "coding",
    }
    assert config.text_prompt is None
    assert config.contestant_name_template is None
    assert config.contestant_score_template is None


def test_load_config_allows_phase_scene_overrides() -> None:
    env = {
        "OBS_SCENE_IDLE": "Idle",
        "OBS_SCENE_CODING": "Coding",
        "OBS_SCENE_SCOREBOARD": "Scoreboard",
        "OBS_TEXT_PROMPT": "PromptText",
        "OBS_TEXT_CONTESTANT_NAME_FMT": "Contestant{n}Name",
        "OBS_TEXT_CONTESTANT_SCORE_FMT": "Contestant{n}Score",
    }

    with patch.dict("os.environ", env, clear=False):
        config = load_state_config()

    assert config.scenes == {
        Phase.IDLE: "Idle",
        Phase.CODING: "Coding",
        Phase.SCOREBOARD: "Scoreboard",
    }
    assert config.text_prompt == "PromptText"
    assert config.contestant_name_template == "Contestant{n}Name"
    assert config.contestant_score_template == "Contestant{n}Score"
