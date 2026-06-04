from pathlib import Path

import pytest

from deepagents_talon.config import TalonConfig, TalonConfigError


def test_from_env_creates_assistant_home(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant-1",
            "AGENT_MODEL": "provider:model",
            "UNRELATED_SECRET": "ignored",
        },
        base_home=tmp_path,
    )

    assert config.assistant_id == "assistant-1"
    assert config.model == "provider:model"
    assert config.home == tmp_path / "assistant-1"
    assert config.env == {
        "AGENT_ASSISTANT_ID": "assistant-1",
        "AGENT_MODEL": "provider:model",
    }

    home = config.ensure_home()

    assert home == tmp_path / "assistant-1"
    assert home.stat().st_mode & 0o777 == 0o700
    assert config.manifest_dir.stat().st_mode & 0o777 == 0o700
    assert config.cron_dir.stat().st_mode & 0o777 == 0o700
    assert config.channel_dir.stat().st_mode & 0o777 == 0o700
    assert config.inbound_media_dir.stat().st_mode & 0o777 == 0o700


def test_from_env_keeps_langsmith_env(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant-1",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_API_KEY": "key",
            "LANGSMITH_PROJECT": "project",
        },
        base_home=tmp_path,
    )

    assert config.env["LANGSMITH_TRACING"] == "true"
    assert config.env["LANGSMITH_API_KEY"] == "key"
    assert config.env["LANGSMITH_PROJECT"] == "project"


def test_from_env_keeps_openai_env(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant-1",
            "OPENAI_API_KEY": "key",
            "OPENAI_BASE_URL": "https://openai-compatible.example.com/v1",
        },
        base_home=tmp_path,
    )

    assert config.env["OPENAI_API_KEY"] == "key"
    assert config.env["OPENAI_BASE_URL"] == "https://openai-compatible.example.com/v1"


def test_from_env_keeps_legacy_speech_env(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant-1",
            "SPEECH_ENABLED": "true",
            "SPEECH_DEVICE": "cuda",
        },
        base_home=tmp_path,
    )

    assert config.env["SPEECH_ENABLED"] == "true"
    assert config.env["SPEECH_DEVICE"] == "cuda"


@pytest.mark.parametrize("assistant_id", ["", ".", "..", "../bad", "bad/slash", "bad space"])
def test_from_env_rejects_unsafe_assistant_id(tmp_path: Path, assistant_id: str) -> None:
    with pytest.raises(TalonConfigError):
        TalonConfig.from_env({"AGENT_ASSISTANT_ID": assistant_id}, base_home=tmp_path)
