from pathlib import Path

import pytest

from deepagents_talon.config import TalonConfig, TalonConfigError
from deepagents_talon.tool_approvals import ToolApprovalStore


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
    assert config.manifest_dir == config.home
    assert config.manifest_dir.stat().st_mode & 0o777 == 0o700
    assert config.agents_dir.stat().st_mode & 0o777 == 0o700
    assert config.cron_dir.stat().st_mode & 0o777 == 0o700
    assert config.channel_dir.stat().st_mode & 0o777 == 0o700
    assert config.checkpoint_path == config.home / "checkpoints.sqlite"
    assert config.conversation_state_path == config.home / "conversations.json"
    assert config.inbound_media_dir.stat().st_mode & 0o777 == 0o700


def test_ensure_home_materializes_usable_tool_approvals(tmp_path: Path) -> None:
    config = TalonConfig(assistant_id="assistant-1", home=tmp_path / "assistant-1")

    config.ensure_home()

    snapshot = ToolApprovalStore(config.home / "tools.json").read()
    assert snapshot.approvals == {
        "update_tool_approvals": True,
        "delete_conversations": True,
        "update_mcp_server": True,
        "start_async_task": True,
    }
    assert snapshot.interrupt_on == {
        name: {"allowed_decisions": ["approve", "reject"]} for name in snapshot.approvals
    }


def test_ensure_home_preserves_custom_tool_approvals(tmp_path: Path) -> None:
    config = TalonConfig(assistant_id="assistant-1", home=tmp_path / "assistant-1")
    config.home.mkdir()
    path = config.home / "tools.json"
    raw = b'{ "execute": true, "update_tool_approvals": false }\n'
    path.write_bytes(raw)

    for _ in range(2):
        config.ensure_home()
        assert path.read_bytes() == raw
        assert ToolApprovalStore(path).read().approvals == {
            "execute": True,
            "update_tool_approvals": False,
        }


@pytest.mark.parametrize(
    ("raw", "message"),
    [(b"not-json", "Expecting value"), (b'{"execute": "true"}', "Invalid tool approval entry")],
)
def test_ensure_home_rejects_invalid_tool_approvals(
    tmp_path: Path, raw: bytes, message: str
) -> None:
    config = TalonConfig(assistant_id="assistant-1", home=tmp_path / "assistant-1")
    config.home.mkdir()
    path = config.home / "tools.json"
    path.write_bytes(raw)

    with pytest.raises(ValueError, match=message):
        config.ensure_home()

    assert path.read_bytes() == raw


def test_checkpoint_path_rejects_symlinked_assistant_home(tmp_path: Path) -> None:
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    linked_home = tmp_path / "assistant-1"
    linked_home.symlink_to(real_home, target_is_directory=True)
    config = TalonConfig(assistant_id="assistant-1", home=linked_home)

    with pytest.raises(TalonConfigError, match="checkpoint database"):
        _ = config.checkpoint_path


def test_from_env_defaults_to_deepagents_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "assistant-1"})

    assert config.home == tmp_path / ".deepagents" / "assistant-1"


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


def test_from_env_keeps_runtime_env_vars(tmp_path: Path) -> None:
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant-1",
            "LANGSMITH_TENANT_ID": "tenant",
            "LANGSMITH_ORGANIZATION_ID": "org",
            "LANGSMITH_USER_ID": "user",
            "BUILTIN_MCP_URL": "https://tools.example/mcp",
            "LANGSMITH_HOST_URL": "https://langsmith.example/api",
            "HOST_LANGCHAIN_API_URL": "https://langsmith.example/api-host",
        },
        base_home=tmp_path,
    )

    assert not hasattr(config, "fleet_dir")
    assert config.env["LANGSMITH_TENANT_ID"] == "tenant"
    assert config.env["LANGSMITH_ORGANIZATION_ID"] == "org"
    assert config.env["LANGSMITH_USER_ID"] == "user"
    assert config.env["BUILTIN_MCP_URL"] == "https://tools.example/mcp"
    assert config.env["LANGSMITH_HOST_URL"] == "https://langsmith.example/api"
    assert config.env["HOST_LANGCHAIN_API_URL"] == "https://langsmith.example/api-host"


@pytest.mark.parametrize("assistant_id", ["", ".", "..", "../bad", "bad/slash", "bad space"])
def test_from_env_rejects_unsafe_assistant_id(tmp_path: Path, assistant_id: str) -> None:
    with pytest.raises(TalonConfigError):
        TalonConfig.from_env({"AGENT_ASSISTANT_ID": assistant_id}, base_home=tmp_path)


@pytest.mark.parametrize(
    "env_key",
    ["DEEPAGENTS_TALON_FLEET_DIR", "AGENT_FLEET_DIR", "FLEET_DIR"],
)
def test_from_env_has_no_fleet_dir_field(tmp_path: Path, env_key: str) -> None:
    """Legacy Fleet direct-run env vars no longer configure a Fleet source."""
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "assistant-1",
            env_key: str(tmp_path / "fleet"),
            "AGENT_MODEL": "provider:model",
        },
        base_home=tmp_path,
    )

    assert not hasattr(config, "fleet_dir")
    assert config.model == "provider:model"
    assert config.assistant_id == "assistant-1"
