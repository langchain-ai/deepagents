"""Tests for deploy configuration parsing and validation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from deepagents_cli.deploy.config import (
    AGENTS_MD_FILENAME,
    DEFAULT_CONFIG_FILENAME,
    MCP_FILENAME,
    SKILLS_DIRNAME,
    VALID_SANDBOX_PROVIDERS,
    AgentConfig,
    DeployConfig,
    SandboxConfig,
    _parse_config,
    _validate_mcp_for_deploy,
    _validate_model_credentials,
    _validate_sandbox_credentials,
    find_config,
    load_config,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_valid_construction(self) -> None:
        cfg = AgentConfig(name="my-agent")
        assert cfg.name == "my-agent"
        assert cfg.model == "anthropic:claude-sonnet-4-6"

    def test_custom_model(self) -> None:
        cfg = AgentConfig(name="a", model="openai:gpt-5.3-codex")
        assert cfg.model == "openai:gpt-5.3-codex"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            AgentConfig(name="")

    def test_whitespace_only_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            AgentConfig(name="   ")

    def test_frozen(self) -> None:
        cfg = AgentConfig(name="x")
        with pytest.raises(AttributeError):
            cfg.name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SandboxConfig
# ---------------------------------------------------------------------------


class TestSandboxConfig:
    def test_defaults(self) -> None:
        cfg = SandboxConfig()
        assert cfg.provider == "none"
        assert cfg.template == "deepagents-deploy"
        assert cfg.image == "python:3"
        assert cfg.scope == "thread"

    def test_custom_values(self) -> None:
        cfg = SandboxConfig(
            provider="langsmith",
            template="custom",
            image="node:20",
            scope="assistant",
        )
        assert cfg.provider == "langsmith"
        assert cfg.scope == "assistant"

    def test_frozen(self) -> None:
        cfg = SandboxConfig()
        with pytest.raises(AttributeError):
            cfg.provider = "modal"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DeployConfig
# ---------------------------------------------------------------------------


class TestDeployConfig:
    def test_defaults(self) -> None:
        cfg = DeployConfig(agent=AgentConfig(name="x"))
        assert cfg.sandbox.provider == "none"

    def test_validate_missing_agents_md(self, tmp_path: Path) -> None:
        cfg = DeployConfig(agent=AgentConfig(name="x"))
        errors = cfg.validate(tmp_path)
        assert any(AGENTS_MD_FILENAME in e for e in errors)

    def test_validate_valid_project(self, tmp_path: Path) -> None:
        (tmp_path / AGENTS_MD_FILENAME).write_text("# Agent", encoding="utf-8")
        cfg = DeployConfig(agent=AgentConfig(name="x"))
        # Filter out credential warnings (env-dependent).
        structural = [e for e in cfg.validate(tmp_path) if "API key" not in e]
        assert structural == []

    def test_validate_skills_must_be_dir(self, tmp_path: Path) -> None:
        (tmp_path / AGENTS_MD_FILENAME).write_text("# Agent", encoding="utf-8")
        (tmp_path / SKILLS_DIRNAME).write_text("oops", encoding="utf-8")
        cfg = DeployConfig(agent=AgentConfig(name="x"))
        errors = cfg.validate(tmp_path)
        assert any("must be a directory" in e for e in errors)

    def test_validate_mcp_stdio_rejected(self, tmp_path: Path) -> None:
        (tmp_path / AGENTS_MD_FILENAME).write_text("# Agent", encoding="utf-8")
        mcp = {"mcpServers": {"local": {"type": "stdio", "command": "node"}}}
        (tmp_path / MCP_FILENAME).write_text(json.dumps(mcp), encoding="utf-8")
        cfg = DeployConfig(agent=AgentConfig(name="x"))
        errors = cfg.validate(tmp_path)
        assert any("stdio" in e for e in errors)


# ---------------------------------------------------------------------------
# _parse_config
# ---------------------------------------------------------------------------


class TestParseConfig:
    def test_minimal(self) -> None:
        cfg = _parse_config({"agent": {"name": "bot"}})
        assert cfg.agent.name == "bot"
        assert cfg.agent.model == "anthropic:claude-sonnet-4-6"
        assert cfg.sandbox == SandboxConfig()

    def test_full(self) -> None:
        data: dict[str, Any] = {
            "agent": {"name": "bot", "model": "openai:gpt-5.3-codex"},
            "sandbox": {
                "provider": "daytona",
                "template": "t",
                "image": "img",
                "scope": "assistant",
            },
        }
        cfg = _parse_config(data)
        assert cfg.agent.model == "openai:gpt-5.3-codex"
        assert cfg.sandbox.provider == "daytona"
        assert cfg.sandbox.scope == "assistant"

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValueError, match=r"name.*required"):
            _parse_config({"agent": {}})

    def test_unknown_section_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown section"):
            _parse_config({"agent": {"name": "x"}, "tools": {}})

    def test_unknown_agent_key_raises(self) -> None:
        with pytest.raises(ValueError, match=r"Unknown key.*\[agent\]"):
            _parse_config({"agent": {"name": "x", "timeout": 30}})

    def test_unknown_sandbox_key_raises(self) -> None:
        with pytest.raises(ValueError, match=r"Unknown key.*\[sandbox\]"):
            _parse_config(
                {
                    "agent": {"name": "x"},
                    "sandbox": {"provider": "none", "typo": "val"},
                }
            )

    def test_defaults_come_from_dataclass(self) -> None:
        """Ensure _parse_config without optional keys uses dataclass defaults."""
        cfg = _parse_config({"agent": {"name": "x"}})
        assert cfg.agent.model == AgentConfig(name="x").model
        assert cfg.sandbox == SandboxConfig()


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "missing.toml")

    def test_valid_toml(self, tmp_path: Path) -> None:
        toml = tmp_path / DEFAULT_CONFIG_FILENAME
        toml.write_text(
            '[agent]\nname = "hello"\n',
            encoding="utf-8",
        )
        cfg = load_config(toml)
        assert cfg.agent.name == "hello"

    def test_malformed_toml_raises_valueerror(self, tmp_path: Path) -> None:
        toml = tmp_path / DEFAULT_CONFIG_FILENAME
        toml.write_text("[[[[bad toml", encoding="utf-8")
        with pytest.raises(ValueError, match="Syntax error"):
            load_config(toml)


# ---------------------------------------------------------------------------
# find_config
# ---------------------------------------------------------------------------


class TestFindConfig:
    def test_finds_in_directory(self, tmp_path: Path) -> None:
        (tmp_path / DEFAULT_CONFIG_FILENAME).write_text("", encoding="utf-8")
        result = find_config(tmp_path)
        assert result is not None
        assert result.name == DEFAULT_CONFIG_FILENAME

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert find_config(tmp_path) is None


# ---------------------------------------------------------------------------
# MCP validation
# ---------------------------------------------------------------------------


class TestValidateMcpForDeploy:
    def test_http_allowed(self, tmp_path: Path) -> None:
        mcp = {"mcpServers": {"s": {"type": "http", "url": "http://x"}}}
        p = tmp_path / "mcp.json"
        p.write_text(json.dumps(mcp), encoding="utf-8")
        assert _validate_mcp_for_deploy(p) == []

    def test_stdio_rejected(self, tmp_path: Path) -> None:
        mcp = {"mcpServers": {"s": {"type": "stdio", "command": "node"}}}
        p = tmp_path / "mcp.json"
        p.write_text(json.dumps(mcp), encoding="utf-8")
        errors = _validate_mcp_for_deploy(p)
        assert len(errors) == 1
        assert "stdio" in errors[0]

    def test_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "mcp.json"
        p.write_text("{bad", encoding="utf-8")
        errors = _validate_mcp_for_deploy(p)
        assert len(errors) == 1
        assert "Could not read" in errors[0]


# ---------------------------------------------------------------------------
# Credential validators
# ---------------------------------------------------------------------------


class TestValidateModelCredentials:
    def test_no_colon_skips(self) -> None:
        assert _validate_model_credentials("bare-model") == []

    def test_unknown_provider_skips(self) -> None:
        assert _validate_model_credentials("custom:model") == []

    def test_missing_key_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        errors = _validate_model_credentials("anthropic:claude-sonnet-4-6")
        assert len(errors) == 1
        assert "ANTHROPIC_API_KEY" in errors[0]

    def test_present_key_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert _validate_model_credentials("anthropic:claude-sonnet-4-6") == []


class TestValidateSandboxCredentials:
    def test_unknown_provider_skips(self) -> None:
        assert _validate_sandbox_credentials("none") == []

    def test_missing_key_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
        errors = _validate_sandbox_credentials("daytona")
        assert len(errors) == 1
        assert "DAYTONA_API_KEY" in errors[0]

    def test_any_key_suffices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_SANDBOX_API_KEY", raising=False)
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2-test")
        assert _validate_sandbox_credentials("langsmith") == []


# ---------------------------------------------------------------------------
# Cross-module consistency
# ---------------------------------------------------------------------------


class TestCrossModuleConsistency:
    def test_sandbox_blocks_matches_valid_providers(self) -> None:
        """SANDBOX_BLOCKS keys in templates.py must match VALID_SANDBOX_PROVIDERS."""
        from deepagents_cli.deploy.templates import SANDBOX_BLOCKS

        assert frozenset(SANDBOX_BLOCKS.keys()) == VALID_SANDBOX_PROVIDERS
