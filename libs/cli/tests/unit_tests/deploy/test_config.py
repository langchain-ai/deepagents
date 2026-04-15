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
    SUBAGENTS_DIRNAME,
    VALID_SANDBOX_PROVIDERS,
    AgentConfig,
    AsyncSubAgentConfig,
    DeployConfig,
    SandboxConfig,
    SubAgentConfig,
    SubAgentProject,
    _parse_config,
    _validate_mcp_for_deploy,
    _validate_model_credentials,
    _validate_sandbox_credentials,
    find_config,
    load_config,
    load_subagents,
    validate_subagent_names,
)

from pathlib import Path

if TYPE_CHECKING:
    pass


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

    def test_description_default(self) -> None:
        cfg = AgentConfig(name="my-agent")
        assert cfg.description == ""

    def test_description_custom(self) -> None:
        cfg = AgentConfig(name="my-agent", description="A helpful bot")
        assert cfg.description == "A helpful bot"

    def test_frozen(self) -> None:
        cfg = AgentConfig(name="x")
        with pytest.raises(AttributeError):
            cfg.name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AsyncSubAgentConfig
# ---------------------------------------------------------------------------


class TestAsyncSubAgentConfig:
    def test_valid_construction(self) -> None:
        cfg = AsyncSubAgentConfig(
            name="researcher", description="Does research", graph_id="researcher"
        )
        assert cfg.name == "researcher"
        assert cfg.description == "Does research"
        assert cfg.graph_id == "researcher"
        assert cfg.url == ""
        assert cfg.headers == {}

    def test_with_url_and_headers(self) -> None:
        cfg = AsyncSubAgentConfig(
            name="researcher",
            description="Does research",
            graph_id="researcher",
            url="https://example.com",
            headers={"Authorization": "Bearer tok"},
        )
        assert cfg.url == "https://example.com"
        assert cfg.headers == {"Authorization": "Bearer tok"}

    def test_frozen(self) -> None:
        cfg = AsyncSubAgentConfig(
            name="x", description="d", graph_id="g"
        )
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

    def test_description_parsed(self) -> None:
        cfg = _parse_config({"agent": {"name": "bot", "description": "A bot"}})
        assert cfg.agent.description == "A bot"

    def test_description_optional(self) -> None:
        cfg = _parse_config({"agent": {"name": "bot"}})
        assert cfg.agent.description == ""

    def test_async_subagents_parsed(self) -> None:
        data: dict[str, Any] = {
            "agent": {"name": "bot"},
            "async_subagents": [
                {
                    "name": "researcher",
                    "description": "Does research",
                    "graph_id": "researcher",
                    "url": "https://example.com",
                    "headers": {"Authorization": "Bearer tok"},
                },
            ],
        }
        cfg = _parse_config(data)
        assert len(cfg.async_subagents) == 1
        sub = cfg.async_subagents[0]
        assert sub.name == "researcher"
        assert sub.description == "Does research"
        assert sub.graph_id == "researcher"
        assert sub.url == "https://example.com"
        assert sub.headers == {"Authorization": "Bearer tok"}

    def test_async_subagents_defaults_empty(self) -> None:
        cfg = _parse_config({"agent": {"name": "bot"}})
        assert cfg.async_subagents == []

    def test_async_subagents_missing_required_field(self) -> None:
        data: dict[str, Any] = {
            "agent": {"name": "bot"},
            "async_subagents": [
                {"name": "researcher", "description": "Does research"},
            ],
        }
        with pytest.raises(ValueError, match="graph_id"):
            _parse_config(data)

    def test_async_subagents_unknown_key(self) -> None:
        data: dict[str, Any] = {
            "agent": {"name": "bot"},
            "async_subagents": [
                {
                    "name": "researcher",
                    "description": "Does research",
                    "graph_id": "researcher",
                    "bogus": "value",
                },
            ],
        }
        with pytest.raises(ValueError, match="Unknown key"):
            _parse_config(data)

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


# ---------------------------------------------------------------------------
# load_subagents
# ---------------------------------------------------------------------------


class TestLoadSubagents:
    @staticmethod
    def _make_subagent(
        parent: Path, name: str, *, description: str = "A subagent"
    ) -> Path:
        """Create a minimal valid subagent directory structure."""
        sub_dir = parent / SUBAGENTS_DIRNAME / name
        sub_dir.mkdir(parents=True, exist_ok=True)
        toml_content = f'[agent]\nname = "{name}"\ndescription = "{description}"\n'
        (sub_dir / DEFAULT_CONFIG_FILENAME).write_text(toml_content, encoding="utf-8")
        (sub_dir / AGENTS_MD_FILENAME).write_text(
            "# Subagent instructions", encoding="utf-8"
        )
        return sub_dir

    def test_no_subagents_dir(self, tmp_path: Path) -> None:
        result = load_subagents(tmp_path)
        assert result == {}

    def test_empty_subagents_dir(self, tmp_path: Path) -> None:
        (tmp_path / SUBAGENTS_DIRNAME).mkdir()
        result = load_subagents(tmp_path)
        assert result == {}

    def test_single_subagent(self, tmp_path: Path) -> None:
        self._make_subagent(tmp_path, "helper")
        result = load_subagents(tmp_path)
        assert len(result) == 1
        assert "helper" in result
        proj = result["helper"]
        assert isinstance(proj, SubAgentProject)
        assert isinstance(proj.config, SubAgentConfig)
        assert proj.config.agent.name == "helper"
        assert proj.config.agent.description == "A subagent"
        assert proj.root == tmp_path / SUBAGENTS_DIRNAME / "helper"

    def test_multiple_subagents(self, tmp_path: Path) -> None:
        self._make_subagent(tmp_path, "alpha", description="First")
        self._make_subagent(tmp_path, "beta", description="Second")
        result = load_subagents(tmp_path)
        assert len(result) == 2
        assert result["alpha"].config.agent.description == "First"
        assert result["beta"].config.agent.description == "Second"

    def test_missing_agents_md_raises(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / SUBAGENTS_DIRNAME / "bad"
        sub_dir.mkdir(parents=True)
        (sub_dir / DEFAULT_CONFIG_FILENAME).write_text(
            '[agent]\nname = "bad"\ndescription = "d"\n', encoding="utf-8"
        )
        with pytest.raises(ValueError, match=r"(?i)AGENTS\.md.*required"):
            load_subagents(tmp_path)

    def test_missing_toml_raises(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / SUBAGENTS_DIRNAME / "bad"
        sub_dir.mkdir(parents=True)
        (sub_dir / AGENTS_MD_FILENAME).write_text("# hi", encoding="utf-8")
        with pytest.raises(ValueError, match=r"(?i)deepagents\.toml.*required"):
            load_subagents(tmp_path)

    def test_missing_description_raises(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / SUBAGENTS_DIRNAME / "nodesc"
        sub_dir.mkdir(parents=True)
        (sub_dir / DEFAULT_CONFIG_FILENAME).write_text(
            '[agent]\nname = "nodesc"\n', encoding="utf-8"
        )
        (sub_dir / AGENTS_MD_FILENAME).write_text("# hi", encoding="utf-8")
        with pytest.raises(ValueError, match=r"(?i)description.*required"):
            load_subagents(tmp_path)

    def test_sandbox_section_rejected(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / SUBAGENTS_DIRNAME / "bad"
        sub_dir.mkdir(parents=True)
        (sub_dir / DEFAULT_CONFIG_FILENAME).write_text(
            '[agent]\nname = "bad"\ndescription = "d"\n\n[sandbox]\nprovider = "none"\n',
            encoding="utf-8",
        )
        (sub_dir / AGENTS_MD_FILENAME).write_text("# hi", encoding="utf-8")
        with pytest.raises(ValueError, match=r"sandbox.*not allowed"):
            load_subagents(tmp_path)

    def test_async_subagents_section_rejected(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / SUBAGENTS_DIRNAME / "bad"
        sub_dir.mkdir(parents=True)
        (sub_dir / DEFAULT_CONFIG_FILENAME).write_text(
            '[agent]\nname = "bad"\ndescription = "d"\n\n'
            '[[async_subagents]]\nname = "x"\ndescription = "x"\ngraph_id = "x"\n',
            encoding="utf-8",
        )
        (sub_dir / AGENTS_MD_FILENAME).write_text("# hi", encoding="utf-8")
        with pytest.raises(ValueError, match=r"async_subagents.*not allowed"):
            load_subagents(tmp_path)

    def test_nested_subagents_rejected(self, tmp_path: Path) -> None:
        sub_dir = self._make_subagent(tmp_path, "outer")
        (sub_dir / SUBAGENTS_DIRNAME).mkdir()
        with pytest.raises(ValueError, match=r"Nested.*not allowed"):
            load_subagents(tmp_path)

    def test_subagent_with_skills(self, tmp_path: Path) -> None:
        sub_dir = self._make_subagent(tmp_path, "skilled")
        (sub_dir / SKILLS_DIRNAME).mkdir()
        result = load_subagents(tmp_path)
        assert "skilled" in result
        assert result["skilled"].root == sub_dir

    def test_subagent_mcp_validated(self, tmp_path: Path) -> None:
        sub_dir = self._make_subagent(tmp_path, "mcpbad")
        mcp = {"mcpServers": {"local": {"type": "stdio", "command": "node"}}}
        (sub_dir / MCP_FILENAME).write_text(json.dumps(mcp), encoding="utf-8")
        with pytest.raises(ValueError, match="stdio"):
            load_subagents(tmp_path)


class TestCrossModuleConsistency:
    def test_sandbox_blocks_matches_valid_providers(self) -> None:
        """SANDBOX_BLOCKS keys in templates.py must match VALID_SANDBOX_PROVIDERS."""
        from deepagents_cli.deploy.templates import SANDBOX_BLOCKS

        assert frozenset(SANDBOX_BLOCKS.keys()) == VALID_SANDBOX_PROVIDERS


# ---------------------------------------------------------------------------
# validate_subagent_names
# ---------------------------------------------------------------------------


class TestValidateSubagentNames:
    def test_no_conflict(self) -> None:
        """One async 'a' and one sync 'b' → no errors."""
        async_subs = [AsyncSubAgentConfig(name="a", description="d", graph_id="g")]
        sync_subs = {
            "b": SubAgentProject(
                config=SubAgentConfig(agent=AgentConfig(name="b", description="d")),
                root=Path("/fake"),
            )
        }
        errors = validate_subagent_names(async_subs, sync_subs)
        assert errors == []

    def test_duplicate_across_sync_and_async(self) -> None:
        """Both async and sync have 'researcher' → 1 error containing 'researcher'."""
        async_subs = [
            AsyncSubAgentConfig(name="researcher", description="d", graph_id="g")
        ]
        sync_subs = {
            "researcher": SubAgentProject(
                config=SubAgentConfig(
                    agent=AgentConfig(name="researcher", description="d")
                ),
                root=Path("/fake"),
            )
        }
        errors = validate_subagent_names(async_subs, sync_subs)
        assert len(errors) == 1
        assert "researcher" in errors[0]

    def test_duplicate_async_names(self) -> None:
        """Two async entries both named 'r' → 1 error containing 'r'."""
        async_subs = [
            AsyncSubAgentConfig(name="r", description="d", graph_id="g1"),
            AsyncSubAgentConfig(name="r", description="d2", graph_id="g2"),
        ]
        sync_subs: dict[str, SubAgentProject] = {}
        errors = validate_subagent_names(async_subs, sync_subs)
        assert len(errors) == 1
        assert "r" in errors[0]
