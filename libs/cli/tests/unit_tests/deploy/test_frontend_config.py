"""Tests for `[frontend]` parsing in deepagents.toml."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import pytest

from deepagents_cli.deploy.config import (
    AgentConfig,
    AuthConfig,
    DeployConfig,
    FrontendConfig,
    _parse_config,
)


def test_frontend_config_defaults():
    fc = FrontendConfig()
    assert fc.enabled is False
    assert fc.app_name is None
    assert fc.subtitle is None
    assert fc.prompts is None


def test_frontend_section_parses_subtitle_and_prompts():
    cfg = _parse_config(
        {
            "agent": {"name": "my-agent"},
            "auth": {"provider": "supabase"},
            "frontend": {
                "enabled": True,
                "subtitle": "Custom subtitle",
                "prompts": ["one", "two", "three"],
            },
        }
    )
    assert cfg.frontend is not None
    assert cfg.frontend.subtitle == "Custom subtitle"
    # Stored as tuple for hashability of the frozen dataclass.
    assert cfg.frontend.prompts == ("one", "two", "three")


def test_frontend_prompts_must_be_list_of_strings():
    with pytest.raises(ValueError, match="must be a list of strings"):
        _parse_config(
            {
                "agent": {"name": "my-agent"},
                "frontend": {"enabled": True, "prompts": [1, 2, 3]},
            }
        )


def test_frontend_section_parses_enabled_true():
    cfg = _parse_config(
        {
            "agent": {"name": "my-agent"},
            "auth": {"provider": "supabase"},
            "frontend": {"enabled": True},
        }
    )
    assert cfg.frontend is not None
    assert cfg.frontend.enabled is True
    assert cfg.frontend.app_name is None


def test_frontend_section_parses_app_name():
    cfg = _parse_config(
        {
            "agent": {"name": "my-agent"},
            "auth": {"provider": "clerk"},
            "frontend": {"enabled": True, "app_name": "My App"},
        }
    )
    assert cfg.frontend is not None
    assert cfg.frontend.app_name == "My App"


def test_frontend_section_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown key"):
        _parse_config(
            {
                "agent": {"name": "my-agent"},
                "auth": {"provider": "supabase"},
                "frontend": {"enabled": True, "theme": "dark"},
            }
        )


def test_frontend_omitted_defaults_to_none():
    cfg = _parse_config({"agent": {"name": "my-agent"}})
    assert cfg.frontend is None


def _write_project(tmp_path: Path) -> Path:
    (tmp_path / "AGENTS.md").write_text("prompt", encoding="utf-8")
    return tmp_path


def test_frontend_enabled_without_auth_validates_clean(tmp_path, monkeypatch):
    """[frontend].enabled with no [auth] is valid (anonymous mode)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    _write_project(tmp_path)
    cfg = DeployConfig(
        agent=AgentConfig(name="a"),
        frontend=FrontendConfig(enabled=True),
    )
    errors = cfg.validate(tmp_path)
    assert errors == [], f"unexpected errors: {errors}"


def test_frontend_disabled_no_auth_is_fine(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    _write_project(tmp_path)
    cfg = DeployConfig(
        agent=AgentConfig(name="a"),
        frontend=FrontendConfig(enabled=False),
    )
    errors = cfg.validate(tmp_path)
    assert not any("[frontend]" in e for e in errors)


def test_frontend_clerk_requires_publishable_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("CLERK_SECRET_KEY", "k")
    monkeypatch.delenv("CLERK_PUBLISHABLE_KEY", raising=False)
    _write_project(tmp_path)
    cfg = DeployConfig(
        agent=AgentConfig(name="a"),
        auth=AuthConfig(provider="clerk"),
        frontend=FrontendConfig(enabled=True),
    )
    errors = cfg.validate(tmp_path)
    assert any("CLERK_PUBLISHABLE_KEY" in e for e in errors)


def test_frontend_supabase_needs_no_extra_env_vars(tmp_path, monkeypatch):
    """Supabase needs no extra env vars beyond [auth].

    Supabase reuses SUPABASE_URL + SUPABASE_PUBLISHABLE_DEFAULT_KEY
    already required by [auth]. No extra VITE_* duplication.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY", "k")
    _write_project(tmp_path)
    cfg = DeployConfig(
        agent=AgentConfig(name="a"),
        auth=AuthConfig(provider="supabase"),
        frontend=FrontendConfig(enabled=True),
    )
    errors = cfg.validate(tmp_path)
    assert not any("[frontend]" in e for e in errors)


def test_frontend_clerk_all_env_vars_present_no_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("CLERK_SECRET_KEY", "k")
    monkeypatch.setenv("CLERK_PUBLISHABLE_KEY", "pk_test_x")
    _write_project(tmp_path)
    cfg = DeployConfig(
        agent=AgentConfig(name="a"),
        auth=AuthConfig(provider="clerk"),
        frontend=FrontendConfig(enabled=True),
    )
    errors = cfg.validate(tmp_path)
    assert not any("[frontend]" in e for e in errors)
