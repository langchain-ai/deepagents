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


def test_frontend_section_parses_enabled_true():
    cfg = _parse_config({
        "agent": {"name": "my-agent"},
        "auth": {"provider": "supabase"},
        "frontend": {"enabled": True},
    })
    assert cfg.frontend is not None
    assert cfg.frontend.enabled is True
    assert cfg.frontend.app_name is None


def test_frontend_section_parses_app_name():
    cfg = _parse_config({
        "agent": {"name": "my-agent"},
        "auth": {"provider": "clerk"},
        "frontend": {"enabled": True, "app_name": "My App"},
    })
    assert cfg.frontend is not None
    assert cfg.frontend.app_name == "My App"


def test_frontend_section_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown key"):
        _parse_config({
            "agent": {"name": "my-agent"},
            "auth": {"provider": "supabase"},
            "frontend": {"enabled": True, "theme": "dark"},
        })


def test_frontend_omitted_defaults_to_none():
    cfg = _parse_config({"agent": {"name": "my-agent"}})
    assert cfg.frontend is None
