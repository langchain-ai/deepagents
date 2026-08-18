"""Tests for extension configuration resolution."""

import os
from pathlib import Path

import pytest

from deepagents_code._env_vars import (
    EXTENSIONS,
    EXTENSIONS_PATHS,
    EXTENSIONS_TRUST,
)
from deepagents_code.extensions import TrustPolicy, load_extension_settings


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point extension settings at an isolated user config file."""
    path = tmp_path / "config.toml"
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", path)
    return path


def test_loads_extension_config(config_path: Path) -> None:
    """Typed TOML values should populate extension settings."""
    config_path.write_text(
        """
[extensions]
enabled = false
paths = ["~/one.py", "/tmp/two"]
trust = "always"
""",
        encoding="utf-8",
    )

    settings = load_extension_settings()

    assert settings.enabled is False
    assert settings.paths == (Path("~/one.py").expanduser(), Path("/tmp/two"))
    assert settings.trust is TrustPolicy.ALWAYS


def test_environment_overrides_config(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid environment values should take precedence over TOML."""
    config_path.write_text(
        '[extensions]\nenabled = false\npaths = ["old.py"]\ntrust = "never"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv(EXTENSIONS, "yes")
    monkeypatch.setenv(EXTENSIONS_PATHS, f"one.py{os.pathsep}two.py")
    monkeypatch.setenv(EXTENSIONS_TRUST, "ask")

    settings = load_extension_settings()

    assert settings.enabled is True
    assert settings.paths == (Path("one.py"), Path("two.py"))
    assert settings.trust is TrustPolicy.ASK


def test_invalid_environment_falls_through_to_config(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed overrides must not disagree with effective TOML behavior."""
    config_path.write_text(
        '[extensions]\nenabled = false\ntrust = "never"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv(EXTENSIONS, "maybe")
    monkeypatch.setenv(EXTENSIONS_TRUST, "sometimes")

    settings = load_extension_settings()

    assert settings.enabled is False
    assert settings.trust is TrustPolicy.NEVER


def test_malformed_config_uses_safe_defaults(config_path: Path) -> None:
    """Wrong TOML types and invalid policy strings should use defaults."""
    config_path.write_text(
        '[extensions]\nenabled = "false"\npaths = "one.py"\ntrust = "bogus"\n',
        encoding="utf-8",
    )

    settings = load_extension_settings()

    assert settings.enabled is True
    assert not settings.paths
    assert settings.trust is TrustPolicy.ASK
