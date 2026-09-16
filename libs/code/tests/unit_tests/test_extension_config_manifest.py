"""Tests for extension entries in the configuration catalog."""

from pathlib import Path
from unittest.mock import patch

import pytest

from deepagents_code._env_vars import EXTENSIONS, EXTENSIONS_TRUST
from deepagents_code.config_manifest import (
    OptionKind,
    get_option,
)
from unit_tests.conftest import resolve_option_for_test


def test_extension_options_are_cataloged() -> None:
    """The config CLI should describe every user-facing extension setting."""
    enabled = get_option("extensions.enabled")
    trust = get_option("extensions.trust")

    assert enabled is not None
    assert enabled.kind is OptionKind.BOOL
    assert get_option("extensions.extra_paths") is not None
    assert trust is not None
    assert trust.kind is OptionKind.EXTENSION_TRUST_DELEGATE


def test_invalid_trust_env_matches_runtime_fallback() -> None:
    """A malformed trust override falls through to typed TOML."""
    trust = get_option("extensions.trust")
    assert trust is not None
    toml = {"extensions": {"trust": "never"}}

    with patch.dict("os.environ", {EXTENSIONS_TRUST: "sometimes"}, clear=False):
        assert resolve_option_for_test(trust, toml_data=toml) == (
            "never",
            "config.toml",
        )


def test_runtime_settings_honor_managed_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed extension policy must override user and environment values."""
    from deepagents_code.configuration import resolver as resolver_module
    from deepagents_code.configuration.types import TomlSnapshot
    from deepagents_code.extensions import settings as settings_module
    from deepagents_code.extensions.settings import TrustPolicy, load_extension_settings

    resolver = resolver_module.resolver_from_snapshots(
        managed=TomlSnapshot.from_table(
            "managed config", {"extensions": {"enabled": False, "trust": "never"}}
        ),
        user=TomlSnapshot.from_table(
            "config.toml",
            {
                "extensions": {
                    "enabled": True,
                    "trust": "always",
                    "extra_paths": [str(tmp_path / "one.py"), str(tmp_path / "more")],
                }
            },
        ),
    )
    monkeypatch.setattr(settings_module, "get_config_resolver", lambda: resolver)
    monkeypatch.setenv(EXTENSIONS, "true")
    monkeypatch.setenv(EXTENSIONS_TRUST, "always")

    settings = load_extension_settings()

    assert not settings.enabled
    assert settings.trust is TrustPolicy.NEVER
    assert settings.extra_paths == (tmp_path / "one.py", tmp_path / "more")
