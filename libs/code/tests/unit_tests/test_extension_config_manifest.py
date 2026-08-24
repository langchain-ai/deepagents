"""Tests for extension entries in the configuration catalog."""

from unittest.mock import patch

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
    assert get_option("extensions.extra_files") is not None
    assert get_option("extensions.extra_dirs") is not None
    assert trust is not None
    assert trust.kind is OptionKind.EXTENSION_TRUST_DELEGATE


def test_invalid_env_values_match_runtime_fallback() -> None:
    """Malformed overrides should fall through to typed TOML values."""
    enabled = get_option("extensions.enabled")
    trust = get_option("extensions.trust")
    assert enabled is not None
    assert trust is not None
    toml = {"extensions": {"enabled": False, "trust": "never"}}

    with patch.dict(
        "os.environ",
        {EXTENSIONS: "maybe", EXTENSIONS_TRUST: "sometimes"},
        clear=False,
    ):
        assert resolve_option_for_test(enabled, toml_data=toml) == (
            False,
            "config.toml",
        )
        assert resolve_option_for_test(trust, toml_data=toml) == (
            "never",
            "config.toml",
        )
