"""Tests for extension entries in the configuration catalog."""

from unittest.mock import patch

from deepagents_code._env_vars import EXTENSIONS_TRUST
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
