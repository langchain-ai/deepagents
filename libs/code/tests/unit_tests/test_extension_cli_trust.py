"""Tests for interactive project-extension trust."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from deepagents_code._env_vars import EXPERIMENTAL
from deepagents_code.extensions.settings import ExtensionSettings, TrustPolicy
from deepagents_code.main import (
    _check_project_extensions_trust,
    _TrustAction,
    _TrustPromptOutcome,
    parse_args,
)


@pytest.fixture(autouse=True)
def _enable_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXPERIMENTAL, "1")


def test_experimental_mode_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabled extensions do not inspect projects or prompt for trust."""
    monkeypatch.delenv(EXPERIMENTAL)
    assert not _check_project_extensions_trust(trust_flag=True)


@pytest.mark.parametrize(
    "argv",
    [
        ["dcode", "--extension", "extension.py"],
        ["dcode", "--trust-project-extensions"],
    ],
)
def test_extension_flags_require_experimental_mode(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit extension flags fail instead of being silently ignored."""
    monkeypatch.delenv(EXPERIMENTAL, raising=False)
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit, match="2"):
        parse_args()


def test_never_policy_overrides_explicit_flag() -> None:
    """A configured hard stop wins over the CLI flag."""
    with patch(
        "deepagents_code.extensions.settings.load_extension_settings",
        return_value=ExtensionSettings(trust=TrustPolicy.NEVER),
    ):
        assert not _check_project_extensions_trust(trust_flag=True)
