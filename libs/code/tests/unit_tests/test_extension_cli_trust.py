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


@pytest.mark.parametrize(
    ("action", "expected", "persist"),
    [
        (_TrustAction.ALLOW_ONCE, True, False),
        (_TrustAction.REMEMBER, True, True),
        (_TrustPromptOutcome.CANCELLED, _TrustPromptOutcome.CANCELLED, False),
    ],
)
def test_prompt_decisions(
    tmp_path: Path,
    action: _TrustAction | _TrustPromptOutcome,
    expected: bool | _TrustPromptOutcome,
    persist: bool,
) -> None:
    """Prompt outcomes grant, persist, or cancel as selected."""
    (tmp_path / ".deepagents" / "extensions").mkdir(parents=True)
    with (
        patch(
            "deepagents_code.project_utils.ProjectContext.from_user_cwd",
            return_value=SimpleNamespace(project_root=tmp_path, user_cwd=tmp_path),
        ),
        patch(
            "deepagents_code.extensions.trust.is_project_extensions_trusted",
            return_value=False,
        ),
        patch(
            "deepagents_code.extensions.trust.trust_project_extensions",
            return_value=True,
        ) as trust,
        patch("deepagents_code.main._select_trust_action", return_value=action),
    ):
        assert _check_project_extensions_trust() is expected
    assert trust.call_count == int(persist)
