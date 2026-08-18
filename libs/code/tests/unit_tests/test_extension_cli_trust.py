"""Tests for interactive project extension trust decisions."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from deepagents_code.extensions import ExtensionSettings, TrustPolicy
from deepagents_code.main import (
    _check_project_extensions_trust,
    _TrustAction,
    _TrustPromptOutcome,
)


def _project(tmp_path: Path) -> SimpleNamespace:
    """Create a project context containing an extensions directory."""
    (tmp_path / ".deepagents" / "extensions").mkdir(parents=True)
    return SimpleNamespace(project_root=tmp_path, user_cwd=tmp_path)


def test_never_policy_overrides_explicit_flag(tmp_path: Path) -> None:
    """A user-level never policy should remain a hard stop."""
    context = _project(tmp_path)
    with (
        patch(
            "deepagents_code.extensions.settings.load_extension_settings",
            return_value=ExtensionSettings(trust=TrustPolicy.NEVER),
        ),
        patch(
            "deepagents_code.project_utils.ProjectContext.from_user_cwd",
            return_value=context,
        ),
    ):
        assert not _check_project_extensions_trust(trust_flag=True)


def test_allow_once_grants_current_session(tmp_path: Path) -> None:
    """An allow-once response should grant without persisting."""
    context = _project(tmp_path)
    with (
        patch(
            "deepagents_code.project_utils.ProjectContext.from_user_cwd",
            return_value=context,
        ),
        patch(
            "deepagents_code.extensions.trust.is_project_extensions_trusted",
            return_value=False,
        ),
        patch(
            "deepagents_code.main._select_trust_action",
            return_value=_TrustAction.ALLOW_ONCE,
        ),
    ):
        assert _check_project_extensions_trust() is True


def test_remember_persists_project_trust(tmp_path: Path) -> None:
    """The remember response should write the canonical project decision."""
    context = _project(tmp_path)
    with (
        patch(
            "deepagents_code.project_utils.ProjectContext.from_user_cwd",
            return_value=context,
        ),
        patch(
            "deepagents_code.extensions.trust.is_project_extensions_trusted",
            return_value=False,
        ),
        patch(
            "deepagents_code.extensions.trust.trust_project_extensions",
            return_value=True,
        ) as trust,
        patch(
            "deepagents_code.main._select_trust_action",
            return_value=_TrustAction.REMEMBER,
        ),
    ):
        assert _check_project_extensions_trust() is True

    trust.assert_called_once_with(tmp_path)


def test_cancel_propagates_to_startup(tmp_path: Path) -> None:
    """Cancelling the selector should abort rather than silently deny."""
    context = _project(tmp_path)
    with (
        patch(
            "deepagents_code.project_utils.ProjectContext.from_user_cwd",
            return_value=context,
        ),
        patch(
            "deepagents_code.extensions.trust.is_project_extensions_trusted",
            return_value=False,
        ),
        patch(
            "deepagents_code.main._select_trust_action",
            return_value=_TrustPromptOutcome.CANCELLED,
        ),
    ):
        assert _check_project_extensions_trust() is _TrustPromptOutcome.CANCELLED
