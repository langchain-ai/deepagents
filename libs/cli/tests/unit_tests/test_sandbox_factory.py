"""Tests for sandbox factory optional dependency handling."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from deepagents_cli.integrations.sandbox_factory import (
    _get_provider,
    verify_sandbox_deps,
)


@pytest.mark.parametrize(
    ("provider", "package"),
    [
        ("daytona", "langchain-daytona"),
        ("modal", "langchain-modal"),
        ("runloop", "langchain-runloop"),
    ],
)
def test_get_provider_raises_helpful_error_for_missing_optional_dependency(
    provider: str,
    package: str,
) -> None:
    """Provider construction should explain which CLI extra to install."""
    error = (
        rf"The '{provider}' sandbox provider requires the "
        rf"'{package}' package"
    )
    with (
        patch(
            "deepagents_cli.integrations.sandbox_factory.importlib.import_module",
            side_effect=ImportError("missing dependency"),
        ),
        pytest.raises(ImportError, match=error),
    ):
        _get_provider(provider)


def test_agentcore_get_or_create_raises_for_missing_dep() -> None:
    """AgentCore should explain which package to install."""
    error = (
        r"The 'agentcore' sandbox provider requires the "
        r"'langchain-agentcore-codeinterpreter' package"
    )

    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.get_credentials.return_value = MagicMock()
    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        provider = _get_provider("agentcore")

    with (
        patch(
            "deepagents_cli.integrations.sandbox_factory.importlib.import_module",
            side_effect=ImportError("missing dependency"),
        ),
        pytest.raises(ImportError, match=error),
    ):
        provider.get_or_create()


def test_agentcore_raises_on_missing_aws_credentials() -> None:
    """AgentCore should raise ValueError without AWS creds."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.get_credentials.return_value = None
    with (
        patch.dict(sys.modules, {"boto3": mock_boto3}),
        pytest.raises(ValueError, match="AWS credentials not found"),
    ):
        _get_provider("agentcore")


def test_agentcore_rejects_sandbox_id() -> None:
    """AgentCore should raise NotImplementedError for sandbox_id."""
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.get_credentials.return_value = MagicMock()
    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        provider = _get_provider("agentcore")

    with pytest.raises(NotImplementedError, match="does not support reconnecting"):
        provider.get_or_create(sandbox_id="some-id")


class TestVerifySandboxDeps:
    """Tests for the early sandbox dependency check."""

    @pytest.mark.parametrize(
        ("provider", "expected_module"),
        [
            ("agentcore", "langchain_agentcore_codeinterpreter"),
            ("daytona", "langchain_daytona"),
            ("modal", "langchain_modal"),
            ("runloop", "langchain_runloop"),
        ],
    )
    def test_raises_import_error_when_backend_missing(
        self, provider: str, expected_module: str
    ) -> None:
        """Should raise ImportError with install instructions."""
        mock_find_spec = patch(
            "deepagents_cli.integrations.sandbox_factory.importlib.util.find_spec",
            return_value=None,
        )
        with (
            mock_find_spec as find_spec,
            pytest.raises(
                ImportError,
                match=rf"Missing dependencies for '{provider}' sandbox.*"
                rf"pip install 'deepagents-cli\[{provider}\]'",
            ),
        ):
            verify_sandbox_deps(provider)

        find_spec.assert_called_once_with(expected_module)

    @pytest.mark.parametrize(
        "provider",
        ["agentcore", "daytona", "modal", "runloop"],
    )
    def test_passes_when_backend_installed(self, provider: str) -> None:
        """Should not raise when the backend module is found."""
        spec_sentinel = object()
        with patch(
            "deepagents_cli.integrations.sandbox_factory.importlib.util.find_spec",
            return_value=spec_sentinel,
        ):
            verify_sandbox_deps(provider)  # should not raise

    @pytest.mark.parametrize(
        "exc_cls",
        [ImportError, ValueError],
    )
    def test_raises_when_find_spec_throws(self, exc_cls: type) -> None:
        """find_spec can raise ImportError/ValueError in corrupted envs."""
        with (
            patch(
                "deepagents_cli.integrations.sandbox_factory.importlib.util.find_spec",
                side_effect=exc_cls("broken"),
            ),
            pytest.raises(ImportError, match="Missing dependencies"),
        ):
            verify_sandbox_deps("daytona")

    @pytest.mark.parametrize("provider", ["none", "langsmith", "", None])
    def test_skips_builtin_and_empty_providers(self, provider: str | None) -> None:
        """Built-in and empty providers should be silently accepted."""
        verify_sandbox_deps(provider)  # type: ignore[arg-type]

    def test_skips_unknown_provider(self) -> None:
        """Unknown providers are passed through for downstream handling."""
        verify_sandbox_deps("unknown_provider")  # should not raise
