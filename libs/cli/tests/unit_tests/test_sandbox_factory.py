"""Tests for sandbox factory optional dependency handling."""

from __future__ import annotations

from unittest.mock import patch

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


class TestVerifySandboxDeps:
    """Tests for the early sandbox dependency check."""

    @pytest.mark.parametrize(
        "provider",
        ["daytona", "modal", "runloop"],
    )
    def test_raises_import_error_when_backend_missing(self, provider: str) -> None:
        """Should raise ImportError with install instructions."""
        with (
            patch(
                "deepagents_cli.integrations.sandbox_factory.importlib.util.find_spec",
                return_value=None,
            ),
            pytest.raises(
                ImportError,
                match=rf"Missing dependencies for '{provider}' sandbox.*"
                rf"pip install 'deepagents-cli\[{provider}\]'",
            ),
        ):
            verify_sandbox_deps(provider)

    @pytest.mark.parametrize(
        "provider",
        ["daytona", "modal", "runloop"],
    )
    def test_passes_when_backend_installed(self, provider: str) -> None:
        """Should not raise when the backend module is found."""
        spec_sentinel = object()
        with patch(
            "deepagents_cli.integrations.sandbox_factory.importlib.util.find_spec",
            return_value=spec_sentinel,
        ):
            verify_sandbox_deps(provider)  # should not raise

    @pytest.mark.parametrize("provider", ["none", "langsmith", "", None])
    def test_skips_builtin_and_empty_providers(self, provider: str | None) -> None:
        """Built-in and empty providers should be silently accepted."""
        verify_sandbox_deps(provider)  # type: ignore[arg-type]

    def test_skips_unknown_provider(self) -> None:
        """Unknown providers are passed through for downstream handling."""
        verify_sandbox_deps("unknown_provider")  # should not raise
