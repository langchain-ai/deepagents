"""Tests for sandbox factory optional dependency handling."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from deepagents_cli.integrations.sandbox_factory import _get_provider


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
