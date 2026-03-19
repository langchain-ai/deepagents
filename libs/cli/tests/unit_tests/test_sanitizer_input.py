# libs/cli/tests/unit_tests/test_sanitizer_input.py
"""Tests for human-input sanitization in the CLI layer."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from deepagents.middleware.sanitizer import SanitizeFinding, SanitizeResult, SanitizerProvider


def test_create_sanitizer_provider_gitleaks():
    """create_sanitizer_provider returns a GitleaksSanitizerProvider for 'gitleaks'."""
    from deepagents_cli.sanitizer_factory import create_sanitizer_provider

    with patch("deepagents.middleware.sanitizer_gitleaks.shutil.which", return_value="/usr/bin/gitleaks"):
        provider = create_sanitizer_provider("gitleaks")
    assert provider is not None
    assert provider.name == "gitleaks"


def test_create_sanitizer_provider_unknown_returns_none():
    """create_sanitizer_provider returns None for an unknown provider name."""
    from deepagents_cli.sanitizer_factory import create_sanitizer_provider

    provider = create_sanitizer_provider("nosuchprovider")
    assert provider is None


def test_create_sanitizer_provider_none_returns_none():
    """create_sanitizer_provider returns None when given None."""
    from deepagents_cli.sanitizer_factory import create_sanitizer_provider

    provider = create_sanitizer_provider(None)
    assert provider is None


def test_create_sanitizer_provider_missing_binary_returns_none():
    """create_sanitizer_provider returns None when the binary is not found."""
    from deepagents_cli.sanitizer_factory import create_sanitizer_provider

    with patch("deepagents.middleware.sanitizer_gitleaks.shutil.which", return_value=None):
        provider = create_sanitizer_provider("gitleaks")
    assert provider is None
