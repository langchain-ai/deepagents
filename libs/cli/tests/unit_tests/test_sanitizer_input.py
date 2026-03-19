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


# ---------------------------------------------------------------------------
# Human-input sanitization logic (unit-testable without Textual)
# ---------------------------------------------------------------------------


class FakeProvider:
    """Minimal SanitizerProvider for testing."""

    @property
    def name(self) -> str:
        return "fake"

    def sanitize(self, content: str) -> SanitizeResult:
        raise NotImplementedError

    async def asanitize(self, content: str) -> SanitizeResult:
        if "AKIA" in content:
            redacted = content.replace("AKIAIOSFODNN7EXAMPLE", "<REDACTED:aws-access-key>")
            return SanitizeResult(
                content=redacted,
                findings=[SanitizeFinding(rule_id="aws-access-key", redacted_as="<REDACTED:aws-access-key>")],
            )
        return SanitizeResult(content=content, findings=[])


@pytest.mark.asyncio
async def test_sanitize_human_input_with_secret():
    """sanitize_human_input returns redacted text and findings when a secret is present."""
    from deepagents_cli.app import sanitize_human_input

    provider = FakeProvider()
    result = await sanitize_human_input("my key is AKIAIOSFODNN7EXAMPLE ok", provider)
    assert result.redacted == "my key is <REDACTED:aws-access-key> ok"
    assert len(result.findings) == 1
    assert result.findings[0]["rule_id"] == "aws-access-key"


@pytest.mark.asyncio
async def test_sanitize_human_input_clean():
    """sanitize_human_input returns original text and no findings when clean."""
    from deepagents_cli.app import sanitize_human_input

    provider = FakeProvider()
    result = await sanitize_human_input("just a normal message", provider)
    assert result.redacted == "just a normal message"
    assert result.findings == []


@pytest.mark.asyncio
async def test_sanitize_human_input_no_provider():
    """sanitize_human_input returns original text when provider is None."""
    from deepagents_cli.app import sanitize_human_input

    result = await sanitize_human_input("AKIAIOSFODNN7EXAMPLE", None)
    assert result.redacted == "AKIAIOSFODNN7EXAMPLE"
    assert result.findings == []


@pytest.mark.asyncio
async def test_sanitize_human_input_empty_string():
    """sanitize_human_input handles empty string gracefully."""
    from deepagents_cli.app import sanitize_human_input

    provider = FakeProvider()
    result = await sanitize_human_input("", provider)
    assert result.redacted == ""
    assert result.findings == []


class ExplodingProvider:
    """Provider that raises on asanitize."""

    @property
    def name(self) -> str:
        return "exploding"

    def sanitize(self, content: str) -> SanitizeResult:
        raise RuntimeError("boom")

    async def asanitize(self, content: str) -> SanitizeResult:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_sanitize_human_input_provider_error_fallback():
    """sanitize_human_input returns original text when provider raises."""
    from deepagents_cli.app import sanitize_human_input

    provider = ExplodingProvider()
    result = await sanitize_human_input("my secret text", provider)
    assert result.redacted == "my secret text"
    assert result.findings == []
