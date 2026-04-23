"""Tests for the provider video-capability registry."""

from __future__ import annotations

import pytest

from deepagents.middleware._video_capability import (
    VIDEO_CAPABLE_PATTERNS,
    is_video_capable,
)


@pytest.mark.parametrize(
    ("provider", "model_name", "override", "expected"),
    [
        # Gemini variants are capable by default.
        ("google_genai", "gemini-2.0-flash", None, True),
        ("google_vertexai", "gemini-2.5-pro", None, True),
        # Other providers are not capable by default.
        ("anthropic", "claude-sonnet-4-6", None, False),
        ("openai", "gpt-5", None, False),
        ("unknown-provider", "whatever-model", None, False),
        # None / empty inputs never crash.
        # provider=None but valid gemini model_name → True (matching on model_name alone).
        (None, "gemini-2.0-flash", None, True),
        # provider is present but model_name is None → False.
        ("google_genai", None, None, False),
        # provider is present but model_name is empty → False.
        ("google_genai", "", None, False),
        # Case-insensitive matching on model name.
        ("google_genai", "GEMINI-2.0-FLASH", None, True),
        # Override forces capability regardless of registry.
        ("anthropic", "claude-sonnet-4-6", True, True),
        ("google_genai", "gemini-2.0-flash", False, False),
    ],
)
def test_is_video_capable(
    provider: str | None,
    model_name: str | None,
    override: bool | None,  # noqa: FBT001
    expected: bool,  # noqa: FBT001
) -> None:
    assert is_video_capable(provider, model_name, override=override) is expected


def test_registry_is_tuple_of_lowercase_prefixes() -> None:
    assert isinstance(VIDEO_CAPABLE_PATTERNS, tuple)
    for pattern in VIDEO_CAPABLE_PATTERNS:
        assert isinstance(pattern, str)
        assert pattern == pattern.lower()
        assert len(pattern) > 0
