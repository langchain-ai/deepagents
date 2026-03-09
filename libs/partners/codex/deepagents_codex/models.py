"""Codex model allowlist and profile metadata."""

from __future__ import annotations

from typing import Any

# Model profiles for Codex-available models.
# These mirror ChatGPT Plus/Pro subscription models accessible via the Codex backend.
CODEX_MODELS: dict[str, dict[str, Any]] = {
    "gpt-4o": {
        "max_input_tokens": 128_000,
        "max_output_tokens": 16_384,
        "tool_calling": True,
        "text_inputs": True,
        "text_outputs": True,
        "image_inputs": True,
    },
    "gpt-4o-mini": {
        "max_input_tokens": 128_000,
        "max_output_tokens": 16_384,
        "tool_calling": True,
        "text_inputs": True,
        "text_outputs": True,
        "image_inputs": True,
    },
    "o3-mini": {
        "max_input_tokens": 128_000,
        "max_output_tokens": 65_536,
        "tool_calling": True,
        "text_inputs": True,
        "text_outputs": True,
        "image_inputs": False,
    },
    "o3": {
        "max_input_tokens": 200_000,
        "max_output_tokens": 100_000,
        "tool_calling": True,
        "text_inputs": True,
        "text_outputs": True,
        "image_inputs": True,
    },
    "o4-mini": {
        "max_input_tokens": 200_000,
        "max_output_tokens": 100_000,
        "tool_calling": True,
        "text_inputs": True,
        "text_outputs": True,
        "image_inputs": True,
    },
}


def get_available_codex_models() -> list[str]:
    """Return sorted list of available Codex model IDs."""
    return sorted(CODEX_MODELS.keys())
