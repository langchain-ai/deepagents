"""Codex model allowlist and profile metadata."""

from __future__ import annotations

from typing import Any

# Model profiles for Codex-available models.
# These are the models available via the Codex backend (ChatGPT subscription).
CODEX_MODELS: dict[str, dict[str, Any]] = {
    "gpt-5.3-codex": {
        "max_input_tokens": 400_000,
        "max_output_tokens": 16_384,
        "tool_calling": True,
        "text_inputs": True,
        "text_outputs": True,
        "image_inputs": False,
    },
    "gpt-5.2-codex": {
        "max_input_tokens": 400_000,
        "max_output_tokens": 16_384,
        "tool_calling": True,
        "text_inputs": True,
        "text_outputs": True,
        "image_inputs": False,
    },
    "gpt-5.1-codex": {
        "max_input_tokens": 192_000,
        "max_output_tokens": 16_384,
        "tool_calling": True,
        "text_inputs": True,
        "text_outputs": True,
        "image_inputs": False,
    },
    "gpt-5.1-codex-mini": {
        "max_input_tokens": 192_000,
        "max_output_tokens": 16_384,
        "tool_calling": True,
        "text_inputs": True,
        "text_outputs": True,
        "image_inputs": False,
    },
    "codex-mini-latest": {
        "max_input_tokens": 192_000,
        "max_output_tokens": 16_384,
        "tool_calling": True,
        "text_inputs": True,
        "text_outputs": True,
        "image_inputs": False,
    },
}


def get_available_codex_models() -> list[str]:
    """Return sorted list of available Codex model IDs."""
    return sorted(CODEX_MODELS.keys())
