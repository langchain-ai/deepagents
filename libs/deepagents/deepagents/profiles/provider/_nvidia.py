"""Built-in NVIDIA provider profile and helpers.

Injects Deep Agents app-origin attribution into NVIDIA NIM requests via the
header supported by `langchain-nvidia-ai-endpoints`.

Registered directly by `_ensure_builtin_profiles_loaded` during the first
profile-registry access. Not exposed as an `importlib.metadata` entry point —
built-ins ship with the SDK and should not depend on install-time metadata to
activate.
"""

from __future__ import annotations

from typing import Any

from deepagents.profiles.provider.provider_profiles import (
    ProviderProfile,
    _register_provider_profile_impl,
)

_NVIDIA_BILLING_ORIGIN_HEADER = "X-BILLING-INVOKE-ORIGIN"
"""NVIDIA NIM header used to attribute requests to an originating app."""

_NVIDIA_APP_ORIGIN = "DeepAgents"
"""Deep Agents identity reported to NVIDIA NIM."""


def _nvidia_attribution_kwargs() -> dict[str, Any]:
    """Build default NVIDIA NIM app-attribution kwargs.

    `ChatNVIDIA` accepts `default_headers` and merges them into sync, async,
    and streaming requests. Returning a new nested mapping on every call keeps
    profile resolution isolated between model instances.

    Returns:
        Dictionary of kwargs to spread into `init_chat_model`.
    """
    return {
        "default_headers": {
            _NVIDIA_BILLING_ORIGIN_HEADER: _NVIDIA_APP_ORIGIN,
        }
    }


def register() -> None:
    """Register the built-in NVIDIA provider profile."""
    _register_provider_profile_impl(
        "nvidia",
        ProviderProfile(init_kwargs_factory=_nvidia_attribution_kwargs),
    )
