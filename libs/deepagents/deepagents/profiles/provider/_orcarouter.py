"""Built-in OrcaRouter provider profile and helpers.

OrcaRouter is an OpenAI-compatible AI gateway. It exposes a provider/model
namespace across many models (like OpenRouter) and runs gateway-level
security for AI agents on the same endpoint. There is no dedicated
`langchain-orcarouter` integration package, so the model is constructed
through `langchain-openrouter`'s `ChatOpenRouter` (an OpenAI-compatible
client) with the OrcaRouter base URL and app attribution injected here.

Registered directly by `_ensure_builtin_profiles_loaded` during the
first profile-registry access. Not exposed as an `importlib.metadata` entry
point — built-ins ship with the SDK and should not depend on install-time
metadata to activate.
"""

from __future__ import annotations

import os
from typing import Any

from deepagents.profiles.provider.provider_profiles import ProviderProfile, _register_provider_profile_impl

ORCAROUTER_BASE_URL = "https://api.orcarouter.ai/v1"
"""Default OrcaRouter API base URL.

Mapped to `base_url` on `ChatOpenRouter`, which forwards it to the
OpenAI-compatible client as the server URL.
"""

ORCAROUTER_APP_URL = "https://github.com/langchain-ai/deepagents"
"""Default `app_url` (maps to `HTTP-Referer`) for OrcaRouter attribution."""

ORCAROUTER_APP_TITLE = "Deep Agents"
"""Default `app_title` (maps to `X-Title`) for OrcaRouter attribution."""

ORCAROUTER_API_KEY_ENV = "ORCAROUTER_API_KEY"
"""Env var holding the OrcaRouter API key.

The model is built through `ChatOpenRouter`, which reads `OPENROUTER_API_KEY`
by default. This profile reads `ORCAROUTER_API_KEY` and forwards it as
`api_key`, so `orcarouter:<model>` specs authenticate with the OrcaRouter key
without also exporting `OPENROUTER_API_KEY`.
"""


def _orcarouter_init_kwargs() -> dict[str, Any]:
    """Build default OrcaRouter kwargs, deferring to env var overrides.

    `ChatOpenRouter` reads `OPENROUTER_APP_URL` and `OPENROUTER_APP_TITLE`
    via `from_env()` defaults. Explicit kwargs passed to the constructor take
    precedence over those env-var defaults, so we only inject our SDK defaults
    when the corresponding env var is **not** set — otherwise the user's env var
    would be overridden.

    `base_url` is always injected so `orcarouter:<model>` resolves against the
    OrcaRouter endpoint by default. Callers may override it with their own
    `base_url` kwarg, which `apply_provider_profile` layers on top.

    `api_key` is read from `ORCAROUTER_API_KEY` so a single env var
    authenticates the provider; an explicit `api_key` kwarg from the caller
    still wins.

    Returns:
        Dictionary of kwargs to spread into `init_chat_model`.
    """
    kwargs: dict[str, Any] = {"base_url": ORCAROUTER_BASE_URL}
    api_key = os.environ.get(ORCAROUTER_API_KEY_ENV)
    if api_key:
        kwargs["api_key"] = api_key
    if os.environ.get("OPENROUTER_APP_URL") is None:
        kwargs["app_url"] = ORCAROUTER_APP_URL
    if os.environ.get("OPENROUTER_APP_TITLE") is None:
        kwargs["app_title"] = ORCAROUTER_APP_TITLE
    return kwargs


def register() -> None:
    """Register the built-in OrcaRouter provider profile."""
    _register_provider_profile_impl(
        "orcarouter",
        ProviderProfile(
            init_kwargs_factory=_orcarouter_init_kwargs,
        ),
    )
