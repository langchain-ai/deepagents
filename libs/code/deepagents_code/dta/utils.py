"""Utility helpers for Dynamic Tool Allocation (DTA)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from deepagents_code.config import settings
except ImportError:
    settings = None  # type: ignore[assignment]


def get_dta_fast_model() -> str | None:
    """Resolve the fastest available model spec based on the configured provider.

    Checks for an explicit per-session DTA model override stored in
    ``settings.dta_model`` first, then falls back to a curated map of
    provider → cheap fast model.  Returns ``None`` when no provider is
    detected so callers can fall back to the user's primary model.

    Returns:
        A model spec string such as ``"openai:gpt-4o-mini"``, or ``None``
        if the provider cannot be determined.
    """
    if settings is None:
        return None

    # Explicit per-session override (set via /dta-model slash command)
    dta_model: str | None = getattr(settings, "dta_model", None)
    if dta_model:
        return dta_model

    # Auto-resolve: map primary provider → fastest cheap model
    provider: str = getattr(settings, "model_provider", "") or ""
    if provider == "anthropic":
        return "anthropic:claude-3-haiku-20240307"
    if provider == "openai":
        return "openai:gpt-4o-mini"
    if provider == "google_genai":
        return "google_genai:gemini-1.5-flash"

    # Final fallback: use whatever the main model setting is
    return getattr(settings, "model", None)
