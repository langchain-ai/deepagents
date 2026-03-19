# libs/cli/deepagents_cli/sanitizer_factory.py
"""Factory for creating sanitizer providers from CLI flag values."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents.middleware.sanitizer import SanitizerProvider

logger = logging.getLogger(__name__)


def create_sanitizer_provider(name: str | None) -> "SanitizerProvider | None":
    """Create a sanitizer provider by name.

    Args:
        name: Provider name from the ``--sanitizer`` flag, or ``None``.

    Returns:
        An instantiated provider, or ``None`` if the name is unknown or
        the provider binary is missing.
    """
    if not name:
        return None

    from deepagents.middleware.sanitizer_gitleaks import GitleaksSanitizerProvider

    _PROVIDERS: dict[str, type] = {
        "gitleaks": GitleaksSanitizerProvider,
    }

    provider_cls = _PROVIDERS.get(name)
    if provider_cls is None:
        logger.warning("Unknown sanitizer provider: %s — skipping", name)
        return None

    try:
        return provider_cls()
    except FileNotFoundError:
        logger.warning(
            "Sanitizer '%s' binary not found in PATH — install it or remove --sanitizer flag",
            name,
        )
        return None
