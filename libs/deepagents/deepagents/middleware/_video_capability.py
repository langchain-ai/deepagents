"""Registry of model providers whose chat models accept native video content blocks.

The registry is intentionally tiny — only Gemini currently supports video input
across the providers that `deepagents` users commonly bind. New entries are added
as support lands upstream.
"""

from __future__ import annotations

VIDEO_CAPABLE_PATTERNS: tuple[str, ...] = ("gemini-",)
"""Lowercase model-name prefixes that signal native video support.

Matching is `prefix`-style against the lowercased model identifier. Provider is
checked loosely (any provider reporting a Gemini model qualifies).
"""


def is_video_capable(
    provider: str | None,
    model_name: str | None,
    *,
    override: bool | None = None,
) -> bool:
    """Return True when the provider+model combination accepts native video blocks.

    Args:
        provider: The provider string (e.g., `"google_genai"`) reported by
            `langchain` for the bound model, or `None` if unknown.
        model_name: The model identifier (e.g., `"gemini-2.0-flash"`), or
            `None` if unknown.
        override: If `True`, always return `True` regardless of the registry.
            If `False`, always return `False`. If `None` (default), consult
            the registry.

    Returns:
        `True` if the provider+model is known to accept native video content
        blocks, otherwise `False`.
    """
    if override is not None:
        return override
    if not model_name:
        return False
    lowered = model_name.lower()
    return any(lowered.startswith(prefix) for prefix in VIDEO_CAPABLE_PATTERNS)
