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

VIDEO_CAPABLE_PROVIDERS: frozenset[str] = frozenset({"google_genai", "google_vertexai"})
"""Provider strings (lowercased) that qualify as video-capable as a fallback
when `model_name` is empty or unknown."""


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

    When `model_name` is non-empty, the decision is made solely on the model
    name (model overrides provider). When `model_name` is empty or `None`,
    `provider` is consulted as a fallback against `VIDEO_CAPABLE_PROVIDERS`.
    """
    if override is not None:
        return override
    if model_name:
        lowered = model_name.lower()
        return any(lowered.startswith(prefix) for prefix in VIDEO_CAPABLE_PATTERNS)
    # model_name is missing/empty — fall back to provider.
    return bool(provider and provider.lower() in VIDEO_CAPABLE_PROVIDERS)
