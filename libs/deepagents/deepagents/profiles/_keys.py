"""Shared helpers for profile registry keys.

Both `harness_profiles` and `provider_profiles` use the same `provider` or
`provider:model` key shape, so the validation and lookup helpers live here
to avoid duplication.
"""

from __future__ import annotations


def validate_profile_key(key: str) -> None:
    """Validate a `provider` or `provider:model` profile registry key.

    The first colon separates the provider from the complete model identifier.
    Providers must not contain colons, while model identifiers may contain them;
    for example, `ollama:glm-5.2:cloud` identifies provider `ollama` and model
    `glm-5.2:cloud`.

    Only the first colon is structural, but every colon-delimited segment is
    still validated. A key like `"openai:gpt-5.4:"` has an empty trailing
    segment and is rejected: it would otherwise register under a key no model
    spec can reproduce, leaving the registration silently inert.

    Args:
        key: The registry key to check.

    Raises:
        ValueError: If `key` is empty, contains leading/trailing whitespace,
            has whitespace adjacent to any `:`, or has an empty provider or
            model segment.
    """
    if not key:
        msg = "Profile key must be a non-empty string."
        raise ValueError(msg)
    if key != key.strip():
        msg = f"Profile key {key!r} has leading or trailing whitespace; expected 'provider' or 'provider:model'."
        raise ValueError(msg)
    if ":" in key:
        provider, _, model = key.partition(":")
        if not provider or not model:
            msg = f"Profile key {key!r} has an empty provider or model half; expected 'provider:model'."
            raise ValueError(msg)
        # Only the first colon is structural, so `model` is the whole remainder
        # and is nonempty as soon as it holds any character — including another
        # colon. Validate each segment so a stray colon or space deeper in the
        # key cannot register an unreachable entry.
        segments = [provider, *model.split(":")]
        if any(not segment for segment in segments):
            msg = f"Profile key {key!r} has an empty provider or model segment between colons; expected no empty colon-delimited segments."
            raise ValueError(msg)
        if any(segment != segment.strip() for segment in segments):
            msg = f"Profile key {key!r} has whitespace adjacent to ':'; expected 'provider:model' with no spaces around any ':'."
            raise ValueError(msg)
