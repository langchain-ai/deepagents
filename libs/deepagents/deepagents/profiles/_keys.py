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

    Only the first colon is structural. The complete model identifier must be
    nonempty and have no surrounding whitespace; its internal syntax belongs
    to the provider. In particular, Bedrock foundation-model ARNs can contain
    `::` because their account component is empty.

    Args:
        key: The registry key to check.

    Raises:
        ValueError: If `key` is empty, contains leading/trailing whitespace,
            has whitespace adjacent to the first `:`, or has an empty provider
            or model identifier.
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
        if provider != provider.strip() or model != model.strip():
            msg = f"Profile key {key!r} has whitespace adjacent to ':'; expected 'provider:model' with no spaces around the first ':'."
            raise ValueError(msg)
