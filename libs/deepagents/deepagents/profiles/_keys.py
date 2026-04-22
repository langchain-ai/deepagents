"""Shared helpers for profile registry keys.

Both `harness_profiles` and `provider_profiles` use the same `provider` or
`provider:model` key shape, so the validation and lookup helpers live here
to avoid duplication.
"""

from __future__ import annotations


def validate_profile_key(key: str) -> None:
    """Validate a profile registry key.

    Enforces the `provider` or `provider:model` shape used by the lookup
    functions. Rejects empty strings, multiple colons, and empty halves.

    Args:
        key: The registry key to check.

    Raises:
        ValueError: If `key` is empty, has more than one `:`, or either side
            of a `:` separator is empty.
    """
    if not key:
        msg = "Profile key must be a non-empty string."
        raise ValueError(msg)
    if key.count(":") > 1:
        msg = f"Profile key {key!r} has more than one ':'; expected 'provider' or 'provider:model'."
        raise ValueError(msg)
    if ":" in key:
        provider, _, model = key.partition(":")
        if not provider or not model:
            msg = f"Profile key {key!r} has an empty provider or model half; expected 'provider:model'."
            raise ValueError(msg)
