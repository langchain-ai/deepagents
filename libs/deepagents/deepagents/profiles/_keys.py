"""Shared helpers for profile registry keys.

Both `harness_profiles` and `provider_profiles` use the same `provider` or
`provider:model` key shape, so the validation and lookup helpers live here
to avoid duplication.
"""

from __future__ import annotations


def validate_profile_key(key: str) -> None:
    """Validate a profile registry key.

    Enforces the `provider` or `provider:model` shape used by the lookup
    functions. Rejects empty strings, whitespace-only or whitespace-padded
    halves, multiple colons, and empty halves.

    Args:
        key: The registry key to check.

    Raises:
        ValueError: If `key` is empty, contains leading/trailing whitespace,
            has more than one `:`, has whitespace adjacent to `:`, or has
            an empty half on either side of `:`.
    """
    if not key:
        msg = "Profile key must be a non-empty string."
        raise ValueError(msg)
    if key != key.strip():
        msg = f"Profile key {key!r} has leading or trailing whitespace; expected 'provider' or 'provider:model'."
        raise ValueError(msg)
    if key.count(":") > 1:
        msg = f"Profile key {key!r} has more than one ':'; expected 'provider' or 'provider:model'."
        raise ValueError(msg)
    if ":" in key:
        provider, _, model = key.partition(":")
        if not provider or not model:
            msg = f"Profile key {key!r} has an empty provider or model half; expected 'provider:model'."
            raise ValueError(msg)
        if provider != provider.strip() or model != model.strip():
            msg = f"Profile key {key!r} has whitespace adjacent to ':'; expected 'provider:model' with no spaces around ':'."
            raise ValueError(msg)
