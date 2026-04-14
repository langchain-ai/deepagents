"""Harness profiles and provider-specific configuration.

!!! warning

    This is an internal API subject to change without deprecation. It is not
    intended for external use or consumption.

Re-exports the profile dataclass, registry helpers, and provider modules so
internal consumers can import from `deepagents.profiles` directly.
"""

# Provider/model modules register their profiles as a side effect of import.
from deepagents.profiles import _codex as _codex, _openai as _openai
from deepagents.profiles._harness_profiles import (
    _HARNESS_PROFILES,
    _get_harness_profile,
    _HarnessProfile,
    _merge_profiles,
    _register_harness_profile,
)
from deepagents.profiles._openrouter import (
    _OPENROUTER_APP_TITLE,
    _OPENROUTER_APP_URL,
    OPENROUTER_MIN_VERSION,
    _openrouter_attribution_kwargs,
    check_openrouter_version,
)

__all__ = [
    "OPENROUTER_MIN_VERSION",
    "_HARNESS_PROFILES",
    "_OPENROUTER_APP_TITLE",
    "_OPENROUTER_APP_URL",
    "_HarnessProfile",
    "_get_harness_profile",
    "_merge_profiles",
    "_openrouter_attribution_kwargs",
    "_register_harness_profile",
    "check_openrouter_version",
]
