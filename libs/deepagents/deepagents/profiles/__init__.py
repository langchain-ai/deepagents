"""Harness and provider profile registries.

!!! warning

    This is an internal API subject to change without deprecation. It is not
    intended for external use or consumption.

Re-exports the internal profile dataclasses, registries, and provider modules so
internal consumers can import from `deepagents.profiles` directly.
"""

# Provider modules register their provider profiles as a side effect of import.
from deepagents.profiles import _openai as _openai
from deepagents.profiles._harness_profiles import (
    _HARNESS_PROFILES,
    _GeneralPurposeSubagentProfile,
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
from deepagents.profiles._provider_profiles import (
    _PROVIDER_PROFILES,
    _get_provider_profile,
    _merge_provider_profiles,
    _ProviderProfile,
    _register_provider_profile,
)

__all__ = [
    "OPENROUTER_MIN_VERSION",
    "_HARNESS_PROFILES",
    "_OPENROUTER_APP_TITLE",
    "_OPENROUTER_APP_URL",
    "_PROVIDER_PROFILES",
    "_GeneralPurposeSubagentProfile",
    "_HarnessProfile",
    "_ProviderProfile",
    "_get_harness_profile",
    "_get_provider_profile",
    "_merge_profiles",
    "_merge_provider_profiles",
    "_openrouter_attribution_kwargs",
    "_register_harness_profile",
    "_register_provider_profile",
    "check_openrouter_version",
]
