"""Harness profiles and provider-specific configuration.

!!! warning

    This is an internal API subject to change without deprecation. It is not
    intended for external use or consumption.

Re-exports the profile dataclass, registry helpers, and provider modules so
internal consumers can import from `deepagents.profiles` directly.
"""

# Provider modules register their profiles as a side effect of import.
# _openrouter registration fires via the `from` import below.
# _anthropic_opus registers the model-level Opus 4.6 overlay.
# _anthropic_opus47 registers the model-level Opus 4.7 overlay.
from deepagents.profiles import (
    _anthropic as _anthropic,
    _anthropic_opus as _anthropic_opus,
    _anthropic_opus47 as _anthropic_opus47,
    _openai as _openai,
)
from deepagents.profiles._anthropic import (
    _ANTHROPIC_SYSTEM_PROMPT_SUFFIX,
)
from deepagents.profiles._anthropic_opus import (
    _ANTHROPIC_OPUS_SYSTEM_PROMPT_SUFFIX,
    _OPUS_SYSTEM_PROMPT_SUFFIX,
)
from deepagents.profiles._anthropic_opus47 import (
    _ANTHROPIC_OPUS_47_SYSTEM_PROMPT_SUFFIX,
    _OPUS_47_SYSTEM_PROMPT_SUFFIX,
)
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
    "_ANTHROPIC_OPUS_47_SYSTEM_PROMPT_SUFFIX",
    "_ANTHROPIC_OPUS_SYSTEM_PROMPT_SUFFIX",
    "_ANTHROPIC_SYSTEM_PROMPT_SUFFIX",
    "_HARNESS_PROFILES",
    "_OPENROUTER_APP_TITLE",
    "_OPENROUTER_APP_URL",
    "_OPUS_47_SYSTEM_PROMPT_SUFFIX",
    "_OPUS_SYSTEM_PROMPT_SUFFIX",
    "_HarnessProfile",
    "_get_harness_profile",
    "_merge_profiles",
    "_openrouter_attribution_kwargs",
    "_register_harness_profile",
    "check_openrouter_version",
]
