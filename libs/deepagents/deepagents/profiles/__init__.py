"""Public beta APIs for model and harness profiles.

!!! beta

    `deepagents.profiles` exposes beta APIs that may receive minor changes in
    future releases. Refer to the [versioning documentation](https://docs.langchain.com/oss/python/versioning)
    for more details.

Exposes the public `ProviderProfile`, `HarnessProfile`, and
`HarnessProfileConfig` APIs for customizing how `resolve_model` constructs chat
models and how `create_deep_agent` shapes agent runtime behavior.
Registration helpers are additive: re-registering under an existing key merges
on top of the prior registration.
"""

from deepagents.profiles._builtin_profiles import _ensure_builtin_profiles_loaded
from deepagents.profiles.harness_profiles import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    HarnessProfileConfig,
    register_harness_profile,
)
from deepagents.profiles.provider.provider_profiles import (
    ProviderProfile,
    register_provider_profile,
)

# Load built-in provider registrations as a one-time package bootstrap step.
_ensure_builtin_profiles_loaded()

__all__ = [
    "GeneralPurposeSubagentProfile",
    "HarnessProfile",
    "HarnessProfileConfig",
    "ProviderProfile",
    "register_harness_profile",
    "register_provider_profile",
]
