"""Public beta APIs for model and harness profiles.

!!! beta

    `deepagents.profiles` exposes beta APIs that may receive minor changes in
    future releases. Refer to the [versioning documentation](https://docs.langchain.com/oss/python/versioning)
    for more details.

Exposes the public `ProviderProfile` and `HarnessProfile` APIs for customizing
how `resolve_model` constructs chat models and how `create_deep_agent` shapes
agent runtime behavior. Registration helpers are additive: re-registering
under an existing key merges on top of the prior registration.
"""

# Provider modules register their provider profiles as a side effect of import.
from deepagents.profiles import _openai as _openai, _openrouter as _openrouter
from deepagents.profiles.harness_profiles import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    register_harness_profile,
)
from deepagents.profiles.provider_profiles import (
    ProviderProfile,
    register_provider_profile,
)

__all__ = [
    "GeneralPurposeSubagentProfile",
    "HarnessProfile",
    "ProviderProfile",
    "register_harness_profile",
    "register_provider_profile",
]
