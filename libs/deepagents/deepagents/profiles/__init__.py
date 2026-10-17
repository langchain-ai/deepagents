"""Public beta APIs for model and harness profiles.

!!! beta

    `deepagents.profiles` exposes beta APIs that may receive minor changes in
    future releases. Refer to the [versioning documentation](https://docs.langchain.com/oss/python/versioning)
    for more details.

Profiles let Deep Agents tailor behavior to a specific provider or model spec
across two orthogonal phases:

- **Provider profiles** (`ProviderProfile`) control the *model-construction*
    phase. They declare how `resolve_model` builds a chat model — `init_chat_model`
    kwargs, pre-initialization side effects, and kwargs derived from runtime
    state (e.g. environment variables).
- **Harness profiles** (`HarnessProfile`, `HarnessProfileConfig`) control the
    *runtime* phase. They declare how `create_deep_agent` shapes the agent
    *after* the model is built — prompt assembly, tool visibility, middleware,
    and default subagent behavior.

Both kinds live in keyed registries that accept a `provider` or
`provider:model` key. Registration helpers (`register_provider_profile`,
`register_harness_profile`) are additive: re-registering under an existing key
merges on top of the prior registration rather than replacing it.

Directory layout:

- `provider/` — `ProviderProfile` API (`provider_profiles.py`) plus built-in
    provider modules (e.g. `_openai`, `_openrouter`).
- `harness/` — `HarnessProfile` API (`harness_profiles.py`) plus built-in
    harness modules for frontier model specs (e.g. `_anthropic_sonnet_4_6`,
    `_openai_codex`).
- `_builtin_profiles.py` — bootstrap that registers built-in profiles and loads
    third-party plugins (via `importlib.metadata` entry points) lazily on first
    profile-registry access, so importing this package stays cheap.
- `_keys.py` — shared validation and lookup helpers for the `provider` /
    `provider:model` registry keys used by both registries.
"""

from deepagents.profiles.harness.harness_profiles import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    HarnessProfileConfig,
    register_harness_profile,
)
from deepagents.profiles.provider.provider_profiles import (
    ProviderProfile,
    register_provider_profile,
)

# Built-in provider/harness profiles are registered lazily on first
# profile-registry access so importing `deepagents.profiles` stays cheap.

__all__ = [
    "GeneralPurposeSubagentProfile",
    "HarnessProfile",
    "HarnessProfileConfig",
    "ProviderProfile",
    "register_harness_profile",
    "register_provider_profile",
]
