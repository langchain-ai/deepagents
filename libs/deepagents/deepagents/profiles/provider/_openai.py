"""Built-in OpenAI provider profile.

Enables the OpenAI Responses API by default for all `openai:*` models via
`use_responses_api=True`. Users may layer additional kwargs on top
via `register_provider_profile("openai", ...)`.

Registered directly by `_ensure_builtin_profiles_loaded` at
`deepagents.profiles` import time. Not exposed as an `importlib.metadata`
entry point — built-ins ship with the SDK and should not depend on
install-time metadata to activate.
"""

from deepagents.profiles.provider.provider_profiles import (
    ProviderProfile,
    register_provider_profile,
)


def register() -> None:
    """Register the built-in OpenAI provider profile."""
    register_provider_profile(
        "openai",
        ProviderProfile(init_kwargs={"use_responses_api": True}),
    )
