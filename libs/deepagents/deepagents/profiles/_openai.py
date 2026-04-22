"""OpenAI provider profile.

!!! warning

    This is an internal API subject to change without deprecation. It is not
    intended for external use or consumption.
"""

from deepagents.profiles._provider_profiles import _ProviderProfile, _register_provider_profile

_register_provider_profile(
    "openai",
    _ProviderProfile(init_kwargs={"use_responses_api": True}),
)
