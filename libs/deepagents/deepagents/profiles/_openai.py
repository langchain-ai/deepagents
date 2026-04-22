"""Built-in OpenAI provider profile.

Registers the default configuration used when constructing OpenAI chat models
via `resolve_model`. Users may layer additional kwargs on top via
`register_provider_profile("openai", ...)`.
"""

from deepagents.profiles.provider_profiles import ProviderProfile, register_provider_profile

register_provider_profile(
    "openai",
    ProviderProfile(init_kwargs={"use_responses_api": True}),
)
