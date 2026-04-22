"""Built-in OpenAI provider profile.

Enables the OpenAI Responses API by default for all `openai:*` models via
`use_responses_api=True`. Users may layer additional kwargs on top
via `register_provider_profile("openai", ...)`.
"""

from deepagents.profiles.provider.provider_profiles import ProviderProfile, register_provider_profile

register_provider_profile(
    "openai",
    ProviderProfile(init_kwargs={"use_responses_api": True}),
)
