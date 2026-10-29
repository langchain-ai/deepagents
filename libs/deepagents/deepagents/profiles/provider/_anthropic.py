"""Built-in Anthropic streaming default for first-token observability."""

from deepagents.profiles.provider.provider_profiles import (
    ProviderProfile,
    _register_provider_profile_impl,
)


def register() -> None:
    """Register the built-in Anthropic provider profile."""
    _register_provider_profile_impl(
        "anthropic",
        ProviderProfile(init_kwargs={"streaming": True}),
    )
