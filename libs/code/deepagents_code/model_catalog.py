"""Model discovery data shared by the inference host and model pickers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepagents_code.model_config import (
    CODEX_PROVIDER,
    PROVIDER_API_KEY_ENV,
    ModelConfig,
    ModelProfileEntry,
    ProviderAuthSource,
    ProviderAuthState,
    ProviderAuthStatus,
    _get_builtin_providers,
    get_available_models,
    get_model_profiles,
    get_provider_auth_status,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

logger = logging.getLogger(__name__)

# Only presentation fields cross the process boundary. Constructor kwargs and
# arbitrary config/profile extensions can contain credentials or executable data.
_PROFILE_FIELDS = frozenset(
    {
        "name",
        "status",
        "max_input_tokens",
        "max_output_tokens",
        "text_inputs",
        "image_inputs",
        "audio_inputs",
        "pdf_inputs",
        "video_inputs",
        "reasoning_output",
        "tool_calling",
        "structured_output",
    }
)


class CatalogProfile(BaseModel):
    """Displayable model capabilities with override provenance."""

    model_config = ConfigDict(extra="forbid", strict=True)
    profile: dict[str, str | int | bool | None]
    overridden_keys: list[str]


class CatalogProvider(BaseModel):
    """Provider presentation and readiness, without credential values."""

    model_config = ConfigDict(extra="forbid", strict=True)
    state: ProviderAuthState = Field(strict=False)
    source: ProviderAuthSource | None = Field(default=None, strict=False)
    env_var: str | None = None
    detail: str | None = None
    display_name: str | None = None
    short_name: str | None = None
    install_extra: str | None = None

    @model_validator(mode="after")
    def _validate_readiness(self) -> Self:
        """Return readiness validated before the picker renders it."""
        self.auth_status("provider")
        return self

    def auth_status(self, provider: str) -> ProviderAuthStatus:
        """Return the readiness indicator for a provider.

        Args:
            provider: Provider key associated with this catalog entry.
        """
        return ProviderAuthStatus(
            state=self.state,
            provider=provider,
            source=self.source,
            env_var=self.env_var,
            detail=self.detail,
        )


class ModelCatalog(BaseModel):
    """Validated catalog from the environment that constructs models."""

    model_config = ConfigDict(extra="forbid", strict=True)
    models: list[str]
    profiles: dict[str, CatalogProfile]
    providers: dict[str, CatalogProvider]
    allowed_models: list[str] | None = None
    managed_policy: bool = False
    current_spec: str | None = None

    def profile_entries(self) -> dict[str, ModelProfileEntry]:
        """Return profile entries understood by the model detail footer."""
        return {
            spec: ModelProfileEntry(
                profile=entry.profile,
                overridden_keys=frozenset(entry.overridden_keys),
            )
            for spec, entry in self.profiles.items()
        }

    def presentation_config(self) -> ModelConfig:
        """Return display and policy metadata without loading client config."""
        from deepagents_code.model_config import MANAGED_CONFIG_SOURCE, ProviderConfig

        providers: dict[str, ProviderConfig] = {}
        for provider, info in self.providers.items():
            entry: ProviderConfig = {}
            if info.display_name:
                entry["display_name"] = info.display_name
            if info.short_name:
                entry["short_name"] = info.short_name
            providers[provider] = entry
        return ModelConfig(
            providers=providers,
            allowed_models=(
                tuple(self.allowed_models) if self.allowed_models is not None else None
            ),
            allowed_models_source=(
                MANAGED_CONFIG_SOURCE if self.managed_policy else "server config"
            )
            if self.allowed_models is not None
            else None,
        )


def _profile_entry(entry: ModelProfileEntry) -> CatalogProfile:
    """Return only scalar profile fields that the picker displays."""
    profile = {
        key: value
        for key, value in entry["profile"].items()
        if key in _PROFILE_FIELDS and (value is None or type(value) in {str, int, bool})
    }
    return CatalogProfile(
        profile=profile,
        overridden_keys=sorted(entry["overridden_keys"] & profile.keys()),
    )


def _provider_entry(provider: str, config: ModelConfig) -> CatalogProvider:
    """Return provider readiness from the active inference environment."""
    from deepagents_code.config_manifest import (
        is_provider_package_installed,
        provider_install_extra,
    )

    status = get_provider_auth_status(provider)
    extra = provider_install_extra(provider)
    return CatalogProvider(
        state=status.state,
        source=status.source,
        env_var=status.env_var,
        detail=status.detail,
        display_name=config.get_provider_display_name(provider),
        short_name=config.get_provider_short_name(provider),
        install_extra=(
            extra if extra and not is_provider_package_installed(provider) else None
        ),
    )


def load_model_catalog(
    *,
    profile_overrides: dict[str, object] | None = None,
    recommended_models: Sequence[str] = (),
    current_spec: str | None = None,
) -> ModelCatalog:
    """Discover models in the caller's active environment without constructing them.

    Args:
        profile_overrides: Main-model profile overrides; omit for auxiliary roles.
        recommended_models: Additional known models to surface when enabled.
        current_spec: Active model to retain even if discovery does not list it.

    Returns:
        Displayable capabilities, policy, and provider readiness.
    """
    from deepagents_code.model_config import MANAGED_CONFIG_SOURCE

    config = ModelConfig.load()
    if current_spec:
        current_spec = config.canonical_model_spec(current_spec) or current_spec
    available = get_available_models()
    models = [
        f"{provider}:{model}"
        for provider, names in available.items()
        for model in names
    ]
    providers = {provider: _provider_entry(provider, config) for provider in available}
    for spec in dict.fromkeys(
        [*recommended_models, *([current_spec] if current_spec else [])]
    ):
        provider, separator, _name = spec.partition(":")
        if not separator or spec in models or not config.is_model_allowed(spec):
            continue
        if not config.is_provider_enabled(provider):
            continue
        try:
            if provider not in providers:
                providers[provider] = _provider_entry(provider, config)
            info = providers[provider]
            if provider in available or info.install_extra or spec == current_spec:
                models.append(spec)
        except Exception:
            logger.warning(
                "Skipping unavailable catalog provider %s", provider, exc_info=True
            )
    # Custom entries need setup prompts even when an integration is missing or
    # exposes no profiles. Keep readiness on the inference host, independent of
    # the discovered model rows.
    known_providers = (
        _get_builtin_providers().keys()
        | PROVIDER_API_KEY_ENV.keys()
        | config.providers.keys()
        | {CODEX_PROVIDER}
    )
    for provider in sorted(known_providers - providers.keys()):
        if not config.is_provider_enabled(provider):
            continue
        try:
            providers[provider] = _provider_entry(provider, config)
        except Exception:
            logger.warning(
                "Skipping unavailable catalog provider %s", provider, exc_info=True
            )
    profiles = get_model_profiles(cli_override=profile_overrides)
    return ModelCatalog(
        models=models,
        current_spec=current_spec,
        profiles={
            spec: _profile_entry(profiles[spec]) for spec in models if spec in profiles
        },
        providers=providers,
        allowed_models=list(config.allowed_models)
        if config.allowed_models is not None
        else None,
        managed_policy=config.allowed_models_source == MANAGED_CONFIG_SOURCE,
    )
