"""Model configuration management.

Handles loading and saving model configuration from TOML files, providing a
structured way to define available models and providers.
"""

from __future__ import annotations

import importlib
import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import tomli_w

logger = logging.getLogger(__name__)


class ModelConfigError(Exception):
    """Raised when model configuration or creation fails."""


@dataclass(frozen=True)
class ModelSpec:
    """A model specification in `provider:model` format.

    Examples:
        >>> spec = ModelSpec.parse("anthropic:claude-sonnet-4-5")
        >>> spec.provider
        'anthropic'
        >>> spec.model
        'claude-sonnet-4-5'
        >>> str(spec)
        'anthropic:claude-sonnet-4-5'
    """

    provider: str
    """The provider name (e.g., `'anthropic'`, `'openai'`)."""

    model: str
    """The model identifier (e.g., `'claude-sonnet-4-5'`, `'gpt-4o'`)."""

    def __post_init__(self) -> None:
        """Validate the model spec after initialization.

        Raises:
            ValueError: If provider or model is empty.
        """
        if not self.provider:
            msg = "Provider cannot be empty"
            raise ValueError(msg)
        if not self.model:
            msg = "Model cannot be empty"
            raise ValueError(msg)

    @classmethod
    def parse(cls, spec: str) -> ModelSpec:
        """Parse a model specification string.

        Args:
            spec: Model specification in `'provider:model'` format.

        Returns:
            Parsed ModelSpec instance.

        Raises:
            ValueError: If the spec is not in valid `'provider:model'` format.
        """
        if ":" not in spec:
            msg = (
                f"Invalid model spec '{spec}': must be in provider:model format "
                "(e.g., 'anthropic:claude-sonnet-4-5')"
            )
            raise ValueError(msg)
        provider, model = spec.split(":", 1)
        return cls(provider=provider, model=model)

    @classmethod
    def try_parse(cls, spec: str) -> ModelSpec | None:
        """Try to parse a model specification, returning None on failure.

        Args:
            spec: Model specification to parse.

        Returns:
            Parsed ModelSpec if valid, None otherwise.
        """
        try:
            return cls.parse(spec)
        except ValueError:
            return None

    def __str__(self) -> str:
        """Return the model spec as a string in `provider:model` format."""
        return f"{self.provider}:{self.model}"


class ProviderConfig(TypedDict, total=False):
    """Configuration for a model provider."""

    models: list[str]
    """List of model identifiers available from this provider."""

    api_key_env: str
    """Environment variable name containing the API key."""

    base_url: str
    """Custom base URL."""


DEFAULT_CONFIG_DIR = Path.home() / ".deepagents"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"

PROVIDER_API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "google_vertexai": "GOOGLE_CLOUD_PROJECT",
}
"""Mapping of provider names to their API key environment variables."""


def get_available_models() -> dict[str, list[str]]:
    """Get available models dynamically from installed LangChain provider packages.

    Attempts to import model profiles from each provider package and extracts
    model names. Falls back to hardcoded defaults if profiles are unavailable.

    Returns:
        Dictionary mapping provider names to lists of model identifiers.
            Only includes providers whose packages are installed.
    """
    available: dict[str, list[str]] = {}

    # Try to load from langchain provider profile data
    provider_modules = [
        ("anthropic", "langchain_anthropic.data._profiles"),
        ("openai", "langchain_openai.data._profiles"),
        ("google_genai", "langchain_google_genai.data._profiles"),
    ]

    for provider, module_path in provider_modules:
        try:
            module = importlib.import_module(module_path)
            profiles: dict[str, Any] = getattr(module, "_PROFILES", {})

            # Filter to models that support tool calling (required for agents)
            models = [
                name
                for name, profile in profiles.items()
                if profile.get("tool_calling", False)
            ]

            # Sort by model name for consistent display
            models.sort()
            if models:
                available[provider] = models
        except ImportError:
            logger.debug(
                "Could not import profiles from %s (package may not be installed)",
                module_path,
            )

    # Fall back to hardcoded defaults if no profiles found
    if not available:
        available = get_default_models()

    return available


def get_default_models() -> dict[str, list[str]]:
    """Get hardcoded default curated model list.

    This is a curated list of commonly used models that are known to work well
    with the CLI. Used as fallback when profile data is unavailable.

    Returns:
        Dictionary mapping provider names to lists of model identifiers.
    """
    return {
        "anthropic": [
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
        ],
        "openai": [
            "gpt-5.2",
            "gpt-4o",
            "o3",
            "o3-mini",
        ],
        "google_genai": [
            "gemini-3-pro-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
    }


def get_curated_models() -> dict[str, list[str]]:
    """Get curated subset of popular models for display in the selector.

    Returns a smaller, curated list of the most commonly used models
    rather than the full list from profiles. Users can still type
    any valid model name.

    Returns:
        Dictionary mapping provider names to lists of curated model identifiers.
    """
    return {
        "anthropic": [
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
        ],
        "openai": [
            "gpt-5.2",
            "gpt-5",
            "gpt-4o",
            "o3",
            "o3-mini",
        ],
        "google_genai": [
            "gemini-3-pro-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
        "google_vertexai": [
            "gemini-3-pro-preview",
            "claude-sonnet-4-5",  # (Claude on Vertex)
        ],
    }


def has_provider_credentials(provider: str) -> bool:
    """Check if credentials are available for a provider.

    Args:
        provider: Provider name.

    Returns:
        True if the required environment variable is set.
    """
    env_var = PROVIDER_API_KEY_ENV.get(provider)
    if not env_var:
        return False
    return bool(os.environ.get(env_var))


@dataclass
class ModelConfig:
    """Parsed model configuration from `config.toml`."""

    default_model: str | None = None
    """The default model to use when none is specified."""

    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    """Dictionary mapping provider names to their configurations."""

    @classmethod
    def load(cls, config_path: Path | None = None) -> ModelConfig:
        """Load config from file.

        Args:
            config_path: Path to config file. Defaults to ~/.deepagents/config.toml.

        Returns:
            Parsed `ModelConfig` instance.
                Returns empty config if file is missing, unreadable, or contains
                invalid TOML syntax.
        """
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH

        if not config_path.exists():
            return cls()

        try:
            with config_path.open("rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            logger.warning(
                "Config file %s has invalid TOML syntax: %s. "
                "Ignoring config file. Fix the file or delete it to reset.",
                config_path,
                e,
            )
            return cls()
        except (PermissionError, OSError) as e:
            logger.warning("Could not read config file %s: %s", config_path, e)
            return cls()

        config = cls(
            default_model=data.get("default", {}).get("model"),
            providers=data.get("providers", {}),
        )

        # Validate config consistency
        config._validate()

        return config

    def _validate(self) -> None:
        """Validate internal consistency of the config.

        Issues warnings for invalid configurations but does not raise exceptions,
        allowing the app to continue with potentially degraded functionality.
        """
        # Warn if default_model is set but doesn't use provider:model format
        if self.default_model and ":" not in self.default_model:
            logger.warning(
                "default_model '%s' should use provider:model format "
                "(e.g., 'anthropic:claude-sonnet-4-5')",
                self.default_model,
            )

    def get_all_models(self) -> list[tuple[str, str]]:
        """Get all models as `(model_name, provider_name)` tuples.

        Returns:
            List of tuples containing `(model_name, provider_name)`.
        """
        return [
            (model, provider_name)
            for provider_name, provider_config in self.providers.items()
            for model in provider_config.get("models", [])
        ]

    def get_provider_for_model(self, model_name: str) -> str | None:
        """Find the provider that contains this model.

        Args:
            model_name: The model identifier to look up.

        Returns:
            Provider name if found, None otherwise.
        """
        for provider_name, provider_config in self.providers.items():
            if model_name in provider_config.get("models", []):
                return provider_name
        return None

    def has_credentials(self, provider_name: str) -> bool:
        """Check if credentials are available for a provider.

        This is the config-file-driven credential check, supporting custom
        providers (e.g., local Ollama with no key required). For the hardcoded
        `PROVIDER_API_KEY_ENV`-based check used in the hot-swap path, see the
        module-level `has_provider_credentials()`.

        Args:
            provider_name: The provider to check.

        Returns:
            True if credentials are available or not required.
        """
        provider = self.providers.get(provider_name)
        if not provider:
            return False
        env_var = provider.get("api_key_env")
        if not env_var:
            return True  # No key required (e.g., local Ollama)
        return bool(os.environ.get(env_var))

    def get_base_url(self, provider_name: str) -> str | None:
        """Get custom base URL.

        Args:
            provider_name: The provider to get base URL for.

        Returns:
            Base URL if configured, None otherwise.
        """
        provider = self.providers.get(provider_name)
        return provider.get("base_url") if provider else None

    def get_api_key_env(self, provider_name: str) -> str | None:
        """Get the environment variable name for a provider's API key.

        Args:
            provider_name: The provider to get API key env var for.

        Returns:
            Environment variable name if configured, None otherwise.
        """
        provider = self.providers.get(provider_name)
        return provider.get("api_key_env") if provider else None


def save_default_model(model_name: str, config_path: Path | None = None) -> bool:
    """Update the default model in config file.

    Reads existing config (if any), updates the `default.model` value, and
    writes back using proper TOML serialization.

    Args:
        model_name: The model to set as default (`provider:model` format recommended).
        config_path: Path to config file. Defaults to `~/.deepagents/config.toml`.

    Returns:
        True if save succeeded, False if it failed due to I/O errors.

    Note:
        This function does not preserve comments in the config file.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Read existing config or start fresh
        if config_path.exists():
            with config_path.open("rb") as f:
                data = tomllib.load(f)
        else:
            data = {}

        # Update the default model
        if "default" not in data:
            data["default"] = {}
        data["default"]["model"] = model_name

        # Write back with proper TOML formatting
        with config_path.open("wb") as f:
            tomli_w.dump(data, f)
    except (OSError, PermissionError, tomllib.TOMLDecodeError) as e:
        logger.warning("Could not save model preference: %s", e)
        return False
    else:
        return True
