"""Model configuration management for deepagents-cli.

This module handles loading and saving model configuration from TOML files,
providing a structured way to define available models and providers.

The module supports the "provider:model" syntax for model specification,
allowing explicit provider selection (e.g., "anthropic:claude-sonnet-4-5").
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict


class ProviderConfig(TypedDict, total=False):
    """Configuration for a model provider.

    Attributes:
        models: List of model identifiers available from this provider.
        api_key_env: Environment variable name containing the API key.
        base_url: Custom base URL for OpenAI-compatible APIs.
    """

    models: list[str]
    api_key_env: str
    base_url: str


# Default config directory
DEFAULT_CONFIG_DIR = Path.home() / ".deepagents"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"

# Mapping of provider names to their API key environment variables
PROVIDER_API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "google_vertexai": "GOOGLE_CLOUD_PROJECT",
}


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
            import importlib

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
            # Package not installed, skip this provider
            pass

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
            "claude-sonnet-4-5",  # Claude on Vertex
        ],
    }


def has_provider_credentials(provider: str) -> bool:
    """Check if credentials are available for a provider.

    Args:
        provider: Provider name (anthropic, openai, google_genai, google_vertexai)

    Returns:
        True if the required environment variable is set.
    """
    env_var = PROVIDER_API_KEY_ENV.get(provider)
    if not env_var:
        return False
    return bool(os.environ.get(env_var))


@dataclass
class ModelConfig:
    """Parsed model configuration from config.toml.

    Attributes:
        default_model: The default model to use when none is specified.
        providers: Dictionary mapping provider names to their configurations.
    """

    default_model: str | None = None
    providers: dict[str, ProviderConfig] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Path | None = None) -> ModelConfig:
        """Load config from file, returning empty config if not found.

        Args:
            config_path: Path to config file. Defaults to ~/.deepagents/config.toml.

        Returns:
            Parsed ModelConfig instance.
        """
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH

        if not config_path.exists():
            return cls()

        with config_path.open("rb") as f:
            data = tomllib.load(f)

        return cls(
            default_model=data.get("default", {}).get("model"),
            providers=data.get("providers", {}),
        )

    def get_all_models(self) -> list[tuple[str, str]]:
        """Get all models as (model_name, provider_name) tuples.

        Returns:
            List of tuples containing (model_name, provider_name).
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
        """Get custom base URL for OpenAI-compatible providers.

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


def save_default_model(model_name: str, config_path: Path | None = None) -> None:
    """Update the default model in config file.

    If the config file doesn't exist, this creates a minimal config.
    If it exists, only the default.model value is updated.

    Args:
        model_name: The model to set as default.
        config_path: Path to config file. Defaults to ~/.deepagents/config.toml.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        # Read existing content and update default.model
        content = config_path.read_text()

        # Check if [default] section exists
        if "[default]" in content:
            # Update existing model line
            import re

            pattern = r'(\[default\][^\[]*model\s*=\s*)["\']?[^"\'\n]*["\']?'
            replacement = rf'\1"{model_name}"'
            new_content, count = re.subn(pattern, replacement, content)
            if count == 0:
                # [default] exists but no model line, add it
                new_content = content.replace(
                    "[default]", f'[default]\nmodel = "{model_name}"'
                )
        else:
            # Add [default] section at the beginning
            new_content = f'[default]\nmodel = "{model_name}"\n\n{content}'

        config_path.write_text(new_content)
    else:
        # Create minimal config with just the default
        config_path.write_text(f'[default]\nmodel = "{model_name}"\n')


def run_first_run_wizard() -> ModelConfig | None:
    """Interactive first-run setup when no config exists.

    Called from main.py if no config file and no env vars detected.
    Uses simple input() prompts since app isn't running yet.

    Returns:
        Created ModelConfig if setup completed, None if skipped.
    """
    print("\nNo configuration found for deepagents.")
    print("\nAvailable providers:")
    print("  1. Anthropic (Claude models)")
    print("  2. OpenAI (GPT models)")
    print("  3. Google (Gemini models)")
    print("  4. Skip (use environment variables)")

    try:
        choice = input("\nSelect provider [1-4]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if not choice or choice == "4":
        return None

    # Map choice to (provider, env_var, default_model_spec using provider:model format)
    provider_map: dict[str, tuple[str, str, str]] = {
        "1": ("anthropic", "ANTHROPIC_API_KEY", "anthropic:claude-sonnet-4-5"),
        "2": ("openai", "OPENAI_API_KEY", "openai:gpt-4o"),
        "3": ("google_genai", "GOOGLE_API_KEY", "google_genai:gemini-3-pro-preview"),
    }

    if choice not in provider_map:
        print(f"Invalid choice: {choice}")
        return None

    provider_name, env_var, default_model_spec = provider_map[choice]

    # Check if env var exists
    if not os.environ.get(env_var):
        print(f"\nNote: {env_var} is not set.")
        print(f"Set it with: export {env_var}=your_key_here")

    # Create config with provider:model syntax
    config_path = DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Get curated models for this provider
    curated = get_curated_models().get(provider_name, [])
    models_list = curated or [default_model_spec.split(":", 1)[1]]

    config_content = f"""[default]
model = "{default_model_spec}"

[providers.{provider_name}]
models = {models_list}
api_key_env = "{env_var}"
"""

    config_path.write_text(config_content)
    print(f"\nConfiguration saved to {config_path}")

    return ModelConfig.load(config_path)
