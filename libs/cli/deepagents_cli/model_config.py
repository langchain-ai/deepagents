"""Model configuration management.

Handles loading and saving model configuration from TOML files, providing a
structured way to define available models and providers.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import os
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypedDict

import tomli_w

if TYPE_CHECKING:
    from collections.abc import Mapping

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
    """Configuration for a model provider.

    The optional `class_path` field allows bypassing `init_chat_model` entirely
    and instantiating an arbitrary `BaseChatModel` subclass via importlib.

    !!! warning

        Setting `class_path` executes arbitrary Python code from the user's
        config file. This has the same trust model as `pyproject.toml` build
        scripts — the user controls their own machine.
    """

    models: list[str]
    """List of model identifiers available from this provider."""

    api_key_env: str
    """Environment variable name containing the API key."""

    base_url: str
    """Custom base URL."""

    # Level 2: arbitrary BaseChatModel classes

    class_path: str
    """Fully-qualified Python class in `module.path:ClassName` format.

    When set, `create_model` imports this class and instantiates it directly
    instead of calling `init_chat_model`.
    """

    params: dict[str, Any]
    """Extra keyword arguments forwarded to the model constructor.

    Flat keys (e.g., `temperature = 0`) are provider-wide defaults applied to
    every model from this provider. Model-keyed sub-tables (e.g.,
    `[params."qwen3:4b"]`) override individual values for that model only;
    the merge is shallow (model wins on conflict).
    """


DEFAULT_CONFIG_DIR = Path.home() / ".deepagents"
"""Directory for user-level Deep Agents configuration (`~/.deepagents`)."""

DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"
"""Path to the user's model configuration file (`~/.deepagents/config.toml`)."""

PROVIDER_API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "cohere": "COHERE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "google_vertexai": "GOOGLE_CLOUD_PROJECT",
    "groq": "GROQ_API_KEY",
    "huggingface": "HUGGINGFACEHUB_API_TOKEN",
    "ibm": "WATSONX_APIKEY",
    "mistralai": "MISTRAL_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "perplexity": "PPLX_API_KEY",
    "together": "TOGETHER_API_KEY",
    "xai": "XAI_API_KEY",
}
"""Well-known providers mapped to the env var that holds their API key.

Used by `has_provider_credentials` to verify credentials *before* model
creation, so the UI can show a warning icon and a specific error message
(e.g., "ANTHROPIC_API_KEY not set") instead of letting the provider fail at call
time.

Providers not listed here fall through to the config-file check or the langchain
registry fallback.
"""


# Module-level caches — cleared by `clear_caches()`.
_available_models_cache: dict[str, list[str]] | None = None
_builtin_providers_cache: dict[str, Any] | None = None
_default_config_cache: ModelConfig | None = None


def clear_caches() -> None:
    """Reset module-level caches so the next call recomputes from scratch.

    Intended for tests and for a hypothetical "refresh models" UI action.
    """
    global _available_models_cache, _builtin_providers_cache, _default_config_cache  # noqa: PLW0603  # Module-level caches require global statement
    _available_models_cache = None
    _builtin_providers_cache = None
    _default_config_cache = None


def _get_builtin_providers() -> dict[str, Any]:
    """Return langchain's built-in provider registry.

    Tries the newer `_BUILTIN_PROVIDERS` name first, then falls back to
    the legacy `_SUPPORTED_PROVIDERS` for older langchain versions.

    Results are cached after the first call; use `clear_caches()` to reset.

    Returns:
        The provider registry dict from `langchain.chat_models.base`.
    """
    global _builtin_providers_cache  # noqa: PLW0603  # Module-level cache requires global statement
    if _builtin_providers_cache is not None:
        return _builtin_providers_cache

    # Deferred: langchain.chat_models pulls in heavy provider registry,
    # only needed when resolving provider names for model config.
    from langchain.chat_models import base

    registry: dict[str, Any] | None = getattr(base, "_BUILTIN_PROVIDERS", None)
    if registry is None:
        registry = getattr(base, "_SUPPORTED_PROVIDERS", None)
    _builtin_providers_cache = registry if registry is not None else {}
    return _builtin_providers_cache


def _get_provider_profile_modules() -> list[tuple[str, str]]:
    """Build a `(provider, profile_module)` list from langchain's provider registry.

    Reads the built-in provider registry from `langchain.chat_models.base`
    to discover every provider that `init_chat_model` knows about, then derives
    the `<package>.data._profiles` module path for each.

    Returns:
        List of `(provider_name, profile_module_path)` tuples.
    """
    providers = _get_builtin_providers()

    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for provider_name, (module_path, *_rest) in providers.items():
        package_root = module_path.split(".", maxsplit=1)[0]
        profile_module = f"{package_root}.data._profiles"
        key = (provider_name, profile_module)
        if key not in seen:
            seen.add(key)
            result.append((provider_name, profile_module))

    return result


def _load_provider_profiles(module_path: str) -> dict[str, Any]:
    """Load `_PROFILES` from a provider's data module.

    Locating the package on disk with `importlib.util.find_spec` and load *only*
    the `_profiles.py` file via `spec_from_file_location`.

    Args:
        module_path: Dotted module path (e.g., `"langchain_openai.data._profiles"`).

    Returns:
        The `_PROFILES` dictionary from the module, or an empty dict if
            the module has no such attribute.

    Raises:
        ImportError: If the package is not installed or the profile module
            cannot be found on disk.
    """
    parts = module_path.split(".")
    package_root = parts[0]

    spec = importlib.util.find_spec(package_root)
    if spec is None:
        msg = f"Package {package_root} is not installed"
        raise ImportError(msg)

    # Determine the package directory from the spec.
    if spec.origin:
        package_dir = Path(spec.origin).parent
    elif spec.submodule_search_locations:
        package_dir = Path(next(iter(spec.submodule_search_locations)))
    else:
        msg = f"Cannot determine location for {package_root}"
        raise ImportError(msg)

    # Build the path to the target file (e.g., data/_profiles.py).
    relative_parts = parts[1:]  # ["data", "_profiles"]
    profiles_path = package_dir.joinpath(
        *relative_parts[:-1], f"{relative_parts[-1]}.py"
    )

    if not profiles_path.exists():
        msg = f"Profile module not found: {profiles_path}"
        raise ImportError(msg)

    file_spec = importlib.util.spec_from_file_location(module_path, profiles_path)
    if file_spec is None or file_spec.loader is None:
        msg = f"Could not create module spec for {profiles_path}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(file_spec)
    file_spec.loader.exec_module(module)
    return getattr(module, "_PROFILES", {})


def get_available_models() -> dict[str, list[str]]:
    """Get available models dynamically from installed LangChain provider packages.

    Imports model profiles from each provider package and extracts model names.

    Results are cached after the first call; use `clear_caches()` to reset.

    Returns:
        Dictionary mapping provider names to lists of model identifiers.
            Only includes providers whose packages are installed.
    """
    global _available_models_cache  # noqa: PLW0603  # Module-level cache requires global statement
    if _available_models_cache is not None:
        return _available_models_cache

    available: dict[str, list[str]] = {}

    # Try to load from langchain provider profile data.
    # Build the list dynamically from langchain's supported-provider registry
    # so new providers are picked up automatically when langchain adds them.
    provider_modules = _get_provider_profile_modules()

    for provider, module_path in provider_modules:
        try:
            profiles = _load_provider_profiles(module_path)
        except ImportError:
            logger.debug(
                "Could not import profiles from %s (package may not be installed)",
                module_path,
            )
            continue
        except Exception:
            logger.warning(
                "Failed to load profiles from %s, skipping provider '%s'",
                module_path,
                provider,
                exc_info=True,
            )
            continue

        # Filter to models that support tool calling and text I/O.
        models = [
            name
            for name, profile in profiles.items()
            if profile.get("tool_calling", False)
            and profile.get("text_inputs", True) is not False
            and profile.get("text_outputs", True) is not False
        ]

        models.sort()
        if models:
            available[provider] = models

    # Merge in models from config file (custom providers like ollama, fireworks)
    config = ModelConfig.load()
    for provider_name, provider_config in config.providers.items():
        config_models = provider_config.get("models", [])
        if provider_name not in available:
            if config_models:
                available[provider_name] = config_models
        else:
            # Append any config models not already discovered
            existing = set(available[provider_name])
            for model in config_models:
                if model not in existing:
                    available[provider_name].append(model)

    _available_models_cache = available
    return available


def _is_langchain_supported_provider(provider: str) -> bool:
    """Check if a provider is in langchain's built-in provider registry.

    Args:
        provider: Provider name to check.

    Returns:
        True if the provider is known to `init_chat_model`.
    """
    return provider in _get_builtin_providers()


def has_provider_credentials(provider: str) -> bool | None:
    """Check if credentials are available for a provider.

    Checks in order:

    1. Config-file providers (`config.toml`) — takes priority so user
        overrides (e.g., custom `api_key_env` or `base_url`) are respected.
    2. Hardcoded `PROVIDER_API_KEY_ENV` mapping (anthropic, openai, etc.).
    3. Langchain's `_SUPPORTED_PROVIDERS` registry — if the provider is known
        to `init_chat_model`, credential status is unknown; the provider
        itself will report auth failures at model-creation time.

    Args:
        provider: Provider name.

    Returns:
        True if credentials are confirmed available, False if confirmed
            missing, or None if credential status cannot be determined.
    """
    # Config-file providers take priority when api_key_env is specified.
    config = ModelConfig.load()
    if config.providers.get(provider):
        result = config.has_credentials(provider)
        if result is not None:
            return result
        # No api_key_env in config — fall through to hardcoded map.

    # Fall back to hardcoded well-known providers.
    env_var = PROVIDER_API_KEY_ENV.get(provider)
    if env_var:
        return bool(os.environ.get(env_var))

    # If langchain knows this provider, let it through — we can't verify
    # credentials, but the provider will raise its own auth error if needed.
    if _is_langchain_supported_provider(provider):
        return None

    return False


def get_credential_env_var(provider: str) -> str | None:
    """Return the env var name that holds credentials for a provider.

    Checks the config file first (user override), then falls back to the
    hardcoded `PROVIDER_API_KEY_ENV` map.

    Args:
        provider: Provider name.

    Returns:
        Environment variable name, or None if unknown.
    """
    config = ModelConfig.load()
    config_env = config.get_api_key_env(provider)
    if config_env:
        return config_env
    return PROVIDER_API_KEY_ENV.get(provider)


@dataclass(frozen=True)
class ModelConfig:
    """Parsed model configuration from `config.toml`.

    Instances are immutable once constructed. The `providers` mapping is
    wrapped in `MappingProxyType` to prevent accidental mutation of the
    globally cached singleton returned by `load()`.
    """

    default_model: str | None = None
    """The user's intentional default model (from config file `[models].default`)."""

    recent_model: str | None = None
    """The most recently switched-to model (from config file `[models].recent`)."""

    providers: Mapping[str, ProviderConfig] = field(default_factory=dict)
    """Read-only mapping of provider names to their configurations."""

    def __post_init__(self) -> None:
        """Freeze the providers dict into a read-only proxy."""
        if not isinstance(self.providers, MappingProxyType):
            object.__setattr__(self, "providers", MappingProxyType(self.providers))

    @classmethod
    def load(cls, config_path: Path | None = None) -> ModelConfig:
        """Load config from file.

        When called with the default path, results are cached for the
        lifetime of the process. Use `clear_caches()` to reset.

        Args:
            config_path: Path to config file. Defaults to ~/.deepagents/config.toml.

        Returns:
            Parsed `ModelConfig` instance.
                Returns empty config if file is missing, unreadable, or contains
                invalid TOML syntax.
        """
        global _default_config_cache  # noqa: PLW0603  # Module-level cache requires global statement
        is_default = config_path is None
        if is_default and _default_config_cache is not None:
            return _default_config_cache

        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH

        if not config_path.exists():
            fallback = cls()
            if is_default:
                _default_config_cache = fallback
            return fallback

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
            fallback = cls()
            if is_default:
                _default_config_cache = fallback
            return fallback
        except (PermissionError, OSError) as e:
            logger.warning("Could not read config file %s: %s", config_path, e)
            fallback = cls()
            if is_default:
                _default_config_cache = fallback
            return fallback

        models_section = data.get("models", {})
        config = cls(
            default_model=models_section.get("default"),
            recent_model=models_section.get("recent"),
            providers=models_section.get("providers", {}),
        )

        # Validate config consistency
        config._validate()

        if is_default:
            _default_config_cache = config

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

        # Warn if recent_model is set but doesn't use provider:model format
        if self.recent_model and ":" not in self.recent_model:
            logger.warning(
                "recent_model '%s' should use provider:model format "
                "(e.g., 'anthropic:claude-sonnet-4-5')",
                self.recent_model,
            )

        # Validate class_path format and params references
        for name, provider in self.providers.items():
            class_path = provider.get("class_path")
            if class_path and ":" not in class_path:
                logger.warning(
                    "Provider '%s' has invalid class_path '%s': "
                    "must be in module.path:ClassName format "
                    "(e.g., 'my_package.models:MyChatModel')",
                    name,
                    class_path,
                )

            params = provider.get("params", {})
            models = set(provider.get("models", []))
            for key, value in params.items():
                if isinstance(value, dict) and key not in models:
                    logger.warning(
                        "Provider '%s' has params for '%s' "
                        "which is not in its models list",
                        name,
                        key,
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

    def has_credentials(self, provider_name: str) -> bool | None:
        """Check if credentials are available for a provider.

        This is the config-file-driven credential check, supporting custom
        providers (e.g., local Ollama with no key required). For the hardcoded
        `PROVIDER_API_KEY_ENV`-based check used in the hot-swap path, see the
        module-level `has_provider_credentials()`.

        Args:
            provider_name: The provider to check.

        Returns:
            True if credentials are confirmed available, False if confirmed
                missing, or None if no `api_key_env` is configured and
                credential status cannot be determined.
        """
        provider = self.providers.get(provider_name)
        if not provider:
            return False
        env_var = provider.get("api_key_env")
        if not env_var:
            return None  # No key configured — can't verify
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

    def get_class_path(self, provider_name: str) -> str | None:
        """Get the custom class path for a provider.

        Args:
            provider_name: The provider to look up.

        Returns:
            Class path in `module.path:ClassName` format, or None.
        """
        provider = self.providers.get(provider_name)
        return provider.get("class_path") if provider else None

    def get_kwargs(
        self, provider_name: str, *, model_name: str | None = None
    ) -> dict[str, Any]:
        """Get extra constructor kwargs for a provider.

        Reads the `params` table from the provider config. Flat keys are
        provider-wide defaults; model-keyed sub-tables are per-model
        overrides that shallow-merge on top (model wins on conflict).

        Args:
            provider_name: The provider to look up.
            model_name: Optional model name for per-model overrides.

        Returns:
            Dictionary of extra kwargs (empty if none configured).
        """
        provider = self.providers.get(provider_name)
        if not provider:
            return {}
        params = provider.get("params", {})
        # Flat keys are provider-wide defaults; dict values are per-model overrides
        result = {k: v for k, v in params.items() if not isinstance(v, dict)}
        if model_name:
            overrides = params.get(model_name)
            if isinstance(overrides, dict):
                result.update(overrides)
        return result


def _save_model_field(
    field: str, model_spec: str, config_path: Path | None = None
) -> bool:
    """Read-modify-write a `[models].<field>` key in the config file.

    Args:
        field: Key name under the `[models]` table (e.g., `'default'` or `'recent'`).
        model_spec: The model to save in `provider:model` format.
        config_path: Path to config file.

            Defaults to `~/.deepagents/config.toml`.

    Returns:
        True if save succeeded, False if it failed due to I/O errors.
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

        if "models" not in data:
            data["models"] = {}
        data["models"][field] = model_spec

        # Write to temp file then rename to prevent corruption if write is interrupted
        fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                tomli_w.dump(data, f)
            Path(tmp_path).replace(config_path)
        except BaseException:
            # Clean up temp file on any failure
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except (OSError, tomllib.TOMLDecodeError):
        logger.exception("Could not save %s model preference", field)
        return False
    else:
        # Invalidate config cache so the next load() picks up the change.
        global _default_config_cache  # noqa: PLW0603  # Module-level cache requires global statement
        _default_config_cache = None
        return True


def save_default_model(model_spec: str, config_path: Path | None = None) -> bool:
    """Update the default model in config file.

    Reads existing config (if any), updates `[models].default`, and writes
    back using proper TOML serialization.

    Args:
        model_spec: The model to set as default in `provider:model` format.
        config_path: Path to config file.

            Defaults to `~/.deepagents/config.toml`.

    Returns:
        True if save succeeded, False if it failed due to I/O errors.

    Note:
        This function does not preserve comments in the config file.
    """
    return _save_model_field("default", model_spec, config_path)


def clear_default_model(config_path: Path | None = None) -> bool:
    """Remove the default model from the config file.

    Deletes the `[models].default` key so that future launches fall back to
    `[models].recent` or environment auto-detection.

    Args:
        config_path: Path to config file.

            Defaults to `~/.deepagents/config.toml`.

    Returns:
        True if the key was removed (or was already absent), False on I/O error.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return True  # Nothing to clear

    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)

        models_section = data.get("models")
        if not isinstance(models_section, dict) or "default" not in models_section:
            return True  # Already absent

        del models_section["default"]

        fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                tomli_w.dump(data, f)
            Path(tmp_path).replace(config_path)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except (OSError, tomllib.TOMLDecodeError):
        logger.exception("Could not clear default model preference")
        return False
    else:
        global _default_config_cache  # noqa: PLW0603  # Module-level cache requires global statement
        _default_config_cache = None
        return True


def is_warning_suppressed(key: str, config_path: Path | None = None) -> bool:
    """Check if a warning key is suppressed in the config file.

    Reads the `[warnings].suppress` list from `config.toml` and checks
    whether `key` is present.

    Args:
        key: Warning identifier to check (e.g., `'ripgrep'`).
        config_path: Path to config file.

            Defaults to `~/.deepagents/config.toml`.

    Returns:
        `True` if the warning is suppressed, `False` otherwise (including
            when the file is missing, unreadable, or has no
            `[warnings]` section).
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    try:
        if not config_path.exists():
            return False
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        logger.debug(
            "Could not read config file %s for warning suppression check",
            config_path,
            exc_info=True,
        )
        return False

    suppress_list = data.get("warnings", {}).get("suppress", [])
    if not isinstance(suppress_list, list):
        logger.debug(
            "[warnings].suppress in %s should be a list, got %s",
            config_path,
            type(suppress_list).__name__,
        )
        return False
    return key in suppress_list


def suppress_warning(key: str, config_path: Path | None = None) -> bool:
    """Add a warning key to the suppression list in the config file.

    Reads existing config (if any), adds `key` to `[warnings].suppress`,
    and writes back using atomic temp-file rename. Deduplicates entries.

    Args:
        key: Warning identifier to suppress (e.g., `'ripgrep'`).
        config_path: Path to config file.

            Defaults to `~/.deepagents/config.toml`.

    Returns:
        `True` if save succeeded, `False` if it failed due to I/O errors.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if config_path.exists():
            with config_path.open("rb") as f:
                data = tomllib.load(f)
        else:
            data = {}

        if "warnings" not in data:
            data["warnings"] = {}
        suppress_list: list[str] = data["warnings"].get("suppress", [])
        if key not in suppress_list:
            suppress_list.append(key)
        data["warnings"]["suppress"] = suppress_list

        fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                tomli_w.dump(data, f)
            Path(tmp_path).replace(config_path)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except (OSError, tomllib.TOMLDecodeError):
        logger.exception("Could not save warning suppression for '%s'", key)
        return False
    return True


def save_recent_model(model_spec: str, config_path: Path | None = None) -> bool:
    """Update the recently used model in config file.

    Writes to `[models].recent` instead of `[models].default`, so that `/model`
    switches do not overwrite the user's intentional default.

    Args:
        model_spec: The model to save in `provider:model` format.
        config_path: Path to config file.

            Defaults to `~/.deepagents/config.toml`.

    Returns:
        True if save succeeded, False if it failed due to I/O errors.

    Note:
        This function does not preserve comments in the config file.
    """
    return _save_model_field("recent", model_spec, config_path)
