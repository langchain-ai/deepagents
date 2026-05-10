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
import threading
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, TypedDict, cast
from urllib.parse import urlparse

import tomli_w

from deepagents_cli import _env_vars, auth_store

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

_ENV_PREFIX = "DEEPAGENTS_CLI_"


def resolved_env_var_name(canonical: str) -> str:
    """Return whichever env var name actually carries the resolved value.

    Mirrors `resolve_env_var`'s precedence: when the prefixed variant is
    present in `os.environ` (even empty), it wins; otherwise the canonical
    name is returned. Useful for UI labels that need to reflect what the
    CLI is actually reading rather than the canonical name.

    Args:
        canonical: The canonical environment variable name.

    Returns:
        The resolving env var name (prefixed or canonical).
    """
    if not canonical.startswith(_ENV_PREFIX):
        prefixed = f"{_ENV_PREFIX}{canonical}"
        if prefixed in os.environ:
            return prefixed
    return canonical


def resolve_env_var(name: str) -> str | None:
    """Look up an env var with `DEEPAGENTS_CLI_` prefix override.

    Checks `DEEPAGENTS_CLI_{name}` first, then falls back to `{name}`.

    If the prefixed variable is *present* in the environment (even as an empty
    string), the canonical variable is never consulted. This lets users
    set `DEEPAGENTS_CLI_X=""` to shadow a canonically-set key -- the function
    will return `None` (since empty strings are normalized to `None`),
    effectively suppressing the canonical value.

    If `name` already carries the prefix, the double-prefixed lookup is skipped
    to avoid nonsensical `DEEPAGENTS_CLI_DEEPAGENTS_CLI_*` reads
    (e.g., when the name comes from a user's `config.toml`).

    Args:
        name: The canonical environment variable name (e.g.
            `ANTHROPIC_API_KEY`).

    Returns:
        The resolved value, or `None` when absent or empty.
    """
    if not name.startswith(_ENV_PREFIX):
        prefixed = f"{_ENV_PREFIX}{name}"
        if prefixed in os.environ:
            val = os.environ[prefixed]
            if not val and os.environ.get(name):
                logger.debug(
                    "%s is set but empty, blocking non-empty %s. "
                    "Unset %s to use the canonical variable.",
                    prefixed,
                    name,
                    prefixed,
                )
            if val:
                logger.debug("Resolved %s from %s", name, prefixed)
            return val or None
    return os.environ.get(name) or None


PROVIDERS_DOCS_URL = (
    "https://docs.langchain.com/oss/python/deepagents/cli/providers#provider-reference"
)
"""Public CLI docs page for configuring model providers.

Referenced by `UnknownProviderError` and the `/auth` manager so the same
URL is used everywhere a user is sent to read about provider setup.
"""


class ModelConfigError(Exception):
    """Raised when model configuration or creation fails."""


class UnknownProviderError(ModelConfigError):
    """Raised when neither the CLI nor `init_chat_model` can infer a provider.

    Carries the offending model spec as an attribute and exposes
    `PROVIDERS_DOCS_URL` as a class-level constant so callers can render
    a clickable link without string-scanning the formatted message. This
    mirrors how `MissingCredentialsError` exposes `provider` / `env_var`
    for targeted recovery hints.
    """

    docs_url: ClassVar[str] = PROVIDERS_DOCS_URL
    """Provider-reference docs URL. Class-level so callers don't pass it."""

    def __init__(self, *, model_spec: str) -> None:
        """Initialize the error.

        Args:
            model_spec: The bare model name the user supplied (e.g.
                `'mystery-model'`). When the input had a `provider:model`
                form, parsing succeeds and this exception does not fire.

        Raises:
            ValueError: If `model_spec` is empty.
        """
        if not model_spec:
            msg = "model_spec must be non-empty"
            raise ValueError(msg)
        message = (
            f"Unable to infer a model provider for {model_spec!r}. "
            f"Specify one explicitly (e.g. 'anthropic:{model_spec}') "
            f"or see the provider reference at {self.docs_url}."
        )
        super().__init__(message)
        self.model_spec = model_spec


class MissingCredentialsError(ModelConfigError):
    """Raised when a provider is selected but its API key env var is unset.

    Subclasses `ModelConfigError` so existing `except ModelConfigError` blocks
    keep working. Carries the `provider` name and the canonical `env_var` so
    callers can render targeted recovery hints (e.g., "set OPENAI_API_KEY" or
    "run `/model <other_provider>:<model>`") without string-matching on the
    formatted exception message and without re-deriving the env-var name.
    """

    def __init__(
        self, message: str, *, provider: str, env_var: str | None = None
    ) -> None:
        """Initialize the error.

        Args:
            message: Human-readable message describing the missing credential.
            provider: The provider whose credentials are missing
                (e.g., `'openai'`).
            env_var: The canonical env var name expected to hold the
                credential (e.g., `'OPENAI_API_KEY'`). `None` when the
                provider has no registered env-var mapping.
        """
        super().__init__(message)
        self.provider = provider
        self.env_var = env_var


class ProviderAuthState(StrEnum):
    """Credential readiness state for a model provider."""

    CONFIGURED = "configured"
    """An explicit credential source is configured and non-empty."""

    MISSING = "missing"
    """An explicit credential source is required but missing."""

    NOT_REQUIRED = "not_required"
    """This provider configuration does not require API-key credentials."""

    IMPLICIT = "implicit"
    """The provider supports ambient auth outside CLI env-var checks."""

    MANAGED = "managed"
    """A custom provider class is expected to manage auth itself."""

    UNKNOWN = "unknown"
    """The CLI cannot determine whether provider auth is ready."""


class ProviderAuthSource(StrEnum):
    """Origin of a `CONFIGURED` credential, used to discriminate display."""

    STORED = "stored"
    """Persisted via `/auth` in `~/.deepagents/.state/auth.json`."""

    ENV = "env"
    """Resolved from an environment variable."""


@dataclass(frozen=True)
class ProviderAuthStatus:
    """Credential readiness information for a provider.

    Args:
        state: Provider auth state.
        provider: Provider name.
        env_var: Env var name associated with the state, when applicable.
        source: For `CONFIGURED` states, where the credential value came
            from. `None` for non-configured states or when the source is
            not meaningful (e.g., implicit/managed auth).
        detail: Short user-facing context for selectors and logs.
    """

    state: ProviderAuthState
    provider: str
    env_var: str | None = None
    source: ProviderAuthSource | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        """Enforce the source-vs-state invariant.

        Raises:
            ValueError: If `source` is set but `state` is not `CONFIGURED`,
                or if `state` is `CONFIGURED` but no `source` is recorded.
        """
        is_configured = self.state is ProviderAuthState.CONFIGURED
        has_source = self.source is not None
        if is_configured != has_source:
            msg = (
                f"ProviderAuthStatus invariant violated: "
                f"state={self.state!r} requires "
                f"{'a source' if is_configured else 'source=None'}, "
                f"got source={self.source!r}"
            )
            raise ValueError(msg)

    @property
    def blocks_start(self) -> bool:
        """Whether this status should block model creation or switching."""
        return self.state is ProviderAuthState.MISSING

    def as_legacy_bool(self) -> bool | None:
        """Return the historic `has_provider_credentials` tri-state value."""
        if self.state is ProviderAuthState.MISSING:
            return False
        if self.state is ProviderAuthState.UNKNOWN:
            return None
        return True

    def missing_detail(self) -> str:
        """Return a user-facing reason for a missing-credential status."""
        if self.env_var:
            return f"{self.env_var} is not set or is empty"
        if self.detail:
            return self.detail
        return (
            f"provider '{self.provider}' is not recognized. "
            "Add it to ~/.deepagents/config.toml with an api_key_env field"
        )


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
        """Non-raising variant of `parse`.

        Args:
            spec: Model specification in `provider:model` format.

        Returns:
            Parsed `ModelSpec`, or `None` when *spec* is not valid.
        """
        try:
            return cls.parse(spec)
        except ValueError:
            return None

    def __str__(self) -> str:
        """Return the model spec as a string in `provider:model` format."""
        return f"{self.provider}:{self.model}"


class ModelProfileEntry(TypedDict):
    """Profile data for a model with override tracking."""

    profile: dict[str, Any]
    """Merged profile dict (upstream defaults + config.toml overrides).

    Keys vary by provider (e.g., `max_input_tokens`, `tool_calling`).
    """

    overridden_keys: frozenset[str]
    """Keys in `profile` whose values came from config.toml rather than the
    upstream provider package."""


class ProviderConfig(TypedDict, total=False):
    """Configuration for a model provider.

    The optional `class_path` field allows bypassing `init_chat_model` entirely
    and instantiating an arbitrary `BaseChatModel` subclass via importlib.

    !!! warning

        Setting `class_path` executes arbitrary Python code from the user's
        config file. This has the same trust model as `pyproject.toml` build
        scripts — the user controls their own machine.
    """

    enabled: bool
    """Whether this provider appears in the model switcher.

    Defaults to `True`. Set to `False` to hide a package-discovered provider
    and all its models from the `/model` selector. Useful when a LangChain
    provider package is installed as a transitive dependency but should not
    be user-visible.
    """

    models: list[str]
    """List of model identifiers available from this provider."""

    api_key_env: str
    """Name of the environment variable that holds the API key.

    This is the env var *name* (e.g., `"OPENAI_API_KEY"`), not the secret
    itself. The CLI resolves it at startup to verify credentials before model
    creation.
    """

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

    Do not set `api_key` here — the early credential check runs before
    `params` are read, so the CLI will reject the model before it sees the key.
    Use `api_key_env` to point at an environment variable instead.
    """

    profile: dict[str, Any]
    """Overrides merged into the model's runtime profile dict.

    Flat keys (e.g., `max_input_tokens = 4096`) are provider-wide defaults.
    Model-keyed sub-tables (e.g., `[profile."claude-sonnet-4-5"]`) override
    individual values for that model only; the merge is shallow.
    """


DEFAULT_CONFIG_DIR = Path.home() / ".deepagents"
"""Directory for user-level Deep Agents configuration (`~/.deepagents`)."""

DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"
"""Path to the user's model configuration file (`~/.deepagents/config.toml`)."""

DEFAULT_STATE_DIR = DEFAULT_CONFIG_DIR / ".state"
"""Directory for CLI-managed internal state (`~/.deepagents/.state`).

Holds files the CLI writes for its own bookkeeping — OAuth tokens, the
sessions database, version-check caches, input history. Kept separate from
top-level user-facing config and agent directories so listing/iterating
`~/.deepagents` doesn't conflate state with agents.
"""

PROVIDER_API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "baseten": "BASETEN_API_KEY",
    "cohere": "COHERE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "google_vertexai": "GOOGLE_CLOUD_PROJECT",
    "groq": "GROQ_API_KEY",
    "huggingface": "HUGGINGFACEHUB_API_TOKEN",
    "ibm": "WATSONX_APIKEY",
    "litellm": "LITELLM_API_KEY",
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

IMPLICIT_AUTH_PROVIDERS: frozenset[str] = frozenset({"google_vertexai"})
"""Providers that support ambient auth outside CLI env-var checks.

These providers can authenticate without the env var listed in
`PROVIDER_API_KEY_ENV`, so a missing env var should not be treated as a hard
credential failure. Used by `create_model` to skip the early credential check
and by `get_provider_auth_status` for user-facing auth labels.
"""

NO_AUTH_REQUIRED_PROVIDERS: frozenset[str] = frozenset({"ollama"})
"""Providers whose default local configuration does not require API keys."""

OPTIONAL_AUTH_ENV: dict[str, str] = {"ollama": "OLLAMA_API_KEY"}
"""Optional env vars that enable authenticated provider modes when present."""

PROVIDER_HOST_ENV: dict[str, str] = {"ollama": "OLLAMA_HOST"}
"""Provider-specific env vars that can point a local provider at a remote host."""

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
"""Default endpoint assumed when no `base_url` or `OLLAMA_HOST` is configured."""

OLLAMA_DISCOVERY_TIMEOUT_SECONDS = 1.0
"""Socket timeout for Ollama discovery probes.

Kept short so a dead daemon does not stall switcher loading. Discovery runs
off the UI loop in a worker thread and may call `/api/tags` and `/api/show`,
so this caps the worst-case wait visible to the user.
"""


# Module-level caches — cleared by `clear_caches()`.
_available_models_cache: dict[str, list[str]] | None = None
_builtin_providers_cache: dict[str, Any] | None = None
_default_config_cache: ModelConfig | None = None
_provider_profiles_cache: dict[str, dict[str, Any]] = {}
_provider_profiles_lock = threading.Lock()
_ollama_installed_models_cache: dict[str, list[str]] = {}
_ollama_model_profiles_cache: dict[tuple[str, str], dict[str, Any]] = {}
_profiles_cache: Mapping[str, ModelProfileEntry] | None = None
_profiles_override_cache: tuple[int, Mapping[str, ModelProfileEntry]] | None = None


def clear_caches() -> None:
    """Reset module-level caches so the next call recomputes from scratch.

    Intended for tests and for the `/reload` command.
    """
    global _available_models_cache, _builtin_providers_cache, _default_config_cache, _profiles_cache, _profiles_override_cache  # noqa: PLW0603, E501  # Module-level caches require global statement
    _available_models_cache = None
    _builtin_providers_cache = None
    _default_config_cache = None
    _provider_profiles_cache.clear()
    _ollama_installed_models_cache.clear()
    _ollama_model_profiles_cache.clear()
    _profiles_cache = None
    _profiles_override_cache = None
    invalidate_thread_config_cache()


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

    Results are cached by `module_path` so repeated calls (e.g., from both
    `get_available_models` and `get_model_profiles`) reuse the same dict.
    Use `clear_caches()` to reset.

    Locates the package on disk with `importlib.util.find_spec` and loads *only*
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
    with _provider_profiles_lock:
        cached = _provider_profiles_cache.get(module_path)
        if cached is not None:  # `is not None` so empty profile dicts are cached
            return cached

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
        profiles = getattr(module, "_PROFILES", {})
        _provider_profiles_cache[module_path] = profiles
        return profiles


def _profile_module_from_class_path(class_path: str) -> str | None:
    """Derive the profile module path from a `class_path` config value.

    Args:
        class_path: Fully-qualified class in `module.path:ClassName` format.

    Returns:
        Dotted module path like `langchain_baseten.data._profiles`, or None
            if `class_path` is malformed.
    """
    if ":" not in class_path:
        return None
    module_part, _ = class_path.split(":", 1)
    package_root = module_part.split(".", maxsplit=1)[0]
    if not package_root:
        return None
    return f"{package_root}.data._profiles"


def get_available_models() -> dict[str, list[str]]:
    """Get available models dynamically from installed LangChain provider packages.

    Imports model profiles from each provider package and extracts model names.

    Results are cached after the first call; use `clear_caches()` to reset.

    Returns:
        Dictionary mapping provider names to lists of model identifiers.
            Includes providers from the langchain registry, config-file
            providers with explicit model lists, and `class_path` providers
            whose packages expose a `_profiles` module.
    """
    global _available_models_cache  # noqa: PLW0603  # Module-level cache requires global statement
    if _available_models_cache is not None:
        return _available_models_cache

    available: dict[str, list[str]] = {}
    config = ModelConfig.load()

    # Try to load from langchain provider profile data.
    # Build the list dynamically from langchain's supported-provider registry
    # so new providers are picked up automatically when langchain adds them.
    provider_modules = _get_provider_profile_modules()
    registry_providers: set[str] = set()

    for provider, module_path in provider_modules:
        registry_providers.add(provider)
        # Skip providers explicitly disabled in config.
        if not config.is_provider_enabled(provider):
            logger.debug(
                "Provider '%s' is disabled in config; skipping registry discovery",
                provider,
            )
            continue
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
    for provider_name, provider_config in config.providers.items():
        # Respect enabled = false (hide provider entirely).
        if not config.is_provider_enabled(provider_name):
            logger.debug(
                "Provider '%s' is disabled in config; skipping",
                provider_name,
            )
            continue

        config_models = list(provider_config.get("models", []))

        # For class_path providers not in the built-in registry, auto-discover
        # models from the package's _profiles.py when no explicit models list.
        if (
            not config_models
            and provider_name not in registry_providers
            and provider_name not in available
        ):
            class_path = provider_config.get("class_path", "")
            profile_module = _profile_module_from_class_path(class_path)
            if profile_module:
                try:
                    profiles = _load_provider_profiles(profile_module)
                except ImportError:
                    logger.debug(
                        "Could not import profiles from %s for class_path "
                        "provider '%s' (package may not be installed)",
                        profile_module,
                        provider_name,
                    )
                except Exception:
                    logger.warning(
                        "Failed to load profiles from %s for class_path provider '%s'",
                        profile_module,
                        provider_name,
                        exc_info=True,
                    )
                else:
                    config_models = sorted(
                        name
                        for name, profile in profiles.items()
                        if profile.get("tool_calling", False)
                        and profile.get("text_inputs", True) is not False
                        and profile.get("text_outputs", True) is not False
                    )

        if provider_name not in available:
            if config_models:
                available[provider_name] = config_models
        else:
            # Append any config models not already discovered
            existing = set(available[provider_name])
            for model in config_models:
                if model not in existing:
                    available[provider_name].append(model)

    # `langchain-ollama` ships no profile data, so the steps above leave the
    # switcher empty unless the user hand-curates `models = [...]` in config.
    # Probe the daemon for installed models and merge them in,
    # preserving explicit config order (config wins) with discoveries appended.
    # Cached alongside the rest of `available`; refresh by
    # calling `clear_caches()` (e.g. via the `/reload` slash command).
    if (
        _ollama_discovery_enabled()
        and "ollama" in registry_providers
        and config.is_provider_enabled("ollama")
        and importlib.util.find_spec("langchain_ollama") is not None
    ):
        endpoint = _get_provider_endpoint("ollama", config)
        discovered = _get_ollama_installed_models(endpoint)
        if discovered:
            available["ollama"] = list(
                dict.fromkeys([*available.get("ollama", []), *discovered])
            )
        else:
            logger.debug(
                "Ollama discovery returned no models for %s; "
                "daemon may be down or have no pulls",
                endpoint or OLLAMA_DEFAULT_BASE_URL,
            )

    _available_models_cache = available
    return available


def _build_entry(
    base: dict[str, Any],
    overrides: dict[str, Any],
    cli_override: dict[str, Any] | None,
) -> ModelProfileEntry:
    """Build a profile entry by merging base, overrides, and CLI override.

    Args:
        base: Upstream profile dict (empty for config-only models).
        overrides: `config.toml` profile overrides.
        cli_override: Extra fields from `--profile-override`.

    Returns:
        Profile entry with merged data and override tracking.
    """
    merged = {**base, **overrides}
    overridden_keys = set(overrides)
    if cli_override:
        merged = {**merged, **cli_override}
        overridden_keys |= set(cli_override)
    return ModelProfileEntry(
        profile=merged,
        overridden_keys=frozenset(overridden_keys),
    )


def get_model_profiles(
    *,
    cli_override: dict[str, Any] | None = None,
) -> Mapping[str, ModelProfileEntry]:
    """Load upstream profiles merged with config.toml overrides.

    Keyed by `provider:model` spec string. Each entry contains the
    merged profile dict and the set of keys overridden by config.toml.

    Unlike `get_available_models()`, this includes all models from upstream
    profiles regardless of capability filters (tool calling, text I/O).

    Results are cached; use `clear_caches()` to reset. When `cli_override` is
    provided the result is stored in a single-slot cache keyed by
    `id(cli_override)`. This relies on the caller retaining the same dict
    object for the session (the CLI stores it once on the app instance);
    passing a different dict with the same contents will bypass the cache
    and overwrite the previous entry.

    Args:
        cli_override: Extra profile fields from `--profile-override`.

            When provided, these are merged on top of every profile entry
            (after upstream + config.toml) and their keys are added to
            `overridden_keys`.

    Returns:
        Read-only mapping of spec strings to profile entries.
    """
    global _profiles_cache, _profiles_override_cache  # noqa: PLW0603  # Module-level caches require global statement
    if cli_override is None and _profiles_cache is not None:
        return _profiles_cache
    if cli_override is not None and _profiles_override_cache is not None:
        cached_id, cached_result = _profiles_override_cache
        if cached_id == id(cli_override):
            return cached_result

    result: dict[str, ModelProfileEntry] = {}
    config = ModelConfig.load()

    # Collect upstream profiles from provider packages.
    seen_specs: set[str] = set()
    provider_modules = _get_provider_profile_modules()
    registry_providers: set[str] = set()
    for provider, module_path in provider_modules:
        registry_providers.add(provider)
        # Skip providers explicitly disabled in config.
        if not config.is_provider_enabled(provider):
            logger.debug(
                "Provider '%s' is disabled in config; skipping profiles",
                provider,
            )
            continue
        try:
            profiles = _load_provider_profiles(module_path)
        except ImportError:
            logger.debug(
                "Could not import profiles from %s for provider '%s'",
                module_path,
                provider,
            )
            continue
        except Exception:
            logger.warning(
                "Failed to load profiles from %s for provider '%s'",
                module_path,
                provider,
                exc_info=True,
            )
            continue

        for model_name, upstream_profile in profiles.items():
            spec = f"{provider}:{model_name}"
            seen_specs.add(spec)
            overrides = config.get_profile_overrides(provider, model_name=model_name)
            result[spec] = _build_entry(upstream_profile, overrides, cli_override)

    # Add config-only models and class_path provider profiles.
    for provider_name, provider_config in config.providers.items():
        if not config.is_provider_enabled(provider_name):
            logger.debug(
                "Provider '%s' is disabled in config; skipping profiles",
                provider_name,
            )
            continue
        # For class_path providers not in the built-in registry, load
        # upstream profiles from the package's _profiles.py.
        if provider_name not in registry_providers:
            class_path = provider_config.get("class_path", "")
            profile_module = _profile_module_from_class_path(class_path)
            if profile_module:
                try:
                    pkg_profiles = _load_provider_profiles(profile_module)
                except ImportError:
                    logger.debug(
                        "Could not import profiles from %s for class_path "
                        "provider '%s' (package may not be installed)",
                        profile_module,
                        provider_name,
                    )
                except Exception:
                    logger.warning(
                        "Failed to load profiles from %s for class_path provider '%s'",
                        profile_module,
                        provider_name,
                        exc_info=True,
                    )
                else:
                    for model_name, upstream_profile in pkg_profiles.items():
                        spec = f"{provider_name}:{model_name}"
                        seen_specs.add(spec)
                        overrides = config.get_profile_overrides(
                            provider_name, model_name=model_name
                        )
                        result[spec] = _build_entry(
                            upstream_profile, overrides, cli_override
                        )

        config_models = provider_config.get("models", [])
        for model_name in config_models:
            spec = f"{provider_name}:{model_name}"
            if spec not in seen_specs:
                overrides = config.get_profile_overrides(
                    provider_name, model_name=model_name
                )
                result[spec] = _build_entry({}, overrides, cli_override)

    # `langchain-ollama` does not ship static profile data. When discovery is
    # enabled, ask the daemon for model metadata so the selector can show
    # context length and capabilities for locally pulled models.
    if (
        _ollama_discovery_enabled()
        and "ollama" in registry_providers
        and config.is_provider_enabled("ollama")
        and importlib.util.find_spec("langchain_ollama") is not None
    ):
        endpoint = _get_provider_endpoint("ollama", config)
        discovered_model_names = _get_ollama_installed_models(endpoint)
        configured_model_names = [
            spec.removeprefix("ollama:")
            for spec in result
            if spec.startswith("ollama:")
        ]
        model_names = list(
            dict.fromkeys([*configured_model_names, *discovered_model_names])
        )
        if model_names:
            discovered_profiles = _fetch_ollama_installed_model_profiles(
                endpoint,
                model_names,
            )
            for model_name in model_names:
                profile = discovered_profiles.get(model_name, {})
                spec = f"ollama:{model_name}"
                existing = result.get(spec)
                base = dict(existing["profile"]) if existing is not None else {}
                base.update(profile)
                overrides = config.get_profile_overrides(
                    "ollama", model_name=model_name
                )
                result[spec] = _build_entry(base, overrides, cli_override)
                seen_specs.add(spec)

    frozen = MappingProxyType(result)
    if cli_override is None:
        _profiles_cache = frozen
    else:
        _profiles_override_cache = (id(cli_override), frozen)
    return frozen


_LOCAL_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "127.0.0.1",
        "::1",
        "0.0.0.0",  # noqa: S104  # hostname comparison, not socket binding
    }
)


def _is_local_endpoint(url: str | None) -> bool:
    """Return whether a provider endpoint points at the local machine."""
    if not url:
        return True
    if not isinstance(url, str):
        return False

    # Bare hostname literal (no scheme, no port) — short-circuit so IPv6
    # forms like `::1` don't get misparsed by urlparse.
    if url in _LOCAL_HOSTNAMES:
        return True

    candidate = url if "://" in url else f"http://{url}"
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return False
    return parsed.hostname in _LOCAL_HOSTNAMES


def _get_provider_endpoint(provider: str, config: ModelConfig) -> str | None:
    """Return a provider endpoint from config or provider-specific env vars."""
    base_url = config.get_base_url(provider)
    if base_url:
        return base_url

    host_env = PROVIDER_HOST_ENV.get(provider)
    if not host_env:
        return None
    return resolve_env_var(host_env)


_OLLAMA_DISCOVERY_FALSY: frozenset[str] = frozenset({"0", "false", "no", "off"})
"""Normalized values that disable Ollama discovery when set in `OLLAMA_DISCOVERY`."""

_OLLAMA_DISCOVERY_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})
"""Normalized values that enable Ollama discovery when set in `OLLAMA_DISCOVERY`."""


def _ollama_discovery_enabled() -> bool:
    """Return whether Ollama model/profile discovery may run.

    Defaults to enabled. Opt out via `_env_vars.OLLAMA_DISCOVERY` set to a
    falsy value (`0`, `false`, `no`, `off`); truthy values (`1`, `true`,
    `yes`, `on`) explicitly enable. Unrecognized values warn and fall through
    to the default because the user clearly tried to configure something.
    """
    raw = resolve_env_var(_env_vars.OLLAMA_DISCOVERY)
    if raw is None:
        return True
    normalized = raw.strip().lower()
    if normalized in _OLLAMA_DISCOVERY_FALSY:
        return False
    if normalized in _OLLAMA_DISCOVERY_TRUTHY:
        return True
    logger.warning(
        "Unrecognized value for %s: %r; expected one of %s. Defaulting to enabled.",
        _env_vars.OLLAMA_DISCOVERY,
        raw,
        sorted(_OLLAMA_DISCOVERY_FALSY | _OLLAMA_DISCOVERY_TRUTHY),
    )
    return True


def _get_ollama_installed_models(endpoint: str | None) -> list[str]:
    """Return cached Ollama model names for `endpoint`.

    Args:
        endpoint: Base URL of the Ollama daemon. When `None`, defaults to
            `OLLAMA_DEFAULT_BASE_URL`.

    Returns:
        Sorted list of model names reported by `/api/tags`.
    """
    key = (endpoint or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
    cached = _ollama_installed_models_cache.get(key)
    if cached is not None:
        return list(cached)
    models = _fetch_ollama_installed_models(endpoint)
    if models:
        _ollama_installed_models_cache[key] = models
    return list(models)


def _fetch_ollama_installed_models(
    endpoint: str | None,
    *,
    timeout: float = OLLAMA_DISCOVERY_TIMEOUT_SECONDS,
) -> list[str]:
    """Discover models installed in a local or hosted Ollama daemon.

    Issues a `GET {endpoint}/api/tags` and returns the sorted list of model
    names reported by the daemon. The probe is best-effort: any error
    (timeout, connection refused, malformed JSON) yields an empty list and is
    logged at debug level so the model switcher can fall back gracefully.

    When probing a local endpoint and `OLLAMA_API_KEY` (or the
    `DEEPAGENTS_CLI_`-prefixed variant) is set, its value is forwarded as a
    `Bearer` token. Discovery never forwards credentials to non-local endpoints.

    Args:
        endpoint: Base URL of the Ollama daemon. When `None`, defaults to
            `OLLAMA_DEFAULT_BASE_URL`. A trailing `/` is tolerated.
        timeout: Socket timeout in seconds.

    Returns:
        Sorted list of model names; empty when the daemon is unreachable or
            returns no models.
    """
    import json
    from urllib.error import URLError
    from urllib.request import Request, urlopen

    base = (endpoint or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
    if not base.startswith(("http://", "https://")):
        logger.warning(
            "Skipping Ollama discovery: %r has no http:// or https:// scheme. "
            "Set base_url or OLLAMA_HOST to e.g. http://localhost:11434.",
            base,
        )
        return []
    url = f"{base}/api/tags"

    headers = _ollama_discovery_headers(base, content_type=False)
    request = Request(url, headers=headers)  # noqa: S310  # scheme guarded above
    # Catch-all is intentional: discovery is best-effort and must never break
    # the model selector. The narrow tuple is fully subsumed by `Exception`
    # below; we keep it only to log expected transport failures at debug while
    # surfacing unexpected ones at warning so a real bug doesn't disappear.
    # Notably catches `pytest-socket`'s `SocketBlockedError`, which inherits
    # from `Exception` (not `OSError`) and would otherwise propagate during
    # unit tests run with `--disable-socket`. `KeyboardInterrupt` and
    # `SystemExit` derive from `BaseException` and bypass both branches.
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310  # scheme guarded above
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        logger.debug("Ollama model discovery failed for %s: %s", url, exc)
        return []
    except Exception as exc:  # noqa: BLE001  # see comment above
        logger.warning(
            "Ollama model discovery raised unexpected %s for %s: %s",
            type(exc).__name__,
            url,
            exc,
        )
        return []

    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        logger.debug(
            "Ollama discovery: %s returned unexpected payload shape (%s); "
            "endpoint may not be an Ollama daemon",
            url,
            type(payload).__name__,
        )
        return []

    names: list[str] = []
    for entry in payload["models"]:
        if isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    names.sort()
    return names


def _ollama_discovery_headers(endpoint: str, *, content_type: bool) -> dict[str, str]:
    """Build headers for Ollama discovery requests.

    Args:
        endpoint: Base URL for the discovery request.
        content_type: Whether to include a JSON `Content-Type` header.

    Returns:
        HTTP headers including optional bearer auth for local endpoints.
    """
    headers: dict[str, str] = {"Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = "application/json"
    optional_env = OPTIONAL_AUTH_ENV.get("ollama")
    if optional_env and _is_local_endpoint(endpoint):
        api_key = resolve_env_var(optional_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _coerce_positive_int(value: object) -> int | None:
    """Return `value` as a positive integer, or `None` when unavailable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0 and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        if parsed > 0:
            return parsed
    return None


def _profile_from_ollama_show_payload(payload: object) -> dict[str, Any]:
    """Extract LangChain-style profile fields from an Ollama `/api/show` payload.

    Args:
        payload: Decoded JSON response from `POST /api/show`.

    Returns:
        Profile fields understood by the model selector, such as
        `max_input_tokens` and `tool_calling`.
    """
    if not isinstance(payload, dict):
        return {}
    payload_dict = cast("dict[str, object]", payload)

    profile: dict[str, Any] = {}
    model_info = payload_dict.get("model_info")
    if isinstance(model_info, dict):
        context_lengths = [
            length
            for key, value in model_info.items()
            if isinstance(key, str)
            and (key == "context_length" or key.endswith(".context_length"))
            and (length := _coerce_positive_int(value)) is not None
        ]
        if context_lengths:
            profile["max_input_tokens"] = max(context_lengths)

    capabilities = payload_dict.get("capabilities")
    if isinstance(capabilities, list):
        capability_names = {item for item in capabilities if isinstance(item, str)}
        if "completion" in capability_names:
            profile["text_inputs"] = True
            profile["text_outputs"] = True
        if "tools" in capability_names:
            profile["tool_calling"] = True
        if "thinking" in capability_names:
            profile["reasoning_output"] = True

    if not profile and ("model_info" in payload_dict or "capabilities" in payload_dict):
        logger.debug(
            "Ollama profile discovery returned a payload with no recognized "
            "profile fields; top-level keys: %s",
            sorted(str(key) for key in payload_dict),
        )

    return profile


def _fetch_ollama_installed_model_profiles(
    endpoint: str | None,
    model_names: list[str],
    *,
    timeout: float = OLLAMA_DISCOVERY_TIMEOUT_SECONDS,
) -> dict[str, dict[str, Any]]:
    """Discover profile metadata for installed Ollama models.

    Issues `POST {endpoint}/api/show` for each model. The probe is best-effort:
    failures for one model are logged and do not stop profile discovery for the
    remaining models.

    Args:
        endpoint: Base URL of the Ollama daemon. When `None`, defaults to
            `OLLAMA_DEFAULT_BASE_URL`. A trailing `/` is tolerated.
        model_names: Model names to inspect.
        timeout: Socket timeout in seconds.

    Returns:
        Mapping of model name to extracted profile fields.
    """
    import json
    from urllib.error import URLError
    from urllib.request import Request, urlopen

    base = (endpoint or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
    if not base.startswith(("http://", "https://")):
        logger.warning(
            "Skipping Ollama profile discovery: %r has no http:// or https:// scheme. "
            "Set base_url or OLLAMA_HOST to e.g. http://localhost:11434.",
            base,
        )
        return {}

    url = f"{base}/api/show"
    profiles: dict[str, dict[str, Any]] = {}
    headers = _ollama_discovery_headers(base, content_type=True)

    for model_name in model_names:
        cache_key = (base, model_name)
        cached = _ollama_model_profiles_cache.get(cache_key)
        if cached is not None:
            profiles[model_name] = dict(cached)
            continue

        body = json.dumps({"model": model_name}).encode("utf-8")
        request = Request(  # noqa: S310  # scheme guarded above
            url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310  # scheme guarded above
                payload = json.loads(response.read().decode("utf-8"))
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            logger.debug(
                "Ollama profile discovery failed for %s via %s: %s",
                model_name,
                url,
                exc,
            )
            continue
        except Exception as exc:  # noqa: BLE001  # see _fetch_ollama_installed_models
            logger.warning(
                "Ollama profile discovery raised unexpected %s for %s via %s: %s",
                type(exc).__name__,
                model_name,
                url,
                exc,
            )
            continue

        profile = _profile_from_ollama_show_payload(payload)
        if profile:
            _ollama_model_profiles_cache[cache_key] = profile
            profiles[model_name] = profile

    return profiles


def _has_stored_credential(provider: str) -> bool:
    """Return whether `provider` has a credential persisted via `/auth`.

    A corrupt `auth.json` is swallowed (logged, treated as absent) so the
    model selector and other read-side callers can keep listing providers.
    The user-visible signal lives in `AuthManagerScreen` — opening `/auth`
    surfaces a corruption banner directly. Read-side resilience here means
    you can still pick a different provider while the file is broken.
    """
    try:
        return auth_store.get_stored_key(provider) is not None
    except RuntimeError:
        logger.warning(
            "Could not read stored credentials for provider %s; treating as absent",
            provider,
        )
        return False


def resolve_provider_credential(provider: str) -> str | None:
    """Resolve the credential value for `provider` from any configured source.

    Lookup order:

    1. Stored API key in `~/.deepagents/.state/auth.json` (added via `/auth`).
    2. Canonical env var via `resolve_env_var()` (which honors the
        `DEEPAGENTS_CLI_` prefix and dotenv files).

    A user who has *both* a stored key and an env var set gets the stored
    key — entering one in the TUI is the more deliberate, more recent
    action, so "I just typed this in" beats whatever the shell exported.

    Args:
        provider: Provider name (e.g., `"anthropic"`).

    Returns:
        The credential value, or `None` when no source has one or the
        provider has no env-var mapping at all.
    """
    try:
        stored = auth_store.get_stored_key(provider)
    except RuntimeError:
        logger.warning(
            "Could not read stored credentials for provider %s; falling back to env",
            provider,
        )
        stored = None
    if stored:
        return stored
    env_var = get_credential_env_var(provider)
    if env_var:
        return resolve_env_var(env_var)
    return None


def _resolve_configured(provider: str, env_var: str) -> ProviderAuthStatus | None:
    """Return a `CONFIGURED` status if a stored or env credential is set.

    Stored credentials beat env vars (matches `resolve_provider_credential`).

    Args:
        provider: Provider name (e.g., `"anthropic"`).
        env_var: Canonical env var name to check when no stored credential
            exists. Recorded on the returned status either way.

    Returns:
        A `CONFIGURED` status, or `None` when neither source is set.
    """
    if _has_stored_credential(provider):
        return ProviderAuthStatus(
            state=ProviderAuthState.CONFIGURED,
            provider=provider,
            env_var=env_var,
            source=ProviderAuthSource.STORED,
            detail="stored credential",
        )
    if resolve_env_var(env_var):
        return ProviderAuthStatus(
            state=ProviderAuthState.CONFIGURED,
            provider=provider,
            env_var=env_var,
            source=ProviderAuthSource.ENV,
            detail="credentials set",
        )
    return None


def get_provider_auth_status(provider: str) -> ProviderAuthStatus:
    """Return credential readiness details for a provider.

    Combines config, well-known provider metadata, optional provider auth,
    and implicit-auth provider metadata before attempting model creation:

    1. **Config-file providers** (`config.toml`
        `[models.providers.<name>]`):
        - If the section declares `api_key_env`, that env var is checked
            via `resolve_env_var()` (which honors `DEEPAGENTS_CLI_` prefixes).
        - If the section has `class_path` but no `api_key_env`, the provider is
            assumed to manage its own auth (e.g., custom headers, JWT, mTLS).
        - If neither `api_key_env` nor `class_path` is set, falls through
            to provider-specific defaults.
    2. **Hardcoded registry** (`PROVIDER_API_KEY_ENV`): a module-level dict
        mapping well-known provider names to their canonical env var
        (e.g., `"anthropic"` → `"ANTHROPIC_API_KEY"`). The env var is checked
        via `resolve_env_var()`.
    3. **Implicit auth providers** (e.g., Vertex AI ADC): a missing env var is
        not treated as missing credentials.
    4. **Optional auth env vars** (`OPTIONAL_AUTH_ENV`): when present, mark
        the provider as configured for hosted/cloud use.
    5. **No-auth-required providers** (`NO_AUTH_REQUIRED_PROVIDERS`): default
        local endpoints report `NOT_REQUIRED`; non-local endpoints fall back
        to `UNKNOWN` so the SDK can decide.
    6. **Unknown providers** not present in any source defer auth failures to
        the provider SDK.

    Use `has_provider_credentials()` when compatibility with the historic
    `True`/`False`/`None` contract is required.

    Args:
        provider: Provider name (e.g., `"anthropic"`, `"openai"`).

    Returns:
        Provider auth status for selectors, startup checks, and compatibility
            wrappers.
    """
    # Config-file providers take priority when api_key_env is specified.
    config = ModelConfig.load()
    provider_config = config.providers.get(provider)
    if provider_config:
        env_var = provider_config.get("api_key_env")
        if env_var:
            configured = _resolve_configured(provider, env_var)
            if configured:
                return configured
            return ProviderAuthStatus(
                state=ProviderAuthState.MISSING,
                provider=provider,
                env_var=env_var,
                detail=f"{env_var} is not set or is empty",
            )
        # class_path providers that omit api_key_env manage their own auth
        # (e.g., custom headers, JWT, mTLS).
        if provider_config.get("class_path"):
            return ProviderAuthStatus(
                state=ProviderAuthState.MANAGED,
                provider=provider,
                detail="custom auth",
            )
        # No api_key_env in config — fall through to provider-specific and
        # hardcoded maps.

    # Fall back to hardcoded well-known providers.
    env_var = PROVIDER_API_KEY_ENV.get(provider)
    if env_var:
        configured = _resolve_configured(provider, env_var)
        if configured:
            return configured
        if provider in IMPLICIT_AUTH_PROVIDERS:
            return ProviderAuthStatus(
                state=ProviderAuthState.IMPLICIT,
                provider=provider,
                env_var=env_var,
                detail="implicit auth",
            )
        return ProviderAuthStatus(
            state=ProviderAuthState.MISSING,
            provider=provider,
            env_var=env_var,
            detail=f"{env_var} is not set or is empty",
        )

    if provider in IMPLICIT_AUTH_PROVIDERS:
        return ProviderAuthStatus(
            state=ProviderAuthState.IMPLICIT,
            provider=provider,
            detail="implicit auth",
        )

    optional_env = OPTIONAL_AUTH_ENV.get(provider)
    if optional_env:
        configured = _resolve_configured(provider, optional_env)
        if configured:
            return configured

    if provider in NO_AUTH_REQUIRED_PROVIDERS:
        endpoint = _get_provider_endpoint(provider, config)
        if _is_local_endpoint(endpoint):
            return ProviderAuthStatus(
                state=ProviderAuthState.NOT_REQUIRED,
                provider=provider,
                detail="local provider",
            )
        # Remote endpoint may or may not require auth (private network vs.
        # hosted). Don't block; surface the optional env var as a hint.
        detail = (
            f"remote endpoint; set {optional_env} if auth is required"
            if optional_env
            else "remote endpoint"
        )
        return ProviderAuthStatus(
            state=ProviderAuthState.UNKNOWN,
            provider=provider,
            env_var=optional_env,
            detail=detail,
        )

    # Provider not found in config or hardcoded map — credential status is
    # unknown. The provider itself will report auth failures at
    # model-creation time.
    logger.debug(
        "No credential information for provider '%s'; deferring auth to provider",
        provider,
    )
    return ProviderAuthStatus(
        state=ProviderAuthState.UNKNOWN,
        provider=provider,
        detail="credentials unknown",
    )


def has_provider_credentials(provider: str) -> bool | None:
    """Check if credentials are available for a provider.

    This compatibility wrapper preserves the historic tri-state contract while
    `get_provider_auth_status()` carries the richer user-facing distinctions:
    configured credentials, missing credentials, no-auth local providers,
    implicit auth, custom provider-managed auth, and unknown providers.

    Args:
        provider: Provider name (e.g., `"anthropic"`, `"openai"`).

    Returns:
        `True` if auth is configured, implicit, provider-managed, or not
            required.
        `False` if a required env var is known but not set.
        `None` if credential status cannot be determined.
    """
    return get_provider_auth_status(provider).as_legacy_bool()


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


def apply_stored_credentials(provider: str) -> bool:
    """Export this provider's stored API key into `os.environ` for SDK use.

    LangChain's chat-model factories read credentials from process env vars,
    so a stored key only takes effect once it's copied onto the env var name
    registered for that provider. This is a no-op when the provider has no
    env-var mapping (custom auth) or no stored credential.

    The env var is overwritten whether or not it was already set, matching
    the precedence rule documented on `resolve_provider_credential`: a
    credential the user typed in `/auth` is the most recent deliberate
    action and should take effect.

    Args:
        provider: Provider name.

    Returns:
        `True` if a stored key was applied, `False` otherwise.
    """
    env_var = get_credential_env_var(provider)
    if not env_var:
        return False
    try:
        stored = auth_store.get_stored_key(provider)
    except RuntimeError:
        logger.warning("Could not read stored credentials for provider %s", provider)
        return False
    if not stored:
        return False
    if os.environ.get(env_var) == stored:
        return True
    os.environ[env_var] = stored
    return True


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

        # Validate enabled field type and class_path format / params references
        for name, provider in self.providers.items():
            enabled = provider.get("enabled")
            if enabled is not None and not isinstance(enabled, bool):
                logger.warning(
                    "Provider '%s' has non-boolean 'enabled' value %r "
                    "(expected true/false). Provider will remain visible.",
                    name,
                    enabled,
                )

            class_path = provider.get("class_path")
            if class_path and ":" not in class_path:
                logger.warning(
                    "Provider '%s' has invalid class_path '%s': "
                    "must be in module.path:ClassName format "
                    "(e.g., 'my_package.models:MyChatModel')",
                    name,
                    class_path,
                )

            models = set(provider.get("models", []))

            params = provider.get("params", {})
            for key, value in params.items():
                if isinstance(value, dict) and key not in models:
                    logger.warning(
                        "Provider '%s' has params for '%s' "
                        "which is not in its models list",
                        name,
                        key,
                    )

    def is_provider_enabled(self, provider_name: str) -> bool:
        """Check whether a provider should appear in the model switcher.

        A provider is disabled when its config explicitly sets
        `enabled = false`. Providers not present in the config file are
        always considered enabled.

        Args:
            provider_name: The provider to check.

        Returns:
            `False` if the provider is explicitly disabled, `True` otherwise.
        """
        provider = self.providers.get(provider_name)
        if not provider:
            return True
        return provider.get("enabled") is not False

    def get_all_models(self) -> list[tuple[str, str]]:
        """Get all models as `(model_name, provider_name)` tuples.

        Returns raw config data — does not filter by `is_provider_enabled`.
        For the filtered set shown in the model switcher, use
        `get_available_models()`.

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

        Returns raw config data — does not filter by `is_provider_enabled`.

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
        return bool(resolve_env_var(env_var))

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
        result = {k: v for k, v in params.items() if not isinstance(v, dict)}
        if model_name:
            overrides = params.get(model_name)
            if isinstance(overrides, dict):
                result.update(overrides)
        return result

    def get_profile_overrides(
        self, provider_name: str, *, model_name: str | None = None
    ) -> dict[str, Any]:
        """Get profile overrides for a provider.

        Reads the `profile` table from the provider config. Flat keys are
        provider-wide defaults; model-keyed sub-tables are per-model overrides
        that shallow-merge on top (model wins on conflict).

        Args:
            provider_name: The provider to look up.
            model_name: Optional model name for per-model overrides.

        Returns:
            Dictionary of profile overrides (empty if none configured).
        """
        provider = self.providers.get(provider_name)
        if not provider:
            return {}
        profile = provider.get("profile", {})
        result = {k: v for k, v in profile.items() if not isinstance(v, dict)}
        if model_name:
            overrides = profile.get(model_name)
            if isinstance(overrides, dict):
                result.update(overrides)
        return result


def _save_toml_field(
    section: str,
    field: str,
    value: str,
    config_path: Path | None = None,
) -> bool:
    """Read-modify-write a `[section].<field>` key in the config file.

    Args:
        section: TOML table name (e.g., `'models'`, `'agents'`).
        field: Key within the table (e.g., `'default'`, `'recent'`).
        value: String value to persist.
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

        if section not in data:
            data[section] = {}
        data[section][field] = value

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
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError):
        # `TypeError` covers `tomli_w.dump` rejecting a non-serializable
        # payload; `ValueError` covers things like `os.fdopen` on a
        # closed fd. Folding them in keeps the `bool` contract intact for
        # the UI branches that toggle on the return value.
        logger.exception("Could not save %s.%s preference", section, field)
        return False
    else:
        # Invalidate config cache so the next load() picks up the change.
        global _default_config_cache  # noqa: PLW0603  # Module-level cache requires global statement
        _default_config_cache = None
        return True


def _save_model_field(
    field: str, model_spec: str, config_path: Path | None = None
) -> bool:
    """Read-modify-write a `[models].<field>` key in the config file.

    Thin wrapper around `_save_toml_field` for the `[models]` section.

    Args:
        field: Key name under the `[models]` table (e.g., `'default'` or `'recent'`).
        model_spec: The model to save in `provider:model` format.
        config_path: Path to config file.

            Defaults to `~/.deepagents/config.toml`.

    Returns:
        True if save succeeded, False if it failed due to I/O errors.
    """
    return _save_toml_field("models", field, model_spec, config_path)


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
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError):
        # See `_save_toml_field` for why `TypeError` / `ValueError` are
        # folded into the bool return contract.
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
        suppress_list = data["warnings"].get("suppress", [])
        if not isinstance(suppress_list, list):
            logger.debug(
                "[warnings].suppress in %s should be a list, got %s",
                config_path,
                type(suppress_list).__name__,
            )
            suppress_list = []
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


def unsuppress_warning(key: str, config_path: Path | None = None) -> bool:
    """Remove a warning key from the suppression list in the config file.

    Reads existing config (if any), removes `key` from `[warnings].suppress`,
    and writes back using atomic temp-file rename. No-op if the key is not
    present or the file does not exist.

    Args:
        key: Warning identifier to unsuppress (e.g., `'ripgrep'`).
        config_path: Path to config file.

            Defaults to `~/.deepagents/config.toml`.

    Returns:
        `True` if save succeeded, `False` if it failed due to I/O errors.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    try:
        if not config_path.exists():
            return True  # nothing to remove

        with config_path.open("rb") as f:
            data = tomllib.load(f)

        suppress_list = data.get("warnings", {}).get("suppress", [])
        if not isinstance(suppress_list, list):
            logger.debug(
                "[warnings].suppress in %s should be a list, got %s",
                config_path,
                type(suppress_list).__name__,
            )
            return True  # treat as nothing to remove
        if key not in suppress_list:
            return True  # already unsuppressed

        suppress_list.remove(key)
        data.setdefault("warnings", {})["suppress"] = suppress_list

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
        logger.exception("Could not remove warning suppression for '%s'", key)
        return False
    return True


THREAD_COLUMN_DEFAULTS: dict[str, bool] = {
    "thread_id": False,
    "messages": True,
    "created_at": True,
    "updated_at": True,
    "git_branch": False,
    "cwd": False,
    "initial_prompt": True,
    "agent_name": False,
}
"""Default visibility for thread selector columns."""


class ThreadConfig(NamedTuple):
    """Coalesced thread-selector configuration read from a single TOML parse."""

    columns: dict[str, bool]
    """Column visibility settings."""

    relative_time: bool
    """Whether to display timestamps as relative time."""

    sort_order: str
    """`'updated_at'` or `'created_at'`."""


_thread_config_cache: ThreadConfig | None = None


def load_thread_config(config_path: Path | None = None) -> ThreadConfig:
    """Load all thread-selector settings from one config file read.

    Returns a cached result when reading the default config path. The
    prewarm worker calls this at startup so subsequent opens of the
    `/threads` modal avoid disk I/O entirely.

    Args:
        config_path: Path to config file.

    Returns:
        Coalesced thread configuration.
    """
    global _thread_config_cache  # noqa: PLW0603  # Module-level cache requires global statement

    if config_path is None:
        if _thread_config_cache is not None:
            return _thread_config_cache
        config_path = DEFAULT_CONFIG_PATH
    use_default = config_path == DEFAULT_CONFIG_PATH

    columns = dict(THREAD_COLUMN_DEFAULTS)
    relative_time = True
    sort_order = "updated_at"

    try:
        if not config_path.exists():
            result = ThreadConfig(columns, relative_time, sort_order)
            if use_default:
                _thread_config_cache = result
            return result
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        threads_section = data.get("threads", {})

        # columns
        raw_columns = threads_section.get("columns", {})
        if isinstance(raw_columns, dict):
            for key in columns:
                if key in raw_columns and isinstance(raw_columns[key], bool):
                    columns[key] = raw_columns[key]

        # relative_time
        rt_value = threads_section.get("relative_time")
        if isinstance(rt_value, bool):
            relative_time = rt_value

        # sort_order
        so_value = threads_section.get("sort_order")
        if so_value in {"updated_at", "created_at"}:
            sort_order = so_value
    except (OSError, tomllib.TOMLDecodeError):
        logger.warning("Could not read thread config; using defaults", exc_info=True)
        # Do not cache on error — allow retry on next call in case the
        # file is fixed or permissions are restored.
        return ThreadConfig(columns, relative_time, sort_order)

    result = ThreadConfig(columns, relative_time, sort_order)
    if use_default:
        _thread_config_cache = result
    return result


def invalidate_thread_config_cache() -> None:
    """Clear the cached `ThreadConfig` so the next load re-reads disk."""
    global _thread_config_cache  # noqa: PLW0603  # Module-level cache requires global statement
    _thread_config_cache = None


def load_thread_columns(config_path: Path | None = None) -> dict[str, bool]:
    """Load thread column visibility from config file.

    Args:
        config_path: Path to config file.

    Returns:
        Dict mapping column names to visibility booleans.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    result = dict(THREAD_COLUMN_DEFAULTS)
    try:
        if not config_path.exists():
            return result
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        columns = data.get("threads", {}).get("columns", {})
        if isinstance(columns, dict):
            for key in result:
                if key in columns and isinstance(columns[key], bool):
                    result[key] = columns[key]
    except (OSError, tomllib.TOMLDecodeError):
        logger.debug("Could not read thread column config", exc_info=True)
    return result


def save_thread_columns(
    columns: dict[str, bool], config_path: Path | None = None
) -> bool:
    """Save thread column visibility to config file.

    Args:
        columns: Dict mapping column names to visibility booleans.
        config_path: Path to config file.

    Returns:
        True if save succeeded, False on I/O error.
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

        if "threads" not in data:
            data["threads"] = {}
        data["threads"]["columns"] = columns

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
        logger.exception("Could not save thread column preferences")
        return False
    invalidate_thread_config_cache()
    return True


def load_thread_relative_time(config_path: Path | None = None) -> bool:
    """Load the relative-time display preference for thread timestamps.

    Args:
        config_path: Path to config file.

    Returns:
        True if timestamps should display as relative time.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    try:
        if not config_path.exists():
            return True
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        value = data.get("threads", {}).get("relative_time")
        if isinstance(value, bool):
            return value
    except (OSError, tomllib.TOMLDecodeError):
        logger.debug("Could not read thread relative_time config", exc_info=True)
    return True


def save_thread_relative_time(enabled: bool, config_path: Path | None = None) -> bool:
    """Save the relative-time display preference for thread timestamps.

    Args:
        enabled: Whether to display relative timestamps.
        config_path: Path to config file.

    Returns:
        True if save succeeded, False on I/O error.
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
        if "threads" not in data:
            data["threads"] = {}
        data["threads"]["relative_time"] = enabled
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
        logger.exception("Could not save thread relative_time preference")
        return False
    invalidate_thread_config_cache()
    return True


def load_thread_sort_order(config_path: Path | None = None) -> str:
    """Load the sort order preference for the thread selector.

    Args:
        config_path: Path to config file.

    Returns:
        `"updated_at"` or `"created_at"`.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    try:
        if not config_path.exists():
            return "updated_at"
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        value = data.get("threads", {}).get("sort_order")
        if value in {"updated_at", "created_at"}:
            return value
    except (OSError, tomllib.TOMLDecodeError):
        logger.debug("Could not read thread sort_order config", exc_info=True)
    return "updated_at"


def save_thread_sort_order(sort_order: str, config_path: Path | None = None) -> bool:
    """Save the sort order preference for the thread selector.

    Args:
        sort_order: `"updated_at"` or `"created_at"`.
        config_path: Path to config file.

    Returns:
        True if save succeeded, False on I/O error.

    Raises:
        ValueError: If `sort_order` is not a recognised value.
    """
    if sort_order not in {"updated_at", "created_at"}:
        msg = (
            f"Invalid sort_order {sort_order!r}; expected 'updated_at' or 'created_at'"
        )
        raise ValueError(msg)
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if config_path.exists():
            with config_path.open("rb") as f:
                data = tomllib.load(f)
        else:
            data = {}
        if "threads" not in data:
            data["threads"] = {}
        data["threads"]["sort_order"] = sort_order
        fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                tomli_w.dump(data, f)
            Path(tmp_path).replace(config_path)
        except Exception:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except (OSError, tomllib.TOMLDecodeError):
        logger.exception("Could not save thread sort_order preference")
        return False
    invalidate_thread_config_cache()
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


def save_recent_agent(agent_name: str, config_path: Path | None = None) -> bool:
    """Update the recently used agent in config file.

    Writes to `[agents].recent` so a later bare `deepagents` launch (no
    `-a`) can bring the user back to their last agent instead of the
    default.

    Args:
        agent_name: The agent directory name (e.g., `'coder'`).
        config_path: Path to config file.

            Defaults to `~/.deepagents/config.toml`.

    Returns:
        True if save succeeded, False if it failed due to I/O errors.
    """
    return _save_toml_field("agents", "recent", agent_name, config_path)


def load_recent_agent(config_path: Path | None = None) -> str | None:
    """Read `[agents].recent` from the config file.

    Args:
        config_path: Path to config file.

            Defaults to `~/.deepagents/config.toml`.

    Returns:
        The saved agent name, or `None` if the file or key is missing or
        the file is unreadable.
    """
    return _load_agents_field("recent", config_path)


def save_default_agent(agent_name: str, config_path: Path | None = None) -> bool:
    """Update the default agent in config file.

    Writes to `[agents].default`. This is the user's intentional sticky
    default — set via `Ctrl+S` in the `/agents` picker — and takes
    precedence over `[agents].recent` on bare-launch resolution.

    Args:
        agent_name: The agent directory name (e.g., `'coder'`).
        config_path: Path to config file.

            Defaults to `~/.deepagents/config.toml`.

    Returns:
        True if save succeeded, False if it failed due to I/O errors.
    """
    return _save_toml_field("agents", "default", agent_name, config_path)


def clear_default_agent(config_path: Path | None = None) -> bool:
    """Remove the default agent from the config file.

    Deletes the `[agents].default` key so that future launches fall back
    to `[agents].recent` and then `DEFAULT_AGENT_NAME`.

    Args:
        config_path: Path to config file.

            Defaults to `~/.deepagents/config.toml`.

    Returns:
        True if the key was removed (or was already absent), False on I/O error.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return True

    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)

        agents_section = data.get("agents")
        if not isinstance(agents_section, dict) or "default" not in agents_section:
            return True

        del agents_section["default"]

        fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                tomli_w.dump(data, f)
            Path(tmp_path).replace(config_path)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError):
        # See `_save_toml_field` for why `TypeError` / `ValueError` are
        # folded into the bool return contract.
        logger.exception("Could not clear default agent preference")
        return False
    else:
        global _default_config_cache  # noqa: PLW0603  # Module-level cache requires global statement
        _default_config_cache = None
        return True


def load_default_agent(config_path: Path | None = None) -> str | None:
    """Read `[agents].default` from the config file.

    Args:
        config_path: Path to config file.

            Defaults to `~/.deepagents/config.toml`.

    Returns:
        The saved agent name, or `None` if the file or key is missing or
        the file is unreadable.
    """
    return _load_agents_field("default", config_path)


def _load_agents_field(field: str, config_path: Path | None = None) -> str | None:
    """Read `[agents].<field>` from the config file.

    Args:
        field: Key under the `[agents]` table (e.g., `'recent'`, `'default'`).
        config_path: Path to config file.

            Defaults to `~/.deepagents/config.toml`.

    Returns:
        The trimmed string value, or `None` if the file, section, or key
        is missing or the file is unreadable.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return None
    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        logger.warning("Could not read agents.%s from config", field, exc_info=True)
        return None
    agents_section = data.get("agents", {})
    value = agents_section.get(field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
