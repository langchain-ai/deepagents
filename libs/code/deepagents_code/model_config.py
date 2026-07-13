"""Model configuration management.

Handles loading and saving model configuration from TOML files, providing a
structured way to define available models and providers.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
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

from deepagents_code import _env_vars, auth_store

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)

_ENV_PREFIX = "DEEPAGENTS_CODE_"


def resolved_env_var_name(canonical: str) -> str:
    """Return whichever env var name actually carries the resolved value.

    Mirrors `resolve_env_var`'s precedence: when the prefixed variant is
    present in `os.environ` (even empty), it wins; otherwise the canonical
    name is returned. Useful for UI labels that need to reflect what the
    app is actually reading rather than the canonical name.

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
    """Look up an env var with `DEEPAGENTS_CODE_` prefix override.

    Checks `DEEPAGENTS_CODE_{name}` first, then falls back to `{name}`.

    If the prefixed variable is *present* in the environment (even as an empty
    string), the canonical variable is never consulted. This lets users
    set `DEEPAGENTS_CODE_X=""` to shadow a canonically-set key -- the function
    will return `None` (since empty strings are normalized to `None`),
    effectively suppressing the canonical value.

    If `name` already carries the prefix, the double-prefixed lookup is skipped
    to avoid nonsensical `DEEPAGENTS_CODE_DEEPAGENTS_CODE_*` reads
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
    "https://docs.langchain.com/oss/python/deepagents/code/providers#provider-reference"
)
"""Public docs page for configuring model providers.

Referenced by `UnknownProviderError` and the `/auth` manager so the same
URL is used everywhere a user is sent to read about provider setup.
"""


class ModelConfigError(Exception):
    """Raised when model configuration or creation fails."""


class NoCredentialsConfiguredError(ModelConfigError):
    """Raised when no credentials are configured for any default-resolvable provider.

    Distinct from `MissingCredentialsError` (which targets a specific provider
    the user has selected): this fires from `_get_default_model_spec()` when
    auto-detection finds no usable credentials at all. Callers (the deferred-
    start path in the TUI and CLI) `isinstance`-check this type to recover by
    launching the TUI with model creation deferred, rather than string-matching
    the formatted message.
    """


class UnknownProviderError(ModelConfigError):
    """Raised when neither the app nor `init_chat_model` can infer a provider.

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


class MissingProviderPackageError(ModelConfigError):
    """Raised when a provider is selected but its LangChain package is not installed.

    Subclasses `ModelConfigError` so existing `except ModelConfigError` blocks
    keep working. Carries the `provider` name and the `package` to install so
    callers can render targeted recovery hints (e.g., suggest `/install fireworks`
    or the `/model` slash command) without string-matching on the formatted
    exception message.
    """

    def __init__(self, message: str, *, provider: str, package: str) -> None:
        """Initialize the error.

        Args:
            message: Human-readable message describing the missing package.
            provider: The provider whose package is missing (e.g., `'fireworks'`).
            package: The pip-installable package name (e.g.,
                `'langchain-fireworks'`).
        """
        super().__init__(message)
        self.provider = provider
        self.package = package


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
    """The app cannot determine whether provider auth is ready."""


class ProviderAuthSource(StrEnum):
    """Origin of a `CONFIGURED` credential, used to discriminate display."""

    STORED = "stored"
    """Persisted in a local credential store under `~/.deepagents/.state`.

    Usually the `/auth` API-key map (`auth.json`), but also covers the
    file-backed ChatGPT OAuth token used by the codex provider
    (`chatgpt-auth.json`).
    """

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
    """The model identifier (e.g., `'claude-sonnet-4-5'`, `'gpt-5.5'`)."""

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
    itself. The app resolves it at startup to verify credentials before model
    creation.
    """

    display_name: str
    """Human-readable provider name shown in auth UI.

    Useful for arbitrary providers whose config key is optimized for machine use
    (e.g., `my_gateway`) but whose UI label should include spaces or brand
    capitalization.
    """

    short_name: str
    """Compact brand label for space-constrained UI (e.g. the `/model` Recent
    tag), where the full `display_name` — which may carry a parenthetical
    qualifier like `"OpenAI Codex (ChatGPT login)"` — is too long. Optional;
    when unset, callers fall back to `display_name`.
    """

    api_key_url: str
    """Provider page where users can create or manage API keys.

    Used by `/auth` as an acquisition link before the API-key input. The value is
    a URL, not a credential. Must use an `http` or `https` scheme to render as a
    clickable link; values with other schemes are ignored with a warning.
    """

    base_url: str
    """Custom base URL."""

    base_url_env: str
    """Name of the environment variable that holds this provider's base URL.

    Parallel to `api_key_env`: lets a provider that is not one of the built-in
    `PROVIDER_BASE_URL_ENV` entries participate in endpoint resolution and in
    the key/endpoint pairing applied by `apply_stored_credentials` (so a stored
    `/auth` override clears an inherited gateway URL). The static `base_url`
    field still wins over this when both are set.
    """

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
    `params` are read, so the app will reject the model before it sees the key.
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
"""Directory for app-managed internal state (`~/.deepagents/.state`).

Holds files the app writes for its own bookkeeping — OAuth tokens, the
sessions database, version-check caches, input history. Kept separate from
top-level user-facing config and agent directories so listing/iterating
`~/.deepagents` doesn't conflate state with agents.
"""

RECENT_MODELS_FILENAME = "recent_models.json"
"""Filename under `DEFAULT_STATE_DIR` for the MRU list shown in `/model`."""

RECENT_MODELS_LIMIT = 5
"""Maximum number of `provider:model` specs retained in the recent list.

Sized to fit comfortably above the provider-grouped list in `/model` without
pushing the rest of the catalog off-screen on a typical terminal.
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

LANGSMITH_SERVICE = "langsmith"
"""Service name for LangSmith tracing in `SERVICE_API_KEY_ENV`.

Storing a key for this service via `/auth` also enables tracing at startup
(see `config._apply_stored_langsmith_tracing`) and can carry a custom project
name, so it gets special handling beyond a plain key copy.
"""

TAVILY_SERVICE = "tavily"
"""Service name for Tavily web search in `SERVICE_API_KEY_ENV`.

Storing a key for this service via `/auth` gates the spawn-time `web_search`
tool (see `server_graph._build_tools`), so a key added to a running server
takes effect only after a respawn — the app offers that restart, and this
constant is the single name its `/auth` handling compares against.
"""

SERVICE_API_KEY_ENV: dict[str, str] = {
    LANGSMITH_SERVICE: "LANGSMITH_API_KEY",
    TAVILY_SERVICE: "TAVILY_API_KEY",
}
"""Non-model services configurable via `/auth`, mapped to their API-key env var.

These are not LLM providers — they back features such as web search (Tavily) or
agent tracing (LangSmith) — but their credentials follow the same store-on-disk
model as model providers, so they appear in the `/auth` manager and can be
entered directly in the TUI instead of being exported as environment variables
before launch.
"""

CODEX_PROVIDER = "openai_codex"
"""Provider name for `_ChatOpenAICodex` models authenticated via ChatGPT OAuth.

Distinct from `"openai"` (which uses an `OPENAI_API_KEY`) because the auth
source, model class, and request endpoint all differ. See
`deepagents_code.integrations.openai_codex` for the OAuth flow.
"""

CODEX_MODELS: frozenset[str] = frozenset(
    {
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.3-codex",
        "gpt-5.2",
    }
)
"""Curated allowlist of models the Codex (ChatGPT OAuth) backend serves.

The provider mirrors `openai` profiles, but only models in this set are
exposed under `openai_codex`. The Codex backend serves a narrower lineup than
the full `openai` API, so mirroring every openai model would surface specs the
backend rejects at call time.
"""


RETRY_PARAM_BY_PROVIDER: dict[str, str] = {
    "anthropic": "max_retries",
    "azure_openai": "max_retries",
    "baseten": "max_retries",
    "bedrock": "max_retries",
    "deepseek": "max_retries",
    "fireworks": "max_retries",
    "google_genai": "max_retries",
    "google_vertexai": "max_retries",
    "groq": "max_retries",
    "litellm": "max_retries",
    "mistralai": "max_retries",
    "openai": "max_retries",
    "openrouter": "max_retries",
    "perplexity": "max_retries",
    "together": "max_retries",
    "xai": "max_retries",
}
"""Maps a provider to the constructor kwarg that sets its retry count.

The value is the kwarg name to pass to the provider's chat model constructor.
It is uniformly `max_retries` for every provider listed today, but this is a
`dict` rather than a `set` of providers because retry-kwarg names diverge across
the ecosystem -- some integrations expose a differently named kwarg -- and the
value column lets a future provider register its own name without restructuring
callers.

Membership is verified against each provider's chat model constructor (e.g.
`ChatGoogleGenerativeAI` exposes `max_retries`, not `retries`), not inferred
from naming. Providers absent from this map either lack an integer retry-count
kwarg or are not yet wired as a credential-resolvable provider in this module;
a `[retries]` config for them is ignored with a warning by `_resolve_retry_kwargs`.
"""

PROVIDER_BASE_URL_ENV: dict[str, tuple[str, ...]] = {
    # Each tuple lists every base-URL env var the provider's LangChain
    # integration and underlying SDK may read, canonical name first. Names were
    # verified against the integration and SDK source, not inferred:
    #   anthropic     langchain_anthropic reads ANTHROPIC_API_URL; the anthropic
    #                 SDK reads ANTHROPIC_BASE_URL.
    #   azure_openai  AzureChatOpenAI and the openai SDK both read
    #                 AZURE_OPENAI_ENDPOINT.
    #   baseten       ChatBaseten reads BASETEN_BASE_URL, then falls back to
    #                 BASETEN_API_BASE.
    #   cohere        langchain_cohere passes base_url=None, so the cohere SDK's
    #                 CO_API_URL is what takes effect.
    #   deepseek      ChatDeepSeek reads DEEPSEEK_API_BASE (alias base_url).
    #   fireworks     ChatFireworks reads FIREWORKS_API_BASE; when unset the
    #                 fireworks SDK reads FIREWORKS_BASE_URL.
    #   google_genai  the google-genai SDK reads GOOGLE_GEMINI_BASE_URL (the lone
    #                 name langchain_google_genai threads through HttpOptions).
    #   groq          ChatGroq reads GROQ_API_BASE; when unset the groq SDK reads
    #                 GROQ_BASE_URL.
    #   huggingface   the integration and huggingface_hub both read
    #                 HF_INFERENCE_ENDPOINT.
    #   ibm           ChatWatsonx reads WATSONX_URL.
    #   mistralai     ChatMistralAI reads MISTRAL_BASE_URL.
    #   nvidia        ChatNVIDIA reads NVIDIA_BASE_URL.
    #   openai        langchain_openai reads OPENAI_API_BASE; the openai SDK
    #                 reads OPENAI_BASE_URL.
    #   openrouter    ChatOpenRouter reads OPENROUTER_API_BASE (alias base_url).
    #   perplexity    the integration passes no base_url, so the perplexity SDK's
    #                 PERPLEXITY_BASE_URL is what takes effect.
    #   together      ChatTogether reads TOGETHER_API_BASE (alias base_url).
    #   xai           ChatXAI reads XAI_API_BASE (alias base_url).
    #
    # OpenAI-compatible providers (deepseek, openrouter, together, xai, baseten)
    # sit on the openai SDK, whose only base-URL env var is the shared
    # OPENAI_BASE_URL. That name is intentionally NOT listed under those
    # providers: writing or clearing it under another provider's name would
    # clobber the user's real OpenAI endpoint. Each is listed above under its own
    # dedicated name(s) instead. In practice the integration always passes
    # base_url explicitly, so the shared fallback never fires.
    #
    # Omitted (no dedicated, provider-specific endpoint env var): litellm
    # (api_base arg, per-provider env), google_vertexai (endpoint derived from the
    # region). A `/auth` endpoint for these still resolves through the
    # stored-credential step of `get_base_url` and reaches the model as the
    # `base_url` kwarg.
    "anthropic": ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_URL"),
    "azure_openai": ("AZURE_OPENAI_ENDPOINT",),
    "baseten": ("BASETEN_BASE_URL", "BASETEN_API_BASE"),
    "cohere": ("CO_API_URL",),
    "deepseek": ("DEEPSEEK_API_BASE",),
    "fireworks": ("FIREWORKS_BASE_URL", "FIREWORKS_API_BASE"),
    "google_genai": ("GOOGLE_GEMINI_BASE_URL",),
    "groq": ("GROQ_BASE_URL", "GROQ_API_BASE"),
    "huggingface": ("HF_INFERENCE_ENDPOINT",),
    "ibm": ("WATSONX_URL",),
    "mistralai": ("MISTRAL_BASE_URL",),
    "nvidia": ("NVIDIA_BASE_URL",),
    "openai": ("OPENAI_BASE_URL", "OPENAI_API_BASE"),
    "openrouter": ("OPENROUTER_API_BASE",),
    "perplexity": ("PERPLEXITY_BASE_URL",),
    "together": ("TOGETHER_API_BASE",),
    "xai": ("XAI_API_BASE",),
}
"""Every base-URL env var a provider's SDK may read.

Element `[0]` is the *canonical* name — the one we write a stored endpoint to.
`get_base_url` reads each name in tuple order through `resolve_env_var`, so every
base URL gets the same `DEEPAGENTS_CODE_*` > plain-var precedence as API keys.
The remaining names are alternates the SDK might also honor;
`apply_stored_credentials` clears them when applying or resetting an endpoint, so
a stale value (e.g. an inherited gateway URL) can't leak through. Clearing every
name is what lets the write path treat the canonical as authoritative regardless
of which name the SDK prefers.

The key and its endpoint are a coherent pair: a gateway key only works against
the gateway URL, a provider-native key only against the provider's own endpoint,
so both must resolve from the same source.
"""


def _canonical_base_url_env(provider: str) -> str | None:
    """Return the canonical (written) base-URL env var name for a provider.

    The canonical name is element `[0]` of the provider's `PROVIDER_BASE_URL_ENV`
    tuple. Returns `None` for providers outside the built-in set.

    Args:
        provider: Provider name.

    Returns:
        Canonical env var name, or `None` if the provider has no built-in entry.
    """
    names = PROVIDER_BASE_URL_ENV.get(provider)
    return names[0] if names else None


IMPLICIT_AUTH_PROVIDERS: frozenset[str] = frozenset({"google_vertexai"})
"""Providers that support ambient auth outside app env-var checks.

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

PROVIDER_CUSTOM_HEADERS_ENV: dict[str, str] = {"anthropic": "ANTHROPIC_CUSTOM_HEADERS"}
"""Provider SDK env vars that inject custom request headers (e.g. gateway auth)."""

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

    # Mirror the curated `CODEX_MODELS` subset of `openai` models under a
    # dedicated `openai_codex` provider entry so the switcher offers them under
    # their own ChatGPT-OAuth auth context. Eligibility is filtered by the
    # allowlist because the Codex backend serves a narrower lineup than the
    # full `openai` API and rejects unsupported models at call time.
    if config.is_provider_enabled(CODEX_PROVIDER):
        openai_models = available.get("openai")
        if openai_models:
            mirrored = [name for name in openai_models if name in CODEX_MODELS]
            codex_models = list(
                dict.fromkeys([*available.get(CODEX_PROVIDER, []), *mirrored])
            )
            # Place `openai_codex` directly after `openai` so the switcher
            # keeps the two OpenAI-backed providers adjacent (codex before
            # azure_openai etc.) instead of trailing it at the end of the
            # dict. dict insertion order is the switcher's display order, so
            # rebuild the dict, dropping any prior codex entry and re-inserting
            # it right after `openai`.
            reordered: dict[str, list[str]] = {}
            for name, models in available.items():
                if name == CODEX_PROVIDER:
                    continue
                reordered[name] = models
                if name == "openai":
                    reordered[CODEX_PROVIDER] = codex_models
            available = reordered

    _available_models_cache = available
    return available


def _build_entry(
    base: dict[str, Any],
    overrides: dict[str, Any],
    cli_override: dict[str, Any] | None,
) -> ModelProfileEntry:
    """Build a profile entry by merging base, overrides, and app override.

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
    object for the session (the app stores it once on the app instance);
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
            # Mirror the curated `CODEX_MODELS` subset of openai profiles under
            # the `openai_codex` provider so `/model openai_codex:<model>`
            # resolves to the same upstream profile without duplicating data.
            # Filtered by the allowlist — see the note in `get_available_models`.
            if (
                provider == "openai"
                and model_name in CODEX_MODELS
                and config.is_provider_enabled(CODEX_PROVIDER)
            ):
                codex_spec = f"{CODEX_PROVIDER}:{model_name}"
                seen_specs.add(codex_spec)
                codex_overrides = config.get_profile_overrides(
                    CODEX_PROVIDER, model_name=model_name
                )
                result[codex_spec] = _build_entry(
                    upstream_profile, codex_overrides, cli_override
                )

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


def _is_local_endpoint(url: object) -> bool:
    """Return whether a provider endpoint points at the local machine.

    Accepts `object` rather than `str | None` because the endpoint originates
    from untyped TOML; the `isinstance` guard below defends against drift.
    """
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
    `DEEPAGENTS_CODE_`-prefixed variant) is set, its value is forwarded as a
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
        `DEEPAGENTS_CODE_` prefix and dotenv files).

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


def _get_codex_auth_status() -> ProviderAuthStatus:
    """Translate the ChatGPT OAuth on-disk state into a `ProviderAuthStatus`.

    The codex provider uses a file-backed OAuth token store rather than
    `auth_store`'s API-key map, so it gets its own branch in
    `get_provider_auth_status`. The `STORED` source is reused only to satisfy
    the `ProviderAuthStatus` "CONFIGURED implies a source" invariant; it is
    cosmetic here, since `format_auth_badge` routes the codex provider to its
    own `[chatgpt]` / `[sign in to chatgpt]` badge before the source is ever
    consulted.

    Returns:
        `CONFIGURED` / `STORED` when a token bundle sits at the upstream
            default store path; `MISSING` otherwise. Expired access tokens
            are still reported as configured because the file-backed model
            provider can refresh them with the saved refresh token when the
            model is constructed.
    """
    from deepagents_code.integrations import openai_codex

    status = openai_codex.get_status()
    if status.unreadable_reason:
        return ProviderAuthStatus(
            state=ProviderAuthState.MISSING,
            provider=CODEX_PROVIDER,
            detail=f"token store unreadable: {status.unreadable_reason}",
        )
    if not status.logged_in:
        return ProviderAuthStatus(
            state=ProviderAuthState.MISSING,
            provider=CODEX_PROVIDER,
            detail="not signed in to ChatGPT",
        )
    detail = "signed in to ChatGPT"
    if status.plan_type:
        detail = f"signed in to ChatGPT ({status.plan_type})"
    if status.is_expired:
        detail = f"{detail}; access token will refresh on use"
    return ProviderAuthStatus(
        state=ProviderAuthState.CONFIGURED,
        provider=CODEX_PROVIDER,
        source=ProviderAuthSource.STORED,
        detail=detail,
    )


def get_provider_auth_status(provider: str) -> ProviderAuthStatus:
    """Return credential readiness details for a provider.

    Combines config, well-known provider metadata, optional provider auth,
    and implicit-auth provider metadata before attempting model creation:

    1. **Config-file providers** (`config.toml`
        `[models.providers.<name>]`):
        - If the section declares `api_key_env`, that env var is checked
            via `resolve_env_var()` (which honors `DEEPAGENTS_CODE_` prefixes).
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
    # ChatGPT-OAuth-backed codex provider has no env var and stores tokens
    # in its own on-disk JSON; route it through a dedicated helper before
    # the standard config / env-var lookup so callers get the codex-specific
    # `[chatgpt]` / `[sign in to chatgpt]` badge and a "signed in as <plan>"
    # detail.
    if provider == CODEX_PROVIDER:
        return _get_codex_auth_status()

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


def get_base_url_env_vars(provider: str) -> tuple[str, ...]:
    """Return base-URL env var names for a provider in resolution order.

    Checks the config file's `base_url_env` first (user override), then falls
    back to the hardcoded `PROVIDER_BASE_URL_ENV` map.

    Args:
        provider: Provider name.

    Returns:
        Environment variable names, or an empty tuple if the provider has no
        base-URL env var (config-declared or built-in).
    """
    config = ModelConfig.load()
    config_env = config.get_base_url_env(provider)
    if config_env:
        return (config_env,)
    return PROVIDER_BASE_URL_ENV.get(provider, ())


def get_base_url_env_var(provider: str) -> str | None:
    """Return the canonical base-URL env var name for a provider.

    Checks the config file's `base_url_env` first (user override), then falls
    back to the canonical name in the hardcoded `PROVIDER_BASE_URL_ENV` map.
    Parallel to `get_credential_env_var`.

    Args:
        provider: Provider name.

    Returns:
        Environment variable name, or None if the provider has no base-URL env
        var (config-declared or built-in).
    """
    env_vars = get_base_url_env_vars(provider)
    return env_vars[0] if env_vars else None


def get_default_base_url_env(provider: str) -> str | None:
    """Return the env var that supplies a provider's endpoint when none is stored.

    Answers "what does leaving the `/auth` base-URL field blank fall back to?"
    A blank save clears the *plain* endpoint env vars (so an inherited gateway
    URL can't leak through — see `apply_stored_credentials`), so the only env
    var that still supplies a value afterward is the `DEEPAGENTS_CODE_`-prefixed
    one. The name is returned (not its value) for display next to the field, so
    the user sees the knob rather than a long or sensitive URL.

    Returns `None` when that variable holds no value — the endpoint then comes
    from a `config.toml` literal or the provider SDK's own default, neither of
    which is a single env var to name here.

    Args:
        provider: Provider name.

    Returns:
        The `DEEPAGENTS_CODE_`-prefixed env var name still in effect after a
        blank save, or `None`.
    """
    for env_var in get_base_url_env_vars(provider):
        prefixed = f"{_ENV_PREFIX}{env_var}"
        if os.environ.get(prefixed):
            return prefixed
    return None


def is_service(name: str) -> bool:
    """Return whether `name` is a non-model service configurable via `/auth`."""
    return name in SERVICE_API_KEY_ENV


def is_langsmith(name: str) -> bool:
    """Return whether `name` is the LangSmith tracing service.

    Centralizes the identity check so the LangSmith-specific branches (project
    field instead of a base URL, tracing auto-enable) share one definition
    rather than scattering `== LANGSMITH_SERVICE` comparisons.
    """
    return name == LANGSMITH_SERVICE


def get_service_auth_status(service: str) -> ProviderAuthStatus:
    """Return credential readiness for a non-model service (e.g. `"tavily"`).

    Mirrors `get_provider_auth_status` but is scoped to `SERVICE_API_KEY_ENV`,
    so a stored key beats the env var and the `/auth` manager can render the
    same `[stored]` / `[env: ...]` / `[missing]` badges.

    Args:
        service: Service name (e.g. `"tavily"`).

    Returns:
        `CONFIGURED` when a stored or env credential is set, else `MISSING`.
    """
    env_var = SERVICE_API_KEY_ENV[service]
    configured = _resolve_configured(service, env_var)
    if configured:
        return configured
    return ProviderAuthStatus(
        state=ProviderAuthState.MISSING,
        provider=service,
        env_var=env_var,
        detail=f"{env_var} is not set or is empty",
    )


def apply_stored_service_credentials() -> None:
    """Export every stored service key into `os.environ`.

    Services (e.g. web search via Tavily) have no base URL to reconcile, so
    this is a plain key copy onto the canonical env var name the underlying
    SDK reads. A stored key takes precedence over an existing plain env var,
    matching `apply_stored_credentials`; a `DEEPAGENTS_CODE_`-prefixed override
    is left authoritative because the app already treats it as the top-priority
    per-session credential.
    """
    for service, env_var in SERVICE_API_KEY_ENV.items():
        try:
            stored = auth_store.get_stored_key(service)
        except RuntimeError:
            logger.warning(
                "Could not read stored credentials for service %s; the credential "
                "file may be corrupt. Re-add the key via /auth.",
                service,
            )
            continue
        if not stored:
            continue
        prefixed = f"{_ENV_PREFIX}{env_var}"
        if prefixed in os.environ:
            continue
        if os.environ.get(env_var) != stored:
            os.environ[env_var] = stored


def apply_stored_credentials(provider: str) -> bool:
    """Export this provider's stored key *and endpoint* into `os.environ`.

    LangChain's chat-model factories read credentials from process env vars,
    so a stored key only takes effect once it's copied onto the env var name
    registered for that provider. This is a no-op when the provider has no
    env-var mapping (custom auth) or no stored credential.

    The key env var is overwritten whether or not it was already set, matching
    the precedence rule documented on `resolve_provider_credential`: a
    credential the user typed in `/auth` is the most recent deliberate
    action and should take effect.

    Because a key and its endpoint are a coherent pair (a gateway key only
    works against the gateway URL; a provider-native key only against the
    provider's own endpoint), the base URL is applied atomically with the key:

    - A stored `base_url` is written to the provider's canonical base-URL env
        var, and every *other* base-URL name the SDK reads is cleared so an
        inherited gateway URL can't leak through an alternate variable.
    - No stored `base_url` (the user left the field blank) clears *all* of the
        provider's base-URL env vars, so the SDK falls back to the provider
        default rather than an inherited gateway URL. This is what prevents a
        personal key from being shipped to the gateway.

    Only the unprefixed canonical names are written, so an explicit
    `DEEPAGENTS_CODE_{VAR}` override still wins via `resolve_env_var`.

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
        stored_base_url = auth_store.get_stored_base_url(provider)
    except RuntimeError:
        logger.warning("Could not read stored credentials for provider %s", provider)
        return False
    if not stored:
        return False
    # Reconcile the endpoint first: it resolves env-var names (which can touch
    # the config) and so is the only step that might raise. Doing it before the
    # key write means the key is never left applied while an inherited gateway
    # URL stays uncleared — the key and endpoint move together.
    _apply_stored_base_url(provider, stored_base_url)
    if os.environ.get(env_var) != stored:
        os.environ[env_var] = stored
    return True


def _apply_stored_base_url(provider: str, base_url: str | None) -> None:
    """Reconcile a provider's base-URL env vars with a `/auth` credential.

    Writes `base_url` to the canonical name and clears the alternates, or
    clears every name when `base_url` is `None` (reset to the provider
    default). See `apply_stored_credentials` for the pairing rationale.

    When switching to a provider-native key (no `base_url`), also clears the
    provider's custom-headers env var (e.g. `ANTHROPIC_CUSTOM_HEADERS`) so a
    gateway-provisioned auth header isn't sent to the native endpoint.

    Args:
        provider: Provider name.
        base_url: The stored endpoint, or `None` to reset to the default.
    """
    canonical = get_base_url_env_var(provider)
    # Clear every name the SDK might read: the built-in alternates plus any
    # config-declared `base_url_env` (which extends pairing to providers
    # outside the hardcoded set).
    names = set(PROVIDER_BASE_URL_ENV.get(provider, ()))
    if canonical:
        names.add(canonical)
    if not names:
        return
    configured_base_url_survives = _configured_base_url_survives_env_clear(provider)
    for name in names:
        if base_url and name == canonical:
            os.environ[name] = base_url
        else:
            os.environ.pop(name, None)

    # A provider SDK's custom-header env var (e.g. `ANTHROPIC_CUSTOM_HEADERS`)
    # injects headers into every request. A gateway-provisioned environment
    # often sets it to `X-Api-Key: <gateway-key>`, which overrides the SDK's
    # own `api_key`-derived header. When switching to a provider-native key
    # (no stored `base_url`), that header must also be cleared — otherwise the
    # gateway key is sent to the native endpoint and rejected.
    custom_headers_env = PROVIDER_CUSTOM_HEADERS_ENV.get(provider)
    if custom_headers_env and not base_url:
        if not configured_base_url_survives:
            if os.environ.pop(custom_headers_env, None) is not None:
                # Log the env var name only — never its value, which carries
                # auth headers. Surfaces the removal for the user who set a
                # header deliberately for the native endpoint and later wonders
                # where it went.
                logger.info(
                    "Cleared %s while applying a provider-native %s key",
                    custom_headers_env,
                    provider,
                )
        elif os.environ.get(custom_headers_env) is not None:
            # A provider base URL still routes (config or a prefixed env var),
            # so the custom-header env is deliberately kept. Log the name only —
            # never the value — so the retention is observable when a user later
            # wonders why a gateway header is still in effect after applying a
            # native key.
            logger.debug(
                "Kept %s: a %s base URL is still configured",
                custom_headers_env,
                provider,
            )


def _configured_base_url_survives_env_clear(provider: str) -> bool:
    """Return whether endpoint config still routes after plain env cleanup."""
    config = ModelConfig.load()
    provider_cfg = config.providers.get(provider)
    if provider_cfg and provider_cfg.get("base_url"):
        return True
    for env_var in get_base_url_env_vars(provider):
        if os.environ.get(f"{_ENV_PREFIX}{env_var}"):
            return True
    return False


def warn_on_split_credential_source(provider: str) -> None:
    """Log when a provider's key and endpoint resolve from different env tiers.

    The `DEEPAGENTS_CODE_` prefix is a *per-variable* override, not a credential
    bundle: setting `DEEPAGENTS_CODE_OPENAI_API_KEY` while leaving the endpoint to
    a plain `OPENAI_BASE_URL` makes the key resolve from the prefixed tier and the
    endpoint from the unprefixed one. A key and its endpoint are a coherent pair
    (see `PROVIDER_BASE_URL_ENV`), so a split source is a likely misconfiguration
    -- e.g. a provider-native key shipped to a gateway URL, or vice versa.

    This is purely diagnostic: it never mutates `os.environ` or changes
    resolution. Only the env var *names* are logged, never the secret value or
    the URL. It is emitted at DEBUG because the `deepagents_code` package logger
    only attaches a handler when `DEEPAGENTS_CODE_DEBUG` is set, and DEBUG stays
    below `logging.lastResort`'s WARNING stderr threshold so it cannot bleed onto
    stderr and corrupt the Textual TUI. The `DEEPAGENTS_CODE_DEBUG` file log is
    where someone chasing a wrong-endpoint bug will look.

    A `config.toml` `base_url` literal wins over env vars in `get_base_url`, so
    when one is set there is no env-tier split to flag and this returns early.

    Args:
        provider: Provider name (e.g. `"openai"`).
    """
    key_env = get_credential_env_var(provider)
    base_env = get_base_url_env_var(provider)
    if not key_env or not base_env:
        return
    config = ModelConfig.load()
    provider_cfg = config.providers.get(provider)
    if provider_cfg and provider_cfg.get("base_url"):
        return
    prefixed_key = f"{_ENV_PREFIX}{key_env}"
    prefixed_base = f"{_ENV_PREFIX}{base_env}"
    # Key must actually resolve from the prefixed tier (present and non-empty),
    # while the endpoint falls back to the plain tier: no prefixed override
    # present (an empty prefixed var would shadow the plain one in
    # `resolve_env_var`, so its mere presence means the endpoint is not "plain").
    key_from_prefixed = bool(os.environ.get(prefixed_key))
    base_from_plain = prefixed_base not in os.environ and bool(os.environ.get(base_env))
    if key_from_prefixed and base_from_plain:
        logger.debug(
            "Provider %s: API key resolved from %s but base URL resolved from "
            "the unprefixed %s. Key and endpoint came from different sources and "
            "may not be a matching pair. Set %s to pin the endpoint, or unset %s.",
            provider,
            prefixed_key,
            base_env,
            prefixed_base,
            base_env,
        )


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
                Returns empty config if file is missing, unreadable, contains
                invalid TOML syntax, or is structurally invalid (valid TOML of
                the wrong shape, e.g. a scalar `[models]`).
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
            models_section = data.get("models", {})
            config = cls(
                default_model=models_section.get("default"),
                recent_model=models_section.get("recent"),
                providers=models_section.get("providers", {}),
            )
        except tomllib.TOMLDecodeError as e:
            logger.warning(
                "Config file %s has invalid TOML syntax: %s. "
                "Ignoring config file. Fix the file or delete it to reset.",
                config_path,
                e,
            )
            config = cls()
        except (PermissionError, OSError) as e:
            logger.warning("Could not read config file %s: %s", config_path, e)
            config = cls()
        except (AttributeError, TypeError) as e:
            # Syntactically valid TOML can still have the wrong shape — a scalar
            # `[models]`, a non-table `providers` — which surfaces here as an
            # AttributeError from `.get(...)` or a TypeError from the dataclass
            # constructor. Treat it like any other unreadable config rather than
            # letting it crash callers (e.g. the /auth modal on Ctrl+R) that
            # assume load() is total and never raises.
            logger.warning(
                "Config file %s is structurally invalid: %s. "
                "Ignoring config file. Fix the file or delete it to reset.",
                config_path,
                e,
            )
            config = cls()

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
            # `enabled` originates from untyped TOML; cast to `object` so the
            # runtime non-bool validation below stays reachable (the TypedDict
            # types it as `bool`, which would otherwise mark this branch dead).
            enabled = cast("object", provider.get("enabled"))
            if enabled is not None and not isinstance(enabled, bool):
                logger.warning(
                    "Provider '%s' has non-boolean 'enabled' value %r "
                    "(expected true/false). Provider will remain visible.",
                    name,
                    enabled,
                )

            # `display_name`/`api_key_url` also originate from untyped TOML; cast
            # to `object` so the runtime non-string checks stay reachable (the
            # TypedDict types them as `str`).
            display_name = cast("object", provider.get("display_name"))
            if display_name is not None and not isinstance(display_name, str):
                logger.warning(
                    "Provider '%s' has non-string 'display_name' value %r "
                    "(expected a string). Falling back to the default label.",
                    name,
                    display_name,
                )

            short_name = cast("object", provider.get("short_name"))
            if short_name is not None and not isinstance(short_name, str):
                logger.warning(
                    "Provider '%s' has non-string 'short_name' value %r "
                    "(expected a string). Falling back to the display name.",
                    name,
                    short_name,
                )

            api_key_url = cast("object", provider.get("api_key_url"))
            if api_key_url is not None and not isinstance(api_key_url, str):
                logger.warning(
                    "Provider '%s' has non-string 'api_key_url' value %r "
                    "(expected a string). Ignoring it.",
                    name,
                    api_key_url,
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
        """Get the configured base URL for a provider.

        Resolution order (first match wins):

        1. `base_url` in the provider's `config.toml` section.
        2. The provider's base-URL env vars via `resolve_env_var`, in provider
            precedence order, so `DEEPAGENTS_CODE_{VAR}` beats the plain `{VAR}`
            for each name — mirroring how API keys resolve. This also surfaces
            the value `apply_stored_credentials` bridged in from a `/auth`
            credential, and the gateway-provisioned URL in the default
            (no-override) case.
        3. The endpoint stored with a `/auth` credential. This is the source
            for providers that have no base-URL env var (e.g. an OpenAI-
            compatible provider like Litellm): step 2 has no name to read, so
            the stored endpoint is taken directly. It then reaches the model as
            the `base_url` constructor kwarg via
            `_get_provider_kwargs`, the same path a `config.toml` literal uses.
            For providers that *do* have an env var, the stored endpoint already
            arrives via step 2 (it was bridged onto the env var), so this step
            is a redundant — and consistent — fallback.

        This function only *resolves* the endpoint; whether it takes effect is a
        separate contract owned by the provider's LangChain class. The value is
        delivered as the `base_url` kwarg (see `_get_provider_kwargs`), which the
        OpenAI/Anthropic-compatible classes accept via a Pydantic `base_url`
        alias. A class that names the field differently may silently
        ignore `base_url` — Pydantic models default to `extra="ignore"` — so for
        those the endpoint must be set via `params`.

        A corrupt credential store is treated as "no stored endpoint" rather than
        propagating, so endpoint resolution never newly raises.

        Args:
            provider_name: The provider to get base URL for.

        Returns:
            Base URL if configured, None otherwise.
        """
        provider = self.providers.get(provider_name)
        config_url = provider.get("base_url") if provider else None
        if config_url:
            return config_url
        config_env = provider.get("base_url_env") if provider else None
        env_vars = (
            (config_env,)
            if config_env
            else PROVIDER_BASE_URL_ENV.get(provider_name, ())
        )
        for env_var in env_vars:
            resolved = resolve_env_var(env_var)
            if resolved:
                return resolved
        try:
            return auth_store.get_stored_base_url(provider_name)
        except RuntimeError:
            return None

    def get_api_key_env(self, provider_name: str) -> str | None:
        """Get the environment variable name for a provider's API key.

        Args:
            provider_name: The provider to get API key env var for.

        Returns:
            Environment variable name if configured, None otherwise.
        """
        provider = self.providers.get(provider_name)
        return provider.get("api_key_env") if provider else None

    def get_provider_display_name(self, provider_name: str) -> str | None:
        """Get the configured display name for a provider.

        Args:
            provider_name: The provider to look up.

        Returns:
            Human-readable display name if configured, None otherwise.
        """
        provider = self.providers.get(provider_name)
        name = provider.get("display_name") if provider else None
        return name if isinstance(name, str) else None

    def get_provider_short_name(self, provider_name: str) -> str | None:
        """Get the configured compact brand name for a provider.

        Args:
            provider_name: The provider to look up.

        Returns:
            Compact brand name if configured, None otherwise.
        """
        provider = self.providers.get(provider_name)
        name = provider.get("short_name") if provider else None
        return name if isinstance(name, str) else None

    def get_provider_api_key_url(self, provider_name: str) -> str | None:
        """Get the configured API-key management URL for a provider.

        Args:
            provider_name: The provider to look up.

        Returns:
            API-key management URL if configured, None otherwise.
        """
        provider = self.providers.get(provider_name)
        url = provider.get("api_key_url") if provider else None
        return url if isinstance(url, str) else None

    def get_base_url_env(self, provider_name: str) -> str | None:
        """Get the environment variable name for a provider's base URL.

        Args:
            provider_name: The provider to get the base-URL env var for.

        Returns:
            Environment variable name if configured, None otherwise.
        """
        provider = self.providers.get(provider_name)
        return provider.get("base_url_env") if provider else None

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


@dataclass(frozen=True, order=True)
class McpProjectServerApproval:
    """A project-scoped, definition-bound MCP server approval.

    Membership in a `McpServerTrustLists.approvals` set *is* the trust decision
    (`is_enabled` reconstructs an approval and tests `approval in approvals`), so
    value equality on the three raw strings must line up between the write side
    (`add_enabled_project_mcp_servers`) and the read side (`is_enabled`). Build
    runtime approvals through `create`, never the raw constructor, so both sides
    normalize the root and compute the fingerprint identically; the raw
    constructor is reserved for deserializing already-computed fields from TOML.
    `order=True` exists only so `sorted()` yields deterministic TOML output.
    """

    project_root: str
    """Resolved project root that originated the approval."""

    name: str
    """MCP server name within the project config."""

    fingerprint: str
    """Fingerprint of the approved MCP server definition."""

    def __post_init__(self) -> None:
        """Reject degenerate approvals so a bad one can't silently never match.

        An empty `project_root`, `name`, or `fingerprint` can only ever equal a
        malformed peer, so forbid the state entirely rather than let it persist.

        Raises:
            ValueError: If any field is empty or whitespace-only.
        """
        if not (
            self.project_root.strip() and self.name.strip() and self.fingerprint.strip()
        ):
            msg = (
                "McpProjectServerApproval requires non-empty project_root, name, "
                "and fingerprint"
            )
            raise ValueError(msg)

    @classmethod
    def create(
        cls, *, project_root: str | Path | None, name: str, server: object
    ) -> McpProjectServerApproval | None:
        """Build an approval, normalizing the root and fingerprinting `server`.

        The single construction path for runtime approvals, so the write and read
        sides cannot drift on how the root is normalized or the server hashed.

        Args:
            project_root: Project root to normalize.
            name: MCP server name.
            server: Parsed MCP server definition to fingerprint.

        Returns:
            The approval, or `None` when `project_root` cannot be normalized.
        """
        normalized_root = normalize_mcp_project_root(project_root)
        if normalized_root is None:
            return None
        return cls(
            project_root=normalized_root,
            name=name,
            fingerprint=fingerprint_mcp_server_config(server),
        )

    def as_toml(self) -> dict[str, str]:
        """Return a TOML-serializable representation."""
        return {
            "project_root": self.project_root,
            "name": self.name,
            "fingerprint": self.fingerprint,
        }


def normalize_mcp_project_root(project_root: str | Path | None) -> str | None:
    """Normalize a project root for persisted MCP trust comparisons.

    Args:
        project_root: Project root path to normalize.

    Returns:
        Resolved absolute project root string, or `None` when `project_root`
            is `None`. On an `OSError` from `resolve()`, returns the expanded
            but *unresolved* path as a fallback; a transient failure on only one
            of the write/read sides then yields different strings and a spurious
            re-prompt (fail-closed), never a false match.
    """
    if project_root is None:
        return None
    try:
        return str(Path(project_root).expanduser().resolve())
    except OSError:
        logger.warning(
            "Could not resolve MCP project root %s",
            project_root,
            exc_info=True,
        )
        return str(Path(project_root).expanduser())


def fingerprint_mcp_server_config(server: object) -> str:
    """Return a stable fingerprint for an MCP server definition.

    Typed `object` because callers thread the server through as `object`, but
    the contract is a JSON-serializable value (in practice the `dict` parsed
    from `.mcp.json`); a non-serializable input raises `TypeError` from
    `json.dumps`. `sort_keys=True` makes the digest independent of key order,
    so reordering fields in the config does not force a re-prompt.

    Args:
        server: Parsed MCP server config (a JSON-serializable value).

    Returns:
        A SHA-256 fingerprint over the canonical JSON representation.
    """
    encoded = json.dumps(
        server,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class McpServerTrustLists:
    """User-level allow/deny lists for project MCP servers.

    Sourced only from the user's own configuration — the home `config.toml`, the
    global `~/.deepagents/.env`, and shell-exported env — never from a repo, so a
    committed `.mcp.json` cannot self-approve. Persisted approvals are scoped to
    a resolved project root and the approved server definition's fingerprint.
    Env-sourced approvals remain explicit process-wide name approvals.

    The "reject wins" invariant — a name in both approval and rejection data is
    only rejected — is enforced in `__post_init__`, so every instance is disjoint
    no matter how it was constructed; callers need not pre-subtract.
    """

    enabled: frozenset[str]
    """Env-sourced server names pre-approved for any project config."""

    disabled: frozenset[str]
    """Server names always rejected; reject wins over approvals and over trust."""

    approvals: frozenset[McpProjectServerApproval] = frozenset()
    """Project-scoped approvals loaded from user `config.toml`."""

    read_error: str | None = field(default=None, compare=False)
    """Non-`None` when the user's `config.toml` existed but its trust policy
    could not be fully read: the file was unreadable/unparseable, its `[mcp]`
    value was not a table, or its `disabled_project_servers` was a wrong type
    that could not be interpreted as a deny list. Callers must treat this as
    fail-closed (do not grant whole-config project trust) and surface it, rather
    than proceeding with a deny list that may not have loaded — use `load_failed`
    for that check. Note the resolved `enabled`/`disabled` sets are not
    necessarily empty here: names from a still-readable source (the env vars)
    continue to apply. Excluded from equality so a failed load still compares
    equal to empty lists for tests that only care about the resolved names."""

    legacy_ignored: frozenset[str] = field(default=frozenset(), compare=False)
    """Names found in a legacy `[mcp].enabled_project_servers` list that this
    build no longer honors. Non-empty means the user relied on the removed flat
    allowlist, so those servers silently stopped loading; callers should surface
    it (a bare `logger.warning` is invisible outside debug mode) so
    non-interactive paths can explain the change. Diagnostic, not resolved
    policy — excluded from equality like `read_error`."""

    def __post_init__(self) -> None:
        """Enforce reject precedence by stripping disabled names from both sets.

        A rejected name must never survive in `enabled` or `approvals`, whatever
        the caller passed, so a future allow-first consumer can't be tricked
        into loading a denied server. Frozen dataclass, so assign via
        `object.__setattr__`.
        """
        if self.enabled & self.disabled:
            object.__setattr__(self, "enabled", self.enabled - self.disabled)
        if any(approval.name in self.disabled for approval in self.approvals):
            object.__setattr__(
                self,
                "approvals",
                frozenset(
                    approval
                    for approval in self.approvals
                    if approval.name not in self.disabled
                ),
            )

    @property
    def load_failed(self) -> bool:
        """Whether the user's trust policy failed to load (see `read_error`).

        Callers gating on trust MUST check this and fail closed: a failed load
        means a configured deny may be missing, so whole-config project trust
        must not be honored. Named so the fail-closed contract is discoverable
        rather than resting on every caller remembering the `read_error`
        sentinel.
        """
        return self.read_error is not None

    def is_enabled(
        self,
        name: str,
        *,
        project_root: str | Path | None,
        server: object,
    ) -> bool:
        """Return whether `server` is approved by name or scoped fingerprint.

        Args:
            name: MCP server name.
            project_root: Resolved project root for the config that defined it.
            server: Parsed MCP server config for fingerprint comparison.

        Returns:
            `True` when the server is approved and not disabled.
        """
        if not name.strip():
            # A blank name can only come from a malformed config. Fail closed
            # here rather than let `McpProjectServerApproval.create` raise
            # `ValueError` from its non-empty invariant out of the trust filter.
            return False
        if name in self.disabled:
            return False
        if name in self.enabled:
            return True
        approval = McpProjectServerApproval.create(
            project_root=project_root, name=name, server=server
        )
        if approval is None:
            return False
        return approval in self.approvals


def _parse_csv_env(name: str) -> list[str] | None:
    """Parse a comma-separated env var into a list of trimmed, non-empty names.

    Returns:
        The parsed list when the variable is set (possibly empty after
            trimming), or `None` when the variable is unset so callers can
            distinguish "unset, fall back to TOML" from "set but empty".
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _toml_str_list(
    value: object, *, key: str, config_path: Path
) -> tuple[list[str], bool]:
    """Coerce a raw TOML value into a list of trimmed, non-empty server names.

    A bare string is *split on commas* (e.g. `disabled_project_servers = "a, b"`
    yields `["a", "b"]`), so a scalar written in the TOML parses identically to
    the comma-separated env form in `_parse_csv_env` — the two forms can never
    silently diverge into one bogus `"a, b"` token that matches no server. Non-
    string list elements are dropped (with a log) while the surrounding valid
    names survive. A genuinely wrong type (number, table, bool) cannot be
    interpreted as names at all: it yields an empty list *and* flags `malformed`,
    so a caller enforcing a deny list can fail closed rather than silently drop
    the rejection.

    Args:
        value: The raw value read from the `[mcp]` table (or `None` when the
            key is absent).
        key: The TOML key name, used only for log context.
        config_path: The config file the value came from, for log context.

    Returns:
        `(names, malformed)`. `names` are the trimmed, non-empty server names.
            `malformed` is `True` only when `value` is present but neither a
            string nor a list (so it could not be read as names); it is `False`
            for an absent value, a string, or any list — even one whose non-
            string elements were dropped.
    """
    if value is None:
        return [], False
    if isinstance(value, str):
        # Split on commas so a bare string parses exactly like the env form; a
        # single name with no comma still yields a one-element list.
        return [item.strip() for item in value.split(",") if item.strip()], False
    if not isinstance(value, list):
        logger.warning(
            "[mcp].%s in %s should be a list of strings, got %s; ignoring it",
            key,
            config_path,
            type(value).__name__,
        )
        return [], True
    result: list[str] = []
    discarded = 0
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
        else:
            discarded += 1
    if discarded:
        logger.warning(
            "[mcp].%s in %s: ignored %d non-string or empty entr%s",
            key,
            config_path,
            discarded,
            "y" if discarded == 1 else "ies",
        )
    return result, False


def _toml_project_server_approvals(
    value: object, *, config_path: Path
) -> list[McpProjectServerApproval]:
    """Parse `[mcp].enabled_project_server_approvals` entries.

    Args:
        value: Raw TOML value from the `[mcp]` table.
        config_path: Config file the value came from, for log context.

    Returns:
        Well-formed project-scoped approvals. Malformed entries are dropped,
        which is fail-closed for an allowlist.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        logger.warning(
            "[mcp].enabled_project_server_approvals in %s should be a list of "
            "tables; ignoring it",
            config_path,
        )
        return []

    approvals: list[McpProjectServerApproval] = []
    for item in value:
        if not isinstance(item, dict):
            logger.warning(
                "[mcp].enabled_project_server_approvals in %s ignored a "
                "non-table entry",
                config_path,
            )
            continue
        project_root = item.get("project_root")
        name = item.get("name")
        fingerprint = item.get("fingerprint")
        if not (
            isinstance(project_root, str)
            and project_root.strip()
            and isinstance(name, str)
            and name.strip()
            and isinstance(fingerprint, str)
            and fingerprint.strip()
        ):
            logger.warning(
                "[mcp].enabled_project_server_approvals in %s ignored an "
                "entry missing project_root, name, or fingerprint",
                config_path,
            )
            continue
        approvals.append(
            McpProjectServerApproval(
                project_root=(
                    normalize_mcp_project_root(project_root) or project_root.strip()
                ),
                name=name.strip(),
                fingerprint=fingerprint.strip(),
            )
        )
    return approvals


def load_mcp_server_trust_lists(
    config_path: Path | None = None,
) -> McpServerTrustLists:
    """Load per-server project MCP allow/deny lists from user-level config.

    Security boundary: this reads the `[mcp]` table only from the user-level
    `config.toml` (`DEFAULT_CONFIG_PATH`, i.e. `~/.deepagents/config.toml`) and
    the `DEEPAGENTS_CODE_DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS` /
    `DEEPAGENTS_CODE_DISABLED_PROJECT_MCP_SERVERS` process env vars — never from
    a project's `.mcp.json` or any repo-committed file. There is no
    project-level `config.toml` discovery, so an attacker who commits a
    malicious `.mcp.json` plus an in-repo config cannot pre-approve their own
    servers; the approval must live in the user's home config. This mirrors
    Claude Code's "untrusted folder → only non-checked-in settings" rule.

    Source resolution differs by list, matching each one's security direction:

    - `enabled` (permissive): the env var is an explicit process-wide name
        allowlist. When set, it *replaces* scoped TOML approvals
        (env-beats-config, as elsewhere). Clearing it via an empty env value is
        fail-closed — it only ever pre-approves fewer servers.
    - `approvals` (permissive): TOML approvals are scoped to project root and a
        server-definition fingerprint. Legacy flat TOML `enabled_project_servers`
        entries are ignored because they cannot be safely scoped.
    - `disabled` (restrictive): the env var *unions* with the TOML list — denies
        accumulate and a lower-effort source can never silently empty a deny
        entry set in the other, which would be a fail-open. There is
        deliberately no way to *remove* a configured deny via env.

    Rejection wins: a name appearing in approval and disabled data is reported
    only in `disabled`.

    Args:
        config_path: Config file to read. Defaults to `DEFAULT_CONFIG_PATH`;
            callers should not point this at a project path — doing so would
            defeat the boundary above.

    Returns:
        The resolved `McpServerTrustLists`. A missing file yields empty lists
            (the normal "unset" case). `read_error` is set (so callers can fail
            closed instead of treating a broken config as "nothing denied") when
            the file exists but cannot be read/parsed, when `[mcp]` is not a
            table, or when `disabled_project_servers` is a wrong type that cannot
            be read as a deny list; env-sourced names still apply in that case.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    toml_approvals: list[McpProjectServerApproval] = []
    toml_disabled: list[str] = []
    legacy_ignored: list[str] = []
    read_error: str | None = None
    try:
        if config_path.exists():
            with config_path.open("rb") as f:
                data = tomllib.load(f)
            mcp_section = data.get("mcp", {})
            if isinstance(mcp_section, dict):
                toml_approvals = _toml_project_server_approvals(
                    mcp_section.get("enabled_project_server_approvals"),
                    config_path=config_path,
                )
                legacy_enabled, _ = _toml_str_list(
                    mcp_section.get("enabled_project_servers"),
                    key="enabled_project_servers",
                    config_path=config_path,
                )
                if legacy_enabled:
                    legacy_ignored = legacy_enabled
                    logger.warning(
                        "[mcp].enabled_project_servers in %s is ignored; run "
                        "the project MCP approval prompt again to save "
                        "project-scoped approvals",
                        config_path,
                    )
                toml_disabled, disabled_malformed = _toml_str_list(
                    mcp_section.get("disabled_project_servers"),
                    key="disabled_project_servers",
                    config_path=config_path,
                )
                if disabled_malformed:
                    # A wrong-typed deny list cannot be read, so proceeding as
                    # if nothing were denied would be a fail-open. Surface it and
                    # fail closed, mirroring the unreadable-file path below.
                    read_error = (
                        f"[mcp].disabled_project_servers in {config_path} must be "
                        "a list of strings; refusing to proceed with an "
                        "unenforced deny list"
                    )
            else:
                # An `[mcp]` value that is not a table means the deny list is
                # unreadable too; fail closed rather than leave it unenforced.
                read_error = (
                    f"[mcp] in {config_path} must be a table, got "
                    f"{type(mcp_section).__name__}"
                )
                logger.warning(
                    "[mcp] in %s should be a table, got %s; treating project "
                    "configs as untrusted",
                    config_path,
                    type(mcp_section).__name__,
                )
    except (OSError, tomllib.TOMLDecodeError) as exc:
        # The file exists but is unreadable/unparseable. Record it so callers
        # fail closed rather than silently proceeding with an empty deny list.
        read_error = f"Could not read MCP trust lists from {config_path}: {exc}"
        logger.warning(
            "Could not read %s for MCP server trust lists; treating project "
            "configs as untrusted",
            config_path,
            exc_info=True,
        )

    env_enabled = _parse_csv_env(_env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS)
    env_disabled = _parse_csv_env(_env_vars.DISABLED_PROJECT_MCP_SERVERS)

    # Enabled env remains an explicit process-wide name allowlist. TOML approvals
    # are scoped to the project root and server fingerprint; a set env var
    # replaces those TOML approvals to preserve env-beats-config semantics.
    enabled = frozenset(env_enabled or ())
    approvals = frozenset(() if env_enabled is not None else toml_approvals)
    disabled = frozenset(toml_disabled) | frozenset(env_disabled or ())
    # Reject precedence is enforced by `McpServerTrustLists.__post_init__`, so no
    # subtraction here.
    return McpServerTrustLists(
        enabled=enabled,
        disabled=disabled,
        approvals=approvals,
        read_error=read_error,
        legacy_ignored=frozenset(legacy_ignored),
    )


def add_enabled_project_mcp_servers(
    names: Iterable[str],
    config_path: Path | None = None,
    *,
    project_root: str | Path | None = None,
    server_configs: Mapping[str, object] | None = None,
) -> bool:
    """Persist project-scoped MCP server approvals.

    Backs the interactive approval prompt's "always allow" choice: the given
    names are added to the user-level `config.toml` allowlist with the current
    project root and each server definition's fingerprint, so a different repo
    or a changed command/URL under the same name asks for approval again.

    Defaults to the user-level config (`DEFAULT_CONFIG_PATH`), the sole source
    `load_mcp_server_trust_lists` reads the allowlist from — so writing to the
    user's home config is what preserves the read-side trust boundary (a
    committed `.mcp.json` can never self-approve). The write is atomic
    (`tempfile.mkstemp` + `Path.replace`), matching `suppress_warning`.

    Args:
        names: Server names to add to the allowlist. Blank/whitespace-only
            names are ignored; a call with no usable names is a no-op success.
        config_path: Config file to write. Defaults to `DEFAULT_CONFIG_PATH`
            (`~/.deepagents/config.toml`). Callers should not point this at a
            project path: the loader only ever reads the user-level config, so
            an allowlist written elsewhere is never honored.
        project_root: Project root whose MCP server definitions were approved.
        server_configs: Current server definitions keyed by server name.

    Returns:
        `True` if the save succeeded (or there was nothing to add), `False` on
            I/O, parse failure, or missing project/server context.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    clean_names = [name.strip() for name in names if name and name.strip()]
    if not clean_names:
        return True

    normalized_root = normalize_mcp_project_root(project_root)
    if normalized_root is None or server_configs is None:
        logger.error(
            "Cannot save enabled project MCP servers without project root and "
            "server definitions"
        )
        return False

    approvals_to_add: list[McpProjectServerApproval] = []
    for name in clean_names:
        if name not in server_configs:
            logger.error("Cannot save unknown project MCP server %r", name)
            return False
        approval = McpProjectServerApproval.create(
            project_root=normalized_root, name=name, server=server_configs[name]
        )
        if approval is None:
            # normalized_root is already non-None here, so create cannot fail;
            # guard anyway so a future change can't smuggle in a bad approval.
            logger.error("Could not build approval for project MCP server %r", name)
            return False
        approvals_to_add.append(approval)

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if config_path.exists():
            with config_path.open("rb") as f:
                data = tomllib.load(f)
        else:
            data = {}

        mcp_section = data.get("mcp")
        if not isinstance(mcp_section, dict):
            mcp_section = {}
        existing = _toml_project_server_approvals(
            mcp_section.get("enabled_project_server_approvals"),
            config_path=config_path,
        )
        merged = set(existing) | set(approvals_to_add)
        mcp_section["enabled_project_server_approvals"] = [
            approval.as_toml() for approval in sorted(merged)
        ]
        data["mcp"] = mcp_section

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
        logger.exception(
            "Could not save enabled project MCP servers to %s", config_path
        )
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

    scope: str
    """`'cwd'` (current working directory) or `'all'` (all directories)."""


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
    scope = "cwd"

    try:
        if not config_path.exists():
            result = ThreadConfig(columns, relative_time, sort_order, scope)
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

        # scope
        scope_value = threads_section.get("scope")
        if scope_value in {"cwd", "all"}:
            scope = scope_value
    except (OSError, tomllib.TOMLDecodeError):
        logger.warning("Could not read thread config; using defaults", exc_info=True)
        # Do not cache on error — allow retry on next call in case the
        # file is fixed or permissions are restored.
        return ThreadConfig(columns, relative_time, sort_order, scope)

    result = ThreadConfig(columns, relative_time, sort_order, scope)
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


STARTUP_MODE_MANUAL = "manual"
"""Startup approval mode that keeps human-in-the-loop approvals enabled."""

STARTUP_MODE_DANGEROUSLY_AUTO = "dangerously-auto"
"""Startup approval mode that auto-approves gated tool calls at launch."""

VALID_STARTUP_MODES = frozenset({STARTUP_MODE_MANUAL, STARTUP_MODE_DANGEROUSLY_AUTO})
"""Accepted values for the `[startup].mode` config option."""

DEFAULT_STARTUP_MODE = STARTUP_MODE_MANUAL
"""Fallback startup mode when `[startup].mode` is missing, unreadable, or invalid."""


def load_startup_mode(config_path: Path | None = None) -> str:
    """Load the default startup approval mode from config.toml.

    Reads `[startup].mode`, which controls whether the interactive TUI launches
    with human-in-the-loop approvals enabled (`manual`) or auto-approved
    (`dangerously-auto`). The explicit `-y`/`--auto-approve` flag overrides this.

    Args:
        config_path: Path to config file.

    Returns:
        `"manual"` or `"dangerously-auto"`; falls back to `"manual"` when unset,
        unreadable, or invalid.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    try:
        if not config_path.exists():
            return DEFAULT_STARTUP_MODE
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        startup = data.get("startup")
        value = startup.get("mode") if isinstance(startup, dict) else None
        # `value` may be any TOML type; guard against non-strings (e.g. an
        # array or table) before the frozenset membership test, which would
        # otherwise raise `TypeError: unhashable type` and crash startup.
        if isinstance(value, str) and value in VALID_STARTUP_MODES:
            return value
        if value is not None:
            logger.warning(
                "Ignoring [startup].mode=%r (expected 'manual' or 'dangerously-auto')",
                value,
            )
    except (OSError, tomllib.TOMLDecodeError):
        logger.debug("Could not read startup mode config", exc_info=True)
    return DEFAULT_STARTUP_MODE


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


def save_thread_scope(scope: str, config_path: Path | None = None) -> bool:
    """Save the directory-scope preference for the thread selector.

    Args:
        scope: `"cwd"` (current working directory) or `"all"` (all directories).
        config_path: Path to config file.

    Returns:
        True if save succeeded, False on I/O error.

    Raises:
        ValueError: If `scope` is not a recognised value.
    """
    if scope not in {"cwd", "all"}:
        msg = f"Invalid scope {scope!r}; expected 'cwd' or 'all'"
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
        data["threads"]["scope"] = scope
        fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                tomli_w.dump(data, f)
            Path(tmp_path).replace(config_path)
        except BaseException:
            # Clean up temp file on any failure, including interrupts.
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError):
        # `TypeError`/`ValueError` cover `tomli_w.dump` rejecting a payload
        # from a pre-existing config that does not round-trip; folding them in
        # keeps the `bool` contract intact for `_persist_scope`'s failure toast.
        logger.exception("Could not save thread scope preference")
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


def _recent_models_path(state_dir: Path | None = None) -> Path:
    """Resolve the JSON file path for the recent-models MRU cache.

    Args:
        state_dir: Override for the state directory (test hook).

    Returns:
        Absolute path to `recent_models.json` under the chosen state dir.
    """
    return (state_dir or DEFAULT_STATE_DIR) / RECENT_MODELS_FILENAME


def load_recent_models(state_dir: Path | None = None) -> list[str]:
    """Read the most-recent-first list of `provider:model` specs.

    Missing or malformed files yield an empty list rather than raising; the
    recent section is a non-essential UI affordance and must not block the
    selector from rendering.

    Args:
        state_dir: Override for the state directory (test hook).

    Returns:
        Ordered list of recent `provider:model` specs, most recent first.
            Capped at `RECENT_MODELS_LIMIT` and de-duplicated.
    """
    path = _recent_models_path(state_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read recent models cache at %s", path, exc_info=True)
        return []
    raw = data.get("models") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for entry in raw:
        if not isinstance(entry, str) or ":" not in entry or entry in seen:
            continue
        seen.add(entry)
        out.append(entry)
        if len(out) >= RECENT_MODELS_LIMIT:
            break
    return out


def touch_recent_model(model_spec: str, state_dir: Path | None = None) -> bool:
    """Promote `model_spec` to the front of the recent-models MRU list.

    Existing entries for the same spec are moved (not duplicated); the list
    is capped at `RECENT_MODELS_LIMIT`. Best-effort: returns `False` on I/O
    error so callers can degrade silently — recents are a nice-to-have, not
    a correctness requirement.

    Args:
        model_spec: The `provider:model` string just selected.
        state_dir: Override for the state directory (test hook).

    Returns:
        `True` on success, `False` on I/O error or invalid spec.
    """
    if not model_spec or ":" not in model_spec:
        return False
    existing = load_recent_models(state_dir)
    deduped = [entry for entry in existing if entry != model_spec]
    new_list = [model_spec, *deduped][:RECENT_MODELS_LIMIT]
    path = _recent_models_path(state_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"models": new_list}, f)
            Path(tmp_path).replace(path)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except OSError:
        logger.warning(
            "Could not update recent models cache at %s", path, exc_info=True
        )
        return False
    return True


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
