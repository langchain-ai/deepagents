"""Provider profile registry for model- and provider-specific configuration.

Defines the `ProviderProfile` dataclass and the provider profile registry used
by `resolve_model` and `create_deep_agent` to apply provider- and
model-specific configuration (init kwargs, extra middleware, system prompt
patches, tool overrides).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import TYPE_CHECKING, Any

from packaging.version import Version

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain.agents.middleware.types import AgentMiddleware

OPENROUTER_MIN_VERSION = "0.2.0"
"""Minimum required version of `langchain-openrouter`.

Used by both the SDK (`resolve_model`) and the CLI (`config.py`) to enforce a
consistent version floor at runtime.
"""

_OPENROUTER_APP_URL = "https://github.com/langchain-ai/deepagents"
"""Default `app_url` (maps to `HTTP-Referer`) for OpenRouter attribution.

See https://openrouter.ai/docs/app-attribution for details.
"""

_OPENROUTER_APP_TITLE = "Deep Agents"
"""Default `app_title` (maps to `X-Title`) for OpenRouter attribution."""


# ---------------------------------------------------------------------------
# ProviderProfile — declarative model/provider customization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderProfile:
    """Declarative configuration for the Deep Agent harness.

    Applied based on the selected model or provider. Each field is optional —
    its default means "no change from baseline behavior". Profiles are looked
    up by `get_provider_profile` (exact model spec first, then provider prefix)
    and consumed by `resolve_model` (for `init_kwargs` / `pre_init`) and
    `create_deep_agent` (for everything else).

    Register profiles via `register_provider_profile`.
    """

    init_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra keyword arguments forwarded to `init_chat_model` when resolving
    a string model spec (e.g. `{"use_responses_api": True}` for OpenAI)."""

    pre_init: Callable[[str], None] | None = None
    """Optional callable invoked with the raw model spec string *before*
    `init_chat_model` runs.  Use for version checks or other preconditions
    (e.g. `check_openrouter_version`).  Must raise on failure."""

    init_kwargs_factory: Callable[[], dict[str, Any]] | None = None
    """Optional factory called at init time to produce dynamic kwargs that
    are merged *on top of* `init_kwargs`.  Use when values depend on runtime
    state like environment variables (e.g. OpenRouter attribution headers
    that defer to env var overrides)."""

    system_prompt_suffix: str | None = None
    """Text appended to the system prompt after `BASE_AGENT_PROMPT`.
    `None` means no suffix."""

    exclude_tools: frozenset[str] = frozenset()
    """Names of built-in tools to remove from the agent's tool set."""

    tool_description_overrides: dict[str, str] = field(default_factory=dict)
    """Per-tool description replacements, keyed by tool name."""

    extra_middleware: Sequence[AgentMiddleware] | Callable[[], Sequence[AgentMiddleware]] = ()
    """Provider-specific middleware appended to every middleware stack (main
    agent, general-purpose subagent, and per-subagent).

    May be a static sequence or a zero-arg factory that returns one (use a
    factory when the middleware instances should not be shared/reused across
    stacks).
    """


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

_PROVIDER_PROFILES: dict[str, ProviderProfile] = {}
"""Registry mapping profile keys to `ProviderProfile` instances.

Keys are either a full `provider:model` spec (for per-model overrides) or a
bare provider name (for provider-wide defaults).  Lookup order:
exact spec → provider prefix → empty default.
"""


def register_provider_profile(key: str, profile: ProviderProfile) -> None:
    """Register a `ProviderProfile` for a provider or specific model.

    Args:
        key: A provider name (e.g. `"openai"`) for provider-wide defaults,
            or a full `provider:model` spec (e.g. `"openai:o3-pro"`) for a
            per-model override.
        profile: The profile to register.
    """
    _PROVIDER_PROFILES[key] = profile


def get_provider_profile(spec: str) -> ProviderProfile:
    """Look up the `ProviderProfile` for a model spec.

    Resolution order:

    1. Exact match on `spec` (supports per-model overrides).
    2. Provider prefix (everything before the first `:`; for bare names
        without a colon, the full string is used).
    3. A default empty `ProviderProfile`.

    Args:
        spec: Model spec in `provider:model` format, or a bare model name.

    Returns:
        The matching `ProviderProfile`, or an empty default.
    """
    profile = _PROVIDER_PROFILES.get(spec)
    if profile is not None:
        return profile

    provider, _, _ = spec.partition(":")
    if provider:
        profile = _PROVIDER_PROFILES.get(provider)
        if profile is not None:
            return profile

    return ProviderProfile()


# ---------------------------------------------------------------------------
# OpenRouter helpers (used by the built-in openrouter profile)
# ---------------------------------------------------------------------------


def _openrouter_attribution_kwargs() -> dict[str, Any]:
    """Build OpenRouter attribution kwargs, deferring to env var overrides.

    `ChatOpenRouter` reads `OPENROUTER_APP_URL` and `OPENROUTER_APP_TITLE` via
    `from_env()` defaults. Explicit kwargs passed to the constructor take
    precedence over those env-var defaults, so we only inject our SDK defaults
    when the corresponding env var is **not** set — otherwise the user's env var
    would be overridden.

    Returns:
        Dictionary of attribution kwargs to spread into `init_chat_model`.
    """
    kwargs: dict[str, Any] = {}
    if not os.environ.get("OPENROUTER_APP_URL"):
        kwargs["app_url"] = _OPENROUTER_APP_URL
    if not os.environ.get("OPENROUTER_APP_TITLE"):
        kwargs["app_title"] = _OPENROUTER_APP_TITLE
    return kwargs


def check_openrouter_version() -> None:
    """Raise if the installed `langchain-openrouter` is below the minimum.

    If the package is not installed at all the check is skipped;
    `init_chat_model` will surface its own missing-dependency error downstream.

    Raises:
        ImportError: If the installed version is too old.
    """
    try:
        installed = pkg_version("langchain-openrouter")
    except PackageNotFoundError:
        return
    if Version(installed) < Version(OPENROUTER_MIN_VERSION):
        msg = (
            f"deepagents requires langchain-openrouter>={OPENROUTER_MIN_VERSION}, "
            f"but {installed} is installed. "
            f"Run: pip install 'langchain-openrouter>={OPENROUTER_MIN_VERSION}'"
        )
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# Built-in provider profiles
# ---------------------------------------------------------------------------


def _anthropic_extra_middleware() -> Sequence[AgentMiddleware]:
    """Build Anthropic prompt-caching middleware (deferred import).

    Returns:
        Single-element sequence containing an `AnthropicPromptCachingMiddleware`
        configured with `unsupported_model_behavior="ignore"`.
    """
    from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware  # noqa: PLC0415

    return [AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")]


register_provider_profile(
    "openai",
    ProviderProfile(init_kwargs={"use_responses_api": True}),
)

register_provider_profile(
    "openrouter",
    ProviderProfile(
        pre_init=lambda _spec: check_openrouter_version(),
        init_kwargs_factory=_openrouter_attribution_kwargs,
    ),
)

register_provider_profile(
    "anthropic",
    ProviderProfile(extra_middleware=_anthropic_extra_middleware),
)
