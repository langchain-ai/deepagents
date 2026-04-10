"""Harness profile registry for model- and provider-specific configuration.

!!! warning

    This is an internal API subject to change without deprecation. It is not
    intended for external use or consumption.

Defines the `HarnessProfile` dataclass and the harness profile registry used
by `resolve_model` and `create_deep_agent` to apply provider- and model-specific
configuration.
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

OPENROUTER_MIN_VERSION = "0.2.0"  # app attribution support added
"""Minimum required version of `langchain-openrouter`.

Used to enforce a consistent version floor at runtime.
"""

_OPENROUTER_APP_URL = "https://github.com/langchain-ai/deepagents"
"""Default `app_url` (maps to `HTTP-Referer`) for OpenRouter attribution.

See https://openrouter.ai/docs/app-attribution for details.
"""

_OPENROUTER_APP_TITLE = "Deep Agents"
"""Default `app_title` (maps to `X-Title`) for OpenRouter attribution."""


# ---------------------------------------------------------------------------
# HarnessProfile — declarative model/provider customization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarnessProfile:
    """Declarative configuration for the Deep Agent harness.

    Applied based on the selected model or provider. Each field is optional —
    its default means "no change from baseline behavior". Profiles are looked
    up by `get_harness_profile` (exact model spec first, then provider prefix)
    and consumed by `resolve_model` (for `init_kwargs` / `pre_init`) and
    `create_deep_agent` (for everything else).

    Register profiles via `register_harness_profile`.
    """

    init_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra keyword arguments forwarded to `init_chat_model` when resolving
    a string model spec (e.g. `{"use_responses_api": True}` for OpenAI)."""

    pre_init: Callable[[str], None] | None = None
    """Optional callable invoked with the raw model spec string *before*
    `init_chat_model` runs.

    Use for version checks or other preconditions
    (e.g. `check_openrouter_version`).

    Must raise on failure.
    """

    init_kwargs_factory: Callable[[], dict[str, Any]] | None = None
    """Optional factory called at init time to produce dynamic kwargs that
    are merged *on top of* `init_kwargs`.

    Use when values depend on runtime state like environment variables
    (e.g. OpenRouter attribution headers that defer to env var overrides).
    """

    base_system_prompt: str | None = None
    """When set, completely replaces `BASE_AGENT_PROMPT` as the base system
    prompt.  `None` (default) means use `BASE_AGENT_PROMPT` unchanged.

    If both `base_system_prompt` and `system_prompt_suffix` are set, the
    suffix is appended to this custom base.
    """

    system_prompt_suffix: str | None = None
    """Text appended to the base system prompt (either `BASE_AGENT_PROMPT`
    or the profile's `base_system_prompt` when set).

    `None` means no suffix.
    """

    tool_description_overrides: dict[str, str] = field(default_factory=dict)
    """Per-tool description replacements, keyed by tool name.

    Applied only where Deep Agents has a stable description hook: built-in
    filesystem tools, the `task` tool, and user-supplied `BaseTool` / dict
    tools. Plain callable tools are left unchanged.

    !!! warning

        Keys are matched by tool name string. If a built-in tool is renamed
        or removed, stale keys silently become no-ops with no error. Keep
        overrides minimal and verify against the current tool names.
    """

    excluded_tools: frozenset[str] = frozenset()
    """Tool names to remove from the tool set for this provider/model.

    Applies to both user-supplied tools (filtered before passing to
    `create_agent`) and middleware-injected tools (filtered via a
    `_ToolExclusionMiddleware` that strips them from `request.tools` before
    the model sees them).

    Merged via union when profiles are combined, so provider-level exclusions
    and model-level exclusions accumulate.
    """

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

_HARNESS_PROFILES: dict[str, HarnessProfile] = {}
"""Registry mapping profile keys to `HarnessProfile` instances.

Keys are either a full `provider:model` spec (for per-model overrides) or a
bare provider name (for provider-wide defaults).  Lookup order:
exact spec → provider prefix → empty default.
"""


def register_harness_profile(key: str, profile: HarnessProfile) -> None:
    """Register a `HarnessProfile` for a provider or specific model.

    Args:
        key: A provider name (e.g. `"openai"`) for provider-wide defaults,
            or a full `provider:model` spec (e.g. `"openai:o3-pro"`) for a
            per-model override.
        profile: The profile to register.
    """
    _HARNESS_PROFILES[key] = profile


def get_harness_profile(spec: str) -> HarnessProfile:
    """Look up the `HarnessProfile` for a model spec.

    Resolution order:

    1. Exact match on `spec` (supports per-model overrides).
    2. Provider prefix (everything before the first `:`; for bare names
        without a colon, the full string is used).
    3. A default empty `HarnessProfile`.

    When both an exact-model profile and a provider-level profile exist, they
    are merged: the provider profile serves as the base and the exact-model
    profile is layered on top. This ensures per-model tweaks inherit provider
    defaults (e.g. `use_responses_api` for OpenAI, prompt-caching middleware
    for Anthropic) instead of silently dropping them.

    Args:
        spec: Model spec in `provider:model` format, or a bare model name.

    Returns:
        The matching `HarnessProfile`, or an empty default.
    """
    exact = _HARNESS_PROFILES.get(spec)

    provider, sep, _ = spec.partition(":")
    base = _HARNESS_PROFILES.get(provider) if sep else None

    if exact is not None and base is not None:
        return _merge_profiles(base, exact)
    if exact is not None:
        return exact
    if base is not None:
        return base
    return HarnessProfile()


def _resolve_middleware_seq(
    middleware: Sequence[AgentMiddleware] | Callable[[], Sequence[AgentMiddleware]],
) -> Sequence[AgentMiddleware]:
    """Resolve middleware to a concrete sequence, calling factory if needed."""
    if callable(middleware):
        return middleware()  # ty: ignore[call-top-callable]  # Callable & Sequence union confuses ty
    return middleware


def _merge_middleware(
    base_mw: Sequence[AgentMiddleware] | Callable[[], Sequence[AgentMiddleware]],
    over_mw: Sequence[AgentMiddleware] | Callable[[], Sequence[AgentMiddleware]],
) -> Sequence[AgentMiddleware] | Callable[[], Sequence[AgentMiddleware]]:
    """Merge two middleware sequences by type.

    If the override supplies a middleware whose type already exists in the base,
    the override instance replaces it in-place (preserving position). Novel
    override entries are appended.

    Example: a provider profile registers `CachingMiddleware(ttl=60)` and a
    model-specific profile registers `CachingMiddleware(ttl=0)`. The merged
    result contains a single `CachingMiddleware(ttl=0)` — the model-specific
    instance replaces the provider-level one rather than duplicating it.

    Args:
        base_mw: Base middleware (lower priority).
        over_mw: Override middleware (higher priority).

    Returns:
        Merged middleware sequence or factory.
    """
    if not base_mw or not over_mw:
        return over_mw or base_mw

    def factory() -> Sequence[AgentMiddleware]:
        base_seq = _resolve_middleware_seq(base_mw)
        over_seq = _resolve_middleware_seq(over_mw)
        over_by_type: dict[type, AgentMiddleware] = {type(m): m for m in over_seq}
        merged: list[AgentMiddleware] = []
        seen: set[type] = set()
        for m in base_seq:
            mtype = type(m)
            if mtype in over_by_type:
                merged.append(over_by_type[mtype])
                seen.add(mtype)
            else:
                merged.append(m)
        merged.extend(m for m in over_seq if type(m) not in seen)
        return merged

    return factory


def _merge_profiles(base: HarnessProfile, override: HarnessProfile) -> HarnessProfile:
    """Merge two profiles, layering `override` on top of `base`.

    Dict fields are merged (override wins per-key). Callables (`pre_init`,
    `init_kwargs_factory`) are chained. Middleware sequences are merged by
    type: if the override supplies a middleware whose type already exists in
    the base, it replaces the base entry (preserving position); novel override
    entries are appended. Scalar fields (e.g. prompts) use the override
    value when set, otherwise fall back to the base.

    Args:
        base: Provider-level profile (lower priority).
        override: Exact-model profile (higher priority).

    Returns:
        A new merged `HarnessProfile`.
    """
    # Chain pre_init callables
    if base.pre_init is not None and override.pre_init is not None:
        base_pre = base.pre_init
        over_pre = override.pre_init

        def chained_pre_init(spec: str) -> None:
            base_pre(spec)
            over_pre(spec)

        pre_init: Callable[[str], None] | None = chained_pre_init
    else:
        pre_init = override.pre_init or base.pre_init

    # Chain init_kwargs_factory callables
    if base.init_kwargs_factory is not None and override.init_kwargs_factory is not None:
        base_fac = base.init_kwargs_factory
        over_fac = override.init_kwargs_factory

        def chained_factory() -> dict[str, Any]:
            result = base_fac()
            result.update(over_fac())
            return result

        init_kwargs_factory: Callable[[], dict[str, Any]] | None = chained_factory
    else:
        init_kwargs_factory = override.init_kwargs_factory or base.init_kwargs_factory

    return HarnessProfile(
        init_kwargs={**base.init_kwargs, **override.init_kwargs},
        pre_init=pre_init,
        init_kwargs_factory=init_kwargs_factory,
        base_system_prompt=(override.base_system_prompt if override.base_system_prompt is not None else base.base_system_prompt),
        system_prompt_suffix=(override.system_prompt_suffix if override.system_prompt_suffix is not None else base.system_prompt_suffix),
        tool_description_overrides={
            **base.tool_description_overrides,
            **override.tool_description_overrides,
        },
        excluded_tools=base.excluded_tools | override.excluded_tools,
        extra_middleware=_merge_middleware(base.extra_middleware, override.extra_middleware),
    )


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
# Built-in harness profiles
# ---------------------------------------------------------------------------


register_harness_profile(
    "openai",
    HarnessProfile(init_kwargs={"use_responses_api": True}),
)

register_harness_profile(
    "openrouter",
    HarnessProfile(
        pre_init=lambda _spec: check_openrouter_version(),
        init_kwargs_factory=_openrouter_attribution_kwargs,
    ),
)
