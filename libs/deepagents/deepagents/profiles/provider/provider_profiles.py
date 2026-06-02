"""Beta APIs for configuring model-construction behavior.

!!! beta

    `deepagents.profiles` exposes beta APIs that may receive minor changes in
    future releases. Refer to the [versioning documentation](https://docs.langchain.com/oss/python/versioning)
    for more details.

Provider profiles declare how Deep Agents should construct a chat model for a
given provider or specific model spec. The registry is consumed by
`resolve_model` and is the extension point for controlling `init_chat_model`
kwargs, running pre-initialization side effects, and deriving kwargs from
runtime state (e.g. environment variables).
"""
# Built-in profiles are registered lazily on first registry access for
# `"openai"` (enables the Responses API by default) and `"openrouter"`
# (enforces a minimum version and injects app-attribution headers).
# Additional providers or per-model overrides can be registered with
# `register_provider_profile`.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from deepagents.profiles._keys import validate_profile_key

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderProfile:
    """Declarative configuration for constructing a chat model.

    !!! beta

        `deepagents.profiles` exposes beta APIs that may receive minor changes in
        future releases. Refer to the [versioning documentation](https://docs.langchain.com/oss/python/versioning)
        for more details.

    A `ProviderProfile` describes provider- or model-specific kwargs,
    pre-initialization side effects, and runtime-derived kwargs that should be
    applied when `resolve_model` turns a string spec (e.g. `"openai:gpt-5.4"`)
    into a `BaseChatModel`. Profiles are registered via
    `register_provider_profile` under a provider key (`"openai"`) or a full
    `provider:model` key (`"openai:gpt-5.4"`).

    Profiles handle model-construction concerns only — things that shape how
    `init_chat_model` assembles the client instance. Typical examples:
    constructor kwargs like `use_responses_api`, `temperature`, `max_tokens`,
    or `base_url`; provider-specific headers such as OpenRouter app
    attribution; pre-construction checks like minimum-version enforcement;
    and env-var-aware defaults.

    Runtime and harness behavior — system-prompt assembly, tool description
    overrides, excluded tools, extra middleware, general-purpose subagent
    configuration — belongs in `HarnessProfile`, the separate harness
    profile system consumed by `create_deep_agent`, not here.

    Example:
        Set a default temperature for a hypothetical provider:

        ```python
        from deepagents import ProviderProfile, register_provider_profile

        register_provider_profile(
            "my_provider",
            ProviderProfile(init_kwargs={"temperature": 0.7}),
        )
        ```
    """

    init_kwargs: Mapping[str, Any] = field(default_factory=dict)
    """Static keyword arguments forwarded to `init_chat_model`.

    Once a profile is constructed, its kwargs can be read but not
    rewritten — for example, `profile.init_kwargs["temperature"] = 2.0`
    raises `TypeError`. The registry stores its own defensive copy, so
    mutating the dict you passed into the constructor after the fact won't
    affect the registered profile either. To change a registered profile's
    kwargs, re-register (which merges on top) or construct a new profile.

    When both `init_kwargs` and `init_kwargs_factory` are set on the same
    profile, the factory's output overrides `init_kwargs` on key collision.
    """

    pre_init: Callable[[str], None] | None = None
    """Optional callable invoked with the raw model spec before initialization.

    Runs before `init_kwargs_factory` is invoked and before `init_chat_model`
    is called; if it raises, the factory does not run and no model is
    constructed. Use for side-effectful checks that must run before
    `init_chat_model` (e.g. minimum-version enforcement). Raise to abort
    model construction.
    """

    init_kwargs_factory: Callable[[], dict[str, Any]] | None = None
    """Optional factory producing dynamic init kwargs at resolution time.

    Use when values depend on runtime state such as environment variables.

    Within a single profile, whenever the factory and `init_kwargs` set the
    same key, the factory's value wins. For example:

    ```python
    ProviderProfile(
        init_kwargs={"temperature": 0, "timeout": 30},
        init_kwargs_factory=lambda: {"temperature": 0.7, "base_url": os.environ["BASE_URL"]},
    )
    ```

    At resolution, this forwards `temperature=0.7` (factory wins),
    `timeout=30` (only in static), and `base_url=<env value>` (only from
    factory) to `init_chat_model`.

    When merging profiles, if both the base and override profiles define a
    factory, both run at every resolution — base first, then override — and
    their outputs merge with the override's values winning on any shared
    keys. A worked example: the built-in OpenRouter factory always sets
    `app_url` and `app_title`; layer a user profile whose factory reads a
    per-tenant `OPENROUTER_APP_TITLE_TENANT` env var and sets `app_title`
    from it. Every resolution runs both factories; the user's `app_title`
    replaces the built-in, while the built-in's `app_url` is preserved.
    """

    def __post_init__(self) -> None:
        """Freeze `init_kwargs` to prevent post-construction mutation.

        `@dataclass(frozen=True)` only prevents rebinding attributes (e.g.,
        `profile.init_kwargs = {...}`). It does not prevent mutating the
        contents of a mutable value, so without this hook a caller could
        silently corrupt a registered profile:

        ```python
        shared = {"temperature": 0}
        profile = ProviderProfile(init_kwargs=shared)
        register_provider_profile("openai", profile)

        # Later, somewhere else in the program:
        shared["temperature"] = 2.0  # mutates caller's dict
        profile.init_kwargs["timeout"] = 5  # mutates the registered profile
        ```

        Both of those were ways to change what `resolve_model` forwarded to
        `init_chat_model` long after registration, with no audit trail.
        This method defensively copies `init_kwargs` into a fresh dict and
        wraps it in `MappingProxyType` — a read-only view — so both
        scenarios become errors: the first because the registry holds its
        own copy independent of `shared`, and the second because item
        assignment on a `MappingProxyType` raises `TypeError`.
        """
        if not isinstance(self.init_kwargs, MappingProxyType):
            object.__setattr__(
                self,
                "init_kwargs",
                MappingProxyType(dict(self.init_kwargs)),
            )


_PROVIDER_PROFILES: dict[str, ProviderProfile] = {}
"""Internal registry mapping provider-profile keys to `ProviderProfile` instances."""


def _ensure_provider_profiles_loaded() -> None:
    """Ensure the lazy built-in/profile-plugin bootstrap has completed."""
    from deepagents.profiles._builtin_profiles import _ensure_builtin_profiles_loaded  # noqa: PLC0415

    _ensure_builtin_profiles_loaded()


def _register_provider_profile_impl(key: str, profile: ProviderProfile) -> None:
    """Core implementation behind `register_provider_profile`.

    Callers are responsible for any lazy-bootstrap coordination.
    """
    validate_profile_key(key)
    existing = _PROVIDER_PROFILES.get(key)
    if existing is not None:
        logger.info(
            "Merging ProviderProfile under %r on top of existing registration; "
            "init_kwargs and factory outputs merge with the new profile winning "
            "on shared keys, and pre_init callables chain.",
            key,
        )
        profile = _merge_provider_profiles(existing, profile)
    _PROVIDER_PROFILES[key] = profile


def register_provider_profile(key: str, profile: ProviderProfile) -> None:
    """Register a `ProviderProfile` for a provider or specific model.

    !!! beta

        `deepagents.profiles` exposes beta APIs that may receive minor changes in
        future releases. Refer to the [versioning documentation](https://docs.langchain.com/oss/python/versioning)
        for more details.

    Registrations are **additive**: if a profile is already registered under
    `key` (including a built-in profile loaded during lazy bootstrap), the new
    profile is merged on top rather than replacing it. The incoming profile's
    fields win on conflicts; unspecified fields inherit from the existing
    profile.
    `pre_init` callables chain (existing runs first), and `init_kwargs_factory`
    callables chain — both factories are invoked at every resolution (base
    first, then override) and their outputs merge with the override's values
    winning on shared keys.

    To layer additional kwargs onto a built-in profile, register under the
    same provider key. To override a built-in default (e.g. disable the
    OpenAI Responses API), set the conflicting key explicitly:

    ```python
    from deepagents import ProviderProfile, register_provider_profile

    # Adds temperature alongside the built-in `use_responses_api=True`.
    register_provider_profile("openai", ProviderProfile(init_kwargs={"temperature": 0}))

    # Explicitly disables Responses API for OpenAI. (This will break usage,
    # this example is purely illustrative.)
    register_provider_profile(
        "openai",
        ProviderProfile(init_kwargs={"use_responses_api": False}),
    )
    ```

    Args:
        key: Either a provider name (no colon) for provider-wide defaults,
            or a full `provider:model` spec for a per-model override. Valid
            shapes:

            - `"openai"` — provider-wide
            - `"openai:gpt-5.4"` — specific model

        profile: The provider profile to register.

    Raises:
        ValueError: If `key` is empty, contains more than one `:`, or has an
            empty provider/model half.
    """
    _ensure_provider_profiles_loaded()
    _register_provider_profile_impl(key, profile)


def get_provider_profile(spec: str) -> ProviderProfile | None:
    """Look up the `ProviderProfile` for a model spec.

    !!! beta

        `deepagents.profiles` exposes beta APIs that may receive minor changes in
        future releases. Refer to the [versioning documentation](https://docs.langchain.com/oss/python/versioning)
        for more details.

    Resolution order:

    1. Exact match on `spec`.
    2. Provider prefix (everything before the first `:`), when `spec`
        contains a colon and both halves are non-empty.
    3. `None` when neither matches.

    When both an exact-model profile and a provider-level profile exist, they
    are merged via `_merge_provider_profiles` with the exact-model entry
    overriding the provider-level entry on conflicts.

    When only the provider-level profile matches, a debug breadcrumb is
    emitted so registrations layered on an exact key can be traced when they
    don't apply (e.g. typo'd specs falling through to the provider default).

    Malformed specs (empty string, more than one `:`, or a `:` with an empty
    provider/model half) return `None` without consulting the registry. This
    prevents a spec like `"openai:"` from silently matching the provider-wide
    `"openai"` registration.

    !!! note "Prefer `apply_provider_profile` for model construction"

        This function is intended for *inspection* (tooling, conditional logic
        on `pre_init` presence). To actually build a model, reach for
        `apply_provider_profile` — it composes lookup, `pre_init` invocation,
        and kwargs merging into a single call.

    Args:
        spec: Model spec in `provider:model` format, or a bare provider/model
            identifier.

    Returns:
        The matching `ProviderProfile`, or `None` when no registered profile matches.
    """
    if not spec or spec.count(":") > 1:
        return None

    provider, sep, model = spec.partition(":")
    if sep and (not provider or not model):
        return None

    _ensure_provider_profiles_loaded()
    exact = _PROVIDER_PROFILES.get(spec)
    base = _PROVIDER_PROFILES.get(provider) if sep else None

    if exact is not None and base is not None:
        return _merge_provider_profiles(base, exact)
    if exact is not None:
        return exact
    if base is not None:
        logger.debug(
            "No exact ProviderProfile for %r; using provider %r profile.",
            spec,
            provider,
        )
        return base
    return None


def apply_provider_profile(
    spec: str,
    kwargs: Mapping[str, Any] | None = None,
    *,
    run_pre_init: bool = True,
) -> dict[str, Any]:
    """Compose `init_chat_model` kwargs from the registered profile for `spec`.

    !!! beta

        `deepagents.profiles` exposes beta APIs that may receive minor changes in
        future releases. Refer to the [versioning documentation](https://docs.langchain.com/oss/python/versioning)
        for more details.

    This is the recommended entry point for honoring `ProviderProfile`
    registrations when building a chat model — it composes lookup, `pre_init`
    invocation, and kwargs merging into a single call. `get_provider_profile`
    only returns the registered object; pair it with this helper (or call
    this helper directly) any time you intend to actually instantiate a
    model. Used by `resolve_model` and intended for any harness that layers
    config-file values on top of SDK defaults.

    Looks up the profile via `get_provider_profile`, runs its `pre_init` hook
    (unless suppressed), and returns a fresh dict combining `init_kwargs`,
    `init_kwargs_factory()` output, and `kwargs`. Caller-supplied `kwargs`
    take highest precedence — profile defaults sit beneath them — so
    user-provided values from config files or explicit overrides are never
    silently replaced.

    When no profile is registered for `spec`, returns a copy of `kwargs`
    unchanged. This keeps the helper safe to call unconditionally.

    Args:
        spec: Model spec in `provider:model` format, or a bare provider/model
            identifier.

            Same shape accepted by `get_provider_profile`.
        kwargs: Caller-supplied kwargs that override profile defaults on
            shared keys.

            Defaults to an empty mapping.
        run_pre_init: When `True` (default), invokes the profile's `pre_init`
            hook before composing kwargs.

            Set `False` to inspect the merged kwargs without firing
            side effects (e.g. dry-run, validation).

    Returns:
        A fresh `dict` ready to spread into `init_chat_model`.
    """
    base: dict[str, Any] = dict(kwargs) if kwargs else {}
    profile = get_provider_profile(spec)
    if profile is None:
        return base

    if run_pre_init and profile.pre_init is not None:
        profile.pre_init(spec)

    merged: dict[str, Any] = dict(profile.init_kwargs)
    if profile.init_kwargs_factory is not None:
        merged.update(profile.init_kwargs_factory())
    merged.update(base)
    return merged


def _merge_provider_profiles(base: ProviderProfile, override: ProviderProfile) -> ProviderProfile:
    """Merge two provider profiles, layering `override` on top of `base`.

    `init_kwargs` dicts are merged with override winning per key. For example,
    `{"a": 1, "shared": "base"}` merged with `{"b": 2, "shared": "over"}`
    yields `{"a": 1, "b": 2, "shared": "over"}`.

    `pre_init` callables chain: both run in order (base first, then override)
    when both are set. Exceptions from either propagate and halt the chain.

    `init_kwargs_factory` callables chain: both are invoked at resolution time
    and their outputs merged with override winning per key. When only one
    profile sets a field, the merged profile uses that side directly.

    Args:
        base: Lower-priority profile, typically from the provider.
        override: Higher-priority profile, typically from the exact model.

    Returns:
        A merged `ProviderProfile`.
    """
    if base.pre_init is not None and override.pre_init is not None:
        base_pre = base.pre_init
        over_pre = override.pre_init

        def chained_pre_init(spec: str) -> None:
            try:
                base_pre(spec)
            except Exception:
                logger.exception(
                    "Base pre_init in chained ProviderProfile raised for spec %r; override pre_init will not run.",
                    spec,
                )
                raise
            try:
                over_pre(spec)
            except Exception:
                logger.exception(
                    "Override pre_init in chained ProviderProfile raised for spec %r.",
                    spec,
                )
                raise

        pre_init: Callable[[str], None] | None = chained_pre_init
    else:
        pre_init = override.pre_init or base.pre_init

    if base.init_kwargs_factory is not None and override.init_kwargs_factory is not None:
        base_factory = base.init_kwargs_factory
        override_factory = override.init_kwargs_factory

        def chained_factory() -> dict[str, Any]:
            try:
                result = {**base_factory()}
            except Exception:
                logger.exception("Base init_kwargs_factory in chained ProviderProfile raised; override factory will not run.")
                raise
            try:
                result.update(override_factory())
            except Exception:
                logger.exception("Override init_kwargs_factory in chained ProviderProfile raised.")
                raise
            return result

        init_kwargs_factory: Callable[[], dict[str, Any]] | None = chained_factory
    else:
        init_kwargs_factory = override.init_kwargs_factory or base.init_kwargs_factory

    return ProviderProfile(
        init_kwargs={**base.init_kwargs, **override.init_kwargs},
        pre_init=pre_init,
        init_kwargs_factory=init_kwargs_factory,
    )
