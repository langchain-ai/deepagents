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
# Built-in profiles are registered at import time for `"openai"` (enables the
# Responses API by default) and `"openrouter"` (enforces a minimum version and
# injects app-attribution headers). Additional providers or per-model overrides
# can be registered with `register_provider_profile`.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class ProviderProfile:
    """Declarative configuration for constructing a chat model.

    !!! beta

        `ProviderProfile` is a beta API. It is safe for production use, but
        may receive minor changes in future releases.

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
        Register custom attribution headers for a hypothetical provider:

        ```python
        from deepagents import ProviderProfile, register_provider_profile

        register_provider_profile(
            "my_provider",
            ProviderProfile(init_kwargs={"temperature": 0.7}),
        )
        ```
    """

    init_kwargs: dict[str, Any] = field(default_factory=dict)
    """Static keyword arguments forwarded to `init_chat_model`.

    When both `init_kwargs` and `init_kwargs_factory` are set on the same
    profile, the factory's output overrides `init_kwargs` on key collision.
    """

    pre_init: Callable[[str], None] | None = None
    """Optional callable invoked with the raw model spec before initialization.

    Use for side-effectful checks that must run before `init_chat_model` (e.g.
    minimum-version enforcement). Raise to abort model construction.
    """

    init_kwargs_factory: Callable[[], dict[str, Any]] | None = None
    """Optional factory producing dynamic init kwargs at resolution time.

    Use when values depend on runtime state such as environment variables.

    Factory output overrides static `init_kwargs` on any key collision within
    the same profile.
    """
    # (e.g. the built-in OpenRouter attribution headers that defer to
    # `OPENROUTER_APP_URL` / `OPENROUTER_APP_TITLE`).


_PROVIDER_PROFILES: dict[str, ProviderProfile] = {}
"""Internal registry mapping provider-profile keys to `ProviderProfile` instances."""


def register_provider_profile(key: str, profile: ProviderProfile) -> None:
    """Register a `ProviderProfile` for a provider or specific model.

    !!! beta

        `register_provider_profile` is a beta API. It is safe for production
        use, but may receive minor changes in future releases.

    Registrations are **additive**: if a profile is already registered under
    `key` (including a built-in profile loaded at import time), the new profile
    is merged on top rather than replacing it. The incoming profile's fields
    win on conflicts; unspecified fields inherit from the existing profile.
    `pre_init` callables chain (existing runs first), and `init_kwargs_factory`
    callables chain with the incoming factory's output overriding the
    existing factory's output per key.

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
        key: A provider name like `"openai"` for provider-wide defaults, or a
            full `provider:model` spec like `"openai:gpt-5.4"` for a per-model
            override.
        profile: The provider profile to register.
    """
    existing = _PROVIDER_PROFILES.get(key)
    if existing is not None:
        profile = _merge_provider_profiles(existing, profile)
    _PROVIDER_PROFILES[key] = profile


def _get_provider_profile(spec: str) -> ProviderProfile:
    """Look up the `ProviderProfile` for a model spec.

    Resolution order:

    1. Exact match on `spec`.
    2. Provider prefix (everything before the first `:`), when `spec` contains a colon.
    3. A default empty `ProviderProfile`.

    When both an exact-model profile and a provider-level profile exist, they
    are merged via `_merge_provider_profiles` with the exact-model entry
    overriding the provider-level entry on conflicts.

    Args:
        spec: Model spec in `provider:model` format, or a bare provider/model
            identifier.

    Returns:
        The matching `ProviderProfile`, or an empty default when no registered
        profile matches.
    """
    exact = _PROVIDER_PROFILES.get(spec)

    provider, sep, _ = spec.partition(":")
    base = _PROVIDER_PROFILES.get(provider) if sep else None

    if exact is not None and base is not None:
        return _merge_provider_profiles(base, exact)
    if exact is not None:
        return exact
    if base is not None:
        return base
    return ProviderProfile()


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
            base_pre(spec)
            over_pre(spec)

        pre_init: Callable[[str], None] | None = chained_pre_init
    else:
        pre_init = override.pre_init or base.pre_init

    if base.init_kwargs_factory is not None and override.init_kwargs_factory is not None:
        base_factory = base.init_kwargs_factory
        override_factory = override.init_kwargs_factory

        def chained_factory() -> dict[str, Any]:
            result = {**base_factory()}
            result.update(override_factory())
            return result

        init_kwargs_factory: Callable[[], dict[str, Any]] | None = chained_factory
    else:
        init_kwargs_factory = override.init_kwargs_factory or base.init_kwargs_factory

    return ProviderProfile(
        init_kwargs={**base.init_kwargs, **override.init_kwargs},
        pre_init=pre_init,
        init_kwargs_factory=init_kwargs_factory,
    )
