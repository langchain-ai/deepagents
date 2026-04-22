"""Provider profile registry for model-construction configuration.

!!! warning

    This is an internal API subject to change without deprecation. It is not
    intended for external use or consumption.

Defines the `_ProviderProfile` dataclass and the provider profile registry used
by `resolve_model` to apply provider- and model-specific initialization
behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class _ProviderProfile:
    """Declarative configuration for model construction.

    Provider profiles are resolved from model specs and consumed by
    `resolve_model`. They handle model-construction concerns only and are kept
    separate from harness/runtime behavior, which belongs in `_HarnessProfile`.
    """

    init_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra keyword arguments forwarded to `init_chat_model`.

    When both `init_kwargs` and `init_kwargs_factory` are set on the same
    profile, the factory's output overrides `init_kwargs` on key collision.
    """

    pre_init: Callable[[str], None] | None = None
    """Optional callable invoked with the raw model spec before initialization."""

    init_kwargs_factory: Callable[[], dict[str, Any]] | None = None
    """Optional factory producing dynamic init kwargs at resolution time.

    Use when values depend on runtime state like environment variables
    (e.g. OpenRouter attribution headers that defer to env var overrides).

    Factory output overrides static `init_kwargs` on any key collision within
    the same profile.
    """


_PROVIDER_PROFILES: dict[str, _ProviderProfile] = {}
"""Registry mapping provider-profile keys to `_ProviderProfile` instances."""


def _register_provider_profile(key: str, profile: _ProviderProfile) -> None:
    """Register a `_ProviderProfile` for a provider or specific model."""
    _PROVIDER_PROFILES[key] = profile


def _get_provider_profile(spec: str) -> _ProviderProfile:
    """Look up the `_ProviderProfile` for a model spec.

    Resolution order:

    1. Exact match on `spec`.
    2. Provider prefix (everything before the first `:`), when `spec` contains a colon.
    3. A default empty `_ProviderProfile`.

    When both an exact-model profile and a provider-level profile exist, they
    are merged via `_merge_provider_profiles` with the exact-model entry
    overriding the provider-level entry on conflicts.

    Args:
        spec: Model spec in `provider:model` format, or a bare provider/model
            identifier.

    Returns:
        The matching `_ProviderProfile`, or an empty default when no registered
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
    return _ProviderProfile()


def _merge_provider_profiles(base: _ProviderProfile, override: _ProviderProfile) -> _ProviderProfile:
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
        A merged `_ProviderProfile`.
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

    return _ProviderProfile(
        init_kwargs={**base.init_kwargs, **override.init_kwargs},
        pre_init=pre_init,
        init_kwargs_factory=init_kwargs_factory,
    )
