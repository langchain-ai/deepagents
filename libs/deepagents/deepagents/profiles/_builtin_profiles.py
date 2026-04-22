"""Bootstrap for built-in and third-party profile plugins.

Built-in provider profiles (OpenAI, OpenRouter) are registered via
explicit module imports — not entry points — so a malformed or missing
`dist-info` in the environment cannot silently disable the SDK's own
defaults. Third parties plug in via `importlib.metadata` entry points
under two groups:

- `deepagents.provider_profiles` — plugins that call
    `register_provider_profile(...)` to declare provider- or model-keyed
    `ProviderProfile` entries.
- `deepagents.harness_profiles` — plugins that call
    `register_harness_profile(...)` to declare provider- or model-keyed
    `HarnessProfile` entries.

Each entry point resolves to a zero-arg callable whose sole job is to
perform the registrations. Built-ins load first, so third-party plugins
registering under the same key layer on top via the additive merge
semantics of `register_*_profile`.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from deepagents.profiles.harness_profiles import _HARNESS_PROFILES
from deepagents.profiles.provider import _openai, _openrouter

logger = logging.getLogger(__name__)

_PROVIDER_PROFILE_GROUP = "deepagents.provider_profiles"
"""Entry-point group name for third-party `ProviderProfile` plugins."""

_HARNESS_PROFILE_GROUP = "deepagents.harness_profiles"
"""Entry-point group name for third-party `HarnessProfile` plugins."""

_BOOTSTRAP_HARNESS_KEYS: frozenset[str] = frozenset()
"""Snapshot of harness-profile keys registered during bootstrap.

Populated once by `_ensure_builtin_profiles_loaded`. Captures every
harness key in the registry immediately after the built-in and
entry-point phases complete — so both Deep Agents' own defaults and any
third-party harness plugins the user has installed are treated
uniformly as "bootstrap-provided." `_has_any_harness_profile` subtracts
this set from the live registry to distinguish those defaults from
profiles the user registers explicitly after import.
"""

_loaded: bool = False
"""Guards `_ensure_builtin_profiles_loaded` against re-running.

Registration callables are not guaranteed idempotent — repeat
invocations would chain `pre_init` hooks or re-merge kwargs with
themselves. The flag ensures the bootstrap runs exactly once per
interpreter, even if the function is called directly from tests or a
reload scenario.
"""


def _ensure_builtin_profiles_loaded() -> None:
    """Register built-in profiles and discover third-party plugins.

    Runs two phases, both idempotent:

    1. Call the built-in provider `register` functions directly.
        Any exception propagates — a broken built-in is a deepagents
        bug and must surface loudly, not degrade silently.
    2. Iterate `importlib.metadata` entry points in the
        `deepagents.provider_profiles` and `deepagents.harness_profiles`
        groups. Third-party failures are logged at `WARNING` and skipped
        so one misbehaving distribution cannot prevent `deepagents.profiles`
        from importing.

    Built-ins run first, so third-party plugins registering under the
    same key layer on top via additive merge semantics in
    `register_*_profile`.

    After both phases complete, snapshots the harness registry so
    downstream callers can distinguish bootstrap-registered profiles
    from profiles registered later via user code.
    """
    global _loaded, _BOOTSTRAP_HARNESS_KEYS  # noqa: PLW0603
    if _loaded:
        return
    _openai.register()
    _openrouter.register()
    _invoke_profile_plugins(_PROVIDER_PROFILE_GROUP)
    _invoke_profile_plugins(_HARNESS_PROFILE_GROUP)
    _BOOTSTRAP_HARNESS_KEYS = frozenset(_HARNESS_PROFILES)
    _loaded = True


def _invoke_profile_plugins(group: str) -> None:
    """Invoke every entry-point callable in `group`, isolating failures.

    Any of the following conditions is logged at `WARNING` and skipped
    without affecting sibling plugins:

    1. `entry_points(group=...)` itself raises (e.g. malformed
        `dist-info` metadata). The whole group is skipped.
    2. `ep.load()` raises (missing dependency, import-time error).
    3. The entry-point target resolves to something that is not callable.
    4. The registration callable raises when invoked.

    Plugins are iterated in whatever order
    `importlib.metadata.entry_points` returns — callers MUST NOT rely on
    a specific ordering when two plugins register under the same key.
    Registration semantics are additive (`register_*_profile` merges on
    top), so later entries layer on earlier ones.

    Args:
        group: Entry-point group name to iterate (e.g.
            `deepagents.provider_profiles`).
    """
    try:
        eps = entry_points(group=group)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Failed to enumerate %s entry points; no third-party plugins in this group will load.",
            group,
            exc_info=True,
        )
        return
    for ep in eps:
        try:
            register = ep.load()
        except Exception:  # noqa: BLE001
            logger.warning(
                "Skipping %s plugin %r: failed to load entry point %r.",
                group,
                ep.name,
                ep.value,
                exc_info=True,
            )
            continue
        if not callable(register):
            logger.warning(
                "Skipping %s plugin %r: entry point %r did not resolve to a callable.",
                group,
                ep.name,
                ep.value,
            )
            continue
        try:
            register()
        except Exception:  # noqa: BLE001
            logger.warning(
                "Skipping %s plugin %r: registration callable %r raised.",
                group,
                ep.name,
                ep.value,
                exc_info=True,
            )
