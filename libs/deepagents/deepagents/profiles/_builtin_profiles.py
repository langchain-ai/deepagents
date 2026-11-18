"""Bootstrap for built-in and third-party profile plugins.

Built-in provider and harness profiles are registered via explicit
module imports — not entry points — so a malformed or missing
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
import threading
import warnings
from importlib.metadata import EntryPoint, entry_points

from deepagents.profiles.harness import (
    _anthropic_haiku_4_5,
    _anthropic_opus_4_7,
    _anthropic_sonnet_4_6,
    _nvidia_nemotron_3_ultra,
    _openai_codex,
)
from deepagents.profiles.harness.harness_profiles import _HARNESS_PROFILES
from deepagents.profiles.provider import _nvidia, _openai, _openrouter
from deepagents.profiles.provider.provider_profiles import _PROVIDER_PROFILES

logger = logging.getLogger(__name__)


def _format_plugin_label(ep: EntryPoint) -> str:
    """Return a human-readable identifier for a plugin entry point.

    Includes the source distribution name when available so logs can point
    at the misbehaving package, not just the entry-point name (which can
    collide across distributions).
    """
    dist = getattr(ep, "dist", None)
    dist_name = getattr(dist, "name", None) if dist is not None else None
    if isinstance(dist_name, str) and dist_name:
        return f"{ep.name!r} (dist={dist_name!r})"
    return repr(ep.name)


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

_BOOTSTRAP_CONDITION = threading.Condition()
"""Coordinates first-time lazy bootstrap across threads.

One thread performs the bootstrap while concurrent threads wait for it
to finish. The condition is also used to permit same-thread re-entry:
plugin registration callables invoked *during* bootstrap often call the
public `register_*_profile` helpers, which must short-circuit rather
than deadlock or recurse.
"""

_loading_thread_id: int | None = None
"""Thread currently performing `_ensure_builtin_profiles_loaded`, if any.

Used to distinguish same-thread re-entry (short-circuit) from
cross-thread first access (wait for bootstrap completion).
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

    The function is invoked lazily from `register_*_profile` and
    `get_*_profile` entry points; importing `deepagents.profiles` itself
    does not trigger bootstrap. Same-thread re-entrant calls that occur
    *during* bootstrap (for example, plugin registration helpers calling
    the public `register_*_profile` APIs) short-circuit, while other
    threads block until bootstrap completes so they never observe a
    partially populated registry.
    """
    global _loaded, _BOOTSTRAP_HARNESS_KEYS, _loading_thread_id  # noqa: PLW0603
    thread_id = threading.get_ident()
    with _BOOTSTRAP_CONDITION:
        if _loaded:
            return
        if _loading_thread_id == thread_id:
            return
        while _loading_thread_id is not None:
            _BOOTSTRAP_CONDITION.wait()
            if _loaded:
                return
        _loading_thread_id = thread_id
    saved_provider_profiles = dict(_PROVIDER_PROFILES)
    saved_harness_profiles = dict(_HARNESS_PROFILES)
    saved_bootstrap_harness_keys = _BOOTSTRAP_HARNESS_KEYS
    try:
        _nvidia.register()
        _openai.register()
        _openrouter.register()
        _anthropic_opus_4_7.register()
        _anthropic_sonnet_4_6.register()
        _anthropic_haiku_4_5.register()
        _nvidia_nemotron_3_ultra.register()
        _openai_codex.register()
        plugins = _discover_profile_plugins()
        _invoke_profile_plugins(_PROVIDER_PROFILE_GROUP, plugins[_PROVIDER_PROFILE_GROUP])
        _invoke_profile_plugins(_HARNESS_PROFILE_GROUP, plugins[_HARNESS_PROFILE_GROUP])
        bootstrap_harness_keys = frozenset(_HARNESS_PROFILES)
    except Exception:
        logger.exception("Built-in profile bootstrap failed; restoring pre-bootstrap registry state.")
        # Restore in place so modules holding registry references keep seeing
        # the same dict objects after rollback.
        _PROVIDER_PROFILES.clear()
        _PROVIDER_PROFILES.update(saved_provider_profiles)
        _HARNESS_PROFILES.clear()
        _HARNESS_PROFILES.update(saved_harness_profiles)
        with _BOOTSTRAP_CONDITION:
            _BOOTSTRAP_HARNESS_KEYS = saved_bootstrap_harness_keys
            _loading_thread_id = None
            _BOOTSTRAP_CONDITION.notify_all()
        raise
    with _BOOTSTRAP_CONDITION:
        _BOOTSTRAP_HARNESS_KEYS = bootstrap_harness_keys
        _loaded = True
        _loading_thread_id = None
        _BOOTSTRAP_CONDITION.notify_all()


def _discover_profile_plugins() -> dict[str, list[EntryPoint]]:
    """Return the entry points of both profile groups from a single scan.

    Enumerating entry points reads `entry_points.txt` from every installed
    distribution, and `entry_points(group=...)` repeats that whole scan for
    each call. This runs on the first `create_deep_agent()` of a process, so
    both groups are partitioned out of one enumeration instead of two.

    If enumeration itself raises (e.g. malformed `dist-info` metadata), the
    failure is logged at `WARNING` — environment-level, not attributable to a
    specific plugin — and both groups come back empty.
    """
    groups = (_PROVIDER_PROFILE_GROUP, _HARNESS_PROFILE_GROUP)
    try:
        # `.select()` rather than iterating: on Python 3.11 a bare
        # `entry_points()` is the deprecated `SelectableGroups` mapping, whose
        # iteration yields group names. Selecting from it does not rescan.
        all_eps = entry_points()
        return {group: list(all_eps.select(group=group)) for group in groups}
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to enumerate entry points; no third-party profile plugins will load: {type(exc).__name__}: {exc}"
        logger.warning(msg, exc_info=True)
        warnings.warn(msg, stacklevel=3)
        return {group: [] for group in groups}


def _invoke_profile_plugins(group: str, eps: list[EntryPoint]) -> None:
    """Invoke every entry-point callable in `eps`, isolating failures.

    Plugin-level failures are each logged and skipped so one broken plugin
    cannot block its siblings:

    1. `ep.load()` raises (missing dependency, import-time error). Logged
        at `ERROR` with the source distribution name — a structural bug
        in the plugin that the plugin author needs to fix.
    2. The entry-point target resolves to something that is not callable.
        Logged at `ERROR` (declaring a non-callable as a registration hook
        is a plugin bug).
    3. The registration callable raises when invoked. Logged at `ERROR` —
        the plugin attempted to register but produced a `ValueError` /
        `TypeError` / etc. The plugin's registrations are silently absent
        if this is suppressed, so the elevated level helps users notice.

    Plugins are iterated in whatever order
    `importlib.metadata.entry_points` returns — callers MUST NOT rely on
    a specific ordering when two plugins register under the same key.
    Registration semantics are additive (`register_*_profile` merges on
    top), so later entries layer on earlier ones.

    Args:
        group: Entry-point group name, used in log messages (e.g.
            `deepagents.provider_profiles`).
        eps: The group's entry points, from `_discover_profile_plugins`.
    """
    for ep in eps:
        plugin_label = _format_plugin_label(ep)
        try:
            register = ep.load()
        except Exception as exc:
            msg = f"Skipping {group} plugin {plugin_label}: failed to load entry point {ep.value!r}: {type(exc).__name__}: {exc}"
            logger.exception(msg)
            warnings.warn(msg, stacklevel=2)
            continue
        if not callable(register):
            msg = f"Skipping {group} plugin {plugin_label}: entry point {ep.value!r} did not resolve to a callable."
            logger.error(msg)
            warnings.warn(msg, stacklevel=2)
            continue
        try:
            register()
        except Exception as exc:
            msg = f"Skipping {group} plugin {plugin_label}: registration callable {ep.value!r} raised: {type(exc).__name__}: {exc}"
            logger.exception(msg)
            warnings.warn(msg, stacklevel=2)
