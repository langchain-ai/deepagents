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
import threading
import warnings
from importlib.metadata import entry_points

from deepagents.profiles.harness import (
    _anthropic_haiku_4_5,
    _anthropic_opus_4_7,
    _anthropic_sonnet_4_6,
    _openai_codex,
)
from deepagents.profiles.harness_profiles import _HARNESS_PROFILES
from deepagents.profiles.provider import _openai, _openrouter
from deepagents.profiles.provider.provider_profiles import _PROVIDER_PROFILES

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
        _openai.register()
        _openrouter.register()
        _anthropic_opus_4_7.register()
        _anthropic_sonnet_4_6.register()
        _anthropic_haiku_4_5.register()
        _openai_codex.register()
        _invoke_profile_plugins(_PROVIDER_PROFILE_GROUP)
        _invoke_profile_plugins(_HARNESS_PROFILE_GROUP)
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
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to enumerate {group} entry points; no third-party plugins in this group will load: {type(exc).__name__}: {exc}"
        logger.warning(msg, exc_info=True)
        warnings.warn(msg, stacklevel=2)
        return
    for ep in eps:
        try:
            register = ep.load()
        except Exception as exc:  # noqa: BLE001
            msg = f"Skipping {group} plugin {ep.name!r}: failed to load entry point {ep.value!r}: {type(exc).__name__}: {exc}"
            logger.warning(msg, exc_info=True)
            warnings.warn(msg, stacklevel=2)
            continue
        if not callable(register):
            msg = f"Skipping {group} plugin {ep.name!r}: entry point {ep.value!r} did not resolve to a callable."
            logger.warning(msg)
            warnings.warn(msg, stacklevel=2)
            continue
        try:
            register()
        except Exception as exc:  # noqa: BLE001
            msg = f"Skipping {group} plugin {ep.name!r}: registration callable {ep.value!r} raised: {type(exc).__name__}: {exc}"
            logger.warning(msg, exc_info=True)
            warnings.warn(msg, stacklevel=2)
