"""Internal bootstrap for built-in profile registrations.

Imports built-in provider and harness modules whose top-level code registers
default profile entries, then snapshots the harness registry so later
user-registered profiles can be told apart from built-ins.
"""

from importlib import import_module

from deepagents.profiles.harness_profiles import _HARNESS_PROFILES

_BUILTIN_HARNESS_KEYS: frozenset[str] = frozenset()
"""Snapshot of harness-profile keys registered by the built-in bootstrap.

Populated once by `_ensure_builtin_profiles_loaded`. Used by
`_has_any_harness_profile` to distinguish user registrations from built-ins
without tracking provenance on the registry itself.
"""


def _ensure_builtin_profiles_loaded() -> None:
    """Register built-in provider and harness profiles.

    Called once from `deepagents.profiles.__init__`. The imported modules
    register their profiles as a top-level side effect, so re-importing is a
    no-op thanks to the Python import cache. After the modules load, snapshot
    the harness keys so later user registrations can be told apart from the
    built-ins.
    """
    import_module("deepagents.profiles.provider._openai")
    import_module("deepagents.profiles.provider._openrouter")

    # Snapshot has to replace the module-level binding so downstream callers
    # see the frozen post-bootstrap set, not the empty default.
    global _BUILTIN_HARNESS_KEYS  # noqa: PLW0603
    _BUILTIN_HARNESS_KEYS = frozenset(_HARNESS_PROFILES)
