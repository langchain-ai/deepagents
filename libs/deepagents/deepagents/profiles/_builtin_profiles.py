"""Internal bootstrap for built-in profile registrations.

Imports built-in provider modules whose top-level code registers default
`ProviderProfile` entries. Kept separate from `deepagents.profiles.__init__`
so the public package surface stays focused on the public beta APIs.
"""

from importlib import import_module


def _ensure_builtin_profiles_loaded() -> None:
    """Register built-in provider profiles.

    Called once from `deepagents.profiles.__init__`. The imported modules
    register their profiles as a top-level side effect, so re-importing is a
    no-op thanks to the Python import cache.
    """
    import_module("deepagents.profiles._openai")
    import_module("deepagents.profiles._openrouter")
