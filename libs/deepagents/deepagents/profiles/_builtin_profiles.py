"""Internal bootstrap for built-in profile registrations.

Imports built-in provider modules whose top-level code registers default
`ProviderProfile` entries. Kept separate from `deepagents.profiles.__init__`
so the public package surface stays focused on the public beta APIs.
"""

from functools import cache
from importlib import import_module


@cache
def _ensure_builtin_profiles_loaded() -> None:
    """Ensure built-in provider profiles are registered once."""
    import_module("deepagents.profiles._openai")
    import_module("deepagents.profiles._openrouter")
