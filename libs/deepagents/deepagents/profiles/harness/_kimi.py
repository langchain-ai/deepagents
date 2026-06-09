"""Built-in Kimi harness profile.

Mirrors the suffix-only MiniMax profile: it appends the same behavioral
suffix and makes no tool changes. Registered for the Kimi family so we can
cross-check whether the suffix that was null on MiniMax behaves differently
on Kimi (a different model/provider).
"""

from deepagents.profiles.harness._minimax import _SYSTEM_PROMPT_SUFFIX
from deepagents.profiles.harness.harness_profiles import (
    HarnessProfile,
    _register_harness_profile_impl,
)

_KIMI_MODEL_SPECS: tuple[str, ...] = (
    "fireworks:accounts/fireworks/models/kimi-k2p6",
    "fireworks:accounts/fireworks/models/kimi-k2p5",
    "baseten:moonshotai/Kimi-K2.6",
    "baseten:moonshotai/Kimi-K2.5",
    "openrouter:moonshotai/kimi-k2.6",
    "openrouter:moonshotai/kimi-k2.5",
)
"""Model specs that receive the Kimi harness profile.

The suffix is shared with the MiniMax profile (imported, not copied) so the
two stay identical for the cross-model comparison.
"""


def register() -> None:
    """Register the built-in Kimi harness profile for each Kimi spec."""
    profile = HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX)
    for spec in _KIMI_MODEL_SPECS:
        _register_harness_profile_impl(spec, profile)
