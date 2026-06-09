"""Built-in GLM harness profile.

Mirrors the suffix-only MiniMax profile: it appends the same behavioral
suffix and makes no tool changes. Registered for the GLM family so we can
cross-check whether the suffix behaves differently on GLM (a different
model/provider).
"""

from deepagents.profiles.harness._minimax import _SYSTEM_PROMPT_SUFFIX
from deepagents.profiles.harness.harness_profiles import (
    HarnessProfile,
    _register_harness_profile_impl,
)

_GLM_MODEL_SPECS: tuple[str, ...] = (
    "fireworks:accounts/fireworks/models/glm-5p1",
    "fireworks:accounts/fireworks/models/glm-5",
    "baseten:zai-org/GLM-5",
    "openrouter:z-ai/glm-5.1",
)
"""Model specs that receive the GLM harness profile.

The suffix is shared with the MiniMax profile (imported, not copied) so the
profiles stay identical for the cross-model comparison.
"""


def register() -> None:
    """Register the built-in GLM harness profile for each GLM spec."""
    profile = HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX)
    for spec in _GLM_MODEL_SPECS:
        _register_harness_profile_impl(spec, profile)
