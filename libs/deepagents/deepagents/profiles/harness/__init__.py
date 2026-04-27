"""Harness profile package: `HarnessProfile` API and built-in registrations.

Individual built-in modules register their profiles as a top-level import side
effect. The lazy `_builtin_profiles` bootstrap imports them once on first
profile-registry access.
"""

from deepagents.profiles.harness.harness_profiles import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    HarnessProfileConfig,
    register_harness_profile,
)

__all__ = [
    "GeneralPurposeSubagentProfile",
    "HarnessProfile",
    "HarnessProfileConfig",
    "register_harness_profile",
]
