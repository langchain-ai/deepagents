"""Harness profile package: `HarnessProfile` API and built-in registrations.

Individual built-in modules expose a zero-arg `register()` callable; the lazy
`_builtin_profiles` bootstrap invokes them once on first profile-registry
access. Built-ins must not register at module import time — registration runs
under the bootstrap mutex, so a top-level call would race with concurrent
lookups and bypass the additive-merge semantics.
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
