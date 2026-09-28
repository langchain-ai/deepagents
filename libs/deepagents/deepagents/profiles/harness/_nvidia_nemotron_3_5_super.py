"""Nemotron 3.5 Super harness profile using the verified Nemotron controls.

The NVIDIA API Catalog identifier is pending publication. Update these exact
model keys when the public API Catalog entry is available.
"""

from __future__ import annotations

from deepagents.profiles.harness._nvidia_nemotron_3_ultra import build_nemotron_profile
from deepagents.profiles.harness.harness_profiles import _register_harness_profile_impl

_NEMOTRON_SUPER_MODEL_SPECS: tuple[str, ...] = (
    "NVIDIA:nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
    "nvidia:nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
)


def register() -> None:
    """Register the initial Nemotron 3.5 Super harness profile."""
    profile = build_nemotron_profile()
    for spec in _NEMOTRON_SUPER_MODEL_SPECS:
        _register_harness_profile_impl(spec, profile)
