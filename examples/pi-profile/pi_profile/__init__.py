"""Pi-style harness profile for Deep Agents.

A callable `HarnessProfile` factory that mirrors the prompt, tool
descriptions, and guideline style of the Pi coding agent
(https://pi.dev/, https://github.com/earendil-works/pi). Register it
against any provider or `provider:model` key supported by Deep Agents.
"""

from pi_profile.profile import (
    PI_BASE_SYSTEM_PROMPT,
    PI_TOOL_DESCRIPTIONS,
    pi_harness_profile,
    register_pi_harness,
)

__all__ = [
    "PI_BASE_SYSTEM_PROMPT",
    "PI_TOOL_DESCRIPTIONS",
    "pi_harness_profile",
    "register_pi_harness",
]
