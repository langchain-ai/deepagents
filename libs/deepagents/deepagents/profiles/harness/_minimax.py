"""Built-in MiniMax harness profile.

Registers a `HarnessProfile` for each MiniMax model spec
"""

# ruff: noqa: E501
# Prompt sections are single lines by design so they read as one instruction
# each; hard-wrapping them would not change behavior but would make edits harder
# to diff.

from deepagents.profiles.harness.harness_profiles import (
    HarnessProfile,
    _register_harness_profile_impl,
)

_MINIMAX_MODEL_SPECS: tuple[str, ...] = (
    "openrouter:minimax/minimax-m3",
    "openrouter:minimax/minimax-m2.7",
    "fireworks:accounts/fireworks/models/minimax-m2p7",
    "fireworks:accounts/fireworks/models/minimax-m2p5",
    "baseten:MiniMaxAI/MiniMax-M2.5",
)
"""Model specs that receive the MiniMax harness profile.

The MiniMax family shares the same response style across backends, so a single
profile applies to all of them. Add or remove specs here when a new MiniMax
variant ships or a backend ID changes.

The `ollama:minimax-*:cloud` specs are intentionally absent: their identifiers
contain a second colon (`minimax-m2.7:cloud`), which the `provider:model`
profile-key grammar rejects. Registering under the bare `ollama` provider would
be too broad (it would capture every ollama model), so ollama is left on stock
defaults.
"""

_SYSTEM_PROMPT_SUFFIX: str = """\
<completing_state_changes>
When a task asks you to change something — create, edit, delete, submit, or update state through tools — confirm the change actually took effect before you end your turn. Verify against the system's real state rather than assuming a tool call succeeded, and don't stop after only part of the requested change is done. A correct plan with incomplete or unverified execution is still an incomplete task.
</completing_state_changes>

<find_a_permitted_path>
When you can't do exactly what was asked, work out the user's underlying goal and check whether a different sequence of allowed actions reaches it before reporting the request as impossible to the user. If a rule or limit blocks the direct route, state the constraint and its consequence, then take or offer the alternative — don't silently settle for a lesser outcome or quietly work around a limit the user would want to know about.
</find_a_permitted_path>

<report_back>
When you finish, tell the user what you did and give them the specific information they asked for, in your final message. The last message should stand on its own — put the actual answer there rather than pointing back to earlier steps or tool output.
</report_back>

<manage_context>
If a compact_conversation tool is available, call it when starting an unrelated task or after a large read, so stale context doesn't crowd out what the current task needs.
</manage_context>"""
"""Text appended to the assembled base system prompt for MiniMax models."""


def register() -> None:
    """Register the built-in MiniMax harness profile for each MiniMax spec."""
    profile = HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX)
    for spec in _MINIMAX_MODEL_SPECS:
        _register_harness_profile_impl(spec, profile)
