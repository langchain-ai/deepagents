"""Built-in MiniMax harness profile.

Registers a `HarnessProfile` for each MiniMax model spec
"""

# ruff: noqa: E501
# Prompt sections are single lines by design so they read as one instruction
# each; hard-wrapping them would not change behavior but would make edits harder
# to diff.

from deepagents.profiles.harness._reasoning_gate import ReasoningGateMiddleware
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
<track_and_verify>
For any request with multiple parts, or any action that changes state, keep a running checklist and verify it against reality before you report — don't rely on your memory of what you intended.

- Track every distinct thing the user asks for, plus any rule or limit you discover that affects their request, as items in write_todos, and keep that list current as the conversation continues.
- Before telling the user something is done, confirm it actually took effect: the matching tool call succeeded and its result matches what you're claiming.
- If a rule prevents doing exactly what the user asked, tell them explicitly and propose the best allowed alternative — don't apply a limit silently or quietly settle for less.

For simple, single-step requests, skip this and just answer.
</track_and_verify>

<report_back>
When you finish, tell the user what you did and give them the specific information they asked for, in your final message. The last message should stand on its own — put the actual answer there rather than pointing back to earlier steps or tool output.
</report_back>"""
"""Text appended to the assembled base system prompt for MiniMax models."""


def register() -> None:
    """Register the built-in MiniMax harness profile for each MiniMax spec.

    The profile pairs the `track_and_verify` prompt framework with a
    `ReasoningGateMiddleware` controller: it classifies each turn and, only when
    the turn needs hard reasoning, grades the agent's work against a fixed
    process rubric and re-runs once on a real violation.

    The classifier + grader run on a MiniMax-family model served by Fireworks
    (`minimax-m2p7`): the grader uses structured output, which MiniMax M3 on
    OpenRouter (the actor) does not support (no endpoint accepts a forced
    `tool_choice`). m2.7 is <= m3 in capability, so this adds no frontier
    assistance — the verification gain comes from a fresh-perspective grade
    against the rubric, not a stronger grader.
    """
    profile = HarnessProfile(
        system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX,
        extra_middleware=lambda: [
            ReasoningGateMiddleware(
                grader_model="fireworks:accounts/fireworks/models/minimax-m2p7"
            )
        ],
    )
    for spec in _MINIMAX_MODEL_SPECS:
        _register_harness_profile_impl(spec, profile)
