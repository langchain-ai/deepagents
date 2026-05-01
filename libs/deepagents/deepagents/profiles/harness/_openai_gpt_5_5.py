"""Built-in OpenAI GPT-5.5 harness profile.

Registers a `HarnessProfile` for `openai:gpt-5.5` with a concise,
outcome-first `system_prompt_suffix` drawn from OpenAI's GPT-5.5 prompting
guidance: define success criteria and stopping conditions, keep user-visible
updates short, use tools selectively, and validate before finalizing.

The profile is keyed to GPT-5.5 rather than the `"openai"` provider because
OpenAI recommends treating GPT-5.5 as a new model family to tune for, not as a
drop-in replacement for earlier GPT-5 models.
"""

from deepagents.profiles.harness.harness_profiles import (
    HarnessProfile,
    _register_harness_profile_impl,
)

_MODEL_SPEC = "openai:gpt-5.5"
"""Model spec that receives the GPT-5.5 harness profile."""

_SYSTEM_PROMPT_SUFFIX: str = """\
## GPT-5.5-Specific Behavior

- Prefer outcome-first execution: identify the target result, success criteria, constraints, and available evidence, then choose an efficient path.
- Avoid process-heavy or mechanical step sequences unless the user or task requires them. Use strict words only for true invariants.
- If the request is clear enough and the next step is reversible and low-risk, proceed with reasonable assumptions. Ask only for missing information \
that would materially change the outcome or create risk.

## User Updates

- For multi-step or tool-heavy work, send a brief user-visible update before the first tool call that states the first step.
- Keep progress updates sparse and outcome-based. Do not narrate routine tool calls.

## Tool Use and Stopping Rules

- Use tools when they materially improve correctness, completeness, grounding, or validation.
- Prefer independent parallel retrieval when it reduces wall-clock time, but do not parallelize steps with real dependencies.
- Stop once you can satisfy the user's core request with sufficient evidence. Do not add extra searches only to improve phrasing or gather details.

## Validation

- Before finalizing, check that the answer satisfies the request, is grounded in context or tool outputs, and follows the requested format.
- If validation cannot be run, state why and describe the next best check."""
"""Text appended to the assembled base system prompt."""


def register() -> None:
    """Register the built-in GPT-5.5 harness profile."""
    _register_harness_profile_impl(
        _MODEL_SPEC,
        HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX),
    )
