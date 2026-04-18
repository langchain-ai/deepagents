"""Anthropic Haiku 4.5 model harness profile.

!!! warning

    This is an internal API subject to change without deprecation. It is not
    intended for external use or consumption.

Registers an intentionally empty model-level profile for
``anthropic:claude-haiku-4-5``.  Same audit conclusion as the Sonnet 4.6
profile (see :mod:`deepagents.profiles._anthropic_sonnet`): the
provider-level Anthropic profile already covers every prompt-suffix
recommendation that Anthropic's `prompting best practices
<https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices>`_
attributes to "Claude's latest models" (the scope sentence explicitly
includes Haiku 4.5 in that set), and no Haiku-specific behavioral
tendencies are documented anywhere in the article or the `Haiku 4.5
migration guide
<https://platform.claude.com/docs/en/about-claude/models/migration-guide#migrating-to-claude-haiku-4-5>`_:

* The best-practices article contains only two incidental Haiku mentions
  (scope sentence + context-awareness note).  It contains zero Haiku
  behavioral-prompt recommendations.  The overengineering, overthinking,
  and subagent-overuse sections are all explicitly tagged Opus-only and
  do not apply here — the same reasoning that excluded them from Sonnet.
* The migration guide's Haiku 4.5 section is entirely API/config changes
  (model ID, rate limits, tool versions, sampling parameters, ``refusal``
  stop reason, extended-thinking configuration) — none of which are
  prompt concerns.  There is no behavior-change list comparable to the
  Opus 4.7 migration notes.

One wrinkle worth documenting in case it is revisited: Haiku 4.5 predates
the 4.6 generation and uses extended thinking with ``budget_tokens``, not
adaptive thinking (the article notes "Extended thinking is deprecated in
Claude 4.6 or newer models").  The provider-level
``<tool_result_reflection>`` section was originally motivated by
adaptive-/interleaved-thinking behavior, but its text is plain
behavioral guidance that does not depend on a specific thinking mode, so
inheriting it on Haiku is appropriate.  If future evals show the
section misfires on Haiku specifically, the fix belongs here as a
model-level override rather than a provider-level change.

Haiku 4.5 is frequently selected as a fast / low-cost subagent model in
Deep Agent setups.  The audit anchor is particularly valuable here:
"make Haiku prompts shorter / more efficient" is a plausible future
edit, but nothing in the source documentation supports it, and any such
steering should require new evidence (evals, documented behavior
changes) before landing.

The empty override composes with the Anthropic provider profile via
``_merge_profiles``: a ``None`` ``system_prompt_suffix`` on the override
falls back to the provider suffix, so Haiku 4.5 resolves to
``_ANTHROPIC_SYSTEM_PROMPT_SUFFIX`` verbatim.  The registration is
behaviorally a no-op; its value is the anchored audit and the tests.
"""

from deepagents.profiles._harness_profiles import _HarnessProfile, _register_harness_profile

_register_harness_profile(
    "anthropic:claude-haiku-4-5",
    _HarnessProfile(),
)
