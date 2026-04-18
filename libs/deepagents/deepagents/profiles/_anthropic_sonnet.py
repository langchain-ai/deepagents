"""Anthropic Sonnet 4.6 model harness profile.

!!! warning

    This is an internal API subject to change without deprecation. It is not
    intended for external use or consumption.

Registers an intentionally empty model-level profile for
``anthropic:claude-sonnet-4-6``.  The provider-level Anthropic profile
(:data:`~deepagents.profiles._anthropic._ANTHROPIC_SYSTEM_PROMPT_SUFFIX`)
already covers every prompt-suffix recommendation that Anthropic's
`prompting best practices
<https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices>`_
attributes to "Claude's latest models" as a class (parallel tool calling,
grounded responses, post-tool reflection, decisive execution).  Nothing in
that article — or in the `Sonnet 4.6 migration guide
<https://platform.claude.com/docs/en/about-claude/models/migration-guide#migrating-to-claude-sonnet-4-6>`_
— documents Sonnet-specific behavioral tendencies that warrant additional
prompt steering:

* The article's overeagerness, overthinking, and subagent-overuse warnings
  are all explicitly tagged "Opus 4.5 / 4.6" (not Sonnet).  The Opus 4.6
  overlay in :mod:`deepagents.profiles._anthropic_opus` already carries
  those and notes they "do not apply to Sonnet / Haiku".
* Opus 4.7's migration guide lists seven concrete behavior changes (fewer
  tool calls, fewer subagents, more literal instruction following, etc.)
  that justify its overlay in
  :mod:`deepagents.profiles._anthropic_opus47`.  Sonnet 4.6's migration
  section contains no comparable behavior-change list — only API/config
  changes (prefill deprecation, adaptive thinking, effort parameter) that
  are handled at the SDK/API layer, not the prompt.
* Sonnet 4.6-specific callouts in the article (default ``high`` effort,
  computer-use best-in-class with adaptive thinking, positioning for
  fast-turnaround workloads) are API kwargs or use-case guidance, not
  prompt content.  Forcing defaults like ``effort=medium`` at the harness
  level would hurt users who legitimately want ``high`` effort for
  autonomous coding; those choices belong at the call site via
  ``init_kwargs``.

Registering an empty profile (rather than omitting this module) serves
three purposes:

1. Makes the audit discoverable — anyone grepping for ``sonnet`` in the
   profiles package lands here and finds the documented rationale.
2. Provides a drop-in template if a future Sonnet release documents
   behavior changes the way Opus 4.7 did.
3. Gives the test suite (``TestBuiltInProfiles``) a stable anchor to
   assert "Sonnet 4.6 uses the provider suffix unchanged", preventing
   silent divergence.

The empty override is composed with the Anthropic provider profile by
``_merge_profiles``: a ``None`` ``system_prompt_suffix`` on the override
falls back to the provider suffix, so Sonnet 4.6 resolves to
``_ANTHROPIC_SYSTEM_PROMPT_SUFFIX`` verbatim — identical to what a bare
provider-prefix lookup would return.  The extra registration is
behaviorally a no-op; its value is the anchored audit and the tests.
"""

from deepagents.profiles._harness_profiles import _HarnessProfile, _register_harness_profile

_register_harness_profile(
    "anthropic:claude-sonnet-4-6",
    _HarnessProfile(),
)
