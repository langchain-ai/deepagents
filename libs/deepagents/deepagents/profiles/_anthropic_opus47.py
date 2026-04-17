"""Anthropic Opus 4.7 model harness profile.

!!! warning

    This is an internal API subject to change without deprecation. It is not
    intended for external use or consumption.

Registers a model-level profile for ``anthropic:claude-opus-4-7`` that layers
Opus 4.7-specific agentic guidance on top of the Anthropic provider profile.

Opus 4.7's native tendencies are inverted from Opus 4.6 on several axes: the
`migration guide
<https://platform.claude.com/docs/en/about-claude/models/migration-guide#migrating-to-claude-opus-4-7>`_
documents that Opus 4.7 spawns fewer subagents by default, uses tools less
often (preferring reasoning), and scopes its work more strictly to what was
asked. As a result, the Opus 4.6 overlay sections (``<minimal_changes>``,
``<subagent_discipline>``, ``<focused_exploration>``) would compound these
tendencies in the wrong direction for a Deep Agent and are deliberately *not*
included here.

The two sections below are sourced from explicit Anthropic recommendations in
the 4.7 migration guide:

* **Tool usage** — counteracts the reduced tool-calling default (#7).
* **Subagent usage** — counteracts the reduced subagent-spawning default (#5).

Because ``_merge_profiles`` replaces ``system_prompt_suffix`` (scalar override),
this module composes its suffix from ``_ANTHROPIC_SYSTEM_PROMPT_SUFFIX`` (the
provider-level sections) plus the Opus 4.7 additions so the provider guidance
is preserved.
"""
# ruff: noqa: E501

from deepagents.profiles._anthropic import _ANTHROPIC_SYSTEM_PROMPT_SUFFIX
from deepagents.profiles._harness_profiles import _HarnessProfile, _register_harness_profile

_OPUS_47_SYSTEM_PROMPT_SUFFIX = """\
<tool_usage>
When a task depends on the state of files, tests, or system output, use tools to observe that state directly rather than reasoning from memory about what it probably contains. Read files before describing them. Run tests before claiming they pass. Search the codebase before asserting a symbol does or does not exist. Active investigation with tools is the default mode of working, not a fallback.
</tool_usage>

<subagent_usage>
Consider spawning subagents when a task has genuinely independent workstreams that can run in parallel, when a subtask needs isolated context to avoid polluting the main thread, or when delegating a large well-scoped unit of work would reduce orchestration overhead. Subagents are especially useful for independent research, parallel code exploration across unrelated areas, and large self-contained implementations. Use them when the parallelism or isolation pays for the coordination cost.
</subagent_usage>"""
"""Opus 4.7-specific prompt sections appended after the provider-level Anthropic sections.

* **Tool usage** — sourced from migration guide behavior change #7
  ("Fewer tool calls by default").  Reinforces active investigation via tools
  rather than reasoning from memory.  Complements the provider-level
  ``<grounded_responses>`` which targets hallucinated *claims*; this section
  targets the *working loop*.
* **Subagent usage** — sourced from migration guide behavior change #5
  ("Fewer subagents spawned by default").  Mild, specific encouragement for
  the cases where subagents genuinely help.  Moderate tone per the "more
  literal instruction following" caveat in the migration guide (#2).
"""

_ANTHROPIC_OPUS_47_SYSTEM_PROMPT_SUFFIX = _ANTHROPIC_SYSTEM_PROMPT_SUFFIX + "\n\n" + _OPUS_47_SYSTEM_PROMPT_SUFFIX
"""Full system prompt suffix for Opus 4.7: Anthropic provider sections + Opus 4.7 overlay."""

_register_harness_profile(
    "anthropic:claude-opus-4-7",
    _HarnessProfile(system_prompt_suffix=_ANTHROPIC_OPUS_47_SYSTEM_PROMPT_SUFFIX),
)
