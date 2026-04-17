"""Anthropic Opus 4.6 model harness profile.

!!! warning

    This is an internal API subject to change without deprecation. It is not
    intended for external use or consumption.

Registers a model-level profile for ``anthropic:claude-opus-4-6`` that layers
Opus-specific agentic guidance on top of the Anthropic provider profile.  The
three additional sections address behavioral tendencies documented in Anthropic's
`prompting best practices for Claude 4.6
<https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices>`_
that are specific to Opus 4.6 (and do not apply to Sonnet / Haiku):

* **Minimal changes** — curbs overengineering and scope creep.
* **Subagent discipline** — prevents excessive delegation for simple tasks.
* **Focused exploration** — constrains broad upfront codebase scanning.

Because ``_merge_profiles`` replaces ``system_prompt_suffix`` (scalar override),
this module composes its suffix from ``_ANTHROPIC_SYSTEM_PROMPT_SUFFIX`` (the
provider-level sections) plus the Opus-specific additions so nothing is lost.
"""
# ruff: noqa: E501

from deepagents.profiles._anthropic import _ANTHROPIC_SYSTEM_PROMPT_SUFFIX
from deepagents.profiles._harness_profiles import _HarnessProfile, _register_harness_profile

_OPUS_SYSTEM_PROMPT_SUFFIX = """\
<minimal_changes>
Only make changes that are directly requested or clearly necessary to fulfill the request. Do not add features, refactor surrounding code, or introduce abstractions beyond what the task requires. A bug fix does not need nearby code cleaned up. A simple feature does not need extra configurability. Do not add documentation, type annotations, or error handling to code you did not change. The right amount of complexity is the minimum needed for the current task.
</minimal_changes>

<subagent_discipline>
Prefer working directly over delegating to subagents when the task is simple — single-file edits, quick lookups, sequential operations, or anything that takes only a few tool calls. Reserve subagents for work that genuinely benefits from parallelism, isolated context, or independent workstreams. When in doubt, do the work yourself.
</subagent_discipline>

<focused_exploration>
When starting a task, read the specific files relevant to the problem rather than broadly scanning the codebase. Form a plan from what you find and begin executing. Expand your investigation only when you encounter something that requires additional context — do not front-load extensive exploration.
</focused_exploration>"""
"""Opus-specific prompt sections appended after the provider-level Anthropic sections.

* **Minimal changes** — sourced from "Overeagerness" (article section on Opus 4.5
  and Opus 4.6 overengineering).  Language is moderate to avoid undertriggering on
  legitimate refactoring tasks.
* **Subagent discipline** — sourced from "Subagent orchestration" (article notes
  Opus 4.6 has a strong predilection for subagents).  Complements the existing
  ``TASK_SYSTEM_PROMPT`` "when NOT to use" guidance by framing it as a model-level
  behavioral preference rather than a tool-level rule.
* **Focused exploration** — sourced from "Overthinking and excessive thoroughness"
  (article notes Opus 4.6 does significantly more upfront exploration).  Strengthens
  the provider-level ``<decisive_execution>`` which gives mild general coverage.
"""

_ANTHROPIC_OPUS_SYSTEM_PROMPT_SUFFIX = _ANTHROPIC_SYSTEM_PROMPT_SUFFIX + "\n\n" + _OPUS_SYSTEM_PROMPT_SUFFIX
"""Full system prompt suffix for Opus 4.6: Anthropic provider sections + Opus overlay."""

_register_harness_profile(
    "anthropic:claude-opus-4-6",
    _HarnessProfile(system_prompt_suffix=_ANTHROPIC_OPUS_SYSTEM_PROMPT_SUFFIX),
)
