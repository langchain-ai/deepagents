"""Built-in Claude Sonnet 4.6 harness profile.

Layers Anthropic's universal Claude guidance onto
`anthropic:claude-sonnet-4-6` — parallel tool calls, grounded (non-
speculative) answers, and post-tool-result reflection.

No Claude-Sonnet-4.6-specific overlays. Anthropic's published guidance
for Sonnet 4.6 centers on API-level configuration (effort defaults,
adaptive thinking, `budget_tokens` deprecation) rather than system-
prompt adjustments; the overeagerness, overthinking, and subagent-
overuse prompt snippets in the guide are tagged for Claude Opus 4.5 /
Claude Opus 4.6 and do not apply to Sonnet 4.6. This module exists as
the audit anchor: its presence documents the review and justifies the
absence of model-specific prompt content. If a future revision of the
prompting guide adds Sonnet-4.6-specific guidance, add it here rather
than at the provider key so it does not leak onto other Anthropic
models.

Source: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices
"""

# ruff: noqa: E501
# Prompt sections are single lines by design to match Anthropic's
# published samples verbatim; hard-wrapping them would diverge from the
# source of truth and make future updates harder to diff.

from deepagents.profiles.harness_profiles import (
    HarnessProfile,
    _register_harness_profile_impl,
)

_SYSTEM_PROMPT_SUFFIX = """\
<use_parallel_tool_calls>
If you intend to call multiple tools and there are no dependencies between the tool calls, make all of the independent tool calls in parallel. Prioritize calling tools simultaneously whenever the actions can be done in parallel rather than sequentially. For example, when reading 3 files, run 3 tool calls in parallel to read all 3 files into context at the same time. Maximize use of parallel tool calls where possible to increase speed and efficiency. However, if some tool calls depend on previous calls to inform dependent values like the parameters, do NOT call these tools in parallel and instead call them sequentially. Never use placeholders or guess missing parameters in tool calls.
</use_parallel_tool_calls>

<investigate_before_answering>
Never speculate about code you have not opened. If the user references a specific file, you MUST read the file before answering. Make sure to investigate and read relevant files BEFORE answering questions about the codebase. Never make any claims about code before investigating unless you are certain of the correct answer - give grounded and hallucination-free answers.
</investigate_before_answering>

<tool_result_reflection>
After receiving tool results, carefully reflect on their quality and determine optimal next steps before proceeding. Use your thinking to plan and iterate based on this new information, and then take the best next action.
</tool_result_reflection>"""
"""Text appended to the assembled base system prompt."""


def register() -> None:
    """Register the built-in Claude Sonnet 4.6 harness profile."""
    _register_harness_profile_impl(
        "anthropic:claude-sonnet-4-6",
        HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX),
    )
