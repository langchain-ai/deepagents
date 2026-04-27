"""Built-in Claude Opus 4.7 harness profile.

Layers a system-prompt suffix onto `anthropic:claude-opus-4-7` tuned to
Claude Opus 4.7's documented behaviors:

- Universal Claude guidance that applies to every recent Claude —
  parallel tool calls, grounded (non-speculative) answers, and
  post-tool-result reflection.
- Claude Opus 4.7-specific overlays that counter the model's documented
  tendency to use tools and spawn subagents less aggressively than
  prior Opus generations when not prompted otherwise.

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
</tool_result_reflection>

<tool_usage>
When a task depends on the state of files, tests, or system output, use tools to observe that state directly rather than reasoning from memory about what it probably contains. Read files before describing them. Run tests before claiming they pass. Search the codebase before asserting a symbol does or does not exist. Active investigation with tools is the default mode of working, not a fallback.
</tool_usage>

<subagent_usage>
Do not spawn a subagent for work you can complete directly in a single response (e.g. refactoring a function you can already see).

Spawn multiple subagents in the same turn when fanning out across items or reading multiple files.
</subagent_usage>"""
"""Text appended to the assembled base system prompt."""


def register() -> None:
    """Register the built-in Claude Opus 4.7 harness profile."""
    _register_harness_profile_impl(
        "anthropic:claude-opus-4-7",
        HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX),
    )
