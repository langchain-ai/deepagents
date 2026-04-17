"""OpenAI Codex model harness profiles.

!!! warning

    This is an internal API subject to change without deprecation. It is not
    intended for external use or consumption.

Registers per-model profiles for Codex variants (``gpt-5.2-codex``,
``gpt-5.3-codex``) that layer on top of the provider-wide OpenAI profile.
The merge gives us ``use_responses_api`` from the provider level plus
Codex-specific tuning here.

Prompt additions are derived from the official Codex Prompting Guide
(https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide)
and optimized for the Deep Agents tool set.

Context compaction on Codex uses the OpenAI Responses ``/compact`` endpoint
rather than the default LLM-based summarization. Compaction returns an
opaque ``encrypted_content`` item that preserves procedural fidelity better
than an ad-hoc text summary, which matters for long multi-turn sessions
with tool use. On ``/compact`` failure, the middleware transparently falls
back to the standard ``SummarizationMiddleware`` pathway — no crash, no
dropped context.
"""

from deepagents.profiles._harness_profiles import _HarnessProfile, _register_harness_profile

_CODEX_SYSTEM_PROMPT_SUFFIX = """\
## Codex-Specific Behavior

- You are an autonomous senior engineer. Once given a direction, proactively \
gather context, plan, implement, and verify without waiting for additional \
prompts at each step.
- Persist until the task is fully handled end-to-end within the current turn \
whenever feasible. Do not stop at analysis or partial fixes; carry changes \
through implementation, verification, and a clear explanation of outcomes.
- Bias to action: default to implementing with reasonable assumptions. Do not \
end your turn with clarifications unless truly blocked.
- Do not communicate an upfront plan or status preamble before acting. Just act.
- Avoid excessive looping or repetition. If you have already called a tool and \
received a valid response, use that information — do not re-invoke the same \
tool with identical arguments. If you find yourself re-reading, re-querying, \
or re-editing without clear progress, stop and summarize what is blocking you.

## Parallel Tool Use

- Before any tool call, decide ALL files and resources you will need.
- Batch reads, searches, and other independent operations into parallel tool \
calls instead of issuing them one at a time.
- Only make sequential calls when you truly cannot determine the next step \
without seeing a prior result.

## Plan Hygiene

- Before finishing, reconcile every TODO or plan item created via write_todos. \
Mark each as done, blocked (with a one-sentence reason), or cancelled. Do not \
finish with pending items.

## File Editing

- Use apply_patch for file edits — it uses the V4A diff format you are trained \
on. Batch logical changes into one patch instead of many small ones.
- For bulk operations across many files, use shell commands (sed, awk, etc.)."""

_CODEX_TOOL_ALIASES: dict[str, str] = {
    "execute": "shell_command",
    "list_dir": "ls",
}

_CODEX_TOOL_DESCRIPTION_OVERRIDES: dict[str, str] = {
    "execute": (
        "Runs a shell command and returns its output. "
        "Use for git operations, build commands, tests, and other terminal tasks. "
        "Prefer dedicated tools (read_file, grep, glob, list_dir) over "
        "shell_command equivalents."
    ),
}

_CODEX_PROFILE = _HarnessProfile(
    init_kwargs={"reasoning_effort": "medium"},
    system_prompt_suffix=_CODEX_SYSTEM_PROMPT_SUFFIX,
    tool_aliases=_CODEX_TOOL_ALIASES,
    tool_description_overrides=_CODEX_TOOL_DESCRIPTION_OVERRIDES,
    include_apply_patch=True,
    use_codex_compaction=True,
    excluded_tools=frozenset({"edit_file"}),
)

for _model in ("openai:gpt-5.2-codex", "openai:gpt-5.3-codex"):
    _register_harness_profile(_model, _CODEX_PROFILE)
