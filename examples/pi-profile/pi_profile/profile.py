"""Pi-style `HarnessProfile` factory and registration helper.

Bottles up the Pi coding agent's minimal-prompt + tool-description style as
a reusable `deepagents.HarnessProfile`. Pi keeps the system prompt small and
pushes detail into per-tool descriptions and a short list of guidelines.

Tool-name mapping (Pi -> Deep Agents):

* `read`  -> `read_file`
* `write` -> `write_file`
* `edit`  -> `edit_file`
* `bash`  -> `execute`
* `find`  -> `glob`
* `ls`    -> `ls`
* `grep`  -> `grep`

Source: https://github.com/earendil-works/pi/tree/main/packages/coding-agent
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from deepagents import HarnessProfile, register_harness_profile

if TYPE_CHECKING:
    from collections.abc import Mapping


PI_BASE_SYSTEM_PROMPT: str = """\
You are an expert coding assistant operating inside a Deep Agents harness in \
Pi style. You help users by reading files, executing commands, editing code, \
and writing new files.

Guidelines:
- Prefer grep/glob/ls tools over execute for file exploration (faster, \
respects .gitignore).
- Use read_file to examine files instead of cat or sed.
- Use edit_file for precise changes (each edit's old_string must match exactly).
- When changing multiple separate locations in one file, prefer one edit_file \
call per location over batching unrelated edits.
- Use write_file only for new files or complete rewrites.
- Be concise in your responses.
- Show file paths clearly when working with files."""
"""Pi-flavored replacement for the default Deep Agents base system prompt.

Mirrors the minimalist structure of `buildSystemPrompt` in Pi's coding agent:
a one-line role statement followed by a short list of tool-driven guidelines.
Detail lives on per-tool descriptions, not the system prompt.
"""


_PI_TOOL_DESCRIPTIONS: dict[str, str] = {
    "read_file": (
        "Read the contents of a file. Supports text files. For text files, "
        "output is truncated. Use offset/limit for large files. When you need "
        "the full file, continue with offset until complete."
    ),
    "write_file": (
        "Write content to a file. Creates the file if it doesn't exist, "
        "overwrites if it does. Automatically creates parent directories. Use "
        "this only for new files or complete rewrites."
    ),
    "edit_file": (
        "Edit a single file using exact text replacement. The old_string must "
        "match a unique, non-overlapping region of the original file. If two "
        "changes affect the same block or nearby lines, merge them into one "
        "edit instead of emitting overlapping edits. Do not include large "
        "unchanged regions just to connect distant changes."
    ),
    "ls": (
        "List directory contents. Returns entries sorted alphabetically, with "
        "'/' suffix for directories. Includes dotfiles."
    ),
    "glob": (
        "Search for files by glob pattern. Returns matching file paths "
        "relative to the search directory. Respects .gitignore."
    ),
    "grep": (
        "Search file contents for a pattern. Returns matching lines with file "
        "paths and line numbers. Respects .gitignore."
    ),
    "execute": (
        "Execute a shell command in the current working directory. Returns "
        "stdout and stderr. Output is truncated to the tail when long. "
        "Optionally provide a timeout in seconds."
    ),
}
"""Pi-style descriptions mapped to Deep Agents tool names.

Adapted from Pi's tool definitions in
`packages/coding-agent/src/core/tools/` (read.ts, write.ts, edit.ts,
ls.ts, find.ts, grep.ts, bash.ts). Truncation byte/line counts in the
original strings are dropped here because Deep Agents' filesystem tools
use different limits.
"""

PI_TOOL_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(_PI_TOOL_DESCRIPTIONS)
"""Read-only public view of `_PI_TOOL_DESCRIPTIONS` for inspection by callers."""


def pi_harness_profile() -> HarnessProfile:
    """Build a Pi-style `HarnessProfile`.

    The returned profile replaces the assembled base system prompt with
    `PI_BASE_SYSTEM_PROMPT` and overrides each Deep Agents filesystem tool
    description with the Pi-flavored equivalent in `PI_TOOL_DESCRIPTIONS`.

    No middleware is excluded — Pi's "no sub-agents / no plan mode /
    no permission popups" philosophy is a packaging stance, not something
    that maps onto Deep Agents' required scaffolding (`FilesystemMiddleware`,
    `SubAgentMiddleware`, permission middleware). Callers who want to drop
    the general-purpose subagent can layer
    `general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)`
    on top via `register_harness_profile`'s additive merge.

    Returns:
        A fresh `HarnessProfile` instance each call. The instance carries
        immutable fields, so reuse across registrations is safe; the helper
        returns a new object only so callers can mutate the dict shape
        before registration if they want to.
    """
    return HarnessProfile(
        base_system_prompt=PI_BASE_SYSTEM_PROMPT,
        tool_description_overrides=dict(PI_TOOL_DESCRIPTIONS),
    )


def register_pi_harness(key: str) -> HarnessProfile:
    """Register the Pi-style profile under `key`.

    `key` follows the standard Deep Agents profile-key shape: a bare
    provider name (e.g. `"anthropic"`) or a `provider:model` spec (e.g.
    `"anthropic:claude-sonnet-4-6"`). Registration is additive — if a
    profile already exists under `key`, the Pi fields layer on top via
    the documented merge semantics.

    Args:
        key: Provider or `provider:model` registry key.

    Returns:
        The `HarnessProfile` that was registered, so callers can inspect or
        further customize it.
    """
    profile = pi_harness_profile()
    register_harness_profile(key, profile)
    return profile
