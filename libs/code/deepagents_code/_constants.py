"""Lightweight shared constants for the app.

This module is intentionally dependency-free (no third-party imports, no
sibling-module imports) so any other module — including the startup-critical
`main.py` and the heavy `agent.py` — can import from it without triggering a
chain of expensive imports.
"""

from __future__ import annotations

from typing import Final

DEFAULT_AGENT_NAME: Final[str] = "agent"
"""Default agent / assistant identifier when no `-a` flag is given."""

FS_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute"}
)
"""Mirror of the SDK's `FsToolName` literal members.

Hardcoded here rather than derived from `deepagents.FsToolName` because
`deepagents` must not be imported on the arg-parsing hot path (see AGENTS.md
"Startup performance"); this module is dependency-free and safe for `main.py` to
import. Consumers (`main._parse_allow_fs_tools_flag`,
`tool_catalog.collect_built_in_tools`) alias this set, and `get_args(FsToolName)`
drift guards in `test_main_args` and `test_tool_catalog` pin it so a new or
renamed SDK filesystem tool fails a test instead of silently diverging.
"""

FIREWORKS_PROVIDER_ID_PREFIX: Final[str] = "accounts/fireworks/"
"""Prefix used to infer Fireworks from fully-qualified IDs."""

FIREWORKS_MODEL_ID_PREFIXES: Final[tuple[str, ...]] = (
    "accounts/fireworks/models/",
    "accounts/fireworks/routers/",
)
"""Model and router ID prefixes used for stripping and classification."""

MCP_REENABLED_PENDING_ERROR: Final[str] = "Re-enabled — press Ctrl+R to load."
"""User-facing reconnect guidance shown for an MCP server that was optimistically
re-enabled but whose agent has not yet reconnected.

Set as `MCPServerInfo.error` by `app._apply_optimistic_disabled_state` (alongside
`pending_reconnect=True`, which is what `/tools` actually keys off). Named here
so the producer and the tests asserting the message share one literal.
"""

SYSTEM_MESSAGE_PREFIX: Final[str] = "[SYSTEM]"
"""Prefix for synthetic human messages (e.g. interrupt cancellation notices).

Such messages are written to the `messages` channel for the agent's benefit on
resume but are not user-authored, so they are filtered out of both the rendered
transcript and a thread's initial prompt. Shared here so the single producer
(`textual_adapter`) and its consumers (`app`, `sessions`) agree on one literal.
"""
