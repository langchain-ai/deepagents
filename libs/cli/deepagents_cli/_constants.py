"""Lightweight shared constants for the CLI.

This module is intentionally dependency-free (no third-party imports, no
sibling-module imports) so any other module — including the startup-critical
`main.py` and the heavy `agent.py` — can import from it without triggering a
chain of expensive imports.
"""

from __future__ import annotations

DEFAULT_AGENT_NAME = "agent"
"""Single source of truth for the default agent / assistant identifier.

Re-exported as `agent.DEFAULT_AGENT_NAME`, `_server_config.DEFAULT_ASSISTANT_ID`,
and imported as `_DEFAULT_AGENT_NAME` in `main.py`. Used as the default for
`ServerConfig.assistant_id` and as the fallback when no `-a` flag is given.
"""
