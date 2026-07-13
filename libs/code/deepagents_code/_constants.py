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

FIREWORKS_PROVIDER_ID_PREFIX: Final[str] = "accounts/fireworks/"
"""Broad stem shared by every fully-qualified Fireworks ID.

Used for provider *inference* (does this ID belong to Fireworks?), in the same
spirit as LangChain's own `accounts/fireworks`-based inference. The trailing
slash keeps the match anchored to the account namespace so lookalikes such as
`accounts/fireworks-enterprise/...` do not resolve to Fireworks. Deliberately
broader than `FIREWORKS_MODEL_ID_PREFIXES` so any current or future
`accounts/fireworks/*` namespace resolves to the provider, not just `models/`
and `routers/`."""

FIREWORKS_MODEL_ID_PREFIXES: Final[tuple[str, ...]] = (
    "accounts/fireworks/models/",
    "accounts/fireworks/routers/",
)
"""Fully-qualified prefixes for Fireworks model and router IDs.

Narrower than `FIREWORKS_PROVIDER_ID_PREFIX`: these enumerate the exact
namespaces used for display stripping and reasoning-effort classification. The
full namespace segment (through `models/` or `routers/`) is included so that
stripping leaves only the bare model name rather than a leftover
`models/`/`routers/` fragment."""

SYSTEM_MESSAGE_PREFIX: Final[str] = "[SYSTEM]"
"""Prefix for synthetic human messages (e.g. interrupt cancellation notices).

Such messages are written to the `messages` channel for the agent's benefit on
resume but are not user-authored, so they are filtered out of both the rendered
transcript and a thread's initial prompt. Shared here so the single producer
(`textual_adapter`) and its consumers (`app`, `sessions`) agree on one literal.
"""
