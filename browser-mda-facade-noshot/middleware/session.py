"""Release the Browserbase session when the agent finishes.

Without this the session lives out the TTL passed to `browserbase.launch`,
holding a concurrency slot while doing nothing. Measured: a single 100-task
suite at concurrency 8 pinned the project at all 75 live sessions, and a
second arm started alongside it failed every call with
`BrowserbaseSessionError: Failed to create a Browserbase session`.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware

from tools.facade import release

logger = logging.getLogger(__name__)


class ReleaseBrowserSession(AgentMiddleware):
    """Close the thread's browser session on the way out."""

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        try:
            await release(runtime)
        except Exception:  # noqa: BLE001 - teardown must not fail a finished run
            logger.debug("releasing the browser session failed", exc_info=True)
        return None
