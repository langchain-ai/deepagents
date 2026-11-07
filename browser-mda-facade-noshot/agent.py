"""The facade arm with its eyes closed: `snapshot` and `run`, no `screenshot`.

An ablation, not a new design. On WebTailBench the facade arm spent **93.8% of
its tool output on screenshots** -- 42 calls across 11 tasks at a median 89KB
of base64 each -- while agent-browser, which cannot see at all, used 16% fewer
tokens and scored 17 points lower.

So vision is doing something. This arm asks how much. If accuracy holds, those
screenshots are waste and the facade gets strictly cheaper; if it falls toward
agent-browser's 0.67, the screenshots were buying the gap and the cost is the
price of sight.

Deployed separately rather than toggled on `browser-agent-facade`, so a run
already in flight against that deployment is not disturbed.
"""

import os

# Set before importing tools.facade: FACADE_TOOLS is built at import time, so
# flipping the flag afterwards would silently keep the screenshot tool.
os.environ.setdefault("FACADE_SCREENSHOTS", "0")

from managed_deepagents import define_deep_agent  # noqa: E402

from middleware.session import ReleaseBrowserSession
from middleware.tool_errors import ToolErrorsAreMessages
from tools.facade import FACADE_TOOLS

agent = define_deep_agent(
    name="browser-agent-facade-noshot",
    model="anthropic:claude-sonnet-5-5",
    tools=FACADE_TOOLS,
    # An exception out of a tool ends the whole run; a browsing failure is
    # almost always something the next turn can route around.
    middleware=[ToolErrorsAreMessages(), ReleaseBrowserSession()],
)
