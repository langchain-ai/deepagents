"""Browserbase's hosted MCP server as this agent's browser.

Third point of comparison. The other variants install Chrome into the agent's
own sandbox and drive it by shelling out; here the browser is Browserbase's and
the tools arrive over MCP, so the agent needs no sandbox, no `execute`, and no
provision step.

Two consequences worth stating before reading any numbers against the sandbox
variants:

* Stagehand's `act`/`extract`/`observe` run on Browserbase's side and make their
  own model calls, which never appear in this agent's trace. Its step count and
  cost therefore understate the real work, and are not comparable like-for-like
  with a variant whose reasoning is all local.
* That reasoning model is pinned to the REPL variant's via `STAGEHAND_MODEL` /
  `STAGEHAND_MODEL_KEY` below, so the two variants differ in interface rather
  than in model. It remains a different model from the one driving this agent.

Auth is a query parameter rather than a header -- the endpoint takes
`browserbaseApiKey` on the URL.
"""

import os
from urllib.parse import urlencode

from managed_deepagents import define_mcp

_KEY = os.environ.get("BROWSERBASE_API_KEY", "")

# Hold the Stagehand reasoning model equal to the REPL variant's.
#
# This server defaults `modelName` to `google/gemini-2.5-flash-lite` -- verified
# from its own responses, not just the docs. `browser-mda-repl` now matches that
# default, so neither arm overrides and the plumbing below is a no-op until
# someone sets `STAGEHAND_MODEL_KEY`. It is kept because the model is worth
# varying deliberately later: act/observe/extract are structured-output calls, act/observe/extract are structured-output calls, and in the
# 30-task run 82 of this variant's 88 `act` failures were the single string
# `AI_NoObjectGeneratedError: No object generated: response did not match schema`
# -- a 37.8% hard failure rate on the one verb that changes page state. A
# comparison against the REPL arm with this unpinned measures two models, not two
# interfaces.
#
# `modelApiKey` is required whenever `modelName` is non-default, so both are set
# together or neither is: without a key the server rejects the override, and
# falling back to the documented default is better than failing every tool call.
_MODEL = os.environ.get("STAGEHAND_MODEL", "google/gemini-2.5-flash-lite")
_MODEL_KEY = os.environ.get("STAGEHAND_MODEL_KEY", "")

_params = {"browserbaseApiKey": _KEY, "keepAlive": "true"}
if _MODEL_KEY:
    _params["modelName"] = _MODEL
    _params["modelApiKey"] = _MODEL_KEY

mcp = define_mcp(
    servers={
        "browserbase": {
            # keepAlive=true: without it the server drops the SSE stream part way
            # through a task and the run dies with
            # MCPError(-32000, 'SSE stream ended without a response').
            "url": f"https://mcp.browserbase.com/mcp?{urlencode(_params)}",
            # "http" is Browserbase's hosted Streamable HTTP endpoint -- not a
            # legacy SSE transport. SSE is the response-stream mechanism inside
            # Streamable HTTP, which is why long tasks still fail with
            # MCPError(-32000, 'SSE stream ended without a response') even with
            # keepAlive=true. An explicit timeout keeps a stalled stream from
            # hanging the whole run.
            "transport": "http",
            "timeout": 180,
        }
    }
)
