"""Browsing agent that writes JavaScript instead of calling tools one at a time.

The browser tools are NOT given to the model. They are exposed only inside a
QuickJS REPL as `tools.*`, so the only thing the agent can emit is code.

The motive is measured. Across 180 runs of the other two harnesses, every
browser action costs a model call, and in the MCP variant a single
`browserbase_navigate` injects a median 11,160 tokens of serialized CDP object
graph that the agent has no way to trim -- 238 such calls carried 20.7M tokens,
67% of that variant's entire input budget.

Code mode attacks both. One `eval` can navigate, extract, filter and loop, and
the snippet decides what comes back, so a page the agent did not ask to read
never enters the context.
"""

from langchain_quickjs import CodeInterpreterMiddleware
from managed_deepagents import define_deep_agent

from middleware.tool_errors import ToolErrorsAreMessages
from tools.browser import BROWSER_TOOLS

agent = define_deep_agent(
    name="browser-agent-repl",
    model="anthropic:claude-sonnet-5-5",
    # Tools go to PTC, not to the model: `tools=` would also surface them as
    # ordinary tool calls and the agent would use those instead of writing code.
    middleware=[
        # Outermost, so it also contains failures raised by the REPL itself.
        ToolErrorsAreMessages(),
        CodeInterpreterMiddleware(
            ptc=BROWSER_TOOLS,
            # A Stagehand act/extract on a slow page routinely passes 30s. When the
            # REPL deadline fired mid-call the QuickJS context was torn down under
            # the in-flight host call and the run died on ValueError("already
            # closed") rather than returning a timeout the agent could read.
            timeout=180.0,
            max_result_chars=8000,
            subagents=False,
        )
    ],
)
