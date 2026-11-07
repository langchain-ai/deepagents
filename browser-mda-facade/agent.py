"""Browsing agent built on Stagehand's facade contract: snapshot, run, screenshot.

Fourth arm of the comparison, and the one where no second model is involved.
The MCP arm's `act`/`extract`/`observe` each spend a Stagehand-side inference
call that never appears in our trace, so its measured cost understates the real
work. Here the agent reads the accessibility tree and decides for itself, so
every token is ours and visible.

Tools go straight to the model rather than into a QuickJS REPL as in
`browser-mda-repl`. `run` already executes model-authored JavaScript, so
wrapping it in an interpreter would nest a code sandbox inside a code sandbox.
"""

from managed_deepagents import define_deep_agent

from middleware.session import ReleaseBrowserSession
from middleware.tool_errors import ToolErrorsAreMessages
from tools.facade import FACADE_TOOLS

agent = define_deep_agent(
    name="browser-agent-facade",
    model="anthropic:claude-sonnet-5-5",
    tools=FACADE_TOOLS,
    # An exception out of a tool ends the whole run; a browsing failure is
    # almost always something the next turn can route around.
    middleware=[ToolErrorsAreMessages(), ReleaseBrowserSession()],
)
