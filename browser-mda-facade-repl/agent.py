"""The facade contract again, but reached by writing code instead of calling tools.

Fifth arm, and the only clean A/B in the set. `browser-mda-facade` gives the
model `snapshot`, `run` and `screenshot` directly; this gives it the *same three
tools* through `CodeInterpreterMiddleware`, so the only variable is the
invocation surface. Every earlier comparison moved the tool surface and the
invocation style together and could not separate them.

The mechanism under test is turn collapsing. Acting on an element costs the
direct arm three model turns -- `snapshot`, read the tree, `run` the action.
Here one snippet can snapshot, match the node it wants by text, and fire the
action without a second model call ever seeing the tree:

    const tree = await tools.snapshot();
    const id = tree.match(/\\[([\\d-]+)\\][^\\n]*Sign in/)?.[1];
    await tools.run({actions: [{op: "click", id}]});

Whether that is cheaper in practice is the question; `run`'s `code` form does
mean JavaScript calling a tool that runs JavaScript, which is a real cost.
"""

from langchain_quickjs import CodeInterpreterMiddleware
from managed_deepagents import define_deep_agent

from middleware.session import ReleaseBrowserSession
from middleware.tool_errors import ToolErrorsAreMessages
from tools.facade import FACADE_TOOLS

agent = define_deep_agent(
    name="browser-agent-facade-repl",
    model="anthropic:claude-sonnet-5-5",
    # No `tools=`: the facade tools go to PTC only, so the sole thing the model
    # can call is `eval`.
    middleware=[
        ToolErrorsAreMessages(),
        ReleaseBrowserSession(),
        CodeInterpreterMiddleware(
            ptc=FACADE_TOOLS,
            # A snapshot on a heavy page plus an action batch passes 30s often
            # enough that the old default tore the context down mid-call.
            timeout=180.0,
            max_result_chars=8000,
            subagents=False,
        ),
    ],
)
