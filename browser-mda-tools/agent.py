"""Browsing agent that drives the sandbox browser through typed tools.

Same browser as the `browse`-CLI variant -- a real headless Chrome baked into
this project's sandbox by `sandbox/setup.sh`. The difference is the surface the
model sees. Here it calls `navigate(url)` / `click(ref)` / `read_page()` instead
of composing shell strings, and the tools do the quoting, JSON parsing, and
output trimming that the model otherwise spends steps on.

The tools run in the Agent Server and reach the sandbox through
`runtime.backend.aexecute`, so the browser still lives in the sandbox.
"""

from managed_deepagents import define_deep_agent

from tools.browser import BROWSER_TOOLS

agent = define_deep_agent(
    name="browser-agent-tools",
    model="anthropic:claude-sonnet-5-5",
    tools=BROWSER_TOOLS,
)
