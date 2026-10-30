"""A Managed Deep Agent that browses with Vercel's agent-browser CLI.

Same shape as the `browse` CLI arm, and for the same reason: MDA tools run in
the Agent Server process while the sandbox is a separate box, so a browser
launched from this file would not be the sandbox's browser. `sandbox/setup.sh`
bakes Google Chrome and `agent-browser` into the snapshot instead, and the
agent drives them through the built-in `execute` shell tool.

That also makes this the only way to benchmark agent-browser without a browser
on the machine running the suite -- the CI runs of this arm installed Chrome on
the runner, which worked but is not the same environment as the other arms.

The sandbox is picked up from `sandbox/__init__.py`; behaviour lives in
`instructions.md`.
"""

from managed_deepagents import define_deep_agent

agent = define_deep_agent(
    name="browser-agent-agent-browser",
    model="anthropic:claude-sonnet-5-5",
)
