"""A Managed Deep Agent that browses the web from inside its own sandbox.

The browser is NOT a tool defined here. MDA tools run in the Agent Server
process, while the sandbox is a separate box; a browser launched from this
file would not be the sandbox's browser. Instead `sandbox/setup.sh` bakes
Google Chrome and the Browserbase `browse` CLI into the sandbox snapshot, and the
agent drives them through the sandbox's built-in `execute` shell tool. That is
what makes the browser genuinely local to the sandbox -- no Browserbase cloud
session, no BROWSERBASE_API_KEY.

Behaviour lives in `instructions.md` and `skills/web-browsing/SKILL.md`.
"""

from managed_deepagents import define_deep_agent

agent = define_deep_agent(
    name="browser-agent",
    model="anthropic:claude-sonnet-5-5",
)
