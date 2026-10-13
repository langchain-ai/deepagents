"""Browsing agent backed by Browserbase's hosted MCP server.

No sandbox: the browser lives on Browserbase and the tools arrive over MCP, so
nothing here depends on `execute`. That also makes this the one variant Harbor
can evaluate as-is, since eval trials withhold the sandbox backend.

The MCP servers come from `tools/mcp.py`, which a static agent picks up
automatically; passing `mcp=` here is only for factory entrypoints.
"""

from managed_deepagents import define_deep_agent

agent = define_deep_agent(
    name="browser-agent-mcp",
    model="anthropic:claude-sonnet-5-5",
)
