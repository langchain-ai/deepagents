"""Managed Deep Agent: an agentic shopper.

Browses the web with Stagehand (Browserbase) and checks out with the Stripe
Link CLI, which runs inside the MDA sandbox. MDA supplies the harness and
hosted runtime; this file only wires the agent's model and tools.

Layout (Managed Deep Agents project):
- agent.py            -> this file, exports `agent`
- instructions.md     -> system prompt (synced to Context Hub)
- tools/stagehand.py  -> official Stagehand<->Deep Agents tools (run/snapshot/screenshot),
                         vendored from browserbase/stagehand
                         packages/integrations/deepagents/examples/managed
- tools/search.py     -> Tavily web search as an authored tool
- sandbox/            -> installs the Link CLI (setup.sh) + define_sandbox
- identity.py         -> caller identity (LangSmith API key)
- pyproject.toml/.env -> deps and secrets

The model is served through the LangSmith LLM Gateway (OpenAI-compatible
endpoint) so it authenticates with the workspace's configured provider secret
via a LangSmith key — no separate Anthropic key needed. Because that endpoint
does not expose a provider-native web-search block, search is an authored tool
backed by Browserbase's hosted search — the same API key as the browser, and
no browser session consumed.

Payments run through the sandbox's `execute` tool (built in), not over MCP:
a loopback MCP server is not reachable from the hosted runtime, whereas the
Link CLI installed in the sandbox is co-located with the agent.
"""

from __future__ import annotations

from managed_deepagents import define_deep_agent

from tools.link import link_user_info
from tools.search import fetch_page, web_search
from tools.stagehand import run, screenshot, snapshot

# Model is served through the LangSmith LLM Gateway's Anthropic-native endpoint.
# langchain-anthropic reads ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY from the env
# (both set in .env / forwarded as deployment secrets): the base URL points at
# the gateway and the "key" is the LangSmith gateway key, which the gateway
# swaps for the workspace's configured Anthropic secret. Using the `anthropic:`
# provider keeps langchain-anthropic (already a dep) as the only model package.
agent = define_deep_agent(
    name="agentic-shopper",
    model="anthropic:anthropic/claude-sonnet-4-6",
    tools=[web_search, fetch_page, run, snapshot, screenshot, link_user_info],
)
