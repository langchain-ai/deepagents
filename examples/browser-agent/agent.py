"""A deep agent that browses, over a sandbox backend.

Two browser providers, one agent. `BROWSER=local` launches Chrome on this
machine; anything else uses a hosted Browserbase session. The agent's
filesystem and `execute` tool come from a sandbox backend either way, so the
browser is the only thing that changes.

    uv run --with "deepagents[browser]" python agent.py "find the top story on hn"
    BROWSER=local uv run --with "deepagents[browser]" python agent.py "..."
"""

import asyncio
import os
import sys

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from deepagents.middleware.browser import Browserbase, BrowserMiddleware, LocalChrome

INSTRUCTIONS = """You browse with three tools.

1. `run` with `code` to navigate: `await page.goto("https://example.com")`.
2. `snapshot` to read the page -- it returns an accessibility tree where every
   interactive element has a bracketed ID.
3. `run` with `{"actions": [{"op": "click", "id": "1-42"}]}` to act on those IDs.
   Re-snapshot afterwards; old IDs go stale.

`run` with `code` is usually cheapest -- the snippet picks what comes back:

    await page.goto("https://news.ycombinator.com");
    return await page.locator(".titleline a").first().innerText();

`page` and `context` are Playwright-shaped, so `goto`, `locator`, `title` and
friends all work, and the whole snippet is one round trip.

Answer from what the page actually showed. If a site blocks you, say so. Never
guess a value you did not read."""


def build():
    provider = LocalChrome() if os.environ.get("BROWSER") == "local" else Browserbase()
    return create_deep_agent(
        model="anthropic:claude-sonnet-5",
        system_prompt=INSTRUCTIONS,
        backend=LocalShellBackend(),
        middleware=[BrowserMiddleware(provider)],
    )


async def main() -> None:
    task = " ".join(sys.argv[1:]) or "Go to https://example.com and tell me the page title."
    result = await build().ainvoke(
        {"messages": [{"role": "user", "content": task}]},
        config={"configurable": {"thread_id": "browser-example"}, "recursion_limit": 100},
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
