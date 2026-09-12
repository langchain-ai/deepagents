"""Smoke-test the deployed agentic-shopper against its live Agent Server.

Runs two tasks:
  1. Payments/sandbox check — confirm the Link CLI is installed and reachable
     in the sandbox (no real transaction; just `link-cli --version` + auth status).
  2. Browser-use — a small real Stagehand browse via Browserbase.

Auth: the deployment uses LangSmith-API-key identity, so the client sends the
key as `x-api-key`. Reads LANGSMITH_API_KEY from the environment.
"""

from __future__ import annotations

import asyncio
import os
import sys

from langgraph_sdk import get_client

URL = "https://agentic-shopper-e2243e654a5c5858a80dcc090c681dc2.us.langgraph.app"
GRAPH = "agentic-shopper"


async def run_task(client, prompt: str) -> str:
    thread = await client.threads.create()
    final = ""
    async for chunk in client.runs.stream(
        thread["thread_id"],
        GRAPH,
        input={"messages": [{"role": "user", "content": prompt}]},
        stream_mode="values",
    ):
        if chunk.event == "values" and isinstance(chunk.data, dict):
            msgs = chunk.data.get("messages") or []
            if msgs:
                final = msgs[-1].get("content", "")
    return final if isinstance(final, str) else str(final)


async def main() -> None:
    key = os.environ.get("LANGSMITH_API_KEY")
    if not key:
        raise SystemExit("LANGSMITH_API_KEY is required.")
    client = get_client(url=URL, api_key=key)

    which = sys.argv[1] if len(sys.argv) > 1 else "sandbox"
    if which == "sandbox":
        prompt = (
            "Do NOT make any purchase. Just verify your payment tooling: run "
            "`link-cli --version` and `link-cli auth status` in your sandbox "
            "and report exactly what they print, including whether you are "
            "authenticated. Do not call `link-cli auth login`."
        )
    else:
        prompt = (
            "Go to news.ycombinator.com and tell me the titles of the top 3 "
            "stories. Do not make any purchase."
        )

    print(f"--- task: {which} ---")
    print(await run_task(client, prompt))


if __name__ == "__main__":
    asyncio.run(main())
