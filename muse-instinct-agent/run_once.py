"""Run one task against the live deployment and print the full message trail."""

from __future__ import annotations

import asyncio
import os
import sys

from langgraph_sdk import get_client

URL = "https://agentic-shopper-e2243e654a5c5858a80dcc090c681dc2.us.langgraph.app"
GRAPH = "agentic-shopper"


async def main() -> None:
    key = os.environ["LANGSMITH_API_KEY"]
    client = get_client(url=URL, api_key=key)

    which = sys.argv[1] if len(sys.argv) > 1 else "sandbox"
    if which == "sandbox":
        prompt = (
            "Do NOT make any purchase. Just verify your payment tooling: run "
            "`link-cli --version` and `link-cli auth status` in your sandbox "
            "and report exactly what they print. Do not call `link-cli auth login`."
        )
    else:
        prompt = (
            "Go to news.ycombinator.com and tell me the titles of the top 3 "
            "stories. Do not make any purchase."
        )

    thread = await client.threads.create()
    tid = thread["thread_id"]
    print(f"thread={tid}  task={which}")

    run = await client.runs.create(
        tid, GRAPH, input={"messages": [{"role": "user", "content": prompt}]}
    )
    result = await client.runs.join(tid, run["run_id"])

    state = await client.threads.get_state(tid)
    msgs = state["values"].get("messages", [])
    print(f"--- {len(msgs)} messages ---")
    for m in msgs:
        role = m.get("type") or m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in content
            )
        tcs = m.get("tool_calls") or []
        line = f"[{role}] {str(content)[:1500]}"
        for tc in tcs:
            line += f"\n    ->tool {tc.get('name')}({str(tc.get('args'))[:300]})"
        print(line)
    print("--- run status:", run.get("status"), "->", (result or {}).get("status", "joined"))


if __name__ == "__main__":
    asyncio.run(main())
