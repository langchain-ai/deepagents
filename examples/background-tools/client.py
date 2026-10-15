"""Chat with the agent running under `langgraph dev` to watch background tools finish."""

import asyncio

from langgraph_sdk import get_client
from langgraph_sdk.client import LangGraphClient


async def ask(client: LangGraphClient, thread_id: str, message: str) -> None:
    """Send a message and print the agent's reply."""
    result = await client.runs.wait(thread_id, "agent", input={"messages": [{"role": "user", "content": message}]})
    print(f"\n> {message}\n{result['messages'][-1]['content']}")  # noqa: T201


async def main() -> None:
    """Start two background tools, chat while they run, then ask for results."""
    client = get_client(url="http://localhost:2024")
    thread = await client.threads.create()
    await ask(client, thread["thread_id"], "Run the benchmark with 5 steps and research the history of the transistor.")
    await ask(client, thread["thread_id"], "While those run: what's 17 * 23?")
    await asyncio.sleep(30)
    await ask(client, thread["thread_id"], "Any updates?")


if __name__ == "__main__":
    asyncio.run(main())
