# ruff: noqa: INP001, D100, D103, ANN201, PLE1205, E501
import asyncio
from logging import getLogger

from langgraph_sdk import get_client

logger = getLogger(__name__)


async def main():
    client = get_client(url="http://localhost:2024")

    thread = await client.threads.create(graph_id="agent")

    thread_id = thread.get("thread_id")
    logger.info("\n\nthread_id", thread_id, "\n\n")

    stream = client.runs.stream(
        thread_id,
        "agent",
        stream_mode="values",
        config={"configurable": {"repo": {"owner": "langchain-ai", "name": "open-swe"}}},
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "The repo open-swe is already cloned. Can you please edit the readme to add a deprecation notice? Say something along the lines of this repo is deprecated and won't be updated anymore.",
                }
            ]
        },
    )

    async for update in stream:
        messages = update.data.get("messages")
        last_message = messages[-1] if messages else None
        logger.info("\n\nUpdate\n%s\n\n", last_message)


if __name__ == "__main__":
    asyncio.run(main())
