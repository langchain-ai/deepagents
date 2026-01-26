import asyncio

from langgraph_sdk import get_client


async def main():
    client = get_client(url="http://localhost:2024")

    # thread = await client.threads.create(graph_id="agent")

    # thread_id = thread.get("thread_id")
    thread_id = "cd05c990-858f-4ddb-92ac-37437d75bec6"
    print(f"\n\nthread_id\n{thread_id}\n\n")

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
                    # "content": "The repo open-swe is already cloned. Can you please run git status and tell me the output?",
                }
            ]
        },
    )

    async for update in stream:
        messages = update.data.get("messages")
        last_message = messages[-1] if messages else None
        print(f"\n---Update---\n{last_message}\n---")

    print("\n\nDone streaming updates\n\n")


if __name__ == "__main__":
    asyncio.run(main())
