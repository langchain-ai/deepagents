"""Test user memory persistence across threads."""

import asyncio
import os

from langgraph_sdk import get_client

DEPLOY_URL = "https://deepagents-deploy-content-w-6909480a63d7575eb597d5a1b3c6e61e.us.langgraph.app"
USER_ID = "test-user-sydney"


async def run_thread(client, assistant_id, message, user_id=None, label=""):
    """Run a thread and return the final AI message."""
    thread = await client.threads.create()
    print(f"  Thread ID: {thread['thread_id']}")

    config = {}
    if user_id:
        config = {"configurable": {"user_id": user_id}}

    final_response = None
    async for event in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input={"messages": [{"role": "user", "content": message}]},
        config=config,
        stream_mode="values",
    ):
        if isinstance(event.data, dict) and "messages" in event.data:
            msgs = event.data["messages"]
            for msg in msgs:
                if isinstance(msg, dict) and msg.get("type") == "ai" and msg.get("content"):
                    content = msg["content"]
                    if isinstance(content, list):
                        # Tool use blocks
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                final_response = block["text"]
                    elif isinstance(content, str):
                        final_response = content

    return final_response


async def main():
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            for line in open(env_path):
                if line.startswith("LANGSMITH_API_KEY="):
                    api_key = line.strip().split("=", 1)[1]
                    os.environ["LANGSMITH_API_KEY"] = api_key

    client = get_client(url=DEPLOY_URL)

    assistants = await client.assistants.search()
    assistant_id = assistants[0]["assistant_id"]
    print(f"Using assistant: {assistant_id}\n")

    # --- Thread 1: Ask the agent to remember a preference ---
    print("=== Thread 1: Setting preference ===")
    resp1 = await run_thread(
        client, assistant_id,
        "I prefer concise, bullet-point style content. Please remember this preference.",
        user_id=USER_ID,
    )
    print(f"  Response (last 300 chars): ...{resp1[-300:] if resp1 else 'NONE'}\n")

    # --- Thread 2: New thread, same user — check if memory persists ---
    print("=== Thread 2: Checking memory persistence (same user) ===")
    resp2 = await run_thread(
        client, assistant_id,
        "What are my content preferences? Read your memory files and tell me.",
        user_id=USER_ID,
    )
    print(f"  Response (last 500 chars): ...{resp2[-500:] if resp2 else 'NONE'}\n")

    # --- Thread 3: Different user — should NOT see the preference ---
    print("=== Thread 3: Different user (should NOT see preference) ===")
    resp3 = await run_thread(
        client, assistant_id,
        "What are my content preferences? Read your memory files and tell me.",
        user_id="other-user-xyz",
    )
    print(f"  Response (last 500 chars): ...{resp3[-500:] if resp3 else 'NONE'}\n")

    # --- Thread 4: No user_id — should gracefully skip user memory ---
    print("=== Thread 4: No user_id (should skip user memory gracefully) ===")
    try:
        resp4 = await run_thread(
            client, assistant_id,
            "Hello, just say hi back briefly.",
        )
        print(f"  Response: {resp4[:200] if resp4 else 'NONE'}")
        print("  SUCCESS: No user_id handled gracefully\n")
    except Exception as e:
        print(f"  ERROR with no user_id: {e}\n")

    print("=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
