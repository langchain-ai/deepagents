"""Test client that sends follow-up messages during an active run.

Usage:
    1. Start the server:  langgraph dev
    2. Run this script:   python test_client.py

The script:
    1. Creates a thread
    2. Sends an initial message (which triggers slow_think tool, ~10s)
    3. While that run is executing, enqueues a follow-up message
    4. Waits for the first run to complete
    5. Checks the final state to see if the follow-up was absorbed mid-run

If QueueLookaheadMiddleware is working, the follow-up message should appear
in the conversation BEFORE the model's final response (injected during the
before_model step after the tool call), rather than as a separate run.
"""

from __future__ import annotations

import asyncio
import time

from langgraph_sdk import get_client


async def main() -> None:
    client = get_client(url="http://localhost:2024")

    # 1. Create a thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    print(f"Created thread: {thread_id}")

    # 2. Send the initial message (starts a run with slow_think tool)
    print("\n--- Sending initial message ---")
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id="agent",
        input={"messages": [{"role": "user", "content": "Tell me about Python."}]},
        multitask_strategy="enqueue",
    )
    run_id = run["run_id"]
    print(f"Run started: {run_id}")

    # 3. Wait a few seconds for the tool to start executing, then enqueue a follow-up
    print("\nWaiting 3 seconds for tool execution to begin...")
    await asyncio.sleep(3)

    print("\n--- Sending follow-up message (should be enqueued) ---")
    followup_run = await client.runs.create(
        thread_id=thread_id,
        assistant_id="agent",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Actually, focus specifically on Python 3.12 features.",
                }
            ]
        },
        multitask_strategy="enqueue",
    )
    print(f"Follow-up enqueued as run: {followup_run['run_id']}")

    # 4. Check pending runs
    pending = await client.runs.list(thread_id=thread_id, status="pending")
    print(f"\nPending runs on thread: {len(pending)}")
    for r in pending:
        print(f"  - {r['run_id']} (status: {r['status']})")

    # 5. Wait for the first run to complete
    print("\nWaiting for first run to complete...")
    start = time.time()
    while True:
        run_status = await client.runs.get(thread_id=thread_id, run_id=run_id)
        status = run_status["status"]
        if status in ("success", "error", "interrupted", "timeout"):
            elapsed = time.time() - start
            print(f"Run finished with status: {status} ({elapsed:.1f}s)")
            break
        await asyncio.sleep(1)

    # 6. Check the final thread state
    print("\n--- Final thread state ---")
    state = await client.threads.get_state(thread_id=thread_id)
    messages = state["values"]["messages"]

    print(f"\nTotal messages: {len(messages)}")
    for i, msg in enumerate(messages):
        role = msg.get("type", msg.get("role", "unknown"))
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > 120:
            content = content[:120] + "..."
        print(f"  [{i}] {role}: {content}")

    # 7. Verify the follow-up was absorbed
    human_contents = [
        msg["content"]
        for msg in messages
        if msg.get("type") == "human" or msg.get("role") == "user"
    ]
    if "Actually, focus specifically on Python 3.12 features." in human_contents:
        print("\n[PASS] Follow-up message was absorbed into the conversation!")
    else:
        print("\n[FAIL] Follow-up message was NOT found in the conversation.")
        print("       It may be queued as a separate run instead.")

    # 8. Check if the follow-up run was cancelled (consumed by middleware)
    try:
        followup_status = await client.runs.get(
            thread_id=thread_id, run_id=followup_run["run_id"]
        )
        print(f"\nFollow-up run status: {followup_status['status']}")
        if followup_status["status"] == "interrupted":
            print("[PASS] Follow-up run was cancelled by middleware (interrupted).")
        elif followup_status["status"] == "pending":
            print("[INFO] Follow-up run is still pending (middleware may not have drained it yet).")
        else:
            print(f"[INFO] Follow-up run status: {followup_status['status']}")
    except Exception as e:  # noqa: BLE001
        print(f"\nCouldn't check follow-up run status: {e}")


if __name__ == "__main__":
    asyncio.run(main())
