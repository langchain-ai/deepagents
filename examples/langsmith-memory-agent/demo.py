"""Walk through every kind of memory against a running agent server.

Start `langgraph dev` first, or pass `--url` with a LangSmith deployment URL.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from langgraph_sdk import get_client
from langsmith import Client
from langsmith.utils import LangSmithError

from agent import APP_NAMESPACE, CONTEXT_HUB_REPO

if TYPE_CHECKING:
    from langgraph_sdk.client import LangGraphClient

ASSISTANT_ID = "memory_agent"
USERS = ("alice", "bob")


@dataclass(frozen=True)
class Step:
    """One scripted message, the thread it belongs to, and what it demonstrates."""

    user_id: str
    thread: str
    message: str
    shows: str


STEPS = (
    Step(
        "alice",
        "alice-1",
        "Hi! I'm Alice, a data engineer on the payments team. I prefer short answers with Python examples.",
        "Semantic memory: Alice's profile goes to the LangGraph store.",
    ),
    Step(
        "alice",
        "alice-1",
        "A convention for everyone on this team: incident summaries must end with a 'Follow-ups' section where every item has an owner.",
        "Procedural memory: the convention becomes a Context Hub commit.",
    ),
    Step(
        "alice",
        "alice-1",
        "Draft an on-call handoff checklist in /scratch/handoff.md, then tell me the three most important items.",
        "Short-term memory: scratch files live only in this thread.",
    ),
    Step(
        "alice",
        "alice-2",
        "New day! What do you remember about me, what did we work on last time, and is /scratch/handoff.md still there?",
        "New thread: the profile loads, past episodes are searchable, and scratch files are gone.",
    ),
    Step(
        "bob",
        "bob-1",
        "Hi, I'm Bob. What do you know about me?",
        "User isolation: Bob cannot see Alice's profile or conversations.",
    ),
    Step(
        "bob",
        "bob-1",
        "Write an incident summary: the checkout API returned 500s for 12 minutes after a bad config push.",
        "Shared procedural memory: Bob gets the curated skill plus Alice's team convention.",
    ),
)


def message_text(message: dict[str, Any]) -> str:
    """Return the plain text of a serialized message."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content or [] if isinstance(block, dict) and block.get("type") == "text")


def print_turn(messages: list[dict[str, Any]]) -> None:
    """Print the memory tool calls and final answer from the latest turn."""
    start = max(i for i, message in enumerate(messages) if message["type"] == "human")
    for message in messages[start + 1 :]:
        for call in message.get("tool_calls") or []:
            target = call["args"].get("file_path") or call["args"].get("query") or ""
            print(f"  -> {call['name']}({target})")
    print(f"\nAssistant: {message_text(messages[-1])}\n")


async def run_step(client: LangGraphClient, step: Step, threads: dict[str, str]) -> None:
    """Send one scripted message as the step's user, creating its thread on first use."""
    if step.thread not in threads:
        thread = await client.threads.create(metadata={"user_id": step.user_id})
        threads[step.thread] = thread["thread_id"]
    print(f"=== {step.user_id} / {step.thread} | {step.shows}\n{step.user_id}: {step.message}")
    state = await client.runs.wait(
        threads[step.thread],
        ASSISTANT_ID,
        input={"messages": [{"role": "user", "content": step.message}]},
        context={"user_id": step.user_id},
    )
    if not isinstance(state, dict) or "messages" not in state:
        msg = f"Run failed: {state}"
        raise RuntimeError(msg)
    print_turn(state["messages"])


async def print_store(client: LangGraphClient) -> None:
    """Print what each user has in the LangGraph store."""
    print("=== LangGraph store")
    for user_id in USERS:
        files = await client.store.search_items((APP_NAMESPACE, user_id, "files"))
        episodes = await client.store.search_items((APP_NAMESPACE, user_id, "episodes"), limit=100)
        print(f"\n[{user_id}] {len(episodes['items'])} episode(s)")
        for item in files["items"]:
            print(f"--- {item['key']}\n{item['value']['content']}")


def print_context_hub() -> None:
    """Print the latest shared agent memory from Context Hub."""
    print(f"\n=== Context Hub: {CONTEXT_HUB_REPO}")
    try:
        context = Client().pull_agent(CONTEXT_HUB_REPO)
    except LangSmithError as exc:
        print(f"Could not read Context Hub: {exc}")
        return
    print(f"Latest commit: {context.commit_hash}")
    agents_md = context.files.get("AGENTS.md")
    print(getattr(agents_md, "content", "(no AGENTS.md yet)"))


async def main() -> None:
    """Run every scripted step, then show where each memory ended up."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:2024", help="Agent server URL")
    args = parser.parse_args()

    load_dotenv()
    client = get_client(url=args.url)
    threads: dict[str, str] = {}
    for step in STEPS:
        await run_step(client, step, threads)
    await print_store(client)
    print_context_hub()


if __name__ == "__main__":
    asyncio.run(main())
