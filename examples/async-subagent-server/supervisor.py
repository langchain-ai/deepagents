"""Supervisor — Async Subagent Example.

An interactive REPL that demonstrates the five async subagent operations
against the FastAPI server in server.py.

The supervisor delegates research tasks to the server-hosted researcher
via Agent Protocol (through the LangGraph SDK). Tasks run in the
background — the supervisor returns a task ID immediately and lets you
check in when you're ready.

Run (after starting server.py in another terminal):
    ANTHROPIC_API_KEY=... python supervisor.py

Try these prompts:
    > research the latest developments in quantum computing
    > check status of <task-id>
    > update <task-id> to focus on commercial applications only
    > cancel <task-id>
    > list all tasks
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from deepagents import create_deep_agent
from deepagents.middleware.async_subagents import AsyncSubAgent

load_dotenv(Path(__file__).parent / ".env")

import os  # noqa: E402

RESEARCHER_URL = os.environ.get("RESEARCHER_URL", "http://localhost:2024")

# ── Agent setup ───────────────────────────────────────────────────────────────

async_subagents: list[AsyncSubAgent] = [
    {
        "name": "researcher",
        "description": (
            "A research agent that investigates any topic using web search. "
            "Runs in the background and returns a detailed summary."
        ),
        "graph_id": "researcher",
        "url": RESEARCHER_URL,
        "headers": {"x-auth-scheme": "custom"},
    },
]

checkpointer = MemorySaver()
thread_id = str(uuid.uuid4())

supervisor = create_deep_agent(
    model=ChatAnthropic(model="claude-sonnet-4-5"),
    checkpointer=checkpointer,
    system_prompt=(
        "You are a research supervisor coordinating a background researcher agent.\n\n"
        "For general questions, answer directly — do NOT launch a researcher.\n\n"
        'Only launch the researcher when the user says "research", "investigate", "look into", or "find out".\n\n'
        "START: When the user asks to research something:\n"
        '  1. Call start_async_task with subagent_type "researcher" and the topic.\n'
        "  2. Report the task_id and stop. Do NOT immediately check status.\n\n"
        "CHECK: When the user asks for status or results:\n"
        "  1. Call check_async_task with the exact task_id.\n"
        "  2. Report what the tool returns. If still running, say so and stop.\n\n"
        "UPDATE: When the user asks to change what the researcher is working on:\n"
        "  1. Call update_async_task with the task_id and new instructions.\n"
        "  2. Confirm the update.\n\n"
        "CANCEL: When the user asks to cancel a task:\n"
        "  1. Call cancel_async_task with the exact task_id.\n"
        "  2. Confirm the cancellation.\n\n"
        "LIST: When the user asks to list tasks or check all statuses:\n"
        "  1. Call list_async_tasks.\n"
        "  2. Present the live statuses.\n\n"
        "Rules:\n"
        "- Never report a stale status from memory. Always call a tool.\n"
        "- Never poll in a loop. One tool call per user request.\n"
        "- Always show the full task_id — never truncate it."
    ),
    subagents=async_subagents,
)


# ── REPL ──────────────────────────────────────────────────────────────────────

async def chat(user_input: str) -> None:
    """Send a message to the supervisor and print the response."""
    result = await supervisor.ainvoke(
        {"messages": [HumanMessage(user_input)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    last = result["messages"][-1]
    content = last.content
    print(
        "\n"
        + (content if isinstance(content, str) else __import__("json").dumps(content, indent=2))
        + "\n"
    )


async def main() -> None:
    """Run the interactive REPL."""
    print(f"Supervisor connected to researcher at {RESEARCHER_URL}")
    print("Type a message and press Enter. Ctrl+C or Ctrl+D to exit.\n")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not user_input:
            continue
        try:
            await chat(user_input)
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
