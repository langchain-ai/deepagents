"""Slow agent for testing QueueLookaheadMiddleware.

This agent has a tool that sleeps, giving you time to send follow-up
messages while it's running. The middleware drains those pending messages
before each model call so the agent sees them in-context.

Start with:
    langgraph dev

Then run the test script in another terminal:
    python test_client.py
"""

import time

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from deepagents.middleware.queue_lookahead import QueueLookaheadMiddleware


@tool
def slow_think(thought: str) -> str:
    """Think slowly about something. Takes 10 seconds.

    Args:
        thought: What to think about.
    """
    print(f"[slow_think] Thinking about: {thought}")  # noqa: T201
    time.sleep(10)
    return f"Done thinking about: {thought}"


model = init_chat_model("anthropic:claude-sonnet-4-20250514")

agent = create_agent(
    model=model,
    tools=[slow_think],
    prompt="You are a helpful assistant. You MUST use the slow_think tool at least once before answering. "
    "Always acknowledge any follow-up messages that appear in the conversation.",
    middleware=[QueueLookaheadMiddleware()],
)
