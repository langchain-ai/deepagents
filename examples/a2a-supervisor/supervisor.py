"""Run a Deep Agents supervisor over remote A2A agents."""

import asyncio
import os
from urllib.parse import urlsplit

from deepagents import create_deep_agent
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from middleware import A2ASupervisorMiddleware


async def main() -> None:
    """Start an interactive supervisor with approval-gated network actions."""
    agent_url = os.environ["A2A_AGENT_URL"]
    host = urlsplit(agent_url).hostname
    if host is None:
        raise ValueError("A2A_AGENT_URL must include a host")
    middleware = A2ASupervisorMiddleware(
        agents=[agent_url],
        allowed_hosts=[host],
    )
    async with middleware:
        agent = create_deep_agent(
            model=os.environ.get("MODEL", "anthropic:claude-sonnet-4-6"),
            system_prompt="Delegate specialized work to available A2A agents and report their results.",
            middleware=[middleware],
            interrupt_on={
                "delegate_to_a2a_agent": True,
                "cancel_a2a_task": True,
            },
            checkpointer=InMemorySaver(),
        )
        config: RunnableConfig = {"configurable": {"thread_id": "a2a-supervisor"}}
        while prompt := input("> ").strip():
            result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]}, config)
            while result.get("__interrupt__"):
                requests = result["__interrupt__"][0].value["action_requests"]
                print(f"Approval required: {requests}")
                decision = "approve" if input("Approve? [y/N] ").lower() == "y" else "reject"
                result = await agent.ainvoke(
                    Command(resume={"decisions": [{"type": decision} for _ in requests]}),
                    config,
                )
            print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
