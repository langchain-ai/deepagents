"""A deep agent whose code runs in a credential-less QuickJS worker.

The agent is composed as usual: `create_deep_agent` with
`CodeInterpreterMiddleware`, whose `slot_factory` runs the model's JavaScript in
a separate worker process with an empty environment, seccomp and recycling.
The ticket tools need a host-only API token and check the calling user's
permissions, so they stay in this process; the model's code reaches them as
`tools.listTickets(...)` / `tools.getAsset(...)`.

Uses `anthropic:claude-sonnet-4-6` when `ANTHROPIC_API_KEY` is set (via the
environment or a `.env`), and a scripted stand-in model otherwise.

Usage:
    uv run python agent.py
"""

import asyncio
import os

from deepagents import create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import ToolException, tool
from langchain_quickjs import CodeInterpreterMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from qjs_worker.host import WorkerConfig, WorkerPool

MODEL = "anthropic:claude-sonnet-4-6"
TASK = "Summarize my open tickets with the name and severity of each affected asset. Use one eval call."

TICKETS = [
    {"id": "T-101", "asset_id": "a-1", "title": "TLS cert expires in 5 days", "status": "open"},
    {"id": "T-102", "asset_id": "a-2", "title": "Outdated OpenSSH", "status": "open"},
    {"id": "T-103", "asset_id": "a-3", "title": "Open S3 bucket", "status": "open"},
    {"id": "T-104", "asset_id": "a-1", "title": "Weak cipher suites", "status": "closed"},
]
ASSETS = {
    "a-1": {"name": "payments-api", "severity": "critical", "team": "payments"},
    "a-2": {"name": "hr-portal", "severity": "high", "team": "people"},
    "a-3": {"name": "payments-exports", "severity": "high", "team": "payments"},
}
USER_TEAMS = {"alice": "payments"}


def require_token() -> None:
    # A real tool would call an API with this. It exists only in the host:
    # the worker running the model's code starts with an empty environment.
    if not os.environ.get("TICKETS_API_TOKEN"):
        msg = "TICKETS_API_TOKEN is not set"
        raise ToolException(msg)


@tool
async def list_tickets(status: str) -> list[dict]:
    """List security tickets with the given status ("open" or "closed")."""
    require_token()
    return [ticket for ticket in TICKETS if ticket["status"] == status]


@tool
async def get_asset(asset_id: str, config: RunnableConfig) -> dict:
    """Get an asset's name and severity. Only assets owned by the caller's team are visible."""
    require_token()
    user = config["configurable"]["user_id"]
    asset = ASSETS[asset_id]
    if asset["team"] != USER_TEAMS[user]:
        msg = f"{user} is not allowed to read asset {asset_id}"
        raise ToolException(msg)
    return asset


SCRIPTED_CODE = """\
const tickets = await tools.listTickets({ status: "open" });
const assets = await Promise.allSettled(tickets.map((t) => tools.getAsset({ asset_id: t.asset_id })));
tickets.map((t, i) => ({
  ticket: `${t.id} ${t.title}`,
  asset: assets[i].status === "fulfilled" ? assets[i].value : assets[i].reason.message,
}))"""


class ScriptedModel(GenericFakeChatModel):
    """Stand-in model for runs without an API key: one `eval` call, then an answer."""

    def bind_tools(self, tools: object, **kwargs: object) -> "ScriptedModel":
        return self


def scripted_model() -> ScriptedModel:
    return ScriptedModel(
        messages=iter(
            [
                AIMessage("", tool_calls=[{"name": "eval", "args": {"code": SCRIPTED_CODE}, "id": "call_1"}]),
                AIMessage("(scripted model) Your open tickets and assets are in the eval result above."),
            ]
        )
    )


async def main() -> None:
    load_dotenv(find_dotenv())
    os.environ.setdefault("TICKETS_API_TOKEN", "demo-token")  # injected by the deployment in real life
    model = MODEL if os.environ.get("ANTHROPIC_API_KEY") else scripted_model()
    print(f"model: {MODEL if isinstance(model, str) else 'scripted stand-in (no ANTHROPIC_API_KEY)'}")

    pool = WorkerPool(WorkerConfig(), size=2)
    await pool.start()
    agent = create_deep_agent(
        model=model,
        tools=[list_tickets, get_asset],
        middleware=[
            CodeInterpreterMiddleware(ptc=["list_tickets", "get_asset"], subagents=False, slot_factory=pool.slot)
        ],
        checkpointer=InMemorySaver(),
    )

    config: RunnableConfig = {"configurable": {"thread_id": "demo", "user_id": "alice"}}
    result = await agent.ainvoke({"messages": [{"role": "user", "content": TASK}]}, config=config)
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                print(f"\n[{call['name']}]\n{call['args'].get('code', call['args'])}")
            if message.content:
                print(f"\n[answer]\n{message.text}")
        elif isinstance(message, ToolMessage):
            print(f"\n[{message.name} result]\n{message.content}")
    worker = pool.slots[0].worker
    print(f"\nthe code ran in worker {worker.ready}")

    await pool.aclose()


if __name__ == "__main__":
    asyncio.run(main())
