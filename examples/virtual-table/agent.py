"""Run the virtual-table middleware prototype over example customer feedback."""

from __future__ import annotations

import argparse
import asyncio
import json

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langgraph.graph.state import CompiledStateGraph

from virtual_table import VirtualTableMiddleware

FEEDBACK = [
    {"file": "/feedback/acme.txt", "customer": "Acme", "plan": "enterprise"},
    {"file": "/feedback/beacon.txt", "customer": "Beacon", "plan": "starter"},
    {"file": "/feedback/cedar.txt", "customer": "Cedar", "plan": "enterprise"},
    {"file": "/feedback/delta.txt", "customer": "Delta", "plan": "pro"},
    {"file": "/feedback/elm.txt", "customer": "Elm", "plan": "pro"},
]

FILES = {
    "/feedback/acme.txt": "The dashboard is fast, but exports time out every Friday.",
    "/feedback/beacon.txt": "Setup was easy. I wish the API docs had more Python examples.",
    "/feedback/cedar.txt": "Support fixed our SSO issue quickly, although audit logs are hard to search.",
    "/feedback/delta.txt": "The new search is excellent and cut our investigation time in half.",
    "/feedback/elm.txt": "Billing pages sometimes show stale usage numbers.",
}

DEFAULT_QUESTION = "Classify each feedback item by sentiment and product area, then count feedback by plan, sentiment, and product area."


def create_agent(model: str) -> CompiledStateGraph:
    """Create the prototype agent."""
    backend = StateBackend()
    return create_deep_agent(
        model=model,
        backend=backend,
        middleware=[VirtualTableMiddleware(backend=backend, initial_tables={"feedback": FEEDBACK})],
    )


async def main() -> None:
    """Run one analysis request and print the final answer and materialized rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--model", default="anthropic:claude-sonnet-4-6")
    args = parser.parse_args()

    agent = create_agent(args.model)
    files = {path: {"content": content, "encoding": "utf-8"} for path, content in FILES.items()}
    result = await agent.ainvoke({"messages": [{"role": "user", "content": args.question}], "files": files})
    print(result["messages"][-1].text)
    print("\nMaterialized feedback table:")
    print(json.dumps(result.get("_virtual_tables", {}).get("feedback", []), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
