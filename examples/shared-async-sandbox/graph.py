"""Parent and remote worker sharing a sandbox, not a checkpoint thread."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langsmith.sandbox import ResourceNotFoundError, SandboxClient

from deepagents import create_deep_agent
from deepagents.backends import LangSmithSandbox


def sandbox_name(thread_id: str) -> str:
    """Map an Agent Server thread UUID to a stable sandbox name."""
    return f"shared-{UUID(thread_id).hex}"


@asynccontextmanager
async def parent(config: RunnableConfig) -> AsyncIterator[CompiledStateGraph]:
    """Create or reconnect the parent's sandbox for this run."""
    name = sandbox_name(config["configurable"]["thread_id"])
    with SandboxClient() as client:
        try:
            sandbox = client.get_sandbox(name)
        except ResourceNotFoundError:
            sandbox = client.create_sandbox(
                snapshot_name=os.environ["SANDBOX_SNAPSHOT"], name=name
            )
        backend = LangSmithSandbox(sandbox=sandbox)
        try:
            yield create_deep_agent(
                backend=backend,
                system_prompt="Delegate background file work to worker. Report task IDs; check results when asked.",
                subagents=[
                    {
                        "name": "worker",
                        "description": "Works in your shared sandbox in a separate background run.",
                        "graph_id": "worker",
                        "url": os.environ.get(
                            "AGENT_SERVER_URL", "http://localhost:2024"
                        ),
                    }
                ],
            )
        finally:
            await backend.aclose()


@asynccontextmanager
async def worker(config: RunnableConfig) -> AsyncIterator[CompiledStateGraph]:
    """Attach using the parent's identity while retaining the worker's thread."""
    name = sandbox_name(config["configurable"]["parent_thread_id"])
    with SandboxClient() as client:
        backend = LangSmithSandbox(sandbox=client.get_sandbox(name))
        try:
            yield create_deep_agent(
                backend=backend,
                system_prompt="Complete the delegated task in the shared sandbox. Do not overwrite unrelated files.",
            )
        finally:
            await backend.aclose()
