"""Agent middleware: an `eval` tool whose code runs in a credential-less worker."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from qjs_worker.host import Interpreter, WorkerPool

if TYPE_CHECKING:
    from collections.abc import Sequence


class WorkerInterpreterMiddleware(AgentMiddleware):
    """Give the agent an `eval` tool backed by out-of-process QuickJS workers.

    `host_tools` are not given to the model directly: its code reaches them as
    `tools.<name>(input)`, and they run in this process with the run's config.
    Each thread keeps its own worker, and so its JavaScript state, until the
    worker is recycled.
    """

    def __init__(self, pool: WorkerPool, host_tools: Sequence[BaseTool]) -> None:
        super().__init__()
        self.pool = pool
        self.host_tools = list(host_tools)
        self.interpreters: dict[str, Interpreter] = {}
        self.tools = [self._eval_tool()]

    def _eval_tool(self) -> BaseTool:
        async def run(code: str, config: RunnableConfig) -> str:
            thread_id = config["configurable"]["thread_id"]
            if thread_id not in self.interpreters:
                self.interpreters[thread_id] = Interpreter(self.pool, self.host_tools)
            return json.dumps(asdict(await self.interpreters[thread_id].eval(code, config)))

        signatures = "\n".join(f"- tools.{t.name}({json.dumps(t.args)}): {t.description}" for t in self.host_tools)
        return StructuredTool.from_function(
            coroutine=run,
            name="eval",
            description=(
                "Run JavaScript in a sandboxed worker and return the value of the last expression. "
                "Top-level await works; console.log output is returned too. Call host tools with "
                f"`await tools.<name>({{...args}})`, and use Promise.all for parallel calls:\n{signatures}"
            ),
        )

    async def aclose(self) -> None:
        """Kill every thread's worker."""
        for interpreter in self.interpreters.values():
            await interpreter.aclose()
