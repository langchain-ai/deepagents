"""Minimal agent runtime used to bootstrap the Talon host."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, cast

from deepagents import create_deep_agent

from deepagents_talon.interfaces import AgentRequest, AgentResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool


class EchoAgentRuntime:
    """Small placeholder runtime for host bootstrapping and tests."""

    async def start(self) -> None:
        """Initialize the placeholder runtime."""

    async def stop(self) -> None:
        """Release placeholder runtime resources."""

    async def invoke(self, request: AgentRequest) -> AgentResult:
        """Return the request text as a trivial agent response.

        Args:
            request: Agent request supplied by the Talon host.

        Returns:
            Echo response tagged as placeholder runtime output.
        """
        return AgentResult(text=request.text)


class DeepAgentRuntime:
    """Deep Agents-backed runtime for Talon.

    Args:
        model: Chat model identifier for `create_deep_agent`.
        tools: Tools exposed to the agent.
        system_prompt: Optional system prompt from the materialized manifest.
    """

    def __init__(
        self,
        *,
        model: str,
        tools: Sequence[BaseTool] = (),
        system_prompt: str | None = None,
    ) -> None:
        """Initialize without constructing the graph."""
        self.model = model
        self.tools = tuple(tools)
        self.system_prompt = system_prompt
        self._graph: object | None = None

    async def start(self) -> None:
        """Construct the Deep Agents graph."""
        self._graph = create_deep_agent(
            model=self.model,
            tools=list(self.tools),
            system_prompt=self.system_prompt,
        )

    async def stop(self) -> None:
        """Release runtime resources."""
        self._graph = None

    async def invoke(self, request: AgentRequest) -> AgentResult:
        """Invoke the Deep Agents graph for one Talon request.

        Args:
            request: Agent request supplied by the Talon host.

        Returns:
            Final assistant text from the graph.

        Raises:
            RuntimeError: If the runtime has not been started.
        """
        if self._graph is None:
            msg = "DeepAgentRuntime must be started before invoke"
            raise RuntimeError(msg)

        ainvoke = getattr(self._graph, "ainvoke", None)
        if not callable(ainvoke):
            msg = "Deep Agents graph does not expose async invocation"
            raise TypeError(msg)

        invoke = cast("Callable[..., Awaitable[object]]", ainvoke)
        state = await invoke(
            {"messages": [{"role": "user", "content": request.text}]},
            config={"configurable": {"thread_id": request.conversation_id}},
        )
        return AgentResult(text=_last_text(state))


def _last_text(state: object) -> str:
    if not isinstance(state, Mapping):
        return ""
    data = cast("Mapping[str, object]", state)
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    content = getattr(messages[-1], "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_content_block_text(block) for block in content).strip()
    return ""


def _content_block_text(block: object) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, Mapping):
        data = cast("Mapping[str, object]", block)
        text = data.get("text")
        if isinstance(text, str):
            return text
    return ""
