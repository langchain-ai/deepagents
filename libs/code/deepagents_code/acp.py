"""Dcode-specific ACP approval-mode adapter."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from acp.schema import PromptResponse, TextContentBlock
from deepagents_acp.server import AgentServerACP as BaseAgentServerACP
from langchain_core.messages import HumanMessage

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code.approval_mode import (
    APPROVAL_MODE_NAMESPACE,
    ApprovalMode,
    approval_mode_key,
    approval_mode_payload,
)
from deepagents_code.auto_mode import USER_PROMPT_METADATA_KEY, user_prompt_metadata

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from langchain_core.runnables import RunnableConfig
    from langgraph.pregel import Pregel
    from langgraph.store.base import BaseStore
    from langgraph.types import Command

_prompt: ContextVar[tuple[str, str] | None] = ContextVar(
    "acp_auto_prompt", default=None
)


class _AutoGraph:
    """Add trusted Auto control state to ACP graph runs."""

    def __init__(self, graph: Pregel[Any, Any, Any, Any], store: BaseStore) -> None:
        self._graph = graph
        self._store = store
        self.checkpointer = graph.checkpointer
        self._turns: dict[str, tuple[str, str]] = {}

    async def astream(
        self,
        value: dict[str, Any] | Command,
        *,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        session_id = config["configurable"]["thread_id"]
        prompt = _prompt.get()
        if prompt is not None:
            turn_id = uuid4().hex
            self._turns[session_id] = (turn_id, prompt[1])
        else:
            turn_id, _text = self._turns[session_id]
        key = approval_mode_key(session_id)
        self._store.put(
            APPROVAL_MODE_NAMESPACE,
            key,
            dict(approval_mode_payload(mode=ApprovalMode.AUTO)),
        )
        if prompt is not None and isinstance(value, dict):
            messages = list(value.get("messages", []))
            if messages:
                messages[-1] = HumanMessage(
                    content=messages[-1]["content"],
                    additional_kwargs={
                        USER_PROMPT_METADATA_KEY: user_prompt_metadata(
                            prompt[1], [], turn_id=turn_id
                        )
                    },
                )
                value = {**value, "messages": messages}
        context = CLIContextSchema(
            approval_mode=ApprovalMode.AUTO.value,
            auto_approve=True,
            approval_mode_key=key,
            thread_id=session_id,
            turn_id=turn_id,
        )
        graph = cast("Any", self._graph)
        async for chunk in graph.astream(
            value, config=config, context=context, **kwargs
        ):
            yield chunk

    async def aget_state(self, config: RunnableConfig) -> object:
        return await self._graph.aget_state(config)


class AgentServerACP(BaseAgentServerACP):
    """ACP server that supplies trusted classifier context in Auto mode."""

    def __init__(
        self,
        agent: Pregel[Any, Any, Any, Any],
        *,
        store: BaseStore,
    ) -> None:
        """Initialize the Auto-aware ACP server."""
        super().__init__(cast("Any", _AutoGraph(agent, store)))

    async def prompt(
        self,
        prompt: Sequence[Any],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        """Run an ACP prompt with trusted classifier metadata.

        Returns:
            The ACP prompt response.
        """
        text = "\n".join(
            block.text for block in prompt if isinstance(block, TextContentBlock)
        )
        token = _prompt.set((session_id, text))
        try:
            return await super().prompt(
                list(prompt), session_id, message_id=message_id, **kwargs
            )
        finally:
            _prompt.reset(token)
