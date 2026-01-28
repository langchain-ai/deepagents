"""Regression test for missing system prompt in SubAgentMiddleware."""

from langchain.agents import create_agent
from collections.abc import Callable, Sequence

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from deepagents.middleware.subagents import SubAgentMiddleware


class FixedGenericChatModel(GenericFakeChatModel):
    """Generic chat model with tool binding for tests."""

    def bind_tools(
        self,
        _tools: Sequence[dict[str, object] | type | Callable | BaseTool],
        *,
        _tool_choice: str | None = None,
        **_kwargs: object,
    ) -> Runnable:
        return self


def test_subagent_middleware_missing_system_prompt() -> None:
    """Ensure SubAgentMiddleware handles requests without `system_prompt`."""
    model = FixedGenericChatModel(messages=iter([AIMessage(content="ok")]))

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[
            SubAgentMiddleware(
                default_model=model,
                default_tools=[],
                subagents=[],
            )
        ],
    )

    result = agent.invoke({"messages": [HumanMessage(content="hello")]})

    assert "messages" in result
    assert result["messages"][-1].content == "ok"
