"""Subagent refusals are reported to the parent as task results."""

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRefusalMiddleware
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from deepagents.graph import create_deep_agent
from deepagents.middleware.async_subagents import _final_content
from deepagents.middleware.subagents import CompiledSubAgent
from tests.unit_tests.chat_model import GenericFakeChatModel

RETRY_HINT = " Retrying the same request is unlikely to succeed; rephrase it or continue without it."

ANTHROPIC_REFUSAL = AIMessage(
    content=[],
    response_metadata={
        "model_provider": "anthropic",
        "stop_reason": "refusal",
        "stop_details": {"type": "refusal", "category": "cyber", "explanation": "Blocked."},
    },
)
ANTHROPIC_NOTICE = f'The subagent stopped because a model refused the request. Category: cyber. Explanation: "Blocked."{RETRY_HINT}'

BARE_REFUSAL = AIMessage(content=[], response_metadata={"model_provider": "anthropic", "stop_reason": "refusal"})
BARE_NOTICE = f"The subagent stopped because a model refused the request.{RETRY_HINT}"

OPENAI_REFUSAL = AIMessage(
    content="",
    additional_kwargs={"refusal": "I can't help with that."},
    response_metadata={"model_provider": "openai"},
)
OPENAI_NOTICE = f'The subagent stopped because a model refused the request. Explanation: "I can\'t help with that."{RETRY_HINT}'


class _Report(BaseModel):
    summary: str


def _parent_model() -> GenericFakeChatModel:
    return GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {"description": "Scan this code", "subagent_type": "scanner"},
                            "id": "call_scan",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="The scan could not be completed."),
            ]
        )
    )


async def _task_results(subagent: object, *, asynchronous: bool) -> list[str]:
    agent = create_deep_agent(
        model=_parent_model(),
        subagents=[CompiledSubAgent(name="scanner", description="Scans code.", runnable=subagent)],
    )
    inputs = {"messages": [HumanMessage("Scan my code")]}
    result = await agent.ainvoke(inputs) if asynchronous else agent.invoke(inputs)
    assert result["messages"][-1].content == "The scan could not be completed."
    return [message.content for message in result["messages"] if message.type == "tool"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("raises", [False, True], ids=["returns-refusal", "raises-refusal"])
@pytest.mark.parametrize(
    ("refusal", "notice"),
    [(ANTHROPIC_REFUSAL, ANTHROPIC_NOTICE), (BARE_REFUSAL, BARE_NOTICE), (OPENAI_REFUSAL, OPENAI_NOTICE)],
    ids=["anthropic", "bare", "openai"],
)
async def test_subagent_refusal_becomes_task_result(*, refusal: AIMessage, notice: str, asynchronous: bool, raises: bool) -> None:
    middleware: list[AgentMiddleware] = [ModelRefusalMiddleware()] if raises else []
    subagent = create_agent(GenericFakeChatModel(messages=iter([refusal])), middleware=middleware)
    assert await _task_results(subagent, asynchronous=asynchronous) == [notice]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_structured_subagent_refusal_becomes_task_result(*, asynchronous: bool) -> None:
    subagent = create_agent(
        GenericFakeChatModel(messages=iter([ANTHROPIC_REFUSAL])),
        response_format=ProviderStrategy(_Report),
    )
    assert await _task_results(subagent, asynchronous=asynchronous) == [ANTHROPIC_NOTICE]


def test_async_subagent_result_reports_refusal() -> None:
    assert _final_content({"messages": [ANTHROPIC_REFUSAL]}) == [{"type": "text", "text": ANTHROPIC_NOTICE}]
