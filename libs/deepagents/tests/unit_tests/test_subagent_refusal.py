"""Subagent refusals are reported to the parent as task results."""

from typing import TypedDict
from unittest.mock import patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelFallbackMiddleware, ModelRefusalMiddleware
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.exceptions import ModelRefusalError
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START, StateGraph
from pydantic import BaseModel

from deepagents.graph import create_deep_agent
from deepagents.middleware.async_subagents import _final_content
from deepagents.middleware.subagents import CompiledSubAgent
from deepagents.profiles import HarnessProfile
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


class _PlainMessagesState(TypedDict):
    messages: list[dict[str, str]]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_subagent_with_plain_dict_messages_is_not_a_refusal(*, asynchronous: bool) -> None:
    graph = StateGraph(_PlainMessagesState)
    graph.add_node("reply", lambda _: {"messages": [{"role": "assistant", "content": "Done."}]})
    graph.add_edge(START, "reply")
    results = await _task_results(graph.compile(), asynchronous=asynchronous)
    assert len(results) == 1
    assert "refused" not in results[0]


def _raise_content_filter(_: _PlainMessagesState) -> None:
    msg = "Model output was stopped by a content filter"
    raise ModelRefusalError(msg, raw_message=AIMessage(content=""))


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_refusal_error_without_block_keeps_its_message(*, asynchronous: bool) -> None:
    graph = StateGraph(_PlainMessagesState)
    graph.add_node("generate", _raise_content_filter)
    graph.add_edge(START, "generate")
    notice = f'The subagent stopped because a model refused the request. Explanation: "Model output was stopped by a content filter"{RETRY_HINT}'
    assert await _task_results(graph.compile(), asynchronous=asynchronous) == [notice]


def _ask(agent: object) -> list[object]:
    return agent.invoke({"messages": [HumanMessage("Scan my code")]})["messages"]  # type: ignore[attr-defined]


def test_main_agent_raises_on_refusal_by_default() -> None:
    agent = create_deep_agent(model=GenericFakeChatModel(messages=iter([ANTHROPIC_REFUSAL])))
    with pytest.raises(ModelRefusalError):
        _ask(agent)


def test_fallback_middleware_catches_default_refusal() -> None:
    fallback = GenericFakeChatModel(messages=iter([AIMessage(content="Scan complete.")]))
    agent = create_deep_agent(
        model=GenericFakeChatModel(messages=iter([ANTHROPIC_REFUSAL])),
        middleware=[ModelFallbackMiddleware(fallback)],
    )
    assert _ask(agent)[-1].content == "Scan complete."  # type: ignore[attr-defined]


def test_refusal_middleware_can_be_excluded() -> None:
    profile = HarnessProfile(excluded_middleware=frozenset({"ModelRefusalMiddleware"}))
    with patch("deepagents.graph._harness_profile_for_model", return_value=profile):
        agent = create_deep_agent(model=GenericFakeChatModel(messages=iter([ANTHROPIC_REFUSAL])))
    assert _ask(agent)[-1].response_metadata["stop_reason"] == "refusal"  # type: ignore[attr-defined]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_declarative_subagent_refusal_becomes_task_result(*, asynchronous: bool) -> None:
    agent = create_deep_agent(
        model=_parent_model(),
        subagents=[
            {
                "name": "scanner",
                "description": "Scans code.",
                "system_prompt": "Scan code.",
                "model": GenericFakeChatModel(messages=iter([ANTHROPIC_REFUSAL])),
            }
        ],
    )
    inputs = {"messages": [HumanMessage("Scan my code")]}
    result = await agent.ainvoke(inputs) if asynchronous else agent.invoke(inputs)
    assert [m.content for m in result["messages"] if m.type == "tool"] == [ANTHROPIC_NOTICE]
