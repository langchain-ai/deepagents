"""Media from `tools.*` calls reaches the model with the eval's result."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_quickjs import CodeInterpreterMiddleware
from tests._common import FakeChatModel

CALLS: list[str] = []
PAUSES: list[Any] = []


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    CALLS.clear()
    PAUSES.clear()


def _image(name: str) -> dict[str, Any]:
    return {"type": "image", "base64": name, "mime_type": "image/png"}


def _label(seq: int, name: str = "screenshot") -> dict[str, Any]:
    return {"type": "text", "text": f"tools.{name} call {seq}:"}


@tool
async def screenshot(page: str, delay: float = 0) -> list[dict[str, Any]]:
    """Take a screenshot of a page."""
    await asyncio.sleep(delay)
    CALLS.append(page)
    return [{"type": "text", "text": f"shot of {page}"}, _image(page)]


@tool(response_format="content_and_artifact")
def weather(city: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Get the weather, like an MCP tool with structured content."""
    data = {"city": city, "temp": 21}
    return [{"type": "text", "text": str(data)}, _image("map")], {
        "structured_content": data
    }


@tool
def echo(x: str) -> str:
    """Return the text."""
    return x


@tool
def danger(x: str) -> str:
    """Do something that needs approval."""
    CALLS.append(f"danger:{x}")
    return f"danger:{x}"


def _agent(code: str, **middleware_kwargs: Any) -> Any:
    script = [
        AIMessage(
            content="",
            tool_calls=[{"name": "eval", "args": {"code": code}, "id": "c1"}],
        ),
        AIMessage("done"),
    ]
    tools = [screenshot, weather, echo, danger]
    return create_agent(
        model=FakeChatModel(messages=iter(script)),
        tools=tools,
        middleware=[
            CodeInterpreterMiddleware(ptc=[t.name for t in tools], **middleware_kwargs),
            HumanInTheLoopMiddleware(
                interrupt_on={"danger": True}, interrupt_mode="per_call"
            ),
        ],
        checkpointer=InMemorySaver(),
    )


async def _eval_content(
    agent: Any, invoke_mode: Literal["invoke", "ainvoke"] = "ainvoke"
) -> str | list[Any]:
    """Run to completion, approving every pause; return the eval's content."""
    config = {"configurable": {"thread_id": "t"}}
    payload: Any = {"messages": [HumanMessage("go")]}
    while True:
        if invoke_mode == "ainvoke":
            out = await agent.ainvoke(payload, config, version="v2")
        else:
            out = agent.invoke(payload, config, version="v2")
        if not out.interrupts:
            break
        PAUSES.append([i.value for i in out.interrupts])
        payload = Command(resume={i.id: {"type": "approve"} for i in out.interrupts})
    return next(m for m in out.value["messages"] if isinstance(m, ToolMessage)).content


@pytest.mark.parametrize("invoke_mode", ["invoke", "ainvoke"])
async def test_screenshot_from_code_is_shown_to_the_model(
    invoke_mode: Literal["invoke", "ainvoke"],
) -> None:
    code = "(await tools.screenshot({page: 'home'})).map((b) => b.type).join()"
    content = await _eval_content(_agent(code), invoke_mode)

    # JS still gets the tool's data: its content blocks.
    assert content == [
        {"type": "text", "text": "<result>text,image</result>"},
        _label(0),
        _image("home"),
    ]


async def test_artifact_tool_gives_js_its_artifact_and_the_model_its_image() -> None:
    code = "(await tools.weather({city: 'Oslo'})).structured_content.temp"
    content = await _eval_content(_agent(code))

    assert content == [
        {"type": "text", "text": "<result>21</result>"},
        _label(0, "weather"),
        _image("map"),
    ]


async def test_media_follows_issue_order_not_completion_order() -> None:
    code = (
        "await Promise.all([['a', 0.2], ['b', 0.1], ['c', 0]].map("
        "([page, delay]) => tools.screenshot({page, delay}))); 'ok'"
    )
    content = await _eval_content(_agent(code))

    assert CALLS == ["c", "b", "a"]
    assert content[1:] == [
        _label(0),
        _image("a"),
        _label(1),
        _image("b"),
        _label(2),
        _image("c"),
    ]


async def test_only_the_last_media_blocks_are_kept() -> None:
    code = "for (let i = 0; i < 10; i++) await tools.screenshot({page: 'p' + i}); 'ok'"
    content = await _eval_content(_agent(code))

    images = [b for b in content if b["type"] == "image"]
    assert images == [_image(f"p{i}") for i in range(2, 10)]
    assert content[1] == {
        "type": "text",
        "text": "[2 earlier media blocks from `tools.*` calls omitted; "
        "only the last 8 are kept.]",
    }
    assert content[2] == _label(2)


@pytest.mark.parametrize(
    ("limit", "count"), [(None, 10), (3, 3)], ids=["uncapped", "three"]
)
async def test_media_limit_is_configurable(limit: int | None, count: int) -> None:
    code = "for (let i = 0; i < 10; i++) await tools.screenshot({page: 'p' + i}); 'ok'"
    content = await _eval_content(_agent(code, max_tool_media_blocks=limit))

    assert sum(b["type"] == "image" for b in content) == count


async def test_zero_limit_shows_no_media() -> None:
    content = await _eval_content(
        _agent("await tools.screenshot({page: 'home'}); 'ok'", max_tool_media_blocks=0)
    )

    assert content == "<result>ok</result>"


async def test_eval_without_media_returns_plain_text() -> None:
    content = await _eval_content(_agent("await tools.echo({x: 'hi'})"))

    assert content == "<result>hi</result>"


async def test_media_is_shown_when_the_eval_throws_afterwards() -> None:
    content = await _eval_content(
        _agent("await tools.screenshot({page: 'home'}); null.boom")
    )

    assert content[0]["text"].startswith('<error type="TypeError">')
    assert content[1:] == [_label(0), _image("home")]


async def test_media_from_before_a_pause_survives_the_resume() -> None:
    code = (
        "await tools.screenshot({page: 'before'});"
        " await tools.danger({x: 'b'});"
        " await tools.screenshot({page: 'after'}); 'ok'"
    )
    content = await _eval_content(_agent(code))

    # The first screenshot ran once; the re-run reused its journaled result.
    assert len(PAUSES) == 1
    assert CALLS == ["before", "danger:b", "after"]
    assert content == [
        {"type": "text", "text": "<result>ok</result>"},
        _label(0),
        _image("before"),
        _label(2),
        _image("after"),
    ]


def test_negative_media_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_tool_media_blocks"):
        CodeInterpreterMiddleware(max_tool_media_blocks=-1)
