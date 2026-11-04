"""An eval whose final value is message content reaches the model as blocks."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.subagents import CompiledSubAgent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

from langchain_quickjs import CodeInterpreterMiddleware
from langchain_quickjs import _format as format_module
from tests._common import FakeChatModel

if TYPE_CHECKING:
    from pathlib import Path

# A 1x1 transparent PNG.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA"
    "60e6kgAAAABJRU5ErkJggg=="
)
_PNG_B64 = base64.b64encode(_PNG).decode()


def _script(code: str) -> FakeChatModel:
    return FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "eval",
                            "args": {"code": code},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Done."),
            ]
        )
    )


def _eval_message(result: dict[str, Any]) -> ToolMessage:
    [message] = [
        m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "eval"
    ]
    return message


def _run(code: str, tmp_path: Path) -> ToolMessage:
    (tmp_path / "a.png").write_bytes(_PNG)
    middleware = CodeInterpreterMiddleware(ptc=["read_file"])
    agent = create_deep_agent(
        model=_script(code),
        middleware=[middleware],
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
    )
    try:
        result = agent.invoke({"messages": [HumanMessage(content="go")]})
    finally:
        middleware._registry.close()
    return _eval_message(result)


def _image(message: ToolMessage) -> dict[str, Any]:
    assert isinstance(message.content, list)
    [image] = [
        b for b in message.content if isinstance(b, dict) and b["type"] == "image"
    ]
    return image


def test_read_file_image_as_final_value_reaches_model(tmp_path: Path) -> None:
    message = _run(
        'console.log("loaded"); await tools.readFile({file_path: "/a.png"})',
        tmp_path,
    )

    assert isinstance(message.content, list)
    header = message.content[0]
    assert isinstance(header, dict)
    assert header["type"] == "text"
    assert "<stdout>\nloaded\n</stdout>" in header["text"]
    assert "<result>[image/png]</result>" in header["text"]
    image = _image(message)
    assert image["base64"] == _PNG_B64
    assert image["mime_type"] == "image/png"


def test_json_stringify_opts_out_to_text(tmp_path: Path) -> None:
    message = _run(
        'JSON.stringify(await tools.readFile({file_path: "/a.png"}))', tmp_path
    )

    assert isinstance(message.content, str)
    assert _PNG_B64 in message.content


def test_mixed_caption_and_block_keeps_order(tmp_path: Path) -> None:
    message = _run(
        'const [img] = await tools.readFile({file_path: "/a.png"}); ["a pixel", img]',
        tmp_path,
    )

    assert isinstance(message.content, list)
    header, caption, image = message.content
    assert isinstance(header, dict)
    assert "<result>[text] [image/png]</result>" in header["text"]
    assert caption == {"type": "text", "text": "a pixel"}
    assert isinstance(image, dict)
    assert image["type"] == "image"


@pytest.mark.parametrize(
    "code",
    [
        '({type: "image", mime_type: "image/png"})',
        '({type: "image", base64: 42})',
        '["a", "b"]',
        "[]",
        '[{type: "image", base64: "AAAA"}, 1]',
    ],
)
def test_non_content_value_stays_text(code: str, tmp_path: Path) -> None:
    message = _run(code, tmp_path)

    assert isinstance(message.content, str)
    assert "<result>" in message.content


def test_oversized_blocks_fall_back_to_text_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(format_module, "MAX_CONTENT_BLOCK_BYTES", 100)

    message = _run(
        '[{type: "image", base64: "A".repeat(500), mime_type: "image/png"}]',
        tmp_path,
    )

    assert isinstance(message.content, str)
    assert "Content blocks not shown: 500 bytes" in message.content
    assert "AAAA" not in message.content


def _approval_subagent() -> CompiledSubAgent:
    def ask(state: MessagesState) -> dict[str, Any]:
        decision = interrupt("approve?")
        return {"messages": [AIMessage(content=decision)]}

    graph = StateGraph(MessagesState)
    graph.add_node("ask", ask)
    graph.add_edge(START, "ask")
    graph.add_edge("ask", END)
    return CompiledSubAgent(
        name="approver", description="Ask for approval.", runnable=graph.compile()
    )


async def test_image_survives_interrupt_and_resume(tmp_path: Path) -> None:
    """Resume re-executes the eval, which rebuilds the same image block."""
    (tmp_path / "a.png").write_bytes(_PNG)
    middleware = CodeInterpreterMiddleware(ptc=["read_file"])
    code = (
        'const img = await tools.readFile({file_path: "/a.png"});\n'
        'await task({description: "ok?", subagentType: "approver"});\n'
        "img"
    )
    agent = create_deep_agent(
        model=_script(code),
        middleware=[middleware],
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        subagents=[_approval_subagent()],
        checkpointer=InMemorySaver(),
    )
    config = RunnableConfig(configurable={"thread_id": "image-resume"})
    try:
        await agent.ainvoke({"messages": [HumanMessage(content="go")]}, config)
        snapshot = await agent.aget_state(config)
        assert len(snapshot.interrupts) == 1
        result = await agent.ainvoke(
            Command(resume={snapshot.interrupts[0].id: "approved"}), config
        )
    finally:
        middleware._registry.close()

    message = _eval_message(result)
    assert _image(message)["base64"] == _PNG_B64
