"""Shared fixtures for the skill tool disclosure tests.

Builds skills on disk, the skill tools they name, scripted tool calls, and
HTTP stubs for real provider chat models, so tests can assert on the payload
that leaves the process.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
import langchain_anthropic.chat_models as anthropic_chat_models
from langchain.tools import ToolRuntime  # noqa: TC002  # `@tool` resolves the injected `runtime` annotation at runtime
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.graph import create_deep_agent

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

SKILLS_SOURCE = "/skills/"
CRM_PATH = "/skills/crm/SKILL.md"


def skill_md(name: str, include_tools: str | None, description: str = "Manage customer requests") -> str:
    """Return a `SKILL.md` whose frontmatter names `include_tools`."""
    metadata = f"metadata:\n  include_tools: {include_tools}\n" if include_tools is not None else ""
    return f"---\nname: {name}\ndescription: {description}\n{metadata}---\n\n# {name}\n\nFollow these steps.\n"


def write_skill(root: Path, name: str, include_tools: str | None = None, *, content: str | None = None) -> str:
    """Write a skill under `root/skills/<name>/` and return its virtual `SKILL.md` path."""
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content if content is not None else skill_md(name, include_tools))
    return f"/skills/{name}/SKILL.md"


def skills_backend(root: Path) -> FilesystemBackend:
    """Return a virtual-mode backend rooted at `root`, so paths look like `/skills/...`."""
    return FilesystemBackend(root_dir=str(root), virtual_mode=True)


def skills_agent(root: Path, model: BaseChatModel | str, **kwargs: Any) -> CompiledStateGraph:
    """Build a deep agent over the skills under `root`, with `create_customer_request` as a skill tool."""
    kwargs.setdefault("skill_tools", [create_customer_request])
    return create_deep_agent(model=model, backend=skills_backend(root), skills=[SKILLS_SOURCE], **kwargs)


@tool
def create_customer_request(title: str, runtime: ToolRuntime) -> str:
    """Create a customer request."""
    return f"created {title} ({runtime.tool_call_id})"


@tool
def list_customer_requests() -> str:
    """List customer requests."""
    return "no requests"


@tool(extras={"defer_loading": True})
def search_tickets(query: str) -> str:
    """Search support tickets."""
    return f"tickets for {query}"


def call(name: str, call_id: str, **args: Any) -> ToolCall:
    """Return a tool call as the model would emit it."""
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def read(call_id: str, path: str = CRM_PATH, **args: Any) -> ToolCall:
    """Return a `read_file` call for `path`."""
    return call("read_file", call_id, file_path=path, **args)


def ai(*calls: ToolCall, content: str = "") -> AIMessage:
    """Return an assistant turn making `calls`."""
    return AIMessage(content=content, tool_calls=list(calls))


def invoke(agent: CompiledStateGraph, state: dict[str, Any], mode: str, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Run `agent` through its sync or async entry point."""
    if mode == "sync":
        return agent.invoke(state, config)
    return asyncio.run(agent.ainvoke(state, config))


def tool_messages(result: dict[str, Any], name: str) -> list[Any]:
    """Return the `ToolMessage`s in `result` produced for tool `name`."""
    return [m for m in result["messages"] if m.type == "tool" and m.name == name]


def bound_tool_names(entry: dict[str, Any]) -> list[str]:
    """Return the names of the tools bound on one `GenericFakeChatModel` call."""
    return [t["name"] if isinstance(t, dict) else t.name for t in entry["tools"]]


Turn = str | list[ToolCall] | httpx.Response
"""A scripted model turn: text, tool calls, or a raw HTTP response."""


@dataclass
class ProviderStub:
    """Answer provider HTTP requests from a script and record each request."""

    render: Callable[[int, str | list[ToolCall]], dict[str, Any]]
    turns: list[Turn]
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Record `request` and return the next scripted response."""
        index = len(self.requests)
        self.requests.append(request)
        turn = self.turns[index]
        if isinstance(turn, httpx.Response):
            return turn
        return httpx.Response(200, json=self.render(index, turn))

    @property
    def bodies(self) -> list[dict[str, Any]]:
        """Return every request body, decoded."""
        return [json.loads(request.content) for request in self.requests]


def anthropic_message(index: int, turn: str | list[ToolCall]) -> dict[str, Any]:
    """Render a scripted turn as an Anthropic Messages API response."""
    if isinstance(turn, str):
        content = [{"type": "text", "text": turn}]
    else:
        content = [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]} for c in turn]
    return {
        "id": f"msg_{index}",
        "type": "message",
        "role": "assistant",
        "model": "claude",
        "content": content,
        "stop_reason": "end_turn" if isinstance(turn, str) else "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def openai_response(index: int, turn: str | list[ToolCall]) -> dict[str, Any]:
    """Render a scripted turn as an OpenAI Responses API response."""
    if isinstance(turn, str):
        output = [
            {
                "type": "message",
                "id": f"msg_{index}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": turn, "annotations": []}],
            }
        ]
    else:
        output = [
            {
                "type": "function_call",
                "id": f"fc_{c['id']}",
                "call_id": c["id"],
                "name": c["name"],
                "arguments": json.dumps(c["args"]),
                "status": "completed",
            }
            for c in turn
        ]
    return {
        "id": f"resp_{index}",
        "object": "response",
        "created_at": 0,
        "model": "gpt",
        "status": "completed",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def openai_chat_completion(index: int, turn: str | list[ToolCall]) -> dict[str, Any]:
    """Render a scripted turn as an OpenAI Chat Completions response."""
    tool_calls = None
    if not isinstance(turn, str):
        tool_calls = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for c in turn]
    return {
        "id": f"chatcmpl-{index}",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": turn if isinstance(turn, str) else None, "tool_calls": tool_calls},
                "finish_reason": "stop" if isinstance(turn, str) else "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def stub_anthropic(
    monkeypatch: pytest.MonkeyPatch, turns: list[Turn], model: str = "claude-opus-5-5", **model_kwargs: Any
) -> tuple[ChatAnthropic, ProviderStub]:
    """Return a real `ChatAnthropic` whose HTTP transport answers from `turns`."""
    stub = ProviderStub(anthropic_message, list(turns))
    transport = httpx.MockTransport(stub)
    monkeypatch.setattr(anthropic_chat_models, "_get_default_httpx_client", lambda **_: httpx.Client(transport=transport))
    monkeypatch.setattr(anthropic_chat_models, "_get_default_async_httpx_client", lambda **_: httpx.AsyncClient(transport=transport))
    return ChatAnthropic(model=model, api_key="test-key", max_retries=0, **model_kwargs), stub


def stub_openai(
    turns: list[Turn], model: str = "gpt-6-astra", *, use_responses_api: bool = True, **model_kwargs: Any
) -> tuple[ChatOpenAI, ProviderStub]:
    """Return a real `ChatOpenAI` whose HTTP transport answers from `turns`."""
    stub = ProviderStub(openai_response if use_responses_api else openai_chat_completion, list(turns))
    transport = httpx.MockTransport(stub)
    chat_model = ChatOpenAI(
        model=model,
        api_key="test-key",
        max_retries=0,
        use_responses_api=use_responses_api,
        http_client=httpx.Client(transport=transport),
        http_async_client=httpx.AsyncClient(transport=transport),
        **model_kwargs,
    )
    return chat_model, stub
