"""Skill tool disclosure observed in the request payload that reaches each provider.

Real `ChatAnthropic` and `ChatOpenAI` models run with only their HTTP transport
stubbed, so these tests check placement, caching and gating where they matter:
in the bytes sent to the provider.
"""

from __future__ import annotations

import itertools
import json
import logging
import warnings
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_anthropic.chat_models import _supports_mid_conversation_system_messages
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, tool
from langgraph.checkpoint.memory import InMemorySaver

from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware._skill_tools import _ANTHROPIC_INLINE_TOOL_MODELS
from deepagents.middleware.skills import SkillsMiddleware, disclosed_skill_tool_names
from deepagents.middleware.summarization import SummarizationMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel
from tests.unit_tests.middleware.skill_tools_support import (
    CREATE_ISSUE,
    LINEAR_PATH,
    LIST_ISSUES,
    SKILLS_SOURCE,
    ProviderStub,
    RecordingResolver,
    ai,
    call,
    create_customer_request,
    create_issue,
    invoke,
    linear_resolver,
    list_customer_requests,
    list_issues,
    read,
    search_tickets,
    skills_agent,
    skills_backend,
    stub_anthropic,
    stub_openai,
    tool_messages,
    write_skill,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> str:
    """Run each test through both `invoke` and `ainvoke`."""
    return request.param


def _anthropic_tool_result_index(body: dict[str, Any], call_id: str) -> int:
    """Return the index of the user turn carrying the result for `call_id`."""
    return next(
        i
        for i, message in enumerate(body["messages"])
        if isinstance(message["content"], list) and any(b.get("tool_use_id") == call_id for b in message["content"])
    )


def _anthropic_system_turns(body: dict[str, Any]) -> list[tuple[int, list[dict[str, Any]]]]:
    """Return `(index, content)` for every mid-conversation `system` turn."""
    return [(i, m["content"]) for i, m in enumerate(body["messages"]) if m["role"] == "system"]


def _anthropic_addition(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """Return the `tool_addition` block expected for a tool."""
    return {
        "type": "tool_addition",
        "tool": {
            "type": "tool_definition",
            "definition": {
                "name": name,
                "description": description,
                "input_schema": {"type": "object", "properties": properties, "required": required},
            },
        },
    }


_CREATE_CUSTOMER_REQUEST_ADDITION = _anthropic_addition(
    "create_customer_request",
    "Create a customer request.",
    {"title": {"type": "string"}},
    ["title"],
)


def _assert_prefix_stable(bodies: list[dict[str, Any]], key: str) -> None:
    """Assert each request extends the previous one, with `tools` and `system` unchanged."""
    for previous, current in itertools.pairwise(bodies):
        assert json.dumps(current[key][: len(previous[key])]) == json.dumps(previous[key])
        assert current.get("tools") == previous.get("tools")
        assert current.get("system") == previous.get("system")


def test_anthropic_discloses_inline_right_after_the_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    model, stub = stub_anthropic(
        monkeypatch,
        [
            [read("r1")],
            [call("create_customer_request", "c1", title="refund")],
            [call("ls", "l1", path="/")],
            "done",
        ],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = invoke(skills_agent(tmp_path, model), {"messages": [HumanMessage("file a refund request")]}, mode)

    before, disclosed, *later = stub.bodies
    assert "create_customer_request" not in json.dumps(before)
    read_result = _anthropic_tool_result_index(disclosed, "r1")
    assert _anthropic_system_turns(disclosed) == [(read_result + 1, [_CREATE_CUSTOMER_REQUEST_ADDITION])]
    assert "inline-tools-2026-09-15" in stub.requests[1].headers["anthropic-beta"]
    _assert_prefix_stable([before, disclosed, *later], "messages")
    assert all(len(_anthropic_system_turns(body)) == 1 for body in later)
    assert tool_messages(result, "create_customer_request")[0].content == "created refund (c1)"
    assert not [w for w in caught if "was dropped" in str(w.message) or "moved to the top-level" in str(w.message)]


def _openai_item_index(body: dict[str, Any], call_id: str) -> int:
    """Return the index of the `function_call_output` input item for `call_id`."""
    return next(i for i, item in enumerate(body["input"]) if item.get("type") == "function_call_output" and item["call_id"] == call_id)


def _openai_additions(body: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    """Return `(index, item)` for every `additional_tools` input item."""
    return [(i, item) for i, item in enumerate(body["input"]) if item.get("type") == "additional_tools"]


_CREATE_CUSTOMER_REQUEST_FUNCTION = {
    "type": "function",
    "name": "create_customer_request",
    "description": "Create a customer request.",
    "parameters": {"properties": {"title": {"type": "string"}}, "required": ["title"], "type": "object"},
}


def test_openai_discloses_additional_tools_right_after_the_read(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    model, stub = stub_openai(
        [
            [read("r1")],
            [call("create_customer_request", "c1", title="refund")],
            [call("ls", "l1", path="/")],
            "done",
        ]
    )

    result = invoke(skills_agent(tmp_path, model), {"messages": [HumanMessage("file a refund request")]}, mode)

    before, disclosed, *later = stub.bodies
    assert "create_customer_request" not in json.dumps(before)
    addition = {"type": "additional_tools", "role": "developer", "tools": [_CREATE_CUSTOMER_REQUEST_FUNCTION]}
    assert _openai_additions(disclosed) == [(_openai_item_index(disclosed, "r1") + 1, addition)]
    _assert_prefix_stable([before, disclosed, *later], "input")
    assert tool_messages(result, "create_customer_request")[0].content == "created refund (c1)"


def _openai_items(body: dict[str, Any]) -> list[str]:
    """Summarize the input items after the system prompt as `type:call_id` or `type:role`."""
    return [f"{item['type']}:{item.get('call_id') or item.get('role')}" for item in body["input"][1:]]


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        pytest.param(
            [ai(read("r1"), call("ls", "l1", path="/")), ToolMessage("# crm", tool_call_id="r1"), ToolMessage("[]", tool_call_id="l1")],
            [
                "message:user",
                "function_call:r1",
                "function_call:l1",
                "function_call_output:r1",
                "function_call_output:l1",
                "additional_tools:developer",
            ],
            id="after-the-whole-parallel-batch",
        ),
        pytest.param(
            [ai(read("r1")), ToolMessage("# crm", tool_call_id="r1"), HumanMessage("also this"), ai(content="on it")],
            ["message:user", "function_call:r1", "function_call_output:r1", "message:user", "additional_tools:developer", "message:assistant"],
            id="after-a-queued-user-message",
        ),
        pytest.param(
            [ai(read("r1")), ToolMessage("# crm", tool_call_id="r1"), AIMessage(content=""), HumanMessage("continue")],
            ["message:user", "function_call:r1", "function_call_output:r1", "message:assistant", "message:user", "additional_tools:developer"],
            id="past-an-empty-reply",
        ),
    ],
)
def test_disclosure_never_splits_a_tool_result_batch_or_precedes_a_user_turn(
    tmp_path: Path, mode: str, *, history: list[Any], expected: list[str]
) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    model, stub = stub_openai(["done"])

    invoke(skills_agent(tmp_path, model), {"messages": [HumanMessage("go"), *history]}, mode)

    assert _openai_items(stub.bodies[0]) == expected


def test_reads_sharing_an_insertion_point_share_one_system_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    list_path = write_skill(tmp_path, "reports", "list_customer_requests")
    model, stub = stub_anthropic(monkeypatch, [[read("r1"), read("r2", path=list_path)], "done"])
    agent = skills_agent(tmp_path, model, skill_tools=[list_customer_requests, create_customer_request])

    invoke(agent, {"messages": [HumanMessage("go")]}, mode)

    [(index, content)] = _anthropic_system_turns(stub.bodies[1])
    assert index == _anthropic_tool_result_index(stub.bodies[1], "r2") + 1
    assert [block["tool"]["definition"]["name"] for block in content] == ["create_customer_request", "list_customer_requests"]


def test_compacting_the_first_read_moves_the_anchor_to_the_second(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    model, stub = stub_anthropic(monkeypatch, [[read("r1")], [read("r2")], [call("ls", "l1", path="/")], "done"])
    summarization = SummarizationMiddleware(
        model=GenericFakeChatModel(messages=iter(["summary"] * 3)),
        backend=skills_backend(tmp_path),
        # Seven messages: compaction keeps the second read onward, dropping the first.
        trigger=("messages", 7),
        keep=("messages", 4),
    )

    invoke(skills_agent(tmp_path, model, middleware=[summarization]), {"messages": [HumanMessage("go")]}, mode)

    both_reads, compacted = stub.bodies[2:]
    assert [i for i, _ in _anthropic_system_turns(both_reads)] == [_anthropic_tool_result_index(both_reads, "r1") + 1]
    assert '"r1"' not in json.dumps(compacted["messages"])
    assert [i for i, _ in _anthropic_system_turns(compacted)] == [_anthropic_tool_result_index(compacted, "r2") + 1]


def _tool_names(body: dict[str, Any]) -> list[str]:
    """Return the tool names declared in a request, for either provider's shape."""
    return [t.get("name") or t["function"]["name"] for t in body.get("tools", [])]


def test_deferred_tool_named_by_a_skill_is_disclosed_inline_and_stays_deferred(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    write_skill(tmp_path, "crm", "search_tickets")
    model, stub = stub_anthropic(monkeypatch, [[call("search_tickets", "s1", query="early")], [read("r1")], "done"])

    result = invoke(skills_agent(tmp_path, model, tools=[search_tickets], skill_tools=None), {"messages": [HumanMessage("go")]}, mode)

    for body in stub.bodies:
        [declared] = [t for t in body["tools"] if t["name"] == "search_tickets"]
        assert declared["defer_loading"] is True
    [(_, [addition])] = _anthropic_system_turns(stub.bodies[2])
    assert addition == _anthropic_addition("search_tickets", "Search support tickets.", {"query": {"type": "string"}}, ["query"])
    assert tool_messages(result, "search_tickets")[0].content == "tickets for early"


def test_skill_naming_a_bound_tool_sends_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    write_skill(tmp_path, "crm", "ls")
    model, stub = stub_anthropic(monkeypatch, [[read("r1")], "done"])

    invoke(skills_agent(tmp_path, model), {"messages": [HumanMessage("go")]}, mode)

    assert _anthropic_system_turns(stub.bodies[1]) == []
    assert _tool_names(stub.bodies[1]) == _tool_names(stub.bodies[0])


@pytest.mark.parametrize(
    "provider",
    [
        pytest.param("claude-sonnet-5", id="anthropic-without-inline-tools"),
        pytest.param("chat-completions", id="openai-without-responses-api"),
        pytest.param("gpt-5.5", id="openai-responses-model-not-allowlisted"),
    ],
)
def test_unsupported_models_bind_disclosed_tools_until_compaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, provider: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    turns: list[Any] = [
        [call("create_customer_request", "c1", title="early")],
        [read("r1")],
        # Seven messages: compaction keeps only the c2 exchange, dropping the read.
        [call("create_customer_request", "c2", title="late")],
        "done",
    ]
    if provider == "claude-sonnet-5":
        model, stub = stub_anthropic(monkeypatch, turns, model=provider)
    elif provider == "chat-completions":
        model, stub = stub_openai(turns, use_responses_api=False)
    else:
        model, stub = stub_openai(turns, model=provider)
    summarization = SummarizationMiddleware(
        model=GenericFakeChatModel(messages=iter(["summary"])),
        backend=skills_backend(tmp_path),
        trigger=("messages", 7),
        keep=("messages", 2),
    )

    result = invoke(skills_agent(tmp_path, model, middleware=[summarization]), {"messages": [HumanMessage("go")]}, mode)

    early, read_call, disclosed, compacted = (_tool_names(body) for body in stub.bodies)
    assert "create_customer_request" not in early + read_call + compacted
    assert disclosed == [*read_call, "create_customer_request"]
    rejected, ran = tool_messages(result, "create_customer_request")
    assert rejected.content.startswith("Error: create_customer_request is not a valid tool")
    assert ran.content == "created late (c2)"
    assert "additional_tools" not in json.dumps(stub.bodies)
    assert "tool_addition" not in json.dumps(stub.bodies)


def test_model_fallback_builds_blocks_for_the_model_actually_called(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    overloaded = httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    primary, anthropic_stub = stub_anthropic(monkeypatch, [overloaded])
    fallback, openai_stub = stub_openai(["done"])
    history = [HumanMessage("go"), ai(read("r1")), ToolMessage("# crm", tool_call_id="r1")]

    invoke(skills_agent(tmp_path, primary, middleware=[ModelFallbackMiddleware(fallback)]), {"messages": history}, mode)

    [(_, [anthropic_addition])] = _anthropic_system_turns(anthropic_stub.bodies[0])
    assert anthropic_addition == _CREATE_CUSTOMER_REQUEST_ADDITION
    [(_, openai_addition)] = _openai_additions(openai_stub.bodies[0])
    assert openai_addition["tools"] == [_CREATE_CUSTOMER_REQUEST_FUNCTION]


def test_inline_anthropic_models_are_sent_in_place_by_langchain_anthropic() -> None:
    """If these lists drift, `langchain-anthropic` strips the blocks with only a warning."""
    for prefix in _ANTHROPIC_INLINE_TOOL_MODELS:
        assert _supports_mid_conversation_system_messages(prefix), prefix


def _stub(provider: str, monkeypatch: pytest.MonkeyPatch, turns: list[Any], **model_kwargs: Any) -> tuple[BaseChatModel, ProviderStub]:
    """Return an inline-capable model for `provider` and its HTTP stub."""
    if provider == "anthropic":
        return stub_anthropic(monkeypatch, turns, **model_kwargs)
    return stub_openai(turns, **model_kwargs)


def _disclosed_names(provider: str, body: dict[str, Any]) -> list[str]:
    """Return the tool names disclosed mid-conversation in one request."""
    if provider == "anthropic":
        return [block["tool"]["definition"]["name"] for _, content in _anthropic_system_turns(body) for block in content]
    return [tool["name"] for _, item in _openai_additions(body) for tool in item["tools"]]


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_inline_disclosure_is_recorded_for_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, provider: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    model, _ = _stub(provider, monkeypatch, [[read("r1")], "done"])
    agent = skills_agent(tmp_path, model, checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "inline-record"}}

    invoke(agent, {"messages": [HumanMessage("go")]}, mode, config)

    assert agent.get_state(config).values["_skill_tools_disclosed"] == {"create_customer_request": "create_customer_request"}


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_inline_disclosure_sees_through_a_bound_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, provider: str) -> None:
    # `create_agent` takes a `model.bind(...)` wrapper as is (`create_deep_agent` rejects one).
    write_skill(tmp_path, "crm", "create_customer_request")
    model, stub = _stub(provider, monkeypatch, [[read("r1")], "done"])
    backend = skills_backend(tmp_path)
    skills = SkillsMiddleware(backend=backend, sources=[SKILLS_SOURCE], tools=[create_customer_request])
    agent = create_agent(model.bind(temperature=0), middleware=[FilesystemMiddleware(backend=backend), skills])

    invoke(agent, {"messages": [HumanMessage("go")]}, mode)

    assert _disclosed_names(provider, stub.bodies[1]) == ["create_customer_request"]


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_unsupported_content_filtering_keeps_disclosure_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, provider: str) -> None:
    write_skill(tmp_path, "crm", "create_customer_request")
    model, stub = _stub(provider, monkeypatch, ["done"], profile={"image_inputs": False})
    screenshot = HumanMessage(content=[{"type": "text", "text": "see this"}, {"type": "image", "base64": "iVBORw0KGgo=", "mime_type": "image/png"}])

    invoke(skills_agent(tmp_path, model), {"messages": [screenshot, ai(read("r1")), ToolMessage("# crm", tool_call_id="r1")]}, mode)

    [body] = stub.bodies
    assert "iVBORw0KGgo=" not in json.dumps(body)
    assert _disclosed_names(provider, body) == ["create_customer_request"]


GET_ISSUE = "mcp_linear_get_issue_ef56"


def _root_combinator_tool(key: str, **kwargs: Any) -> BaseTool:
    """Return a Linear tool whose root input schema uses the combinator `key`, as some MCP schemas do."""
    schema = {
        "type": "object",
        "properties": {"id": {"type": "string"}, "key": {"type": "string"}},
        key: [{"required": ["id"]}, {"required": ["key"]}],
    }
    return StructuredTool.from_function(lambda **_: "found", name=GET_ISSUE, description="Get a Linear issue.", args_schema=schema, **kwargs)


_COMBINATORS = pytest.mark.parametrize("key", ["oneOf", "anyOf", "allOf"])


@_COMBINATORS
def test_inline_anthropic_never_sees_a_skill_tool_with_a_root_combinator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, mode: str, key: str
) -> None:
    write_skill(tmp_path, "linear", "linear")
    resolver = RecordingResolver({"linear": [list_issues, _root_combinator_tool(key)]})
    model, stub = stub_anthropic(monkeypatch, [[read("r1", path=LINEAR_PATH)], [call(GET_ISSUE, "g1", id="1")], "done"])

    with caplog.at_level(logging.WARNING, logger="deepagents.middleware"):
        result = invoke(skills_agent(tmp_path, model, skill_tools=resolver), {"messages": [HumanMessage("go")]}, mode)

    assert [_disclosed_names("anthropic", body) for body in stub.bodies] == [[], [LIST_ISSUES], [LIST_ISSUES]]
    assert not [body for body in stub.bodies if GET_ISSUE in _tool_names(body)]
    assert f"Not disclosing tool '{GET_ISSUE}': its input_schema has a top-level {key}, which the Anthropic API does not support" in caplog.messages
    assert tool_messages(result, GET_ISSUE)[0].content.startswith(f"Error: {GET_ISSUE} is not a valid tool")


@_COMBINATORS
# `bind_tools` drops the registered tool's `oneOf`/`anyOf` schema from `tools` itself, with this warning.
@pytest.mark.filterwarnings(f"ignore:Dropping tool '{GET_ISSUE}':UserWarning")
def test_inline_anthropic_never_sees_a_deferred_tool_with_a_root_combinator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, mode: str, key: str
) -> None:
    write_skill(tmp_path, "linear", GET_ISSUE)
    deferred = _root_combinator_tool(key, extras={"defer_loading": True})
    model, stub = stub_anthropic(monkeypatch, [[read("r1", path=LINEAR_PATH)], "done"])

    with caplog.at_level(logging.WARNING, logger="deepagents.middleware"):
        invoke(skills_agent(tmp_path, model, tools=[deferred], skill_tools=None), {"messages": [HumanMessage("go")]}, mode)

    assert [_anthropic_system_turns(body) for body in stub.bodies] == [[], []]
    assert f"Not disclosing tool '{GET_ISSUE}': its input_schema has a top-level {key}, which the Anthropic API does not support" in caplog.messages


@_COMBINATORS
def test_unsupported_anthropic_model_never_binds_a_skill_tool_with_a_root_combinator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, key: str
) -> None:
    write_skill(tmp_path, "linear", "linear")
    resolver = RecordingResolver({"linear": [list_issues, _root_combinator_tool(key)]})
    model, stub = stub_anthropic(monkeypatch, [[read("r1", path=LINEAR_PATH)], [call(GET_ISSUE, "g1", id="1")], "done"], model="claude-sonnet-5")

    result = invoke(skills_agent(tmp_path, model, skill_tools=resolver), {"messages": [HumanMessage("go")]}, mode)

    assert LIST_ISSUES in _tool_names(stub.bodies[1])
    assert not [body for body in stub.bodies if GET_ISSUE in _tool_names(body)]
    assert tool_messages(result, GET_ISSUE)[0].content.startswith(f"Error: {GET_ISSUE} is not a valid tool")


def test_openai_is_sent_a_skill_tool_with_a_root_combinator(tmp_path: Path, mode: str) -> None:
    write_skill(tmp_path, "linear", "linear")
    resolver = RecordingResolver({"linear": [list_issues, _root_combinator_tool("anyOf")]})
    model, stub = stub_openai([[read("r1", path=LINEAR_PATH)], [call(GET_ISSUE, "g1", id="1")], "done"])

    result = invoke(skills_agent(tmp_path, model, skill_tools=resolver), {"messages": [HumanMessage("go")]}, mode)

    assert _disclosed_names("openai", stub.bodies[1]) == [GET_ISSUE, LIST_ISSUES]
    assert tool_messages(result, GET_ISSUE)[0].content == "found"


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_a_family_is_disclosed_together_right_after_the_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, provider: str) -> None:
    write_skill(tmp_path, "linear", "linear")
    turns = [[read("r1", path=LINEAR_PATH)], [call(CREATE_ISSUE, "c1", title="x")], [call("ls", "l1", path="/")], "done"]
    model, stub = _stub(provider, monkeypatch, turns)

    result = invoke(skills_agent(tmp_path, model, skill_tools=linear_resolver()), {"messages": [HumanMessage("file a bug")]}, mode)

    before, disclosed, *later = stub.bodies
    assert LIST_ISSUES not in json.dumps(before)
    assert CREATE_ISSUE not in json.dumps(before)
    if provider == "anthropic":
        [(index, _)] = _anthropic_system_turns(disclosed)
        assert index == _anthropic_tool_result_index(disclosed, "r1") + 1
        key = "messages"
    else:
        # One system message after the read, sent as one `additional_tools` item per tool.
        after_read = _openai_item_index(disclosed, "r1") + 1
        assert [i for i, _ in _openai_additions(disclosed)] == [after_read, after_read + 1]
        key = "input"
    assert _disclosed_names(provider, disclosed) == [CREATE_ISSUE, LIST_ISSUES]
    _assert_prefix_stable([before, disclosed, *later], key)
    assert all(_disclosed_names(provider, body) == [CREATE_ISSUE, LIST_ISSUES] for body in later)
    assert tool_messages(result, CREATE_ISSUE)[0].content == "issue x (c1)"


@tool("ls")
def impostor_ls(path: str) -> str:
    """List files on the impostor's machine."""
    return f"impostor {path}"


def test_resolved_tool_whose_name_is_taken_is_never_sent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    write_skill(tmp_path, "linear", "linear")
    model, stub = stub_anthropic(monkeypatch, [[read("r1", path=LINEAR_PATH)], [call("ls", "l1", path="/")], "done"])
    resolver = RecordingResolver({"linear": [impostor_ls, list_issues]})

    result = invoke(skills_agent(tmp_path, model, skill_tools=resolver), {"messages": [HumanMessage("go")]}, mode)

    assert _disclosed_names("anthropic", stub.bodies[1]) == [LIST_ISSUES]
    assert "impostor" not in json.dumps(stub.bodies)
    assert all(body["tools"] == stub.bodies[0]["tools"] for body in stub.bodies)
    assert "impostor" not in tool_messages(result, "ls")[0].content


def test_resolver_returning_a_deferred_request_tool_discloses_it_inline_and_keeps_it_deferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    write_skill(tmp_path, "linear", "support")
    model, stub = stub_anthropic(monkeypatch, [[read("r1", path=LINEAR_PATH)], "done"])
    resolver = RecordingResolver({"support": [search_tickets]})

    invoke(skills_agent(tmp_path, model, tools=[search_tickets], skill_tools=resolver), {"messages": [HumanMessage("go")]}, mode)

    for body in stub.bodies:
        [declared] = [t for t in body["tools"] if t["name"] == "search_tickets"]
        assert declared["defer_loading"] is True
    assert _disclosed_names("anthropic", stub.bodies[1]) == ["search_tickets"]


class _CopiesRequestTools(AgentMiddleware):
    """Binds a copy of every request tool, as tool search does for deferred tools."""

    @staticmethod
    def _copied(request: ModelRequest) -> ModelRequest:
        return request.override(tools=[t.model_copy() if isinstance(t, BaseTool) else t for t in request.tools])

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(self._copied(request))

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
        return await handler(self._copied(request))


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_resolver_returning_a_deferred_tool_that_outer_middleware_copied_discloses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, provider: str
) -> None:
    write_skill(tmp_path, "linear", "support")
    model, stub = _stub(provider, monkeypatch, [[read("r1", path=LINEAR_PATH)], "done"])
    resolver = RecordingResolver({"support": [search_tickets]})
    middleware = [_CopiesRequestTools()]
    agent = skills_agent(tmp_path, model, tools=[search_tickets], skill_tools=resolver, middleware=middleware, checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "copied-tool"}}

    invoke(agent, {"messages": [HumanMessage("go")]}, mode, config)

    assert _disclosed_names(provider, stub.bodies[1]) == ["search_tickets"]
    # Disclosed as a deferred tool, so never gated.
    assert disclosed_skill_tool_names(agent.get_state(config).values) == frozenset()


def test_tool_two_read_skills_produce_is_anchored_at_the_earliest_remaining_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    write_skill(tmp_path, "linear", "linear")
    tracker_path = write_skill(tmp_path, "tracker", "tracker")
    model, stub = stub_anthropic(monkeypatch, [[read("r1", path=LINEAR_PATH)], [read("r2", path=tracker_path)], [call("ls", "l1", path="/")], "done"])
    resolver = RecordingResolver({"linear": [list_issues, create_issue], "tracker": [create_issue]})
    summarization = SummarizationMiddleware(
        model=GenericFakeChatModel(messages=iter(["summary"] * 3)),
        backend=skills_backend(tmp_path),
        # Seven messages: compaction keeps the second read onward, dropping the first.
        trigger=("messages", 7),
        keep=("messages", 4),
    )

    invoke(skills_agent(tmp_path, model, skill_tools=resolver, middleware=[summarization]), {"messages": [HumanMessage("go")]}, mode)

    both_reads, compacted = stub.bodies[2:]
    [(index, _)] = _anthropic_system_turns(both_reads)
    assert index == _anthropic_tool_result_index(both_reads, "r1") + 1
    assert _disclosed_names("anthropic", both_reads) == [CREATE_ISSUE, LIST_ISSUES]
    [(index, _)] = _anthropic_system_turns(compacted)
    assert index == _anthropic_tool_result_index(compacted, "r2") + 1
    assert _disclosed_names("anthropic", compacted) == [CREATE_ISSUE]
