"""Exercise exact-command permissions through the ACP approval path."""

from unittest.mock import AsyncMock

import pytest
from acp.exceptions import RequestError
from acp.schema import AllowedOutcome, DeniedOutcome, RequestPermissionResponse
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Interrupt, StateSnapshot

from deepagents_acp.server import AgentServerACP
from tests.chat_model import GenericFakeChatModel


@pytest.fixture
def server() -> AgentServerACP:
    graph = create_deep_agent(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="done")])),
        checkpointer=MemorySaver(),
    )
    server = AgentServerACP(agent=graph)
    client = AsyncMock()
    server.on_connect(client)
    server._session_cwds["session"] = "/project"
    return server


async def request(
    server: AgentServerACP,
    args: dict[str, object],
    option: str = "approve_always",
    session: str = "session",
) -> list[dict[str, str]] | None:
    server._conn.request_permission.return_value = RequestPermissionResponse(
        outcome=AllowedOutcome(outcome="selected", option_id=option)
    )
    state = StateSnapshot(
        values={},
        next=("tools",),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(Interrupt(value={"action_requests": [{"name": "execute", "args": args}]}),),
    )
    return await server._handle_interrupts(current_state=state, session_id=session)


@pytest.mark.parametrize(
    ("approved", "changed"),
    [
        ("ls", "ls -la"),
        ("ls", "ls "),
        ("ls", " ls"),
        ("ls", "ls && printf marker #'"),
        ("ls", "ls; printf marker #'"),
        ("ls", "ls\nprintf marker"),
        ("ls", "ls | printf marker"),
        ("ls", "ls $(printf marker)"),
        ("ls", "ls > output"),
        ("ls", "ls # comment"),
        ("python -c 'print(1)'", "python -c 'print(2)'"),
        ("find . -print", "find . -exec printf marker +"),
        ("git status", "git -c core.pager='printf marker' log"),
        ("ls && pwd", "ls"),
    ],
)
async def test_changed_command_requires_approval(
    server: AgentServerACP, approved: str, changed: str
) -> None:
    assert await request(server, {"command": approved}) == [{"type": "approve"}]
    server._conn.request_permission.reset_mock()
    assert await request(server, {"command": changed}, "reject") == [{"type": "reject"}]
    server._conn.request_permission.assert_awaited_once()
    assert server._conn.request_permission.call_args.kwargs["tool_call"].raw_input == {
        "command": changed
    }


@pytest.mark.parametrize("command", ["ls", "ls && pwd", "printf 'a;b'", "ls #'", "ls\npwd"])
async def test_identical_approved_command_reuses_permission(
    server: AgentServerACP, command: str
) -> None:
    assert await request(server, {"command": command}) == [{"type": "approve"}]
    server._conn.request_permission.reset_mock()
    assert await request(server, {"command": command}, "reject") == [{"type": "approve"}]
    server._conn.request_permission.assert_not_awaited()


@pytest.mark.parametrize(
    ("approved", "changed"),
    [
        ({"command": "ls"}, {"command": "ls", "timeout": 60}),
        ({"command": "ls", "timeout": 10}, {"command": "ls", "timeout": 20}),
        ({"command": "ls", "cwd": "/a"}, {"command": "ls", "cwd": "/b"}),
        ({"command": "ls", "env": {"PATH": "/a"}}, {"command": "ls", "env": {"PATH": "/b"}}),
    ],
)
async def test_changed_arguments_require_approval(
    server: AgentServerACP, approved: dict[str, object], changed: dict[str, object]
) -> None:
    await request(server, approved)
    server._conn.request_permission.reset_mock()
    assert await request(server, changed, "reject") == [{"type": "reject"}]
    server._conn.request_permission.assert_awaited_once()


async def test_argument_key_order_does_not_change_command(server: AgentServerACP) -> None:
    await request(server, {"command": "ls", "timeout": 10})
    server._conn.request_permission.reset_mock()
    assert await request(server, {"timeout": 10, "command": "ls"}) == [{"type": "approve"}]
    server._conn.request_permission.assert_not_awaited()


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("_session_cwds", "/other"), ("_session_modes", "different"), ("_session_models", "other")],
)
async def test_changed_session_context_requires_approval(
    server: AgentServerACP, attribute: str, value: str
) -> None:
    await request(server, {"command": "ls"})
    getattr(server, attribute)["session"] = value
    server._conn.request_permission.reset_mock()
    assert await request(server, {"command": "ls"}, "reject") == [{"type": "reject"}]
    server._conn.request_permission.assert_awaited_once()


async def test_other_session_cannot_reuse_permission(server: AgentServerACP) -> None:
    await request(server, {"command": "ls"})
    server._session_cwds["other"] = "/project"
    server._conn.request_permission.reset_mock()
    assert await request(server, {"command": "ls"}, "reject", "other") == [{"type": "reject"}]
    server._conn.request_permission.assert_awaited_once()


@pytest.mark.parametrize("option", ["approve", "reject", "unknown"])
async def test_only_always_allow_grants_reusable_permission(
    server: AgentServerACP, option: str
) -> None:
    expected = "approve" if option == "approve" else "reject"
    assert await request(server, {"command": "ls"}, option) == [{"type": expected}]
    server._conn.request_permission.reset_mock()
    assert await request(server, {"command": "ls"}, "reject") == [{"type": "reject"}]
    server._conn.request_permission.assert_awaited_once()
    assert not server._allowed_execute_commands


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"command": ""},
        {"command": "  "},
        {"command": None},
        {"command": 42},
        {"command": "ls", "extra": object()},
        {"command": "ls", "extra": (1, 2)},
        {"command": "ls", "extra": {1: "value"}},
        {"command": "ls", "extra": float("nan")},
    ],
)
async def test_invalid_request_has_no_always_allow_option(
    server: AgentServerACP, args: dict[str, object]
) -> None:
    assert await request(server, args) == [{"type": "reject"}]
    options = server._conn.request_permission.call_args.kwargs["options"]
    assert {option.option_id for option in options} == {"approve", "reject"}
    assert not server._allowed_execute_commands
    assert not server._allowed_command_types


async def test_missing_context_cannot_grant_reusable_permission(server: AgentServerACP) -> None:
    server._session_cwds.clear()
    assert await request(server, {"command": "ls"}) == [{"type": "reject"}]
    assert not server._allowed_execute_commands


async def test_legacy_command_types_cannot_authorize_execute(server: AgentServerACP) -> None:
    server._allowed_command_types["session"] = {("execute", "ls"), ("execute", None)}
    assert await request(server, {"command": "ls"}, "reject") == [{"type": "reject"}]
    server._conn.request_permission.assert_awaited_once()


async def test_full_command_is_sent_for_review(server: AgentServerACP) -> None:
    command = "printf " + "x" * 200 + "; printf unapproved_suffix #'"
    await request(server, {"command": command})
    call = server._conn.request_permission.call_args.kwargs
    assert call["tool_call"].title == f"Execute: `{command}`"
    assert call["tool_call"].raw_input == {"command": command}
    assert call["options"][-1].name == "Always allow this exact command in this session"


async def test_permission_error_cannot_grant_approval(server: AgentServerACP) -> None:
    server._conn.request_permission.side_effect = RequestError(400, "cancelled")
    with pytest.raises(RequestError):
        await request(server, {"command": "ls"})
    assert not server._allowed_execute_commands


async def test_client_cancellation_rejects(server: AgentServerACP) -> None:
    server._conn.request_permission.side_effect = lambda **kwargs: RequestPermissionResponse(
        outcome=DeniedOutcome(outcome="cancelled")
    )
    assert await request(server, {"command": "ls"}) == [{"type": "reject"}]
    assert not server._allowed_execute_commands


async def test_forgetting_session_discards_permission(server: AgentServerACP) -> None:
    await request(server, {"command": "ls"})
    server._forget_session("session")
    server._session_cwds["session"] = "/project"
    server._conn.request_permission.reset_mock()
    assert await request(server, {"command": "ls"}, "reject") == [{"type": "reject"}]
    server._conn.request_permission.assert_awaited_once()
