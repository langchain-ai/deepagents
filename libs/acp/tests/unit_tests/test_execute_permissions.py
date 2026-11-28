"""Exercise exact-command permissions through the ACP approval path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from acp.exceptions import RequestError
from acp.schema import (
    AllowedOutcome,
    DeniedOutcome,
    RequestPermissionResponse,
    TextContentBlock,
)
from langgraph.types import Command, Interrupt

from deepagents_acp._permissions import _Permissions
from deepagents_acp.server import AgentServerACP, _Session

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from deepagents_acp.server import AgentSessionContext


@dataclass
class ExecuteGate:
    """A graph whose every turn asks to run `args` and records the decisions it gets."""

    args: object = None
    decisions: list[Any] = field(default_factory=list)
    before_asking: Callable[[], None] = lambda: None
    checkpointer: object = True

    async def astream(self, graph_input: Any, **kwargs: Any) -> AsyncIterator[Any]:
        if isinstance(graph_input, Command):
            [answer] = graph_input.resume.values()
            self.decisions.append(answer["decisions"])
            return
        self.before_asking()
        value = {"action_requests": [{"name": "execute", "args": self.args}]}
        yield ((), "updates", {"__interrupt__": (Interrupt(value=value, id="gate"),)})

    async def aget_state(self, config: Any) -> Any:
        return None


class GatedServer(AgentServerACP):
    """A server whose sessions all run one `ExecuteGate`."""

    def __init__(self) -> None:
        self.gate = ExecuteGate()
        models = [{"value": "default", "name": "Default"}, {"value": "other", "name": "Other"}]
        super().__init__(agent=self._build, models=models)

    def _build(self, context: AgentSessionContext) -> Any:
        return self.gate


@pytest.fixture
async def server() -> GatedServer:
    server = GatedServer()
    server.on_connect(AsyncMock())
    await new_session(server)
    return server


async def new_session(server: GatedServer) -> str:
    return (await server.new_session(cwd="/project", mcp_servers=[])).session_id


def first_session(server: GatedServer) -> str:
    return next(iter(server._sessions))


async def request(
    server: GatedServer,
    args: dict[str, object],
    option: str = "approve_always",
    session: str | None = None,
) -> list[dict[str, str]]:
    """Prompt a turn that asks to run `args`, answering with `option`."""
    server.gate.args = args
    server._conn.request_permission.return_value = RequestPermissionResponse(
        outcome=AllowedOutcome(outcome="selected", option_id=option)
    )
    prompt = [TextContentBlock(type="text", text="run it")]
    await server.prompt(prompt, session_id=session or first_session(server))
    return server.gate.decisions.pop()


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
    server: GatedServer, approved: str, changed: str
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
    server: GatedServer, command: str
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
    server: GatedServer, approved: dict[str, object], changed: dict[str, object]
) -> None:
    await request(server, approved)
    server._conn.request_permission.reset_mock()
    assert await request(server, changed, "reject") == [{"type": "reject"}]
    server._conn.request_permission.assert_awaited_once()


async def test_argument_key_order_does_not_change_command(server: GatedServer) -> None:
    await request(server, {"command": "ls", "timeout": 10})
    server._conn.request_permission.reset_mock()
    assert await request(server, {"timeout": 10, "command": "ls"}) == [{"type": "approve"}]
    server._conn.request_permission.assert_not_awaited()


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("cwd", "/other"),
        ("mode", "different"),
        ("model", "other"),
        ("options", {"effort": "high"}),
    ],
)
async def test_changed_session_context_requires_approval(
    server: GatedServer, attribute: str, value: object
) -> None:
    await request(server, {"command": "ls"})
    setattr(server._sessions[first_session(server)], attribute, value)
    server._conn.request_permission.reset_mock()
    assert await request(server, {"command": "ls"}, "reject") == [{"type": "reject"}]
    server._conn.request_permission.assert_awaited_once()


async def test_other_session_cannot_reuse_permission(server: GatedServer) -> None:
    await request(server, {"command": "ls"})
    other = await new_session(server)
    server._conn.request_permission.reset_mock()
    assert await request(server, {"command": "ls"}, "reject", other) == [{"type": "reject"}]
    server._conn.request_permission.assert_awaited_once()


@pytest.mark.parametrize("option", ["approve", "reject", "unknown"])
async def test_only_always_allow_grants_reusable_permission(
    server: GatedServer, option: str
) -> None:
    expected = "approve" if option == "approve" else "reject"
    assert await request(server, {"command": "ls"}, option) == [{"type": expected}]
    server._conn.request_permission.reset_mock()
    assert await request(server, {"command": "ls"}, "reject") == [{"type": "reject"}]
    server._conn.request_permission.assert_awaited_once()


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
    server: GatedServer, args: dict[str, object]
) -> None:
    assert await request(server, args) == [{"type": "reject"}]
    options = server._conn.request_permission.call_args.kwargs["options"]
    assert {option.option_id for option in options} == {"approve", "reject"}
    assert server._sessions[first_session(server)].permissions == _Permissions()


async def test_session_forgotten_mid_turn_cannot_grant_reusable_permission(
    server: GatedServer,
) -> None:
    session_id = first_session(server)
    server.gate.before_asking = server._sessions.clear
    assert await request(server, {"command": "ls"}, session=session_id) == [{"type": "reject"}]
    assert server._sessions == {}


async def test_tool_wide_approval_cannot_authorize_execute(server: GatedServer) -> None:
    server._sessions[first_session(server)].permissions.allowed_tools.add("execute")
    assert await request(server, {"command": "ls"}, "reject") == [{"type": "reject"}]
    server._conn.request_permission.assert_awaited_once()


async def test_full_command_is_sent_for_review(server: GatedServer) -> None:
    command = "printf " + "x" * 200 + "; printf unapproved_suffix #'"
    await request(server, {"command": command})
    call = server._conn.request_permission.call_args.kwargs
    assert call["tool_call"].title == f"Execute: `{command}`"
    assert call["tool_call"].raw_input == {"command": command}
    names = {option.option_id: option.name for option in call["options"]}
    assert names["approve_always"] == "Always allow this exact command in this session"


async def test_permission_error_cannot_grant_approval(server: GatedServer) -> None:
    server._conn.request_permission.side_effect = RequestError(400, "cancelled")
    with pytest.raises(RequestError):
        await request(server, {"command": "ls"})
    assert server._sessions[first_session(server)].permissions == _Permissions()


async def test_client_cancellation_rejects(server: GatedServer) -> None:
    server._conn.request_permission.side_effect = lambda **kwargs: RequestPermissionResponse(
        outcome=DeniedOutcome(outcome="cancelled")
    )
    assert await request(server, {"command": "ls"}) == [{"type": "reject"}]
    assert server._sessions[first_session(server)].permissions == _Permissions()


async def test_forgetting_session_discards_permission(server: GatedServer) -> None:
    session_id = first_session(server)
    await request(server, {"command": "ls"})
    server._forget_session(session_id)
    server._sessions[session_id] = _Session(cwd="/project")
    server._conn.request_permission.reset_mock()
    assert await request(server, {"command": "ls"}, "reject") == [{"type": "reject"}]
    server._conn.request_permission.assert_awaited_once()
