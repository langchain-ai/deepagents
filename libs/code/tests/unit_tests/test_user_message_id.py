"""Tests that user-facing input construction assigns stable IDs to HumanMessages.

The reducer assigns a fresh random UUID to any message that arrives with
id=None, so two separate graph invocations with identical content produce
different IDs. The fix: stamp a UUID onto the user_msg dict before it enters
the graph so the reducer receives an already-identified message.
"""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage, convert_to_messages

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _is_uuid4(value: str | None) -> bool:
    return bool(value and UUID_RE.match(value))


# ---------------------------------------------------------------------------
# Mechanism test: verifies the dict → HumanMessage pipeline preserves "id"
# ---------------------------------------------------------------------------


def test_convert_to_messages_preserves_id_field() -> None:
    """A user-role dict with an explicit 'id' key produces a matching HumanMessage.

    This is the mechanism the call-site fix relies on.
    """
    msg_id = str(uuid.uuid4())
    [msg] = convert_to_messages([{"role": "user", "content": "hello", "id": msg_id}])
    assert isinstance(msg, HumanMessage)
    assert msg.id == msg_id, f"expected id={msg_id!r}, got {msg.id!r}"


def test_convert_to_messages_without_id_gives_none() -> None:
    """Baseline: without the fix the user dict has no 'id'.

    The resulting HumanMessage has id=None and the reducer assigns a fresh
    UUID each run.
    """
    [msg] = convert_to_messages([{"role": "user", "content": "hello"}])
    assert isinstance(msg, HumanMessage)
    assert msg.id is None, "pre-fix baseline: id should be None when not supplied"


# ---------------------------------------------------------------------------
# Call-site tests: non_interactive and textual_adapter stamp a UUID
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_agent_loop_user_msg_has_id() -> None:
    """_run_agent_loop must include a UUID4 'id' in the user_msg dict.

    It passes the dict to agent.astream so the LangGraph reducer receives an
    already-identified HumanMessage rather than id=None.
    """
    from deepagents_code.non_interactive import _run_agent_loop

    captured: list[dict] = []

    async def fake_astream(  # noqa: RUF029  # must be async to replace agent.astream
        stream_input: dict, **_kwargs: Any
    ) -> AsyncIterator[Any]:
        captured.append(stream_input)
        for _ in ():
            yield

    agent = MagicMock()
    agent.astream = fake_astream

    await _run_agent_loop(
        agent=agent,
        message="hello world",
        config={"configurable": {"thread_id": "t1"}},
        console=MagicMock(),
        file_op_tracker=MagicMock(),
        quiet=True,
    )

    assert captured, "agent.astream should have been called"
    msgs = captured[0].get("messages", [])
    assert msgs, "stream_input must contain messages"
    user_msg = msgs[0]
    assert "id" in user_msg, (
        f"user_msg dict must include 'id' so the reducer receives a stable "
        f"HumanMessage ID; got keys: {list(user_msg)}"
    )
    assert _is_uuid4(user_msg["id"]), (
        f"'id' must be a UUID4 string, got {user_msg['id']!r}"
    )
