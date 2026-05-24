"""Unit tests for _messages_delta_reducer."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from deepagents._messages_reducer import _messages_delta_reducer


def test_idless_message_gets_id_and_order_preserved() -> None:
    existing = [AIMessage(content="hello", id="existing-1")]
    new_msg = HumanMessage(content="follow-up")  # no id

    result = _messages_delta_reducer(existing, [[new_msg]])

    assert len(result) == 2
    assert result[0].id == "existing-1"
    assert result[1] is new_msg
    assert result[1].id is not None


def test_state_is_none() -> None:
    new_msg = HumanMessage(content="hello")

    result = _messages_delta_reducer(None, [[new_msg]])

    assert len(result) == 1
    assert result[0] is new_msg
    assert result[0].id is not None
