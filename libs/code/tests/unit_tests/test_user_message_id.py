"""Tests that HumanMessage IDs are stable across resumed conversations.

The root fix lives in LangGraph (ensure_message_ids in put_writes()), which
assigns a stable UUID to id=None BaseMessages before they are serialised to
the checkpoint. This test verifies the end-to-end property: the same human
message keeps its ID when a thread is resumed across multiple invocations.
"""
from __future__ import annotations

import asyncio
import re

import pytest
from langchain_core.messages import HumanMessage, convert_to_messages


UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def test_convert_to_messages_preserves_id_field() -> None:
    """A user-role dict with an explicit 'id' key produces a HumanMessage
    whose .id matches — the pipeline the LangGraph fix relies on."""
    from uuid import uuid4
    msg_id = str(uuid4())
    [msg] = convert_to_messages([{"role": "user", "content": "hello", "id": msg_id}])
    assert isinstance(msg, HumanMessage)
    assert msg.id == msg_id


def test_convert_to_messages_without_id_gives_none() -> None:
    """Without an explicit id, the LangGraph checkpointer layer assigns it."""
    [msg] = convert_to_messages([{"role": "user", "content": "hello"}])
    assert isinstance(msg, HumanMessage)
    assert msg.id is None
