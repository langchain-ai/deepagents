"""Tests for forwarding delta-channel history through langgraph-api's adapter."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated
from unittest.mock import patch

import pytest
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph

from deepagents_code._checkpointer_adapter import install_delta_history_forwarding

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from langchain_core.runnables import RunnableConfig


def _extend(base: list[str] | None, writes: Sequence[list[str]]) -> list[str]:
    return [*(base or []), *[item for batch in writes for item in batch]]


@dataclass
class _State:
    turns: Annotated[int, operator.add]
    items: Annotated[list[str], DeltaChannel(_extend, snapshot_frequency=4)] = field(
        default_factory=list
    )


def _adapter(inner: BaseCheckpointSaver) -> BaseCheckpointSaver:
    # Imported late: `langgraph_api.config` needs the env set by `_server_env`.
    from langgraph_api._checkpointer._adapter import (
        CheckpointerCapabilities,
        _CustomCheckpointerAdapter,
    )

    return _CustomCheckpointerAdapter(
        inner, CheckpointerCapabilities.from_type(type(inner))
    )


@pytest.fixture(autouse=True)
def _server_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Satisfy `langgraph_api.config`, which reads server env vars on import."""
    monkeypatch.setenv("REDIS_URI", "fake")


async def _seed_thread(saver: AsyncSqliteSaver, config: RunnableConfig) -> None:
    builder = StateGraph(_State)
    builder.add_node("step", lambda state: {"items": [f"i{state.turns}"]})
    builder.add_edge("__start__", "step")
    graph = builder.compile(checkpointer=saver)
    for _ in range(11):
        await graph.ainvoke(_State(turns=1), config)


async def test_adapter_uses_inner_fast_path_with_same_values(tmp_path: Path) -> None:
    config: RunnableConfig = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "s.db")) as inner:
        await _seed_thread(inner, config)
        adapter = _adapter(inner)
        walked = await BaseCheckpointSaver.aget_delta_channel_history(
            adapter, config=config, channels=["items"]
        )
        install_delta_history_forwarding()
        with patch.object(
            AsyncSqliteSaver,
            "aget_delta_channel_history",
            autospec=True,
            side_effect=AsyncSqliteSaver.aget_delta_channel_history,
        ) as fast:
            forwarded = await adapter.aget_delta_channel_history(
                config=config, channels=["items"]
            )

    fast.assert_awaited_once()
    assert walked["items"]["writes"]
    assert forwarded == walked


async def test_adapter_keeps_base_walk_without_inner_override(
    tmp_path: Path,
) -> None:
    class PlainSaver(AsyncSqliteSaver):
        aget_delta_channel_history = BaseCheckpointSaver.aget_delta_channel_history

    config: RunnableConfig = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    async with PlainSaver.from_conn_string(str(tmp_path / "s.db")) as inner:
        await _seed_thread(inner, config)
        adapter = _adapter(inner)
        install_delta_history_forwarding()
        history = await adapter.aget_delta_channel_history(
            config=config, channels=["items"]
        )
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "s.db")) as fast:
            expected = await fast.aget_delta_channel_history(
                config=config, channels=["items"]
            )

    assert history == expected
