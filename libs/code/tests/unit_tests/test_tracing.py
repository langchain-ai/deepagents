"""Tests for trace grouping helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.types import Command

from deepagents_code._tracing import RESUME_TRACE_TAG, stream_trace_config

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


def _turn_config() -> RunnableConfig:
    """Build a stream config shaped like one `build_stream_config` returns."""
    return {
        "configurable": {"thread_id": "thread-1"},
        "metadata": {
            "thread_id": "thread-1",
            "turn_id": "turn-1",
            "turn_number": 1,
        },
        "tags": ["existing"],
    }


def test_initial_run_is_returned_untagged() -> None:
    """A turn's first round carries no resume marker."""
    config = _turn_config()

    assert stream_trace_config(config, {"messages": []}) is config


def test_resume_tags_a_copy_and_leaves_the_callers_config_alone() -> None:
    """Tagging must not leak the marker into the config reused next round."""
    config = _turn_config()

    resumed = stream_trace_config(config, Command(resume=[]))

    assert resumed is not config
    assert resumed["tags"] == ["existing", RESUME_TRACE_TAG]
    assert config["tags"] == ["existing"]


def test_resume_preserves_turn_grouping_keys() -> None:
    """Sibling roots only group if every round keeps the same turn identity."""
    config = _turn_config()

    resumed = stream_trace_config(config, Command(resume=[]))

    assert resumed["metadata"] == config["metadata"]
    assert resumed["configurable"] == config["configurable"]


def test_repeated_tagging_does_not_duplicate_the_marker() -> None:
    """A base config that already carries the marker gains no second copy."""
    config: RunnableConfig = {"tags": [RESUME_TRACE_TAG]}

    resumed = stream_trace_config(config, Command(resume=[]))

    assert resumed["tags"] == [RESUME_TRACE_TAG]


def test_resume_tags_a_config_that_has_no_tags_key() -> None:
    """`build_stream_config` emits no `tags`, so the absent-key path is live."""
    config: RunnableConfig = {"configurable": {"thread_id": "thread-1"}}

    resumed = stream_trace_config(config, Command(resume=[]))

    assert resumed["tags"] == [RESUME_TRACE_TAG]
    assert "tags" not in config
