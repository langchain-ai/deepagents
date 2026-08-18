"""Tests for trace grouping helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_code._tracing import RESUME_TRACE_TAG, resume_trace_config

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


def test_resume_trace_config_preserves_turn_metadata_and_existing_tags() -> None:
    """A resume round keeps turn grouping data while adding its marker."""
    metadata = {
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "turn_number": 1,
    }
    config: RunnableConfig = {
        "configurable": {"thread_id": "thread-1"},
        "metadata": metadata,
        "tags": ["existing"],
    }

    resumed = resume_trace_config(config)

    assert resumed is not config
    assert resumed["tags"] == ["existing", RESUME_TRACE_TAG]
    assert config["tags"] == ["existing"]
    assert resumed["metadata"] is metadata


def test_resume_trace_config_does_not_duplicate_tag() -> None:
    """Repeated derivation leaves exactly one resume marker."""
    config: RunnableConfig = {"tags": [RESUME_TRACE_TAG]}

    resumed = resume_trace_config(resume_trace_config(config))

    assert resumed["tags"] == [RESUME_TRACE_TAG]
