"""Ensure first-party tools expose per-argument schema descriptions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated
from unittest.mock import Mock

import pytest
from langchain_core.tools import BaseTool, StructuredTool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from deepagents_code.ask_user import AskUserMiddleware
from deepagents_code.auto_mode import AutoModeHITLMiddleware
from deepagents_code.goal_tools import GoalToolsMiddleware
from deepagents_code.offload_middleware import CLICompactionMiddleware
from deepagents_code.tools import fetch_url, get_current_thread_id, web_search

if TYPE_CHECKING:
    from pathlib import Path

    from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig

ToolLike = BaseTool | Callable[..., object]


def _as_tool(candidate: ToolLike) -> BaseTool:
    """Normalize bare callables the same way agent construction does."""
    if isinstance(candidate, BaseTool):
        return candidate
    return StructuredTool.from_function(func=candidate)


def _model_properties(candidate: ToolLike) -> dict[str, object]:
    """Return OpenAI-style parameter properties for a tool."""
    schema = convert_to_openai_tool(_as_tool(candidate))
    params = schema["function"]["parameters"]
    assert isinstance(params, dict)
    props = params.get("properties") or {}
    assert isinstance(props, dict)
    return props


def _assert_model_args_described(
    candidate: ToolLike, *, expected: set[str] | None = None
) -> None:
    """Assert every model-visible property has a non-empty description."""
    props = _model_properties(candidate)
    if expected is not None:
        assert set(props) == expected
    missing = [
        name
        for name, spec in props.items()
        if not isinstance(spec, dict) or not str(spec.get("description") or "").strip()
    ]
    tool_name = getattr(_as_tool(candidate), "name", type(candidate).__name__)
    assert not missing, f"{tool_name} missing property descriptions: {missing}"


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (get_current_thread_id, set()),
        (
            web_search,
            {"query", "max_results", "topic", "include_raw_content"},
        ),
        (fetch_url, {"url", "timeout"}),
    ],
)
def test_core_custom_tools_have_described_args(
    candidate: ToolLike, expected: set[str]
) -> None:
    """`tools.py` callables should expose Field descriptions in the model schema."""
    _assert_model_args_described(candidate, expected=expected)


def test_goal_tools_have_described_model_args() -> None:
    """Goal tools keep injected state hidden and describe model-facing fields."""
    middleware = GoalToolsMiddleware()
    by_name = {item.name: item for item in middleware.tools}
    _assert_model_args_described(by_name["update_goal"], expected={"status", "note"})


def test_ask_user_has_described_questions_arg() -> None:
    """`ask_user` should describe its questions list at the property level."""
    middleware = AskUserMiddleware()
    _assert_model_args_described(middleware.tools[0], expected={"questions"})


def test_compact_conversation_has_no_model_args() -> None:
    """Forced-compaction injects handle model-facing args only via runtime."""
    middleware = CLICompactionMiddleware(Mock())
    compact = next(t for t in middleware.tools if t.name == "compact_conversation")
    _assert_model_args_described(compact, expected=set())


def test_auto_mode_temp_artifact_tools_have_described_args(
    tmp_path: Path,
) -> None:
    """Auto mode scratch tools expose Field descriptions for model args."""
    config: InterruptOnConfig = {"allowed_decisions": ["approve", "reject"]}
    middleware = AutoModeHITLMiddleware(
        {
            "compact_conversation": config,
            "delete": config,
            "execute": config,
            "write_file": config,
            "edit_file": config,
            "task": config,
            "mcp_mutate": config,
            "mcp_read": config,
        },
        worktree_root=tmp_path,
        classifier_timeout_seconds=1,
    )
    by_name = {item.name: item for item in middleware.tools}
    _assert_model_args_described(
        by_name["create_temp_artifact"], expected={"content", "suffix"}
    )
    _assert_model_args_described(
        by_name["delete_temp_artifact"], expected={"file_path"}
    )


def test_annotated_field_survives_from_function_wrapper() -> None:
    """Regression: bare `from_function` keeps Annotated Field descriptions."""

    def sample(value: str) -> str:
        """Sample tool."""
        return value

    def sample_described(
        value: Annotated[str, Field(description="A described value.")],
    ) -> str:
        """Sample tool."""
        return value

    plain = tool(sample)
    described = StructuredTool.from_function(func=sample_described)
    plain_props = convert_to_openai_tool(plain)["function"]["parameters"]["properties"]
    described_props = convert_to_openai_tool(described)["function"]["parameters"][
        "properties"
    ]
    assert "description" not in plain_props["value"]
    assert described_props["value"]["description"] == "A described value."
