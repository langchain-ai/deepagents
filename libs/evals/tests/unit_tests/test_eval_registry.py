"""Tests for the eval registry and mock tool exports.

These tests verify the registry shape (keys, callable values) and mock tool
exports — without hardcoding specific eval configurations, so they don't break
when evals are added, removed, or reconfigured.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from deepagents_evals.eval_registry import EVALS
from deepagents_evals.mock_tools import (
    INCIDENT_GRAPH_TOOLS,
    RELATIONAL_TOOLS,
    TOOL_SELECTION_TOOLS,
    get_weather_fake,
    incident_graph_tool_error_middleware,
)
from deepagents_evals.mock_tools.bfcl import BFCL_CLASS_REGISTRY, make_bfcl_tools

# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


def test_registry_is_non_empty() -> None:
    """Registry has at least one eval."""
    assert len(EVALS) > 0


def test_registry_values_are_callables() -> None:
    """Every registry value is callable."""
    for builder in EVALS.values():
        assert callable(builder)


def test_registry_keys_are_test_names() -> None:
    """Every registry key starts with 'test_'."""
    for name in EVALS:
        assert name.startswith("test_"), f"Unexpected registry key: {name!r}"


def test_skill_path_eval_registered() -> None:
    """The multi-path skill eval is registered under its pytest eval name."""
    assert "test_find_skill_in_correct_path" in EVALS
    assert "test_edit_correct_skill_from_multiple_sources" not in EVALS


# ---------------------------------------------------------------------------
# Mock tool exports
# ---------------------------------------------------------------------------


def test_tool_selection_tools_non_empty() -> None:
    """Tool selection module exports a non-empty tool list."""
    assert len(TOOL_SELECTION_TOOLS) > 0


def test_relational_tools_non_empty() -> None:
    """Relational module exports a non-empty tool list."""
    assert len(RELATIONAL_TOOLS) > 0


def test_incident_graph_tools_non_empty() -> None:
    """Incident graph module exports a non-empty tool list."""
    assert len(INCIDENT_GRAPH_TOOLS) > 0


def test_weather_tool_exists() -> None:
    """Subagents module exports the weather tool."""
    assert get_weather_fake.name == "get_weather_fake"


def test_incident_graph_middleware_exists() -> None:
    """Incident graph module exports the error middleware."""
    assert incident_graph_tool_error_middleware is not None


# ---------------------------------------------------------------------------
# Builder behavior
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal stand-in for BaseChatModel to test builders without network."""

    def __str__(self) -> str:
        return "FakeModel"


@pytest.fixture
def fake_model() -> _FakeModel:
    return _FakeModel()


def test_bare_eval_builder_calls_create_deep_agent(fake_model: _FakeModel) -> None:
    """A simple eval builder calls create_deep_agent with just model."""
    builder = EVALS["test_read_file_seeded_state_backend_file"]
    with patch("deepagents_evals.eval_registry.create_deep_agent") as mock_create:
        builder(fake_model)
        mock_create.assert_called_once_with(model=fake_model)


def test_memory_eval_builder_passes_memory(fake_model: _FakeModel) -> None:
    """A memory eval builder forwards memory to create_deep_agent."""
    builder = EVALS["test_memory_basic_recall"]
    with patch("deepagents_evals.eval_registry.create_deep_agent") as mock_create:
        builder(fake_model)
        assert mock_create.call_args.kwargs["memory"] == ["/project/AGENTS.md"]


def test_composite_backend_builder_seeds_memory(fake_model: _FakeModel) -> None:
    """Composite-backend eval preserves its pytest memory setup."""
    builder = EVALS["test_memory_middleware_composite_backend"]
    with patch("deepagents_evals.eval_registry.create_deep_agent") as mock_create:
        builder(fake_model)

    store = mock_create.call_args.kwargs["store"]
    item = store.get(("filesystem",), "/AGENTS.md")

    assert item is not None
    assert item.value["content"] == ["Your name is Jackson"]


def test_make_bfcl_tools_scopes_to_involved_classes() -> None:
    """make_bfcl_tools binds only requested classes; default binds all."""
    full = make_bfcl_tools()
    scoped = make_bfcl_tools(involved_classes=["MessageAPI"])
    assert 0 < len(scoped) < len(full)
    assert len(make_bfcl_tools(involved_classes=list(BFCL_CLASS_REGISTRY))) == len(full)


def test_factory_produced_builders_are_unique_callables() -> None:
    """Factory-produced builders are distinct closures (catches copy-paste bugs).

    Shared builders like `_bare` may be registered under multiple eval names —
    that's intentional. But factory-produced builders (summarization,
    memory_bench, todo_middleware) must be distinct closures; sharing the same
    object means the factory wasn't called.
    """
    factory_names = {
        "test_summarize_continues_task",
        "test_summarization_offloads_to_filesystem",
        "test_compact_tool_new_task",
        "test_compact_tool_not_overly_sensitive",
        "test_compact_tool_large_reads",
        "test_conflict_resolution",
        "test_time_learning",
        "test_memory_agent_bench_ci",
        "test_memory_agent_bench_ci_fileseeded",
        "test_density_rank_lands_in_final_message",
        "test_population_compare_lands_in_final_message",
        "test_trivial_arithmetic_skips_write_todos",
        "test_rank_with_unknown_lookup_lands_in_final_message",
        "test_design_api_lands_in_final_message",
        "test_density_cairo_lands_in_final_message",
        "test_trivial_plan_skips_write_todos",
    }
    seen: dict[int, str] = {}
    for name in factory_names:
        builder = EVALS[name]
        obj_id = id(builder)
        if obj_id in seen:
            msg = (
                f"Builder for {name!r} is the same object as {seen[obj_id]!r} — "
                "did you forget to call the factory?"
            )
            raise AssertionError(msg)
        seen[obj_id] = name
