"""Tests for the eval registry infrastructure and mock tool extraction.

These tests verify the *machinery* — EvalSpec.build(), mock tool module
exports — without hardcoding specific eval names or configurations, so they
don't break when evals are added, removed, or reconfigured.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

# ruff: noqa: ANN401, ARG001
from deepagents_evals.eval_registry import EVALS, EvalSpec
from deepagents_evals.mock_tools import (
    INCIDENT_GRAPH_TOOLS,
    RELATIONAL_TOOLS,
    TOOL_SELECTION_TOOLS,
    get_weather_fake,
    incident_graph_tool_error_middleware,
)

# ---------------------------------------------------------------------------
# Registry shape (not contents)
# ---------------------------------------------------------------------------


def test_registry_is_non_empty() -> None:
    """Registry has at least one eval."""
    assert len(EVALS) > 0


def test_registry_values_are_eval_specs() -> None:
    """Every registry value is an EvalSpec."""
    for spec in EVALS.values():
        assert isinstance(spec, EvalSpec)


def test_registry_keys_are_test_names() -> None:
    """Every registry key starts with 'test_'."""
    for name in EVALS:
        assert name.startswith("test_"), f"Unexpected registry key: {name!r}"


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
# EvalSpec.build() mechanics — uses synthetic specs, not registry lookups
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal stand-in for BaseChatModel to test build() without network."""

    def __str__(self) -> str:
        return "FakeModel"


@pytest.fixture
def fake_model() -> Any:
    return _FakeModel()


def test_build_default_calls_create_deep_agent(fake_model: Any) -> None:
    """A bare EvalSpec.build() calls create_deep_agent with just model."""
    spec = EvalSpec(name="synthetic")
    with patch("deepagents_evals.eval_registry.create_deep_agent") as mock_create:
        spec.build(fake_model)
        mock_create.assert_called_once_with(model=fake_model)


def test_build_passes_system_prompt(fake_model: Any) -> None:
    """build() forwards system_prompt to create_deep_agent."""
    spec = EvalSpec(name="synthetic", system_prompt="You are a test agent.")
    with patch("deepagents_evals.eval_registry.create_deep_agent") as mock_create:
        spec.build(fake_model)
        mock_create.assert_called_once_with(
            model=fake_model,
            system_prompt="You are a test agent.",
        )


def test_build_passes_memory(fake_model: Any) -> None:
    """build() forwards memory to create_deep_agent."""
    spec = EvalSpec(name="synthetic", memory=["/project/AGENTS.md"])
    with patch("deepagents_evals.eval_registry.create_deep_agent") as mock_create:
        spec.build(fake_model)
        mock_create.assert_called_once_with(
            model=fake_model,
            memory=["/project/AGENTS.md"],
        )


def test_build_passes_skills(fake_model: Any) -> None:
    """build() forwards skills to create_deep_agent."""
    spec = EvalSpec(name="synthetic", skills=["/skills/user/"])
    with patch("deepagents_evals.eval_registry.create_deep_agent") as mock_create:
        spec.build(fake_model)
        mock_create.assert_called_once_with(
            model=fake_model,
            skills=["/skills/user/"],
        )


def test_build_passes_tools(fake_model: Any) -> None:
    """build() forwards tools to create_deep_agent."""
    spec = EvalSpec(name="synthetic", tools=TOOL_SELECTION_TOOLS)
    with patch("deepagents_evals.eval_registry.create_deep_agent") as mock_create:
        spec.build(fake_model)
        mock_create.assert_called_once_with(
            model=fake_model,
            tools=TOOL_SELECTION_TOOLS,
        )


def test_build_passes_middleware(fake_model: Any) -> None:
    """build() forwards middleware to create_deep_agent."""
    mw = [incident_graph_tool_error_middleware]
    spec = EvalSpec(name="synthetic", middleware=mw)
    with patch("deepagents_evals.eval_registry.create_deep_agent") as mock_create:
        spec.build(fake_model)
        mock_create.assert_called_once_with(
            model=fake_model,
            middleware=mw,
        )


def test_build_with_builder_overrides_fields(fake_model: Any) -> None:
    """When a builder is set, it takes precedence over field-based config."""
    sentinel = object()

    def _builder(model: Any, **kwargs: Any) -> object:
        return sentinel

    spec = EvalSpec(name="synthetic", builder=_builder)
    result = spec.build(fake_model)
    assert result is sentinel


def test_build_repl_name_passed_to_builder(fake_model: Any) -> None:
    """build() forwards repl_name to the builder."""
    captured: dict[str, Any] = {}

    def _builder(model: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    spec = EvalSpec(name="synthetic", builder=_builder, supports_repl=True)
    spec.build(fake_model, repl_name="quickjs")
    assert captured.get("repl_name") == "quickjs"


def test_build_repl_name_defaults_to_none(fake_model: Any) -> None:
    """build() defaults repl_name to None when not provided."""
    captured: dict[str, Any] = {}

    def _builder(model: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    spec = EvalSpec(name="synthetic", builder=_builder, supports_repl=True)
    spec.build(fake_model)
    assert captured.get("repl_name") is None
