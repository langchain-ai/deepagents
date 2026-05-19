"""Unit tests for the swarm_task PTC tool.

Covers normalize_schema, VariantCache, and the tool created by
create_swarm_task_tool (subagent validation, agent mode dispatch,
invoke mode, schema-constrained variant caching).
"""

from __future__ import annotations

import json
from collections.abc import Iterator  # noqa: TC003
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from langchain_quickjs._swarm_task import (
    SwarmSubAgent,
    VariantCache,
    create_swarm_task_tool,
    normalize_schema,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeModel(GenericFakeChatModel):
    """Fake model that passes Pydantic validation on SwarmSubAgent.model."""

    messages: Iterator[AIMessage | str] = Field(
        default_factory=lambda: iter([AIMessage(content="")]), exclude=True
    )

    def bind_tools(self, tools: Any, **_: Any) -> _FakeModel:
        return self


def _make_mock_model(response: str = "model response") -> MagicMock:
    """Create a mock model that returns an AIMessage with the given content."""
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value=AIMessage(content=response))
    return model


def _make_runnable(afunc: Any) -> RunnableLambda:
    """Wrap an async function in a RunnableLambda (func arg is required)."""
    return RunnableLambda(func=lambda _: None, afunc=afunc)


def _make_fake_agent(
    content: str = "agent response",
    structured_response: Any = None,
) -> RunnableLambda:
    """Create a fake agent that returns a state dict with messages."""
    result: dict[str, Any] = {
        "messages": [AIMessage(content=content)],
    }
    if structured_response is not None:
        result["structured_response"] = structured_response

    async def _afunc(_: Any) -> dict[str, Any]:
        return result

    return _make_runnable(_afunc)


# ---------------------------------------------------------------------------
# normalizeSchema
# ---------------------------------------------------------------------------


class TestNormalizeSchema:
    """Tests for the normalize_schema function."""

    def test_adds_additional_properties_false_to_top_level_object(self) -> None:
        result = normalize_schema(
            {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            }
        )
        assert result["additionalProperties"] is False

    def test_preserves_existing_additional_properties_false(self) -> None:
        result = normalize_schema(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            }
        )
        assert result["additionalProperties"] is False

    def test_recurses_into_nested_object_properties(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "counts": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}},
                    "required": ["a"],
                },
            },
            "required": ["counts"],
        }
        result = normalize_schema(schema)
        counts = result["properties"]["counts"]
        assert counts["additionalProperties"] is False

    def test_recurses_into_array_items(self) -> None:
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        }
        result = normalize_schema(schema)
        items = result["items"]
        assert items["additionalProperties"] is False

    def test_handles_deeply_nested_objects(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "counts": {
                                "type": "object",
                                "properties": {"a": {"type": "number"}},
                            },
                        },
                    },
                },
            },
        }
        result = normalize_schema(schema)
        items = result["properties"]["results"]["items"]
        counts = items["properties"]["counts"]
        assert counts["additionalProperties"] is False

    def test_passes_through_non_object_array_types_unchanged(self) -> None:
        schema = {"type": "string"}
        assert normalize_schema(schema) == {"type": "string"}

    def test_preserves_min_items_on_array_types(self) -> None:
        assert (
            normalize_schema(
                {"type": "array", "minItems": 6, "items": {"type": "string"}}
            )["minItems"]
            == 6
        )
        assert (
            normalize_schema(
                {"type": "array", "minItems": 0, "items": {"type": "string"}}
            )["minItems"]
            == 0
        )
        assert (
            normalize_schema(
                {"type": "array", "minItems": 1, "items": {"type": "string"}}
            )["minItems"]
            == 1
        )

    def test_preserves_max_items_on_array_types(self) -> None:
        result = normalize_schema(
            {"type": "array", "maxItems": 10, "items": {"type": "string"}}
        )
        assert result["maxItems"] == 10


# ---------------------------------------------------------------------------
# VariantCache (isolated unit tests)
# ---------------------------------------------------------------------------


class TestVariantCache:
    """Tests for the VariantCache class."""

    def test_returns_factory_value_on_cache_miss(self) -> None:
        cache = VariantCache(ttl_s=60.0)
        value = cache.get_or_create("key1", lambda: "created")
        assert value == "created"
        assert cache.size == 1

    def test_returns_cached_value_on_hit_without_calling_factory(self) -> None:
        cache = VariantCache(ttl_s=60.0)
        cache.get_or_create("key1", lambda: "first")

        factory = MagicMock(return_value="second")
        value = cache.get_or_create("key1", factory)

        assert value == "first"
        factory.assert_not_called()

    def test_stores_separate_entries_for_different_keys(self) -> None:
        cache = VariantCache(ttl_s=60.0)
        cache.get_or_create("a", lambda: "alpha")
        cache.get_or_create("b", lambda: "beta")

        assert cache.size == 2
        assert cache.get_or_create("a", lambda: "unused") == "alpha"
        assert cache.get_or_create("b", lambda: "unused") == "beta"

    def test_evicts_entries_that_exceed_ttl(self) -> None:
        cache = VariantCache(ttl_s=1.0)

        with patch("langchain_quickjs._swarm_task.time") as mock_time:
            mock_time.monotonic = MagicMock(side_effect=[0.0, 0.0, 1.5, 1.5, 1.5])
            cache.get_or_create("key1", lambda: "value1")

            factory = MagicMock(return_value="value2")
            value = cache.get_or_create("key1", factory)

        assert value == "value2"
        factory.assert_called_once()

    def test_keeps_entries_alive_when_accessed_within_ttl(self) -> None:
        cache = VariantCache(ttl_s=1.0)

        with patch("langchain_quickjs._swarm_task.time") as mock_time:
            mock_time.monotonic = MagicMock(
                side_effect=[
                    0.0,
                    0.0,  # create key1 (sweep reads 0.0, set reads 0.0)
                    0.8,
                    0.8,
                    0.8,  # access at 0.8 — not expired
                    1.6,
                    1.6,
                    1.6,  # access at 1.6 — diff=0.8 < 1.0, alive
                ]
            )
            cache.get_or_create("key1", lambda: "value1")
            cache.get_or_create("key1", lambda: "unused")

            factory = MagicMock(return_value="replaced")
            value = cache.get_or_create("key1", factory)

        assert value == "value1"
        factory.assert_not_called()

    def test_sweeps_multiple_expired_entries_in_one_call(self) -> None:
        cache = VariantCache(ttl_s=1.0)

        with patch("langchain_quickjs._swarm_task.time") as mock_time:
            mock_time.monotonic = MagicMock(
                side_effect=[
                    0.0,
                    0.0,  # create a
                    0.0,
                    0.0,  # create b
                    0.0,
                    0.0,  # create c
                    1.5,
                    1.5,  # create d (sweep evicts a, b, c)
                ]
            )
            cache.get_or_create("a", lambda: "1")
            cache.get_or_create("b", lambda: "2")
            cache.get_or_create("c", lambda: "3")
            assert cache.size == 3

            cache.get_or_create("d", lambda: "4")

        assert cache.size == 1

    def test_only_evicts_expired_entries_not_active_ones(self) -> None:
        cache = VariantCache(ttl_s=1.0)

        with patch("langchain_quickjs._swarm_task.time") as mock_time:
            mock_time.monotonic = MagicMock(
                side_effect=[
                    0.0,
                    0.0,  # create "old": sweep(now=0.0) + set(0.0)
                    0.8,
                    0.8,  # create "new": old@0.0 diff<1.0, ok
                    1.1,
                    1.1,  # create "trigger": old evicted, new ok
                ]
            )
            cache.get_or_create("old", lambda: "stale")
            cache.get_or_create("new", lambda: "fresh")
            cache.get_or_create("trigger", lambda: "sweep")

        assert cache.size == 2


# ---------------------------------------------------------------------------
# Subagent validation
# ---------------------------------------------------------------------------


class TestSubagentValidation:
    """Tests for subagent validation in create_swarm_task_tool."""

    async def test_throws_when_subagent_type_not_in_configured_list(self) -> None:
        with patch(
            "langchain_quickjs._swarm_task.create_agent",
            return_value=_make_fake_agent(),
        ):
            tool = create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(
                        name="screener",
                        description="A screener",
                        system_prompt="Screen.",
                    ),
                ],
                default_model=_make_mock_model(),
            )

        with pytest.raises(
            Exception, match='Unknown swarm subagent type "nonexistent"'
        ):
            await tool.ainvoke(
                {"description": "do work", "subagent_type": "nonexistent"}
            )

    async def test_includes_available_subagent_names_in_error_message(self) -> None:
        with patch(
            "langchain_quickjs._swarm_task.create_agent",
            return_value=_make_fake_agent(),
        ):
            tool = create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(name="alpha", description="A", system_prompt="A."),
                    SwarmSubAgent(name="beta", description="B", system_prompt="B."),
                ],
                default_model=_make_mock_model(),
            )

        with pytest.raises(Exception, match="alpha, beta"):
            await tool.ainvoke({"description": "do work", "subagent_type": "gamma"})

    async def test_agent_mode_requires_subagent_type(self) -> None:
        with patch(
            "langchain_quickjs._swarm_task.create_agent",
            return_value=_make_fake_agent(),
        ):
            tool = create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(name="worker", description="W", system_prompt="W."),
                ],
                default_model=_make_mock_model(),
            )

        with pytest.raises(Exception, match="agent mode requires subagent_type"):
            await tool.ainvoke({"description": "do work"})

    async def test_agent_mode_none_configured_message(self) -> None:
        tool = create_swarm_task_tool(
            subagents=[],
            default_model=_make_mock_model(),
        )

        with pytest.raises(Exception, match=r"\(none configured\)"):
            await tool.ainvoke({"description": "do work"})


# ---------------------------------------------------------------------------
# Agent mode
# ---------------------------------------------------------------------------


class TestAgentMode:
    """Tests for agent dispatch mode (default)."""

    async def test_dispatches_to_correct_subagent_by_name(self) -> None:
        alpha_agent = _make_fake_agent("alpha result")
        beta_agent = _make_fake_agent("beta result")

        agents = [alpha_agent, beta_agent]
        agent_idx = 0

        def mock_create_agent(**_: Any) -> RunnableLambda:
            nonlocal agent_idx
            agent = agents[agent_idx]
            agent_idx += 1
            return agent

        with patch(
            "langchain_quickjs._swarm_task.create_agent", side_effect=mock_create_agent
        ):
            tool = create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(name="alpha", description="A", system_prompt="A."),
                    SwarmSubAgent(name="beta", description="B", system_prompt="B."),
                ],
                default_model=_make_mock_model(),
            )

        result = await tool.ainvoke({"description": "do work", "subagent_type": "beta"})
        assert result == "beta result"

    async def test_passes_description_as_human_message_content(self) -> None:
        invoked_with: list[Any] = []

        def capture_agent(**_: Any) -> RunnableLambda:
            async def run(state: dict[str, Any]) -> dict[str, Any]:
                invoked_with.append(state)
                return {"messages": [AIMessage(content="done")]}

            return _make_runnable(run)

        with patch(
            "langchain_quickjs._swarm_task.create_agent", side_effect=capture_agent
        ):
            tool = create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(name="worker", description="W", system_prompt="W.")
                ],
                default_model=_make_mock_model(),
            )

        await tool.ainvoke(
            {"description": "classify this trace", "subagent_type": "worker"}
        )

        state = invoked_with[-1]
        messages = state["messages"]
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "classify this trace"

    async def test_returns_structured_response_as_json_when_present(self) -> None:
        agent = _make_fake_agent(
            content="ignored",
            structured_response={"label": "positive", "score": 0.95},
        )

        with patch("langchain_quickjs._swarm_task.create_agent", return_value=agent):
            tool = create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(name="worker", description="W", system_prompt="W.")
                ],
                default_model=_make_mock_model(),
            )

        result = await tool.ainvoke(
            {"description": "classify", "subagent_type": "worker"}
        )
        assert json.loads(result) == {"label": "positive", "score": 0.95}

    async def test_returns_last_message_content_when_no_structured_response(
        self,
    ) -> None:
        async def run(_: Any) -> dict[str, Any]:
            return {
                "messages": [
                    AIMessage(content="first"),
                    AIMessage(content="last message"),
                ],
            }

        with patch(
            "langchain_quickjs._swarm_task.create_agent",
            return_value=_make_runnable(run),
        ):
            tool = create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(name="worker", description="W", system_prompt="W.")
                ],
                default_model=_make_mock_model(),
            )

        result = await tool.ainvoke({"description": "work", "subagent_type": "worker"})
        assert result == "last message"

    async def test_returns_task_completed_when_no_messages(self) -> None:
        async def run(_: Any) -> dict[str, Any]:
            return {"messages": []}

        with patch(
            "langchain_quickjs._swarm_task.create_agent",
            return_value=_make_runnable(run),
        ):
            tool = create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(name="worker", description="W", system_prompt="W.")
                ],
                default_model=_make_mock_model(),
            )

        result = await tool.ainvoke({"description": "work", "subagent_type": "worker"})
        assert result == "Task completed"

    async def test_compiles_new_agent_with_response_format_when_schema_provided(
        self,
    ) -> None:
        with patch(
            "langchain_quickjs._swarm_task.create_agent",
            return_value=_make_fake_agent(),
        ) as mock_create:
            tool = create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(
                        name="worker", description="W", system_prompt="Analyze."
                    )
                ],
                default_model=_make_mock_model(),
            )

            assert mock_create.call_count == 1

            schema = {
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
            }

            await tool.ainvoke(
                {
                    "description": "classify",
                    "subagent_type": "worker",
                    "response_schema": schema,
                }
            )

            assert mock_create.call_count == 2
            last_call_kwargs = mock_create.call_args_list[-1]
            assert last_call_kwargs.kwargs["response_format"] == {
                "type": "object",
                "additionalProperties": False,
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
            }

    async def test_does_not_compile_new_agent_when_schema_omitted(self) -> None:
        with patch(
            "langchain_quickjs._swarm_task.create_agent",
            return_value=_make_fake_agent(),
        ) as mock_create:
            tool = create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(name="worker", description="W", system_prompt="W.")
                ],
                default_model=_make_mock_model(),
            )
            assert mock_create.call_count == 1

            await tool.ainvoke({"description": "work", "subagent_type": "worker"})
            assert mock_create.call_count == 1

    async def test_validates_response_schema_must_have_type_object(self) -> None:
        with patch(
            "langchain_quickjs._swarm_task.create_agent",
            return_value=_make_fake_agent(),
        ):
            tool = create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(name="worker", description="W", system_prompt="W.")
                ],
                default_model=_make_mock_model(),
            )

        with pytest.raises(Exception, match='response_schema must have type: "object"'):
            await tool.ainvoke(
                {
                    "description": "work",
                    "subagent_type": "worker",
                    "response_schema": {"type": "array", "items": {"type": "string"}},
                }
            )

    async def test_normalizes_schema_adding_additional_properties_false(self) -> None:
        with patch(
            "langchain_quickjs._swarm_task.create_agent",
            return_value=_make_fake_agent(),
        ) as mock_create:
            tool = create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(name="worker", description="W", system_prompt="W.")
                ],
                default_model=_make_mock_model(),
            )

            await tool.ainvoke(
                {
                    "description": "work",
                    "subagent_type": "worker",
                    "response_schema": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                }
            )

            last_call_kwargs = mock_create.call_args_list[-1]
            assert (
                last_call_kwargs.kwargs["response_format"]["additionalProperties"]
                is False
            )


# ---------------------------------------------------------------------------
# Invoke mode
# ---------------------------------------------------------------------------


class TestInvokeMode:
    """Tests for invoke dispatch mode."""

    async def test_calls_model_directly_with_human_message(self) -> None:
        model = _make_mock_model("classified: positive")

        tool = create_swarm_task_tool(subagents=[], default_model=model)

        await tool.ainvoke({"description": "classify this", "mode": "invoke"})

        model.ainvoke.assert_awaited_once()
        messages = model.ainvoke.call_args[0][0]
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "classify this"

    async def test_does_not_call_create_agent_in_invoke_mode(self) -> None:
        with patch(
            "langchain_quickjs._swarm_task.create_agent",
            return_value=_make_fake_agent(),
        ) as mock_create:
            tool = create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(
                        name="worker",
                        description="W",
                        system_prompt="W.",
                        model=_FakeModel(),
                    ),
                ],
                default_model=_make_mock_model(),
            )
            after_construction = mock_create.call_count

            await tool.ainvoke({"description": "work", "mode": "invoke"})
            assert mock_create.call_count == after_construction

    async def test_uses_bind_tools_when_response_schema_provided(self) -> None:
        structured_result = {"label": "positive"}
        bound_model = AsyncMock()
        bound_model.ainvoke = AsyncMock(
            return_value=AIMessage(
                content=[],
                tool_calls=[
                    {
                        "name": "structured_output",
                        "args": structured_result,
                        "id": "tc_1",
                    }
                ],
            )
        )

        model = MagicMock()
        model.bind_tools = MagicMock(return_value=bound_model)

        tool = create_swarm_task_tool(subagents=[], default_model=model)

        result = await tool.ainvoke(
            {
                "description": "work",
                "mode": "invoke",
                "response_schema": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                },
            }
        )

        model.bind_tools.assert_called_once_with(
            [
                {
                    "name": "structured_output",
                    "description": "Return the structured result.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"label": {"type": "string"}},
                    },
                },
            ],
            tool_choice="structured_output",
        )
        bound_model.ainvoke.assert_awaited_once()
        assert result == json.dumps(structured_result)

    async def test_throws_when_model_does_not_support_bind_tools(self) -> None:
        model = _make_mock_model()
        del model.bind_tools

        tool = create_swarm_task_tool(subagents=[], default_model=model)

        with pytest.raises(Exception, match="bind_tools"):
            await tool.ainvoke(
                {
                    "description": "work",
                    "mode": "invoke",
                    "response_schema": {
                        "type": "object",
                        "properties": {"label": {"type": "string"}},
                    },
                }
            )

    async def test_returns_string_content_from_model_response(self) -> None:
        model = _make_mock_model("the answer")
        tool = create_swarm_task_tool(subagents=[], default_model=model)

        result = await tool.ainvoke({"description": "work", "mode": "invoke"})
        assert result == "the answer"

    async def test_validates_response_schema_must_have_type_object(self) -> None:
        tool = create_swarm_task_tool(subagents=[], default_model=_make_mock_model())

        with pytest.raises(Exception, match='response_schema must have type: "object"'):
            await tool.ainvoke(
                {
                    "description": "work",
                    "mode": "invoke",
                    "response_schema": {"type": "array", "items": {"type": "string"}},
                }
            )

    async def test_works_without_response_schema(self) -> None:
        model = _make_mock_model("plain response")
        tool = create_swarm_task_tool(subagents=[], default_model=model)

        result = await tool.ainvoke({"description": "work", "mode": "invoke"})

        assert result == "plain response"
        model.ainvoke.assert_awaited_once()


# ---------------------------------------------------------------------------
# Mode defaulting
# ---------------------------------------------------------------------------


class TestModeDefaulting:
    """Tests for mode default behavior."""

    async def test_defaults_to_agent_mode_when_mode_not_provided(self) -> None:
        subagent_model = _FakeModel()
        default_model = _make_mock_model()

        with patch(
            "langchain_quickjs._swarm_task.create_agent",
            return_value=_make_fake_agent(),
        ):
            tool = create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(
                        name="worker",
                        description="W",
                        system_prompt="W.",
                        model=subagent_model,
                    )
                ],
                default_model=default_model,
            )

        await tool.ainvoke({"description": "work", "subagent_type": "worker"})

        # default_model.ainvoke should NOT have been called (that's invoke mode)
        default_model.ainvoke.assert_not_called()


# ---------------------------------------------------------------------------
# Multiple subagents
# ---------------------------------------------------------------------------


class TestMultipleSubagents:
    """Tests for multiple subagent configurations."""

    def test_uses_subagent_own_model_when_specified(self) -> None:
        screener_model = _FakeModel()
        default_model = _make_mock_model()

        with patch(
            "langchain_quickjs._swarm_task.create_agent",
            return_value=_make_fake_agent(),
        ) as mock_create:
            create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(
                        name="screener",
                        description="S",
                        system_prompt="Screen.",
                        model=screener_model,
                    ),
                ],
                default_model=default_model,
            )

        mock_create.assert_called_once()
        assert mock_create.call_args.kwargs["model"] is screener_model

    def test_falls_back_to_default_model_when_subagent_has_no_model(self) -> None:
        default_model = _make_mock_model()

        with patch(
            "langchain_quickjs._swarm_task.create_agent",
            return_value=_make_fake_agent(),
        ) as mock_create:
            create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(name="worker", description="W", system_prompt="W.")
                ],
                default_model=default_model,
            )

        mock_create.assert_called_once()
        assert mock_create.call_args.kwargs["model"] is default_model


# ---------------------------------------------------------------------------
# TTL variant cache integration
# ---------------------------------------------------------------------------


class TestTTLVariantCacheIntegration:
    """Tests for TTL-based variant caching in agent mode."""

    async def test_reuses_compiled_agent_on_repeated_calls_with_same_schema(
        self,
    ) -> None:
        with patch(
            "langchain_quickjs._swarm_task.create_agent",
            return_value=_make_fake_agent(),
        ) as mock_create:
            tool = create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(name="worker", description="W", system_prompt="W.")
                ],
                default_model=_make_mock_model(),
            )

            schema = {
                "type": "object",
                "properties": {"label": {"type": "string"}},
            }

            await tool.ainvoke(
                {
                    "description": "row 1",
                    "subagent_type": "worker",
                    "response_schema": schema,
                }
            )
            await tool.ainvoke(
                {
                    "description": "row 2",
                    "subagent_type": "worker",
                    "response_schema": schema,
                }
            )
            await tool.ainvoke(
                {
                    "description": "row 3",
                    "subagent_type": "worker",
                    "response_schema": schema,
                }
            )

            # 1 at construction + 1 for the schema variant (reused for rows 2 and 3)
            assert mock_create.call_count == 2

    async def test_compiles_separate_variants_for_distinct_schemas(self) -> None:
        with patch(
            "langchain_quickjs._swarm_task.create_agent",
            return_value=_make_fake_agent(),
        ) as mock_create:
            tool = create_swarm_task_tool(
                subagents=[
                    SwarmSubAgent(name="worker", description="W", system_prompt="W.")
                ],
                default_model=_make_mock_model(),
            )

            await tool.ainvoke(
                {
                    "description": "work",
                    "subagent_type": "worker",
                    "response_schema": {
                        "type": "object",
                        "properties": {"a": {"type": "string"}},
                    },
                }
            )
            await tool.ainvoke(
                {
                    "description": "work",
                    "subagent_type": "worker",
                    "response_schema": {
                        "type": "object",
                        "properties": {"b": {"type": "number"}},
                    },
                }
            )

            # 1 at construction + 2 schema variants
            assert mock_create.call_count == 3


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestToolMetadata:
    """Tests for the tool's name and schema."""

    def test_tool_name_is_swarm_task(self) -> None:
        tool = create_swarm_task_tool(subagents=[], default_model=_make_mock_model())
        assert tool.name == "swarm_task"

    def test_tool_has_description(self) -> None:
        tool = create_swarm_task_tool(subagents=[], default_model=_make_mock_model())
        assert "swarm subagent" in tool.description.lower()
