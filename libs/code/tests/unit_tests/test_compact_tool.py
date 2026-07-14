"""CLI-specific tests for compact_conversation tool (HITL gating, display).

Core compact tool logic tests live in the SDK at
`libs/deepagents/tests/unit_tests/middleware/test_compact_tool.py`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code.offload_middleware import (
    COMPACTION_FAILURE_PREFIX,
    CLICompactionMiddleware,
    _runtime_model_config,
)
from deepagents_code.tool_display import format_tool_display


class TestHITLGating:
    """Test that compact_conversation HITL gating respects the constant."""

    def test_hitl_gating_when_enabled(self) -> None:
        """With REQUIRE_COMPACT_TOOL_APPROVAL=True, tool should be gated."""
        with patch("deepagents_code.agent.REQUIRE_COMPACT_TOOL_APPROVAL", True):
            from deepagents_code.agent import _add_interrupt_on

            result = _add_interrupt_on()
            assert "compact_conversation" in result

    def test_hitl_gating_when_disabled(self) -> None:
        """With REQUIRE_COMPACT_TOOL_APPROVAL=False, tool should NOT be gated."""
        with patch("deepagents_code.agent.REQUIRE_COMPACT_TOOL_APPROVAL", False):
            from deepagents_code.agent import _add_interrupt_on

            result = _add_interrupt_on()
            assert "compact_conversation" not in result


class TestDisplayFormatting:
    """Test tool display formatting for compact_conversation."""

    def test_display_formatting(self) -> None:
        """format_tool_display should return the expected string."""
        result = format_tool_display("compact_conversation", {})
        assert "compact_conversation()" in result


class TestCLICompactionMiddleware:
    """Cover dcode's explicit `/offload` behavior layered over the SDK tool."""

    @staticmethod
    def _summarization() -> MagicMock:
        summarization = MagicMock()
        summarization._backend = object()
        summarization._apply_event_to_messages.side_effect = lambda messages, _event: (
            messages
        )
        summarization._determine_cutoff_index.return_value = 2
        summarization._partition_messages.side_effect = lambda messages, cutoff: (
            messages[:cutoff],
            messages[cutoff:],
        )
        summarization._acreate_summary = AsyncMock(return_value="Summary")
        summarization._aoffload_to_backend = AsyncMock(
            return_value="/conversation_history/thread.md"
        )
        summarization._build_new_messages_with_path.return_value = [
            HumanMessage(content="Summary")
        ]
        summarization._compute_state_cutoff.return_value = 2
        return summarization

    async def test_force_bypasses_sdk_eligibility_gate(self) -> None:
        """Forced compaction partitions directly even below the proactive gate."""
        summarization = self._summarization()
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None
        runtime.state = {"messages": [HumanMessage("one"), HumanMessage("two")]}
        runtime.tool_call_id = "tool-call"

        result = await middleware._arun_forced_compact(runtime)

        summarization._is_eligible_for_compaction.assert_not_called()
        summarization._acreate_summary.assert_awaited_once()
        assert result.update is not None
        assert result.update["_summarization_event"]["cutoff_index"] == 2

    def test_runtime_model_builds_matching_summarizer(self) -> None:
        """A `/model` override selects the summarizer used by `/offload`."""
        startup = self._summarization()
        middleware = CLICompactionMiddleware(startup)
        runtime = MagicMock()
        runtime.context = {
            "model": "provider:active-model",
            "model_params": {"temperature": 0},
        }
        active_model = object()
        result = SimpleNamespace(model=active_model)
        selected = MagicMock()

        with (
            patch(
                "deepagents_code.config.create_model", return_value=result
            ) as create_model,
            patch(
                "deepagents_code.offload_middleware.create_summarization_middleware",
                return_value=selected,
            ) as create_summarization,
        ):
            actual = middleware._summarization_for_runtime(runtime)

        assert actual is selected
        create_model.assert_called_once_with(
            "provider:active-model",
            extra_kwargs={"temperature": 0},
            profile_overrides=None,
        )
        create_summarization.assert_called_once_with(active_model, startup._backend)

    def test_runtime_profile_overrides_and_context_limit_are_applied(self) -> None:
        """Server-side offload uses the CLI's effective model profile."""
        startup = self._summarization()
        middleware = CLICompactionMiddleware(startup)
        runtime = MagicMock()
        runtime.context = {
            "model": "provider:active-model",
            "model_params": {},
            "profile_overrides": {"max_input_tokens": 32_000},
            "model_context_limit": 24_000,
        }
        active_model = SimpleNamespace(profile={"max_input_tokens": 200_000})
        result = SimpleNamespace(model=active_model)
        selected = MagicMock()

        with (
            patch(
                "deepagents_code.config.create_model", return_value=result
            ) as create_model,
            patch(
                "deepagents_code.offload_middleware.create_summarization_middleware",
                return_value=selected,
            ) as create_summarization,
        ):
            actual = middleware._summarization_for_runtime(runtime)

        assert actual is selected
        create_model.assert_called_once_with(
            "provider:active-model",
            extra_kwargs=None,
            profile_overrides={"max_input_tokens": 32_000},
        )
        assert active_model.profile["max_input_tokens"] == 24_000
        create_summarization.assert_called_once_with(active_model, startup._backend)

    async def test_force_noops_when_nothing_old_enough(self) -> None:
        """Forced compaction still no-ops at cutoff 0 (bypasses only the gate)."""
        summarization = self._summarization()
        summarization._determine_cutoff_index.return_value = 0
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None
        runtime.state = {"messages": [HumanMessage("one")]}
        runtime.tool_call_id = "tool-call"

        result = await middleware._arun_forced_compact(runtime)

        assert result.update is not None
        assert "_summarization_event" not in result.update
        summarization._acreate_summary.assert_not_awaited()
        assert "Nothing to compact" in result.update["messages"][0].content

    async def test_forced_compact_error_when_summary_fails(self) -> None:
        """A summary failure returns the failure prefix and does not compact."""
        summarization = self._summarization()
        summarization._acreate_summary = AsyncMock(side_effect=RuntimeError("boom"))
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None
        runtime.state = {"messages": [HumanMessage("one"), HumanMessage("two")]}
        runtime.tool_call_id = "tool-call"

        result = await middleware._arun_forced_compact(runtime)

        # The failure must NOT persist an event, and must carry the stable
        # prefix the `/offload` client keys on.
        assert result.update is not None
        assert "_summarization_event" not in result.update
        content = result.update["messages"][0].content
        assert content.startswith(COMPACTION_FAILURE_PREFIX)
        assert "RuntimeError" in content

    def test_sync_forced_compact_compacts(self) -> None:
        """The synchronous forced path mirrors the async one."""
        summarization = self._summarization()
        summarization._create_summary.return_value = "Summary"
        summarization._offload_to_backend.return_value = (
            "/conversation_history/thread.md"
        )
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None
        runtime.state = {"messages": [HumanMessage("one"), HumanMessage("two")]}
        runtime.tool_call_id = "tool-call"

        result = middleware._run_forced_compact(runtime)

        summarization._create_summary.assert_called_once()
        assert result.update is not None
        assert result.update["_summarization_event"]["cutoff_index"] == 2

    def test_force_is_hidden_from_model_schema(self) -> None:
        """`force` must not appear in the schema the model sees."""
        middleware = CLICompactionMiddleware(self._summarization())
        tool = middleware.tools[0]
        # `tool_call_schema` is a pydantic model (or, rarely, a dict); either
        # way the model-facing property set must not expose `force`.
        schema: Any = tool.tool_call_schema
        props = (
            schema.get("properties", {})
            if isinstance(schema, dict)
            else schema.model_json_schema().get("properties", {})
        )
        assert "force" not in props

    def test_force_false_delegates_to_gated_path(self) -> None:
        """`force=False` — the model's only reachable path — uses the SDK gate."""
        middleware = CLICompactionMiddleware(self._summarization())
        tool: Any = middleware.tools[0]
        runtime = MagicMock()
        with (
            patch.object(middleware, "_run_compact", return_value="gated") as gated,
            patch.object(
                middleware, "_run_forced_compact", return_value="forced"
            ) as forced,
        ):
            assert tool.func(runtime, force=False) == "gated"
            gated.assert_called_once_with(runtime)
            forced.assert_not_called()
            assert tool.func(runtime, force=True) == "forced"

    async def test_force_false_delegates_to_gated_path_async(self) -> None:
        """The async tool likewise routes `force=False` to the gated path."""
        middleware = CLICompactionMiddleware(self._summarization())
        tool: Any = middleware.tools[0]
        runtime = MagicMock()
        with (
            patch.object(
                middleware,
                "_arun_compact",
                new_callable=AsyncMock,
                return_value="gated",
            ) as gated,
            patch.object(
                middleware,
                "_arun_forced_compact",
                new_callable=AsyncMock,
                return_value="forced",
            ) as forced,
        ):
            assert await tool.coroutine(runtime, force=False) == "gated"
            gated.assert_awaited_once_with(runtime)
            forced.assert_not_awaited()
            assert await tool.coroutine(runtime, force=True) == "forced"

    def test_sync_forced_compact_noops_when_nothing_old_enough(self) -> None:
        """The sync forced path also no-ops at cutoff 0 (mirrors the async one)."""
        summarization = self._summarization()
        summarization._determine_cutoff_index.return_value = 0
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None
        runtime.state = {"messages": [HumanMessage("one")]}
        runtime.tool_call_id = "tool-call"

        result = middleware._run_forced_compact(runtime)

        assert result.update is not None
        assert "_summarization_event" not in result.update
        summarization._create_summary.assert_not_called()
        assert "Nothing to compact" in result.update["messages"][0].content

    def test_sync_forced_compact_error_when_summary_fails(self) -> None:
        """A sync summary failure returns the failure prefix and does not compact."""
        summarization = self._summarization()
        summarization._create_summary = MagicMock(side_effect=RuntimeError("boom"))
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None
        runtime.state = {"messages": [HumanMessage("one"), HumanMessage("two")]}
        runtime.tool_call_id = "tool-call"

        result = middleware._run_forced_compact(runtime)

        assert result.update is not None
        assert "_summarization_event" not in result.update
        content = result.update["messages"][0].content
        assert content.startswith(COMPACTION_FAILURE_PREFIX)
        assert "RuntimeError" in content

    def test_forced_compact_error_starts_with_prefix(self) -> None:
        """The prefix position is the load-bearing failure-detection contract."""
        command = CLICompactionMiddleware._forced_compact_error(
            "call-1", RuntimeError("boom")
        )
        assert command.update is not None
        (message,) = command.update["messages"]
        assert message.content.startswith(COMPACTION_FAILURE_PREFIX)
        assert message.tool_call_id == "call-1"
        assert "RuntimeError" in message.content

    def test_factory_builds_cli_middleware_threading_system_prompt(self) -> None:
        """The factory returns a CLI middleware carrying the SDK's config."""
        from deepagents_code import offload_middleware as om

        sdk = MagicMock()
        sdk._summarization = MagicMock()
        sdk.system_prompt = "SYSTEM PROMPT"
        backend: Any = object()
        with patch.object(
            om, "create_summarization_tool_middleware", return_value=sdk
        ) as factory:
            result = om._create_cli_compaction_middleware("provider:model", backend)

        factory.assert_called_once()
        assert isinstance(result, om.CLICompactionMiddleware)
        assert result.system_prompt == "SYSTEM PROMPT"
        assert result._summarization is sdk._summarization


class TestRuntimeModelConfig:
    """Cover the three context shapes `_runtime_model_config` accepts."""

    @staticmethod
    def _runtime(context: object) -> MagicMock:
        runtime = MagicMock()
        runtime.context = context
        return runtime

    def test_schema_instance(self) -> None:
        ctx = CLIContextSchema(model="p:m", model_params={"temperature": 0})
        assert _runtime_model_config(self._runtime(ctx)) == (
            "p:m",
            {"temperature": 0},
            {},
            None,
        )

    def test_serialized_dict(self) -> None:
        ctx = {"model": "p:m2", "model_params": {"x": 1}}
        assert _runtime_model_config(self._runtime(ctx)) == (
            "p:m2",
            {"x": 1},
            {},
            None,
        )

    def test_dict_with_bad_types_normalizes(self) -> None:
        ctx = {"model": 123, "model_params": None}
        assert _runtime_model_config(self._runtime(ctx)) == (None, {}, {}, None)

    def test_unknown_shape(self) -> None:
        assert _runtime_model_config(self._runtime(object())) == (None, {}, {}, None)
