"""CLI-specific tests for compact_conversation tool (HITL gating, display).

Core compact tool logic tests live in the SDK at
`libs/deepagents/tests/unit_tests/middleware/test_compact_tool.py`.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents.backends.protocol import FileDownloadResponse, WriteResult
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code.config import MODEL_RETRIES_ATTR
from deepagents_code.offload_middleware import (
    _SUMMARY_TRIM_FALLBACK,
    CLICompactionMiddleware,
    _ArchiveReadGuard,
    _install_lazy_summary_model,
    _require_helper_slot,
    _RetryingModelInvoker,
    _runtime_model_config,
    _summary_trim_limit,
)

_NO_BACKOFF = "deepagents_code.model_retry._retry_delay_seconds"

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from langchain_core.messages import AnyMessage


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


class TestArchiveReadGuard:
    """Cover fail-closed archive writes after backend read errors."""

    def test_sync_error_response_blocks_write(self) -> None:
        """A synchronous error response must not permit a truncating write."""
        response = FileDownloadResponse(
            path="/conversation_history/thread.md",
            error="permission_denied",
        )
        backend = MagicMock()
        backend.download_files.return_value = [response]
        backend.write.return_value = WriteResult(path=response.path)
        guard = _ArchiveReadGuard(backend)

        assert guard.download_files([response.path]) == [response]
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            guard.write(response.path, "new history")

        backend.write.assert_not_called()

    async def test_async_error_response_blocks_write(self) -> None:
        """An asynchronous error response must not permit a truncating write."""
        response = FileDownloadResponse(
            path="/conversation_history/thread.md",
            error="transient backend error",
        )
        backend = MagicMock()
        backend.adownload_files = AsyncMock(return_value=[response])
        backend.awrite = AsyncMock(return_value=WriteResult(path=response.path))
        guard = _ArchiveReadGuard(backend)

        assert await guard.adownload_files([response.path]) == [response]
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            await guard.awrite(response.path, "new history")

        backend.awrite.assert_not_awaited()

    def test_missing_archive_allows_create(self) -> None:
        """A missing archive remains the expected first-write path."""
        response = FileDownloadResponse(
            path="/conversation_history/thread.md",
            error="file_not_found",
        )
        backend = MagicMock()
        backend.download_files.return_value = [response]
        expected = WriteResult(path=response.path)
        backend.write.return_value = expected
        guard = _ArchiveReadGuard(backend)

        assert guard.download_files([response.path]) == [response]
        assert guard.write(response.path, "new history") == expected


class TestCLICompactionMiddleware:
    """Cover dcode's explicit `/offload` behavior layered over the SDK tool."""

    @staticmethod
    def _summarization() -> MagicMock:
        summarization = MagicMock()
        backend = MagicMock()
        backend.adownload_files = AsyncMock(
            return_value=[
                FileDownloadResponse(
                    path="/conversation_history/thread.md",
                    error="file_not_found",
                )
            ]
        )
        summarization._backend = backend
        summarization._get_history_path.return_value = "/conversation_history/thread.md"
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

    async def test_operation_plan_defers_archive_until_checkpoint_reservation(
        self,
    ) -> None:
        """Planning may spend on a summary but cannot mutate archive storage."""
        summarization = self._summarization()
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None

        plan = await middleware._aplan_forced_compaction_update(
            {"messages": [HumanMessage("one"), HumanMessage("two")]}, runtime
        )

        assert plan is not None
        assert plan.update(None)["_summarization_event"]["file_path"] is None
        summarization._aoffload_to_backend.assert_not_awaited()

    async def test_operation_path_returns_an_absolute_cutoff(self) -> None:
        """The committed event must carry the absolute cutoff, not the relative one.

        `_determine_cutoff_index` is relative to the *effective* conversation
        (post-previous-summary), while the persisted `cutoff_index` indexes the
        full message list — `_compute_state_cutoff` converts between them. The
        two coincide on a thread's first `/offload`, so returning the relative
        value passes every other test here and only corrupts the *second*
        `/offload`, which reads this back as its base.
        """
        summarization = self._summarization()
        summarization._determine_cutoff_index.return_value = 2
        summarization._compute_state_cutoff.return_value = 9
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None
        prior = {"cutoff_index": 7, "summary_message": None, "file_path": None}

        result = await middleware.arun_forced_compaction_update(
            cast(
                "Any",
                {
                    "messages": [HumanMessage("one"), HumanMessage("two")],
                    "_summarization_event": prior,
                },
            ),
            runtime,
        )

        assert result is not None
        event = result["_summarization_event"]
        summarization._compute_state_cutoff.assert_called_once_with(prior, 2)
        assert event["cutoff_index"] == 9
        assert event["file_path"] == "/conversation_history/thread.md"
        assert isinstance(event["summary_message"], HumanMessage)

    async def test_operation_path_threads_and_persists_the_session_id(self) -> None:
        """`/offload` must reuse and re-commit the SDK's archive-file id.

        The SDK's `_offload_to_backend` names the archive by `session_id`, and
        the committed `_summarization_session_id` is what makes a later
        compaction append to the same file instead of starting a new one. The
        server operation bypasses the SDK's own state update, so it has to
        thread the id through and write it back itself.
        """
        summarization = self._summarization()
        summarization._get_session_id.return_value = "session_abc"
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None

        result = await middleware.arun_forced_compaction_update(
            {"messages": [HumanMessage("one"), HumanMessage("two")]}, runtime
        )

        assert result is not None
        assert result["_summarization_session_id"] == "session_abc"
        assert summarization._aoffload_to_backend.await_args.args[2] == "session_abc"

    async def test_operation_path_rejects_an_empty_conversation(self) -> None:
        """An empty `messages` must raise rather than report a clean no-op.

        The server operation normally handles an empty thread before invoking
        compaction. A direct caller that bypasses that service check still gets
        an explicit error instead of a misleading successful no-op.
        """
        summarization = self._summarization()
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None

        with pytest.raises(ValueError, match="checkpointed conversation"):
            await middleware.arun_forced_compaction_update(
                cast("Any", {"messages": [], "_summarization_event": None}), runtime
            )

        summarization._acreate_summary.assert_not_awaited()

    async def test_operation_path_logs_a_failed_archive_write(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A `None` archive path must leave a trace naming this call site.

        `_aoffload_to_backend` catches every write failure and returns `None` —
        including `_ArchiveReadGuard`'s deliberate "refusing to overwrite
        existing history" `RuntimeError`. The compaction still commits (the
        client reports the missing archive to the user), but without this the
        only server-side record is a warning inside the SDK that names neither
        the thread nor `/offload`.
        """
        summarization = self._summarization()
        summarization._aoffload_to_backend = AsyncMock(return_value=None)
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None

        with caplog.at_level("ERROR"):
            result = await middleware.arun_forced_compaction_update(
                {"messages": [HumanMessage("one"), HumanMessage("two")]}, runtime
            )

        assert result is not None
        assert result["_summarization_event"]["file_path"] is None
        assert "archive write failed" in caplog.text

    def test_runtime_model_builds_matching_summarizer(self) -> None:
        """A `/model` override selects the summarizer used by `/offload`."""
        startup = self._summarization()
        middleware = CLICompactionMiddleware(startup, cli_max_retries=0)
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
            cli_max_retries=0,
        )
        create_summarization.assert_called_once()
        assert create_summarization.call_args.args[0] is active_model
        # The summarizer gets the composite backend itself, not the
        # `_ArchiveReadGuard` wrapper: it reads `artifacts_root` to prefix the
        # archive path, and the guard exposes no such attribute. The server
        # operation applies the guard separately at the write site.
        assert create_summarization.call_args.args[1] is startup._backend
        assert isinstance(selected._lc_helper._summary_model, _RetryingModelInvoker)

    async def test_runtime_summary_override_replaces_only_summary_invoker(self) -> None:
        startup = self._summarization()
        middleware = CLICompactionMiddleware(startup, cli_max_retries=0)
        runtime = MagicMock()
        runtime.context = {
            "model": "provider:main-model",
            "summarization_model": "provider:summary-model",
        }
        main_model = SimpleNamespace(profile={"max_input_tokens": 100_000})
        summary_model = SimpleNamespace(
            profile={"max_input_tokens": 10_000}, _llm_type="summary"
        )
        selected = MagicMock()
        selected._lc_helper.trim_tokens_to_summarize = None
        selected._lc_helper._acreate_summary = AsyncMock(return_value="summary")

        with (
            patch(
                "deepagents_code.config.create_model",
                side_effect=[
                    SimpleNamespace(model=main_model),
                    SimpleNamespace(model=summary_model),
                ],
            ) as create_model,
            patch(
                "deepagents_code.offload_middleware.create_summarization_middleware",
                return_value=selected,
            ) as create_summarization,
        ):
            actual = middleware._summarization_for_runtime(runtime)
            assert create_model.call_count == 1
            await actual._lc_helper._acreate_summary([])

        assert actual is selected
        assert create_model.call_args_list[0].args == ("provider:main-model",)
        assert create_model.call_args_list[1].args == ("provider:summary-model",)
        assert create_summarization.call_args.args[0] is main_model
        assert selected._lc_helper._summary_model._model is summary_model
        assert selected._lc_helper.trim_tokens_to_summarize == 8_000
        assert middleware._summarization is startup

    def test_summary_override_without_a_main_model_reuses_the_startup_model(
        self,
    ) -> None:
        """The dominant path: `--summarization-model` with no `--model`.

        The per-turn context carries `model=self._model_override`, which is
        `None` for any user who never ran `--model` or `/model`. So this branch
        -- not the both-specs branch -- is what a plain
        `--summarization-model` user hits on every turn.
        """
        startup = self._summarization()
        startup.model = SimpleNamespace(profile={"max_input_tokens": 100_000})
        middleware = CLICompactionMiddleware(startup, cli_max_retries=0)
        runtime = MagicMock()
        runtime.context = {
            "summarization_model": "provider:summary-model",
            # A context limit is present but must not be applied: `model` here
            # is the shared startup instance, so patching its profile would
            # leak into every later turn.
            "model_context_limit": 42_000,
        }
        selected = MagicMock()

        with (
            patch("deepagents_code.config.create_model") as create_model,
            patch(
                "deepagents_code.offload_middleware.create_summarization_middleware",
                return_value=selected,
            ) as create_summarization,
        ):
            actual = middleware._summarization_for_runtime(runtime)

        assert actual is selected
        # Nothing is resolved eagerly: the main model is reused as-is and the
        # summary model is deferred to the first actual compaction.
        create_model.assert_not_called()
        assert create_summarization.call_args.args[0] is startup.model
        assert startup.model.profile == {"max_input_tokens": 100_000}
        assert middleware._summarization is startup

    def test_explicit_clear_overrides_startup_summary_model(self) -> None:
        """A runtime clear selects the main model despite a graph-level default."""
        from deepagents_code._cli_context import INHERIT_SUMMARIZATION_MODEL

        startup = self._summarization()
        middleware = CLICompactionMiddleware(
            startup, summarization_model_spec="provider:startup-summary"
        )
        runtime = MagicMock()
        runtime.context = {"summarization_model": INHERIT_SUMMARIZATION_MODEL}

        with patch(
            "deepagents_code.offload_middleware.create_summarization_middleware"
        ) as create_summarization:
            actual = middleware._summarization_for_runtime(runtime)

        assert actual is startup
        create_summarization.assert_not_called()

    async def test_runtime_summary_override_uses_summary_counter_and_fallback(
        self,
    ) -> None:
        startup = self._summarization()
        middleware = CLICompactionMiddleware(
            startup, summarization_model_spec="provider:summary-model"
        )
        runtime = MagicMock()
        runtime.context = {"model": "provider:main-model"}
        main_model = SimpleNamespace(profile={"max_input_tokens": 100_000})
        summary_model = SimpleNamespace(profile=None)
        selected = MagicMock()
        selected._lc_helper._acreate_summary = AsyncMock(return_value="summary")
        counter = MagicMock()

        with (
            patch(
                "deepagents_code.config.create_model",
                side_effect=[
                    SimpleNamespace(model=main_model),
                    SimpleNamespace(model=summary_model),
                ],
            ) as create_model,
            patch(
                "deepagents_code.offload_middleware.create_summarization_middleware",
                return_value=selected,
            ),
            patch(
                "deepagents_code.offload_middleware._get_approximate_token_counter",
                return_value=counter,
            ) as get_counter,
        ):
            middleware._summarization_for_runtime(runtime)
            assert create_model.call_count == 1
            await selected._lc_helper._acreate_summary([])

        get_counter.assert_called_once_with(summary_model)
        assert selected._lc_helper.token_counter is counter
        assert selected._lc_helper._partial_token_counter.func is counter
        assert selected._lc_helper._partial_token_counter.keywords == {
            "use_usage_metadata_scaling": False
        }
        assert selected._lc_helper.trim_tokens_to_summarize == _SUMMARY_TRIM_FALLBACK

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
            cli_max_retries=None,
        )
        assert active_model.profile["max_input_tokens"] == 24_000
        create_summarization.assert_called_once()
        assert create_summarization.call_args.args[0] is active_model
        assert create_summarization.call_args.args[1] is startup._backend

    async def test_operation_read_failure_never_truncates_archive(self) -> None:
        """A transient archive read failure blocks the server operation's write."""
        from deepagents.middleware.summarization import SummarizationMiddleware

        summarization = self._summarization()
        backend = MagicMock()
        backend.adownload_files = AsyncMock(side_effect=RuntimeError("read failed"))
        backend.awrite = AsyncMock()
        backend.aedit = AsyncMock()
        summarization._backend = backend
        summarization._get_history_path.return_value = "/conversation_history/thread.md"
        summarization._filter_summary_messages.side_effect = lambda messages: messages

        async def sdk_offload(
            guarded: BackendProtocol, messages: list[AnyMessage], session_id: str
        ) -> str | None:
            return await SummarizationMiddleware._aoffload_to_backend(
                summarization, guarded, messages, session_id
            )

        summarization._aoffload_to_backend = AsyncMock(side_effect=sdk_offload)
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None

        result = await middleware.arun_forced_compaction_update(
            {"messages": [HumanMessage("one"), HumanMessage("two")]}, runtime
        )

        backend.awrite.assert_not_awaited()
        backend.aedit.assert_not_awaited()
        assert result is not None
        assert result["_summarization_event"]["file_path"] is None

    def test_factory_builds_cli_middleware_threading_system_prompt(self) -> None:
        """The factory returns a CLI middleware carrying the SDK's config."""
        from deepagents_code import offload_middleware as om

        sdk = MagicMock()
        sdk._summarization = MagicMock()
        sdk._summarization.name = "SummarizationMiddleware"
        sdk.system_prompt = "SYSTEM PROMPT"
        backend: Any = object()
        with patch.object(
            om, "create_summarization_tool_middleware", return_value=sdk
        ) as factory:
            result = om._create_cli_compaction_middleware(
                "provider:model", backend, cli_max_retries=0
            )

        factory.assert_called_once()
        assert isinstance(result, om.CLICompactionMiddleware)
        assert result.name == "SummarizationMiddleware"
        assert result.system_prompt == "SYSTEM PROMPT"
        assert result._summarization is sdk._summarization
        assert result._cli_max_retries == 0
        assert isinstance(
            result._summarization._lc_helper._summary_model,
            _RetryingModelInvoker,
        )


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
            None,
            {"temperature": 0},
            {},
            None,
        )

    def test_serialized_dict(self) -> None:
        ctx = {"model": "p:m2", "model_params": {"x": 1}}
        assert _runtime_model_config(self._runtime(ctx)) == (
            "p:m2",
            None,
            {"x": 1},
            {},
            None,
        )

    def test_dict_with_bad_types_normalizes(self) -> None:
        ctx = {"model": 123, "model_params": None}
        assert _runtime_model_config(self._runtime(ctx)) == (None, None, {}, {}, None)

    def test_unknown_shape(self) -> None:
        assert _runtime_model_config(self._runtime(object())) == (
            None,
            None,
            {},
            {},
            None,
        )

    def test_summarization_model_accepts_schema_and_serialized_context(self) -> None:
        schema = CLIContextSchema(summarization_model="openai:gpt-5.4-mini")
        assert (
            _runtime_model_config(self._runtime(schema)).summarization_model_spec
            == "openai:gpt-5.4-mini"
        )
        assert (
            _runtime_model_config(
                self._runtime({"summarization_model": "anthropic:claude-haiku-4-5"})
            ).summarization_model_spec
            == "anthropic:claude-haiku-4-5"
        )

    def test_named_fields_disambiguate_the_two_dict_slots(self) -> None:
        """The two `dict` slots are addressable by name, not just position.

        `model_params` and `profile_overrides` are structurally identical, so a
        positional swap would be invisible; named-field access pins each to the
        right source value.
        """
        ctx = CLIContextSchema(
            model="p:m",
            model_params={"temperature": 0},
            profile_overrides={"max_input_tokens": 99},
            model_context_limit=7,
        )
        config = _runtime_model_config(self._runtime(ctx))
        assert config.model_params == {"temperature": 0}
        assert config.profile_overrides == {"max_input_tokens": 99}
        assert config.context_limit == 7


class TestSummaryTrimLimit:
    """The bound on history handed to a dedicated summary model.

    dcode leaves SDK trimming off (`trim_tokens_to_summarize=None` skips it
    entirely), so this value is the only thing standing between a deliberately
    smaller summary model and the whole conversation. A `0` would make
    `trim_messages` reject everything; an unbounded value overflows the model
    this feature exists to protect.
    """

    @pytest.mark.parametrize(
        ("profile", "expected"),
        [
            pytest.param({"max_input_tokens": 10_000}, 8_000, id="reserves-a-fifth"),
            pytest.param({"max_input_tokens": 1}, 1, id="floor-never-zero"),
            pytest.param({"max_input_tokens": 4}, 3, id="floor-not-reached"),
            pytest.param(None, _SUMMARY_TRIM_FALLBACK, id="no-profile-dict"),
            pytest.param("not-a-dict", _SUMMARY_TRIM_FALLBACK, id="profile-not-a-dict"),
            pytest.param({}, _SUMMARY_TRIM_FALLBACK, id="limit-absent"),
            pytest.param(
                {"max_input_tokens": "200k"}, _SUMMARY_TRIM_FALLBACK, id="limit-str"
            ),
            pytest.param(
                {"max_input_tokens": 0}, _SUMMARY_TRIM_FALLBACK, id="limit-zero"
            ),
            pytest.param(
                {"max_input_tokens": -5}, _SUMMARY_TRIM_FALLBACK, id="limit-negative"
            ),
            # `bool` is an `int` subclass: without the explicit check,
            # `True * 4 // 5` is `0` and `max(1, 0)` yields a 1-token budget.
            pytest.param(
                {"max_input_tokens": True}, _SUMMARY_TRIM_FALLBACK, id="limit-bool"
            ),
        ],
    )
    def test_budget(self, profile: object, expected: int) -> None:
        model = SimpleNamespace(profile=profile)
        assert _summary_trim_limit(cast("Any", model)) == expected

    def test_missing_profile_attribute_falls_back(self) -> None:
        """A provider exposing no `profile` at all must not raise."""

        class _NoProfile:
            pass

        assert _summary_trim_limit(cast("Any", _NoProfile())) == _SUMMARY_TRIM_FALLBACK


class TestSummarySlotGuards:
    """Every write into `_lc_helper` must fail loudly if the slot is renamed.

    Each of these assignments replaces SDK behavior. Writing to a slot nothing
    reads would leave the SDK's own behavior in place -- a silent no-op that
    reports nothing, which is exactly the failure these guards convert into an
    exception.
    """

    def test_require_helper_slot_names_the_slot_and_the_loss(self) -> None:
        with pytest.raises(AttributeError, match=r"'_gone'.*cannot do the thing"):
            _require_helper_slot(SimpleNamespace(), "_gone", "do the thing")

    def test_require_helper_slot_accepts_a_present_slot(self) -> None:
        _require_helper_slot(SimpleNamespace(_present=None), "_present", "unused")


class TestLazySummaryModel:
    """Deferred construction of the dedicated summary model.

    Resolution is deferred so `create_model`'s provider imports stay off the
    startup path, which means a bad spec is first discovered mid-compaction --
    the worst possible moment to fail. So it degrades instead.
    """

    @staticmethod
    def _summarizer() -> SimpleNamespace:
        helper = SimpleNamespace(
            _create_summary=lambda _messages: "main-model-summary",
            _acreate_summary=AsyncMock(return_value="main-model-summary"),
            _summary_model=None,
            trim_tokens_to_summarize=None,
            token_counter=None,
            _partial_token_counter=None,
        )
        return SimpleNamespace(
            _lc_helper=helper,
            model=SimpleNamespace(profile={"max_input_tokens": 100_000}),
        )

    def test_model_is_not_built_until_a_summary_is_requested(self) -> None:
        summarization = self._summarizer()
        with patch("deepagents_code.config.create_model") as create_model:
            _install_lazy_summary_model(cast("Any", summarization), "p:summary", 0)
            create_model.assert_not_called()

    def test_first_summary_installs_the_override(self) -> None:
        summarization = self._summarizer()
        summary_model = SimpleNamespace(
            profile={"max_input_tokens": 10_000}, _llm_type="summary"
        )
        _install_lazy_summary_model(cast("Any", summarization), "p:summary", 0)

        with patch(
            "deepagents_code.config.create_model",
            return_value=SimpleNamespace(model=summary_model),
        ) as create_model:
            assert summarization._lc_helper._create_summary([]) == "main-model-summary"

        create_model.assert_called_once_with("p:summary", cli_max_retries=0)
        assert summarization._lc_helper._summary_model._model is summary_model
        assert summarization._lc_helper.trim_tokens_to_summarize == 8_000

    def test_unresolvable_spec_degrades_to_the_main_model(self) -> None:
        """A broken summary model is a broken optimization, not a broken turn.

        Raising here would fail the turn that triggered compaction and would
        also break `/compact` -- the tool the user reaches for precisely when
        the context is full.
        """
        summarization = self._summarizer()
        _install_lazy_summary_model(cast("Any", summarization), "p:bad", None)

        with patch(
            "deepagents_code.config.create_model",
            side_effect=RuntimeError("no such provider"),
        ):
            assert summarization._lc_helper._create_summary([]) == "main-model-summary"

        assert isinstance(
            summarization._lc_helper._summary_model, _RetryingModelInvoker
        )
        assert summarization._lc_helper._summary_model._model is summarization.model
        assert summarization._lc_helper.trim_tokens_to_summarize is None

    def test_failure_warns_once_then_retries_the_override(self) -> None:
        summarization = self._summarizer()
        _install_lazy_summary_model(cast("Any", summarization), "p:flaky", None)
        summary_model = SimpleNamespace(
            profile={"max_input_tokens": 10_000}, _llm_type="summary"
        )

        with (
            patch(
                "deepagents_code.config.create_model",
                side_effect=[
                    RuntimeError("transient"),
                    RuntimeError("transient"),
                    SimpleNamespace(model=summary_model),
                ],
            ),
            patch("deepagents_code.offload_middleware.logger") as log,
        ):
            summarization._lc_helper._create_summary([])
            summarization._lc_helper._create_summary([])
            # Compaction is rare, but a permanently bad spec must not log on
            # every attempt.
            assert log.warning.call_count == 1
            summarization._lc_helper._create_summary([])

        assert summarization._lc_helper._summary_model._model is summary_model

    def test_blocking_error_is_not_swallowed(self) -> None:
        """`BlockingError` means the caller ran this on the server event loop.

        That is a defect in the caller, not a bad model spec, so it must not be
        absorbed by the degrade-to-main-model path.
        """

        class BlockingError(Exception):
            pass

        summarization = self._summarizer()
        _install_lazy_summary_model(cast("Any", summarization), "p:summary", None)

        with (
            patch(
                "deepagents_code.config.create_model",
                side_effect=BlockingError("blocking call in event loop"),
            ),
            pytest.raises(BlockingError),
        ):
            summarization._lc_helper._create_summary([])

    async def test_async_summary_installs_the_override(self) -> None:
        summarization = self._summarizer()
        summary_model = SimpleNamespace(
            profile={"max_input_tokens": 10_000}, _llm_type="summary"
        )
        _install_lazy_summary_model(cast("Any", summarization), "p:summary", None)

        with patch(
            "deepagents_code.config.create_model",
            return_value=SimpleNamespace(model=summary_model),
        ):
            assert (
                await summarization._lc_helper._acreate_summary([])
                == "main-model-summary"
            )

        assert summarization._lc_helper._summary_model._model is summary_model

    @staticmethod
    def _dispatching_summarizer() -> SimpleNamespace:
        """A summarizer whose hooks invoke the installed `_summary_model`.

        Mirrors the SDK helper, which reads `_summary_model` at call time, so
        the lazy wrapper's captured bound method follows a model swap.
        """
        helper = SimpleNamespace(
            _summary_model=None,
            trim_tokens_to_summarize=None,
            token_counter="main-counter",
            _partial_token_counter="main-partial-counter",
        )
        helper._create_summary = lambda messages: helper._summary_model.invoke(messages)

        async def _acreate(messages: object) -> object:
            return await helper._summary_model.ainvoke(messages)

        helper._acreate_summary = _acreate
        return SimpleNamespace(
            _lc_helper=helper,
            model=SimpleNamespace(profile={"max_input_tokens": 100_000}),
        )

    @staticmethod
    def _broken_summary_model() -> SimpleNamespace:
        """A dedicated model that builds but raises on every invocation."""

        def _fail(_input: object, **_kwargs: object) -> object:
            msg = "provider has no such model"
            raise RuntimeError(msg)

        async def _afail(_input: object, **_kwargs: object) -> object:
            await asyncio.sleep(0)
            msg = "provider has no such model"
            raise RuntimeError(msg)

        return SimpleNamespace(
            profile={"max_input_tokens": 10_000},
            _llm_type="summary",
            invoke=_fail,
            ainvoke=_afail,
        )

    def test_invocation_failure_degrades_to_the_main_model(self) -> None:
        """A dedicated model that builds but cannot generate must not wedge compaction.

        `create_model` succeeding only proves the spec parses; an unknown model
        ID or missing provider access surfaces on the first summary call, and
        raising there would abort compaction on every later turn once the
        context threshold is reached.
        """
        summarization = self._dispatching_summarizer()
        summarization.model.invoke = lambda _input, **_kwargs: "main-model-summary"
        _install_lazy_summary_model(cast("Any", summarization), "p:broken", None)

        with patch(
            "deepagents_code.config.create_model",
            return_value=SimpleNamespace(model=self._broken_summary_model()),
        ):
            assert summarization._lc_helper._create_summary([]) == "main-model-summary"

        helper = summarization._lc_helper
        assert helper._summary_model._model is summarization.model
        assert helper.token_counter == "main-counter"
        assert helper._partial_token_counter == "main-partial-counter"
        assert helper.trim_tokens_to_summarize is None

    def test_concurrent_invocation_failures_both_fall_back(self) -> None:
        """Concurrent failures must not observe another call's reset state."""
        summarization = self._dispatching_summarizer()
        summarization.model.invoke = lambda _input, **_kwargs: "main-model-summary"
        _install_lazy_summary_model(cast("Any", summarization), "p:broken", None)
        first_started = Event()
        two_started = Event()
        allow_failure = Event()
        count_lock = Lock()
        calls = 0

        def _fail(_input: object, **_kwargs: object) -> object:
            nonlocal calls
            with count_lock:
                calls += 1
                first_started.set()
                if calls == 2:
                    two_started.set()
            assert allow_failure.wait(timeout=5)
            msg = "provider has no such model"
            raise RuntimeError(msg)

        model = SimpleNamespace(
            profile={"max_input_tokens": 10_000},
            _llm_type="summary",
            invoke=_fail,
        )
        with (
            patch(
                "deepagents_code.config.create_model",
                return_value=SimpleNamespace(model=model),
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            first = executor.submit(summarization._lc_helper._create_summary, [])
            assert first_started.wait(timeout=5)
            second = executor.submit(summarization._lc_helper._create_summary, [])
            assert not two_started.wait(timeout=0.1)
            allow_failure.set()

            assert first.result(timeout=5) == "main-model-summary"
            assert second.result(timeout=5) == "main-model-summary"

    def test_invocation_failure_retries_the_override_on_the_next_summary(
        self,
    ) -> None:
        """A transient invocation failure must not disable the override for good.

        Matches the build-failure path: after degrading, the next summary
        rebuilds and retries the dedicated model.
        """
        summarization = self._dispatching_summarizer()
        summarization.model.invoke = lambda _input, **_kwargs: "main-model-summary"
        _install_lazy_summary_model(cast("Any", summarization), "p:flaky", None)

        calls = 0

        def _flaky(_input: object, **_kwargs: object) -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                msg = "provider overloaded"
                raise RuntimeError(msg)
            return "dedicated-summary"

        model = SimpleNamespace(
            profile={"max_input_tokens": 10_000},
            _llm_type="summary",
            invoke=_flaky,
        )
        with patch(
            "deepagents_code.config.create_model",
            return_value=SimpleNamespace(model=model),
        ):
            assert summarization._lc_helper._create_summary([]) == "main-model-summary"
            assert summarization._lc_helper._create_summary([]) == "dedicated-summary"

        assert summarization._lc_helper._summary_model._model is model

    def test_invocation_failure_warns_once(self) -> None:
        summarization = self._dispatching_summarizer()
        summarization.model.invoke = lambda _input, **_kwargs: "main-model-summary"
        _install_lazy_summary_model(cast("Any", summarization), "p:broken", None)

        with (
            patch(
                "deepagents_code.config.create_model",
                return_value=SimpleNamespace(model=self._broken_summary_model()),
            ),
            patch("deepagents_code.offload_middleware.logger") as log,
        ):
            summarization._lc_helper._create_summary([])
            summarization._lc_helper._create_summary([])
            assert log.warning.call_count == 1

    def test_invocation_blocking_error_is_not_swallowed(self) -> None:
        """`BlockingError` on the summary call is a caller defect, re-raised."""

        class BlockingError(Exception):
            pass

        def _fail(_input: object, **_kwargs: object) -> object:
            msg = "blocking call in event loop"
            raise BlockingError(msg)

        summarization = self._dispatching_summarizer()
        _install_lazy_summary_model(cast("Any", summarization), "p:summary", None)
        model = SimpleNamespace(
            profile={"max_input_tokens": 10_000},
            _llm_type="summary",
            invoke=_fail,
        )
        with (
            patch(
                "deepagents_code.config.create_model",
                return_value=SimpleNamespace(model=model),
            ),
            pytest.raises(BlockingError),
        ):
            summarization._lc_helper._create_summary([])

    def test_main_model_failure_is_not_masked_by_the_fallback(self) -> None:
        """A main-model summary failure propagates instead of degrading.

        The fallback exists for a broken dedicated model; when the override was
        never installed there is nothing to fall back to, and restoring the
        captured placeholders would clobber the helper's real token counters.
        """
        summarization = self._dispatching_summarizer()

        def _fail(_input: object, **_kwargs: object) -> object:
            msg = "main model down"
            raise RuntimeError(msg)

        summarization.model.invoke = _fail
        _install_lazy_summary_model(cast("Any", summarization), "p:bad", None)

        with (
            patch(
                "deepagents_code.config.create_model",
                side_effect=RuntimeError("no such provider"),
            ),
            pytest.raises(RuntimeError, match="main model down"),
        ):
            summarization._lc_helper._create_summary([])

        helper = summarization._lc_helper
        assert helper.token_counter == "main-counter"
        assert helper._partial_token_counter == "main-partial-counter"

    async def test_async_invocation_failure_degrades_to_the_main_model(self) -> None:
        summarization = self._dispatching_summarizer()

        async def _main_ainvoke(_input: object, **_kwargs: object) -> str:
            await asyncio.sleep(0)
            return "main-model-summary"

        summarization.model.ainvoke = _main_ainvoke
        _install_lazy_summary_model(cast("Any", summarization), "p:broken", None)

        with patch(
            "deepagents_code.config.create_model",
            return_value=SimpleNamespace(model=self._broken_summary_model()),
        ):
            assert (
                await summarization._lc_helper._acreate_summary([])
                == "main-model-summary"
            )

        assert summarization._lc_helper._summary_model._model is summarization.model

    async def test_concurrent_async_invocation_failures_both_fall_back(self) -> None:
        """Async failures serialize the shared configure-and-fallback state."""
        summarization = self._dispatching_summarizer()

        async def _main_ainvoke(_input: object, **_kwargs: object) -> str:
            await asyncio.sleep(0)
            return "main-model-summary"

        summarization.model.ainvoke = _main_ainvoke
        _install_lazy_summary_model(cast("Any", summarization), "p:broken", None)
        first_started = asyncio.Event()
        two_started = asyncio.Event()
        allow_failure = asyncio.Event()
        calls = 0

        async def _fail(_input: object, **_kwargs: object) -> object:
            nonlocal calls
            calls += 1
            first_started.set()
            if calls == 2:
                two_started.set()
            await asyncio.wait_for(allow_failure.wait(), timeout=5)
            msg = "provider has no such model"
            raise RuntimeError(msg)

        model = SimpleNamespace(
            profile={"max_input_tokens": 10_000},
            _llm_type="summary",
            ainvoke=_fail,
        )
        with patch(
            "deepagents_code.config.create_model",
            return_value=SimpleNamespace(model=model),
        ):
            first = asyncio.create_task(summarization._lc_helper._acreate_summary([]))
            await asyncio.wait_for(first_started.wait(), timeout=5)
            second = asyncio.create_task(summarization._lc_helper._acreate_summary([]))
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(two_started.wait(), timeout=0.1)
            allow_failure.set()

            assert await asyncio.wait_for(first, timeout=5) == "main-model-summary"
            assert await asyncio.wait_for(second, timeout=5) == "main-model-summary"


class TestRetryingModelInvoker:
    """The wrapper that replaces LangChain's summary-model retries.

    It stands in for `with_retry()`, so a wrapper that does not actually retry
    silently drops compaction summarization to a single attempt.
    """

    @staticmethod
    def _model(calls: list[str]) -> SimpleNamespace:
        """Build a model that fails once, then succeeds.

        Returns:
            A stub chat model carrying the default dcode retry budget.
        """
        transient = TimeoutError("provider unavailable")

        def invoke(_input: object, **_kwargs: object) -> AIMessage:
            calls.append("sync")
            if len(calls) == 1:
                raise transient
            return AIMessage(content="summary")

        async def ainvoke(_input: object, **_kwargs: object) -> AIMessage:
            await asyncio.sleep(0)
            calls.append("async")
            if len(calls) == 1:
                raise transient
            return AIMessage(content="summary")

        model = SimpleNamespace(invoke=invoke, ainvoke=ainvoke)
        setattr(model, MODEL_RETRIES_ATTR, 2)
        return model
