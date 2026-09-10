"""Tests for _session_stats module."""

from __future__ import annotations

import logging
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from deepagents_code._session_stats import (
    ModelStats,
    RecordedRequest,
    SessionStats,
    UsageLedgerKey,
    finalize_recorded_requests,
    record_message_usage,
    record_model_usage_event,
    usage_table_enabled,
)
from deepagents_code.cost_tracking import MODEL_USAGE_EVENT_VERSION


class TestFormatCost:
    """Tests for compact USD formatting."""


class TestFormatCostEstimate:
    """Tests for the rounded, approximate cost formatting used for estimates."""


class TestFormatTokenCount:
    """Tests for format_token_count()."""


class TestModelStats:
    """Tests for ModelStats dataclass."""


class TestSessionStats:
    """Tests for SessionStats accumulation logic."""


class TestRecordMessageUsage:
    """Client-side accounting for usage arriving on the message stream."""

    @staticmethod
    def _chunk(
        input_tokens: int,
        output_tokens: int,
        *,
        message_id: str | None = "run-1",
        names_model: bool = True,
    ) -> AIMessageChunk:
        return AIMessageChunk(
            content="",
            id=message_id,
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            response_metadata=(
                {"model_name": "gpt-5.5", "model_provider": "openai"}
                if names_model
                else {"model_provider": "openai"}
            ),
        )

    def test_replayed_chunk_after_a_round_boundary_is_not_recounted(self) -> None:
        """A HITL resume replays chunks; closing the round makes them idempotent.

        Without the boundary the replayed chunk looks like a legitimate later
        delta and merges again, doubling the request's tokens and cost.
        """
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}
        record_message_usage(stats, self._chunk(1_000, 100), recorded_requests=ledger)

        finalize_recorded_requests(ledger)
        replayed = record_message_usage(
            stats, self._chunk(1_000, 100), recorded_requests=ledger
        )

        assert replayed is None
        assert stats.request_count == 1
        assert stats.input_tokens == 1_000
        assert stats.output_tokens == 100

    def test_completed_message_replay_is_recorded_once(self) -> None:
        """A resumed stream replays a completed message; it must not re-count."""
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}
        message = AIMessage(
            content="done",
            id="run-1",
            usage_metadata={
                "input_tokens": 1_000,
                "output_tokens": 100,
                "total_tokens": 1_100,
            },
        )

        first = record_message_usage(stats, message, recorded_requests=ledger)
        second = record_message_usage(stats, message, recorded_requests=ledger)

        assert first is not None
        assert second is None
        assert stats.request_count == 1
        assert stats.output_tokens == 100

    def test_completion_replaces_a_partial_chunk_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A completed message corrects a request built from partial chunks.

        The chunks priced the request under the fallback model; the completion
        names the real one and carries the whole usage. The partial record must
        be replaced, not added to, and the reported delta must be signed so the
        provisional display can follow the correction down.
        """
        monkeypatch.setattr(
            "deepagents_code.cost_tracking.estimate_cost",
            lambda _usage, model, _provider="": (
                0.5 if model == "fallback-model" else 0.05
            ),
        )
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}
        chunk = AIMessageChunk(
            content="",
            id="child-1",
            usage_metadata={
                "input_tokens": 900,
                "output_tokens": 10,
                "total_tokens": 910,
            },
            response_metadata={"model_provider": "openai"},
        )
        completion = AIMessage(
            content="done",
            id="child-1",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 5,
                "total_tokens": 105,
                "input_token_details": {"cache_read": 80},
            },
            response_metadata={"model_name": "real-model", "model_provider": "openai"},
        )

        chunk_usage = record_message_usage(
            stats,
            chunk,
            fallback_model="fallback-model",
            fallback_provider="openai",
            recorded_requests=ledger,
        )
        completion_usage = record_message_usage(
            stats,
            completion,
            fallback_model="fallback-model",
            fallback_provider="openai",
            recorded_requests=ledger,
        )

        assert chunk_usage is not None
        assert chunk_usage.cost_usd == pytest.approx(0.5)
        assert completion_usage is not None
        assert completion_usage.cost_usd == pytest.approx(-0.45)
        assert completion_usage.request_tokens == 105
        assert completion_usage.request_id == "child-1"
        assert stats.request_count == 1
        assert stats.total_cost_usd == pytest.approx(0.05)
        assert stats.input_tokens == 100
        assert stats.output_tokens == 5
        assert stats.cache_read_tokens == 80
        assert stats.per_model["openai", "real-model"].request_count == 1
        assert stats.per_kind["assistant"].request_count == 1
        assert ledger["child-1"].finalized is True

    def test_replaying_a_completion_after_partial_chunks_is_a_no_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The completed event arrives twice; the second must not re-count.

        Once from the message stream and once as a nested event.
        """
        monkeypatch.setattr(
            "deepagents_code.cost_tracking.estimate_cost",
            lambda _usage, _model, _provider="": 0.05,
        )
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}
        chunk = AIMessageChunk(
            content="",
            id="child-1",
            usage_metadata={
                "input_tokens": 900,
                "output_tokens": 10,
                "total_tokens": 910,
            },
            response_metadata={"model_provider": "openai"},
        )
        completion = AIMessage(
            content="done",
            id="child-1",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 5,
                "total_tokens": 105,
            },
            response_metadata={"model_name": "real-model", "model_provider": "openai"},
        )
        record_message_usage(stats, chunk, recorded_requests=ledger)
        first = record_message_usage(stats, completion, recorded_requests=ledger)
        replay = record_message_usage(stats, completion, recorded_requests=ledger)

        assert first is not None
        assert replay is None
        assert stats.request_count == 1
        assert stats.input_tokens == 100
        assert stats.total_cost_usd == pytest.approx(0.05)

    def test_chunks_after_a_completion_cannot_reopen_the_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A finalization closes the ledger entry against later chunk deltas."""
        monkeypatch.setattr(
            "deepagents_code.cost_tracking.estimate_cost",
            lambda _usage, _model, _provider="": 0.05,
        )
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}
        completion = AIMessage(
            content="done",
            id="child-1",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 5,
                "total_tokens": 105,
            },
            response_metadata={"model_name": "real-model", "model_provider": "openai"},
        )
        record_message_usage(stats, completion, recorded_requests=ledger)
        stray = record_message_usage(
            stats,
            AIMessageChunk(
                content="",
                id="child-1",
                usage_metadata={
                    "input_tokens": 50,
                    "output_tokens": 2,
                    "total_tokens": 52,
                },
                response_metadata={"model_provider": "openai"},
            ),
            recorded_requests=ledger,
        )

        assert stray is None
        assert stats.request_count == 1
        assert stats.input_tokens == 100
        assert stats.total_cost_usd == pytest.approx(0.05)

    def test_completion_before_any_chunks_opens_the_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A completion with no prior partial record opens the request.

        It is an ordinary first record, not a correction.
        """
        monkeypatch.setattr(
            "deepagents_code.cost_tracking.estimate_cost",
            lambda _usage, _model, _provider="": 0.05,
        )
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}
        completion = AIMessage(
            content="done",
            id="child-1",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 5,
                "total_tokens": 105,
            },
            response_metadata={"model_name": "real-model", "model_provider": "openai"},
        )

        recorded = record_message_usage(stats, completion, recorded_requests=ledger)

        assert recorded is not None
        assert recorded.cost_usd == pytest.approx(0.05)
        assert recorded.request_id == "child-1"
        assert stats.request_count == 1

    def test_retry_attempt_scoping_separates_completions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two attempts reusing one message ID are distinct requests.

        A completion finalizes only its own attempt's entry.
        """
        monkeypatch.setattr(
            "deepagents_code.cost_tracking.estimate_cost",
            lambda _usage, _model, _provider="": 0.1,
        )
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}
        chunk = AIMessageChunk(
            content="",
            id="run-1",
            usage_metadata={
                "input_tokens": 900,
                "output_tokens": 10,
                "total_tokens": 910,
            },
            response_metadata={"model_provider": "openai"},
        )
        completion = AIMessage(
            content="done",
            id="run-1",
            usage_metadata={
                "input_tokens": 1_000,
                "output_tokens": 20,
                "total_tokens": 1_020,
            },
            response_metadata={"model_provider": "openai"},
        )

        record_message_usage(stats, chunk, recorded_requests=ledger, attempt_scope=1)
        finalized = record_message_usage(
            stats, completion, recorded_requests=ledger, attempt_scope=1
        )
        retry = record_message_usage(
            stats, chunk, recorded_requests=ledger, attempt_scope=2
        )

        assert finalized is not None
        assert finalized.request_id == "run-1"
        assert retry is not None
        assert stats.request_count == 2
        assert stats.input_tokens == 1_900
        assert stats.output_tokens == 30
        assert ledger[1, "run-1"].finalized is True
        assert ledger[2, "run-1"].finalized is False

    def test_missing_response_model_uses_request_specific_configured_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A child's configured model beats the main agent's fallback.

        The request metadata describes this very call, so it must win over the
        caller's `runtime_state` fallback, which describes the parent.
        """
        priced_models: list[str] = []

        def fake_price(_usage: object, model: str, _provider: str = "") -> float | None:
            priced_models.append(model)
            return 0.25 if model == "child-model" else 9.99

        monkeypatch.setattr("deepagents_code.cost_tracking.estimate_cost", fake_price)
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}
        chunk = AIMessageChunk(
            content="",
            id="child-1",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
            },
            response_metadata={"model_provider": "openai"},
        )

        recorded = record_message_usage(
            stats,
            chunk,
            fallback_model="main-model",
            fallback_provider="openai",
            request_metadata={
                "deepagents_code_configured_model": "child-model",
                "deepagents_code_configured_provider": "openai",
            },
            kind="subagent",
            recorded_requests=ledger,
        )

        assert recorded is not None
        assert recorded.cost_usd == pytest.approx(0.25)
        assert stats.per_model["openai", "child-model"].request_count == 1
        assert ("openai", "main-model") not in stats.per_model
        assert priced_models == ["child-model"]

    def test_explicit_response_model_still_beats_configured_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A response that names its model keeps precedence over request metadata.

        The configured model only fills in when the response omits one; the
        existing response-first chain must not be inverted.
        """
        priced_models: list[str] = []

        def fake_price(_usage: object, model: str, _provider: str = "") -> float | None:
            priced_models.append(model)
            return 0.25

        monkeypatch.setattr("deepagents_code.cost_tracking.estimate_cost", fake_price)
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}
        chunk = AIMessageChunk(
            content="",
            id="child-1",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
            },
            response_metadata={
                "model_name": "response-model",
                "model_provider": "openai",
            },
        )

        record_message_usage(
            stats,
            chunk,
            fallback_model="main-model",
            fallback_provider="openai",
            request_metadata={
                "deepagents_code_configured_model": "child-model",
                "deepagents_code_configured_provider": "openai",
            },
            kind="subagent",
            recorded_requests=ledger,
        )

        assert stats.per_model["openai", "response-model"].request_count == 1
        assert ("openai", "child-model") not in stats.per_model
        assert priced_models == ["response-model"]

    def test_configured_provider_alias_wins_over_generic_response_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An aliased configured provider still replaces a generic response one.

        `azure_openai` maps to the `genai-prices` identifier the catalog knows;
        pricing under the response's bare `openai` would use the wrong rates.
        """
        priced_providers: list[str] = []

        def fake_price(_usage: object, _model: str, provider: str = "") -> float | None:
            priced_providers.append(provider)
            return 0.25

        monkeypatch.setattr("deepagents_code.cost_tracking.estimate_cost", fake_price)
        stats = SessionStats()
        chunk = AIMessageChunk(
            content="",
            id="child-1",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
            },
            response_metadata={"model_name": "gpt-5.5", "model_provider": "openai"},
        )

        record_message_usage(
            stats,
            chunk,
            fallback_model="main-model",
            fallback_provider="openai",
            request_metadata={
                "deepagents_code_configured_model": "gpt-5.5",
                "deepagents_code_configured_provider": "azure_openai",
            },
            kind="subagent",
            recorded_requests={},
        )

        assert priced_providers == ["azure_openai"]
        assert stats.per_model["azure_openai", "gpt-5.5"].request_count == 1

    def test_without_request_metadata_the_parent_fallback_stays_subagent_shy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No request metadata and a nested call keeps prior behavior.

        The parent fallback must not silently price the child as the parent's
        model when the response names nothing.
        """
        priced_models: list[str] = []

        def fake_price(_usage: object, model: str, _provider: str = "") -> float | None:
            priced_models.append(model)
            return 0.25

        monkeypatch.setattr("deepagents_code.cost_tracking.estimate_cost", fake_price)
        stats = SessionStats()
        chunk = AIMessageChunk(
            content="",
            id="child-1",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
            },
            response_metadata={},
        )

        record_message_usage(
            stats,
            chunk,
            fallback_model="main-model",
            fallback_provider="openai",
            kind="subagent",
            recorded_requests={},
        )

        # Pre-existing behavior: an unnamed nested response is priced under the
        # caller's fallback because nothing request-specific is known.
        assert stats.per_model["openai", "main-model"].request_count == 1
        assert priced_models == ["main-model"]


class TestParallelChildrenAndLateCorrections:
    """Interleaved children, backend totals, and late usage corrections."""

    @staticmethod
    def _chunk(
        message_id: str,
        input_tokens: int,
        *,
        names_model: bool = True,
    ) -> AIMessageChunk:
        return AIMessageChunk(
            content="",
            id=message_id,
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": 10,
                "total_tokens": input_tokens + 10,
            },
            response_metadata=(
                {"model_name": "expensive-model", "model_provider": "openai"}
                if names_model
                else {"model_provider": "openai"}
            ),
        )

    def test_a_late_correction_reports_a_negative_delta_for_its_own_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two children recorded provisionally, then a completion corrects.

        The second child's completion corrects it down after the first's
        backend total already reset the display.
        """
        monkeypatch.setattr(
            "deepagents_code.cost_tracking.estimate_cost",
            lambda _usage, model, _provider="": 0.05 if model == "real-model" else 0.5,
        )
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}
        record_message_usage(
            stats, self._chunk("child-1", 900), recorded_requests=ledger
        )
        record_message_usage(
            stats, self._chunk("child-2", 900), recorded_requests=ledger
        )

        completion = AIMessage(
            content="done",
            id="child-2",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 5,
                "total_tokens": 105,
            },
            response_metadata={"model_name": "real-model", "model_provider": "openai"},
        )
        correction = record_message_usage(stats, completion, recorded_requests=ledger)

        assert correction is not None
        assert correction.request_id == "child-2"
        # Signed so the caller can reconcile request-keyed provisional spend;
        # applying it to the whole accumulator would subtract child-1's $0.50.
        assert correction.cost_usd == pytest.approx(-0.45)
        assert stats.request_count == 2

    def test_event_with_changed_model_replaces_the_partial_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A nested event correcting model and tokens re-files its request."""
        priced: dict[str, float] = {}

        def fake_price(usage: object, model: str, _provider: str = "") -> float | None:
            rates = {"expensive-model": 1.0, "cheap-model": 0.01, "": 0.0}
            priced[model] = rates[model]
            input_tokens = cast("dict[str, Any]", usage)["input_tokens"]
            return input_tokens / 1000 * rates[model]

        monkeypatch.setattr("deepagents_code.cost_tracking.estimate_cost", fake_price)
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}
        partial = {
            "type": "model_usage",
            "version": 1,
            "request_id": "child-1",
            "usage_metadata": {
                "input_tokens": 1_000,
                "output_tokens": 100,
                "total_tokens": 1_100,
            },
            "model_name": "expensive-model",
            "provider": "openai",
            "thread_id": "thread-1",
            "scope": "tools:task",
        }
        # A nested event arrives complete by construction, but the message
        # stream can have recorded partial chunks for the same request first.
        record_message_usage(
            stats, self._chunk("child-1", 1_000), recorded_requests=ledger
        )

        corrected = record_model_usage_event(
            stats,
            partial
            | {
                "model_name": "cheap-model",
                "usage_metadata": {
                    "input_tokens": 500,
                    "output_tokens": 50,
                    "total_tokens": 550,
                    "input_token_details": {"cache_read": 400},
                },
            },
            active_thread_id="thread-1",
            recorded_requests=ledger,
        )

        assert corrected is not None
        assert corrected.cost_usd == pytest.approx(
            500 / 1000 * 0.01 - 1_000 / 1000 * 1.0
        )
        assert corrected.request_id == "child-1"
        assert stats.request_count == 1
        assert stats.total_cost_usd == pytest.approx(500 / 1000 * 0.01)
        assert stats.input_tokens == 500
        assert stats.cache_read_tokens == 400
        assert ("openai", "cheap-model") in stats.per_model
        assert ("openai", "expensive-model") not in stats.per_model


class TestRecordModelUsageEvent:
    """Nested usage custom events share ordinary message accounting."""

    @staticmethod
    def _event() -> dict[str, object]:
        return {
            "type": "model_usage",
            "version": 1,
            "request_id": "child-1",
            "usage_metadata": {
                "input_tokens": 1_000,
                "output_tokens": 100,
                "total_tokens": 1_100,
                "input_token_details": {"cache_read": 800},
            },
            "model_name": "gpt-5.5",
            "provider": "openai",
            "thread_id": "thread-1",
            "scope": "tools:task",
        }

    def test_records_subagent_usage_once(self) -> None:
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}

        first = record_model_usage_event(
            stats,
            self._event(),
            active_thread_id="thread-1",
            recorded_requests=ledger,
        )
        replay = record_model_usage_event(
            stats,
            self._event(),
            active_thread_id="thread-1",
            recorded_requests=ledger,
        )

        assert first is not None
        assert replay is None
        assert stats.request_count == 1
        assert stats.per_kind["subagent"].request_count == 1
        assert stats.cache_read_tokens == 800
        assert ("openai", "gpt-5.5") in stats.per_model

    def test_deduplicates_with_ordinary_message(self) -> None:
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}
        message = AIMessage(
            content="done",
            id="child-1",
            usage_metadata={
                "input_tokens": 1_000,
                "output_tokens": 100,
                "total_tokens": 1_100,
            },
            response_metadata={
                "model_name": "gpt-5.5",
                "model_provider": "openai",
            },
        )

        record_message_usage(stats, message, kind="subagent", recorded_requests=ledger)
        replay = record_model_usage_event(
            stats,
            self._event(),
            active_thread_id="thread-1",
            recorded_requests=ledger,
        )

        assert replay is None
        assert stats.request_count == 1

    @pytest.mark.parametrize(
        "update",
        [
            {"version": True},
            {"version": MODEL_USAGE_EVENT_VERSION + 1},
            {"request_id": ""},
            {"usage_metadata": "tokens"},
            {"scope": ""},
        ],
    )
    def test_rejects_malformed_event(self, update: dict[str, object]) -> None:
        event = self._event() | update
        stats = SessionStats()

        assert (
            record_model_usage_event(
                stats,
                event,
                active_thread_id="thread-1",
                recorded_requests={},
            )
            is None
        )
        assert stats.request_count == 0

    def test_rejects_another_thread(self) -> None:
        stats = SessionStats()

        assert (
            record_model_usage_event(
                stats,
                self._event(),
                active_thread_id="other-thread",
                recorded_requests={},
            )
            is None
        )
        assert stats.request_count == 0


class TestClassifyUsageKind:
    """Request classification for cost breakdowns."""


class TestPrintUsageTable:
    """Tests for `print_usage_table` output."""


class TestUsageTableEnabled:
    """Test the gate that decides whether the usage table renders."""

    def test_resolution_failure_keeps_the_table(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A raising resolver logs and returns `True` instead of propagating.

        Both callers run at teardown: in the TUI an escaping exception is caught
        by the handler that rewrites a clean exit into `1` plus a traceback, and
        in the headless run it would skip the `AGENT_COMPLETED` notification and
        the `session.end` hooks. Failing open on a cosmetic table is the cheap
        outcome; failing shut on session teardown is not.
        """

        def _boom(
            _key: str,
            *,
            fallback: bool,  # noqa: ARG001
            on_rejected: object = None,  # noqa: ARG001
        ) -> bool:
            msg = "managed policy refresh exploded"
            raise RuntimeError(msg)

        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_bool_display_preference", _boom
        )

        with caplog.at_level(logging.WARNING, logger="deepagents_code._session_stats"):
            assert usage_table_enabled() is True

        assert "show_usage_stats" in caplog.text
        # `exc_info=True`, so the cause is diagnosable rather than swallowed.
        assert "managed policy refresh exploded" in caplog.text

    def test_blocking_error_is_not_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`BlockingError` propagates instead of failing open.

        The fail-open exists for config hiccups. Blocking I/O on the event loop
        is a regression in the caller — this runs directly inside the async
        headless teardown — and swallowing it would hide the violation *and*
        silently ignore the user's opt-out. Matched by class name because
        `blockbuster` is not a runtime dependency here, so the test defines its
        own class rather than importing one.
        """

        class BlockingError(Exception):
            """Stands in for `blockbuster.BlockingError`."""

        def _blocked(
            _key: str,
            *,
            fallback: bool,  # noqa: ARG001
            on_rejected: object = None,  # noqa: ARG001
        ) -> bool:
            msg = "blocking call to io.TextIOWrapper.read"
            raise BlockingError(msg)

        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_bool_display_preference", _blocked
        )

        with pytest.raises(BlockingError):
            usage_table_enabled()


class TestAttemptScopedUsage:
    """Attempt-scoped dedupe for retries that reuse a provider message ID."""

    def test_chunks_and_corrections_merge_within_one_attempt(self) -> None:
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}

        record_message_usage(
            stats, self._chunk(1_000, 60), recorded_requests=ledger, attempt_scope=1
        )
        record_message_usage(
            stats, self._chunk(-200, 40), recorded_requests=ledger, attempt_scope=1
        )

        assert stats.request_count == 1
        assert stats.input_tokens == 800
        assert stats.output_tokens == 100

    def test_finalize_closes_scoped_entries(self) -> None:
        """The round boundary applies to scoped keys exactly as to bare ones."""
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}

        record_message_usage(
            stats, self._chunk(1_000, 100), recorded_requests=ledger, attempt_scope=1
        )
        finalize_recorded_requests(ledger)
        replay = record_message_usage(
            stats, self._chunk(1_000, 100), recorded_requests=ledger, attempt_scope=1
        )

        assert replay is None
        assert stats.request_count == 1

    def test_model_correction_hits_the_attempt_it_belongs_to(self) -> None:
        """A late model-naming chunk must re-file its own attempt's request."""
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}

        record_message_usage(
            stats,
            self._chunk(1_000, 100, names_model=False),
            fallback_model="configured-model",
            fallback_provider="openai",
            recorded_requests=ledger,
            attempt_scope=1,
        )
        # A second attempt of the same message ID completes without naming a
        # model; the first attempt's late correction must not touch it.
        record_message_usage(
            stats,
            AIMessage(
                content="done",
                id="run-1",
                usage_metadata={
                    "input_tokens": 2_000,
                    "output_tokens": 50,
                    "total_tokens": 2_050,
                },
            ),
            recorded_requests=ledger,
            attempt_scope=2,
        )
        correction = record_message_usage(
            stats,
            self._chunk(0, 0, names_model=True),
            fallback_model="configured-model",
            fallback_provider="openai",
            recorded_requests=ledger,
            attempt_scope=1,
        )

        assert correction is not None
        assert stats.request_count == 2
        entry = stats.per_model["openai", "gpt-5.5"]
        assert entry.request_count == 1
        assert entry.input_tokens == 1_000
        assert entry.output_tokens == 100
        assert stats.input_tokens == 3_000
        assert stats.output_tokens == 150
        assert stats.per_kind["assistant"].request_count == 2

    def test_model_usage_event_dedupes_per_attempt_scope(self) -> None:
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}
        event = {
            "type": "model_usage",
            "version": 1,
            "request_id": "child-1",
            "usage_metadata": {
                "input_tokens": 1_000,
                "output_tokens": 100,
                "total_tokens": 1_100,
            },
            "model_name": "gpt-5.5",
            "provider": "openai",
            "thread_id": "thread-1",
            "scope": "tools:task",
        }

        first = record_model_usage_event(
            stats,
            event,
            active_thread_id="thread-1",
            recorded_requests=ledger,
            attempt_scope="attempt-a",
        )
        retry = record_model_usage_event(
            stats,
            event,
            active_thread_id="thread-1",
            recorded_requests=ledger,
            attempt_scope="attempt-b",
        )
        replay = record_model_usage_event(
            stats,
            event,
            active_thread_id="thread-1",
            recorded_requests=ledger,
            attempt_scope="attempt-a",
        )

        assert first is not None
        assert retry is not None
        assert replay is None
        assert stats.request_count == 2
        assert stats.per_kind["subagent"].request_count == 2

    def test_none_scope_and_scoped_attempt_are_distinct_requests(self) -> None:
        """Unscoped legacy recording must not collide with scoped attempts."""
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}

        legacy = record_message_usage(
            stats, self._chunk(1_000, 100), recorded_requests=ledger
        )
        scoped = record_message_usage(
            stats, self._chunk(1_000, 100), recorded_requests=ledger, attempt_scope=1
        )

        assert legacy is not None
        assert scoped is not None
        assert stats.request_count == 2

    def test_none_scope_preserves_legacy_dedupe(self) -> None:
        """Without a scope, a completed-message replay still counts once."""
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}
        message = AIMessage(
            content="done",
            id="run-1",
            usage_metadata={
                "input_tokens": 1_000,
                "output_tokens": 100,
                "total_tokens": 1_100,
            },
        )

        first = record_message_usage(stats, message, recorded_requests=ledger)
        replay = record_message_usage(stats, message, recorded_requests=ledger)

        assert first is not None
        assert replay is None
        assert stats.request_count == 1
        assert stats.input_tokens == 1_000
        assert stats.output_tokens == 100

    def test_resume_replay_credits_the_attempt_that_succeeded(self) -> None:
        """After a retry, the projected row carries the surviving attempt."""
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}

        record_message_usage(
            stats,
            self._chunk(1_000, 100),
            recorded_requests=ledger,
            attempt_scope=((), "call-1", 0),
        )
        record_message_usage(
            stats,
            self._chunk(2_000, 200),
            recorded_requests=ledger,
            attempt_scope=((), "call-1", 1),
        )
        # Both attempts are real spend and both counted.
        assert stats.request_count == 2

        finalize_recorded_requests(ledger)
        replay = record_message_usage(
            stats, self._chunk(2_000, 200), recorded_requests=ledger
        )

        assert replay is None
        assert stats.request_count == 2
        # The bare-id projection took the last attempt written, which is the one
        # that actually succeeded.
        assert ledger["run-1"].input_tokens == 2_000

    def test_same_message_id_counts_once_per_attempt(self) -> None:
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}

        first = record_message_usage(
            stats, self._chunk(1_000, 100), recorded_requests=ledger, attempt_scope=1
        )
        retry = record_message_usage(
            stats, self._chunk(1_000, 100), recorded_requests=ledger, attempt_scope=2
        )

        assert first is not None
        assert retry is not None
        assert stats.request_count == 2
        assert stats.input_tokens == 2_000
        assert stats.output_tokens == 200
        assert stats.per_model["openai", "gpt-5.5"].request_count == 2

    def test_scoped_request_is_not_recounted_on_a_hitl_resume_replay(self) -> None:
        """A turn that crosses a HITL pause must not count its spend twice.

        The record pass keys by `(attempt_scope, message_id)`, but the attempt
        scope closes when the attempt completes -- and `model_attempt(complete)`
        always fires before the tool node interrupts. So the resume pass replays
        with no scope open and keys by the bare message id. Closing the ledger at
        the round boundary has to bridge the two shapes, or every turn containing
        one tool approval reports double the tokens and cost.
        """
        stats = SessionStats()
        ledger: dict[UsageLedgerKey, RecordedRequest] = {}
        scope = ((), "call-1", 0)

        recorded = record_message_usage(
            stats,
            self._chunk(1_000, 100),
            recorded_requests=ledger,
            attempt_scope=scope,
        )
        assert recorded is not None
        assert stats.request_count == 1

        # End of the stream round: the tool node interrupts for approval.
        finalize_recorded_requests(ledger)

        # Resume pass. The scope is long closed, so this replays unscoped.
        replay = record_message_usage(
            stats, self._chunk(1_000, 100), recorded_requests=ledger
        )

        assert replay is None
        assert stats.request_count == 1
        assert stats.input_tokens == 1_000
        assert stats.output_tokens == 100

    @staticmethod
    def _chunk(
        input_tokens: int,
        output_tokens: int,
        *,
        names_model: bool = True,
    ) -> AIMessageChunk:
        return AIMessageChunk(
            content="",
            id="run-1",
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            response_metadata=(
                {"model_name": "gpt-5.5", "model_provider": "openai"}
                if names_model
                else {"model_provider": "openai"}
            ),
        )
