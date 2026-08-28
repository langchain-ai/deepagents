"""Tests for _session_stats module."""

from __future__ import annotations

import logging
from io import StringIO
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from rich.console import Console

from deepagents_code._session_stats import (
    ModelStats,
    RecordedRequest,
    SessionStats,
    UsageLedgerKey,
    classify_usage_kind,
    finalize_recorded_requests,
    format_cost,
    format_cost_estimate,
    format_token_count,
    print_usage_table,
    record_message_usage,
    record_model_usage_event,
    usage_table_enabled,
)
from deepagents_code.cost_tracking import MODEL_USAGE_EVENT_VERSION

if TYPE_CHECKING:
    from pathlib import Path


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
