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

    @pytest.mark.parametrize(
        ("cost_usd", "expected"),
        [
            (0.0, "$0.00"),
            (-1.0, "$0.00"),
            (0.0001, "<$0.01"),
            (0.009, "<$0.01"),
            (0.01, "$0.01"),
            (0.42, "$0.42"),
            (12.5, "$12.50"),
        ],
    )
    def test_format(self, cost_usd: float, expected: str) -> None:
        assert format_cost(cost_usd) == expected


class TestFormatCostEstimate:
    """Tests for the rounded, approximate cost formatting used for estimates."""

    @pytest.mark.parametrize(
        ("cost_usd", "expected"),
        [
            # Edge conventions match `format_cost`.
            (0.0, "$0.00"),
            (-1.0, "$0.00"),
            (0.0001, "<$0.01"),
            (0.009, "<$0.01"),
            # Sub-dime values keep cent-level precision and round upward.
            (0.01, "~$0.01"),
            (0.042, "~$0.05"),
            (0.0999, "~$0.10"),
            # Two significant figures from a dime upward, always rounding up
            # so an upper-bound estimate never understates the cost.
            (0.6234, "~$0.63"),
            (0.5062, "~$0.51"),
            (0.999, "~$1.0"),
            (1.234, "~$1.3"),
            (1.25, "~$1.3"),
            (1.15, "~$1.2"),
            (1.0, "~$1.0"),
            (9.9, "~$9.9"),
            (9.99, "~$10"),
            (12.34, "~$13"),
            (99.9, "~$100"),
            (123.4, "~$130"),
            (999.9, "~$1000"),
        ],
    )
    def test_format(self, cost_usd: float, expected: str) -> None:
        assert format_cost_estimate(cost_usd) == expected

    def test_rounds_up_to_preserve_upper_bound(self) -> None:
        """A displayed upper bound must never be lower than the estimate."""
        assert format_cost_estimate(0.105) == "~$0.11"
        assert format_cost_estimate(0.115) == "~$0.12"
        assert format_cost_estimate(1.24) == "~$1.3"


class TestFormatTokenCount:
    """Tests for format_token_count()."""

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (0, "0"),
            (1, "1"),
            (999, "999"),
        ],
    )
    def test_small_counts(self, count: int, expected: str) -> None:
        assert format_token_count(count) == expected

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (1000, "1.0K"),
            (1500, "1.5K"),
            (12_500, "12.5K"),
            (999_999, "1000.0K"),
        ],
    )
    def test_thousands(self, count: int, expected: str) -> None:
        assert format_token_count(count) == expected

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (1_000_000, "1.0M"),
            (1_200_000, "1.2M"),
            (10_000_000, "10.0M"),
        ],
    )
    def test_millions(self, count: int, expected: str) -> None:
        assert format_token_count(count) == expected


class TestModelStats:
    """Tests for ModelStats dataclass."""

    def test_defaults(self) -> None:
        stats = ModelStats()
        assert stats.request_count == 0
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.cost_usd == pytest.approx(0.0)
        assert stats.priced_request_count == 0
        assert stats.provider == ""


class TestSessionStats:
    """Tests for SessionStats accumulation logic."""

    def test_defaults(self) -> None:
        stats = SessionStats()
        assert stats.request_count == 0
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.total_cost_usd == pytest.approx(0.0)
        assert stats.priced_request_count == 0
        assert stats.wall_time_seconds == pytest.approx(0.0)
        assert stats.per_model == {}

    def test_record_request_increments_totals(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 50)
        assert stats.request_count == 1
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50

    def test_record_request_accumulates(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 50)
        stats.record_request("gpt-5.5", 200, 75)
        assert stats.request_count == 2
        assert stats.input_tokens == 300
        assert stats.output_tokens == 125

    def test_record_request_populates_per_model(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 50)
        assert ("", "gpt-5.5") in stats.per_model
        model = stats.per_model["", "gpt-5.5"]
        assert model.request_count == 1
        assert model.input_tokens == 100
        assert model.output_tokens == 50
        assert model.model_name == "gpt-5.5"

    def test_record_request_multiple_models(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 50)
        stats.record_request("claude-sonnet-4-5", 200, 75)
        assert len(stats.per_model) == 2
        assert stats.per_model["", "gpt-5.5"].input_tokens == 100
        assert stats.per_model["", "claude-sonnet-4-5"].input_tokens == 200
        assert stats.request_count == 2
        assert stats.input_tokens == 300

    def test_record_request_records_provider(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 50, provider="openai")
        assert stats.per_model["openai", "gpt-5.5"].provider == "openai"

    def test_record_request_splits_same_model_by_provider(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 50, provider="openai")
        stats.record_request("gpt-5.5", 200, 75, provider="azure")

        assert len(stats.per_model) == 2
        assert stats.per_model["openai", "gpt-5.5"].input_tokens == 100
        assert stats.per_model["azure", "gpt-5.5"].input_tokens == 200

    def test_record_request_empty_model_skips_per_model(self) -> None:
        stats = SessionStats()
        stats.record_request("", 100, 50)
        assert stats.request_count == 1
        assert stats.input_tokens == 100
        assert stats.per_model == {}

    def test_record_request_accumulates_cost(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 50, cost_usd=0.01)
        stats.record_request("gpt-5.5", 200, 75, cost_usd=0.02)
        assert stats.total_cost_usd == pytest.approx(0.03)
        assert stats.priced_request_count == 2
        assert stats.per_model["", "gpt-5.5"].cost_usd == pytest.approx(0.03)

    def test_missing_cost_does_not_inflate_total(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 50, cost_usd=0.01)
        stats.record_request("unknown", 200, 75, cost_usd=None)
        assert stats.total_cost_usd == pytest.approx(0.01)
        assert stats.priced_request_count == 1
        assert stats.per_model["", "unknown"].priced_request_count == 0

    def test_literal_zero_cost_is_recorded_but_does_not_inflate_total(self) -> None:
        stats = SessionStats()
        stats.record_request("free-model", 100, 50, cost_usd=0.0)
        assert stats.total_cost_usd == pytest.approx(0.0)
        assert stats.priced_request_count == 1
        assert stats.per_model["", "free-model"].priced_request_count == 1

    def test_merge_combines_totals(self) -> None:
        a = SessionStats(
            request_count=1,
            input_tokens=100,
            output_tokens=50,
            wall_time_seconds=1.5,
        )
        b = SessionStats(
            request_count=2,
            input_tokens=200,
            output_tokens=75,
            wall_time_seconds=2.0,
        )
        a.merge(b)
        assert a.request_count == 3
        assert a.input_tokens == 300
        assert a.output_tokens == 125
        assert a.wall_time_seconds == pytest.approx(3.5)

    def test_merge_combines_cost(self) -> None:
        first = SessionStats()
        first.record_request("gpt-5.5", 100, 50, cost_usd=0.01)
        second = SessionStats()
        second.record_request("gpt-5.5", 200, 75, cost_usd=0.02)

        first.merge(second)

        assert first.total_cost_usd == pytest.approx(0.03)
        assert first.priced_request_count == 2
        assert first.per_model["", "gpt-5.5"].cost_usd == pytest.approx(0.03)

    def test_merge_combines_per_model(self) -> None:
        a = SessionStats()
        a.record_request("gpt-5.5", 100, 50)

        b = SessionStats()
        b.record_request("gpt-5.5", 200, 75)
        b.record_request("claude-sonnet-4-5", 300, 100)

        a.merge(b)
        assert a.per_model["", "gpt-5.5"].input_tokens == 300
        assert a.per_model["", "gpt-5.5"].request_count == 2
        assert a.per_model["", "claude-sonnet-4-5"].input_tokens == 300

    def test_merge_carries_provider(self) -> None:
        a = SessionStats()
        b = SessionStats()
        b.record_request("gpt-5.5", 200, 75, provider="openai")

        a.merge(b)
        assert a.per_model["openai", "gpt-5.5"].provider == "openai"

    def test_merge_splits_same_model_by_provider(self) -> None:
        a = SessionStats()
        a.record_request("gpt-5.5", 100, 50, provider="openai")

        b = SessionStats()
        b.record_request("gpt-5.5", 200, 75, provider="azure")

        a.merge(b)
        assert len(a.per_model) == 2
        assert a.per_model["openai", "gpt-5.5"].input_tokens == 100
        assert a.per_model["azure", "gpt-5.5"].input_tokens == 200

    def test_record_request_tracks_kind(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 50, cost_usd=0.01, kind="assistant")
        stats.record_request("gpt-5.5", 20, 5, cost_usd=0.02, kind="offload")
        assert stats.per_kind["assistant"].cost_usd == pytest.approx(0.01)
        assert stats.per_kind["offload"].cost_usd == pytest.approx(0.02)
        assert stats.per_kind["offload"].request_count == 1

    def test_side_call_keeps_explicit_provider_over_parent_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A cross-provider subagent must not inherit the main provider."""
        from deepagents_code import cost_tracking

        monkeypatch.setattr(cost_tracking, "estimate_cost", lambda *_args: 0.25)
        stats = SessionStats()
        message = AIMessage(
            content="done",
            usage_metadata={
                "input_tokens": 1_000,
                "output_tokens": 100,
                "total_tokens": 1_100,
            },
            response_metadata={
                "model_name": "side-model",
                "model_provider": "anthropic",
            },
        )

        recorded = record_message_usage(
            stats,
            message,
            fallback_model="parent-model",
            fallback_provider="openai_codex",
            kind="subagent",
        )

        assert recorded is not None
        assert recorded.cost_usd is not None
        assert ("anthropic", "side-model") in stats.per_model

    @pytest.mark.parametrize("configured_provider", ["azure_openai", "openai_codex"])
    def test_side_call_uses_its_request_provider_alias(
        self,
        monkeypatch: pytest.MonkeyPatch,
        configured_provider: str,
    ) -> None:
        """An inherited side model keeps its actual configured provider."""
        from deepagents_code import cost_tracking

        priced_providers: list[str] = []

        def price(
            usage_metadata: object,
            model_name: str,
            provider: str = "",
        ) -> float:
            assert usage_metadata
            assert model_name == "side-model"
            priced_providers.append(provider)
            return 0.25

        monkeypatch.setattr(cost_tracking, "estimate_cost", price)
        stats = SessionStats()
        message = AIMessage(
            content="done",
            usage_metadata={
                "input_tokens": 1_000,
                "output_tokens": 100,
                "total_tokens": 1_100,
            },
            response_metadata={
                "model_name": "side-model",
                "model_provider": "openai",
            },
        )

        recorded = record_message_usage(
            stats,
            message,
            fallback_model="parent-model",
            fallback_provider=configured_provider,
            request_metadata={
                cost_tracking._CONFIGURED_PROVIDER_METADATA_KEY: configured_provider
            },
            kind="subagent",
        )

        assert recorded is not None
        assert priced_providers == [configured_provider]
        assert (configured_provider, "side-model") in stats.per_model

    def test_merge_combines_kinds(self) -> None:
        first = SessionStats()
        first.record_request("gpt-5.5", 100, 50, cost_usd=0.01, kind="subagent")
        second = SessionStats()
        second.record_request("gpt-5.5", 200, 75, cost_usd=0.03, kind="subagent")
        first.merge(second)
        assert first.per_kind["subagent"].cost_usd == pytest.approx(0.04)
        assert first.per_kind["subagent"].request_count == 2

    def test_merge_empty_into_populated(self) -> None:
        a = SessionStats(request_count=5, input_tokens=500)
        b = SessionStats()
        a.merge(b)
        assert a.request_count == 5
        assert a.input_tokens == 500


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
        ledger: dict[str, RecordedRequest] = {}
        record_message_usage(stats, self._chunk(1_000, 100), recorded_requests=ledger)

        finalize_recorded_requests(ledger)
        replayed = record_message_usage(
            stats, self._chunk(1_000, 100), recorded_requests=ledger
        )

        assert replayed is None
        assert stats.request_count == 1
        assert stats.input_tokens == 1_000
        assert stats.output_tokens == 100

    def test_round_boundary_does_not_break_incremental_chunks(self) -> None:
        """Chunks within one round must still revise, not start a new request."""
        stats = SessionStats()
        ledger: dict[str, RecordedRequest] = {}

        record_message_usage(stats, self._chunk(1_000, 60), recorded_requests=ledger)
        record_message_usage(stats, self._chunk(0, 40), recorded_requests=ledger)

        assert stats.request_count == 1
        assert stats.input_tokens == 1_000
        assert stats.output_tokens == 100

    def test_reprice_to_an_unpriceable_model_retracts_the_estimate(self) -> None:
        """The caller must be told to drop the estimate it already displayed.

        Reporting `None` would leave the provisional display holding a cost the
        accumulator no longer has, so the status bar and `/cost` disagree.
        """
        stats = SessionStats()
        ledger: dict[str, RecordedRequest] = {}
        first = record_message_usage(
            stats,
            self._chunk(1_000, 60, names_model=False),
            fallback_model="gpt-5.5",
            fallback_provider="openai",
            recorded_requests=ledger,
        )
        assert first is not None
        assert first.cost_usd is not None

        uncatalogued = AIMessageChunk(
            content="",
            id="run-1",
            usage_metadata={
                "input_tokens": 0,
                "output_tokens": 40,
                "total_tokens": 40,
            },
            response_metadata={
                "model_name": "totally-made-up-model-zzz",
                "model_provider": "nowhere",
            },
        )
        second = record_message_usage(
            stats,
            uncatalogued,
            fallback_model="gpt-5.5",
            fallback_provider="openai",
            recorded_requests=ledger,
        )

        assert second is not None
        assert second.cost_usd == pytest.approx(-first.cost_usd)
        assert stats.total_cost_usd == pytest.approx(0.0)
        assert stats.priced_request_count == 0

    def test_retracting_a_kind_s_only_request_drops_its_row(self) -> None:
        """An all-zero kind row would show as an empty line in the breakdown."""
        stats = SessionStats()
        ledger: dict[str, RecordedRequest] = {}
        record_message_usage(
            stats, self._chunk(1_000, 60), kind="subagent", recorded_requests=ledger
        )

        record_message_usage(
            stats, self._chunk(0, 40), kind="subagent", recorded_requests=ledger
        )

        assert [entry.request_count for entry in stats.per_kind.values()] == [1]

    def test_per_chunk_deltas_sum_into_one_request(self) -> None:
        """Google reports an incremental delta on every chunk of one message.

        Dropping chunks after the first would lose most of the request's output
        tokens and cost; counting each chunk as its own request would inflate
        the request and priced-request counts instead. Equal consecutive deltas
        must survive too, so token counts cannot stand in for stream position.
        """
        stats = SessionStats()
        ledger: dict[str, RecordedRequest] = {}

        for output_tokens in (34, 33, 33):
            record_message_usage(
                stats,
                self._chunk(1_000, output_tokens),
                recorded_requests=ledger,
            )

        assert stats.output_tokens == 100
        assert stats.request_count == 1
        assert stats.priced_request_count == 1
        assert stats.per_kind["assistant"].request_count == 1
        assert [entry.request_count for entry in stats.per_model.values()] == [1]

    def test_model_named_only_on_the_final_chunk_owns_the_whole_request(
        self,
    ) -> None:
        """Google attaches `model_name` only to the chunk with `finish_reason`.

        Without carrying the model forward, one API call would straddle a
        fallback-model row and a real-model row in the breakdown.
        """
        stats = SessionStats()
        ledger: dict[str, RecordedRequest] = {}

        record_message_usage(
            stats,
            self._chunk(1_000, 60, names_model=False),
            fallback_model="configured-model",
            fallback_provider="openai",
            recorded_requests=ledger,
        )
        record_message_usage(
            stats,
            self._chunk(0, 40, names_model=False),
            fallback_model="configured-model",
            fallback_provider="openai",
            recorded_requests=ledger,
        )
        record_message_usage(
            stats,
            self._chunk(0, 0, names_model=True),
            fallback_model="configured-model",
            fallback_provider="openai",
            recorded_requests=ledger,
        )

        assert list(stats.per_model) == [("openai", "gpt-5.5")]
        entry = stats.per_model["openai", "gpt-5.5"]
        assert entry.request_count == 1
        assert entry.output_tokens == 100
        assert stats.request_count == 1

        # The cost must be re-derived under the model finally named. The early
        # chunks were priced against an unpriceable fallback, so keeping their
        # estimates would file a priceable request under `gpt-5.5` showing
        # nothing spent.
        reference = SessionStats()
        record_message_usage(
            reference,
            self._chunk(1_000, 100, names_model=True),
            fallback_model="configured-model",
            fallback_provider="openai",
            recorded_requests={},
        )
        assert stats.priced_request_count == 1
        assert stats.total_cost_usd > 0
        assert stats.total_cost_usd == pytest.approx(reference.total_cost_usd)
        assert entry.cost_usd == pytest.approx(reference.total_cost_usd)

    def test_split_stream_costs_the_same_as_an_unsplit_one(self) -> None:
        """Chunking is a transport detail; it must not change the estimate."""
        split = SessionStats()
        ledger: dict[str, RecordedRequest] = {}
        for chunk in (
            self._chunk(1_000, 60, names_model=False),
            self._chunk(0, 40, names_model=True),
        ):
            record_message_usage(
                split,
                chunk,
                fallback_model="totally-unknown-model",
                fallback_provider="openai",
                recorded_requests=ledger,
            )

        whole = SessionStats()
        record_message_usage(
            whole,
            self._chunk(1_000, 100, names_model=True),
            fallback_model="totally-unknown-model",
            fallback_provider="openai",
            recorded_requests={},
        )

        assert whole.total_cost_usd > 0
        assert split.total_cost_usd == pytest.approx(whole.total_cost_usd)
        assert split.priced_request_count == whole.priced_request_count == 1

    def test_negative_input_correction_lowers_the_displayed_tokens(self) -> None:
        """Gemini can revise its prompt count *down* on the final chunk.

        `langchain-google-genai` emits a negative `input_tokens` delta to
        compensate, treating the lower cumulative count as ground truth. Per
        message that floors to zero, so displaying the summed per-chunk counts
        would leave the token readout above the count the cost was based on.
        """
        stats = SessionStats()
        ledger: dict[str, RecordedRequest] = {}

        record_message_usage(
            stats,
            self._chunk(1_000, 60, names_model=False),
            recorded_requests=ledger,
        )
        record_message_usage(
            stats,
            self._chunk(-200, 40, names_model=True),
            recorded_requests=ledger,
        )

        corrected = SessionStats()
        record_message_usage(corrected, self._chunk(800, 100), recorded_requests={})

        assert stats.input_tokens == 800
        assert stats.output_tokens == 100
        assert stats.request_count == 1
        assert stats.total_cost_usd == pytest.approx(corrected.total_cost_usd)
        entry = stats.per_model["openai", "gpt-5.5"]
        assert entry.input_tokens == 800

    def test_correction_only_chunk_still_revises_the_request(self) -> None:
        """A correction with no positive counts must not be discarded."""
        stats = SessionStats()
        ledger: dict[str, RecordedRequest] = {}

        record_message_usage(stats, self._chunk(1_000, 100), recorded_requests=ledger)
        record_message_usage(stats, self._chunk(-200, 0), recorded_requests=ledger)

        assert stats.input_tokens == 800
        assert stats.output_tokens == 100
        assert stats.request_count == 1

    def test_cached_token_details_survive_the_merge(self) -> None:
        """Cache buckets carry their own rates, so merging must keep them."""
        stats = SessionStats()
        ledger: dict[str, RecordedRequest] = {}

        for output_tokens in (60, 40):
            message = AIMessageChunk(
                content="",
                id="run-1",
                usage_metadata={
                    "input_tokens": 1_000 if output_tokens == 60 else 0,
                    "output_tokens": output_tokens,
                    "total_tokens": (1_000 if output_tokens == 60 else 0)
                    + output_tokens,
                    "input_token_details": {
                        "cache_read": 800 if output_tokens == 60 else 0
                    },
                },
                response_metadata={
                    "model_name": "gpt-5.5",
                    "model_provider": "openai",
                },
            )
            record_message_usage(stats, message, recorded_requests=ledger)

        uncached = SessionStats()
        record_message_usage(uncached, self._chunk(1_000, 100), recorded_requests={})

        # Cache reads are cheaper than ordinary input, so losing the detail in
        # the merge would silently overprice the request or status metric.
        assert stats.total_cost_usd < uncached.total_cost_usd
        assert stats.cache_read_tokens == 800
        assert stats.cache_write_tokens == 0

    def test_reports_message_delta_and_running_request_tokens(self) -> None:
        """Pricing needs a delta while context display needs the running total."""
        stats = SessionStats()
        ledger: dict[str, RecordedRequest] = {}

        first = record_message_usage(
            stats, self._chunk(1_000, 60), recorded_requests=ledger
        )
        second = record_message_usage(
            stats, self._chunk(0, 40), recorded_requests=ledger
        )

        assert first is not None
        assert second is not None
        assert first.output_tokens == 60
        assert second.output_tokens == 40
        assert second.input_tokens == 0
        assert first.request_tokens == 1_060
        assert second.request_tokens == 1_100
        assert first.cost_usd is not None
        assert second.cost_usd is not None
        assert first.cost_usd + second.cost_usd == pytest.approx(stats.total_cost_usd)

    def test_completed_message_replay_is_recorded_once(self) -> None:
        """A resumed stream replays a completed message; it must not re-count."""
        stats = SessionStats()
        ledger: dict[str, RecordedRequest] = {}
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

    def test_completed_replay_after_chunks_is_not_added_on_top(self) -> None:
        """Chunks mark their ID so a later whole-message replay is skipped."""
        stats = SessionStats()
        ledger: dict[str, RecordedRequest] = {}

        record_message_usage(stats, self._chunk(1_000, 100), recorded_requests=ledger)
        replay = record_message_usage(
            stats,
            AIMessage(
                content="done",
                id="run-1",
                usage_metadata={
                    "input_tokens": 1_000,
                    "output_tokens": 100,
                    "total_tokens": 1_100,
                },
            ),
            recorded_requests=ledger,
        )

        assert replay is None
        assert stats.request_count == 1
        assert stats.output_tokens == 100

    def test_usage_without_an_id_is_always_recorded(self) -> None:
        """Anthropic's usage-bearing chunk carries no ID, so it cannot dedupe."""
        stats = SessionStats()
        ledger: dict[str, RecordedRequest] = {}

        for _ in range(2):
            record_message_usage(
                stats,
                self._chunk(1_000, 100, message_id=None),
                recorded_requests=ledger,
            )

        assert stats.request_count == 2
        assert not ledger

    def test_unusable_message_is_not_entered_in_the_ledger(self) -> None:
        """An unusable first pass must not suppress the real usage later."""
        stats = SessionStats()
        ledger: dict[str, RecordedRequest] = {}
        empty = AIMessage(
            content="",
            id="run-1",
            usage_metadata={
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )
        real = AIMessage(
            content="done",
            id="run-1",
            usage_metadata={
                "input_tokens": 1_000,
                "output_tokens": 100,
                "total_tokens": 1_100,
            },
        )

        assert record_message_usage(stats, empty, recorded_requests=ledger) is None
        assert record_message_usage(stats, real, recorded_requests=ledger) is not None
        assert stats.request_count == 1
        assert stats.output_tokens == 100

    def test_total_only_usage_falls_back_to_input(self) -> None:
        stats = SessionStats()
        message = AIMessageChunk(
            content="",
            id="run-1",
            usage_metadata={
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 1_200,
            },
        )

        recorded = record_message_usage(stats, message)

        assert recorded is not None
        assert recorded.input_tokens == 1_200
        assert recorded.output_tokens == 0

    def test_non_mapping_usage_is_ignored(self) -> None:
        stats = SessionStats()

        assert record_message_usage(stats, SimpleNamespace(usage_metadata=None)) is None
        assert record_message_usage(stats, SimpleNamespace()) is None
        assert stats.request_count == 0


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
        ledger: dict[str, RecordedRequest] = {}

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
        ledger: dict[str, RecordedRequest] = {}
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

    def test_nested_namespace_is_subagent(self) -> None:
        assert (
            classify_usage_kind(
                is_main_agent=False,
                metadata={"lc_source": "summarization"},
            )
            == "subagent"
        )

    def test_summarization_source_is_offload(self) -> None:
        assert (
            classify_usage_kind(
                is_main_agent=True, metadata={"lc_source": "summarization"}
            )
            == "offload"
        )

    def test_auto_classifier_source(self) -> None:
        assert (
            classify_usage_kind(
                is_main_agent=True,
                metadata={"lc_source": "auto_mode_classifier"},
            )
            == "auto"
        )

    def test_default_is_assistant(self) -> None:
        assert classify_usage_kind(is_main_agent=True, metadata=None) == "assistant"


class TestPrintUsageTable:
    """Tests for `print_usage_table` output."""

    def test_no_model_called_skips_unknown_row(self) -> None:
        """When no model was called, the table should not show 'unknown'."""
        stats = SessionStats()
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=1.5, console=console)
        output = buf.getvalue()
        assert "unknown" not in output
        assert "Usage Stats" not in output
        assert "Agent active" in output

    def test_single_model_shows_name(self) -> None:
        """Single-model session should display the model name."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50, cost_usd=0.42)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=2.0, console=console)
        output = buf.getvalue()
        assert "gpt-4" in output
        assert "Cost" in output
        assert "$0.42" in output
        assert "unknown" not in output

    def test_unpriced_model_does_not_render_zero_cost(self) -> None:
        stats = SessionStats()
        stats.record_request("self-hosted", 100, 50, cost_usd=None)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=0.0, console=console)
        output = buf.getvalue()
        assert "Cost" in output
        assert "$0.00" not in output
        assert "—" in output

    def test_shows_provider_name(self) -> None:
        """The table should include the provider for each model."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50, provider="openai")
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=2.0, console=console)
        output = buf.getvalue()
        assert "Provider" in output
        assert "openai" in output
        assert "gpt-4" in output

    def test_multi_model_shows_all_names_and_total(self) -> None:
        """Multi-model session should show each model and a Total row."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)
        stats.record_request("claude-opus-4-6", 200, 80)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=2.0, console=console)
        output = buf.getvalue()
        assert "gpt-4" in output
        assert "claude-opus-4-6" in output
        assert "Total" in output
        assert "unknown" not in output

    def test_same_model_with_different_providers_shows_separate_rows(self) -> None:
        """Same-name models from different providers should render separately."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50, provider="openai")
        stats.record_request("gpt-4", 200, 80, provider="azure")
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=2.0, console=console)
        output = buf.getvalue()
        assert "openai" in output
        assert "azure" in output
        assert "Total" in output
        # Two distinct rows, not a collapsed one: each provider's per-row token
        # counts must appear (100/50 and 200/80), alongside the 300/130 totals.
        assert "100" in output
        assert "50" in output
        assert "200" in output
        assert "80" in output

    def test_tokens_with_no_wall_time_omits_timing_line(self) -> None:
        """Token table should print but timing line should be absent."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=0.0, console=console)
        output = buf.getvalue()
        assert "gpt-4" in output
        assert "Agent active" not in output

    def test_no_requests_no_time_prints_nothing(self) -> None:
        """Empty stats with negligible wall time should print nothing."""
        stats = SessionStats()
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=0.01, console=console)
        output = buf.getvalue()
        assert output.strip() == ""


class TestUsageTableEnabled:
    """Test the gate that decides whether the usage table renders."""

    def test_enabled_by_default(self) -> None:
        """With no configuration in play the table renders."""
        assert usage_table_enabled() is True

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

    def test_import_error_is_not_reported_as_a_config_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken `config_manifest` import propagates, not fails open.

        The deferred import sits outside the `try` on purpose: an `ImportError`
        means the package is broken, not that the option could not be read, and
        reporting it as the latter would send a debugger to the wrong place.

        A `None` entry in `sys.modules` is the documented way to make an import
        of an otherwise-importable module fail.
        """
        import sys

        monkeypatch.setitem(sys.modules, "deepagents_code.config_manifest", None)

        with pytest.raises(ImportError):
            usage_table_enabled()

    def test_rejected_value_warns_on_stderr(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A misparsed value reaches the user, not just the log buffer.

        `show_usage_stats = "false"` is the expected typo — there is no
        `dcode config set`, so the only way to set this is hand-edited TOML.
        Falling through to `True` produces exactly the table the user meant to
        hide, and the resolver's own warning has no reader outside the TUI
        Debug Console. This is the one bool display option where that silence
        is the whole bug, so it is the one that prints.
        """
        import deepagents_code._session_stats as session_stats

        config_path = tmp_path / "config.toml"
        config_path.write_text('[ui]\nshow_usage_stats = "false"\n', encoding="utf-8")
        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_CONFIG_PATH", config_path
        )
        # Process-wide dedupe, so a leaked entry from another test would make
        # this pass for the wrong reason.
        monkeypatch.setattr(session_stats, "_warned_usage_stats_rejections", set())

        assert usage_table_enabled() is True

        captured = capsys.readouterr()
        assert "show_usage_stats" in captured.err
        # The warning is the entire point; it must not go to stdout, which a
        # headless caller may be piping as the agent's answer.
        assert captured.out == ""

    def test_rejected_value_warns_once_per_process(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Repeated resolution does not repeat an identical warning.

        Teardown resolves once per session today, but `dcode config` walks the
        whole manifest, and a line repeated verbatim reads as two separate
        problems. Dedupe is per reason, so two genuinely different rejections
        still both print.
        """
        import deepagents_code._session_stats as session_stats

        config_path = tmp_path / "config.toml"
        config_path.write_text("[ui]\nshow_usage_stats = 0\n", encoding="utf-8")
        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_CONFIG_PATH", config_path
        )
        monkeypatch.setattr(session_stats, "_warned_usage_stats_rejections", set())

        assert usage_table_enabled() is True
        assert usage_table_enabled() is True

        assert capsys.readouterr().err.count("Warning: ") == 1

    def test_shadowed_rejection_prints_nothing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A rejection at a tier that lost stays silent.

        With `DEEPAGENTS_CODE_SHOW_USAGE_STATS=0` winning over a user config
        holding the quoted `"false"` typo, the table is off and a warning
        announcing it would contradict what the user sees. The rejection is
        still logged through `_emit_ranked_diagnostics`; it just does not
        reach stderr.
        """
        import deepagents_code._session_stats as session_stats

        config_path = tmp_path / "config.toml"
        config_path.write_text('[ui]\nshow_usage_stats = "false"\n', encoding="utf-8")
        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_CONFIG_PATH", config_path
        )
        monkeypatch.setenv("DEEPAGENTS_CODE_SHOW_USAGE_STATS", "0")
        monkeypatch.setattr(session_stats, "_warned_usage_stats_rejections", set())

        assert usage_table_enabled() is False

        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out == ""

    def test_valid_value_prints_nothing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A value that parses stays silent on both streams.

        Guards the obvious regression in the other direction: a warning that
        fires for every well-formed config would be worse than no warning.
        """
        import deepagents_code._session_stats as session_stats

        config_path = tmp_path / "config.toml"
        config_path.write_text("[ui]\nshow_usage_stats = false\n", encoding="utf-8")
        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_CONFIG_PATH", config_path
        )
        monkeypatch.setattr(session_stats, "_warned_usage_stats_rejections", set())

        assert usage_table_enabled() is False

        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out == ""

    def test_env_var_overrides_the_config_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The env var outranks `config.toml`, per the resolver's tiers.

        This is the option's reason for declaring an env var at all: the
        headless run is where suppression matters, and a CI runner has env vars
        rather than a `~/.deepagents/config.toml`.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text("[ui]\nshow_usage_stats = true\n", encoding="utf-8")
        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_CONFIG_PATH", config_path
        )
        monkeypatch.setenv("DEEPAGENTS_CODE_SHOW_USAGE_STATS", "0")

        assert usage_table_enabled() is False

    def test_empty_env_var_disables_the_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty env var reads as off, matching the other default-on toggles.

        `empty_env_is_false` is what makes a bare `DEEPAGENTS_CODE_SHOW_USAGE_STATS=`
        in a CI env file mean "off" rather than falling through to the default.
        """
        monkeypatch.setenv("DEEPAGENTS_CODE_SHOW_USAGE_STATS", "")

        assert usage_table_enabled() is False
