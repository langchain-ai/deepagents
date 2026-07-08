"""Tests for formatting module."""

from __future__ import annotations

import locale
import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from deepagents_code.formatting import (
    format_duration,
    format_message_timestamp,
    uses_24_hour_clock,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def _utc_timezone() -> Iterator[None]:
    """Pin the process timezone to UTC for the duration of the block."""
    previous_tz = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    if hasattr(time, "tzset"):
        time.tzset()
    try:
        yield
    finally:
        if previous_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous_tz
        if hasattr(time, "tzset"):
            time.tzset()


class TestFormatDuration:
    """Tests for format_duration() helper."""

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0s"),
            (1, "1s"),
            (5, "5s"),
            (59, "59s"),
        ],
    )
    def test_whole_seconds(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.5, "0.5s"),
            (1.3, "1.3s"),
            (59.9, "59.9s"),
        ],
    )
    def test_fractional_seconds(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (60, "1m 0s"),
            (61, "1m 1s"),
            (90, "1m 30s"),
            (125, "2m 5s"),
            (3599, "59m 59s"),
        ],
    )
    def test_minutes(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (3600, "1h 0m 0s"),
            (3661, "1h 1m 1s"),
            (7384, "2h 3m 4s"),
        ],
    )
    def test_hours(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected

    def test_boundary_rounds_up_to_minute(self) -> None:
        """59.95 rounds to 60.0 which should render as 1m 0s."""
        assert format_duration(59.95) == "1m 0s"

    def test_whole_float_renders_without_decimal(self) -> None:
        """A float like 5.0 should render as '5s', not '5.0s'."""
        assert format_duration(5.0) == "5s"


class TestFormatMessageTimestamp:
    """Tests for format_message_timestamp() helper."""

    def test_today_omits_date(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Today's messages show only the time, not the date."""
        monkeypatch.setattr(
            "deepagents_code.formatting.uses_24_hour_clock", lambda: False
        )
        today = (
            datetime.now()
            .astimezone()
            .replace(hour=12, minute=0, second=5, microsecond=0)
        )
        assert format_message_timestamp(today.timestamp()) == "12:00:05 PM"

    def test_other_day_includes_date(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Messages from other days keep the leading date."""
        monkeypatch.setattr(
            "deepagents_code.formatting.uses_24_hour_clock", lambda: False
        )
        with _utc_timezone():
            # 2024-01-01 12:00:05 UTC — a fixed past date.
            assert format_message_timestamp(1_704_110_405.0) == "Jan 1, 12:00:05 PM"

    def test_24_hour_clock_drops_am_pm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 24-hour clock renders time without an AM/PM suffix."""
        monkeypatch.setattr(
            "deepagents_code.formatting.uses_24_hour_clock", lambda: True
        )
        with _utc_timezone():
            # 2024-01-01 13:00:05 UTC — a fixed past afternoon time.
            assert format_message_timestamp(1_704_114_005.0) == "Jan 1, 13:00:05"

    def test_midnight_12_hour_renders_as_12_am(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 12-hour branch renders midnight as `12:..:.. AM`, not `0:`."""
        monkeypatch.setattr(
            "deepagents_code.formatting.uses_24_hour_clock", lambda: False
        )
        with _utc_timezone():
            # 2024-01-01 00:00:05 UTC — exercises the `hour % 12 or 12` path.
            assert format_message_timestamp(1_704_067_205.0) == "Jan 1, 12:00:05 AM"

    def test_invalid_timestamp_returns_none(self) -> None:
        """An out-of-range timestamp degrades to `None` rather than raising."""
        assert format_message_timestamp(float("inf")) is None


class TestUses24HourClock:
    """Tests for the system 12-/24-hour clock detection."""

    @staticmethod
    def _clear_cache() -> None:
        uses_24_hour_clock.cache_clear()

    def test_macos_force_24_hour_preference_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On macOS the explicit 24-hour preference is honored over locale."""
        import deepagents_code.formatting as fmt

        self._clear_cache()
        monkeypatch.setattr(fmt.sys, "platform", "darwin")
        monkeypatch.setattr(fmt, "macos_force_24_hour_time", lambda: True)
        try:
            assert fmt.uses_24_hour_clock() is True
        finally:
            self._clear_cache()

    def test_macos_force_12_hour_preference_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On macOS an explicit 12-hour preference overrides the locale."""
        import deepagents_code.formatting as fmt

        self._clear_cache()
        monkeypatch.setattr(fmt.sys, "platform", "darwin")
        monkeypatch.setattr(fmt, "macos_force_24_hour_time", lambda: False)
        try:
            assert fmt.uses_24_hour_clock() is False
        finally:
            self._clear_cache()

    def test_macos_unset_defaults_to_24_hour(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On macOS an unset preference defaults to 24h without probing locale."""
        import deepagents_code.formatting as fmt

        self._clear_cache()
        probed = False

        def _spy_setlocale(*_args: object, **_kwargs: object) -> str:
            nonlocal probed
            probed = True
            return "C"

        monkeypatch.setattr(fmt.sys, "platform", "darwin")
        monkeypatch.setattr(fmt, "macos_force_24_hour_time", lambda: None)
        monkeypatch.setattr(fmt.locale, "setlocale", _spy_setlocale)
        try:
            assert fmt.uses_24_hour_clock() is True
            # The locale probe is meaningless on macOS, so it must be skipped.
            assert probed is False
        finally:
            self._clear_cache()

    def test_locale_failure_defaults_to_24_hour(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unresolvable locale falls back to a 24-hour clock."""
        import deepagents_code.formatting as fmt

        self._clear_cache()
        # Force the locale-probe path (non-macOS, or macOS preference unset).
        monkeypatch.setattr(fmt.sys, "platform", "linux")

        def _raise(*_args: object, **_kwargs: object) -> None:
            raise locale.Error

        monkeypatch.setattr(fmt.locale, "setlocale", _raise)
        try:
            assert fmt.uses_24_hour_clock() is True
        finally:
            self._clear_cache()
