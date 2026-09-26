from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from deepagents_talon.cron import CronJobError, CronJobStore, CronOrigin, CronSchedule

NEW_YORK = ZoneInfo("America/New_York")


def _runs(text: str, start: datetime, count: int = 3) -> list[str]:
    """Chain `next_after` from `start`, reporting each run as local wall-clock text."""
    schedule = CronSchedule.parse(text)
    zone = ZoneInfo(str(schedule.timezone))
    runs: list[str] = []
    current = start
    for _ in range(count):
        current = schedule.next_after(current)
        runs.append(current.astimezone(zone).strftime("%Y-%m-%d %H:%M"))
    return runs


def _local(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=NEW_YORK)


@pytest.mark.parametrize(
    "text",
    [
        "cron * * * * * UTC",
        "cron */15 9-17 * * mon-fri America/New_York",
        "cron 0,30 8-18/2 1,15 jan-mar,dec 1-5 UTC",
        "cron 5/10 * * * * UTC",
        "cron 0 12 * * SAT,SUN UTC",
        "cron 0 9 * * 7 UTC",
        "cron 0 9 L * * UTC",
        "cron 0 9 LW * * UTC",
        "cron 0 9 15W * * UTC",
        "cron 0 17 * * 5L UTC",
        "cron 0 9 * * 2#1 UTC",
        "cron 0 9 * * fri#3 UTC",
        "cron @daily UTC",
        "cron @HOURLY UTC",
    ],
)
def test_parse_accepts_cron_expressions(text: str) -> None:
    schedule = CronSchedule.parse(text)

    assert schedule.form == "cron"
    assert schedule.kind == "recurring"
    assert CronSchedule.from_dict(schedule.to_dict()) == schedule


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("cron * * * * UTC", "five fields"),
        ("cron 0 * * * * * UTC", "five fields"),
        ("cron * * * * *", "five fields"),
        ("cron 60 * * * * UTC", "minute '60' is outside 0-59"),
        ("cron * 24 * * * UTC", "hour '24' is outside 0-23"),
        ("cron * * 0 * * UTC", "day-of-month '0' is outside 1-31"),
        ("cron * * * 13 * UTC", "month '13' is outside 1-12"),
        ("cron * * * * 8 UTC", "day-of-week '8' is outside 0-7"),
        ("cron */0 * * * * UTC", "step must be at least 1"),
        ("cron 5-1 * * * * UTC", "runs backward"),
        ("cron 1,,2 * * * * UTC", "must be a number"),
        ("cron ? * * * * UTC", "must be a number"),
        ("cron * * * * 6#6 UTC", "occurrence must be 1-5"),
        ("cron @reboot UTC", "unsupported cron macro"),
        ("cron * * * * * Mars/Olympus", "Mars/Olympus"),
    ],
)
def test_parse_rejects_malformed_cron_expressions(text: str, reason: str) -> None:
    with pytest.raises(CronJobError, match=reason):
        CronSchedule.parse(text)


def test_parse_canonicalizes_case_but_keeps_timezone_case() -> None:
    schedule = CronSchedule.parse("CRON 0 12 * * SAT,SUN America/New_York")

    assert schedule.expression == "0 12 * * sat,sun"
    assert schedule.display == "cron 0 12 * * sat,sun America/New_York"


def test_weekdays_in_june_every_fifteen_minutes() -> None:
    start = _local(2026, 9, 23, 12, 0)

    assert _runs("cron */15 * * jun mon-fri America/New_York", start) == [
        "2027-06-01 00:00",
        "2027-06-01 00:15",
        "2027-06-01 00:30",
    ]


def test_weekday_schedule_skips_the_weekend() -> None:
    friday_night = _local(2027, 6, 4, 23, 40)

    assert _runs("cron */15 * * jun mon-fri America/New_York", friday_night) == [
        "2027-06-04 23:45",
        "2027-06-07 00:00",
        "2027-06-07 00:15",
    ]


def test_weekends_at_noon() -> None:
    wednesday = _local(2026, 9, 23, 13, 0)

    assert _runs("cron 0 12 * * sat,sun America/New_York", wednesday) == [
        "2026-09-26 12:00",
        "2026-09-27 12:00",
        "2026-10-03 12:00",
    ]


def test_restricted_day_fields_match_either() -> None:
    # Fridays in October 2026 are the 2nd, 9th, 16th; the 13th is a Tuesday.
    start = datetime(2026, 10, 1, tzinfo=UTC)

    assert _runs("cron 0 0 13 * 5 UTC", start, 4) == [
        "2026-10-02 00:00",
        "2026-10-09 00:00",
        "2026-10-13 00:00",
        "2026-10-16 00:00",
    ]


def test_starred_day_field_requires_both() -> None:
    # `*/2` starts with `*`, so a day must be odd AND a Friday.
    start = datetime(2026, 10, 1, tzinfo=UTC)

    assert _runs("cron 0 0 */2 * 5 UTC", start) == [
        "2026-10-09 00:00",
        "2026-10-23 00:00",
        "2026-11-13 00:00",
    ]


@pytest.mark.parametrize(
    ("text", "start", "expected"),
    [
        ("cron 0 9 L * * UTC", (2026, 2, 1), ["2026-02-28", "2026-03-31", "2026-04-30"]),
        # May 31, 2026 is a Sunday.
        ("cron 0 9 LW * * UTC", (2026, 5, 1), ["2026-05-29", "2026-06-30", "2026-07-31"]),
        # August 1, 2026 is a Saturday; W never crosses back into July.
        ("cron 0 9 1W * * UTC", (2026, 8, 1), ["2026-08-03", "2026-09-01", "2026-10-01"]),
        # October 31 is a Saturday, November has no 31st, December 31 is a Thursday.
        ("cron 0 9 31W * * UTC", (2026, 10, 1), ["2026-10-30", "2026-12-31", "2027-01-29"]),
        ("cron 0 9 * * 5L UTC", (2026, 9, 1), ["2026-09-25", "2026-10-30", "2026-11-27"]),
        ("cron 0 9 * * 2#1 UTC", (2026, 9, 1), ["2026-09-01", "2026-10-06", "2026-11-03"]),
        ("cron 0 9 * * 7 UTC", (2026, 9, 23), ["2026-09-27", "2026-10-04", "2026-10-11"]),
        ("cron @weekly UTC", (2026, 9, 23), ["2026-09-27", "2026-10-04", "2026-10-11"]),
    ],
)
def test_calendar_extensions(text: str, start: tuple[int, int, int], expected: list[str]) -> None:
    runs = _runs(text, datetime(*start, tzinfo=UTC))

    assert [run.split()[0] for run in runs] == expected


def test_leap_day_skips_the_2100_non_leap_year() -> None:
    start = datetime(2097, 1, 1, tzinfo=UTC)

    assert _runs("cron 0 0 29 2 * UTC", start, 1) == ["2104-02-29 00:00"]


def test_rare_match_beyond_a_decade_is_found() -> None:
    # A fifth Friday in February needs a leap year whose February starts on a Friday.
    start = datetime(2093, 1, 1, tzinfo=UTC)

    assert _runs("cron 0 0 * 2 5#5 UTC", start, 1) == ["2104-02-29 00:00"]


def test_impossible_expression_is_rejected_at_create(tmp_path) -> None:
    store = CronJobStore(assistant_id="assistant", cron_dir=tmp_path / "cron")

    with pytest.raises(CronJobError, match="never fires"):
        store.create_job(
            prompt="never",
            schedule=CronSchedule.parse("cron 0 0 31 2 * UTC"),
            origin=CronOrigin(conversation_id="chat"),
            now=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_spring_forward_gap_does_not_fire_twice() -> None:
    # 02:00-03:00 does not exist in New York on 2026-03-08; 02:00 through 02:45
    # all snap to 03:00, which must fire once.
    start = _local(2026, 3, 8, 1, 50)

    assert _runs("cron */15 * * * * America/New_York", start, 3) == [
        "2026-03-08 03:00",
        "2026-03-08 03:15",
        "2026-03-08 03:30",
    ]


def test_fall_back_repeated_time_fires_once() -> None:
    schedule = CronSchedule.parse("cron 30 1 * * * America/New_York")
    first = schedule.next_after(_local(2026, 10, 31, 12, 0))
    second = schedule.next_after(first)

    assert first == datetime(2026, 11, 1, 5, 30, tzinfo=UTC)  # 01:30 EDT
    assert second == datetime(2026, 11, 2, 6, 30, tzinfo=UTC)  # 01:30 EST, next day


def test_cron_holds_local_time_across_daylight_saving() -> None:
    before = _local(2026, 10, 31, 0, 0)

    assert _runs("cron 0 9 * * * America/New_York", before, 3) == [
        "2026-10-31 09:00",
        "2026-11-01 09:00",
        "2026-11-02 09:00",
    ]
