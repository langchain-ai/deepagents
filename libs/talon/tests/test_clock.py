from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.clock import current_time, current_time_snapshot
from deepagents_talon.timezones import (
    TimeZoneError,
    _key_from_localtime_target,
    resolve_zone,
    system_timezone_name,
)

if TYPE_CHECKING:
    from pathlib import Path


def _link(tmp_path: Path, target: str) -> Path:
    link = tmp_path / "localtime"
    link.symlink_to(target)
    return link


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("/var/db/timezone/zoneinfo/America/New_York", "America/New_York"),
        ("/usr/share/zoneinfo/Europe/Berlin", "Europe/Berlin"),
        ("../usr/share/zoneinfo/Asia/Kolkata", "Asia/Kolkata"),
        ("/usr/share/zoneinfo/UTC", "UTC"),
        ("/usr/share/zoneinfo/Etc/UTC", "Etc/UTC"),
        ("/some/copy/of/a/tzfile", None),
        ("/usr/share/zoneinfo/", None),
    ],
)
def test_key_from_localtime_target_handles_platform_layouts(target, expected) -> None:
    assert _key_from_localtime_target(target) == expected


def test_system_timezone_prefers_env_over_symlink(tmp_path) -> None:
    link = _link(tmp_path, "/usr/share/zoneinfo/Europe/Berlin")

    name = system_timezone_name(
        env={"TZ": "Asia/Kolkata"},
        localtime_path=link,
        timezone_path=tmp_path / "missing",
    )

    assert name == "Asia/Kolkata"


def test_system_timezone_strips_posix_colon_prefix(tmp_path) -> None:
    name = system_timezone_name(
        env={"TZ": ":America/New_York"},
        localtime_path=tmp_path / "missing",
        timezone_path=tmp_path / "missing",
    )

    assert name == "America/New_York"


def test_unusable_env_value_yields_no_name_rather_than_a_disagreeing_one(tmp_path) -> None:
    # TZ governs the clock astimezone() reports, so naming Europe/Berlin from
    # the symlink here would describe a wall clock the process is not running
    # on. No name is the honest answer; the caller still gets a real offset.
    link = _link(tmp_path, "/usr/share/zoneinfo/Europe/Berlin")

    name = system_timezone_name(
        env={"TZ": "EST5EDT"},
        localtime_path=link,
        timezone_path=tmp_path / "missing",
    )

    assert name is None


def test_present_but_empty_env_value_is_utc(tmp_path) -> None:
    # POSIX: TZ="" means UTC, and that is what astimezone() reports.
    link = _link(tmp_path, "/usr/share/zoneinfo/Europe/Berlin")

    name = system_timezone_name(
        env={"TZ": ""},
        localtime_path=link,
        timezone_path=tmp_path / "missing",
    )

    assert name == "UTC"


def test_undecodable_timezone_file_is_not_fatal(tmp_path) -> None:
    path = tmp_path / "timezone"
    path.write_bytes(b"\xff\xfe not utf-8")

    name = system_timezone_name(
        env={},
        localtime_path=tmp_path / "missing",
        timezone_path=path,
    )

    assert name is None


def test_undecodable_timezone_file_never_read_when_symlink_resolves(tmp_path) -> None:
    # UnicodeDecodeError is a ValueError, so an eagerly-read corrupt file used
    # to blow up detection even when a higher-priority source was fine.
    link = _link(tmp_path, "/usr/share/zoneinfo/Europe/Berlin")
    path = tmp_path / "timezone"
    path.write_bytes(b"\xff\xfe not utf-8")

    name = system_timezone_name(env={}, localtime_path=link, timezone_path=path)

    assert name == "Europe/Berlin"


def test_system_timezone_reads_debian_timezone_file(tmp_path) -> None:
    path = tmp_path / "timezone"
    path.write_text("Australia/Sydney\n", encoding="utf-8")

    name = system_timezone_name(
        env={},
        localtime_path=tmp_path / "missing",
        timezone_path=path,
    )

    assert name == "Australia/Sydney"


def test_system_timezone_is_none_when_nothing_resolves(tmp_path) -> None:
    link = _link(tmp_path, "/some/copy/of/a/tzfile")

    name = system_timezone_name(
        env={},
        localtime_path=link,
        timezone_path=tmp_path / "missing",
    )

    assert name is None


@pytest.mark.parametrize(
    ("zone", "offset"),
    [("Asia/Kolkata", "+05:30"), ("UTC", "+00:00"), ("Etc/UTC", "+00:00")],
)
def test_snapshot_reports_requested_zone_offset(zone, offset) -> None:
    snapshot = current_time_snapshot(zone)

    assert snapshot["timezone"] == zone
    assert snapshot["utc_offset"] == offset
    assert "note" not in snapshot


def test_snapshot_local_and_utc_describe_the_same_instant() -> None:
    snapshot = current_time_snapshot("Asia/Kolkata")

    local = datetime.fromisoformat(snapshot["local"])
    utc = datetime.fromisoformat(snapshot["utc"])

    assert local == utc
    assert snapshot["date"] == local.strftime("%Y-%m-%d")
    assert snapshot["day_of_week"] == local.strftime("%A")


def test_snapshot_without_a_zone_name_still_reports_offset(monkeypatch) -> None:
    monkeypatch.setattr("deepagents_talon.clock.system_timezone_name", lambda: None)

    snapshot = current_time_snapshot()

    assert snapshot["timezone"] is None
    assert "note" in snapshot
    assert datetime.fromisoformat(snapshot["local"]) == datetime.fromisoformat(snapshot["utc"])
    assert snapshot["utc_offset"].startswith(("+", "-"))


def test_snapshot_uses_the_detected_host_zone(monkeypatch) -> None:
    monkeypatch.setattr(
        "deepagents_talon.clock.system_timezone_name",
        lambda: "America/New_York",
    )

    snapshot = current_time_snapshot()

    assert snapshot["timezone"] == "America/New_York"
    assert snapshot["utc_offset"] in {"-05:00", "-04:00"}


@pytest.mark.parametrize(
    "zone",
    ["America/NewYork", "EST5EDT", "EST", "+02:00", "", "/etc/localtime", "../../etc/passwd"],
)
def test_resolve_zone_rejects_unusable_names(zone) -> None:
    with pytest.raises(TimeZoneError):
        resolve_zone(zone)


@pytest.mark.parametrize(
    "zone",
    ["America/NewYork", "EST5EDT", "+02:00", "../../etc/passwd"],
)
def test_tool_returns_an_error_without_leaking_host_details(zone) -> None:
    result = current_time.invoke({"timezone": zone})

    # The rejected input is echoed so the agent can correct itself, but nothing
    # about where the host keeps its tz database is disclosed.
    assert "error" in result
    assert zone in result["error"]
    assert "/etc/localtime" not in result["error"]
    assert "/usr/share" not in result["error"]
    assert "/var/db" not in result["error"]
    assert "Traceback" not in result["error"]


def test_tool_invocation_returns_a_snapshot() -> None:
    result = current_time.invoke({})

    assert "error" not in result
    assert set(result) >= {
        "utc",
        "local",
        "date",
        "time",
        "day_of_week",
        "timezone",
        "utc_offset",
        "abbreviation",
    }
