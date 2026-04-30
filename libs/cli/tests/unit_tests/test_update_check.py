"""Tests for the background update check module."""

from __future__ import annotations

import json
import time
import tomllib
from unittest.mock import MagicMock, patch

import pytest
from packaging.version import InvalidVersion, Version

from deepagents_cli.update_check import (
    CACHE_TTL,
    _extract_release_times,
    _latest_from_releases,
    _parse_version,
    clear_update_notified,
    format_age_suffix,
    format_release_age,
    format_sdk_age_suffix,
    format_sdk_release_age,
    get_latest_version,
    get_release_time,
    get_sdk_release_time,
    get_seen_version,
    is_auto_update_enabled,
    is_update_available,
    mark_update_notified,
    mark_version_seen,
    set_auto_update,
    should_notify_update,
)


@pytest.fixture
def cache_file(tmp_path):
    """Override CACHE_FILE to use a temporary directory."""
    path = tmp_path / "latest_version.json"
    with patch("deepagents_cli.update_check.CACHE_FILE", path):
        yield path


def _mock_pypi_response(
    version: str = "99.0.0",
    releases: dict[str, list[dict[str, object]]] | None = None,
    release_times: dict[str, str] | None = None,
) -> MagicMock:
    if releases is None:
        releases = {version: [{"filename": "fake.tar.gz"}]}
    release_times = release_times or {}
    # Stamp upload_time_iso_8601 onto the first file of each release so the
    # real extraction path runs in tests.
    for ver, iso in release_times.items():
        files = releases.get(ver)
        if files:
            files[0]["upload_time_iso_8601"] = iso
    resp = MagicMock()
    resp.json.return_value = {
        "info": {"version": version},
        "releases": releases,
    }
    resp.raise_for_status = MagicMock()
    return resp


class TestParseVersion:
    def test_basic(self) -> None:
        assert _parse_version("1.2.3") == Version("1.2.3")

    def test_single_digit(self) -> None:
        assert _parse_version("0") == Version("0")

    def test_whitespace(self) -> None:
        assert _parse_version("  1.0.0  ") == Version("1.0.0")

    def test_prerelease(self) -> None:
        result = _parse_version("1.2.3rc1")
        assert result == Version("1.2.3rc1")
        assert result.is_prerelease

    def test_alpha(self) -> None:
        result = _parse_version("1.2.3a1")
        assert result == Version("1.2.3a1")
        assert result.is_prerelease

    def test_empty_raises(self) -> None:
        with pytest.raises(InvalidVersion):
            _parse_version("")

    def test_ordering(self) -> None:
        assert _parse_version("1.0.0a1") < _parse_version("1.0.0a2")
        assert _parse_version("1.0.0a2") < _parse_version("1.0.0b1")
        assert _parse_version("1.0.0b1") < _parse_version("1.0.0rc1")
        assert _parse_version("1.0.0rc1") < _parse_version("1.0.0")


class TestLatestFromReleases:
    def test_stable_only(self) -> None:
        releases = {
            "1.0.0": [{"filename": "a.tar.gz"}],
            "1.1.0a1": [{"filename": "b.tar.gz"}],
            "0.9.0": [{"filename": "c.tar.gz"}],
        }
        assert _latest_from_releases(releases, include_prereleases=False) == "1.0.0"

    def test_include_prereleases(self) -> None:
        releases = {
            "1.0.0": [{"filename": "a.tar.gz"}],
            "1.1.0a1": [{"filename": "b.tar.gz"}],
        }
        assert _latest_from_releases(releases, include_prereleases=True) == "1.1.0a1"

    def test_skips_empty_releases(self) -> None:
        releases = {
            "2.0.0": [],
            "1.0.0": [{"filename": "a.tar.gz"}],
        }
        assert _latest_from_releases(releases, include_prereleases=False) == "1.0.0"

    def test_skips_invalid_versions(self) -> None:
        releases = {
            "not-a-version": [{"filename": "a.tar.gz"}],
            "1.0.0": [{"filename": "b.tar.gz"}],
        }
        assert _latest_from_releases(releases, include_prereleases=False) == "1.0.0"

    def test_empty_releases(self) -> None:
        assert _latest_from_releases({}, include_prereleases=False) is None

    def test_no_stable_releases(self) -> None:
        releases = {
            "1.0.0a1": [{"filename": "a.tar.gz"}],
            "1.0.0b1": [{"filename": "b.tar.gz"}],
        }
        assert _latest_from_releases(releases, include_prereleases=False) is None
        assert _latest_from_releases(releases, include_prereleases=True) == "1.0.0b1"


class TestGetLatestVersion:
    def test_fresh_fetch(self, cache_file) -> None:
        """Successful PyPI fetch writes cache and returns version."""
        with patch("requests.get", return_value=_mock_pypi_response("2.0.0")):
            result = get_latest_version()

        assert result == "2.0.0"
        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert data["version"] == "2.0.0"
        assert "checked_at" in data

    def test_fresh_fetch_prerelease(self, cache_file) -> None:
        """PyPI fetch with include_prereleases returns pre-release version."""
        releases = {
            "2.0.0": [{"filename": "a.tar.gz"}],
            "2.1.0a1": [{"filename": "b.tar.gz"}],
        }
        with patch(
            "requests.get",
            return_value=_mock_pypi_response("2.0.0", releases=releases),
        ):
            result = get_latest_version(include_prereleases=True)

        assert result == "2.1.0a1"
        data = json.loads(cache_file.read_text())
        assert data["version"] == "2.0.0"
        assert data["version_prerelease"] == "2.1.0a1"

    def test_cached_hit(self, cache_file) -> None:
        """Fresh cache returns version without HTTP call."""
        cache_file.write_text(
            json.dumps({"version": "1.5.0", "checked_at": time.time()})
        )
        with patch("requests.get") as mock_get:
            result = get_latest_version()

        assert result == "1.5.0"
        mock_get.assert_not_called()

    def test_cached_hit_prerelease(self, cache_file) -> None:
        """Fresh cache returns pre-release version without HTTP call."""
        cache_file.write_text(
            json.dumps(
                {
                    "version": "1.5.0",
                    "version_prerelease": "1.6.0a1",
                    "checked_at": time.time(),
                }
            )
        )
        with patch("requests.get") as mock_get:
            result = get_latest_version(include_prereleases=True)

        assert result == "1.6.0a1"
        mock_get.assert_not_called()

    def test_cached_null_prerelease_is_cache_hit(self, cache_file) -> None:
        """Cache with null prerelease returns None without hitting PyPI."""
        cache_file.write_text(
            json.dumps(
                {
                    "version": "1.5.0",
                    "version_prerelease": None,
                    "checked_at": time.time(),
                }
            )
        )
        with patch("requests.get") as mock_get:
            result = get_latest_version(include_prereleases=True)

        assert result is None
        mock_get.assert_not_called()

    def test_cached_missing_prerelease_key_triggers_fetch(self, cache_file) -> None:
        """Cache without pre-release key triggers PyPI fetch."""
        cache_file.write_text(
            json.dumps({"version": "1.5.0", "checked_at": time.time()})
        )
        releases = {
            "1.5.0": [{"filename": "a.tar.gz"}],
            "1.6.0a1": [{"filename": "b.tar.gz"}],
        }
        with patch(
            "requests.get",
            return_value=_mock_pypi_response("1.5.0", releases=releases),
        ):
            result = get_latest_version(include_prereleases=True)

        assert result == "1.6.0a1"

    def test_stale_cache(self, cache_file) -> None:
        """Expired cache triggers a new HTTP call."""
        cache_file.write_text(
            json.dumps(
                {
                    "version": "1.0.0",
                    "checked_at": time.time() - CACHE_TTL - 1,
                }
            )
        )
        with patch(
            "requests.get", return_value=_mock_pypi_response("2.0.0")
        ) as mock_get:
            result = get_latest_version()

        assert result == "2.0.0"
        mock_get.assert_called_once()

    def test_network_error(self, cache_file) -> None:  # noqa: ARG002  # fixture overrides CACHE_FILE
        """Network failure returns None."""
        with patch("requests.get", side_effect=OSError("no network")):
            result = get_latest_version()

        assert result is None

    def test_corrupt_cache(self, cache_file) -> None:
        """Malformed cache JSON triggers PyPI fetch instead of crashing."""
        cache_file.write_text("not valid json")
        with patch("requests.get", return_value=_mock_pypi_response("3.0.0")):
            result = get_latest_version()

        assert result == "3.0.0"

    def test_cache_missing_version_key(self, cache_file) -> None:
        """Cache with missing version key triggers PyPI fetch."""
        cache_file.write_text(json.dumps({"checked_at": time.time()}))
        with patch("requests.get", return_value=_mock_pypi_response("3.0.0")):
            result = get_latest_version()

        assert result == "3.0.0"


class TestIsUpdateAvailable:
    def test_newer_available(self) -> None:
        with patch(
            "deepagents_cli.update_check.get_latest_version", return_value="99.0.0"
        ):
            available, latest = is_update_available()

        assert available is True
        assert latest == "99.0.0"

    def test_current_version(self) -> None:
        """User on the latest version sees `available=False` but keeps `latest`.

        The version string is preserved so callers can distinguish "up to date"
        from "PyPI unreachable" (which returns `latest=None`).
        """
        with (
            patch(
                "deepagents_cli.update_check.get_latest_version", return_value="0.0.1"
            ),
            patch("deepagents_cli.update_check.__version__", "0.0.1"),
        ):
            available, latest = is_update_available()

        assert available is False
        assert latest == "0.0.1"

    def test_ahead_of_pypi(self) -> None:
        """Dev build ahead of PyPI should not flag an update."""
        with (
            patch(
                "deepagents_cli.update_check.get_latest_version", return_value="0.0.1"
            ),
            patch("deepagents_cli.update_check.__version__", "99.0.0"),
        ):
            available, latest = is_update_available()

        assert available is False
        assert latest == "0.0.1"

    def test_fetch_failure(self) -> None:
        with patch("deepagents_cli.update_check.get_latest_version", return_value=None):
            available, latest = is_update_available()

        assert available is False
        assert latest is None

    def test_up_to_date_distinguishable_from_fetch_failure(self) -> None:
        """Callers must distinguish `None` (fetch failed) from a version string.

        An up-to-date install returns `(False, "1.2.3")` and a PyPI fetch
        failure returns `(False, None)`; collapsing the two would conflate
        transient network errors with being on the latest release.
        """
        with (
            patch(
                "deepagents_cli.update_check.get_latest_version", return_value="1.2.3"
            ),
            patch("deepagents_cli.update_check.__version__", "1.2.3"),
        ):
            up_to_date = is_update_available()

        with patch("deepagents_cli.update_check.get_latest_version", return_value=None):
            fetch_failed = is_update_available()

        assert up_to_date == (False, "1.2.3")
        assert fetch_failed == (False, None)

    def test_prerelease_user_sees_newer_prerelease(self) -> None:
        """User on alpha sees a newer alpha as available."""
        with (
            patch(
                "deepagents_cli.update_check.get_latest_version",
                return_value="1.0.0a2",
            ),
            patch("deepagents_cli.update_check.__version__", "1.0.0a1"),
        ):
            available, latest = is_update_available()

        assert available is True
        assert latest == "1.0.0a2"

    def test_prerelease_user_sees_stable_release(self) -> None:
        """User on alpha sees the stable release as available."""
        with (
            patch(
                "deepagents_cli.update_check.get_latest_version",
                return_value="1.0.0",
            ),
            patch("deepagents_cli.update_check.__version__", "1.0.0a1"),
        ):
            available, latest = is_update_available()

        assert available is True
        assert latest == "1.0.0"

    def test_stable_user_does_not_see_prerelease(self) -> None:
        """Stable user on current version sees no update available."""
        with (
            patch(
                "deepagents_cli.update_check.get_latest_version",
                return_value="1.0.0",
            ),
            patch("deepagents_cli.update_check.__version__", "1.0.0"),
        ):
            available, latest = is_update_available()

        assert available is False
        assert latest == "1.0.0"

    def test_include_prereleases_kwarg_passed(self) -> None:
        """Verify include_prereleases is True when installed version is pre-release."""
        with (
            patch(
                "deepagents_cli.update_check.get_latest_version",
                return_value=None,
            ) as mock_get,
            patch("deepagents_cli.update_check.__version__", "1.0.0a1"),
        ):
            is_update_available()

        mock_get.assert_called_once_with(bypass_cache=False, include_prereleases=True)

    def test_include_prereleases_false_for_stable(self) -> None:
        """Verify include_prereleases is False when installed version is stable."""
        with (
            patch(
                "deepagents_cli.update_check.get_latest_version",
                return_value=None,
            ) as mock_get,
            patch("deepagents_cli.update_check.__version__", "1.0.0"),
        ):
            is_update_available()

        mock_get.assert_called_once_with(bypass_cache=False, include_prereleases=False)

    def test_invalid_installed_version(self) -> None:
        """Non-PEP 440 installed version disables update check gracefully."""
        with patch("deepagents_cli.update_check.__version__", "not-a-version"):
            available, latest = is_update_available()

        assert available is False
        assert latest is None

    def test_unparseable_pypi_version(self) -> None:
        """Malformed PyPI version string does not crash."""
        with (
            patch(
                "deepagents_cli.update_check.get_latest_version",
                return_value="not-a-version",
            ),
            patch("deepagents_cli.update_check.__version__", "1.0.0"),
        ):
            available, latest = is_update_available()

        assert available is False
        assert latest is None


class TestExtractReleaseTimes:
    def test_stable_only(self) -> None:
        payload = {
            "releases": {
                "1.0.0": [
                    {
                        "filename": "a.tar.gz",
                        "upload_time_iso_8601": "2026-04-15T12:00:00Z",
                    }
                ],
            },
        }
        times = _extract_release_times(payload, stable="1.0.0", prerelease=None)
        assert times == {"1.0.0": "2026-04-15T12:00:00Z"}

    def test_stable_and_prerelease(self) -> None:
        payload = {
            "releases": {
                "1.0.0": [
                    {
                        "filename": "a.tar.gz",
                        "upload_time_iso_8601": "2026-04-15T12:00:00Z",
                    }
                ],
                "1.1.0a1": [
                    {
                        "filename": "b.tar.gz",
                        "upload_time_iso_8601": "2026-04-18T09:30:00Z",
                    }
                ],
            },
        }
        times = _extract_release_times(payload, stable="1.0.0", prerelease="1.1.0a1")
        assert times == {
            "1.0.0": "2026-04-15T12:00:00Z",
            "1.1.0a1": "2026-04-18T09:30:00Z",
        }

    def test_releases_key_absent(self) -> None:
        """Payload with no `releases` key yields an empty result."""
        payload: dict[str, object] = {}
        assert _extract_release_times(payload, stable="1.0.0", prerelease=None) == {}

    def test_non_dict_releases_skipped(self) -> None:
        """A non-dict `releases` value is ignored rather than crashing."""
        payload: dict[str, object] = {"releases": []}
        times = _extract_release_times(payload, stable="1.0.0", prerelease="1.1.0a1")
        assert times == {}

    def test_missing_release_entry(self) -> None:
        """A version with no release entry is silently dropped."""
        payload = {
            "releases": {
                "1.0.0": [
                    {
                        "filename": "a.tar.gz",
                        "upload_time_iso_8601": "2026-04-15T12:00:00Z",
                    }
                ],
            },
        }
        times = _extract_release_times(payload, stable="1.0.0", prerelease="1.1.0a1")
        assert times == {"1.0.0": "2026-04-15T12:00:00Z"}

    def test_stable_lookup_independent_of_info_version(self) -> None:
        """Stable timestamp is read from `releases[stable]`, not `info.version`.

        Regression guard: an earlier implementation used `payload["urls"][0]`,
        which reflects the project's `info.version` and could diverge from
        the requested `stable` when the newest release on PyPI is a
        pre-release.
        """
        payload = {
            "info": {"version": "1.1.0a1"},
            "urls": [{"upload_time_iso_8601": "2026-04-20T00:00:00Z"}],
            "releases": {
                "1.0.0": [
                    {
                        "filename": "a.tar.gz",
                        "upload_time_iso_8601": "2026-04-15T12:00:00Z",
                    }
                ],
                "1.1.0a1": [
                    {
                        "filename": "b.tar.gz",
                        "upload_time_iso_8601": "2026-04-20T00:00:00Z",
                    }
                ],
            },
        }
        times = _extract_release_times(payload, stable="1.0.0", prerelease=None)
        assert times == {"1.0.0": "2026-04-15T12:00:00Z"}

    def test_malformed_entries_skipped(self) -> None:
        payload = {
            "releases": {
                "1.0.0": [{"filename": "no-timestamp"}],
                "1.1.0a1": [{"upload_time_iso_8601": 12345}],
            },
        }
        assert (
            _extract_release_times(payload, stable="1.0.0", prerelease="1.1.0a1") == {}
        )


class TestGetReleaseTime:
    def test_reads_cached_time(self, cache_file) -> None:
        cache_file.write_text(
            json.dumps(
                {
                    "version": "1.0.0",
                    "release_times": {"1.0.0": "2026-04-15T12:00:00Z"},
                    "checked_at": time.time(),
                }
            )
        )
        assert get_release_time("1.0.0") == "2026-04-15T12:00:00Z"

    def test_unknown_version(self, cache_file) -> None:
        cache_file.write_text(
            json.dumps(
                {
                    "release_times": {"1.0.0": "2026-04-15T12:00:00Z"},
                    "checked_at": time.time(),
                }
            )
        )
        assert get_release_time("9.9.9") is None

    def test_missing_cache(self, cache_file) -> None:  # noqa: ARG002
        """No cache file yet → no known release time."""
        assert get_release_time("1.0.0") is None

    def test_cache_without_release_times_key(self, cache_file) -> None:
        """Cache entry lacking the `release_times` field returns `None`."""
        cache_file.write_text(
            json.dumps({"version": "1.0.0", "checked_at": time.time()})
        )
        assert get_release_time("1.0.0") is None

    def test_release_times_is_list_not_dict(self, cache_file) -> None:
        """A list-shaped `release_times` (wrong type) degrades to `None`."""
        cache_file.write_text(
            json.dumps(
                {
                    "release_times": ["1.0.0", "2026-04-15T12:00:00Z"],
                    "checked_at": time.time(),
                }
            )
        )
        assert get_release_time("1.0.0") is None

    def test_corrupted_cache_json(self, cache_file) -> None:
        """Unparseable cache contents return `None` without raising."""
        cache_file.write_text("{not valid json")
        assert get_release_time("1.0.0") is None

    def test_none_version_returns_none(self, cache_file) -> None:  # noqa: ARG002
        """A `None` input short-circuits without touching the cache."""
        assert get_release_time(None) is None


class TestFormatReleaseAge:
    def test_returns_released_prefix(self, cache_file) -> None:
        from datetime import UTC, datetime, timedelta

        iso = (datetime.now(tz=UTC) - timedelta(days=3)).isoformat()
        cache_file.write_text(
            json.dumps(
                {
                    "release_times": {"1.0.0": iso},
                    "checked_at": time.time(),
                }
            )
        )
        age = format_release_age("1.0.0")
        assert age.startswith("released ")
        assert age.endswith("ago")

    def test_unknown_version_returns_empty(self, cache_file) -> None:  # noqa: ARG002
        assert format_release_age("1.0.0") == ""

    def test_empty_relative_timestamp_returns_empty(self, cache_file) -> None:
        """When the relative-timestamp helper returns `""`, the wrapper does too."""
        cache_file.write_text(
            json.dumps(
                {
                    "release_times": {"1.0.0": "not-a-timestamp"},
                    "checked_at": time.time(),
                }
            )
        )
        with patch(
            "deepagents_cli.sessions.format_relative_timestamp", return_value=""
        ):
            assert format_release_age("1.0.0") == ""


class TestFormatAgeSuffix:
    def test_returns_separator_prefixed_age(self, cache_file) -> None:
        """Known age is prefixed with `", "` for splicing into parentheticals."""
        cache_file.write_text(
            json.dumps(
                {
                    "release_times": {"1.0.0": "2026-04-15T12:00:00Z"},
                    "checked_at": time.time(),
                }
            )
        )
        with patch(
            "deepagents_cli.sessions.format_relative_timestamp", return_value="3d ago"
        ):
            assert format_age_suffix("1.0.0") == ", released 3d ago"

    def test_unknown_age_returns_empty(self, cache_file) -> None:  # noqa: ARG002
        """Unknown age collapses to `""` so callers can concat unconditionally."""
        assert format_age_suffix("1.0.0") == ""

    def test_none_version_returns_empty(self, cache_file) -> None:  # noqa: ARG002
        assert format_age_suffix(None) == ""


def _mock_sdk_pypi_response(
    releases: dict[str, list[dict[str, object]]] | None = None,
) -> MagicMock:
    """Build a minimal PyPI response for the `deepagents` SDK.

    The SDK lookup reads from the `releases` map (keyed by version) rather
    than `info.version`, so only that field is required.
    """
    resp = MagicMock()
    resp.json.return_value = {"releases": releases or {}}
    resp.raise_for_status = MagicMock()
    return resp


class TestGetSdkReleaseTime:
    def test_returns_none_for_none_version(self, cache_file) -> None:  # noqa: ARG002
        assert get_sdk_release_time(None) is None

    def test_reads_from_cache(self, cache_file) -> None:
        """A cached SDK release time short-circuits the PyPI fetch."""
        cache_file.write_text(
            json.dumps(
                {
                    "sdk_release_times": {"0.5.0": "2026-04-01T12:00:00Z"},
                }
            )
        )
        with patch("requests.get") as mock_get:
            assert get_sdk_release_time("0.5.0") == "2026-04-01T12:00:00Z"
            mock_get.assert_not_called()

    def test_fetches_on_cache_miss(self, cache_file) -> None:
        """On cache miss the function hits PyPI and writes the result back."""
        iso = "2026-04-10T09:30:00Z"
        releases = {"0.5.0": [{"upload_time_iso_8601": iso}]}
        with patch(
            "requests.get", return_value=_mock_sdk_pypi_response(releases=releases)
        ):
            assert get_sdk_release_time("0.5.0") == iso

        data = json.loads(cache_file.read_text())
        assert data["sdk_release_times"] == {"0.5.0": iso}

    def test_unknown_version_returns_none(self, cache_file) -> None:  # noqa: ARG002
        """A version PyPI doesn't know about yields `None` without raising."""
        with patch(
            "requests.get",
            return_value=_mock_sdk_pypi_response(
                releases={"0.4.0": [{"upload_time_iso_8601": "2026-01-01T00:00:00Z"}]}
            ),
        ):
            assert get_sdk_release_time("9.9.9") is None

    def test_network_error_returns_none(self, cache_file) -> None:  # noqa: ARG002
        """A `requests` failure degrades to `None` without raising."""
        import requests

        with patch("requests.get", side_effect=requests.ConnectionError("boom")):
            assert get_sdk_release_time("0.5.0") is None

    def test_bypass_cache_refetches(self, cache_file) -> None:
        """`bypass_cache=True` ignores the cached value and hits PyPI."""
        cache_file.write_text(
            json.dumps(
                {
                    "sdk_release_times": {"0.5.0": "2026-01-01T00:00:00Z"},
                }
            )
        )
        fresh = "2026-04-15T12:00:00Z"
        with patch(
            "requests.get",
            return_value=_mock_sdk_pypi_response(
                releases={"0.5.0": [{"upload_time_iso_8601": fresh}]}
            ),
        ):
            assert get_sdk_release_time("0.5.0", bypass_cache=True) == fresh

        data = json.loads(cache_file.read_text())
        assert data["sdk_release_times"]["0.5.0"] == fresh

    def test_preserves_existing_sdk_entries(self, cache_file) -> None:
        """Writing a new SDK timestamp leaves other cached versions intact."""
        cache_file.write_text(
            json.dumps(
                {
                    "sdk_release_times": {"0.4.0": "2026-01-01T00:00:00Z"},
                }
            )
        )
        iso = "2026-04-10T09:30:00Z"
        with patch(
            "requests.get",
            return_value=_mock_sdk_pypi_response(
                releases={"0.5.0": [{"upload_time_iso_8601": iso}]}
            ),
        ):
            assert get_sdk_release_time("0.5.0") == iso

        data = json.loads(cache_file.read_text())
        assert data["sdk_release_times"] == {
            "0.4.0": "2026-01-01T00:00:00Z",
            "0.5.0": iso,
        }

    def test_overwrites_corrupt_cache(self, cache_file) -> None:
        """A corrupt cache JSON must be overwritten, not preserved.

        Regression guard: an earlier implementation skipped the write when
        decoding the existing cache raised, so every call paid the PyPI
        round-trip until the file was deleted by hand.
        """
        cache_file.write_text("{not valid json")
        iso = "2026-04-10T09:30:00Z"
        with patch(
            "requests.get",
            return_value=_mock_sdk_pypi_response(
                releases={"0.5.0": [{"upload_time_iso_8601": iso}]}
            ),
        ):
            assert get_sdk_release_time("0.5.0") == iso

        data = json.loads(cache_file.read_text())
        assert data["sdk_release_times"] == {"0.5.0": iso}


class TestFormatSdkReleaseAge:
    def test_returns_released_prefix(self, cache_file) -> None:
        from datetime import UTC, datetime, timedelta

        iso = (datetime.now(tz=UTC) - timedelta(days=2)).isoformat()
        cache_file.write_text(json.dumps({"sdk_release_times": {"0.5.0": iso}}))
        age = format_sdk_release_age("0.5.0")
        assert age.startswith("released ")
        assert age.endswith("ago")

    def test_unknown_version_with_no_network_returns_empty(self, cache_file) -> None:  # noqa: ARG002
        """Cache miss + PyPI failure collapses to `""` (no exception)."""
        import requests

        with patch("requests.get", side_effect=requests.ConnectionError("boom")):
            assert format_sdk_release_age("0.5.0") == ""


class TestFormatSdkAgeSuffix:
    def test_returns_separator_prefixed_age(self, cache_file) -> None:
        cache_file.write_text(
            json.dumps({"sdk_release_times": {"0.5.0": "2026-04-10T12:00:00Z"}})
        )
        with patch(
            "deepagents_cli.sessions.format_relative_timestamp", return_value="1w ago"
        ):
            assert format_sdk_age_suffix("0.5.0") == ", released 1w ago"

    def test_none_version_returns_empty(self, cache_file) -> None:  # noqa: ARG002
        assert format_sdk_age_suffix(None) == ""


class TestGetLatestVersionReleaseTimes:
    def test_release_times_cached_on_fresh_fetch(self, cache_file) -> None:
        """A fresh PyPI fetch captures stable upload_time_iso_8601 into the cache."""
        with patch(
            "requests.get",
            return_value=_mock_pypi_response(
                "2.0.0",
                release_times={"2.0.0": "2026-04-15T12:00:00Z"},
            ),
        ):
            get_latest_version()

        data = json.loads(cache_file.read_text())
        assert data["release_times"] == {"2.0.0": "2026-04-15T12:00:00Z"}

    def test_release_times_cached_for_prerelease(self, cache_file) -> None:
        """Prerelease fetch captures both stable and prerelease timestamps."""
        releases = {
            "2.0.0": [{"filename": "a.tar.gz"}],
            "2.1.0a1": [{"filename": "b.tar.gz"}],
        }
        with patch(
            "requests.get",
            return_value=_mock_pypi_response(
                "2.0.0",
                releases=releases,
                release_times={
                    "2.0.0": "2026-04-15T12:00:00Z",
                    "2.1.0a1": "2026-04-18T09:30:00Z",
                },
            ),
        ):
            get_latest_version(include_prereleases=True)

        data = json.loads(cache_file.read_text())
        assert data["release_times"] == {
            "2.0.0": "2026-04-15T12:00:00Z",
            "2.1.0a1": "2026-04-18T09:30:00Z",
        }


class TestSetAutoUpdate:
    @pytest.fixture
    def config_path(self, tmp_path):
        """Override DEFAULT_CONFIG_PATH to use a temporary file."""
        path = tmp_path / "config.toml"
        with patch("deepagents_cli.update_check.DEFAULT_CONFIG_PATH", path):
            yield path

    def test_enable_creates_config(self, config_path) -> None:
        """Creates config.toml with auto_update = true when file doesn't exist."""
        set_auto_update(True)
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        assert data["update"]["auto_update"] is True

    def test_disable(self, config_path) -> None:
        """Sets auto_update = false."""
        set_auto_update(True)
        set_auto_update(False)
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        assert data["update"]["auto_update"] is False

    def test_preserves_existing_config(self, config_path) -> None:
        """Doesn't clobber unrelated config sections."""
        import tomli_w

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("wb") as f:
            tomli_w.dump({"ui": {"theme": "monokai"}}, f)

        set_auto_update(True)
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        assert data["ui"]["theme"] == "monokai"
        assert data["update"]["auto_update"] is True

    def test_preserves_sibling_update_keys(self, config_path) -> None:
        """Doesn't clobber sibling keys in [update] section."""
        import tomli_w

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("wb") as f:
            tomli_w.dump({"update": {"check": False}}, f)

        set_auto_update(True)
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        assert data["update"]["check"] is False
        assert data["update"]["auto_update"] is True

    def test_round_trip_with_is_auto_update_enabled(self, config_path) -> None:  # noqa: ARG002
        """set_auto_update(True) makes is_auto_update_enabled() return True."""
        set_auto_update(True)
        with (
            patch("deepagents_cli.config._is_editable_install", return_value=False),
            patch.dict("os.environ", {}, clear=False),
        ):
            import os

            os.environ.pop("DEEPAGENTS_CLI_AUTO_UPDATE", None)
            assert is_auto_update_enabled() is True


class TestIsAutoUpdateEnabled:
    @pytest.fixture
    def config_path(self, tmp_path):
        """Override DEFAULT_CONFIG_PATH to use a temporary file."""
        path = tmp_path / "config.toml"
        with patch("deepagents_cli.update_check.DEFAULT_CONFIG_PATH", path):
            yield path

    def test_default_is_false(self, config_path) -> None:  # noqa: ARG002
        """Auto-update defaults to disabled."""
        with (
            patch("deepagents_cli.config._is_editable_install", return_value=False),
            patch.dict("os.environ", {}, clear=False),
        ):
            import os

            os.environ.pop("DEEPAGENTS_CLI_AUTO_UPDATE", None)
            assert is_auto_update_enabled() is False

    def test_env_var_enables(self, config_path) -> None:  # noqa: ARG002
        """DEEPAGENTS_CLI_AUTO_UPDATE=1 enables auto-update."""
        with (
            patch("deepagents_cli.config._is_editable_install", return_value=False),
            patch.dict("os.environ", {"DEEPAGENTS_CLI_AUTO_UPDATE": "1"}),
        ):
            assert is_auto_update_enabled() is True

    def test_editable_install_always_disabled(self, config_path) -> None:
        """Editable installs never auto-update, even with config set."""
        set_auto_update(True)
        assert config_path.exists()
        with patch("deepagents_cli.config._is_editable_install", return_value=True):
            assert is_auto_update_enabled() is False


class TestShouldNotifyUpdate:
    @pytest.fixture
    def state_file(self, tmp_path):
        """Override UPDATE_STATE_FILE to use a temporary directory."""
        path = tmp_path / "update_state.json"
        with patch("deepagents_cli.update_check.UPDATE_STATE_FILE", path):
            yield path

    def test_no_file_returns_true(self, state_file) -> None:  # noqa: ARG002
        """First-run case: no state file exists."""
        assert should_notify_update("2.0.0") is True

    def test_same_version_within_ttl(self, state_file) -> None:
        """Same version notified recently — suppress."""
        state_file.write_text(
            json.dumps({"notified_at": time.time(), "notified_version": "2.0.0"})
        )
        assert should_notify_update("2.0.0") is False

    def test_different_version_within_ttl(self, state_file) -> None:
        """New version available — notify even within TTL window."""
        state_file.write_text(
            json.dumps({"notified_at": time.time(), "notified_version": "1.9.0"})
        )
        assert should_notify_update("2.0.0") is True

    def test_same_version_ttl_expired(self, state_file) -> None:
        """TTL expired — re-notify for same version."""
        state_file.write_text(
            json.dumps(
                {
                    "notified_at": time.time() - CACHE_TTL - 1,
                    "notified_version": "2.0.0",
                }
            )
        )
        assert should_notify_update("2.0.0") is True

    def test_corrupt_json(self, state_file) -> None:
        """Malformed JSON — fail-open (show banner)."""
        state_file.write_text("not valid json")
        assert should_notify_update("2.0.0") is True

    def test_non_dict_json(self, state_file) -> None:
        """JSON array instead of object — fail-open."""
        state_file.write_text(json.dumps([1, 2, 3]))
        assert should_notify_update("2.0.0") is True

    def test_non_numeric_notified_at(self, state_file) -> None:
        """notified_at is a string — treated as invalid, show banner."""
        state_file.write_text(
            json.dumps({"notified_at": "not-a-number", "notified_version": "2.0.0"})
        )
        assert should_notify_update("2.0.0") is True

    def test_missing_notified_at_key(self, state_file) -> None:
        """File exists but missing notified_at — defaults to 0, TTL expired."""
        state_file.write_text(json.dumps({"notified_version": "2.0.0"}))
        assert should_notify_update("2.0.0") is True


class TestMarkUpdateNotified:
    @pytest.fixture
    def state_file(self, tmp_path):
        """Override UPDATE_STATE_FILE to use a temporary directory."""
        path = tmp_path / "update_state.json"
        with patch("deepagents_cli.update_check.UPDATE_STATE_FILE", path):
            yield path

    def test_creates_file(self, state_file) -> None:
        """Creates state file when none exists."""
        mark_update_notified("2.0.0")
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["notified_version"] == "2.0.0"
        assert isinstance(data["notified_at"], float)

    def test_overwrites_previous(self, state_file) -> None:
        """Overwrites previous notification marker."""
        mark_update_notified("1.0.0")
        mark_update_notified("2.0.0")
        data = json.loads(state_file.read_text())
        assert data["notified_version"] == "2.0.0"

    def test_round_trip(self, state_file) -> None:  # noqa: ARG002
        """Mark then should_notify returns False for same version."""
        mark_update_notified("2.0.0")
        assert should_notify_update("2.0.0") is False

    def test_round_trip_different_version(self, state_file) -> None:  # noqa: ARG002
        """Mark then should_notify returns True for different version."""
        mark_update_notified("1.9.0")
        assert should_notify_update("2.0.0") is True

    def test_clear_makes_should_notify_true_again(
        self,
        state_file,  # noqa: ARG002
    ) -> None:
        """clear_update_notified undoes a previous mark."""
        mark_update_notified("2.0.0")
        assert should_notify_update("2.0.0") is False
        clear_update_notified()
        assert should_notify_update("2.0.0") is True

    def test_clear_removes_marker_keys_from_state(self, state_file) -> None:
        """clear_update_notified pops the keys rather than writing sentinels."""
        mark_update_notified("2.0.0")
        clear_update_notified()
        data = json.loads(state_file.read_text())
        assert "notified_at" not in data
        assert "notified_version" not in data

    def test_clear_preserves_other_state_keys(self, state_file) -> None:
        """Clearing notification markers leaves unrelated keys intact."""
        mark_version_seen("1.0.0")
        mark_update_notified("2.0.0")
        clear_update_notified()
        data = json.loads(state_file.read_text())
        assert data["seen_version"] == "1.0.0"

    def test_write_failure_does_not_raise(self, state_file) -> None:
        """Write failure is absorbed gracefully."""
        with patch(
            "deepagents_cli.update_check.UPDATE_STATE_FILE",
            type(state_file)("/nonexistent/readonly/path/state.json"),
        ):
            mark_update_notified("2.0.0")  # should not raise

    def test_does_not_touch_cache_file(self, state_file, cache_file) -> None:
        """Notification state is independent of version cache."""
        cache_file.write_text(
            json.dumps(
                {
                    "version": "2.0.0",
                    "checked_at": time.time(),
                }
            )
        )
        mark_update_notified("2.0.0")
        # Cache file should be untouched
        cache_data = json.loads(cache_file.read_text())
        assert "notified_at" not in cache_data
        assert "notified_version" not in cache_data
        # State file should have the marker
        assert state_file.exists()
        state_data = json.loads(state_file.read_text())
        assert state_data["notified_version"] == "2.0.0"

    def test_get_latest_version_does_not_clobber_notify(
        self,
        state_file,  # noqa: ARG002
        cache_file,  # noqa: ARG002
    ) -> None:
        """get_latest_version writing cache doesn't destroy notification state."""
        mark_update_notified("2.0.0")
        with patch("requests.get", return_value=_mock_pypi_response("3.0.0")):
            get_latest_version(bypass_cache=True)
        # Notification marker should survive
        assert should_notify_update("2.0.0") is False

    def test_preserves_seen_version(self, state_file) -> None:
        """Marking notification preserves existing seen_version data."""
        mark_version_seen("1.0.0")
        mark_update_notified("2.0.0")
        data = json.loads(state_file.read_text())
        assert data["seen_version"] == "1.0.0"
        assert data["notified_version"] == "2.0.0"


class TestGetSeenVersion:
    @pytest.fixture
    def state_file(self, tmp_path):
        """Override UPDATE_STATE_FILE to use a temporary directory."""
        path = tmp_path / "update_state.json"
        with patch("deepagents_cli.update_check.UPDATE_STATE_FILE", path):
            yield path

    def test_no_file_returns_none(self, state_file) -> None:  # noqa: ARG002
        """No state file -> None."""
        assert get_seen_version() is None

    def test_round_trip(self, state_file) -> None:  # noqa: ARG002
        """Mark then get returns the same version."""
        mark_version_seen("1.0.0")
        assert get_seen_version() == "1.0.0"

    def test_corrupt_json_returns_none(self, state_file) -> None:
        """Corrupt state file -> None."""
        state_file.write_text("not json")
        assert get_seen_version() is None

    def test_non_string_value_returns_none(self, state_file) -> None:
        """Non-string seen_version -> None (type guard)."""
        state_file.write_text(json.dumps({"seen_version": 123}))
        assert get_seen_version() is None

    def test_preserves_notification_keys(self, state_file) -> None:  # noqa: ARG002
        """Marking seen preserves existing notification data."""
        mark_update_notified("2.0.0")
        mark_version_seen("1.0.0")
        assert get_seen_version() == "1.0.0"
        assert should_notify_update("2.0.0") is False


class TestShouldShowWhatsNew:
    @pytest.fixture
    def state_file(self, tmp_path):
        """Override UPDATE_STATE_FILE to use a temporary directory."""
        path = tmp_path / "update_state.json"
        with patch("deepagents_cli.update_check.UPDATE_STATE_FILE", path):
            yield path

    def test_first_run_returns_false_and_marks(self, state_file) -> None:
        """First run: returns False and writes current version as seen."""
        from deepagents_cli.update_check import should_show_whats_new

        with patch("deepagents_cli.update_check.__version__", "1.0.0"):
            assert should_show_whats_new() is False
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["seen_version"] == "1.0.0"

    def test_same_version_returns_false(self, state_file) -> None:  # noqa: ARG002
        """Current version == seen version -> False."""
        from deepagents_cli.update_check import should_show_whats_new

        mark_version_seen("1.0.0")
        with patch("deepagents_cli.update_check.__version__", "1.0.0"):
            assert should_show_whats_new() is False

    def test_newer_version_returns_true(self, state_file) -> None:  # noqa: ARG002
        """Current version > seen version -> True."""
        from deepagents_cli.update_check import should_show_whats_new

        mark_version_seen("1.0.0")
        with patch("deepagents_cli.update_check.__version__", "2.0.0"):
            assert should_show_whats_new() is True

    def test_coexists_with_notification_state(self, state_file) -> None:  # noqa: ARG002
        """What's-new and notification state don't interfere."""
        from deepagents_cli.update_check import should_show_whats_new

        mark_update_notified("2.0.0")
        mark_version_seen("1.0.0")
        with patch("deepagents_cli.update_check.__version__", "2.0.0"):
            assert should_show_whats_new() is True
        # Notification throttle still works
        assert should_notify_update("2.0.0") is False
