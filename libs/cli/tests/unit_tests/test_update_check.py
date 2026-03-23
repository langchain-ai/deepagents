"""Tests for the background update check module."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from packaging.version import InvalidVersion, Version

from deepagents_cli.update_check import (
    CACHE_TTL,
    _latest_from_releases,
    _parse_version,
    get_latest_version,
    is_update_available,
)


@pytest.fixture
def cache_file(tmp_path):
    """Override CACHE_FILE to use a temporary directory."""
    path = tmp_path / "latest_version.json"
    with patch("deepagents_cli.update_check.CACHE_FILE", path):
        yield path


def _mock_pypi_response(
    version: str = "99.0.0",
    releases: dict[str, list[object]] | None = None,
) -> MagicMock:
    if releases is None:
        releases = {version: [{"filename": "fake.tar.gz"}]}
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
        with (
            patch(
                "deepagents_cli.update_check.get_latest_version", return_value="0.0.1"
            ),
            patch("deepagents_cli.update_check.__version__", "0.0.1"),
        ):
            available, latest = is_update_available()

        assert available is False
        assert latest is None

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
        assert latest is None

    def test_fetch_failure(self) -> None:
        with patch("deepagents_cli.update_check.get_latest_version", return_value=None):
            available, latest = is_update_available()

        assert available is False
        assert latest is None

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
        """Stable user is not nudged to a pre-release."""
        with (
            patch(
                "deepagents_cli.update_check.get_latest_version",
                return_value="1.0.0",
            ),
            patch("deepagents_cli.update_check.__version__", "1.0.0"),
        ):
            available, latest = is_update_available()

        assert available is False
        assert latest is None

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
