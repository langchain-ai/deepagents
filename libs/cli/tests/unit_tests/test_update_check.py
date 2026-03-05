"""Tests for the background update check module."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from deepagents_cli.update_check import (
    CACHE_TTL,
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


def _mock_pypi_response(version: str = "99.0.0") -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"info": {"version": version}}
    resp.raise_for_status = MagicMock()
    return resp


class TestParseVersion:
    def test_basic(self) -> None:
        assert _parse_version("1.2.3") == (1, 2, 3)

    def test_single_digit(self) -> None:
        assert _parse_version("0") == (0,)

    def test_whitespace(self) -> None:
        assert _parse_version("  1.0.0  ") == (1, 0, 0)

    def test_prerelease_raises(self) -> None:
        """Pre-release suffixes like rc1 are not parseable."""
        with pytest.raises(ValueError, match="invalid literal"):
            _parse_version("1.2.3rc1")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid literal"):
            _parse_version("")


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

    def test_cached_hit(self, cache_file) -> None:
        """Fresh cache returns version without HTTP call."""
        cache_file.write_text(
            json.dumps({"version": "1.5.0", "checked_at": time.time()})
        )
        with patch("requests.get") as mock_get:
            result = get_latest_version()

        assert result == "1.5.0"
        mock_get.assert_not_called()

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

    def test_unparseable_pypi_version(self) -> None:
        """Malformed PyPI version string does not crash."""
        with patch(
            "deepagents_cli.update_check.get_latest_version",
            return_value="1.2.3rc1",
        ):
            available, latest = is_update_available()

        assert available is False
        assert latest is None
