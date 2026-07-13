"""Tests for the background update check module."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import signal
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence  # noqa: TC003
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from packaging.version import InvalidVersion, Version

from deepagents_code._version import __version__
from deepagents_code.extras_info import ExtrasIntrospectionError, installed_extra_names
from deepagents_code.update_check import (
    CACHE_TTL,
    INSTALLED_STALE_NOTICE_DAYS,
    DependencyChange,
    InstallMethod,
    ShadowedDcode,
    ToolRequirementIntrospectionError,
    _extract_release_times,
    _install_extra_uv_tool_command,
    _latest_from_releases,
    _note_install_baseline,
    _parse_version,
    _requires_prerelease_dependency,
    _run_install_subprocess,
    _terminate_install_process,
    _uv_tool_bin_dir,
    cleanup_update_logs,
    clear_resume_auto_update_deferral,
    clear_startup_auto_update_failure,
    clear_update_notified,
    create_update_log_path,
    dependency_refresh_command,
    dependency_refresh_dry_run_command,
    dependency_refresh_supported,
    detect_install_method,
    detect_shadowed_dcode,
    detect_shadowed_dcode_safe,
    editable_extra_hint,
    editable_package_hint,
    format_age_suffix,
    format_dependency_changes,
    format_installed_age_suffix,
    format_release_age,
    format_release_age_parenthetical,
    format_sdk_age_suffix,
    format_sdk_release_age,
    format_shadowed_dcode_fix_command,
    format_shadowed_dcode_warning,
    get_cached_update_available,
    get_last_update_check_time,
    get_latest_version,
    get_release_time,
    get_sdk_release_time,
    get_seen_version,
    install_extra_command,
    install_extra_recovery_command,
    install_extras_command,
    install_package_command,
    installed_days_old,
    is_auto_update_enabled,
    is_auto_update_explicitly_set,
    is_installation_stale,
    is_installed_version_at_least,
    is_update_available,
    is_valid_extra_name,
    is_valid_package_name,
    mark_auto_update_default_acknowledged,
    mark_startup_auto_update_failed,
    mark_update_notified,
    mark_version_seen,
    parse_dependency_changes,
    perform_dependency_refresh,
    perform_dependency_refresh_dry_run,
    perform_install_extra,
    perform_install_package,
    perform_upgrade,
    prerelease_upgrade_supported,
    release_requires_prereleases,
    set_auto_update,
    should_announce_auto_update_default,
    should_defer_startup_auto_update_for_resume,
    should_notify_update,
    should_skip_startup_auto_update_after_failure,
    upgrade_command,
    upgrade_install_command,
)


@pytest.fixture
def cache_file(tmp_path):
    """Override CACHE_FILE to use a temporary directory."""
    path = tmp_path / "latest_version.json"
    with patch("deepagents_code.update_check.CACHE_FILE", path):
        yield path


@pytest.fixture
def update_log_dir(tmp_path):
    """Override UPDATE_LOG_DIR to use a temporary directory."""
    path = tmp_path / "update_logs"
    with patch("deepagents_code.update_check.UPDATE_LOG_DIR", path):
        yield path


def _mock_pypi_response(
    version: str = "99.0.0",
    releases: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    release_times: dict[str, str] | None = None,
    requires_dist: Sequence[str] | None = None,
) -> MagicMock:
    if releases is None:
        releases = {version: [{"filename": "fake.tar.gz"}]}
    releases_data = {
        ver: [dict(file) for file in files] for ver, files in releases.items()
    }
    release_times = release_times or {}
    # Stamp upload_time_iso_8601 onto the first file of each release so the
    # real extraction path runs in tests.
    for ver, iso in release_times.items():
        files = releases_data.get(ver)
        if files:
            files[0]["upload_time_iso_8601"] = iso
    info: dict[str, object] = {"version": version}
    if requires_dist is not None:
        info["requires_dist"] = list(requires_dist)
    resp = MagicMock()
    resp.json.return_value = {
        "info": info,
        "releases": releases_data,
    }
    resp.raise_for_status = MagicMock()
    return resp


def _write_dist_info(
    root: Path,
    name: str,
    *,
    version: str = "1.0.0",
    requires: tuple[str, ...] = (),
) -> None:
    normalized = name.replace("-", "_")
    dist_info = root / f"{normalized}-{version}.dist-info"
    dist_info.mkdir()
    metadata = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    metadata.extend(f"Requires-Dist: {req}" for req in requires)
    dist_info.joinpath("METADATA").write_text("\n".join(metadata), encoding="utf-8")


def _write_uv_receipt(
    root: Path,
    requirements: str,
    *,
    python: str | None = None,
) -> None:
    python_line = f'python = "{python}"\n' if python is not None else ""
    root.joinpath("uv-receipt.toml").write_text(
        f"[tool]\n{python_line}requirements = [{requirements}]\n",
        encoding="utf-8",
    )


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


class TestInstalledVersionAtLeast:
    def test_true_when_distribution_metadata_matches_target(self) -> None:
        with patch("importlib.metadata.version", return_value="2.0.0"):
            assert is_installed_version_at_least("2.0.0") is True

    def test_true_when_distribution_metadata_is_newer(self) -> None:
        with patch("importlib.metadata.version", return_value="2.0.1"):
            assert is_installed_version_at_least("2.0.0") is True

    def test_false_when_distribution_metadata_is_older(self) -> None:
        with patch("importlib.metadata.version", return_value="1.9.9"):
            assert is_installed_version_at_least("2.0.0") is False


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


class TestCachedUpdateAvailable:
    def test_fresh_cache_reports_update_without_http(self, cache_file) -> None:
        """Fresh cache can drive startup auto-update without network access."""
        cache_file.write_text(
            json.dumps({"version": "99.0.0", "checked_at": time.time()}),
            encoding="utf-8",
        )

        with patch("requests.get") as mock_get:
            assert get_cached_update_available() == (True, "99.0.0")

        mock_get.assert_not_called()

    def test_stale_cache_returns_no_answer_without_http(self, cache_file) -> None:
        """Stale cache must not trigger a startup network request."""
        cache_file.write_text(
            json.dumps(
                {"version": "99.0.0", "checked_at": time.time() - CACHE_TTL - 1}
            ),
            encoding="utf-8",
        )

        with patch("requests.get") as mock_get:
            assert get_cached_update_available() == (False, None)

        mock_get.assert_not_called()

    @pytest.mark.parametrize("checked_at", [float("nan"), float("inf"), 1e100])
    def test_invalid_numeric_checked_at_returns_no_answer_without_http(
        self, cache_file, checked_at: float
    ) -> None:
        """Corrupt numeric timestamps must not be treated as fresh cache."""
        cache_file.write_text(
            json.dumps({"version": "99.0.0", "checked_at": checked_at}),
            encoding="utf-8",
        )

        with patch("requests.get") as mock_get:
            assert get_cached_update_available() == (False, None)

        mock_get.assert_not_called()

    def test_missing_cache_returns_no_answer_without_http(self, cache_file) -> None:
        """Missing cache should not block startup on a network request."""
        assert not cache_file.exists()
        with patch("requests.get") as mock_get:
            assert get_cached_update_available() == (False, None)

        mock_get.assert_not_called()

    def test_fresh_current_cache_reports_no_update(self, cache_file) -> None:
        """A fresh cache at the installed version should not update."""
        cache_file.write_text(
            json.dumps({"version": __version__, "checked_at": time.time()}),
            encoding="utf-8",
        )

        assert get_cached_update_available() == (False, __version__)


class TestGetLastUpdateCheckTime:
    def test_reads_checked_at(self, cache_file) -> None:
        """The stored `checked_at` epoch is returned as a float."""
        now = time.time()
        cache_file.write_text(json.dumps({"checked_at": now}), encoding="utf-8")
        assert get_last_update_check_time() == pytest.approx(now)

    def test_missing_cache_returns_none(self, cache_file) -> None:  # noqa: ARG002
        """An absent cache yields `None`."""
        assert get_last_update_check_time() is None

    def test_corrupt_cache_returns_none(self, cache_file) -> None:
        """Unparseable cache data fails soft to `None`."""
        cache_file.write_text("{not valid json", encoding="utf-8")
        assert get_last_update_check_time() is None

    def test_non_numeric_checked_at_returns_none(self, cache_file) -> None:
        """A non-numeric (or boolean) `checked_at` is ignored."""
        cache_file.write_text(json.dumps({"checked_at": "soon"}), encoding="utf-8")
        assert get_last_update_check_time() is None
        cache_file.write_text(json.dumps({"checked_at": True}), encoding="utf-8")
        assert get_last_update_check_time() is None

    @pytest.mark.parametrize("checked_at", [float("nan"), float("inf"), 1e100])
    def test_invalid_numeric_checked_at_returns_none(
        self, cache_file, checked_at: float
    ) -> None:
        """Invalid numeric `checked_at` values are ignored."""
        cache_file.write_text(json.dumps({"checked_at": checked_at}), encoding="utf-8")
        assert get_last_update_check_time() is None


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

    def test_fresh_fetch_caches_prerelease_dependency_requirement(
        self, cache_file
    ) -> None:
        """Stable dcode releases can intentionally pin pre-release dependencies."""
        with patch(
            "requests.get",
            return_value=_mock_pypi_response(
                "2.0.0",
                requires_dist=("deepagents==0.7.0a2",),
            ),
        ):
            result = get_latest_version()

        assert result == "2.0.0"
        assert release_requires_prereleases("2.0.0") is True
        data = json.loads(cache_file.read_text())
        assert data["release_requires_prereleases"] == {"2.0.0": True}

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
            json.dumps(
                {
                    "version": "1.5.0",
                    "release_times": {__version__: "2026-04-01T12:00:00Z"},
                    "checked_at": time.time(),
                }
            )
        )
        with patch("requests.get") as mock_get:
            result = get_latest_version()

        assert result == "1.5.0"
        mock_get.assert_not_called()

    def test_cached_hit_missing_installed_release_time_triggers_fetch(
        self, cache_file
    ) -> None:
        """Old cache files are refreshed so installed age notices have data."""
        cache_file.write_text(
            json.dumps({"version": "1.5.0", "checked_at": time.time()})
        )
        releases = {
            "2.0.0": [{"filename": "a.tar.gz"}],
            __version__: [{"filename": "installed.tar.gz"}],
        }
        with patch(
            "requests.get",
            return_value=_mock_pypi_response(
                "2.0.0",
                releases=releases,
                release_times={__version__: "2026-04-01T12:00:00Z"},
            ),
        ) as mock_get:
            result = get_latest_version()

        assert result == "2.0.0"
        mock_get.assert_called_once()
        data = json.loads(cache_file.read_text())
        assert data["release_times"][__version__] == "2026-04-01T12:00:00Z"

    def test_cached_hit_missing_installed_release_time_falls_back_on_fetch_error(
        self, cache_file
    ) -> None:
        """Age metadata refresh failures must not discard a fresh cached version."""
        cache_file.write_text(
            json.dumps({"version": "1.5.0", "checked_at": time.time()})
        )
        with patch("requests.get", side_effect=OSError("offline")) as mock_get:
            result = get_latest_version()

        assert result == "1.5.0"
        mock_get.assert_called_once()

    def test_cached_hit_prerelease(self, cache_file) -> None:
        """Fresh cache returns pre-release version without HTTP call."""
        cache_file.write_text(
            json.dumps(
                {
                    "version": "1.5.0",
                    "version_prerelease": "1.6.0a1",
                    "release_times": {__version__: "2026-04-01T12:00:00Z"},
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
                    "release_times": {__version__: "2026-04-01T12:00:00Z"},
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

    def test_fresh_fetch_preserves_other_release_prerelease_entries(
        self, cache_file
    ) -> None:
        """A refresh keeps cached pre-release answers for unrelated versions."""
        cache_file.write_text(
            json.dumps(
                {
                    "release_requires_prereleases": {"1.0.0": True},
                    "checked_at": time.time(),
                }
            ),
            encoding="utf-8",
        )
        with patch("requests.get", return_value=_mock_pypi_response("2.0.0")):
            result = get_latest_version()

        assert result == "2.0.0"
        data = json.loads(cache_file.read_text())
        assert data["release_requires_prereleases"] == {"1.0.0": True, "2.0.0": False}

    def test_fresh_fetch_non_dict_info_returns_cached(self, cache_file) -> None:
        """A PyPI payload whose `info` is not an object falls back to cache."""
        resp = MagicMock()
        resp.json.return_value = {"info": "not-a-dict", "releases": {}}
        resp.raise_for_status = MagicMock()
        with patch("requests.get", return_value=resp):
            result = get_latest_version()

        assert result is None
        assert not cache_file.exists()

    def test_fresh_fetch_non_str_version_returns_cached(self, cache_file) -> None:
        """A PyPI payload whose `info.version` is not a string falls back."""
        resp = MagicMock()
        resp.json.return_value = {"info": {"version": 123}, "releases": {}}
        resp.raise_for_status = MagicMock()
        with patch("requests.get", return_value=resp):
            result = get_latest_version()

        assert result is None
        assert not cache_file.exists()


class TestRequiresPrereleaseDependency:
    """Unit tests for the `Requires-Dist` pre-release detection helper."""

    @pytest.mark.parametrize(
        ("requirements", "expected"),
        [
            (None, False),
            ((), False),
            (("deepagents==0.7.0",), False),
            (("deepagents==0.7.0a2",), True),
            (("deepagents>=0.7.0a2",), True),
            (("deepagents~=0.7.0a2",), True),
            # Operator-agnostic by design: even an exclusion of a pre-release
            # flags the release. Errs toward enabling --prerelease (safe).
            (("deepagents!=0.7.0a1",), True),
            # Marker-gated pre-release pins still flag the release (conservative).
            (('deepagents==0.7.0a2; extra=="x"',), True),
            (("deepagents>=0.7.0,<0.8",), False),
            (("deepagents",), False),  # no version specifier
            (("requests>=2", "deepagents==0.7.0a2"), True),  # one of many
            (("not a valid requirement !!!",), False),  # unparseable -> skipped
            (("deepagents===not.a.version",), False),  # unparseable version
            ((123, "deepagents==0.7.0a2"), True),  # non-str entry skipped
            ((123, 456), False),  # all non-str entries
        ],
    )
    def test_detects_prerelease_specifiers(self, requirements, expected) -> None:
        assert _requires_prerelease_dependency(requirements) is expected


class TestReleaseRequiresPrereleases:
    """Unit tests for `release_requires_prereleases`."""

    def test_none_version_is_false(self) -> None:
        """A missing version never needs the pre-release resolver."""
        assert release_requires_prereleases(None) is False

    def test_cache_hit_true(self, cache_file) -> None:
        """A cached `True` short-circuits without a network call."""
        cache_file.write_text(
            json.dumps({"release_requires_prereleases": {"1.1.0": True}}),
            encoding="utf-8",
        )
        with patch("requests.get") as get_mock:
            assert release_requires_prereleases("1.1.0") is True
        get_mock.assert_not_called()

    def test_cache_hit_false(self, cache_file) -> None:
        """A cached `False` short-circuits without a network call."""
        cache_file.write_text(
            json.dumps({"release_requires_prereleases": {"1.1.0": False}}),
            encoding="utf-8",
        )
        with patch("requests.get") as get_mock:
            assert release_requires_prereleases("1.1.0") is False
        get_mock.assert_not_called()

    def test_cache_miss_fetches_and_writes(self, cache_file) -> None:
        """A cache miss fetches per-version metadata and caches the result."""
        with patch(
            "requests.get",
            return_value=_mock_pypi_response(
                "1.1.0", requires_dist=("deepagents==0.7.0a2",)
            ),
        ) as get_mock:
            assert release_requires_prereleases("1.1.0") is True
        assert get_mock.call_args.args[0].endswith("/deepagents-code/1.1.0/json")
        data = json.loads(cache_file.read_text())
        assert data["release_requires_prereleases"]["1.1.0"] is True

    def test_bypass_cache_forces_fetch(self, cache_file) -> None:
        """`bypass_cache` re-fetches even when a cached value exists."""
        cache_file.write_text(
            json.dumps({"release_requires_prereleases": {"1.1.0": False}}),
            encoding="utf-8",
        )
        with patch(
            "requests.get",
            return_value=_mock_pypi_response(
                "1.1.0", requires_dist=("deepagents==0.7.0a2",)
            ),
        ) as get_mock:
            result = release_requires_prereleases("1.1.0", bypass_cache=True)
        assert result is True
        get_mock.assert_called_once()

    def test_network_failure_returns_false(self, cache_file) -> None:
        """A PyPI failure conservatively reports stable-only resolution."""
        import requests

        with patch("requests.get", side_effect=requests.RequestException("boom")):
            assert release_requires_prereleases("1.1.0") is False
        # Nothing cached, so a later successful lookup can still self-correct.
        assert not cache_file.exists()

    def test_missing_requests_returns_false(self, cache_file) -> None:
        """Without `requests`, the lookup degrades to stable-only resolution."""
        with patch.dict(sys.modules, {"requests": None}):
            assert release_requires_prereleases("1.1.0") is False
        assert not cache_file.exists()

    def test_malformed_info_returns_false(self, cache_file) -> None:
        """A payload with a non-object `info` is treated as no requirement."""
        resp = MagicMock()
        resp.json.return_value = {"info": "not-a-dict"}
        resp.raise_for_status = MagicMock()
        with patch("requests.get", return_value=resp):
            assert release_requires_prereleases("1.1.0") is False
        data = json.loads(cache_file.read_text())
        assert data["release_requires_prereleases"]["1.1.0"] is False


class TestIsUpdateAvailable:
    def test_newer_available(self) -> None:
        with patch(
            "deepagents_code.update_check.get_latest_version", return_value="99.0.0"
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
                "deepagents_code.update_check.get_latest_version", return_value="0.0.1"
            ),
            patch("deepagents_code.update_check.__version__", "0.0.1"),
        ):
            available, latest = is_update_available()

        assert available is False
        assert latest == "0.0.1"

    def test_ahead_of_pypi(self) -> None:
        """Dev build ahead of PyPI should not flag an update."""
        with (
            patch(
                "deepagents_code.update_check.get_latest_version", return_value="0.0.1"
            ),
            patch("deepagents_code.update_check.__version__", "99.0.0"),
        ):
            available, latest = is_update_available()

        assert available is False
        assert latest == "0.0.1"

    def test_fetch_failure(self) -> None:
        with patch(
            "deepagents_code.update_check.get_latest_version", return_value=None
        ):
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
                "deepagents_code.update_check.get_latest_version", return_value="1.2.3"
            ),
            patch("deepagents_code.update_check.__version__", "1.2.3"),
        ):
            up_to_date = is_update_available()

        with patch(
            "deepagents_code.update_check.get_latest_version", return_value=None
        ):
            fetch_failed = is_update_available()

        assert up_to_date == (False, "1.2.3")
        assert fetch_failed == (False, None)

    def test_prerelease_user_sees_newer_prerelease(self) -> None:
        """User on alpha sees a newer alpha as available."""
        with (
            patch(
                "deepagents_code.update_check.get_latest_version",
                return_value="1.0.0a2",
            ),
            patch("deepagents_code.update_check.__version__", "1.0.0a1"),
        ):
            available, latest = is_update_available()

        assert available is True
        assert latest == "1.0.0a2"

    def test_prerelease_user_sees_stable_release(self) -> None:
        """User on alpha sees the stable release as available."""
        with (
            patch(
                "deepagents_code.update_check.get_latest_version",
                return_value="1.0.0",
            ),
            patch("deepagents_code.update_check.__version__", "1.0.0a1"),
        ):
            available, latest = is_update_available()

        assert available is True
        assert latest == "1.0.0"

    def test_stable_user_does_not_see_prerelease(self) -> None:
        """Stable user on current version sees no update available."""
        with (
            patch(
                "deepagents_code.update_check.get_latest_version",
                return_value="1.0.0",
            ),
            patch("deepagents_code.update_check.__version__", "1.0.0"),
        ):
            available, latest = is_update_available()

        assert available is False
        assert latest == "1.0.0"

    def test_include_prereleases_kwarg_passed(self) -> None:
        """Verify include_prereleases is True when installed version is pre-release."""
        with (
            patch(
                "deepagents_code.update_check.get_latest_version",
                return_value=None,
            ) as mock_get,
            patch("deepagents_code.update_check.__version__", "1.0.0a1"),
        ):
            is_update_available()

        mock_get.assert_called_once_with(bypass_cache=False, include_prereleases=True)

    def test_include_prereleases_false_for_stable(self) -> None:
        """Verify include_prereleases is False when installed version is stable."""
        with (
            patch(
                "deepagents_code.update_check.get_latest_version",
                return_value=None,
            ) as mock_get,
            patch("deepagents_code.update_check.__version__", "1.0.0"),
        ):
            is_update_available()

        mock_get.assert_called_once_with(bypass_cache=False, include_prereleases=False)

    def test_explicit_include_prereleases_overrides_stable_install(self) -> None:
        """Explicit `include_prereleases=True` beats a stable installed version."""
        with (
            patch(
                "deepagents_code.update_check.get_latest_version",
                return_value=None,
            ) as mock_get,
            patch("deepagents_code.update_check.__version__", "1.0.0"),
        ):
            is_update_available(include_prereleases=True)

        mock_get.assert_called_once_with(bypass_cache=False, include_prereleases=True)

    def test_explicit_exclude_prereleases_overrides_prerelease_install(self) -> None:
        """Explicit `include_prereleases=False` beats a pre-release install."""
        with (
            patch(
                "deepagents_code.update_check.get_latest_version",
                return_value=None,
            ) as mock_get,
            patch("deepagents_code.update_check.__version__", "1.0.0a1"),
        ):
            is_update_available(include_prereleases=False)

        mock_get.assert_called_once_with(bypass_cache=False, include_prereleases=False)

    def test_invalid_installed_version(self) -> None:
        """Non-PEP 440 installed version disables update check gracefully."""
        with patch("deepagents_code.update_check.__version__", "not-a-version"):
            available, latest = is_update_available()

        assert available is False
        assert latest is None

    def test_unparseable_pypi_version(self) -> None:
        """Malformed PyPI version string does not crash."""
        with (
            patch(
                "deepagents_code.update_check.get_latest_version",
                return_value="not-a-version",
            ),
            patch("deepagents_code.update_check.__version__", "1.0.0"),
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

    def test_includes_installed_version_when_provided(self) -> None:
        payload = {
            "releases": {
                "1.0.0": [{"upload_time_iso_8601": "2026-04-15T12:00:00Z"}],
                "0.9.0": [{"upload_time_iso_8601": "2026-04-01T12:00:00Z"}],
            },
        }
        times = _extract_release_times(
            payload, stable="1.0.0", prerelease=None, installed="0.9.0"
        )
        assert times == {
            "1.0.0": "2026-04-15T12:00:00Z",
            "0.9.0": "2026-04-01T12:00:00Z",
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
            "deepagents_code.sessions.format_relative_timestamp", return_value=""
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
            "deepagents_code.sessions.format_relative_timestamp", return_value="3d ago"
        ):
            assert format_age_suffix("1.0.0") == ", released 3d ago"

    def test_unknown_age_returns_empty(self, cache_file) -> None:  # noqa: ARG002
        """Unknown age collapses to `""` so callers can concat unconditionally."""
        assert format_age_suffix("1.0.0") == ""

    def test_none_version_returns_empty(self, cache_file) -> None:  # noqa: ARG002
        assert format_age_suffix(None) == ""


class TestFormatReleaseAgeParenthetical:
    def test_returns_parenthesized_release_age(self, cache_file) -> None:
        """Known age is formatted for update-available lead sentences."""
        cache_file.write_text(
            json.dumps(
                {
                    "release_times": {"1.0.0": "2026-04-15T12:00:00Z"},
                    "checked_at": time.time(),
                }
            )
        )
        with patch(
            "deepagents_code.sessions.format_relative_timestamp", return_value="3d ago"
        ):
            assert format_release_age_parenthetical("1.0.0") == " (released 3d ago)"

    def test_unknown_age_returns_empty(self, cache_file) -> None:  # noqa: ARG002
        assert format_release_age_parenthetical("1.0.0") == ""


class TestFormatInstalledAgeSuffix:
    def test_returns_days_old_for_versions_at_least_one_week_old(
        self, cache_file
    ) -> None:
        """Installed version age is shown only after it crosses the threshold."""
        from datetime import UTC, datetime, timedelta

        iso = (datetime.now(tz=UTC) - timedelta(days=8)).isoformat()
        cache_file.write_text(
            json.dumps(
                {
                    "release_times": {"1.0.0": iso},
                    "checked_at": time.time(),
                }
            )
        )
        assert format_installed_age_suffix("1.0.0") == " (8 days old)"

    def test_omits_versions_newer_than_one_week(self, cache_file) -> None:
        from datetime import UTC, datetime, timedelta

        iso = (datetime.now(tz=UTC) - timedelta(days=6)).isoformat()
        cache_file.write_text(
            json.dumps(
                {
                    "release_times": {"1.0.0": iso},
                    "checked_at": time.time(),
                }
            )
        )
        assert format_installed_age_suffix("1.0.0") == ""

    def test_unknown_age_returns_empty(self, cache_file) -> None:  # noqa: ARG002
        assert format_installed_age_suffix("1.0.0") == ""


def _write_installed_release_time(
    cache_file: Path, *, days_ago: int, latest_version: str | None = None
) -> None:
    """Seed the cache with a release time for the running version."""
    from datetime import UTC, datetime, timedelta

    iso = (datetime.now(tz=UTC) - timedelta(days=days_ago)).isoformat()
    data: dict[str, object] = {
        "release_times": {__version__: iso},
        "checked_at": time.time(),
    }
    if latest_version is not None:
        data["version"] = latest_version
    cache_file.write_text(
        json.dumps(data),
        encoding="utf-8",
    )


class TestInstalledDaysOld:
    def test_returns_days_for_known_release(self, cache_file) -> None:
        _write_installed_release_time(cache_file, days_ago=21)
        assert installed_days_old() == 21

    def test_unknown_release_returns_none(self, cache_file) -> None:  # noqa: ARG002
        assert installed_days_old() is None

    def test_malformed_timestamp_returns_none(self, cache_file: Path) -> None:
        cache_file.write_text(
            json.dumps(
                {
                    "release_times": {__version__: "not-a-timestamp"},
                    "checked_at": time.time(),
                }
            ),
            encoding="utf-8",
        )
        assert installed_days_old() is None

    def test_none_when_release_lookup_raises(self, cache_file) -> None:  # noqa: ARG002
        """An unexpected error degrades to `None` instead of propagating."""
        with patch(
            "deepagents_code.update_check.get_release_time",
            side_effect=OSError("boom"),
        ):
            assert installed_days_old() is None


class TestIsInstallationStale:
    def test_true_when_older_than_threshold(self, cache_file) -> None:
        _write_installed_release_time(
            cache_file,
            days_ago=INSTALLED_STALE_NOTICE_DAYS + 1,
            latest_version="99.0.0",
        )
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
        ):
            assert is_installation_stale() is True

    def test_false_when_current_version_is_old(self, cache_file) -> None:
        _write_installed_release_time(
            cache_file,
            days_ago=INSTALLED_STALE_NOTICE_DAYS + 30,
            latest_version=__version__,
        )
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
        ):
            assert is_installation_stale() is False

    def test_true_at_exact_threshold(self, cache_file) -> None:
        """Exactly `INSTALLED_STALE_NOTICE_DAYS` old is stale (inclusive `>=`)."""
        _write_installed_release_time(
            cache_file,
            days_ago=INSTALLED_STALE_NOTICE_DAYS,
            latest_version="99.0.0",
        )
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
        ):
            assert is_installation_stale() is True

    def test_false_when_newer_than_threshold(self, cache_file) -> None:
        _write_installed_release_time(
            cache_file,
            days_ago=INSTALLED_STALE_NOTICE_DAYS - 1,
            latest_version="99.0.0",
        )
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
        ):
            assert is_installation_stale() is False

    def test_false_when_editable_check_raises(self, cache_file) -> None:  # noqa: ARG002
        """A raising editable check degrades to `False`, never aborting startup."""
        with patch(
            "deepagents_code.config._is_editable_install",
            side_effect=PermissionError("direct_url.json unreadable"),
        ):
            assert is_installation_stale() is False

    def test_false_for_editable_install(self, cache_file) -> None:
        _write_installed_release_time(
            cache_file, days_ago=INSTALLED_STALE_NOTICE_DAYS + 30
        )
        with patch("deepagents_code.config._is_editable_install", return_value=True):
            assert is_installation_stale() is False

    def test_false_when_update_checks_disabled(self, cache_file) -> None:
        _write_installed_release_time(
            cache_file, days_ago=INSTALLED_STALE_NOTICE_DAYS + 30
        )
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=False,
            ),
        ):
            assert is_installation_stale() is False

    def test_false_on_cold_cache(self, cache_file) -> None:  # noqa: ARG002
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
        ):
            assert is_installation_stale() is False


class TestDetectInstallMethod:
    def test_non_editable_non_uv_non_brew_returns_other(self) -> None:
        """The fallback bucket is not a positive pip detection."""
        with (
            patch("deepagents_code.update_check.sys.prefix", "/tmp/dcode-venv"),
            patch("deepagents_code.config._is_editable_install", return_value=False),
        ):
            assert detect_install_method() == "other"


class TestUvToolBinDir:
    """Coverage for uv's documented executable-directory precedence chain.

    `detect_shadowed_dcode` compares the user's PATH against whatever
    `_uv_tool_bin_dir` returns, so any drift between this helper and uv's
    actual install location causes false-positive shadow warnings *and*
    skipped auto-update restarts. Each candidate in uv's precedence list
    gets explicit coverage.
    """

    def test_uv_tool_bin_dir_env_wins(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`UV_TOOL_BIN_DIR` overrides every other candidate."""
        override = tmp_path / "uv-tool-bin"
        override.mkdir()
        xdg_bin = tmp_path / "xdg-bin"
        xdg_bin.mkdir()
        monkeypatch.setenv("UV_TOOL_BIN_DIR", str(override))
        monkeypatch.setenv("XDG_BIN_HOME", str(xdg_bin))

        assert _uv_tool_bin_dir() == override.resolve()

    def test_xdg_bin_home_wins_when_uv_var_unset(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`XDG_BIN_HOME` is the second-precedence candidate per uv's docs.

        Hits the branch users hit on Linux when they've adopted the XDG
        Base Directory convention but haven't set uv-specific overrides.
        Without this branch the detector would skip past XDG_BIN_HOME to
        ~/.local/bin and silently warn on every successful upgrade.
        """
        xdg_bin = tmp_path / "xdg-bin"
        xdg_bin.mkdir()
        # Also create a `~/.local/bin` candidate to prove XDG_BIN_HOME
        # wins even when later candidates exist.
        home = tmp_path / "home"
        (home / ".local" / "bin").mkdir(parents=True)
        monkeypatch.delenv("UV_TOOL_BIN_DIR", raising=False)
        monkeypatch.setenv("XDG_BIN_HOME", str(xdg_bin))
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))

        assert _uv_tool_bin_dir() == xdg_bin.resolve()

    def test_xdg_data_home_parent_bin_wins_when_only_xdg_data_set(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`$XDG_DATA_HOME/../bin` is uv's third-precedence candidate.

        Pins the intermediate fallback rather than collapsing it into the
        `~/.local/bin` default — a setup where the user has redirected
        XDG_DATA_HOME (e.g. to a non-standard prefix) must land on the
        sibling bin dir uv itself would target.
        """
        data_root = tmp_path / "alt-data-root"
        data_root.mkdir()
        sibling_bin = tmp_path / "bin"
        sibling_bin.mkdir()
        home = tmp_path / "home"
        (home / ".local" / "bin").mkdir(parents=True)
        monkeypatch.delenv("UV_TOOL_BIN_DIR", raising=False)
        monkeypatch.delenv("XDG_BIN_HOME", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(data_root))
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))

        # `$XDG_DATA_HOME/../bin` resolves to `tmp_path/bin` (sibling).
        assert _uv_tool_bin_dir() == sibling_bin.resolve()

    def test_local_bin_fallback_when_no_env_vars(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`~/.local/bin` is the documented final fallback on Unix and Windows.

        The path most real users hit: no env vars set, default home,
        `~/.local/bin` exists. Without coverage here a regression that
        broke the fallback would only show up in production.
        """
        home = tmp_path / "home"
        local_bin = home / ".local" / "bin"
        local_bin.mkdir(parents=True)
        monkeypatch.delenv("UV_TOOL_BIN_DIR", raising=False)
        monkeypatch.delenv("XDG_BIN_HOME", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))

        assert _uv_tool_bin_dir() == local_bin.resolve()

    def test_returns_none_when_no_candidate_exists(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No env vars and no `~/.local/bin` → `None`, not a bogus path.

        Returning a non-existent path would make `detect_shadowed_dcode`
        report every install as shadowed against a directory the user
        couldn't possibly have on PATH. `None` is the right signal so the
        detector short-circuits silently.
        """
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.delenv("UV_TOOL_BIN_DIR", raising=False)
        monkeypatch.delenv("XDG_BIN_HOME", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))

        assert _uv_tool_bin_dir() is None

    def test_skips_missing_candidate_and_falls_through(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A higher-precedence candidate that doesn't exist falls through.

        An env var set to a non-existent path must not bind the answer to
        that bad value — the helper should keep walking the precedence
        list. This is what makes the env override safe to set
        unconditionally in dotfiles even when the directory hasn't been
        created yet.
        """
        missing = tmp_path / "does-not-exist"
        # Deliberately do not mkdir.
        home = tmp_path / "home"
        local_bin = home / ".local" / "bin"
        local_bin.mkdir(parents=True)
        monkeypatch.setenv("UV_TOOL_BIN_DIR", str(missing))
        monkeypatch.delenv("XDG_BIN_HOME", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))

        assert _uv_tool_bin_dir() == local_bin.resolve()

    def test_resolve_failure_falls_through_to_next_candidate(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A candidate that raises on `resolve()` must not bind the answer.

        Distinct from the "missing candidate" case: here the higher-precedence
        candidate exists but `resolve()` raises `OSError` (a vanished mount,
        a permission glitch). The helper's `except OSError: continue` exists so
        a transient failure doesn't downgrade the answer to a less-preferred
        path *or* poison it with the bad candidate — it must keep walking to
        the next entry, exactly like the missing-candidate path.
        """
        override = tmp_path / "uv-tool-bin"
        override.mkdir()
        home = tmp_path / "home"
        local_bin = home / ".local" / "bin"
        local_bin.mkdir(parents=True)
        monkeypatch.setenv("UV_TOOL_BIN_DIR", str(override))
        monkeypatch.delenv("XDG_BIN_HOME", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))

        real_resolve = Path.resolve

        def _resolve(self: Path, strict: bool = False) -> Path:
            # Only the first (UV_TOOL_BIN_DIR) candidate raises; everything
            # else — including the eventual `~/.local/bin` winner — resolves
            # normally so the test pins fallthrough, not a blanket failure.
            if self == override:
                msg = "simulated resolve failure"
                raise OSError(msg)
            return real_resolve(self, strict)

        monkeypatch.setattr(Path, "resolve", _resolve)

        assert _uv_tool_bin_dir() == real_resolve(local_bin)


class TestDetectShadowedDcode:
    """Regression coverage for the post-upgrade shadowing detector.

    The detector is the only thing standing between a successful `uv tool
    upgrade` and the user silently relaunching into a pre-uv `dcode` earlier on
    PATH, so each branch of the comparison is covered explicitly.
    """

    def test_returns_none_for_non_uv_install(self, tmp_path) -> None:
        """Non-uv installs cannot describe an 'upgraded shim' location."""
        uv_bin = tmp_path / "bin"
        uv_bin.mkdir()
        (uv_bin / "dcode").write_text("")
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="brew",
            ),
            patch.dict(os.environ, {"UV_TOOL_BIN_DIR": str(uv_bin)}),
            patch("shutil.which", return_value=str(uv_bin / "dcode")),
        ):
            assert detect_shadowed_dcode() is None

    def test_returns_none_when_path_resolves_into_uv_bin_dir(self, tmp_path) -> None:
        """The happy path: PATH points at the directory uv installs into."""
        uv_bin = tmp_path / "bin"
        uv_bin.mkdir()
        shim = uv_bin / "dcode"
        shim.write_text("")
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch.dict(os.environ, {"UV_TOOL_BIN_DIR": str(uv_bin)}),
            patch("shutil.which", return_value=str(shim)),
        ):
            assert detect_shadowed_dcode() is None

    def test_checks_deepagents_code_when_dcode_is_healthy(self, tmp_path) -> None:
        """A healthy `dcode` must not hide a shadowed `deepagents-code`."""
        uv_bin = tmp_path / "uv-bin"
        uv_bin.mkdir()
        (uv_bin / "dcode").write_text("")
        (uv_bin / "deepagents-code").write_text("")
        stale_bin = tmp_path / "stale-bin"
        stale_bin.mkdir()
        stale = stale_bin / "deepagents-code"
        stale.write_text("")

        def _which(name: str) -> str | None:
            if name == "dcode":
                return str(uv_bin / "dcode")
            if name == "deepagents-code":
                return str(stale)
            return None

        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch.dict(os.environ, {"UV_TOOL_BIN_DIR": str(uv_bin)}),
            patch("shutil.which", side_effect=_which),
        ):
            shadow = detect_shadowed_dcode()

        assert shadow == ShadowedDcode(
            shadowing_bin=stale,
            upgraded_bin_dir=uv_bin.resolve(),
        )

    def test_returns_none_for_uv_symlink_shim(self, tmp_path) -> None:
        """A uv-style symlink shim under the user bin dir is NOT a shadow.

        On a healthy uv tool install, `~/.local/bin/dcode` is a symlink to
        `~/.local/share/uv/tools/deepagents-code/bin/dcode`. If we followed
        that symlink the parent would be the tool venv's internal bin dir,
        which differs from uv's user-facing bin dir and would make every
        healthy install look shadowed. The detector must compare the
        PATH-entry directory, not the symlink target.
        """
        uv_bin = tmp_path / "uv-bin"
        uv_bin.mkdir()
        tool_internal_bin = tmp_path / "tools" / "deepagents-code" / "bin"
        tool_internal_bin.mkdir(parents=True)
        real_entry_point = tool_internal_bin / "dcode"
        real_entry_point.write_text("")
        shim = uv_bin / "dcode"
        shim.symlink_to(real_entry_point)

        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch.dict(os.environ, {"UV_TOOL_BIN_DIR": str(uv_bin)}),
            patch("shutil.which", return_value=str(shim)),
        ):
            assert detect_shadowed_dcode() is None

    def test_returns_shadow_when_path_resolves_outside_uv_bin_dir(
        self, tmp_path
    ) -> None:
        """A different `dcode` earlier on PATH is the bug we're protecting against.

        Also pins the reported `shadowing_bin` to the PATH-visible path
        (not the resolved symlink target), since that's the file the user
        needs to act on.
        """
        uv_bin = tmp_path / "uv-bin"
        uv_bin.mkdir()
        (uv_bin / "dcode").write_text("")  # the upgraded shim uv would install
        stale_bin = tmp_path / "stale-bin"
        stale_bin.mkdir()
        stale = stale_bin / "dcode"
        stale.write_text("")

        def _which(name: str) -> str | None:
            return str(stale) if name == "dcode" else None

        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch.dict(os.environ, {"UV_TOOL_BIN_DIR": str(uv_bin)}),
            patch("shutil.which", side_effect=_which),
        ):
            shadow = detect_shadowed_dcode()

        assert shadow == ShadowedDcode(
            shadowing_bin=stale,
            upgraded_bin_dir=uv_bin.resolve(),
        )

    def test_returns_shadow_for_symlink_shim_in_wrong_directory(self, tmp_path) -> None:
        """A symlinked `dcode` outside uv's bin dir is still a real shadow.

        Distinguishes the genuine shadow case (symlink in some other PATH
        directory) from the false-positive case the previous test covers
        (uv's own symlinks under its bin dir). Without separating these,
        a fix for either could regress the other.
        """
        uv_bin = tmp_path / "uv-bin"
        uv_bin.mkdir()
        (uv_bin / "dcode").write_text("")
        other_bin = tmp_path / "homebrew-bin"
        other_bin.mkdir()
        target = tmp_path / "Cellar" / "dcode" / "bin"
        target.mkdir(parents=True)
        real_dcode = target / "dcode"
        real_dcode.write_text("")
        stale_shim = other_bin / "dcode"
        stale_shim.symlink_to(real_dcode)

        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch.dict(os.environ, {"UV_TOOL_BIN_DIR": str(uv_bin)}),
            patch("shutil.which", return_value=str(stale_shim)),
        ):
            shadow = detect_shadowed_dcode()

        assert shadow is not None
        # The reported path is the PATH-entry symlink, not the resolved
        # target — that's what the user needs to delete or demote.
        assert shadow.shadowing_bin == stale_shim
        assert shadow.upgraded_bin_dir == uv_bin.resolve()

    def test_returns_none_when_no_dcode_on_path(self, tmp_path) -> None:
        """Without any `dcode` on PATH there's nothing to be shadowed by."""
        uv_bin = tmp_path / "uv-bin"
        uv_bin.mkdir()
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch.dict(os.environ, {"UV_TOOL_BIN_DIR": str(uv_bin)}),
            patch("shutil.which", return_value=None),
        ):
            assert detect_shadowed_dcode() is None

    def test_falls_back_to_deepagents_code_binary_name(self, tmp_path) -> None:
        """The `deepagents-code` binary is checked when `dcode` is missing.

        Mirrors the install-script verification loop so an install that only
        exposes `deepagents-code` (e.g. an older `uv tool install` that
        predates the `dcode` entry point) still gets shadow-checked.
        """
        uv_bin = tmp_path / "uv-bin"
        uv_bin.mkdir()
        (uv_bin / "deepagents-code").write_text("")
        stale_bin = tmp_path / "stale-bin"
        stale_bin.mkdir()
        stale = stale_bin / "deepagents-code"
        stale.write_text("")

        def _which(name: str) -> str | None:
            if name == "dcode":
                return None
            if name == "deepagents-code":
                return str(stale)
            return None

        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch.dict(os.environ, {"UV_TOOL_BIN_DIR": str(uv_bin)}),
            patch("shutil.which", side_effect=_which),
        ):
            shadow = detect_shadowed_dcode()

        assert shadow is not None
        assert shadow.shadowing_bin == stale
        assert shadow.upgraded_bin_dir == uv_bin.resolve()

    def test_warning_text_includes_both_paths(self, tmp_path) -> None:
        """The user-facing warning must name the shadowing binary AND the shim.

        Without both paths the user can't tell which one is wrong or how to
        fix their PATH, so this guards the message contract callers rely on.
        The suggested command is intentionally session-scoped and
        non-destructive because the shadowing binary may be package-managed.
        """
        shadow = ShadowedDcode(
            shadowing_bin=tmp_path / "old-bin" / "dcode",
            upgraded_bin_dir=tmp_path / "uv-bin",
        )
        rendered = format_shadowed_dcode_warning(shadow)
        assert str(shadow.shadowing_bin) in rendered
        assert str(shadow.upgraded_bin_dir / "dcode") in rendered
        assert "earlier on your PATH" in rendered
        command = format_shadowed_dcode_fix_command(shadow)
        assert command.replace("\n", "\n  ") in rendered
        assert "hash -r" in rendered
        assert "rm " not in rendered

    def test_warning_text_quotes_fix_command_path(self, tmp_path) -> None:
        """The suggested PATH command must be safe to copy with odd paths."""
        shadow = ShadowedDcode(
            shadowing_bin=tmp_path / "old bin" / "dcode",
            upgraded_bin_dir=tmp_path / "uv bin's dir",
        )

        rendered = format_shadowed_dcode_warning(shadow)
        command = format_shadowed_dcode_fix_command(shadow)
        quoted_bin_dir = shlex.quote(str(shadow.upgraded_bin_dir))

        assert (
            command
            == f"export PATH={quoted_bin_dir}:$PATH\nhash -r 2>/dev/null || true"
        )
        assert command.replace("\n", "\n  ") in rendered

    def test_windows_fix_command_uses_powershell_literal_path(self, tmp_path) -> None:
        """PowerShell paths must not expand `$` or evaluate subexpressions."""
        shadow = ShadowedDcode(
            shadowing_bin=tmp_path / "old-bin" / "dcode",
            upgraded_bin_dir=tmp_path / "uv $dcode's $(bin)",
        )

        with patch("deepagents_code.update_check.os.name", "nt"):
            command = format_shadowed_dcode_fix_command(shadow)

        quoted_bin_dir = str(shadow.upgraded_bin_dir).replace("'", "''")
        assert command == f"$env:PATH = '{quoted_bin_dir};' + $env:PATH"

    def test_canonicalize_failure_continues_to_deepagents_code_name(
        self, tmp_path
    ) -> None:
        """A `resolve()` failure on `dcode` must not hide another shadow.

        The detector deliberately `continue`s to the `deepagents-code` name
        when canonicalizing `dcode`'s PATH directory raises, rather than
        returning `None` (which would silently report "no shadow"). This pins
        that fall-through: `dcode`'s directory raises, but `deepagents-code`
        resolves to a stale directory and is still reported as the shadow. A
        regression that turned the `continue` into `return None` would
        re-introduce the exact silent-hide bug the inline comment warns about.
        """
        uv_bin = (tmp_path / "uv-bin").resolve()
        uv_bin.mkdir()
        bad_dir = tmp_path / "bad-dir"
        bad_dir.mkdir()
        (bad_dir / "dcode").write_text("")
        stale_bin = tmp_path / "stale-bin"
        stale_bin.mkdir()
        stale_deepagents_code = stale_bin / "deepagents-code"
        stale_deepagents_code.write_text("")

        def _which(name: str) -> str | None:
            if name == "dcode":
                return str(bad_dir / "dcode")
            if name == "deepagents-code":
                return str(stale_deepagents_code)
            return None

        real_resolve = Path.resolve

        def _resolve(self: Path, strict: bool = False) -> Path:
            # Only `dcode`'s PATH-entry directory raises; the other binary's
            # directory resolves cleanly so the loop can reach a real answer.
            if self == bad_dir:
                msg = "simulated resolve failure"
                raise OSError(msg)
            return real_resolve(self, strict)

        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check._uv_tool_bin_dir",
                return_value=uv_bin,
            ),
            patch("shutil.which", side_effect=_which),
            patch.object(Path, "resolve", _resolve),
        ):
            shadow = detect_shadowed_dcode()

        assert shadow is not None
        assert shadow.shadowing_bin == stale_deepagents_code
        assert shadow.upgraded_bin_dir == uv_bin


class TestDetectShadowedDcodeSafe:
    """The never-raises wrapper used at every post-upgrade call site.

    Shadow detection only decorates an already-successful upgrade, so a
    detector defect must degrade to "no shadow" rather than turning a working
    upgrade into a user-facing failure.
    """

    def test_passes_through_shadow(self, tmp_path) -> None:
        """A detected shadow flows through unchanged."""
        shadow = ShadowedDcode(
            shadowing_bin=tmp_path / "stale" / "dcode",
            upgraded_bin_dir=tmp_path / "uv-bin",
        )
        with patch(
            "deepagents_code.update_check.detect_shadowed_dcode",
            return_value=shadow,
        ):
            assert detect_shadowed_dcode_safe() == shadow

    def test_passes_through_none(self) -> None:
        """The common "no shadow" answer flows through unchanged."""
        with patch(
            "deepagents_code.update_check.detect_shadowed_dcode",
            return_value=None,
        ):
            assert detect_shadowed_dcode_safe() is None

    def test_swallows_unexpected_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unexpected raise becomes `None`, not a propagated crash.

        This is the whole reason the wrapper exists: the success path that
        calls it has already committed the upgrade, so a detector bug must not
        surface as "update failed". The failure is logged at warning level so
        it stays diagnosable.
        """
        with (
            patch(
                "deepagents_code.update_check.detect_shadowed_dcode",
                side_effect=RuntimeError("boom"),
            ),
            caplog.at_level(logging.WARNING, logger="deepagents_code.update_check"),
        ):
            assert detect_shadowed_dcode_safe() is None
        assert any("Shadow detection failed" in r.message for r in caplog.records)


class TestUpdateLogs:
    def test_create_update_log_path_uses_log_dir(self, update_log_dir) -> None:
        path = create_update_log_path()
        assert path.parent == update_log_dir
        assert path.name.endswith("-update.log")

    def test_cleanup_update_logs_removes_old_and_excess(self, update_log_dir) -> None:
        update_log_dir.mkdir(parents=True)
        now = time.time()
        paths = []
        for idx in range(4):
            path = update_log_dir / f"{idx}-update.log"
            path.write_text(str(idx))
            os.utime(path, (now - idx, now - idx))
            paths.append(path)
        old = update_log_dir / "old-update.log"
        old.write_text("old")
        os.utime(old, (now - 30 * 86_400, now - 30 * 86_400))

        cleanup_update_logs(retention_days=14, max_files=2)

        remaining = {path.name for path in update_log_dir.glob("*.log")}
        assert remaining == {paths[0].name, paths[1].name}

    async def test_perform_upgrade_runs_when_log_cannot_be_created(
        self, tmp_path
    ) -> None:
        """Log persistence is best-effort and must not block the updater."""
        blocked_parent = tmp_path / "not-a-dir"
        blocked_parent.write_text("file")
        log_path = blocked_parent / "update.log"

        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            # Stub the receipt-aware command builder so the test doesn't
            # depend on a real `uv-receipt.toml`; the assertion is about
            # log-creation failure surfacing through.
            patch(
                "deepagents_code.update_check.upgrade_install_command",
                return_value="printf 'ok\\n'",
            ),
        ):
            success, output = await perform_upgrade(log_path=log_path)

        assert success is True
        assert output == "ok"

    async def test_perform_upgrade_ignores_log_close_failure(self, tmp_path) -> None:
        """A close-time log flush failure must not fail a successful upgrade."""
        log_path = tmp_path / "update.log"
        opener = mock_open()
        opener.return_value.close.side_effect = OSError("flush failed")

        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            # `perform_upgrade` now calls `upgrade_install_command`, which
            # reads the uv receipt and distribution metadata. Stub those
            # out so the test can focus on the log-close-failure assertion
            # rather than fight with the broad `pathlib.Path.open` mock.
            patch(
                "deepagents_code.update_check.upgrade_install_command",
                return_value="printf 'ok\\n'",
            ),
            patch("pathlib.Path.open", opener),
        ):
            success, output = await perform_upgrade(log_path=log_path)

        assert success is True
        assert output == "ok"

    async def test_perform_upgrade_refuses_other_install(self) -> None:
        """Unknown non-editable installs must not upgrade a separate uv tool env."""
        with patch(
            "deepagents_code.update_check.detect_install_method",
            return_value="other",
        ):
            success, output = await perform_upgrade()

        assert success is False
        assert "Unsupported install method" in output

    async def test_perform_upgrade_uses_uv_prerelease_command(self) -> None:
        """Pre-release upgrades pass uv's explicit pre-release strategy.

        Uses `uv tool install -U` (not `uv tool upgrade`) so any stale
        `==<version>` pin in the receipt — left over from a prior install
        or dependency refresh — is cleared, letting uv resolve to the
        latest available release.
        """
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value=frozenset(),
            ),
            patch(
                "deepagents_code.update_check._uv_tool_python",
                return_value=None,
            ),
            patch(
                "deepagents_code.update_check._uv_tool_with_packages",
                return_value=(),
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as run_mock,
        ):
            success, _output = await perform_upgrade(include_prereleases=True)

        assert success is True
        run_mock.assert_awaited_once()
        await_args = run_mock.await_args
        assert await_args is not None
        assert await_args.args[0] == (
            "uv tool install -U deepagents-code --prerelease allow"
        )

    async def test_perform_upgrade_allows_prereleases_for_target_dependency(
        self, cache_file
    ) -> None:
        """A stable target that pins an alpha dependency opts uv into prereleases."""
        cache_file.write_text(
            json.dumps(
                {
                    "release_requires_prereleases": {"1.1.0": True},
                    "checked_at": time.time(),
                }
            ),
            encoding="utf-8",
        )
        with (
            patch("deepagents_code.update_check.__version__", "1.0.0"),
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value=frozenset(),
            ),
            patch(
                "deepagents_code.update_check._uv_tool_python",
                return_value=None,
            ),
            patch(
                "deepagents_code.update_check._uv_tool_with_packages",
                return_value=(),
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as run_mock,
        ):
            success, _output = await perform_upgrade(target_version="1.1.0")

        assert success is True
        run_mock.assert_awaited_once()
        await_args = run_mock.await_args
        assert await_args is not None
        assert await_args.args[0] == (
            "uv tool install -U deepagents-code==1.1.0 --prerelease allow"
        )

    async def test_perform_upgrade_fallback_pins_target_with_prerelease_deps(
        self, cache_file
    ) -> None:
        """The bare fallback also avoids floating stable targets to app prereleases."""
        cache_file.write_text(
            json.dumps(
                {
                    "release_requires_prereleases": {"1.1.0": True},
                    "checked_at": time.time(),
                }
            ),
            encoding="utf-8",
        )
        with (
            patch("deepagents_code.update_check.__version__", "1.0.0"),
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                side_effect=ExtrasIntrospectionError("metadata unreadable"),
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as run_mock,
        ):
            success, _output = await perform_upgrade(target_version="1.1.0")

        assert success is True
        run_mock.assert_awaited_once()
        await_args = run_mock.await_args
        assert await_args is not None
        assert await_args.args[0] == (
            "uv tool install -U deepagents-code==1.1.0 --prerelease allow"
        )

    async def test_perform_upgrade_follows_installed_prerelease_channel(self) -> None:
        """Omitted pre-release preference follows an installed pre-release."""
        with (
            patch("deepagents_code.update_check.__version__", "1.0.0rc1"),
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value=frozenset(),
            ),
            patch(
                "deepagents_code.update_check._uv_tool_python",
                return_value=None,
            ),
            patch(
                "deepagents_code.update_check._uv_tool_with_packages",
                return_value=(),
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as run_mock,
        ):
            success, _output = await perform_upgrade()

        assert success is True
        run_mock.assert_awaited_once()
        await_args = run_mock.await_args
        assert await_args is not None
        assert await_args.args[0] == (
            "uv tool install -U deepagents-code --prerelease allow"
        )

    async def test_perform_upgrade_uses_unpinned_uv_install_by_default(self) -> None:
        """Stable upgrades shell out to `uv tool install -U`, not `uv tool upgrade`.

        `uv tool upgrade` respects the receipt's requirement string, so a
        previously-pinned install (e.g. via `DEEPAGENTS_CODE_VERSION` or a
        prior dependency refresh that wrote `==<version>` into the receipt)
        would silently keep the user on the old version. Using `uv tool
        install -U deepagents-code` (no version) rewrites the receipt to an
        unpinned requirement and re-resolves to the latest release.
        """
        with (
            patch("deepagents_code.update_check.__version__", "1.0.0"),
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value=frozenset(),
            ),
            patch(
                "deepagents_code.update_check._uv_tool_python",
                return_value=None,
            ),
            patch(
                "deepagents_code.update_check._uv_tool_with_packages",
                return_value=(),
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as run_mock,
        ):
            success, _output = await perform_upgrade()

        assert success is True
        run_mock.assert_awaited_once()
        await_args = run_mock.await_args
        assert await_args is not None
        assert await_args.args[0] == "uv tool install -U deepagents-code"

    async def test_perform_upgrade_preserves_installed_extras(self) -> None:
        """An upgrade must not silently drop the user's installed extras.

        The unpinned-install fix to the receipt-pin bug could otherwise
        reinstall a bare `deepagents-code` and remove every extra the user
        had set up. Receipt-aware command building keeps them in the
        requirement so they survive the reinstall.
        """
        with (
            patch("deepagents_code.update_check.__version__", "1.0.0"),
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value=frozenset({"quickjs", "nvidia"}),
            ),
            patch(
                "deepagents_code.update_check._uv_tool_python",
                return_value=None,
            ),
            patch(
                "deepagents_code.update_check._uv_tool_with_packages",
                return_value=(),
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as run_mock,
        ):
            success, _output = await perform_upgrade()

        assert success is True
        run_mock.assert_awaited_once()
        await_args = run_mock.await_args
        assert await_args is not None
        assert await_args.args[0] == (
            "uv tool install -U 'deepagents-code[nvidia,quickjs]'"
        )

    async def test_perform_upgrade_falls_back_when_receipt_introspection_fails(
        self,
    ) -> None:
        """Receipt failures must not block the upgrade — fall back to bare.

        Dropping extras is bad, but silently keeping the user pinned to an
        old version is worse. The fallback path runs the bare upgrade
        command rather than refusing the upgrade outright.
        """
        with (
            patch("deepagents_code.update_check.__version__", "1.0.0"),
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                side_effect=ExtrasIntrospectionError("metadata unreadable"),
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as run_mock,
        ):
            success, _output = await perform_upgrade()

        assert success is True
        run_mock.assert_awaited_once()
        await_args = run_mock.await_args
        assert await_args is not None
        assert await_args.args[0] == "uv tool install -U deepagents-code"

    async def test_perform_upgrade_fallback_warns_user_about_dropped_extras(
        self,
    ) -> None:
        """The bare fallback surfaces the extras caveat to the user, not just logs.

        When receipt introspection fails, `perform_upgrade` still upgrades via
        the bare command but may drop extras / `--with` packages. The user's
        only window into the upgrade is the progress stream, so the caveat must
        be emitted there; a log-only warning is invisible in the TUI and the
        user would discover the missing extra later as an unrelated-looking
        import error.
        """
        progress_lines: list[str] = []
        with (
            patch("deepagents_code.update_check.__version__", "1.0.0"),
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                side_effect=ExtrasIntrospectionError("metadata unreadable"),
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
        ):
            success, _output = await perform_upgrade(progress=progress_lines.append)

        assert success is True
        assert any("may not carry over" in line for line in progress_lines)

    async def test_perform_upgrade_refuses_prerelease_for_brew(self) -> None:
        """Pre-release channel switching is only safe for uv tool installs."""
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="brew",
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
            ) as run_mock,
        ):
            success, output = await perform_upgrade(include_prereleases=True)

        assert success is False
        assert "Pre-release updates aren't supported for this install" in output
        # The refusal must short-circuit before shelling out to `brew`.
        run_mock.assert_not_awaited()

    def test_upgrade_command_prerelease(self) -> None:
        """Manual fallback command includes uv's pre-release strategy.

        Uses `uv tool install -U` (not `uv tool upgrade`): see the docstring
        on `_UPGRADE_COMMANDS` for why we avoid the receipt-respecting
        `upgrade` form.
        """
        assert (
            upgrade_command(include_prereleases=True)
            == "uv tool install -U deepagents-code --prerelease allow"
        )

    def test_upgrade_command_pins_target_with_prerelease_deps(self) -> None:
        """Manual fallback pins root dcode when prerelease deps are allowed."""
        assert (
            upgrade_command(
                include_prereleases=True,
                version="1.1.0",
            )
            == "uv tool install -U deepagents-code==1.1.0 --prerelease allow"
        )

    def test_dependency_refresh_command_pins_current_version(
        self, tmp_path, monkeypatch
    ) -> None:
        """Dependency refresh keeps dcode on the running version."""
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        with patch(
            "deepagents_code.extras_info.installed_extra_names",
            return_value=frozenset(),
        ):
            assert (
                dependency_refresh_command(version="1.2.3")
                == "uv tool install -U deepagents-code==1.2.3"
            )

    def test_dependency_refresh_command_preserves_extras(
        self, tmp_path, monkeypatch
    ) -> None:
        """Dependency refresh must not drop already-installed extras."""
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        with patch(
            "deepagents_code.extras_info.installed_extra_names",
            return_value=frozenset({"quickjs", "nvidia"}),
        ):
            assert (
                dependency_refresh_command(
                    version="1.2.3",
                    include_prereleases=True,
                )
                == "uv tool install -U "
                "'deepagents-code[nvidia,quickjs]==1.2.3' --prerelease allow"
            )

    def test_dependency_refresh_command_preserves_with_packages(
        self, tmp_path, monkeypatch
    ) -> None:
        """Dependency refresh must not drop packages installed via `--with`."""
        _write_uv_receipt(
            tmp_path,
            (
                '{ name = "deepagents-code" }, '
                '{ name = "langchain-custom" }, '
                '{ name = "langchain.another_provider" }'
            ),
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))

        with patch(
            "deepagents_code.extras_info.installed_extra_names",
            return_value=frozenset(),
        ):
            assert dependency_refresh_command(version="1.2.3") == (
                "uv tool install -U deepagents-code==1.2.3 "
                "--with langchain-custom --with langchain.another_provider"
            )

    def test_dependency_refresh_command_preserves_uv_python(
        self, tmp_path, monkeypatch
    ) -> None:
        """Dependency refresh must keep uv's recorded interpreter selection."""
        _write_uv_receipt(
            tmp_path,
            '{ name = "deepagents-code" }',
            python="3.13",
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))

        with patch(
            "deepagents_code.extras_info.installed_extra_names",
            return_value=frozenset(),
        ):
            assert dependency_refresh_command(version="1.2.3") == (
                "uv tool install -U --python 3.13 deepagents-code==1.2.3"
            )

    def test_dependency_refresh_command_quotes_uv_python(
        self, tmp_path, monkeypatch
    ) -> None:
        """Recorded interpreter paths are shell-quoted before execution."""
        _write_uv_receipt(
            tmp_path,
            '{ name = "deepagents-code" }, { name = "langchain-custom" }',
            python="/opt/Python 3.13/bin/python",
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))

        with patch(
            "deepagents_code.extras_info.installed_extra_names",
            return_value=frozenset(),
        ):
            assert dependency_refresh_command(version="1.2.3") == (
                "uv tool install -U --python '/opt/Python 3.13/bin/python' "
                "deepagents-code==1.2.3 --with langchain-custom"
            )

    def test_dependency_refresh_command_refuses_malformed_receipt(
        self, tmp_path, monkeypatch
    ) -> None:
        """Malformed uv receipts must not silently drop `--with` packages."""
        tmp_path.joinpath("uv-receipt.toml").write_text(
            "[tool\nrequirements = []\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))

        with (
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value=frozenset(),
            ),
            pytest.raises(ToolRequirementIntrospectionError, match="Could not read"),
        ):
            dependency_refresh_command(version="1.2.3")

    def test_dependency_refresh_command_refuses_unpreservable_with_requirement(
        self, tmp_path, monkeypatch
    ) -> None:
        """Unsupported receipt entries are refused instead of rewritten lossy."""
        _write_uv_receipt(
            tmp_path,
            (
                '{ name = "deepagents-code" }, '
                '{ name = "langchain-custom", editable = "/tmp/pkg" }'
            ),
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))

        with (
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value=frozenset(),
            ),
            pytest.raises(
                ToolRequirementIntrospectionError,
                match="cannot be preserved automatically",
            ),
        ):
            dependency_refresh_command(version="1.2.3")

    def test_dependency_refresh_command_invalid_metadata_extra_reraised(
        self, tmp_path, monkeypatch
    ) -> None:
        """Malformed metadata extras surface through the typed error contract."""
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        with (
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value=frozenset({"not a valid extra"}),
            ),
            pytest.raises(ExtrasIntrospectionError),
        ):
            dependency_refresh_command(version="1.2.3")

    def test_dependency_refresh_dry_run_command_targets_current_python(
        self, tmp_path, monkeypatch
    ) -> None:
        """Dry-run planning resolves against the running tool environment."""
        _write_uv_receipt(
            tmp_path,
            '{ name = "deepagents-code" }, { name = "langchain-custom" }',
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        with patch(
            "deepagents_code.extras_info.installed_extra_names",
            return_value=frozenset({"quickjs"}),
        ):
            assert dependency_refresh_dry_run_command(
                version="1.2.3",
                include_prereleases=True,
                python="/opt/Dcode Python/bin/python",
            ) == (
                "uv pip install --dry-run --python "
                "'/opt/Dcode Python/bin/python' -U "
                "'deepagents-code[quickjs]==1.2.3' langchain-custom "
                "--prerelease allow"
            )

    async def test_perform_dependency_refresh_dry_run_uses_pinned_uv_pip_command(
        self,
    ) -> None:
        """Dependency dry run shells out without mutating the tool environment."""
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch("shutil.which", return_value="/usr/bin/uv"),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value=frozenset(),
            ),
            patch(
                "deepagents_code.update_check._uv_tool_with_packages",
                return_value=(),
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as run_mock,
        ):
            success, _output = await perform_dependency_refresh_dry_run()

        assert success is True
        run_mock.assert_awaited_once()
        await_args = run_mock.await_args
        assert await_args is not None
        assert await_args.args[0] == (
            f"uv pip install --dry-run --python {shlex.quote(sys.executable)} "
            f"-U deepagents-code=={__version__}"
        )

    async def test_perform_dependency_refresh_uses_pinned_uv_command(self) -> None:
        """Dependency refresh shells out without allowing a dcode version bump."""
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch("shutil.which", return_value="/usr/bin/uv"),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value=frozenset(),
            ),
            patch(
                "deepagents_code.update_check._uv_tool_with_packages",
                return_value=(),
            ),
            patch(
                "deepagents_code.update_check._uv_tool_python",
                return_value=None,
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as run_mock,
        ):
            success, _output = await perform_dependency_refresh()

        assert success is True
        run_mock.assert_awaited_once()
        await_args = run_mock.await_args
        assert await_args is not None
        assert await_args.args[0] == (
            f"uv tool install -U deepagents-code=={__version__}"
        )

    async def test_perform_dependency_refresh_reports_with_package_errors(
        self,
    ) -> None:
        """Refresh refuses rather than dropping unknown `--with` packages."""
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch("shutil.which", return_value="/usr/bin/uv"),
            patch(
                "deepagents_code.update_check.dependency_refresh_command",
                side_effect=ToolRequirementIntrospectionError("receipt broken"),
            ),
        ):
            success, output = await perform_dependency_refresh()

        assert success is False
        assert "ToolRequirementIntrospectionError" in output
        assert "receipt broken" in output

    async def test_perform_dependency_refresh_refuses_brew(self) -> None:
        """Brew cannot refresh deps without taking the app formula update."""
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="brew",
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
            ) as run_mock,
        ):
            success, output = await perform_dependency_refresh()

        assert success is False
        assert "dependency-only refresh is not supported" in output
        run_mock.assert_not_awaited()

    async def test_perform_dependency_refresh_refuses_editable(self) -> None:
        """Editable installs can't be re-resolved as a tool environment."""
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="unknown",
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
            ) as run_mock,
        ):
            success, output = await perform_dependency_refresh()

        assert success is False
        assert "Editable install detected" in output
        run_mock.assert_not_awaited()

    async def test_perform_dependency_refresh_refuses_other(self) -> None:
        """An unrecognized install method is refused, not guessed at."""
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="other",
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
            ) as run_mock,
        ):
            success, output = await perform_dependency_refresh()

        assert success is False
        assert "Unsupported install method detected" in output
        run_mock.assert_not_awaited()

    async def test_perform_dependency_refresh_refuses_when_uv_missing(self) -> None:
        """A uv-managed install still needs `uv` on PATH to refresh."""
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch("shutil.which", return_value=None),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
            ) as run_mock,
        ):
            success, output = await perform_dependency_refresh()

        assert success is False
        assert "`uv` not found on PATH." in output
        run_mock.assert_not_awaited()

    def test_prerelease_upgrade_supported_for_uv(self) -> None:
        """The uv install method can be steered onto the pre-release channel."""
        supported, reason = prerelease_upgrade_supported("uv")

        assert supported is True
        assert reason is None

    @pytest.mark.parametrize("method", ["brew", "other", "unknown"])
    def test_prerelease_upgrade_unsupported_for_non_uv(
        self,
        method: InstallMethod,
    ) -> None:
        """Non-uv installs are refused with a user-facing reason."""
        supported, reason = prerelease_upgrade_supported(method)

        assert supported is False
        assert reason is not None
        assert "aren't supported for this install" in reason


class TestUpgradeInstallCommand:
    """Direct coverage for the receipt-aware unpinned-upgrade command builder.

    `perform_upgrade`'s uv path delegates to `upgrade_install_command`, but its
    tests stub `_uv_tool_python`/`_uv_tool_with_packages` to empty, so the
    `--python` and `--with` assembly branches are never exercised through this
    function there. The structurally-similar `dependency_refresh_command` has
    its own coverage, but it is a different function — a `shlex.quote` slip in
    this builder would pass every `perform_upgrade` test. These pin the command
    string end-to-end against a real receipt.
    """

    def test_unpinned_bare_command(self, tmp_path, monkeypatch) -> None:
        """No extras, no `--with`, no recorded python → the bare unpinned form.

        The version pin is *always* stripped (unlike `dependency_refresh_command`),
        because clearing a stale receipt pin is the entire point of routing
        `/update` through this builder.
        """
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        with patch(
            "deepagents_code.extras_info.installed_extra_names",
            return_value=frozenset(),
        ):
            assert upgrade_install_command() == "uv tool install -U deepagents-code"

    def test_pins_target_version_when_requested(self, tmp_path, monkeypatch) -> None:
        """Target pins prevent prerelease dependency mode from floating the app."""
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        with patch(
            "deepagents_code.extras_info.installed_extra_names",
            return_value=frozenset({"openai"}),
        ):
            assert upgrade_install_command(
                version="1.1.0",
                include_prereleases=True,
            ) == (
                "uv tool install -U 'deepagents-code[openai]==1.1.0' --prerelease allow"
            )

    def test_preserves_extras_and_prerelease(self, tmp_path, monkeypatch) -> None:
        """Installed extras survive the unpinned reinstall; prerelease opt-in too."""
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        with patch(
            "deepagents_code.extras_info.installed_extra_names",
            return_value=frozenset({"quickjs", "nvidia"}),
        ):
            assert upgrade_install_command(include_prereleases=True) == (
                "uv tool install -U 'deepagents-code[nvidia,quickjs]' "
                "--prerelease allow"
            )

    def test_preserves_with_packages(self, tmp_path, monkeypatch) -> None:
        """Packages installed via `--with` must survive the unpinned reinstall."""
        _write_uv_receipt(
            tmp_path,
            (
                '{ name = "deepagents-code" }, '
                '{ name = "langchain-custom" }, '
                '{ name = "langchain.another_provider" }'
            ),
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        with patch(
            "deepagents_code.extras_info.installed_extra_names",
            return_value=frozenset(),
        ):
            assert upgrade_install_command() == (
                "uv tool install -U deepagents-code "
                "--with langchain-custom --with langchain.another_provider"
            )

    def test_quotes_uv_python(self, tmp_path, monkeypatch) -> None:
        """A recorded interpreter path with spaces is shell-quoted, not split.

        This is the branch `perform_upgrade`'s tests never reach (they stub
        `_uv_tool_python` to `None`). A dropped `shlex.quote` here would shell
        out to a broken, word-split `--python` argument.
        """
        _write_uv_receipt(
            tmp_path,
            '{ name = "deepagents-code" }, { name = "langchain-custom" }',
            python="/opt/Python 3.13/bin/python",
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        with patch(
            "deepagents_code.extras_info.installed_extra_names",
            return_value=frozenset(),
        ):
            assert upgrade_install_command() == (
                "uv tool install -U --python '/opt/Python 3.13/bin/python' "
                "deepagents-code --with langchain-custom"
            )

    def test_propagates_extras_introspection_error(self, tmp_path, monkeypatch) -> None:
        """Unreadable extras metadata propagates rather than silently dropping.

        `perform_upgrade` catches this and falls back to the bare command, but
        the builder itself must surface the failure so that decision stays at
        the caller, matching the docstring's documented contract.
        """
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        with (
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                side_effect=ExtrasIntrospectionError("metadata unreadable"),
            ),
            pytest.raises(ExtrasIntrospectionError),
        ):
            upgrade_install_command()

    def test_invalid_metadata_extra_reraised_as_introspection_error(
        self, tmp_path, monkeypatch
    ) -> None:
        """A malformed extra name from metadata surfaces as the typed error.

        `_dcode_extras_requirement` raises a bare `ValueError` on a PEP
        508-invalid extra name. Since the extras here come from the
        distribution's own metadata, such a name signals malformed metadata —
        the builder re-raises it as `ExtrasIntrospectionError` so `perform_upgrade`
        handles it through its typed fallback rather than relying on a broad
        `ValueError` catch that could also mask an unrelated builder bug.
        """
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        with (
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value=frozenset({"not a valid extra"}),
            ),
            pytest.raises(ExtrasIntrospectionError),
        ):
            upgrade_install_command()


class TestParseDependencyChanges:
    """`parse_dependency_changes` collapses uv's env diff into changes."""

    def test_version_bump_pairs_removed_and_added(self) -> None:
        """A `- old` / `+ new` pair for one package becomes one bump entry."""
        output = (
            "Resolved 120 packages in 12ms\n"
            " - langchain-openai==1.3.2\n"
            " + langchain-openai==1.5.0\n"
            "Installed 1 executable: dcode\n"
        )
        assert parse_dependency_changes(output) == [
            DependencyChange(name="langchain-openai", old="1.3.2", new="1.5.0"),
        ]

    def test_new_package_has_no_old(self) -> None:
        """A lone `+` line is reported as a new package."""
        assert parse_dependency_changes(" + httpx==0.28.1\n") == [
            DependencyChange(name="httpx", old=None, new="0.28.1"),
        ]

    def test_removed_package_has_no_new(self) -> None:
        """A lone `-` line is reported as a removed package."""
        assert parse_dependency_changes(" - httpx==0.28.1\n") == [
            DependencyChange(name="httpx", old="0.28.1", new=None),
        ]

    def test_preserves_first_seen_order(self) -> None:
        """Packages keep the order uv first mentioned them in."""
        output = " - b-pkg==1.0\n + b-pkg==2.0\n - a-pkg==1.0\n + a-pkg==2.0\n"
        names = [change.name for change in parse_dependency_changes(output)]
        assert names == ["b-pkg", "a-pkg"]

    def test_ignores_non_diff_lines(self) -> None:
        """Resolver chatter without `+`/`-` markers is skipped."""
        assert parse_dependency_changes("Resolved 3 packages\nAudited 3\n") == []


class TestFormatDependencyChanges:
    """`format_dependency_changes` renders an aligned summary."""

    def test_empty_returns_empty_string(self) -> None:
        """No changes renders to an empty string."""
        assert format_dependency_changes([]) == ""

    def test_renders_bump_new_and_removed(self) -> None:
        """Each change kind gets its own rendering, column-aligned."""
        changes = [
            DependencyChange(name="langchain-openai", old="1.3.2", new="1.5.0"),
            DependencyChange(name="httpx", old=None, new="0.28.1"),
            DependencyChange(name="old-pkg", old="1.0", new=None),
        ]
        rendered = format_dependency_changes(changes)
        assert "langchain-openai  1.3.2 -> 1.5.0" in rendered
        assert "0.28.1 (new)" in rendered
        assert "1.0 (removed)" in rendered
        assert "httpx             0.28.1 (new)" in rendered


class TestDependencyChangeKind:
    """`DependencyChange.kind` classifies the three legal shapes."""

    def test_bumped_when_both_sides_present(self) -> None:
        """Both `old` and `new` set is an in-place bump."""
        assert DependencyChange(name="a", old="1.0", new="2.0").kind == "bumped"

    def test_added_when_only_new(self) -> None:
        """Only `new` set is a newly added package."""
        assert DependencyChange(name="a", old=None, new="1.0").kind == "added"

    def test_removed_when_only_old(self) -> None:
        """Only `old` set is a removed package."""
        assert DependencyChange(name="a", old="1.0", new=None).kind == "removed"

    def test_empty_shape_is_rejected(self) -> None:
        """`(None, None)` is meaningless and must raise rather than mis-render."""
        with pytest.raises(ValueError, match="neither an old nor new version"):
            _ = DependencyChange(name="a", old=None, new=None).kind


class TestDependencyChangeAnnotations:
    """`parse_dependency_changes` tolerates uv's source annotations."""

    def test_source_annotation_suffix_is_parsed(self) -> None:
        """A non-PyPI source suffix doesn't hide the version change."""
        output = (
            " - example==0.1.0 (from file:///old)\n"
            " + example==0.2.0 (from file:///new)\n"
        )
        assert parse_dependency_changes(output) == [
            DependencyChange(name="example", old="0.1.0", new="0.2.0"),
        ]


class TestDependencyRefreshSupported:
    """`dependency_refresh_supported` gates the dependency-only refresh."""

    def test_uv_is_supported(self) -> None:
        """uv-managed installs can re-resolve dependencies in place."""
        supported, reason = dependency_refresh_supported("uv")

        assert supported is True
        assert reason is None

    @pytest.mark.parametrize(
        ("method", "needle"),
        [
            ("unknown", "Editable install detected"),
            ("brew", "Homebrew install detected"),
            ("other", "Unsupported install method detected"),
        ],
    )
    def test_non_uv_methods_are_refused_with_reason(
        self,
        method: InstallMethod,
        needle: str,
    ) -> None:
        """Each non-uv method is refused with a distinct, user-facing reason."""
        supported, reason = dependency_refresh_supported(method)

        assert supported is False
        assert reason is not None
        assert needle in reason


class TestInstallExtraCommand:
    """`install_extra_command` builds the promoted install-script string."""

    def test_basic(self) -> None:
        """Extras are passed to the install script through its environment."""
        assert (
            install_extras_command(["quickjs"])
            == "curl -LsSf https://langch.in/dcode | "
            "DEEPAGENTS_CODE_EXTRAS=quickjs bash"
        )

    def test_provider_extra(self) -> None:
        assert (
            install_extras_command(["fireworks"])
            == "curl -LsSf https://langch.in/dcode | "
            "DEEPAGENTS_CODE_EXTRAS=fireworks bash"
        )

    def test_installed_extra_names_missing_distribution_returns_empty(self) -> None:
        """Display-only introspection stays forgiving when metadata is absent."""
        assert installed_extra_names("does-not-exist-pkg-xyz-abc") == set()

    def test_install_extra_command_ignores_missing_distribution(self) -> None:
        """Display-only commands tolerate missing distribution metadata."""
        assert (
            install_extra_command("quickjs", distribution_name="missing-dcode-test")
            == "curl -LsSf https://langch.in/dcode | "
            "DEEPAGENTS_CODE_EXTRAS=quickjs bash"
        )

    def test_no_installed_extras_from_clean_metadata(
        self, tmp_path, monkeypatch
    ) -> None:
        """Clean metadata with no installed optional deps is distinct from failure."""
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        _write_dist_info(
            tmp_path,
            "deepagents-code",
            requires=('definitely-absent-dcode-test-quickjs-xyz; extra == "quickjs"',),
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))

        assert installed_extra_names("deepagents-code") == set()
        assert (
            install_extra_command("quickjs") == "curl -LsSf https://langch.in/dcode | "
            "DEEPAGENTS_CODE_EXTRAS=quickjs bash"
        )

    def test_install_extra_command_preserves_detected_extras(
        self, tmp_path, monkeypatch
    ) -> None:
        """Install-script guidance keeps existing extras when metadata reveals them."""
        _write_dist_info(tmp_path, "definitely-present-dcode-test-nvidia")
        _write_dist_info(
            tmp_path,
            "deepagents-code",
            requires=(
                'definitely-present-dcode-test-nvidia; extra == "nvidia"',
                'definitely-absent-dcode-test-quickjs-xyz; extra == "quickjs"',
            ),
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        assert installed_extra_names("deepagents-code") == {"nvidia"}
        assert (
            install_extra_command("quickjs") == "curl -LsSf https://langch.in/dcode | "
            "DEEPAGENTS_CODE_EXTRAS=nvidia,quickjs bash"
        )

    def test_recovery_command_preserves_uv_receipt_context(
        self, tmp_path, monkeypatch
    ) -> None:
        """UV recovery guidance matches the automatic context-preserving install."""
        _write_uv_receipt(
            tmp_path,
            '{ name = "deepagents-code" }, { name = "langchain-custom" }',
            python="/opt/Python 3.13/bin/python",
        )
        _write_dist_info(tmp_path, "definitely-present-dcode-test-nvidia")
        _write_dist_info(
            tmp_path,
            "deepagents-code",
            requires=('definitely-present-dcode-test-nvidia; extra == "nvidia"',),
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setattr(
            "deepagents_code.update_check.detect_install_method", lambda: "uv"
        )

        assert install_extra_recovery_command("quickjs") == (
            "uv tool install --reinstall -U --python '/opt/Python 3.13/bin/python' "
            f"'deepagents-code[nvidia,quickjs]=={__version__}' "
            "--with langchain-custom --prerelease allow"
        )

    def test_recovery_command_uses_script_for_non_uv_without_receipt(
        self, tmp_path, monkeypatch
    ) -> None:
        """Unsupported install recovery does not require uv receipt introspection."""
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }, "bad"')
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        monkeypatch.setattr(
            "deepagents_code.update_check.detect_install_method", lambda: "other"
        )
        monkeypatch.setattr(
            "deepagents_code.extras_info.installed_extra_names",
            lambda _distribution_name="deepagents-code": set(),
        )

        assert install_extra_recovery_command("quickjs") == (
            "curl -LsSf https://langch.in/dcode | DEEPAGENTS_CODE_EXTRAS=quickjs bash"
        )

    def test_uv_install_extra_command_refuses_invalid_metadata(
        self, tmp_path, monkeypatch
    ) -> None:
        """Malformed optional-dependency metadata must not drop existing extras."""
        _write_dist_info(
            tmp_path,
            "deepagents-code",
            requires=("not a valid requirement ; ;",),
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(ExtrasIntrospectionError, match="Could not parse"):
            _install_extra_uv_tool_command(
                "quickjs", distribution_name="deepagents-code"
            )

    def test_uv_install_extra_command_preserves_installed_extras(
        self, tmp_path, monkeypatch
    ) -> None:
        """Installing a new extra keeps already-installed extras selected."""
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        _write_dist_info(tmp_path, "definitely-present-dcode-test-nvidia")
        _write_dist_info(
            tmp_path,
            "deepagents-code",
            requires=(
                'definitely-present-dcode-test-nvidia; extra == "nvidia"',
                'definitely-absent-dcode-test-baseten-xyz; extra == "baseten"',
            ),
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))

        assert installed_extra_names("deepagents-code") == {"nvidia"}
        assert _install_extra_uv_tool_command(
            "baseten", distribution_name="deepagents-code"
        ) == (
            "uv tool install --reinstall -U "
            f"'deepagents-code[baseten,nvidia]=={__version__}' --prerelease allow"
        )

    def test_uv_install_extra_command_dedupes_existing_extra(
        self, tmp_path, monkeypatch
    ) -> None:
        """Installing an already-present extra does not duplicate it."""
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        _write_dist_info(tmp_path, "definitely-present-dcode-test-nvidia")
        _write_dist_info(
            tmp_path,
            "deepagents-code",
            requires=('definitely-present-dcode-test-nvidia; extra == "nvidia"',),
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))

        assert _install_extra_uv_tool_command(
            "nvidia", distribution_name="deepagents-code"
        ) == (
            "uv tool install --reinstall -U "
            f"'deepagents-code[nvidia]=={__version__}' --prerelease allow"
        )

    def test_uv_install_extra_command_drops_composite_extras(
        self, tmp_path, monkeypatch
    ) -> None:
        """Composite extras are not echoed back into uv reinstall commands."""
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        _write_dist_info(tmp_path, "definitely-present-dcode-test-nvidia")
        _write_dist_info(tmp_path, "definitely-present-dcode-test-openai")
        _write_dist_info(
            tmp_path,
            "deepagents-code",
            requires=(
                'definitely-present-dcode-test-nvidia; extra == "nvidia"',
                'definitely-present-dcode-test-openai; extra == "all-providers"',
            ),
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))

        assert installed_extra_names("deepagents-code") == {"nvidia"}
        assert _install_extra_uv_tool_command(
            "baseten", distribution_name="deepagents-code"
        ) == (
            "uv tool install --reinstall -U "
            f"'deepagents-code[baseten,nvidia]=={__version__}' --prerelease allow"
        )

    def test_uv_install_extra_command_preserves_receipt_python_and_with_packages(
        self, tmp_path, monkeypatch
    ) -> None:
        """Installing an extra preserves the uv tool interpreter and `--with` deps."""
        _write_uv_receipt(
            tmp_path,
            '{ name = "deepagents-code" }, { name = "langchain-custom" }',
            python="/opt/Python 3.13/bin/python",
        )
        _write_dist_info(tmp_path, "definitely-present-dcode-test-nvidia")
        _write_dist_info(
            tmp_path,
            "deepagents-code",
            requires=('definitely-present-dcode-test-nvidia; extra == "nvidia"',),
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))

        command = _install_extra_uv_tool_command(
            "baseten", distribution_name="deepagents-code"
        )

        assert command == (
            "uv tool install --reinstall -U --python '/opt/Python 3.13/bin/python' "
            f"'deepagents-code[baseten,nvidia]=={__version__}' "
            "--with langchain-custom --prerelease allow"
        )

    def test_sorts_extras_deterministically(self) -> None:
        assert (
            install_extras_command({"quickjs", "baseten", "nvidia"})
            == "curl -LsSf https://langch.in/dcode | "
            "DEEPAGENTS_CODE_EXTRAS=baseten,nvidia,quickjs bash"
        )

    def test_rejects_shell_metacharacters(self) -> None:
        assert not is_valid_extra_name("quickjs']; touch /tmp/pwned; '")
        with pytest.raises(ValueError, match="Invalid extra name"):
            install_extra_command("quickjs']; touch /tmp/pwned; '")
        with pytest.raises(ValueError, match="Invalid extra name"):
            install_extras_command(["quickjs", "bad;name"])


class TestEditableExtraHint:
    """`editable_extra_hint` is the shared editable-install action hint."""

    def test_contains_uv_command_and_bracketed_extra(self) -> None:
        hint = editable_extra_hint("quickjs")
        assert "uv tool install --editable" in hint
        assert "--with 'deepagents-code[quickjs]'" in hint

    def test_extra_is_interpolated_into_brackets(self) -> None:
        # The bracket fragment is load-bearing — Rich-markup call sites
        # must `escape()` this output, so the bracketed extra must always
        # be present in the hint (callers rely on this contract).
        assert "[fireworks]" in editable_extra_hint("fireworks")


class TestInstallPackageCommand:
    """`install_package_command` builds a uv tool package install string."""

    def test_basic_no_extras(self, tmp_path, monkeypatch) -> None:
        """Clean metadata with no extras yields the version-pinned requirement."""
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        _write_dist_info(
            tmp_path,
            "deepagents-code",
            requires=('definitely-absent-dcode-test-quickjs-xyz; extra == "quickjs"',),
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))

        assert install_package_command(
            "langchain-custom", distribution_name="deepagents-code"
        ) == (
            "uv tool install --reinstall -U "
            f"deepagents-code=={__version__} --with langchain-custom "
            "--prerelease allow"
        )

    def test_allows_pep508_name_separators(self, tmp_path, monkeypatch) -> None:
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        _write_dist_info(
            tmp_path,
            "deepagents-code",
            requires=('definitely-absent-dcode-test-quickjs-xyz; extra == "quickjs"',),
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))

        assert install_package_command(
            "langchain.custom_provider", distribution_name="deepagents-code"
        ) == (
            "uv tool install --reinstall -U "
            f"deepagents-code=={__version__} --with langchain.custom_provider "
            "--prerelease allow"
        )

    def test_preserves_installed_extras(self, tmp_path, monkeypatch) -> None:
        """Adding a package keeps already-installed extras selected."""
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        _write_dist_info(tmp_path, "definitely-present-dcode-test-nvidia")
        _write_dist_info(
            tmp_path,
            "deepagents-code",
            requires=(
                'definitely-present-dcode-test-nvidia; extra == "nvidia"',
                'definitely-absent-dcode-test-baseten-xyz; extra == "baseten"',
            ),
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))

        assert installed_extra_names("deepagents-code") == {"nvidia"}
        assert install_package_command(
            "langchain-custom", distribution_name="deepagents-code"
        ) == (
            "uv tool install --reinstall -U "
            f"'deepagents-code[nvidia]=={__version__}' --with langchain-custom "
            "--prerelease allow"
        )

    def test_preserves_receipt_python_and_with_packages(
        self, tmp_path, monkeypatch
    ) -> None:
        """Adding a package keeps uv receipt interpreter and `--with` packages."""
        _write_uv_receipt(
            tmp_path,
            '{ name = "deepagents-code" }, { name = "langchain-first" }',
            python="/opt/Python 3.13/bin/python",
        )
        _write_dist_info(tmp_path, "definitely-present-dcode-test-nvidia")
        _write_dist_info(
            tmp_path,
            "deepagents-code",
            requires=('definitely-present-dcode-test-nvidia; extra == "nvidia"',),
        )
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))

        command = install_package_command(
            "langchain-second", distribution_name="deepagents-code"
        )

        assert command == (
            "uv tool install --reinstall -U --python '/opt/Python 3.13/bin/python' "
            f"'deepagents-code[nvidia]=={__version__}' --with langchain-first "
            "--with langchain-second --prerelease allow"
        )

    def test_does_not_duplicate_existing_receipt_package(
        self, tmp_path, monkeypatch
    ) -> None:
        """Reinstalling an existing package does not emit duplicate `--with` args."""
        _write_uv_receipt(
            tmp_path,
            '{ name = "deepagents-code" }, { name = "langchain-custom" }',
        )
        _write_dist_info(tmp_path, "deepagents-code")
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))

        command = install_package_command(
            "LangChain_Custom", distribution_name="deepagents-code"
        )

        assert command == (
            "uv tool install --reinstall -U "
            f"deepagents-code=={__version__} --with langchain-custom "
            "--prerelease allow"
        )

    def test_pins_prerelease_app_version(self, tmp_path, monkeypatch) -> None:
        """Adding a package to a pre-release install keeps that exact app version."""
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        _write_dist_info(tmp_path, "deepagents-code")
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))

        with patch("deepagents_code.update_check.__version__", "1.0.0a1"):
            command = install_package_command(
                "langchain-custom", distribution_name="deepagents-code"
            )

        assert command == (
            "uv tool install --reinstall -U deepagents-code==1.0.0a1 "
            "--with langchain-custom --prerelease allow"
        )

    def test_stable_install_pins_app_and_allows_prerelease_deps(
        self, tmp_path, monkeypatch
    ) -> None:
        """A stable app reinstall keeps the app pinned while allowing rc deps."""
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        _write_dist_info(tmp_path, "deepagents-code")
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))

        with patch("deepagents_code.update_check.__version__", "1.0.0"):
            command = install_package_command(
                "langchain-custom", distribution_name="deepagents-code"
            )

        assert command == (
            "uv tool install --reinstall -U deepagents-code==1.0.0 "
            "--with langchain-custom --prerelease allow"
        )

    def test_appends_new_with_package_after_sorted_receipt_packages(
        self, tmp_path, monkeypatch
    ) -> None:
        """A new `--with` package is appended after preserved ones, not re-sorted.

        Preserved receipt packages come back sorted, and the new package is
        appended afterward regardless of where it would sort. Using a new name
        (`langchain-alpha`) that sorts *before* the receipt's (`langchain-zeta`)
        distinguishes this append-after-preserved contract from a plain
        alphabetical sort of the union, which the same-order cases cannot.
        """
        _write_uv_receipt(
            tmp_path,
            '{ name = "deepagents-code" }, { name = "langchain-zeta" }',
        )
        _write_dist_info(tmp_path, "deepagents-code")
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))

        command = install_package_command(
            "langchain-alpha", distribution_name="deepagents-code"
        )

        assert command == (
            "uv tool install --reinstall -U "
            f"deepagents-code=={__version__} --with langchain-zeta "
            "--with langchain-alpha --prerelease allow"
        )

    def test_unpreservable_receipt_with_requirement_raises(
        self, tmp_path, monkeypatch
    ) -> None:
        """A `--with` requirement uv can't re-express by name aborts the build.

        Exercises the `unsupported_keys` arm through `install_package_command`'s
        newly-added receipt read: a source-pinned `--with` entry (e.g. an
        editable install) carries keys beyond `name`, so it cannot be safely
        re-expressed as a `--with <name>` and must raise rather than be silently
        dropped from the rebuilt command.
        """
        _write_uv_receipt(
            tmp_path,
            '{ name = "deepagents-code" }, '
            '{ name = "langchain-custom", editable = true }',
        )
        _write_dist_info(tmp_path, "deepagents-code")
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(
            ToolRequirementIntrospectionError,
            match="cannot be preserved automatically",
        ):
            install_package_command(
                "langchain-new", distribution_name="deepagents-code"
            )

    def test_refuses_missing_distribution(self) -> None:
        """Reinstalls must not drop extras when metadata is unavailable."""
        with pytest.raises(ExtrasIntrospectionError, match="cannot preserve"):
            install_package_command(
                "langchain-custom", distribution_name="missing-dcode-test"
            )

    def test_refuses_invalid_metadata(self, tmp_path, monkeypatch) -> None:
        """Malformed optional-dependency metadata must not drop existing extras."""
        _write_dist_info(
            tmp_path,
            "deepagents-code",
            requires=("not a valid requirement ; ;",),
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(ExtrasIntrospectionError, match="Could not parse"):
            install_package_command(
                "langchain-custom", distribution_name="deepagents-code"
            )

    def test_rejects_shell_metacharacters(self) -> None:
        """A bad package name raises before extras introspection runs.

        Validation precedes the distribution lookup, so the rejection holds
        regardless of metadata availability.
        """
        with pytest.raises(ValueError, match="Invalid package name"):
            install_package_command("langchain-custom; touch /tmp/pwned")


class TestPerformInstallExtra:
    """`perform_install_extra` execution paths."""

    async def test_editable_install_refuses(self) -> None:
        """Editable installs cannot accept extras via uv tool install."""
        with patch(
            "deepagents_code.update_check.detect_install_method",
            return_value="unknown",
        ):
            success, output = await perform_install_extra("quickjs")
        assert success is False
        assert "Editable install" in output
        assert "uv tool install --editable" in output
        assert "--with 'deepagents-code[quickjs]'" in output

    async def test_brew_install_refuses(self) -> None:
        """Homebrew formula doesn't expose extras."""
        with patch(
            "deepagents_code.update_check.detect_install_method",
            return_value="brew",
        ):
            success, output = await perform_install_extra("quickjs")
        assert success is False
        assert "Homebrew" in output

    async def test_other_install_refuses(self) -> None:
        """Unknown non-editable installs cannot be updated through uv tool."""
        with patch(
            "deepagents_code.update_check.detect_install_method",
            return_value="other",
        ):
            success, output = await perform_install_extra("quickjs")
        assert success is False
        assert "Unsupported install method" in output

    @pytest.mark.parametrize(
        ("method", "needle"),
        [
            ("brew", "Homebrew install detected"),
            ("other", "Unsupported install method detected"),
        ],
    )
    async def test_non_uv_install_refuses_before_reading_uv_receipt(
        self,
        method: InstallMethod,
        needle: str,
        tmp_path,
        monkeypatch,
    ) -> None:
        """Unsupported install guidance wins over uv receipt introspection errors."""
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }, "bad"')
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        # Isolate from the host's installed extras so the script command is
        # deterministic — install_extra_command merges real distribution
        # metadata, which would otherwise leak the dev env's extras in.
        monkeypatch.setattr(
            "deepagents_code.extras_info.installed_extra_names",
            lambda _distribution_name="deepagents-code": set(),
        )
        with patch(
            "deepagents_code.update_check.detect_install_method",
            return_value=method,
        ):
            success, output = await perform_install_extra("quickjs")
        assert success is False
        assert needle in output
        assert "ToolRequirementIntrospectionError" not in output
        assert "curl -LsSf https://langch.in/dcode" in output
        assert "DEEPAGENTS_CODE_EXTRAS=quickjs bash" in output

    async def test_invalid_extra_refuses_before_detecting_install(self) -> None:
        """Malformed forced extras must never reach command construction."""
        with patch(
            "deepagents_code.update_check.detect_install_method",
        ) as detect:
            success, output = await perform_install_extra("quickjs']; echo nope; '")
        assert success is False
        assert "Invalid extra name" in output
        detect.assert_not_called()

    async def test_uv_install_runs(self, tmp_path) -> None:
        """`uv` method runs the subprocess and returns success."""
        log_path = tmp_path / "install.log"
        # Inject a no-op command in place of the real uv tool install so the
        # subprocess actually exits 0 without touching the environment.
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check.shutil.which",
                return_value="/usr/bin/uv",
            ),
            patch(
                "deepagents_code.update_check._install_extra_uv_tool_command",
                return_value="printf 'ok\\n'",
            ),
        ):
            success, output = await perform_install_extra("quickjs", log_path=log_path)
        assert success is True
        assert output == "ok"

    async def test_uv_receipt_failure_is_reported(self, tmp_path, monkeypatch) -> None:
        """A malformed uv receipt is reported instead of dropping install context."""
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }, "bad"')
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check.shutil.which",
                return_value="/usr/bin/uv",
            ),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value=frozenset(),
            ),
        ):
            success, output = await perform_install_extra("quickjs")
        assert success is False
        assert "ToolRequirementIntrospectionError" in output
        assert "non-table requirement" in output

    async def test_uv_missing_returns_actionable_error(self) -> None:
        """When `uv` is not on PATH, surface a clear error before exec."""
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check.shutil.which",
                return_value=None,
            ),
        ):
            success, output = await perform_install_extra("quickjs")
        assert success is False
        assert "uv" in output
        assert "not found" in output


class TestIsValidPackageName:
    """`is_valid_package_name` accepts PEP 508 names, rejects the rest."""

    def test_accepts_plain_and_separated_names(self) -> None:
        assert is_valid_package_name("langchain-custom")
        assert is_valid_package_name("langchain.custom_provider")

    def test_rejects_shell_metacharacters(self) -> None:
        assert not is_valid_package_name("langchain-custom; touch /tmp/pwned")

    def test_rejects_option_injection_leading_dash(self) -> None:
        """A leading dash would smuggle uv options into `--with <name>`.

        The command is `uv tool install --reinstall -U deepagents-code==<version>
        --with <name>`; a name
        like `-rreqs.txt` or `--editable` would be parsed by uv as a flag, not a
        package. The validator must reject these regardless of `--force`/`--yes`.
        """
        assert not is_valid_package_name("-rreqs.txt")
        assert not is_valid_package_name("--force")
        assert not is_valid_package_name("-e.")

    def test_rejects_boundary_separators_and_whitespace(self) -> None:
        """Leading/trailing separators and internal whitespace are rejected."""
        for bad in (".foo", "foo.", "-foo", "foo-", "_foo", "foo_", "foo bar"):
            assert not is_valid_package_name(bad), bad

    def test_rejects_non_ascii(self) -> None:
        r"""The pattern is ASCII-only; a `\w`-based regex would wrongly accept."""
        assert not is_valid_package_name("foöbar")

    def test_rejects_empty(self) -> None:
        assert not is_valid_package_name("")


class TestEditablePackageHint:
    """`editable_package_hint` names the package without a raw `uv` command."""

    def test_names_package_without_uv_command(self) -> None:
        hint = editable_package_hint("langchain-custom")
        assert "langchain-custom" in hint
        # We intentionally don't surface raw `uv tool` commands to the user.
        assert "uv tool" not in hint


class TestPerformInstallPackage:
    """`perform_install_package` execution paths."""

    async def test_editable_install_refuses(self) -> None:
        """Editable installs cannot accept packages via uv tool install."""
        with patch(
            "deepagents_code.update_check.detect_install_method",
            return_value="unknown",
        ):
            success, output = await perform_install_package("langchain-custom")
        assert success is False
        assert "Editable install" in output
        assert "langchain-custom" in output
        # No raw `uv tool` command is surfaced to the user.
        assert "uv tool" not in output

    async def test_brew_install_refuses(self) -> None:
        """Homebrew formula can't add packages to the tool env."""
        with patch(
            "deepagents_code.update_check.detect_install_method",
            return_value="brew",
        ):
            success, output = await perform_install_package("langchain-custom")
        assert success is False
        assert "Homebrew" in output

    async def test_other_install_refuses(self) -> None:
        """Unknown non-editable installs cannot be updated through uv tool."""
        with patch(
            "deepagents_code.update_check.detect_install_method",
            return_value="other",
        ):
            success, output = await perform_install_package("langchain-custom")
        assert success is False
        assert "Unsupported install method" in output

    async def test_invalid_package_refuses_before_detecting_install(self) -> None:
        """Malformed package names must never reach command construction."""
        with patch(
            "deepagents_code.update_check.detect_install_method",
        ) as detect:
            success, output = await perform_install_package("custom; echo nope")
        assert success is False
        assert "Invalid package name" in output
        detect.assert_not_called()

    async def test_uv_install_runs(self, tmp_path) -> None:
        """`uv` method runs the subprocess and returns success."""
        log_path = tmp_path / "install.log"
        # Inject a no-op command in place of the real uv tool install so the
        # subprocess actually exits 0 without touching the environment.
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check.shutil.which",
                return_value="/usr/bin/uv",
            ),
            patch(
                "deepagents_code.update_check.install_package_command",
                return_value="printf 'ok\\n'",
            ),
        ):
            success, output = await perform_install_package(
                "langchain-custom", log_path=log_path
            )
        assert success is True
        assert output == "ok"

    async def test_uv_missing_returns_actionable_error(self) -> None:
        """When `uv` is not on PATH, surface a clear error before exec."""
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check.shutil.which",
                return_value=None,
            ),
        ):
            success, output = await perform_install_package("langchain-custom")
        assert success is False
        assert "uv" in output
        assert "not found" in output

    async def test_extras_introspection_failure_is_reported_and_logged(
        self, caplog
    ) -> None:
        """Unreadable distribution metadata surfaces as a reported, logged error.

        Guards the `ExtrasIntrospectionError` arm distinctly from the
        `ValueError` arm: a narrowing back to `except ValueError` would let the
        error escape unhandled, and dropping the log would erase the only
        breadcrumb for what is an environment-corruption signal.
        """
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check.shutil.which",
                return_value="/usr/bin/uv",
            ),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                side_effect=ExtrasIntrospectionError("metadata unreadable"),
            ),
            caplog.at_level(logging.WARNING, logger="deepagents_code.update_check"),
        ):
            success, output = await perform_install_package("langchain-custom")
        assert success is False
        assert "ExtrasIntrospectionError" in output
        assert "metadata unreadable" in output
        assert "introspect installed extras" in caplog.text

    async def test_uv_receipt_failure_is_reported_and_logged(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        """An unreadable uv receipt surfaces as a reported, logged error.

        `install_package_command` now reads the uv tool receipt to preserve the
        interpreter and `--with` packages, so a malformed receipt raises
        `ToolRequirementIntrospectionError`. The executor must report it rather
        than let it escape unhandled — narrowing back to `except
        ExtrasIntrospectionError` would let the error crash the caller.
        """
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }, "bad"')
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check.shutil.which",
                return_value="/usr/bin/uv",
            ),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value=frozenset(),
            ),
            caplog.at_level(logging.WARNING, logger="deepagents_code.update_check"),
        ):
            success, output = await perform_install_package("langchain-custom")
        assert success is False
        assert "ToolRequirementIntrospectionError" in output
        assert "non-table requirement" in output
        assert "uv receipt" in caplog.text

    async def test_invalid_app_version_is_reported(self, tmp_path, monkeypatch) -> None:
        """A malformed app version pin is reported instead of escaping."""
        _write_uv_receipt(tmp_path, '{ name = "deepagents-code" }')
        _write_dist_info(tmp_path, "deepagents-code")
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check.shutil.which",
                return_value="/usr/bin/uv",
            ),
            patch("deepagents_code.update_check.__version__", "not-a-version"),
        ):
            success, output = await perform_install_package("langchain-custom")
        assert success is False
        assert "ValueError" in output
        assert "Invalid deepagents-code version" in output


class TestRunInstallSubprocessFailureModes:
    """Failure-mode coverage routed through `perform_install_extra`.

    Exercises the shared `_run_install_subprocess` helper since it has no
    public entry point of its own.
    """

    async def test_timeout_kills_process(self, tmp_path) -> None:
        """A subprocess that exceeds `_UPGRADE_TIMEOUT` is killed and reported."""
        log_path = tmp_path / "install.log"
        with (
            patch("deepagents_code.update_check._UPGRADE_TIMEOUT", 0.05),
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check.shutil.which",
                return_value="/usr/bin/uv",
            ),
            patch(
                "deepagents_code.update_check._install_extra_uv_tool_command",
                return_value="sleep 5",
            ),
        ):
            success, output = await perform_install_extra("quickjs", log_path=log_path)
        assert success is False
        assert "timed out" in output

    async def test_timeout_kills_process_group_after_shell_exits(
        self, tmp_path
    ) -> None:
        """Timeout cleanup kills descendants after the POSIX shell exits."""
        if os.name != "posix":
            pytest.skip("process groups are POSIX-specific")
        log_path = tmp_path / "install.log"
        with (
            patch("deepagents_code.update_check._UPGRADE_TIMEOUT", 0.05),
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check.shutil.which",
                return_value="/usr/bin/uv",
            ),
            patch(
                "deepagents_code.update_check._install_extra_uv_tool_command",
                return_value="sleep 5 & exit 0",
            ),
            patch("deepagents_code.update_check.os.killpg", wraps=os.killpg) as killpg,
        ):
            success, output = await perform_install_extra("quickjs", log_path=log_path)
        assert success is False
        assert "timed out" in output
        killpg.assert_called_once()
        assert killpg.call_args.args[1] == signal.SIGKILL

    async def test_cancellation_terminates_process_and_propagates(
        self, tmp_path
    ) -> None:
        """Cancelling mid-install kills the process group and re-raises."""
        if os.name != "posix":
            pytest.skip("process groups are POSIX-specific")
        log_path = tmp_path / "install.log"
        started = asyncio.Event()

        def progress(line: str) -> None:
            if "ready" in line:
                started.set()

        with patch("deepagents_code.update_check.os.killpg", wraps=os.killpg) as killpg:
            task = asyncio.ensure_future(
                _run_install_subprocess(
                    "echo ready; sleep 30", progress=progress, log_path=log_path
                )
            )
            # Cancel only once the subprocess is actually running (it emitted a
            # line), so cancellation lands inside the install, not before it.
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        # Cancellation must not be swallowed, and the descendant `sleep` must be
        # killed via the process group rather than orphaned.
        killpg.assert_called_once()
        assert killpg.call_args.args[1] == signal.SIGKILL

    async def test_terminate_falls_back_to_direct_kill_on_permission_error(
        self,
    ) -> None:
        """When `killpg` is denied, the direct child is still reaped."""
        if os.name != "posix":
            pytest.skip("process groups are POSIX-specific")
        proc = await asyncio.create_subprocess_shell(
            "sleep 30",
            stdin=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        with patch(
            "deepagents_code.update_check.os.killpg",
            side_effect=PermissionError,
        ):
            await _terminate_install_process(proc)

        # The EPERM fallback (`proc.kill()`) must have reaped the child.
        assert proc.returncode is not None

    async def test_oserror_includes_exception_detail(self, tmp_path) -> None:
        """An OSError during exec must surface the exception class + message."""
        log_path = tmp_path / "install.log"

        def _raise(*_args: object, **_kwargs: object) -> None:
            raise FileNotFoundError(2, "No such file or directory", "uv")

        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check.shutil.which",
                return_value="/usr/bin/uv",
            ),
            patch(
                "deepagents_code.update_check._install_extra_uv_tool_command",
                return_value=(
                    "uv tool install --reinstall -U 'deepagents-code[quickjs]'"
                ),
            ),
            patch("asyncio.create_subprocess_shell", side_effect=_raise),
        ):
            success, output = await perform_install_extra("quickjs", log_path=log_path)
        assert success is False
        assert "FileNotFoundError" in output
        assert "No such file" in output

    async def test_nonzero_exit_returns_combined_output(self, tmp_path) -> None:
        """A failing subprocess returns False with stderr in the output."""
        log_path = tmp_path / "install.log"
        with (
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch(
                "deepagents_code.update_check.shutil.which",
                return_value="/usr/bin/uv",
            ),
            patch(
                "deepagents_code.update_check._install_extra_uv_tool_command",
                return_value="sh -c 'printf boom 1>&2; exit 1'",
            ),
        ):
            success, output = await perform_install_extra("quickjs", log_path=log_path)
        assert success is False
        assert "boom" in output


def _mock_sdk_pypi_response(
    releases: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
) -> MagicMock:
    """Build a minimal PyPI response for the `deepagents` SDK.

    The SDK lookup reads from the `releases` map (keyed by version) rather
    than `info.version`, so only that field is required.
    """
    releases_data = (
        {ver: [dict(file) for file in files] for ver, files in releases.items()}
        if releases is not None
        else {}
    )
    resp = MagicMock()
    resp.json.return_value = {"releases": releases_data}
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
            "requests.get",
            return_value=_mock_sdk_pypi_response(releases=releases),
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
            "deepagents_code.sessions.format_relative_timestamp", return_value="1w ago"
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

    def test_installed_release_time_cached_on_fresh_fetch(self, cache_file) -> None:
        """The current install's release timestamp is cached for age notices."""
        releases = {
            "2.0.0": [{"filename": "a.tar.gz"}],
            __version__: [{"filename": "installed.tar.gz"}],
        }
        with patch(
            "requests.get",
            return_value=_mock_pypi_response(
                "2.0.0",
                releases=releases,
                release_times={
                    "2.0.0": "2026-04-15T12:00:00Z",
                    __version__: "2026-04-01T12:00:00Z",
                },
            ),
        ):
            get_latest_version()

        data = json.loads(cache_file.read_text())
        assert data["release_times"][__version__] == "2026-04-01T12:00:00Z"

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
        with patch("deepagents_code.update_check.DEFAULT_CONFIG_PATH", path):
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
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch.dict("os.environ", {}, clear=False),
        ):
            import os

            os.environ.pop("DEEPAGENTS_CODE_AUTO_UPDATE", None)
            assert is_auto_update_enabled() is True


class TestIsAutoUpdateEnabled:
    @pytest.fixture
    def config_path(self, tmp_path):
        """Override DEFAULT_CONFIG_PATH to use a temporary file."""
        path = tmp_path / "config.toml"
        with patch("deepagents_code.update_check.DEFAULT_CONFIG_PATH", path):
            yield path

    def test_default_is_true(self, config_path) -> None:  # noqa: ARG002
        """Auto-update defaults to enabled (opt-out)."""
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch.dict("os.environ", {}, clear=False),
        ):
            import os

            os.environ.pop("DEEPAGENTS_CODE_AUTO_UPDATE", None)
            assert is_auto_update_enabled() is True

    def test_env_var_enables(self, config_path) -> None:  # noqa: ARG002
        """DEEPAGENTS_CODE_AUTO_UPDATE=1 enables auto-update."""
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch.dict("os.environ", {"DEEPAGENTS_CODE_AUTO_UPDATE": "1"}),
        ):
            assert is_auto_update_enabled() is True

    def test_env_var_disables(self, config_path) -> None:  # noqa: ARG002
        """DEEPAGENTS_CODE_AUTO_UPDATE=0 opts out of auto-update."""
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch.dict("os.environ", {"DEEPAGENTS_CODE_AUTO_UPDATE": "0"}),
        ):
            assert is_auto_update_enabled() is False

    def test_config_disables(self, config_path) -> None:
        """`[update].auto_update = false` opts out of auto-update."""
        set_auto_update(False)
        assert config_path.exists()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch.dict("os.environ", {}, clear=False),
        ):
            import os

            os.environ.pop("DEEPAGENTS_CODE_AUTO_UPDATE", None)
            assert is_auto_update_enabled() is False

    def test_empty_env_disables(self, config_path, monkeypatch) -> None:  # noqa: ARG002
        """An explicitly-empty env value is treated as falsy (opt-out)."""
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTO_UPDATE", "")
        with patch("deepagents_code.config._is_editable_install", return_value=False):
            assert is_auto_update_enabled() is False

    def test_unrecognized_env_falls_through_to_default(
        self, config_path, monkeypatch, caplog
    ) -> None:
        """A garbage env value is ignored (with a warning) and uses the default.

        Guards the `classify_env_bool(...) is None` branch: a typo'd disable
        attempt must not be mistaken for a real value. With no config written
        it falls through to the opt-out default of `True`.
        """
        assert not config_path.exists()  # no config backs the result
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTO_UPDATE", "ture")
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            caplog.at_level(logging.WARNING, logger="deepagents_code.update_check"),
        ):
            assert is_auto_update_enabled() is True
        assert "expected bool" in caplog.text

    def test_unrecognized_env_falls_through_to_config(
        self, config_path, monkeypatch
    ) -> None:
        """A garbage env value yields to `config.toml` rather than overriding it."""
        set_auto_update(False)
        assert config_path.exists()
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTO_UPDATE", "maybe")
        with patch("deepagents_code.config._is_editable_install", return_value=False):
            assert is_auto_update_enabled() is False

    def test_env_overrides_config_to_disable(self, config_path, monkeypatch) -> None:  # noqa: ARG002
        """A falsy env var wins over `[update].auto_update = true`."""
        set_auto_update(True)
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTO_UPDATE", "0")
        with patch("deepagents_code.config._is_editable_install", return_value=False):
            assert is_auto_update_enabled() is False

    def test_env_overrides_config_to_enable(self, config_path, monkeypatch) -> None:  # noqa: ARG002
        """A truthy env var wins over `[update].auto_update = false`."""
        set_auto_update(False)
        monkeypatch.setenv("DEEPAGENTS_CODE_AUTO_UPDATE", "1")
        with patch("deepagents_code.config._is_editable_install", return_value=False):
            assert is_auto_update_enabled() is True

    def test_editable_install_always_disabled(self, config_path) -> None:
        """Editable installs never auto-update, even with config set."""
        set_auto_update(True)
        assert config_path.exists()
        with patch("deepagents_code.config._is_editable_install", return_value=True):
            assert is_auto_update_enabled() is False

    def test_corrupt_config_fails_closed(
        self, config_path, monkeypatch, caplog
    ) -> None:
        """A present-but-corrupt config disables auto-update despite the default.

        The opt-out default is `True`, but a corrupt `config.toml` may hold an
        explicit `auto_update = false`. Silently re-enabling auto-update (which
        upgrades and re-execs) over an unreadable opt-out would be worse than
        skipping, so a parse error must fail closed rather than fall through to
        the default.
        """
        config_path.write_text("this = is not [valid toml", encoding="utf-8")
        monkeypatch.delenv("DEEPAGENTS_CODE_AUTO_UPDATE", raising=False)
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            caplog.at_level(logging.WARNING, logger="deepagents_code.update_check"),
        ):
            assert is_auto_update_enabled() is False
        assert "disabling auto-update" in caplog.text


class TestAutoUpdateDefaultMigration:
    @pytest.fixture
    def config_path(self, tmp_path):
        """Override DEFAULT_CONFIG_PATH to use a temporary file."""
        path = tmp_path / "config.toml"
        with patch("deepagents_code.update_check.DEFAULT_CONFIG_PATH", path):
            yield path

    @pytest.fixture
    def state_file(self, tmp_path):
        """Override UPDATE_STATE_FILE to use a temporary file."""
        path = tmp_path / "update_state.json"
        with patch("deepagents_code.update_check.UPDATE_STATE_FILE", path):
            yield path

    def test_explicit_config_is_not_default(self, config_path, state_file) -> None:  # noqa: ARG002
        """An explicit config choice counts as explicitly set."""
        set_auto_update(True)
        import os

        os.environ.pop("DEEPAGENTS_CODE_AUTO_UPDATE", None)
        assert is_auto_update_explicitly_set() is True
        assert should_announce_auto_update_default() is False

    def test_explicit_env_is_not_default(self, config_path, state_file) -> None:  # noqa: ARG002
        """A recognized env value counts as explicitly set."""
        with patch.dict("os.environ", {"DEEPAGENTS_CODE_AUTO_UPDATE": "1"}):
            assert is_auto_update_explicitly_set() is True
            assert should_announce_auto_update_default() is False

    def test_implicit_default_announces_once(self, config_path, state_file) -> None:  # noqa: ARG002
        """With no explicit choice, the migration notice fires exactly once."""
        import os

        os.environ.pop("DEEPAGENTS_CODE_AUTO_UPDATE", None)
        assert is_auto_update_explicitly_set() is False
        assert should_announce_auto_update_default() is True
        mark_auto_update_default_acknowledged()
        assert should_announce_auto_update_default() is False

    def test_unrecognized_env_is_not_explicit(self, config_path, state_file) -> None:  # noqa: ARG002
        """A garbage env token does not count as an explicit choice."""
        import os

        os.environ.pop("DEEPAGENTS_CODE_AUTO_UPDATE", None)
        with patch.dict("os.environ", {"DEEPAGENTS_CODE_AUTO_UPDATE": "ture"}):
            assert is_auto_update_explicitly_set() is False
            assert should_announce_auto_update_default() is True

    def test_corrupt_state_refires_notice(self, config_path, state_file) -> None:  # noqa: ARG002
        """A corrupt state file fails open: the one-time notice fires again.

        `_read_update_state` returns `{}` on unreadable JSON, so the
        acknowledgement reads as absent. Re-showing the notice is the safe
        direction (versus silently auto-updating as if it had been seen).
        """
        import os

        os.environ.pop("DEEPAGENTS_CODE_AUTO_UPDATE", None)
        state_file.write_text("{ not valid json", encoding="utf-8")
        assert should_announce_auto_update_default() is True

    def test_corrupt_config_is_not_explicit(self, config_path, state_file) -> None:  # noqa: ARG002
        """A corrupt config reads as 'no explicit choice' for the notice gate.

        `is_auto_update_enabled` fails closed on a corrupt config, so the notice
        gate never re-enables an unreadable opt-out; this documents that
        `is_auto_update_explicitly_set` itself treats an unparseable file as
        absent rather than raising.
        """
        import os

        os.environ.pop("DEEPAGENTS_CODE_AUTO_UPDATE", None)
        config_path.write_text("not [ valid toml", encoding="utf-8")
        assert is_auto_update_explicitly_set() is False

    def test_baseline_acknowledges_implicit_default(
        self,
        config_path,  # noqa: ARG002
        state_file,  # noqa: ARG002
    ) -> None:
        """`_note_install_baseline` pre-acknowledges the implicit-default notice.

        On a fresh install the migration notice has no meaning, so stamping the
        baseline suppresses it.
        """
        import os

        os.environ.pop("DEEPAGENTS_CODE_AUTO_UPDATE", None)
        assert should_announce_auto_update_default() is True
        _note_install_baseline()
        assert should_announce_auto_update_default() is False

    def test_baseline_skips_when_explicitly_set(self, config_path, state_file) -> None:  # noqa: ARG002
        """An explicit preference means no migration scenario, so no stamp.

        With an explicit choice the notice is already suppressed; the baseline
        must not record an acknowledgement. Asserting on state *content* (rather
        than file existence) keeps the test robust to unrelated state another
        code path might write to the same file.
        """
        with patch.dict("os.environ", {"DEEPAGENTS_CODE_AUTO_UPDATE": "1"}):
            _note_install_baseline()
        if state_file.exists():
            state = json.loads(state_file.read_text(encoding="utf-8"))
            assert "auto_update_default_acknowledged" not in state

    def test_baseline_tolerates_write_failure(self, config_path, tmp_path) -> None:  # noqa: ARG002
        """The fresh-install baseline stays fail-soft when the state write fails.

        An unwritable state dir must not crash startup; the stamp is simply
        retried on the next launch (see `should_show_whats_new`).
        """
        import os

        os.environ.pop("DEEPAGENTS_CODE_AUTO_UPDATE", None)
        # Point the state file beneath an existing *file* so the parent
        # `mkdir`/write raises `OSError`, simulating an unwritable state dir.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        with patch(
            "deepagents_code.update_check.UPDATE_STATE_FILE", blocker / "state.json"
        ):
            _note_install_baseline()  # must not raise

    def test_mark_tolerates_write_failure(self, config_path, tmp_path) -> None:  # noqa: ARG002
        """A failed acknowledgement write returns `False` without raising.

        The notice will re-fire next launch (surfaced to the user), but startup
        must not crash because the state directory is unwritable.
        """
        # Point the state file beneath an existing *file* so the parent
        # `mkdir`/write raises `OSError`, simulating an unwritable state dir.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        with patch(
            "deepagents_code.update_check.UPDATE_STATE_FILE", blocker / "state.json"
        ):
            assert mark_auto_update_default_acknowledged() is False


class TestStartupAutoUpdateFailureCooldown:
    @pytest.fixture
    def state_file(self, tmp_path):
        """Override UPDATE_STATE_FILE to use a temporary file."""
        path = tmp_path / "update_state.json"
        with patch("deepagents_code.update_check.UPDATE_STATE_FILE", path):
            yield path

    def test_recent_same_version_skips(self, state_file) -> None:  # noqa: ARG002
        """A recent startup auto-update failure suppresses the same version."""
        with patch("deepagents_code.update_check.time.time", return_value=100.0):
            assert mark_startup_auto_update_failed("2.0.0") is True
        with patch("deepagents_code.update_check.time.time", return_value=101.0):
            assert should_skip_startup_auto_update_after_failure("2.0.0") is True

    def test_different_version_does_not_skip(self, state_file) -> None:  # noqa: ARG002
        """A failure marker is scoped to the failed target version."""
        with patch("deepagents_code.update_check.time.time", return_value=100.0):
            mark_startup_auto_update_failed("2.0.0")
        with patch("deepagents_code.update_check.time.time", return_value=101.0):
            assert should_skip_startup_auto_update_after_failure("2.0.1") is False

    def test_expired_failure_does_not_skip(self, state_file) -> None:  # noqa: ARG002
        """The startup failure cooldown expires after the configured window."""
        with patch("deepagents_code.update_check.time.time", return_value=100.0):
            mark_startup_auto_update_failed("2.0.0")
        with patch(
            "deepagents_code.update_check.time.time",
            return_value=100.0 + CACHE_TTL + 1,
        ):
            assert should_skip_startup_auto_update_after_failure("2.0.0") is False

    def test_clear_removes_matching_failure_marker(self, state_file) -> None:
        """Clearing a matching marker preserves unrelated update state."""
        mark_version_seen("1.0.0")
        mark_startup_auto_update_failed("2.0.0")

        clear_startup_auto_update_failure("2.0.0")

        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["seen_version"] == "1.0.0"
        assert "startup_auto_update_failed_version" not in data
        assert "startup_auto_update_failed_at" not in data

    def test_clear_ignores_different_version(self, state_file) -> None:
        """Clearing a different version leaves the failure marker intact."""
        mark_startup_auto_update_failed("2.0.0")

        clear_startup_auto_update_failure("2.0.1")

        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["startup_auto_update_failed_version"] == "2.0.0"

    def test_corrupted_timestamp_does_not_skip(self, state_file) -> None:
        """A hand-edited non-numeric timestamp must not crash or skip."""
        state_file.write_text(
            json.dumps(
                {
                    "startup_auto_update_failed_version": "2.0.0",
                    "startup_auto_update_failed_at": "not-a-number",
                }
            ),
            encoding="utf-8",
        )
        assert should_skip_startup_auto_update_after_failure("2.0.0") is False


class TestResumeAutoUpdateGracePeriod:
    @pytest.fixture
    def state_file(self, tmp_path):
        """Override UPDATE_STATE_FILE to use a temporary file."""
        path = tmp_path / "update_state.json"
        with patch("deepagents_code.update_check.UPDATE_STATE_FILE", path):
            yield path

    def test_first_resume_starts_grace_period(self, state_file) -> None:
        """The first resumed launch records and uses the grace period."""
        with patch("deepagents_code.update_check.time.time", return_value=100.0):
            assert should_defer_startup_auto_update_for_resume() is True

        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["resume_auto_update_deferred_at"] == pytest.approx(100.0)

    def test_repeated_resume_does_not_extend_grace_period(self, state_file) -> None:
        """Repeated resumes preserve the first deferral timestamp."""
        with patch("deepagents_code.update_check.time.time", return_value=100.0):
            should_defer_startup_auto_update_for_resume()
        with patch("deepagents_code.update_check.time.time", return_value=200.0):
            assert should_defer_startup_auto_update_for_resume() is True

        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["resume_auto_update_deferred_at"] == pytest.approx(100.0)

    def test_expired_grace_period_runs_update_path(self, state_file) -> None:  # noqa: ARG002
        """Resume-only use stops bypassing startup updates after seven days."""
        from deepagents_code.update_check import RESUME_AUTO_UPDATE_GRACE_PERIOD

        with patch("deepagents_code.update_check.time.time", return_value=100.0):
            should_defer_startup_auto_update_for_resume()
        with patch(
            "deepagents_code.update_check.time.time",
            return_value=100.0 + RESUME_AUTO_UPDATE_GRACE_PERIOD,
        ):
            assert should_defer_startup_auto_update_for_resume() is False

    def test_normal_launch_resets_grace_period(self, state_file) -> None:
        """A normal launch lets a later resume begin a fresh grace period."""
        with patch("deepagents_code.update_check.time.time", return_value=100.0):
            should_defer_startup_auto_update_for_resume()

        clear_resume_auto_update_deferral()

        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert "resume_auto_update_deferred_at" not in data
        with patch("deepagents_code.update_check.time.time", return_value=200.0):
            assert should_defer_startup_auto_update_for_resume() is True

    def test_unwritable_marker_does_not_bypass_update(self, tmp_path) -> None:
        """A persistence failure cannot create an unbounded resume bypass."""
        state_file = tmp_path / "missing" / "update_state.json"
        with (
            patch("deepagents_code.update_check.UPDATE_STATE_FILE", state_file),
            patch("pathlib.Path.mkdir", side_effect=OSError("read-only")),
        ):
            assert should_defer_startup_auto_update_for_resume() is False


class TestShouldNotifyUpdate:
    @pytest.fixture
    def state_file(self, tmp_path):
        """Override UPDATE_STATE_FILE to use a temporary directory."""
        path = tmp_path / "update_state.json"
        with patch("deepagents_code.update_check.UPDATE_STATE_FILE", path):
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
        with patch("deepagents_code.update_check.UPDATE_STATE_FILE", path):
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
            "deepagents_code.update_check.UPDATE_STATE_FILE",
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
        with patch("deepagents_code.update_check.UPDATE_STATE_FILE", path):
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
        with patch("deepagents_code.update_check.UPDATE_STATE_FILE", path):
            yield path

    def test_first_run_returns_false_and_marks(
        self,
        state_file,
        config_path,  # noqa: ARG002
    ) -> None:
        """First run: returns False and writes current version as seen.

        Injects `config_path` so the first-run baseline's explicit-preference
        check reads a clean temp config instead of the developer's real one.
        """
        from deepagents_code.update_check import should_show_whats_new

        with patch("deepagents_code.update_check.__version__", "1.0.0"):
            assert should_show_whats_new() is False
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["seen_version"] == "1.0.0"

    def test_same_version_returns_false(self, state_file) -> None:  # noqa: ARG002
        """Current version == seen version -> False."""
        from deepagents_code.update_check import should_show_whats_new

        mark_version_seen("1.0.0")
        with patch("deepagents_code.update_check.__version__", "1.0.0"):
            assert should_show_whats_new() is False

    def test_newer_version_returns_true(self, state_file) -> None:  # noqa: ARG002
        """Current version > seen version -> True."""
        from deepagents_code.update_check import should_show_whats_new

        mark_version_seen("1.0.0")
        with patch("deepagents_code.update_check.__version__", "2.0.0"):
            assert should_show_whats_new() is True

    def test_coexists_with_notification_state(self, state_file) -> None:  # noqa: ARG002
        """What's-new and notification state don't interfere."""
        from deepagents_code.update_check import should_show_whats_new

        mark_update_notified("2.0.0")
        mark_version_seen("1.0.0")
        with patch("deepagents_code.update_check.__version__", "2.0.0"):
            assert should_show_whats_new() is True
        # Notification throttle still works
        assert should_notify_update("2.0.0") is False
        # Notification throttle still works
        assert should_notify_update("2.0.0") is False

    def test_first_run_suppresses_auto_update_notice(
        self,
        state_file,
        config_path,  # noqa: ARG002
    ) -> None:
        """A fresh install's first run pre-acknowledges the migration notice.

        The auto-update default notice only applies to users who predate the
        opt-out default, so a brand-new install must never see it.
        """
        import os

        from deepagents_code.update_check import should_show_whats_new

        os.environ.pop("DEEPAGENTS_CODE_AUTO_UPDATE", None)
        assert should_announce_auto_update_default() is True
        with patch("deepagents_code.update_check.__version__", "1.0.0"):
            assert should_show_whats_new() is False
        assert should_announce_auto_update_default() is False
        # The baseline write must not clobber the adjacent `seen_version` write;
        # both land in the same state file on first run.
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["seen_version"] == "1.0.0"

    def test_existing_install_still_sees_notice(
        self,
        state_file,  # noqa: ARG002
        config_path,  # noqa: ARG002
    ) -> None:
        """An upgrade for an existing install does not pre-acknowledge.

        When a prior `seen_version` already exists, the first-run baseline does
        not fire, so a returning user still gets the one-time migration notice.
        """
        import os

        from deepagents_code.update_check import should_show_whats_new

        os.environ.pop("DEEPAGENTS_CODE_AUTO_UPDATE", None)
        mark_version_seen("1.0.0")
        with patch("deepagents_code.update_check.__version__", "2.0.0"):
            assert should_show_whats_new() is True
        assert should_announce_auto_update_default() is True

    @pytest.fixture
    def config_path(self, tmp_path):
        """Override DEFAULT_CONFIG_PATH so explicit-preference checks are clean."""
        path = tmp_path / "config.toml"
        with patch("deepagents_code.update_check.DEFAULT_CONFIG_PATH", path):
            yield path
