"""Python compatibility checks and receipt-preserving uv tool migrations."""

from __future__ import annotations

import os
import shlex
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from packaging.specifiers import SpecifierSet

from deepagents_code import update_check
from deepagents_code.extras_info import ExtrasIntrospectionError


class VersionInfo(NamedTuple):
    major: int
    minor: int
    micro: int


def _response(info: object) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"info": info}
    return response


@pytest.mark.parametrize("version", ["2.0.0", "2.0.0rc1"])
def test_release_requirement_reads_exact_target(version: str) -> None:
    with patch(
        "requests.get",
        return_value=_response({"version": version, "requires_python": ">=3.14,<3.15"}),
    ) as get:
        requirement = update_check._release_python_requirement(version)
    assert requirement == SpecifierSet(">=3.14,<3.15")
    assert get.call_args.args[0].endswith(f"/{version}/json")
    assert get.call_args.kwargs["timeout"] > 0


@pytest.mark.parametrize(
    "info",
    [
        None,
        {},
        {"version": "2.0.1", "requires_python": ">=3.14"},
        {"version": "invalid", "requires_python": ">=3.14"},
        {"version": "2.0.0"},
        {"version": "2.0.0", "requires_python": None},
        {"version": "2.0.0", "requires_python": " "},
        {"version": "2.0.0", "requires_python": "not a specifier"},
    ],
)
def test_release_requirement_refuses_untrusted_metadata(info: object) -> None:
    with (
        patch("requests.get", return_value=_response(info)),
        pytest.raises(ValueError, match=r"metadata|specifier"),
    ):
        update_check._release_python_requirement("2.0.0")


@pytest.mark.parametrize(
    "error", [requests.ConnectionError("offline"), ValueError("bad JSON")]
)
def test_release_requirement_refuses_network_or_json_failure(error: Exception) -> None:
    with (
        patch("requests.get", side_effect=error),
        pytest.raises(ValueError, match="metadata"),
    ):
        update_check._release_python_requirement("2.0.0")


@pytest.mark.parametrize("payload", [None, [], "invalid"])
def test_release_requirement_refuses_non_object_response(payload: object) -> None:
    response = MagicMock()
    response.json.return_value = payload
    with (
        patch("requests.get", return_value=response),
        pytest.raises(ValueError, match="metadata"),
    ):
        update_check._release_python_requirement("2.0.0")


def test_release_requirement_refuses_http_error() -> None:
    response = _response({"version": "2.0.0", "requires_python": ">=3.14,<3.15"})
    response.raise_for_status.side_effect = requests.HTTPError("503 unavailable")
    with (
        patch("requests.get", return_value=response),
        pytest.raises(ValueError, match="503 unavailable"),
    ):
        update_check._release_python_requirement("2.0.0")


@pytest.mark.parametrize(
    ("running", "requirement", "expected"),
    [
        ((3, 12, 9), ">=3.14,<3.15", "3.14"),
        ((3, 13, 4), ">=3.14,<3.15", "3.14"),
        ((3, 14, 0), ">=3.14,<3.15", None),
        ((3, 13, 4), ">=3.12", None),
    ],
)
def test_upgrade_python_only_migrates_incompatible_interpreters(
    running: tuple[int, int, int], requirement: str, expected: str | None
) -> None:
    with (
        patch.object(update_check.sys, "version_info", running),
        patch.object(
            update_check,
            "_release_python_requirement",
            return_value=SpecifierSet(requirement),
        ),
    ):
        assert update_check._upgrade_python("2.0.0") == expected


def test_upgrade_python_refuses_unsupported_python_floor() -> None:
    with (
        patch.object(update_check.sys, "version_info", VersionInfo(3, 13, 4)),
        patch.object(
            update_check,
            "_release_python_requirement",
            return_value=SpecifierSet(">=3.15"),
        ),
        pytest.raises(ValueError, match="compatible interpreter"),
    ):
        update_check._upgrade_python("2.0.0")


@pytest.fixture
def uv_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    tmp_path.joinpath("uv-receipt.toml").write_text(
        '[tool]\npython = "3.13"\nrequirements = ['
        '{ name = "deepagents-code", extras = ["openai", "quickjs"] }, '
        '{ name = "langchain-custom" }]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(update_check.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(update_check, "detect_install_method", lambda: "uv")
    monkeypatch.setattr(update_check, "update_install_lock", lambda: nullcontext(True))
    monkeypatch.setattr(update_check, "release_prerelease_pins", lambda _version: [])
    monkeypatch.setattr(
        update_check,
        "_release_python_requirement",
        lambda _version: SpecifierSet(">=3.14,<3.15"),
    )
    monkeypatch.setattr(
        update_check, "read_installed_distribution_version", lambda: "2.0.0"
    )
    monkeypatch.setattr(
        update_check, "_read_migrated_distribution_version", lambda: "2.0.0"
    )
    run = AsyncMock(return_value=(True, "installed"))
    monkeypatch.setattr(update_check, "_run_install_subprocess", run)
    return run


@pytest.mark.parametrize("minor", [12, 13])
async def test_migration_preserves_install_context_and_announces_before_install(
    uv_install: AsyncMock, minor: int
) -> None:
    notices: list[str] = []

    def install(command: str, **_kwargs: object) -> tuple[bool, str]:
        assert notices
        assert "3.14" in notices[0]
        args = shlex.split(command)
        assert args[:4] == ["uv", "tool", "install", "-U"]
        assert args[args.index("--python") + 1] == "3.14"
        assert "deepagents-code[openai,quickjs]==2.0.0" in args
        assert args[args.index("--with") + 1] == "langchain-custom"
        assert "uninstall" not in args
        return True, "installed"

    uv_install.side_effect = install
    with patch.object(update_check.sys, "version_info", VersionInfo(3, minor, 4)):
        result = await update_check.perform_upgrade(
            target_version="2.0.0", include_prereleases=False, on_notice=notices.append
        )
    assert result == (True, "installed", "2.0.0")
    uv_install.assert_awaited_once()


async def test_compatible_upgrade_keeps_receipt_interpreter(
    uv_install: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        update_check,
        "_release_python_requirement",
        lambda _version: SpecifierSet(">=3.12"),
    )
    notice = MagicMock()
    with patch.object(update_check.sys, "version_info", VersionInfo(3, 13, 4)):
        success, _, version = await update_check.perform_upgrade(
            target_version="2.0.0", include_prereleases=False, on_notice=notice
        )
    assert success
    assert version == "2.0.0"
    args = shlex.split(uv_install.call_args.args[0])
    assert args[args.index("--python") + 1] == "3.13"
    notice.assert_not_called()


async def test_missing_target_is_resolved_then_pinned(
    uv_install: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    latest = MagicMock(return_value="2.0.0")
    monkeypatch.setattr(update_check, "get_latest_version", latest)
    with patch.object(update_check.sys, "version_info", VersionInfo(3, 13, 4)):
        success, _, version = await update_check.perform_upgrade(
            include_prereleases=False
        )
    assert success
    assert version == "2.0.0"
    assert "deepagents-code[openai,quickjs]==2.0.0" in shlex.split(
        uv_install.call_args.args[0]
    )
    latest.assert_called_once()


async def test_prerelease_migration_remains_exactly_pinned(
    uv_install: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        update_check, "_read_migrated_distribution_version", lambda: "2.0.0rc1"
    )
    with patch.object(update_check.sys, "version_info", VersionInfo(3, 13, 4)):
        result = await update_check.perform_upgrade(
            target_version="2.0.0rc1", include_prereleases=True
        )
    assert result[0]
    assert result[2] == "2.0.0rc1"
    args = shlex.split(uv_install.call_args.args[0])
    assert "deepagents-code[openai,quickjs]==2.0.0rc1" in args
    assert args[args.index("--prerelease") + 1] == "allow"


async def test_migration_preserves_targeted_prerelease_constraints(
    uv_install: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        update_check,
        "release_prerelease_pins",
        lambda _version: ["deepagents==0.7.0a7"],
    )

    def install(command: str, **_kwargs: object) -> tuple[bool, str]:
        args = shlex.split(command)
        assert args[args.index("--python") + 1] == "3.14"
        assert "deepagents-code[openai,quickjs]==2.0.0" in args
        assert args[args.index("--prerelease") + 1] == "if-necessary-or-explicit"
        constraints = Path(args[args.index("--constraints") + 1])
        assert "deepagents==0.7.0a7" in constraints.read_text(encoding="utf-8")
        return True, "installed"

    uv_install.side_effect = install
    with patch.object(update_check.sys, "version_info", VersionInfo(3, 13, 4)):
        result = await update_check.perform_upgrade(target_version="2.0.0")
    assert result == (True, "installed", "2.0.0")


@pytest.mark.parametrize("migrating", [False, True])
@pytest.mark.parametrize("installed", [None, "invalid", "1.9.9", "2.0.1"])
async def test_installer_success_requires_exact_verified_version(
    uv_install: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
    migrating: bool,
    installed: str | None,
) -> None:
    monkeypatch.setattr(
        update_check, "_upgrade_python", lambda _version: "3.14" if migrating else None
    )
    monkeypatch.setattr(
        update_check, "read_installed_distribution_version", lambda: installed
    )
    monkeypatch.setattr(
        update_check, "_read_migrated_distribution_version", lambda: installed
    )
    success, output, version = await update_check.perform_upgrade(
        target_version="2.0.0", include_prereleases=False
    )
    assert not success
    assert version is None
    assert output
    uv_install.assert_awaited_once()


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("metadata unavailable"),
        ExtrasIntrospectionError("missing extras"),
        update_check.ToolRequirementIntrospectionError("missing receipt"),
    ],
)
async def test_preflight_failure_never_runs_bare_installer(
    uv_install: AsyncMock, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    helper = (
        "_upgrade_python"
        if isinstance(failure, ValueError)
        else "upgrade_install_command"
    )
    monkeypatch.setattr(update_check, helper, MagicMock(side_effect=failure))
    success, output, version = await update_check.perform_upgrade(
        target_version="2.0.0", include_prereleases=False
    )
    assert not success
    assert version is None
    assert str(failure) in output
    uv_install.assert_not_awaited()


async def test_unresolved_target_refuses_install(
    uv_install: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(update_check, "get_latest_version", lambda **_kwargs: None)
    success, output, version = await update_check.perform_upgrade(
        include_prereleases=False
    )
    assert not success
    assert version is None
    assert output
    uv_install.assert_not_awaited()


@pytest.mark.parametrize("receipt", [None, '[tool]\nrequirements = ["broken"]\n'])
async def test_missing_or_malformed_receipt_refuses_migration(
    uv_install: AsyncMock, tmp_path: Path, receipt: str | None
) -> None:
    path = tmp_path / "uv-receipt.toml"
    if receipt is None:
        path.unlink()
    else:
        path.write_text(receipt, encoding="utf-8")
    with patch.object(update_check.sys, "version_info", VersionInfo(3, 13, 4)):
        success, output, version = await update_check.perform_upgrade(
            target_version="2.0.0", include_prereleases=False
        )
    assert not success
    assert version is None
    assert "receipt" in output.lower()
    uv_install.assert_not_awaited()


@pytest.mark.parametrize(
    "failure",
    [
        "dependency conflict",
        "Python downloads disabled",
        "interpreter unavailable",
        "network connection failed",
    ],
)
async def test_migration_install_failure_has_actionable_recovery(
    uv_install: AsyncMock,
    failure: str,
) -> None:
    notices: list[str] = []
    uv_install.return_value = False, failure
    with patch.object(update_check.sys, "version_info", VersionInfo(3, 13, 4)):
        success, output, version = await update_check.perform_upgrade(
            target_version="2.0.0", include_prereleases=False, on_notice=notices.append
        )
    assert not success
    assert version is None
    assert failure in output
    assert notices
    assert "network" in notices[-1].lower()
    assert "3.14" in output
    assert "network" in output.lower() or "download" in output.lower()


def test_migrated_readback_uses_rebuilt_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(update_check.sys, "prefix", str(tmp_path))
    with patch(
        "subprocess.run", return_value=subprocess.CompletedProcess([], 0, "2.0.0\n")
    ) as run:
        assert update_check._read_migrated_distribution_version() == "2.0.0"
    args = run.call_args.args[0]
    assert args[0] == str(
        tmp_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    assert "-I" in args
    assert "(3, 14)" in args[-1]
    assert run.call_args.kwargs["check"] is True
    assert run.call_args.kwargs["timeout"] > 0


@pytest.mark.parametrize(
    "error",
    [
        OSError("missing interpreter"),
        subprocess.CalledProcessError(1, "python"),
        subprocess.TimeoutExpired("python", 10),
    ],
)
def test_migrated_readback_refuses_unverified_interpreter(error: Exception) -> None:
    with patch("subprocess.run", side_effect=error):
        assert update_check._read_migrated_distribution_version() is None


def test_migrated_readback_refuses_invalid_version() -> None:
    with patch(
        "subprocess.run", return_value=subprocess.CompletedProcess([], 0, "invalid\n")
    ):
        assert update_check._read_migrated_distribution_version() is None
