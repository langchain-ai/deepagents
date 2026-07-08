"""Unit tests for `deepagents_code.managed_tools`."""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tarfile
import warnings
import zipfile
from email.message import Message
from pathlib import Path
from unittest import mock

import pytest

from deepagents_code import managed_tools
from deepagents_code._env_vars import OFFLINE, RIPGREP_INSTALLER
from deepagents_code.managed_tools import (
    ChecksumMismatchError,
    ManagedToolUnavailableError,
)

_EXPECTED_PLATFORM_ARCHS = {
    ("darwin", "arm64"),
    ("darwin", "x86_64"),
    ("linux", "arm64"),
    ("linux", "x86_64"),
    ("win32", "arm64"),
    ("win32", "x86_64"),
}


def test_ripgrep_assets_has_all_expected_keys() -> None:
    assert set(managed_tools.RIPGREP_ASSETS.keys()) == _EXPECTED_PLATFORM_ARCHS


def test_ripgrep_assets_filenames_match_platform_arch() -> None:
    """Each asset filename must encode the platform/arch it serves.

    Stronger than a tautology key-set check: catches mismatches like a
    `darwin x86_64` entry pointing at an `aarch64` asset.
    """
    expected_triples = {
        ("darwin", "arm64"): "aarch64-apple-darwin",
        ("darwin", "x86_64"): "x86_64-apple-darwin",
        ("linux", "arm64"): "aarch64-unknown-linux",
        ("linux", "x86_64"): "x86_64-unknown-linux",
        # Both Windows entries intentionally point at the x86_64 build.
        ("win32", "arm64"): "x86_64-pc-windows",
        ("win32", "x86_64"): "x86_64-pc-windows",
    }
    for key, expected_triple in expected_triples.items():
        asset, _sha = managed_tools.RIPGREP_ASSETS[key]
        assert expected_triple in asset, (key, asset, expected_triple)


def test_ripgrep_assets_values_are_well_formed() -> None:
    for (platform_, arch), entry in managed_tools.RIPGREP_ASSETS.items():
        asset, sha256 = entry
        assert managed_tools.RIPGREP_VERSION in asset, (platform_, arch, asset)
        assert len(sha256) == 64
        int(sha256, 16)


def test_prepend_managed_bin_to_path_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", f"/usr/bin{os.pathsep}/bin")
    managed_tools.prepend_managed_bin_to_path()
    after_first = os.environ["PATH"]
    managed_tools.prepend_managed_bin_to_path()
    assert os.environ["PATH"] == after_first
    assert after_first.startswith(f"{managed_tools.BIN_DIR}{os.pathsep}")


def test_prepend_managed_bin_to_path_dedupes_existing_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed_str = str(managed_tools.BIN_DIR)
    monkeypatch.setenv("PATH", f"/usr/bin{os.pathsep}{managed_str}{os.pathsep}/bin")
    managed_tools.prepend_managed_bin_to_path()
    parts = os.environ["PATH"].split(os.pathsep)
    assert parts[0] == managed_str
    assert parts.count(managed_str) == 1


async def test_ensure_ripgrep_returns_managed_when_current(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A current managed `rg` is returned without re-install.

    Probes the binary's reported version via a fake `subprocess.run`
    rather than stubbing `_managed_binary_is_current` (the branch logic
    under test).
    """
    managed = tmp_path / "rg"
    managed.write_bytes(b"#!/bin/sh\necho rg\n")
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: managed)

    fake = mock.Mock()
    fake.returncode = 0
    fake.stdout = f"ripgrep {managed_tools.RIPGREP_VERSION} (rev abc)\n"
    with mock.patch.object(subprocess, "run", return_value=fake):
        assert await managed_tools.ensure_ripgrep() == managed


async def test_ensure_ripgrep_short_circuits_on_system_rg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: tmp_path / "absent")
    with mock.patch("shutil.which", return_value="/usr/bin/rg"):
        result = await managed_tools.ensure_ripgrep()
    assert result == Path("/usr/bin/rg")


async def test_ensure_ripgrep_short_circuits_when_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(OFFLINE, "1")
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: tmp_path / "absent")
    with mock.patch("shutil.which", return_value=None):
        assert await managed_tools.ensure_ripgrep() is None


def test_ripgrep_installer_defaults_to_managed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RIPGREP_INSTALLER, raising=False)
    assert managed_tools.ripgrep_installer() == managed_tools.INSTALLER_MANAGED
    assert managed_tools.prefers_system_ripgrep() is False


def test_ripgrep_installer_system(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RIPGREP_INSTALLER, "System")
    assert managed_tools.ripgrep_installer() == managed_tools.INSTALLER_SYSTEM
    assert managed_tools.prefers_system_ripgrep() is True


def test_ripgrep_installer_unrecognized_falls_back_to_managed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RIPGREP_INSTALLER, "bogus")
    assert managed_tools.ripgrep_installer() == managed_tools.INSTALLER_MANAGED


async def test_ensure_ripgrep_short_circuits_when_system_installer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`RIPGREP_INSTALLER=system` skips the managed download (no system rg)."""
    monkeypatch.delenv(OFFLINE, raising=False)
    monkeypatch.setenv(RIPGREP_INSTALLER, "system")
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: tmp_path / "absent")

    def _no_download(_url: str, _dest: Path) -> None:
        msg = "_download_to must not be called in system installer mode"
        raise AssertionError(msg)

    monkeypatch.setattr(managed_tools, "_download_to", _no_download)
    with mock.patch("shutil.which", return_value=None):
        assert await managed_tools.ensure_ripgrep() is None


async def test_ensure_ripgrep_system_installer_ignores_current_managed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "managed-bin"
    bin_dir.mkdir()
    managed = bin_dir / "rg"
    managed.write_bytes(b"current-managed")
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: managed)
    monkeypatch.delenv(OFFLINE, raising=False)
    monkeypatch.setenv(RIPGREP_INSTALLER, "system")
    monkeypatch.setenv("PATH", str(bin_dir))

    with (
        mock.patch("shutil.which", return_value=None),
        mock.patch.object(
            subprocess,
            "run",
            side_effect=AssertionError("managed rg must not be version-probed"),
        ),
    ):
        assert await managed_tools.ensure_ripgrep() is None


async def test_ensure_ripgrep_system_installer_uses_non_managed_path_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "managed-bin"
    system_bin = tmp_path / "system-bin"
    bin_dir.mkdir()
    system_bin.mkdir()
    managed = bin_dir / "rg"
    system_rg = system_bin / "rg"
    managed.write_bytes(b"current-managed")
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: managed)
    monkeypatch.delenv(OFFLINE, raising=False)
    monkeypatch.setenv(RIPGREP_INSTALLER, "system")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{system_bin}")

    def _which(cmd: str, path: str | None = None) -> str | None:
        assert cmd == "rg"
        assert path == str(system_bin)
        return str(system_rg)

    with mock.patch("shutil.which", side_effect=_which):
        assert await managed_tools.ensure_ripgrep() == system_rg


def test_path_without_managed_bin_returns_none_when_path_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unset/empty `PATH` yields `None` so `shutil.which` uses its default."""
    monkeypatch.setattr(managed_tools, "BIN_DIR", tmp_path / "managed-bin")
    monkeypatch.delenv("PATH", raising=False)
    assert managed_tools._path_without_managed_bin() is None


def test_path_without_managed_bin_leaves_other_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A `PATH` without the managed dir is returned unchanged."""
    bin_dir = tmp_path / "managed-bin"
    other = tmp_path / "usr-bin"
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setenv("PATH", str(other))
    assert managed_tools._path_without_managed_bin() == str(other)


def test_path_without_managed_bin_removes_managed_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The managed dir is dropped while sibling entries survive."""
    bin_dir = tmp_path / "managed-bin"
    other = tmp_path / "usr-bin"
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{other}")
    assert managed_tools._path_without_managed_bin() == str(other)


def test_path_without_managed_bin_removes_non_canonical_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An entry that differs textually but `resolve()`s equal is still removed.

    Guards the `Path(part).resolve()` comparison against a regression to a
    plain string compare, which would miss `.../managed-bin/.` and friends.
    """
    bin_dir = tmp_path / "managed-bin"
    other = tmp_path / "usr-bin"
    alias = f"{bin_dir}{os.sep}."
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setenv("PATH", f"{alias}{os.pathsep}{other}")
    assert managed_tools._path_without_managed_bin() == str(other)


def test_path_without_managed_bin_preserves_empty_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty `PATH` components are kept verbatim, not resolved to cwd.

    Resolving an empty entry would collapse it to the current directory, which
    could spuriously match `BIN_DIR`; the `not part` short-circuit avoids that.
    """
    bin_dir = tmp_path / "managed-bin"
    other = tmp_path / "usr-bin"
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.pathsep}{other}")
    assert managed_tools._path_without_managed_bin() == f"{os.pathsep}{other}"


async def test_ensure_ripgrep_reports_unsupported_android(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: tmp_path / "absent")
    monkeypatch.delenv(OFFLINE, raising=False)
    monkeypatch.setattr(managed_tools.sys, "platform", "android")
    with (
        mock.patch("shutil.which", return_value=None),
        pytest.raises(ManagedToolUnavailableError) as exc_info,
    ):
        await managed_tools.ensure_ripgrep()
    assert exc_info.value.reason == "unsupported"
    assert "android" in exc_info.value.message


async def test_ensure_ripgrep_reports_unsupported_arch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unsupported arch (e.g. s390x) raises before any download."""
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: tmp_path / "absent")
    monkeypatch.delenv(OFFLINE, raising=False)
    monkeypatch.setattr(managed_tools, "_normalized_arch", lambda: None)

    def _no_download(_url: str, _dest: Path) -> None:
        msg = "_download_to must not be called on unsupported arch"
        raise AssertionError(msg)

    monkeypatch.setattr(managed_tools, "_download_to", _no_download)
    with (
        mock.patch("shutil.which", return_value=None),
        pytest.raises(ManagedToolUnavailableError) as exc_info,
    ):
        await managed_tools.ensure_ripgrep()
    assert exc_info.value.reason == "unsupported"


async def test_ensure_ripgrep_reports_missing_asset_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A supported arch with no pinned asset is a permanent unavailable state."""
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: tmp_path / "absent")
    monkeypatch.delenv(OFFLINE, raising=False)
    monkeypatch.setattr(managed_tools.sys, "platform", "linux")
    monkeypatch.setattr(managed_tools, "_normalized_arch", lambda: "x86_64")
    monkeypatch.delitem(managed_tools.RIPGREP_ASSETS, ("linux", "x86_64"))

    with (
        mock.patch("shutil.which", return_value=None),
        pytest.raises(ManagedToolUnavailableError) as exc_info,
    ):
        await managed_tools.ensure_ripgrep()
    assert exc_info.value.reason == "unsupported"
    assert "linux/x86_64" in exc_info.value.message


async def test_ensure_ripgrep_preserves_stale_when_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An offline user keeps their stale managed binary rather than losing it.

    Regression for ordering: removal must not run before the offline gate.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    managed = bin_dir / "rg"
    managed.write_bytes(b"stale-but-working")
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: managed)
    monkeypatch.setenv(OFFLINE, "1")

    fake = mock.Mock()
    fake.returncode = 0
    fake.stdout = "ripgrep 1.0.0 (rev stale)\n"
    with mock.patch.object(subprocess, "run", return_value=fake):
        result = await managed_tools.ensure_ripgrep()

    assert result is None
    assert managed.exists(), "stale binary should not be removed when offline"


async def test_ensure_ripgrep_preserves_stale_on_unsupported_arch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stale managed binary survives when no asset matches platform/arch."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    managed = bin_dir / "rg"
    managed.write_bytes(b"stale-but-working")
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: managed)
    monkeypatch.delenv(OFFLINE, raising=False)
    monkeypatch.setattr(managed_tools, "_normalized_arch", lambda: None)

    fake = mock.Mock()
    fake.returncode = 0
    fake.stdout = "ripgrep 1.0.0 (rev stale)\n"
    with (
        mock.patch.object(subprocess, "run", return_value=fake),
        pytest.raises(ManagedToolUnavailableError),
    ):
        await managed_tools.ensure_ripgrep()
    assert managed.exists()


def _make_fake_tarball(
    rg_bytes: bytes, *, member_name: str = "ripgrep-14.1.1-test-triple/rg"
) -> bytes:
    """Build an in-memory tar.gz containing `ripgrep-x.y.z-triple/rg`."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(rg_bytes)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(rg_bytes))
    return buf.getvalue()


def _make_fake_zip(
    rg_bytes: bytes, *, member_name: str = "ripgrep-14.1.1/rg.exe"
) -> bytes:
    """Build an in-memory zip containing a single `rg.exe` member."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member_name, rg_bytes)
    return buf.getvalue()


def _patch_legacy_tar_extractall(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate Python 3.11 patch versions without `extractall(filter=...)`."""
    original = tarfile.TarFile.extractall

    def _legacy_extractall(
        self: tarfile.TarFile,
        path: str | os.PathLike[str] = ".",
        members: list[tarfile.TarInfo] | None = None,
        *,
        numeric_owner: bool = False,
        **kwargs: object,
    ) -> None:
        if "filter" in kwargs:
            msg = "TarFile.extractall() got an unexpected keyword argument 'filter'"
            raise TypeError(msg)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Python 3.14 will, by default, filter extracted tar archives",
                category=DeprecationWarning,
            )
            original(self, path, members=members, numeric_owner=numeric_owner)

    monkeypatch.setattr(tarfile.TarFile, "extractall", _legacy_extractall)


def test_extract_rg_supports_legacy_tar_extractall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tar extraction falls back when Python 3.11 lacks `filter=` support."""
    rg_payload = b"#!/bin/sh\necho fake rg\n"
    archive = tmp_path / "ripgrep-test.tar.gz"
    archive.write_bytes(_make_fake_tarball(rg_payload))
    _patch_legacy_tar_extractall(monkeypatch)

    extracted = managed_tools._extract_rg(archive, tmp_path / "unpacked")

    assert extracted.read_bytes() == rg_payload


def test_extract_rg_legacy_fallback_rejects_unsafe_member(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Legacy extraction still rejects path traversal members."""
    archive = tmp_path / "ripgrep-test.tar.gz"
    archive.write_bytes(_make_fake_tarball(b"bad", member_name="../rg"))
    _patch_legacy_tar_extractall(monkeypatch)

    with pytest.raises(tarfile.TarError, match="unsafe tar member"):
        managed_tools._extract_rg(archive, tmp_path / "unpacked")


def test_extract_rg_extracts_zip_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Zip extraction places `rg.exe` at the expected location."""
    monkeypatch.setattr(managed_tools.sys, "platform", "win32")
    payload = b"fake-windows-rg-exe"
    archive = tmp_path / "ripgrep-test.zip"
    archive.write_bytes(_make_fake_zip(payload))

    extracted = managed_tools._extract_rg(archive, tmp_path / "unpacked")

    assert extracted.read_bytes() == payload
    assert extracted.name == "rg.exe"


def test_extract_rg_rejects_zip_slip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A zip member with `..` in its path is refused even before extraction."""
    monkeypatch.setattr(managed_tools.sys, "platform", "win32")
    archive = tmp_path / "ripgrep-evil.zip"
    archive.write_bytes(_make_fake_zip(b"bad", member_name="../rg.exe"))

    with pytest.raises(zipfile.BadZipFile, match="unsafe zip member"):
        managed_tools._extract_rg(archive, tmp_path / "unpacked")


def test_extract_rg_missing_binary_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Archive missing the `rg` member surfaces a clear error."""
    monkeypatch.setattr(managed_tools.sys, "platform", "linux")
    archive = tmp_path / "ripgrep-no-rg.tar.gz"
    archive.write_bytes(
        _make_fake_tarball(b"readme contents", member_name="ripgrep-14.1.1/README")
    )

    with pytest.raises(FileNotFoundError, match="Could not find rg"):
        managed_tools._extract_rg(archive, tmp_path / "unpacked")


@pytest.mark.parametrize("platform_name", ["linux", "darwin", "win32"])
def test_install_ripgrep_sync_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform_name: str
) -> None:
    """Download + verify + extract + install across platform variants."""
    rg_payload = b"#!/bin/sh\necho fake rg\n"
    is_windows = platform_name == "win32"
    archive_bytes = (
        _make_fake_zip(rg_payload, member_name="ripgrep-14.1.1-test/rg.exe")
        if is_windows
        else _make_fake_tarball(rg_payload)
    )
    sha = hashlib.sha256(archive_bytes).hexdigest()

    bin_dir = tmp_path / "bin"
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setattr(
        managed_tools,
        "managed_rg_path",
        lambda: bin_dir / ("rg.exe" if is_windows else "rg"),
    )
    monkeypatch.setattr(managed_tools.sys, "platform", platform_name)

    def _fake_download(url: str, dest: Path) -> None:
        assert "ripgrep" in url
        dest.write_bytes(archive_bytes)

    monkeypatch.setattr(managed_tools, "_download_to", _fake_download)

    asset_name = (
        "ripgrep-14.1.1-test.zip" if is_windows else "ripgrep-14.1.1-test.tar.gz"
    )
    installed = managed_tools._install_ripgrep_sync(asset_name, sha)
    expected = bin_dir / ("rg.exe" if is_windows else "rg")
    assert installed == expected
    assert installed.read_bytes() == rg_payload
    if not is_windows:
        assert installed.stat().st_mode & 0o777 == 0o755


def test_install_ripgrep_sync_rejects_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tar_bytes = _make_fake_tarball(b"hi")
    bin_dir = tmp_path / "bin"
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: bin_dir / "rg")
    monkeypatch.setattr(managed_tools.sys, "platform", "linux")
    monkeypatch.setattr(
        managed_tools,
        "_download_to",
        lambda _url, dest: dest.write_bytes(tar_bytes),
    )
    with pytest.raises(ChecksumMismatchError, match="Checksum mismatch"):
        managed_tools._install_ripgrep_sync("ripgrep-14.1.1-test.tar.gz", "00" * 32)
    assert not (bin_dir / "rg").exists()


async def test_ensure_ripgrep_propagates_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`ensure_ripgrep` must raise on checksum mismatch, not return `None`.

    Callers rely on the distinct exception type to surface a loud,
    user-visible notice — a silent fall-through would mask a
    supply-chain anomaly.
    """
    bin_dir = tmp_path / "bin"
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: bin_dir / "rg")
    monkeypatch.delenv(OFFLINE, raising=False)
    monkeypatch.setattr(managed_tools.sys, "platform", "linux")
    monkeypatch.setattr(managed_tools, "_normalized_arch", lambda: "x86_64")

    tar_bytes = _make_fake_tarball(b"hi")
    monkeypatch.setitem(
        managed_tools.RIPGREP_ASSETS,
        ("linux", "x86_64"),
        ("ripgrep-test.tar.gz", "00" * 32),
    )
    monkeypatch.setattr(
        managed_tools,
        "_download_to",
        lambda _url, dest: dest.write_bytes(tar_bytes),
    )

    with (
        mock.patch("shutil.which", return_value=None),
        pytest.raises(ChecksumMismatchError),
    ):
        await managed_tools.ensure_ripgrep()


async def test_ensure_ripgrep_downloads_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "bin"
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: bin_dir / "rg")
    monkeypatch.delenv(OFFLINE, raising=False)
    monkeypatch.setattr(managed_tools.sys, "platform", "linux")
    monkeypatch.setattr(managed_tools, "_normalized_arch", lambda: "x86_64")

    rg_payload = b"fake-binary"
    tar_bytes = _make_fake_tarball(rg_payload)
    sha = hashlib.sha256(tar_bytes).hexdigest()
    monkeypatch.setitem(
        managed_tools.RIPGREP_ASSETS,
        ("linux", "x86_64"),
        ("ripgrep-test.tar.gz", sha),
    )
    monkeypatch.setattr(
        managed_tools,
        "_download_to",
        lambda _url, dest: dest.write_bytes(tar_bytes),
    )

    with mock.patch("shutil.which", return_value=None):
        result = await managed_tools.ensure_ripgrep()
    assert result is not None
    assert result == bin_dir / "rg"
    assert result.exists()
    assert result.is_symlink()
    assert result.readlink() == Path(f"rg-{managed_tools.RIPGREP_VERSION}")
    versioned = bin_dir / f"rg-{managed_tools.RIPGREP_VERSION}"
    assert versioned.read_bytes() == rg_payload
    assert os.environ["PATH"].split(os.pathsep)[0] == str(bin_dir)


async def test_ensure_ripgrep_repairs_dangling_managed_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    managed = bin_dir / "rg"
    managed.symlink_to("missing-rg")
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: managed)
    monkeypatch.delenv(OFFLINE, raising=False)
    monkeypatch.setattr(managed_tools.sys, "platform", "linux")
    monkeypatch.setattr(managed_tools, "_normalized_arch", lambda: "x86_64")

    rg_payload = b"repaired-binary"
    tar_bytes = _make_fake_tarball(rg_payload)
    sha = hashlib.sha256(tar_bytes).hexdigest()
    monkeypatch.setitem(
        managed_tools.RIPGREP_ASSETS,
        ("linux", "x86_64"),
        ("ripgrep-test.tar.gz", sha),
    )
    monkeypatch.setattr(
        managed_tools,
        "_download_to",
        lambda _url, dest: dest.write_bytes(tar_bytes),
    )

    with mock.patch("shutil.which", return_value="/usr/bin/rg"):
        result = await managed_tools.ensure_ripgrep()

    assert result == managed
    assert managed.read_bytes() == rg_payload
    assert managed.is_symlink()


async def test_ensure_ripgrep_redownloads_stale_managed_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale managed binary is replaced end-to-end, ignoring system `rg`.

    Verifies the resolution-order guarantee: once the user has a managed
    binary, the system `rg` is not silently substituted when the pin
    bumps. The stale bytes are also replaced — a regression letting them
    persist would silently ship outdated functionality.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    managed = bin_dir / "rg"
    managed.write_bytes(b"stale-bytes")
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: managed)
    monkeypatch.delenv(OFFLINE, raising=False)
    monkeypatch.setattr(managed_tools.sys, "platform", "linux")
    monkeypatch.setattr(managed_tools, "_normalized_arch", lambda: "x86_64")

    new_payload = b"new-binary"
    tar_bytes = _make_fake_tarball(new_payload)
    sha = hashlib.sha256(tar_bytes).hexdigest()
    monkeypatch.setitem(
        managed_tools.RIPGREP_ASSETS,
        ("linux", "x86_64"),
        ("ripgrep-test.tar.gz", sha),
    )
    monkeypatch.setattr(
        managed_tools,
        "_download_to",
        lambda _url, dest: dest.write_bytes(tar_bytes),
    )

    fake_probe = mock.Mock()
    fake_probe.returncode = 0
    fake_probe.stdout = "ripgrep 1.0.0 (rev stale)\n"
    with (
        mock.patch.object(subprocess, "run", return_value=fake_probe),
        mock.patch("shutil.which", return_value="/usr/bin/rg"),
    ):
        result = await managed_tools.ensure_ripgrep()

    assert result == managed
    assert managed.read_bytes() == new_payload
    assert managed.is_symlink()


async def test_ensure_ripgrep_reports_missing_artifact_on_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "bin"
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: bin_dir / "rg")
    monkeypatch.delenv(OFFLINE, raising=False)
    monkeypatch.setattr(managed_tools.sys, "platform", "linux")
    monkeypatch.setattr(managed_tools, "_normalized_arch", lambda: "x86_64")

    import urllib.error

    err = urllib.error.HTTPError(
        "https://example.test/rg.tar.gz", 404, "Not Found", hdrs=Message(), fp=None
    )

    def _boom(_url: str, _dest: Path) -> None:
        raise err

    monkeypatch.setattr(managed_tools, "_download_to", _boom)
    with (
        mock.patch("shutil.which", return_value=None),
        pytest.raises(ManagedToolUnavailableError) as exc_info,
    ):
        await managed_tools.ensure_ripgrep()
    assert exc_info.value.reason == "artifact_not_found"
    assert "linux/x86_64" in exc_info.value.message


async def test_ensure_ripgrep_returns_none_on_download_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "bin"
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: bin_dir / "rg")
    monkeypatch.delenv(OFFLINE, raising=False)
    monkeypatch.setattr(managed_tools.sys, "platform", "linux")
    monkeypatch.setattr(managed_tools, "_normalized_arch", lambda: "x86_64")

    import urllib.error

    err = urllib.error.URLError("connection refused")

    def _boom(_url: str, _dest: Path) -> None:
        raise err

    monkeypatch.setattr(managed_tools, "_download_to", _boom)
    with mock.patch("shutil.which", return_value=None):
        assert await managed_tools.ensure_ripgrep() is None


async def test_ensure_ripgrep_returns_none_on_http_download_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Non-404 HTTP failures are transient download failures, not unavailability."""
    bin_dir = tmp_path / "bin"
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: bin_dir / "rg")
    monkeypatch.delenv(OFFLINE, raising=False)
    monkeypatch.setattr(managed_tools.sys, "platform", "linux")
    monkeypatch.setattr(managed_tools, "_normalized_arch", lambda: "x86_64")

    import urllib.error

    err = urllib.error.HTTPError(
        "https://example.test/rg.tar.gz",
        503,
        "Service Unavailable",
        hdrs=Message(),
        fp=None,
    )

    def _boom(_url: str, _dest: Path) -> None:
        raise err

    monkeypatch.setattr(managed_tools, "_download_to", _boom)
    with mock.patch("shutil.which", return_value=None):
        assert await managed_tools.ensure_ripgrep() is None


async def test_ensure_ripgrep_preserves_stale_on_download_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed replacement install must not delete the existing stale binary.

    Regression: a transient network failure during a pin bump would
    otherwise strand the user with no `rg`. Atomic replace means the
    stale copy stays in place until a verified replacement is ready.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    managed = bin_dir / "rg"
    stale_bytes = b"stale-but-usable"
    managed.write_bytes(stale_bytes)
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: managed)
    monkeypatch.delenv(OFFLINE, raising=False)
    monkeypatch.setattr(managed_tools.sys, "platform", "linux")
    monkeypatch.setattr(managed_tools, "_normalized_arch", lambda: "x86_64")
    monkeypatch.setitem(
        managed_tools.RIPGREP_ASSETS,
        ("linux", "x86_64"),
        ("ripgrep-test.tar.gz", "00" * 32),
    )

    import urllib.error

    err = urllib.error.URLError("connection refused")

    def _boom(_url: str, _dest: Path) -> None:
        raise err

    monkeypatch.setattr(managed_tools, "_download_to", _boom)

    fake_probe = mock.Mock()
    fake_probe.returncode = 0
    fake_probe.stdout = "ripgrep 1.0.0 (rev stale)\n"
    with (
        mock.patch.object(subprocess, "run", return_value=fake_probe),
        mock.patch("shutil.which", return_value="/usr/bin/rg"),
    ):
        result = await managed_tools.ensure_ripgrep()

    assert result is None
    assert managed.exists()
    assert managed.read_bytes() == stale_bytes


def test_managed_binary_is_current_detects_match(tmp_path: Path) -> None:
    binary = tmp_path / "rg"
    binary.write_text("")
    fake = mock.Mock()
    fake.returncode = 0
    fake.stdout = f"ripgrep {managed_tools.RIPGREP_VERSION} (rev abc)\n"
    with mock.patch.object(subprocess, "run", return_value=fake):
        assert managed_tools._managed_binary_is_current(binary) is True


def test_managed_binary_is_current_detects_stale(tmp_path: Path) -> None:
    binary = tmp_path / "rg"
    binary.write_text("")
    fake = mock.Mock()
    fake.returncode = 0
    fake.stdout = "ripgrep 13.0.0 (rev abc)\n"
    with mock.patch.object(subprocess, "run", return_value=fake):
        assert managed_tools._managed_binary_is_current(binary) is False


def test_managed_binary_is_current_treats_oserror_as_stale(tmp_path: Path) -> None:
    """A binary that won't even exec (corrupt, wrong-arch) is not trusted."""
    binary = tmp_path / "rg"
    binary.write_bytes(b"not-a-real-binary")
    with mock.patch.object(subprocess, "run", side_effect=OSError("ENOEXEC")):
        assert managed_tools._managed_binary_is_current(binary) is False


def test_managed_binary_is_current_treats_nonzero_exit_as_stale(
    tmp_path: Path,
) -> None:
    """A binary that prints the right version but exits non-zero is not trusted."""
    binary = tmp_path / "rg"
    binary.write_text("")
    fake = mock.Mock()
    fake.returncode = 1
    fake.stdout = f"ripgrep {managed_tools.RIPGREP_VERSION} (rev abc)\n"
    with mock.patch.object(subprocess, "run", return_value=fake):
        assert managed_tools._managed_binary_is_current(binary) is False


def test_managed_binary_is_current_treats_empty_stdout_as_stale(
    tmp_path: Path,
) -> None:
    """A binary that exits 0 with no output is not trusted."""
    binary = tmp_path / "rg"
    binary.write_text("")
    fake = mock.Mock()
    fake.returncode = 0
    fake.stdout = ""
    with mock.patch.object(subprocess, "run", return_value=fake):
        assert managed_tools._managed_binary_is_current(binary) is False


def test_managed_binary_is_current_falls_open_on_timeout(tmp_path: Path) -> None:
    """A timed-out probe (sandboxed subprocess) does not force a redownload."""
    binary = tmp_path / "rg"
    binary.write_text("")
    timeout = subprocess.TimeoutExpired(cmd=[str(binary), "--version"], timeout=5)
    with mock.patch.object(subprocess, "run", side_effect=timeout):
        assert managed_tools._managed_binary_is_current(binary) is True


def test_download_to_enforces_total_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A slow trickle exceeding the deadline raises `TimeoutError`."""
    from typing import Self

    class _SlowResponse:
        def __init__(self) -> None:
            self._calls = 0

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            self._calls += 1
            return b"x" * 4

    monkeypatch.setattr(managed_tools, "_DOWNLOAD_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: _SlowResponse()
    )

    dest = tmp_path / "archive"
    with pytest.raises(TimeoutError, match="deadline"):
        managed_tools._download_to("https://example.invalid/x", dest)


def test_download_to_rejects_non_200_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-200 response is rejected before any bytes are written.

    Guards against a proxy interstitial or unfollowed redirect being
    streamed to disk and only caught later as a misleading checksum failure.
    """
    import urllib.error
    from typing import Self

    class _Non200Response:
        status = 503

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            msg = "read must not be called on a non-200 response"
            raise AssertionError(msg)

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: _Non200Response()
    )

    dest = tmp_path / "archive"
    with pytest.raises(urllib.error.URLError, match="HTTP 503"):
        managed_tools._download_to("https://example.invalid/x", dest)
    assert dest.read_bytes() == b""


def test_extract_tar_data_reraises_unrelated_typeerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unrelated `TypeError` propagates instead of falling to legacy tar."""
    archive = tmp_path / "ripgrep-test.tar.gz"
    archive.write_bytes(_make_fake_tarball(b"payload"))

    def _boom_extractall(
        _self: tarfile.TarFile, *_args: object, **_kwargs: object
    ) -> None:
        msg = "something else entirely"
        raise TypeError(msg)

    monkeypatch.setattr(tarfile.TarFile, "extractall", _boom_extractall)

    with pytest.raises(TypeError, match="something else entirely"):
        managed_tools._extract_rg(archive, tmp_path / "unpacked")
