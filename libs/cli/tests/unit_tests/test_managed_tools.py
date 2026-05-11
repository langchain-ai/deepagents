"""Unit tests for `deepagents_cli.managed_tools`."""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
from pathlib import Path
from unittest import mock

import pytest

from deepagents_cli import managed_tools
from deepagents_cli._env_vars import OFFLINE

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
    managed = tmp_path / "rg"
    managed.write_bytes(b"#!/bin/sh\necho rg\n")
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: managed)
    monkeypatch.setattr(
        managed_tools, "_managed_binary_is_current", lambda _binary: True
    )
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


async def test_ensure_ripgrep_short_circuits_on_android(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: tmp_path / "absent")
    monkeypatch.delenv(OFFLINE, raising=False)
    monkeypatch.setattr(managed_tools.sys, "platform", "android")
    with mock.patch("shutil.which", return_value=None):
        assert await managed_tools.ensure_ripgrep() is None


def _make_fake_tarball(rg_bytes: bytes) -> bytes:
    """Build an in-memory tar.gz containing `ripgrep-x.y.z-triple/rg`."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="ripgrep-14.1.1-test-triple/rg")
        info.size = len(rg_bytes)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(rg_bytes))
    return buf.getvalue()


def test_install_ripgrep_sync_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Download + verify + extract + chmod with monkeypatched urlopen."""
    rg_payload = b"#!/bin/sh\necho fake rg\n"
    tar_bytes = _make_fake_tarball(rg_payload)
    sha = hashlib.sha256(tar_bytes).hexdigest()

    bin_dir = tmp_path / "bin"
    monkeypatch.setattr(managed_tools, "BIN_DIR", bin_dir)
    monkeypatch.setattr(managed_tools, "managed_rg_path", lambda: bin_dir / "rg")
    monkeypatch.setattr(managed_tools.sys, "platform", "linux")

    def _fake_download(url: str, dest: Path) -> None:
        assert "ripgrep" in url
        dest.write_bytes(tar_bytes)

    monkeypatch.setattr(managed_tools, "_download_to", _fake_download)

    installed = managed_tools._install_ripgrep_sync("ripgrep-14.1.1-test.tar.gz", sha)
    assert installed == bin_dir / "rg"
    assert installed.read_bytes() == rg_payload
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
    with pytest.raises(ValueError, match="Checksum mismatch"):
        managed_tools._install_ripgrep_sync("ripgrep-14.1.1-test.tar.gz", "00" * 32)
    # Bin dir may exist but no `rg` should be present.
    assert not (bin_dir / "rg").exists()


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
    assert os.environ["PATH"].split(os.pathsep)[0] == str(bin_dir)


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


def test_managed_binary_is_current_detects_match(tmp_path: Path) -> None:
    binary = tmp_path / "rg"
    binary.write_text("")
    fake = mock.Mock()
    fake.stdout = f"ripgrep {managed_tools.RIPGREP_VERSION} (rev abc)\n"
    with mock.patch("subprocess.run", return_value=fake):
        assert managed_tools._managed_binary_is_current(binary) is True


def test_managed_binary_is_current_detects_stale(tmp_path: Path) -> None:
    binary = tmp_path / "rg"
    binary.write_text("")
    fake = mock.Mock()
    fake.stdout = "ripgrep 13.0.0 (rev abc)\n"
    with mock.patch("subprocess.run", return_value=fake):
        assert managed_tools._managed_binary_is_current(binary) is False
