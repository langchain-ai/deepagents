"""Auto-install pinned upstream binaries for optional tools.

Today this only manages `ripgrep`. The SDK's `subprocess.run(["rg", ...])`
in `deepagents.backends.filesystem._ripgrep_search` resolves `rg` through
`PATH`, so installing into `~/.deepagents/bin/` and prepending it to
`os.environ["PATH"]` early enough is sufficient — no SDK change required.

Modeled on pi-mono's `tools-manager.ts`. The pinned `RIPGREP_VERSION` and
`RIPGREP_ASSETS` table is the single source of truth for what gets
downloaded and verified. Bumping the version is a quarterly chore: refresh
both the version and the six SHA-256 entries together.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from deepagents_cli._env_vars import OFFLINE, is_env_truthy

logger = logging.getLogger(__name__)

RIPGREP_VERSION = "14.1.1"
"""Pinned upstream ripgrep release. Bump alongside `RIPGREP_ASSETS`."""

_RELEASE_URL_PREFIX = (
    "https://github.com/BurntSushi/ripgrep/releases/download/" + RIPGREP_VERSION
)

RIPGREP_ASSETS: dict[tuple[str, str], tuple[str, str]] = {
    ("darwin", "arm64"): (
        f"ripgrep-{RIPGREP_VERSION}-aarch64-apple-darwin.tar.gz",
        "24ad76777745fbff131c8fbc466742b011f925bfa4fffa2ded6def23b5b937be",
    ),
    ("darwin", "x86_64"): (
        f"ripgrep-{RIPGREP_VERSION}-x86_64-apple-darwin.tar.gz",
        "fc87e78f7cb3fea12d69072e7ef3b21509754717b746368fd40d88963630e2b3",
    ),
    ("linux", "arm64"): (
        f"ripgrep-{RIPGREP_VERSION}-aarch64-unknown-linux-gnu.tar.gz",
        "c827481c4ff4ea10c9dc7a4022c8de5db34a5737cb74484d62eb94a95841ab2f",
    ),
    ("linux", "x86_64"): (
        f"ripgrep-{RIPGREP_VERSION}-x86_64-unknown-linux-musl.tar.gz",
        "4cf9f2741e6c465ffdb7c26f38056a59e2a2544b51f7cc128ef28337eeae4d8e",
    ),
    # Windows on ARM runs x64 binaries via emulation; upstream does not
    # ship an arm64-windows build for ripgrep, so both Windows entries
    # point at the same x86_64 MSVC asset.
    ("win32", "arm64"): (
        f"ripgrep-{RIPGREP_VERSION}-x86_64-pc-windows-msvc.zip",
        "d0f534024c42afd6cb4d38907c25cd2b249b79bbe6cc1dbee8e3e37c2b6e25a1",
    ),
    ("win32", "x86_64"): (
        f"ripgrep-{RIPGREP_VERSION}-x86_64-pc-windows-msvc.zip",
        "d0f534024c42afd6cb4d38907c25cd2b249b79bbe6cc1dbee8e3e37c2b6e25a1",
    ),
}
"""`(sys.platform, normalized arch) -> (asset filename, sha256 hex)`."""

BIN_DIR: Path = Path.home() / ".deepagents" / "bin"
"""Directory holding managed binaries. Prepended to `PATH` on startup."""

_DOWNLOAD_TIMEOUT_SECONDS = 120
_VERSION_CHECK_TIMEOUT_SECONDS = 5
_ARCH_ALIASES = {
    "aarch64": "arm64",
    "arm64": "arm64",
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "x64": "x86_64",
}


def _normalized_arch() -> str | None:
    """Return a normalized arch key matching `RIPGREP_ASSETS`.

    Returns `None` for unsupported architectures (e.g. 32-bit, ppc, s390x).
    """
    import platform

    raw = platform.machine().lower()
    return _ARCH_ALIASES.get(raw)


def managed_rg_path() -> Path:
    """Return the managed ripgrep binary path (`.exe` on Windows)."""
    name = "rg.exe" if sys.platform == "win32" else "rg"
    return BIN_DIR / name


def is_offline() -> bool:
    """Return whether managed-tool downloads are disabled via env var."""
    return is_env_truthy(OFFLINE)


def prepend_managed_bin_to_path() -> None:
    """Idempotently prepend `BIN_DIR` to `os.environ["PATH"]`.

    Safe to call on every startup. Callers do not need to check whether
    the directory exists — adding a non-existent directory to `PATH` is
    harmless and matches behavior of common version managers.
    """
    bin_str = str(BIN_DIR)
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if parts and parts[0] == bin_str:
        return
    parts = [bin_str, *(p for p in parts if p != bin_str)]
    os.environ["PATH"] = os.pathsep.join(parts)


def _managed_binary_is_current(binary: Path) -> bool:
    """Return whether the on-disk managed `rg` matches `RIPGREP_VERSION`.

    Spawning `rg --version` adds ~10 ms but lets users pick up a bumped
    `RIPGREP_VERSION` without manually deleting `~/.deepagents/bin/rg`.
    Falls open (treats the binary as current) on any failure so a flaky
    fork or a sandboxed `subprocess` does not trigger an unwanted
    re-download.
    """
    import subprocess  # noqa: S404  # fixed-argv probe of a managed binary

    try:
        result = subprocess.run(  # noqa: S603  # fixed argv, managed path
            [str(binary), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=_VERSION_CHECK_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("rg --version probe failed for %s", binary, exc_info=True)
        return True
    first_line = (result.stdout or "").splitlines()[:1]
    if not first_line:
        return True
    return RIPGREP_VERSION in first_line[0]


def _download_to(url: str, dest: Path) -> None:
    """Stream `url` to `dest` with a bounded timeout."""
    import shutil
    import urllib.request

    with (
        urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as resp,  # noqa: S310  # fixed https GitHub release URL
        dest.open("wb") as fh,
    ):
        shutil.copyfileobj(resp, fh)


def _verify_sha256(path: Path, expected_hex: str) -> None:
    """Verify `path` matches `expected_hex`.

    Raises:
        ValueError: When the SHA-256 of `path` differs from `expected_hex`.
    """
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_hex:
        msg = (
            f"Checksum mismatch for {path.name}: expected {expected_hex}, got {actual}"
        )
        raise ValueError(msg)


def _extract_rg(archive: Path, extract_root: Path) -> Path:
    """Extract `archive` and locate the `rg` binary inside.

    Handles both `.tar.gz` and `.zip` archives. Release archives nest the
    binary under `ripgrep-<ver>-<triple>/`, so we walk the tree to find it
    rather than hard-coding the prefix.

    Returns:
        Absolute path to the extracted `rg` (or `rg.exe`) binary.

    Raises:
        FileNotFoundError: When the archive does not contain an `rg` binary.
    """
    import tarfile
    import zipfile

    if archive.suffix == ".zip":
        # Archive SHA-256 is verified before extraction.
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_root)  # noqa: S202  # verified above
    else:
        with tarfile.open(archive, mode="r:*") as tf:
            tf.extractall(extract_root, filter="data")

    target_name = "rg.exe" if sys.platform == "win32" else "rg"
    for path in extract_root.rglob(target_name):
        if path.is_file():
            return path
    msg = f"Could not find {target_name} inside {archive.name}"
    raise FileNotFoundError(msg)


def _install_ripgrep_sync(asset: str, sha256: str) -> Path:
    """Download, verify, extract, and install ripgrep.

    Atomic-ish: extracts under a `TemporaryDirectory` and only moves the
    binary into `BIN_DIR` on success. Concurrent CLI invocations are safe
    — the loser overwrites identical bytes via `shutil.move`.

    Returns:
        Absolute path to the installed `rg` binary.
    """
    import shutil
    import tempfile

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{_RELEASE_URL_PREFIX}/{asset}"
    with tempfile.TemporaryDirectory(prefix="deepagents-rg-") as tmp_str:
        tmp = Path(tmp_str)
        archive = tmp / asset
        _download_to(url, archive)
        _verify_sha256(archive, sha256)
        extracted = _extract_rg(archive, tmp / "unpacked")
        dest = managed_rg_path()
        # `shutil.move` replaces an existing file on POSIX and Windows.
        shutil.move(str(extracted), str(dest))
        if sys.platform != "win32":
            dest.chmod(0o755)
        return dest


def _remove_stale_binary(binary: Path) -> None:
    """Best-effort removal of a stale managed binary before re-install."""
    import contextlib

    with contextlib.suppress(OSError):
        binary.unlink()


async def ensure_ripgrep() -> Path | None:
    """Ensure a usable `rg` binary is available, installing if necessary.

    Resolution order:

    1. If a managed `rg` exists *and* matches `RIPGREP_VERSION`, return it.
    2. If a managed `rg` exists but is stale, remove it and fall through
       to step 4 (the system `rg` is not re-checked because the user
       previously opted into the managed binary).
    3. If a system `rg` is on `PATH`, return its resolved path.
    4. If offline, on Android, or no asset matches the platform/arch,
       return `None` so callers fall back to the existing notification +
       slow path.
    5. Otherwise download → SHA-256 verify → extract → install → return
       the installed path. On any failure, log and return `None`.

    Returns:
        Path to a usable `rg` binary, or `None` when one could not be
        located or installed.
    """
    import asyncio
    import platform
    import shutil
    import tarfile
    import urllib.error
    import zipfile

    managed = managed_rg_path()
    if managed.exists():
        if _managed_binary_is_current(managed):
            return managed
        logger.info(
            "Managed ripgrep at %s is stale; replacing with %s",
            managed,
            RIPGREP_VERSION,
        )
        _remove_stale_binary(managed)
    else:
        system_rg = shutil.which("rg")
        if system_rg is not None:
            return Path(system_rg)

    if is_offline():
        logger.debug("Skipping ripgrep install: %s is set", OFFLINE)
        return None
    if sys.platform == "android":
        logger.debug("Skipping ripgrep install: unsupported platform 'android'")
        return None

    arch = _normalized_arch()
    if arch is None:
        logger.debug(
            "Skipping ripgrep install: unsupported arch %r", platform.machine()
        )
        return None

    asset_entry = RIPGREP_ASSETS.get((sys.platform, arch))
    if asset_entry is None:
        logger.debug(
            "Skipping ripgrep install: no asset for (%s, %s)", sys.platform, arch
        )
        return None
    asset, sha256 = asset_entry

    try:
        installed = await asyncio.to_thread(_install_ripgrep_sync, asset, sha256)
    except (urllib.error.URLError, TimeoutError):
        logger.warning(
            "Could not download ripgrep from %s", _RELEASE_URL_PREFIX, exc_info=True
        )
        return None
    except ValueError:
        # Checksum mismatch — already detailed in the exception message.
        logger.exception("ripgrep install aborted: checksum mismatch")
        return None
    except (OSError, tarfile.TarError, zipfile.BadZipFile, FileNotFoundError):
        logger.exception("ripgrep install failed")
        return None
    else:
        prepend_managed_bin_to_path()
        return installed
