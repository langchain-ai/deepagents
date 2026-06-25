"""Auto-install pinned upstream binaries for optional tools.

Today this only manages `ripgrep`. The SDK shells out to `rg` via `PATH`,
so installing into `~/.deepagents/bin/` and prepending that directory to
`os.environ["PATH"]` is sufficient — no SDK change required.

The pinned `RIPGREP_VERSION` and `RIPGREP_ASSETS` table is the single
source of truth for what gets downloaded and verified. When bumping the
version, refresh both the version and the SHA-256 entries together.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code._env_vars import OFFLINE, is_env_truthy

if TYPE_CHECKING:
    import tarfile
    import zipfile

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
_DOWNLOAD_CHUNK_BYTES = 1 << 16
_ARCH_ALIASES = {
    "aarch64": "arm64",
    "arm64": "arm64",
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "x64": "x86_64",
}


class ChecksumMismatchError(Exception):
    """Raised when a downloaded archive fails SHA-256 verification.

    Distinct from generic install failure so callers can surface a loud,
    user-visible notice — a checksum mismatch is a supply-chain anomaly
    (CDN poisoning, MITM, tampered mirror) and must not be silently
    treated like "you're offline".
    """


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

    Returns `False` on any concrete failure (`OSError`, non-zero exit,
    empty stdout, version mismatch) so a corrupted or wrong-arch
    binary written by a previously crashed install gets re-fetched. Only
    `TimeoutExpired` "falls open" — that case suggests a sandboxed
    subprocess rather than a broken binary.
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
    except subprocess.TimeoutExpired:
        logger.debug("rg --version probe timed out for %s; assuming current", binary)
        return True
    except OSError:
        logger.debug(
            "rg --version probe failed for %s; treating as stale",
            binary,
            exc_info=True,
        )
        return False
    if result.returncode != 0:
        logger.debug(
            "rg --version exited %d for %s; treating as stale",
            result.returncode,
            binary,
        )
        return False
    first_line = (result.stdout or "").splitlines()[:1]
    if not first_line:
        return False
    return RIPGREP_VERSION in first_line[0]


def _download_to(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, bounded by a wall-clock deadline.

    `urlopen(timeout=...)` only bounds per-operation socket waits, so a
    slow trickle of bytes from a flaky peer could otherwise stretch the
    transfer well beyond the configured timeout. The chunked read here
    enforces an end-to-end deadline, checked between chunk reads.

    A non-200 response is rejected before any bytes are written: a proxy
    interstitial or an unfollowed redirect returned with a non-200 status
    must not be streamed to disk and then surface downstream as a
    misleading SHA-256 failure (which reads as a supply-chain anomaly).
    `urlopen` already raises `HTTPError` for 4xx/5xx, so this guards the
    residual 2xx/3xx cases.

    Raises:
        TimeoutError: When total transfer time exceeds the deadline.
        urllib.error.URLError: When the response status is not 200.
    """
    import time
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + _DOWNLOAD_TIMEOUT_SECONDS
    with (
        urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as resp,  # noqa: S310  # fixed https GitHub release URL
        dest.open("wb") as fh,
    ):
        status = getattr(resp, "status", None)
        if status is not None and status != 200:  # noqa: PLR2004  # HTTP 200 OK
            msg = f"Unexpected HTTP {status} response fetching {url}"
            raise urllib.error.URLError(msg)
        while True:
            if time.monotonic() > deadline:
                msg = (
                    f"Download of {url} exceeded {_DOWNLOAD_TIMEOUT_SECONDS}s deadline"
                )
                raise TimeoutError(msg)
            chunk = resp.read(_DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            fh.write(chunk)


def _verify_sha256(path: Path, expected_hex: str) -> None:
    """Verify `path` matches `expected_hex`.

    Raises:
        ChecksumMismatchError: When the SHA-256 of `path` differs from
            `expected_hex`.
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
        raise ChecksumMismatchError(msg)


def _validate_legacy_tar_member(member: tarfile.TarInfo, extract_root: Path) -> None:
    """Reject tar members that cannot be safely extracted without filters.

    Raises:
        tarfile.TarError: If a member would extract outside `extract_root`
            or uses a tar entry type this fallback does not support.
    """
    import tarfile

    target = extract_root / member.name
    try:
        target.resolve().relative_to(extract_root.resolve())
    except ValueError as exc:
        msg = f"Refusing to extract unsafe tar member {member.name!r}"
        raise tarfile.TarError(msg) from exc

    if not (member.isfile() or member.isdir()):
        msg = f"Refusing to extract unsupported tar member {member.name!r}"
        raise tarfile.TarError(msg)


def _extract_tar_data(tf: tarfile.TarFile, extract_root: Path) -> None:
    """Extract a tar archive with `data` filtering when available.

    Python versions before the PEP 706 backport (3.11.0-3.11.3) lack the
    `filter` keyword on `extractall`, and this package supports those patch
    versions. The fallback validates the pinned release archive before
    using the legacy API.

    Raises:
        TypeError: Re-raised when `extractall` rejects a non-`filter`
            keyword (i.e. an unrelated `TypeError` we should not swallow).
    """
    try:
        tf.extractall(extract_root, filter="data")
    except TypeError as exc:
        if "filter" not in str(exc):
            raise
        members = tf.getmembers()
        for member in members:
            _validate_legacy_tar_member(member, extract_root)
        tf.extractall(extract_root, members=members)  # noqa: S202  # validated above


def _extract_rg(archive: Path, extract_root: Path) -> Path:
    """Extract `archive` and locate the `rg` binary inside.

    Handles both `.tar.gz` and `.zip` archives. Release archives nest the
    binary under `ripgrep-<ver>-<triple>/`, so we walk the tree to find it
    rather than hard-coding the prefix. Malformed archives or unsafe
    members propagate `tarfile.TarError` / `zipfile.BadZipFile`.

    Returns:
        Absolute path to the extracted `rg` (or `rg.exe`) binary.

    Raises:
        FileNotFoundError: When the archive does not contain an `rg` binary.
    """
    import tarfile
    import zipfile

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            _extract_zip_validated(zf, extract_root)
    else:
        with tarfile.open(archive, mode="r:*") as tf:
            _extract_tar_data(tf, extract_root)

    target_name = "rg.exe" if sys.platform == "win32" else "rg"
    for path in extract_root.rglob(target_name):
        if path.is_file():
            return path
    msg = f"Could not find {target_name} inside {archive.name}"
    raise FileNotFoundError(msg)


def _extract_zip_validated(zf: zipfile.ZipFile, extract_root: Path) -> None:
    """Extract a zip archive after validating each member's path.

    `ZipFile.extractall` does sanitize absolute paths and parent-relative
    components on modern Python, but defense-in-depth here keeps the
    SHA-256-verified archive from being the only line of defense against
    a zip-slip variant in a future upstream archive.

    Raises:
        zipfile.BadZipFile: If a member would extract outside `extract_root`.
    """
    import zipfile

    extract_root.mkdir(parents=True, exist_ok=True)
    root = extract_root.resolve()
    for member in zf.infolist():
        target = (extract_root / member.filename).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            msg = f"Refusing to extract unsafe zip member {member.filename!r}"
            raise zipfile.BadZipFile(msg) from exc
    zf.extractall(extract_root)  # noqa: S202  # validated above


def _install_ripgrep_sync(asset: str, sha256: str) -> Path:
    """Download, verify, extract, and install ripgrep atomically.

    Staging happens *inside* `BIN_DIR` so the final rename is on the same
    filesystem and therefore atomic on POSIX. On Windows it is also atomic
    when the destination is not in use; a process holding the existing
    `rg.exe` open will see `PermissionError`, which the caller surfaces
    rather than silently corrupts. `_verify_sha256` propagates
    `ChecksumMismatchError` to abort install before any move.

    Returns:
        Absolute path to the installed `rg` binary.
    """
    import tempfile

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{_RELEASE_URL_PREFIX}/{asset}"
    with tempfile.TemporaryDirectory(prefix=".deepagents-rg-", dir=BIN_DIR) as tmp_str:
        tmp = Path(tmp_str)
        archive = tmp / asset
        _download_to(url, archive)
        _verify_sha256(archive, sha256)
        extracted = _extract_rg(archive, tmp / "unpacked")
        if sys.platform != "win32":
            extracted.chmod(0o755)
        dest = managed_rg_path()
        extracted.replace(dest)
        return dest


async def ensure_ripgrep() -> Path | None:
    """Ensure a usable `rg` binary is available, installing if necessary.

    Resolution order:

    1. If a managed `rg` exists *and* matches `RIPGREP_VERSION`, return it.
    2. Otherwise, if a system `rg` is on `PATH` and no managed binary
        exists, return its resolved path. This is gated on the *absence*
        of a managed binary: once a managed `rg` exists, the pinned
        version always wins, so a stale managed binary is re-fetched
        rather than deferring to a system `rg` and the resolved version
        stays deterministic.
    3. If offline, on an unsupported platform, or no asset matches the
        platform/arch, return `None` so callers fall back to the existing
        notification + slow path.
    4. Otherwise download → SHA-256 verify → extract → install →
        prepend `BIN_DIR` to `PATH` → return the installed path. On a
        checksum mismatch, raises `ChecksumMismatchError` so callers can
        surface a loud notice; other failures log and return `None`.

    A stale managed binary is never proactively deleted. The atomic
    replace in `_install_ripgrep_sync` overwrites it on success, and on
    failure the user is strictly better off keeping the older copy than
    being left with no `rg` at all.

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
    managed_exists = managed.exists()
    if managed_exists and _managed_binary_is_current(managed):
        return managed

    if not managed_exists:
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

    if managed_exists:
        logger.info(
            "Managed ripgrep at %s is stale; replacing with %s",
            managed,
            RIPGREP_VERSION,
        )

    try:
        # `_install_ripgrep_sync` atomically replaces the destination on
        # success, so we deliberately leave any stale binary in place
        # until the verified replacement is ready. A failed download must
        # not strand the user with no `rg` at all.
        installed = await asyncio.to_thread(_install_ripgrep_sync, asset, sha256)
    except (urllib.error.URLError, TimeoutError):
        logger.warning(
            "Could not download ripgrep from %s", _RELEASE_URL_PREFIX, exc_info=True
        )
        return None
    except (tarfile.TarError, zipfile.BadZipFile, FileNotFoundError) as exc:
        logger.exception(
            "ripgrep install failed: archive error (%s)", type(exc).__name__
        )
        return None
    except PermissionError:
        logger.exception(
            "ripgrep install failed: cannot write to %s — check permissions", BIN_DIR
        )
        return None
    except OSError as exc:
        logger.exception(
            "ripgrep install failed: %s (errno=%s)", type(exc).__name__, exc.errno
        )
        return None
    else:
        prepend_managed_bin_to_path()
        return installed
