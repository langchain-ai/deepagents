"""Auto-install pinned upstream binaries for optional tools.

Today this only manages `ripgrep`. The SDK shells out to `rg` via `PATH`,
so installing inside the dcode tool environment and prepending that directory
to `os.environ["PATH"]` is sufficient — no SDK change required. Keeping helper
binaries installation-scoped lets multiple profiles reuse one verified binary.

`FALLBACK_BIN_DIR` covers the case where that shared directory is not writable
(a system or root-owned `sys.prefix`). Run `dcode doctor` to see which of the
two locations is actually in use.

The pinned `RIPGREP_VERSION`, archive hashes in `RIPGREP_ASSETS`, and extracted
binary hashes in `RIPGREP_BINARY_SHA256` are the source of truth for what gets
downloaded and executed. Refresh all three together when bumping the version.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from deepagents_code._env_vars import OFFLINE, RIPGREP_INSTALLER, is_env_truthy
from deepagents_code._paths import (
    PATHS,
    PathState,
    classify_path,
    first_writable,
)

if TYPE_CHECKING:
    import tempfile
    import zipfile

logger = logging.getLogger(__name__)

RIPGREP_VERSION = "14.1.1"
"""Pinned release. Bump alongside both SHA-256 tables."""

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

RIPGREP_BINARY_SHA256: dict[tuple[str, str], str] = {
    ("darwin", "arm64"): (
        "0e0cb83f5195f1f51bb8feef1fff5b0b171e82bd1db6bd35deee701a3e7102f8"
    ),
    ("darwin", "x86_64"): (
        "923dcc25cab57d33f4e7dd0476d4b74a554401a38817e246a8d6101dcd51c50f"
    ),
    ("linux", "arm64"): (
        "e07d5c85fa9ca740ff4ab8bbac60a1e11c7a5ce242435f7820a03f7c20ef6276"
    ),
    ("linux", "x86_64"): (
        "f401154e2393f9002ac77e419f9ee5521c18f4f8cd3e32293972f493ba06fce7"
    ),
    ("win32", "arm64"): (
        "f162b54de2adfc72d78adb1dbada2dedda111ae0a5e2f6e9500f4f909664c5d2"
    ),
    ("win32", "x86_64"): (
        "f162b54de2adfc72d78adb1dbada2dedda111ae0a5e2f6e9500f4f909664c5d2"
    ),
}
"""SHA-256 of the extracted `rg` binary in each pinned release asset."""

BIN_DIR: Path = PATHS.installation.managed_bin_dir
"""Preferred directory for managed binaries, shared by every profile.

Prepended to `PATH` on startup. Tied to the installation rather than the
profile so relocating `DEEPAGENTS_HOME` reuses one verified download.
"""

FALLBACK_BIN_DIR: Path = PATHS.profile.bin_dir
"""Profile-scoped bin directory used when `BIN_DIR` is not writable.

A system or root-owned install prefix (`pip install --break-system-packages`,
a packaged interpreter) leaves `BIN_DIR` unwritable for a normal user, which
would otherwise mean no managed ripgrep at all.
"""

_FALLBACK_SHIM: (
    tuple[tempfile.TemporaryDirectory[str], tuple[str, tuple[int, int, int]]] | None
) = None
"""Process-private `PATH` shim and the fallback binary identity it exposes."""


def managed_bin_dirs() -> tuple[Path, ...]:
    """Return both managed bin locations in preference order.

    Read from the module globals on each call rather than frozen into a
    constant, so a test (or a future runtime override) that patches `BIN_DIR`
    is honored by lookup, `PATH` assembly, and install alike.

    Returns:
        The preferred installation directory followed by the profile fallback.
    """
    return (BIN_DIR, FALLBACK_BIN_DIR)


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


UnavailableReason = Literal["unsupported", "artifact_not_found", "permission_denied"]
"""Stable reason token for logging and telemetry."""


class ManagedToolUnavailableError(Exception):
    """Raised when no managed helper binary is available for this system.

    Distinct from transient network/download failures so callers can tell users
    whether retrying can help or whether they need a different install path.
    """

    def __init__(
        self, *, tool: Literal["ripgrep"], reason: UnavailableReason, message: str
    ) -> None:
        """Initialize the unavailable-tool error.

        Args:
            tool: Managed helper name.
            reason: Stable reason token for logging and telemetry.
            message: User-facing remediation message.
        """
        super().__init__(message)
        self.tool = tool
        self.reason = reason
        self.message = message


def _normalized_arch() -> str | None:
    """Return a normalized arch key matching `RIPGREP_ASSETS`.

    Returns `None` for unsupported architectures (e.g. 32-bit, ppc, s390x).
    """
    import platform

    raw = platform.machine().lower()
    return _ARCH_ALIASES.get(raw)


def _unsupported_ripgrep_error(
    platform_name: str, arch: str | None
) -> ManagedToolUnavailableError:
    """Return a clear unsupported-platform error for managed ripgrep."""
    target = platform_name if arch is None else f"{platform_name}/{arch}"
    return ManagedToolUnavailableError(
        tool="ripgrep",
        reason="unsupported",
        message=(
            f"Managed ripgrep is not available for this system ({target}). "
            "Install ripgrep manually, or set DEEPAGENTS_CODE_RIPGREP_INSTALLER=system."
        ),
    )


def _unwritable_bin_dir_error() -> ManagedToolUnavailableError:
    """Return a clear write-failure error naming both managed bin directories.

    The wording covers every reason a write can fail, not permissions alone.
    A read-only filesystem, a full disk, and an exceeded quota all reach this
    error, and telling those users to "check the permissions" sends them after
    the wrong cause.
    """
    return ManagedToolUnavailableError(
        tool="ripgrep",
        reason="permission_denied",
        message=(
            f"Could not write ripgrep to {BIN_DIR} or {FALLBACK_BIN_DIR}. "
            "Check that one of them is writable and that the filesystem is "
            "not full or read-only, or install ripgrep with your package "
            "manager."
        ),
    )


def _artifact_not_found_error(
    platform_name: str, arch: str
) -> ManagedToolUnavailableError:
    """Return a clear missing-artifact error for managed ripgrep."""
    return ManagedToolUnavailableError(
        tool="ripgrep",
        reason="artifact_not_found",
        message=(
            f"Managed ripgrep artifact for {platform_name}/{arch} was not found "
            f"in pinned ripgrep {RIPGREP_VERSION}. Install ripgrep manually, or "
            "try a newer dcode version."
        ),
    )


def managed_rg_filename() -> str:
    """Return the managed ripgrep filename for this platform."""
    return "rg.exe" if sys.platform == "win32" else "rg"


def managed_rg_path() -> Path:
    """Return the managed ripgrep binary path (`.exe` on Windows).

    Returns:
        A current candidate across both locations, the first existing stale
        candidate, or the preferred location when neither exists.

    Note:
        The version probe runs only when both locations hold a binary, which
        is the one case where the answer is not already determined. That probe
        starts a subprocess, and this function is called several times per
        launch, so `_managed_binary_is_current` memoizes on file identity.
    """
    candidates = [directory / managed_rg_filename() for directory in managed_bin_dirs()]
    existing = [
        candidate
        for candidate in candidates
        if classify_path(candidate) is PathState.EXISTS
    ]
    if not existing:
        return candidates[0]
    if len(existing) == 1:
        return existing[0]
    return next(
        (candidate for candidate in existing if _managed_binary_is_current(candidate)),
        existing[0],
    )


def is_offline() -> bool:
    """Return whether managed-tool downloads are disabled via env var."""
    return is_env_truthy(OFFLINE)


RipgrepInstaller = Literal["managed", "system"]
"""The two recognized ripgrep installer modes."""

INSTALLER_MANAGED: RipgrepInstaller = "managed"
"""Default installer mode: fetch the pinned, checksummed upstream binary."""

INSTALLER_SYSTEM: RipgrepInstaller = "system"
"""Installer mode that defers ripgrep to the system package manager / `PATH`."""


def ripgrep_installer() -> RipgrepInstaller:
    """Return the configured ripgrep installer mode.

    Reads `RIPGREP_INSTALLER` and normalizes it to `INSTALLER_MANAGED` or
    `INSTALLER_SYSTEM`, falling back to `INSTALLER_MANAGED` for unset or
    unrecognized values. The `strip().lower()` normalization must stay in
    sync with the `case` block in `scripts/install.sh` so both layers agree
    on the parsed mode.
    """
    raw = os.environ.get(RIPGREP_INSTALLER, "").strip().lower()
    if raw == INSTALLER_SYSTEM:
        return INSTALLER_SYSTEM
    if raw and raw != INSTALLER_MANAGED:
        logger.warning(
            "Unrecognized %s=%r; expected %r or %r. Defaulting to %r.",
            RIPGREP_INSTALLER,
            raw,
            INSTALLER_MANAGED,
            INSTALLER_SYSTEM,
            INSTALLER_MANAGED,
        )
    return INSTALLER_MANAGED


def prefers_system_ripgrep() -> bool:
    """Return whether the user opted into the `system` ripgrep installer.

    In `system` mode ripgrep is provisioned by the OS package manager or an
    existing `PATH` entry rather than the managed download.
    """
    return ripgrep_installer() == INSTALLER_SYSTEM


def prepend_managed_bin_to_path() -> None:
    """Idempotently expose managed ripgrep through `os.environ["PATH"]`.

    Safe to call on every startup. The installation-scoped directory is
    prepended directly. A verified profile fallback is exposed through a
    process-private shim containing only `rg`, because the profile may be
    repository-controlled. Both managed directories are removed from the rest
    of `PATH` so neither a stale copy nor a fallback sibling can shadow it.

    Prepending only one directory matters for the profile fallback. TB14
    permits a `DEEPAGENTS_HOME` inside a checkout, so `<profile>/bin` can be a
    repository-controlled directory. Verifying `rg` does not make siblings
    such as `git` trustworthy, so that directory never enters `PATH`.
    """
    candidate = managed_rg_path()
    active_dir = candidate.parent
    if active_dir == FALLBACK_BIN_DIR:
        active_dir = _verified_fallback_shim(candidate) or BIN_DIR
    active = str(active_dir)
    managed = {str(directory) for directory in managed_bin_dirs()}
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    desired = [active, *(p for p in parts if p not in managed)]
    if parts == desired:
        return
    os.environ["PATH"] = os.pathsep.join(desired)


def _verified_fallback_shim(binary: Path) -> Path | None:
    """Return a private directory exposing only a checksum-verified `binary`."""
    global _FALLBACK_SHIM  # noqa: PLW0603  # process-lifetime shim cache

    identity = _binary_identity(binary)
    if identity is None:
        return None
    source = (str(binary), identity)
    if _FALLBACK_SHIM is not None and _FALLBACK_SHIM[1] == source:
        return Path(_FALLBACK_SHIM[0].name)
    shim = _create_verified_fallback_shim(binary)
    if shim is None:
        return None
    previous = _FALLBACK_SHIM
    _FALLBACK_SHIM = (shim, source)
    if previous is not None:
        previous[0].cleanup()
    return Path(shim.name)


def _create_verified_fallback_shim(
    binary: Path,
) -> tempfile.TemporaryDirectory[str] | None:
    """Create a private snapshot that contains no profile-controlled siblings.

    Returns:
        The live temporary directory, or `None` if the entrypoint cannot be
        created and verified.
    """
    import shutil
    import tempfile

    shim = tempfile.TemporaryDirectory(
        prefix="deepagents-rg-shim-", ignore_cleanup_errors=True
    )
    target = Path(shim.name) / managed_rg_filename()
    try:
        shutil.copy2(binary, target)
        if _managed_binary_is_verified(target):
            return shim
    except OSError:
        logger.warning(
            "Could not create an isolated PATH shim for ripgrep at %s",
            binary,
            exc_info=True,
        )
    shim.cleanup()
    return None


def _cleanup_fallback_shim() -> None:
    """Remove the process-private fallback shim at interpreter shutdown."""
    global _FALLBACK_SHIM  # noqa: PLW0603  # process-lifetime shim cache

    if _FALLBACK_SHIM is not None:
        _FALLBACK_SHIM[0].cleanup()
        _FALLBACK_SHIM = None


atexit.register(_cleanup_fallback_shim)


def _path_without_managed_bin() -> str | None:
    """Return `PATH` with every managed bin dir removed."""
    current = os.environ.get("PATH")
    if not current:
        return None

    managed_dirs = {d.resolve() for d in managed_bin_dirs()}
    parts = [
        part
        for part in current.split(os.pathsep)
        if not part or Path(part).resolve() not in managed_dirs
    ]
    return os.pathsep.join(parts)


def _binary_identity(binary: Path) -> tuple[int, int, int] | None:
    """Return a stat identity for `binary`, or `None` when it cannot be read.

    Used as a memo key for the version probe. An install replaces the file, so
    the inode, size, or mtime changes and the next probe misses the memo.

    Returns:
        The inode, size, and mtime in nanoseconds, or `None`.
    """
    try:
        stat_result = binary.stat()
    except OSError:
        return None
    return (stat_result.st_ino, stat_result.st_size, stat_result.st_mtime_ns)


def _managed_binary_is_current(binary: Path) -> bool:
    """Return whether the on-disk managed `rg` matches `RIPGREP_VERSION`.

    The binary's pinned SHA-256 is checked before it is executed. Returns
    `False` on any concrete failure (checksum mismatch, `OSError`, non-zero
    exit, empty stdout, version mismatch) so an unverified profile fallback or
    a corrupted install gets replaced. Only `TimeoutExpired` "falls open" —
    after checksum verification, that case suggests a sandboxed subprocess
    rather than a broken binary.

    The result is memoized on the binary's stat identity. `managed_rg_path`
    calls this whenever both bin directories hold a binary, and several call
    sites reach `managed_rg_path` on one launch, so an uncached probe starts
    the same subprocess four to six times.
    """
    identity = _binary_identity(binary)
    if identity is not None:
        memo_key = (str(binary), identity)
        cached = _VERSION_PROBE_MEMO.get(memo_key)
        if cached is not None:
            return cached
        result_is_current = _managed_binary_is_verified(
            binary
        ) and _probe_managed_binary_version(binary)
        _VERSION_PROBE_MEMO[memo_key] = result_is_current
        return result_is_current
    return _managed_binary_is_verified(binary) and _probe_managed_binary_version(binary)


_VERSION_PROBE_MEMO: dict[tuple[str, tuple[int, int, int]], bool] = {}
"""Memo for `_managed_binary_is_current`, keyed on path and stat identity."""


def _managed_binary_is_verified(binary: Path) -> bool:
    """Return whether `binary` matches the pinned upstream executable bytes."""
    arch = _normalized_arch()
    expected = None if arch is None else RIPGREP_BINARY_SHA256.get((sys.platform, arch))
    if expected is None:
        return False
    try:
        return _sha256(binary) == expected
    except OSError:
        logger.debug("Could not checksum managed ripgrep at %s", binary, exc_info=True)
        return False


def _probe_managed_binary_version(binary: Path) -> bool:
    """Run `rg --version` and report whether it matches `RIPGREP_VERSION`.

    Returns:
        Whether the binary reports the pinned version.
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


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of `path`."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected_hex: str) -> None:
    """Verify `path` matches `expected_hex`.

    Raises:
        ChecksumMismatchError: When the SHA-256 of `path` differs from
            `expected_hex`.
    """
    actual = _sha256(path)
    if actual != expected_hex:
        msg = (
            f"Checksum mismatch for {path.name}: expected {expected_hex}, got {actual}"
        )
        raise ChecksumMismatchError(msg)


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
            tf.extractall(extract_root, filter="data")

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


class _NoWritableBinDirError(OSError):
    """Raised when no managed bin directory can be created or written.

    Its own type rather than a bare `OSError` because the caller must map this
    to a visible message regardless of why the write failed. Selecting on
    `PermissionError` catches EACCES and EPERM only, so a read-only filesystem
    or a full disk would fall through to the generic handler and produce the
    "ripgrep is not installed" hint for a problem no package manager can fix.
    """


def _resolve_install_bin_dir() -> Path:
    """Return the first managed bin dir that can be created.

    Returns:
        A usable, existing bin directory.

    Raises:
        _NoWritableBinDirError: If no candidate can be created or written. Its
            message names both directories; `first_writable` has already logged
            each candidate's own `OSError` with a traceback.
    """
    directory = first_writable(managed_bin_dirs(), what="Managed bin")
    if directory is None:
        msg = (
            f"No managed bin directory could be created or written: tried "
            f"{BIN_DIR} and {FALLBACK_BIN_DIR}."
        )
        raise _NoWritableBinDirError(msg)
    if directory != BIN_DIR:
        logger.warning(
            "Installing ripgrep to the profile directory %s because the "
            "shared installation directory %s is not writable",
            directory,
            BIN_DIR,
        )
    return directory


def _install_ripgrep_sync(asset: str, sha256: str) -> Path:
    """Download, verify, extract, and install ripgrep atomically.

    Staging happens *inside* the chosen bin directory so the final rename is on
    the same filesystem and therefore atomic on POSIX. Windows keeps replacing
    the user-facing `rg.exe` directly because symlink support varies by
    developer mode and policy. POSIX installs use a versioned real binary plus a
    relative `rg` symlink so moving or bind-mounting the tool environment does
    not bake in its original absolute path. `_verify_sha256` propagates
    `ChecksumMismatchError` to abort install before any move.

    Returns:
        Absolute path to the installed `rg` entrypoint.
    """
    import os
    import tempfile

    bin_dir = _resolve_install_bin_dir()
    url = f"{_RELEASE_URL_PREFIX}/{asset}"
    with tempfile.TemporaryDirectory(prefix=".deepagents-rg-", dir=bin_dir) as tmp_str:
        tmp = Path(tmp_str)
        archive = tmp / asset
        _download_to(url, archive)
        _verify_sha256(archive, sha256)
        extracted = _extract_rg(archive, tmp / "unpacked")
        if sys.platform != "win32":
            extracted.chmod(0o755)
        name = "rg.exe" if sys.platform == "win32" else "rg"
        dest = bin_dir / name
        if sys.platform == "win32":
            extracted.replace(dest)
            return dest

        real = bin_dir / f"rg-{RIPGREP_VERSION}"
        extracted.replace(real)
        link = tmp / "rg-link"
        link.symlink_to(os.path.relpath(real, start=bin_dir))
        link.replace(dest)
        return dest


async def ensure_ripgrep() -> Path | None:
    """Ensure a usable `rg` binary is available, installing if necessary.

    Resolution order:

    1. If the `system` installer is selected, return a non-managed `rg`
        found on `PATH`, or `None` when only the managed binary is present.
    2. If a managed `rg` exists *and* matches `RIPGREP_VERSION`, return it.
    3. Otherwise, if a system `rg` is on `PATH` and no managed binary
        exists, return its resolved path. This is gated on the *absence*
        of a managed binary: once a managed `rg` exists, the pinned
        version always wins, so a stale managed binary is re-fetched
        rather than deferring to a system `rg` and the resolved version
        stays deterministic.
    4. If offline, return `None` so callers fall back to the existing
        notification + slow path.
    5. If no managed asset matches the platform/arch, return a non-managed
        `rg` on `PATH` when one exists; otherwise raise
        `ManagedToolUnavailableError` so callers can explain that retrying will
        not help.
    6. Otherwise download → SHA-256 verify → extract → install →
        prepend the active managed bin dir to `PATH` → return the installed path. On a
        checksum mismatch, raises `ChecksumMismatchError` so callers can
        surface a loud notice. On a 404, or when neither managed bin directory
        is writable, raises `ManagedToolUnavailableError`; other failures log
        and return `None`.

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
    managed_exists = managed.exists() or managed.is_symlink()

    def non_managed_rg() -> Path | None:
        system_rg = shutil.which("rg", path=_path_without_managed_bin())
        if system_rg is not None:
            return Path(system_rg)
        return None

    if prefers_system_ripgrep():
        system_rg_path = non_managed_rg()
        if system_rg_path is not None:
            return system_rg_path
        logger.debug(
            "Skipping managed ripgrep download: %s=%s",
            RIPGREP_INSTALLER,
            INSTALLER_SYSTEM,
        )
        return None

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
        system_rg_path = non_managed_rg()
        if system_rg_path is not None:
            return system_rg_path
        error = _unsupported_ripgrep_error(sys.platform, None)
        raise error

    arch = _normalized_arch()
    if arch is None:
        logger.debug(
            "Skipping ripgrep install: unsupported arch %r", platform.machine()
        )
        system_rg_path = non_managed_rg()
        if system_rg_path is not None:
            return system_rg_path
        error = _unsupported_ripgrep_error(sys.platform, platform.machine())
        raise error

    asset_entry = RIPGREP_ASSETS.get((sys.platform, arch))
    if asset_entry is None:
        logger.debug(
            "Skipping ripgrep install: no asset for (%s, %s)", sys.platform, arch
        )
        system_rg_path = non_managed_rg()
        if system_rg_path is not None:
            return system_rg_path
        error = _unsupported_ripgrep_error(sys.platform, arch)
        raise error
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
    except urllib.error.HTTPError as exc:
        if exc.code == 404:  # noqa: PLR2004  # HTTP 404 Not Found
            logger.warning(
                "Managed ripgrep artifact was not found: %s/%s", sys.platform, arch
            )
            error = _artifact_not_found_error(sys.platform, arch)
            raise error from exc
        logger.warning(
            "Could not download ripgrep from %s", _RELEASE_URL_PREFIX, exc_info=True
        )
        return None
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
    except (_NoWritableBinDirError, PermissionError) as exc:
        # The bin directory could not be written. That happens before the
        # download (`_resolve_install_bin_dir`) or after it (the install
        # itself), and both reach here. Returning None would send the user the
        # caller's generic "ripgrep is not installed — brew install ripgrep"
        # hint for a problem `brew` cannot fix. Raise instead: every caller
        # renders this message visibly, while the log line here is invisible
        # without --debug.
        logger.exception(
            "ripgrep install failed: cannot write to %s or %s",
            BIN_DIR,
            FALLBACK_BIN_DIR,
        )
        error = _unwritable_bin_dir_error()
        raise error from exc
    except OSError as exc:
        logger.exception(
            "ripgrep install failed: %s (errno=%s)", type(exc).__name__, exc.errno
        )
        return None
    else:
        prepend_managed_bin_to_path()
        return installed
