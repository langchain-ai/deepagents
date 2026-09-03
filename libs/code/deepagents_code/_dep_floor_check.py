"""Best-effort runtime dependency floor check for editable dev installs.

Editable installs resolve dependencies once at install time and nothing
re-checks them afterwards, so after `pyproject.toml` floors are bumped on
`main` a stale editable venv silently runs new source against old deps.
This module detects editable installs via PEP 610 metadata and warns at
startup when an installed runtime dependency is older than the floor the
checkout declares. Released installs (uv tool / PyPI wheel) carry no
editable `direct_url.json` record and skip the check entirely, so end
users pay no startup cost here.

Interactive launches on a terminal stop on a blocking pre-TUI prompt
(continue / mute / abort); headless launches, subcommands, and interactive
launches whose stdin is piped get a single stderr warning per launch instead,
since they have no answerable prompt and must never block. A dismissed
(muted) mismatch is remembered per checkout as a fingerprint of the offending
distributions and stays silent until the mismatch itself changes.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import shlex
import subprocess  # noqa: S404  # only `list2cmdline` string quoting, never executed
import sys
import tempfile
import threading
import tomllib
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code.config import _is_editable_install

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents_code.main import _TrustPromptOutcome

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FloorViolation:
    """A runtime dependency whose installed version is below the declared floor.

    Attributes:
        dist_name: Distribution name as passed to `importlib.metadata.version`.
        installed: The installed version string.
        floor: The highest declared floor version `installed` sorts below.
        required: The requirement spec as declared in `pyproject.toml`, kept
            for display only. `floor` is the parsed value to compare against;
            this string is `packaging`'s rendering and may shift across
            `packaging` releases.
    """

    dist_name: str
    installed: str
    floor: str
    required: str

    def describe(self) -> str:
        """Return the one-line advisory row for this violation.

        Both the stderr warning and the interactive prompt render violations,
        so the sentence lives here rather than being duplicated per channel.
        """
        return (
            f"{self.dist_name} {self.installed} is installed, "
            f"but {self.required} is required"
        )


def _load_cli_requirements() -> list[str] | None:
    """Read the hard dependency list from the editable source checkout.

    The installed `Requires-Dist` metadata is frozen at install time, so a
    floor bump on `main` would not reach a stale editable venv through it.
    The checkout's own `pyproject.toml` is live, and editable installs keep
    the console entry point importing from the checkout, so the package
    directory's parent holds it.

    Returns:
        The `project.dependencies` entries, or `None` when the source
        checkout or its dependency list cannot be read.
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        entries = data["project"]["dependencies"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        logger.debug(
            "Could not read dependency floors from the source checkout",
            exc_info=True,
        )
        return None
    if not isinstance(entries, list) or not all(
        isinstance(entry, str) for entry in entries
    ):
        return None
    return entries


def _find_floor_violations(entries: list[str]) -> list[_FloorViolation]:
    """Compare each applicable requirement's floor against installed versions.

    Args:
        entries: Requirement strings from `project.dependencies`.

    Returns:
        One `_FloorViolation` per requirement whose installed version parses
        below its declared `>=`/`~=`/`==` floor. Requirements that fail to
        parse, whose marker does not apply to this environment, whose only
        equality specifier is a `==X.*` wildcard, or whose distribution is
        not installed are skipped.
    """
    # `packaging` is a runtime dependency, but this module must remain
    # importable when an editable environment is stale enough to lack it.
    # The caller treats failures in this best-effort check as non-fatal.
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.version import InvalidVersion, Version

    violations: list[_FloorViolation] = []
    for entry in entries:
        try:
            req = Requirement(entry)
        except InvalidRequirement:
            logger.debug("Unparseable dependency entry %r; skipping", entry)
            continue
        except Exception:  # any unexpected parser failure skips the entry
            logger.debug(
                "Requirement parser failed on %r; skipping", entry, exc_info=True
            )
            continue
        if req.marker is not None:
            try:
                if not req.marker.evaluate():
                    continue
            except Exception:  # an unevaluatable marker must not abort the whole check
                logger.debug(
                    "Could not evaluate marker for %r; skipping", entry, exc_info=True
                )
                continue
        # `==` pins act as floors here too: a hard pin (e.g. the SDK's
        # `deepagents==X.Y.Z`) still means "this version or drift" in an
        # editable venv, and an older-than-pinned install is exactly the
        # staleness this check exists to surface. `==X.*` wildcard equality
        # has no parseable single version, so it stays out.
        floors = [
            spec.version
            for spec in req.specifier
            if spec.operator in {">=", "~="}
            or (spec.operator == "==" and not spec.version.endswith(".*"))
        ]
        if not floors:
            continue
        try:
            installed = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            # Workspace `[tool.uv.sources]` path deps and not-yet-installed
            # dists both land here; a missing dist is not a stale floor.
            logger.debug("Distribution %r is not installed; skipping", req.name)
            continue
        except Exception:
            # A corrupt or half-written `.dist-info` raises from the metadata
            # parser. Skip just this entry: letting it escape would discard
            # violations already found and leave the rest of the list
            # unexamined, reporting a false all-clear for every dependency.
            logger.debug(
                "Could not read the installed version of %r; skipping",
                req.name,
                exc_info=True,
            )
            continue
        try:
            installed_version = Version(installed)
        except InvalidVersion:
            logger.debug("Unparseable installed version %r for %r", installed, req.name)
            continue
        floor = max(floors, key=_version_key)
        try:
            floor_version = Version(floor)
        except InvalidVersion:
            # `_version_key` sorts unparseable floors lowest, so this is only
            # reachable when every floor for the entry is unparseable.
            logger.debug("Unparseable floor %r for %r; skipping", floor, req.name)
            continue
        if installed_version < floor_version:
            violations.append(
                _FloorViolation(
                    dist_name=req.name,
                    installed=installed,
                    floor=floor,
                    required=str(req),
                )
            )
    return violations


def _quote_arg(arg: str) -> str:
    """Quote a command argument for the platform's command-line conventions.

    `shlex.quote` emits POSIX single quotes, which `cmd.exe` does not treat
    as quoting — on Windows the quote characters would become part of the
    path, so the printed refresh command would not work there.
    `subprocess.list2cmdline` follows the Windows argv rules instead.

    Args:
        arg: The raw argument string.

    Returns:
        The argument quoted for the current platform's shell conventions.
    """
    if sys.platform == "win32":
        return subprocess.list2cmdline([arg])
    return shlex.quote(arg)


def _version_key(version: str) -> tuple[int, ...]:
    """Return a sortable key for a floor version string.

    Args:
        version: A PEP 440 version string from a `>=`/`~=` specifier.

    Returns:
        The release tuple when parseable, else an empty tuple so malformed
        floors sort lowest and a parseable floor always wins the `max`.
    """
    from packaging.version import InvalidVersion, Version

    try:
        return Version(version).release
    except InvalidVersion:
        return ()


_DEBUG_VIOLATION = _FloorViolation(
    dist_name="packaging",
    installed="0.0.1",
    floor="26.2",
    required="packaging>=26.2",
)
"""Fake below-floor violation used by `DEEPAGENTS_CODE_DEBUG_DEP_FLOOR`.

Lets the warn/prompt/mute flow be exercised end to end without a genuinely
stale environment or monkeypatching internals. This one instance deliberately
misreports the environment, so "an instance describes a real installed
version" is not an invariant of `_FloorViolation`.
"""


def _checkout_root() -> Path:
    """Return the editable source checkout root this module runs from."""
    return Path(__file__).resolve().parent.parent


def refresh_command() -> str:
    """Return the shell command that refreshes this editable environment.

    Shared by the stderr warning and the interactive prompt so the two
    channels cannot drift apart.
    """
    return (
        "uv pip install --python "
        f"{_quote_arg(sys.executable)} -e {_quote_arg(str(_checkout_root()))} "
        "--upgrade"
    )


def _collect_violations() -> list[_FloorViolation]:
    """Detect below-floor dependencies for this editable install.

    `DEEPAGENTS_CODE_DEBUG_DEP_FLOOR` short-circuits to a synthetic
    violation before the editable gate and any real metadata reads.

    Returns:
        The offending distributions; empty when the install is released,
        the checkout cannot be read, or every floor is satisfied.
    """
    from deepagents_code._env_vars import DEBUG_DEP_FLOOR, is_env_truthy

    if is_env_truthy(DEBUG_DEP_FLOOR):
        return [_DEBUG_VIOLATION]
    if not _is_editable_install():
        return []
    entries = _load_cli_requirements()
    if entries is None:
        return []
    return _find_floor_violations(entries)


def _mismatch_fingerprint(violations: list[_FloorViolation]) -> str:
    """Hash the mismatch so a dismissal stays valid only while it is unchanged.

    Args:
        violations: The offending distributions.

    Returns:
        A hex digest over sorted `name/installed/floor` rows. Any change —
        a refreshed dep, a moved floor, a newly stale package — produces a
        different fingerprint and re-arms the warning.

        The rows deliberately exclude `required`: it is `packaging`'s rendering
        of the requirement, so hashing it would let a `packaging` upgrade that
        reorders specifiers silently invalidate every stored dismissal.
    """
    rows = sorted(f"{v.dist_name}/{v.installed}/{v.floor}" for v in violations)
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def format_dep_floor_warning(violations: list[_FloorViolation]) -> str:
    """Build the full advisory: one line per violation plus the refresh command.

    Args:
        violations: The offending distributions.

    Returns:
        Plain-text (no Rich markup) advisory listing each violation, then the
        refresh command, ending with a "continuing anyway" caveat.
    """
    lines = [
        (
            "this editable dcode install is running against dependencies "
            "older than the floors declared in the checkout's pyproject.toml:"
        )
    ]
    lines.extend(f"  - {v.describe()}" for v in violations)
    refresh = refresh_command()
    lines.append(
        "Refresh the active environment:\n"
        f"  {refresh}\n"
        "Continuing anyway; behavior may be broken."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dismissal ("mute") store — per-checkout fingerprint of a muted mismatch.
# Mirrors the hooks trust store's locking/atomic-write pattern, but the file
# records a *specific mismatch* rather than a durable grant: unlike hook
# trust, muting never disables the check, it only silences the exact mismatch
# the user already saw.
# ---------------------------------------------------------------------------

_DISMISSAL_STORE_LOCK_TIMEOUT_SECONDS = 5.0
_DISMISSAL_STORE_THREAD_LOCK = threading.Lock()


def _default_dismissal_path() -> Path:
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "dep_floor_dismissed.json"


def _checkout_key() -> str:
    """Return the canonical per-checkout key: the resolved editable source root."""
    return str(_checkout_root())


@contextmanager
def _dismissal_store_lock(path: Path) -> Iterator[None]:
    """Serialize read-merge-write updates to the dismissal store.

    Args:
        path: Path to the dismissal JSON file.

    Yields:
        Control while the caller exclusively holds the mutation lock.
    """
    from filelock import FileLock

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        with suppress(OSError):
            path.parent.chmod(0o700)
    file_lock = FileLock(
        str(path.with_name(f"{path.name}.lock")),
        timeout=_DISMISSAL_STORE_LOCK_TIMEOUT_SECONDS,
        thread_local=False,
    )
    with _DISMISSAL_STORE_THREAD_LOCK, file_lock:
        yield


def _read_dismissed_fingerprints(path: Path) -> dict[str, str]:
    """Read the dismissal store, tolerating corruption and absence.

    Args:
        path: Path to the dismissal JSON file.

    Returns:
        Map of checkout key to muted mismatch fingerprint; empty on any
        read/parse failure (a lost dismissal simply re-arms the prompt).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    dismissed = data.get("dismissed")
    if not isinstance(dismissed, dict):
        return {}
    return {k: v for k, v in dismissed.items() if isinstance(v, str)}


def _write_dismissed_fingerprints(path: Path, dismissed: dict[str, str]) -> None:
    """Atomically replace the dismissal store.

    Args:
        path: Path to the dismissal JSON file.
        dismissed: Full map of checkout key to muted mismatch fingerprint.
    """
    payload = json.dumps({"dismissed": dismissed}, indent=2, sort_keys=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f"{path.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        # `fdopen` takes ownership of `fd`, so it is only closed by hand when
        # the wrap itself failed; otherwise the `with` block closes it.
        try:
            handle = os.fdopen(fd, "w", encoding="utf-8")
        except BaseException:
            with suppress(OSError):
                os.close(fd)
            raise
        with handle:
            handle.write(payload)
        if os.name != "nt":
            with suppress(OSError):
                tmp.chmod(0o600)
        tmp.replace(path)
    except BaseException:
        with suppress(OSError):
            tmp.unlink()
        raise


def is_dep_floor_mismatch_muted(fingerprint: str) -> bool:
    """Return whether this checkout's muted fingerprint matches *fingerprint*.

    Reads without taking the store lock. Writes land via an atomic
    `Path.replace`, so a concurrent mute is either wholly visible or wholly
    absent — never torn. Locking here would instead put a `filelock` import
    and a 5-second timeout on the startup path of every editable launch, and
    a second `dcode` mid-write would stall this one for no benefit.

    Args:
        fingerprint: Fingerprint of the currently detected mismatch.

    Returns:
        `True` when the user muted this exact mismatch for this checkout.
        Any store read failure returns `False`, re-arming the prompt.
    """
    try:
        dismissed = _read_dismissed_fingerprints(_default_dismissal_path())
        return dismissed.get(_checkout_key()) == fingerprint
    except Exception:
        logger.debug("Could not read dependency floor dismissal store", exc_info=True)
        return False


def mute_dep_floor_mismatch(fingerprint: str) -> bool:
    """Persist *fingerprint* as this checkout's muted mismatch.

    Args:
        fingerprint: Fingerprint of the mismatch the user chose to mute.

    Returns:
        `True` when the dismissal was persisted; `False` on any write
        failure (the caller then treats the choice as session-only).
    """
    path = _default_dismissal_path()
    try:
        with _dismissal_store_lock(path):
            dismissed = _read_dismissed_fingerprints(path)
            dismissed[_checkout_key()] = fingerprint
            _write_dismissed_fingerprints(path, dismissed)
    except Exception:
        logger.debug("Could not persist dependency floor dismissal", exc_info=True)
        return False
    return True


def warn_if_editable_deps_stale() -> None:
    """Print the stale-dependency warning to stderr for non-TUI launches.

    Skips silently when there are no violations or the exact mismatch was
    muted for this checkout. This is strictly best-effort: any unexpected
    failure degrades to a debug log and never raises. The warning goes to
    stderr so it never contaminates the stdout these launches produce for a
    caller to consume (a piped `-n` answer, a subcommand's output).
    """
    try:
        violations = _collect_violations()
        if not violations:
            return
        if is_dep_floor_mismatch_muted(_mismatch_fingerprint(violations)):
            return
        from rich.console import Console
        from rich.markup import escape

        Console(stderr=True).print(
            f"[bold yellow]Warning:[/bold yellow] "
            f"{escape(format_dep_floor_warning(violations))}",
            highlight=False,
        )
    except Exception:  # strictly best-effort: a check failure must never break startup
        logger.debug("Dependency floor check failed", exc_info=True)


def prompt_if_editable_deps_stale() -> _TrustPromptOutcome | None:
    """Block on a continue/mute/abort prompt when an interactive launch is stale.

    Prompts only when violations exist and the exact mismatch was not muted
    for this checkout. The prompt itself is implemented in `main` next to the
    other pre-TUI trust prompts; this function handles detection, muting, and
    the confirmation prints.

    Returns:
        `None` to continue the launch (no violation, muted, or the user
        chose to continue), `INTERRUPTED` on Ctrl+C (caller exits 130), or
        `CANCELLED` on Esc/abort (caller exits 0).
    """
    try:
        return _prompt_if_editable_deps_stale()
    except Exception:  # strictly best-effort: check failures must not break startup
        logger.debug("Interactive dependency floor check failed", exc_info=True)
        return None


def _prompt_if_editable_deps_stale() -> _TrustPromptOutcome | None:
    """Run the interactive stale-dependency check after its fail-open boundary.

    Returns:
        The abort outcome, or `None` when launch should continue.
    """
    from rich.console import Console

    from deepagents_code.main import (
        _TrustAction,
        _TrustPromptOutcome,
        prompt_for_dep_floor_mismatch,
    )

    violations = _collect_violations()
    if not violations:
        return None
    fingerprint = _mismatch_fingerprint(violations)
    if is_dep_floor_mismatch_muted(fingerprint):
        return None

    console = Console(stderr=True)
    action = prompt_for_dep_floor_mismatch(console, violations)
    if action in {_TrustPromptOutcome.INTERRUPTED, _TrustPromptOutcome.CANCELLED}:
        return action
    if action is _TrustAction.REMEMBER:
        if mute_dep_floor_mismatch(fingerprint):
            console.print(
                "[dim]Muted until the dependency mismatch changes.[/dim]",
                highlight=False,
            )
        else:
            console.print(
                "[yellow]The dismissal could not be saved; you will be asked "
                "again next launch.[/yellow]",
                highlight=False,
            )
    elif action is _TrustAction.ALLOW_ONCE:
        console.print(
            "[dim]Continuing this session; you will be asked again next launch.[/dim]",
            highlight=False,
        )
    else:
        # `abort_on_deny` should have collapsed a refusal into `CANCELLED`
        # already, so `DENY` (or any future outcome) only lands here if that
        # mapping regresses. Refuse rather than continuing: matching "continue"
        # by default is what previously turned "Abort launch" into a launch.
        logger.debug("Unexpected dependency floor prompt outcome %r; aborting", action)
        return _TrustPromptOutcome.CANCELLED
    return None
