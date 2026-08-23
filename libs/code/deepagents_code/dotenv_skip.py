"""Records of project directories whose `.env` should be skipped.

Two independent skip sources live here: a persistent store on disk ("never load
in this project") and a process-local set ("not this session"). The session set
is relayed to the server subprocess through
`_env_vars.SERVER_DOTENV_SESSION_SKIPS`, because the server reloads settings in
its own interpreter and would otherwise load a file the user just refused.

Separate from the hooks trust store (`hooks/trust.py`): choosing not to load a
project's `.env` is an independent decision from trusting its hooks, so the two
stores do not share state. The shape and concurrency model (versioned JSON,
atomic write, process + cross-process locking) mirror that module.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, ValidationError

from deepagents_code._env_vars import SERVER_DOTENV_SESSION_SKIPS

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

_STORE_VERSION: Literal[1] = 1
_STORE_LOCK_TIMEOUT_SECONDS = 5.0
_STORE_THREAD_LOCK = threading.Lock()
"""Process-local guard for skip-store mutations (see `hooks.trust`)."""

_SESSION_SKIP_LOCK = threading.Lock()
"""Guard for project skips that last only for the current process."""

_SESSION_SKIPPED_PROJECTS: set[str] = set()
"""Canonical `.env` parent directories skipped for the current process."""

_SESSION_SKIPS_SEEDED = False
"""Whether the relayed skips from the launch environment were read yet."""

_STDERR_WARN_LOCK = threading.Lock()
_STDERR_WARNED: set[str] = set()
"""Warnings already written to stderr, so a reload cannot repeat them."""


def _tui_owns_the_terminal() -> bool:
    """Return whether a Textual app is currently mounted.

    Read through `sys.modules` so a pre-TUI launch — the case the stderr
    pairing exists for — does not pay a `textual` import to find out.
    """
    app_module = sys.modules.get("textual.app")
    active_app = getattr(app_module, "active_app", None)
    if active_app is None:
        return False
    try:
        return active_app.get(None) is not None
    except LookupError:  # pragma: no cover - defensive
        return False


def _warn_dropped_decision(message: str) -> None:
    """Report a dropped `.env` decision where the user can see it.

    Every caller means a decision the user made is not being applied, so a bare
    `logger.warning` will not do: `deepagents_code.__init__` attaches an
    in-memory handler to the package logger, which suppresses the stderr
    `lastResort` fallback, and these paths run before the TUI (and its Debug
    Console) exists. Pairing the log with an explicit stderr line matches
    `_debug`.

    The store is also read on every reload and cwd-switch preview, when Textual
    does own the terminal. Raw stderr there would paint over the interface, so
    the stderr half is skipped once the TUI is up — the Debug Console shows the
    logged copy instead — and never repeats a message it already printed.

    Args:
        message: The warning text, without the `Warning: ` prefix.
    """
    logger.warning("%s", message)
    if _tui_owns_the_terminal():
        return
    with _STDERR_WARN_LOCK:
        if message in _STDERR_WARNED:
            return
        _STDERR_WARNED.add(message)
    print(f"Warning: {message}", file=sys.stderr)  # noqa: T201


class DotenvSkipEntry(BaseModel):
    """Persisted record for one canonical project root whose `.env` is skipped."""

    model_config = ConfigDict(extra="ignore")

    skipped_at: str
    """UTC ISO-8601 timestamp when the skip was recorded."""


class DotenvAllowEntry(BaseModel):
    """Persisted record for one canonical project root whose `.env` is trusted."""

    model_config = ConfigDict(extra="ignore")

    allowed_at: str
    """UTC ISO-8601 timestamp when the allow was recorded."""


class DotenvSkipStore(BaseModel):
    """Versioned on-disk store of remembered project `.env` decisions."""

    model_config = ConfigDict(extra="ignore")

    version: Literal[1] = _STORE_VERSION
    """Schema version; unsupported versions are ignored on read."""

    projects: dict[str, DotenvSkipEntry] = {}
    """Map of canonical project roots to skip entries."""

    allowed: dict[str, DotenvAllowEntry] = {}
    """Map of canonical project roots whose `.env` should load without asking.

    Read only by the prompt, never by the loader: an allow suppresses the
    question, it does not grant a load that `startup.read_project_dotenv` or a
    skip already refused. That keeps the store one-directional — it can silence
    or skip, never force.
    """


def _default_store_path() -> Path:
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "dotenv_skip.json"


def _project_key(project_root: Path | str) -> str:
    return str(Path(project_root).expanduser().resolve())


def _warn_relayed_skips_unusable(reason: str) -> None:
    """Report relayed session skips that could not be read, visibly.

    Every call means the user answered the advisory prompt in the client and
    this process will load the `.env` anyway, so the warning uses the same
    stderr + logger pairing as `_warn_store_unusable`.

    Args:
        reason: Short description of what was wrong with the relayed value.
    """
    _warn_dropped_decision(
        f"could not read {SERVER_DOTENV_SESSION_SKIPS}: {reason}. "
        "Session project .env skips are not applied in this process."
    )


def _relayed_session_skips() -> set[str]:
    """Read session skips relayed from the launching process.

    Returns:
        Canonical keys from `SERVER_DOTENV_SESSION_SKIPS`, or an empty set when
        the variable is absent, empty, or unusable.
    """
    raw = os.environ.get(SERVER_DOTENV_SESSION_SKIPS)
    if not raw:
        return set()
    try:
        relayed: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        _warn_relayed_skips_unusable(f"it is not valid JSON ({exc})")
        return set()
    if not isinstance(relayed, list):
        _warn_relayed_skips_unusable("it is not a JSON array of strings")
        return set()
    keys: set[str] = set()
    for key in relayed:
        if not isinstance(key, str):
            _warn_relayed_skips_unusable("it is not a JSON array of strings")
            return set()
        keys.add(key)
    return keys


def _session_skips_locked() -> set[str]:
    """Return the session skip set, seeding it from the environment once.

    The relayed value is read lazily rather than at import so a test (or a
    caller that sets the variable after import) sees the same behavior as a
    freshly launched server. Callers must hold `_SESSION_SKIP_LOCK`.
    """
    global _SESSION_SKIPS_SEEDED  # noqa: PLW0603
    if not _SESSION_SKIPS_SEEDED:
        _SESSION_SKIPS_SEEDED = True
        _SESSION_SKIPPED_PROJECTS.update(_relayed_session_skips())
    return _SESSION_SKIPPED_PROJECTS


def reset_session_skips() -> None:
    """Drop every session skip and re-arm seeding from the environment.

    For tests only: the session set is process-global, so a test that records a
    skip would otherwise leak it into every later test in the same process.
    """
    global _SESSION_SKIPS_SEEDED  # noqa: PLW0603
    with _SESSION_SKIP_LOCK:
        _SESSION_SKIPPED_PROJECTS.clear()
        _SESSION_SKIPS_SEEDED = False
    os.environ.pop(SERVER_DOTENV_SESSION_SKIPS, None)


def skip_project_dotenv_for_session(project_root: Path | str) -> None:
    """Skip one project's `.env` for the rest of the current process.

    Also exports the full session set to `SERVER_DOTENV_SESSION_SKIPS`, which
    is how the decision reaches processes this one starts or becomes: the server
    subprocess inherits `os.environ` (`client.launch.server._build_server_env`)
    and so does the startup auto-update's `os.execv`. Without that the server
    would reload settings in its own interpreter and load the very file the
    user just refused.

    Args:
        project_root: Directory that owns the discovered project `.env`.
    """
    key = _project_key(project_root)
    with _SESSION_SKIP_LOCK:
        keys = _session_skips_locked()
        keys.add(key)
        exported = json.dumps(sorted(keys))
    os.environ[SERVER_DOTENV_SESSION_SKIPS] = exported


def is_project_dotenv_skipped_for_session(project_root: Path | str) -> bool:
    """Return whether this process skips the project's `.env`.

    Args:
        project_root: Directory that owns the discovered project `.env`.

    Returns:
        `True` when the canonical project key was skipped in this process,
        either by a prompt answered here or by a skip relayed from the client.
    """
    key = _project_key(project_root)
    with _SESSION_SKIP_LOCK:
        return key in _session_skips_locked()


def skip_key_for_start_path(start_path: Path | None) -> str | None:
    """Resolve the skip-store key that governs the `.env` found from a directory.

    The key is the parent of the discovered `.env` file — the directory that
    actually owns the file — not the invocation directory. A non-Git project
    has no `ProjectContext.project_root`, so keying on `user_cwd` would miss
    the ancestor `.env` that `_find_dotenv_from_start_path` walks up to; keying
    on the file's parent makes "never load" follow the file whether the launch
    is from its own directory or a subdirectory.

    Args:
        start_path: Directory to start `.env` discovery from; cwd when `None`.

    Returns:
        The canonical key for the discovered `.env`, or `None` when no project
        `.env` is present (nothing to skip).
    """
    from deepagents_code.config import _find_dotenv_from_start_path

    try:
        dotenv_path = _find_dotenv_from_start_path(
            Path(start_path) if start_path is not None else Path.cwd()
        )
    except OSError:
        logger.debug("Could not locate a project .env for the skip key")
        return None
    if dotenv_path is None:
        return None
    return _project_key(dotenv_path.parent)


def _warn_store_unusable(path: Path, reason: str) -> None:
    """Report a skip store that could not be used, visibly.

    Every call means a remembered "never load this project's `.env`" decision
    was dropped and the file will load, so the user has to be able to see it
    (see `_warn_dropped_decision`).

    Args:
        path: Store path that could not be used.
        reason: Short description of what was wrong with it.
    """
    _warn_dropped_decision(
        f"could not read {path}: {reason}. "
        f"Remembered project .env skips are not applied."
    )


def _warn_entry_unusable(path: Path, reason: str) -> None:
    """Report one ignored entry without implying the whole store was dropped."""
    _warn_dropped_decision(
        f"could not use an entry from {path}: {reason}. Only this entry is "
        "ignored; other valid remembered project .env decisions still apply."
    )


def _store_lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


@contextmanager
def _store_lock(path: Path) -> Iterator[None]:
    """Serialize read-merge-write updates to the dotenv skip store.

    Combines a single process-local threading lock with a cross-process
    `FileLock` on a sibling `.lock` file so concurrent `dcode` processes cannot
    drop each other's entries. Callers handle `filelock.Timeout` and `OSError`.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.parent.chmod(0o700)
    file_lock = FileLock(
        str(_store_lock_path(path)),
        timeout=_STORE_LOCK_TIMEOUT_SECONDS,
        thread_local=False,
    )
    with _STORE_THREAD_LOCK, file_lock:
        yield


class _InvalidEntryMapError(Exception):
    """One of the store's entry maps is present but not an object.

    A dedicated type rather than `TypeError` so the caller's handler cannot
    also swallow an unrelated `TypeError` from validation and then misreport it
    as a malformed field.
    """


def _parse_entries[EntryT: BaseModel](
    raw_entries: object,
    entry_model: type[EntryT],
    *,
    field: str,
    path: Path,
) -> dict[str, EntryT]:
    """Parse one of the store's entry maps, dropping structurally invalid rows.

    Args:
        raw_entries: The map, straight from JSON.
        entry_model: Model each value must validate against.
        field: Field name, used to identify the map in messages.
        path: Store path, used only to identify the file in warnings.

    Returns:
        Validated map, with unusable entries dropped. Empty when `raw_entries`
        is `None`.

    Raises:
        _InvalidEntryMapError: When `raw_entries` is present but not a mapping.
    """
    if raw_entries is None:
        return {}
    if not isinstance(raw_entries, dict):
        msg = f"its {field} field is not an object"
        raise _InvalidEntryMapError(msg)

    entries: dict[str, EntryT] = {}
    for key, value in raw_entries.items():
        if not isinstance(key, str):
            _warn_entry_unusable(path, f"a {field} key is not a string: {key!r}")
            continue
        try:
            entries[key] = entry_model.model_validate(value)
        except ValidationError as exc:
            _warn_entry_unusable(path, f"the {field} entry for {key} is invalid: {exc}")
    return entries


def _load_store(path: Path, *, strict: bool = False) -> DotenvSkipStore:
    """Load and validate the dotenv skip store.

    Args:
        path: Store path.
        strict: When `True`, raise on unreadable or structurally invalid stores
            so writers refuse to overwrite them.

    Returns:
        Validated store. Missing files yield an empty store; unreadable or
        unsupported stores yield an empty store when not `strict`.

    Raises:
        OSError: When `strict` and the file exists but cannot be read; a
            missing file always yields an empty store.
        TypeError: When `strict` and the top-level shape is not a mapping.
        _InvalidEntryMapError: When `strict` and `projects` or `allowed` is present
            but not a mapping. Individual invalid *entries* are dropped with a
            warning in both modes, so a write silently discards them.
        ValueError: When `strict` and the version is unsupported.
        json.JSONDecodeError: When `strict` and the file is not valid JSON.
        UnicodeDecodeError: When `strict` and the file is not UTF-8 text.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DotenvSkipStore()
    except (OSError, UnicodeDecodeError) as exc:
        if strict:
            raise
        _warn_store_unusable(path, str(exc))
        return DotenvSkipStore()

    try:
        data: object = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        if strict:
            raise
        _warn_store_unusable(path, f"it is not valid JSON ({exc})")
        return DotenvSkipStore()

    if not isinstance(data, dict):
        msg = f"dotenv skip store must be a JSON object: {path}"
        if strict:
            raise TypeError(msg)
        _warn_store_unusable(path, "it is not a JSON object")
        return DotenvSkipStore()

    version = data.get("version")
    if version != _STORE_VERSION:
        msg = f"Unsupported dotenv skip store version: {version!r}"
        if strict:
            raise ValueError(msg)
        _warn_store_unusable(path, f"its version {version!r} is not supported")
        return DotenvSkipStore()

    try:
        projects = _parse_entries(
            data.get("projects"), DotenvSkipEntry, field="projects", path=path
        )
        allowed = _parse_entries(
            data.get("allowed"), DotenvAllowEntry, field="allowed", path=path
        )
    except _InvalidEntryMapError as exc:
        if strict:
            raise
        _warn_store_unusable(path, str(exc))
        return DotenvSkipStore()

    return DotenvSkipStore(version=_STORE_VERSION, projects=projects, allowed=allowed)


def _write_store(path: Path, store: DotenvSkipStore) -> None:
    """Atomically write the skip store with restrictive permissions."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.parent.chmod(0o700)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(store.model_dump(mode="json"), handle, sort_keys=True)
            handle.write("\n")
        if os.name != "nt":
            tmp_path.chmod(0o600)
        tmp_path.replace(path)
        if os.name != "nt":
            path.chmod(0o600)
    except BaseException:
        with suppress(OSError):
            tmp_path.unlink()
        raise


def is_project_dotenv_skipped(
    project_root: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Return whether the project `.env` is skipped for a canonical key.

    Args:
        project_root: The skip key to inspect — the parent directory of the
            discovered `.env` (see `skip_key_for_start_path`).
        store_path: Alternate store path for tests.

    Returns:
        `True` when the key is in the skip store.
    """
    path = store_path or _default_store_path()
    store = _load_store(path)
    return _project_key(project_root) in store.projects


def skip_project_dotenv(
    project_root: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Persist skipping the project `.env` for a canonical project root.

    Args:
        project_root: Project root whose `.env` should be skipped.
        store_path: Alternate store path for tests.

    Returns:
        `True` when the decision was saved.

    Note:
        Failures (unreadable store, lock timeout, I/O errors) return `False`
        without mutating the on-disk store; callers treat `False` as a real
        persistence failure, not an implicit session decision.
    """
    path = store_path or _default_store_path()
    try:
        with _store_lock(path):
            try:
                store = _load_store(path, strict=True)
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
                _InvalidEntryMapError,
            ):
                logger.exception(
                    "Refusing to overwrite unreadable dotenv skip store %s", path
                )
                return False

            key = _project_key(project_root)
            projects = dict(store.projects)
            projects[key] = DotenvSkipEntry(skipped_at=datetime.now(UTC).isoformat())
            # A skip and an allow are contradictory answers to the same
            # question, so the newer one replaces the older.
            allowed = {k: v for k, v in store.allowed.items() if k != key}
            _write_store(
                path,
                DotenvSkipStore(
                    version=_STORE_VERSION, projects=projects, allowed=allowed
                ),
            )
    except Timeout:
        logger.exception("Timed out waiting to persist dotenv skip store %s", path)
        return False
    except OSError:
        logger.exception("Could not save dotenv skip store %s", path)
        return False
    return True


def is_project_dotenv_allowed(
    project_root: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Return whether the project `.env` is trusted for a canonical key.

    Read by the advisory prompt only. It answers "stop asking about this
    `.env`", not "load it": `startup.read_project_dotenv` and both skip sources
    are checked first and still win.

    Args:
        project_root: The key to inspect — the parent directory of the
            discovered `.env` (see `skip_key_for_start_path`).
        store_path: Alternate store path for tests.

    Returns:
        `True` when the key is in the allow map.
    """
    path = store_path or _default_store_path()
    store = _load_store(path)
    return _project_key(project_root) in store.allowed


def allow_project_dotenv(
    project_root: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Persist loading the project `.env` without asking again.

    Args:
        project_root: Project root whose `.env` should load without a prompt.
        store_path: Alternate store path for tests.

    Returns:
        `True` when the decision was saved. Failures return `False` without
        mutating the on-disk store, exactly as `skip_project_dotenv` does; the
        caller reports that the answer was not remembered rather than implying
        it was.
    """
    path = store_path or _default_store_path()
    try:
        with _store_lock(path):
            try:
                store = _load_store(path, strict=True)
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
                _InvalidEntryMapError,
            ):
                logger.exception(
                    "Refusing to overwrite unreadable dotenv skip store %s", path
                )
                return False

            key = _project_key(project_root)
            allowed = dict(store.allowed)
            allowed[key] = DotenvAllowEntry(allowed_at=datetime.now(UTC).isoformat())
            # Contradictory answers to the same question; the newer one wins.
            projects = {k: v for k, v in store.projects.items() if k != key}
            _write_store(
                path,
                DotenvSkipStore(
                    version=_STORE_VERSION, projects=projects, allowed=allowed
                ),
            )
    except Timeout:
        logger.exception("Timed out waiting to persist dotenv skip store %s", path)
        return False
    except OSError:
        logger.exception("Could not save dotenv skip store %s", path)
        return False
    return True
