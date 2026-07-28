"""Persistent workspace trust for project-scoped hooks."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, ValidationError

from deepagents_code.project_utils import ProjectContext

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

_STORE_VERSION: Literal[1] = 1
_TRUST_STORE_LOCK_TIMEOUT_SECONDS = 5.0
_TRUST_STORE_THREAD_LOCK = threading.Lock()
"""Process-local guard for trust-store mutations.

A single lock is sufficient: one store backs the process, and the alternate
paths tests pass only ever serialize against each other harmlessly. Contention
across distinct stores is not worth per-path lock bookkeeping.
"""


class HooksTrustEntry(BaseModel):
    """Persisted trust record for one canonical workspace root."""

    model_config = ConfigDict(extra="ignore")

    trusted_at: str
    """UTC ISO-8601 timestamp when the workspace was trusted."""


class HooksTrustStore(BaseModel):
    """Versioned on-disk trust store for project-scoped hooks."""

    model_config = ConfigDict(extra="ignore")

    version: Literal[1] = _STORE_VERSION
    """Schema version; unsupported versions are ignored on read."""

    projects: dict[str, HooksTrustEntry] = {}
    """Map of canonical workspace roots to trust entries."""


def _default_store_path() -> Path:
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "hooks_trust.json"


def _project_key(project_root: Path | str) -> str:
    return str(Path(project_root).expanduser().resolve())


def _trust_store_lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


@contextmanager
def _trust_store_lock(path: Path) -> Iterator[None]:
    """Serialize read-merge-write updates to the hooks trust store.

    Combines a single process-local threading lock with a cross-process
    `FileLock` on a sibling `.lock` file so concurrent `dcode` processes cannot
    drop each other's workspace entries.

    Args:
        path: Path to `hooks_trust.json`.

    Yields:
        Control while the caller exclusively holds the mutation lock.

    Callers should handle `filelock.Timeout` (lock wait expired) and `OSError`
    (lock directory creation failure).
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.parent.chmod(0o700)
    file_lock = FileLock(
        str(_trust_store_lock_path(path)),
        timeout=_TRUST_STORE_LOCK_TIMEOUT_SECONDS,
        thread_local=False,
    )
    with _TRUST_STORE_THREAD_LOCK, file_lock:
        yield


def _parse_projects(
    raw_projects: object,
    *,
    path: Path,
) -> dict[str, HooksTrustEntry]:
    """Parse trust entries, skipping structurally invalid ones.

    Args:
        raw_projects: Raw `projects` value from JSON.
        path: Store path used in warning messages.

    Returns:
        Validated project map. Empty when `raw_projects` is missing or not a
        mapping.

    Raises:
        TypeError: When `raw_projects` is present but not a mapping (strict
            callers refuse to overwrite such stores).
    """
    if raw_projects is None:
        return {}
    if not isinstance(raw_projects, dict):
        msg = f"hooks trust store projects must be an object: {path}"
        raise TypeError(msg)

    projects: dict[str, HooksTrustEntry] = {}
    for key, value in raw_projects.items():
        if not isinstance(key, str):
            logger.warning(
                "Skipping non-string hooks trust project key in %s: %r",
                path,
                key,
            )
            continue
        try:
            projects[key] = HooksTrustEntry.model_validate(value)
        except ValidationError as exc:
            logger.warning(
                "Skipping invalid hooks trust entry for %s in %s: %s",
                key,
                path,
                exc,
            )
    return projects


def _load_store(path: Path, *, strict: bool = False) -> HooksTrustStore:
    """Load and validate the hooks trust store.

    Args:
        path: Trust store path.
        strict: When `True`, raise on unreadable or structurally invalid stores
            so writers refuse to overwrite them.

    Returns:
        Validated store. Missing files yield an empty store. Unsupported versions
        and unreadable files yield an empty store when not `strict`.

    Raises:
        OSError: When `strict` and the file cannot be read.
        TypeError: When `strict` and `projects` is not a mapping.
        ValueError: When `strict` and the version or top-level shape is invalid.
        json.JSONDecodeError: When `strict` and the file is not valid JSON.
        UnicodeDecodeError: When `strict` and the file is not UTF-8 text.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return HooksTrustStore()
    except (OSError, UnicodeDecodeError) as exc:
        # Decoding happens during the read, so non-UTF-8 stores surface here
        # rather than at `json.loads` below.
        if strict:
            raise
        logger.warning("Could not read hooks trust store %s: %s", path, exc)
        return HooksTrustStore()

    try:
        data: object = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        if strict:
            raise
        logger.warning("Could not parse hooks trust store %s: %s", path, exc)
        return HooksTrustStore()

    if not isinstance(data, dict):
        msg = f"hooks trust store must be a JSON object: {path}"
        if strict:
            raise TypeError(msg)
        logger.warning(msg)
        return HooksTrustStore()

    version = data.get("version")
    if version != _STORE_VERSION:
        msg = f"Unsupported hooks trust store version: {version!r}"
        if strict:
            raise ValueError(msg)
        logger.warning(
            "Ignoring hooks trust store with unsupported version %r", version
        )
        return HooksTrustStore()

    try:
        projects = _parse_projects(data.get("projects"), path=path)
    except TypeError:
        if strict:
            raise
        logger.warning(
            "Ignoring hooks trust store with invalid projects field at %s",
            path,
        )
        return HooksTrustStore()

    # Tolerate unknown top-level fields via HooksTrustStore.extra="ignore".
    return HooksTrustStore(version=_STORE_VERSION, projects=projects)


def _write_store(path: Path, store: HooksTrustStore) -> None:
    """Atomically write the trust store with restrictive permissions.

    Args:
        path: Destination path.
        store: Validated store payload.
    """
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
            json.dump(
                store.model_dump(mode="json"),
                handle,
                sort_keys=True,
            )
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


def is_project_hooks_trusted(
    project_root: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Return whether project hooks are trusted for a canonical workspace root.

    Args:
        project_root: Workspace root to inspect.
        store_path: Alternate trust store path for tests.

    Returns:
        `True` when the canonical workspace root is trusted.
    """
    path = store_path or _default_store_path()
    store = _load_store(path)
    return _project_key(project_root) in store.projects


def trust_project_hooks(
    project_root: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Persist project-hook trust for a workspace root.

    Args:
        project_root: Workspace root to trust.
        store_path: Alternate trust store path for tests.

    Returns:
        `True` when trust was saved.

    Note:
        Failures (unreadable store, lock timeout, I/O errors) return `False`
        without mutating the on-disk store. Callers must treat `False` as a
        real persistence failure, not as an implicit session grant.
    """
    path = store_path or _default_store_path()
    try:
        with _trust_store_lock(path):
            try:
                store = _load_store(path, strict=True)
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                logger.exception(
                    "Refusing to overwrite unreadable hooks trust store %s",
                    path,
                )
                return False

            projects = dict(store.projects)
            projects[_project_key(project_root)] = HooksTrustEntry(
                trusted_at=datetime.now(UTC).isoformat()
            )
            _write_store(
                path,
                HooksTrustStore(version=_STORE_VERSION, projects=projects),
            )
    except Timeout:
        logger.exception("Timed out waiting to persist hooks trust store %s", path)
        return False
    except OSError:
        logger.exception("Could not save hooks trust store %s", path)
        return False
    return True


def project_root_for(cwd: Path | str) -> Path:
    """Resolve the workspace root that governs hook trust for a directory.

    Args:
        cwd: Session working directory.

    Returns:
        The enclosing project root, or the directory itself when it is not
        inside a project.
    """
    context = ProjectContext.from_user_cwd(Path(cwd))
    return context.project_root or context.user_cwd


@dataclass(frozen=True, slots=True)
class WorkspaceTrust:
    """Decides whether project-scoped hooks may run in a given directory.

    Trust is a property of the workspace, not of the session, so it must be
    re-resolved every time the working directory moves. A session that starts in
    a trusted project and later moves into an untrusted one must not carry the
    original grant forward.

    Callers hand this policy to `HooksManager`, which resolves it on load and on
    every reload; nothing upstream needs to hold or reinterpret the decision.
    """

    session_grants: frozenset[str] = frozenset()
    """Canonical workspace roots trusted for this session only.

    Populated from an explicit CLI grant or an in-session `allow once` choice,
    neither of which is persisted to the trust store.
    """

    consult_store: bool = True
    """Whether persisted trust may satisfy the policy.

    Headless runs set this to `False`: executing repository hooks there requires
    an explicit opt-in, so a workspace remembered during an interactive session
    must not silently qualify a later `dcode -n` invocation.
    """

    store_path: Path | None = None
    """Alternate trust store path for tests."""

    @classmethod
    def none(cls) -> WorkspaceTrust:
        """Return a policy that trusts no workspace.

        Returns:
            A policy that consults only the persisted trust store.
        """
        return cls()

    @classmethod
    def for_session(
        cls,
        cwd: Path | str,
        *,
        granted: bool,
        store_path: Path | None = None,
    ) -> WorkspaceTrust:
        """Build a policy from a launch-time trust decision.

        Args:
            cwd: Directory the trust decision was made for.
            granted: Whether the user trusted project hooks for this session
                without persisting that choice.
            store_path: Alternate trust store path for tests.

        Returns:
            A policy that grants `cwd`'s workspace root for this session and
            defers to the persisted store everywhere else.
        """
        return cls(
            session_grants=cls._grants_for(cwd) if granted else frozenset(),
            store_path=store_path,
        )

    @classmethod
    def explicit_only(
        cls,
        cwd: Path | str,
        *,
        granted: bool,
        store_path: Path | None = None,
    ) -> WorkspaceTrust:
        """Build a policy that ignores persisted trust entirely.

        For contexts where running repository hooks must be an explicit opt-in
        rather than something a previous interactive session can enable —
        notably headless runs, where the operator may never have seen the
        interactive trust prompt.

        Args:
            cwd: Directory the trust decision was made for.
            granted: Whether project hooks were explicitly opted into.
            store_path: Alternate trust store path for tests.

        Returns:
            A policy that allows `cwd`'s workspace root only when `granted`, and
            allows nothing otherwise.
        """
        return cls(
            session_grants=cls._grants_for(cwd) if granted else frozenset(),
            consult_store=False,
            store_path=store_path,
        )

    @staticmethod
    def _grants_for(cwd: Path | str) -> frozenset[str]:
        try:
            return frozenset({_project_key(project_root_for(cwd))})
        except (OSError, ValueError):
            logger.warning(
                "Could not resolve workspace root for session hook trust",
                exc_info=True,
            )
            return frozenset()

    def allows(self, cwd: Path | str) -> bool:
        """Return whether project hooks may run for a working directory.

        Args:
            cwd: Directory to resolve trust for.

        Returns:
            `True` when the enclosing workspace root was granted for this
            session, or is recorded in the trust store and `consult_store` is
            set. Unresolvable directories fail closed.
        """
        try:
            root = project_root_for(cwd)
        except (OSError, ValueError):
            # The raised exception carries the offending path; don't restate it.
            logger.warning(
                "Could not resolve workspace root; treating project hooks as untrusted",
                exc_info=True,
            )
            return False
        if _project_key(root) in self.session_grants:
            return True
        if not self.consult_store:
            return False
        return is_project_hooks_trusted(root, store_path=self.store_path)
