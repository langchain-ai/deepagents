"""Trust store for skill directories that resolve outside trusted roots.

`load_skill_content` refuses to read a `SKILL.md` whose resolved path falls
outside every trusted skill root — this stops a symlink inside a skill
directory from reading arbitrary files. The static escape hatch is the
`DEEPAGENTS_CODE_EXTRA_SKILLS_DIRS` env var / `[skills].extra_allowed_dirs`
config allowlist.

This module adds an in-the-moment, persistent approval path (mirroring
`deepagents_code.mcp_trust`): when a skill resolves outside the trusted roots,
the user is asked once to allow the resolved target directory, and the decision
is remembered. Trust is keyed by the approved target directory — the canonical
path resolved and shown to the user at approval time, stored as-is and never
re-resolved.

Two distinct post-approval swaps are caught by two distinct layers, so neither
grants access the user never approved:

* Re-pointing the *discovery* symlink (the `SKILL.md` path) at a new target is
  caught by containment enforcement in `load_skill_content`: the new target is
  not on the allowlist, so the read is refused and the user is re-prompted. The
  stored trust entry — the original resolved target — is untouched.
* Replacing the *stored* directory itself (or one of its parents) with a symlink
  is caught by the `resolve()`-to-self re-verification in
  `load_trusted_skill_dirs`, which drops the stale entry rather than following
  the injected symlink to a directory the user never approved.

Trust entries are app-managed bookkeeping (a set of approved directories), not
user-facing configuration, so they live alongside the other state files under
`~/.deepagents/.state/skill_trust.json` rather than in the hand-editable
`config.toml`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

_STORAGE_VERSION = 1
"""Schema version stamped into `skill_trust.json`; bump on incompatible changes."""


class _TrustEntry(TypedDict):
    """One trusted-directory record in the store's `dirs` map."""

    trusted_at: str
    """ISO-8601 UTC timestamp of when the directory was approved."""


class _TrustStore(TypedDict):
    """On-disk shape of `skill_trust.json`."""

    version: int
    dirs: dict[str, _TrustEntry]


class RevokeResult(Enum):
    """Outcome of a `revoke_skill_dir_trust` call.

    Distinguishing `NOT_FOUND` from `REMOVED` lets the CLI print an honest
    message instead of a false success when the target was never trusted (a
    plain bool collapsed the two).
    """

    REMOVED = "removed"
    """An entry existed and was removed from the store."""
    NOT_FOUND = "not_found"
    """No matching entry existed; the store was left unchanged."""
    ERROR = "error"
    """The store could not be read or the removal could not be persisted."""


def _default_store_path() -> Path:
    """Return `~/.deepagents/.state/skill_trust.json`.

    Resolved at call time (not import time) so tests can redirect storage by
    monkeypatching `deepagents_code.model_config.DEFAULT_STATE_DIR` — the same
    pattern `deepagents_code.mcp_trust._default_store_path` uses.
    """
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "skill_trust.json"


def _normalize(target_dir: Path | str) -> str:
    """Return the resolved absolute string form of a directory key."""
    return str(Path(target_dir).expanduser().resolve())


def _approved_key(target_dir: Path | str) -> str:
    """Return the already-approved directory key without resolving again."""
    return str(Path(target_dir).expanduser())


def _load_store(store_path: Path, *, strict: bool = False) -> dict[str, Any]:
    """Read the JSON trust store file.

    Args:
        store_path: Path to the trust store file.
        strict: When `True`, a store that exists but cannot be read or parsed
            re-raises instead of degrading to `{}`. Read/modify/write callers
            pass `strict=True` so a transient read error aborts the write
            rather than silently rebuilding the store from an empty dict (which
            would clobber every prior approval). The audit path passes it too so
            it can report an unreadable store instead of claiming nothing is
            trusted. Enforcement callers leave it `False` to stay fail-closed.

    Returns:
        Parsed JSON data, or an empty dict when the file is missing, or (only
        when `strict` is `False`) when it is unreadable or corrupt. A corrupt
        store degrades to "nothing trusted" so a bad file can't crash startup.
        It is *not* self-healed on the next approval: ordinary writes read with
        `strict=True` and refuse rather than clobber a store they can't parse,
        so recovery from a corrupt file requires `skills trust clear` (or
        `clear_trusted_skill_dirs`, the only writer that overwrites blindly).

    Raises:
        OSError: When `strict` and an existing store cannot be read.
        json.JSONDecodeError: When `strict` and an existing store is not valid
            JSON.
        ValueError: When `strict` and the store's top-level value is not a JSON
            object, or its `version` is unrecognized (non-integer, or newer than
            this build understands).
    """
    # A missing store is a normal first-run state, never an error — return
    # empty even under `strict` so callers don't have to special-case it.
    if not store_path.exists():
        return {}
    try:
        data = json.loads(store_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if strict:
            raise
        # A corrupt store silently drops every prior approval and forces a
        # re-prompt, so log at WARNING (not DEBUG) to leave a breadcrumb for
        # the otherwise-unexplained re-prompt.
        logger.warning(
            "Skill trust store %s is corrupt; treating as empty: %s", store_path, exc
        )
        return {}
    except OSError as exc:
        if strict:
            raise
        logger.warning(
            "Could not read skill trust store %s; treating as empty: %s",
            store_path,
            exc,
        )
        return {}
    if not isinstance(data, dict):
        if strict:
            msg = f"Skill trust store {store_path} is not a JSON object"
            raise ValueError(msg)
        logger.warning(
            "Skill trust store %s is not a JSON object; ignoring", store_path
        )
        return {}
    # A store written by a newer build may carry an incompatible schema. Reading
    # its `dirs` regardless could misinterpret entries, so refuse: fail-closed
    # (treat as nothing trusted) for enforcement, and surface the error for the
    # audit path. A present-but-non-integer `version` is unrecognized in the same
    # way (only tampering or a corrupt write produces it, since every writer
    # stamps an int), so it is refused too rather than falling through and
    # trusting `dirs`. A missing `version` stays tolerated: an empty `{}` file
    # has no `dirs` to trust anyway. Together this makes the `_STORAGE_VERSION`
    # "bump on incompatible changes" contract enforceable rather than
    # aspirational.
    version = data.get("version")
    if version is not None and (
        not isinstance(version, int) or version > _STORAGE_VERSION
    ):
        if strict:
            msg = (
                f"Skill trust store {store_path} has an unrecognized schema "
                f"version {version!r} (this build understands <= {_STORAGE_VERSION}); "
                f"refusing to read it"
            )
            raise ValueError(msg)
        logger.warning(
            "Skill trust store %s has an unrecognized schema version %r "
            "(this build understands <= %s); treating as empty",
            store_path,
            version,
            _STORAGE_VERSION,
        )
        return {}
    return data


def _save_store(data: Mapping[str, Any], store_path: Path) -> bool:
    """Atomic write of JSON trust data to `store_path`.

    Uses `tempfile.mkstemp` + `Path.replace` for crash safety.

    Args:
        data: Full store dict to write.
        store_path: Destination path.

    Returns:
        `True` on success, `False` on I/O failure.
    """
    try:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=store_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            Path(tmp_path).replace(store_path)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except (OSError, ValueError):
        logger.exception("Failed to save skill trust store to %s", store_path)
        return False
    return True


def _read_dirs(store_path: Path, *, strict: bool = False) -> dict[str, Any]:
    """Return the `dirs` mapping from the store, or an empty dict.

    Args:
        store_path: Path to the trust store file.
        strict: Propagated to `_load_store`; see its docstring.
    """
    dirs = _load_store(store_path, strict=strict).get("dirs", {})
    return dirs if isinstance(dirs, dict) else {}


def is_skill_dir_trusted(
    target_dir: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Check whether a resolved skill directory has been trusted.

    Warning:
        This resolves `target_dir` and checks raw membership; it does NOT
        re-verify that a stored entry still resolves to itself. It is therefore
        **not** a safe containment-enforcement primitive on its own — a stored
        directory swapped for a symlink after approval would still report
        trusted here. Enforcement builds the containment allowlist from
        `load_trusted_skill_dirs`, which drops post-approval symlink swaps. Use
        this only for informational "is this exact resolved dir on record?"
        checks.

        The lookup resolves `target_dir` (via `_normalize`), but `trust_skill_dir`
        stores the expanduser-only `_approved_key`. In the live flow the two
        coincide because callers approve an already-resolved path, so the keys
        are identical. A caller that trusted a *non-canonical* path would see a
        false negative here (the only failure direction, and the safe one). Pass
        an already-resolved directory to keep the check meaningful.

    Args:
        target_dir: Directory to check; resolved before lookup.
        store_path: Path to the trust store file. Defaults to
            `~/.deepagents/.state/skill_trust.json`.

    Returns:
        `True` if the resolved directory is present in the store.
    """
    if store_path is None:
        store_path = _default_store_path()
    return _normalize(target_dir) in _read_dirs(store_path)


def trust_skill_dir(
    target_dir: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Persist trust for a resolved skill directory.

    Args:
        target_dir: Canonical directory to trust. This is expected to be the
            already-resolved path shown to the user, and is not resolved again
            before storing so a post-approval symlink swap cannot change what
            gets persisted.
        store_path: Path to the trust store file. Defaults to
            `~/.deepagents/.state/skill_trust.json`.

    Returns:
        `True` if the entry was saved successfully.
    """
    if store_path is None:
        store_path = _default_store_path()

    # Read strictly: if an existing store can't be read, abort rather than
    # rebuild it from `{}` and overwrite (which would drop every prior
    # approval). A transient read error should re-prompt next time, not
    # silently erase the store.
    try:
        data = _load_store(store_path, strict=True)
    except (OSError, ValueError):
        logger.exception(
            "Refusing to persist skill trust: could not read existing store %s",
            store_path,
        )
        return False
    # The key is stored expanduser-only and never re-resolved (that is the
    # anti-symlink-swap property). That only holds the invariant "the stored key
    # is the canonical dir the user approved" if the caller already passed a
    # canonical path. If it did not, `load_trusted_skill_dirs` will later drop
    # the entry at its resolve()-to-self check and the approval silently never
    # persists (re-prompt every session). Warn at the write boundary so that
    # caller bug surfaces here instead of as a mysterious never-remembered trust.
    key = _approved_key(target_dir)
    try:
        is_canonical = key == _normalize(target_dir)
    except OSError:
        # Resolving for the diagnostic failed; skip the warning rather than
        # abort the write. The read-time resolve()-to-self check is the actual
        # safety net, not this best-effort boundary hint.
        is_canonical = True
    if not is_canonical:
        logger.warning(
            "trust_skill_dir called with a non-canonical path %r; the stored "
            "entry will be dropped at read time. Pass an already-resolved "
            "directory.",
            target_dir,
        )

    dirs = data.get("dirs")
    if not isinstance(dirs, dict):
        dirs = {}
    dirs[key] = _TrustEntry(trusted_at=datetime.now(UTC).isoformat())
    return _save_store(_TrustStore(version=_STORAGE_VERSION, dirs=dirs), store_path)


def revoke_skill_dir_trust(
    target_dir: Path | str,
    *,
    store_path: Path | None = None,
) -> RevokeResult:
    """Remove trust for a skill directory.

    Matches on both the approved (expanduser-only) key form that
    `trust_skill_dir` stores and the fully-resolved form, so a caller can
    revoke either by the path they see in `skills trust list` or by the
    original symlink path.

    Args:
        target_dir: Directory to revoke.
        store_path: Path to the trust store file. Defaults to
            `~/.deepagents/.state/skill_trust.json`.

    Returns:
        `RevokeResult.REMOVED` if a matching entry was removed and persisted,
        `RevokeResult.NOT_FOUND` if no entry matched (store left unchanged), or
        `RevokeResult.ERROR` if the store could not be read or the write failed.
    """
    if store_path is None:
        store_path = _default_store_path()

    # Read strictly so a transient read error aborts rather than rebuilding
    # from `{}` and dropping the other entries on the next save.
    try:
        data = _load_store(store_path, strict=True)
    except (OSError, ValueError):
        logger.exception(
            "Refusing to revoke skill trust: could not read existing store %s",
            store_path,
        )
        return RevokeResult.ERROR
    dirs = data.get("dirs")
    if not isinstance(dirs, dict):
        return RevokeResult.NOT_FOUND
    keys = {_approved_key(target_dir), _normalize(target_dir)}
    removed = False
    for key in keys:
        if key in dirs:
            del dirs[key]
            removed = True
    if not removed:
        return RevokeResult.NOT_FOUND
    data["version"] = _STORAGE_VERSION
    data["dirs"] = dirs
    return RevokeResult.REMOVED if _save_store(data, store_path) else RevokeResult.ERROR


def clear_trusted_skill_dirs(*, store_path: Path | None = None) -> bool:
    """Remove all trusted skill directories.

    Args:
        store_path: Path to the trust store file. Defaults to
            `~/.deepagents/.state/skill_trust.json`.

    Returns:
        `True` if the store was cleared (or was already empty).
    """
    if store_path is None:
        store_path = _default_store_path()

    if not store_path.exists():
        return True
    return _save_store(_TrustStore(version=_STORAGE_VERSION, dirs={}), store_path)


def list_trusted_skill_dirs(
    *,
    store_path: Path | None = None,
    strict: bool = False,
) -> list[str]:
    """Return the sorted list of trusted skill directory paths.

    Args:
        store_path: Path to the trust store file. Defaults to
            `~/.deepagents/.state/skill_trust.json`.
        strict: When `True`, an existing-but-unreadable store re-raises instead
            of degrading to an empty list. The audit command (`skills trust
            list`) passes `strict=True` so it can report an error rather than
            falsely printing "No trusted skill directories" while entries the
            user cannot then see or revoke sit in an unreadable file.

    Returns:
        Sorted absolute directory paths previously trusted.

        When `strict`, an existing-but-unreadable or corrupt store propagates
        the underlying error (`OSError` / `json.JSONDecodeError` / `ValueError`)
        from `_load_store` instead of returning a list.
    """
    if store_path is None:
        store_path = _default_store_path()
    return sorted(_read_dirs(store_path, strict=strict))


def list_trusted_skill_dir_entries(
    *,
    store_path: Path | None = None,
    strict: bool = False,
) -> list[tuple[str, str]]:
    """Return trusted directories paired with their approval timestamps.

    The audit surface for the `trusted_at` metadata that `trust_skill_dir`
    records: `list_trusted_skill_dirs` returns only paths (all enforcement
    needs), so this is the one reader of the timestamp, used by `skills trust
    list` to show *when* each directory was approved.

    Args:
        store_path: Path to the trust store file. Defaults to
            `~/.deepagents/.state/skill_trust.json`.
        strict: Propagated to `_load_store`; see `list_trusted_skill_dirs`.

    Returns:
        `(path, trusted_at)` tuples sorted by path. `trusted_at` is the stored
        ISO-8601 string, or `""` when a hand-edited entry omitted or malformed
        it (the path is still listed so it remains visible and revocable).
    """
    if store_path is None:
        store_path = _default_store_path()
    entries: list[tuple[str, str]] = []
    for path, entry in _read_dirs(store_path, strict=strict).items():
        trusted_at = entry.get("trusted_at", "") if isinstance(entry, dict) else ""
        entries.append((path, trusted_at if isinstance(trusted_at, str) else ""))
    return sorted(entries)


def load_trusted_skill_dirs(*, store_path: Path | None = None) -> list[Path]:
    """Return verified trusted skill directories as canonical `Path` objects.

    Used to extend the containment allowlist passed to `load_skill_content`.

    Stored entries are the exact canonical directory the user approved (already
    resolved at trust time). Each entry is re-verified here rather than blindly
    re-resolved: if a stored path no longer resolves to itself — because it, or
    a parent component, was replaced with a symlink after approval — the current
    resolution would point somewhere the user never approved. Such entries are
    dropped (and logged) instead of silently allowlisting the swapped target, so
    a post-approval symlink swap re-prompts rather than granting access.

    Args:
        store_path: Path to the trust store file. Defaults to
            `~/.deepagents/.state/skill_trust.json`.

    Returns:
        Canonical directory paths that still resolve to themselves; empty when
        nothing is trusted.
    """
    verified: list[Path] = []
    for entry in list_trusted_skill_dirs(store_path=store_path):
        stored = Path(entry)
        try:
            resolves_to_self = stored.resolve() == stored
        except (OSError, RuntimeError):
            # A single unresolvable entry (e.g. a symlink cycle introduced under
            # the stored path) must not abort discovery of every other skill.
            # Drop it like the swap case below. `RuntimeError` is caught
            # alongside `OSError` to match the resolve guard in
            # `app._prompt_skill_trust_and_retry` (some Python builds surface a
            # symlink loop as `RuntimeError`).
            logger.warning(
                "Trusted skill directory %s could not be resolved; "
                "ignoring the trust entry.",
                entry,
                exc_info=True,
            )
            continue
        if resolves_to_self:
            verified.append(stored)
        else:
            logger.warning(
                "Trusted skill directory %s no longer resolves to itself "
                "(a symlink may have been introduced since approval); "
                "ignoring the stale trust entry.",
                entry,
            )
    return verified
