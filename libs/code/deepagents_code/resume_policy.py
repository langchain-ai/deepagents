"""Thread-resume policy shared by the TUI and non-interactive mode.

Kept free of Textual imports so the headless path can enforce the same
resume-age cutoff and working-directory comparison without loading `app.py`.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ResumeCutoff = tuple[datetime, str, bool]
"""The cutoff, the config source that set it, and whether it is rolling."""

CwdMatch = Literal["match", "unavailable", "mismatch"]
"""How a thread's stored cwd compares with the current one."""


def resume_cutoff_explanation(source: str) -> str:
    """Explain why the active resume cutoff exists and who set it.

    Returns:
        A short user-facing provenance and rationale sentence.
    """
    if source == "managed config":
        owner = "Your administrator set this policy through managed config"
    elif source == "config.toml":
        owner = "This policy was set in your config.toml"
    else:
        owner = "This policy was set by active configuration"
    return (
        f"{owner} to keep older threads from restoring stale context after "
        "model or policy changes."
    )


def resume_cutoff() -> ResumeCutoff | None:
    """Resolve the strictest configured thread-resume cutoff.

    Returns:
        The cutoff, its source, and whether it is rolling, or `None`.
    """
    from deepagents_code.config_manifest import (
        _emit_ranked_diagnostics,
        _ranked_source,
        get_option,
        parse_duration_seconds,
    )
    from deepagents_code.configuration.resolver import get_config_resolver

    resolver = get_config_resolver()
    cutoffs: list[ResumeCutoff] = []
    for key in ("threads.resume_after", "threads.max_resume_age"):
        option = get_option(key)
        if option is None:
            continue
        resolved = resolver.get(option)
        _emit_ranked_diagnostics(option, resolved)
        value = resolved.value
        if key.endswith("resume_after") and isinstance(value, str):
            cutoffs.append(
                (datetime.fromisoformat(value), _ranked_source(resolved), False)
            )
        elif key.endswith("max_resume_age") and isinstance(value, str):
            seconds = parse_duration_seconds(value)
            if seconds is not None:
                cutoff = datetime.min.replace(tzinfo=UTC)
                with suppress(OverflowError):
                    cutoff = datetime.now(UTC) - timedelta(seconds=seconds)
                cutoffs.append((cutoff, _ranked_source(resolved), True))
    return max(cutoffs, default=None)


async def thread_resume_block(
    thread_id: str, policy: ResumeCutoff | None
) -> str | None:
    """Return why `policy` blocks resuming a thread, if it does.

    Args:
        thread_id: Thread being resumed.
        policy: The active cutoff from `resume_cutoff`, or `None` for no policy.

    Returns:
        A user-facing explanation, or `None` when the thread may be resumed.
    """
    from deepagents_code.config_manifest import normalize_iso_datetime
    from deepagents_code.sessions import get_thread_updated_at

    if policy is None:
        return None
    cutoff, source, rolling = policy
    explanation = resume_cutoff_explanation(source)
    updated_value = await get_thread_updated_at(thread_id)
    normalized = normalize_iso_datetime(updated_value)
    if normalized is None:
        logger.warning(
            "Blocking resume for thread %s: missing or invalid updated_at %r",
            thread_id,
            updated_value,
        )
        return (
            f"Thread {thread_id} cannot be resumed because its last-updated "
            f"time could not be verified.\n\n{explanation}"
        )
    if datetime.fromisoformat(normalized) >= cutoff:
        return None
    reason = (
        "it is older than the configured maximum age"
        if rolling
        else f"it was last updated before {cutoff.isoformat()}"
    )
    return f"Thread {thread_id} cannot be resumed because {reason}.\n\n{explanation}"


def classify_thread_cwd(raw: str, current_cwd: str) -> tuple[CwdMatch, Path | None]:
    """Classify a stored thread cwd against the current cwd.

    Args:
        raw: The cwd recorded in the thread's checkpoint metadata. May be
            relative or use `~`; both are normalized here.
        current_cwd: The current working directory.

    Returns:
        A `(status, path)` pair. `path` is only set when `status` is
        `"mismatch"`; it is `None` otherwise. `status` is one of:

        - `"match"`: the stored cwd resolves to the current cwd; no action.
        - `"unavailable"`: the stored cwd is relative/malformed, or names an
            absolute directory that no longer exists — it cannot be honored.
        - `"mismatch"`: the stored cwd is a real directory that differs from
            the current cwd.
    """
    target = Path(raw).expanduser()
    if not target.is_absolute() or not target.is_dir():
        # Relative/malformed or missing directory: cannot be honored.
        return "unavailable", None
    try:
        current = Path(current_cwd).expanduser().resolve()
        resolved = target.resolve()
    except OSError:
        # Symlink resolution failed (e.g. ELOOP, permission on a path
        # component). Fall back to a non-resolving comparison, which can
        # report a spurious mismatch for symlinked-but-equal paths; log so
        # the degraded comparison is traceable.
        logger.debug(
            "Could not resolve cwd paths for mismatch check (%r vs %r); "
            "falling back to non-resolving comparison",
            current_cwd,
            raw,
            exc_info=True,
        )
        current = Path(current_cwd).expanduser().absolute()
        resolved = target.absolute()
    if current == resolved:
        return "match", None
    return "mismatch", resolved
