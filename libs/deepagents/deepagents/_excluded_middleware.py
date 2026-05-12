"""Filtering helpers for `HarnessProfile.excluded_middleware`.

These functions validate, apply, and audit exclusions against assembled
middleware stacks. The set of *required scaffolding* — classes/names that
must remain in the stack for the agent to function — is owned by
`deepagents.graph` and threaded through as parameters so that policy stays
next to `create_deep_agent`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentMiddleware

    from deepagents.profiles import HarnessProfile

logger = logging.getLogger(__name__)


def _validate_excluded_middleware_config(
    profile: HarnessProfile,
    *,
    required_classes: frozenset[type[AgentMiddleware[Any, Any, Any]]],
    required_names: frozenset[str],
) -> None:
    """Validate stack-independent guards on `profile.excluded_middleware`.

    Rejects required-scaffolding entries (class or the equivalent `.name`
    string). Grammar-level checks (empty strings, multi-colon, underscore
    prefix on plain names) already fire at `HarnessProfile` construction;
    this function focuses on the assembly-time invariant that scaffolding
    middleware must remain present for the agent to function.

    Args:
        profile: Profile whose `excluded_middleware` is validated.
        required_classes: Scaffolding classes that must not be excluded.
        required_names: Scaffolding `.name` values that must not be excluded.

    Raises:
        ValueError: If any entry is required scaffolding.
    """
    excluded = profile.excluded_middleware
    if not excluded:
        return

    excluded_classes: set[type[AgentMiddleware[Any, Any, Any]]] = set()
    excluded_names: set[str] = set()
    for entry in excluded:
        if isinstance(entry, type):
            excluded_classes.add(entry)
        else:
            excluded_names.add(entry)

    forbidden_classes = excluded_classes & required_classes
    forbidden_names = excluded_names & required_names
    if forbidden_classes or forbidden_names:
        # Lazy import: harness_profiles owns the per-class guidance text.
        from deepagents.profiles.harness.harness_profiles import _format_scaffolding_rejection  # noqa: PLC0415

        labels = [cls.__name__ for cls in forbidden_classes] + [f"{name!r} (string)" for name in forbidden_names]
        raise ValueError(_format_scaffolding_rejection(labels))


def _raise_on_name_collisions(
    name_matched_types: dict[str, set[type[AgentMiddleware[Any, Any, Any]]]],
) -> None:
    """Raise `ValueError` if any string exclusion matched multiple distinct classes.

    A string entry that drops instances of more than one concrete class is
    almost always a surprise — e.g. a user middleware whose `.name`
    accidentally collides with a built-in alias. Force the caller to use a
    class-form exclusion via the runtime `HarnessProfile` to disambiguate.
    """
    collisions = {name: classes for name, classes in name_matched_types.items() if len(classes) > 1}
    if not collisions:
        return
    labels = sorted(f"{name!r} matched {sorted(cls.__name__ for cls in classes)}" for name, classes in collisions.items())
    msg = (
        "HarnessProfile.excluded_middleware name entry matched multiple "
        "distinct middleware classes within a single stack: "
        f"{'; '.join(labels)}. Use a class-form exclusion via the runtime "
        "`HarnessProfile` to disambiguate."
    )
    raise ValueError(msg)


def _apply_excluded_middleware(
    stack: list[AgentMiddleware[Any, Any, Any]],
    profile: HarnessProfile,
    *,
    matched_classes: set[type[AgentMiddleware[Any, Any, Any]]] | None = None,
    matched_names: set[str] | None = None,
) -> list[AgentMiddleware[Any, Any, Any]]:
    """Drop middleware in the stack matched by `profile.excluded_middleware`.

    Class entries match on exact type (not `isinstance`), mirroring the
    slot-identity semantics of `_merge_middleware` so a subclass introduced
    by the caller is preserved when the profile excludes the base class.
    String entries match `AgentMiddleware.name` exactly — defaults to the
    class's `__name__` but is overridable when the public alias differs from
    the impl class (e.g. `SummarizationMiddleware` for
    `_DeepAgentsSummarizationMiddleware`).

    When `matched_classes` / `matched_names` are supplied, matches are
    recorded there so `_verify_excluded_middleware_coverage` can confirm
    every entry matched *somewhere* across the stacks the profile applies to
    (main agent + GP subagent). Per-stack checking would be too strict —
    a profile legitimately targets middleware only one stack carries. Omit
    the sets for single-stack filters where aggregation isn't meaningful.

    Args:
        stack: Fully assembled middleware list for a single agent/subagent.
        profile: Profile whose `excluded_middleware` drives the filter.
        matched_classes: Optional mutable set recording class matches across
            calls for the same profile.
        matched_names: Optional mutable set recording name matches, same
            lifetime semantics as `matched_classes`.

    Returns:
        A new list with excluded entries removed. Always a fresh list, even
            when no exclusions apply, so callers can mutate the result freely.
    """
    excluded = profile.excluded_middleware
    if not excluded:
        return list(stack)

    excluded_classes: set[type[AgentMiddleware[Any, Any, Any]]] = set()
    excluded_names: set[str] = set()
    for entry in excluded:
        if isinstance(entry, type):
            excluded_classes.add(entry)
        else:
            excluded_names.add(entry)

    filtered: list[AgentMiddleware[Any, Any, Any]] = []
    name_matched_types: dict[str, set[type[AgentMiddleware[Any, Any, Any]]]] = {}
    for mw in stack:
        mw_type = type(mw)
        mw_name = mw.name
        if mw_type in excluded_classes:
            if matched_classes is not None:
                matched_classes.add(mw_type)
            continue
        if mw_name in excluded_names:
            name_matched_types.setdefault(mw_name, set()).add(mw_type)
            if matched_names is not None:
                matched_names.add(mw_name)
            continue
        filtered.append(mw)

    _raise_on_name_collisions(name_matched_types)

    removed_count = len(stack) - len(filtered)
    if removed_count:
        logger.debug(
            "Dropped %d middleware instance(s) from stack per profile.excluded_middleware=%r (matched classes=%s, names=%s)",
            removed_count,
            sorted(repr(entry) for entry in profile.excluded_middleware),
            sorted(cls.__name__ for cls in excluded_classes),
            sorted(excluded_names),
        )
    return filtered


def _verify_excluded_middleware_coverage(
    profile: HarnessProfile,
    matched_classes: set[type[AgentMiddleware[Any, Any, Any]]],
    matched_names: set[str],
    *,
    required_classes: frozenset[type[AgentMiddleware[Any, Any, Any]]],
    required_names: frozenset[str],
) -> None:
    """Raise `ValueError` if any `profile.excluded_middleware` entry matched nothing.

    Run after every stack has been filtered so the accumulated `matched_*`
    sets reflect matches anywhere. An entry that matched nothing is almost
    always a typo or stale profile. Required-scaffolding and `_`-prefixed
    entries are skipped — rejected earlier by
    `_validate_excluded_middleware_config`.

    Args:
        profile: Profile whose `excluded_middleware` is being audited.
        matched_classes: Accumulated class matches across filter calls.
        matched_names: Accumulated name matches across filter calls.
        required_classes: Scaffolding classes; subtracted from unmatched so
            scaffolding exclusions (already rejected upstream) don't surface
            here as defense-in-depth.
        required_names: Scaffolding `.name` values; same purpose as
            `required_classes`.

    Raises:
        ValueError: If any entry is missing from the corresponding `matched_*` set.
    """
    excluded = profile.excluded_middleware
    if not excluded:
        return

    excluded_classes: set[type[AgentMiddleware[Any, Any, Any]]] = set()
    excluded_names: set[str] = set()
    for entry in excluded:
        if isinstance(entry, type):
            excluded_classes.add(entry)
        else:
            excluded_names.add(entry)

    unmatched_classes = excluded_classes - matched_classes - required_classes
    unmatched_names = excluded_names - matched_names - required_names
    # Private-prefix names are rejected by the config guard; skip them here
    # so the coverage error stays focused on legitimate "didn't match" cases.
    unmatched_names = {name for name in unmatched_names if not name.startswith("_")}
    if not unmatched_classes and not unmatched_names:
        return

    labels = sorted({cls.__name__ for cls in unmatched_classes} | {f"{name!r} (string)" for name in unmatched_names})
    msg = (
        f"HarnessProfile.excluded_middleware entries matched no middleware "
        f"across any assembled stack: {', '.join(labels)}. Typo or stale "
        f"profile — every exclusion must correspond to a middleware actually "
        f"present at runtime. (Tip: use class-form exclusion when the class "
        f"is available to catch typos at import time.)"
    )
    raise ValueError(msg)
