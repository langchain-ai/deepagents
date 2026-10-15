"""Code-local skills middleware adapter for plugin namespaces."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, NotRequired, cast

from deepagents.backends.protocol import FileInfo, LsResult
from deepagents.backends.utils import to_posix_path
from deepagents.middleware import skills as sdk_skills
from deepagents.middleware.skills import SkillsMiddleware
from langchain.agents.middleware.types import OmitFromOutput
from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer
from langgraph.types import Overwrite, interrupt

from deepagents_code.goal_state_notice import (
    GOAL_CONTROL_MESSAGE_SOURCE,
    is_internal_message,
    message_source,
)
from deepagents_code.plugins.adapters.skills import (
    CodeSkillSource,
    SkillNamespace,
    namespaced_skill_name,
)
from deepagents_code.skills.invocation import extra_skill_roots, pinned_skill_body
from deepagents_code.skills.load import load_skill_content
from deepagents_code.skills.load_errors import skills_load_errors_event
from deepagents_code.skills.merge import merge_skill
from deepagents_code.skills.trust import (
    SKILL_TRUST_INTERRUPT_TYPE,
    SkillTrustRequest,
    trust_skill_dir,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from deepagents.backends.protocol import BackendProtocol
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

_PLUGIN_SKILL_SOURCE_LENGTH = 3
_SKILL_FILE = "SKILL.md"

_PROCESS_ID = uuid.uuid4().hex
"""Random ID of this process, recorded with each skill load.

A thread's skills were loaded by an earlier process when its recorded ID
differs. `/reload` restarts the agent server, so every thread rescans its skill
sources on its next turn.
"""


class _PluginSkillsState(sdk_skills.SkillsState):
    """SDK skills state plus the process that loaded `skills_metadata`."""

    skills_loaded_by: NotRequired[Annotated[str, OmitFromOutput]]
    """`_PROCESS_ID` of the process whose scan wrote `skills_metadata`.

    Not private, so a forked subagent inherits it with `skills_metadata` and
    does not rescan.
    """


class _SkillsStateUpdate(sdk_skills.SkillsStateUpdate, total=False):
    """SDK skills update that can also replace the pending `pinned_skills`."""

    pinned_skills: Overwrite
    skills_loaded_by: str


def _skill_invocation(state: sdk_skills.SkillsState) -> dict[str, object]:
    """Return the `__skill` marker of the newest visible human message.

    Stops at that message, so an older turn's selection never applies. A goal
    continuation is hidden but starts its own turn, so it also stops the search.
    """
    for message in reversed(state.get("messages", [])):
        if not isinstance(message, HumanMessage):
            continue
        if message_source(message) == GOAL_CONTROL_MESSAGE_SOURCE:
            return {}
        if not is_internal_message(message):
            marker = message.additional_kwargs.get("__skill")
            return marker if isinstance(marker, dict) else {}
    return {}


def _pending_pin_update(state: sdk_skills.SkillsState) -> Overwrite | None:
    """Return an update retaining only the current turn's explicit selection.

    The SDK `pinned_skills` reducer appends any requested name. Only the name
    from this turn's trust-checked `/skill:` invocation may be pinned, and a pin
    left over from a failed turn is dropped.
    """
    names = state.get("pinned_skills") or []
    name = _skill_invocation(state).get("name")
    selected = [name] if isinstance(name, str) and name in names else []
    return Overwrite(selected) if selected != names else None


def _loaded_here(state: Mapping[str, object]) -> bool:
    """Return whether this process wrote the loaded `skills_metadata`.

    Reads `state` as a mapping because the SDK hooks type it as `SkillsState`,
    which lacks the `_PluginSkillsState` key.
    """
    return state.get("skills_loaded_by") == _PROCESS_ID


def _needs_load(state: sdk_skills.SkillsState) -> bool:
    """Return whether `before_agent` must scan the skill sources.

    Metadata loaded by an earlier process can be stale (see `_PROCESS_ID`).
    Current metadata serves a pending pin when it lists the selected skill.
    Otherwise the skill may have been added since the load.
    """
    loaded = state.get("skills_metadata")
    if loaded is None or not _loaded_here(state):
        return True
    name = _skill_invocation(state).get("name")
    if name not in (state.get("pinned_skills") or []):
        # No pin survives this turn, so the loaded metadata is enough.
        return False
    return all(skill["name"] != name for skill in loaded)


def _loaded_update(state: sdk_skills.SkillsState) -> _SkillsStateUpdate | None:
    """Return the pin trim for a turn that keeps the loaded metadata.

    Returns:
        An update replacing `pinned_skills`, or `None` when nothing changes.
    """
    pins = _pending_pin_update(state)
    if pins is None:
        return None
    return _SkillsStateUpdate(
        skills_metadata=state.get("skills_metadata") or [], pinned_skills=pins
    )


def _trusted_directory(skill: sdk_skills.SkillMetadata, denied: str) -> str:
    """Ask the user to trust the skill's resolved directory, then save it.

    `interrupt` returns the client's answer when the run resumes, so the
    calling node runs again from the start with the same skill.

    Args:
        skill: Skill whose `SKILL.md` resolves outside the trusted roots.
        denied: Containment error to report if the user does not approve.

    Returns:
        The approved directory, which is now in the trust store.

    Raises:
        ValueError: The path cannot be resolved, the user denied trust, or
            the approval could not be saved.
    """
    name = skill["name"]
    try:
        directory = str(Path(skill["path"]).resolve().parent)
    except (OSError, RuntimeError) as exc:
        raise ValueError(denied) from exc
    answer = interrupt(
        SkillTrustRequest(
            type=SKILL_TRUST_INTERRUPT_TYPE, skill=name, directory=directory
        )
    )
    if not isinstance(answer, dict) or answer.get("approved") is not True:
        raise ValueError(denied)
    if not trust_skill_dir(directory):
        msg = (
            f"Could not save trust for {directory}, so skill '{name}' cannot "
            "run. Check that the state directory is writable, or add the "
            "directory to [skills].extra_allowed_dirs."
        )
        raise ValueError(msg)
    return directory


def _entries(
    ls_result: object, path: str, errors: list[str] | None = None
) -> list[FileInfo]:
    """Normalize a backend `ls` result to a list of entry dicts.

    Appends a listing error for `path` to `errors`, in the SDK's format for
    skill source errors.

    Returns:
        The listing entries, or an empty list when the result is empty or an
        unexpected shape.
    """
    if isinstance(ls_result, LsResult):
        if ls_result.error and errors is not None:
            errors.append(sdk_skills._format_skills_source_error(path, ls_result.error))
        return list(ls_result.entries or [])
    if isinstance(ls_result, list):
        return cast("list[FileInfo]", ls_result)
    return []


def _child_dirs(entries: list[FileInfo], root: str) -> list[tuple[str, str]]:
    """Return `(name, path)` for each immediate subdirectory in `entries`.

    Returns:
        Name/path pairs for each immediate subdirectory, excluding `root`.
    """
    root_posix = PurePosixPath(to_posix_path(root))
    dirs: list[tuple[str, str]] = []
    for entry in entries:
        if not entry.get("is_dir"):
            continue
        path = entry["path"]
        name = PurePosixPath(to_posix_path(path)).name
        # Skip the source dir itself if a backend echoes it back.
        if PurePosixPath(to_posix_path(path)) == root_posix:
            continue
        dirs.append((name, path))
    return dirs


def _has_skill_file(entries: list[FileInfo], root: str) -> bool:
    """Return whether `entries` contains a `SKILL.md` directly under `root`."""
    root_posix = PurePosixPath(to_posix_path(root))
    for entry in entries:
        path = PurePosixPath(to_posix_path(entry["path"]))
        if path.name == _SKILL_FILE and path.parent == root_posix:
            return True
    return False


def _missing(path: str) -> bool:
    """Return whether `path` does not exist, so its listing error is expected."""
    try:
        return not Path(path).exists()
    except OSError:
        # Cannot be checked, for example a parent without search permission.
        return False


def _report_unreadable(errors: list[str]) -> None:
    """Stream errors for skill directories that exist but cannot be read.

    The SDK shows skill load errors only to the model. The client shows these
    to the user, because the agent silently loses the skills they hold.
    """
    if not errors:
        return
    try:
        get_stream_writer()(skills_load_errors_event(errors))
    except Exception:
        # Best effort: a hook called outside a graph run has no stream.
        logger.debug("Could not stream skill load errors", exc_info=True)


def _skill_md_path(skill_dir: str) -> str:
    """Return the `SKILL.md` path inside a skill directory."""
    return str(PurePosixPath(to_posix_path(skill_dir)) / _SKILL_FILE)


def _namespace_skill(
    skill: sdk_skills.SkillMetadata,
    namespace: SkillNamespace,
    subfolders: tuple[str, ...],
) -> sdk_skills.SkillMetadata:
    """Return a copy of `skill` with a namespace-qualified name."""
    return cast(
        "sdk_skills.SkillMetadata",
        {
            **skill,
            "name": namespaced_skill_name(namespace, skill["name"], subfolders),
        },
    )


def discover_skill_dirs(
    backend: BackendProtocol,
    source_path: str,
    *,
    errors: list[str] | None = None,
) -> list[tuple[str, tuple[str, ...]]]:
    """Return `(skill_dir, subfolders)` pairs found under `source_path`.

    Walks the source tree, treating any directory that directly contains a
    `SKILL.md` as a skill directory (a recursion leaf, like a plugin walker).
    `subfolders` holds the directory names between the source
    root and the skill directory, excluding the skill directory's own name.
    A directory that cannot be listed adds a message to `errors`.

    Returns:
        Skill directories paired with their intermediate subfolder segments.
    """
    found: list[tuple[str, tuple[str, ...]]] = []
    # `path_segments` accumulates directory names from the source root down to
    # and including `current`. A skill directory's own name is dropped when
    # naming, since the skill's terminal identifier is its frontmatter name;
    # only the directories above it form the namespace segments.
    source_root = Path(source_path).resolve()
    visited: set[Path] = set()
    stack: list[tuple[str, tuple[str, ...]]] = [(str(source_root), ())]
    while stack:
        current, path_segments = stack.pop()
        try:
            resolved = Path(current).resolve()
        except (OSError, RuntimeError):
            logger.warning("Could not resolve plugin skill directory %s", current)
            continue
        if not resolved.is_relative_to(source_root) or resolved in visited:
            continue
        visited.add(resolved)
        resolved_path = str(resolved)
        entries = _entries(backend.ls(resolved_path), resolved_path, errors)
        if _has_skill_file(entries, resolved_path):
            found.append((resolved_path, path_segments[:-1]))
            continue
        for name, path in _child_dirs(entries, resolved_path):
            stack.append((path, (*path_segments, name)))
    return found


async def adiscover_skill_dirs(
    backend: BackendProtocol,
    source_path: str,
    *,
    errors: list[str] | None = None,
) -> list[tuple[str, tuple[str, ...]]]:
    """Async counterpart of `discover_skill_dirs`.

    Returns:
        Skill directories paired with their intermediate subfolder segments.
    """
    found: list[tuple[str, tuple[str, ...]]] = []
    source_root = await asyncio.to_thread(Path(source_path).resolve)
    visited: set[Path] = set()
    stack: list[tuple[str, tuple[str, ...]]] = [(str(source_root), ())]
    while stack:
        current, path_segments = stack.pop()
        try:
            resolved = await asyncio.to_thread(Path(current).resolve)
        except (OSError, RuntimeError):
            logger.warning("Could not resolve plugin skill directory %s", current)
            continue
        if not resolved.is_relative_to(source_root) or resolved in visited:
            continue
        visited.add(resolved)
        resolved_path = str(resolved)
        entries = _entries(await backend.als(resolved_path), resolved_path, errors)
        if _has_skill_file(entries, resolved_path):
            found.append((resolved_path, path_segments[:-1]))
            continue
        for name, path in _child_dirs(entries, resolved_path):
            stack.append((path, (*path_segments, name)))
    return found


def load_namespaced_skills(
    backend: BackendProtocol,
    source_path: str,
    namespace: SkillNamespace,
    *,
    errors: list[str] | None = None,
) -> list[sdk_skills.SkillMetadata]:
    """Load and namespace every skill found under a plugin source.

    Reads each discovered skill directory's `SKILL.md` directly, since the SDK
    loader only scans one level below a source and would not read a leaf
    directory's own `SKILL.md`. Nested directories become `:`-joined namespace
    segments (e.g. `plugin:foo:bar:review`). A directory that cannot be listed
    adds a message to `errors`.

    Returns:
        Namespace-qualified skill metadata for the source.
    """
    skill_dirs = discover_skill_dirs(backend, source_path, errors=errors)
    if not skill_dirs:
        return []
    paths = [_skill_md_path(skill_dir) for skill_dir, _ in skill_dirs]
    responses = backend.download_files(paths)
    skills: list[sdk_skills.SkillMetadata] = []
    for (skill_dir, segments), path, response in zip(
        skill_dirs, paths, responses, strict=True
    ):
        skill = sdk_skills._skill_metadata_from_response(response, skill_dir, path)
        if skill is not None:
            skills.append(_namespace_skill(skill, namespace, segments))
    return skills


async def aload_namespaced_skills(
    backend: BackendProtocol,
    source_path: str,
    namespace: SkillNamespace,
    *,
    errors: list[str] | None = None,
) -> list[sdk_skills.SkillMetadata]:
    """Async counterpart of `load_namespaced_skills`.

    Returns:
        Namespace-qualified skill metadata for the source.
    """
    skill_dirs = await adiscover_skill_dirs(backend, source_path, errors=errors)
    if not skill_dirs:
        return []
    paths = [_skill_md_path(skill_dir) for skill_dir, _ in skill_dirs]
    responses = await backend.adownload_files(paths)
    skills: list[sdk_skills.SkillMetadata] = []
    for (skill_dir, segments), path, response in zip(
        skill_dirs, paths, responses, strict=True
    ):
        skill = sdk_skills._skill_metadata_from_response(response, skill_dir, path)
        if skill is not None:
            skills.append(_namespace_skill(skill, namespace, segments))
    return skills


class PluginSkillsMiddleware(SkillsMiddleware):
    """Load namespaced plugin skills without extending the SDK source API.

    Wraps the SDK `SkillsMiddleware`. Sources without a namespace load exactly
    as the SDK loads them. Sources carrying a plugin namespace are walked
    recursively so nested skill directories (`skills/foo/bar/review/SKILL.md`)
    are discovered, and each skill's name is qualified as
    `plugin_id:foo:bar:review` before the last-one-wins merge — matching
    the plugin skill naming convention.
    """

    state_schema = _PluginSkillsState

    def __init__(
        self,
        *,
        backend: BackendProtocol,
        sources: Sequence[CodeSkillSource],
        system_prompt: str | None = sdk_skills.SKILLS_SYSTEM_PROMPT,
        path_base: Path | None = None,
    ) -> None:
        """Initialize the middleware with Code-local plugin source tuples.

        Args:
            backend: Backend used to load skill files.
            sources: Ordered Code skill sources, optionally including a plugin
                namespace as the third tuple item.
            system_prompt: Skills prompt template passed to the SDK middleware.
            path_base: Directory that relative `skills.extra_allowed_dirs`
                entries resolve against. If `None`, the process working
                directory is used.
        """
        self._allowed_roots = tuple(Path(source[0]).resolve() for source in sources)
        self._path_base = path_base
        sdk_sources = [(source[0], source[1]) for source in sources]
        super().__init__(
            backend=backend,
            sources=sdk_sources,
            system_prompt=system_prompt,
        )
        self._namespaces = tuple(
            source[2] if len(source) == _PLUGIN_SKILL_SOURCE_LENGTH else None
            for source in sources
        )

    @staticmethod
    def _read_skill(skill: sdk_skills.SkillMetadata, roots: Sequence[Path]) -> str:
        """Return the selected skill's `SKILL.md` read within trusted roots.

        A skill outside every root interrupts the run to ask the user to trust
        its directory. An approval is saved, because later turns read trust
        only from the saved store.

        Errors are raised as `ValueError` because the agent server replaces the
        message of most other exception types, including `PermissionError`,
        before it reaches the client.

        Raises:
            ValueError: No trusted roots exist, the user denied trust, the
                approval could not be saved, or the file is unreadable.
        """
        name = skill["name"]
        # `load_skill_content` skips containment when given no roots, so an
        # empty list must fail here rather than read an unchecked path.
        if not roots:
            msg = f"Skill '{name}' cannot run: no trusted skill directories exist."
            raise ValueError(msg)
        try:
            content = load_skill_content(skill["path"], allowed_roots=roots)
        except PermissionError as exc:
            directory = _trusted_directory(skill, str(exc))
            try:
                content = load_skill_content(
                    skill["path"], allowed_roots=[*roots, Path(directory)]
                )
            except PermissionError as retry_exc:
                msg = (
                    f"Skill '{name}' changed location after you approved it, "
                    "so it was not read."
                )
                raise ValueError(msg) from retry_exc
        if content is None:
            msg = (
                f"Could not read instructions for skill '{name}'. Check that "
                "its SKILL.md exists, is UTF-8, and is not too large."
            )
            raise ValueError(msg)
        return content

    def before_model(
        self, state: sdk_skills.SkillsState, runtime: Runtime
    ) -> dict[str, object] | None:
        """Return SDK pin updates after validating skill selection and trust.

        Returns:
            An update that appends one pinned-skill message per requested name
                and clears `pinned_skills`, or `None` when nothing is pending.

        Raises:
            ValueError: A requested skill is missing on the server, or fails
                the trust or content checks.
        """
        del runtime
        names = state.get("pinned_skills")
        if not names:
            return None
        by_name = {skill["name"]: skill for skill in state.get("skills_metadata") or []}
        roots = [*self._allowed_roots, *extra_skill_roots(self._path_base)]
        messages: list[HumanMessage] = []
        for name in dict.fromkeys(names):
            if name not in by_name:
                msg = (
                    f"Skill '{name}' was not found. Check the name, or run "
                    "/reload if you just added it."
                )
                raise ValueError(msg)
            skill = by_name[name]
            body = pinned_skill_body(skill, self._read_skill(skill, roots))
            messages.append(sdk_skills._pinned_skill_message(skill, body))
        return {"pinned_skills": Overwrite([]), "messages": messages}

    async def abefore_model(
        self, state: sdk_skills.SkillsState, runtime: Runtime
    ) -> dict[str, object] | None:
        """Return trust-checked pin updates without blocking the event loop."""
        return await asyncio.to_thread(self.before_model, state, runtime)

    @staticmethod
    def _state_update(
        all_skills: dict[str, sdk_skills.SkillMetadata],
        errors: list[str],
        unreadable: list[str],
        state: sdk_skills.SkillsState,
    ) -> _SkillsStateUpdate:
        """Build the middleware state update, logging any load errors.

        Args:
            all_skills: Merged skill metadata keyed by name.
            errors: Every source error, shown to the model.
            unreadable: Errors for sources that exist, also shown to the user.
            state: State whose pending pins are trimmed.

        Returns:
            The state update carrying merged skill metadata and any errors.
        """
        # Always write the errors so warnings from an earlier load are cleared.
        update = _SkillsStateUpdate(
            skills_metadata=list(all_skills.values()),
            skills_load_errors=errors,
            skills_loaded_by=_PROCESS_ID,
        )
        if (pins := _pending_pin_update(state)) is not None:
            update["pinned_skills"] = pins
        if errors:
            logger.warning("Skills load errors: %s", errors)
        _report_unreadable(unreadable)
        return update

    def _load_source(
        self, source_path: str, namespace: SkillNamespace | None
    ) -> tuple[list[sdk_skills.SkillMetadata], list[str]]:
        """Return one source's skills and the errors from listing it."""
        if namespace is None:
            skills, error = sdk_skills._list_skills_with_errors(
                self._backend, source_path
            )
            return skills, [] if error is None else [error]
        errors: list[str] = []
        skills = load_namespaced_skills(
            self._backend, source_path, namespace, errors=errors
        )
        return skills, errors

    async def _aload_source(
        self, source_path: str, namespace: SkillNamespace | None
    ) -> tuple[list[sdk_skills.SkillMetadata], list[str]]:
        """Async counterpart of `_load_source`.

        Returns:
            The source's skills and the errors from listing it.
        """
        if namespace is None:
            skills, error = await sdk_skills._alist_skills_with_errors(
                self._backend, source_path
            )
            return skills, [] if error is None else [error]
        errors: list[str] = []
        skills = await aload_namespaced_skills(
            self._backend, source_path, namespace, errors=errors
        )
        return skills, errors

    def before_agent(
        self,
        state: sdk_skills.SkillsState,
        runtime: Runtime,  # noqa: ARG002
        config: RunnableConfig,  # noqa: ARG002
    ) -> sdk_skills.SkillsStateUpdate | None:
        """Load and namespace plugin skills before collision resolution.

        Returns:
            A state update containing collision-safe skill metadata and the
                trimmed `pinned_skills`, or `None` when the loaded skills serve
                the turn unchanged.
        """
        if not _needs_load(state):
            return _loaded_update(state)

        all_skills: dict[str, sdk_skills.SkillMetadata] = {}
        merged_source_labels: dict[str, str | None] = {}
        errors: list[str] = []
        unreadable: list[str] = []

        # `self.sources`, `self.source_labels`, and `self._namespaces` are all
        # built from the same source sequence at the same indices (see
        # `__init__` and the SDK base), so this zip is aligned by construction.
        # `strict=True` turns future *length* drift into a loud error; it does
        # not catch a same-length reorder, which would still mispair silently.
        for source_path, source_label, namespace in zip(
            self.sources, self.source_labels, self._namespaces, strict=True
        ):
            source_skills, source_errors = self._load_source(source_path, namespace)
            errors.extend(source_errors)
            # Optional skill directories often do not exist; that is not news.
            if source_errors and not _missing(source_path):
                unreadable.extend(source_errors)
            for skill in source_skills:
                merge_skill(
                    all_skills,
                    merged_source_labels,
                    skill,
                    source_label=source_label,
                )

        return self._state_update(all_skills, errors, unreadable, state)

    async def abefore_agent(
        self,
        state: sdk_skills.SkillsState,
        runtime: Runtime,  # noqa: ARG002
        config: RunnableConfig,  # noqa: ARG002
    ) -> sdk_skills.SkillsStateUpdate | None:
        """Asynchronously load and namespace skills before collision resolution.

        Returns:
            A state update containing collision-safe skill metadata and the
                trimmed `pinned_skills`, or `None` when the loaded skills serve
                the turn unchanged.
        """
        if not await asyncio.to_thread(_needs_load, state):
            return _loaded_update(state)

        all_skills: dict[str, sdk_skills.SkillMetadata] = {}
        merged_source_labels: dict[str, str | None] = {}
        errors: list[str] = []
        unreadable: list[str] = []

        # See `before_agent`: the three sequences are index-aligned by
        # construction, and `strict=True` guards against future length drift.
        for source_path, source_label, namespace in zip(
            self.sources, self.source_labels, self._namespaces, strict=True
        ):
            source_skills, source_errors = await self._aload_source(
                source_path, namespace
            )
            errors.extend(source_errors)
            if source_errors and not await asyncio.to_thread(_missing, source_path):
                unreadable.extend(source_errors)
            for skill in source_skills:
                merge_skill(
                    all_skills,
                    merged_source_labels,
                    skill,
                    source_label=source_label,
                )

        return self._state_update(all_skills, errors, unreadable, state)
