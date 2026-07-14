"""Code-local skills middleware adapter for plugin namespaces."""

from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, cast

from deepagents.backends.protocol import FileInfo, LsResult
from deepagents.backends.utils import to_posix_path
from deepagents.middleware import skills as sdk_skills
from deepagents.middleware.skills import SkillsMiddleware

from deepagents_code.plugins.adapters.skills import (
    CodeSkillSource,
    SkillNamespace,
    namespaced_skill_name,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

_PLUGIN_SKILL_SOURCE_LENGTH = 3
_SKILL_FILE = "SKILL.md"


def _entries(ls_result: object) -> list[FileInfo]:
    """Normalize a backend `ls` result to a list of entry dicts.

    Returns:
        The listing entries, or an empty list when the result is empty or an
        unexpected shape.
    """
    if isinstance(ls_result, LsResult):
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
) -> list[tuple[str, tuple[str, ...]]]:
    """Return `(skill_dir, subfolders)` pairs found under `source_path`.

    Walks the source tree, treating any directory that directly contains a
    `SKILL.md` as a skill directory (a recursion leaf, like Claude Code's
    plugin walker). `subfolders` holds the directory names between the source
    root and the skill directory, excluding the skill directory's own name.

    Returns:
        Skill directories paired with their intermediate subfolder segments.
    """
    found: list[tuple[str, tuple[str, ...]]] = []
    # `path_segments` accumulates directory names from the source root down to
    # and including `current`. A skill directory's own name is dropped when
    # naming, since the skill's terminal identifier is its frontmatter name;
    # only the directories above it form the namespace segments.
    stack: list[tuple[str, tuple[str, ...]]] = [(source_path, ())]
    while stack:
        current, path_segments = stack.pop()
        entries = _entries(backend.ls(current))
        if _has_skill_file(entries, current):
            found.append((current, path_segments[:-1]))
            continue
        for name, path in _child_dirs(entries, current):
            stack.append((path, (*path_segments, name)))
    return found


async def adiscover_skill_dirs(
    backend: BackendProtocol,
    source_path: str,
) -> list[tuple[str, tuple[str, ...]]]:
    """Async counterpart of `discover_skill_dirs`.

    Returns:
        Skill directories paired with their intermediate subfolder segments.
    """
    found: list[tuple[str, tuple[str, ...]]] = []
    stack: list[tuple[str, tuple[str, ...]]] = [(source_path, ())]
    while stack:
        current, path_segments = stack.pop()
        entries = _entries(await backend.als(current))
        if _has_skill_file(entries, current):
            found.append((current, path_segments[:-1]))
            continue
        for name, path in _child_dirs(entries, current):
            stack.append((path, (*path_segments, name)))
    return found


def load_namespaced_skills(
    backend: BackendProtocol,
    source_path: str,
    namespace: SkillNamespace,
) -> list[sdk_skills.SkillMetadata]:
    """Load and namespace every skill found under a plugin source.

    Reads each discovered skill directory's `SKILL.md` directly, since the SDK
    loader only scans one level below a source and would not read a leaf
    directory's own `SKILL.md`. Nested directories become `:`-joined namespace
    segments (e.g. `plugin:foo:bar:review`).

    Returns:
        Namespace-qualified skill metadata for the source.
    """
    skill_dirs = discover_skill_dirs(backend, source_path)
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
) -> list[sdk_skills.SkillMetadata]:
    """Async counterpart of `load_namespaced_skills`.

    Returns:
        Namespace-qualified skill metadata for the source.
    """
    skill_dirs = await adiscover_skill_dirs(backend, source_path)
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
    Claude Code's plugin skill naming.
    """

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES,
        sources: Sequence[CodeSkillSource],
        system_prompt: str | None = sdk_skills.SKILLS_SYSTEM_PROMPT,
    ) -> None:
        """Initialize the middleware with Code-local plugin source tuples.

        Args:
            backend: Backend used to load skill files.
            sources: Ordered Code skill sources, optionally including a plugin
                namespace as the third tuple item.
            system_prompt: Skills prompt template passed to the SDK middleware.
        """
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
    def _state_update(
        all_skills: dict[str, sdk_skills.SkillMetadata],
        errors: list[str],
    ) -> sdk_skills.SkillsStateUpdate:
        """Build the middleware state update, logging any load errors.

        Returns:
            The state update carrying merged skill metadata and any errors.
        """
        update = sdk_skills.SkillsStateUpdate(skills_metadata=list(all_skills.values()))
        if errors:
            logger.warning("Skills load errors: %s", errors)
            update["skills_load_errors"] = errors
        return update

    def before_agent(
        self,
        state: sdk_skills.SkillsState,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> sdk_skills.SkillsStateUpdate | None:
        """Load and namespace plugin skills before collision resolution.

        Returns:
            A state update containing collision-safe skill metadata, or `None`
            when skills are already loaded.
        """
        if "skills_metadata" in state:
            return None

        backend = self._get_backend(state, runtime, config)
        all_skills: dict[str, sdk_skills.SkillMetadata] = {}
        errors: list[str] = []

        for source_path, namespace in zip(self.sources, self._namespaces, strict=True):
            if namespace is None:
                source_skills, source_error = sdk_skills._list_skills_with_errors(
                    backend, source_path
                )
                if source_error is not None:
                    errors.append(source_error)
            else:
                source_skills = load_namespaced_skills(backend, source_path, namespace)
            for skill in source_skills:
                all_skills[skill["name"]] = skill

        return self._state_update(all_skills, errors)

    async def abefore_agent(
        self,
        state: sdk_skills.SkillsState,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> sdk_skills.SkillsStateUpdate | None:
        """Asynchronously load and namespace skills before collision resolution.

        Returns:
            A state update containing collision-safe skill metadata, or `None`
            when skills are already loaded.
        """
        if "skills_metadata" in state:
            return None

        backend = self._get_backend(state, runtime, config)
        all_skills: dict[str, sdk_skills.SkillMetadata] = {}
        errors: list[str] = []

        for source_path, namespace in zip(self.sources, self._namespaces, strict=True):
            if namespace is None:
                (
                    source_skills,
                    source_error,
                ) = await sdk_skills._alist_skills_with_errors(backend, source_path)
                if source_error is not None:
                    errors.append(source_error)
            else:
                source_skills = await aload_namespaced_skills(
                    backend, source_path, namespace
                )
            for skill in source_skills:
                all_skills[skill["name"]] = skill

        return self._state_update(all_skills, errors)
