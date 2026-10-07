"""Code-local skills middleware adapter for plugin namespaces."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, cast

from deepagents.backends.protocol import FileDownloadResponse, FileInfo, LsResult
from deepagents.backends.utils import to_posix_path
from deepagents.middleware import skills as sdk_skills
from deepagents.middleware.skills import SkillsMiddleware
from langchain_core.messages import HumanMessage

from deepagents_code.config import _use_extra_skills_path_base
from deepagents_code.config_manifest import get_option
from deepagents_code.configuration.resolver import get_config_resolver
from deepagents_code.goal_state_notice import is_internal_message
from deepagents_code.plugins.adapters.skills import (
    CodeSkillSource,
    SkillNamespace,
    namespaced_skill_name,
)
from deepagents_code.skills.load import load_skill_content
from deepagents_code.skills.merge import merge_skill
from deepagents_code.skills.trust import load_trusted_skill_dirs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents.backends.protocol import BackendProtocol
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
    `SKILL.md` as a skill directory (a recursion leaf, like a plugin walker).
    `subfolders` holds the directory names between the source
    root and the skill directory, excluding the skill directory's own name.

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
        entries = _entries(backend.ls(resolved_path))
        if _has_skill_file(entries, resolved_path):
            found.append((resolved_path, path_segments[:-1]))
            continue
        for name, path in _child_dirs(entries, resolved_path):
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
        entries = _entries(await backend.als(resolved_path))
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
    the plugin skill naming convention.
    """

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
            path_base: User working directory for relative extra skill roots.
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

    def _pin_roots(self) -> list[Path]:
        """Return configured and currently approved skill directories."""
        roots = [*self._allowed_roots, *load_trusted_skill_dirs()]
        option = get_option("skills.extra_allowed_dirs")
        if option is not None:
            with _use_extra_skills_path_base(self._path_base):
                extra_roots = cast(
                    "list[Path] | None", get_config_resolver().get(option).value
                )
            roots.extend(Path(root).resolve() for root in extra_roots or ())
        return roots

    @staticmethod
    def _pin_response(
        skill: sdk_skills.SkillMetadata,
        expected_path: str | None,
        roots: Sequence[Path],
    ) -> FileDownloadResponse:
        """Return the selected skill read within trusted directories.

        Raises:
            PermissionError: No trusted roots exist or the skill escapes them.
            ValueError: The selected skill moved or cannot be read.
        """
        path = Path(skill["path"]).resolve()
        if expected_path is not None and str(path) != expected_path:
            msg = f"Skill '{skill['name']}' changed location; invoke it again."
            raise ValueError(msg)
        if not roots:
            msg = "No trusted skill directories are configured."
            raise PermissionError(msg)
        content = load_skill_content(str(path), allowed_roots=roots)
        response = FileDownloadResponse(
            path=skill["path"], content=content.encode("utf-8") if content else None
        )
        if not sdk_skills._pinned_skill_body(skill, response):
            msg = f"Could not read instructions for skill '{skill['name']}'."
            raise ValueError(msg)
        return response

    def before_model(
        self, state: sdk_skills.SkillsState, runtime: Runtime
    ) -> dict[str, object] | None:
        """Return SDK pin updates after validating skill selection and trust.

        Raises:
            ValueError: A requested skill is no longer available.
        """
        del runtime
        names = state.get("pinned_skills")
        if not names:
            return None
        by_name = {skill["name"]: skill for skill in state.get("skills_metadata") or []}
        expected: dict[str, str] = {}
        for message in reversed(state.get("messages", [])):
            if is_internal_message(message):
                continue
            if isinstance(message, HumanMessage):
                marker = message.additional_kwargs.get("__skill")
                if isinstance(marker, dict) and isinstance(marker.get("path"), str):
                    expected[marker.get("name", "")] = marker["path"]
                break
        roots = self._pin_roots()
        skills: list[sdk_skills.SkillMetadata] = []
        responses: list[FileDownloadResponse] = []
        for name in dict.fromkeys(names):
            if name not in by_name:
                msg = f"Skill '{name}' is no longer available; invoke it again."
                raise ValueError(msg)
            skill = by_name[name]
            responses.append(self._pin_response(skill, expected.get(name), roots))
            skills.append(skill)
        return sdk_skills._pin_update(skills, responses)

    async def abefore_model(
        self, state: sdk_skills.SkillsState, runtime: Runtime
    ) -> dict[str, object] | None:
        """Return trust-checked pin updates without blocking the event loop."""
        return await asyncio.to_thread(self.before_model, state, runtime)

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
        runtime: Runtime,  # noqa: ARG002
        config: RunnableConfig,  # noqa: ARG002
    ) -> sdk_skills.SkillsStateUpdate | None:
        """Load and namespace plugin skills before collision resolution.

        Returns:
            A state update containing collision-safe skill metadata, or `None`
            when skills are already loaded.
        """
        if "skills_metadata" in state and not state.get("pinned_skills"):
            return None

        backend = self._backend
        all_skills: dict[str, sdk_skills.SkillMetadata] = {}
        merged_source_labels: dict[str, str | None] = {}
        errors: list[str] = []

        # `self.sources`, `self.source_labels`, and `self._namespaces` are all
        # built from the same source sequence at the same indices (see
        # `__init__` and the SDK base), so this zip is aligned by construction.
        # `strict=True` turns future *length* drift into a loud error; it does
        # not catch a same-length reorder, which would still mispair silently.
        for source_path, source_label, namespace in zip(
            self.sources, self.source_labels, self._namespaces, strict=True
        ):
            if namespace is None:
                source_skills, source_error = sdk_skills._list_skills_with_errors(
                    backend, source_path
                )
                if source_error is not None:
                    errors.append(source_error)
            else:
                source_skills = load_namespaced_skills(backend, source_path, namespace)
            for skill in source_skills:
                merge_skill(
                    all_skills,
                    merged_source_labels,
                    skill,
                    source_label=source_label,
                )

        return self._state_update(all_skills, errors)

    async def abefore_agent(
        self,
        state: sdk_skills.SkillsState,
        runtime: Runtime,  # noqa: ARG002
        config: RunnableConfig,  # noqa: ARG002
    ) -> sdk_skills.SkillsStateUpdate | None:
        """Asynchronously load and namespace skills before collision resolution.

        Returns:
            A state update containing collision-safe skill metadata, or `None`
            when skills are already loaded.
        """
        if "skills_metadata" in state and not state.get("pinned_skills"):
            return None

        backend = self._backend
        all_skills: dict[str, sdk_skills.SkillMetadata] = {}
        merged_source_labels: dict[str, str | None] = {}
        errors: list[str] = []

        # See `before_agent`: the three sequences are index-aligned by
        # construction, and `strict=True` guards against future length drift.
        for source_path, source_label, namespace in zip(
            self.sources, self.source_labels, self._namespaces, strict=True
        ):
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
                merge_skill(
                    all_skills,
                    merged_source_labels,
                    skill,
                    source_label=source_label,
                )

        return self._state_update(all_skills, errors)
