"""Code-local skills middleware adapter for plugin namespaces."""

from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, cast

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


class PluginSkillsMiddleware(SkillsMiddleware):
    """Load namespaced plugin skills without extending the SDK source API."""

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
    def _direct_skill_path(source_path: str) -> str:
        return str(PurePosixPath(to_posix_path(source_path)) / "SKILL.md")

    @classmethod
    def _load_direct_skill(
        cls,
        backend: BackendProtocol,
        source_path: str,
    ) -> sdk_skills.SkillMetadata | None:
        skill_path = cls._direct_skill_path(source_path)
        response = backend.download_files([skill_path])[0]
        return sdk_skills._skill_metadata_from_response(
            response,
            source_path,
            skill_path,
        )

    @classmethod
    async def _aload_direct_skill(
        cls,
        backend: BackendProtocol,
        source_path: str,
    ) -> sdk_skills.SkillMetadata | None:
        skill_path = cls._direct_skill_path(source_path)
        response = (await backend.adownload_files([skill_path]))[0]
        return sdk_skills._skill_metadata_from_response(
            response,
            source_path,
            skill_path,
        )

    @staticmethod
    def _namespace_skills(
        skills: list[sdk_skills.SkillMetadata],
        namespace: SkillNamespace | None,
    ) -> list[sdk_skills.SkillMetadata]:
        if namespace is None:
            return skills
        return [
            cast(
                "sdk_skills.SkillMetadata",
                {
                    **skill,
                    "name": namespaced_skill_name(namespace, skill["name"]),
                },
            )
            for skill in skills
        ]

    @staticmethod
    def _merge_skills(
        all_skills: dict[str, sdk_skills.SkillMetadata],
        source_skills: list[sdk_skills.SkillMetadata],
        namespace: SkillNamespace | None,
    ) -> None:
        for skill in PluginSkillsMiddleware._namespace_skills(
            source_skills,
            namespace,
        ):
            all_skills[skill["name"]] = skill

    @staticmethod
    def _state_update(
        all_skills: dict[str, sdk_skills.SkillMetadata],
        errors: list[str],
    ) -> sdk_skills.SkillsStateUpdate:
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

        for source_path, namespace in zip(
            self.sources,
            self._namespaces,
            strict=True,
        ):
            source_skills, source_error = sdk_skills._list_skills_with_errors(
                backend,
                source_path,
            )
            if namespace is not None:
                direct_skill = self._load_direct_skill(backend, source_path)
                if direct_skill is not None and all(
                    skill["path"] != direct_skill["path"] for skill in source_skills
                ):
                    source_skills.insert(0, direct_skill)
            if source_error is not None:
                errors.append(source_error)
            self._merge_skills(all_skills, source_skills, namespace)

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

        for source_path, namespace in zip(
            self.sources,
            self._namespaces,
            strict=True,
        ):
            source_skills, source_error = await sdk_skills._alist_skills_with_errors(
                backend, source_path
            )
            if namespace is not None:
                direct_skill = await self._aload_direct_skill(backend, source_path)
                if direct_skill is not None and all(
                    skill["path"] != direct_skill["path"] for skill in source_skills
                ):
                    source_skills.insert(0, direct_skill)
            if source_error is not None:
                errors.append(source_error)
            self._merge_skills(all_skills, source_skills, namespace)

        return self._state_update(all_skills, errors)
