"""Unit tests for skill-name collision (override) debug logging."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from deepagents.backends.filesystem import FilesystemBackend

from deepagents_code.plugins.adapters.skills_middleware import PluginSkillsMiddleware

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents.middleware import skills as sdk_skills
    from langgraph.runtime import Runtime


_MERGE_LOGGER = "deepagents_code.skills.merge"

# The middleware entry points ignore `runtime`; only `state` drives the guard.
_RUNTIME = cast("Runtime", None)


def _create_skill(skill_dir: Path, name: str, description: str) -> None:
    """Create a minimal skill directory with a valid `SKILL.md`."""
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: {description}
---
Content
""")


class TestMergeSkillHelper:
    """Directly exercise `merge_skill`."""


class TestListSkillsCollisionLogging:
    """Exercise collision logging through the CLI `list_skills` discovery path."""


class TestMiddlewareCollisionLogging:
    """Exercise collision logging through `PluginSkillsMiddleware` (sync + async).

    These lock in the new three-way `zip(self.sources, self.source_labels,
    self._namespaces, ...)` wiring: both entry points must merge through
    `merge_skill` and log overrides identically.
    """

    @staticmethod
    def _middleware(user_dir: Path, project_dir: Path) -> PluginSkillsMiddleware:
        """Build a middleware over two colliding, non-namespaced sources."""
        _create_skill(user_dir / "review", "review", "User review")
        _create_skill(project_dir / "review", "review", "Project review")
        return PluginSkillsMiddleware(
            backend=FilesystemBackend(virtual_mode=False),
            sources=[(str(user_dir), "User"), (str(project_dir), "Project")],
            system_prompt=None,
        )

    @staticmethod
    def _namespaced_middleware(dir_a: Path, dir_b: Path) -> PluginSkillsMiddleware:
        """Build a middleware over two colliding plugin (namespaced) sources.

        Both sources share the `myplugin` namespace and a `review` skill, so
        both qualify to `myplugin:review` and drive the namespaced (`else`)
        loop branch through `load_namespaced_skills` — the branch a plain-source
        collision never reaches.
        """
        _create_skill(dir_a / "review", "review", "Plugin A review")
        _create_skill(dir_b / "review", "review", "Plugin B review")
        return PluginSkillsMiddleware(
            backend=FilesystemBackend(virtual_mode=False),
            sources=[
                (str(dir_a), "Plugin A", "myplugin"),
                (str(dir_b), "Plugin B", "myplugin"),
            ],
            system_prompt=None,
        )


class TestMiddlewareReloadGuard:
    """Lock the adapter's reload guard to the SDK's `skills_metadata is None` contract.

    `PluginSkillsMiddleware` overrides `before_model`/`abefore_model`, so it does
    not inherit the SDK's guard. A membership check (`"skills_metadata" in state`)
    would treat an explicit `None` as loaded and silently skip the reload.
    """

    @staticmethod
    def _middleware(skills_dir: Path) -> PluginSkillsMiddleware:
        """Build a middleware over one plain source holding a single skill."""
        _create_skill(skills_dir / "review", "review", "User review")
        return PluginSkillsMiddleware(
            backend=FilesystemBackend(virtual_mode=False),
            sources=[(str(skills_dir), "User")],
            system_prompt=None,
        )

    @staticmethod
    def _state(**overrides: Any) -> sdk_skills.SkillsState:
        """Build a minimal middleware state with the given skills keys."""
        return cast("sdk_skills.SkillsState", {"messages": [], **overrides})

    def test_before_model_reloads_when_metadata_is_none(self, tmp_path: Path) -> None:
        """A stored `None` means not loaded, so the sync entry point loads."""
        middleware = self._middleware(tmp_path / "skills")

        result = middleware.before_model(
            self._state(skills_metadata=None), _RUNTIME, {}
        )

        assert result is not None
        assert [skill["name"] for skill in result["skills_metadata"]] == ["review"]

    def test_before_model_skips_when_metadata_is_a_list(self, tmp_path: Path) -> None:
        """An existing list (even empty) means loaded, so the load is skipped."""
        middleware = self._middleware(tmp_path / "skills")

        assert (
            middleware.before_model(self._state(skills_metadata=[]), _RUNTIME, {})
            is None
        )

    def test_before_model_clears_stale_load_errors(self, tmp_path: Path) -> None:
        """Every load rewrites `skills_load_errors`, clearing earlier warnings."""
        middleware = self._middleware(tmp_path / "skills")
        state = self._state(
            skills_metadata=None,
            skills_load_errors=["Cannot load skills from '/old': denied"],
        )

        result = middleware.before_model(state, _RUNTIME, {})

        assert result is not None
        assert result["skills_load_errors"] == []

    async def test_abefore_model_reloads_when_metadata_is_none(
        self, tmp_path: Path
    ) -> None:
        """The async entry point must honour the same reset contract."""
        middleware = self._middleware(tmp_path / "skills")

        result = await middleware.abefore_model(
            self._state(skills_metadata=None), _RUNTIME, {}
        )

        assert result is not None
        assert [skill["name"] for skill in result["skills_metadata"]] == ["review"]

    async def test_abefore_model_skips_when_metadata_is_a_list(
        self, tmp_path: Path
    ) -> None:
        """An existing list (even empty) means loaded, so the async load is skipped."""
        middleware = self._middleware(tmp_path / "skills")

        result = await middleware.abefore_model(
            self._state(skills_metadata=[]), _RUNTIME, {}
        )

        assert result is None
