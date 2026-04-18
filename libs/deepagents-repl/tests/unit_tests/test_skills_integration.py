"""End-to-end tests: skill install + dynamic import in the REPL.

Exercises the full pipeline — enumerate a skill on a
``FilesystemBackend`` → build a ``ModuleScope`` → install on a Context
→ ``await import("@/skills/<name>")`` from guest code. Lives next to
the unit tests because it needs no network or model call.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import SkillMetadata

from deepagents_repl._repl import _Registry
from deepagents_repl._skills import SkillScopeInvalid


def _metadata(name: str, *, path: str, module: str | None = None) -> SkillMetadata:
    m: SkillMetadata = {
        "name": name,
        "description": "x",
        "path": path,
        "metadata": {},
        "license": None,
        "compatibility": None,
        "allowed_tools": [],
    }
    if module is not None:
        m["module"] = module
    return m


def _write(backend: FilesystemBackend, files: dict[str, str]) -> None:
    pairs = [(p, c.encode("utf-8")) for p, c in files.items()]
    for r in backend.upload_files(pairs):
        assert r.error is None, f"upload of {r.path} failed: {r.error}"


@pytest.fixture
def registry() -> _Registry:
    reg = _Registry(memory_limit=64 * 1024 * 1024, timeout=5.0, capture_console=True)
    try:
        yield reg
    finally:
        reg.close()


async def test_dynamic_import_roundtrip(registry: _Registry, tmp_path: Path) -> None:
    """Install one skill, then import its export from guest code."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "slugify")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: slugify\ndescription: x\n---\n",
            f"{skill_dir}/index.js": (
                "export function toSlug(s) { return s.toLowerCase().replace(/ /g, '-'); }"
            ),
        },
    )
    meta = _metadata("slugify", path=f"{skill_dir}/SKILL.md", module="index.js")

    repl = registry.get("t1")
    errors = await registry.aensure_skills_installed(
        frozenset({"slugify"}), {"slugify": meta}, backend, repl._ctx
    )
    assert errors == []

    outcome = await repl.eval_async(
        "const m = await import(\"@/skills/slugify\"); globalThis.r = m.toSlug('Hello World');"
    )
    assert outcome.error_type is None
    after = await repl.eval_async("globalThis.r")
    assert after.result == "hello-world"


async def test_dynamic_import_of_ts_skill_strips_types(registry: _Registry, tmp_path: Path) -> None:
    """TS types survive install (are stripped) and the skill works."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "ts-skill")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: ts-skill\ndescription: x\n---\n",
            f"{skill_dir}/index.ts": (
                "export function add(a: number, b: number): number { return a + b; }"
            ),
        },
    )
    meta = _metadata("ts-skill", path=f"{skill_dir}/SKILL.md", module="index.ts")

    repl = registry.get("t1")
    errors = await registry.aensure_skills_installed(
        frozenset({"ts-skill"}), {"ts-skill": meta}, backend, repl._ctx
    )
    assert errors == []

    import_outcome = await repl.eval_async(
        'const m = await import("@/skills/ts-skill"); globalThis.r = m.add(2, 3);'
    )
    assert import_outcome.error_type is None
    after = await repl.eval_async("globalThis.r")
    assert after.result == "5"


async def test_multi_file_skill_relative_import(registry: _Registry, tmp_path: Path) -> None:
    """A skill's entrypoint relative-imports a helper file."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "multi")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: multi\ndescription: x\n---\n",
            f"{skill_dir}/index.ts": (
                'import { value } from "./util.ts";\nexport const doubled = value * 2;\n'
            ),
            f"{skill_dir}/util.ts": "export const value = 7;\n",
        },
    )
    meta = _metadata("multi", path=f"{skill_dir}/SKILL.md", module="index.ts")

    repl = registry.get("t1")
    errors = await registry.aensure_skills_installed(
        frozenset({"multi"}), {"multi": meta}, backend, repl._ctx
    )
    assert errors == []

    import_outcome = await repl.eval_async(
        'const m = await import("@/skills/multi"); globalThis.r = m.doubled;'
    )
    assert import_outcome.error_type is None
    after = await repl.eval_async("globalThis.r")
    assert after.result == "14"


async def test_install_cache_avoids_second_fetch(registry: _Registry, tmp_path: Path) -> None:
    """Second install request for the same skill doesn't re-load."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "cached")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: cached\ndescription: x\n---\n",
            f"{skill_dir}/index.js": "export const k = 1;",
        },
    )
    meta = _metadata("cached", path=f"{skill_dir}/SKILL.md", module="index.js")

    repl = registry.get("t1")
    await registry.aensure_skills_installed(
        frozenset({"cached"}), {"cached": meta}, backend, repl._ctx
    )
    assert "cached" in registry._skill_installs
    first_loaded = registry._skill_installs["cached"].loaded

    # Second pass — no backend I/O, no new install.
    await registry.aensure_skills_installed(
        frozenset({"cached"}), {"cached": meta}, backend, repl._ctx
    )
    assert registry._skill_installs["cached"].loaded is first_loaded


async def test_installed_skill_visible_from_second_thread(
    registry: _Registry, tmp_path: Path
) -> None:
    """Install is per-Runtime, not per-Context. Second thread's Context
    can import the skill without a new install pass."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "shared")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: shared\ndescription: x\n---\n",
            f"{skill_dir}/index.js": "export const k = 99;",
        },
    )
    meta = _metadata("shared", path=f"{skill_dir}/SKILL.md", module="index.js")

    repl_a = registry.get("thread-a")
    await registry.aensure_skills_installed(
        frozenset({"shared"}), {"shared": meta}, backend, repl_a._ctx
    )

    # Thread B — no new install call — can still import.
    repl_b = registry.get("thread-b")
    outcome = await repl_b.eval_async(
        'const m = await import("@/skills/shared"); globalThis.r = m.k;'
    )
    assert outcome.error_type is None
    after = await repl_b.eval_async("globalThis.r")
    assert after.result == "99"


async def test_unavailable_skill_returns_error(registry: _Registry, tmp_path: Path) -> None:
    """Referencing a skill that has no metadata entry surfaces as an error."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    repl = registry.get("t1")

    errors = await registry.aensure_skills_installed(frozenset({"nope"}), {}, backend, repl._ctx)
    assert len(errors) == 1
    assert "nope" in str(errors[0])
    assert "not available" in str(errors[0])


async def test_broken_skill_failure_is_cached(registry: _Registry, tmp_path: Path) -> None:
    """A failing install is cached so subsequent references fail fast."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "broken")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: broken\ndescription: x\n---\n",
            # `module` points at a file we never created.
        },
    )
    meta = _metadata("broken", path=f"{skill_dir}/SKILL.md", module="missing.ts")

    repl = registry.get("t1")
    errors1 = await registry.aensure_skills_installed(
        frozenset({"broken"}), {"broken": meta}, backend, repl._ctx
    )
    assert len(errors1) == 1
    assert isinstance(errors1[0], SkillScopeInvalid)

    # Same error comes back on second call — from the cache, not a new
    # backend load. We can't directly assert "no I/O happened", but we
    # can confirm the cache entry is populated.
    errors2 = await registry.aensure_skills_installed(
        frozenset({"broken"}), {"broken": meta}, backend, repl._ctx
    )
    assert len(errors2) == 1
    cached = registry._skill_installs["broken"]
    assert cached.error is not None
    assert cached.loaded is None


async def test_unknown_specifier_rejects_at_import(registry: _Registry, tmp_path: Path) -> None:
    """If a skill specifier wasn't installed, dynamic import rejects."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    # Empty: no skills installed.
    await registry.aensure_skills_installed(frozenset(), {}, backend, registry.get("t1")._ctx)
    repl = registry.get("t1")
    outcome = await repl.eval_async('await import("@/skills/nonexistent")')
    assert outcome.error_type is not None
