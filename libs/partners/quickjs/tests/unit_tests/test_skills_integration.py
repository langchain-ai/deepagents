"""End-to-end tests: skill install + dynamic import in the REPL.

Exercises the full pipeline — enumerate a skill on a
``FilesystemBackend`` → build a ``ModuleScope`` → install on a Context
→ ``await import("@/skills/<name>")`` from guest code. Lives next to
the unit tests because it needs no network or model call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from deepagents.backends.filesystem import FilesystemBackend

from langchain_quickjs._repl import _Registry

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents.middleware.skills import SkillMetadata


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


def _cache_key(meta: SkillMetadata) -> tuple[str, str, str | None]:
    return (meta["name"], meta["path"], meta.get("module"))


@pytest.fixture
def registry() -> _Registry:
    reg = _Registry(memory_limit=64 * 1024 * 1024, timeout=5.0, capture_console=True)
    try:
        yield reg
    finally:
        reg.close()


async def test_dynamic_import_roundtrip(registry: _Registry, tmp_path: Path) -> None:
    """Eval installs referenced skills, then import resolves in guest code."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "slugify")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: slugify\ndescription: x\n---\n",
            f"{skill_dir}/index.js": (
                "export function toSlug(s) {\n"
                "  return s.toLowerCase().replace(/ /g, '-');\n"
                "}"
            ),
        },
    )
    meta = _metadata("slugify", path=f"{skill_dir}/SKILL.md", module="index.js")

    repl = registry.get("t1")
    outcome = await repl.eval_async(
        'const m = await import("@/skills/slugify");\n'
        "globalThis.r = m.toSlug('Hello World');",
        skills={"slugify": meta},
        skills_backend=backend,
    )
    assert outcome.error_type is None
    after = await repl.eval_async("globalThis.r")
    assert after.result == "hello-world"


def test_sync_dynamic_import_roundtrip(registry: _Registry, tmp_path: Path) -> None:
    """Sync eval auto-installs referenced skills when metadata/backend are passed."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "sync-skill")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: sync-skill\ndescription: x\n---\n",
            f"{skill_dir}/index.js": "export const v = 11;",
        },
    )
    meta = _metadata("sync-skill", path=f"{skill_dir}/SKILL.md", module="index.js")

    repl = registry.get("t1")
    import_outcome = repl.eval_sync(
        'const m = await import("@/skills/sync-skill"); globalThis.r = m.v;',
        skills={"sync-skill": meta},
        skills_backend=backend,
    )
    assert import_outcome.error_type is None
    after = repl.eval_sync("globalThis.r")
    assert after.result == "11"


async def test_dynamic_import_of_ts_skill_strips_types(
    registry: _Registry, tmp_path: Path
) -> None:
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
    import_outcome = await repl.eval_async(
        'const m = await import("@/skills/ts-skill"); globalThis.r = m.add(2, 3);',
        skills={"ts-skill": meta},
        skills_backend=backend,
    )
    assert import_outcome.error_type is None
    after = await repl.eval_async("globalThis.r")
    assert after.result == "5"


async def test_multi_file_skill_relative_import(
    registry: _Registry, tmp_path: Path
) -> None:
    """A skill's entrypoint relative-imports a helper file."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "multi")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: multi\ndescription: x\n---\n",
            f"{skill_dir}/index.ts": (
                'import { value } from "./util.ts";\n'
                "export const doubled = value * 2;\n"
            ),
            f"{skill_dir}/util.ts": "export const value = 7;\n",
        },
    )
    meta = _metadata("multi", path=f"{skill_dir}/SKILL.md", module="index.ts")

    repl = registry.get("t1")
    import_outcome = await repl.eval_async(
        'const m = await import("@/skills/multi"); globalThis.r = m.doubled;',
        skills={"multi": meta},
        skills_backend=backend,
    )
    assert import_outcome.error_type is None
    after = await repl.eval_async("globalThis.r")
    assert after.result == "14"


async def test_install_cache_avoids_second_fetch(
    registry: _Registry, tmp_path: Path
) -> None:
    """Second install request in one thread skips already-installed keys."""
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
    await repl.eval_async(
        'await import("@/skills/cached")',
        skills={"cached": meta},
        skills_backend=backend,
    )
    key = _cache_key(meta)
    assert key in repl._installed_skills
    first_count = len(repl._installed_skills)

    # Second pass on the same REPL should hit local cache.
    await repl.eval_async(
        'await import("@/skills/cached")',
        skills={"cached": meta},
        skills_backend=backend,
    )
    assert key in repl._installed_skills
    assert len(repl._installed_skills) == first_count


async def test_slot_skill_cache_is_cleared_on_slot_eviction(tmp_path: Path) -> None:
    """TTL eviction clears slot-local skill cache for that thread."""
    reg = _Registry(
        memory_limit=32 * 1024 * 1024,
        timeout=5.0,
        capture_console=True,
        idle_ttl_sec=60.0,
    )
    try:
        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        skill_dir = str(tmp_path / "skills" / "persist")
        _write(
            backend,
            {
                f"{skill_dir}/SKILL.md": "---\nname: persist\ndescription: x\n---\n",
                f"{skill_dir}/index.js": "export const k = 7;",
            },
        )
        meta = _metadata("persist", path=f"{skill_dir}/SKILL.md", module="index.js")

        repl_a = reg.get("t1")
        await repl_a.eval_async(
            'await import("@/skills/persist")',
            skills={"persist": meta},
            skills_backend=backend,
        )
        key = _cache_key(meta)
        assert key in repl_a._installed_skills

        # Evict t1 (simulate long idle), then return — fresh slot.
        reg._slots["t1"].last_used -= 120.0
        repl_a_new = reg.get("t1")
        assert repl_a_new is not repl_a
        assert key not in repl_a_new._installed_skills

        # Re-install on the fresh slot and verify import still resolves.
        await repl_a_new.eval_async(
            'await import("@/skills/persist")',
            skills={"persist": meta},
            skills_backend=backend,
        )
        assert key in repl_a_new._installed_skills
        outcome = await repl_a_new.eval_async(
            'const m = await import("@/skills/persist"); m.k'
        )
        assert outcome.error_type is None
        assert outcome.result == "7"
    finally:
        reg.close()


async def test_skill_cache_isolated_across_threads(
    registry: _Registry, tmp_path: Path
) -> None:
    """Same skill in two threads builds separate slot-local cache entries."""
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
    await repl_a.eval_async(
        'await import("@/skills/shared")',
        skills={"shared": meta},
        skills_backend=backend,
    )
    key = _cache_key(meta)
    assert key in repl_a._installed_skills

    repl_b = registry.get("thread-b")
    # Thread-b installs independently and gets its own cache entry.
    await repl_b.eval_async(
        'await import("@/skills/shared")',
        skills={"shared": meta},
        skills_backend=backend,
    )
    assert key in repl_b._installed_skills

    outcome = await repl_b.eval_async(
        'const m = await import("@/skills/shared"); globalThis.r = m.k;'
    )
    assert outcome.error_type is None
    after = await repl_b.eval_async("globalThis.r")
    assert after.result == "99"


async def test_eval_reports_skill_not_available_when_metadata_missing(
    registry: _Registry, tmp_path: Path
) -> None:
    """Implicit install path surfaces missing metadata as ``SkillNotAvailable``."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    repl = registry.get("t1")

    outcome = await repl.eval_async(
        'await import("@/skills/nope")',
        skills={},
        skills_backend=backend,
    )
    assert outcome.error_type == "SkillNotAvailable"
    assert "nope" in outcome.error_message
    assert "not available" in outcome.error_message


async def test_broken_skill_failure_is_not_tracked_as_installed(
    registry: _Registry, tmp_path: Path
) -> None:
    """A failing install does not mark the skill as installed."""
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
    first = await repl.eval_async(
        'await import("@/skills/broken")',
        skills={"broken": meta},
        skills_backend=backend,
    )
    assert first.error_type == "SkillNotAvailable"
    assert "no JS/TS files" in first.error_message

    # Second call still fails and the key remains uninstalled.
    second = await repl.eval_async(
        'await import("@/skills/broken")',
        skills={"broken": meta},
        skills_backend=backend,
    )
    assert second.error_type == "SkillNotAvailable"
    assert _cache_key(meta) not in repl._installed_skills


async def test_unknown_specifier_rejects_at_import(
    registry: _Registry, tmp_path: Path
) -> None:
    """If a skill specifier wasn't installed, dynamic import rejects."""
    repl = registry.get("t1")
    outcome = await repl.eval_async('await import("@/skills/nonexistent")')
    assert outcome.error_type is not None
