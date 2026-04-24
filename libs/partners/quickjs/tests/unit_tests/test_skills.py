"""Unit tests for the skill-module loader.

Covers the pure loader pipeline (enumerate → download → scope) and the
pre-eval specifier scan. Install-cache behaviour and the end-to-end
REPL wiring are covered in test_repl_middleware.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import SkillMetadata
from quickjs_rs import ModuleScope

from langchain_quickjs._skills import (
    SkillInstallError,
    SkillScopeInvalid,
    aload_skill,
    load_skill,
    scan_skill_references,
)


def _metadata(
    name: str,
    *,
    path: str,
    module: str | None = None,
    description: str = "x",
) -> SkillMetadata:
    m: SkillMetadata = {
        "name": name,
        "description": description,
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
    pairs = [(path, content.encode("utf-8")) for path, content in files.items()]
    responses = backend.upload_files(pairs)
    for r in responses:
        assert r.error is None, f"upload of {r.path} failed: {r.error}"


# ---- load_skill ---------------------------------------------------


def test_load_skill_single_file(tmp_path: Path) -> None:
    """A minimal skill with one index.js at the root."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "slugify")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: slugify\ndescription: x\n---\n",
            f"{skill_dir}/index.js": "export function toSlug(s) { return s; }",
        },
    )
    meta = _metadata("slugify", path=f"{skill_dir}/SKILL.md", module="index.js")

    loaded = load_skill(meta, backend)

    assert loaded.name == "slugify"
    assert loaded.specifier == "@/skills/slugify"
    assert isinstance(loaded.scope, ModuleScope)
    # Only the one source file installed at the canonical key.
    assert dict(loaded.scope.modules) == {
        "index.js": "export function toSlug(s) { return s; }",
    }


def test_load_skill_multi_file_with_subdir(tmp_path: Path) -> None:
    """Entrypoint + helper + nested helper all land in the scope with POSIX keys."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "pdf-extract")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: pdf-extract\ndescription: x\n---\n",
            f"{skill_dir}/index.ts": 'import { parsePdf } from "./parser.ts";\n',
            f"{skill_dir}/parser.ts": 'import { decode } from "./lib/utf.ts";\n',
            f"{skill_dir}/lib/utf.ts": "export function decode() {}",
        },
    )
    meta = _metadata("pdf-extract", path=f"{skill_dir}/SKILL.md", module="index.ts")

    loaded = load_skill(meta, backend)

    assert set(loaded.scope.modules.keys()) == {
        "index.ts",
        "parser.ts",
        "lib/utf.ts",
    }


def test_load_skill_renames_non_index_entrypoint(tmp_path: Path) -> None:
    """If the author's entrypoint isn't index.<ext>, install it under index.<ext>
    AND keep it at its original key so relative imports still resolve."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "custom")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: custom\ndescription: x\n---\n",
            f"{skill_dir}/entry.ts": "export const x = 1;",
            f"{skill_dir}/helper.ts": "export const y = 2;",
        },
    )
    meta = _metadata("custom", path=f"{skill_dir}/SKILL.md", module="entry.ts")

    loaded = load_skill(meta, backend)

    # index.ts is the canonical entrypoint key; entry.ts stays as a
    # sibling so anything importing "./entry.ts" still works.
    assert set(loaded.scope.modules.keys()) == {"index.ts", "entry.ts", "helper.ts"}
    assert loaded.scope.modules["index.ts"] == loaded.scope.modules["entry.ts"]


def test_load_skill_ignores_non_code_files(tmp_path: Path) -> None:
    """SKILL.md, JSON data files, etc. are enumerated but not installed."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "s")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: s\ndescription: x\n---\n",
            f"{skill_dir}/index.js": "export const k = 1;",
            f"{skill_dir}/data.json": '{"k": 1}',
            f"{skill_dir}/README.md": "# docs",
        },
    )
    meta = _metadata("s", path=f"{skill_dir}/SKILL.md", module="index.js")

    loaded = load_skill(meta, backend)

    # Only the code file — data/markdown filtered at enumeration time.
    assert set(loaded.scope.modules.keys()) == {"index.js"}


def test_load_skill_missing_module_raises(tmp_path: Path) -> None:
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "prose")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: prose\ndescription: x\n---\n",
        },
    )
    meta = _metadata("prose", path=f"{skill_dir}/SKILL.md")  # no module

    with pytest.raises(SkillScopeInvalid, match="no `module` frontmatter"):
        load_skill(meta, backend)


def test_load_skill_module_path_not_in_dir_raises(tmp_path: Path) -> None:
    """Frontmatter points at a file that doesn't exist — clean error."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "typo")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: typo\ndescription: x\n---\n",
            f"{skill_dir}/index.ts": "export const x = 1;",
        },
    )
    meta = _metadata("typo", path=f"{skill_dir}/SKILL.md", module="entry.ts")

    with pytest.raises(SkillScopeInvalid, match="did not match any file"):
        load_skill(meta, backend)


def test_load_skill_empty_dir_raises(tmp_path: Path) -> None:
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "empty")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: empty\ndescription: x\n---\n",
        },
    )
    meta = _metadata("empty", path=f"{skill_dir}/SKILL.md", module="index.js")

    with pytest.raises(SkillScopeInvalid, match="no JS/TS files"):
        load_skill(meta, backend)


def test_load_skill_invalid_name_rejected(tmp_path: Path) -> None:
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "bad name")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: 'bad name'\ndescription: x\n---\n",
            f"{skill_dir}/index.ts": "export const x = 1;",
        },
    )
    meta = _metadata("bad name", path=f"{skill_dir}/SKILL.md", module="index.ts")

    with pytest.raises(SkillScopeInvalid, match="kebab-case"):
        load_skill(meta, backend)


def test_load_skill_preserves_utf8_content(tmp_path: Path) -> None:
    """Non-ASCII source survives the bytes → str round trip."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "emoji")
    source = 'export const greet = "hello 🎉";'
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: emoji\ndescription: x\n---\n",
            f"{skill_dir}/index.js": source,
        },
    )
    meta = _metadata("emoji", path=f"{skill_dir}/SKILL.md", module="index.js")

    loaded = load_skill(meta, backend)

    assert loaded.scope.modules["index.js"] == source


def test_load_skill_rejects_non_utf8(tmp_path: Path) -> None:
    """A non-UTF-8 file on disk must surface as a clean install error."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "binary")
    # Write the SKILL.md normally, but stash non-UTF-8 bytes in a .js
    # file directly through the filesystem — bypassing upload_files,
    # which would re-encode.
    Path(skill_dir).mkdir(parents=True)
    (Path(skill_dir) / "SKILL.md").write_bytes(
        b"---\nname: binary\ndescription: x\n---\n"
    )
    (Path(skill_dir) / "index.js").write_bytes(b"\x80\x81\x82")
    meta = _metadata("binary", path=f"{skill_dir}/SKILL.md", module="index.js")

    with pytest.raises(SkillInstallError, match="not valid UTF-8"):
        load_skill(meta, backend)


# ---- aload_skill --------------------------------------------------


async def test_aload_skill_matches_sync(tmp_path: Path) -> None:
    """Async path produces the same result as sync."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "dual")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: dual\ndescription: x\n---\n",
            f"{skill_dir}/index.ts": "export const k = 1;",
            f"{skill_dir}/util.ts": "export const u = 2;",
        },
    )
    meta = _metadata("dual", path=f"{skill_dir}/SKILL.md", module="index.ts")

    sync_loaded = load_skill(meta, backend)
    async_loaded = await aload_skill(meta, backend)

    assert sync_loaded.name == async_loaded.name
    assert sync_loaded.specifier == async_loaded.specifier
    assert dict(sync_loaded.scope.modules) == dict(async_loaded.scope.modules)


# ---- scan_skill_references ----------------------------------------


def test_scan_static_import() -> None:
    source = 'import { parse } from "@/skills/pdf-extract";'
    assert scan_skill_references(source) == frozenset({"pdf-extract"})


def test_scan_dynamic_import() -> None:
    source = 'const m = await import("@/skills/slugify");'
    assert scan_skill_references(source) == frozenset({"slugify"})


def test_scan_mixed() -> None:
    source = """
    import { a } from "@/skills/one";
    const m = await import('@/skills/two');
    console.log(a);
    """
    assert scan_skill_references(source) == frozenset({"one", "two"})


def test_scan_dedups() -> None:
    """Multiple references to the same skill collapse to one entry."""
    source = """
    import a from "@/skills/x";
    const b = await import("@/skills/x");
    """
    assert scan_skill_references(source) == frozenset({"x"})


def test_scan_ignores_non_matches() -> None:
    """Similar-looking but wrong specifiers don't trigger."""
    source = """
    import x from "@/other/foo";
    import y from "skills/foo";
    import z from "@/skills/";  // empty name
    import q from "@/skills/Foo";  // uppercase — not kebab
    """
    assert scan_skill_references(source) == frozenset()


def test_scan_ignores_template_literal() -> None:
    """Template literals are out of scope — can't be statically resolved."""
    source = "const m = await import(`@/skills/${name}`);"
    assert scan_skill_references(source) == frozenset()


def test_scan_both_quote_styles() -> None:
    """Single and double quotes both parse."""
    source = """
    import a from "@/skills/one";
    import b from '@/skills/two';
    """
    assert scan_skill_references(source) == frozenset({"one", "two"})


# ---- sanity: scope is installable ---------------------------------


def test_loaded_scope_installs_on_context(tmp_path: Path) -> None:
    """The scope we build is acceptable to `ctx.install`.

    Catches shape bugs (e.g. illegal keys) before they reach the full
    REPL wiring. Does not exercise import() — that's in the integration
    tests.
    """
    from quickjs_rs import Runtime

    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill_dir = str(tmp_path / "skills" / "hello")
    _write(
        backend,
        {
            f"{skill_dir}/SKILL.md": "---\nname: hello\ndescription: x\n---\n",
            f"{skill_dir}/index.js": "export const n = 42;",
        },
    )
    meta = _metadata("hello", path=f"{skill_dir}/SKILL.md", module="index.js")
    loaded = load_skill(meta, backend)

    with Runtime() as rt, rt.new_context():
        rt.install(ModuleScope({loaded.specifier: loaded.scope}))


__all__: list[Any] = []
