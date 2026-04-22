"""Skill module loader for the REPL.

Turns a skill's ``SkillMetadata`` (parsed by ``SkillsMiddleware``) plus a
``BackendProtocol`` into a ``ModuleScope`` installable on a QuickJS
context under the bare specifier ``@/skills/<name>``. The guest can
then ``await import("@/skills/<name>")`` to pull the skill's entrypoint
exports.

This module is the enumeration + scope-build half; the install-cache
and the pre-eval specifier scan live on ``_ThreadREPL`` in
``_repl.py`` so they share the Context / Runtime locks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from quickjs_rs import ModuleScope

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware.skills import SkillMetadata

logger = logging.getLogger(__name__)

# Extensions quickjs-rs's resolver + oxidase TS-strip understand. Order
# matters for the `index.<ext>` picker in quickjs-rs
# (`src/modules.rs:296-308`) — we mirror the same preference list so the
# author's choice of `index.ts` vs `index.js` produces predictable
# resolution.
SKILL_MODULE_EXTENSIONS: tuple[str, ...] = (
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".mts",
    ".cts",
    ".jsx",
    ".tsx",
)

# Hard cap on how many bytes we'll pull for one skill's bundle. A skill
# dir with a gigabyte of generated code would otherwise block the event
# loop during install and blow through the context's memory limit. 1 MiB
# is generous for hand-written skills and small enough that a runaway
# generated dir trips this check instead of the OOM killer.
_MAX_BUNDLE_BYTES = 1 * 1024 * 1024

# Identifier rule for a skill's bare-specifier install key. Matches
# Agent Skills spec's kebab-case name constraint. We re-validate here
# (rather than trust the parsed metadata) so a future loader-side
# concern — guarding specifier injection into the `ModuleScope` dict —
# stays local. See `_skill_specifier`.
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillLoadError(Exception):
    """Base class for skill-load failures surfaced at install time."""


class SkillScopeInvalid(SkillLoadError):
    """Skill directory contains nothing installable.

    Either no module-extension files were found, or the frontmatter
    ``module`` path doesn't match any of them.
    """


class SkillInstallError(SkillLoadError):
    """Backend fetch failed or produced unreadable content for a skill."""


@dataclass(frozen=True)
class LoadedSkill:
    """A skill's install-ready state.

    Attributes:
        name: Spec-validated skill name (kebab-case).
        specifier: The bare specifier we install under. Always
            ``"@/skills/<name>"``.
        scope: A ``ModuleScope`` carrying every code file from the
            skill directory, with the entrypoint renamed to
            ``index.<ext>`` if the author picked a different name.
    """

    name: str
    specifier: str
    scope: ModuleScope


def _skill_specifier(name: str) -> str:
    """Return the bare specifier a skill installs under.

    Raises:
        SkillScopeInvalid: If the skill name is not a spec-valid
            kebab-case identifier. Guards against a malformed name
            silently becoming a weird install key.
    """
    if not _SKILL_NAME_RE.match(name):
        msg = f"skill name {name!r} is not a valid kebab-case identifier"
        raise SkillScopeInvalid(msg)
    return f"@/skills/{name}"


def _skill_dir_from_metadata(metadata: SkillMetadata) -> str:
    """Extract the skill directory from a ``SkillMetadata``.

    ``path`` is the SKILL.md path; the skill dir is its parent.
    """
    return str(PurePosixPath(metadata["path"]).parent)


def _enumerate_code_files(
    backend: BackendProtocol,
    skill_dir: str,
) -> list[str]:
    """List every code-extension file under ``skill_dir`` (recursive).

    Makes one ``glob`` call per extension. We'd prefer one call with a
    brace-expansion pattern, but the backend protocol doesn't require
    brace support and ``FilesystemBackend.glob`` uses ``pathlib.rglob``
    which doesn't expand them. Per-extension calls sidestep that cross-
    backend inconsistency. Paths are deduped (a single file can only
    match one extension) and sorted for determinism so the model's
    system prompt never flakes based on filesystem order.
    """
    seen: set[str] = set()
    for ext in SKILL_MODULE_EXTENSIONS:
        pattern = f"**/*{ext}"
        result = backend.glob(pattern, skill_dir)
        if result.error:
            msg = f"failed to list skill dir {skill_dir}: {result.error}"
            raise SkillInstallError(msg)
        for match in result.matches or []:
            seen.add(match["path"])
    return sorted(seen)


async def _aenumerate_code_files(
    backend: BackendProtocol,
    skill_dir: str,
) -> list[str]:
    """Async sibling of `_enumerate_code_files`."""
    seen: set[str] = set()
    for ext in SKILL_MODULE_EXTENSIONS:
        pattern = f"**/*{ext}"
        result = await backend.aglob(pattern, skill_dir)
        if result.error:
            msg = f"failed to list skill dir {skill_dir}: {result.error}"
            raise SkillInstallError(msg)
        for match in result.matches or []:
            seen.add(match["path"])
    return sorted(seen)


def _relative(skill_dir: str, absolute_path: str) -> str:
    """Return `absolute_path` expressed relative to `skill_dir`.

    Both arguments are POSIX paths. The result has no leading `/` and
    uses `/` separators. A path that doesn't start with `skill_dir` is
    a backend bug; we raise rather than silently produce a broken key.
    """
    skill_root = PurePosixPath(skill_dir)
    abs_path = PurePosixPath(absolute_path)
    try:
        rel = abs_path.relative_to(skill_root)
    except ValueError as exc:
        msg = f"file {absolute_path!r} is not under skill dir {skill_dir!r}"
        raise SkillInstallError(msg) from exc
    return str(rel)


def _pick_installed_index_name(entry_rel: str) -> str:
    """Pick the `index.<ext>` key we install the entrypoint under.

    If the author already named their entrypoint `index.<ext>` at the
    skill dir root, we keep the name. Otherwise we rewrite to
    `index.<ext>` using the author's extension so quickjs-rs's
    subscope resolver auto-picks it for bare-specifier imports.

    Args:
        entry_rel: Entrypoint path relative to the skill dir
            (e.g. `"index.ts"` or `"src/entry.ts"`).

    Returns:
        The key we install this file under inside the scope. Always
        `index.<ext>` at the top level — a non-root entrypoint is
        flattened, which means any other file at the same depth or
        deeper is preserved so sibling relative imports still work.
    """
    entry_path = PurePosixPath(entry_rel)
    ext = entry_path.suffix
    return f"index{ext}"


def _validate_bundle_size(
    paths_and_contents: list[tuple[str, bytes]],
    skill_name: str,
) -> None:
    total = sum(len(c) for _, c in paths_and_contents)
    if total > _MAX_BUNDLE_BYTES:
        msg = f"skill {skill_name!r} bundle exceeds {_MAX_BUNDLE_BYTES} bytes (total {total})"
        raise SkillInstallError(msg)


def _build_scope_modules(
    skill_dir: str,
    entry_rel: str,
    file_pairs: list[tuple[str, bytes]],
    skill_name: str,
) -> dict[str, str | ModuleScope]:
    """Build the ``ModuleScope`` contents dict for one skill.

    The dict has only `str` entries — no nested subscopes. That's the
    isolation guarantee from the spec: a skill scope can see only its
    own files, so it cannot bare-import other skills.

    If the entrypoint isn't already `index.<ext>` at the skill dir
    root, we install its source under the canonical `index.<ext>` key
    so `@/skills/<name>` bare-specifier imports auto-resolve.
    """
    files: dict[str, str | ModuleScope] = {}
    installed_index = _pick_installed_index_name(entry_rel)
    entry_installed = False

    for abs_path, raw in file_pairs:
        rel = _relative(skill_dir, abs_path)
        try:
            source = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            msg = f"skill {skill_name!r} file {rel!r} is not valid UTF-8"
            raise SkillInstallError(msg) from exc

        if rel == entry_rel:
            # Always install the entrypoint under the canonical
            # `index.<ext>` key. If the author already named it
            # `index.<ext>` at the root, this is a no-op rename.
            files[installed_index] = source
            entry_installed = True
            # Also expose the file at its original key so relative
            # imports (e.g. `import "./index.ts"` from inside a sub-
            # directory) still resolve. Skipped when entry_rel already
            # equals installed_index.
            if rel != installed_index:
                files[rel] = source
        else:
            files[rel] = source

    if not entry_installed:
        msg = (
            f"skill {skill_name!r}: module path {entry_rel!r} "
            "did not match any file in the skill directory"
        )
        raise SkillScopeInvalid(msg)
    return files


def _require_module_path(metadata: SkillMetadata) -> str:
    entry_rel = metadata.get("module")
    if not entry_rel:
        msg = (
            f"skill {metadata['name']!r} has no `module` frontmatter key; "
            "only skills with a declared entrypoint are installable"
        )
        raise SkillScopeInvalid(msg)
    return entry_rel


def load_skill(
    metadata: SkillMetadata,
    backend: BackendProtocol,
) -> LoadedSkill:
    """Load one skill into a ``LoadedSkill``.

    Raises:
        SkillScopeInvalid: Metadata has no `module` key, or the
            entrypoint doesn't match any file in the skill directory.
        SkillInstallError: Backend fetch failed, content was
            non-UTF-8, or the bundle exceeded the size cap.
    """
    name = metadata["name"]
    entry_rel = _require_module_path(metadata)
    specifier = _skill_specifier(name)
    skill_dir = _skill_dir_from_metadata(metadata)

    code_files = _enumerate_code_files(backend, skill_dir)
    if not code_files:
        msg = f"skill {name!r}: no JS/TS files under {skill_dir!r}"
        raise SkillScopeInvalid(msg)

    responses = backend.download_files(code_files)
    file_pairs: list[tuple[str, bytes]] = []
    for resp in responses:
        if resp.error or resp.content is None:
            msg = f"skill {name!r}: failed to download {resp.path!r}: {resp.error}"
            raise SkillInstallError(msg)
        file_pairs.append((resp.path, resp.content))

    _validate_bundle_size(file_pairs, name)
    files = _build_scope_modules(skill_dir, entry_rel, file_pairs, name)
    return LoadedSkill(name=name, specifier=specifier, scope=ModuleScope(files))


async def aload_skill(
    metadata: SkillMetadata,
    backend: BackendProtocol,
) -> LoadedSkill:
    """Async sibling of :func:`load_skill`."""
    name = metadata["name"]
    entry_rel = _require_module_path(metadata)
    specifier = _skill_specifier(name)
    skill_dir = _skill_dir_from_metadata(metadata)

    code_files = await _aenumerate_code_files(backend, skill_dir)
    if not code_files:
        msg = f"skill {name!r}: no JS/TS files under {skill_dir!r}"
        raise SkillScopeInvalid(msg)

    responses = await backend.adownload_files(code_files)
    file_pairs: list[tuple[str, bytes]] = []
    for resp in responses:
        if resp.error or resp.content is None:
            msg = f"skill {name!r}: failed to download {resp.path!r}: {resp.error}"
            raise SkillInstallError(msg)
        file_pairs.append((resp.path, resp.content))

    _validate_bundle_size(file_pairs, name)
    files = _build_scope_modules(skill_dir, entry_rel, file_pairs, name)
    return LoadedSkill(name=name, specifier=specifier, scope=ModuleScope(files))


# ---- pre-eval specifier scan ---------------------------------------

# Matches `"@/skills/<name>"` inside JS source. Bounded to the two
# string-literal forms the model realistically writes — single-quoted
# and double-quoted. Template literals aren't caught; the model can't
# static-import with a template literal anyway, and a dynamic
# `import(`${x}`)` would have a computed value we can't resolve.
_SKILL_SPECIFIER_RE = re.compile(
    r"""["']@/skills/([a-z0-9]+(?:-[a-z0-9]+)*)["']""",
)


def scan_skill_references(source: str) -> frozenset[str]:
    """Return the set of skill names the source imports from.

    Extracts every literal ``"@/skills/<name>"`` specifier the source
    contains. The caller is responsible for rejecting unknown names
    with a ``SkillNotAvailable``-style error — this is a scan, not a
    validator.

    A returned name is not proof the skill exists or installs cleanly.
    Dynamic imports with computed specifiers are not detected; those
    can only succeed if a literal reference elsewhere in the session
    has already triggered install.
    """
    return frozenset(_SKILL_SPECIFIER_RE.findall(source))


__all__ = [
    "LoadedSkill",
    "SKILL_MODULE_EXTENSIONS",
    "SkillInstallError",
    "SkillLoadError",
    "SkillScopeInvalid",
    "aload_skill",
    "load_skill",
    "scan_skill_references",
]
