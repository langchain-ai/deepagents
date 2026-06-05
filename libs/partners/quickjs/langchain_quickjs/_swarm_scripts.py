"""Loader for the bundled swarm interpreter scripts.

The TypeScript sources under ``_swarm_scripts/`` are vendored from the
``swarm`` skill in ``langchain-ai/langchain-skills`` (``config/skills/swarm``).
They implement the handle-based table API — ``create`` / ``run`` / ``rows`` —
that the agent imports as ``import { create, run, rows } from "swarm"``.

At runtime the scripts read five functions off ``globalThis.tools``
(``swarmTask``, ``glob``, ``readFile``, ``writeFile``, ``editFile``); the
swarm extension installs those as host functions and assembles the ``tools``
namespace. This module just turns the ``.ts`` files into a ``ModuleScope``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from quickjs_rs import ModuleScope

from langchain_quickjs._skills import _rewrite_js_imports_to_ts

# The bare specifier the guest imports from: `import ... from "swarm"`.
SWARM_SPECIFIER = "swarm"

_SCRIPTS_DIR = Path(__file__).parent / "_swarm_scripts"


@lru_cache(maxsize=1)
def swarm_module_scope() -> ModuleScope:
    """Build the ``ModuleScope`` for the bundled swarm scripts.

    Reads every ``.ts`` file under ``_swarm_scripts/`` into one scope keyed
    by filename (``index.ts`` is the entrypoint a bare ``import "swarm"``
    resolves to). Cross-file imports use the TS ``./name.js`` convention
    while the files are ``.ts``; ``_rewrite_js_imports_to_ts`` rewrites
    those specifiers so quickjs-rs's exact-match resolver finds them.

    Cached: the sources are static package data, identical every call.
    """
    files: dict[str, str | ModuleScope] = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(_SCRIPTS_DIR.glob("*.ts"))
    }
    if "index.ts" not in files:
        msg = "swarm scripts bundle is missing index.ts"
        raise RuntimeError(msg)
    _rewrite_js_imports_to_ts(files)
    return ModuleScope({SWARM_SPECIFIER: ModuleScope(files)})


__all__ = ["SWARM_SPECIFIER", "swarm_module_scope"]
