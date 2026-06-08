"""`glob(...)` baseline extension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_quickjs._baseline.shared import (
    error,
    normalize_abs_path,
    raise_backend_error,
    reject_unknown_keys,
    require_backend,
    require_dict,
    require_str,
)

if TYPE_CHECKING:
    from langchain_quickjs._extensions import ExtensionContext

GLOB_SYSTEM_PROMPT = """Baseline helper available:

- `await glob(pattern, { cwd? })`
- Returns matched paths.
"""


@dataclass(frozen=True)
class GlobExtension:
    """Expose top-level `glob(pattern, { cwd? })`."""

    system_prompt: str | None = GLOB_SYSTEM_PROMPT
    exported_globals: tuple[str, ...] = ("glob",)

    def on_setup(self, ctx: ExtensionContext) -> None:
        backend = ctx.backend

        @ctx.function(name="__ciGlob")
        async def _glob(payload: Any = None) -> list[str]:
            args = require_dict(payload, "glob")
            reject_unknown_keys(args, allowed={"pattern", "cwd"}, call="glob")
            pattern = require_str(args, "pattern", "glob")
            cwd = args.get("cwd", "/")
            if not isinstance(cwd, str):
                error("ERR_INVALID_ARG_TYPE", "glob `cwd` must be a string")
            resolved_cwd = normalize_abs_path(cwd)
            fs_backend = require_backend(backend, surface="glob")
            result = await fs_backend.aglob(pattern, resolved_cwd)
            if result.error:
                raise_backend_error(result.error, fallback_code="ENOTSUP")
            paths: list[str] = []
            for entry in result.matches or []:
                value = entry.get("path")
                if isinstance(value, str):
                    paths.append(value)
            paths.sort()
            return paths

        ctx.eval(
            "globalThis.glob = async function(pattern, options) {"
            " const opts = options ?? {};"
            " return __ciGlob({ pattern, ...opts });"
            "};"
            "undefined"
        )
