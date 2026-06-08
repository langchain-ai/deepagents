"""`editFile(...)` baseline extension."""

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

EDIT_FILE_SYSTEM_PROMPT = """Baseline helper available:

- `await editFile({ filePath, oldString, newString, replaceAll? })`
- Performs exact string replacement on an existing file.
"""


@dataclass(frozen=True)
class EditFileExtension:
    """Expose top-level `editFile({ filePath, oldString, newString, replaceAll? })`."""

    system_prompt: str | None = EDIT_FILE_SYSTEM_PROMPT
    exported_globals: tuple[str, ...] = ("editFile",)

    def on_setup(self, ctx: ExtensionContext) -> None:
        backend = ctx.backend

        @ctx.function(name="__ciEditFile")
        async def _edit_file(payload: Any = None) -> str:
            args = require_dict(payload, "editFile")
            reject_unknown_keys(
                args,
                allowed={"filePath", "oldString", "newString", "replaceAll"},
                call="editFile",
            )
            path = normalize_abs_path(require_str(args, "filePath", "editFile"))
            old = require_str(args, "oldString", "editFile")
            new = require_str(args, "newString", "editFile")
            replace_all = args.get("replaceAll", False)
            if not isinstance(replace_all, bool):
                error("ERR_INVALID_ARG_TYPE", "editFile `replaceAll` must be boolean")

            fs_backend = require_backend(backend, surface="editFile")
            result = await fs_backend.aedit(path, old, new, replace_all)
            if result.error:
                raise_backend_error(result.error)
            return result.path or path

        ctx.eval(
            "globalThis.editFile = async function(options) {"
            " return __ciEditFile(options ?? {});"
            "};"
            "undefined"
        )
