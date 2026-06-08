"""`fs` baseline extension."""

from __future__ import annotations

import base64
import posixpath
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_quickjs._baseline.shared import (
    READ_LIMIT,
    coerce_file_text,
    error,
    is_missing_error,
    map_backend_error,
    normalize_abs_path,
    raise_backend_error,
    reject_unknown_keys,
    require_backend,
    require_dict,
    require_str,
)

if TYPE_CHECKING:
    from langchain_quickjs._extensions import ExtensionContext

FILESYSTEM_SYSTEM_PROMPT = """Baseline filesystem capability available (`fs`):

- `await fs.readFile(path, { encoding?: "utf8" | "base64" })`
- `await fs.writeFile(path, data, { encoding?: "utf8" | "base64", flag?: "w" | "wx" })`
- `await fs.readdir(path = ".")`
- Relative paths resolve from `/`.
"""


@dataclass(frozen=True)
class FilesystemExtension:
    """Expose Node-like `fs` subset (`readFile`, `writeFile`, `readdir`)."""

    system_prompt: str | None = FILESYSTEM_SYSTEM_PROMPT
    exported_globals: tuple[str, ...] = ("fs",)

    def on_setup(self, ctx: ExtensionContext) -> None:  # noqa: C901, PLR0915
        backend = ctx.backend

        @ctx.function(name="__ciFsReadFile")
        async def _read_file(payload: Any = None) -> str:
            args = require_dict(payload, "fs.readFile")
            reject_unknown_keys(args, allowed={"path", "encoding"}, call="fs.readFile")
            path = normalize_abs_path(require_str(args, "path", "fs.readFile"))
            encoding = args.get("encoding", "utf8")
            if encoding not in {"utf8", "base64"}:
                error(
                    "ERR_INVALID_ENCODING",
                    "fs.readFile encoding must be `utf8` or `base64`",
                )
            fs_backend = require_backend(backend, surface="fs.readFile")
            result = await fs_backend.aread(path, 0, READ_LIMIT)
            if result.error:
                raise_backend_error(result.error, fallback_code="ENOENT")
            file_data = result.file_data or {}
            content = coerce_file_text(file_data.get("content", ""))
            stored_encoding = str(file_data.get("encoding", "utf-8")).lower()
            if encoding == "utf8":
                if stored_encoding in {"utf-8", "utf8"}:
                    return content
                if stored_encoding == "base64":
                    try:
                        return base64.standard_b64decode(content).decode("utf-8")
                    except Exception as exc:  # noqa: BLE001
                        error(
                            "ERR_INVALID_ENCODING",
                            f"fs.readFile cannot decode binary data as utf8: {exc}",
                        )
                error(
                    "ERR_INVALID_ENCODING",
                    f"fs.readFile got unsupported stored encoding `{stored_encoding}`",
                )
            if stored_encoding == "base64":
                return content
            return base64.standard_b64encode(content.encode("utf-8")).decode("ascii")

        @ctx.function(name="__ciFsWriteFile")
        async def _write_file(payload: Any = None) -> None:  # noqa: C901
            args = require_dict(payload, "fs.writeFile")
            reject_unknown_keys(
                args,
                allowed={"path", "data", "encoding", "flag"},
                call="fs.writeFile",
            )
            path = normalize_abs_path(require_str(args, "path", "fs.writeFile"))
            data = args.get("data")
            if not isinstance(data, str):
                error("ERR_INVALID_ARG_TYPE", "fs.writeFile `data` must be a string")
            encoding = args.get("encoding", "utf8")
            if encoding not in {"utf8", "base64"}:
                error(
                    "ERR_INVALID_ENCODING",
                    "fs.writeFile encoding must be `utf8` or `base64`",
                )
            flag = args.get("flag", "w")
            if flag not in {"w", "wx"}:
                error("ERR_INVALID_ARG", "fs.writeFile `flag` must be `w` or `wx`")

            if encoding == "utf8":
                data_bytes = data.encode("utf-8")
            else:
                try:
                    data_bytes = base64.standard_b64decode(data)
                except Exception as exc:  # noqa: BLE001
                    error(
                        "ERR_INVALID_ENCODING",
                        f"fs.writeFile got invalid base64 input: {exc}",
                    )

            fs_backend = require_backend(backend, surface="fs.writeFile")
            if flag == "wx":
                existing = await fs_backend.aread(path, 0, 1)
                if existing.error is None:
                    error(
                        "EEXIST",
                        f"cannot write `{path}` because it already exists (`flag=wx`)",
                    )
                if existing.error and not is_missing_error(existing.error):
                    raise_backend_error(existing.error)

            upload_result = await fs_backend.aupload_files([(path, data_bytes)])
            if upload_result and upload_result[0].error:
                err = str(upload_result[0].error)
                if flag == "wx" and map_backend_error(err) == "EEXIST":
                    error(
                        "EEXIST",
                        f"cannot write `{path}` because it already exists (`flag=wx`)",
                    )
                raise_backend_error(err)

        @ctx.function(name="__ciFsReaddir")
        async def _readdir(payload: Any = None) -> list[str]:
            args = require_dict(payload, "fs.readdir")
            reject_unknown_keys(args, allowed={"path"}, call="fs.readdir")
            raw_path = args.get("path", ".")
            if not isinstance(raw_path, str):
                error("ERR_INVALID_ARG_TYPE", "fs.readdir `path` must be a string")
            path = normalize_abs_path(raw_path)
            fs_backend = require_backend(backend, surface="fs.readdir")
            result = await fs_backend.als(path)
            if result.error:
                raise_backend_error(result.error, fallback_code="ENOENT")
            names: set[str] = set()
            for entry in result.entries or []:
                value = entry.get("path")
                if not isinstance(value, str):
                    continue
                name = posixpath.basename(value.rstrip("/"))
                if name in {"", ".", ".."}:
                    continue
                names.add(name)
            return sorted(names)

        ctx.eval(
            "globalThis.fs = {"
            " readFile: async function(path, options) {"
            "   const opts = options ?? {};"
            "   return __ciFsReadFile({ path, ...opts });"
            " },"
            " writeFile: async function(path, data, options) {"
            "   const opts = options ?? {};"
            "   return __ciFsWriteFile({ path, data, ...opts });"
            " },"
            " readdir: async function(path = '.') {"
            "   return __ciFsReaddir({ path });"
            " },"
            "};"
            "undefined"
        )
