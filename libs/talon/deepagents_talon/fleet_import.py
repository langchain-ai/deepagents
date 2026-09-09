"""Fleet zip import support for Talon local agent directories."""

from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

_AGENT_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_ZIP_FILE_TYPE_MASK = 0o170000
_ZIP_SYMLINK_TYPE = 0o120000
_SUBAGENT_FILE_PARTS = 3
_MAX_ZIP_ENTRY_COUNT = 10_000
_MAX_ZIP_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_ZIP_COMPRESSION_RATIO = 100
_COPY_CHUNK_SIZE = 1024 * 1024


class FleetImportError(ValueError):
    """Raised when a Fleet zip cannot be materialized into a Talon agent directory."""


@dataclass(frozen=True, slots=True)
class FleetImportResult:
    """Summary of a completed Fleet zip import.

    Args:
        target_dir: Directory that received the materialized Talon agent files.
        root_prompt_count: Number of root prompt files written.
        subagent_prompt_count: Number of subagent prompt files written.
        config_ignored: Whether the Fleet zip contained a root `config.json`.
    """

    target_dir: Path
    root_prompt_count: int
    subagent_prompt_count: int
    config_ignored: bool


def import_fleet_zip(
    zip_path: Path,
    *,
    target_dir: Path,
    assistant_home: Path | None = None,
) -> FleetImportResult:
    """Materialize a Fleet zip export into a Talon local agent directory.

    Args:
        zip_path: Fleet export zip file to import.
        target_dir: Talon assistant directory to refresh with materialized files.
        assistant_home: Assistant state directory that should receive local
            subagents. Defaults to `target_dir`, keeping all writes under the
            explicit target.

    Returns:
        Summary of the materialized files.

    Raises:
        FleetImportError: If the zip is structurally unsafe, missing required
            prompts, or cannot be written.
    """
    source = zip_path.expanduser()
    target = target_dir.expanduser()
    home = assistant_home.expanduser() if assistant_home is not None else target
    try:
        with zipfile.ZipFile(source) as archive:
            entries = _validated_entries(archive)
            if "AGENTS.md" not in entries:
                msg = "AGENTS.md: missing required root prompt"
                raise FleetImportError(msg)
            with tempfile.TemporaryDirectory(prefix="deepagents-talon-import-") as raw:
                staging = Path(raw)
                _materialize_staging(archive, entries, staging)
                config_ignored = "config.json" in entries
                _refresh_target(staging, target, home)
    except zipfile.BadZipFile as exc:
        msg = f"{source}: invalid zip file"
        raise FleetImportError(msg) from exc
    except OSError as exc:
        msg = f"{target}: {exc}"
        raise FleetImportError(msg) from exc

    return FleetImportResult(
        target_dir=target,
        root_prompt_count=1,
        subagent_prompt_count=len(_subagent_prompt_paths(home)),
        config_ignored=config_ignored,
    )


def format_import_stdout(result: FleetImportResult) -> str:
    """Render a concise user-facing import summary.

    Args:
        result: Completed import summary.

    Returns:
        Text suitable for printing to stdout.
    """
    lines = [
        "Fleet import complete.",
        f"Agent files imported to: {result.target_dir}",
        f"Root prompts written: {result.root_prompt_count}",
        f"Subagent prompts written: {result.subagent_prompt_count}",
        f"config.json: {'ignored' if result.config_ignored else 'not present'}",
    ]
    return "\n".join(lines) + "\n"


def _validated_entries(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    entries: dict[str, zipfile.ZipInfo] = {}
    total_size = 0
    for info in archive.infolist():
        name = _normalized_zip_name(info.filename)
        if name is None:
            continue
        if _is_unsafe_zip_path(name):
            msg = f"{info.filename}: unsafe zip path"
            raise FleetImportError(msg)
        if _is_symlink(info):
            msg = f"{name}: symlink entries are not supported"
            raise FleetImportError(msg)
        if info.is_dir():
            continue
        _validate_zip_entry_size(name, info)
        if len(entries) >= _MAX_ZIP_ENTRY_COUNT:
            msg = f"{archive.filename}: too many zip entries"
            raise FleetImportError(msg)
        total_size += info.file_size
        if total_size > _MAX_ZIP_UNCOMPRESSED_BYTES:
            msg = f"{archive.filename}: zip uncompressed size exceeds limit"
            raise FleetImportError(msg)
        entries[name] = info
    return entries


def _normalized_zip_name(name: str) -> str | None:
    normalized = name.replace("\\", "/")
    if not normalized or normalized.endswith("/"):
        return None
    return normalized


def _is_unsafe_zip_path(name: str) -> bool:
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    return (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive != ""
        or any(part in {"", ".", ".."} for part in posix.parts)
    )


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    file_type = (info.external_attr >> 16) & _ZIP_FILE_TYPE_MASK
    return file_type == _ZIP_SYMLINK_TYPE


def _validate_zip_entry_size(name: str, info: zipfile.ZipInfo) -> None:
    if info.file_size > _MAX_ZIP_UNCOMPRESSED_BYTES:
        msg = f"{name}: zip entry uncompressed size exceeds limit"
        raise FleetImportError(msg)
    if info.compress_size == 0:
        return
    if info.file_size > info.compress_size * _MAX_ZIP_COMPRESSION_RATIO:
        msg = f"{name}: zip entry compression ratio exceeds limit"
        raise FleetImportError(msg)


def _materialize_staging(
    archive: zipfile.ZipFile,
    entries: Mapping[str, zipfile.ZipInfo],
    staging: Path,
) -> None:
    _copy_zip_file(archive, entries["AGENTS.md"], staging / "AGENTS.md")

    for name, info in entries.items():
        if name.startswith("skills/"):
            _copy_zip_file(archive, info, staging / name)
        elif _is_subagent_prompt_path(name):
            subagent = PurePosixPath(name).parts[1]
            _validate_agent_name(subagent, name)
            _copy_zip_file(archive, info, staging / "agents" / subagent / "AGENTS.md")


def _copy_zip_file(archive: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> None:
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    copied = 0
    with archive.open(info) as src, target.open("wb") as dst:
        while chunk := src.read(_COPY_CHUNK_SIZE):
            copied += len(chunk)
            if copied > info.file_size or copied > _MAX_ZIP_UNCOMPRESSED_BYTES:
                msg = f"{info.filename}: zip entry expanded beyond declared size"
                raise FleetImportError(msg)
            dst.write(chunk)
    target.chmod(0o600)


def _validate_agent_name(name: str, path: str) -> None:
    if not _AGENT_ID_PATTERN.fullmatch(name) or name in {".", ".."}:
        msg = f"{path}: unsafe subagent name {name!r}"
        raise FleetImportError(msg)


def _is_subagent_prompt_path(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (
        len(parts) == _SUBAGENT_FILE_PARTS and parts[0] == "subagents" and parts[2] == "AGENTS.md"
    )


def _refresh_target(staging: Path, target: Path, assistant_home: Path) -> None:
    assistant_home.mkdir(mode=0o700, parents=True, exist_ok=True)
    assistant_home.chmod(0o700)
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    target.chmod(0o700)
    _replace_file(staging / "AGENTS.md", target / "AGENTS.md")

    _replace_tree(staging / "skills", target / "skills")
    _replace_tree(staging / "agents", assistant_home / "agents")
    if assistant_home != target:
        _remove_path(target / "agents")
    _remove_path(target / "subagents")


def _replace_file(source: Path, target: Path) -> None:
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.tmp")
    shutil.copy2(source, temp)
    temp.chmod(0o600)
    temp.replace(target)


def _replace_tree(source: Path, target: Path) -> None:
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    if source.is_dir():
        shutil.copytree(source, target)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _subagent_prompt_paths(assistant_home: Path) -> list[Path]:
    agents = assistant_home / "agents"
    if not agents.is_dir():
        return []
    return sorted(path for path in agents.glob("*/AGENTS.md") if path.is_file())
