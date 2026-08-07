"""Assemble the prebuilt Switchyard runtime staged into Harbor's agent project."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path, PurePosixPath

_PROVIDER_SAFE_ADAPTER_MARKERS = {
    "content_mapper.py": "def ai_message_from_switchyard(",
    "request_mapper.py": "_ContentMapper.ai_message_from_switchyard(",
    "response_mapper.py": "_ContentMapper.ai_message_from_switchyard(",
}


def _destination(root: Path, member: str) -> Path:
    path = PurePosixPath(member)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        msg = f"Unsafe wheel member path: {member!r}"
        raise ValueError(msg)
    return root.joinpath(*path.parts)


def _validate_adapter(adapter: Path) -> None:
    """Require the provider-safe tool-call conversion patch before packaging.

    Args:
        adapter: Patched `langchain_nvidia_switchyard` package directory.

    Raises:
        FileNotFoundError: If the adapter package is incomplete.
        ValueError: If normalized tool calls can still leak into provider content.
    """
    if not (adapter / "__init__.py").is_file():
        msg = f"LangChain Switchyard adapter package is missing: {adapter}"
        raise FileNotFoundError(msg)
    for relative, marker in _PROVIDER_SAFE_ADAPTER_MARKERS.items():
        path = adapter / relative
        if not path.is_file():
            msg = f"LangChain Switchyard adapter file is missing: {path}"
            raise FileNotFoundError(msg)
        if marker not in path.read_text(encoding="utf-8"):
            msg = f"LangChain Switchyard adapter lacks provider-safe tool calls: {path}"
            raise ValueError(msg)


def assemble_runtime(wheel: Path, adapter: Path, output: Path) -> None:
    """Unpack a native wheel and add the pinned LangChain adapter source.

    Args:
        wheel: Patched `nemo-switchyard` wheel built for the Harbor agent runtime.
        adapter: `langchain_nvidia_switchyard` package directory from the pinned checkout.
        output: New directory to upload as the runtime artifact.

    Raises:
        FileExistsError: If `output` already exists.
        FileNotFoundError: If either input or a required runtime file is missing.
        ValueError: If the wheel contains an unsafe path.
    """
    if output.exists():
        msg = f"Runtime output already exists: {output}"
        raise FileExistsError(msg)
    if not wheel.is_file():
        msg = f"Switchyard wheel is missing: {wheel}"
        raise FileNotFoundError(msg)
    _validate_adapter(adapter)

    output.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            destination = _destination(output, member.filename)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)

    shutil.copytree(adapter, output / "langchain_nvidia_switchyard")
    required = [
        output / "switchyard" / "libsy" / "algorithms.py",
        output / "langchain_nvidia_switchyard" / "__init__.py",
    ]
    for path in required:
        if not path.is_file():
            msg = f"Assembled Switchyard runtime is missing: {path.relative_to(output)}"
            raise FileNotFoundError(msg)
    native = list((output / "switchyard_rust").glob("_switchyard_rust*.so"))
    if len(native) != 1:
        msg = f"Expected one native Switchyard extension, found {len(native)}"
        raise FileNotFoundError(msg)


def main() -> None:
    """Parse CLI arguments and assemble one runtime directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assemble_runtime(args.wheel, args.adapter, args.output)


if __name__ == "__main__":
    main()
