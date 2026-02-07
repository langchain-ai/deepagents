"""Unit tests for Harbor backend edit behavior."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from deepagents_harbor.backend import HarborSandbox


@dataclass
class _LocalExecResult:
    stdout: str
    stderr: str
    return_code: int


class _LocalEnvironment:
    async def exec(self, command: str) -> _LocalExecResult:
        process = await asyncio.create_subprocess_shell(
            command,
            executable="/bin/bash",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return _LocalExecResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            return_code=process.returncode or 0,
        )


@pytest.mark.asyncio
async def test_aedit_multiline_payload_replaces_full_block_once(tmp_path: Path) -> None:
    """`aedit` should support multiline old/new strings (summarization append pattern)."""
    file_path = tmp_path / "history.md"
    old = "## Summarized at t1\n\nline1\nline2\n\n"
    new = old + "## Summarized at t2\n\nline3\n\n"
    file_path.write_text(old, encoding="utf-8")

    sandbox = HarborSandbox(environment=_LocalEnvironment())  # type: ignore[arg-type]
    result = await sandbox.aedit(str(file_path), old, new)

    assert result.error is None
    assert result.path == str(file_path)
    assert result.occurrences == 1
    assert file_path.read_text(encoding="utf-8") == new


@pytest.mark.asyncio
async def test_aedit_multiline_payload_errors_when_multiple_without_replace_all(
    tmp_path: Path,
) -> None:
    """`aedit` should preserve uniqueness checks for non-replace-all mode."""
    file_path = tmp_path / "dup.md"
    block = "alpha\nbeta\n"
    file_path.write_text(f"{block}x\n{block}y\n", encoding="utf-8")

    sandbox = HarborSandbox(environment=_LocalEnvironment())  # type: ignore[arg-type]
    result = await sandbox.aedit(str(file_path), block, "replaced\n")

    assert result.path is None
    assert result.occurrences is None
    assert result.error is not None
    assert "appears multiple times" in result.error
