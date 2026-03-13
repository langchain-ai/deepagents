"""External editor support for composing prompts."""

from __future__ import annotations

import contextlib
import os
import shlex
import subprocess  # noqa: S404
import sys
import tempfile
from pathlib import Path

GUI_WAIT_FLAG: dict[str, str] = {
    "code": "--wait",
    "cursor": "--wait",
    "zed": "--wait",
    "atom": "--wait",
    "subl": "-w",
    "windsurf": "--wait",
}
"""Mapping of GUI editor base names to their blocking flag."""

VIM_EDITORS = {"vi", "vim", "nvim"}


def resolve_editor() -> list[str] | None:
    """Resolve editor command from environment.

    Checks $VISUAL, then $EDITOR, then falls back to platform default.

    Returns:
        Tokenized command list, or None if no editor could be resolved.
    """
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if not editor:
        if sys.platform == "win32":
            return ["notepad"]
        return ["vi"]
    return shlex.split(editor)


def _prepare_command(cmd: list[str], filepath: str) -> list[str]:
    """Build the full command list with appropriate flags.

    Adds --wait/-w for GUI editors and -i NONE for vim.

    Returns:
        The complete command list with flags and filepath appended.
    """
    cmd = list(cmd)  # copy
    exe = Path(cmd[0]).stem.lower()

    # Auto-inject wait flag for GUI editors
    if exe in GUI_WAIT_FLAG:
        flag = GUI_WAIT_FLAG[exe]
        if flag not in cmd:
            cmd.insert(1, flag)

    # Vim workaround: avoid viminfo errors in temp environments
    if exe in VIM_EDITORS and "-i" not in cmd:
        cmd.extend(["-i", "NONE"])

    cmd.append(filepath)
    return cmd


def open_in_editor(current_text: str) -> str | None:
    """Open current_text in an external editor.

    Creates a temp .md file, launches the editor, and reads back the result.

    Returns:
        The edited text, or None if the editor failed or the user cancelled.
    """
    cmd = resolve_editor()
    if cmd is None:
        return None

    with tempfile.NamedTemporaryFile(
        suffix=".md",
        prefix="deepagents-edit-",
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as tmp:
        tmp.write(current_text)

    try:
        full_cmd = _prepare_command(cmd, tmp.name)

        # S603: editor command comes from user's own $EDITOR env var
        result = subprocess.run(  # noqa: S603
            full_cmd,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=False,
        )
        if result.returncode != 0:
            return None

        edited = Path(tmp.name).read_text(encoding="utf-8")

        # Normalize line endings
        edited = edited.replace("\r\n", "\n").replace("\r", "\n")

        # Treat empty result as cancellation
        if not edited.strip():
            return None

    except FileNotFoundError:
        return None
    else:
        return edited
    finally:
        contextlib.suppress(OSError)
        Path(tmp.name).unlink(missing_ok=True)
