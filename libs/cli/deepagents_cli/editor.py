"""External editor support for composing prompts."""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
import subprocess  # noqa: S404
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

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
"""Set of vim-family editor base names that receive the `-i NONE` flag."""


def resolve_editor() -> list[str] | None:
    """Resolve editor command from environment.

    Checks $VISUAL, then $EDITOR, then falls back to platform default.

    Returns:
        Tokenized command list, or `None` if the env var was set but empty after
            tokenization.
    """
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if not editor:
        if sys.platform == "win32":
            return ["notepad"]
        return ["vi"]
    tokens = shlex.split(editor)
    return tokens or None


def _prepare_command(cmd: list[str], filepath: str) -> list[str]:
    """Build the full command list with appropriate flags.

    Adds --wait/-w for GUI editors and `-i NONE` for vim-family editors.

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

    Args:
        current_text: The text to pre-populate in the editor.

    Returns:
        The edited text with normalized line endings, or `None` if the editor
            exited with a non-zero status, was not found, or the result was
            empty/whitespace-only.
    """
    cmd = resolve_editor()
    if cmd is None:
        return None

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".md",
            prefix="deepagents-edit-",
            delete=False,
            mode="w",
            encoding="utf-8",
        ) as tmp:
            tmp_path = tmp.name
            tmp.write(current_text)

        full_cmd = _prepare_command(cmd, tmp_path)

        # S603: editor command comes from user's own $EDITOR env var
        result = subprocess.run(  # noqa: S603
            full_cmd,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "Editor exited with code %d: %s", result.returncode, full_cmd
            )
            return None

        edited = Path(tmp_path).read_text(encoding="utf-8")

        # Normalize line endings
        edited = edited.replace("\r\n", "\n").replace("\r", "\n")

        # Treat empty result as cancellation
        if not edited.strip():
            return None

    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("Editor failed", exc_info=True)
        return None
    else:
        return edited
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink(missing_ok=True)
