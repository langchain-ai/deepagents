"""VS Code integration for previewing code changes as diffs."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console

console = Console(highlight=False)


def is_vscode_available() -> bool:
    """Check if VS Code CLI is available on the system.

    Returns:
        True if 'code' command is available, False otherwise.
    """
    return shutil.which("code") is not None


def open_diff_in_vscode(
    file_path: str | Path,
    before_content: str,
    after_content: str,
    *,
    wait: bool = False,
) -> bool:
    """Open a diff view in VS Code comparing before and after content.

    Creates temporary files for the before/after content and opens them
    in VS Code's diff view. The temporary files are created with appropriate
    extensions to enable syntax highlighting.

    Args:
        file_path: The target file path (used for naming and syntax highlighting).
        before_content: The original content (before changes).
        after_content: The new content (after changes).
        wait: If True, wait for VS Code window to close before returning.

    Returns:
        True if VS Code was opened successfully, False otherwise.
    """
    if not is_vscode_available():
        console.print("[yellow]VS Code CLI ('code' command) not found.[/yellow]")
        console.print("Install VS Code and ensure 'code' is in your PATH.")
        return False

    # Get file extension for syntax highlighting
    file_path = Path(file_path)
    extension = file_path.suffix

    # Create temporary directory for diff files
    temp_dir = Path(tempfile.mkdtemp(prefix="deepagents_diff_"))
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create temporary files with appropriate names
        before_file = temp_dir / f"{file_path.stem}_BEFORE{extension}"
        after_file = temp_dir / f"{file_path.stem}_AFTER{extension}"

        # Write content to temporary files
        before_file.write_text(before_content, encoding="utf-8")
        after_file.write_text(after_content, encoding="utf-8")

        # Build VS Code command
        cmd = ["code", "--diff", str(before_file), str(after_file)]

        # Add --wait flag if requested
        if wait:
            cmd.insert(1, "--wait")

        # Open diff in VS Code
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            console.print(f"[yellow]VS Code diff failed: {result.stderr}[/yellow]")
            return False

        # Show user feedback
        console.print(f"[dim]→ Opened diff in VS Code: {file_path.name}[/dim]")
        return True

    except Exception as e:
        console.print(f"[yellow]Error opening VS Code diff: {e}[/yellow]")
        return False
    finally:
        # Clean up temporary files (unless VS Code is still using them)
        if not wait:
            # Small delay to allow VS Code to read files
            import time
            time.sleep(0.5)
            try:
                shutil.rmtree(temp_dir)
            except OSError:
                # If cleanup fails, files will be cleaned up by OS eventually
                pass


def open_file_in_vscode(file_path: str | Path) -> bool:
    """Open a file in VS Code for viewing.

    Args:
        file_path: Path to the file to open.

    Returns:
        True if VS Code was opened successfully, False otherwise.
    """
    if not is_vscode_available():
        return False

    try:
        result = subprocess.run(
            ["code", str(file_path)],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return False

        console.print(f"[dim]→ Opened file in VS Code: {Path(file_path).name}[/dim]")
        return True

    except Exception:
        return False
