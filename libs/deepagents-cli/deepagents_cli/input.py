"""Input handling, completers, and prompt session for the CLI."""

import os
import re
import shutil
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import (
    Completer,
    Completion,
    PathCompleter,
    WordCompleter,
    merge_completers,
)
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings

from .config import COLORS, COMMANDS, COMMON_BASH_COMMANDS, SessionState, console


class DirectoryAwarePathCompleter(Completer):
    """PathCompleter wrapper that adds trailing slashes to directories."""

    def __init__(self):
        self.path_completer = PathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        """Get completions and modify directory entries to include trailing /."""
        for completion in self.path_completer.get_completions(document, complete_event):
            # Check if this completion represents a directory
            # Build the full path from document text + completion text
            base_text = document.text[: document.cursor_position]
            completed_path = base_text + completion.text

            try:
                # Check if the completed path is a directory
                p = Path(completed_path).expanduser()
                if not p.is_absolute():
                    p = Path.cwd() / p

                # If it's a directory and doesn't already end with /
                if p.is_dir() and not completion.text.endswith("/"):
                    # Modify the completion to include trailing /
                    yield Completion(
                        text=completion.text + "/",
                        start_position=completion.start_position,
                        display=completion.display or (completion.text + "/"),
                        display_meta=completion.display_meta,
                    )
                else:
                    yield completion
            except Exception:
                # If path checking fails, yield original completion
                yield completion


class FilePathCompleter(Completer):
    """File path completer that triggers on @ symbol."""

    def __init__(self):
        self.path_completer = DirectoryAwarePathCompleter()

    def get_completions(self, document, complete_event):
        """Get file path completions when @ is detected."""
        text = document.text_before_cursor

        # Find the position of the last @ symbol
        last_at_pos = text.rfind("@")
        if last_at_pos >= 0:
            # Get the part after the @
            path_part = text[last_at_pos + 1 :]

            # Calculate cursor position within the path part
            cursor_in_path = document.cursor_position - last_at_pos - 1

            # Let PathCompleter handle everything - it knows how to deal with paths
            path_doc = Document(path_part, cursor_in_path)
            for completion in self.path_completer.get_completions(path_doc, complete_event):
                yield completion


class CommandCompleter(Completer):
    """Command completer for / commands."""

    def __init__(self):
        self.word_completer = WordCompleter(
            list(COMMANDS.keys()),
            meta_dict=COMMANDS,
            sentence=True,
            ignore_case=True,
        )

    def get_completions(self, document, complete_event):
        """Get command completions when / is at the start."""
        text = document.text

        # Only complete if line starts with /
        if text.startswith("/"):
            # Remove / for word completion
            cmd_text = text[1:]
            adjusted_doc = Document(
                cmd_text, document.cursor_position - 1 if document.cursor_position > 0 else 0
            )

            for completion in self.word_completer.get_completions(adjusted_doc, complete_event):
                yield completion


class BashCompleter(Completer):
    """Bash command completer for ! commands."""

    def __init__(self):
        self.word_completer = WordCompleter(
            list(COMMON_BASH_COMMANDS.keys()),
            meta_dict=COMMON_BASH_COMMANDS,
            sentence=True,
            ignore_case=True,
        )

    def get_completions(self, document, complete_event):
        """Get bash command completions when ! is at the start."""
        text = document.text

        # Only complete if line starts with !
        if text.startswith("!"):
            # Remove ! for word completion
            cmd_text = text[1:]
            adjusted_doc = Document(
                cmd_text, document.cursor_position - 1 if document.cursor_position > 0 else 0
            )

            for completion in self.word_completer.get_completions(adjusted_doc, complete_event):
                yield completion


def parse_file_mentions(text: str) -> tuple[str, list[Path]]:
    """Extract @file mentions and return cleaned text with resolved file paths."""
    # Simple pattern: @ followed by valid filename characters
    # Stops at whitespace and punctuation (?, !, ,, etc.)
    pattern = r"@([A-Za-z0-9._/~-]+)"
    matches = re.findall(pattern, text)

    files = []
    for match in matches:
        # Expand ~ to home directory
        path = Path(match).expanduser()

        # Try to resolve relative to cwd
        if not path.is_absolute():
            path = Path.cwd() / path

        try:
            path = path.resolve()
            if path.exists() and path.is_file():
                files.append(path)
            else:
                console.print(f"[yellow]Warning: File not found: {match}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: Invalid path {match}: {e}[/yellow]")

    return text, files


def get_bottom_toolbar(session_state: SessionState):
    """Return toolbar function that shows auto-approve status."""

    def toolbar():
        if session_state.auto_approve:
            # Green background when auto-approve is ON
            return [("class:toolbar-green", "auto-accept ON (CTRL+T to toggle)")]
        # Orange background when manual accept (auto-approve OFF)
        return [("class:toolbar-orange", "manual accept (CTRL+T to toggle)")]

    return toolbar


def create_prompt_session(assistant_id: str, session_state: SessionState) -> PromptSession:
    """Create a configured PromptSession with all features."""
    # Set default editor if not already set
    if "EDITOR" not in os.environ:
        os.environ["EDITOR"] = "nano"

    # Create key bindings
    kb = KeyBindings()

    # Bind Ctrl+T to toggle auto-approve
    @kb.add("c-t")
    def _(event):
        """Toggle auto-approve mode."""
        session_state.toggle_auto_approve()
        # Force UI refresh to update toolbar
        event.app.invalidate()

    # Bind backspace to update completions
    @kb.add("backspace")
    def _(event):
        """Handle backspace - delete char and update completions if menu was open."""
        buffer = event.current_buffer

        # Remember if completion menu was showing
        had_completions = buffer.complete_state is not None

        # Do the actual backspace
        buffer.delete_before_cursor(count=1)

        # If we had completions showing, retrigger them for the new text
        if had_completions:
            buffer.start_completion(select_first=False)

    # Bind regular Enter to submit (intuitive behavior)
    @kb.add("enter")
    def _(event):
        """Enter submits the input, unless completion menu is active."""
        buffer = event.current_buffer

        # If completion menu is showing, apply the current completion
        if buffer.complete_state:
            # Get the current completion (the highlighted one)
            current_completion = buffer.complete_state.current_completion

            # If no completion is selected (user hasn't navigated), select and apply the first one
            if not current_completion and buffer.complete_state.completions:
                # Move to the first completion
                buffer.complete_next()
                # Now apply it
                buffer.apply_completion(buffer.complete_state.current_completion)
            elif current_completion:
                # Apply the already-selected completion
                buffer.apply_completion(current_completion)

                # If the completion ends with /, it's a directory - keep completing
                # (DirectoryAwarePathCompleter automatically adds / to directories)
                if current_completion.text.endswith("/"):
                    # Trigger completions again to show directory contents
                    buffer.start_completion(select_first=False)
            else:
                # No completions available, close menu
                buffer.complete_state = None
        # Don't submit if buffer is empty or only whitespace
        elif buffer.text.strip():
            # Normal submit
            buffer.validate_and_handle()
            # If empty, do nothing (don't submit)

    # Alt+Enter for newlines (press ESC then Enter, or Option+Enter on Mac)
    @kb.add("escape", "enter")
    def _(event):
        """Alt+Enter inserts a newline for multi-line input."""
        event.current_buffer.insert_text("\n")

    # Ctrl+E to open in external editor
    @kb.add("c-e")
    def _(event):
        """Open the current input in an external editor (nano by default)."""
        event.current_buffer.open_in_editor()

    from prompt_toolkit.styles import Style

    # Define styles for the toolbar
    toolbar_style = Style.from_dict(
        {
            "bottom-toolbar": "noreverse",
            "toolbar-green": "bg:#10b981 #000000",  # Green for auto-approve ON
            "toolbar-orange": "bg:#f59e0b #000000",  # Orange for manual approve
        }
    )

    # Calculate rows to reserve for the autocomplete menu so it stays visible.
    try:
        terminal_height = shutil.get_terminal_size().lines
    except OSError:
        terminal_height = 24
    # Keep a small cushion and cap it to avoid eating the whole screen.
    reserve_rows = max(4, min(10, max(terminal_height // 4, 1)))

    # Create the session
    session = PromptSession(
        message=HTML(f'<style fg="{COLORS["user"]}">></style> '),
        prompt_continuation="",  # Empty continuation (no extra prompts on resize)
        multiline=True,  # Keep multiline support but Enter submits
        key_bindings=kb,
        completer=merge_completers([CommandCompleter(), BashCompleter(), FilePathCompleter()]),
        editing_mode=EditingMode.EMACS,
        complete_while_typing=True,  # Show completions as you type
        mouse_support=False,
        enable_open_in_editor=True,  # Allow Ctrl+X Ctrl+E to open external editor
        reserve_space_for_menu=reserve_rows,
        bottom_toolbar=get_bottom_toolbar(session_state),  # Status toolbar
        style=toolbar_style,
    )

    return session
