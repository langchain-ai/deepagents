"""Input handling, completers, and prompt session for the CLI."""

import os
import re
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


class FilePathCompleter(Completer):
    """File path completer that triggers on @ symbol with case-insensitive matching."""

    def __init__(self):
        self.path_completer = PathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        """Get file path completions when @ is detected."""
        text = document.text_before_cursor

        # Check if we're after an @ symbol
        if "@" in text:
            # Get the part after the last @
            parts = text.split("@")
            if len(parts) >= 2:
                after_at = parts[-1]
                # Create a document for just the path part
                path_doc = Document(after_at, len(after_at))

                # Get all completions from PathCompleter
                all_completions = list(
                    self.path_completer.get_completions(path_doc, complete_event)
                )

                # If user has typed something, filter case-insensitively
                if after_at.strip():
                    # Extract just the filename part for matching (not the full path)
                    search_parts = after_at.split("/")
                    search_term = search_parts[-1].lower() if search_parts else ""

                    # Filter completions case-insensitively
                    filtered_completions = [
                        c for c in all_completions if search_term in c.text.lower()
                    ]
                else:
                    # No search term, show all completions
                    filtered_completions = all_completions

                # Yield filtered completions
                for completion in filtered_completions:
                    yield Completion(
                        text=completion.text,
                        start_position=completion.start_position,
                        display=completion.display,
                        display_meta=completion.display_meta,
                        style=completion.style,
                    )


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
    pattern = r"@((?:[^\s@]|(?<=\\)\s)+)"  # Match @filename, allowing escaped spaces
    matches = re.findall(pattern, text)

    files = []
    for match in matches:
        # Remove escape characters
        clean_path = match.replace("\\ ", " ")
        path = Path(clean_path).expanduser()

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

    # Define styles for the toolbar with full-width background colors
    toolbar_style = Style.from_dict(
        {
            "bottom-toolbar": "noreverse",  # Disable default reverse video
            "toolbar-green": "bg:#10b981 #000000",  # Green for auto-accept ON
            "toolbar-orange": "bg:#f59e0b #000000",  # Orange for manual accept
        }
    )

    # Create the session
    session = PromptSession(
        message=HTML(f'<style fg="{COLORS["user"]}">></style> '),
        multiline=True,  # Keep multiline support but Enter submits
        key_bindings=kb,
        completer=merge_completers([CommandCompleter(), BashCompleter(), FilePathCompleter()]),
        editing_mode=EditingMode.EMACS,
        complete_while_typing=True,  # Show completions as you type
        mouse_support=False,
        enable_open_in_editor=True,  # Allow Ctrl+X Ctrl+E to open external editor
        bottom_toolbar=get_bottom_toolbar(session_state),  # Persistent status bar at bottom
        style=toolbar_style,  # Apply toolbar styling
    )

    return session
