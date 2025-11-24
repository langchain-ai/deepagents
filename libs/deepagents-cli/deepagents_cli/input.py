"""Input handling, completers, and prompt session for the CLI."""

import asyncio
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import (
    Completer,
    Completion,
    PathCompleter,
    merge_completers,
)
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings

from .config import COLORS, COMMANDS, SessionState, console

# Regex patterns for context-aware completion
AT_MENTION_RE = re.compile(r"@(?P<path>(?:[^\s@]|(?<=\\)\s)*)$")
SLASH_COMMAND_RE = re.compile(r"^/(?P<command>[a-z]*)$")

EXIT_CONFIRM_WINDOW = 3.0


class FilePathCompleter(Completer):
    """Activate filesystem completion only when cursor is after '@'."""

    def __init__(self) -> None:
        self.path_completer = PathCompleter(
            expanduser=True,
            min_input_len=0,
            only_directories=False,
        )

    def get_completions(self, document, complete_event):
        """Get file path completions when @ is detected."""
        text = document.text_before_cursor

        # Use regex to detect @path pattern at end of line
        m = AT_MENTION_RE.search(text)
        if not m:
            return  # Not in an @path context

        path_fragment = m.group("path")

        # Unescape the path for PathCompleter (it doesn't understand escape sequences)
        unescaped_fragment = path_fragment.replace("\\ ", " ")

        # Strip trailing backslash if present (user is in the process of typing an escape)
        unescaped_fragment = unescaped_fragment.removesuffix("\\")

        # Create temporary document for the unescaped path fragment
        temp_doc = Document(text=unescaped_fragment, cursor_position=len(unescaped_fragment))

        # Get completions from PathCompleter and use its start_position
        # PathCompleter returns suffix text with start_position=0 (insert at cursor)
        for comp in self.path_completer.get_completions(temp_doc, complete_event):
            # Add trailing / for directories so users can continue navigating
            completed_path = Path(unescaped_fragment + comp.text).expanduser()
            # Re-escape spaces in the completion text for the command line
            completion_text = comp.text.replace(" ", "\\ ")
            if completed_path.is_dir() and not completion_text.endswith("/"):
                completion_text += "/"

            yield Completion(
                text=completion_text,
                start_position=comp.start_position,  # Use PathCompleter's position (usually 0)
                display=comp.display,
                display_meta=comp.display_meta,
            )


class CommandCompleter(Completer):
    """Activate command completion only when line starts with '/'."""

    def get_completions(self, document, _complete_event):
        """Get command completions when / is at the start."""
        text = document.text_before_cursor

        # Use regex to detect /command pattern at start of line
        m = SLASH_COMMAND_RE.match(text)
        if not m:
            return  # Not in a /command context

        command_fragment = m.group("command")

        # Match commands that start with the fragment (case-insensitive)
        for cmd_name, cmd_desc in COMMANDS.items():
            if cmd_name.startswith(command_fragment.lower()):
                yield Completion(
                    text=cmd_name,
                    start_position=-len(command_fragment),  # Fixed position for original document
                    display=cmd_name,
                    display_meta=cmd_desc,
                )


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


def get_bottom_toolbar(
    session_state: SessionState, session_ref: dict
) -> Callable[[], list[tuple[str, str]]]:
    """Return toolbar function that shows auto-approve status and BASH MODE."""

    def toolbar() -> list[tuple[str, str]]:
        parts = []

        # Check if we're in BASH mode (input starts with !)
        try:
            session = session_ref.get("session")
            if session:
                current_text = session.default_buffer.text
                if current_text.startswith("!"):
                    parts.append(("bg:#ff1493 fg:#ffffff bold", " BASH MODE "))
                    parts.append(("", " | "))
        except (AttributeError, TypeError):
            # Silently ignore - toolbar is non-critical and called frequently
            pass

        # Base status message
        if session_state.auto_approve:
            base_msg = "auto-accept ON (CTRL+T to toggle)"
            base_class = "class:toolbar-green"
        else:
            base_msg = "manual accept (CTRL+T to toggle)"
            base_class = "class:toolbar-orange"

        parts.append((base_class, base_msg))

        # Show exit confirmation hint if active
        hint_until = session_state.exit_hint_until
        if hint_until is not None:
            now = time.monotonic()
            if now < hint_until:
                parts.append(("", " | "))
                parts.append(("class:toolbar-exit", " Ctrl+C again to exit "))
            else:
                session_state.exit_hint_until = None

        return parts

    return toolbar


def create_prompt_session(_assistant_id: str, session_state: SessionState) -> PromptSession:
    """Create a configured PromptSession with all features."""
    # Set default editor if not already set
    if "EDITOR" not in os.environ:
        os.environ["EDITOR"] = "nano"

    # Create key bindings
    kb = KeyBindings()

    @kb.add("c-c")
    def _(event) -> None:
        """Require double Ctrl+C within a short window to exit."""
        app = event.app
        now = time.monotonic()

        if session_state.exit_hint_until is not None and now < session_state.exit_hint_until:
            handle = session_state.exit_hint_handle
            if handle:
                handle.cancel()
                session_state.exit_hint_handle = None
            session_state.exit_hint_until = None
            app.invalidate()
            app.exit(exception=KeyboardInterrupt())
            return

        session_state.exit_hint_until = now + EXIT_CONFIRM_WINDOW

        handle = session_state.exit_hint_handle
        if handle:
            handle.cancel()

        loop = asyncio.get_running_loop()
        app_ref = app

        def clear_hint() -> None:
            if (
                session_state.exit_hint_until is not None
                and time.monotonic() >= session_state.exit_hint_until
            ):
                session_state.exit_hint_until = None
                session_state.exit_hint_handle = None
                app_ref.invalidate()

        session_state.exit_hint_handle = loop.call_later(EXIT_CONFIRM_WINDOW, clear_hint)

        app.invalidate()

    # Bind Ctrl+T to toggle auto-approve
    @kb.add("c-t")
    def _(event) -> None:
        """Toggle auto-approve mode."""
        session_state.toggle_auto_approve()
        # Force UI refresh to update toolbar
        event.app.invalidate()

    # Bind regular Enter to submit (intuitive behavior)
    @kb.add("enter")
    def _(event) -> None:
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
    def _(event) -> None:
        """Alt+Enter inserts a newline for multi-line input."""
        event.current_buffer.insert_text("\n")

    # Ctrl+E to open in external editor
    @kb.add("c-e")
    def _(event) -> None:
        """Open the current input in an external editor (nano by default)."""
        event.current_buffer.open_in_editor()

    # Backspace handler to retrigger completions after deletion
    @kb.add("backspace")
    def _(event) -> None:
        """Handle backspace and retrigger completion if in @ or / context."""
        buffer = event.current_buffer

        # Perform the normal backspace action
        buffer.delete_before_cursor(count=1)

        # Check if we're in a completion context (@ or /)
        text = buffer.document.text_before_cursor
        if AT_MENTION_RE.search(text) or SLASH_COMMAND_RE.match(text):
            # Retrigger completion
            buffer.start_completion(select_first=False)

    from prompt_toolkit.styles import Style

    # Define styles for the toolbar with full-width background colors
    toolbar_style = Style.from_dict(
        {
            "bottom-toolbar": "noreverse",  # Disable default reverse video
            "toolbar-green": "bg:#10b981 #000000",  # Green for auto-accept ON
            "toolbar-orange": "bg:#f59e0b #000000",  # Orange for manual accept
            "toolbar-exit": "bg:#2563eb #ffffff",  # Blue for exit hint
        }
    )

    # Create session reference dict for toolbar to access session
    session_ref = {}

    # Create the session
    session = PromptSession(
        message=HTML(f'<style fg="{COLORS["user"]}">></style> '),
        multiline=True,  # Keep multiline support but Enter submits
        key_bindings=kb,
        completer=merge_completers([CommandCompleter(), FilePathCompleter()]),
        editing_mode=EditingMode.EMACS,
        complete_while_typing=True,  # Show completions as you type
        complete_in_thread=True,  # Async completion prevents menu freezing
        mouse_support=False,
        enable_open_in_editor=True,  # Allow Ctrl+X Ctrl+E to open external editor
        bottom_toolbar=get_bottom_toolbar(
            session_state, session_ref
        ),  # Persistent status bar at bottom
        style=toolbar_style,  # Apply toolbar styling
        reserve_space_for_menu=7,  # Reserve space for completion menu to show 5-6 results
    )

    # Store session reference for toolbar to access
    session_ref["session"] = session

    return session
