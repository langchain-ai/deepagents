"""Input handling utilities including image tracking and file mention parsing."""

import logging
import re
import shlex
from pathlib import Path
from urllib.parse import unquote, urlparse

from deepagents_cli.config import console
from deepagents_cli.image_utils import ImageData

logger = logging.getLogger(__name__)

PATH_CHAR_CLASS = r"A-Za-z0-9._~/\\:-"
"""Characters allowed in file paths.

Includes alphanumeric, period, underscore, tilde (home), forward/back slashes
(path separators), colon (Windows drive letters), and hyphen.
"""

FILE_MENTION_PATTERN = re.compile(r"@(?P<path>(?:\\.|[" + PATH_CHAR_CLASS + r"])+)")
"""Pattern for extracting `@file` mentions from input text.

Matches `@` followed by one or more path characters or escaped character
pairs (backslash + any character, e.g., `\\ ` for spaces in paths).

Uses `+` (not `*`) because a bare `@` without a path is not a valid
file reference.
"""

EMAIL_PREFIX_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]$")
"""Pattern to detect email-like text preceding an `@` symbol.

If the character immediately before `@` matches this pattern, the `@mention`
is likely part of an email address (e.g., `user@example.com`) rather than
a file reference.
"""

INPUT_HIGHLIGHT_PATTERN = re.compile(
    r"(^\/[a-zA-Z0-9_-]+|@(?:\\.|[" + PATH_CHAR_CLASS + r"])+)"
)
"""Pattern for highlighting `@mentions` and `/commands` in rendered
user messages.

Matches either:
- Slash commands at the start of the string (e.g., `/help`)
- `@file` mentions anywhere in the text (e.g., `@README.md`)

Note: The `^` anchor matches start of string, not start of line. The consumer
in `UserMessage.compose()` additionally checks `start == 0` before styling
slash commands, so a `/` mid-string is not highlighted.
"""

IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[image (?P<id>\d+)\]")
"""Pattern for image placeholders with a named `id` capture group.

Used to extract numeric IDs from placeholder tokens so the tracker can prune
stale entries and compute the next available ID.
"""


class ImageTracker:
    """Track pasted images in the current conversation."""

    def __init__(self) -> None:
        """Initialize an empty image tracker.

        Sets up an empty list to store images and initializes the ID counter
        to 1 for generating unique placeholder identifiers.
        """
        self.images: list[ImageData] = []
        self.next_id = 1

    def add_image(self, image_data: ImageData) -> str:
        """Add an image and return its placeholder text.

        Args:
            image_data: The image data to track

        Returns:
            Placeholder string like "[image 1]"
        """
        placeholder = f"[image {self.next_id}]"
        image_data.placeholder = placeholder
        self.images.append(image_data)
        self.next_id += 1
        return placeholder

    def get_images(self) -> list[ImageData]:
        """Get all tracked images.

        Returns:
            Copy of the list of tracked images.
        """
        return self.images.copy()

    def clear(self) -> None:
        """Clear all tracked images and reset counter."""
        self.images.clear()
        self.next_id = 1

    def sync_to_text(self, text: str) -> None:
        """Retain only images still referenced by placeholders in current text.

        Args:
            text: Current input text shown to the user.
        """
        placeholders = {
            match.group(0) for match in IMAGE_PLACEHOLDER_PATTERN.finditer(text)
        }
        if not placeholders:
            self.clear()
            return

        self.images = [img for img in self.images if img.placeholder in placeholders]
        if not self.images:
            self.next_id = 1
            return

        max_id = 0
        for image in self.images:
            match = IMAGE_PLACEHOLDER_PATTERN.fullmatch(image.placeholder)
            if match is None:
                continue
            max_id = max(max_id, int(match.group("id")))

        self.next_id = max_id + 1 if max_id else len(self.images) + 1


def parse_file_mentions(text: str) -> tuple[str, list[Path]]:
    r"""Extract `@file` mentions and return the text with resolved file paths.

    Parses `@file` mentions from the input text and resolves them to absolute
    file paths. Files that do not exist or cannot be resolved are excluded with
    a warning printed to the console.

    Email addresses (e.g., `user@example.com`) are automatically excluded by
    detecting email-like characters before the `@` symbol.

    Backslash-escaped spaces in paths (e.g., `@my\ folder/file.txt`) are
    unescaped before resolution. Tilde paths (e.g., `@~/file.txt`) are expanded
    via `Path.expanduser()`. Only regular files are returned; directories are
    excluded.

    This function does not raise exceptions; invalid paths are handled
    internally with a console warning.

    Args:
        text: Input text potentially containing `@file` mentions.

    Returns:
        Tuple of (original text unchanged, list of resolved file paths that exist).
    """
    matches = FILE_MENTION_PATTERN.finditer(text)

    files = []
    for match in matches:
        # Skip if this looks like an email address
        text_before = text[: match.start()]
        if text_before and EMAIL_PREFIX_PATTERN.search(text_before):
            continue

        raw_path = match.group("path")
        clean_path = raw_path.replace("\\ ", " ")

        try:
            path = Path(clean_path).expanduser()

            if not path.is_absolute():
                path = Path.cwd() / path

            resolved = path.resolve()
            if resolved.exists() and resolved.is_file():
                files.append(resolved)
            else:
                console.print(f"[yellow]Warning: File not found: {raw_path}[/yellow]")
        except (OSError, RuntimeError) as e:
            console.print(f"[yellow]Warning: Invalid path {raw_path}: {e}[/yellow]")

    return text, files


def parse_pasted_file_paths(text: str) -> list[Path]:
    r"""Parse a paste payload that may contain dragged-and-dropped file paths.

    The parser is strict on purpose: it only returns paths when the entire paste
    payload can be interpreted as one or more existing files. Any invalid token
    falls back to normal text paste behavior by returning an empty list.

    Supports common dropped-path formats:

    - Absolute/relative paths
    - POSIX shell quoting and escaping
    - `file://` URLs

    Args:
        text: Raw paste payload from the terminal.

    Returns:
        List of resolved file paths, or an empty list when parsing fails.
    """
    payload = text.strip()
    if not payload:
        return []

    tokens: list[str] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line_tokens = _split_paste_line(line)
        if not line_tokens:
            return []
        tokens.extend(line_tokens)

    if not tokens:
        return []

    paths: list[Path] = []
    for token in tokens:
        path = _token_to_path(token)
        if path is None:
            return []
        try:
            resolved = path.expanduser().resolve()
        except (OSError, RuntimeError) as e:
            logger.debug("Path resolution failed for token %r: %s", token, e)
            return []
        if not resolved.exists() or not resolved.is_file():
            return []
        paths.append(resolved)

    return paths


def _split_paste_line(line: str) -> list[str]:
    """Split a single pasted line into path-like tokens.

    Args:
        line: A single line from the paste payload.

    Returns:
        Parsed shell-like tokens, or an empty list when parsing fails.
    """
    try:
        return shlex.split(line, posix=True)
    except ValueError:
        # Unbalanced quotes or other tokenization errors: treat as plain text.
        return []


def _token_to_path(token: str) -> Path | None:
    """Convert a pasted token into a path candidate.

    Args:
        token: A single shell-split token from the paste payload.

    Returns:
        A parsed path candidate, or `None` when token parsing fails.
    """
    value = token.strip()
    if not value:
        return None

    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()
        if not value:
            return None

    if value.startswith("file://"):
        parsed = urlparse(value)
        path_text = unquote(parsed.path or "")
        if parsed.netloc and parsed.netloc != "localhost":
            path_text = f"//{parsed.netloc}{path_text}"
        if (
            path_text.startswith("/")
            and len(path_text) > 2  # noqa: PLR2004  # '/C:' minimum for Windows file URI
            and path_text[2] == ":"
            and path_text[1].isalpha()
        ):
            # `file:///C:/...` on Windows includes an extra leading slash.
            path_text = path_text[1:]
        if not path_text:
            return None
        return Path(path_text)

    return Path(value)
