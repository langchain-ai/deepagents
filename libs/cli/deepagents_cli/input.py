"""Input handling utilities including image tracking and file mention parsing."""

import re
from pathlib import Path

from deepagents_cli.config import console
from deepagents_cli.image_utils import ImageData


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


def parse_file_mentions(text: str) -> tuple[str, list[Path]]:
    """Extract @file mentions and return cleaned text with resolved file paths.

    Returns:
        Tuple of (original text, list of resolved file paths).
    """
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
