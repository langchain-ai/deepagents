"""Utilities for handling image paste from clipboard."""

import base64
import io
import logging
import os
import pathlib
import shutil

# S404: subprocess needed for clipboard access via pngpaste/osascript
import subprocess  # noqa: S404
import sys
import tempfile
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _get_executable(name: str) -> str | None:
    """Get full path to an executable using shutil.which().

    Args:
        name: Name of the executable to find

    Returns:
        Full path to executable, or None if not found.
    """
    return shutil.which(name)


@dataclass
class ImageData:
    """Represents a pasted image with its base64 encoding."""

    base64_data: str
    format: str  # "png", "jpeg", etc.
    placeholder: str  # Display text like "[image 1]"

    def to_message_content(self) -> dict:
        """Convert to LangChain message content format.

        Returns:
            Dict with type and image_url for multimodal messages.
        """
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/{self.format};base64,{self.base64_data}"},
        }


def get_clipboard_image() -> ImageData | None:
    """Attempt to read an image from the system clipboard.

    Supports macOS via `pngpaste` or `osascript`.

    Returns:
        ImageData if an image is found, None otherwise.
    """
    if sys.platform == "darwin":
        return _get_macos_clipboard_image()
    logger.warning(
        "Clipboard image paste is not supported on %s. "
        "Only macOS is currently supported. "
        "You can still attach images by dragging and dropping file paths.",
        sys.platform,
    )
    return None


def get_image_from_path(path: pathlib.Path) -> ImageData | None:
    """Read and encode an image file from disk.

    Args:
        path: Path to the image file.

    Returns:
        `ImageData` when the file is a valid image, otherwise `None`.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        image_bytes = path.read_bytes()
        if not image_bytes:
            return None

        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = (image.format or "").lower()

        if image_format == "jpg":
            image_format = "jpeg"
        if not image_format:
            suffix = path.suffix.lower().removeprefix(".")
            image_format = "jpeg" if suffix == "jpg" else suffix
        if not image_format:
            image_format = "png"

        return ImageData(
            base64_data=encode_image_to_base64(image_bytes),
            format=image_format,
            placeholder="[image]",
        )
    except (UnidentifiedImageError, OSError) as e:
        logger.debug("Failed to load image from %s: %s", path, e, exc_info=True)
        return None


def _get_macos_clipboard_image() -> ImageData | None:
    """Get clipboard image on macOS using pngpaste or osascript.

    First tries pngpaste (faster if installed), then falls back to osascript.

    Returns:
        ImageData if an image is found, None otherwise.
    """
    from PIL import Image, UnidentifiedImageError

    # Try pngpaste first (fast if installed)
    pngpaste_path = _get_executable("pngpaste")
    if pngpaste_path:
        try:
            # S603: pngpaste_path is validated via shutil.which(), args are hardcoded
            result = subprocess.run(  # noqa: S603
                [pngpaste_path, "-"],
                capture_output=True,
                check=False,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout:
                # Successfully got PNG data - validate it's a real image
                try:
                    Image.open(io.BytesIO(result.stdout))
                    base64_data = base64.b64encode(result.stdout).decode("utf-8")
                    return ImageData(
                        base64_data=base64_data,
                        format="png",  # 'pngpaste -' always outputs PNG
                        placeholder="[image]",
                    )
                except (
                    # UnidentifiedImageError: corrupted or non-image data
                    UnidentifiedImageError,
                    OSError,  # OSError: I/O errors during image processing
                ) as e:
                    logger.debug(
                        "Invalid image data from pngpaste: %s", e, exc_info=True
                    )
        except FileNotFoundError:
            # pngpaste not installed - expected on systems without it
            logger.debug("pngpaste not found, falling back to osascript")
        except subprocess.TimeoutExpired:
            logger.debug("pngpaste timed out after 2 seconds")

    # Fallback to osascript with temp file (built-in but slower)
    return _get_clipboard_via_osascript()


def _get_clipboard_via_osascript() -> ImageData | None:
    """Get clipboard image via osascript using a temp file.

    osascript outputs data in a special format that can't be captured as raw binary,
    so we write to a temp file instead.

    Returns:
        ImageData if an image is found, None otherwise.
    """
    from PIL import Image, UnidentifiedImageError

    # Get osascript path - it's a macOS builtin so should always exist
    osascript_path = _get_executable("osascript")
    if not osascript_path:
        return None

    # Create a temp file for the image
    fd, temp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)

    try:
        # First check if clipboard has PNG data
        # S603: osascript_path is validated via shutil.which(), args are hardcoded
        check_result = subprocess.run(  # noqa: S603
            [osascript_path, "-e", "clipboard info"],
            capture_output=True,
            check=False,
            timeout=2,
            text=True,
        )

        if check_result.returncode != 0:
            return None

        # Check for PNG or TIFF in clipboard info
        clipboard_info = check_result.stdout.lower()
        if "pngf" not in clipboard_info and "tiff" not in clipboard_info:
            return None

        # Try to get PNG first, fall back to TIFF
        if "pngf" in clipboard_info:
            get_script = f"""
            set pngData to the clipboard as «class PNGf»
            set theFile to open for access POSIX file "{temp_path}" with write permission
            write pngData to theFile
            close access theFile
            return "success"
            """  # noqa: E501
        else:
            get_script = f"""
            set tiffData to the clipboard as TIFF picture
            set theFile to open for access POSIX file "{temp_path}" with write permission
            write tiffData to theFile
            close access theFile
            return "success"
            """  # noqa: E501

        # S603: osascript_path validated via shutil.which(), script is internal
        result = subprocess.run(  # noqa: S603
            [osascript_path, "-e", get_script],
            capture_output=True,
            check=False,
            timeout=3,
            text=True,
        )

        if result.returncode != 0 or "success" not in result.stdout:
            return None

        # Check if file was created and has content
        if (
            not pathlib.Path(temp_path).exists()
            or pathlib.Path(temp_path).stat().st_size == 0
        ):
            return None

        # Read and validate the image
        image_data = pathlib.Path(temp_path).read_bytes()

        try:
            image = Image.open(io.BytesIO(image_data))
            # Convert to PNG if it's not already (e.g., if we got TIFF)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return ImageData(
                base64_data=base64_data,
                format="png",
                placeholder="[image]",
            )
        except (
            # UnidentifiedImageError: corrupted or non-image data
            UnidentifiedImageError,
            OSError,  # OSError: I/O errors during image processing
        ) as e:
            logger.debug(
                "Failed to process clipboard image via osascript: %s", e, exc_info=True
            )
            return None

    except subprocess.TimeoutExpired:
        logger.debug("osascript timed out while accessing clipboard")
        return None
    except OSError as e:
        logger.debug("OSError accessing clipboard via osascript: %s", e)
        return None
    finally:
        # Clean up temp file
        try:
            pathlib.Path(temp_path).unlink()
        except OSError as e:
            logger.debug("Failed to clean up temp file %s: %s", temp_path, e)


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Base64-encoded string.
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def create_multimodal_content(text: str, images: list[ImageData]) -> list[dict]:
    """Create multimodal message content with text and images.

    Args:
        text: Text content of the message
        images: List of ImageData objects

    Returns:
        List of content blocks in LangChain format.
    """
    content_blocks = []

    # Add text block
    if text.strip():
        content_blocks.append({"type": "text", "text": text})

    # Add image blocks
    content_blocks.extend(image.to_message_content() for image in images)

    return content_blocks
