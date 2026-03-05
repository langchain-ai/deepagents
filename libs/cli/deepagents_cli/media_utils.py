"""Utilities for handling image and video media from clipboard and files."""

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages.content import VideoContentBlock

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mov",
        ".avi",
        ".webm",
        ".m4v",
        ".wmv",
    }
)
"""Video file extensions with validated magic-byte support."""


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


@dataclass
class VideoData:
    """Represents a pasted video with its base64 encoding."""

    base64_data: str
    format: str  # "mp4", "quicktime", etc.
    placeholder: str  # Display text like "[video 1]"

    def to_message_content(self) -> "VideoContentBlock":
        """Convert to LangChain `VideoContentBlock` format.

        Returns:
            `VideoContentBlock` with base64 data and mime_type.
        """
        from langchain_core.messages.content import create_video_block

        return create_video_block(
            base64=self.base64_data,
            mime_type=f"video/{self.format}",
        )


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
            base64_data=encode_to_base64(image_bytes),
            format=image_format,
            placeholder="[image]",
        )
    except (UnidentifiedImageError, OSError) as e:
        logger.debug("Failed to load image from %s: %s", path, e, exc_info=True)
        return None


def _detect_video_format(data: bytes) -> str | None:
    """Detect video MIME subtype from magic bytes.

    Args:
        data: Raw file bytes (at least 12 bytes for reliable detection).

    Returns:
        MIME subtype (e.g. "mp4", "webm") or `None` if unrecognized.
    """
    min_avi_len = 12
    if data[4:8] == b"ftyp":
        # ftyp box: major brand at bytes 8-12 distinguishes MOV vs MP4
        brand = data[8:12]
        if brand == b"qt  ":
            return "quicktime"
        return "mp4"
    if data[:4] == b"RIFF" and len(data) >= min_avi_len and data[8:12] == b"AVI ":
        return "avi"
    if data[:4] == b"\x30\x26\xb2\x75":  # ASF/WMV
        return "x-ms-wmv"
    if data[:4] == b"\x1a\x45\xdf\xa3":  # WebM/Matroska (EBML header)
        return "webm"
    return None


def get_video_from_path(path: pathlib.Path) -> VideoData | None:
    """Read and encode a video file from disk.

    Args:
        path: Path to the video file.

    Returns:
        `VideoData` when the file is a valid video, otherwise `None`.
    """
    # 20 MB — keeps base64 payload under ~27 MB, well within API limits
    max_video_bytes = 20 * 1024 * 1024

    suffix = path.suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        return None

    try:
        file_size = path.stat().st_size
        if file_size == 0:
            logger.debug("Video file is empty: %s", path)
            return None
        if file_size > max_video_bytes:
            logger.warning(
                "Video file %s is too large (%d MB, max %d MB)",
                path,
                file_size // (1024 * 1024),
                max_video_bytes // (1024 * 1024),
            )
            return None

        video_bytes = path.read_bytes()

        # Validate it's a real video file by checking magic bytes
        # MP4 starts with ftyp, MOV also uses ftyp, AVI starts with RIFF
        min_video_len = 8
        if len(video_bytes) < min_video_len:
            logger.debug("Video file too small (%d bytes): %s", len(video_bytes), path)
            return None

        # Detect format from magic bytes (not extension) so renamed files
        # get the correct MIME type.
        detected_format = _detect_video_format(video_bytes)
        if detected_format is None:
            logger.warning(
                "Video file %s has unrecognized signature for extension '%s'; "
                "skipping. If this is a valid video, the format may not be "
                "supported yet.",
                path,
                suffix,
            )
            return None

        return VideoData(
            base64_data=encode_to_base64(video_bytes),
            format=detected_format,
            placeholder="[video]",
        )
    except OSError as e:
        logger.warning("Failed to load video from %s: %s", path, e, exc_info=True)
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


def encode_to_base64(data: bytes) -> str:
    """Encode raw bytes to a base64 string.

    Args:
        data: Raw bytes to encode.

    Returns:
        Base64-encoded string.
    """
    return base64.b64encode(data).decode("utf-8")


def create_multimodal_content(
    text: str, images: list[ImageData], videos: list[VideoData] | None = None
) -> list[dict]:
    """Create multimodal message content with text, images, and videos.

    Args:
        text: Text content of the message
        images: List of ImageData objects
        videos: Optional list of VideoData objects

    Returns:
        List of content blocks in LangChain message format.
    """
    content_blocks = []

    # Add text block
    if text.strip():
        content_blocks.append({"type": "text", "text": text})

    # Add image blocks
    content_blocks.extend(image.to_message_content() for image in images)

    # Add video blocks
    if videos:
        content_blocks.extend(video.to_message_content() for video in videos)

    return content_blocks
