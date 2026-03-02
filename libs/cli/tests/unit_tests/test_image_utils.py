"""Tests for image utilities.

Covers clipboard detection, base64 encoding, and multimodal content.
"""

import base64
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from deepagents_cli.image_utils import (
    ImageData,
    create_multimodal_content,
    encode_image_to_base64,
    get_clipboard_image,
    get_image_from_path,
)
from deepagents_cli.input import ImageTracker


class TestImageData:
    """Tests for ImageData dataclass."""

    def test_to_message_content_png(self) -> None:
        """Test converting PNG image data to LangChain message format."""
        image = ImageData(
            base64_data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            format="png",
            placeholder="[image 1]",
        )
        result = image.to_message_content()

        assert result["type"] == "image_url"
        assert "image_url" in result
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_to_message_content_jpeg(self) -> None:
        """Test converting JPEG image data to LangChain message format."""
        image = ImageData(
            base64_data="abc123",
            format="jpeg",
            placeholder="[image 2]",
        )
        result = image.to_message_content()

        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/jpeg;base64,")


class TestImageTracker:
    """Tests for ImageTracker class."""

    def test_add_image_increments_counter(self) -> None:
        """Test that adding images increments the counter correctly."""
        tracker = ImageTracker()

        img1 = ImageData(base64_data="abc", format="png", placeholder="")
        img2 = ImageData(base64_data="def", format="png", placeholder="")

        placeholder1 = tracker.add_image(img1)
        placeholder2 = tracker.add_image(img2)

        assert placeholder1 == "[image 1]"
        assert placeholder2 == "[image 2]"
        assert img1.placeholder == "[image 1]"
        assert img2.placeholder == "[image 2]"

    def test_get_images_returns_copy(self) -> None:
        """Test that get_images returns a copy, not the original list."""
        tracker = ImageTracker()
        img = ImageData(base64_data="abc", format="png", placeholder="")
        tracker.add_image(img)

        images = tracker.get_images()
        images.clear()  # Modify the returned list

        # Original should be unchanged
        assert len(tracker.get_images()) == 1

    def test_clear_resets_counter(self) -> None:
        """Test that clear resets both images and counter."""
        tracker = ImageTracker()
        img = ImageData(base64_data="abc", format="png", placeholder="")
        tracker.add_image(img)
        tracker.add_image(img)

        assert tracker.next_id == 3
        assert len(tracker.images) == 2

        tracker.clear()

        assert tracker.next_id == 1
        assert len(tracker.images) == 0

    def test_add_after_clear_starts_at_one(self) -> None:
        """Test that adding after clear starts from [image 1] again."""
        tracker = ImageTracker()
        img = ImageData(base64_data="abc", format="png", placeholder="")

        tracker.add_image(img)
        tracker.add_image(img)
        tracker.clear()

        new_img = ImageData(base64_data="xyz", format="png", placeholder="")
        placeholder = tracker.add_image(new_img)

        assert placeholder == "[image 1]"

    def test_sync_to_text_resets_when_placeholders_removed(self) -> None:
        """Removing placeholders from input should clear tracked images and IDs."""
        tracker = ImageTracker()
        img = ImageData(base64_data="abc", format="png", placeholder="")

        tracker.add_image(img)
        tracker.add_image(img)
        tracker.sync_to_text("")

        assert tracker.images == []
        assert tracker.next_id == 1

    def test_sync_to_text_keeps_referenced_images(self) -> None:
        """Sync should prune unreferenced images while preserving next ID order."""
        tracker = ImageTracker()
        img1 = ImageData(base64_data="abc", format="png", placeholder="")
        img2 = ImageData(base64_data="def", format="png", placeholder="")

        tracker.add_image(img1)
        tracker.add_image(img2)
        tracker.sync_to_text("keep [image 2] only")

        assert tracker.next_id == 3
        assert len(tracker.images) == 1
        assert tracker.images[0].placeholder == "[image 2]"


class TestEncodeImageToBase64:
    """Tests for base64 encoding."""

    def test_encode_image_bytes(self) -> None:
        """Test encoding raw bytes to base64."""
        test_bytes = b"test image data"
        result = encode_image_to_base64(test_bytes)

        # Verify it's valid base64
        decoded = base64.b64decode(result)
        assert decoded == test_bytes

    def test_encode_png_bytes(self) -> None:
        """Test encoding actual PNG bytes."""
        # Create a small PNG in memory
        img = Image.new("RGB", (10, 10), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        result = encode_image_to_base64(png_bytes)

        # Should be valid base64
        decoded = base64.b64decode(result)
        assert decoded == png_bytes


class TestCreateMultimodalContent:
    """Tests for creating multimodal message content."""

    def test_text_only(self) -> None:
        """Test creating content with text only (no images)."""
        result = create_multimodal_content("Hello world", [])

        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Hello world"

    def test_text_and_one_image(self) -> None:
        """Test creating content with text and one image."""
        img = ImageData(base64_data="abc123", format="png", placeholder="[image 1]")
        result = create_multimodal_content("Describe this:", [img])

        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Describe this:"
        assert result[1]["type"] == "image_url"

    def test_text_and_multiple_images(self) -> None:
        """Test creating content with text and multiple images."""
        img1 = ImageData(base64_data="abc", format="png", placeholder="[image 1]")
        img2 = ImageData(base64_data="def", format="png", placeholder="[image 2]")
        result = create_multimodal_content("Compare these:", [img1, img2])

        assert len(result) == 3
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"
        assert result[2]["type"] == "image_url"

    def test_empty_text_with_image(self) -> None:
        """Test that empty text is not included in content."""
        img = ImageData(base64_data="abc", format="png", placeholder="[image 1]")
        result = create_multimodal_content("", [img])

        # Should only have the image, no empty text block
        assert len(result) == 1
        assert result[0]["type"] == "image_url"

    def test_whitespace_only_text(self) -> None:
        """Test that whitespace-only text is not included."""
        img = ImageData(base64_data="abc", format="png", placeholder="[image 1]")
        result = create_multimodal_content("   \n\t  ", [img])

        assert len(result) == 1
        assert result[0]["type"] == "image_url"


class TestGetClipboardImage:
    """Tests for clipboard image detection."""

    @patch("deepagents_cli.image_utils.sys.platform", "linux")
    def test_unsupported_platform_returns_none_and_warns(self) -> None:
        """Test that non-macOS platforms return None and log a warning."""
        with patch("deepagents_cli.image_utils.logger") as mock_logger:
            result = get_clipboard_image()
            assert result is None
            mock_logger.warning.assert_called_once()
            assert "linux" in mock_logger.warning.call_args[0][1]

    @patch("deepagents_cli.image_utils.sys.platform", "darwin")
    @patch("deepagents_cli.image_utils._get_macos_clipboard_image")
    def test_macos_calls_macos_function(self, mock_macos_fn: MagicMock) -> None:
        """Test that macOS platform calls the macOS-specific function."""
        mock_macos_fn.return_value = None
        get_clipboard_image()
        mock_macos_fn.assert_called_once()

    @patch("deepagents_cli.image_utils.sys.platform", "darwin")
    @patch("deepagents_cli.image_utils.subprocess.run")
    @patch("deepagents_cli.image_utils._get_executable")
    def test_pngpaste_success(
        self, mock_get_executable: MagicMock, mock_run: MagicMock
    ) -> None:
        """Test successful image retrieval via pngpaste."""
        # Mock _get_executable to return a path for pngpaste
        mock_get_executable.return_value = "/usr/local/bin/pngpaste"

        # Create a small valid PNG
        img = Image.new("RGB", (10, 10), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=png_bytes,
        )

        result = get_clipboard_image()

        assert result is not None
        assert result.format == "png"
        assert len(result.base64_data) > 0

    @patch("deepagents_cli.image_utils.sys.platform", "darwin")
    @patch("deepagents_cli.image_utils.subprocess.run")
    @patch("deepagents_cli.image_utils._get_executable")
    def test_pngpaste_not_installed_falls_back(
        self, mock_get_executable: MagicMock, mock_run: MagicMock
    ) -> None:
        """Test fallback to osascript when pngpaste is not installed."""
        # pngpaste not found, but osascript is available
        mock_get_executable.side_effect = lambda name: (
            "/usr/bin/osascript" if name == "osascript" else None
        )

        # osascript clipboard info returns no image info (no "pngf" in output)
        mock_run.return_value = MagicMock(returncode=0, stdout="text data")

        result = get_clipboard_image()

        # Should return None since clipboard has no image
        assert result is None
        # Should have tried osascript (clipboard info check)
        assert mock_run.call_count == 1

    @patch("deepagents_cli.image_utils.sys.platform", "darwin")
    @patch("deepagents_cli.image_utils._get_clipboard_via_osascript")
    @patch("deepagents_cli.image_utils.subprocess.run")
    def test_no_image_in_clipboard(
        self, mock_run: MagicMock, mock_osascript: MagicMock
    ) -> None:
        """Test behavior when clipboard has no image."""
        # pngpaste fails
        mock_run.return_value = MagicMock(returncode=1, stdout=b"")
        # osascript fallback also returns None
        mock_osascript.return_value = None

        result = get_clipboard_image()
        assert result is None


class TestGetImageFromPath:
    """Tests for loading local images from dropped file paths."""

    def test_get_image_from_path_png(self, tmp_path: Path) -> None:
        """Valid PNG files should be returned as ImageData."""
        img_path = tmp_path / "dropped.png"
        img = Image.new("RGB", (4, 4), color="red")
        img.save(img_path, format="PNG")

        result = get_image_from_path(img_path)

        assert result is not None
        assert result.format == "png"
        assert result.placeholder == "[image]"
        assert base64.b64decode(result.base64_data)

    def test_get_image_from_path_non_image_returns_none(self, tmp_path: Path) -> None:
        """Non-image files should be ignored."""
        file_path = tmp_path / "notes.txt"
        file_path.write_text("not an image")

        assert get_image_from_path(file_path) is None

    def test_get_image_from_path_missing_returns_none(self, tmp_path: Path) -> None:
        """Missing files should return None instead of raising."""
        file_path = tmp_path / "missing.png"
        assert get_image_from_path(file_path) is None

    def test_get_image_from_path_jpeg_normalizes_format(self, tmp_path: Path) -> None:
        """JPEG images should normalize 'JPEG' format to 'jpeg'."""
        img_path = tmp_path / "photo.jpg"
        img = Image.new("RGB", (4, 4), color="green")
        img.save(img_path, format="JPEG")

        result = get_image_from_path(img_path)

        assert result is not None
        assert result.format == "jpeg"


class TestSyncToTextWithIDGaps:
    """Tests for ImageTracker.sync_to_text with non-contiguous IDs."""

    def test_sync_to_text_with_id_gap_preserves_max_id(self) -> None:
        """Deleting the middle image should set next_id based on max surviving ID."""
        tracker = ImageTracker()
        img1 = ImageData(base64_data="a", format="png", placeholder="")
        img2 = ImageData(base64_data="b", format="png", placeholder="")
        img3 = ImageData(base64_data="c", format="png", placeholder="")

        tracker.add_image(img1)
        tracker.add_image(img2)
        tracker.add_image(img3)

        # Remove the middle placeholder — IDs 1 and 3 remain
        tracker.sync_to_text("[image 1] and [image 3]")

        assert len(tracker.images) == 2
        assert tracker.images[0].placeholder == "[image 1]"
        assert tracker.images[1].placeholder == "[image 3]"
        assert tracker.next_id == 4
