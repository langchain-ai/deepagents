"""Tests for image utilities (clipboard detection, base64 encoding, multimodal content)."""

import base64
import io
from unittest.mock import MagicMock, patch

from PIL import Image

from deepagents_cli.image_utils import (
    ImageData,
    create_multimodal_content,
    encode_image_to_base64,
    get_clipboard_image,
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

    def test_remove_image_and_reset_counter(self) -> None:
        """Test removing an image resets the counter appropriately."""
        tracker = ImageTracker()
        img1 = ImageData(base64_data="abc", format="png", placeholder="")
        img2 = ImageData(base64_data="def", format="png", placeholder="")

        tracker.add_image(img1)
        tracker.add_image(img2)

        # Simulate what happens on backspace delete
        tracker.images.pop(1)  # Remove image 2
        tracker.next_id = len(tracker.images) + 1

        assert tracker.next_id == 2
        assert len(tracker.images) == 1


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
    def test_unsupported_platform_returns_none(self) -> None:
        """Test that non-macOS platforms return None."""
        result = get_clipboard_image()
        assert result is None

    @patch("deepagents_cli.image_utils.sys.platform", "darwin")
    @patch("deepagents_cli.image_utils._get_macos_clipboard_image")
    def test_macos_calls_macos_function(self, mock_macos_fn: MagicMock) -> None:
        """Test that macOS platform calls the macOS-specific function."""
        mock_macos_fn.return_value = None
        get_clipboard_image()
        mock_macos_fn.assert_called_once()

    @patch("deepagents_cli.image_utils.sys.platform", "darwin")
    @patch("deepagents_cli.image_utils.subprocess.run")
    def test_pngpaste_success(self, mock_run: MagicMock) -> None:
        """Test successful image retrieval via pngpaste."""
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
    def test_pngpaste_not_installed_falls_back(self, mock_run: MagicMock) -> None:
        """Test fallback to osascript when pngpaste is not installed."""
        # First call (pngpaste) raises FileNotFoundError
        # Second call (osascript clipboard info) returns no image info
        mock_run.side_effect = [
            FileNotFoundError("pngpaste not found"),
            MagicMock(returncode=0, stdout="text data"),  # clipboard info - no pngf
        ]

        result = get_clipboard_image()

        # Should return None since clipboard has no image
        assert result is None
        # Should have tried both methods
        assert mock_run.call_count == 2

    @patch("deepagents_cli.image_utils.sys.platform", "darwin")
    @patch("deepagents_cli.image_utils._get_clipboard_via_osascript")
    @patch("deepagents_cli.image_utils.subprocess.run")
    def test_no_image_in_clipboard(self, mock_run: MagicMock, mock_osascript: MagicMock) -> None:
        """Test behavior when clipboard has no image."""
        # pngpaste fails
        mock_run.return_value = MagicMock(returncode=1, stdout=b"")
        # osascript fallback also returns None
        mock_osascript.return_value = None

        result = get_clipboard_image()
        assert result is None
