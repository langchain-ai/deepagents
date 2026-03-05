"""Tests for media utilities.

Covers clipboard detection, base64 encoding, and multimodal content.
"""

import base64
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from deepagents_cli.input import ImageTracker, MediaTracker
from deepagents_cli.media_utils import (
    ImageData,
    VideoData,
    create_multimodal_content,
    encode_to_base64,
    get_clipboard_image,
    get_image_from_path,
    get_video_from_path,
)


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

        assert tracker.next_image_id == 3
        assert len(tracker.images) == 2

        tracker.clear()

        assert tracker.next_image_id == 1
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
        assert tracker.next_image_id == 1

    def test_sync_to_text_keeps_referenced_images(self) -> None:
        """Sync should prune unreferenced images while preserving next ID order."""
        tracker = ImageTracker()
        img1 = ImageData(base64_data="abc", format="png", placeholder="")
        img2 = ImageData(base64_data="def", format="png", placeholder="")

        tracker.add_image(img1)
        tracker.add_image(img2)
        tracker.sync_to_text("keep [image 2] only")

        assert tracker.next_image_id == 3
        assert len(tracker.images) == 1
        assert tracker.images[0].placeholder == "[image 2]"


class TestEncodeImageToBase64:
    """Tests for base64 encoding."""

    def test_encode_image_bytes(self) -> None:
        """Test encoding raw bytes to base64."""
        test_bytes = b"test image data"
        result = encode_to_base64(test_bytes)

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

        result = encode_to_base64(png_bytes)

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

    @patch("deepagents_cli.media_utils.sys.platform", "linux")
    def test_unsupported_platform_returns_none_and_warns(self) -> None:
        """Test that non-macOS platforms return None and log a warning."""
        with patch("deepagents_cli.media_utils.logger") as mock_logger:
            result = get_clipboard_image()
            assert result is None
            mock_logger.warning.assert_called_once()
            assert "linux" in mock_logger.warning.call_args[0][1]

    @patch("deepagents_cli.media_utils.sys.platform", "darwin")
    @patch("deepagents_cli.media_utils._get_macos_clipboard_image")
    def test_macos_calls_macos_function(self, mock_macos_fn: MagicMock) -> None:
        """Test that macOS platform calls the macOS-specific function."""
        mock_macos_fn.return_value = None
        get_clipboard_image()
        mock_macos_fn.assert_called_once()

    @patch("deepagents_cli.media_utils.sys.platform", "darwin")
    @patch("deepagents_cli.media_utils.subprocess.run")
    @patch("deepagents_cli.media_utils._get_executable")
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

    @patch("deepagents_cli.media_utils.sys.platform", "darwin")
    @patch("deepagents_cli.media_utils.subprocess.run")
    @patch("deepagents_cli.media_utils._get_executable")
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

    @patch("deepagents_cli.media_utils.sys.platform", "darwin")
    @patch("deepagents_cli.media_utils._get_clipboard_via_osascript")
    @patch("deepagents_cli.media_utils.subprocess.run")
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
        assert tracker.next_image_id == 4


class TestVideoData:
    """Tests for VideoData dataclass."""

    def test_to_message_content_mp4(self) -> None:
        """Test converting MP4 video data to LangChain file block format."""
        video = VideoData(
            base64_data="AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            format="mp4",
            placeholder="[video 1]",
        )
        result = video.to_message_content()

        assert result["type"] == "file"
        assert result["base64"] == video.base64_data
        assert result["mime_type"] == "video/mp4"

    def test_to_message_content_mov(self) -> None:
        """Test converting MOV video data to LangChain file block format."""
        video = VideoData(
            base64_data="abc123",
            format="quicktime",
            placeholder="[video 2]",
        )
        result = video.to_message_content()

        assert result["type"] == "file"
        assert result["mime_type"] == "video/quicktime"


class TestGetVideoFromPath:
    """Tests for loading video files from disk."""

    def test_get_video_from_path_mp4(self, tmp_path: Path) -> None:
        """Valid MP4 files should be returned as VideoData."""
        # Create a minimal valid MP4 file (ftyp box)
        mp4_content = (
            b"\x00\x00\x00\x14"  # box size (20 bytes)
            b"ftyp"  # box type
            b"mp42"  # major brand
            b"\x00\x00\x00\x00"  # minor version
            b"mp42"  # compatible brand
        )
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(mp4_content)

        result = get_video_from_path(video_path)

        assert result is not None
        assert result.format == "mp4"
        assert result.placeholder == "[video]"
        assert base64.b64decode(result.base64_data) == mp4_content

    def test_get_video_from_path_jpg_returns_none(self, tmp_path: Path) -> None:
        """Non-video files should return None."""
        file_path = tmp_path / "test.jpg"
        file_path.write_bytes(b"fake jpg content")

        assert get_video_from_path(file_path) is None

    def test_get_video_from_path_txt_returns_none(self, tmp_path: Path) -> None:
        """Text files should return None."""
        file_path = tmp_path / "test.txt"
        file_path.write_bytes(b"not a video")

        assert get_video_from_path(file_path) is None

    def test_get_video_from_path_missing_returns_none(self, tmp_path: Path) -> None:
        """Missing files should return None."""
        file_path = tmp_path / "missing.mp4"
        assert get_video_from_path(file_path) is None

    def test_get_video_from_path_oversized_returns_none(self, tmp_path: Path) -> None:
        """Videos exceeding the size limit should be rejected."""
        video_path = tmp_path / "huge.mp4"
        # Create a file that reports > 20 MB via stat
        # Use a sparse approach: write header then seek to create large file
        with video_path.open("wb") as f:
            # Valid ftyp header
            f.write(b"\x00\x00\x00\x14ftypmp42\x00\x00\x00\x00mp42")
            # Pad to exceed 20 MB
            f.seek(21 * 1024 * 1024)
            f.write(b"\x00")

        assert get_video_from_path(video_path) is None

    def test_get_video_from_path_invalid_signature_returns_none(
        self, tmp_path: Path
    ) -> None:
        """Files with valid video extension but invalid signature should be rejected."""
        video_path = tmp_path / "fake.mp4"
        video_path.write_bytes(b"this is not a real video file at all")

        assert get_video_from_path(video_path) is None

    def test_get_video_from_path_mov(self, tmp_path: Path) -> None:
        """MOV files should be detected correctly."""
        # MOV files also use ftyp
        mov_content = (
            b"\x00\x00\x00\x14"  # box size
            b"ftyp"  # box type
            b"qt  "  # major brand (QuickTime)
            b"\x00\x00\x00\x00"  # minor version
            b"qt  "  # compatible brand
        )
        video_path = tmp_path / "test.mov"
        video_path.write_bytes(mov_content)

        result = get_video_from_path(video_path)

        assert result is not None
        assert result.format == "quicktime"


class TestMediaTrackerVideo:
    """Tests for MediaTracker video functionality."""

    def test_add_video_increments_counter(self) -> None:
        """Test that adding videos increments the video counter correctly."""
        tracker = MediaTracker()

        vid1 = VideoData(base64_data="abc", format="mp4", placeholder="")
        vid2 = VideoData(base64_data="def", format="mp4", placeholder="")

        placeholder1 = tracker.add_video(vid1)
        placeholder2 = tracker.add_video(vid2)

        assert placeholder1 == "[video 1]"
        assert placeholder2 == "[video 2]"
        assert vid1.placeholder == "[video 1]"
        assert vid2.placeholder == "[video 2]"

    def test_get_videos_returns_copy(self) -> None:
        """Test that get_videos returns a copy, not the original list."""
        tracker = MediaTracker()
        vid = VideoData(base64_data="abc", format="mp4", placeholder="")
        tracker.add_video(vid)

        videos = tracker.get_videos()
        videos.clear()  # Modify the returned list

        # Original should be unchanged
        assert len(tracker.get_videos()) == 1

    def test_clear_resets_video_counter(self) -> None:
        """Test that clear resets both videos and video counter."""
        tracker = MediaTracker()
        vid = VideoData(base64_data="abc", format="mp4", placeholder="")
        tracker.add_video(vid)
        tracker.add_video(vid)

        assert tracker.next_video_id == 3
        assert len(tracker.videos) == 2

        tracker.clear()

        assert tracker.next_video_id == 1
        assert len(tracker.videos) == 0

    def test_add_video_after_clear_starts_at_one(self) -> None:
        """Test that adding video after clear starts from [video 1] again."""
        tracker = MediaTracker()
        vid = VideoData(base64_data="abc", format="mp4", placeholder="")

        tracker.add_video(vid)
        tracker.add_video(vid)
        tracker.clear()

        new_vid = VideoData(base64_data="xyz", format="mp4", placeholder="")
        placeholder = tracker.add_video(new_vid)

        assert placeholder == "[video 1]"

    def test_sync_to_text_prunes_unreferenced_videos(self) -> None:
        """Sync should prune unreferenced videos while preserving video ID order."""
        tracker = MediaTracker()

        vid1 = VideoData(base64_data="abc", format="mp4", placeholder="")
        vid2 = VideoData(base64_data="def", format="mp4", placeholder="")

        tracker.add_video(vid1)
        tracker.add_video(vid2)
        tracker.sync_to_text("keep [video 2] only")

        assert tracker.next_video_id == 3
        assert len(tracker.videos) == 1
        assert tracker.videos[0].placeholder == "[video 2]"

    def test_image_and_video_tracking_work_together(self) -> None:
        """Test that images and videos can be tracked independently."""
        tracker = MediaTracker()

        img = ImageData(base64_data="img", format="png", placeholder="")
        vid = VideoData(base64_data="vid", format="mp4", placeholder="")

        img_placeholder = tracker.add_image(img)
        vid_placeholder = tracker.add_video(vid)

        assert img_placeholder == "[image 1]"
        assert vid_placeholder == "[video 1]"
        assert len(tracker.images) == 1
        assert len(tracker.videos) == 1

    def test_sync_to_text_handles_both_images_and_videos(self) -> None:
        """Sync should handle both image and video placeholders."""
        tracker = MediaTracker()

        img = ImageData(base64_data="img", format="png", placeholder="")
        vid = VideoData(base64_data="vid", format="mp4", placeholder="")

        tracker.add_image(img)
        tracker.add_video(vid)
        tracker.sync_to_text("[image 1] and [video 1]")

        assert len(tracker.images) == 1
        assert len(tracker.videos) == 1

    def test_sync_to_text_clears_all_when_no_placeholders(self) -> None:
        """Sync with no placeholders should clear both images and videos."""
        tracker = MediaTracker()

        img = ImageData(base64_data="img", format="png", placeholder="")
        vid = VideoData(base64_data="vid", format="mp4", placeholder="")

        tracker.add_image(img)
        tracker.add_video(vid)
        tracker.sync_to_text("no media here")

        assert len(tracker.images) == 0
        assert len(tracker.videos) == 0
        assert tracker.next_image_id == 1
        assert tracker.next_video_id == 1


class TestCreateMultimodalContentWithVideo:
    """Tests for creating multimodal content with videos."""

    def test_text_and_video(self) -> None:
        """Test creating content with text and one video."""
        vid = VideoData(base64_data="abc", format="mp4", placeholder="[video 1]")
        result = create_multimodal_content("Analyze this:", [], [vid])

        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "file"

    def test_text_image_and_video(self) -> None:
        """Test creating content with text, image, and video."""
        img = ImageData(base64_data="img", format="png", placeholder="[image 1]")
        vid = VideoData(base64_data="vid", format="mp4", placeholder="[video 1]")
        result = create_multimodal_content("Compare:", [img], [vid])

        assert len(result) == 3
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"
        assert result[2]["type"] == "file"

    def test_video_only(self) -> None:
        """Test that empty text is not included when only video is present."""
        vid = VideoData(base64_data="vid", format="mp4", placeholder="[video 1]")
        result = create_multimodal_content("", [], [vid])

        assert len(result) == 1
        assert result[0]["type"] == "file"

    def test_multiple_videos(self) -> None:
        """Test creating content with multiple videos."""
        vid1 = VideoData(base64_data="vid1", format="mp4", placeholder="[video 1]")
        vid2 = VideoData(
            base64_data="vid2",
            format="quicktime",
            placeholder="[video 2]",
        )
        result = create_multimodal_content("Compare these videos:", [], [vid1, vid2])

        assert len(result) == 3
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "file"
        assert result[2]["type"] == "file"
