"""Tests for media utilities.

Covers clipboard detection, base64 encoding, and multimodal content.
"""

import base64
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from deepagents_code.input import MediaTracker
from deepagents_code.media_utils import (
    ImageData,
    VideoData,
    _detect_video_format,
    create_multimodal_content,
    encode_to_base64,
    get_clipboard_image,
    get_image_from_path,
    get_video_from_path,
    is_media_path,
    strip_media_placeholders,
)


class TestIsMediaPath:
    """Tests for the extension-only media classifier."""

    def test_image_extension_is_media(self) -> None:
        """Image extensions are recognized regardless of case."""
        assert is_media_path(Path("/tmp/a.png"))
        assert is_media_path(Path("/tmp/a.JPG"))

    def test_video_extension_is_media(self) -> None:
        """Video extensions are recognized as media."""
        assert is_media_path(Path("/tmp/clip.mp4"))

    def test_non_media_extension_is_not_media(self) -> None:
        """Text and other files are not treated as media."""
        assert not is_media_path(Path("/tmp/notes.txt"))
        assert not is_media_path(Path("/tmp/no_extension"))


class TestImageData:
    """Tests for ImageData dataclass."""


class TestMediaTracker:
    """Tests for MediaTracker class."""

    def test_add_image_skips_placeholder_already_in_text(self) -> None:
        """A literal placeholder in the draft is not rebound to new media."""
        tracker = MediaTracker()
        img = ImageData(base64_data="abc", format="png", placeholder="")

        placeholder = tracker.add_image(img, existing_text="restore [image 1]")

        assert placeholder == "[image 2]"
        assert img.placeholder == "[image 2]"
        assert tracker.next_image_id == 3

    def test_add_video_skips_placeholder_already_in_text(self) -> None:
        """Video attachment IDs skip literal placeholders in the draft."""
        tracker = MediaTracker()
        vid = VideoData(base64_data="abc", format="mp4", placeholder="")

        placeholder = tracker.add_video(vid, existing_text="restore [video 1]")

        assert placeholder == "[video 2]"
        assert vid.placeholder == "[video 2]"
        assert tracker.next_video_id == 3

    def test_sync_to_text_tracks_duplicate_inserted_before_placeholder(self) -> None:
        """A typed duplicate before the display token does not steal the media span."""
        tracker = MediaTracker()
        img = ImageData(base64_data="abc", format="png", placeholder="")
        tracker.add_image(img)
        tracker.sync_to_text("[image 1]")

        text = "literal [image 1] then actual [image 1]"
        tracker.sync_to_text(text, previous_text="[image 1]", cursor_offset=30)

        assert img.placeholder_span == (30, 39)

    def test_sync_to_text_preserves_mapped_duplicate_after_prefix_edit(self) -> None:
        """Edits before duplicates keep the mapped display token bound to media."""
        tracker = MediaTracker()
        img = ImageData(base64_data="abc", format="png", placeholder="")
        tracker.add_image(img)
        tracker.sync_to_text("[image 1]")

        previous_text = "literal [image 1] actual [image 1]"
        tracker.sync_to_text(
            previous_text,
            previous_text="[image 1]",
            cursor_offset=25,
        )

        text = "Xliteral [image 1] actual [image 1]"
        tracker.sync_to_text(text, previous_text=previous_text, cursor_offset=1)
        result = create_multimodal_content(text, tracker.images)

        assert img.placeholder_span == (26, 35)
        assert result[0]["text"] == "Xliteral [image 1] actual"

    def test_sync_to_text_tracks_duplicate_inserted_after_placeholder(self) -> None:
        """A typed duplicate after the display token does not steal the media span."""
        tracker = MediaTracker()
        img = ImageData(base64_data="abc", format="png", placeholder="")
        tracker.add_image(img)
        tracker.sync_to_text("[image 1]")

        text = "[image 1] literal [image 1]"
        tracker.sync_to_text(text, previous_text="[image 1]")

        assert img.placeholder_span == (0, 9)

    def test_sync_to_text_edit_between_placeholder_and_duplicate_keeps_actual(
        self,
    ) -> None:
        """Typing between an actual placeholder and a duplicate does not rebind it."""
        tracker = MediaTracker()
        img = ImageData(base64_data="abc", format="png", placeholder="")
        tracker.add_image(img)
        tracker.sync_to_text("[image 1]")
        tracker.sync_to_text("[image 1] literal [image 1]", previous_text="[image 1]")

        text = "[image 1] edited literal [image 1]"
        tracker.sync_to_text(
            text,
            previous_text="[image 1] literal [image 1]",
            cursor_offset=17,
        )

        assert img.placeholder_span == (0, 9)

    def test_remap_spans_to_text_shifts_span_and_strips_correct_duplicate(
        self,
    ) -> None:
        """Remapping keeps the real (second) token bound after an offset shift.

        Regression: spans are captured against the draft but consumed after
        submit-time rewrites (whitespace trim, paste expansion) shift offsets.
        Without remapping, the stale span is discarded and the token-count
        fallback strips the *first* occurrence — the user's literal — leaving the
        real display token in the model-facing text. Remapping fixes the offset.
        """
        tracker = MediaTracker()
        img = ImageData(base64_data="abc", format="png", placeholder="")
        tracker.add_image(img)
        tracker.sync_to_text("[image 1]")

        # A literal duplicate precedes the real display token in the draft.
        draft = "see [image 1] then real [image 1]"
        tracker.sync_to_text(draft, previous_text="[image 1]", cursor_offset=24)
        assert img.placeholder_span == (24, 33)

        # Submit expands a paste before the token, shifting it right by 15 chars.
        final = "EXPANDED PASTE see [image 1] then real [image 1]"
        tracker.remap_spans_to_text(final, previous_text=draft)
        assert img.placeholder_span == (39, 48)

        result = create_multimodal_content(final, tracker.get_images())
        assert result[0]["text"] == "EXPANDED PASTE see [image 1] then real"

    def test_remap_spans_to_text_drops_span_when_token_edited(self) -> None:
        """A span whose token was edited in the transformed text becomes None."""
        tracker = MediaTracker()
        img = ImageData(base64_data="abc", format="png", placeholder="")
        tracker.add_image(img)
        tracker.sync_to_text("[image 1]")
        assert img.placeholder_span == (0, 9)

        # The token characters themselves changed: cannot be cleanly mapped.
        tracker.remap_spans_to_text("[image 2]", previous_text="[image 1]")
        assert img.placeholder_span is None

    def test_remap_spans_to_text_ignores_items_without_span(self) -> None:
        """Items with no captured span are left as None (token-fallback path)."""
        tracker = MediaTracker()
        img = ImageData(base64_data="abc", format="png", placeholder="[image 1]")
        tracker.images.append(img)  # attached but never span-synced

        tracker.remap_spans_to_text("[image 1]", previous_text="[image 1]")
        assert img.placeholder_span is None


class TestEncodeImageToBase64:
    """Tests for base64 encoding."""


class TestCreateMultimodalContent:
    """Tests for creating multimodal message content."""

    def test_empty_text_with_image(self) -> None:
        """Test that empty text is not included in content."""
        img = ImageData(base64_data="abc", format="png", placeholder="[image 1]")
        result = create_multimodal_content("", [img])

        # Should only have the image, no empty text block
        assert len(result) == 1
        assert result[0]["type"] == "image_url"

    def test_placeholder_not_leaked_into_text_block(self) -> None:
        """The `[image N]` display placeholder must not reach model-facing text.

        Regression test: sending an image previously serialized the display
        placeholder as literal user-authored text in the traced/model message.
        """
        img = ImageData(base64_data="abc", format="png", placeholder="[image 1]")
        result = create_multimodal_content("[image 1] what's in this image?", [img])

        assert len(result) == 2
        assert result[0]["type"] == "text"
        # Placeholder is gone but the surrounding user text is preserved.
        assert "[image 1]" not in result[0]["text"]
        assert result[0]["text"] == "what's in this image?"
        assert result[1]["type"] == "image_url"

    def test_literal_duplicate_placeholder_preserved_in_text_block(self) -> None:
        """Only the display placeholder occurrence is stripped from text."""
        img = ImageData(
            base64_data="abc",
            format="png",
            placeholder="[image 1]",
            placeholder_span=(0, 9),
        )
        result = create_multimodal_content(
            "[image 1] compare with literal [image 1]",
            [img],
        )

        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "compare with literal [image 1]"
        assert result[1]["type"] == "image_url"

    def test_literal_duplicate_before_placeholder_preserved_in_text_block(self) -> None:
        """A literal duplicate before the display placeholder is preserved."""
        img = ImageData(
            base64_data="abc",
            format="png",
            placeholder="[image 1]",
            placeholder_span=(30, 39),
        )
        result = create_multimodal_content(
            "literal [image 1] then actual [image 1]",
            [img],
        )

        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "literal [image 1] then actual"
        assert result[1]["type"] == "image_url"

    def test_placeholder_removed_when_only_placeholder(self) -> None:
        """A message that is only a placeholder yields no text block."""
        img = ImageData(base64_data="abc", format="png", placeholder="[image 1]")
        result = create_multimodal_content("[image 1]", [img])

        assert len(result) == 1
        assert result[0]["type"] == "image_url"

    def test_placeholders_removed_for_video(self) -> None:
        """Video placeholders are also stripped from model-facing text."""
        vid = VideoData(base64_data="vid", format="mp4", placeholder="[video 1]")
        result = create_multimodal_content("summarize [video 1] please", [], [vid])

        assert result[0]["type"] == "text"
        assert "[video 1]" not in result[0]["text"]
        assert result[0]["text"] == "summarize please"

    def test_multiple_placeholders_all_stripped(self) -> None:
        """All placeholders are removed while surrounding text is preserved."""
        img1 = ImageData(base64_data="a", format="png", placeholder="[image 1]")
        img2 = ImageData(base64_data="b", format="png", placeholder="[image 2]")
        result = create_multimodal_content(
            "[image 1] and [image 2] differ how?", [img1, img2]
        )

        assert result[0]["type"] == "text"
        assert "[image 1]" not in result[0]["text"]
        assert "[image 2]" not in result[0]["text"]
        assert result[0]["text"] == "and differ how?"

    def test_mixed_image_and_video_placeholders_stripped(self) -> None:
        """Image and video placeholders in one message are both stripped.

        Exercises the combined `images + videos` placeholder assembly and the
        heterogeneous `|`-token strip together, which same-type cases miss.
        """
        img = ImageData(base64_data="a", format="png", placeholder="[image 1]")
        vid = VideoData(base64_data="v", format="mp4", placeholder="[video 1]")
        result = create_multimodal_content("[image 1] vs [video 1]", [img], [vid])

        assert result[0]["type"] == "text"
        assert result[0]["text"] == "vs"
        # Both media blocks follow the single text block.
        assert result[1]["type"] == "image_url"
        assert result[2]["type"] == "video"

    def test_unbound_placeholder_like_text_preserved(self) -> None:
        """Placeholder-shaped text not bound to attached media is preserved.

        Regression test for false positives: a user attaches image 1 but their
        prompt also literally mentions `[image 2]` (which is not attached). Only
        the real `[image 1]` token is stripped; the literal `[image 2]` stays.
        """
        img = ImageData(base64_data="abc", format="png", placeholder="[image 1]")
        result = create_multimodal_content(
            "[image 1] see the note about [image 2] in the docs", [img]
        )

        assert result[0]["type"] == "text"
        assert result[0]["text"] == "see the note about [image 2] in the docs"

    def test_placeholder_like_text_without_attachment_untouched(self) -> None:
        """With no media attached, placeholder-shaped text is never stripped."""
        result = create_multimodal_content("compare [image 1] vs [image 2]", [])

        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "compare [image 1] vs [image 2]"


class TestStripMediaPlaceholders:
    """Tests for `strip_media_placeholders`."""

    def test_leading_placeholder(self) -> None:
        """A leading placeholder is removed along with its trailing space."""
        assert strip_media_placeholders("[image 1] hello", ["[image 1]"]) == "hello"

    def test_inline_placeholder_no_double_space(self) -> None:
        """An inline placeholder does not leave a double space behind."""
        assert (
            strip_media_placeholders("before [image 1] after", ["[image 1]"])
            == "before after"
        )

    def test_duplicate_literal_placeholder_preserved(self) -> None:
        """Duplicate literal text is preserved when one matching media is attached."""
        assert (
            strip_media_placeholders(
                "[image 1] describe literal [image 1]",
                ["[image 1]"],
                placeholder_spans=[(0, 9)],
            )
            == "describe literal [image 1]"
        )

    def test_literal_duplicate_before_placeholder_preserved(self) -> None:
        """Span tracking strips the display token, not the first duplicate."""
        assert (
            strip_media_placeholders(
                "literal [image 1] then actual [image 1]",
                ["[image 1]"],
                placeholder_spans=[(30, 39)],
            )
            == "literal [image 1] then actual"
        )

    def test_only_bound_placeholders_removed(self) -> None:
        """Only tokens in the bound set are removed; look-alikes are preserved."""
        result = strip_media_placeholders(
            "keep [image 2] drop [image 1]", ["[image 1]"]
        )
        assert result == "keep [image 2] drop"

    def test_stale_span_is_discarded_and_falls_back_to_token(self) -> None:
        """A span whose slice no longer matches the token is ignored, not used.

        Guards the graceful-degradation contract: an out-of-date span (here the
        offset points at non-token text) must not delete arbitrary characters;
        the token fallback removes the real occurrence instead.
        """
        result = strip_media_placeholders(
            "hello [image 1] world",
            ["[image 1]"],
            placeholder_spans=[(0, 5)],  # points at "hello", not the token
        )
        assert result == "hello world"

    def test_ambiguous_fallback_strips_leading_occurrence(self) -> None:
        """With no span and duplicate tokens, the fallback strips the first one."""
        result = strip_media_placeholders(
            "first [image 1] second [image 1]",
            ["[image 1]"],
        )
        assert result == "first second [image 1]"

    def test_mixed_image_and_video_tokens_removed(self) -> None:
        """Heterogeneous image and video tokens are stripped in one pass."""
        result = strip_media_placeholders(
            "[image 1] and [video 1] together", ["[image 1]", "[video 1]"]
        )
        assert result == "and together"

    def test_text_without_placeholder_unchanged(self) -> None:
        """Text without a bound placeholder is unchanged (aside from trim)."""
        assert (
            strip_media_placeholders("plain text only", ["[image 1]"])
            == "plain text only"
        )

    def test_no_placeholders_returns_text_verbatim(self) -> None:
        """An empty placeholder set returns the text unchanged, including whitespace."""
        assert strip_media_placeholders("  [image 1]  ", []) == "  [image 1]  "

    def test_only_placeholder_becomes_empty(self) -> None:
        """A string that is only a bound placeholder becomes empty."""
        assert strip_media_placeholders("[video 2]", ["[video 2]"]) == ""

    def test_newlines_preserved(self) -> None:
        """Newlines around a placeholder are preserved."""
        assert (
            strip_media_placeholders("line1\n[image 1]\nline2", ["[image 1]"])
            == "line1\n\nline2"
        )

    def test_special_regex_chars_in_placeholder_are_escaped(self) -> None:
        """Placeholder tokens are treated literally, not as regex."""
        assert strip_media_placeholders("a [image (1)] b", ["[image (1)]"]) == "a b"

    def test_indentation_preserved_after_placeholder(self) -> None:
        r"""Leading newlines and code indentation after a placeholder survive.

        Regression: ``.strip()`` removed all leading whitespace, so attaching
        an image before indented code (``[image 1]\n    def foo():``) would
        lose the four-space indent. Only spaces/tabs on each edge are trimmed.
        """
        assert (
            strip_media_placeholders("[image 1]\n    def foo():", ["[image 1]"])
            == "\n    def foo():"
        )


class TestGetClipboardImage:
    """Tests for clipboard image detection."""


class TestGetImageFromPath:
    """Tests for loading local images from dropped file paths."""


class TestSyncToTextWithIDGaps:
    """Tests for MediaTracker.sync_to_text with non-contiguous IDs."""


class TestVideoData:
    """Tests for VideoData dataclass."""

    def test_to_message_content_mp4(self) -> None:
        """Test converting MP4 video data to LangChain video block format."""
        video = VideoData(
            base64_data="AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            format="mp4",
            placeholder="[video 1]",
        )
        result = video.to_message_content()

        assert result["type"] == "video"
        assert result["base64"] == video.base64_data
        assert result["mime_type"] == "video/mp4"

    def test_to_message_content_mov(self) -> None:
        """Test converting MOV video data to LangChain video block format."""
        video = VideoData(
            base64_data="abc123",
            format="quicktime",
            placeholder="[video 2]",
        )
        result = video.to_message_content()

        assert result["type"] == "video"
        assert result["mime_type"] == "video/quicktime"


class TestGetVideoFromPath:
    """Tests for loading video files from disk."""


class TestMediaTrackerVideo:
    """Tests for MediaTracker video functionality."""


class TestCreateMultimodalContentWithVideo:
    """Tests for creating multimodal content with videos."""


class TestDetectVideoFormat:
    """Tests for _detect_video_format magic-byte detection."""


class TestGetVideoFromPathEdgeCases:
    """Edge-case tests for get_video_from_path."""
