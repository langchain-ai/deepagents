"""Tests for media utilities.

Covers clipboard detection, base64 encoding, and multimodal content.
"""

import base64
import io
from itertools import pairwise
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from deepagents_code.input import (
    MAX_DETACHED_MEDIA,
    MAX_DETACHED_MEDIA_BYTES,
    MediaTracker,
)
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


class TestMediaTracker:
    """Tests for MediaTracker class."""

    def test_add_image_increments_counter(self) -> None:
        """Test that adding images increments the counter correctly."""
        tracker = MediaTracker()

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
        tracker = MediaTracker()
        img = ImageData(base64_data="abc", format="png", placeholder="")
        tracker.add_image(img)

        images = tracker.get_images()
        images.clear()  # Modify the returned list

        # Original should be unchanged
        assert len(tracker.get_images()) == 1

    def test_clear_resets_counter(self) -> None:
        """Test that clear resets both images and counter."""
        tracker = MediaTracker()
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
        tracker = MediaTracker()
        img = ImageData(base64_data="abc", format="png", placeholder="")

        tracker.add_image(img)
        tracker.add_image(img)
        tracker.clear()

        new_img = ImageData(base64_data="xyz", format="png", placeholder="")
        placeholder = tracker.add_image(new_img)

        assert placeholder == "[image 1]"

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

    def test_sync_to_text_detaches_images_and_keeps_ids(self) -> None:
        """Removing every placeholder detaches the payloads and reserves their IDs.

        The counter deliberately does not reset: a reused ID would let an undo
        that restores an old token bind it to a newer payload.
        """
        tracker = MediaTracker()

        tracker.add_image(ImageData(base64_data="abc", format="png", placeholder=""))
        tracker.add_image(ImageData(base64_data="def", format="png", placeholder=""))
        tracker.sync_to_text("")

        assert tracker.images == []
        assert tracker.next_image_id == 3

    def test_sync_to_text_keeps_referenced_images(self) -> None:
        """Sync should prune unreferenced images while preserving next ID order."""
        tracker = MediaTracker()
        img1 = ImageData(base64_data="abc", format="png", placeholder="")
        img2 = ImageData(base64_data="def", format="png", placeholder="")

        tracker.add_image(img1)
        tracker.add_image(img2)
        tracker.sync_to_text("keep [image 2] only")

        assert tracker.next_image_id == 3
        assert len(tracker.images) == 1
        assert tracker.images[0].placeholder == "[image 2]"

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

    def test_sync_to_text_reattaches_image_when_placeholder_returns(self) -> None:
        """An undo that restores a deleted token re-attaches its image.

        Regression: deleting `[image 1]` dropped the attachment permanently, so
        ctrl+z restored the text without the image — the token stopped deleting
        atomically and the image never reached the model.
        """
        tracker = MediaTracker()
        img = ImageData(base64_data="abc", format="png", placeholder="")
        tracker.add_image(img)
        tracker.sync_to_text("[image 1]")

        edit_token = object()
        tracker.sync_to_text("", previous_text="[image 1]", edit_token=edit_token)
        assert tracker.get_images() == []

        tracker.sync_to_text(
            "[image 1]",
            previous_text="",
            undo_previous_text="",
            undo_token=edit_token,
        )

        assert len(tracker.get_images()) == 1
        assert tracker.get_images()[0].base64_data == "abc"
        assert tracker.next_image_id == 2

    def test_sync_to_text_reattaches_video_when_placeholder_returns(self) -> None:
        """An undo that restores a deleted token re-attaches its video."""
        tracker = MediaTracker()
        vid = VideoData(base64_data="abc", format="mp4", placeholder="")
        tracker.add_video(vid)
        tracker.sync_to_text("[video 1]")

        edit_token = object()
        tracker.sync_to_text("", previous_text="[video 1]", edit_token=edit_token)
        assert tracker.get_videos() == []

        tracker.sync_to_text(
            "[video 1]",
            previous_text="",
            undo_previous_text="",
            undo_token=edit_token,
        )

        assert len(tracker.get_videos()) == 1
        assert tracker.get_videos()[0].base64_data == "abc"

    def test_sync_to_text_reattaches_only_the_undone_kind(self) -> None:
        """Deleting one kind leaves the other attached, and undo restores it.

        The `self.clear()` call removed from `sync_to_text` was the only code
        that touched both kinds at once, so a mixed draft pins that its per-kind
        replacements are correct. (`MediaTracker.clear()` itself still exists and
        is still called on the agent dispatch path.)
        """
        tracker = MediaTracker()
        tracker.add_image(ImageData(base64_data="img", format="png", placeholder=""))
        tracker.add_video(VideoData(base64_data="vid", format="mp4", placeholder=""))
        full_text = "[image 1] [video 1]"
        tracker.sync_to_text(full_text)

        edit_token = object()
        tracker.sync_to_text(
            "[image 1]", previous_text=full_text, edit_token=edit_token
        )
        assert [img.base64_data for img in tracker.get_images()] == ["img"]
        assert tracker.get_videos() == []

        tracker.sync_to_text(
            full_text,
            previous_text="[image 1]",
            undo_previous_text="[image 1]",
            undo_token=edit_token,
        )

        assert [img.base64_data for img in tracker.get_images()] == ["img"]
        assert [vid.base64_data for vid in tracker.get_videos()] == ["vid"]

    def test_sync_to_text_reattached_image_keeps_id_order(self) -> None:
        """A re-attached image is ordered by placeholder ID, not re-attach order."""
        tracker = MediaTracker()
        img1 = ImageData(base64_data="one", format="png", placeholder="")
        img2 = ImageData(base64_data="two", format="png", placeholder="")
        tracker.add_image(img1)
        tracker.add_image(img2)
        tracker.sync_to_text("[image 1] [image 2]")

        edit_token = object()
        tracker.sync_to_text(
            "[image 2]",
            previous_text="[image 1] [image 2]",
            edit_token=edit_token,
        )
        tracker.sync_to_text(
            "[image 1] [image 2]",
            previous_text="[image 2]",
            undo_previous_text="[image 2]",
            undo_token=edit_token,
        )

        assert [img.base64_data for img in tracker.get_images()] == ["one", "two"]

    def test_add_image_reserves_detached_placeholder_for_undo(self) -> None:
        """A replacement cannot reuse an ID retained by undo history."""
        tracker = MediaTracker()
        old = ImageData(base64_data="old", format="png", placeholder="")
        tracker.add_image(old)
        tracker.sync_to_text("[image 1]")
        edit_token = object()
        tracker.sync_to_text("", previous_text="[image 1]", edit_token=edit_token)
        assert tracker.next_image_id == 2

        new = ImageData(base64_data="new", format="png", placeholder="")
        assert tracker.add_image(new) == "[image 2]"
        tracker.sync_to_text("", previous_text="[image 2]")
        tracker.sync_to_text(
            "[image 1]",
            previous_text="",
            undo_previous_text="",
            undo_token=edit_token,
        )

        assert [img.base64_data for img in tracker.get_images()] == ["old"]

    def test_manually_reinserted_placeholder_does_not_reattach_media(self) -> None:
        """Typing a deleted media token cannot reclaim its detached payload."""
        tracker = MediaTracker()
        tracker.add_image(ImageData(base64_data="secret", format="png", placeholder=""))
        tracker.sync_to_text("[image 1]")
        tracker.sync_to_text("", previous_text="[image 1]")

        tracker.sync_to_text("[image 1]", previous_text="")

        assert tracker.get_images() == []

    def test_undo_of_later_literal_edit_does_not_reattach_media(self) -> None:
        """An undo marker must match the edit which detached the payload."""
        tracker = MediaTracker()
        tracker.add_image(ImageData(base64_data="secret", format="png", placeholder=""))
        tracker.sync_to_text("[image 1]")
        deletion_token = object()
        tracker.sync_to_text("", previous_text="[image 1]", edit_token=deletion_token)

        tracker.sync_to_text(
            "[image 1]",
            previous_text="[image 1",
            undo_previous_text="[image 1",
            undo_token=object(),
        )

        assert tracker.get_images() == []

    def test_detached_pool_cap_evicts_oldest_and_strands_its_token(self) -> None:
        """Past the count cap, only the newest payloads survive an undo.

        The evicted tokens can still come back as text, so they are reported as
        stranded rather than silently shipped as bare `[image N]` text.
        """
        total = MAX_DETACHED_MEDIA + 3
        tracker = MediaTracker()
        placeholders = []
        for index in range(total):
            img = ImageData(base64_data=str(index), format="png", placeholder="")
            placeholders.append(tracker.add_image(img))
        full_text = " ".join(placeholders)
        tracker.sync_to_text(full_text)
        assert len(tracker.get_images()) == total

        edit_token = object()
        for index in range(total):
            tracker.sync_to_text(
                " ".join(placeholders[index + 1 :]),
                previous_text=" ".join(placeholders[index:]),
                edit_token=edit_token,
            )
        tracker.sync_to_text(
            full_text,
            previous_text="",
            undo_previous_text="",
            undo_token=edit_token,
        )

        assert [img.base64_data for img in tracker.get_images()] == [
            str(index) for index in range(3, total)
        ]
        assert tracker.take_stranded_placeholders() == [
            "[image 1]",
            "[image 2]",
            "[image 3]",
        ]

    def test_detached_pool_cap_is_bounded_by_bytes(self) -> None:
        """Oversized payloads are evicted by byte budget, not just by count."""
        tracker = MediaTracker()
        half_budget = "x" * (MAX_DETACHED_MEDIA_BYTES // 2 + 1)
        placeholders = [
            tracker.add_image(
                ImageData(base64_data=half_budget, format="png", placeholder="")
            )
            for _ in range(3)
        ]
        full_text = " ".join(placeholders)
        tracker.sync_to_text(full_text)

        edit_token = object()
        tracker.sync_to_text("", previous_text=full_text, edit_token=edit_token)
        tracker.sync_to_text(
            full_text,
            previous_text="",
            undo_previous_text="",
            undo_token=edit_token,
        )

        # Two payloads exceed the budget, so only the newest is restorable even
        # though the count cap would have allowed all three.
        assert [img.placeholder for img in tracker.get_images()] == [placeholders[-1]]
        assert tracker.take_stranded_placeholders() == placeholders[:2]

    def test_detached_pool_always_keeps_the_newest_payload(self) -> None:
        """The most recent delete stays undoable even if it alone busts the budget."""
        tracker = MediaTracker()
        oversized = "x" * (MAX_DETACHED_MEDIA_BYTES + 1)
        placeholder = tracker.add_image(
            ImageData(base64_data=oversized, format="png", placeholder="")
        )
        tracker.sync_to_text(placeholder)

        edit_token = object()
        tracker.sync_to_text("", previous_text=placeholder, edit_token=edit_token)
        tracker.sync_to_text(
            placeholder,
            previous_text="",
            undo_previous_text="",
            undo_token=edit_token,
        )

        assert [img.placeholder for img in tracker.get_images()] == [placeholder]
        assert tracker.take_stranded_placeholders() == []

    def test_stranded_placeholder_reported_once_per_undo(self) -> None:
        """A stranded token is surfaced once per undo, then drained."""
        tracker = MediaTracker()
        oversized = "x" * (MAX_DETACHED_MEDIA_BYTES + 1)
        first = tracker.add_image(
            ImageData(base64_data=oversized, format="png", placeholder="")
        )
        second = tracker.add_image(
            ImageData(base64_data=oversized, format="png", placeholder="")
        )
        full_text = f"{first} {second}"
        tracker.sync_to_text(full_text)
        edit_token = object()
        tracker.sync_to_text("", previous_text=full_text, edit_token=edit_token)

        tracker.sync_to_text(
            full_text,
            previous_text="",
            undo_previous_text="",
            undo_token=edit_token,
        )

        assert tracker.take_stranded_placeholders() == [first]
        assert tracker.take_stranded_placeholders() == []

    def test_evicted_token_not_stranded_until_an_undo_restores_it(self) -> None:
        """Eviction alone raises nothing; only a failed undo restore does.

        Placeholder-shaped text the user types is ordinary text, so it must not
        trigger a "media is gone" warning.
        """
        tracker = MediaTracker()
        oversized = "x" * (MAX_DETACHED_MEDIA_BYTES + 1)
        first = tracker.add_image(
            ImageData(base64_data=oversized, format="png", placeholder="")
        )
        second = tracker.add_image(
            ImageData(base64_data=oversized, format="png", placeholder="")
        )
        full_text = f"{first} {second}"
        tracker.sync_to_text(full_text)
        tracker.sync_to_text("", previous_text=full_text)

        assert tracker.take_stranded_placeholders() == []

        # Typed, not undone: stays literal text and stays silent.
        tracker.sync_to_text(full_text, previous_text="")

        assert tracker.take_stranded_placeholders() == []

    def test_clear_releases_detached_media(self) -> None:
        """`clear()` releases detached payloads so a consumed message frees them."""
        tracker = MediaTracker()
        tracker.add_image(ImageData(base64_data="abc", format="png", placeholder=""))
        tracker.sync_to_text("[image 1]")
        tracker.sync_to_text("", previous_text="[image 1]")
        assert tracker._detached_images

        tracker.clear()
        tracker.sync_to_text("[image 1]", previous_text="", undo_previous_text="")

        assert tracker.get_images() == []

    def test_release_detached_keeps_attached_media(self) -> None:
        """`release_detached()` frees undo payloads without dropping attachments."""
        tracker = MediaTracker()
        tracker.add_image(ImageData(base64_data="kept", format="png", placeholder=""))
        tracker.add_image(ImageData(base64_data="gone", format="png", placeholder=""))
        tracker.sync_to_text("[image 1] [image 2]")
        tracker.sync_to_text("[image 1]", previous_text="[image 1] [image 2]")

        tracker.release_detached()
        tracker.sync_to_text(
            "[image 1] [image 2]",
            previous_text="[image 1]",
            undo_previous_text="[image 1]",
        )

        assert [img.base64_data for img in tracker.get_images()] == ["kept"]

    def test_snapshot_and_restore_carry_attached_media_only(self) -> None:
        """Snapshot/restore round-trips attachments and drops undo payloads.

        Submission replaces the draft via `clear_text`, which resets the undo
        history, so a detached payload carried in the snapshot could never be
        restored — it would only pin base64 on every transcript message.
        """
        tracker = MediaTracker()
        tracker.add_image(ImageData(base64_data="kept", format="png", placeholder=""))
        tracker.add_image(ImageData(base64_data="gone", format="png", placeholder=""))
        tracker.sync_to_text("[image 1] [image 2]")
        tracker.sync_to_text("[image 1]", previous_text="[image 1] [image 2]")

        snapshot = tracker.snapshot()
        tracker.clear()
        tracker.restore(snapshot)

        assert [img.base64_data for img in tracker.get_images()] == ["kept"]

        tracker.sync_to_text(
            "[image 1] [image 2]",
            previous_text="[image 1]",
            undo_previous_text="[image 1]",
        )

        assert [img.base64_data for img in tracker.get_images()] == ["kept"]

    def test_restore_releases_detached_without_a_preceding_clear(self) -> None:
        """`restore()` frees the abandoned draft's undo payloads on its own.

        The interrupt-restore path calls `restore()` directly, without the
        `clear()` that other tests happen to perform first. The restored text
        arrives with a fresh undo history, so a payload held for the abandoned
        draft is unreachable and must not survive to bind a same-numbered token
        in the restored message.
        """
        interrupted = MediaTracker()
        interrupted.add_image(
            ImageData(base64_data="restored", format="png", placeholder="")
        )
        interrupted.sync_to_text("[image 1]")
        snapshot = interrupted.snapshot()

        tracker = MediaTracker()
        tracker.add_image(
            ImageData(base64_data="abandoned", format="png", placeholder="")
        )
        tracker.sync_to_text("[image 1]")
        marker = object()
        tracker.sync_to_text(
            "", previous_text="[image 1]", edit_token=marker, undo_token=marker
        )
        assert tracker._detached_images

        tracker.restore(snapshot)

        assert tracker._detached_images == []
        # Even a genuine undo of that exact edit cannot resurrect it.
        tracker.sync_to_text(
            "[image 1]",
            previous_text="",
            undo_previous_text="",
            edit_token=marker,
            undo_token=marker,
        )
        assert [img.base64_data for img in tracker.get_images()] == ["restored"]

    def test_take_stranded_placeholders_orders_ids_numerically(self) -> None:
        """Stranded tokens are listed in ID order, not lexicographic order.

        A lexicographic sort puts `[image 10]` before `[image 2]`, which reads as
        a mistake in the warning the user sees.
        """
        tracker = MediaTracker()
        tracker._stranded_placeholders.update(
            {"[image 10]", "[image 2]", "[image 3]", "[video 2]"}
        )

        assert tracker.take_stranded_placeholders() == [
            "[image 2]",
            "[image 3]",
            "[image 10]",
            "[video 2]",
        ]
        assert tracker.take_stranded_placeholders() == []

    def test_kind_state_keeps_attached_and_detached_disjoint(self) -> None:
        """The per-kind state never lists one token as both attached and detached.

        `_rebind` is the single writer of that invariant, so a partition that
        violated it must be corrected there rather than reaching the model as a
        payload that is simultaneously live and held for undo.
        """
        from deepagents_code.input import (
            IMAGE_PLACEHOLDER_PATTERN,
            _MediaKindState,
            _MediaPartition,
        )

        state: _MediaKindState[ImageData] = _MediaKindState(
            IMAGE_PLACEHOLDER_PATTERN, "image"
        )
        live = ImageData(base64_data="live", format="png", placeholder="[image 1]")
        stale = ImageData(base64_data="stale", format="png", placeholder="[image 1]")

        state._rebind(
            _MediaPartition(
                attached=[live],
                detached=[stale],
                evicted=[],
                detached_edits={"[image 1]": object()},
                evicted_edits={},
            )
        )

        assert [img.base64_data for img in state.attached] == ["live"]
        assert state.detached == []
        assert state.detached_edits == {}
        # The counter still reserves the token that was handed out.
        assert state.next_id == 2

    def test_kind_state_ids_never_go_backwards(self) -> None:
        """A kind's next ID keeps rising while a payload holds its token.

        This is what stops a later paste from reusing an ID that an undo could
        still restore, which would bind a restored token to a different payload.
        """
        from deepagents_code.input import (
            IMAGE_PLACEHOLDER_PATTERN,
            _MediaKindState,
        )

        state: _MediaKindState[ImageData] = _MediaKindState(
            IMAGE_PLACEHOLDER_PATTERN, "image"
        )
        first = ImageData(base64_data="one", format="png")
        assert state.bind(first, "") == "[image 1]"
        assert state.next_id == 2

        marker = object()
        state.sync(
            "",
            equal_spans=None,
            previous_text="[image 1]",
            cursor_offset=None,
            edit_token=marker,
            undo_token=None,
        )
        assert [img.placeholder for img in state.detached] == ["[image 1]"]
        assert state.next_id == 2

        second = ImageData(base64_data="two", format="png")
        assert state.bind(second, "") == "[image 2]"

    def test_unchanged_spans_matches_a_full_diff(self) -> None:
        """Prefix/suffix trimming agrees with diffing the whole draft.

        The trimming exists purely for speed — a character-level diff over a
        whole draft is quadratic — so it must not change which occurrences count
        as carried over.
        """
        cases = [
            ("[image 1] tail", "head [image 1] tail"),
            ("abc", "abc"),
            ("", "[image 1]"),
            ("[image 1]", ""),
            ("a [image 1] b", "a [image 1] b [image 1]"),
            ("see [imaxge 1] here", "see [image 1] here"),
            ("[image 1][image 2]", "[image 2]"),
        ]
        for previous_text, text in cases:
            spans = MediaTracker._unchanged_spans(previous_text, text)
            # Every reported span must genuinely be common to both texts.
            for start, end in spans:
                assert text[start:end] in previous_text, (previous_text, text)
            # Spans are ascending and non-overlapping.
            assert spans == sorted(spans), (previous_text, text)
            for (_, first_end), (second_start, _) in pairwise(spans):
                assert first_end <= second_start, (previous_text, text)
            # Identical text is entirely unchanged.
            if previous_text == text and text:
                assert spans == [(0, len(text))]

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

    @patch("deepagents_code.media_utils.sys.platform", "linux")
    def test_unsupported_platform_returns_none_and_warns(self) -> None:
        """Test that non-macOS platforms return None and log a warning."""
        with patch("deepagents_code.media_utils.logger") as mock_logger:
            result = get_clipboard_image()
            assert result is None
            mock_logger.warning.assert_called_once()
            assert "linux" in mock_logger.warning.call_args[0][1]

    @patch("deepagents_code.media_utils.sys.platform", "darwin")
    @patch("deepagents_code.media_utils._get_macos_clipboard_image")
    def test_macos_calls_macos_function(self, mock_macos_fn: MagicMock) -> None:
        """Test that macOS platform calls the macOS-specific function."""
        mock_macos_fn.return_value = None
        get_clipboard_image()
        mock_macos_fn.assert_called_once()

    @patch("deepagents_code.media_utils.sys.platform", "darwin")
    @patch("deepagents_code.media_utils.subprocess.run")
    @patch("deepagents_code.media_utils._get_executable")
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

    @patch("deepagents_code.media_utils.sys.platform", "darwin")
    @patch("deepagents_code.media_utils.subprocess.run")
    @patch("deepagents_code.media_utils._get_executable")
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

    @patch("deepagents_code.media_utils.sys.platform", "darwin")
    @patch("deepagents_code.media_utils._get_clipboard_via_osascript")
    @patch("deepagents_code.media_utils.subprocess.run")
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
        # Loaders return unbound payloads; only `add_media` assigns a token.
        assert result.placeholder == ""
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

    def test_get_image_from_path_empty_returns_none(self, tmp_path: Path) -> None:
        """Empty image files should return None."""
        img_path = tmp_path / "empty.png"
        img_path.write_bytes(b"")

        assert get_image_from_path(img_path) is None

    def test_get_image_from_path_oversized_returns_none(self, tmp_path: Path) -> None:
        """Images exceeding the size limit should be rejected."""
        img_path = tmp_path / "huge.png"
        with img_path.open("wb") as f:
            # Write a valid PNG header then pad to exceed 20 MB
            img = Image.new("RGB", (4, 4), color="red")
            img.save(f, format="PNG")
            f.seek(21 * 1024 * 1024)
            f.write(b"\x00")

        assert get_image_from_path(img_path) is None


class TestSyncToTextWithIDGaps:
    """Tests for MediaTracker.sync_to_text with non-contiguous IDs."""

    def test_sync_to_text_with_id_gap_preserves_max_id(self) -> None:
        """Deleting the middle image should set next_id based on max surviving ID."""
        tracker = MediaTracker()
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
        # Loaders return unbound payloads; only `add_media` assigns a token.
        assert result.placeholder == ""
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

    def test_sync_to_text_detaches_both_kinds_when_no_placeholders(self) -> None:
        """Sync with no placeholders detaches both kinds, leaving IDs reserved."""
        tracker = MediaTracker()

        img = ImageData(base64_data="img", format="png", placeholder="")
        vid = VideoData(base64_data="vid", format="mp4", placeholder="")

        tracker.add_image(img)
        tracker.add_video(vid)
        tracker.sync_to_text("no media here")

        assert len(tracker.images) == 0
        assert len(tracker.videos) == 0
        assert tracker.next_image_id == 2
        assert tracker.next_video_id == 2


class TestCreateMultimodalContentWithVideo:
    """Tests for creating multimodal content with videos."""

    def test_text_and_video(self) -> None:
        """Test creating content with text and one video."""
        vid = VideoData(base64_data="abc", format="mp4", placeholder="[video 1]")
        result = create_multimodal_content("Analyze this:", [], [vid])

        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "video"

    def test_text_image_and_video(self) -> None:
        """Test creating content with text, image, and video."""
        img = ImageData(base64_data="img", format="png", placeholder="[image 1]")
        vid = VideoData(base64_data="vid", format="mp4", placeholder="[video 1]")
        result = create_multimodal_content("Compare:", [img], [vid])

        assert len(result) == 3
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"
        assert result[2]["type"] == "video"

    def test_video_only(self) -> None:
        """Test that empty text is not included when only video is present."""
        vid = VideoData(base64_data="vid", format="mp4", placeholder="[video 1]")
        result = create_multimodal_content("", [], [vid])

        assert len(result) == 1
        assert result[0]["type"] == "video"

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
        assert result[1]["type"] == "video"
        assert result[2]["type"] == "video"


class TestDetectVideoFormat:
    """Tests for _detect_video_format magic-byte detection."""

    def test_mp4_ftyp_mp42(self) -> None:
        """MP4 ftyp box with mp42 brand returns 'mp4'."""
        data = b"\x00\x00\x00\x14ftypmp42\x00\x00\x00\x00"
        assert _detect_video_format(data) == "mp4"

    def test_mp4_ftyp_isom(self) -> None:
        """MP4 ftyp box with isom brand returns 'mp4'."""
        data = b"\x00\x00\x00\x14ftypisom\x00\x00\x00\x00"
        assert _detect_video_format(data) == "mp4"

    def test_mov_ftyp_qt(self) -> None:
        """MOV ftyp box with 'qt  ' brand returns 'quicktime'."""
        data = b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00"
        assert _detect_video_format(data) == "quicktime"

    def test_avi_riff(self) -> None:
        """AVI RIFF header returns 'avi'."""
        data = b"RIFF\x00\x00\x00\x00AVI \x00\x00\x00\x00"
        assert _detect_video_format(data) == "avi"

    def test_wmv_asf(self) -> None:
        """WMV/ASF magic bytes return 'x-ms-wmv'."""
        data = b"\x30\x26\xb2\x75" + b"\x00" * 12
        assert _detect_video_format(data) == "x-ms-wmv"

    def test_webm_ebml(self) -> None:
        """WebM/EBML magic bytes return 'webm'."""
        data = b"\x1a\x45\xdf\xa3" + b"\x00" * 12
        assert _detect_video_format(data) == "webm"

    def test_garbage_returns_none(self) -> None:
        """Unrecognized bytes return None."""
        data = b"this is not a video file at all!!"
        assert _detect_video_format(data) is None

    def test_empty_returns_none(self) -> None:
        """Empty bytes return None."""
        assert _detect_video_format(b"") is None

    def test_short_riff_not_avi(self) -> None:
        """RIFF prefix with < 12 bytes should not match AVI."""
        data = b"RIFF\x00\x00\x00\x00"
        assert _detect_video_format(data) is None

    def test_riff_non_avi_subtype(self) -> None:
        """RIFF header with non-AVI subtype (e.g. WAVE) returns None."""
        data = b"RIFF\x00\x00\x00\x00WAVE\x00\x00\x00\x00"
        assert _detect_video_format(data) is None


class TestGetVideoFromPathEdgeCases:
    """Edge-case tests for get_video_from_path."""

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        """Zero-byte video file should return None."""
        video_path = tmp_path / "empty.mp4"
        video_path.write_bytes(b"")

        assert get_video_from_path(video_path) is None

    def test_too_small_file_returns_none(self, tmp_path: Path) -> None:
        """Video file smaller than minimum magic-byte length should return None."""
        video_path = tmp_path / "tiny.mp4"
        video_path.write_bytes(b"\x00\x00\x00\x01")

        assert get_video_from_path(video_path) is None
