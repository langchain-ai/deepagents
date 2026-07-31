"""Input handling utilities including image/video tracking and file mention parsing."""

import logging
import re
import shlex
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Generic, Literal, TypeVar
from urllib.parse import unquote, urlparse

from rich.markup import escape as escape_markup

from deepagents_code.config import console
from deepagents_code.media_utils import ImageData, VideoData

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

MediaKind = Literal["image", "video"]
"""Accepted values for the `kind` parameter in `MediaTracker` methods."""

IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[image (?P<id>\d+)\]")
"""Pattern for image placeholders with a named `id` capture group.

Used to extract numeric IDs from placeholder tokens so the tracker can prune
stale entries and compute the next available ID.
"""

VIDEO_PLACEHOLDER_PATTERN = re.compile(r"\[video (?P<id>\d+)\]")
"""Pattern for video placeholders with a named `id` capture group.

Used to extract numeric IDs from placeholder tokens so the tracker can prune
stale entries and compute the next available ID.
"""

_UNICODE_SPACE_EQUIVALENTS = str.maketrans(
    {
        "\u00a0": " ",  # NO-BREAK SPACE
        "\u202f": " ",  # NARROW NO-BREAK SPACE
    }
)
"""Translation table used to normalize Unicode space variants.

Some macOS-generated filenames (for example screenshots) may contain non-ASCII
space code points that look identical to normal spaces when pasted.
"""

_WINDOWS_DRIVE_PATH_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
"""Pattern for Windows drive-letter paths like `C:\\Users\\...`."""

_MediaItemT = TypeVar("_MediaItemT", ImageData, VideoData)
"""Media payload type for the tracker's partition helpers.

Value-restricted rather than bound so `_MediaItemT` can never solve to
`ImageData | VideoData`: a single heterogeneous list is rejected outright. Note
that a mismatched *pair* of lists (images passed alongside videos) is already
rejected by `list` invariance, with or without the value restriction.
"""

MAX_DETACHED_MEDIA = 10
"""Cap on the number of detached media items retained per kind for undo.

Guards against many small pastes; `MAX_DETACHED_MEDIA_BYTES` is the binding
limit for large ones. Both caps apply per kind and drop the oldest entries first,
stopping at the first payload that does not fit rather than best-fitting smaller
ones around it.
"""

MAX_DETACHED_MEDIA_BYTES = 32 * 1024 * 1024
"""Byte budget for detached payloads retained per kind for undo.

A single item can approach `media_utils.MAX_MEDIA_BYTES` (~27 MB once
base64-encoded), so a count-only cap of ten would admit hundreds of megabytes
per kind. Undo realistically needs only the most recent deletions, so the pool
drops oldest-first until it fits. The newest item is always retained even when
it alone exceeds the budget, so undoing the most recent delete never loses its
payload.
"""


@dataclass(frozen=True)
class _MediaPartition(Generic[_MediaItemT]):
    """Result of re-splitting one kind's media against the current draft.

    Named rather than a tuple because three of the four fields are same-typed
    lists: positionally, `attached` and `detached` are interchangeable, and
    swapping them would send the model the payloads the user deleted while
    suppressing the ones they kept — a mistake no type checker can catch.
    """

    attached: list[_MediaItemT]
    """Media bound to a placeholder present in the current text."""

    detached: list[_MediaItemT]
    """Media retained for undo after its placeholder left the text."""

    evicted: list[_MediaItemT]
    """Detached payloads dropped to stay within the pool caps."""

    evicted_edits: dict[str, object]
    """Detaching edit identity per evicted token, for stranded reporting."""


@dataclass(frozen=True)
class ParsedPastedPathPayload:
    """Unified parse result for dropped-path payload detection.

    Attributes:
        paths: Resolved file paths parsed from the input payload.
        token_end: End index (exclusive) of the parsed leading token when the
            payload starts with a path followed by trailing text.

            `None` means the entire payload was parsed as path-only content.
    """

    paths: list[Path]
    token_end: int | None = None


class MediaTracker:
    """Track pasted images and videos in the current conversation."""

    def __init__(self) -> None:
        """Initialize an empty media tracker.

        Sets up empty lists to store images and videos, and initializes the
        ID counters to 1 for generating unique placeholder identifiers.
        """
        self.images: list[ImageData] = []
        self.videos: list[VideoData] = []
        self.next_image_id: int = 1
        self.next_video_id: int = 1
        # Media whose placeholder was removed from the draft but whose payload is
        # retained so an undo (ctrl+z) of that edit re-attaches it. Whole-draft
        # clears do not come through here: both `ChatInput.discard_text` and
        # `_submit_value` bump `_skip_media_sync_events` so the sync is skipped.
        # History recall (`set_text_from_history`) *does* reach this, because its
        # `_skip_history_change_events` guard is checked after the media sync.
        self._detached_images: list[ImageData] = []
        self._detached_videos: list[VideoData] = []
        # Identity markers for the concrete TextArea edit batches that detached
        # each payload. The marker, rather than token text, proves that a later
        # undo reverses the edit which removed this exact attachment occurrence.
        self._detached_image_edits: dict[str, object] = {}
        self._detached_video_edits: dict[str, object] = {}
        # Identity markers whose payload the pool caps dropped, retained so
        # undoing that exact batch can report its media is no longer available.
        self._evicted_placeholder_edits: dict[str, object] = {}
        # Evicted tokens an undo has since restored as text. They look attached
        # but carry nothing, so they are reported to the user and then drained.
        self._stranded_placeholders: set[str] = set()

    def add_media(
        self,
        data: ImageData | VideoData,
        kind: MediaKind,
        *,
        existing_text: str = "",
    ) -> str:
        """Add a media item and return its placeholder text.

        IDs still held by a detached payload are also skipped. Reusing one would
        let an undo that restores the old token bind it to this new item,
        delivering a different image than the restored draft was written about.

        Args:
            data: The image or video data to track.
            kind: Media type key.
            existing_text: Current draft text. Placeholder IDs already present
                here are skipped so literal user text is not bound to new media.

        Returns:
            Placeholder string like "[image 1]" or "[video 1]".
        """
        if kind == "image":
            reserved = {item.placeholder for item in self._detached_images}
            while (
                f"[image {self.next_image_id}]" in existing_text
                or f"[image {self.next_image_id}]" in reserved
            ):
                self.next_image_id += 1
            placeholder = f"[image {self.next_image_id}]"
            data.placeholder = placeholder
            self.images.append(data)  # ty: ignore[invalid-argument-type]
            self.next_image_id += 1
        else:
            reserved = {item.placeholder for item in self._detached_videos}
            while (
                f"[video {self.next_video_id}]" in existing_text
                or f"[video {self.next_video_id}]" in reserved
            ):
                self.next_video_id += 1
            placeholder = f"[video {self.next_video_id}]"
            data.placeholder = placeholder
            self.videos.append(data)  # ty: ignore[invalid-argument-type]
            self.next_video_id += 1
        return placeholder

    def add_image(self, image_data: ImageData, *, existing_text: str = "") -> str:
        """Add an image and return its placeholder text.

        Args:
            image_data: The image data to track.
            existing_text: Current draft text. Placeholder IDs already present
                here are skipped so literal user text is not bound to new media.

        Returns:
            Placeholder string like "[image 1]".
        """
        return self.add_media(image_data, "image", existing_text=existing_text)

    def add_video(self, video_data: VideoData, *, existing_text: str = "") -> str:
        """Add a video and return its placeholder text.

        Args:
            video_data: The video data to track.
            existing_text: Current draft text. Placeholder IDs already present
                here are skipped so literal user text is not bound to new media.

        Returns:
            Placeholder string like "[video 1]".
        """
        return self.add_media(video_data, "video", existing_text=existing_text)

    def get_images(self) -> list[ImageData]:
        """Get all tracked images.

        Returns:
            Copy of the list of tracked images.
        """
        return list(self.images)

    def get_videos(self) -> list[VideoData]:
        """Get all tracked videos.

        Returns:
            Copy of the list of tracked videos.
        """
        return list(self.videos)

    def clear(self) -> None:
        """Clear all tracked media, detached undo payloads, and reset counters."""
        self.images.clear()
        self.videos.clear()
        self.release_detached()
        self.next_image_id = 1
        self.next_video_id = 1

    def release_detached(self) -> None:
        """Release detached undo payloads without touching attached media.

        Used on submission paths that never consume the draft's media (slash and
        shell commands), where the payloads would otherwise stay re-attachable
        into an unrelated later draft.
        """
        self._detached_images.clear()
        self._detached_videos.clear()
        self._detached_image_edits.clear()
        self._detached_video_edits.clear()
        self._evicted_placeholder_edits.clear()
        self._stranded_placeholders.clear()

    def take_stranded_placeholders(self) -> list[str]:
        """Return tokens an undo restored without their payload, then drain them.

        Reported per undo attempt rather than per keystroke, and only for tokens
        the undo actually tried to restore — placeholder-shaped text the user
        typed is ordinary text and must not raise a warning.

        Returns:
            Stranded placeholder tokens in ascending ID order, or an empty list
            if none. Ordered numerically rather than lexicographically so a
            warning listing `[image 2]` and `[image 10]` reads in draft order.
        """
        stranded = sorted(
            self._stranded_placeholders, key=self._placeholder_token_sort_key
        )
        self._stranded_placeholders.clear()
        return stranded

    @staticmethod
    def _placeholder_token_sort_key(token: str) -> tuple[str, int]:
        """Return a numeric ordering key for a placeholder token.

        Args:
            token: Placeholder token such as `[image 10]`.

        Returns:
            The token's kind and numeric ID, falling back to `(token, 0)` so
            unparsable tokens still order deterministically.
        """
        for pattern in (IMAGE_PLACEHOLDER_PATTERN, VIDEO_PLACEHOLDER_PATTERN):
            match = pattern.fullmatch(token)
            if match is not None:
                return (pattern.pattern, int(match.group("id")))
        return (token, 0)

    def _record_stranded(
        self,
        restored_spans: dict[str, list[tuple[int, int]]],
        attached: list[ImageData] | list[VideoData],
        undo_token: object | None,
    ) -> None:
        """Flag undo-restored tokens whose payload the pool had already evicted.

        Args:
            restored_spans: Token occurrences introduced by an actual undo.
            attached: Media bound after this sync, used to spot tokens that
                came back as text without a payload behind them.
            undo_token: Identity of the TextArea edit batch being undone.
        """
        if (
            not restored_spans
            or undo_token is None
            or not self._evicted_placeholder_edits
        ):
            return
        bound = {item.placeholder for item in attached}
        self._stranded_placeholders.update(
            token
            for token in restored_spans
            if token not in bound
            and self._evicted_placeholder_edits.get(token) is undo_token
        )

    def _record_evicted(
        self,
        evicted: list[ImageData] | list[VideoData],
        edit_tokens: dict[str, object],
    ) -> None:
        """Note payloads the pool caps dropped so their tokens can be reported.

        Args:
            evicted: Media items removed from the detached pool.
            edit_tokens: Detaching edit identities keyed by evicted token.
        """
        if not evicted:
            return
        tokens = [item.placeholder for item in evicted]
        self._evicted_placeholder_edits.update(edit_tokens)
        logger.warning(
            "Detached media pool full; dropped %d payload(s) %s. "
            "Undo can no longer restore the media behind these tokens.",
            len(tokens),
            ", ".join(tokens),
        )

    def snapshot(self) -> "MediaTracker":
        """Return an independent copy of the currently attached media.

        Detached undo payloads are deliberately excluded. Submission replaces
        the draft via `ChatTextArea.clear_text`, which resets the undo history,
        so no undo could reach them — carrying them would pin base64 payloads on
        every transcript message for the life of the session.
        """
        tracker = MediaTracker()
        tracker.images = [replace(img) for img in self.images]
        tracker.videos = [replace(vid) for vid in self.videos]
        tracker.next_image_id = self.next_image_id
        tracker.next_video_id = self.next_video_id
        return tracker

    def restore(self, snapshot: "MediaTracker") -> None:
        """Replace current media state with an independent snapshot copy.

        Any detached payloads held for the abandoned draft are released: the
        restored text comes with a fresh undo history, so they are unreachable.

        Args:
            snapshot: Previously captured media state to restore.
        """
        self.images = [replace(img) for img in snapshot.images]
        self.videos = [replace(vid) for vid in snapshot.videos]
        self.release_detached()
        self.next_image_id = snapshot.next_image_id
        self.next_video_id = snapshot.next_video_id

    def sync_to_text(
        self,
        text: str,
        *,
        previous_text: str | None = None,
        cursor_offset: int | None = None,
        undo_previous_text: str | None = None,
        edit_token: object | None = None,
        undo_token: object | None = None,
    ) -> None:
        """Re-bind tracked media to the placeholder tokens in the current text.

        Media whose placeholder disappeared is detached rather than dropped, and
        a detached item is re-attached only when an undo restores its exact token
        occurrence. Detached payloads are released by `clear()` when a message is
        dispatched to the agent, or by `release_detached()` on the slash and
        shell paths that never consume them.

        Args:
            text: Current input text shown to the user.
            previous_text: Previous input text, used to keep tracking the same
                placeholder occurrence when duplicate literal tokens are added.
            cursor_offset: Current cursor offset, used to disambiguate whole-paste
                edits that create duplicate placeholder tokens.
            undo_previous_text: Text immediately before an actual undo operation.
                Supplying this marks only placeholder occurrences inserted by that
                undo as eligible to reclaim detached payloads.
            edit_token: Identity of the TextArea edit batch that produced `text`.
                Media removed by the edit is bound to this identity for undo.
            undo_token: Identity of the TextArea edit batch being reversed. A
                detached payload returns only when this matches the identity
                recorded when that payload was detached, so placeholder-shaped
                text from any other source stays ordinary text.
        """
        # Both kinds ask the same question of the same edit, so the diff is
        # computed once here rather than per kind.
        equal_spans = (
            None
            if undo_previous_text is None
            else self._unchanged_spans(undo_previous_text, text)
        )
        restored_images = self._restored_placeholder_spans(
            equal_spans, text, IMAGE_PLACEHOLDER_PATTERN
        )
        restored_videos = self._restored_placeholder_spans(
            equal_spans, text, VIDEO_PLACEHOLDER_PATTERN
        )
        self._sync_kind_images(
            text,
            previous_text=previous_text,
            cursor_offset=cursor_offset,
            restored_spans=restored_images,
            edit_token=edit_token,
            undo_token=undo_token,
        )
        self._sync_kind_videos(
            text,
            previous_text=previous_text,
            cursor_offset=cursor_offset,
            restored_spans=restored_videos,
            edit_token=edit_token,
            undo_token=undo_token,
        )

    def remap_spans_to_text(self, text: str, *, previous_text: str) -> None:
        """Re-map tracked placeholder spans onto a transformed copy of the text.

        Submission rewrites the draft before the display placeholders are
        stripped from the model-facing text: whitespace is trimmed, collapsed
        pastes expand back to full content, dropped paths become placeholders,
        and a mode prefix may be prepended. Every one of those shifts character
        offsets, so a `placeholder_span` captured against the draft would be
        stale by the time `strip_media_placeholders` consumes it — silently
        stripping the wrong occurrence when a user-typed duplicate is present.

        Re-mapping each span through the same before/after diff keeps it pointing
        at its own display token in `text`. Spans that cannot be cleanly mapped
        become `None`, degrading to the token-count fallback rather than a wrong
        strip.

        Args:
            text: The transformed text the spans must line up with.
            previous_text: The draft text the current spans were captured
                against.
        """
        for item in (*self.images, *self.videos):
            if item.placeholder_span is None:
                continue
            item.placeholder_span = self._map_placeholder_span(
                item.placeholder_span, previous_text, text
            )

    def _sync_kind_images(
        self,
        text: str,
        *,
        previous_text: str | None = None,
        cursor_offset: int | None = None,
        restored_spans: dict[str, list[tuple[int, int]]],
        edit_token: object | None = None,
        undo_token: object | None = None,
    ) -> None:
        """Sync image list to surviving placeholders in text.

        Also rewrites the detached pool with images whose token left the text,
        and records any payload the pool caps evicted.

        Args:
            text: Current input text.
            previous_text: Previous input text, used to map existing spans.
            cursor_offset: Current cursor offset for duplicate disambiguation.
            restored_spans: Token occurrences introduced by an actual undo.
            edit_token: Identity of the edit that produced `text`.
            undo_token: Identity of the edit currently being undone.
        """
        matches = list(IMAGE_PLACEHOLDER_PATTERN.finditer(text))
        placeholders = {m.group(0) for m in matches}
        partition = self._partition_media(
            attached=self.images,
            detached=self._detached_images,
            detached_edits=self._detached_image_edits,
            placeholders=placeholders,
            pattern=IMAGE_PLACEHOLDER_PATTERN,
            restored_spans=restored_spans,
            edit_token=edit_token,
            undo_token=undo_token,
        )
        self.images = partition.attached
        self._detached_images = partition.detached
        self._record_evicted(partition.evicted, partition.evicted_edits)
        self._record_stranded(restored_spans, self.images, undo_token)
        self._update_placeholder_spans(
            self.images,
            matches,
            text,
            previous_text,
            cursor_offset,
            restored_spans,
        )
        retained = [*self.images, *self._detached_images]
        if retained:
            self.next_image_id = max(
                self.next_image_id,
                self._max_placeholder_id(
                    retained, IMAGE_PLACEHOLDER_PATTERN, len(retained)
                ),
            )

    def _sync_kind_videos(
        self,
        text: str,
        *,
        previous_text: str | None = None,
        cursor_offset: int | None = None,
        restored_spans: dict[str, list[tuple[int, int]]],
        edit_token: object | None = None,
        undo_token: object | None = None,
    ) -> None:
        """Sync video list to surviving placeholders in text.

        Also rewrites the detached pool with videos whose token left the text,
        and records any payload the pool caps evicted.

        Args:
            text: Current input text.
            previous_text: Previous input text, used to map existing spans.
            cursor_offset: Current cursor offset for duplicate disambiguation.
            restored_spans: Token occurrences introduced by an actual undo.
            edit_token: Identity of the edit that produced `text`.
            undo_token: Identity of the edit currently being undone.
        """
        matches = list(VIDEO_PLACEHOLDER_PATTERN.finditer(text))
        placeholders = {m.group(0) for m in matches}
        partition = self._partition_media(
            attached=self.videos,
            detached=self._detached_videos,
            detached_edits=self._detached_video_edits,
            placeholders=placeholders,
            pattern=VIDEO_PLACEHOLDER_PATTERN,
            restored_spans=restored_spans,
            edit_token=edit_token,
            undo_token=undo_token,
        )
        self.videos = partition.attached
        self._detached_videos = partition.detached
        self._record_evicted(partition.evicted, partition.evicted_edits)
        self._record_stranded(restored_spans, self.videos, undo_token)
        self._update_placeholder_spans(
            self.videos,
            matches,
            text,
            previous_text,
            cursor_offset,
            restored_spans,
        )
        retained = [*self.videos, *self._detached_videos]
        if retained:
            self.next_video_id = max(
                self.next_video_id,
                self._max_placeholder_id(
                    retained, VIDEO_PLACEHOLDER_PATTERN, len(retained)
                ),
            )

    def _partition_media(
        self,
        *,
        attached: list[_MediaItemT],
        detached: list[_MediaItemT],
        detached_edits: dict[str, object],
        placeholders: set[str],
        pattern: re.Pattern[str],
        restored_spans: dict[str, list[tuple[int, int]]],
        edit_token: object | None,
        undo_token: object | None,
    ) -> _MediaPartition[_MediaItemT]:
        """Split media into what the current text references and what it does not.

        Attached items survive while their bound token remains. Detached items
        can return only through occurrences identified as belonging to an actual
        undo, so same-looking text typed or pasted later stays ordinary text.

        Args:
            attached: Media currently bound to tokens in the draft.
            detached: Media retained for undo after its token was removed.
            detached_edits: Detaching edit identities keyed by token.
            placeholders: Placeholder tokens present in the current text.
            pattern: Placeholder regex with an `id` group, used for ordering.
            restored_spans: Token occurrences introduced by an actual undo.
            edit_token: Identity of the edit that produced the current text.
            undo_token: Identity of the edit currently being undone.

        Returns:
            The partition for the current text. Attached items are sorted by
            placeholder ID so a re-attached item returns to its original
            position instead of appending after higher-ID items. Evicted items
            are the detached payloads dropped to stay within the pool caps;
            their tokens can no longer be restored by undo.
        """
        attached_by_token: dict[str, _MediaItemT] = {}
        for item in attached:
            if item.placeholder in attached_by_token:
                # Reachable when a caller adds the same payload object twice:
                # `add_media` assigns `data.placeholder` in place, so both list
                # entries alias one object carrying the newer token. Distinct
                # payloads cannot collide, because `next_image_id` only ever
                # rises (the `max()` folds in `_sync_kind_images` /
                # `_sync_kind_videos`) and `restore()` releases the detached
                # pool alongside lowering it. A duplicate token is inherently
                # unresolvable — two payloads cannot share one occurrence — so
                # warn rather than drop a payload in silence.
                logger.warning(
                    "Duplicate placeholder %r among attached media; "
                    "keeping only the most recent payload",
                    item.placeholder,
                )
            attached_by_token[item.placeholder] = item
        detached_by_token = {
            item.placeholder: item
            for item in detached
            if item.placeholder not in attached_by_token
        }
        restored_tokens = {
            token
            for token in restored_spans
            if undo_token is not None and detached_edits.get(token) is undo_token
        }
        still_attached = [
            item
            for item in attached_by_token.values()
            if item.placeholder in placeholders
        ]
        still_attached.extend(
            item
            for item in detached_by_token.values()
            if item.placeholder in restored_tokens
        )
        still_attached.sort(key=lambda item: self._placeholder_sort_key(item, pattern))
        now_detached = [
            item
            for item in detached_by_token.values()
            if item.placeholder not in restored_tokens
        ]
        newly_detached = [
            item
            for item in attached_by_token.values()
            if item.placeholder not in placeholders
        ]
        now_detached.extend(newly_detached)

        next_edits = {
            item.placeholder: detached_edits[item.placeholder]
            for item in now_detached
            if item.placeholder in detached_edits
        }
        if edit_token is not None:
            next_edits.update({item.placeholder: edit_token for item in newly_detached})

        retained, evicted = self._trim_detached(now_detached)
        evicted_edits = {
            item.placeholder: next_edits[item.placeholder]
            for item in evicted
            if item.placeholder in next_edits
        }
        detached_edits.clear()
        detached_edits.update(
            {
                item.placeholder: next_edits[item.placeholder]
                for item in retained
                if item.placeholder in next_edits
            }
        )
        return _MediaPartition(
            attached=still_attached,
            detached=retained,
            evicted=evicted,
            evicted_edits=evicted_edits,
        )

    @staticmethod
    def _trim_detached(
        detached: list[_MediaItemT],
    ) -> tuple[list[_MediaItemT], list[_MediaItemT]]:
        """Drop oldest detached payloads to fit the pool caps, keeping the newest.

        Args:
            detached: Detached pool in oldest-first order.

        Returns:
            The `(retained, evicted)` split, both oldest-first. The newest entry
            is always retained so the most recent delete stays undoable even
            when its payload alone exceeds `MAX_DETACHED_MEDIA_BYTES` — in that
            case the pool deliberately does not fit the byte cap.
        """
        kept: list[_MediaItemT] = []
        budget = MAX_DETACHED_MEDIA_BYTES
        for index, item in enumerate(reversed(detached)):
            size = len(item.base64_data)
            # `break`, not `continue`: retention stops at the first item that
            # does not fit, so one oversized newer payload evicts every older
            # entry even where a smaller one would still have fit the budget.
            # The slice below depends on `kept` being a contiguous suffix, so
            # relaxing this into a best-fit `continue` would also need to change
            # how `evicted` is computed.
            if index and (len(kept) >= MAX_DETACHED_MEDIA or size > budget):
                break
            budget -= size
            kept.append(item)
        kept.reverse()
        return kept, detached[: len(detached) - len(kept)]

    @staticmethod
    def _unchanged_spans(previous_text: str, text: str) -> list[tuple[int, int]]:
        """Return the spans of `text` that an edit left untouched.

        The shared prefix and suffix are unchanged by construction, so only the
        differing middle needs a real diff. That keeps the cost proportional to
        the size of the edit rather than the size of the draft: a character-level
        `SequenceMatcher` over a whole draft is quadratic, which made a single
        ctrl+z on a large draft block the UI for seconds.

        Args:
            previous_text: Text immediately before the edit.
            text: Text produced by the edit.

        Returns:
            Ascending, non-overlapping spans of `text` carried over unchanged.
        """
        limit = min(len(previous_text), len(text))
        prefix = 0
        while prefix < limit and previous_text[prefix] == text[prefix]:
            prefix += 1
        suffix = 0
        while (
            suffix < limit - prefix
            and previous_text[len(previous_text) - 1 - suffix]
            == text[len(text) - 1 - suffix]
        ):
            suffix += 1

        spans: list[tuple[int, int]] = []
        if prefix:
            spans.append((0, prefix))
        old_middle = previous_text[prefix : len(previous_text) - suffix]
        new_middle = text[prefix : len(text) - suffix]
        if old_middle and new_middle:
            matcher = SequenceMatcher(a=old_middle, b=new_middle, autojunk=False)
            for tag, _old_start, _old_end, new_start, new_end in matcher.get_opcodes():
                if tag == "equal":
                    spans.append((prefix + new_start, prefix + new_end))
        if suffix:
            spans.append((len(text) - suffix, len(text)))
        return spans

    @staticmethod
    def _restored_placeholder_spans(
        equal_spans: list[tuple[int, int]] | None,
        text: str,
        pattern: re.Pattern[str],
    ) -> dict[str, list[tuple[int, int]]]:
        """Return placeholder occurrences inserted by an undo operation.

        A match lying wholly inside an unchanged span was already in the draft
        before the edit; anything else was introduced by it. Containment rather
        than overlap is required so a token merely abutting an unchanged region
        is not mistaken for a survivor.

        Args:
            equal_spans: Spans of `text` the edit left untouched, or `None` when
                the current edit was not an undo.
            text: Text produced by the undo.
            pattern: Placeholder regex for the media kind being examined.

        Returns:
            Restored spans grouped by placeholder token.
        """
        if equal_spans is None:
            return {}

        restored: dict[str, list[tuple[int, int]]] = {}
        for match in pattern.finditer(text):
            start, end = match.span()
            survived = any(
                equal_start <= start and end <= equal_end
                for equal_start, equal_end in equal_spans
            )
            if not survived:
                restored.setdefault(match.group(0), []).append(match.span())
        return restored

    @staticmethod
    def _placeholder_sort_key(
        item: ImageData | VideoData, pattern: re.Pattern[str]
    ) -> int:
        """Return the numeric placeholder ID used to order tracked media.

        This ordering decides the order of the content blocks handed to the
        model, so an unparsable placeholder is logged rather than quietly
        sorted: `media_utils` builds every payload with the un-numbered sentinel
        `[image]`/`[video]` and relies on `add_media` to replace it, so a missed
        `add_media` would otherwise mis-pair images with their references.

        Args:
            item: Media item whose placeholder should be parsed.
            pattern: Placeholder regex with an `id` group.

        Returns:
            The parsed ID, or `-1` when the placeholder is not a numbered token,
            keeping such items ahead of every real ID deterministically.
        """
        match = pattern.fullmatch(item.placeholder)
        if match is None:
            logger.warning(
                "Media item has non-numbered placeholder %r; content block order "
                "may not match placeholder order in the message",
                item.placeholder,
            )
            return -1
        return int(match.group("id"))

    def _update_placeholder_spans(
        self,
        items: list[ImageData] | list[VideoData],
        matches: list[re.Match[str]],
        text: str,
        previous_text: str | None,
        cursor_offset: int | None,
        restored_spans: dict[str, list[tuple[int, int]]],
    ) -> None:
        """Refresh tracked placeholder spans for surviving media items.

        Args:
            items: Surviving tracked media items.
            matches: Placeholder regex matches in the current text.
            text: Current input text.
            previous_text: Previous input text, used to map existing spans.
            cursor_offset: Current cursor offset for duplicate disambiguation.
            restored_spans: Token occurrences introduced by an actual undo.
        """
        spans_by_token: dict[str, list[tuple[int, int]]] = {}
        for match in matches:
            spans_by_token.setdefault(match.group(0), []).append(match.span())

        for item in items:
            if restored := restored_spans.get(item.placeholder):
                # Detached items retain the span of their originally bound token.
                # When undo restores duplicate look-alikes, that exact occurrence
                # must win over the first same-looking token in the text.
                spans = spans_by_token.get(item.placeholder, [])
                if item.placeholder_span not in spans:
                    item.placeholder_span = restored[0]
                continue
            spans = spans_by_token.get(item.placeholder, [])
            cursor_span = self._placeholder_span_after_cursor(spans, cursor_offset)
            mapped = self._map_placeholder_span(
                item.placeholder_span, previous_text, text
            )
            had_duplicate = (
                previous_text is not None and previous_text.count(item.placeholder) > 1
            )
            if had_duplicate and mapped is not None and mapped in spans:
                item.placeholder_span = mapped
            elif cursor_span is not None and (
                item.placeholder_span is None or mapped != item.placeholder_span
            ):
                item.placeholder_span = cursor_span
            elif len(spans) == 1:
                item.placeholder_span = spans[0]
            elif mapped is not None and mapped in spans:
                item.placeholder_span = mapped
            elif item.placeholder_span not in spans:
                item.placeholder_span = None

    @staticmethod
    def _placeholder_span_after_cursor(
        spans: list[tuple[int, int]], cursor_offset: int | None
    ) -> tuple[int, int] | None:
        """Return the first duplicate placeholder span at or after the cursor.

        Only meaningful when the token is duplicated (`len(spans) > 1`); returns
        `None` otherwise, or when no span starts at/after the cursor.

        Args:
            spans: Placeholder spans for one token in current text.
            cursor_offset: Current cursor offset, or `None` when unknown.
        """
        if cursor_offset is None or len(spans) <= 1:
            return None
        for span in spans:
            start, _end = span
            if start >= cursor_offset:
                return span
        return None

    @staticmethod
    def _map_placeholder_span(
        span: tuple[int, int] | None,
        previous_text: str | None,
        text: str,
    ) -> tuple[int, int] | None:
        """Map a placeholder span from the previous text into current text.

        Uses a `SequenceMatcher` diff so the span survives edits elsewhere in the
        text: it returns the shifted span only when its whole range falls inside
        an unchanged (`equal`) block, and `None` when the token's own characters
        were edited or the span cannot be located.

        Args:
            span: The placeholder span in `previous_text`, or `None`.
            previous_text: Text the span was captured against, or `None`.
            text: Text to map the span into.

        Returns:
            The mapped span when the same placeholder occurrence survives,
            otherwise `None`.
        """
        if span is None or previous_text is None:
            return None
        start, end = span
        if not (0 <= start < end <= len(previous_text)):
            return None

        matcher = SequenceMatcher(a=previous_text, b=text, autojunk=False)
        for tag, old_start, old_end, new_start, _new_end in matcher.get_opcodes():
            if tag == "equal" and old_start <= start and end <= old_end:
                offset = new_start - old_start
                return start + offset, end + offset
            if old_start <= start and end <= old_end:
                return None
            if old_start > end:
                break
        return None

    @staticmethod
    def _max_placeholder_id(
        items: list[ImageData] | list[VideoData],
        pattern: re.Pattern[str],
        fallback_count: int,
    ) -> int:
        """Compute next ID from the highest surviving placeholder.

        Args:
            items: Surviving media items.
            pattern: Placeholder regex with an `id` group.
            fallback_count: Fallback when no IDs can be parsed.

        Returns:
            Next ID value (max_id + 1).
        """
        max_id = 0
        for item in items:
            match = pattern.fullmatch(item.placeholder)
            if match is not None:
                max_id = max(max_id, int(match.group("id")))
        return max_id + 1 if max_id else fallback_count + 1


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
                console.print(
                    f"[yellow]Warning: File not found: "
                    f"{escape_markup(raw_path)}[/yellow]"
                )
        except (OSError, RuntimeError) as e:
            console.print(
                f"[yellow]Warning: Invalid path "
                f"{escape_markup(raw_path)}: "
                f"{escape_markup(str(e))}[/yellow]"
            )

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
        resolved = _resolve_existing_pasted_path(path)
        if resolved is None:
            return []
        paths.append(resolved)

    return paths


def parse_pasted_path_payload(
    text: str, *, allow_leading_path: bool = False
) -> ParsedPastedPathPayload | None:
    """Parse dropped-path payload variants through one entrypoint.

    Parsing order is:
    1. strict multi-path payload parsing (`parse_pasted_file_paths`)
    2. single-path normalization/parsing (`parse_single_pasted_file_path`)
    3. optional leading-path extraction (`extract_leading_pasted_file_path`)

    Args:
        text: Input payload to parse.
        allow_leading_path: Whether to parse a leading path token followed by
            trailing prompt text.

    Returns:
        Parsed payload details, otherwise `None`.
    """
    paths = parse_pasted_file_paths(text)
    if paths:
        return ParsedPastedPathPayload(paths=paths)

    single_path = parse_single_pasted_file_path(text)
    if single_path is not None:
        return ParsedPastedPathPayload(paths=[single_path])

    if not allow_leading_path:
        return None

    leading = extract_leading_pasted_file_path(text)
    if leading is None:
        return None

    path, token_end = leading
    return ParsedPastedPathPayload(paths=[path], token_end=token_end)


def looks_like_dropped_payload(text: str) -> bool:
    """Return whether a payload has the shape a terminal uses for a file drop.

    Terminals deliver a dragged file as an absolute path (POSIX or Windows
    drive/UNC), a `~/` path, or a `file://` URL, so requiring that shape keeps a
    payload that *begins* with a hand-typed relative path (`assets/logo.png`)
    from being mistaken for a drop. Without this guard `parse_pasted_path_payload`
    would resolve such a token against the working directory, and a caller that
    rejects drops would swallow ordinary text. The check is leading-token only:
    once the first token passes, later relative tokens still resolve against the
    working directory.

    Pure string inspection, so callers can apply it on the event loop to decide
    whether the filesystem-touching parse is worth a thread hop at all.

    Args:
        text: Raw pasted/dropped text payload.

    Returns:
        `True` when the payload begins like a dropped path.
    """
    value = text.strip().lstrip("<'\"")
    return bool(
        value.startswith(("/", "~/", "file://", "\\\\"))
        or _WINDOWS_DRIVE_PATH_PATTERN.match(value)
    )


def dropped_payload_paths(text: str) -> list[Path]:
    """Return resolved file paths from a payload that looks like a file drop.

    Applies `parse_pasted_path_payload` — the same parser the chat input's drop
    handling uses, which only resolves paths that exist on disk — behind a shape
    guard that requires the drop form terminals actually emit (see
    `looks_like_dropped_payload`). Text-only inputs use this to detect a
    dragged file so it can be rejected instead of inserted as a path.

    Leading-path-plus-trailing-text payloads (`<path> what is this?`) are
    deliberately out of scope: `allow_leading_path` stays off, matching the
    chat input's own drop-time calls, which handle that shape later at submit
    time instead.

    Args:
        text: Raw pasted/dropped text payload.

    Returns:
        Resolved file paths found in the payload, or an empty list.
    """
    if not looks_like_dropped_payload(text):
        return []
    parsed = parse_pasted_path_payload(text)
    if parsed is None:
        return []
    return list(parsed.paths)


def parse_single_pasted_file_path(text: str) -> Path | None:
    """Parse and resolve a single pasted path payload.

    Unlike `parse_pasted_file_paths`, this helper only accepts one path token
    and is intended for fallback handling when a paste event carries a
    single path representation.

    Args:
        text: Raw pasted text payload.

    Returns:
        Resolved path when payload is a single existing file, otherwise `None`.
    """
    candidate = normalize_pasted_path(text)
    if candidate is None:
        return None
    return _resolve_existing_pasted_path(candidate)


def extract_leading_pasted_file_path(text: str) -> tuple[Path, int] | None:
    """Extract and resolve a leading pasted path token from input text.

    This is used for submit-time recovery when a user message starts with a
    path token followed by additional prompt text.

    Args:
        text: Input text to inspect.

    Returns:
        Tuple of `(resolved_path, token_end_index)` or `None` when no valid
        leading file path token exists.
    """
    if not text:
        return None

    start = len(text) - len(text.lstrip())
    payload = text[start:]
    token_end = _leading_token_end(payload)
    if token_end is None:
        return None

    token_text = payload[:token_end]
    path = parse_single_pasted_file_path(token_text)
    if path is None:
        spaced = _extract_unquoted_leading_path_with_spaces(payload)
        if spaced is None:
            return None
        spaced_path, spaced_end = spaced
        return spaced_path, start + spaced_end

    return path, start + token_end


def normalize_pasted_path(text: str) -> Path | None:
    """Normalize pasted text that may represent a single filesystem path.

    Supports:

    - quoted and shell-escaped single paths
    - `file://` URLs
    - Windows drive-letter and UNC paths

    Args:
        text: Raw pasted text payload.

    Returns:
        Parsed `Path` if payload is a single path token, otherwise `None`.
    """
    payload = text.strip()
    if not payload:
        return None

    unquoted = (
        payload.removeprefix('"').removesuffix('"')
        if payload.startswith('"') and payload.endswith('"')
        else payload
    )
    unquoted = (
        unquoted.removeprefix("'").removesuffix("'")
        if unquoted.startswith("'") and unquoted.endswith("'")
        else unquoted
    )

    if unquoted.startswith("file://"):
        return _token_to_path(unquoted)

    windows_path = _normalize_windows_pasted_path(unquoted)
    if windows_path is not None:
        return windows_path

    posix_path = _normalize_posix_pasted_path(unquoted)
    if posix_path is not None:
        return posix_path

    parts = _split_paste_line(payload)
    if len(parts) != 1:
        return None
    token = parts[0]
    path = _token_to_path(token)
    if path is None:
        return None
    windows_token_path = _normalize_windows_pasted_path(str(path))
    if windows_token_path is not None:
        return windows_token_path
    return path


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
        try:
            parsed = urlparse(value)
        except ValueError as e:
            # Malformed authority (e.g. `file://[::1/x.png`) raises rather than
            # returning a partial result; treat it as ordinary text.
            logger.debug("file:// URL parsing failed for %r: %s", value, e)
            return None
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


def _leading_token_end(text: str) -> int | None:
    """Return the end index of the first shell-like token.

    Args:
        text: Input text beginning with a token.

    Returns:
        End index (exclusive), or `None` when token parsing fails.
    """
    if not text:
        return None

    if text[0] in {'"', "'"}:
        quote = text[0]
        escaped = False
        for index in range(1, len(text)):
            char = text[index]
            if char == "\\" and not escaped:
                escaped = True
                continue
            if char == quote and not escaped:
                return index + 1
            escaped = False
        return None

    escaped = False
    for index, char in enumerate(text):
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char.isspace() and not escaped:
            return index
        escaped = False
    return len(text)


def _extract_unquoted_leading_path_with_spaces(text: str) -> tuple[Path, int] | None:
    """Extract a leading unquoted path that may contain spaces.

    This fallback is intentionally POSIX-oriented (`/` and `~/`) because the
    slash-command conflict it addresses is specific to inputs that begin with
    `/`.

    Args:
        text: Input text beginning with a potential path.

    Returns:
        Tuple of `(resolved_path, token_end_index)` or `None` when no matching
        leading path prefix resolves to an existing file.
    """
    if not text or ("\n" in text or "\r" in text):
        return None
    if not text.startswith(("/", "~/")):
        return None
    if " " not in text and "\u00a0" not in text and "\u202f" not in text:
        return None

    boundaries = [index for index, char in enumerate(text) if char.isspace()]
    boundaries.append(len(text))
    for end in reversed(boundaries):
        candidate = text[:end].rstrip()
        if not candidate:
            continue
        path = parse_single_pasted_file_path(candidate)
        if path is not None:
            return path, len(candidate)
    return None


def _normalize_windows_pasted_path(text: str) -> Path | None:
    """Return a `Path` for unquoted Windows drive/UNC path inputs.

    Args:
        text: Potential Windows path input.

    Returns:
        Parsed `Path` when `text` is Windows drive-letter or UNC style,
        otherwise `None`.
    """
    if _WINDOWS_DRIVE_PATH_PATTERN.match(text) or text.startswith("\\\\"):
        return Path(text)
    return None


def _normalize_posix_pasted_path(text: str) -> Path | None:
    """Return a `Path` for likely POSIX absolute/home path payloads.

    Some terminals paste dropped absolute paths with spaces as raw text without
    quoting/escaping. In that case shell tokenization splits on spaces even
    though the full payload is intended to be a single path.

    Args:
        text: Potential POSIX path input.

    Returns:
        Parsed `Path` when `text` looks like a raw POSIX absolute/home path,
        otherwise `None`.
    """
    if "\n" in text or "\r" in text:
        return None
    if text.startswith("~/"):
        return Path(text)
    if text.startswith("/") and "/" in text[1:]:
        return Path(text)
    return None


def _safe_exists(path: Path) -> bool:
    """Return whether `path` exists, treating OS rejections as non-existent.

    Filesystem probes (`exists`/`is_file`/`is_dir`) issue an `os.stat` that can
    raise `OSError` for inputs the OS refuses outright — notably `ENAMETOOLONG`
    when a path component exceeds the filesystem limit. Whether `pathlib`
    swallows such an error is version-dependent (Python <=3.13 ignores only a
    small set of errnos and lets `ENAMETOOLONG` propagate; 3.14 routes these
    through `os.path.*`, which swallows more), so we guard unconditionally for
    uniform behavior. Callers here only care whether the path is usable, so a
    failed probe is equivalent to "not there".

    `ValueError` needs no guard here: every supported interpreter already
    absorbs an embedded NUL inside these three probes (3.11-3.13 catch it in
    `pathlib`, 3.14 delegates to `os.path.*`, which catches it). Only
    `resolve()` propagates it — see `_resolve_existing_pasted_path`.

    Args:
        path: Path candidate to probe.

    Returns:
        `True` if the path exists, `False` if it does not or cannot be probed.
    """
    try:
        return path.exists()
    except OSError as e:
        logger.debug("exists() check failed for %r: %s", path, e)
        return False


def _safe_is_file(path: Path) -> bool:
    """Return whether `path` is an existing file, ignoring stat failures.

    See `_safe_exists` for why probes are guarded.

    Args:
        path: Path candidate to probe.

    Returns:
        `True` if the path is a regular file, `False` otherwise or on failure.
    """
    try:
        return path.is_file()
    except OSError as e:
        logger.debug("is_file() check failed for %r: %s", path, e)
        return False


def _safe_is_dir(path: Path) -> bool:
    """Return whether `path` is an existing directory, ignoring stat failures.

    See `_safe_exists` for why probes are guarded.

    Args:
        path: Path candidate to probe.

    Returns:
        `True` if the path is a directory, `False` otherwise or on failure.
    """
    try:
        return path.is_dir()
    except OSError as e:
        logger.debug("is_dir() check failed for %r: %s", path, e)
        return False


def _resolve_existing_pasted_path(path: Path) -> Path | None:
    """Resolve a pasted path candidate to an existing file.

    Performs an exact resolution first, then a Unicode-space-tolerant lookup.

    Args:
        path: Parsed path candidate.

    Returns:
        Resolved existing file path, otherwise `None`.
    """
    try:
        resolved = path.expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as e:
        # ValueError covers an embedded NUL, which `resolve()` rejects outright.
        logger.debug("Path resolution failed for %r: %s", path, e)
        return None
    if _safe_is_file(resolved):
        return resolved

    fuzzy = _resolve_with_unicode_space_variants(path)
    if fuzzy is None:
        return None
    try:
        resolved_fuzzy = fuzzy.resolve()
    except (OSError, RuntimeError, ValueError) as e:
        logger.debug("Unicode-space resolution failed for %r: %s", fuzzy, e)
        return None
    if _safe_is_file(resolved_fuzzy):
        return resolved_fuzzy
    return None


def _normalize_unicode_spaces(text: str) -> str:
    """Normalize Unicode lookalike spaces to ASCII spaces.

    Args:
        text: Text to normalize.

    Returns:
        Normalized text with Unicode-space variants converted to ASCII spaces.
    """
    return text.translate(_UNICODE_SPACE_EQUIVALENTS)


def _resolve_with_unicode_space_variants(path: Path) -> Path | None:
    """Resolve path by matching filename segments with Unicode space variants.

    Args:
        path: Path candidate that may differ from disk by space code points.

    Returns:
        Matching filesystem path, or `None` when no variant match exists.
    """
    expanded = path.expanduser()
    if expanded.is_absolute():
        current = Path(expanded.anchor)
        parts = expanded.parts[1:]
    else:
        current = Path.cwd()
        parts = expanded.parts

    for index, part in enumerate(parts):
        candidate = current / part
        if _safe_exists(candidate):
            current = candidate
            continue

        if not _safe_is_dir(current):
            return None
        if " " not in part and "\u00a0" not in part and "\u202f" not in part:
            return None

        normalized_part = _normalize_unicode_spaces(part)
        try:
            matches = [
                entry
                for entry in current.iterdir()
                if _normalize_unicode_spaces(entry.name) == normalized_part
            ]
        except OSError as e:
            logger.debug("Failed listing %s for Unicode-space lookup: %s", current, e)
            return None

        if not matches:
            return None

        is_last = index == len(parts) - 1
        if is_last:
            file_matches = [entry for entry in matches if _safe_is_file(entry)]
            if file_matches:
                matches = file_matches
        else:
            dir_matches = [entry for entry in matches if _safe_is_dir(entry)]
            if dir_matches:
                matches = dir_matches

        matches.sort(key=lambda entry: entry.name)
        current = matches[0]

    return current
