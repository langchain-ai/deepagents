"""Media helpers shared by channel adapters and the Talon host.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import base64
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Literal
from urllib.parse import urlparse

from deepagents_talon.interfaces import ChannelMedia

MAX_INBOUND_IMAGE_BYTES = 5 * 1024 * 1024
MAX_TEXT_DOCUMENT_BYTES = 100 * 1024

READABLE_DOCUMENT_EXTENSIONS = frozenset(
    {
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".xml",
        ".yaml",
        ".yml",
        ".log",
        ".py",
        ".js",
        ".ts",
        ".html",
        ".css",
    },
)
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm", ".3gp", ".m4v"})

ContentBlock = dict[str, object]
OutboundMediaType = Literal["image", "video", "document", "audio", "voice"]

_MARKDOWN_MEDIA_PATTERN = re.compile(r"!\[([^\]]*)]\(([^)\s]+)\)")
_FENCE_PATTERN = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_PATTERN = re.compile(r"`[^`\n]+`")
_FENCE_PLACEHOLDER = "\x00TALONFENCE"
_CODE_PLACEHOLDER = "\x00TALONCODE"


@dataclass(frozen=True, slots=True)
class MarkdownMediaRef:
    """Markdown media reference extracted from an agent response.

    Args:
        alt: Markdown alt text.
        path: Local path referenced by the response.
    """

    alt: str
    path: Path


def build_inbound_text(text: str, metadata: dict[str, object]) -> str:
    """Append bounded user-visible fallback text for inbound media.

    Args:
        text: Original channel message body.
        metadata: Channel metadata containing media paths and media type.

    Returns:
        Message text enriched with readable document content or media fallback text.
    """
    media_paths = _media_paths(metadata)
    if not media_paths:
        return text

    media_type = _normalized_media_type(metadata)
    parts = [text] if text.strip() else []

    if media_type == "document":
        parts.extend(_document_parts(media_paths))
    elif media_type == "image":
        parts.extend(_image_fallback_parts(media_paths))
    elif media_type == "video":
        parts.extend(_binary_media_parts(media_paths, media_type="video"))
    elif media_type in {"voice", "audio"}:
        return text
    else:
        parts.extend(_binary_media_parts(media_paths, media_type="media"))

    return "\n\n".join(part for part in parts if part.strip()) or text


def build_model_content(text: str, metadata: dict[str, object]) -> str | list[ContentBlock]:
    """Build provider-agnostic multimodal content for inbound photos.

    Args:
        text: Text passed to the model alongside any image blocks.
        metadata: Channel metadata containing media paths and MIME types.

    Returns:
        Plain text for non-photo messages, or LangChain-compatible content blocks
        containing a text block followed by image data URLs.
    """
    if _normalized_media_type(metadata) != "image":
        return text

    blocks: list[ContentBlock] = []
    mime_types = _media_mime_types(metadata)
    for index, path in enumerate(_media_paths(metadata)):
        mime = _image_mime(path, mime_types[index] if index < len(mime_types) else None)
        if mime is None:
            continue
        try:
            if path.stat().st_size > MAX_INBOUND_IMAGE_BYTES:
                continue
            data = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            continue
        blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})

    if not blocks:
        return text
    return [{"type": "text", "text": text.strip() or "(image)"}, *blocks]


def extract_markdown_media(text: str) -> tuple[str, list[MarkdownMediaRef]]:
    """Strip markdown media references from text and return local path refs.

    Args:
        text: Agent response text.

    Returns:
        Cleaned response text and extracted media references in source order.
    """
    if not text:
        return text, []

    masked, fences, codes = _mask_code(text)
    refs: list[MarkdownMediaRef] = []

    def record(match: re.Match[str]) -> str:
        refs.append(
            MarkdownMediaRef(
                alt=match.group(1),
                path=Path(match.group(2)),
            ),
        )
        return ""

    cleaned = _MARKDOWN_MEDIA_PATTERN.sub(record, masked)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return _restore_code(cleaned, fences, codes).strip(), refs


def outbound_channel_media(
    ref: MarkdownMediaRef,
    *,
    caption: str | None = None,
    root: Path | None = None,
) -> ChannelMedia:
    """Build an outbound channel media payload for a markdown media ref.

    Args:
        ref: Markdown media reference.
        caption: Optional caption for the media.
        root: Optional directory that must contain the referenced media. When
            supplied, Markdown references must be relative paths resolved inside
            this root.

    Returns:
        Channel media payload.

    Raises:
        ValueError: If the referenced path is unsafe or is not an image or video.
    """
    path = (
        resolve_bounded_media_path(ref.path, root, require_relative=True)
        if root is not None
        else ref.path.expanduser()
    )
    media_type = _outbound_media_type(path)
    if media_type is None:
        msg = f"unsupported outbound media file type: {path}"
        raise ValueError(msg)
    return ChannelMedia(path=path, media_type=media_type, caption=caption)


def resolve_bounded_media_path(
    path: Path,
    root: Path,
    *,
    require_relative: bool = False,
) -> Path:
    """Resolve a local media path and enforce containment under a trusted root.

    Args:
        path: Candidate media path.
        root: Directory that must contain the media after symlink resolution.
        require_relative: Whether to reject absolute candidate paths up front.

    Returns:
        Canonical media path under `root`.

    Raises:
        ValueError: If the path is unsafe, unavailable, or escapes `root`.
    """
    raw = str(path)
    parsed = urlparse(raw)
    windows_path = PureWindowsPath(raw)
    is_windows_drive_path = bool(windows_path.drive)
    if parsed.scheme and not is_windows_drive_path:
        msg = f"media path must be a local filesystem path: {path}"
        raise ValueError(msg)
    if raw.startswith("~") or (
        require_relative
        and (path.is_absolute() or windows_path.is_absolute() or is_windows_drive_path)
    ):
        msg = f"media path must be a local relative path under the outbound root: {path}"
        raise ValueError(msg)
    if ".." in path.parts or ".." in windows_path.parts:
        msg = f"media path must not contain parent-directory traversal: {path}"
        raise ValueError(msg)

    root_resolved = root.expanduser().resolve()
    candidate_input = root_resolved / path if not path.is_absolute() else path
    try:
        candidate = candidate_input.resolve(strict=True)
    except OSError as exc:
        msg = f"media file is unavailable: {path}"
        raise ValueError(msg) from exc
    if not candidate.is_relative_to(root_resolved):
        msg = f"media path escapes outbound root: {path}"
        raise ValueError(msg)
    if not candidate.is_file():
        msg = f"media path is not a regular file: {path}"
        raise ValueError(msg)
    return candidate


def _document_parts(paths: list[Path]) -> list[str]:
    parts: list[str] = []
    for path in paths:
        if path.suffix.lower() not in READABLE_DOCUMENT_EXTENSIONS:
            parts.append(f"_(Received unsupported document attachment: {path.name}.)_")
            continue
        try:
            if path.stat().st_size > MAX_TEXT_DOCUMENT_BYTES:
                parts.append(f"_(Document attachment is too large to read inline: {path.name}.)_")
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            parts.append(f"_(Document attachment was unavailable: {path.name}.)_")
            continue
        parts.append(f"[Content of {path.name}]:\n{content}")
    return parts


def _image_fallback_parts(paths: list[Path]) -> list[str]:
    parts: list[str] = []
    for path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            parts.append(f"_(Image attachment was unavailable: {path.name}.)_")
            continue
        if size > MAX_INBOUND_IMAGE_BYTES:
            parts.append(f"_(Image attachment is too large to inspect: {path.name}.)_")
    return parts


def _binary_media_parts(paths: list[Path], *, media_type: str) -> list[str]:
    return [f"_(Received {media_type} attachment: {path}.)_" for path in paths]


def _media_paths(metadata: dict[str, object]) -> list[Path]:
    values: list[object] = []
    raw_many = metadata.get("media_paths") or metadata.get("media_urls")
    if isinstance(raw_many, list):
        values.extend(raw_many)
    raw_one = metadata.get("media_path") or metadata.get("voice_path")
    if raw_one is not None:
        values.append(raw_one)

    paths: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        path = _path_from_value(value)
        if path is None or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def _path_from_value(value: object) -> Path | None:
    if isinstance(value, Path):
        return value.expanduser()
    if isinstance(value, str) and value:
        return Path(value).expanduser()
    return None


def _media_mime_types(metadata: dict[str, object]) -> list[str]:
    raw = metadata.get("media_mime_types") or metadata.get("mime_types")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str) and "/" in item]


def _normalized_media_type(metadata: dict[str, object]) -> str:
    raw = metadata.get("media_type") or metadata.get("message_type")
    value = raw.lower() if isinstance(raw, str) else ""
    if "image" in value or value in {"photo", "sticker"}:
        return "image"
    if "audio" in value or value in {"voice", "ptt"}:
        return "voice"
    if "video" in value:
        return "video"
    if value in {"document", "file"}:
        return "document"
    return value or "unknown"


def _image_mime(path: Path, reported: str | None) -> str | None:
    if reported is not None and reported.startswith("image/"):
        return reported
    guessed, _ = mimetypes.guess_type(path)
    if guessed is not None and guessed.startswith("image/"):
        return guessed
    return _sniff_image_mime(path)


def _sniff_image_mime(path: Path) -> str | None:
    try:
        header = path.read_bytes()[:12]
    except OSError:
        return None
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if header[:6] in {b"GIF87a", b"GIF89a"}:
        return "image/gif"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def _outbound_media_type(path: Path) -> OutboundMediaType | None:
    guessed, _ = mimetypes.guess_type(path)
    if guessed is not None:
        if guessed.startswith("image/"):
            return "image"
        if guessed.startswith("video/"):
            return "video"
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return "video"
    return None


def _mask_code(text: str) -> tuple[str, list[str], list[str]]:
    fences: list[str] = []
    codes: list[str] = []

    def save_fence(match: re.Match[str]) -> str:
        fences.append(match.group(0))
        return f"{_FENCE_PLACEHOLDER}{len(fences) - 1}\x00"

    def save_code(match: re.Match[str]) -> str:
        codes.append(match.group(0))
        return f"{_CODE_PLACEHOLDER}{len(codes) - 1}\x00"

    masked = _FENCE_PATTERN.sub(save_fence, text)
    return _INLINE_CODE_PATTERN.sub(save_code, masked), fences, codes


def _restore_code(text: str, fences: list[str], codes: list[str]) -> str:
    restored = text
    for index, fence in enumerate(fences):
        restored = restored.replace(f"{_FENCE_PLACEHOLDER}{index}\x00", fence)
    for index, code in enumerate(codes):
        restored = restored.replace(f"{_CODE_PLACEHOLDER}{index}\x00", code)
    return restored
