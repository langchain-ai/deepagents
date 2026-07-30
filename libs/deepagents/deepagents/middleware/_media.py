"""Shared helpers for detecting and decoding inline media content blocks.

Used by:

- `SummarizationMiddleware` — offloading inline media out of the conversation
    history before it is rendered and summarized.
- `_overflow_clip` — archiving inline media out of the trailing ToolMessage
    batch on the context-overflow fallback path.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import urllib.parse
from typing import Any

logger = logging.getLogger(__name__)


def _is_data_url(url: str) -> bool:
    """Return whether `url` is an inline `data:` URL.

    Any `data:` URL is treated as inline media to offload, because the XML
    history renderer drops `data:` URL blocks entirely (only `http(s)`-style
    references survive). This covers both base64 (`data:<mime>;base64,<payload>`)
    and percent-encoded / plaintext (`data:<mime>,<payload>`, e.g. an inline SVG)
    forms; whether the payload actually decodes is left to `_decode_data_url`.
    """
    return url.startswith("data:")


def _extract_data_url(block: Any) -> str | None:  # noqa: ANN401
    """Return the embedded `data:` URL for an inline-media content block.

    Detects the three inline-data content-block shapes that appear across
    LangChain messages:

    1. A standard content block with an explicit `base64` field.
    2. A `data:` URL on the `url` field.
    3. An OpenAI-style `image_url` block whose `url` is a `data:` URL.

    Both base64 (`;base64,`) and percent-encoded / plaintext `data:` URLs are
    detected -- e.g. an inline SVG (`data:image/svg+xml,<svg .../>`) -- because
    the XML history renderer drops *any* inline `data:` URL, so all of them must
    be offloaded to a referenceable path rather than left inline.

    Shape 3 is defensive: `content_blocks` normalizes most `image_url` blocks
    (a base64 `data:` URL becomes shape 1; an `https` URL becomes a plain `url`
    image block), so this branch rarely fires for normalized input; it is kept
    for raw, un-normalized blocks.

    This is pure detection and never raises: it reports *whether* a block
    carries inline data, leaving decoding (which can fail) to `_decode_data_url`.

    Args:
        block: A single content block (usually a dict).

    Returns:
        The block's `data:` URL, or `None` if the block carries no inline data.
    """
    if not isinstance(block, dict):
        return None

    # 1. Standard content block with an explicit base64 field.
    raw_b64 = block.get("base64")
    if raw_b64:
        mime = block.get("mime_type") or "application/octet-stream"
        return f"data:{mime};base64,{raw_b64}"

    # 2. Top-level data: URL.
    url = block.get("url", "")
    if isinstance(url, str) and _is_data_url(url):
        return url

    # 3. OpenAI-style image_url with a data: URL.
    image_url = block.get("image_url")
    if isinstance(image_url, dict):
        inner = image_url.get("url", "")
        if isinstance(inner, str) and _is_data_url(inner):
            return inner

    return None


def _decode_data_url(data_url: str) -> tuple[bytes, str, str] | None:
    """Decode a `data:` URL to raw bytes, a file extension, and a MIME type.

    Handles both encodings a `data:` URL can use: a `;base64,` payload is
    base64-decoded, while a plain `data:<mime>,<payload>` payload is treated as
    percent-encoded text (e.g. an inline SVG).

    Args:
        data_url: A `data:<mime>[;base64],<payload>` URL.

    Returns:
        A `(raw_bytes, extension, mime_type)` tuple, or `None` if decoding fails
            (including a malformed URL with no `,` payload separator). A failure
            is logged here and never swallowed silently: the summarization caller
            surfaces it as a failed-offload placeholder counted toward its
            aggregate warning, and the overflow-clip caller preserves the
            undecodable block verbatim in the result's manifest.
    """
    try:
        header, payload = data_url.split(",", 1)
        mime = header.split(":")[1].split(";")[0] if ":" in header else "application/octet-stream"
        ext = (mimetypes.guess_extension(mime) or ".bin").lstrip(".")
        is_base64 = "base64" in header.lower().split(";")
        raw = base64.b64decode(payload) if is_base64 else urllib.parse.unquote_to_bytes(payload)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to decode data: content block (%s): %s", type(e).__name__, e)
        return None
    else:
        return raw, ext, mime
