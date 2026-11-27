"""Downscale images returned by `read_file` before they enter the thread."""

from __future__ import annotations

import base64
import binascii
import io
import logging
import warnings
from typing import Final

logger = logging.getLogger(__name__)

MAX_IMAGE_SIDE: Final = 2000
"""Longest side, in pixels, of an image `read_file` returns.

Anthropic applies a stricter per-image dimension limit to every image once a
request carries more than 20 of them, and the thread resends every earlier
image on each turn. An image over the limit then fails every later request in
the thread. Shrinking at read time keeps the thread append-only: editing an
earlier image on a later request would instead invalidate Claude's replayed
thinking blocks.
"""

_RESIZABLE_FORMATS: Final = frozenset({"PNG", "JPEG", "WEBP"})
"""Formats re-encoded in place. GIF is skipped, and so is animated WebP, so animation isn't flattened."""


def downscale_image(data: str) -> str:
    """Return base64 image `data` with its longest side at most `MAX_IMAGE_SIDE`.

    Smaller images, animated images, other formats, and anything Pillow
    can't decode come back unchanged. Without Pillow installed this is a no-op.

    Args:
        data: Base64-encoded image bytes.

    Returns:
        Base64-encoded image bytes in the original format.
    """
    try:
        from PIL import Image  # noqa: PLC0415 - Pillow is optional
    except ImportError:
        return data
    try:
        raw = base64.b64decode(data, validate=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                animated = getattr(image, "is_animated", False)
                if animated or image.format not in _RESIZABLE_FORMATS or max(image.size) <= MAX_IMAGE_SIDE:
                    return data
                image_format = image.format
                image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                image.save(output, format=image_format)
    except (binascii.Error, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        logger.debug("Image could not be downscaled; returning it unchanged", exc_info=True)
        return data
    return base64.b64encode(output.getvalue()).decode("ascii")
