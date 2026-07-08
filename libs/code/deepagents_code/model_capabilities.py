"""Model capability checks for multimodal (image/video/audio) inputs.

Provides a pre-call scan that raises a user-facing error when the caller
tries to send image blocks to a model whose profile advertises no vision
support, plus a defense-in-depth translator that converts provider
`BadRequestError` traces mentioning multimodal input into the same
user-facing error type. The TUI catches `UserFacingModelError` at the
turn boundary and renders its message instead of a raw traceback.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class UserFacingModelError(RuntimeError):
    """Model call failed for a reason the user can act on."""


_MULTIMODAL_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"does not support multimodal", re.IGNORECASE),
    re.compile(r"multimodal \(image", re.IGNORECASE),
    re.compile(r"does not support image", re.IGNORECASE),
    re.compile(r"image[_\s-]*inputs?[^a-z]*not[^a-z]*supported", re.IGNORECASE),
    re.compile(r"vision[^a-z]*not[^a-z]*supported", re.IGNORECASE),
    re.compile(r"model does not support (?:vision|images?)", re.IGNORECASE),
)
"""Provider-error phrases that indicate the failure is a vision-support gap."""


def _model_identifier(model: object) -> str:
    """Return a best-effort human-readable name for a chat model.

    Returns:
        First non-empty string attribute among `model_name`, `model`, and
        `model_id`, falling back to the class name.
    """
    for attr in ("model_name", "model", "model_id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


def model_supports_vision(model: object) -> bool:
    """Return whether the model's profile advertises image input support.

    Reads the `image_inputs` flag from the model's `profile` dict (the same
    source `deepagents_code.config.create_model` consults for
    `unsupported_modalities`). Absent or unknown profiles default to `True`
    so unknown providers are not silently downgraded — the defense-in-depth
    error translator still catches provider rejections.
    """
    profile = getattr(model, "profile", None)
    if isinstance(profile, dict):
        image_inputs = profile.get("image_inputs")
        if image_inputs is False:
            return False
    return True


def _iter_content_blocks(content: object) -> Iterable[dict[str, Any]]:
    """Yield content blocks from a message's `.content` when it is a list."""
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict):
            yield block


def message_has_image_block(message: object) -> bool:
    """Return whether a message carries at least one image content block."""
    content = getattr(message, "content", None)
    for block in _iter_content_blocks(content):
        block_type = block.get("type")
        if block_type in {"image_url", "image"}:
            return True
    return False


def messages_have_image_blocks(messages: Sequence[object]) -> bool:
    """Return whether any message in the list carries an image content block."""
    return any(message_has_image_block(m) for m in messages)


def _vision_error_message(model: object) -> str:
    """Return the actionable "switch model" message shown to the user.

    Returns:
        A single-sentence message naming the offending model and listing a
        few vision-capable alternatives.
    """
    name = _model_identifier(model)
    return (
        f"Selected model {name} does not support images. Switch to a "
        "vision-capable model (e.g. gpt-5, claude-sonnet-4, gemini-2.5-pro) "
        "or describe the screenshot in text."
    )


def check_multimodal_compatibility(
    model: object, messages: Sequence[BaseMessage]
) -> None:
    """Raise if `messages` carry images the model rejects.

    Called before dispatch so the user sees a friendly message instead of a
    provider `BadRequestError` traceback.

    Raises:
        UserFacingModelError: When the model profile advertises no image
            support and at least one message carries an image content block.
    """
    if model_supports_vision(model):
        return
    if not messages_have_image_blocks(messages):
        return
    raise UserFacingModelError(_vision_error_message(model))


def is_multimodal_bad_request(exc: BaseException) -> bool:
    """Return whether `exc` looks like a provider "no multimodal support" error.

    Duck-typed on the exception class name so we don't hard-import provider
    SDKs (openai / anthropic / etc.). We look for `BadRequestError` in the
    class hierarchy and match the message against known phrases.
    """
    class_names = {cls.__name__ for cls in type(exc).__mro__}
    if "BadRequestError" not in class_names:
        return False
    message = str(exc)
    return any(pattern.search(message) for pattern in _MULTIMODAL_ERROR_PATTERNS)


def translate_multimodal_error(
    exc: BaseException, model: object
) -> UserFacingModelError | None:
    """Return a `UserFacingModelError` when `exc` is a multimodal rejection.

    Returns `None` when the exception is unrelated so callers can re-raise.
    """
    if not is_multimodal_bad_request(exc):
        return None
    logger.debug(
        "Translating provider multimodal rejection for %s: %s",
        _model_identifier(model),
        exc,
    )
    return UserFacingModelError(_vision_error_message(model))


__all__ = [
    "UserFacingModelError",
    "check_multimodal_compatibility",
    "is_multimodal_bad_request",
    "message_has_image_block",
    "messages_have_image_blocks",
    "model_supports_vision",
    "translate_multimodal_error",
]
