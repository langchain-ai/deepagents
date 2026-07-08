"""Tests for the multimodal capability pre-check and error translator."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code.model_capabilities import (
    UserFacingModelError,
    check_multimodal_compatibility,
    is_multimodal_bad_request,
    message_has_image_block,
    model_supports_vision,
    translate_multimodal_error,
)


def _model(name: str, *, image_inputs: bool | None = None) -> SimpleNamespace:
    """Build a fake chat model with an optional `profile.image_inputs` flag."""
    profile: dict[str, Any] | None = (
        None if image_inputs is None else {"image_inputs": image_inputs}
    )
    return SimpleNamespace(model_name=name, profile=profile)


def _image_message() -> HumanMessage:
    return HumanMessage(
        content=[
            {"type": "text", "text": "what do you see?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAAA"},
            },
        ]
    )


def _text_message() -> HumanMessage:
    return HumanMessage(content="plain text only")


class TestModelSupportsVision:
    def test_missing_profile_defaults_to_true(self) -> None:
        assert model_supports_vision(_model("unknown")) is True

    def test_explicit_true(self) -> None:
        assert model_supports_vision(_model("gpt-5", image_inputs=True)) is True

    def test_explicit_false(self) -> None:
        assert (
            model_supports_vision(_model("zai-org/GLM-5.2", image_inputs=False))
            is False
        )


class TestMessageHasImageBlock:
    def test_string_content_has_no_image(self) -> None:
        assert message_has_image_block(_text_message()) is False

    def test_image_url_block(self) -> None:
        assert message_has_image_block(_image_message()) is True

    def test_image_block(self) -> None:
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "hi"},
                {"type": "image", "source_type": "base64", "data": "AAAA"},
            ]
        )
        assert message_has_image_block(msg) is True


class TestCheckMultimodalCompatibility:
    def test_vision_model_with_image_ok(self) -> None:
        check_multimodal_compatibility(
            _model("gpt-5", image_inputs=True), [_image_message()]
        )

    def test_non_vision_model_without_image_ok(self) -> None:
        check_multimodal_compatibility(
            _model("zai-org/GLM-5.2", image_inputs=False), [_text_message()]
        )

    def test_non_vision_model_with_image_raises(self) -> None:
        model = _model("zai-org/GLM-5.2", image_inputs=False)
        with pytest.raises(UserFacingModelError) as excinfo:
            check_multimodal_compatibility(model, [_image_message()])
        message = str(excinfo.value)
        assert "zai-org/GLM-5.2" in message
        assert "vision-capable" in message

    def test_unknown_profile_with_image_ok(self) -> None:
        check_multimodal_compatibility(_model("unknown"), [_image_message()])

    def test_scans_all_messages(self) -> None:
        model = _model("baseten:foo", image_inputs=False)
        messages = [AIMessage(content="prior"), _text_message(), _image_message()]
        with pytest.raises(UserFacingModelError):
            check_multimodal_compatibility(model, messages)


class _FakeBadRequestError(Exception):
    """Duck-typed stand-in for provider BadRequestError classes."""


_FakeBadRequestError.__name__ = "BadRequestError"


class _UnrelatedProviderError(Exception):
    """Duck-typed stand-in for an unrelated BadRequestError message."""


_UnrelatedProviderError.__name__ = "BadRequestError"


class TestIsMultimodalBadRequest:
    def test_baseten_phrasing(self) -> None:
        exc = _FakeBadRequestError(
            "400 Bad Request: This model does not support multimodal "
            "(image/video/audio) inputs."
        )
        assert is_multimodal_bad_request(exc) is True

    def test_openai_phrasing(self) -> None:
        exc = _FakeBadRequestError("model does not support images in this deployment")
        assert is_multimodal_bad_request(exc) is True

    def test_unrelated_bad_request_is_not_multimodal(self) -> None:
        exc = _UnrelatedProviderError("Invalid API key")
        assert is_multimodal_bad_request(exc) is False

    def test_non_bad_request_is_not_multimodal(self) -> None:
        exc = RuntimeError("does not support multimodal inputs")
        assert is_multimodal_bad_request(exc) is False


class TestTranslateMultimodalError:
    def test_returns_none_for_unrelated_error(self) -> None:
        translated = translate_multimodal_error(RuntimeError("boom"), _model("x"))
        assert translated is None

    def test_translates_provider_error(self) -> None:
        exc = _FakeBadRequestError(
            "This model does not support multimodal (image/video/audio) inputs."
        )
        translated = translate_multimodal_error(
            exc, _model("baseten:zai-org/GLM-5.2", image_inputs=False)
        )
        assert isinstance(translated, UserFacingModelError)
        assert "baseten:zai-org/GLM-5.2" in str(translated)
        assert "vision-capable" in str(translated)
