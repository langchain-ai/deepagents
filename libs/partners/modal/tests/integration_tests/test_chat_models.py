"""Test `ChatModal` chat model."""

from __future__ import annotations

import os

import pytest
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_modal.chat_models import ChatModal

MODEL_NAME = os.environ.get("MODAL_TEST_MODEL", "")

pytestmark = pytest.mark.skipif(
    not os.environ.get("MODAL_PROXY_TOKEN") or not MODEL_NAME,
    reason="Set MODAL_PROXY_TOKEN and MODAL_TEST_MODEL to run Modal integration tests.",
)


class TestChatModal(ChatModelIntegrationTests):
    """Test `ChatModal` chat model."""

    @property
    def chat_model_class(self) -> type[ChatModal]:
        """Return class of chat model being tested."""
        return ChatModal

    @property
    def chat_model_params(self) -> dict:
        """Parameters to create chat model instance for testing."""
        return {
            "model": MODEL_NAME,
            "temperature": 0,
        }
