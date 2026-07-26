"""Unit tests for Telegram and WhatsApp send_message on empty text.

These tests verify that ``send_message("")`` does not raise
``UnboundLocalError`` when ``chunk_text`` returns an empty list.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from deepagents_talon.channels.telegram import TelegramChannel, TelegramChannelConfig
from deepagents_talon.channels.whatsapp import WhatsAppChannel, WhatsAppChannelConfig
from deepagents_talon.interfaces import SendResult

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


def _make_telegram_channel() -> TelegramChannel:
    """Create a TelegramChannel with a mocked transport."""
    config = TelegramChannelConfig(
        bot_token="fake-token",  # noqa: S106  # Test-only dummy value
        session_dir=MagicMock(),
    )
    transport = MagicMock()
    transport.call = AsyncMock(return_value={"result": {"message_id": 1}})
    return TelegramChannel(config, transport=transport)


class TestTelegramSendMessageEmpty:
    """Telegram send_message must not crash on empty text."""

    @pytest.mark.asyncio
    async def test_empty_text_does_not_raise(self) -> None:
        """send_message('') must not raise UnboundLocalError."""
        channel = _make_telegram_channel()
        result = await channel.send_message("chat-1", "")
        assert isinstance(result, SendResult)
        assert result.success is True


# ---------------------------------------------------------------------------
# WhatsApp
# ---------------------------------------------------------------------------


def _make_whatsapp_channel() -> WhatsAppChannel:
    """Create a WhatsAppChannel with a mocked transport."""
    config = WhatsAppChannelConfig(
        session_dir=MagicMock(),
    )
    transport = MagicMock()
    transport.post = AsyncMock(return_value={"success": True, "message_id": "m1"})
    return WhatsAppChannel(config, transport=transport)


class TestWhatsAppSendMessageEmpty:
    """WhatsApp send_message must not crash on empty text."""

    @pytest.mark.asyncio
    async def test_empty_text_does_not_raise(self) -> None:
        """send_message('') must not raise UnboundLocalError."""
        channel = _make_whatsapp_channel()
        result = await channel.send_message("chat-1", "")
        assert isinstance(result, SendResult)
        assert result.success is True
