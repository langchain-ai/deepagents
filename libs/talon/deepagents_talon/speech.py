"""Optional inbound voice transcription for Talon channels."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from deepagents_talon.interfaces import ChannelMessage

if TYPE_CHECKING:
    from deepagents_talon.config import TalonConfig

logger = logging.getLogger(__name__)


class VoiceTranscriber(Protocol):
    """Turn channel voice payloads into text."""

    async def transcribe(self, message: ChannelMessage) -> str | None:
        """Return transcribed text for a voice message.

        Args:
            message: Inbound channel message.

        Returns:
            Transcribed text, or `None` when the message cannot be transcribed.
        """


@dataclass(frozen=True, slots=True)
class OpenAIVoiceTranscriber:
    """Voice transcriber backed by the optional OpenAI SDK.

    Args:
        model: Audio transcription model identifier configured by the operator.
    """

    model: str

    async def transcribe(self, message: ChannelMessage) -> str | None:
        """Transcribe the local audio path in message metadata.

        Args:
            message: Inbound channel message with `voice_path` or `media_path`.

        Returns:
            Transcribed text, or `None` when the SDK or media file is unavailable.
        """
        path = _voice_path(message)
        if path is None:
            return None

        try:
            module = importlib.import_module("openai")
        except ImportError:
            logger.warning("Voice transcription requested, but the OpenAI SDK is not installed")
            return None

        if not path.is_file():
            logger.warning("Voice transcription skipped because media file is missing: %s", path)
            return None

        client = module.AsyncOpenAI()
        with path.open("rb") as audio:
            transcript = await client.audio.transcriptions.create(model=self.model, file=audio)
        text = getattr(transcript, "text", None)
        return text if isinstance(text, str) and text else None


def build_voice_transcriber(config: TalonConfig) -> VoiceTranscriber | None:
    """Build the configured voice transcriber, if enabled.

    Args:
        config: Talon runtime configuration.

    Returns:
        A transcriber when voice transcription is enabled and configured,
        otherwise `None`.
    """
    enabled = config.env.get("DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_ENABLED", "").lower()
    if enabled not in {"1", "true", "yes"}:
        return None
    model = config.env.get("DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_MODEL")
    if not model:
        logger.warning("Voice transcription enabled without a configured model")
        return None
    return OpenAIVoiceTranscriber(model=model)


async def transcribe_voice_message(
    transcriber: VoiceTranscriber | None,
    message: ChannelMessage,
) -> ChannelMessage:
    """Return a message with voice text appended when transcription succeeds.

    Args:
        transcriber: Optional voice transcriber.
        message: Inbound channel message.

    Returns:
        Original or transcribed channel message.
    """
    if transcriber is None or not _is_voice_message(message):
        return message

    try:
        text = await transcriber.transcribe(message)
    except Exception:
        logger.exception("Voice transcription failed")
        return message

    if not text:
        return message
    content = text if not message.text.strip() else f"{message.text}\n\n{text}"
    return ChannelMessage(
        conversation_id=message.conversation_id,
        text=content,
        sender_id=message.sender_id,
        message_id=message.message_id,
        metadata={**message.metadata, "voice_transcribed": True},
    )


def _is_voice_message(message: ChannelMessage) -> bool:
    return message.metadata.get("media_type") == "voice" or "voice_path" in message.metadata


def _voice_path(message: ChannelMessage) -> Path | None:
    value = message.metadata.get("voice_path") or message.metadata.get("media_path")
    if isinstance(value, str) and value:
        return Path(value).expanduser()
    if isinstance(value, Path):
        return value.expanduser()
    return None
