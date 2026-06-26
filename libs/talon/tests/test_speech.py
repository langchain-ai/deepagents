from pathlib import Path

from deepagents_talon.config import TalonConfig
from deepagents_talon.interfaces import ChannelMessage
from deepagents_talon.speech import (
    DEFAULT_LOCAL_VOICE_TRANSCRIPTION_MODEL,
    LocalParakeetVoiceTranscriber,
    OpenAIVoiceTranscriber,
    build_voice_transcriber,
    transcribe_voice_message,
)


def _config(env: dict[str, str], tmp_path: Path) -> TalonConfig:
    return TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test", **env}, base_home=tmp_path)


def test_build_voice_transcriber_uses_default_local_model(tmp_path: Path) -> None:
    transcriber = build_voice_transcriber(
        _config({"DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_ENABLED": "true"}, tmp_path)
    )

    assert isinstance(transcriber, LocalParakeetVoiceTranscriber)
    assert transcriber.model == DEFAULT_LOCAL_VOICE_TRANSCRIPTION_MODEL
    assert transcriber.device == "cpu"


def test_build_voice_transcriber_supports_legacy_speech_env(tmp_path: Path) -> None:
    transcriber = build_voice_transcriber(
        _config({"SPEECH_ENABLED": "true", "SPEECH_DEVICE": "cuda"}, tmp_path)
    )

    assert isinstance(transcriber, LocalParakeetVoiceTranscriber)
    assert transcriber.model == DEFAULT_LOCAL_VOICE_TRANSCRIPTION_MODEL
    assert transcriber.device == "cuda"


def test_build_voice_transcriber_uses_explicit_local_model(tmp_path: Path) -> None:
    transcriber = build_voice_transcriber(
        _config(
            {
                "DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_ENABLED": "true",
                "DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_MODEL": "nvidia/parakeet-tdt-0.6b-v3",
                "DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_DEVICE": "cuda",
            },
            tmp_path,
        )
    )

    assert isinstance(transcriber, LocalParakeetVoiceTranscriber)
    assert transcriber.model == "nvidia/parakeet-tdt-0.6b-v3"
    assert transcriber.device == "cuda"


def test_build_voice_transcriber_preserves_openai_model_override(tmp_path: Path) -> None:
    transcriber = build_voice_transcriber(
        _config(
            {
                "DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_ENABLED": "true",
                "DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_MODEL": "gpt-4o-transcribe",
            },
            tmp_path,
        )
    )

    assert isinstance(transcriber, OpenAIVoiceTranscriber)
    assert transcriber.model == "gpt-4o-transcribe"


async def test_transcribe_voice_message_transcribes_video() -> None:
    class Transcriber:
        def __init__(self) -> None:
            self.calls = 0

        async def transcribe(self, message: ChannelMessage) -> str | None:  # noqa: ARG002
            self.calls += 1
            return "transcribed"

    transcriber = Transcriber()
    message = ChannelMessage(
        conversation_id="chat",
        text="video",
        metadata={"media_type": "video", "media_path": "clip.mp4"},
    )

    updated = await transcribe_voice_message(transcriber, message)

    assert transcriber.calls == 1
    assert "transcribed" in updated.text


async def test_transcribe_voice_message_transcribes_audio_document() -> None:
    class Transcriber:
        def __init__(self) -> None:
            self.calls = 0

        async def transcribe(self, message: ChannelMessage) -> str | None:  # noqa: ARG002
            self.calls += 1
            return "transcribed"

    transcriber = Transcriber()
    message = ChannelMessage(
        conversation_id="chat",
        text="",
        metadata={
            "media_type": "voice",
            "media_path": "song.mp3",
            "media_mime_types": ["audio/mpeg"],
        },
    )

    updated = await transcribe_voice_message(transcriber, message)

    assert transcriber.calls == 1
    assert "transcribed" in updated.text


async def test_transcribe_voice_message_ignores_plain_document() -> None:
    class Transcriber:
        def __init__(self) -> None:
            self.calls = 0

        async def transcribe(self, message: ChannelMessage) -> str | None:  # noqa: ARG002
            self.calls += 1
            return "transcribed"

    transcriber = Transcriber()
    message = ChannelMessage(
        conversation_id="chat",
        text="doc",
        metadata={"media_type": "document", "media_path": "report.pdf"},
    )

    updated = await transcribe_voice_message(transcriber, message)

    assert updated == message
    assert transcriber.calls == 0
