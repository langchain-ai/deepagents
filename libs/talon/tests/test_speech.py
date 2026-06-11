from pathlib import Path

from deepagents_talon.config import TalonConfig
from deepagents_talon.speech import (
    DEFAULT_LOCAL_VOICE_TRANSCRIPTION_MODEL,
    LocalParakeetVoiceTranscriber,
    OpenAIVoiceTranscriber,
    build_voice_transcriber,
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
