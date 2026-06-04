"""Optional inbound voice transcription for Talon channels."""

from __future__ import annotations

import asyncio
import importlib
import logging
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from deepagents_talon.interfaces import ChannelMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents_talon.config import TalonConfig

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_VOICE_TRANSCRIPTION_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
_DEFAULT_LOCAL_VOICE_DEVICE = "cpu"
_local_models: dict[tuple[str, str], _LocalSpeechModel] = {}
_local_model_lock = threading.Lock()


class VoiceTranscriber(Protocol):
    """Turn channel voice payloads into text."""

    async def transcribe(self, message: ChannelMessage) -> str | None:
        """Return transcribed text for a voice message.

        Args:
            message: Inbound channel message.

        Returns:
            Transcribed text, or `None` when the message cannot be transcribed.
        """


class _LocalSpeechModel(Protocol):
    def to(self, device: str) -> _LocalSpeechModel:
        """Move the model to the target inference device."""

    def eval(self) -> None:
        """Put the model in inference mode."""

    def transcribe(self, audio_paths: list[str]) -> Sequence[object]:
        """Transcribe audio paths and return NeMo result objects."""


@dataclass(frozen=True, slots=True)
class LocalParakeetVoiceTranscriber:
    """Voice transcriber backed by local NVIDIA Parakeet ASR through NeMo.

    Args:
        model: NeMo model identifier to load.
        device: Inference device for the local model.
    """

    model: str = DEFAULT_LOCAL_VOICE_TRANSCRIPTION_MODEL
    device: str = _DEFAULT_LOCAL_VOICE_DEVICE

    async def transcribe(self, message: ChannelMessage) -> str | None:
        """Transcribe the local audio path in message metadata.

        Args:
            message: Inbound channel message with `voice_path` or `media_path`.

        Returns:
            Transcribed text, or `None` when local transcription is unavailable.
        """
        path = _voice_path(message)
        if path is None:
            return None
        if not path.is_file():
            logger.warning("Voice transcription skipped because media file is missing: %s", path)
            return None
        text = await _transcribe_local(path, model=self.model, device=self.device)
        return text or None


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
    enabled = _first_config_value(
        config,
        "DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_ENABLED",
        "SPEECH_ENABLED",
    ).lower()
    if enabled not in {"1", "true", "yes"}:
        return None
    model = _first_config_value(config, "DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_MODEL")
    if not model or _is_local_voice_model(model):
        device = _first_config_value(
            config,
            "DEEPAGENTS_TALON_VOICE_TRANSCRIPTION_DEVICE",
            "SPEECH_DEVICE",
            default=_DEFAULT_LOCAL_VOICE_DEVICE,
        )
        return LocalParakeetVoiceTranscriber(
            model=model or DEFAULT_LOCAL_VOICE_TRANSCRIPTION_MODEL,
            device=device,
        )
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


def _first_config_value(config: TalonConfig, *keys: str, default: str = "") -> str:
    for key in keys:
        value = config.env.get(key)
        if value:
            return value
    return default


def _is_local_voice_model(model: str) -> bool:
    return model == DEFAULT_LOCAL_VOICE_TRANSCRIPTION_MODEL or model.startswith("nvidia/parakeet")


async def _transcribe_local(path: Path, *, model: str, device: str) -> str:
    return await asyncio.to_thread(_transcribe_local_sync, path, model=model, device=device)


def _transcribe_local_sync(path: Path, *, model: str, device: str) -> str:
    wav_path: Path | None = None
    try:
        wav_path = _convert_to_wav(path)
        with _local_model_lock:
            speech_model = _load_local_model(model, device)
            outputs = speech_model.transcribe([str(wav_path)])
        if not outputs:
            return ""
        result = outputs[0]
        text = getattr(result, "text", None)
        return text.strip() if isinstance(text, str) else str(result).strip()
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("Voice transcription failed for %s: %s", path, exc)
        return ""
    finally:
        if wav_path is not None:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not delete temporary voice transcription file: %s", wav_path)


def _load_local_model(model: str, device: str) -> _LocalSpeechModel:
    key = (model, device)
    cached = _local_models.get(key)
    if cached is not None:
        return cached

    try:
        module = importlib.import_module("nemo.collections.asr")
    except ImportError as exc:
        msg = (
            "nemo_toolkit[asr] is required for local voice transcription. "
            "Install it with `nemo_toolkit[asr]` and ensure ffmpeg is on PATH."
        )
        raise ImportError(msg) from exc

    logger.info("Loading local voice transcription model %s on device=%s", model, device)
    asr_model = module.models.ASRModel
    loaded = cast("_LocalSpeechModel", asr_model.from_pretrained(model_name=model))
    loaded = loaded.to(device)
    loaded.eval()
    _local_models[key] = loaded
    logger.info("Local voice transcription model %s ready on device=%s", model, device)
    return loaded


def _convert_to_wav(path: Path) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    output = Path(tmp_path)
    try:
        proc = subprocess.run(  # noqa: S603  # ffmpeg receives local media paths only
            [  # noqa: S607  # use ffmpeg from PATH, matching the local example workflow
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "wav",
                str(output),
            ],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError as exc:
        msg = "ffmpeg not found on PATH; install ffmpeg to enable voice transcription."
        raise RuntimeError(msg) from exc
    except subprocess.TimeoutExpired as exc:
        msg = f"ffmpeg timed out while converting {path}"
        raise RuntimeError(msg) from exc

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        msg = f"ffmpeg conversion failed with exit code {proc.returncode}: {stderr}"
        raise RuntimeError(msg)
    return output
