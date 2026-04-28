"""Local speech-to-text transcription using NVIDIA Parakeet TDT 0.6B v3.

The NeMo model is loaded lazily on the first call and cached for subsequent
ones (thread-safe double-checked singleton). Inference runs on CPU by default;
set ``SPEECH_DEVICE=cuda`` to use a GPU if one is available.

OGG/Opus files — the codec WhatsApp uses for push-to-talk messages — are
converted to 16 kHz mono WAV via ffmpeg before being handed to NeMo, which
requires WAV or FLAC input.

Enable with ``SPEECH_ENABLED=true`` in the environment. The ``nemo_toolkit``
package must be installed separately (see the ``speech`` optional dependency
in ``pyproject.toml``).
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"

_model: object | None = None
_lock = threading.Lock()  # serializes both model load and inference calls


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_model() -> object:
    """Return the cached NeMo ASR model, loading it on first call.

    Raises:
        ImportError: If ``nemo_toolkit[asr]`` is not installed.
    """
    global _model  # noqa: PLW0603  # intentional module-level singleton

    if _model is not None:
        return _model

    try:
        import nemo.collections.asr as nemo_asr  # noqa: PLC0415  # deferred import
    except ImportError as exc:
        msg = (
            "nemo_toolkit[asr] is required for local speech transcription. "
            "Install it with:  pip install 'nemo_toolkit[asr]'"
        )
        raise ImportError(msg) from exc

    device = os.getenv("SPEECH_DEVICE", "cpu")
    logger.info(
        "Loading %s on device=%s (first run will download ~2.4 GB)...",
        _MODEL_NAME,
        device,
    )

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=_MODEL_NAME)
    model = model.to(device)
    model.eval()
    _model = model

    logger.info("Parakeet TDT 0.6B v3 ready on %s", device)
    return _model


def _convert_to_wav(src: str) -> str:
    """Convert *src* to a 16 kHz mono WAV temp file using ffmpeg.

    The caller is responsible for deleting the returned path.

    Args:
        src: Path to the source audio file (OGG/Opus, MP3, M4A, etc.).

    Returns:
        Path to the temporary 16 kHz mono WAV file.

    Raises:
        RuntimeError: If ffmpeg is missing or conversion fails.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    dst = tmp.name

    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src,
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                dst,
            ],
            capture_output=True,
            timeout=120,
        )
    except FileNotFoundError as exc:
        msg = (
            "ffmpeg not found on PATH. "
            "Install it (e.g. `apt install ffmpeg` or `brew install ffmpeg`) "
            "to enable voice message transcription."
        )
        raise RuntimeError(msg) from exc
    except subprocess.TimeoutExpired as exc:
        msg = f"ffmpeg timed out while converting {src}"
        raise RuntimeError(msg) from exc

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        msg = f"ffmpeg conversion failed (exit {proc.returncode}): {stderr}"
        raise RuntimeError(msg)

    return dst


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transcribe(audio_path: str) -> str:
    """Transcribe *audio_path* to text using Parakeet TDT 0.6B v3 on CPU.

    The audio file is first converted to 16 kHz mono WAV via ffmpeg (in-process
    subprocess, fast), then fed to the NeMo ASR model. The model is loaded once
    and reused; inference calls are serialized via a threading lock to avoid
    memory thrashing on CPU.

    Args:
        audio_path: Path to the input audio file (OGG, WAV, FLAC, MP3, etc.).

    Returns:
        Transcribed text, or an empty string if transcription fails or the
        audio contains no recognisable speech.
    """
    wav_path: str | None = None
    try:
        wav_path = _convert_to_wav(audio_path)

        with _lock:
            model = _load_model()
            outputs = model.transcribe([wav_path])

        if not outputs:
            return ""

        result = outputs[0]
        # NeMo 2.x returns objects with a .text attribute; older builds
        # returned plain strings — handle both gracefully.
        text: str = result.text if hasattr(result, "text") else str(result)
        return text.strip()

    except Exception as exc:
        logger.warning("Voice transcription failed for %s: %s", audio_path, exc)
        return ""
    finally:
        if wav_path:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except OSError:
                pass
