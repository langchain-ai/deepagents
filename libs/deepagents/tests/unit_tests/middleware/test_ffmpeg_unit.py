"""Unit tests for the FFmpeg wrapper (mocked subprocess; no ffmpeg required)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from deepagents.middleware._ffmpeg import (
    ExtractedFrame,
    ExtractionParams,
    ExtractionFailedError,
    FFmpegMissingError,
    FileCorruptError,
    NoVideoStreamError,
    check_ffmpeg_available,
)


class TestDataclasses:
    def test_extraction_params_immutable(self) -> None:
        params = ExtractionParams(
            max_frames=95, scene_threshold=0.3, max_width=1024, jpeg_quality=5
        )
        with pytest.raises(FrozenInstanceError):
            params.max_frames = 10  # type: ignore[misc]

    def test_extracted_frame_immutable(self) -> None:
        frame = ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff", timestamp_s=1.5)
        with pytest.raises(FrozenInstanceError):
            frame.timestamp_s = 2.0  # type: ignore[misc]

    def test_extraction_params_hashable(self) -> None:
        # Required so extraction params can be part of the cache key.
        p1 = ExtractionParams(
            max_frames=95, scene_threshold=0.3, max_width=1024, jpeg_quality=5
        )
        p2 = ExtractionParams(
            max_frames=95, scene_threshold=0.3, max_width=1024, jpeg_quality=5
        )
        assert hash(p1) == hash(p2)


class TestCheckFfmpegAvailable:
    def test_returns_true_when_both_binaries_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import deepagents.middleware._ffmpeg as module

        module.check_ffmpeg_available.cache_clear()
        monkeypatch.setattr(module.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert check_ffmpeg_available() is True

    def test_returns_false_when_ffmpeg_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import deepagents.middleware._ffmpeg as module

        module.check_ffmpeg_available.cache_clear()
        monkeypatch.setattr(
            module.shutil,
            "which",
            lambda name: None if name == "ffmpeg" else "/usr/bin/ffprobe",
        )
        assert check_ffmpeg_available() is False

    def test_returns_false_when_ffprobe_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import deepagents.middleware._ffmpeg as module

        module.check_ffmpeg_available.cache_clear()
        monkeypatch.setattr(
            module.shutil,
            "which",
            lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
        )
        assert check_ffmpeg_available() is False


class TestErrorHierarchy:
    def test_all_errors_inherit_from_extraction_error(self) -> None:
        from deepagents.middleware._ffmpeg import ExtractionError

        for cls in (FFmpegMissingError, NoVideoStreamError, FileCorruptError, ExtractionFailedError):
            assert issubclass(cls, ExtractionError)
            assert issubclass(cls, Exception)


import json
import subprocess
from pathlib import Path

from deepagents.middleware._ffmpeg import (
    probe_duration,
    probe_has_video_stream,
)


def _fake_completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ffprobe"], returncode=returncode, stdout=stdout, stderr=""
    )


class TestProbeDuration:
    def test_returns_float_for_valid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import deepagents.middleware._ffmpeg as module

        payload = json.dumps({"streams": [{"codec_type": "video"}], "format": {"duration": "45.25"}})
        monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: _fake_completed(payload))

        assert probe_duration(Path("/tmp/video.mp4")) == pytest.approx(45.25)

    def test_returns_none_for_missing_duration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import deepagents.middleware._ffmpeg as module

        payload = json.dumps({"streams": [{"codec_type": "video"}], "format": {}})
        monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: _fake_completed(payload))

        assert probe_duration(Path("/tmp/video.mp4")) is None

    def test_returns_none_on_ffprobe_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import deepagents.middleware._ffmpeg as module

        def raise_error(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.CalledProcessError(returncode=1, cmd=["ffprobe"])

        monkeypatch.setattr(module.subprocess, "run", raise_error)
        assert probe_duration(Path("/tmp/video.mp4")) is None

    def test_returns_none_on_unparseable_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import deepagents.middleware._ffmpeg as module

        monkeypatch.setattr(
            module.subprocess, "run", lambda *a, **kw: _fake_completed("not json")
        )
        assert probe_duration(Path("/tmp/video.mp4")) is None


class TestProbeHasVideoStream:
    def test_true_when_video_stream_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import deepagents.middleware._ffmpeg as module

        payload = json.dumps({"streams": [{"codec_type": "video"}]})
        monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: _fake_completed(payload))

        assert probe_has_video_stream(Path("/tmp/video.mp4")) is True

    def test_false_when_only_audio_streams(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import deepagents.middleware._ffmpeg as module

        payload = json.dumps({"streams": [{"codec_type": "audio"}]})
        monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: _fake_completed(payload))

        assert probe_has_video_stream(Path("/tmp/mp3-as-mp4.mp4")) is False

    def test_false_when_no_streams_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import deepagents.middleware._ffmpeg as module

        monkeypatch.setattr(
            module.subprocess, "run", lambda *a, **kw: _fake_completed(json.dumps({}))
        )
        assert probe_has_video_stream(Path("/tmp/x.mp4")) is False

    def test_false_on_ffprobe_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import deepagents.middleware._ffmpeg as module

        def raise_error(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.CalledProcessError(returncode=1, cmd=["ffprobe"])

        monkeypatch.setattr(module.subprocess, "run", raise_error)
        assert probe_has_video_stream(Path("/tmp/x.mp4")) is False
