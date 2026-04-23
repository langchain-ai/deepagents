"""Unit tests for the FFmpeg wrapper (mocked subprocess; no ffmpeg required)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import deepagents.middleware._ffmpeg as _ffmpeg_module
from deepagents.middleware._ffmpeg import (
    ExtractedFrame,
    ExtractionError,
    ExtractionFailedError,
    ExtractionParams,
    FFmpegMissingError,
    FileCorruptError,
    NoVideoStreamError,
    check_ffmpeg_available,
    probe_duration,
    probe_has_video_stream,
)


def _fake_completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["ffprobe"], returncode=returncode, stdout=stdout, stderr="")


class TestDataclasses:
    def test_extraction_params_immutable(self) -> None:
        params = ExtractionParams(max_frames=95, scene_threshold=0.3, max_width=1024, jpeg_quality=5)
        with pytest.raises(FrozenInstanceError):
            params.max_frames = 10  # type: ignore[misc]

    def test_extracted_frame_immutable(self) -> None:
        frame = ExtractedFrame(jpeg_bytes=b"\xff\xd8\xff", timestamp_s=1.5)
        with pytest.raises(FrozenInstanceError):
            frame.timestamp_s = 2.0  # type: ignore[misc]

    def test_extraction_params_hashable(self) -> None:
        # Required so extraction params can be part of the cache key.
        p1 = ExtractionParams(max_frames=95, scene_threshold=0.3, max_width=1024, jpeg_quality=5)
        p2 = ExtractionParams(max_frames=95, scene_threshold=0.3, max_width=1024, jpeg_quality=5)
        assert hash(p1) == hash(p2)


class TestCheckFfmpegAvailable:
    def test_returns_true_when_both_binaries_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ffmpeg_module.check_ffmpeg_available.cache_clear()
        monkeypatch.setattr(_ffmpeg_module.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert check_ffmpeg_available() is True

    def test_returns_false_when_ffmpeg_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ffmpeg_module.check_ffmpeg_available.cache_clear()
        monkeypatch.setattr(
            _ffmpeg_module.shutil,
            "which",
            lambda name: None if name == "ffmpeg" else "/usr/bin/ffprobe",
        )
        assert check_ffmpeg_available() is False

    def test_returns_false_when_ffprobe_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ffmpeg_module.check_ffmpeg_available.cache_clear()
        monkeypatch.setattr(
            _ffmpeg_module.shutil,
            "which",
            lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
        )
        assert check_ffmpeg_available() is False


class TestErrorHierarchy:
    def test_all_errors_inherit_from_extraction_error(self) -> None:
        for cls in (FFmpegMissingError, NoVideoStreamError, FileCorruptError, ExtractionFailedError):
            assert issubclass(cls, ExtractionError)
            assert issubclass(cls, Exception)


class TestProbeDuration:
    def test_returns_float_for_valid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = json.dumps({"streams": [{"codec_type": "video"}], "format": {"duration": "45.25"}})
        monkeypatch.setattr(_ffmpeg_module.subprocess, "run", lambda *_a, **_kw: _fake_completed(payload))

        assert probe_duration(Path("/tmp/video.mp4")) == pytest.approx(45.25)  # noqa: S108

    def test_returns_none_for_missing_duration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = json.dumps({"streams": [{"codec_type": "video"}], "format": {}})
        monkeypatch.setattr(_ffmpeg_module.subprocess, "run", lambda *_a, **_kw: _fake_completed(payload))

        assert probe_duration(Path("/tmp/video.mp4")) is None  # noqa: S108

    def test_returns_none_on_ffprobe_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_error(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.CalledProcessError(returncode=1, cmd=["ffprobe"])

        monkeypatch.setattr(_ffmpeg_module.subprocess, "run", raise_error)
        assert probe_duration(Path("/tmp/video.mp4")) is None  # noqa: S108

    def test_returns_none_on_unparseable_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_ffmpeg_module.subprocess, "run", lambda *_a, **_kw: _fake_completed("not json"))
        assert probe_duration(Path("/tmp/video.mp4")) is None  # noqa: S108


class TestProbeHasVideoStream:
    def test_true_when_video_stream_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = json.dumps({"streams": [{"codec_type": "video"}]})
        monkeypatch.setattr(_ffmpeg_module.subprocess, "run", lambda *_a, **_kw: _fake_completed(payload))

        assert probe_has_video_stream(Path("/tmp/video.mp4")) is True  # noqa: S108

    def test_false_when_only_audio_streams(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = json.dumps({"streams": [{"codec_type": "audio"}]})
        monkeypatch.setattr(_ffmpeg_module.subprocess, "run", lambda *_a, **_kw: _fake_completed(payload))

        assert probe_has_video_stream(Path("/tmp/mp3-as-mp4.mp4")) is False  # noqa: S108

    def test_false_when_no_streams_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_ffmpeg_module.subprocess, "run", lambda *_a, **_kw: _fake_completed(json.dumps({})))
        assert probe_has_video_stream(Path("/tmp/x.mp4")) is False  # noqa: S108

    def test_false_on_ffprobe_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_error(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.CalledProcessError(returncode=1, cmd=["ffprobe"])

        monkeypatch.setattr(_ffmpeg_module.subprocess, "run", raise_error)
        assert probe_has_video_stream(Path("/tmp/x.mp4")) is False  # noqa: S108


class TestExtractFrames:
    def test_builds_expected_ffmpeg_command(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Verify the exact ffmpeg command, including the escaped filter chain."""
        captured: dict[str, object] = {}

        def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            captured["cmd"] = cmd
            # Simulate ffmpeg writing two frames and a metadata file.
            # Real ffmpeg metadata=print output puts pts:N between frame:N and
            # pts_time:T on the same line, so test data mirrors that.
            tmpdir = Path(cmd[-1]).parent
            (tmpdir / "frame_00001.jpg").write_bytes(b"\xff\xd8\xff\xe0jpg1")
            (tmpdir / "frame_00002.jpg").write_bytes(b"\xff\xd8\xff\xe0jpg2")
            (tmpdir / "ts.txt").write_text(
                "frame:0    pts:0       pts_time:0.000000\n"
                "lavfi.scene_score=0.000000\n"
                "frame:1    pts:90000   pts_time:1.000000\n"
                "lavfi.scene_score=0.123456\n"
            )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        _ffmpeg_module.check_ffmpeg_available.cache_clear()
        monkeypatch.setattr(_ffmpeg_module.shutil, "which", lambda _name: f"/usr/bin/{_name}")
        monkeypatch.setattr(_ffmpeg_module, "probe_duration", lambda _p: 5.0)
        monkeypatch.setattr(_ffmpeg_module, "probe_has_video_stream", lambda _p: True)
        monkeypatch.setattr(_ffmpeg_module.subprocess, "run", fake_run)

        src = tmp_path / "src.mp4"
        src.write_bytes(b"fake-bytes")
        params = ExtractionParams(max_frames=95, scene_threshold=0.3, max_width=1024, jpeg_quality=5)

        frames, duration = _ffmpeg_module.extract_frames(src, params)

        assert duration == pytest.approx(5.0)
        assert len(frames) == 2
        assert frames[0].timestamp_s == pytest.approx(0.0)
        assert frames[1].timestamp_s == pytest.approx(1.0)
        assert frames[0].jpeg_bytes == b"\xff\xd8\xff\xe0jpg1"

        cmd = captured["cmd"]
        assert isinstance(cmd, list)
        cmd_strs: list[str] = [str(s) for s in cmd]
        assert cmd_strs[0] == "ffmpeg"
        joined = " ".join(cmd_strs)
        # baseline_interval = max(1.0, 5.0 / 95) = 1.0
        assert "gte(t-prev_selected_t\\,1.0)" in joined
        assert "gt(scene\\,0.3)" in joined
        assert "scale='min(1024\\,iw)':-2" in joined
        assert "-q:v" in cmd_strs and "5" in cmd_strs
        assert "-fps_mode" in cmd_strs and "vfr" in cmd_strs

    def test_trims_to_max_frames(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            tmpdir = Path(cmd[-1]).parent
            ts_lines = []
            for i in range(5):  # ffmpeg overshoots; wrapper must trim.
                (tmpdir / f"frame_{i + 1:05d}.jpg").write_bytes(b"jpg")
                ts_lines.append(f"frame:{i}    pts:{i * 90000}   pts_time:{i}.000000\n")
            (tmpdir / "ts.txt").write_text("".join(ts_lines))
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        _ffmpeg_module.check_ffmpeg_available.cache_clear()
        monkeypatch.setattr(_ffmpeg_module.shutil, "which", lambda _name: f"/usr/bin/{_name}")
        monkeypatch.setattr(_ffmpeg_module, "probe_duration", lambda _p: 10.0)
        monkeypatch.setattr(_ffmpeg_module, "probe_has_video_stream", lambda _p: True)
        monkeypatch.setattr(_ffmpeg_module.subprocess, "run", fake_run)

        src = tmp_path / "src.mp4"
        src.write_bytes(b"bytes")
        params = ExtractionParams(max_frames=3, scene_threshold=0.3, max_width=1024, jpeg_quality=5)

        frames, duration = _ffmpeg_module.extract_frames(src, params)
        assert duration == pytest.approx(10.0)
        assert len(frames) == 3
        assert [f.timestamp_s for f in frames] == pytest.approx([0.0, 1.0, 2.0])

    def test_raises_ffmpeg_missing_when_unavailable(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _ffmpeg_module.check_ffmpeg_available.cache_clear()
        monkeypatch.setattr(_ffmpeg_module.shutil, "which", lambda _name: None)

        src = tmp_path / "src.mp4"
        src.write_bytes(b"x")
        params = ExtractionParams(max_frames=95, scene_threshold=0.3, max_width=1024, jpeg_quality=5)

        with pytest.raises(FFmpegMissingError):
            _ffmpeg_module.extract_frames(src, params)

    def test_raises_no_video_stream(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _ffmpeg_module.check_ffmpeg_available.cache_clear()
        monkeypatch.setattr(_ffmpeg_module.shutil, "which", lambda _name: f"/usr/bin/{_name}")
        monkeypatch.setattr(_ffmpeg_module, "probe_has_video_stream", lambda _p: False)

        src = tmp_path / "src.mp4"
        src.write_bytes(b"x")
        params = ExtractionParams(max_frames=95, scene_threshold=0.3, max_width=1024, jpeg_quality=5)

        with pytest.raises(NoVideoStreamError):
            _ffmpeg_module.extract_frames(src, params)

    def test_raises_file_corrupt_for_missing_duration(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _ffmpeg_module.check_ffmpeg_available.cache_clear()
        monkeypatch.setattr(_ffmpeg_module.shutil, "which", lambda _name: f"/usr/bin/{_name}")
        monkeypatch.setattr(_ffmpeg_module, "probe_has_video_stream", lambda _p: True)
        monkeypatch.setattr(_ffmpeg_module, "probe_duration", lambda _p: None)

        src = tmp_path / "src.mp4"
        src.write_bytes(b"x")
        params = ExtractionParams(max_frames=95, scene_threshold=0.3, max_width=1024, jpeg_quality=5)

        with pytest.raises(FileCorruptError):
            _ffmpeg_module.extract_frames(src, params)

    def test_raises_extraction_failed_on_nonzero_exit(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _ffmpeg_module.check_ffmpeg_available.cache_clear()
        monkeypatch.setattr(_ffmpeg_module.shutil, "which", lambda _name: f"/usr/bin/{_name}")
        monkeypatch.setattr(_ffmpeg_module, "probe_has_video_stream", lambda _p: True)
        monkeypatch.setattr(_ffmpeg_module, "probe_duration", lambda _p: 5.0)

        def raise_called(*_a: object, **_kw: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.CalledProcessError(1, ["ffmpeg"], stderr="boom")

        monkeypatch.setattr(_ffmpeg_module.subprocess, "run", raise_called)

        src = tmp_path / "src.mp4"
        src.write_bytes(b"x")
        params = ExtractionParams(max_frames=95, scene_threshold=0.3, max_width=1024, jpeg_quality=5)

        with pytest.raises(ExtractionFailedError, match="boom"):
            _ffmpeg_module.extract_frames(src, params)
