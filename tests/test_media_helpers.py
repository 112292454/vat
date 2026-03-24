"""media helpers 契约测试。"""

import json
import subprocess
from types import SimpleNamespace

import pytest

from vat.media.audio import extract_audio_ffmpeg
from vat.media.probe import probe_media_info


class TestProbeMediaInfo:
    def test_probe_media_info_parses_ffprobe_json(self, monkeypatch, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"00")
        payload = {
            "format": {"duration": "10.5", "size": "12345", "bit_rate": "999"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080, "r_frame_rate": "30000/1001"},
                {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2},
            ],
        }
        monkeypatch.setattr(
            "subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(payload)),
        )

        info = probe_media_info(video)

        assert info["duration"] == 10.5
        assert info["video"]["fps"] > 29
        assert info["audio"]["codec"] == "aac"

    def test_probe_media_info_returns_none_when_ffprobe_missing(self, monkeypatch, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"00")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("ffprobe missing")),
        )

        assert probe_media_info(video) is None


class TestExtractAudioFfmpeg:
    def test_extract_audio_ffmpeg_raises_when_input_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_audio_ffmpeg(tmp_path / "missing.mp4", tmp_path / "audio.wav")

    def test_extract_audio_ffmpeg_invokes_ffmpeg(self, monkeypatch, tmp_path):
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        video.write_bytes(b"00")
        calls = []

        def fake_run(cmd, check, capture_output, text):
            calls.append(cmd)
            audio.write_bytes(b"11")
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)

        extract_audio_ffmpeg(video, audio)

        assert calls
        assert audio.exists()

    def test_extract_audio_ffmpeg_raises_runtime_error_on_failure(self, monkeypatch, tmp_path):
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        video.write_bytes(b"00")

        def fake_run(cmd, check, capture_output, text):
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="failed")

        monkeypatch.setattr("subprocess.run", fake_run)

        with pytest.raises(RuntimeError, match="音频提取失败"):
            extract_audio_ffmpeg(video, audio)
