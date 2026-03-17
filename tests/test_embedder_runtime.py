"""embedder runtime 契约测试。"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from vat.embedder.ffmpeg_wrapper import FFmpegWrapper


class TestFFmpegWrapperInit:
    def test_init_raises_when_ffmpeg_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)

        with pytest.raises(RuntimeError, match="ffmpeg未安装"):
            FFmpegWrapper()


class TestFFmpegWrapperHelpers:
    def test_check_encoder_support_returns_false_on_subprocess_error(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

        assert wrapper._check_encoder_support("h264_nvenc") is False

    def test_get_video_info_parses_ffprobe_json(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
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

        info = wrapper.get_video_info(video)

        assert info["duration"] == 10.5
        assert info["video"]["codec"] == "h264"
        assert info["audio"]["codec"] == "aac"

    def test_get_video_info_returns_none_on_subprocess_failure(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        video.write_bytes(b"00")
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

        assert wrapper.get_video_info(video) is None

    def test_scale_ass_style_applies_portrait_rules(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        style = "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,2,0,7,10,10,10,1"

        scaled = wrapper._scale_ass_style(style, 1.5, video_width=720, video_height=1280)

        assert ",30," in scaled  # fontsize 20 -> 30
        assert ",36,36,30," in scaled  # MarginL/R 至少 5% 宽度，MarginV 翻倍


class TestFFmpegWrapperSoftEmbedContracts:
    def test_embed_subtitle_soft_returns_false_when_video_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()

        result = wrapper.embed_subtitle_soft(tmp_path / "missing.mp4", tmp_path / "sub.srt", tmp_path / "out.mkv")

        assert result is False

    def test_embed_subtitle_soft_returns_false_when_subtitle_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        video.write_bytes(b"00")

        result = wrapper.embed_subtitle_soft(video, tmp_path / "missing.srt", tmp_path / "out.mkv")

        assert result is False

    def test_embed_subtitle_soft_mp4_uses_mov_text_and_succeeds(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        calls = []

        def fake_run(cmd, capture_output, text, check):
            calls.append(cmd)
            out.write_bytes(b"11")
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)

        result = wrapper.embed_subtitle_soft(video, sub, out)

        assert result is True
        assert "-c:s" in calls[0]
        assert "mov_text" in calls[0]

    def test_extract_audio_returns_false_when_input_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()

        assert wrapper.extract_audio(tmp_path / "missing.mp4", tmp_path / "audio.wav") is False

    def test_extract_audio_returns_true_when_ffmpeg_creates_output(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        video.write_bytes(b"00")

        def fake_run(cmd, capture_output, text, check):
            audio.write_bytes(b"11")
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)

        assert wrapper.extract_audio(video, audio) is True

    def test_extract_audio_returns_false_on_ffmpeg_failure(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        video.write_bytes(b"00")

        class _CalledProcessError(Exception):
            stderr = "failed"

        def fake_run(cmd, capture_output, text, check):
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="failed")

        monkeypatch.setattr("subprocess.run", fake_run)

        assert wrapper.extract_audio(video, audio) is False


class TestFFmpegWrapperHardEmbedContracts:
    def test_embed_subtitle_hard_rejects_cpu_gpu_device(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.srt"
        out = tmp_path / "out.mp4"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")

        with pytest.raises(RuntimeError, match="禁止 CPU 回退"):
            wrapper.embed_subtitle_hard(video, sub, out, gpu_device="cpu")

    def test_embed_subtitle_hard_raises_when_nvenc_slot_acquire_times_out(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.srt"
        out = tmp_path / "out.mp4"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")

        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: None)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.select_gpu", lambda: 1)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire", lambda gpu_id, timeout=600: False)

        with pytest.raises(RuntimeError, match="获取超时"):
            wrapper.embed_subtitle_hard(video, sub, out, gpu_device="auto")

    def test_embed_subtitle_hard_releases_session_when_nvenc_not_supported(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.srt"
        out = tmp_path / "out.mp4"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        released = []

        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: None)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.select_gpu", lambda: 2)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire", lambda gpu_id, timeout=600: True)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: released.append(gpu_id))
        monkeypatch.setattr(wrapper, "_check_nvenc_support", lambda: False)

        with pytest.raises(RuntimeError, match="不支持 NVENC"):
            wrapper.embed_subtitle_hard(video, sub, out, gpu_device="auto")

        assert released == [2]
