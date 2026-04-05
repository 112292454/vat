"""embedder runtime 契约测试。"""

import json
import subprocess
import sys
import tempfile
import types
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


class TestFFmpegWrapperConvertVideoContracts:
    def test_prepare_convert_video_preflight_creates_output_directory(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        out = tmp_path / "nested" / "output.mp4"

        wrapper._prepare_convert_video_preflight(output_path=out)

        assert out.parent.is_dir()

    def test_plan_convert_video_command_builds_expected_ffmpeg_args(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        src = tmp_path / "input.mov"
        out = tmp_path / "nested" / "output.mp4"

        cmd = wrapper._plan_convert_video_command(
            input_path=src,
            output_path=out,
            video_codec="libx265",
            audio_codec="copy",
            crf=19,
            preset="slow",
        )

        assert cmd == [
            "ffmpeg",
            "-i", str(src),
            "-c:v", "libx265",
            "-crf", "19",
            "-preset", "slow",
            "-c:a", "copy",
            "-y",
            str(out),
        ]

    def test_convert_video_delegates_preflight_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        src = tmp_path / "input.mov"
        out = tmp_path / "nested" / "output.mp4"
        delegated = []

        monkeypatch.setattr(
            wrapper,
            "_prepare_convert_video_preflight",
            lambda **kwargs: delegated.append(("preflight", kwargs)),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_convert_video_command",
            lambda **kwargs: delegated.append(("plan", kwargs)) or ["ffmpeg", "planned", str(out)],
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_run_convert_video_runtime_stage",
            lambda **kwargs: delegated.append(("run", kwargs)) or True,
            raising=False,
        )

        result = wrapper.convert_video(
            src,
            out,
            video_codec="libx265",
            audio_codec="copy",
            crf=19,
            preset="slow",
        )

        assert result is True
        assert delegated == [
            ("preflight", {"output_path": out}),
            ("plan", {
                "input_path": src,
                "output_path": out,
                "video_codec": "libx265",
                "audio_codec": "copy",
                "crf": 19,
                "preset": "slow",
            }),
            ("run", {"cmd": ["ffmpeg", "planned", str(out)]}),
        ]

    def test_convert_video_delegates_command_planning_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        src = tmp_path / "input.mov"
        out = tmp_path / "nested" / "output.mp4"
        planned = []
        calls = []

        def fake_plan(**kwargs):
            planned.append(kwargs)
            return ["ffmpeg", "planned", str(out)]

        def fake_run(cmd, capture_output, text, check):
            calls.append(cmd)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(wrapper, "_plan_convert_video_command", fake_plan, raising=False)
        monkeypatch.setattr("subprocess.run", fake_run)

        result = wrapper.convert_video(
            src,
            out,
            video_codec="libx265",
            audio_codec="copy",
            crf=19,
            preset="slow",
        )

        assert result is True
        assert planned == [{
            "input_path": src,
            "output_path": out,
            "video_codec": "libx265",
            "audio_codec": "copy",
            "crf": 19,
            "preset": "slow",
        }]
        assert calls == [["ffmpeg", "planned", str(out)]]

    def test_convert_video_returns_true_when_ffmpeg_succeeds(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        src = tmp_path / "input.mov"
        out = tmp_path / "nested" / "output.mp4"
        calls = []

        def fake_run(cmd, capture_output, text, check):
            calls.append(cmd)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)

        result = wrapper.convert_video(src, out)

        assert result is True
        assert out.parent.exists()
        assert calls[0][0] == "ffmpeg"

    def test_convert_video_returns_false_and_reports_stderr_on_ffmpeg_failure(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        src = tmp_path / "input.mov"
        out = tmp_path / "nested" / "output.mp4"

        def fake_run(cmd, capture_output, text, check):
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="failed")

        monkeypatch.setattr("subprocess.run", fake_run)

        result = wrapper.convert_video(src, out)

        assert result is False
        captured = capsys.readouterr()
        assert "视频转换失败: failed" in captured.out

    def test_run_convert_video_runtime_stage_returns_true_when_ffmpeg_succeeds(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        calls = []

        def fake_run(cmd, capture_output, text, check):
            calls.append(cmd)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)

        result = wrapper._run_convert_video_runtime_stage(cmd=["ffmpeg", "planned", "out.mp4"])

        assert result is True
        assert calls == [["ffmpeg", "planned", "out.mp4"]]

    def test_run_convert_video_runtime_stage_returns_false_and_reports_stderr_on_failure(self, monkeypatch, capsys):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()

        def fake_run(cmd, capture_output, text, check):
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="failed")

        monkeypatch.setattr("subprocess.run", fake_run)

        result = wrapper._run_convert_video_runtime_stage(cmd=["ffmpeg", "planned", "out.mp4"])

        assert result is False
        captured = capsys.readouterr()
        assert "视频转换失败: failed" in captured.out

    def test_convert_video_delegates_runtime_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        src = tmp_path / "input.mov"
        out = tmp_path / "nested" / "output.mp4"
        delegated = []

        monkeypatch.setattr(wrapper, "_plan_convert_video_command", lambda **kwargs: ["ffmpeg", "planned", str(out)], raising=False)
        monkeypatch.setattr(
            wrapper,
            "_run_convert_video_runtime_stage",
            lambda **kwargs: delegated.append(kwargs) or True,
            raising=False,
        )
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, capture_output, text, check: SimpleNamespace(returncode=0),
        )

        result = wrapper.convert_video(src, out)

        assert result is True
        assert out.parent.exists()
        assert delegated == [{"cmd": ["ffmpeg", "planned", str(out)]}]


class TestFFmpegWrapperExtractThumbnailContracts:
    def test_prepare_extract_thumbnail_preflight_creates_output_directory(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        out = tmp_path / "nested" / "thumb.jpg"

        wrapper._prepare_extract_thumbnail_preflight(output_path=out)

        assert out.parent.is_dir()

    def test_plan_extract_thumbnail_command_builds_expected_ffmpeg_args(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "input.mov"
        out = tmp_path / "nested" / "thumb.jpg"

        cmd = wrapper._plan_extract_thumbnail_command(
            video_path=video,
            output_path=out,
            time_position="00:00:09",
        )

        assert cmd == [
            "ffmpeg",
            "-ss", "00:00:09",
            "-i", str(video),
            "-vframes", "1",
            "-q:v", "2",
            "-y",
            str(out),
        ]

    def test_extract_thumbnail_delegates_preflight_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "input.mov"
        out = tmp_path / "nested" / "thumb.jpg"
        delegated = []

        monkeypatch.setattr(
            wrapper,
            "_prepare_extract_thumbnail_preflight",
            lambda **kwargs: delegated.append(("preflight", kwargs)),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_extract_thumbnail_command",
            lambda **kwargs: delegated.append(("plan", kwargs)) or ["ffmpeg", "planned", str(out)],
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_run_extract_thumbnail_runtime_stage",
            lambda **kwargs: delegated.append(("run", kwargs)) or True,
            raising=False,
        )

        result = wrapper.extract_thumbnail(video, out, time_position="00:00:09")

        assert result is True
        assert delegated == [
            ("preflight", {"output_path": out}),
            ("plan", {
                "video_path": video,
                "output_path": out,
                "time_position": "00:00:09",
            }),
            ("run", {"cmd": ["ffmpeg", "planned", str(out)]}),
        ]

    def test_extract_thumbnail_delegates_command_planning_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "input.mov"
        out = tmp_path / "nested" / "thumb.jpg"
        planned = []
        calls = []

        def fake_plan(**kwargs):
            planned.append(kwargs)
            return ["ffmpeg", "planned", str(out)]

        def fake_run(cmd, capture_output, text, check):
            calls.append(cmd)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(wrapper, "_plan_extract_thumbnail_command", fake_plan, raising=False)
        monkeypatch.setattr("subprocess.run", fake_run)

        result = wrapper.extract_thumbnail(video, out, time_position="00:00:09")

        assert result is True
        assert out.parent.exists()
        assert planned == [{
            "video_path": video,
            "output_path": out,
            "time_position": "00:00:09",
        }]
        assert calls == [["ffmpeg", "planned", str(out)]]

    def test_extract_thumbnail_returns_true_when_ffmpeg_succeeds(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "input.mov"
        out = tmp_path / "nested" / "thumb.jpg"
        calls = []

        def fake_run(cmd, capture_output, text, check):
            calls.append(cmd)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)

        result = wrapper.extract_thumbnail(video, out)

        assert result is True
        assert out.parent.exists()
        assert calls[0][0] == "ffmpeg"

    def test_extract_thumbnail_returns_false_and_reports_stderr_on_ffmpeg_failure(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "input.mov"
        out = tmp_path / "nested" / "thumb.jpg"

        def fake_run(cmd, capture_output, text, check):
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="failed")

        monkeypatch.setattr("subprocess.run", fake_run)

        result = wrapper.extract_thumbnail(video, out)

        assert result is False
        captured = capsys.readouterr()
        assert "缩略图提取失败: failed" in captured.out

    def test_run_extract_thumbnail_runtime_stage_returns_true_when_ffmpeg_succeeds(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        calls = []

        def fake_run(cmd, capture_output, text, check):
            calls.append(cmd)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)

        result = wrapper._run_extract_thumbnail_runtime_stage(cmd=["ffmpeg", "planned", "thumb.jpg"])

        assert result is True
        assert calls == [["ffmpeg", "planned", "thumb.jpg"]]

    def test_run_extract_thumbnail_runtime_stage_returns_false_and_reports_stderr_on_failure(self, monkeypatch, capsys):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()

        def fake_run(cmd, capture_output, text, check):
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="failed")

        monkeypatch.setattr("subprocess.run", fake_run)

        result = wrapper._run_extract_thumbnail_runtime_stage(cmd=["ffmpeg", "planned", "thumb.jpg"])

        assert result is False
        captured = capsys.readouterr()
        assert "缩略图提取失败: failed" in captured.out

    def test_extract_thumbnail_delegates_runtime_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "input.mov"
        out = tmp_path / "nested" / "thumb.jpg"
        delegated = []

        monkeypatch.setattr(wrapper, "_plan_extract_thumbnail_command", lambda **kwargs: ["ffmpeg", "planned", str(out)], raising=False)
        monkeypatch.setattr(
            wrapper,
            "_run_extract_thumbnail_runtime_stage",
            lambda **kwargs: delegated.append(kwargs) or True,
            raising=False,
        )
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, capture_output, text, check: SimpleNamespace(returncode=0),
        )

        result = wrapper.extract_thumbnail(video, out)

        assert result is True
        assert out.parent.exists()
        assert delegated == [{"cmd": ["ffmpeg", "planned", str(out)]}]


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

    def test_prepare_soft_subtitle_preflight_returns_false_when_video_missing(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        subtitle = tmp_path / "sub.srt"
        subtitle.write_text("dummy", encoding="utf-8")

        result = wrapper._prepare_soft_subtitle_preflight(
            video_path=tmp_path / "missing.mp4",
            subtitle_path=subtitle,
            output_path=tmp_path / "nested" / "out.mkv",
        )

        assert result is False
        captured = capsys.readouterr()
        assert f"错误: 输入视频文件不存在: {tmp_path / 'missing.mp4'}" in captured.out

    def test_prepare_soft_subtitle_preflight_returns_false_when_subtitle_missing(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        video.write_bytes(b"00")

        result = wrapper._prepare_soft_subtitle_preflight(
            video_path=video,
            subtitle_path=tmp_path / "missing.srt",
            output_path=tmp_path / "nested" / "out.mkv",
        )

        assert result is False
        captured = capsys.readouterr()
        assert f"错误: 字幕文件不存在: {tmp_path / 'missing.srt'}" in captured.out

    def test_prepare_soft_subtitle_preflight_creates_output_directory_and_returns_true(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        subtitle = tmp_path / "sub.srt"
        output_path = tmp_path / "nested" / "out.mkv"
        video.write_bytes(b"00")
        subtitle.write_text("dummy", encoding="utf-8")

        result = wrapper._prepare_soft_subtitle_preflight(
            video_path=video,
            subtitle_path=subtitle,
            output_path=output_path,
        )

        assert result is True
        assert output_path.parent.is_dir()

    def test_embed_subtitle_soft_delegates_preflight_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "nested" / "out.mp4"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        delegated = []

        monkeypatch.setattr(
            wrapper,
            "_prepare_soft_subtitle_preflight",
            lambda **kwargs: delegated.append(kwargs) or False,
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_soft_subtitle_command",
            lambda **kwargs: pytest.fail("unexpected command planning"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_run_soft_subtitle_runtime_stage",
            lambda **kwargs: pytest.fail("unexpected runtime stage"),
            raising=False,
        )

        result = wrapper.embed_subtitle_soft(video, sub, out)

        assert result is False
        assert delegated == [{
            "video_path": video,
            "subtitle_path": sub,
            "output_path": out,
        }]

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

    def test_plan_soft_subtitle_command_for_mkv_preserves_ass_and_default_disposition(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mkv"

        cmd = wrapper._plan_soft_subtitle_command(
            video_path=video,
            subtitle_path=sub,
            output_path=out,
            subtitle_language="zho",
            subtitle_title="中文字幕",
        )

        assert cmd == [
            "ffmpeg",
            "-i", str(video),
            "-i", str(sub),
            "-c:v", "copy",
            "-c:a", "copy",
            "-c:s", "copy",
            "-metadata:s:s:0", "language=zho",
            "-metadata:s:s:0", "title=中文字幕",
            "-disposition:s:0", "default",
            "-y",
            str(out),
        ]

    def test_plan_soft_subtitle_command_for_mkv_transcodes_non_ass_subtitle_to_srt(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.srt"
        out = tmp_path / "out.mkv"

        cmd = wrapper._plan_soft_subtitle_command(
            video_path=video,
            subtitle_path=sub,
            output_path=out,
            subtitle_language="chi",
            subtitle_title="中文",
        )

        assert cmd == [
            "ffmpeg",
            "-i", str(video),
            "-i", str(sub),
            "-c:v", "copy",
            "-c:a", "copy",
            "-c:s", "srt",
            "-metadata:s:s:0", "language=chi",
            "-metadata:s:s:0", "title=中文",
            "-disposition:s:0", "default",
            "-y",
            str(out),
        ]

    def test_plan_soft_subtitle_command_for_mp4_uses_mov_text_without_default_disposition(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"

        cmd = wrapper._plan_soft_subtitle_command(
            video_path=video,
            subtitle_path=sub,
            output_path=out,
            subtitle_language="chi",
            subtitle_title="中文",
        )

        assert cmd == [
            "ffmpeg",
            "-i", str(video),
            "-i", str(sub),
            "-c:v", "copy",
            "-c:a", "copy",
            "-c:s", "mov_text",
            "-metadata:s:s:0", "language=chi",
            "-metadata:s:s:0", "title=中文",
            "-y",
            str(out),
        ]

    def test_embed_subtitle_soft_delegates_command_planning_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "nested" / "out.mp4"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        delegated = []
        calls = []

        def fake_plan(**kwargs):
            delegated.append(kwargs)
            return ["ffmpeg", "planned", str(out)]

        def fake_run(cmd, capture_output, text, check):
            calls.append(cmd)
            out.write_bytes(b"11")
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(wrapper, "_plan_soft_subtitle_command", fake_plan, raising=False)
        monkeypatch.setattr("subprocess.run", fake_run)

        result = wrapper.embed_subtitle_soft(video, sub, out)

        assert result is True
        assert delegated == [{
            "video_path": video,
            "subtitle_path": sub,
            "output_path": out,
            "subtitle_language": "chi",
            "subtitle_title": "中文",
        }]
        assert calls == [["ffmpeg", "planned", str(out)]]

    def test_run_soft_subtitle_runtime_stage_returns_true_when_ffmpeg_creates_output(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        sub.write_text("dummy", encoding="utf-8")
        calls = []

        def fake_run(cmd, capture_output, text, check):
            calls.append(cmd)
            out.write_bytes(b"11")
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)

        result = wrapper._run_soft_subtitle_runtime_stage(
            cmd=["ffmpeg", "planned", str(out)],
            subtitle_path=sub,
            output_path=out,
        )

        assert result is True
        assert calls == [["ffmpeg", "planned", str(out)]]

    def test_run_soft_subtitle_runtime_stage_returns_false_when_output_missing(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        sub.write_text("dummy", encoding="utf-8")

        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, capture_output, text, check: SimpleNamespace(returncode=0),
        )

        result = wrapper._run_soft_subtitle_runtime_stage(
            cmd=["ffmpeg", "planned", str(out)],
            subtitle_path=sub,
            output_path=out,
        )

        assert result is False
        captured = capsys.readouterr()
        assert f"错误: 软字幕嵌入完成但未生成文件: {out}" in captured.out

    def test_run_soft_subtitle_runtime_stage_reports_mp4_ass_hint_on_failure(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        sub.write_text("dummy", encoding="utf-8")

        def fake_run(cmd, capture_output, text, check):
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="failed")

        monkeypatch.setattr("subprocess.run", fake_run)

        result = wrapper._run_soft_subtitle_runtime_stage(
            cmd=["ffmpeg", "planned", str(out)],
            subtitle_path=sub,
            output_path=out,
        )

        assert result is False
        captured = capsys.readouterr()
        assert "软字幕嵌入失败: failed" in captured.out
        assert "提示: MP4容器不完全支持ASS字幕样式，建议使用MKV容器或硬字幕" in captured.out

    def test_embed_subtitle_soft_delegates_runtime_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        delegated = []

        monkeypatch.setattr(wrapper, "_plan_soft_subtitle_command", lambda **kwargs: ["ffmpeg", "planned"], raising=False)
        monkeypatch.setattr(
            wrapper,
            "_run_soft_subtitle_runtime_stage",
            lambda **kwargs: delegated.append(kwargs) or True,
            raising=False,
        )

        result = wrapper.embed_subtitle_soft(video, sub, out)

        assert result is True
        assert delegated == [{
            "cmd": ["ffmpeg", "planned"],
            "subtitle_path": sub,
            "output_path": out,
        }]

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


class TestFFmpegWrapperHardEmbedPlanning:
    def test_resolve_hard_embed_gpu_device_parses_explicit_cuda_device(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()

        gpu_id = wrapper._resolve_hard_embed_gpu_device("cuda:7")

        assert gpu_id == 7

    def test_resolve_hard_embed_gpu_device_selects_balanced_gpu_for_auto(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.select_gpu", lambda: 3)

        gpu_id = wrapper._resolve_hard_embed_gpu_device("auto")

        assert gpu_id == 3

    def test_resolve_hard_embed_gpu_device_raises_on_invalid_cuda_format(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()

        with pytest.raises(ValueError, match="无效的 GPU 设备格式"):
            wrapper._resolve_hard_embed_gpu_device("cuda:oops")

    def test_embed_subtitle_hard_delegates_gpu_device_planning_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.srt"
        out = tmp_path / "out.mp4"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        planned = []

        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: None)
        monkeypatch.setattr(wrapper, "_resolve_hard_embed_gpu_device", lambda gpu_device: planned.append(gpu_device) or 5, raising=False)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire", lambda gpu_id, timeout=600: True)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: None)
        monkeypatch.setattr(wrapper, "_check_nvenc_support", lambda: True)
        monkeypatch.setattr(wrapper, "get_video_info", lambda _path: {"bit_rate": 1000})
        monkeypatch.setattr(wrapper, "_build_hard_embed_ffmpeg_command", lambda **kwargs: ["ffmpeg", str(kwargs["gpu_id"])], raising=False)
        monkeypatch.setattr(wrapper, "_run_ffmpeg_embed_process", lambda **kwargs: True, raising=False)

        result = wrapper.embed_subtitle_hard(video, sub, out, gpu_device="cuda:5")

        assert result is True
        assert planned == ["cuda:5"]

    def test_prepare_hard_embed_preflight_returns_false_when_video_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        missing_video = tmp_path / "missing.mp4"
        sub = tmp_path / "sub.srt"
        out = tmp_path / "out.mp4"
        sub.write_text("dummy", encoding="utf-8")

        should_not_run = []
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: should_not_run.append(max_per_gpu))

        result = wrapper._prepare_hard_embed_preflight(
            video_path=missing_video,
            subtitle_path=sub,
            output_path=out,
            gpu_device="auto",
            max_nvenc_sessions=5,
        )

        assert result is False
        assert should_not_run == []

    def test_prepare_hard_embed_preflight_initializes_nvenc_and_prepares_output_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.srt"
        out = tmp_path / "nested" / "out.mp4"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        init_calls = []

        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: init_calls.append(max_per_gpu))

        result = wrapper._prepare_hard_embed_preflight(
            video_path=video,
            subtitle_path=sub,
            output_path=out,
            gpu_device="auto",
            max_nvenc_sessions=7,
        )

        assert result is True
        assert init_calls == [7]
        assert out.parent.exists() is True

    def test_prepare_hard_embed_preflight_raises_on_cpu_gpu_device(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.srt"
        out = tmp_path / "nested" / "out.mp4"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        init_calls = []

        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: init_calls.append(max_per_gpu))

        with pytest.raises(RuntimeError, match="禁止 CPU 回退"):
            wrapper._prepare_hard_embed_preflight(
                video_path=video,
                subtitle_path=sub,
                output_path=out,
                gpu_device="cpu",
                max_nvenc_sessions=5,
            )

        assert init_calls == [5]
        assert out.parent.exists() is False

    def test_embed_subtitle_hard_delegates_preflight_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.srt"
        out = tmp_path / "out.mp4"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        delegated = []

        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.init",
            lambda max_per_gpu=5: pytest.fail("unexpected inline nvenc init"),
        )
        monkeypatch.setattr(
            Path,
            "mkdir",
            lambda self, parents=False, exist_ok=False: pytest.fail("unexpected inline output dir setup"),
        )
        monkeypatch.setattr(
            wrapper,
            "_prepare_hard_embed_preflight",
            lambda **kwargs: delegated.append(kwargs) or True,
            raising=False,
        )
        monkeypatch.setattr(wrapper, "_resolve_hard_embed_gpu_device", lambda gpu_device: 0, raising=False)
        monkeypatch.setattr(wrapper, "_prepare_hard_embed_nvenc_session", lambda **kwargs: None, raising=False)
        monkeypatch.setattr(wrapper, "_probe_hard_embed_original_bitrate", lambda video_path: 1000, raising=False)
        monkeypatch.setattr(wrapper, "_build_hard_embed_ffmpeg_command", lambda **kwargs: ["ffmpeg", "planned"], raising=False)
        monkeypatch.setattr(wrapper, "_run_ffmpeg_embed_process", lambda **kwargs: True, raising=False)
        monkeypatch.setattr(wrapper, "_finalize_hard_embed_resources", lambda **kwargs: None, raising=False)

        result = wrapper.embed_subtitle_hard(
            video,
            sub,
            out,
            gpu_device="auto",
            max_nvenc_sessions=9,
        )

        assert result is True
        assert delegated == [{
            "video_path": video,
            "subtitle_path": sub,
            "output_path": out,
            "gpu_device": "auto",
            "max_nvenc_sessions": 9,
        }]

    def test_plan_hard_embed_subtitle_inputs_uses_original_subtitle_for_non_ass(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        sub = tmp_path / "sub.srt"
        sub.write_text("dummy", encoding="utf-8")
        planned = []

        monkeypatch.setattr(
            wrapper,
            "_build_hard_embed_subtitle_filter",
            lambda **kwargs: planned.append(kwargs) or "subtitles='planned'",
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_prepare_hard_embed_ass_subtitle",
            lambda **kwargs: pytest.fail("unexpected ass preprocess"),
            raising=False,
        )

        temp_files_to_cleanup, vf = wrapper._plan_hard_embed_subtitle_inputs(
            video_path=tmp_path / "video.mp4",
            subtitle_path=sub,
            subtitle_style=None,
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=720,
        )

        assert temp_files_to_cleanup == []
        assert vf == "subtitles='planned'"
        assert planned == [{
            "subtitle_ext": ".srt",
            "processed_subtitle": sub,
            "fonts_dir": "/fonts",
        }]

    def test_plan_hard_embed_subtitle_inputs_delegates_ass_preprocess_before_filter(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        processed_ass = tmp_path / "processed.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        prepared = []
        planned = []

        monkeypatch.setattr(
            wrapper,
            "_prepare_hard_embed_ass_subtitle",
            lambda **kwargs: prepared.append(kwargs) or (processed_ass, [str(tmp_path / "temp.ass")]),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_build_hard_embed_subtitle_filter",
            lambda **kwargs: planned.append(kwargs) or "ass='planned'",
            raising=False,
        )

        temp_files_to_cleanup, vf = wrapper._plan_hard_embed_subtitle_inputs(
            video_path=video,
            subtitle_path=sub,
            subtitle_style="named-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=900,
        )

        assert temp_files_to_cleanup == [str(tmp_path / "temp.ass")]
        assert vf == "ass='planned'"
        assert prepared == [{
            "video_path": video,
            "subtitle_path": sub,
            "subtitle_style": "named-style",
            "style_dir": "/styles",
            "fonts_dir": "/fonts",
            "reference_height": 900,
        }]
        assert planned == [{
            "subtitle_ext": ".ass",
            "processed_subtitle": processed_ass,
            "fonts_dir": "/fonts",
        }]

    def test_embed_subtitle_hard_delegates_subtitle_planning_handoff(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        delegated = []
        execution = []
        runtime = []

        monkeypatch.setattr(wrapper, "_prepare_hard_embed_preflight", lambda **kwargs: True, raising=False)
        monkeypatch.setattr(
            wrapper,
            "_prepare_hard_embed_ass_subtitle",
            lambda **kwargs: pytest.fail("unexpected inline ass preprocess"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_build_hard_embed_subtitle_filter",
            lambda **kwargs: pytest.fail("unexpected inline subtitle filter planning"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_hard_embed_subtitle_inputs",
            lambda **kwargs: delegated.append(kwargs) or ([str(tmp_path / "temp.ass")], "ass='planned'"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_hard_embed_execution",
            lambda **kwargs: execution.append(kwargs) or (4, ["ffmpeg", "planned"]),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_run_hard_embed_runtime_stage",
            lambda **kwargs: runtime.append(kwargs) or True,
            raising=False,
        )

        result = wrapper.embed_subtitle_hard(
            video,
            sub,
            out,
            gpu_device="auto",
            subtitle_style="named-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=900,
        )

        assert result is True
        assert delegated == [{
            "video_path": video,
            "subtitle_path": sub,
            "subtitle_style": "named-style",
            "style_dir": "/styles",
            "fonts_dir": "/fonts",
            "reference_height": 900,
        }]
        assert execution == [{
            "video_path": video,
            "output_path": out,
            "vf": "ass='planned'",
            "video_codec": "hevc",
            "audio_codec": "copy",
            "crf": 28,
            "preset": "p4",
            "gpu_device": "auto",
            "max_nvenc_sessions": 5,
        }]
        assert runtime == [{
            "gpu_id": 4,
            "cmd": ["ffmpeg", "planned"],
            "output_path": out,
            "progress_callback": None,
            "temp_files_to_cleanup": [str(tmp_path / "temp.ass")],
        }]

    def test_plan_hard_embed_execution_resolves_gpu_prepares_session_and_delegates_command_planning(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "out.mp4"
        video.write_bytes(b"00")
        planned = []

        monkeypatch.setattr(
            wrapper,
            "_resolve_hard_embed_gpu_device",
            lambda gpu_device: planned.append(("gpu", gpu_device)) or 4,
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_prepare_hard_embed_nvenc_session",
            lambda **kwargs: planned.append(("session", kwargs)),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_probe_hard_embed_original_bitrate",
            lambda video_path: pytest.fail("unexpected inline bitrate probe"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_build_hard_embed_ffmpeg_command",
            lambda **kwargs: pytest.fail("unexpected inline ffmpeg command build"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_hard_embed_command",
            lambda **kwargs: planned.append(("command", kwargs)) or ["ffmpeg", "planned"],
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_finalize_hard_embed_resources",
            lambda **kwargs: pytest.fail("unexpected planning-stage cleanup on success path"),
            raising=False,
        )

        gpu_id, cmd = wrapper._plan_hard_embed_execution(
            video_path=video,
            output_path=out,
            vf="ass='planned'",
            video_codec="hevc",
            audio_codec="copy",
            crf=28,
            preset="p6",
            gpu_device="cuda:4",
            max_nvenc_sessions=7,
        )

        assert gpu_id == 4
        assert cmd == ["ffmpeg", "planned"]
        assert planned == [
            ("gpu", "cuda:4"),
            ("session", {"gpu_id": 4, "max_nvenc_sessions": 7}),
            ("command", {
                "video_path": video,
                "output_path": out,
                "vf": "ass='planned'",
                "video_codec": "hevc",
                "audio_codec": "copy",
                "crf": 28,
                "preset": "p6",
                "gpu_id": 4,
            }),
        ]

    def test_plan_hard_embed_execution_releases_nvenc_session_when_command_planning_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "out.mp4"
        video.write_bytes(b"00")
        planned = []
        finalized = []

        monkeypatch.setattr(
            wrapper,
            "_resolve_hard_embed_gpu_device",
            lambda gpu_device: planned.append(("gpu", gpu_device)) or 4,
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_prepare_hard_embed_nvenc_session",
            lambda **kwargs: planned.append(("session", kwargs)),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_hard_embed_command",
            lambda **kwargs: planned.append(("command", kwargs)) or (_ for _ in ()).throw(RuntimeError("boom")),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_finalize_hard_embed_resources",
            lambda **kwargs: finalized.append(kwargs),
            raising=False,
        )

        with pytest.raises(RuntimeError, match="boom"):
            wrapper._plan_hard_embed_execution(
                video_path=video,
                output_path=out,
                vf="ass='planned'",
                video_codec="hevc",
                audio_codec="copy",
                crf=28,
                preset="p6",
                gpu_device="cuda:4",
                max_nvenc_sessions=7,
            )

        assert planned == [
            ("gpu", "cuda:4"),
            ("session", {"gpu_id": 4, "max_nvenc_sessions": 7}),
            ("command", {
                "video_path": video,
                "output_path": out,
                "vf": "ass='planned'",
                "video_codec": "hevc",
                "audio_codec": "copy",
                "crf": 28,
                "preset": "p6",
                "gpu_id": 4,
            }),
        ]
        assert finalized == [{
            "gpu_id": 4,
            "temp_files_to_cleanup": [],
        }]

    def test_plan_hard_embed_command_probes_bitrate_before_building_ffmpeg_command(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "out.mp4"
        video.write_bytes(b"00")
        planned = []

        monkeypatch.setattr(
            wrapper,
            "_probe_hard_embed_original_bitrate",
            lambda video_path: planned.append(("bitrate", video_path)) or 4321,
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_build_hard_embed_ffmpeg_command",
            lambda **kwargs: planned.append(("cmd", kwargs)) or ["ffmpeg", "planned"],
            raising=False,
        )

        cmd = wrapper._plan_hard_embed_command(
            video_path=video,
            output_path=out,
            vf="ass='planned'",
            video_codec="hevc",
            audio_codec="copy",
            crf=28,
            preset="p6",
            gpu_id=4,
        )

        assert cmd == ["ffmpeg", "planned"]
        assert planned == [
            ("bitrate", video),
            ("cmd", {
                "video_path": video,
                "output_path": out,
                "vf": "ass='planned'",
                "video_codec": "hevc",
                "audio_codec": "copy",
                "crf": 28,
                "preset": "p6",
                "gpu_id": 4,
                "original_bitrate": 4321,
            }),
        ]

    def test_embed_subtitle_hard_delegates_execution_planning_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        processed_ass = tmp_path / "processed.ass"
        temp_ass = tmp_path / "temp.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        delegated = []
        ran = []
        finalized = []

        monkeypatch.setattr(wrapper, "_prepare_hard_embed_preflight", lambda **kwargs: True, raising=False)
        monkeypatch.setattr(
            wrapper,
            "_plan_hard_embed_subtitle_inputs",
            lambda **kwargs: ([str(temp_ass)], "ass='planned'"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_resolve_hard_embed_gpu_device",
            lambda gpu_device: pytest.fail("unexpected inline gpu resolution"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_prepare_hard_embed_nvenc_session",
            lambda **kwargs: pytest.fail("unexpected inline nvenc session prepare"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_probe_hard_embed_original_bitrate",
            lambda video_path: pytest.fail("unexpected inline bitrate probe"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_build_hard_embed_ffmpeg_command",
            lambda **kwargs: pytest.fail("unexpected inline ffmpeg command build"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_hard_embed_execution",
            lambda **kwargs: delegated.append(kwargs) or (4, ["ffmpeg", "planned"]),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_run_ffmpeg_embed_process",
            lambda **kwargs: ran.append(kwargs) or True,
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_finalize_hard_embed_resources",
            lambda **kwargs: finalized.append(kwargs),
            raising=False,
        )

        result = wrapper.embed_subtitle_hard(
            video,
            sub,
            out,
            video_codec="hevc",
            audio_codec="aac",
            crf=23,
            preset="p6",
            gpu_device="cuda:4",
            subtitle_style="named-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=900,
            max_nvenc_sessions=7,
        )

        assert result is True
        assert delegated == [{
            "video_path": video,
            "output_path": out,
            "vf": "ass='planned'",
            "video_codec": "hevc",
            "audio_codec": "aac",
            "crf": 23,
            "preset": "p6",
            "gpu_device": "cuda:4",
            "max_nvenc_sessions": 7,
        }]
        assert ran == [{
            "cmd": ["ffmpeg", "planned"],
            "output_path": out,
            "progress_callback": None,
        }]
        assert finalized == [{
            "gpu_id": 4,
            "temp_files_to_cleanup": [str(temp_ass)],
        }]

    def test_build_hard_embed_subtitle_filter_uses_ass_and_fontsdir_with_escaped_paths(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()

        vf = wrapper._build_hard_embed_subtitle_filter(
            subtitle_ext=".ass",
            processed_subtitle=Path("/tmp/C:/subs/demo.ass"),
            fonts_dir="/tmp/C:/fonts/demo",
        )

        assert vf == "ass='/tmp/C\\:/subs/demo.ass':fontsdir='/tmp/C\\:/fonts/demo'"

    def test_build_hard_embed_subtitle_filter_uses_subtitles_filter_for_non_ass(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()

        vf = wrapper._build_hard_embed_subtitle_filter(
            subtitle_ext=".srt",
            processed_subtitle=Path("/tmp/C:/subs/demo.srt"),
            fonts_dir="/tmp/C:/fonts/demo",
        )

        assert vf == "subtitles='/tmp/C\\:/subs/demo.srt'"

    def test_embed_subtitle_hard_delegates_subtitle_filter_planning_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        planned = []

        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: None)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.select_gpu", lambda: 0)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire", lambda gpu_id, timeout=600: True)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: None)
        monkeypatch.setattr(wrapper, "_check_nvenc_support", lambda: True)
        monkeypatch.setattr(wrapper, "get_video_info", lambda _path: {"bit_rate": 1000})
        monkeypatch.setattr(
            wrapper,
            "_build_hard_embed_subtitle_filter",
            lambda **kwargs: planned.append(kwargs) or "ass='planned'",
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_build_hard_embed_ffmpeg_command",
            lambda **kwargs: ["ffmpeg", kwargs["vf"]],
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_run_ffmpeg_embed_process",
            lambda **kwargs: True,
            raising=False,
        )

        result = wrapper.embed_subtitle_hard(video, sub, out, gpu_device="auto", fonts_dir="/fonts")

        assert result is True
        assert planned == [{
            "subtitle_ext": ".ass",
            "processed_subtitle": sub,
            "fonts_dir": "/fonts",
        }]

    def test_prepare_hard_embed_ass_subtitle_uses_named_style_wraps_in_place_and_tracks_temp_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        temp_ass = tmp_path / "temp.ass"
        style_calls = []
        scale_calls = []
        from_calls = []
        to_ass_calls = []
        auto_wrap_calls = []

        class _FakeSubtitleData:
            def to_ass(self, **kwargs):
                to_ass_calls.append(kwargs)
                return "ass-content"

        class _FakeASRData:
            @staticmethod
            def from_subtitle_file(path):
                from_calls.append(path)
                return _FakeSubtitleData()

        class _FakeTempFile:
            def __init__(self, path):
                self.name = str(path)
                self._path = path

            def write(self, content):
                self._path.write_text(content, encoding="utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_asr_module = types.ModuleType("vat.asr")
        fake_asr_module.ASRData = _FakeASRData
        fake_subtitle_module = types.ModuleType("vat.asr.subtitle")

        def _fake_get_subtitle_style(name, style_dir=None):
            style_calls.append((name, style_dir))
            return "named-style"

        def _fake_compute_subtitle_scale_factor(width, height, reference_height):
            return 1.25

        def _fake_auto_wrap_ass_file(path, fonts_dir=None):
            auto_wrap_calls.append((path, fonts_dir))
            return path

        fake_subtitle_module.get_subtitle_style = _fake_get_subtitle_style
        fake_subtitle_module.compute_subtitle_scale_factor = _fake_compute_subtitle_scale_factor
        fake_subtitle_module.auto_wrap_ass_file = _fake_auto_wrap_ass_file

        monkeypatch.setitem(sys.modules, "vat.asr", fake_asr_module)
        monkeypatch.setitem(sys.modules, "vat.asr.subtitle", fake_subtitle_module)
        monkeypatch.setattr(wrapper, "_get_video_resolution", lambda path: (1920, 1080))
        monkeypatch.setattr(
            wrapper,
            "_scale_ass_style",
            lambda style_str, scale_factor, video_width, video_height: scale_calls.append(
                (style_str, scale_factor, video_width, video_height)
            ) or "scaled-style",
        )
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", lambda **kwargs: _FakeTempFile(temp_ass))

        processed_subtitle, temp_files_to_cleanup = wrapper._prepare_hard_embed_ass_subtitle(
            video_path=video,
            subtitle_path=sub,
            subtitle_style="named-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=720,
        )

        assert processed_subtitle == temp_ass
        assert temp_files_to_cleanup == [str(temp_ass)]
        assert style_calls == [("named-style", "/styles")]
        assert scale_calls == [("named-style", 1.25, 1920, 1080)]
        assert from_calls == [str(sub)]
        assert to_ass_calls == [{"style_str": "scaled-style", "video_width": 1920, "video_height": 1080}]
        assert auto_wrap_calls == [(str(temp_ass), "/fonts")]
        assert temp_ass.read_text(encoding="utf-8") == "ass-content"

    def test_prepare_hard_embed_ass_subtitle_tracks_extra_temp_file_when_auto_wrap_returns_distinct_output(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        temp_ass = tmp_path / "temp.ass"
        wrapped_ass = tmp_path / "wrapped.ass"
        auto_wrap_calls = []

        class _FakeSubtitleData:
            def to_ass(self, **kwargs):
                return "ass-content"

        class _FakeASRData:
            @staticmethod
            def from_subtitle_file(path):
                return _FakeSubtitleData()

        class _FakeTempFile:
            def __init__(self, path):
                self.name = str(path)
                self._path = path

            def write(self, content):
                self._path.write_text(content, encoding="utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_asr_module = types.ModuleType("vat.asr")
        fake_asr_module.ASRData = _FakeASRData
        fake_subtitle_module = types.ModuleType("vat.asr.subtitle")
        fake_subtitle_module.get_subtitle_style = lambda name, style_dir=None: "named-style"
        fake_subtitle_module.compute_subtitle_scale_factor = lambda width, height, reference_height: 1.0
        fake_subtitle_module.auto_wrap_ass_file = lambda path, fonts_dir=None: auto_wrap_calls.append((path, fonts_dir)) or str(wrapped_ass)

        monkeypatch.setitem(sys.modules, "vat.asr", fake_asr_module)
        monkeypatch.setitem(sys.modules, "vat.asr.subtitle", fake_subtitle_module)
        monkeypatch.setattr(wrapper, "_get_video_resolution", lambda path: (1280, 720))
        monkeypatch.setattr(
            wrapper,
            "_scale_ass_style",
            lambda style_str, scale_factor, video_width, video_height: "scaled-style",
        )
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", lambda **kwargs: _FakeTempFile(temp_ass))

        processed_subtitle, temp_files_to_cleanup = wrapper._prepare_hard_embed_ass_subtitle(
            video_path=video,
            subtitle_path=sub,
            subtitle_style="named-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=720,
        )

        assert processed_subtitle == wrapped_ass
        assert temp_files_to_cleanup == [str(temp_ass), str(wrapped_ass)]
        assert auto_wrap_calls == [(str(temp_ass), "/fonts")]
        assert temp_ass.read_text(encoding="utf-8") == "ass-content"

    def test_prepare_hard_embed_ass_subtitle_falls_back_to_original_when_style_lookup_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        style_calls = []

        class _FakeASRData:
            @staticmethod
            def from_subtitle_file(path):
                pytest.fail("unexpected ass conversion")

        fake_asr_module = types.ModuleType("vat.asr")
        fake_asr_module.ASRData = _FakeASRData
        fake_subtitle_module = types.ModuleType("vat.asr.subtitle")

        def _fake_get_subtitle_style(name, style_dir=None):
            style_calls.append((name, style_dir))
            raise AssertionError(f"样式文件不存在: {style_dir}/{name}.txt")

        fake_subtitle_module.get_subtitle_style = _fake_get_subtitle_style
        fake_subtitle_module.compute_subtitle_scale_factor = (
            lambda width, height, reference_height: pytest.fail("unexpected scale computation")
        )
        fake_subtitle_module.auto_wrap_ass_file = lambda path, fonts_dir=None: pytest.fail("unexpected auto wrap")

        monkeypatch.setitem(sys.modules, "vat.asr", fake_asr_module)
        monkeypatch.setitem(sys.modules, "vat.asr.subtitle", fake_subtitle_module)
        monkeypatch.setattr(wrapper, "_get_video_resolution", lambda path: (1920, 1080))
        monkeypatch.setattr(
            wrapper,
            "_scale_ass_style",
            lambda style_str, scale_factor, video_width, video_height: pytest.fail("unexpected style scaling"),
        )
        monkeypatch.setattr(
            tempfile,
            "NamedTemporaryFile",
            lambda **kwargs: pytest.fail("unexpected temp file creation"),
        )

        processed_subtitle, temp_files_to_cleanup = wrapper._prepare_hard_embed_ass_subtitle(
            video_path=video,
            subtitle_path=sub,
            subtitle_style="missing-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=720,
        )

        assert processed_subtitle == sub
        assert temp_files_to_cleanup == []
        assert style_calls == [("missing-style", "/styles")]

    def test_prepare_hard_embed_ass_subtitle_falls_back_to_original_and_keeps_temp_file_on_wrap_failure(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        temp_ass = tmp_path / "temp.ass"

        class _FakeSubtitleData:
            def to_ass(self, **kwargs):
                return "ass-content"

        class _FakeASRData:
            @staticmethod
            def from_subtitle_file(path):
                return _FakeSubtitleData()

        class _FakeTempFile:
            def __init__(self, path):
                self.name = str(path)
                self._path = path

            def write(self, content):
                self._path.write_text(content, encoding="utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_asr_module = types.ModuleType("vat.asr")
        fake_asr_module.ASRData = _FakeASRData
        fake_subtitle_module = types.ModuleType("vat.asr.subtitle")
        fake_subtitle_module.get_subtitle_style = lambda name, style_dir=None: "named-style"
        fake_subtitle_module.compute_subtitle_scale_factor = lambda width, height, reference_height: 1.0
        fake_subtitle_module.auto_wrap_ass_file = lambda path, fonts_dir=None: (_ for _ in ()).throw(RuntimeError("wrap failed"))

        monkeypatch.setitem(sys.modules, "vat.asr", fake_asr_module)
        monkeypatch.setitem(sys.modules, "vat.asr.subtitle", fake_subtitle_module)
        monkeypatch.setattr(wrapper, "_get_video_resolution", lambda path: (1280, 720))
        monkeypatch.setattr(
            wrapper,
            "_scale_ass_style",
            lambda style_str, scale_factor, video_width, video_height: "scaled-style",
        )
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", lambda **kwargs: _FakeTempFile(temp_ass))

        processed_subtitle, temp_files_to_cleanup = wrapper._prepare_hard_embed_ass_subtitle(
            video_path=video,
            subtitle_path=sub,
            subtitle_style="named-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=720,
        )

        assert processed_subtitle == sub
        assert temp_files_to_cleanup == [str(temp_ass)]
        assert temp_ass.read_text(encoding="utf-8") == "ass-content"

    def test_prepare_hard_embed_ass_subtitle_falls_back_to_original_and_tracks_created_temp_file_when_ass_write_fails(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        temp_ass = tmp_path / "temp.ass"
        auto_wrap_calls = []

        class _FakeSubtitleData:
            def to_ass(self, **kwargs):
                return "ass-content"

        class _FakeASRData:
            @staticmethod
            def from_subtitle_file(path):
                return _FakeSubtitleData()

        class _FakeTempFile:
            def __init__(self, path):
                self.name = str(path)
                self._path = path

            def write(self, content):
                raise RuntimeError("write failed")

            def __enter__(self):
                self._path.write_text("", encoding="utf-8")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_asr_module = types.ModuleType("vat.asr")
        fake_asr_module.ASRData = _FakeASRData
        fake_subtitle_module = types.ModuleType("vat.asr.subtitle")
        fake_subtitle_module.get_subtitle_style = lambda name, style_dir=None: "named-style"
        fake_subtitle_module.compute_subtitle_scale_factor = lambda width, height, reference_height: 1.0
        fake_subtitle_module.auto_wrap_ass_file = lambda path, fonts_dir=None: auto_wrap_calls.append((path, fonts_dir)) or str(tmp_path / "wrapped.ass")

        monkeypatch.setitem(sys.modules, "vat.asr", fake_asr_module)
        monkeypatch.setitem(sys.modules, "vat.asr.subtitle", fake_subtitle_module)
        monkeypatch.setattr(wrapper, "_get_video_resolution", lambda path: (1280, 720))
        monkeypatch.setattr(
            wrapper,
            "_scale_ass_style",
            lambda style_str, scale_factor, video_width, video_height: "scaled-style",
        )
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", lambda **kwargs: _FakeTempFile(temp_ass))

        processed_subtitle, temp_files_to_cleanup = wrapper._prepare_hard_embed_ass_subtitle(
            video_path=video,
            subtitle_path=sub,
            subtitle_style="named-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=720,
        )

        assert processed_subtitle == sub
        assert temp_files_to_cleanup == [str(temp_ass)]
        assert auto_wrap_calls == []
        assert temp_ass.exists() is True

    def test_prepare_hard_embed_ass_subtitle_falls_back_to_original_and_tracks_created_temp_file_when_to_ass_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        temp_ass = tmp_path / "temp.ass"
        auto_wrap_calls = []

        class _FakeSubtitleData:
            def to_ass(self, **kwargs):
                raise RuntimeError("to_ass failed")

        class _FakeASRData:
            @staticmethod
            def from_subtitle_file(path):
                return _FakeSubtitleData()

        class _FakeTempFile:
            def __init__(self, path):
                self.name = str(path)
                self._path = path

            def write(self, content):
                pytest.fail("unexpected temp ass write")

            def __enter__(self):
                self._path.write_text("", encoding="utf-8")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_asr_module = types.ModuleType("vat.asr")
        fake_asr_module.ASRData = _FakeASRData
        fake_subtitle_module = types.ModuleType("vat.asr.subtitle")
        fake_subtitle_module.get_subtitle_style = lambda name, style_dir=None: "named-style"
        fake_subtitle_module.compute_subtitle_scale_factor = lambda width, height, reference_height: 1.0
        fake_subtitle_module.auto_wrap_ass_file = lambda path, fonts_dir=None: auto_wrap_calls.append((path, fonts_dir)) or str(tmp_path / "wrapped.ass")

        monkeypatch.setitem(sys.modules, "vat.asr", fake_asr_module)
        monkeypatch.setitem(sys.modules, "vat.asr.subtitle", fake_subtitle_module)
        monkeypatch.setattr(wrapper, "_get_video_resolution", lambda path: (1280, 720))
        monkeypatch.setattr(
            wrapper,
            "_scale_ass_style",
            lambda style_str, scale_factor, video_width, video_height: "scaled-style",
        )
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", lambda **kwargs: _FakeTempFile(temp_ass))

        processed_subtitle, temp_files_to_cleanup = wrapper._prepare_hard_embed_ass_subtitle(
            video_path=video,
            subtitle_path=sub,
            subtitle_style="named-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=720,
        )

        assert processed_subtitle == sub
        assert temp_files_to_cleanup == [str(temp_ass)]
        assert auto_wrap_calls == []
        assert temp_ass.exists() is True

    def test_prepare_hard_embed_ass_subtitle_falls_back_to_original_without_temp_files_when_asr_data_load_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        style_calls = []
        scale_calls = []
        auto_wrap_calls = []

        class _FakeASRData:
            @staticmethod
            def from_subtitle_file(path):
                raise RuntimeError("asr data load failed")

        fake_asr_module = types.ModuleType("vat.asr")
        fake_asr_module.ASRData = _FakeASRData
        fake_subtitle_module = types.ModuleType("vat.asr.subtitle")

        def _fake_get_subtitle_style(name, style_dir=None):
            style_calls.append((name, style_dir))
            return "named-style"

        fake_subtitle_module.get_subtitle_style = _fake_get_subtitle_style
        fake_subtitle_module.compute_subtitle_scale_factor = lambda width, height, reference_height: 1.0
        fake_subtitle_module.auto_wrap_ass_file = lambda path, fonts_dir=None: auto_wrap_calls.append((path, fonts_dir)) or pytest.fail("unexpected auto wrap")

        monkeypatch.setitem(sys.modules, "vat.asr", fake_asr_module)
        monkeypatch.setitem(sys.modules, "vat.asr.subtitle", fake_subtitle_module)
        monkeypatch.setattr(wrapper, "_get_video_resolution", lambda path: (1280, 720))
        monkeypatch.setattr(
            wrapper,
            "_scale_ass_style",
            lambda style_str, scale_factor, video_width, video_height: scale_calls.append(
                (style_str, scale_factor, video_width, video_height)
            ) or "scaled-style",
        )
        monkeypatch.setattr(
            tempfile,
            "NamedTemporaryFile",
            lambda **kwargs: pytest.fail("unexpected temp file creation"),
        )

        processed_subtitle, temp_files_to_cleanup = wrapper._prepare_hard_embed_ass_subtitle(
            video_path=video,
            subtitle_path=sub,
            subtitle_style="named-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=720,
        )

        assert processed_subtitle == sub
        assert temp_files_to_cleanup == []
        assert style_calls == [("named-style", "/styles")]
        assert scale_calls == [("named-style", 1.0, 1280, 720)]
        assert auto_wrap_calls == []

    def test_prepare_hard_embed_ass_subtitle_falls_back_to_original_without_temp_files_when_scale_ass_style_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        style_calls = []
        scale_calls = []
        from_calls = []
        auto_wrap_calls = []

        class _FakeASRData:
            @staticmethod
            def from_subtitle_file(path):
                from_calls.append(path)
                pytest.fail("unexpected ASR data load")

        fake_asr_module = types.ModuleType("vat.asr")
        fake_asr_module.ASRData = _FakeASRData
        fake_subtitle_module = types.ModuleType("vat.asr.subtitle")

        def _fake_get_subtitle_style(name, style_dir=None):
            style_calls.append((name, style_dir))
            return "named-style"

        def _fake_compute_subtitle_scale_factor(width, height, reference_height):
            return 1.0

        fake_subtitle_module.get_subtitle_style = _fake_get_subtitle_style
        fake_subtitle_module.compute_subtitle_scale_factor = _fake_compute_subtitle_scale_factor
        fake_subtitle_module.auto_wrap_ass_file = lambda path, fonts_dir=None: auto_wrap_calls.append((path, fonts_dir)) or pytest.fail("unexpected auto wrap")

        monkeypatch.setitem(sys.modules, "vat.asr", fake_asr_module)
        monkeypatch.setitem(sys.modules, "vat.asr.subtitle", fake_subtitle_module)
        monkeypatch.setattr(wrapper, "_get_video_resolution", lambda path: (1280, 720))
        monkeypatch.setattr(
            wrapper,
            "_scale_ass_style",
            lambda style_str, scale_factor, video_width, video_height: scale_calls.append(
                (style_str, scale_factor, video_width, video_height)
            ) or (_ for _ in ()).throw(RuntimeError("scale style failed")),
        )
        monkeypatch.setattr(
            tempfile,
            "NamedTemporaryFile",
            lambda **kwargs: pytest.fail("unexpected temp file creation"),
        )

        processed_subtitle, temp_files_to_cleanup = wrapper._prepare_hard_embed_ass_subtitle(
            video_path=video,
            subtitle_path=sub,
            subtitle_style="named-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=720,
        )

        assert processed_subtitle == sub
        assert temp_files_to_cleanup == []
        assert style_calls == [("named-style", "/styles")]
        assert scale_calls == [("named-style", 1.0, 1280, 720)]
        assert from_calls == []
        assert auto_wrap_calls == []

    def test_prepare_hard_embed_ass_subtitle_falls_back_to_original_without_temp_files_when_compute_scale_factor_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        style_calls = []
        scale_calls = []
        from_calls = []
        auto_wrap_calls = []

        class _FakeASRData:
            @staticmethod
            def from_subtitle_file(path):
                from_calls.append(path)
                pytest.fail("unexpected ASR data load")

        fake_asr_module = types.ModuleType("vat.asr")
        fake_asr_module.ASRData = _FakeASRData
        fake_subtitle_module = types.ModuleType("vat.asr.subtitle")

        def _fake_get_subtitle_style(name, style_dir=None):
            style_calls.append((name, style_dir))
            return "named-style"

        def _fake_compute_subtitle_scale_factor(width, height, reference_height):
            scale_calls.append((width, height, reference_height))
            raise RuntimeError("compute scale failed")

        fake_subtitle_module.get_subtitle_style = _fake_get_subtitle_style
        fake_subtitle_module.compute_subtitle_scale_factor = _fake_compute_subtitle_scale_factor
        fake_subtitle_module.auto_wrap_ass_file = lambda path, fonts_dir=None: auto_wrap_calls.append((path, fonts_dir)) or pytest.fail("unexpected auto wrap")

        monkeypatch.setitem(sys.modules, "vat.asr", fake_asr_module)
        monkeypatch.setitem(sys.modules, "vat.asr.subtitle", fake_subtitle_module)
        monkeypatch.setattr(wrapper, "_get_video_resolution", lambda path: (1280, 720))
        monkeypatch.setattr(
            wrapper,
            "_scale_ass_style",
            lambda style_str, scale_factor, video_width, video_height: pytest.fail("unexpected style scaling"),
        )
        monkeypatch.setattr(
            tempfile,
            "NamedTemporaryFile",
            lambda **kwargs: pytest.fail("unexpected temp file creation"),
        )

        processed_subtitle, temp_files_to_cleanup = wrapper._prepare_hard_embed_ass_subtitle(
            video_path=video,
            subtitle_path=sub,
            subtitle_style="named-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=720,
        )

        assert processed_subtitle == sub
        assert temp_files_to_cleanup == []
        assert style_calls == [("named-style", "/styles")]
        assert scale_calls == [(1280, 720, 720)]
        assert from_calls == []
        assert auto_wrap_calls == []

    def test_prepare_hard_embed_ass_subtitle_falls_back_to_original_without_temp_files_when_temp_file_creation_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        style_calls = []
        scale_calls = []
        from_calls = []
        to_ass_calls = []
        auto_wrap_calls = []

        class _FakeSubtitleData:
            def to_ass(self, **kwargs):
                to_ass_calls.append(kwargs)
                pytest.fail("unexpected to_ass call")

        class _FakeASRData:
            @staticmethod
            def from_subtitle_file(path):
                from_calls.append(path)
                return _FakeSubtitleData()

        fake_asr_module = types.ModuleType("vat.asr")
        fake_asr_module.ASRData = _FakeASRData
        fake_subtitle_module = types.ModuleType("vat.asr.subtitle")

        def _fake_get_subtitle_style(name, style_dir=None):
            style_calls.append((name, style_dir))
            return "named-style"

        def _fake_compute_subtitle_scale_factor(width, height, reference_height):
            scale_calls.append((width, height, reference_height))
            return 1.0

        fake_subtitle_module.get_subtitle_style = _fake_get_subtitle_style
        fake_subtitle_module.compute_subtitle_scale_factor = _fake_compute_subtitle_scale_factor
        fake_subtitle_module.auto_wrap_ass_file = lambda path, fonts_dir=None: auto_wrap_calls.append((path, fonts_dir)) or pytest.fail("unexpected auto wrap")

        monkeypatch.setitem(sys.modules, "vat.asr", fake_asr_module)
        monkeypatch.setitem(sys.modules, "vat.asr.subtitle", fake_subtitle_module)
        monkeypatch.setattr(wrapper, "_get_video_resolution", lambda path: (1280, 720))
        monkeypatch.setattr(
            wrapper,
            "_scale_ass_style",
            lambda style_str, scale_factor, video_width, video_height: "scaled-style",
        )
        monkeypatch.setattr(
            tempfile,
            "NamedTemporaryFile",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("temp file create failed")),
        )

        processed_subtitle, temp_files_to_cleanup = wrapper._prepare_hard_embed_ass_subtitle(
            video_path=video,
            subtitle_path=sub,
            subtitle_style="named-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=720,
        )

        assert processed_subtitle == sub
        assert temp_files_to_cleanup == []
        assert style_calls == [("named-style", "/styles")]
        assert scale_calls == [(1280, 720, 720)]
        assert from_calls == [str(sub)]
        assert to_ass_calls == []
        assert auto_wrap_calls == []

    def test_embed_subtitle_hard_delegates_ass_preprocess_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        processed_ass = tmp_path / "processed.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        prepared = []
        planned = []

        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: None)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: None)
        monkeypatch.setattr(wrapper, "_resolve_hard_embed_gpu_device", lambda gpu_device: 0, raising=False)
        monkeypatch.setattr(wrapper, "_prepare_hard_embed_nvenc_session", lambda **kwargs: None, raising=False)
        monkeypatch.setattr(wrapper, "_get_video_resolution", lambda path: pytest.fail("unexpected inline ass preprocess"))
        monkeypatch.setattr(
            wrapper,
            "_prepare_hard_embed_ass_subtitle",
            lambda **kwargs: prepared.append(kwargs) or (processed_ass, [str(tmp_path / "temp.ass")]),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_build_hard_embed_subtitle_filter",
            lambda **kwargs: planned.append(kwargs) or "ass='planned'",
            raising=False,
        )
        monkeypatch.setattr(wrapper, "_probe_hard_embed_original_bitrate", lambda video_path: 1000, raising=False)
        monkeypatch.setattr(wrapper, "_build_hard_embed_ffmpeg_command", lambda **kwargs: ["ffmpeg", "planned"], raising=False)
        monkeypatch.setattr(wrapper, "_run_ffmpeg_embed_process", lambda **kwargs: True, raising=False)

        result = wrapper.embed_subtitle_hard(
            video,
            sub,
            out,
            gpu_device="auto",
            subtitle_style="named-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=900,
        )

        assert result is True
        assert prepared == [{
            "video_path": video,
            "subtitle_path": sub,
            "subtitle_style": "named-style",
            "style_dir": "/styles",
            "fonts_dir": "/fonts",
            "reference_height": 900,
        }]
        assert planned and planned[0]["processed_subtitle"] == processed_ass

    def test_embed_subtitle_hard_cleans_up_ass_temp_files_from_prepare_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        temp_ass = tmp_path / "temp.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        temp_ass.write_text("temp", encoding="utf-8")
        released = []

        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: None)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: released.append(gpu_id))
        monkeypatch.setattr(wrapper, "_resolve_hard_embed_gpu_device", lambda gpu_device: 0, raising=False)
        monkeypatch.setattr(wrapper, "_prepare_hard_embed_nvenc_session", lambda **kwargs: None, raising=False)
        monkeypatch.setattr(
            wrapper,
            "_prepare_hard_embed_ass_subtitle",
            lambda **kwargs: (temp_ass, [str(temp_ass)]),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_build_hard_embed_subtitle_filter",
            lambda **kwargs: "ass='planned'" if kwargs["processed_subtitle"] == temp_ass else pytest.fail("unexpected processed subtitle"),
            raising=False,
        )
        monkeypatch.setattr(wrapper, "_probe_hard_embed_original_bitrate", lambda video_path: 1000, raising=False)
        monkeypatch.setattr(wrapper, "_build_hard_embed_ffmpeg_command", lambda **kwargs: ["ffmpeg", "planned"], raising=False)
        monkeypatch.setattr(wrapper, "_run_ffmpeg_embed_process", lambda **kwargs: True, raising=False)

        result = wrapper.embed_subtitle_hard(
            video,
            sub,
            out,
            gpu_device="auto",
            subtitle_style="named-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=900,
        )

        assert result is True
        assert released == [0]
        assert temp_ass.exists() is False

    def test_embed_subtitle_hard_cleans_up_multiple_ass_temp_files_when_prepare_returns_multiple_cleanup_paths(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        processed_ass = tmp_path / "processed.ass"
        temp_ass = tmp_path / "temp.ass"
        wrapped_ass = tmp_path / "wrapped.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        processed_ass.write_text("processed", encoding="utf-8")
        temp_ass.write_text("temp", encoding="utf-8")
        wrapped_ass.write_text("wrapped", encoding="utf-8")
        released = []

        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: None)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: released.append(gpu_id))
        monkeypatch.setattr(wrapper, "_resolve_hard_embed_gpu_device", lambda gpu_device: 0, raising=False)
        monkeypatch.setattr(wrapper, "_prepare_hard_embed_nvenc_session", lambda **kwargs: None, raising=False)
        monkeypatch.setattr(
            wrapper,
            "_prepare_hard_embed_ass_subtitle",
            lambda **kwargs: (processed_ass, [str(temp_ass), str(wrapped_ass)]),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_build_hard_embed_subtitle_filter",
            lambda **kwargs: "ass='planned'" if kwargs["processed_subtitle"] == processed_ass else pytest.fail("unexpected processed subtitle"),
            raising=False,
        )
        monkeypatch.setattr(wrapper, "_probe_hard_embed_original_bitrate", lambda video_path: 1000, raising=False)
        monkeypatch.setattr(wrapper, "_build_hard_embed_ffmpeg_command", lambda **kwargs: ["ffmpeg", "planned"], raising=False)
        monkeypatch.setattr(wrapper, "_run_ffmpeg_embed_process", lambda **kwargs: True, raising=False)

        result = wrapper.embed_subtitle_hard(
            video,
            sub,
            out,
            gpu_device="auto",
            subtitle_style="named-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=900,
        )

        assert result is True
        assert released == [0]
        assert temp_ass.exists() is False
        assert wrapped_ass.exists() is False
        assert processed_ass.exists() is True

    def test_finalize_hard_embed_resources_releases_nvenc_session_and_cleans_temp_files(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        temp_ass = tmp_path / "temp.ass"
        wrapped_ass = tmp_path / "wrapped.ass"
        temp_ass.write_text("temp", encoding="utf-8")
        wrapped_ass.write_text("wrapped", encoding="utf-8")
        released = []

        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.release",
            lambda gpu_id: released.append(gpu_id),
        )

        wrapper._finalize_hard_embed_resources(
            gpu_id=3,
            temp_files_to_cleanup=[str(temp_ass), str(wrapped_ass)],
        )

        assert released == [3]
        assert temp_ass.exists() is False
        assert wrapped_ass.exists() is False

    def test_finalize_hard_embed_resources_ignores_temp_file_cleanup_errors(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        temp_ass = tmp_path / "temp.ass"
        released = []
        attempted = []

        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.release",
            lambda gpu_id: released.append(gpu_id),
        )
        monkeypatch.setattr(
            Path,
            "unlink",
            lambda self, missing_ok=True: attempted.append((str(self), missing_ok)) or (_ for _ in ()).throw(RuntimeError("boom")),
        )

        wrapper._finalize_hard_embed_resources(
            gpu_id=4,
            temp_files_to_cleanup=[str(temp_ass)],
        )

        assert released == [4]
        assert attempted == [(str(temp_ass), True)]

    def test_embed_subtitle_hard_delegates_finalize_cleanup_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        processed_ass = tmp_path / "processed.ass"
        temp_ass = tmp_path / "temp.ass"
        wrapped_ass = tmp_path / "wrapped.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        finalized = []

        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: None)
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.release",
            lambda gpu_id: pytest.fail("unexpected inline release"),
        )
        monkeypatch.setattr(
            Path,
            "unlink",
            lambda self, missing_ok=True: pytest.fail("unexpected inline cleanup"),
        )
        monkeypatch.setattr(wrapper, "_resolve_hard_embed_gpu_device", lambda gpu_device: 0, raising=False)
        monkeypatch.setattr(wrapper, "_prepare_hard_embed_nvenc_session", lambda **kwargs: None, raising=False)
        monkeypatch.setattr(
            wrapper,
            "_prepare_hard_embed_ass_subtitle",
            lambda **kwargs: (processed_ass, [str(temp_ass), str(wrapped_ass)]),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_build_hard_embed_subtitle_filter",
            lambda **kwargs: "ass='planned'",
            raising=False,
        )
        monkeypatch.setattr(wrapper, "_probe_hard_embed_original_bitrate", lambda video_path: 1000, raising=False)
        monkeypatch.setattr(wrapper, "_build_hard_embed_ffmpeg_command", lambda **kwargs: ["ffmpeg", "planned"], raising=False)
        monkeypatch.setattr(wrapper, "_run_ffmpeg_embed_process", lambda **kwargs: True, raising=False)
        monkeypatch.setattr(
            wrapper,
            "_finalize_hard_embed_resources",
            lambda **kwargs: finalized.append(kwargs),
            raising=False,
        )

        result = wrapper.embed_subtitle_hard(
            video,
            sub,
            out,
            gpu_device="auto",
            subtitle_style="named-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=900,
        )

        assert result is True
        assert finalized == [{
            "gpu_id": 0,
            "temp_files_to_cleanup": [str(temp_ass), str(wrapped_ass)],
        }]

    def test_prepare_hard_embed_nvenc_session_acquires_slot_and_checks_support(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        acquired = []
        released = []
        checked = []

        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire",
            lambda gpu_id, timeout=600: acquired.append((gpu_id, timeout)) or True,
        )
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.release",
            lambda gpu_id: released.append(gpu_id),
        )
        monkeypatch.setattr(wrapper, "_check_nvenc_support", lambda: checked.append(True) or True)

        wrapper._prepare_hard_embed_nvenc_session(gpu_id=4, max_nvenc_sessions=5)

        assert acquired == [(4, 600)]
        assert checked == [True]
        assert released == []

    def test_prepare_hard_embed_nvenc_session_raises_on_acquire_timeout(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        checked = []
        released = []

        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire",
            lambda gpu_id, timeout=600: False,
        )
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.release",
            lambda gpu_id: released.append(gpu_id),
        )
        monkeypatch.setattr(wrapper, "_check_nvenc_support", lambda: checked.append(True) or True)

        with pytest.raises(RuntimeError, match="NVENC 会话获取超时: GPU 2"):
            wrapper._prepare_hard_embed_nvenc_session(gpu_id=2, max_nvenc_sessions=5)

        assert checked == []
        assert released == []

    def test_prepare_hard_embed_nvenc_session_releases_slot_when_nvenc_unsupported(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        released = []

        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire",
            lambda gpu_id, timeout=600: True,
        )
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.release",
            lambda gpu_id: released.append(gpu_id),
        )
        monkeypatch.setattr(wrapper, "_check_nvenc_support", lambda: False)

        with pytest.raises(RuntimeError, match="当前环境不支持 NVENC"):
            wrapper._prepare_hard_embed_nvenc_session(gpu_id=1, max_nvenc_sessions=5)

        assert released == [1]

    def test_embed_subtitle_hard_delegates_nvenc_session_prepare_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.srt"
        out = tmp_path / "out.mp4"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        delegated = []

        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: None)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.select_gpu", lambda: 0)
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire",
            lambda gpu_id, timeout=600: pytest.fail("unexpected inline acquire call"),
        )
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: None)
        monkeypatch.setattr(
            wrapper,
            "_check_nvenc_support",
            lambda: pytest.fail("unexpected inline nvenc support check"),
        )
        monkeypatch.setattr(
            wrapper,
            "_prepare_hard_embed_nvenc_session",
            lambda **kwargs: delegated.append(kwargs),
            raising=False,
        )
        monkeypatch.setattr(wrapper, "_probe_hard_embed_original_bitrate", lambda video_path: 1000, raising=False)
        monkeypatch.setattr(wrapper, "_build_hard_embed_ffmpeg_command", lambda **kwargs: ["ffmpeg", "planned"], raising=False)
        monkeypatch.setattr(wrapper, "_run_ffmpeg_embed_process", lambda **kwargs: True, raising=False)

        result = wrapper.embed_subtitle_hard(video, sub, out, gpu_device="auto")

        assert result is True
        assert delegated == [{"gpu_id": 0, "max_nvenc_sessions": 5}]

    def test_probe_hard_embed_original_bitrate_returns_bit_rate_from_video_info(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        video.write_bytes(b"00")
        monkeypatch.setattr(wrapper, "get_video_info", lambda path: {"bit_rate": 2468})

        bitrate = wrapper._probe_hard_embed_original_bitrate(video)

        assert bitrate == 2468

    def test_probe_hard_embed_original_bitrate_falls_back_to_zero_when_probe_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        video.write_bytes(b"00")
        monkeypatch.setattr(wrapper, "get_video_info", lambda path: None)

        bitrate = wrapper._probe_hard_embed_original_bitrate(video)

        assert bitrate == 0

    def test_embed_subtitle_hard_delegates_bitrate_probe_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.srt"
        out = tmp_path / "out.mp4"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        delegated = []
        planned = []

        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: None)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.select_gpu", lambda: 0)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire", lambda gpu_id, timeout=600: True)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: None)
        monkeypatch.setattr(wrapper, "_check_nvenc_support", lambda: True)
        monkeypatch.setattr(wrapper, "get_video_info", lambda _path: {"bit_rate": 1000})
        monkeypatch.setattr(
            wrapper,
            "_probe_hard_embed_original_bitrate",
            lambda video_path: delegated.append(video_path) or 4321,
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_build_hard_embed_ffmpeg_command",
            lambda **kwargs: planned.append(kwargs) or ["ffmpeg", "planned"],
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_run_ffmpeg_embed_process",
            lambda **kwargs: True,
            raising=False,
        )

        result = wrapper.embed_subtitle_hard(video, sub, out, gpu_device="auto")

        assert result is True
        assert delegated == [video]
        assert planned and planned[0]["original_bitrate"] == 4321

    def test_build_hard_embed_ffmpeg_command_uses_hevc_nvenc_and_vbr_bitrate(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "out.mp4"

        cmd = wrapper._build_hard_embed_ffmpeg_command(
            video_path=video,
            output_path=out,
            vf="subtitles='sub.srt'",
            video_codec="hevc",
            audio_codec="copy",
            crf=28,
            preset="p6",
            gpu_id=3,
            original_bitrate=1000,
        )

        assert cmd[:8] == [
            "ffmpeg",
            "-hwaccel",
            "cuda",
            "-hwaccel_device",
            "3",
            "-i",
            str(video),
            "-vf",
        ]
        assert "subtitles='sub.srt'" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "hevc_nvenc"
        assert cmd[cmd.index("-rc") + 1] == "vbr"
        assert cmd[cmd.index("-cq") + 1] == "28"
        assert cmd[cmd.index("-b:v") + 1] == "1100"
        assert cmd[cmd.index("-maxrate") + 1] == "1500"
        assert cmd[cmd.index("-bufsize") + 1] == "3000"
        assert cmd[cmd.index("-preset") + 1] == "p6"
        assert cmd[cmd.index("-gpu") + 1] == "3"
        assert cmd[cmd.index("-c:a") + 1] == "copy"
        assert cmd[-2:] == ["-y", str(out)]

    def test_build_hard_embed_ffmpeg_command_falls_back_to_hevc_when_av1_nvenc_unavailable(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "out.mp4"
        monkeypatch.setattr(wrapper, "_check_encoder_support", lambda name: False)

        cmd = wrapper._build_hard_embed_ffmpeg_command(
            video_path=video,
            output_path=out,
            vf="subtitles='sub.srt'",
            video_codec="av1",
            audio_codec="aac",
            crf=23,
            preset="slow",
            gpu_id=0,
            original_bitrate=0,
        )

        assert cmd[cmd.index("-c:v") + 1] == "hevc_nvenc"
        assert cmd[cmd.index("-rc") + 1] == "constqp"
        assert cmd[cmd.index("-qp") + 1] == "23"
        assert cmd[cmd.index("-preset") + 1] == "p4"
        assert cmd[cmd.index("-gpu") + 1] == "0"
        assert cmd[cmd.index("-c:a") + 1] == "aac"

    def test_embed_subtitle_hard_delegates_command_planning_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.srt"
        out = tmp_path / "out.mp4"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        planned = []
        delegated = []

        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: None)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.select_gpu", lambda: 0)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire", lambda gpu_id, timeout=600: True)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: None)
        monkeypatch.setattr(wrapper, "_check_nvenc_support", lambda: True)
        monkeypatch.setattr(wrapper, "get_video_info", lambda _path: {"bit_rate": 1000})
        monkeypatch.setattr(
            wrapper,
            "_build_hard_embed_ffmpeg_command",
            lambda **kwargs: planned.append(kwargs) or ["ffmpeg", "planned"],
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_run_ffmpeg_embed_process",
            lambda **kwargs: delegated.append(kwargs) or True,
            raising=False,
        )

        result = wrapper.embed_subtitle_hard(video, sub, out, gpu_device="auto")

        assert result is True
        assert planned and planned[0]["original_bitrate"] == 1000
        assert planned[0]["gpu_id"] == 0
        assert delegated == [{"cmd": ["ffmpeg", "planned"], "output_path": out, "progress_callback": None}]


class TestFFmpegWrapperHardEmbedRuntime:
    def test_run_hard_embed_runtime_stage_runs_ffmpeg_and_finalizes_resources(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        out = tmp_path / "out.mp4"
        temp_ass = tmp_path / "temp.ass"
        calls = []

        monkeypatch.setattr(
            wrapper,
            "_run_ffmpeg_embed_process",
            lambda **kwargs: calls.append(("run", kwargs)) or True,
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_finalize_hard_embed_resources",
            lambda **kwargs: calls.append(("finalize", kwargs)),
            raising=False,
        )

        result = wrapper._run_hard_embed_runtime_stage(
            gpu_id=4,
            cmd=["ffmpeg", "planned"],
            output_path=out,
            progress_callback=None,
            temp_files_to_cleanup=[str(temp_ass)],
        )

        assert result is True
        assert calls == [
            ("run", {
                "cmd": ["ffmpeg", "planned"],
                "output_path": out,
                "progress_callback": None,
            }),
            ("finalize", {
                "gpu_id": 4,
                "temp_files_to_cleanup": [str(temp_ass)],
            }),
        ]

    def test_run_hard_embed_runtime_stage_finalizes_resources_when_ffmpeg_process_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        out = tmp_path / "out.mp4"
        temp_ass = tmp_path / "temp.ass"
        finalized = []

        monkeypatch.setattr(
            wrapper,
            "_run_ffmpeg_embed_process",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_finalize_hard_embed_resources",
            lambda **kwargs: finalized.append(kwargs),
            raising=False,
        )

        with pytest.raises(RuntimeError, match="boom"):
            wrapper._run_hard_embed_runtime_stage(
                gpu_id=2,
                cmd=["ffmpeg", "planned"],
                output_path=out,
                progress_callback=None,
                temp_files_to_cleanup=[str(temp_ass)],
            )

        assert finalized == [{
            "gpu_id": 2,
            "temp_files_to_cleanup": [str(temp_ass)],
        }]

    def test_embed_subtitle_hard_delegates_runtime_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        processed_ass = tmp_path / "processed.ass"
        temp_ass = tmp_path / "temp.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        delegated = []

        monkeypatch.setattr(wrapper, "_prepare_hard_embed_preflight", lambda **kwargs: True, raising=False)
        monkeypatch.setattr(
            wrapper,
            "_plan_hard_embed_subtitle_inputs",
            lambda **kwargs: ([str(temp_ass)], "ass='planned'"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_hard_embed_execution",
            lambda **kwargs: (4, ["ffmpeg", "planned"]),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_run_ffmpeg_embed_process",
            lambda **kwargs: pytest.fail("unexpected inline ffmpeg execution"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_finalize_hard_embed_resources",
            lambda **kwargs: pytest.fail("unexpected inline finalize cleanup"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_run_hard_embed_runtime_stage",
            lambda **kwargs: delegated.append(kwargs) or True,
            raising=False,
        )

        result = wrapper.embed_subtitle_hard(
            video,
            sub,
            out,
            gpu_device="cuda:4",
            subtitle_style="named-style",
            style_dir="/styles",
            fonts_dir="/fonts",
            reference_height=900,
        )

        assert result is True
        assert delegated == [{
            "gpu_id": 4,
            "cmd": ["ffmpeg", "planned"],
            "output_path": out,
            "progress_callback": None,
            "temp_files_to_cleanup": [str(temp_ass)],
        }]

    def test_embed_subtitle_hard_cleans_temp_files_when_execution_planning_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        temp_ass = tmp_path / "temp.ass"
        wrapped_ass = tmp_path / "wrapped.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        temp_ass.write_text("temp", encoding="utf-8")
        wrapped_ass.write_text("wrapped", encoding="utf-8")

        monkeypatch.setattr(wrapper, "_prepare_hard_embed_preflight", lambda **kwargs: True, raising=False)
        monkeypatch.setattr(
            wrapper,
            "_plan_hard_embed_subtitle_inputs",
            lambda **kwargs: ([str(temp_ass), str(wrapped_ass)], "ass='planned'"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_hard_embed_execution",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("planning boom")),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_run_hard_embed_runtime_stage",
            lambda **kwargs: pytest.fail("unexpected runtime stage"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_finalize_hard_embed_resources",
            lambda **kwargs: pytest.fail("unexpected gpu finalize for top-level temp cleanup"),
            raising=False,
        )

        with pytest.raises(RuntimeError, match="planning boom"):
            wrapper.embed_subtitle_hard(
                video,
                sub,
                out,
                gpu_device="cuda:4",
                subtitle_style="named-style",
                style_dir="/styles",
                fonts_dir="/fonts",
                reference_height=900,
            )

        assert temp_ass.exists() is False
        assert wrapped_ass.exists() is False

    def test_embed_subtitle_hard_re_raises_command_planning_error_releases_nvenc_once_and_cleans_temp_files_once(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        temp_ass = tmp_path / "temp.ass"
        wrapped_ass = tmp_path / "wrapped.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        temp_ass.write_text("temp", encoding="utf-8")
        wrapped_ass.write_text("wrapped", encoding="utf-8")
        session = []
        released = []
        unlink_calls = []
        original_unlink = Path.unlink

        monkeypatch.setattr(wrapper, "_prepare_hard_embed_preflight", lambda **kwargs: True, raising=False)
        monkeypatch.setattr(
            wrapper,
            "_plan_hard_embed_subtitle_inputs",
            lambda **kwargs: ([str(temp_ass), str(wrapped_ass)], "ass='planned'"),
            raising=False,
        )
        monkeypatch.setattr(wrapper, "_resolve_hard_embed_gpu_device", lambda gpu_device: 4, raising=False)
        monkeypatch.setattr(
            wrapper,
            "_prepare_hard_embed_nvenc_session",
            lambda **kwargs: session.append(kwargs),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_hard_embed_command",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("command boom")),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_run_hard_embed_runtime_stage",
            lambda **kwargs: pytest.fail("unexpected runtime stage"),
            raising=False,
        )
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.release",
            lambda gpu_id: released.append(gpu_id),
        )
        monkeypatch.setattr(
            Path,
            "unlink",
            lambda self, missing_ok=True: unlink_calls.append((str(self), missing_ok)) or original_unlink(self, missing_ok=missing_ok),
        )

        with pytest.raises(RuntimeError, match="command boom"):
            wrapper.embed_subtitle_hard(
                video,
                sub,
                out,
                gpu_device="cuda:4",
                subtitle_style="named-style",
                style_dir="/styles",
                fonts_dir="/fonts",
                reference_height=900,
                max_nvenc_sessions=7,
            )

        assert session == [{"gpu_id": 4, "max_nvenc_sessions": 7}]
        assert released == [4]
        assert unlink_calls == [
            (str(temp_ass), True),
            (str(wrapped_ass), True),
        ]
        assert temp_ass.exists() is False
        assert wrapped_ass.exists() is False

    def test_embed_subtitle_hard_cleans_temp_files_when_nvenc_not_supported_after_ass_prepare(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        temp_ass = tmp_path / "temp.ass"
        wrapped_ass = tmp_path / "wrapped.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        temp_ass.write_text("temp", encoding="utf-8")
        wrapped_ass.write_text("wrapped", encoding="utf-8")
        released = []
        unlink_calls = []
        original_unlink = Path.unlink

        monkeypatch.setattr(wrapper, "_prepare_hard_embed_preflight", lambda **kwargs: True, raising=False)
        monkeypatch.setattr(
            wrapper,
            "_plan_hard_embed_subtitle_inputs",
            lambda **kwargs: ([str(temp_ass), str(wrapped_ass)], "ass='planned'"),
            raising=False,
        )
        monkeypatch.setattr(wrapper, "_resolve_hard_embed_gpu_device", lambda gpu_device: 4, raising=False)
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire",
            lambda gpu_id, timeout=600: True,
        )
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.release",
            lambda gpu_id: released.append(gpu_id),
        )
        monkeypatch.setattr(wrapper, "_check_nvenc_support", lambda: False)
        monkeypatch.setattr(
            wrapper,
            "_run_hard_embed_runtime_stage",
            lambda **kwargs: pytest.fail("unexpected runtime stage"),
            raising=False,
        )
        monkeypatch.setattr(
            Path,
            "unlink",
            lambda self, missing_ok=True: unlink_calls.append((str(self), missing_ok)) or original_unlink(self, missing_ok=missing_ok),
        )

        with pytest.raises(RuntimeError, match="不支持 NVENC"):
            wrapper.embed_subtitle_hard(
                video,
                sub,
                out,
                gpu_device="cuda:4",
                subtitle_style="named-style",
                style_dir="/styles",
                fonts_dir="/fonts",
                reference_height=900,
            )

        assert released == [4]
        assert unlink_calls == [
            (str(temp_ass), True),
            (str(wrapped_ass), True),
        ]
        assert temp_ass.exists() is False
        assert wrapped_ass.exists() is False

    def test_embed_subtitle_hard_cleans_temp_files_when_nvenc_slot_acquire_times_out_after_ass_prepare(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        temp_ass = tmp_path / "temp.ass"
        wrapped_ass = tmp_path / "wrapped.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        temp_ass.write_text("temp", encoding="utf-8")
        wrapped_ass.write_text("wrapped", encoding="utf-8")
        released = []
        unlink_calls = []
        original_unlink = Path.unlink

        monkeypatch.setattr(wrapper, "_prepare_hard_embed_preflight", lambda **kwargs: True, raising=False)
        monkeypatch.setattr(
            wrapper,
            "_plan_hard_embed_subtitle_inputs",
            lambda **kwargs: ([str(temp_ass), str(wrapped_ass)], "ass='planned'"),
            raising=False,
        )
        monkeypatch.setattr(wrapper, "_resolve_hard_embed_gpu_device", lambda gpu_device: 4, raising=False)
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire",
            lambda gpu_id, timeout=600: False,
        )
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.release",
            lambda gpu_id: released.append(gpu_id),
        )
        monkeypatch.setattr(
            wrapper,
            "_run_hard_embed_runtime_stage",
            lambda **kwargs: pytest.fail("unexpected runtime stage"),
            raising=False,
        )
        monkeypatch.setattr(
            Path,
            "unlink",
            lambda self, missing_ok=True: unlink_calls.append((str(self), missing_ok)) or original_unlink(self, missing_ok=missing_ok),
        )

        with pytest.raises(RuntimeError, match="获取超时"):
            wrapper.embed_subtitle_hard(
                video,
                sub,
                out,
                gpu_device="cuda:4",
                subtitle_style="named-style",
                style_dir="/styles",
                fonts_dir="/fonts",
                reference_height=900,
                max_nvenc_sessions=7,
            )

        assert released == []
        assert unlink_calls == [
            (str(temp_ass), True),
            (str(wrapped_ass), True),
        ]
        assert temp_ass.exists() is False
        assert wrapped_ass.exists() is False

    def test_embed_subtitle_hard_cleans_temp_files_when_gpu_device_resolution_raises_after_ass_prepare(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        sub = tmp_path / "sub.ass"
        out = tmp_path / "out.mp4"
        temp_ass = tmp_path / "temp.ass"
        wrapped_ass = tmp_path / "wrapped.ass"
        video.write_bytes(b"00")
        sub.write_text("dummy", encoding="utf-8")
        temp_ass.write_text("temp", encoding="utf-8")
        wrapped_ass.write_text("wrapped", encoding="utf-8")
        released = []
        unlink_calls = []
        prepared = []
        original_unlink = Path.unlink

        monkeypatch.setattr(wrapper, "_prepare_hard_embed_preflight", lambda **kwargs: True, raising=False)
        monkeypatch.setattr(
            wrapper,
            "_plan_hard_embed_subtitle_inputs",
            lambda **kwargs: ([str(temp_ass), str(wrapped_ass)], "ass='planned'"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_prepare_hard_embed_nvenc_session",
            lambda **kwargs: prepared.append(kwargs),
            raising=False,
        )
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.release",
            lambda gpu_id: released.append(gpu_id),
        )
        monkeypatch.setattr(
            wrapper,
            "_run_hard_embed_runtime_stage",
            lambda **kwargs: pytest.fail("unexpected runtime stage"),
            raising=False,
        )
        monkeypatch.setattr(
            Path,
            "unlink",
            lambda self, missing_ok=True: unlink_calls.append((str(self), missing_ok)) or original_unlink(self, missing_ok=missing_ok),
        )

        with pytest.raises(ValueError, match="无效的 GPU 设备格式"):
            wrapper.embed_subtitle_hard(
                video,
                sub,
                out,
                gpu_device="cuda:oops",
                subtitle_style="named-style",
                style_dir="/styles",
                fonts_dir="/fonts",
                reference_height=900,
            )

        assert prepared == []
        assert released == []
        assert unlink_calls == [
            (str(temp_ass), True),
            (str(wrapped_ass), True),
        ]
        assert temp_ass.exists() is False
        assert wrapped_ass.exists() is False

    def test_run_ffmpeg_embed_process_reports_progress_writes_log_and_succeeds(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr("time.sleep", lambda _seconds: None)
        monkeypatch.setattr("time.time", lambda: 10.0)
        wrapper = FFmpegWrapper()
        output = tmp_path / "out.mp4"
        callbacks = []

        class _FakeStderr:
            def __init__(self, lines, remaining=""):
                self._lines = list(lines)
                self._remaining = remaining

            def readline(self):
                return self._lines.pop(0) if self._lines else ""

            def read(self):
                return self._remaining

        class _FakeProcess:
            def __init__(self):
                self.stderr = _FakeStderr(
                    [
                        "Duration: 00:00:10.00\n",
                        "frame=1 time=00:00:05.00 bitrate=1000kbits/s\n",
                    ]
                )

            def poll(self):
                return None

            def wait(self):
                output.write_bytes(b"ok")
                return 0

        monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: _FakeProcess())

        result = wrapper._run_ffmpeg_embed_process(
            cmd=["ffmpeg", "-i", "input.mp4", str(output)],
            output_path=output,
            progress_callback=lambda progress, message: callbacks.append((progress, message)),
        )

        assert result is True
        assert callbacks[-1] == ("100", "合成完成")
        log_text = (output.parent / "ffmpeg_embed.log").read_text(encoding="utf-8")
        assert "=== FFmpeg 命令 ===" in log_text
        assert "Duration: 00:00:10.00" in log_text

    def test_run_ffmpeg_embed_process_reports_100_percent_progress_only_once_before_completion(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr("time.sleep", lambda _seconds: None)
        monkeypatch.setattr("time.time", lambda: 10.0)
        wrapper = FFmpegWrapper()
        output = tmp_path / "out.mp4"
        callbacks = []

        class _FakeStderr:
            def __init__(self, lines, remaining=""):
                self._lines = list(lines)
                self._remaining = remaining

            def readline(self):
                return self._lines.pop(0) if self._lines else ""

            def read(self):
                return self._remaining

        class _FakeProcess:
            def __init__(self):
                self.stderr = _FakeStderr(
                    [
                        "Duration: 00:00:10.00\n",
                        "frame=1 time=00:00:10.00 bitrate=1000kbits/s\n",
                        "frame=2 time=00:00:10.00 bitrate=1000kbits/s\n",
                    ]
                )

            def poll(self):
                return None

            def wait(self):
                output.write_bytes(b"ok")
                return 0

        monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: _FakeProcess())

        result = wrapper._run_ffmpeg_embed_process(
            cmd=["ffmpeg", "-i", "input.mp4", str(output)],
            output_path=output,
            progress_callback=lambda progress, message: callbacks.append((progress, message)),
        )

        progress_100_callbacks = [
            (progress, message)
            for progress, message in callbacks
            if message == "正在合成" and progress.startswith("100%")
        ]

        assert result is True
        assert len(progress_100_callbacks) == 1
        assert callbacks.count(("100", "合成完成")) == 1
        assert output.exists() is True
        log_text = (output.parent / "ffmpeg_embed.log").read_text(encoding="utf-8")
        assert "=== FFmpeg 命令 ===" in log_text
        assert "frame=2 time=00:00:10.00" in log_text

    def test_run_ffmpeg_embed_process_returns_false_and_persists_failure_log(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr("time.sleep", lambda _seconds: None)
        wrapper = FFmpegWrapper()
        output = tmp_path / "out.mp4"

        class _FakeStderr:
            def __init__(self, lines, remaining=""):
                self._lines = list(lines)
                self._remaining = remaining

            def readline(self):
                return self._lines.pop(0) if self._lines else ""

            def read(self):
                return self._remaining

        class _FakeProcess:
            def __init__(self):
                self.stderr = _FakeStderr(
                    [
                        "Duration: 00:00:10.00\n",
                        "Error while filtering\n",
                        "failed to encode\n",
                    ],
                    remaining="fatal tail\n",
                )

            def poll(self):
                return None

            def wait(self):
                return 1

        monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: _FakeProcess())

        result = wrapper._run_ffmpeg_embed_process(
            cmd=["ffmpeg", "-i", "input.mp4", str(output)],
            output_path=output,
            progress_callback=None,
        )

        assert result is False
        log_text = (output.parent / "ffmpeg_embed.log").read_text(encoding="utf-8")
        assert "Error while filtering" in log_text
        assert "fatal tail" in log_text

    def test_run_ffmpeg_embed_process_does_not_report_completion_callback_when_ffmpeg_fails(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr("time.sleep", lambda _seconds: None)
        monkeypatch.setattr("time.time", lambda: 10.0)
        wrapper = FFmpegWrapper()
        output = tmp_path / "out.mp4"
        callbacks = []

        class _FakeStderr:
            def __init__(self, lines, remaining=""):
                self._lines = list(lines)
                self._remaining = remaining

            def readline(self):
                return self._lines.pop(0) if self._lines else ""

            def read(self):
                return self._remaining

        class _FakeProcess:
            def __init__(self):
                self.stderr = _FakeStderr(
                    [
                        "Duration: 00:00:10.00\n",
                        "frame=1 time=00:00:10.00 bitrate=1000kbits/s\n",
                        "Error while filtering\n",
                    ],
                    remaining="fatal tail\n",
                )

            def poll(self):
                return None

            def wait(self):
                return 1

        monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: _FakeProcess())

        result = wrapper._run_ffmpeg_embed_process(
            cmd=["ffmpeg", "-i", "input.mp4", str(output)],
            output_path=output,
            progress_callback=lambda progress, message: callbacks.append((progress, message)),
        )

        assert result is False
        assert ("100", "合成完成") not in callbacks

    def test_run_ffmpeg_embed_process_returns_false_and_does_not_report_completion_when_output_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr("time.sleep", lambda _seconds: None)
        monkeypatch.setattr("time.time", lambda: 10.0)
        wrapper = FFmpegWrapper()
        output = tmp_path / "out.mp4"
        callbacks = []

        class _FakeStderr:
            def __init__(self, lines, remaining=""):
                self._lines = list(lines)
                self._remaining = remaining

            def readline(self):
                return self._lines.pop(0) if self._lines else ""

            def read(self):
                return self._remaining

        class _FakeProcess:
            def __init__(self):
                self.stderr = _FakeStderr(
                    [
                        "Duration: 00:00:10.00\n",
                        "frame=1 time=00:00:10.00 bitrate=1000kbits/s\n",
                    ]
                )

            def poll(self):
                return None

            def wait(self):
                return 0

        monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: _FakeProcess())

        result = wrapper._run_ffmpeg_embed_process(
            cmd=["ffmpeg", "-i", "input.mp4", str(output)],
            output_path=output,
            progress_callback=lambda progress, message: callbacks.append((progress, message)),
        )

        assert result is False
        assert output.exists() is False
        assert ("100", "合成完成") not in callbacks

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
        released = []

        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: None)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.select_gpu", lambda: 1)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire", lambda gpu_id, timeout=600: False)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: released.append(gpu_id))
        monkeypatch.setattr(wrapper, "_check_nvenc_support", lambda: pytest.fail("unexpected nvenc support check"))

        with pytest.raises(RuntimeError, match="获取超时"):
            wrapper.embed_subtitle_hard(video, sub, out, gpu_device="auto")

        assert released == []
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


class TestFFmpegWrapperMaskViolationPlanning:
    def test_prepare_mask_violation_context_returns_none_when_video_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        missing_video = tmp_path / "missing.mp4"
        out = tmp_path / "nested" / "masked.mp4"

        monkeypatch.setattr(wrapper, "get_video_info", lambda path: pytest.fail("unexpected video info fetch"))

        context = wrapper._prepare_mask_violation_context(
            video_path=missing_video,
            output_path=out,
            violation_ranges=[(10.0, 12.0)],
            margin_sec=1.5,
        )

        assert context is None
        assert out.parent.exists() is False

    def test_prepare_mask_violation_context_reads_video_info_and_merges_ranges(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "nested" / "masked.mp4"
        video.write_bytes(b"00")
        video_info = {
            "duration": 120.0,
            "video": {"width": 1280, "height": 720},
            "bit_rate": 2468,
        }
        merge_calls = []

        monkeypatch.setattr(wrapper, "get_video_info", lambda path: video_info)
        monkeypatch.setattr(
            wrapper,
            "_merge_ranges",
            lambda ranges, margin, max_duration: merge_calls.append((ranges, margin, max_duration)) or [(8.5, 13.5)],
            raising=False,
        )

        resolved_video_info, width, height, merged = wrapper._prepare_mask_violation_context(
            video_path=video,
            output_path=out,
            violation_ranges=[(10.0, 12.0)],
            margin_sec=1.5,
        )

        assert resolved_video_info == video_info
        assert width == 1280
        assert height == 720
        assert merged == [(8.5, 13.5)]
        assert merge_calls == [([(10.0, 12.0)], 1.5, 120.0)]
        assert out.parent.exists() is True

    def test_plan_mask_violation_filters_builds_expected_vf_and_af_with_cjk_font(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()

        monkeypatch.setattr(
            wrapper,
            "_find_cjk_font",
            lambda: "/fonts/Noto CJK:Bold's.otf",
            raising=False,
        )

        vf, af = wrapper._plan_mask_violation_filters(
            width=1280,
            height=720,
            merged=[(8.5, 13.5), (20.0, 25.0)],
            mask_text="违规:'片段':说明",
        )

        assert vf == (
            "drawbox=x=0:y=0:w=1280:h=720:color=black:t=fill:enable='between(t,8.5,13.5)',"
            "drawtext=text='违规\\:\\'片段\\'\\:说明':fontfile='/fonts/Noto CJK\\:Bold\\'s.otf':fontsize=24:fontcolor=white"
            ":x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,8.5,13.5)',"
            "drawbox=x=0:y=0:w=1280:h=720:color=black:t=fill:enable='between(t,20.0,25.0)',"
            "drawtext=text='违规\\:\\'片段\\'\\:说明':fontfile='/fonts/Noto CJK\\:Bold\\'s.otf':fontsize=24:fontcolor=white"
            ":x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,20.0,25.0)'"
        )
        assert af == (
            "volume=enable='between(t,8.5,13.5)':volume=0,"
            "volume=enable='between(t,20.0,25.0)':volume=0"
        )

    def test_plan_mask_violation_execution_selects_gpu_acquires_session_and_builds_vbr_command(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "masked.mp4"
        video.write_bytes(b"00")
        planned = []

        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.init",
            lambda max_per_gpu=5: planned.append(("init", max_per_gpu)),
        )
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.select_gpu",
            lambda: planned.append(("select", None)) or 3,
        )
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire",
            lambda gpu_id, timeout=300: planned.append(("acquire", {"gpu_id": gpu_id, "timeout": timeout})) or True,
        )

        gpu_id, cmd = wrapper._plan_mask_violation_execution(
            video_path=video,
            output_path=out,
            vf="vf=planned",
            af="af=planned",
            video_info={"bit_rate": 2000},
            gpu_device="auto",
        )

        assert gpu_id == 3
        assert cmd == [
            "ffmpeg",
            "-hwaccel", "cuda",
            "-hwaccel_device", "3",
            "-i", str(video),
            "-vf", "vf=planned",
            "-af", "af=planned",
            "-c:v", "hevc_nvenc",
            "-gpu", "3",
            "-rc", "vbr",
            "-cq", "23",
            "-b:v", "2200",
            "-maxrate", "3000",
            "-preset", "p4",
            "-c:a", "aac",
            "-movflags", "+faststart",
            "-y",
            str(out),
        ]
        assert planned == [
            ("init", 5),
            ("select", None),
            ("acquire", {"gpu_id": 3, "timeout": 300}),
        ]

    def test_plan_mask_violation_execution_falls_back_to_gpu_zero_and_constqp_when_bitrate_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "masked.mp4"
        video.write_bytes(b"00")
        acquired = []

        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: None)
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.select_gpu",
            lambda: pytest.fail("unexpected auto gpu selection"),
        )
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire",
            lambda gpu_id, timeout=300: acquired.append((gpu_id, timeout)) or True,
        )

        gpu_id, cmd = wrapper._plan_mask_violation_execution(
            video_path=video,
            output_path=out,
            vf="vf=planned",
            af="af=planned",
            video_info={"bit_rate": 0},
            gpu_device="cuda:not-a-number",
        )

        assert gpu_id == 0
        assert acquired == [(0, 300)]
        assert cmd == [
            "ffmpeg",
            "-hwaccel", "cuda",
            "-hwaccel_device", "0",
            "-i", str(video),
            "-vf", "vf=planned",
            "-af", "af=planned",
            "-c:v", "hevc_nvenc",
            "-gpu", "0",
            "-rc", "constqp",
            "-qp", "23",
            "-preset", "p4",
            "-c:a", "aac",
            "-movflags", "+faststart",
            "-y",
            str(out),
        ]

    def test_plan_mask_violation_execution_returns_none_when_nvenc_acquire_times_out(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "masked.mp4"
        video.write_bytes(b"00")

        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: None)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.select_gpu", lambda: 2)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire", lambda gpu_id, timeout=300: False)

        planned = wrapper._plan_mask_violation_execution(
            video_path=video,
            output_path=out,
            vf="vf=planned",
            af="af=planned",
            video_info={"bit_rate": 2468},
            gpu_device="auto",
        )

        assert planned is None

    def test_plan_mask_violation_execution_parses_explicit_gpu_device_into_command(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "masked.mp4"
        video.write_bytes(b"00")
        acquired = []

        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: None)
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.select_gpu",
            lambda: pytest.fail("unexpected auto gpu selection"),
        )
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire",
            lambda gpu_id, timeout=300: acquired.append((gpu_id, timeout)) or True,
        )

        gpu_id, cmd = wrapper._plan_mask_violation_execution(
            video_path=video,
            output_path=out,
            vf="vf=planned",
            af="af=planned",
            video_info={"bit_rate": 2468},
            gpu_device="cuda:4",
        )

        assert gpu_id == 4
        assert acquired == [(4, 300)]
        assert cmd[cmd.index("-hwaccel_device") + 1] == "4"
        assert cmd[cmd.index("-gpu") + 1] == "4"

    def test_mask_violation_segments_returns_false_when_execution_planning_fails(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "masked.mp4"
        video.write_bytes(b"00")
        released = []

        monkeypatch.setattr(
            wrapper,
            "_prepare_mask_violation_context",
            lambda **kwargs: ({"duration": 120.0, "video": {"width": 1280, "height": 720}, "bit_rate": 2468}, 1280, 720, [(8.5, 13.5)]),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_mask_violation_filters",
            lambda **kwargs: ("vf=planned", "af=planned"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_mask_violation_execution",
            lambda **kwargs: None,
            raising=False,
        )
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: pytest.fail("unexpected ffmpeg run"))
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: released.append(gpu_id))

        result = wrapper.mask_violation_segments(
            video_path=video,
            output_path=out,
            violation_ranges=[(10.0, 12.0)],
            mask_text="此处内容因平台合规要求已被遮罩",
            gpu_device="auto",
            margin_sec=1.5,
        )

        assert result is False
        assert released == []

    def test_run_mask_violation_runtime_stage_runs_ffmpeg_checks_output_and_releases_gpu(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "masked.mp4"
        video.write_bytes(b"00")
        released = []

        def fake_run(cmd, capture_output, text, timeout):
            assert cmd == ["ffmpeg", "planned"]
            assert capture_output is True
            assert text is True
            assert timeout == 3600
            out.write_bytes(b"ok")
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: released.append(gpu_id))

        result = wrapper._run_mask_violation_runtime_stage(
            video_path=video,
            output_path=out,
            gpu_id=4,
            cmd=["ffmpeg", "planned"],
        )

        assert result is True
        assert released == [4]

    def test_run_mask_violation_runtime_stage_returns_false_when_output_missing_and_releases_gpu(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "masked.mp4"
        video.write_bytes(b"00")
        released = []

        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, capture_output, text, timeout: SimpleNamespace(returncode=0, stderr=""),
        )
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: released.append(gpu_id))

        result = wrapper._run_mask_violation_runtime_stage(
            video_path=video,
            output_path=out,
            gpu_id=2,
            cmd=["ffmpeg", "planned"],
        )

        assert result is False
        assert released == [2]

    def test_run_mask_violation_runtime_stage_returns_false_when_subprocess_run_raises_and_releases_gpu(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "masked.mp4"
        video.write_bytes(b"00")
        released = []

        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, capture_output, text, timeout: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: released.append(gpu_id))

        result = wrapper._run_mask_violation_runtime_stage(
            video_path=video,
            output_path=out,
            gpu_id=6,
            cmd=["ffmpeg", "planned"],
        )

        assert result is False
        assert released == [6]

    def test_run_mask_violation_runtime_stage_returns_false_when_ffmpeg_exits_nonzero_and_releases_gpu(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "masked.mp4"
        video.write_bytes(b"00")
        released = []

        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, capture_output, text, timeout: SimpleNamespace(returncode=1, stderr="encode failed"),
        )
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: released.append(gpu_id))

        result = wrapper._run_mask_violation_runtime_stage(
            video_path=video,
            output_path=out,
            gpu_id=5,
            cmd=["ffmpeg", "planned"],
        )

        assert result is False
        assert released == [5]

    def test_run_mask_violation_runtime_stage_returns_false_when_ffmpeg_times_out_and_releases_gpu(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "masked.mp4"
        video.write_bytes(b"00")
        released = []

        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, capture_output, text, timeout: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)),
        )
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: released.append(gpu_id))

        result = wrapper._run_mask_violation_runtime_stage(
            video_path=video,
            output_path=out,
            gpu_id=7,
            cmd=["ffmpeg", "planned"],
        )

        assert result is False
        assert released == [7]

    def test_mask_violation_segments_delegates_runtime_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "masked.mp4"
        video.write_bytes(b"00")
        delegated = []

        monkeypatch.setattr(
            wrapper,
            "_prepare_mask_violation_context",
            lambda **kwargs: ({"duration": 120.0, "video": {"width": 1280, "height": 720}, "bit_rate": 2468}, 1280, 720, [(8.5, 13.5)]),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_mask_violation_filters",
            lambda **kwargs: ("vf=planned", "af=planned"),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_mask_violation_execution",
            lambda **kwargs: (4, ["ffmpeg", "planned"]),
            raising=False,
        )
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: pytest.fail("unexpected inline subprocess.run"))
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: pytest.fail("unexpected inline gpu release"))
        monkeypatch.setattr(
            wrapper,
            "_run_mask_violation_runtime_stage",
            lambda **kwargs: delegated.append(kwargs) or True,
            raising=False,
        )

        result = wrapper.mask_violation_segments(
            video_path=video,
            output_path=out,
            violation_ranges=[(10.0, 12.0)],
            mask_text="此处内容因平台合规要求已被遮罩",
            gpu_device="cuda:4",
            margin_sec=1.5,
        )

        assert result is True
        assert delegated == [{
            "video_path": video,
            "output_path": out,
            "gpu_id": 4,
            "cmd": ["ffmpeg", "planned"],
        }]

    def test_mask_violation_segments_delegates_execution_planning_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "masked.mp4"
        video.write_bytes(b"00")
        delegated = []

        video_info = {"duration": 120.0, "video": {"width": 1280, "height": 720}, "bit_rate": 2468}
        monkeypatch.setattr(
            wrapper,
            "_prepare_mask_violation_context",
            lambda **kwargs: (video_info, 1280, 720, [(8.5, 13.5)]),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_mask_violation_filters",
            lambda **kwargs: ("vf=planned", "af=planned"),
            raising=False,
        )
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.init",
            lambda max_per_gpu=5: pytest.fail("unexpected inline nvenc init"),
        )
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.select_gpu",
            lambda: pytest.fail("unexpected inline gpu selection"),
        )
        monkeypatch.setattr(
            "vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire",
            lambda gpu_id, timeout=300: pytest.fail("unexpected inline nvenc acquire"),
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_mask_violation_execution",
            lambda **kwargs: delegated.append(kwargs) or (4, ["ffmpeg", "planned"]),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_run_mask_violation_runtime_stage",
            lambda **kwargs: True,
            raising=False,
        )

        result = wrapper.mask_violation_segments(
            video_path=video,
            output_path=out,
            violation_ranges=[(10.0, 12.0)],
            mask_text="此处内容因平台合规要求已被遮罩",
            gpu_device="cuda:4",
            margin_sec=1.5,
        )

        assert result is True
        assert delegated == [{
            "video_path": video,
            "output_path": out,
            "vf": "vf=planned",
            "af": "af=planned",
            "video_info": video_info,
            "gpu_device": "cuda:4",
        }]

    def test_mask_violation_segments_delegates_filter_planning_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "masked.mp4"
        video.write_bytes(b"00")
        delegated = []

        monkeypatch.setattr(
            wrapper,
            "_prepare_mask_violation_context",
            lambda **kwargs: (
                {"duration": 120.0, "video": {"width": 1280, "height": 720}, "bit_rate": 2468},
                1280,
                720,
                [(8.5, 13.5)],
            ),
            raising=False,
        )
        monkeypatch.setattr(
            wrapper,
            "_plan_mask_violation_filters",
            lambda **kwargs: delegated.append(kwargs) or ("vf=planned", "af=planned"),
            raising=False,
        )
        monkeypatch.setattr(wrapper, "_find_cjk_font", lambda: pytest.fail("unexpected inline font lookup"), raising=False)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: None)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.select_gpu", lambda: 0)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire", lambda gpu_id, timeout=300: True)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: None)

        def fake_run(cmd, capture_output, text, timeout):
            assert cmd[cmd.index("-vf") + 1] == "vf=planned"
            assert cmd[cmd.index("-af") + 1] == "af=planned"
            out.write_bytes(b"ok")
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)

        result = wrapper.mask_violation_segments(
            video_path=video,
            output_path=out,
            violation_ranges=[(10.0, 12.0)],
            mask_text="此处内容因平台合规要求已被遮罩",
            gpu_device="auto",
            margin_sec=1.5,
        )

        assert result is True
        assert delegated == [{
            "width": 1280,
            "height": 720,
            "merged": [(8.5, 13.5)],
            "mask_text": "此处内容因平台合规要求已被遮罩",
        }]

    def test_mask_violation_segments_delegates_context_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
        wrapper = FFmpegWrapper()
        video = tmp_path / "video.mp4"
        out = tmp_path / "masked.mp4"
        video.write_bytes(b"00")
        delegated = []

        monkeypatch.setattr(
            wrapper,
            "_prepare_mask_violation_context",
            lambda **kwargs: delegated.append(kwargs) or (
                {"duration": 120.0, "video": {"width": 1280, "height": 720}, "bit_rate": 2468},
                1280,
                720,
                [(8.5, 13.5)],
            ),
            raising=False,
        )
        monkeypatch.setattr(wrapper, "get_video_info", lambda path: pytest.fail("unexpected inline video info fetch"))
        monkeypatch.setattr(wrapper, "_merge_ranges", lambda *args, **kwargs: pytest.fail("unexpected inline range merge"), raising=False)
        monkeypatch.setattr(
            Path,
            "mkdir",
            lambda self, parents=False, exist_ok=False: pytest.fail("unexpected inline output dir setup"),
        )
        monkeypatch.setattr(wrapper, "_find_cjk_font", lambda: None, raising=False)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.init", lambda max_per_gpu=5: None)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.select_gpu", lambda: 0)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.acquire", lambda gpu_id, timeout=300: True)
        monkeypatch.setattr("vat.embedder.ffmpeg_wrapper._nvenc_manager.release", lambda gpu_id: None)

        def fake_run(cmd, capture_output, text, timeout):
            out.write_bytes(b"ok")
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)

        result = wrapper.mask_violation_segments(
            video_path=video,
            output_path=out,
            violation_ranges=[(10.0, 12.0)],
            gpu_device="auto",
            margin_sec=1.5,
        )

        assert result is True
        assert delegated == [{
            "video_path": video,
            "output_path": out,
            "violation_ranges": [(10.0, 12.0)],
            "margin_sec": 1.5,
        }]
