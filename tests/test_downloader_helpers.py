"""下载器 helper 契约测试。"""

import subprocess
from pathlib import Path
from types import SimpleNamespace

from vat.downloaders.base import BaseDownloader
from vat.downloaders.direct_url import DirectURLDownloader
from vat.downloaders.local import LocalImporter, generate_content_based_id
from vat.downloaders.youtube import VideoInfoResult, is_upcoming_event_error, is_video_permanently_unavailable


class TestLocalImporterHelpers:
    def test_generate_content_based_id_is_stable(self, tmp_path):
        video = tmp_path / "sample.mp4"
        video.write_bytes(b"abc123")

        assert generate_content_based_id(video) == generate_content_based_id(video)

    def test_validate_source_accepts_supported_video_file(self, tmp_path):
        video = tmp_path / "sample.mp4"
        video.write_bytes(b"abc")

        assert LocalImporter().validate_source(str(video)) is True

    def test_validate_source_rejects_unsupported_extension(self, tmp_path):
        video = tmp_path / "sample.txt"
        video.write_text("abc", encoding="utf-8")

        assert LocalImporter().validate_source(str(video)) is False


class TestDirectUrlHelpers:
    def test_guess_extension_prefers_url_suffix(self):
        downloader = DirectURLDownloader()
        resp = SimpleNamespace(headers={"content-type": "video/webm"})

        assert downloader._guess_extension("https://cdn.example.com/video.mkv", resp) == ".mkv"

    def test_guess_extension_falls_back_to_content_type(self):
        downloader = DirectURLDownloader()
        resp = SimpleNamespace(headers={"content-type": "video/webm"})

        assert downloader._guess_extension("https://cdn.example.com/download", resp) == ".webm"

    def test_title_from_url_uses_filename_stem(self):
        assert DirectURLDownloader._title_from_url("https://cdn.example.com/path/video-name.mp4") == "video-name"

    def test_title_from_url_returns_empty_for_generic_name(self):
        assert DirectURLDownloader._title_from_url("https://cdn.example.com/download") == ""


class TestYoutubeHelperContracts:
    def test_video_info_result_properties(self):
        ok = VideoInfoResult(status="ok", info={"upload_date": "20250101"})
        unavailable = VideoInfoResult(status="unavailable", error_message="private")

        assert ok.ok is True
        assert ok.upload_date == "20250101"
        assert unavailable.is_unavailable is True
        assert unavailable.upload_date is None

    def test_is_upcoming_event_error_detects_premiere_messages(self):
        assert is_upcoming_event_error("This live event will begin in 2 hours") is True
        assert is_upcoming_event_error("normal error") is False

    def test_is_video_permanently_unavailable_detects_private_removed(self):
        assert is_video_permanently_unavailable("This video is private") is True
        assert is_video_permanently_unavailable("Connection reset by peer") is False


class TestBaseDownloaderProbeVideoMetadata:
    def test_probe_video_metadata_parses_ffprobe_json(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(stdout="""
            {
              "format": {"duration": "12.5", "size": "1024", "bit_rate": "800000"},
              "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080, "r_frame_rate": "30000/1001"},
                {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2}
              ]
            }
            """),
        )

        info = BaseDownloader.probe_video_metadata(tmp_path / "video.mp4")

        assert info["duration"] == 12.5
        assert info["video"]["codec"] == "h264"
        assert info["audio"]["codec"] == "aac"

    def test_probe_video_metadata_returns_none_when_ffprobe_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("ffprobe missing")),
        )

        assert BaseDownloader.probe_video_metadata(tmp_path / "video.mp4") is None
