"""youtube downloader 行为测试。"""

from contextlib import contextmanager
from pathlib import Path

import pytest

from vat.downloaders.youtube import YouTubeDownloader


class TestGetVideoInfoFallback:
    def test_get_video_info_uses_isolated_cookiefile_copy(self, monkeypatch, tmp_path):
        source_cookiefile = tmp_path / "cookies.txt"
        source_cookiefile.write_text(
            "# Netscape HTTP Cookie File\n"
            ".youtube.com\tTRUE\t/\tFALSE\t0\tSID\ttest-cookie\n",
            encoding="utf-8",
        )

        seen_cookiefiles = []

        class FakeYoutubeDL:
            def __init__(self, opts):
                cookiefile = opts.get("cookiefile")
                seen_cookiefiles.append(cookiefile)
                assert cookiefile is not None
                assert cookiefile != str(source_cookiefile)
                assert Path(cookiefile).read_text(encoding="utf-8") == source_cookiefile.read_text(encoding="utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download=False):
                return {
                    "id": "test12345678",
                    "title": "Test Title",
                    "description": "Test Description",
                    "duration": 123,
                    "uploader": "Test Uploader",
                    "upload_date": "20250101",
                    "thumbnail": "https://example.com/thumb.jpg",
                    "live_status": "was_live",
                }

        monkeypatch.setattr("vat.downloaders.youtube.YoutubeDL", FakeYoutubeDL)

        downloader = YouTubeDownloader(cookies_file=str(source_cookiefile))

        result = downloader.get_video_info("https://www.youtube.com/watch?v=test12345678")

        assert result.ok
        assert len(seen_cookiefiles) == 1
        assert not Path(seen_cookiefiles[0]).exists()

    def test_get_video_info_falls_back_to_mweb_when_default_client_needs_reload(self, monkeypatch):
        calls = []

        class FakeYoutubeDL:
            def __init__(self, opts):
                self.opts = opts
                calls.append(opts)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download=False):
                extractor_args = self.opts.get("extractor_args")
                if not extractor_args:
                    raise RuntimeError("ERROR: [youtube] test12345678: The page needs to be reloaded.")
                assert extractor_args == {"youtube": {"player_client": ["mweb"]}}
                return {
                    "id": "test12345678",
                    "title": "Test Title",
                    "description": "Test Description",
                    "duration": 123,
                    "uploader": "Test Uploader",
                    "upload_date": "20250101",
                    "thumbnail": "https://example.com/thumb.jpg",
                    "live_status": "was_live",
                }

        monkeypatch.setattr("vat.downloaders.youtube.YoutubeDL", FakeYoutubeDL)

        downloader = YouTubeDownloader(
            proxy="http://proxy.local:1080",
            cookies_file=str(Path("/tmp/not-used-cookies.txt")),
            remote_components=["ejs:github"],
        )

        result = downloader.get_video_info("https://www.youtube.com/watch?v=test12345678")

        assert result.ok
        assert result.info["thumbnail"] == "https://example.com/thumb.jpg"
        assert result.info["upload_date"] == "20250101"
        assert len(calls) == 2
        assert "extractor_args" not in calls[0]
        assert calls[1]["extractor_args"] == {"youtube": {"player_client": ["mweb"]}}


class TestDownloadPathResilience:
    def test_download_requires_lock_db_path(self, monkeypatch, tmp_path):
        downloader = YouTubeDownloader()

        monkeypatch.setattr(
            downloader,
            "_extract_info_with_retry",
            lambda *_args, **_kwargs: {
                "id": "abc123def45",
                "title": "Test Title",
                "description": "Test Description",
                "duration": 123,
                "uploader": "Uploader",
                "upload_date": "20250101",
                "thumbnail": "https://example.com/thumb.jpg",
                "subtitles": {},
                "automatic_captions": {},
            },
        )
        monkeypatch.setattr(
            downloader,
            "_download_with_retry",
            lambda *_args, **_kwargs: (tmp_path / "abc123def45.mp4").write_bytes(b"00"),
        )

        with pytest.raises(RuntimeError, match="lock_db_path"):
            downloader.download("https://www.youtube.com/watch?v=abc123def45", tmp_path)

    def test_download_acquires_global_resource_lock(self, monkeypatch, tmp_path):
        calls = []

        @contextmanager
        def fake_resource_lock(**kwargs):
            calls.append(kwargs)
            yield object()

        downloader = YouTubeDownloader(
            lock_db_path=str(tmp_path / "locks.db"),
            download_cooldown=17,
            max_concurrent_downloads=2,
        )

        monkeypatch.setattr("vat.downloaders.youtube.resource_lock", fake_resource_lock, raising=False)
        monkeypatch.setattr(
            downloader,
            "_extract_info_with_retry",
            lambda *_args, **_kwargs: {
                "id": "abc123def45",
                "title": "Test Title",
                "description": "Test Description",
                "duration": 123,
                "uploader": "Uploader",
                "upload_date": "20250101",
                "thumbnail": "https://example.com/thumb.jpg",
                "subtitles": {},
                "automatic_captions": {},
            },
        )
        monkeypatch.setattr(
            downloader,
            "_download_with_retry",
            lambda *_args, **_kwargs: (tmp_path / "abc123def45.mp4").write_bytes(b"00"),
        )

        result = downloader.download("https://www.youtube.com/watch?v=abc123def45", tmp_path)

        assert result["video_path"] == tmp_path / "abc123def45.mp4"
        assert calls == [{
            "db_path": str(tmp_path / "locks.db"),
            "resource_type": "youtube_download",
            "cooldown_seconds": 17,
            "timeout_seconds": 1800,
            "lock_ttl_seconds": 5400,
            "max_concurrent": 2,
        }]

    def test_get_ydl_opts_does_not_force_player_clients_by_default(self, tmp_path):
        downloader = YouTubeDownloader()

        opts = downloader._get_ydl_opts(tmp_path)

        assert "extractor_args" not in opts

    def test_download_with_retry_uses_isolated_cookiefile_copy(self, monkeypatch, tmp_path):
        source_cookiefile = tmp_path / "cookies.txt"
        source_cookiefile.write_text(
            "# Netscape HTTP Cookie File\n"
            ".youtube.com\tTRUE\t/\tFALSE\t0\tSID\ttest-cookie\n",
            encoding="utf-8",
        )

        seen_cookiefiles = []

        class FakeYoutubeDL:
            def __init__(self, opts):
                cookiefile = opts.get("cookiefile")
                seen_cookiefiles.append(cookiefile)
                assert cookiefile is not None
                assert cookiefile != str(source_cookiefile)
                assert Path(cookiefile).read_text(encoding="utf-8") == source_cookiefile.read_text(encoding="utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def download(self, urls):
                return 0

        monkeypatch.setattr("vat.downloaders.youtube.YoutubeDL", FakeYoutubeDL)

        downloader = YouTubeDownloader(cookies_file=str(source_cookiefile))
        opts = downloader._get_ydl_opts(tmp_path)

        downloader._download_with_retry("https://www.youtube.com/watch?v=test12345678", opts, "test12345678")

        assert len(seen_cookiefiles) == 1
        assert not Path(seen_cookiefiles[0]).exists()

    def test_download_with_retry_retries_on_nonzero_return_with_retryable_error(self, monkeypatch, tmp_path):
        attempts = []

        class FakeYoutubeDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def download(self, urls):
                attempts.append(list(urls))
                if len(attempts) == 1:
                    self.opts["logger"].error("[download] Got error: EOF occurred in violation of protocol (_ssl.c:997)")
                    return 1
                return 0

        monkeypatch.setattr("vat.downloaders.youtube.YoutubeDL", FakeYoutubeDL)
        monkeypatch.setattr("vat.downloaders.youtube.time.sleep", lambda *_args, **_kwargs: None)

        downloader = YouTubeDownloader()
        opts = downloader._get_ydl_opts(tmp_path)

        downloader._download_with_retry("https://www.youtube.com/watch?v=test12345678", opts, "test12345678")

        assert len(attempts) == 2

    def test_extract_info_with_retry_falls_back_to_no_cookie_when_cookie_triggers_reload(self, monkeypatch, tmp_path):
        source_cookiefile = tmp_path / "cookies.txt"
        source_cookiefile.write_text(
            "# Netscape HTTP Cookie File\n"
            ".youtube.com\tTRUE\t/\tFALSE\t0\tSID\ttest-cookie\n",
            encoding="utf-8",
        )

        seen_cookiefiles = []
        warnings = []

        class FakeYoutubeDL:
            def __init__(self, opts):
                self.opts = opts
                seen_cookiefiles.append(opts.get("cookiefile"))

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download=False):
                if self.opts.get("cookiefile"):
                    raise RuntimeError("ERROR: [youtube] test12345678: The page needs to be reloaded.")
                return {
                    "id": "test12345678",
                    "title": "Test Title",
                    "description": "Test Description",
                    "duration": 123,
                    "uploader": "Test Uploader",
                    "upload_date": "20250101",
                    "thumbnail": "https://example.com/thumb.jpg",
                    "live_status": "was_live",
                }

        monkeypatch.setattr("vat.downloaders.youtube.YoutubeDL", FakeYoutubeDL)
        monkeypatch.setattr("vat.downloaders.youtube.logger.warning", lambda msg: warnings.append(msg))

        downloader = YouTubeDownloader(cookies_file=str(source_cookiefile))
        opts = downloader._get_ydl_opts(tmp_path)
        extract_opts = {k: v for k, v in opts.items() if k != "ignoreerrors"}

        info = downloader._extract_info_with_retry(
            "https://www.youtube.com/watch?v=test12345678",
            extract_opts,
        )

        assert info["id"] == "test12345678"
        assert len(seen_cookiefiles) == 2
        assert seen_cookiefiles[0] is not None
        assert seen_cookiefiles[1] is None
        assert any("youtube.com cookies" in msg for msg in warnings)

    def test_download_with_retry_falls_back_to_no_cookie_when_cookie_triggers_reload(self, monkeypatch, tmp_path):
        source_cookiefile = tmp_path / "cookies.txt"
        source_cookiefile.write_text(
            "# Netscape HTTP Cookie File\n"
            ".youtube.com\tTRUE\t/\tFALSE\t0\tSID\ttest-cookie\n",
            encoding="utf-8",
        )

        seen_cookiefiles = []
        warnings = []

        class FakeYoutubeDL:
            def __init__(self, opts):
                self.opts = opts
                seen_cookiefiles.append(opts.get("cookiefile"))

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def download(self, urls):
                if self.opts.get("cookiefile"):
                    raise RuntimeError("ERROR: [youtube] test12345678: The page needs to be reloaded.")
                return 0

        monkeypatch.setattr("vat.downloaders.youtube.YoutubeDL", FakeYoutubeDL)
        monkeypatch.setattr("vat.downloaders.youtube.logger.warning", lambda msg: warnings.append(msg))

        downloader = YouTubeDownloader(cookies_file=str(source_cookiefile))
        opts = downloader._get_ydl_opts(tmp_path)

        downloader._download_with_retry(
            "https://www.youtube.com/watch?v=test12345678",
            opts,
            "test12345678",
        )

        assert len(seen_cookiefiles) == 2
        assert seen_cookiefiles[0] is not None
        assert seen_cookiefiles[1] is None
        assert any("youtube.com cookies" in msg for msg in warnings)
