"""
downloaders 模块单元测试

测试 YouTubeDownloader URL 解析、playlist 识别、ydl_opts 生成。
不测试实际下载（需要网络）。
"""
import pytest
from pathlib import Path
from vat.downloaders import YouTubeDownloader


@pytest.fixture
def dl():
    return YouTubeDownloader()


class TestIsPlaylistUrl:

    def test_playlist_url(self, dl):
        assert dl.is_playlist_url("https://www.youtube.com/playlist?list=PLxxx")

    def test_video_with_list_param(self, dl):
        assert dl.is_playlist_url("https://youtube.com/watch?v=xxx&list=PLyyy")

    def test_plain_video_url(self, dl):
        assert not dl.is_playlist_url("https://youtube.com/watch?v=xxx")

    def test_short_url(self, dl):
        assert not dl.is_playlist_url("https://youtu.be/xxx")


class TestExtractPlaylistId:

    def test_from_playlist_url(self, dl):
        assert dl.extract_playlist_id(
            "https://youtube.com/playlist?list=PLtest123"
        ) == "PLtest123"

    def test_from_video_with_list(self, dl):
        assert dl.extract_playlist_id(
            "https://youtube.com/watch?v=xxx&list=PLfoo"
        ) == "PLfoo"


class TestExtractVideoId:

    def test_standard_url(self, dl):
        vid = dl.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert vid == "dQw4w9WgXcQ"

    def test_short_url(self, dl):
        vid = dl.extract_video_id("https://youtu.be/dQw4w9WgXcQ")
        assert vid == "dQw4w9WgXcQ"

    def test_invalid_url_returns_none(self, dl):
        assert dl.extract_video_id("https://example.com") is None


class TestYdlOpts:
    """_get_ydl_opts 配置生成"""

    def test_basic_opts(self, dl):
        opts = dl._get_ydl_opts(Path("/tmp/test"))
        assert opts['format'] == dl.video_format
        assert 'outtmpl' in opts

    def test_proxy_included_when_set(self):
        dl = YouTubeDownloader(proxy="http://proxy:8080")
        opts = dl._get_ydl_opts(Path("/tmp/test"))
        assert opts['proxy'] == "http://proxy:8080"

    def test_proxy_absent_when_empty(self, dl):
        opts = dl._get_ydl_opts(Path("/tmp/test"))
        assert 'proxy' not in opts

    def test_cookies_included_when_file_exists(self, tmp_path):
        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text("# Netscape cookies")
        dl = YouTubeDownloader(cookies_file=str(cookie_file))
        opts = dl._get_ydl_opts(Path("/tmp/test"))
        assert opts['cookiefile'] == str(cookie_file)

    def test_cookies_skipped_when_file_missing(self):
        dl = YouTubeDownloader(cookies_file="/nonexistent/cookies.txt")
        opts = dl._get_ydl_opts(Path("/tmp/test"))
        assert 'cookiefile' not in opts

    def test_cookies_skipped_when_empty(self, dl):
        opts = dl._get_ydl_opts(Path("/tmp/test"))
        assert 'cookiefile' not in opts

    def test_remote_components_included(self):
        dl = YouTubeDownloader(remote_components=["ejs:github"])
        opts = dl._get_ydl_opts(Path("/tmp/test"))
        assert opts['remote_components'] == ["ejs:github"]

    def test_remote_components_absent_when_empty(self, dl):
        opts = dl._get_ydl_opts(Path("/tmp/test"))
        assert 'remote_components' not in opts

    def test_subtitle_opts_when_enabled(self, dl):
        opts = dl._get_ydl_opts(Path("/tmp/test"), download_subs=True,
                                sub_langs=["ja", "en"])
        assert opts['writeautomaticsub'] is True
        assert opts['writesubtitles'] is True
        assert opts['subtitleslangs'] == ["ja", "en"]

    def test_no_subtitle_opts_by_default(self, dl):
        opts = dl._get_ydl_opts(Path("/tmp/test"))
        assert 'writesubtitles' not in opts


class TestGenerateVideoId:
    """generate_video_id_from_url 哈希 ID 生成"""

    def test_deterministic(self):
        url = "https://youtube.com/watch?v=abc"
        id1 = YouTubeDownloader.generate_video_id_from_url(url)
        id2 = YouTubeDownloader.generate_video_id_from_url(url)
        assert id1 == id2

    def test_length_16(self):
        vid = YouTubeDownloader.generate_video_id_from_url("https://youtube.com/watch?v=x")
        assert len(vid) == 16

    def test_different_urls_different_ids(self):
        id1 = YouTubeDownloader.generate_video_id_from_url("https://youtube.com/watch?v=a")
        id2 = YouTubeDownloader.generate_video_id_from_url("https://youtube.com/watch?v=b")
        assert id1 != id2

    def test_consistent_with_create_video_from_url(self):
        """与 create_video_from_url 使用相同的哈希逻辑"""
        import hashlib
        url = "https://youtube.com/watch?v=test123"
        expected = hashlib.md5(url.encode()).hexdigest()[:16]
        assert YouTubeDownloader.generate_video_id_from_url(url) == expected


class TestChannelPattern:
    """频道 URL 正则匹配"""

    def test_at_handle(self, dl):
        assert dl.channel_pattern.search("https://www.youtube.com/@SomeChannel")

    def test_channel_path(self, dl):
        assert dl.channel_pattern.search("https://www.youtube.com/channel/UCxxxxxxx")

    def test_user_path(self, dl):
        assert dl.channel_pattern.search("https://www.youtube.com/user/SomeName")

    def test_non_channel_url(self, dl):
        assert not dl.channel_pattern.search("https://example.com/video")
