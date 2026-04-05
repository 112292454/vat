"""Bilibili 上传器基础 API 契约测试。"""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vat.uploaders.bilibili import BilibiliUploader, UploadResult, create_bilibili_uploader


def _make_uploader(tmp_path):
    uploader = BilibiliUploader.__new__(BilibiliUploader)
    uploader.cookies_file = tmp_path / "cookies.json"
    uploader.line = "AUTO"
    uploader.threads = 3
    uploader.lock_db_path = str(tmp_path / "locks.db")
    uploader.upload_interval = 60
    uploader.cookie_data = {"SESSDATA": "s", "bili_jct": "csrf", "DedeUserID": "uid"}
    uploader._raw_cookie_data = {"cookie_info": {"cookies": []}}
    uploader._cookie_loaded = True
    uploader._load_cookie = MagicMock()
    return uploader


class _FakeData:
    def __init__(self):
        self.copyright = None
        self.title = ""
        self.desc = ""
        self.tid = None
        self.tags = None
        self.source = ""
        self.dynamic = ""
        self.cover = ""
        self.delay = None
        self.appended = []

    def set_tag(self, tags):
        self.tags = tags

    def delay_time(self, dtime):
        self.delay = dtime

    def append(self, video_part):
        self.appended.append(video_part)


class _FakeBiliClient:
    def __init__(self, submit_payload=None, upload_result=None, cover_error=None):
        self.submit_payload = submit_payload or {"code": 0, "data": {"bvid": "BV1", "aid": 123}}
        self.upload_result = upload_result or {"filename": "file1"}
        self.cover_error = cover_error
        self.login_calls = []
        self.cover_calls = []
        self.upload_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def login_by_cookies(self, raw_cookie_data):
        self.login_calls.append(raw_cookie_data)

    def cover_up(self, path):
        self.cover_calls.append(path)
        if self.cover_error:
            raise self.cover_error
        return "http://cover.example/img.jpg"

    def upload_file(self, path, lines, tasks):
        self.upload_calls.append((path, lines, tasks))
        return self.upload_result

    def submit_web(self):
        return self.submit_payload


class TestUploadContracts:
    def test_upload_requires_lock_db_path(self, tmp_path, monkeypatch):
        uploader = _make_uploader(tmp_path)
        uploader.lock_db_path = ""
        video = tmp_path / "video.mp4"
        video.write_bytes(b"00")
        monkeypatch.setattr("vat.uploaders.bilibili.Data", _FakeData)
        monkeypatch.setattr("vat.uploaders.bilibili.BiliBili", lambda _data: _FakeBiliClient())

        with pytest.raises(RuntimeError, match="lock_db_path"):
            uploader.upload(
                video_path=video,
                title="标题",
                description="简介",
                tid=21,
                tags=["tag"],
            )

    def test_upload_acquires_global_resource_lock(self, tmp_path, monkeypatch):
        uploader = _make_uploader(tmp_path)
        uploader.upload_interval = 42
        video = tmp_path / "video.mp4"
        video.write_bytes(b"00")
        calls = []

        @contextmanager
        def fake_resource_lock(**kwargs):
            calls.append(kwargs)
            yield object()

        monkeypatch.setattr("vat.uploaders.bilibili.resource_lock", fake_resource_lock, raising=False)
        monkeypatch.setattr("vat.uploaders.bilibili.Data", _FakeData)
        monkeypatch.setattr("vat.uploaders.bilibili.BiliBili", lambda _data: _FakeBiliClient())

        result = uploader.upload(
            video_path=video,
            title="标题",
            description="简介",
            tid=21,
            tags=["tag"],
        )

        assert result.success is True
        assert calls == [{
            "db_path": uploader.lock_db_path,
            "resource_type": "bilibili_upload",
            "cooldown_seconds": 42,
            "timeout_seconds": 1800,
            "lock_ttl_seconds": 7200,
        }]

    def test_upload_returns_failure_when_video_path_missing(self, tmp_path):
        uploader = _make_uploader(tmp_path)

        result = uploader.upload(
            video_path=tmp_path / "missing.mp4",
            title="标题",
            description="简介",
            tid=21,
            tags=["tag"],
        )

        assert isinstance(result, UploadResult)
        assert result.success is False
        assert "视频文件不存在" in result.error

    def test_upload_returns_failure_when_load_cookie_fails(self, tmp_path):
        uploader = _make_uploader(tmp_path)
        uploader._load_cookie.side_effect = RuntimeError("cookie fail")
        video = tmp_path / "video.mp4"
        video.write_bytes(b"00")

        result = uploader.upload(
            video_path=video,
            title="标题",
            description="简介",
            tid=21,
            tags=["tag"],
        )

        assert result.success is False
        assert "加载cookie失败" in result.error

    def test_upload_continues_when_cover_upload_fails(self, tmp_path, monkeypatch):
        uploader = _make_uploader(tmp_path)
        video = tmp_path / "video.mp4"
        cover = tmp_path / "cover.jpg"
        video.write_bytes(b"00")
        cover.write_bytes(b"11")
        client = _FakeBiliClient(cover_error=RuntimeError("cover fail"))
        monkeypatch.setattr("vat.uploaders.bilibili.Data", _FakeData)
        monkeypatch.setattr("vat.uploaders.bilibili.BiliBili", lambda _data: client)

        result = uploader.upload(
            video_path=video,
            title="标题",
            description="简介",
            tid=21,
            tags=["tag"],
            cover_path=cover,
        )

        assert result.success is True
        assert client.cover_calls == [str(cover)]
        assert client.upload_calls == [(str(video), "AUTO", 3)]

    def test_upload_returns_failure_when_submit_web_returns_error(self, tmp_path, monkeypatch):
        uploader = _make_uploader(tmp_path)
        video = tmp_path / "video.mp4"
        video.write_bytes(b"00")
        client = _FakeBiliClient(submit_payload={"code": -1, "message": "submit failed"})
        monkeypatch.setattr("vat.uploaders.bilibili.Data", _FakeData)
        monkeypatch.setattr("vat.uploaders.bilibili.BiliBili", lambda _data: client)

        result = uploader.upload(
            video_path=video,
            title="标题",
            description="简介",
            tid=21,
            tags=["tag"],
        )

        assert result.success is False
        assert result.error == "submit failed"

    def test_upload_retries_keyerror_then_succeeds(self, tmp_path, monkeypatch):
        uploader = _make_uploader(tmp_path)
        video = tmp_path / "video.mp4"
        video.write_bytes(b"00")
        sleep_calls = []

        class _RetryClient(_FakeBiliClient):
            def __init__(self):
                super().__init__(submit_payload={"code": 0, "data": {"bvid": "BV2", "aid": 456}})
                self._attempt = 0

            def upload_file(self, path, lines, tasks):
                self._attempt += 1
                if self._attempt < 3:
                    raise KeyError("auth")
                return {"filename": "retry-file"}

        client = _RetryClient()
        monkeypatch.setattr("vat.uploaders.bilibili.Data", _FakeData)
        monkeypatch.setattr("vat.uploaders.bilibili.BiliBili", lambda _data: client)
        monkeypatch.setattr("vat.uploaders.bilibili.time.sleep", lambda seconds: sleep_calls.append(seconds))

        result = uploader.upload(
            video_path=video,
            title="标题",
            description="简介",
            tid=21,
            tags=["tag"],
        )

        assert result.success is True
        assert sleep_calls == [120, 240]


class TestUploadResultContracts:
    def test_upload_result_bool_follows_success_flag(self):
        assert bool(UploadResult(success=True)) is True
        assert bool(UploadResult(success=False)) is False


class TestUploadWithMetadataContracts:
    def test_upload_with_metadata_requires_title(self, tmp_path):
        uploader = _make_uploader(tmp_path)
        video = tmp_path / "video.mp4"
        video.write_bytes(b"00")

        with pytest.raises(ValueError, match="缺少 title"):
            uploader.upload_with_metadata(video, {"desc": "简介"})

    def test_upload_with_metadata_forwards_to_upload(self, tmp_path, monkeypatch):
        uploader = _make_uploader(tmp_path)
        video = tmp_path / "video.mp4"
        video.write_bytes(b"00")
        captured = {}
        monkeypatch.setattr(
            uploader,
            "upload",
            lambda **kwargs: captured.update(kwargs) or UploadResult(success=True, bvid="BV1", aid=123),
        )

        result = uploader.upload_with_metadata(
            video,
            {"title": "标题", "desc": "简介", "tid": 17, "tags": ["a", "b"]},
        )

        assert result.success is True
        assert captured["title"] == "标题"


class TestReplacementUploadContracts:
    def test_upload_replacement_file_acquires_global_resource_lock(self, tmp_path, monkeypatch):
        uploader = _make_uploader(tmp_path)
        uploader.upload_interval = 33
        video = tmp_path / "replacement.mp4"
        video.write_bytes(b"00")
        calls = []

        @contextmanager
        def fake_resource_lock(**kwargs):
            calls.append(kwargs)
            yield object()

        monkeypatch.setattr("vat.uploaders.bilibili.resource_lock", fake_resource_lock, raising=False)
        monkeypatch.setattr("vat.uploaders.bilibili.Data", _FakeData)
        monkeypatch.setattr("vat.uploaders.bilibili.BiliBili", lambda _data: _FakeBiliClient(upload_result={"filename": "repl.mp4"}))

        filename = uploader._upload_replacement_file(video, {"title": "old", "desc": "old desc"})

        assert filename == "repl.mp4"
        assert calls == [{
            "db_path": uploader.lock_db_path,
            "resource_type": "bilibili_upload",
            "cooldown_seconds": 33,
            "timeout_seconds": 1800,
            "lock_ttl_seconds": 7200,
        }]


class TestValidateCredentialsContracts:
    def test_validate_credentials_returns_true_when_required_cookies_present(self, tmp_path):
        uploader = _make_uploader(tmp_path)

        assert uploader.validate_credentials() is True

    def test_validate_credentials_returns_false_when_required_cookie_missing(self, tmp_path):
        uploader = _make_uploader(tmp_path)
        uploader.cookie_data = {"SESSDATA": "s"}

        assert uploader.validate_credentials() is False

    def test_validate_credentials_returns_false_when_load_cookie_raises(self, tmp_path):
        uploader = _make_uploader(tmp_path)
        uploader._load_cookie.side_effect = RuntimeError("bad cookie")

        assert uploader.validate_credentials() is False


class TestSeasonCreationAndListingContracts:
    def test_list_seasons_parses_sections_ep_count(self, tmp_path):
        uploader = _make_uploader(tmp_path)
        session = MagicMock()
        session.get.return_value = MagicMock(json=lambda: {
            "code": 0,
            "data": {
                "seasons": [
                    {
                        "season": {"id": 42, "title": "合集A", "desc": "desc", "cover": "cover.jpg"},
                        "sections": {"sections": [{"epCount": 7}]},
                    }
                ]
            },
        })
        uploader._get_authenticated_session = MagicMock(return_value=session)

        assert uploader.list_seasons() == [{
            "season_id": 42,
            "name": "合集A",
            "description": "desc",
            "cover": "cover.jpg",
            "total": 7,
        }]

    def test_list_seasons_returns_empty_on_api_failure(self, tmp_path):
        uploader = _make_uploader(tmp_path)
        session = MagicMock()
        session.get.return_value = MagicMock(json=lambda: {"code": -1, "message": "failed"})
        uploader._get_authenticated_session = MagicMock(return_value=session)

        assert uploader.list_seasons() == []

    def test_list_seasons_returns_empty_on_exception(self, tmp_path):
        uploader = _make_uploader(tmp_path)
        session = MagicMock()
        session.get.side_effect = RuntimeError("boom")
        uploader._get_authenticated_session = MagicMock(return_value=session)

        assert uploader.list_seasons() == []

    def test_create_season_rewrites_minus_400_error(self, tmp_path):
        uploader = _make_uploader(tmp_path)
        session = MagicMock()
        session.post.return_value = MagicMock(json=lambda: {"code": -400, "message": "bad request"})
        uploader._get_authenticated_session = MagicMock(return_value=session)

        result = uploader.create_season("新合集", "简介")

        assert result["success"] is False
        assert "API 请求错误" in result["error"]

    def test_create_season_returns_season_id_on_success(self, tmp_path):
        uploader = _make_uploader(tmp_path)
        session = MagicMock()
        session.post.return_value = MagicMock(json=lambda: {"code": 0, "data": {"season_id": 888}})
        uploader._get_authenticated_session = MagicMock(return_value=session)

        result = uploader.create_season("新合集", "简介")

        assert result == {"success": True, "season_id": 888}

    def test_create_season_returns_error_on_exception(self, tmp_path):
        uploader = _make_uploader(tmp_path)
        session = MagicMock()
        session.post.side_effect = RuntimeError("boom")
        uploader._get_authenticated_session = MagicMock(return_value=session)

        result = uploader.create_season("新合集", "简介")

        assert result["success"] is False
        assert "boom" in result["error"]


class TestBilibiliUploaderHelpers:
    def test_get_upload_limit_contains_expected_keys(self, tmp_path):
        uploader = _make_uploader(tmp_path)

        info = uploader.get_upload_limit()

        assert info["max_size"] > 0
        assert "supported_formats" in info

    def test_get_categories_contains_common_partition(self, tmp_path):
        uploader = _make_uploader(tmp_path)

        categories = uploader.get_categories()

        assert categories[160] == "生活"

    def test_create_bilibili_uploader_uses_config_values(self):
        config = SimpleNamespace(
            storage=SimpleNamespace(
                database_path="/tmp/test.db",
            ),
            uploader=SimpleNamespace(
                bilibili=SimpleNamespace(
                    cookies_file="cookies.json",
                    line="bda2",
                    threads=5,
                    upload_interval=75,
                )
            )
        )

        uploader = create_bilibili_uploader(config)

        assert isinstance(uploader, BilibiliUploader)
        assert str(uploader.cookies_file).endswith("cookies.json")
        assert uploader.line == "bda2"
        assert uploader.threads == 5
        assert uploader.lock_db_path == "/tmp/test.db"
        assert uploader.upload_interval == 75
