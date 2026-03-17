"""Bilibili 其余 API wrapper 契约测试。"""

from pathlib import Path
from unittest.mock import MagicMock

from vat.uploaders.bilibili import BilibiliUploader


def _make_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    return resp


def _make_uploader():
    uploader = BilibiliUploader.__new__(BilibiliUploader)
    uploader.cookie_data = {"bili_jct": "csrf-token"}
    uploader._get_authenticated_session = MagicMock()
    return uploader


class TestDeleteVideoContracts:
    def test_delete_video_returns_true_on_success(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.post.return_value = _make_response({"code": 0})
        uploader._get_authenticated_session.return_value = session

        assert uploader.delete_video(12345) is True

    def test_delete_video_returns_false_on_api_failure(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.post.return_value = _make_response({"code": -1, "message": "failed"})
        uploader._get_authenticated_session.return_value = session

        assert uploader.delete_video(12345) is False

    def test_delete_video_returns_false_on_exception(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.post.side_effect = RuntimeError("boom")
        uploader._get_authenticated_session.return_value = session

        assert uploader.delete_video(12345) is False


class TestMyVideosContracts:
    def test_get_my_videos_parses_archive_list(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.return_value = _make_response({
            "code": 0,
            "data": {
                "page": {"count": 2},
                "arc_audits": [
                    {"Archive": {"aid": 1, "bvid": "BV1", "title": "A", "state": 0, "state_desc": "正常"}},
                    {"Archive": {"aid": 2, "bvid": "BV2", "title": "B", "state": -30, "state_desc": "退回"}},
                ],
            },
        })
        uploader._get_authenticated_session.return_value = session

        result = uploader.get_my_videos(page=2, page_size=5)

        assert result == {
            "total": 2,
            "videos": [
                {"aid": 1, "bvid": "BV1", "title": "A", "state": 0, "state_desc": "正常"},
                {"aid": 2, "bvid": "BV2", "title": "B", "state": -30, "state_desc": "退回"},
            ],
        }

    def test_get_my_videos_returns_none_on_api_failure(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.return_value = _make_response({"code": -1, "message": "failed"})
        uploader._get_authenticated_session.return_value = session

        assert uploader.get_my_videos() is None

    def test_get_my_videos_returns_none_on_exception(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.side_effect = RuntimeError("boom")
        uploader._get_authenticated_session.return_value = session

        assert uploader.get_my_videos() is None


class TestVideoDetailContracts:
    def test_get_video_detail_returns_data_on_success(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.return_value = _make_response({"code": 0, "data": {"aid": 123, "title": "标题"}})
        uploader._get_authenticated_session.return_value = session

        assert uploader.get_video_detail(123) == {"aid": 123, "title": "标题"}

    def test_get_video_detail_returns_none_on_failure(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.return_value = _make_response({"code": -1, "message": "failed"})
        uploader._get_authenticated_session.return_value = session

        assert uploader.get_video_detail(123) is None

    def test_get_video_detail_returns_none_on_exception(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.side_effect = RuntimeError("boom")
        uploader._get_authenticated_session.return_value = session

        assert uploader.get_video_detail(123) is None


class TestArchiveDetailContracts:
    def test_get_archive_detail_returns_data_on_success(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.return_value = _make_response({"code": 0, "data": {"archive": {"aid": 123}, "videos": []}})
        uploader._get_authenticated_session.return_value = session

        assert uploader.get_archive_detail(123) == {"archive": {"aid": 123}, "videos": []}

    def test_get_archive_detail_returns_none_on_failure(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.return_value = _make_response({"code": -1, "message": "failed"})
        uploader._get_authenticated_session.return_value = session

        assert uploader.get_archive_detail(123) is None

    def test_get_archive_detail_returns_none_on_exception(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.side_effect = RuntimeError("boom")
        uploader._get_authenticated_session.return_value = session

        assert uploader.get_archive_detail(123) is None


class TestFullDescContracts:
    def test_get_full_desc_returns_desc_on_success(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.return_value = _make_response({"code": 0, "data": {"desc": "完整简介"}})

        assert uploader._get_full_desc(123, session=session) == "完整简介"

    def test_get_full_desc_returns_none_on_failure(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.return_value = _make_response({"code": -1})

        assert uploader._get_full_desc(123, session=session) is None

    def test_get_full_desc_returns_none_on_exception(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.side_effect = RuntimeError("boom")

        assert uploader._get_full_desc(123, session=session) is None


class TestBvidToAidContracts:
    def test_bvid_to_aid_returns_aid_on_success(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.return_value = _make_response({"code": 0, "data": {"aid": 12345}})
        uploader._get_authenticated_session.return_value = session

        assert uploader.bvid_to_aid("BVxxxx") == 12345

    def test_bvid_to_aid_returns_none_on_failure(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.return_value = _make_response({"code": -1, "message": "failed"})
        uploader._get_authenticated_session.return_value = session

        assert uploader.bvid_to_aid("BVxxxx") is None

    def test_bvid_to_aid_returns_none_on_exception(self):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.side_effect = RuntimeError("boom")
        uploader._get_authenticated_session.return_value = session

        assert uploader.bvid_to_aid("BVxxxx") is None


class TestDownloadVideoContracts:
    def test_download_video_returns_false_when_playurl_has_no_dash(self, monkeypatch, tmp_path):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.side_effect = [
            _make_response({"code": 0, "data": {"pages": [{"cid": 999}]}}),
            _make_response({"code": 0, "data": {}}),
        ]
        uploader._get_authenticated_session.return_value = session
        uploader.get_archive_detail = MagicMock(return_value=None)
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ffmpeg should not run")))

        assert uploader.download_video(12345, tmp_path / "out.mp4") is False

    def test_download_video_returns_false_when_dash_streams_empty(self, monkeypatch, tmp_path):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.side_effect = [
            _make_response({"code": 0, "data": {"pages": [{"cid": 999}]}}),
            _make_response({"code": 0, "data": {"dash": {"video": [], "audio": []}}}),
        ]
        uploader._get_authenticated_session.return_value = session
        uploader.get_archive_detail = MagicMock(return_value=None)
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ffmpeg should not run")))

        assert uploader.download_video(12345, tmp_path / "out.mp4") is False

    def test_download_video_returns_false_when_ffmpeg_fails(self, monkeypatch, tmp_path):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.side_effect = [
            _make_response({"code": 0, "data": {"pages": [{"cid": 999}]}}),
            _make_response({
                "code": 0,
                "data": {
                    "dash": {
                        "video": [{"bandwidth": 2, "baseUrl": "video", "codecs": "h264", "width": 1920, "height": 1080}],
                        "audio": [{"bandwidth": 1, "baseUrl": "audio", "codecs": "aac"}],
                    }
                },
            }),
        ]
        uploader._get_authenticated_session.return_value = session
        uploader.get_archive_detail = MagicMock(return_value=None)
        monkeypatch.setattr(
            "subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="ffmpeg failed"),
        )

        assert uploader.download_video(12345, tmp_path / "out.mp4") is False

    def test_download_video_returns_false_when_ffmpeg_succeeds_without_output_file(self, monkeypatch, tmp_path):
        uploader = _make_uploader()
        session = MagicMock()
        session.get.side_effect = [
            _make_response({"code": 0, "data": {"pages": [{"cid": 999}]}}),
            _make_response({
                "code": 0,
                "data": {
                    "dash": {
                        "video": [{"bandwidth": 2, "baseUrl": "video", "codecs": "h264", "width": 1920, "height": 1080}],
                        "audio": [{"bandwidth": 1, "baseUrl": "audio", "codecs": "aac"}],
                    }
                },
            }),
        ]
        uploader._get_authenticated_session.return_value = session
        uploader.get_archive_detail = MagicMock(return_value=None)
        monkeypatch.setattr(
            "subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""),
        )

        assert uploader.download_video(12345, tmp_path / "out.mp4") is False
