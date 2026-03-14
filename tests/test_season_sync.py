"""season_sync 契约测试。"""

import os
import tempfile

import pytest

from vat.database import Database
from vat.models import Playlist, SourceType, Task, TaskStatus, TaskStep, Video
from vat.uploaders.bilibili import season_sync


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(path)
    yield database
    os.unlink(path)


def _add_playlist_video(db, video_id, *, playlist_id="PL1", title=None, metadata=None):
    if db.get_playlist(playlist_id) is None:
        db.add_playlist(
            Playlist(
                id=playlist_id,
                title=f"Playlist {playlist_id}",
                source_url=f"https://youtube.com/playlist?list={playlist_id}",
            )
        )
    db.add_video(
        Video(
            id=video_id,
            source_type=SourceType.YOUTUBE,
            source_url=f"https://youtube.com/watch?v={video_id}",
            title=title or video_id,
            metadata=metadata or {},
        )
    )
    db.add_video_to_playlist(video_id, playlist_id, playlist_index=1)


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class _FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        return _FakeResponse(self.payload)


class _FakeUploader:
    def __init__(self, *, add_results=None, season_map=None, archive_payload=None, auto_sort_result=True):
        self.add_results = list(add_results or [])
        self.season_map = dict(season_map or {})
        self.archive_payload = archive_payload or {"code": 0, "data": {}}
        self.auto_sort_result = auto_sort_result
        self.add_calls = []
        self.sort_calls = []
        self.session = _FakeSession(self.archive_payload)

    def add_to_season(self, aid, season_id):
        self.add_calls.append((aid, season_id))
        if self.add_results:
            result = self.add_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return True

    def get_season_episodes(self, season_id):
        return self.season_map.get(season_id)

    def auto_sort_season(self, season_id):
        self.sort_calls.append(season_id)
        return self.auto_sort_result

    def _get_authenticated_session(self):
        return self.session


class TestSeasonSyncContracts:
    def test_successful_pending_video_marks_db_and_sorts_season(self, db, monkeypatch):
        _add_playlist_video(
            db,
            "v1",
            metadata={
                "bilibili_aid": 123,
                "bilibili_target_season_id": 42,
                "bilibili_season_added": False,
            },
        )
        uploader = _FakeUploader(
            add_results=[True],
            season_map={42: {"episodes": [{"aid": 123}]}},
        )
        monkeypatch.setattr("vat.uploaders.bilibili.time.sleep", lambda _seconds: None)

        result = season_sync(db, uploader, "PL1")

        assert result["total"] == 1
        assert result["success"] == 1
        assert result["failed"] == 0
        assert result["season_ids"] == {42}
        assert uploader.add_calls == [(123, 42)]
        assert uploader.sort_calls == [42]
        assert db.get_video("v1").metadata["bilibili_season_added"] is True

    def test_upload_completed_without_aid_is_reported_even_when_nothing_pending(self, db):
        _add_playlist_video(
            db,
            "v_no_aid",
            metadata={"bilibili_target_season_id": 42},
        )
        db.add_task(Task(video_id="v_no_aid", step=TaskStep.UPLOAD, status=TaskStatus.COMPLETED))
        uploader = _FakeUploader()

        result = season_sync(db, uploader, "PL1")

        assert result["total"] == 0
        assert result["success"] == 0
        assert result["diagnostics"]["upload_completed_no_aid"] == [("v_no_aid", "v_no_aid")]
        assert uploader.add_calls == []
        assert uploader.sort_calls == []

    def test_failed_add_reports_aid_not_found_diagnostic(self, db, monkeypatch):
        _add_playlist_video(
            db,
            "v_missing",
            metadata={
                "bilibili_aid": 456,
                "bilibili_target_season_id": 77,
                "bilibili_season_added": False,
            },
        )
        uploader = _FakeUploader(
            add_results=[False],
            season_map={77: {"episodes": []}},
            archive_payload={"code": -404, "message": "not found"},
        )
        monkeypatch.setattr("vat.uploaders.bilibili.time.sleep", lambda _seconds: None)

        result = season_sync(db, uploader, "PL1")

        assert result["failed"] == 1
        assert result["failed_videos"] == ["v_missing"]
        assert result["diagnostics"]["aid_not_found_on_bilibili"] == [("v_missing", 456, "v_missing")]
        assert db.get_video("v_missing").metadata["bilibili_season_added"] is False
        assert uploader.sort_calls == [77]

    def test_desync_repair_failure_resets_db_flag(self, db, monkeypatch):
        _add_playlist_video(
            db,
            "v_desync",
            metadata={
                "bilibili_aid": 789,
                "bilibili_target_season_id": 88,
                "bilibili_season_added": True,
            },
        )
        uploader = _FakeUploader(
            add_results=[False],
            season_map={88: {"episodes": []}},
        )
        monkeypatch.setattr("vat.uploaders.bilibili.time.sleep", lambda _seconds: None)

        result = season_sync(db, uploader, "PL1")

        assert result["total"] == 0
        assert result["diagnostics"]["desync_failed"] == 1
        assert result["diagnostics"].get("desync_fixed", 0) == 0
        assert db.get_video("v_desync").metadata["bilibili_season_added"] is False
        assert uploader.add_calls == [(789, 88)]
        assert uploader.sort_calls == [88]
