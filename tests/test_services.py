"""
services 模块单元测试

测试 PlaylistService 的进度统计（含 partial_completed）。
"""
import os
import tempfile
import pytest
from vat.database import Database
from vat.models import (
    Video, Task, Playlist, TaskStep, TaskStatus, SourceType,
    DEFAULT_STAGE_SEQUENCE,
)
from vat.services import PlaylistService


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    database = Database(path)
    yield database
    os.unlink(path)


def _setup_playlist(db, playlist_id="PL1"):
    db.add_playlist(Playlist(
        id=playlist_id, title="Test PL",
        source_url=f"https://youtube.com/playlist?list={playlist_id}",
    ))


def _add_pl_video(db, video_id, playlist_id="PL1", index=1, unavailable=False):
    metadata = {"unavailable": True} if unavailable else {}
    v = Video(id=video_id, source_type=SourceType.YOUTUBE,
              source_url=f"https://youtube.com/watch?v={video_id}",
              title=video_id, metadata=metadata)
    db.add_video(v)
    db.add_video_to_playlist(video_id, playlist_id, playlist_index=index)


def _complete_all(db, vid):
    for step in DEFAULT_STAGE_SEQUENCE:
        db.add_task(Task(video_id=vid, step=step, status=TaskStatus.COMPLETED))


def _complete_partial(db, vid, steps):
    for step in steps:
        db.add_task(Task(video_id=vid, step=step, status=TaskStatus.COMPLETED))


def _fail(db, vid, step):
    db.add_task(Task(video_id=vid, step=step, status=TaskStatus.FAILED))


class TestPlaylistProgress:
    """get_playlist_progress 进度统计"""

    def _setup(self, db):
        _setup_playlist(db)
        _add_pl_video(db, "v_comp", index=1)
        _add_pl_video(db, "v_part", index=2)
        _add_pl_video(db, "v_fail", index=3)
        _add_pl_video(db, "v_pend", index=4)
        _add_pl_video(db, "v_unavail", index=5, unavailable=True)

        _complete_all(db, "v_comp")
        _complete_partial(db, "v_part", [TaskStep.DOWNLOAD, TaskStep.WHISPER])
        _complete_partial(db, "v_fail", [TaskStep.DOWNLOAD])
        _fail(db, "v_fail", TaskStep.WHISPER)

    def test_total(self, db):
        self._setup(db)
        svc = PlaylistService(db)
        p = svc.get_playlist_progress("PL1")
        assert p['total'] == 5

    def test_completed(self, db):
        self._setup(db)
        p = PlaylistService(db).get_playlist_progress("PL1")
        assert p['completed'] == 1

    def test_partial_completed(self, db):
        self._setup(db)
        p = PlaylistService(db).get_playlist_progress("PL1")
        assert p['partial_completed'] == 1

    def test_failed(self, db):
        self._setup(db)
        p = PlaylistService(db).get_playlist_progress("PL1")
        assert p['failed'] == 1

    def test_unavailable(self, db):
        self._setup(db)
        p = PlaylistService(db).get_playlist_progress("PL1")
        assert p['unavailable'] == 1

    def test_pending_is_truly_unprocessed(self, db):
        """pending = processable - completed - partial_completed - failed"""
        self._setup(db)
        p = PlaylistService(db).get_playlist_progress("PL1")
        # processable=4, completed=1, partial=1, failed=1 -> pending=1 (v_pend)
        assert p['pending'] == 1

    def test_consistency(self, db):
        """processable = completed + partial_completed + failed + pending; total = processable + unavailable"""
        self._setup(db)
        p = PlaylistService(db).get_playlist_progress("PL1")
        processable = p['total'] - p['unavailable']
        assert processable == p['completed'] + p['partial_completed'] + p['failed'] + p['pending']

    def test_by_step_present(self, db):
        self._setup(db)
        p = PlaylistService(db).get_playlist_progress("PL1")
        assert 'by_step' in p
        for step in DEFAULT_STAGE_SEQUENCE:
            assert step.value in p['by_step']

    def test_by_step_download_counts(self, db):
        """by_step 中 download 的 completed/pending/failed 应与实际一致"""
        self._setup(db)
        p = PlaylistService(db).get_playlist_progress("PL1")
        dl = p['by_step']['download']
        # v_comp: download completed, v_part: download completed,
        # v_fail: download completed, v_pend: no task, v_unavail: skipped
        assert dl['completed'] == 3
        assert dl['failed'] == 0

    def test_by_step_whisper_counts(self, db):
        """by_step 中 whisper 应反映部分完成和失败"""
        self._setup(db)
        p = PlaylistService(db).get_playlist_progress("PL1")
        wh = p['by_step']['whisper']
        # v_comp: completed, v_part: completed, v_fail: failed
        assert wh['completed'] == 2
        assert wh['failed'] == 1


class TestGetPendingVideos:

    def test_excludes_completed(self, db):
        _setup_playlist(db)
        _add_pl_video(db, "v_done", index=1)
        _add_pl_video(db, "v_todo", index=2)
        _complete_all(db, "v_done")

        svc = PlaylistService(db)
        pending = svc.get_pending_videos("PL1")
        assert len(pending) == 1
        assert pending[0].id == "v_todo"

    def test_all_completed_returns_empty(self, db):
        _setup_playlist(db)
        _add_pl_video(db, "v1", index=1)
        _complete_all(db, "v1")
        assert PlaylistService(db).get_pending_videos("PL1") == []


class TestGetPlaylistVideosOrdering:

    def test_order_by_playlist_index(self, db):
        _setup_playlist(db)
        for i in [3, 1, 2]:
            _add_pl_video(db, f"v{i}", index=i)

        svc = PlaylistService(db)
        videos = svc.get_playlist_videos("PL1", order_by="playlist_index")
        indices = [v.playlist_index for v in videos]
        assert indices == [1, 2, 3]


class TestSyncPlaylistContracts:
    def test_sync_playlist_uses_explicit_target_playlist_id(self, db):
        service = PlaylistService(db)
        service._downloader = type(
            "FakeDownloader",
            (),
            {
                "get_playlist_info": lambda _self, _url: {
                    "id": "UC_raw",
                    "title": "Channel Videos",
                    "uploader": "Uploader",
                    "uploader_id": "channel-1",
                    "entries": [],
                }
            },
        )()

        result = service.sync_playlist(
            "https://www.youtube.com/@demo/videos",
            fetch_upload_dates=False,
            target_playlist_id="UC_raw-videos",
        )

        assert result.playlist_id == "UC_raw-videos"
        assert db.get_playlist("UC_raw-videos") is not None
        assert db.get_playlist("UC_raw") is None

    def test_sync_playlist_links_existing_video_without_duplicating_video_record(self, db):
        existing_video = Video(
            id="vid_existing",
            source_type=SourceType.YOUTUBE,
            source_url="https://www.youtube.com/watch?v=vid_existing",
            title="Existing Video",
            metadata={},
        )
        db.add_video(existing_video)

        service = PlaylistService(db)
        service._downloader = type(
            "FakeDownloader",
            (),
            {
                "get_playlist_info": lambda _self, _url: {
                    "id": "PL_SYNC",
                    "title": "Sync Target",
                    "uploader": "Uploader",
                    "uploader_id": "channel-2",
                    "entries": [
                        {"id": "vid_existing", "title": "Existing Video"},
                    ],
                }
            },
        )()

        result = service.sync_playlist(
            "https://youtube.com/playlist?list=PL_SYNC",
            fetch_upload_dates=False,
        )

        assert result.new_videos == ["vid_existing"]
        playlist_videos = db.get_playlist_video_ids("PL_SYNC")
        assert playlist_videos == {"vid_existing"}
        stored = db.get_video("vid_existing")
        assert stored is not None
        assert stored.title == "Existing Video"

    def test_sync_playlist_updates_playlist_index_for_existing_members(self, db):
        _setup_playlist(db, "PL_IDX")
        _add_pl_video(db, "vid1", playlist_id="PL_IDX", index=7)

        service = PlaylistService(db)
        service._downloader = type(
            "FakeDownloader",
            (),
            {
                "get_playlist_info": lambda _self, _url: {
                    "id": "PL_IDX",
                    "title": "Indexed Playlist",
                    "uploader": "Uploader",
                    "uploader_id": "channel-3",
                    "entries": [
                        {"id": "vid1", "title": "Video One"},
                    ],
                }
            },
        )()

        result = service.sync_playlist(
            "https://youtube.com/playlist?list=PL_IDX",
            fetch_upload_dates=False,
        )

        assert result.existing_videos == ["vid1"]
        pv_info = db.get_playlist_video_info("PL_IDX", "vid1")
        assert pv_info["playlist_index"] == 1


class TestPlaylistRecoveryHelpers:
    def test_calc_interpolated_date_between_newer_and_older(self, db):
        service = PlaylistService(db)

        date = service._calc_interpolated_date(
            2,
            {
                1: "20250110",  # 更新
                3: "20250106",  # 更旧
            },
        )

        assert date == "20250108"

    def test_calc_interpolated_date_for_oldest_video_uses_previous_day(self, db):
        service = PlaylistService(db)

        date = service._calc_interpolated_date(
            5,
            {
                2: "20250110",
            },
        )

        assert date == "20250109"

    def test_process_failed_fetches_marks_unavailable_with_interpolated_date(self, db):
        _setup_playlist(db, "PL_FAIL")
        _add_pl_video(db, "v_known_newer", playlist_id="PL_FAIL", index=1)
        _add_pl_video(db, "v_target", playlist_id="PL_FAIL", index=2)
        _add_pl_video(db, "v_known_older", playlist_id="PL_FAIL", index=3)

        db.update_video("v_known_newer", metadata={"upload_date": "20250110"})
        db.update_video("v_known_older", metadata={"upload_date": "20250106"})

        service = PlaylistService(db)
        messages = []

        class _Result:
            ok = False
            is_unavailable = True
            upload_date = None
            error_message = "video unavailable"

        service._process_failed_fetches(
            "PL_FAIL",
            [("v_target", _Result())],
            messages.append,
        )

        target = db.get_video("v_target")
        assert target.metadata["upload_date"] == "20250108"
        assert target.metadata["upload_date_interpolated"] is True
        assert target.metadata["unavailable"] is True
        assert target.metadata["unavailable_reason"] == "video unavailable"
        assert any("永久不可用" in msg for msg in messages)

    def test_process_failed_fetches_error_only_interpolates_without_unavailable(self, db):
        _setup_playlist(db, "PL_ERR")
        _add_pl_video(db, "v_known_newer", playlist_id="PL_ERR", index=1)
        _add_pl_video(db, "v_target", playlist_id="PL_ERR", index=2)
        _add_pl_video(db, "v_known_older", playlist_id="PL_ERR", index=3)

        db.update_video("v_known_newer", metadata={"upload_date": "20250110"})
        db.update_video("v_known_older", metadata={"upload_date": "20250106"})

        service = PlaylistService(db)
        messages = []

        class _Result:
            ok = False
            is_unavailable = False
            upload_date = None
            error_message = "temporary network issue"

        service._process_failed_fetches(
            "PL_ERR",
            [("v_target", _Result())],
            messages.append,
        )

        target = db.get_video("v_target")
        assert target.metadata["upload_date"] == "20250108"
        assert target.metadata["upload_date_interpolated"] is True
        assert "unavailable" not in target.metadata
        assert "unavailable_reason" not in target.metadata
        assert any("获取失败" in msg for msg in messages)
