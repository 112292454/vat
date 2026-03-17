"""
services 模块单元测试

测试 PlaylistService 的进度统计（含 partial_completed）。
"""
import os
import tempfile
from types import SimpleNamespace
import pytest
from vat.database import Database
from vat.models import (
    Video, Task, Playlist, TaskStep, TaskStatus, SourceType,
    DEFAULT_STAGE_SEQUENCE,
)
from vat.services import PlaylistService
from vat.downloaders import VideoInfoResult


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
    def test_sync_playlist_raises_when_playlist_info_unavailable(self, db):
        service = PlaylistService(db)
        service._downloader = type(
            "FakeDownloader",
            (),
            {"get_playlist_info": lambda _self, _url: None},
        )()

        with pytest.raises(ValueError, match="无法获取 Playlist 信息"):
            service.sync_playlist("https://youtube.com/playlist?list=PL_MISSING", fetch_upload_dates=False)

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

    def test_sync_playlist_ignores_none_entries_and_entries_without_id(self, db):
        service = PlaylistService(db)
        service._downloader = type(
            "FakeDownloader",
            (),
            {
                "get_playlist_info": lambda _self, _url: {
                    "id": "PL_SKIP",
                    "title": "Skip Playlist",
                    "uploader": "Uploader",
                    "uploader_id": "channel-3",
                    "entries": [
                        None,
                        {"title": "No ID"},
                        {"id": "valid1", "title": "Valid"},
                    ],
                }
            },
        )()

        result = service.sync_playlist("https://youtube.com/playlist?list=PL_SKIP", fetch_upload_dates=False)

        assert result.new_videos == ["valid1"]
        assert db.get_video("valid1") is not None

    def test_sync_playlist_auto_add_videos_false_does_not_create_video_records(self, db):
        service = PlaylistService(db)
        service._downloader = type(
            "FakeDownloader",
            (),
            {
                "get_playlist_info": lambda _self, _url: {
                    "id": "PL_NOADD",
                    "title": "No Add",
                    "uploader": "Uploader",
                    "uploader_id": "channel-4",
                    "entries": [
                        {"id": "vid_noadd", "title": "Valid"},
                    ],
                }
            },
        )()

        result = service.sync_playlist(
            "https://youtube.com/playlist?list=PL_NOADD",
            auto_add_videos=False,
            fetch_upload_dates=False,
        )

        assert result.new_videos == ["vid_noadd"]
        assert db.get_video("vid_noadd") is None

    def test_sync_playlist_fetch_upload_dates_updates_existing_video_missing_date(self, db, monkeypatch):
        _setup_playlist(db, "PL_FETCH")
        _add_pl_video(db, "vid_existing", playlist_id="PL_FETCH", index=1)
        db.update_video("vid_existing", metadata={
            "unavailable": True,
            "unavailable_reason": "old reason",
            "upload_date_interpolated": True,
        })

        service = PlaylistService(db)
        service._downloader = type(
            "FakeDownloader",
            (),
            {
                "get_playlist_info": lambda _self, _url: {
                    "id": "PL_FETCH",
                    "title": "Fetch Playlist",
                    "uploader": "Uploader",
                    "uploader_id": "channel-1",
                    "entries": [{"id": "vid_existing", "title": "Existing"}],
                },
                "get_video_info": lambda _self, _url: VideoInfoResult(
                    status="ok",
                    info={
                        "id": "vid_existing",
                        "title": "Fetched Title",
                        "uploader": "Fetched Uploader",
                        "upload_date": "20250110",
                        "duration": 120,
                        "thumbnail": "thumb.jpg",
                        "view_count": 10,
                        "like_count": 2,
                    },
                ),
            },
        )()

        translate_calls = []
        monkeypatch.setattr(service, "_submit_translate_task", lambda vid, info, force=False: translate_calls.append((vid, info, force)))

        class _ImmediateFuture:
            def __init__(self, value):
                self._value = value

            def result(self, timeout=None):
                return self._value

        class _ImmediateExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, vid):
                return _ImmediateFuture(fn(vid))

        monkeypatch.setattr("vat.services.playlist_service.ThreadPoolExecutor", _ImmediateExecutor)

        result = service.sync_playlist("https://youtube.com/playlist?list=PL_FETCH", fetch_upload_dates=True)

        assert result.existing_videos == ["vid_existing"]
        video = db.get_video("vid_existing")
        assert video.metadata["upload_date"] == "20250110"
        assert video.metadata["duration"] == 120
        assert video.metadata["thumbnail"] == "thumb.jpg"
        assert "unavailable" not in video.metadata
        assert "unavailable_reason" not in video.metadata
        assert "upload_date_interpolated" not in video.metadata
        assert translate_calls and translate_calls[0][0] == "vid_existing"

    def test_sync_playlist_fetch_upload_dates_timeout_falls_back_to_interpolated_error_result(self, db, monkeypatch):
        _setup_playlist(db, "PL_TIMEOUT")
        _add_pl_video(db, "vid_newer", playlist_id="PL_TIMEOUT", index=1)
        _add_pl_video(db, "vid_target", playlist_id="PL_TIMEOUT", index=2)
        _add_pl_video(db, "vid_older", playlist_id="PL_TIMEOUT", index=3)
        db.update_video("vid_newer", metadata={"upload_date": "20250110"})
        db.update_video("vid_older", metadata={"upload_date": "20250106"})

        service = PlaylistService(db)
        service._downloader = type(
            "FakeDownloader",
            (),
            {
                "get_playlist_info": lambda _self, _url: {
                    "id": "PL_TIMEOUT",
                    "title": "Timeout Playlist",
                    "uploader": "Uploader",
                    "uploader_id": "channel-2",
                    "entries": [
                        {"id": "vid_newer", "title": "newer"},
                        {"id": "vid_target", "title": "target"},
                        {"id": "vid_older", "title": "older"},
                    ],
                },
                "get_video_info": lambda _self, _url: VideoInfoResult(
                    status="ok",
                    info={"id": _url.rsplit("=", 1)[-1], "upload_date": "20250101"},
                ),
            },
        )()
        monkeypatch.setattr(service, "_submit_translate_task", lambda *args, **kwargs: None)

        class _FutureOK:
            def __init__(self, value):
                self._value = value

            def result(self, timeout=None):
                return self._value

        class _FutureBoom:
            def result(self, timeout=None):
                raise TimeoutError("boom")

        class _Executor:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, vid):
                if vid == "vid_target":
                    return _FutureBoom()
                return _FutureOK(fn(vid))

        monkeypatch.setattr("vat.services.playlist_service.ThreadPoolExecutor", _Executor)

        result = service.sync_playlist("https://youtube.com/playlist?list=PL_TIMEOUT", fetch_upload_dates=True)

        assert result.existing_videos == ["vid_newer", "vid_target", "vid_older"]
        target = db.get_video("vid_target")
        assert target.metadata["upload_date"] == "20250108"
        assert target.metadata["upload_date_interpolated"] is True


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


class TestSubmitTranslateTaskContracts:
    def test_submit_translate_task_skips_when_translated_exists_and_not_forced(self, db, monkeypatch):
        _add_pl_video(db, "v_translated")
        db.update_video("v_translated", metadata={"translated": {"title_translated": "已有翻译"}})

        service = PlaylistService(db)

        submit_calls = []

        class _InlineExecutor:
            def submit(self, fn):
                submit_calls.append("submitted")
                fn()

        monkeypatch.setattr("vat.services.playlist_service._translate_executor", _InlineExecutor())
        monkeypatch.setattr(
            "vat.config.load_config",
            lambda: SimpleNamespace(
                llm=SimpleNamespace(is_available=lambda: True, model="model"),
                downloader=SimpleNamespace(
                    video_info_translate=SimpleNamespace(model="", api_key="", base_url="")
                ),
                get_stage_proxy=lambda _stage: "",
            ),
        )
        monkeypatch.setattr(
            "vat.llm.video_info_translator.VideoInfoTranslator",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not instantiate translator")),
        )

        service._submit_translate_task(
            "v_translated",
            {"title": "原始标题", "uploader": "频道", "description": "", "tags": []},
            force=False,
        )

        assert submit_calls == ["submitted"]
        assert db.get_video("v_translated").metadata["translated"] == {"title_translated": "已有翻译"}

    def test_submit_translate_task_force_retranslates_and_updates_metadata(self, db, monkeypatch):
        _add_pl_video(db, "v_force")
        db.update_video("v_force", metadata={"translated": {"title_translated": "旧翻译"}})

        service = PlaylistService(db)

        class _InlineExecutor:
            def submit(self, fn):
                fn()

        class _TranslatedInfo:
            recommended_tid_name = "日常"

            def to_dict(self):
                return {
                    "title_translated": "新翻译标题",
                    "description_translated": "新翻译简介",
                    "description_summary": "摘要",
                    "tags_translated": ["标签A"],
                    "tags_generated": ["标签B"],
                    "recommended_tid": 21,
                    "recommended_tid_name": "日常",
                    "tid_reason": "test",
                    "original_title": "原始标题",
                    "original_description": "",
                    "original_tags": [],
                }

        created = {}

        class _FakeTranslator:
            def __init__(self, **kwargs):
                created["kwargs"] = kwargs

            def translate(self, **kwargs):
                created["translate_kwargs"] = kwargs
                return _TranslatedInfo()

        monkeypatch.setattr("vat.services.playlist_service._translate_executor", _InlineExecutor())
        monkeypatch.setattr(
            "vat.config.load_config",
            lambda: SimpleNamespace(
                llm=SimpleNamespace(is_available=lambda: True, model="fallback-model"),
                downloader=SimpleNamespace(
                    video_info_translate=SimpleNamespace(model="vit-model", api_key="k", base_url="u")
                ),
                get_stage_proxy=lambda _stage: "http://proxy:7890",
            ),
        )
        monkeypatch.setattr("vat.llm.video_info_translator.VideoInfoTranslator", _FakeTranslator)

        service._submit_translate_task(
            "v_force",
            {
                "id": "v_force",
                "webpage_url": "https://www.youtube.com/watch?v=v_force",
                "title": "原始标题",
                "uploader": "频道",
                "description": "简介",
                "tags": ["tag1"],
                "duration": 120,
                "upload_date": "20250101",
                "thumbnail": "thumb",
                "width": 1920,
                "height": 1080,
            },
            force=True,
        )

        updated = db.get_video("v_force")
        assert updated.title == "原始标题"
        assert updated.metadata["translated"]["title_translated"] == "新翻译标题"
        assert updated.metadata["_video_info"]["title"] == "原始标题"
        assert created["kwargs"]["model"] == "vit-model"
        assert created["kwargs"]["proxy"] == "http://proxy:7890"
        assert created["translate_kwargs"]["uploader"] == "频道"


class TestPlaylistVideoSelectionContracts:
    def test_get_playlist_videos_order_by_upload_date_places_missing_dates_last(self, db):
        _setup_playlist(db, "PL_DATE")
        _add_pl_video(db, "v_old", playlist_id="PL_DATE", index=2)
        _add_pl_video(db, "v_none", playlist_id="PL_DATE", index=1)
        _add_pl_video(db, "v_new", playlist_id="PL_DATE", index=3)
        db.update_video("v_old", metadata={"upload_date": "20240101"})
        db.update_video("v_new", metadata={"upload_date": "20240201"})

        videos = PlaylistService(db).get_playlist_videos("PL_DATE", order_by="upload_date")

        assert [v.id for v in videos] == ["v_old", "v_new", "v_none"]

    def test_get_pending_videos_filters_specific_target_step(self, db):
        _setup_playlist(db, "PL_STEP")
        _add_pl_video(db, "v_download_done", playlist_id="PL_STEP", index=1)
        _add_pl_video(db, "v_whisper_done", playlist_id="PL_STEP", index=2)
        db.add_task(Task(video_id="v_download_done", step=TaskStep.DOWNLOAD, status=TaskStatus.COMPLETED))
        db.add_task(Task(video_id="v_whisper_done", step=TaskStep.DOWNLOAD, status=TaskStatus.COMPLETED))
        db.add_task(Task(video_id="v_whisper_done", step=TaskStep.WHISPER, status=TaskStatus.COMPLETED))

        pending = PlaylistService(db).get_pending_videos("PL_STEP", target_step="whisper")

        assert [v.id for v in pending] == ["v_download_done"]

    def test_get_completed_videos_only_returns_fully_completed(self, db):
        _setup_playlist(db, "PL_DONE")
        _add_pl_video(db, "v_done", playlist_id="PL_DONE", index=1)
        _add_pl_video(db, "v_partial", playlist_id="PL_DONE", index=2)
        _complete_all(db, "v_done")
        _complete_partial(db, "v_partial", [TaskStep.DOWNLOAD])

        completed = PlaylistService(db).get_completed_videos("PL_DONE")

        assert [v.id for v in completed] == ["v_done"]


class TestDeletePlaylistContracts:
    def test_delete_playlist_without_videos_only_removes_playlist(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        outdir = tempfile.mkdtemp()
        try:
            db = Database(path, output_base_dir=outdir)
            _setup_playlist(db, "PL_DEL")
            _add_pl_video(db, "v_keep", playlist_id="PL_DEL", index=1)
            service = PlaylistService(db)

            result = service.delete_playlist("PL_DEL", delete_videos=False)

            assert result == {"deleted_videos": 0}
            assert db.get_playlist("PL_DEL") is None
            assert db.get_video("v_keep") is not None
        finally:
            os.unlink(path)
            import shutil
            shutil.rmtree(outdir)


class TestRefreshVideosContracts:
    def _make_service(self, db, monkeypatch, video_info_by_id):
        service = PlaylistService(db)
        service._downloader = type(
            "FakeDownloader",
            (),
            {
                "get_video_info": lambda _self, url: video_info_by_id[url.rsplit("=", 1)[-1]],
            },
        )()
        translate_calls = []
        monkeypatch.setattr(
            service,
            "_submit_translate_task",
            lambda video_id, video_info, force=False: translate_calls.append(
                {"video_id": video_id, "video_info": video_info, "force": force}
            ),
        )
        return service, translate_calls

    def test_refresh_videos_merge_mode_only_fills_missing_fields(self, db, monkeypatch):
        _setup_playlist(db, "PL_REFRESH")
        _add_pl_video(db, "v_missing", playlist_id="PL_REFRESH", index=1)
        db.update_video("v_missing", metadata={"thumbnail": "", "translated": {"title_translated": "已有"}})

        service, translate_calls = self._make_service(
            db,
            monkeypatch,
            {
                "v_missing": {
                    "id": "v_missing",
                    "webpage_url": "https://www.youtube.com/watch?v=v_missing",
                    "title": "标题A",
                    "uploader": "频道A",
                    "description": "简介A",
                    "duration": 120,
                    "upload_date": "20250101",
                    "thumbnail": "thumbA",
                    "tags": ["tagA"],
                    "width": 1920,
                    "height": 1080,
                    "view_count": 99,
                    "like_count": 7,
                }
            },
        )

        result = service.refresh_videos("PL_REFRESH")

        assert result == {"refreshed": 1, "skipped": 0, "failed": 0}
        video = db.get_video("v_missing")
        assert video.title == "v_missing"
        assert video.metadata["thumbnail"] == "thumbA"
        assert video.metadata["upload_date"] == "20250101"
        assert video.metadata["translated"] == {"title_translated": "已有"}
        assert "_video_info" in video.metadata
        assert video.metadata["_video_info"]["title"] == "标题A"
        assert translate_calls == []

    def test_refresh_videos_merge_mode_auto_submits_translate_when_missing(self, db, monkeypatch):
        _setup_playlist(db, "PL_MERGE_TRANSLATE")
        _add_pl_video(db, "v_need_translate", playlist_id="PL_MERGE_TRANSLATE", index=1)

        service, translate_calls = self._make_service(
            db,
            monkeypatch,
            {
                "v_need_translate": {
                    "id": "v_need_translate",
                    "webpage_url": "https://www.youtube.com/watch?v=v_need_translate",
                    "title": "标题B",
                    "uploader": "频道B",
                    "description": "简介B",
                    "duration": 100,
                    "upload_date": "20250102",
                    "thumbnail": "thumbB",
                    "tags": ["tagB"],
                    "width": 1280,
                    "height": 720,
                    "view_count": 88,
                    "like_count": 6,
                }
            },
        )

        result = service.refresh_videos("PL_MERGE_TRANSLATE")

        assert result == {"refreshed": 1, "skipped": 0, "failed": 0}
        assert translate_calls == [{
            "video_id": "v_need_translate",
            "video_info": {
                "id": "v_need_translate",
                "webpage_url": "https://www.youtube.com/watch?v=v_need_translate",
                "title": "标题B",
                "uploader": "频道B",
                "description": "简介B",
                "duration": 100,
                "upload_date": "20250102",
                "thumbnail": "thumbB",
                "tags": ["tagB"],
                "width": 1280,
                "height": 720,
                "view_count": 88,
                "like_count": 6,
            },
            "force": False,
        }]

    def test_refresh_videos_force_refetch_preserves_translated_without_force_retranslate(self, db, monkeypatch):
        _setup_playlist(db, "PL_FORCE")
        _add_pl_video(db, "v_force", playlist_id="PL_FORCE", index=1)
        db.update_video("v_force", metadata={
            "translated": {"title_translated": "旧翻译"},
            "upload_date_interpolated": True,
        })

        service, translate_calls = self._make_service(
            db,
            monkeypatch,
            {
                "v_force": {
                    "id": "v_force",
                    "webpage_url": "https://www.youtube.com/watch?v=v_force",
                    "title": "标题C",
                    "uploader": "频道C",
                    "description": "简介C",
                    "duration": 90,
                    "upload_date": "20250103",
                    "thumbnail": "thumbC",
                    "tags": ["tagC"],
                    "width": 1920,
                    "height": 1080,
                    "view_count": 66,
                    "like_count": 5,
                }
            },
        )

        result = service.refresh_videos("PL_FORCE", force_refetch=True, force_retranslate=False)

        assert result == {"refreshed": 1, "skipped": 0, "failed": 0}
        video = db.get_video("v_force")
        assert video.metadata["translated"] == {"title_translated": "旧翻译"}
        assert "upload_date_interpolated" not in video.metadata
        assert translate_calls == []

    def test_refresh_videos_force_refetch_and_force_retranslate_resubmits_translation(self, db, monkeypatch):
        _setup_playlist(db, "PL_FORCE_RETRANS")
        _add_pl_video(db, "v_force_retrans", playlist_id="PL_FORCE_RETRANS", index=1)
        db.update_video("v_force_retrans", metadata={"translated": {"title_translated": "旧翻译"}})

        service, translate_calls = self._make_service(
            db,
            monkeypatch,
            {
                "v_force_retrans": {
                    "id": "v_force_retrans",
                    "webpage_url": "https://www.youtube.com/watch?v=v_force_retrans",
                    "title": "标题D",
                    "uploader": "频道D",
                    "description": "简介D",
                    "duration": 80,
                    "upload_date": "20250104",
                    "thumbnail": "thumbD",
                    "tags": ["tagD"],
                    "width": 1920,
                    "height": 1080,
                    "view_count": 55,
                    "like_count": 4,
                }
            },
        )

        result = service.refresh_videos("PL_FORCE_RETRANS", force_refetch=True, force_retranslate=True)

        assert result == {"refreshed": 1, "skipped": 0, "failed": 0}
        assert translate_calls == [{
            "video_id": "v_force_retrans",
            "video_info": {
                "id": "v_force_retrans",
                "webpage_url": "https://www.youtube.com/watch?v=v_force_retrans",
                "title": "标题D",
                "uploader": "频道D",
                "description": "简介D",
                "duration": 80,
                "upload_date": "20250104",
                "thumbnail": "thumbD",
                "tags": ["tagD"],
                "width": 1920,
                "height": 1080,
                "view_count": 55,
                "like_count": 4,
            },
            "force": True,
        }]

    def test_refresh_videos_skips_unavailable_and_counts_failures(self, db, monkeypatch):
        _setup_playlist(db, "PL_FAIL_CASE")
        _add_pl_video(db, "v_unavailable", playlist_id="PL_FAIL_CASE", index=1, unavailable=True)
        _add_pl_video(db, "v_ok", playlist_id="PL_FAIL_CASE", index=2)
        _add_pl_video(db, "v_fail", playlist_id="PL_FAIL_CASE", index=3)

        def _get_video_info(video_id):
            if video_id == "v_fail":
                raise RuntimeError("network error")
            return {
                "id": video_id,
                "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
                "title": f"标题-{video_id}",
                "uploader": "频道",
                "description": "简介",
                "duration": 70,
                "upload_date": "20250105",
                "thumbnail": "thumb",
                "tags": [],
                "width": 1920,
                "height": 1080,
                "view_count": 1,
                "like_count": 1,
            }

        service = PlaylistService(db)
        service._downloader = type(
            "FakeDownloader",
            (),
            {"get_video_info": lambda _self, url: _get_video_info(url.rsplit("=", 1)[-1])},
        )()
        translate_calls = []
        monkeypatch.setattr(
            service,
            "_submit_translate_task",
            lambda video_id, video_info, force=False: translate_calls.append(video_id),
        )

        result = service.refresh_videos("PL_FAIL_CASE", force_refetch=True)

        assert result == {"refreshed": 1, "skipped": 1, "failed": 1}
        assert translate_calls == ["v_ok"]


class TestRetranslateVideosContracts:
    def test_retranslate_videos_skips_unavailable_and_missing_source_info(self, db, monkeypatch):
        _setup_playlist(db, "PL_RETRANS")
        _add_pl_video(db, "v_unavail", playlist_id="PL_RETRANS", index=1, unavailable=True)
        _add_pl_video(db, "v_meta", playlist_id="PL_RETRANS", index=2)
        _add_pl_video(db, "v_skip", playlist_id="PL_RETRANS", index=3)
        db.update_video("v_meta", metadata={
            "_video_info": {"title": "标题", "description": "简介", "tags": [], "uploader": "频道"}
        })
        db.update_video("v_skip", title="", metadata={"description": "", "uploader": ""})

        service = PlaylistService(db)
        calls = []
        monkeypatch.setattr(
            service,
            "_submit_translate_task",
            lambda video_id, video_info, force=False: calls.append((video_id, video_info, force)),
        )

        result = service.retranslate_videos("PL_RETRANS")

        assert result == {"submitted": 1, "skipped": 2}
        assert calls == [("v_meta", {"title": "标题", "description": "简介", "tags": [], "uploader": "频道"}, True)]

    def test_retranslate_videos_falls_back_to_metadata_when_video_info_cache_missing(self, db, monkeypatch):
        _setup_playlist(db, "PL_RETRANS_META")
        _add_pl_video(db, "v_meta_fallback", playlist_id="PL_RETRANS_META", index=1)
        db.update_video("v_meta_fallback", metadata={"description": "简介", "tags": ["t1"], "uploader": "频道"})

        service = PlaylistService(db)
        calls = []
        monkeypatch.setattr(
            service,
            "_submit_translate_task",
            lambda video_id, video_info, force=False: calls.append((video_id, video_info, force)),
        )

        result = service.retranslate_videos("PL_RETRANS_META")

        assert result == {"submitted": 1, "skipped": 0}
        assert calls == [(
            "v_meta_fallback",
            {"title": "v_meta_fallback", "description": "简介", "tags": ["t1"], "uploader": "频道"},
            True,
        )]


class TestDownloaderPropertyContracts:
    def test_downloader_uses_injected_config_without_loading_global_config(self, db, monkeypatch):
        fake_config = SimpleNamespace(
            get_stage_proxy=lambda stage: "http://proxy:8000",
            downloader=SimpleNamespace(
                youtube=SimpleNamespace(
                    format="best",
                    cookies_file="cookies.json",
                    remote_components={"translate": True},
                )
            ),
        )
        service = PlaylistService(db, config=fake_config)
        created = {}

        class _FakeDownloader:
            def __init__(self, **kwargs):
                created["kwargs"] = kwargs

        monkeypatch.setattr("vat.services.playlist_service.YouTubeDownloader", _FakeDownloader)
        monkeypatch.setattr(
            "vat.config.load_config",
            lambda: (_ for _ in ()).throw(AssertionError("should not load config")),
        )

        downloader = service.downloader

        assert isinstance(downloader, _FakeDownloader)
        assert created["kwargs"] == {
            "proxy": "http://proxy:8000",
            "video_format": "best",
            "cookies_file": "cookies.json",
            "remote_components": {"translate": True},
        }

    def test_downloader_loads_global_config_when_missing(self, db, monkeypatch):
        fake_config = SimpleNamespace(
            get_stage_proxy=lambda stage: "",
            downloader=SimpleNamespace(
                youtube=SimpleNamespace(
                    format="bestvideo",
                    cookies_file="cookie2.json",
                    remote_components={},
                )
            ),
        )
        service = PlaylistService(db, config=None)
        created = {}

        class _FakeDownloader:
            def __init__(self, **kwargs):
                created["kwargs"] = kwargs

        monkeypatch.setattr("vat.services.playlist_service.YouTubeDownloader", _FakeDownloader)
        monkeypatch.setattr("vat.config.load_config", lambda: fake_config)

        downloader = service.downloader

        assert isinstance(downloader, _FakeDownloader)
        assert created["kwargs"]["video_format"] == "bestvideo"
        assert created["kwargs"]["cookies_file"] == "cookie2.json"

    def test_delete_playlist_with_videos_removes_records_and_runs_file_cleanup(self, monkeypatch):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        outdir = tempfile.mkdtemp()
        try:
            db = Database(path, output_base_dir=outdir)
            _setup_playlist(db, "PL_PURGE")
            _add_pl_video(db, "v1", playlist_id="PL_PURGE", index=1)
            _add_pl_video(db, "v2", playlist_id="PL_PURGE", index=2)
            os.makedirs(os.path.join(outdir, "v1"), exist_ok=True)
            os.makedirs(os.path.join(outdir, "v2"), exist_ok=True)

            deleted_dirs = []
            monkeypatch.setattr(
                "vat.utils.file_ops.delete_processed_files",
                lambda path: deleted_dirs.append(str(path)),
            )

            service = PlaylistService(db)
            result = service.delete_playlist("PL_PURGE", delete_videos=True)

            assert result == {"deleted_videos": 2}
            assert db.get_playlist("PL_PURGE") is None
            assert db.get_video("v1") is None
            assert db.get_video("v2") is None
            assert deleted_dirs == [os.path.join(outdir, "v1"), os.path.join(outdir, "v2")]
        finally:
            os.unlink(path)
            import shutil
            shutil.rmtree(outdir)
