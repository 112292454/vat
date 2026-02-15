"""
database.py 单元测试

测试数据库 CRUD、统计（含 partial_completed）、分页过滤、批量进度查询。
所有测试使用临时数据库，互不干扰。
"""
import os
import tempfile
import pytest
from vat.database import Database
from vat.models import (
    Video, Task, Playlist, TaskStep, TaskStatus, SourceType,
    DEFAULT_STAGE_SEQUENCE,
)


@pytest.fixture
def db():
    """创建临时数据库，测试结束后删除"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    database = Database(path)
    yield database
    os.unlink(path)


def _add_video(db, video_id, title=None):
    """辅助：添加一个视频"""
    v = Video(
        id=video_id,
        source_type=SourceType.YOUTUBE,
        source_url=f"https://youtube.com/watch?v={video_id}",
        title=title or video_id,
    )
    db.add_video(v)
    return v


def _complete_all_steps(db, video_id):
    """辅助：标记视频所有 7 个阶段为 completed"""
    for step in DEFAULT_STAGE_SEQUENCE:
        db.add_task(Task(video_id=video_id, step=step, status=TaskStatus.COMPLETED))


def _complete_partial_steps(db, video_id, steps):
    """辅助：标记部分阶段为 completed"""
    for step in steps:
        db.add_task(Task(video_id=video_id, step=step, status=TaskStatus.COMPLETED))


def _fail_step(db, video_id, step):
    """辅助：标记某个阶段为 failed"""
    db.add_task(Task(video_id=video_id, step=step, status=TaskStatus.FAILED,
                     error_message="test failure"))


def _run_step(db, video_id, step):
    """辅助：标记某个阶段为 running"""
    db.add_task(Task(video_id=video_id, step=step, status=TaskStatus.RUNNING))


# ==================== Video CRUD ====================

class TestVideoCRUD:

    def test_add_and_get(self, db):
        _add_video(db, "v1", "Test Video")
        loaded = db.get_video("v1")
        assert loaded is not None
        assert loaded.id == "v1"
        assert loaded.title == "Test Video"
        assert loaded.source_type == SourceType.YOUTUBE

    def test_get_nonexistent(self, db):
        assert db.get_video("nonexistent") is None

    def test_list_videos(self, db):
        for i in range(3):
            _add_video(db, f"v{i}")
        videos = db.list_videos()
        assert len(videos) == 3


# ==================== Task CRUD ====================

class TestTaskCRUD:

    def test_add_and_get_task(self, db):
        _add_video(db, "v1")
        db.add_task(Task(video_id="v1", step=TaskStep.DOWNLOAD, status=TaskStatus.COMPLETED))
        task = db.get_task("v1", TaskStep.DOWNLOAD)
        assert task is not None
        assert task.status == TaskStatus.COMPLETED

    def test_get_tasks_returns_all(self, db):
        _add_video(db, "v1")
        for step in [TaskStep.DOWNLOAD, TaskStep.WHISPER]:
            db.add_task(Task(video_id="v1", step=step, status=TaskStatus.PENDING))
        tasks = db.get_tasks("v1")
        assert len(tasks) == 2

    def test_update_task_status(self, db):
        _add_video(db, "v1")
        db.add_task(Task(video_id="v1", step=TaskStep.DOWNLOAD, status=TaskStatus.PENDING))
        db.update_task_status("v1", TaskStep.DOWNLOAD, TaskStatus.COMPLETED)
        task = db.get_task("v1", TaskStep.DOWNLOAD)
        assert task.status == TaskStatus.COMPLETED

    def test_get_pending_steps(self, db):
        _add_video(db, "v1")
        _complete_partial_steps(db, "v1", [TaskStep.DOWNLOAD, TaskStep.WHISPER, TaskStep.SPLIT])
        pending = db.get_pending_steps("v1")
        assert TaskStep.OPTIMIZE in pending
        assert TaskStep.TRANSLATE in pending
        assert TaskStep.EMBED in pending
        assert TaskStep.UPLOAD in pending
        assert TaskStep.DOWNLOAD not in pending

    def test_get_pending_steps_all_done(self, db):
        _add_video(db, "v1")
        _complete_all_steps(db, "v1")
        pending = db.get_pending_steps("v1")
        assert pending == []

    def test_delete_tasks_for_video(self, db):
        _add_video(db, "v1")
        for step in [TaskStep.DOWNLOAD, TaskStep.WHISPER]:
            db.add_task(Task(video_id="v1", step=step, status=TaskStatus.PENDING))
        deleted = db.delete_tasks_for_video("v1")
        assert deleted == 2
        assert db.get_tasks("v1") == []


# ==================== Playlist CRUD ====================

class TestPlaylistCRUD:

    def test_add_get_update_delete(self, db):
        pl = Playlist(id="PL1", title="Test", source_url="https://youtube.com/playlist?list=PL1",
                      video_count=10)
        db.add_playlist(pl)

        loaded = db.get_playlist("PL1")
        assert loaded.title == "Test"
        assert loaded.video_count == 10

        db.update_playlist("PL1", video_count=20)
        assert db.get_playlist("PL1").video_count == 20

        db.delete_playlist("PL1")
        assert db.get_playlist("PL1") is None

    def test_list_playlists(self, db):
        for i in range(3):
            db.add_playlist(Playlist(id=f"PL{i}", title=f"PL{i}",
                                     source_url=f"https://youtube.com/playlist?list=PL{i}"))
        assert len(db.list_playlists()) == 3

    def test_video_playlist_association(self, db):
        db.add_playlist(Playlist(id="PL1", title="PL1", source_url="https://example.com"))
        for i in [3, 1, 2]:
            _add_video(db, f"v{i}")
            db.add_video_to_playlist(f"v{i}", "PL1", playlist_index=i)
        videos = db.list_videos(playlist_id="PL1")
        assert len(videos) == 3
        indices = [v.playlist_index for v in videos]
        assert indices == sorted(indices), "应按 playlist_index 排序"


# ==================== Statistics ====================

class TestStatistics:
    """get_statistics() 测试，含 partial_completed"""

    def _setup_mixed_videos(self, db):
        """构造混合状态的视频集合：
        - v_complete: 全部完成
        - v_partial: 部分完成（download+whisper 完成，无失败）
        - v_failed: 有失败步骤
        - v_running: 有运行中步骤
        - v_pending: 无任何任务记录（纯待处理）
        """
        for vid in ["v_complete", "v_partial", "v_failed", "v_running", "v_pending"]:
            _add_video(db, vid)

        _complete_all_steps(db, "v_complete")
        _complete_partial_steps(db, "v_partial", [TaskStep.DOWNLOAD, TaskStep.WHISPER])
        _complete_partial_steps(db, "v_failed", [TaskStep.DOWNLOAD])
        _fail_step(db, "v_failed", TaskStep.WHISPER)
        _complete_partial_steps(db, "v_running", [TaskStep.DOWNLOAD])
        _run_step(db, "v_running", TaskStep.WHISPER)

    def test_total_videos(self, db):
        self._setup_mixed_videos(db)
        stats = db.get_statistics()
        assert stats['total_videos'] == 5

    def test_completed_videos(self, db):
        self._setup_mixed_videos(db)
        stats = db.get_statistics()
        assert stats['completed_videos'] == 1

    def test_partial_completed_videos(self, db):
        self._setup_mixed_videos(db)
        stats = db.get_statistics()
        assert stats['partial_completed_videos'] == 1

    def test_failed_videos(self, db):
        self._setup_mixed_videos(db)
        stats = db.get_statistics()
        assert stats['failed_videos'] == 1

    def test_running_videos(self, db):
        self._setup_mixed_videos(db)
        stats = db.get_statistics()
        assert stats['running_videos'] == 1

    def test_pending_not_counted_in_any_category(self, db):
        """纯待处理视频（无 task 记录）不计入任何统计"""
        self._setup_mixed_videos(db)
        stats = db.get_statistics()
        accounted = (stats['completed_videos'] + stats['partial_completed_videos']
                     + stats['failed_videos'] + stats['running_videos'])
        assert accounted == 4  # v_pending 不在统计中

    def test_empty_database(self, db):
        stats = db.get_statistics()
        assert stats['total_videos'] == 0
        assert stats['completed_videos'] == 0
        assert stats['partial_completed_videos'] == 0


# ==================== Pagination & Filters ====================

class TestListVideosPaginated:

    def _setup_filterable_videos(self, db):
        """构造 5 个视频用于过滤测试"""
        for vid in ["v_comp", "v_part", "v_fail", "v_run", "v_pend"]:
            _add_video(db, vid)
        _complete_all_steps(db, "v_comp")
        _complete_partial_steps(db, "v_part", [TaskStep.DOWNLOAD, TaskStep.WHISPER])
        _complete_partial_steps(db, "v_fail", [TaskStep.DOWNLOAD])
        _fail_step(db, "v_fail", TaskStep.WHISPER)
        _complete_partial_steps(db, "v_run", [TaskStep.DOWNLOAD])
        _run_step(db, "v_run", TaskStep.WHISPER)

    def test_no_filter_returns_all(self, db):
        self._setup_filterable_videos(db)
        result = db.list_videos_paginated(page=1, per_page=50)
        assert result['total'] == 5

    def test_filter_completed(self, db):
        self._setup_filterable_videos(db)
        result = db.list_videos_paginated(status='completed')
        assert result['total'] == 1
        assert result['videos'][0].id == "v_comp"

    def test_filter_partial_completed(self, db):
        self._setup_filterable_videos(db)
        result = db.list_videos_paginated(status='partial_completed')
        assert result['total'] == 1
        assert result['videos'][0].id == "v_part"

    def test_filter_failed(self, db):
        self._setup_filterable_videos(db)
        result = db.list_videos_paginated(status='failed')
        assert result['total'] == 1
        assert result['videos'][0].id == "v_fail"

    def test_filter_running(self, db):
        self._setup_filterable_videos(db)
        result = db.list_videos_paginated(status='running')
        assert result['total'] == 1
        assert result['videos'][0].id == "v_run"

    def test_pagination(self, db):
        for i in range(10):
            _add_video(db, f"v{i:02d}")
        result = db.list_videos_paginated(page=1, per_page=3)
        assert len(result['videos']) == 3
        assert result['total'] == 10
        assert result['total_pages'] == 4

    def test_search_by_title(self, db):
        _add_video(db, "v1", title="Python Tutorial")
        _add_video(db, "v2", title="Java Guide")
        result = db.list_videos_paginated(search="Python")
        assert result['total'] == 1
        assert result['videos'][0].id == "v1"


# ==================== Batch Progress ====================

class TestBatchGetVideoProgress:

    def test_completed_video(self, db):
        _add_video(db, "v1")
        _complete_all_steps(db, "v1")
        progress = db.batch_get_video_progress(["v1"])
        assert progress["v1"]["completed"] == 7
        assert progress["v1"]["progress"] == 100
        assert progress["v1"]["has_failed"] is False

    def test_partial_video(self, db):
        _add_video(db, "v1")
        _complete_partial_steps(db, "v1", [TaskStep.DOWNLOAD, TaskStep.WHISPER])
        progress = db.batch_get_video_progress(["v1"])
        assert progress["v1"]["completed"] == 2
        assert progress["v1"]["progress"] == 28  # 2/7*100 ≈ 28

    def test_failed_video_has_blocked_steps(self, db):
        """失败视频的后续步骤应标记为 blocked"""
        _add_video(db, "v1")
        _complete_partial_steps(db, "v1", [TaskStep.DOWNLOAD])
        _fail_step(db, "v1", TaskStep.WHISPER)
        progress = db.batch_get_video_progress(["v1"])
        assert progress["v1"]["has_failed"] is True
        # whisper 之后的步骤应为 blocked
        assert progress["v1"]["task_status"]["split"]["status"] == "blocked"
        assert progress["v1"]["task_status"]["optimize"]["status"] == "blocked"

    def test_no_tasks_video(self, db):
        """无任务记录的视频"""
        _add_video(db, "v1")
        progress = db.batch_get_video_progress(["v1"])
        assert progress["v1"]["completed"] == 0
        assert progress["v1"]["progress"] == 0
        # 所有步骤应为 pending
        for step in DEFAULT_STAGE_SEQUENCE:
            assert progress["v1"]["task_status"][step.value]["status"] == "pending"


# ==================== Batch Playlist Progress ====================

class TestBatchGetPlaylistProgress:

    def _setup_playlist_with_videos(self, db):
        """构造 playlist 包含不同状态的视频"""
        db.add_playlist(Playlist(id="PL1", title="Test", source_url="https://example.com"))
        videos = {
            "v_comp": "completed",
            "v_part": "partial",
            "v_fail": "failed",
            "v_pend": "pending",
            "v_unavail": "unavailable",
        }
        for i, (vid, _) in enumerate(videos.items()):
            v = Video(id=vid, source_type=SourceType.YOUTUBE,
                      source_url=f"https://youtube.com/watch?v={vid}", title=vid)
            if vid == "v_unavail":
                v.metadata = {"unavailable": True}
            db.add_video(v)
            db.add_video_to_playlist(vid, "PL1", playlist_index=i + 1)

        _complete_all_steps(db, "v_comp")
        _complete_partial_steps(db, "v_part", [TaskStep.DOWNLOAD, TaskStep.WHISPER])
        _complete_partial_steps(db, "v_fail", [TaskStep.DOWNLOAD])
        _fail_step(db, "v_fail", TaskStep.WHISPER)

    def test_progress_categories(self, db):
        self._setup_playlist_with_videos(db)
        progress = db.batch_get_playlist_progress()
        p = progress["PL1"]
        assert p['total'] == 5
        assert p['completed'] == 1
        assert p['partial_completed'] == 1
        assert p['failed'] == 1
        assert p['unavailable'] == 1
        assert p['pending'] == 2  # processable(4) - completed(1) - partial(1) = 2 (failed 不从 pending 中扣除)

    def test_consistency(self, db):
        """processable = completed + partial_completed + pending; total = processable + unavailable"""
        self._setup_playlist_with_videos(db)
        progress = db.batch_get_playlist_progress()
        p = progress["PL1"]
        processable = p['total'] - p['unavailable']
        assert processable == p['completed'] + p['partial_completed'] + p['pending']


# ==================== DB Version ====================

class TestDatabaseVersion:

    def test_fresh_db_has_version(self, db):
        from vat.database import DB_VERSION
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version FROM db_version LIMIT 1")
            row = cursor.fetchone()
            assert row['version'] == DB_VERSION

    def test_tables_exist(self, db):
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row['name'] for row in cursor.fetchall()}
            for expected in ['videos', 'tasks', 'playlists', 'playlist_videos', 'db_version']:
                assert expected in tables, f"表 {expected} 不存在"
