"""
Watch 服务单元测试

测试 vat.services.watch_service.WatchService 的核心逻辑：
- Session 创建与管理
- 视频筛选（processable / retry candidates）
- Round 记录
- 冲突 session 检测
- once 模式
"""

import json
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import pytest

from vat.services.watch_service import WatchService
from vat.models import TaskStep, TaskStatus, DEFAULT_STAGE_SEQUENCE


# ========== 测试用 Fixture ==========

@dataclass
class FakeVideo:
    """测试用简化 Video 对象"""
    id: str
    title: str = ""
    source_type: str = "youtube"
    source_url: str = ""
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FakePlaylist:
    """测试用简化 Playlist 对象"""
    id: str
    title: str = "Test Playlist"
    source_url: str = "https://www.youtube.com/playlist?list=PLtest"


@dataclass
class FakeSyncResult:
    """测试用 SyncResult"""
    new_videos: List[str] = field(default_factory=list)
    total_videos: int = 0


@pytest.fixture
def tmp_db(tmp_path):
    """创建带 watch 表的临时数据库"""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    
    # 创建 watch 表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watch_sessions (
            session_id TEXT PRIMARY KEY,
            playlist_ids TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'running',
            pid INTEGER,
            config TEXT,
            started_at TIMESTAMP NOT NULL,
            last_check_at TIMESTAMP,
            next_check_at TIMESTAMP,
            total_rounds INTEGER DEFAULT 0,
            total_new_found INTEGER DEFAULT 0,
            total_jobs_submitted INTEGER DEFAULT 0,
            error TEXT,
            stopped_at TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watch_rounds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            round_number INTEGER NOT NULL,
            playlist_id TEXT NOT NULL,
            started_at TIMESTAMP NOT NULL,
            finished_at TIMESTAMP,
            new_videos_found INTEGER DEFAULT 0,
            jobs_submitted INTEGER DEFAULT 0,
            submitted_video_ids TEXT,
            submitted_job_ids TEXT,
            retry_video_ids TEXT,
            error TEXT,
            FOREIGN KEY (session_id) REFERENCES watch_sessions(session_id)
        )
    """)
    
    # 创建 tasks 表（供视频筛选查询）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            step TEXT NOT NULL,
            status TEXT NOT NULL,
            gpu_id INTEGER,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            error_message TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def mock_config():
    """创建 mock Config"""
    config = MagicMock()
    config.watch.default_interval = 60
    config.watch.default_stages = "all"
    config.watch.max_new_videos_per_round = 0
    config.watch.default_concurrency = 1
    config.watch.max_retries = 3
    config.storage.work_dir = "/tmp/vat_test"
    return config


@pytest.fixture
def mock_db(tmp_db):
    """创建 mock Database（使用真实 SQLite 但 mock 方法签名）"""
    db = MagicMock()
    db.db_path = Path(tmp_db)
    
    # get_connection 返回真实的 SQLite 连接
    from contextlib import contextmanager
    
    @contextmanager
    def get_connection():
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    db.get_connection = get_connection
    return db


@pytest.fixture(autouse=True)
def mock_job_manager():
    """自动 mock process submitter builder，使 WatchService 不依赖真实 web.jobs。
    
    所有测试自动获得此 fixture。需要检查 submit_job 调用的测试可显式声明参数名获取 mock 实例。
    """
    submitter = MagicMock()
    submitter.return_value = "test-job-id"
    with patch('vat.services.watch_service.build_process_job_submitter', return_value=submitter):
        yield submitter


def _insert_task(db_path: str, video_id: str, step: str, status: str):
    """辅助：向 tasks 表插入记录"""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO tasks (video_id, step, status) VALUES (?, ?, ?)",
        (video_id, step, status)
    )
    conn.commit()
    conn.close()


# ========== Session 管理测试 ==========

class TestSessionManagement:
    
    def test_create_session(self, mock_config, mock_db):
        """测试 session 创建"""
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"], once=True,
        )
        service._session_id = "test-123"
        service._create_session()
        
        with mock_db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM watch_sessions WHERE session_id = 'test-123'"
            ).fetchone()
            assert row is not None
            assert row['status'] == 'running'
            assert row['pid'] == os.getpid()
            assert json.loads(row['playlist_ids']) == ["PL_test"]
    
    def test_finalize_session(self, mock_config, mock_db):
        """测试 session 结束时状态更新"""
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"], once=True,
        )
        service._session_id = "test-fin"
        service._create_session()
        service._finalize_session()
        
        with mock_db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM watch_sessions WHERE session_id = 'test-fin'"
            ).fetchone()
            assert row['status'] == 'stopped'
            assert row['stopped_at'] is not None
    
    def test_session_error_preserved(self, mock_config, mock_db):
        """测试 error 状态在 finalize 时不被覆盖"""
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"], once=True,
        )
        service._session_id = "test-err"
        service._create_session()
        service._update_session_error("some error")
        service._finalize_session()
        
        with mock_db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM watch_sessions WHERE session_id = 'test-err'"
            ).fetchone()
            # error 状态应保留，不被 finalize 覆盖为 stopped
            assert row['status'] == 'error'
            assert row['error'] == 'some error'
    
    def test_detect_dead_session(self, mock_config, mock_db, tmp_db):
        """测试检测到死亡 session 时自动清理"""
        # 插入一个"死进程"的 running session
        conn = sqlite3.connect(tmp_db)
        conn.execute("""
            INSERT INTO watch_sessions (session_id, playlist_ids, status, pid, started_at)
            VALUES ('dead-sess', '["PL_test"]', 'running', 99999999, ?)
        """, (datetime.now().isoformat(),))
        conn.commit()
        conn.close()
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"], once=True,
        )
        # 不应抛异常，应自动清理死亡 session
        service._check_existing_sessions()
        
        with mock_db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM watch_sessions WHERE session_id = 'dead-sess'"
            ).fetchone()
            assert row['status'] == 'stopped'
    
    def test_detect_alive_session_raises(self, mock_config, mock_db, tmp_db):
        """测试检测到存活 session 时抛出错误"""
        # 插入一个当前进程的 running session（模拟另一个存活的 watch）
        conn = sqlite3.connect(tmp_db)
        conn.execute("""
            INSERT INTO watch_sessions (session_id, playlist_ids, status, pid, started_at)
            VALUES ('alive-sess', '["PL_test"]', 'running', ?, ?)
        """, (os.getpid(), datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"], once=True,
        )
        with pytest.raises(RuntimeError, match="已被 session"):
            service._check_existing_sessions()


# ========== 视频筛选测试 ==========

class TestVideoFiltering:
    
    def test_new_video_is_processable(self, mock_config, mock_db, tmp_db):
        """没有任何 task 的视频应该可以处理"""
        mock_db.get_video = MagicMock(return_value=FakeVideo(id="vid_new"))
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"],
        )
        result = service._get_processable_videos(["vid_new"])
        assert "vid_new" in result
    
    def test_completed_video_excluded(self, mock_config, mock_db, tmp_db):
        """所有阶段都 completed 的视频应被排除"""
        mock_db.get_video = MagicMock(return_value=FakeVideo(id="vid_done"))
        
        # 为所有阶段插入 completed task
        for step in DEFAULT_STAGE_SEQUENCE:
            _insert_task(tmp_db, "vid_done", step.value, TaskStatus.COMPLETED.value)
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"],
        )
        result = service._get_processable_videos(["vid_done"])
        assert "vid_done" not in result
    
    def test_running_video_excluded(self, mock_config, mock_db, tmp_db):
        """有 running task 的视频应被排除"""
        mock_db.get_video = MagicMock(return_value=FakeVideo(id="vid_run"))
        _insert_task(tmp_db, "vid_run", "download", TaskStatus.RUNNING.value)
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"],
        )
        result = service._get_processable_videos(["vid_run"])
        assert "vid_run" not in result
    
    def test_unavailable_video_excluded(self, mock_config, mock_db, tmp_db):
        """unavailable 视频应被排除"""
        mock_db.get_video = MagicMock(
            return_value=FakeVideo(id="vid_unavail", metadata={"unavailable": True})
        )
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"],
        )
        result = service._get_processable_videos(["vid_unavail"])
        assert "vid_unavail" not in result
    
    def test_failed_video_is_retry_candidate(self, mock_config, mock_db, tmp_db):
        """有 failed task 的视频应是重试候选"""
        mock_db.get_video = MagicMock(return_value=FakeVideo(id="vid_fail"))
        _insert_task(tmp_db, "vid_fail", "download", TaskStatus.COMPLETED.value)
        _insert_task(tmp_db, "vid_fail", "whisper", TaskStatus.FAILED.value)
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"],
        )
        result = service._get_retry_candidates(["vid_fail"])
        assert "vid_fail" in result
    
    def test_retry_count_exceeded(self, mock_config, mock_db, tmp_db):
        """超过最大重试次数的视频不再重试"""
        mock_db.get_video = MagicMock(return_value=FakeVideo(id="vid_exhaust"))
        _insert_task(tmp_db, "vid_exhaust", "download", TaskStatus.FAILED.value)
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"],
        )
        # 预设重试次数已达上限
        service._retry_counts["vid_exhaust"] = 3
        
        result = service._get_retry_candidates(["vid_exhaust"])
        assert "vid_exhaust" not in result


# ========== Round 记录测试 ==========

class TestRoundRecording:
    
    def test_record_round(self, mock_config, mock_db, tmp_db):
        """测试 round 记录"""
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"],
        )
        service._session_id = "sess-round"
        service._round_number = 1
        
        service._record_round(
            playlist_id="PL_test",
            started_at=datetime.now(),
            new_videos_found=3,
            jobs_submitted=1,
            submitted_video_ids=["v1", "v2", "v3"],
            submitted_job_ids=["job-1"],
        )
        
        with mock_db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM watch_rounds WHERE session_id = 'sess-round'"
            ).fetchone()
            assert row is not None
            assert row['new_videos_found'] == 3
            assert row['jobs_submitted'] == 1
            assert json.loads(row['submitted_video_ids']) == ["v1", "v2", "v3"]
    
    def test_record_round_error(self, mock_config, mock_db, tmp_db):
        """测试 round 错误记录"""
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"],
        )
        service._session_id = "sess-err"
        service._round_number = 1
        
        service._record_round_error("PL_test", "sync failed")
        
        with mock_db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM watch_rounds WHERE session_id = 'sess-err'"
            ).fetchone()
            assert row is not None
            assert row['error'] == "sync failed"


# ========== 集成测试（mock 外部调用） ==========

class TestWatchOnceMode:
    
    def test_once_mode_no_new_videos(self, mock_job_manager, mock_config, mock_db, tmp_db):
        """once 模式：无新视频时正常退出"""
        # Mock PlaylistService
        mock_pl_service = MagicMock()
        mock_pl_service.get_playlist.return_value = FakePlaylist(id="PL_test")
        mock_pl_service.sync_playlist.return_value = FakeSyncResult(
            new_videos=[], total_videos=5
        )
        mock_pl_service.get_playlist_videos.return_value = []
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"], once=True,
        )
        service.playlist_service = mock_pl_service
        
        service.run()
        
        # 验证 session 被正确记录
        with mock_db.get_connection() as conn:
            sessions = conn.execute("SELECT * FROM watch_sessions").fetchall()
            assert len(sessions) == 1
            assert sessions[0]['status'] == 'stopped'
        
        # 没有提交任何任务
        mock_job_manager.assert_not_called()
    
    def test_once_mode_with_new_videos(self, mock_job_manager, mock_config, mock_db, tmp_db):
        """once 模式：有新视频时提交任务"""
        mock_config.storage.work_dir = str(Path(tmp_db).parent)
        
        mock_pl_service = MagicMock()
        mock_pl_service.get_playlist.return_value = FakePlaylist(id="PL_test")
        mock_pl_service.sync_playlist.return_value = FakeSyncResult(
            new_videos=["vid_a", "vid_b"], total_videos=5
        )
        mock_pl_service.get_playlist_videos.return_value = [
            FakeVideo(id="vid_a"), FakeVideo(id="vid_b")
        ]
        
        # mock get_video 返回有效视频
        mock_db.get_video = MagicMock(side_effect=lambda vid: FakeVideo(id=vid))
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"], once=True,
        )
        service.playlist_service = mock_pl_service
        
        service.run()
        
        # 验证通过 JobManager 提交了 process job
        mock_job_manager.assert_called_once()
        call_kwargs = mock_job_manager.call_args.kwargs
        assert set(call_kwargs['video_ids']) == {"vid_a", "vid_b"}
        
        # 验证 session 统计
        with mock_db.get_connection() as conn:
            sess = conn.execute("SELECT * FROM watch_sessions").fetchone()
            assert sess['total_rounds'] == 1
            assert sess['total_new_found'] == 2
            assert sess['total_jobs_submitted'] == 1
    
    def test_once_mode_multiple_playlists(self, mock_job_manager, mock_config, mock_db, tmp_db):
        """once 模式：多个 playlist 各自独立处理"""
        mock_config.storage.work_dir = str(Path(tmp_db).parent)
        
        mock_pl_service = MagicMock()
        # 两个 playlist 各自返回不同的新视频
        mock_pl_service.get_playlist.side_effect = lambda pid: FakePlaylist(id=pid, title=f"PL {pid}")
        
        def sync_side_effect(**kwargs):
            target = kwargs.get('target_playlist_id', '')
            if target == "PL_A":
                return FakeSyncResult(new_videos=["vA1"], total_videos=3)
            else:
                return FakeSyncResult(new_videos=["vB1", "vB2"], total_videos=5)
        
        mock_pl_service.sync_playlist.side_effect = sync_side_effect
        mock_pl_service.get_playlist_videos.side_effect = lambda pid: (
            [FakeVideo(id="vA1")] if pid == "PL_A" else [FakeVideo(id="vB1"), FakeVideo(id="vB2")]
        )
        
        mock_db.get_video = MagicMock(side_effect=lambda vid: FakeVideo(id=vid))
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_A", "PL_B"], once=True,
        )
        service.playlist_service = mock_pl_service
        
        service.run()
        
        # 两个 playlist 各提交一个任务
        assert mock_job_manager.call_count == 2
        
        # 验证 round 记录数
        with mock_db.get_connection() as conn:
            rounds = conn.execute("SELECT * FROM watch_rounds").fetchall()
            assert len(rounds) == 2
            pl_ids = {r['playlist_id'] for r in rounds}
            assert pl_ids == {"PL_A", "PL_B"}
    
    def test_force_mode_reprocesses_completed(self, mock_job_manager, mock_config, mock_db, tmp_db):
        """force 模式下 sync 报告为新增的已完成视频也会被重处理
        
        关键设计：force 只影响 sync 报告的新视频是否跳过已完成状态，
        不会导致全量 playlist 历史视频被重处理（那是 vat process -f 的职责）。
        """
        mock_config.storage.work_dir = str(Path(tmp_db).parent)
        
        # 插入全部已完成的 task
        for step in DEFAULT_STAGE_SEQUENCE:
            _insert_task(tmp_db, "vid_done", step.value, TaskStatus.COMPLETED.value)
        
        mock_pl_service = MagicMock()
        mock_pl_service.get_playlist.return_value = FakePlaylist(id="PL_test")
        # sync 报告 vid_done 为新增视频（可能是重建 playlist 关联）
        mock_pl_service.sync_playlist.return_value = FakeSyncResult(new_videos=["vid_done"], total_videos=1)
        
        mock_db.get_video = MagicMock(return_value=FakeVideo(id="vid_done"))
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"], once=True,
            force=True,
        )
        service.playlist_service = mock_pl_service
        
        service.run()
        
        # force=True 使 sync 报告的已完成视频也被提交
        mock_job_manager.assert_called_once()
    
    def test_no_new_videos_no_submission(self, mock_job_manager, mock_config, mock_db, tmp_db):
        """sync 无新视频时，即使 playlist 有未完成视频也不提交（防止全量重处理）"""
        mock_config.storage.work_dir = str(Path(tmp_db).parent)
        
        mock_pl_service = MagicMock()
        mock_pl_service.get_playlist.return_value = FakePlaylist(id="PL_test")
        # sync 无新增视频
        mock_pl_service.sync_playlist.return_value = FakeSyncResult(new_videos=[], total_videos=100)
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"], once=True,
        )
        service.playlist_service = mock_pl_service
        
        service.run()
        
        # 无新视频 → 不提交任何任务（即使 playlist 有大量未完成视频）
        mock_job_manager.assert_not_called()
    
    def test_max_new_per_round_limit(self, mock_job_manager, mock_config, mock_db, tmp_db):
        """max_new_videos_per_round 限制每轮提交数量"""
        mock_config.storage.work_dir = str(Path(tmp_db).parent)
        mock_config.watch.max_new_videos_per_round = 2  # 限制每轮最多 2 个
        
        # 返回 5 个新视频
        all_vids = [FakeVideo(id=f"v{i}") for i in range(5)]
        
        mock_pl_service = MagicMock()
        mock_pl_service.get_playlist.return_value = FakePlaylist(id="PL_test")
        mock_pl_service.sync_playlist.return_value = FakeSyncResult(
            new_videos=[f"v{i}" for i in range(5)], total_videos=5
        )
        mock_pl_service.get_playlist_videos.return_value = all_vids
        
        mock_db.get_video = MagicMock(side_effect=lambda vid: FakeVideo(id=vid))
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"], once=True,
        )
        service.playlist_service = mock_pl_service
        
        service.run()
        
        # 验证 submit_job 调用中的 video_ids 被截断为 2 个
        call_kwargs = mock_job_manager.call_args.kwargs
        assert len(call_kwargs['video_ids']) == 2
    
    def test_sync_failure_continues(self, mock_job_manager, mock_config, mock_db, tmp_db):
        """playlist sync 失败时记录错误并继续"""
        mock_pl_service = MagicMock()
        mock_pl_service.get_playlist.return_value = FakePlaylist(id="PL_test")
        mock_pl_service.sync_playlist.side_effect = Exception("Network error")
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"], once=True,
        )
        service.playlist_service = mock_pl_service
        
        # 不应抛出异常
        service.run()
        
        # session 正常结束
        with mock_db.get_connection() as conn:
            sess = conn.execute("SELECT * FROM watch_sessions").fetchone()
            assert sess['status'] == 'stopped'
            
            # round 记录了错误
            rd = conn.execute("SELECT * FROM watch_rounds").fetchone()
            assert rd is not None
            assert "Network error" in rd['error']
        
        # 没有提交任何任务
        mock_job_manager.assert_not_called()
    
    def test_mixed_new_and_retry_videos(self, mock_job_manager, mock_config, mock_db, tmp_db):
        """新视频 + 本 session 提交过的失败视频在第二轮混合处理
        
        关键设计：retry 只重试本 session 之前提交过的视频，不扫描全量 playlist 历史。
        首轮只处理 sync 的新视频；失败后在下一轮作为 retry 出现。
        """
        mock_config.storage.work_dir = str(Path(tmp_db).parent)
        
        # vid_fail 有失败的 task（只有在本 session 提交过才会被 retry）
        _insert_task(tmp_db, "vid_fail", "download", TaskStatus.COMPLETED.value)
        _insert_task(tmp_db, "vid_fail", "whisper", TaskStatus.FAILED.value)
        
        mock_pl_service = MagicMock()
        mock_pl_service.get_playlist.return_value = FakePlaylist(id="PL_test")
        mock_pl_service.sync_playlist.return_value = FakeSyncResult(
            new_videos=["vid_new"], total_videos=3
        )
        
        mock_db.get_video = MagicMock(side_effect=lambda vid: FakeVideo(id=vid))
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"], once=True,
        )
        service.playlist_service = mock_pl_service
        
        # 模拟 vid_fail 在之前的轮次中被本 session 提交过
        service._session_submitted_ids.add("vid_fail")
        
        service.run()
        
        # 验证提交了任务（包含新视频+重试视频）
        mock_job_manager.assert_called_once()
        video_ids = mock_job_manager.call_args.kwargs['video_ids']
        assert "vid_new" in video_ids
        assert "vid_fail" in video_ids
    
    def test_submit_job_has_correct_params(self, mock_job_manager, mock_config, mock_db, tmp_db):
        """验证 submit_job 传递了正确的参数"""
        mock_config.storage.work_dir = str(Path(tmp_db).parent)
        
        mock_pl_service = MagicMock()
        mock_pl_service.get_playlist.return_value = FakePlaylist(id="PL_test")
        mock_pl_service.sync_playlist.return_value = FakeSyncResult(
            new_videos=["v1"], total_videos=1
        )
        
        mock_db.get_video = MagicMock(return_value=FakeVideo(id="v1"))
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"], once=True,
            stages="download,whisper",
            gpu_device="cuda:1",
            concurrency=2,
            force=True,
            fail_fast=True,
        )
        service.playlist_service = mock_pl_service
        
        service.run()
        
        mock_job_manager.assert_called_once()
        kw = mock_job_manager.call_args.kwargs
        assert kw['video_ids'] == ["v1"]
        assert kw['steps'] == ["download", "whisper"]
        assert kw['gpu_device'] == "cuda:1"
        assert kw['concurrency'] == 2
        assert kw['force'] is True
        assert kw['fail_fast'] is True
        assert kw['playlist_id'] == "PL_test"


class TestEdgeCases:
    """边界场景测试"""
    
    def test_submit_exception_returns_no_job(self, mock_job_manager, mock_config, mock_db, tmp_db):
        """JobManager.submit_job 抛异常时，任务提交失败但不崩溃"""
        mock_config.storage.work_dir = str(Path(tmp_db).parent)
        
        mock_pl_service = MagicMock()
        mock_pl_service.get_playlist.return_value = FakePlaylist(id="PL_test")
        mock_pl_service.sync_playlist.return_value = FakeSyncResult(
            new_videos=["v1"], total_videos=1
        )
        
        mock_db.get_video = MagicMock(return_value=FakeVideo(id="v1"))
        mock_job_manager.side_effect = Exception("DB connection failed")
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"], once=True,
        )
        service.playlist_service = mock_pl_service
        
        # 不应抛异常
        service.run()
        
        # 验证 round 记录了 0 个 jobs_submitted
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        rounds = conn.execute("SELECT * FROM watch_rounds").fetchall()
        conn.close()
        assert any(r['jobs_submitted'] == 0 for r in rounds)
    
    def test_playlist_not_found_skips(self, mock_job_manager, mock_config, mock_db, tmp_db):
        """playlist 不存在时跳过，不报错"""
        mock_pl_service = MagicMock()
        mock_pl_service.get_playlist.return_value = None  # 不存在
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_nonexistent"], once=True,
        )
        service.playlist_service = mock_pl_service
        
        service.run()
        
        # 不应调用 sync
        mock_pl_service.sync_playlist.assert_not_called()
        mock_job_manager.assert_not_called()
    
    def test_brand_new_video_no_tasks(self, mock_job_manager, mock_config, mock_db, tmp_db):
        """全新视频（DB 中无任何 task 记录）应被视为可处理"""
        mock_config.storage.work_dir = str(Path(tmp_db).parent)
        
        # 不插入任何 task 记录
        mock_pl_service = MagicMock()
        mock_pl_service.get_playlist.return_value = FakePlaylist(id="PL_test")
        mock_pl_service.sync_playlist.return_value = FakeSyncResult(
            new_videos=["brand_new"], total_videos=1
        )
        
        mock_db.get_video = MagicMock(return_value=FakeVideo(id="brand_new"))
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"], once=True,
        )
        service.playlist_service = mock_pl_service
        
        service.run()
        
        # 应提交任务
        mock_job_manager.assert_called_once()
        video_ids = mock_job_manager.call_args.kwargs['video_ids']
        assert "brand_new" in video_ids
    
    def test_force_overrides_existing_session(self, mock_config, mock_db, tmp_db):
        """force=True 时覆盖已有 running session（PID 已死的情况）"""
        # 插入一个 "running" 但 PID 已死的 session
        conn = sqlite3.connect(tmp_db)
        conn.execute("""
            INSERT INTO watch_sessions (session_id, playlist_ids, status, pid, config, started_at)
            VALUES (?, ?, 'running', ?, '{}', ?)
        """, ("old_sess", '["PL_test"]', 99999, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        mock_pl_service = MagicMock()
        mock_pl_service.get_playlist.return_value = FakePlaylist(id="PL_test")
        mock_pl_service.sync_playlist.return_value = FakeSyncResult(
            new_videos=[], total_videos=0
        )
        mock_pl_service.get_playlist_videos.return_value = []
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_test"], once=True,
        )
        service.playlist_service = mock_pl_service
        
        # 不应抛异常（PID 已死，会自动清理）
        service.run()
        
        # 验证 old_sess 被标记为 stopped
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        old = conn.execute(
            "SELECT status FROM watch_sessions WHERE session_id = 'old_sess'"
        ).fetchone()
        conn.close()
        assert old['status'] == 'stopped'


# ========== 全流程集成测试（端到端 mock 验证） ==========

class TestFullLifecycle:
    """
    全流程集成测试：模拟真实场景下 Watch 的完整工作流，
    验证 fake playlist 数据从同步→筛选→提交→记录的全链路正确性。
    """
    
    def test_mixed_video_states_precise_filtering(self, mock_job_manager, mock_config, mock_db, tmp_db):
        """
        场景：Playlist 有 6 个视频，各处于不同状态：
        - v_new1, v_new2: 全新视频（无任何 task 记录）→ 应处理
        - v_done: 所有阶段 completed → 应排除
        - v_running: 有 running task → 应排除
        - v_failed: 有 failed task → 应进入重试列表
        - v_unavail: metadata 标记 unavailable → 应排除
        
        验证：提交的 video_ids 只包含 v_new1, v_new2, v_failed
        """
        mock_config.storage.work_dir = str(Path(tmp_db).parent)
        
        # 准备 tasks 数据
        for step in DEFAULT_STAGE_SEQUENCE:
            _insert_task(tmp_db, "v_done", step.value, TaskStatus.COMPLETED.value)
        _insert_task(tmp_db, "v_running", TaskStep.DOWNLOAD.value, TaskStatus.RUNNING.value)
        _insert_task(tmp_db, "v_failed", TaskStep.DOWNLOAD.value, TaskStatus.COMPLETED.value)
        _insert_task(tmp_db, "v_failed", TaskStep.WHISPER.value, TaskStatus.FAILED.value)
        
        all_videos = [
            FakeVideo(id="v_new1"),
            FakeVideo(id="v_new2"),
            FakeVideo(id="v_done"),
            FakeVideo(id="v_running"),
            FakeVideo(id="v_failed"),
            FakeVideo(id="v_unavail", metadata={'unavailable': True}),
        ]
        
        mock_pl_service = MagicMock()
        mock_pl_service.get_playlist.return_value = FakePlaylist(id="PL_mixed")
        mock_pl_service.sync_playlist.return_value = FakeSyncResult(
            new_videos=["v_new1", "v_new2"], total_videos=6
        )
        mock_pl_service.get_playlist_videos.return_value = all_videos
        
        mock_db.get_video = MagicMock(
            side_effect=lambda vid: next((v for v in all_videos if v.id == vid), None)
        )
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_mixed"], once=True,
        )
        service.playlist_service = mock_pl_service
        
        # 模拟 v_failed 在之前的轮次中被本 session 提交过
        service._session_submitted_ids.add("v_failed")
        
        service.run()
        
        # 验证 submit_job 调用的 video_ids 包含正确的视频
        mock_job_manager.assert_called_once()
        video_ids = mock_job_manager.call_args.kwargs['video_ids']
        
        assert "v_new1" in video_ids, "全新视频 v_new1 应被提交"
        assert "v_new2" in video_ids, "全新视频 v_new2 应被提交"
        assert "v_failed" in video_ids, "失败视频 v_failed 应被重试（本 session 提交过）"
        assert "v_done" not in video_ids, "已完成视频 v_done 不应被提交"
        assert "v_running" not in video_ids, "运行中视频 v_running 不应被提交"
        assert "v_unavail" not in video_ids, "不可用视频 v_unavail 不应被提交"
        
        # 验证 round 记录
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        rd = conn.execute("SELECT * FROM watch_rounds").fetchone()
        conn.close()
        
        assert rd['new_videos_found'] == 2, "应记录 2 个新视频（不含重试）"
        assert rd['jobs_submitted'] == 1
        submitted_ids = json.loads(rd['submitted_video_ids'])
        assert set(submitted_ids) == {"v_new1", "v_new2", "v_failed"}
        retry_ids = json.loads(rd['retry_video_ids'])
        assert retry_ids == ["v_failed"]
    
    def test_multi_round_retry_flow(self, mock_job_manager, mock_config, mock_db, tmp_db):
        """
        场景：模拟两轮 watch，验证重试机制：
        
        第 1 轮：playlist 有 v1, v2（全新） → 提交处理
        第 2 轮：v1 完成，v2 失败，v3 新增 → 提交 v2(重试) + v3(新)
        
        通过两次调用 once 模式模拟两轮，中间修改 DB 状态。
        """
        mock_config.storage.work_dir = str(Path(tmp_db).parent)
        
        # ====== 第 1 轮 ======
        mock_pl_service = MagicMock()
        mock_pl_service.get_playlist.return_value = FakePlaylist(id="PL_retry")
        mock_pl_service.sync_playlist.return_value = FakeSyncResult(
            new_videos=["v1", "v2"], total_videos=2
        )
        mock_pl_service.get_playlist_videos.return_value = [
            FakeVideo(id="v1"), FakeVideo(id="v2")
        ]
        mock_db.get_video = MagicMock(side_effect=lambda vid: FakeVideo(id=vid))
        
        service1 = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_retry"], once=True,
        )
        service1.playlist_service = mock_pl_service
        service1.run()
        
        # 验证第 1 轮提交了 v1, v2
        assert mock_job_manager.call_count == 1
        vid_ids_r1 = mock_job_manager.call_args.kwargs['video_ids']
        assert "v1" in vid_ids_r1 and "v2" in vid_ids_r1
        
        # ====== 模拟任务执行结果 ======
        # v1: 所有阶段完成
        for step in DEFAULT_STAGE_SEQUENCE:
            _insert_task(tmp_db, "v1", step.value, TaskStatus.COMPLETED.value)
        # v2: download 完成，whisper 失败
        _insert_task(tmp_db, "v2", TaskStep.DOWNLOAD.value, TaskStatus.COMPLETED.value)
        _insert_task(tmp_db, "v2", TaskStep.WHISPER.value, TaskStatus.FAILED.value)
        
        # ====== 第 2 轮 ======
        mock_job_manager.reset_mock()
        
        mock_pl_service2 = MagicMock()
        mock_pl_service2.get_playlist.return_value = FakePlaylist(id="PL_retry")
        mock_pl_service2.sync_playlist.return_value = FakeSyncResult(
            new_videos=["v3"], total_videos=3
        )
        # 现在 playlist 有 v1(已完成)、v2(失败)、v3(新)
        mock_pl_service2.get_playlist_videos.return_value = [
            FakeVideo(id="v1"), FakeVideo(id="v2"), FakeVideo(id="v3")
        ]
        
        service2 = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_retry"], once=True,
        )
        service2.playlist_service = mock_pl_service2
        # 从 service1 继承 session 提交记录，模拟同一 session 的多轮
        service2._session_submitted_ids = service1._session_submitted_ids.copy()
        service2.run()
        
        # 验证第 2 轮提交了 v2(重试) + v3(新)，但不包含 v1(已完成)
        mock_job_manager.assert_called_once()
        vid_ids_r2 = mock_job_manager.call_args.kwargs['video_ids']
        assert "v2" in vid_ids_r2, "失败的 v2 应被重试"
        assert "v3" in vid_ids_r2, "新视频 v3 应被提交"
        assert "v1" not in vid_ids_r2, "已完成的 v1 不应被重新提交"
        
        # 验证第 2 轮 round 记录
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        rounds = conn.execute(
            "SELECT * FROM watch_rounds ORDER BY id"
        ).fetchall()
        conn.close()
        
        # 应该有 2 条 round（来自两个不同的 session）
        assert len(rounds) >= 2
        # 最新一条应记录 1 个新视频 + 1 个重试
        latest = rounds[-1]
        assert latest['new_videos_found'] == 1  # 只有 v3 是新的
        retry_ids = json.loads(latest['retry_video_ids']) if latest['retry_video_ids'] else []
        assert "v2" in retry_ids
    
    def test_session_state_lifecycle(self, mock_config, mock_db, tmp_db):
        """
        验证 session 状态的完整生命周期：
        running → stopped（正常退出）
        running → error（异常退出）
        """
        # === 正常退出 ===
        mock_pl_service = MagicMock()
        mock_pl_service.get_playlist.return_value = FakePlaylist(id="PL_life")
        mock_pl_service.sync_playlist.return_value = FakeSyncResult(
            new_videos=[], total_videos=0
        )
        mock_pl_service.get_playlist_videos.return_value = []
        
        service = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_life"], once=True,
        )
        service.playlist_service = mock_pl_service
        service.run()
        
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        sess = conn.execute("SELECT * FROM watch_sessions ORDER BY rowid DESC LIMIT 1").fetchone()
        conn.close()
        
        assert sess['status'] == 'stopped'
        assert sess['stopped_at'] is not None
        assert sess['pid'] == os.getpid()
        
        # === 异常退出 ===
        # 注意：playlist 处理异常被内层 try/except 捕获，不会导致 session error。
        # 要测试 session error 状态，需要在外层循环基础设施（如 _update_session_stats）中触发异常。
        mock_pl_service2 = MagicMock()
        mock_pl_service2.get_playlist.return_value = FakePlaylist(id="PL_life")
        mock_pl_service2.sync_playlist.return_value = FakeSyncResult(
            new_videos=[], total_videos=0
        )
        mock_pl_service2.get_playlist_videos.return_value = []
        
        service_err = WatchService(
            config=mock_config, db=mock_db,
            playlist_ids=["PL_life"], once=True,
        )
        service_err.playlist_service = mock_pl_service2
        # 在 _update_session_stats 中注入异常，模拟 DB 故障
        service_err._update_session_stats = MagicMock(side_effect=RuntimeError("DB crashed"))
        
        with pytest.raises(RuntimeError, match="DB crashed"):
            service_err.run()
        
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        sess_err = conn.execute("SELECT * FROM watch_sessions ORDER BY rowid DESC LIMIT 1").fetchone()
        conn.close()
        
        assert sess_err['status'] == 'error'
        assert 'DB crashed' in sess_err['error']
    
    def test_max_retries_stops_retry(self, mock_job_manager, mock_config, mock_db, tmp_db):
        """
        验证重试次数上限：当同一视频重试超过 max_retries 次后不再重试
        """
        mock_config.storage.work_dir = str(Path(tmp_db).parent)
        mock_config.watch.max_retries = 2  # 最多重试 2 次
        
        # v_stubborn 有 failed task
        _insert_task(tmp_db, "v_stubborn", TaskStep.DOWNLOAD.value, TaskStatus.FAILED.value)
        
        mock_pl_service = MagicMock()
        mock_pl_service.get_playlist.return_value = FakePlaylist(id="PL_max")
        mock_pl_service.sync_playlist.return_value = FakeSyncResult(
            new_videos=[], total_videos=1
        )
        mock_pl_service.get_playlist_videos.return_value = [FakeVideo(id="v_stubborn")]
        mock_db.get_video = MagicMock(return_value=FakeVideo(id="v_stubborn"))
        
        # 模拟 3 次运行（超过 max_retries=2）
        for i in range(3):
            mock_job_manager.reset_mock()
            svc = WatchService(
                config=mock_config, db=mock_db,
                playlist_ids=["PL_max"], once=True,
            )
            svc.playlist_service = mock_pl_service
            # 手动设置重试计数（模拟跨 session 的累计）
            svc._retry_counts = {"v_stubborn": i}
            # v_stubborn 必须在 session 提交记录中才能被 retry
            svc._session_submitted_ids.add("v_stubborn")
            svc.run()
            
            if i < 2:
                # 前 2 次应提交重试
                assert mock_job_manager.call_count == 1, f"第 {i+1} 次应提交重试"
            else:
                # 第 3 次超过限制，不再重试
                mock_job_manager.assert_not_called()


class TestJobManagerCommandConstruction:
    """
    测试 API → JobManager → CLI 命令构建的全链路。
    验证 WebUI 提交 watch 请求时，JobManager 构建出正确的 CLI 命令。
    """
    
    def test_watch_command_basic(self):
        """基本 watch 命令构建"""
        from vat.web.jobs import JobManager
        
        params = {
            'playlist_ids': ['PL_A', 'PL_B'],
            'once': True,
            'force': False,
            'fail_fast': False,
        }
        cmd = JobManager._build_tools_command('watch', params)
        
        assert cmd[:5] == ["python", "-m", "vat", "tools", "watch"]
        assert "--playlist" in cmd
        assert "PL_A" in cmd
        assert "PL_B" in cmd
        assert "--once" in cmd
        # force=False 和 fail_fast=False 不应出现
        assert "--force" not in cmd
        assert "--fail-fast" not in cmd
    
    def test_watch_command_full_params(self):
        """完整参数 watch 命令构建"""
        from vat.web.jobs import JobManager
        
        params = {
            'playlist_ids': ['PL_X'],
            'interval': 30,
            'once': False,
            'stages': 'download,whisper',
            'gpu': 'cuda:1',
            'concurrency': 4,
            'force': True,
            'fail_fast': True,
        }
        cmd = JobManager._build_tools_command('watch', params)
        cmd_str = " ".join(cmd)
        
        assert "--playlist PL_X" in cmd_str
        assert "--interval 30" in cmd_str
        assert "--stages download,whisper" in cmd_str
        assert "--gpu cuda:1" in cmd_str
        assert "--concurrency 4" in cmd_str
        assert "--force" in cmd_str
        assert "--fail-fast" in cmd_str
        # once=False 不应出现
        assert "--once" not in cmd_str
    
    def test_watch_command_multi_playlist(self):
        """多 playlist 参数展开为多个 --playlist"""
        from vat.web.jobs import JobManager
        
        params = {
            'playlist_ids': ['PL_1', 'PL_2', 'PL_3'],
            'once': True,
            'force': False,
            'fail_fast': False,
        }
        cmd = JobManager._build_tools_command('watch', params)
        
        # 每个 playlist ID 前面都应有 --playlist
        playlist_indices = [i for i, c in enumerate(cmd) if c == '--playlist']
        assert len(playlist_indices) == 3
        for idx in playlist_indices:
            assert cmd[idx + 1] in ['PL_1', 'PL_2', 'PL_3']
