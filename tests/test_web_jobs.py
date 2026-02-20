"""
web/jobs.py 单元测试

测试 JobManager CRUD、进度日志解析、WebJob 序列化。
不测试子进程启动（需要真实 CLI 环境）。
"""
import os
import tempfile
import pytest
from pathlib import Path
from vat.web.jobs import JobManager, JobStatus, WebJob


@pytest.fixture
def job_env():
    """创建临时 DB + log 目录"""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    log_dir = os.path.join(tmpdir, "logs")
    jm = JobManager(db_path, log_dir)
    yield jm, tmpdir, log_dir
    import shutil
    shutil.rmtree(tmpdir)


class TestJobManagerCRUD:

    def test_init_creates_table(self, job_env):
        jm, tmpdir, _ = job_env
        import sqlite3
        conn = sqlite3.connect(os.path.join(tmpdir, "test.db"))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='web_jobs'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_get_nonexistent_job(self, job_env):
        jm, _, _ = job_env
        assert jm.get_job("nonexistent") is None

    def test_list_empty(self, job_env):
        jm, _, _ = job_env
        assert jm.list_jobs() == []


class TestWebJobSerialization:

    def test_to_dict_includes_all_fields(self):
        from datetime import datetime
        job = WebJob(
            job_id="abc123",
            video_ids=["v1", "v2"],
            steps=["download", "whisper"],
            gpu_device="auto",
            force=True,
            status=JobStatus.RUNNING,
            pid=12345,
            log_file="/tmp/log.txt",
            progress=0.5,
            error=None,
            created_at=datetime(2025, 1, 1),
            started_at=datetime(2025, 1, 1, 0, 1),
            finished_at=None,
        )
        d = job.to_dict()
        assert d["job_id"] == "abc123"
        assert d["task_id"] == "abc123"  # 兼容字段
        assert d["video_ids"] == ["v1", "v2"]
        assert d["steps"] == ["download", "whisper"]
        assert d["force"] is True
        assert d["status"] == "running"
        assert d["progress"] == 0.5
        assert d["pid"] == 12345

    def test_to_dict_status_is_string(self):
        from datetime import datetime
        job = WebJob(
            job_id="x", video_ids=[], steps=[], gpu_device="auto",
            force=False, status=JobStatus.PARTIAL_COMPLETED, pid=None,
            log_file=None, progress=0.8, error=None,
            created_at=datetime.now(), started_at=None, finished_at=None,
        )
        d = job.to_dict()
        assert d["status"] == "partial_completed"
        assert isinstance(d["status"], str)


class TestProgressLogParsing:
    """测试 _parse_progress_from_log 实际方法"""

    def _make_jm(self):
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        log_dir = os.path.join(tmpdir, "logs")
        return JobManager(db_path, log_dir), tmpdir

    def test_empty_log(self):
        jm, tmpdir = self._make_jm()
        log_file = os.path.join(tmpdir, "empty.log")
        Path(log_file).write_text("")
        assert jm._parse_progress_from_log(log_file) == 0.0
        import shutil; shutil.rmtree(tmpdir)

    def test_total_progress_format(self):
        jm, tmpdir = self._make_jm()
        log_file = os.path.join(tmpdir, "test.log")
        Path(log_file).write_text("[TOTAL:50%] [75%] 处理中\n")
        assert jm._parse_progress_from_log(log_file) == 0.5
        import shutil; shutil.rmtree(tmpdir)

    def test_fallback_to_per_video(self):
        jm, tmpdir = self._make_jm()
        log_file = os.path.join(tmpdir, "test.log")
        Path(log_file).write_text("[75%] 处理中\n")
        assert jm._parse_progress_from_log(log_file) == 0.75
        import shutil; shutil.rmtree(tmpdir)

    def test_total_preferred_over_per_video(self):
        jm, tmpdir = self._make_jm()
        log_file = os.path.join(tmpdir, "test.log")
        Path(log_file).write_text("[TOTAL:33%] [100%] 阶段完成\n")
        assert jm._parse_progress_from_log(log_file) == 0.33
        import shutil; shutil.rmtree(tmpdir)

    def test_multi_line_last_wins(self):
        jm, tmpdir = self._make_jm()
        log_file = os.path.join(tmpdir, "test.log")
        Path(log_file).write_text(
            "[TOTAL:10%] 开始\n"
            "[TOTAL:50%] 中间\n"
            "[TOTAL:90%] 快完成\n"
        )
        assert jm._parse_progress_from_log(log_file) == 0.9
        import shutil; shutil.rmtree(tmpdir)

    def test_nonexistent_file(self):
        jm, tmpdir = self._make_jm()
        assert jm._parse_progress_from_log("/nonexistent/path.log") == 0.0
        import shutil; shutil.rmtree(tmpdir)

    def test_none_file(self):
        jm, tmpdir = self._make_jm()
        assert jm._parse_progress_from_log(None) == 0.0
        import shutil; shutil.rmtree(tmpdir)


class TestVideoDeduplication:
    """测试 running job 中的视频去重逻辑：已完成所有步骤的视频不再阻塞"""

    @pytest.fixture
    def env(self):
        """创建包含 web_jobs + tasks 表的测试环境"""
        import sqlite3, shutil
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        log_dir = os.path.join(tmpdir, "logs")
        jm = JobManager(db_path, log_dir)
        # 在同一 DB 中创建 tasks 表（生产环境由 Database 类创建）
        conn = sqlite3.connect(db_path)
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
        yield jm, db_path
        shutil.rmtree(tmpdir)

    def _insert_running_job(self, jm, job_id, video_ids, steps):
        """插入一个 running 状态的 job"""
        import json
        from datetime import datetime
        with jm._get_connection() as conn:
            conn.execute("""
                INSERT INTO web_jobs (job_id, video_ids, steps, status, created_at)
                VALUES (?, ?, ?, 'running', ?)
            """, (job_id, json.dumps(video_ids), json.dumps(steps),
                  datetime.now().isoformat()))

    def _insert_task(self, db_path, video_id, step, status):
        """插入一条 task 记录"""
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT INTO tasks (video_id, step, status) VALUES (?, ?, ?)
        """, (video_id, step, status))
        conn.commit()
        conn.close()

    def test_all_videos_pending_all_blocked(self, env):
        """所有视频都未处理 → 全部阻塞"""
        jm, db_path = env
        self._insert_running_job(jm, "job1", ["v1", "v2", "v3"], ["download", "whisper"])
        result = jm.get_running_video_ids()
        assert result == {"v1", "v2", "v3"}

    def test_completed_video_unblocked(self, env):
        """v1 完成所有步骤 → v1 不阻塞，v2/v3 仍阻塞"""
        jm, db_path = env
        self._insert_running_job(jm, "job1", ["v1", "v2", "v3"], ["download", "whisper"])
        self._insert_task(db_path, "v1", "download", "completed")
        self._insert_task(db_path, "v1", "whisper", "completed")
        result = jm.get_running_video_ids()
        assert "v1" not in result
        assert result == {"v2", "v3"}

    def test_partially_completed_still_blocked(self, env):
        """v1 只完成 download 未完成 whisper → 仍阻塞"""
        jm, db_path = env
        self._insert_running_job(jm, "job1", ["v1", "v2"], ["download", "whisper"])
        self._insert_task(db_path, "v1", "download", "completed")
        result = jm.get_running_video_ids()
        assert "v1" in result

    def test_failed_video_still_blocked(self, env):
        """v1 有失败步骤（可能被重试）→ 仍阻塞"""
        jm, db_path = env
        self._insert_running_job(jm, "job1", ["v1", "v2"], ["download", "whisper"])
        self._insert_task(db_path, "v1", "download", "completed")
        self._insert_task(db_path, "v1", "whisper", "failed")
        result = jm.get_running_video_ids()
        assert "v1" in result

    def test_latest_task_wins(self, env):
        """v1 先失败后成功（重试成功）→ 解除阻塞"""
        jm, db_path = env
        self._insert_running_job(jm, "job1", ["v1"], ["download", "whisper"])
        # 第一次：失败
        self._insert_task(db_path, "v1", "download", "completed")
        self._insert_task(db_path, "v1", "whisper", "failed")
        # 重试：成功（id 更大 → 最新记录）
        self._insert_task(db_path, "v1", "whisper", "completed")
        result = jm.get_running_video_ids()
        assert "v1" not in result

    def test_get_running_job_for_completed_video(self, env):
        """已完成视频查询 active job 应返回 None"""
        jm, db_path = env
        self._insert_running_job(jm, "job1", ["v1", "v2"], ["download"])
        self._insert_task(db_path, "v1", "download", "completed")
        assert jm.get_running_job_for_video("v1") is None
        # v2 未完成，仍能找到 job
        job = jm.get_running_job_for_video("v2")
        assert job is not None
        assert job.job_id == "job1"

    def test_multiple_running_jobs(self, env):
        """多个 running job，各自独立判断"""
        jm, db_path = env
        self._insert_running_job(jm, "job1", ["v1", "v2"], ["download"])
        self._insert_running_job(jm, "job2", ["v3", "v4"], ["download", "whisper"])
        # v1 在 job1 中完成
        self._insert_task(db_path, "v1", "download", "completed")
        # v3 在 job2 中只完成 download（还差 whisper）
        self._insert_task(db_path, "v3", "download", "completed")
        result = jm.get_running_video_ids()
        assert "v1" not in result  # 完成了 job1 的所有步骤
        assert "v2" in result      # 未完成
        assert "v3" in result      # 只完成部分步骤
        assert "v4" in result      # 未完成

    def test_tools_job_no_video_ids(self, env):
        """tools 类型 job（video_ids 为空）不贡献阻塞"""
        jm, db_path = env
        self._insert_running_job(jm, "job1", [], ["fix-violation"])
        result = jm.get_running_video_ids()
        assert result == set()
