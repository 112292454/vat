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
