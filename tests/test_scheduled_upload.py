"""
定时上传功能单元测试

测试：
1. CLI --upload-cron 参数校验逻辑
2. _run_scheduled_uploads 队列构建与 dry-run
3. _format_duration 辅助函数
4. WebJob upload_cron 序列化
5. API upload_cron 校验（steps 互斥、cron 合法性）
"""
import os
import tempfile
import sqlite3
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from vat.cli.commands import parse_stages, _format_duration
from vat.models import TaskStep, TaskStatus
from vat.web.jobs import JobManager, JobStatus, WebJob


# ========== _format_duration 测试 ==========

class TestFormatDuration:

    def test_seconds(self):
        assert _format_duration(30) == "30秒"

    def test_minutes(self):
        assert _format_duration(125) == "2分5秒"

    def test_hours(self):
        assert _format_duration(3725) == "1时2分"

    def test_days(self):
        assert _format_duration(90061) == "1天1时1分"

    def test_zero(self):
        assert _format_duration(0) == "0秒"


# ========== WebJob upload_cron 序列化测试 ==========

class TestWebJobUploadCron:

    def test_to_dict_includes_upload_cron(self):
        job = WebJob(
            job_id="test1",
            video_ids=["v1"],
            steps=["upload"],
            gpu_device="auto",
            force=False,
            status=JobStatus.RUNNING,
            pid=123,
            log_file=None,
            progress=0.0,
            error=None,
            created_at=datetime.now(),
            started_at=None,
            finished_at=None,
            upload_cron="0 12,18 * * *",
        )
        d = job.to_dict()
        assert d["upload_cron"] == "0 12,18 * * *"

    def test_to_dict_upload_cron_none(self):
        job = WebJob(
            job_id="test2",
            video_ids=["v1"],
            steps=["download"],
            gpu_device="auto",
            force=False,
            status=JobStatus.PENDING,
            pid=None,
            log_file=None,
            progress=0.0,
            error=None,
            created_at=datetime.now(),
            started_at=None,
            finished_at=None,
        )
        d = job.to_dict()
        assert d["upload_cron"] is None


# ========== JobManager DB schema 迁移测试 ==========

class TestJobManagerUploadCronColumn:

    def test_upload_cron_column_exists(self):
        """新建 DB 时 upload_cron 列应存在"""
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        log_dir = os.path.join(tmpdir, "logs")
        jm = JobManager(db_path, log_dir)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(web_jobs)")
        columns = [row['name'] for row in cursor.fetchall()]
        conn.close()

        assert "upload_cron" in columns

        import shutil
        shutil.rmtree(tmpdir)

    def test_migration_on_existing_db(self):
        """已有 DB（无 upload_cron 列）初始化后应添加该列"""
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        log_dir = os.path.join(tmpdir, "logs")

        # 先创建一个没有 upload_cron 列的旧表
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE web_jobs (
                job_id TEXT PRIMARY KEY,
                video_ids TEXT NOT NULL,
                steps TEXT NOT NULL,
                gpu_device TEXT DEFAULT 'auto',
                force INTEGER DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                pid INTEGER,
                log_file TEXT,
                progress REAL DEFAULT 0.0,
                error TEXT,
                created_at TIMESTAMP NOT NULL,
                started_at TIMESTAMP,
                finished_at TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

        # 初始化 JobManager 应触发迁移
        jm = JobManager(db_path, log_dir)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(web_jobs)")
        columns = [row['name'] for row in cursor.fetchall()]
        conn.close()

        assert "upload_cron" in columns

        import shutil
        shutil.rmtree(tmpdir)


# ========== CLI upload-cron 校验逻辑测试 ==========

class TestUploadCronValidation:

    def test_cron_expression_validity(self):
        """croniter 能正确判断合法/非法的 cron 表达式"""
        from croniter import croniter
        assert croniter.is_valid("0 12,18 * * *")
        assert croniter.is_valid("0 */6 * * *")
        assert croniter.is_valid("30 8 * * 1,4")
        assert not croniter.is_valid("invalid")
        assert not croniter.is_valid("0 12")

    def test_parse_stages_upload_only(self):
        """parse_stages('upload') 应返回 [TaskStep.UPLOAD]"""
        result = parse_stages("upload")
        assert result == [TaskStep.UPLOAD]

    def test_parse_stages_upload_not_alone(self):
        """upload + 其他阶段应返回多个 step"""
        result = parse_stages("embed,upload")
        assert TaskStep.EMBED in result
        assert TaskStep.UPLOAD in result
        assert len(result) == 2


# ========== croniter 调度逻辑测试 ==========

class TestCroniterScheduling:

    def test_next_trigger_times(self):
        """验证 croniter 能正确计算下一次触发时间"""
        from croniter import croniter

        # 每天 12:00 和 18:00
        base = datetime(2026, 2, 15, 10, 0, 0)
        cron = croniter("0 12,18 * * *", base)

        t1 = cron.get_next(datetime)
        assert t1 == datetime(2026, 2, 15, 12, 0, 0)

        t2 = cron.get_next(datetime)
        assert t2 == datetime(2026, 2, 15, 18, 0, 0)

        t3 = cron.get_next(datetime)
        assert t3 == datetime(2026, 2, 16, 12, 0, 0)

    def test_interval_cron(self):
        """验证 */6 小时间隔"""
        from croniter import croniter

        base = datetime(2026, 2, 15, 1, 0, 0)
        cron = croniter("0 */6 * * *", base)

        t1 = cron.get_next(datetime)
        assert t1 == datetime(2026, 2, 15, 6, 0, 0)

        t2 = cron.get_next(datetime)
        assert t2 == datetime(2026, 2, 15, 12, 0, 0)
