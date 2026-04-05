"""
web/jobs.py 单元测试

测试 JobManager CRUD、进度日志解析、WebJob 序列化。
不测试子进程启动（需要真实 CLI 环境）。
"""
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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

    def test_find_latest_job_is_not_limited_to_last_200_rows(self, job_env):
        jm, _, _ = job_env
        import json
        from datetime import datetime, timedelta

        with jm._get_connection() as conn:
            target_time = datetime(2026, 3, 24, 12, 0, 0)
            conn.execute("""
                INSERT INTO web_jobs (
                    job_id, video_ids, steps, gpu_device, force, status, log_file, created_at,
                    task_type, task_params
                ) VALUES (?, ?, ?, 'auto', 0, 'running', NULL, ?, ?, ?)
            """, (
                "target-job",
                json.dumps([]),
                json.dumps(["sync-playlist"]),
                target_time.isoformat(),
                "sync-playlist",
                json.dumps({"playlist_id": "PL_TARGET"}),
            ))

            for idx in range(205):
                created_at = target_time + timedelta(minutes=idx + 1)
                conn.execute("""
                    INSERT INTO web_jobs (
                        job_id, video_ids, steps, gpu_device, force, status, log_file, created_at,
                        task_type, task_params
                    ) VALUES (?, ?, ?, 'auto', 0, 'completed', NULL, ?, ?, ?)
                """, (
                    f"newer-{idx}",
                    json.dumps([]),
                    json.dumps(["sync-playlist"]),
                    created_at.isoformat(),
                    "sync-playlist",
                    json.dumps({"playlist_id": f"PL_{idx}"}),
                ))

        job = jm.find_latest_job("sync-playlist", {"playlist_id": "PL_TARGET"})
        assert job is not None
        assert job.job_id == "target-job"


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


class TestJobProcessStartupCleanup:

    def test_start_job_process_kills_spawned_process_when_db_update_fails(self, job_env, monkeypatch):
        jm, _, _ = job_env
        fake_process = MagicMock(pid=43210)
        kill_calls = []

        monkeypatch.setattr(jm, "_build_process_command", lambda *args, **kwargs: ["python", "-m", "vat", "process"])
        monkeypatch.setattr("vat.web.jobs.subprocess.Popen", lambda *args, **kwargs: fake_process)
        monkeypatch.setattr("vat.web.jobs.os.getpgid", lambda pid: 98765)
        monkeypatch.setattr("vat.web.jobs.os.killpg", lambda pgid, sig: kill_calls.append((pgid, sig)))

        @contextmanager
        def _failing_connection():
            raise RuntimeError("db update failed")
            yield

        monkeypatch.setattr(jm, "_get_connection", _failing_connection)

        with pytest.raises(RuntimeError, match="db update failed"):
            jm._start_job_process(
                job_id="job1",
                video_ids=["v1"],
                steps=["download"],
                gpu_device="auto",
                force=False,
                log_file=os.path.join(jm.log_dir, "job1.log"),
            )

        assert kill_calls, "启动后的子进程应在 DB 更新失败时被终止"


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

    def test_skipped_step_unblocks_video(self, env):
        """requested step 为 skipped 也视为该 job 已满足。"""
        jm, db_path = env
        self._insert_running_job(jm, "job1", ["v1", "v2"], ["download", "whisper"])
        self._insert_task(db_path, "v1", "download", "completed")
        self._insert_task(db_path, "v1", "whisper", "skipped")

        result = jm.get_running_video_ids()
        assert "v1" not in result
        assert result == {"v2"}

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

    def test_task_params_todo_video_ids_are_included_in_hide_processing_set(self, env):
        """running job 的 task_params 待处理视频也应被首页过滤隐藏。"""
        jm, db_path = env
        import json
        from datetime import datetime

        with jm._get_connection() as conn:
            conn.execute("""
                INSERT INTO web_jobs (
                    job_id, video_ids, steps, status, created_at, task_params
                ) VALUES (?, ?, ?, 'running', ?, ?)
            """, (
                "job1",
                json.dumps(["v_processing"]),
                json.dumps(["download"]),
                datetime.now().isoformat(),
                json.dumps({"todo_video_ids": ["v_todo"]}),
            ))

        result = jm.get_running_and_queued_video_ids()
        assert result == {"v_processing", "v_todo"}


class TestTaskParamsPersistence:
    """测试 task_params 统一持久化：process 特有参数存入 task_params JSON 字段"""

    def test_submit_job_rolls_back_db_record_when_start_fails(self):
        """子进程启动失败时，不应遗留 pending job 记录。"""
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            log_dir = os.path.join(tmpdir, "logs")
            jm = JobManager(db_path, log_dir)

            from unittest.mock import patch
            with patch.object(jm, '_start_job_process', side_effect=RuntimeError("spawn failed")):
                with pytest.raises(RuntimeError, match="spawn failed"):
                    jm.submit_job(
                        video_ids=["v1"],
                        steps=["download"],
                    )

            assert jm.list_jobs() == []
        finally:
            shutil.rmtree(tmpdir)

    def test_process_params_merged_into_task_params(self):
        """process 任务的 playlist_id/upload_batch_size/upload_mode 应合并到 task_params"""
        import json, sqlite3, tempfile, shutil
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            log_dir = os.path.join(tmpdir, "logs")
            jm = JobManager(db_path, log_dir)

            # 直接写入 DB（不启动子进程）
            from unittest.mock import patch
            with patch.object(jm, '_start_job_process'):
                job_id = jm.submit_job(
                    video_ids=["v1"],
                    steps=["upload"],
                    playlist_id="PL_abc",
                    upload_batch_size=3,
                    upload_mode="dtime",
                    upload_cron="0 12 * * *",
                )

            job = jm.get_job(job_id)
            assert job is not None
            params = job.task_params
            assert params['playlist_id'] == 'PL_abc'
            assert params['upload_batch_size'] == 3
            assert params['upload_mode'] == 'dtime'
        finally:
            shutil.rmtree(tmpdir)

    def test_process_default_params_not_stored(self):
        """默认值的参数不写入 task_params（减少噪音）"""
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            log_dir = os.path.join(tmpdir, "logs")
            jm = JobManager(db_path, log_dir)

            from unittest.mock import patch
            with patch.object(jm, '_start_job_process'):
                job_id = jm.submit_job(
                    video_ids=["v1"],
                    steps=["upload"],
                    # playlist_id=None, upload_batch_size=1, upload_mode='cron' (all defaults)
                )

            job = jm.get_job(job_id)
            assert job.task_params == {}
        finally:
            shutil.rmtree(tmpdir)

    def test_tools_params_stored_directly(self):
        """tools 任务的 task_params 直接存储"""
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            log_dir = os.path.join(tmpdir, "logs")
            jm = JobManager(db_path, log_dir)

            from unittest.mock import patch
            with patch.object(jm, '_start_job_process'):
                job_id = jm.submit_job(
                    video_ids=[],
                    steps=["fix-violation"],
                    task_type="fix-violation",
                    task_params={"aid": 12345, "max_rounds": 5},
                )

            job = jm.get_job(job_id)
            assert job.task_params['aid'] == 12345
            assert job.task_params['max_rounds'] == 5
            assert job.task_type == 'fix-violation'
        finally:
            shutil.rmtree(tmpdir)

    def test_build_process_command_reads_task_params(self):
        """_build_process_command 从 task_params 读取 playlist_id/upload_batch_size/upload_mode"""
        cmd = JobManager._build_process_command(
            video_ids=["v1", "v2"],
            steps=["upload"],
            gpu_device="auto",
            force=False,
            concurrency=1,
            upload_cron="0 12 * * *",
            fail_fast=False,
            task_params={
                'playlist_id': 'PL_test',
                'upload_batch_size': 5,
                'upload_mode': 'dtime',
            },
        )
        assert "-p" in cmd
        idx_p = cmd.index("-p")
        assert cmd[idx_p + 1] == "PL_test"
        assert "--upload-batch-size" in cmd
        idx_bs = cmd.index("--upload-batch-size")
        assert cmd[idx_bs + 1] == "5"
        assert "--upload-mode" in cmd
        idx_m = cmd.index("--upload-mode")
        assert cmd[idx_m + 1] == "dtime"

    def test_build_process_command_includes_group_config_path(self):
        """process 子进程应继承 web 启动时的 group 级 --config。"""
        cmd = JobManager._build_process_command(
            video_ids=["v1"],
            steps=["download"],
            gpu_device="auto",
            force=False,
            concurrency=1,
            upload_cron=None,
            fail_fast=False,
            task_params={},
            config_path="config/custom.yaml",
        )
        assert cmd[:5] == [sys.executable, "-m", "vat", "-c", "config/custom.yaml"]
        assert cmd[5] == "process"

    def test_build_process_command_empty_task_params(self):
        """task_params 为空时不生成额外参数"""
        cmd = JobManager._build_process_command(
            video_ids=["v1"],
            steps=["download"],
            gpu_device="auto",
            force=False,
            concurrency=1,
            upload_cron=None,
            fail_fast=False,
            task_params={},
        )
        assert "-p" not in cmd
        assert "--upload-batch-size" not in cmd
        assert "--upload-mode" not in cmd
        assert "--delay-start" not in cmd

    def test_delay_start_merged_into_task_params(self):
        """delay_start > 0 时应合并到 task_params"""
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            log_dir = os.path.join(tmpdir, "logs")
            jm = JobManager(db_path, log_dir)

            from unittest.mock import patch
            with patch.object(jm, '_start_job_process'):
                job_id = jm.submit_job(
                    video_ids=["v1"],
                    steps=["upload"],
                    delay_start=300,
                )

            job = jm.get_job(job_id)
            assert job.task_params['delay_start'] == 300
        finally:
            shutil.rmtree(tmpdir)

    def test_delay_start_zero_not_stored(self):
        """delay_start=0 不写入 task_params"""
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            log_dir = os.path.join(tmpdir, "logs")
            jm = JobManager(db_path, log_dir)

            from unittest.mock import patch
            with patch.object(jm, '_start_job_process'):
                job_id = jm.submit_job(
                    video_ids=["v1"],
                    steps=["upload"],
                    delay_start=0,
                )

            job = jm.get_job(job_id)
            assert 'delay_start' not in job.task_params
        finally:
            shutil.rmtree(tmpdir)

    def test_build_process_command_with_delay_start(self):
        """_build_process_command 从 task_params 读取 delay_start"""
        cmd = JobManager._build_process_command(
            video_ids=["v1"],
            steps=["upload"],
            gpu_device="auto",
            force=False,
            concurrency=1,
            upload_cron=None,
            fail_fast=False,
            task_params={'delay_start': 120},
        )
        assert "--delay-start" in cmd
        idx = cmd.index("--delay-start")
        assert cmd[idx + 1] == "120"

    def test_build_tools_command_includes_group_config_path_for_watch(self):
        """tools/watch 子进程应继承 web 启动时的 group 级 --config。"""
        cmd = JobManager._build_tools_command(
            "watch",
            {"playlist_ids": ["PL1"], "once": True},
            config_path="config/custom.yaml",
        )
        assert cmd[:6] == [sys.executable, "-m", "vat", "-c", "config/custom.yaml", "tools"]
        assert cmd[6] == "watch"


class TestProcessJobResultContracts:
    @pytest.fixture
    def env(self):
        import sqlite3, shutil
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
        log_dir = os.path.join(tmpdir, "logs")
        jm = JobManager(db_path, log_dir)
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

    def _insert_job(self, jm, job_id, video_ids, steps, pid=1234):
        import json
        from datetime import datetime
        with jm._get_connection() as conn:
            conn.execute("""
                INSERT INTO web_jobs (job_id, video_ids, steps, gpu_device, force, status, pid, created_at)
                VALUES (?, ?, ?, 'auto', 0, 'running', ?, ?)
            """, (job_id, json.dumps(video_ids), json.dumps(steps), pid, datetime.now().isoformat()))

    def _insert_task(self, db_path, video_id, step, status, error_message=None):
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT INTO tasks (video_id, step, status, error_message) VALUES (?, ?, ?, ?)
        """, (video_id, step, status, error_message))
        conn.commit()
        conn.close()

    def test_determine_job_result_fails_when_no_task_records_exist(self, env):
        jm, _ = env
        self._insert_job(jm, "job1", ["v1"], ["download", "whisper"])
        job = jm.get_job("job1")

        status, error, progress = jm._determine_job_result(job)

        assert status == JobStatus.FAILED
        assert progress == 0.0
        assert error is not None
        assert "缺少" in error or "未完成" in error

    def test_determine_job_result_marks_missing_videos_as_partial_failure(self, env):
        jm, db_path = env
        self._insert_job(jm, "job2", ["v1", "v2"], ["download", "whisper"])
        self._insert_task(db_path, "v1", "download", "completed")
        self._insert_task(db_path, "v1", "whisper", "completed")
        job = jm.get_job("job2")

        status, error, progress = jm._determine_job_result(job)

        assert status == JobStatus.PARTIAL_COMPLETED
        assert progress == 0.5
        assert error is not None
        assert "v2" in error


class TestCancelJobContracts:
    def test_cancel_job_marks_cancel_requested_and_sends_sigterm(self, job_env, monkeypatch):
        jm, _, _ = job_env
        from datetime import datetime
        import json

        with jm._get_connection() as conn:
            conn.execute("""
                INSERT INTO web_jobs (
                    job_id, video_ids, steps, gpu_device, force, status, pid, created_at
                ) VALUES (?, ?, ?, 'auto', 0, 'running', ?, ?)
            """, ("job-cancel", json.dumps(["v1"]), json.dumps(["download"]), 4321, datetime.now().isoformat()))

        killpg_calls = []
        monkeypatch.setattr("vat.web.jobs.JobManager.update_job_status", lambda _self, _job_id: None)
        monkeypatch.setattr("vat.web.jobs.os.getpgid", lambda pid: 9876)
        monkeypatch.setattr("vat.web.jobs.os.killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))

        assert jm.cancel_job("job-cancel") is True
        assert killpg_calls == [(9876, __import__("signal").SIGTERM)]
        job = jm.get_job("job-cancel")
        assert job.status == JobStatus.RUNNING
        assert job.cancel_requested is True

    def test_cancel_job_returns_false_when_status_refresh_finishes_job(self, job_env, monkeypatch):
        jm, _, _ = job_env
        from datetime import datetime
        import json

        with jm._get_connection() as conn:
            conn.execute("""
                INSERT INTO web_jobs (
                    job_id, video_ids, steps, gpu_device, force, status, pid, created_at
                ) VALUES (?, ?, ?, 'auto', 0, 'running', ?, ?)
            """, ("job-finished", json.dumps(["v1"]), json.dumps(["download"]), 4322, datetime.now().isoformat()))

        def mark_completed(_self, job_id):
            with jm._get_connection() as conn:
                conn.execute(
                    "UPDATE web_jobs SET status = ?, finished_at = ? WHERE job_id = ?",
                    (JobStatus.COMPLETED.value, datetime.now().isoformat(), job_id),
                )

        monkeypatch.setattr("vat.web.jobs.JobManager.update_job_status", mark_completed)

        assert jm.cancel_job("job-finished") is False
        assert jm.get_job("job-finished").status == JobStatus.COMPLETED

    def test_update_job_status_marks_cancel_requested_job_as_cancelled(self, job_env, monkeypatch):
        jm, _, _ = job_env
        from datetime import datetime
        import json

        with jm._get_connection() as conn:
            conn.execute("""
                INSERT INTO web_jobs (
                    job_id, video_ids, steps, gpu_device, force, status, pid, created_at, cancel_requested
                ) VALUES (?, ?, ?, 'auto', 0, 'running', ?, ?, 1)
            """, ("job-cancelled", json.dumps(["v1"]), json.dumps(["download"]), 4323, datetime.now().isoformat()))

        monkeypatch.setattr("vat.web.jobs.JobManager._is_process_alive", lambda _self, _pid: False)
        monkeypatch.setattr("vat.web.jobs.JobManager._parse_progress_from_log", lambda _self, _log_file: 0.4)

        jm.update_job_status("job-cancelled")

        job = jm.get_job("job-cancelled")
        assert job.status == JobStatus.CANCELLED
        assert job.progress == 0.4
        assert job.finished_at is not None

    def test_is_process_alive_treats_zombie_as_exited(self, job_env, monkeypatch):
        jm, _, _ = job_env

        waitpid_calls = []

        monkeypatch.setattr("vat.web.jobs.os.kill", lambda _pid, _sig: None)
        monkeypatch.setattr("vat.web.jobs.os.waitpid", lambda pid, flags: waitpid_calls.append((pid, flags)))

        class _StatusFile:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return "Name:\tpython\nState:\tZ (zombie)\n"

        monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: _StatusFile())

        assert jm._is_process_alive(4324) is False
        assert waitpid_calls == [(4324, __import__("os").WNOHANG)]
