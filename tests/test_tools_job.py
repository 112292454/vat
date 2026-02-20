"""
测试 tools 任务系统：
- JobManager 泛化（task_type, task_params）
- _build_tools_command 命令构建
- _determine_tools_job_result 日志标记解析
"""
import json
import tempfile
import os
from pathlib import Path
from unittest import TestCase

from vat.web.jobs import JobManager, WebJob, JobStatus, TOOLS_TASK_TYPES


class TestBuildToolsCommand(TestCase):
    """测试 _build_tools_command 命令构建"""

    def test_fix_violation_basic(self):
        cmd = JobManager._build_tools_command('fix-violation', {'aid': 12345})
        self.assertEqual(cmd[:5], ['python', '-m', 'vat', 'tools', 'fix-violation'])
        self.assertIn('--aid', cmd)
        self.assertIn('12345', cmd)

    def test_fix_violation_full(self):
        cmd = JobManager._build_tools_command('fix-violation', {
            'aid': 99,
            'video_path': '/tmp/video.mp4',
            'margin': 2.0,
            'mask_text': '遮罩',
            'dry_run': True,
        })
        self.assertIn('--aid', cmd)
        self.assertIn('--video-path', cmd)
        self.assertIn('--margin', cmd)
        self.assertIn('--mask-text', cmd)
        self.assertIn('--dry-run', cmd)

    def test_fix_violation_dry_run_false(self):
        """dry_run=False 不应出现 --dry-run"""
        cmd = JobManager._build_tools_command('fix-violation', {
            'aid': 1,
            'dry_run': False,
        })
        self.assertNotIn('--dry-run', cmd)

    def test_sync_playlist(self):
        cmd = JobManager._build_tools_command('sync-playlist', {
            'playlist_id': 'PL_abc',
            'url': 'https://youtube.com/playlist?list=PL_abc',
        })
        self.assertIn('--playlist', cmd)
        self.assertIn('PL_abc', cmd)
        self.assertIn('--url', cmd)

    def test_refresh_playlist(self):
        cmd = JobManager._build_tools_command('refresh-playlist', {
            'playlist_id': 'PL_xyz',
            'force_refetch': True,
            'force_retranslate': False,
        })
        self.assertIn('--force-refetch', cmd)
        self.assertNotIn('--force-retranslate', cmd)

    def test_sync_db(self):
        cmd = JobManager._build_tools_command('sync-db', {
            'season_id': 7376902,
            'playlist_id': 'PL_test',
            'dry_run': True,
        })
        self.assertIn('--season', cmd)
        self.assertIn('7376902', cmd)
        self.assertIn('--playlist', cmd)
        self.assertIn('--dry-run', cmd)

    def test_season_sync(self):
        cmd = JobManager._build_tools_command('season-sync', {
            'playlist_id': 'PL_ss',
        })
        self.assertEqual(cmd[:5], ['python', '-m', 'vat', 'tools', 'season-sync'])
        self.assertIn('--playlist', cmd)

    def test_none_values_skipped(self):
        """None 值的参数不应出现在命令中"""
        cmd = JobManager._build_tools_command('fix-violation', {
            'aid': 1,
            'video_path': None,
            'margin': None,
        })
        self.assertNotIn('--video-path', cmd)
        self.assertNotIn('--margin', cmd)
        self.assertIn('--aid', cmd)

    def test_unknown_task_type(self):
        """未知 task_type 应返回基础命令"""
        cmd = JobManager._build_tools_command('unknown-type', {})
        self.assertEqual(cmd, ['python', '-m', 'vat', 'tools', 'unknown-type'])


class TestDetermineToolsJobResult(TestCase):
    """测试 _determine_tools_job_result 日志标记解析"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.jm = JobManager(
            db_path=os.path.join(self.tmpdir, 'test_jobs.db'),
            log_dir=os.path.join(self.tmpdir, 'logs')
        )

    def _make_job(self, log_content: str) -> WebJob:
        """创建一个 tools 类型的 WebJob 并写入日志"""
        log_file = os.path.join(self.tmpdir, 'test.log')
        Path(log_file).write_text(log_content)
        return WebJob(
            job_id='test123',
            video_ids=[],
            steps=['fix-violation'],
            gpu_device='auto',
            force=False,
            status=JobStatus.RUNNING,
            pid=None,
            log_file=log_file,
            progress=0.0,
            error=None,
            created_at=None,
            started_at=None,
            finished_at=None,
            task_type='fix-violation',
            task_params={'aid': 123},
        )

    def test_success(self):
        job = self._make_job("开始处理...\n[50%]\n[SUCCESS] 修复完成")
        status, error, progress = self.jm._determine_tools_job_result(job)
        self.assertEqual(status, JobStatus.COMPLETED)
        self.assertIsNone(error)
        self.assertEqual(progress, 1.0)

    def test_success_no_message(self):
        job = self._make_job("处理中...\n[SUCCESS]")
        status, error, progress = self.jm._determine_tools_job_result(job)
        self.assertEqual(status, JobStatus.COMPLETED)

    def test_failed(self):
        job = self._make_job("开始处理...\n[30%]\n[FAILED] 文件不存在")
        status, error, progress = self.jm._determine_tools_job_result(job)
        self.assertEqual(status, JobStatus.FAILED)
        self.assertIn("文件不存在", error)

    def test_failed_no_message(self):
        job = self._make_job("[FAILED]")
        status, error, progress = self.jm._determine_tools_job_result(job)
        self.assertEqual(status, JobStatus.FAILED)

    def test_no_marker_crash(self):
        """无标记=进程崩溃"""
        job = self._make_job("开始处理...\n[20%]\nTraceback: something broke")
        status, error, progress = self.jm._determine_tools_job_result(job)
        self.assertEqual(status, JobStatus.FAILED)
        self.assertIn("异常终止", error)

    def test_no_log_file(self):
        job = self._make_job("")
        job.log_file = "/nonexistent/path.log"
        status, error, progress = self.jm._determine_tools_job_result(job)
        self.assertEqual(status, JobStatus.FAILED)


class TestWebJobToolsFields(TestCase):
    """测试 WebJob 的 tools 相关字段"""

    def test_is_tools_task(self):
        job = WebJob(
            job_id='t1', video_ids=[], steps=['fix-violation'],
            gpu_device='auto', force=False, status=JobStatus.PENDING,
            pid=None, log_file=None, progress=0, error=None,
            created_at=None, started_at=None, finished_at=None,
            task_type='fix-violation', task_params={'aid': 1}
        )
        self.assertTrue(job.is_tools_task)

    def test_is_not_tools_task(self):
        job = WebJob(
            job_id='t2', video_ids=['vid1'], steps=['download'],
            gpu_device='auto', force=False, status=JobStatus.PENDING,
            pid=None, log_file=None, progress=0, error=None,
            created_at=None, started_at=None, finished_at=None,
        )
        self.assertFalse(job.is_tools_task)

    def test_to_dict_includes_task_fields(self):
        job = WebJob(
            job_id='t3', video_ids=[], steps=['season-sync'],
            gpu_device='auto', force=False, status=JobStatus.RUNNING,
            pid=999, log_file='/tmp/log', progress=0.5, error=None,
            created_at=None, started_at=None, finished_at=None,
            task_type='season-sync', task_params={'playlist_id': 'PL1'}
        )
        d = job.to_dict()
        self.assertEqual(d['task_type'], 'season-sync')
        self.assertEqual(d['task_params'], {'playlist_id': 'PL1'})


class TestJobManagerPersistence(TestCase):
    """测试 JobManager DB 持久化（task_type, task_params 列）"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.jm = JobManager(
            db_path=os.path.join(self.tmpdir, 'test.db'),
            log_dir=os.path.join(self.tmpdir, 'logs')
        )

    def test_submit_and_get_tools_job(self):
        """提交 tools 任务后能正确读取 task_type 和 task_params"""
        import sqlite3
        # 直接写 DB（不启动子进程）
        from datetime import datetime
        job_id = 'test_tj'
        now = datetime.now()
        with self.jm._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO web_jobs 
                (job_id, video_ids, steps, gpu_device, force, status, log_file, created_at,
                 task_type, task_params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id, '[]', '["fix-violation"]', 'auto', 0,
                'pending', '/tmp/test.log', now,
                'fix-violation', json.dumps({'aid': 12345})
            ))

        job = self.jm.get_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job.task_type, 'fix-violation')
        self.assertEqual(job.task_params, {'aid': 12345})
        self.assertTrue(job.is_tools_task)

    def test_backward_compat_no_task_type_column(self):
        """旧 DB 没有 task_type 列时，默认为 process"""
        from datetime import datetime
        job_id = 'old_job'
        now = datetime.now()
        with self.jm._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO web_jobs 
                (job_id, video_ids, steps, gpu_device, force, status, log_file, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id, '["vid1"]', '["download"]', 'auto', 0,
                'completed', '/tmp/old.log', now,
            ))

        job = self.jm.get_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job.task_type, 'process')
        self.assertFalse(job.is_tools_task)


class TestToolsTaskTypes(TestCase):
    """验证 TOOLS_TASK_TYPES 常量"""

    def test_all_types_defined(self):
        expected = {
            'fix-violation', 'sync-playlist', 'refresh-playlist',
            'retranslate-playlist', 'upload-sync', 'update-info',
            'sync-db', 'season-sync',
        }
        self.assertEqual(TOOLS_TASK_TYPES, expected)


class TestFixViolationLoop(TestCase):
    """测试 fix-violation 自动循环逻辑

    通过 mock 外部依赖，验证：
    - 循环控制（dry-run/通过/失败/耗尽轮次）
    - 等待时间计算（upload_duration*2 / 下限900s / 显式指定）
    - 成功后调用 add-to-season
    """

    def _make_fix_result(self, success=True, upload_duration=500, all_ranges=None):
        """构造 fix_violation 的返回值"""
        return {
            'success': success,
            'new_ranges': [(100, 110)],
            'all_ranges': all_ranges or [(100, 110)],
            'masked_path': '/tmp/masked.mp4',
            'source': 'local',
            'message': '修复完成' if success else '修复失败',
            'upload_duration': upload_duration,
        }

    def _invoke(self, mock_uploader, extra_args=None, **overrides):
        """用 CliRunner 调用 tools_fix_violation，返回 CliRunner.Result"""
        from unittest.mock import patch, MagicMock
        from click.testing import CliRunner
        from vat.cli.tools import tools_fix_violation

        mock_config = MagicMock()
        mock_config.storage.database_path = ':memory:'
        mock_config.storage.output_dir = '/tmp'

        args = ['--aid', '12345', '--max-rounds', str(overrides.get('max_rounds', 3)),
                '--wait-seconds', str(overrides.get('wait_seconds', 1))]
        if overrides.get('dry_run'):
            args.append('--dry-run')
        if extra_args:
            args.extend(extra_args)

        patches = {
            'vat.config.load_config': MagicMock(return_value=mock_config),
            'vat.database.Database': MagicMock(),
            'vat.cli.tools._get_bilibili_uploader': MagicMock(return_value=mock_uploader),
            'vat.cli.tools._find_local_video_for_aid': MagicMock(return_value=None),
            'vat.cli.tools._try_add_to_season': MagicMock(),
            'vat.cli.commands._get_previous_violation_ranges': MagicMock(return_value=[]),
            'vat.cli.commands._save_violation_ranges': MagicMock(),
            'time.sleep': MagicMock(),
        }

        runner = CliRunner()
        import contextlib
        stack = contextlib.ExitStack()
        mocks = {}
        for target, mock_obj in patches.items():
            mocks[target] = stack.enter_context(patch(target, mock_obj))
        with stack:
            result = runner.invoke(tools_fix_violation, args, catch_exceptions=False)

        return result, mocks

    def test_dry_run_no_loop(self):
        """dry-run 模式只做一次 mask，不循环"""
        from unittest.mock import MagicMock, patch
        mock_uploader = MagicMock()
        mock_uploader.fix_violation.return_value = self._make_fix_result()

        result, mocks = self._invoke(mock_uploader, dry_run=True)

        self.assertEqual(mock_uploader.fix_violation.call_count, 1)
        self.assertIn('[SUCCESS]', result.output)
        self.assertIn('dry-run', result.output)
        # 不应调用 get_rejected_videos（不等待/不检查）
        mock_uploader.get_rejected_videos.assert_not_called()

    def test_single_round_passes(self):
        """第1轮修复后审核通过 → 不再循环"""
        from unittest.mock import MagicMock, patch
        mock_uploader = MagicMock()
        mock_uploader.fix_violation.return_value = self._make_fix_result()
        # 审核通过：aid 不在退回列表中
        mock_uploader.get_rejected_videos.return_value = []

        result, mocks = self._invoke(mock_uploader, max_rounds=5, wait_seconds=1)

        self.assertEqual(mock_uploader.fix_violation.call_count, 1)
        self.assertIn('[SUCCESS]', result.output)
        self.assertIn('修复流程完成', result.output)
        # 应调用 add-to-season
        mocks['vat.cli.tools._try_add_to_season'].assert_called_once()

    def test_two_rounds_then_passes(self):
        """第1轮被退回，第2轮通过"""
        from unittest.mock import MagicMock, patch
        mock_uploader = MagicMock()
        mock_uploader.fix_violation.return_value = self._make_fix_result()
        # 第1次检查：仍被退回；第2次检查：已通过
        mock_uploader.get_rejected_videos.side_effect = [
            [{'aid': 12345, 'title': 'test'}],  # 第1轮后检查
            [],                                   # 第2轮后检查
        ]

        result, mocks = self._invoke(mock_uploader, max_rounds=5, wait_seconds=1)

        self.assertEqual(mock_uploader.fix_violation.call_count, 2)
        self.assertIn('[SUCCESS]', result.output)

    def test_fix_fails_stops_immediately(self):
        """fix_violation 返回 success=False → 立即停止"""
        from unittest.mock import MagicMock, patch
        mock_uploader = MagicMock()
        mock_uploader.fix_violation.return_value = self._make_fix_result(success=False)

        result, mocks = self._invoke(mock_uploader, max_rounds=5, wait_seconds=1)

        self.assertEqual(mock_uploader.fix_violation.call_count, 1)
        self.assertIn('[FAILED]', result.output)
        # 不应等待或检查
        mock_uploader.get_rejected_videos.assert_not_called()

    def test_max_rounds_exhausted(self):
        """所有轮次用完，每轮都被退回"""
        from unittest.mock import MagicMock, patch
        mock_uploader = MagicMock()
        mock_uploader.fix_violation.return_value = self._make_fix_result()
        # 每轮检查都被退回（最后一轮不检查）
        mock_uploader.get_rejected_videos.return_value = [{'aid': 12345}]

        result, mocks = self._invoke(mock_uploader, max_rounds=3, wait_seconds=1)

        self.assertEqual(mock_uploader.fix_violation.call_count, 3)
        # 最后一轮不等待/不检查，所以 get_rejected_videos 只调用 2 次
        self.assertEqual(mock_uploader.get_rejected_videos.call_count, 2)
        self.assertIn('[SUCCESS]', result.output)
        self.assertIn('3 轮修复提交', result.output)

    def test_wait_time_minimum_900s(self):
        """upload_duration 很短时，等待时间不低于900s"""
        from unittest.mock import MagicMock, patch
        mock_uploader = MagicMock()
        # upload_duration = 100s → 100*2 = 200s < 900s → 用 900s
        mock_uploader.fix_violation.return_value = self._make_fix_result(upload_duration=100)
        mock_uploader.get_rejected_videos.return_value = []

        result, mocks = self._invoke(mock_uploader, max_rounds=2, wait_seconds=0)

        self.assertIn('等待审核 900s', result.output)

    def test_wait_time_uses_upload_duration(self):
        """upload_duration 足够大时，等待时间 = upload_duration * 2"""
        from unittest.mock import MagicMock, patch
        mock_uploader = MagicMock()
        # upload_duration = 600s → 600*2 = 1200s > 900s → 用 1200s
        mock_uploader.fix_violation.return_value = self._make_fix_result(upload_duration=600)
        mock_uploader.get_rejected_videos.return_value = []

        result, mocks = self._invoke(mock_uploader, max_rounds=2, wait_seconds=0)

        self.assertIn('等待审核 1200s', result.output)

    def test_explicit_wait_seconds_overrides(self):
        """显式指定 --wait-seconds 覆盖自动计算"""
        from unittest.mock import MagicMock, patch
        mock_uploader = MagicMock()
        mock_uploader.fix_violation.return_value = self._make_fix_result(upload_duration=600)
        mock_uploader.get_rejected_videos.return_value = []

        result, mocks = self._invoke(mock_uploader, max_rounds=2, wait_seconds=60)

        self.assertIn('等待审核 60s', result.output)

    def test_ranges_accumulated_across_rounds(self):
        """各轮次间 ranges 正确累积"""
        from unittest.mock import MagicMock, patch, call
        mock_uploader = MagicMock()
        # 第1轮返回 ranges A, 第2轮返回 ranges A+B
        mock_uploader.fix_violation.side_effect = [
            self._make_fix_result(all_ranges=[(100, 110)]),
            self._make_fix_result(all_ranges=[(100, 110), (200, 210)]),
        ]
        mock_uploader.get_rejected_videos.side_effect = [
            [{'aid': 12345}],  # 第1轮后仍被退回
            [],                 # 第2轮后通过
        ]

        result, mocks = self._invoke(mock_uploader, max_rounds=5, wait_seconds=1)

        # 验证第2轮调用时 previous_ranges 是第1轮的 all_ranges
        calls = mock_uploader.fix_violation.call_args_list
        self.assertEqual(calls[1].kwargs.get('previous_ranges') or calls[1][1].get('previous_ranges'),
                         [(100, 110)])
