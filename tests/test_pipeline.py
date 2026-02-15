"""
pipeline 模块单元测试

测试 create_video_from_url、ProgressTracker、Pipeline 异常层次、
VideoProcessor.process() 调度逻辑（mock _run_* 方法验证编排行为）。
"""
import os
import shutil
import inspect
import tempfile
import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from vat.database import Database
from vat.models import (
    Video, Task, TaskStep, TaskStatus, SourceType,
    DEFAULT_STAGE_SEQUENCE,
)


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    database = Database(path)
    yield database
    os.unlink(path)


class TestCreateVideoFromUrl:

    def test_first_creation_creates_7_pending_tasks(self, db):
        from vat.pipeline.executor import create_video_from_url
        video_id = create_video_from_url("https://youtube.com/watch?v=test1", db)
        tasks = db.get_tasks(video_id)
        assert len(tasks) == 7
        assert all(t.status == TaskStatus.PENDING for t in tasks)

    def test_covers_all_steps(self, db):
        from vat.pipeline.executor import create_video_from_url
        video_id = create_video_from_url("https://youtube.com/watch?v=test2", db)
        task_steps = {t.step for t in db.get_tasks(video_id)}
        assert task_steps == set(DEFAULT_STAGE_SEQUENCE)

    def test_same_url_same_id(self, db):
        from vat.pipeline.executor import create_video_from_url
        url = "https://youtube.com/watch?v=dup"
        assert create_video_from_url(url, db) == create_video_from_url(url, db)

    def test_duplicate_no_extra_tasks(self, db):
        from vat.pipeline.executor import create_video_from_url
        url = "https://youtube.com/watch?v=dup2"
        vid = create_video_from_url(url, db)
        create_video_from_url(url, db)
        assert len(db.get_tasks(vid)) == 7

    def test_duplicate_resets_completed(self, db):
        from vat.pipeline.executor import create_video_from_url
        url = "https://youtube.com/watch?v=reset"
        vid = create_video_from_url(url, db)
        db.update_task_status(vid, TaskStep.DOWNLOAD, TaskStatus.COMPLETED)
        create_video_from_url(url, db)
        assert all(t.status == TaskStatus.PENDING for t in db.get_tasks(vid))


class TestProgressTracker:

    def _make(self, stages, callback=None):
        from vat.pipeline.progress import ProgressTracker
        return ProgressTracker(stages=stages, callback=callback)

    def test_initial_zero(self):
        assert self._make(['a', 'b']).get_overall_progress() == 0.0

    def test_one_of_two_stages(self):
        t = self._make(['a', 'b'])
        t.start_stage('a'); t.complete_stage('a')
        assert abs(t.get_overall_progress() - 0.5) < 0.01

    def test_all_complete(self):
        t = self._make(['a', 'b'])
        for s in ['a', 'b']:
            t.start_stage(s); t.complete_stage(s)
        assert abs(t.get_overall_progress() - 1.0) < 0.01

    def test_partial_items(self):
        t = self._make(['x', 'y'])
        t.start_stage('x'); t.set_total_items(4); t.increment_completed(2)
        assert abs(t.get_overall_progress() - 0.25) < 0.01

    def test_callback_fires(self):
        msgs = []
        t = self._make(['a'], callback=lambda p, m: msgs.append(p))
        t.start_stage('a'); t.complete_stage('a')
        assert len(msgs) >= 2

    def test_progress_info(self):
        t = self._make(['a', 'b'])
        t.start_stage('a'); t.complete_stage('a')
        info = t.get_progress_info()
        assert 'a' in info['completed_stages']
        assert info['stages']['a']['progress'] == 1.0
        assert info['stages']['b']['progress'] == 0.0

    def test_set_stage_progress(self):
        t = self._make(['embed'])
        t.start_stage('embed'); t.set_stage_progress(0.6)
        assert abs(t.get_overall_progress() - 0.6) < 0.01

    def test_empty_stages(self):
        assert self._make([]).get_overall_progress() == 0.0

    def test_download_events(self):
        from vat.pipeline.progress import ProgressEvent
        t = self._make(['download'])
        t.start_stage('download')
        t.report_event(ProgressEvent.DOWNLOAD_INFO_FETCHED)
        assert abs(t.get_overall_progress() - 0.2) < 0.01
        t.report_event(ProgressEvent.DOWNLOAD_VIDEO_DONE)
        assert abs(t.get_overall_progress() - 0.8) < 0.01


class TestPipelineExceptions:

    def test_subclass_hierarchy(self):
        from vat.pipeline.exceptions import (
            PipelineError, ASRError, TranslateError,
            EmbedError, DownloadError, UploadError,
        )
        for cls in [ASRError, TranslateError, EmbedError, DownloadError, UploadError]:
            assert issubclass(cls, PipelineError)
            assert issubclass(cls, Exception)

    def test_original_error(self):
        from vat.pipeline.exceptions import ASRError
        orig = ValueError("orig")
        err = ASRError("msg", original_error=orig)
        assert err.original_error is orig
        assert str(err) == "msg"

    def test_catch_generic(self):
        from vat.pipeline.exceptions import PipelineError, DownloadError
        with pytest.raises(PipelineError):
            raise DownloadError("fail")


class TestVideoProcessorInterface:

    def test_init_params(self):
        from vat.pipeline.executor import VideoProcessor
        sig = inspect.signature(VideoProcessor.__init__)
        params = set(sig.parameters.keys())
        for p in ['video_id', 'config', 'gpu_id', 'force', 'video_index', 'total_videos']:
            assert p in params, f"缺少参数 {p}"

    def test_process_has_steps_param(self):
        from vat.pipeline.executor import VideoProcessor
        sig = inspect.signature(VideoProcessor.process)
        assert 'steps' in sig.parameters
        assert 'force' not in sig.parameters

    def test_stage_methods_exist(self):
        from vat.pipeline.executor import VideoProcessor
        members = {m[0] for m in inspect.getmembers(VideoProcessor, predicate=inspect.isfunction)}
        for method in ['_run_download', '_run_whisper', '_run_split',
                       '_run_optimize', '_run_translate', '_run_embed', '_run_upload']:
            assert method in members, f"缺少方法 {method}"


# ==================== VideoProcessor.process() 调度逻辑 mock 测试 ====================

def _make_vp(tmp_path, force=False, video_metadata=None):
    """
    构造可控的 VideoProcessor 实例（绕过真实 __init__，手动注入依赖）
    
    返回 (vp, db) 元组，所有 _run_* 方法已替换为返回 True 的 mock。
    """
    from vat.pipeline.executor import VideoProcessor
    from vat.config import load_config

    db_path = str(tmp_path / "test.db")
    database = Database(db_path)

    # 添加测试视频
    v = Video(
        id="test_vid", source_type=SourceType.YOUTUBE,
        source_url="https://youtube.com/watch?v=test",
        title="Test", output_dir=str(tmp_path / "output"),
        metadata=video_metadata or {},
    )
    database.add_video(v)
    for step in DEFAULT_STAGE_SEQUENCE:
        database.add_task(Task(video_id="test_vid", step=step, status=TaskStatus.PENDING))

    # 创建输出目录
    (tmp_path / "output").mkdir(exist_ok=True)

    # 绕过 __init__，手动构造实例
    vp = object.__new__(VideoProcessor)
    config = load_config()
    # 覆盖存储路径
    config.storage.database_path = db_path
    config.storage.output_dir = str(tmp_path / "output")
    config.storage.cache_dir = str(tmp_path / "cache")

    vp.video_id = "test_vid"
    vp.config = config
    vp.gpu_id = None
    vp.force = force
    vp.video_index = 0
    vp.total_videos = 1
    vp.logger = logging.getLogger("test.pipeline")
    vp.db = database
    vp.video = database.get_video("test_vid")
    vp.output_dir = tmp_path / "output"
    vp._progress_tracker = None
    vp._passthrough_stages = set()
    vp._config_backup = None
    vp._downloader = None
    vp._asr = None
    vp._ffmpeg = None

    # 进度回调记录
    vp._log = []
    def _cb(msg):
        vp._log.append(msg)
    vp.progress_callback = _cb
    vp._default_progress_callback = _cb

    # Mock 所有 _run_* 方法为返回 True
    for step in DEFAULT_STAGE_SEQUENCE:
        setattr(vp, f"_run_{step.value}", MagicMock(return_value=True))

    return vp, database


class TestProcessOrchestration:
    """VideoProcessor.process() 调度逻辑（mock _run_* 验证编排行为）"""

    def test_all_pending_steps_executed(self, tmp_path):
        """steps=None 时，执行所有 pending 步骤"""
        vp, db = _make_vp(tmp_path)
        result = vp.process(steps=None)
        assert result is True
        # 所有 7 个步骤都被调用
        for step in DEFAULT_STAGE_SEQUENCE:
            getattr(vp, f"_run_{step.value}").assert_called_once()
        # DB 中所有步骤都标记为 COMPLETED
        for step in DEFAULT_STAGE_SEQUENCE:
            task = db.get_task("test_vid", step)
            assert task.status == TaskStatus.COMPLETED, f"{step} 应为 COMPLETED"

    def test_skip_completed_steps(self, tmp_path):
        """force=False 时，跳过已完成步骤"""
        vp, db = _make_vp(tmp_path)
        # 预先标记 download 和 whisper 为完成
        db.update_task_status("test_vid", TaskStep.DOWNLOAD, TaskStatus.COMPLETED)
        db.update_task_status("test_vid", TaskStep.WHISPER, TaskStatus.COMPLETED)

        result = vp.process(steps=None)
        assert result is True
        # download 和 whisper 不应被调用（已跳过）
        vp._run_download.assert_not_called()
        vp._run_whisper.assert_not_called()
        # 其余步骤应被调用
        vp._run_split.assert_called_once()
        vp._run_translate.assert_called_once()

    def test_force_re_executes_completed(self, tmp_path):
        """force=True 时，重新执行已完成步骤"""
        vp, db = _make_vp(tmp_path, force=True)
        db.update_task_status("test_vid", TaskStep.DOWNLOAD, TaskStatus.COMPLETED)

        result = vp.process(steps=['download', 'whisper'])
        assert result is True
        # force 模式下 download 应被重新执行
        vp._run_download.assert_called_once()
        vp._run_whisper.assert_called_once()

    def test_failure_stops_pipeline(self, tmp_path):
        """某步骤失败后应停止后续步骤"""
        vp, db = _make_vp(tmp_path)
        # whisper 返回 False（失败）
        vp._run_whisper = MagicMock(return_value=False)

        result = vp.process(steps=None)
        assert result is False
        # download 成功，whisper 失败
        vp._run_download.assert_called_once()
        vp._run_whisper.assert_called_once()
        # split 不应被调用（因 whisper 失败后 break）
        vp._run_split.assert_not_called()
        # whisper 在 DB 中应标记为 FAILED
        task = db.get_task("test_vid", TaskStep.WHISPER)
        assert task.status == TaskStatus.FAILED

    def test_pipeline_error_marks_failed(self, tmp_path):
        """PipelineError 异常应标记步骤为 FAILED 并停止"""
        from vat.pipeline.exceptions import ASRError
        vp, db = _make_vp(tmp_path)
        vp._run_whisper = MagicMock(side_effect=ASRError("模型加载失败"))

        result = vp.process(steps=None)
        assert result is False
        task = db.get_task("test_vid", TaskStep.WHISPER)
        assert task.status == TaskStatus.FAILED
        assert "ASRError" in task.error_message
        # 后续步骤不执行
        vp._run_split.assert_not_called()

    def test_generic_exception_marks_failed(self, tmp_path):
        """非 PipelineError 异常也应标记为 FAILED"""
        vp, db = _make_vp(tmp_path)
        vp._run_download = MagicMock(side_effect=RuntimeError("磁盘满了"))

        result = vp.process(steps=None)
        assert result is False
        task = db.get_task("test_vid", TaskStep.DOWNLOAD)
        assert task.status == TaskStatus.FAILED
        assert "RuntimeError" in task.error_message

    def test_stage_group_expansion(self, tmp_path):
        """传入阶段组名 'asr' 应展开为 whisper + split"""
        vp, db = _make_vp(tmp_path)
        result = vp.process(steps=['asr'])
        assert result is True
        vp._run_whisper.assert_called_once()
        vp._run_split.assert_called_once()
        # download 不应被调用（不在展开结果中，且不在 gap 范围内）
        vp._run_download.assert_not_called()

    def test_explicit_steps_only(self, tmp_path):
        """只传入 download 和 embed 时，中间阶段以直通模式执行"""
        vp, db = _make_vp(tmp_path)
        result = vp.process(steps=['download', 'embed'])
        assert result is True
        # download 和 embed 必须被调用
        vp._run_download.assert_called_once()
        vp._run_embed.assert_called_once()
        # 中间阶段也被执行（直通填充）
        vp._run_whisper.assert_called_once()

    def test_unavailable_video_skips_all(self, tmp_path):
        """不可用视频应跳过所有处理，标记全部完成"""
        vp, db = _make_vp(tmp_path, video_metadata={"unavailable": True})
        result = vp.process(steps=None)
        assert result is True
        # 所有 _run_* 不应被调用
        for step in DEFAULT_STAGE_SEQUENCE:
            getattr(vp, f"_run_{step.value}").assert_not_called()

    def test_empty_pending_returns_true(self, tmp_path):
        """所有步骤已完成时应直接返回 True"""
        vp, db = _make_vp(tmp_path)
        for step in DEFAULT_STAGE_SEQUENCE:
            db.update_task_status("test_vid", step, TaskStatus.COMPLETED)
        result = vp.process(steps=None)
        assert result is True
        for step in DEFAULT_STAGE_SEQUENCE:
            getattr(vp, f"_run_{step.value}").assert_not_called()

    def test_running_status_set_before_execution(self, tmp_path):
        """执行步骤前应先将状态设为 RUNNING"""
        vp, db = _make_vp(tmp_path)
        statuses_during_run = []

        def capture_status():
            task = db.get_task("test_vid", TaskStep.DOWNLOAD)
            statuses_during_run.append(task.status)
            return True

        vp._run_download = MagicMock(side_effect=capture_status)
        vp.process(steps=['download'])
        # 在 _run_download 执行时，DB 中状态应该是 RUNNING
        assert statuses_during_run[0] == TaskStatus.RUNNING


class TestResolveStageGaps:
    """_resolve_stage_gaps 直通填充逻辑"""

    def _make_vp_for_gaps(self, tmp_path):
        vp, _ = _make_vp(tmp_path)
        return vp

    def test_contiguous_no_gaps(self, tmp_path):
        vp = self._make_vp_for_gaps(tmp_path)
        result = vp._resolve_stage_gaps(['download', 'whisper', 'split'])
        assert result == ['download', 'whisper', 'split']
        assert vp._passthrough_stages == set()

    def test_noncontiguous_fills_gaps(self, tmp_path):
        vp = self._make_vp_for_gaps(tmp_path)
        result = vp._resolve_stage_gaps(['whisper', 'embed'])
        # 应填充 whisper -> split -> optimize -> translate -> embed
        assert 'split' in result
        assert 'optimize' in result
        assert 'translate' in result
        assert {'split', 'optimize', 'translate'} == vp._passthrough_stages

    def test_single_step_no_gap(self, tmp_path):
        vp = self._make_vp_for_gaps(tmp_path)
        result = vp._resolve_stage_gaps(['whisper'])
        assert result == ['whisper']
        assert vp._passthrough_stages == set()

    def test_empty_input(self, tmp_path):
        vp = self._make_vp_for_gaps(tmp_path)
        assert vp._resolve_stage_gaps([]) == []


class TestPassthroughConfigBackupRestore:
    """直通模式 config 备份与恢复"""

    def test_config_restored_after_process(self, tmp_path):
        """process() 结束后 config 应恢复到原始值"""
        vp, _ = _make_vp(tmp_path)
        original_split_enable = vp.config.asr.split.enable
        original_skip_translate = vp.config.translator.skip_translate

        # 执行包含直通的步骤序列
        vp.process(steps=['whisper', 'embed'])

        # config 应被恢复
        assert vp.config.asr.split.enable == original_split_enable
        assert vp.config.translator.skip_translate == original_skip_translate

    def test_config_restored_even_on_failure(self, tmp_path):
        """即使步骤失败，config 也应恢复"""
        vp, _ = _make_vp(tmp_path)
        original_split_enable = vp.config.asr.split.enable
        vp._run_embed = MagicMock(side_effect=RuntimeError("boom"))

        vp.process(steps=['whisper', 'embed'])

        assert vp.config.asr.split.enable == original_split_enable


class TestExecuteStepDispatch:
    """_execute_step 步骤分发"""

    def test_all_steps_dispatched_correctly(self, tmp_path):
        """每个 TaskStep 都分发到对应的 _run_* 方法"""
        vp, _ = _make_vp(tmp_path)
        for step in DEFAULT_STAGE_SEQUENCE:
            mock_fn = getattr(vp, f"_run_{step.value}")
            mock_fn.reset_mock()
            vp._execute_step(step)
            mock_fn.assert_called_once()

    def test_unknown_step_raises(self, tmp_path):
        """未知 TaskStep 应抛出 ValueError"""
        vp, _ = _make_vp(tmp_path)
        # 构造一个不在 handlers 中的 step（通过 mock TaskStep）
        fake_step = MagicMock()
        fake_step.value = "nonexistent"
        with pytest.raises(ValueError, match="未知步骤"):
            vp._execute_step(fake_step)


# ==================== Scheduler download_delay mock 测试 ====================

class TestSchedulerDownloadDelay:
    """SingleGPUScheduler 视频间延迟"""

    @patch('vat.pipeline.scheduler.VideoProcessor')
    @patch('time.sleep')
    def test_delay_between_videos(self, mock_sleep, MockVP, tmp_path):
        """多视频处理时应在视频间插入延迟"""
        from vat.pipeline.scheduler import SingleGPUScheduler
        from vat.config import load_config

        config = load_config()
        config.storage.database_path = str(tmp_path / "test.db")
        db = Database(config.storage.database_path)

        # 添加测试视频
        for vid in ["v1", "v2", "v3"]:
            db.add_video(Video(id=vid, source_type=SourceType.YOUTUBE,
                               source_url=f"https://youtube.com/watch?v={vid}"))

        # Mock VideoProcessor 实例
        mock_instance = MagicMock()
        mock_instance.process.return_value = True
        MockVP.return_value = mock_instance

        scheduler = SingleGPUScheduler(config, gpu_id=0)
        config.downloader.youtube.download_delay = 5

        scheduler.run(["v1", "v2", "v3"])

        # VideoProcessor 应被实例化 3 次
        assert MockVP.call_count == 3
        # sleep 应被调用 2 次（第 2、3 个视频前）
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(5)

    @patch('vat.pipeline.scheduler.VideoProcessor')
    @patch('time.sleep')
    def test_no_delay_when_zero(self, mock_sleep, MockVP, tmp_path):
        """download_delay=0 时不应有延迟"""
        from vat.pipeline.scheduler import SingleGPUScheduler
        from vat.config import load_config

        config = load_config()
        config.storage.database_path = str(tmp_path / "test.db")
        db = Database(config.storage.database_path)

        for vid in ["v1", "v2"]:
            db.add_video(Video(id=vid, source_type=SourceType.YOUTUBE,
                               source_url=f"https://youtube.com/watch?v={vid}"))

        mock_instance = MagicMock()
        mock_instance.process.return_value = True
        MockVP.return_value = mock_instance

        scheduler = SingleGPUScheduler(config, gpu_id=0)
        config.downloader.youtube.download_delay = 0

        scheduler.run(["v1", "v2"])

        mock_sleep.assert_not_called()

    @patch('vat.pipeline.scheduler.VideoProcessor')
    @patch('time.sleep')
    def test_single_video_no_delay(self, mock_sleep, MockVP, tmp_path):
        """单视频时不应有延迟"""
        from vat.pipeline.scheduler import SingleGPUScheduler
        from vat.config import load_config

        config = load_config()
        config.storage.database_path = str(tmp_path / "test.db")
        db = Database(config.storage.database_path)

        db.add_video(Video(id="v1", source_type=SourceType.YOUTUBE,
                           source_url="https://youtube.com/watch?v=v1"))

        mock_instance = MagicMock()
        mock_instance.process.return_value = True
        MockVP.return_value = mock_instance

        scheduler = SingleGPUScheduler(config, gpu_id=0)
        config.downloader.youtube.download_delay = 30

        scheduler.run(["v1"])

        mock_sleep.assert_not_called()
