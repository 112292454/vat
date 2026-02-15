"""
pipeline 模块单元测试

测试 create_video_from_url、ProgressTracker、Pipeline 异常层次。
"""
import os
import inspect
import tempfile
import pytest
from vat.database import Database
from vat.models import TaskStep, TaskStatus, DEFAULT_STAGE_SEQUENCE


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
